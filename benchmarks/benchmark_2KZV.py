import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from diff_integrator.loss import JointLoss, LossTerm
from diff_integrator.optimizer import IntegrativeRefiner
from diff_integrator.terms.chemical_shifts import CAShiftLoss
from diff_integrator.terms.geometry import GeometryLoss

from diff_biophys.geometry.backbone import (
    compute_phi_psi,
    get_backbone_coords,
    get_residue_info,
    load_pdb_model,
    make_backbone_builder,
)
from diff_biophys.nmr.io import load_rdc_table
from diff_biophys.nmr.rdc import make_rdc_refinement_fns

BENCH_DIR = Path("../diff-biophys/benchmarks/2KZV").resolve()
sys.path.insert(0, str(BENCH_DIR))
from parse_bmrb import load_bmrb_shifts

class LegacyRDCLoss(LossTerm):
    def __init__(self, rdc_loss_fn, tensor_fn):
        self.rdc_loss_fn = rdc_loss_fn
        self.tensor_fn = tensor_fn
    
    def __call__(self, params, coords):
        tensor = self.tensor_fn(coords)
        return self.rdc_loss_fn(coords, tensor)

def main():
    print("Running diff-integrator benchmark: 2KZV (CvR118A)")
    
    struct = load_pdb_model(BENCH_DIR / "2KZV.pdb", model_id=1)
    res_ids, res_names = get_residue_info(struct)
    coords = get_backbone_coords(struct)
    build_backbone = make_backbone_builder(len(res_ids), coords[:3])
    init_phi, init_psi = compute_phi_psi(coords)
    
    bmrb = load_bmrb_shifts(BENCH_DIR / "bmrb17020.str")
    ca_exp = bmrb["CA"]
    ca_loss = CAShiftLoss(ca_exp["res_id"], ca_exp["shift"], res_ids, list(res_names))
    
    rdc_pag = load_rdc_table(BENCH_DIR / "rdc_PAG.tsv")["PAG"]
    rdc_peg = load_rdc_table(BENCH_DIR / "rdc_PEG.tsv")["PEG"]
    
    loss_pag, q_pag, tensor_pag, n_pag = make_rdc_refinement_fns(rdc_pag["res_id"], rdc_pag["rdc"], res_ids)
    loss_peg, q_peg, tensor_peg, n_peg = make_rdc_refinement_fns(rdc_peg["res_id"], rdc_peg["rdc"], res_ids)
    
    rdc_term_pag = LegacyRDCLoss(loss_pag, tensor_pag)
    rdc_term_peg = LegacyRDCLoss(loss_peg, tensor_peg)
    
    geom_loss = GeometryLoss(target_coords=coords, target_weight=1.0)
    
    joint_loss = JointLoss([
        (geom_loss, 5.0),    # Strong anchor to prevent unravelling
        (ca_loss, 1.0),
        (rdc_term_pag, 1.0),
        (rdc_term_peg, 1.0)
    ])
    
    refiner = IntegrativeRefiner(loss_fn=joint_loss)
    final_params, history = refiner.run(
        init_params=(init_phi, init_psi),
        epochs=500,
        learning_rate=0.01,
        kinematics_fn=lambda p: build_backbone(p[0], p[1])
    )
    
    final_phi, final_psi = final_params
    final_coords = build_backbone(final_phi, final_psi)
    init_coords = build_backbone(init_phi, init_psi)
    
    print("\nResults:")
    print(f"  CA Shift RMSD (Init): {ca_loss((init_phi, init_psi), init_coords):.3f}")
    print(f"  CA Shift RMSD (Final): {ca_loss((final_phi, final_psi), final_coords):.3f}")
    print(f"  PAG Q-factor (Init): {q_pag(init_coords):.3f}")
    print(f"  PAG Q-factor (Final): {q_pag(final_coords):.3f}")
    print(f"  PEG Q-factor (Init): {q_peg(init_coords):.3f}")
    print(f"  PEG Q-factor (Final): {q_peg(final_coords):.3f}")

    # Compute structural RMSD
    import biotite.structure as struc
    from biotite.structure.io.pdb import PDBFile
    # We can just compute a simple centered RMSD or kabsch since coords are CA
    def kabsch_rmsd(A, B):
        A = np.array(A).reshape(-1, 3)
        B = np.array(B).reshape(-1, 3)
        A = A - A.mean(axis=0)
        B = B - B.mean(axis=0)
        U, S, Vt = np.linalg.svd(A.T @ B)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = U @ Vt
        A_rot = A @ R
        return np.sqrt(np.mean(np.sum((A_rot - B)**2, axis=1)))
        
    rmsd = kabsch_rmsd(init_coords, final_coords)
    print(f"  Structural RMSD to Init: {rmsd:.3f} Å")

if __name__ == "__main__":
    main()
