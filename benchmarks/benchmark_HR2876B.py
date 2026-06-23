import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from diff_integrator.loss import JointLoss
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

BENCH_DIR = Path("../diff-biophys/benchmarks/HR2876B").resolve()
sys.path.insert(0, str(BENCH_DIR))
from parse_nmrstar import load_bmrb_shifts

def main():
    print("Running diff-integrator benchmark: HR2876B (2LTM)")
    
    struct = load_pdb_model(BENCH_DIR / "2LTM.pdb", model_id=1)
    res_ids, res_names = get_residue_info(struct)
    coords = get_backbone_coords(struct)
    build_backbone = make_backbone_builder(len(res_ids), coords[:3])
    init_phi, init_psi = compute_phi_psi(coords)
    
    bmrb = load_bmrb_shifts(BENCH_DIR / "bmrb18489_HR2876B.str")
    ca_exp = bmrb["CA"]
    ca_loss = CAShiftLoss(ca_exp["res_id"], ca_exp["shift"], res_ids, list(res_names))
    
    geom_loss = GeometryLoss(target_coords=coords, target_weight=1.0)
    
    joint_loss = JointLoss([
        (geom_loss, 1.0),
        (ca_loss, 1.0)
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

if __name__ == "__main__":
    main()
