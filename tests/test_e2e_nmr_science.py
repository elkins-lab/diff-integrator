import sys
from pathlib import Path
import pytest
import jax.numpy as jnp
import numpy as np

from diff_integrator.loss import JointLoss
from diff_integrator.optimizer import IntegrativeRefiner
from diff_integrator.terms.chemical_shifts import CAShiftLoss
from diff_integrator.terms.geometry import GeometryLoss
from diff_integrator.loss import LossTerm

from diff_biophys.geometry.backbone import (
    compute_phi_psi,
    get_backbone_coords,
    get_residue_info,
    load_pdb_model,
    make_backbone_builder,
)

# Assume diff-biophys is cloned alongside diff-integrator
BIOPHYS_DIR = Path(__file__).parent.parent.parent / "diff-biophys"

class LegacyRDCLoss(LossTerm):
    def __init__(self, rdc_loss_fn, tensor_fn):
        self.rdc_loss_fn = rdc_loss_fn
        self.tensor_fn = tensor_fn
        self.rdc_scale = 1.0 # default scale for tests
    
    def __call__(self, params, coords):
        tensor = self.tensor_fn(coords)
        return self.rdc_loss_fn(coords, tensor)

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

@pytest.mark.slow
def test_2kzv_joint_refinement_published_validation():
    """
    Validates end-to-end refinement of 2KZV using published C-alpha shifts and RDCs.
    Reference: Li et al. (2023).
    Ensures that Q-factors decrease without structural unravelling.
    """
    bench_dir = BIOPHYS_DIR / "benchmarks" / "2KZV"
    if not bench_dir.exists():
        pytest.skip(f"Benchmark data not found at {bench_dir}")
        
    sys.path.insert(0, str(bench_dir))
    from parse_bmrb import load_bmrb_shifts
    from diff_biophys.nmr.io import load_rdc_table
    from diff_biophys.nmr.rdc import make_rdc_refinement_fns

    struct = load_pdb_model(bench_dir / "2KZV.pdb", model_id=1)
    res_ids, res_names = get_residue_info(struct)
    coords = get_backbone_coords(struct)
    build_backbone = make_backbone_builder(len(res_ids), coords[:3])
    init_phi, init_psi = compute_phi_psi(coords)

    bmrb = load_bmrb_shifts(bench_dir / "bmrb17020.str")
    ca_exp = bmrb["CA"]
    ca_loss = CAShiftLoss(ca_exp["res_id"], ca_exp["shift"], res_ids, list(res_names))

    rdc_pag = load_rdc_table(bench_dir / "rdc_PAG.tsv")["PAG"]
    rdc_peg = load_rdc_table(bench_dir / "rdc_PEG.tsv")["PEG"]

    loss_pag, q_pag_fn, tensor_pag, n_pag = make_rdc_refinement_fns(rdc_pag["res_id"], rdc_pag["rdc"], res_ids)
    loss_peg, q_peg_fn, tensor_peg, n_peg = make_rdc_refinement_fns(rdc_peg["res_id"], rdc_peg["rdc"], res_ids)

    rdc_term_pag = LegacyRDCLoss(loss_pag, tensor_pag)
    rdc_term_pag.rdc_scale = float(np.sum(rdc_pag["rdc"]**2))
    
    rdc_term_peg = LegacyRDCLoss(loss_peg, tensor_peg)
    rdc_term_peg.rdc_scale = float(np.sum(rdc_peg["rdc"]**2))
    
    geom_loss = GeometryLoss(target_coords=coords, target_weight=1.0)

    joint_loss = JointLoss([
        (geom_loss, 5.0),
        (ca_loss, 1.0),
        (rdc_term_pag, 1.0),
        (rdc_term_peg, 1.0)
    ])

    refiner = IntegrativeRefiner(loss_fn=joint_loss)
    final_params, history = refiner.run(
        init_params=(init_phi, init_psi),
        epochs=150, # Sufficient for tests
        learning_rate=0.01,
        kinematics_fn=lambda p: build_backbone(p[0], p[1])
    )

    final_phi, final_psi = final_params
    final_coords = build_backbone(final_phi, final_psi)
    init_coords = build_backbone(init_phi, init_psi)

    # Re-evaluate properties
    final_ca_rmsd = np.sqrt(float(ca_loss((final_phi, final_psi), final_coords) / len(ca_exp["res_id"])))
    
    # Calculate RDC Q-factors directly without weights
    def q_pag(c):
        return float(jnp.sqrt(rdc_term_pag((), c) / rdc_term_pag.rdc_scale))
    def q_peg(c):
        return float(jnp.sqrt(rdc_term_peg((), c) / rdc_term_peg.rdc_scale))
        
    init_q_pag = q_pag(init_coords)
    final_q_pag = q_pag(final_coords)
    
    init_q_peg = q_peg(init_coords)
    final_q_peg = q_peg(final_coords)

    rmsd = kabsch_rmsd(init_coords, final_coords)

    # Assertions based on published expectations
    assert final_q_pag < init_q_pag, "PAG Q-factor failed to descend."
    assert final_q_peg < init_q_peg, "PEG Q-factor failed to descend."
    
    # Structural RMSD should be constrained to < 2.5 A (prevents unravelling to <10 A)
    assert rmsd < 2.5, f"Structural drift too high: {rmsd:.2f} A. GeometryLoss failed."
    assert final_q_peg < 0.20, f"PEG Q-factor did not descend sufficiently: {final_q_peg:.3f}"

@pytest.mark.slow
def test_gmr58a_shift_refinement():
    """
    Validates end-to-end refinement of GmR58A using BMRB C-alpha shifts.
    """
    bench_dir = BIOPHYS_DIR / "benchmarks" / "GmR58A"
    if not bench_dir.exists():
        pytest.skip(f"Benchmark data not found at {bench_dir}")
        
    sys.path.insert(0, str(bench_dir))
    from parse_nmrstar import load_bmrb_shifts

    struct = load_pdb_model(bench_dir / "2KUT.pdb", model_id=1)
    res_ids, res_names = get_residue_info(struct)
    coords = get_backbone_coords(struct)
    build_backbone = make_backbone_builder(len(res_ids), coords[:3])
    init_phi, init_psi = compute_phi_psi(coords)

    bmrb = load_bmrb_shifts(bench_dir / "bmrb16746_GmR58A.str")
    ca_exp = bmrb["CA"]
    ca_loss = CAShiftLoss(ca_exp["res_id"], ca_exp["shift"], res_ids, list(res_names))
    geom_loss = GeometryLoss(target_coords=coords, target_weight=1.0)
    
    joint_loss = JointLoss([
        (geom_loss, 1.0),
        (ca_loss, 1.0)
    ])

    refiner = IntegrativeRefiner(loss_fn=joint_loss)
    final_params, _ = refiner.run(
        init_params=(init_phi, init_psi),
        epochs=150,
        learning_rate=0.01,
        kinematics_fn=lambda p: build_backbone(p[0], p[1])
    )

    final_phi, final_psi = final_params
    final_coords = build_backbone(final_phi, final_psi)
    init_coords = build_backbone(init_phi, init_psi)

    init_loss = float(ca_loss((init_phi, init_psi), init_coords))
    final_loss = float(ca_loss((final_phi, final_psi), final_coords))
    
    # Assert C-alpha shift loss descends
    assert final_loss < init_loss, "C-alpha shift loss failed to descend."
    
    rmsd = kabsch_rmsd(init_coords, final_coords)
    assert rmsd < 2.0, f"Structural drift too high: {rmsd:.2f} A."
