"""
End-to-end scientific validation tests for diff-integrator.

These tests use published experimental NMR data to verify that the
IntegrativeRefiner produces physically correct and scientifically meaningful
results.  Each test asserts that gradient-based optimization:

1. Reduces the relevant observables (Q-factors, shift RMSD) below their
   starting values.
2. Does *not* cause structural unravelling (Kabsch RMSD is bounded).

Published data sources
----------------------
* **2KZV (CvR118A)**: Li et al. (2023), PMID 37257257.
  Cα chemical shifts from BMRB 17020; ¹⁵N-¹H RDCs in PAG (23 res) and PEG
  (16 res) from Roberto Tejero, RPI.
* **GmR58A**: BMRB 16746 (NESG target).

All data files are bundled in ``tests/data/`` so the suite is fully
self-contained and does not require a separate diff-biophys checkout.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from diff_biophys.geometry.backbone import (
    compute_phi_psi,
    get_backbone_coords,
    get_residue_info,
    load_pdb_model,
    make_backbone_builder,
)
from diff_biophys.nmr.io import load_rdc_table
from diff_biophys.nmr.rdc import make_rdc_refinement_fns

from diff_integrator.loss import JointLoss
from diff_integrator.optimizer import IntegrativeRefiner
from diff_integrator.terms.chemical_shifts import CAShiftLoss
from diff_integrator.terms.geometry import GeometryLoss
from diff_integrator.terms.nmr import FixedTensorRDCLoss

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
UTILS_DIR = Path(__file__).parent / "utils"
sys.path.insert(0, str(UTILS_DIR))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def kabsch_rmsd(A: np.ndarray, B: np.ndarray) -> float:
    """Kabsch-optimal RMSD between two coordinate sets (in Å)."""
    A = np.array(A).reshape(-1, 3)
    B = np.array(B).reshape(-1, 3)
    A = A - A.mean(axis=0)
    B = B - B.mean(axis=0)
    U, _S, Vt = np.linalg.svd(A.T @ B)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    return float(np.sqrt(np.mean(np.sum((A @ R - B) ** 2, axis=1))))


# ---------------------------------------------------------------------------
# Test 1: 2KZV joint refinement (published NMR validation)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_2kzv_joint_refinement_published_validation() -> None:
    """
    Validates end-to-end joint refinement of 2KZV against published experimental data.

    Reference: Li et al. (2023), PMID 37257257.

    Assertions
    ----------
    * Both PAG and PEG Q-factors must decrease from their baseline values.
    * PEG Q-factor must drop below 0.20 (showing genuine improvement).
    * Kabsch RMSD to starting structure must stay below 2.5 Å (no unravelling).
    """
    bench_dir = DATA_DIR / "2KZV"
    if not bench_dir.exists():
        pytest.skip(f"Benchmark data not found at {bench_dir}")

    from parse_bmrb import load_bmrb_shifts  # noqa: PLC0415 (bundled utility)

    # --- Load structure ---
    struct = load_pdb_model(bench_dir / "2KZV.pdb", model_id=1)
    res_ids, res_names = get_residue_info(struct)
    coords = get_backbone_coords(struct)
    build_backbone = make_backbone_builder(len(res_ids), coords[:3])
    init_phi, init_psi = compute_phi_psi(coords)

    # --- Chemical shifts ---
    bmrb = load_bmrb_shifts(bench_dir / "bmrb17020.str")
    ca_exp = bmrb["CA"]
    ca_loss = CAShiftLoss(ca_exp["res_id"], ca_exp["shift"], res_ids, list(res_names))

    # --- RDC losses (fixed-tensor) ---
    rdc_pag = load_rdc_table(bench_dir / "rdc_PAG.tsv")["PAG"]
    rdc_peg = load_rdc_table(bench_dir / "rdc_PEG.tsv")["PEG"]

    loss_pag, q_pag_fn, tensor_pag, _ = make_rdc_refinement_fns(
        rdc_pag["res_id"], rdc_pag["rdc"], res_ids
    )
    loss_peg, q_peg_fn, tensor_peg, _ = make_rdc_refinement_fns(
        rdc_peg["res_id"], rdc_peg["rdc"], res_ids
    )

    rdc_term_pag = FixedTensorRDCLoss(loss_pag, tensor_pag, update_interval=25)
    rdc_term_peg = FixedTensorRDCLoss(loss_peg, tensor_peg, update_interval=25)
    geom_loss = GeometryLoss(target_coords=coords, target_weight=1.0)

    # --- Baseline Q-factors (before refinement) ---
    init_coords = build_backbone(init_phi, init_psi)
    rdc_term_pag.initialize_tensor(init_coords)
    rdc_term_peg.initialize_tensor(init_coords)

    init_q_pag = float(q_pag_fn(init_coords))
    init_q_peg = float(q_peg_fn(init_coords))

    # --- Optimization (abbreviated for test speed) ---
    import jax  # noqa: PLC0415
    import optax  # noqa: PLC0415

    joint_loss = JointLoss(
        [
            (geom_loss, 5.0),
            (ca_loss, 1.0),
            (rdc_term_pag, 1.0),
            (rdc_term_peg, 1.0),
        ]
    )

    optimizer = optax.adam(0.01)
    opt_state = optimizer.init((init_phi, init_psi))
    params = (init_phi, init_psi)

    @jax.jit
    def step(p, state):  # type: ignore[no-untyped-def]
        def obj(pp):  # type: ignore[no-untyped-def]
            c = build_backbone(pp[0], pp[1])
            return joint_loss(pp, c)

        loss_val, grads = jax.value_and_grad(obj)(p)
        updates, new_state = optimizer.update(grads, state)
        return optax.apply_updates(p, updates), new_state, loss_val

    EPOCHS = 150
    for epoch in range(EPOCHS):
        curr_coords = build_backbone(params[0], params[1])
        rdc_term_pag.maybe_update_tensor(curr_coords, epoch)
        rdc_term_peg.maybe_update_tensor(curr_coords, epoch)
        params, opt_state, _ = step(params, opt_state)

    # --- Final evaluation using the dedicated q_eval_fn (re-fits tensor fresh) ---
    final_phi, final_psi = params
    final_coords = build_backbone(final_phi, final_psi)

    final_q_pag = float(q_pag_fn(final_coords))
    final_q_peg = float(q_peg_fn(final_coords))
    rmsd = kabsch_rmsd(init_coords, final_coords)

    # --- Assertions ---
    assert final_q_pag < init_q_pag, (
        f"PAG Q-factor did not decrease: {init_q_pag:.3f} → {final_q_pag:.3f}"
    )
    assert final_q_peg < init_q_peg, (
        f"PEG Q-factor did not decrease: {init_q_peg:.3f} → {final_q_peg:.3f}"
    )
    assert final_q_peg < 0.20, (
        f"PEG Q-factor did not descend sufficiently: {final_q_peg:.3f} (expected < 0.20)"
    )
    assert rmsd < 2.5, (
        f"Structural drift too high: {rmsd:.2f} Å — GeometryLoss failed to restrain backbone"
    )


# ---------------------------------------------------------------------------
# Test 2: GmR58A Cα shift refinement
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_gmr58a_shift_refinement() -> None:
    """
    Validates end-to-end Cα shift-driven dihedral refinement of GmR58A.

    Data source: BMRB 16746 (NESG target GmR58A).

    Assertions
    ----------
    * Cα shift loss (RMSD in ppm) must decrease.
    * Kabsch RMSD to starting structure must stay below 2.0 Å.
    """
    bench_dir = DATA_DIR / "GmR58A"
    if not bench_dir.exists():
        pytest.skip(f"Benchmark data not found at {bench_dir}")

    from parse_nmrstar import load_bmrb_shifts  # noqa: PLC0415 (bundled utility)

    struct = load_pdb_model(bench_dir / "2KUT.pdb", model_id=1)
    res_ids, res_names = get_residue_info(struct)
    coords = get_backbone_coords(struct)
    build_backbone = make_backbone_builder(len(res_ids), coords[:3])
    init_phi, init_psi = compute_phi_psi(coords)

    bmrb = load_bmrb_shifts(bench_dir / "bmrb16746_GmR58A.str")
    ca_exp = bmrb["CA"]
    ca_loss = CAShiftLoss(ca_exp["res_id"], ca_exp["shift"], res_ids, list(res_names))
    geom_loss = GeometryLoss(target_coords=coords, target_weight=1.0)

    joint_loss = JointLoss(
        [
            (geom_loss, 1.0),
            (ca_loss, 1.0),
        ]
    )

    init_coords = build_backbone(init_phi, init_psi)
    init_loss = float(ca_loss((init_phi, init_psi), init_coords))

    refiner = IntegrativeRefiner(loss_fn=joint_loss)
    result = refiner.run(
        init_params=(init_phi, init_psi),
        epochs=150,
        learning_rate=0.01,
        kinematics_fn=lambda p: build_backbone(p[0], p[1]),
    )

    final_phi, final_psi = result.final_params
    final_coords = build_backbone(final_phi, final_psi)
    final_loss = float(ca_loss((final_phi, final_psi), final_coords))
    rmsd = kabsch_rmsd(init_coords, final_coords)

    assert final_loss < init_loss, (
        f"Cα shift loss did not decrease: {init_loss:.3f} → {final_loss:.3f} ppm"
    )
    assert rmsd < 2.0, f"Structural drift too high: {rmsd:.2f} Å"
