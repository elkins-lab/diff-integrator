"""
Benchmark: Joint NMR refinement of 2KZV (CvR118A) using diff-integrator.

Protein
-------
CV_0373(175-257) from *Chromobacterium violaceum*, NESG target CvR118A.
PDB: 2KZV  |  BMRB: 17020

Reference
---------
Li, Spaman, Tejero, Montelione et al. (2023).
*Blind assessment of monomeric AlphaFold2 protein structure models with
experimental NMR data.*  PMID 37257257.

Experimental data (provided by Roberto Tejero, RPI)
----------------------------------------------------
* 91 Cα chemical shifts from BMRB 17020.
* 23 ¹⁵N–¹H RDCs in PAG alignment medium.
* 16 ¹⁵N–¹H RDCs in PEG alignment medium.

Published Q-factor benchmarks (Table 5 of Li et al., NMR medoid)
-----------------------------------------------------------------
* PAG: Q = 0.18  (primary benchmark; 23 RDCs, ratio 4.6× tensor params)
* PEG: Q = 0.36  (supplementary only; 16 RDCs, ratio 3.2× — underdetermined)

Baseline from NMR model 1 (pre-refinement)
-------------------------------------------
* PAG: Q = 0.31
* PEG: Q = 0.37

Design choices
--------------
1. **Internal coordinates**: Optimization is over backbone (φ, ψ) dihedral
   angles, not Cartesian coordinates.  A NeRF-based builder reconstructs
   Cartesian coordinates at each step.

2. **Fixed-tensor RDC loss** (``FixedTensorRDCLoss``):
   The Saupe alignment tensor is fitted from the current backbone, then held
   fixed during gradient descent using ``jax.lax.stop_gradient``.  The tensor
   is re-fitted every ``TENSOR_UPDATE_INTERVAL`` epochs via a
   ``per_epoch_callbacks`` entry in ``refiner.run()``.
   This is the standard X-PLOR/CNS/PALES approach and prevents the optimizer
   from trivially driving Q→0 by exploiting tensor degeneracy.

3. **GeometryLoss anchor**:
   A harmonic restraint to the initial NMR coordinates prevents physically
   impossible global backbone unravelling under the highly degenerate RDC
   potential.  The restraint is strong enough to limit structural drift to
   ~1–2 Å while allowing genuine local improvements.

4. **PEG caution**:
   With only 16 RDCs against 5 tensor parameters (ratio 3.2×), the PEG medium
   is severely underdetermined.  Any final Q(PEG) well below the published
   NMR medoid value (0.36) should be interpreted as overfitting, not as a
   genuine structural improvement.  PAG (23 RDCs, ratio 4.6×) is the primary
   validation metric.

5. **RDC cross-validation** (PAG only):
   20% of PAG measurements (≈4 RDCs) are held out as a validation set.  The
   training loss uses only the remaining 19 RDCs; ``evaluate_validation_q()``
   evaluates the held-out set with the *training-fitted* tensor.  If training
   Q drops while validation Q stays flat, overfitting is occurring.  PEG is
   not split (only 16 RDCs; any split would leave < 6 for training).

6. **Auto-weight by overdetermination ratio**:
   ``FixedTensorRDCLoss.suggested_weight()`` scales the RDC term weight
   proportionally to ``n_train_rdcs / (5 × 10)`` — a medium at the ideal
   ratio of 10× gets ``base_weight`` unchanged.  PAG (≈19 train, ratio 3.8×)
   gets weight ≈ 0.38; PEG (16 total, ratio 3.2×) gets weight ≈ 0.32.  Both
   are automatically down-weighted relative to the geometry anchor.

7. **Training loop**: A single ``IntegrativeRefiner.run()`` call with
   ``per_epoch_callbacks`` for periodic tensor updates and Q-factor logging.
   No external loop or manual optax bookkeeping is needed.
"""

import sys
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import optax
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
from diff_integrator.terms.nmr import FixedTensorRDCLoss, make_rdc_cv_refinement_fns

# ---------------------------------------------------------------------------
# Data location — diff-biophys benchmark directory
# ---------------------------------------------------------------------------

BENCH_DIR = Path("../diff-biophys/benchmarks/2KZV").resolve()
sys.path.insert(0, str(BENCH_DIR))
from parse_bmrb import load_bmrb_shifts  # noqa: E402 (local benchmark utility)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

EPOCHS = 500
LEARNING_RATE = 0.01
TENSOR_UPDATE_INTERVAL = 50  # Re-fit Saupe tensor every N epochs
LOG_INTERVAL = 100           # Print Q diagnostics every N epochs

# Loss weights
WEIGHT_GEOMETRY = 5.0    # Strong anchor against degenerate RDC potential
WEIGHT_CA_SHIFTS = 1.0
# RDC weights are set automatically via suggested_weight() below;
# BASE_WEIGHT_RDC is the reference for a medium at the ideal ratio (10×).
BASE_WEIGHT_RDC = 1.0

# Cross-validation fraction for PAG (PEG is too small to split).
CV_FRACTION_PAG = 0.2


def kabsch_rmsd(A: np.ndarray, B: np.ndarray) -> float:
    """Compute the least-RMSD between two coordinate sets after optimal superposition.

    Uses the Kabsch algorithm to find the rotation that minimises the RMSD.

    Args:
        A: ``(N, 3)`` reference coordinates.
        B: ``(N, 3)`` mobile coordinates.

    Returns:
        RMSD in Å after optimal superposition.
    """
    A = np.array(A).reshape(-1, 3)
    B = np.array(B).reshape(-1, 3)
    A = A - A.mean(axis=0)
    B = B - B.mean(axis=0)
    U, _S, Vt = np.linalg.svd(A.T @ B)
    R = U @ Vt
    # Correct improper rotation (reflection) if det(R) < 0
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    A_rot = A @ R
    return float(np.sqrt(np.mean(np.sum((A_rot - B) ** 2, axis=1))))


def main() -> None:
    print("=" * 60)
    print("diff-integrator Benchmark: 2KZV (CvR118A)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load structure
    # ------------------------------------------------------------------
    struct = load_pdb_model(BENCH_DIR / "2KZV.pdb", model_id=1)
    res_ids, res_names = get_residue_info(struct)
    coords = get_backbone_coords(struct)  # (3N, 3) N-CA-C backbone
    n_res = len(res_ids)

    # Build the NeRF backbone function: (φ, ψ) → (3N, 3) Cartesian coords.
    # The first three atoms (N₀, CA₀, C₀) are used as the frame anchor.
    build_backbone = make_backbone_builder(n_res, coords[:3])
    init_phi, init_psi = compute_phi_psi(coords)

    print(f"\nStructure: {n_res} residues, {len(coords)} backbone atoms")

    # ------------------------------------------------------------------
    # 2. Set up Cα chemical shift loss
    # ------------------------------------------------------------------
    bmrb = load_bmrb_shifts(BENCH_DIR / "bmrb17020.str")
    ca_exp = bmrb["CA"]
    ca_loss = CAShiftLoss(ca_exp["res_id"], ca_exp["shift"], res_ids, list(res_names))
    print(f"Cα shifts: {len(ca_exp['res_id'])} matched residues")

    # ------------------------------------------------------------------
    # 3. Set up RDC losses (fixed-tensor approach with CV split for PAG)
    # ------------------------------------------------------------------
    rdc_pag = load_rdc_table(BENCH_DIR / "rdc_PAG.tsv")["PAG"]
    rdc_peg = load_rdc_table(BENCH_DIR / "rdc_PEG.tsv")["PEG"]

    # PAG: use cross-validation split (20% held out for monitoring).
    # make_rdc_cv_refinement_fns returns:
    #   loss_fn      — MSE on training RDCs only (used in gradient)
    #   q_eval_fn    — Q-factor on ALL matched RDCs (monitoring; refits tensor)
    #   make_tensor_fn — fits tensor from training RDCs (for periodic updates)
    #   val_q_fn     — Q on held-out RDCs using training-fitted tensor
    #   n_train, n_val — split counts
    loss_pag, q_pag, tensor_pag, val_q_pag, n_pag_train, n_pag_val = (
        make_rdc_cv_refinement_fns(
            rdc_pag["res_id"], rdc_pag["rdc"], res_ids,
            cv_fraction=CV_FRACTION_PAG,
        )
    )

    # PEG: too few RDCs to split; use all for training, no CV.
    loss_peg, q_peg, tensor_peg, n_peg = make_rdc_refinement_fns(
        rdc_peg["res_id"], rdc_peg["rdc"], res_ids
    )

    print(f"RDC PAG: {n_pag_train} train + {n_pag_val} val residues  "
          f"(CV fraction {CV_FRACTION_PAG:.0%})")
    print(f"RDC PEG: {n_peg} matched residues (no CV — underdetermined)")

    # Wrap in FixedTensorRDCLoss.
    rdc_term_pag = FixedTensorRDCLoss(
        loss_pag, tensor_pag,
        update_interval=TENSOR_UPDATE_INTERVAL,
        n_rdcs=n_pag_train,
        val_q_eval_fn=val_q_pag,
    )
    rdc_term_peg = FixedTensorRDCLoss(
        loss_peg, tensor_peg,
        update_interval=TENSOR_UPDATE_INTERVAL,
        n_rdcs=n_peg,
    )

    # Auto-weight: scale each term by its overdetermination ratio.
    weight_pag = rdc_term_pag.suggested_weight(BASE_WEIGHT_RDC)
    weight_peg = rdc_term_peg.suggested_weight(BASE_WEIGHT_RDC)
    print(f"  Auto-weights: PAG={weight_pag:.3f}  PEG={weight_peg:.3f}  "
          f"(base={BASE_WEIGHT_RDC}, ideal ratio=10×)")

    # ------------------------------------------------------------------
    # 4. Set up geometry anchor
    # ------------------------------------------------------------------
    # Harmonic restraint to NMR model 1 coordinates prevents unravelling.
    # The restrained coords are the raw PDB backbone, not a rebuilt chain.
    geom_loss = GeometryLoss(target_coords=coords, target_weight=1.0)

    # ------------------------------------------------------------------
    # 5. Build joint loss and refiner
    # ------------------------------------------------------------------
    joint_loss = JointLoss(
        [
            (geom_loss, WEIGHT_GEOMETRY),    # Strong anchor vs. RDC degeneracy
            (ca_loss, WEIGHT_CA_SHIFTS),
            (rdc_term_pag, weight_pag),       # Auto-weighted by ratio
            (rdc_term_peg, weight_peg),       # Auto-weighted by ratio
        ]
    )

    # Use plain Adam (no gradient clipping) to match the original design.
    adam = optax.adam(LEARNING_RATE)
    refiner = IntegrativeRefiner(loss_fn=joint_loss)

    # ------------------------------------------------------------------
    # 6. Evaluate baseline (before refinement)
    # ------------------------------------------------------------------
    init_coords = build_backbone(init_phi, init_psi)
    rdc_term_pag.initialize_tensor(init_coords)
    rdc_term_peg.initialize_tensor(init_coords)

    print("\n--- Baseline (NMR model 1, pre-refinement) ---")
    print(f"  Cα RMSD:  {float(ca_loss((init_phi, init_psi), init_coords)):.3f} ppm")
    print(f"  Q (PAG):  {float(q_pag(init_coords)):.3f}  (published NMR medoid: 0.18)")
    print(f"  Q (PEG):  {float(q_peg(init_coords)):.3f}  (published NMR medoid: 0.36)")

    # ------------------------------------------------------------------
    # 7. Per-epoch callback: tensor updates + Q logging
    # ------------------------------------------------------------------
    print(f"\nRefining for {EPOCHS} epochs "
          f"(tensor update every {TENSOR_UPDATE_INTERVAL} steps)...")

    def epoch_callback(epoch: int, params: Any, coords_arg: jnp.ndarray) -> None:
        # Re-fit Saupe tensors outside the gradient tape
        rdc_term_pag.maybe_update_tensor(coords_arg, epoch)
        rdc_term_peg.maybe_update_tensor(coords_arg, epoch)

        # Q-factor logging
        if (epoch + 1) % LOG_INTERVAL == 0:
            q_p_train = float(q_pag(coords_arg))
            q_p_val   = rdc_term_pag.evaluate_validation_q(coords_arg)
            q_e       = float(q_peg(coords_arg))
            val_str   = f"  Q(PAG val)={q_p_val:.3f}" if q_p_val is not None else ""
            print(
                f"  Epoch {epoch + 1:4d}: "
                f"Q(PAG train)={q_p_train:.3f}{val_str}  Q(PEG)={q_e:.3f}"
            )

    # ------------------------------------------------------------------
    # 8. Refinement — single refiner.run() call
    # ------------------------------------------------------------------
    result = refiner.run(
        init_params=(init_phi, init_psi),
        epochs=EPOCHS,
        optimizer=adam,                                       # plain Adam, no clip
        kinematics_fn=lambda p: build_backbone(p[0], p[1]),
        per_epoch_callbacks=[epoch_callback],
    )

    # ------------------------------------------------------------------
    # 9. Evaluate final results
    # ------------------------------------------------------------------
    final_phi, final_psi = result.final_params
    final_coords = build_backbone(final_phi, final_psi)

    rmsd = kabsch_rmsd(init_coords, final_coords)

    q_pag_final     = float(q_pag(final_coords))
    q_pag_val_final = rdc_term_pag.evaluate_validation_q(final_coords)
    q_peg_final     = float(q_peg(final_coords))

    print("\n--- Final Results ---")
    print(f"  Cα RMSD:             "
          f"{float(ca_loss((final_phi, final_psi), final_coords)):.3f} ppm")
    print(f"  Q (PAG train):       {q_pag_final:.3f}  (published target: ≤0.22)")
    if q_pag_val_final is not None:
        print(f"  Q (PAG val):         {q_pag_val_final:.3f}  "
              f"(held-out {n_pag_val} RDCs — overfitting check)")
    print(f"  Q (PEG):             {q_peg_final:.3f}  (caution: underdetermined)")
    print(f"  Structural RMSD:     {rmsd:.3f} Å to NMR model 1")

    if q_peg_final < 0.18:
        print("\n  ⚠️  WARNING: Q(PEG) is suspiciously low — likely overfitting the "
              "16-RDC dataset.")
        print("     PAG Q-factor is the reliable primary metric.")
    if q_pag_val_final is not None and q_pag_val_final > q_pag_final + 0.05:
        print("\n  ⚠️  WARNING: PAG validation Q is notably higher than training Q.")
        print("     The improvement may not generalise to held-out measurements.")

    # ------------------------------------------------------------------
    # 10. Save artefacts
    # ------------------------------------------------------------------
    results_dir = Path("benchmarks/results/2KZV")
    results_dir.mkdir(parents=True, exist_ok=True)

    np.save(results_dir / "loss_history.npy", np.array(result.loss_history))

    import biotite.structure.io.pdb as pdb  # noqa: PLC0415

    # Save the raw PDB backbone as initial.pdb (NOT the NeRF rebuild,
    # which may have minor numeric differences from the true model 1).
    backbone_mask = np.isin(struct.atom_name, ["N", "CA", "C"])
    init_struct = struct[backbone_mask].copy()
    f_init = pdb.PDBFile()
    f_init.set_structure(init_struct)
    f_init.write(str(results_dir / "initial.pdb"))

    # Save the refined coordinates into the same atom layout.
    final_struct = init_struct.copy()
    final_struct.coord = np.array(final_coords)
    f_final = pdb.PDBFile()
    f_final.set_structure(final_struct)
    f_final.write(str(results_dir / "final.pdb"))

    print(f"\n  Results saved to {results_dir}/")


if __name__ == "__main__":
    main()
