"""
Benchmark: Cartesian Cα chemical shift + RDC joint refinement of HR2876B (2LTM).

Protein
-------
N-terminal domain of human NFU1 (Iron-Sulfur Cluster Scaffold Homolog).
NESG target: HR2876B.
PDB: 2LTM | BMRB: 18489

Purpose
-------
This benchmark demonstrates the **Cartesian + bond-angle penalty** approach to
protein backbone refinement extended with **two RDC media** from BMRB 18489.

HR2876B has two exceptionally well-determined ¹⁵N–¹H RDC datasets:
  * RDC_list_1 (PEG alignment medium):  72 RDCs over residues 14–107  (14.4×)
  * RDC_list_2 (Pf1 phage medium):      75 RDCs over residues 13–107  (15.0×)

Both are far above the minimum recommended ratio (~5×), making them ideal inputs for
``FixedTensorRDCLoss``.  Adding them to the Cartesian benchmark substantially increases
the experimental constraint-to-DOF ratio and is expected to yield larger Cα RMSD
improvements and meaningful Q-factor reductions.

Comparison with previous runs
------------------------------
NeRF benchmark (benchmark_HR2876B.py):
  * Parameterisation: (φ, ψ) dihedral angles, NeRF builder
  * Observables: Cα shifts only
  * NeRF drift: ~14 Å — severely distorts gradient landscape
  * Cα RMSD improvement: −0.011 ppm (500 epochs)

Cartesian benchmark, run 1 (shifts only):
  * Parameterisation: raw Cα coordinates, BondLength + BondAngle penalties
  * Observables: Cα shifts only
  * NeRF drift: zero
  * Cα RMSD improvement: −0.145 ppm (stopped at epoch 894 / 2000)

Cartesian benchmark, run 2 (shifts + RDCs — this file):
  * Same Cartesian parameterisation as run 1
  * Added: FixedTensorRDCLoss for RDC_list_1 (72 ¹⁵N–¹H PEG,  14.4×)
           FixedTensorRDCLoss for RDC_list_2 (75 ¹⁵N–¹H Pf1,  15.0×)
  * Both RDC media: 20% CV split + suggested_weight() auto-scaling
  * Training loop: IntegrativeRefiner.run() with per_epoch_callbacks for
    periodic tensor updates and Q-factor monitoring, weight_schedules for
    the annealed anchor, and built-in EarlyStopping on the Cα shift term.

Design choices
--------------
1. **Direct Cartesian coordinates** — raw PDB backbone, no NeRF builder.

2. **Bond-length penalty** (``BondLengthPenalty``, weight=50):
   Harmonic restraint on N–CA, CA–C, C–N bonds to Engh & Huber ideal values.

3. **Bond-angle penalty** (``BondAnglePenalty``, weight=10):
   Harmonic restraint on N–CA–C, CA–C–N, C–N–CA angles.

4. **Annealed position anchor** (``GeometryLoss``, term index 0):
   Decays 10.0 → 0.1 over τ=100 epochs via ``ExponentialDecaySchedule``,
   passed as ``weight_schedules={0: anchor_schedule}`` to ``refiner.run()``.

5. **FixedTensorRDCLoss** (both media, term indices 4 & 5):
   ``jax.lax.stop_gradient`` freezes the tensor during backprop; re-fitted
   every ``TENSOR_UPDATE_INTERVAL`` epochs via a ``per_epoch_callbacks`` entry.
   CV split monitors overfitting.

6. **Per-epoch Q monitoring** (``per_epoch_callbacks``):
   A single callback prints Cα RMSD and Q-factors every ``LOG_INTERVAL``
   epochs and re-fits both Saupe tensors at ``TENSOR_UPDATE_INTERVAL``
   boundaries — all inside ``refiner.run()`` without a manual outer loop.

7. **Built-in EarlyStopping** on Cα shift term (index 1):
   ``EarlyStopping(term_index=1, patience=75)`` is passed directly to
   ``refiner.run()``.  No manual stopping logic is needed.
"""

import sys
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from diff_biophys.geometry.backbone import (
    get_backbone_coords,
    get_residue_info,
    load_pdb_model,
)

from diff_integrator.loss import JointLoss
from diff_integrator.optimizer import EarlyStopping, IntegrativeRefiner
from diff_integrator.schedules import ExponentialDecaySchedule
from diff_integrator.terms.bond_geometry import make_backbone_bond_geometry
from diff_integrator.terms.chemical_shifts import CartesianCAShiftLoss
from diff_integrator.terms.geometry import GeometryLoss
from diff_integrator.terms.nmr import FixedTensorRDCLoss, make_rdc_cv_refinement_fns

# ---------------------------------------------------------------------------
# Data location
# ---------------------------------------------------------------------------

BENCH_DIR = Path("../diff-biophys/benchmarks/HR2876B").resolve()
sys.path.insert(0, str(BENCH_DIR))
from parse_nmrstar import load_bmrb_rdcs, load_bmrb_shifts  # noqa: E402

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

EPOCHS = 2000
LEARNING_RATE = 0.005

WEIGHT_ANCHOR_INITIAL = 10.0
WEIGHT_ANCHOR_FINAL   = 0.1
WEIGHT_ANCHOR_DECAY   = 100     # τ epochs

WEIGHT_BOND      = 50.0
WEIGHT_ANGLE     = 10.0
WEIGHT_CA_SHIFTS = 1.0
BASE_WEIGHT_RDC  = 1.0   # reference at ideal 10× overdetermination

TENSOR_UPDATE_INTERVAL = 50   # re-fit Saupe tensor every N epochs
CV_FRACTION_RDC        = 0.2
LOG_INTERVAL           = 100  # print diagnostics every N epochs

# Early stopping on Cα shift term (term index 1)
ES_PATIENCE  = 75
ES_MIN_DELTA = 5e-5


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


def main() -> None:
    print("=" * 65)
    print("diff-integrator Benchmark: HR2876B (2LTM) — Cartesian + RDC")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Load structure
    # ------------------------------------------------------------------
    struct = load_pdb_model(BENCH_DIR / "2LTM.pdb", model_id=1)
    res_ids, res_names = get_residue_info(struct)
    coords = get_backbone_coords(struct)     # (3N, 3) raw PDB backbone
    n_res = len(res_ids)

    print(f"\nStructure: {n_res} residues, {len(coords)} backbone atoms")
    print("Parameterisation: Cartesian coordinates (no NeRF builder)")

    # ------------------------------------------------------------------
    # 2. Cα chemical shift loss  (JointLoss term index 1)
    # ------------------------------------------------------------------
    bmrb = load_bmrb_shifts(BENCH_DIR / "bmrb18489_HR2876B.str")
    ca_exp = bmrb["CA"]
    ca_loss = CartesianCAShiftLoss(
        ca_exp["res_id"], ca_exp["shift"], res_ids, list(res_names)
    )
    print(f"Cα shifts: {len(ca_exp['res_id'])} matched residues")

    # ------------------------------------------------------------------
    # 3. RDC losses  (JointLoss term indices 4, 5)
    # ------------------------------------------------------------------
    rdcs = load_bmrb_rdcs(BENCH_DIR / "bmrb18489_HR2876B.str")

    rdc1_data = rdcs["RDC_list_1"]   # 72 ¹⁵N–¹H, PEG medium
    rdc2_data = rdcs["RDC_list_2"]   # 75 ¹⁵N–¹H, Pf1 phage

    (loss_rdc1, q_rdc1, tensor_rdc1, val_q_rdc1, n1_train, n1_val) = (
        make_rdc_cv_refinement_fns(
            rdc1_data["res_id"], rdc1_data["rdc"], res_ids,
            cv_fraction=CV_FRACTION_RDC,
        )
    )
    (loss_rdc2, q_rdc2, tensor_rdc2, val_q_rdc2, n2_train, n2_val) = (
        make_rdc_cv_refinement_fns(
            rdc2_data["res_id"], rdc2_data["rdc"], res_ids,
            cv_fraction=CV_FRACTION_RDC,
        )
    )

    rdc_term1 = FixedTensorRDCLoss(
        loss_rdc1, tensor_rdc1,
        update_interval=TENSOR_UPDATE_INTERVAL,
        n_rdcs=n1_train,
        val_q_eval_fn=val_q_rdc1,
    )
    rdc_term2 = FixedTensorRDCLoss(
        loss_rdc2, tensor_rdc2,
        update_interval=TENSOR_UPDATE_INTERVAL,
        n_rdcs=n2_train,
        val_q_eval_fn=val_q_rdc2,
    )

    weight_rdc1 = rdc_term1.suggested_weight(BASE_WEIGHT_RDC)
    weight_rdc2 = rdc_term2.suggested_weight(BASE_WEIGHT_RDC)

    ratio1 = len(rdc1_data["res_id"]) / 5
    ratio2 = len(rdc2_data["res_id"]) / 5
    print(f"RDC list 1 (PEG):  {len(rdc1_data['res_id'])} total → "
          f"{n1_train} train + {n1_val} val  "
          f"(ratio={ratio1:.1f}×, weight={weight_rdc1:.3f})")
    print(f"RDC list 2 (Pf1):  {len(rdc2_data['res_id'])} total → "
          f"{n2_train} train + {n2_val} val  "
          f"(ratio={ratio2:.1f}×, weight={weight_rdc2:.3f})")

    # ------------------------------------------------------------------
    # 4. Bond penalties  (indices 2, 3)
    # ------------------------------------------------------------------
    bond_pen, angle_pen = make_backbone_bond_geometry(n_res)
    n_bonds  = bond_pen.bond_pairs.shape[0]
    n_angles = angle_pen.angle_triples.shape[0]
    print(f"Bond-length penalty: {n_bonds} bonds  (weight={WEIGHT_BOND})")
    print(f"Bond-angle penalty:  {n_angles} angles (weight={WEIGHT_ANGLE})")
    print(f"  Initial bond RMSD:  {bond_pen.bond_rmsd(coords):.4f} Å")
    print(f"  Initial angle RMSD: {angle_pen.angle_rmsd_deg(coords):.4f}°")

    # ------------------------------------------------------------------
    # 5. Annealed position anchor  (index 0)
    # ------------------------------------------------------------------
    anchor_loss     = GeometryLoss(target_coords=coords, target_weight=1.0)
    anchor_schedule = ExponentialDecaySchedule(
        initial_weight=WEIGHT_ANCHOR_INITIAL,
        final_weight=WEIGHT_ANCHOR_FINAL,
        decay_epochs=WEIGHT_ANCHOR_DECAY,
    )

    # ------------------------------------------------------------------
    # 6. Joint loss
    #    index 0 → anchor (annealed)       weight_schedules={0: anchor_schedule}
    #    index 1 → Cα shifts               ← EarlyStopping watches this
    #    index 2 → bond-length penalty
    #    index 3 → bond-angle penalty
    #    index 4 → RDC list 1 (PEG)
    #    index 5 → RDC list 2 (Pf1)
    # ------------------------------------------------------------------
    joint_loss = JointLoss([
        (anchor_loss, WEIGHT_ANCHOR_INITIAL),
        (ca_loss,     WEIGHT_CA_SHIFTS),
        (bond_pen,    WEIGHT_BOND),
        (angle_pen,   WEIGHT_ANGLE),
        (rdc_term1,   weight_rdc1),
        (rdc_term2,   weight_rdc2),
    ])

    refiner = IntegrativeRefiner(loss_fn=joint_loss)

    # ------------------------------------------------------------------
    # 7. Baseline
    # ------------------------------------------------------------------
    init_cj = jnp.array(coords)
    init_ca_rmsd = float(ca_loss(init_cj, init_cj))

    # Prime both tensors on the initial structure before refiner.run()
    rdc_term1.maybe_update_tensor(init_cj, epoch=0)
    rdc_term2.maybe_update_tensor(init_cj, epoch=0)
    init_q1 = float(q_rdc1(init_cj))
    init_q2 = float(q_rdc2(init_cj))

    print("\n--- Baseline (NMR model 1, pre-refinement) ---")
    print(f"  Cα RMSD:        {init_ca_rmsd:.3f} ppm")
    print(f"  Q (RDC list 1, PEG): {init_q1:.3f}  (14.4× overdetermined)")
    print(f"  Q (RDC list 2, Pf1): {init_q2:.3f}  (15.0× overdetermined)")

    # ------------------------------------------------------------------
    # 8. Per-epoch callback
    #
    # Registered via ``per_epoch_callbacks`` in refiner.run().  Called once
    # per epoch with (epoch, params, coords).  Handles:
    #   (a) Re-fitting both Saupe tensors at TENSOR_UPDATE_INTERVAL boundaries.
    #   (b) Printing Cα RMSD and Q-factors at LOG_INTERVAL boundaries.
    # ------------------------------------------------------------------
    print(f"\nRefining for {EPOCHS} epochs  "
          f"(lr={LEARNING_RATE}, anchor τ={WEIGHT_ANCHOR_DECAY}, "
          f"tensor update every {TENSOR_UPDATE_INTERVAL})...")
    print("  Epoch |  Cα(ppm) |  Q(PEG) Qval |  Q(Pf1) Qval")
    print("  " + "-" * 55)

    def epoch_callback(epoch: int, params: Any, coords_arg: jnp.ndarray) -> None:
        # Tensor update: re-fit Saupe matrices from current backbone
        rdc_term1.maybe_update_tensor(coords_arg, epoch=epoch)
        rdc_term2.maybe_update_tensor(coords_arg, epoch=epoch)

        # Periodic Q / Cα RMSD logging
        if epoch % LOG_INTERVAL == 0:
            ca   = float(ca_loss(coords_arg, coords_arg))
            q1   = float(q_rdc1(coords_arg))
            q2   = float(q_rdc2(coords_arg))
            q1v  = rdc_term1.evaluate_validation_q(coords_arg)
            q2v  = rdc_term2.evaluate_validation_q(coords_arg)
            q1vs = f"{q1v:.3f}" if q1v is not None else "  —  "
            q2vs = f"{q2v:.3f}" if q2v is not None else "  —  "
            print(f"  {epoch + 1:5d} | {ca:.3f}    | {q1:.3f}  {q1vs} | {q2:.3f}  {q2vs}")

    # ------------------------------------------------------------------
    # 9. Refinement — single refiner.run() call
    # ------------------------------------------------------------------
    result = refiner.run(
        init_params=jnp.array(coords),
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_schedules={0: anchor_schedule},
        per_epoch_callbacks=[epoch_callback],
        log_interval=LOG_INTERVAL,
        early_stopping=EarlyStopping(
            term_index=1,           # Cα shift term
            patience=ES_PATIENCE,
            min_delta=ES_MIN_DELTA,
        ),
    )

    # ------------------------------------------------------------------
    # 10. Final results
    # ------------------------------------------------------------------
    best_cj = jnp.array(result.best_params)

    final_ca_rmsd = float(ca_loss(best_cj, best_cj))
    final_q1      = float(q_rdc1(best_cj))
    final_q2      = float(q_rdc2(best_cj))
    final_q1_val  = rdc_term1.evaluate_validation_q(best_cj)
    final_q2_val  = rdc_term2.evaluate_validation_q(best_cj)
    final_bond    = bond_pen.bond_rmsd(best_cj)
    final_angle   = angle_pen.angle_rmsd_deg(best_cj)
    struct_rmsd   = kabsch_rmsd(coords, np.array(result.best_params))

    print("\n--- Final Results (best checkpoint) ---")
    print(f"  Cα RMSD:             {final_ca_rmsd:.3f} ppm  "
          f"(Δ = {final_ca_rmsd - init_ca_rmsd:+.3f})")
    print(f"  Q (RDC list 1, PEG): {final_q1:.3f}  "
          f"(Δ = {final_q1 - init_q1:+.3f})"
          + (f"  val={final_q1_val:.3f}" if final_q1_val is not None else ""))
    print(f"  Q (RDC list 2, Pf1): {final_q2:.3f}  "
          f"(Δ = {final_q2 - init_q2:+.3f})"
          + (f"  val={final_q2_val:.3f}" if final_q2_val is not None else ""))
    print(f"  Bond RMSD:           {final_bond:.4f} Å  (target < 0.05 Å)")
    print(f"  Angle RMSD:          {final_angle:.3f}°  (target < 3°)")
    print(f"  Structural RMSD:     {struct_rmsd:.3f} Å vs raw PDB model 1")
    print(f"\n  Epochs run:          {result.epochs_run} / {EPOCHS}")
    print(f"  Stopped early:       {result.stopped_early}")
    if result.stopped_early:
        print(f"  Stopped at epoch:    {result.stopped_at_epoch + 1}")
        print(f"  Triggered by:        {result.early_stopping_triggered_by}")
    print(f"  Best checkpoint:     epoch {result.best_epoch + 1}")

    if final_q1_val is not None and final_q1_val > final_q1 + 0.05:
        print("\n  ⚠️  WARNING: RDC list 1 val Q notably higher than train Q.")
    if final_q2_val is not None and final_q2_val > final_q2 + 0.05:
        print("  ⚠️  WARNING: RDC list 2 val Q notably higher than train Q.")

    # ------------------------------------------------------------------
    # 11. Save artefacts
    # ------------------------------------------------------------------
    results_dir = Path("benchmarks/results/HR2876B_cartesian")
    results_dir.mkdir(parents=True, exist_ok=True)

    np.save(results_dir / "loss_history.npy", np.array(result.loss_history))
    np.save(results_dir / "rdc1_q_before.npy", np.array(init_q1))
    np.save(results_dir / "rdc1_q_after.npy",  np.array(final_q1))
    np.save(results_dir / "rdc2_q_before.npy", np.array(init_q2))
    np.save(results_dir / "rdc2_q_after.npy",  np.array(final_q2))

    import biotite.structure.io.pdb as pdb  # noqa: PLC0415

    backbone_mask = np.isin(struct.atom_name, ["N", "CA", "C"])
    init_struct = struct[backbone_mask].copy()

    f_init = pdb.PDBFile()
    f_init.set_structure(init_struct)
    f_init.write(str(results_dir / "initial.pdb"))

    best_struct = init_struct.copy()
    best_struct.coord = np.array(result.best_params)
    f_best = pdb.PDBFile()
    f_best.set_structure(best_struct)
    f_best.write(str(results_dir / "final.pdb"))

    print(f"\n  Results saved to {results_dir}/")
    print(f"  Best checkpoint at epoch {result.best_epoch + 1} / {result.epochs_run}")


if __name__ == "__main__":
    main()
