"""
Benchmark: Cartesian Cα chemical shift refinement of HR2876B (2LTM).

Protein
-------
N-terminal domain of human NFU1 (Iron-Sulfur Cluster Scaffold Homolog).
NESG target: HR2876B.
PDB: 2LTM | BMRB: 18489

Purpose
-------
This benchmark demonstrates the **Cartesian + bond-angle penalty** approach to
protein backbone refinement, which eliminates the NeRF geometric drift problem
that limits the standard internal-coordinate (φ, ψ dihedral) approach.

HR2876B is the natural demonstration target because its 107 residues produce
approximately **14 Å** of NeRF drift (NeRF-rebuilt backbone vs. raw PDB), which
means the NeRF parameterisation starts from a qualitatively different structure
than the real NMR model.  The Cartesian approach has no such limitation.

Comparison with NeRF benchmark
-------------------------------
The standard ``benchmark_HR2876B.py`` uses NeRF internal coordinates:
  * Parameters:   (φ, ψ) angles, 2 × 107 = 214 parameters
  * Geometry:     exactly ideal Engh & Huber (hard constraint via NeRF builder)
  * NeRF drift:   ~14 Å from raw PDB → the optimizer fights this from epoch 0
  * Cα RMSD improvement: only −0.011 ppm over 500 epochs

This Cartesian benchmark:
  * Parameters:   (X, Y, Z) per backbone atom, 3 × 107 × 3 = 963 parameters
  * Geometry:     soft harmonic restraints on bond lengths and angles
  * NeRF drift:   zero — the starting coordinates ARE the raw PDB model 1
  * Expected improvement: substantially larger Cα RMSD reduction

Design choices
--------------
1. **Direct Cartesian coordinates**: The PDB backbone coordinates are used
   directly as the starting parameters.  No NeRF reconstruction is performed.

2. **Bond-length penalty** (``BondLengthPenalty``):
   Harmonic restraint on N–CA, CA–C, and C–N backbone bonds.  Weight = 50.0
   (stiff bonds; should maintain |Δbond| < 0.05 Å throughout optimization).

3. **Bond-angle penalty** (``BondAnglePenalty``):
   Harmonic restraint on N–CA–C, CA–C–N, and C–N–CA angles.  Weight = 10.0
   (softer than bonds; should maintain |Δangle| < 3°).

4. **Annealed position anchor** (``GeometryLoss``):
   Harmonic restraint to the raw PDB coordinates prevents global rigid-body
   drift and out-of-basin exploration.  Decays from 10.0 → 0.1 over τ = 100
   epochs (same schedule as GmR58A benchmark), allowing the shift gradient to
   dominate later in training.

5. **No NeRF**: ``kinematics_fn=None`` → ``IntegrativeRefiner`` uses the
   identity function (params ARE coords).

6. **Best-checkpoint**: ``result.best_params`` holds the iterate with the
   lowest total training loss.
"""

import sys
from pathlib import Path

import numpy as np
from diff_biophys.geometry.backbone import (
    get_backbone_coords,
    get_residue_info,
    load_pdb_model,
)

from diff_integrator.loss import JointLoss
from diff_integrator.optimizer import IntegrativeRefiner
from diff_integrator.schedules import ExponentialDecaySchedule
from diff_integrator.terms.bond_geometry import make_backbone_bond_geometry
from diff_integrator.terms.chemical_shifts import CartesianCAShiftLoss
from diff_integrator.terms.geometry import GeometryLoss

# ---------------------------------------------------------------------------
# Data location
# ---------------------------------------------------------------------------

BENCH_DIR = Path("../diff-biophys/benchmarks/HR2876B").resolve()
sys.path.insert(0, str(BENCH_DIR))
from parse_nmrstar import load_bmrb_shifts  # noqa: E402 (local benchmark utility)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

EPOCHS = 500
LEARNING_RATE = 0.005   # Lower LR than NeRF: Cartesian space has larger gradient norms

# Position anchor: strong early (prevents rigid-body drift), relaxed late
WEIGHT_ANCHOR_INITIAL = 10.0
WEIGHT_ANCHOR_FINAL   = 0.1
WEIGHT_ANCHOR_DECAY   = 100     # τ = 100 epochs

# Bond/angle penalties (hold geometry near Engh & Huber ideal throughout)
WEIGHT_BOND   = 50.0    # Stiff: target bond RMSD < 0.05 Å
WEIGHT_ANGLE  = 10.0    # Softer: target angle RMSD < 3°

# Experimental observable
WEIGHT_CA_SHIFTS = 1.0


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
    print("=" * 60)
    print("diff-integrator Benchmark: HR2876B (2LTM) — Cartesian")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load structure — use raw PDB coordinates directly
    # ------------------------------------------------------------------
    struct = load_pdb_model(BENCH_DIR / "2LTM.pdb", model_id=1)
    res_ids, res_names = get_residue_info(struct)
    coords = get_backbone_coords(struct)     # (3N, 3) raw PDB backbone
    n_res = len(res_ids)

    print(f"\nStructure: {n_res} residues, {len(coords)} backbone atoms")
    print("Parameterisation: Cartesian coordinates (no NeRF builder)")

    # ------------------------------------------------------------------
    # 2. Cα chemical shift loss
    # ------------------------------------------------------------------
    bmrb = load_bmrb_shifts(BENCH_DIR / "bmrb18489_HR2876B.str")
    ca_exp = bmrb["CA"]
    ca_loss = CartesianCAShiftLoss(
        ca_exp["res_id"], ca_exp["shift"], res_ids, list(res_names)
    )
    print(f"Cα shifts: {len(ca_exp['res_id'])} matched residues")

    # ------------------------------------------------------------------
    # 3. Bond-length and bond-angle penalties
    # ------------------------------------------------------------------
    bond_pen, angle_pen = make_backbone_bond_geometry(n_res)
    n_bonds = bond_pen.bond_pairs.shape[0]
    n_angles = angle_pen.angle_triples.shape[0]
    print(f"Bond-length penalty: {n_bonds} bonds  (weight={WEIGHT_BOND})")
    print(f"Bond-angle penalty:  {n_angles} angles (weight={WEIGHT_ANGLE})")

    # Measure initial geometry quality (from raw PDB — should be near ideal)
    init_bond_rmsd  = bond_pen.bond_rmsd(coords)
    init_angle_rmsd = angle_pen.angle_rmsd_deg(coords)
    print(f"  Initial bond RMSD:  {init_bond_rmsd:.4f} Å  "
          f"(ideal = 0; PDB bonds are not ideal Engh & Huber)")
    print(f"  Initial angle RMSD: {init_angle_rmsd:.4f}°")

    # ------------------------------------------------------------------
    # 4. Annealed position anchor (raw PDB coords as reference)
    # ------------------------------------------------------------------
    # The anchor target is the raw PDB model 1 backbone (NOT a NeRF rebuild).
    anchor_loss = GeometryLoss(target_coords=coords, target_weight=1.0)

    anchor_schedule = ExponentialDecaySchedule(
        initial_weight=WEIGHT_ANCHOR_INITIAL,
        final_weight=WEIGHT_ANCHOR_FINAL,
        decay_epochs=WEIGHT_ANCHOR_DECAY,
    )

    # ------------------------------------------------------------------
    # 5. Joint loss
    #    Term index 0 → anchor (annealed)
    #    Term index 1 → Cα shifts
    #    Term index 2 → bond-length penalty
    #    Term index 3 → bond-angle penalty
    # ------------------------------------------------------------------
    joint_loss = JointLoss([
        (anchor_loss, WEIGHT_ANCHOR_INITIAL),   # index 0 — weight will be annealed
        (ca_loss,     WEIGHT_CA_SHIFTS),          # index 1
        (bond_pen,    WEIGHT_BOND),               # index 2
        (angle_pen,   WEIGHT_ANGLE),              # index 3
    ])

    refiner = IntegrativeRefiner(loss_fn=joint_loss)

    # ------------------------------------------------------------------
    # 6. Baseline (before refinement)
    # ------------------------------------------------------------------
    init_ca_rmsd = float(ca_loss(coords, coords))
    print("\n--- Baseline (NMR model 1, pre-refinement) ---")
    print(f"  Cα RMSD: {init_ca_rmsd:.3f} ppm")

    # ------------------------------------------------------------------
    # 7. Optimize (Cartesian: kinematics_fn = None → identity)
    # ------------------------------------------------------------------
    print(f"\nRefining for {EPOCHS} epochs  (lr={LEARNING_RATE}, "
          f"anchor τ={WEIGHT_ANCHOR_DECAY})...")

    result = refiner.run(
        init_params=coords,           # raw PDB backbone coordinates
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        kinematics_fn=None,           # identity: params ARE the coords
        weight_schedules={0: anchor_schedule},
        log_interval=50,
    )

    # ------------------------------------------------------------------
    # 8. Per-checkpoint geometry monitoring
    # ------------------------------------------------------------------
    print("\n  Epoch | Cα RMSD | Bond RMSD | Angle RMSD | Anchor wt")
    print("  " + "-" * 58)

    final_c = result.best_params
    for ep_frac, loss_frac in [(0.2, 0.2), (0.4, 0.4), (0.6, 0.6), (0.8, 0.8), (1.0, 1.0)]:
        # We don't have per-epoch coord snapshots; evaluate at final only.
        # For live monitoring the benchmark loop prints at 100-epoch intervals.
        pass

    # Evaluate at final / best checkpoint
    best_c  = result.best_params
    final_c_raw = result.final_params

    def _report(label: str, c: np.ndarray) -> None:
        import jax.numpy as jnp
        c_jax = jnp.array(c)
        ca_rmsd    = float(ca_loss(c_jax, c_jax))
        bond_rmsd  = bond_pen.bond_rmsd(c_jax)
        angle_rmsd = angle_pen.angle_rmsd_deg(c_jax)
        struct_rmsd = kabsch_rmsd(coords, c)
        print(f"  {label}")
        print(f"    Cα RMSD:         {ca_rmsd:.3f} ppm  (Δ = {ca_rmsd - init_ca_rmsd:+.3f})")
        print(f"    Bond RMSD:       {bond_rmsd:.4f} Å  (target < 0.05 Å)")
        print(f"    Angle RMSD:      {angle_rmsd:.3f}°  (target < 3°)")
        print(f"    Structural RMSD: {struct_rmsd:.3f} Å vs raw PDB model 1")

    print("\n--- Results ---")
    _report("Best checkpoint (lowest training loss):", best_c)
    if result.best_epoch != result.epochs_run - 1:
        _report("Final epoch:", final_c_raw)

    # ------------------------------------------------------------------
    # 9. Save artefacts
    # ------------------------------------------------------------------
    results_dir = Path("benchmarks/results/HR2876B_cartesian")
    results_dir.mkdir(parents=True, exist_ok=True)

    np.save(results_dir / "loss_history.npy", np.array(result.loss_history))

    import biotite.structure.io.pdb as pdb  # noqa: PLC0415

    backbone_mask = np.isin(struct.atom_name, ["N", "CA", "C"])
    init_struct = struct[backbone_mask].copy()

    f_init = pdb.PDBFile()
    f_init.set_structure(init_struct)
    f_init.write(str(results_dir / "initial.pdb"))

    # Save best-checkpoint structure
    best_struct = init_struct.copy()
    best_struct.coord = np.array(best_c)
    f_best = pdb.PDBFile()
    f_best.set_structure(best_struct)
    f_best.write(str(results_dir / "final.pdb"))

    print(f"\n  Results saved to {results_dir}/")
    print(f"  Best checkpoint at epoch {result.best_epoch} / {result.epochs_run}")


if __name__ == "__main__":
    main()
