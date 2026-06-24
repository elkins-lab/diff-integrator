"""
Benchmark: Cα chemical shift refinement of GmR58A (2KUT) using diff-integrator.

Protein
-------
GmR58A from *Geobacter metallireducens*, NESG target GmR58A.
PDB: 2KUT (10-model NMR ensemble) | BMRB: 16746

Data source: BMRB 16746 (fully public).  All observables (shifts + 3 RDC media)
are available from a single NMR-STAR v3 file.

This is a stronger scientific validation dataset than 2KZV because the BMRB entry
contains three independent alignment media (stretched gel, negative gel, PEG),
which dramatically over-determines the Saupe tensor and makes the benchmark
much harder to overfit than a single-medium measurement.

This benchmark uses only the Cα chemical shifts (no RDCs).  See the diff-biophys
GmR58A benchmark for a full RDC refinement example.

Design choices
--------------
1. **Internal coordinates**: Optimization over backbone (φ, ψ) dihedral angles.
   A NeRF-based builder reconstructs Cartesian coordinates at each step.

2. **GeometryLoss anchor**: Harmonic restraint to the initial NMR coordinates
   prevents the optimizer from freely distorting the global fold while
   minimizing the Cα shift prediction error.
"""

import sys
from pathlib import Path

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

# ---------------------------------------------------------------------------
# Data location
# ---------------------------------------------------------------------------

BENCH_DIR = Path("../diff-biophys/benchmarks/GmR58A").resolve()
sys.path.insert(0, str(BENCH_DIR))
from parse_nmrstar import load_bmrb_shifts  # noqa: E402 (local benchmark utility)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

EPOCHS = 500
LEARNING_RATE = 0.01
WEIGHT_GEOMETRY = 1.0
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
    print("diff-integrator Benchmark: GmR58A (2KUT)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load structure
    # ------------------------------------------------------------------
    struct = load_pdb_model(BENCH_DIR / "2KUT.pdb", model_id=1)
    res_ids, res_names = get_residue_info(struct)
    coords = get_backbone_coords(struct)
    n_res = len(res_ids)

    build_backbone = make_backbone_builder(n_res, coords[:3])
    init_phi, init_psi = compute_phi_psi(coords)
    print(f"\nStructure: {n_res} residues, {len(coords)} backbone atoms")

    # ------------------------------------------------------------------
    # 2. Set up Cα chemical shift loss
    # ------------------------------------------------------------------
    bmrb = load_bmrb_shifts(BENCH_DIR / "bmrb16746_GmR58A.str")
    ca_exp = bmrb["CA"]
    ca_loss = CAShiftLoss(ca_exp["res_id"], ca_exp["shift"], res_ids, list(res_names))
    print(f"Cα shifts: {len(ca_exp['res_id'])} matched residues")

    # ------------------------------------------------------------------
    # 3. Set up geometry anchor
    # ------------------------------------------------------------------
    geom_loss = GeometryLoss(target_coords=coords, target_weight=1.0)

    # ------------------------------------------------------------------
    # 4. Build joint loss and refiner
    # ------------------------------------------------------------------
    joint_loss = JointLoss([
        (geom_loss, WEIGHT_GEOMETRY),
        (ca_loss,   WEIGHT_CA_SHIFTS),
    ])

    refiner = IntegrativeRefiner(loss_fn=joint_loss)

    # ------------------------------------------------------------------
    # 5. Baseline
    # ------------------------------------------------------------------
    init_coords = build_backbone(init_phi, init_psi)
    init_ca_rmsd = float(ca_loss((init_phi, init_psi), init_coords))
    print(f"\n--- Baseline (NMR model 1, pre-refinement) ---")
    print(f"  Cα RMSD: {init_ca_rmsd:.3f} ppm")

    # ------------------------------------------------------------------
    # 6. Optimize
    # ------------------------------------------------------------------
    print(f"\nRefining for {EPOCHS} epochs...")
    final_params, history = refiner.run(
        init_params=(init_phi, init_psi),
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        kinematics_fn=lambda p: build_backbone(p[0], p[1]),
    )

    # ------------------------------------------------------------------
    # 7. Evaluate
    # ------------------------------------------------------------------
    final_phi, final_psi = final_params
    final_coords = build_backbone(final_phi, final_psi)
    final_ca_rmsd = float(ca_loss((final_phi, final_psi), final_coords))
    rmsd = kabsch_rmsd(init_coords, final_coords)

    print("\n--- Final Results ---")
    print(f"  Cα RMSD before: {init_ca_rmsd:.3f} ppm")
    print(f"  Cα RMSD after:  {final_ca_rmsd:.3f} ppm  (Δ = {final_ca_rmsd - init_ca_rmsd:+.3f})")
    print(f"  Structural RMSD: {rmsd:.3f} Å to NMR model 1")

    # ------------------------------------------------------------------
    # 8. Save artefacts
    # ------------------------------------------------------------------
    results_dir = Path("benchmarks/results/GmR58A")
    results_dir.mkdir(parents=True, exist_ok=True)

    np.save(results_dir / "loss_history.npy", np.array(history))

    import biotite.structure.io.pdb as pdb  # noqa: PLC0415

    backbone_mask = np.isin(struct.atom_name, ["N", "CA", "C"])
    init_struct = struct[backbone_mask].copy()
    f_init = pdb.PDBFile()
    f_init.set_structure(init_struct)
    f_init.write(str(results_dir / "initial.pdb"))

    final_struct = init_struct.copy()
    final_struct.coord = np.array(final_coords)
    f_final = pdb.PDBFile()
    f_final.set_structure(final_struct)
    f_final.write(str(results_dir / "final.pdb"))

    print(f"\n  Results saved to {results_dir}/")


if __name__ == "__main__":
    main()
