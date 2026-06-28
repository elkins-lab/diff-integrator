"""
Benchmark: Joint NMR refinement of GmR58A (2KUT) using diff-integrator.

Protein
-------
GmR58A from *Geobacter metallireducens*, NESG target GmR58A.
PDB: 2KUT (10-model NMR ensemble) | BMRB: 16746

Data source: BMRB 16746 (fully public).  All observables (shifts + 3 RDC media)
are available from a single NMR-STAR v3 file.

This benchmark uses all three sources of experimental data simultaneously:
  * 114 Cα chemical shifts
  * 43  ¹⁵N–¹H RDCs in gel (RDC_list_1)       — ratio 8.6×  ✅ reliable
  * 59  ¹⁵N–¹H RDCs in negative gel (RDC_list_2) — ratio 11.8× ✅ reliable
  * 53  ¹⁵N–¹H RDCs in PEG (RDC_list_3)         — ratio 10.6× ✅ reliable

GmR58A is a stronger scientific validation dataset than 2KZV because all three
media are strongly overdetermined (>8× the 5 Saupe tensor parameters), making
the dataset essentially immune to the tensor-degeneracy overfitting problem.

Annealed geometry weight
------------------------
The geometry restraint starts strong (weight=10.0) and decays exponentially
toward 0.1 over 300 epochs via ``ExponentialDecaySchedule``, passed to
``refiner.run()`` as ``weight_schedules={0: geom_schedule}``.  This prevents
structural unravelling in the early epochs when the RDC tensors are poorly
estimated, then relaxes the anchor so experimental gradients can dominate
as the tensors stabilise.

Design choices
--------------
1. **Internal coordinates**: Optimization over backbone (φ, ψ) dihedral angles.
   A NeRF-based builder reconstructs Cartesian coordinates at each step.

2. **Fixed-tensor RDC losses** (``FixedTensorRDCLoss``):
   The Saupe alignment tensor for each medium is held fixed during gradient
   descent and re-fitted every ``TENSOR_UPDATE_INTERVAL`` epochs via a
   ``per_epoch_callbacks`` entry in ``refiner.run()``.

3. **Annealed GeometryLoss anchor**:
   Uses ``ExponentialDecaySchedule`` via ``weight_schedules={0: geom_schedule}``
   in ``refiner.run()``.  No manual weight-passing trick to the JIT step
   function is needed — ``IntegrativeRefiner`` handles this internally by
   passing the current weight vector as a dynamic JAX array to the compiled
   objective, so weight changes take effect every epoch without recompilation.

4. **Best-checkpoint by mean Q-factor**:
   ``RefinementResult.best_params`` tracks the lowest total training loss.
   GmR58A uses mean Q-factor across all three media as the primary checkpoint
   criterion (lower mean Q = structurally better for this multi-media dataset).
   A ``per_epoch_callbacks`` entry accumulates the best-Q checkpoint every
   10 epochs, which is stored in ``best_q_params`` and reported separately
   alongside ``result.best_params`` (lowest loss).
"""

import sys
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from diff_biophys.geometry.backbone import (
    compute_phi_psi,
    get_backbone_coords,
    get_residue_info,
    load_pdb_model,
    make_backbone_builder,
)
from diff_biophys.nmr.rdc import make_rdc_refinement_fns

from diff_integrator.loss import JointLoss
from diff_integrator.optimizer import IntegrativeRefiner
from diff_integrator.schedules import ExponentialDecaySchedule
from diff_integrator.terms.chemical_shifts import CAShiftLoss
from diff_integrator.terms.geometry import GeometryLoss
from diff_integrator.terms.nmr import FixedTensorRDCLoss

# ---------------------------------------------------------------------------
# Data location
# ---------------------------------------------------------------------------

BENCH_DIR = Path("../diff-biophys/benchmarks/GmR58A").resolve()
sys.path.insert(0, str(BENCH_DIR))
from parse_nmrstar import load_bmrb_rdcs, load_bmrb_shifts  # noqa: E402

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

EPOCHS = 500
LEARNING_RATE = 0.01
TENSOR_UPDATE_INTERVAL = 50  # Re-fit Saupe tensor every N epochs
LOG_INTERVAL = 100           # Print diagnostics every N epochs
BEST_Q_CHECK_INTERVAL = 10   # Check best mean-Q every N epochs

# Loss weights
WEIGHT_GEOMETRY_INITIAL = 10.0   # Strong anchor early in training
WEIGHT_GEOMETRY_FINAL   = 0.1    # Relaxed anchor at convergence
WEIGHT_GEOMETRY_DECAY   = 100    # τ = 100 epochs
WEIGHT_RAMACHANDRAN = 0.5        # Sequence-aware Ramachandran prior
WEIGHT_CA_SHIFTS = 1.0
WEIGHT_RDC_GEL     = 1.0
WEIGHT_RDC_NEG_GEL = 1.0
WEIGHT_RDC_PEG     = 1.0


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
    print("diff-integrator Benchmark: GmR58A (2KUT) — Full RDC")
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
    # 2. Load Cα chemical shifts
    # ------------------------------------------------------------------
    bmrb = load_bmrb_shifts(BENCH_DIR / "bmrb16746_GmR58A.str")
    ca_exp = bmrb["CA"]
    ca_loss = CAShiftLoss(ca_exp["res_id"], ca_exp["shift"], res_ids, list(res_names))
    print(f"Cα shifts: {len(ca_exp['res_id'])} matched residues")

    # ------------------------------------------------------------------
    # 3. Load and set up all three RDC media
    # ------------------------------------------------------------------
    rdcs = load_bmrb_rdcs(BENCH_DIR / "bmrb16746_GmR58A.str")
    rdc_list_names = sorted(rdcs.keys())  # ["RDC_list_1", "RDC_list_2", "RDC_list_3"]

    rdc_terms:  list[FixedTensorRDCLoss] = []
    q_eval_fns: list[Any] = []
    rdc_labels: list[str] = []
    for name in rdc_list_names:
        d = rdcs[name]
        loss_fn, q_fn, tensor_fn, n_matched = make_rdc_refinement_fns(
            d["res_id"], d["rdc"], res_ids
        )
        term = FixedTensorRDCLoss(loss_fn, tensor_fn, update_interval=TENSOR_UPDATE_INTERVAL)
        rdc_terms.append(term)
        q_eval_fns.append(q_fn)
        rdc_labels.append(name)
        print(f"RDC {name}: {n_matched} matched residues  "
              f"(ratio {n_matched / 5:.1f}× tensor params)")

    # ------------------------------------------------------------------
    # 4. Build joint loss: geometry(0) + rama(1) + shifts(2) + 3 RDC terms(3,4,5)
    # ------------------------------------------------------------------
    from diff_integrator.terms.ramachandran import RamachandranLoss  # noqa: PLC0415

    geom_loss = GeometryLoss(target_coords=coords, target_weight=1.0)
    rama_loss = RamachandranLoss(residue_types=list(res_names))
    rdc_weights = [WEIGHT_RDC_GEL, WEIGHT_RDC_NEG_GEL, WEIGHT_RDC_PEG]
    joint_loss = JointLoss(
        [(geom_loss, WEIGHT_GEOMETRY_INITIAL)]               # index 0 — annealed
        + [(rama_loss, WEIGHT_RAMACHANDRAN)]                  # index 1 — fixed
        + [(ca_loss, WEIGHT_CA_SHIFTS)]                      # index 2
        + [(t, w) for t, w in zip(rdc_terms, rdc_weights, strict=False)]  # indices 3,4,5
    )

    # Annealed geometry weight: strong anchor → relaxed, via weight_schedules.
    geom_schedule = ExponentialDecaySchedule(
        initial_weight=WEIGHT_GEOMETRY_INITIAL,
        final_weight=WEIGHT_GEOMETRY_FINAL,
        decay_epochs=WEIGHT_GEOMETRY_DECAY,
    )

    refiner = IntegrativeRefiner(loss_fn=joint_loss)

    # ------------------------------------------------------------------
    # 5. Baseline
    # ------------------------------------------------------------------
    init_coords = build_backbone(init_phi, init_psi)
    for term in rdc_terms:
        term.initialize_tensor(init_coords)

    print("\n--- Baseline (NMR model 1, pre-refinement) ---")
    print(f"  Cα RMSD: {float(ca_loss((init_phi, init_psi), init_coords)):.3f} ppm")
    for label, q_fn in zip(rdc_labels, q_eval_fns, strict=False):
        print(f"  Q ({label}): {float(q_fn(init_coords)):.3f}")

    # ------------------------------------------------------------------
    # 6. Per-epoch callback: tensor updates + Q logging + best-Q checkpoint
    #
    # ``IntegrativeRefiner.run()`` tracks best_params by total training loss.
    # For GmR58A, mean Q-factor across all three media is the scientifically
    # correct checkpoint criterion.  We maintain a separate best-Q tracker in
    # the callback closure and expose it via ``best_q_params`` / ``best_q_epoch``.
    # ------------------------------------------------------------------
    print(f"\nRefining for {EPOCHS} epochs "
          f"(tensor update every {TENSOR_UPDATE_INTERVAL} steps, "
          f"geometry weight annealed {WEIGHT_GEOMETRY_INITIAL}→{WEIGHT_GEOMETRY_FINAL})...")

    # Mutable best-Q state tracked across callback invocations
    _best_q_state: dict[str, Any] = {
        "mean_q":  float("inf"),
        "params":  (init_phi, init_psi),
        "epoch":   0,
    }

    def epoch_callback(epoch: int, params: Any, coords_arg: jnp.ndarray) -> None:
        # Tensor updates outside the gradient tape
        for term in rdc_terms:
            term.maybe_update_tensor(coords_arg, epoch)

        # Best-Q checkpoint (every BEST_Q_CHECK_INTERVAL epochs)
        if (epoch + 1) % BEST_Q_CHECK_INTERVAL == 0:
            mean_q = sum(float(q_fn(coords_arg)) for q_fn in q_eval_fns) / len(q_eval_fns)
            if mean_q < _best_q_state["mean_q"]:
                _best_q_state["mean_q"]  = mean_q
                _best_q_state["params"]  = params
                _best_q_state["epoch"]   = epoch + 1

        # Q-factor and geometry-weight logging
        if (epoch + 1) % LOG_INTERVAL == 0:
            q_strs = "  ".join(
                f"Q({lbl.replace('RDC_list_', 'L')})={float(q_fn(coords_arg)):.3f}"
                for lbl, q_fn in zip(rdc_labels, q_eval_fns, strict=False)
            )
            print(f"  Epoch {epoch + 1:4d}: {q_strs}")

    # ------------------------------------------------------------------
    # 7. Refinement — single refiner.run() call
    # ------------------------------------------------------------------
    result = refiner.run(
        init_params=(init_phi, init_psi),
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        kinematics_fn=lambda p: build_backbone(p[0], p[1]),
        weight_schedules={0: geom_schedule},
        per_epoch_callbacks=[epoch_callback],
    )

    # ------------------------------------------------------------------
    # 8. Final evaluation — report best-Q checkpoint AND last iterate
    # ------------------------------------------------------------------
    final_phi, final_psi = result.final_params
    final_coords = build_backbone(final_phi, final_psi)

    best_phi, best_psi = _best_q_state["params"]
    best_coords = build_backbone(best_phi, best_psi)
    best_epoch  = _best_q_state["epoch"]

    init_ca_rmsd  = float(ca_loss((init_phi,  init_psi),  init_coords))
    final_ca_rmsd = float(ca_loss((final_phi, final_psi), final_coords))
    best_ca_rmsd  = float(ca_loss((best_phi,  best_psi),  best_coords))
    rmsd_final    = kabsch_rmsd(init_coords, final_coords)
    rmsd_best     = kabsch_rmsd(init_coords, best_coords)

    print(f"\n--- Best Checkpoint by mean Q-factor (epoch {best_epoch}) ---")
    print(f"  Cα RMSD: {best_ca_rmsd:.3f} ppm  (baseline: {init_ca_rmsd:.3f})")
    for label, q_fn in zip(rdc_labels, q_eval_fns, strict=False):
        b = float(q_fn(init_coords))
        a = float(q_fn(best_coords))
        print(f"  Q ({label}): {b:.3f} → {a:.3f}  (Δ {a - b:+.3f})")
    print(f"  Structural RMSD: {rmsd_best:.3f} Å to NMR model 1")

    print(f"\n--- Final Iterate (epoch {result.epochs_run}) ---")
    print(f"  Cα RMSD: {final_ca_rmsd:.3f} ppm")
    for label, q_fn in zip(rdc_labels, q_eval_fns, strict=False):
        print(f"  Q ({label}): {float(q_fn(final_coords)):.3f}")
    print(f"  Structural RMSD: {rmsd_final:.3f} Å to NMR model 1")

    # ------------------------------------------------------------------
    # 9. Save artefacts  (best-Q checkpoint is the primary output)
    # ------------------------------------------------------------------
    results_dir = Path("benchmarks/results/GmR58A")
    results_dir.mkdir(parents=True, exist_ok=True)

    np.save(results_dir / "loss_history.npy", np.array(result.loss_history))
    np.save(results_dir / "geometry_weight_history.npy",
            np.array(list(result.weight_history.get(0, []))))
    for label, q_fn in zip(rdc_labels, q_eval_fns, strict=False):
        tag = label.replace("RDC_list_", "rdc_list")
        np.save(results_dir / f"q_{tag}_after.npy",
                np.array([float(q_fn(best_coords))]))
        np.save(results_dir / f"q_{tag}_before.npy",
                np.array([float(q_fn(init_coords))]))

    import biotite.structure.io.pdb as pdb  # noqa: PLC0415

    backbone_mask = np.isin(struct.atom_name, ["N", "CA", "C"])
    init_struct = struct[backbone_mask].copy()
    f_init = pdb.PDBFile()
    f_init.set_structure(init_struct)
    f_init.write(str(results_dir / "initial.pdb"))

    best_struct = init_struct.copy()
    best_struct.coord = np.array(best_coords)
    f_best = pdb.PDBFile()
    f_best.set_structure(best_struct)
    f_best.write(str(results_dir / "final.pdb"))  # best-Q = canonical output

    print(f"\n  Results saved to {results_dir}/  "
          f"(best-Q checkpoint at epoch {best_epoch})")


if __name__ == "__main__":
    main()
