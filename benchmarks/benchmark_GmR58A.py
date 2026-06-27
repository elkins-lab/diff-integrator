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
toward 0.1 over 300 epochs via ``ExponentialDecaySchedule``.  This prevents
structural unravelling in the early epochs when the RDC tensors are poorly
estimated, then relaxes the anchor so experimental gradients can dominate
as the tensors stabilise.

Design choices
--------------
1. **Internal coordinates**: Optimization over backbone (φ, ψ) dihedral angles.
   A NeRF-based builder reconstructs Cartesian coordinates at each step.

2. **Fixed-tensor RDC losses** (``FixedTensorRDCLoss``):
   The Saupe alignment tensor for each medium is held fixed during gradient
   descent and re-fitted every ``TENSOR_UPDATE_INTERVAL`` epochs.

3. **Annealed GeometryLoss anchor**:
   Uses ``ExponentialDecaySchedule`` via the ``weight_schedules`` mechanism
   in ``IntegrativeRefiner``.
"""

import sys
from pathlib import Path

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

# Loss weights
WEIGHT_GEOMETRY_INITIAL = 10.0   # Strong anchor early in training
WEIGHT_GEOMETRY_FINAL = 0.1     # Relaxed anchor in later epochs
WEIGHT_GEOMETRY_DECAY = 300     # Exponential decay time-constant (epochs)
WEIGHT_CA_SHIFTS = 1.0
WEIGHT_RDC_GEL = 1.0
WEIGHT_RDC_NEG_GEL = 1.0
WEIGHT_RDC_PEG = 1.0


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
    rdc_list_names = sorted(rdcs.keys())  # e.g. ["RDC_list_1", "RDC_list_2", "RDC_list_3"]

    rdc_terms: list[FixedTensorRDCLoss] = []
    q_eval_fns = []
    rdc_labels = []
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
    # 4. Build joint loss: geometry(0) + shifts(1) + 3 RDC terms(2,3,4)
    # ------------------------------------------------------------------
    geom_loss = GeometryLoss(target_coords=coords, target_weight=1.0)
    rdc_weights = [WEIGHT_RDC_GEL, WEIGHT_RDC_NEG_GEL, WEIGHT_RDC_PEG]
    joint_loss = JointLoss(
        [(geom_loss, WEIGHT_GEOMETRY_INITIAL)]        # index 0 — annealed
        + [(ca_loss, WEIGHT_CA_SHIFTS)]               # index 1
        + [(t, w) for t, w in zip(rdc_terms, rdc_weights)]  # indices 2,3,4
    )

    # Annealed geometry weight: strong anchor → relaxed
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
    for label, q_fn in zip(rdc_labels, q_eval_fns):
        print(f"  Q ({label}): {float(q_fn(init_coords)):.3f}")

    # ------------------------------------------------------------------
    # 6. Custom training loop with periodic tensor updates
    # ------------------------------------------------------------------
    print(f"\nRefining for {EPOCHS} epochs "
          f"(tensor update every {TENSOR_UPDATE_INTERVAL} steps, "
          f"geometry weight annealed {WEIGHT_GEOMETRY_INITIAL}→{WEIGHT_GEOMETRY_FINAL})...")

    import jax  # noqa: PLC0415
    import optax  # noqa: PLC0415

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(LEARNING_RATE),
    )
    opt_state = optimizer.init((init_phi, init_psi))
    params = (init_phi, init_psi)
    loss_history: list[float] = []
    weight_history: list[float] = []

    @jax.jit
    def step(p, state):  # type: ignore[no-untyped-def]
        def objective(pp):  # type: ignore[no-untyped-def]
            c = build_backbone(pp[0], pp[1])
            return joint_loss(pp, c)

        loss_val, grads = jax.value_and_grad(objective)(p)
        updates, new_state = optimizer.update(grads, state)
        new_p = optax.apply_updates(p, updates)
        return new_p, new_state, loss_val

    for epoch in range(EPOCHS):
        # 6a. Update geometry weight (annealing)
        new_geom_weight = geom_schedule(epoch)
        joint_loss.set_weight(0, new_geom_weight)
        weight_history.append(new_geom_weight)

        # 6b. Update RDC tensors (outside the gradient)
        curr_coords = build_backbone(params[0], params[1])
        for term in rdc_terms:
            term.maybe_update_tensor(curr_coords, epoch)

        # 6c. Gradient step
        params, opt_state, loss_val = step(params, opt_state)
        loss_history.append(float(loss_val))

        if (epoch + 1) % 100 == 0:
            c = build_backbone(params[0], params[1])
            q_strs = "  ".join(
                f"Q({lbl.replace('RDC_list_', 'L')})={float(q_fn(c)):.3f}"
                for lbl, q_fn in zip(rdc_labels, q_eval_fns)
            )
            print(f"  Epoch {epoch + 1:4d}: loss={loss_val:.4f}  {q_strs}  "
                  f"geom_w={new_geom_weight:.3f}")

    # ------------------------------------------------------------------
    # 7. Final evaluation
    # ------------------------------------------------------------------
    final_phi, final_psi = params
    final_coords = build_backbone(final_phi, final_psi)
    init_ca_rmsd = float(ca_loss((init_phi, init_psi), init_coords))
    final_ca_rmsd = float(ca_loss((final_phi, final_psi), final_coords))
    rmsd = kabsch_rmsd(init_coords, final_coords)

    print("\n--- Final Results ---")
    print(f"  Cα RMSD before: {init_ca_rmsd:.3f} ppm")
    print(f"  Cα RMSD after:  {final_ca_rmsd:.3f} ppm  (Δ = {final_ca_rmsd - init_ca_rmsd:+.3f})")
    for label, q_fn in zip(rdc_labels, q_eval_fns):
        print(f"  Q ({label}): {float(q_fn(final_coords)):.3f}")
    print(f"  Structural RMSD: {rmsd:.3f} Å to NMR model 1")

    # ------------------------------------------------------------------
    # 8. Save artefacts
    # ------------------------------------------------------------------
    results_dir = Path("benchmarks/results/GmR58A")
    results_dir.mkdir(parents=True, exist_ok=True)

    np.save(results_dir / "loss_history.npy", np.array(loss_history))
    np.save(results_dir / "geometry_weight_history.npy", np.array(weight_history))
    for label, q_fn in zip(rdc_labels, q_eval_fns):
        tag = label.replace("RDC_list_", "rdc_list")
        np.save(results_dir / f"q_{tag}_after.npy",
                np.array([float(q_fn(final_coords))]))
        np.save(results_dir / f"q_{tag}_before.npy",
                np.array([float(q_fn(init_coords))]))

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
