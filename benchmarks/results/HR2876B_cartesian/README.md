# HR2876B Cartesian Refinement Benchmark

**Protein**: N-terminal domain of human NFU1 (Iron-Sulfur Cluster Scaffold Homolog)
**NESG Target**: HR2876B
**PDB**: [2LTM](https://www.rcsb.org/structure/2LTM) | **BMRB**: [18489](https://bmrb.io/data_library/summary/index.php?bmrbId=18489)

**Demonstrates**: Cartesian + bond-geometry + chirality-guard parameterisation, eliminating NeRF geometric drift and Cα L→D inversion.

---

## Purpose

HR2876B is the natural demonstration target for the Cartesian parameterisation because its 107-residue chain produces approximately **14 Å** of NeRF drift — the discrepancy between a NeRF-reconstructed backbone (using ideal Engh & Huber geometry) and the raw PDB model 1 coordinates.  This means the NeRF-based optimizer starts from a qualitatively different structure than the real NMR model, severely limiting its ability to improve against chemical shift data.

The Cartesian approach eliminates this problem entirely by optimising directly in Cartesian coordinate space, using the raw PDB coordinates as the starting point.

---

## Files

| File | Description |
|---|---|
| `initial.pdb` | Raw NMR model 1 backbone (N, CA, C; from RCSB 2LTM) |
| `final.pdb` | Best-checkpoint backbone after 2000 epochs of Cartesian refinement (Sprint 2 run) |
| `loss_history.npy` | Per-epoch total weighted loss trace |
| `chirality_violations_before.npy` | Cα D-inversions in raw PDB (5) |
| `chirality_violations_after.npy` | Cα D-inversions after refinement (0) |

---

## Benchmark Design

### Parameterisation

**Parameters**: `(3N, 3)` Cartesian coordinates — the backbone N, CA, C atom positions of all 107 residues (321 atoms, 963 scalar parameters).

**Kinematics**: identity — `IntegrativeRefiner` is called with `kinematics_fn=None`, meaning `coords = params` directly.  No NeRF reconstruction.

### Loss terms

| Term | Index | Weight | Role |
|---|---|---|---|
| `GeometryLoss(target_coords=pdb_coords)` | 0 | 10.0 → 0.1 (annealed) | Position anchor; prevents rigid-body drift |
| `CartesianCAShiftLoss` | 1 | 1.0 | Cα chemical shift RMSD (ppm) — EarlyStopping monitors this |
| `BondLengthPenalty` | 2 | 50.0 | Harmonic restraint on 320 backbone bonds |
| `BondAnglePenalty` | 3 | 10.0 | Harmonic restraint on 319 backbone angles |
| `FixedTensorRDCLoss` (PEG) | 4 | auto | ¹⁵N–¹H RDC list 1, PEG alignment medium (72 RDCs, 14.4×) |
| `FixedTensorRDCLoss` (Pf1) | 5 | auto | ¹⁵N–¹H RDC list 2, Pf1 phage medium (75 RDCs, 15.0×) |
| `ChiralityPenalty` | 6 | 20.0 | Half-harmonic Cα L→D inversion guard (Sprint 2) |

### Annealed position anchor

The geometry anchor decays exponentially from weight 10.0 → 0.1 over τ = 100 epochs.  This prevents rigid-body drift and out-of-basin exploration early in training, then relaxes so the experimental gradient can dominate from epoch ~300 onward.

### CartesianCAShiftLoss

A `LossTerm` in `diff_integrator/terms/chemical_shifts.py` that:
1. Extracts φ/ψ torsion angles from the current Cartesian coordinates via `compute_phi_psi(coords)` (differentiable using JAX)
2. Evaluates the SPARTA+ empirical chemical shift RMSD against BMRB 18489

The gradient flows `shift_loss → φ/ψ(coords) → coords` through the torsion extraction — no discontinuities.

---

## Results (Sprint 2 — with ChiralityPenalty)

Optimization: 2000 epochs, Adam optimizer (lr=0.005), annealed geometry anchor
(10.0 → 0.1, τ=100), bond weight=50, angle weight=10, chirality weight=20.

### Comparison with NeRF benchmark (`benchmark_HR2876B.py`)

| Metric | NeRF (internal coords) | **Cartesian (this benchmark)** | Improvement |
|---|---|---|---|
| Starting coords | NeRF-rebuilt (14 Å drift) | **Raw PDB model 1** | — |
| Cα RMSD baseline | 1.710 ppm | 1.710 ppm | — |
| Cα RMSD final | 1.699 ppm | **1.587 ppm** | **11× larger improvement** |
| Δ Cα RMSD | −0.011 ppm | **−0.123 ppm** | 11× |
| Structural drift | 6.356 Å | **0.545 Å** | 12× less drift |
| Bond geometry | Exactly ideal (hard) | **0.0056 Å RMSD** ✅ | Maintained |
| Angle geometry | Exactly ideal (hard) | **2.611°** ✅ | Maintained |
| Chirality violations | N/A | **0** (was 5 in raw PDB) ✅ | Fully corrected |

### RDC Q-factors

| Medium | Q before | Q after | Improvement |
|---|---|---|---|
| PEG (list 1) | 0.440 | **0.163** | −0.277 (63% reduction) |
| Pf1 (list 2) | 0.443 | **0.162** | −0.281 (63% reduction) |

### Geometry quality

The bond-length, bond-angle, and chirality penalties maintain physically valid geometry throughout optimization:

- **Bond RMSD**: 0.0056 Å (target < 0.05 Å) ✅
- **Angle RMSD**: 2.611° (target < 3°) ✅
- **Cα D-inversions**: 0 (was 5 in raw PDB, was 6 after the pre-Sprint-2 run) ✅

> [!NOTE]
> The starting raw PDB model 1 has small deviations from ideal Engh & Huber geometry
> (bond RMSD = 0.0196 Å, angle RMSD = 1.623°) because real NMR structures are refined
> with different restraint sets.  The angle RMSD of 2.611° is slightly wider than the
> pre-Sprint-2 run (0.458°) because correcting 5 chirality-inverted Cα centers requires
> genuine backbone rearrangement that stresses adjacent bond angles — still well within
> the 3° acceptance threshold.

### Interpretation

The 11× improvement in Cα RMSD reduction (−0.123 ppm vs. −0.011 ppm) comes almost entirely from eliminating the NeRF drift problem:

- **NeRF**: starts 14 Å from the NMR model, so most of the budget is consumed fighting structural inconsistency rather than improving against experimental data
- **Cartesian**: starts exactly at NMR model 1, so every gradient step immediately exploits the chemical shift and RDC information

The 0.545 Å structural RMSD (vs. 0.211 Å in the pre-Sprint-2 run) reflects the chirality correction: the optimizer moved 5 inverted Cα centers back to L-configuration, which requires genuine backbone displacement.  The larger structural change is physically meaningful, not drift.

The **63% RDC Q-factor reduction** (0.44 → 0.16 on both media independently) is a major new result — these well-overdetermined datasets strongly constrain backbone orientation and contribute substantially to the chemical-shift improvement.

---

## NMR Observable Modules Used

| Observable | Source |
|---|---|
| Cα chemical shifts (Cartesian) | `diff_integrator.terms.chemical_shifts.CartesianCAShiftLoss` |
| Bond-length penalty | `diff_integrator.terms.bond_geometry.BondLengthPenalty` |
| Bond-angle penalty | `diff_integrator.terms.bond_geometry.BondAnglePenalty` |
| Cα chirality guard | `diff_integrator.terms.chirality.ChiralityPenalty` |
| Factory (ideal geometry) | `diff_integrator.terms.bond_geometry.make_backbone_bond_geometry` |
| Factory (chirality) | `diff_integrator.terms.chirality.make_backbone_chirality` |
| RDC loss | `diff_integrator.terms.nmr.FixedTensorRDCLoss` |
| RDC factory | `diff_integrator.terms.nmr.make_rdc_cv_refinement_fns` |
| Position anchor | `diff_integrator.terms.geometry.GeometryLoss` |
| φ/ψ extraction | `diff_biophys.geometry.backbone.compute_phi_psi` |
| Optimizer | `optax.adam` (via `IntegrativeRefiner`) |
| Annealed weight | `diff_integrator.schedules.ExponentialDecaySchedule` |
