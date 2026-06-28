# HR2876B Cartesian Refinement Benchmark

**Protein**: N-terminal domain of human NFU1 (Iron-Sulfur Cluster Scaffold Homolog)
**NESG Target**: HR2876B
**PDB**: [2LTM](https://www.rcsb.org/structure/2LTM) | **BMRB**: [18489](https://bmrb.io/data_library/summary/index.php?bmrbId=18489)

**Demonstrates**: Cartesian + bond-angle penalty parameterisation, eliminating NeRF geometric drift.

---

## Purpose

HR2876B is the natural demonstration target for the Cartesian parameterisation because its 107-residue chain produces approximately **14 Å** of NeRF drift — the discrepancy between a NeRF-reconstructed backbone (using ideal Engh & Huber geometry) and the raw PDB model 1 coordinates.  This means the NeRF-based optimizer starts from a qualitatively different structure than the real NMR model, severely limiting its ability to improve against chemical shift data.

The Cartesian approach eliminates this problem entirely by optimising directly in Cartesian coordinate space, using the raw PDB coordinates as the starting point.

---

## Files

| File | Description |
|---|---|
| `initial.pdb` | Raw NMR model 1 backbone (N, CA, C; from RCSB 2LTM) |
| `final.pdb` | Best-checkpoint backbone after 500 epochs of Cartesian refinement |
| `loss_history.npy` | Per-epoch total weighted loss trace |

---

## Benchmark Design

### Parameterisation

**Parameters**: `(3N, 3)` Cartesian coordinates — the backbone N, CA, C atom positions of all 107 residues (321 atoms, 963 scalar parameters).

**Kinematics**: identity — `IntegrativeRefiner` is called with `kinematics_fn=None`, meaning `coords = params` directly.  No NeRF reconstruction.

### Loss terms

| Term | Weight | Role |
|---|---|---|
| `GeometryLoss(target_coords=pdb_coords)` | 10.0 → 0.1 (annealed) | Position anchor; prevents rigid-body drift |
| `CartesianCAShiftLoss` | 1.0 | Cα chemical shift RMSD (ppm) |
| `BondLengthPenalty` | 50.0 | Harmonic restraint on 320 backbone bonds |
| `BondAnglePenalty` | 10.0 | Harmonic restraint on 319 backbone angles |

### Annealed position anchor

The geometry anchor decays exponentially from weight 10.0 → 0.1 over τ = 100 epochs.  This prevents rigid-body drift and out-of-basin exploration early in training, then relaxes so the experimental gradient can dominate from epoch ~300 onward.

### CartesianCAShiftLoss

A new `LossTerm` in `diff_integrator/terms/chemical_shifts.py` that:
1. Extracts φ/ψ torsion angles from the current Cartesian coordinates via `compute_phi_psi(coords)` (differentiable using JAX)
2. Evaluates the SPARTA+ empirical chemical shift RMSD against BMRB 18489

The gradient flows `shift_loss → φ/ψ(coords) → coords` through the torsion extraction — no discontinuities.

---

## Results

Optimization: 500 epochs, Adam optimizer (lr=0.005), annealed geometry anchor
(10.0 → 0.1, τ=100), bond-length weight=50, angle weight=10.

### Comparison with NeRF benchmark (`benchmark_HR2876B.py`)

| Metric | NeRF (internal coords) | **Cartesian (this benchmark)** | Improvement |
|---|---|---|---|
| Starting coords | NeRF-rebuilt (14 Å drift) | **Raw PDB model 1** | — |
| Cα RMSD baseline | 1.710 ppm | 1.710 ppm | — |
| Cα RMSD final | 1.699 ppm | **1.567 ppm** | **13× larger improvement** |
| Δ Cα RMSD | −0.011 ppm | **−0.144 ppm** | 13× |
| Structural drift | 6.356 Å | **0.211 Å** | 30× less drift |
| Bond geometry | Exactly ideal (hard) | **0.0010 Å RMSD** ✅ | Maintained |
| Angle geometry | Exactly ideal (hard) | **0.458°** ✅ | Maintained |

### Geometry quality

The bond-length and bond-angle penalties successfully maintain physically valid geometry throughout optimization:

- **Bond RMSD**: 0.0010 Å (target < 0.05 Å) ✅
- **Angle RMSD**: 0.458° (target < 3°) ✅

> [!NOTE]
> The starting raw PDB model 1 already has small deviations from ideal Engh & Huber
> geometry (bond RMSD = 0.0196 Å, angle RMSD = 1.62°) because real NMR structures
> are refined with different restraint sets.  During Cartesian optimization, the
> bond/angle penalties pull the coordinates toward ideal geometry — producing a
> **better** bond RMSD (0.0010 Å) than the input PDB.

### Interpretation

The 13× improvement in Cα RMSD reduction (−0.144 ppm vs. −0.011 ppm) comes almost entirely from eliminating the NeRF drift problem:

- **NeRF**: starts 14 Å from the NMR model, so most of the 500-epoch budget is consumed fighting this structural inconsistency rather than improving against the experimental data
- **Cartesian**: starts exactly at NMR model 1, so every gradient step can immediately exploit the chemical shift information

The 0.211 Å structural RMSD confirms that the backbone moved a physically meaningful but small amount — the optimizer found genuine improvements, not artificial coordinate drift.

---

## NMR Observable Modules Used

| Observable | Source |
|---|---|
| Cα chemical shifts (Cartesian) | `diff_integrator.terms.chemical_shifts.CartesianCAShiftLoss` |
| Bond-length penalty | `diff_integrator.terms.bond_geometry.BondLengthPenalty` |
| Bond-angle penalty | `diff_integrator.terms.bond_geometry.BondAnglePenalty` |
| Factory (ideal geometry) | `diff_integrator.terms.bond_geometry.make_backbone_bond_geometry` |
| Position anchor | `diff_integrator.terms.geometry.GeometryLoss` |
| φ/ψ extraction | `diff_biophys.geometry.backbone.compute_phi_psi` |
| Optimizer | `optax.adam` (via `IntegrativeRefiner`) |
| Annealed weight | `diff_integrator.schedules.ExponentialDecaySchedule` |
