# Algorithmic Improvements & Known Limitations

This document catalogues the known limitations of `diff-integrator`'s current
implementation, the algorithmic approaches used to address them, and open avenues
for future improvement.  Each limitation is linked to the benchmark that
demonstrates it most clearly.

---

## 1. NeRF Geometric Drift

### What it is

`diff-integrator` parameterises backbone conformation using **dihedral angles (φ,
ψ)** and reconstructs Cartesian coordinates via the **NeRF** (Natural Extension
Reference Frame) algorithm.  NeRF builds a chain sequentially, positioning each
residue relative to the previous three atoms using ideal peptide bond lengths and
angles.

The critical limitation is that NeRF uses *idealised* rather than *actual* bond
geometry.  Every deviation of the real PDB structure from ideal geometry compounds
along the chain, producing a starting backbone that may already differ
significantly from the raw PDB coordinates.  For **HR2876B (107 residues)** this
drift is approximately **14 Å** (Cα RMSD against raw PDB model 1).

### Impact on benchmarks

| Benchmark | Residues | NeRF drift (NeRF start vs. raw PDB) | Structural RMSD (NeRF start vs. final) |
|---|---|---|---|
| 2KZV | 92 | ~2–3 Å | 1.6 Å |
| GmR58A | 122 | ~3–5 Å | 2.1 Å |
| HR2876B | 107 | ~14 Å | 6.4 Å |

### Proposed solutions (not yet implemented)

1. **Anchored-segment NeRF** — Split the chain into independently-anchored
   segments (every 20–30 residues), bounding drift within each segment.
2. **Cartesian parameterisation + bond/angle penalties** — Optimise directly in
   Cartesian space with harmonic restraints on bond lengths and angles via
   `GeometryLoss`.  The `bonds` and `ideal_bond_lengths` fields already exist in
   [`GeometryLoss`](../diff_integrator/terms/geometry.py); they need only be
   populated with pre-computed ideal values.
3. **Redundant internal coordinates** — Denavit-Hartenberg frames with periodic
   overlap corrections (as used by OpenMM's BAT module).

---

## 2. Geometry Restraint Competing with Experimental Gradients

### What it is

The `GeometryLoss` harmonic anchor (a mean squared distance from the starting
coordinates) is essential for preventing structural unravelling under the
degenerate RDC or chemical-shift potential.  However, if the geometry weight
equals the experimental term weights, the two competing forces cancel and very
little net improvement occurs.

The original GmR58A shift-only benchmark showed only **Δ = −0.001 ppm** improvement
with equal weights (`weight_geometry = weight_shifts = 1.0`).

### Solution: Annealed Geometry Weight

The key insight is that a **strong geometry anchor is most needed in the early
epochs**, when RDC tensors are poorly estimated and chemical-shift gradients
point in unreliable directions.  As the optimization matures and tensors
stabilise, the anchor can be progressively relaxed to let experimental
gradients dominate.

`diff-integrator` implements this via two new components:

#### `ExponentialDecaySchedule` (`diff_integrator/schedules.py`)

Computes a weight that decreases from `initial_weight` toward `final_weight`
with time constant `decay_epochs`:

$$w(t) = w_f + (w_i - w_f)\,e^{-t/\tau}$$

```python
from diff_integrator.schedules import ExponentialDecaySchedule

schedule = ExponentialDecaySchedule(
    initial_weight=10.0,   # Strong anchor at epoch 0
    final_weight=0.1,      # Relaxed anchor at convergence
    decay_epochs=300,      # Time constant in epochs
)
```

#### `weight_schedules` in `IntegrativeRefiner.run()`

Any `LossTerm` in the `JointLoss` can be assigned a schedule by term index:

```python
result = refiner.run(
    init_params=...,
    epochs=500,
    weight_schedules={0: schedule},  # Index 0 = GeometryLoss
)

# Inspect the weight history
print(result.weight_history[0])  # List of weight values per epoch
```

The weight applied at each epoch is recorded in `RefinementResult.weight_history`
for full reproducibility and visualisation.

#### Recommendation for new benchmarks

| Phase | Geometry weight | Rationale |
|---|---|---|
| Epochs 0–100 | ~10 | Prevent early unravelling while tensors are poor |
| Epochs 100–300 | Decaying | Allow experimental gradients increasing influence |
| Epochs 300+ | ~0.1 | Experimental gradients dominate |

---

## 3. RDC Overfitting for Underdetermined Media

### What it is

The Saupe alignment tensor has **5 free parameters**.  When the number of
measured RDCs is less than ~3× the tensor parameters, the optimizer can find
distortions of the backbone that satisfy the RDCs without genuine structural
improvement — **overfitting via tensor degeneracy**.

The 2KZV benchmark exhibits this: the PEG medium has only 16 RDCs (ratio 3.2×),
and the final Q(PEG) = 0.047 is far below the published NMR medoid (0.36),
confirming the optimizer is exploiting tensor degeneracy rather than improving the
structure.

### Current mitigation

`FixedTensorRDCLoss` prevents the optimizer from differentiating *through* the
tensor fit (via `jax.lax.stop_gradient`), so the tensor itself cannot be driven
to degenerate values.  However, small backbone distortions can still satisfy
under-determined RDC constraints.

### Best practice: minimum overdetermination ratio

| Ratio (n_RDCs / 5) | Reliability | Recommendation |
|---|---|---|
| < 3× | Poor | Do not use as a refinement term |
| 3–5× | Marginal | Use with caution; monitor for overfitting |
| > 8× | Reliable | Use freely |

The **GmR58A full-RDC benchmark** demonstrates the reliable regime: all three
media have ratios of 8.6×, 11.8×, and 10.6×.

### Proposed future solutions (not yet implemented)

1. **RDC cross-validation** — Hold back a fraction of measurements as a
   validation set; monitor validation Q independently from training Q.
2. **Regularised tensor fitting** — Tikhonov penalty on tensor magnitude during
   `fit_saupe_tensor` (requires change in `diff_biophys`).
3. **Auto-weight by ratio** — Scale RDC term weight proportionally to the
   overdetermination ratio, automatically down-weighting marginal media.

---

## 4. Sequence-Independent Ramachandran Prior

### What it is

[`RamachandranLoss`](../diff_integrator/terms/ramachandran.py) penalises φ/ψ
pairs that lie far from three canonical basins (α-helix, β-strand, left-handed
α).  The basins are **the same for all residues**, ignoring the well-documented
residue-type-specific distributions:

- **Glycine**: No Cβ, so essentially the entire Ramachandran plot is accessible.
- **Proline**: Restricted to cis/trans ω; φ is confined to ~−75° ± 25°.
- **Pre-proline**: Shifted β-strand basin.
- **Alanine, Val, Ile**: Narrow α-helix and β-strand basins.

### Proposed solution

A `residue_types` parameter accepting a sequence of amino acid codes, with
per-residue basin centres and widths from a lookup table.  The MolProbity
Ramachandran distributions are freely available and have been encoded as
a static dictionary in `diff_integrator/terms/ramachandran.py`.

### Implementation (diff-integrator ≥ 0.1.1)

```python
from diff_integrator.terms.ramachandran import RamachandranLoss

# Sequence-aware mode: per-residue basins from the MolProbity lookup table
loss = RamachandranLoss(residue_types=["ALA", "GLY", "PRO", "VAL", ...])
```

Key improvements:

| Residue | Basins | σ | Rationale |
|---|---|---|---|
| Default (all others) | α, β, L-α (3) | 0.5 | Standard Ramachandran |
| **Gly** | α, β, L-α, ε/mirror-β (4) | **1.0** | No Cβ → broader & 4th basin |
| **Pro** | down pucker, up pucker (2) | **0.35** | Ring-constrained φ ≈ −65° |

The ε-basin for Gly (φ ≈ +60°, ψ ≈ −120°) is unique to glycine and was
previously penalised as "forbidden" — causing the optimizer to push Gly residues
into sterically inappropriate conformations. The narrow Pro basins prevent the
optimizer from exploring physically inaccessible φ values.

---

## 5. Future Directions

| Direction | Benefit | Complexity |
|---|---|---|
| Add GmR58A RDCs (3 media, 155 total) | Dramatic Q improvement; benchmarks reliable regime | Low (data available, tooling ready) ✅ **Implemented** |
| Annealed geometry weight (τ=100) | Schedule completes within 500-epoch budget; Q(gel) −82%, Q(PEG) −84% | Low ✅ **Implemented** |
| Sequence-aware Ramachandran prior | Physically accurate backbone prior (GLY ε-basin, PRO ring constraint) | Low ✅ **Implemented** |
| Best-checkpoint saving | Returns the iterate with best loss/Q rather than the last one | Low ✅ **Implemented** |
| Anchored-segment NeRF for long chains | Eliminates HR2876B drift problem | Medium |
| Cartesian + bond-angle penalty | Eliminates NeRF drift entirely | Medium |
| RDC cross-validation split | Guard against overfitting in low-ratio media | Low |
| Auto-weight RDC by ratio | Prevent accidental use of under-determined media | Low |
| Riemannian Adam for torsion angles | Respects periodic (−π, π) topology | Medium |
| Per-term early stopping | Stop on experimental observable plateau, not total loss | Low |
