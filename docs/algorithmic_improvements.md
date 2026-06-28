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

### Proposed solutions

**1. RDC cross-validation (implemented ✅)**

Hold back a fraction of measurements as a validation set; monitor validation Q
independently from training Q.

```python
from diff_integrator.terms.nmr import make_rdc_cv_refinement_fns, FixedTensorRDCLoss

loss_fn, q_eval_fn, make_tensor_fn, val_q_fn, n_train, n_val = \
    make_rdc_cv_refinement_fns(
        rdc["res_id"], rdc["rdc"], res_ids, cv_fraction=0.2
    )

term = FixedTensorRDCLoss(
    loss_fn=loss_fn, make_tensor_fn=make_tensor_fn,
    n_rdcs=n_train, val_q_eval_fn=val_q_fn,
)
```

`evaluate_validation_q(coords)` returns the Q-factor on held-out measurements
using the current training-fitted tensor.  If training Q drops while validation
Q stays flat or rises, overfitting is occurring.

**2KZV experiment (2025-06)**: With PAG cross-validation (18 train / 5 held-out
RDCs) the training Q improved from 0.309 → 0.298 (−4%) while the validation Q
*worsened* from 0.309 → 0.471 (+52%).  This definitively shows that at ratio
3.6×, the PAG medium is too underdetermined for reliable structural improvement.
Without the CV split this overfitting would have been invisible.

**2. Auto-weight by overdetermination ratio (implemented ✅)**

Scale RDC term weight proportionally to `n_train_rdcs / (5 × 10)`, automatically
down-weighting marginal media and up-weighting well-determined ones.

```python
weight = rdc_term.suggested_weight(base_weight=1.0)
# e.g. PAG (ratio 3.6×): weight = 0.36
#      PEG (ratio 3.2×): weight = 0.32
#      GmR58A L1 (ratio 8.6×): weight = 0.86
```

Result is clamped to `[0.1 × base, 2.0 × base]`.

**3. Regularised tensor fitting** — Tikhonov penalty on tensor magnitude
during `fit_saupe_tensor` (requires change in `diff_biophys`, not yet implemented).

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

## 4. Cartesian + Bond-Angle Penalty (implemented ✅)

Optimise directly in Cartesian coordinate space instead of over backbone (φ, ψ)
dihedral angles.  Bond-length and bond-angle penalties replace the hard NeRF
geometric constraints with soft harmonic restraints.

### The NeRF drift problem

The NeRF backbone builder starts from a 3-atom seed and propagates ideal
bond lengths and angles along the chain.  For long proteins (>50 residues),
float32 accumulation causes the NeRF-rebuilt backbone to diverge from the
raw PDB coordinates.  For HR2876B (107 residues) this drift is approximately
**14 Å**, meaning the optimizer starts from a structure qualitatively
different from the real NMR model.

### Solution: BondLengthPenalty + BondAnglePenalty

New module `diff_integrator/terms/bond_geometry.py` provides:

- `BondLengthPenalty` — harmonic MSE on backbone bond lengths
- `BondAnglePenalty` — harmonic MSE on backbone bond angles (arccos-safe)
- `make_backbone_bond_geometry(n_residues)` — factory pre-populated with
  Engh & Huber (1991) ideal values for all N–CA–C backbone bonds/angles

Also added `CartesianCAShiftLoss` to `diff_integrator/terms/chemical_shifts.py`,
which extracts φ/ψ torsion angles from the current Cartesian coordinates via
`compute_phi_psi(coords)` and evaluates the chemical shift RMSD.  This makes
the gradient `shift_loss → φ/ψ(coords) → coords` fully differentiable.

```python
from diff_integrator.terms.bond_geometry import make_backbone_bond_geometry
from diff_integrator.terms.chemical_shifts import CartesianCAShiftLoss
from diff_integrator.terms.geometry import GeometryLoss

bond_pen, angle_pen = make_backbone_bond_geometry(n_residues)
ca_loss = CartesianCAShiftLoss(exp_res_ids, exp_shifts, res_ids, res_names)
anchor  = GeometryLoss(target_coords=pdb_coords, target_weight=1.0)

joint_loss = JointLoss([
    (anchor,   10.0),   # annealed 10→0.1 over τ=100 epochs
    (ca_loss,  1.0),
    (bond_pen, 50.0),   # stiff bonds
    (angle_pen, 10.0),  # softer angles
])

# Cartesian: params ARE coords; kinematics_fn=None → identity
result = IntegrativeRefiner(joint_loss).run(
    init_params=pdb_backbone_coords,
    kinematics_fn=None,
    weight_schedules={0: anchor_schedule},
)
```

### HR2876B benchmark results (2025-06)

| Metric | NeRF | **Cartesian** | Improvement |
|---|---|---|---|
| Starting coords | NeRF-rebuilt (14 Å drift) | Raw PDB | — |
| Δ Cα RMSD | −0.011 ppm | **−0.144 ppm** | **13×** |
| Structural drift | 6.356 Å | **0.211 Å** | 30× less |
| Bond RMSD (final) | 0 (hard) | **0.0010 Å** ✅ | Maintained |
| Angle RMSD (final) | 0 (hard) | **0.458°** ✅ | Maintained |

The 13× improvement comes entirely from eliminating NeRF drift: the Cartesian
optimizer starts at the actual NMR structure rather than 14 Å away.

---


## 5. Future Directions

| Direction | Benefit | Complexity |
|---|---|---|
| Add GmR58A RDCs (3 media, 155 total) | Dramatic Q improvement; benchmarks reliable regime | Low ✅ **Implemented** |
| Annealed geometry weight (τ=100) | Schedule completes within 500-epoch budget; Q(gel) −82%, Q(PEG) −84% | Low ✅ **Implemented** |
| Sequence-aware Ramachandran prior | Physically accurate backbone prior (GLY ε-basin, PRO ring constraint) | Low ✅ **Implemented** |
| Best-checkpoint saving | Returns the iterate with best loss/Q rather than the last one | Low ✅ **Implemented** |
| RDC cross-validation split | Revealed PAG overfitting in 2KZV at ratio 3.6× | Low ✅ **Implemented** |
| Auto-weight RDC by ratio | PAG=0.36, PEG=0.32 for 2KZV; reduces incentive to overfit underdetermined media | Low ✅ **Implemented** |
| Anchored-segment NeRF for long chains | Eliminates HR2876B drift problem | Medium |
| Cartesian + bond-angle penalty | 13× larger Cα RMSD improvement on HR2876B; structural drift 30× reduced | Medium ✅ **Implemented** |
| Regularised tensor fitting | Tikhonov penalty in fit_saupe_tensor (diff_biophys change required) | Medium |
| Riemannian Adam for torsion angles | Respects periodic (−π, π) topology | Medium |
| Per-term early stopping | Stop on experimental observable plateau, not total loss | Low |
