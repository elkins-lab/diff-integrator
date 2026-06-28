# 2KZV diff-integrator Benchmark

**Protein**: CV_0373(175-257) from *Chromobacterium violaceum*
**NESG Target**: CvR118A
**PDB**: [2KZV](https://www.rcsb.org/structure/2KZV) | **BMRB**: [17020](https://bmrb.io/data_library/summary/index.php?bmrbId=17020)

**Reference**: Li, Spaman, Tejero, Montelione et al. (2023). *Blind assessment of monomeric AlphaFold2 protein structure models with experimental NMR data.* PMID [37257257](https://pubmed.ncbi.nlm.nih.gov/37257257/).

---

## Files

| File | Description |
|---|---|
| `initial.pdb` | NMR model 1 backbone (N, CA, C atoms; raw from RCSB) |
| `final.pdb` | Refined backbone after 500 epochs |
| `loss_history.npy` | Per-epoch total weighted loss trace |

---

## Benchmark Design

This benchmark performs a joint multi-objective optimization against three simultaneous experimental constraints:

1. **$C_\alpha$ Chemical Shifts**: RMSD against 91 measured $C_\alpha$ shifts from BMRB 17020.
2. **RDCs (PAG medium)**: MSE against 23 ¹⁵N–¹H RDCs. Primary validation metric.
3. **RDCs (PEG medium)**: MSE against 16 ¹⁵N–¹H RDCs. Supplementary — see caution below.
4. **Geometry Restraint**: Harmonic position restraint anchoring to NMR model 1, preventing non-physical backbone unravelling.

Optimization uses backbone dihedral angles (φ, ψ) as parameters, with a NeRF builder reconstructing Cartesian coordinates at each step.

### Fixed-Tensor RDC Strategy

The Saupe alignment tensor (5 free parameters: $D_a$, $R$, and 3 Euler angles) is held **fixed** during gradient descent and re-fitted from the current backbone every 50 epochs. This is the standard approach used by X-PLOR, CNS, and PALES.

Fitting the tensor *inside* the gradient computation would allow the optimizer to trivially drive Q→0 by exploiting the degeneracy between backbone orientation and tensor parameters — producing non-physical results without genuine structural improvement. `FixedTensorRDCLoss` in `diff_integrator/terms/nmr.py` enforces the correct behaviour via `jax.lax.stop_gradient`.

### RDC Cross-Validation Split (PAG only)

20% of PAG measurements (5 RDCs) are held out as a validation set.  The training loss uses only the remaining **18 RDCs**; `evaluate_validation_q()` evaluates the held-out set with the *training-fitted* tensor.

PEG is **not** split: splitting 16 RDCs would leave fewer than 6 for training — insufficient for reliable Saupe tensor fitting.

### Auto-Weight by Overdetermination Ratio

RDC term weights are set automatically via `suggested_weight(base_weight=1.0)`:

| Medium | RDCs | Train RDCs | Ratio | Auto-weight |
|---|---|---|---|---|
| PAG | 23 | 18 | **3.6×** | **0.360** |
| PEG | 16 | 16 | **3.2×** | **0.320** |

Both are down-weighted from the previous hard-coded value of 1.0, reducing the optimizer's incentive to fit the relatively underdetermined RDC constraints at the expense of geometry and chemical shifts.

### Why PEG Results Must Be Interpreted with Caution

The Saupe tensor has 5 free parameters. A reliable fit requires the number of RDCs to substantially exceed the tensor parameters (Bax & Tjandra recommend ≥20 per medium):

| Medium | RDCs | Tensor params | Ratio | Role |
|---|---|---|---|---|
| PAG | 23 | 5 | **4.6×** | **Primary benchmark** |
| PEG | 16 | 5 | **3.2×** | Supplementary only ⚠️ |

With only 16 data points, there exist many small backbone distortions that can satisfy the RDC constraints without globally improving the structure. Any Q(PEG) well below the published NMR medoid value (0.36) should be treated as overfitting, not genuine improvement.

---

## Results

Optimization: 500 epochs, Adam optimizer (lr=0.01), tensor update every 50 epochs,
fixed geometry weight = 5.0, auto-weighted RDC terms (PAG=0.360, PEG=0.320),
PAG cross-validation split (18 train / 5 held-out RDCs).

| Metric | Before Refinement | After Refinement | Change |
|---|---|---|---|
| Cα RMSD | 1.542 ppm | **1.537 ppm** | −0.005 ppm |
| Q (PAG train, 18 RDCs) | 0.309 | **0.298** | −4% |
| Q (PAG val, 5 RDCs) | 0.309 | **0.471** | **+52% ⚠️** |
| Q (PEG, 16 RDCs) | 0.373 | 0.108 ⚠️ | −71% |
| Structural drift | — | **1.616 Å** RMSD | — |

### Interpretation

**The cross-validation split reveals overfitting on PAG.**

The training Q(PAG) improved modestly from 0.309 → 0.298 (−4%), but the held-out validation Q *worsened* from 0.309 → 0.471 (+52%).  This is a clear signature of overfitting: the optimizer is finding backbone distortions that satisfy the 18 training RDCs while simultaneously moving further from the 5 held-out measurements.  With only 18 training RDCs (ratio 3.6×), the system is too underdetermined for reliable, generalisable structural improvement.

This is the cross-validation system working exactly as designed — the Q-factor without CV would have shown 0.298, which looks like an improvement; the CV split reveals it is not.

**Q(PEG) = 0.108** is far below the published NMR medoid (0.36), confirming severe overfitting to the underdetermined 16-RDC PEG dataset. This is expected and consistent with prior runs.

**Cα RMSD** improved by 0.005 ppm — marginal but numerically consistent.

**Structural drift** of 1.616 Å is tight; the geometry anchor is holding the backbone close to the native structure throughout.

### Scientific Conclusion

For 2KZV, the RDC data available in BMRB 17020 (PAG: 23 RDCs, ratio 4.6×; PEG: 16 RDCs, ratio 3.2×) are **too underdetermined** to produce reliable structural improvement via RDC-guided refinement alone. The cross-validation experiment confirms this definitively. The published benchmark (Li et al. 2023, Table 5) uses much richer data — the full NMR ensemble and more sophisticated protocols — and achieves Q(PAG) = 0.18.

For meaningful RDC-guided refinement with diff-integrator, ratios ≥ 8× are recommended (as demonstrated by GmR58A, where all three media have ratios 8.6–11.8×).

---

## NMR Observable Modules Used

| Observable | Source |
|---|---|
| Cα chemical shifts | `diff_biophys.nmr.chemical_shifts.make_ca_shift_loss` |
| RDC back-calculation | `diff_biophys.nmr.rdc.calculate_rdc_from_tensor` |
| Fixed-tensor loss | `diff_integrator.terms.nmr.FixedTensorRDCLoss` |
| CV split factory | `diff_integrator.terms.nmr.make_rdc_cv_refinement_fns` |
| Alignment tensor fit | `diff_biophys.nmr.rdc.fit_saupe_tensor` |
| Q-factor | `diff_biophys.nmr.rdc.calculate_q_factor` |
| Auto-weight | `FixedTensorRDCLoss.suggested_weight()` |
| Backbone builder | `diff_biophys.geometry.backbone.make_backbone_builder` (NeRF) |
| Optimizer | `optax.adam` |
