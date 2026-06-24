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

Fitting the tensor *inside* the gradient computation (differentiating through the SVD) would allow the optimizer to trivially drive Q→0 by exploiting the degeneracy between backbone orientation and tensor parameters — producing non-physical results without genuine structural improvement. `FixedTensorRDCLoss` in `diff_integrator/terms/nmr.py` enforces the correct behaviour via `jax.lax.stop_gradient`.

### Why PEG Results Must Be Interpreted with Caution

The Saupe tensor has 5 free parameters. A reliable fit requires the number of RDCs to substantially exceed the tensor parameters (Bax & Tjandra recommend ≥20 per medium):

| Medium | RDCs | Tensor params | Ratio | Role |
|---|---|---|---|---|
| PAG | 23 | 5 | **4.6×** | **Primary benchmark** |
| PEG | 16 | 5 | **3.2×** | Supplementary only ⚠️ |

With only 16 data points, there exist many small backbone distortions that can satisfy the RDC constraints without globally improving the structure. Any Q(PEG) well below the published NMR medoid value (0.36) should be treated as overfitting, not genuine improvement. The benchmark script prints an explicit warning when this occurs.

---

## Results

Optimization: 500 epochs, Adam optimizer (lr=0.01), tensor update every 50 epochs.

| Metric | Before Refinement | After Refinement | Published Target |
|---|---|---|---|
| Cα RMSD | 1.542 ppm | **1.538 ppm** | — |
| Q (PAG, 23 res) | 0.309 | **0.290** | AF2=0.22, NMR medoid=0.18 |
| Q (PEG, 16 res) | 0.373 | 0.047 ⚠️ | AF2=0.35, NMR medoid=0.36 |
| Structural drift | — | **1.628 Å** RMSD | — |

### Interpretation

**PAG (primary)**: Q improved from 0.309 → 0.290. The starting point is NMR model 1, which is not the medoid of the ensemble; the published NMR medoid target is 0.18. Continued optimization with looser geometry restraints or more epochs would be expected to approach this target.

**PEG (supplementary)**: Q = 0.047 is far below the published NMR medoid (0.36), confirming overfitting to the underdetermined 16-RDC dataset. This is expected behaviour and not a claimed result. The geometry restraint successfully held the backbone to within **1.628 Å** of the native structure, preventing physically impossible distortion.

---

## NMR Observable Modules Used

| Observable | Source |
|---|---|
| Cα chemical shifts | `diff_biophys.nmr.chemical_shifts.make_ca_shift_loss` |
| RDC back-calculation | `diff_biophys.nmr.rdc.calculate_rdc_from_tensor` |
| Fixed-tensor loss | `diff_integrator.terms.nmr.FixedTensorRDCLoss` |
| Alignment tensor fit | `diff_biophys.nmr.rdc.fit_saupe_tensor` |
| Q-factor | `diff_biophys.nmr.rdc.calculate_q_factor` |
| Backbone builder | `diff_biophys.geometry.backbone.make_backbone_builder` (NeRF) |
| Optimizer | `optax.adam` |
