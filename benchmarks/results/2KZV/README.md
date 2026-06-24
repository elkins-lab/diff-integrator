# 2KZV diff-integrator Benchmark

**Protein**: CV_0373(175-257) from *Chromobacterium violaceum*
**NESG Target**: CvR118A
**PDB**: [2KZV](https://www.rcsb.org/structure/2KZV) | **BMRB**: [17020](https://bmrb.io/data_library/summary/index.php?bmrbId=17020)

**Reference**: Li, Spaman, Tejero, Montelione et al. (2023). *Blind assessment of monomeric AlphaFold2 protein structure models with experimental NMR data.* PMID [37257257](https://pubmed.ncbi.nlm.nih.gov/37257257/).

---

## Files

| File | Description |
|---|---|
| `initial.pdb` | NMR initial structure (Model 1, backbone only) |
| `final.pdb` | Refined output structure |
| `loss_history.npy` | Per-epoch total loss trace |

---

## Benchmark Design

This benchmark demonstrates the core functionality of `diff-integrator` by performing a joint multi-objective optimization using three simultaneous target functions:
1. **$C_\alpha$ Chemical Shifts**: RMSD constraint
2. **Residual Dipolar Couplings (RDCs)**: Dynamic Saupe tensor fitting in two distinct alignment media (PAG and PEG)
3. **Geometry Restraint**: A weak $1.6 \text{ \AA}$ harmonic anchor to the initial NMR state to prevent unphysical backbone unravelling

### Dynamic Tensor Fitting vs. Fixed-Tensor

Unlike traditional molecular dynamics solvers (like X-PLOR or `diff-biophys`) which freeze the alignment tensor for hundreds of steps to calculate gradients, `diff-integrator` fits the Saupe tensor **dynamically at every single optimization step**. 

Because `diff-integrator` can auto-differentiate through the SVD (Singular Value Decomposition) used to fit the tensor, the optimizer computes the "true" gradient of the RDC landscape. This removes the need for discontinuous "refit-and-freeze" cycles, leading to smoother and more rapid descent.

### The Overfitting Danger

RDCs are extremely degenerate. Because they only measure the orientation of local bond vectors relative to a global magnetic field, an unrestrained protein will rapidly twist into a non-physical "pretzel" to satisfy the RDCs (driving the Q-factor to zero but destroying the global fold). 

To prevent this, this benchmark implements `diff_integrator.terms.geometry.GeometryLoss`. We balance the RDC and Chemical Shift forces against a geometric prior, finding a stable local minimum that dramatically improves the PEG alignment while preserving the native structure.

---

## Results

Optimization performed over 500 epochs using the Optax Adam optimizer.

| Metric | Before Refinement | After Refinement |
|---|---|---|
| Cα Shift RMSD | 1.542 ppm | 1.539 ppm |
| Q-factor (PAG, 23 res) | 0.309 | **0.290** |
| Q-factor (PEG, 16 res) | 0.373 | **0.060** |
| Structural Drift | — | **1.604 Å** RMSD |

The optimizer dramatically improved the alignment to the PEG medium RDCs (Q-factor from 0.373 $\rightarrow$ 0.060) while maintaining the structural integrity of the protein (1.6 $\text{\AA}$ RMSD). This perfectly illustrates the power of `JointLoss` multi-objective refinement.
