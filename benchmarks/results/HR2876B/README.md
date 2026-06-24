# HR2876B diff-integrator Benchmark

**Protein**: N-terminal domain of human NFU1 (Iron-Sulfur Cluster Scaffold Homolog)
**NESG Target**: HR2876B
**PDB**: [2LTM](https://www.rcsb.org/structure/2LTM) | **BMRB**: [18489](https://bmrb.io/data_library/summary/index.php?bmrbId=18489)
**Residues**: 107

**Reference**: Rosato, A., et al. (2015). *The second round of critical assessment of automated structure determination of proteins by NMR: CASD-NMR-2013.* J Biomol NMR **62**, 413–424. DOI: [10.1007/s10858-015-9928-5](https://doi.org/10.1007/s10858-015-9928-5).

---

## Files

| File | Description |
|---|---|
| `initial.pdb` | NMR model 1 backbone (N, CA, C atoms; raw from RCSB) |
| `final.pdb` | Refined backbone after 500 epochs |
| `loss_history.npy` | Per-epoch total weighted loss trace |

---

## Benchmark Design

This benchmark refines HR2876B using $C_\alpha$ chemical shifts against the CASD-NMR 2013 experimental dataset. No RDC data is used. BMRB 18489 also contains two well-determined RDC media (72 and 75 ¹⁵N–¹H values; 14–15× over-determined) which are strong candidates for a future `FixedTensorRDCLoss` extension.

**Optimization**: 500 epochs of Adam (lr=0.01) over backbone (φ, ψ) dihedral angles.

### NeRF Reconstruction Drift

`make_backbone_builder` uses ideal peptide bond lengths and angles (from the NeRF parameterization) rather than the actual PDB geometry. Over 107 residues this accumulates to significant positional drift. The structural RMSD of **6.443 Å** in the results below is measured against the NeRF-rebuilt starting structure, *not* the raw PDB model 1 — so this value reflects both genuine optimization displacement and NeRF-vs-PDB geometric differences.

The diff-biophys HR2876B benchmark independently characterizes this drift as ~14 Å Cα RMSD between raw PDB and the NeRF-rebuilt start. The 6.443 Å reported here is relative to the NeRF starting point (i.e., the geometry anchor is doing significant work restraining the structure against NeRF drift).

---

## Results

Optimization: 500 epochs, Adam (lr=0.01), geometry restraint weight=1.0.

| Metric | Before Refinement | After Refinement | Change |
|---|---|---|---|
| Cα RMSD | 1.710 ppm | **1.705 ppm** | −0.005 ppm |
| Structural drift | — | **6.443 Å** RMSD (vs NeRF start) | — |

### Interpretation

The $C_\alpha$ shift RMSD improved by only **0.005 ppm**. The large structural drift (6.443 Å) despite the geometry restraint is attributable to the NeRF reconstruction drift across 107 residues: the optimizer is simultaneously fighting the geometry anchor pulling towards the NeRF-idealized geometry and the shift gradient pulling towards better shift agreement. The two competing forces largely cancel, producing minimal net shift improvement.

The diff-biophys HR2876B benchmark documents the same phenomenon: the NeRF-rebuilt backbone already starts ~14 Å from the raw PDB structure, so the effective gradient landscape is significantly different from what would be obtained by optimizing directly in Cartesian space.

**This benchmark highlights a key limitation**: for longer proteins (>80–90 residues), NeRF-based parameterization accumulates geometric error that competes with the experimental gradient. This is a known open problem in differentiable protein structure refinement. Future work could use an alternative backbone parameterization (e.g., redundant torsion frames or Cartesian coordinates with bond-length penalties) to eliminate NeRF drift for larger proteins.

---

## NMR Observable Modules Used

| Observable | Source |
|---|---|
| Cα chemical shifts | `diff_biophys.nmr.chemical_shifts.make_ca_shift_loss` |
| Backbone builder | `diff_biophys.geometry.backbone.make_backbone_builder` (NeRF) |
| Geometry restraint | `diff_integrator.terms.geometry.GeometryLoss` |
| Optimizer | `optax.adam` |

## Available Data Not Yet Used

| Observable | Count | Notes |
|---|---|---|
| RDC list 1 (PEG) | 72 ¹⁵N–¹H | Ratio: 14.4× — excellent for `FixedTensorRDCLoss` |
| RDC list 2 (Pf1 phage) | 75 ¹⁵N–¹H | Ratio: 15.0× — gold-standard overdetermination |
