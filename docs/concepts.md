# 🌉 The Interdisciplinary Bridge: Concepts & Context

DiffBiophys sits at the intersection of two distinct fields: **Deep Learning / Machine Learning (ML)** and **Structural Biology**. This page acts as a "Rosetta Stone" to help specialists from either domain understand the context, terminology, and value of this library. A [Glossary](#glossary) for undergraduates is provided at the end.

---

## 📖 For Machine Learning Engineers

You know how to build, train, and deploy neural networks in PyTorch or JAX. The biology may be unfamiliar, but the abstractions map cleanly onto what you already know.

### The Problem: Proteins Are Dynamic, and Biology Lives in 1D Spectra

AlphaFold revolutionised biology by predicting 3D atomic coordinates from a 1D amino acid sequence. But AlphaFold returns a *single static snapshot*. Real proteins exist in solution — constantly breathing, folding, and visiting many conformations as they perform their function. To characterise this motion experimentally, biologists collect spectra: **SAXS** (Small-Angle X-ray Scattering) and **NMR** (Nuclear Magnetic Resonance) are the two most important.

Neither experiment gives you a 3D picture. Instead, each produces a **1D curve** (or a sparse set of scalar observables) that encodes an *ensemble average* over all the conformations the protein visits. SAXS measures how X-rays scatter off the protein in solution, encoding information about its overall shape and compactness as a function of momentum transfer $$q$$. NMR chemical shifts report on the local electronic environment of individual atoms, sensitive to hydrogen-bonding, secondary structure, and ring currents. RDCs (Residual Dipolar Couplings) report bond-vector orientations relative to an external magnetic field.

### Why Legacy Code Was Not Differentiable

For decades the software that converts 3D coordinates → predicted spectra was written in Fortran 77 or C, packaged as compiled binaries with rigid I/O conventions (think Xplor-NIH, CRYSOL, SPARTA+). These programs were:

- **Opaque to autograd engines** — no computation graph is exposed.
- **Not composable** — you cannot fuse a SAXS forward pass with an NMR forward pass inside a single loss function.
- **Batch-unfriendly** — running 10,000 ensemble members required launching 10,000 subprocess calls.

DiffBiophys rewrites the physics as **pure JAX kernels**: every arithmetic operation is tracked by JAX's tracing engine, so gradients flow back through the physics all the way to whatever upstream parameters (neural network weights, torsion angles, latent embeddings) you care about.

### What NeRF Is — The Robot Arm Analogy

Predicting or optimising *torsion angles* is natural for both biophysics (Ramachandran plot) and ML (bounded, periodic parameters). But loss functions almost always need *Cartesian coordinates* — distances, dot products, scattering integrals — which require converting angles to XYZ. This is exactly what robot kinematics does when it converts joint angles to end-effector position.

**NeRF (Natural Extension Reference Frame)** is the differentiable forward-kinematics algorithm for protein chains. Think of each peptide bond as a rigid segment and each $$\phi$$, $$\psi$$, $$\omega$$ angle as a revolute joint. NeRF walks the chain bond by bond, placing each atom in the coordinate frame defined by the previous three atoms. Because the entire procedure is matrix multiplications and rotations, JAX can differentiate through it.

### Plugging Into a JAX/PyTorch Pipeline

DiffBiophys is designed to be a **drop-in differentiable physics layer**. A minimal integration looks like:

```python
import jax
import jax.numpy as jnp
from diff_biophys.saxs import debye_saxs
from diff_biophys.nerf import torsions_to_coords

# coords: (N_atoms, 3) — could come from a neural network decoder
def loss_fn(torsions, q_vals, I_exp):
    coords  = torsions_to_coords(torsions)     # NeRF: angles → XYZ
    I_pred  = debye_saxs(coords, q_vals)       # Debye: XYZ → SAXS curve
    return jnp.mean((I_pred - I_exp) ** 2)     # MSE / chi-squared

# Exact gradients w.r.t. torsion angles — no finite differences
grads = jax.grad(loss_fn)(torsions, q_vals, I_exp)
```

For PyTorch interop, JAX arrays are exported via `dlpack` (zero-copy on GPU) or converted with `np.array()` on CPU:

```python
import torch
import jax

# Zero-copy GPU transfer from JAX → PyTorch
torch_tensor = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(jax_array))
```

### Concept Mapping: ML → Biophysics

| Machine Learning | Biophysics Equivalent | Notes |
|---|---|---|
| Tensor of shape `(N, 3)` | Cartesian coordinates of $$N$$ atoms | Rows = atoms, cols = $$x, y, z$$ |
| Latent vector $$z$$ | Torsion-angle vector $$(\phi, \psi, \chi)$$ | Internal (intrinsic) coordinates |
| Forward pass | Physics forward model (Debye, Karplus, …) | Maps coords → observable |
| Loss function | $$\chi^2$$ / energy penalty vs. experiment | Minimised during refinement |
| Backpropagation | Gradient of energy w.r.t. atomic positions | Same chain rule, different domain |
| Batching / `vmap` | Ensemble of conformers | $$N_\text{ens}$$ structures evaluated in parallel |
| Model weights | Atomic coordinates / force-field parameters | The parameters being optimised |
| Regularisation | Stereochemical restraints (bond lengths, angles) | Prevent unphysical geometries |

---

## 🧬 For Structural Biologists

You understand the physics of NMR, the Debye equation for SAXS, and the subtleties of conformational ensembles. What makes DiffBiophys different from Xplor-NIH, CNS, or Rosetta — and why should you care about "automatic differentiation"?

### What Automatic Differentiation Actually Is

People often conflate AD with *numerical* differentiation (finite differences). They are fundamentally different:

**Finite differences** approximate the derivative:

$$\frac{\partial f}{\partial x} \approx \frac{f(x + \varepsilon) - f(x)}{\varepsilon}$$

This requires two forward-model evaluations per parameter, is polluted by floating-point cancellation error for small $$\varepsilon$$, and scales as $$\mathcal{O}(N_\text{params})$$ — catastrophic for a protein with tens of thousands of atoms.

**Automatic differentiation** (specifically *reverse-mode AD*, i.e., backpropagation) computes the *exact* mathematical derivative by applying the chain rule symbolically through the computation graph. The cost is a single backward pass regardless of how many parameters you have. For a protein with 5,000 atoms (15,000 Cartesian degrees of freedom), AD computes all 15,000 partial derivatives at roughly the cost of **one extra forward pass**. The gradients are not approximate; they are exact to floating-point precision.

### Why Gradient Descent Beats Monte Carlo for High-Dimensional Problems

Simulated annealing and Monte Carlo methods explore conformation space by random perturbation. Their convergence rate scales poorly with dimensionality — in a space with $$D$$ degrees of freedom the expected number of random steps to escape a basin grows exponentially. For small peptides this is manageable; for a 200-residue protein ensemble with 10,000 members, it becomes intractable.

Gradient descent exploits the *local geometry* of the loss surface:

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \, \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_t)$$

The gradient $$\nabla \mathcal{L}$$ points *directly* towards the steepest descent direction. Adaptive optimisers (Adam, L-BFGS) further rescale steps per parameter, making optimisation robust to ill-conditioned loss landscapes — exactly the situation in ensemble refinement where SAXS and NMR restraints have very different scales and curvatures.

### Gradient descent finds local minima, not global ones.

DiffBiophys is most powerful when combined with a good initialisation (e.g., an AlphaFold prediction) or a stochastic initialisation strategy, with gradient descent performing fast, accurate local refinement.

#### ⛰️ The Real-World Limitation: Rugged Landscapes
Protein conformational space is notoriously "rugged." It is filled with thousands of local minima (traps) where a simple gradient descent ball will get stuck before reaching the "true" global minimum.

**Why differentiate anyway?**
While gradients don't solve the "global search" problem on their own, they solve the "convergence" problem. Once you are in the correct "neighborhood" (e.g., via an AlphaFold model or a `synth-pdb` generated decoy), gradient descent can find the exact experimental solution 1,000x faster than Monte Carlo sampling.

Additionally, experimental observables (like SAXS and RDCs) often have a **smoothing effect** on the landscape. Because these signals depend on the global shape of the molecule, they can "pull" the structure through small energetic bumps that would trap a purely physics-based simulation.

### How JAX `jit` + `vmap` Delivers GPU Speed for Ensembles

Two JAX primitives are central to DiffBiophys performance:

- **`jit` (just-in-time compilation):** JAX traces Python functions and compiles them to XLA (Accelerated Linear Algebra) machine code. A Debye scattering calculation that takes 40 ms in interpreted NumPy runs in < 1 ms after `jit` compilation on a GPU. Compilation happens once; subsequent calls use the cached binary.

- **`vmap` (vectorising map):** This lifts a function written for a *single* structure to operate over a *batch* of structures without any explicit loop. For ensemble calculations:

```python
from diff_biophys.saxs import debye_saxs
import jax

# ensemble_coords: (N_ens, N_atoms, 3)
batch_saxs = jax.vmap(debye_saxs, in_axes=(0, None))
I_ensemble = batch_saxs(ensemble_coords, q_vals)   # shape: (N_ens, N_q)
```

On a single A100 GPU this evaluates 10,000 SAXS profiles in approximately 200 ms — work that would take hours serially in CRYSOL.

### Software Comparison

| Feature | diff-biophys | Xplor-NIH | Rosetta | CRYSOL / ATSAS |
|---|---|---|---|---|
| Gradient-based optimisation | ✅ Exact AD | ⚠️ Finite diff. | ⚠️ Finite diff. | ❌ |
| GPU acceleration | ✅ Native XLA | ❌ | ❌ | ❌ |
| Ensemble calculations | ✅ `vmap` | ⚠️ Serial | ⚠️ Serial | ❌ |
| Differentiable loss | ✅ | ❌ | ❌ | ❌ |
| NMR restraints | ✅ | ✅ | ⚠️ Limited | ❌ |
| SAXS forward model | ✅ Debye | ⚠️ External | ❌ | ✅ |
| Python-native API | ✅ | ⚠️ Tcl/Python | ⚠️ C++/PyRosetta | ❌ |
| Composable loss functions | ✅ | ❌ | ❌ | ❌ |

### Compatibility Notes

DiffBiophys arrays are JAX `DeviceArray` objects, interoperable with the full scientific Python ecosystem:

- **NumPy:** `np.array(jax_array)` copies to CPU. Suitable for passing results to MDAnalysis, MDTraj, or Biopython.
- **PyTorch (zero-copy):** `torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(x))` avoids any host-side copy when both tensors live on the same GPU.
- **PDB export:** `diff_biophys.io.write_pdb(coords, topology, path)` writes standard PDB files from optimised coordinates for downstream use in PyMOL, UCSF ChimeraX, or PHENIX.

---

## 🎓 Deep Dive into JAX

To truly master differentiable biophysics, it helps to understand the engine under the hood. We recommend exploring the broader JAX ecosystem:

### Core Learning
- **[Official JAX Documentation](https://jax.readthedocs.io/):** The source of truth for the library.
- **[JAX 101 Tutorials](https://jax.readthedocs.io/en/latest/jax-101/index.html):** A linear path from "What is JAX?" to advanced vectorization and compilation.
- **[Knife-Edge Performance (The Sharp Bits)](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html):** Essential reading for avoiding common errors with side effects and state.

### Scientific Context
- **[JAX, M.D. (ArXiv)](https://arxiv.org/abs/1912.04232):** Learn how the concepts in `diff-biophys` apply to large-scale molecular dynamics and physics simulations.
- **[Automatic Differentiation in Machine Learning: A Survey](https://arxiv.org/abs/1502.05767):** A deep dive into the math behind the gradients we use for structural refinement.

### High-Level Frameworks
- **[Optax](https://github.com/google-deepmind/optax):** The gradient processing and optimization library used in our tutorials for managing the Adam optimizer.
- **[Equinox](https://docs.kidger.site/equinox/):** Provides an elegant, class-based way to build models in JAX, making it very accessible for PyTorch users.
- **[Flax](https://flax.readthedocs.io/):** Google's flagship neural network library for JAX, ideal for building the AI models that feed into `diff-biophys` loss functions.

---

## 📚 Glossary

*A plain-English reference for undergraduates. Terms are defined in the context of DiffBiophys and structural biology. Cross-references to other documentation pages are noted where relevant.*

---

**Automatic Differentiation (AD)** — A computational technique that applies the chain rule of calculus *exactly* through a computer program, yielding analytical gradients. Unlike finite differences, AD is exact to floating-point precision and requires only a single backward pass regardless of the number of parameters. See [For Structural Biologists](#for-structural-biologists) and [theory.md](theory.md).

**Backpropagation** — The specific algorithm that implements reverse-mode automatic differentiation in a computation graph. It propagates the gradient of a scalar loss backwards through each operation in sequence. In DiffBiophys, "backprop" flows through the physics forward model (e.g., the Debye equation) rather than through neural-network layers.

**Bond Angle** — The angle formed at a central atom between two covalent bonds, e.g., the N–Cα–C angle in a peptide (~111°). Bond angles are usually held near their ideal values by restraints during structure refinement. Significant deviations indicate steric strain or poor stereochemistry.

**Bond Length** — The equilibrium distance between two covalently bonded atoms (e.g., Cα–C ≈ 1.52 Å). Bond lengths are among the most tightly constrained geometric quantities in structure refinement because quantum chemistry fixes them to within a few hundredths of an ångström.

**Cα Chemical Shift** — The NMR resonance frequency of a protein's alpha-carbon, referenced to a standard (DSS at 0 ppm). Cα shifts are sensitive to backbone conformation: helical residues resonate ~3 ppm downfield and β-strand residues ~2 ppm upfield relative to random-coil values, making them reliable secondary-structure reporters. See [theory.md](theory.md).

**Circular Dichroism (CD)** — A spectroscopic technique measuring the differential absorption of left- versus right-circularly polarised light by a chiral molecule. Protein far-UV CD spectra (190–250 nm) report on bulk secondary-structure content (% helix, strand, coil) and provide a fast global restraint on conformational ensembles.

**Chirality (L vs D amino acids)** — The handedness of the Cα tetrahedral center. All natural amino acids are L-configured: the four substituents (H, NH₂, COOH, side chain) are arranged with a specific sense relative to the backbone. D-amino acids (mirror images) are rare in nature but can arise in Cartesian refinement when large gradients push a Cα past the L→D inversion boundary. DiffIntegrator's `ChiralityPenalty` guards against this using the signed scalar triple product $\chi_i = (\vec{N}_i - \vec{CA}_i) \times (\vec{C}_i - \vec{CA}_i) \cdot (\vec{C}_{i-1} - \vec{CA}_i)$, which is negative for all L-amino acids.

**Conformational Ensemble** — A set of many different 3D structures all representing the same protein under the same solution conditions. Because proteins are dynamic, experimental NMR and SAXS observables represent *population-weighted averages* over the ensemble. DiffBiophys optimises ensemble weights and coordinates simultaneously via differentiable ensemble averaging.

**Cryo-EM (Cryo-Electron Microscopy)** — An imaging technique in which an electron beam is directed at flash-frozen protein samples, producing 2D projections that are computationally reconstructed into 3D electron-density maps. Resolutions below 2 Å are now routinely achieved. DiffBiophys can in principle compute differentiable density-map agreement scores for Cryo-EM–guided refinement.

**Debye Formula** — The foundational equation for computing a theoretical SAXS profile from atomic coordinates:

$$I(q) = \sum_{i} \sum_{j} f_i(q)\, f_j(q)\, \frac{\sin(q\, r_{ij})}{q\, r_{ij}}$$

where $$f_i(q)$$ are atomic form factors and $$r_{ij}$$ are all pairwise interatomic distances. DiffBiophys implements this as a JAX-differentiable kernel. See [theory.md](theory.md).

**Dihedral Angle** — The angle between two planes each defined by three consecutive atoms in a four-atom chain, measured by rotation about the central bond. Backbone dihedrals $$\phi$$ (C–N–Cα–C) and $$\psi$$ (N–Cα–C–N) define protein secondary structure and are the primary optimisation parameters in DiffBiophys. See also *Torsion Angle* and *Ramachandran Plot*.

**Differentiable Function** — A function whose derivative exists at every point in its domain. In the context of DiffBiophys, a differentiable forward model means JAX can compute $$\partial\mathcal{L}/\partial\theta$$ exactly for any parameter $$\theta$$ using the chain rule, enabling gradient-based structure refinement.

**Form Factor** — The amplitude with which an atom or group of atoms scatters X-rays as a function of momentum transfer $$q$$. Each element has a characteristic form factor $$f(q)$$ that peaks at $$q=0$$ and decreases with increasing $$q$$. Atomic form factors enter directly into the Debye formula.

**Forward Kinematics** — The process of computing Cartesian positions from a set of joint angles — exactly what NeRF does for protein chains. Given $$\phi, \psi$$ angles at each residue, forward kinematics places every backbone atom in 3D space. This is the differentiable building block of DiffBiophys's coordinate-generation pipeline. See *NeRF*.

**FSC (Fourier Shell Correlation)** — A resolution metric for Cryo-EM maps that measures the correlation between two independently determined half-maps in concentric shells of reciprocal space. Resolution is conventionally reported at the FSC = 0.143 threshold. DiffBiophys can compute differentiable map-model agreement scores analogous to FSC.

**Gradient** — The vector of all partial derivatives of a scalar function with respect to its inputs:

$$\nabla_\theta \mathcal{L} = \left(\frac{\partial \mathcal{L}}{\partial \theta_1},\, \frac{\partial \mathcal{L}}{\partial \theta_2},\, \ldots\right)$$

The gradient points in the direction of steepest *increase* of $$\mathcal{L}$$; gradient descent moves in the opposite direction to minimise the loss.

**Gradient Descent** — An iterative optimisation algorithm that updates parameters in the direction of the negative gradient of a loss function. Adam (Adaptive Moment estimation) and SGD (stochastic gradient descent) are the two most common variants. DiffBiophys uses gradient descent for structure refinement via the `optax` library. See [For Structural Biologists](#for-structural-biologists).

**GPU (Graphics Processing Unit)** — A massively parallel processor with thousands of cores optimised for floating-point matrix operations. Modern GPUs (NVIDIA A100, H100) deliver tens of teraFLOPS, enabling large protein ensemble calculations in seconds. JAX targets GPUs via the XLA compiler with no hand-written CUDA required. See *XLA*.

**Hamiltonian** — In classical mechanics, the total energy of a system (kinetic + potential). In molecular mechanics, the Hamiltonian is approximated by a force field (AMBER, CHARMM) summing bond, angle, dihedral, and non-bonded terms. DiffBiophys does not require a full force-field Hamiltonian, but physical energy terms can be added as differentiable regularisers to the loss function.

**Hydration Shell** — The ordered layer of water molecules surrounding a protein in solution, typically 1–3 Å thick and with elevated electron density relative to bulk solvent. The hydration shell contributes significantly to the low-$$q$$ SAXS signal and must be modelled, typically via a contrast parameter $$\delta\rho$$. DiffBiophys includes a differentiable hydration-shell correction.

**Internal Coordinates** — A description of molecular geometry using bond lengths, bond angles, and dihedral angles rather than Cartesian $$(x, y, z)$$ positions. Internal coordinates are more compact ($$3N-6$$ values vs $$3N$$) and directly interpretable in chemical terms. NeRF converts internal coordinates to Cartesian. See *NeRF* and *Torsion Angle*.

**JAX** — A Python numerical computing library from Google Research combining NumPy-compatible syntax with automatic differentiation (`grad`), just-in-time compilation to XLA (`jit`), and automatic vectorisation (`vmap`). DiffBiophys is built entirely on JAX. See [jax.readthedocs.io](https://jax.readthedocs.io) and [the Why JAX section](index.md).

**JIT Compilation (Just-In-Time)** — A compilation strategy in which code is compiled to machine instructions at the moment of first call rather than ahead of time. JAX's `@jax.jit` decorator traces a Python function, converts it to an XLA computation graph, and emits optimised GPU/CPU binaries. Subsequent calls reuse the compiled binary, essentially eliminating Python overhead.

**Karplus Equation** — An empirical relationship between an NMR scalar J-coupling constant (Hz) and a dihedral angle $$\theta$$:

$$J = A\cos^2\!\theta + B\cos\theta + C$$

where $$A$$, $$B$$, $$C$$ are atom-type-dependent constants determined by quantum-chemical calculations. DiffBiophys implements differentiable Karplus equations for backbone $${}^3J_{HN\text{-}H\alpha}$$ couplings and sidechain $$\chi$$-angle couplings. See [theory.md](theory.md).

**Kabsch Algorithm** — An algorithm that finds the optimal rigid-body rotation (and optionally translation) to superimpose two corresponding sets of 3D points, minimising their RMSD. DiffBiophys provides a differentiable Kabsch implementation used for ensemble alignment and RMSD-based loss terms.

**Loss Function** — A scalar-valued function measuring how poorly a model's predictions match experimental data; the quantity that gradient descent minimises. In DiffBiophys a typical loss is:

$$\mathcal{L} = w_\text{SAXS}\,\chi^2_\text{SAXS} + w_\text{NMR}\,\chi^2_\text{NMR} + w_\text{stereo}\,E_\text{stereo}$$

where the weights $$w$$ balance the contribution of each restraint type.

**Magic Angle** — The angle $$\theta_m = \arccos(1/\sqrt{3}) \approx 54.74°$$ at which the function $$(3\cos^2\!\theta - 1)$$ equals zero. In solid-state NMR, spinning samples at the magic angle removes broadening from dipolar couplings and chemical-shift anisotropy. In solution NMR, the magic angle appears in the angular dependence of RDCs and quadrupolar couplings.

**Mean Squared Error (MSE)** — The average squared difference between predicted and observed values:

$$\text{MSE} = \frac{1}{N}\sum_{i=1}^N \bigl(\hat{y}_i - y_i\bigr)^2$$

MSE is differentiable everywhere and is the default loss metric in DiffBiophys example notebooks. In SAXS refinement a noise-weighted variant gives the standard $$\chi^2$$ statistic.

**NeRF (Natural Extension Reference Frame)** — A differentiable algorithm that converts a chain of bond lengths, bond angles, and dihedral angles into 3D Cartesian atomic positions using successive rotation-matrix operations. Analogous to forward kinematics in robotics: each amino acid's backbone is like a rigid link connected by revolute joints. NeRF is the core coordinate-building primitive in DiffBiophys. See [For ML Engineers](#for-machine-learning-engineers) and [theory.md](theory.md).

**NMR (Nuclear Magnetic Resonance)** — A spectroscopic technique exploiting the quantum-mechanical spin of atomic nuclei (${}^1$H, ${}^{13}$C, ${}^{15}$N) to report on molecular structure and dynamics in solution. NMR provides residue-level observables — chemical shifts, J-couplings, NOEs, RDCs — all of which DiffBiophys can predict from 3D coordinates and differentiate.

**NOE (Nuclear Overhauser Effect)** — A through-space magnetization transfer between two protons closer than ~5–6 Å.  NOE intensities decay as $1/r^6$, providing distance bounds on inter-proton separations.  In DiffIntegrator, these are encoded as flat-bottomed harmonic restraints in `NOELoss`: zero energy within $[d^\text{lower}, d^\text{upper}]$, quadratic outside.  NOE-derived distance restraints are the most information-dense NMR observable for determining protein fold topology.

**Optimizer (Adam / SGD)** — An algorithm that updates model parameters using gradient information. **SGD** applies a uniform learning rate $$\eta$$ to all parameters. **Adam** maintains per-parameter estimates of the first and second moments of the gradient, adapting the effective step size and converging faster on heterogeneous loss landscapes. DiffBiophys uses `optax.adam` by default.

**Pair-Distance Distribution P(r)** — The histogram of all pairwise interatomic distances in a molecule, related to the SAXS intensity by a Fourier-sine transform:

$$I(q) = 4\pi \int_0^{D_\text{max}} P(r)\, \frac{\sin(qr)}{qr}\, \mathrm{d}r$$

The peak of $$P(r)$$ gives the most probable distance; $$D_\text{max}$$ is the maximum particle dimension. DiffBiophys computes $$P(r)$$ differentiably from atomic coordinates.

**Parameter** — Any numerical quantity adjusted during optimisation. In structure refinement, parameters are typically atomic coordinates or torsion angles; in neural networks they are weights and biases. DiffBiophys differentiates the loss with respect to whichever leaf variables `jax.grad` is given.

**Ramachandran Plot** — A 2D scatter plot of backbone dihedral angles $$(\phi, \psi)$$ for every residue in a protein. Allowed regions correspond to geometrically favourable secondary structures (helix ~−60°/−40°; strand ~−120°/120°); disallowed regions indicate steric clashes. DiffBiophys can add a differentiable Ramachandran prior as a regularisation term.

**Random Coil** — The idealised disordered state of a polymer in which each bond rotates freely, producing a Gaussian distribution of end-to-end distances. In NMR, tabulated random-coil chemical shift values (one per residue type) serve as the reference from which secondary-structure-induced deviations (secondary chemical shifts) are measured.

**RDC (Residual Dipolar Coupling)** — An NMR observable arising when a protein is weakly aligned in an anisotropic medium (liquid crystal, stretched gel). RDCs report on the time-averaged orientation of internuclear bond vectors relative to the magnetic field and are sensitive to global fold topology. DiffBiophys computes RDCs from coordinates and the Saupe tensor:

$$D_{HN} = D_a\!\left[(3\cos^2\!\theta - 1) + \tfrac{3}{2}\eta\sin^2\!\theta\cos 2\phi\right]$$

See also *Saupe Tensor*. See [theory.md](theory.md) for derivation.

**Radius of Gyration ($$R_g$$)** — The root-mean-square distance of atoms from the protein's centre of mass, weighted by scattering length:

$$R_g^2 = \frac{\sum_i f_i\,|\mathbf{r}_i - \bar{\mathbf{r}}|^2}{\sum_i f_i}$$

$$R_g$$ is measurable from the low-$$q$$ Guinier region of a SAXS curve and is a primary target observable in DiffBiophys SAXS refinement workflows.

**Ring Current Shift** — A contribution to NMR proton chemical shifts arising from the anisotropic magnetic field induced by aromatic rings (Phe, Tyr, Trp, His). Protons above or below the ring plane experience shielding (upfield shift); those in the plane experience deshielding. DiffBiophys models ring current effects using the Haigh–Mallion formalism, fully differentiable with respect to atomic positions.

**Rosetta** — A widely used software suite for protein structure prediction and design, built on a physics-based energy function and Monte Carlo sampling. Rosetta is not natively differentiable, although PyRosetta exposes a Python API. The two tools are complementary: Rosetta for diverse global sampling, DiffBiophys for fast, gradient-based local refinement.

**SAXS (Small-Angle X-ray Scattering)** — A solution-state experiment in which a monochromatic X-ray beam illuminates a protein solution and scattered photon intensity is recorded as a function of momentum transfer $$q = 4\pi\sin\theta/\lambda$$. SAXS requires no crystals, is sensitive to overall shape, size, and flexibility, and is the primary long-range restraint implemented in DiffBiophys. See [theory.md](theory.md) and *Debye Formula*.

**Saupe Tensor** — A $$3\times 3$$ symmetric traceless matrix $$\mathbf{S}$$ describing the degree and symmetry of molecular alignment in an anisotropic medium. The five independent elements of $$\mathbf{S}$$ (equivalently the axial component $$D_a$$, rhombicity $$\eta$$, and three Euler angles) uniquely determine all RDC observables for a given structure. Fitting $$\mathbf{S}$$ to experimental RDCs is a core inverse problem that DiffBiophys solves via gradient descent.

**Secondary Structure** — The local, regular folding patterns of a polypeptide chain, stabilised by hydrogen bonds. The three principal types are: **α-helix** (right-handed, 3.6 residues/turn, $$\phi \approx -60°$$, $$\psi \approx -40°$$), **β-strand** (extended, $$\phi \approx -120°$$, $$\psi \approx 120°$$), and **random coil** (disordered). Secondary structure is strongly correlated with backbone dihedral angles and NMR chemical shifts.

**Simulated Annealing** — A stochastic global optimisation algorithm that accepts uphill moves with probability $$\exp(-\Delta E / T)$$, where the fictitious temperature $$T$$ is slowly decreased ("annealed"). This allows escape from local minima. However, simulated annealing scales poorly with dimensionality and is orders of magnitude slower than gradient descent for high-dimensional ensemble refinement.

**Softmax** — A differentiable function mapping a real-valued vector to a probability simplex:

$$\operatorname{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

In DiffBiophys, softmax is used to parameterise conformer weights in ensemble averaging: weights $$w_k = \operatorname{softmax}(\mathbf{a})_k$$ automatically satisfy $$\sum_k w_k = 1$$ and $$w_k > 0$$ throughout gradient-descent optimisation.

**Structure Refinement** — The computational process of adjusting a protein model to improve agreement with experimental data while maintaining chemically reasonable geometry. DiffBiophys performs differentiable structure refinement using gradient descent, replacing the Monte Carlo or molecular dynamics protocols used by traditional software packages.

**Tensor** — In the ML sense, a multi-dimensional array of numbers (generalisation of scalars, vectors, and matrices). In DiffBiophys, a tensor of shape `(N_atoms, 3)` stores Cartesian coordinates. In the physics sense, a tensor is a geometric object whose components transform covariantly under rotations; the Saupe tensor is a rank-2 physical tensor.

**Torsion Angle** — Synonym for *dihedral angle*: the angle of rotation about a chemical bond defined by four consecutive bonded atoms. Backbone torsion angles $$\phi$$ and $$\psi$$ are the primary internal degrees of freedom of a protein and are the natural parameters for DiffBiophys optimisation when working in internal-coordinate space.

**Transition Dipole** — A quantum-mechanical quantity describing the direction and magnitude of the oscillating electric dipole moment during an electronic or vibrational transition. Transition dipoles determine orientation-dependent absorption (e.g., in membrane proteins or aligned samples) and are relevant to DiffBiophys's circular dichroism forward model.

**vmap** — JAX's automatic vectorisation transformation. `jax.vmap(f)` converts a function `f` operating on a single input into one that operates on a batch of inputs in parallel, without any explicit loop and with no penalty for the vectorisation itself. In DiffBiophys, `vmap` is the key primitive for efficient ensemble calculations: it evaluates thousands of conformers simultaneously on a GPU in a single kernel launch. See [For Structural Biologists](#for-structural-biologists).

**XLA (Accelerated Linear Algebra)** — A domain-specific compiler for linear algebra operations developed at Google. JAX compiles computation graphs to XLA, which generates highly optimised machine code for CPUs, NVIDIA GPUs (via CUDA/cuBLAS), and Google TPUs. XLA enables DiffBiophys kernels to achieve near-peak hardware throughput without any hand-written low-level code.

---

*Last updated: 2026-06. For corrections or additions to the Glossary, please open an issue on the [GitHub repository](https://github.com/diff-biophys/diff-biophys).*
