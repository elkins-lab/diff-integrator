# Use Cases

DiffBiophys makes every biophysical forward model **differentiable end-to-end** in JAX.
The same kernel that computes a predicted observable also returns exact gradients with
respect to atomic coordinates, ensemble weights, tensor components, or neural-network
parameters — whichever you are optimising.  The use cases below span the full range from
interactive student notebooks to large-scale multi-observable refinement pipelines.

---

## At a Glance

| # | Use Case | Primary Observable(s) | Typical Audience |
|---|----------|-----------------------|-----------------|
| 1 | [Experimental Structure Refinement](#1-experimental-structure-refinement) | SAXS, RDC | Structural biologists, computational chemists |
| 2 | [Physics-Informed Machine Learning](#2-physics-informed-machine-learning) | SAXS (as loss) | ML researchers, protein engineers |
| 3 | [Ensemble Weight Optimisation](#3-ensemble-weight-optimisation) | SAXS, any observable | Biophysicists, IDP researchers |
| 4 | [Differentiable MD Analysis](#4-differentiable-molecular-dynamics-analysis) | Any χ² misfit | Computational chemists, MD practitioners |
| 5 | [Automated Tensor Fitting](#5-automated-tensor-fitting) | RDC, Saupe tensor | NMR spectroscopists |
| 6 | [CD-Driven Secondary Structure Monitoring](#6-cd-driven-secondary-structure-monitoring) | Circular dichroism | Biophysicists, folding researchers |
| 7 | [Multi-Observable Joint Refinement](#7-multi-observable-joint-refinement) | SAXS + NMR shifts + RDC + CD | Advanced structural biologists |
| 8 | [Undergraduate Teaching](#8-undergraduate-teaching) | All of the above | Students, course instructors |

---

## 1. Experimental Structure Refinement

### The Problem

A homology model or cryo-EM map often leaves backbone dihedral angles poorly determined,
particularly in loop regions and disordered tails.  Classical refinement relies on
stochastic sampling (simulated annealing, Monte Carlo), which is expensive and hard to
couple tightly to solution-state data such as Small-Angle X-ray Scattering (SAXS) profiles
or Residual Dipolar Couplings (RDCs).

### How DiffBiophys Solves It

Because `debye_saxs` and `calculate_rdc` are written entirely in JAX primitives,
`jax.grad` returns exact gradients of the misfit $$\chi^2$$ with respect to every atomic
coordinate simultaneously.  A gradient-based optimiser such as Adam can therefore drive
the backbone toward a structure that simultaneously satisfies the scattering envelope
**and** the dipolar coupling restraints in a single, unified loss landscape — no alternating
or decoupled refinement loops needed.

### Code Sketch

```python
import jax
import jax.numpy as jnp
import optax
from diff_biophys.saxs import debye_saxs
from diff_biophys.nmr.rdc import calculate_rdc_from_tensor

def chi2_saxs(coords, q_vals, ff, I_exp, sigma):
    I_calc = debye_saxs(coords, q_vals, ff)
    return jnp.sum(((I_calc - I_exp) / sigma) ** 2)

def chi2_rdc(coords, bond_fn, saupe, d_max, rdc_exp):
    vecs = bond_fn(coords)          # (N, 3) unit bond vectors
    rdc_calc = calculate_rdc_from_tensor(vecs, saupe, d_max)
    return jnp.sum((rdc_calc - rdc_exp) ** 2)

def loss(coords):
    return (chi2_saxs(coords, q, ff, I_exp, sigma)
            + 0.5 * chi2_rdc(coords, get_NH_vectors, S, d_max, D_exp))

# Gradient-based refinement with Adam
optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(coords)

@jax.jit
def step(coords, opt_state):
    grads = jax.grad(loss)(coords)
    updates, opt_state = optimizer.update(grads, opt_state)
    return optax.apply_updates(coords, updates), opt_state

for _ in range(500):
    coords, opt_state = step(coords, opt_state)
```

### Key Advantage

Exact gradients converge orders of magnitude faster than stochastic sampling,
making previously impractical high-resolution, multi-restraint refinements
routine on a single GPU.

---

## 2. Physics-Informed Machine Learning

### The Problem

Generative models for protein structures (VAEs, diffusion models, flow matching) learn
from sequence–structure databases but have no inherent incentive to produce structures
consistent with solution-state experimental data.  Adding a biophysical regulariser
usually requires finite-difference gradients or a surrogate model, both of which
introduce approximation error.

### How DiffBiophys Solves It

`debye_saxs` is a pure JAX function, so it composes directly into any JAX or
JAX-compatible (e.g. Flax, Haiku) neural-network loss.  During training, the
auto-decoder reconstructs coordinates from a latent vector, the Debye kernel
evaluates the predicted scattering curve, and the SAXS misfit is back-propagated
through the decoder weights in a single `jax.grad` call — no finite differences,
no surrogate.

### Code Sketch

```python
import jax
import jax.numpy as jnp
from flax import linen as nn
from diff_biophys.saxs import debye_saxs

class StructureDecoder(nn.Module):
    n_atoms: int

    @nn.compact
    def __call__(self, z):
        h = nn.Dense(256)(z)
        h = nn.relu(h)
        h = nn.Dense(128)(h)
        h = nn.relu(h)
        coords = nn.Dense(self.n_atoms * 3)(h).reshape(self.n_atoms, 3)
        return coords

def biophysical_loss(params, z, q_vals, ff, I_exp, sigma, beta=0.1):
    coords = decoder.apply(params, z)            # decode latent → coords
    I_calc = debye_saxs(coords, q_vals, ff)      # physics forward model
    saxs_term = jnp.mean(((I_calc - I_exp) / sigma) ** 2)
    recon_term = jnp.mean((coords - coords_ref) ** 2)  # reconstruction
    return recon_term + beta * saxs_term

grads = jax.grad(biophysical_loss)(params, z, q_vals, ff, I_exp, sigma)
```

### Key Advantage

The physical constraint is propagated through the decoder without any surrogate
approximation, forcing the latent space to remain in a region of conformational
space that is consistent with real experimental data.

---

## 3. Ensemble Weight Optimisation

### The Problem

SAXS, SANS, and NMR observables measured in solution represent **population averages**
over a conformational ensemble, not a single rigid structure.  For disordered proteins or
multidomain systems, fitting a single conformation leads to systematic bias.
The challenge is to recover population weights $$\{w_i\}$$ for a pre-generated library of
$$M$$ conformers such that $$\bar{I}(q) = \sum_i w_i\, I_i(q)$$ matches the experiment.

### How DiffBiophys Solves It

JAX's `vmap` evaluates the Debye kernel for all $$M$$ conformers in a single
vectorised pass across the batch dimension, making the forward computation
nearly as cheap as a single-conformation calculation on a GPU.  Because the
weighted sum is differentiable, `jax.grad` returns $$\partial \chi^2 / \partial w_i$$
for every conformer simultaneously, enabling efficient gradient-based weight fitting with
an optional entropy regulariser to prevent over-fitting to sparse data.

### Code Sketch

```python
import jax
import jax.numpy as jnp
import optax
from diff_biophys.ensemble import calculate_ensemble_saxs

# coords_library : (M, N, 3)  — pre-generated conformer library
# log_weights are unconstrained; softmax enforces w_i > 0, sum = 1

log_w = jnp.zeros(M)   # initialise uniform

def loss(log_w):
    w = jax.nn.softmax(log_w)               # normalised weights
    I_calc = calculate_ensemble_saxs(
        coords_library, w, q_vals, ff)      # vmapped Debye over M conformers
    chi2 = jnp.sum(((I_calc - I_exp) / sigma) ** 2)
    # Entropy regularisation prevents all weight collapsing onto one conformer
    entropy = -jnp.sum(w * jnp.log(w + 1e-12))
    return chi2 - 0.01 * entropy

optimizer = optax.adam(5e-3)
opt_state = optimizer.init(log_w)

@jax.jit
def step(log_w, state):
    g = jax.grad(loss)(log_w)
    updates, state = optimizer.update(g, state)
    return optax.apply_updates(log_w, updates), state

for _ in range(1000):
    log_w, opt_state = step(log_w, opt_state)

weights = jax.nn.softmax(log_w)
```

### Key Advantage

`vmap` over conformers requires **zero hand-written loops**, scales linearly with
ensemble size on GPU/TPU, and fits naturally into gradient-based Maximum Entropy
or Bayesian reweighting frameworks.

---

## 4. Differentiable Molecular Dynamics Analysis

### The Problem

Molecular dynamics simulations produce trajectories of atomic coordinates, but the
connection to solution-state experimental observables is typically post-hoc: the
experimenter computes predicted SAXS curves or RDCs frame-by-frame and compares to
data, but cannot easily determine which atoms or degrees of freedom are responsible
for the misfit — or use that information to steer the simulation.

### How DiffBiophys Solves It

The gradient $$\nabla_{\mathbf{r}}\,\chi^2$$ is the **experimental force** on every atom
implied by the mismatch between prediction and experiment.  Because DiffBiophys kernels
are differentiable JAX functions, this force can be computed in one `jax.grad` call and
either used analytically to diagnose which atoms drive the discrepancy, or directly added
to the MD force field as a biasing potential to restrain the trajectory toward
experimental consistency.

### Code Sketch

```python
import jax
import jax.numpy as jnp
from diff_biophys.saxs import debye_saxs

def chi2(coords, q_vals, ff, I_exp, sigma):
    I_calc = debye_saxs(coords, q_vals, ff)
    return jnp.sum(((I_calc - I_exp) / sigma) ** 2)

# Experimental force on each atom (shape: same as coords, i.e. (N, 3))
experimental_force = jax.grad(chi2)(coords, q_vals, ff, I_exp, sigma)

# Per-atom force magnitude — highlights which regions drive the misfit
force_magnitude = jnp.linalg.norm(experimental_force, axis=-1)  # (N,)

# Add as a biasing term in an MD integrator (e.g. JAX-MD)
def total_force(coords):
    md_force = md_force_fn(coords)
    exp_force = -jax.grad(chi2)(coords, q_vals, ff, I_exp, sigma)
    return md_force + lambda_exp * exp_force
```

### Key Advantage

A single `jax.grad` call simultaneously identifies the atoms most responsible for
experimental misfit **and** provides the restoring force needed to correct them —
analysis and steering in one line.

---

## 5. Automated Tensor Fitting

### The Problem

Fitting Residual Dipolar Couplings to a protein structure requires knowing the alignment
tensor (Saupe tensor **S**) that describes how the molecule orients in the liquid-crystal
medium.  The tensor depends on the structure, but the structure is what we are trying to
refine — an inherently circular problem that classical workflows solve with slow,
manual iterations between tensor estimation and coordinate refinement.

### How DiffBiophys Solves It

`fit_saupe_tensor` performs a differentiable SVD-based least-squares fit of
**S** from current bond vectors and experimental RDCs.  Because both the tensor
fitting and the subsequent RDC back-calculation are JAX-differentiable,
the entire pipeline — from Cartesian coordinates to the RDC Q-factor — can be
differentiated in one pass.  Coordinates and tensor parameters can therefore be
**jointly optimised** in a single gradient loop without any manual alternation.

### Code Sketch

```python
import jax
import jax.numpy as jnp
from diff_biophys.nmr.rdc import (
    fit_saupe_tensor,
    calculate_rdc_from_tensor,
    calculate_q_factor,
)

def rdc_loss(coords):
    # Bond vectors from current coordinates
    vecs = get_NH_unit_vectors(coords)           # (N, 3)

    # Differentiable SVD tensor fit (no manual tensor guessing)
    S = fit_saupe_tensor(vecs, D_exp, d_max=21.7)

    # Back-calculate RDCs from the freshly fitted tensor
    D_calc = calculate_rdc_from_tensor(vecs, S, d_max=21.7)

    # Q-factor as the optimisation target
    return calculate_q_factor(D_calc, D_exp)

# Gradient of Q w.r.t. atomic coords — drives joint structure + tensor refinement
grad_coords = jax.grad(rdc_loss)(coords)
```

### Key Advantage

The tensor is implicitly optimised at every gradient step without any alternating
minimisation bookkeeping, and the SVD fitting step never leaves the JAX computation
graph — gradients flow through it automatically.

---

## 6. CD-Driven Secondary Structure Monitoring

### The Problem

Circular dichroism (CD) spectroscopy is sensitive to secondary structure content:
α-helices produce a characteristic double-minimum at 208 nm and 222 nm, while
β-sheets produce a minimum near 218 nm.  During in-silico folding simulations
it is difficult to monitor how helical content evolves in real time, and even
harder to identify which chromophores (amide groups) are responsible for deviations
from a target CD signature.

### How DiffBiophys Solves It

`simulate_cd_matrix` implements the coupled-oscillator (DeVoe matrix) theory of CD
in JAX: it computes the full dipole–dipole interaction matrix between amide chromophores
and diagonalises it to obtain a wavelength-resolved molar ellipticity spectrum.
The function is end-to-end differentiable, so `jax.grad` with respect to
`peptide_positions` reveals the spatial gradient $$\partial[\theta](\lambda)/\partial \mathbf{r}_i$$
— telling you **exactly which amide groups to move** to increase or decrease
helical CD signal.

### Code Sketch

```python
import jax
import jax.numpy as jnp
from diff_biophys.cd.kernels import simulate_cd_matrix

wavelengths = jnp.linspace(185.0, 260.0, 150)   # nm

def cd_helix_content(peptide_positions, dipole_dirs):
    """Proxy for helical content: negative ellipticity at 222 nm."""
    theta = simulate_cd_matrix(
        peptide_positions, dipole_dirs, wavelengths,
        f_osc=0.2, gamma=10.0, lambda_0=190.0,
    )
    # Index closest to 222 nm
    idx_222 = jnp.argmin(jnp.abs(wavelengths - 222.0))
    return theta[idx_222]   # more negative → more helix

# Gradient w.r.t. chromophore positions (shape: same as peptide_positions)
grad_pos = jax.grad(cd_helix_content)(peptide_positions, dipole_dirs)

# Use as a biasing force during a folding simulation to encourage helical geometry
folding_force = -lambda_cd * grad_pos

# Monitor helical content across a trajectory
theta_trajectory = jax.vmap(
    lambda pos: simulate_cd_matrix(pos, dipole_dirs, wavelengths)
)(trajectory_positions)   # (T, M) — T frames, M wavelengths
```

### Key Advantage

Gradients of a CD spectrum with respect to chromophore positions provide a
physically interpretable, per-residue helicity sensitivity map that no
conventional MD analysis tool can produce.

---

## 7. Multi-Observable Joint Refinement

### The Problem

No single experimental observable fully constrains a protein structure: SAXS
gives global shape but is blind to local geometry; RDCs constrain bond orientations
but not distances; Cα chemical shifts are sensitive to secondary structure but
not tertiary packing; and CD reports on bulk helical content only.
Combining these data streams in a single self-consistent refinement has historically
required fragile custom code that chains separate fitting programs.

### How DiffBiophys Solves It

Because every DiffBiophys kernel returns a JAX array, their losses can be summed
algebraically into one composite objective.  Weights $$\lambda_k$$ balance
contributions from each data type; gradients from all terms flow back through
the same `jax.grad` call; and JIT compilation fuses the entire computation into
a single, optimised XLA kernel.  The result is a single gradient-descent loop
that simultaneously satisfies constraints from four independent experimental
techniques.

$$
\mathcal{L}(\mathbf{r}) =
  \lambda_\text{SAXS}\,\chi^2_\text{SAXS}
  + \lambda_\text{shift}\,\chi^2_\text{shift}
  + \lambda_\text{RDC}\,\chi^2_\text{RDC}
  + \lambda_\text{CD}\,\chi^2_\text{CD}
$$

### Code Sketch

```python
import jax
import jax.numpy as jnp
import optax
from diff_biophys.saxs import debye_saxs
from diff_biophys.nmr.chemical_shifts import predict_ca_shifts
from diff_biophys.nmr.rdc import (
    fit_saupe_tensor, calculate_rdc_from_tensor, calculate_q_factor
)
from diff_biophys.cd.kernels import simulate_cd_matrix

# --- Observable weights (tuned to comparable scale) ---
LAMBDA = dict(saxs=1.0, shift=5.0, rdc=2.0, cd=0.5)

def composite_loss(coords):
    # 1. SAXS: global shape
    I_calc = debye_saxs(coords, q_vals, ff)
    chi2_saxs = jnp.sum(((I_calc - I_exp_saxs) / sigma_saxs) ** 2)

    # 2. Cα chemical shifts: secondary structure
    phi, psi = get_backbone_torsions(coords)
    delta_calc = predict_ca_shifts(phi, psi, rc_shifts)
    chi2_shift = jnp.sum(((delta_calc - delta_exp) / sigma_shift) ** 2)

    # 3. RDCs: bond orientation
    vecs = get_NH_unit_vectors(coords)
    S = fit_saupe_tensor(vecs, D_exp, d_max=21.7)
    D_calc = calculate_rdc_from_tensor(vecs, S, d_max=21.7)
    chi2_rdc = calculate_q_factor(D_calc, D_exp)

    # 4. CD: helical content
    theta_calc = simulate_cd_matrix(
        get_amide_positions(coords), get_dipole_dirs(coords), wl_cd
    )
    chi2_cd = jnp.sum(((theta_calc - theta_exp) / sigma_cd) ** 2)

    return (LAMBDA["saxs"]  * chi2_saxs
          + LAMBDA["shift"] * chi2_shift
          + LAMBDA["rdc"]   * chi2_rdc
          + LAMBDA["cd"]    * chi2_cd)

# Single jit-compiled refinement step
optimizer = optax.adam(5e-4)
opt_state = optimizer.init(coords)

@jax.jit
def step(coords, state):
    loss_val, grads = jax.value_and_grad(composite_loss)(coords)
    updates, state = optimizer.update(grads, state)
    return optax.apply_updates(coords, updates), state, loss_val

for epoch in range(2000):
    coords, opt_state, loss_val = step(coords, opt_state)
    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | loss = {loss_val:.4f}")
```

### Key Advantage

All four experimental constraints share one gradient tape: JIT compilation fuses
the forward passes into a single XLA kernel, so four-observable refinement
costs only marginally more GPU time than a single-observable run.

---

## 8. Undergraduate Teaching

### The Problem

Biophysics courses traditionally teach forward models (the Debye formula, the Karplus
equation, the dipole-coupling RDC formula) as pencil-and-paper exercises.  Students
gain little intuition for **why** experimental observables constrain structure, or
how sensitive each observable is to different degrees of freedom.

### How DiffBiophys Solves It

DiffBiophys makes every forward model interactive and inspectable.  A student can
load a PDB file, compute a SAXS curve, perturb a single dihedral angle, and watch
the curve change — all in a Jupyter notebook.  More powerfully, they can call
`jax.grad` to produce a **gradient map**: a vector field over all atomic coordinates
that shows how much each atom shifts the predicted observable.  This turns abstract
partial derivatives into a vivid, three-dimensional visualisation of physical
sensitivity.

The library ships three tutorial notebooks that build up progressively:

| Notebook | Topic | Key JAX concept |
|----------|-------|-----------------|
| `01_saxs_basics.ipynb` | Computing and fitting a SAXS curve | `jax.jit`, `jax.grad` |
| `02_rdc_tensor.ipynb` | Fitting a Saupe tensor with SVD | Differentiable linear algebra |
| `03_ensemble_reweighting.ipynb` | Ensemble weight optimisation | `jax.vmap`, `optax` |

### Code Sketch

```python
# Recommended classroom exercise: visualise the SAXS gradient map
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from diff_biophys.saxs import debye_saxs

# Forward model: scattering at a single q value
def I_at_q(coords, q_scalar, ff):
    q = jnp.array([q_scalar])
    return debye_saxs(coords, q, ff)[0]

# Gradient of scattering intensity w.r.t. every atomic coordinate
grad_fn = jax.grad(I_at_q)
sensitivity = grad_fn(coords, q_target, ff)   # shape (N, 3) — same as coords

# Per-atom sensitivity magnitude: which atoms matter most at this q?
per_atom = jnp.linalg.norm(sensitivity, axis=-1)  # (N,)

# Plot on a bar chart — students immediately see which atoms drive scattering
residue_indices = np.arange(len(per_atom))
plt.bar(residue_indices, np.array(per_atom))
plt.xlabel("Atom index")
plt.ylabel(r"$\|\nabla_{\mathbf{r}_i} I(q)\|$")
plt.title(f"SAXS sensitivity map at q = {q_target:.2f} Å⁻¹")
plt.show()
```

!!! tip "Pedagogical Note"
    Ask students to repeat the sensitivity plot at several values of $$q$$.  At low
    $$q$$ the gradient is dominated by atoms in the molecular periphery (global shape);
    at high $$q$$ inner-shell atoms gain prominence (local electron density fluctuations).
    This concretely demonstrates the physical meaning of the scattering vector without
    any hand-waving.

!!! note "Getting the Notebooks"
    The tutorial notebooks live in the `examples/` directory of the repository and are
    also rendered in the **Examples** section of this documentation site.  Each notebook
    is self-contained and requires only `diff-biophys`, `jax`, `optax`, and `matplotlib`.

### Key Advantage

Automatic differentiation converts every forward model into a built-in sensitivity
analysis tool, giving students physical intuition that is impossible to develop
from static equations alone.
