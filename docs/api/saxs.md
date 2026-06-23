# 🔬 SAXS API

The `diff_biophys.saxs` subpackage implements differentiable small-angle X-ray
scattering kernels.  The core function `debye_saxs` computes the full
$O(N^2)$ pairwise Debye sum, GPU-accelerated via JAX `vmap`, with an
optional excluded-volume hydration shell correction.

---

## Debye Scattering

The **Debye formula** computes solution-state X-ray scattering intensity
from atomic coordinates and form factors:

$$I(q) = \sum_i \sum_j f_i(q)\, f_j(q)\, \frac{\sin(q r_{ij})}{q r_{ij}}$$

where $q$ is the momentum transfer (Å⁻¹), $r_{ij}$ is the pairwise
inter-atomic distance, and $f_i(q)$ are atomic form factors.

**Hydration shell correction** (Fraser et al. 1978): a solvent layer
surrounding the protein contributes excess scattering density.  The
correction subtracts a bulk-solvent term scaled by the excluded volume
of each atom.

**Gradient meaning:** $\partial I(q) / \partial \mathbf{r}_i$ reveals which
atom, if displaced, would most change the scattering intensity at
momentum transfer $q$.  At low $q$, this is dominated by the
overall shape (Rg); at high $q$, by inter-atomic distances.

```python
from diff_biophys.saxs.kernels import debye_saxs
import jax, jax.numpy as jnp

# coords: (N, 3) atomic positions in Å
# q_vals: (M,) momentum transfer grid in Å⁻¹
# form_factors: (N, M) or (N,) atomic form factors

coords       = jnp.array(...)          # (N, 3)
q_vals       = jnp.linspace(0.01, 0.5, 100)
form_factors = jnp.ones(len(coords))   # uniform (simplified)

I_q = debye_saxs(coords, q_vals, form_factors)    # (M,) in a.u.

# Chi-squared loss vs experimental profile
def saxs_chi2(c, I_exp, sigma=1.0):
    I_calc = debye_saxs(c, q_vals, form_factors)
    return jnp.mean(((I_calc - I_exp) / sigma) ** 2)

grad_coords = jax.grad(saxs_chi2)(coords, I_exp)

# JIT for speed
saxs_jit = jax.jit(debye_saxs)
```

### Multi-structure ensemble averaging

Use `jax.vmap` to evaluate the Debye sum over an ensemble of conformers
and optimise population weights:

```python
import jax

# ensemble: (K, N, 3) — K conformers
ensemble    = jnp.array(...)
weights     = jax.nn.softmax(jnp.zeros(K))   # uniform start

batch_saxs  = jax.vmap(lambda c: debye_saxs(c, q_vals, form_factors))
I_ensemble  = jnp.einsum("k,km->m", weights, batch_saxs(ensemble))

def ensemble_loss(w):
    I = jnp.einsum("k,km->m", jax.nn.softmax(w), batch_saxs(ensemble))
    return jnp.mean((I - I_exp) ** 2)

grad_w = jax.grad(ensemble_loss)(jnp.zeros(K))
```

### Guinier analysis (Rg from low-q slope)

At low $q$, $\ln I(q) \approx \ln I_0 - q^2 R_g^2 / 3$.
Fit a line to $\ln I$ vs $q^2$ to extract $R_g$:

```python
from diff_biophys.geometry.macroscopic import compute_rg

# Direct differentiable Rg (faster than Guinier fitting)
rg = compute_rg(coords)

# Rg restraint loss
rg_target = 15.0  # Å from experiment
rg_loss = (compute_rg(coords) - rg_target) ** 2
```

::: diff_biophys.saxs.kernels
