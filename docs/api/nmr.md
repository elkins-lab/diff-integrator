# ⚛️ NMR API

The `diff_biophys.nmr` subpackage implements four categories of differentiable NMR
observables.  Each kernel accepts JAX arrays and returns JAX arrays, making them
trivially composable into multi-observable loss functions.

---

## Karplus J-coupling

`diff_biophys.nmr.karplus` implements the **Karplus equation** relating a three-bond
scalar coupling constant $^3J$ (Hz) to a backbone dihedral angle:

$$J(\theta) = A\cos^2(\theta) + B\cos(\theta) + C$$

For the backbone HN–Hα coupling, the dihedral $\theta = \phi - 60°$ (Vuister & Bax 1993
offset convention).  Default parameters: $A = 6.98$, $B = -1.38$, $C = 1.72$ Hz.

**Gradient meaning:** $\partial J / \partial \theta$ tells you how fast the
coupling changes with the backbone angle — the key quantity for
J-coupling-driven structure refinement.

```python
from diff_biophys.nmr.karplus import calculate_karplus_j
import jax, jax.numpy as jnp

phi = jnp.array([-0.995])          # φ in radians (≈ −57° helix)
theta = phi - jnp.deg2rad(60.0)   # offset for HN-Hα

J = calculate_karplus_j(theta, A=6.98, B=-1.38, C=1.72)   # → [Hz]
dJ_dtheta = jax.grad(lambda t: jnp.sum(calculate_karplus_j(t, 6.98, -1.38, 1.72)))(theta)
```

::: diff_biophys.nmr.karplus

---

## Cα Chemical Shifts

`diff_biophys.nmr.chemical_shifts` predicts **Cα chemical shifts** (ppm) from
backbone torsion angles using a softmax-weighted Gaussian secondary-structure
detector in (φ, ψ) space.

Secondary chemical shifts (relative to random-coil):

| Structure | Δδ(Cα) |
|---|---|
| α-helix | +3.1 ppm |
| β-sheet | −1.5 ppm |
| Random coil | 0 ppm |

The library provides `RANDOM_COIL_CA` — a dictionary of per-residue random-coil
reference values.

```python
from diff_biophys.nmr.chemical_shifts import predict_ca_shifts, RANDOM_COIL_CA
import jax, jax.numpy as jnp

n_res = 10
phi = jnp.full((n_res,), jnp.deg2rad(-57.0))   # α-helix
psi = jnp.full((n_res,), jnp.deg2rad(-47.0))
rc  = jnp.full((n_res,), RANDOM_COIL_CA["ALA"])

shifts = predict_ca_shifts(phi, psi, rc)   # (n_res,) in ppm

# Refine φ to match target shifts
target = shifts + 0.5   # perturb target
loss = lambda p: jnp.mean((predict_ca_shifts(p, psi, rc) - target) ** 2)
grad_phi = jax.grad(loss)(phi)
```

::: diff_biophys.nmr.chemical_shifts

---

## Residual Dipolar Couplings (RDCs)

`diff_biophys.nmr.rdc` computes **Residual Dipolar Couplings** (Hz) from bond
vectors and a Saupe alignment tensor:

$$D_{NH} = D_{\max} \sum_{i,j} v_i S_{ij} v_j$$

where $\mathbf{v}$ is the unit bond vector and $\mathbf{S}$ is the Saupe tensor
(3×3 symmetric traceless matrix).  At the **magic angle** ($\theta \approx 54.74°$
from the alignment axis), $D = 0$.

Also provides:

- `fit_saupe_tensor` — SVD-based least-squares fitting of **S** from observed RDCs
- `calculate_rdc` — axially-symmetric simplified form ($D \propto 3\cos^2\theta - 1$)
- `calculate_q_factor` — R-factor for RDC quality assessment

```python
from diff_biophys.nmr.rdc import calculate_rdc_from_tensor, fit_saupe_tensor
import jax.numpy as jnp

# N–H bond unit vectors  (n_res, 3)
bond_vecs = jnp.array(...)

# Saupe alignment tensor (3, 3) — axially symmetric
S = jnp.diag(jnp.array([-0.05, -0.05, 0.10]))

rdcs = calculate_rdc_from_tensor(bond_vecs, S, d_max=21585.0)   # Hz

# Fit tensor from experimental RDCs
S_fit = fit_saupe_tensor(bond_vecs, rdcs_experimental)
```

::: diff_biophys.nmr.rdc

---

## Ring Current Shifts

`diff_biophys.nmr.ring_currents` implements the **Johnson-Bovey model** of
aromatic ring current shielding.  Protons directly above the ring plane are
**shielded** (upfield shift, negative Δδ); protons in the plane are
**deshielded** (downfield shift, positive Δδ).

The shift falls off as $\sim 1/r^3$ with distance from the ring centre.

```python
from diff_biophys.nmr.ring_currents import calculate_ring_current_shift
import jax, jax.numpy as jnp

# Proton position relative to ring centre
proton_pos = jnp.array([0.0, 0.0, 3.5])   # 3.5 Å above ring
ring_pos   = jnp.array([0.0, 0.0, 0.0])
ring_normal = jnp.array([0.0, 0.0, 1.0])   # ring in xy-plane

delta = calculate_ring_current_shift(proton_pos, ring_pos, ring_normal)
# delta < 0  (shielded above the ring)

grad = jax.grad(calculate_ring_current_shift)(proton_pos, ring_pos, ring_normal)
# tells you: move the proton in which direction to maximise shielding
```

::: diff_biophys.nmr.ring_currents

---

## Fixed-Tensor RDC Loss

### `FixedTensorRDCLoss`

`diff_integrator.terms.nmr.FixedTensorRDCLoss`

A `LossTerm` that computes the RDC loss while keeping the Saupe alignment tensor frozen during backpropagation via `jax.lax.stop_gradient`. The tensor is re-fitted from current coordinates every `update_interval` epochs, preventing the degeneracy exploit where gradient descent drives Q→0 unphysically by distorting the tensor rather than the structure.

**Constructor**

```python
FixedTensorRDCLoss(
    loss_fn,
    tensor_fn,
    update_interval: int = 50,
    n_rdcs: int | None = None,
    val_q_eval_fn: Callable | None = None,
)
```

| Parameter | Type | Description |
|---|---|---|
| `loss_fn` | `Callable` | Training-set RDC loss function `(bond_vecs, S) → scalar` returned by `make_rdc_cv_refinement_fns` |
| `tensor_fn` | `Callable` | Tensor-fitting function `(bond_vecs, rdcs) → S` returned by `make_rdc_cv_refinement_fns` |
| `update_interval` | `int` | Re-fit the Saupe tensor every this many epochs (default `50`) |
| `n_rdcs` | `int \| None` | Number of training RDCs; used by `suggested_weight()` |
| `val_q_eval_fn` | `Callable \| None` | Held-out Q-factor evaluator returned by `make_rdc_cv_refinement_fns`; enables `evaluate_validation_q()` |

**Methods**

#### `maybe_update_tensor(coords, epoch)`

Re-fits the Saupe tensor from the current Cartesian coordinates if `epoch % update_interval == 0`. Should be called at the top of each training step before the gradient computation.

```python
rdc_term.maybe_update_tensor(coords, epoch=epoch)
```

#### `suggested_weight(base_weight=1.0)`

Returns a weight scaled by the overdetermination ratio relative to an ideal 10× ratio:

$$w = \text{base\_weight} \times \frac{n\_rdcs / 10}{5}$$

Use this to auto-scale the RDC term so that systems with fewer RDCs are not over-penalised.

```python
rdc_weight = rdc_term.suggested_weight(base_weight=1.0)
```

#### `evaluate_validation_q(coords)`

Returns the Q-factor on the held-out cross-validation split, or `None` if `val_q_eval_fn` was not provided.

```python
val_q = rdc_term.evaluate_validation_q(coords)
```

---

### `make_rdc_cv_refinement_fns`

`diff_integrator.terms.nmr.make_rdc_cv_refinement_fns`

Factory function that partitions experimental RDCs into a training set and a held-out cross-validation set, then returns pre-built closures ready for `FixedTensorRDCLoss`.

**Signature**

```python
make_rdc_cv_refinement_fns(
    rdc_res_ids: ArrayLike,
    exp_rdcs: ArrayLike,
    struct_res_ids: ArrayLike,
    cv_fraction: float = 0.2,
) -> tuple[Callable, Callable, Callable, Callable, int, int]
```

| Parameter | Type | Description |
|---|---|---|
| `rdc_res_ids` | `ArrayLike` | Residue IDs corresponding to each experimental RDC value |
| `exp_rdcs` | `ArrayLike` | Experimental RDC values (Hz) |
| `struct_res_ids` | `ArrayLike` | Residue IDs present in the structure (used for index mapping) |
| `cv_fraction` | `float` | Fraction of RDCs to hold out for cross-validation (default `0.2`) |

**Returns** `(loss_fn, q_eval_fn, tensor_fn, val_q_fn, n_train, n_val)`

| Return value | Description |
|---|---|
| `loss_fn` | Training-set loss closure `(bond_vecs, S) → scalar` |
| `q_eval_fn` | Training-set Q-factor evaluator `(bond_vecs, S) → float` |
| `tensor_fn` | Tensor-fitting closure `(bond_vecs) → S` |
| `val_q_fn` | Validation Q-factor evaluator `(bond_vecs, S) → float` |
| `n_train` | Number of training RDCs |
| `n_val` | Number of held-out validation RDCs |

---

## EarlyStopping

### `EarlyStopping`

`diff_integrator.optimizer.EarlyStopping`

A dataclass passed as `early_stopping=` to `IntegrativeRefiner.run()` that halts training when a monitored loss term stops improving.

**Fields**

| Field | Type | Default | Description |
|---|---|---|---|
| `term_index` | `int` | — | Index into `JointLoss.terms` of the term to monitor (unweighted value) |
| `patience` | `int` | — | Number of epochs with no improvement before stopping |
| `min_delta` | `float` | `1e-5` | Minimum change in monitored value to count as an improvement |
| `mode` | `str` | `"min"` | `"min"` for loss-type metrics (lower is better); `"max"` for score-type metrics (higher is better) |

**Behaviour**

- The **unweighted** value of the monitored term (not the contribution to total loss) is tracked.
- When no improvement exceeding `min_delta` occurs for `patience` consecutive epochs, training stops and the best-checkpoint parameters are returned.
- The stopping event is recorded in `RefinementResult.stopped_early` and `RefinementResult.early_stopping_triggered_by`.

**Example**

```python
from diff_integrator.optimizer import EarlyStopping, IntegrativeRefiner

refiner = IntegrativeRefiner(loss_fn=joint_loss)
result = refiner.run(
    init_params=starting_coords,
    epochs=2000,
    learning_rate=0.005,
    early_stopping=EarlyStopping(
        term_index=1,      # monitor RDC term (index 1 in JointLoss)
        patience=50,
        min_delta=1e-4,
        mode="min",
    ),
)
print(f"Stopped at epoch {result.best_epoch} / 2000")
print(f"Triggered by term {result.early_stopping_triggered_by}")
```

---

## NOE Distance Restraints

### `NOELoss`

`diff_integrator.terms.noe.NOELoss`

A `LossTerm` implementing the standard flat-bottomed harmonic NOE (Nuclear Overhauser
Effect) distance restraint used in XPLOR, CNS, and ARIA.  The energy is zero when a
distance is within bounds and grows quadratically once a bound is violated:

$$E(d) = \frac{k}{M} \sum_{m=1}^{M} \left[ \max(0,\, d_m - d_m^{\text{upper}})^2 + \max(0,\, d_m^{\text{lower}} - d_m)^2 \right]$$

where $M$ is the number of restraints and $k$ is `force_const`.  A mean (not sum) is
used so the weight has a consistent interpretation across datasets of different sizes.

**Constructor**

```python
NOELoss(
    atom_pairs:  jnp.ndarray,             # (M, 2) integer atom index pairs
    d_upper:     jnp.ndarray,             # (M,)  upper-bound distances in Å
    d_lower:     jnp.ndarray | None = None,  # (M,) optional lower bounds
    force_const: float = 1.0,
)
```

| Parameter | Type | Description |
|---|---|---|
| `atom_pairs` | `(M, 2) int` | Each row `[i, j]` defines one restraint between atoms `i` and `j` |
| `d_upper` | `(M,) float` | Upper-bound distances in Å — penalty fires when `d > d_upper` |
| `d_lower` | `(M,) float \| None` | Lower-bound distances in Å — penalty fires when `d < d_lower`.  `None` (default) means upper-bound only, which is the standard NMR convention |
| `force_const` | `float` | Harmonic force constant.  Default `1.0`.  Typical values: `5`–`50` |

**`name`** attribute: `"noe"`

**Methods**

#### `count_violations(coords) → dict[str, int]`

Returns `{"upper": n, "lower": n, "total": n}` — count of atoms violating each bound.
Pure diagnostic; not used in the gradient.

#### `rms_violation(coords) → float`

Root-mean-square distance violation across all restraints, in Å.

**Property**

#### `n_restraints → int`

Number of distance restraints.

**Example**

```python
from diff_integrator.terms.noe import NOELoss

noe_loss = NOELoss(
    atom_pairs  = jnp.array([[4, 31], [7, 52]]),   # Cα pairs
    d_upper     = jnp.array([6.0, 4.5]),            # Å
    d_lower     = jnp.array([1.8, 1.8]),            # Å (optional)
    force_const = 10.0,
)
```

---

### `make_noe_restraints`

`diff_integrator.terms.noe.make_noe_restraints`

Factory that maps `(res_id, atom_name)` observations to flat atom indices using the
structure's residue ordering, then returns a ready-to-use `NOELoss`.

**Signature**

```python
make_noe_restraints(
    noe_list:   list[dict],
    res_ids:    np.ndarray,
    atom_names: list[str] | None = None,   # default ["N", "CA", "C"]
    force_const: float = 1.0,
) -> NOELoss
```

Each dict in `noe_list` must contain:

| Key | Type | Description |
|---|---|---|
| `"res_i"` | `int` | Residue number of atom i |
| `"atom_i"` | `str` | Atom name of atom i (e.g. `"CA"`) |
| `"res_j"` | `int` | Residue number of atom j |
| `"atom_j"` | `str` | Atom name of atom j |
| `"d_upper"` | `float` | Upper-bound distance in Å |
| `"d_lower"` | `float` | *(optional)* Lower-bound distance in Å |

**Example**

```python
from diff_integrator.terms.noe import make_noe_restraints
import numpy as np

noe_observations = [
    {"res_i":  5, "atom_i": "CA", "res_j": 20, "atom_j": "CA",
     "d_upper": 6.0, "d_lower": 1.8},
    {"res_i": 12, "atom_i": "N",  "res_j": 45, "atom_j": "CA",
     "d_upper": 5.5},
]

noe_term = make_noe_restraints(noe_observations, struct_res_ids)
violations = noe_term.count_violations(coords)
print(f"{violations['total']} NOE violations at current coordinates")
```

