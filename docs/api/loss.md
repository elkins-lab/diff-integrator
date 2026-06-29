# 📉 Loss & Core API

The `diff_integrator.loss` module provides the core abstractions for building differentiable integrative refinement objectives.

---

## `LossTerm`

`diff_integrator.loss.LossTerm`

Abstract base class for all differentiable loss terms.

All experimental and geometric restraints (like NOE distance bounds, RDCs, chemical shifts, backbone geometry) inherit from `LossTerm`.

### Abstract Methods

#### `__call__(params, coords) -> jnp.ndarray`

Evaluates the loss term.

| Parameter | Type | Description |
|---|---|---|
| `params` | `Any` | The parameters being optimized (e.g., internal angles or Cartesian coordinates). |
| `coords` | `jnp.ndarray` | `(N, 3)` atomic Cartesian coordinates resulting from the kinematics function. |

Returns a scalar `jnp.ndarray` representing the loss.

---

## `JointLoss`

`diff_integrator.loss.JointLoss`

Combines multiple `LossTerm` objects with scalar weights to create the single objective function minimized by `IntegrativeRefiner`.

Supports multi-phase refinement workflows (e.g., geometry-only for 200 epochs, then adding experimental terms) via its term-freezing API.

### Constructor

```python
JointLoss(terms: list[tuple[LossTerm, float]])
```

| Parameter | Type | Description |
|---|---|---|
| `terms` | `list[tuple[LossTerm, float]]` | A list of tuples containing `(LossTerm, weight)`. |

### Methods

#### `__call__(params, coords) -> jnp.ndarray`
Evaluates the total weighted loss, skipping any frozen terms.

#### `set_weight(term_index, weight)`
Updates the weight of a single term in-place. Useful for dynamic weight schedules.

#### `freeze_term(term_index)`
Excludes a term from the gradient objective while keeping it visible to `evaluate_terms` for diagnostic monitoring.

#### `unfreeze_term(term_index)`
Re-enables a previously frozen term.

#### `is_frozen(term_index) -> bool`
Returns `True` if the term at `term_index` is currently frozen.

#### `evaluate_terms(params, coords, unweighted=False) -> dict[str, float]`
Evaluates each term individually and returns a dictionary mapping the term's name to its value. Useful for logging.
- `unweighted`: If `True`, returns the raw, unweighted term value. If `False` (default), returns the weighted value.
