# 🚀 Optimizer API

The `diff_integrator.optimizer` module provides tools for minimizing integrative structural biology loss functions using Optax.

---

## `IntegrativeRefiner`

`diff_integrator.optimizer.IntegrativeRefiner`

Refines atomic coordinates by minimizing a `JointLoss` using Optax.

### Constructor

```python
IntegrativeRefiner(loss_fn: JointLoss)
```

### `run()`

Runs the refinement optimization.

```python
def run(
    self,
    init_params: Any,
    epochs: int = 1000,
    learning_rate: float = 0.01,
    optimizer: optax.GradientTransformation | None = None,
    kinematics_fn: Callable[[Any], jnp.ndarray] | None = None,
    patience: int = 0,
    min_delta: float = 1e-5,
    validation_loss: JointLoss | None = None,
    log_interval: int = 1,
    weight_schedules: dict[int, Callable[[int], float]] | None = None,
    early_stopping: "EarlyStopping | list[EarlyStopping] | None" = None,
    per_epoch_callbacks: "list[Callable] | None" = None,
) -> RefinementResult:
```

| Parameter | Description |
|---|---|
| `init_params` | Starting parameters for optimization. |
| `epochs` | Number of optimization steps. |
| `learning_rate` | Step size for the optimizer (if default Adam is used). |
| `optimizer` | A custom Optax optimizer. If None, `optax.chain(clip_by_global_norm(1.0), adam)` is used. |
| `kinematics_fn` | A function that maps `params` to Cartesian coordinates. Defaults to identity. |
| `patience` | If > 0, stops when no improvement in the total training loss for this many epochs. |
| `weight_schedules` | Mapping from term index to a callable `(epoch: int) -> float` to dynamically update term weights. |
| `early_stopping` | One or more `EarlyStopping` configurations for per-term stopping. |
| `per_epoch_callbacks` | Optional list of callables invoked before each gradient step. |

Returns a `RefinementResult` containing the final parameters, best parameters, loss history, and early-stopping diagnostics.

---

## `EarlyStopping`

`diff_integrator.optimizer.EarlyStopping`

Configuration for per-term early stopping.

Monitors a single `JointLoss` term and requests refinement to stop when that term's **unweighted** value fails to improve by at least `min_delta` for `patience` consecutive epochs.

### Constructor

```python
EarlyStopping(
    term_index: int,
    patience: int,
    min_delta: float = 1e-5,
    mode: str = "min",
)
```

| Parameter | Description |
|---|---|
| `term_index` | Zero-based index of the term to watch in the `JointLoss`. |
| `patience` | Number of consecutive epochs without an improvement. |
| `min_delta` | Minimum change to qualify as an improvement. |
| `mode` | `"min"` (stop when stops decreasing) or `"max"` (stop when stops increasing). |

---

## `RefinementResult`

`diff_integrator.optimizer.RefinementResult`

Results from an `IntegrativeRefiner` optimization run.

| Field | Description |
|---|---|
| `final_params` | The parameters at the last epoch run. |
| `best_params` | The parameters at the epoch with the lowest loss. |
| `best_epoch` | The epoch (0-based) where `best_params` were obtained. |
| `loss_history` | A list of total loss values per epoch. |
| `per_term_history` | A dictionary mapping term names to lists of their values per epoch. |
| `stopped_early` | `True` if early stopping was triggered. |
| `early_stopping_triggered_by` | String explanation of which criterion caused the stop. |
