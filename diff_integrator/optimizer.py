from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
import optax
from jax import jit, value_and_grad

from diff_integrator.loss import JointLoss


@dataclass
class EarlyStopping:
    """Configuration for per-term early stopping.

    Monitors a single :class:`~diff_integrator.loss.JointLoss` term and
    requests refinement to stop when that term's **unweighted** value fails
    to improve by at least ``min_delta`` for ``patience`` consecutive epochs.

    Using the unweighted value means the threshold is expressed in the term's
    natural units (e.g. ppm for a chemical-shift RMSD, Å² for a bond-length
    penalty) and is independent of any weight schedule running on that term.

    Args:
        term_index: Zero-based index of the term to watch in the
            :class:`~diff_integrator.loss.JointLoss` term list.
        patience: Number of consecutive epochs without an improvement of at
            least ``min_delta`` before stopping is triggered.
        min_delta: Minimum decrease in the monitored value that counts as an
            improvement.  Must be ≥ 0.
        mode: ``"min"`` (default) — stop when the value stops *decreasing*.
            ``"max"`` — stop when the value stops *increasing* (useful for
            monitoring a score rather than a loss).

    Example::

        from diff_integrator.optimizer import EarlyStopping, IntegrativeRefiner

        result = IntegrativeRefiner(joint_loss).run(
            init_params=coords,
            epochs=2000,
            early_stopping=[
                EarlyStopping(term_index=1, patience=50, min_delta=1e-4),
            ],
        )
        print(f"Stopped at epoch {result.stopped_at_epoch}")
        print(f"Reason: {result.early_stopping_triggered_by}")
    """

    term_index: int
    patience: int
    min_delta: float = 1e-5
    mode: str = "min"

    def __post_init__(self) -> None:
        if self.patience <= 0:
            raise ValueError(f"EarlyStopping.patience must be > 0, got {self.patience}")
        if self.min_delta < 0:
            raise ValueError(
                f"EarlyStopping.min_delta must be >= 0, got {self.min_delta}"
            )
        if self.mode not in ("min", "max"):
            raise ValueError(
                f"EarlyStopping.mode must be 'min' or 'max', got {self.mode!r}"
            )


@dataclass
class RefinementResult:
    """Results from an IntegrativeRefiner optimization run."""

    final_params: Any
    loss_history: list[float]
    per_term_history: dict[str, list[float]] = field(default_factory=dict)
    per_term_epochs: list[int] = field(default_factory=list)
    validation_history: list[float] = field(default_factory=list)
    weight_history: dict[int, list[float]] = field(default_factory=dict)
    best_params: Any = None
    best_epoch: int = 0
    epochs_run: int = 0
    stopped_early: bool = False
    stopped_at_epoch: int = -1
    early_stopping_triggered_by: str = ""


class IntegrativeRefiner:
    """
    Refines atomic coordinates by minimizing a JointLoss using Optax.
    """

    def __init__(self, loss_fn: JointLoss):
        """
        Args:
            loss_fn: The JointLoss objective to minimize.
        """
        self.loss_fn = loss_fn

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
        per_epoch_callbacks: "list[Callable[[int, Any, jnp.ndarray], None]] | None" = None,
    ) -> RefinementResult:
        """
        Run the refinement optimization.

        Args:
            init_params: Starting parameters for optimization
                (e.g. coordinates or angles).
            epochs: Number of optimization steps.
            learning_rate: Step size for the optimizer
                (if default Adam is used).
            optimizer: A custom Optax optimizer. If None,
                optax.chain(clip_by_global_norm(1.0), adam) is used.
            kinematics_fn: A function that maps `params` to Cartesian
                coordinates `(N, 3)`. If None, the identity function
                is used (assumes params ARE the coordinates).
            patience: If > 0, stop when no improvement in the **total
                training loss** for this many epochs.  0 means disabled.
                See ``early_stopping`` for per-term monitoring.
            min_delta: Minimum change to qualify as an improvement for the
                global ``patience`` criterion.
            validation_loss: A JointLoss evaluated each epoch for
                diagnostics but not used in the gradient.
            log_interval: How often to evaluate per-term diagnostics.
            weight_schedules: Optional mapping from term index to a
                callable ``(epoch: int) -> float``.  Before each gradient
                step the corresponding ``JointLoss`` term weight is updated
                to ``schedule_fn(epoch)``.  Use
                :class:`~diff_integrator.schedules.ExponentialDecaySchedule`
                for the annealed geometry restraint pattern.  The weight
                values applied each epoch are recorded in
                ``RefinementResult.weight_history``.
            early_stopping: One or more :class:`EarlyStopping` configurations
                for **per-term** stopping.  Each watches a single
                :class:`~diff_integrator.loss.JointLoss` term's *unweighted*
                value and fires when that term fails to improve by
                ``min_delta`` for ``patience`` consecutive epochs.  A single
                :class:`EarlyStopping` instance or a list are both accepted.
                Whichever criterion (global ``patience`` or any per-term
                rule) fires first terminates the run.  The result fields
                ``stopped_at_epoch`` and ``early_stopping_triggered_by``
                record which rule fired and at which epoch.
            per_epoch_callbacks: Optional list of callables invoked **before
                each gradient step** with signature
                ``(epoch: int, params: Any, coords: jnp.ndarray) -> None``.
                Use this to perform operations that must run outside the
                gradient tape every epoch, such as periodically re-fitting a
                Saupe tensor via
                :meth:`~diff_integrator.terms.nmr.FixedTensorRDCLoss.maybe_update_tensor`::

                    result = refiner.run(
                        ...,
                        per_epoch_callbacks=[
                            lambda epoch, p, c: rdc_term.maybe_update_tensor(c, epoch)
                        ],
                    )

                Coords are computed from ``params`` via ``kinematics_fn``
                once per epoch when callbacks are present; this incurs one
                extra forward pass per epoch regardless of ``log_interval``.

        Returns:
            A :class:`RefinementResult` with all tracked data.

            ``best_params`` holds the parameter values at the epoch with the
            lowest loss — measured by ``validation_loss`` if supplied, otherwise
            by the total training loss.  ``best_epoch`` records which epoch
            that was (0-based).  ``final_params`` always holds the last
            iterate, preserving backward compatibility.
        """
        if optimizer is None:
            optimizer = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adam(learning_rate),
            )

        if kinematics_fn is None:

            def default_kinematics(x: Any) -> jnp.ndarray:
                return jnp.asarray(x)

            kinematics_fn = default_kinematics

        # Initialize optimizer state
        opt_state = optimizer.init(init_params)
        params = init_params

        # Define the step function.
        #
        # The weights are passed as an explicit JAX array rather than being
        # captured as Python-float closure variables.  This is critical for
        # weight_schedules to work correctly: JAX @jit traces the function
        # once and caches the compiled XLA code keyed on input shapes/dtypes.
        # Python floats captured in the closure are baked in as compile-time
        # constants — changing them via set_weight would have NO effect on
        # the compiled computation.  By passing weights as a dynamic array
        # argument, JAX substitutes the current values each call without
        # recompilation, while value_and_grad still differentiates only
        # w.r.t. current_params (not the weights).
        terms = self.loss_fn.terms  # stable reference to the term list

        @jit
        def step(
            current_params: Any,
            current_opt_state: optax.OptState,
            current_weights: jnp.ndarray,
        ) -> tuple[Any, optax.OptState, jnp.ndarray]:
            def objective(p: Any) -> jnp.ndarray:
                coords = kinematics_fn(p)
                total = jnp.array(0.0)
                for i, (term, _) in enumerate(terms):
                    total = total + current_weights[i] * term(p, coords)
                return total

            loss_val, grads = value_and_grad(objective)(current_params)
            updates, new_opt_state = optimizer.update(
                grads, current_opt_state
            )
            new_params = optax.apply_updates(current_params, updates)
            return new_params, new_opt_state, loss_val

        loss_history: list[float] = []
        per_term_history: dict[str, list[float]] = {}
        per_term_epochs: list[int] = []
        validation_history: list[float] = []
        weight_history: dict[int, list[float]] = {
            idx: [] for idx in (weight_schedules or {})
        }
        # Best-checkpoint tracking: keyed on validation_loss when provided,
        # otherwise on total training loss.
        _best_metric = float("inf")
        best_params = init_params
        best_epoch = 0
        _track_by_validation = validation_loss is not None
        best_loss_for_patience = float("inf")
        epochs_since_improvement = 0
        stopped_early = False
        stopped_at_epoch = -1
        early_stopping_triggered_by = ""
        epochs_run = 0

        # Normalise early_stopping into a list of EarlyStopping configs.
        _es_configs: list[EarlyStopping]
        if early_stopping is None:
            _es_configs = []
        elif isinstance(early_stopping, EarlyStopping):
            _es_configs = [early_stopping]
        else:
            _es_configs = list(early_stopping)

        # Validate term indices early so callers get a clear error before
        # any computation begins rather than an IndexError mid-run.
        n_terms = len(self.loss_fn.terms)
        for _cfg in _es_configs:
            if _cfg.term_index < 0 or _cfg.term_index >= n_terms:
                raise IndexError(
                    f"EarlyStopping.term_index={_cfg.term_index} is out of range "
                    f"for a JointLoss with {n_terms} term(s)."
                )

        # Per-config patience counters and best-value trackers.
        # We track best_value in the natural direction (lower is better for
        # mode="min"; higher for mode="max").
        _es_best: list[float] = [
            float("inf") if cfg.mode == "min" else float("-inf")
            for cfg in _es_configs
        ]
        _es_counter: list[int] = [0] * len(_es_configs)

        for epoch in range(epochs):
            # Apply weight schedules, then pass current weights to JIT step
            if weight_schedules:
                for term_idx, schedule_fn in weight_schedules.items():
                    new_weight = schedule_fn(epoch)
                    self.loss_fn.set_weight(term_idx, new_weight)
                    weight_history[term_idx].append(new_weight)

            # Per-epoch callbacks (e.g. tensor re-fitting) — before gradient step.
            # Coords are computed from current params here so that callbacks such
            # as FixedTensorRDCLoss.maybe_update_tensor receive up-to-date
            # geometry without requiring a separate external loop.
            if per_epoch_callbacks:
                _cb_coords = kinematics_fn(params)
                for _cb in per_epoch_callbacks:
                    _cb(epoch, params, _cb_coords)

            current_weights = jnp.array([w for _, w in self.loss_fn.terms])
            params, opt_state, loss_val = step(params, opt_state, current_weights)
            current_loss = float(loss_val)
            loss_history.append(current_loss)
            epochs_run = epoch + 1

            # Per-term diagnostics (outside JIT)
            if epoch % log_interval == 0:
                coords = kinematics_fn(params)
                term_values = self.loss_fn.evaluate_terms(
                    params, coords
                )
                for name, value in term_values.items():
                    if name not in per_term_history:
                        per_term_history[name] = []
                    per_term_history[name].append(value)
                per_term_epochs.append(epoch)

            # Validation loss (outside JIT)
            if validation_loss is not None:
                coords = kinematics_fn(params)
                val_loss = float(
                    validation_loss(params, coords)
                )
                validation_history.append(val_loss)
                # Best-checkpoint keyed on validation loss
                if _track_by_validation and val_loss < _best_metric:
                    _best_metric = val_loss
                    best_params = params
                    best_epoch = epoch

            # Best-checkpoint keyed on total training loss (no validation_loss)
            if not _track_by_validation and current_loss < _best_metric:
                _best_metric = current_loss
                best_params = params
                best_epoch = epoch

            # Global early stopping (total training loss)
            if patience > 0:
                if current_loss < best_loss_for_patience - min_delta:
                    best_loss_for_patience = current_loss
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1
                if epochs_since_improvement >= patience:
                    stopped_early = True
                    stopped_at_epoch = epoch
                    early_stopping_triggered_by = (
                        f"global patience={patience} (total loss)"
                    )
                    break

            # Per-term early stopping
            if _es_configs:
                # Reuse coords already computed in the diagnostics block above
                # (or compute here if log_interval skipped this epoch).
                if epoch % log_interval != 0:
                    coords = kinematics_fn(params)
                _fire = False
                for _i, _cfg in enumerate(_es_configs):
                    _term_obj, _ = self.loss_fn.terms[_cfg.term_index]
                    _raw = float(_term_obj(params, coords))
                    if _cfg.mode == "min":
                        _improved = _raw < _es_best[_i] - _cfg.min_delta
                    else:
                        _improved = _raw > _es_best[_i] + _cfg.min_delta
                    if _improved:
                        _es_best[_i] = _raw
                        _es_counter[_i] = 0
                    else:
                        _es_counter[_i] += 1
                    if _es_counter[_i] >= _cfg.patience:
                        _term_name = _term_obj.name or f"term_{_cfg.term_index}"
                        stopped_early = True
                        stopped_at_epoch = epoch
                        early_stopping_triggered_by = (
                            f"term_{_cfg.term_index} ({_term_name}) "
                            f"patience={_cfg.patience}"
                        )
                        _fire = True
                        break
                if _fire:
                    break

        return RefinementResult(
            final_params=params,
            loss_history=loss_history,
            per_term_history=per_term_history,
            per_term_epochs=per_term_epochs,
            validation_history=validation_history,
            weight_history=weight_history,
            best_params=best_params,
            best_epoch=best_epoch,
            epochs_run=epochs_run,
            stopped_early=stopped_early,
            stopped_at_epoch=stopped_at_epoch,
            early_stopping_triggered_by=early_stopping_triggered_by,
        )
