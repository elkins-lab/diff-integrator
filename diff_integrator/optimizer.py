from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
import optax
from jax import jit, value_and_grad

from diff_integrator.loss import JointLoss


@dataclass
class RefinementResult:
    """Results from an IntegrativeRefiner optimization run."""

    final_params: Any
    loss_history: list[float]
    per_term_history: dict[str, list[float]] = field(default_factory=dict)
    validation_history: list[float] = field(default_factory=list)
    weight_history: dict[int, list[float]] = field(default_factory=dict)
    best_params: Any = None
    best_epoch: int = 0
    epochs_run: int = 0
    stopped_early: bool = False


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
            patience: If > 0, stop when no improvement for this many
                epochs. 0 means disabled.
            min_delta: Minimum change to qualify as an improvement.
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
        epochs_run = 0

        for epoch in range(epochs):
            # Apply weight schedules, then pass current weights to JIT step
            if weight_schedules:
                for term_idx, schedule_fn in weight_schedules.items():
                    new_weight = schedule_fn(epoch)
                    self.loss_fn.set_weight(term_idx, new_weight)
                    weight_history[term_idx].append(new_weight)

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

            # Early stopping
            if patience > 0:
                if current_loss < best_loss_for_patience - min_delta:
                    best_loss_for_patience = current_loss
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1
                if epochs_since_improvement >= patience:
                    stopped_early = True
                    break

        return RefinementResult(
            final_params=params,
            loss_history=loss_history,
            per_term_history=per_term_history,
            validation_history=validation_history,
            weight_history=weight_history,
            best_params=best_params,
            best_epoch=best_epoch,
            epochs_run=epochs_run,
            stopped_early=stopped_early,
        )
