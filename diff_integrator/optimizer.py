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

        Returns:
            A RefinementResult with all tracked data.
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

        # Define the step function
        @jit
        def step(
            current_params: Any, current_opt_state: optax.OptState
        ) -> tuple[Any, optax.OptState, jnp.ndarray]:
            def objective(p: Any) -> jnp.ndarray:
                coords = kinematics_fn(p)
                return self.loss_fn(p, coords)

            loss_val, grads = value_and_grad(objective)(current_params)
            updates, new_opt_state = optimizer.update(
                grads, current_opt_state
            )
            new_params = optax.apply_updates(current_params, updates)
            return new_params, new_opt_state, loss_val

        loss_history: list[float] = []
        per_term_history: dict[str, list[float]] = {}
        validation_history: list[float] = []
        best_loss = float("inf")
        epochs_since_improvement = 0
        stopped_early = False
        epochs_run = 0

        for epoch in range(epochs):
            params, opt_state, loss_val = step(params, opt_state)
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

            # Early stopping
            if patience > 0:
                if current_loss < best_loss - min_delta:
                    best_loss = current_loss
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
            epochs_run=epochs_run,
            stopped_early=stopped_early,
        )
