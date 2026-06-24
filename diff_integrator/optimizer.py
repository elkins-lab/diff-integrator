from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import optax
from jax import jit, value_and_grad

from diff_integrator.loss import JointLoss


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
    ) -> tuple[Any, list[float]]:
        """
        Run the refinement optimization.

        Args:
            init_params: Starting parameters for optimization (e.g. coordinates or angles).
            epochs: Number of optimization steps.
            learning_rate: Step size for the optimizer (if default Adam is used).
            optimizer: A custom Optax optimizer. If None, optax.adam is used.
            kinematics_fn: A function that maps `params` to Cartesian coordinates `(N, 3)`.
                If None, the identity function is used (assumes params ARE the coordinates).

        Returns:
            A tuple of (final_params, loss_history).
        """
        if optimizer is None:
            optimizer = optax.adam(learning_rate)

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
            updates, new_opt_state = optimizer.update(grads, current_opt_state)
            new_params = optax.apply_updates(current_params, updates)
            return new_params, new_opt_state, loss_val

        loss_history = []
        for _ in range(epochs):
            params, opt_state, loss_val = step(params, opt_state)
            loss_history.append(float(loss_val))

        return params, loss_history
