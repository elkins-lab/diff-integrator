from typing import Any

import jax.numpy as jnp

from diff_integrator.loss import JointLoss, LossTerm
from diff_integrator.optimizer import IntegrativeRefiner


class DummyTargetLoss(LossTerm):
    def __init__(self, target_coords: jnp.ndarray):
        self.target_coords = target_coords

    def __call__(self, params: Any, coords: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum((coords - self.target_coords) ** 2)


def test_optimizer_convergence():
    target_coords = jnp.array([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])

    init_coords = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    loss_fn = JointLoss([(DummyTargetLoss(target_coords), 1.0)])

    refiner = IntegrativeRefiner(loss_fn=loss_fn)

    final_coords, loss_history = refiner.run(init_params=init_coords, epochs=100, learning_rate=0.1)

    assert len(loss_history) == 100
    assert loss_history[-1] < loss_history[0]
    assert jnp.allclose(final_coords, target_coords, atol=1e-2)
