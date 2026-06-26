from typing import Any

import jax.numpy as jnp

from diff_integrator.loss import JointLoss, LossTerm
from diff_integrator.optimizer import IntegrativeRefiner, RefinementResult


class DummyTargetLoss(LossTerm):
    def __init__(self, target_coords: jnp.ndarray):
        self.target_coords = target_coords

    def __call__(self, params: Any, coords: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum((coords - self.target_coords) ** 2)


class DummySumLoss(LossTerm):
    def __call__(self, params: Any, coords: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(coords)


def test_optimizer_convergence():
    target_coords = jnp.array(
        [[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]
    )
    init_coords = jnp.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )

    loss_fn = JointLoss([(DummyTargetLoss(target_coords), 1.0)])
    refiner = IntegrativeRefiner(loss_fn=loss_fn)

    result = refiner.run(
        init_params=init_coords, epochs=100, learning_rate=0.1
    )

    assert isinstance(result, RefinementResult)
    assert len(result.loss_history) == 100
    assert result.loss_history[-1] < result.loss_history[0]
    assert jnp.allclose(result.final_params, target_coords, atol=1e-2)
    assert result.epochs_run == 100
    assert result.stopped_early is False


def test_optimizer_per_term_history():
    target_coords = jnp.array(
        [[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]
    )
    init_coords = jnp.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )

    term_a = DummyTargetLoss(target_coords)
    term_a.name = "target_loss"
    term_b = DummySumLoss()
    term_b.name = "sum_loss"

    loss_fn = JointLoss([(term_a, 1.0), (term_b, 0.5)])
    refiner = IntegrativeRefiner(loss_fn=loss_fn)

    result = refiner.run(
        init_params=init_coords, epochs=50, learning_rate=0.1
    )

    assert "target_loss" in result.per_term_history
    assert "sum_loss" in result.per_term_history
    assert len(result.per_term_history["target_loss"]) == 50
    assert len(result.per_term_history["sum_loss"]) == 50


def test_optimizer_early_stopping():
    target_coords = jnp.array(
        [[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]
    )
    init_coords = jnp.array(
        [[0.99, 0.99, 0.99], [-0.99, -0.99, -0.99]]
    )

    loss_fn = JointLoss([(DummyTargetLoss(target_coords), 1.0)])
    refiner = IntegrativeRefiner(loss_fn=loss_fn)

    epochs = 5000
    result = refiner.run(
        init_params=init_coords,
        epochs=epochs,
        learning_rate=0.1,
        patience=10,
        min_delta=1e-5,
    )

    assert result.stopped_early is True
    assert result.epochs_run < epochs


def test_optimizer_validation_loss():
    target_coords = jnp.array(
        [[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]
    )
    init_coords = jnp.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )

    loss_fn = JointLoss([(DummyTargetLoss(target_coords), 1.0)])

    val_target = jnp.array(
        [[1.1, 1.1, 1.1], [-1.1, -1.1, -1.1]]
    )
    val_loss = JointLoss([(DummyTargetLoss(val_target), 1.0)])

    refiner = IntegrativeRefiner(loss_fn=loss_fn)

    result = refiner.run(
        init_params=init_coords,
        epochs=50,
        learning_rate=0.1,
        validation_loss=val_loss,
    )

    assert len(result.validation_history) == 50
    assert all(isinstance(v, float) for v in result.validation_history)


def test_optimizer_gradient_clipping():
    """Verify default optimizer (with gradient clipping) converges."""
    target_coords = jnp.array(
        [[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]
    )
    init_coords = jnp.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )

    loss_fn = JointLoss([(DummyTargetLoss(target_coords), 1.0)])
    refiner = IntegrativeRefiner(loss_fn=loss_fn)

    result = refiner.run(
        init_params=init_coords, epochs=200, learning_rate=0.1
    )

    assert result.loss_history[-1] < result.loss_history[0]
    assert jnp.allclose(result.final_params, target_coords, atol=1e-1)
