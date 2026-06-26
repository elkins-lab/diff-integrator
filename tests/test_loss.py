from typing import Any

import jax.numpy as jnp
from jax import grad

from diff_integrator.loss import JointLoss, LossTerm


class DummyLossA(LossTerm):
    def __call__(self, params: Any, coords: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(coords**2)


class DummyLossB(LossTerm):
    def __call__(self, params: Any, coords: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(coords)


def test_joint_loss():
    loss_a = DummyLossA()
    loss_b = DummyLossB()

    # Joint loss with weights
    joint_loss = JointLoss(terms=[(loss_a, 2.0), (loss_b, 1.0)])

    coords = jnp.array([1.0, 2.0, 3.0])

    # Loss A = 1^2 + 2^2 + 3^2 = 14
    # Loss B = 1 + 2 + 3 = 6
    # Total = 2.0 * 14 + 1.0 * 6 = 28 + 6 = 34

    val = joint_loss(None, coords)
    assert jnp.isclose(val, 34.0)


def test_joint_loss_gradient():
    loss_a = DummyLossA()
    loss_b = DummyLossB()

    joint_loss = JointLoss(terms=[(loss_a, 2.0), (loss_b, 1.0)])

    coords = jnp.array([1.0, 2.0, 3.0])

    # f(x) = 2.0 * sum(x^2) + 1.0 * sum(x)
    # df/dx = 4.0 * x + 1.0

    expected_grad = 4.0 * coords + 1.0

    grad_fn = grad(lambda c: joint_loss(None, c))
    actual_grad = grad_fn(coords)

    assert jnp.allclose(actual_grad, expected_grad)


def test_evaluate_terms():
    loss_a = DummyLossA()
    loss_a.name = "loss_a"
    loss_b = DummyLossB()
    loss_b.name = "loss_b"

    joint_loss = JointLoss(terms=[(loss_a, 2.0), (loss_b, 1.0)])
    coords = jnp.array([1.0, 2.0, 3.0])

    result = joint_loss.evaluate_terms(None, coords)
    assert "loss_a" in result
    assert "loss_b" in result
    assert result["loss_a"] == 2.0 * 14.0  # 2.0 * sum([1,4,9])
    assert result["loss_b"] == 1.0 * 6.0   # 1.0 * sum([1,2,3])


def test_evaluate_terms_unnamed():
    loss_a = DummyLossA()
    loss_b = DummyLossB()

    joint_loss = JointLoss(terms=[(loss_a, 1.0), (loss_b, 1.0)])
    coords = jnp.array([1.0, 2.0, 3.0])

    result = joint_loss.evaluate_terms(None, coords)
    assert "term_0" in result
    assert "term_1" in result
