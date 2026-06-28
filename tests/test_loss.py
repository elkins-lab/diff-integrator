from typing import Any

import jax.numpy as jnp
import pytest
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


def test_joint_loss_set_weight():
    """set_weight updates the effective contribution of a term."""
    loss_a = DummyLossA()
    loss_b = DummyLossB()
    joint_loss = JointLoss(terms=[(loss_a, 2.0), (loss_b, 1.0)])
    coords = jnp.array([1.0, 2.0, 3.0])

    # Before: 2.0 * 14 + 1.0 * 6 = 34
    assert jnp.isclose(joint_loss(None, coords), 34.0)

    # Change weight of term 0 from 2.0 → 0.0 — term_a should contribute nothing
    joint_loss.set_weight(0, 0.0)
    assert jnp.isclose(joint_loss(None, coords), 6.0)  # only term_b

    # Change weight of term 1 from 1.0 → 3.0
    joint_loss.set_weight(1, 3.0)
    assert jnp.isclose(joint_loss(None, coords), 18.0)  # 0 + 3.0 * 6


def test_joint_loss_set_weight_out_of_range():
    """set_weight raises IndexError for invalid term index."""
    import pytest
    joint_loss = JointLoss(terms=[(DummyLossA(), 1.0)])
    with pytest.raises(IndexError):
        joint_loss.set_weight(1, 2.0)  # only index 0 exists
    with pytest.raises(IndexError):
        joint_loss.set_weight(-1, 2.0)


# ---------------------------------------------------------------------------
# evaluate_terms unweighted
# ---------------------------------------------------------------------------


def test_evaluate_terms_unweighted_false_returns_weighted():
    """Default (unweighted=False) returns weight * raw value."""
    loss_a = DummyLossA()
    loss_a.name = "loss_a"
    joint_loss = JointLoss(terms=[(loss_a, 3.0)])
    coords = jnp.array([1.0, 2.0, 3.0])   # sum(x^2) = 14
    result = joint_loss.evaluate_terms(None, coords)
    assert result["loss_a"] == pytest.approx(3.0 * 14.0)


def test_evaluate_terms_unweighted_true_returns_raw():
    """unweighted=True returns the raw term value, independent of weight."""
    loss_a = DummyLossA()
    loss_a.name = "loss_a"
    joint_loss = JointLoss(terms=[(loss_a, 99.0)])  # high weight should NOT affect output
    coords = jnp.array([1.0, 2.0, 3.0])   # sum(x^2) = 14
    result = joint_loss.evaluate_terms(None, coords, unweighted=True)
    assert result["loss_a"] == pytest.approx(14.0)


def test_evaluate_terms_unweighted_independent_of_weight_change():
    """After set_weight the unweighted result must not change."""
    import pytest  # noqa: PLC0415
    loss_a = DummyLossA()
    loss_a.name = "a"
    joint_loss = JointLoss(terms=[(loss_a, 1.0)])
    coords = jnp.array([2.0, 0.0, 0.0])   # raw = 4.0

    raw_before = joint_loss.evaluate_terms(None, coords, unweighted=True)["a"]
    joint_loss.set_weight(0, 50.0)
    raw_after = joint_loss.evaluate_terms(None, coords, unweighted=True)["a"]

    assert raw_before == pytest.approx(raw_after)


# ---------------------------------------------------------------------------
# freeze_term / unfreeze_term / is_frozen
# ---------------------------------------------------------------------------


def test_freeze_term_removes_contribution():
    """Frozen term should contribute zero to __call__."""
    loss_a = DummyLossA()
    loss_a.name = "a"
    loss_b = DummyLossB()
    loss_b.name = "b"
    joint_loss = JointLoss(terms=[(loss_a, 1.0), (loss_b, 1.0)])
    coords = jnp.array([2.0, 0.0, 0.0])

    full_loss = float(joint_loss(None, coords))   # a + b contribution

    joint_loss.freeze_term(0)                     # freeze term a
    frozen_loss = float(joint_loss(None, coords)) # only b should contribute

    assert joint_loss.is_frozen(0)
    assert not joint_loss.is_frozen(1)
    # Frozen term a contributed 4.0 (sum(coords^2)=4); only b remains.
    # DummyLossB returns sum(coords) = 2.0.
    assert frozen_loss < full_loss
    assert frozen_loss == pytest.approx(2.0, rel=1e-5)


def test_unfreeze_term_restores_contribution():
    """Unfreezing a term should restore it to the gradient objective."""
    loss_a = DummyLossA()
    loss_a.name = "a"
    joint_loss = JointLoss(terms=[(loss_a, 1.0)])
    coords = jnp.array([2.0, 0.0, 0.0])

    original = float(joint_loss(None, coords))
    joint_loss.freeze_term(0)
    assert float(joint_loss(None, coords)) == pytest.approx(0.0, abs=1e-7)

    joint_loss.unfreeze_term(0)
    restored = float(joint_loss(None, coords))
    assert restored == pytest.approx(original, rel=1e-6)
    assert not joint_loss.is_frozen(0)


def test_is_frozen_default_false():
    """No terms should be frozen at construction."""
    loss_a = DummyLossA()
    loss_a.name = "a"
    joint_loss = JointLoss(terms=[(loss_a, 1.0)])
    assert not joint_loss.is_frozen(0)


def test_freeze_out_of_range_raises():
    loss_a = DummyLossA()
    loss_a.name = "a"
    joint_loss = JointLoss(terms=[(loss_a, 1.0)])
    with pytest.raises(IndexError, match="out of range"):
        joint_loss.freeze_term(5)


def test_unfreeze_out_of_range_raises():
    loss_a = DummyLossA()
    loss_a.name = "a"
    joint_loss = JointLoss(terms=[(loss_a, 1.0)])
    with pytest.raises(IndexError, match="out of range"):
        joint_loss.unfreeze_term(-1)


def test_evaluate_terms_includes_frozen_with_suffix():
    """Frozen terms should appear in evaluate_terms output with '(frozen)' suffix."""
    loss_a = DummyLossA()
    loss_a.name = "a"
    loss_b = DummyLossB()
    loss_b.name = "b"
    joint_loss = JointLoss(terms=[(loss_a, 1.0), (loss_b, 1.0)])
    coords = jnp.array([2.0, 0.0, 0.0])

    joint_loss.freeze_term(0)
    result = joint_loss.evaluate_terms(None, coords)

    # term 'a' is frozen — key should be 'a(frozen)'
    assert "a(frozen)" in result
    # term 'b' is not frozen — key should be plain 'b'
    assert "b" in result
    # both values should still be evaluated
    assert result["a(frozen)"] == pytest.approx(4.0, rel=1e-5)   # raw=4.0 * weight 1.0


def test_freeze_multiple_terms():
    """Multiple terms can be frozen simultaneously."""
    loss_a = DummyLossA()
    loss_a.name = "a"
    loss_b = DummyLossB()
    loss_b.name = "b"
    joint_loss = JointLoss(terms=[(loss_a, 2.0), (loss_b, 3.0)])
    coords = jnp.array([2.0, 0.0, 0.0])

    joint_loss.freeze_term(0)
    joint_loss.freeze_term(1)
    assert float(joint_loss(None, coords)) == pytest.approx(0.0, abs=1e-7)
    assert joint_loss.is_frozen(0)
    assert joint_loss.is_frozen(1)

