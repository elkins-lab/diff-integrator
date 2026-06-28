"""
Tests for diff_integrator/schedules.py — LinearSchedule and CosineAnnealingSchedule.

ExponentialDecaySchedule is already tested in test_optimizer.py; those tests are
not duplicated here.
"""

import math

import pytest

from diff_integrator.schedules import CosineAnnealingSchedule, LinearSchedule

# ---------------------------------------------------------------------------
# LinearSchedule
# ---------------------------------------------------------------------------


def test_linear_schedule_at_zero():
    sched = LinearSchedule(10.0, 0.0, 100)
    assert sched(0) == pytest.approx(10.0)


def test_linear_schedule_at_midpoint():
    sched = LinearSchedule(10.0, 0.0, 100)
    assert sched(50) == pytest.approx(5.0)


def test_linear_schedule_at_end():
    sched = LinearSchedule(10.0, 0.0, 100)
    assert sched(100) == pytest.approx(0.0)


def test_linear_schedule_clamped_beyond_decay_epochs():
    """After decay_epochs the weight must stay at final_weight."""
    sched = LinearSchedule(10.0, 0.0, 100)
    assert sched(200) == pytest.approx(0.0)
    assert sched(1000) == pytest.approx(0.0)


def test_linear_schedule_increasing_warmup():
    """LinearSchedule supports initial < final (warm-up use case)."""
    sched = LinearSchedule(0.0, 10.0, 100)
    assert sched(0) == pytest.approx(0.0)
    assert sched(50) == pytest.approx(5.0)
    assert sched(100) == pytest.approx(10.0)


def test_linear_schedule_monotonic_decreasing():
    sched = LinearSchedule(5.0, 0.5, 100)
    weights = [sched(e) for e in range(0, 101, 5)]
    for a, b in zip(weights, weights[1:], strict=False):
        assert a >= b


def test_linear_schedule_monotonic_increasing():
    sched = LinearSchedule(0.5, 5.0, 100)
    weights = [sched(e) for e in range(0, 101, 5)]
    for a, b in zip(weights, weights[1:], strict=False):
        assert a <= b


def test_linear_schedule_invalid_decay_epochs():
    with pytest.raises(ValueError, match="decay_epochs must be > 0"):
        LinearSchedule(10.0, 0.0, 0)


def test_linear_schedule_invalid_final_weight():
    with pytest.raises(ValueError, match="final_weight must be >= 0"):
        LinearSchedule(10.0, -1.0, 100)


def test_linear_schedule_exact_formula():
    """Verify the exact interpolation formula at an arbitrary point."""
    sched = LinearSchedule(8.0, 2.0, 200)
    # at epoch 80: t=0.4, expected = 8 + (2-8)*0.4 = 8 - 2.4 = 5.6
    assert sched(80) == pytest.approx(5.6)


def test_linear_schedule_equal_weights():
    """Constant schedule (initial == final) should always return that value."""
    sched = LinearSchedule(3.0, 3.0, 50)
    for epoch in [0, 10, 50, 100]:
        assert sched(epoch) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# CosineAnnealingSchedule
# ---------------------------------------------------------------------------


def test_cosine_schedule_at_zero():
    sched = CosineAnnealingSchedule(10.0, 0.0, 100)
    assert sched(0) == pytest.approx(10.0, abs=1e-9)


def test_cosine_schedule_at_end():
    sched = CosineAnnealingSchedule(10.0, 0.0, 100)
    assert sched(100) == pytest.approx(0.0, abs=1e-9)


def test_cosine_schedule_at_midpoint():
    """At exactly half the decay period the cosine is exactly at the midpoint."""
    sched = CosineAnnealingSchedule(10.0, 0.0, 100)
    # cos(pi * 0.5) = 0, so cos_factor = 0.5 → value = 10 + (0-10)*0.5 = 5.0
    assert sched(50) == pytest.approx(5.0, abs=1e-9)


def test_cosine_schedule_clamped_beyond_decay_epochs():
    sched = CosineAnnealingSchedule(10.0, 0.0, 100)
    assert sched(200) == pytest.approx(0.0, abs=1e-9)
    assert sched(9999) == pytest.approx(0.0, abs=1e-9)


def test_cosine_schedule_monotonic_decreasing():
    sched = CosineAnnealingSchedule(5.0, 0.5, 100)
    weights = [sched(e) for e in range(0, 101, 2)]
    for a, b in zip(weights, weights[1:], strict=False):
        assert a >= b - 1e-10  # allow for floating-point equality at clamped region


def test_cosine_schedule_starts_slower_than_linear():
    """Cosine schedule should change more slowly than linear at the start."""
    cosine = CosineAnnealingSchedule(10.0, 0.0, 100)
    linear = LinearSchedule(10.0, 0.0, 100)
    # At epoch 10 (t'=0.1): cosine should have decayed LESS than linear
    # Linear: 10 + (0-10)*0.1 = 9.0
    # Cosine: cos_factor=(1-cos(0.1π))/2 ≈ 0.0245 → 10 + (0-10)*0.0245 = 9.755
    assert cosine(10) > linear(10)


def test_cosine_schedule_ends_slower_than_linear():
    """Cosine schedule should have less remaining weight than linear near the end."""
    cosine = CosineAnnealingSchedule(10.0, 0.0, 100)
    linear = LinearSchedule(10.0, 0.0, 100)
    # At epoch 90 (t'=0.9): cosine should be closer to final than linear
    # Linear: 10 + (0-10)*0.9 = 1.0
    # Cosine: cos_factor=(1-cos(0.9π))/2 ≈ 0.9755 → 10 + (0-10)*0.9755 ≈ 0.245
    assert cosine(90) < linear(90)


def test_cosine_schedule_increasing_warmup():
    sched = CosineAnnealingSchedule(0.0, 10.0, 100)
    assert sched(0) == pytest.approx(0.0, abs=1e-9)
    assert sched(100) == pytest.approx(10.0, abs=1e-9)
    assert sched(50) == pytest.approx(5.0, abs=1e-9)


def test_cosine_schedule_invalid_decay_epochs():
    with pytest.raises(ValueError, match="decay_epochs must be > 0"):
        CosineAnnealingSchedule(10.0, 0.0, 0)


def test_cosine_schedule_invalid_final_weight():
    with pytest.raises(ValueError, match="final_weight must be >= 0"):
        CosineAnnealingSchedule(10.0, -1.0, 100)


def test_cosine_schedule_exact_formula():
    """Verify exact cosine formula at t'=0.25."""
    sched = CosineAnnealingSchedule(8.0, 0.0, 100)
    # t'=0.25, cos_factor=(1-cos(0.25*pi))/2 = (1-sqrt(2)/2)/2
    cos_factor = (1.0 - math.cos(math.pi * 0.25)) / 2.0
    expected = 8.0 * (1.0 - cos_factor)  # 8 + (0-8)*cos_factor
    assert sched(25) == pytest.approx(expected, rel=1e-9)


def test_cosine_schedule_equal_weights():
    """Constant schedule should always return that value."""
    sched = CosineAnnealingSchedule(3.0, 3.0, 50)
    for epoch in [0, 10, 50, 100]:
        assert sched(epoch) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Integration: use LinearSchedule and CosineAnnealingSchedule in IntegrativeRefiner
# ---------------------------------------------------------------------------


def test_linear_schedule_used_in_refiner():
    """LinearSchedule correctly updates weight_history inside IntegrativeRefiner."""
    from typing import Any

    import jax.numpy as jnp

    from diff_integrator.loss import JointLoss, LossTerm
    from diff_integrator.optimizer import IntegrativeRefiner

    class QuadLoss(LossTerm):
        name = "quad"
        def __call__(self, params: Any, coords: jnp.ndarray) -> jnp.ndarray:
            return jnp.mean(params ** 2)

    sched = LinearSchedule(10.0, 0.0, 20)
    loss_fn = JointLoss([(QuadLoss(), 10.0)])
    refiner = IntegrativeRefiner(loss_fn=loss_fn)
    result = refiner.run(
        init_params=jnp.ones((2,)),
        epochs=10,
        weight_schedules={0: sched},
    )
    for epoch, w in enumerate(result.weight_history[0]):
        assert w == pytest.approx(sched(epoch), rel=1e-9)


def test_cosine_schedule_used_in_refiner():
    """CosineAnnealingSchedule correctly updates weight_history inside IntegrativeRefiner."""
    from typing import Any

    import jax.numpy as jnp

    from diff_integrator.loss import JointLoss, LossTerm
    from diff_integrator.optimizer import IntegrativeRefiner

    class QuadLoss(LossTerm):
        name = "quad"
        def __call__(self, params: Any, coords: jnp.ndarray) -> jnp.ndarray:
            return jnp.mean(params ** 2)

    sched = CosineAnnealingSchedule(10.0, 0.0, 20)
    loss_fn = JointLoss([(QuadLoss(), 10.0)])
    refiner = IntegrativeRefiner(loss_fn=loss_fn)
    result = refiner.run(
        init_params=jnp.ones((2,)),
        epochs=10,
        weight_schedules={0: sched},
    )
    for epoch, w in enumerate(result.weight_history[0]):
        assert w == pytest.approx(sched(epoch), rel=1e-9)
