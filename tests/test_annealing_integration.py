"""
Integration tests for the annealed geometry weight pattern.

These tests verify the end-to-end behaviour of ExponentialDecaySchedule
combined with the weight_schedules mechanism in IntegrativeRefiner — the
"annealed geometry restraint" pattern described in docs/algorithmic_improvements.md.

The key scientific property under test: when the geometry weight is annealed
from strong → weak, the experimental loss term (chemical shifts / target loss)
should improve *more* than with a fixed strong geometry weight, because the
experimental gradient gets increasing influence over time.
"""

from typing import Any

import jax.numpy as jnp
import pytest

from diff_integrator.loss import JointLoss, LossTerm
from diff_integrator.optimizer import IntegrativeRefiner
from diff_integrator.schedules import ExponentialDecaySchedule


# ---------------------------------------------------------------------------
# Minimal synthetic loss terms for controlled testing
# ---------------------------------------------------------------------------


class HarmonicAnchorLoss(LossTerm):
    """Harmonic restraint to a fixed anchor — analogous to GeometryLoss."""

    name = "anchor"

    def __init__(self, anchor: jnp.ndarray) -> None:
        self.anchor = anchor

    def __call__(self, params: Any, coords: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean((coords - self.anchor) ** 2)


class TargetLoss(LossTerm):
    """Drives coords toward a target — analogous to a chemical-shift or RDC term."""

    name = "target"

    def __init__(self, target: jnp.ndarray) -> None:
        self.target = target

    def __call__(self, params: Any, coords: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean((coords - self.target) ** 2)


# ---------------------------------------------------------------------------
# Test: geometry weight decays correctly end-to-end
# ---------------------------------------------------------------------------


def test_annealed_weight_decays_in_run():
    """The geometry weight in weight_history should follow the schedule exactly."""
    anchor = jnp.zeros((2, 3))
    target = jnp.ones((2, 3))
    init = jnp.array([[0.1, 0.1, 0.1], [-0.1, -0.1, -0.1]])

    schedule = ExponentialDecaySchedule(
        initial_weight=8.0,
        final_weight=0.5,
        decay_epochs=50,
    )

    loss_fn = JointLoss([
        (HarmonicAnchorLoss(anchor), 8.0),  # index 0 — annealed
        (TargetLoss(target), 1.0),           # index 1 — fixed
    ])
    refiner = IntegrativeRefiner(loss_fn=loss_fn)

    result = refiner.run(
        init_params=init,
        epochs=20,
        learning_rate=0.05,
        weight_schedules={0: schedule},
    )

    # Weight history should be exactly the schedule values
    for epoch, w in enumerate(result.weight_history[0]):
        expected = schedule(epoch)
        assert abs(w - expected) < 1e-9, (
            f"Epoch {epoch}: expected weight {expected:.6f}, got {w:.6f}"
        )

    # Weight should be strictly decreasing
    wh = result.weight_history[0]
    assert all(a > b for a, b in zip(wh, wh[1:]))


# ---------------------------------------------------------------------------
# Test: annealing improves experimental loss vs. fixed strong weight
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_annealing_improves_experimental_loss_vs_fixed_strong():
    """
    With equal experimental and geometry terms, an annealed geometry weight
    should yield better experimental-term improvement than a fixed strong weight.

    Scientific rationale: a fixed weight_geometry=8.0 competes strongly against
    the target gradient throughout training.  The annealed version starts equally
    strong but relaxes, allowing the target gradient to dominate later.
    """
    anchor = jnp.zeros((3, 3))    # Anchor: origin
    target = jnp.array([          # Target: far from origin
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0],
    ])
    init = jnp.zeros((3, 3))

    def run_with_weight(geom_weight_fixed: float | None,
                        schedule: ExponentialDecaySchedule | None) -> float:
        """Return final target loss value."""
        loss_fn = JointLoss([
            (HarmonicAnchorLoss(anchor), geom_weight_fixed or 8.0),
            (TargetLoss(target), 1.0),
        ])
        refiner = IntegrativeRefiner(loss_fn=loss_fn)
        ws = {0: schedule} if schedule is not None else None
        result = refiner.run(
            init_params=init,
            epochs=200,
            learning_rate=0.05,
            weight_schedules=ws,
        )
        # Evaluate target loss at final params
        final_coords = result.final_params
        return float(TargetLoss(target)(None, final_coords))

    # Fixed strong geometry weight throughout
    fixed_target_loss = run_with_weight(8.0, None)

    # Annealed: starts at 8.0, decays to 0.1 over 100 epochs
    sched = ExponentialDecaySchedule(8.0, 0.1, 100)
    annealed_target_loss = run_with_weight(None, sched)

    assert annealed_target_loss < fixed_target_loss, (
        f"Annealed target loss ({annealed_target_loss:.4f}) should be less than "
        f"fixed weight target loss ({fixed_target_loss:.4f})"
    )


# ---------------------------------------------------------------------------
# Test: weight_history is empty when no schedules provided
# ---------------------------------------------------------------------------


def test_no_schedules_leaves_weight_history_empty():
    """Confirm weight_history is {} when weight_schedules is not provided."""
    anchor = jnp.zeros((2, 3))
    loss_fn = JointLoss([(HarmonicAnchorLoss(anchor), 1.0)])
    refiner = IntegrativeRefiner(loss_fn=loss_fn)

    result = refiner.run(init_params=jnp.ones((2, 3)), epochs=5)
    assert result.weight_history == {}
