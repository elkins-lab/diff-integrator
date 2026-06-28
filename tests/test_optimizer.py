from typing import Any

import jax.numpy as jnp

from diff_integrator.loss import JointLoss, LossTerm
from diff_integrator.optimizer import IntegrativeRefiner, RefinementResult
from diff_integrator.schedules import ExponentialDecaySchedule


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


# ---------------------------------------------------------------------------
# ExponentialDecaySchedule tests
# ---------------------------------------------------------------------------


def test_exponential_decay_schedule_initial():
    """At epoch 0 the weight equals initial_weight."""
    sched = ExponentialDecaySchedule(initial_weight=10.0, final_weight=0.1, decay_epochs=200)
    assert abs(sched(0) - 10.0) < 1e-9


def test_exponential_decay_schedule_asymptote():
    """At a very large epoch the weight is close to final_weight."""
    sched = ExponentialDecaySchedule(initial_weight=10.0, final_weight=0.1, decay_epochs=100)
    # After 10 time-constants (1000 epochs) should be within 0.01 of final
    assert abs(sched(1000) - 0.1) < 0.01


def test_exponential_decay_schedule_monotonic():
    """Weights should be strictly decreasing for initial > final."""
    sched = ExponentialDecaySchedule(initial_weight=5.0, final_weight=0.5, decay_epochs=100)
    weights = [sched(e) for e in range(0, 500, 10)]
    for a, b in zip(weights, weights[1:], strict=False):
        assert a > b, f"Not monotonically decreasing: {a} <= {b}"


def test_exponential_decay_schedule_one_time_constant():
    """At epoch == decay_epochs, ~36.8% of the initial-to-final range remains."""
    import math
    sched = ExponentialDecaySchedule(initial_weight=10.0, final_weight=0.0, decay_epochs=200)
    expected = 10.0 * math.exp(-1)  # ≈ 3.679
    assert abs(sched(200) - expected) < 1e-9


def test_exponential_decay_schedule_invalid_decay_epochs():
    """decay_epochs <= 0 should raise ValueError."""
    import pytest
    with pytest.raises(ValueError, match="decay_epochs must be > 0"):
        ExponentialDecaySchedule(10.0, 0.1, 0)


def test_exponential_decay_schedule_invalid_final_weight():
    """Negative final_weight should raise ValueError."""
    import pytest
    with pytest.raises(ValueError, match="final_weight must be >= 0"):
        ExponentialDecaySchedule(10.0, -1.0, 100)


# ---------------------------------------------------------------------------
# weight_schedules integration tests
# ---------------------------------------------------------------------------


def test_weight_schedules_applied_to_loss():
    """weight_schedules should update term weight before each gradient step."""
    target_coords = jnp.array([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
    init_coords = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    term = DummyTargetLoss(target_coords)
    loss_fn = JointLoss([(term, 99.0)])  # Initial weight deliberately wrong
    refiner = IntegrativeRefiner(loss_fn=loss_fn)

    # Schedule: always return weight 1.0 regardless of epoch
    constant_schedule = lambda epoch: 1.0  # noqa: E731

    result = refiner.run(
        init_params=init_coords,
        epochs=10,
        learning_rate=0.1,
        weight_schedules={0: constant_schedule},
    )

    # weight_history for term 0 should have 10 entries, all == 1.0
    assert 0 in result.weight_history
    assert len(result.weight_history[0]) == 10
    assert all(abs(w - 1.0) < 1e-9 for w in result.weight_history[0])


def test_weight_history_recorded_in_result():
    """weight_history keys match weight_schedules keys; length equals epochs_run."""
    target_coords = jnp.array([[1.0, 0.0, 0.0]])
    init_coords = jnp.array([[0.0, 0.0, 0.0]])

    loss_fn = JointLoss([
        (DummyTargetLoss(target_coords), 1.0),
        (DummySumLoss(), 0.5),
    ])
    refiner = IntegrativeRefiner(loss_fn=loss_fn)

    sched_0 = ExponentialDecaySchedule(5.0, 0.1, 50)
    sched_1 = ExponentialDecaySchedule(2.0, 0.5, 50)

    result = refiner.run(
        init_params=init_coords,
        epochs=30,
        weight_schedules={0: sched_0, 1: sched_1},
    )

    assert set(result.weight_history.keys()) == {0, 1}
    assert len(result.weight_history[0]) == 30
    assert len(result.weight_history[1]) == 30
    # First recorded weight should match schedule at epoch 0
    assert abs(result.weight_history[0][0] - sched_0(0)) < 1e-6
    assert abs(result.weight_history[1][0] - sched_1(0)) < 1e-6


def test_no_weight_history_without_schedules():
    """weight_history is empty when no weight_schedules are provided."""
    target_coords = jnp.array([[1.0, 0.0, 0.0]])
    init_coords = jnp.array([[0.0, 0.0, 0.0]])

    loss_fn = JointLoss([(DummyTargetLoss(target_coords), 1.0)])
    refiner = IntegrativeRefiner(loss_fn=loss_fn)

    result = refiner.run(init_params=init_coords, epochs=5)
    assert result.weight_history == {}


# ---------------------------------------------------------------------------
# EarlyStopping dataclass validation
# ---------------------------------------------------------------------------

import pytest  # noqa: E402

from diff_integrator.optimizer import EarlyStopping  # noqa: E402


def test_early_stopping_invalid_patience():
    with pytest.raises(ValueError, match="patience must be > 0"):
        EarlyStopping(term_index=0, patience=0)


def test_early_stopping_invalid_min_delta():
    with pytest.raises(ValueError, match="min_delta must be >= 0"):
        EarlyStopping(term_index=0, patience=5, min_delta=-1.0)


def test_early_stopping_invalid_mode():
    with pytest.raises(ValueError, match="mode must be"):
        EarlyStopping(term_index=0, patience=5, mode="median")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FlatLoss(LossTerm):
    name: str = "flat"

    def __call__(self, params, coords):
        import jax.numpy as jnp
        return jnp.array(1.0)


# ---------------------------------------------------------------------------
# Per-term early stopping integration tests
# ---------------------------------------------------------------------------


def test_per_term_early_stopping_fires():
    target = jnp.array([[1.0, 0.0, 0.0]])
    init = jnp.array([[0.0, 0.0, 0.0]])
    loss_fn = JointLoss([(DummyTargetLoss(target), 1.0), (FlatLoss(), 0.5)])
    result = IntegrativeRefiner(loss_fn=loss_fn).run(
        init_params=init, epochs=500, learning_rate=0.1,
        early_stopping=EarlyStopping(term_index=1, patience=10, min_delta=1e-6),
    )
    assert result.stopped_early is True
    assert result.epochs_run <= 15
    assert result.stopped_at_epoch >= 0
    assert "term_1" in result.early_stopping_triggered_by
    assert "flat" in result.early_stopping_triggered_by
    assert "patience=10" in result.early_stopping_triggered_by


def test_per_term_early_stopping_does_not_fire_if_improving():
    target = jnp.array([[1.0, 0.0, 0.0]])
    init = jnp.array([[0.0, 0.0, 0.0]])
    loss_fn = JointLoss([(DummyTargetLoss(target), 1.0)])
    result = IntegrativeRefiner(loss_fn=loss_fn).run(
        init_params=init, epochs=50, learning_rate=0.1,
        early_stopping=EarlyStopping(term_index=0, patience=40, min_delta=1e-8),
    )
    assert result.stopped_early is False
    assert result.epochs_run == 50
    assert result.stopped_at_epoch == -1
    assert result.early_stopping_triggered_by == ""


def test_per_term_early_stopping_multiple_terms():
    target = jnp.array([[1.0, 0.0, 0.0]])
    init = jnp.array([[0.0, 0.0, 0.0]])
    flat_fast = FlatLoss(); flat_fast.name = "flat_fast"
    flat_slow = FlatLoss(); flat_slow.name = "flat_slow"
    loss_fn = JointLoss([
        (DummyTargetLoss(target), 1.0), (flat_fast, 0.5), (flat_slow, 0.5),
    ])
    result = IntegrativeRefiner(loss_fn=loss_fn).run(
        init_params=init, epochs=500, learning_rate=0.1,
        early_stopping=[
            EarlyStopping(term_index=1, patience=5,  min_delta=1e-6),
            EarlyStopping(term_index=2, patience=20, min_delta=1e-6),
        ],
    )
    assert result.stopped_early is True
    assert "term_1" in result.early_stopping_triggered_by
    assert result.epochs_run <= 10


def test_per_term_early_stopping_result_fields():
    target = jnp.array([[1.0, 0.0, 0.0]])
    init = jnp.array([[0.0, 0.0, 0.0]])
    flat = FlatLoss(); flat.name = "observable"
    loss_fn = JointLoss([(DummyTargetLoss(target), 1.0), (flat, 1.0)])
    result = IntegrativeRefiner(loss_fn=loss_fn).run(
        init_params=init, epochs=500,
        early_stopping=EarlyStopping(term_index=1, patience=7, min_delta=1e-9),
    )
    assert result.stopped_at_epoch >= 0
    assert result.stopped_at_epoch == result.epochs_run - 1
    assert "term_1" in result.early_stopping_triggered_by
    assert "observable" in result.early_stopping_triggered_by
    assert "patience=7" in result.early_stopping_triggered_by


def test_per_term_early_stopping_single_instance():
    target = jnp.array([[1.0, 0.0, 0.0]])
    init = jnp.array([[0.0, 0.0, 0.0]])
    loss_fn = JointLoss([(DummyTargetLoss(target), 1.0), (FlatLoss(), 0.5)])
    result = IntegrativeRefiner(loss_fn=loss_fn).run(
        init_params=init, epochs=500, learning_rate=0.1,
        early_stopping=EarlyStopping(term_index=1, patience=5, min_delta=1e-9),
    )
    assert result.stopped_early is True
    assert result.epochs_run < 500


def test_global_and_per_term_stopping_coexist():
    target = jnp.array([[1.0, 0.0, 0.0]])
    init = jnp.array([[0.0, 0.0, 0.0]])
    loss_fn = JointLoss([(DummyTargetLoss(target), 1.0), (FlatLoss(), 0.5)])
    result = IntegrativeRefiner(loss_fn=loss_fn).run(
        init_params=init, epochs=2000, learning_rate=0.1,
        patience=1000, min_delta=1e-10,
        early_stopping=EarlyStopping(term_index=1, patience=5, min_delta=1e-9),
    )
    assert result.stopped_early is True
    assert result.epochs_run <= 10
    assert "term_1" in result.early_stopping_triggered_by


def test_per_term_early_stopping_no_fire_when_none():
    target = jnp.array([[1.0, 0.0, 0.0]])
    init = jnp.array([[0.0, 0.0, 0.0]])
    loss_fn = JointLoss([(DummyTargetLoss(target), 1.0)])
    result = IntegrativeRefiner(loss_fn=loss_fn).run(init_params=init, epochs=10)
    assert result.stopped_at_epoch == -1
    assert result.early_stopping_triggered_by == ""


# ---------------------------------------------------------------------------
# log_interval behavior
# ---------------------------------------------------------------------------


def test_log_interval_shortens_per_term_history():
    """When log_interval=5, per_term_history should have ceil(epochs/5) entries."""
    target = jnp.array([[1.0, 0.0, 0.0]])
    init = jnp.array([[0.0, 0.0, 0.0]])

    class NamedLoss(LossTerm):
        name = "named"
        def __call__(self, params: Any, coords: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum((coords - target) ** 2)

    loss_fn = JointLoss([(NamedLoss(), 1.0)])
    result = IntegrativeRefiner(loss_fn=loss_fn).run(
        init_params=init,
        epochs=20,
        log_interval=5,
    )
    # Epochs 0, 5, 10, 15 are logged → 4 entries
    expected_len = len(range(0, 20, 5))
    assert len(result.per_term_history["named"]) == expected_len
    # loss_history still has one entry per epoch
    assert len(result.loss_history) == 20


def test_per_term_epochs_matches_log_interval():
    """per_term_epochs records exactly which epoch indices were logged."""
    target = jnp.array([[1.0, 0.0, 0.0]])
    init = jnp.array([[0.0, 0.0, 0.0]])

    class NamedLoss(LossTerm):
        name = "nl"
        def __call__(self, params: Any, coords: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum((coords - target) ** 2)

    loss_fn = JointLoss([(NamedLoss(), 1.0)])
    result = IntegrativeRefiner(loss_fn=loss_fn).run(
        init_params=init,
        epochs=15,
        log_interval=3,
    )
    expected = list(range(0, 15, 3))  # [0, 3, 6, 9, 12]
    assert result.per_term_epochs == expected
    # Length of per_term_history must match per_term_epochs
    assert len(result.per_term_history["nl"]) == len(result.per_term_epochs)


def test_per_term_epochs_default_log_interval():
    """With log_interval=1 (default), per_term_epochs == list(range(epochs))."""
    target = jnp.array([[1.0, 0.0, 0.0]])
    init = jnp.array([[0.0, 0.0, 0.0]])

    class NamedLoss(LossTerm):
        name = "nl"
        def __call__(self, params: Any, coords: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum((coords - target) ** 2)

    loss_fn = JointLoss([(NamedLoss(), 1.0)])
    result = IntegrativeRefiner(loss_fn=loss_fn).run(init_params=init, epochs=8)
    assert result.per_term_epochs == list(range(8))


# ---------------------------------------------------------------------------
# per_epoch_callbacks
# ---------------------------------------------------------------------------


def test_per_epoch_callbacks_fired_every_epoch():
    """per_epoch_callbacks must be called exactly once per epoch."""
    target = jnp.array([[1.0, 0.0, 0.0]])
    init = jnp.array([[0.0, 0.0, 0.0]])

    class NamedLoss(LossTerm):
        name = "quad"
        def __call__(self, params: Any, coords: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum((coords - target) ** 2)

    loss_fn = JointLoss([(NamedLoss(), 1.0)])

    call_log: list[int] = []

    def callback(epoch: int, params: Any, coords: jnp.ndarray) -> None:
        call_log.append(epoch)

    EPOCHS = 7
    IntegrativeRefiner(loss_fn=loss_fn).run(
        init_params=init,
        epochs=EPOCHS,
        per_epoch_callbacks=[callback],
    )

    assert call_log == list(range(EPOCHS)), (
        f"Expected callbacks at epochs 0..{EPOCHS-1}, got {call_log}"
    )


def test_per_epoch_callbacks_fired_even_when_log_interval_skips():
    """Callbacks fire at EVERY epoch, not only at log_interval epochs."""
    target = jnp.array([[1.0, 0.0, 0.0]])
    init = jnp.array([[0.0, 0.0, 0.0]])

    class NamedLoss(LossTerm):
        name = "quad"
        def __call__(self, params: Any, coords: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum((coords - target) ** 2)

    loss_fn = JointLoss([(NamedLoss(), 1.0)])

    epochs_seen: list[int] = []

    def callback(epoch: int, params: Any, coords: jnp.ndarray) -> None:
        epochs_seen.append(epoch)

    EPOCHS = 12
    IntegrativeRefiner(loss_fn=loss_fn).run(
        init_params=init,
        epochs=EPOCHS,
        log_interval=5,          # only 0, 5, 10 would be logged
        per_epoch_callbacks=[callback],
    )

    assert epochs_seen == list(range(EPOCHS)), (
        "Callbacks must fire at ALL epochs, not just log_interval epochs"
    )


def test_per_epoch_callbacks_multiple_callbacks():
    """Multiple callbacks in the list are all called."""
    jnp.array([[0.0]])
    init = jnp.array([[1.0]])

    class SimpleLoss(LossTerm):
        name = "simple"
        def __call__(self, params: Any, coords: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(coords ** 2)

    loss_fn = JointLoss([(SimpleLoss(), 1.0)])

    count_a: list[int] = []
    count_b: list[int] = []

    IntegrativeRefiner(loss_fn=loss_fn).run(
        init_params=init,
        epochs=5,
        per_epoch_callbacks=[
            lambda e, p, c: count_a.append(e),
            lambda e, p, c: count_b.append(e),
        ],
    )

    assert len(count_a) == 5
    assert len(count_b) == 5


def test_per_epoch_callback_receives_pre_step_params():
    """The params passed to each callback are the CURRENT (pre-update) params."""
    jnp.array([[0.0]])
    init = jnp.array([[5.0]])

    class SimpleLoss(LossTerm):
        name = "simple"
        def __call__(self, params: Any, coords: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(coords ** 2)

    loss_fn = JointLoss([(SimpleLoss(), 1.0)])
    params_at_each_epoch: list[float] = []

    def callback(epoch: int, params: Any, coords: jnp.ndarray) -> None:
        params_at_each_epoch.append(float(params[0, 0]))

    IntegrativeRefiner(loss_fn=loss_fn).run(
        init_params=init,
        epochs=5,
        learning_rate=0.5,
        per_epoch_callbacks=[callback],
    )

    # params must start at the initial value at epoch 0
    assert params_at_each_epoch[0] == 5.0
    # And decrease (optimizer is reducing the loss)
    assert params_at_each_epoch[0] > params_at_each_epoch[-1]


# ---------------------------------------------------------------------------
# EarlyStopping term_index out-of-range caught before loop
# ---------------------------------------------------------------------------


def test_early_stopping_invalid_term_index_raises_before_loop():
    """EarlyStopping with out-of-range term_index must raise IndexError immediately."""
    import pytest  # noqa: PLC0415

    from diff_integrator.optimizer import EarlyStopping  # noqa: PLC0415

    target = jnp.array([[1.0, 0.0, 0.0]])
    init = jnp.array([[0.0, 0.0, 0.0]])
    loss_fn = JointLoss([(DummyTargetLoss(target), 1.0)])  # 1 term only
    refiner = IntegrativeRefiner(loss_fn=loss_fn)

    with pytest.raises(IndexError, match="out of range"):
        refiner.run(
            init_params=init,
            epochs=100,
            early_stopping=EarlyStopping(
                term_index=5,   # invalid: only index 0 exists
                patience=10,
            ),
        )


def test_early_stopping_negative_term_index_raises_before_loop():
    """EarlyStopping with negative term_index must raise IndexError immediately."""
    import pytest  # noqa: PLC0415

    from diff_integrator.optimizer import EarlyStopping  # noqa: PLC0415

    target = jnp.array([[1.0, 0.0, 0.0]])
    init = jnp.array([[0.0, 0.0, 0.0]])
    loss_fn = JointLoss([(DummyTargetLoss(target), 1.0)])
    refiner = IntegrativeRefiner(loss_fn=loss_fn)

    with pytest.raises(IndexError, match="out of range"):
        refiner.run(
            init_params=init,
            epochs=100,
            early_stopping=EarlyStopping(
                term_index=-1,   # negative index
                patience=10,
            ),
        )


def test_early_stopping_invalid_term_index_in_list_raises_before_loop():
    """Same IndexError check for a list-of-EarlyStopping config."""
    import pytest  # noqa: PLC0415

    from diff_integrator.optimizer import EarlyStopping  # noqa: PLC0415

    target = jnp.array([[1.0, 0.0, 0.0]])
    init = jnp.array([[0.0, 0.0, 0.0]])

    class ValidLoss(LossTerm):
        name = "v"
        def __call__(self, params: Any, coords: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(coords ** 2)

    # 2-term loss, but early_stopping watches index 3
    loss_fn = JointLoss([
        (DummyTargetLoss(target), 1.0),
        (ValidLoss(), 0.5),
    ])
    refiner = IntegrativeRefiner(loss_fn=loss_fn)

    with pytest.raises(IndexError, match="out of range"):
        refiner.run(
            init_params=init,
            epochs=100,
            early_stopping=[
                EarlyStopping(term_index=0, patience=10),  # valid
                EarlyStopping(term_index=3, patience=10),  # INVALID
            ],
        )
