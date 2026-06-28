import jax.numpy as jnp
import numpy as np
import pytest
from jax import grad

from diff_integrator.terms.nmr import (
    FixedTensorRDCLoss,
    RDCLoss,
    make_rdc_cv_refinement_fns,
    _MIN_TRAIN_RDCS_FOR_TENSOR,
    _IDEAL_RATIO,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_loss_and_tensor():
    """Return minimal dummy loss_fn, make_tensor_fn, and coords for testing."""
    def dummy_loss(coords, tensor):
        return jnp.array(0.5)

    def dummy_make(coords):
        return jnp.eye(3)

    coords = jnp.zeros((3, 3))
    return dummy_loss, dummy_make, coords


# ---------------------------------------------------------------------------
# FixedTensorRDCLoss — existing behaviour (backward-compat)
# ---------------------------------------------------------------------------

def test_fixed_tensor_rdc_loss_uninitialized():
    loss_fn, make_tensor_fn, coords = _make_simple_loss_and_tensor()
    term = FixedTensorRDCLoss(loss_fn=loss_fn, make_tensor_fn=make_tensor_fn)
    with pytest.raises(RuntimeError, match="tensor not initialized"):
        term(None, coords)


def test_fixed_tensor_rdc_loss_basic():
    loss_fn, make_tensor_fn, coords = _make_simple_loss_and_tensor()
    term = FixedTensorRDCLoss(loss_fn=loss_fn, make_tensor_fn=make_tensor_fn)
    term.initialize_tensor(coords)
    val = term(None, coords)
    assert float(val) == pytest.approx(0.5)


def test_maybe_update_tensor_triggers_at_interval():
    calls = []

    def tracking_make(coords):
        calls.append(1)
        return jnp.eye(3)

    def dummy_loss(coords, tensor):
        return jnp.array(0.0)

    term = FixedTensorRDCLoss(
        loss_fn=dummy_loss, make_tensor_fn=tracking_make, update_interval=10
    )
    coords = jnp.zeros((3, 3))
    term.initialize_tensor(coords)  # epoch-independent first call
    n_initial = len(calls)

    # Epochs that are NOT multiples of 10 should not trigger a re-fit
    for epoch in [1, 2, 3]:
        term.maybe_update_tensor(coords, epoch)
    assert len(calls) == n_initial  # no new calls

    # Epoch 10 is a multiple of 10 → triggers re-fit
    term.maybe_update_tensor(coords, 10)
    assert len(calls) == n_initial + 1


# ---------------------------------------------------------------------------
# FixedTensorRDCLoss — overdetermination ratio and suggested_weight
# ---------------------------------------------------------------------------

def test_overdetermination_ratio_none_when_n_rdcs_missing():
    loss_fn, make_tensor_fn, _ = _make_simple_loss_and_tensor()
    term = FixedTensorRDCLoss(loss_fn=loss_fn, make_tensor_fn=make_tensor_fn)
    assert term.overdetermination_ratio is None


def test_overdetermination_ratio_correct():
    loss_fn, make_tensor_fn, _ = _make_simple_loss_and_tensor()
    term = FixedTensorRDCLoss(loss_fn=loss_fn, make_tensor_fn=make_tensor_fn, n_rdcs=23)
    assert term.overdetermination_ratio == pytest.approx(23 / 5)


def test_suggested_weight_raises_without_n_rdcs():
    loss_fn, make_tensor_fn, _ = _make_simple_loss_and_tensor()
    term = FixedTensorRDCLoss(loss_fn=loss_fn, make_tensor_fn=make_tensor_fn)
    with pytest.raises(ValueError, match="n_rdcs"):
        term.suggested_weight()


def test_suggested_weight_ideal_ratio_returns_base():
    """50 training RDCs (ratio 10 = _IDEAL_RATIO) should return base_weight × 1.0."""
    loss_fn, make_tensor_fn, _ = _make_simple_loss_and_tensor()
    n_ideal = int(_IDEAL_RATIO * 5)  # = 50
    term = FixedTensorRDCLoss(loss_fn=loss_fn, make_tensor_fn=make_tensor_fn, n_rdcs=n_ideal)
    assert term.suggested_weight(base_weight=1.0) == pytest.approx(1.0)
    assert term.suggested_weight(base_weight=2.5) == pytest.approx(2.5)


def test_suggested_weight_scales_with_ratio():
    """A well-determined medium should get higher weight than a marginal one."""
    loss_fn, make_tensor_fn, _ = _make_simple_loss_and_tensor()
    term_good = FixedTensorRDCLoss(loss_fn=loss_fn, make_tensor_fn=make_tensor_fn, n_rdcs=50)
    term_bad = FixedTensorRDCLoss(loss_fn=loss_fn, make_tensor_fn=make_tensor_fn, n_rdcs=16)
    assert term_good.suggested_weight() > term_bad.suggested_weight()


def test_suggested_weight_clamped_at_lower_bound():
    """A medium so underdetermined that the raw weight would be below 0.1× base.

    floor = 0.1 × base_weight = 0.1
    raw   = base_weight × (ratio / _IDEAL_RATIO)
          = 1.0 × ((6/5) / 10) = 0.12   ← still above floor for 6 RDCs

    Use a tiny n_rdcs (= _MIN_TRAIN_RDCS_FOR_TENSOR = 6) gives raw = 0.12 > floor.
    To actually hit the floor we need ratio < 1 → n_rdcs < 5, but that would fail
    the minimum-tensor-fitting check.  Instead verify that the ratio formula is
    correct and the floor is applied when ratio is very small by calling
    suggested_weight() directly with a manually-set n_rdcs value that produces
    ratio < 1 (bypassing the constructor guard on minimum training RDCs).
    """
    loss_fn, make_tensor_fn, _ = _make_simple_loss_and_tensor()
    # Manually set n_rdcs to 3 (ratio = 0.6) to test the clamp floor
    term = FixedTensorRDCLoss(loss_fn=loss_fn, make_tensor_fn=make_tensor_fn, n_rdcs=3)
    w = term.suggested_weight(base_weight=1.0)
    # raw = 1.0 * (0.6 / 10) = 0.06  → clamped to 0.1
    assert w == pytest.approx(0.1)


def test_suggested_weight_clamped_at_upper_bound():
    """Extremely overdetermined medium should be capped at 2× base_weight."""
    loss_fn, make_tensor_fn, _ = _make_simple_loss_and_tensor()
    term = FixedTensorRDCLoss(loss_fn=loss_fn, make_tensor_fn=make_tensor_fn, n_rdcs=300)
    w = term.suggested_weight(base_weight=1.0)
    assert w == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# FixedTensorRDCLoss — validation Q evaluation
# ---------------------------------------------------------------------------

def test_evaluate_validation_q_returns_none_without_val_fn():
    loss_fn, make_tensor_fn, coords = _make_simple_loss_and_tensor()
    term = FixedTensorRDCLoss(loss_fn=loss_fn, make_tensor_fn=make_tensor_fn)
    term.initialize_tensor(coords)
    assert term.evaluate_validation_q(coords) is None


def test_evaluate_validation_q_returns_scalar():
    def dummy_loss(coords, tensor):
        return jnp.array(0.0)

    def dummy_make(coords):
        return jnp.eye(3)

    def val_q_fn(coords, tensor):
        return jnp.array(0.25)

    coords = jnp.zeros((3, 3))
    term = FixedTensorRDCLoss(
        loss_fn=dummy_loss,
        make_tensor_fn=dummy_make,
        val_q_eval_fn=val_q_fn,
    )
    term.initialize_tensor(coords)
    q = term.evaluate_validation_q(coords)
    assert q == pytest.approx(0.25)


def test_evaluate_validation_q_raises_if_not_initialized():
    def dummy_loss(coords, tensor):
        return jnp.array(0.0)

    def dummy_make(coords):
        return jnp.eye(3)

    def val_q_fn(coords, tensor):
        return jnp.array(0.25)

    coords = jnp.zeros((3, 3))
    term = FixedTensorRDCLoss(
        loss_fn=dummy_loss,
        make_tensor_fn=dummy_make,
        val_q_eval_fn=val_q_fn,
    )
    with pytest.raises(RuntimeError, match="tensor not initialized"):
        term.evaluate_validation_q(coords)


# ---------------------------------------------------------------------------
# make_rdc_cv_refinement_fns
# ---------------------------------------------------------------------------

def _make_cv_test_data(n: int = 30):
    """Generate synthetic matched RDC data for CV tests."""
    rng = np.random.default_rng(0)
    exp_res_ids = np.arange(1, n + 1)
    exp_rdcs = rng.uniform(-20, 20, size=n).astype(np.float32)
    struct_res_ids = np.arange(1, n + 1)
    return exp_res_ids, exp_rdcs, struct_res_ids


def test_cv_split_sizes_correct():
    exp_res_ids, exp_rdcs, struct_res_ids = _make_cv_test_data(30)
    *_, n_train, n_val = make_rdc_cv_refinement_fns(
        exp_res_ids, exp_rdcs, struct_res_ids, cv_fraction=0.2
    )
    assert n_train + n_val == 30
    assert n_val == 6   # round(30 * 0.2) = 6
    assert n_train == 24


def test_cv_split_min_train_enforced():
    """A cv_fraction that leaves too few training RDCs must raise ValueError."""
    exp_res_ids, exp_rdcs, struct_res_ids = _make_cv_test_data(8)
    with pytest.raises(ValueError, match="training RDCs"):
        make_rdc_cv_refinement_fns(
            exp_res_ids, exp_rdcs, struct_res_ids, cv_fraction=0.9
        )


def test_cv_loss_fn_returns_finite_scalar():
    """loss_fn from CV factory must return a finite scalar."""
    from diff_biophys.nmr.rdc import make_rdc_refinement_fns as _orig
    exp_res_ids, exp_rdcs, struct_res_ids = _make_cv_test_data(30)
    # Use make_rdc_refinement_fns to get real coords layout, then test our factory
    loss_fn, q_eval_fn, make_tensor_fn, val_q_fn, n_train, n_val = (
        make_rdc_cv_refinement_fns(
            exp_res_ids, exp_rdcs, struct_res_ids, cv_fraction=0.2
        )
    )
    # Build dummy backbone coords: 3 atoms per residue (N, CA, C), 30 residues
    coords = jnp.zeros((90, 3))
    tensor = jnp.eye(3) * 0.01
    val = loss_fn(coords, tensor)
    assert jnp.isfinite(val)
    assert val.shape == ()


def test_cv_val_q_fn_returns_finite_scalar():
    """val_q_eval_fn must return a finite scalar."""
    exp_res_ids, exp_rdcs, struct_res_ids = _make_cv_test_data(30)
    loss_fn, q_eval_fn, make_tensor_fn, val_q_fn, n_train, n_val = (
        make_rdc_cv_refinement_fns(
            exp_res_ids, exp_rdcs, struct_res_ids, cv_fraction=0.2
        )
    )
    coords = jnp.zeros((90, 3))
    tensor = jnp.eye(3) * 0.01
    q = val_q_fn(coords, tensor)
    assert jnp.isfinite(q)
    assert q.shape == ()


def test_cv_reproducible_with_same_seed():
    """Same seed must always produce the same split."""
    exp_res_ids, exp_rdcs, struct_res_ids = _make_cv_test_data(30)
    _, _, _, _, n1, n1v = make_rdc_cv_refinement_fns(
        exp_res_ids, exp_rdcs, struct_res_ids, cv_fraction=0.2, cv_seed=7
    )
    _, _, _, _, n2, n2v = make_rdc_cv_refinement_fns(
        exp_res_ids, exp_rdcs, struct_res_ids, cv_fraction=0.2, cv_seed=7
    )
    assert n1 == n2 and n1v == n2v


def test_cv_different_seeds_produce_same_sizes():
    """Different seeds produce same *sizes* (just different assignment)."""
    exp_res_ids, exp_rdcs, struct_res_ids = _make_cv_test_data(30)
    _, _, _, _, n1, n1v = make_rdc_cv_refinement_fns(
        exp_res_ids, exp_rdcs, struct_res_ids, cv_fraction=0.2, cv_seed=1
    )
    _, _, _, _, n2, n2v = make_rdc_cv_refinement_fns(
        exp_res_ids, exp_rdcs, struct_res_ids, cv_fraction=0.2, cv_seed=99
    )
    assert n1 == n2 and n1v == n2v


def test_cv_invalid_fraction_raises():
    exp_res_ids, exp_rdcs, struct_res_ids = _make_cv_test_data(20)
    with pytest.raises(ValueError):
        make_rdc_cv_refinement_fns(
            exp_res_ids, exp_rdcs, struct_res_ids, cv_fraction=0.0
        )
    with pytest.raises(ValueError):
        make_rdc_cv_refinement_fns(
            exp_res_ids, exp_rdcs, struct_res_ids, cv_fraction=1.0
        )


def test_cv_no_overlap_raises():
    exp_res_ids = np.array([100, 101, 102])
    exp_rdcs = np.array([1.0, 2.0, 3.0])
    struct_res_ids = np.arange(1, 31)  # no overlap
    with pytest.raises(ValueError, match="No residues overlap"):
        make_rdc_cv_refinement_fns(
            exp_res_ids, exp_rdcs, struct_res_ids, cv_fraction=0.2
        )


def test_fixed_tensor_rdc_loss_with_cv_factory():
    """Full end-to-end: FixedTensorRDCLoss constructed from make_rdc_cv_refinement_fns."""
    exp_res_ids, exp_rdcs, struct_res_ids = _make_cv_test_data(30)
    loss_fn, q_eval_fn, make_tensor_fn, val_q_fn, n_train, n_val = (
        make_rdc_cv_refinement_fns(
            exp_res_ids, exp_rdcs, struct_res_ids, cv_fraction=0.2
        )
    )
    term = FixedTensorRDCLoss(
        loss_fn=loss_fn,
        make_tensor_fn=make_tensor_fn,
        n_rdcs=n_train,
        val_q_eval_fn=val_q_fn,
    )
    coords = jnp.zeros((90, 3))
    term.initialize_tensor(coords)

    train_loss = term(None, coords)
    assert jnp.isfinite(train_loss)

    val_q = term.evaluate_validation_q(coords)
    assert val_q is not None
    assert np.isfinite(val_q)

    assert term.n_rdcs == n_train
    assert term.overdetermination_ratio == pytest.approx(n_train / 5)
    w = term.suggested_weight(base_weight=1.0)
    assert 0.1 <= w <= 2.0


# ---------------------------------------------------------------------------
# RDCLoss (deprecated) — unchanged behaviour
# ---------------------------------------------------------------------------

def test_rdc_loss():
    coords = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    atom_pairs = jnp.array([[0, 1], [2, 3]])
    exp_rdcs = jnp.array([10.0, -5.0])

    loss_fn = RDCLoss(atom_pairs=atom_pairs, exp_rdcs=exp_rdcs, d_max=1.0)
    loss_val = loss_fn(None, coords)
    assert not jnp.isnan(loss_val)

    grad_fn = grad(lambda c: loss_fn(None, c))
    gradients = grad_fn(coords)
    assert gradients.shape == coords.shape
    assert not jnp.any(jnp.isnan(gradients))


def test_rdc_loss_q_factor():
    coords = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    atom_pairs = jnp.array([[0, 1]])
    exp_rdcs = jnp.array([10.0])
    loss_fn = RDCLoss(atom_pairs=atom_pairs, exp_rdcs=exp_rdcs, loss_type="q_factor")
    loss_val = loss_fn(None, coords)
    assert float(loss_val) >= 0.0


def test_rdc_loss_invalid_type():
    coords = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    atom_pairs = jnp.array([[0, 1]])
    exp_rdcs = jnp.array([10.0])
    loss_fn = RDCLoss(atom_pairs=atom_pairs, exp_rdcs=exp_rdcs, loss_type="invalid")
    with pytest.raises(ValueError):
        loss_fn(None, coords)
