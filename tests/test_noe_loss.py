"""
tests/test_noe_loss.py — Tests for diff_integrator.terms.noe.NOELoss.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diff_integrator.terms.noe import NOELoss, make_noe_restraints

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coords(n: int = 10) -> jnp.ndarray:
    """Simple coordinates: atom i is at (i, 0, 0)."""
    return jnp.array([[float(i), 0.0, 0.0] for i in range(n)])


# ---------------------------------------------------------------------------
# Construction and properties
# ---------------------------------------------------------------------------


def test_n_restraints_property() -> None:
    loss = NOELoss(
        atom_pairs=jnp.array([[0, 1], [2, 3]]),
        d_upper=jnp.array([5.0, 5.0]),
    )
    assert loss.n_restraints == 2


def test_shape_mismatch_d_upper_raises() -> None:
    with pytest.raises(ValueError, match="d_upper must have shape"):
        NOELoss(
            atom_pairs=jnp.array([[0, 1], [2, 3]]),
            d_upper=jnp.array([5.0]),  # wrong: need 2 elements
        )


def test_shape_mismatch_d_lower_raises() -> None:
    with pytest.raises(ValueError, match="d_lower must have shape"):
        NOELoss(
            atom_pairs=jnp.array([[0, 1], [2, 3]]),
            d_upper=jnp.array([5.0, 5.0]),
            d_lower=jnp.array([1.0]),  # wrong: need 2 elements
        )


# ---------------------------------------------------------------------------
# Upper-bound violations
# ---------------------------------------------------------------------------


def test_upper_bound_no_violation() -> None:
    """Atoms exactly at d_upper — penalty should be zero."""
    coords = _coords()  # atom 0 at x=0, atom 4 at x=4 → d=4.0
    loss = NOELoss(
        atom_pairs=jnp.array([[0, 4]]),
        d_upper=jnp.array([4.0]),
    )
    assert float(loss(None, coords)) == pytest.approx(0.0, abs=1e-6)


def test_upper_bound_violated() -> None:
    """Atoms further apart than d_upper — penalty > 0."""
    coords = _coords()  # atom 0 at x=0, atom 5 at x=5 → d=5.0
    loss = NOELoss(
        atom_pairs=jnp.array([[0, 5]]),
        d_upper=jnp.array([3.0]),  # d_upper=3, violation of 2.0
    )
    val = float(loss(None, coords))
    # Expected: force_const * (5-3)^2 / 1 = 4.0
    assert val == pytest.approx(4.0, rel=1e-4)


def test_upper_bound_gradient_analytical() -> None:
    """Gradient at violation matches chain-rule formula analytically.

    loss = force_const * (d - d_upper)^2 / n_restraints
    d = sqrt(sum((r_i - r_j)^2) + eps) ≈ 6.0  for atoms at (0,0,0) and (6,0,0)

    d_loss/d_x0 = 2 * (d - d_upper) * (x0 - x1) / d
               = 2 * (6 - 5) * (0 - 6) / 6  = -2.0
    """
    coords = jnp.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]])  # d=6, d_upper=5
    loss = NOELoss(
        atom_pairs=jnp.array([[0, 1]]),
        d_upper=jnp.array([5.0]),
        force_const=1.0,
    )
    grad = jax.grad(lambda c: loss(None, c))(coords)
    # d_loss/d_x0 = 2*(6-5) * (0-6)/6 = -2.0
    expected_grad_atom0_x = 2.0 * (6.0 - 5.0) * (0.0 - 6.0) / 6.0
    assert float(grad[0, 0]) == pytest.approx(expected_grad_atom0_x, rel=1e-3)


# ---------------------------------------------------------------------------
# Lower-bound violations
# ---------------------------------------------------------------------------


def test_lower_bound_no_violation() -> None:
    """Atoms further apart than d_lower — no lower-bound penalty."""
    coords = _coords()  # atom 0 at x=0, atom 5 at x=5 → d=5.0
    loss = NOELoss(
        atom_pairs=jnp.array([[0, 5]]),
        d_upper=jnp.array([8.0]),
        d_lower=jnp.array([3.0]),  # d_lower=3, actual d=5 → no violation
    )
    assert float(loss(None, coords)) == pytest.approx(0.0, abs=1e-6)


def test_lower_bound_violated() -> None:
    """Atoms closer than d_lower — penalty > 0."""
    coords = _coords()  # atom 0 at x=0, atom 2 at x=2 → d=2.0
    loss = NOELoss(
        atom_pairs=jnp.array([[0, 2]]),
        d_upper=jnp.array([8.0]),
        d_lower=jnp.array([5.0]),  # d_lower=5, violation of 3.0
    )
    val = float(loss(None, coords))
    # Expected: force_const * (5-2)^2 / 1 = 9.0
    assert val == pytest.approx(9.0, rel=1e-3)


def test_no_lower_bound_below_upper_is_zero() -> None:
    """Without d_lower, distance below d_upper contributes nothing."""
    coords = _coords()  # atom 0 at x=0, atom 1 at x=1 → d=1.0
    loss = NOELoss(
        atom_pairs=jnp.array([[0, 1]]),
        d_upper=jnp.array([5.0]),  # not violated
    )
    assert float(loss(None, coords)) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Multiple restraints, partial violations
# ---------------------------------------------------------------------------


def test_multiple_restraints_partial_violation() -> None:
    """Only violated restraints contribute; non-violated ones contribute 0."""
    coords = _coords()
    # Pair (0,3): d=3.0, d_upper=5.0 → no violation
    # Pair (0,7): d=7.0, d_upper=4.0 → violation of 3.0
    loss = NOELoss(
        atom_pairs=jnp.array([[0, 3], [0, 7]]),
        d_upper=jnp.array([5.0, 4.0]),
        force_const=1.0,
    )
    # Mean over 2: (0 + 3^2) / 2 = 4.5
    assert float(loss(None, coords)) == pytest.approx(4.5, rel=1e-4)


def test_force_const_scaling() -> None:
    """Doubling force_const doubles the loss."""
    coords = _coords()
    loss1 = NOELoss(
        atom_pairs=jnp.array([[0, 6]]),
        d_upper=jnp.array([4.0]),
        force_const=1.0,
    )
    loss2 = NOELoss(
        atom_pairs=jnp.array([[0, 6]]),
        d_upper=jnp.array([4.0]),
        force_const=2.0,
    )
    assert float(loss2(None, coords)) == pytest.approx(
        2.0 * float(loss1(None, coords)), rel=1e-5
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_count_violations_none() -> None:
    coords = _coords()  # all within 5 Å of 0
    loss = NOELoss(
        atom_pairs=jnp.array([[0, 2], [0, 3]]),
        d_upper=jnp.array([5.0, 5.0]),
    )
    v = loss.count_violations(coords)
    assert v == {"upper": 0, "lower": 0, "total": 0}


def test_count_violations_upper_only() -> None:
    coords = _coords()  # atom 0 to 7: d=7 > 4
    loss = NOELoss(
        atom_pairs=jnp.array([[0, 7], [0, 2]]),
        d_upper=jnp.array([4.0, 5.0]),  # first violated, second not
    )
    v = loss.count_violations(coords)
    assert v == {"upper": 1, "lower": 0, "total": 1}


def test_rms_violation_zero_when_satisfied() -> None:
    coords = _coords()
    loss = NOELoss(
        atom_pairs=jnp.array([[0, 3]]),
        d_upper=jnp.array([5.0]),
    )
    assert loss.rms_violation(coords) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def test_make_noe_restraints_basic() -> None:
    res_ids = np.array([1, 2, 3, 4, 5])
    noe_list = [
        {"res_i": 1, "atom_i": "CA", "res_j": 5, "atom_j": "CA",
         "d_upper": 8.0, "d_lower": 1.8},
    ]
    loss = make_noe_restraints(noe_list, res_ids)
    assert loss.n_restraints == 1
    assert float(loss.d_upper[0]) == pytest.approx(8.0)
    assert loss.d_lower is not None
    assert float(loss.d_lower[0]) == pytest.approx(1.8)


def test_make_noe_restraints_unknown_residue_raises() -> None:
    res_ids = np.array([1, 2, 3])
    noe_list = [
        {"res_i": 99, "atom_i": "CA", "res_j": 1, "atom_j": "CA", "d_upper": 5.0}
    ]
    with pytest.raises(ValueError, match="residue 99 not found"):
        make_noe_restraints(noe_list, res_ids)


def test_make_noe_restraints_unknown_atom_raises() -> None:
    res_ids = np.array([1, 2, 3])
    noe_list = [
        {"res_i": 1, "atom_i": "CB", "res_j": 2, "atom_j": "CA", "d_upper": 5.0}
    ]
    with pytest.raises(ValueError, match="atom name 'CB' not in"):
        make_noe_restraints(noe_list, res_ids)


def test_make_noe_restraints_no_lower_bound() -> None:
    """When no d_lower is specified, d_lower should be None."""
    res_ids = np.array([1, 2])
    noe_list = [
        {"res_i": 1, "atom_i": "CA", "res_j": 2, "atom_j": "N", "d_upper": 5.0},
    ]
    loss = make_noe_restraints(noe_list, res_ids)
    assert loss.d_lower is None


def test_make_noe_restraints_correct_atom_indices() -> None:
    """Atom index for CA of residue 3 (block 2) should be 2*3+1=7."""
    res_ids = np.array([1, 2, 3])  # blocks 0,1,2  → atoms 0-8
    noe_list = [
        {"res_i": 3, "atom_i": "CA", "res_j": 1, "atom_j": "N", "d_upper": 6.0}
    ]
    loss = make_noe_restraints(noe_list, res_ids, atom_names=["N", "CA", "C"])
    # residue 3 is at block index 2 → CA offset 1 → atom_index = 2*3+1 = 7
    assert int(loss.atom_pairs[0, 0]) == 7
    # residue 1 is at block index 0 → N offset 0 → atom_index = 0*3+0 = 0
    assert int(loss.atom_pairs[0, 1]) == 0
