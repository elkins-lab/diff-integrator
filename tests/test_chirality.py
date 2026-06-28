"""
tests/test_chirality.py — Tests for diff_integrator.terms.chirality.ChiralityPenalty.
"""

import jax
import jax.numpy as jnp
import pytest

from diff_integrator.terms.chirality import ChiralityPenalty, make_backbone_chirality

# ---------------------------------------------------------------------------
# Helpers: build minimal 2-residue backbones with known chirality
# ---------------------------------------------------------------------------
#
# Standard N-CA-C layout: atom index = 3*r + {0:N, 1:CA, 2:C}
# 2-residue chain: N0(0) CA0(1) C0(2) N1(3) CA1(4) C1(5)
#
# ChiralityPenalty covers *interior* residues (r>=1, needs C_{r-1}).
# For r=1: CA=4, N=3, C=5, C_prev=2.
#
# L-amino acid: chi = dot(cross(N-CA, C-CA), C_prev-CA) < 0
# We build coords that produce a known chi value.


def _l_amino_coords() -> jnp.ndarray:
    """Minimal 2-residue backbone with a clearly L-configured residue 1.

    Place atoms so that the triple product is comfortably negative:
      CA1 at origin, N1 along +x, C1 along +y → cross(N-CA, C-CA) = +z.
      C_prev along +z → chi = dot(+z, +z) = +positive ... wait that's D.

    Let's think carefully:
      u = N1 - CA1 = (1, 0, 0)
      v = C1 - CA1 = (0, 1, 0)
      cross(u, v) = (0, 0, 1)   (points in +z)
      w = C0 - CA1 → we want dot((0,0,1), w) < 0 → w must have negative z.
      Set C0 = (0, 0, -2) → chi = dot((0,0,1), (0,0,-2)) = -2.  ✓ L-amino acid.
    """
    coords = jnp.array([
        [0.0, 0.0,  0.0],  # 0: N0  (residue 0)
        [0.0, 0.0,  0.0],  # 1: CA0 (residue 0)
        [0.0, 0.0, -2.0],  # 2: C0  (residue 0) ← C_prev for residue 1
        [1.0, 0.0,  0.0],  # 3: N1  (residue 1)
        [0.0, 0.0,  0.0],  # 4: CA1 (residue 1) ← the center being tested
        [0.0, 1.0,  0.0],  # 5: C1  (residue 1)
    ])
    return coords


def _d_amino_coords() -> jnp.ndarray:
    """Same geometry but with C_prev flipped to +z → chi = +2 (D-amino acid)."""
    coords = jnp.array([
        [0.0, 0.0,  0.0],  # 0: N0
        [0.0, 0.0,  0.0],  # 1: CA0
        [0.0, 0.0,  2.0],  # 2: C0  ← C_prev now at +z → D-configuration
        [1.0, 0.0,  0.0],  # 3: N1
        [0.0, 0.0,  0.0],  # 4: CA1
        [0.0, 1.0,  0.0],  # 5: C1
    ])
    return coords


def _make_pen(margin: float = 0.1) -> ChiralityPenalty:
    """ChiralityPenalty for interior residue 1 in a 2-residue chain."""
    return ChiralityPenalty(
        ca_indices=jnp.array([4]),
        n_indices=jnp.array([3]),
        c_indices=jnp.array([5]),
        cprev_indices=jnp.array([2]),
        margin=margin,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_n_centers_property() -> None:
    pen = _make_pen()
    assert pen.n_centers == 1


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="n_indices must have shape"):
        ChiralityPenalty(
            ca_indices=jnp.array([4, 7]),
            n_indices=jnp.array([3]),        # wrong: need 2 elements
            c_indices=jnp.array([5, 8]),
            cprev_indices=jnp.array([2, 5]),
        )


# ---------------------------------------------------------------------------
# Chi values
# ---------------------------------------------------------------------------


def test_chi_value_l_config() -> None:
    """L-configured residue should have chi < 0."""
    pen = _make_pen()
    coords = _l_amino_coords()
    chi = pen.chi_values(coords)
    assert float(chi[0]) < 0.0


def test_chi_value_d_config() -> None:
    """D-configured residue should have chi > 0."""
    pen = _make_pen()
    coords = _d_amino_coords()
    chi = pen.chi_values(coords)
    assert float(chi[0]) > 0.0


# ---------------------------------------------------------------------------
# Penalty values
# ---------------------------------------------------------------------------


def test_l_config_zero_penalty() -> None:
    """Well-folded L-amino acid (chi = -2, margin = 0.1) → penalty = 0."""
    pen = _make_pen(margin=0.1)
    coords = _l_amino_coords()
    # chi = -2.0;  chi + margin = -1.9 < 0 → max(0, -1.9)^2 = 0
    assert float(pen(None, coords)) == pytest.approx(0.0, abs=1e-7)


def test_d_config_positive_penalty() -> None:
    """D-configured residue (chi = +2) → penalty = (2 + margin)^2."""
    margin = 0.1
    pen = _make_pen(margin=margin)
    coords = _d_amino_coords()
    val = float(pen(None, coords))
    expected = (2.0 + margin) ** 2  # one residue, mean over 1
    assert val == pytest.approx(expected, rel=1e-5)


def test_penalty_exactly_at_margin() -> None:
    """chi = -margin → violation = max(0, 0) = 0 → penalty = 0."""
    margin = 1.0
    # Need chi = -margin = -1.0.
    # With u=(1,0,0), v=(0,1,0), cross=(0,0,1):
    # chi = dot((0,0,1), w).  Want chi = -1.0 → w = (0,0,-1).
    coords = jnp.array([
        [0.0, 0.0,  0.0],  # N0
        [0.0, 0.0,  0.0],  # CA0
        [0.0, 0.0, -1.0],  # C0 → chi = -1.0 = -margin
        [1.0, 0.0,  0.0],  # N1
        [0.0, 0.0,  0.0],  # CA1
        [0.0, 1.0,  0.0],  # C1
    ])
    pen = _make_pen(margin=margin)
    assert float(pen(None, coords)) == pytest.approx(0.0, abs=1e-7)


def test_penalty_just_inside_violation() -> None:
    """chi = -margin + epsilon → tiny positive penalty."""
    margin = 1.0
    eps = 0.01
    # chi = -margin + eps = -0.99  → violation = eps = 0.01
    coords = jnp.array([
        [0.0, 0.0,  0.0],
        [0.0, 0.0,  0.0],
        [0.0, 0.0,  -(margin - eps)],  # C_prev at -(margin-eps) on z
        [1.0, 0.0,  0.0],
        [0.0, 0.0,  0.0],
        [0.0, 1.0,  0.0],
    ])
    pen = _make_pen(margin=margin)
    val = float(pen(None, coords))
    expected = eps ** 2
    assert val == pytest.approx(expected, rel=1e-3)


# ---------------------------------------------------------------------------
# Gradient
# ---------------------------------------------------------------------------


def test_gradient_flows_for_violation() -> None:
    """Gradient should be nonzero for a D-configured residue."""
    pen = _make_pen()
    coords = _d_amino_coords()
    grad = jax.grad(lambda c: pen(None, c))(coords)
    # At least the CA atom (index 4) should have a nonzero gradient
    assert float(jnp.linalg.norm(grad[4])) > 0.0


def test_gradient_zero_for_l_config() -> None:
    """Gradient should be exactly zero for a well-satisfied L-amino acid."""
    pen = _make_pen(margin=0.1)
    coords = _l_amino_coords()
    grad = jax.grad(lambda c: pen(None, c))(coords)
    # All gradients should be zero (relu returns 0, gradient is 0 too)
    assert float(jnp.max(jnp.abs(grad))) == pytest.approx(0.0, abs=1e-7)


# ---------------------------------------------------------------------------
# Count violations
# ---------------------------------------------------------------------------


def test_count_violations_l_config() -> None:
    pen = _make_pen()
    coords = _l_amino_coords()
    assert pen.count_violations(coords) == 0


def test_count_violations_d_config() -> None:
    pen = _make_pen()
    coords = _d_amino_coords()
    assert pen.count_violations(coords) == 1


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def test_factory_n_residues_2() -> None:
    """2-residue chain → 1 interior residue monitored."""
    pen = make_backbone_chirality(n_residues=2)
    assert pen.n_centers == 1


def test_factory_n_residues_10() -> None:
    """10-residue chain → 9 interior residues monitored."""
    pen = make_backbone_chirality(n_residues=10)
    assert pen.n_centers == 9


def test_factory_n_residues_1_raises() -> None:
    with pytest.raises(ValueError, match="n_residues ≥ 2"):
        make_backbone_chirality(n_residues=1)


def test_factory_correct_indices() -> None:
    """For a 3-residue chain, residue 1 should use atoms 3(N), 4(CA), 5(C), 2(C_prev)."""
    pen = make_backbone_chirality(n_residues=3)
    # Residue 1: CA=3*1+1=4, N=3*1=3, C=3*1+2=5, C_prev=3*0+2=2
    assert int(pen.ca_indices[0])    == 4
    assert int(pen.n_indices[0])     == 3
    assert int(pen.c_indices[0])     == 5
    assert int(pen.cprev_indices[0]) == 2


def test_factory_l_amino_acid_zero_penalty() -> None:
    """L-configured backbone from make_backbone_chirality should yield zero penalty."""
    pen = make_backbone_chirality(n_residues=2)
    coords = _l_amino_coords()
    assert float(pen(None, coords)) == pytest.approx(0.0, abs=1e-7)


def test_factory_d_amino_acid_positive_penalty() -> None:
    """D-configured backbone should yield positive penalty."""
    pen = make_backbone_chirality(n_residues=2)
    coords = _d_amino_coords()
    assert float(pen(None, coords)) > 0.0
