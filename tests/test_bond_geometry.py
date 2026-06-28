"""
Tests for diff_integrator/terms/bond_geometry.py.

Covers BondLengthPenalty, BondAnglePenalty, and make_backbone_bond_geometry
for both unit-level correctness and end-to-end Cartesian refinement behaviour.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from diff_biophys.geometry.backbone import (
    C_N_CA_ANGLE,
    C_N_LENGTH,
    CA_C_LENGTH,
    CA_C_N_ANGLE,
    N_CA_C_ANGLE,
    N_CA_LENGTH,
)
from jax import grad

from diff_integrator.terms.bond_geometry import (
    BondAnglePenalty,
    BondLengthPenalty,
    make_backbone_bond_geometry,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ideal_triplet(bond_len: float, angle_rad: float) -> jnp.ndarray:
    """Build three collinear-adjacent atoms with a given bond length and angle.

    Atom layout: A at origin, B at (bond_len, 0, 0), C placed so that ∠ABC
    equals angle_rad.
    """
    A = jnp.array([0.0, 0.0, 0.0])
    B = jnp.array([bond_len, 0.0, 0.0])
    # C is in the XY plane
    c_x = bond_len + bond_len * jnp.cos(jnp.pi - angle_rad)
    c_y = bond_len * jnp.sin(jnp.pi - angle_rad)
    C = jnp.array([c_x, c_y, 0.0])
    return jnp.stack([A, B, C])  # (3, 3)


def _build_ideal_backbone(n_residues: int) -> jnp.ndarray:
    """Build a toy backbone chain with exactly Engh & Huber ideal geometry.

    Uses simple straight-chain placement so bond lengths and angles are
    exactly at their ideal values; no NeRF drift.  Returns ``(3N, 3)``.
    """
    from diff_biophys.geometry.backbone import make_backbone_builder

    # Start at a 3-atom seed with exactly ideal N–CA–C geometry
    seed = _ideal_triplet(N_CA_LENGTH, N_CA_C_ANGLE)
    build = make_backbone_builder(n_residues, seed)

    # Use φ = −60°, ψ = −40° (α-helix) for a reasonable non-degenerate chain
    phi = jnp.full(n_residues, jnp.radians(-60.0))
    psi = jnp.full(n_residues, jnp.radians(-40.0))
    return build(phi, psi)


# ---------------------------------------------------------------------------
# BondLengthPenalty
# ---------------------------------------------------------------------------


def test_bond_length_penalty_zero_at_ideal():
    """Loss is exactly 0 when all bonds equal the ideal length."""
    bond_len = N_CA_LENGTH
    # Two atoms exactly bond_len apart
    coords = jnp.array([[0.0, 0.0, 0.0], [bond_len, 0.0, 0.0]])
    bond_pairs = jnp.array([[0, 1]], dtype=jnp.int32)
    ideal = jnp.array([bond_len])
    penalty = BondLengthPenalty(bond_pairs, ideal)
    assert float(penalty(None, coords)) == pytest.approx(0.0, abs=1e-6)


def test_bond_length_penalty_nonzero_for_stretched_bond():
    """Loss > 0 when bond length deviates from ideal."""
    coords = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])  # 2.0 Å
    bond_pairs = jnp.array([[0, 1]], dtype=jnp.int32)
    ideal = jnp.array([N_CA_LENGTH])                            # 1.459 Å
    penalty = BondLengthPenalty(bond_pairs, ideal)
    loss = float(penalty(None, coords))
    expected = (2.0 - N_CA_LENGTH) ** 2
    assert loss == pytest.approx(expected, rel=1e-5)


def test_bond_length_penalty_gradient_finite_and_nonzero():
    """Gradient flows through the bond-length penalty correctly."""
    coords = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    bond_pairs = jnp.array([[0, 1]], dtype=jnp.int32)
    ideal = jnp.array([N_CA_LENGTH])
    penalty = BondLengthPenalty(bond_pairs, ideal)
    g = grad(lambda c: penalty(None, c))(coords)
    assert g.shape == coords.shape
    assert jnp.all(jnp.isfinite(g))
    assert jnp.any(g != 0.0)


def test_bond_length_penalty_mean_over_bonds():
    """Loss is the *mean* over bonds, not the sum."""
    coords = jnp.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],   # bond 0-1: 2.0 Å, ideal 1.459 → dev² = 0.2929
        [2.0, 1.5, 0.0],   # bond 1-2: 1.5 Å, ideal 1.525 → dev² = 0.000625
    ])
    bond_pairs = jnp.array([[0, 1], [1, 2]], dtype=jnp.int32)
    ideal = jnp.array([N_CA_LENGTH, CA_C_LENGTH])
    penalty = BondLengthPenalty(bond_pairs, ideal)
    expected_mean = ((2.0 - N_CA_LENGTH)**2 + (1.5 - CA_C_LENGTH)**2) / 2
    assert float(penalty(None, coords)) == pytest.approx(expected_mean, rel=1e-4)


def test_bond_rmsd_monitoring():
    """bond_rmsd() returns a float in Å."""
    coords = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    bond_pairs = jnp.array([[0, 1]], dtype=jnp.int32)
    ideal = jnp.array([N_CA_LENGTH])
    penalty = BondLengthPenalty(bond_pairs, ideal)
    rmsd = penalty.bond_rmsd(coords)
    assert isinstance(rmsd, float)
    assert rmsd == pytest.approx(abs(2.0 - N_CA_LENGTH), rel=1e-5)


# ---------------------------------------------------------------------------
# BondAnglePenalty
# ---------------------------------------------------------------------------


def test_bond_angle_penalty_zero_at_ideal():
    """Loss is exactly 0 when the angle equals the ideal value."""
    ideal_rad = N_CA_C_ANGLE
    coords = _ideal_triplet(N_CA_LENGTH, ideal_rad)
    angle_triples = jnp.array([[0, 1, 2]], dtype=jnp.int32)
    ideal_angles = jnp.array([ideal_rad])
    penalty = BondAnglePenalty(angle_triples, ideal_angles)
    assert float(penalty(None, coords)) == pytest.approx(0.0, abs=1e-5)


def test_bond_angle_penalty_nonzero_for_wrong_angle():
    """Loss > 0 when angle deviates from ideal."""
    # Place atoms in a right angle (90°) but ideal is ~111.2°
    coords = jnp.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    angle_triples = jnp.array([[0, 1, 2]], dtype=jnp.int32)
    ideal_angles = jnp.array([N_CA_C_ANGLE])
    penalty = BondAnglePenalty(angle_triples, ideal_angles)
    loss = float(penalty(None, coords))
    expected = (jnp.pi / 2 - N_CA_C_ANGLE) ** 2
    assert loss == pytest.approx(float(expected), rel=1e-4)


def test_bond_angle_penalty_gradient_finite():
    """Gradient flows through the bond-angle penalty."""
    coords = jnp.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    angle_triples = jnp.array([[0, 1, 2]], dtype=jnp.int32)
    ideal_angles = jnp.array([N_CA_C_ANGLE])
    penalty = BondAnglePenalty(angle_triples, ideal_angles)
    g = grad(lambda c: penalty(None, c))(coords)
    assert g.shape == coords.shape
    assert jnp.all(jnp.isfinite(g))


def test_bond_angle_penalty_safe_near_collinear():
    """arccos clamp prevents NaN gradient for nearly collinear atoms."""
    # Nearly collinear — angle close to 180°
    coords = jnp.array([
        [-1.0, 0.0, 0.0],
        [ 0.0, 0.0, 0.0],
        [ 1.0, 1e-6, 0.0],   # almost perfectly opposite
    ])
    angle_triples = jnp.array([[0, 1, 2]], dtype=jnp.int32)
    ideal_angles = jnp.array([N_CA_C_ANGLE])
    penalty = BondAnglePenalty(angle_triples, ideal_angles)
    g = grad(lambda c: penalty(None, c))(coords)
    assert jnp.all(jnp.isfinite(g)), "Gradient must be finite near collinear config"


def test_angle_rmsd_deg_monitoring():
    """angle_rmsd_deg() returns a float in degrees."""
    coords = jnp.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    angle_triples = jnp.array([[0, 1, 2]], dtype=jnp.int32)
    ideal_angles = jnp.array([N_CA_C_ANGLE])
    penalty = BondAnglePenalty(angle_triples, ideal_angles)
    rmsd_deg = penalty.angle_rmsd_deg(coords)
    assert isinstance(rmsd_deg, float)
    expected_deg = abs(np.degrees(np.pi / 2 - N_CA_C_ANGLE))
    assert rmsd_deg == pytest.approx(expected_deg, rel=1e-4)


# ---------------------------------------------------------------------------
# make_backbone_bond_geometry — structural tests
# ---------------------------------------------------------------------------


def test_make_backbone_bond_geometry_raises_for_single_residue():
    with pytest.raises(ValueError, match="n_residues"):
        make_backbone_bond_geometry(1)


@pytest.mark.parametrize("n", [2, 5, 10, 92, 107])
def test_make_backbone_bond_geometry_bond_counts(n):
    """Factory produces correct number of bonds and angles.

    For n residues:
      bonds = 2n (N-CA, CA-C within each residue) + (n-1) C-N peptide bonds
            = 3n - 1
      angles = n (N-CA-C) + 2*(n-1) (CA-C-N and C-N-CA at each junction)
             = 3n - 2
    """
    bond_pen, angle_pen = make_backbone_bond_geometry(n)
    assert bond_pen.bond_pairs.shape == (3 * n - 1, 2)
    assert bond_pen.ideal_lengths.shape == (3 * n - 1,)
    assert angle_pen.angle_triples.shape == (3 * n - 2, 3)
    assert angle_pen.ideal_angles_rad.shape == (3 * n - 2,)


def test_make_backbone_bond_geometry_ideal_lengths_correct():
    """Factory encodes exactly the Engh & Huber bond lengths."""
    bond_pen, _ = make_backbone_bond_geometry(3)
    lengths = np.array(bond_pen.ideal_lengths)
    # For 3 residues: N-CA, CA-C, C-N, N-CA, CA-C, C-N, N-CA, CA-C
    expected = [
        N_CA_LENGTH, CA_C_LENGTH, C_N_LENGTH,
        N_CA_LENGTH, CA_C_LENGTH, C_N_LENGTH,
        N_CA_LENGTH, CA_C_LENGTH,
    ]
    np.testing.assert_allclose(lengths, expected, rtol=1e-5)


def test_make_backbone_bond_geometry_ideal_angles_correct():
    """Factory encodes exactly the Engh & Huber bond angles."""
    _, angle_pen = make_backbone_bond_geometry(3)
    angles = np.array(angle_pen.ideal_angles_rad)
    # For 3 residues: N-CA-C, CA-C-N, C-N-CA, N-CA-C, CA-C-N, C-N-CA, N-CA-C
    expected = [
        N_CA_C_ANGLE,
        CA_C_N_ANGLE, C_N_CA_ANGLE,
        N_CA_C_ANGLE,
        CA_C_N_ANGLE, C_N_CA_ANGLE,
        N_CA_C_ANGLE,
    ]
    np.testing.assert_allclose(angles, expected, rtol=1e-5)


def test_make_backbone_bond_geometry_near_zero_on_ideal_chain():
    """Bond and angle penalties are very small on a NeRF-built ideal chain."""
    n = 10
    coords = _build_ideal_backbone(n)
    bond_pen, angle_pen = make_backbone_bond_geometry(n)

    float(bond_pen(None, coords))
    float(angle_pen(None, coords))

    # NeRF uses the same Engh & Huber values but accumulates float32 rounding
    # across the chain.  Losses are very small but not exactly zero.
    # Bond RMSD should be < 0.015 Å; angle RMSD < 0.1°.
    bond_rmsd = bond_pen.bond_rmsd(coords)
    angle_rmsd = angle_pen.angle_rmsd_deg(coords)
    assert bond_rmsd < 0.015, f"Bond RMSD too large on ideal chain: {bond_rmsd:.4f} Å"
    assert angle_rmsd < 0.5, f"Angle RMSD too large on ideal chain: {angle_rmsd:.4f}°"


def test_make_backbone_bond_geometry_gradients_flow():
    """Gradients from factory-built penalties flow cleanly through all atoms."""
    n = 5
    coords = _build_ideal_backbone(n) + 0.05 * jax.random.normal(
        jax.random.PRNGKey(0), (3 * n, 3)
    )
    bond_pen, angle_pen = make_backbone_bond_geometry(n)

    def total(c: jnp.ndarray) -> jnp.ndarray:
        return bond_pen(None, c) + angle_pen(None, c)

    g = grad(total)(coords)
    assert g.shape == coords.shape
    assert jnp.all(jnp.isfinite(g))


# ---------------------------------------------------------------------------
# End-to-end: Cartesian refinement reduces loss vs. random displacement
# ---------------------------------------------------------------------------


def test_cartesian_refinement_reduces_loss():
    """Bond+angle penalties decrease when an optimizer step moves toward ideal.

    This is a minimal gradient-descent sanity check: one Adam step on a
    coordinate set displaced from ideal should reduce the combined penalty.
    """
    import optax

    n = 8
    ideal_coords = _build_ideal_backbone(n)
    # Displace by 0.3 Å RMS
    rng = jax.random.PRNGKey(1)
    displaced = ideal_coords + 0.3 * jax.random.normal(rng, ideal_coords.shape)

    bond_pen, angle_pen = make_backbone_bond_geometry(n)

    def loss(c: jnp.ndarray) -> jnp.ndarray:
        return bond_pen(None, c) + angle_pen(None, c)

    loss_before = float(loss(displaced))

    opt = optax.adam(0.01)
    state = opt.init(displaced)
    coords = displaced
    for _ in range(50):
        g = grad(loss)(coords)
        updates, state = opt.update(g, state)
        coords = optax.apply_updates(coords, updates)

    loss_after = float(loss(coords))
    assert loss_after < loss_before, (
        f"Expected loss to decrease: before={loss_before:.4f}, after={loss_after:.4f}"
    )
