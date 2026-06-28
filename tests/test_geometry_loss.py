import jax.numpy as jnp
from jax import grad

from diff_integrator.terms.geometry import GeometryLoss


def test_geometry_loss():
    coords = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],  # Bond is 2.0 A long
            [0.0, 1.0, 0.0],  # Atom 2 is 1.0 A away from Atom 0
        ]
    )

    bonds = jnp.array(
        [
            [0, 1],
        ]
    )
    ideal_bond_lengths = jnp.array([1.5])

    clash_pairs = jnp.array(
        [
            [0, 2],
        ]
    )
    min_clash_dists = jnp.array([2.0])

    geom_loss = GeometryLoss(
        bonds=bonds,
        ideal_bond_lengths=ideal_bond_lengths,
        clash_pairs=clash_pairs,
        min_clash_dists=min_clash_dists,
        bond_weight=1.0,
        clash_weight=1.0,
    )

    # Bond loss: (2.0 - 1.5)^2 = 0.25
    # Clash loss: Atom 0 and 2 distance is 1.0. Minimum is 2.0.
    # clash_loss = (2.0 - 1.0)^2 = 1.0
    # Total loss = 1.25

    loss_val = geom_loss(None, coords)
    assert jnp.isclose(loss_val, 1.25)

    # Check gradients
    grad_fn = grad(lambda c: geom_loss(None, c))
    gradients = grad_fn(coords)

    assert gradients.shape == coords.shape
    assert not jnp.any(jnp.isnan(gradients))


def test_geometry_loss_target_only():
    """GeometryLoss with only target_coords computes harmonic position restraint."""
    from jax import grad  # noqa: PLC0415

    target = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    current = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    geom = GeometryLoss(target_coords=target, target_weight=1.0)
    loss_val = geom(None, current)

    # mean over rows of sum-of-sq per row:
    # row0: (1^2+0+0)=1, row1: (0+1^2+0)=1 → mean = 1.0
    assert jnp.isclose(loss_val, 1.0)

    g = grad(lambda c: geom(None, c))(current)
    assert g.shape == current.shape
    assert jnp.all(jnp.isfinite(g))
    assert jnp.any(g != 0.0)


def test_geometry_loss_all_none_returns_zero():
    """GeometryLoss with no components active returns exactly 0."""
    geom = GeometryLoss()  # all optional args default to None
    coords = jnp.ones((3, 3))
    assert float(geom(None, coords)) == 0.0


def test_geometry_loss_name_attribute():
    """GeometryLoss must have a non-empty name so per_term_history keys are meaningful."""
    assert GeometryLoss.name != ""
    assert GeometryLoss().name == "geometry"
