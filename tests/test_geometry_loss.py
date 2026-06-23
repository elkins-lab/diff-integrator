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
