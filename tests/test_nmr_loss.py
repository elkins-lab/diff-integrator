import jax.numpy as jnp
from jax import grad

from diff_integrator.terms.nmr import FixedTensorRDCLoss, RDCLoss


def test_rdc_loss():
    # Coords of 4 atoms (2 bonds)
    coords = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],  # Bond 1 along Z
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],  # Bond 2 along X
        ]
    )

    # Bond pairs (atom_a, atom_b)
    atom_pairs = jnp.array(
        [
            [0, 1],
            [2, 3],
        ]
    )

    # Experimental RDCs
    # Assume some specific saupe tensor that gives RDCs for Z and X bonds.
    # Let's say Szz = 10, Sxx = -5 (these aren't actual tensor values, just dummy)
    exp_rdcs = jnp.array([10.0, -5.0])

    loss_fn = RDCLoss(
        atom_pairs=atom_pairs,
        exp_rdcs=exp_rdcs,
        d_max=1.0,  # arbitrary
    )

    loss_val = loss_fn(None, coords)
    assert not jnp.isnan(loss_val)

    grad_fn = grad(lambda c: loss_fn(None, c))
    gradients = grad_fn(coords)

    assert gradients.shape == coords.shape
    assert not jnp.any(jnp.isnan(gradients))


def test_rdc_loss_q_factor():
    coords = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
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
    import pytest
    with pytest.raises(ValueError):
        loss_fn(None, coords)


def test_fixed_tensor_rdc_loss_uninitialized():
    coords = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    def dummy_loss(coords, tensor): return jnp.array(0.0)
    def dummy_make(coords): return jnp.array(0.0)
    loss_fn = FixedTensorRDCLoss(loss_fn=dummy_loss, make_tensor_fn=dummy_make)
    
    import pytest
    with pytest.raises(RuntimeError, match="tensor not initialized"):
        loss_fn(None, coords)

