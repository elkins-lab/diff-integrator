import jax.numpy as jnp
from jax import grad

from diff_integrator.terms.ramachandran import RamachandranLoss


def test_ramachandran_loss_alpha_helix_low_penalty():
    """phi/psi sitting exactly on the alpha-helix centre should be near zero."""
    phi = jnp.array([-1.05])
    psi = jnp.array([-0.78])

    loss = RamachandranLoss()
    penalty = loss((phi, psi), jnp.zeros((1, 3)))

    assert jnp.isclose(penalty, 0.0, atol=1e-3)


def test_ramachandran_loss_forbidden_high_penalty():
    """phi=0, psi=0 is far from all three basins and should be penalised."""
    phi = jnp.array([0.0])
    psi = jnp.array([0.0])

    loss = RamachandranLoss()
    penalty = loss((phi, psi), jnp.zeros((1, 3)))

    assert penalty > 0.1


def test_ramachandran_loss_ordering():
    """Alpha-helix penalty must be strictly less than forbidden-region penalty."""
    loss = RamachandranLoss()

    alpha_penalty = loss(
        (jnp.array([-1.05]), jnp.array([-0.78])),
        jnp.zeros((1, 3)),
    )
    forbidden_penalty = loss(
        (jnp.array([0.0]), jnp.array([0.0])),
        jnp.zeros((1, 3)),
    )

    assert alpha_penalty < forbidden_penalty


def test_ramachandran_loss_gradient():
    """Gradients through the Ramachandran loss must be finite and non-NaN."""
    loss = RamachandranLoss()

    def fn(phi: jnp.ndarray) -> jnp.ndarray:
        return loss((phi, jnp.array([0.0, -0.78])), jnp.zeros((2, 3)))

    phi = jnp.array([-1.05, 0.5])
    grads = grad(fn)(phi)

    assert grads.shape == phi.shape
    assert jnp.all(jnp.isfinite(grads))


def test_ramachandran_loss_custom_sigma():
    """Changing sigma must change the computed penalty."""
    phi = jnp.array([0.0, 0.5])
    psi = jnp.array([0.0, -1.0])
    coords = jnp.zeros((2, 3))

    loss_default = RamachandranLoss(sigma=0.5)
    loss_small = RamachandranLoss(sigma=0.1)

    val_default = loss_default((phi, psi), coords)
    val_small = loss_small((phi, psi), coords)

    assert not jnp.isclose(val_default, val_small)
