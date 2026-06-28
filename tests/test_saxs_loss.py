import jax.numpy as jnp
from jax import grad

from diff_integrator.terms.saxs import SAXSLoss


def test_saxs_loss_basic():
    # Dummy coords
    coords = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.0, 1.5, 0.0],
        ]
    )

    q_values = jnp.linspace(0.01, 0.3, 10)
    # Dummy form factors (N, M)
    form_factors = jnp.ones((3, 10))
    # Dummy volumes
    volumes = jnp.array([10.0, 10.0, 10.0])

    # Let's create an experimental intensity array using the same coords
    # so we can test if the loss is close to 0 when coords are perfect.
    from diff_biophys.saxs.kernels import debye_saxs

    exp_intensities = debye_saxs(coords, q_values, form_factors, volumes=volumes)

    saxs_loss = SAXSLoss(
        q_values=q_values,
        exp_intensities=exp_intensities,
        form_factors=form_factors,
        volumes=volumes,
    )

    # Calculate loss with perfect coords
    loss_val = saxs_loss(None, coords)
    assert jnp.isclose(loss_val, 0.0, atol=1e-5)

    # Perturb coords non-translationally
    perturbed_coords = coords.at[0].add(0.5)
    loss_val_perturbed = saxs_loss(None, perturbed_coords)
    assert loss_val_perturbed > 1e-4


def test_saxs_loss_gradients():
    coords = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.0, 1.5, 0.0],
        ]
    )

    q_values = jnp.linspace(0.01, 0.3, 10)
    form_factors = jnp.ones((3, 10))
    exp_intensities = jnp.ones(10)

    saxs_loss = SAXSLoss(
        q_values=q_values,
        exp_intensities=exp_intensities,
        form_factors=form_factors,
    )

    # Check that gradient is computable and shape is correct
    grad_fn = grad(lambda c: saxs_loss(None, c))
    gradients = grad_fn(coords)

    assert gradients.shape == coords.shape
    assert not jnp.any(jnp.isnan(gradients))


def test_saxs_loss_no_scaling():
    coords = jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    q_values = jnp.array([0.1])
    form_factors = jnp.ones((2, 1))
    exp_intensities = jnp.array([1.0])
    saxs_loss = SAXSLoss(
        q_values=q_values,
        exp_intensities=exp_intensities,
        form_factors=form_factors,
        scale_mode="none",
    )
    loss_val = saxs_loss(None, coords)
    assert not jnp.isnan(loss_val)


def test_saxs_loss_invalid_scale_mode_raises():
    """SAXSLoss must raise ValueError when given an unrecognised scale_mode."""
    import pytest  # noqa: PLC0415
    jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    q_values = jnp.array([0.1])
    form_factors = jnp.ones((2, 1))
    exp_intensities = jnp.array([1.0])
    with pytest.raises(ValueError, match="scale_mode"):
        SAXSLoss(
            q_values=q_values,
            exp_intensities=exp_intensities,
            form_factors=form_factors,
            scale_mode="log",  # not a valid mode
        )


def test_saxs_loss_name_attribute():
    """SAXSLoss must expose a non-empty name for per_term_history keying."""
    from diff_integrator.terms.saxs import SAXSLoss as _SAXSLoss  # noqa: PLC0415
    assert _SAXSLoss.name != ""
    assert _SAXSLoss.name == "saxs"
