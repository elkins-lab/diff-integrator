import jax.numpy as jnp
import numpy as np
from jax import grad

from diff_integrator.terms.chemical_shifts import CartesianCAShiftLoss, CAShiftLoss


def test_ca_shift_loss() -> None:
    # Setup mock data for 3 residues
    exp_res_ids = np.array([1, 2, 3])
    exp_shifts = np.array([55.0, 56.0, 57.0])
    struct_res_ids = np.array([1, 2, 3])
    struct_res_names = ["ALA", "VAL", "LEU"]

    # Initialize loss
    ca_loss = CAShiftLoss(exp_res_ids, exp_shifts, struct_res_ids, struct_res_names)
    assert ca_loss.n_matched == 3

    # Setup dummy phi, psi (internal parameters)
    phi = jnp.array([-1.0, -1.0, -1.0])
    psi = jnp.array([-0.8, -0.8, -0.8])
    params = (phi, psi)

    # Coordinates are ignored
    coords = jnp.zeros((3, 3))

    # Evaluate loss
    loss_val = ca_loss(params, coords)
    assert loss_val.shape == ()
    assert float(loss_val) >= 0.0


def test_ca_shift_loss_name_attribute():
    """CAShiftLoss must expose name='ca_shift'."""
    assert CAShiftLoss.name == "ca_shift"


# ---------------------------------------------------------------------------
# CartesianCAShiftLoss
# ---------------------------------------------------------------------------


def _make_cartesian_shift_loss() -> CartesianCAShiftLoss:
    """Build a CartesianCAShiftLoss for 5 residues with synthetic data."""
    exp_res_ids = np.array([1, 2, 3, 4, 5])
    exp_shifts = np.array([55.0, 56.0, 57.0, 58.0, 59.0])
    struct_res_ids = np.array([1, 2, 3, 4, 5])
    struct_res_names = ["ALA", "GLY", "VAL", "LEU", "ILE"]
    return CartesianCAShiftLoss(exp_res_ids, exp_shifts, struct_res_ids, struct_res_names)


def _build_straight_backbone(n_residues: int) -> jnp.ndarray:
    """Build a minimal N-CA-C backbone using the NeRF builder.

    Uses a proper non-collinear seed (ideal N-CA-C angle) so that
    ``compute_phi_psi`` returns non-trivial dihedral angles and the JAX
    gradient is non-zero.
    """
    import math  # noqa: PLC0415

    from diff_biophys.geometry.backbone import (  # noqa: PLC0415
        CA_C_LENGTH,
        N_CA_C_ANGLE,
        N_CA_LENGTH,
        make_backbone_builder,
    )

    # N at origin, CA along x-axis, C placed so angle N-CA-C = N_CA_C_ANGLE
    n_pos = jnp.array([0.0, 0.0, 0.0])
    ca_pos = jnp.array([N_CA_LENGTH, 0.0, 0.0])
    cx = N_CA_LENGTH + CA_C_LENGTH * math.cos(math.pi - N_CA_C_ANGLE)
    cy = CA_C_LENGTH * math.sin(math.pi - N_CA_C_ANGLE)
    c_pos = jnp.array([cx, cy, 0.0])
    seed_coords = jnp.stack([n_pos, ca_pos, c_pos])

    build = make_backbone_builder(n_residues, seed_coords)
    phi = jnp.full(n_residues, jnp.radians(-60.0))
    psi = jnp.full(n_residues, jnp.radians(-40.0))
    return build(phi, psi)


def test_cartesian_ca_shift_loss_scalar_output():
    """CartesianCAShiftLoss must return a scalar (shape ())."""
    loss_term = _make_cartesian_shift_loss()
    coords = _build_straight_backbone(5)
    result = loss_term(coords, coords)  # params == coords in Cartesian mode
    assert result.shape == ()


def test_cartesian_ca_shift_loss_nonnegative():
    """CartesianCAShiftLoss (RMSD) must be non-negative."""
    loss_term = _make_cartesian_shift_loss()
    coords = _build_straight_backbone(5)
    result = float(loss_term(coords, coords))
    assert result >= 0.0


def test_cartesian_ca_shift_loss_gradient_flows():
    """Gradient must flow through compute_phi_psi(coords) -> shift_loss.

    This is the critical chain: CartesianCAShiftLoss calls compute_phi_psi(coords)
    internally, so the gradient path is loss -> phi/psi(coords) -> coords.
    If JAX cannot differentiate through compute_phi_psi the gradient will be
    all-zero or NaN.
    """
    loss_term = _make_cartesian_shift_loss()
    coords = _build_straight_backbone(5)

    def fn(c: jnp.ndarray) -> jnp.ndarray:
        return loss_term(c, c)  # params == coords in Cartesian mode

    g = grad(fn)(coords)
    assert g.shape == coords.shape
    assert jnp.all(jnp.isfinite(g)), "Gradient contains NaN or Inf"
    # Gradient should not be identically zero (loss depends on coords)
    assert jnp.any(g != 0.0), "Gradient is all-zero — chain may be broken"


def test_cartesian_ca_shift_loss_gradient_finite_on_perturbed_coords():
    """Gradient must remain finite after a small random perturbation."""
    import jax  # noqa: PLC0415

    loss_term = _make_cartesian_shift_loss()
    coords = _build_straight_backbone(5)
    perturbed = coords + 0.1 * jax.random.normal(jax.random.PRNGKey(42), coords.shape)

    g = grad(lambda c: loss_term(c, c))(perturbed)
    assert jnp.all(jnp.isfinite(g))


def test_cartesian_ca_shift_loss_name_attribute():
    """CartesianCAShiftLoss must expose a distinct name for per_term_history keying."""
    assert CartesianCAShiftLoss.name == "ca_shift_cartesian"
    assert CartesianCAShiftLoss.name != CAShiftLoss.name
