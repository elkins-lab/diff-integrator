import jax.numpy as jnp
import numpy as np

from diff_integrator.terms.chemical_shifts import CAShiftLoss


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
