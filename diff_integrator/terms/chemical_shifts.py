from typing import Any

import jax.numpy as jnp
import numpy as np
from diff_biophys.geometry.backbone import compute_phi_psi
from diff_biophys.nmr.chemical_shifts import make_ca_shift_loss

from diff_integrator.loss import LossTerm


class CAShiftLoss(LossTerm):
    """
    Computes the C-alpha Chemical Shift RMSD.
    Assumes `params` is a tuple of (phi, psi) dihedral angles.
    """

    name: str = "ca_shift"

    def __init__(
        self,
        exp_res_ids: np.ndarray,
        exp_shifts: np.ndarray,
        struct_res_ids: np.ndarray,
        struct_res_names: list[str],
    ) -> None:
        """
        Initialize the C-alpha shift loss.

        Args:
            exp_res_ids: Experimental residue IDs.
            exp_shifts: Experimental C-alpha shifts (ppm).
            struct_res_ids: Residue IDs in the structural model.
            struct_res_names: Residue names in the structural model.
        """
        self.loss_fn, self.n_matched = make_ca_shift_loss(
            exp_res_ids, exp_shifts, struct_res_ids, struct_res_names
        )

    def __call__(self, params: Any, coords: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate the chemical shift RMSD.

        Args:
            params: Tuple of (phi, psi) dihedral angles in radians.
            coords: Ignored for chemical shifts.

        Returns:
            Scalar jnp.ndarray representing the RMSD (ppm).
        """
        phi, psi = params
        return jnp.asarray(self.loss_fn(phi, psi))


class CartesianCAShiftLoss(LossTerm):
    """Cα chemical shift RMSD for Cartesian-coordinate refinement.

    In the Cartesian parameterisation, ``params`` are the backbone Cartesian
    coordinates ``(3N, 3)``.  This term extracts φ/ψ torsion angles from the
    current coordinates via :func:`~diff_biophys.geometry.backbone.compute_phi_psi`
    (which is differentiable) and then evaluates the chemical shift RMSD.

    This is the Cartesian-regime equivalent of :class:`CAShiftLoss`.

    Args:
        exp_res_ids: Experimental residue IDs.
        exp_shifts: Experimental Cα chemical shifts (ppm).
        struct_res_ids: Residue IDs in the structural model.
        struct_res_names: Residue names in the structural model.
    """

    def __init__(
        self,
        exp_res_ids: np.ndarray,
        exp_shifts: np.ndarray,
        struct_res_ids: np.ndarray,
        struct_res_names: list[str],
    ) -> None:
        self.loss_fn, self.n_matched = make_ca_shift_loss(
            exp_res_ids, exp_shifts, struct_res_ids, struct_res_names
        )

    def __call__(self, params: Any, coords: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the Cα chemical shift RMSD from Cartesian coordinates.

        Args:
            params: ``(3N, 3)`` Cartesian backbone coordinates — same as
                ``coords`` when used with :class:`~diff_integrator.optimizer.IntegrativeRefiner`
                and ``kinematics_fn=None``.
            coords: ``(3N, 3)`` backbone coordinates (same array as ``params``).

        Returns:
            Scalar jnp.ndarray RMSD in ppm.
        """
        phi, psi = compute_phi_psi(coords)
        return jnp.asarray(self.loss_fn(phi, psi))
