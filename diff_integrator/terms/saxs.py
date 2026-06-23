from typing import Any

import jax.numpy as jnp
from diff_biophys.saxs.kernels import debye_saxs

from diff_integrator.loss import LossTerm


class SAXSLoss(LossTerm):
    """
    Computes the loss between predicted SAXS intensities and experimental data.
    """

    def __init__(
        self,
        q_values: jnp.ndarray,
        exp_intensities: jnp.ndarray,
        form_factors: jnp.ndarray,
        volumes: jnp.ndarray | None = None,
        solvent_density: float = 0.334,
        scale_mode: str = "lsq",
    ):
        """
        Args:
            q_values: (M,) scattering vector magnitudes (A^-1).
            exp_intensities: (M,) experimental scattering intensities.
            form_factors: (N, M) q-dependent vacuum atomic form factors.
            volumes: (N,) atomic volumes for excluded volume correction.
            solvent_density: Bulk solvent electron density.
            scale_mode: How to scale theoretical to experimental data. "lsq" means
                least-squares optimal scaling factor.
        """
        self.q_values = q_values
        self.exp_intensities = exp_intensities
        self.form_factors = form_factors
        self.volumes = volumes
        self.solvent_density = solvent_density
        self.scale_mode = scale_mode

    def __call__(self, params: Any, coords: jnp.ndarray) -> jnp.ndarray:
        # Predict theoretical intensities
        calc_intensities = debye_saxs(
            coords=coords,
            q_values=self.q_values,
            form_factors=self.form_factors,
            volumes=self.volumes,
            solvent_density=self.solvent_density,
        )

        if self.scale_mode == "lsq":
            # Optimal scaling factor c that minimizes sum( (exp - c*calc)^2 )
            # c = sum(exp * calc) / sum(calc * calc)
            # Add small epsilon to prevent division by zero
            c = jnp.sum(self.exp_intensities * calc_intensities) / (
                jnp.sum(calc_intensities**2) + 1e-12
            )
            scaled_calc = c * calc_intensities
        else:
            scaled_calc = calc_intensities

        # Mean Squared Error
        mse = jnp.mean((self.exp_intensities - scaled_calc) ** 2)
        return mse
