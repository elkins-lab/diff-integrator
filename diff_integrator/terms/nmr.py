from typing import Any

import jax.numpy as jnp
from diff_biophys.nmr.rdc import calculate_rdc_from_tensor, fit_saupe_tensor

from diff_integrator.loss import LossTerm


class RDCLoss(LossTerm):
    """
    Computes the loss between predicted RDCs and experimental RDCs,
    by analytically fitting the Saupe tensor at each step.
    """

    def __init__(
        self,
        atom_pairs: jnp.ndarray,
        exp_rdcs: jnp.ndarray,
        d_max: float = 1.0,
        loss_type: str = "mse",
    ):
        """
        Args:
            atom_pairs: (N, 2) indices of the two atoms forming the bond for each RDC.
            exp_rdcs: (N,) experimental RDC values.
            d_max: Maximum dipolar coupling constant (Hz) for this specific nuclear pair.
            loss_type: 'mse' (Mean Squared Error) or 'q_factor' (Cornilescu Q-factor).
        """
        self.atom_pairs = atom_pairs
        self.exp_rdcs = exp_rdcs
        self.d_max = d_max
        self.loss_type = loss_type

    def __call__(self, params: Any, coords: jnp.ndarray) -> jnp.ndarray:
        # Extract the two atoms for each bond
        coords_a = coords[self.atom_pairs[:, 0]]
        coords_b = coords[self.atom_pairs[:, 1]]

        # Calculate bond vectors
        bond_vecs = coords_a - coords_b

        # Normalize to unit vectors
        norms = jnp.linalg.norm(bond_vecs, axis=-1, keepdims=True)
        # Avoid division by zero
        bond_vecs_norm = bond_vecs / (norms + 1e-12)

        # Analytically fit the Saupe tensor to the current structure's bond vectors
        saupe_tensor = fit_saupe_tensor(
            bond_vectors=bond_vecs_norm,
            experimental_rdcs=self.exp_rdcs,
            d_max=self.d_max,
        )

        # Back-calculate the theoretical RDCs from the fitted tensor
        calc_rdcs = calculate_rdc_from_tensor(
            bond_vectors=bond_vecs_norm,
            saupe_tensor=saupe_tensor,
            d_max=self.d_max,
        )

        if self.loss_type == "mse":
            return jnp.mean((calc_rdcs - self.exp_rdcs) ** 2)
        elif self.loss_type == "q_factor":
            # Q = sqrt( sum((D_calc - D_exp)^2) / sum(D_exp^2) )
            diff_sq = jnp.sum((calc_rdcs - self.exp_rdcs) ** 2)
            exp_sq = jnp.sum(self.exp_rdcs**2)
            q = jnp.sqrt(diff_sq / jnp.maximum(exp_sq, 1e-10))
            return jnp.where(exp_sq > 0.0, q, 0.0)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
