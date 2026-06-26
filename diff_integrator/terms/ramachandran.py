from typing import Any

import jax
import jax.numpy as jnp

from diff_integrator.loss import LossTerm


class RamachandranLoss(LossTerm):
    """Soft Ramachandran prior potential for backbone dihedral angles.

    Penalises phi/psi pairs that lie far from the three canonical allowed
    regions of the Ramachandran plot:

      * Alpha-helix:          phi ~ -60°  (-1.05 rad), psi ~ -45°  (-0.78 rad)
      * Beta-strand:          phi ~ -120° (-2.09 rad), psi ~ +135° (+2.35 rad)
      * Left-handed alpha:    phi ~ +60°  (+1.05 rad), psi ~ +45°  (+0.78 rad)

    For each residue the squared distance to every region centre is computed,
    and a soft minimum is taken via the log-sum-exp trick so that only the
    nearest basin contributes significantly.  This is analogous to the
    Ramachandran torsion-angle potentials used in CNS and Rosetta.

    The ``sigma`` parameter controls the softness of the minimum: smaller
    values make the selection sharper (closer to a hard minimum), while larger
    values smooth the landscape.

    This is a port of the ``ramachandran_penalty`` from torsion-tuner.
    """

    def __init__(self, sigma: float = 0.5) -> None:
        """
        Args:
            sigma: Softness parameter for the log-sum-exp smooth minimum.
        """
        self.sigma = sigma

    def __call__(self, params: Any, coords: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the Ramachandran penalty.

        Args:
            params: A tuple ``(phi, psi)`` where each element is a 1-D
                ``jnp.ndarray`` of shape ``(N,)`` containing backbone
                dihedral angles in radians.
            coords: Unused by this term (retained for ABC compatibility).

        Returns:
            A scalar ``jnp.ndarray`` with the mean per-residue penalty.
        """
        phi, psi = params

        # Region centres (radians)
        alpha_phi, alpha_psi = -1.05, -0.78
        beta_phi, beta_psi = -2.09, 2.35
        lalpha_phi, lalpha_psi = 1.05, 0.78

        # Squared distance to each region
        d_alpha = (phi - alpha_phi) ** 2 + (psi - alpha_psi) ** 2
        d_beta = (phi - beta_phi) ** 2 + (psi - beta_psi) ** 2
        d_lalpha = (phi - lalpha_phi) ** 2 + (psi - lalpha_psi) ** 2

        # Stack into (N, 3)
        dist_sq = jnp.stack([d_alpha, d_beta, d_lalpha], axis=-1)

        # Soft minimum via log-sum-exp
        penalty = -self.sigma * jax.nn.logsumexp(
            -dist_sq / self.sigma, axis=-1
        )

        return jnp.mean(penalty)
