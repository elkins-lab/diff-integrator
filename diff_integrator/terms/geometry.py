from typing import Any

import jax.numpy as jnp

from diff_integrator.loss import LossTerm


class GeometryLoss(LossTerm):
    """
    Computes a loss penalty for geometric violations like extreme bond lengths
    or atomic clashes.
    """

    def __init__(
        self,
        bonds: jnp.ndarray | None = None,
        ideal_bond_lengths: jnp.ndarray | None = None,
        clash_pairs: jnp.ndarray | None = None,
        min_clash_dists: jnp.ndarray | None = None,
        bond_weight: float = 1.0,
        clash_weight: float = 1.0,
    ):
        """
        Args:
            bonds: (N, 2) indices of bonded atoms.
            ideal_bond_lengths: (N,) target distances for bonds.
            clash_pairs: (M, 2) indices of atom pairs to check for clashes.
            min_clash_dists: (M,) minimum allowed distance for each clash pair.
            bond_weight: Weight multiplier for the bond loss.
            clash_weight: Weight multiplier for the clash loss.
        """
        self.bonds = bonds
        self.ideal_bond_lengths = ideal_bond_lengths
        self.clash_pairs = clash_pairs
        self.min_clash_dists = min_clash_dists
        self.bond_weight = bond_weight
        self.clash_weight = clash_weight

    def __call__(self, params: Any, coords: jnp.ndarray) -> jnp.ndarray:
        total_loss = jnp.array(0.0)

        # Bond constraints (harmonic potential)
        if self.bonds is not None and self.ideal_bond_lengths is not None:
            c1 = coords[self.bonds[:, 0]]
            c2 = coords[self.bonds[:, 1]]
            dists = jnp.linalg.norm(c1 - c2, axis=-1)
            bond_loss = jnp.sum((dists - self.ideal_bond_lengths) ** 2)
            total_loss += self.bond_weight * bond_loss

        # Clash constraints (half-harmonic repulsion)
        if self.clash_pairs is not None and self.min_clash_dists is not None:
            c1 = coords[self.clash_pairs[:, 0]]
            c2 = coords[self.clash_pairs[:, 1]]
            dists = jnp.linalg.norm(c1 - c2, axis=-1)
            # Only penalize if dists < min_clash_dists
            violations = jnp.maximum(0.0, self.min_clash_dists - dists)
            clash_loss = jnp.sum(violations**2)
            total_loss += self.clash_weight * clash_loss

        return total_loss
