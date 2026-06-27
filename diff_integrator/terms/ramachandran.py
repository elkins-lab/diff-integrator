"""
diff_integrator/terms/ramachandran.py — Soft Ramachandran prior potential.

Two modes:
  * **Uniform** (``residue_types=None``): the original sequence-independent
    3-basin model (α-helix, β-strand, L-α) applied identically to every
    residue.  Fully backward-compatible.
  * **Sequence-aware** (``residue_types=[list of 3-letter codes]``): per-residue
    basin centres and widths chosen from a MolProbity-derived lookup table.
    Key special cases:

    * **Glycine** — no Cβ steric clash, so the full φ/ψ plane is essentially
      accessible.  Modelled with 4 basins (adds the ε / mirror-β basin unique
      to Gly: φ ≈ +60°, ψ ≈ −120°) and a broader σ = 1.0 rad².
    * **Proline** — ring-constrained to φ ≈ −65° ± 20°.  Modelled with 2
      basins (down pucker ψ ≈ +150°; up pucker ψ ≈ −30°) and a narrow
      σ = 0.35 rad².
    * **All others** — the standard 3-basin model with σ = 0.5 rad².

Basin centres are drawn from:
  Lovell et al. (2003) *Proteins* 50:437–450 (MolProbity backbone geometry).
  Chen et al. (2010) *Acta Cryst.* D66:12–21.

Usage::

    from diff_integrator.terms.ramachandran import RamachandranLoss

    # Uniform (backward-compatible)
    loss = RamachandranLoss()

    # Sequence-aware
    loss = RamachandranLoss(residue_types=["ALA", "GLY", "PRO", "VAL", ...])
    value = loss((phi, psi), coords)
"""

from typing import Any

import jax
import jax.numpy as jnp

from diff_integrator.loss import LossTerm

# ---------------------------------------------------------------------------
# Basin lookup table
# ---------------------------------------------------------------------------

# All centres in radians (converted from degrees in comments).
# Each entry is padded to _N_BASINS entries with duplicates so that JAX can
# use a single fixed-shape (N, _N_BASINS, 2) array for all residue types.

_N_BASINS = 4

# (list-of-4-(phi, psi)-centres, sigma)
_BASIN_TABLE: dict[str, tuple[list[tuple[float, float]], float]] = {
    "GLY": (
        [
            (-1.05, -0.78),  # α-helix   (φ ≈ −60°, ψ ≈ −44°)
            (-2.09,  2.35),  # β-strand  (φ ≈ −120°, ψ ≈ +135°)
            ( 1.05,  0.78),  # L-α       (φ ≈ +60°, ψ ≈ +44°)
            ( 1.05, -2.09),  # ε mirror-β (φ ≈ +60°, ψ ≈ −120°) — unique to Gly
        ],
        1.0,   # broader: no Cβ steric clash
    ),
    "PRO": (
        [
            (-1.13,  2.62),  # down pucker (φ ≈ −65°, ψ ≈ +150°) — most common
            (-1.13, -0.52),  # up pucker   (φ ≈ −65°, ψ ≈  −30°)
            (-1.13,  2.62),  # pad
            (-1.13,  2.62),  # pad
        ],
        0.35,  # narrow: φ ring-constrained to ~−65° ± 20°
    ),
}

_DEFAULT_CENTERS: list[tuple[float, float]] = [
    (-1.05, -0.78),  # α-helix
    (-2.09,  2.35),  # β-strand
    ( 1.05,  0.78),  # L-α
    (-1.05, -0.78),  # pad (duplicate of α)
]
_DEFAULT_SIGMA = 0.5


def _lookup(res: str) -> tuple[list[tuple[float, float]], float]:
    """Return (centres-padded-to-4, sigma) for a residue type string."""
    key = res.strip().upper()
    if key in _BASIN_TABLE:
        return _BASIN_TABLE[key]
    return _DEFAULT_CENTERS, _DEFAULT_SIGMA


# ---------------------------------------------------------------------------
# Loss term
# ---------------------------------------------------------------------------


class RamachandranLoss(LossTerm):
    """Soft Ramachandran prior potential for backbone dihedral angles.

    **Uniform mode** (``residue_types=None``, default):

    Penalises (φ, ψ) pairs that lie far from three canonical regions:

    * Alpha-helix:       φ ≈ −60°, ψ ≈ −44°
    * Beta-strand:       φ ≈ −120°, ψ ≈ +135°
    * Left-handed alpha: φ ≈ +60°, ψ ≈ +44°

    For each residue the squared distance to every region centre is computed
    and a soft minimum is taken via log-sum-exp so that only the nearest basin
    contributes significantly.

    **Sequence-aware mode** (``residue_types=[...]``):

    Uses per-residue basin centres and widths from a MolProbity-derived lookup
    table.  Key improvements over the uniform model:

    * Glycine receives a 4th basin (ε / mirror-β) and a broader σ = 1.0,
      preventing incorrect penalisation of legitimately accessible conformations.
    * Proline uses 2 narrow basins (down/up pucker) centred at the
      ring-constrained φ ≈ −65°, correcting the uniform model's over-permissive
      treatment of Pro φ.

    Args:
        sigma: Softness for the log-sum-exp smooth minimum in **uniform** mode.
            Smaller values produce a sharper (harder) minimum.  Ignored when
            ``residue_types`` is provided (per-residue σ from the lookup table
            is used instead).
        residue_types: Optional list of 3-letter residue codes (e.g. ``"ALA"``,
            ``"GLY"``, ``"PRO"``) with the same length as the ``phi`` / ``psi``
            arrays passed at call time.  If ``None``, the uniform 3-basin model
            is used (backward-compatible behaviour).

    Examples:
        >>> import jax.numpy as jnp
        >>> from diff_integrator.terms.ramachandran import RamachandranLoss
        >>> phi = jnp.array([-1.05, -1.05])
        >>> psi = jnp.array([-0.78, -0.78])
        >>> # Uniform mode
        >>> loss_u = RamachandranLoss()
        >>> float(loss_u((phi, psi), None)) < 0.1
        True
        >>> # Sequence-aware mode
        >>> loss_s = RamachandranLoss(residue_types=["ALA", "GLY"])
        >>> float(loss_s((phi, psi), None)) < 0.1
        True
    """

    def __init__(
        self,
        sigma: float = 0.5,
        residue_types: list[str] | None = None,
    ) -> None:
        self.sigma = sigma
        self.residue_types = residue_types

        if residue_types is not None:
            centres_list: list[list[tuple[float, float]]] = []
            sigmas_list: list[float] = []
            for res in residue_types:
                centres, sig = _lookup(res)
                centres_list.append(centres)
                sigmas_list.append(sig)
            # (N, _N_BASINS, 2)
            self.basin_centers: jnp.ndarray | None = jnp.array(
                centres_list, dtype=jnp.float32
            )
            # (N,)
            self.sigma_per_res: jnp.ndarray | None = jnp.array(
                sigmas_list, dtype=jnp.float32
            )
        else:
            self.basin_centers = None
            self.sigma_per_res = None

    def __call__(self, params: Any, coords: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the Ramachandran penalty.

        Args:
            params: Tuple ``(phi, psi)`` of 1-D arrays of shape ``(N,)``
                containing backbone dihedral angles in radians.
            coords: Unused; retained for ``LossTerm`` ABC compatibility.

        Returns:
            Scalar mean per-residue penalty.
        """
        phi, psi = params

        if self.basin_centers is not None:
            # -----------------------------------------------------------------
            # Sequence-aware path  (vectorised, JIT-friendly)
            # -----------------------------------------------------------------
            # stacked:  (N, 2)
            stacked = jnp.stack([phi, psi], axis=-1)
            # diff:     (N, _N_BASINS, 2)
            diff = stacked[:, None, :] - self.basin_centers
            # dist_sq:  (N, _N_BASINS)
            dist_sq = jnp.sum(diff ** 2, axis=-1)
            # exponents: (N, _N_BASINS)
            exponents = -dist_sq / self.sigma_per_res[:, None]
            # penalty:  (N,)
            penalty = -self.sigma_per_res * jax.nn.logsumexp(exponents, axis=-1)
            return jnp.mean(penalty)

        # ---------------------------------------------------------------------
        # Uniform path  (original 3-basin formula — backward compatible)
        # ---------------------------------------------------------------------
        alpha_phi, alpha_psi = -1.05, -0.78
        beta_phi, beta_psi = -2.09, 2.35
        lalpha_phi, lalpha_psi = 1.05, 0.78

        d_alpha = (phi - alpha_phi) ** 2 + (psi - alpha_psi) ** 2
        d_beta = (phi - beta_phi) ** 2 + (psi - beta_psi) ** 2
        d_lalpha = (phi - lalpha_phi) ** 2 + (psi - lalpha_psi) ** 2

        dist_sq = jnp.stack([d_alpha, d_beta, d_lalpha], axis=-1)
        penalty = -self.sigma * jax.nn.logsumexp(
            -dist_sq / self.sigma, axis=-1
        )
        return jnp.mean(penalty)
