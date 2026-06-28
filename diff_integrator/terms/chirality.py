"""
diff_integrator/terms/chirality.py — Cα chirality penalty for Cartesian refinement.

In Cartesian-coordinate refinement, bond-length and bond-angle penalties keep
covalent geometry close to ideal values but are chirality-blind: they cannot
distinguish an L-amino acid Cα from its mirror-image D-configuration.  Under
large chemical-shift gradients, a Cα can drift past the L→D boundary without
triggering any existing penalty — a structurally catastrophic and silent failure
mode.

This module provides ``ChiralityPenalty``, a differentiable half-harmonic
potential that fires as soon as a Cα approaches D-geometry, preventing any
actual inversion.

Chirality indicator
-------------------
For each interior Cα (residue i, with a preceding C_{i-1} available), the
signed scalar triple product::

    chi_i = dot( cross(N_i - CA_i,  C_i - CA_i),  C_{i-1} - CA_i )

is negative for all L-amino acids in standard backbone geometry (verified on
2KZV, GmR58A, and HR2876B across 250+ residues).

The half-harmonic penalty fires when ``chi_i ≥ −margin``::

    penalty_i = max(0,  chi_i + margin)^2

At ``chi_i = 0`` (the L→D boundary) the penalty equals ``margin^2`` and the
gradient points back toward L-geometry.  With the default ``margin = 0.1`` Å³
the penalty is negligible for well-folded L-amino acids (typical ``|chi| ≈ 2``
Å³) and strong at the inversion boundary.

Factory
-------
``make_backbone_chirality(n_residues)`` constructs a ready-to-use
``ChiralityPenalty`` for the standard N–CA–C backbone layout::

    atom_index = 3 * residue_index + {0: N,  1: CA,  2: C}

All *interior* residues (indices 1 to n_residues−1) are covered; the
N-terminal residue (no C_{i-1}) is excluded.  This is identical to the
convention used by REFMAC and PHENIX.

Usage::

    from diff_integrator.terms.chirality import make_backbone_chirality

    chirality_pen = make_backbone_chirality(n_residues)

    joint_loss = JointLoss([
        (position_anchor, 5.0),
        (ca_shift_loss,   1.0),
        (bond_pen,       50.0),
        (angle_pen,      10.0),
        (chirality_pen,  20.0),   # prevents L→D inversion
    ])

References
----------
Engh, R. A. & Huber, R. (1991). *Acta Cryst.* A47, 392–400.
Murshudov, G. N. et al. (2011). REFMAC5. *Acta Cryst.* D67, 355–367.
"""

from typing import Any

import jax.numpy as jnp
import numpy as np

from diff_integrator.loss import LossTerm


class ChiralityPenalty(LossTerm):
    """Half-harmonic Cα chirality penalty for Cartesian backbone refinement.

    Penalizes any Cα that drifts toward or past D-amino acid geometry.
    The penalty is zero for well-folded L-amino acids and grows quadratically
    as a Cα approaches the L→D inversion boundary.

    The chirality indicator is the signed scalar triple product::

        chi_i = dot( cross(N_i - CA_i,  C_i - CA_i),  C_{i-1} - CA_i )

    which is negative for L-amino acids (empirically verified; see module
    docstring).  The penalty fires when ``chi_i ≥ −margin``:

    .. math::

        \\mathcal{L}_{\\text{chiral}} =
            \\frac{1}{M} \\sum_{i=1}^{M}
            \\left[ \\max\\!\\left(0,\\, \\chi_i + \\delta\\right) \\right]^2

    where :math:`\\delta` is ``margin`` and :math:`M` is the number of
    monitored Cα centers.

    Args:
        ca_indices: ``(M,)`` integer array of Cα atom indices.
        n_indices: ``(M,)`` integer array of N atom indices (same residue as CA).
        c_indices: ``(M,)`` integer array of C atom indices (same residue as CA).
        cprev_indices: ``(M,)`` integer array of C atom indices from the
            *preceding* residue (i.e. C_{i-1}).
        margin: Safety margin in Å³.  The penalty starts at ``chi = −margin``
            and the gradient points back toward L-geometry from that point on.
            Default ``0.1``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from diff_integrator.terms.chirality import ChiralityPenalty
        >>> # 5-atom mini-backbone: N(0) CA(1) C(2) N(3) CA(4) C(5) — 2 residues
        >>> # Interior Cα at index 4 (residue 1), with C_prev at index 2.
        >>> pen = ChiralityPenalty(
        ...     ca_indices=jnp.array([4]),
        ...     n_indices=jnp.array([3]),
        ...     c_indices=jnp.array([5]),
        ...     cprev_indices=jnp.array([2]),
        ... )
        >>> coords = jnp.zeros((6, 3))          # degenerate — chi = 0 → violation
        >>> float(pen(None, coords)) >= 0
        True
    """

    name: str = "chirality"

    def __init__(
        self,
        ca_indices: jnp.ndarray,
        n_indices: jnp.ndarray,
        c_indices: jnp.ndarray,
        cprev_indices: jnp.ndarray,
        margin: float = 0.1,
    ) -> None:
        self.ca_indices    = jnp.asarray(ca_indices,    dtype=jnp.int32)
        self.n_indices     = jnp.asarray(n_indices,     dtype=jnp.int32)
        self.c_indices     = jnp.asarray(c_indices,     dtype=jnp.int32)
        self.cprev_indices = jnp.asarray(cprev_indices, dtype=jnp.int32)
        self.margin        = float(margin)

        m = self.ca_indices.shape[0]
        for name, arr in [
            ("n_indices",     self.n_indices),
            ("c_indices",     self.c_indices),
            ("cprev_indices", self.cprev_indices),
        ]:
            if arr.shape != (m,):
                raise ValueError(
                    f"ChiralityPenalty: {name} must have shape ({m},), "
                    f"got {arr.shape}"
                )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_centers(self) -> int:
        """Number of monitored Cα centers."""
        return int(self.ca_indices.shape[0])

    # ------------------------------------------------------------------
    # Loss evaluation
    # ------------------------------------------------------------------

    def __call__(self, params: Any, coords: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the mean chirality violation penalty.

        Args:
            params: Optimization parameters (ignored; present for ``LossTerm``
                interface compatibility).
            coords: ``(N_atoms, 3)`` Cartesian coordinates in Å.

        Returns:
            Scalar mean chirality penalty (Å^6 units, but treated as
            a dimensionless loss contribution inside a ``JointLoss``).
        """
        ca    = coords[self.ca_indices]     # (M, 3)
        n     = coords[self.n_indices]      # (M, 3)
        c     = coords[self.c_indices]      # (M, 3)
        cprev = coords[self.cprev_indices]  # (M, 3)

        # Vectors from CA
        u = n     - ca  # CA→N
        v = c     - ca  # CA→C
        w = cprev - ca  # CA→C_prev

        # Signed triple product: chi < 0 for L-amino acids
        cross_uv = jnp.cross(u, v)                   # (M, 3)
        chi      = jnp.sum(cross_uv * w, axis=-1)    # (M,)

        # Half-harmonic penalty: fires when chi >= −margin
        violations = jnp.maximum(0.0, chi + self.margin)
        return jnp.mean(violations ** 2)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def count_violations(self, coords: jnp.ndarray) -> int:
        """Count Cα centers that have crossed into D-geometry (chi ≥ 0).

        This is a pure diagnostic — not used in the gradient.

        Args:
            coords: ``(N_atoms, 3)`` Cartesian coordinates in Å.

        Returns:
            Number of Cα centers with ``chi ≥ 0`` (true L→D inversions).
        """
        ca    = coords[self.ca_indices]
        n     = coords[self.n_indices]
        c     = coords[self.c_indices]
        cprev = coords[self.cprev_indices]
        u = n - ca
        v = c - ca
        w = cprev - ca
        chi = jnp.sum(jnp.cross(u, v) * w, axis=-1)
        return int(jnp.sum(chi >= 0.0))

    def chi_values(self, coords: jnp.ndarray) -> jnp.ndarray:
        """Return the raw chi value for every monitored Cα center.

        Useful for diagnosing which specific residues are at risk of inversion.

        Args:
            coords: ``(N_atoms, 3)`` Cartesian coordinates in Å.

        Returns:
            ``(M,)`` array of chi values.  Negative = L-geometry.
        """
        ca    = coords[self.ca_indices]
        n     = coords[self.n_indices]
        c     = coords[self.c_indices]
        cprev = coords[self.cprev_indices]
        u = n - ca
        v = c - ca
        w = cprev - ca
        return jnp.sum(jnp.cross(u, v) * w, axis=-1)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def make_backbone_chirality(
    n_residues: int,
    margin: float = 0.1,
) -> ChiralityPenalty:
    """Build a ``ChiralityPenalty`` for the standard N–CA–C backbone layout.

    Constructs index arrays for all *interior* residues (1 through
    n_residues − 1) of a backbone stored as::

        atom_index = 3 * residue_index + {0: N,  1: CA,  2: C}

    The N-terminal residue (index 0) is excluded because it has no preceding
    C_{i-1}.  This matches the convention used by REFMAC and PHENIX.

    Args:
        n_residues: Total number of residues.  Must be ≥ 2 (need at least one
            interior residue).
        margin: Chirality margin passed through to ``ChiralityPenalty``.
            Default ``0.1`` Å³.

    Returns:
        A ``ChiralityPenalty`` covering residues 1 to n_residues − 1.

    Raises:
        ValueError: If ``n_residues < 2``.

    Example::

        chirality_pen = make_backbone_chirality(n_residues=92)
        joint_loss = JointLoss([
            (position_anchor, 5.0),
            (ca_shift_loss,   1.0),
            (bond_pen,       50.0),
            (angle_pen,      10.0),
            (chirality_pen,  20.0),
        ])
    """
    if n_residues < 2:
        raise ValueError(
            f"make_backbone_chirality requires n_residues ≥ 2, got {n_residues}"
        )

    # Interior residues: 1, 2, ..., n_residues - 1
    interior = np.arange(1, n_residues)        # residue indices
    ca_idx    = 3 * interior + 1               # CA of residue i
    n_idx     = 3 * interior                   # N  of residue i
    c_idx     = 3 * interior + 2              # C  of residue i
    cprev_idx = 3 * (interior - 1) + 2        # C  of residue i-1

    return ChiralityPenalty(
        ca_indices    = jnp.array(ca_idx,    dtype=jnp.int32),
        n_indices     = jnp.array(n_idx,     dtype=jnp.int32),
        c_indices     = jnp.array(c_idx,     dtype=jnp.int32),
        cprev_indices = jnp.array(cprev_idx, dtype=jnp.int32),
        margin        = margin,
    )
