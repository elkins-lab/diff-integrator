"""
diff_integrator/terms/bond_geometry.py — Cartesian backbone geometry penalties.

Provides two differentiable loss terms that enforce physically valid backbone
geometry when optimising **directly in Cartesian coordinate space** (as opposed
to the NeRF internal-coordinate parameterisation):

* ``BondLengthPenalty`` — harmonic penalty on backbone bond lengths.
* ``BondAnglePenalty``  — harmonic penalty on backbone bond angles.

Together these replace the hard geometric constraints imposed by the NeRF
builder, allowing the optimizer to move atoms freely while keeping bond lengths
and angles close to the Engh & Huber (1991) ideal values.

Factory function
----------------

``make_backbone_bond_geometry(n_residues)`` returns a pre-populated
``(BondLengthPenalty, BondAnglePenalty)`` tuple for a backbone of ``n_residues``
residues in the standard N–CA–C layout::

    coords layout:  N₀  CA₀  C₀  N₁  CA₁  C₁  …  Nₙ  CAₙ  Cₙ
    atom index:      0    1    2   3    4    5  …  3n  3n+1  3n+2

Usage example (Cartesian refinement)::

    from diff_integrator.terms.bond_geometry import make_backbone_bond_geometry

    bond_pen, angle_pen = make_backbone_bond_geometry(n_residues)

    joint_loss = JointLoss([
        (geom_anchor,  weight_anchor),   # position restraint
        (ca_shift_loss, 1.0),
        (bond_pen,     50.0),            # stiff bonds
        (angle_pen,    10.0),            # softer angles
    ])

    result = IntegrativeRefiner(joint_loss).run(
        init_params=init_coords,         # (3N, 3) Cartesian coordinates
        kinematics_fn=None,              # identity: params ARE the coords
        ...
    )

References
----------
Engh, R. A. & Huber, R. (1991). Accurate bond and angle parameters for X-ray
protein structure refinement. *Acta Cryst.* A47, 392–400.
"""

from typing import Any

import jax.numpy as jnp
import numpy as np

from diff_biophys.geometry.backbone import (
    CA_C_LENGTH,
    CA_C_N_ANGLE,
    C_N_CA_ANGLE,
    C_N_LENGTH,
    N_CA_C_ANGLE,
    N_CA_LENGTH,
)

from diff_integrator.loss import LossTerm

# ---------------------------------------------------------------------------
# Loss terms
# ---------------------------------------------------------------------------


class BondLengthPenalty(LossTerm):
    """Harmonic penalty on backbone bond lengths.

    Penalises deviations of bond lengths from ideal (Engh & Huber) values.
    The loss is the **mean** squared deviation across all bonds:

    .. math::

        \\mathcal{L}_{\\text{bond}} =
            \\frac{1}{B}\\sum_{b=1}^{B}
            \\left(\\|\\mathbf{r}_{i_b} - \\mathbf{r}_{j_b}\\| - d_b^{\\text{ideal}}\\right)^2

    A mean (rather than sum) is used so the loss magnitude is independent of
    chain length and the weight has a consistent interpretation across proteins
    of different sizes.

    Args:
        bond_pairs: ``(B, 2)`` integer array of atom index pairs.
            Each row ``[i, j]`` specifies one covalent bond.
        ideal_lengths: ``(B,)`` float array of ideal bond lengths in Å.
        weight: Scalar multiplier applied to the entire term when used inside
            a ``JointLoss``.  Has no effect when the term is called directly.
    """

    name: str = "bond_length"

    def __init__(
        self,
        bond_pairs: jnp.ndarray,
        ideal_lengths: jnp.ndarray,
        weight: float = 1.0,
    ) -> None:
        self.bond_pairs = bond_pairs
        self.ideal_lengths = ideal_lengths
        self.weight = weight

    def __call__(self, params: Any, coords: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the bond-length penalty.

        Args:
            params: Ignored (present for ``LossTerm`` interface compatibility).
            coords: ``(N_atoms, 3)`` Cartesian coordinates in Å.

        Returns:
            Scalar mean squared bond-length deviation.
        """
        r_i = coords[self.bond_pairs[:, 0]]
        r_j = coords[self.bond_pairs[:, 1]]
        lengths = jnp.linalg.norm(r_i - r_j, axis=-1)
        return jnp.mean((lengths - self.ideal_lengths) ** 2)

    def bond_rmsd(self, coords: jnp.ndarray) -> float:
        """Root-mean-square bond-length deviation from ideal (in Å).

        Convenience diagnostic for monitoring — not used in the gradient.

        Args:
            coords: ``(N_atoms, 3)`` Cartesian coordinates.

        Returns:
            Bond RMSD in Å.
        """
        r_i = coords[self.bond_pairs[:, 0]]
        r_j = coords[self.bond_pairs[:, 1]]
        lengths = jnp.linalg.norm(r_i - r_j, axis=-1)
        return float(jnp.sqrt(jnp.mean((lengths - self.ideal_lengths) ** 2)))


class BondAnglePenalty(LossTerm):
    """Harmonic penalty on backbone bond angles.

    Penalises deviations of bond angles from ideal (Engh & Huber) values.
    Each angle is defined by three consecutive atoms (i, j, k) with j at the
    apex; the angle ∠(i, j, k) is computed as:

    .. math::

        \\theta = \\arccos\\!\\left(
            \\frac{(\\mathbf{r}_i - \\mathbf{r}_j) \\cdot
                   (\\mathbf{r}_k - \\mathbf{r}_j)}
                  {\\|\\mathbf{r}_i - \\mathbf{r}_j\\|\\,
                   \\|\\mathbf{r}_k - \\mathbf{r}_j\\|}
        \\right)

    A small epsilon is added to denominators to guard against degenerate
    (collinear) configurations.  The loss is the **mean** squared angular
    deviation in radians:

    .. math::

        \\mathcal{L}_{\\text{angle}} =
            \\frac{1}{A}\\sum_{a=1}^{A}
            \\left(\\theta_a - \\theta_a^{\\text{ideal}}\\right)^2

    Args:
        angle_triples: ``(A, 3)`` integer array of atom index triples.
            Each row ``[i, j, k]`` specifies one bond angle at atom ``j``.
        ideal_angles_rad: ``(A,)`` float array of ideal angles **in radians**.
        weight: Scalar multiplier applied inside a ``JointLoss``.
    """

    name: str = "bond_angle"

    def __init__(
        self,
        angle_triples: jnp.ndarray,
        ideal_angles_rad: jnp.ndarray,
        weight: float = 1.0,
    ) -> None:
        self.angle_triples = angle_triples
        self.ideal_angles_rad = ideal_angles_rad
        self.weight = weight

    def __call__(self, params: Any, coords: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the bond-angle penalty.

        Args:
            params: Ignored (present for ``LossTerm`` interface compatibility).
            coords: ``(N_atoms, 3)`` Cartesian coordinates in Å.

        Returns:
            Scalar mean squared bond-angle deviation (in rad²).
        """
        r_i = coords[self.angle_triples[:, 0]]
        r_j = coords[self.angle_triples[:, 1]]
        r_k = coords[self.angle_triples[:, 2]]

        v1 = r_i - r_j  # (A, 3)  vector j→i
        v2 = r_k - r_j  # (A, 3)  vector j→k

        norm1 = jnp.linalg.norm(v1, axis=-1, keepdims=True)
        norm2 = jnp.linalg.norm(v2, axis=-1, keepdims=True)

        u1 = v1 / (norm1 + 1e-8)
        u2 = v2 / (norm2 + 1e-8)

        # Clamp to [-1, 1] to make arccos safe at degenerate configurations
        cos_theta = jnp.sum(u1 * u2, axis=-1)
        cos_theta = jnp.clip(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
        theta = jnp.arccos(cos_theta)

        return jnp.mean((theta - self.ideal_angles_rad) ** 2)

    def angle_rmsd_deg(self, coords: jnp.ndarray) -> float:
        """Root-mean-square bond-angle deviation from ideal (in degrees).

        Convenience diagnostic for monitoring — not used in the gradient.

        Args:
            coords: ``(N_atoms, 3)`` Cartesian coordinates.

        Returns:
            Angle RMSD in degrees.
        """
        r_i = coords[self.angle_triples[:, 0]]
        r_j = coords[self.angle_triples[:, 1]]
        r_k = coords[self.angle_triples[:, 2]]

        v1 = r_i - r_j
        v2 = r_k - r_j

        u1 = v1 / (jnp.linalg.norm(v1, axis=-1, keepdims=True) + 1e-8)
        u2 = v2 / (jnp.linalg.norm(v2, axis=-1, keepdims=True) + 1e-8)

        cos_theta = jnp.clip(jnp.sum(u1 * u2, axis=-1), -1.0 + 1e-7, 1.0 - 1e-7)
        theta = jnp.arccos(cos_theta)
        rmsd_rad = float(jnp.sqrt(jnp.mean((theta - self.ideal_angles_rad) ** 2)))
        return float(np.degrees(rmsd_rad))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_backbone_bond_geometry(
    n_residues: int,
) -> tuple[BondLengthPenalty, BondAnglePenalty]:
    """Build bond-length and bond-angle penalty terms for an N–CA–C backbone.

    Constructs all backbone bond pairs and angle triples for a chain of
    ``n_residues`` residues using the atom layout::

        atom index = 3·res + {0: N,  1: CA,  2: C}

    Ideal values are taken from Engh & Huber (1991):

    .. list-table:: Bond lengths
       :header-rows: 1

       * - Bond
         - Ideal (Å)
       * - N–CA
         - 1.459
       * - CA–C
         - 1.525
       * - C–N (peptide)
         - 1.329

    .. list-table:: Bond angles
       :header-rows: 1

       * - Angle
         - Ideal (°)
         - Ideal (rad)
       * - N–CA–C (at Cα)
         - 111.2
         - 1.9406
       * - CA–C–N (at C)
         - 116.2
         - 2.0281
       * - C–N–CA (at N, next residue)
         - 121.7
         - 2.1240

    Args:
        n_residues: Number of residues.  Must be ≥ 2.

    Returns:
        Tuple ``(bond_penalty, angle_penalty)`` with all backbone bonds and
        angles pre-populated.  Both are ready to be passed directly to
        ``JointLoss``.

    Raises:
        ValueError: If ``n_residues < 2``.

    Example::

        bond_pen, angle_pen = make_backbone_bond_geometry(n_residues=92)
        joint_loss = JointLoss([
            (position_anchor, 5.0),
            (ca_shift_loss, 1.0),
            (bond_pen, 50.0),
            (angle_pen, 10.0),
        ])
    """
    if n_residues < 2:
        raise ValueError(
            f"make_backbone_bond_geometry requires n_residues ≥ 2, got {n_residues}"
        )

    # ------------------------------------------------------------------
    # Bond pairs
    # ------------------------------------------------------------------
    # Within each residue: N(3r)–CA(3r+1)  and  CA(3r+1)–C(3r+2)
    # Between residues:    C(3r+2)–N(3(r+1))
    # ------------------------------------------------------------------
    bond_pairs_list: list[tuple[int, int]] = []
    ideal_lengths_list: list[float] = []

    for r in range(n_residues):
        n_idx  = 3 * r
        ca_idx = 3 * r + 1
        c_idx  = 3 * r + 2

        # N–CA
        bond_pairs_list.append((n_idx, ca_idx))
        ideal_lengths_list.append(N_CA_LENGTH)

        # CA–C
        bond_pairs_list.append((ca_idx, c_idx))
        ideal_lengths_list.append(CA_C_LENGTH)

        # C–N (peptide bond to next residue)
        if r < n_residues - 1:
            n_next = 3 * (r + 1)
            bond_pairs_list.append((c_idx, n_next))
            ideal_lengths_list.append(C_N_LENGTH)

    bond_pairs = jnp.array(bond_pairs_list, dtype=jnp.int32)
    ideal_lengths = jnp.array(ideal_lengths_list, dtype=jnp.float32)

    # ------------------------------------------------------------------
    # Angle triples  (i, j, k)  — angle at apex j
    # ------------------------------------------------------------------
    # Within residue:    N–CA–C  (at Cα)
    # Inter-residue:     CA–C–N  (at C_i)   and   C–N–CA  (at N_{i+1})
    # ------------------------------------------------------------------
    angle_triples_list: list[tuple[int, int, int]] = []
    ideal_angles_list: list[float] = []

    for r in range(n_residues):
        n_idx  = 3 * r
        ca_idx = 3 * r + 1
        c_idx  = 3 * r + 2

        # N–CA–C  (within residue, at Cα)
        angle_triples_list.append((n_idx, ca_idx, c_idx))
        ideal_angles_list.append(N_CA_C_ANGLE)

        if r < n_residues - 1:
            n_next  = 3 * (r + 1)
            ca_next = 3 * (r + 1) + 1

            # CA–C–N  (at C of residue r)
            angle_triples_list.append((ca_idx, c_idx, n_next))
            ideal_angles_list.append(CA_C_N_ANGLE)

            # C–N–CA  (at N of residue r+1)
            angle_triples_list.append((c_idx, n_next, ca_next))
            ideal_angles_list.append(C_N_CA_ANGLE)

    angle_triples = jnp.array(angle_triples_list, dtype=jnp.int32)
    ideal_angles_rad = jnp.array(ideal_angles_list, dtype=jnp.float32)

    bond_penalty = BondLengthPenalty(bond_pairs, ideal_lengths)
    angle_penalty = BondAnglePenalty(angle_triples, ideal_angles_rad)

    return bond_penalty, angle_penalty
