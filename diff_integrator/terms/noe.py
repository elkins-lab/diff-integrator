"""
diff_integrator/terms/noe.py — NOE distance restraint loss term.

Nuclear Overhauser Effect (NOE) measurements report upper (and optionally lower)
bounds on inter-atomic distances.  This module provides a flat-bottomed harmonic
loss that is zero when the distance satisfies the bounds and grows quadratically
once a bound is violated.

The standard XPLOR/CNS flat-well NOE energy function is:

    E(d) = k * (d - d_upper)^2    if d > d_upper   (upper-bound violation)
           k * (d_lower - d)^2    if d < d_lower    (lower-bound violation)
           0                      otherwise

where ``d = ||r_i - r_j||`` is the Euclidean distance between the two atoms
and ``k`` is ``force_const``.

Because the distance is computed entirely via JAX operations, gradients flow
back to both atomic positions without any special handling.

Usage::

    from diff_integrator.terms.noe import NOELoss, make_noe_restraints

    # Direct construction from pre-indexed arrays
    restraints = NOELoss(
        atom_pairs  = jnp.array([[0, 5], [12, 30]]),
        d_upper     = jnp.array([5.0, 4.5]),
        d_lower     = jnp.array([1.5, 1.8]),   # optional
        force_const = 50.0,
    )

    # Or use the factory to map (res_id, atom_name) onto backbone atom indices
    restraints = make_noe_restraints(
        noe_list   = [
            {"res_i": 3, "atom_i": "CA", "res_j": 10, "atom_j": "CA",
             "d_upper": 5.0, "d_lower": 1.8},
        ],
        res_ids    = struct_res_ids,     # (N_res,) array of residue numbers
        atom_names = ["N", "CA", "C"],  # ordering within each residue block
    )

References
----------
Nilges, M., Clore, G. M. & Gronenborn, A. M. (1988). Determination of
three-dimensional structures of proteins from interproton distance data by
dynamical simulated annealing from a random array of atoms.  *FEBS Lett.*
229, 317–324.
"""

from typing import Any

import jax.numpy as jnp
import numpy as np

from diff_integrator.loss import LossTerm


class NOELoss(LossTerm):
    """Flat-bottomed harmonic NOE distance restraint loss.

    Evaluates the total NOE restraint violation energy for a set of atom
    pairs with distance bounds.  The loss is zero when all distances are
    within bounds and grows quadratically once any bound is violated:

    .. math::

        E = \\frac{k}{M} \\sum_{m=1}^{M}
            \\left[
                \\max(0,\\, d_m - d_m^{\\text{upper}})^2 +
                \\max(0,\\, d_m^{\\text{lower}} - d_m)^2
            \\right]

    where :math:`M` is the number of restraints, :math:`d_m` is the
    Euclidean distance between atoms :math:`i_m` and :math:`j_m`, and
    :math:`k` is ``force_const``.  A mean (rather than sum) is used so
    the loss magnitude is independent of the number of restraints and the
    weight has a consistent interpretation across datasets of different sizes.

    Args:
        atom_pairs: ``(M, 2)`` integer array of atom index pairs.  Each row
            ``[i, j]`` specifies one restraint between atoms ``i`` and ``j``.
        d_upper: ``(M,)`` float array of upper-bound distances in Å.  A
            penalty fires whenever ``d > d_upper[m]``.
        d_lower: Optional ``(M,)`` float array of lower-bound distances in Å.
            A penalty fires whenever ``d < d_lower[m]``.  If ``None``
            (default) only upper-bound violations are penalized, which is
            the most common convention for solution NMR distance restraints.
        force_const: Scalar force constant ``k`` (unitless multiplier for the
            quadratic energy).  Default ``1.0``.  Increase for stiffer
            restraints.

    Examples:
        >>> import jax.numpy as jnp
        >>> from diff_integrator.terms.noe import NOELoss
        >>> coords = jnp.zeros((6, 3))
        >>> # Atoms 0 and 5 are coincident — distance 0.0, which violates d_lower=1.8
        >>> restraint = NOELoss(
        ...     atom_pairs=jnp.array([[0, 5]]),
        ...     d_upper=jnp.array([5.0]),
        ...     d_lower=jnp.array([1.8]),
        ... )
        >>> float(restraint(None, coords)) > 0
        True
    """

    name: str = "noe"

    def __init__(
        self,
        atom_pairs: jnp.ndarray,
        d_upper: jnp.ndarray,
        d_lower: jnp.ndarray | None = None,
        force_const: float = 1.0,
    ) -> None:
        self.atom_pairs = jnp.asarray(atom_pairs, dtype=jnp.int32)
        self.d_upper = jnp.asarray(d_upper, dtype=jnp.float32)
        self.d_lower = (
            jnp.asarray(d_lower, dtype=jnp.float32) if d_lower is not None else None
        )
        self.force_const = force_const

        n = self.atom_pairs.shape[0]
        if self.d_upper.shape != (n,):
            raise ValueError(
                f"d_upper must have shape ({n},), got {self.d_upper.shape}"
            )
        if self.d_lower is not None and self.d_lower.shape != (n,):
            raise ValueError(
                f"d_lower must have shape ({n},), got {self.d_lower.shape}"
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_restraints(self) -> int:
        """Number of distance restraints."""
        return int(self.atom_pairs.shape[0])

    # ------------------------------------------------------------------
    # Loss evaluation
    # ------------------------------------------------------------------

    def __call__(self, params: Any, coords: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the total NOE restraint violation energy.

        Args:
            params: Optimization parameters (ignored; present for ``LossTerm``
                interface compatibility).
            coords: ``(N_atoms, 3)`` Cartesian coordinates in Å.

        Returns:
            Scalar mean restraint violation energy.
        """
        r_i = coords[self.atom_pairs[:, 0]]  # (M, 3)
        r_j = coords[self.atom_pairs[:, 1]]  # (M, 3)
        # Add small epsilon to avoid zero-gradient at d=0
        d = jnp.sqrt(jnp.sum((r_i - r_j) ** 2, axis=-1) + 1e-12)  # (M,)

        # Upper-bound violations
        upper_viol = jnp.maximum(0.0, d - self.d_upper)
        energy = jnp.sum(upper_viol ** 2)

        # Lower-bound violations (optional)
        if self.d_lower is not None:
            lower_viol = jnp.maximum(0.0, self.d_lower - d)
            energy = energy + jnp.sum(lower_viol ** 2)

        return self.force_const * energy / self.n_restraints

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def count_violations(self, coords: jnp.ndarray) -> dict[str, int]:
        """Count upper- and lower-bound violations at the current coordinates.

        This is a pure diagnostic — not used in the gradient.  Useful for
        reporting during refinement callbacks.

        Args:
            coords: ``(N_atoms, 3)`` Cartesian coordinates in Å.

        Returns:
            Dict with keys ``"upper"``, ``"lower"`` (always 0 if ``d_lower``
            was not supplied), and ``"total"``.
        """
        r_i = coords[self.atom_pairs[:, 0]]
        r_j = coords[self.atom_pairs[:, 1]]
        d = jnp.sqrt(jnp.sum((r_i - r_j) ** 2, axis=-1) + 1e-12)

        n_upper = int(jnp.sum(d > self.d_upper))
        n_lower = 0
        if self.d_lower is not None:
            n_lower = int(jnp.sum(d < self.d_lower))
        return {"upper": n_upper, "lower": n_lower, "total": n_upper + n_lower}

    def rms_violation(self, coords: jnp.ndarray) -> float:
        """Root-mean-square distance violation across all restraints (in Å).

        Violations are signed: positive for upper-bound violations, negative
        for lower-bound violations; the squared sum is taken before the RMS.

        Args:
            coords: ``(N_atoms, 3)`` Cartesian coordinates in Å.

        Returns:
            RMS violation in Å.
        """
        r_i = coords[self.atom_pairs[:, 0]]
        r_j = coords[self.atom_pairs[:, 1]]
        d = jnp.sqrt(jnp.sum((r_i - r_j) ** 2, axis=-1) + 1e-12)

        upper_viol = jnp.maximum(0.0, d - self.d_upper)
        sq_viols = upper_viol ** 2

        if self.d_lower is not None:
            lower_viol = jnp.maximum(0.0, self.d_lower - d)
            sq_viols = sq_viols + lower_viol ** 2

        return float(jnp.sqrt(jnp.mean(sq_viols)))


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def make_noe_restraints(
    noe_list: list[dict],
    res_ids: np.ndarray,
    atom_names: list[str] | None = None,
    force_const: float = 1.0,
) -> "NOELoss":
    """Build an ``NOELoss`` from a list of NOE observations.

    Maps ``(res_id, atom_name)`` pairs to flat atom indices using the
    structure's residue ordering and the per-residue atom layout.

    This factory supports any fixed atom ordering within a residue block.
    For the standard backbone N–CA–C layout pass ``atom_names=["N", "CA", "C"]``
    (the default), which gives atom indices::

        atom_index = res_block_start + atom_names.index(atom_name)
        where res_block_start = 3 * position_of(res_id in res_ids)

    Args:
        noe_list: List of dicts, each with keys:

            * ``"res_i"``   — residue number of atom i (int)
            * ``"atom_i"``  — atom name of atom i (str, e.g. ``"CA"``)
            * ``"res_j"``   — residue number of atom j (int)
            * ``"atom_j"``  — atom name of atom j (str)
            * ``"d_upper"`` — upper-bound distance in Å (float)
            * ``"d_lower"`` — lower-bound distance in Å (float, optional)

        res_ids: ``(N_res,)`` array of residue numbers in the structure,
            in the same order as the backbone coordinate array.
        atom_names: Ordered list of atom names within each residue block.
            Default ``["N", "CA", "C"]`` (standard backbone layout).
        force_const: Force constant passed to ``NOELoss``.  Default ``1.0``.

    Returns:
        An ``NOELoss`` instance with all matched restraints.

    Raises:
        ValueError: If any ``(res_id, atom_name)`` pair cannot be mapped to
            an atom index (residue not in structure, or atom name not in
            ``atom_names``).

    Example::

        restraints = make_noe_restraints(
            noe_list=[
                {"res_i": 5, "atom_i": "CA", "res_j": 20, "atom_j": "CA",
                 "d_upper": 6.0},
            ],
            res_ids=struct_res_ids,
        )
    """
    if atom_names is None:
        atom_names = ["N", "CA", "C"]

    res_ids_arr = np.asarray(res_ids)
    res_id_to_block: dict[int, int] = {
        int(rid): i for i, rid in enumerate(res_ids_arr)
    }
    n_atoms_per_res = len(atom_names)
    atom_name_to_offset: dict[str, int] = {
        name: i for i, name in enumerate(atom_names)
    }

    def _lookup(res_id: int, atom_name: str) -> int:
        if int(res_id) not in res_id_to_block:
            raise ValueError(
                f"make_noe_restraints: residue {res_id} not found in structure "
                f"(res_ids range {int(res_ids_arr.min())}–{int(res_ids_arr.max())})."
            )
        if atom_name not in atom_name_to_offset:
            raise ValueError(
                f"make_noe_restraints: atom name '{atom_name}' not in "
                f"atom_names {atom_names}.  Supply a custom atom_names list "
                f"or map indices externally."
            )
        block = res_id_to_block[int(res_id)]
        return block * n_atoms_per_res + atom_name_to_offset[atom_name]

    pairs: list[list[int]] = []
    d_upper_list: list[float] = []
    d_lower_list: list[float | None] = []
    has_lower = False

    for obs in noe_list:
        idx_i = _lookup(obs["res_i"], obs["atom_i"])
        idx_j = _lookup(obs["res_j"], obs["atom_j"])
        pairs.append([idx_i, idx_j])
        d_upper_list.append(float(obs["d_upper"]))
        lower = obs.get("d_lower")
        d_lower_list.append(float(lower) if lower is not None else None)
        if lower is not None:
            has_lower = True

    atom_pairs = jnp.array(pairs, dtype=jnp.int32)
    d_upper_arr = jnp.array(d_upper_list, dtype=jnp.float32)
    d_lower_arr: jnp.ndarray | None = None
    if has_lower:
        d_lower_arr = jnp.array(
            [v if v is not None else 0.0 for v in d_lower_list], dtype=jnp.float32
        )

    return NOELoss(
        atom_pairs=atom_pairs,
        d_upper=d_upper_arr,
        d_lower=d_lower_arr,
        force_const=force_const,
    )
