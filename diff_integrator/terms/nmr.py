"""
diff_integrator/terms/nmr.py — NMR-based loss terms for integrative refinement.

Two classes are provided:

* ``FixedTensorRDCLoss`` (recommended):
    The scientifically correct implementation for structure refinement.
    The Saupe alignment tensor is held **fixed** during each gradient step
    (via ``jax.lax.stop_gradient``) and updated periodically *outside* the
    gradient computation.  This mirrors the standard X-PLOR/CNS/PALES approach
    and prevents the optimizer from trivially driving Q→0 by exploiting the
    degeneracy of the 5-parameter tensor.

    Additional features:

    * **Cross-validation split** (``cv_fraction``): a reproducible random subset
      of RDC measurements is held out of the training loss and used only for
      monitoring via ``evaluate_validation_q()``.  This makes overfitting
      immediately visible — if training Q drops while validation Q stays flat,
      the backbone is being distorted to fit the training RDCs without genuine
      structural improvement.

    * **Auto-weight by overdetermination ratio** (``suggested_weight``): returns
      an advisory weight proportional to ``n_train_rdcs / (5 × 10)`` so that
      well-determined media are up-weighted and underdetermined media are
      down-weighted automatically.

* ``RDCLoss`` (deprecated, monitoring only):
    Fits the tensor analytically inside the gradient.  While mathematically
    differentiable through the SVD, this is **not** suitable for structure
    refinement because it allows the optimizer to satisfy RDCs via degenerate
    backbone distortions rather than genuine structural improvement.  Use only
    for single-step evaluation or monitoring, never inside a training loop.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from diff_biophys.nmr.rdc import (
    calculate_q_factor,
    calculate_rdc_from_tensor,
    fit_saupe_tensor,
)

from diff_integrator.loss import LossTerm

# Minimum number of training RDCs to reliably fit the 5-parameter Saupe tensor.
_MIN_TRAIN_RDCS_FOR_TENSOR = 6

# Ideal overdetermination ratio for suggested_weight() normalisation.
# A ratio of exactly this value maps to base_weight × 1.0.
_IDEAL_RATIO: float = 10.0


class FixedTensorRDCLoss(LossTerm):
    """
    RDC loss for structure refinement with a periodically-updated fixed tensor.

    The Saupe alignment tensor is held fixed (via ``jax.lax.stop_gradient``)
    during gradient computation to prevent the optimizer from exploiting the
    degeneracy between backbone orientation and tensor parameters.  Callers
    are responsible for calling ``maybe_update_tensor(coords, epoch)`` at the
    start of each epoch to keep the tensor current.

    This approach mirrors the standard used by X-PLOR, CNS, and PALES: the
    tensor is fitted from the *current* best structure, then held constant for
    a block of gradient steps, then re-fitted.

    Cross-validation split
    ----------------------
    When ``cv_fraction > 0`` a reproducible random subset of the matched RDC
    measurements is held out of training and evaluated via
    ``evaluate_validation_q(coords)``.  The split is determined at construction
    time using a fixed random seed, so results are fully reproducible.

    The training loss is computed over the training subset only; the
    validation Q is computed with the *training-fitted* tensor applied to the
    held-out subset.  This is intentional — the goal is to test whether the
    *structure* has improved, not whether a freshly fitted tensor could explain
    the held-out data.

    This requires that ``val_q_eval_fn`` is supplied (see below).

    Auto-weight by overdetermination ratio
    ---------------------------------------
    The ``suggested_weight()`` helper returns an advisory weight proportional
    to the overdetermination ratio of the *training* RDCs:

    .. math::

        w = \\mathrm{clamp}\\!\\left(\\frac{n_{\\mathrm{train}}}{5 \\times r_{\\mathrm{ideal}}},\\,
                                  0.1\\,w_{\\mathrm{base}},\\, 2\\,w_{\\mathrm{base}}\\right)

    where :math:`r_{\\mathrm{ideal}} = 10` and the clamp bounds are
    :math:`0.1 \\times w_{\\mathrm{base}}` and :math:`2 \\times w_{\\mathrm{base}}`.
    A medium with exactly 50 training RDCs (ratio 10×) returns
    ``base_weight`` unchanged.

    This is a **hint** only.  Callers retain full control.

    Args:
        loss_fn: The training-set ``loss_fn`` returned by
            ``diff_biophys.nmr.rdc.make_rdc_refinement_fns`` (or a
            CV-split variant).  Signature: ``(coords, fixed_tensor) -> scalar``.
        make_tensor_fn: Fits and returns the current Saupe tensor from backbone
            coordinates.  Signature: ``(coords) -> (3, 3) ndarray``.
        update_interval: Number of epochs between tensor re-fits.  Default 50.
        n_rdcs: Total number of *training* RDC measurements (after any CV split).
            Required for ``suggested_weight()`` and for reporting the
            overdetermination ratio.
        val_q_eval_fn: Optional callable ``(coords) -> float`` that evaluates the
            Q-factor on the held-out validation RDC measurements.  Supply this
            when constructing with ``cv_fraction > 0`` (see
            ``make_rdc_cv_refinement_fns``).
    """

    name: str = "rdc"

    def __init__(
        self,
        loss_fn: Any,
        make_tensor_fn: Any,
        update_interval: int = 50,
        n_rdcs: int | None = None,
        val_q_eval_fn: Any | None = None,
    ) -> None:
        self._loss_fn = loss_fn
        self._make_tensor_fn = make_tensor_fn
        self.update_interval = update_interval
        self._tensor: jnp.ndarray | None = None
        self._epoch: int = 0
        self.n_rdcs: int | None = n_rdcs
        self._val_q_eval_fn = val_q_eval_fn

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def overdetermination_ratio(self) -> float | None:
        """Training RDCs / 5 (Saupe tensor free parameters).

        Returns ``None`` if ``n_rdcs`` was not supplied at construction.
        A ratio < 3 is unreliable; ≥ 8 is considered well-determined.
        """
        if self.n_rdcs is None:
            return None
        return self.n_rdcs / 5.0

    def suggested_weight(self, base_weight: float = 1.0) -> float:
        """Return an advisory loss weight scaled by the overdetermination ratio.

        A medium with exactly 50 training RDCs (ratio 10×, ``_IDEAL_RATIO``)
        returns ``base_weight`` unchanged.  The result is clamped to
        ``[0.1 × base_weight, 2.0 × base_weight]``.

        This is a **hint** only.  Callers retain full control over the final
        weight passed to ``JointLoss``.

        Args:
            base_weight: Reference weight for a medium at the ideal ratio.

        Returns:
            Suggested weight as a float.

        Raises:
            ValueError: If ``n_rdcs`` was not supplied at construction.
        """
        if self.overdetermination_ratio is None:
            raise ValueError(
                "suggested_weight() requires n_rdcs to be supplied at construction."
            )
        ratio = self.overdetermination_ratio
        raw = base_weight * (ratio / _IDEAL_RATIO)
        return float(np.clip(raw, 0.1 * base_weight, 2.0 * base_weight))

    # ------------------------------------------------------------------
    # Tensor management
    # ------------------------------------------------------------------

    def initialize_tensor(self, coords: jnp.ndarray) -> None:
        """Fit the tensor from an initial set of coordinates.

        Must be called once before the first gradient step.

        Args:
            coords: ``(3N, 3)`` backbone atom coordinates.
        """
        self._tensor = self._make_tensor_fn(coords)

    def maybe_update_tensor(self, coords: jnp.ndarray, epoch: int) -> None:
        """Re-fit the tensor if the update interval has elapsed.

        Call this at the start of each epoch, *before* the gradient step.

        Args:
            coords: Current backbone coordinates (outside the gradient tape).
            epoch: The current epoch index (0-based).
        """
        if self._tensor is None or epoch % self.update_interval == 0:
            self._tensor = self._make_tensor_fn(coords)
        self._epoch = epoch

    # ------------------------------------------------------------------
    # Loss evaluation
    # ------------------------------------------------------------------

    def __call__(self, params: Any, coords: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the fixed-tensor RDC MSE loss (training split only).

        The stored tensor is wrapped in ``jax.lax.stop_gradient`` before being
        passed to the underlying loss function, ensuring gradients flow only
        through the coordinate path.

        Args:
            params: Optimization parameters (ignored; present for interface
                compatibility).
            coords: ``(3N, 3)`` backbone atom coordinates.

        Returns:
            Scalar MSE between back-calculated and experimental RDCs.

        Raises:
            RuntimeError: If ``initialize_tensor`` has not been called first.
        """
        if self._tensor is None:
            raise RuntimeError(
                "FixedTensorRDCLoss: tensor not initialized. "
                "Call initialize_tensor(coords) before the first gradient step."
            )
        import typing

        frozen_tensor = jax.lax.stop_gradient(self._tensor)
        return typing.cast(jnp.ndarray, self._loss_fn(coords, frozen_tensor))

    def evaluate_validation_q(self, coords: jnp.ndarray) -> float | None:
        """Evaluate the Q-factor on the held-out validation RDC measurements.

        Uses the current (training-fitted) tensor applied to the held-out
        subset.  This tests whether the *structure* has improved without
        access to the validation data during training.

        Args:
            coords: Current backbone coordinates.

        Returns:
            Scalar Q-factor on the validation split, or ``None`` if no
            ``val_q_eval_fn`` was supplied at construction.
        """
        if self._val_q_eval_fn is None:
            return None
        if self._tensor is None:
            raise RuntimeError(
                "FixedTensorRDCLoss: tensor not initialized. "
                "Call initialize_tensor(coords) before evaluate_validation_q()."
            )
        frozen_tensor = jax.lax.stop_gradient(self._tensor)
        return float(self._val_q_eval_fn(coords, frozen_tensor))


# ---------------------------------------------------------------------------
# CV-aware factory function
# ---------------------------------------------------------------------------


def make_rdc_cv_refinement_fns(
    exp_res_ids: np.ndarray,
    exp_rdcs: np.ndarray,
    struct_res_ids: np.ndarray,
    d_max: float = 21.7,
    cv_fraction: float = 0.2,
    cv_seed: int = 42,
) -> tuple[Any, Any, Any, Any, int, int]:
    """Build train/validation-split RDC callables for ``FixedTensorRDCLoss``.

    This is a CV-aware wrapper around the standard
    ``diff_biophys.nmr.rdc.make_rdc_refinement_fns`` pattern.  It splits the
    matched RDC measurements into a training set (used for the gradient) and a
    held-out validation set (used only for monitoring via
    ``evaluate_validation_q``).

    The split is reproducible: the same ``cv_seed`` always produces the same
    partition.  Only the *training* measurements are used to fit or evaluate
    the Saupe tensor during optimization; the validation Q is computed by
    applying the *training-fitted* tensor to the held-out measurements.

    Args:
        exp_res_ids: ``(M,)`` residue IDs where RDCs were measured.
        exp_rdcs: ``(M,)`` experimental RDC values in Hz.
        struct_res_ids: ``(N,)`` residue IDs present in the structure.
        d_max: Maximum dipolar coupling constant in Hz.  Default 21.7 Hz (¹⁵N–¹H).
        cv_fraction: Fraction of *matched* RDC measurements to hold out.
            Must be in ``(0, 1)``.  The actual number held out is
            ``max(1, round(n_matched × cv_fraction))``.
        cv_seed: NumPy random seed for the index shuffle.  Default 42.

    Returns:
        Tuple ``(loss_fn, q_eval_fn, make_tensor_fn, n_train, n_val)`` where:

        * ``loss_fn(coords, fixed_tensor) -> scalar``: MSE on training RDCs.
        * ``q_eval_fn(coords) -> scalar``: Q-factor on all matched RDCs
          (standard monitoring, refits tensor from scratch — do NOT use inside
          gradient).
        * ``make_tensor_fn(coords) -> (3,3)``: fits tensor from *training* RDCs.
        * ``val_q_eval_fn(coords, fixed_tensor) -> scalar``: Q-factor on the
          held-out validation RDCs using the *provided* tensor (no refitting).
          Pass to ``FixedTensorRDCLoss(val_q_eval_fn=...)``.
        * ``n_train``: number of training RDC measurements.
        * ``n_val``: number of validation RDC measurements.

    Raises:
        ValueError: If no residues overlap, or if the training split would be
            too small to fit the Saupe tensor (< ``_MIN_TRAIN_RDCS_FOR_TENSOR``).

    Example::

        loss_fn, q_eval_fn, make_tensor_fn, val_q_fn, n_train, n_val = \\
            make_rdc_cv_refinement_fns(rdc["res_id"], rdc["rdc"], res_ids, cv_fraction=0.2)

        rdc_term = FixedTensorRDCLoss(
            loss_fn=loss_fn,
            make_tensor_fn=make_tensor_fn,
            n_rdcs=n_train,
            val_q_eval_fn=val_q_fn,
        )
    """
    from diff_biophys.nmr.rdc import nh_bond_vectors  # type: ignore[import]

    if not 0.0 < cv_fraction < 1.0:
        raise ValueError(f"cv_fraction must be in (0, 1), got {cv_fraction}")

    # ---- Match residues -------------------------------------------------------
    res_id_to_idx = {int(rid): i for i, rid in enumerate(struct_res_ids)}
    matched_struct_idx: list[int] = []
    matched_rdcs_list: list[float] = []
    for rid, rdc_val in zip(exp_res_ids, exp_rdcs, strict=False):
        if int(rid) in res_id_to_idx:
            matched_struct_idx.append(res_id_to_idx[int(rid)])
            matched_rdcs_list.append(float(rdc_val))

    if not matched_struct_idx:
        raise ValueError(
            "No residues overlap between exp_res_ids and struct_res_ids."
        )

    n_matched = len(matched_struct_idx)
    all_rdcs = np.array(matched_rdcs_list, dtype=np.float32)
    all_atom_idx = np.array(matched_struct_idx, dtype=np.int32)

    # ---- Build CV split -------------------------------------------------------
    rng = np.random.default_rng(cv_seed)
    perm = rng.permutation(n_matched)
    n_val = max(1, int(round(n_matched * cv_fraction)))
    n_train = n_matched - n_val

    if n_train < _MIN_TRAIN_RDCS_FOR_TENSOR:
        raise ValueError(
            f"cv_fraction={cv_fraction} leaves only {n_train} training RDCs "
            f"(need ≥ {_MIN_TRAIN_RDCS_FOR_TENSOR} to fit the Saupe tensor). "
            f"Reduce cv_fraction or disable cross-validation for this medium."
        )

    train_idx = perm[:n_train]  # indices into matched arrays
    val_idx = perm[n_train:]

    train_atom_idx = jnp.array(all_atom_idx[train_idx], dtype=jnp.int32)
    train_rdcs = jnp.array(all_rdcs[train_idx], dtype=jnp.float32)
    val_atom_idx = jnp.array(all_atom_idx[val_idx], dtype=jnp.int32)
    val_rdcs = jnp.array(all_rdcs[val_idx], dtype=jnp.float32)
    all_atom_idx_jax = jnp.array(all_atom_idx, dtype=jnp.int32)
    all_rdcs_jax = jnp.array(all_rdcs, dtype=jnp.float32)

    # ---- Closures ------------------------------------------------------------
    from typing import cast as _cast

    def _nh_train(coords: jnp.ndarray) -> jnp.ndarray:
        return _cast(jnp.ndarray, nh_bond_vectors(coords)[train_atom_idx])

    def _nh_val(coords: jnp.ndarray) -> jnp.ndarray:
        return _cast(jnp.ndarray, nh_bond_vectors(coords)[val_atom_idx])

    def _nh_all(coords: jnp.ndarray) -> jnp.ndarray:
        return _cast(jnp.ndarray, nh_bond_vectors(coords)[all_atom_idx_jax])

    def loss_fn(coords: jnp.ndarray, fixed_tensor: jnp.ndarray) -> jnp.ndarray:
        """Fixed-tensor MSE on training RDCs; gradient flows through coords."""
        tensor = jax.lax.stop_gradient(fixed_tensor)
        calc = calculate_rdc_from_tensor(_nh_train(coords), tensor, d_max=d_max)
        return _cast(jnp.ndarray, jnp.mean((calc - train_rdcs) ** 2))

    def q_eval_fn(coords: jnp.ndarray) -> jnp.ndarray:
        """Re-fit tensor from ALL matched RDCs and return Q (monitoring only)."""
        nh = _nh_all(coords)
        tensor = fit_saupe_tensor(nh, all_rdcs_jax, d_max=d_max)
        calc = calculate_rdc_from_tensor(nh, tensor, d_max=d_max)
        return _cast(jnp.ndarray, calculate_q_factor(calc, all_rdcs_jax))

    def make_tensor_fn(coords: jnp.ndarray) -> jnp.ndarray:
        """Fit Saupe tensor from *training* RDCs only (for periodic updates)."""
        return _cast(jnp.ndarray, fit_saupe_tensor(_nh_train(coords), train_rdcs, d_max=d_max))

    def val_q_eval_fn(coords: jnp.ndarray, fixed_tensor: jnp.ndarray) -> jnp.ndarray:
        """Q-factor on held-out validation RDCs using the provided tensor."""
        tensor = jax.lax.stop_gradient(fixed_tensor)
        calc = calculate_rdc_from_tensor(_nh_val(coords), tensor, d_max=d_max)
        return _cast(jnp.ndarray, calculate_q_factor(calc, val_rdcs))

    return loss_fn, q_eval_fn, make_tensor_fn, val_q_eval_fn, n_train, n_val


class RDCLoss(LossTerm):
    """
    .. deprecated::
        ``RDCLoss`` fits the Saupe tensor inside the gradient computation.
        This is **not** suitable for structure refinement; use
        ``FixedTensorRDCLoss`` instead.  See module docstring for details.

    Computes the RDC loss by analytically fitting the Saupe tensor at each
    gradient step.  Because the tensor is fitted inside the gradient tape, the
    optimizer can exploit the 5 tensor degrees of freedom to trivially reduce
    the loss without improving the structure.  This is suitable only for
    single-step monitoring or evaluation (where no gradient is computed).

    Args:
        atom_pairs: ``(N, 2)`` indices of the two atoms forming the bond.
        exp_rdcs: ``(N,)`` experimental RDC values in Hz.
        d_max: Maximum dipolar coupling constant in Hz (default 21.7 for ¹⁵N–¹H).
        loss_type: ``'mse'`` or ``'q_factor'``.
    """

    def __init__(
        self,
        atom_pairs: jnp.ndarray,
        exp_rdcs: jnp.ndarray,
        d_max: float = 21.7,
        loss_type: str = "mse",
    ) -> None:
        import warnings

        warnings.warn(
            "RDCLoss fits the tensor inside the gradient and is not suitable for "
            "structure refinement. Use FixedTensorRDCLoss instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.atom_pairs = atom_pairs
        self.exp_rdcs = exp_rdcs
        self.d_max = d_max
        self.loss_type = loss_type

    def __call__(self, params: Any, coords: jnp.ndarray) -> jnp.ndarray:
        coords_a = coords[self.atom_pairs[:, 0]]
        coords_b = coords[self.atom_pairs[:, 1]]
        bond_vecs = coords_a - coords_b
        norms = jnp.linalg.norm(bond_vecs, axis=-1, keepdims=True)
        bond_vecs_norm = bond_vecs / (norms + 1e-12)
        saupe_tensor = fit_saupe_tensor(
            bond_vectors=bond_vecs_norm,
            experimental_rdcs=self.exp_rdcs,
            d_max=self.d_max,
        )
        calc_rdcs = calculate_rdc_from_tensor(
            bond_vectors=bond_vecs_norm,
            saupe_tensor=saupe_tensor,
            d_max=self.d_max,
        )
        if self.loss_type == "mse":
            return jnp.mean((calc_rdcs - self.exp_rdcs) ** 2)
        elif self.loss_type == "q_factor":
            diff_sq = jnp.sum((calc_rdcs - self.exp_rdcs) ** 2)
            exp_sq = jnp.sum(self.exp_rdcs**2)
            q = jnp.sqrt(diff_sq / jnp.maximum(exp_sq, 1e-10))
            return jnp.where(exp_sq > 0.0, q, 0.0)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
