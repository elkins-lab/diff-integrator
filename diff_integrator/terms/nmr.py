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
from diff_biophys.nmr.rdc import calculate_rdc_from_tensor, fit_saupe_tensor

from diff_integrator.loss import LossTerm


class FixedTensorRDCLoss(LossTerm):
    """
    RDC MSE loss for structure refinement using a periodically-updated fixed tensor.

    The Saupe alignment tensor is held fixed (via ``jax.lax.stop_gradient``)
    during gradient computation to prevent the optimizer from exploiting the
    degeneracy between backbone orientation and tensor parameters.  Callers
    are responsible for calling ``update_tensor(coords)`` at an appropriate
    interval (e.g. every 50–100 steps) to keep the tensor current.

    This approach is the standard used by X-PLOR, CNS, and PALES: the tensor
    is fitted from the *current* best structure, then held constant for a block
    of gradient steps, then re-fitted.

    Example usage inside a training loop::

        rdc_loss = FixedTensorRDCLoss(loss_fn, make_tensor_fn, update_interval=50)

        def step_fn(params, opt_state, epoch):
            coords = build_backbone(*params)
            rdc_loss.maybe_update_tensor(coords, epoch)
            ...

    Args:
        loss_fn: The ``loss_fn`` returned by
            ``diff_biophys.nmr.rdc.make_rdc_refinement_fns``.
            Its signature is ``(coords, fixed_tensor) -> scalar``.
        make_tensor_fn: The ``make_tensor_fn`` returned by
            ``make_rdc_refinement_fns``.  Fits and returns the current
            Saupe tensor from backbone coordinates.
        update_interval: Number of epochs between tensor re-fits.
            Lower values track the evolving structure more closely but
            slow training slightly.  Default is 50.
    """

    def __init__(
        self,
        loss_fn: Any,
        make_tensor_fn: Any,
        update_interval: int = 50,
    ) -> None:
        self._loss_fn = loss_fn
        self._make_tensor_fn = make_tensor_fn
        self.update_interval = update_interval
        self._tensor: jnp.ndarray | None = None
        self._epoch: int = 0

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

    def __call__(self, params: Any, coords: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the fixed-tensor RDC MSE loss.

        The stored tensor is wrapped in ``jax.lax.stop_gradient`` before being
        passed to the underlying loss function, ensuring gradients flow only
        through the coordinate path.

        Args:
            params: Optimization parameters (ignored; present for interface compatibility).
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
        # Stop gradient flows through the fixed tensor; gradients only
        # flow through the coords path inside loss_fn.
        import typing

        frozen_tensor = jax.lax.stop_gradient(self._tensor)
        return typing.cast(jnp.ndarray, self._loss_fn(coords, frozen_tensor))


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
