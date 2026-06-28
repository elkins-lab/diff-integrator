"""
diff_integrator/loss.py — Core loss abstractions for integrative refinement.

Provides two building blocks:

* ``LossTerm`` — abstract base class for any differentiable constraint
  (chemical shifts, RDCs, SAXS, geometry, etc.).
* ``JointLoss`` — weighted sum of ``LossTerm`` objects used as the single
  objective passed to ``IntegrativeRefiner``.

The ``JointLoss.set_weight`` method allows the optimizer to update individual
term weights mid-run, which is needed by the weight-schedule mechanism in
``IntegrativeRefiner``.

The ``JointLoss.freeze_term`` / ``unfreeze_term`` pair supports multi-phase
refinement: frozen terms are excluded from the gradient objective but remain
active in ``evaluate_terms`` for diagnostic monitoring.
"""

import abc
from typing import Any

import jax.numpy as jnp


class LossTerm(abc.ABC):
    """Abstract base class for all differentiable loss terms."""

    name: str = ""

    @abc.abstractmethod
    def __call__(self, params: Any, coords: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate the loss term.

        Args:
            params: The parameters being optimized (e.g., coordinates, or internal angles).
            coords: (N, 3) atomic coordinates resulting from the kinematics function.

        Returns:
            A scalar jnp.ndarray representing the loss.
        """
        pass  # pragma: no cover


class JointLoss:
    """Combines multiple ``LossTerm`` objects with scalar weights.

    Terms can be temporarily excluded from the gradient objective via
    ``freeze_term`` while remaining visible to ``evaluate_terms`` for
    diagnostic monitoring.  This supports multi-phase refinement workflows
    such as "geometry-only for the first 200 epochs, then add experimental
    terms" without rebuilding the loss object.

    Example::

        joint_loss = JointLoss([
            (geometry_loss, 5.0),   # index 0
            (rdc_loss,      1.0),   # index 1
        ])

        # Phase 1: geometry only
        joint_loss.freeze_term(1)
        refiner.run(init_params=..., epochs=200, ...)

        # Phase 2: add RDC term
        joint_loss.unfreeze_term(1)
        refiner.run(init_params=result.final_params, epochs=300, ...)
    """

    def __init__(self, terms: list[tuple[LossTerm, float]]):
        """
        Args:
            terms: A list of tuples containing ``(LossTerm, weight)``.
        """
        self.terms = terms
        self._frozen: set[int] = set()

    def __call__(self, params: Any, coords: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate the total weighted loss, skipping any frozen terms.

        Args:
            params: The parameters being optimized.
            coords: (N, 3) atomic coordinates resulting from the kinematics function.

        Returns:
            A scalar jnp.ndarray representing the total loss.
        """
        total_loss = jnp.array(0.0)
        for i, (term, weight) in enumerate(self.terms):
            if i in self._frozen:
                continue
            total_loss += weight * term(params, coords)
        return total_loss

    # ------------------------------------------------------------------
    # Weight management
    # ------------------------------------------------------------------

    def set_weight(self, term_index: int, weight: float) -> None:
        """Update the weight of a single term in-place.

        This is called by ``IntegrativeRefiner`` each epoch when a
        ``weight_schedules`` mapping is provided.  It mutates ``self.terms``
        so the new weight is visible to the next gradient computation.

        Args:
            term_index: Zero-based index into ``self.terms``.
            weight: New weight value for that term.

        Raises:
            IndexError: If ``term_index`` is out of range.
        """
        if term_index < 0 or term_index >= len(self.terms):
            raise IndexError(
                f"term_index {term_index} is out of range for "
                f"JointLoss with {len(self.terms)} terms."
            )
        term, _ = self.terms[term_index]
        self.terms[term_index] = (term, weight)

    # ------------------------------------------------------------------
    # Term freezing (multi-phase refinement)
    # ------------------------------------------------------------------

    def freeze_term(self, term_index: int) -> None:
        """Exclude a term from the gradient objective.

        The frozen term is still evaluated in ``evaluate_terms`` for
        diagnostic monitoring but contributes zero to the loss returned
        by ``__call__``.  ``IntegrativeRefiner`` automatically passes a
        zero weight for frozen terms so the JIT-compiled step function
        remains correct without recompilation.

        Args:
            term_index: Zero-based index into ``self.terms``.

        Raises:
            IndexError: If ``term_index`` is out of range.
        """
        if term_index < 0 or term_index >= len(self.terms):
            raise IndexError(
                f"freeze_term: term_index {term_index} is out of range for "
                f"JointLoss with {len(self.terms)} terms."
            )
        self._frozen.add(term_index)

    def unfreeze_term(self, term_index: int) -> None:
        """Re-enable a previously frozen term.

        If the term was not frozen this is a no-op.

        Args:
            term_index: Zero-based index into ``self.terms``.

        Raises:
            IndexError: If ``term_index`` is out of range.
        """
        if term_index < 0 or term_index >= len(self.terms):
            raise IndexError(
                f"unfreeze_term: term_index {term_index} is out of range for "
                f"JointLoss with {len(self.terms)} terms."
            )
        self._frozen.discard(term_index)

    def is_frozen(self, term_index: int) -> bool:
        """Return ``True`` if the term at ``term_index`` is currently frozen.

        Args:
            term_index: Zero-based index into ``self.terms``.

        Returns:
            ``True`` if frozen, ``False`` otherwise.
        """
        return term_index in self._frozen

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def evaluate_terms(
        self,
        params: Any,
        coords: jnp.ndarray,
        unweighted: bool = False,
    ) -> dict[str, float]:
        """Evaluate each term individually and return a dict of name -> value.

        Args:
            params: The parameters being optimized.
            coords: (N, 3) atomic coordinates.
            unweighted: If ``True``, return each term's raw (unweighted) scalar
                so that values are independent of the current weight schedule
                and comparable across runs.  If ``False`` (default), return
                ``weight * term(...)`` — the original behaviour.

        Returns:
            Dict mapping term name (or ``"term_i"`` fallback) to float value.
            Frozen terms are included (for monitoring) and their keys are
            suffixed with ``"(frozen)"`` to make their status visible.
        """
        results: dict[str, float] = {}
        for i, (term, weight) in enumerate(self.terms):
            term_name = term.name or f"term_{i}"
            if i in self._frozen:
                term_name = f"{term_name}(frozen)"
            raw = float(term(params, coords))
            results[term_name] = raw if unweighted else raw * weight
        return results
