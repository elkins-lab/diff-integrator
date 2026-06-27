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
    """Combines multiple LossTerms with weights."""

    def __init__(self, terms: list[tuple[LossTerm, float]]):
        """
        Args:
            terms: A list of tuples containing (LossTerm, weight).
        """
        self.terms = terms

    def __call__(self, params: Any, coords: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate the total weighted loss.

        Args:
            params: The parameters being optimized.
            coords: (N, 3) atomic coordinates resulting from the kinematics function.

        Returns:
            A scalar jnp.ndarray representing the total loss.
        """
        total_loss = jnp.array(0.0)
        for term, weight in self.terms:
            total_loss += weight * term(params, coords)
        return total_loss

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

    def evaluate_terms(
        self, params: Any, coords: jnp.ndarray
    ) -> dict[str, float]:
        """Evaluate each term individually and return a dict of
        name -> weighted loss value."""
        results: dict[str, float] = {}
        for i, (term, weight) in enumerate(self.terms):
            term_name = term.name or f"term_{i}"
            results[term_name] = float(weight * term(params, coords))
        return results
