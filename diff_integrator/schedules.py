"""
diff_integrator/schedules.py — Weight schedule utilities for integrative refinement.

Weight schedules allow loss term weights to evolve dynamically during
optimization.  The primary motivation is the **annealed geometry restraint**
pattern: start with a strong positional anchor (preventing fold distortion in
early epochs) and gradually relax it so that experimental gradients dominate
in later epochs.

Usage with ``IntegrativeRefiner``::

    from diff_integrator.schedules import ExponentialDecaySchedule
    from diff_integrator.optimizer import IntegrativeRefiner

    # Geometry term is index 0 in the JointLoss list
    schedule = ExponentialDecaySchedule(
        initial_weight=10.0,
        final_weight=0.1,
        decay_epochs=300,
    )

    result = refiner.run(
        init_params=...,
        epochs=500,
        weight_schedules={0: schedule},
    )

    # Inspect how the weight evolved
    print(result.weight_history[0])  # list of 500 float values

Any callable with signature ``(epoch: int) -> float`` can be used as a schedule.
``ExponentialDecaySchedule`` is the most common choice for structural refinement.
"""

import math


class ExponentialDecaySchedule:
    """Exponential decay weight schedule.

    Returns a weight that decreases smoothly from ``initial_weight`` toward
    ``final_weight`` over ``decay_epochs`` epochs:

    .. math::

        w(t) = w_f + (w_i - w_f) \\cdot \\exp\\!\\left(-\\frac{t}{\\tau}\\right)

    where :math:`t` is the current epoch index (0-based), :math:`w_i` is
    ``initial_weight``, :math:`w_f` is ``final_weight``, and :math:`\\tau` is
    ``decay_epochs``.

    After ``decay_epochs`` the weight has decayed to approximately
    :math:`w_f + 0.368 \\cdot (w_i - w_f)` (one time-constant).  After
    ``3 * decay_epochs`` it is within 5 % of ``final_weight``.

    Parameters
    ----------
    initial_weight:
        Weight at epoch 0.  Should be the larger of the two values when used
        as a restraint that is being relaxed.
    final_weight:
        Asymptotic weight as epoch → ∞.  Must be ≥ 0.
    decay_epochs:
        Time constant τ in epochs.  Larger values produce a slower decay.
        Must be > 0.

    Examples
    --------
    >>> sched = ExponentialDecaySchedule(10.0, 0.1, 200)
    >>> round(sched(0), 4)
    10.0
    >>> round(sched(200), 4)       # one time-constant: ~37 % of range remaining
    3.7455
    >>> round(sched(1000), 6)      # essentially at final_weight
    0.100454
    """

    def __init__(
        self,
        initial_weight: float,
        final_weight: float,
        decay_epochs: int,
    ) -> None:
        if decay_epochs <= 0:
            raise ValueError(f"decay_epochs must be > 0, got {decay_epochs}")
        if final_weight < 0:
            raise ValueError(f"final_weight must be >= 0, got {final_weight}")
        self.initial_weight = initial_weight
        self.final_weight = final_weight
        self.decay_epochs = decay_epochs

    def __call__(self, epoch: int) -> float:
        """Return the weight for the given epoch index.

        Parameters
        ----------
        epoch:
            Current epoch index (0-based).

        Returns
        -------
        float
            Weight value for this epoch.
        """
        decay = math.exp(-epoch / self.decay_epochs)
        return self.final_weight + (self.initial_weight - self.final_weight) * decay

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ExponentialDecaySchedule("
            f"initial_weight={self.initial_weight}, "
            f"final_weight={self.final_weight}, "
            f"decay_epochs={self.decay_epochs})"
        )


class LinearSchedule:
    """Linear interpolation weight schedule.

    Returns a weight that changes at a constant rate from ``initial_weight``
    to ``final_weight`` over ``decay_epochs`` epochs, then holds at
    ``final_weight``::

        w(t) = initial_weight + (final_weight - initial_weight) * min(t / decay_epochs, 1)

    Parameters
    ----------
    initial_weight:
        Weight at epoch 0.
    final_weight:
        Weight at ``decay_epochs`` (and all subsequent epochs).  May be larger
        or smaller than ``initial_weight`` to produce warm-up or cool-down
        schedules.  Must be ≥ 0.
    decay_epochs:
        Number of epochs over which the transition occurs.  Must be > 0.

    Examples
    --------
    >>> sched = LinearSchedule(10.0, 0.0, 100)
    >>> round(sched(0), 4)
    10.0
    >>> round(sched(50), 4)
    5.0
    >>> round(sched(100), 4)
    0.0
    >>> round(sched(200), 4)   # clamped at final_weight
    0.0
    """

    def __init__(
        self,
        initial_weight: float,
        final_weight: float,
        decay_epochs: int,
    ) -> None:
        if decay_epochs <= 0:
            raise ValueError(f"decay_epochs must be > 0, got {decay_epochs}")
        if final_weight < 0:
            raise ValueError(f"final_weight must be >= 0, got {final_weight}")
        self.initial_weight = initial_weight
        self.final_weight = final_weight
        self.decay_epochs = decay_epochs

    def __call__(self, epoch: int) -> float:
        """Return the weight for the given epoch index.

        Parameters
        ----------
        epoch:
            Current epoch index (0-based).

        Returns
        -------
        float
            Weight value for this epoch.
        """
        t = min(epoch / self.decay_epochs, 1.0)
        return self.initial_weight + (self.final_weight - self.initial_weight) * t

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"LinearSchedule("
            f"initial_weight={self.initial_weight}, "
            f"final_weight={self.final_weight}, "
            f"decay_epochs={self.decay_epochs})"
        )


class CosineAnnealingSchedule:
    """Cosine annealing weight schedule.

    Returns a weight that follows a half-cosine curve from ``initial_weight``
    to ``final_weight`` over ``decay_epochs`` epochs, then holds at
    ``final_weight``:

    .. math::

        w(t) = w_i + (w_f - w_i) \\cdot \\frac{1 - \\cos(\\pi \\cdot t')}{2}, \\quad
        t' = \\min\\!\\left(\\frac{t}{\\tau}, 1\\right)

    The cosine curve starts flat near ``initial_weight``, accelerates through
    the midpoint, and decelerates to land smoothly at ``final_weight``.  This
    avoids the abrupt start of a linear schedule while still completing the
    transition in a predictable number of epochs.

    Parameters
    ----------
    initial_weight:
        Weight at epoch 0.
    final_weight:
        Asymptotic weight at ``decay_epochs``.  Must be ≥ 0.
    decay_epochs:
        Number of epochs for one half-cycle.  Must be > 0.

    Examples
    --------
    >>> sched = CosineAnnealingSchedule(10.0, 0.0, 100)
    >>> round(sched(0), 6)
    10.0
    >>> round(sched(100), 6)
    0.0
    >>> # At midpoint (t=50) the cosine is exactly at the half-way value
    >>> round(sched(50), 4)
    5.0
    """

    def __init__(
        self,
        initial_weight: float,
        final_weight: float,
        decay_epochs: int,
    ) -> None:
        if decay_epochs <= 0:
            raise ValueError(f"decay_epochs must be > 0, got {decay_epochs}")
        if final_weight < 0:
            raise ValueError(f"final_weight must be >= 0, got {final_weight}")
        self.initial_weight = initial_weight
        self.final_weight = final_weight
        self.decay_epochs = decay_epochs

    def __call__(self, epoch: int) -> float:
        """Return the weight for the given epoch index.

        Parameters
        ----------
        epoch:
            Current epoch index (0-based).

        Returns
        -------
        float
            Weight value for this epoch.
        """
        t_prime = min(epoch / self.decay_epochs, 1.0)
        cos_factor = (1.0 - math.cos(math.pi * t_prime)) / 2.0
        return self.initial_weight + (self.final_weight - self.initial_weight) * cos_factor

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"CosineAnnealingSchedule("
            f"initial_weight={self.initial_weight}, "
            f"final_weight={self.final_weight}, "
            f"decay_epochs={self.decay_epochs})"
        )
