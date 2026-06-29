# ⏳ Schedules API

The `diff_integrator.schedules` module provides weight schedule utilities for integrative refinement.

Weight schedules allow loss term weights to evolve dynamically during optimization. The primary motivation is the **annealed geometry restraint** pattern: start with a strong positional anchor (preventing fold distortion in early epochs) and gradually relax it so that experimental gradients dominate in later epochs.

---

## `ExponentialDecaySchedule`

`diff_integrator.schedules.ExponentialDecaySchedule`

Exponential decay weight schedule. Returns a weight that decreases smoothly from `initial_weight` toward `final_weight` over `decay_epochs` epochs.

### Constructor

```python
ExponentialDecaySchedule(
    initial_weight: float,
    final_weight: float,
    decay_epochs: int,
)
```

---

## `LinearSchedule`

`diff_integrator.schedules.LinearSchedule`

Linear interpolation weight schedule. Returns a weight that changes at a constant rate from `initial_weight` to `final_weight` over `decay_epochs` epochs, then holds at `final_weight`.

### Constructor

```python
LinearSchedule(
    initial_weight: float,
    final_weight: float,
    decay_epochs: int,
)
```

---

## `CosineAnnealingSchedule`

`diff_integrator.schedules.CosineAnnealingSchedule`

Cosine annealing weight schedule. Returns a weight that follows a half-cosine curve from `initial_weight` to `final_weight` over `decay_epochs` epochs, then holds at `final_weight`.

### Constructor

```python
CosineAnnealingSchedule(
    initial_weight: float,
    final_weight: float,
    decay_epochs: int,
)
```
