
import jax.numpy as jnp
from jax import grad

from diff_integrator.terms.ramachandran import RamachandranLoss



def test_gly_epsilon_basin_lower_penalty_than_general():
    """GLY should penalise the ε-basin (φ≈+60°, ψ≈-120°) less than ALA does.

    The ε-basin is unique to Gly (no Cβ steric clash) and is explicitly
    included as a 4th basin in the GLY lookup.  The general (ALA) model has
    no such basin and should assign higher penalty at those coordinates.
    """
    phi = jnp.array([1.05])   # +60°
    psi = jnp.array([-2.09])  # -120°

    loss_gly = RamachandranLoss(residue_types=["GLY"])
    loss_ala = RamachandranLoss(residue_types=["ALA"])

    penalty_gly = float(loss_gly((phi, psi), None))
    penalty_ala = float(loss_ala((phi, psi), None))

    assert penalty_gly < penalty_ala, (
        f"GLY ε-basin penalty ({penalty_gly:.4f}) should be < "
        f"ALA penalty ({penalty_ala:.4f}) at same coordinates"
    )


def test_pro_down_pucker_low_penalty():
    """PRO should give low penalty near the down-pucker centre (φ≈-65°, ψ≈+150°)."""
    phi = jnp.array([-1.13])
    psi = jnp.array([2.62])

    loss_pro = RamachandranLoss(residue_types=["PRO"])
    penalty = float(loss_pro((phi, psi), None))

    assert penalty < 0.1, f"PRO down-pucker penalty should be near 0, got {penalty:.4f}"


def test_pro_phi_high_penalty_away_from_ring():
    """PRO should strongly penalise φ values far from −65° (ring constraint)."""
    phi_bad = jnp.array([1.05])   # +60°: physically impossible for Pro
    phi_good = jnp.array([-1.13])
    psi = jnp.array([2.62])

    loss_pro = RamachandranLoss(residue_types=["PRO"])
    assert float(loss_pro((phi_bad, psi), None)) > float(loss_pro((phi_good, psi), None))


def test_sequence_aware_gradient_finite():
    """Sequence-aware path must produce finite, non-NaN gradients."""
    residue_types = ["ALA", "GLY", "PRO", "VAL", "ILE", "LEU"]
    phi = jnp.array([-1.05, 1.05, -1.13, -2.09, -1.05, -2.09])
    psi = jnp.array([-0.78, 0.78, 2.62,  2.35, -0.78,  2.35])

    loss = RamachandranLoss(residue_types=residue_types)

    def fn(p: jnp.ndarray) -> jnp.ndarray:
        return loss((p, psi), None)

    grads = grad(fn)(phi)
    assert jnp.all(jnp.isfinite(grads)), "Gradients contain NaN or Inf"


def test_sequence_aware_returns_scalar():
    """Both uniform and sequence-aware modes must return scalar outputs."""
    phi = jnp.array([-1.05, -2.09])
    psi = jnp.array([-0.78,  2.35])

    out_uniform = RamachandranLoss()((phi, psi), None)
    out_aware = RamachandranLoss(residue_types=["ALA", "GLY"])((phi, psi), None)

    assert out_uniform.shape == ()
    assert out_aware.shape == ()


# ---------------------------------------------------------------------------
# Best-checkpoint tests (live in test_ramachandran_loss.py for proximity
# to the motivating improvement, but test optimizer behaviour)
# ---------------------------------------------------------------------------


def test_best_params_populated_after_run():
    """IntegrativeRefiner.run() should always populate best_params and best_epoch."""
    from typing import Any

    from diff_integrator.loss import JointLoss, LossTerm
    from diff_integrator.optimizer import IntegrativeRefiner

    class QuadLoss(LossTerm):
        name = "quad"
        def __call__(self, params: Any, coords: jnp.ndarray) -> jnp.ndarray:
            return jnp.mean(params ** 2)

    loss_fn = JointLoss([(QuadLoss(), 1.0)])
    refiner = IntegrativeRefiner(loss_fn=loss_fn)
    result = refiner.run(init_params=jnp.ones((3,)) * 5.0, epochs=50)

    assert result.best_params is not None
    assert 0 <= result.best_epoch < 50
    # The best loss should be ≤ the final training loss
    # (loss_history[-1] is the last iterate's loss)
    best_train_loss = result.loss_history[result.best_epoch]
    final_train_loss = result.loss_history[-1]
    assert best_train_loss <= final_train_loss + 1e-6


def test_best_params_by_validation_loss():
    """When validation_loss is provided, best_params tracks its minimum."""
    from typing import Any

    from diff_integrator.loss import JointLoss, LossTerm
    from diff_integrator.optimizer import IntegrativeRefiner

    class TrainLoss(LossTerm):
        name = "train"
        def __call__(self, params: Any, coords: jnp.ndarray) -> jnp.ndarray:
            return jnp.mean(params ** 2)

    class ValLoss(LossTerm):
        name = "val"
        target = jnp.array([2.0, 2.0, 2.0])
        def __call__(self, params: Any, coords: jnp.ndarray) -> jnp.ndarray:
            return jnp.mean((params - self.target) ** 2)

    train_fn = JointLoss([(TrainLoss(), 1.0)])
    val_fn = JointLoss([(ValLoss(), 1.0)])
    refiner = IntegrativeRefiner(loss_fn=train_fn)

    result = refiner.run(
        init_params=jnp.zeros((3,)),
        epochs=30,
        validation_loss=val_fn,
    )

    assert len(result.validation_history) == 30
    # best_epoch should correspond to the minimum in validation_history
    best_val = result.validation_history[result.best_epoch]
    assert best_val == min(result.validation_history)
