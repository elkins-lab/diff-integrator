import jax.numpy as jnp
import numpy as np
import pytest

from diff_integrator.loss import JointLoss
from diff_integrator.optimizer import IntegrativeRefiner
from diff_integrator.terms.geometry import GeometryLoss
from diff_integrator.terms.saxs import SAXSLoss

from diff_biophys.geometry.macroscopic import compute_rg
from diff_biophys.saxs.kernels import debye_saxs

@pytest.fixture
def small_protein_coords():
    """Generates a small synthetic backbone for SAXS testing."""
    rng = np.random.default_rng(42)
    # A random walk of 30 C-alpha atoms
    coords = np.zeros((30, 3))
    for i in range(1, 30):
        step = rng.normal(size=3)
        step = 3.8 * step / np.linalg.norm(step)
        coords[i] = coords[i-1] + step
    return jnp.array(coords, dtype=jnp.float32)

@pytest.fixture
def saxs_setup(small_protein_coords):
    """Sets up SAXS parameters."""
    q_values = jnp.linspace(0.01, 0.3, 50, dtype=jnp.float32)
    # Mock uniform form factors
    form_factors = jnp.ones((len(small_protein_coords), len(q_values)), dtype=jnp.float32)
    return q_values, form_factors

def test_saxs_joint_refinement_science(small_protein_coords, saxs_setup):
    """
    Validates SAXS structural refinement.
    Starts with a structure, perturbs it to expand the Rg, then optimizes against
    the SAXS curve of the original structure to compress it back while maintaining geometry.
    """
    q_values, form_factors = saxs_setup
    
    # 1. Generate theoretical SAXS target from the 'native' compact structure
    target_iq = debye_saxs(small_protein_coords, q_values, form_factors)
    target_rg = float(compute_rg(small_protein_coords))
    
    # 2. Perturb structure (expand it outward from center)
    center = jnp.mean(small_protein_coords, axis=0)
    expanded_coords = center + (small_protein_coords - center) * 1.5
    expanded_rg = float(compute_rg(expanded_coords))
    assert expanded_rg > target_rg + 2.0, "Setup failed: Structure not expanded."

    # 3. Create SAXS Loss
    saxs_loss = SAXSLoss(
        q_values=q_values,
        exp_intensities=target_iq,
        form_factors=form_factors,
        scale_mode="lsq"
    )

    # 4. We want to restrict the geometry using distance constraints for consecutive C-alphas
    # Since we don't have full bonds in this simple test, we just use a weak geometry loss
    # anchoring to the initial expanded structure to prevent atoms from flying away
    geom_loss = GeometryLoss(target_coords=expanded_coords, target_weight=1.0)
    
    joint_loss = JointLoss([
        (saxs_loss, 1.0),
        (geom_loss, 0.01) # Very weak geometric anchor
    ])

    refiner = IntegrativeRefiner(loss_fn=joint_loss)
    
    # Run optimization (Cartesian parameter space)
    final_coords, history = refiner.run(
        init_params=expanded_coords,
        epochs=150,
        learning_rate=0.05
    )

    final_rg = float(compute_rg(final_coords))
    
    # Assert SAXS successfully compressed the structure back towards target Rg
    assert final_rg < expanded_rg, "SAXS refinement failed to compact the structure."
    
    # Check that SAXS chi-sq/MSE descended
    final_saxs_mse = float(saxs_loss((), final_coords))
    init_saxs_mse = float(saxs_loss((), expanded_coords))
    assert final_saxs_mse < init_saxs_mse, "SAXS loss failed to descend."
