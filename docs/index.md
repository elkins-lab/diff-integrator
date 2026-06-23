# ⚙️ Diff-Integrator

[![Tests](https://github.com/elkins-lab/diff-integrator/actions/workflows/test.yml/badge.svg)](https://github.com/elkins-lab/diff-integrator/actions/workflows/test.yml)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/diff-integrator.svg)](https://pypi.org/project/diff-integrator/)

Welcome to **Diff-Integrator**, a high-performance JAX optimization engine designed for integrative structural biology. 

It acts as the "orchestrator" that combines differentiable observables from **diff-biophys** into multi-objective loss functions.

By cleanly separating the optimization loop from the underlying biophysical kernels, `diff-integrator` enables robust, joint refinement of protein structures against diverse experimental data (e.g., SAXS, NMR Chemical Shifts, NMR RDCs) simultaneously.

## Why Diff-Integrator?

While `diff-biophys` provides the forward models (Structure -> Observable) and gradients, `diff-integrator` provides the **inverse modeling** engine.

1. **Multi-Objective Optimization**: Easily weight and combine multiple experimental constraints via `JointLoss`.
2. **Abstract Parameterization**: Optimize arbitrary parameter spaces—from Cartesian coordinates to internal backbone angles (phi/psi)—via user-defined `kinematics_fn` mappers.
3. **Dynamic Fitting**: Analytically refit nuisance parameters (like Saupe alignment tensors or SAXS scaling factors) dynamically during gradient descent.

## Getting Started

Check out the **[API Reference](api/geometry.md)** or dive into the **[Interactive Tutorials](https://colab.research.google.com/github/elkins-lab/diff-integrator/blob/main/examples/interactive_tutorials/01_hello_gradient_descent.ipynb)**!
