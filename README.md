# ⚙️ Diff-Integrator: The Integrative Refinement Engine

[![Tests](https://github.com/elkins-lab/diff-integrator/actions/workflows/test.yml/badge.svg)](https://github.com/elkins-lab/diff-integrator/actions/workflows/test.yml)
[![PyPI version](https://img.shields.io/pypi/v/diff-integrator.svg)](https://pypi.org/project/diff-integrator/)
[![Python versions](https://img.shields.io/pypi/pyversions/diff-integrator.svg?cache=bust)](https://pypi.org/project/diff-integrator/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy-lang.org/)

**Diff-Integrator** is a JAX-accelerated optimization engine designed for integrative structural biology. It acts as the "orchestrator" that combines differentiable observables from **[diff-biophys](https://github.com/elkins-lab/diff-biophys)** into multi-objective loss functions.

By cleanly separating the optimization loop from the underlying biophysical kernels, `diff-integrator` enables robust, joint refinement of protein structures against diverse experimental data (e.g., SAXS, NMR Chemical Shifts, NMR RDCs) simultaneously.

---

## 🎯 Vision

The goal of `diff-integrator` is to provide a seamless **optax**-based refinement pipeline that handles:
1. **Multi-Objective Optimization**: Easily weight and combine multiple experimental constraints via `JointLoss`.
2. **Abstract Parameterization**: Optimize arbitrary parameter spaces—from Cartesian coordinates to internal backbone angles (phi/psi)—via user-defined `kinematics_fn` mappers.
3. **Dynamic Fitting**: Analytically refit nuisance parameters (like Saupe alignment tensors or SAXS scaling factors) dynamically during gradient descent.

---

## 📚 Interactive Tutorials

Experience **Diff-Integrator** directly in your browser with our Colab tutorials:

| Tutorial | Audience | Description | Action |
| :--- | :--- | :--- | :--- |
| [**📉 Results Dashboard**](examples/interactive_tutorials/visualize_benchmarks.ipynb) | Graduate / researcher | Visualizes the loss descent, Q-factors, chemical shift accuracy, NeRF drift, and structural changes across all three benchmarks. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins-lab/diff-integrator/blob/main/examples/interactive_tutorials/visualize_benchmarks.ipynb) |
| [**🧪 Refinement Concepts**](examples/interactive_tutorials/concepts.ipynb) | Student / researcher | Educational notebook explaining NMR observables, NeRF coordinate parameterization, RDC tensor degeneracy, and the fixed-tensor protocol. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins-lab/diff-integrator/blob/main/examples/interactive_tutorials/concepts.ipynb) |
| [**⚠️ Method Limitations**](examples/interactive_tutorials/limitations.ipynb) | Reviewer / scientist | Honest quantitative assessment of the current method's failure modes: NeRF geometric drift, RDC overfitting on PEG data, and degrees-of-freedom imbalance. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins-lab/diff-integrator/blob/main/examples/interactive_tutorials/limitations.ipynb) |

---

## ⚡ Core Components

### `IntegrativeRefiner`
The core optimization engine. Built on `optax` (defaulting to the Adam optimizer), it manages the training loop, gradient calculation, and loss tracking.
*   **Abstract Support**: Accepts arbitrary parameter sets (`init_params`) and maps them to Cartesian space using an optional `kinematics_fn`.

### `JointLoss`
A container for combining multiple `LossTerm` objects. It computes the total weighted loss by evaluating each term on the current parameters and generated coordinates.

### `LossTerm` (Interface)
An abstract base class for defining differentiable constraints. All terms implement `__call__(self, params, coords)`.

### Included Observables
*   **`GeometryLoss`**: Implements basic structural priors, including harmonic restraints to a target Cartesian structure.
*   **`SAXSLoss`**: Dynamically scales and fits theoretical SAXS profiles against experimental data using Debye kernels.
*   **`RDCLoss`**: Analytically fits the Saupe tensor to current bond vectors at each step to compute Residual Dipolar Coupling (RDC) Mean Squared Errors or Q-factors.
*   **`CAShiftLoss`**: Wraps the ring-current and secondary structure shift predictor to compute $C_\alpha$ chemical shift RMSDs.

---

## 🚀 Usage Example

```python
import jax.numpy as jnp
from diff_integrator.loss import JointLoss
from diff_integrator.optimizer import IntegrativeRefiner
from diff_integrator.terms.geometry import GeometryLoss
from diff_integrator.terms.nmr import RDCLoss

# 1. Define loss terms
target_coords = jnp.load("target.npy")
geom_term = GeometryLoss(target_coords, loss_type="rmsd")

rdc_term = RDCLoss(
    atom_pairs=bond_indices,
    exp_rdcs=experimental_rdc_values,
    d_max=21700.0,
    loss_type="q_factor"
)

# 2. Combine into a joint loss
joint_loss = JointLoss([
    (geom_term, 10.0),  # Weight = 10.0
    (rdc_term, 1.0)     # Weight = 1.0
])

# 3. Refine
refiner = IntegrativeRefiner(loss_fn=joint_loss)
final_coords, history = refiner.run(
    init_params=starting_coords,
    epochs=1000,
    learning_rate=0.01
)
```

---

## 🔬 Scientific Validation

`diff-integrator` is being validated against several experimental NMR datasets:
- **2KZV (CvR118A)**: Joint refinement using $C_\alpha$ Chemical Shifts and dual-medium (PAG/PEG) RDCs, lowering the $C_\alpha$ RMSD and bringing RDC Q-factors near zero.
- **GmR58A & HR2876B**: Successful gradient-based minimization of $C_\alpha$ shift RMSD using internal coordinates (dihedrals).

---

## 📂 Repository Structure

```text
diff-integrator/
├── diff_integrator/       # Core package
│   ├── loss.py            # JointLoss and LossTerm interface
│   ├── optimizer.py       # IntegrativeRefiner engine
│   └── terms/             # Concrete loss implementations
│       ├── geometry.py    # Harmonic restraints, RMSD
│       ├── saxs.py        # Debye scattering loss
│       ├── nmr.py         # RDC and Q-factor loss
│       └── chemical_shifts.py # C-alpha shift loss
├── benchmarks/            # Real-world optimization tests
├── tests/                 # Unit tests (100% coverage)
├── docs/                  # MkDocs documentation
└── pyproject.toml         # Build configuration
```

## 🚀 Installation

Ensure you have JAX installed, then install `diff-integrator` locally:

```bash
pip install -e .
```

## 🤝 Contributing

Contributions are welcome! Please run the test suite and ensure `mypy` typing passes before submitting PRs:
```bash
pytest --cov=diff_integrator
mypy .
```

## ⚖️ License

MIT License — see [LICENSE](LICENSE) for details.
