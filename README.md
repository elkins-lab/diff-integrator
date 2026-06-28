# ⚙️ Diff-Integrator: The Integrative Refinement Engine

[![Tests](https://github.com/elkins-lab/diff-integrator/actions/workflows/test.yml/badge.svg)](https://github.com/elkins-lab/diff-integrator/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/elkins-lab/diff-integrator/branch/main/graph/badge.svg)](https://codecov.io/gh/elkins-lab/diff-integrator)
[![PyPI version](https://img.shields.io/pypi/v/diff-integrator.svg)](https://pypi.org/project/diff-integrator/)
[![Python versions](https://img.shields.io/pypi/pyversions/diff-integrator.svg?cache=bust)](https://pypi.org/project/diff-integrator/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![JAX](https://img.shields.io/badge/framework-JAX-9cf.svg?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyQzYuNDggMiAyIDYuNDggMiAxMnM0LjQ4IDEwIDEwIDEwIDEwLTQuNDggMTAtMTBTMTcuNTIgMiAxMiAyeiIvPjwvc3ZnPg==)](https://github.com/google/jax)
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
| [**📉 Results Dashboard**](examples/interactive_tutorials/visualize_benchmarks.ipynb) | Graduate / researcher | Visualizes the loss descent, Q-factors, chemical shift accuracy, NeRF drift, and a Cartesian vs. NeRF comparison across all four benchmarks (2KZV, GmR58A, HR2876B NeRF, HR2876B Cartesian). | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins-lab/diff-integrator/blob/main/examples/interactive_tutorials/visualize_benchmarks.ipynb) |
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
*   **`FixedTensorRDCLoss`**: Fixed-tensor RDC loss that holds the Saupe alignment tensor frozen during backpropagation (using `jax.lax.stop_gradient`) and re-fits it every `update_interval` epochs. Includes `cv_fraction` cross-validation split and `suggested_weight()` auto-scaling by overdetermination ratio.
*   **`CAShiftLoss`**: Wraps the ring-current and secondary structure shift predictor to compute $C_\alpha$ chemical shift RMSDs from backbone torsion angles.
*   **`CartesianCAShiftLoss`**: Cartesian-space variant that extracts φ/ψ on-the-fly from raw coordinates via `compute_phi_psi`, enabling chemical shift refinement without a NeRF builder.
*   **`BondLengthPenalty` / `BondAnglePenalty`**: Harmonic restraints on backbone bond lengths and angles to Engh & Huber ideal values. Used in Cartesian refinement to replace the hard geometric constraints of the NeRF builder.
*   **`RamachandranLoss`**: Sequence-aware Ramachandran prior with residue-specific Gaussian wells. Handles GLY ε-basin (φ > 0) and PRO ring constraint correctly.

---

## 🚀 Usage Example

```python
from diff_integrator.loss import JointLoss
from diff_integrator.optimizer import EarlyStopping, IntegrativeRefiner
from diff_integrator.schedules import ExponentialDecaySchedule
from diff_integrator.terms.geometry import GeometryLoss
from diff_integrator.terms.nmr import FixedTensorRDCLoss, make_rdc_cv_refinement_fns

# 1. Build loss terms
geom_term = GeometryLoss(target_coords=starting_coords)

# FixedTensorRDCLoss holds the Saupe tensor fixed during backprop,
# preventing the degeneracy exploit that drives Q→0 unphysically.
loss_fn, q_eval, tensor_fn, val_q_fn, n_train, n_val = make_rdc_cv_refinement_fns(
    rdc_res_ids, exp_rdcs, struct_res_ids, cv_fraction=0.2
)
rdc_term = FixedTensorRDCLoss(
    loss_fn, tensor_fn, update_interval=50,
    n_rdcs=n_train, val_q_eval_fn=val_q_fn
)
# Auto-scale weight by overdetermination ratio (ideal = 10×)
rdc_weight = rdc_term.suggested_weight(base_weight=1.0)

# 2. Combine into a joint loss
joint_loss = JointLoss([
    (geom_term, 5.0),
    (rdc_term, rdc_weight),
])

# 3. Annealed geometry weight: strong early, relaxed late
anchor_schedule = ExponentialDecaySchedule(
    initial_weight=10.0, final_weight=0.1, decay_epochs=100
)

# 4. Refine — result is a RefinementResult dataclass
refiner = IntegrativeRefiner(loss_fn=joint_loss)
result = refiner.run(
    init_params=starting_coords,
    epochs=2000,
    learning_rate=0.005,
    weight_schedules={0: anchor_schedule},      # anneal geometry anchor
    early_stopping=EarlyStopping(              # stop when RDC Q plateaus
        term_index=1, patience=50, min_delta=1e-4
    ),
)
print(f"Best checkpoint: epoch {result.best_epoch}")
print(f"Stopped early:   {result.stopped_early} ({result.early_stopping_triggered_by})")
refined_coords = result.best_params
```

---

## 🔬 Scientific Validation

`diff-integrator` is being validated against several experimental NMR datasets:
- **2KZV (CvR118A)**: Joint refinement using $C_\alpha$ Chemical Shifts and dual-medium (PAG/PEG) RDCs, lowering the $C_\alpha$ RMSD and bringing RDC Q-factors near zero.
- **GmR58A & HR2876B**: Successful gradient-based minimization of $C_\alpha$ shift RMSD using internal coordinates (dihedrals).
- **HR2876B Cartesian (2025)**: Cartesian + bond-geometry refinement achieved **13× larger** Cα RMSD improvement (−0.145 ppm vs −0.011 ppm) and **30× less** structural drift (0.24 Å vs 6.4 Å) compared to the NeRF approach, with per-term `EarlyStopping` stopping at epoch 894/2000 (55% compute saved).

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
