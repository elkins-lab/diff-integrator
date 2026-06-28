# 📐 Geometry API

The `diff_biophys.geometry` subpackage provides differentiable structural primitives —
the mathematical backbone for converting between internal coordinates (bond lengths,
bond angles, dihedral angles) and Cartesian 3D positions, plus tools for alignment
and macroscopic property calculation.

All functions are compiled with `jax.jit` and are fully compatible with `jax.grad`,
`jax.vmap`, and `jax.pmap`.

---

## NeRF — Forward Kinematics

`diff_biophys.geometry.nerf` implements the **Natural Extension Reference Frame**
algorithm.  Given three reference atoms and a set of internal coordinates, it places
a new atom in 3D space.  The recurrence can be chained to build an entire polymer
backbone from scratch — entirely differentiably.

**When to use:**

- Convert torsion-angle parameters (φ, ψ, ω) to Cartesian coordinates
- Build synthetic structures for gradient-descent refinement
- Provide a differentiable mapping from parameter space to observable space

```python
from diff_biophys.geometry.nerf import position_atom_3d, chain_nerf
import jax.numpy as jnp

# Place one atom given three reference atoms + internal coordinates
p1 = jnp.array([0.0, 0.0, 0.0])
p2 = jnp.array([1.52, 0.0, 0.0])
p3 = jnp.array([1.52 + 1.52 * jnp.cos(jnp.pi - 1.94), 1.52 * jnp.sin(jnp.pi - 1.94), 0.0])

p4 = position_atom_3d(
    p1, p2, p3,
    bond_length=jnp.array(1.33),      # Å  (Cα–C)
    bond_angle=jnp.array(1.94),       # rad
    dihedral=jnp.array(-0.994),       # rad  (helix ψ)
)
print(p4)  # → (3,) array with Cartesian coordinates
```

::: diff_biophys.geometry.nerf

---

## Torsions — Internal Coordinate Extraction

`diff_biophys.geometry.torsions` extracts bond lengths, bond angles, and dihedral
(torsion) angles from a Cartesian coordinate array.  Together with NeRF this forms a
**round-trip**: internal → Cartesian → internal.

```python
from diff_biophys.geometry.torsions import compute_dihedrals, compute_bond_lengths, compute_bond_angles

# coords: (N, 3) backbone atom positions
dihedrals   = compute_dihedrals(coords)    # (N-3,)
bond_lengths = compute_bond_lengths(coords) # (N-1,)
bond_angles  = compute_bond_angles(coords)  # (N-2,)
```

::: diff_biophys.geometry.torsions

---

## Superposition — Kabsch Alignment

`diff_biophys.geometry.superposition` implements the **Kabsch algorithm** for
optimal RMSD superposition of two structures via SVD.  The rotation matrix and
translation vector are returned, allowing the aligned RMSD to be used as a
differentiable loss term.

```python
from diff_biophys.geometry.superposition import kabsch_alignment
import jax, jax.numpy as jnp

# P: mobile structure (N, 3),  Q: reference (N, 3)
R, t = kabsch_alignment(P, Q)
P_aligned = P @ R.T + t
rmsd = jnp.sqrt(jnp.mean(jnp.sum((P_aligned - Q) ** 2, axis=-1)))

# Gradient of RMSD w.r.t. mobile coordinates
grad = jax.grad(lambda p: jnp.sqrt(jnp.mean(jnp.sum(
    (kabsch_alignment(p, Q)[0] @ p.T).T - Q, axis=-1) ** 2)))(P)
```

::: diff_biophys.geometry.superposition

---

## Macroscopic Properties

`diff_biophys.geometry.macroscopic` computes bulk structural descriptors.

### Radius of Gyration

$$R_g^2 = \frac{\sum_i m_i \|\mathbf{r}_i - \mathbf{r}_{cm}\|^2}{\sum_i m_i}$$

Useful as a **compaction restraint**: minimise $(R_g - R_g^{target})^2$ to drive a
structure toward a target size extracted from the Guinier region of a SAXS profile.

```python
from diff_biophys.geometry.macroscopic import compute_rg
import jax, jax.numpy as jnp

coords = jnp.array(...)         # (N, 3)
masses = jnp.ones(len(coords))  # uniform masses

rg = compute_rg(coords, masses)

# Gradient: which atoms, if moved, most change Rg?
grad_rg = jax.grad(lambda c: compute_rg(c, masses))(coords)
```

::: diff_biophys.geometry.macroscopic

---

## Backbone Bond-Geometry Penalties

`diff_integrator.terms.geometry.BondLengthPenalty` and `diff_integrator.terms.geometry.BondAnglePenalty`

Harmonic restraints on backbone bond lengths and angles toward **Engh & Huber (1991)** ideal values. These are the Cartesian-mode analogues of the hard geometric constraints enforced implicitly by the NeRF builder, and should be included in any Cartesian refinement to prevent backbone distortion during gradient descent.

### `make_backbone_bond_geometry`

Factory function that constructs both penalty terms simultaneously from residue names:

```python
from diff_integrator.terms.geometry import make_backbone_bond_geometry

bond_penalty, angle_penalty = make_backbone_bond_geometry(residue_names)
```

| Parameter | Type | Description |
|---|---|---|
| `residue_names` | `Sequence[str]` | One-letter or three-letter residue codes in chain order |

Returns `(BondLengthPenalty, BondAnglePenalty)`.

### `BondLengthPenalty`

Harmonic penalty on the three backbone bond types per residue:

| Bond | Engh & Huber ideal (Å) |
|---|---|
| N–Cα | 1.458 |
| Cα–C | 1.525 |
| C–N | 1.329 |

**`__call__(params, coords)`** — `coords` must be `(3N, 3)` Cartesian backbone atom positions (N, Cα, C interleaved). Returns a scalar penalty.

Typical weight in `JointLoss`: **50.0** (stiff, geometry must be tight).

At convergence on HR2876B: bond RMSD **0.0007 Å**.

### `BondAnglePenalty`

Harmonic penalty on the three backbone angle types per residue:

| Angle | Engh & Huber ideal (°) |
|---|---|
| N–Cα–C | 111.2 |
| Cα–C–N | 116.2 |
| C–N–Cα | 121.7 |

**`__call__(params, coords)`** — same `(3N, 3)` Cartesian layout as `BondLengthPenalty`. Returns a scalar penalty.

Typical weight in `JointLoss`: **10.0** (softer than bond lengths).

At convergence on HR2876B: angle RMSD **0.33°**.

**Example**

```python
from diff_integrator.loss import JointLoss
from diff_integrator.terms.geometry import GeometryLoss, make_backbone_bond_geometry

bond_pen, angle_pen = make_backbone_bond_geometry(residue_names)
anchor = GeometryLoss(target_coords=starting_coords)

joint_loss = JointLoss([
    (anchor,    5.0),
    (bond_pen,  50.0),
    (angle_pen, 10.0),
])
```

---

## Cartesian Cα Chemical Shift Loss

### `CartesianCAShiftLoss`

`diff_integrator.terms.geometry.CartesianCAShiftLoss`

A `LossTerm` that predicts Cα chemical shifts directly from **Cartesian backbone coordinates** by extracting φ/ψ angles on-the-fly through the differentiable `compute_phi_psi` function. Use this instead of `CAShiftLoss` when operating in Cartesian mode (`kinematics_fn=None`).

**Constructor**

```python
CartesianCAShiftLoss(
    exp_res_ids: ArrayLike,
    exp_shifts: ArrayLike,
    struct_res_ids: ArrayLike,
    struct_res_names: Sequence[str],
)
```

| Parameter | Type | Description |
|---|---|---|
| `exp_res_ids` | `ArrayLike` | Residue IDs for each experimental shift value |
| `exp_shifts` | `ArrayLike` | Experimental Cα chemical shift values (ppm) |
| `struct_res_ids` | `ArrayLike` | Residue IDs present in the structure (for index alignment) |
| `struct_res_names` | `Sequence[str]` | Residue names in chain order (used for random-coil reference lookup) |

**`name`** attribute: `"ca_shift"`

**`__call__(params, coords)`** — `coords` are `(3N, 3)` Cartesian backbone atom positions. Internally calls `compute_phi_psi(coords)` to extract torsion angles, then feeds them to `predict_ca_shifts`. Returns MSE loss (ppm²) over matched residues.

**When to use vs `CAShiftLoss`**

| | `CAShiftLoss` | `CartesianCAShiftLoss` |
|---|---|---|
| Parameter space | Dihedrals (φ, ψ) | Cartesian (x, y, z) |
| Requires `kinematics_fn` | Yes (NeRF builder) | No |
| φ/ψ extraction | Direct (params *are* angles) | `compute_phi_psi` on-the-fly |
| Typical use | NeRF-mode refinement | Cartesian-mode refinement |

**Example**

```python
from diff_integrator.terms.geometry import CartesianCAShiftLoss

shift_term = CartesianCAShiftLoss(
    exp_res_ids=exp_res_ids,
    exp_shifts=exp_ca_shifts,
    struct_res_ids=struct_res_ids,
    struct_res_names=residue_names,
)
```
