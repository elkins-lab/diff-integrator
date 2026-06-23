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
