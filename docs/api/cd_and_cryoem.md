# 💡 CD & Cryo-EM API

---

## Circular Dichroism — CD Matrix Method

`diff_biophys.cd.kernels` implements the **DeVoe coupled-oscillator (matrix) model**
for simulating circular dichroism spectra from atomic positions.

Each amide bond in the backbone acts as a **chromophore** with a transition dipole
moment $\boldsymbol{\mu}_i$.  When two dipoles interact, their coupling splits the
transition into symmetric and antisymmetric combinations with different energies.
The **rotational strength** of each coupled transition determines the sign and
magnitude of the CD signal at each wavelength.

The kernel computes:

$$\Delta\varepsilon(\lambda) = \varepsilon_L(\lambda) - \varepsilon_R(\lambda)$$

as a function of chromophore positions and transition dipole orientations.

**Gradient meaning:** $\partial [\theta](\lambda) / \partial \mathbf{r}_i$ identifies
which chromophore, if moved, most changes the CD signal at wavelength $\lambda$.
At 222 nm (the canonical helix marker), moving chromophores that have large coupling
with their helical neighbours will have the largest gradient.

```python
from diff_biophys.cd.kernels import simulate_cd_matrix
import jax, jax.numpy as jnp

# chromophore_positions:    (N, 3)  amide nitrogen positions (Å)
# dipole_orientations:      (N, 3)  unit transition-dipole vectors
# wavelengths:              (M,)    wavelengths in nm

wavelengths = jnp.linspace(180.0, 260.0, 81)

cd_spectrum = simulate_cd_matrix(
    chromophore_positions,
    dipole_orientations,
    wavelengths,
    f_osc=0.2,       # oscillator strength
    gamma=10.0,      # linewidth (nm)
    lambda_0=190.0,  # transition wavelength (nm)
)
# cd_spectrum: (M,) molar ellipticity [deg cm² dmol⁻¹]

# Gradient of [θ] at 222 nm w.r.t. chromophore positions
idx_222 = jnp.argmin(jnp.abs(wavelengths - 222.0))
grad_222 = jax.grad(
    lambda pos: simulate_cd_matrix(pos, dipole_orientations, wavelengths)[idx_222]
)(chromophore_positions)
# grad_222: (N, 3) — largest magnitude → most influential chromophore
```

### Typical α-helix signature

| Wavelength | Sign | Assignment |
|---|---|---|
| 222 nm | negative | $n \to \pi^*$ parallel |
| 208 nm | negative | $\pi \to \pi^*$ perpendicular |
| 193 nm | positive | $\pi \to \pi^*$ parallel |

!!! note "Building chromophore coordinates"
    See [**Notebook 03 · CD Spectroscopy**](../examples/refinement.md) for a complete
    example of building a helix from scratch and computing its CD spectrum.

::: diff_biophys.cd.kernels

---

## Cryo-EM — Fourier Shell Correlation

`diff_biophys.cryo_em` implements the **Fourier Shell Correlation (FSC)**,
the standard figure-of-merit for cryo-EM reconstruction quality.

The FSC measures the normalised cross-correlation between two independently
reconstructed half-maps as a function of spatial frequency:

$$\text{FSC}(\nu) = \frac{\sum_{\mathbf{k} \in \text{shell}} F_1(\mathbf{k})\, F_2^*(\mathbf{k})}{\sqrt{\sum |F_1|^2 \cdot \sum |F_2|^2}}$$

The **gold-standard 0.143 threshold** gives the resolution at which the
two half-maps are no longer correlated — i.e., the spatial frequency up
to which the reconstruction is reliable.

**Gradient meaning:** $\partial \text{FSC}(\nu) / \partial \text{map}_1$ shows
which voxels, if improved, would most increase the correlation at frequency $\nu$.
This can drive iterative map sharpening or density modification.

```python
from diff_biophys.cryo_em import compute_fsc
import jax, jax.numpy as jnp

# map1, map2: (D, H, W) float32 real-space density maps
# voxel_size: (dz, dy, dx) in Å

frequencies, fsc_curve = compute_fsc(
    map1, map2,
    voxel_size=(1.0, 1.0, 1.0)
)
# frequencies: (n_shells,) in Å⁻¹
# fsc_curve:   (n_shells,) values in [−1, 1]

# Resolution at 0.143 threshold
resolution_mask = fsc_curve > 0.143
resolution_Å = 1.0 / float(frequencies[resolution_mask][-1])

# Gradient w.r.t. first half-map
def fsc_sum(m1):
    _, fsc = compute_fsc(m1, map2, voxel_size=(1.0, 1.0, 1.0))
    return jnp.sum(fsc)

grad_map1 = jax.grad(fsc_sum)(map1)
```

::: diff_biophys.cryo_em
