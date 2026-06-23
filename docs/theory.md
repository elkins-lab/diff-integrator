# Theory Reference

This page develops the mathematical and physical foundations underlying every forward model
implemented in **diff-biophys**. Sections progress from physical intuition → governing
equations → differentiability and gradient interpretation. All math is written in the
convention used by the code; deviations from common textbook notation are flagged
explicitly.

> [!NOTE]
> Throughout this document, atom positions are three-dimensional column vectors
> $\mathbf{r}_i \in \mathbb{R}^3$. Bold lower-case letters are vectors; bold upper-case
> letters are matrices. Greek letters denote angles unless otherwise stated.

---

## 1. Natural Extension Reference Frame (NeRF)

### Physical Intuition

Protein structure can be specified in two complementary coordinate systems.
**Cartesian coordinates** place every atom at an absolute position $(x,y,z)$ in space —
convenient for computing distances and energies, but highly redundant: a rigid-body rotation
or translation of the whole chain changes every coordinate without changing the
*shape* of the molecule. **Internal coordinates** instead describe the chain geometry
through physically meaningful local quantities that are invariant to global rotation and
translation:

- **Bond length** $b_i$ — the distance between bonded atoms $i-1$ and $i$.
- **Bond angle** $\theta_i$ — the angle at atom $i-1$ subtended by the $i-2 \to i-1 \to i$
  triple (equivalently, the supplement of the angle between successive bond vectors).
- **Dihedral (torsion) angle** $\phi_i$ — the angle of rotation about the $i-2 \to i-1$
  bond axis, measured between the plane containing $i-3, i-2, i-1$ and the plane containing
  $i-2, i-1, i$. This is the quantity optimised during conformation search.

The **Natural Extension Reference Frame (NeRF)** algorithm, introduced by Parsons *et al.*
(2005), converts internal coordinates back to Cartesian space via a closed-form recurrence.
This is exactly what is needed for a differentiable structure generator: pass dihedral angles
through NeRF → get atom positions → evaluate any experimental observable.

### The NeRF Recurrence Formula

Given three already-placed atoms $\mathbf{r}_{i-3}$, $\mathbf{r}_{i-2}$, $\mathbf{r}_{i-1}$
and the internal coordinates $(b_i, \theta_i, \phi_i)$ for the new atom $i$, NeRF proceeds
as follows.

**Step 1 — Build the local right-handed frame.**

$$
\hat{\mathbf{d}} = \frac{\mathbf{r}_{i-1} - \mathbf{r}_{i-2}}{\|\mathbf{r}_{i-1} - \mathbf{r}_{i-2}\|}
$$

$$
\hat{\mathbf{n}} = \frac{(\mathbf{r}_{i-2} - \mathbf{r}_{i-3}) \times \hat{\mathbf{d}}}{\|(\mathbf{r}_{i-2} - \mathbf{r}_{i-3}) \times \hat{\mathbf{d}}\|}
$$

$$
\hat{\mathbf{m}} = \hat{\mathbf{n}} \times \hat{\mathbf{d}}
$$

The triad $\{\hat{\mathbf{d}}, \hat{\mathbf{n}}, \hat{\mathbf{m}}\}$ is an orthonormal basis
anchored at $\mathbf{r}_{i-1}$. The vector $\hat{\mathbf{d}}$ points along the most recent
bond direction; $\hat{\mathbf{n}}$ is the normal to the plane of the last three atoms;
$\hat{\mathbf{m}}$ completes the right-handed set.

**Step 2 — Express the new atom in local coordinates.**

In the local frame the new atom lies at bond length $b_i$ from $\mathbf{r}_{i-1}$, deflected
by the supplement of bond angle $\theta_i$ away from $\hat{\mathbf{d}}$, and rotated by
dihedral $\phi_i$ about $\hat{\mathbf{d}}$:

$$
\mathbf{r}_i^{\,\mathrm{local}} = b_i \begin{pmatrix} -\cos\theta_i \\ \sin\theta_i \cos\phi_i \\ \sin\theta_i \sin\phi_i \end{pmatrix}
$$

**Step 3 — Rotate back to global Cartesian space.**

$$
\boxed{\mathbf{r}_i = \mathbf{r}_{i-1} + b_i\bigl[-\cos\theta_i\,\hat{\mathbf{d}} + \sin\theta_i\cos\phi_i\,\hat{\mathbf{m}} + \sin\theta_i\sin\phi_i\,\hat{\mathbf{n}}\bigr]}
$$

This is the NeRF recurrence. Each atom depends only on its three predecessors and its own
internal coordinates. The chain is built atom-by-atom from N-terminus to C-terminus.

### Why NeRF is Differentiable

Every operation above — cross products, normalisation, trigonometric functions — is smooth
and composed of elementary arithmetic. JAX traces through the computation graph automatically
using forward or reverse-mode automatic differentiation. The critical subtlety is the
cross-product normalisation: it introduces a $1/\|\cdot\|$ factor that can be numerically
problematic when three atoms become nearly co-linear (i.e., the bond angle approaches
$0°$ or $180°$). In diff-biophys, bond angles are constrained away from these singularities
during optimisation.

> [!IMPORTANT]
> The gradient $\partial \mathbf{r}_i / \partial \phi_j$ is **non-zero for all** $i > j$
> because a rotation about bond $j$ rigidly moves every downstream atom. This long-range
> dependency is why torsion-space optimisation of proteins is non-trivial but is also why
> dihedral gradients encode rich structural information.

### Gradient Interpretation

The gradient of any downstream loss $\mathcal{L}$ with respect to a dihedral $\phi_j$
aggregates contributions from all atoms $i > j$:

$$
\frac{\partial \mathcal{L}}{\partial \phi_j} = \sum_{i>j} \frac{\partial \mathcal{L}}{\partial \mathbf{r}_i} \cdot \frac{\partial \mathbf{r}_i}{\partial \phi_j}
$$

A large $|\partial \mathcal{L}/\partial \phi_j|$ means that rotating about bond $j$ would
most efficiently reduce the experimental discrepancy — directly identifying the flexible
region the structure needs to adjust.

---

## 2. Small-Angle X-ray Scattering (SAXS)

### Physical Intuition

When a monochromatic X-ray beam of wavelength $\lambda$ illuminates a solution of identical
molecules, the electrons in each molecule scatter the photons. At small angles (typically
$2\theta < 5°$), the scattered waves from different parts of the *same* molecule interfere
constructively and destructively depending on the inter-atomic distances. The resulting
intensity profile $I(q)$ — recorded on a detector as a function of the **momentum transfer**
$q$ — encodes the overall shape and size of the molecule without requiring crystals.

$$
q = \frac{4\pi \sin\theta}{\lambda} \quad [\text{Å}^{-1}]
$$

Here $2\theta$ is the scattering angle. The quantity $2\pi/q$ is the length scale being
probed: small $q$ corresponds to large-scale (global) structure; large $q$ resolves finer
detail.

### The Debye Formula

For a molecule with $N$ atoms at positions $\{\mathbf{r}_i\}$, each with **atomic form
factor** $f_i(q)$ (the Fourier transform of its electron density), the solution-averaged
scattering intensity is exactly:

$$
\boxed{I(q) = \sum_{i=1}^{N}\sum_{j=1}^{N} f_i(q)\, f_j(q)\, \frac{\sin(q r_{ij})}{q r_{ij}}}
$$

where $r_{ij} = |\mathbf{r}_i - \mathbf{r}_j|$ is the inter-atomic distance. The
$\sin(x)/x$ (sinc) kernel arises from averaging the complex exponential $e^{i\mathbf{q}\cdot\mathbf{r}_{ij}}$
over all orientations of the molecule.

Atomic form factors are tabulated as sums of Gaussians (Waasmaier & Kirfel 1995):

$$
f(q) = \sum_{k=1}^{4} a_k \exp\!\left(-b_k \left(\frac{q}{4\pi}\right)^2\right) + c
$$

The form factor decreases monotonically with $q$, reflecting the finite spatial extent of
electron density around each nucleus.

> [!NOTE]
> The Debye formula has $O(N^2)$ complexity. For a 200-residue protein ($\approx$1600 atoms),
> this means ~2.6 million pair distances per $q$ point. diff-biophys uses JAX `vmap` and
> GPU acceleration to make this tractable. For very large assemblies, a histogram
> approximation is optionally substituted.

### Guinier Approximation and Radius of Gyration

At very small $q$ ($q R_g \ll 1$, typically $q < 1.3/R_g$), the intensity simplifies to
the **Guinier approximation**:

$$
\boxed{I(q) \approx I_0 \exp\!\left(-\frac{q^2 R_g^2}{3}\right)}
$$

where $I_0 = I(0)$ is the forward scattering intensity and $R_g$ is the **radius of
gyration** (Section 10). A plot of $\ln I(q)$ versus $q^2$ (the Guinier plot) is linear in
the valid range; the slope gives $-R_g^2/3$. This provides a rapid, model-independent
estimate of molecular size directly from raw data.

### Hydration Shell and Excluded-Volume Correction

Real SAXS measurements are performed in aqueous solution. The protein does two things to
the bulk solvent: it **excludes** water from its interior (reducing the effective scattering
density of that region) and it **orders** water molecules at its surface into a hydration
shell with higher electron density than bulk. The net scattering amplitude is:

$$
F(\mathbf{q}) = F_{\mathrm{protein}}(\mathbf{q}) - \rho_0\, F_{\mathrm{excluded}}(\mathbf{q}) + \delta\rho\, F_{\mathrm{shell}}(\mathbf{q})
$$

Following Fraser *et al.* (1978), the excluded volume is approximated by assigning each
atom a Gaussian-distributed excluded volume $v_i$ so that:

$$
F_{\mathrm{excluded}}(\mathbf{q}) = \sum_i v_i \exp\!\left(-\frac{v_i^{2/3} q^2}{4\pi}\right) e^{i\mathbf{q}\cdot\mathbf{r}_i}
$$

In practice, diff-biophys uses the two-parameter CRYSOL-style correction with amplitude $c_1$
(excluded volume scale) and $c_2$ (shell contrast $\delta\rho$), which are refined
alongside the structural model.

### The Kratky Plot

The **Kratky plot** displays $q^2 I(q)$ versus $q$. This transformation dramatically
amplifies the high-$q$ signal and serves as a rapid qualitative diagnostic:

| State | Kratky appearance |
|-------|------------------|
| Compact globular | Bell-shaped peak, returns toward zero |
| Partially unfolded | Broad, asymmetric peak |
| Intrinsically disordered | Monotonic rise (plateau or diverges) |

The physical basis is that a Gaussian chain gives $I(q) \propto q^{-2}$ at high $q$, so
$q^2 I(q)$ approaches a constant, while a compact sphere gives $I(q) \propto q^{-4}$, so
$q^2 I(q) \propto q^{-2} \to 0$.

### Pair-Distance Distribution P(r)

The real-space counterpart of $I(q)$ is the **pair-distance distribution function**:

$$
P(r) = \frac{r^2}{2\pi^2} \int_0^\infty I(q)\, q^2\, \frac{\sin(qr)}{qr}\, \mathrm{d}q
$$

$P(r)$ is the histogram of all inter-atomic distances weighted by the product of electron
densities at each pair. It is zero beyond the maximum inter-atomic distance $D_{\max}$ of the
molecule. The shape of $P(r)$ is diagnostic: a peak at small $r$ with a long tail indicates
an elongated molecule; a symmetric bell shape indicates a sphere.

> [!TIP]
> $R_g$ can be extracted directly from $P(r)$: $R_g^2 = \frac{\int r^2 P(r)\,\mathrm{d}r}{2\int P(r)\,\mathrm{d}r}$. This is more robust than the Guinier approximation when the low-$q$ region is noisy.

### Gradient Interpretation

$$
\frac{\partial I(q)}{\partial \mathbf{r}_k} = 2\sum_{j \neq k} f_k(q)\,f_j(q)\,\frac{\partial}{\partial \mathbf{r}_k}\left[\frac{\sin(q r_{kj})}{q r_{kj}}\right]
$$

$$
= 2\sum_{j \neq k} f_k f_j \cdot \frac{\cos(q r_{kj}) - \mathrm{sinc}(q r_{kj})}{r_{kj}^2} \cdot (\mathbf{r}_k - \mathbf{r}_j)
$$

The gradient $\partial I(q)/\partial \mathbf{r}_k$ is a 3-vector pointing in the direction
that atom $k$ should move to most increase $I(q)$ at that scattering angle. Atoms far apart
contribute most to low-$q$ gradients; nearby atoms drive high-$q$ gradients. A loss based
on the $\chi^2$ discrepancy between predicted and experimental $I(q)$ therefore pushes atoms
to rearrange in a physically interpretable, spatially resolved way.

---

## 3. NMR: Karplus J-Coupling

### Physical Intuition

**Scalar coupling** (J-coupling) is a through-bond interaction between nuclear spins,
transmitted by the electrons in intervening chemical bonds. The **three-bond** (vicinal)
coupling ${}^3J$ depends exquisitely on the dihedral angle $\theta$ spanning those three
bonds. This makes it one of the most direct NMR restraints on backbone and side-chain
geometry: measure a coupling constant in Hz, read off the torsion angle.

### The Karplus Equation

$$
\boxed{J(\theta) = A\cos^2\theta + B\cos\theta + C}
$$

This empirical relation was derived by Karplus (1959) from valence-bond theory. The parameters
$A$, $B$, $C$ depend on which three-bond pathway is being measured and on the electronegativity
of substituents. For the backbone ${}^3J_{\mathrm{H_N H_\alpha}}$ coupling, which is the most
frequently measured in protein NMR, **Vuister & Bax (1993)** calibrated:

$$
A = 6.98\,\text{Hz}, \quad B = -1.38\,\text{Hz}, \quad C = 1.72\,\text{Hz}
$$

using a database of proteins with known crystal structures.

### Connection to the Backbone Dihedral $\phi$

The ${}^3J_{\mathrm{H_N H_\alpha}}$ coupling is measured between the amide proton H$_\mathrm{N}$
and the alpha proton H$_\alpha$ of the *same* residue. The dihedral angle $\theta$ in the
Karplus equation is the H$_\mathrm{N}$–N–C$_\alpha$–H$_\alpha$ dihedral. In diff-biophys,
the backbone stores the conventional N–C$_\alpha$–C(O)–N dihedral $\phi$ (defined on heavy
atoms). The relationship is:

$$
\theta = \phi - 60°
$$

so that $J = A\cos^2(\phi - 60°) + B\cos(\phi - 60°) + C$.

> [!NOTE]
> The offset of $-60°$ arises from the fixed tetrahedral geometry at C$_\alpha$: the alpha
> proton is roughly $60°$ displaced from the carbonyl carbon in the staggered conformation.
> Getting this offset right is critical — a $60°$ error in $\phi$ maps to the wrong J value.

The Karplus curve is **not injective**: multiple $\phi$ values give the same $J$. This
four-fold ambiguity means J-couplings must be used in combination with other restraints
(NOEs, chemical shifts, RDCs).

### Gradient Interpretation

$$
\frac{\partial J}{\partial \phi} = -2A\cos(\phi-60°)\sin(\phi-60°) - B\sin(\phi-60°) = -\sin(\phi-60°)\bigl[2A\cos(\phi-60°) + B\bigr]
$$

The gradient is large when the Karplus curve has a steep slope — near $\phi \approx -120°$
(extended $\beta$-strand region) the curve is steep, so J-coupling restraints are strongly
informative. Near the extrema of the curve ($\phi \approx 0°$ or $\phi \approx 180°$) the
gradient vanishes: J-couplings provide little torsional information there, and other
restraints must compensate.

---

## 4. NMR: Chemical Shifts

### Physical Intuition

The **chemical shift** $\delta$ (in ppm) measures how far the resonance frequency of a
nucleus deviates from a standard reference compound. This deviation is caused by the local
electronic environment: nearby electrons shield (or deshield) the nucleus from the external
field. Because secondary structure organises the backbone into regular hydrogen-bond
geometry, helices and strands produce characteristic, reproducible deviations from the shifts
expected for an unstructured ("random coil") chain.

### Reference Values and Secondary Shifts

The **random-coil chemical shift** $\delta^{\mathrm{RC}}_{\alpha}$ for each amino acid type
is measured in short unstructured peptides. The experimental **secondary shift** is then:

$$
\Delta\delta = \delta^{\mathrm{obs}} - \delta^{\mathrm{RC}}
$$

For C$_\alpha$ nuclei (the most sensitive indicator), the empirical signatures are:

| Secondary structure | $\Delta\delta_{\mathrm{C}\alpha}$ |
|---------------------|----------------------------------|
| $\alpha$-helix | $\approx +3$ ppm |
| $\beta$-sheet | $\approx -1.5$ ppm |
| Random coil | $\approx 0$ ppm |

These offsets reflect systematic changes in the C$_\alpha$–C$_\beta$ and N–C$_\alpha$ bond
lengths and angles imposed by secondary-structure hydrogen bonding. Helical geometry
elongates the C$_\alpha$–C bond slightly, shifting C$_\alpha$ downfield.

### Softmax-Weighted Gaussian Detector in $(\phi, \psi)$ Space

Predicting C$_\alpha$ shifts from backbone dihedrals requires mapping Ramachandran space
$(\phi, \psi)$ onto a continuous scalar. diff-biophys implements a lightweight probabilistic
model:

**Step 1 — Define reference Gaussians.** For each secondary-structure class
$k \in \{\text{helix}, \text{sheet}, \text{coil}\}$, place a 2D Gaussian in $(\phi,\psi)$
space centred at its Ramachandran basin $(\mu_k^\phi, \mu_k^\psi)$ with covariance
$\Sigma_k$:

$$
\mathcal{G}_k(\phi,\psi) = \exp\!\left(-\tfrac{1}{2}\,\mathbf{x}^T\Sigma_k^{-1}\mathbf{x}\right), \quad \mathbf{x} = \begin{pmatrix}\phi - \mu_k^\phi \\ \psi - \mu_k^\psi\end{pmatrix}
$$

**Step 2 — Softmax weighting.** Convert the Gaussian scores to class probabilities:

$$
p_k(\phi,\psi) = \frac{\mathcal{G}_k(\phi,\psi)}{\sum_{k'} \mathcal{G}_{k'}(\phi,\psi)}
$$

**Step 3 — Weighted shift prediction.**

$$
\hat{\delta}_{C\alpha}(\phi,\psi) = \delta^{\mathrm{RC}} + \sum_k p_k(\phi,\psi)\,\Delta\delta_k
$$

This function is everywhere smooth and differentiable in $(\phi,\psi)$, making it directly
usable in gradient-based optimisation.

> [!TIP]
> The Ramachandran Gaussian centres used in diff-biophys are:
> helix $(-57°, -47°)$; sheet $(-119°, 113°)$; coil broadly distributed.
> Increasing the Gaussian widths $\Sigma_k$ softens the secondary-structure boundaries and
> produces smoother gradients at the expense of predictive sharpness.

### Gradient Interpretation

$$
\frac{\partial \hat{\delta}_{C\alpha}}{\partial \phi} = \sum_k \frac{\partial p_k}{\partial \phi}\,\Delta\delta_k
$$

The gradient points in $(\phi,\psi)$ space toward the secondary-structure basin that would
best account for the observed shift. If the observed C$_\alpha$ shift is $+2.5$ ppm (helix-
like) but the current $\phi$ is in the sheet region, the gradient will push $\phi$ toward
the helical basin — directly encoding secondary-structure propensity as a differentiable
force.

---

## 5. NMR: Residual Dipolar Couplings

### Physical Intuition

In free solution, molecular tumbling averages the direct (dipolar) coupling between two
nuclear spins to zero. When the molecule is **partially aligned** — for example, by dissolving
it in a dilute liquid crystal, bicelle medium, or by using a paramagnetic tag — the tumbling
becomes anisotropic. A small net orientation survives, and the dipolar coupling does not
fully average away. The **residual dipolar coupling** (RDC) $D$ that remains encodes the
orientation of each inter-nuclear vector relative to the molecular alignment frame.

### The Saupe Alignment Tensor

The partial alignment is described by the **Saupe order tensor** $\mathbf{S}$, a
$3\times3$ real symmetric traceless matrix:

$$
\mathbf{S} = \begin{pmatrix} S_{xx} & S_{xy} & S_{xz} \\ S_{xy} & S_{yy} & S_{yz} \\ S_{xz} & S_{yz} & S_{zz} \end{pmatrix}, \quad \mathrm{tr}(\mathbf{S}) = 0
$$

Because it is symmetric and traceless, $\mathbf{S}$ has five independent components. In its
**principal axis frame (PAF)**, it is diagonal:

$$
\mathbf{S}^{\mathrm{PAF}} = \begin{pmatrix} S_{xx}^P & 0 & 0 \\ 0 & S_{yy}^P & 0 \\ 0 & 0 & S_{zz}^P \end{pmatrix}, \quad S_{xx}^P + S_{yy}^P + S_{zz}^P = 0
$$

The largest eigenvalue defines the principal axis (the preferred molecular orientation); the
asymmetry parameter $\eta = (S_{xx}^P - S_{yy}^P)/S_{zz}^P$ quantifies the rhombicity of the
alignment.

### The Dipolar Coupling Formula

For a bond vector $\hat{\mathbf{v}}$ (unit vector from nucleus $i$ to nucleus $j$), the
predicted RDC is:

$$
\boxed{D_{ij} = D_{\max}\sum_{\alpha,\beta} v_\alpha\, S_{\alpha\beta}\, v_\beta = D_{\max}\,\hat{\mathbf{v}}^T \mathbf{S}\,\hat{\mathbf{v}}}
$$

where the prefactor $D_{\max}$ is the maximum possible dipolar coupling:

$$
D_{\max} = -\frac{\mu_0}{4\pi}\frac{\gamma_I \gamma_S \hbar}{r_{IS}^3}
$$

with $\gamma_I, \gamma_S$ the gyromagnetic ratios of the two nuclei and $r_{IS}$ the
internuclear distance (fixed for covalent bonds). For the backbone $^1\mathrm{H}$–$^{15}\mathrm{N}$
bond ($r_{NH} = 1.04$ Å), $D_{\max} \approx -21.7$ kHz.

### The Magic Angle

In the PAF with axial symmetry ($\eta = 0$, $S_{xx}^P = S_{yy}^P = -S_{zz}^P/2$):

$$
D \propto \frac{3\cos^2\theta - 1}{2}
$$

where $\theta$ is the angle between the bond vector and the principal axis $\hat{z}$. This
function equals zero at:

$$
\theta^* = \arccos\!\left(\frac{1}{\sqrt{3}}\right) \approx 54.74°
$$

This is the **magic angle**: any bond at this angle produces zero RDC regardless of the
alignment magnitude. Conversely, a bond parallel to the principal axis ($\theta = 0$) yields
the maximum coupling $D_{\max}\,S_{zz}$.

> [!NOTE]
> The magic angle at $\approx 54.74°$ also appears in magic-angle spinning (MAS) solid-state
> NMR, where mechanical rotation at this angle about the field averages dipolar couplings to
> zero — the same underlying geometry.

### SVD-Based Tensor Fitting

Given a set of experimental RDCs $\{D_{ij}^{\mathrm{exp}}\}$ and known bond vector orientations
$\{\hat{\mathbf{v}}_{ij}\}$, the five independent components of $\mathbf{S}$ can be
determined by linear least squares. Define for each bond the 5-element basis vector:

$$
\mathbf{A}_{ij} = D_{\max}\!\begin{pmatrix} v_x^2 - v_z^2 \\ v_y^2 - v_z^2 \\ 2v_x v_y \\ 2v_x v_z \\ 2v_y v_z \end{pmatrix}
$$

Stack into a matrix $\mathbf{A}$ and solve $\mathbf{A}\,\mathbf{s} = \mathbf{D}^{\mathrm{exp}}$
via **singular value decomposition (SVD)**. The SVD is differentiable in JAX via
`jax.numpy.linalg.svd`, enabling end-to-end gradient flow through tensor fitting into the
structural coordinates.

### Gradient Interpretation

$$
\frac{\partial D_{ij}}{\partial \mathbf{r}_i} = D_{\max}\,\frac{\partial}{\partial \mathbf{r}_i}\bigl[\hat{\mathbf{v}}^T\mathbf{S}\hat{\mathbf{v}}\bigr] = \frac{2 D_{\max}}{r_{ij}}\left(\mathbf{S}\hat{\mathbf{v}} - (\hat{\mathbf{v}}^T\mathbf{S}\hat{\mathbf{v}})\hat{\mathbf{v}}\right)
$$

This gradient is zero when $\hat{\mathbf{v}}$ is an eigenvector of $\mathbf{S}$ — the bond
is already aligned with a principal axis of the tensor. Otherwise, the gradient tilts the
bond toward the axis that reduces the RDC residual, providing orientational restraining
forces on every inter-nuclear vector simultaneously.

---

## 6. NMR: Ring Current Shifts

### Physical Intuition

Aromatic rings (Phe, Tyr, Trp, His) support a **ring current**: the delocalized $\pi$
electrons circulate in the plane of the ring when placed in a magnetic field, generating a
secondary magnetic field. This secondary field has a distinctive spatial pattern — it
**reinforces** the external field in the plane of the ring (deshielding nearby nuclei) and
**opposes** it along the ring axis (shielding nuclei above and below). Protons positioned
above the face of a benzene ring therefore resonate at unusually high field (low $\delta$).

### Johnson-Bovey Model

The classic model (Johnson & Bovey 1958) treats each aromatic ring as a circular current
loop. The additional shielding $\Delta\sigma$ at a point $P$ displaced by vector $\mathbf{r}$
from the ring centre is:

$$
\Delta\sigma = \frac{i\, e^2}{6mc^2 a}\!\left[\frac{1}{k^2}\!\left(\frac{2-k^2}{k'}\,K(k) - \frac{1}{k'}\,E(k)\right) - 1\right] \cdot \frac{a}{R}
$$

where $K(k)$ and $E(k)$ are complete elliptic integrals, $a$ is the ring radius, $R$ is the
distance from $P$ to the ring plane, $k^2 = 4aR/[(a+R)^2 + z^2]$, and $i$ is the ring-current
intensity. For practical use in diff-biophys, this is simplified to the **Haigh-Mallion**
approximation:

$$
\boxed{\Delta\delta = i_{\mathrm{ring}}\sum_{\mathrm{triangles}} B_n \frac{3\cos^2\xi_n - 1}{r_n^3}}
$$

where the sum is over triangles tessellating the ring, $r_n$ is the distance from the proton
to the centroid of triangle $n$, and $\xi_n$ is the angle between the proton vector and the
ring normal. The $r_n^{-3}$ dependence is the hallmark of a magnetic dipole field.

> [!NOTE]
> The ring-current shielding cone is commonly depicted as an hourglass (negative $\Delta\delta$
> above and below the ring) and a torus (positive $\Delta\delta$ in the ring plane). This
> spatial pattern is extremely useful in NMR structure determination: an upfield-shifted
> proton ($\Delta\delta < 0$) almost certainly sits above an aromatic ring, constraining the
> relative geometry.

### Distance Dependence and Gradient

The $1/r^3$ dependence makes ring-current shifts **very sensitive to short-range geometry**:
doubling the distance reduces the effect eightfold. The gradient:

$$
\frac{\partial \Delta\delta}{\partial \mathbf{r}_H} \propto \frac{1}{r^4}
$$

is therefore strongly peaked at close approach. In structure refinement, ring-current shift
restraints act as a precise positional anchor for protons near aromatic residues, often
resolving ambiguities that distance restraints alone cannot.

---

## 7. Circular Dichroism (CD)

### Physical Intuition

**Circular dichroism** measures the differential absorption of left- and right-handed
circularly polarised light by a chiral molecule:

$$
\Delta A(\lambda) = A_L(\lambda) - A_R(\lambda)
$$

Proteins absorb UV light primarily through **amide chromophores** (the peptide bond C=O and
N–H groups, absorbing near 190–230 nm). Because each amide has a fixed orientation along
the backbone, secondary structure places the amide transition dipoles in regular geometric
arrangements. These arrangements determine how the electric and magnetic transition moments
of adjacent amides couple — and that coupling determines the CD spectrum shape.

### Transition Dipole Moments and Amide Transitions

Each peptide amide has two main electronic transitions relevant to far-UV CD:

| Transition | Wavelength | Character |
|------------|------------|-----------|
| $n \to \pi^*$ | ~220 nm | Weak, $\mathbf{m}$ along C=O axis |
| $\pi \to \pi^*$ | ~190 nm | Strong, $\boldsymbol{\mu}$ in amide plane |

The $\alpha$-helical spectrum shows a characteristic **double minimum at 208 nm and 222 nm**
from the splitting of the $\pi\to\pi^*$ band by exciton coupling between neighbouring
amides, plus the $n\to\pi^*$ band. $\beta$-sheets show a single minimum near 216 nm (from
their antiparallel dipole arrangement) and a positive band near 195 nm.

### DeVoe Coupled-Oscillator / Matrix Method

For a protein with $N$ amide chromophores, the CD is computed via the **matrix method**
(Bayley *et al.*, Tinoco *et al.*). Each chromophore $i$ has a transition frequency
$\nu_i$, electric transition dipole moment $\boldsymbol{\mu}_i$, and magnetic transition
dipole moment $\mathbf{m}_i$. Off-diagonal coupling elements $V_{ij}$ between chromophores
$i$ and $j$ are computed from the dipole-dipole interaction:

$$
V_{ij} = \frac{1}{r_{ij}^3}\left[\boldsymbol{\mu}_i \cdot \boldsymbol{\mu}_j - 3(\boldsymbol{\mu}_i\cdot\hat{\mathbf{r}}_{ij})(\boldsymbol{\mu}_j\cdot\hat{\mathbf{r}}_{ij})\right]
$$

The full interaction Hamiltonian is diagonalised:

$$
\mathbf{H}\,\mathbf{c}_k = E_k\,\mathbf{c}_k
$$

The $k$-th normal mode has **rotational strength** $R_k$:

$$
R_k = \mathrm{Im}\!\left(\boldsymbol{\mu}_k \cdot \mathbf{m}_k\right), \quad \boldsymbol{\mu}_k = \sum_i c_{ki}\,\boldsymbol{\mu}_i, \quad \mathbf{m}_k = \sum_i c_{ki}\,\mathbf{m}_i
$$

The CD spectrum is then:

$$
\Delta\varepsilon(\lambda) = \sum_k R_k\, L(\lambda - \lambda_k)
$$

where $L$ is a Gaussian line shape centred at wavelength $\lambda_k$.

> [!IMPORTANT]
> The rotational strength $R_k$ is non-zero only if the electric and magnetic transition
> dipoles of the $k$-th mode have a non-zero dot product — requiring chirality. An achiral
> molecule has $\sum_k R_k = 0$ for each band (the Kuhn sum rule). This is why CD is
> uniquely sensitive to the handedness of secondary structure.

### Why Helices and Sheets Differ

In a right-handed $\alpha$-helix, adjacent amide C=O dipoles point roughly parallel and
spiral around the helix axis. The exciton coupling splits the $\pi\to\pi^*$ band into a
positive component near 193 nm (the parallel exciton, $A$ component) and a negative component
near 208 nm (the perpendicular exciton). The $n\to\pi^*$ band at 222 nm remains negative.
The net result: helices are identified by a large negative double minimum at 208 and 222 nm
and a strong positive band at 193 nm.

In antiparallel $\beta$-sheets, neighbouring strand amides are antiparallel. The coupling is
much weaker and produces a single negative minimum near 216 nm and a positive band near
195–198 nm — a distinctly different signature.

### Gradient Interpretation

The CD depends on atomic positions through the inter-chromophore distances $r_{ij}$ and
the orientations of transition dipole moments $\boldsymbol{\mu}_i$:

$$
\frac{\partial \Delta\varepsilon}{\partial \mathbf{r}_k} = \sum_\ell \frac{\partial \Delta\varepsilon}{\partial V_\ell} \cdot \frac{\partial V_\ell}{\partial \mathbf{r}_k}
$$

The gradient tells you which backbone atom, if moved, would most change the predicted CD
spectrum at a given wavelength — powerful for refining secondary-structure content against
experimental CD data.

---

## 8. Cryo-EM: Fourier Shell Correlation (FSC)

### Physical Intuition

In single-particle cryo-EM, tens of thousands of particle images are collected and classified
into viewing orientations. The gold-standard protocol divides the dataset randomly into two
**half-datasets**, each reconstructed independently into a 3D density map. Comparing these
two half-maps tells you how much signal (vs. noise) is present at each spatial frequency —
this is the **Fourier Shell Correlation (FSC)**.

### The FSC as a Function of Spatial Frequency

The FSC is defined in reciprocal space. For each shell of spatial frequency $|\mathbf{k}|$
(radius $s$ in the Fourier volume), compute the normalised cross-correlation between the two
half-map Fourier transforms $F_1$ and $F_2$:

$$
\boxed{\mathrm{FSC}(s) = \frac{\displaystyle\sum_{|\mathbf{k}|=s} F_1(\mathbf{k})\, F_2^*(\mathbf{k})}{\sqrt{\displaystyle\sum_{|\mathbf{k}|=s} |F_1(\mathbf{k})|^2 \cdot \displaystyle\sum_{|\mathbf{k}|=s} |F_2(\mathbf{k})|^2}}}
$$

FSC ranges from 1 (perfect agreement — pure signal) to 0 (no correlation — pure noise).
The spatial frequency at which $\mathrm{FSC}(s)$ crosses a threshold determines the
**resolution** of the reconstruction.

### The 0.143 Gold-Standard Threshold

The **0.143 criterion** (Rosenthal & Henderson 2003; Scheres & Chen 2012) is the standard
threshold for reporting cryo-EM resolution. The value 0.143 was derived from the half-map
FSC relationship to the full-dataset FSC via the Rosenthal-Henderson formula:

$$
\mathrm{FSC}_{\mathrm{full}}(s) = \frac{2\,\mathrm{FSC}_{1/2}(s)}{1 + \mathrm{FSC}_{1/2}(s)}
$$

When $\mathrm{FSC}_{1/2}(s) = 0.143$, $\mathrm{FSC}_{\mathrm{full}}(s) = 0.25$, which
corresponds roughly to a signal-to-noise ratio of 1 in the full map. Reporting the spatial
frequency $s^*$ where $\mathrm{FSC}_{1/2}(s^*) = 0.143$ as the map resolution is
conservative: it relies only on internal data consistency, not on reference model bias.

> [!WARNING]
> Resolution defined by FSC is the *average* resolution across the entire map. Local
> resolution varies enormously within a particle — flexible loops can be at 8 Å while the
> core rigid domain is at 2.5 Å. Use local resolution estimation (e.g., MonoRes,
> ResMap) to identify well-resolved regions for restraint-based fitting.

### Differentiability with Respect to the Map

In diff-biophys, the FSC loss is used to refine an atomic model into a density map. The
predicted density map $\rho_{\mathrm{pred}}(\mathbf{r})$ is computed by convolving atom
positions with Gaussian kernels:

$$
\rho_{\mathrm{pred}}(\mathbf{r}) = \sum_i w_i \exp\!\left(-\frac{|\mathbf{r} - \mathbf{r}_i|^2}{2\sigma^2}\right)
$$

The FSC between $\rho_{\mathrm{pred}}$ and the experimental map $\rho_{\mathrm{exp}}$ is
computed via 3D FFT. Since the FFT is a linear operation and the Gaussian blurring is
differentiable, the gradient $\partial \mathrm{FSC}(s) / \partial \mathbf{r}_i$ is computed
automatically via JAX autodiff. The gradient at each atom encodes: *moving atom $i$ in this
direction would improve phase agreement with the experimental map at spatial frequency $s$*.

### Gradient Interpretation

$$
\frac{\partial \mathrm{FSC}(s)}{\partial \mathbf{r}_i} = \frac{\partial \mathrm{FSC}}{\partial \rho_{\mathrm{pred}}} \cdot \frac{\partial \rho_{\mathrm{pred}}}{\partial \mathbf{r}_i}
$$

$$
\frac{\partial \rho_{\mathrm{pred}}}{\partial \mathbf{r}_i} = w_i\,\frac{\mathbf{r} - \mathbf{r}_i}{\sigma^2}\,\exp\!\left(-\frac{|\mathbf{r} - \mathbf{r}_i|^2}{2\sigma^2}\right)
$$

This gradient has a spatial range of $\sim \sigma$ around each atom. Atoms in high-density
regions of the experimental map and in high-FSC shells receive larger gradient contributions —
exactly the regions where the structural restraint is most reliable.

---

## 9. Kabsch Alignment

### Physical Intuition

Before computing RMSD between two sets of atom positions (e.g., a predicted model and a
crystal structure), the structures must be **superimposed**: translated so that their
centres of mass coincide, then rotated to minimise the residual distance. The **Kabsch
algorithm** (Kabsch 1976) finds the *optimal* rotation matrix for this superposition in
closed form via SVD.

### RMSD Minimisation

Given two sets of $N$ corresponding atom positions $\{\mathbf{p}_i\}$ (model) and
$\{\mathbf{q}_i\}$ (reference), both centred at the origin:

$$
\mathrm{RMSD} = \sqrt{\frac{1}{N}\sum_{i=1}^N |\mathbf{R}\,\mathbf{p}_i - \mathbf{q}_i|^2}
$$

Minimising RMSD over all proper rotations $\mathbf{R} \in SO(3)$ is equivalent to
maximising $\mathrm{tr}(\mathbf{R}\,\mathbf{H})$ where $\mathbf{H}$ is the
**cross-covariance matrix**:

$$
\mathbf{H} = \sum_{i=1}^N \mathbf{p}_i\,\mathbf{q}_i^T
$$

### SVD-Based Optimal Rotation

Compute the SVD of $\mathbf{H}$:

$$
\mathbf{H} = \mathbf{U}\,\boldsymbol{\Sigma}\,\mathbf{V}^T
$$

The optimal rotation matrix is:

$$
\boxed{\mathbf{R}^* = \mathbf{V}\,\mathrm{diag}(1,\,1,\,d)\,\mathbf{U}^T}
$$

where $d = \mathrm{sign}(\det(\mathbf{V}\mathbf{U}^T)) \in \{+1, -1\}$. The $d$ factor
ensures that $\mathbf{R}^*$ is a proper rotation ($\det\mathbf{R}^* = +1$, not a reflection).

> [!IMPORTANT]
> When two or more singular values of $\mathbf{H}$ are equal (a degenerate case), the
> optimal rotation is not unique. This can cause discontinuous gradient behaviour during
> optimisation. diff-biophys adds a small regularisation to $\mathbf{H}$ to lift degeneracies.

### Differentiability Through SVD

JAX's `jax.numpy.linalg.svd` is differentiable, so the entire Kabsch alignment pipeline
is differentiable. The gradient of RMSD with respect to model atom positions $\mathbf{p}_i$
propagates through:

1. The centring step (subtract mean — linear, trivially differentiable).
2. The cross-covariance $\mathbf{H}$ (bilinear in $\mathbf{p}$ and $\mathbf{q}$).
3. The SVD (differentiable via matrix perturbation theory).
4. The rotation application and distance sum.

$$
\frac{\partial \mathrm{RMSD}}{\partial \mathbf{p}_i} = \frac{1}{N\,\mathrm{RMSD}}\left(\mathbf{R}^* \mathbf{p}_i - \mathbf{q}_i\right) + \text{(rotation gradient)}
$$

The first term is the direct positional residual; the second term accounts for how moving
atom $i$ changes the optimal rotation itself — a non-trivial but automatically handled
contribution.

### Gradient Interpretation

The gradient $\partial \mathrm{RMSD}/\partial \mathbf{p}_i$ points from the current
(rotated) position of atom $i$ toward its target position in the reference structure. The
magnitude is inversely proportional to the current RMSD (it diverges as RMSD $\to 0$, but
the loss $\mathrm{RMSD}^2$ has well-behaved gradients). Atoms with large displacement from
the reference contribute most to the gradient, making RMSD minimisation naturally focus on
the worst-fitting regions.

---

## 10. Radius of Gyration

### Physical Intuition

The **radius of gyration** $R_g$ is a single number that characterises how compact or
extended a molecular structure is — the root-mean-square distance of all atoms from the
centre of mass. A tightly folded globular protein has a small $R_g$; an unfolded chain or
elongated rod has a large one. $R_g$ appears naturally in both SAXS (Guinier approximation)
and as a standalone compactness restraint in structure prediction.

### Definition

For $N$ atoms at positions $\{\mathbf{r}_i\}$, the uniform-weight radius of gyration is:

$$
R_g^2 = \frac{1}{N}\sum_{i=1}^N |\mathbf{r}_i - \mathbf{r}_{\mathrm{cm}}|^2
$$

where the centre of mass is $\mathbf{r}_{\mathrm{cm}} = \frac{1}{N}\sum_i \mathbf{r}_i$.

### Mass-Weighted Version

In real molecules each atom has a different mass $m_i$ (or, in a coarse-grained
representation, a different weight). The mass-weighted $R_g$ is:

$$
\boxed{R_g^2 = \frac{\sum_i m_i\,|\mathbf{r}_i - \mathbf{r}_{\mathrm{cm}}|^2}{\sum_i m_i}, \quad \mathbf{r}_{\mathrm{cm}} = \frac{\sum_i m_i\,\mathbf{r}_i}{\sum_i m_i}}
$$

For SAXS, the relevant $R_g$ is electron-density weighted (effectively weighted by the
number of electrons per atom, $Z_i$). diff-biophys uses $Z_i$ weights by default when
computing $R_g$ from an all-atom model for Guinier analysis.

### Scaling Laws

Polymer physics predicts that for an unfolded chain of $N$ residues, $R_g \propto N^\nu$
where the Flory exponent $\nu \approx 0.6$ for a good solvent. For a compact globular
protein:

$$
R_g \approx r_0 N^{1/3}, \quad r_0 \approx 4.75\,\text{Å} \quad (\text{Dima \& Thirumalai 2004})
$$

This scaling allows a rough sanity check: if the computed $R_g$ is much larger than
$r_0 N^{1/3}$, the model is likely unfolded or incorrect.

### Use as a Compaction Restraint

A loss term penalising deviation from a target radius of gyration $R_g^{\mathrm{target}}$:

$$
\mathcal{L}_{R_g} = \lambda_{R_g}\,\left(R_g - R_g^{\mathrm{target}}\right)^2
$$

provides a gentle, global compaction or expansion force. Unlike pairwise distance
restraints, $R_g$ acts on all atoms simultaneously through the centre of mass.

### Gradient

$$
\frac{\partial R_g}{\partial \mathbf{r}_k} = \frac{m_k}{R_g \sum_j m_j}\left[(\mathbf{r}_k - \mathbf{r}_{\mathrm{cm}}) - \frac{m_k}{\sum_j m_j}\sum_i m_i(\mathbf{r}_i - \mathbf{r}_{\mathrm{cm}})\right]
$$

For uniform masses this simplifies to:

$$
\frac{\partial R_g}{\partial \mathbf{r}_k} = \frac{1}{N\,R_g}\left(\mathbf{r}_k - \mathbf{r}_{\mathrm{cm}}\right)
$$

**Gradient interpretation:** The gradient for atom $k$ points radially outward from the
centre of mass. A large $|\partial R_g / \partial \mathbf{r}_k|$ means that atom $k$ is
far from the centroid — it is a peripheral atom whose movement would most change the overall
compaction. When the target $R_g$ is smaller than the current one, the loss gradient drives
all peripheral atoms inward, globally collapsing the structure.

> [!TIP]
> The $R_g$ restraint is particularly valuable in the early stages of *ab initio* folding from
> dihedrals: without it, unconstrained torsion-space optimisation often generates highly
> extended structures that satisfy local restraints (J-couplings, chemical shifts) but are
> unphysically large. A mild $\lambda_{R_g}$ provides a global shape prior.

---

## References

1. **Parsons, J., Holmes, J.B., Rojas, J.M., Tsai, J., Strauss, C.E.M.** (2005).
   Practical conversion from torsion space to Cartesian space for in silico protein
   synthesis. *J. Comput. Chem.* **26**, 1063–1068.

2. **Debye, P.** (1915). Zerstreuung von Röntgenstrahlen. *Ann. Phys.* **351**, 809–823.

3. **Guinier, A.** (1939). La diffraction des rayons X aux très petits angles:
   applications à l'étude de phénomènes ultramicroscopiques. *Ann. Phys.* **12**, 161–237.

4. **Fraser, R.D.B., MacRae, T.P., Suzuki, E.** (1978). An improved method for
   calculating the contribution of solvent to the X-ray diffraction pattern of biological
   molecules. *J. Appl. Crystallogr.* **11**, 693–694.

5. **Waasmaier, D., Kirfel, A.** (1995). New analytical scattering-factor functions for
   free atoms and ions. *Acta Crystallogr. A* **51**, 416–431.

6. **Karplus, M.** (1959). Contact electron-spin coupling of nuclear magnetic moments.
   *J. Chem. Phys.* **30**, 11–15.

7. **Vuister, G.W., Bax, A.** (1993). Quantitative J correlation: a new approach for
   measuring homonuclear three-bond $J(\text{H}^N\text{H}^\alpha)$ coupling constants in
   $^{15}$N-enriched proteins. *J. Am. Chem. Soc.* **115**, 7772–7777.

8. **Wishart, D.S., Sykes, B.D.** (1994). The $^{13}$C chemical-shift index: a simple
   method for the identification of protein secondary structure using $^{13}$C
   chemical-shift data. *J. Biomol. NMR* **4**, 171–180.

9. **Saupe, A.** (1968). Recent results in the field of liquid crystals. *Angew. Chem.
   Int. Ed. Engl.* **7**, 97–112.

10. **Bax, A., Tjandra, N.** (1997). High-resolution heteronuclear NMR of human ubiquitin
    in an aqueous liquid crystalline medium. *J. Biomol. NMR* **10**, 289–292.

11. **Johnson, C.E., Bovey, F.A.** (1958). Calculation of nuclear magnetic resonance
    spectra of aromatic hydrocarbons. *J. Chem. Phys.* **29**, 1012–1014.

12. **Haigh, C.W., Mallion, R.B.** (1979). Ring current theories in nuclear magnetic
    resonance. *Prog. NMR Spectrosc.* **13**, 303–344.

13. **Bayley, P.M., Nielsen, E.B., Schellman, J.A.** (1969). The rotatory properties of
    molecules containing two peptide groups: theory. *J. Phys. Chem.* **73**, 228–243.

14. **Rosenthal, P.B., Henderson, R.** (2003). Optimal determination of particle
    orientation, absolute hand, and contrast loss in single-particle electron
    cryomicroscopy. *J. Mol. Biol.* **333**, 721–745.

15. **Scheres, S.H.W., Chen, S.** (2012). Prevention of overfitting in cryo-EM structure
    determination. *Nat. Methods* **9**, 853–854.

16. **Kabsch, W.** (1976). A solution for the best rotation to relate two sets of vectors.
    *Acta Crystallogr. A* **32**, 922–923.

17. **Dima, R.I., Thirumalai, D.** (2004). Asymmetry in the shapes of folded and
    denatured states of proteins. *J. Phys. Chem. B* **108**, 6564–6570.
