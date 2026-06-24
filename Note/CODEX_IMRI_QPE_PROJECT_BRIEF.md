# Codex Project Brief: IMRI Minidisk Limit-Cycle Model for Ansky-like QPEs

## 0. Scientific goal

This project implements and tests a semi-analytic/numerical model for quasi-periodic eruptions (QPEs) from an intermediate-mass-ratio inspiral (IMRI) embedded in a transient tidal-disruption-event (TDE) disk.

The central model is **not** that all gas inside the Hill sphere suddenly accretes once the total Hill mass exceeds a threshold. The more defensible version is:

> Gas captured through the IMBH Hill sphere forms a tidally truncated circumsecondary disk. The disk undergoes a thermal--viscous low-to-high state transition when the surface density in an unstable annulus crosses the upper turning point of an S-curve. The Hill flow supplies mass, angular momentum, entropy, and shock heating; the bound minidisk produces the limit cycle.

The implementation should answer four questions:

1. Does captured gas circularize inside the IMBH Hill sphere?
2. Does the tidally truncated circumsecondary disk possess two stable branches separated by an unstable branch?
3. Can the resulting limit cycle reproduce the observed recurrence time, duty cycle, flare energy, and period evolution?
4. Can a simple wind/photosphere model map the burst accretion rate to soft X-ray and UV observables?

---

## 1. Fiducial system and notation

Use cgs units internally.

Fiducial parameters:

```text
M_smbh = 1e6 Msun
M2     = 1e4 Msun
q      = M2 / M_smbh = 1e-2
a      = 50 r_g,smbh
mu     = 0.62
kappa_es = 0.34 cm^2 g^-1
```

Core definitions:

```math
q = M_2/M_\bullet,
```

```math
R_H = a (q/3)^{1/3},
```

```math
\Omega_b = (G M_\bullet/a^3)^{1/2},
```

```math
G M_2 = 3\Omega_b^2 R_H^3.
```

Captured gas has specific angular momentum

```math
j_{\rm in}=\lambda \Omega_b R_H^2,
```

so its circularization radius is

```math
R_c = \frac{j_{\rm in}^2}{G M_2} = \frac{\lambda^2}{3}R_H.
```

A circumsecondary disk exists only if

```math
R_{\rm ISCO} < R_c < R_{\rm out}, \qquad R_{\rm out}=f_t R_H.
```

---

## 2. Recommended repository structure

A clean structure for the Codex project is:

```text
imri_qpe_model/
  README.md
  CODEX_IMRI_QPE_PROJECT_BRIEF.md
  papers/
    README.md
    ansky_observations/
    disk_instability/
    radiation_mhd/
    hill_flow_cpd/
    tde_disk_evolution/
  src/
    imri_qpe/
      __init__.py
      constants.py
      parameters.py
      units.py
      layer1_hill_flow/
        __init__.py
        hill_geometry.py
        capture_diagnostics.py
        stress_diagnostics.py
        source_terms.py
      layer2_scurve/
        __init__.py
        vertical_structure.py
        stress_laws.py
        opacity.py
        thermal_equilibrium.py
        turning_points.py
      layer3_minidisk_1d/
        __init__.py
        grid.py
        source_profiles.py
        diffusion_solver.py
        thermal_solver.py
        limit_cycle.py
      layer4_emission/
        __init__.py
        wind.py
        photosphere.py
        lightcurve.py
        uv_delay.py
  notebooks/
    01_fiducial_scales.ipynb
    02_scurve_demo.ipynb
    03_limit_cycle_demo.ipynb
    04_emission_demo.ipynb
  tests/
    test_hill_geometry.py
    test_vertical_structure.py
    test_scurve.py
    test_limit_cycle.py
  outputs/
    figures/
    tables/
    runs/
```

Start with Python modules and notebooks. Keep hydrodynamic simulation data interfaces separate from the analytic solvers so that the project can run without hydro data at first.

---

# 3. Four-layer implementation program

## Layer 1: Three-dimensional Hill-flow hydrodynamics interface

### Scientific purpose

The hydro layer should determine what the large-scale disk delivers to the IMBH. It should **not** be asked to self-consistently produce a radiation-pressure limit cycle unless radiation physics is included.

The key outputs are:

```math
\dot M_{\rm cap}(t),\quad j_{\rm in}(t),\quad s_{\rm in}(t),\quad R_c,
```

```math
t_{\rm recyc},\quad Q_{\rm stream}^{+}(R,t),\quad \alpha_{\rm shock}(R,t).
```

### Minimal analytic functions to implement first

File: `src/imri_qpe/layer1_hill_flow/hill_geometry.py`

Functions:

```python
def hill_radius(a_cm: float, q: float) -> float:
    """Return R_H = a (q/3)^(1/3)."""


def binary_omega(M_smbh_g: float, a_cm: float) -> float:
    """Return orbital frequency around SMBH."""


def circularization_radius(R_H_cm: float, lambda_j: float) -> float:
    """Return R_c = lambda_j^2 R_H / 3."""


def is_minidisk_allowed(R_c_cm: float, R_isco_cm: float, R_out_cm: float) -> bool:
    """Check R_ISCO < R_c < R_out."""
```

File: `src/imri_qpe/layer1_hill_flow/capture_diagnostics.py`

Functions:

```python
def capture_fraction(mdot_cap, mdot_leak):
    """Return f_cap = Mdot_cap / Mdot_leak, with safe handling of zero input."""


def mean_specific_angular_momentum(mass_flux, j_flux):
    """Return flux-weighted specific angular momentum."""


def recycling_time(M_H, mdot_in, mdot_cap):
    """Estimate Hill-flow recycling time from mass budget."""
```

File: `src/imri_qpe/layer1_hill_flow/stress_diagnostics.py`

Key diagnostic:

```math
W_{R\phi}^{\rm shock}=\int \rho\,\delta v_R\delta v_\phi\,dz,
```

```math
\alpha_{\rm shock}=\frac{W_{R\phi}^{\rm shock}}{\int P\,dz}.
```

Functions:

```python
def reynolds_stress(rho, dv_R, dv_phi, dz):
    """Compute vertically integrated Reynolds/shock stress."""


def alpha_shock(W_Rphi, P, dz):
    """Compute alpha_shock = W_Rphi / int P dz."""
```

### Hydro data expected later

The hydro code should output at least:

```text
rho(x,y,z,t)
P(x,y,z,t)
vx, vy, vz
Phi_secondary
passive tracer or bound/unbound flag, if available
sink accretion rate
mass flux through Hill boundary
angular momentum flux through Hill boundary
```

### Codex task

Implement the Layer-1 functions and unit tests first. Do not assume a particular hydro data format; design simple functions that work on numpy arrays.

---

## Layer 2: Vertical-equilibrium and S-curve solver

### Scientific purpose

This is the first decisive theoretical layer. It asks whether a tidally truncated circum-IMBH disk has a multivalued equilibrium curve.

At each radius and surface density, solve for

```math
T_c,\quad H,\quad \tau,\quad T_{\rm eff},\quad \dot M(\Sigma;R),
```

and then identify

```math
\Sigma_{\max}(R),\quad \Sigma_{\min}(R).
```

### Vertical structure equations

At radius `R`, around secondary mass `M2`, define

```math
\Omega_K = (G M_2/R^3)^{1/2}.
```

Use

```math
\rho_c = \frac{\Sigma}{2H},
```

```math
P_c \simeq \rho_c \Omega_K^2 H^2 = \frac{1}{2}\Sigma\Omega_K^2H,
```

```math
P_c=P_{\rm gas}+P_{\rm rad}
=\rho_c {\cal R}T_c + \frac{a_r T_c^4}{3},
```

where

```math
{\cal R}=k_B/(\mu m_p).
```

The diffusion cooling rate is

```math
Q_{\rm rad}^{-}=\frac{16\sigma_{\rm SB}T_c^4}{3\kappa\Sigma}.
```

### Stress closure

Implement a generalized stress law:

```math
\tau_{R\phi}=\alpha P_{\rm gas}^{\mu}P_{\rm tot}^{1-\mu}.
```

Important special cases:

```text
mu = 0: total-pressure stress, classically unstable in radiation-pressure regime.
mu = 1: gas-pressure stress, usually thermally stable.
```

In the radiation-pressure regime, thermal instability requires

```math
\mu < 4/7.
```

### Useful analytic estimate

The transition scale can be estimated by setting `P_rad ≈ P_gas` on the cool branch:

```math
T_{\rm tr}^4 = \frac{32\sigma_{\rm SB}\Omega_K}{3\alpha_c\kappa a_r^2},
```

```math
\Sigma_{\rm tr}
=\left(\frac{128\sigma_{\rm SB}}{27\alpha_c\kappa {\cal R}\Omega_K}\right)^{1/2}T_{\rm tr}^{3/2}.
```

The participating mass is roughly

```math
M_{\rm crit}\simeq f_A\pi R_u^2\Sigma_{\max}.
```

A radial calculation should instead use

```math
\Delta M_{\rm cyc}=2\pi\int_{R_1}^{R_2}
[\Sigma_{\max}(R)-\Sigma_{\min}(R)]R\,dR.
```

### Recommended files

File: `src/imri_qpe/layer2_scurve/vertical_structure.py`

Functions:

```python
def omega_k(M2_g, R_cm):
    """Keplerian angular frequency around secondary."""


def gas_pressure(rho, T, mu=0.62):
    """Return rho * k_B T / (mu m_p)."""


def radiation_pressure(T):
    """Return a_rad T^4 / 3."""


def solve_scale_height(Sigma, T, M2_g, R_cm, mu=0.62):
    """Solve vertical hydrostatic balance for H."""
```

File: `src/imri_qpe/layer2_scurve/stress_laws.py`

```python
def stress_pressure(Pgas, Ptot, alpha, mu_stress):
    """Return alpha * Pgas^mu * Ptot^(1-mu)."""


def instability_mu_criterion(mu_stress):
    """Return True if mu < 4/7."""
```

File: `src/imri_qpe/layer2_scurve/thermal_equilibrium.py`

```python
def q_rad_minus(T, Sigma, kappa=0.34):
    """Optically thick radiative cooling per disk face convention as adopted in note."""


def q_plus_alpha(Sigma, T, R_cm, M2_g, alpha, mu_stress):
    """Viscous heating for chosen stress law."""


def equilibrium_residual(T, Sigma, R_cm, M2_g, params):
    """Return Qplus - Qminus - Qadv - Qwind + stream terms."""
```

File: `src/imri_qpe/layer2_scurve/turning_points.py`

```python
def compute_scurve(R_cm, Sigma_grid, params):
    """Return equilibrium branches T(Sigma), Mdot(Sigma), stability flags."""


def find_turning_points(Sigma_grid, Mdot_grid):
    """Locate Sigma_min and Sigma_max from local extrema or slope changes."""
```

### Codex task

Implement the analytic transition estimates first. Then implement a numerical equilibrium solver that scans over `Sigma` and roots for `T_c`.

Start without advection/wind, then add optional stabilizing terms. The immediate test is whether the solver recovers the expected gas-pressure stable scaling and the radiation-pressure instability condition.

---

## Layer 3: One-dimensional time-dependent minidisk

### Scientific purpose

Layer 3 determines whether the local S-curve becomes a coherent time-dependent limit cycle with the correct recurrence time and duty cycle.

The mass equation is

```math
\frac{\partial\Sigma}{\partial t}
+\frac{1}{R}\frac{\partial}{\partial R}(R\Sigma v_R)
=S_m(R,t)-\dot\Sigma_w.
```

The angular-momentum equation is

```math
\frac{\partial(\Sigma l)}{\partial t}
+\frac{1}{R}\frac{\partial}{\partial R}(R\Sigma v_R l)
=\frac{1}{R}\frac{\partial}{\partial R}(R^2 W_{R\phi})
+\Sigma\Lambda_{\rm tide}
+S_m l_{\rm in}
-\dot\Sigma_w l_w.
```

The energy equation is

```math
\frac{\partial U}{\partial t}
+\frac{1}{R}\frac{\partial}{\partial R}(R U v_R)
=Q_{\rm visc}^{+}+Q_{\rm stream}^{+}+Q_{\rm tide}^{+}+Q_{\rm irr}^{+}
-Q_{\rm rad}^{-}-Q_{\rm adv}^{-}-Q_{\rm wind}^{-}.
```

### Reduced one-zone limit-cycle model

Before writing the full 1D solver, implement the reduced model:

```math
M_{\min}=\zeta M_{\max},
```

```math
\Delta M=(1-\zeta)M_{\max},
```

```math
t_{\rm load}\simeq\frac{\Delta M}{\dot M_{\rm cap}-\dot M_{\rm low}},
```

```math
t_{\rm high}\simeq\frac{1}{\alpha_h(H/R)^2_h\Omega_K},
```

```math
P_{\rm QPE}\simeq t_{\rm load}+t_{\rm trans}+t_{\rm high},
```

```math
D\simeq\frac{t_{\rm high}}{P_{\rm QPE}},
```

```math
\dot M_{\rm burst}\simeq\frac{\Delta M}{t_{\rm high}}.
```

### Recommended files

File: `src/imri_qpe/layer3_minidisk_1d/limit_cycle.py`

```python
def one_zone_cycle(Mmax, zeta, mdot_cap, mdot_low, alpha_hot, H_over_R_hot, Omega_K, t_trans=0.0):
    """Return DeltaM, t_load, t_high, P_QPE, duty_cycle, Mdot_burst."""
```

File: `src/imri_qpe/layer3_minidisk_1d/grid.py`

```python
def make_log_grid(R_in, R_out, n):
    """Return radial cell centers and edges."""
```

File: `src/imri_qpe/layer3_minidisk_1d/source_profiles.py`

```python
def gaussian_source(R, R_c, width, Mdot_cap):
    """Return S_m(R) normalized to total Mdot_cap."""
```

File: `src/imri_qpe/layer3_minidisk_1d/diffusion_solver.py`

Start with a simple viscous diffusion equation for `Sigma`, then couple to thermal branches later.

### Codex task

Implement the one-zone model first and reproduce the fiducial numbers in the note:

```text
Mmax = 1.6e-4 Msun
zeta = 0.2
Mdot_cap = 1e-2 Msun/yr
alpha_hot * (H/R)^2 = 1e-3
Expected: P_QPE ~ 6-7 d, duty cycle ~ 0.3, Mdot_burst ~ few 1e-2 Msun/yr
```

Then implement a prototype radial diffusion model with a prescribed S-curve switch.

---

## Layer 4: Wind and synthetic-emission model

### Scientific purpose

Layer 4 maps dynamical accretion rates to observables:

```math
L_X(t),\quad T_{\rm col}(t),\quad R_{\rm ph}(t),\quad L_{\rm UV}(t).
```

The dynamical model predicts available accretion power, not directly the observed X-ray luminosity. At super-Eddington rates, photon trapping, winds, reprocessing, and beaming can dominate the observed signal.

### Minimal energetics

```math
L_{\rm acc}=\eta \dot M_{\rm burst}c^2.
```

But for super-Eddington flow, use a saturated luminosity model such as

```math
L_{\rm bol}\sim L_{\rm Edd}[1+\ln(\dot m)],
```

with possible beaming or reprocessing factors.

### Wind/photosphere estimates

Use a first-pass wind photosphere model:

```math
R_{\rm ph}\sim \frac{\kappa \dot M_w}{4\pi v_w},
```

```math
T_{\rm eff}\sim \left(\frac{L_{\rm bol}}{4\pi\sigma_{\rm SB}R_{\rm ph}^2}\right)^{1/4}.
```

Delayed UV emission can be modeled with either a light-travel delay

```math
t_{\rm lag}\sim R_{\rm rep}/c,
```

or a diffusion delay

```math
t_{\rm diff}\sim \tau R_{\rm rep}/c.
```

### Recommended files

File: `src/imri_qpe/layer4_emission/wind.py`

```python
def eddington_luminosity(M_g):
    """Return L_Edd."""


def eddington_mdot(M_g, eta=0.1):
    """Return Mdot_Edd = L_Edd / (eta c^2)."""


def wind_photosphere_radius(mdot_w, v_w, kappa=0.34):
    """Return R_ph = kappa Mdot_w / (4 pi v_w)."""
```

File: `src/imri_qpe/layer4_emission/lightcurve.py`

```python
def supereddington_luminosity(mdot, M_g, eta=0.1, model="log"):
    """Return a bolometric luminosity prescription."""


def blackbody_temperature(L, R_ph):
    """Return effective temperature from L and R_ph."""
```

File: `src/imri_qpe/layer4_emission/uv_delay.py`

```python
def light_travel_delay(R):
    """Return R/c."""


def diffusion_delay(tau, R):
    """Return tau R/c."""
```

### Codex task

Implement a minimal post-processing model that takes `Mdot_burst(t)` from Layer 3 and produces `Lbol(t)`, `Rph(t)`, `Teff(t)`, and a delayed UV proxy.

---

# 4. Minimal milestones

## Milestone 1: Fiducial scales notebook

Notebook: `notebooks/01_fiducial_scales.ipynb`

Goal: reproduce the fiducial scales in the note.

Outputs:

```text
R_H
R_c
Omega_K(R_c)
T_tr
Sigma_tr
Mcrit
Eddington luminosity and Eddington accretion rate for M2
```

## Milestone 2: S-curve notebook

Notebook: `notebooks/02_scurve_demo.ipynb`

Goal: compute local equilibrium curves for a grid of `Sigma` and different `mu_stress`.

Success criterion:

```text
mu < 4/7 should show radiation-pressure thermal instability in the simplified model.
mu >= 4/7 should suppress it.
```

## Milestone 3: One-zone limit-cycle notebook

Notebook: `notebooks/03_limit_cycle_demo.ipynb`

Goal: use the one-zone relaxation oscillator to reproduce day-scale recurrence and duty cycle.

Outputs:

```text
M(t)
Mdot_acc(t)
P_QPE
D
burst energy
```

## Milestone 4: Emission post-processing notebook

Notebook: `notebooks/04_emission_demo.ipynb`

Goal: turn `Mdot_acc(t)` into toy `L_X(t)`, `T_eff(t)`, and delayed UV proxy.

---

# 5. Key falsification checks

The model is disfavored if any of the following is true:

1. Captured gas does not circularize inside the allowed minidisk region: `R_ISCO < R_c < R_out` fails.
2. The local equilibrium curve is single-valued for plausible stress laws and opacity.
3. The unstable annulus lies outside the tidally truncated disk.
4. The participating mass is too small to supply the flare energy.
5. The participating mass is too large to reload on the observed timescale.
6. The hot branch drains much faster than the observed multi-day flare duration.
7. The required captured supply exceeds plausible TDE disk supply.
8. Wind/reprocessing cannot produce the observed soft X-ray luminosity and UV lag.

Conservation-law checks:

```math
E_{\rm flare}=\int L_{\rm bol}(t)dt,
```

```math
\Delta M_{\rm acc}\geq \frac{E_{\rm flare}}{\eta_{\max}c^2},
```

```math
\langle \dot M_{\rm cap}\rangle \geq \frac{\Delta M_{\rm acc}}{P_{\rm QPE}}.
```

---

# 6. Reference papers to download into the Codex project

Suggested folder structure:

```text
papers/
  ansky_observations/
  disk_instability/
  radiation_mhd/
  hill_flow_cpd/
  tde_disk_evolution/
```

## 6.1 Ansky / QPE observations

| Short key | Paper | Download / source links | Suggested folder |
|---|---|---|---|
| `Chakraborty2026_Ansky_Pdot` | Chakraborty et al. 2026, *A positive period derivative in the quasi-periodic eruptions of ZTF19acnskyy* | arXiv abs: https://arxiv.org/abs/2602.16776 ; PDF: https://arxiv.org/pdf/2602.16776 | `papers/ansky_observations/` |
| `HernandezGarcia2025_Ansky_NICER` | Hernández-García et al. 2025, *NICER observations reveal doubled timescales in Ansky's QPEs* | arXiv abs: https://arxiv.org/abs/2509.16304 ; PDF: https://arxiv.org/pdf/2509.16304 | `papers/ansky_observations/` |
| `Guo2026_Ansky_UV_delay` | Guo et al. 2026, *Evidence for a Delayed UV Counterpart to X-ray Quasi-periodic Eruptions in Ansky* | arXiv abs: https://arxiv.org/abs/2603.02517 ; PDF: https://arxiv.org/pdf/2603.02517 | `papers/ansky_observations/` |

## 6.2 Disk instability and accretion theory

| Short key | Paper | Download / source links | Suggested folder |
|---|---|---|---|
| `ShakuraSunyaev1973_alpha_disk` | Shakura & Sunyaev 1973, *Black holes in binary systems. Observational appearance* | ADS: https://ui.adsabs.harvard.edu/abs/1973A%26A....24..337S/abstract ; ADS full: https://adsabs.harvard.edu/full/1973A%26A....24..337S | `papers/disk_instability/` |
| `LightmanEardley1974_instability` | Lightman & Eardley 1974, *Black holes in binary systems: instability of disk accretion* | ADS: https://ui.adsabs.harvard.edu/abs/1974ApJ...187L...1L/abstract ; ADS full: https://adsabs.harvard.edu/full/1974ApJ...187L...1L ; DOI: https://doi.org/10.1086/181377 | `papers/disk_instability/` |
| `Abramowicz1988_slim_disks` | Abramowicz et al. 1988, *Slim Accretion Disks* | ADS: https://ui.adsabs.harvard.edu/abs/1988ApJ...332..646A/abstract ; DOI: https://doi.org/10.1086/166683 | `papers/disk_instability/` |
| `Pan2022_QPE_disk_instability` | Pan et al. 2022, *A disk instability model for the quasi-periodic eruptions of GSN 069* | arXiv abs: https://arxiv.org/abs/2203.12137 ; PDF: https://arxiv.org/pdf/2203.12137 ; DOI: https://doi.org/10.3847/2041-8213/ac5faf | `papers/disk_instability/` |

## 6.3 Radiation-MHD stability and magnetic support

| Short key | Paper | Download / source links | Suggested folder |
|---|---|---|---|
| `Hirose2009_rad_dominated_stable` | Hirose, Krolik & Blaes 2009, *Radiation-Dominated Disks Are Thermally Stable* | arXiv abs: https://arxiv.org/abs/0809.1708 ; PDF: https://arxiv.org/pdf/0809.1708 | `papers/radiation_mhd/` |
| `Jiang2013_rad_thermal_stability` | Jiang, Stone & Davis 2013, *On the Thermal Stability of Radiation Dominated Accretion Disks* | arXiv abs: https://arxiv.org/abs/1309.5646 ; PDF: https://arxiv.org/pdf/1309.5646 | `papers/radiation_mhd/` |
| `Kaur2023_magnetic_TDE_QPE` | Kaur, Stone & Gilbaum 2023, *Magnetically Dominated Disks in Tidal Disruption Events and Quasi-Periodic Eruptions* | arXiv abs: https://arxiv.org/abs/2211.00704 ; PDF: https://arxiv.org/pdf/2211.00704 | `papers/radiation_mhd/` |

## 6.4 Hill-flow / circumplanetary-disk analogs

| Short key | Paper | Download / source links | Suggested folder |
|---|---|---|---|
| `Tanigawa2012_CPD_accretion` | Tanigawa, Ohtsuki & Machida 2012, *Distribution of Accreting Gas and Angular Momentum onto Circumplanetary Disks* | arXiv abs: https://arxiv.org/abs/1112.3706 ; PDF: https://arxiv.org/pdf/1112.3706 | `papers/hill_flow_cpd/` |
| `Zhu2016_shock_CPD` | Zhu, Ju & Stone 2016, *Shock-driven Accretion in Circumplanetary Disks: Observables and Satellite Formation* | arXiv abs: https://arxiv.org/abs/1609.09250 ; PDF: https://arxiv.org/pdf/1609.09250 | `papers/hill_flow_cpd/` |

## 6.5 TDE disk evolution

| Short key | Paper | Download / source links | Suggested folder |
|---|---|---|---|
| `ShenMatzner2014_TDE_disk_evolution` | Shen & Matzner 2014, *Evolution of Accretion Disks in Tidal Disruption Events* | arXiv abs: https://arxiv.org/abs/1305.5570 ; PDF: https://arxiv.org/pdf/1305.5570 | `papers/tde_disk_evolution/` |

---

# 7. Suggested first Codex prompt

Paste the following into Codex after creating the repository:

```text
We are implementing a semi-analytic/numerical model for QPEs from an IMRI embedded in a transient TDE disk. Read CODEX_IMRI_QPE_PROJECT_BRIEF.md and implement the project in layers.

Start with Layer 2 and the fiducial-scales notebook. Use cgs units. Implement constants.py, parameters.py, layer1_hill_flow/hill_geometry.py, and layer2_scurve/vertical_structure.py. Then write tests that reproduce the fiducial R_H, R_c, Omega_K, T_tr, Sigma_tr, and Mcrit estimates in the research note.

Do not build a full hydro solver. Layer 1 should only provide diagnostic functions for later hydro outputs. Keep all functions modular and unit-tested.
```

---

# 8. Recommended initial coding order

1. `constants.py` and `units.py`
2. `parameters.py` with a `FiducialParams` dataclass
3. `layer1_hill_flow/hill_geometry.py`
4. `layer2_scurve/vertical_structure.py`
5. analytic transition estimates `T_tr`, `Sigma_tr`, `Mcrit`
6. `notebooks/01_fiducial_scales.ipynb`
7. tests for all fiducial numbers
8. numerical S-curve root solver
9. one-zone limit-cycle solver
10. toy emission model

---

# 9. Notes on interpretation

Important: the model should be treated as conditional. A successful implementation should not hard-code the existence of a limit cycle. Instead, it should determine whether the assumed stress, opacity, advection, wind, and tidal truncation prescriptions produce the required two stable branches and participating mass.

The decisive first deliverable is not a light curve. It is the S-curve and the allowed unstable radial interval.
