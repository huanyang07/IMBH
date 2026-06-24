# IMRI Minidisk Slim/Wind Upgrade Notes for Codex

This file is intended to be inserted into the Codex project as an implementation addendum to the current `README.md` status note. It focuses on upgrading the current local, parameterized hot-branch treatment into a radially consistent slim-disk / wind-stabilized minidisk model.

The current project status is encouraging:

- Fiducial IMRI scales match the analytic note.
- The Layer-2 local vertical-equilibrium solver produces a cool stable branch and a radiation-pressure unstable branch.
- A local advective closure with imposed `xi` produces a stable hot branch and participating masses in the desired range.
- The Layer-3 one-zone model reproduces day-scale recurrence and a large duty cycle.

The main caveat is:

```text
Q_adv is still local and parameterized by xi.
```

In a faithful slim-disk/minidisk calculation, `xi` should be computed from the radial entropy gradient, not imposed as a constant.

The key upgrade is therefore:

```text
Replace local xi-based advection with a radially computed entropy-advection term,
then add stream feeding, tidal truncation, and super-Eddington wind loss in a
conservative 1D minidisk model.
```

---

## 0. Conceptual Message

The current result should be interpreted as a **promising local diagnostic**, not yet as proof of a physical stable hot branch.

A local algebraic root with

```text
Q_adv = xi * Mdot * P / (2 pi R^2 rho)
```

is useful, but the hot branch becomes physically meaningful only if it survives when

```text
Q_adv = Sigma v_R T ds/dR
```

is computed from the radial disk solution.

The central target for the next implementation is:

```text
A global stream-fed minidisk solution in which Q_adv is computed from ds/dR
and a stable hot/slim branch survives without prescribing xi.
```

## 0.1 Current Implementation Status

The near-term nearly Keplerian global diagnostic described below has now been
implemented in `src/imri_qpe/layer3_minidisk_1d/global_slim.py`.

Implemented pieces:

- `v_R(R)` from the reduced angular-momentum equation.
- `Q_adv = Sigma v_R T ds/dR` from the radial entropy gradient.
- `xi_eff(R)` as a diagnostic, not an input.
- Energy-limited wind cooling with non-negative radiative cooling.
- A fixed-Sigma temperature relaxation against the global energy residual.

Current result:

```text
outputs/figures/global_slim_wind_audit.png
outputs/figures/global_slim_audit_hardened.png
outputs/tables/global_slim_audit_hardened.md
outputs/figures/isolated_slim_branch_continuation.png
outputs/tables/isolated_slim_branch_summary.md
```

The local `xi = 0.3` hot roots do not yet survive as a faithful global hot
branch. Evaluated globally, the local-root candidate has median
`xi_eff ~= 2.2`, integrated `Q_adv/Q_visc ~= 5.8`, and global residual `-4.8`.
The fixed-Sigma relaxed profile closes the formal energy residual, but it has
broad `xi_eff`, integrated `Q_adv/Q_visc ~= -0.09`, and one outward-flow radial
zone. Therefore the current result is a useful failure of the parameterized
hot branch, not a validated slim/wind limit cycle.

The hardened audit adds L1/L2/max residuals, boundary flags, independent
entropy-gradient checks, stencil comparison, and continuity residuals. Its
resolution ladder still finds L1 residuals of order 5 for the stitched local
hot roots, so the failure is not just signed cancellation or one bad boundary
cell.

Sprint B implemented and repaired an isolated no-wind constant-`Mdot`
benchmark. It turns off stream, tide, and wind, uses a physical ISCO
angular-momentum constant, applies a consistent two-face heating/cooling
convention, and solves the Keplerian angular-momentum and energy equations
simultaneously for `Sigma(R)` and `T(R)`. The repaired solver recovers the
thin disk and follows a smooth reduced advective sequence through
`Mdot/Mdot_Edd ~= 10`; for example, `Q_adv/Q_visc ~= 0.19` at
`Mdot/Mdot_Edd = 1` and `~= 0.77` at `Mdot/Mdot_Edd = 10`. The limiting
caveat is now geometric: `H/R` exceeds `0.4` by a few `Mdot_Edd`, so the
nearly Keplerian reduction is not trustworthy at the QPE target
`Mdot/Mdot_Edd ~= 94`. This points toward a transonic slim-disk solver with
radial momentum and sonic regularity before adding stream, tide, or wind.

---

# 1. Vertically Integrated Slim-Disk Energy Equation

## 1.1 Sign convention

Use

```text
Mdot(R,t) = - 2 pi R Sigma v_R > 0
```

for inward accretion. Therefore `v_R < 0` for inflow.

Let all `Q` quantities be vertically integrated energy rates per unit disk surface area, summed over both disk faces.

## 1.2 Time-dependent entropy equation

Implement the vertically integrated entropy equation as

```math
\Sigma T
\left(
\frac{\partial s}{\partial t}
+
v_R \frac{\partial s}{\partial R}
\right)
=
Q_{\rm visc}^{+}
+
Q_{\rm stream}^{+}
+
Q_{\rm tide}^{+}
-
Q_{\rm rad}^{-}
-
Q_{\rm wind}^{-}.
```

The steady advective term is

```math
Q_{\rm adv}
\equiv
\Sigma v_R T \frac{ds}{dR}
=
-\frac{\dot M}{2\pi R}T\frac{ds}{dR}.
```

The steady energy balance is then

```math
Q_{\rm visc}^{+}
+
Q_{\rm stream}^{+}
+
Q_{\rm tide}^{+}
=
Q_{\rm rad}^{-}
+
Q_{\rm adv}
+
Q_{\rm wind}^{-}.
```

The current parameterized form

```math
Q_{\rm adv}
=
\xi \frac{\dot M P}{2\pi R^2 \rho}
```

is recovered if

```math
\xi(R)
=
-\frac{R\rho}{P}T\frac{ds}{dR}.
```

Therefore `xi` should become a **diagnostic**:

```text
xi_eff(R) = - R * rho / P * TdsdR
```

rather than an input parameter.

## 1.3 Numerically convenient entropy derivative

Use the first law:

```math
T\frac{ds}{dR}
=
\frac{de}{dR}
-
\frac{P}{\rho^2}\frac{d\rho}{dR}.
```

For a one-temperature gas-radiation mixture,

```math
P=P_{\rm gas}+P_{\rm rad},
```

```math
P_{\rm gas}=\frac{\rho k_B T}{\mu m_p},
\qquad
P_{\rm rad}=\frac{a_{\rm r}T^4}{3}.
```

The specific internal energy is

```math
e(\rho,T)
=
\frac{P_{\rm gas}}{(\gamma_{\rm gas}-1)\rho}
+
\frac{a_{\rm r}T^4}{\rho}.
```

This is more robust numerically than trying to use approximate analytic `Gamma_1` and `Gamma_3` formulae.

## 1.4 Finite-volume implementation

For each radial cell `i`:

1. Compute vertical structure:
   ```text
   H_i, rho_i, P_i, e_i
   ```

2. Compute slope-limited radial gradients:
   ```text
   de_dR_i
   drho_dR_i
   ```

3. Compute:
   ```text
   TdsdR_i = de_dR_i - P_i / rho_i**2 * drho_dR_i
   ```

4. Compute:
   ```text
   Q_adv_i = Sigma_i * v_R_i * TdsdR_i
   ```

5. Diagnose:
   ```text
   xi_eff_i = - R_i * rho_i / P_i * TdsdR_i
   ```

Do **not** force `Q_adv` to be positive. Entropy advection may act as cooling or heating depending on the entropy gradient.

## 1.5 Minimal versus full slim-disk implementation

A tractable first upgrade can assume nearly Keplerian rotation and compute `v_R` from mass and angular-momentum conservation.

A more complete slim-disk implementation should eventually solve radial momentum:

```math
\frac{\partial v_R}{\partial t}
+
v_R\frac{\partial v_R}{\partial R}
=
R(\Omega^2-\Omega_K^2)
-
\frac{1}{\Sigma}\frac{\partial \Pi}{\partial R}
+
f_{R,\rm tide},
```

where

```math
\Pi=\int P\,dz.
```

The full equation is needed to locate the sonic point and obtain a true transonic inner solution. For the next implementation milestone, this can be deferred, but it should be listed as a future validation requirement.

---

# 2. Mass Conservation, Angular Momentum, and `v_R`

Do not infer `Mdot` from local viscous heating. That creates a circular closure because `Q_adv` depends on `Mdot`, while `Mdot` is then being inferred from the same local energy balance.

Instead, compute `v_R` and `Mdot` from mass and angular-momentum conservation.

## 2.1 Mass conservation

Implement

```math
\frac{\partial\Sigma}{\partial t}
+
\frac{1}{R}
\frac{\partial}{\partial R}
\left(R\Sigma v_R\right)
=
S_\Sigma(R,t)
-
\dot\Sigma_w.
```

Here:

- `S_Sigma` is the stream-deposition source term.
- `dotSigma_w` is the wind mass-loss rate.

Normalize the source as

```math
\int 2\pi R S_\Sigma\,dR
=
\dot M_{\rm cap}.
```

A useful first source profile is a Gaussian centered on the circularization radius:

```math
S_\Sigma(R)
=
\frac{\dot M_{\rm cap}}
{2\pi R_c\sqrt{2\pi}\Delta R_c}
\exp\left[
-\frac{(R-R_c)^2}{2\Delta R_c^2}
\right].
```

The source width `Delta R_c` should eventually come from the Layer-1 Hill-flow simulation.

## 2.2 Angular-momentum conservation

Let

```math
l=R^2\Omega.
```

Define `W > 0` as the magnitude of the outward viscous stress:

```math
W=-\nu\Sigma R\frac{d\Omega}{dR}.
```

The angular-momentum equation is

```math
\frac{\partial(\Sigma l)}{\partial t}
+
\frac{1}{R}
\frac{\partial}{\partial R}
\left(R\Sigma v_R l\right)
=
-\frac{1}{R}
\frac{\partial}{\partial R}
\left(R^2W\right)
+
\Sigma\Lambda_{\rm tide}
+
S_\Sigma l_{\rm in}
-
\dot\Sigma_w l_w.
```

Subtracting `l` times the continuity equation gives, for a prescribed time-independent rotation profile,

```math
\Sigma v_R\frac{dl}{dR}
=
-\frac{1}{R}\frac{\partial}{\partial R}(R^2W)
+
\Sigma\Lambda_{\rm tide}
+
S_\Sigma(l_{\rm in}-l)
-
\dot\Sigma_w(l_w-l).
```

Therefore,

```math
v_R
=
\frac{
-R^{-1}\partial_R(R^2W)
+\Sigma\Lambda_{\rm tide}
+S_\Sigma(l_{\rm in}-l)
-\dot\Sigma_w(l_w-l)
}{
\Sigma\,dl/dR
}.
```

Then compute

```math
\dot M=-2\pi R\Sigma v_R.
```

## 2.3 Stream injection

The stream injection should be implemented as a **source term inside the domain**, not only as an outer boundary inflow.

Recommended first choices:

```text
R_c      = lambda_j^2 R_H / 3
R_out    = f_t R_H
DeltaRc = 0.05--0.2 R_c
l_in     = sqrt(G M2 R_c)
```

where `lambda_j` and `f_t` should eventually be calibrated from Layer 1.

## 2.4 Inner boundary

First implementation:

```text
R_in ~= R_ISCO
W(R_in) = 0
outflow-only mass boundary
```

Later full slim-disk implementation:

```text
Replace zero torque with sonic-point regularity.
```

The sonic-point condition turns one disk parameter, often the inner angular momentum, into an eigenvalue.

## 2.5 Outer boundary

Use

```math
R_{\rm out}=f_tR_H.
```

The preferable treatment is:

1. Deposit the stream near `R_c`.
2. Apply a tidal torque density near `R_out`.
3. Use an outflow-only or nearly zero-flux outer boundary.

Avoid the combination:

```text
gas injection through the outer boundary + hard reflecting wall
```

because it can create artificial pile-up.

A smooth tidal barrier near `R_out` is acceptable for the first version, but the final `Lambda_tide(R)` should be calibrated from the Layer-1 simulation.

---

# 3. Super-Eddington Wind Mass Loss

Once the hot branch is super-Eddington, wind loss should enter through the radial mass budget.

With the convention `Mdot > 0` inward,

```math
\frac{d\dot M}{dR}
=
2\pi R\dot\Sigma_w.
```

This means `Mdot` is larger at large radii and decreases inward as mass is lost to the wind.

## 3.1 Energy-limited wind closure

Use a locally Eddington-regulated energy-limited wind as the primary closure.

The vertical gravity at the photosphere is approximately

```math
g_z\simeq\Omega_K^2H.
```

The Eddington flux from one disk face is

```math
F_{\rm Edd,z}
=
\frac{c\Omega_K^2H}{\kappa}.
```

For both disk faces,

```math
Q_{\rm Edd,z}
=
\frac{2c\Omega_K^2H}{\kappa}.
```

Define the energy available after advection:

```math
Q_{\rm avail}
=
Q_{\rm visc}^{+}
+
Q_{\rm stream}^{+}
+
Q_{\rm tide}^{+}
-
Q_{\rm adv}.
```

Then partition the excess above the local vertical Eddington flux into wind:

```math
Q_{\rm wind}^{-}
=
\epsilon_w
\left[
Q_{\rm avail}-Q_{\rm Edd,z}
\right]_+,
```

```math
Q_{\rm rad}^{-}
=
Q_{\rm avail}-Q_{\rm wind}^{-}.
```

Here

```math
[x]_+ = \max(x,0).
```

The wind mass flux is

```math
\dot\Sigma_w
=
\frac{Q_{\rm wind}^{-}}{\mathcal E_w}.
```

A simple nonmagnetic wind energy requirement is

```math
\mathcal E_w
\simeq
\frac{GM_2}{2R}
+
\frac{v_\infty^2}{2}.
```

The first term unbinds material from a Keplerian orbit, and the second gives terminal kinetic energy.

Recommended scan:

```text
epsilon_w = 0, 0.1, 0.3, 1
v_infty   = 0, v_esc, or a parameterized fraction of v_esc
```

## 3.2 Phenomenological alternative

As a comparison or regression test, implement the standard power-law mass-loss profile:

```math
\dot M(R)
=
\dot M_{\rm out}
\left(\frac{R}{R_{\rm sph}}\right)^s,
\qquad
0\lesssim s\lesssim1,
```

inside the spherization radius.

Use this only as a bracketing model. The energy-limited wind is preferable as the main closure because it enforces energy accounting locally.

---

# 4. Wind Angular Momentum and Energy

Wind effects must be separated into:

1. Removal of mass.
2. Removal of angular momentum.
3. Removal of energy.

## 4.1 Angular momentum carried by wind

The angular-momentum sink is

```math
-\dot\Sigma_w l_w.
```

For the hydro/radiative baseline, use

```math
l_w=l.
```

This means wind removes the angular momentum carried by the lost disk mass but does not apply an extra torque to the remaining disk.

A magnetic lever arm can be included later as

```math
l_w=\lambda_w l,\qquad \lambda_w>1.
```

Then the additional wind torque on the disk is governed by

```math
\dot\Sigma_w(l_w-l).
```

Do not assume `lambda_w > 1` in a non-MHD baseline.

## 4.2 Wind energy sink

A conservative total-energy equation is ideal. In a reduced internal-energy model, subtract

```math
Q_{\rm wind}^{-}
=
\dot\Sigma_w\mathcal E_w.
```

A more complete specific energy loss is

```math
\mathcal E_w
=
\frac{GM_2}{2R}
+
\frac{v_\infty^2}{2}
+
h_w
+
\Omega(l_w-l).
```

Here:

- `GM2/(2R)` unbinds gas from a Keplerian orbit.
- `v_infty^2/2` is the terminal kinetic energy.
- `h_w` is wind enthalpy.
- `Omega(l_w-l)` is work done by an additional wind torque.

For the nonmagnetic baseline, set

```math
l_w=l.
```

Then the last term vanishes.

## 4.3 Avoid double counting

Use one of the following, not both:

```text
Option A:
    compute Q_wind from excess energy and derive dotSigma_w = Q_wind/E_w

Option B:
    prescribe dotSigma_w independently and subtract Q_wind = dotSigma_w E_w
```

Do not both remove a fraction of `Q_avail` and then separately subtract `dotSigma_w E_w` again.

---

# 5. Tests for a Real Stable Hot Branch

The current local result is encouraging, but a local algebraic root generated with prescribed `xi` is not yet evidence for a physical slim branch.

The following tests should be implemented before interpreting the hot branch as real.

## Test A: Recover a standard isolated slim disk

Before adding stream feeding, tides, or wind, reproduce an isolated slim-disk sequence:

- cool thin branch at low accretion rate;
- increasing `H/R` and advected fraction at high accretion rate;
- decreasing radiative efficiency at high `Mdot`;
- smooth connection to the inner region.

This validates the energy and radial-advection implementation.

## Test B: Replace `xi` and audit `xi_eff`

For every global solution, calculate

```math
\xi_{\rm eff}(R)
=
-\frac{R\rho}{P}T\frac{ds}{dR}.
```

A credible hot branch should have:

- smooth `xi_eff(R)`;
- no dependence on arbitrary clipping;
- no grid-scale entropy noise;
- a value following from the solution, not tuned to produce it.

`xi_eff` may vary with radius and can change sign.

## Test C: Global conservation

Compute residuals for mass, angular momentum, and energy.

For energy, define for example:

```math
\mathcal R_E
=
\frac{
L_{\rm heat}
-
L_{\rm rad}
-
L_{\rm wind}
-
L_{\rm adv,in}
-
dE_{\rm disk}/dt
}{
L_{\rm heat}
}.
```

Development target:

```text
|R_E| <= 1e-3--1e-2
```

with convergence as radial resolution is increased.

## Test D: Thermal perturbation test

Take a converged point on each candidate branch and perturb:

```text
T -> T * (1 +/- 0.01)
Sigma -> Sigma * (1 +/- 0.01)
```

Then integrate the time-dependent equations.

Expected behavior:

```text
stable branch   -> returns to equilibrium
unstable branch -> moves away
```

This is stronger than labeling a root stable from a local derivative while advection is imposed externally.

## Test E: Sonic-point regularity

Eventually, a physical hot solution should connect to an inner transonic flow.

Reject branches that:

- terminate before the inner boundary;
- require discontinuous `v_R`;
- cross the sonic point without regularity;
- imply unphysical velocities.

For the near-term Keplerian model, this test can be marked as future work.

## Test F: Resolution and boundary convergence

Repeat with variations in:

```text
N_R
R_in
R_out
Delta R_c
source profile
tidal torque taper
```

The branch location and participating mass should converge.

If the participating mass changes by an order of magnitude when changing source width or outer ghost-cell conditions, the result is probably numerical.

## Test G: Remove closures one by one

Run controlled experiments:

```text
Q_adv = 0
Q_wind = 0
Q_stream = 0
Lambda_tide = 0
```

Expected behavior:

- An advective hot branch should disappear or shift predictably when advection is removed.
- A wind-stabilized branch should respond systematically to `epsilon_w`.
- A branch that survives only because of a temperature ceiling or `H/R` cap is not physical.

## Test H: Closure validity

At each radius verify:

```text
tau_eff >= 1      where diffusion cooling is used
H/R <= 1          unless using an explicitly thick-flow model
|v_R| <= c_s      outside the sonic region
Mdot(R) decreases inward when winds operate
```

If `tau_eff < 1`, replace the diffusion cooling law with a thin/thick bridging formula or flag the solution as outside validity.

## Test I: Hysteresis and true limit cycles

A true limit cycle should show distinct transition surface densities:

```math
\Sigma_{\max}(R),
\qquad
\Sigma_{\min}(R).
```

The time-dependent disk should traverse these without manual switching.

Outputs to track:

```text
P_QPE
t_high
duty cycle
DeltaM_cycle
Mdot_peak
```

## Test J: Stress-law sensitivity

Repeat over:

```text
mu_stress
alpha
```

Because radiation-pressure instability is highly sensitive to the stress closure.

The generalized stress law is

```math
\tau_{R\phi}
=
\alpha P_{\rm gas}^{\mu}
P_{\rm tot}^{1-\mu}.
```

In the radiation-pressure-dominated regime, at fixed `Sigma`,

```math
Q^+ \propto T_c^{8-7\mu},
\qquad
Q^- \propto T_c^4.
```

The thermal instability condition is therefore

```math
8-7\mu>4,
```

or

```math
\mu<\frac{4}{7}.
```

Thus:

```text
mu = 0  -> strong classical instability
mu = 1  -> gas-pressure-like stress, thermally stable
```

This should be tested explicitly.

---

# 6. Recommended Implementation Order

Implement in this order.

## Step 1: Replace constant `xi`

Add routines to compute:

```text
e(rho,T)
TdsdR
Q_adv = Sigma v_R TdsdR
xi_eff
```

Keep wind off initially.

## Step 2: Build steady global nearly Keplerian slim disk

Before stream feeding, solve an isolated disk and recover the expected slim-disk behavior.

## Step 3: Add stream source and tidal truncation

Add:

```text
S_Sigma(R)
l_in(R)
Q_stream_plus(R)
Lambda_tide(R)
R_out = f_t R_H
```

Use Layer-1 outputs when available.

## Step 4: Time-dependent entropy equation

Evolve `Sigma(R,t)` and `T(R,t)` or an equivalent internal-energy variable.

Test branch stability by perturbation.

## Step 5: Add energy-limited wind

First use:

```text
l_w = l
epsilon_w = 0.1--1
```

Then later explore `lambda_w > 1`.

## Step 6: Synthetic light curve

Only after the branch physics is robust, map:

```text
Mdot_ISCO(t) -> L_X(t), T_col(t), R_ph(t)
```

using a wind/photosphere model.

---

# 7. Suggested Code Modules and Functions

The exact file names can be adjusted to match the existing repository. This section suggests a clean separation.

## `src/imri_qpe/thermo.py`

Functions:

```python
def gas_pressure(rho, T, mu):
    ...

def rad_pressure(T):
    ...

def total_pressure(rho, T, mu):
    ...

def specific_internal_energy(rho, T, mu, gamma_gas):
    ...

def vertical_structure(Sigma, T, R, M2, params):
    """Return H, rho, Pgas, Prad, Ptot, tau, etc."""
```

## `src/imri_qpe/advection.py`

Functions:

```python
def slope_limited_gradient(y, R):
    ...

def entropy_gradient(R, rho, T, P, e):
    """Return T ds/dR = de/dR - P/rho^2 d rho/dR."""
    ...

def q_advective(Sigma, vR, TdsdR):
    ...

def xi_eff(R, rho, P, TdsdR):
    ...
```

## `src/imri_qpe/radial_transport.py`

Functions:

```python
def stress_W(Sigma, nu, R, Omega):
    ...

def stream_source(R, Mdot_cap, Rc, dRc):
    ...

def tidal_torque_density(R, Rout, params):
    ...

def radial_velocity_from_angular_momentum(
    R, Sigma, l, W, Lambda_tide, S_Sigma, l_in, dotSigma_w, l_w
):
    ...

def mdot_from_vr(R, Sigma, vR):
    ...
```

## `src/imri_qpe/winds.py`

Functions:

```python
def q_edd_vertical(OmegaK, H, kappa):
    ...

def q_available(Qvisc, Qstream, Qtide, Qadv):
    ...

def wind_energy_per_mass(M2, R, v_inf=0.0, h_w=0.0, torque_work=0.0):
    ...

def energy_limited_wind(Qavail, Qedd, Ew, epsilon_w):
    """Return Qwind, Qrad, dotSigma_w."""
    ...
```

## `src/imri_qpe/global_minidisk.py`

Classes/functions:

```python
class MinidiskState:
    R: array
    Sigma: array
    T: array
    ...

def compute_rhs(state, params):
    """Return dSigma/dt and dU/dt or dT/dt."""
    ...

def steady_residual(state, params):
    ...

def integrate_time_dependent(state0, params, t_span):
    ...
```

---

# 8. Regression Tests to Add

Add tests under `tests/`.

## `test_entropy_advection.py`

- Check that `TdsdR = 0` for a constant-entropy analytic profile.
- Check sign convention for `Q_adv`.
- Check that `xi_eff` is recovered for a synthetic entropy profile with known `xi`.

## `test_mass_angular_momentum.py`

- Verify source normalization:
  ```text
  integral 2 pi R S_Sigma dR = Mdot_cap
  ```
- Verify `Mdot = -2 pi R Sigma vR`.
- Verify zero wind gives constant `Mdot` for a steady isolated disk.

## `test_winds.py`

- Verify wind activates only when `Qavail > Qedd`.
- Verify no double counting:
  ```text
  Qrad + Qwind = Qavail
  ```
- Verify inward `Mdot` decreases toward smaller radii when wind is on.

## `test_global_conservation.py`

- Check energy residual decreases with resolution for a simple steady case.
- Check mass conservation with stream source and wind sink.

## `test_stability.py`

- Perturb a stable branch and verify return.
- Perturb an unstable branch and verify departure.

---

# 9. Acceptance Criteria for the Next Milestone

The next implementation milestone should be considered successful if the code can do the following.

## Minimum milestone

- Compute `Q_adv` from the entropy gradient.
- Produce `xi_eff(R)` as a diagnostic.
- Solve a steady or quasi-steady global minidisk with smooth `Q_adv`.
- Reproduce the existing local hot-branch scale to within an order of magnitude without imposing constant `xi`.

## Strong milestone

- Demonstrate a stable hot/slim branch in the global stream-fed model.
- Show energy conservation residuals below `1e-2`.
- Show convergence of participating mass with radial resolution.
- Produce a time-dependent limit cycle without manual branch switching.

## Decisive milestone

Show that the model simultaneously reproduces:

```text
recurrence time ~ days
high-state duration ~ days
duty cycle ~ 0.3--0.6
burst mass ~ 1e-5--1e-4 Msun
burst rate ~ 1e-3--1e-2 Msun/yr or larger
```

and that this is robust to stress-law and wind-closure variations.

---

# 10. Physical Interpretation to Preserve

The model should not be described as:

```text
Gas mass inside the Hill sphere exceeds a value and all gas suddenly accretes.
```

The more precise interpretation is:

```text
Stream-fed gas accumulates in a tidally truncated circumsecondary disk. When
the disk surface density crosses an upper turning point, part of the minidisk
leaves its cool branch and enters a hot advective/wind-regulated branch. The
burst is the rapid draining of the participating annulus, not the instantaneous
collapse of the full Hill-sphere reservoir.
```

The critical mass is therefore:

```math
\Delta M_{\rm cyc}
=
2\pi
\int_{R_1}^{R_2}
[
\Sigma_{\max}(R)-\Sigma_{\min}(R)
]R\,dR.
```

The Hill-flow simulation supplies:

```text
Mdot_cap
j_in
R_c
Q_stream
alpha_shock
t_recyc
Lambda_tide
```

The 1D radiation/slim/wind minidisk model supplies:

```text
Sigma_max(R)
Sigma_min(R)
DeltaM_cycle
t_load
t_high
Mdot_burst
synthetic L_X(t)
```

---

# 11. References and Links for Background Folder

These are the papers most relevant to implementing and validating the upgrade. Download them into `literature/` or a similar folder.

## Slim disks, advection, and super-Eddington winds

1. Abramowicz et al. 1988, **Slim accretion disks**, ApJ 332, 646.  
   ADS: https://ui.adsabs.harvard.edu/abs/1988ApJ...332..646A/abstract

2. Abramowicz 2004, **Super-Eddington black hole accretion: Polish doughnuts and slim disks**.  
   arXiv: https://arxiv.org/abs/astro-ph/0411185

3. Szuszkiewicz & Miller / related time-dependent slim-disk work: **Non-linear evolution of thermally unstable slim accretion disks**.  
   arXiv: https://arxiv.org/abs/astro-ph/0107257

4. Poutanen et al. 2007, **Super-critically accreting stellar-mass black holes as ultraluminous X-ray sources**.  
   arXiv: https://arxiv.org/abs/astro-ph/0609274

5. Feng et al. 2019, **Global solution to a slim disk with radiation-driven outflows**.  
   arXiv: https://arxiv.org/abs/1909.07559

## Radiation-pressure stability and MHD cautions

6. Hirose, Krolik & Blaes 2009, **Radiation-Dominated Disks Are Thermally Stable**.  
   arXiv: https://arxiv.org/abs/0809.1708

7. Jiang, Stone & Davis 2013, **On the Thermal Stability of Radiation Dominated Accretion Disks**.  
   arXiv: https://arxiv.org/abs/1309.5646

8. Kaur, Stone & Gilbaum 2022, **Magnetically Dominated Disks in Tidal Disruption Events and Quasi-Periodic Eruptions**.  
   arXiv: https://arxiv.org/abs/2211.00704

## QPE disk-instability context

9. Pan et al. 2022, **A Disk Instability Model for the Quasi-periodic Eruptions of GSN 069**.  
   ADS: https://ui.adsabs.harvard.edu/abs/2022ApJ...928L..18P/abstract

10. Application/extension of the disk-instability model to QPE populations.  
    arXiv: https://arxiv.org/abs/2507.11100

## TDE disk evolution

11. Shen & Matzner 2013, **Evolution of Accretion Disks in Tidal Disruption Events**.  
    arXiv: https://arxiv.org/abs/1305.5570

12. TDE disk evolution with magnetically driven winds.  
    arXiv: https://arxiv.org/abs/2312.15415

## Hill-sphere, circumsecondary, and minidisk simulations

13. Tanigawa et al. 2012, **Three-dimensional flow around a protoplanet and circumplanetary-disk accretion**.  
    arXiv: https://arxiv.org/abs/1112.3706

14. Shock-driven accretion in circumplanetary disks.  
    arXiv: https://arxiv.org/abs/1609.09250

15. Bowen et al. 2017, **Quasi-Periodic Behavior of Mini-Disks in Binary Black Holes Approaching Merger**.  
    arXiv: https://arxiv.org/abs/1712.05451

16. Ryan & MacFadyen 2017, **Minidisks in Binary Black Hole Accretion**.  
    arXiv: https://arxiv.org/abs/1611.00341

17. Duffell et al. 2024, **The Santa Barbara Binary-Disk Code Comparison**.  
    arXiv: https://arxiv.org/abs/2402.13039

---

# 12. Short Prompt for Codex

Use this if you want a compact instruction to paste into a Codex session:

```text
Upgrade the IMRI QPE minidisk model from a local xi-parameterized hot branch to
a radially consistent slim/wind minidisk model.

First, add entropy-gradient advection:
    T ds/dR = de/dR - P/rho^2 d rho/dR
    Q_adv = Sigma v_R T ds/dR
    xi_eff = - R rho/P * T ds/dR
where e includes gas and radiation internal energy.

Second, compute v_R and Mdot from mass and angular-momentum conservation rather
than inferring Mdot from local heating. Add stream source S_Sigma(R) centered at
Rc and a tidal torque near Rout.

Third, add an energy-limited super-Eddington wind:
    Q_Edd,z = 2 c Omega_K^2 H / kappa
    Q_avail = Q_visc + Q_stream + Q_tide - Q_adv
    Q_wind = epsilon_w [Q_avail - Q_Edd,z]_+
    dotSigma_w = Q_wind / E_w
with baseline l_w = l.

Fourth, add tests: xi_eff recovery, source normalization, global conservation,
wind energy partition, stability perturbations, and resolution/boundary
convergence.

The goal is to determine whether a stable hot/slim branch and a time-dependent
limit cycle survive without imposing constant xi.
```
