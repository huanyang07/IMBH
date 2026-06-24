# Codex Implementation Plan: Repairing the Isolated Slim-Disk Benchmark Before Adding IMRI Physics

This markdown file is intended to be inserted into the Codex project as the next-step implementation brief.

It responds to the latest Codex outputs:

- `outputs/figures/global_slim_audit_hardened.png`
- `outputs/figures/isolated_slim_branch_continuation.png`
- `GPT_HANDOFF_NEXT_STEP.md`
- `src/imri_qpe/layer3_minidisk_1d/entropy_advection.py`
- `src/imri_qpe/layer3_minidisk_1d/isolated_slim_solver.py`
- `src/imri_qpe/layer3_minidisk_1d/global_slim.py`

The key current result is:

```text
The local xi = 0.3 hot branch does not survive its first global audit.
```

However, the isolated continuation failure is **not yet evidence that no physical slim branch exists**, because the current isolated solver already fails at a very thin point, around

```text
Mdot/Mdot_Edd = 0.03
max H/R ~= 8e-3
```

At such small `H/R`, radial-pressure corrections are of order `(H/R)^2 ~ 1e-4`, so missing radial momentum and sonic regularity cannot plausibly explain an order-unity energy residual. This points first to numerical/closure issues in the reduced solver.

The next goal is therefore:

```text
Repair and validate the reduced isolated nearly Keplerian slim/thin solver
before implementing a full transonic solver or re-adding stream/tide/wind.
```

Post-repair implementation result:

```text
This repair has now been implemented. The reduced isolated solver recovers the
thin disk and follows a smooth advective sequence through Mdot/Mdot_Edd ~= 10.
The old failure near Mdot/Mdot_Edd = 0.03 was a solver/convention artifact.
The remaining failure is physical/model-form: H/R exceeds 0.4 by a few
Mdot_Edd, so the nearly Keplerian reduction is not valid at the QPE target
Mdot/Mdot_Edd ~= 94. The next step is a transonic slim-disk solver with radial
momentum and sonic regularity.
```

---

# 1. Current Scientific Diagnosis

## 1.1 What has been established

The hardened global audit shows that the stitched local hot roots are not self-consistent when evaluated with radial entropy advection:

```text
Q_adv = Sigma v_R T ds/dR
xi_eff = - R rho/P * T ds/dR
```

The global audit finds roughly:

```text
median xi_eff        ~= 2.5--2.7
integrated Qadv/Qvisc ~= 6
energy L1 residual    ~= 5
```

This failure persists under resolution changes and derivative-stencil checks. Therefore, the local imposed-`xi` branch should be treated only as a local target, not as a validated global solution.

## 1.2 What had not yet been established before the repair

Before the repair, the isolated no-wind benchmark found only the lowest
thin-disk point cleanly. It failed at modest `Mdot/Mdot_Edd`, even while the
disk remained geometrically very thin.

This meant:

```text
The reduced solver was failing before it reached the regime where a
true slim-disk treatment is required.
```

That diagnosis motivated the repair described in this file. The post-repair
status is now: the reduced solver passes the thin/moderate-rate benchmark and
the next step is a transonic slim-disk solver.

---

# 2. Immediate Code-Level Fixes

## 2.1 Use a single energy-flux convention

The current code uses, in both `isolated_slim_solver.py` and `global_slim.py`,

```python
Q_visc = 9.0 / 8.0 * nu * Sigma * Omega**2
Q_rad  = 16.0 * sigma_SB * T**4 / (3.0 * kappa * Sigma)
```

The radiative term is the standard **two-face** diffusion cooling if

```text
tau = kappa Sigma / 2.
```

But

```text
Q_visc = 9/8 nu Sigma Omega^2
```

is the standard **one-face** Keplerian viscous dissipation.

Use one consistent convention everywhere. Recommended:

```python
Q_visc = 9.0 / 4.0 * nu * Sigma * Omega**2
Q_rad  = 16.0 * sigma_SB * T**4 / (3.0 * kappa * Sigma)
```

where both are vertically integrated over both disk faces.

Even better, compute viscous heating from the stress:

```math
Q_{\rm visc}^{+}
=
W\left(-R\frac{d\Omega}{dR}\right).
```

For a Keplerian disk,

```math
W = \frac{3}{2}\nu\Sigma\Omega,
\qquad
-R\frac{d\Omega}{dR} = \frac{3}{2}\Omega,
```

so

```math
Q_{\rm visc}^{+}
=
\frac{9}{4}\nu\Sigma\Omega^2.
```

### Required regression test

For a standard two-face Newtonian thin disk, verify that the code recovers

```math
Q_{\rm visc}^{+}
=
\frac{3GM_2\dot M}{4\pi R^3}
\left(1-\sqrt{\frac{R_0}{R}}\right).
```

If using one-face quantities instead, divide both sides by two consistently.

---

## 2.2 Make `Q_rad_diffusion` the actual no-wind radiative term

In `global_slim.py`, the code computes

```python
Q_rad_diffusion = 16.0 * SIGMA_SB * T**4 / (3.0 * params.kappa * Sigma)
```

but the energy residual is evaluated using `Q_rad_limited` returned by the wind helper:

```python
Q_wind, Q_rad_limited, dotSigma_w = energy_limited_wind(...)
energy_residual = Q_visc - Q_rad_limited - Q_adv - Q_wind
```

This is not appropriate for the no-wind isolated benchmark. When wind is off, use

```python
Q_wind = 0
Q_rad = Q_rad_diffusion
energy_residual = Q_visc - Q_rad - Q_adv
```

Do not use the wind partitioning routine in no-wind benchmarks. Otherwise the residual can become partly tautological if the wind helper partitions available energy into radiation and wind.

### Implementation instruction

Add a flag or separate function:

```python
def energy_residual_no_wind(Q_visc, Q_rad_diffusion, Q_adv):
    return Q_visc - Q_rad_diffusion - Q_adv
```

Use this for:

```text
isolated no-wind solver
isolated no-wind audit
thin-disk validation
```

Only reintroduce `energy_limited_wind` after the isolated no-wind solver passes.

---

## 2.3 Verify the physical inner torque constant

The current isolated solver defines the inner angular-momentum constant as:

```python
if l_in is None:
    return keplerian_specific_angular_momentum(M2_g, R_in)
```

If `R_in` is the first radius of the plotted grid, for example

```text
R_min = 0.06 R_H,
```

then the code is imposing zero torque at the numerical grid boundary, not at the physical ISCO.

For the fiducial system,

```text
R_H ~= 747 r_g,2
0.06 R_H ~= 45 r_g,2
```

So a zero-torque condition at `0.06 R_H` is much too far out.

### Required fix

For the reduced Keplerian benchmark, use:

```python
R_ISCO = 6.0 * G * M2 / c**2   # Schwarzschild baseline
l0 = sqrt(G * M2 * R_ISCO)
```

Then use

```math
W_{\rm required}(R)
=
\frac{\dot M(l-l_0)}{2\pi R^2}.
```

The numerical grid may start at a radius larger than `R_ISCO`, but the integration constant should still correspond to the physical inner torque radius, not the first cell.

### Required diagnostic plot

At each failed cell, plot:

```math
W_{\rm model}(\Sigma; T, R)
```

against

```math
W_{\rm required}(R)
=
\frac{\dot M(l-l_0)}{2\pi R^2}.
```

This distinguishes:

1. no physical root at the chosen `T`;
2. too narrow `Sigma` bounds;
3. wrong torque constant;
4. branch jumping;
5. numerical bracketing failure.

---

## 2.4 Replace nested `Sigma(T)` relaxation with a simultaneous root solve

The current isolated solver:

1. fixes `T(R)`;
2. solves the angular-momentum equation locally for `Sigma(R)`;
3. updates `T(R)` by a damped residual step.

This is fragile because the thermal residual is not monotonic in `T` on radiation-pressure or advective branches. A physical root may exist even though the relaxation update moves away from it.

### New unknown vector

Use logarithmic variables:

```math
z =
(\ln\Sigma_1,\ldots,\ln\Sigma_N,
 \ln T_1,\ldots,\ln T_N).
```

This enforces positivity.

### Residuals

Use two residuals per radial cell.

Angular-momentum residual:

```math
F_{J,i}
=
\ln\left[
\frac{2\pi R_i^2 W_i}
{\dot M(l_i-l_0)}
\right].
```

Energy residual:

```math
F_{E,i}
=
\frac{
Q_{{\rm visc},i}
-
Q_{{\rm rad},i}
-
Q_{{\rm adv},i}
}{
|Q_{{\rm visc},i}|
+
|Q_{{\rm rad},i}|
+
|Q_{{\rm adv},i}|
+\epsilon
}.
```

Solve:

```math
F(z)=0.
```

### Numerical method

Use one of:

```text
scipy.optimize.least_squares
scipy.optimize.root
Newton-Krylov
damped sparse Newton
```

Recommended first implementation:

```python
scipy.optimize.least_squares(
    residual_vector,
    z0,
    loss="linear",
    x_scale="jac",
    max_nfev=...
)
```

Because `Q_adv` couples neighboring cells through radial derivatives, the Jacobian is sparse and nearly banded, but a dense finite-difference Jacobian is acceptable for small grids during development.

### Why this matters

Do **not** use physical thermal relaxation as an equilibrium finder. A physical thermal relaxation should not converge to unstable equilibrium branches. A mathematical root finder should be able to find them.

---

## 2.5 Fix continuation logic

The current continuation keeps a profile whenever:

```python
result.profile is not None
```

even if `result.converged` is false. This contaminates all later continuation steps.

### Required fix

Only update the continuation seed when:

```python
result.converged is True
```

If a step fails:

1. return to the last converged solution;
2. halve `Delta ln Mdot`;
3. retry;
4. stop only when `Delta ln Mdot` falls below a minimum threshold.

Recommended initial step:

```text
Delta ln Mdot = 0.03--0.1
```

not jumps like:

```text
0.01, 0.03, 0.1, 0.3, ...
```

### Remove unconditional line-search acceptance

The current code accepts the smallest line-search step even if it worsens the residual:

```python
if trial_metrics.L1 <= current_score or factor == 0.0625:
```

Remove the second clause. A line search should fail if all attempted steps worsen the objective.

Use:

```python
if trial_metrics.L1 < current_score:
    accept
else:
    continue
```

or a proper Armijo condition.

---

# 3. Hardened Audit Tests to Keep

The current entropy-advection module is useful and should be retained.

## 3.1 Entropy-gradient cross-checks

Two formulae should agree:

First-law form:

```math
T\frac{ds}{dR}
=
\frac{de}{dR}
-
\frac{P}{\rho^2}\frac{d\rho}{dR}.
```

Log-gradient form:

```math
T\frac{ds}{dR}
=
{\cal R}T
\left[
\frac{1}{\gamma-1}\frac{d\ln T}{dR}
-
\frac{d\ln\rho}{dR}
\right]
+
\frac{4a_{\rm r}T^4}{\rho}
\left[
\frac{d\ln T}{dR}
-
\frac{1}{3}\frac{d\ln\rho}{dR}
\right].
```

Manufactured tests:

```text
gas-isentropic:       T ∝ rho^(gamma-1)
radiation-isentropic: T ∝ rho^(1/3)
```

The corresponding entropy-gradient contribution should vanish.

## 3.2 Residual norms

For each global or isolated solution, compute pointwise residual:

```math
r_E(R)=
\frac{
Q^+-Q_{\rm rad}-Q_{\rm adv}-Q_{\rm wind}
}{
|Q^+|+Q_{\rm rad}+|Q_{\rm adv}|+Q_{\rm wind}+\epsilon
}.
```

Report:

```math
\max_R |r_E|,
```

```math
{\cal R}_{E,1}
=
\frac{
\int 2\pi R |Q^+-Q_{\rm rad}-Q_{\rm adv}-Q_{\rm wind}|\,dR
}{
\int 2\pi R |Q^+|\,dR
},
```

and signed residual separately.

Do not rely only on signed integrated residuals.

## 3.3 Boundary-zone flags

Report diagnostics with and without the first/last two or three cells.

Track whether spikes in `xi_eff`, `Q_adv/Qvisc`, or residuals are boundary-localized.

---

# 4. Minimal Reduced Isolated Solver to Implement Next

Before implementing a full transonic slim disk, build a corrected reduced nearly Keplerian solver.

## 4.1 Assumptions

For this benchmark only:

```text
no stream source
no tidal torque
no wind
Omega = Omega_K
constant imposed Mdot
physical l0 near ISCO
```

Unknowns:

```text
Sigma(R), T(R)
```

Equations:

```text
angular momentum closure
energy closure
```

## 4.2 Equations

Keplerian angular momentum:

```math
l(R)=\sqrt{GM_2R}.
```

Physical inner torque constant:

```math
l_0\simeq l_K(R_{\rm ISCO}).
```

Stress:

```math
W=\frac{3}{2}\nu\Sigma\Omega_K.
```

Viscosity:

```math
\nu=\alpha H^2\Omega_K.
```

Angular-momentum equation:

```math
\dot M(l-l_0)=2\pi R^2 W.
```

Radial velocity:

```math
v_R=-\frac{\dot M}{2\pi R\Sigma}.
```

Entropy advection:

```math
Q_{\rm adv}
=
\Sigma v_R T\frac{ds}{dR}.
```

Energy equation:

```math
Q_{\rm visc}^{+}
=
Q_{\rm rad}^{-}
+
Q_{\rm adv}.
```

with two-face convention:

```math
Q_{\rm visc}^{+}
=
\frac{9}{4}\nu\Sigma\Omega_K^2,
```

```math
Q_{\rm rad}^{-}
=
\frac{16\sigma_{\rm SB}T_c^4}{3\kappa\Sigma}.
```

## 4.3 Validation target

At low `Mdot`, where advection is negligible, recover the analytic thin disk:

```math
\nu\Sigma
=
\frac{\dot M}{3\pi}
\left(1-\sqrt{\frac{R_0}{R}}\right)
```

and

```math
Q_{\rm visc}^{+}
\simeq
Q_{\rm rad}^{-}.
```

Acceptance:

```text
energy L1 residual < 1e-3--1e-2
thin-disk flux agreement < 1 percent away from boundaries
constant Mdot to numerical precision
smooth Sigma, T, xi_eff
```

Only after this succeeds should we interpret failures near high `Mdot` physically.

---

# 5. When the Reduced Solver Is No Longer Enough

Use the following decision rule:

```text
H/R < 0.2:
    reduced Keplerian solver is acceptable for development

H/R = 0.2--0.4:
    reduced solver is approximate; begin checking radial momentum effects

H/R > 0.4:
    implement full radial momentum and sonic regularity
```

The current failure at `H/R ~ 8e-3` is not due to missing radial momentum.

However, the eventual QPE target around

```text
Mdot/Mdot_Edd ~ 94
```

will likely require radial momentum, non-Keplerian rotation, and winds.

---

# 6. Minimum Faithful Transonic Slim-Disk Solver

After the corrected reduced benchmark works, implement an isolated one-temperature slim disk. Keep stream, tides, and wind off.

## 6.1 Continuity

```math
\dot M=-2\pi R\Sigma v_R.
```

## 6.2 Radial momentum

```math
v_R\frac{dv_R}{dR}
=
R(\Omega^2-\Omega_K^2)
-
\frac{1}{\Sigma}\frac{d\Pi}{dR}.
```

Here:

```math
\Pi=\int P\,dz.
```

A first approximation may use:

```math
\Pi\simeq 2HP_c.
```

## 6.3 Angular momentum

```math
W_{R\phi}
=
-\nu\Sigma R\frac{d\Omega}{dR}.
```

```math
\dot M(l-l_0)=2\pi R^2W_{R\phi}.
```

This gives:

```math
\frac{d\Omega}{dR}
=
-\frac{\dot M(l-l_0)}
{2\pi\nu\Sigma R^3}.
```

## 6.4 Energy

```math
Q_{\rm visc}^{+}
=
Q_{\rm rad}^{-}
+
Q_{\rm adv}.
```

with

```math
Q_{\rm visc}^{+}
=
W_{R\phi}\left(-R\frac{d\Omega}{dR}\right),
```

```math
Q_{\rm adv}
=
\Sigma v_R T\frac{ds}{dR}.
```

and

```math
T\frac{ds}{dR}
=
\frac{de}{dR}
-
\frac{P}{\rho^2}\frac{d\rho}{dR}.
```

## 6.5 Vertical closure

Use:

```math
H\simeq\frac{c_s}{\Omega_\perp},
```

```math
\Sigma=2\rho H,
```

```math
P=P_{\rm gas}+P_{\rm rad}.
```

The unknown functions can be:

```text
ln(-v_R), ln T, Omega
```

with `Sigma` obtained from continuity, and with `l0` as an unknown scalar eigenvalue.

---

# 7. Boundary Conditions for the Transonic Solver

## 7.1 Outer boundary

Choose an outer radius well outside the advective region, if possible:

```text
R_out ~ 1e3 r_g,2
```

or as large as allowed by the minidisk truncation in later IMRI runs.

Specify:

```text
Mdot = Mdot_out
Omega ~= Omega_K
T or entropy from the standard thin-disk solution
```

Do not overconstrain by imposing both `T` and `Sigma` if equation count does not permit it.

## 7.2 Inner boundary

Do not impose both:

```text
zero torque at first radial cell
sonic regularity
```

For a faithful slim disk:

```text
the solution must pass regularly through a sonic point
l0 is an eigenvalue
```

A practical first implementation:

```text
use a pseudo-Newtonian potential
place inner boundary near/inside the sonic region
solve for l0 as an unknown parameter
require smooth critical-point passage
```

For the reduced benchmark, use fixed `l0 ~= l_K(R_ISCO)`.

---

# 8. Numerical Method Recommendation

## 8.1 Reduced solver

Use sparse nonlinear least squares for:

```text
ln Sigma_i, ln T_i
```

with residuals:

```text
F_J,i = angular-momentum residual
F_E,i = energy residual
```

Use adaptive continuation in `ln Mdot`.

## 8.2 Full transonic solver

Use one of:

```text
scipy.integrate.solve_bvp with l0 as an unknown parameter
Henyey/Newton relaxation on a logarithmic radial grid
Newton-Krylov relaxation
```

Direct shooting may be fragile because of the sonic critical point.

## 8.3 Continuation

Use:

```text
Delta ln Mdot = 0.03--0.1
```

and only continue from converged profiles.

Near folds, use pseudo-arclength continuation rather than using `Mdot` itself as the only continuation coordinate.

---

# 9. Validation Sequence

Before returning to the IMRI minidisk, require:

## 9.1 Thin-disk test

Recover:

```math
\nu\Sigma
=
\frac{\dot M}{3\pi}
\left(1-\sqrt{\frac{R_0}{R}}\right).
```

## 9.2 Flux-convention test

Verify:

```math
Q_{\rm visc}^{+}=Q_{\rm rad}^{-}
```

for a standard thin-disk solution.

## 9.3 Constant-Mdot test

For no source and no wind:

```math
\frac{d\dot M}{dR}=0.
```

## 9.4 Entropy-gradient test

Keep the manufactured gas-isentropic and radiation-isentropic tests.

## 9.5 Global energy test

Require:

```math
\frac{
\int |Q^+-Q^--Q_{\rm adv}|\,2\pi R\,dR
}{
\int Q^+\,2\pi R\,dR
}
<10^{-3}--10^{-2}.
```

## 9.6 Sonic test

For the full transonic solver, Mach number should cross unity smoothly once, with no discontinuity in:

```text
v_R
T
Omega
Sigma
```

## 9.7 Published-profile test

Reproduce at least one published slim-disk sequence around:

```text
Mdot/Mdot_Edd ~ 1
Mdot/Mdot_Edd ~ 10
```

before attempting the QPE target.

---

# 10. Delay Stream, Tides, and Wind

Do not add these yet:

```text
stream source
tidal torque
wind
time-dependent limit cycle
```

until the isolated no-wind solver passes.

Recommended order:

```text
1. Thin-disk benchmark
2. Corrected isolated reduced solver
3. Isolated transonic slim disk
4. Stream source + tidal truncation
5. Energy-limited wind
6. Time-dependent limit cycle
7. Synthetic light curve
```

Adding wind now would add an adjustable energy sink and could hide errors in the basic disk equations.

---

# 11. What If No No-Wind Hot Branch Exists?

If a validated isolated solver still finds no steady no-wind hot branch, this does **not** automatically kill the QPE model.

Possible outcomes:

```text
A. A steady hot branch exists only with radial momentum and sonic regularity.
B. A steady hot branch exists only with wind regulation.
C. No steady branch exists, but time-dependent thermal runaway/draining cycles exist.
D. The minidisk limit-cycle model is disfavored under the tested closures.
```

But do not attempt B, C, or D until the isolated thin/slim benchmark is validated.

---

# 12. Go/No-Go Criteria After the Next Implementation

## Continue if

```text
- the corrected reduced solver recovers the thin disk;
- a global advective branch is found or fails only after H/R becomes large;
- xi_eff(R) is smooth and converged;
- energy L1 residual is below 1e-2;
- continuation is robust to resolution and step size;
- no artificial H/R, xi_eff, or Qadv clipping is needed.
```

## Move to full transonic solver if

```text
- reduced solver works at low/moderate Mdot;
- branch approaches H/R > 0.3--0.4;
- Keplerian assumptions become questionable.
```

## Treat the limit-cycle explanation as fragile if

```text
- the hot branch exists only because of imposed caps or clipping;
- the result depends strongly on boundary placement or source width;
- the solution requires unphysical xi_eff or H/R;
- strong winds remove the reservoir before it can burst.
```

## Disfavor this specific model if

```text
- validated isolated slim-disk solver works, but no stream-fed/tidally truncated
  disk over plausible parameters has hysteresis or bursts;
- time-dependent calculations always settle smoothly without eruptions;
- participating mass or burst duration is orders of magnitude from QPE requirements.
```

---

# 13. Compact Prompt for Codex

Use this as the next Codex prompt:

```text
Repair the isolated slim-disk benchmark before adding more IMRI physics.

1. Use one consistent two-face energy convention:
       Qvisc = (9/4) nu Sigma Omega_K^2
       Qrad  = 16 sigma_SB T^4 / (3 kappa Sigma)
   or compute Qvisc = W(-R dOmega/dR).

2. In no-wind mode, use Qrad_diffusion directly:
       residual = Qvisc - Qrad_diffusion - Qadv
   Do not call the wind partitioning routine.

3. Set the angular-momentum integration constant l0 from the physical ISCO,
   not from the first grid cell:
       l0 = sqrt(G M2 R_ISCO).

4. Replace nested Sigma(T) relaxation with a simultaneous root solve in
   lnSigma_i and lnT_i, using angular-momentum residuals and energy residuals.

5. Fix continuation:
   update the seed only after converged=True, remove unconditional acceptance
   of the smallest line-search step, and use adaptive Delta ln Mdot.

6. Add diagnostics:
   W_model(Sigma;T,R) vs W_required at failed cells,
   thin-disk flux test,
   residual L1/max/signed norms,
   entropy-gradient cross-checks,
   boundary-zone flags.

7. First recover the standard thin disk from Mdot/Mdot_Edd ~ 1e-3 to 0.1.
   Only then continue toward the advective regime. If H/R exceeds 0.3--0.4,
   implement the full transonic slim-disk solver with radial momentum and
   sonic regularity before adding stream/tide/wind.
```

---

# 14. Notes on Eddington-Rate Convention

Always print the convention used for `Mdot_Edd`.

Recommended:

```math
\dot M_{\rm Edd}
=
\frac{L_{\rm Edd}}{\eta c^2},
\qquad
\eta=0.1.
```

Then:

```math
L_{\rm Edd}=\frac{4\pi GMc}{\kappa}.
```

Different slim-disk papers sometimes differ by factors of order ten depending on whether `eta` is included.

Every figure caption and table involving `Mdot/Mdot_Edd` should state the convention.
