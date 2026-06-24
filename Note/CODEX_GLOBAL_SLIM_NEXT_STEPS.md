# Codex Implementation Addendum: Global Slim/Wind Branch Audit and Next Steps

This document is a Codex-ready implementation plan for the next development sprint of the **IMRI QPE minidisk limit-cycle model**.

It responds to the current audit result:

```text
The imposed local xi = 0.3 hot branch does not yet survive as a faithful global advective/wind hot branch.
```

This is a useful negative result, not a project failure. It means the local, parameterized advective hot branch must be replaced by a radially self-consistent slim/wind calculation.

The key interpretation is:

```text
The local xi-parameterized hot branch failed its first global audit, but the full physical model has not yet failed.
```

The next decisive calculation is **not** another scan in imposed `xi`. It is:

```text
1. strengthen the global audit,
2. build and validate an isolated global slim-disk solver,
3. add stream feeding and tidal truncation,
4. add wind only after the advective branch is under control,
5. then perform time-dependent limit-cycle calculations.
```

---

## 0. Current Situation to Preserve in the README

Current audit outputs show:

```text
Local hot roots evaluated globally:
    residual                  ~= -4.8
    median xi_eff             ~= 2.2
    integrated Qadv/Qvisc     ~= 5.8
    max H/R                   ~= 1.5

Fixed-Sigma energy-relaxed profile:
    residual                  ~= 7e-19
    median xi_eff             ~= 2.7
    integrated Qadv/Qvisc     ~= -0.09
    one outward-flow radial zone
    max H/R                   ~= 0.67
```

Interpretation:

```text
The stitched local roots are not a self-consistent global solution.
The fixed-Sigma relaxation closes the energy equation by construction, but it is not a full steady disk solution because it does not simultaneously solve mass conservation, angular-momentum conservation, radial momentum, and inner regularity.
```

Do **not** describe the energy-relaxed profile as a validated hot branch.

Use this wording:

```text
The current global slim/wind evaluator is a consistency audit. It shows that the imposed local xi branch is not self-consistent under the present radial model. The next step is a coupled global solver, starting from an isolated slim-disk benchmark.
```

Update after Sprint B:

```text
The isolated no-wind nearly Keplerian benchmark has now been implemented.
It has also been repaired: the code now uses a consistent two-face energy
convention, a physical ISCO angular-momentum constant, direct diffusion
cooling in no-wind cases, and a simultaneous solve for Sigma(R) and T(R).
The repaired solver recovers the thin disk and follows a smooth reduced
advective sequence through Mdot/Mdot_Edd ~= 10. The output files are:

outputs/figures/isolated_slim_branch_continuation.png
outputs/tables/isolated_slim_branch_summary.md
```

Interpretation:

```text
The isolated reduced solver now passes the thin/moderate-rate benchmark, but
it exits its physical validity range before the QPE target: H/R exceeds 0.4 by
a few Mdot_Edd, while the target is Mdot/Mdot_Edd ~= 94. Before adding stream,
tide, or wind, the next physical step is a transonic slim-disk solver with
radial momentum and sonic regularity.
```

---

# 1. Strengthen the Existing Global Audit

The first task is to improve diagnostics before changing the physical model. The current failure might be partly physical and partly numerical, especially near the outer boundary.

## 1.1 Add local residual diagnostics

Define the pointwise normalized energy residual:

```math
r_E(R)=
\frac{
Q^+ - Q_{\rm rad} - Q_{\rm adv} - Q_{\rm wind}
}{
|Q^+| + Q_{\rm rad} + |Q_{\rm adv}| + Q_{\rm wind} + \epsilon
}.
```

Here:

```text
Q+ = Q_visc + Q_stream + Q_tide
```

and `epsilon` is a small floor to avoid division by zero.

Report:

```text
signed global residual
L1 residual
L2 residual
max absolute residual
max absolute residual excluding boundary zones
location of max residual
```

Define integrated residuals:

```math
\mathcal R_{E,1}
=
\frac{
\int 2\pi R |Q^+ - Q_{\rm rad}-Q_{\rm adv}-Q_{\rm wind}|\,dR
}{
\int 2\pi R |Q^+|\,dR
},
```

```math
\mathcal R_{E,{\rm signed}}
=
\frac{
\int 2\pi R (Q^+ - Q_{\rm rad}-Q_{\rm adv}-Q_{\rm wind})\,dR
}{
\int 2\pi R |Q^+|\,dR
}.
```

Do not rely only on the signed residual, because positive and negative errors can cancel.

### Suggested implementation

Add to:

```text
src/imri_qpe/layer3_minidisk_1d/global_slim.py
```

or create:

```text
src/imri_qpe/layer3_minidisk_1d/audit_metrics.py
```

Suggested functions:

```python
def pointwise_energy_residual(Qplus, Qrad, Qadv, Qwind, floor=1e-300):
    denom = np.abs(Qplus) + Qrad + np.abs(Qadv) + Qwind + floor
    return (Qplus - Qrad - Qadv - Qwind) / denom


def integrated_energy_residuals(R, Qplus, Qrad, Qadv, Qwind, floor=1e-300):
    dA = 2.0 * np.pi * R
    raw = Qplus - Qrad - Qadv - Qwind
    norm = np.trapz(dA * np.abs(Qplus), R) + floor
    return {
        "signed": np.trapz(dA * raw, R) / norm,
        "L1": np.trapz(dA * np.abs(raw), R) / norm,
        "L2": np.sqrt(np.trapz(dA * raw**2, R)) / (np.sqrt(np.trapz(dA * Qplus**2, R)) + floor),
        "max_abs": np.max(np.abs(raw) / (np.abs(Qplus) + Qrad + np.abs(Qadv) + Qwind + floor)),
    }
```

---

## 1.2 Recompute entropy gradients in two independent ways

Current expression:

```math
T\frac{ds}{dR}
=
\frac{de}{dR}
-
\frac{P}{\rho^2}\frac{d\rho}{dR}.
```

Add an independent gas+radiation entropy-gradient formula:

```math
\begin{split}
T\frac{ds}{dR}
={}&
\mathcal R T
\left[
\frac{1}{\gamma-1}\frac{d\ln T}{dR}
-
\frac{d\ln\rho}{dR}
\right]
\\
&+
\frac{4a_{\rm r}T^4}{\rho}
\left[
\frac{d\ln T}{dR}
-
\frac{1}{3}\frac{d\ln\rho}{dR}
\right],
\end{split}
```

where:

```math
\mathcal R = \frac{k_B}{\mu m_p}.
```

These two methods should agree cell by cell.

### Suggested function

Add to:

```text
src/imri_qpe/layer3_minidisk_1d/entropy_advection.py
```

```python
def entropy_gradient_log_formula(R, rho, T, mu, gamma_gas):
    """Return T ds/dR for a gas+radiation mixture using log gradients."""
    Rgas = k_B / (mu * m_p)
    dlnT_dR = slope_limited_gradient(np.log(T), R)
    dlnrho_dR = slope_limited_gradient(np.log(rho), R)

    gas = Rgas * T * ((1.0 / (gamma_gas - 1.0)) * dlnT_dR - dlnrho_dR)
    rad = (4.0 * a_rad * T**4 / rho) * (dlnT_dR - (1.0 / 3.0) * dlnrho_dR)
    return gas + rad
```

Add a comparison diagnostic:

```python
def entropy_gradient_consistency_error(TdsdR_first_law, TdsdR_log):
    denom = np.maximum(np.abs(TdsdR_first_law) + np.abs(TdsdR_log), tiny)
    return np.abs(TdsdR_first_law - TdsdR_log) / denom
```

---

## 1.3 Manufactured entropy-gradient tests

Add tests to:

```text
tests/test_entropy_advection.py
```

### Test 1: gas-pressure isentropic profile

For a gas-pressure adiabatic profile:

```math
T \propto \rho^{\gamma-1},
```

with negligible radiation pressure, the gas entropy gradient should vanish.

Construct synthetic arrays:

```python
rho = rho0 * (R/R0)**p
T = T0 * (rho/rho0)**(gamma - 1)
```

Use a sufficiently low `T0` or turn off radiation contribution for this manufactured test.

Expected:

```text
max |TdsdR| / characteristic_energy_gradient < tolerance
```

### Test 2: radiation-entropy isentropic profile

For radiation entropy per unit mass roughly constant:

```math
T \propto \rho^{1/3}.
```

Construct:

```python
rho = rho0 * (R/R0)**p
T = T0 * (rho/rho0)**(1/3)
```

Expected radiation entropy contribution approximately vanishes.

### Test 3: known xi synthetic profile

Construct synthetic entropy gradient such that:

```math
\xi_{\rm target}=-\frac{R\rho}{P}T\frac{ds}{dR}
```

is constant. Verify recovered:

```text
median xi_eff ~= xi_target
```

---

## 1.4 Resolution and stencil convergence

Run audit at:

```text
N_R
2 N_R
4 N_R
```

Compare:

```text
centered gradients
slope-limited gradients
higher-order gradients if available
```

Report:

```text
median xi_eff excluding boundary cells
max xi_eff excluding boundary cells
integrated Qadv/Qvisc
signed residual
L1 residual
outermost-cell behavior
```

Do not use the first and last 2--3 active cells for headline max values, but continue plotting them.

Interpretation:

```text
If the xi_eff spike moves outward and grows with refinement, it is likely a boundary artifact.
If it converges at a fixed physical radius, it may be physical.
```

---

# 2. Understand the Outward-Flow Zone

The current fixed-Sigma relaxed profile has one outward-flow radial zone. This is not automatically fatal.

With sign convention:

```math
\dot M=-2\pi R\Sigma v_R>0
```

steady mass conservation gives:

```math
\frac{d\dot M}{dR}=2\pi R(\dot\Sigma_w-S_\Sigma).
```

Equivalently:

```math
\dot M(R_2)-\dot M(R_1)
=
2\pi\int_{R_1}^{R_2}R(\dot\Sigma_w-S_\Sigma)\,dR.
```

Add a numerical budget check:

```python
def mdot_continuity_residual(R, Mdot, S_Sigma, dotSigma_w):
    dMdot_dR = slope_limited_gradient(Mdot, R)
    rhs = 2.0 * np.pi * R * (dotSigma_w - S_Sigma)
    return dMdot_dR - rhs
```

Interpretation:

```text
Outward flow outside the stream deposition radius can be physical in a stream-fed truncated disk, because the outer disk may act like a decretion region carrying angular momentum to the tidal boundary.
```

But in the current audit:

```text
If there is no localized stream source and no consistent tidal torque, an outward zone only near the outer boundary is more likely a boundary artifact.
```

---

# 3. Do Not Add More Wind Complexity Yet

Do **not** attempt to fix the current inconsistency by adding more adjustable wind physics.

Recommended order:

```text
1. isolated advective disk
2. stream-fed truncated disk
3. wind-regulated disk
4. time-dependent limit cycle
```

Reason:

```text
The immediate problem is excessive and sign-changing radial advection. Wind can hide this by adding another adjustable energy sink. First establish that the advective branch is radially consistent without wind.
```

For now, wind should remain off in the isolated benchmark:

```text
dotSigma_w = 0
Q_wind     = 0
l_w        unused
```

---

# 4. Build an Isolated Global Slim-Disk Benchmark

This is the next decisive calculation.

Turn off:

```text
S_Sigma       = 0
Lambda_tide   = 0
dotSigma_w    = 0
Q_stream      = 0
Q_tide        = 0
```

Specify a constant external accretion rate:

```math
\dot M(R)=\dot M_0.
```

Solve a global isolated disk satisfying:

```math
Q_{\rm visc}=Q_{\rm rad}+Q_{\rm adv}.
```

Do **not** hold `Sigma(R)` fixed and relax only `T(R)`. That only gives an energy-balanced profile, not a disk solution.

## 4.1 Unknowns

Start with nearly Keplerian rotation and solve for:

```text
Sigma(R)
T(R)
v_R(R) or torque W(R)
```

A practical choice:

```text
x_i = ln Sigma_i
y_i = ln T_i
```

so positivity is automatic.

## 4.2 Equations for nearly Keplerian isolated disk

Use:

```math
\dot M=-2\pi R\Sigma v_R = \dot M_0.
```

Then:

```math
v_R=-\frac{\dot M_0}{2\pi R\Sigma}.
```

For angular momentum, use the standard steady disk relation with inner boundary constant:

```math
\dot M_0(l-l_{\rm in}) = 2\pi R^2 W.
```

or equivalently solve the differential angular-momentum equation.

For Keplerian rotation:

```math
W=-\nu\Sigma R\frac{d\Omega}{dR}
=\frac{3}{2}\nu\Sigma\Omega.
```

The relation above becomes a closure between `Sigma`, `T`, and `nu`.

Energy equation:

```math
Q_{\rm visc} = Q_{\rm rad} + Q_{\rm adv}.
```

with:

```math
Q_{\rm adv} = \Sigma v_R T\frac{ds}{dR}.
```

## 4.3 Solver strategy

Use nonlinear root-finding rather than local relaxation.

Suggested approach:

```text
1. construct radial grid in log R
2. set initial thin-disk profiles for Sigma and T
3. define residual vector consisting of:
   - angular momentum residual in each cell
   - energy residual in each cell
   - inner boundary condition
   - outer boundary condition
4. solve using damped Newton / scipy.optimize.root / least_squares if scipy is allowed
5. otherwise implement simple Newton-Krylov or pseudo-time relaxation
```

If external dependencies are restricted, a pseudo-time method is acceptable, but keep it explicitly marked as such.

## 4.4 Branch continuation

Do not jump directly to the target high accretion rate.

Start at low accretion rate where the thin solution is easy:

```text
Mdot = 0.01 Mdot_Edd,2
```

Then increase gradually:

```math
\dot M_{n+1}=(1+\delta)\dot M_n,
```

with:

```text
delta = 0.05--0.2
```

Use each converged solution as the initial guess for the next one.

This is much more reliable than constructing a radial profile from independent local roots.

## 4.5 Isolated benchmark success criteria

The isolated global slim-disk benchmark should show:

```text
Mdot(R) nearly constant
smooth Sigma(R)
smooth T(R)
smooth H/R
smooth xi_eff(R)
energy L1 residual < 1e-2
0 <= integrated Qadv/Qvisc <= 1 for advective high state
no large boundary spikes in xi_eff after convergence
```

Local negative `Q_adv` can be allowed, but the main hot region should have net positive advective cooling.

---

# 5. Decide When Radial Momentum Is Required

The current relaxed profile has:

```text
max H/R ~= 0.67
```

This is too thick for a purely Keplerian vertically averaged model to be fully trustworthy.

Use this rule:

```text
H/R <= 0.3:
    Keplerian approximation is acceptable for development.

H/R >= 0.4--0.5:
    radial momentum and sonic regularity should be promoted to the next task.
```

The radial momentum equation is:

```math
v_R\frac{dv_R}{dR}
=
R(\Omega^2-\Omega_K^2)
-
\frac{1}{\Sigma}\frac{d\Pi}{dR}
+
f_{R,{\rm tide}},
```

where:

```math
\Pi=\int P\,dz.
```

A physical hot slim solution should pass through a sonic point. If the hot state requires `H/R > 0.5`, a future solver must include:

```text
radial momentum
non-Keplerian angular velocity
sonic point condition
regularity/eigenvalue condition
```

Do not claim a high-`H/R` Keplerian energy-balanced profile is a physical slim branch.

---

# 6. Add Stream Source and Tidal Truncation After the Isolated Benchmark

Only after the isolated benchmark passes should the model add the IMRI-specific structure.

## 6.1 Stream source

Add a Gaussian stream source:

```math
S_\Sigma(R)=
\frac{\dot M_{\rm cap}}
{2\pi R_c\sqrt{2\pi}\Delta R_c}
\exp\left[-\frac{(R-R_c)^2}{2\Delta R_c^2}\right].
```

Normalize numerically:

```math
\int 2\pi R S_\Sigma\,dR=\dot M_{\rm cap}.
```

Add angular momentum through:

```math
S_\Sigma(l_{\rm in}-l).
```

Use:

```math
l_{\rm in}=\sqrt{GM_2R_c}
```

for the first version.

Add stream-impact heating:

```math
Q_{\rm stream}
\simeq
\frac{1}{2}S_\Sigma |\mathbf v_{\rm in}-\mathbf v_d|^2.
```

If detailed velocities are unavailable, parameterize:

```math
Q_{\rm stream}
=\epsilon_{\rm stream}\frac{GM_2}{2R_c}S_\Sigma.
```

Scan:

```text
epsilon_stream = 0, 0.1, 0.3, 1
```

## 6.2 Tidal torque

Add a smooth tidal torque density near `R_out`.

A simple first form:

```math
\Lambda_{\rm tide}(R)=
\Lambda_0
\left(\frac{R}{R_{\rm out}}\right)^n
\exp\left[-\frac{(R_{\rm out}-R)^2}{2\Delta_{\rm tide}^2}\right]
```

for `R` near `R_out`, with sign chosen so that the tidal torque removes angular momentum from the outer minidisk.

Calibration target:

```text
outer disk should be truncated near R_out = f_t R_H
```

Scan:

```text
f_t = 0.2, 0.3, 0.5
Delta_tide = 0.03--0.1 R_out
```

Eventually, `Lambda_tide(R)` should be replaced by Layer-1 hydro output.

## 6.3 Interpreting outward flow

In the stream-fed model, an outward-flow region outside `R_c` can be physical if:

```text
mass continuity with source and wind is satisfied
angular momentum is removed by tidal torque
outer boundary does not artificially reflect material
```

An outer decretion-like region is not automatically a failure.

---

# 7. Add Wind Only After the Advective Branch Is Controlled

Once an advection-only global branch exists, add wind.

## 7.1 Energy-limited wind closure

Vertical Eddington flux from both faces:

```math
Q_{{\rm Edd},z}=\frac{2c\Omega_K^2H}{\kappa}.
```

Energy available after advection:

```math
Q_{\rm avail}
=
Q_{\rm visc}+Q_{\rm stream}+Q_{\rm tide}-Q_{\rm adv}.
```

Wind energy:

```math
Q_{\rm wind}=\epsilon_w[Q_{\rm avail}-Q_{{\rm Edd},z}]_+.
```

Radiative cooling:

```math
Q_{\rm rad}=Q_{\rm avail}-Q_{\rm wind}.
```

Mass loss:

```math
\dot\Sigma_w=\frac{Q_{\rm wind}}{\mathcal E_w}.
```

with:

```math
\mathcal E_w=\frac{GM_2}{2R}+\frac{v_\infty^2}{2}.
```

Recommended scans:

```text
epsilon_w = 0, 0.1, 0.3, 1
v_infty   = 0, 0.5 v_esc, v_esc
```

## 7.2 Wind angular momentum

Baseline non-MHD wind:

```math
l_w=l.
```

This removes the angular momentum carried by the lost mass but does not apply an extra torque.

Optional later magnetic lever arm:

```math
l_w=\lambda_w l,
\qquad
\lambda_w>1.
```

Do not use `lambda_w > 1` in the baseline hydro/radiative model.

## 7.3 Wind consistency checks

Verify:

```math
Q_{\rm rad}+Q_{\rm wind}=Q_{\rm avail}
```

and:

```math
\frac{d\dot M}{dR}=2\pi R\dot\Sigma_w
```

away from stream source regions.

Do not double-count wind energy.

Use either:

```text
Option A: compute Q_wind from excess energy and derive dotSigma_w
```

or:

```text
Option B: prescribe dotSigma_w and subtract Q_wind = dotSigma_w E_w
```

but not both simultaneously.

---

# 8. A Steady Hot Branch Is Not the Only Possible Outcome

If no exact steady global hot branch is found, the model is not automatically dead.

The disk may undergo a transient high state:

```text
thermal runaway -> expansion -> outflow/draining -> return to cool state
```

In this case, the burst must be demonstrated by the time-dependent equations rather than by a static S-curve.

Therefore:

```text
Failure to find a steady hot branch means the high state must be demonstrated dynamically.
It does not immediately rule out a QPE-like burst.
```

However, if the time-dependent model always settles smoothly without hysteresis or eruptions, then this specific limit-cycle explanation is disfavored.

---

# 9. Time-Dependent Model After Steady Solvers Pass

Once the steady isolated and stream-fed solvers are validated, implement the time-dependent equations.

## 9.1 Evolve surface density

```math
\frac{\partial\Sigma}{\partial t}
+
\frac{1}{R}\frac{\partial}{\partial R}(R\Sigma v_R)
=
S_\Sigma-\dot\Sigma_w.
```

## 9.2 Evolve internal energy or temperature

Use vertically integrated internal energy `U` if possible:

```math
\frac{\partial U}{\partial t}
+
\frac{1}{R}\frac{\partial}{\partial R}(RUv_R)
=
Q_{\rm visc}+Q_{\rm stream}+Q_{\rm tide}
-Q_{\rm rad}-Q_{\rm wind}
-\text{PdV/advection terms handled conservatively}.
```

If evolving `T` directly, ensure consistency with:

```math
\Sigma T\left(\frac{\partial s}{\partial t}+v_R\frac{\partial s}{\partial R}\right)
=
Q^+ - Q^-.
```

## 9.3 Stability perturbation tests

For each candidate branch:

```text
T -> T * (1 +/- 0.01)
Sigma -> Sigma * (1 +/- 0.01)
```

Expected:

```text
stable branch returns
unstable branch departs
```

## 9.4 Limit-cycle outputs

Track:

```text
P_QPE
t_high
duty cycle
DeltaM_cycle
Mdot_peak
Mdot_ISCO(t)
wind mass loss
radiated energy
```

A true limit cycle should show distinct transition states analogous to:

```math
\Sigma_{\max}(R),
\qquad
\Sigma_{\min}(R).
```

without manual switching.

---

# 10. Go / No-Go Criteria

## Continue with the model if

```text
isolated global advective branch is recovered
branch remains after stream feeding and tidal truncation
xi_eff(R) is smooth and converged
global conservation residuals are < 1e-2
thermal perturbation returns to the hot branch
time-dependent calculation crosses distinct upper/lower transition states
```

## Regard the model as fragile if

```text
branch exists only for one finely tuned stress prescription
branch requires clipping H/R, Q_adv, or xi_eff
branch disappears under modest boundary/source changes
strong winds remove the reservoir before a burst develops
```

## Regard this specific limit-cycle explanation as disfavored if

```text
a validated isolated slim-disk solver works,
but no corresponding branch exists once realistic stream feeding and tidal truncation are included,
across plausible parameters
```

or if:

```text
time-dependent disk evolution always settles smoothly without hysteresis or eruptions.
```

---

# 11. Recommended Next Codex Sprint

Use this as the immediate task list.

## Sprint A: Audit hardening

Implement:

```text
pointwise energy residuals
L1/L2/max residuals
independent entropy-gradient formula
manufactured isentropic tests
resolution/stencil comparison
boundary-zone flags
continuity residual for Mdot
```

Expected deliverables:

```text
outputs/figures/global_slim_audit_hardened.png
outputs/tables/global_slim_audit_hardened.md
new tests in tests/test_entropy_advection.py and tests/test_global_slim.py
```

## Sprint B: Isolated global no-wind slim disk

Implement:

```text
S_Sigma = 0
Lambda_tide = 0
dotSigma_w = 0
Q_stream = 0
constant Mdot branch continuation
solve for Sigma(R), T(R), and v_R/torque consistently
```

Expected deliverables:

```text
outputs/figures/isolated_slim_branch_continuation.png
outputs/tables/isolated_slim_branch_summary.md
```

Success criteria:

```text
Mdot constant with R
smooth xi_eff
energy L1 residual < 1e-2
integrated Qadv/Qvisc between 0 and 1 for advective branch
```

## Sprint C: Decide on radial momentum

If isolated branch requires:

```text
H/R > 0.4
```

then implement or plan radial momentum and sonic regularity before adding stream/tide physics.

Deliverable:

```text
outputs/tables/radial_momentum_need_assessment.md
```

## Sprint D: Add stream source and tidal truncation

Only after Sprint B passes.

Implement:

```text
Gaussian S_Sigma(R)
stream angular momentum source
a simple Q_stream heating term
smooth tidal torque near Rout
source-integrated continuity and angular-momentum checks
```

Deliverables:

```text
outputs/figures/stream_fed_truncated_branch.png
outputs/tables/stream_fed_truncated_budget.md
```

## Sprint E: Add energy-limited wind

Only after Sprint D passes.

Implement:

```text
Q_Edd,z
Q_avail
Q_wind = epsilon_w [Q_avail - Q_Edd,z]_+
dotSigma_w = Q_wind / E_w
baseline l_w = l
wind budget tests
```

Deliverables:

```text
outputs/figures/wind_regulated_branch.png
outputs/tables/wind_budget_summary.md
```

## Sprint F: Time-dependent limit cycle

Only after steady branch and conservation tests are validated.

Implement:

```text
time evolution of Sigma and U/T
thermal perturbation tests
long integration for cycles
```

Deliverables:

```text
outputs/figures/time_dependent_limit_cycle.png
outputs/tables/limit_cycle_metrics.md
```

Metrics:

```text
P_QPE ~ days
t_high ~ days
duty cycle ~ 0.3--0.6
DeltaM_cycle ~ 1e-5--1e-4 Msun
Mdot_peak consistent with required luminosity
```

---

# 12. Compact Prompt for Codex

Paste this into a Codex task if you want a short instruction.

```text
The current IMRI QPE code shows that the local xi=0.3 hot branch fails its first global slim/wind audit. Do not keep scanning imposed xi. Implement the next-stage global solver program.

First harden the audit: add pointwise/L1/L2/max energy residuals, an independent gas+radiation entropy-gradient formula, manufactured isentropic tests, resolution/stencil convergence, boundary-zone flags, and continuity residuals for Mdot.

Then build an isolated no-wind global slim-disk benchmark with S_Sigma=0, Lambda_tide=0, dotSigma_w=0, Q_stream=0. Solve for Sigma(R), T(R), and v_R or torque consistently. Do not hold Sigma fixed and relax only T. Use branch continuation in Mdot from a low thin-disk state upward. The success criteria are constant Mdot(R), smooth xi_eff(R), energy L1 residual <1e-2, and integrated Qadv/Qvisc between 0 and 1 for an advective branch.

Only after the isolated benchmark passes, add stream feeding as a Gaussian S_Sigma centered at Rc, add stream angular momentum and heating, and add a smooth tidal torque near Rout. Check integrated mass and angular-momentum budgets. Outward flow outside Rc is allowed only if it satisfies the source/tidal budget.

Only after that, add energy-limited wind with Q_Edd,z = 2 c Omega_K^2 H/kappa, Q_avail = Q_visc + Q_stream + Q_tide - Q_adv, Q_wind = epsilon_w [Q_avail - Q_Edd,z]_+, dotSigma_w = Q_wind/E_w, and baseline l_w=l. Verify Qrad + Qwind = Qavail and dMdot/dR = 2 pi R dotSigma_w away from source regions.

Finally, run time-dependent Sigma and energy/T evolution to test for true hysteresis and limit cycles without manual branch switching. The model is promising only if a branch or transient high state survives these global conservation and stability tests.
```

---

# 13. Files to Update

Suggested existing files:

```text
src/imri_qpe/layer3_minidisk_1d/global_slim.py
src/imri_qpe/layer3_minidisk_1d/entropy_advection.py
src/imri_qpe/layer3_minidisk_1d/winds.py
scripts/plot_global_slim_wind_audit.py
tests/test_global_slim.py
README.md
```

Suggested new files:

```text
src/imri_qpe/layer3_minidisk_1d/audit_metrics.py
src/imri_qpe/layer3_minidisk_1d/isolated_slim_solver.py
src/imri_qpe/layer3_minidisk_1d/stream_sources.py
src/imri_qpe/layer3_minidisk_1d/tidal_torques.py
tests/test_entropy_advection.py
tests/test_audit_metrics.py
tests/test_isolated_slim_solver.py
scripts/plot_isolated_slim_branch.py
scripts/plot_stream_fed_branch.py
outputs/tables/global_slim_audit_hardened.md
outputs/tables/isolated_slim_branch_summary.md
```

---

# 14. Scientific Summary for README

Use this language in project summaries:

```text
The local xi-parameterized hot branch was a useful target but failed the first global entropy-advection audit. This failure is not fatal: it shows that the problem must be solved as a coupled radial slim/wind minidisk calculation. The next decisive step is an isolated global slim-disk benchmark with Q_adv computed from ds/dR, followed by a stream-fed tidally truncated calculation. Only after those pass should wind regulation and time-dependent limit cycles be interpreted physically.
```
