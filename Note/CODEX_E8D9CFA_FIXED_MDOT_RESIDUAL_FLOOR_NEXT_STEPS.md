# Codex Next-Step Brief: Fixed-\(\dot M\) Transonic Residual Floor After Commit `e8d9cfa`

Repository:

```text
https://github.com/huanyang07/IMBH
commit: e8d9cfa
```

This note focuses on the fixed-\(\dot M\) transonic residual-floor analysis, especially:

```text
outputs/tables/transonic_rout_continuation_audit.md
outputs/tables/transonic_symmetric_sonic_polish_audit.md
outputs/tables/transonic_outer_thermal_match_audit.md
src/imri_qpe/layer3_minidisk_1d/transonic_collocation.py
```

The main question is why the fixed-\(\dot M\) polish still stalls at residuals of
order \(10^{-5}\), and what Codex should do next.

---

# 1. Executive Evaluation

The latest audits are useful and mostly positive. They show that the earlier
\(\sim10^{-4}\) residual floor was substantially reduced once the exact finite-radius
Keplerian outer boundary was replaced by a pressure/matched outer closure.

However, the remaining \(\sim1.5\times10^{-5}\) floor is **not yet a physical
failure of the transonic branch**. It is most likely a combination of:

```text
1. a still-imperfect outer asymptotic/matching closure,
2. a nonsmooth nested local solve inside the outer boundary residual,
3. midpoint collocation/truncation error,
4. finite-difference Jacobian noise,
5. and insufficiently converged fixed-Mdot Newton/polish solves.
```

The correct next step is **not** to add wind, stream feeding, tidal truncation,
or time dependence. The next step is to make the fixed-\(\dot M\) boundary-value
problem smooth, explicit, and mesh-convergent.

---

# 2. What the Latest Audits Establish

## 2.1 Gradual \(R_{\rm out}\) continuation confirms the outer-boundary diagnosis

The gradual \(R_{\rm out}\) audit uses:

```text
fixed Mdot/Edd ~= 0.9028
N = 64
outer closure = matched_full
sonic pivot = C1
```

The residual improves strongly as \(R_{\rm out}\) is moved outward:

```text
R_out = 3000 rg  -> selected final ~5.7e-5
R_out = 4000 rg  -> selected final ~3.2e-5
R_out = 5000 rg  -> selected final ~1.5e-5
R_out = 6500 rg  -> selected final ~2.0e-5
R_out = 8000 rg  -> selected final ~2.3e-5
R_out = 10000 rg -> selected final ~2.5e-5
```

The best point is near:

```text
R_out ~= 5000 rg
selected final ~= 1.5e-5
Rson/rg ~= 5.83
lambda0 ~= 3.675
max H/R ~= 0.15
outer H/R ~= 0.0144
int Qadv/Qvisc ~= 0.074
```

Interpretation:

```text
The finite-radius outer boundary really was the dominant 1e-4-level problem.
After correcting it, the floor is now at the 1e-5 level.
```

But this does **not** yet prove an exact fixed-\(\dot M\) root exists at N64.
The solves still terminate by `max_nfev`, not by true optimizer convergence.

---

## 2.2 Symmetric sonic polishing shows the sonic pivot is not the main remaining problem

The symmetric sonic audit starts from the best \(R_{\rm out}=5000r_g\) matched-full
checkpoint.

It compares:

```text
square_C1
square_C2
symmetric D,C1,C2
symmetric D,C1,C2,K with different weights
```

All variants stall near:

```text
selected final ~= 1.5e-5
```

with similar profiles:

```text
Rson/rg ~= 5.831--5.832
lambda0 ~= 3.675
max H/R ~= 0.1499
outer H/R ~= 0.01444
int Qadv/Qvisc ~= 0.0736
```

Changing the sonic residual set only redistributes the residual among:

```text
outer_2
C1
C2
```

It does not remove the floor.

Interpretation:

```text
The remaining floor is not mainly caused by choosing C1 instead of C2, or by
dropping/adding K. The sonic block is not the leading problem anymore.
```

Keep \(D,C_1,C_2,K\) as diagnostics, but stop spending the next sprint on
sonic-pivot variations alone.

---

## 2.3 The outer thermal-match audit shows `matched_full` is the best current closure

At \(R_{\rm out}=3000r_g\), the audit compares:

```text
pressure_thin_direct
matched_pressure_thin
matched_full
```

and slope sources:

```text
polyfit
local_ode
```

The fixed-\(\dot M\) solve results show:

```text
polyfit + pressure_thin_direct      -> residual ~2.6e-4
polyfit + matched_pressure_thin     -> residual ~1.65e-4
polyfit + matched_full              -> residual ~4.7e-5

local_ode + pressure_thin_direct    -> residual ~5.1e-5
local_ode + matched_pressure_thin   -> residual ~3.9e-5
local_ode + matched_full            -> residual ~3.7e-5
```

So:

```text
matched_full is the right direction.
```

However, the `local_ode` slope source reports huge slopes in the initial diagnostic, e.g.

```text
g_u ~ 2772
g_T ~ 857
```

at \(R_{\rm out}=3000r_g\). Those numbers are not physically acceptable as
outer asymptotic slopes. They are a sign that the local ODE slope extraction is
ill-conditioned near the endpoint or close to a critical/singular local matrix.

Use the smooth polyfit/reduced-solver slopes as the baseline. Treat `local_ode`
slopes only as diagnostics until they are regularized.

---

# 3. The Main Remaining Problem: `matched_outer_state` Is a Nested Optimizer Inside the Residual

In `transonic_collocation.py`, the current matched closure effectively does:

```python
def _outer_matched_state_boundary_residual(logR, y, lambda0, params):
    g_match = ...
    y_match = matched_outer_state(logR, lambda0, params, g_match=g_match, initial_y=y)
    return y - y_match
```

and `matched_outer_state()` itself calls `scipy.optimize.least_squares()`.

This means the global BVP residual contains a **nested nonlinear solve** inside
every residual evaluation.

That has several problems:

```text
1. The residual is not a simple smooth function of the global unknown vector.
2. Finite-difference Jacobians become unreliable because every perturbation
   triggers a new local optimization.
3. The local optimizer can switch roots or terminate at slightly different
   tolerances under tiny perturbations.
4. The global optimizer sees the outer boundary through the numerical noise of
   another optimizer.
5. It is expensive and makes max_nfev termination more likely.
```

This is probably a major reason why the residual stalls near \(10^{-5}\).

---

# 4. Replace the Nested Matched-State Closure With a Direct Smooth Outer Residual

The next Codex sprint should implement an explicit outer derivative/asymptotic
closure that avoids a nested solve.

## 4.1 Preferred closure: direct full-equation slope match

At the outer boundary, use the matched slopes

```math
g_{\rm match}=
\left(
\frac{d\ln u}{d\ln R},
\frac{d\ln T}{d\ln R}
\right)_{\rm match}.
```

Then impose the local scaled differential equations directly:

```math
B_R
=
F_R(R_{\rm out}, y_{\rm out}, g_{\rm match}, \lambda_0)=0,
```

```math
B_E
=
F_E(R_{\rm out}, y_{\rm out}, g_{\rm match}, \lambda_0)=0.
```

Here \(F_R,F_E\) are the same scaled radial-momentum and energy residuals used
inside the collocation intervals.

In code, add:

```python
outer_closure = "full_slope_match"
```

with:

```python
def _outer_full_slope_match_residual(logR, y, lambda0, params):
    g_match = params.outer_match_log_slopes
    if g_match is None:
        g_match = reduced_outer_log_slopes(params, lambda0)
    return _scaled_local_differential_residual(logR, y, g_match, lambda0, params)
```

This gives two boundary equations without a nested optimizer.

It is mathematically close to the current `matched_full` idea, but much cleaner
for Newton/Jacobian work.

## 4.2 Keep `matched_outer_state` only as a diagnostic

After implementing `full_slope_match`, keep `matched_outer_state` only for:

```text
diagnostic target-state calculation
plotting
cross-checking
```

Do not use it inside the production residual.

## 4.3 Immediate comparison test

At:

```text
Mdot/Edd = 0.9028
N = 64
R_out = 5000 rg
pivot = C1 and C2
```

compare:

```text
matched_outer_state     # old nested-solve closure
full_slope_match        # new direct closure
```

Targets:

```text
selected residual < 1e-6 if a root exists at N64
D,C1,C2,K all < 1e-6
unused compatibility < 1e-6
no max_nfev termination
```

If `full_slope_match` converges while `matched_outer_state` stalls, the nested
outer solve was the culprit.

---

# 5. Stabilize the Outer Slope Source

## 5.1 Use polyfit/reduced slopes as baseline

The current polyfit slopes are physically plausible, for example:

```text
g_u ~ -0.55 at R_out=5000
g_T ~ -0.85 at R_out=5000
```

Use those as the production baseline.

Do not use `local_ode` slopes in production until the enormous values are fixed.

## 5.2 Smooth slopes across continuation steps

When continuing in \(R_{\rm out}\), update the outer slopes with an exponential
moving average:

```math
g_{\rm used}^{(k+1)}
=
(1-\eta)g_{\rm used}^{(k)}
+
\eta g_{\rm new}^{(k+1)},
```

with:

```text
eta = 0.2--0.5.
```

This prevents small slope-fit noise from appearing as boundary residual noise.

## 5.3 Add slope sensitivity diagnostics

At each fixed-\(\dot M\) polish, report:

```text
g_u
g_T
d(selected residual)/d g_u
d(selected residual)/d g_T
```

by finite differences with:

```text
delta g = 1e-3, 3e-3, 1e-2.
```

If the residual floor is highly sensitive to \(g_{\rm match}\), then the outer
closure is still controlling the solution and \(R_{\rm out}\) must be moved
farther out or matched more carefully.

---

# 6. Use Integrated Collocation Defects as the Production Residual

The code already supports:

```text
interval_residual_form = "integrated"
integrated_residual_weighting = ...
```

but the new audits still look dominated by the same \(10^{-5}\) floor. The
next sprint should explicitly make the integrated defect the production mode
and audit its conditioning.

Use:

```math
\widetilde F_i
=
\bar A_i (y_{i+1}-y_i)
+
\Delta x_i \bar c_i.
```

This is equivalent to:

```math
\bar A_i\frac{y_{i+1}-y_i}{\Delta x_i}+\bar c_i=0
```

because \(\Delta x_i>0\), but it avoids explicit \(1/\Delta x_i\) amplification
in the Jacobian.

Recommended settings to test:

```text
interval_residual_form = integrated
integrated_residual_weighting = none
integrated_residual_weighting = inverse_sqrt_dx
```

Avoid `inverse_dx` for production because it reintroduces the differential
conditioning problem.

## Required audit

At the same physical point:

```text
Mdot/Edd = 0.9028
R_out = 5000 rg
outer_closure = full_slope_match
```

run:

```text
N = 33, 65, 129
```

and report:

```text
raw condition number
equilibrated condition number
Newton iterations
line-search cuts
final residual
D,C1,C2,K
```

The key question is whether the residual floor and condition number improve
with integrated defects.

---

# 7. Do Not Overinterpret the One-Shot \(R_{\rm out}\) Probe Rows

The outer thermal-match audit explicitly states that the \(R_{\rm out}\neq3000\)
rows are one-shot remap probes, not independent continuation branches.

Those rows show large residuals, but they do not prove that those radii fail.

The gradual \(R_{\rm out}\) continuation audit is more informative. It shows
that \(R_{\rm out}\sim5000r_g\) is currently best.

## Recommended \(R_{\rm out}\) strategy

After implementing `full_slope_match`:

```text
1. Start at R_out = 3000 rg from the old branch.
2. Continue gradually to 4000, 5000, 6500, 8000, 10000 rg.
3. Use small steps in ln R_out, e.g. Delta ln R_out = 0.05--0.1.
4. Polish at each radius.
5. Only compare independently polished roots.
```

Do not compare raw remapped residuals.

---

# 8. Revisit \(R_{\rm out}=5000r_g\) Before Returning to \(\dot M\)-Continuation

The best current state is near:

```text
Mdot/Edd = 0.9028
R_out = 5000 rg
selected residual ~= 1.5e-5
Rson/rg ~= 5.83
lambda0 ~= 3.675
max H/R ~= 0.15
int Qadv/Qvisc ~= 0.074
```

This is the correct debugging target.

## Immediate fixed-\(\dot M\) root experiment

Use:

```text
outer_closure = full_slope_match
interval_residual_form = integrated
integrated_residual_weighting = none or inverse_sqrt_dx
sonic pivot = C1 and C2
N = 64
R_out = 5000 rg
max_nfev >= 500
dense/equilibrated Newton or SVD step
```

Targets:

```text
selected residual < 1e-6
unused compatibility < 1e-6
optimizer success
no active bounds
```

If this fails, diagnose by residual block:

```text
interval R
interval E
outer 1
outer 2
D
C1
C2
K
```

Do not resume pseudo-arclength until this fixed-\(\dot M\) root is genuinely
polished.

---

# 9. Consider Higher-Order Collocation If the Floor Persists

If the direct outer closure and integrated defects still leave a stable
\(\sim10^{-5}\) floor, the likely culprit is discretization.

The current midpoint collocation is low order. Move to a higher-order defect
for the interval equations.

## Option A: trapezoidal defect

For the ODE form:

```math
y' = f(x,y),
```

use:

```math
y_{i+1}-y_i
-
\frac{\Delta x_i}{2}
\left[
f(x_i,y_i)+f(x_{i+1},y_{i+1})
\right]
=0.
```

Here \(f\) is obtained by solving:

```math
\bar A g = -\bar c.
```

away from the sonic point. Since the first interval is close to the sonic
point, keep midpoint for the first interval if needed.

## Option B: Hermite-Simpson defect

Use midpoint plus endpoint slopes for fourth-order accuracy:

```math
y_{i+1}-y_i
-
\frac{\Delta x_i}{6}
(f_i+4f_m+f_{i+1})
=0.
```

This is more accurate but requires robust evaluation of \(f\) at endpoints.

## Development path

Do not immediately replace the whole solver. First implement:

```text
interval_scheme = midpoint
interval_scheme = trapezoid
```

and compare \(N=33,65,129\).

A real solution should converge under scheme refinement.

---

# 10. Independent Nested-Grid Convergence

The current \(N=64\) result is useful, but not enough. Use nested grids:

```text
N = 33, 65, 129
```

or:

```text
N = 65, 129, 257
```

Use derivative-aware prolongation:

```text
PCHIP in logR for logu, logT
eventually Hermite using ODE slopes.
```

Procedure:

```text
1. Solve N=33 from low Mdot to 0.9028.
2. Prolong to N=65 and polish.
3. Prolong to N=129 and polish.
4. Compare independently polished roots.
```

Compare:

```text
Rson
lambda0
max H/R
outer H/R
integrated Qadv/Qvisc
Sigma(R)
T(R)
Omega/OmegaK
xi_eff away from first/last two cells
```

Do not use one-shot interpolation residuals as convergence evidence.

---

# 11. Once Fixed-\(\dot M\) Roots Are Polished, Restart Pseudo-Arclength

Only after obtaining polished roots at:

```text
Mdot/Edd = 0.90
0.965
0.996
```

with:

```text
residual < 1e-6
D,C1,C2,K < 1e-6
```

restart pseudo-arclength continuation.

Discard or mark old \(10^{-5}\)-floor anchors as `legacy`.

Use the root-polished checkpoints only.

---

# 12. What Not To Do Next

Do not spend the next sprint on:

```text
- wind;
- stream source;
- tidal torque;
- time-dependent limit cycle;
- synthetic emission;
- more sonic-pivot weight scans;
- interpreting a physical fold near Mdot/Edd~1;
- increasing max_nfev alone without changing the residual formulation.
```

The current difficulty is still a deterministic boundary-value-solver issue.

---

# 13. Concrete Code Changes

## `transonic_collocation.py`

Add:

```python
outer_closure = "full_slope_match"
```

Implement:

```python
def _outer_full_slope_match_residual(logR, y, lambda0, params):
    g_match = params.outer_match_log_slopes
    if g_match is None:
        g_match = reduced_outer_log_slopes(params, lambda0)
    return _scaled_local_differential_residual(logR, y, g_match, lambda0, params)
```

Keep:

```python
matched_outer_state()
```

but do not call it inside production residuals.

Make integrated defects the default for transonic fixed-\(\dot M\) solves:

```python
interval_residual_form = "integrated"
integrated_residual_weighting = "none"   # first test
```

Then test:

```python
integrated_residual_weighting = "inverse_sqrt_dx"
```

Add residual-block output for the direct outer closure.

## New scripts

Create:

```text
scripts/run_transonic_full_slope_match_audit.py
scripts/run_transonic_integrated_defect_fixed_root.py
scripts/run_transonic_nested_grid_fixed_mdot.py
```

Outputs:

```text
outputs/tables/transonic_full_slope_match_audit.md
outputs/tables/transonic_integrated_defect_fixed_root.md
outputs/tables/transonic_nested_grid_fixed_mdot.md
outputs/figures/transonic_full_slope_match_blocks.png
outputs/figures/transonic_integrated_defect_conditioning.png
outputs/figures/transonic_nested_grid_profiles.png
```

---

# 14. Suggested Immediate Experiment Matrix

## Experiment A: direct outer closure at best current point

```text
Mdot/Edd = 0.9028
R_out = 5000 rg
N = 64
pivot = C1, C2
outer_closure = full_slope_match
interval_residual_form = differential, integrated
integrated_residual_weighting = none, inverse_sqrt_dx
```

Report:

```text
selected residual
dominant block
D,C1,C2,K
outer residuals
interval residuals
condition estimates
nfev
success flag
```

## Experiment B: outer-slope sensitivity

Perturb:

```text
g_u -> g_u + delta
g_T -> g_T + delta
```

with:

```text
delta = 1e-3, 3e-3, 1e-2
```

and measure how the final residual changes.

## Experiment C: nested-grid fixed root

Use:

```text
N = 33 -> 65 -> 129
```

with the best outer closure.

## Experiment D: R_out continuation with new closure

Use:

```text
R_out = 3000 -> 4000 -> 5000 -> 6500 -> 8000 -> 10000 rg
```

but only compare polished roots.

---

# 15. Go / No-Go Criteria

## Proceed to pseudo-arclength continuation if

```text
- fixed-Mdot root at 0.9028 reaches residual < 1e-6;
- C1 and C2 pivots converge to the same profile;
- D,C1,C2,K are all < 1e-6;
- nested-grid roots agree to a few percent;
- conditioning improves with integrated defects;
- no nested optimizer is used inside the production residual.
```

## Revisit outer asymptotics if

```text
- full_slope_match still leaves a residual dominated by outer rows;
- residual depends strongly on g_match;
- R_out continuation does not converge as R_out increases.
```

## Revisit collocation order if

```text
- residual floor is dominated by interval rows;
- residual decreases with N at the expected midpoint-collocation rate;
- higher-order defects reduce the floor.
```

## Revisit sonic/eigenvalue equations only if

```text
- outer and interval residuals polish below 1e-6;
- compatibility residuals remain the dominant block;
- C1 and C2 pivots converge to distinct states.
```

---

# 16. Compact Codex Prompt

```text
The latest e8d9cfa audits show that matched_full and gradual R_out continuation
reduced the fixed-Mdot residual floor from ~1e-4 to ~1.5e-5, with the best
state near R_out=5000 rg. Symmetric sonic polishing does not remove the floor,
so sonic pivot choice is not the main issue. The outer thermal audit shows
matched_full is the best closure, but it currently calls a nested least_squares
matched_outer_state inside the global residual, which makes the residual
nonsmooth and the finite-difference Jacobian unreliable.

Implement next:

1. Add outer_closure="full_slope_match":
       B_outer = scaled_differential_residual(R_out, y_out, g_match, lambda0)
   using smooth reduced/polyfit slopes g_match. Do not call matched_outer_state
   inside the production residual.

2. Keep matched_outer_state only as a diagnostic.

3. Make integrated collocation defects the production interval residual:
       residual_i = Abar_i @ (y_{i+1}-y_i) + dx_i*cbar_i
   Test weighting none and inverse_sqrt_dx.

4. At Mdot/Edd=0.9028, R_out=5000 rg, N=64, pivots C1/C2, compare:
       old matched_outer_state closure
       new full_slope_match closure
       differential vs integrated defects
   Require residual <1e-6 if a root exists.

5. Use polyfit/reduced slopes as baseline. Do not use local_ode slopes in
   production; current local_ode slopes can be enormous and ill-conditioned.

6. Add slope-sensitivity diagnostics for g_u and g_T.

7. Use nested grids N=33,65,129 and independently polished roots. Do not use
   one-shot remap residuals as convergence tests.

8. Restart pseudo-arclength only after fixed-Mdot roots at 0.90,0.965,0.996 are
   polished below 1e-6 with D,C1,C2,K all small.

Do not add wind, stream, tides, or time dependence yet.
```

---

# 17. Bottom Line

The latest results are progress:

```text
- The original 1e-4 finite-Rout/Keplerian-boundary floor has been reduced.
- Matched-full outer closure and R_out continuation move the system to a much
  better ~1.5e-5 residual state.
- Sonic-pivot variations do not cure the remaining floor.
```

The most likely next bottleneck is not physics; it is the **nested matched-state
outer residual plus low-order/finite-difference collocation machinery**.

Replace the nested outer solve with a direct full-slope boundary residual, make
integrated defects the production interval residual, and then perform nested-grid
fixed-\(\dot M\) polishing before returning to high-\(\dot M\) pseudo-arclength.
