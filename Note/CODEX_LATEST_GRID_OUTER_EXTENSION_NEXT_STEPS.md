# Codex Next-Step Brief: Latest Fixed-\(\dot M\) Transonic Results and Plan to Proceed

Repository reviewed:

```text
https://github.com/huanyang07/IMBH
branch: main, latest accessible state
```

Primary files reviewed:

```text
Note/CODEX_SLOPE_LAW_GRID_VALIDATION_NEXT_STEPS.md
outputs/tables/transonic_slope_unknown_root.md
outputs/tables/transonic_staged_resolution_continuation.md
outputs/tables/transonic_trapezoid_collocation_audit.md
outputs/tables/transonic_controlled_slope_rout_continuation.md
outputs/tables/transonic_outer_slope_law_audit.md
outputs/tables/transonic_high_rate_ladder.md
src/imri_qpe/layer3_minidisk_1d/transonic_collocation.py
src/imri_qpe/layer3_minidisk_1d/transonic_continuation.py
```

## Executive assessment

The latest results are **not a failure of the physical IMRI/QPE model**. They are a numerical-boundary-value-solver issue.

The key result remains:

```text
A good local fixed-Mdot transonic root exists near
Mdot/Mdot_Edd ~= 0.9028,
R_out ~= 6500 rg,
N ~= 65,
R_son ~= 5.828 rg,
lambda0 ~= 3.675,
int Qadv/Qvisc ~= 0.074,
physical residual ~= 2e-6.
```

However, that root is **not yet a validated continuum solution**, because it has not survived robust grid refinement and collocation-scheme tests.

The most important new findings are:

```text
1. Promoting outer slopes to weakly constrained unknowns did not actually move
   the slopes; dg_u and dg_T are essentially zero in the accepted N65 runs.

2. Staged resolution continuation from N65 becomes worse already at N67--N77.
   The residual rises from ~2e-6 to ~7e-6, ~1e-5, ~2e-5, and worse.
   The dominant block becomes interval_R or C2.

3. Mixed trapezoidal collocation currently fails badly compared with midpoint:
   at N65, midpoint gives ~2e-6 while trapezoid gives physical residual
   ~2.6e-3.

4. Controlled R_out continuation with raw/EMA/held slopes shows that the outer
   slope closure still controls the solution. Moving R_out while carrying a
   calibrated slope law often makes the residual worse.

5. The high-rate ladder is still only a scout. It should not be interpreted
   because the fixed-Mdot root near 0.9 Eddington is not yet mesh validated.
```

Therefore the next scientific milestone should be:

```text
Mesh-converged fixed-Mdot transonic root at Mdot/Mdot_Edd ~= 0.9.
```

Do **not** add wind, stream feeding, tidal truncation, time dependence, or
synthetic emission yet.

---

# 1. Current status from the latest tables

## 1.1 Slope-unknown fixed-\(\dot M\) root

The slope-unknown audit reports several N65 runs with:

```text
physical residual ~= 1.93e-6
R_son/rg ~= 5.828
lambda0 ~= 3.675
int adv ~= 0.07381
science pass = yes
```

But the solved slope corrections are essentially zero:

```text
dg_u ~ 1e-12
dg_T ~ 1e-13
```

Interpretation:

```text
The "slope unknown" implementation has not yet tested genuine slope freedom.
The priors are effectively acting as hard constraints, or the existing prior
happens to lie close to the local optimum. This does not solve the outer
slope-dependence problem.
```

## 1.2 Staged resolution continuation

The staged resolution table starts from the N65 anchor and tries small resolution increases.

The best sequence shows:

```text
N65 anchor: physical ~= 1.93e-6, science pass yes
N67 release/branch_polish: physical ~= 7e-6, bridge pass only
N69 release/branch_polish: physical ~= 1.2e-5
N73 release/branch_polish: physical ~= 2e-5
N77 release: physical ~= 4.5e-5
```

The staged runs also show increasing branch distances relative to the N65 anchor.

Interpretation:

```text
The root is not stable under the current refinement/prolongation strategy.
The failure begins immediately when moving off N65, so this is not a high-rate
physical issue.
```

## 1.3 Trapezoidal collocation audit

The trapezoidal audit is very revealing:

```text
N65 midpoint: physical ~= 1.93e-6
N65 trapezoid_mixed: physical ~= 2.6e-3
```

At N73/N81, trapezoid remains much worse.

Interpretation:

```text
The mixed trapezoid implementation is not yet a valid production discretization.
It may have a sign/scaling bug, an endpoint-slope problem near the sonic
interval, or may be using f=-A^{-1}c too close to a singular point.
Do not use the current trapezoid result as evidence against the branch.
```

## 1.4 Controlled \(R_{\rm out}\) continuation

The controlled R_out audit shows that slope handling is still fragile.

Examples:

```text
raw calibrated:
    R_out 5000 -> residual 4.3e-6
    R_out 5500 -> residual 3.1e-5
    R_out 6000 -> residual 3.0e-5
    R_out 6500 -> residual 1.9e-5

EMA calibrated:
    often worsens to ~1e-4--3e-4

held calibrated:
    worsens even more as R_out is moved
```

Interpretation:

```text
The slope closure is still selecting the numerical solution. It is not yet a
robust asymptotic boundary condition.
```

## 1.5 High-rate ladder

The scout high-rate run reaches moderately high rates on coarse grids, but
confirmation is poor. For example, N48 reaches:

```text
Mdot/Mdot_Edd ~= 0.7
max H/R ~= 0.125
```

but an N64 confirmation at:

```text
Mdot/Mdot_Edd = 0.5
```

fails with:

```text
interval_R ~= 2.9e-3
```

Interpretation:

```text
High-rate continuation is premature. The coarse high-rate branch may be useful
for intuition, but it is not a validated physical branch.
```

---

# 2. Main diagnosis

The bottleneck has moved. It is no longer the old local-\(\xi\) hot branch, nor
the exact finite-radius Keplerian boundary, nor the nested matched-state
optimizer.

The current bottleneck is:

```text
outer-asymptotic closure + grid refinement + collocation consistency.
```

More specifically:

```text
1. The outer slope closure is too influential.
2. The slope-unknown formulation did not actually free the slopes.
3. Remapping/refinement moves the solution to a nearby but not equivalent
   discrete problem.
4. Midpoint collocation has a good local root, but not a demonstrated
   convergence sequence.
5. The attempted trapezoid formulation is not yet internally validated.
```

The next task is not to tune slopes further. The next task is to remove the
slope closure as an external input.

---

# 3. Recommended strategic change: replace slope closure with a two-domain outer extension

The cleanest way forward is to stop prescribing outer slopes altogether.

Instead, solve a two-domain BVP:

```text
Domain I: inner transonic domain
    R_son -> R_match

Domain II: outer asymptotic domain
    R_match -> R_far
```

Use the same full differential equations in both domains. Then impose simple
thin-disk asymptotic boundary conditions only at \(R_{\rm far}\), where
\(H/R\) is genuinely small enough that the exact Keplerian/thin closure is valid.

This turns the outer slope from an input into an output.

## 3.1 Why this is better

The current `full_slope_match` boundary condition imposes:

```text
F_local(R_out, y_out, g_match, lambda0) = 0
```

where `g_match` is estimated from a reduced/asymptotic fit.

But the solution is sensitive to changes in `g_match` of order \(10^{-3}\).
That means `g_match` is not a harmless asymptotic parameter.

The two-domain method replaces this with:

```text
solve for the outer-domain profile directly;
let the outer-domain collocation determine g_out.
```

Then at the far boundary use:

```text
Omega/Omega_K -> 1
Q_adv/Q_visc -> 0
Q_visc_thin = Q_rad
```

at a radius where these are actually accurate.

## 3.2 Suggested layout

Pick:

```text
R_match = 3000--6500 r_g
R_far   = 3e4--1e5 r_g
```

Start with:

```text
R_match = 6500 r_g
R_far   = 5e4 r_g
```

Use two computational grids:

```text
inner: xi in [0,1], R_son -> R_match
outer: eta in [0,1], R_match -> R_far
```

Unknowns:

```text
inner logu_i, logT_i
outer logu_j, logT_j
log R_son
lambda0
```

Residuals:

```text
inner interval collocation
outer interval collocation
sonic regularity at R_son
interface continuity at R_match:
    logu_inner_end = logu_outer_start
    logT_inner_end = logT_outer_start
far outer boundary:
    ln(Omega/Omega_K)=0
    Qvisc_thin = Qrad
```

The interface does not need prescribed slopes. The ODE residuals on both sides
determine slopes.

## 3.3 First implementation mode

Use midpoint collocation only at first, because the current midpoint scheme has
a working local root.

Do **not** use trapezoid until it passes independent tests.

## 3.4 Acceptance criteria

The two-domain root is accepted only if:

```text
physical differential residual < 1e-6--3e-6
D,C1,C2,K all small
R_son ~= 5.8 r_g
lambda0 ~= 3.675
int adv ~= 0.07--0.08
solution insensitive to R_match
solution insensitive to R_far once H/R_far is small
```

If this works, the entire outer-slope-law problem disappears.

---

# 4. If two-domain implementation is too large: fix slope-unknown formulation

If Codex needs an incremental path before the two-domain solver, revise the
slope-unknown method.

The current slope-unknown table shows:

```text
dg_u ~ 0
dg_T ~ 0
```

so the slopes were effectively not allowed to move.

## 4.1 Treat slope priors as soft objective penalties, not science residuals

Use unknowns:

```text
g_u_out
g_T_out
```

Boundary equations:

```text
B_outer_R = 0
B_outer_E = 0
```

Soft penalties:

```math
P_u = w_u(g_u-g_{u,\rm prior}),
```

```math
P_T = w_T(g_T-g_{T,\rm prior}).
```

But do **not** include \(P_u,P_T\) in the physical max residual.

Report:

```text
physical residual from disk equations only
prior penalty separately
```

## 4.2 Scan prior weights

Run:

```text
sigma_g = 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2
```

Expected behavior:

```text
hard priors -> dg ~ 0, physical residual similar to current
moderate priors -> dg ~ 1e-3, physical residual may improve
very loose priors -> unphysical slopes, reject
```

The useful regime is:

```text
dg_u, dg_T ~ 1e-3--1e-2
physical residual improves
outer slopes remain smooth with R_out and N
```

## 4.3 Do not call this a final boundary condition

This slope-unknown method is a diagnostic/bridge. The final robust method
should still be two-domain outer extension or a properly derived asymptotic
expansion.

---

# 5. Fix refinement/prolongation before more N studies

The staged resolution continuation currently fails almost immediately. This
could be a real discretization problem, but it could also be a bad remap.

## 5.1 Use nested grids

Do not use:

```text
N=65 -> 67 -> 69 -> 73 -> 77
```

as the main convergence path.

Use nested grids:

```text
N=65 -> 129 -> 257
```

or:

```text
N=33 -> 65 -> 129
```

so that every old node is preserved.

## 5.2 Use Hermite prolongation from ODE slopes

PCHIP values alone are not enough because the residual depends sensitively on
radial derivatives.

At each accepted coarse node or midpoint, compute:

```math
g = \frac{d(\ln u,\ln T)}{d\ln R}
\]

from the local differential system:

```math
A g = -c.
```

Then prolong using cubic Hermite interpolation in \(\ln R\).

Near the sonic node, use one of:

```text
- the regular sonic derivative, if available;
- one-sided derivative from the first midpoint;
- or keep the first interval midpoint-based during the first refinement.
```

## 5.3 Refinement acceptance

After prolongation to the fine grid:

1. Hold \(R_{\rm son}\) and \(\lambda_0\) fixed.
2. Solve only nodal values.
3. Free \(R_{\rm son}\).
4. Free \(\lambda_0\).
5. Run full polish.

This staged release is safer than freeing all variables immediately.

---

# 6. Debug the trapezoid scheme separately

The current trapezoid mixed scheme is much worse than midpoint. That is not
normal enough to ignore.

Before using trapezoid for physics, run unit tests.

## 6.1 Manufactured ODE test

Construct a simple known solution:

```math
y_1(x)=a_1+b_1x+c_1x^2,
```

```math
y_2(x)=a_2+b_2x+c_2x^2.
```

Define artificial:

```math
A=I,
```

```math
c=-y'(x).
```

Then midpoint and trapezoid defects should converge with expected order.

## 6.2 Physical ODE consistency test

At the N65 midpoint root, compute local ODE slopes:

```math
g_i = -A_i^{-1}c_i
```

away from the sonic point. Then evaluate trapezoid defects **without solving**.

Plot which intervals dominate. If the first few intervals dominate, the issue
is likely sonic-adjacent slope evaluation. If all intervals dominate, the
trapezoid residual is formulated with inconsistent scaling/sign.

## 6.3 Do not use trapezoid as production until it passes

The current trapezoid residuals are too large to guide the solver.

---

# 7. Residual localization is now essential

For every fixed-\(\dot M\) solve, output a per-interval table:

```text
i
R_mid/r_g
dx
interval_R
interval_E
H/R
Qadv/Qvisc
xi_eff
condition(A)
```

and make a plot of:

```text
|interval_R| and |interval_E| versus R.
```

This will answer whether the residual floor is:

```text
sonic-localized
outer-boundary-localized
spread through the disk
associated with one bad interval
```

Current tables only report the max block. We need localization.

---

# 8. Reinterpret the high-rate ladder

The high-rate ladder should be treated as exploratory only.

The N48 scout reaching:

```text
Mdot/Edd ~ 0.7
H/R ~ 0.125
```

is encouraging, but N64 confirmation failures show the branch is not
resolution robust.

Do not use these runs to infer whether the no-wind branch can reach the QPE
target.

Restart high-rate continuation only after:

```text
fixed-Mdot root at 0.9 Edd is mesh-converged
and
the continuation method reproduces it on at least N65 and N129.
```

---

# 9. Recommended immediate Codex sprint

## Task A: residual localization

Add:

```text
outputs/tables/transonic_interval_residual_profile.md
outputs/figures/transonic_interval_residual_profile.png
```

for the N65 fixed root and the failed N67/N73/N129 cases.

## Task B: two-domain prototype

Implement a prototype two-domain outer extension:

```text
inner domain: R_son -> R_match
outer domain: R_match -> R_far
```

Start with small grids:

```text
N_inner = 65
N_outer = 32
R_match = 6500 r_g
R_far = 5e4 r_g
```

Use midpoint collocation only.

## Task C: slope-unknown diagnostic scan

If two-domain is too big for the sprint, scan slope-prior weights:

```text
sigma_g = 1e-4 ... 3e-2
```

and separate physical residual from slope-prior penalty.

## Task D: Hermite prolongation

Implement ODE-slope Hermite prolongation from N65 to N129.

## Task E: trapezoid unit test

Add manufactured ODE tests before using trapezoid in the real solver.

---

# 10. Code-level suggestions

## `transonic_collocation.py`

Add optional two-domain classes:

```python
@dataclass(frozen=True)
class TwoDomainTransonicParams:
    inner: TransonicSlimParams
    R_match_rg: float
    R_far_rg: float
    n_outer: int
```

Add residual builder:

```python
def two_domain_residual(z, params):
    # unpack inner and outer nodes
    # inner intervals
    # outer intervals
    # sonic residuals
    # interface continuity
    # far outer thin BC
```

Add diagnostics:

```python
def interval_residual_profile(z, params):
    ...
```

## `transonic_continuation.py`

Add Hermite remapping:

```python
def hermite_remap_profile(old_profile, old_params, new_logR):
    # compute ODE slopes g at old nodes/midpoints
    # use cubic Hermite in logR
```

Add branch-lock staged refinement:

```python
def staged_refine_root(old_result, new_params):
    # fixed eigenparameters
    # free Rson
    # free lambda0
    # full polish
```

## New scripts

```text
scripts/run_transonic_interval_residual_profile.py
scripts/run_transonic_two_domain_outer_extension.py
scripts/run_transonic_slope_prior_scan.py
scripts/run_transonic_hermite_refinement.py
scripts/run_transonic_trapezoid_unit_audit.py
```

---

# 11. Concrete acceptance criteria

Before returning to high-rate continuation, require:

```text
1. N65 fixed root residual <= 3e-6.
2. N129 fixed root residual <= 3e-6.
3. N65 and N129 agree:
       |Delta Rson| < 0.02 rg
       |Delta lambda0| < 5e-4
       |Delta int_adv| < 5e-3
       profile differences decrease with N.
4. No unexplained branch jump to Rson ~ 5.1.
5. Trapezoid/Hermite scheme tests pass manufactured problems.
6. Residual profile shows no unresolved localized spike.
7. The outer boundary is either two-domain or uses slope priors with small,
   smooth, grid-converged corrections.
```

---

# 12. Compact Codex prompt

```text
The latest results show a good local N65 fixed-Mdot root near Mdot/Edd=0.9028,
R_out=6500 rg, Rson=5.828 rg, lambda0=3.675, int_adv=0.074, residual~2e-6.
But staged resolution continuation fails almost immediately and N129 jumps to
a different basin. The high-rate ladder is therefore premature.

Main diagnosis: the outer slope closure and grid/prolongation strategy still
control the solution. Promoting slopes to unknowns did not actually free them
(dg_u,dg_T~0), and the mixed trapezoid scheme currently fails badly.

Next tasks:
1. Add per-interval residual localization plots.
2. Implement a two-domain outer extension:
      inner domain Rson->Rmatch
      outer domain Rmatch->Rfar
   with interface continuity and far thin boundary, so g_out becomes an output
   rather than an input.
3. If two-domain is too large, redo slope-unknown mode with true soft priors:
      scan sigma_g=1e-4...3e-2
      report physical residual separately from prior penalty.
4. Implement Hermite remapping using ODE slopes, then refine N65->N129.
5. Debug trapezoid on manufactured ODE tests before using it for physics.
6. Pause high-Mdot continuation until the fixed-Mdot root is mesh-converged.
```

---

# 13. Bottom line

The solver has found a credible local fixed-\(\dot M\) root. The bottleneck is
now validation:

```text
Does that local root converge to a continuum transonic solution as the grid and
outer boundary are refined?
```

The answer is not yet. The next step is to remove prescribed outer slopes
through a two-domain outer extension and to establish mesh convergence at
\(\dot M/\dot M_{\rm Edd}\simeq0.9\).
