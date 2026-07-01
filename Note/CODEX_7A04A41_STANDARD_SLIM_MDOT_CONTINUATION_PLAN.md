# Codex Next-Step Brief: Robust Standard Slim-Disk Mdot Continuation After Commit `7a04a41`

Repository reviewed:

```text
https://github.com/huanyang07/IMBH
commit: 7a04a41
```

Start from:

```text
Note/GPT_REVIEW_PROMPT_MDOT_CONTINUATION.md
Note/CODEX_5EEBBEE_STANDARD_SLIM_BENCHMARK_FIX_PLAN.md
outputs/tables/slim_benchmark_rout_injection_ladder.md
outputs/tables/slim_benchmark_mdot_injection_rout10000_onepercent.md
outputs/tables/slim_benchmark_mdot_injection_rout10000_twopercent.md
outputs/tables/slim_benchmark_mdot_injection_rout10000_2pct_ladder.md
scripts/run_standard_slim_mdot_injection_ladder.py
scripts/run_standard_slim_rout_injection_ladder.py
src/imri_qpe/layer3_minidisk_1d/transonic_collocation.py
src/imri_qpe/layer3_minidisk_1d/transonic_local.py
```

## Executive summary

The standard single-BH no-wind slim benchmark is now substantially recovered at:

```text
Mdot/Mdot_Edd = 1e-3
R_out = 10000 rg
N = 128
R_son ~= 5.920 rg
lambda0/lK_ISCO ~= 1.00007
full residual ~= 1.43e-6
```

This is an important success. It means the core transonic machinery can find a
global standard no-wind slim/thin solution in the single-BH benchmark.

This result changes the interpretation of the IMRI/minidisk failure:

```text
The previous IMRI no-wind obstruction is not simply because the solver can
never find a global no-wind slim disk. The solver can recover at least the
standard low-Mdot no-wind benchmark.
```

However, the Mdot continuation remains fragile. Near \(10^{-3}\), one to two
percent steps work, while five to ten percent jumps fail. The failures are
usually dominated by `interval_R`.

The most likely current bottleneck is:

```text
poor Mdot predictor/remapping + insufficient continuation corrector,
not physical branch stiffness.
```

The next implementation sprint should build a proper Mdot continuation method,
not add wind and not return to IMRI yet.

---

# 1. What has been established physically

## 1.1 The standard no-wind branch exists at low Mdot

The recovered \(R_{\rm out}\) ladder shows smooth solutions from:

```text
R_out = 1000 rg to 10000 rg
```

at:

```text
Mdot/Mdot_Edd = 1e-3.
```

The key accepted rows include:

```text
R_out=1000,  N=80,  full=1.98e-6, Rson=5.911 rg
R_out=3000,  N=96,  full=1.31e-6, Rson=5.910 rg
R_out=10000, N=128, full=1.43e-6, Rson=5.920 rg
```

The solution has:

```text
lambda0/lK_ISCO ~= 1
H/R ~= 1.2e-4 at the sonic point
Qadv/Qvisc is tiny
```

So the standard thin/slim low-Mdot benchmark is basically recovered.

## 1.2 What this means for the IMRI program

This does **not** imply that the IMRI no-wind minidisk must also have a global
solution.

It means something more precise:

```text
The basic transonic equations and solver can recover a global standard no-wind
branch in the simplest geometry.
```

Therefore, if the IMRI/minidisk no-wind branch remains blocked after this
benchmark is robust, that obstruction is more likely due to:

```text
- minidisk-specific boundary geometry;
- finite outer reservoir/truncation;
- stress/vertical closure in the minidisk setup;
- missing wind or energy/angular-momentum loss;
- or the stationary no-wind hot branch being physically absent.
```

It is no longer fair to say:

```text
maybe the solver just cannot solve any slim disk.
```

But it is still too early to say:

```text
the IMRI no-wind branch physically fails.
```

because the Mdot continuation benchmark is still fragile.

---

# 2. What the Mdot continuation result says

## 2.1 One-percent and two-percent steps work

At:

```text
R_out = 10000 rg
N = 128
anchor Mdot/Edd = 1e-3
```

one-percent tests pass:

```text
0.00099: full ~= 1.65e-6
0.00101: full ~= 2.52e-6
```

two-percent tests pass as single steps:

```text
0.00098: full ~= 3.46e-6
0.00102: full ~= 4.38e-6
```

The multi-step two-percent ladder works for a few steps:

```text
down:
  0.00098   accepted, full ~= 3.46e-6
  0.0009604 accepted, full ~= 7.58e-6
  0.0009412 failed, full ~= 2.76e-5

up:
  0.00102   accepted, full ~= 4.38e-6
  0.0010404 accepted, full ~= 8.05e-6
  0.0010612 accepted, full ~= 9.81e-6
  0.0010824 accepted, full ~= 8.99e-6
  0.0011041 failed, full ~= 1.10e-5
```

This is a classic sign of a continuation predictor/corrector becoming gradually
less accurate, not of a physical fold.

## 2.2 Five- and ten-percent jumps fail

The review prompt reports that five-percent and ten-percent jumps fail,
usually with dominant `interval_R`.

The one- and two-percent remapped profiles already have large raw residuals:

```text
1 percent remap full ~ 0.012
2 percent remap full ~ 0.025
```

This is enormous compared with the target \(10^{-6}\) to \(10^{-5}\). The
solver then reduces that residual by four orders of magnitude, but eventually
stalls when the step is too large or after repeated steps.

Interpretation:

```text
The current remap is not close enough to the new-Mdot solution for large steps.
```

---

# 3. What is the actual bottleneck?

Ranked diagnosis:

## Primary bottleneck: Mdot remapping / predictor

The evidence:

```text
- remap_full is already ~1e-2 for 1 percent steps;
- injected_full is still ~1e-2 to ~3e-2;
- larger Mdot jumps fail before any physically thick or advective state appears;
- failures are dominated by interval_R, which means the radial momentum/gradient
  structure of the profile is not being predicted correctly.
```

The current predictor is mostly:

```text
interpolate old profile to new grid
scale T by Mdot_factor^0.25
reinject local sonic root
polish
```

That is too crude for a global BVP over four decades in radius.

## Secondary bottleneck: corrector/polish depth

Accepted rows often have:

```text
optimizer_success = False
message = max number of function evaluations exceeded
```

This is okay when the residual is already below tolerance, but it also means
the corrector is working very hard.

The fallback least-squares polish uses only about:

```text
220 evaluations
```

in the reported Mdot ladder. That is probably too small for larger steps.

## Secondary bottleneck: interval_R/collocation conditioning

The failures are dominated by `interval_R`.

That suggests:

```text
the radial momentum part of the collocation residual is the sensitive equation
during Mdot changes.
```

This could be due to:

```text
- derivative mismatch after remap;
- differential residual 1/dx amplification;
- insufficient profile smoothness after interpolation;
- first-order Mdot scaling errors in u and T;
- collocation discretization floor.
```

## Less likely at current Mdot: physical branch stiffness

Near \(10^{-3}\):

```text
H/R is tiny
Qadv/Qvisc is tiny
Omega is nearly Keplerian
lambda0 remains close to lK_ISCO
Rson stays around 5.92 rg
```

This is not yet a physically stiff slim-disk branch. It is still essentially a
thin disk. So the step sensitivity is numerical/continuation-related.

---

# 4. Thin-disk Mdot scalings to improve the predictor

For a low-Mdot, gas-pressure/electron-scattering thin disk with alpha stress,
the approximate Shakura-Sunyaev scalings are:

```text
T_c     ~ Mdot^(2/5)
Sigma   ~ Mdot^(3/5)
H       ~ Mdot^(1/5)
u       ~ Mdot^(2/5)
```

The current temperature scaling:

```text
T -> T * Mdot_factor^0.25
```

is probably too weak in much of the disk.

But fixed analytic power laws are still not the best solution, because the
inner region and outer region can have different regimes.

Recommended predictor hierarchy:

```text
1. local algebraic thin predictor at every radius;
2. linearized BVP tangent predictor;
3. adaptive continuation.
```

---

# 5. Best next implementation sprint

The best next sprint is:

```text
Mdot tangent predictor + adaptive Mdot continuation + residual localization.
```

Do not implement wind yet.

---

# 6. Sprint A: residual localization for failed Mdot steps

Before changing the solver, localize the residual.

Create:

```text
scripts/run_standard_slim_mdot_residual_profile.py
```

Run for:

```text
Mdot/Edd = 0.0009412 failed down step
Mdot/Edd = 0.0011041 failed up step
Mdot/Edd = 0.00102 accepted up step
Mdot/Edd = 0.00098 accepted down step
```

Output:

```text
outputs/tables/slim_benchmark_mdot_residual_profiles.md
outputs/figures/slim_benchmark_mdot_residual_profiles.png
```

For each interval report:

```text
R_mid/rg
interval_R
interval_E
Delta logu relative to anchor
Delta logT relative to anchor
Omega/OmegaK - 1
pressure-support correction
Qadv/Qvisc
H/R
condition(A)
smin/smax(A)
```

Goal:

```text
Find whether interval_R failure is localized near the sonic region, outer
domain, or distributed over the disk.
```

This tells us whether to fix remap globally or only near a region.

---

# 7. Sprint B: build better predictors

## 7.1 Predictor 1: local algebraic thin predictor

At fixed \(R\), for the new Mdot solve the local thin equations:

```math
\dot M(l_K-l_0)=2\pi R^2 W,
```

```math
Q_{\rm visc,thin}=Q_{\rm rad},
```

with continuity:

```math
u=\frac{\dot M}{2\pi R\Sigma}.
```

Use this to predict:

```text
logu_new
logT_new
```

at every radial node.

Keep:

```text
Rson_new = Rson_old
lambda0_new = lambda0_old
```

as first guess, then sonic-root inject.

This predictor should dramatically reduce remap_full relative to the current
0.012--0.025 values.

## 7.2 Predictor 2: linearized BVP tangent predictor

This is the main robust method.

Let:

```math
F(z,\mu)=0,
qquad
\mu=\ln\dot M.
```

At an accepted solution:

```math
J_z \frac{dz}{d\mu} = -F_\mu.
```

Compute:

```python
dz_dmu = solve(J_z, -F_mu)
```

where:

```text
J_z = square collocation Jacobian at the accepted root
F_mu = finite-difference derivative of square residual with respect to log Mdot
```

Then predict:

```math
z_{\rm pred} =
z_k + \Delta\mu \frac{dz}{d\mu}.
```

Use this for the next Mdot.

Diagnostics:

```text
old remap_full
thin_predictor_full
tangent_predictor_full
```

Expected:

```text
tangent_predictor_full << old remap_full
```

This should allow 5--10 percent steps if the branch is smooth.

## 7.3 Predictor 3: secant predictor

After two accepted solutions:

```math
z_{\rm pred} =
z_k +
\frac{\mu_{k+1}-\mu_k}{\mu_k-\mu_{k-1}}
(z_k-z_{k-1}).
```

Use this only if the tangent predictor is not available.

## 7.4 Predictor comparison table

Create:

```text
outputs/tables/slim_benchmark_mdot_predictor_audit.md
```

Rows:

```text
target Mdot
predictor type
initial residual
dominant initial block
final residual after fixed polish
nfev
accepted
```

Test targets:

```text
1.01e-3
1.02e-3
1.05e-3
1.10e-3
0.99e-3
0.98e-3
0.95e-3
0.90e-3
```

---

# 8. Sprint C: adaptive Mdot step controller

Once the tangent predictor is implemented, use an adaptive controller in
\(\mu=\ln\dot M\).

## 8.1 Step control

Initial step:

```text
Delta_mu = 0.01
```

Rules:

```text
if predictor residual < 1e-3 and final residual < 3e-6:
    Delta_mu *= 1.5

if final residual < 1e-5:
    accept

if final residual > 1e-5:
    reject, Delta_mu *= 0.5, retry

if dominant interval_R and line search stalls:
    reject, Delta_mu *= 0.5, try integrated-defect pre-polish

if accepted but residual > 5e-6:
    mark scout, but do not use as tangent anchor until polished
```

Maximum step near the thin branch:

```text
Delta_mu <= 0.1
```

but let the controller decide.

## 8.2 Anchor versus scout points

Define:

```text
anchor:
    residual <= 3e-6
    D,C1,C2,K <= few e-6
    optimizer or Newton polish acceptable
    can be used to compute next tangent

scout:
    residual <= 1e-5
    can be plotted
    cannot be used as tangent anchor unless polished
```

This avoids contaminating the ladder with marginal points.

## 8.3 Target sequence

Do not predefine a rigid list. Instead set final goals:

```text
down target = 1e-4
up target   = 1e-2
```

Let the adaptive stepper generate intermediate points.

---

# 9. Sprint D: improve the corrector

## 9.1 Use integrated-defect pre-polish

For each target Mdot:

```text
Step 1: integrated residual solve
Step 2: differential residual polish
Step 3: physical audit
```

The integrated residual helps condition radial interval equations; the
differential residual remains the science metric.

## 9.2 Increase evaluation limits

For benchmark validation use:

```text
SOURCE_POLISH_NFEV = 2000
POLISH_NFEV = 4000
FALLBACK_LSQ_NFEV = 1000
```

The current 220 fallback evaluations are useful for quick tests, but not for a
production continuation benchmark.

## 9.3 Use block Jacobian by default

Set:

```text
use_block_jacobian = True
```

for square polish and LSQ fallback.

## 9.4 Add condition diagnostics

For each polish report:

```text
condition estimate of square Jacobian
line-search reductions
Newton iterations
least-squares nfev
final dominant block
```

---

# 10. Sprint E: two-parameter continuation in \(R_{\rm out}\) and Mdot

There are two possible routes to \(10^{-2}\) and \(10^{-4}\):

## Route 1: fixed \(R_{\rm out}=10000r_g\) Mdot ladder

This is the current route.

Pros:

```text
tests the hardest final configuration directly.
```

Cons:

```text
N=128 and huge radial span make continuation sensitive.
```

## Route 2: solve Mdot at smaller \(R_{\rm out}\), then extend \(R_{\rm out}\)

Recommended robust route:

```text
For each target Mdot:
    solve at R_out = 1000 or 2000 rg
    extend R_out through 3000, 5000, 7000, 10000
```

This uses the already successful \(R_{\rm out}\) injection ladder strategy.

Implementation:

```text
For upward Mdot continuation:
    at R_out=1000, continue Mdot to 1e-2
    for selected Mdot checkpoints, extend R_out to 10000

For downward Mdot continuation:
    at R_out=1000, continue Mdot to 1e-4
    then extend selected checkpoints to R_out=10000
```

This should be much more robust and will tell whether failures are caused by
Mdot itself or by large-radius remapping at full \(R_{\rm out}\).

---

# 11. Recommended immediate experiments

## Experiment 1: predictor audit at fixed \(R_{\rm out}=10000\)

Run:

```text
target Mdot = 1.02e-3, 1.05e-3, 1.10e-3
target Mdot = 0.98e-3, 0.95e-3, 0.90e-3
```

Compare:

```text
current remap
thin algebraic predictor
linear tangent predictor
```

Goal:

```text
reduce initial residual from ~1e-2 to <=1e-3, ideally <=1e-4.
```

## Experiment 2: adaptive fixed-\(R_{\rm out}\) ladder

Use tangent predictor and adaptive controller.

Goal:

```text
reach 1e-2 upward and 1e-4 downward with anchor residual <=3e-6.
```

## Experiment 3: small-\(R_{\rm out}\) Mdot ladder

Run at:

```text
R_out = 1000 rg
N = 80
```

Goal:

```text
test whether Mdot ladder can reach 1e-2 and 1e-4 more easily.
```

Then extend selected target states to \(R_{\rm out}=10000r_g\).

## Experiment 4: interval_R residual localization

Run on failed and accepted cases.

Goal:

```text
distinguish global remap error from local sonic or outer problem.
```

---

# 12. What would count as success?

## Near-term success

```text
from 1e-3 to 3e-3 and 3e-4
with residual <=3e-6 anchors.
```

## Medium-term success

```text
full ladder from 1e-4 to 1e-2
at R_out=10000
with no step-size pathology.
```

## Benchmark success

```text
standard no-wind slim disk sequence:
    1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1
```

Before moving past:

```text
Mdot/Edd ~ 0.1
```

the code should verify that advection and H/R behave smoothly.

---

# 13. What does this mean for the IMRI no-wind question?

The fact that the standard benchmark is recovered at \(10^{-3}\) means:

```text
The transonic formulation is not fundamentally incapable of finding a global
no-wind solution.
```

But it does **not** mean:

```text
the IMRI no-wind branch must exist.
```

The IMRI problem has additional constraints:

```text
finite minidisk outer boundary
stream/tidal geometry
different thermal reservoir
possible super-Eddington high state
```

The correct conclusion is:

```text
The previous IMRI no-wind obstruction is not fully trustworthy until the
standard benchmark Mdot ladder is robust.
```

If the standard benchmark reaches higher Mdot smoothly but the IMRI branch
still folds, then the IMRI no-wind obstruction is likely physical/boundary
related.

If the standard benchmark also develops similar folds as Mdot increases, then
the issue is still in the standard closure or transonic implementation.

---

# 14. Concrete code changes

## `scripts/run_standard_slim_mdot_predictor_audit.py`

New script.

Inputs:

```text
anchor checkpoint
target Mdot list
predictor types
```

Outputs:

```text
outputs/tables/slim_benchmark_mdot_predictor_audit.md
outputs/figures/slim_benchmark_mdot_predictor_audit.png
```

## `scripts/run_standard_slim_adaptive_mdot_ladder.py`

New production ladder.

Features:

```text
adaptive Delta_mu
tangent predictor
anchor/scout distinction
rollback on rejection
integrated pre-polish option
```

Outputs:

```text
outputs/tables/slim_benchmark_adaptive_mdot_ladder.md
outputs/figures/slim_benchmark_adaptive_mdot_ladder.png
```

## `scripts/run_standard_slim_mdot_rout_surface.py`

Two-parameter continuation.

Outputs:

```text
outputs/tables/slim_benchmark_mdot_rout_surface.md
outputs/figures/slim_benchmark_mdot_rout_surface.png
```

## `scripts/run_standard_slim_mdot_residual_profile.py`

Residual localization.

Outputs:

```text
outputs/tables/slim_benchmark_mdot_residual_profiles.md
outputs/figures/slim_benchmark_mdot_residual_profiles.png
```

---

# 15. Core implementation details

## 15.1 Tangent predictor

Add helper:

```python
def mdot_tangent_predictor(z_anchor, params_anchor, params_target, pivot="C2"):
    mu0 = np.log(params_anchor.Mdot_g_s)
    mu1 = np.log(params_target.Mdot_g_s)
    dmu = mu1 - mu0

    Fz = square_collocation_jacobian(z_anchor, params_anchor, pivot=pivot)
    Fmu = finite_difference_mdot_column(z_anchor, params_anchor, pivot=pivot)

    dz_dmu = solve(Fz, -Fmu)
    return z_anchor + dmu * dz_dmu
```

Use finite-difference step:

```text
h_mu = 1e-4, 3e-5, 1e-5
```

and audit directional error.

## 15.2 Thin algebraic predictor

Add helper:

```python
def thin_algebraic_mdot_remap(profile_old, params_target):
    # use same R grid
    # solve local W_req and Qrad=Qvisc_thin
    # return logu, logT
```

This provides a non-Newton baseline predictor.

## 15.3 Adaptive controller

Pseudo-code:

```python
while mu not at target:
    z_pred = tangent_predictor(anchor, target_mu)
    z_polished = polish(z_pred)

    if residual <= anchor_tol:
        accept as anchor
        maybe increase step
    elif residual <= scout_tol:
        accept as scout but do not tangent from it
        reduce next step
    else:
        reject
        reduce step
```

---

# 16. Do not do next

Do not yet:

```text
add wind
return to IMRI high-Mdot branch
use the 1e-2 failed row as physical evidence
jump Mdot by 5--10 percent without a tangent predictor
declare the benchmark complete after only 1e-3
```

---

# 17. Compact Codex prompt

```text
The standard slim benchmark is now recovered at Mdot/Edd=1e-3 out to
Rout=10000 rg: residual~1.43e-6, Rson~5.92 rg, lambda0/lK~1.00007. This means
the core no-wind transonic solver can find a global standard solution.

The remaining problem is Mdot continuation. One- and two-percent steps near
1e-3 pass, but repeated two-percent steps and 5--10 percent jumps fail,
usually dominated by interval_R. The disk is still thin and non-advective, so
this is likely predictor/corrector/collocation sensitivity, not physical branch
stiffness.

Implement:
1. residual localization for failed and accepted Mdot steps;
2. thin algebraic local predictor;
3. linearized BVP tangent predictor dz/dlnMdot = -Jz^{-1} F_mu;
4. adaptive log-Mdot step controller with anchor/scout states;
5. integrated-defect pre-polish followed by differential residual audit;
6. larger nfev and block Jacobian by default;
7. two-parameter strategy: first continue Mdot at Rout=1000, then extend Rout
   to 10000 for selected checkpoints.

Goal:
robust ladder from 1e-4 to 1e-2 before returning to IMRI or wind.
```

---

# 18. Bottom line

The current bottleneck is not physical branch stiffness.

It is:

```text
Mdot predictor/remapping and continuation globalization.
```

Fix that first.
