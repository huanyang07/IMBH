# Codex Handoff: Re-evaluation at commit `118bcf2`

**Repository:** `huanyang07/IMBH`
**Commit:** `118bcf2` — `Add Mdot5 eta continuation diagnostics`
**Main notes:**

- `Note/CODEX_MDOT5_ADAPTIVE_ACTIVE_WINDOW_ETA_CONTINUATION_RESULTS.md`
- `Note/GPT_PROMPT_MDOT5_TWO_PASS_SONIC_WALL.md`
- `Note/CODEX_MDOT5_ETA_TANGENT_CONTINUATION_RESULTS.md`
- `Note/CODEX_SOURCE_BAND_PRODUCTION_HSFV_RESULTS.md`

## 1. Updated status

The previous plan that focused on an adaptive active-window corrector is now partly implemented and partly superseded by the new results in commit `118bcf2`.

The important update is:

```text
The source-band HS/FV replacement formulation is no longer the immediate blocker.
The eta_E continuation has now reached eta_E = 98.25 under the compatible
source-band global-replacement score.
The next non-strict point is eta_E = 98.21875.
```

The latest strict checkpoint is:

```text
outputs/checkpoints/m5_eta_two_pass_sonic12_from98p4375_N164/
    stage_03_etaE_98p25_N164.npz
```

The best strict sequence now includes:

```text
eta_E = 98.75      score = 9.032989e-06
eta_E = 98.6875    score = 9.419567e-06
eta_E = 98.65625   score = 9.809904e-06
eta_E = 98.625     score = 9.495741e-06
eta_E = 98.59375   score = 8.961460e-06
eta_E = 98.5625    score = 9.327786e-06
eta_E = 98.50      score = 9.414047e-06
eta_E = 98.46875   score = 9.716850e-06
eta_E = 98.4375    score = 9.189044e-06
eta_E = 98.40625   score = 9.044458e-06
eta_E = 98.375     score = 9.099071e-06
eta_E = 98.3125    score = 9.221698e-06
eta_E = 98.25      score = 9.857180e-06
```

The next attempted point is slightly non-strict:

```text
eta_E = 98.21875
score = 1.018509e-05
leading rows:
    old_sonic_pivot / R ~ 5.30 rg
    old_mass        / R ~ 5.93 rg
mass increment rows:
    active_mass_increment_int  ~ 9.408090e-06
    active_mass_increment_link ~ 9.408090e-06
```

So the new wall is *very close* to strict, but the pattern matters: the leading residuals are now an inner sonic/mass pair, not source-band HS/FV rows and not mass-increment bookkeeping.

## 2. Updated interpretation

The project has moved through several numerical bottlenecks:

```text
Solved / mostly solved:
    eta_E = 100 source-band HS/FV production replacement;
    direct FV mass replacement;
    mass-increment split rows;
    balanced DeltaM initialization;
    broad eta tangent predictor in mu = 1 / eta_E;
    row localization of old_kind = other -> old_sonic_pivot;
    sonic-local and two-pass sonic correction.

Current blocker:
    coupled inner sonic + old_mass residual floor near R ~ 5.3--6 rg.
```

This is not convincing evidence of physical branch loss. It is also not a compact source-annulus defect. The strictness statements here are for the **compatible source-band global-replacement score**, not the legacy midpoint `final_full`, which remains large for these replacement-formulation runs.

The two-pass sonic prepass is useful, but its latest variants suggest the staged formulation is close to exhausted:

```text
soft sonic anchor 1e-3: unchanged;
wider sonic prepass R < 12 rg: unchanged;
prepass + inner old_mass R < 8 rg: worse;
prepass + inner old_mass R < 15 rg: worse.
```

That failure mode is informative. It says simply adding mass rows into the sonic prepass gives a poor search direction. The likely fix is not a broader prepass; it is a **single coupled inner-window solve** with sonic rows, nearby mass rows, radial/energy rows, and mass-increment guard rows all balanced simultaneously.

## 3. Main recommendation

Replace the staged

```text
sonic prepass -> adaptive active-window corrector
```

with a **single coupled inner-window least-squares corrector** for the current wall.

The coupled corrector should target the actual residual pair at the wall:

```text
old_sonic_pivot at R ~ 5.30 rg
old_mass        at R ~ 5.93 rg
```

but it must also include neighboring radial/energy rows so the solver does not fix mass by creating a radial/energy floor.

## 4. Cheap diagnostic before the larger implementation

Before writing the full coupled corrector, run one cheap microstep test from the strict `eta_E=98.25` checkpoint:

```text
98.25 -> 98.234375 -> 98.21875
```

using the existing two-pass sonic12 setup.

If this passes, the `98.21875` wall was mostly a predictor/step-size artifact. Still keep the coupled corrector plan, because the same coupled sonic/mass floor is likely to reappear at lower `eta_E`.

If this does not pass, proceed directly to the coupled inner-window corrector.

## 5. Coupled inner-window corrector formulation

### 5.1 Unknowns

Use a local window initially spanning:

```text
R = 4.8--7.2 rg
```

If the peak row moves outward, expand gradually:

```text
R = 4.8--8.5 rg
R = 4.8--10 rg
```

Release variables in the window:

```text
logu_i
logT_i
logMdot_i
```

Also release the sonic/global variables already used by the sonic prepass:

```text
logRson or equivalent sonic-location variable
lambda0 / eigenvalue / angular-momentum eigenvalue variable, if present
```

Do **not** release source-band HS/FV aux arrays as primary variables in this inner-window corrector unless the window touches the source-band replacement region. They should be refreshed after accepted updates.

### 5.2 Active rows

The local least-squares residual should include these row groups:

```text
G_sonic:
    old_sonic_pivot
    old_sonic_D
    old_inner_mdot if relevant

G_mass:
    old_mass rows from the first few intervals outside the sonic point,
    including the R ~ 5.93 rg row and neighboring mass rows.

G_RE:
    adjacent old_interval_radial rows
    adjacent old_interval_energy rows

G_boundary:
    weak edge anchors at both ends of the local window
    optional first-difference or second-difference smoothness priors

G_source_guard:
    mass_increment_int/link rows as guard rows, not dominant target rows
    source-band implicit/midpoint/Simpson/interface rows as evaluate-only or soft guards
```

Do not optimize old_mass alone. Earlier tests showed mass-only patches move the defect outward rather than solving the coupled response.

### 5.3 Row scaling / weighting

Use group-balanced scaling. A useful first choice:

```text
scale_sonic = max(current_G_sonic, 3e-6)
scale_mass  = max(current_G_mass,  3e-6)
scale_RE    = max(current_G_RE,    3e-6)
scale_guard = max(current_G_guard, 5e-6)
```

Then minimize scaled rows, but accept/reject using the unscaled compatible source-band score.

Recommended group weights:

```text
w_sonic = 1.0
w_mass  = 1.0
w_RE    = 0.3--1.0
w_boundary_anchor = 1e-3 to 1e-2 initially
w_source_guard = 0.3 initially, raised only if guard rows approach tolerance
```

Use a filter, not only a scalar merit:

```text
Accept if:
    compatible source-band score decreases or becomes <= 1e-5;
    old_sonic_pivot <= 1e-5;
    local old_mass peak <= 1e-5;
    mass_increment_int/link remain <= 1e-5;
    source-band HS/FV guard rows do not degrade above tolerance;
    physical diagnostics are smooth.
```

A step that improves sonic but worsens mass, or improves mass but worsens sonic, should be rejected unless the total compatible score becomes clearly strict and the other group remains below tolerance.

### 5.4 Variable scaling / trust region

Start with a conservative trust ladder:

```text
max |delta logu|     = 3e-4, 1e-3, 3e-3, 1e-2
max |delta logT|     = 3e-4, 1e-3, 3e-3, 1e-2
max |delta logMdot|  = 3e-4, 1e-3, 3e-3, 1e-2
max |delta logRson|  = 1e-5, 3e-5, 1e-4, 3e-4
```

Let the line search decide the final step length. Report peak state changes and their radii.

## 6. Better sonic variable basis

The repeated `old_sonic_pivot` floor suggests that the current variables may be poorly conditioned near the sonic point. Add a diagnostic and, if cheap, a local basis option.

### 6.1 Sonic sensitivity audit

At the `eta_E=98.21875` wall, compute the local Jacobian block:

```text
rows:
    old_sonic_pivot
    old_sonic_D
    old_inner_mdot
    first 2--4 old_mass rows
    adjacent radial/energy rows

columns:
    logRson
    lambda0 / eigenvalue variable
    logu/logT/logMdot nodes in R = 4.8--7.2 rg
```

Output:

```text
singular values
right singular vectors
row-scaled condition number
column-scaled condition number
sensitivity of old_sonic_pivot to logRson and lambda0
sensitivity of old_mass 5.93 rg to logRson and lambda0
```

### 6.2 Sonic tangent basis

If the local Jacobian is ill-conditioned, introduce a reduced sonic tangent coordinate:

```text
q_sonic = local tangent direction that decreases sonic_pivot and sonic_D
q_mass  = local tangent direction that decreases the inner old_mass row
```

Then solve a 2D or low-dimensional subspace problem:

```text
state_trial = state + a q_sonic + b q_mass
```

with a two-dimensional line search over `(a,b)`. This is a cheap diagnostic and may be enough to cross `eta_E=98.21875` before building a full local Newton solver.

## 7. Do not switch acceptance metrics yet

Continue to use the compatible source-band global-replacement score for this eta ladder. Do not switch acceptance to legacy `final_full`, because the legacy source-band midpoint rows have been intentionally replaced.

However, keep evaluate-only audits at every accepted checkpoint:

```text
compatible source-band score
legacy final_full
old midpoint source audit
mass_increment_int/link
implicit_ode
midpoint
simpson
interface
FV energy audit
FV angular-momentum audit, if available
```

Also keep the caveat that full unit-weight HS/FV collocation (`SOURCE_BAND_CHI_IMPL=1`) is not yet the acceptance view for this eta continuation.

## 8. Continuation after fixing eta_E=98.21875

Once `eta_E=98.21875` becomes strict, continue with small steps:

```text
98.21875 -> 98.1875 -> 98.15625 -> 98.125 -> 98.0625 -> 98.0
```

Use the coupled inner-window corrector when the leading row is sonic/mass near the inner region.

Step controller:

```text
if score <= 8e-6 for 3 consecutive steps:
    allow mild growth in dmu or eta spacing

if score is 8e-6--1e-5:
    keep step size

if score > 1e-5:
    retry with smaller eta step or coupled inner-window corrector

if dmu collapses below ~1e-6 repeatedly:
    stop; do not force continuation; consider pseudo-arclength or revised sonic coordinate
```

Do not jump to `eta_E=95` or `eta_E=90` until the `98.x` ladder is boring.

## 9. Validation timing

Do not start a full N192/N224 validation campaign yet. First make the N164 continuation robust through at least:

```text
eta_E = 98.0
```

A limited spot check is reasonable at:

```text
eta_E = 98.25
eta_E = 98.0, if reached
```

Use it only to catch a gross N164 artifact, not to certify the full branch.

## 10. Acceptance criteria

For the next strict checkpoint at `eta_E=98.21875`:

```text
compatible source-band score <= 1e-5
old_sonic_pivot <= 1e-5
leading old_mass row <= 1e-5
mass_increment_int <= 1e-5
mass_increment_link <= 1e-5
implicit_ode <= 3e-6 preferred, <=1e-5 acceptable
midpoint <= 3e-6 preferred, <=1e-5 acceptable
simpson <= 3e-6 preferred, <=1e-5 acceptable
interface <= 1e-6--1e-5
```

Physical smoothness:

```text
Mdot_outer/Mdot_inner change <= 1e-4--3e-4 per small step
Lrad/LEdd smooth
Rson change <= 1e-3--1e-2 rg
f_adv_global/f_adv_inner smooth
max H/R smooth
no new source-band defect
no new outer-buffer defect
```

## 11. If the coupled corrector fails

If the coupled inner-window corrector cannot make `eta_E=98.21875` strict, run these diagnostics before changing physics:

```text
1. Half-step eta ladder from 98.25 to 98.21875.
2. Two-vector subspace solve using sonic-prepass direction plus inner-mass direction.
3. Sonic sensitivity / singular-vector audit.
4. Slightly different window boundaries:
       4.8--6.8 rg
       4.8--7.5 rg
       4.8--8.5 rg
5. Different row scaling of sonic vs mass groups.
```

Only if all of these fail and the residual remains anchored at the same sonic/mass pair should Codex consider a more invasive sonic-following inner micro-domain or pseudo-arclength continuation.

## 12. Paste-ready Codex prompt

```text
Please review commit 118bcf2 and implement the next step for the Mdot=5 local-Mdot eta continuation.

Relevant notes:
- Note/CODEX_MDOT5_ADAPTIVE_ACTIVE_WINDOW_ETA_CONTINUATION_RESULTS.md
- Note/GPT_PROMPT_MDOT5_TWO_PASS_SONIC_WALL.md
- Note/CODEX_MDOT5_ETA_TANGENT_CONTINUATION_RESULTS.md
- Note/CODEX_SOURCE_BAND_PRODUCTION_HSFV_RESULTS.md

Current model:
- Mdot_inner/Edd = 5
- Rout = 335 rg
- Rinj = 240 rg
- f_s = 0.80
- compact-C2 stream source
- torque_delta_l_fraction = +0.005
- local-Mdot mass-loaded wind
- N = 164
- eta continuation in mu = 1 / eta_E

Current strict checkpoint:
outputs/checkpoints/m5_eta_two_pass_sonic12_from98p4375_N164/stage_03_etaE_98p25_N164.npz

Current result:
- Source-band HS/FV replacement is not the immediate blocker.
- The compatible source-band global-replacement ladder reaches eta_E = 98.25.
- Next point eta_E = 98.21875 is slightly non-strict:
    score = 1.018509e-05
    leading rows:
        old_sonic_pivot / R ~ 5.30 rg
        old_mass        / R ~ 5.93 rg
    mass_increment_int/link ~ 9.408090e-06, still below tolerance.
- Variants tried:
    softer sonic anchor 1e-3: unchanged;
    wider sonic prepass R < 12 rg: unchanged;
    prepass including inner old_mass R < 8 rg: worse;
    prepass including inner old_mass R < 15 rg: worse.

Interpretation:
- This is not a source-band HS/FV defect.
- This is not finite-volume mass-increment bookkeeping.
- This is not physical branch loss.
- The current wall is a coupled inner sonic + old_mass residual floor.
- The staged sonic prepass + adaptive corrector is close to exhausted.

Requested next implementation:
Build a single coupled inner-window least-squares corrector, replacing the staged sonic prepass at the current wall.

Initial window:
    R = 4.8--7.2 rg
expand if needed to:
    4.8--8.5 rg or 4.8--10 rg

Released variables:
    logu_i, logT_i, logMdot_i inside the window;
    logRson / sonic-location variable;
    lambda0 or equivalent eigenvalue variable if present.

Active row groups:
    G_sonic:
        old_sonic_pivot
        old_sonic_D
        old_inner_mdot if relevant
    G_mass:
        old_mass row at R ~ 5.93 rg and neighboring old_mass rows
    G_RE:
        adjacent old_interval_radial rows
        adjacent old_interval_energy rows
    G_boundary:
        weak edge anchors and optional smoothness priors
    G_source_guard:
        mass_increment_int/link as guard rows
        source-band implicit/midpoint/Simpson/interface rows as soft guards

Do not optimize mass-only. Prior mass-only patches move the defect outward.

Use group-balanced scaling:
    w_sonic = 1
    w_mass = 1
    w_RE = 0.3--1
    w_boundary_anchor = 1e-3--1e-2
    w_source_guard = 0.3 initially

Use a filter acceptance rule:
    compatible source-band score <= 1e-5;
    old_sonic_pivot <= 1e-5;
    leading old_mass <= 1e-5;
    mass_increment_int/link <= 1e-5;
    source-band HS/FV guard rows remain strict;
    physical diagnostics smooth.

Before full implementation, run a cheap half-step diagnostic:
    98.25 -> 98.234375 -> 98.21875
using existing two-pass sonic12 settings.

If that passes, continue with smaller steps but still keep the coupled corrector for later walls.
If it fails, use the coupled inner-window corrector.

Add a sonic sensitivity audit:
    rows = sonic_pivot, sonic_D, inner_mdot, first old_mass rows, adjacent radial/energy rows
    columns = logRson, lambda0, logu/logT/logMdot nodes in 4.8--7.2 rg
    output singular values, scaled condition number, and leading right singular vectors.

Optionally add a 2D subspace diagnostic:
    q_sonic = direction from sonic prepass
    q_mass = direction from inner-mass correction
    solve state_trial = state + a q_sonic + b q_mass
with a 2D line search on compatible source-band score.

After eta_E=98.21875 is strict, continue:
    98.21875 -> 98.1875 -> 98.15625 -> 98.125 -> 98.0625 -> 98.0
Do not jump to eta_E=95 or 90 yet.
Do not start full N192/N224 validation until N164 reaches at least eta_E~98.0 robustly.

Deliverables:
- Note/CODEX_MDOT5_COUPLED_INNER_WINDOW_ETA_CONTINUATION_RESULTS.md
- JSON/MD tables for:
    half-step diagnostic,
    coupled inner-window corrector,
    sonic sensitivity audit,
    optional two-vector subspace diagnostic,
    continued eta ladder.
```

## Bottom line

Commit `118bcf2` is better than my previous assessment: Codex has already implemented the adaptive active-window strategy and advanced the strict compatible ladder to `eta_E=98.25`. The new wall is narrow and specific: a coupled sonic-pivot plus first inner old-mass residual at `eta_E=98.21875`. The next fix should be a single coupled inner-window least-squares corrector with group-balanced scaling, not more staged sonic prepass variants and not more source-band formulation work.
