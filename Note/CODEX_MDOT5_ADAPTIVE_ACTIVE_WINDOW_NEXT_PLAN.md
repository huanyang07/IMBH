# Codex Handoff: Mdot=5 Eta Continuation After Source-Band HS/FV Replacement

Date: 2026-07-08
Repository: `huanyang07/IMBH`
Latest visible GitHub commit reviewed: `f05e4d4` — `Add source-band replacement diagnostics`

## Executive summary

The project has moved past the original source-band formulation wall. The `eta_E=100` source-band HS/FV replacement formulation is now strict enough to serve as the production anchor. The latest continuation has reached a strict `eta_E=98.75` checkpoint at `N=164` under the compatible identity-aware source-band replacement score.

The current blocker is **not** the compact source annulus, and it is **not** the HS/FV source-band residual. The current blocker is an **adaptive eta-continuation / outside-old mass-profile correction problem**.

The residual limiter now alternates between:

```text
old_sonic_pivot near R ~ 5.30 rg
old_mass near R ~ 7--9 rg
sometimes old_mass near R ~ 125 rg
```

The latest strict endpoint is:

```text
eta_E = 98.75
N = 164
active source-band replacement score ~= 9.03e-6
top active row = old_mass / R ~ 8.56 rg
```

The next move should be:

> Implement an adaptive active-window corrector driven by the actual top active row, including adjacent mass/radial/energy rows, with source-band aux refresh after every accepted update. Continue in small steps of `mu = 1/eta_E`. Do not jump to `eta_E=95` or `eta_E=90` yet.

---

## Current status

### 1. Source-band HS/FV replacement is working at `eta_E=100`

The source-band production formulation now uses:

```text
finite-volume mass / mass-increment rows
implicit Hermite-Simpson radial/energy rows
midpoint state variables
auxiliary slopes
checkpoint-persisted source-band auxiliary arrays
old midpoint source rows as audit only
```

The best `eta_E=100` production anchor is:

```text
outputs/checkpoints/m5_source_band_freezeaux_polish16_eta100_N164/
    stage_00_etaE_100_N164.npz
```

Representative diagnostics:

```text
final_score              = 4.202018e-6
outside_old_energy       = 4.202018e-6 near R ~ 31.93 rg
mass_increment_int       = 3.802756e-6
mass_increment_link      = 3.881196e-6
implicit_ode             = 9.537768e-8
midpoint                 = 1.323996e-7
simpson                  = 1.317595e-7
```

Interpretation:

```text
source-band HS/FV rows are no longer the immediate numerical blocker.
eta_E=100 is a usable identity-aware production anchor.
legacy old source rows should remain audit-only inside the replacement band.
```

### 2. Eta tangent predictor works for the broad response

Continuation now uses:

```text
mu = 1 / eta_E
```

with a tangent predictor:

```text
F_mu ~= [F(x, mu + dmu_fd) - F(x, mu)] / dmu_fd
J_x t = -F_mu
x_pred = x + dmu * t
```

The predictor includes global physical disk variables:

```text
logu
logT
logMdot
```

and keeps source-band auxiliary variables frozen for the tangent step.

This is much better than direct eta stepping:

```text
eta_E=99.9 raw score      ~ 1.48e-5
eta_E=99.9 tangent score  ~ 6.87e-6
outside mass after tangent ~ 1e-7
```

### 3. Sonic pivot localization and correction helped

The previously mysterious `old_kind=other` row at `R ~ 5.30 rg` is now correctly identified as:

```text
old_sonic_pivot
```

The sonic-local correction removes that row cheaply. With tangent plus sonic relax, the strict ladder extended from:

```text
eta_E = 99.85
```

to:

```text
eta_E = 99.65
```

After that, the limiter becomes active `old_mass` near the inner grid.

### 4. Balanced mass increments helped

The new default:

```text
SOURCE_BAND_MASS_INCREMENT_INIT=balanced
```

removes the false `active_mass_increment_link` wall near `R ~ 157 rg` without making the integral row large.

This was important bookkeeping. The previous wall near `eta_E ~ 99.45` was partly auxiliary initialization, not physics.

### 5. Latest strict endpoint

With forced sonic-local correction and balanced mass increments, the latest strict sequence reaches:

```text
eta_E = 98.75
N = 164
active score = 9.032989e-6
top active row = old_mass / R ~ 8.56 rg
```

This is real progress. But it is still not enough to claim a robust continuation to `eta_E=95` or `eta_E=90`.

---

## Updated diagnosis

The project is now in a different numerical regime than before.

### Solved enough for now

```text
source-band row replacement
explicit g=-A^{-1}c instability avoided
mass-increment split rows
checkpoint persistence of HS/FV aux variables
eta_E=100 identity-aware production anchor
broad eta tangent predictor
sonic pivot localization
balanced mass-increment initialization
```

### Current main problem

```text
adaptive coupled outside-old mass-profile correction during eta continuation
```

The key evidence is:

```text
mass-only local patches move the defect outward instead of solving it;
wider inner relax creates radial floors;
global damped mass-profile predictor creates large radial residuals;
block corrector helps only slightly when it targets mass alone;
sonic-local window fixes sonic rows but exposes neighboring old_mass rows.
```

Therefore:

> The limiter is a coupled state-response problem. The correction must include old mass rows together with adjacent radial/energy rows and sonic rows when relevant. It should not be mass-only.

---

## What not to do next

Do **not** do these yet:

```text
do not jump directly to eta_E=95 or eta_E=90;
do not lower eta_E using mass-only prediction;
do not re-enable old midpoint source rows as production rows;
do not keep doing source-band HS/FV surgery unless the source-band rows become the dominant defect again;
do not use legacy final_full as the acceptance metric for replacement-formulation experiments;
do not claim physical wind-branch certification from N=164 alone.
```

---

## Recommended next plan

## Phase 1 — Freeze anchors

Freeze the following as named regression states:

```text
A. eta_E=100 source-band HS/FV anchor
   outputs/checkpoints/m5_source_band_freezeaux_polish16_eta100_N164/
       stage_00_etaE_100_N164.npz

B. eta_E=99.65 tangent + sonic-relax checkpoint
   last strict pre-balanced/sonic local ladder state

C. eta_E=98.75 latest strict endpoint
   forced sonic-local, 5--10 rg, balanced mass increments
```

For each anchor, report:

```text
active source-band replacement score
top active rows
outside_old_mass
outside_old_energy
mass_increment_int
mass_increment_link
implicit_ode
midpoint
simpson
interface
Mdot_outer/Mdot_inner
wind_sink_fraction
source_integral/Mdot_inner
Lrad/LEdd
f_adv_global
f_adv_inner
max H/R
Rson
```

---

## Phase 2 — Implement adaptive active-window corrector

After each tangent-predicted eta step:

1. Evaluate active source-band global-replacement residual.
2. Dump top active rows.
3. Select correction window from the actual top active row.
4. Solve a coupled local block.
5. Line-search using the active source-band replacement score.
6. Refresh source-band aux rows after any accepted update.
7. Re-evaluate top active rows.

### Required top-row diagnostic fields

For each top active row, output:

```text
row index
old_group
old_kind
active row type
R_mid/rg
residual value
row normalization
neighboring row types
inside/outside source band
inside sonic window?
inside active correction window?
```

---

## Phase 3 — Window selection logic

### Case A: peak row is `old_sonic_pivot` or `old_sonic_D`

Use a sonic-local window.

Initial window:

```text
R = 5.0--8.0 rg
```

If the residual moves outward:

```text
R = 5.0--10.0 rg
```

Release variables:

```text
logu
logT
logMdot
logRson / sonic eigenvalue variables if supported
lambda0/global variables if required by the sonic rows
```

Include residual rows:

```text
old_sonic_pivot
old_sonic_D
neighboring old_mass rows
neighboring radial rows
neighboring energy rows
```

### Case B: peak row is `old_mass` at `R < 15 rg`

Use an inner coupled mass/radial/energy window.

Window rule:

```text
include peak row plus at least 3 neighboring intervals on each side
if residual shifts outward, expand to cover all active old_mass peaks in 5--12 rg
```

Release:

```text
logu
logT
logMdot
```

Include residual rows:

```text
old_mass
old_interval_radial
old_interval_energy
sonic rows if R < 6 rg overlaps the window
```

Do **not** solve mass rows alone.

### Case C: peak row is `old_mass` around `R ~ 100--150 rg`

Use a mid-disk block.

Initial window:

```text
R_peak / 1.25 < R < R_peak * 1.25
```

Then widen until top 2--3 active mass rows are included.

Release:

```text
logu
logT
logMdot
```

Include:

```text
old_mass
adjacent old_interval_radial
adjacent old_interval_energy
```

### Case D: peak row is `active_mass_increment_int` or `active_mass_increment_link`

First refresh/check:

```text
SOURCE_BAND_MASS_INCREMENT_INIT=balanced
```

Then:

```text
release DeltaM only
re-evaluate
```

Only if still large:

```text
release DeltaM + nearby source-band logMdot
```

Do not run a broad physical-state correction first.

---

## Phase 4 — Coupled local block formulation

Let the local unknown vector be

```text
y_B = {logu_j, logT_j, logMdot_j}_{j in window}
```

with optional global/sonic variables:

```text
logRson
lambda0
other sonic eigen variables
```

The local residual should include grouped rows:

```text
G_M = old_mass rows in window
G_R = adjacent old_interval_radial rows
G_E = adjacent old_interval_energy rows
G_S = sonic rows if relevant
G_G = guard rows from source-band replacement if window touches source/aux band
```

Use a damped least-squares solve:

```text
min_delta || W_B [F_B(y_B + delta) ] ||_2^2
          + lambda_anchor || D_anchor delta ||_2^2
```

Line search must use:

```text
source_band_global_replacement_active_score
```

not the raw old `final_full`.

Acceptance should also require:

```text
source-band aux rows remain strict;
mass-increment rows remain strict;
physical diagnostics remain smooth;
source-band aux arrays are refreshed after accepted updates.
```

---

## Phase 5 — Jacobian strategy

The current global finite-difference corrector is too expensive. The active-window corrector should use one of these:

### Option 1: local finite-difference Jacobian only within window

This is easiest.

```text
finite difference only y_B variables
not the whole global state
not source-band aux variables unless released
```

### Option 2: block-colored finite differences

Use row/column sparsity to perturb non-overlapping nodes together.

### Option 3: semi-analytic rows where easy

For mass rows, use local derivatives with respect to `logMdot`.

For radial/energy rows, local finite differences may be enough initially.

Do not resurrect the old global finite-difference augmented solve as the default.

---

## Phase 6 — Eta continuation ladder

Continue in:

```text
mu = 1 / eta_E
```

Use small steps around the current frontier.

Recommended immediate ladder:

```text
98.75 -> 98.625 -> 98.50 -> 98.375 -> 98.25
98.25 -> 98.00
98.00 -> 97.75 -> 97.50
97.50 -> 97.00 -> 96.50 -> 96.00
then 95.00 only after multiple boring strict steps
```

Do not jump to `eta_E=90`.

### Step-size controller

```text
if active_score <= 8e-6 and corrector cost is small:
    grow dmu by 1.2

if active_score <= 1e-5 but correction was needed:
    keep dmu

if active_score > 1e-5:
    shrink dmu by 0.5 and retry

if the same window fails twice:
    widen the window and include radial/energy rows

if dmu < 1e-6 repeatedly:
    stop and report tangent/corrector failure
```

---

## Phase 7 — Residual-view audit at frozen checkpoints

There is an important caveat:

```text
The eta tangent continuation is validated under the compatible production
mass-increment/global-replacement view.
```

It is **not yet** a certification of full unit-weight HS/FV source-band collocation if `SOURCE_BAND_CHI_IMPL=1` gives an order-unity score.

At every frozen checkpoint, output an evaluate-only audit:

```text
compatible production score
full HS/FV source-band score if available
old midpoint source audit
FV mass audit
FV energy audit
implicit ODE audit
midpoint/Simpson audit
top active rows
```

Do **not** switch acceptance metrics mid-continuation. But if the full HS/FV audit grows catastrophically at lower `eta_E`, stop and diagnose the residual-view mismatch before continuing much farther.

---

## Phase 8 — Validation after continuation stabilizes

Do not run a full N campaign yet. First make the N164 eta ladder boring to at least:

```text
eta_E ~ 98
```

preferably:

```text
eta_E ~ 95
```

Then run spot checks:

```text
N = 164 baseline
N = 192 or 224 spot check
```

at:

```text
eta_E = 98.75
eta_E = 95 if reached
```

Acceptance:

```text
Mdot_outer/Mdot_inner stable within ~1e-3
Lrad/LEdd stable within ~0.3--1%
Rson stable within ~1e-2 rg
f_adv_global and f_adv_inner stable within ~1--2%
no new source-band residual wall
no new sonic defect
no new outer-buffer wall
```

---

## Physical interpretation checkpoint

The mass-loaded wind branch is numerically progressing, but it is not yet a physically certified wind-regulated QPE high state.

A physically meaningful wind branch will need:

```text
stable continuation to eta_E substantially below 100;
mesh/N validation;
smooth Mdot(R);
reasonable s_eff = dlnMdot/dlnR in source-free wind-active regions;
mass/energy/angular-momentum budget audits;
eventual equilibrium/stability map.
```

For now, the target is numerical certification of the local-Mdot mass-loaded wind continuation.

---

## Codex-ready prompt

```text
Please review the latest Mdot=5 eta tangent continuation results and implement
the next adaptive active-window eta continuation step.

Primary note:
- Note/CODEX_MDOT5_ETA_TANGENT_CONTINUATION_RESULTS.md

Important previous note:
- Note/CODEX_SOURCE_BAND_PRODUCTION_HSFV_RESULTS.md

Current physical/numerical target:
- Mdot_inner/Edd = 5
- Rout = 335 rg
- Rinj = 240 rg
- f_s = 0.80
- compact-C2 stream source
- torque_delta_l_fraction = +0.005
- local-Mdot mass-loaded wind
- N = 164
- continue downward in eta_E from the current strict eta_E=98.75 endpoint

Current anchor:
- eta_E=100 source-band HS/FV production checkpoint:
  outputs/checkpoints/m5_source_band_freezeaux_polish16_eta100_N164/
      stage_00_etaE_100_N164.npz

Current status:
- eta_E=100 identity-aware source-band production is strict.
- Source-band mass-increment rows, implicit ODE rows, midpoint rows, and
  Simpson rows are no longer the blocker.
- Coupled tangent predictor in mu = 1/eta_E works for the broad mass-loading
  response.
- Row localization now identifies the near-inner limiter as old_sonic_pivot
  when applicable.
- Sonic-local correction fixes the sonic pivot cheaply.
- Balanced mass-increment initialization removes the false
  active_mass_increment_link wall.
- Latest strict endpoint is eta_E=98.75 at N164 under the source-band
  replacement score.
- Legacy final_full remains dominated by old production rows and is not the
  acceptance metric for these replacement-formulation experiments.

Current limiter:
- active outside-old/global mass-profile rows during eta continuation.
- Limiter alternates between:
    old_sonic_pivot near R~5.30 rg
    old_mass near R~7--9 rg
    sometimes old_mass near R~125 rg
- Local mass-only patches move the defect outward and are not sufficient.

Do not:
- jump directly to eta_E=95 or eta_E=90;
- use mass-only logMdot prediction;
- use broad damped global mass-profile predictor;
- re-enable old midpoint source rows as production rows;
- change source/wind physics yet;
- claim physical wind-branch certification before N/mesh validation.

Main request:
Implement an adaptive active-window corrector driven by the actual top active
source-band replacement row.

Algorithm:

1. After each tangent-predicted eta step, evaluate the active source-band
   replacement residual and dump the top active rows.

Required row data:
    row index
    old_group
    old_kind
    active row type
    R_mid/rg
    residual value
    row normalization
    neighboring row types
    window classification:
        sonic, inner, mid-disk, source-band, buffer, outside-old

2. Select correction window from the peak active row.

If peak row is old_sonic_pivot or old_sonic_D:
    window = sonic-local
    start with R = 5.0--8.0 rg
    if residual moves outward, expand to R = 5.0--10.0 rg
    release:
        logu, logT, logMdot
        logRson / lambda0 / sonic global variables if supported
    include:
        sonic rows
        neighboring old_mass rows
        neighboring radial/energy rows

If peak row is old_mass near R < 15 rg:
    window = inner coupled mass/radial/energy block
    include the peak row plus at least 3 neighboring intervals on each side
    include old_mass, old_interval_radial, old_interval_energy rows
    include sonic rows if window overlaps R < 6 rg
    release logu, logT, logMdot
    do not optimize mass rows alone

If peak row is old_mass near R ~100--150 rg:
    window = mid-disk coupled block
    start with R_peak/1.25 < R < R_peak*1.25
    widen until the top 2--3 active old_mass rows are included
    release logu, logT, logMdot
    include adjacent radial/energy rows

If peak row is active_mass_increment_int or active_mass_increment_link:
    first refresh/check balanced DeltaM initialization
    release DeltaM only
    re-evaluate
    if still large, release DeltaM + nearby source-band logMdot

3. The corrector must be coupled.

The local residual should include:
    top active old_mass rows
    neighboring old_interval_radial rows
    neighboring old_interval_energy rows
    sonic rows if in or near the window
    source-band replacement guard rows if affected
    interface rows if the window touches source-band/buffer boundary

Do not optimize old_mass alone.

4. Use source-band replacement active score as the line-search metric.

Acceptance metric:
    source_band_global_replacement_active_score

Guardrails:
    source-band implicit/midpoint/Simpson rows remain strict
    mass-increment int/link rows remain strict
    physical diagnostics remain smooth
    aux arrays are refreshed after every accepted state update

After accepted update:
    recompute source-band replacement residual rows
    recompute/persist aux arrays
    write post-update top-row diagnostics

5. Continue in mu = 1/eta_E.

Recommended next eta ladder:
    98.75 -> 98.625 -> 98.50 -> 98.375 -> 98.25
    then 98.0, 97.75, 97.5
    then 97, 96.5, 96, 95 only after several boring strict steps

Step controller:
    if active_score <= 8e-6 and correction cost is small:
        grow dmu by 1.2
    if active_score <= 1e-5 but correction needed:
        keep dmu
    if active_score > 1e-5:
        shrink dmu by 0.5 and retry
    if same window fails twice:
        widen window and include radial/energy rows
    if dmu < 1e-6 repeatedly:
        stop and report tangent/corrector failure

6. Keep balanced mass increments as default.

Set:
    SOURCE_BAND_MASS_INCREMENT_INIT=balanced

For every eta step report:
    max active_mass_increment_int
    max active_mass_increment_link
    max raw FV mass audit
    location of max mass-increment rows

7. Reconcile production residual views by audit.

At accepted checkpoints report:
    compatible production score
    full HS/FV source-band score if available
    old midpoint source audit
    FV mass/energy audits
    implicit ODE audit
    midpoint/Simpson audit

Do not switch acceptance metrics mid-continuation. But if full HS/FV score
grows catastrophically, stop and diagnose the residual-view mismatch.

8. Freeze validation checkpoints.

Freeze at least:
    eta_E = 100
    eta_E = 99.5
    eta_E = 99.0
    eta_E = 98.75
    eta_E = 98.0 if reached
    eta_E = 95 if reached

For each frozen checkpoint report:
    active source-band replacement score
    top active rows
    outside_old_mass
    outside_old_energy
    mass_increment_int/link
    implicit_ode
    midpoint
    simpson
    interface
    Mdot_outer/Mdot_inner
    wind_sink_fraction
    source_integral/Mdot_inner
    Lrad/LEdd
    f_adv_global
    f_adv_inner
    max H/R
    Rson
    s_eff(R) = dlnMdot/dlnR in source-free wind-active regions

9. Mesh/N validation comes after eta continuation is stable.

First make N164 continuation robust to eta_E~98 or eta_E~95.
Then run spot checks:
    N = 192 or 224
at:
    eta_E = 98.75
    eta_E = 95 if reached

Acceptance:
    Mdot_outer/Mdot_inner within ~1e-3
    Lrad within ~0.3--1%
    Rson within ~1e-2 rg
    f_adv_global/f_adv_inner within ~1--2%
    no new source-band residual wall

10. Pseudo-arclength later only if needed.

Do not implement pseudo-arclength yet.
Use it only if:
    dmu collapses below ~1e-6 repeatedly;
    tangent norm grows rapidly;
    accepted points show a real turning/fold signature;
    corrector converges backward in eta_E.

Expected deliverables:
- Note/CODEX_MDOT5_ADAPTIVE_ACTIVE_WINDOW_ETA_CONTINUATION_RESULTS.md
- JSON/MD tables for:
    adaptive-window eta ladder
    rejected-step diagnostics
    top active-row localization
    window choices
    before/after correction residuals
- checkpoint directories for every accepted eta point
- profile JSONs around active windows

Bottom line:
The source-band HS/FV problem is solved enough for eta continuation. The current
limiter is the active outside-old mass/sonic response during eta changes.
Implement an adaptive active-window corrector that selects the correction window
from the actual top active row, includes adjacent mass/radial/energy rows,
refreshes aux arrays after accepted updates, and continues in small
mu=1/eta_E steps. Do not jump to eta_E=95/90 until this is boring.
```
