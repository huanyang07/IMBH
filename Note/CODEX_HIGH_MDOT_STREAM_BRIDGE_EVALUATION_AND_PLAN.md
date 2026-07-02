# Codex Handoff: High-`Mdot` Advective Branch Evaluation and Stream-Fed Bridge Plan

**Date:** 2026-07-01  
**Project:** IMRI/QPE stream-fed minidisk / slim-disk solver  
**Purpose:** Provide a Codex-ready evaluation and implementation plan after the standard no-wind slim branch was recovered to high accretion rate and the finite-minidisk stream source/sink sign was corrected.

---

## 0. Executive Decision

Use **the recovered standard no-wind high-`Mdot` slim branch as the backbone**, then continue it into the finite-minidisk stream-fed model.

The best next path is:

```text
standard high-Mdot no-wind branch
    -> finite Rout no-stream branch
    -> finite Rout conservative stream mass source
    -> stream angular momentum source / torque
    -> stream heating
    -> wind only if no-wind topology is absent or physically incomplete
    -> equilibrium/stability map and eventually time-dependent cycle
```

Do **not** try to infer the QPE hot branch from the current fixed-`Mdot_inner/Edd = 1` stream-source case. That case is valuable, but it is mainly a **source-bookkeeping and numerical-stiffness benchmark**, not the hot branch.

Answer to the specific strategic choices:

| Choice | Recommendation | Reason |
|---|---:|---|
| 1. Raise `Mdot_inner` in stream-fed model | **Yes, but as part of the bridge** | True high inner accretion rate is the control parameter that produced strong advection in the standard branch. |
| 2. Add stream torque/heating/wind before raising `Mdot` | **No for wind; torque/heating later** | Adding wind/heating too early can hide whether the conservative finite minidisk can access the high branch. |
| 3. Fix `interval_E` stiffness near high source fraction first | **Do in parallel** | Important infrastructure, but not the main science blocker. |
| 4. Continue from no-wind `Mdot > 1` slim branch into finite stream/wind model | **Yes — primary plan** | This directly connects the two existing tracks rather than treating them as unrelated puzzles. |

Bottom line:

```text
Use option 4 as the spine.
Use option 1 as the main physical continuation parameter.
Use option 3 as a targeted numerical cleanup.
Delay option 2, especially wind, until the no-wind high-Mdot bridge is tested.
```

---

## 1. Current State of the Two Tracks

### Track A — Standard no-wind slim disk

The standard no-wind slim-disk branch has been recovered to:

```text
Mdot/Edd = 5
```

This is the most important update. It demonstrates that the solver can find a physical advective/slim branch when the actual inner supplied accretion rate is raised above Eddington.

Certified/accepted sequence:

| `Mdot/Edd` | N | residual | anchor? | `f_adv_global` | `f_adv_inner(R<20rg)` | `Lrad/LEdd` | max `H/R` | `Rson/rg` |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| `1` | `512` | `5.175e-6` | no | `0.0347` | `-0.1151` | `0.5934` | `0.1572` | `5.084` |
| `2` | `640` | `6.450e-6` | no | `0.1865` | `0.0943` | `0.9660` | `0.2269` | `4.661` |
| `3` | `640` | `7.262e-6` | no | `0.3018` | `0.2562` | `1.213` | `0.2677` | `4.502` |
| `4` | `640` | `2.507e-6` | yes | `0.3894` | `0.3830` | `1.396` | `0.2956` | `4.415` |
| `5` | `768` | `2.293e-6` | yes | `0.4534` | `0.4666` | `1.541` | `0.3164` | `4.360` |

At `Mdot/Edd = 5`, the key diagnostics are:

```text
residual                  = 2.293e-6
f_adv_global              = 0.4534
f_adv_inner(R < 20 rg)    = 0.4666
Lrad/LEdd                 = 1.541
max H/R                   = 0.3164
Rson                      = 4.360 rg
```

### Evaluation of Track A

This is a real signature of a hot/advective slim branch:

1. **Advection grows monotonically with accretion rate.**  
   `f_adv_global` rises from `0.0347` at `Mdot/Edd = 1` to `0.4534` at `Mdot/Edd = 5`.

2. **Inner advection becomes strongly positive.**  
   `f_adv_inner(R<20rg)` crosses positive above roughly `Mdot/Edd ~ 1.5--1.6` and reaches `0.4666` at `Mdot/Edd = 5`.

3. **Luminosity grows sublinearly.**  
   At `Mdot/Edd = 5`, `Lrad/LEdd = 1.541`, not roughly 5. This is the expected photon-trapping/slim-disk signature.

4. **The sonic point moves inward smoothly.**  
   `Rson` shifts from `5.084 rg` at `Mdot/Edd = 1` to `4.360 rg` at `Mdot/Edd = 5`, with no obvious branch jump.

5. **The disk thickens but remains height-integrated/slim.**  
   `max H/R = 0.3164` at `Mdot/Edd = 5`, which is thick enough to be slim/advective but not so thick that the calculation is immediately outside the intended regime.

Conclusion:

```text
The standard no-wind advective slim branch is recovered to Mdot/Edd = 5.
```

This does **not** yet prove that the finite stream-fed IMRI minidisk has a hot branch. It proves that the solver can find one in the standard problem.

---

### Track B — Corrected finite-minidisk stream source/sink model

The latest stream source/sink sign convention has been corrected for inward-positive accretion rate:

```math
\dot M = -2\pi R\Sigma v_R > 0
```

Steady continuity with wind sink and stream source is now:

```math
\frac{d\dot M}{d\ln R} = \dot M'_w - \dot M'_s
```

where:

```math
\dot M'_s = 2\pi R^2 S_\Sigma \ge 0
```

is stream mass added to the disk, and:

```math
\dot M'_w = 2\pi R^2 \dot\Sigma_w \ge 0
```

is wind mass removed from the disk.

Therefore:

```text
pure stream source, no wind:
    dMdot/dlnR < 0
    Mdot_outer < Mdot_inner

pure wind sink, no stream:
    dMdot/dlnR > 0
    Mdot_outer > Mdot_inner
```

The corrected source-only finite-minidisk benchmark at fixed inner accretion rate is:

```text
Mdot_inner/Edd            = 1
Rout                      = 300 rg
N                         = 640
source annulus center      = 0.8 Rout
source annulus width       = 0.08
source fraction            = 0.49
Mdot_outer/Mdot_inner      = 0.511844
source integral/Mdot_inner = 0.4881
full residual              = 2.438e-6
dominant residual          = interval_E
max H/R                    = 0.1571
integrated advective frac  = 0.03404
```

### Evaluation of Track B

This result is good news numerically, but it is not the hot branch.

It shows:

```text
source/sink bookkeeping is now correct;
positive stream source correctly gives Mdot_outer < Mdot_inner;
large conservative source fractions can be reached with strict residual;
the current bottleneck is interval_E near high source fraction.
```

It does **not** show:

```text
a strongly advective finite-minidisk branch;
a QPE hot state;
a limit cycle;
a wind-regulated upper branch.
```

At `Mdot_inner/Edd = 1`, the stream-fed model has:

```text
max H/R ~ 0.157
integrated advective fraction ~ 0.034
```

Those values are close to the mild near-Eddington standard branch. This is expected. The run fixes the mass-source bookkeeping, but it does not raise the true inner accretion rate into the high-`Mdot` regime where Track A becomes advective.

Conclusion:

```text
Track B is a corrected bookkeeping/regression anchor, not a hot-branch anchor.
```

---

## 2. Physical Interpretation

The important conceptual update is that the project now has a proven high-`Mdot` standard backbone.

The question is no longer:

```text
Can the solver find a slim/advective branch at all?
```

The answer is yes, in the standard no-wind problem.

The real question is now:

```text
Can a finite, stream-fed, tidally truncated minidisk continue onto or access
that high-Mdot advective branch once realistic finite-boundary, source,
angular-momentum, heating, and eventually wind terms are introduced?
```

The fixed-`Mdot_inner/Edd = 1`, high-source-fraction stream run cannot answer that question. It only tests whether mass can be added in the outer annulus while keeping the inner rate at roughly Eddington.

The finite-minidisk problem is more constrained than the standard slim disk because it has:

```text
finite outer radius;
finite reservoir entropy;
stream mass injection;
stream angular momentum injection;
possible stream shock heating;
tidal truncation;
possibly wind mass/energy/angular-momentum loss.
```

So the high-`Mdot` branch may survive, deform, terminate, or require wind. The next plan must measure this continuously instead of assuming it.

---

## 3. Required Diagnostics Before Calling Anything a Hot Branch

A stream-fed solution should not be called a true hot/advective branch merely because it converges.

Require these diagnostics:

```text
1. Continuation link:
   The solution should continue smoothly from the standard high-Mdot Track A
   parent, or from the finite-Rout high-Mdot no-stream parent.

2. Inner advection:
   f_adv_inner(R < 20 rg) should be clearly positive.
   Rough targets:
      Mdot_inner/Edd = 2: f_adv_inner ~ 0.1 or larger
      Mdot_inner/Edd = 3: f_adv_inner ~ 0.25 or larger
      Mdot_inner/Edd = 5: f_adv_inner ~ 0.4 or larger

3. Global advection:
   f_adv_global should not be a tiny signed-cancellation artifact.

4. Positive advection:
   f_adv_pos should be reported to separate real advective transport from
   signed cancellation.

5. Luminosity:
   Lrad/LEdd should grow sublinearly with Mdot_inner.

6. Sonic point:
   Rson should move smoothly, with no discontinuous branch jump.

7. Thickness:
   max H/R should remain plausibly slim, preferably below ~0.4 for the first
   accepted bridge runs.

8. Mass budget:
   Mdot_outer/Mdot_inner must match the integrated source/wind budget.

9. Mesh convergence:
   Compare at least N = 640 and N = 768, with source-refined mesh where needed.

10. Residual localization:
    The dominant residual block must be understood. A source-annulus interval_E
    spike should not be mistaken for physical branch termination.
```

Recommended definitions:

```math
f_{adv,global}
=
\frac{\int Q_{adv}\,2\pi R\,dR}{\int Q_{visc}\,2\pi R\,dR}
```

```math
f_{adv,inner}
=
\frac{\int_{R<20r_g} Q_{adv}\,2\pi R\,dR}{\int_{R<20r_g} Q_{visc}\,2\pi R\,dR}
```

```math
f_{adv,pos}
=
\frac{\int \max(Q_{adv},0)\,2\pi R\,dR}{\int Q_{visc}\,2\pi R\,dR}
```

---

## 4. Main Continuation Plan

### Phase 0 — Freeze benchmark anchors

Create or preserve two protected benchmark anchors.

#### Anchor A: Standard no-wind high-`Mdot` branch

Use the certified sequence:

```text
Mdot/Edd = 1, 2, 3, 4, 5
```

Most important anchor:

```text
Mdot/Edd                  = 5
residual                  = 2.293e-6
f_adv_global              = 0.4534
f_adv_inner(R < 20 rg)    = 0.4666
Lrad/LEdd                 = 1.541
max H/R                   = 0.3164
Rson                      = 4.360 rg
```

This is the current hot/slim backbone.

#### Anchor B: Corrected stream-source bookkeeping at `Mdot_inner/Edd = 1`

Use:

```text
Mdot_inner/Edd            = 1
Rout                      = 300 rg
source fraction            = 0.49
Mdot_outer/Mdot_inner      = 0.511844
source integral/Mdot_inner = 0.4881
residual                   = 2.438e-6
dominant residual          = interval_E
max H/R                    = 0.1571
integrated advective frac  = 0.03404
```

This is a source-sign/mass-budget/regression anchor, not a hot-branch anchor.

Codex tasks:

```text
- Add/verify regression test for source-only mass budget sign.
- Add/verify benchmark table for Track A high-Mdot sequence.
- Ensure both anchors are reproducible from clean checkout.
- Store outputs in clear tables with date/stem names.
```

---

### Phase 1 — Build finite-`Rout` no-stream high-`Mdot` bridge

Before turning on stream physics, shrink the recovered standard high-`Mdot` branch to minidisk-sized outer radius.

Run:

```text
Mdot_inner/Edd = 2, 3, 5
Rout/rg        = 10000 -> 3000 -> 1000 -> 500 -> 300
stream source  = 0
wind           = 0
stream torque  = 0
stream heating = 0
outer closure  = pressure-supported, not hard Keplerian/Omega Dirichlet
```

Purpose:

```text
Ask whether the high-Mdot advective branch survives finite truncation to
Rout ~ 300 rg before stream physics is added.
```

Acceptance criteria:

```text
residual <= few x 1e-6, or at least <= 1e-5 for scout runs;
Rson evolves smoothly;
f_adv_inner remains close to the large-Rout parent branch;
Lrad/LEdd remains sublinear with Mdot_inner;
max H/R remains in slim-disk range, preferably <~0.4;
no new outer-boundary angular residual pathology.
```

Deliverable table columns:

```text
Mdot_inner/Edd
Rout/rg
N
residual
dominant_block
f_adv_global
f_adv_inner_Rlt20
f_adv_pos
Lrad_LEdd
max_H_over_R
Rson_rg
lambda0_or_l0
notes
```

Decision logic:

```text
If finite-Rout no-stream branch survives to Rout = 300 rg:
    proceed to Phase 2.

If it fails before Rout = 300 rg:
    diagnose whether failure is outer closure, mesh, sonic matching, or true
    finite-boundary incompatibility before adding stream terms.
```

---

### Phase 2 — Turn on conservative stream mass source at high `Mdot_inner`

After finite-`Rout` no-stream parents exist, add stream mass source while keeping wind, torque, and heating off.

Run first with a broad source annulus:

```text
Mdot_inner/Edd = 2, 3, 5
Rout           = 300 rg
Rinj/Rout      = 0.8
width          = 0.30
source fraction f_s = 0.05, 0.10, 0.20, 0.30
wind           = 0
stream torque  = 0
stream heating = 0
```

Then test the narrower source annulus:

```text
width = 0.08
same Mdot_inner and f_s ladder
```

Reason for broad first:

```text
The narrow width = 0.08 annulus is likely where source-localized interval_E
stiffness appears. A broad annulus tests the physics before testing the hardest
numerical localization.
```

Main comparison:

```text
For each Mdot_inner = 2, 3, 5:
    compare stream-source run against its finite-Rout no-stream parent.
```

Decision logic:

```text
If f_adv_inner stays high:
    conservative stream feeding preserves the hot branch.

If f_adv_inner collapses only for narrow/high f_s:
    source localization or entropy/mass loading is likely the issue.

If f_adv_inner collapses even for broad/low f_s:
    finite stream mass loading may physically suppress the advective branch.
```

Required outputs:

```text
Mdot_inner/Edd
f_s
Rinj/Rout
width
Mdot_outer/Mdot_inner
source_integral/Mdot_inner
mass_budget_error
residual
dominant_block
f_adv_global
f_adv_inner_Rlt20
f_adv_pos
Lrad_LEdd
max_H_over_R
Rson_rg
peak_interval_E_location
notes
```

---

### Phase 3 — Diagnose and fix `interval_E` stiffness near high source fraction

Do this in parallel with Phases 1--2. Do not let it hijack the whole project, but do fix it because it will matter for later high-source runs.

Stress-test case:

```text
Mdot_inner/Edd = 1
Rout           = 300 rg
Rinj/Rout      = 0.8
width          = 0.08
f_s            = 0.49
```

Dump radial profiles of:

```text
R_mid/rg
interval_E
interval_R
interval_Omega
Mdot(R)
dMdot/dlnR
stream_source_prime(R)
Qadv/Qvisc
Qstream/Qvisc
H/R
local mesh spacing
condition number or singular-value diagnostics, if available
```

Classify residual localization:

```text
If interval_E peaks inside source annulus:
    add source-annulus adaptive mesh monitor.

If interval_E peaks near outer edge:
    reuse/adapt the outer remeshing strategy that fixed Track A.

If interval_E peaks near sonic point:
    inspect sonic regularity and inner transonic matching.

If interval_E is broad:
    inspect energy-equation scaling and Jacobian conditioning.
```

Add a source-annulus mesh monitor using some combination of:

```text
|interval_E|
stream_source_prime
|dMdot/dlnR|
Qstream/Qvisc
```

Test:

```text
N = 640 baseline
N = 768 source-refined
N = 896 source-refined, if needed
```

Optional continuation improvement:

For nonlinear system:

```math
F(z, f_s) = 0
```

compute tangent predictor:

```math
J_z \frac{dz}{df_s} = -F_{f_s}
```

Use this to predict larger source-fraction steps through the stiff region near `f_s ~ 0.49`.

---

### Phase 4 — Add stream angular momentum source / torque before stream heating

Only after the source-only high-`Mdot` bridge is mapped, add angular momentum injection.

Reason:

```text
Angular momentum can decide whether the finite-boundary BVP intersects the
inner transonic branch. It should be tested before adding stream shock heating
or wind.
```

Run:

```text
Mdot_inner/Edd       = 2, 3, 5
Rout                 = 300 rg
f_s                  = 0.10, 0.30
width                = 0.30 first, then 0.08
l_stream/l_K(Rinj)   = 0.8, 1.0, 1.2
wind                 = 0
stream heating       = 0
```

Implementation preference:

```text
Use a distributed source/torque term.
Avoid hard outer Omega or hard circularization-radius Dirichlet boundaries as
the first implementation, because they are brittle finite-boundary constraints.
```

Diagnostics:

```text
same as Phase 2, plus:
angular_momentum_budget_error
stream_torque_integral
l_stream/l_K
Omega_outer behavior
lambda0/l0 trend
```

Decision logic:

```text
If angular momentum source smoothly deforms the high branch:
    proceed to stream heating.

If branch fails at specific l_stream/l_K:
    map the allowed angular-momentum window.

If branch fails for all l_stream/l_K:
    inspect whether the finite outer closure is overconstrained or whether wind
    angular-momentum loss is physically required.
```

---

### Phase 5 — Add stream heating only after mass + angular momentum are stable

Stream heating is physically important but should not be added before mass and angular momentum are understood.

Suggested first heating ladder:

```text
eta_heat or epsilon_sh = 0, 0.01, 0.03, 0.10
```

Use the repository's existing convention if a different normalization is already implemented, but avoid jumping immediately to an extreme heating value unless previous tests require that convention.

Run on successful Phase 4 cases:

```text
Mdot_inner/Edd       = 2, 3, 5
Rout                 = 300 rg
f_s                  = 0.10, 0.30
width                = 0.30 first
l_stream/l_K(Rinj)   = best/smooth cases from Phase 4
wind                 = 0
```

Track whether heating:

```text
raises H/R too much;
raises Lrad/LEdd too much;
reduces or enhances f_adv_inner;
creates local interval_E stiffness;
moves Rson discontinuously.
```

Warning criterion:

```text
If max H/R approaches or exceeds ~0.5, flag the solution as possibly outside
height-integrated slim-disk comfort zone.
```

---

### Phase 6 — Add wind only after no-wind topology is known

Do not add wind as the next immediate move.

Add wind if one of these occurs:

```text
1. high-Mdot no-wind finite-minidisk branch is absent;
2. conservative stream-fed branch cannot reach the hot state;
3. hot branch is too radiative/super-Eddington without mass loss;
4. hot branch is too geometrically thick for the model assumptions;
5. equilibrium map lacks a viable upper branch.
```

Minimal wind equations should preserve the corrected sign convention:

```math
\frac{d\dot M}{d\ln R} = \dot M'_w - \dot M'_s
```

Energy equation schematic:

```math
Q_{visc} + Q_{stream} = Q_{rad} + Q_{adv} + Q_{wind}
```

Start with simplest angular momentum assumption:

```text
l_w = l
```

Then scan lever arm only if needed:

```text
l_w = lambda_w l
lambda_w = 1.2, 1.5, 2.0
```

Suggested wind efficiency scan:

```text
epsilon_w = 0.1, 0.3, 1.0
```

Required wind diagnostics:

```text
Mdot_inner/Edd
Mdot_outer/Edd
integrated wind loss / Mdot_inner
integrated stream source / Mdot_inner
Mdot_outer/Mdot_inner
mass_budget_error
Qwind/Qvisc integrated
angular_momentum_budget_error
f_adv_inner
Lrad/LEdd
max H/R
Rson
residual
dominant_block
```

---

### Phase 7 — Build finite-minidisk equilibrium and stability map

A steady hot branch is not a QPE limit cycle. A limit cycle requires an equilibrium map and stability labels.

After the bridge is established, build a map parameterized by one or more reservoir/load variables:

```text
Mdisk
Sigma_out
source normalization
outer entropy/load parameter
```

Output:

```text
Mdot_inner(Mdisk)
Lrad(Mdisk)
f_adv_inner(Mdisk)
f_adv_global(Mdisk)
f_adv_pos(Mdisk)
max H/R(Mdisk)
Rson(Mdisk)
```

Add stability labels using a thermal stability diagnostic such as:

```math
\left.\frac{\partial(Q^+ - Q^-)}{\partial T}\right|_\Sigma
```

or the project’s preferred finite-difference equivalent.

Need to identify:

```text
cool stable branch;
unstable middle branch;
hot advective or wind-regulated branch;
upper transition threshold;
lower transition threshold;
Delta M_cyc estimate;
rough recurrence-time estimate from stream-fed loading.
```

Only after that should the project claim a limit-cycle skeleton.

---

## 5. Minimal Decisive Experiment

The smallest experiment that directly connects Track A and Track B is:

```text
Experiment name:
    high_Mdot_finite_Rout_stream_bridge

Goal:
    Determine whether the corrected finite stream-fed minidisk remains on or
    connects to the recovered high-Mdot advective slim branch.
```

### A. Finite-`Rout` no-stream baseline

```text
Mdot_inner/Edd = 2, 3, 5
Rout           = 300 rg
f_s            = 0
wind           = 0
torque         = 0
heating        = 0
```

### B. Broad conservative stream source

```text
Mdot_inner/Edd = 2, 3, 5
Rout           = 300 rg
Rinj/Rout      = 0.8
width          = 0.30
f_s            = 0.10, 0.30
wind           = 0
torque         = 0
heating        = 0
```

### C. Narrow conservative stream source

```text
same as B, but width = 0.08
```

### D. Add stream angular momentum

```text
l_stream/l_K = 0.8, 1.0, 1.2
use f_s      = 0.10 and 0.30
use width    = 0.30 first
```

### E. Add stream heating only after D succeeds

```text
eta_heat or epsilon_sh = 0.01, 0.03, 0.10
```

This experiment answers:

```text
Is the corrected finite stream-fed minidisk weakly advective only because the
latest run used Mdot_inner/Edd = 1, or does finite stream-fed boundary/source
physics prevent access to the high-Mdot advective branch?
```

Working expectation:

```text
The weak advection in the latest stream-source run is probably mostly because
Mdot_inner/Edd = 1. But finite-boundary and source-annulus physics can still
deform or terminate the branch, so this must be tested by continuation rather
than assumed.
```

---

## 6. Codex Implementation Checklist

### Immediate tasks

```text
[ ] Freeze/reproduce Track A high-Mdot table through Mdot/Edd = 5.
[ ] Freeze/reproduce Track B corrected source-sign f_s = 0.49 table.
[ ] Add or verify automated mass-budget sign tests:
    - pure stream source: Mdot_outer < Mdot_inner
    - pure wind sink: Mdot_outer > Mdot_inner
    - mixed source/wind: matches integral budget
[ ] Add f_adv_inner and f_adv_pos to all relevant output tables.
[ ] Add peak residual localization columns, especially peak interval_E radius.
```

### Phase 1 implementation

```text
[ ] Implement/run finite-Rout high-Mdot no-stream continuation:
    Mdot_inner/Edd = 2, 3, 5
    Rout/rg = 10000 -> 3000 -> 1000 -> 500 -> 300
[ ] Use pressure-supported outer closure.
[ ] Save full diagnostic table and representative radial profiles.
```

### Phase 2 implementation

```text
[ ] Start from finite-Rout no-stream parents at Rout = 300 rg.
[ ] Add conservative stream mass source:
    Rinj/Rout = 0.8
    width = 0.30 first, then 0.08
    f_s = 0.05, 0.10, 0.20, 0.30
[ ] Run for Mdot_inner/Edd = 2, 3, 5.
[ ] Save mass-budget and advective-branch diagnostics.
```

### Phase 3 implementation

```text
[ ] For Mdot_inner/Edd = 1, f_s = 0.49, dump interval_E profile.
[ ] Identify whether interval_E peak is in source annulus, outer edge, sonic
    region, or broad/global.
[ ] Add source-annulus adaptive mesh if localized near injection.
[ ] Test N = 640, 768, 896 as needed.
[ ] Add source-fraction tangent predictor if continuation stalls.
```

### Later tasks

```text
[ ] Add stream angular momentum source/torque after source-only bridge.
[ ] Add stream heating after mass + angular momentum bridge.
[ ] Add wind only after no-wind topology is known.
[ ] Build equilibrium/stability map after steady bridge is credible.
```

---

## 7. Suggested Output Files

Use names like:

```text
outputs/tables/high_mdot_standard_nowind_anchor.md
outputs/tables/high_mdot_finite_Rout_nowind_bridge.md
outputs/tables/high_mdot_stream_source_bridge_broad.md
outputs/tables/high_mdot_stream_source_bridge_narrow.md
outputs/tables/source_fraction_interval_E_diagnostics.md
outputs/tables/stream_angular_momentum_bridge.md
outputs/tables/stream_heating_bridge.md
outputs/tables/wind_bridge_if_needed.md
```

For radial profiles:

```text
outputs/profiles/high_mdot_finite_Rout_nowind_*.csv
outputs/profiles/high_mdot_stream_source_*.csv
outputs/profiles/source_fraction_interval_E_*.csv
```

For plots:

```text
outputs/figures/f_adv_vs_Mdot_standard_and_stream.png
outputs/figures/Lrad_vs_Mdot_standard_and_stream.png
outputs/figures/Rson_vs_Mdot_standard_and_stream.png
outputs/figures/interval_E_source_annulus_diagnostic.png
outputs/figures/Mdot_profile_stream_source_sign_check.png
```

---

## 8. Pass/Fail Summary for the Next Commit

A good next Codex update should be able to answer these questions:

```text
1. Does the high-Mdot no-wind branch survive finite Rout = 300 rg?
2. At Mdot_inner/Edd = 5, does conservative stream source preserve strong
   inner advection for f_s = 0.10 and 0.30?
3. Is the high-source interval_E residual localized, and has the mesh monitor
   reduced it?
4. Are f_adv_inner, f_adv_pos, Lrad/LEdd, max H/R, and Rson reported for every
   bridge run?
5. Does the mass budget close under the corrected source/sink convention?
```

Recommended acceptance for a first successful bridge:

```text
Mdot_inner/Edd = 5
Rout = 300 rg
f_s = 0.10 or 0.30
width = 0.30
wind = 0
torque = 0
heating = 0
residual <= 1e-5
f_adv_inner(R<20rg) clearly positive, ideally ~0.4 or not far below parent
Lrad/LEdd sublinear relative to Mdot_inner
max H/R <~ 0.4
Rson smooth relative to parent
mass budget error small
```

If this succeeds, the project has its first credible bridge from the standard high-`Mdot` slim branch into the finite stream-fed minidisk.

If this fails, the failure is scientifically meaningful and should be classified as one of:

```text
outer finite-boundary incompatibility;
source mass-loading/entropy suppression;
angular-momentum mismatch;
energy-equation/source-annulus numerical stiffness;
need for wind mass/energy/angular-momentum loss.
```

---

## 9. Codex-Ready Prompt

```text
Use the recovered standard no-wind high-Mdot slim branch as the backbone and
connect it to the corrected finite-minidisk stream model.

Important evaluation:
- The standard no-wind slim branch is recovered to Mdot/Edd = 5.
- At Mdot/Edd = 5: residual = 2.293e-6, f_adv_global = 0.4534,
  f_adv_inner(R<20rg) = 0.4666, Lrad/LEdd = 1.541, max H/R = 0.3164,
  Rson = 4.360 rg.
- This is a real hot/advective slim-branch signature in the standard no-wind
  problem.
- The corrected stream-fed finite-minidisk source test at Mdot_inner/Edd = 1,
  Rout = 300 rg, f_s = 0.49 has residual = 2.438e-6,
  Mdot_outer/Mdot_inner = 0.511844, max H/R = 0.1571, and integrated
  advective fraction = 0.03404.
- That stream case proves source bookkeeping, not a hot branch.

Plan:
1. Freeze Track A and Track B benchmarks as regression anchors.
2. Build high-Mdot finite-Rout no-stream bridge:
   Mdot_inner/Edd = 2,3,5;
   Rout/rg = 10000 -> 3000 -> 1000 -> 500 -> 300;
   stream = wind = torque = heating = 0;
   pressure-supported outer closure.
3. Add conservative stream mass source at high Mdot_inner:
   Mdot_inner/Edd = 2,3,5;
   Rout = 300 rg;
   Rinj/Rout = 0.8;
   width = 0.30 first, then 0.08;
   f_s = 0.05, 0.10, 0.20, 0.30;
   no wind, no torque, no heating.
4. In parallel, diagnose Mdot_inner/Edd = 1, f_s = 0.49 interval_E stiffness:
   output interval_E(R), interval_R(R), interval_Omega(R), Mdot(R),
   dMdot/dlnR, stream_source_prime(R), Qadv/Qvisc, Qstream/Qvisc, H/R,
   local mesh spacing, and conditioning/singular values if available.
5. Add source-annulus adaptive mesh monitor if interval_E is localized in the
   injection annulus. Add source-fraction tangent predictor if continuation
   stalls near high f_s.
6. After source-only high-Mdot bridge succeeds, add stream angular momentum:
   l_stream/l_K = 0.8, 1.0, 1.2. Avoid hard outer Omega/l Dirichlet boundaries.
7. Add stream heating only after mass + angular-momentum source runs are stable.
8. Add wind only if no-wind high-Mdot finite stream-fed topology is absent,
   too radiative, too thick, or lacks a viable upper branch.
9. Final deliverable: one bridge table showing how the Track A high-Mdot
   advective branch changes under finite Rout, stream source, angular momentum
   source, heating, and eventually wind. Include pass/fail flags for hot-branch
   criteria.
```

---

## 10. Bottom Line

There **is** a strong hot/advective signature once true `Mdot` is high in the standard no-wind problem, especially at `Mdot/Edd = 5`.

There is **not yet** a demonstrated hot/advective branch in the finite stream-fed IMRI minidisk.

The next decisive move is therefore:

```text
continue the recovered high-Mdot no-wind slim branch into finite Rout and then
into corrected conservative stream feeding at Mdot_inner/Edd = 2, 3, 5.
```

Do not add wind as the immediate next move. Use wind only after the no-wind high-`Mdot` stream-fed topology is actually mapped.
