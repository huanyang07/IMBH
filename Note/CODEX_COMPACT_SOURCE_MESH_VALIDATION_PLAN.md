# Codex Handoff: Compact No-Tail Stream Source at `f_s = 0.82`

Date: 2026-07-02

Repository: `huanyang07/IMBH`

Primary files to review:

```text
Note/CODEX_COMPACT_SOURCE_SHAPE_RESULTS.md
Note/GPT_PROMPT_COMPACT_SOURCE_MESH_VALIDATION.md
```

Related prior context:

```text
Note/CODEX_ADAPTIVE_SOURCE_FRONT_0808_RESULTS.md
Note/GPT_PROMPT_STREAM_FRONT_0808.md
GPT_HIGH_MDOT_NOWIND_HANDOFF.md
IMRI_QPE_CONVERSATION_HANDOFF_SUMMARY.md
```

---

## 1. Executive Summary

The newest compact-source result is a real improvement.

The no-wind stream-fed slim-disk branch at

```text
Mdot_inner/Edd = 2
Rout = 300 rg
f_s = 0.82
torque_delta_l_fraction = +0.005
source shape = compact_c2, no tail at Rout
```

now reaches a compact no-tail stream-source endpoint using a normalized source-shape homotopy.

My evaluation:

```text
The compact no-tail branch at f_s = 0.82 is conditionally mesh-supported.
It is supported when residual-aware remeshing is treated as part of the numerical method.
It is not robust under naive/plain N-change remaps.
```

The result strongly weakens the old explanation that the previous wall was merely caused by the tanh source tail touching the outer boundary. The compact C2 no-tail source reaches essentially the same physical branch.

However, the branch still depends on proper mesh/outer-boundary defect control. The remaining limitation is most likely:

```text
outer/source residual control + outer angular closure/slope refresh,
not a physical loss of the inner advective branch.
```

This is still **not** the strongly hot QPE branch. It is a credible finite stream-fed, no-wind, mildly advective bridge at `Mdot_inner/Edd = 2`. The true hot target remains `Mdot_inner/Edd = 3` and especially `5` at `Rout = 300 rg`.

---

## 2. Current Accepted Numerical Result

### 2.1 Full compact endpoint

The direct tanh-to-compact source replacement had a huge seed residual:

```text
initial seed residual ~ 1.75e1
```

The normalized source-shape homotopy succeeded and reached the full compact endpoint:

```text
source_shape = compact_c2
f_s = 0.82
Mdot_inner/Edd = 2
Rout = 300 rg
torque_delta_l_fraction = +0.005

final_full = 5.693e-06
Mdot_outer/Mdot_inner = 0.18
source_integral/Mdot_inner = 0.82
f_adv_global = 0.2042
f_adv_inner = 0.09517
Lrad/LEdd ≈ 0.8672
max H/R = 0.2269
Rson = 4.66 rg
```

Interpretation:

```text
This is a valid no-wind stream-fed branch point, but only mildly advective.
It is not the Mdot/Edd=5-like hot slim branch.
```

### 2.2 Mesh validation status

Plain N-change remaps failed because they generated outer/source `interval_E` defects.

PCHIP remap did not rescue the situation and made initial defects worse.

Residual-aware remeshing rescued the validation:

```text
N = 640:
    final_full = 1.137e-06
    strict = yes

N = 768:
    final_full = 5.693e-06
    accepted, but marginal and outer_omega-dominated

N = 896:
    final_full = 1.043e-06
    strict = yes
```

Physical diagnostics are stable across `N = 640, 768, 896`:

```text
f_adv_global ≈ 0.20419
f_adv_inner ≈ 0.0947 - 0.0952
Lrad/LEdd ≈ 0.8672
max H/R ≈ 0.2269
Rson ≈ 4.6599 rg
Mdot_outer/Mdot_inner ≈ 0.18
source_integral/Mdot_inner ≈ 0.82
```

This is a clean physics-convergence result, even though the residual machinery remains sensitive.

---

## 3. Physical Interpretation

There are now three distinct status levels:

```text
1. Standard no-wind high-Mdot benchmark:
   Solid to Mdot/Edd = 5.
   This is the real hot/slim backbone.

2. Finite stream-fed no-wind branch at Mdot_inner/Edd = 2:
   Now credible to f_s = 0.82 with compact no-tail source,
   but only mildly advective.

3. Full IMRI/QPE hot/wind/limit-cycle branch:
   Not demonstrated yet.
```

The standard no-wind benchmark at `Mdot/Edd = 5` remains the genuine hot-branch reference:

```text
Mdot/Edd = 5
residual = 2.293e-6
f_adv_global = 0.4534
f_adv_inner(R<20rg) = 0.4666
Lrad/LEdd = 1.541
max H/R = 0.3164
Rson = 4.360 rg
```

By contrast, the compact stream-fed `Mdot_inner/Edd = 2`, `f_s = 0.82` branch has:

```text
f_adv_global ≈ 0.204
f_adv_inner ≈ 0.095
max H/R ≈ 0.227
Lrad/LEdd ≈ 0.867
```

So the compact-source result is scientifically important, but it should be described as:

```text
A credible finite stream-fed mildly advective bridge branch,
not yet the QPE hot branch.
```

---

## 4. Answers to the Four Main Questions

### Q1. Is the compact no-tail branch at `f_s = 0.82` mesh-supported?

Yes, with a qualifier.

Recommended wording:

```text
The f_s = 0.82 compact C2 no-tail branch is mesh-supported under residual-aware
remeshing, with stable physical diagnostics across N = 640, 768, and 896.
It should not be called mesh-independent under naive power-grid, direct N-change,
or PCHIP remaps.
```

Residual-aware remeshing is not just a cosmetic rescue anymore. It should be treated as part of the high-source-fraction finite-boundary discretization strategy.

The important result is not that every remap works. The important result is that, once the mesh is adapted to the actual residual/source defects, the physical diagnostics remain stable across resolution.

---

### Q2. Continue `f_s` upward now, or first implement a better adaptive mesh continuation loop?

Do **not** perform a long brute-force push upward from `f_s = 0.82` yet.

Recommended path:

```text
First implement a defect-preserving residual-aware adaptive mesh continuation loop.
Then run only a short f_s-upward pilot from the N = 896 compact checkpoint.
```

A small scouting push is okay after the loop exists:

```text
f_s = 0.8225, 0.825, 0.83, 0.84, 0.85
```

But this should be diagnostic, not the main production run. Stop if the solve starts requiring tiny steps, if `outer_omega` refreshes after every repolish, or if `interval_E` collapses into a single unresolved outer/source cell.

Reasoning:

```text
The previous f_s ≈ 0.808 front could be moved by stronger outer-tail remeshing,
but only with tiny steps and large nfev. The new compact result shows that
manual one-off rescue works. Now the rescue needs to become an automated method.
```

---

### Q3. Should we add an explicit outer-slope Picard or boundary-only correction before pushing `f_s`?

Yes.

Implement a cheap outer-boundary correction before a serious `f_s` push.

Recommended feature:

```text
outer_slope_picard = True
```

Algorithm:

```text
1. Solve/polish with the current outer closure metadata.
2. Recompute outer_match_log_slopes from the polished state.
3. Update only the outer closure metadata.
4. Repolish from the same state.
5. Repeat until:
       |outer_omega| < outer_tol,
       or slope update is tiny,
       or max Picard iterations reached.
```

Suggested controls:

```text
max_outer_picard = 3 to 5
outer_slope_damping = 0.3, 0.5, 0.8, 1.0
outer_tol = 1e-6
repolish_tol = 1e-8
```

Acceptance for this correction:

```text
It should reduce outer_omega without changing:
    f_adv_inner by more than 0.5-1%,
    f_adv_global by more than 0.5-1%,
    Lrad/LEdd by more than 0.5-1%,
    Rson by more than 1e-3 to 1e-2 rg,
    Mdot budget beyond a few x 1e-4.
```

If this boundary-only Picard fails to reduce `outer_omega`, then implement a soft/Robin reservoir closure as the next boundary upgrade.

---

### Q4. What acceptance criteria should be required before adding stream heating or wind?

Before adding stream heating or wind, require the no-wind compact branch to pass five gates:

```text
1. residual robustness,
2. mesh robustness,
3. source-shape robustness,
4. boundary-closure robustness,
5. physical diagnostic smoothness.
```

Minimum required checkpoints:

```text
f_s = 0.70, 0.80, 0.82
```

Optional stretch checkpoints:

```text
f_s = 0.85, 0.90
```

For each checkpoint:

```text
N = 640, 768, 896
source_shape = compact_c2
residual_aware_remesh = on
outer_slope_picard = on/off comparison
```

Pass criteria:

```text
final_full <= 3e-6 preferred
final_full <= 1e-5 acceptable only for exploratory endpoints

mass budget:
    |source_integral/Mdot_inner - f_s| <= 1e-4
    |Mdot_outer/Mdot_inner - (1 - f_s)| <= few x 1e-4

physics stability across N:
    f_adv_inner changes < 1-2%
    f_adv_global changes < 1-2%
    Lrad/LEdd changes < 1%
    max H/R changes < 1%
    Rson changes < 1e-3 to 1e-2 rg

residual profile:
    no single unresolved outer/source cell dominates
    median interval_E remains tiny
    peak interval_E location is explained by mesh/source structure

cost:
    nfev does not grow catastrophically with N
    no need for absurdly tiny df_s except near a documented fold/front
```

Only after this should stream heating be added.

Wind should wait even longer. Do not add wind until no-wind plus heating topology is understood or shown physically incomplete.

---

## 5. Recommended Concrete Implementation Plan

### Phase 1 — Freeze compact-source regression anchors

Save these as named regression anchors:

```text
A. compact_c2 full endpoint, N=768 original residual-remeshed grid
   f_s = 0.82
   final_full = 5.693e-06
   dominant = outer_omega
   Mdot_outer/Mdot_inner = 0.18
   source_integral/Mdot_inner = 0.82
   f_adv_global = 0.2042
   f_adv_inner = 0.09517
   Lrad/LEdd ≈ 0.8672
   max H/R = 0.2269
   Rson = 4.66 rg

B. compact_c2 residual-remesh, N=640
   final_full = 1.137e-06
   strict = yes

C. compact_c2 residual-remesh, N=896
   final_full = 1.043e-06
   strict = yes
```

Also reconcile the test-count metadata mismatch:

```text
CODEX_COMPACT_SOURCE_SHAPE_RESULTS.md says 141 passed.
GPT_PROMPT_COMPACT_SOURCE_MESH_VALIDATION.md says 142 passed.
```

This mismatch is not scientifically important, but it should be cleaned up in the notes/regression docs.

---

### Phase 2 — Promote residual-aware remeshing into the continuation loop

Do not leave residual remeshing as a rescue script.

Implement either a new driver:

```text
continue_stream_source_with_residual_remesh.py
```

or options in the existing stream-source continuation driver:

```text
IMBH_STANDARD_SLIM_STREAM_MASS_RESIDUAL_REMESH_EVERY_STEP=1
IMBH_STANDARD_SLIM_STREAM_MASS_RESIDUAL_REMESH_ON_REJECT=1
IMBH_STANDARD_SLIM_STREAM_MASS_RESIDUAL_REMESH_ON_OUTER_E=1
```

Suggested mesh monitor:

```text
M(R) = 1
     + A * normalized(|interval_E|)
     + B * normalized(stream_source_prime)
     + C * normalized(|dMdot/dlnR|)
     + D * normalized(|dQstream/dlnR|)
     + E * outer_boundary_layer_weight
```

Minimum requirements:

```text
- preserve exact Rout;
- preserve source normalization;
- preserve mass budget;
- remap state smoothly;
- repolish after remap;
- write old/new residual profiles;
- write old/new source integrals;
- write peak residual radius;
- write number of grid nodes in outer 1%, 2%, and 5% of the domain;
- compare diagnostics before and after remap.
```

Recommended output table columns:

```text
f_s
N
mesh_tag
source_shape
outer_picard_on
final_full
dominant_residual
peak_interval_E
R_peak_interval_E
outer_omega
source_integral/Mdot_inner
Mdot_outer/Mdot_inner
f_adv_global
f_adv_inner
Lrad/LEdd
max_H_over_R
Rson
nfev
accepted
```

---

### Phase 3 — Implement outer-slope Picard / boundary-only correction

This should happen before a serious `f_s` push.

Suggested implementation knobs:

```text
IMBH_STANDARD_SLIM_OUTER_SLOPE_PICARD=1
IMBH_STANDARD_SLIM_OUTER_SLOPE_PICARD_MAXITER=3
IMBH_STANDARD_SLIM_OUTER_SLOPE_PICARD_DAMPING=0.5
IMBH_STANDARD_SLIM_OUTER_SLOPE_PICARD_TOL=1e-6
```

Experiment matrix at `f_s = 0.82`:

```text
N = 640, 768, 896
outer_picard = off, on
damping = 0.3, 0.5, 1.0
max_picard = 3
```

Success criteria:

```text
outer_omega decreases;
final_full becomes strict or closer to strict;
physical diagnostics remain unchanged;
Newton cost does not get worse;
source/mass budget remains closed.
```

If successful, make this the default for high-source finite-boundary branches.

---

### Phase 4 — Short `f_s`-upward pilot from N=896 compact checkpoint

After Phases 2 and 3, run a short pilot:

```text
start:
    N = 896
    f_s = 0.82
    source_shape = compact_c2
    residual_remesh = on
    torque_delta_l_fraction = +0.005
    outer_slope_picard = on

targets:
    f_s = 0.8225, 0.825, 0.83, 0.84, 0.85
```

Use:

```text
residual-remesh every accepted step;
residual-remesh on rejected step;
outer-slope Picard enabled;
cost-aware step control;
guarded secant/current-state/tangent seed comparison if available.
```

Stop conditions:

```text
nfev > 150 repeatedly;
df_s < 1e-4 repeatedly;
outer_omega refreshes after every repolish;
interval_E localizes into one unresolved outer/source cell;
source budget drifts;
Mdot_outer/Mdot_inner no longer equals 1 - f_s;
Rson jumps;
H/R jumps;
f_adv diagnostics jump;
solution requires increasingly pathological mesh clustering.
```

The purpose of this pilot is not to brag about a larger `f_s`. The purpose is to see whether the new adaptive machinery makes continuation boring and reproducible.

---

### Phase 5 — Add true tangent predictor or pseudo-arclength continuation if needed

If the upward pilot still needs tiny steps, implement the true tangent predictor.

For the residual system:

```math
F(z, f_s) = 0
```

compute:

```math
J_z \frac{dz}{df_s} = -F_{f_s}.
```

Then predict:

```math
z_{trial} = z + \Delta f_s \frac{dz}{df_s}.
```

Use multiple candidate seeds:

```text
current state
secant predictor
true tangent predictor
damped tangent predictor
```

Choose the seed with the lowest initial full residual.

If the tangent norm grows, the smallest singular value collapses, or the branch appears to fold in `f_s`, switch to pseudo-arclength continuation before declaring a physical endpoint.

Pseudo-arclength condition:

```math
t_z \cdot (z - z_0) + t_f (f_s - f_{s,0}) - \Delta s = 0.
```

Do not call a physical endpoint unless the fold is:

```text
mesh-independent,
boundary-closure independent,
source-shape independent,
and accompanied by a genuine Jacobian/singular-value signature.
```

---

### Phase 6 — Return to the true hot branch: `Mdot_inner/Edd = 3` and `5`

The current `Mdot_inner/Edd = 2` branch is useful, but it is not the strongly advective QPE hot branch.

After residual remeshing and outer-boundary correction are robust, retry the high-Mdot finite-radius no-stream bridges:

```text
Mdot_inner/Edd = 3:
    continue Rout from current front to 300 rg;
    prioritize interval_E residual remeshing.

Mdot_inner/Edd = 5:
    continue Rout from current front to 300 rg;
    prioritize outer_omega / outer-slope Picard / soft-Robin outer closure.
```

Only after `Mdot = 3` or `Mdot = 5` reaches `Rout = 300 rg` should Codex add compact stream source:

```text
Mdot_inner/Edd = 3, 5
Rout = 300 rg
source_shape = compact_c2
f_s = 0.05, 0.10, 0.30 first
torque_delta_l_fraction = 0, +0.005
stream_heating = 0
wind = 0
```

This is the actual hot-branch bridge test.

---

### Phase 7 — Add stream heating before wind

After the no-wind compact branch passes validation, add stream heating cautiously.

Suggested scan:

```text
stream_heating_efficiency = 0.001, 0.003, 0.01, 0.03
```

Start tiny.

Heating acceptance criteria:

```text
heating energy budget closes;
Lrad/LEdd increases smoothly;
H/R does not jump pathologically;
Rson moves smoothly;
f_adv diagnostics remain interpretable;
no artificial outer/source interval_E wall appears;
no single-cell residual defect dominates.
```

---

### Phase 8 — Add wind last

Do not add wind yet.

Wind becomes justified only if one of these is true:

```text
1. Mdot = 3 or 5 finite stream-fed no-wind branch cannot be continued;
2. the no-wind branch becomes too luminous or too thick;
3. stream heating breaks no-wind energy closure;
4. the equilibrium/stability map lacks a viable high branch;
5. observations require mass/energy removal not representable by no-wind/heating models.
```

When wind is finally added, start with the simplest conservative sign convention:

```math
\frac{d\dot M}{d\ln R} = \dot M'_w - \dot M'_s.
```

Use a local energy-limited wind before adding angular-momentum lever arms:

```text
l_w = l first
then scan l_w = lambda_w l only if needed
```

---

## 6. Acceptance Criteria Before Claiming a Robust No-Wind Compact Stream Branch

A point should be called robust only if:

```text
1. residual:
   final_full <= 3e-6 preferred;
   final_full <= 1e-5 acceptable only for exploratory endpoints.

2. mass/source budget:
   source_integral/Mdot_inner agrees with f_s to <= 1e-4;
   Mdot_outer/Mdot_inner agrees with 1 - f_s to <= few x 1e-4.

3. mesh convergence:
   N = 640, 768, 896 all converge under residual-aware remeshing;
   diagnostics stable across N.

4. remesh robustness:
   physical diagnostics do not change after residual-aware remesh;
   residual peaks do not simply migrate to another unresolved cell.

5. boundary robustness:
   outer_slope_picard or soft/Robin closure reduces outer_omega;
   physical diagnostics remain unchanged under reasonable boundary correction.

6. source-shape robustness:
   compact_c2 and prior source forms give the same qualitative branch
   at fixed f_s, after normalization.

7. physics smoothness:
   Rson changes smoothly;
   H/R changes smoothly;
   f_adv_global and f_adv_inner change smoothly;
   Lrad/LEdd changes smoothly.

8. cost sanity:
   nfev does not grow catastrophically with N;
   no repeated need for df_s < 1e-4 except near a documented fold/front.
```

---

## 7. Minimal Next Run Matrix

Run this before adding heating or wind:

```text
Checkpoint validation:
    f_s = 0.70, 0.80, 0.82
    N = 640, 768, 896
    source_shape = compact_c2
    residual_remesh = on
    outer_picard = off, on

Upward pilot:
    start from N=896 f_s=0.82 compact_c2 residual-remeshed checkpoint
    f_s targets = 0.8225, 0.825, 0.83, 0.84, 0.85
    residual_remesh = every step and on reject
    outer_picard = on

High-Mdot bridge after infrastructure passes:
    Mdot_inner/Edd = 3, 5
    Rout -> 300 rg first with no stream
    then compact_c2 stream source f_s = 0.05, 0.10, 0.30
    torque_delta_l_fraction = 0, +0.005
    heating = 0
    wind = 0
```

---

## 8. Codex-Ready Prompt

```text
Review the latest compact-source results and implement the next numerical
infrastructure before adding heating or wind.

Current accepted status:
- No-wind stream-fed branch:
    Mdot_inner/Edd = 2
    Rout = 300 rg
    f_s = 0.82
    torque_delta_l_fraction = +0.005
    source shape = compact_c2
- Direct tanh-to-compact replacement had huge seed residual ~1.75e1.
- Normalized source-shape homotopy succeeded.
- Full compact endpoint:
    final_full = 5.693e-06
    dominant = outer_omega
    Mdot_outer/Mdot_inner = 0.18
    source_integral/Mdot_inner = 0.82
    f_adv_global = 0.2042
    f_adv_inner = 0.09517
    Lrad/LEdd = 0.8672
    max H/R = 0.2269
    Rson = 4.66 rg
- Plain N remaps and PCHIP remaps fail from large outer/source interval_E
  defects.
- Residual-aware remeshing validates the endpoint:
    N=640 final_full = 1.137e-06 strict
    N=896 final_full = 1.043e-06 strict
    physical diagnostics stable across N=640/768/896.

Interpretation:
- The f_s=0.82 compact no-tail source branch is conditionally mesh-supported
  when residual-aware remeshing is part of the method.
- It is not proven robust under naive grid remaps.
- The tanh tail at Rout is not the main cause of the branch wall.
- The remaining limiter is outer angular closure / slope refresh plus
  source-boundary residual control.
- This is a credible Mdot_inner/Edd=2 mildly advective bridge branch, not yet
  the strongly hot QPE branch.
- Do not add wind yet.

Tasks:
1. Freeze regression anchors:
   - compact_c2 f_s=0.82 N=768 original-grid endpoint;
   - compact_c2 f_s=0.82 N=640 residual-remesh strict endpoint;
   - compact_c2 f_s=0.82 N=896 residual-remesh strict endpoint.

2. Reconcile test-count metadata:
   - CODEX_COMPACT_SOURCE_SHAPE_RESULTS.md says 141 passed;
   - GPT_PROMPT_COMPACT_SOURCE_MESH_VALIDATION.md says 142 passed.

3. Promote residual-aware remeshing into the continuation loop:
   - remesh every accepted high-source step;
   - remesh on rejected step;
   - monitor = |interval_E| + stream_source_prime + |dMdot/dlnR|
               + |dQstream/dlnR| + outer-boundary-layer weight;
   - preserve source normalization and mass budget;
   - output before/after residual profiles and source integrals.

4. Implement explicit outer-slope Picard / boundary-only correction:
   - after polish, refresh outer_match_log_slopes from current state;
   - repolish;
   - damped Picard options: 0.3, 0.5, 1.0;
   - max 3-5 outer Picard iterations;
   - require physics diagnostics to remain unchanged.

5. Run f_s upward pilot from N=896 compact residual-remeshed checkpoint:
   - f_s = 0.8225, 0.825, 0.83, 0.84, 0.85;
   - residual-remesh every step;
   - outer-slope Picard enabled;
   - cost-aware step controller;
   - stop if nfev repeatedly >150, df_s <1e-4, or residual localizes
     into one unresolved outer/source cell.

6. If pilot remains expensive, implement true tangent predictor:
   - solve J_z dz/df_s = -F_f_s;
   - compare current-state, secant, tangent, and damped tangent seeds;
   - choose lowest initial residual;
   - add pseudo-arclength continuation if tangent norm suggests a fold.

7. After compact Mdot=2 branch is robust, retry the true hot branch:
   - Mdot_inner/Edd = 3 finite-Rout to 300 rg using interval_E remeshing;
   - Mdot_inner/Edd = 5 finite-Rout to 300 rg using outer-slope Picard or
     soft/Robin outer closure;
   - then add compact stream source at f_s = 0.05, 0.10, 0.30.

8. Add stream heating only after no-wind compact branch passes validation.
   Add wind only after no-wind + heating topology is understood or shown
   physically incomplete.
```

---

## 9. Bottom Line

```text
Yes, the f_s = 0.82 compact no-tail source branch is now a supported no-wind
stream-fed branch, provided residual-aware remeshing is treated as part of the
method.

The next move should be infrastructure-first:
    residual-aware remeshing inside continuation,
    outer-slope Picard / boundary-only correction,
    short f_s upward pilot,
    then return to Mdot_inner/Edd = 3 and 5.

Do not add wind yet.
Do not call the Mdot=2 branch the final hot QPE branch.
```
