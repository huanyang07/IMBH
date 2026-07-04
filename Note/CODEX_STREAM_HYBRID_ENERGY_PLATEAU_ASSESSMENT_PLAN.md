# Codex Handoff: Stream-Fed Hybrid Energy Plateau Assessment and Next Plan

Date: 2026-07-03  
Repository: `huanyang07/IMBH`  
Starting note: `Note/GPT_PROMPT_STREAM_HYBRID_ENERGY_PLATEAU.md`

## Executive verdict

The current obstruction just above

```text
f_s ≈ 0.898078125
```

is **not convincing evidence for physical loss of the stream-fed branch**. It is also **not mainly a source-fraction predictor problem** and not something likely to be solved by simply increasing `N` or doing ordinary grid remaps.

The main problem right now is:

```text
A physical differential energy residual floor / plateau near the
source + outer-buffer transition, caused by inconsistency between
the energy residual objective, grid transfer/refinement, and the
Newton/Jacobian scaling used near the high-source boundary layer.
```

The solver can still find excellent weighted/integrated collocation solutions, and the physical disk diagnostics remain smooth and stable. But once acceptance is judged by the honest physical differential energy residual, the branch is pinned at about

```text
physical_E ≈ 3e-5
```

with the latest clean anchor at:

```text
f_s = 0.898078125
Mdot_inner/Edd = 2
Rout = 335 rg
Rinj = 240 rg
source shape = compact_c2
torque_delta_l_fraction = +0.005
N = 896
interval form = integrated_physical_energy
physical_E = 2.989e-5
```

The next move should be **energy-residual numerical infrastructure**, not wind, stream heating, or more brute-force `f_s` continuation.

---

## Current project status

### 1. Standard high-Mdot no-wind slim disk

The standard no-wind slim-disk benchmark remains the strongest hot-branch anchor.

At `Mdot/Edd = 5`:

```text
residual = 2.293e-6
f_adv_global = 0.4534
f_adv_inner(R<20rg) = 0.4666
Lrad/LEdd = 1.541
max H/R = 0.3164
Rson = 4.360 rg
```

This is a real advective slim branch.

### 2. Finite stream-fed no-wind branch at Mdot_inner/Edd = 2

The finite stream-fed compact-source branch has made major progress:

```text
Mdot_inner/Edd = 2
Rout = 335 rg
Rinj = 240 rg
source shape = compact_c2
torque_delta_l_fraction = +0.005
no wind
no stream heating
```

It has now been continued to just below `f_s ≈ 0.8981`, with:

```text
Mdot_outer/Mdot_inner ≈ 0.1019
source integral/Mdot_inner ≈ 0.8982
f_adv_global ≈ 0.2043
f_adv_inner ≈ 0.0944
f_adv_pos ≈ 0.2736
Lrad/LEdd ≈ 0.8666
max H/R ≈ 0.2269
Rson ≈ 4.66 rg
```

This is a **real, mildly advective stream-fed bridge branch**, not the final QPE hot branch.

### 3. Full IMRI/QPE hot/wind/limit-cycle branch

Still not demonstrated.

The current `Mdot_inner/Edd = 2` stream-fed branch is mildly advective. The genuinely hot target remains:

```text
Mdot_inner/Edd = 3 and 5
finite Rout ~ 300 rg
compact source
then torque/heating/wind only after the no-wind branch is understood
```

---

## What changed in the latest Codex run

The latest update is important because Codex stopped accepting points based only on friendly integrated/weighted residuals.

New machinery includes:

```text
1. physical_E gate:
   raw physical-domain differential energy residual must pass.

2. integrated_physical_energy residual form:
   radial residual remains integrated,
   energy residual uses raw scaled differential energy residual.

3. corrected hybrid acceptance convention:
   when interval form is integrated_physical_energy,
   effective acceptance tolerance follows PHYSICAL_E_TOL.

4. grid homotopy tests:
   fixed-physics grid moves were tested.

5. high-N remap tests:
   N1024 target/resampled grids were tested.

6. Newton audit:
   LSMR, damping, line search, and direct solve variants were tested.
```

The result is that the old apparent progress toward larger `f_s` was partly residual-accounting optimism. The current clean frontier is now much more honest.

---

## Key numerical facts

### Physical-gated continuation

The physical gate automation showed that weighted residuals alone are not enough:

```text
f_s = 0.8985:
    weighted full = 7.557e-08
    raw physical_E = 3.864e-05
    rejected

f_s = 0.89825:
    weighted full = 6.532e-08
    raw physical_E = 3.340e-05
    rejected
```

Interpretation:

```text
The weighted collocation solution can look excellent while the honest
physical differential energy audit fails.
```

### Differential cleanup

A differential-form cleanup pass did not move the frontier in any meaningful way:

```text
f_s = 0.8985000:
    solver full = 7.594e-08
    physical_E = 3.883e-05
    rejected

f_s = 0.8982500:
    solver full = 6.609e-08
    physical_E = 3.379e-05
    rejected

f_s = 0.8981250:
    solver full = 6.028e-08
    physical_E = 3.082e-05
    rejected

f_s = 0.8980625:
    solver full = 5.738e-08
    physical_E = 2.934e-05
    accepted
```

The note correctly interprets this as a discretization/formulation issue near the source/outer-buffer transition, not a predictor failure.

### Targeted remeshing and N1024

Local target remeshing and direct high-N remapping failed to preserve the clean anchor:

```text
aggressive target remesh at f_s=0.8980625:
    remesh physical_E = 5.464e-03

gentle target remesh to f_s=0.898125:
    remesh physical_E = 3.669e-02

N1024 focused target grid at f_s=0.8980625:
    physical_E = 2.256e-02

N1024 resampled current grid at f_s=0.8980625:
    physical_E = 1.098e-03
```

This means the current issue is **not** solved by “just add more nodes.” Direct grid movement/remapping injects large differential-energy defects.

### Grid homotopy

Even tiny fixed-physics grid homotopy steps break the raw physical energy audit:

```text
eta = 0.0015625:
    physical_E = 1.835e-04

eta = 0.00078125:
    physical_E = 1.004e-04

eta = 0.000390625:
    physical_E = 5.891e-05
```

The weighted residual remains excellent, so the old weighted/integrated objective can walk onto a nearby numerically good but physically dirty state.

### Hybrid physical-energy residual

The new `integrated_physical_energy` residual form is conceptually right because it makes the Newton objective care about the same physical differential energy residual used by the audit.

But it currently stalls just above the gate:

```text
f_s = 0.898125:
    short run physical_E = 3.112e-05
    long run physical_E = 3.086e-05

f_s = 0.89809375:
    physical_E = 3.062e-05
```

The quarter-step ladder gives the latest clean point:

```text
f_s = 0.898078125:
    physical_E = 2.989e-05
    nfev = 74
    accepted

f_s = 0.89809375:
    physical_E = 3.025e-05
    nfev = 71
    rejected
```

The eighth-step retry from the new anchor did not help:

```text
target f_s = 0.8980859375:
    physical_E = 3.116e-05
    nfev = 69
    rejected
```

This is the clearest sign that source step size is not the main problem.

---

## Diagnosis ranking

### 1. Residual-objective/Jacobian/energy-block conditioning: primary

The hybrid residual is the right direction, but the Newton correction is now plateauing at the energy gate. Evidence:

```text
- final_full = physical_E in forced hybrid mode;
- physical_E stalls near 3.0e-5;
- more Newton iterations help weakly and are expensive;
- LSMR uses ~5000-6000 iterations per Newton step;
- LSMR residual norm plateaus;
- direct sparse Newton directions are too aggressive;
- lower damping makes the solve worse;
- accepted line-search steps still do not push physical_E cleanly below gate.
```

This points to poor energy-block scaling/preconditioning and/or an inaccurate local Jacobian for the physical differential energy residual.

### 2. Differential energy formulation near source/outer buffer: co-primary

The physical peak is near the source/buffer structure:

```text
peak physical_E R ≈ 259.2 rg
buffer/interval_E peak R ≈ 333-334 rg
Rout = 335 rg
Rinj = 240 rg
outer buffer inner = 300 rg
```

That geometry is suspicious. The differential energy residual is probably acting like a sensitive local truncation-error detector near a source/buffer transition.

A conservative finite-volume energy balance may be a more physically natural square residual, while the pointwise differential residual should remain an audit/convergence estimator.

### 3. Mesh/grid transfer: important but not solvable by naive remap

The mesh is part of the solution. Current direct remaps, PCHIP, target grids, and fixed-physics grid homotopy all break the physical audit. That does not mean the branch is fake. It means grid transfer is not defect-preserving.

Use nested/local interval splitting or conservative prolongation, not arbitrary grid movement.

### 4. Outer boundary closure: secondary right now

`outer_omega` has been a recurring problem in earlier runs. But in the latest plateau, the decisive acceptance failure is `physical_E` / `interval_E`, while outer_omega is only at the few `1e-6` level in the shown rows.

So do not lead with outer-slope Picard here. Keep it available, but the next primary fix should target the energy residual.

### 5. Predictor limitation: not the main issue

The current/eighth/quarter step behavior shows that smaller `df_s` alone does not solve the plateau. Tangent seeds may be imperfect, but the branch is already close enough that the remaining failure is not mainly a predictor problem.

### 6. Physical loss of branch: least supported

The physical disk diagnostics are almost boringly stable:

```text
f_adv_global ≈ 0.2043
f_adv_inner ≈ 0.0944
Lrad/LEdd ≈ 0.8666
max H/R ≈ 0.2269
Rson ≈ 4.66 rg
```

There is no sonic failure, no H/R blow-up, no advection collapse, and no luminosity discontinuity. The wall is numerical/discretization/formulation dominated.

---

## Is the `physical_E <= 3e-5` gate too strict?

It is a good **provisional anchor gate** because it caught false progress from the old weighted residual. Do not discard it casually.

But it should not be treated as a universal physics criterion until mesh/order convergence is demonstrated. The current misses are tiny:

```text
3.025e-5 vs 3.000e-5
3.062e-5 vs 3.000e-5
3.086e-5 vs 3.000e-5
```

That is close enough to the apparent discretization floor that the right standard is not one fixed number at one grid. The right standard is:

```text
a mesh-converged physical residual criterion plus stable physical diagnostics.
```

Recommended wording:

```text
Use physical_E <= 3e-5 for strict anchor status at N=896,
but define scientific robustness by N/refinement convergence and stable
global diagnostics, not by a single barely missed threshold.
```

---

## Recommended next sequence

### Step 1: Freeze the current clean frontier

Freeze this as the honest latest anchor:

```text
checkpoint:
outputs/checkpoints/high_mdot_stream_outer_buffer_phys_gate_hybrid_quartersteps_accept3e5/
phys_gate_hybrid_quartersteps_accept3e5_mass_0p898078125_torque_0p005_mdot_2_N896.npz

f_s = 0.898078125
physical_E = 2.989e-05
Mdot_outer/Mdot_inner ≈ 0.101922
source_integral ≈ 0.8982
f_adv_global ≈ 0.2043
f_adv_inner ≈ 0.09443
f_adv_pos ≈ 0.2736
Lrad/LEdd ≈ 0.8666
max H/R ≈ 0.2269
Rson ≈ 4.66 rg
```

Also explicitly mark the old `f_s≈0.90` scout as **not a scientific anchor**.

### Step 2: Add plateau autopsy diagnostics

Before changing more solver machinery, write one focused diagnostic table for:

```text
anchor:
    f_s = 0.898078125

failed near points:
    f_s = 0.8980859375
    f_s = 0.89809375
    f_s = 0.898125
```

For each point, output interval-wise data around:

```text
R ≈ 259.2 rg
R ≈ 333-334 rg
```

Required columns:

```text
R_left, R_mid, R_right, dlnR
physical_interval_E
signed physical_interval_E
scaled physical_interval_E
unscaled energy numerator
energy denominator / scale
Qvisc
Qrad
Qadv
Qstream_mass_term
Qtorque_or_angular_source_term
dMdot/dlnR
stream_source_prime
Omega, dOmega/dlnR
Sigma, T, H/R
mesh spacing
row scale
column scales for local variables
linearized residual prediction vs actual residual after step
```

Purpose:

```text
Determine whether the 3e-5 plateau is caused by a real local energy-balance
defect, a denominator/scaling artifact, a derivative/source interpolation
problem, or a Jacobian linearization mismatch.
```

### Step 3: Implement energy-focused hybrid Newton merit and scaling

For forced hybrid solves, do not let the line search judge steps only by a generic square residual. Add an energy-focused block merit.

Suggested merit:

```text
phi =
    max(
        ||R_nonenergy||_2 / tol_nonenergy,
        ||R_energy_phys||_inf / physical_E_tol,
        ||R_energy_phys||_2 / physical_E_tol_2
    )
```

or a smooth Huber/max blend:

```text
phi =
    a * ||R_all||_2^2
  + b * huber_max(R_energy_phys / physical_E_tol)^2
  + c * ||R_outer||_2^2
```

Line-search rule:

```text
Accept candidate step only if it reduces:
    1. global hybrid residual, and
    2. peak physical_E or energy-focused merit,
unless explicitly in a safeguarded fallback mode.
```

Also add row/column equilibration:

```text
- scale energy rows by local Q scale and dlnR sensitivity;
- scale variables by log variable magnitudes;
- report condition/LSMR iteration counts before and after scaling;
- use preconditioned LSMR or a block diagonal preconditioner if available.
```

Acceptance for this step:

```text
At f_s = 0.89809375:
    physical_E should fall below 3e-5,
    or the audit should show clearly why it cannot.
```

### Step 4: Improve the local Jacobian for physical interval_E

The current finite-difference Jacobian is probably too weak near the source/buffer energy residual.

Implement one of these, in increasing order of work:

```text
A. local finite-difference Jacobian for energy rows only:
   - smaller variable-specific finite-difference steps;
   - per-variable relative steps;
   - row coloring for local stencil;
   - compare predicted vs actual physical_E reduction.

B. semi-analytic energy-block Jacobian:
   - analytic derivatives of Qvisc, Qrad, source terms, and dMdot terms
     where easy;
   - finite differences only for difficult closure terms.

C. local patch Newton:
   - freeze variables outside windows around R≈259.2 and R≈333-334;
   - solve only local variables plus matching constraints;
   - use this as a diagnostic and possibly as a smoother.
```

Minimal diagnostic:

```text
At f_s = 0.89809375, take the failed hybrid solution.
Run a local patch solve centered on the peak physical_E intervals.
If the local patch solve drops physical_E below 3e-5, the problem is global
Newton/preconditioning. If it cannot, the problem is residual formulation or
discretization floor.
```

### Step 5: Add a conservative finite-volume energy residual option

The pointwise physical differential residual is useful, but near a source/buffer transition the physically correct discrete equation may be better represented as a conservative interval energy balance.

Add a new residual option:

```text
interval_residual_form = conservative_physical_energy
```

Definition:

```text
For each interval:
    energy residual =
        Δ(advective/mechanical energy flux)
      - ∫(Qvisc + source work - Qrad) 2πR dR
```

or, in the current notation:

```text
integral over interval of:
    Qvisc + Qstream_or_torque_terms - Qrad - Qadv
```

but using the same source normalization and quadrature as the mass/angular momentum equations.

Important:

```text
Do not hide the differential residual.
Still audit raw physical differential_E separately.
```

Acceptance convention:

```text
scientific anchor requires:
    conservative energy residual clean,
    physical differential_E either <= 3e-5 or converging with refinement,
    physical diagnostics stable.
```

This avoids the old mistake of accepting only integrated residuals while still acknowledging that a finite-volume equation may be the right discrete physics near source terms.

### Step 6: Use nested/defect-preserving refinement, not arbitrary remap

Direct remapping has failed repeatedly. The next mesh test should avoid moving old nodes.

Implement nested local interval splitting:

```text
1. Start from clean f_s = 0.898078125 N896 checkpoint.
2. Identify top K physical_E intervals near R≈259.2 and buffer intervals near R≈333-334.
3. Split those intervals by inserting new nodes.
4. Keep old nodes fixed.
5. Initialize new-node variables by high-order local interpolation.
6. Re-solve at fixed f_s with hybrid/conservative energy objective.
7. Accept refined grid only if the old clean anchor remains clean or improves.
```

Then retry:

```text
f_s = 0.8980859375
f_s = 0.89809375
f_s = 0.898125
```

This test distinguishes:

```text
If nested refinement preserves the anchor and lowers physical_E:
    the plateau was mostly mesh/truncation.

If nested refinement preserves physics but not differential_E:
    reformulate the energy residual / audit scaling.

If nested refinement breaks the anchor like previous remaps:
    the transfer/prolongation or differential audit is too fragile.
```

### Step 7: Only after Steps 2-6, resume f_s continuation

Resume source-fraction continuation only after the plateau test is clearer.

Pilot targets:

```text
f_s = 0.8980859375
f_s = 0.89809375
f_s = 0.898125
f_s = 0.89825
f_s = 0.899
```

Settings:

```text
interval form = integrated_physical_energy or conservative_physical_energy
energy-focused merit = on
energy-block scaling = on
local energy Jacobian = on
nested refinement = on if needed
physical gate = on
```

Stop conditions:

```text
physical_E stalls above gate despite improved Jacobian and nested refinement;
LSMR iterations remain thousands with no energy reduction;
peak physical_E moves discontinuously with grid;
global diagnostics jump;
Rson jumps;
H/R jumps;
mass budget fails.
```

### Step 8: Outer-slope Picard later, not first

Add outer-slope Picard or soft/Robin closure after the energy plateau is understood, or only if a specific failed row is dominated by `outer_omega`.

Right now:

```text
outer_omega ~ few e-6
physical_E ~ 3e-5
```

So energy residual comes first.

### Step 9: Do not add heating or wind yet

No wind and no stream heating until the no-wind compact-source branch has a clear numerical foundation.

Required gates before heating:

```text
1. f_s ≈ 0.898 branch passes physical-energy convergence criteria.
2. Same branch is stable under N/refinement:
       N896 plus nested refinement, preferably N1024/N1152 equivalent.
3. physical diagnostics stable:
       f_adv_inner, f_adv_global, Lrad, max H/R, Rson.
4. source and mass budgets close.
5. no unresolved single-cell physical_E spike.
```

Required gates before wind:

```text
1. no-wind compact branch topology understood;
2. stream heating, if added, has an energy budget that closes;
3. Mdot_inner/Edd = 3 and 5 finite-Rout branches have been retried with
   the improved energy machinery;
4. a physical reason for wind exists:
       branch absent, too thick, too luminous, or no viable equilibrium map.
```

---

## Acceptance criteria going forward

Use three labels:

### Exploratory

```text
weighted solver residual <= 1e-5
physical_E may exceed gate
physics diagnostics smooth
not a scientific anchor
```

### Clean anchor

```text
physical_E <= 3e-5 at current N/grid
mass budget closes to <= few x 1e-4
dominant residual understood
physics diagnostics smooth
```

### Mesh-supported scientific anchor

```text
physical_E <= 3e-5
or physical_E demonstrably converges with nested/refined grid

and:

f_adv_inner stable to < 1-2%
f_adv_global stable to < 1-2%
Lrad/LEdd stable to < 1%
max H/R stable to < 1%
Rson stable to < 1e-2 rg
Mdot_outer/Mdot_inner = 1 - f_s within budget tolerance
source integral = f_s within budget tolerance
no unresolved one-cell source/buffer physical_E spike
```

Do not classify `f_s > 0.898078125` as a scientific anchor until it meets at least the clean-anchor standard.

---

## Minimal diagnostic experiment

The best next minimal experiment is a three-way test at fixed target:

```text
target f_s = 0.89809375
start = clean f_s = 0.898078125 hybrid checkpoint
```

Run three variants:

### A. Current hybrid residual, but energy-focused merit/scaling

```text
interval form = integrated_physical_energy
energy-focused max/Huber merit = on
row/column equilibration = on
```

Question:

```text
Can better Newton/Jacobian scaling reduce physical_E below 3e-5?
```

### B. Local energy patch solve

```text
freeze variables outside windows around:
    R ≈ 259.2 rg
    R ≈ 333-334 rg

solve local variables using physical_E-focused residual
```

Question:

```text
Is the plateau local and correctable, or does the local residual resist
reduction even when the patch has freedom?
```

### C. Conservative finite-volume energy residual

```text
interval form = conservative_physical_energy
physical differential_E still audited
```

Question:

```text
Is the pointwise differential audit stricter than the physically relevant
finite-volume energy balance near source/buffer terms?
```

Interpretation:

```text
If A succeeds:
    main problem was Newton scaling/merit.

If B succeeds but A fails:
    main problem was global Jacobian/preconditioning.

If C succeeds and differential_E converges with refinement:
    main problem was residual formulation.

If all fail but global physics remains smooth:
    treat physical_E=3e-5 as current discretization floor and define
    acceptance by mesh convergence, not a single N896 hard gate.

If all fail and physical diagnostics jump:
    then revisit possible physical branch obstruction.
```

---

## Codex-ready prompt

```text
Please implement the next principled numerical diagnostics for the
Mdot_inner/Edd=2 compact-source no-wind stream branch plateau.

Current honest frontier:
- Mdot_inner/Edd = 2
- Rout = 335 rg
- Rinj = 240 rg
- source shape = compact_c2
- torque_delta_l_fraction = +0.005
- outer buffer inner = 300 rg
- N = 896
- interval form = integrated_physical_energy
- physical_E gate = 3e-5
- latest clean anchor:
    f_s = 0.898078125
    physical_E = 2.989e-05
    f_adv_global ≈ 0.2043
    f_adv_inner ≈ 0.09443
    Lrad/LEdd ≈ 0.8666
    max H/R ≈ 0.2269
    Rson ≈ 4.66 rg
- next attempted points:
    f_s = 0.8980859375 -> physical_E = 3.116e-05
    f_s = 0.89809375   -> physical_E = 3.025e-05
    f_s = 0.898125     -> physical_E ≈ 3.086e-05 to 3.191e-05
- weighted/integrated residuals are excellent, but honest physical
  differential energy residual stalls near the gate.

Interpretation:
- Do not add heating or wind.
- Do not brute-force larger f_s by relaxing the gate.
- This is not convincing physical branch loss.
- It is most likely an energy residual/Jacobian/scaling/formulation problem
  near the source + outer-buffer transition.

Tasks:

1. Freeze f_s=0.898078125 as the current clean anchor.
   Mark old f_s≈0.90 scout as exploratory only, not a scientific anchor.

2. Add plateau autopsy diagnostics for:
   f_s=0.898078125, 0.8980859375, 0.89809375, 0.898125.
   Output interval-wise physical_E decomposition near R≈259.2 rg and
   R≈333-334 rg:
       signed/scaled/unscaled energy residual,
       Qvisc, Qrad, Qadv, source/torque energy terms,
       dMdot/dlnR, stream_source_prime,
       dlnR, row scale, variable scales,
       predicted vs actual residual reduction.

3. Implement an energy-focused hybrid Newton merit:
       phi = max(
           ||R_nonenergy|| / tol_nonenergy,
           ||R_energy_phys||_inf / physical_E_tol,
           ||R_energy_phys||_2 / physical_E_tol_2
       )
   or equivalent Huber/max blend.
   The line search should require reduction of the peak physical energy
   residual unless in a documented fallback.

4. Add row/column equilibration and/or preconditioning for the hybrid energy
   block. Report LSMR iterations, linear residual norm, and predicted vs actual
   physical_E reduction.

5. Improve the physical interval_E Jacobian:
   start with local finite-difference steps or colored local energy-row
   finite differences; then consider semi-analytic Qvisc/Qrad/Qadv/source
   derivatives.

6. Add a local patch solve diagnostic:
   freeze variables outside windows around R≈259.2 and R≈333-334 rg;
   solve only local variables plus matching constraints.
   If this drops physical_E below 3e-5, the issue is global
   Newton/preconditioning. If it does not, the issue is residual formulation
   or discretization floor.

7. Add a conservative finite-volume energy residual option:
       interval_residual_form = conservative_physical_energy
   Use it as the square residual while still auditing raw physical
   differential_E separately.

8. Replace arbitrary grid remapping with nested/defect-preserving refinement:
   keep old nodes fixed, split the top physical_E intervals, initialize new
   nodes locally, and repolish at fixed f_s.
   First prove the f_s=0.898078125 anchor survives nested refinement, then
   retry f_s=0.89809375 and 0.898125.

9. Keep physical_E <= 3e-5 as the strict N896 anchor gate, but define
   scientific robustness by mesh/refinement convergence and stable physical
   diagnostics, not by a single barely missed threshold.

10. Only after this plateau is diagnosed should continuation resume toward
    f_s=0.899 and beyond. Heating and wind remain gated until the no-wind
    compact branch is robust.
```

---

## Bottom line

The project has made real progress: the finite stream-fed no-wind branch at `Mdot_inner/Edd=2` now reaches almost `f_s=0.9` with smooth physical diagnostics.

But the main problem right now is no longer the stream source shape or the source-fraction predictor. It is the **honest physical differential energy residual plateau** near the source/outer-buffer transition.

The next principled solution is:

```text
energy-focused hybrid Newton merit
+ better energy-block scaling/Jacobian
+ local plateau diagnostics
+ conservative finite-volume energy residual option
+ nested defect-preserving refinement
```

Only then should Codex resume pushing `f_s`, and only much later should it add heating or wind.
