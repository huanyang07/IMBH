# Codex Handoff: Mdot=5 Energy-Limited Wind Endpoint Review and Next Plan

Date: 2026-07-04

Repository: `huanyang07/IMBH`

Reviewed latest GitHub state, especially commit `04515c9` (`Add high-Mdot wind endpoint continuation results`) and these files:

- `Note/CODEX_MDOT5_WIND_SPRINT_RESULTS.md`
- `Note/GPT_PROMPT_MDOT5_WIND_ENDPOINT_REVIEW.md`
- `src/imri_qpe/layer3_minidisk_1d/winds.py`
- `src/imri_qpe/layer3_minidisk_1d/transonic_local.py`
- `src/imri_qpe/layer3_minidisk_1d/transonic_collocation.py`
- `scripts/run_standard_slim_stream_mass_annulus_scan.py`
- high-wind endpoint output tables, especially the raw-epsilon and eta-continuation rows

---

## Executive verdict

The latest result is a **major numerical success**. The previous strict continuation wall near

```text
epsilon_w ~ 0.065
```

was almost certainly **not physical branch loss**. It was a Newton/Jacobian defect in the active wind contribution to the interval energy residual. After adding the wind-aware interval-energy Jacobian, the branch continues strictly to very high wind efficiencies:

```text
raw epsilon continuation:
  epsilon_w = 0.98      residual ~ 2.16e-10, int Qwind/Qvisc ~ 0.198
  epsilon_w = 0.997     residual ~ 1.21e-09, int Qwind/Qvisc ~ 0.690

eta continuation, eta = -log(1 - epsilon_w):
  eta = 6.20, epsilon_w = 0.997970569, residual ~ 3.68e-10, int Qwind/Qvisc ~ 0.783
  eta = 6.30, epsilon_w = 0.998163695, residual ~ 1.46e-10, int Qwind/Qvisc ~ 0.802
  eta = 6.35, epsilon_w = 0.998253253, residual ~ 1.59e-10, int Qwind/Qvisc ~ 0.811
```

So the current main problem is **no longer basic continuation of the energy-wind branch**.

The main problem is now **physical interpretation and validation**:

```text
Is this high-epsilon branch a real wind-regulated high-Mdot state,
or is it a closure/activation/smoothing artifact of the current energy-only wind sink?
```

My answer: **it is a credible candidate wind-regulated high-Mdot steady branch, but not yet a certified physical wind branch.**

The biggest caveat is that the current energy-limited wind appears to be an **energy sink only** in these runs:

```text
wind_sink_fraction = 0
wind_sink_integral_over_inner = 0
Mdot_outer/Mdot_inner remains ~0.20
```

while the energy wind sink becomes huge:

```text
int Qwind/Qvisc ~ 0.811 at eta = 6.35.
```

That is useful as a cooling/energy-loss experiment, but a physical wind should eventually remove **mass and angular momentum** as well as energy.

---

## Current project status

The project now has several important pieces in place:

```text
1. Standard no-wind slim disk:
   recovered to Mdot/Edd = 5.
   This remains the clean high-Mdot advective backbone.

2. Finite stream-fed no-wind minidisk:
   Mdot_inner/Edd = 5, Rout = 335 rg, f_s = 0.80 is strict and strongly advective.

3. Conservative stream heating:
   does not create a separate stronger hot branch.
   Most added heat radiates rather than increasing global advection.

4. Bookkeeping wind:
   mass-budget/sign convention works:
       Mdot_outer/Mdot_inner ~= 1 - f_s + f_wind.

5. Energy-limited wind, energy-only form:
   now continues to very high epsilon_w after the wind-aware interval_E Jacobian.
```

The project has therefore moved from **branch recovery** to **branch certification and physical closure**.

---

## What we understand

### 1. The no-wind `Mdot=5` stream-fed branch is real

At `Mdot_inner/Edd=5`, `Rout=335 rg`, `f_s=0.80`, no heating, no wind, the solution is a strong high-Mdot slim/advective anchor:

```text
f_adv_global ~ 0.499
f_adv_inner ~ 0.472
Lrad/LEdd ~ 1.300
max H/R ~ 0.315
Rson ~ 4.361 rg
```

This is not the weak `Mdot=2` bridge branch. It is the finite stream-fed continuation of the high-Mdot slim backbone.

### 2. Conservative stream heating is not the missing hot-branch trigger

The heating ladder showed:

```text
case            f_adv_global   Lrad/LEdd   max H/R   Rson
no heating      ~0.499         ~1.300      ~0.315    ~4.361 rg
eta_heat=0.1    ~0.496         ~1.315      ~0.316    ~4.361 rg
eta_heat=1.0    ~0.458         ~1.455      ~0.326    ~4.349 rg
```

So stream heating makes the disk slightly brighter and thicker, but it does not create a separate stronger advective branch.

### 3. The previous wind wall at `epsilon_w ~ 0.065` was numerical

Before the wind-aware Jacobian, strict continuation stalled at an `interval_E` floor near `R ~ 65.8 rg`.

After adding the active-wind derivative into the interval energy Jacobian, continuation became strict through:

```text
epsilon_w = 0.98
raw epsilon = 0.997
eta = 6.35, epsilon_w = 0.998253253
```

This is decisive evidence that the old wall was a corrector/Jacobian issue, not physical branch loss.

### 4. Eta is a better endpoint coordinate than raw epsilon

Near `epsilon_w -> 1`, raw epsilon becomes a poor continuation coordinate. The eta coordinate

```text
eta = -log(1 - epsilon_w)
```

is better because it resolves the endpoint asymptotically. The current `eta=6.4` attempt was too aggressive, while a smaller `6.30 -> 6.35` step succeeded. That looks like endpoint stiffness / step-size control, not a clean branch end.

### 5. The high-wind branch changes character

As wind feedback grows:

```text
int Qwind/Qvisc rises from ~0.20 at epsilon_w=0.98
                   to ~0.81 at eta=6.35.

Lrad/LEdd falls from ~1.30 in the no-wind state
             to ~0.54 at eta=6.35.

max H/R falls from ~0.315 in the no-wind state
           to ~0.141 at eta=6.35.

Rson moves outward from ~4.36 rg
                 to ~5.23 rg.

f_adv_global falls from ~0.499
             through ~0
             to slightly negative at eta=6.35.
```

This is exactly what a strong energy sink might do: wind carries energy that advection previously carried, leaving a cooler/thinner high-Mdot flow.

But this could also be a closure artifact unless it survives validation.

---

## What we do not understand yet

### 1. Whether the high-epsilon state is a physical wind or an energy-only cooling closure

The present energy-limited wind appears to remove energy without removing mass:

```text
wind_sink_fraction = 0
wind_sink_integral_over_inner = 0
Mdot_outer/Mdot_inner = 0.20
```

A physical disk wind should normally obey something like:

```text
dMdot/dlnR = Mdot_wind_prime - Mdot_stream_prime
Qwind = Mdot_wind_prime * E_wind / (2 pi R^2)
Jwind = Mdot_wind_prime * l_w
```

or else explicitly state why the sink is an energy-only loss channel rather than a mass-loaded wind.

### 2. Whether negative advection is physical

At high eta, `f_adv_global` becomes small/negative and `f_adv_inner` is negative. That may be physical if wind extracts most of the heat and reverses/reduces the entropy gradient.

But it may also be an artifact of:

```text
- the softened vertical-Eddington trigger,
- the chosen chi_edd = 0.99,
- the activation width fraction = 0.005,
- the energy-only nature of the wind sink,
- outer-buffer weighting,
- or the differential residual formulation.
```

Do not call the high-eta endpoint a physical wind-regulated branch until this is checked.

### 3. Whether the branch is mesh independent

The current high-wind endpoint is at `N=896`. It has excellent residuals, but the earlier project repeatedly showed that high-source/high-wind fronts can be mesh- and remap-sensitive. The high-eta states need N and residual-localization validation.

### 4. Whether exact `epsilon_w=1` matters

Probably not yet. The physically meaningful controls are not `epsilon_w` itself but:

```text
integrated Qwind/Qvisc
Mwind/Mdot_inner, once mass-coupled
Lrad/LEdd
H/R
Rson
stability / position on the equilibrium map
```

Chasing exact `epsilon_w=1` is less useful than validating the already strong `Qwind/Qvisc ~ 0.2-0.8` states.

---

# Answers to Codex's specific questions

## 1. Is the wind-aware interval_E Jacobian physically/numerically consistent?

**Numerically: yes, provisionally.**

It is consistent with the current residual form:

```text
Q_visc + Q_stream - Q_rad - Q_adv - Q_wind = 0
```

provided the Jacobian uses the correct signs:

```text
R_E = Q_avail - Q_rad - Q_wind
Q_avail = Q_visc + Q_stream - Q_adv
Q_wind = epsilon_w * S(Q_avail - chi_edd Q_Edd,z)

therefore:

dR_E = dQ_avail - dQ_rad - dQ_wind

dQ_wind = epsilon_w * S' * (dQ_avail - chi_edd dQ_Edd,z)
```

The fact that the old `epsilon_w ~ 0.065` wall vanished and the strict ladder reaches high epsilon is strong evidence that the Jacobian is much closer to the true local differential residual.

**Physically: only for the current energy-only closure.**

It is not yet a complete physical wind Jacobian unless the wind also couples to:

```text
- continuity via Mdot_wind_prime,
- angular momentum via l_w,
- and the radial Mdot profile.
```

### Required Jacobian audits

Add or run these before treating the high-wind branch as certified:

```text
A. Directional finite-difference audit of the full interval_E Jacobian:
   points = epsilon_w 0, 0.98, 0.997, eta 6.20, eta 6.35
   directions = random, tangent, temperature-localized, wind-activation-localized
   report median/max relative derivative error.

B. Activation derivative audit:
   report dQwind/dQavail and dQwind/dQedd profiles.
   report active interval fraction, transition-zone fraction, cap-active fraction.

C. No-wind regression:
   epsilon_w = 0 must reproduce the previous no-wind Jacobian and no-wind anchors.

D. Width-zero limit:
   if wind_activation_width_fraction -> 0, hard-trigger rows should converge to the old hard-threshold behavior away from the kink.

E. Cap derivative audit:
   if Qwind uses min(excess, max(Qavail, 0)), report where the cap is active.
   The derivative must switch consistently or be smoothed.
```

---

## 2. Best endpoint strategy: adaptive eta, pseudo-arclength, or beta solve?

Recommended order:

```text
1. Adaptive eta stepping
2. Pseudo-arclength only if adaptive eta shows fold/stall signatures
3. Beta = 1 - epsilon_w endpoint solve only as a diagnostic/asymptotic fit, not primary
```

### Why adaptive eta first

Eta already works better than raw epsilon. The failed `eta=6.4` step had a large initial residual, but the smaller `6.30 -> 6.35` step succeeded. That is a classic step-size-control problem.

Implement:

```text
coordinate: eta = -log(1 - epsilon_w)

initial step near current endpoint:
    d_eta = 0.025

adaptive rules:
    if initial_full < 0.01 and nfev < 8:
        grow d_eta by 1.2
    if initial_full > 0.03 or nfev > 20:
        shrink d_eta by 0.5
    if initial_full > 0.08 or nfev > 50:
        reject and shrink d_eta by 0.25
```

Suggested next eta targets:

```text
6.375, 6.400, 6.425, 6.450
```

but only after validation at `eta=6.2-6.35` starts.

### When to use pseudo-arclength

Switch to pseudo-arclength only if you see evidence of a fold or parameter-coordinate failure:

```text
- tangent norm diverges,
- tangent direction changes abruptly,
- smallest singular value collapses,
- adaptive eta repeatedly shrinks below d_eta ~ 1e-3,
- accepted solutions reverse direction in physical diagnostics.
```

### Why beta is not primary

The variable

```text
beta = 1 - epsilon_w
```

is useful for endpoint asymptotics, but `beta -> 0` is numerically stiff and exact `beta=0` is not obviously a physical target. Eta is just `-log beta`, so it is the safer endpoint coordinate.

Use beta only to fit endpoint trends:

```text
Lrad(beta), H/R(beta), Rson(beta), Qwind/Qvisc(beta), f_adv(beta)
```

not as the main solve coordinate.

---

## 3. Validation required before calling this a real advective/wind hot branch

Do **not** call the high-eta state a certified physical wind branch yet.

Validation gates:

### Gate A: N/mesh validation

Run at:

```text
epsilon_w = 0.98
raw epsilon_w = 0.997
eta = 6.20
eta = 6.35
```

with:

```text
N = 768, 896, 1024
```

Acceptance:

```text
final_full <= 1e-6 preferred, <= few e-6 acceptable
physical_E <= same order
f_adv_global stable to < 2%
f_adv_inner stable to < 2-3%
Lrad/LEdd stable to < 1-2%
max H/R stable to < 1-2%
Rson stable to < 1e-2 rg
int Qwind/Qvisc stable to < 2-3%
```

### Gate B: residual localization

For each high-wind point, dump profiles:

```text
interval_E(R)
interval_R(R)
physical_E and buffer_E partitions
outer_omega
Qvisc(R)
Qrad(R)
Qadv(R)
Qwind(R)
Q_Edd,z(R)
Qavail(R)
activation argument = Qavail - chi Q_Edd,z
dQwind/dQavail
dQwind/dQedd
H/R
T
Sigma
Mdot(R)
```

Acceptance:

```text
no unresolved single-cell interval_E spike;
peak residual location explained and stable with N;
physical and buffer partitions both clean;
outer_omega does not dominate the honest physical residual.
```

### Gate C: wind closure sensitivity

At fixed target wind strength, not fixed epsilon, compare:

```text
chi_edd = 0.995, 0.990, 0.985
wind_activation_width_fraction = 0.001, 0.0025, 0.005, 0.010
```

Use target states such as:

```text
int Qwind/Qvisc ~ 0.2
int Qwind/Qvisc ~ 0.5
int Qwind/Qvisc ~ 0.8
```

Acceptance:

```text
qualitative branch behavior is stable;
negative-advection transition does not appear/disappear only because of width;
launch/activation region is physically interpretable;
active interval fraction is not just a smoothing artifact.
```

### Gate D: outer-boundary and buffer sensitivity

Run high-wind states with:

```text
R_buffer = 290, 300, 310 rg
outer buffer weights varied by factor ~3
outer closure = current pressure_supported_thin_energy and source-aware Robin/matched variant if available
```

Acceptance:

```text
same high-wind physics;
no dependence on buffer weighting strong enough to change f_adv sign or H/R trend.
```

### Gate E: mass-coupled wind

Before final physical interpretation, implement a wind mass sink tied to the energy sink:

```text
Mdot_wind_prime = 2 pi R^2 Qwind / E_wind
```

with a simple first escape-energy model, for example:

```text
E_wind = eta_esc * GM / (2R)
eta_esc = 1 initially
```

and angular momentum loss:

```text
l_w = lambda_w * l
lambda_w = 1 initially
then 1.2, 1.5, 2.0
```

Mass continuity must become:

```text
dMdot/dlnR = Mdot_wind_prime - Mdot_stream_prime
```

Acceptance:

```text
Mdot_outer/Mdot_inner increases by the integrated wind mass loss;
energy budget closes;
angular momentum budget closes;
branch survives with smooth Rson, H/R, and luminosity;
wind mass loss is not absurdly larger than the disk supply.
```

Without Gate E, the current result should be described as:

```text
high-Mdot energy-loss / wind-cooling branch candidate
```

not yet:

```text
fully physical mass-loaded wind branch.
```

---

## 4. Are small/negative advection diagnostics at high eta physical or closure artifact?

Current best answer:

```text
They are plausible but not yet proven physical.
```

Why they could be physical:

```text
- Qwind carries energy that Qadv previously carried.
- Lrad/LEdd drops strongly.
- H/R drops strongly.
- Rson moves outward smoothly.
- Residuals remain tiny.
```

This is consistent with a **wind-cooled high-Mdot state**, not necessarily an advective hot state.

Why they could be artifact:

```text
- the wind currently removes energy but not mass/angular momentum;
- wind_activation_width_fraction = 0.005 may produce broad weak activation;
- wind_active_interval_fraction can reach 1.0 in smoothed runs;
- chi_edd = 0.99 is a model parameter, not yet physically derived;
- high-eta endpoint is sensitive to continuation coordinate;
- mesh/N/closure validation is not done yet.
```

Recommended language:

```text
At high eta, the solution is no longer advective-dominated.
It is a candidate wind-regulated / wind-cooled high-Mdot state.
The negative advective fraction is not automatically bad, but it is not yet a certified physical result.
```

---

## 5. Next priority: validation, diagnostics, or continuation improvements?

Priority order:

```text
1. Validate the existing high-wind branch.
2. Audit the wind-aware Jacobian and energy budget.
3. Add mass/AM-coupled wind physics.
4. Only then improve endpoint continuation further.
```

Do **not** make the next main goal “push epsilon closer to 1.”

The current branch already reaches:

```text
int Qwind/Qvisc ~ 0.81
```

which is dynamically significant. That is enough to start physics validation.

---

# Concrete next implementation plan

## Phase 0: Freeze current anchors

Freeze these checkpoint anchors:

```text
No-wind Mdot=5 stream-fed:
  f_s = 0.80, eta_heat = 0, epsilon_w = 0

Weak/intermediate wind:
  epsilon_w = 0.50
  epsilon_w = 0.80
  epsilon_w = 0.98

High raw-epsilon wind:
  epsilon_w = 0.997

High eta wind:
  eta = 6.20, epsilon_w = 0.997970569
  eta = 6.30, epsilon_w = 0.998163695
  eta = 6.35, epsilon_w = 0.998253253
```

For each anchor, save:

```text
full residual
physical_E and buffer_E
interval_E peak radius
outer_omega
Qwind/Qvisc integrated
Qwind/Qrad integrated
Qwind/Qadv_abs integrated
Lrad/LEdd
f_adv_global
f_adv_inner
f_adv_pos
max H/R
Rson
Mdot_outer/Mdot_inner
source integral
wind mass integral, if any
```

## Phase 1: Wind Jacobian audit

Implement or run:

```text
scripts/audit_wind_interval_jacobian.py
```

Test points:

```text
epsilon_w = 0
0.98
0.997
eta = 6.20
eta = 6.35
```

Report:

```text
median relative directional derivative error
max relative directional derivative error
best finite-difference step
error by residual block: interval_R, interval_E, outer, sonic
error localized near wind activation/cap regions
```

Pass criteria:

```text
median relative error < 1e-4 preferred, < 1e-3 acceptable
max relative error localized and explained
no sign error in active wind intervals
no degradation of no-wind Jacobian
```

## Phase 2: N/mesh validation

Run:

```text
N = 768, 896, 1024
```

at:

```text
epsilon_w = 0.98
eta = 6.20
eta = 6.35
```

Use the same `Rout=335 rg`, `Rinj=240 rg`, `f_s=0.80`, `chi=0.99`, `width=0.005`, `eta_heat=0` setup.

Output one table:

```text
wind_state, N, residual, physical_E, buffer_E, peak_E_R,
Qwind/Qvisc, Lrad, f_adv_global, f_adv_inner, f_adv_pos, H/R, Rson
```

## Phase 3: Closure sensitivity at fixed wind strength

Do **not** compare only at fixed epsilon.

For each target:

```text
int Qwind/Qvisc = 0.2, 0.5, 0.8
```

find solutions across:

```text
chi_edd = 0.995, 0.990, 0.985
width = 0.001, 0.0025, 0.005, 0.010
```

This answers whether the high-wind state is robust or just a consequence of one soft-trigger choice.

## Phase 4: Add mass-coupled wind

Implement an optional mode:

```text
wind_mass_coupled = True
E_wind = eta_esc * GM / (2R)
Mdot_wind_prime = 2 pi R^2 Qwind / E_wind
l_w = lambda_w * l
```

Start with:

```text
eta_esc = 1
lambda_w = 1
```

Use a homotopy between current energy-only sink and mass-coupled wind:

```text
zeta_masswind = 0, 0.25, 0.50, 0.75, 1.0
```

where:

```text
zeta_masswind = 0: current energy-only wind
zeta_masswind = 1: fully mass/AM-coupled wind
```

Test first at moderate wind:

```text
int Qwind/Qvisc ~ 0.2, epsilon_w ~ 0.98 equivalent
```

then high wind:

```text
int Qwind/Qvisc ~ 0.5 and 0.8
```

Expected result if physical:

```text
Mdot_outer/Mdot_inner should increase above 0.20.
The increase should match the integrated Mdot_wind_prime budget.
```

## Phase 5: Adaptive eta stepping, but only after validation begins

Implement cost-aware eta stepping:

```text
start from eta = 6.35
try d_eta = 0.025
```

Step controller:

```text
if accepted and initial_full < 0.01 and nfev < 8:
    d_eta *= 1.2
elif accepted and nfev < 20:
    keep d_eta
elif initial_full > 0.03 or nfev > 20:
    d_eta *= 0.5
elif initial_full > 0.08 or nfev > 50:
    reject and d_eta *= 0.25
```

Stop if:

```text
d_eta < 1e-3 repeatedly;
physical diagnostics jump;
Jacobian audit fails;
N validation fails;
mass-coupled wind cannot reproduce energy-only behavior smoothly.
```

## Phase 6: Pseudo-arclength only if needed

Add pseudo-arclength continuation if adaptive eta suggests a fold or coordinate singularity:

```text
F(z, eta) = 0
arclength constraint: t_z dot (z - z0) + t_eta (eta - eta0) = ds
```

Do not spend major time on this before validating the already achieved high-wind states.

## Phase 7: Return to the QPE limit-cycle problem

Once the wind branch is validated and mass-coupled, build the finite-minidisk equilibrium map.

Control variable should be a reservoir/disk variable, not imposed `Mdot_inner` only:

```text
Mdisk
Sigma_out
stream supply normalization
outer entropy/load parameter
```

Outputs:

```text
Mdot_inner
Lrad
Mdot_wind
f_adv_global
f_adv_inner
H/R
Rson
thermal/viscous stability label
```

Look for:

```text
low stable branch
unstable middle branch
upper advective/wind-regulated branch
turn-on threshold
turn-off threshold
hysteresis
reload/drain mass budget
```

A steady high-wind solution is not yet a QPE limit cycle.

---

# Codex-ready task prompt

```text
Review commit 04515c9 and continue from the Mdot_inner/Edd=5, Rout=335 rg,
f_s=0.80, compact-source, no-heating, energy-limited wind branch.

Interpretation:
- The wind-aware interval_E Jacobian fixed the old epsilon_w~0.065 wall.
- The strict branch now reaches epsilon_w=0.98, raw epsilon_w=0.997,
  and eta=6.35 / epsilon_w=0.998253253 with residual ~1.6e-10.
- This is a major numerical success, but not yet a certified physical
  mass-loaded wind branch.
- The current energy wind appears to be an energy sink only: wind_sink_fraction=0,
  wind_sink_integral=0, while int Qwind/Qvisc reaches ~0.811.
- High eta produces a wind-cooled state: Lrad and H/R drop, Rson moves outward,
  and f_adv_global crosses near/under zero. This may be physical, but it may also
  be a closure/smoothing artifact.

Next tasks, in order:

1. Freeze anchors:
   - epsilon_w = 0, 0.50, 0.80, 0.98, 0.997
   - eta = 6.20, 6.30, 6.35
   - record full residual, physical_E, buffer_E, Qwind/Qvisc, Lrad, f_adv,
     H/R, Rson, Mdot budget, wind mass integral.

2. Add/run wind interval_E Jacobian audit:
   - directional finite-difference check at epsilon_w=0, 0.98, 0.997,
     eta=6.20, eta=6.35.
   - report derivative errors by residual block and location.
   - include activation/cap derivative diagnostics.

3. Validate high-wind states with N checks:
   - N = 768, 896, 1024.
   - states: epsilon_w=0.98, eta=6.20, eta=6.35.
   - require stable Qwind/Qvisc, Lrad, f_adv, H/R, Rson, residual localization.

4. Test closure sensitivity at fixed wind strength, not fixed epsilon:
   - target int Qwind/Qvisc = 0.2, 0.5, 0.8.
   - chi_edd = 0.995, 0.990, 0.985.
   - wind_activation_width_fraction = 0.001, 0.0025, 0.005, 0.010.
   - determine whether negative advection and H/R collapse are robust.

5. Implement mass/AM-coupled energy wind:
   - Mdot_wind_prime = 2 pi R^2 Qwind / E_wind,
     with E_wind = eta_esc GM/(2R), eta_esc=1 first.
   - l_w = lambda_w l, lambda_w=1 first.
   - continuity: dMdot/dlnR = Mdot_wind_prime - Mdot_stream_prime.
   - homotopy from energy-only to mass-coupled wind with zeta=0,0.25,0.5,0.75,1.
   - require Mdot_outer/Mdot_inner to increase consistently with integrated wind mass.

6. Only after validation begins, add adaptive eta stepping:
   - start from eta=6.35 with d_eta=0.025.
   - shrink/grow based on initial residual and nfev.
   - switch to pseudo-arclength only if adaptive eta shows fold-like behavior.

7. Do not chase exact epsilon_w=1 as the main goal.
   The physically meaningful targets are Qwind/Qvisc, Mwind/Min, Lrad, H/R,
   Rson, and stability on the equilibrium map.

8. After mass-coupled wind is validated, build the finite-minidisk equilibrium
   map versus Mdisk/Sigma_out/stream loading and label stability. The final QPE
   claim requires a reload/drain limit cycle, not just a steady high-wind solution.
```

---

## Bottom line

The new wind-aware Jacobian is a real breakthrough. It turns the old `epsilon_w~0.065` wall into a solved numerical issue and exposes a high-wind, high-Mdot branch with `Qwind/Qvisc` approaching order unity.

But the next step should be **certification and physical closure**, not simply pushing closer to `epsilon_w=1`.

The immediate priorities are:

```text
1. Jacobian directional audit.
2. N/mesh and residual-localization validation.
3. Closure sensitivity in chi_edd and activation width.
4. Coupling Qwind to wind mass and angular momentum loss.
5. Then adaptive eta / pseudo-arclength if the endpoint still matters.
```
