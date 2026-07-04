# IMRI/QPE Minidisk Project — Mdot=5 Hot-State Interpretation and Next Plan

Date: 2026-07-04

Prepared as a Codex-facing handoff after reviewing the latest discussion and the current GitHub note:

```text
Note/CODEX_MDOT3_BUFFER_AND_MDOT5_STREAM_RESULTS.md
```

Related prior handoff context:

```text
IMRI_QPE_CONVERSATION_HANDOFF_SUMMARY(1).md
GPT_HIGH_MDOT_NOWIND_HANDOFF.md
CODEX_LATEST_MDOT5_STREAM_WIND_NEXT_PLAN.md
```

---

## 1. Executive Summary

I mostly agree with Codex's latest interpretation, with one important wording correction.

The current `Mdot_inner/Edd=5`, compact-source, stream-fed finite-minidisk solution **is a real high-Mdot advective/slim steady branch**. It is not fake, and it is not merely the old weakly advective `Mdot_inner/Edd=2` branch. At `f_s=0.80`, it has roughly:

```text
Mdot_inner/Edd         = 5
Rout                  = 335 rg
Rinj                  ≈ 240 rg in the fiducial case
f_s                   = 0.80
Mdot_outer/Mdot_inner = 0.20
f_adv_global          ≈ 0.499
f_adv_inner           ≈ 0.472
Lrad/LEdd             ≈ 1.300
max H/R               ≈ 0.315
Rson                  ≈ 4.361 rg
```

That is a strong finite stream-fed slim-disk anchor.

However, Codex is right that **adding conservative stream heating does not produce a new, stronger, hotter, more advective branch**. In the heating ladder, increasing `eta_heat` to `1.0` makes the disk only slightly thicker and more luminous, while the global advective fraction decreases:

| case | `f_adv_global` | `f_adv_inner` | `Lrad/LEdd` | max `H/R` | `Rson/rg` | integrated `Qstream/Qvisc` |
|---|---:|---:|---:|---:|---:|---:|
| no heating | ~0.499 | ~0.4715 | ~1.300 | ~0.315 | ~4.361 | 0 |
| `eta_heat=0.1` | ~0.496 | ~0.4713 | ~1.315 | ~0.316 | ~4.361 | ~3.27e-3 |
| `eta_heat=1.0` | ~0.458 | ~0.4608 | ~1.455 | ~0.326 | ~4.349 | ~3.29e-2 |

So the precise statement should be:

```text
The current Mdot=5 solution is a robust steady high-Mdot slim/advective anchor,
but it is not yet the QPE high state of a reservoir-fed limit cycle. Conservative
stream heating alone does not trigger a new stronger advective/hot branch; most
of the added heat appears as extra luminosity.
```

This distinction matters. The project has likely found the **steady upper-branch building block**, but not yet the **self-consistent eruptive QPE high state**.

---

## 2. Current Project Status

### 2.1 Standard no-wind high-Mdot slim branch

The standard single-BH no-wind slim branch has already been recovered to `Mdot/Edd=5`.

Certified high-Mdot benchmark at `Mdot/Edd=5`:

```text
residual        = 2.293e-6
f_adv_global    = 0.4534
f_adv_inner     = 0.4666
Lrad/LEdd       = 1.541
max H/R         = 0.3164
Rson            = 4.360 rg
```

Interpretation:

```text
The solver can recover a real high-Mdot advective slim-disk branch when the true
inner accretion rate is raised.
```

### 2.2 Finite stream-fed no-wind branch at Mdot_inner/Edd=2

The finite stream-fed branch at `Mdot_inner/Edd=2` has become numerically credible to very high source fraction, but it is only mildly advective.

Representative status:

```text
Mdot_inner/Edd  = 2
f_s             ≈ 0.91 certified in recent work
f_adv_global    ≈ 0.20
f_adv_inner     ≈ 0.095
max H/R         ≈ 0.227
```

Interpretation:

```text
This is a useful stream-fed bridge branch, but not the strongly advective QPE high state.
```

### 2.3 Finite stream-fed no-wind branch at Mdot_inner/Edd=3

The `Mdot_inner/Edd=3` compact branch has been supported at least to `f_s=0.50`.

Representative diagnostics:

```text
Mdot_inner/Edd = 3
f_s            = 0.50
f_adv_global   ≈ 0.3304
f_adv_inner    ≈ 0.266
Lrad/LEdd      ≈ 1.081
max H/R        ≈ 0.2676
Rson           ≈ 4.502 rg
```

Interpretation:

```text
Advection increases substantially relative to Mdot=2, supporting the idea that
true inner Mdot is the main hot-branch control parameter.
```

### 2.4 Finite stream-fed no-wind branch at Mdot_inner/Edd=5

This is the major current success.

At `Mdot_inner/Edd=5`, the compact source branch has now been continued to `f_s=0.80` with stable physical diagnostics.

Representative sequence:

| `f_s` | `Mdot_outer/Mdot_inner` | `f_adv_global` | `f_adv_inner` | `Lrad/LEdd` | max `H/R` | `Rson/rg` |
|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 0.70 | ~0.488 | ~0.471 | ~1.342 | ~0.316 | ~4.361 |
| 0.50 | 0.50 | ~0.493 | ~0.472 | ~1.324 | ~0.316 | ~4.361 |
| 0.80 | 0.20 | ~0.499 | ~0.472 | ~1.300 | ~0.315 | ~4.361 |

Robustness checks already performed:

```text
N checks at f_s=0.30: N=640, 768, 896 stable.
Buffer checks at f_s=0.50 and f_s=0.80: R_buffer=295, 300, 305 rg stable.
Source geometry scan at f_s=0.80: Rinj/Rout=0.70, 0.75, 0.80 survives.
Compatible source-aware Robin closure homotopy: survives without physical changes.
```

Interpretation:

```text
The finite stream-fed minidisk can access a high-Mdot advective/slim branch.
This answers an earlier major uncertainty. The remaining problem is no longer
'can the branch exist?' but 'does this branch become the high state of a QPE
limit cycle once reservoir loading, stability, and wind are included?'
```

---

## 3. Clarifying the Semantic Issue: What Does “Hot Branch” Mean?

There are three different meanings that should not be conflated.

### Meaning A: Standard high-Mdot slim/advective branch

Question:

```text
Can the solver recover a high-Mdot slim disk with strong advection?
```

Answer:

```text
Yes. The standard no-wind branch at Mdot/Edd=5 is recovered.
```

### Meaning B: Finite stream-fed high-Mdot steady branch

Question:

```text
Can the finite stream-fed minidisk access a high-Mdot advective branch?
```

Answer:

```text
Yes, at least in the no-wind compact-source setup. The Mdot_inner/Edd=5,
f_s=0.80 solution is strongly advective and robust under several checks.
```

### Meaning C: QPE high state of a limit cycle

Question:

```text
Does the model now demonstrate the high state of a self-consistent QPE cycle?
```

Answer:

```text
Not yet.
```

The QPE high state is not merely a steady solution with high imposed `Mdot_inner`. It must be reached by reservoir loading, drain the disk, connect to a lower branch through instability/hysteresis, and probably include wind or mass-energy loss.

Therefore, the best wording is:

```text
The current Mdot=5 stream-fed branch is a strong steady upper-branch anchor,
but not yet the full reservoir-triggered QPE high state.
```

---

## 4. Do We Agree With Codex About Stream Heating?

Yes.

Codex's diagnostic argument is sound:

```text
If conservative stream heating created a stronger advective/hot branch, we might
expect clear increases in f_adv_global, f_adv_inner, H/R, entropy/temperature
support, or a meaningful sonic-structure change.
```

But the actual heating ladder shows:

```text
eta_heat=0:
    f_adv_global ≈ 0.499
    Lrad/LEdd    ≈ 1.300
    max H/R      ≈ 0.315
    Rson         ≈ 4.361 rg

eta_heat=1:
    f_adv_global ≈ 0.458
    Lrad/LEdd    ≈ 1.455
    max H/R      ≈ 0.326
    Rson         ≈ 4.349 rg
```

Also:

```text
integrated Qstream/Qvisc at eta_heat=1 ≈ 3.3e-2
```

So the heating term is globally modest even when local `max Qstream/Qvisc` looks large. The pointwise maximum is denominator-sensitive and should not be used as the main branch indicator.

Conclusion:

```text
Conservative stream heating mostly increases luminosity and slightly thickens
the disk. It does not trigger a separate stronger advective branch.
```

This suggests the next missing physics is likely:

```text
wind / mass-energy loss / angular-momentum loss / reservoir-regulated stability,
not more conservative heating alone.
```

---

## 5. What State Are We Actually Looking For?

We are looking for the **upper state of a reservoir-fed QPE limit cycle**, not simply a solution with an even larger `f_adv`.

Desired cycle skeleton:

```text
cool stable minidisk
→ stream-fed mass accumulation
→ radiation-pressure/slim instability
→ rapid rise of Mdot_inner
→ advective and/or wind-regulated hot state
→ disk drains
→ cooling transition
→ stream-fed reload
→ repeat
```

The desired high state should have these properties:

### 5.1 Reached by reservoir loading, not imposed by hand

Right now, the solver imposes:

```text
Mdot_inner/Edd = 5
```

For a QPE limit cycle, the high state should emerge when a reservoir variable crosses a threshold, such as:

```text
Mdisk
Sigma_out
outer entropy/load parameter
stream supply normalization
```

The real control relation should eventually be something like:

```text
Mdot_inner = Mdot_inner(Mdisk)
```

or:

```text
Mdot_inner = Mdot_inner(Sigma_out)
```

### 5.2 Drains the disk

During the high state, the inner drain plus wind loss should exceed the external reload rate:

```text
Mdot_inner + Mdot_wind > Mdot_stream_supply
```

That is what makes it an eruption rather than a steady high-Mdot disk.

### 5.3 Advective and/or wind-regulated

The high state does not require `f_adv` to increase monotonically under every added physics term.

With wind, the energy equation should become:

```text
Q_visc + Q_stream = Q_rad + Q_adv + Q_wind
```

Then the high state could be:

```text
advective-dominated:
    large Qadv, weak wind

wind-regulated:
    moderate Qadv, significant Qwind, capped luminosity

mixed:
    both Qadv and Qwind important
```

Once wind is added, `f_adv` might decrease because wind is carrying the excess energy. That would not necessarily be bad.

### 5.4 Sits on an S-curve or folded equilibrium map

A QPE limit cycle needs an equilibrium/stability structure, not just a smooth imposed-Mdot sequence.

Desired branch structure:

```text
low stable branch:
    low Mdot_inner
    thin/cool
    slow stream-fed loading

middle unstable branch:
    radiation-pressure dominated
    thermal/viscous instability

upper branch:
    high Mdot_inner
    advective and/or wind-regulated
    rapid draining
```

The current `Mdot=5` branch is probably an upper-branch anchor, but the map and stability labels still need to be built.

---

## 6. What We Understand Now

### 6.1 True inner Mdot is the main advection control knob

The trend is now clear:

```text
Mdot_inner/Edd=2:
    f_adv_global ≈ 0.204
    f_adv_inner  ≈ 0.095

Mdot_inner/Edd=3:
    f_adv_global ≈ 0.330
    f_adv_inner  ≈ 0.266

Mdot_inner/Edd=5:
    f_adv_global ≈ 0.499
    f_adv_inner  ≈ 0.472
```

The branch becomes hot/slim primarily because the true inner accretion rate is high, not because the source fraction or conservative heating alone forces a transition.

### 6.2 The finite stream-fed geometry no longer appears to kill the high-Mdot branch

Earlier, this was a major uncertainty. Now the `Mdot=5`, `f_s=0.80` branch survives:

```text
high source fraction,
compact source,
source geometry scan,
small outer-buffer shifts,
compatible source-aware Robin closure check.
```

This is a big positive result.

### 6.3 Conservative stream heating is not the missing trigger

The heating ladder is smooth but not transformative:

```text
f_adv_global decreases at large eta_heat,
Lrad increases,
H/R increases only mildly,
Rson barely moves,
integrated stream heating remains modest.
```

### 6.4 Pointwise max Qstream/Qvisc is not a reliable branch diagnostic

The local maximum can be huge because the denominator can be small. The integrated heating ratio is more reliable.

### 6.5 Numerical infrastructure is much healthier

The project now has working tools for:

```text
compact C2 source,
source-shape homotopy,
residual-aware remeshing,
physical interval-E gate,
source-fraction continuation,
high-Mdot finite stream branch continuation.
```

The remaining technical caveat is that compatible Robin closure currently passes when the target is source-aware, but the Newton/Jacobian machinery does not yet strongly reduce deliberately mismatched Robin angular residuals.

---

## 7. What We Do Not Yet Know

### 7.1 Whether the branch is dynamically stable

A steady branch is not automatically stable. Need thermal/viscous stability labels.

### 7.2 Whether the model has a folded equilibrium map

The key missing product is:

```text
Mdot_inner(Mdisk) or Mdot_inner(Sigma_out)
```

with stable/unstable segments and turning points.

### 7.3 Whether the disk naturally jumps to Mdot_inner/Edd≈3–5

Right now, high `Mdot_inner` is imposed. The next phase must determine whether reservoir loading can produce the jump.

### 7.4 Whether wind creates the physically relevant high state

Wind may:

```text
remove mass,
carry energy,
remove angular momentum,
cap luminosity,
alter branch stability,
create or erase turning points.
```

The current no-wind branch is a necessary anchor, not the final physical model.

### 7.5 Whether the model can produce repeated QPE-like cycles

Still missing:

```text
time-dependent reload/drain calculation,
cycle period,
burst duration,
burst amplitude,
Delta M per cycle,
connection to observed QPE luminosity and temperature.
```

---

## 8. Main Problem Right Now

The main problem is no longer:

```text
Can the finite stream-fed minidisk access a high-Mdot advective branch?
```

That now appears to be yes.

The main problem is:

```text
How do we turn the recovered steady high-Mdot stream-fed branch into a
self-consistent QPE high state produced by reservoir loading, regulated by
advection/wind, and connected to a cool branch through instability and hysteresis?
```

So the project should now move from **branch recovery** to **physical cycle construction**.

---

## 9. What We Expect To See Next

### 9.1 If no-wind Mdot=5 is continued from f_s=0.80 upward

Expected behavior, if the branch remains smooth:

```text
f_s:                     0.80 → 0.90
Mdot_outer/Mdot_inner:   0.20 → 0.10
f_adv_global:            stays near ~0.50
f_adv_inner:             stays near ~0.47
Lrad/LEdd:               stays around ~1.29–1.30, maybe slightly lower/higher depending source geometry
max H/R:                 stays near ~0.315
Rson:                    stays near ~4.36 rg
```

This would further certify the high-source no-wind branch, but it would still not prove a limit cycle.

### 9.2 If conservative heating is increased further

Expected behavior:

```text
more Lrad,
slightly larger H/R,
no major increase in f_adv,
sonic radius nearly unchanged,
likely no new branch transition.
```

Recommendation:

```text
Do not spend much effort pushing eta_heat beyond 1 unless needed for diagnostics.
```

### 9.3 If wind is added

At fixed `Mdot_inner`, a wind sink should change the mass budget roughly as:

```text
Mdot_outer/Mdot_inner ≈ 1 - f_stream + f_wind
```

Qualitative expectations:

```text
Mdot_outer/Mdot_inner increases relative to no-wind at fixed Mdot_inner and f_s.
Lrad/LEdd may flatten or decrease if wind removes excess energy.
H/R should not jump pathologically.
Rson should move smoothly.
f_adv may decrease if Qwind carries energy previously advected.
```

Important: a lower `f_adv` after adding wind is not automatically a failure. It may mean wind has become the regulating channel.

---

## 10. Concrete Next Plan

### Phase 1 — Freeze regression anchors

Freeze the following as named checkpoints and tests:

```text
A. Standard no-wind Mdot/Edd=5 benchmark
   f_adv_global = 0.4534
   f_adv_inner  = 0.4666
   Lrad/LEdd    = 1.541
   max H/R      = 0.3164
   Rson         = 4.360 rg

B. Finite stream-fed Mdot_inner/Edd=5, compact source, f_s=0.30
   N=640/768/896 stable

C. Finite stream-fed Mdot_inner/Edd=5, compact source, f_s=0.50
   buffer sensitivity passed

D. Finite stream-fed Mdot_inner/Edd=5, compact source, f_s=0.80
   f_adv_global ≈ 0.499
   f_adv_inner  ≈ 0.472
   Lrad/LEdd    ≈ 1.300
   max H/R      ≈ 0.315
   Rson         ≈ 4.361 rg

E. Heating ladder at Mdot=5, f_s=0.80
   eta_heat = 0, 0.1, 1.0
```

The purpose is to prevent future wind or stability work from accidentally breaking the validated no-wind branch.

---

### Phase 2 — Do a short no-wind f_s extension, but do not over-focus on it

Run:

```text
Mdot_inner/Edd = 5
Rout = 335 rg
compact C2 source
N = 896 baseline
eta_heat = 0
f_s = 0.825, 0.85, 0.875, 0.90
```

Acceptance:

```text
physical interval-E <= 1e-5
mass budget closes
f_adv_inner, f_adv_global, Lrad, H/R, Rson smooth
residual-aware remeshing does not become pathological
```

This is useful certification, but not the main science target. Stop if it starts consuming disproportionate effort.

---

### Phase 3 — Improve source-aware Robin / outer closure machinery

The compatible Robin closure check is encouraging but not complete. Implement:

```text
source-aware Robin target refresh inside Newton/Picard loop
outer angular residual row scaling
Jacobian treatment for Robin target dependence
boundary-only correction / outer-slope Picard if needed
```

Tests:

```text
Mdot=5, f_s=0.80, eta_heat=0
Mdot=5, f_s=0.80, eta_heat=1.0
Rinj/Rout=0.70, 0.75, 0.80
Robin chi=0, 0.25, 0.5, 1.0
```

Acceptance:

```text
Newton actively reduces outer angular residual,
not merely accepts a seed close to the target.
Physical diagnostics unchanged under compatible closure variation.
```

This matters before using Robin closure for deliberately different reservoir conditions.

---

### Phase 4 — Add a controlled bookkeeping wind sink first

Do not jump immediately to a complicated wind. Start with a controlled sink to verify signs, mass budget, and angular-momentum bookkeeping.

Continuity equation:

```text
dMdot/dlnR = Mdot_wind_prime - Mdot_stream_prime
```

Start with:

```text
Mdot_inner/Edd = 5
f_s = 0.80
eta_heat = 0, 0.1, 1.0
f_wind = 0.01, 0.03, 0.10
l_w = l
same compact geometry as the validated no-wind anchor
```

Diagnostics:

```text
Mdot_outer/Mdot_inner
source_integral/Mdot_inner
wind_integral/Mdot_inner
mass budget closure
angular momentum budget closure
f_adv_global
f_adv_inner
Lrad/LEdd
max H/R
Rson
residual block dominance
```

Expected mass-budget check:

```text
Mdot_outer/Mdot_inner ≈ 1 - f_s + f_wind
```

for a simple integrated wind fraction normalized to `Mdot_inner`.

---

### Phase 5 — Add energy-limited wind

Once bookkeeping wind works, implement a minimal energy-limited vertical wind.

Use:

```text
Q_heat = Q_visc + Q_stream
```

```text
Q_wind = epsilon_w * [Q_heat - Q_adv - Q_Edd,z]_+
```

with:

```text
Q_Edd,z = 2 c Omega_K^2 H / kappa
```

and:

```text
Mdot_wind_prime = 2 pi R^2 Q_wind / E_wind
```

Start with:

```text
epsilon_w = 0.01, 0.03, 0.10, 0.30
l_w = l
```

Only after that scan an angular-momentum lever arm:

```text
l_w = lambda_w l
lambda_w = 1.2, 1.5, 2.0
```

Acceptance:

```text
mass budget closes,
energy budget closes,
wind power is positive and localized where expected,
Lrad changes smoothly,
H/R remains within height-integrated/slim regime,
Rson moves smoothly,
no artificial outer/source residual wall appears.
```

Scientific goal:

```text
Determine whether the high-Mdot branch becomes wind-regulated rather than purely advective.
```

---

### Phase 6 — Build the finite-minidisk equilibrium map

This is now the most important conceptual step.

Parameterize steady models by a reservoir/loading variable, not only imposed `Mdot_inner`:

```text
Mdisk
Sigma_out
outer entropy/load parameter
stream supply normalization
```

For each sequence, output:

```text
Mdot_inner
Lrad
Mdot_wind
Mdot_outer
f_adv_global
f_adv_inner
max H/R
Rson
Mdisk
thermal stability label
viscous/drain time estimate
```

Look for:

```text
lower stable branch,
unstable middle branch,
upper advective/wind-regulated branch,
turn-on threshold,
turn-off threshold,
hysteresis.
```

Thermal stability diagnostic:

```text
(∂(Q+ - Q-) / ∂T)_Sigma
```

where:

```text
Q+ = Q_visc + Q_stream
Q- = Q_rad + Q_adv + Q_wind
```

Use finite perturbations around steady solutions if analytic derivatives are inconvenient.

---

### Phase 7 — Estimate cycle viability before time-dependent simulation

Before a full time-dependent run, estimate:

```text
Delta M_cyc between lower and upper turning points,
stream refill time = Delta M_cyc / Mdot_stream_supply,
high-state drain time = Mdisk / (Mdot_inner + Mdot_wind - Mdot_stream_supply),
expected burst duty cycle,
peak Lrad,
wind kinetic/mechanical luminosity if applicable.
```

Compare qualitatively with QPE requirements:

```text
recurrence time,
burst duration,
amplitude,
soft thermal temperature,
peak luminosity.
```

---

### Phase 8 — Time-dependent reload/drain prototype

After the equilibrium map shows a plausible S-curve/hysteresis, implement a reduced time-dependent model:

```text
dMdisk/dt = Mdot_stream_supply - Mdot_inner(Mdisk) - Mdot_wind(Mdisk)
```

with interpolation over the steady equilibrium map.

Then proceed to a full radial time-dependent disk only if the reduced model shows repeated cycles.

---

## 11. Acceptance Criteria Before Claiming a QPE High State

Do not claim the final QPE high state until the following are true:

```text
1. The high branch is reached by a reservoir/loading variable, not just imposed Mdot_inner.
2. The high state drains the disk: Mdot_inner + Mdot_wind > Mdot_stream_supply.
3. The branch has a stability/turning-point context.
4. There is hysteresis between turn-on and turn-off thresholds.
5. Mass, angular momentum, and energy budgets close.
6. Lrad, H/R, Rson, f_adv, and Mdot_wind are mesh-stable.
7. The model can estimate recurrence time and burst duration.
8. A reduced or full time-dependent model produces repeated cycles.
```

The current `Mdot=5`, `f_s=0.80` branch satisfies none of the time-dependent/cycle criteria yet, but it is probably the correct steady upper-branch anchor to build from.

---

## 12. Codex-Ready Prompt

```text
Please use the current Mdot_inner/Edd=5 compact-source results as a robust
steady upper-branch anchor, but do not call them the final QPE high state yet.

Interpretation:
- The Mdot=5, f_s=0.80 no-wind compact-source finite-minidisk solution is a real
  high-Mdot advective/slim steady branch:
    f_adv_global ≈ 0.499
    f_adv_inner ≈ 0.472
    Lrad/LEdd ≈ 1.300
    max H/R ≈ 0.315
    Rson ≈ 4.361 rg
- This branch survives buffer shifts, compact-source geometry changes, and a
  compatible source-aware Robin closure check.
- However, adding conservative stream heating does not create a separate stronger
  hot/advective branch. At eta_heat=1.0:
    f_adv_global drops to ≈0.458
    Lrad/LEdd rises to ≈1.455
    max H/R rises only mildly to ≈0.326
    Rson barely changes to ≈4.349 rg
    integrated Qstream/Qvisc is only ≈3.3e-2
- Therefore the current branch is a steady high-Mdot upper-branch anchor, not yet
  the QPE high state of a reservoir-fed limit cycle.

Main next goal:
Move from branch recovery to physical cycle construction.

Tasks:
1. Freeze regression anchors:
   - standard no-wind Mdot/Edd=5 benchmark;
   - stream-fed Mdot=5 compact source f_s=0.30, 0.50, 0.80;
   - heating ladder at f_s=0.80 for eta_heat=0, 0.1, 1.0.

2. Optional short no-wind source-fraction extension:
   - Mdot_inner/Edd=5, f_s=0.825, 0.85, 0.875, 0.90;
   - stop if numerically expensive; this is certification, not the main science.

3. Improve source-aware Robin/outer closure machinery:
   - refresh Robin target with full stream/source fields;
   - add outer-slope Picard or boundary-only correction;
   - ensure Newton actively reduces outer angular residual, not just accepts a
     seed that already satisfies the compatible target.

4. Add controlled bookkeeping wind first:
   - dMdot/dlnR = Mdot_wind_prime - Mdot_stream_prime;
   - Mdot_inner/Edd=5, f_s=0.80;
   - eta_heat=0, 0.1, 1.0;
   - f_wind=0.01, 0.03, 0.10;
   - l_w=l initially;
   - verify mass and angular-momentum budgets.

5. Add energy-limited wind after bookkeeping wind passes:
   - Q_heat = Q_visc + Q_stream;
   - Q_wind = epsilon_w [Q_heat - Q_adv - Q_Edd,z]_+;
   - Q_Edd,z = 2 c Omega_K^2 H / kappa;
   - epsilon_w = 0.01, 0.03, 0.10, 0.30;
   - l_w=l first, then lambda_w=1.2, 1.5, 2.0 if needed.

6. Build the finite-minidisk equilibrium map:
   - parameterize by Mdisk, Sigma_out, outer entropy/load, or stream supply;
   - output Mdot_inner, Lrad, Mdot_wind, f_adv, H/R, Rson;
   - add thermal/viscous stability labels;
   - look for lower stable branch, unstable middle branch, upper hot/wind branch,
     and hysteresis.

7. Estimate cycle viability:
   - Delta M_cyc;
   - refill time;
   - high-state drain time;
   - duty cycle;
   - burst luminosity and duration.

8. Only after the equilibrium map shows a plausible hysteresis loop should we
   implement a reduced or full time-dependent reload/drain calculation.

Acceptance criteria for claiming QPE high state:
- high state is reached by reservoir loading, not imposed Mdot_inner only;
- disk drains during high state;
- mass, energy, angular momentum budgets close;
- branch has stability/turning-point context;
- hysteresis exists;
- repeated cycles can be produced in a reduced or full time-dependent model.
```

---

## 13. Bottom Line

Codex is right that conservative stream heating did **not** uncover a new stronger advective/hot branch. But the `Mdot_inner/Edd=5`, `f_s=0.80` solution is still extremely important: it is a robust steady high-Mdot stream-fed slim branch and likely the upper-branch anchor needed for the QPE model.

The state we are actually looking for is not simply:

```text
larger f_adv after adding heat
```

It is:

```text
a reservoir-triggered, draining, radiation-pressure/slim upper state, regulated
by advection and probably wind, connected to a cool branch through instability
and hysteresis, and capable of repeated reload/drain cycles.
```

The next missing physics is therefore:

```text
wind + equilibrium/stability map + cycle construction,
not more conservative heating alone.
```
