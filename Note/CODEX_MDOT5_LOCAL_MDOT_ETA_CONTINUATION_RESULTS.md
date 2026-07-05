# Codex Mdot=5 Local-Mdot Eta Continuation Results

Date: 2026-07-05

This sprint follows the Shen-diagnostic plan by adding local `Mdot(R)` residual
localization and testing launch-energy continuation from the prescribed
`zeta=0.03` power-law bridge.

New driver:

```text
scripts/run_mdot5_local_mdot_eta_continuation.py
```

The driver reuses the prototype local-Mdot residual, but now outputs:

```text
R_M(R)
inner logMdot boundary residual
interval_E(R)
Qwind/Qvisc
Mwind_prime/Mdot
Mstream_prime/Mdot
Mdot_tilde/Mdot_inner
s_eff_tilde(R)
Jacobian row/column norm arrays
```

It also supports:

```text
strict outer-buffer mode
resume from local x checkpoints
restoring saved outer slope pairs
outer-slope Picard refresh/repolish
restoring custom checkpoint grids
optional residual-aware remesh pre-pass for local-Mdot checkpoints
seed-only dry-run mode
node-preserving nested refinement seeds
integrated interval residual override
experimental inner-window pre-relaxation
```

## Direct Eta Ladder

Run:

```text
outputs/tables/m5_local_mdot_eta_continuation_zeta0p03_N96.md
```

Settings:

```text
N = 96
anchor = prescribed zeta=0.03 power-law bridge
eta_E = 100 -> 60 -> 40 -> 33.333
outer-buffer weights inherited from the prescribed bridge
```

Results:

| eta_E | final_full | mass_residual_max | peak mass R/rg | Mdot_outer/Mdot_inner |
|---:|---:|---:|---:|---:|
| 100 | 8.704e-05 | 6.807e-06 | 6.053 | 0.2293 |
| 60 | 9.589e-03 | 9.589e-03 | 6.057 | 0.2412 |
| 40 | 2.907e-02 | 2.907e-02 | 6.058 | 0.2438 |
| 33.333 | 3.353e-02 | 3.353e-02 | 6.059 | 0.2449 |

Interpretation:

```text
The direct eta_E jump sequence is not valid continuation.
eta_E=100 is close, but 100 -> 60 is too aggressive and the solve falls into a
compromise with inner-Mdot anchoring error and local mass residual near R~6 rg.
```

A bookkeeping caveat was also identified:

```text
mass_residual_max in the old prototype combined:
    inner logMdot boundary residual
    interval local mass-continuity residual

These are now reported separately.
```

## Strict Outer-Buffer Scouts

The prescribed bridge uses weak outer-buffer weights:

```text
outer_buffer_inner_rg = 300
outer_buffer_radial_weight = 1e-3
outer_buffer_energy_weight = 1e-3
outer_buffer_boundary_weight = 1e-4
```

Those weights are useful for prescribed-bridge continuation but too permissive
for judging a local wind BVP.  I therefore ran strict scouts with the outer
buffer disabled.

### N64 strict eta_E=100

Output:

```text
outputs/tables/m5_local_mdot_eta_continuation_zeta0p03_N64_strict_eta100.md
```

Result:

```text
final_full = 2.579e-04
interval_E = 2.579e-04
outer_omega = 2.270e-04
inner logMdot residual = 6.996e-06
interval mass residual max = 3.878e-06
Mdot_outer/Mdot_inner = 0.2249
```

The mass equation is already reasonably tight.  The remaining strict residual
is mostly energy/outer closure.

### N96 strict eta_E=100 with resume and outer-slope Picard

Best output:

```text
outputs/tables/m5_local_mdot_eta_continuation_zeta0p03_N96_strict_eta100_picard2.md
```

Best result:

```text
final_full = 1.575e-05
outer_omega = 1.575e-05
interval_E = 5.468e-06
inner logMdot residual = -2.246e-06
interval mass residual max = 6.140e-07
Mdot_outer/Mdot_inner = 0.22926
Lrad/LEdd = 0.52738
Rson = 5.298 rg
```

Residual localization:

```text
largest interval_E peaks around R~7.4-9.4 rg
local mass residual peaks near R~6.06 rg but is only ~6e-7
outer omega is now the maximum residual
```

Interpretation:

```text
Strict local-Mdot eta_E=100 is close but not yet accepted.
The local mass equation itself is no longer the main bottleneck at eta_E=100.
Outer angular closure and the inner wind-active energy rows set the floor.
Outer-slope Picard is definitely useful.

### N128 strict eta_E=100 from N96, mass-ODE remap

Output:

```text
outputs/tables/m5_local_mdot_eta_continuation_zeta0p03_N128_strict_eta100_massode_picard.md
```

Result:

```text
seed initial full = 3.015e-01
final_full = 4.214e-05
interval_R = 4.214e-05 at R~283 rg
interval_E = 1.948e-05 at R~7.83 rg
inner logMdot residual = 2.421e-07
interval mass residual max = 2.682e-07
Mdot_outer/Mdot_inner = 0.23282
Lrad/LEdd = 0.52706
Rson = 5.300 rg
```

Interpretation:

```text
N128 does not certify the eta_E=100 local-Mdot solution yet.
The local mass equation is excellent at N128, but the full residual is worse
than the N96 Picard result because the residual floor moves into interval_R in
the outer/source-tail region and interval_E in the inner wind-active region.
```

This is not evidence for a physical failure of the local wind branch.  It is a
grid-transfer/collocation floor: the N96 -> N128 remap starts with a large
source-annulus defect and the finite-difference least-squares correction only
partly removes it.

### Residual-remesh and higher-N scouts

I added an optional residual-aware remesh pre-pass to the eta driver.  The
monitor includes:

```text
|interval_R| + |interval_E| + |local mass residual|
+ stream source/wind gradients + |dlogMdot/dlogR| + outer-layer weight
```

Two scouts from the N128 eta_E=100 checkpoint were attempted:

```text
strength=8, blend=0.7:
    remeshed seed initial_full = 8.484e-01
    interrupted after the finite-difference Jacobian became too expensive

strength=1, blend=0.3:
    remeshed seed initial_full = 7.878e-01
    interrupted as the same cost trap appeared

pure PCHIP remap at strength=1, blend=0.3:
    remeshed seed initial_full = 8.929e-01
    interrupted

N128 -> N160 mass-ODE prolongation:
    seed initial_full = 1.912e+00
    interrupted
```

Interpretation:

```text
Fixed-N residual remeshing and simple N growth are not currently safe seed
operations for the expanded local-Mdot unknown vector.  The large defects are
created by moving/prolonging the grid through the compact source/wind-gradient
region, not by the final eta_E=100 state itself.
```

The driver now correctly restores custom checkpoint grids, so future remeshed
local-Mdot checkpoints will not silently reload onto the baseline grid.

### Seed-only dry-run and nested refinement

I added a dry-run mode:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_SEED_ONLY=1
```

This writes the usual table/profile/checkpoint without entering the expensive
finite-difference Newton/Jacobian solve.  It was used to compare remap seeds.

Corrected seed-only results from the N128 eta_E=100 checkpoint:

| seed | final_full | mass max | dominant location |
|---|---:|---:|---|
| same-grid N128 | 4.214e-05 | 2.682e-07 | interval_R at R~283 rg |
| fixed-N residual remesh, mass_ode | 7.878e-01 | 1.959e-01 | interval_E/mass near R~247 rg |
| fixed-N residual remesh, defect mass | 1.204e+00 | 2.077e-02 | interval_E near R~247 rg |
| ordinary N160 prolongation, mass_ode | 1.912e+00 | 8.754e-02 | interval_E near R~237 rg |
| ordinary N160 prolongation, defect mass | 1.984e+00 | 8.118e-03 | interval_E near R~237 rg |

Interpretation:

```text
The bad remap/prolongation seeds are mainly source-annulus energy defects.
Changing only the local-Mdot reconstruction does not solve fixed-N remeshing or
ordinary N growth.
```

I then added a node-preserving nested refinement mode:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_REMAP_METHOD=nested_mass_ode
IMBH_MDOT5_LOCAL_MDOT_ETA_REMAP_METHOD=nested_defect_preserving
```

This keeps every accepted old node and inserts new nodes in the largest gaps,
rather than moving the whole grid.

Nested seed-only results:

| seed | final_full | mass max | dominant location |
|---|---:|---:|---|
| N160 nested_mass_ode | 1.565e-01 | 1.565e-01 | mass/source annulus near R~245 rg |
| N160 nested_defect_preserving | 4.496e-03 | 1.676e-05 | interval_R near R~6.19 rg |
| N255 nested_defect_preserving | 4.556e-03 | 1.756e-05 | interval_R near R~6.15 rg |
| N136 nested_defect_preserving | 4.237e-03 | 1.483e-05 | interval_R near R~6.11 rg |

Interpretation:

```text
Node preservation plus defect-preserving mass transfer is the first seed method
that removes the catastrophic source-annulus energy defect.  The remaining seed
defect moves to the inner sonic-side radial/energy rows, which is a much more
plausible target for a local corrector.
```

However, actual N136 correction from this seed did not certify:

```text
N136 nested_defect_preserving + one Picard:
    initial_full = 4.237e-03
    final_full = 1.082e-04
    interval_E = 1.082e-04 at R~10.6 rg
    interval_R = 4.201e-05
    interval mass max = 5.037e-07
    inner logMdot residual = -9.658e-06

N136 nested_defect_preserving, no Picard:
    final_full = 1.310e-04
```

So Picard is not the cause of the worse N136 result; it helps slightly.  The
current least-squares differential residual solve slides into an inner energy
floor after the nested seed.

Integrated interval residual was exposed via:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_INTERVAL_FORM=integrated
IMBH_MDOT5_LOCAL_MDOT_ETA_INTEGRATED_WEIGHTING=none|inverse_sqrt_dx
```

Dry-run N136 nested-defect seeds were worse in integrated mode:

```text
integrated, no weighting:      initial_full = 3.601e-02
integrated, inverse_sqrt_dx:   initial_full = 2.016e-01
```

Thus integrated defects are not an immediate seed-level rescue for this local
wind BVP.

### Inner-window pre-relaxation test

I implemented an optional inner-window least-squares pre-relaxation:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_INNER_RELAX_OUTER_RG=<R/rg>
IMBH_MDOT5_LOCAL_MDOT_ETA_INNER_RELAX_INCLUDE_MDOT=0|1
IMBH_MDOT5_LOCAL_MDOT_ETA_INNER_RELAX_INCLUDE_GLOBALS=0|1
IMBH_MDOT5_LOCAL_MDOT_ETA_INNER_RELAX_ANCHOR_WEIGHT=<weight>
```

The intent was to clean the N136 nested-defect seed near R~5.7-10 rg before
global polish.  The result was negative:

| case | final seed residual | dominant problem |
|---|---:|---|
| inner relax to 8 rg, weak anchor, includes Mdot/globals | 7.300e-02 | mass/source annulus near R~250 rg |
| inner relax to 8 rg, y-only | 8.200e-02 | mass/inner rows, worse than seed |
| inner relax to 8 rg, anchor 0.1 | 2.255e-01 | mass/source annulus |
| inner relax to 8 rg, anchor 1.0 | 6.855e-02 | mass/source annulus |
| inner relax to 10-12 rg | interrupted | dense finite-difference Jacobian too slow |

Interpretation:

```text
A generic local least-squares patch is not a good pre-polish.  It over-corrects
the inner rows and destroys the already-good source-annulus/mass budget, even
when anchored.  The problem is not the choice of local radius; it is that the
local finite-difference corrector is unstructured and does not respect the
coupled mass/energy balance.
```

The code remains useful as a controlled diagnostic switch, but it should stay
off for production continuation.

### Targeted nested refinement fix

The earlier node-preserving nested refinement still had a flaw: it inserted new
nodes into the largest gaps anywhere on the grid.  For this solution those gaps
are near the inner sonic-side region, so the N136 seed created a new artificial
R~6 rg defect.

I added targeted nested insertion controls:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_NESTED_REFINE_MIN_RG=<Rmin>
IMBH_MDOT5_LOCAL_MDOT_ETA_NESTED_REFINE_MAX_RG=<Rmax>
```

New nodes are now inserted only into intervals whose midpoint lies in the
requested radial band.  If no interval lies in the band, the code falls back to
the old largest-gap rule.

Seed-only tests from the N128 eta_E=100 checkpoint:

| seed | band | final_full | mass max | dominant |
|---|---:|---:|---:|---|
| N136 targeted | 100-320 rg | 4.216e-05 | 2.878e-07 | interval_R at R~283 rg |
| N136 targeted | 180-320 rg | 2.817e-03 | 2.461e-07 | interval_R at R~213 rg |
| N136 targeted | 250-340 rg | 1.679e+00 | 1.241e-02 | source-annulus interval_E |

The broad `100-320 rg` band is the stable one.  It preserves the inner solution
and avoids the catastrophic source-annulus energy defect.

Polished staged results:

| stage | final_full | interval_R | interval_E | mass max | notes |
|---|---:|---:|---:|---:|---|
| N136, 100-320 rg | 3.586e-05 | 3.586e-05 at R~283 rg | 1.543e-05 | 1.468e-07 | cheap, stable |
| N140, 100-320 rg | 3.318e-05 | 3.318e-05 at R~289 rg | 1.487e-05 | 4.194e-07 | cheap, stable |
| N152, 100-320 rg | 2.515e-05 | 2.515e-05 at R~295 rg | 9.530e-06 | 1.271e-06 | best current checkpoint |

Attempting to continue the same broad-band ladder beyond N152 is not currently
safe:

```text
N160 from N152, 100-320 rg: seed final_full = 4.310e-04
N168 from N152, 100-320 rg: seed final_full = 1.544e+00
N176 from N152, 100-320 rg: seed final_full = 1.682e+00
```

Interpretation:

```text
The original remap/refinement bug is fixed.  The safe ladder is now
N128 -> N136 -> N140 -> N152 using broad 100-320 rg targeted nested insertion.
This lowers the eta_E=100 residual from 4.21e-5 to 2.52e-5 without creating the
old inner/source catastrophe.  The remaining residual floor is a localized
outer/source radial row around R~295 rg, with interval_E now below 1e-5.
```

This is progress but not yet a strict scientific anchor.  The next numerical
problem is the residual floor itself, not the previous bad remap.

### Residual-form, closure, and outer-band corrector audit

Starting checkpoint:

```text
outputs/checkpoints/m5_local_mdot_eta_continuation_zeta0p03_N152_strict_eta100_nested_defect_R100_320_picard/stage_00_etaE_100_N152.npz
```

Differential audit of that checkpoint:

```text
final_full = 2.515e-05
interval_R = 2.515e-05 at R~294.55 rg
interval_E = 9.530e-06 at R~7.83 rg
mass_residual_max = 1.271e-06
outer_omega = -1.161e-05
```

#### 1. Residual-form diagnosis

Seed-only residual-form audits:

| interval form | weighting | seed residual | dominant interpretation |
|---|---|---:|---|
| differential | none | 2.515e-05 | radial row at R~294.55 rg |
| integrated | none | 3.600e-02 | integrated energy at R~39 rg, not useful |
| integrated | inverse_sqrt_dx | 2.015e-01 | worse |
| integrated_physical_energy | none | 1.161e-05 | removes differential radial norm from max; energy stays differential |
| integrated_physical_energy | inverse_sqrt_dx | 1.161e-05 | same max; weighting does not matter here |
| conservative_physical_energy | none | 8.732e-02 | bad source/energy finite-volume row |
| conservative_physical_energy | inverse_sqrt_dx | 6.056e-01 | worse |

Interpretation:

```text
The remaining R~295 rg defect is mainly a radial differential collocation/norm
floor.  It is not a local mass failure and not an energy failure: the energy row
is already below 1e-5 in the physical differential audit.
```

#### 2. Mixed residual pre-polish

I polished once with:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_INTERVAL_FORM=integrated_physical_energy
```

Result:

```text
mixed residual final_full = 1.007e-05
```

Auditing that mixed-polished state back in the original differential residual
gave:

```text
differential seed residual = 4.427e-05
```

So mixed residual alone is not an acceptable physical solution.  However, using
it as a conditioning step and then resuming differential polish helped:

```text
outputs/checkpoints/m5_local_mdot_eta_polish_N152_integrated_physE_then_differential_resume/stage_00_etaE_100_N152.npz

final_full = 2.075e-05
interval_R = 2.075e-05 at R~300.49 rg
interval_E = 6.865e-06 at R~7.83 rg
mass_residual_max = 1.746e-06
outer_omega = -1.335e-05
Mdot_outer/Mdot_inner = 0.232809
Lrad/LEdd = 0.527513
Rson = 5.29806 rg
```

This is the best current eta_E=100 repaired checkpoint, but still not strict.

#### 3. Outer-closure sensitivity

Seed-only closure tests from the same N152 checkpoint:

| closure | seed residual | result |
|---|---:|---|
| pressure_supported_local_energy | 5.537e-03 | large boundary residual |
| full_slope_match | 5.537e-03 | same large boundary residual |
| pressure_supported_entropy_slope | 9.253e-01 | unusable |
| pressure_supported_robin_energy, chi=0.5 | 3.070e-04 | worse than baseline, dominated by outer_omega |

Interpretation:

```text
The existing N152 state is not close to the alternate outer closures.  These
variants add boundary residual; they do not cure the R~295 interval_R floor.
The remaining defect should not be treated as simply a hard outer-closure
artifact.
```

#### 4. Structured outer-band corrector test

I added an opt-in outer/source-band relaxer:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_OUTER_RELAX_MIN_RG=<Rmin>
IMBH_MDOT5_LOCAL_MDOT_ETA_OUTER_RELAX_MAX_RG=<Rmax>
IMBH_MDOT5_LOCAL_MDOT_ETA_OUTER_RELAX_INCLUDE_ENERGY=0|1
IMBH_MDOT5_LOCAL_MDOT_ETA_OUTER_RELAX_ANCHOR_WEIGHT=<weight>
```

Tests around the R~295 peak:

| band | rows | seed residual after relax | result |
|---|---|---:|---|
| 280-305 rg | radial only | 2.069e-03 | damages the state |
| 280-305 rg | radial+energy | 2.617e-05 | slightly worse than baseline |
| 270-310 rg | radial only | 2.634e-03 | damages the state |

Interpretation:

```text
A local outer-band least-squares corrector is not the right fix either.  Like
the inner relaxer, it tends to damage nearby coupled rows unless it is anchored
so strongly that it becomes nearly neutral.
```

## Current Best State

The best repaired eta_E=100 local-Mdot checkpoint is:

```text
outputs/checkpoints/m5_local_mdot_eta_polish_N152_integrated_physE_then_differential_resume/stage_00_etaE_100_N152.npz
```

It is not strict, but it is the cleanest state so far:

```text
full differential residual = 2.075e-05
physical energy residual < 7e-06
mass residual ~1.7e-06
remaining floor = radial differential row near R~300 rg
```

The immediate remap/refinement problem is fixed.  The remaining problem is a
radial collocation/norm floor in the source/outer transition region.

### Same-grid eta_E scout at N96

Output:

```text
outputs/tables/m5_local_mdot_eta_continuation_zeta0p03_N96_strict_eta95_90_picard2.md
```

Results:

| eta_E | seed initial full | final_full | interval_R peak | interval_E peak | interval mass max | Mdot_outer/Mdot_inner |
|---:|---:|---:|---:|---:|---:|---:|
| 95 | 1.028e-03 | 1.659e-05 | 1.442e-05 at R~331 rg | 3.430e-06 at R~38 rg | 1.323e-06 | 0.23128 |
| 90 | 1.144e-03 | 1.555e-05 | 1.410e-05 at R~331 rg | 4.581e-06 at R~7.44 rg | 1.439e-06 | 0.23354 |

Interpretation:

```text
Small eta_E steps are continuous on the same N96 grid, unlike the old direct
100 -> 60 jump.  But they do not beat the strict residual floor.  The dominant
floor is again a localized interval_R/outer-zone row plus a small mass boundary
offset, not a catastrophic local mass-loading failure.
```
```

## What This Means

The local-Mdot infrastructure improved materially:

```text
Before:
    zeta=0.03, eta_E=33.333, N96 one-shot residual ~4.6e-4
    unclear whether mass continuity, energy, or outer closure dominated

Now:
    zeta=0.03, eta_E=100, N96 strict/Picard residual ~1.6e-5
    interval local mass residual ~6e-7
    interval_E ~5.5e-6
    outer_omega ~1.6e-5
    N128 strict/Picard residual ~4.2e-5
    N128 interval local mass residual ~2.7e-7
    same-grid N96 eta_E=95,90 remain continuous but sit at ~1.6e-5
```

So the next bottleneck is not a gross failure of local mass loading.  It is:

```text
1. grid-transfer/prolongation of the expanded local-Mdot unknown vector;
2. localized interval_R residual in the outer/source-tail region;
3. energy-row/Jacobian scaling in the inner wind-active region;
4. outer angular closure refresh/Robin handling once the grid-transfer floor is lower.
```

The old direct `eta_E=100 -> 60 -> 40 -> 33.333` path should be abandoned.

## Recommended Next Move

Do not jump directly to `eta_E=60`.  First certify `eta_E=100`.

Suggested sequence:

```text
1. Do not continue eta_E lower on scientific grounds until eta_E=100 is
   mesh-transfer stable.

2. Treat fixed-N residual remeshing as unsafe for the local-Mdot BVP until the
   source-annulus energy defect can be eliminated by a state-defect-preserving
   remap, not just a mass-defect-preserving remap.

3. Keep node-preserving nested refinement as the best seed route, but do not
   use generic inner-window finite-difference relaxation.

4. Use the repaired targeted nested ladder plus mixed-prepolish as the eta_E=100 baseline:
       N128 -> N136 -> N140 -> N152,
       band = 100-320 rg.

5. The next numerical implementation target is not another generic local
   least-squares patch.  It should be either:
       a. a higher-order/trapezoid radial momentum residual audit for the
          R~300 rg row, or
       b. a block/Jacobian-aware correction that treats the outer/source radial
          row together with its neighboring energy and mass rows.

6. Only after eta_E=100 passes a physical differential audit
   should we lower eta_E toward 90, 80, 70, 60.
```

This keeps the Shen-calibrated local wind path honest: first solve a weak local
wind loading strictly, then lower the launch energy gradually.
