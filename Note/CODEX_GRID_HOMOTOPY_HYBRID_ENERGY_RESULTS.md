# Grid Homotopy and Hybrid Physical-Energy Residual Results

Date: 2026-07-03

## Purpose

Follow up on the targeted-remesh failure near the stream-source wall:

```text
last clean source fraction: f_s = 0.8980625
next attempted source fraction: f_s = 0.898125
physical gate: raw physical-domain differential energy residual <= 3e-5
```

Two fixes were tested:

1. fixed-physics grid homotopy at `f_s=0.8980625`;
2. a hybrid interval residual that directly includes the raw differential energy equation in the Newton objective.

## Code Changes

### New script

Added:

```text
scripts/run_standard_slim_stream_grid_homotopy.py
```

This script:

- loads a fixed-physics checkpoint;
- builds a residual-remesh target grid using the existing monitor;
- introduces a homotopy parameter `eta` between the current grid and target grid;
- remaps/polishes at fixed source fraction;
- accepts only if both solver residual and raw physical `interval_E` pass.

### New residual form

Added `interval_residual_form="integrated_physical_energy"` to `TransonicSlimParams`.

For each interval it uses:

```text
radial residual = integrated radial residual
energy residual = raw scaled differential energy residual
```

This makes the square Newton system minimize the same physical energy residual used by the physical gate, while keeping the better behaved integrated radial equation.

### Regression test

Added a finite-shape test for the new residual form.

Test result:

```text
147 passed
```

## Fixed-Physics Grid Homotopy

Anchor:

```text
outputs/checkpoints/high_mdot_stream_outer_buffer_phys_gate_target_remesh_0898_to0898125/
phys_gate_target_remesh_0898_to0898125_mass_0p8980625_torque_0p005_mdot_2_N896.npz
```

Anchor audit:

```text
full   = 5.738e-08
physE  = 2.934e-05
status = clean
```

Target grid monitor:

```text
target R ~ 259.2 rg
gentle target remesh
N = 896
```

### Eta scan

Output:

```text
outputs/tables/high_mdot_stream_grid_homotopy_08980625_to_gentle_target.md
```

| eta | solver full | physical_E | result |
|---:|---:|---:|---|
| 0.05 | 2.571e-06 | 2.317e-03 | rejected |
| 0.025 | 6.204e-06 | 3.361e-03 | rejected |
| 0.0125 | 1.288e-05 | 7.783e-03 | rejected |
| 0.00625 | 2.974e-06 | 1.557e-03 | rejected |

### Ultra-small eta scan

Output:

```text
outputs/tables/high_mdot_stream_grid_homotopy_08980625_ultratiny_eta.md
```

| eta | solver full | physical_E | result |
|---:|---:|---:|---|
| 0.0015625 | 2.886e-07 | 1.835e-04 | rejected |
| 0.00078125 | 1.732e-07 | 1.004e-04 | rejected |
| 0.000390625 | 1.152e-07 | 5.891e-05 | rejected |

### Grid-homotopy interpretation

Even extremely small grid motion breaks the raw physical energy audit while leaving the weighted solver residual excellent.

This means fixed-physics grid homotopy is not currently viable with the old weighted/integrated objective. The solver moves to a nearby weighted-residual minimum that is not physically clean under the differential energy audit.

## Hybrid Physical-Energy Residual

The first run accidentally inherited the checkpoint residual form. The meaningful runs force:

```text
IMBH_STANDARD_SLIM_STREAM_MASS_FORCE_INTERVAL_FORM=integrated_physical_energy
IMBH_STANDARD_SLIM_STREAM_MASS_FORCE_INTEGRATED_WEIGHTING=none
```

### Full source step

Target:

```text
f_s: 0.8980625 -> 0.898125
df_s = 6.25e-5
```

Output:

```text
outputs/tables/high_mdot_stream_outer_buffer_phys_gate_hybrid_forced_08980625_to0898125.md
```

Short Newton budget:

```text
final_full = physical_E = 3.112e-05
nfev = 13
status = rejected
```

Long Newton budget:

```text
outputs/tables/high_mdot_stream_outer_buffer_phys_gate_hybrid_long_08980625_to0898125.md
```

```text
final_full = physical_E = 3.086e-05
nfev = 107
status = rejected
```

The hybrid residual correctly aligns the solver residual with the physical gate, but the Newton solve stalls just above `3e-5`.

### Half source step

Target:

```text
f_s: 0.8980625 -> 0.89809375
df_s = 3.125e-5
```

Output:

```text
outputs/tables/high_mdot_stream_outer_buffer_phys_gate_hybrid_halfstep_08980625_to0898125.md
```

Result:

```text
final_full = physical_E = 3.062e-05
nfev = 72
status = rejected
```

Even the half step remains slightly above the strict physical gate.

## Main Findings

1. The old grid-remap path is not reliable.
   - Tiny grid changes make physical `interval_E` worse by factors of several to hundreds.

2. The new hybrid residual is conceptually correct.
   - It makes `final_full` equal to the raw physical energy residual.
   - It removes the misleading situation where weighted residuals look excellent while physical `interval_E` fails.

3. The current hybrid Newton solve is not yet numerically strong enough.
   - It stalls at `physical_E ~ 3.06e-5` to `3.09e-5`, just above the `3e-5` gate.
   - More iterations help only weakly and are expensive.

4. The clean scientific frontier remains:

```text
f_s = 0.8980625
```

## Recommended Next Step

Do not spend more runtime on ordinary integrated residual remaps or plain source-fraction step shrinking.

The next useful implementation is a better-conditioned hybrid Newton path:

1. add Newton audit output for the forced hybrid residual;
2. inspect accepted/rejected line-search steps, damping choices, and linear residuals near the plateau;
3. try equilibrated direct/LSMR variants and smaller Jacobian finite-difference steps;
4. add an energy-focused line-search merit or block scaling so the physical energy component cannot stall just above the gate;
5. then retry `f_s=0.89809375` and `0.898125`.

If this still stalls above `3e-5`, the physical gate may need to be treated as a mesh-dependent tolerance until a higher-order or integral-consistent physical-energy audit is implemented.

## Follow-Up: Hybrid Newton Audit and Quarter-Step Ladder

Additional runs tested the hybrid Newton plateau directly.

### Newton audit with `damping=1e-3`

Output:

```text
outputs/tables/high_mdot_stream_outer_buffer_phys_gate_hybrid_audit_08980625_to0898125.md
outputs/tables/high_mdot_stream_outer_buffer_phys_gate_hybrid_audit_08980625_to0898125_newton_audit/
```

Result:

```text
target f_s = 0.898125
final_full = physical_E = 3.191e-05
nfev = 42
accepted = false
```

Audit interpretation:

- every Newton step was accepted;
- all accepted steps used the first damping candidate, `1e-3`;
- line search often reduced the step by factors of 4--16;
- LSMR used roughly `5000--6000` iterations per Newton step;
- condition estimates were modest, but the LSMR residual norm plateaued near `2.85e-6`.

So the plateau is not a hard line-search failure. It is a slow, regularized hybrid Newton descent.

### Lower damping

Output:

```text
outputs/tables/high_mdot_stream_outer_buffer_phys_gate_hybrid_lowdamp_08980625_to0898125.md
```

Result:

```text
target f_s = 0.898125
final_full = physical_E = 6.053e-05
nfev = 101
accepted = false
```

Interpretation:

Lower damping made the solve worse. Undamped LSMR hit its iteration cap and line search collapsed to small alphas.

### Direct sparse solve

Output:

```text
outputs/tables/high_mdot_stream_outer_buffer_phys_gate_hybrid_direct_08980625_to0898125.md
```

Result:

```text
target f_s = 0.898125
final_full = physical_E = 7.663e-05
nfev = 34
accepted = false
```

Interpretation:

Direct Newton directions were too aggressive. The line search reduced to very small alphas and made no useful progress.

### Quarter source-fraction steps

Important acceptance bookkeeping point:

For forced hybrid solves, `final_full` is the raw physical energy residual. Therefore using the old generic solver tolerance `1e-5` is stricter than the physical gate and incorrectly rejects physically clean rows. The quarter-step ladder was rerun with:

```text
ACCEPTANCE_TOL = 3e-5
ANCHOR_TOL = 3e-5
PHYSICAL_E_TOL = 3e-5
```

Output:

```text
outputs/tables/high_mdot_stream_outer_buffer_phys_gate_hybrid_quartersteps_accept3e5.md
outputs/checkpoints/high_mdot_stream_outer_buffer_phys_gate_hybrid_quartersteps_accept3e5/
```

Results:

| f_s | final_full = physical_E | nfev | result |
|---:|---:|---:|---|
| 0.898078125 | 2.989e-05 | 74 | accepted |
| 0.89809375 | 3.025e-05 | 71 | rejected |

The new clean frontier is therefore:

```text
f_s = 0.898078125
```

This is a genuine but tiny advance beyond the previous `0.8980625` anchor.

## Updated Interpretation

The hybrid residual is the right physical objective, and with acceptance aligned to the physical gate it can advance the branch slightly.

The next bottleneck is numerical efficiency and plateau behavior:

- the accepted quarter step costs `~74` function evaluations;
- the next quarter step misses by only `2.5e-7` in raw physical `interval_E`;
- lower damping and direct sparse Newton both make things worse.

Best next move:

1. start from the new clean `f_s=0.898078125` hybrid checkpoint;
2. retry toward `0.89809375` with either a slightly smaller source step or an energy-focused line search;
3. add a runner-level convention that forced hybrid solves use `ACCEPTANCE_TOL = PHYSICAL_E_TOL`, so clean physical rows are not falsely rejected by the old integrated-residual tolerance.

## Follow-Up: Acceptance Convention and Eighth-Step Test

Implemented the runner-level convention:

```text
scripts/run_standard_slim_stream_mass_annulus_scan.py
```

When the physical gate is active and the interval form is `integrated_physical_energy`, the runner now uses:

```text
effective_acceptance_tol = max(ACCEPTANCE_TOL, PHYSICAL_E_TOL)
effective_anchor_tol     = max(ANCHOR_TOL, PHYSICAL_E_TOL)
```

Rows record these effective tolerances. This keeps hybrid solves from being falsely rejected by the old `1e-5` integrated-residual tolerance when the physical gate itself is `3e-5`.

Regression tests after the change:

```text
147 passed
```

### Re-run quarter-step ladder with automatic hybrid acceptance

Output:

```text
outputs/tables/high_mdot_stream_outer_buffer_phys_gate_hybrid_quartersteps_accept3e5.md
outputs/checkpoints/high_mdot_stream_outer_buffer_phys_gate_hybrid_quartersteps_accept3e5/
```

Results:

| f_s | physical_E | nfev | result |
|---:|---:|---:|---|
| 0.898078125 | 2.989e-05 | 74 | accepted |
| 0.89809375 | 3.025e-05 | 71 | rejected |

The accepted checkpoint is:

```text
outputs/checkpoints/high_mdot_stream_outer_buffer_phys_gate_hybrid_quartersteps_accept3e5/
phys_gate_hybrid_quartersteps_accept3e5_mass_0p898078125_torque_0p005_mdot_2_N896.npz
```

### Eighth-step retry from the new clean checkpoint

Output:

```text
outputs/tables/high_mdot_stream_outer_buffer_phys_gate_hybrid_eighthsteps_0898078125_to089809375.md
```

Target:

```text
f_s = 0.898078125 -> 0.8980859375
```

Result:

```text
physical_E = 3.116e-05
nfev = 69
accepted = false
```

The smaller step from the new anchor did not help. The tangent seed from the new anchor was worse (`initial ~4.38e-5`), and the hybrid Newton polish settled above the gate.

## Current Best State

The current clean frontier is:

```text
f_s = 0.898078125
```

This is a real improvement over the earlier `0.8980625` frontier, but the branch remains pinned by the hybrid Newton plateau just above the physical gate.

## Updated Next Recommendation

The next bottleneck is not source step size alone. It is the hybrid Newton correction near the physical-energy wall.

Most useful next work:

1. add an energy-focused line-search merit or block scaling for hybrid solves;
2. compare whether the candidate step reduces raw physical `interval_E`, not only global square merit;
3. only then retry from `f_s=0.898078125` toward `0.8980859375` and `0.89809375`.

The failed low-damping and direct-linear-solve tests suggest that simply changing linear solver/damping is not enough.
