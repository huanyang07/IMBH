# Global Source-On/Source-Off WP5 Results

**Date:** 2026-07-14
**Branch:** `codex/repository-cleanup-after-1e7438e`
**Starting commit:** `35dbd2f`
**Scope:** instantaneous source tendency and matched corrected-seed `N=64`
source-on/source-off trajectories. Tide and wind remain disabled.

## Verdict

WP5 separates two simultaneous early-time effects:

1. the inner plunge and controller response are dominated by relaxation of the
   mapped stationary state under the global finite-volume operator;
2. the stream already adds its exact conservative mass, angular momentum, and
   energy moments in the distant source annulus, but that perturbation has not
   reached the inner flow by `1e-7 t_load`.

The next physical initial condition should therefore be built by relaxing a
source-free reference and then ramping the stream. The source-on trajectory
from the unrelaxed mapping remains a valid numerical boundary test, but its
early inner motion must not be interpreted as stream-driven loading.

## Shared Initial State

Both trajectories start from the same immutable, checksummed corrected-rate
`144/96 -> N64` mapping with a supersonic inner face at `4.5 rg`. They use the
same:

```text
global equations
Roche boundary
serial sparse-forward Jacobian
residual and ledger gates
2% physical-change controller
physical final time
```

The only difference is:

```text
source on:   exact compact stream S_M, S_J, S_E
source off:  exact zero cell source
```

## Instantaneous Tendency

At the identical initial state,

```text
R_source = R_on - R_off
```

reproduces the prescribed stream cell moments to relative error below
`1.21e-16`. The instantaneous inner face flux response is exactly zero in all
four conserved components, as expected for a finite-volume source deposited
near `240 rg`.

Over the matched horizon of `0.1519729662443652 s`, the linearized controller
metrics are:

| Operator | Controlling variable/cell | Metric |
|---|---|---:|
| Source on | relative thickness, cell 0 | `8.968e-2` |
| Source off | relative thickness, cell 0 | `8.968e-2` |
| Source only | log surface density, cell 59 | `8.489e-7` |

The direct stream contribution to the controller metric is about five orders
of magnitude below the source-free inner relaxation at this time.

## Matched Trajectories

Both runs land at exactly

```text
t = 0.1519729662443652 s = 1e-7 t_load
```

and independently choose the same timestep sequence:

```text
15 accepted steps
4 recovered rejected attempts
first accepted dt = 1.25e-8 t_load
remaining accepted dt = 6.25e-9 t_load
```

Every controller attempt in both trajectories selects relative thickness in
cell zero, with the same supersonic Mach number and the same fraction of the
unchanged two-percent limit.

| Quantity | Source on | Source off |
|---|---:|---:|
| Disk mass relative change | `+8.36082e-8` | `-1.63918e-8` |
| Inner mass flux / reference supply | `-0.17197938` | `-0.17197938` |
| Inner angular flux | `-2.3105371e42` | `-2.3105371e42` |
| Inner total-energy flux | `7.2647389e41` | `7.2647389e41` |
| Maximum `H/R` | `0.141104993` | `0.141104993` |
| Maximum ledger defect | `5.27e-16` | `4.81e-16` |

The source-on minus source-off disk mass is

```text
1.0000000049 times the injected stream mass.
```

The final inner mass-flux difference is below `9e-16` relative. The angular
and energy flux differences are likewise at floating-point scale. The sonic
radius differs by only `2.77e-5 cm`, maximum `H/R` is identical at reported
precision, and total internal energy differs by `1.06e-9` relative.

## Interpretation

The outcome is spatially mixed, not contradictory:

- In the source annulus, the stream produces an exact conservative storage
  increment. Its mass, angular-momentum, and energy effects are real.
- In the inner plunge, the source-on and source-off states are indistinguishable.
  The early controller and inner-flux evolution are operator/mapping
  relaxation.
- Global component `L2` norms mix these separated regions and therefore show
  both contributions as comparable. They must not be used alone to claim that
  the source drives the inner transient.

The source-off trajectory is a counterfactual control, not yet a physical
equilibrium. The appropriate next step is to construct a separately relaxed
source-free reference, then activate the stream smoothly and restart the
physical loading clock.

## Reproduction

```bash
PYTHONPATH=src python3 scripts/run_global_source_on_off_control.py \
  --output outputs/tables/global_source_on_off_control.json \
  --target-loading-fraction 1e-7 \
  --maximum-nfev 600 \
  --maximum-accepted-steps 32
```

The shared initial state and every accepted source-on/source-off milestone are
stored through the immutable WP1 checkpoint contract.

## Revised Next Gate

Before the physical `5e-6 t_load` extension:

1. continue the source-free `N=64` state until the inner controller/transient
   reaches a declared relaxation gate or a named stop event;
2. freeze that relaxed state and all conservative ledgers;
3. ramp the physical stream through one declared smooth schedule;
4. define physical loading time zero at the start of that ramp;
5. then extend the no-tide source-on evolution beyond the former failure time.

This is a sequencing correction, not a new physical closure. Tide and wind
remain blocked.
