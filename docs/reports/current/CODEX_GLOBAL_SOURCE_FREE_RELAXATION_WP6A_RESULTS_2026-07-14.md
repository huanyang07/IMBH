# Global Source-Free Relaxation WP6a Results

**Date:** 2026-07-14
**Branch:** `codex/repository-cleanup-after-1e7438e`
**Starting commit:** `35dbd2f`
**Scope:** bounded `N=64` source-free continuation from the WP5 immutable
corrected-rate state. Tide and wind remain disabled.

## Verdict

The source-free `N=64` trajectory does not pass the declared inner
mapping-relaxation gate. It reaches exact milestones at `2e-7`, `5e-7`, and
`1e-6 t_load` with excellent nonlinear and conservation behavior, but stops at
`1e-6` because the minimum velocity-gradient length falls below one cell:

```text
L_v / Delta R_cell = 0.900399
```

This is the predeclared named plunge-resolution stop. The attempted extension
to `2e-6` was terminated and no under-resolved state was accepted as a relaxed
reference.

The next step is one controlled `N=96` source-free refinement through the same
physical milestones. A longer `N=64` run and stream ramp remain blocked.

## Declared Relaxation Gate

Consecutive milestones had to satisfy all of:

```text
inner mass flux change / supply        <= 0.01
inner angular flux relative change     <= 0.02
inner energy flux relative change      <= 0.02
fixed-radius Mach change               <= 0.10
fixed-radius abs(Delta ln Sigma)        <= 0.02
fixed-radius abs(Delta ln T)            <= 0.02
maximum H/R relative change            <= 0.01
minimum L_v / Delta R_cell              >= 1.0
```

No threshold was relaxed after seeing the trajectory.

## Milestones

| `t/t_load` | Inner mass flux / supply | Max `H/R` | `L_v/Delta R` | `L_v/H` |
|---:|---:|---:|---:|---:|
| `2e-7` | `-0.182852` | `0.1411051` | `1.4054` | `2.4834` |
| `5e-7` | `-0.191512` | `0.1411056` | `1.2518` | `1.9890` |
| `1e-6` | `-0.195483` | `0.1411074` | `0.9004` | `1.3065` |

The maximum full-history storage-scaled ledger defect remains
`6.06e-16`. The Roche edge remains closed with zero outer `M/J/E` flux and a
normalized energy margin near `-0.08479`.

The last accepted controller moves rather than settling:

| `t/t_load` | Controller | Cell | Fraction of 2% limit | Cell Mach |
|---:|---|---:|---:|---:|
| `2e-7` | relative thickness | 0 | `0.391` | `-8.37` |
| `5e-7` | relative thickness | 2 | `0.163` | `-0.690` |
| `1e-6` | relative thickness | 0 | `0.670` | `-12.06` |

## Why The Relaxation Gate Fails

Between `2e-7` and `5e-7`:

```text
inner angular-flux change:      4.75%
inner energy-flux change:       4.34%
maximum fixed-radius Mach move: 1.479
maximum abs(Delta ln T):        0.0295
```

Between `5e-7` and `1e-6`:

```text
inner mass-flux change/supply:  0.00397
inner angular-flux change:      2.075%
inner energy-flux change:       1.871%
maximum fixed-radius Mach move: 2.337
maximum abs(Delta ln Sigma):    0.02013
maximum abs(Delta ln T):        0.04871
```

The integrated inner flux is beginning to move more slowly, but the local
plunge state is not stabilized and is no longer resolved on `N=64`. Maximum
outer thickness is almost unchanged; the stop is specifically an inner
gradient-resolution issue, not a global thermal runaway or Roche opening.

## Interpretation

WP5 established that this inner motion is present with or without the stream.
WP6a now shows that it cannot be safely removed by simply evolving the
`N=64` source-free state longer.

The result does not establish a physical instability. It says:

```text
the mapped stationary plunge is not a discrete equilibrium of the global
finite-volume operator, and its N64 relaxation becomes locally unresolved
before a quasi-relaxed reference is obtained.
```

The immutable `2e-7`, `5e-7`, and `1e-6` milestones are retained. The
interrupted candidate beyond `1e-6` is not a result.

## Reproduction

```bash
PYTHONPATH=src python3 scripts/run_global_source_free_relaxation.py \
  --output outputs/tables/global_source_free_relaxation.json \
  --maximum-nfev 600 \
  --maximum-accepted-steps 80
```

The runner reconstructs completed stages from immutable milestones and stops
before requesting the next target when the resolution gate fails.

## Locked Next Step

Run one source-free `N=96` refinement from the corrected accepted-rate mapping
at exact `2e-7`, `5e-7`, and `1e-6 t_load`.

Decision gates:

1. If `L_v/Delta R_cell >= 1` and the fixed-radius/flux comparisons converge,
   continue N96 only until the declared relaxation gate passes, then freeze the
   reference and begin one smooth stream ramp.
2. If N96 also crosses `L_v/Delta R_cell < 1` before relaxation, stop. Run one
   N128 snapshot at the failing physical time solely to distinguish ordinary
   spatial refinement from formation of a narrowing physical layer.
3. Do not continue N64, lower the gradient gate, alter the plunge branch, or
   begin tide/wind work.
