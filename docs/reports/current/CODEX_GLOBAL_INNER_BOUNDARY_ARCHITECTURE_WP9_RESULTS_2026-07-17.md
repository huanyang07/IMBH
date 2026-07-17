# Global Inner-Boundary Architecture WP9 Results

**Date:** 2026-07-17
**Branch:** `codex/repository-cleanup-after-1e7438e`
**Scope:** exact degree-of-freedom comparison and one bounded low-rate causal
excision audit. No evolution, tide, wind, or boundary tuning was run.

## Verdict

Neither of the two authorized inner-boundary candidates passes its existing
production gate.

The one-domain architecture remains the preferred structural direction, but
the accepted low-rate slim solution cannot provide its required causal
excision. The older quasi-steady transonic-response hybrid is already closed
by its refined repeated-step failure. Selecting either one now would override
a declared stop condition.

## Exact Architecture Counts

### A. One-domain causal excision

For N cells:

```text
differential unknowns     4 N
backward-Euler rows       4 N
inner boundary rows       0
required incoming modes   0
```

The cell state is mass, radial momentum, angular momentum, and total energy.
The zero-row inner contract is valid only when every radial characteristic
leaves the domain through the inner face.

### B. Quasi-steady transonic response

For `Ni` inner nodes and `No` outer cells:

```text
unknowns and rows          2 Ni + 5 No + 5
outer storage rank         3 No
```

At the tested `24/16` mesh this is a square `133 x 133` system. It includes
the inner transonic core, outer mass/angular/energy storage, common stress,
radial force, two primitive interface rows, two flux-extraction rows, and one
open-edge row.

## Low-Rate Causal Audit

The solver-generated low-throughput state has

```text
Mdot_inner / Mdot_Edd                 0.025
stationary critical radius            5.996987 rg
u / (H Omega_K) at critical point     0.07419
Euler acoustic Mach at critical point -0.05786
incoming characteristics              1
```

The stationary DAE critical point is therefore not an acoustic sonic point of
the global four-equation time-dependent system for this state.

The same accepted local equations were continued inward without a ballistic,
free-fall, or fitted replacement:

| Edge (`rg`) | `v_R/c` | `c_eff/c` | Acoustic Mach | Incoming modes | Gate |
|---:|---:|---:|---:|---:|---:|
| 4.5 | `-8.54e-6` | `4.95e-3` | `-1.73e-3` | 1 | fail |
| 3.0 | `-3.48e-4` | `1.22e-1` | `-2.84e-3` | 1 | fail |
| 2.1 | `-1.25e-2` | `2.95` | `-4.22e-3` | 1 | fail |
| 2.01 | `-7.09e-2` | `1.49e1` | `-4.74e-3` | 1 | fail |
| 2.001 | `-2.63e-1` | `5.19e1` | `-5.06e-3` | 1 | fail |
| 2.0001 | `-8.62e-1` | `1.65e2` | `-5.22e-3` | 1 | fail |

The pseudo-horizon limit is not a remedy. Before a zero-incoming acoustic
region appears, the Newtonian gas-radiation closure becomes acausal. No edge
in the audited interval passes the simultaneous zero-incoming, subluminal
velocity, and subluminal sound-speed gate.

## Hybrid Stop Evidence

The quasi-steady inner-response DAE already has the following bounded result:

```text
24/16 accepted subcycled steps          2
third-step maximum residual             1.0466e-7
fixed residual gate                     1.0e-7
maximum interface continuity row        1.0466e-7
maximum flux-extraction row              9.1184e-8
```

The cross-interface radial stencil no longer controls that failure. Direct
primitive elimination instead failed in the inner transonic core. This is why
ADR 0012 closed further splice conditioning.

## Scientific Interpretation

The project currently uses two different meanings of “sonic”:

1. singularity of the stationary slim-disk differential matrix; and
2. acoustic characteristics of the global conservative evolution equations.

They agree sufficiently for the high-throughput mapped plunge, which remains
causally outgoing at `4.5 rg`. They do not agree for the low-throughput branch.
This is the controlling inner-physics inconsistency for fresh loading.

The result does not reject time-dependent accumulation or a future hot phase.
It says that those trajectories cannot yet be evolved from low throughput
with a mesh-independent black-hole boundary under the current mixed inner
model.

## Decision And Next Plan

1. Preserve both existing implementations and their evidence.
2. Select neither as the fresh-loading production boundary.
3. Do not tune the low-mass drift, move the edge toward `2 rg`, cap the sound
   speed, force inward velocity, or reopen hybrid interface conditioning.
4. Freeze tide and wind.
5. Write the next work package around one causal inner physical system: its
   stationary critical point, time-dependent characteristics, EOS, and
   excision must be mutually consistent.
6. Begin with an equations/rank prototype, not a trajectory. The minimum gate
   is a low-throughput stationary inflow with subluminal characteristics,
   zero incoming modes at the excision, and mesh-invariant N64/N96 mapping.

Machine-readable evidence:

```text
outputs/tables/global_inner_boundary_architecture_gate.json
```

Reproduction:

```bash
PYTHONPATH=src python3 scripts/run_global_inner_boundary_architecture_gate.py
```
