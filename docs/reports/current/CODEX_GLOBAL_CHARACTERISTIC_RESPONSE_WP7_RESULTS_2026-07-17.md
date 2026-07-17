# Global Characteristic-Response WP7 Results

**Date:** 2026-07-17
**Branch:** `codex/repository-cleanup-after-1e7438e`
**Starting commit:** `8d0691b`
**Scope:** profile and optimize the existing subsonic one-incoming-mode inner
characteristic response without changing its equations, fluxes, tolerances, or
mode count; retry exactly the first N64 source-off remnant step.

## Verdict

The characteristic pressure root was a material part of one finite-difference
Jacobian, and an exact bounded trace cache removes nearly all repeated local
work. The cache is rigorously keyed by the complete first-cell primitive trace,
is disabled by default, and stores only the characteristic flux correction and
its audit. Cached and uncached trajectories agree exactly over a controlled
20-evaluation comparison.

The optimization does not unlock the low-throughput remnant. With the cache
enabled, the bounded N64 coarse retry reaches a maximum normalized residual of
`9.831218e-7` after 600 nonlinear evaluations, above the unchanged `1e-8`
acceptance gate. No timestep is accepted. Source-on, N96, tide, and wind remain
blocked.

WP7 is therefore:

```text
local characteristic equivalence and efficiency: certified
low-throughput initializer unlock:                rejected
```

## Reference Profile

The frozen input is

```text
outputs/checkpoints/global_low_throughput_remnant/projected_N64.npz
```

The profile uses source-off physics, the exact first hold step
`dt/t_load=1e-8`, the production
`characteristic_inner_roche_outer` boundary, and the certified
`sparse_forward` Jacobian. The checkpoint arrays remain bitwise unchanged.

For one nonlinear evaluation and one Jacobian assembly:

| Metric | Uncached |
|---|---:|
| Residual calls | `259` |
| Characteristic calls | `260` |
| Pressure-root solves | `260` |
| Root function calls | `4680` |
| Vertical-state calls | `5460` |
| Jacobian wall time | `0.619541 s` |
| Characteristic wall time | `0.156294 s` |
| Characteristic / Jacobian time | `25.23%` |

This confirms the WP6c diagnosis: repeated finite-difference traces rebuilt the
same local pressure match many times.

## Exact Trace Cache

The optional cache key is the exact tuple

```text
(Sigma, v_R, Omega, T, specific_total_energy)
```

with no rounding, tolerance, nearest-neighbor reuse, or cross-step
linearization. It is a bounded LRU local to one backward-Euler solve. The
default cache size is zero, so existing production behavior is unchanged.

For the same one-evaluation profile with a cache size of 32:

| Metric | Uncached | Cached |
|---|---:|---:|
| Pressure-root solves | `260` | `5` |
| Cache hits / misses | `0 / 260` | `255 / 5` |
| Vertical-state calls | `5460` | `105` |
| Characteristic wall time | `0.156294 s` | `0.006563 s` |
| Jacobian wall time | `0.619541 s` | `0.463130 s` |

The local characteristic work is `23.8x` faster, pressure-root count falls by
`98.08%`, and the complete Jacobian is `1.34x` faster.

## Twenty-Evaluation Equivalence

The uncached and cached N64 solves were each stopped at 20 nonlinear
evaluations. They have identical:

```text
maximum normalized residual       0.025397158634797978
relative ledger defect            4.062933108518594e-10
nonlinear evaluations             20
Jacobian assemblies               10
residual calls                    2591
final update norm                 5.62349212387403e-5
termination message               maximum evaluations exceeded
```

The cached run reduces pressure-root solves from `2592` to `60` and total
measured wall time from `6.498 s` to `4.913 s`. Face mass, radial-momentum,
angular-momentum, and energy flux arrays agree exactly in the dedicated
regression test.

## Bounded Coarse Retry

The one authorized N64 source-off retry used:

```text
maximum nonlinear evaluations     600
cache size                         32
residual gate                      1e-8
ledger gate                        unchanged
controller and timestep            unchanged
```

It returned:

| Metric | Result |
|---|---:|
| Accepted timestep | no |
| Maximum normalized residual | `9.831218e-7` |
| Relative ledger defect | `1.280106e-11` |
| Nonlinear evaluations | `600` |
| Jacobian assemblies | `563` |
| Residual calls | `145292` |
| Pressure-root solves | `2852` |
| Cache hits / misses | `142441 / 2852` |
| Characteristic wall time | `3.605 s` |
| Jacobian wall time | `256.462 s` |
| Characteristic / Jacobian time | `1.406%` |
| Measured wall time | `264.800 s` |

The optimized characteristic map is no longer the controlling expense. The
remaining failure is the complete global nonlinear correction: the residual is
about `98.3` times the gate after the fixed evaluation ceiling.

## Decision

1. Retain the characteristic work telemetry.
2. Retain the exact cache as an opt-in diagnostic/solver aid, disabled by
   default.
3. Do not wire the cache into adaptive production holds because the coarse
   initializer gate failed.
4. Do not add a scalar implicit derivative for the same local map. It would
   preserve the local derivative but cannot address the now-dominant global
   Jacobian and Newton cost.
5. Do not launch source-on, N96, N128, tide, or wind.
6. Do not relax the residual, ledger, timestep, controller, or nonlinear
   evaluation gates.
7. Do not start another colored/global Jacobian optimization architecture.

The next work package must choose exactly one route:

```text
A. one bounded complete-global-residual nonlinear-convergence audit,
   with no physics change and a predeclared adoption gate;

or

B. close the remnant as an implicit initializer and construct a fresh
   low-mass global initial state for an explicit/IMEX startup that avoids
   the large monolithic first correction.
```

These routes must not be developed in parallel. Tide and wind remain blocked
until a no-tide global duration gate is practical and passes.

## Verification

```text
characteristic cache equivalence test:  passed
global evolution module tests:          55 passed
full repository suite:                  378 passed, 4 subtests
input checkpoint mutation:              none (bitwise check)
```

Machine-readable profiles:

```text
outputs/tables/global_characteristic_response_profile_reference_N64.json
outputs/tables/global_characteristic_response_profile_cached_N64.json
outputs/tables/global_characteristic_response_profile_reference_N64_nfev20.json
outputs/tables/global_characteristic_response_profile_cached_N64_nfev20.json
outputs/tables/global_characteristic_response_profile_cached_N64_nfev600.json
```
