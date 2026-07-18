# Causal N128 Mesh Certification WP10c5u Results

Date: 2026-07-18

## Verdict

The independently generated N128 causal trajectory passes the final bounded
spatial-convergence gate authorized by WP10c5s-t.

At the exact common duration,

```text
maximum abs Delta ln(H/R) difference   2.58966956e-3
RMS Delta ln(H/R) difference           1.25635590e-3
```

Both are below the unchanged `5e-3` maximum-response gate. Relative to the
N32/N64 errors, the N64/N128 errors contract with observed orders

```text
maximum response order                 1.36442516
RMS response order                     1.22560983
```

The N128 duration reaches the exact target in 63 extension steps with no
retries. The endpoint descriptor and complete consistency systems remain full
rank, the strict nonlinear residual contract passes, and the physical
five-field and mass ledgers close below `2.5e-12`.

The locked result is:

```text
independent N128 fixed datum                     PASSED
N64/N128 exact-time short gate                   PASSED
strict N128 bounded duration                     PASSED
N64/N128 bounded duration 5e-3 mesh gate         PASSED
bounded first-order spatial convergence          CERTIFIED
further N96/N256 confirmation                    NOT AUTHORIZED
long evolution, tide, wind, stability,
hot-state, and cycle searches                    NOT AUTHORIZED
```

This certifies only the current bounded no-tide response at about `1e-9` of
the loading time. It is not evidence for relaxation, instability, a hot
state, or a limit cycle.

## Fixed Numerical Contract

No physical equation, operator, boundary condition, or physical gate changed
from the accepted N64 replay.

```text
target elapsed time                    8.484232672865630e-4 s
maximum timestep                       1.181237603812410e-5 s
short residual tolerance               1e-10
duration residual tolerance            1e-11
algebraic residual tolerance            1e-11
mass/five-field ledger tolerance        1e-10
```

The short-startup tolerance remains `1e-10`, exactly as at N64. The stricter
`1e-11` solve contract begins with the duration continuation. An initial
diagnostic invocation incorrectly applied `1e-11` to the short startup. It
reached the target and passed every physical, restart, and mass gate, but the
initial polished increment had residual `1.68644e-11`. The final reproducible
run restores the unchanged short contract; no tolerance was relaxed relative
to the certified short workflow.

## Independent N128 Datum

N128 was regenerated from the same analytic physical C2 profile used at
N16-N64. No N64 state was interpolated or prolonged.

```text
descriptor rank                         640 / 640
complete consistency rank              1925 / 1925
equilibrated condition estimate         6.37267e6
maximum consistency defect              7.77156e-15
inner mass flux / stream supply        -1.0000000000000002
maximum H/R                             0.1000000000000002
minimum scattering optical depth       18.5044962
incoming inner characteristics          0
outer Roche channel                     closed, nonchoked
```

The N64/N128 initial-profile differences are:

| Quantity | Maximum difference |
|---|---:|
| `ln Sigma` | `1.38541e-4` |
| `v_R/c` | `8.24346e-6` |
| `v_phi/c` | `1.10704e-5` |
| `ln T` | `1.40947e-4` |
| `ln(H/R)` | `5.00851e-5` |

All fixed-datum gates pass.

## Short Gate

N128 reaches the exact N64 short time

```text
elapsed time                            1.085488357452971e-4 s
accepted steps                          7
rejected attempts                       0
maximum step residual                   5.63686e-11
mass defect                             1.13209e-12
minimum scattering optical depth       18.5046008
```

The short N64/N128 differences are:

| Quantity | Difference | Gate |
|---|---:|---:|
| Mass response / injected mass | `2.84060e-6` | `5e-2` |
| Inner mass flux / supply | `5.71276e-6` | `5e-2` |
| Outer mass flux / supply | `0` | `5e-2` |
| Maximum `Delta ln(H/R)` response | `3.33334e-4` | `5e-3` |
| RMS `Delta ln(H/R)` response | `1.60648e-4` | diagnostic |

The short gate therefore authorizes only the predeclared duration target.

## Bounded N128 Duration

The strict continuation result is:

```text
target elapsed time                     8.484232672865630e-4 s
reached elapsed time                    8.484232672865630e-4 s
extension accepted steps                63
extension rejected attempts             0
total accepted steps                    70
total rejected attempts                 0
maximum extension residual              3.19009e-13
maximum H/R                             0.0999596725
minimum scattering optical depth       18.5192706
elapsed loading-time fraction           9.98142e-10
inner incoming characteristics          0
outer incoming characteristics          2
outer Roche channel                     closed, nonchoked
```

The endpoint rank and conservation gates are:

```text
descriptor rank                         640 / 640
descriptor smallest singular value      5.04960e-3
complete consistency rank              1925 / 1925
equilibrated consistency condition      6.37682e6
equilibrated smallest singular value    7.63307e-7
five-field relative ledger defect       1.77056e-12
mass relative ledger defect             2.48535e-13
first continuation replay               bitwise
final restart roundtrip                 bitwise
```

The full unscaled consistency condition estimate is large
(`1.52737e10`), but equilibration leaves a full-rank system with a
`6.37682e6` condition estimate under the predeclared numerical threshold.

## Spatial Convergence

| Mesh pair | Max response error | RMS response error |
|---|---:|---:|
| N16/N32 | `2.10327575e-2` | `8.10164862e-3` |
| N32/N64 | `6.66771844e-3` | `2.93804203e-3` |
| N64/N128 | `2.58966956e-3` | `1.25635590e-3` |

The final contraction is:

```text
maximum error ratio                     2.57473715
maximum observed order                  1.36442516
RMS error ratio                         2.33854279
RMS observed order                      1.22560983
```

The N64/N128 maximum error is below `5e-3`, all other mesh metrics pass,
and the observed order remains consistent with the first-order face
transport classification from WP10c5r. No operator correction or further
fine-mesh confirmation is justified.

## Scientific Interpretation

WP10c5u closes the bounded spatial-certification question. It establishes
that the causal five-field discretization gives a convergent, conservative,
full-rank early-time response from one fixed source-compatible datum.

It does not establish that the current microsecond-scale controller is a
physical timestep requirement. The final run required roughly two hours of
wall time to evolve only `8.48e-4 s`, or about `1e-9 t_load`. Continuing this
controller geometrically in physical duration is therefore not a credible
route to loading, heating, or cycle timescales.

## Locked Next Work

The next work package is an observable-aware timescale and timestep-ceiling
audit, with no new physics:

1. Define versioned observables for local cooling outside a declared inner
   radius, inner mass flux, `H/R`, and conserved storage.
2. Measure local characteristic, causal-stress, thermal, cooling-response,
   advection, and loading clocks from an accepted checkpoint.
3. At N16, compare one backward-Euler step of `dt` with two steps of
   `dt/2` on a bounded geometric ladder.
4. Preserve all nonlinear, causal, optical, Roche, and ledger gates and
   classify every rejected rung.
5. Run N32 only after N16 identifies a finite observable-accuracy ceiling.
6. Do not run N64/N128 production ladders, longer loading evolution,
   distributed tide, wind, stability, hot-state, or cycle searches in that
   audit.

The user-provided long-timescale note has the right direction, but its
baseline predates WP10c5r-u. Its proposed generalized eigenanalysis, BDF2,
Jacobian reuse, stress reduction, and multirate methods remain later
work packages. The immediate task is the smallest measurement that decides
whether the current microstep ceiling is numerical or observable-physical.

## Verification

```text
focused causal DAE/evolution tests      30 passed
full repository suite                   489 passed, 4 subtests passed
repository hygiene                      passed
Python compilation                      passed
git diff --check                        passed
```

## Reproduction

```text
PYTHONPATH=src python3 \
  scripts/run_causal_five_field_consistent_initial_step_wp10c5d.py \
  --increment-primary-mesh-common-n128-confirmation-audit
```

Machine-readable output:

```text
outputs/tables/causal_five_field_mesh_common_n128_confirmation_wp10c5u.json
```

Restart checkpoints:

```text
outputs/checkpoints/causal_five_field_wp10c5k/causal_wp10c5u_N128_final.npz
outputs/checkpoints/causal_five_field_wp10c5k/causal_wp10c5u_duration_N128_final.npz
```
