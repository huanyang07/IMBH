# Causal N64 Confirmation WP10c5s-t Results

Date: 2026-07-18

## Verdict

The independently generated N64 causal trajectory passes its fixed-datum
short-startup gate. At the bounded common time, the N32/N64 thickness-response
error is

```text
maximum abs Delta ln(H/R) difference   6.66771844e-3
RMS Delta ln(H/R) difference           2.93804203e-3
```

The maximum remains above the unchanged `5e-3` mesh-certification gate, but it
contracts from the N16/N32 error with observed order `1.65737`. The RMS error
contracts with order `1.46336`. Both exceed the predeclared first-order
acceptance floor of `0.75`.

The first N64 duration attempt reached the target and passed every step, state,
rank, restart, and mass gate, but its accepted nonlinear residuals accumulated
to a five-field ledger defect of `1.80820e-10`, narrowly above `1e-10`. One
bounded replay tightened only the nonlinear residual tolerance from `1e-10`
to `1e-11`. It reduced the maximum step residual to `4.89864e-13` and the
aggregate five-field defect to `1.80840e-12`, without relaxing any gate or
changing the physical response.

The locked result is:

```text
N64 fixed analytic datum                         PASSED
N32/N64 exact-time short gate                    PASSED
baseline N64 duration physical/rank gates        PASSED
baseline N64 aggregate five-field ledger         FAILED
ledger-tight N64 duration                        PASSED
N32/N64 bounded duration 5e-3 mesh gate          NOT YET PASSED
minimum contraction order 0.75                   PASSED
one bounded N128 confirmation                    AUTHORIZED
N96, N256, long evolution, tide, wind,
stability, hot-state, and cycle searches         NOT AUTHORIZED
```

This is a bounded numerical convergence result at about `1e-9` of the loading
time. It is not evidence for a relaxed disk, instability, hot state, or limit
cycle.

## N64 Initial Datum

N64 was regenerated directly from the same fixed physical C2 profile used by
N16 and N32. No evolved N32 state was remapped.

```text
descriptor rank                         320 / 320
complete consistency rank               965 / 965
equilibrated condition estimate         6.15387e6
maximum consistency defect              4.44089e-15
initial physical timestep               6.56871e-6 s
minimum scattering optical depth        18.5046
initial incoming inner modes             0
outer Roche channel                     closed, nonchoked
```

The N32/N64 initial-profile differences are small:

| Quantity | Maximum difference |
|---|---:|
| `ln Sigma` | `1.01118e-3` |
| `v_R/c` | `6.01673e-5` |
| `v_phi/c` | `8.08003e-5` |
| `ln T` | `4.67714e-4` |
| `ln(H/R)` | `3.65561e-4` |

All original common-data gates pass.

## Short Gate

N64 reaches the exact N32 short time

```text
t_short = 1.0854883574529712e-4 s
```

in seven accepted steps with zero retries. The N32/N64 response differences
are:

| Quantity | Difference | Gate |
|---|---:|---:|
| Maximum `Delta ln(H/R)` | `8.64931e-4` | `5e-3` |
| RMS `Delta ln(H/R)` | `3.76210e-4` | diagnostic |
| Inner mass flux / supply | `1.13822e-5` | `5e-2` |
| Mass response / injected mass | `1.34875e-5` | `5e-2` |
| Outer mass flux / supply | `0` | `5e-2` |

The short common-data gate therefore passes before any duration extension is
attempted.

## Baseline Ledger Stop

N32 and N64 were advanced to

```text
t_target = 8.48423267286563e-4 s
```

with the same maximum timestep `1.1812376038124097e-5 s`. Each extension uses
63 accepted steps and zero rejected attempts.

The baseline N64 trajectory reaches the exact target with:

```text
all step gates                           passed
final state gate                         passed
descriptor rank                          320 / 320
complete consistency rank                965 / 965
mass ledger defect                       2.48668e-11
five-field ledger defect                 1.80820e-10
maximum accepted scaled residual         4.85457e-11
```

The only failed acceptance condition is the aggregate five-field ledger. The
balance identity agrees with the sum of the individual step closure defects;
this is accumulated accepted nonlinear closure, not a missing conservation
term or summation mismatch.

One strict single-step replay reduces the scaled residual to `4.89864e-13`
and its physical ledger defect to `3.27888e-12`. This bounded diagnostic
justifies one full ledger-tight replay. It does not justify relaxing the
aggregate gate.

## Ledger-Tight Replay

The replay changes only:

```text
step residual tolerance   1e-10 -> 1e-11
```

It does not change the equations, source, boundary, flux, timestep cap,
physical state gates, mass gate, or five-field ledger gate.

The strict N64 result gives:

```text
accepted extension steps                 63
rejected attempts                         0
first-step replay                         bitwise
maximum step residual                    4.89864e-13
maximum algebraic residual               below 1e-11
aggregate mass defect                    2.44727e-13
aggregate five-field defect              1.80840e-12
final H/R maximum                        0.0999615
minimum scattering optical depth         18.5195
incoming inner modes                      0
outer incoming modes                      2
outer Roche channel                      closed, nonchoked
descriptor / consistency rank            320/320, 965/965
```

The strict and baseline N64 physical responses agree:

```text
maximum Delta ln(H/R) response change    5.01175e-10
RMS response change                      1.10798e-10
inner flux / supply change               4.29878e-13
maximum H/R change                       3.18870e-13
accepted-step change                      0
```

The replay therefore repairs numerical closure without changing the measured
spatial response.

## Spatial Contraction

At the exact common target and shared timestep cap:

| Pair | Maximum response error | RMS response error |
|---|---:|---:|
| N16/N32 | `2.10327575e-2` | `8.10164862e-3` |
| N32/N64 | `6.66771844e-3` | `2.93804203e-3` |

This gives:

```text
maximum-error ratio                       3.15442
maximum-error observed order              1.65737
RMS-error ratio                           2.75750
RMS-error observed order                  1.46336
minimum order required for N128           0.75
```

The N32/N64 maximum remains `1.6677e-3` above the `5e-3` gate, so N64 does
not certify the duration mesh comparison. The strong monotone contraction,
together with every causal, rank, optical, Roche, nonlinear, restart, mass,
and five-field gate passing, activates the predeclared N128-only branch.

## Locked Next Work: WP10c5u

Run exactly one bounded N128 confirmation:

1. Regenerate N128 from the same fixed analytic profile. Do not remap N64.
2. Use the certified colored sparse backend and the `1e-11` residual contract
   from the first duration step so the aggregate ledger remains resolvable.
3. Require the unchanged N64/N128 short response gate of `5e-3`.
4. Only if the short gate passes, advance N128 to the same
   `8.48423267286563e-4 s` target with the same
   `1.1812376038124097e-5 s` maximum timestep.
5. Compare against the accepted strict N64 endpoint and report N32/N64 versus
   N64/N128 contraction.
6. Certify the bounded mesh gate only if the N64/N128 maximum response error
   is at most `5e-3` and every existing gate passes.
7. Otherwise stop and reassess the first-order production method.

Do not run N96, N256, a longer duration, distributed tide, wind, stability,
hot-state, or cycle searches in WP10c5u.

## Verification

```text
focused causal DAE/evolution tests    30 passed
full repository suite                 489 passed, 4 subtests passed
repository hygiene                    passed for 628 tracked files
Python compilation                    passed
git diff --check                      passed
```

## Reproduction

```text
PYTHONPATH=src python3 \
  scripts/run_causal_five_field_consistent_initial_step_wp10c5d.py \
  --increment-primary-mesh-common-n64-confirmation-audit

PYTHONPATH=src python3 \
  scripts/run_causal_five_field_consistent_initial_step_wp10c5d.py \
  --increment-primary-mesh-common-n64-ledger-replay-audit
```

Machine-readable outputs:

```text
outputs/tables/causal_five_field_mesh_common_n64_confirmation_wp10c5s.json
outputs/tables/causal_five_field_mesh_common_n64_ledger_replay_wp10c5t.json
```
