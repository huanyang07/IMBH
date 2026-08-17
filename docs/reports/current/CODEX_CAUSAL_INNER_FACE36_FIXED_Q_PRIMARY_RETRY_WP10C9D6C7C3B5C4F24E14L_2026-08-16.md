# Fixed-Q Iteration-Reserve Primary Retry WP10c9d6c7c3b5c4f24e14l

## Classification

`bounded_continuation_failed`

The frozen end-to-end primary retry completed all four accepted main BDF2
roots, advanced the accepted fixed-Q trajectory by `4e-7 s`, reproduced the
final two-root suffix bitwise after restart, closed the cumulative ledgers,
and passed the matched two-half-step audit. The iteration-reserve warm policy
also passed its cost comparison.

The package nevertheless fails its binding scientific classification because
the same-history cold shadow and warm `warm_2` endpoint differ in physical
reaction action by `2.8666087608919947e-8`, above the prospectively frozen
`1e-8` tolerance. Their scaled primitive-state difference is only
`7.859135564558528e-11` and passes its separate `1e-8` gate. The tolerance is
not relaxed and the result is not reclassified as a cost-only failure.

This rejection does not select a physical-admissibility, history, restart,
ledger, or matched-timestep failure. It selects insufficient invariant
endpoint agreement between two accepted nonlinear roots produced from the
same state, BDF history, target, scales, timestep, and predictor.

## Frozen execution identity

```text
execution commit          b0a60a14d8b90b5fef25898190ab2d7dcb2d0c97
execution tree            13e3c55a6f9b5a35c30c2df1ced7d2e1b6ad2c9d
manifest contract SHA256  5f95ab1490ddd008a121ce8aae7967499a28cb4817dddc6c884b4da78f055b75
runner SHA256             a22793a18ba8193e67d371b2fae9b8ed4130c4e0457d75f71a5324c09b2798f0
continuation seed SHA256  929f844ecd1dba520bcdffdeab4e8876c5842d536032ca8cb2d77bfe609cd653
```

BLAS and OpenMP thread counts were pinned to one. The tracked tree was clean
at execution start. The canonical package closes its SHA256 checksums, its
frozen-contract validation passes, and the focused result tests pass `5/5`.

## Accepted main trajectory

Every main root passes the unchanged `1e-10` complete-residual gate and all
physical, storage, reaction, conditioning, history, and checkpoint gates.

| Root | Residual | Evaluations | Exact assemblies | Refresh reason | Wall time |
|---|---:|---:|---:|---|---:|
| `cold_1` | `4.737492683089679e-13` | 19 | 2 | initial, line-search failure | `2741.77 s` |
| `warm_1` | `5.533443390874517e-13` | 8 | 1 | iteration reserve | `1289.61 s` |
| `warm_2` | `5.048217216618925e-13` | 8 | 1 | iteration reserve | `1290.00 s` |
| `warm_3` | `8.748886309915929e-13` | 8 | 1 | iteration reserve | `1301.97 s` |

The three warm roots each follow the carried-matrix path for six iterations,
then use the prospectively frozen iteration-reserve refresh and converge with
one exact correction. No warm root exceeds one exact assembly. No rejected
candidate enters continuation history.

Across the four accepted main endpoints:

```text
largest Q3 relative defect                 2.860487376181398e-16
largest storage-parity defect              3.210489712912401e-14
largest reaction-ledger defect             3.6463428791461168e-22
largest constraint-action defect           2.664591403085430e-16
largest raw Schur condition number          3.469908959920416e4
minimum / maximum reconstruction factor     1.0 / 1.0
largest H/R                                 0.09783748666878898
minimum scattering optical depth            19.254315793914518
largest scaled primitive change             0.004409792597730185
incoming excision characteristics           0
```

The accepted trajectory cumulative ledger defect is
`3.9968871035996406e-16`, below the frozen `4e-12` budget. The main roots use
`6623.34 s` of wall time, corresponding to
`2.1741296479821607e-7` accepted physical seconds per wall hour.

## Restart and matched-step evidence

The checkpoint after `warm_1` was reloaded and the `warm_2 -> warm_3` suffix
was replayed bitwise. Both result objects and continuation states reproduce,
including event traces and the iteration-reserve refresh locations.

The nonpropagating two-half-step control also passes:

```text
half_1 residual                              5.585887182532928e-13
half_2 residual                              6.522280716720812e-13
state difference / full-step change          3.363160546067348e-6
reaction-action relative difference          1.7656403948099872e-5
```

Both half roots pass all inherited physical gates and checkpoint roundtrips.
Their line-failure refreshes also demonstrate that the residual-heavy
backtracking path remains a major cost center, but this does not invalidate
the matched-endpoint audit.

## Binding cold-shadow rejection

The same-history cold shadow starts from the checkpoint before `warm_2` and
uses the identical physical state, BDF history, timestep, target, scales, and
predictor. It differs only in nonlinear initialization: the shadow begins
with an exact complete matrix, while `warm_2` begins with the carried matrix
and refreshes at the iteration reserve.

The cold shadow passes every per-root acceptance gate:

```text
maximum scaled residual                       6.398284679853816e-11
Q3 relative defect                            2.7943081817376983e-16
storage-parity defect                         2.273418159675800e-14
reaction-ledger defect                        1.889442091722278e-16
constraint-action defect                      2.664591407560404e-16
raw Schur condition number                    3.452378615098630e4
minimum / maximum reconstruction factor       1.0 / 1.0
maximum H/R                                   0.0978374774166655
minimum scattering optical depth              19.254319053888015
incoming excision characteristics             0
```

It converges in 13 residual evaluations and `1919.85 s`, versus 8 evaluations
and `1290.00 s` for `warm_2`. The warm/cold evaluation ratio is `0.615385` and
the wall ratio is `0.671928`, so the frozen cost gate passes.

The invariant endpoint comparison is:

```text
scaled primitive-state absolute defect        7.859135564558528e-11  PASS
physical reaction-action relative defect      2.866608760891995e-8   FAIL
frozen tolerance for each                      1.0e-8
```

The warm endpoint residual is `5.048217216618925e-13`, whereas the cold
shadow stops as soon as it reaches the common root gate at
`6.398284679853816e-11`. The reaction-action mismatch is about 448 times the
cold residual in scaled units. This suggests—but does not yet prove—that the
accepted root tolerance leaves too much endpoint uncertainty for the much
tighter invariant-action equivalence gate at this Schur conditioning.

## Scientific interpretation

The result establishes several positive facts despite the binding rejection:

1. Four authentic equal-step BDF2 continuation roots exist and pass every
   declared physical gate.
2. The iteration-reserve refresh policy is robust and substantially cheaper
   than the same-history cold solve.
3. Arbitrary-BDF2 restart and a two-root suffix replay are bitwise.
4. Cumulative constraint/reaction ledgers show no drift over the accepted
   `4e-7 s` trajectory.
5. The full-versus-two-half-step comparison is comfortably within its frozen
   `0.1` diagnostic tolerances.

However, the warm and cold algorithms have not been shown to select the same
physical reaction action to `1e-8` from the same history. Therefore the
primary continuation certificate remains failed. Held-out continuation,
operational-timestep work, physical microbursts, fast averaging, and reduced
slow evolution remain unauthorized.

## Next plan

The next work package should be a definitions-only, nonpropagating endpoint
diagnostic. It must preserve this failure and may not add accepted trajectory
time.

1. Hash-lock the accepted `warm_2` result, accepted cold-shadow result, their
   common starting checkpoint, source hashes, and environment.
2. Reproduce both saved endpoint residuals and physical reaction actions
   exactly before any correction.
3. At the cold-shadow endpoint, assemble one exact complete bordered
   Jacobian and apply one nonpropagating exact Newton correction.
4. Re-evaluate the complete residual and every physical audit at the corrected
   cold candidate, then compare its primitive state and physical reaction
   action with the committed warm endpoint.
5. Record blockwise action differences and the ratio between action change,
   state correction, and residual reduction. Do not use multiplier-coordinate
   agreement as a binding invariant.
6. If the exact correction brings the action defect below `1e-8`, diagnose a
   root-accuracy/equivalence mismatch and prospectively require a tighter
   control-root stopping criterion in a new certificate. Do not retroactively
   pass this package.
7. If the exact correction does not close the action defect, investigate
   state-local reaction-map sensitivity, basis reconstruction, and Schur
   conditioning before another nonlinear trajectory run.

No new primary retry, held-out run, timestep search, microburst, averaging,
or reduced evolution is authorized by this result.

## Canonical evidence

```text
results/canonical/
causal_inner_face36_fixed_q_primary_retry_wp10c9d6c7c3b5c4f24e14l/
```

The package includes all main-root results and checkpoints, the bitwise suffix
replay, cold shadow, two half-step roots, decisive arrays, full solver traces,
profiling, provenance, machine-readable classification, and closing SHA256
checksums.
