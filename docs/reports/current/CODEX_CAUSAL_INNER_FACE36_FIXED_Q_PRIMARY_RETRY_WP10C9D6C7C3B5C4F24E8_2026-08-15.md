# Fixed-Q Schur-Repaired Primary Retry WP10c9d6c7c3b5c4f24e8

## Classification

`fixed_Q_primary_case_recovered_remaining_history_manifest_authorized`

The bounded middle-layout 20 ms, `h=1e-7 s` execution passes one authentic
constrained BDF1 startup, serialized restart, one exact-history constrained
BDF2 step, and bitwise BDF2 replay.

No fixed-`Q` microburst or reduced slow evolution is authorized. The result
authorizes only a fresh definitions-only manifest for the remaining five
cases of the previously frozen physical-history ladder.

## BDF1

The repaired ordinary solver reproduces the saved endpoint recovery:

```text
initial maximum scaled residual              6.342948677146715e-10
accepted maximum scaled residual             4.031505120854680e-13
Newton iterations                            1
function evaluations                         2
exact Jacobian assemblies                    1
line-search alpha                            1
Schur identity closure                       2.140021778427095e-14
```

The complete residual, exact Q3, storage parity, inactive reconstruction,
reaction and action ledgers, physical guards, primitive-change bound, and
outgoing excision all pass. The BDF history is therefore accepted and
serialized for the first time under the corrected exact-increment contract.

## BDF2 and replay

The authentic BDF2 step uses only the accepted BDF1 primitive, mapped-storage,
responsive-height, and timestep history. It reaches

```text
maximum scaled residual                      1.342875810550481e-11
Newton iterations                            7
function evaluations                         16
exact Jacobian assemblies                    1
Schur identity closure                       5.885745432074176e-13
Q3 relative defect                           0
storage parity defect                        2.232443577708263e-14
mapped endpoint/path closure                 2.567721707917881e-10
minimum/maximum reconstruction factor        1 / 1
incoming excision characteristics            0
```

The nonlinear path includes damped Broyden corrections, but remains within the
prospectively frozen one-exact-Jacobian budget. Repeating the complete BDF2
solve from the serialized BDF1 restart reproduces the state, histories,
multipliers, counters, diagnostics, and accepted result bitwise.

## Interpretation

The original primary BDF1 rejection was solely the direct `3x3` Schur inverse
closure. The prospectively selected equilibrated/refined solve removes that
blocker without relaxing `1e-12`, changing the physical reaction action, or
altering the fixed-`Q` equations.

This result also establishes that an authentic accepted BDF1 history can seed
a converged constrained BDF2 root. It does not yet establish the two adjacent
small-step convergence orders at both the 20 ms and held-out 16 ms states.

## Next step

Freeze a remaining-ladder manifest, then run fail-fast:

1. held-out 16 ms state at `h=1e-7 s`;
2. both states at `h=5e-8 s`;
3. both states at `h=2.5e-8 s`.

Each case must preserve exact-increment binding storage, direct-rate parity,
the selected Schur solve, all physical/ledger gates, restart roundtrip, and
bitwise BDF2 replay. Both adjacent observed orders must remain at least `0.9`
for the state-space BDF rate and physical reaction action at both states.

A complete ladder pass may authorize only a definitions-only bounded one-`Q`
execution manifest. It may not directly authorize a microburst or reduced
slow evolution.
