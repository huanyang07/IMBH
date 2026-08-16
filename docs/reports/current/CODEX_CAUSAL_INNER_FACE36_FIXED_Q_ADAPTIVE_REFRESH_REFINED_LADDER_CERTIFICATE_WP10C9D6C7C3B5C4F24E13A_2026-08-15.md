# Fixed-Q Adaptive-Refresh Refined History Ladder Certificate WP10c9d6c7c3b5c4f24e13a

## Final classification

`adaptive_refresh_refined_fixed_Q_history_ladder_certified_one_Q_manifest_authorized`

The authentic constrained BDF1-to-BDF2 history ladder passes at both the
primary 20 ms state and the held-out 16 ms state for

```text
h = 1.0e-7, 5.0e-8, 2.5e-8 s.
```

All four newly executed middle/fine stages pass their complete nonlinear,
constraint, storage, reconstruction, physical, reaction, conditioning,
restart, and bitwise-replay contracts. All eight prospectively binding
adjacent-pair orders for complete state-space BDF rate and physical reaction
action exceed `0.9`.

This certifies the local authentic-history continuous-limit interpretation
under the selective line-search-failure refresh policy. It authorizes only a
new definitions-only bounded one-`Q` continuation/cost manifest. It does not
authorize the fixed-`Q` micro-solver, a physical microburst, fast averaging,
or reduced slow evolution.

## Convergence certificate

Errors are absolute L2 differences from each state’s frozen continuous
constrained reference. Parentheses show the corresponding fixed-reference
relative errors.

| State | Pair | Quantity | Prior error | Current error | Order |
|---|---|---|---:|---:|---:|
| 20 ms | coarse to middle | State-space BDF rate | `6145.2236` (`0.0555301`) | `3109.8304` (`0.0281014`) | `0.9826296` |
| 20 ms | coarse to middle | Physical reaction action | `6119.8664` (`0.0553011`) | `3102.2251` (`0.0280327`) | `0.9801968` |
| 20 ms | middle to fine | State-space BDF rate | `3109.8304` (`0.0281014`) | `1565.3236` (`0.0141447`) | `0.9903750` |
| 20 ms | middle to fine | Physical reaction action | `3102.2251` (`0.0280327`) | `1563.2599` (`0.0141261`) | `0.9887457` |
| 16 ms | coarse to middle | State-space BDF rate | `6479.6950` (`0.0569683`) | `3279.3579` (`0.0288315`) | `0.9825125` |
| 16 ms | coarse to middle | Physical reaction action | `6453.0128` (`0.0567339`) | `3271.2769` (`0.0287605`) | `0.9801190` |
| 16 ms | middle to fine | State-space BDF rate | `3279.3579` (`0.0288315`) | `1650.7845` (`0.0145134`) | `0.9902616` |
| 16 ms | middle to fine | Physical reaction action | `3271.2769` (`0.0287605`) | `1648.5834` (`0.0144941`) | `0.9886270` |

The minimum observed order is `0.9801190`, above the frozen `0.9` gate. The
orders improve modestly from the coarse-to-middle pair to the middle-to-fine
pair and agree closely across the two committed states.

## Solver and physical result

Each of the eight BDF roots satisfies the unchanged `1e-10` scaled-residual
gate. Six roots require the prospectively allowed line-search-failure exact
refresh; two fine roots close using only the initial exact assembly. No root
uses more than two exact Jacobian assemblies.

Across all new stages:

- every fixed-`Q` target and physical reaction ledger closes;
- mapped-storage and responsive-height parity pass;
- every reconstruction path remains inactive with factor exactly one;
- the reaction Schur map retains rank three with condition about
  `3.37e4-3.46e4`;
- height, optical-depth, primitive-change, and outgoing-excision gates pass;
- every BDF1 checkpoint roundtrip and complete BDF2 replay is bitwise.

The selective refresh therefore repairs stale Broyden linearization without
altering the constrained equations, residual tolerance, or accepted history.
The earlier one-assembly rejection remains valid as a solver-budget result.

## Scientific interpretation

The previous synthetic-history BDF2 failure is not reproduced when BDF2 uses
an authentic accepted constrained BDF1 history. The fixed-`Q` residual,
reaction action, and physical history now exhibit state-robust approximately
first-order convergence toward the continuous constrained rate, as expected
for this two-step startup-limit experiment.

This is a local history certificate, not a multi-step trajectory certificate.
It does not establish long-time stability, global BDF2 order, fast attraction,
averaging-window convergence, or a slow closure.

## Verification and canonical evidence

Final canonical evidence is stored under
`results/canonical/causal_inner_face36_fixed_q_adaptive_refresh_refined_ladder_hardened_wp10c9d6c7c3b5c4f24e13a/`.
It aggregates only the four already committed and checksum-validated stage
packages. No physical rung is rerun during finalization.

The final checksums close, and the hardened ladder plus both parent
adaptive-refresh focused suites pass:

```text
19 passed in 1.05 s
```

## Authorized next package

Create a definitions-only bounded one-`Q` continuation and cost manifest on
the middle spatial layout. It should prospectively freeze:

1. a short, restartable multi-step horizon;
2. exact accepted BDF histories and centralized step acceptance;
3. cross-step Jacobian/Broyden reuse and the same line-failure refresh rule;
4. refresh frequency, residual evaluations, wall time, and checkpoint cost;
5. selected step-doubling or half-step consistency audits;
6. stepwise Q3, reaction, constraint-work, storage, reconstruction, physical,
   and excision gates;
7. fail-fast cost and scientific decision branches.

The package must execute no trajectory. Only a subsequently reviewed manifest
may authorize the bounded continuation/cost pilot.
