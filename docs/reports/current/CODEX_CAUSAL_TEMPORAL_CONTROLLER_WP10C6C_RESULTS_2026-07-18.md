# Causal Temporal-Controller WP10c6c Results

Date: 2026-07-18

## Verdict

The observable-controlled backward-Euler step-doubling implementation works
as designed locally, preserves the complete physical contract, reproduces
bitwise across restart, and reduces nonlinear work. It does not pass the
predeclared accumulated-accuracy gate over eight shared timestep ceilings.

The bounded N16 result is:

```text
adaptive accepted steps                   9
adaptive rejected trials                  0
adaptive implicit solves                 27
fixed-reference steps/solves             64
reference/adaptive Jacobian-work ratio   2.1918
maximum accepted local normalized error  0.7189
final normalized error vs reference      3.6948
```

The final adaptive state violates the total cooling, cooling outside `6 rg`,
and `Delta ln(H/R)` accuracy gates. All nonlinear, algebraic, independent
five-field ledger, causal, optical-depth, Roche, and emergency-change gates
pass.

Per the locked coarse-to-fine rule, N32 was not launched. No tolerance was
relaxed and no alternative controller was tried in WP10c6c.

This is a certified bounded negative result for using the WP10c6b local-error
contract as a production accumulated-error controller. It is not a negative
result for the causal DAE or for time evolution.

## Implemented Contract

WP10c6c adds a reusable controller that, for each trial timestep, computes:

1. one backward-Euler full step;
2. one backward-Euler first half step;
3. one backward-Euler second half step;
4. independent state and physical-ledger audits for all three solves;
5. the immutable `causal-five-field-observables-v1` differences;
6. the bounded first-order update
   `clip(0.8/sqrt(error), 0.25, 2.0)`.

A trial advances history only when:

```text
full-step contract                         pass
first-half-step contract                   pass
second-half-step contract                  pass
maximum normalized local observable error <= 1
```

The accepted state is exactly the two-half-step state. Its total physical
increment and accepted timestep become the next predictor history.

The shared source-compatible regression context is now a library fixture
rather than duplicated private runner code. The v1 temporal gates are also a
single exported constant, and configuration rejects missing or additional
gate names.

## Predeclared Campaign

The N16 campaign starts from:

```text
checkpoint       causal_wp10c5q_N016_final.npz
checkpoint SHA   9b8247536daf2ddd2868f571d911751062a90d08690499ffd69794bff9046e7e
elapsed time     8.484232672865630e-4 s
stream           exact circularized C2 regression stream
```

The fixed construction is:

```text
shared passing ceiling        1.921821997458634e-3 s
initial controller timestep   9.609109987293168e-4 s
extension duration            1.537457597966907e-2 s
target elapsed time           1.622299924695563e-2 s
fixed reference steps         64
fixed reference timestep      2.402277496823292e-4 s
restart split                 after accepted step 3
```

The duration is eight shared ceilings. It is only about `1.81e-8` of the
global loading time and is not a physical-duration simulation.

## Adaptive Trajectory

The accepted timestep sequence is:

```text
9.609109987e-4
1.921821997e-3
2.023003328e-3
2.035755057e-3
2.050102558e-3
2.066419207e-3
2.084794175e-3
2.105319139e-3
1.264495195e-4 s
```

The final short step lands exactly on the target. The normalized local-error
sequence is:

```text
0.14625, 0.57758, 0.63201, 0.63107, 0.62993,
0.62877, 0.62758, 0.71890, 0.00273
```

The controller therefore behaves stably and settles near its expected
first-order equilibrium. It never invokes a rejected retry.

Across all 27 diagnostic/accepted implicit solves:

```text
maximum scaled nonlinear residual       9.8192e-12
maximum scaled algebraic residual       2.2424e-14
maximum physical ledger defect          4.7693e-12
maximum Newton iterations               6
```

The final adaptive state retains:

```text
maximum H/R                             0.101079
minimum scattering optical depth       18.8431
inner incoming characteristics          0
outer incoming characteristics          2
Roche channel                           closed, nonchoked
```

## Restart Identity

An independent replay writes and reloads the complete state and controller
history after accepted step three. It then reaches the target with:

```text
same accepted timestep sequence         true
same rejected-trial count               true
final state array bitwise identical     true
final physical increment bitwise equal  true
final dt_next and previous_dt equal     true
checkpoint round trip bitwise           true
```

The selected final checkpoint is:

```text
outputs/checkpoints/causal_five_field_wp10c6c/
  causal_wp10c6c_N016_final.npz

SHA-256
50f939011cfcf383350e3e07fe5eb926dab8a56b19c5d72b72d3c10303914235
```

Generated checkpoints remain ignored under the artifact policy.

## Fixed Reference

The reference performs 64 fixed backward-Euler steps of
`2.402277496823292e-4 s`. Every step passes the same state and independent
physical-ledger contract:

```text
maximum scaled nonlinear residual       2.3245e-12
maximum physical ledger defect          1.3205e-12
completed steps                         64/64
```

Its final state has `H/R=0.100606`, minimum scattering optical depth
`18.8423`, zero inner incoming characteristics, two outer incoming
characteristics, and a closed Roche edge.

The reference is four times finer than the controller's initial timestep and
roughly four to nine times finer than the accepted half steps. WP10c6c does
not yet certify its continuum error; that is the first gate of the next
package.

## Accumulated Accuracy

At the exact common target:

| Observable | Adaptive/reference error | Gate | Ratio |
|---|---:|---:|---:|
| Total cooling proxy | `3.43558e-3` | `1e-3` | `3.436` |
| Cooling outside `6 rg` | `2.26423e-3` | `1e-3` | `2.264` |
| Inner accretion rate | `1.71764e-4` | `1e-3` | `0.172` |
| Maximum `Delta ln(H/R)` | `7.38964e-3` | `2e-3` | `3.695` |
| Integrated conserved fields | `6.34293e-6` | `1e-3` | `0.0063` |
| Baseline-scaled full state | `1.86942e-3` | `2e-3` | `0.935` |

The maximum normalized error is `3.69482`, controlled by the thickness
profile.

This distinction matters:

- each local full/two-half comparison passes;
- the accepted local error settles near `0.63` of its gate;
- accumulated first-order error over the output interval does not pass the
  same global gate.

The implementation is behaving consistently with a local controller. The
WP10c6b contract was insufficient as a global error budget.

## Work Audit

| Work measure | Adaptive | Reference | Reference/adaptive |
|---|---:|---:|---:|
| Implicit solves | 27 | 64 | `2.370` |
| Function evaluations | 5,429 | 11,904 | `2.193` |
| Jacobian evaluations | 146 | 320 | `2.192` |
| Newton iterations | 146 | 320 | `2.192` |

The controller is computationally useful, but that gain cannot certify a
trajectory whose accumulated observables miss their gates.

## Classification

WP10c6c establishes:

```text
controller implementation                  supported
local observable control                   supported
physical-contract preservation             supported
bitwise restart continuation               certified
bounded N16 accumulated accuracy            failed
N32 campaign                               not authorized
production long-duration controller         not certified
```

It does not establish physical relaxation, stability, a hot state, a cycle,
tide, or wind.

## Locked Next Work

WP10c6d should remain N16-only and address global accuracy without changing
physics:

1. save fixed-step endpoints at 32, 64, and 128 subdivisions of the same
   predeclared output interval;
2. demonstrate first-order convergence and require the 64-to-128 reference
   uncertainty to be a declared fraction of every global observable gate;
3. compare the existing controller endpoint with the converged reference;
4. implement one horizon-budget rule in which a step may spend at most its
   fraction `dt / T_output` of the global observable budget;
5. repeat the same restart and work audits;
6. authorize N32 only if the budgeted N16 endpoint passes.

Do not lower the safety factor by fitting this one trajectory, relax any
observable gate, run N32 before the N16 gate, or begin N64/N128 production,
tide, wind, stability, hot-state, or cycle work.

## Verification

```text
focused causal controller suite             38 passed
full repository suite                       497 passed, 4 subtests
N16 adaptive trajectory                     passed locally
N16 interrupted restart replay              bitwise passed
N16 64-step fixed reference                 passed
N16 accumulated observable gate             failed
N32 production campaign                     correctly skipped
repository hygiene                         passed
Python compilation                          passed
git diff --check                            passed
```

## Reproduction

```text
PYTHONPATH=src python3 \
  scripts/run_causal_temporal_controller_wp10c6c.py \
  --n-cells 16 \
  --output outputs/tables/causal_temporal_controller_wp10c6c_N016.json
```

Machine-readable output and checkpoints remain ignored.
