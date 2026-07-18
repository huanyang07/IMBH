# Causal Refined Reference WP10c6e Results

Date: 2026-07-18

## Verdict

The direct N16 backward-Euler reference is now sufficiently resolved for the
bounded horizon-budget closure experiment.

All 896 fixed steps in the predeclared 128/256/512 ladder pass every
nonlinear, algebraic, independent physical-ledger, causal, optical-depth,
Roche, and emergency-change gate. All three accepted endpoints are persisted
as checksummed, bitwise-reloadable restart files.

The six immutable accuracy observables retain clean first-order convergence:

```text
0.99866 <= p <= 1.00075
```

The raw 256-to-512 endpoint differences consume at most:

```text
0.15165 of an observable gate
```

This is below the unchanged maximum reference uncertainty of `0.25`.
Therefore:

```text
N16 refined reference                    certified
N16 horizon-budget closure               authorized
BDF2 disk certification                  not yet authorized
N32 controller campaign                  not authorized
long evolution and new physics           not authorized
```

This is a numerical reference result over a short controller interval. It is
not evidence for physical relaxation, stability, a hot state, or a cycle.

## Locked Contract

WP10c6e retains the exact WP10c6d problem and acceptance rules:

```text
mesh                                  N16 only
initial checkpoint                    accepted WP10c5q restart
output interval                       1.537457597966907e-2 s
reference subdivisions                128, 256, 512
maximum fine uncertainty              0.25 of each global gate
minimum observed order                0.75
negligible-order floor                1e-3 of a gate
adaptive controller                   not run
physics changes                       none
```

For every non-negligible observable, certification requires:

```text
E256-512 / gate <= 0.25
log2(E128-256 / E256-512) >= 0.75
```

No Richardson-extrapolated endpoint replaces the raw-difference gate, and no
gate was changed after seeing the result.

## Initial State

All references start from:

```text
checkpoint
outputs/checkpoints/causal_five_field_wp10c5k/
  causal_wp10c5q_N016_final.npz

SHA-256
9b8247536daf2ddd2868f571d911751062a90d08690499ffd69794bff9046e7e

initial elapsed time                  8.484232672865630e-4 s
target elapsed time                   1.622299924695563e-2 s
stream                                exact circularized C2 regression stream
```

The interval remains far shorter than a loading, cooling, or thermal time and
exists to certify temporal integration accuracy.

## Persisted References

| Subdivisions | Timestep (s) | Steps | Function evaluations | Jacobians | Checkpoint SHA-256 |
|---:|---:|---:|---:|---:|---|
| 128 | `1.201138748e-4` | 128/128 | 23,808 | 640 | `33b11f926c6f1511af7309a654b37038ca9b42474f00494d4bca58f29fd3f258` |
| 256 | `6.005693742e-5` | 256/256 | 47,616 | 1,280 | `18be729107ea561b9f377d51938368adcf4bb21e47314c8fa831ff3c911e2397` |
| 512 | `3.002846871e-5` | 512/512 | 95,232 | 2,560 | `a8bf5043ecf27792acece9c27984d73bc76c0c9d7178898a60bcd0d4f1926e79` |

Every step used five Newton/Jacobian evaluations. The complete ladder required
166,656 function evaluations and 4,480 Jacobians. These files are the reusable
N16 backward-Euler reference and should not be regenerated for each temporal
method.

The checkpoints live under:

```text
outputs/checkpoints/causal_five_field_wp10c6e/
```

They are ignored runtime artifacts, but each path, hash, construction, and
aggregate solver audit is recorded in the machine-readable WP10c6e output and
in the restart provenance.

## Numerical Contracts

| Subdivisions | Max scaled residual | Max physical-ledger defect | Max primitive change |
|---:|---:|---:|---:|
| 128 | `1.156e-12` | `2.200e-12` | `4.025e-3` |
| 256 | `5.856e-13` | `4.162e-12` | `2.032e-3` |
| 512 | `2.984e-13` | `9.022e-12` | `1.021e-3` |

All are below the unchanged contracts. Conservation telescoping defects remain
below `1.56e-16`, algebraic residuals remain below `7.05e-15`, and every
checkpoint reload is bitwise identical.

The final state remains comfortably inside its physical gates:

| Subdivisions | Max `H/R` | Min scattering depth | Inner/outer incoming characteristics |
|---:|---:|---:|---:|
| 128 | `0.1005294` | `18.84212` | `0 / 2` |
| 256 | `0.1004909` | `18.84206` | `0 / 2` |
| 512 | `0.1004716` | `18.84202` | `0 / 2` |

The Roche channel remains closed and nonchoked.

## Reference Convergence

| Observable | `E128-256/gate` | `E256-512/gate` | Order | Pass |
|---|---:|---:|---:|---:|
| Total cooling | `0.28103` | `0.14059` | `0.99920` | yes |
| Cooling outside `6 rg` | `0.18463` | `0.09235` | `0.99950` | yes |
| Inner accretion rate | `0.01403` | `0.00702` | `0.99926` | yes |
| `Delta ln(H/R)` | `0.30302` | `0.15165` | `0.99867` | yes |
| Integrated conserved fields | `0.00052` | `0.00026` | `1.00075` | yes |
| Baseline-scaled full state | `0.07667` | `0.03837` | `0.99866` | yes |

The 128-to-256 `Delta ln(H/R)` difference still consumes `0.303` of its
gate, confirming that stopping at 256 would have failed the declared
uncertainty allowance. The direct 512-step endpoint was the necessary
reference level.

The result also confirms that WP10c6c's accumulated cooling and thickness
errors are ordinary first-order temporal truncation error rather than
nonlinear or conservation noise.

## Classification

WP10c6e establishes:

```text
fixed-step reference implementation          certified
128/256/512 N16 trajectories                 certified
persisted restart/checksum contract          certified
first-order endpoint convergence             certified
512-step reference uncertainty               passes declared gate
N16 horizon-budget closure                   authorized
N32 campaign                                 still blocked
BDF2 disk campaign                           still blocked
long evolution                               not certified
```

It establishes no physical relaxation, stability, hot state, cycle, tide, or
wind.

## Locked Next Work

The next atomic package is WP10c6f:

1. load and verify the saved 128/256/512 reference checkpoints;
2. run exactly one N16 `dt/T_output` horizon-budget trajectory;
3. run one interrupted/restarted replay and require bitwise final equality;
4. compare the controller endpoint with S512;
5. add the measured S256-to-S512 uncertainty to every controller error;
6. require the conservative sum to remain below every immutable gate;
7. record work per simulated second and compare with the S512 reference;
8. freeze backward Euler as a reference/fallback backend after this one
   closure experiment, regardless of whether it passes efficiently.

Do not tune the safety factor, change the budget law, relax an observable
gate, or run N32 backward Euler merely because the N16 endpoint passes. BDF2
implementation may begin only in a separate package after WP10c6f closes the
backward-Euler production question.

The path toward WP10c7d remains:

```text
WP10c6f  one N16 horizon-budget BE closure
WP10c7a  increment-primary BDF1/BDF2 operator and method tests
WP10c7b  fixed-step N16 BDF2 certification against S512
WP10c7c  adaptive N16 BDF2 certification
WP10c7d  matched N32 adaptive BDF2 confirmation
```

No N64/N128 production, long-timescale, tide, wind, stability, hot-state, or
cycle run is authorized by this result.

## Verification

```text
focused temporal-reference/controller tests  13 passed before production
full repository suite                          502 passed, 4 subtests
repository hygiene                            passed for 642 tracked files
128-step fixed reference                      passed and persisted
256-step fixed reference                      passed and persisted
512-step fixed reference                      passed and persisted
checkpoint reuse ladder                       passed
reference order gate                          passed
reference uncertainty gate                    passed
adaptive production branch                    not run
```

## Reproduction

Run or reuse the complete ladder:

```text
PYTHONPATH=src python3 \
  scripts/run_causal_refined_reference_wp10c6e.py \
  --output \
  outputs/tables/causal_refined_reference_wp10c6e_N016.json
```

To recompute one missing rung, add one of:

```text
--subdivisions 128
--subdivisions 256
--subdivisions 512
```

Machine-readable output and restart files remain ignored under the artifact
policy.
