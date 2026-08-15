# Fixed-Q Adaptive-Refresh Primary Revalidation WP10c9d6c7c3b5c4f24e11

## Classification

`adaptive_refresh_primary_nonregression_passed_heldout_retry_manifest_authorized`

The prospective adaptive exact-Jacobian policy does not alter the already
certified primary 20 ms, `h=1e-7 s` constrained BDF1/BDF2 result.

This package authorizes only a definitions-only held-out coarse retry
manifest. It does not authorize the retry itself, refined timesteps, a
fixed-`Q` microburst, or reduced slow evolution.

## Prospective policy

The solver now supports a non-default fail-closed policy:

1. assemble the existing exact complete bordered Jacobian at the first
   Newton iteration;
2. use Broyden secant updates while the frozen merit line search finds
   descent;
3. only after all 12 line-search lengths fail, assemble one additional exact
   Jacobian at the unchanged iterate;
4. preserve the residual, row scales, merit, step bounds, physical gates,
   and `1e-10` root tolerance;
5. cap the complete exact assemblies at two.

The repository default remains the historical per-iteration behavior. The
adaptive policy is selected explicitly only by the prospective fixed-`Q`
audit runner.

## Primary BDF1 non-regression

```text
initial maximum scaled residual              6.342948677146715e-10
accepted maximum scaled residual             4.031505120854680e-13
function evaluations                         2
exact Jacobian assemblies                    1
optional line-failure refreshes               0
Schur identity closure                       2.140021778427095e-14
```

All acceptance gates pass, and the accepted BDF1 history survives restart
serialization bitwise.

## Primary BDF2 non-regression

The adaptive run follows the complete historical solver path, including the
same accepted `alpha=0.0625` and `alpha=0.25` backtracking decisions:

```text
initial maximum scaled residual              2.553433464477697e+0
accepted maximum scaled residual             1.342875810550481e-11
function evaluations                         16
linear solves                                 7
exact Jacobian assemblies                    1
optional line-failure refreshes               0
Schur identity closure                       5.885745432074176e-13
```

The BDF2 replay is bitwise. Exact-increment storage, direct-rate parity,
Q3, reconstruction, reaction/action ledgers, physical guards, primitive
change, and outgoing excision all pass.

## Strong non-regression gate

Every decisive BDF1 and BDF2 array is bitwise equal to the certified
WP10c9d6c7c3b5c4f24e8 baseline:

- primitive charts and increments;
- BDF and interval rates;
- multipliers and physical reaction action;
- complete augmented residuals.

Thus the new branch changes no accepted primary state, solver decision, or
scientific result when the historical Broyden path remains viable.

## Next step

Freeze a held-out coarse retry using the same prospective policy. It must:

- reconstruct the already accepted held-out BDF1 restart exactly;
- reproduce the historical BDF2 path through the `1.56255e-9` stalled
  endpoint;
- record exactly one additional exact assembly with reason
  `line_search_failure`;
- reach the unchanged `1e-10` root gate;
- pass all non-root gates and bitwise BDF2 replay.

Only that pass may authorize restarting the `5e-8` and `2.5e-8 s` convergence
ladder. The historical held-out rejection remains part of the canonical
record.
