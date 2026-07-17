# Causal Five-Field Temporal Storage Increment WP10c5e Results

Date: 2026-07-17

## Verdict

Direct endpoint subtraction is a real cancellation source in the tangent-sized
backward-Euler residual, but replacing it with a converged path-integrated
storage increment does not unlock the N16 step.

```text
path storage identity/convergence       PASS
N16 target change 1e-4                  FAIL at 3.77e-6
N16 target change 1e-3                  FAIL at 1.42e-6
N32                                     NOT ATTEMPTED
physical evolution                      BLOCKED
stationary roots, tide, wind            BLOCKED
```

The remaining floor is associated with an approximately `1.03e10` conditioned
reduced Newton matrix and corrections of order `1e-12`. This work package does
not establish a physical instability or missing DAE condition.

## Endpoint Cancellation Audit

For each cell, the original finite storage uses

```text
Delta U = U(p_new) - U(p_old)
```

and then divides by `c Delta t`. At the first attempted timestep,
`Delta t=1.56892e-8 s`, so roundoff in the subtraction of two large mapped
states is amplified.

The replacement integrates the mapped-state derivative along the declared
straight primitive path:

```text
Delta U = integral_0^1 J_U(p_old + lambda Delta p) Delta p dlambda.
```

A fourth-order centered directional derivative is combined with
Gauss-Legendre quadrature. The responsive-height one-form is integrated on the
same path:

```text
Delta W_H = integral_0^1 Pi(lambda) dlnH/dlambda dlambda,
```

and mapped into all four Killing storage components at each quadrature point.
The path scheme is restricted to reduced states whose conserved and face maps
are exact; the flux-primary endpoint scheme remains the default.

At the final endpoint-scheme candidates, the maximum scaled endpoint/path
rate defects are:

| Target primitive change | Endpoint/path defect |
|---:|---:|
| `1e-4` | `7.05e-6` |
| `1e-3` | `3.16e-7` |

The first discrepancy exceeds the failed endpoint residual. Endpoint
cancellation is therefore demonstrated rather than merely suspected.

## Path Convergence

At the path-scheme candidates:

```text
order 4 versus order 8       <= 3.00e-10
half directional step        <= 2.53e-9
double directional step      <= 1.33e-9
declared convergence gate        5.00e-9
```

The selected order-2 path differs from order 8 by at most `2.85e-10`. The
global mass, radial momentum, angular momentum, Killing energy, and relaxing
stress ledger defects are:

```text
target 1e-4: 5.60e-17
target 1e-3: 3.73e-17
```

Focused tests also include a `Delta lnSigma` of about `1e-12`. The path result
recovers the analytic `D` increment to about `3e-12` relative and improves the
endpoint-subtraction error by more than six orders of magnitude.

## Bounded N16 Rerun

No nonlinear tolerance, physical bound, active Roche branch, or timestep
target was changed.

| Target change | Timestep | Final residual | Controlling row | Result |
|---:|---:|---:|---|---|
| `1e-4` | `1.56892e-8 s` | `3.77e-6` | cell 14 angular momentum | fail |
| `1e-3` | `1.56892e-7 s` | `1.42e-6` | cell 15 angular momentum | fail |

For both attempts:

```text
primitive-map residual       0
face-map residual            0
Roche gate                   closed before and after
minimum scattering depth     about 1.70e4
accepted-state clipping      none
storage convergence          pass
```

The smaller step improves relative to the endpoint value `4.79e-6`; the
larger step does not improve relative to `1.40e-6`. In both cases the Newton
matrix condition estimate is approximately `1.03e10`. The final corrections
fall to `1.5e-12` and `3.2e-12`, after which the line search cannot reduce the
residual.

## Classification

WP10c5e establishes:

1. direct endpoint subtraction was numerically unsafe at these timesteps;
2. the cancellation-safe storage path is internally converged and
   conservative;
3. storage cancellation was not the only source of the nonlinear floor;
4. the unchanged N16 evolution gate still fails.

The bounded classification is:

```text
verified temporal-storage cancellation
plus unresolved double-precision reduced-Newton conditioning
at a deliberately nonstationary N16 seed
```

This is not evidence for equilibrium, marginality, stability, or instability.

## Locked Next Step

If work continues, perform one reduced linear-solve precision audit only:

1. freeze the final N16 path candidate and its `80 x 80` Jacobian/residual;
2. report row/column equilibration factors and singular-vector localization;
3. compare the current solve with one equilibrated solve and iterative
   refinement using the same residual and Jacobian;
4. distinguish Jacobian finite-difference error from linear-solve error;
5. repeat the two N16 gates once only if the audit demonstrates recoverable
   precision;
6. attempt N32 only after an unchanged N16 pass.

Do not alter tolerances, scan finite-difference steps, add boundary conditions,
launch stationary roots, or introduce tide/wind.

This action was completed in WP10c5f. Equilibration reduces the frozen matrix
condition estimate to `27.5`, but the direct and refined corrections agree to
`2.62e-14` and produce the same failed nonlinear residual. See
`CODEX_CAUSAL_FIVE_FIELD_LINEAR_PRECISION_WP10C5F_RESULTS_2026-07-17.md`.

## Verification

```text
PYTHONPATH=src python3 -m pytest -q \
  tests/test_causal_inner_dae_system.py

PYTHONPATH=src python3 \
  scripts/run_causal_five_field_consistent_initial_step_wp10c5d.py \
  --temporal-storage-scheme path_integrated \
  --output \
  outputs/tables/causal_five_field_temporal_storage_increment_wp10c5e.json
```

Machine-readable output:

```text
outputs/tables/causal_five_field_temporal_storage_increment_wp10c5e.json
```
