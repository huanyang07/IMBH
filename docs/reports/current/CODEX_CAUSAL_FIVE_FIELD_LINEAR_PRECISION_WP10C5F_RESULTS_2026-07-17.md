# Causal Five-Field Linear Precision WP10c5f Results

Date: 2026-07-17

## Verdict

The frozen N16 path candidate does not have a recoverable reduced
linear-solve precision defect. LAPACK equilibration and iterative refinement
solve the linear system accurately, but they do not change the Newton
correction or reduce the nonlinear residual.

```text
frozen reduced matrix              80 x 80
raw condition estimate             1.03e10
equilibrated condition estimate    27.50
linear relative residual           1.49e-16
direct/refined correction defect   2.62e-14
recoverable precision              NOT DEMONSTRATED
N16 precision rerun                NOT AUTHORIZED
N32                                NOT ATTEMPTED
```

Stationary roots, physical evolution, tide, and wind remain blocked.

## Frozen Candidate

The audit uses only the final `N=16`, target-change `1e-3`,
path-integrated-storage candidate from WP10c5e:

```text
timestep                           1.56892e-7 s
maximum scaled residual            1.42466e-6
controlling row                    cell 15 angular momentum
maximum scaled primitive change    9.96797e-4
Roche edge                         closed
```

The same residual and declared centered `2e-6` Jacobian are used for every
linear-solve comparison.

## Equilibration And Refinement

LAPACK `dgeequ`/`dgesvx` applies both row and column equilibration:

```text
equed                              B
row scale range                    5.88e-12 to 2.30e-6
column scale range                 1 to 6.43e5
equilibrated condition estimate    27.4978
reciprocal condition estimate      1.83276e-2
dgesvx backward error              9.86e-17
dgesvx forward error bound         4.43e-8
```

The current direct solve, equilibrated/refined solve, and fourth-order
Jacobian solve all have linear relative residual `1.49e-16`. The direct and
`dgesvx` corrections differ by only `2.62e-14` relative.

Applying either full correction gives the same nonlinear result:

```text
correction maximum                 3.23837e-12
nonlinear residual after update    3.34921e-6
```

The nonlinear residual increases rather than passing `1e-8`. Therefore
changing the dense linear solver cannot unlock this candidate.

## Finite-Difference Separation

One fourth-order centered Jacobian was assembled at the same `2e-6` scale.
This is a comparator, not a finite-difference scan.

```text
second-order condition             1.030047792e10
fourth-order condition             1.030047778e10
relative Jacobian Frobenius defect 4.47e-12
relative correction difference     5.98e-8
```

The fourth-order correction also gives nonlinear residual `3.34921e-6`.
Neither second-order truncation nor the direct linear solve explains the
finite nonlinear floor.

## Weakest Mode

The backward-Euler reduced matrix's weakest right direction is localized at
the innermost cell, not at the Roche edge:

```text
right maximum-cell fraction        0.99999999998
right lnT field norm               0.999999993
right outermost-cell fraction      1.54e-17
```

The weakest left direction is likewise localized at cell zero and is mainly
thermal. This is distinct from the outer thermal/stress direction of the
stationary flux-primary embedding and should not be interpreted as the same
mode.

## Classification

WP10c5f excludes:

- inadequate row/column equilibration;
- inaccurate direct linear solution;
- recoverable LAPACK iterative-refinement error;
- a material second-order versus fourth-order Jacobian difference.

The bounded classification is:

```text
nonlinear residual-evaluation/directional-consistency floor
at corrections near 1e-12
```

This is a numerical classification at a deliberately nonstationary seed. It
does not establish a physical mode, instability, or limit cycle.

## Locked Next Step

If work continues, perform one component-wise residual directional-consistency
audit:

1. freeze this candidate and the direct Newton correction;
2. compare `R(p + delta p) - R(p)` with `J delta p` cell by cell;
3. separate face-flux differences, geometric/thermal sources, path storage,
   and responsive-height work;
4. use compensated differences for diagnostics only;
5. change a production residual term only if one component fails a declared
   identity and convergence test;
6. repeat N16 once only after that repair.

Do not change tolerances, run another linear solver, scan finite-difference
steps, attempt N32, or introduce roots, tide, or wind.

## Verification

```text
PYTHONPATH=src python3 \
  scripts/run_causal_five_field_consistent_initial_step_wp10c5d.py \
  --temporal-storage-scheme path_integrated \
  --linear-precision-audit \
  --output outputs/tables/causal_five_field_linear_precision_wp10c5f.json
```

Machine-readable output:

```text
outputs/tables/causal_five_field_linear_precision_wp10c5f.json
```
