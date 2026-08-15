# Exact-Increment Fixed-Q Storage Recertification WP10c9d6c7c3b5c4f24e5

## Classification

`fixed_Q_exact_increment_residual_resolution_failed`

The prospectively frozen package formally fails its quadratic model-error
order gate. Nevertheless, the exact-increment representation repairs the
actual saved endpoint root by a wide margin:

```text
repaired full-step maximum residual       7.934316105590028e-13
unchanged root gate                        1.0e-10
```

The failure and the repair must both be preserved. No physical trajectory or
history ladder was executed, and reduced slow evolution remains blocked.

## What the repair changed

The fixed-`Q` nonlinear unknown already contains the scaled primitive
increment. The repaired binding path now:

1. converts that unknown directly to the physical primitive increment;
2. verifies exactly that `new = old + increment`;
3. reconstructs the affine node increment from the supplied increment;
4. integrates mapped and responsive-height storage along the unchanged state
   path;
5. retains endpoint subtraction and direct-rate evaluation as independent
   audits;
6. stores the exact accepted increment in BDF history and restart state.

The physical operator, endpoint, BDF coefficients, history definition, row
scales, merit norm, and `1e-10` residual tolerance are unchanged.

Seventeen focused monolithic/BDF/fixed-`Q` tests pass after the repair.

## Saved endpoint result

At the same saved 20 ms, `h=1e-7 s` endpoint:

```text
base maximum residual                     4.833036887363917e-10
base residual L2 norm                     1.871942902360754e-09
fresh Newton correction L2 norm           2.377336347918289e-09
linear relative residual                  1.931258503632581e-14
mapped increment/direct defect            4.488600462766586e-16
height increment/direct defect            1.891495918686483e-16
mapped endpoint/path closure defect        8.088209488131910e-11
full-step model error / base residual      8.569473265553016e-04
full-step maximum residual                 7.934316105590028e-13
```

Endpoint repetition is bitwise. The full-step model error is more than two
orders of magnitude below the frozen `0.10` resolution budget, and the root
is about 126 times below the unchanged nonlinear tolerance.

This directly establishes that endpoint/node subtraction, rather than the
bordered matrix or reaction derivative, caused the prior non-descent.

## Why the formal gate still fails

The package also froze a minimum `1.5` order for the first three halvings of
the nonlinear model error. The measured error fractions are

```text
8.57e-4, 5.53e-4, 6.30e-4, 5.96e-4,
6.74e-4, 5.38e-4, 1.33e-4, 1.98e-4.
```

Their first three orders are `0.633`, `-0.188`, and `0.079`, so the frozen
order gate fails. In absolute residual units, however, all eight model errors
are only about `2.5e-13` to `1.6e-12`. They have reached the deterministic
arithmetic floor and therefore cannot display a clean quadratic sequence.

The correct interpretation is:

- the prospectively frozen order certificate is not issued;
- the exact-increment path has resolved the Newton correction sufficiently
  to satisfy the actual nonlinear root gate;
- the evidence does not select another derivative or physical repair.

The machine result's generic `exact_increment_path_derivative_repair` branch
is therefore retained as the formal fail branch but should not be executed
without an independent discrepancy above the measured floor.

## Next plan

Freeze one bounded primary-case recovery manifest before reopening the full
history ladder:

1. start again from the committed middle 20 ms state at `h=1e-7 s`;
2. run the ordinary repaired increment-primary fixed-`Q` solver, not a manual
   endpoint correction;
3. retain the unchanged `1e-10` root, exact-`Q3`, physical, reconstruction,
   Schur, reaction, constraint-work, and storage-parity gates;
4. serialize the accepted exact primitive/mapped/height history;
5. reload it and require bitwise BDF2 replay preparation;
6. record exact Jacobian/Broyden counts and fail closed;
7. authorize the other five frozen history cases only if this first case
   passes through the standard solver and restart path.

If that ordinary first case fails despite the recovered endpoint, audit only
the solver globalization/merit path. Do not alter the equations, row scales,
residual tolerance, reaction, or storage derivative.
