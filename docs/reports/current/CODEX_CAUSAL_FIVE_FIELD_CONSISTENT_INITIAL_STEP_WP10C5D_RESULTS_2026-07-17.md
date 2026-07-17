# Causal Five-Field Consistent Initial Step WP10c5d Results

Date: 2026-07-17

## Verdict

The flux-primary DAE admits index-one consistent initial data at the published
`N=16` nonstationary seed, but the bounded finite backward-Euler step does not
pass the fixed nonlinear residual gate. `N=32` is therefore not attempted.

```text
consistent-data matrix             245 / 245
descriptor conservation rows         80 / 80
maximum consistency residual          2.66e-15
algebraic tangent residual norm        9.05e-15
storage-balance residual norm          3.91e-15
N16 finite step                        FAIL
N32 finite step                        NOT ATTEMPTED
```

This separates two questions that were previously conflated:

1. the DAE has a unique local storage tangent on its algebraic manifold;
2. the current finite temporal-storage evaluation cannot yet support an
   accepted nonlinear step at the tested amplitudes.

No physical evolution, stability, tide, wind, or hot-branch conclusion is
authorized.

## Consistent Initial Data

Let `M` be the continuous storage derivative extracted from the difference
between the backward-Euler and stationary Jacobians. Let `A` be the Jacobian
of the primitive-map and face-map constraints. The initial tangent solves

```text
[ M_conservation ] ydot = [ -R_conservation ]
[ A_algebraic    ]        [         0       ].
```

At `N=16` this is a square `245 x 245` system. It is full rank, while the
storage block has the expected full row rank `80/80`. The seed already closes
all 165 algebraic residuals exactly; its nonzero conservation residual is
balanced by `ydot` rather than incorrectly forced to zero.

The largest scaled primitive rate is

```text
6373.82 per second,
```

and is dominated by the thermal field. This reflects the deliberately
nonstationary smooth seed and is not a measured physical growth rate.

## Bounded Finite Steps

The first timestep was selected so that its tangent predictor changed no
scaled primitive by more than `1e-4`. A single declared retry used `1e-3` to
reduce cancellation in the temporal state difference. Both used:

- exact primitive and face remapping at every residual evaluation;
- a centered `2e-6` reduced Jacobian;
- a direct `80 x 80` Newton solve;
- a bound-aware line search;
- unchanged `1e-8` residual acceptance.

| Target change | Timestep | Final max change | Final residual | Ledger defect | Result |
|---:|---:|---:|---:|---:|---|
| `1e-4` | `1.56892e-8 s` | `9.9968e-5` | `4.79e-6` | `3.58e-9` | fail |
| `1e-3` | `1.56892e-7 s` | `9.9680e-4` | `1.40e-6` | `2.52e-7` | fail |

For both attempts:

```text
primitive-map residual          0
face-map residual               0
Roche gate                      closed before and after
minimum scattering depth        about 1.70e4
accepted-state clipping         none
```

The Newton residual falls by roughly two orders of magnitude, then its
correction reaches about `1e-10` to `1e-11` and the bound-aware line search
cannot find a further decrease. The residual improves when the timestep is
lengthened, while the telescoping evaluation becomes cancellation sensitive.
The controlling residual is in the outermost cell in both cases:

```text
target 1e-4: rest-mass row,          -4.79e-6
target 1e-3: angular-momentum row,   -1.40e-6
```

The maximum absolute vertical temporal-storage contribution is about
`5.53e24`, while the maximum integrated source is about `2.23e21`.
This scale separation and the improvement at the larger increment are
consistent with a finite temporal-storage cancellation floor. They do not
identify a defective equation; each storage component must be compared with
its path-integrated primitive Jacobian before any implementation change.

## Gate Decision

```text
index-one consistency              PASS
N16 finite backward-Euler step     FAIL
N32 attempt                        BLOCKED
physical evolution                 BLOCKED
stationary N64/N96 roots           BLOCKED
tide                               BLOCKED
wind                               BLOCKED
```

Neither lowering the residual gate nor accepting the optimizer's generic
success flag is allowed.

## Locked Next Step

Perform one temporal-storage increment audit before another timestep:

1. identify the controlling conservation field and cell;
2. compare endpoint subtraction of each conserved storage component with a
   path integral of its primitive Jacobian over declared increments;
3. separately audit the finite responsive-height work;
4. quantify cancellation versus increment size;
5. implement a cancellation-safe finite storage difference only if the
   identity and convergence tests pass;
6. repeat the two `N=16` steps once;
7. attempt `N=32` only after `N=16` passes unchanged gates.

Do not add a stationary boundary condition, alter the Roche provider, launch
another root campaign, or introduce tide/wind.

This action was completed in WP10c5e. Endpoint cancellation was confirmed and
a converged path-integrated replacement was implemented, but the two bounded
N16 steps still failed. See
`CODEX_CAUSAL_FIVE_FIELD_TEMPORAL_STORAGE_INCREMENT_WP10C5E_RESULTS_2026-07-17.md`.

## Verification

```text
PYTHONPATH=src python3 -m pytest -q \
  tests/test_causal_inner_dae_system.py

PYTHONPATH=src python3 \
  scripts/run_causal_five_field_consistent_initial_step_wp10c5d.py
```

Machine-readable output:

```text
outputs/tables/causal_five_field_consistent_step_wp10c5d.json
```
