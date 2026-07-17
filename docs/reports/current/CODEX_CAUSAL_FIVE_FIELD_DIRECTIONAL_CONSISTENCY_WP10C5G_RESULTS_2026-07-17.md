# Causal Five-Field Directional Consistency WP10c5g Results

Date: 2026-07-17

## Verdict

The frozen N16 backward-Euler failure is uniquely localized to the
path-integrated conserved-storage increment. Face-flux differences,
geometric/thermal sources, and responsive-height work all follow the Newton
direction to much better than the unchanged `1e-8` residual gate.

One authorized conditioning repair replaced the path derivative's
endpoint-dependent magnitude/direction normalization with an algebraically
equivalent fixed-coordinate Jacobian-vector product. Focused tests pass and
the path quadrature remains converged, but the single N16 retry still stops at
`1.42473e-6`.

```text
pre-repair failing component       path conserved storage only
pre-repair storage defect          3.34921e-6
post-repair N16 residual           1.42473e-6
post-repair storage defect         1.42457e-6
N16 gate                           FAILED
N32                                NOT ATTEMPTED
```

The bounded conclusion is that finite-difference path integration is not a
Newton-consistent production representation of the tiny conserved increment
at this timestep. Physical evolution, roots, tide, and wind remain blocked.

## Frozen Component Audit

The first audit uses the final path-integrated N16 candidate and the direct
Newton correction from the same centered `2e-6` Jacobian:

```text
correction L2 norm                 5.44753e-12
maximum correction                3.23837e-12
linearized residual               2.12e-22
nonlinear residual after update    3.34921e-6
```

The conservation residual is decomposed before scaling into:

```text
face-flux divergence
- integrated geometric/thermal sources
+ path-integrated conserved storage / Delta t
+ responsive-height Killing work / Delta t
```

Long-double compensated differences are used only to diagnose changes between
two already evaluated double-precision component arrays. They do not alter the
production residual.

The actual component changes versus the fourth-order directional prediction
are:

| Component | Maximum scaled defect |
|---|---:|
| Face-flux difference | `2.51e-15` |
| Geometric/thermal sources | `5.13e-15` |
| Path conserved storage | `3.34921e-6` |
| Responsive-height work | `1.91e-13` |

Second- and fourth-order directional predictions for path storage agree to
`2.88e-13`. The component sum reconstructs the residual change to
`2.76e-16`. The defect is therefore not a failure of the decomposition,
linear solve, flux cancellation, or source ledger.

## Authorized Storage Repair

The original path implementation represented the primitive endpoint increment
as a norm and normalized direction. Both were recomputed after every Newton
correction. At corrections near `1e-12`, floating-point changes in that
normalization could exceed the physical storage change.

The repair evaluates the same straight-path integral as

```text
dU/dlambda = sum_a (dU/dq_a) Delta q_a
```

using fourth-order centered derivatives in fixed scaled primitive coordinates.
The endpoint increment enters the Jacobian-vector product directly. No
equation, path, quadrature order, tolerance, or physical closure changes.

Two focused tests verify:

- the analytic rest-mass response to a `1e-12` endpoint correction;
- the multi-field tiny endpoint change against an independently sampled
  directional response.

The path order and step audits remain below their `5e-9` gate.

## Single N16 Retry

The unchanged N16 retry uses:

```text
temporal scheme                    path integrated
target scaled primitive change     1e-3
timestep                           1.56892e-7 s
residual tolerance                 1e-8
finite-difference step             2e-6
```

It stops after two Newton iterations:

```text
maximum scaled residual            1.4247307765e-6
controlling row                    cell 15 angular momentum
solver message                     bound-aware line search failed
reduced condition estimate         1.03005e10
```

The post-repair directional audit again identifies only conserved storage:

| Component | Actual/fourth-order defect | Second/fourth defect |
|---|---:|---:|
| Face-flux difference | `2.15e-15` | `7.97e-20` |
| Geometric/thermal sources | `7.46e-15` | `1.63e-19` |
| Path conserved storage | `1.42457e-6` | `1.74e-11` |
| Responsive-height work | `2.09e-13` | `4.73e-18` |

The base and corrected component sums reproduce the production residual below
`2.0e-16`; the actual component increment reproduces the residual change below
`1.91e-16`. The storage evaluation, not the ledger assembly, remains the
limitation.

## Classification

WP10c5g certifies:

- a unique component-level diagnosis;
- exact reconstruction of the residual from its four physical blocks;
- directional consistency of fluxes, sources, and responsive-height work;
- one locally smoother path-storage representation.

WP10c5g rejects:

- path-integrated finite-difference conserved storage as the N16 evolution
  unlock;
- another storage quadrature, step-size, linear-solver, or tolerance scan;
- N32, stationary roots, tide, wind, or long evolution.

This is a numerical negative result at a deliberately nonstationary seed. It
does not establish a physical instability or nonexistence.

## Recommended Next Architecture

If numerical work continues, use an increment-primary form of the complete
flux-primary backward-Euler DAE:

```text
unknowns = (Delta U_cell, Delta p_cell, Delta F_face)
```

The conservation storage term is then the primary `Delta U_cell / Delta t`,
not an endpoint subtraction or finite-difference path integral. Absolute
primitive and face maps remain algebraic rows and are not amplified by
`1 / Delta t`. The responsive-height one-form may retain its current path
evaluation because it passed the directional audit.

This is a reparameterization of the same `15N+5` DAE, not a new physical
model. It should receive one N16 count/rank and tiny-step gate before any N32
attempt. If that gate fails, close this seed-based startup and construct a
stationary or physically relaxed causal initial state instead.

## Verification

```text
PYTHONPATH=src python3 \
  scripts/run_causal_five_field_consistent_initial_step_wp10c5d.py \
  --temporal-storage-scheme path_integrated \
  --directional-consistency-audit \
  --output \
  outputs/tables/causal_five_field_directional_consistency_wp10c5g_repaired.json
```

The machine-readable pre-repair and post-repair outputs are:

```text
outputs/tables/causal_five_field_directional_consistency_wp10c5g.json
outputs/tables/causal_five_field_directional_consistency_wp10c5g_repaired.json
```

Repository verification:

```text
475 tests passed
4 subtests passed
repository hygiene passed for 613 tracked files
Python compile checks passed
git diff --check passed
```
