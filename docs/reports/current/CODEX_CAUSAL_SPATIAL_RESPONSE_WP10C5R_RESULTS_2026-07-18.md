# Causal Spatial-Response WP10c5r Results

Date: 2026-07-18

## Verdict

The bounded no-evolution spatial audit passes. The N16/N32 duration mismatch
is ordinary coarse-grid truncation from the declared first-order Rusanov face
operator, not a temporal artifact, reconstruction artifact, inconsistent
source, or broken spatial stencil.

```text
term-resolved N16 tangent decomposition                 PASSED
term-resolved N32 tangent decomposition                 PASSED
conservative fine-to-coarse comparison                  PASSED
shared-radius linear/PCHIP comparison                   PASSED
central-flux manufactured order                         PASSED
Rusanov-dissipation manufactured order                  PASSED
full face-transport manufactured order                  PASSED
all production source-term manufactured checks          PASSED
operator correction                                     NOT JUSTIFIED
N64 physical evolution in this package                  NOT EXECUTED
one bounded N64 confirmation in the next package        AUTHORIZED
long evolution, tide, wind, stability, hot/cycle work  NOT AUTHORIZED
```

No physical equation, closure, residual gate, or production flux was changed.
The implementation only exposes an exact diagnostic split of the existing
Rusanov flux and an exact source-component ledger.

## Semidiscrete Decomposition

The complete consistency matrix is used for the full stationary forcing and
for each isolated forcing term. The component tangents therefore obey the
same descriptor and algebraic constraints as the production initial tangent.

| Quantity | N16 | N32 |
|---|---:|---:|
| Residual reconstruction defect | `2.02e-16` | `2.21e-16` |
| Tangent reconstruction defect | `2.62e-11` | `5.15e-12` |
| Maximum full `abs[d ln(H/R)/dt]` | `37.8411 s^-1` | `13.2690 s^-1` |

The largest component amplitudes show a strong physical cancellation:

| Component | N16 maximum | N32 maximum |
|---|---:|---:|
| Face transport | `125.762 s^-1` | `120.113 s^-1` |
| Perfect-fluid geometry | `117.362 s^-1` | `118.140 s^-1` |
| Vertical work | `0.632 s^-1` | `0.856 s^-1` |
| Radiative cooling | `0.368 s^-1` | `0.391 s^-1` |
| Stress geometry | `0.118 s^-1` | `0.131 s^-1` |
| Stream | `1.69e-4 s^-1` | `2.10e-4 s^-1` |
| Stress relaxation | `1.35e-4 s^-1` | `1.99e-4 s^-1` |

Face transport and the perfect-fluid geometric source are individually about
`120 s^-1` and nearly cancel. A first-order face error can therefore be much
larger than the net physical response on a coarse mesh.

## Mesh Localization

Exact nested finite-volume restriction gives the following N16 versus
restricted-N32 `d ln(H/R)/dt` differences:

| Rank | Component | Maximum difference | Radius |
|---:|---|---:|---:|
| 1 | Face transport | `24.0482 s^-1` | `55.5662 rg` |
| 2 | Perfect-fluid geometry | `1.58902 s^-1` | `55.5662 rg` |
| 3 | Vertical work | `0.241641 s^-1` | `2.11935 rg` |
| 4 | Radiative cooling | `2.69319e-3 s^-1` | `2.11935 rg` |
| 5 | Stress geometry | `1.13135e-3 s^-1` | `2.11935 rg` |
| 6 | Stream | `4.70525e-5 s^-1` | `205.236 rg` |
| 7 | Stress relaxation | `4.59024e-5 s^-1` | `2.11935 rg` |

The full tangent difference is `25.6373 s^-1` at `55.5662 rg`. It survives
exact conservative restriction:

```text
fine-to-coarse cell-average maximum   25.6373 s^-1
log-linear shared-radius maximum      25.8410 s^-1
PCHIP shared-radius maximum           25.8111 s^-1
N16 reconstruction spread              2.8903 s^-1
N32 reconstruction spread              0.4499 s^-1
```

The discrepancy is therefore in the coarse semidiscrete operator, not in the
choice of profile reconstruction. It is broad in the same middle-disk region
that controlled the bounded-duration result, not localized to the compact
stream or Roche boundary.

## Manufactured Spatial Order

The production central-plus-Rusanov face flux was evaluated, without time
evolution, on exact nested N16, N32, N64, and N128 samples of the same
fixed-anchor C2 continuum profile. Shared physical faces and integrated cell
balances were compared separately.

The minimum observed orders are:

| Spatial term | Minimum observed order | Required |
|---|---:|---:|
| Central face average | `1.9961` | `>=1.5` |
| Rusanov dissipation | `1.1399` | `>=0.75` |
| Total face transport | `1.1058` | `>=0.75` |
| Local geometry/cooling sources | `1.9917` | `>=1.5` |
| Derivative work/stress sources | `1.9837` | `>=0.75` |

The central shared-face errors converge at second order. The Rusanov
dissipation and total shared-face errors converge at first order, as declared
for the piecewise-constant production flux. Integrated source checks give:

```text
perfect-fluid geometry orders   2.0096, 1.9965
stress geometry orders          2.0054, 2.0051
radiative cooling orders        1.9917, 1.9981
vertical-work orders            2.0132, 2.0001
stress-relaxation orders        1.9837, 2.0012
maximum exact-stream defect     2.2522e-16
```

N64 and N128 here are operator evaluations only. No N64/N128 nonlinear step
or physical evolution was launched.

## Interpretation

WP10c5r closes the ambiguity left by WP10c5o-q:

1. The fixed analytic initial datum is common across meshes.
2. Timestep history is negligible compared with the duration discrepancy.
3. Conservative restriction and two reconstructions agree on the same broad
   spatial mismatch.
4. Face transport supplies about `94%` of the two leading absolute component
   differences (`24.05` versus `1.59 s^-1`).
5. The production face operator and every production source term show their
   expected convergence order.

Changing the Rusanov operator now would be an unrequested numerical-method
upgrade, not a correction of an identified defect. The current first-order
method remains the production baseline.

This result does not certify a physical relaxation, hot state, instability,
or limit cycle. It only classifies why N16/N32 disagree after the bounded
duration.

## Locked Next Work: WP10c5s

Run one bounded N64 confirmation on the same fixed physical datum:

1. Reuse the certified N32 checkpoint and regenerate N64 from the analytic
   profile. Do not remap an evolved N32 state.
2. Run the existing short startup at N64 and compare N32/N64 at the exact
   common short time. The unchanged `5e-3` response gate must pass.
3. Only if the short gate passes, extend N64 to the existing bounded target
   time with the same maximum-timestep temporal-parity control.
4. Compare N16/N32 and N32/N64 duration errors and report the observed
   contraction order.
5. If the N32/N64 duration response passes `5e-3`, certify the mesh gate.
6. If it remains above `5e-3` but contracts with order at least `0.75` while
   all causal, rank, optical, Roche, nonlinear, and ledger gates pass,
   authorize one N128 confirmation only.
7. Otherwise stop and reassess the first-order production method.

Do not add a higher-order reconstruction, change physics, relax tolerances, or
launch N96, long evolution, distributed tide, wind, stability, hot-state, or
cycle searches in WP10c5s.

## Verification

```text
focused causal DAE tests        26 passed
full repository suite           489 passed, 4 subtests passed
repository hygiene              passed for 628 tracked files
Python compilation              passed
git diff --check                passed
```

## Reproduction

```text
PYTHONPATH=src python3 \
  scripts/run_causal_five_field_consistent_initial_step_wp10c5d.py \
  --increment-primary-mesh-common-spatial-response-audit
```

Machine-readable output:

```text
outputs/tables/causal_five_field_mesh_common_spatial_response_wp10c5r.json
```
