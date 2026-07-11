# Coupled Inner/Outer Rank Prototype Results

## Scope

This work package implements the first simultaneous no-wind solve of the
inner transonic eigenproblem and outer common-stress reservoir at the exact
canonical interface radius

```text
R_I = 40.04153642035986 rg.
```

It changes no physical closure. The stream, ideal tidal-wall control,
Paczynski-Wiita potential, vertical thermodynamics, total-pressure alpha
stress, and corrected total-energy identity are inherited from `1146e67`.
The reservoir edge is the corrected finite-minidisk value `335 rg`. Earlier
versions of this runner accidentally inherited the transonic benchmark's
`10000 rg` numerical buffer and are superseded for physical interpretation.

## Boundary Rank

For `N_i` inner nodes and `N_o` outer cells, the unknown and residual count is

```text
inner logu/logT plus R_son/lambda0     2 N_i + 2
outer logSigma/logT/logOmega           3 N_o
signed interface J and F_E                  2
total                               2 N_i + 3 N_o + 4.
```

The residual contains `2 N_i` inner interval/sonic rows, `3 N_o` outer
stress/radial/energy rows, two flux-extraction rows, and two primitive
continuity rows. The hard primitive conditions are `log Sigma` and `log T`;
`Pi`, `Omega`, `H`, and `u` are audits.

The old isolated inner solution's outer boundary residual cannot anchor a
truncated `40 rg` problem. The implemented initialization instead uses the
finite-domain `Sigma,T` state at `40.0415 rg`. Coupling then uses the
rank-preserving target-jump homotopy

```text
C(mu) - (1-mu) C(0) = 0,
```

which drives the measured one-way primitive jump to zero. A direct blend of
the canonical-anchor rows with continuity was tested and rejected because
the inner-state derivatives have opposite signs and the continuation became
artificially singular near `mu=0.06` on the pilot mesh.

## Numerical Implementation

The inner boundary-free residual has a block-local Jacobian. The two anchor
derivatives are exact. Coupled Newton steps use a graph-colored sparse finite
difference Jacobian and bounded backtracking. On the tiny verification case,
the colored Jacobian has no omitted dense-Jacobian entries and differs by
about `2e-5` in Frobenius norm, consistent with its one-sided difference.

No projection, smoothing, clipping, residual reweighting, or tolerance
relaxation is used.

## Results

| `N_i/N_o` | Max residual | `Lrad/LEdd` | Max `H/R` | Jacobian condition |
|---:|---:|---:|---:|---:|
| 96/64 | `1.68e-8` | `1.34822` | `0.31023` | `2.47e5` |

The `96/64` system has 388 unknowns and 388 residuals. At both `mu=0` and
`mu=1`:

```text
full Jacobian rank at 1e-8,1e-10,1e-12: 388
pre-boundary nullity:                         2
interface response rank:                     2
selected sonic-pair rank:                    2
```

At full coupling:

```text
ln Sigma continuity mismatch       1.78e-15
ln T continuity mismatch           1.78e-15
ln Pi audit mismatch               4.44e-5
relative Omega audit mismatch     -9.66e-5
ln H audit mismatch                4.45e-5
ln u audit mismatch                3.55e-15
lambda0                            3.81876001
R_son                              4.36112038 rg
```

The unused sonic compatibility residual is `1.36e-12`. The composite remains
warm/thick rather than collapsing to the open cool state.

Removing the outermost radial row in a rank-only audit exposes an endpoint-
localized null vector, as expected because that one-sided row is the current
numerical endpoint closure. The weakest singular vector of the complete
Jacobian is instead inner-transonic, not outer-endpoint localized. This
endpoint dependence remains a required sensitivity check during mesh and
interface continuation.

## Classification

```text
numerical_status = SUPPORTED BUT NOT FULLY CERTIFIED
physical_status  = DIAGNOSTIC ONLY
```

This establishes that the remaining density mismatch in `1146e67` can be
removed by a square, full-rank coupled eigenproblem while preserving the
conserved fluxes and warm/thick response. It does not yet establish mesh
certification, interface-position invariance, a physical tidal torque and
power, stability, time evolution, or wind.

## Locked Next Gate

1. Prolongate this full root, not the frozen-inner seed, to the N128/N256 mesh
   gate.
2. Continue the certified root to `35,45,50 rg` as an external parameter.
3. Require luminosity spread below 1%, warm/thick metric spread below 2%, and
   audited primitive mismatch below 1%.
4. If either continuation is not mesh supported, stop splice development and
   move to the one-domain signed conservative transonic fallback.
5. Add physical tidal torque/power, stability, time evolution, and wind only
   after those gates pass.
