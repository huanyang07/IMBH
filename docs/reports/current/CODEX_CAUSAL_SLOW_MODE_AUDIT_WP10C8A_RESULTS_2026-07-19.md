# WP10c8a Selected-State Slow-Mode Audit

Date: 2026-07-19

Base commit under test:
`183afa13b21762d7fef49addc297172013981e8b`

## Decision

WP10c8a rejects the proposed global fieldwise slow-manifold split:

```text
decision
wp10c8a_slow_manifold_not_authorized

candidate retained fields       M, J, E
candidate eliminated fields     P_R, chi
selected meshes                 N64, N128
selected times                  0, 0.0375, 0.05 s
finite unstable modes           0 at every selected state
fast P_R/chi block stable       yes at every selected state
global fast/slow gap            absent at every selected state
low-mode median mesh gate       passed
```

The negative result is useful. It rules out an indiscriminate algebraic
elimination of radial momentum and causal stress over the whole radial domain.
It does not rule out a region-selective, trajectory-conditioned, or
microburst-based reduction.

## Descriptor Construction

The `15N+5` DAE Jacobian is partitioned into primitive and algebraic blocks.
The conserved and face-flux variables are eliminated by an exact scaled
algebraic Schur complement, leaving a finite `5N x 5N` primitive descriptor:

```text
M_p dp/dt + K_p dp = 0
```

The new construction preserves the increment-primary scaling and does not
differentiate the algebraic maps in time. A direct N2 finite-difference test
reconstructs both reduced matrices from the full residual and verifies the
algebraic response.

| Mesh | Descriptor rank | Condition estimate | Algebraic solve defect |
|---:|---:|---:|---:|
| N64 | `320/320` | `1.55e9-1.58e9` | `2.55e-17-2.13e-16` |
| N128 | `640/640` | `4.54e9` | `3.95e-17-8.49e-17` |

The maximum descriptor leakage into algebraic residual rows is exactly zero.

## Full Finite Spectrum

Every finite generalized eigenvalue is stable at every selected state.

| Mesh | Time | Maximum real part (`s^-1`) | Minimum real part (`s^-1`) | Unstable |
|---:|---:|---:|---:|---:|
| N64 | `0` | `-1.6305e-4` | `-44.4270` | 0 |
| N64 | `0.0375` | `-1.6326e-4` | `-42.6036` | 0 |
| N64 | `0.05` | `-1.6335e-4` | `-42.0211` | 0 |
| N128 | `0` | `-1.2621e-4` | `-81.4849` | 0 |
| N128 | `0.0375` | `-1.2641e-4` | `-77.7904` | 0 |
| N128 | `0.05` | `-1.2647e-4` | `-76.7697` | 0 |

The largest generalized eigenpair residual is `2.14e-8`, below the locked
`2e-7` gate.

This stability statement is local and frozen-coefficient. The selected states
are evolving, not stationary equilibria, and the result is not a nonlinear or
long-duration stability proof.

## Why the Global Reduction Fails

The proposed split requires every eliminated `P_R/chi` mode to be uniformly
faster than the retained `M/J/E` dynamics. The measured blocks do not have
that structure.

| Mesh | Time | Fast-block damping range | Fastest retained damping | Gap |
|---:|---:|---:|---:|---:|
| N64 | `0` | `0.0241-1406.6 s` | `0.0267 s` | `1.90e-5` |
| N64 | `0.05` | `0.0252-1405.2 s` | `0.0293 s` | `2.09e-5` |
| N128 | `0` | `0.0132-1437.6 s` | `0.0141 s` | `9.82e-6` |
| N128 | `0.05` | `0.0144-1436.2 s` | `0.0156 s` | `1.08e-5` |

There are two distinct causes:

1. outer-domain radial/stress modes remain slow, with damping times near
   `1.4e3 s`;
2. retained conserved fields contain high-wavenumber mesh modes with damping
   times near `0.015-0.03 s`.

Consequently, the labels "fast fields" and "slow fields" are not globally
valid. Setting both candidate fast equations quasi-steady in every cell would
remove slow physical content while retaining faster mesh content.

## Non-Normality

The dynamic matrices are strongly non-normal:

```text
right-eigenvector condition estimate
N64    4.10e15 to 1.35e16
N128   1.03e19 to 9.95e19

dynamic numerical abscissa
N64    +220.9 to +411.4 s^-1
N128   +431.9 to +833.0 s^-1
```

The isolated `P_R/chi` blocks also have positive numerical abscissae despite
stable eigenvalues. Stable modal eigenvalues therefore do not exclude
finite-time transient amplification. A reduction cannot be authorized from
right eigenvalues alone.

## Mesh Comparison

Hungarian matching of the 32 smallest-magnitude finite eigenvalues gives:

| Time | Median relative mismatch | Maximum relative mismatch |
|---:|---:|---:|
| `0` | `0.1860` | `0.7379` |
| `0.0375` | `0.1734` | `0.7395` |
| `0.05` | `0.1719` | `0.7420` |

The predeclared median `0.25` gate passes, so the low end of the spectrum is
not arbitrary across N64/N128. The large maximum mismatch and extreme
eigenvector conditioning prevent a stronger mode-by-mode continuum claim.

## Interpretation

WP10c8a supports four conclusions:

1. the spatially certified causal DAE has no detected local unstable
   eigenvalue through `0.05 s`;
2. the simple global split `Y=(M,J,E)`, `Z=(P_R,chi)` is invalid;
3. the actual trajectory may still occupy a much smaller active subspace than
   the full linear state space;
4. finite-time and region-aware diagnostics are required before any nonlinear
   reduced equation is implemented.

The result also explains why a single local stress time near `0.15 s` is
insufficient to justify eliminating stress everywhere. The outer domain
contains substantially slower causal-stress content.

## Evidence

Runtime artifacts remain ignored by repository policy.

```text
outputs/tables/causal_slow_mode_audit_wp10c8a.json
SHA256 d0af56ab0576e7b09b86592d4563abeb3e583fdebca88ae2dfb85a661604a7b8

outputs/tables/causal_slow_mode_audit_wp10c8a_arrays.npz
SHA256 41ce7b84482f4c1162dc870c8f5658db1d211f406d7f08bd98b91b3c69587891
```

Focused descriptor tests pass `9/9`.

## Next Authorization

WP10c8b is re-scoped to a reduction-feasibility audit. It may:

1. extend the full N64/N128 no-tide reference to the first stress-time rung
   only if the inherited spatial budget remains viable;
2. measure actual trajectory excitation, stress disequilibrium, radial-force
   imbalance, and finite-time transient amplification;
3. distinguish inner fast adjustment from slow outer radial/stress modes;
4. test a region- or mode-selective closure in operator-only form;
5. authorize a nonlinear WP10c8c reduced/full comparison only if the selected
   closure is conservative, finite-time stable, and small on the certified
   full trajectory.

WP10c8b may not implement or calibrate the rejected global three-field model.
WP10c8c, loading-time macrosteps, tide, wind, and hot/cycle work remain
conditional.
