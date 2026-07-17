# Horizon-penetrating Valencia WP10b Results

**Date:** 2026-07-17
**Branch:** `codex/repository-cleanup-after-1e7438e`
**Scope:** architecture selection, local conservative flux, full rotating
characteristics, stationary rank, and exact DAE count. No stationary disk,
global evolution, tide, wind, or physical source continuation was run.

## Verdict

WP10b selects a one-domain ingoing-Kerr-Schild Schwarzschild Valencia column
system as the next causal architecture.

The local prototype passes all authorized mathematical gates:

```text
representative analytic/numerical eigenvalue defect   9.71e-11
regular stationary flux rank                          4
acoustic critical stationary flux rank                3
critical smallest scaled singular value               5.42e-12
maximum characteristic at 1.9 rg                     -0.0392 c
maximum characteristic at 1.5 rg                     -0.1548 c
```

The last two scans cover 171 physical velocity/sound-speed combinations at
each radius. All characteristics leave through the inner boundary.

This is an architecture result, not a production solution.

## Why The PW Continuation Cannot Be Reused

Adding the previously omitted transverse velocity gives:

| Radius (`rg`) | `v_R/c` | `v_phi/c` | Total `v/c` | Radial incoming modes |
|---:|---:|---:|---:|---:|
| 4.5 | `-8.54e-6` | `0.848` | `0.848` | 1 |
| 3.0 | `-3.48e-4` | `1.710` | `1.710` | 1 |
| 2.1 | `-0.0125` | `9.618` | `9.618` | 1 |
| 2.01 | `-0.0709` | `37.27` | `37.27` | 1 |
| 2.001 | `-0.263` | `117.12` | `117.12` | 1 |
| 2.0001 | `-0.862` | `357.06` | `357.06` | 0 |

The old radial-only crossing occurs after the full PW state has become
superluminal. There is no full-state excision candidate on that continuation.

## Horizon-penetrating Geometry

The ingoing light speed is `-c` at all audited radii. The outgoing light
speed changes continuously:

| Radius (`rg`) | Lapse | Shift `beta^R/c` | Outgoing light speed |
|---:|---:|---:|---:|
| 4.5 | `0.8321` | `0.3077` | `0.3846 c` |
| 2.1 | `0.7157` | `0.4878` | `0.02439 c` |
| 2.0 | `0.7071` | `0.5000` | `0` |
| 1.9 | `0.6980` | `0.5128` | `-0.02564 c` |
| 1.5 | `0.6547` | `0.5714` | `-0.14286 c` |

An excision inside `2 rg` therefore has a geometry-controlled zero-incoming
contract. It does not depend on the low-rate radial velocity remaining
supersonic under mesh refinement.

## Conservative Flux Check

The prototype maps a rotating column to

```text
U = (D, S_R, S_phi, tau)
```

and uses the standard Valencia radial flux in ingoing Kerr-Schild geometry.
For a representative rotating state at `4.5 rg`, the analytic speeds are

```text
-0.54750063, -0.44615385, -0.44615385, -0.33822379
```

and numerical differentiation of the conservative flux gives

```text
-0.54750063, -0.44615385, -0.44615385, -0.33822379.
```

The maximum difference is `9.71e-11`. This check includes transverse
rotation; it is not the one-dimensional velocity-addition formula used by
WP10a.

## Stationary And Time-dependent Consistency

At a constructed acoustic critical point, the outgoing characteristic is
zero and the stationary conservative flux has rank three. Away from the
critical point it has rank four.

The production stationary baseline must therefore be solved as a zero of the
same finite-volume residual used in evolution. No independent PW slim-disk
critical condition will be attached.

## Exact DAE Count

For `N` cells:

| Block | Unknowns/rows |
|---|---:|
| Conserved cell state | `4N` |
| Primitive cell state | `4N` |
| All face fluxes | `4(N+1)` |
| **Total unknowns** | **`12N+4`** |
| Backward-Euler conservation | `4N` |
| Primitive map | `4N` |
| Interior face fluxes | `4(N-1)` |
| Inner one-sided flux | `4` |
| Outer provider flux | `4` |
| **Total rows** | **`12N+4`** |

The physical inner-boundary rank is zero. The four inner face rows evaluate
the one-sided flux and do not supply exterior data.

## What Is Not Yet Implemented

The prototype intentionally omits:

1. gas+radiation column primitive recovery in the Valencia chart;
2. discretized Kerr-Schild geometric source terms;
3. radiative cooling and vertical work in relativistic energy variables;
4. relativistic alpha stress, torque work, and its characteristic audit;
5. exact stream source moments in the new conserved variables;
6. the Hill/Roche boundary and Jacobi ledger in the new energy convention;
7. a stationary disk root or implicit evolution step.

No current checkpoint may be relabeled as a Valencia initial state.

## Locked Next Plan

### WP10c1: thermodynamic primitive map

Implement `P -> U -> P` for the shared gas+radiation column EOS. Require:

```text
positive Sigma and T
v_hat_R^2 + v_hat_phi^2 < c^2
0 < a^2 < c^2
round-trip error <= 1e-10
analytic/numerical eigensystem defect <= 1e-7
```

### WP10c2: source-free geometric finite volume

Add proper cell/face geometry and covariant Schwarzschild sources. Pass
constant-state local tests, geodesic/circular-orbit controls, exact global
telescoping, and convergence without floors or clipping.

### WP10c3: stress and thermal ledger

Transform the common stress and paired torque work into the new variables.
Audit the full flux spectrum for causality. Then add radiation and vertical
work without double counting.

### WP10c4: full-domain physical contracts

Migrate exact stream moments and the Hill/Roche provider. Recover the
Newtonian weak-field benchmark before constructing a new low-throughput
stationary state.

### WP10c5: first production gate

Only then solve N64/N96 stationary roots, place the inner face inside `2 rg`,
verify zero incoming modes, conservatively map the root, and attempt one tiny
implicit step.

Tide, wind, and long loading remain blocked.

## Verification

```text
focused causal/Valencia/EOS tests   22 passed
complete repository suite           394 passed, 4 subtests passed
```

Machine-readable evidence:

```text
outputs/tables/causal_inner_valencia_wp10b.json
```

Reproduction:

```bash
PYTHONPATH=src python3 scripts/run_global_inner_boundary_architecture_gate.py
PYTHONPATH=src python3 scripts/run_causal_inner_thermodynamics_wp10a.py
PYTHONPATH=src python3 scripts/run_causal_inner_valencia_wp10b.py
```
