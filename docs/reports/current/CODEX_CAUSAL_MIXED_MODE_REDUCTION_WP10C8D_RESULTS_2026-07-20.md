# WP10c8d Conservation-Constrained Mixed-Mode Audit

Date: 2026-07-20

Base commit under test:
`4247696c2c65039fc4c08d6aaca7cbace8be6636`

## Decision

The tested finite-horizon balanced-POD realization does not provide a stable,
cross-mesh compact Markovian model:

```text
decision                         wp10c8d_compact_cross_mesh_markovian_basis_not_found
selected meshes                  N64, N128
selected times                   0, 0.05, 0.125 s
requested orders                 8, 16, 32, 64, 96, 128
numerically resolved order       39-41 including three exact ledgers
passing compact orders           none
nonlinear ROM                    not authorized
```

This is a bounded rejection of this realization, not a proof that every
mixed-mode or memory model is impossible.

## Exact Descriptor And Input Contract

The existing algebraic Schur complement supplies the finite primitive
descriptor

```text
E dx/dt + K x = R u
```

at all six selected states. The implementation maps it to an explicit linear
system with a joint solve for `-K` and `R`. The solve defects are
`1.76e-16-2.66e-16`.

| Mesh | Rank | Condition estimate | Descriptor wall time |
|---:|---:|---:|---:|
| N64 | `320/320` | `1.51e9-1.58e9` | `22.4-23.1 s` |
| N128 | `640/640` | `4.54e9-4.55e9` | `44.7-45.6 s` |

The primary input is one linked physical stream-amplitude variation. Three
fixed-mass variations of injected radial momentum, angular momentum, and
Killing energy are used only as robustness inputs. A finite-difference test
against the full source-fed residual verifies the input sign and scaling.

The basis construction also includes the actual trajectory secant and
thermal, surface-density, source-band, and stress-adjustment directions.
Outputs include total and exterior cooling, inner accretion, selected
thickness samples and moments, and integrated mass, angular momentum, and
Killing energy.

## Exact Conservation Coordinates

The three integrated ledgers are protected coordinates. The trial basis
reconstructs their values exactly, and the first three Petrov test rows are
the original ledger operators. Their reduced derivatives are therefore the
original full-system ledger derivatives evaluated on the reconstructed
state.

The protected value and dynamics defects remain at numerical precision.
No conservation law is inferred from a fitted mode.

## Finite-Horizon Compressibility

The snapshot Hankel maps have only `36-38` resolvable dynamic directions at
the dimension-aware precision floor. With the three protected coordinates,
the largest meaningful orders are:

| Mesh | `t=0` | `t=0.05 s` | `t=0.125 s` |
|---:|---:|---:|---:|
| N64 | 39 | 39 | 39 |
| N128 | 41 | 41 | 40 |

Orders 64, 96, and 128 are therefore recorded as tested but numerically
unresolved; no artificial modes are manufactured below the Hankel precision
floor.

The low-dimensional subspaces show some mesh agreement. The N64/N128
95th-percentile principal angles are about `3 degrees` at order 8 and
`22-27 degrees` at order 16. They rise to `63-66 degrees` at order 32.

## Why The Markovian Models Fail

Every available reduced operator is unstable even though every full
descriptor mode is stable:

```text
order 8 maximum real part       +7.80 to +8.49 s^-1
order 16 maximum real part      +9.36 to +10.00 s^-1
order 32 maximum real part      +3.75 to +53.80 s^-1
```

Order 16 reproduces the trained scalar outputs well, with maximum relative
errors of roughly `0.006-0.012`, and reproduces trained thickness responses
to about `8.7e-5-1.1e-3`. That apparent fit is not sufficient:

- the reduced dynamics are unstable;
- held-out thickness response errors are about `0.225-0.544`;
- held-out full-state errors are large;
- order 32 becomes less robust and less mesh aligned despite fitting the
  training directions more closely.

This is the expected failure mode of an over-compressed non-normal system:
small trained transfer error does not imply stable off-training dynamics.

## Memory-Necessity Diagnostic

No nonlinear memory model was fitted. The exact oblique unresolved feedback
was measured for the order-32 projection at the certified `0.125 s` state.

| Mesh | Unresolved maximum real part | Measured feedback growth |
|---:|---:|---:|
| N64 | `+4.170 s^-1` | `34.82x` by `1 s` |
| N128 | `+41.725 s^-1` | `18.58x` by `0.1 s` |

Longer exponential actions were deliberately skipped once
`Re(lambda) t > 20`. Reporting those horizons as unsafe is preferable to
overflowing an unstable projected complement and calling it a long-memory
measurement.

The result says that omitted feedback is dynamically important for this
projection. It does not yet say whether a different stable realization could
represent that feedback with a small number of auxiliary states.

## Online-Cost Gate

The linear reduced solves would be inexpensive, but nonlinear operator
compression has not been demonstrated. A small coordinate vector that still
evaluates the complete N128 residual every macrostep does not solve the
loading-time problem.

Therefore:

- no nonlinear ROM is implemented;
- no hyper-reduction is authorized;
- exact structural ledger preservation under hyper-reduction remains open;
- no loading-time speedup is claimed.

## Next Authorization

The broad conservation-constrained BPOD route is closed at the tested
orders and horizon. A next reduction package must be narrower and
stability-preserving. It may:

1. test an observable-specific rational-Krylov, IRKA, or stable transfer
   realization while retaining the exact ledger coordinates;
2. measure held-out transfer and transient-gain errors at N64 and N128;
3. add auxiliary memory states only if their poles are stable and
   cross-mesh consistent;
4. estimate true online nonlinear cost before any nonlinear implementation;
5. stop if the scientific outputs cannot be reproduced without the full
   state.

In parallel, exactly one bounded stationary ledger/rank/root-predictor
preflight is authorized. Neither path may launch loading-time macrosteps,
tide, wind, or hot/cycle physics.

## Evidence

Runtime artifacts remain ignored by repository policy.

```text
outputs/tables/causal_mixed_mode_reduction_audit_wp10c8d.json
SHA256 0abda6624949aaf08258f72299e9ea8401221986f7575a6af046e74aa36351c4

outputs/tables/causal_mixed_mode_reduction_audit_wp10c8d_arrays.npz
SHA256 fa1fdb22094a830ff901b6a58aa22f524e76bd8d45b4ba038ffc1bf7bf6e3460
```

The mixed-reduction and descriptor-focused test set passes `18/18`. The full
repository suite passes `573` tests plus `4` subtests.
