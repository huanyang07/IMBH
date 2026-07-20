# WP10c8f Stable Observable Reduction Audit

Date: 2026-07-20

Base commit under test:
`d6fc8e3cf6b8d45803a6f0111a70726f47c60457`

This report is also the current review entry point for the superseded
standalone WP10c8d mixed-mode and WP10c8e stationary-preflight reports.
Their exact runners and reports remain reproducible from base commit
`d6fc8e3`; their binding decisions, evidence hashes, and decisive numerical
results are retained here, in `docs/PROJECT_STATUS.md`, and in the ignored
runtime JSON consumed by this audit.

## Decision

The exact global ledgers reveal a spatially converged secular evolution on
loading-time scales, but the tested stability-preserving rational Krylov
models do not provide a faithful compact realization:

```text
decision                    wp10c8f_stable_cross_mesh_observable_model_not_found
meshes                      N64, N128
anchors                     0, 0.05, 0.125 s
rational timescales         0.01-1000 s
orders                      8, 16, 24, 32, 48, 64, 96
passing orders              none
nonlinear ROM               not authorized
memory model                not authorized
new nonlinear trajectory    not run
```

This rejects the tested projection and stabilization route. It does not prove
that every observable-specific or equation-free secular model is impossible.

## Exact Secular Ledgers

The full finite-volume state was evaluated at `0`, `0.05`, `0.10`, and
`0.125 s`. For each checkpoint the audit records:

- global conserved storage;
- inner-to-outer boundary transport;
- prescribed stream input;
- geometry, cooling, vertical-work, and stress source contributions;
- net conserved-plus-responsive-height storage rate;
- the external torque or power required to make that state stationary.

At the N128 `0.125 s` state:

| Quantity | Net rate | Accumulation time | Time / `t_load` |
|---|---:|---:|---:|
| Rest mass | `-8.7453e21` | `7.9522e6 s` | `9.355` |
| Angular momentum | `+1.4952e33` | `9.3616e5 s` | `1.101` |
| Killing energy | `-4.8039e22` | `1.4439e6 s` | `1.699` |

The loading time is `8.5007e5 s`. The angular-momentum and energy ledgers
therefore identify genuine secular clocks near the requested physical
duration even though the certified truth trajectory is only `0.125 s`.

N64/N128 agreement at `0.125 s` is strong:

```text
angular-momentum net-rate difference       2.03e-5
angular-momentum timescale difference      4.86e-5
Killing-energy net-rate difference         1.86e-4
Killing-energy timescale difference        3.42e-4
rest-mass net-rate difference              7.91e-4
rest-mass timescale difference             9.49e-4
```

The no-tide state is not close to a stationary angular ledger. At `0.125 s`,
stationarity would require an external angular-momentum sink equal to about
`0.794` of the stream angular-momentum input. It would also require positive
external Killing power equal to about `0.370` of the selected energy scale.
Those missing terms are measured requirements; they are not inserted into
the equations.

The rates remain startup dependent. From `0.05` to `0.125 s`, the N64/N128
rate changes are approximately:

```text
rest mass              29.3%
angular momentum        6.55%
Killing energy           5.79%
```

This prevents a constant-rate extrapolation from being promoted directly to
a loading-time solution.

## Lyapunov Metric Audit

Every full explicit descriptor is Hurwitz, but the dense numerical Lyapunov
energy metric is not a usable certificate. Matrix balancing gives small
relative equation residuals, `2.64e-13-3.57e-13`, while the computed metric
has normalized minimum eigenvalues from about `-2.23e-16` at N64 to
`-3.21e-9` at N128.

The metric spectrum is therefore not positive definite under the unchanged
`1e-12` relative positivity contract. No eigenvalue clipping or diagonal
loading is applied. Such a repair would hide the conditioning caused by the
strongly non-normal full operator.

## Rational Krylov Construction

For every mesh and anchor, the trial space contains:

- exact representers of global mass, angular momentum, and Killing energy;
- linked stream and source-moment directions;
- trajectory, thermal, density, source-band, and stress directions;
- primal resolvents and adjoint observable resolvents at `0.01-1000 s`;
- cooling, exterior cooling, inner accretion, thickness samples and moments,
  and exact global ledgers as selected outputs.

The raw orthogonal Galerkin operators remain unstable:

```text
maximum raw real part across the ladder    +2.18 to +51.41 s^-1
```

This independently reproduces the projection-instability problem seen in
WP10c8d.

## Ledger-Safe Stabilization

The fallback stabilization solves a reduced LQR problem whose correction can
act only in the null space of the three exact global ledger coordinates.
Consequently:

```text
protected-value defects       1.76e-16 to 5.96e-16
protected-dynamics defects    2.24e-17 to 7.39e-17
stabilized maximum real part  <= 7.82e-9 s^-1
```

Stability is achieved for every tested model. At order 64 the correction
costs approximately:

```text
N64     4.03-4.66% of the raw reduced operator
N128    3.10-3.76% of the raw reduced operator
```

Order 96 lowers the correction to about `1.60-2.26%`, but exceeds the
preferred order-64 and N64 five-percent linear-cost contracts.

The correction is therefore numerically controlled at high order. That is
not enough: its input-output fidelity fails.

## Response Failure

No order passes any complete local contract. Across all six descriptors, the
best maximum trained response error is still:

```text
1.000236
```

against the `0.10` gate. Long-timescale thickness transfer errors remain near
one even where the certified short-time thickness response is improved.
Held-out thermal, density, and velocity directions fail more strongly.

The order-64/96 cross-mesh transfer excess remains approximately:

```text
0.320-0.323
```

against `0.25`.

The failure has a useful interpretation. The stabilization can preserve the
three exact ledger derivatives and move every reduced pole into the stable
half-plane, but even a `1.6-4.7%` operator correction changes the delicate
non-normal transfer behavior enough to lose the scientific response. A
stable reduced pole set is not a substitute for a faithful causal transfer
map.

## Consequences

The following routes are now closed under the tested contracts:

1. fieldwise algebraic elimination;
2. region-selective algebraic elimination;
3. finite-horizon BPOD at orders up to its resolved rank;
4. orthogonal rational Krylov followed by ledger-null stabilization;
5. stationary continuation from the source-compatible seeds.

No unrestricted memory fit is authorized. The truth model covers only
`0.125 s`, and the failed compact projections do not provide a stable
unresolved complement from which to extrapolate a long memory kernel.

The positive result is narrower and important:

> Global mass, angular momentum, and Killing energy are exact,
> mesh-consistent secular coordinates, but they do not yet form a closed
> predictive state.

## Next Authorization

One ledger-driven equation-free preflight is authorized. It must use existing
N64/N128 checkpoints before launching new long evolution:

1. compare exact ledger rates and secant rates at common times;
2. test one- and two-window projective predictions only within the existing
   `0.125 s` truth interval;
3. test whether perturbations with identical global `M/J/E` produce
   materially different cooling, thickness, or inner-accretion responses;
4. reject a ledger-only closure if that non-identifiability exceeds the
   observable gates;
5. if needed, add only measured observable coordinates such as luminosity,
   inner accretion, or a few thickness moments;
6. do not authorize a nonlinear macrostep until a factor-two projection
   passes on both meshes and on held-out perturbations.

No loading-time evolution, memory ROM, N256 trajectory, tide, wind, or
hot/cycle claim is authorized by this result.

## Evidence

Runtime artifacts remain ignored by repository policy.

```text
outputs/tables/causal_stable_observable_reduction_audit_wp10c8f.json
SHA256 0302741543cf4fcbd72fd4f4816625341efd67a21a9ce195faa8f57038145e83

outputs/tables/causal_stable_observable_reduction_audit_wp10c8f_arrays.npz
SHA256 90086be02c697d27f4ee155a19c35f212e121e7dc363585218dc2b1fbb71b142
```
