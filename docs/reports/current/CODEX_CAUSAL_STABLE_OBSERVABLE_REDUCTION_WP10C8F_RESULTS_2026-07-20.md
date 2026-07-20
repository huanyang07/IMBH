# WP10c8f-h Stable Observable and Equation-Free Closure Audits

Date: 2026-07-20

Base commit under test:
`d6fc8e3cf6b8d45803a6f0111a70726f47c60457`

This report is also the current review entry point for the superseded
standalone WP10c8d mixed-mode and WP10c8e stationary-preflight reports.
Their exact runners and reports remain reproducible from base commit
`d6fc8e3`; their binding decisions, evidence hashes, and decisive numerical
results are retained here, in `docs/PROJECT_STATUS.md`, and in the ignored
runtime JSON consumed by this audit.

It is also the current entry point for the subsequent WP10c8g global-ledger
and WP10c8h conservative-shell closure preflights, run from base commit
`d89663531dbbce97be01d583e364bc3585448e76`. Their runners remain tracked
independently, while their binding decisions and evidence are consolidated
below to respect the repository artifact limit.

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

## WP10c8g Global-Ledger Closure Preflight

The authorized equation-free preflight used only the existing N64/N128
production and temporal-control checkpoints at `0.05`, `0.075`, `0.10`, and
`0.125 s`. It tested

\[
C_0=(M,J,E)
\]

and an augmented candidate

\[
C_1=(M,J,E,L_{>6r_g},\dot M_{\rm inner},H_1,H_2,H_3).
\]

The ledger-only state appears excellent if judged only along the startup
trajectory: factor-two Euler and AB2 checkpoint errors remain below
`6.7e-7` of a scientific gate, and exact effective-storage rates agree with
checkpoint secants to `9.53e-3`, below the `5e-2` integrity gate.

That apparent success is not closure. At the common `0.125 s` descriptor
state, physically scaled perturbations were projected into the exact
candidate constraint null spaces. The constraint defects remain below
`1.1e-14`, but equal-ledger states produce:

| Test | N64 | N128 | Controlling direction |
|---|---:|---:|---|
| `C0` immediate held-observable response | `19.733` | `19.753` | thermal, `6-60 rg` |
| `C0` projected `0.025 s` response | `13.046` | `13.230` | radial velocity |
| `C1` immediate held-observable response | `2.486` | `2.390` | density, `6-60 rg` |
| `C1` projected held response | `13.046` | `13.230` | radial velocity |
| `C1` projected constrained-coordinate change | `5.960` | `6.227` | radial velocity |

The augmented state also fails direct checkpoint extrapolation: its AB2
endpoint error is `2.227/2.229` gates on N64/N128, controlled by inner
accretion. Production/control agreement is excellent, so controller history
does not explain the failure.

The binding decision is:

```text
decision              wp10c8g_global_equation_free_closure_not_identifiable
nonlinear burst       not run
nonlinear macrostep   not authorized
next authorization    five-shell conservative closure preflight
```

## WP10c8h Conservative-Shell Closure Preflight

The authorized fallback retained exact shell-integrated M/J/E on
mesh-coincident N64/N128 physical regions. The five-shell state has 15
coordinates with edges

```text
1.8, 6.1270, 60.2942, 205.2361, 284.5211, 335.0 rg
```

and the single predeclared eight-shell refinement has 24 coordinates with
edges

```text
1.8, 2.9381, 6.1270, 19.2204, 60.2942,
125.7368, 205.2361, 284.5211, 335.0 rg
```

Both layouts fail the checkpoint projection contract:

| Layout/mesh | Euler to `0.10 s` | Euler to `0.125 s` | AB2 to `0.125 s` |
|---|---:|---:|---:|
| Five-shell N64 | `1.434` | `1.257` | `0.5693` |
| Five-shell N128 | `1.437` | `1.258` | `0.5688` |
| Eight-shell N64 | `3.941` | `3.641` | `1.7518` |
| Eight-shell N128 | `3.942` | `3.641` | `1.7518` |

Acceptance required every normalized error to remain below `0.25`.
Refinement therefore worsens rather than repairs the bounded projective
prediction.

Within-shell redistributions give the decisive identifiability result:

| Layout/test | N64 | N128 | Controlling direction |
|---|---:|---:|---|
| Five-shell immediate observable | `17.541` | `17.778` | thermal, `6-60 rg` |
| Five-shell projected observable | `9.983` | `10.006` | radial, `6-60 rg` |
| Five-shell projected coarse rate | `0.0612` | `0.0668` | radial |
| Eight-shell immediate observable | `17.541` | `17.778` | thermal, `6-60 rg` |
| Eight-shell projected observable | `9.941` | `9.963` | radial, `6-60 rg` |
| Eight-shell projected coarse rate | `0.1185` | `0.1372` | stress |

Every tested perturbation preserves all declared shell M/J/E coordinates to
first order. The shell rates can therefore remain comparatively insensitive
while cooling, thickness, and inner accretion change by many scientific
gates. The missing state is dynamic thermodynamic and radial-transport
structure inside the shells.

The binding decision is:

```text
decision              wp10c8h_compact_conservative_shell_closure_not_identifiable
nonlinear burst       not run
nonlinear macrostep   not authorized
next authorization    retain full DAE microbursts and reassess physical closure
```

This closes compact global and shell-only equation-free macrosteps under the
declared contracts. The full DAE remains the short-time truth model. The next
work must either derive a dynamic continuum/moment hierarchy that retains
thermodynamic and transport state, or construct an independent,
ledger-compatible stationary/bifurcation anchor after the physical
torque/power budget is specified.

## Evidence

Runtime artifacts remain ignored by repository policy.

```text
outputs/tables/causal_stable_observable_reduction_audit_wp10c8f.json
SHA256 0302741543cf4fcbd72fd4f4816625341efd67a21a9ce195faa8f57038145e83

outputs/tables/causal_stable_observable_reduction_audit_wp10c8f_arrays.npz
SHA256 90086be02c697d27f4ee155a19c35f212e121e7dc363585218dc2b1fbb71b142

outputs/tables/causal_ledger_equation_free_preflight_wp10c8g.json
SHA256 6eb6b3c02576d9840797fd840b159b664ed01cecd9e7cf144f0af16209f099ca

outputs/tables/causal_ledger_equation_free_preflight_wp10c8g_arrays.npz
SHA256 7dcbbb0069a313610d630f8311da03b495d4cf29a3d8b0fe75b8b4a34b00e1a0

outputs/tables/causal_shell_closure_preflight_wp10c8h.json
SHA256 7d3064166a2b401b1eeadc267721a2151936ed19c12d3f5bba6f929558b5d173

outputs/tables/causal_shell_closure_preflight_wp10c8h_arrays.npz
SHA256 ef65e92b99cfed3be7d8aa2a6803fb74801a9aeb7b275b4d92497655ff94c857

outputs/checkpoints/causal_five_field_wp10c8h/N064_t_0p125_shell_operators.npz
SHA256 0eb0650783decf7d67fcf82d25fc3cc36760d6afc3175bfd434ca5c0af025851

outputs/checkpoints/causal_five_field_wp10c8h/N128_t_0p125_shell_operators.npz
SHA256 3b7747dfd08b9140688e29dae516fc6493982b31ccb4c12a91d6813aa4178ac8
```
