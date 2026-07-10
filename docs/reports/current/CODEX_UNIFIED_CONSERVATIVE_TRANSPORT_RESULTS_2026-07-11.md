# Unified Conservative Transport Results

Date: 2026-07-11

## Scope

This work replaces the provisional algebraic angular-momentum bookkeeping with
one production formulation for inward mass, angular-momentum, and total-energy
fluxes. The primary radial state is

```text
(log u, log T, F, j, epsilon, log R_son)
```

where `F=Mdot/Mdot_inner`, `j=J/(Mdot_inner*l_ref)`, and
`epsilon=E/(Mdot_inner*c^2)`.

The closure is explicit:

- stream angular momentum and energy are set by the Layer-1 circularization
  radius, `R_circ=248.96693 rg`;
- the baseline wind carries the local disk specific angular momentum,
  `l_w=l_disk`;
- its carried energy is `B_w=B_disk+E_launch+Omega(l_w-l_disk)`;
- external torque and external power are independent terms, so torque work is
  not silently double counted.

Each radial interval enforces finite-volume mass, angular-momentum, and energy
conservation together with radial momentum and the local energy-state
compatibility identity. Optimizer weights are separated from raw certification
audits.

## Energy Identity Gate

The legacy radial equation uses `(1/Sigma)dPi/dlnR`, while the entropy equation
contains `P/rho^2 drho/dlnR`. A naive Bernoulli flux therefore has an order-one
identity defect. Including the exact cumulative vertical-work term closes the
identity on the canonical `N=768` no-wind checkpoint:

```text
raw normalized defect       ~ 1
vertical-work-corrected max  1.225e-15
```

This gate is now tested before conservative energy transport is used.

## Regression Results

### No-wind standard disk

The `Mdot/Edd=5` no-wind benchmark was re-solved with the five-field
conservative sonic conditions. The `N=128` checkpoint has:

```text
radial                 1.350e-7
mass                   1.395e-6
angular momentum       6.467e-7
energy                 1.917e-7
energy compatibility   7.534e-6
sonic                   8.295e-9
```

This confirms that the unified flux variables preserve the standard high-rate
slim-disk solution.

### Compact stream, no wind

For `Mdot_inner/Edd=2`, `Rout=300 rg`, `f_s=0.80`, a normalized source-shape
homotopy converts the actual tanh checkpoint to the compact C2 source with
physical circularization closure. With the conservative sonic gate at `N=128`:

```text
radial                 1.258e-6
mass                   4.991e-6
angular momentum       6.247e-6
energy                 1.645e-8
energy compatibility   1.247e-5
sonic                   4.530e-8
```

Status: exploratory support, narrowly above the preferred `1e-5` gate.

## Energy-limited Wind Continuation

The wind launch multiplier is fixed at `eta_E=98.125`; the physical efficiency
parameter is finite, `0 <= epsilon_w <= 1`. Custom-grid provenance is stored in
every new checkpoint, preventing a remeshed state from being replayed on the
wrong grid.

Certified exploratory continuation (`raw max <=3e-5`) reaches:

```text
epsilon_w = 0.50, N=256, raw max = 2.964e-5
epsilon_w = 0.54, N=320, raw max = 2.970e-5
```

Continuation-only scouts (`raw max <=5e-5`) reach `epsilon_w=0.90`. The
`epsilon_w=1` endpoint has raw maximum `5.123e-5` and is not accepted even as a
`5e-5` scout.

Selected physical diagnostics:

| epsilon_w | N | status | wind/Mdot_inner | f_adv | Lrad/LEdd | max H/R | Rson/rg |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0.00 | 128 | exploratory | 0 | 0.1991 | 0.8273 | 0.2121 | 4.7057 |
| 0.50 | 256 | exploratory | 0.00247 | 0.1947 | 0.8223 | 0.2107 | 4.7201 |
| 0.54 | 320 | exploratory | 0.00264 | 0.1943 | 0.8226 | 0.2109 | 4.7222 |
| 0.90 | 320 | scout only | 0.00440 | 0.1944 | 0.8227 | 0.2109 | 4.7225 |
| 1.00 | 320 | rejected at scout gate | 0.00489 | 0.1944 | 0.8227 | 0.2109 | 4.7226 |

## Numerical Findings

1. The apparent `epsilon_w~0.34` and `~0.46` walls moved with `N` and node
   placement. They were mesh-resolution effects, not sonic or physical branch
   endpoints.
2. Global residual equidistribution damaged the compact source and outer
   buffer. Restricting adaptation to `R<100 rg` preserved those fixed
   structures and produced an accepted `N=256` anchor.
3. A checkpoint without its custom grid can generate an artificial order-one
   radial residual on restart. New continuation and refinement checkpoints
   save and restore `custom_grid_xi`.
4. Direct `N=384` remaps, including a prefix-frozen variant, export residual to
   the first radial interval. They are rejected and not promoted.
5. No mesh-stable singularity, radial fold, or sonic failure appears in the
   accepted wind continuation. A phase-space DAE patch is therefore not
   activated for this branch.

## Scientific Interpretation

The implementation now demonstrates a physical, explicitly closed,
mass-loaded wind solution under one conservative mass/angular/energy ledger.
It does **not** recover the sought strong hot/advective branch:

- the accepted wind removes only about `0.26%` of `Mdot_inner`;
- advection remains near `0.19-0.20` and decreases slightly with wind;
- `H/R`, luminosity, and sonic radius barely change;
- this continuation starts from the `Mdot_inner/Edd=2` compact-stream branch,
  not yet the target `Mdot_inner/Edd=5` stream-fed branch.

The next scientific step is to use this verified formulation at
`Mdot_inner/Edd=5`, first without stream and then with compact stream fractions
`0.05, 0.10, 0.30`. Only after those roots are mesh certified should launch
energy, wind lever arm, and stream heating be varied to search for a genuinely
hot topology.

## Mdot=5 Finite-Minidisk Extension

The planned high-rate extension has now been carried out at
`Rout=335 rg`, `Rinj=240 rg`, with the same Layer-1 circularization closure and
no stream heating.

The standard `Rout=10000 rg` solution was first truncated to `335 rg`. The
resulting no-stream `N=192` root is preferred:

```text
raw maximum             2.203e-6
sonic                    4.922e-8
F_outer                  0.999993
```

Compact C2 stream continuation then gives:

| f_s | N | status | raw maximum | F_outer | f_adv | Lrad/LEdd | max H/R |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0.05 | 192 | exploratory | 1.533e-5 | 0.950026 | 0.4146 | 1.2592 | 0.2905 |
| 0.10 | 256 | exploratory | 1.831e-5 | 0.900035 | 0.4125 | 1.2602 | 0.2906 |
| 0.30 | 384 | exploratory | 2.917e-5 | 0.700117 | 0.4102 | 1.2655 | 0.2908 |

The source integral closes to `0.299976 Mdot_inner` at `f_s=0.30`. Stream
loading therefore changes the outer mass supply as required, but does not
create a new hot topology; advection decreases slightly.

### Mdot=5 wind and launch-energy scans

At `f_s=0.30`, `eta_E=98.125`, the energy-limited wind remains exploratory
through `epsilon_w=0.20`. Higher efficiencies are scouts only:

```text
epsilon_w=0.20: raw max 2.914e-5, wind/Mdot_inner 0.00136
epsilon_w=0.50: raw max 4.175e-5, wind/Mdot_inner 0.00340
epsilon_w=0.80: raw max 7.562e-5
epsilon_w=0.90: raw max 1.123e-4, rejected by 1e-4 scout gate
```

Lowering launch energy at fixed `epsilon_w=0.20` increases mass loading:

| eta_E | status | wind/Mdot_inner | f_adv | max H/R |
|---:|---|---:|---:|---:|
| 98.125 | exploratory | 0.00136 | 0.4101 | 0.2908 |
| 50 | exploratory | 0.00266 | 0.4101 | 0.2908 |
| 40 | exploratory | 0.00333 | 0.4102 | 0.2907 |
| 30 | scout | 0.00444 | 0.4101 | 0.2907 |
| 20 | scout | 0.00665 | 0.4103 | 0.2907 |
| 10 | scout | 0.01327 | 0.4114 | 0.2908 |
| 8 | loose scout | 0.01659 | 0.4114 | 0.2908 |

The `eta_E=7` step exceeds the `2e-4` loose-scout gate, with defects localized
to the compact source band and first radial interval. It is not a sonic
failure. Across the controlled eta range there is no hot-branch transition:
`f_adv`, `H/R`, luminosity, and `Rson` remain nearly unchanged.

The next numerical task is a conservative source-band refinement that can
support `eta_E=O(1-10)`. The next physical task, after that gate passes, is to
test wind lever arm and stream heating separately rather than combining them.

## Reproduction Entry Points

- `scripts/run_unified_conservative_energy_identity_audit.py`
- `scripts/run_unified_conservative_no_wind_regression.py`
- `scripts/run_unified_conservative_compact_source_homotopy.py`
- `scripts/run_unified_conservative_wind_continuation.py`
- `scripts/run_unified_conservative_wind_refinement.py`
- `scripts/run_unified_conservative_mdot5_stream_ladder.py`
- `scripts/run_unified_conservative_mdot5_wind_ladder.py`
- `scripts/run_unified_conservative_mdot5_eta_ladder.py`

The generated checkpoints and tables under `outputs/` are intentionally not
part of the compact Git history.
