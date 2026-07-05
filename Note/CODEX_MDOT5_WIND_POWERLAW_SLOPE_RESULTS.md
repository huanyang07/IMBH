# Mdot=5 Wind Power-Law Slope Diagnostic

Generated after adding the simulation-calibrated wind-slope audit.

## Question

If the energy-only wind loss is interpreted as a physical wind with

```math
\dot\Sigma_w = Q_{\rm wind}/E_w,
```

does it naturally imply a simulation-like radial mass-loss law

```math
\dot M(R) \propto R^s
```

with `s` of order `0.3-1`?

## Implementation

Added reusable wind helpers:

- `wind_mass_loss_prime_from_energy(Q_wind, R, E_w)`
- `effective_wind_powerlaw_slope(Q_wind, R, Mdot, E_w)`
- `required_wind_energy_for_powerlaw_slope(Q_wind, R, Mdot, s_target)`

The audit script is:

```text
scripts/audit_mdot5_wind_powerlaw_slope.py
```

It post-processes the existing `Mdot_inner/Edd=5`, `Rout=335 rg`,
`f_s=0.8`, energy-wind checkpoints and assumes, for the first pass,

```math
E_w = GM/(2R).
```

It reports the wind-only effective slope

```math
s_{\rm eff}
=
\frac{d\ln \dot M}{d\ln R}
=
\frac{2\pi R^2 Q_{\rm wind}}{E_w \dot M},
```

and also the net stream-fed slope

```math
s_{\rm net}
=
\frac{\dot M'_w-\dot M'_s}{\dot M}.
```

`s_net` is useful for the minidisk bookkeeping, but the comparison to isolated
super-Eddington wind simulations should use the wind-only `s_eff`.

## Outputs

- `outputs/tables/m5_energy_wind_powerlaw_slope_diagnostics.md`
- `outputs/tables/m5_energy_wind_powerlaw_slope_diagnostics.json`
- `outputs/tables/m5_energy_wind_powerlaw_slope_profiles.json`
- `outputs/figures/m5_energy_wind_powerlaw_slope_diagnostics.png`

## Main Results

| state | Qwind/Qvisc | Lrad/LEdd | implied Mwind/Min, eta_E=1 | s_eff p10 | s_eff p50 | s_eff p90 | eta_E p50 for s=0.5 | eta_E p50 for s=1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| epsilon_w=0.98 | 0.198 | 1.143 | 0.864 | 0.103 | 0.340 | 0.453 | 0.680 | 0.340 |
| epsilon_w=0.997 | 0.690 | 0.693 | 2.967 | 0.416 | 1.106 | 1.294 | 2.212 | 1.106 |
| eta=6.20 | 0.783 | 0.581 | 3.386 | 0.508 | 1.248 | 1.416 | 2.496 | 1.248 |
| eta=6.35 | 0.811 | 0.541 | 3.523 | 0.542 | 1.296 | 1.455 | 2.592 | 1.296 |
| eta=6.425 | 0.823 | 0.522 | 3.586 | 0.559 | 1.319 | 1.472 | 2.639 | 1.319 |

The plot shows the same result visually: the moderate wind state lies entirely
inside the `s~0.3-1` simulation-motivated band over most of the wind-active
region. The strongest wind states are only modestly above `s=1` for
`E_w=GM/(2R)`.

## Interpretation

This is encouraging. The energy-limited wind closure is not obviously
incompatible with simulation-style radial mass-loss laws.

For the gentle wind state, `E_w=GM/(2R)` naturally gives a median
`s_eff ~= 0.34`.

For the strongest validated states, the same launch energy gives
`s_eff ~= 1.25-1.32`. To force a target `s=1`, only a modestly larger effective
launch energy is needed:

```math
E_w \simeq 1.25-1.32\,GM/(2R).
```

For a target `s=0.5`, the required energy is larger but still not absurd:

```math
E_w \simeq 2.5-2.6\,GM/(2R).
```

This could represent terminal kinetic energy, enthalpy, or extra work required
to launch the wind.

## Caveat

The implied integrated wind mass for high `epsilon_w` is large:

```text
Mwind/Min ~= 3.0-3.6
```

for the strongest states with `eta_E=1`. Therefore the physical branch must
increase the outer supply/reservoir and repolish the BVP. The current result is
still a post-processing diagnostic of the energy-only branch, not yet a solved
mass-loaded wind branch.

## Updated Next Step

Implement a target-`s` or calibrated-`eta_E` mass-coupled wind mode:

1. Start with a target `s=1` equivalent, because it requires only
   `eta_E~1.3` for the strong wind states.
2. Use `l_w=l` first, so wind carries its own angular momentum but applies no
   magnetic lever-arm torque.
3. Homotope the mass-coupling strength from 0 to 1, while refreshing the outer
   reservoir so `Mdot_outer/Min` can exceed the current stream-only value.
4. Then scan `s_target=0.5, 0.7, 1.0` or equivalently
   `eta_E=1, 1.5, 2, 3`.
5. Only call the branch physical if the mass budget closes, `s_eff` remains
   smooth, and the mesh validation survives.
