# Mdot=5 Power-Law Mass-Coupled Wind Bridge Results

Generated: 2026-07-05

## Context

The previous energy-limited wind endpoints were post-processed into an implied
mass-loss law using

```text
dMdot_w/dlnR = 2 pi R^2 Q_wind / E_w .
```

For the first implementation pass I used a prescribed cumulative power-law wind
sink,

```text
Mdot(R) / Mdot_inner = 1 + f_w [ (R/Rin)^s - 1 ] / [ (Rout/Rin)^s - 1 ] - f_stream shape_stream(R),
```

with `s` normalized so that the selected integrated wind sink corresponds to a
fraction `zeta` of the energy-wind post-processing estimate.  This is a bridge
model, not yet the final fully local `Q_wind -> dMdot/dlnR` solved field.

The reference energy-wind checkpoint was the `eta=6.425` Mdot_inner/Edd=5,
Rout=300 rg, N=896 solution.  Assuming `E_w = GM/(2R)`, its post-processed
integrated mass loss is

```text
Mwind / Mdot_inner = 3.58551 .
```

Thus `zeta=0.015` means a prescribed wind sink of

```text
wind_sink_fraction = 0.015 * 3.58551 = 0.0537827 .
```

## Code Changes

- Added wind algebra helpers in `src/imri_qpe/layer3_minidisk_1d/winds.py`:
  - `wind_mass_loss_prime_from_energy`
  - `effective_wind_powerlaw_slope`
  - `required_wind_energy_for_powerlaw_slope`
- Added `wind_sink_shape="powerlaw"` support in the transonic local/collocation
  model.
- Added tests verifying that the normalized power-law wind sink reproduces the
  target `dlnMdot/dlnR = s`.
- Added `scripts/audit_mdot5_wind_powerlaw_slope.py`.
- Added `scripts/audit_mdot5_powerlaw_mass_wind_seed_residuals.py`.
- Added `scripts/run_mdot5_powerlaw_mass_wind_pilot.py`.
- Upgraded the pilot with:
  - resume-from-checkpoint support;
  - automatic current-zeta inference;
  - secant predictor;
  - optional zeta tangent predictor;
  - separate output stems for resumed segments.

## Key Diagnostics

The first `E_w=GM/(2R)` slope audit showed that the strongest energy-wind
states imply simulation-like mass-loss slopes:

| label | Qwind/Qvisc | implied Mwind/Minner | median s_eff |
| --- | ---: | ---: | ---: |
| epsilon_w=0.98 | 0.197638 | 0.864166 | 0.340046 |
| eta=6.425 | 0.822847 | 3.58551 | 1.31936 |

This means the energy closure can in principle be interpreted as an
`Mdot(R) ~ R^s` wind, but the full eta=6.425 implied mass loss is very large.

## Mass-Coupled Pilot Results

The prescribed power-law mass-coupled bridge initially reached strict N=896
anchors up to `zeta=0.015`.

| zeta | wind_sink_fraction | Mdot_outer/Mdot_inner | final residual | predictor | nfev | Lrad/LEdd | f_adv_global | Rson/rg |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1e-5 | 3.586e-5 | 0.200036 | 7.739e-10 | mass_compensated_u | 4 | 0.522255 | -0.006664 | 5.26875 |
| 3e-5 | 1.076e-4 | 0.200108 | 1.540e-9 | secant:1 | 2 | 0.522269 | -0.006662 | 5.26875 |
| 1e-4 | 3.586e-4 | 0.200359 | 9.756e-9 | secant:1 | 2 | 0.522318 | -0.006655 | 5.26872 |
| 3e-4 | 1.076e-3 | 0.201076 | 1.145e-9 | secant:1 | 3 | 0.522458 | -0.006637 | 5.26864 |
| 1e-3 | 3.586e-3 | 0.203586 | 7.325e-10 | secant:1 | 4 | 0.522946 | -0.006572 | 5.26835 |
| 3e-3 | 1.076e-2 | 0.210757 | 1.689e-9 | secant:1 | 5 | 0.524340 | -0.006389 | 5.26753 |
| 0.005 | 0.017928 | 0.217928 | 3.083e-10 | tangent:1 | 5 | 0.525729 | -0.006207 | 5.26671 |
| 0.007 | 0.025099 | 0.225099 | 5.109e-10 | tangent:1 | 5 | 0.527115 | -0.006027 | 5.26590 |
| 0.010 | 0.035855 | 0.235855 | 1.708e-9 | tangent:1 | 5 | 0.529186 | -0.005759 | 5.26468 |
| 0.0115 | 0.041233 | 0.241233 | 4.786e-10 | tangent:1 | 5 | 0.530218 | -0.005627 | 5.26408 |
| 0.013 | 0.046612 | 0.246612 | 9.854e-10 | secant:1 | 5 | 0.531247 | -0.005495 | 5.26348 |
| 0.015 | 0.053783 | 0.253783 | 1.410e-9 | secant:1 | 5 | 0.532616 | -0.005320 | 5.26268 |

Important numerical lesson: the old failure was mainly a predictor issue.
Without secant/tangent prediction, the `zeta=3e-4` step stalled from a
`~2.4e-2` seed residual.  With secant prediction, the same point started from
`1.9e-4` and converged.  The direct `zeta=0.003 -> 0.01` tangent jump was too
large (`~1.37e-1` seed residual), but splitting through `0.005` and `0.007`
worked.

## Adaptive Continuation Update

I added an adaptive `zeta` controller to
`scripts/run_mdot5_powerlaw_mass_wind_pilot.py`.  The controller:

- resumes from any power-law mass-wind checkpoint;
- infers the current `zeta` from `wind_sink_fraction`;
- shrinks a proposed step before Newton if the best predictor residual is too
  large;
- adjusts the next step from the accepted Newton cost;
- records adaptive metadata in the output rows.

With this controller the bridge continued smoothly from `zeta=0.015` to
`zeta=0.10` at N=896.

| zeta | wind_sink_fraction | Mdot_outer/Mdot_inner | final residual | predictor | nfev | Lrad/LEdd | f_adv_global | Rson/rg |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 0.017 | 0.060954 | 0.260954 | 4.919e-10 | tangent:1 | 5 | 0.533981 | -0.005148 | 5.26188 |
| 0.019 | 0.068125 | 0.268125 | 7.554e-10 | secant:1 | 5 | 0.535342 | -0.004976 | 5.26109 |
| 0.021 | 0.075296 | 0.275296 | 5.405e-10 | secant:1 | 5 | 0.536699 | -0.004807 | 5.26030 |
| 0.023 | 0.082467 | 0.282467 | 9.835e-9 | secant:1 | 5 | 0.538052 | -0.004638 | 5.25952 |
| 0.025 | 0.089638 | 0.289638 | 4.983e-10 | secant:1 | 5 | 0.539400 | -0.004472 | 5.25874 |
| 0.027 | 0.096809 | 0.296809 | 8.055e-10 | secant:1 | 5 | 0.540745 | -0.004307 | 5.25796 |
| 0.029 | 0.103980 | 0.303980 | 2.330e-9 | secant:1 | 5 | 0.542085 | -0.004143 | 5.25718 |
| 0.030 | 0.107565 | 0.307565 | 2.965e-9 | secant:1 | 4 | 0.542754 | -0.004062 | 5.25680 |
| 0.032 | 0.114736 | 0.314736 | 2.243e-10 | tangent:1 | 5 | 0.544087 | -0.003900 | 5.25603 |
| 0.034 | 0.121907 | 0.321907 | 1.044e-9 | secant:1 | 5 | 0.545416 | -0.003745 | 5.25526 |
| 0.036 | 0.129078 | 0.329078 | 2.001e-9 | secant:1 | 5 | 0.546741 | -0.003587 | 5.25450 |
| 0.038 | 0.136250 | 0.336250 | 2.465e-9 | secant:1 | 5 | 0.548061 | -0.003430 | 5.25374 |
| 0.040 | 0.143421 | 0.343421 | 2.047e-9 | secant:1 | 5 | 0.549377 | -0.003275 | 5.25298 |
| 0.042 | 0.150592 | 0.350592 | 2.524e-9 | secant:1 | 5 | 0.550687 | -0.003121 | 5.25222 |
| 0.044 | 0.157763 | 0.357763 | 3.937e-9 | secant:1 | 5 | 0.551992 | -0.002969 | 5.25147 |
| 0.046 | 0.164934 | 0.364934 | 1.472e-9 | secant:1 | 5 | 0.553293 | -0.002818 | 5.25073 |
| 0.048 | 0.172105 | 0.372105 | 7.157e-10 | secant:1 | 6 | 0.554588 | -0.002668 | 5.24998 |
| 0.050 | 0.179276 | 0.379276 | 1.546e-9 | secant:1 | 5 | 0.555877 | -0.002520 | 5.24924 |
| 0.054 | 0.193618 | 0.393618 | 8.520e-9 | tangent:1 | 6 | 0.558445 | -0.002227 | 5.24776 |
| 0.058 | 0.207960 | 0.407960 | 3.226e-9 | secant:1 | 7 | 0.560985 | -0.001938 | 5.24630 |
| 0.062 | 0.222302 | 0.422302 | 4.591e-9 | secant:1 | 6 | 0.563439 | -0.001641 | 5.24484 |
| 0.066 | 0.236644 | 0.436644 | 5.778e-9 | secant:1 | 6 | 0.565777 | -0.001337 | 5.24340 |
| 0.070 | 0.250986 | 0.450986 | 6.737e-9 | secant:1 | 6 | 0.568034 | -0.001035 | 5.24197 |
| 0.074 | 0.265328 | 0.465328 | 7.464e-9 | secant:1 | 6 | 0.570214 | -0.000726 | 5.24061 |
| 0.078 | 0.279670 | 0.479670 | 7.995e-9 | secant:1 | 6 | 0.572320 | -0.000431 | 5.23920 |
| 0.082 | 0.294012 | 0.494012 | 8.410e-9 | secant:1 | 6 | 0.574358 | -0.000139 | 5.23780 |
| 0.086 | 0.308354 | 0.508354 | 1.047e-8 | secant:1 | 6 | 0.576330 | 0.000149 | 5.23642 |
| 0.090 | 0.322696 | 0.522696 | 1.591e-8 | secant:1 | 6 | 0.578232 | 0.000432 | 5.23504 |
| 0.094 | 0.337038 | 0.537038 | 2.231e-8 | secant:1 | 6 | 0.580082 | 0.000710 | 5.23367 |
| 0.098 | 0.351380 | 0.551380 | 3.021e-8 | secant:1 | 6 | 0.581896 | 0.000982 | 5.23232 |
| 0.100 | 0.358551 | 0.558551 | 3.490e-8 | secant:1 | 5 | 0.582786 | 0.001115 | 5.23164 |

Mesh spot checks were run at `zeta=0.03` and `zeta=0.05`:

| zeta | N | final residual | Mdot_outer/Mdot_inner | Lrad/LEdd | f_adv_global | Rson/rg |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.03 | 896 | 2.965e-9 | 0.307565 | 0.542754 | -0.004062 | 5.25680 |
| 0.03 | 768 | 9.178e-9 | 0.307565 | 0.542841 | -0.003689 | 5.26013 |
| 0.03 | 640 | 9.370e-9 | 0.307565 | 0.542868 | -0.003586 | 5.26128 |
| 0.05 | 896 | 1.546e-9 | 0.379276 | 0.555877 | -0.002520 | 5.24924 |
| 0.05 | 768 | 3.019e-8 | 0.379276 | 0.555980 | -0.002151 | 5.25255 |
| 0.05 | 640 | 3.279e-8 | 0.379276 | 0.556008 | -0.002049 | 5.25369 |
| 0.10 | 896 | 3.490e-8 | 0.558551 | 0.582786 | 0.001115 | 5.23164 |
| 0.10 | 768 | 1.241e-7 | 0.558551 | 0.582900 | 0.001444 | 5.23482 |
| 0.10 | 640 | 1.623e-7 | 0.558551 | 0.582931 | 0.001544 | 5.23595 |

The mesh checks are encouraging: the physical diagnostics drift smoothly and
only weakly with N.  The N=640/768 residuals at `zeta=0.05` and `zeta=0.10`
are looser than the N=896 anchors, but still accepted and physics-gated.

## Interpretation

This is the first successful mass-coupled wind bridge from the Mdot=5
energy-wind endpoint.  It shows that introducing a physically motivated
`Mdot(R)` power-law sink does not immediately destroy the global transonic
solution.

## Local-Mdot BVP Prototype

I added the first local `Mdot(R)` BVP infrastructure:

- `TransonicSlimParams` now supports `mdot_profile_mode="tabulated"`.
- `stream_mass_rate_and_derivative` can evaluate a tabulated node-wise
  `logMdot(logR)` field and its local derivative.
- `scripts/run_mdot5_local_mdot_bvp_pilot.py` introduces a third unknown field
  `logMdot_i` and enforces

```text
dMdot/dlnR = source_prime - 2 pi R^2 Q_wind / E_w
```

with the same inward-positive sign convention used by the rest of the solver:

```text
dMdot/dlnR = wind_prime - source_prime .
```

This is still a prototype script, not yet merged into the production
square-Newton solver.  It uses the prescribed power-law bridge as the initial
guess, removes the analytic wind mass sink, and calibrates the wind launch
energy with a multiplier

```text
E_w = eta_E GM/(2R).
```

Calibration pilots:

| bridge target | N | eta_E | initial residual | final residual | Mdot_outer/Mdot_inner | Lrad/LEdd | f_adv_global | Rson/rg |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zeta=0.03 | 96 | 33.333 | 0.578 | 4.574e-4 | 0.307304 | 0.553070 | 0.002176 | 5.26372 |
| zeta=0.03 | 128 | 33.333 | 0.379 | 4.055e-3 | 0.305616 | 0.553521 | 0.000565 | 5.26589 |
| zeta=0.05 | 96 | 20 | 0.538 | 7.523e-3 | 0.378536 | 0.583849 | 0.002699 | 5.25314 |
| zeta=0.10 | 96 | 10 | 0.461 | 2.303e-2 | 0.559040 | 0.605375 | 0.013074 | 5.23509 |

The key diagnostic lesson is that using the physical unscaled
`E_w=GM/(2R)` tries to impose the full post-processed wind mass loss and badly
misses the prescribed `zeta` bridge.  The calibrated `eta_E ~= 1/zeta` sequence
is the right debug target and allows the new local BVP to approach the
prescribed branch.  The `zeta=0.03` case is already close; higher loading still
hits a residual floor and needs better scaling/continuation before it becomes a
production-quality local solver.

However, it is not yet the final physical mass-loaded wind branch:

- The strongest robust branch is still the prescribed bridge; the local
  `Mdot(R)` BVP exists only as a prototype and is not yet solved tightly at
  higher loading.
- The reached mass coupling is still small compared with the full implied
  energy-wind mass loss: `zeta=0.10` is only 10 percent of the full
  `Mwind/Minner=3.58551` estimate.
- The solution remains radiatively dominated in these diagnostics
  (`f_adv_global` is only slightly positive at `zeta=0.10`), so this is not yet
  a stronger advective/hot branch.
- Initial N/mesh spot checks are positive, but this is not yet a full
  robustness campaign over source shape, N, and outer closure.

## Next Recommended Steps

1. Improve the local `Mdot(R)` BVP prototype:
   - add staged continuation in `eta_E` and/or `zeta`;
   - add residual localization for the mass-continuity residual;
   - improve residual scaling so N=96 and N=128 converge to the same local
     root;
   - then port the third-field residual into the production square-Newton
     machinery.
2. Keep the prescribed bridge available as a calibration/debug target at
   `zeta=0.03`, `0.05`, and `0.10`.
3. Add fuller N/mesh certification if the prescribed bridge is pushed beyond
   `zeta=0.10`.
4. Only after the mass-coupled no-wind/heating topology is robust should this
   be combined with additional stream heating or wind angular-momentum loss.

## Verification

Full test suite:

```text
158 passed in 2.96s
```

Main result files:

- `outputs/tables/m5_energy_wind_powerlaw_slope_diagnostics.md`
- `outputs/tables/m5_energy_wind_powerlaw_mass_coupled_pilot.md`
- `outputs/tables/m5_energy_wind_powerlaw_mass_coupled_resume_0p003_to_0p01.md`
- `outputs/tables/m5_energy_wind_powerlaw_mass_coupled_resume_0p01_to_0p015.md`
- `outputs/tables/m5_energy_wind_powerlaw_mass_coupled_adaptive_0p015_to_0p03.md`
- `outputs/tables/m5_energy_wind_powerlaw_mass_coupled_adaptive_0p03_to_0p05.md`
- `outputs/tables/m5_energy_wind_powerlaw_mass_coupled_zeta0p03_N768_spotcheck.md`
- `outputs/tables/m5_energy_wind_powerlaw_mass_coupled_zeta0p03_N640_spotcheck.md`
- `outputs/tables/m5_energy_wind_powerlaw_mass_coupled_zeta0p05_N768_spotcheck.md`
- `outputs/tables/m5_energy_wind_powerlaw_mass_coupled_zeta0p05_N640_spotcheck.md`
- `outputs/tables/m5_energy_wind_powerlaw_mass_coupled_adaptive_0p05_to_0p10.md`
- `outputs/tables/m5_energy_wind_powerlaw_mass_coupled_zeta0p10_N768_spotcheck.md`
- `outputs/tables/m5_energy_wind_powerlaw_mass_coupled_zeta0p10_N640_spotcheck.md`
- `outputs/tables/m5_local_mdot_bvp_zeta0p03_N96_etaE33_pilot.md`
- `outputs/tables/m5_local_mdot_bvp_zeta0p03_N128_etaE33_pilot.md`
- `outputs/tables/m5_local_mdot_bvp_zeta0p05_N96_etaE20_pilot.md`
- `outputs/tables/m5_local_mdot_bvp_zeta0p10_N96_etaE10_pilot.md`
