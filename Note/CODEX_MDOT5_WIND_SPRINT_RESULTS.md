# Mdot=5 Wind Sprint Results

Date: 2026-07-04

This note records the run from GPT's requested points 1--5 after
`Note/CODEX_MDOT5_HOT_STATE_INTERPRETATION_AND_NEXT_PLAN.md`.

## Summary

The `Mdot_inner/Edd=5`, `Rout=335 rg`, compact-source branch is now protected by
regression anchors and has a working first wind layer.

Main outcomes:

- Regression anchors passed for the standard `Mdot=5` no-wind slim disk,
  stream-fed `f_s=0.30,0.50,0.80`, and heating anchors
  `eta_heat=0,0.1,1.0`.
- A short no-wind source-fraction extension accepted `f_s=0.825`; the next
  `f_s=0.85` solve became expensive and was stopped because this was only a
  certification task.
- Bookkeeping wind passed for `eta_heat=0,0.1,1.0` through
  `f_wind=0.10`.
- The mass budget behaves as expected:

```text
Mdot_outer/Mdot_inner ~= 1 - f_s + f_wind
```

- The first energy-limited wind residual is implemented and runs, but with the
  current vertical Eddington threshold it is inactive even at
  `eta_heat=1.0`, `epsilon_w=0.30`.

## Code Changes

Implemented:

- `wind_energy_limited_epsilon` in `TransonicSlimParams`.
- Optional local energy sink:

```text
Q_visc + Q_stream - Q_rad - Q_adv - Q_wind = 0
```

where

```text
Q_wind = epsilon_w [Q_visc + Q_stream - Q_adv - Q_Edd,z]_+
Q_Edd,z = 2 c Omega_K^2 H / kappa
```

- `wind_energy_loss_rate(...)` in the local transonic system.
- Stream-mass runner modes:
  - `IMBH_STANDARD_SLIM_STREAM_MASS_WIND_FRACTIONS`
  - `IMBH_STANDARD_SLIM_STREAM_MASS_ENERGY_WIND_EPSILONS`
- Output columns for bookkeeping wind fraction/integral and energy-wind
  diagnostics.
- Updated regression anchor table with heat/wind metadata.

All new wind parameters default to zero, so existing no-wind anchors are
unchanged unless the wind env vars are explicitly set.

## Phase 1: Regression Anchors

Output:

- `outputs/tables/m5_hot_state_regression_anchors.md`

Selected rows:

| anchor | full | Mdot/Edd | f_s | eta_heat | Mout/Min | f_adv_global | Lrad/LEdd | max H/R | Rson/rg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| standard Mdot=5 no-wind | 2.293e-06 | 5 | 0 | 0 | 1.000 | 0.4534 | 1.541 | 0.3164 | 4.360 |
| stream f_s=0.30 | 7.491e-08 | 5 | 0.30 | 0 | 0.700 | 0.4881 | 1.342 | 0.3158 | 4.361 |
| stream f_s=0.50 | 2.639e-08 | 5 | 0.50 | 0 | 0.500 | 0.4929 | 1.324 | 0.3155 | 4.361 |
| stream f_s=0.80 | 2.783e-10 | 5 | 0.80 | 0 | 0.200 | 0.4990 | 1.300 | 0.3152 | 4.361 |
| stream f_s=0.80, heat=0.1 | 1.021e-10 | 5 | 0.80 | 0.1 | 0.200 | 0.4962 | 1.315 | 0.3159 | 4.361 |
| stream f_s=0.80, heat=1 | 9.688e-12 | 5 | 0.80 | 1.0 | 0.200 | 0.4581 | 1.455 | 0.3262 | 4.349 |

All selected anchors passed the `1e-5` strict check used for this sprint.

## Phase 2: Short No-Wind f_s Extension

Output:

- `outputs/tables/high_mdot_stream_m5_compact_N896_080_to090_no_energy_merit.md`
- `outputs/checkpoints/high_mdot_stream_m5_compact_N896_080_to090_no_energy_merit/`

Accepted:

| f_s | final full | Mout/Min | f_adv_global | Lrad/LEdd | max H/R | Rson/rg |
|---:|---:|---:|---:|---:|---:|---:|
| 0.825 | 5.731e-10 | 0.175 | 0.4994 | 1.298 | 0.3152 | 4.361 |

The attempted `f_s=0.85` step became a long sparse linear solve and was
interrupted. This does not look like a branch failure; it is a cost issue in a
certification-only extension.

## Phase 3/4: Bookkeeping Wind

The existing local continuity convention was used:

```text
dMdot/dlnR = Mdot_wind_prime - Mdot_stream_prime
```

The bookkeeping wind scan used the same compact source geometry and `l_w=l`
implicitly, so this is a mass-budget/sign test rather than a full wind physics
model.

Outputs:

- `outputs/tables/high_mdot_stream_m5_fs080_bookkeeping_wind_eta0_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_bookkeeping_wind_eta01_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_bookkeeping_wind_eta1_N896.md`

For `eta_heat=0`:

| f_wind | final full | Mout/Min | source integral | wind integral | f_adv_global | Lrad/LEdd |
|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 1.814e-10 | 0.209998 | 0.8001 | 0.009998 | 0.4988 | 1.300 |
| 0.03 | 5.415e-10 | 0.229993 | 0.8001 | 0.029994 | 0.4985 | 1.302 |
| 0.10 | 1.833e-10 | 0.299976 | 0.8000 | 0.099980 | 0.4974 | 1.307 |

For `eta_heat=1`:

| f_wind | final full | Mout/Min | source integral | wind integral | f_adv_global | Lrad/LEdd |
|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 8.327e-09 | 0.209998 | 0.8001 | 0.009998 | 0.4588 | 1.454 |
| 0.03 | 1.076e-09 | 0.229993 | 0.8000 | 0.029994 | 0.4600 | 1.452 |
| 0.10 | 1.516e-09 | 0.299976 | 0.8000 | 0.099981 | 0.4636 | 1.447 |

The bookkeeping layer therefore passes the sign and budget tests. At fixed
`Mdot_inner`, adding mass loss raises `Mdot_outer/Mdot_inner` by the expected
amount and changes the physical diagnostics smoothly.

## Phase 5: Energy-Limited Wind Pilot

Outputs:

- `outputs/tables/high_mdot_stream_m5_fs080_energy_wind_eta0_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_energy_wind_eta1_quick_N896.md`

No-heating branch:

- `epsilon_w=0.01` and `0.03` accepted.
- `integrated_Qwind/Qvisc = 0`
- active interval fraction = `0`
- the attempted `epsilon_w=0.10` row was interrupted because the remesh solve
  was costly while prior rows showed no wind activation.

Heated branch at `eta_heat=1`:

| epsilon_w | final full | integrated Qwind/Qvisc | active interval fraction | result |
|---:|---:|---:|---:|---|
| 0.01 | 3.418e-10 | 0 | 0 | seed accepted |
| 0.03 | 1.068e-09 | 0 | 0 | seed accepted |
| 0.10 | 4.136e-09 | 0 | 0 | seed accepted |
| 0.30 | 2.051e-08 | 0 | 0 | accepted, but above strict seed tolerance |

Interpretation: with the current prescription,

```text
Q_Edd,z = 2 c Omega_K^2 H / kappa
```

the branch does not exceed the threshold, even with conservative stream heating
at `eta_heat=1`. The energy-wind implementation is wired in, but this closure is
inactive on the present `Mdot=5`, `f_s=0.80` anchor.

## Interpretation

The project now has:

1. A robust `Mdot=5` steady upper-branch anchor.
2. A working bookkeeping wind layer with correct source/sink mass budgets.
3. An implemented but inactive first energy-limited wind closure.

The next scientific question is not whether wind signs are correct; they are.
The question is whether the vertical Eddington threshold is the right activation
criterion for this height-integrated model, or whether the wind should instead
be tied to a different local excess-energy, photon-trapping, or reservoir/escape
condition.

## Recommended Next Step

Do not immediately tune `epsilon_w`; it has no effect if the activation bracket
is zero. First audit the local activation quantity:

```text
Q_visc + Q_stream - Q_adv - Q_Edd,z
```

along the `Mdot=5`, `f_s=0.80`, `eta_heat=0` and `eta_heat=1` profiles. Plot its
maximum and radial profile, and compare with `Q_rad`, `Q_visc`, and local
vertical Eddington flux. If the threshold is everywhere positive by a large
margin, revise the wind trigger before any larger wind scan.

## Follow-Up: Wind-Trigger Audit

Implemented:

- `scripts/run_standard_slim_stream_wind_trigger_audit.py`
- `outputs/tables/high_mdot_stream_m5_wind_trigger_audit.md`
- `outputs/tables/high_mdot_stream_m5_wind_trigger_audit.json`
- `outputs/figures/high_mdot_stream_m5_wind_trigger_audit.png`

The audit evaluates the interval-midpoint activation bracket on the
`Mdot_inner/Edd=5`, `Rout=335 rg`, `f_s=0.80`, compact-source anchors:

```text
Q_trigger = Q_visc + Q_stream - Q_adv - Q_Edd,z
Q_Edd,z = 2 c Omega_K^2 H / kappa
```

Results:

| anchor | eta_heat | max Qavail/Qedd | integrated Qavail/Qedd | max trigger/Qvisc | active interval fraction | peak R/rg |
|---|---:|---:|---:|---:|---:|---:|
| m5_fs080_heat0 | 0 | 0.999890 | 0.999119 | -4.696e-05 | 0 | 6.351 |
| m5_fs080_heat0p1 | 0.1 | 0.999890 | 0.999186 | -4.690e-05 | 0 | 6.364 |
| m5_fs080_heat1 | 1 | 0.999894 | 0.999395 | -4.606e-05 | 0 | 6.351 |

Interpretation:

- The energy-limited wind is inactive because the trigger is negative
  everywhere, but only by about `5e-5` of the local viscous heating at the
  closest point.
- The maximum `Qavail/Q_Edd,z` is already `0.99989`, so the solution is
  effectively vertical-Eddington capped near `R ~= 6.35 rg`.
- The stream-heating term is not what controls this peak: the peak is in the
  inner flow, while the source/heating annulus is near the outer disk.
- Therefore changing `epsilon_w` alone cannot activate the wind. The remaining
  question is the physical activation criterion, not the bookkeeping or sign of
  the implemented sink.

Recommended next experiment:

1. Add a diagnostic-only soft threshold scan, for example
   `max(Qavail - chi_edd Q_Edd,z, 0)` with
   `chi_edd = 1.0, 0.999, 0.995, 0.99, 0.98`, without yet claiming a physical
   wind model.
2. Compare the implied integrated wind power and launch radius with
   photon-trapping, Bernoulli/escape, and vertical-Eddington interpretations.
3. Only after choosing a physically motivated activation rule should the
   energy-limited sink be solved self-consistently in the BVP.

## Follow-Up: Soft Wind-Threshold Diagnostic

Implemented:

- `scripts/run_standard_slim_stream_soft_wind_threshold_scan.py`
- `outputs/tables/high_mdot_stream_m5_soft_wind_threshold_scan.md`
- `outputs/tables/high_mdot_stream_m5_soft_wind_threshold_scan.json`
- `outputs/figures/high_mdot_stream_m5_soft_wind_threshold_scan.png`

This scan does not alter the BVP solution. It evaluates the diagnostic rule

```text
Qwind_raw = max(Qavail - chi_edd Q_Edd,z, 0)
dotSigma_w = Qwind_raw / (GM / 2R)
```

for `chi_edd = 1.0, 0.999, 0.995, 0.99, 0.98` on the same
`Mdot_inner/Edd=5`, `Rout=335 rg`, `f_s=0.80` compact-source anchors.

Selected results:

| anchor | eta_heat | chi_edd | active R range/rg | int Qwind/Qvisc | implied Mwind/Min |
|---|---:|---:|---|---:|---:|
| m5_fs080_heat0 | 0 | 0.999 | 4.373-131.4 | 3.423e-04 | 9.032e-04 |
| m5_fs080_heat0 | 0 | 0.990 | 4.373-232.0 | 4.755e-03 | 2.068e-02 |
| m5_fs080_heat0 | 0 | 0.980 | 4.373-238.3 | 9.707e-03 | 4.370e-02 |
| m5_fs080_heat0p1 | 0.1 | 0.999 | 4.374-136.9 | 3.451e-04 | 9.264e-04 |
| m5_fs080_heat0p1 | 0.1 | 0.990 | 4.374-239.7 | 4.823e-03 | 2.183e-02 |
| m5_fs080_heat0p1 | 0.1 | 0.980 | 4.374-246.1 | 9.840e-03 | 4.598e-02 |
| m5_fs080_heat1 | 1 | 0.999 | 4.362-242.4 | 4.179e-04 | 1.865e-03 |
| m5_fs080_heat1 | 1 | 0.990 | 4.362-252.8 | 5.452e-03 | 3.066e-02 |
| m5_fs080_heat1 | 1 | 0.980 | 4.362-255.0 | 1.105e-02 | 6.281e-02 |

Interpretation:

- A microscopic softening, `chi_edd=0.999`, produces only
  `Mwind/Mdot_inner ~= 9e-4` to `1.9e-3`. That is likely too small to reshape
  the branch.
- A moderate softening, `chi_edd=0.99`, gives a few percent mass-loss scale:
  `Mwind/Mdot_inner ~= 0.021`, `0.022`, and `0.031` for
  `eta_heat = 0, 0.1, 1`.
- A stronger softening, `chi_edd=0.98`, gives `Mwind/Mdot_inner ~= 0.044`,
  `0.046`, and `0.063`.
- The diagnostic launch zone always begins near the inner transonic region
  (`R ~= 4.36-4.37 rg`) and extends farther outward as `chi_edd` is lowered.
  Stream heating changes the amplitude and outer extent, but it does not move
  the launch onset to the stream annulus.

Conclusion:

The branch is close enough to vertical Eddington that a smooth/soft trigger can
produce a nonzero wind without numerical violence. However, a trigger as close
to unity as `chi_edd=0.999` is dynamically weak. A self-consistent BVP wind test
should start with a moderate diagnostic scale such as `chi_edd=0.99` and small
`epsilon_w`, while treating `chi_edd` as a model parameter that must be
physically justified rather than fitted.

## Follow-Up: Self-Consistent Soft-Trigger Wind Pilot

Implemented:

- `wind_eddington_chi` in `TransonicSlimParams`, default `1.0`.
- `energy_limited_wind(..., chi_edd=1.0)`.
- `IMBH_STANDARD_SLIM_STREAM_MASS_ENERGY_WIND_CHI` in
  `scripts/run_standard_slim_stream_mass_annulus_scan.py`.
- Checkpoint/table persistence for `wind_eddington_chi`.

Default behavior is unchanged because `chi_edd=1.0` reproduces the previous hard
vertical-Eddington threshold.

Pilot settings:

```text
Mdot_inner/Edd = 5
Rout = 335 rg
f_s = 0.80
source shape = compact_c2
torque_delta_l_fraction = +0.005
chi_edd = 0.99
N = 896
```

No-heating anchor (`eta_heat=0`):

- table:
  `outputs/tables/high_mdot_stream_m5_fs080_soft_energy_wind_chi099_eta0_N896.md`
- checkpoints:
  `outputs/checkpoints/high_mdot_stream_m5_fs080_soft_energy_wind_chi099_eta0_N896/`

| epsilon_w | final full | int Qwind/Qvisc | active frac | Lrad/LEdd | f_adv global | nfev | result |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.01 | 1.979e-09 | 4.803e-05 | 0.680 | 1.300 | 0.4989 | 4 | accepted |
| 0.03 | 9.602e-09 | 1.470e-04 | 0.680 | 1.299 | 0.4989 | 9 | accepted |

Heated anchor (`eta_heat=1`):

- table:
  `outputs/tables/high_mdot_stream_m5_fs080_soft_energy_wind_chi099_eta1_N896.md`
- checkpoints:
  `outputs/checkpoints/high_mdot_stream_m5_fs080_soft_energy_wind_chi099_eta1_N896/`

| epsilon_w | final full | int Qwind/Qvisc | active frac | Lrad/LEdd | f_adv global | nfev | result |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.01 | 9.862e-09 | 5.507e-05 | 0.772 | 1.455 | 0.4581 | 4 | accepted |
| 0.03 | 4.032e-09 | 1.686e-04 | 0.772 | 1.455 | 0.4580 | 14 | accepted |

Additional attempted scouts:

- Direct no-heating jump from `epsilon_w=0.03` to `0.10` was interrupted after
  several minutes. It was stuck in finite-difference Jacobian/corrector work,
  not in an obvious sonic failure.
- Staged no-heating jump from `0.03` to `0.05` at `N=896` was also interrupted
  after several minutes, again inside the Newton/corrector loop.
- A naive `N=640` remap scout to `epsilon_w=0.05` started from a poor remapped
  seed (`initial_full ~= 4.97e-02`), so it did not provide a clean resolution
  comparison and was interrupted.

Interpretation:

- The softened self-consistent wind is real and numerically clean in the small
  feedback regime. The solved `Qwind/Qvisc` scales roughly linearly with
  `epsilon_w`, as expected.
- The absolute wind power is small for `epsilon_w <= 0.03`, about
  `5e-5` to `1.7e-4` in integrated `Qwind/Qvisc`, so these accepted pilots do
  not yet reshape the high-Mdot branch.
- The next bottleneck is not the existence of a soft-trigger branch. It is
  efficient continuation to stronger wind feedback, likely affected by the
  nonsmooth `max(Qavail - chi Q_Edd,z, 0)` switch and high-N finite-difference
  Jacobian cost.

Recommended next step:

1. Add an optional smooth activation width, default zero, for the energy-limited
   wind excess term:

   ```text
   excess = softplus(Qavail - chi_edd Q_Edd,z; width)
   ```

   where the width is a small fraction of `Q_Edd,z`.
2. Repeat `chi_edd=0.99` with `epsilon_w = 0.03, 0.05, 0.07, 0.10` using the
   smooth activation.
3. If smoothing helps, add a continuation parameter for `epsilon_w` and/or
   analytic/local derivatives for the wind part of `interval_E`.
4. Only after reaching dynamically meaningful wind power should we revisit
   whether this produces a stronger advective/hot branch.

## Follow-Up: Smooth Activation Pilot

Implemented:

- `activation_width` in `energy_limited_wind(...)`.
- `wind_activation_width_fraction` in `TransonicSlimParams`, default `0`.
- `IMBH_STANDARD_SLIM_STREAM_MASS_ENERGY_WIND_WIDTH_FRACTION` in the stream
  scan driver.
- Checkpoint/table persistence for `wind_activation_width_fraction`.

The smooth activation uses a capped softplus:

```text
hard excess = Qavail - chi_edd Q_Edd,z
width       = wind_activation_width_fraction * Q_Edd,z
excess      = softplus(hard excess; width)
excess      = min(excess, max(Qavail, 0))
Qwind       = epsilon_w * excess
```

The cap preserves `Qrad >= 0` and prevents wind from removing more energy than
is locally available. With `wind_activation_width_fraction=0`, the old hard
threshold is recovered.

Pilot settings:

```text
Mdot_inner/Edd = 5
Rout = 335 rg
f_s = 0.80
eta_heat = 0
chi_edd = 0.99
N = 896
```

Accepted rows:

| width fraction | epsilon_w | final full | int Qwind/Qvisc | active frac | nfev | elapsed |
|---:|---:|---:|---:|---:|---:|---:|
| 0.001 | 0.03 | 2.637e-09 | 1.471e-04 | 1.000 | 5 | 22.01 s |
| 0.005 | 0.03 | 5.076e-09 | 1.570e-04 | 1.000 | 8 | 39.48 s |
| 0.005 | 0.04 | 3.175e-09 | 2.113e-04 | 1.000 | 30 | 94.35 s |

Outputs:

- `outputs/tables/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w001_eta0_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_eps004_N896.md`
- `outputs/checkpoints/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w001_eta0_N896/`
- `outputs/checkpoints/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_N896/`
- `outputs/checkpoints/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_eps004_N896/`

Interrupted rows:

- `width=0.001`, `epsilon_w=0.05` from `epsilon_w=0.03`.
- `width=0.005`, `epsilon_w=0.05` from `epsilon_w=0.03`.
- `width=0.005`, `epsilon_w=0.05` from the accepted `epsilon_w=0.04`
  checkpoint.

Interpretation:

- Smooth activation is implemented and behaves consistently.
- `width=0.001` is effectively too close to the hard switch to help.
- `width=0.005` allows a tighter staged continuation from `epsilon_w=0.03` to
  `0.04`, but the next step to `0.05` remains too expensive with the current
  finite-difference global Newton path.
- Therefore the present bottleneck is not simply the nondifferentiable `max`.
  It is the cost/conditioning of the high-N corrector once active wind feedback
  becomes moderately strong.

Recommended next step:

1. Implement a true `epsilon_w` tangent predictor using the square Newton
   Jacobian:

   ```text
   J_z dz/d epsilon_w = -F_epsilon
   ```

2. Use it for small staged continuation:
   `epsilon_w = 0.04, 0.045, 0.05, 0.06, 0.07, 0.10`.
3. If tangent prediction still requires costly correctors, add analytic/local
   derivatives for the wind contribution to `interval_E`, or use lower-N
   continuation with residual-aware remap back to `N=896`.
4. Treat all current wind results as perturbative; integrated `Qwind/Qvisc` is
   still only `~2e-4` at `epsilon_w=0.04`, so it is not yet a dynamically strong
   wind/hot branch.

## Follow-Up: Epsilon-Wind Tangent Predictor

Implemented:

- `finite_difference_energy_wind_epsilon_column(...)`
- `energy_wind_epsilon_tangent(...)`
- `energy_wind_epsilon_seed(...)`
- `run_energy_wind_branch(...)` now compares current, secant, and tangent
  predictor seeds for `wind_energy_limited_epsilon`.

The tangent solves

```text
J_z dz/d epsilon_w = -F_epsilon
```

using the same square collocation Jacobian and equilibrated tangent solver used
for source-fraction continuation. The existing predictor metadata columns now
apply to energy-wind continuation too.

Pilot settings:

```text
Mdot_inner/Edd = 5
Rout = 335 rg
f_s = 0.80
eta_heat = 0
chi_edd = 0.99
wind_activation_width_fraction = 0.005
N = 896
```

Accepted strict/tight rows:

| epsilon_w | predictor | initial current | best initial | final full | int Qwind/Qvisc | nfev | elapsed |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.045 | tangent:1 | 3.448e-05 | 1.975e-06 | 9.234e-09 | 2.388e-04 | 19 | 53.34 s |
| 0.050 | secant:1 | 3.463e-05 | 3.136e-07 | 9.366e-09 | 2.665e-04 | 136 | 206.4 s |

Outputs:

- `outputs/tables/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_tangent_0045_005_N896.md`
- `outputs/checkpoints/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_tangent_0045_005_N896/`

Direct-linear-solver scouts:

| epsilon_w | predictor | best initial | final full | accepted | strict anchor | int Qwind/Qvisc | note |
|---:|---|---:|---:|:---:|:---:|---:|---|
| 0.055 | tangent:1 | 2.588e-06 | 3.934e-06 | yes | no | 2.945e-04 | accepted loose scout |
| 0.060 | tangent:1 | 4.678e-06 | 4.678e-06 | yes | no | 3.227e-04 | no residual improvement |
| 0.070 | tangent:1 | 1.131e-05 | 1.131e-05 | no | no | 3.799e-04 | rejected |

Outputs:

- `outputs/tables/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_tangent_direct_0055_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_tangent_direct_006_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_tangent_direct_007_N896.md`
- `outputs/checkpoints/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_tangent_direct_0055_N896/`
- `outputs/checkpoints/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_tangent_direct_006_N896/`
- `outputs/checkpoints/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_tangent_direct_007_N896/`

Interpretation:

- The predictor problem is mostly solved. At `epsilon_w=0.05`, the current seed
  residual was `3.46e-05`, the tangent seed was `2.26e-06`, and the secant seed
  was `3.14e-07`.
- The remaining bottleneck is the corrector/polish step. For `epsilon_w=0.05`,
  Newton still required `136` function evaluations despite an excellent seed.
- Above `epsilon_w=0.05`, the direct-linear-solver scouts can carry the branch
  as loose accepted states to `0.055` and `0.060`, but they do not polish to
  strict residual. At `0.070`, the tangent seed is outside acceptance.
- The dominant residual remains `interval_E`, with peak around the
  outer-buffer/source-tail region (`R ~= 255-270 rg` in these rows), not a sonic
  failure.
- The wind remains perturbative: even at `epsilon_w=0.060`, integrated
  `Qwind/Qvisc ~= 3.2e-4`. This is not yet dynamically strong enough to claim a
  wind-regulated hot branch.

Recommended next step:

1. Focus on the corrector, not the predictor.
2. Add analytic/local derivative support for the wind contribution to the
   energy interval residual, or cache/reuse the square Jacobian between tiny
   epsilon steps.
3. Add a targeted outer-buffer/source-tail energy patch for the
   `R ~= 255-270 rg` interval_E residual.
4. Treat `epsilon_w=0.05` as the current strict self-consistent soft-wind anchor;
   treat `0.055-0.060` as loose scouts only until they polish tighter.

## Follow-Up: Triggered Local Energy Patch

Implemented:

- Added `IMBH_STANDARD_SLIM_STREAM_MASS_LOCAL_PATCH_TRIGGER_TOL`.
- Local physical/global patching can now be triggered for accepted-but-not-strict
  rows, not only rows rejected by the loose acceptance gate.
- The patch mode switch now uses the stricter trigger tolerance when supplied,
  so an accepted loose row can still enter global patch mode while trying to
  become a strict anchor.

Settings for the focused follow-up runs:

```text
Mdot_inner/Edd = 5
Rout = 335 rg
R_buffer = 300 rg
f_s = 0.80
eta_heat = 0
torque_delta_l_fraction = 0.005
chi_edd = 0.99
wind_activation_width_fraction = 0.005
N = 896
strict trigger = 3e-6
```

New strict/loose sequence:

| epsilon_w | predictor | final full | strict anchor | peak interval_E R/rg | int Qwind/Qvisc | f_adv_global | Rson/rg | note |
|---:|---|---:|:---:|---:|---:|---:|---:|---|
| 0.060 | current + 3 patch passes | 1.769e-06 | yes | 334.85 | 3.227e-04 | 0.4987 | 4.3615 | old loose row polished strict |
| 0.065 | tangent:1 + patch | 2.916e-06 | yes | 65.76 | 3.512e-04 | 0.4987 | 4.3615 | current strict ceiling |
| 0.066 | tangent:1 | 3.141e-06 | no | 65.76 | 3.569e-04 | 0.4987 | 4.3615 | just above strict tolerance |
| 0.0675 | tangent:1 + patch | 3.500e-06 | no | 65.76 | 3.655e-04 | 0.4987 | 4.3615 | loose accepted |
| 0.070 | tangent:1 + patch | 4.156e-06 | no | 65.76 | 3.799e-04 | 0.4987 | 4.3615 | loose accepted |

Negative tests at/above the new ceiling:

- `epsilon_w=0.070` from the strict `0.065` anchor with
  `energy_jacobian_rel_step=1e-6` reproduced the same `4.156e-06` residual.
- Alternative cleanup polish with `integrated_physical_energy`,
  `conservative_physical_energy`, and `differential` returned to the same
  `4.156e-06` state.
- Global-mode local patch from the loose `0.070` state did not adopt:
  `4.156e-06 -> 6.312e-06`.
- A wider/aggressive global patch reduced the physical energy row slightly
  but worsened the full residual: `4.156e-06 -> 4.762e-06`.
- Energy-merit line search at `epsilon_w=0.0675` reproduced the same
  `3.500e-06` loose state.

Outputs:

- `outputs/tables/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_patch4_006_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_tangent_patch_0065_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_tangent_patch_0066_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_tangent_patch_00675_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_tangent_patch_007_from0065_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_globalpatch_007_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_aggrpatch_007_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_cleanup_007_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_energymerit_00675_N896.md`

Interpretation:

- The targeted local patch solved the previous loose `epsilon_w=0.060` row and
  advanced the strict self-consistent soft-wind anchor to `epsilon_w=0.065`.
- The remaining wall is no longer an outer-buffer/source-tail residual. The
  dominant interval_E peak moved inward to `R ~= 65.76 rg`.
- The branch diagnostics are smooth across the loose rows: `f_adv_global ~=
  0.499`, `f_adv_inner ~= 0.471`, `Rson ~= 4.3615 rg`, and the wind power remains
  perturbative with `int Qwind/Qvisc < 4e-4`.
- Therefore the current obstruction is a strict energy-collocation/corrector
  floor near the advective transition, not a sonic failure and not evidence for
  a physical branch endpoint.

Recommended next step:

1. Do not tune the local patch further unless the strict tolerance is relaxed.
2. Implement a real analytic/local Jacobian block for the active wind
   contribution to `interval_E`, including the smooth activation derivative.
3. If that still stalls, move to a formulation change: finite-volume energy
   collocation or a mesh-local refinement around `R ~= 65.8 rg`.
4. Treat `epsilon_w=0.065` as the current strict wind anchor; treat
   `0.066-0.070` as loose scouts only.

## Follow-Up: Wind-Aware Interval-Energy Jacobian

Implemented:

- Added `energy_limited_wind_derivatives(...)`, returning
  `dQ_wind/dQ_avail` and `dQ_wind/dQ_edd` for both hard and smooth activation.
- Updated the local differential matrix so active energy-limited wind uses the
  analytic local slope derivative of the wind term, instead of a unit-slope
  secant through the nonlinear activation.
- Added a hybrid wind-aware interval-energy Jacobian for
  `interval_residual_form="differential"`:
  - finite-difference the smooth no-wind energy row;
  - apply the wind activation derivative analytically through `Q_avail` and
    `Q_edd`;
  - leave the old no-wind path unchanged.
- Added regression tests for the wind derivative and active-wind local matrix
  derivative.

This directly removes the earlier `interval_E` floor near `R ~= 65.8 rg`.

Strict ladder after the Jacobian fix:

| epsilon_w | predictor | initial full | final full | int Qwind/Qvisc | f_adv_global | f_adv_inner | Rson/rg | nfev |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.0675 | tangent:1 | 2.953e-06 | 9.967e-10 | 3.657e-04 | 0.4987 | 0.4713 | 4.3615 | 2 |
| 0.070 | tangent:1 | 4.143e-08 | 1.719e-10 | 3.801e-04 | 0.4987 | 0.4713 | 4.3615 | 2 |
| 0.080 | tangent:1 | 6.675e-07 | 1.779e-10 | 4.386e-04 | 0.4987 | 0.4712 | 4.3615 | 2 |
| 0.090 | tangent:1 | 6.830e-07 | 1.872e-10 | 4.982e-04 | 0.4986 | 0.4712 | 4.3616 | 2 |
| 0.100 | tangent:1 | 6.991e-07 | 1.997e-10 | 5.589e-04 | 0.4986 | 0.4711 | 4.3616 | 2 |
| 0.150 | tangent:1 | 1.798e-05 | 9.414e-10 | 8.819e-04 | 0.4984 | 0.4709 | 4.3617 | 2 |
| 0.200 | tangent:1 | 2.031e-05 | 1.590e-09 | 1.241e-03 | 0.4981 | 0.4706 | 4.3619 | 2 |
| 0.300 | tangent:1 | 9.281e-05 | 1.474e-10 | 2.101e-03 | 0.4976 | 0.4699 | 4.3623 | 3 |
| 0.500 | tangent:1 | 4.942e-04 | 1.624e-10 | 4.792e-03 | 0.4958 | 0.4679 | 4.3635 | 3 |
| 0.800 | tangent:1 | 2.203e-03 | 1.592e-10 | 1.875e-02 | 0.4868 | 0.4571 | 4.3699 | 4 |
| 0.900 | tangent:1 | 1.504e-03 | 1.705e-10 | 4.145e-02 | 0.4720 | 0.4396 | 4.3805 | 4 |
| 0.950 | tangent:1 | 1.466e-03 | 1.716e-10 | 8.453e-02 | 0.4441 | 0.4049 | 4.4015 | 4 |
| 0.980 | tangent:1 | 2.732e-03 | 2.157e-10 | 1.976e-01 | 0.3710 | 0.3140 | 4.4623 | 5 |

Outputs:

- `outputs/tables/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_windjac_00675_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_windjac_007_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_windjac_008_010_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_windjac_015_030_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_windjac_05_10_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_windjac_09_10_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_windjac_095_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_windjac_098_N896.md`

Interpretation:

- The previous strict ceiling at `epsilon_w ~= 0.065` was numerical, not
  physical. It was caused by the active-wind contribution to `interval_E` being
  poorly represented in the local corrector/Jacobian.
- The corrected path is strict through `epsilon_w=0.98`. Near this point the
  wind term is no longer microscopic: `int Qwind/Qvisc ~= 0.20`, and the
  advection diagnostics respond smoothly (`f_adv_global` decreases from
  `~0.50` to `~0.37`).
- Exact `epsilon_w=1.0` remains expensive because the predictor falls back to
  the current state at the hard upper parameter boundary. Attempts from
  `0.8`, `0.9`, and `0.98` started with residuals `~0.04-0.05` and were
  interrupted after slow progress. This looks like a boundary-predictor issue,
  not a branch failure.

Recommended next step:

1. Treat `epsilon_w=0.98` as the current robust near-max wind anchor.
2. Add a one-sided/boundary-aware epsilon tangent predictor before spending more
   time on exactly `epsilon_w=1.0`.
3. Start scientific validation of the high-epsilon wind branch: N checks,
   residual localization, luminosity/energy-budget audit, and source/closure
   sensitivity.
4. Only then decide whether this self-consistent wind sink is enough to claim a
   true hot/wind branch, or whether stream heating and/or mass loss must be
   added.

## Follow-Up: Endpoint Staging Toward Epsilon = 1

Implemented:

- Added `IMBH_STANDARD_SLIM_STREAM_MASS_ENERGY_WIND_PREV_ANCHOR`.
- The energy-wind branch can now seed its first step with a previous checkpoint,
  giving the first row a true one-sided secant predictor even when the script is
  launched from an already-polished current anchor.

Tests from `epsilon_w=0.98`:

- Direct `0.98 -> 1.0` with no previous anchor:
  - predictor stayed `current`;
  - initial residual `5.467e-02`;
  - tangent best was worse (`5.708e-02`).
- Direct `0.98 -> 1.0` with `0.95` loaded as previous anchor:
  - secant best was also worse (`5.563e-02`);
  - tangent and secant were almost parallel (`cosine ~= 0.99995`).

Finer endpoint staging:

| epsilon_w | predictor | initial full | final full | int Qwind/Qvisc | f_adv_global | f_adv_inner | Rson/rg | note |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 0.990 | tangent:0.5 | 7.255e-03 | 6.526e-10 | 3.447e-01 | not re-tabulated | not re-tabulated | not re-tabulated | strict |
| 0.995 | tangent:0.5 | 2.017e-02 | 1.669e-10 | 5.407e-01 | not re-tabulated | not re-tabulated | not re-tabulated | strict |
| 0.997 | tangent:0.5 | 4.150e-02 | 1.206e-09 | 6.898e-01 | not re-tabulated | not re-tabulated | not re-tabulated | strict |
| 0.998 | tangent:0.5 | 1.146e-01 | interrupted | - | - | - | - | too costly with raw epsilon |

Outputs:

- `outputs/tables/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_predictor_audit_10_from098_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_predictor_audit_secant_10_from098_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_windjac_099_10_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_windjac_0997_0999_N896.md`

Interpretation:

- The branch is now strict at least to `epsilon_w=0.997`, with
  `int Qwind/Qvisc ~= 0.69`. This is qualitatively stronger than the earlier
  perturbative wind tests.
- The exact endpoint is not blocked by the old `interval_E` Jacobian issue.
  Instead, raw `epsilon_w` becomes a poor continuation coordinate as
  `epsilon_w -> 1`.
- A one-sided secant in raw epsilon is not enough near the endpoint; the tangent
  and secant directions are nearly parallel and both worsen the exact `1.0`
  seed.

Recommended next step:

1. Treat `epsilon_w=0.997` as the current strict endpoint anchor.
2. If exact `epsilon_w=1.0` is scientifically important, switch continuation
   coordinate from `epsilon_w` to an endpoint variable such as
   `eta = -log(1 - epsilon_w)` or use pseudo-arclength continuation in
   `(z, epsilon_w)`.
3. Before chasing exactly `1.0`, validate the `0.98-0.997` high-wind states
   with N checks and energy-budget audits.

## Follow-Up: Eta Endpoint Coordinate

Implemented:

- Added `IMBH_STANDARD_SLIM_STREAM_MASS_ENERGY_WIND_ETAS`.
- When eta targets are supplied, the script converts

  ```text
  epsilon_w = 1 - exp(-eta)
  ```

  for the physics solve, but uses eta as the continuation coordinate in both
  the secant and tangent predictors.
- The tangent predictor now applies

  ```text
  dz/deta = dz/depsilon_w * (1 - epsilon_w)
  ```

  near the endpoint.
- Existing `IMBH_STANDARD_SLIM_STREAM_MASS_ENERGY_WIND_EPSILONS` behavior is
  unchanged.

Eta continuation from the strict `epsilon_w=0.997` anchor:

| eta | epsilon_w | predictor | initial full | final full | int Qwind/Qvisc | f_adv_global | f_adv_inner | Rson/rg | nfev |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 5.90 | 0.997260555 | tangent:1 | 6.880e-03 | 1.928e-10 | 0.7139 | 0.0476 | -0.1091 | 4.979 | 7 |
| 6.00 | 0.997521248 | tangent:1 | 9.169e-03 | 1.359e-10 | 0.7389 | 0.0337 | -0.1238 | 5.031 | 5 |
| 6.10 | 0.997757132 | tangent:1 | 1.003e-02 | 1.872e-10 | 0.7622 | 0.0214 | -0.1382 | 5.085 | 7 |
| 6.20 | 0.997970569 | tangent:1 | 1.305e-02 | 3.677e-10 | 0.7834 | 0.0107 | -0.1468 | 5.141 | 8 |
| 6.30 | 0.998163695 | tangent:1 | 2.632e-02 | 1.459e-10 | 0.8024 | 0.0018 | -0.1515 | 5.198 | 6 |
| 6.35 | 0.998253253 | tangent:1 | 9.983e-03 | 1.593e-10 | 0.8110 | -0.0019 | -0.1515 | 5.227 | 4 |

Attempted `eta=6.40` from the `eta=6.30` anchor:

- `epsilon_w = 0.998338`
- initial residual `8.485e-02`
- interrupted as too large/expensive for the current fixed-step controller.
- A smaller step to `eta=6.35` succeeded, so this is step-size/end-point
  stiffness rather than a clean branch loss.

Outputs:

- `outputs/tables/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_eta_590_620_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_eta_630_650_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_eta_635_N896.md`

Interpretation:

- Eta is a better endpoint coordinate than raw epsilon. It extended the strict
  branch from `epsilon_w=0.997` to `epsilon_w=0.998253`.
- The endpoint remains stiff: fixed eta steps of `0.1` become too aggressive
  around `eta ~= 6.4`. An adaptive eta step controller or pseudo-arclength
  should be used before pushing closer to `epsilon_w=1`.
- The high-wind solution changes character near this endpoint:
  `int Qwind/Qvisc` rises to `~0.81`, `Rson` moves outward to `~5.23 rg`, and
  the global/inner advection diagnostics cross toward negative values. This is
  potentially important, but it should not be over-interpreted until mesh and
  energy-budget validation are done.

Recommended next step:

1. Implement adaptive eta stepping with rejection/shrink based on initial
   residual and corrector cost.
2. Validate the high-wind states at `eta=6.2-6.35` with N checks and residual
   localization.
3. If adaptive eta still struggles, switch to pseudo-arclength in
   `(z, eta)` or `(z, epsilon_w)`.
