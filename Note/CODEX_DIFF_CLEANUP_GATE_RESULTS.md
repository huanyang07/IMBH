# Differential Cleanup Physical-Gate Test

Date: 2026-07-03

## Purpose

Test whether a targeted differential-form cleanup pass can move the clean no-wind compact stream branch beyond the previous physical-audit frontier at `f_s=0.8980`.

This follows the physical gate automation note:

- weighted solver residuals alone are not enough near the wall;
- a row is only accepted when the raw physical-domain differential energy residual satisfies `physical_E <= 3e-5`;
- rejected rows can be written with lean diagnostics to avoid spending full profile diagnostic time after the gate has already failed.

## Code Changes

Updated `scripts/run_standard_slim_stream_mass_annulus_scan.py` with opt-in controls:

- `IMBH_STANDARD_SLIM_STREAM_MASS_CLEANUP_POLISH_SPECS`
  - examples: `same`, `differential:none`, `integrated:inverse_dx`;
  - cleanup tries these residual forms after the base polish and adopts only if the raw physical energy residual improves without losing solver acceptance.
- `IMBH_STANDARD_SLIM_STREAM_MASS_LEAN_REJECT_DIAGNOSTICS`
  - rejected rows keep the residual, partition, mass-budget, and continuation fields;
  - expensive profile/advection/luminosity diagnostics and Newton-audit writes are skipped for rows that already fail the gate.

Regression tests:

```text
146 passed
```

## Run

Anchor:

```text
outputs/checkpoints/high_mdot_stream_outer_buffer_repolish0898_more/repolish0898_more_mass_0p898_torque_0p005_mdot_2_N896.npz
```

Main settings:

```text
Mdot_inner/Edd = 2
Rout = 335 rg
Rinj = 240 rg
outer buffer inner = 300 rg
N = 896
source shape = compact_c2
torque_delta_l_fraction = +0.005
interval solve form = integrated:none
physical_E gate = 3e-5
cleanup polish specs = differential:none
cleanup passes = 1
adaptive step = 5e-4, shrink to 6.25e-5
```

Outputs:

```text
outputs/tables/high_mdot_stream_outer_buffer_phys_gate_diff_cleanup_0898_to090.md
outputs/figures/high_mdot_stream_outer_buffer_phys_gate_diff_cleanup_0898_to090.png
outputs/checkpoints/high_mdot_stream_outer_buffer_phys_gate_diff_cleanup_0898_to090/
```

## Results

| f_s | solver full | raw physical_E | gate result |
|---:|---:|---:|---|
| 0.8985000 | 7.594e-08 | 3.883e-05 | rejected |
| 0.8982500 | 6.609e-08 | 3.379e-05 | rejected |
| 0.8981250 | 6.028e-08 | 3.082e-05 | rejected |
| 0.8980625 | 5.738e-08 | 2.934e-05 | accepted strict anchor |
| 0.8981250 retry | 6.155e-08 | 3.147e-05 | rejected |

The differential cleanup pass was attempted on all rows, but it was not adopted by the physical-state selector. It did not materially reduce the raw physical energy residual above the wall.

The only new clean extension is very small:

```text
f_s = 0.8980625
```

It has stable physical diagnostics:

```text
f_adv_global = 0.2043
f_adv_inner  = 0.09443
Lrad/LEdd    = 0.8666
max H/R      = 0.2269
Rson         = 4.66 rg
```

## Interpretation

The obstruction is not predictor quality:

- the tangent seeds are already tiny, down to `~8e-8`;
- the weighted solver residual is strict for all attempted rows;
- the raw physical-domain energy residual crosses the `3e-5` threshold almost linearly with `f_s`.

The obstruction is also not fixed by a simple differential-form cleanup pass. With this grid and outer-buffer formulation, the clean physical-audit frontier is approximately:

```text
0.89806 < f_s_clean < 0.898125
```

This points to a discretization/formulation issue in the physical energy residual near the outer/source-buffer transition, not to a physical branch endpoint.

## Recommended Next Step

Do not push to larger `f_s` by relaxing the physical gate unless the goal is only exploratory.

For a robust branch, the next implementation target should be one of:

1. add a physical-energy-targeted residual form for the square Newton solve, so the solver minimizes the same differential physical audit used by the gate;
2. add residual-adaptive local refinement around the peak physical `interval_E` radius near `R ~ 259 rg`, then repolish and test whether the `f_s=0.898125` physical residual drops below `3e-5`;
3. implement an early-stop/cost-aware cleanup rule so differential cleanup is only continued when it actually decreases the raw physical energy residual after the first Newton step.

The best immediate scientific/numerical test is option 2: refine specifically around the physical `interval_E` peak and rerun `f_s=0.898125`.
