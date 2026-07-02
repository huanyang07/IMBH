# Codex Stream Residual-Remesh Results

Date: 2026-07-02

## Scope

This update implements the first two items from the latest GPT plan:

1. Freeze regression anchors for the standard no-wind and stream-fed checkpoints.
2. Add residual-based remeshing for the stream-fed outer/source-tail problem.

It also performs the first robustness matrix requested by GPT:

```text
f_s = 0.70, 0.80, 0.805, 0.808585
N   = 640, 768, 896
```

All tests below use:

```text
Mdot_inner/Edd = 2
Rout = 300 rg
stream_torque_delta_l_fraction = +0.005
stream source centered at Rinj/Rout = 0.80
stream source log width = 0.08
no wind
no stream heating
```

except for the explicit no-wind and no-torque regression anchors.

## Code Added

New scripts:

```text
scripts/run_standard_slim_stream_anchor_regression.py
scripts/run_standard_slim_stream_residual_remesh.py
```

Also updated:

```text
scripts/run_standard_slim_stream_mass_annulus_scan.py
```

The continuation script now preserves checkpoint metadata for:

```text
stream source geometry
stream torque geometry
wind source/sink fields
stream heating efficiency
```

This avoids accidentally changing the physical setup when continuing from a checkpoint without matching environment variables.

## Regression Anchors

Output:

```text
outputs/tables/standard_slim_stream_regression_anchors.md
outputs/tables/standard_slim_stream_regression_anchors.json
```

Results:

| anchor | residual | dominant | Mdot_outer/Mdot_inner | f_adv_global | f_adv_inner | Lrad/LEdd | max H/R | Rson/rg | strict? |
|---|---:|---|---:|---:|---:|---:|---:|---:|:---:|
| Mdot=5 no-wind large-R | 2.293e-6 | interval_E | 1.000 | 0.4534 | 0.4666 | 1.541 | 0.3164 | 4.360 | yes |
| Mdot=2 Rout=300 no-stream | 4.441e-6 | outer_omega | 1.000 | 0.1995 | 0.0931 | 0.8844 | 0.2269 | 4.660 | no |
| Mdot=2 f_s=0.50 no-torque | 1.743e-6 | outer_omega | 0.5019 | 0.2026 | 0.0946 | 0.8735 | 0.2269 | 4.660 | yes |
| Mdot=2 f_s=0.80 torque +0.005 | 3.756e-7 | outer_omega | 0.2030 | 0.2039 | 0.0946 | 0.8675 | 0.2269 | 4.660 | yes |
| Mdot=2 f_s=0.808585 torque +0.005 | 8.532e-8 | interval_E | 0.1945 | 0.2039 | 0.0954 | 0.8673 | 0.2269 | 4.660 | yes |

The only non-strict anchor is the finite-Rout no-stream Mdot=2 checkpoint, which is still accepted at residual 4.44e-6. This matches the earlier caveat and is not a new regression.

## Residual-Based Remeshing

The monitor is:

```text
M = floor + strength * normalize(
      w_E      * |interval_E|
    + w_source * stream_source_prime
    + w_source * wind_sink_prime
    + w_mdot   * |dMdot/dlnR|
    + w_Q      * |dQstream/dlnR|
    + w_outer  * outer_boundary_layer
)^power
```

Default run settings:

```text
strength = 12
blend = 0.75
power = 0.5
reference grid = current checkpoint grid
outer boundary width = 0.018 in xi
```

The code writes tables, checkpoints, and dense monitor profiles. Example output files:

```text
outputs/tables/standard_slim_stream_residual_remesh_fs070_N640_N768_N896_s12.md
outputs/tables/standard_slim_stream_residual_remesh_fs080_N640_N768_N896_s12.md
outputs/tables/standard_slim_stream_residual_remesh_fs0805_N640_N768_N896_s12.md
outputs/tables/standard_slim_stream_residual_remesh_fs0808585_N640_s12.md
outputs/tables/standard_slim_stream_residual_remesh_fs0808585_N768_N896_s12.md
```

## Robustness Matrix

All residual-remeshed cases are strict anchors.

| f_s | N | final residual | dominant | nfev | outer 1% nodes | outer 5% nodes | source integral delta | f_adv_global | f_adv_inner | Lrad/LEdd | max H/R | Rson/rg |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.70 | 640 | 7.508e-9 | outer_omega | 2 | 32 | 144 | +8.35e-6 | 0.2035 | 0.0947 | 0.8694 | 0.2269 | 4.6599 |
| 0.70 | 768 | 2.500e-9 | outer_omega | 1 | 38 | 172 | +1.59e-6 | 0.2035 | 0.0954 | 0.8694 | 0.2269 | 4.6599 |
| 0.70 | 896 | 2.621e-8 | outer_omega | 2 | 44 | 201 | -2.49e-6 | 0.2035 | 0.0960 | 0.8694 | 0.2269 | 4.6599 |
| 0.80 | 640 | 8.498e-7 | interval_E | 79 | 24 | 118 | +2.99e-5 | 0.2039 | 0.0945 | 0.8675 | 0.2269 | 4.6599 |
| 0.80 | 768 | 3.464e-8 | outer_omega | 5 | 29 | 141 | +1.60e-5 | 0.2039 | 0.0947 | 0.8675 | 0.2269 | 4.6599 |
| 0.80 | 896 | 2.247e-8 | outer_omega | 5 | 34 | 165 | +7.57e-6 | 0.2039 | 0.0949 | 0.8675 | 0.2269 | 4.6599 |
| 0.805 | 640 | 3.654e-7 | outer_omega | 86 | 35 | 152 | +3.83e-6 | 0.2039 | 0.0953 | 0.8674 | 0.2269 | 4.6599 |
| 0.805 | 768 | 9.606e-8 | outer_omega | 14 | 42 | 183 | -2.18e-6 | 0.2039 | 0.0957 | 0.8674 | 0.2269 | 4.6599 |
| 0.805 | 896 | 8.254e-8 | outer_omega | 14 | 49 | 213 | -5.81e-6 | 0.2039 | 0.0960 | 0.8674 | 0.2269 | 4.6599 |
| 0.808585 | 640 | 2.937e-8 | outer_omega | 5 | 41 | 170 | +1.34e-5 | 0.2039 | 0.0947 | 0.8673 | 0.2269 | 4.6599 |
| 0.808585 | 768 | 3.165e-8 | outer_omega | 5 | 49 | 204 | +8.70e-6 | 0.2039 | 0.0952 | 0.8673 | 0.2269 | 4.6599 |
| 0.808585 | 896 | 2.696e-8 | outer_omega | 5 | 57 | 238 | +5.86e-6 | 0.2039 | 0.0955 | 0.8673 | 0.2269 | 4.6599 |

## Interpretation

The residual-remeshed grids remove the single-cell outer interval_E wall at the current front. The previous `f_s≈0.8086` state now repolishes cleanly at `N=640,768,896`, with final residuals near `3e-8` and stable physical diagnostics.

The result strongly supports GPT's diagnosis:

```text
The f_s≈0.8086 wall was primarily a mesh/source-tail/boundary-layer problem,
not a sonic failure and not evidence for physical branch death.
```

The branch remains only mildly advective:

```text
f_adv_global ≈ 0.204
f_adv_inner ≈ 0.095
max H/R ≈ 0.227
Rson ≈ 4.66 rg
```

So this is a robust finite stream-fed branch at Mdot_inner/Edd=2, but it is not yet the genuinely hot Mdot=3-5 stream-fed branch.

## Caveats

1. The residual-remesh monitor still uses the current checkpoint grid as a blended reference. A later audit should compare against `REFERENCE=power` or `REFERENCE=uniform`.
2. `f_s=0.80` and `f_s=0.805` at N=640 still needed many function evaluations, even though the final residuals are strict. This means the predictor/outer-closure cost problem is not fully solved.
3. The source-integral drift is small, at most about `3e-5` of inner Mdot in this matrix, but source normalization should continue to be reported.
4. These tests do not yet include soft/Robin outer angular closure or source-shape dependence.

## Verification

```text
PYTHONPATH=src:scripts python -m py_compile \
  scripts/run_standard_slim_stream_mass_annulus_scan.py \
  scripts/run_standard_slim_stream_anchor_regression.py \
  scripts/run_standard_slim_stream_residual_remesh.py

PYTHONPATH=src:scripts python -m pytest
```

Result:

```text
140 passed in 2.70s
```

## Recommended Next Step

Implement the true source-fraction tangent predictor:

```text
J_z dz/df_s = -F_f_s
```

Then compare current-state, secant, and tangent seeds at:

```text
f_s = 0.80 -> 0.805
f_s = 0.805 -> 0.8085
f_s = 0.8085 -> 0.81
```

Use the residual-remeshed grid as the default mesh. If tangent cost still grows or singular values drop near the front, then add pseudo-arclength. Soft/Robin outer closure remains the next physics/numerics improvement after the tangent predictor.
