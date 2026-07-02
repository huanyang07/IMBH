# Compact C2 Source-Shape Homotopy Results

Generated after testing a no-tail compact stream source at the current
Mdot_inner/Edd = 2, Rout = 300 rg, f_s = 0.82, torque_delta_l_fraction = +0.005
standard no-wind stream branch.

## Implementation

- Added opt-in `stream_source_shape` to `TransonicSlimParams`.
  - Default remains `tanh`.
  - New option: `compact_c2`.
- Added `stream_source_shape_blend` in `[0, 1]`.
  - `0` is the original tanh cumulative source.
  - `1` is the compact C2 cumulative source.
  - Intermediate values are normalized blends of the two cumulative profiles,
    so the integrated source fraction is preserved.
- Added checkpoint/table metadata for source shape and blend.
- Added unit coverage for the compact source:
  - no derivative tail at Rout,
  - correct center derivative normalization,
  - integrated source equals the requested fraction.

## Verification

Command:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/imbh_pycache PYTHONPATH=src:scripts \
  /Users/huanyang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest
```

Result:

```text
141 passed in 2.49s
```

## Key numerical findings

Directly replacing the tanh source by full compact C2 at f_s = 0.82 gave a huge
initial residual on the old state:

```text
blend=1 seed residual ~1.75e1
```

A source-shape homotopy on the original residual-remeshed N=768 grid succeeded:

| source blend | final residual | accepted | dominant | Mdot_outer/Mdot_inner | notes |
|---:|---:|:---:|---|---:|---|
| 0.001 | 8.238e-08 | yes | outer_omega | 0.18308 | strict |
| 0.005 | 2.202e-07 | yes | outer_omega | 0.18307 | strict |
| 0.02 | 1.003e-06 | yes | outer_omega | 0.18302 | strict |
| 0.05 | 2.198e-06 | yes | outer_omega | 0.18293 | strict |
| 0.10 | 3.839e-06 | yes | outer_omega | 0.18278 | accepted |
| 0.25 | 7.882e-06 | yes | outer_omega | 0.18231 | accepted |
| 0.40 | 4.200e-06 | yes | outer_omega | 0.18185 | accepted via smaller steps |
| 0.60 | 7.885e-06 | yes | outer_omega | 0.18123 | accepted |
| 0.75 | 6.445e-06 | yes | outer_omega | 0.18077 | accepted via smaller steps |
| 0.93 | 6.328e-06 | yes | outer_omega | 0.18022 | accepted |
| 1.00 | 5.693e-06 | yes | outer_omega | 0.18000 | full compact C2 |

The final full-compact run:

- checkpoint:
  `outputs/checkpoints/high_mdot_stream_source_compact_full_origgrid_fs082_N768/compact_full_origgrid_mass_0p82_torque_0p005_mdot_2_N768.npz`
- table:
  `outputs/tables/high_mdot_stream_source_compact_full_origgrid_fs082_N768.md`
- figure:
  `outputs/figures/high_mdot_stream_source_compact_full_origgrid_fs082_N768.png`

Final diagnostics:

```text
final_full = 5.693e-06
dominant = outer_omega
interval_R = 3.813e-08
interval_E = 5.674e-06
Mdot_outer/Mdot_inner = 0.18
source_integral/Mdot_inner = 0.82
f_adv_global = 0.2042
f_adv_inner = 0.09517
max H/R = 0.2269
Rson = 4.66 rg
```

## Important caveat

Using the `annulus_outer` source-grid remap during the tiny shape-blend tests was
counterproductive. For example, blend = 0.001 on the remapped grid stalled near
residual ~1.1e-3, while the same blend on the original residual-remeshed grid
reached 8.238e-08. This means the compact-source experiment should be separated
from grid remapping unless remapping is done with a dedicated defect-preserving
strategy.

Follow-up mesh checks sharpen this caveat:

- Plain N=640 power-grid remap from the compact endpoint failed:
  `final_full = 2.039e-03`, dominated by `interval_E` near `R~299.4 rg`.
- PCHIP remapping was tested as an opt-in smoother remap, but it made the
  initial defects worse for this endpoint:
  - N=640 power grid: linear `2.17e-2`, PCHIP `4.17e-2`;
  - N=896 power grid: linear `1.98e-2`, PCHIP `4.83e-2`.
- Resampling the existing custom grid distribution alone also did not cure the
  problem cheaply.
- Residual-aware remeshing did cure it.

Residual-remeshed mesh validation at the full compact endpoint:

| N | final residual | strict | dominant | source integral | rel budget err | f_adv global | f_adv inner | Lrad/LEdd | max H/R | Rson/rg |
|---:|---:|:---:|---|---:|---:|---:|---:|---:|---:|---:|
| 640 | 1.137e-06 | yes | interval_E | 0.820051 | 5.09e-05 | 0.204190 | 0.09491 | 0.867201 | 0.226899 | 4.659917 |
| 768 | 5.693e-06 | no | outer_omega | 0.820016 | 1.59e-05 | 0.204189 | 0.09517 | 0.867199 | 0.226899 | 4.659918 |
| 896 | 1.043e-06 | yes | outer_omega | 0.820026 | 2.59e-05 | 0.204193 | 0.09471 | 0.867198 | 0.226900 | 4.659919 |

Mesh-validation outputs:

- N=640 residual-remesh table:
  `outputs/tables/high_mdot_stream_source_compact_full_residual_remesh_N640_s12.md`
- N=640 residual-remesh checkpoint:
  `outputs/checkpoints/high_mdot_stream_source_compact_full_residual_remesh_N640_s12/N640_s12_mass_0p82_torque_0p005_mdot_2_N640.npz`
- N=896 residual-remesh table:
  `outputs/tables/high_mdot_stream_source_compact_full_residual_remesh_N896_s12.md`
- N=896 residual-remesh checkpoint:
  `outputs/checkpoints/high_mdot_stream_source_compact_full_residual_remesh_N896_s12/N896_s12_mass_0p82_torque_0p005_mdot_2_N896.npz`

## Interpretation

The old wall is not simply caused by the tanh source tail at Rout. A fully compact
no-tail C2 source exists at f_s = 0.82 on the current N=768 grid and has smooth,
nearly unchanged physical diagnostics relative to the tanh branch.

The remaining numerical friction is mostly outer angular closure / slope Picard
lag and high-N finite-difference Jacobian cost:

- most failed or marginal steps are dominated by `outer_omega`;
- interval residuals remain acceptable during the compact homotopy, although
  the full compact endpoint has `interval_E` at the same few-e-6 level as
  `outer_omega` while the median interval-E defect remains tiny;
- full compact required 60 Newton function evaluations and ended accepted but
  not strict.

## Suggested next step

Do not add wind yet. The compact-source endpoint is now mesh-supported only when
residual-aware remeshing is used. The best next move is:

1. Treat residual-aware mesh placement as required for compact/high-source
   branches.
2. Continue f_s upward from the residual-remeshed compact checkpoints, starting
   from N=896 if cost remains acceptable.
3. Add an explicit outer-slope Picard loop or cheaper boundary-only correction
   only if the next f_s continuation steps again become dominated by refreshed
   `outer_omega`.
