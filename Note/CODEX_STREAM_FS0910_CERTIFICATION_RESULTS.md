# Stream Branch f_s=0.910 Certification Results

Date: 2026-07-04

## Goal

Certify the newly rescued no-wind compact stream branch before moving to
`Mdot_inner/Edd=3`.

Configuration:

- `Mdot_inner/Edd = 2`
- `Rout = 335 rg`
- compact stream source
- torque delta-l fraction `+0.005`
- strict gates: full residual `<=1e-5` and physical_E `<=1e-5`

## Tests Run

1. Nested physical refinement on `f_s=0.900`, `0.905`, and `0.910`.
2. One higher-N spot check for `f_s=0.910`, remapping `N=896 -> 960`.
3. Smaller-step replay around the former plateau from `f_s=0.89825` to
   `0.899`.

## Nested Refinement

Each run split the top physical interval-E defects, inserted eight nodes, used
local inserted-node initialization, and repolished to `N=904`.

| f_s | N | full before | full final | physical_E before | physical_E final | f_adv_global final | Rson/rg final |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.900 | 896 -> 904 | 3.743e-07 | 3.736e-07 | 8.228e-08 | 3.410e-07 | 0.204349 | 4.65992 |
| 0.905 | 896 -> 904 | 8.387e-07 | 8.322e-07 | 2.265e-07 | 5.695e-07 | 0.204376 | 4.65992 |
| 0.910 | 896 -> 904 | 3.233e-06 | 2.209e-06 | 3.233e-06 | 2.041e-06 | 0.204403 | 4.65992 |

Reports:

- `outputs/tables/high_mdot_stream_nested_refinement_fs0900_validation.md`
- `outputs/tables/high_mdot_stream_nested_refinement_fs0905_validation.md`
- `outputs/tables/high_mdot_stream_nested_refinement_fs0910_validation.md`

Interpretation: the branch survives nested physical refinement. The physical
diagnostics remain smooth and the residuals stay comfortably below the strict
gate.

## N=960 Spot Check

Anchor:

`outputs/checkpoints/high_mdot_stream_local_patch3_0905_to0910/patch3_mass_0p91_torque_0p005_mdot_2_N896.npz`

First N960 attempt with only three patch passes:

- full residual `4.977e-05`
- physical_E `2.130e-07`
- rejected because the remaining full residual was buffer/global interval-E

Second N960 attempt with four patch passes:

- full residual `2.697e-06`
- physical_E `2.130e-07`
- accepted and anchor-eligible
- final patch mode: global

Report:

- `outputs/tables/high_mdot_stream_N960_fs0910_spotcheck_patch4.md`

Checkpoint:

- `outputs/checkpoints/high_mdot_stream_N960_fs0910_spotcheck_patch4/N960p4_mass_0p91_torque_0p005_mdot_2_N960.npz`

Interpretation: the branch survives one higher-N spot check, but the N960 remap
needs the same two-stage cleanup seen before: first physical/source cleanup,
then global/buffer cleanup.

## Smaller-Step Replay

Replay from the clean `f_s=0.89825` anchor:

| f_s | accepted | full final | physical_E final | patch passes |
|---:|:---:|---:|---:|---:|
| 0.898375 | yes | 8.374e-06 | 8.374e-06 | 1 |
| 0.8985 | yes | 8.426e-06 | 8.426e-06 | 1 |
| 0.89875 | yes | 9.206e-06 | 9.206e-06 | 1 |
| 0.899 | yes | 8.061e-07 | 2.046e-07 | 2 |

Report:

- `outputs/tables/high_mdot_stream_smallstep_replay_089825_to0899.md`

Interpretation: the branch crossing near the old `f_s~0.8985` wall is not only
a large-step artifact. Smaller steps also pass the strict gate, usually with
only one local physical patch.

## Certification Status

The `Mdot_inner/Edd=2`, compact no-wind stream branch is now reasonably
certified through `f_s=0.910` for near-term continuation work.

Evidence:

- nested physical refinement passes at `f_s=0.900`, `0.905`, `0.910`;
- an N960 spot check passes at `f_s=0.910`;
- smaller-step replay passes across the old plateau;
- physical diagnostics remain smooth:
  - `f_adv_global ~ 0.20435-0.20440`
  - `Rson ~ 4.6599 rg`
  - `max H/R ~ 0.2269`

Remaining caveat:

- The branch is still numerically rescued. The rescue is now reproducible, but
  the outer-buffer/global cleanup pass is essential at high source fraction and
  higher N. This should remain part of the method until a better global
  predictor/remeshing strategy replaces it.

## Recommended Next Step

Proceed to `Mdot_inner/Edd=3`, but do it cautiously:

1. first no-stream finite-Rout confirmation at the comparable outer radius;
2. then compact stream source with small source fractions, e.g. `f_s=0.05`,
   `0.10`, `0.30`;
3. keep local patch rescue enabled with at least four passes;
4. require the same strict full/physical gates;
5. reserve `Mdot=5` until `Mdot=3` is clean.

## Verification

`PYTHONPATH=src:scripts /Users/huanyang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest -q`

Result: `149 passed`.
