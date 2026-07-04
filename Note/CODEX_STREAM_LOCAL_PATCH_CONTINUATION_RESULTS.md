# Stream Local-Patch Continuation Results

Date: 2026-07-04

## What Was Implemented

The successful local physical-energy patch from the diagnostic stage was
promoted into `scripts/run_standard_slim_stream_mass_annulus_scan.py` as an
opt-in rescue path:

- `IMBH_STANDARD_SLIM_STREAM_MASS_LOCAL_PATCH_ON_REJECT=1`
- localizes the peak physical interval-E residual after an otherwise rejected
  step;
- solves only local `logu/logT` nodes while holding the rest of the branch
  fixed;
- can run multiple passes with
  `IMBH_STANDARD_SLIM_STREAM_MASS_LOCAL_PATCH_MAX_PASSES`;
- after physical energy is below tolerance, a final global configured-residual
  pass can target the outer-buffer/full-residual cell:
  `IMBH_STANDARD_SLIM_STREAM_MASS_LOCAL_PATCH_GLOBAL_AFTER_PHYSICAL=1`.

This means the normal Newton/energy-merit solve still runs first. The patch is
only invoked when the point would otherwise fail the acceptance gate.

## Main Result

The former `f_s~0.8985` wall is now crossed, and the no-wind compact stream
branch continues to `f_s=0.910` at `Mdot_inner/Edd=2`, `Rout=335 rg`,
`N=896`.

Accepted strict anchors:

| f_s | final full | physical_E | raw buffer_E | passes | final patch mode | f_adv_global | Rson/rg |
|---:|---:|---:|---:|---:|---|---:|---:|
| 0.8985 | 4.298e-06 | 5.512e-08 | 4.298e-03 | 2 | physical | 0.20434 | 4.6599 |
| 0.899 | 4.366e-06 | 2.064e-08 | 4.366e-03 | 2 | physical | 0.20434 | 4.6599 |
| 0.900 | 3.743e-07 | 8.228e-08 | 3.743e-04 | 3 | global | 0.20435 | 4.6599 |
| 0.902 | 2.370e-06 | 2.370e-06 | 8.792e-05 | 3 | global | 0.20436 | 4.6599 |
| 0.905 | 8.387e-07 | 2.265e-07 | 8.387e-04 | 3 | global | 0.20438 | 4.6599 |
| 0.910 | 3.233e-06 | 3.233e-06 | 2.261e-03 | 3 | global | 0.20440 | 4.6599 |

All listed points pass the strict `1e-5` full-residual and physical-energy
gate used in these runs.

## Important Intermediate Finding

A two-pass physical-only patch fixed `f_s=0.900` physically but left a weighted
outer-buffer residual just above tolerance:

- physical-only result: `full=1.250e-05`, `physical_E=1.162e-05`
- after allowing a third global pass:
  `full=3.743e-07`, `physical_E=8.228e-08`

So the current continuation bottleneck is a two-stage numerical localization
problem:

1. first clean the physical/source interval-E defect;
2. then clean the weighted outer-buffer interval-E defect.

Neither behavior looks like a sonic or physical branch endpoint in these runs.

## Output Files

Integrated rescue checks:

- `outputs/tables/high_mdot_stream_local_patch2_089825_to08985.md`
- `outputs/tables/high_mdot_stream_local_patch2_08985_to0900.md`
- `outputs/tables/high_mdot_stream_local_patch3_0899_to0900.md`
- `outputs/tables/high_mdot_stream_local_patch3_0900_to0905.md`
- `outputs/tables/high_mdot_stream_local_patch3_0905_to0910.md`

Latest strict checkpoint:

- `outputs/checkpoints/high_mdot_stream_local_patch3_0905_to0910/patch3_mass_0p91_torque_0p005_mdot_2_N896.npz`

## Interpretation

The no-wind stream-fed branch is now numerically recoverable beyond the old
`f_s~0.8985` plateau. The local patch shows that the obstruction was not a
global loss of the branch; it was a localized correction missed by the global
Newton/continuation predictor.

The branch remains mildly advective and smooth over this extension:

- `f_adv_global ~ 0.2043-0.2044`
- `Rson ~ 4.6599 rg`
- `max H/R` remains near the previous `~0.227` level in the output tables

## Current Caveat

This is a numerically rescued branch. It is promising, but not yet a robustness
claim. The accepted points should now be checked with:

1. nested physical refinement plus local inserted-node initialization;
2. N-variation spot checks around `f_s=0.900`, `0.905`, and `0.910`;
3. smaller-step replay to confirm the patch is not hiding a step-size artifact;
4. residual localization plots showing physical and buffer cleanup separately.

## Recommended Next Step

Use the new `f_s=0.910` checkpoint as the anchor and run a cost-aware adaptive
ladder with smaller steps, e.g. `0.912`, `0.915`, `0.920`, while keeping:

- three local patch passes;
- physical-first then global-buffer cleanup;
- strict `1e-5` full and physical-energy gates;
- periodic nested-refinement validation.

Do not add heating or wind yet. The no-wind stream branch is still being
successfully extended.

## Verification

`PYTHONPATH=src:scripts /Users/huanyang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest -q`

Result: `149 passed`.
