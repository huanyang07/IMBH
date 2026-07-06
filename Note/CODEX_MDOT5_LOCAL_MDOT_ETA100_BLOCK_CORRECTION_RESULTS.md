# Codex Mdot=5 Local-Mdot Eta_E=100 Block-Correction Results

Date: 2026-07-05

This sprint implements the coupled block/Jacobian-aware correction recommended
after the residual-floor audit.  It is still a numerical correction sprint, not
a new physical wind/heating model.

## Code changes

Updated driver:

```text
scripts/run_mdot5_local_mdot_eta_continuation.py
```

New opt-in block-correction controls:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_BLOCK_CORRECT
IMBH_MDOT5_LOCAL_MDOT_ETA_BLOCK_HALF_WIDTH
IMBH_MDOT5_LOCAL_MDOT_ETA_BLOCK_MAX_NFEV
IMBH_MDOT5_LOCAL_MDOT_ETA_BLOCK_EDGE_ANCHOR_WEIGHT
IMBH_MDOT5_LOCAL_MDOT_ETA_BLOCK_ALL_ANCHOR_WEIGHT
IMBH_MDOT5_LOCAL_MDOT_ETA_BLOCK_INCLUDE_OUTER
IMBH_MDOT5_LOCAL_MDOT_ETA_BLOCK_INCLUDE_GLOBALS
IMBH_MDOT5_LOCAL_MDOT_ETA_BLOCK_LINE_SEARCH_STEPS
IMBH_MDOT5_LOCAL_MDOT_ETA_BLOCK_ACCEPT_STRICT_GUARDS
```

The corrector:

```text
1. finds the peak local interval_R row;
2. selects a node block around that interval;
3. solves radial + energy + local-Mdot rows together;
4. anchors block-edge node variables;
5. line-searches the step against the original global local-Mdot residual;
6. rejects the move if the guarded global physical residual gets worse.
```

I also added explicit local residual summary fields:

```text
local_interval_R
local_interval_E
peak_interval_R_rg
peak_interval_R
```

This avoids confusion with the older `interval_R/interval_E` columns inherited
from the non-tabulated transonic `z` audit.  For the local-Mdot BVP, use the
new `local_interval_*` columns and `final_full`.

## Starting point

Frozen input checkpoint:

```text
outputs/checkpoints/m5_local_mdot_eta_polish_N152_integrated_physE_then_differential_resume/stage_00_etaE_100_N152.npz
```

Starting local-Mdot residual:

```text
final_full = 2.042e-05
local interval_R = 2.042e-05
local interval_E = 6.865e-06
mass residual = 1.746e-06
```

## Block scan

All rows below use the original N152 grid and eta_E=100.

| run | block | guard | result |
|---|---:|---|---:|
| `m5_local_mdot_eta100_block_q2_a1em2` | q=2, edge anchor 1e-2 | strict | 1.914e-05 |
| `m5_local_mdot_eta100_block_q3_a1em2_n20` | q=3, edge anchor 1e-2 | strict | 1.778e-05 |
| `m5_local_mdot_eta100_block_q4_a1em2` | q=4, edge anchor 1e-2 | strict | 1.786e-05 |
| `m5_local_mdot_eta100_block_q4_a1em1_n20` | q=4, edge anchor 1e-1 | strict | 1.785e-05 |
| `m5_local_mdot_eta100_block_q4_a1_n20` | q=4, edge anchor 1 | strict | 1.785e-05 |
| `m5_local_mdot_eta100_block_q4_a1_relaxed_seed` | q=4, edge anchor 1 | relaxed line search | 1.530e-05 |
| `m5_local_mdot_eta100_block_q5_a1_relaxed_seed` | q=5, edge anchor 1 | relaxed line search | 1.528e-05 |
| `m5_local_mdot_eta100_block_q6_a1_relaxed_seed` | q=6, edge anchor 1 | relaxed line search | 1.401e-05 |

Interpretation:

```text
The block correction works only when the block is wide enough.
q=2-4 improves the state but stalls above 1.7e-5 under strict guards.
q=6 lets the radial row approach the strict threshold but temporarily raises
the local energy row, so a follow-up global differential polish is useful.
```

For the q=6 relaxed seed:

```text
local interval_R = 1.014e-05
local interval_E = 1.401e-05
mass residual = 4.349e-07
```

So the q=6 block moves the radial floor almost exactly to the target, while
energy becomes the temporary limiter.

## Successful repair sequence

Best sequence:

```text
1. q=6 relaxed block seed:
   outputs/checkpoints/m5_local_mdot_eta100_block_q6_a1_relaxed_seed/stage_00_etaE_100_N152.npz

2. global differential polish:
   outputs/checkpoints/m5_local_mdot_eta100_block_q6_relaxed_then_global/stage_00_etaE_100_N152.npz
   final_full = 1.098e-05

3. second q=6 relaxed block pass:
   outputs/checkpoints/m5_local_mdot_eta100_block_q6_second_block_seed/stage_00_etaE_100_N152.npz
   final_full = 9.603e-06
```

Final accepted checkpoint:

```text
outputs/checkpoints/m5_local_mdot_eta100_block_q6_second_block_seed/stage_00_etaE_100_N152.npz
```

Final audit:

```text
outputs/tables/m5_local_mdot_eta100_block_q6_strict_final_audit.md
outputs/tables/m5_local_mdot_eta100_block_q6_strict_final_audit_profiles.json
```

Final local-Mdot residuals:

```text
final_full = 9.603e-06
accepted_exploratory = true
local_interval_R = 9.603e-06 at R = 288.704 rg
local_interval_E = 6.840e-06 at R = 7.832 rg
mass_residual_max = 1.471e-06
interval_mass_residual_max = 3.566e-07
Mdot_outer/Mdot_inner = 0.23280913
f_adv_global = -0.00389125
Lrad/LEdd = 0.52751368
Rson = 5.298056 rg
```

The final physical diagnostics are essentially unchanged from the pre-repair
state.  The correction is numerical, not a new physical branch.

## Final diagnostic caveats

The final radial representation audit still shows strong representation
sensitivity:

| diagnostic | value |
|---|---:|
| current differential radial residual | 9.603e-06 |
| trapezoid-equivalent radial residual | 3.519e-06 |
| Simpson-equivalent radial residual | 7.575e-06 |
| virtual split-interval max residual | 1.764e-04 |
| representation indicator tau | 1.668e-04 |

The local block remains ill-conditioned:

```text
condition estimate = 2.557e+07
```

Interpretation:

```text
The original differential audit is now strict at N152, but the state is still
numerically delicate.  The split-interval residual has not become small.
This means the checkpoint is a valid strict N152 local-Mdot solution candidate,
not yet a mesh-certified physical branch point.
```

## What did not help

A plain global-polish restart from the first q=6 global checkpoint worsened
slightly:

```text
1.098e-05 -> 1.110e-05
```

So the useful pattern is not repeated blind global polish.  It is:

```text
wide coupled block move -> global rebalance -> small wide coupled block move.
```

## Verification

```text
python -m py_compile scripts/run_mdot5_local_mdot_eta_continuation.py
PYTHONPATH=src python -m pytest tests/test_winds.py tests/test_transonic_local.py
```

Focused tests pass:

```text
48 passed
```

## Recommended next step

Do not lower eta_E yet.  First certify this strict N152 checkpoint.

Recommended sequence:

```text
1. Re-audit the final checkpoint with the original differential local-Mdot
   residual and saved local_interval_* columns.

2. Try N140, N152, N160/168 nearby-grid validation from this repaired state.
   Use node-preserving remap; avoid naive source/transition-node insertion.

3. If N160 remap is not safe, build a defect-preserving remap for logu/logT
   as well as logMdot, because the current source/transition remap can still
   create large interval_E defects.

4. Only after eta_E=100 is strict on nearby grids should we lower eta_E to
   95, 90, 80, 70, 60.
```

Current status:

```text
eta_E=100 is now strict at N152.
It is not yet mesh-certified.
```
