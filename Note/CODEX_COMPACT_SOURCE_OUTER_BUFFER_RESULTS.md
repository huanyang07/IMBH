# Compact Source Outer-Buffer Test

Date: 2026-07-03

## Purpose

The previous integrated continuation reached `f_s=0.8625` at `Rout=300 rg`, but the differential
energy defect peaked at the outer edge near `R=298.4 rg`. The question was whether this was a
physical source-annulus defect or a hard outer-boundary defect.

## New Runner Controls

`scripts/run_standard_slim_stream_mass_annulus_scan.py` now supports:

- `IMBH_STANDARD_SLIM_STREAM_MASS_R_OUT_RG`
- `IMBH_STANDARD_SLIM_STREAM_MASS_FIXED_RINJ_RG`
- `IMBH_STANDARD_SLIM_STREAM_MASS_FIXED_TORQUE_RINJ_RG`

These controls remap the loaded checkpoint to a new outer radius while keeping the stream source
and, by default, the torque center at a fixed physical radius. In the tests below:

- `Rinj = 240 rg`
- `f_s = 0.8625`
- `Mdot_inner/Edd = 2`
- `N = 896`
- compact C2 source
- `torque_delta_l_fraction = +0.005`
- no wind, no heating
- integrated interval residual form

## Results

Comparison profile:

- `outputs/tables/high_mdot_stream_compact_outer_buffer_interval_profile_r300_305_310_335.md`
- `outputs/figures/high_mdot_stream_compact_outer_buffer_interval_profile_r300_305_310_335.png`

| case | seed path | final integrated full | accepted | peak interval_E R/rg | differential interval_E | outer angular residual |
|---|---|---:|:---:|---:|---:|---:|
| `Rout=300` | accepted f_s=0.8625 checkpoint | 9.790e-07 | yes | 298.4 | 1.399e-03 | 5.987e-08 |
| `Rout=305` | direct from `Rout=300` | 6.984e-06 | yes | 303.2 | 9.940e-03 | 4.413e-06 |
| `Rout=310` | direct from `Rout=300` | 4.590e-02 | no | 309.5 | 7.911e+00 | 4.590e-02 |
| `Rout=310` | staged from accepted `Rout=305` | 1.199e-04 | no | 307.5 | 1.698e-01 | -5.029e-05 |
| `Rout=335` | direct from `Rout=300` | 2.635e-01 | no | 334.9 | 2.360e+01 | 2.635e-01 |

Key outputs:

- `outputs/tables/high_mdot_stream_compact_outer_buffer_rout305_fixedrinj240_fs08625_N896.md`
- `outputs/tables/high_mdot_stream_compact_outer_buffer_rout310_fixedrinj240_fs08625_N896.md`
- `outputs/tables/high_mdot_stream_compact_outer_buffer_rout310_from305_fixedrinj240_fs08625_N896.md`
- `outputs/tables/high_mdot_stream_compact_outer_buffer_rout335_fixedrinj240_fs08625_N896.md`

## Interpretation

The peak differential energy defect moves with the computational outer boundary:

- `Rout=300` peak: `298.4 rg`
- `Rout=305` peak: `303.2 rg`
- `Rout=310` staged peak: `307.5 rg`
- `Rout=335` peak: `334.9 rg`

This strongly supports the outer-boundary interpretation. The unresolved defect is not locked to
the physical compact source center at `Rinj=240 rg`. A small `5 rg` buffer can be made acceptable
under the integrated residual, but a `10 rg` buffer remains above tolerance even after staging.

The staged `Rout=310` attempt improved dramatically relative to direct remap:

- direct `Rout=310`: `4.590e-02`
- `Rout=305 -> 310`: `1.199e-04`

However, repolishing the staged `Rout=310` checkpoint with a larger Newton iteration cap was too
slow and was interrupted. This suggests the remaining problem is not just one missing Newton
iteration; the hard outer closure produces a stiff localized outer-tail interval defect.

A naive `pressure_supported_robin_energy` probe at `Rout=310`, `chi=0.5`, started with a worse
initial residual (`1.375e+00`) and was interrupted. This does not rule out a better soft closure,
but the current Robin target/scale is not a drop-in fix.

## Current Conclusion

The `f_s=0.8625` compact no-wind branch is conditionally continued by integrated collocation, but
the outer boundary now controls the result. We should not claim a robust physical branch beyond this
point until the outer-tail closure is replaced or buffered in a mesh-convergent way.

## Suggested Next Step

Implement a dedicated outer-tail treatment rather than continuing brute force:

1. Add a two-zone outer buffer:
   - physical source domain through the compact source annulus;
   - outer reservoir/buffer domain with a soft angular/entropy closure.
2. Test existing closures as controlled variants:
   - `matched_outer_state`
   - `full_slope_match`
   - tuned `pressure_supported_robin_energy`
3. Use `Rout=305` as the last accepted hard-closure buffer anchor and require any new closure to
   reach `Rout=310,335,375` without the peak following the outer edge.
4. Only then resume source-fraction continuation toward `f_s=0.90`.

Wind and stream heating should still wait.

## 2026-07-03 Implementation Update: Dedicated Outer-Tail/Buffer Formulation

Implemented an opt-in outer-tail/buffer formulation in
`src/imri_qpe/layer3_minidisk_1d/transonic_collocation.py`.

New `TransonicSlimParams` controls:

- `outer_buffer_inner_rg`
- `outer_buffer_radial_weight`
- `outer_buffer_energy_weight`
- `outer_buffer_boundary_weight`
- `outer_buffer_taper_log_width`

When `outer_buffer_inner_rg` is set, collocation intervals outside that radius are
treated as an outer reservoir/buffer by down-weighting the selected radial and
energy interval residuals. The terminal outer boundary residual can be
down-weighted separately. The residual audit and interval-profile scripts still
report the raw unweighted differential residuals, so the buffer cannot hide the
physical-domain defect in diagnostics.

Also added two opt-in boundary closures for experiments:

- `pressure_supported_local_energy`
- `pressure_supported_entropy_slope`

These were tested at `Rout=310 rg` from the accepted `Rout=305 rg` seed, but
their cheap initial residuals were worse than the existing hard closure:

| closure | initial full residual |
|---|---:|
| hard `pressure_supported_thin_energy` | `4.050e-02` |
| `pressure_supported_local_energy` | `1.408e+00` |
| `pressure_supported_entropy_slope` | `3.469e+00` |

### Buffered Runs

All tests below use:

- `Mdot_inner/Edd = 2`
- `f_s = 0.8625`
- compact C2 source
- `torque_delta_l_fraction = +0.005`
- fixed physical injection center `Rinj = 240 rg`
- no wind, no stream heating
- integrated interval residual form
- `N = 896`

| case | buffer inner | weights `(R,E,B)` | final weighted full | accepted | raw interval_E | raw outer omega | interpretation |
|---|---:|---|---:|:---:|---:|---:|---|
| `Rout=310` from accepted `Rout=305` | `300 rg` | `(1, 1e-3, 1e-2)` | `1.774e-06` | yes | `1.036` | `4.477e-06` | Buffer succeeds as a reservoir anchor. |
| `Rout=335` from buffered `Rout=310` | `300 rg` | `(1, 1e-3, 1e-2)` | `1.715e-03` | no | large | `0.171` | Terminal angular residual still dominates. |
| `Rout=335` from buffered `Rout=310` | `300 rg` | `(1, 1e-3, 1e-4)` | `3.844e-04` | no | large | `0.169` | Boundary no longer sole limiter. |
| `Rout=335` from buffered `Rout=310` | `300 rg` | `(1e-3, 1e-3, 1e-4)` | `3.113e-04` | no | large | `0.169` | Residual piles up near buffer interface. |
| `Rout=335` from buffered `Rout=310` | `290 rg` | `(1e-3, 1e-3, 1e-4)` | `5.182e-05` | no | `15.25` | `0.144` | Outer wall suppressed; remaining weighted defect is source-domain polish/remesh. |

The `Rout=335`, `R_buffer=290 rg` checkpoint was repolished with a larger
Newton cap. It did not improve: the final weighted full residual stayed at
`5.182e-05`. This suggests the next limiter is no longer the hard terminal
boundary alone. The weighted maximum moved to source/inner-buffer intervals
around `R ~ 223-252 rg` and the raw audit peak remains at the outer tail.

Key outputs:

- `outputs/tables/high_mdot_stream_compact_outer_buffered_rout310_E1e3_B1e2_from305_N896.md`
- `outputs/checkpoints/high_mdot_stream_compact_outer_buffered_rout310_E1e3_B1e2_from305_N896/`
- `outputs/tables/high_mdot_stream_compact_outer_buffered_rout335_buffer290_R1e3_E1e3_B1e4_from310_N896.md`
- `outputs/tables/high_mdot_stream_compact_outer_buffered_rout335_buffer290_repolish_N896.md`
- `outputs/tables/high_mdot_stream_compact_outer_buffer_weighted_profile_rout310_335.md`
- `outputs/figures/high_mdot_stream_compact_outer_buffer_weighted_profile_rout310_335.png`

### Interpretation

The dedicated buffer formulation works for a controlled `Rout=305 -> 310 rg`
extension and confirms that the previous wall is largely an outer-tail/reservoir
closure issue, not a sonic failure. It should be treated as a reservoir
formulation anchor rather than a strict full-ODE anchor, because raw unweighted
energy residuals remain intentionally large inside the buffer.

The failed `Rout=335 rg` attempt is nevertheless informative. Once the buffer
weights are strong enough to suppress the terminal closure, the remaining
weighted defect shifts back toward the compact source annulus and buffer
interface. The next numerical step should therefore be staged radius continuation
or residual-aware remeshing focused on the source plus inner-buffer transition,
not wind/heating.

### Verification

`PYTHONPATH=src:scripts python -m pytest` passes: `145 passed`.

## 2026-07-03 Follow-up: Split Audit, Staged Rout Ladder, and Robustness Checks

Implemented a split residual audit:

- `residual_partition_audit_from_state_vector(...)`
- `TransonicResidualPartitionAudit`

The new audit reports raw differential residuals separately in:

- the physical/source domain, `R < R_buffer`;
- the softened outer buffer, `R_buffer < R < R_out`;
- the terminal boundary.

`scripts/run_standard_slim_stream_mass_annulus_scan.py` now writes these
partition fields into JSON/table rows, and
`scripts/run_standard_slim_stream_interval_profile.py` includes physical-vs-buffer
energy columns in its summary table.

### Staged Radius Continuation

Starting from the accepted `Rout=310 rg` buffered checkpoint, I ran:

`310 -> 315 -> 320 -> 325 -> 330 -> 335 rg`

with:

- fixed physical `Rinj = 240 rg`
- compact C2 stream source
- `f_s = 0.8625`
- `Mdot_inner/Edd = 2`
- `N = 896`
- `R_buffer = 300 rg`
- buffer weights `(R,E,B) = (1e-3, 1e-3, 1e-4)`
- integrated interval residual form
- residual-aware remesh after accepted steps and on rejected steps

All staged points accepted as strict anchors:

| Rout/rg | final weighted full | physical raw E max | buffer raw E max | f_adv_global | f_adv_inner | Lrad/LEdd | Rson/rg |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 315 | `2.650e-07` | `3.137e-04` | `2.491e-02` | `0.20426` | `0.09532` | `0.86692` | `4.65992` |
| 320 | `9.232e-08` | `1.049e-04` | `2.811e-02` | `0.20423` | `0.09499` | `0.86708` | `4.65992` |
| 325 | `2.519e-07` | `8.017e-05` | `3.330e-01` | `0.20421` | `0.09569` | `0.86724` | `4.65992` |
| 330 | `1.027e-07` | `1.139e-05` | `1.605e-01` | `0.20417` | `0.09627` | `0.86739` | `4.65992` |
| 335 | `2.342e-07` | `1.185e-05` | `2.478e-01` | `0.20415` | `0.09622` | `0.86754` | `4.65992` |

This resolves the previous direct-jump `Rout=335` failure. The failure was not
a physical endpoint of the stream-fed branch; it was a continuation/remeshing
problem at the outer-tail reservoir.

### Robustness Checks at Rout = 335 rg

Mesh checks:

| N | final weighted full | physical raw E max | buffer raw E max | f_adv_global | f_adv_inner | Lrad/LEdd | Rson/rg |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 768 | `2.104e-07` | `2.416e-06` | `2.722e-01` | `0.20414` | `0.09628` | `0.86754` | `4.65992` |
| 896 | `2.342e-07` | `1.185e-05` | `2.478e-01` | `0.20415` | `0.09622` | `0.86754` | `4.65992` |
| 1024 | `7.469e-07` | `2.392e-04` | `1.464e-01` | `0.20415` | `0.09586` | `0.86754` | `4.65992` |

Buffer-edge checks at `N=896`:

| R_buffer/rg | final weighted full | physical raw E max | buffer raw E max | result |
|---:|---:|---:|---:|---|
| 295 | `2.342e-07` | `1.185e-05` | `2.478e-01` | strict; remesh not adopted |
| 300 | `2.342e-07` | `1.185e-05` | `2.478e-01` | strict baseline |
| 305 | `2.880e-07` | `2.535e-06` | `4.096e-01` | strict after remesh |

Buffer-weight check:

| weights `(R,E,B)` | final weighted full | result |
|---|---:|---|
| `(1e-3,1e-3,1e-4)` | `2.342e-07` | strict baseline |
| `(3e-3,3e-3,3e-4)` | `7.027e-07` | strict without needing remesh |

Key combined outputs:

- `outputs/tables/high_mdot_stream_compact_outer_buffer_ladder_validation_profile.md`
- `outputs/figures/high_mdot_stream_compact_outer_buffer_ladder_validation_profile.png`
- `outputs/tables/high_mdot_stream_compact_outer_buffer_ladder_rout335_N896.md`
- `outputs/checkpoints/high_mdot_stream_compact_outer_buffer_ladder_rout335_N896/`
- `outputs/tables/high_mdot_stream_compact_outer_buffer_validation_N768.md`
- `outputs/tables/high_mdot_stream_compact_outer_buffer_validation_N1024.md`
- `outputs/tables/high_mdot_stream_compact_outer_buffer_validation_buffer295.md`
- `outputs/tables/high_mdot_stream_compact_outer_buffer_validation_buffer305.md`
- `outputs/tables/high_mdot_stream_compact_outer_buffer_validation_weight3e3.md`

### Updated Interpretation

The staged buffer/remesh method is now strong enough to carry the compact
stream-fed no-wind branch from `Rout=310` to `335 rg` and passes first-pass
mesh, buffer-edge, and buffer-weight checks. Because the staged ladder did not
stall, I did not implement the heavier two-domain phase-space/state-vector
formulation in this pass.

The scientific interpretation remains conservative:

- the branch is robust as a reservoir-formulation branch;
- raw outer-buffer differential residuals are intentionally large and must not
  be mistaken for physical-domain convergence;
- the physical/source-domain residuals and global diagnostics are stable enough
  to resume source-fraction continuation or retry larger finite-Rout hot-branch
  tests without adding wind yet.

### Verification

`PYTHONPATH=src:scripts python -m pytest` passes after these changes:
`146 passed`.

## 2026-07-03 Source-Fraction Continuation from f_s = 0.8625

Using the validated `Rout=335 rg`, `R_buffer=300 rg`, `N=896`
reservoir-formulation checkpoint, I resumed compact-source continuation toward
`f_s=0.90`.

Run controls:

- `Mdot_inner/Edd = 2`
- `Rout = 335 rg`
- fixed physical `Rinj = 240 rg`
- compact C2 source, `torque_delta_l_fraction = +0.005`
- no wind, no stream heating
- integrated interval residual
- buffer weights `(R,E,B) = (1e-3, 1e-3, 1e-4)`
- adaptive source-fraction steps with residual-aware remesh

The branch continued cleanly past the old wall, but became expensive near
`f_s ~ 0.876`. I stopped the run after the strict `f_s=0.8759639587` anchor
because the adaptive controller had reached its minimum step and remained
costly. The next attempted step to `f_s=0.8762139587` was interrupted during
the remeshed Newton polish.

| f_s | final weighted full | strict anchor | nfev total | next step | physical raw E max | buffer raw E max | f_adv_global | f_adv_inner | Lrad/LEdd | Rson/rg |
|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.865000000` | `3.129e-07` | yes | 27 | `3.125e-03` | `1.550e-04` | `1.020e-01` | `0.20414` | `0.09580` | `0.86747` | `4.65993` |
| `0.868125000` | `5.905e-06` | no | 89 | `3.125e-03` | `2.032e-03` | `8.172e-02` | `0.20418` | `0.09539` | `0.86739` | `4.65992` |
| `0.871250000` | `1.715e-06` | yes | 164 | `7.8125e-04` | `7.004e-04` | `9.281e-03` | `0.20420` | `0.09539` | `0.86730` | `4.65992` |
| `0.872031250` | `4.688e-08` | yes | 118 | `3.90625e-04` | `1.404e-05` | `5.406e-02` | `0.20420` | `0.09564` | `0.86728` | `4.65992` |
| `0.872421875` | `2.236e-07` | yes | 18 | `4.88281e-04` | `7.108e-05` | `2.645e-02` | `0.20420` | `0.09598` | `0.86727` | `4.65992` |
| `0.872910156` | `2.195e-07` | yes | 7 | `6.10352e-04` | `4.618e-06` | `2.568e-01` | `0.20421` | `0.09443` | `0.86726` | `4.65992` |
| `0.873520508` | `4.068e-07` | yes | 20 | `7.62939e-04` | `4.941e-05` | `6.181e-01` | `0.20420` | `0.09481` | `0.86724` | `4.65992` |
| `0.874283447` | `8.477e-07` | yes | 31 | `9.53674e-04` | `4.134e-04` | `2.755e-02` | `0.20421` | `0.09598` | `0.86722` | `4.65992` |
| `0.875237122` | `2.715e-07` | yes | 104 | `4.76837e-04` | `9.455e-05` | `3.039e-01` | `0.20421` | `0.09463` | `0.86719` | `4.65992` |
| `0.875713959` | `7.108e-06` | no | 115 | `2.5e-04` | `1.942e-03` | `6.333e-02` | `0.20421` | `0.09530` | `0.86718` | `4.65992` |
| `0.875963959` | `1.242e-07` | yes | 136 | `2.5e-04` | `1.473e-05` | `1.669e-01` | `0.20422` | `0.09443` | `0.86717` | `4.65992` |

Key outputs:

- `outputs/tables/high_mdot_stream_compact_outer_buffer_fs08625_to090_N896.md`
- `outputs/figures/high_mdot_stream_compact_outer_buffer_fs08625_to090_N896.png`
- `outputs/checkpoints/high_mdot_stream_compact_outer_buffer_fs08625_to090_N896/`

### Interpretation

The old `f_s ~ 0.86` wall was not a physical branch endpoint: the branch
continues to at least `f_s = 0.8759639587` with smooth global diagnostics.
However, the current continuation is now limited by source-fraction predictor
and Newton/remesh cost. The physical-domain residual spikes intermittently
(`~10^-3`) but can relax back to `~10^-5` after small steps and remeshing.

The next numerical improvement should be a true source-fraction tangent
predictor, or a short pseudo-arclength continuation in `f_s`, before spending
many hours walking from `0.876` to `0.90` at `df_s=2.5e-4`.
