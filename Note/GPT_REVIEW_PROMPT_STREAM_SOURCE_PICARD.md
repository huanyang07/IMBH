# GPT Review Prompt: Stream-Fed Slim Benchmark After Slope-Picard Fix

Please review the GitHub repository `huanyang07/IMBH`, especially these files:

- `Note/CODEX_STANDARD_SLIM_MDOT_EQUILIBRATED_CONTINUATION_RESULTS.md`
- `src/imri_qpe/layer3_minidisk_1d/transonic_local.py`
- `src/imri_qpe/layer3_minidisk_1d/transonic_collocation.py`
- `scripts/run_standard_slim_stream_source_strength_scan.py`
- `scripts/run_standard_slim_stream_heating_profile_diagnostics.py`
- `outputs/tables/slim_benchmark_stream_source_strength_scan_mdot1_rout300_slope_picard.md`
- `outputs/tables/slim_benchmark_stream_source_strength_slope_picard_profile_diagnostics.md`

Current status:

1. The standard no-wind transonic slim benchmark had already been recovered to
   `Mdot/Edd ~ 1` with finite outer radius and adaptive outer mesh.
2. We then added a distributed stream annulus:
   - cumulative torque source,
   - cumulative mass source `Mdot(R)`,
   - stream heating proportional to positive `dMdot/dlnR`.
3. Best current no-wind stream-fed configuration:
   - `Mdot_inner/Edd = 1`
   - `Rout = 300 rg`
   - `Rinj/Rout = 0.4`
   - log width `0.30`
   - torque fraction `Delta l/lK = 0.01`
   - heating efficiency `eta_heat = 10`
4. Direct staged source scan initially looked limited by `outer_omega`, but this
   was diagnosed as stale outer-slope bookkeeping in the pressure-supported
   closure.
5. A slope-Picard polish was added:
   - freeze outer one-sided slopes,
   - Newton-polish the square BVP,
   - recompute outer slopes from the polished state,
   - repeat until residual and slope fixed point are self-consistent.
6. With slope-Picard enabled, the no-wind stream-fed benchmark is strict through
   `f_m = 0.3`.

Key strict slope-Picard results:

| `f_m` | residual | max `Qstream/Qvisc` | integrated `Qstream/Qvisc` | max H/R | integrated advective fraction |
|---:|---:|---:|---:|---:|---:|
| 0.03 | `7.49e-9` | 0.251 | 0.0236 | 0.157 | 0.0322 |
| 0.10 | `9.69e-9` | 0.827 | 0.0781 | 0.158 | 0.0286 |
| 0.20 | `1.55e-8` | 1.625 | 0.155 | 0.204 | 0.0234 |
| 0.30 | `1.95e-8` | 2.388 | 0.232 | 0.254 | 0.0186 |

Interpretation so far:

- The source-fed no-wind benchmark is now numerically robust at `f_m=0.3`.
- The disk develops a localized puffed stream-heated region near `R~104 rg`.
- The integrated advective fraction decreases with increasing source strength.
- Therefore this no-wind source prescription does not by itself recover the
  advective/wind hot branch needed for the QPE model.

Please advise on the next best step.

Specific questions:

1. Should we now add wind/source-sink physics on top of this slope-Picard
   stream-fed benchmark, or first push the no-wind source to stronger/inwarder
   deposition?
2. What minimal wind prescription should be added first so the model can
   produce an advective/wind hot branch without overfitting?
3. How should the wind enter the equations: `Mdot(R)` sink, angular momentum
   sink, energy sink/cooling, Bernoulli/escape-energy sink, or a coupled
   prescription?
4. What diagnostics should be required before claiming an advective/wind branch:
   mesh convergence, energy integral closure, wind energy budget, sonic
   regularity, slope-Picard convergence, or other checks?
5. Are there any red flags in the current stream source implementation or
   pressure-supported outer closure before we build the wind layer?

Desired output:

- A concrete implementation plan for the next sprint.
- Suggested equations/source terms for the first wind-enabled model.
- Acceptance criteria and failure diagnostics.
