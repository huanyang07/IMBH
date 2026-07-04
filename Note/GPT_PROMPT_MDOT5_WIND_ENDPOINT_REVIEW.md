# GPT Prompt: Review Mdot=5 Wind Endpoint Continuation

Please review the latest GitHub state of `huanyang07/IMBH`, focusing on the
Mdot_inner/Edd=5, Rout=335 rg, f_s=0.80, no-heating, energy-limited wind
continuation.

Important implementation changes:

1. A wind-aware interval-energy Jacobian was added:
   - `src/imri_qpe/layer3_minidisk_1d/winds.py`
     - `energy_limited_wind_derivatives(...)`
   - `src/imri_qpe/layer3_minidisk_1d/transonic_local.py`
     - active-wind local slope Jacobian in `differential_matrix(...)`
   - `src/imri_qpe/layer3_minidisk_1d/transonic_collocation.py`
     - hybrid wind-aware `interval_E` Jacobian for differential intervals.
   - Regression tests were added in `tests/test_winds.py` and
     `tests/test_transonic_local.py`.

2. Energy-wind continuation infrastructure was added to
   `scripts/run_standard_slim_stream_mass_annulus_scan.py`:
   - `IMBH_STANDARD_SLIM_STREAM_MASS_ENERGY_WIND_EPSILONS`
   - `IMBH_STANDARD_SLIM_STREAM_MASS_ENERGY_WIND_ETAS`
   - `IMBH_STANDARD_SLIM_STREAM_MASS_ENERGY_WIND_PREV_ANCHOR`
   - epsilon tangent predictor and eta endpoint coordinate predictor.

Key numerical status:

- Before the wind-aware Jacobian, strict continuation stalled near
  `epsilon_w ~= 0.065` because of an `interval_E` floor near `R ~= 65.8 rg`.
- After the wind-aware Jacobian, the branch is strict through high wind
  efficiencies:
  - `epsilon_w=0.98`, residual `2.16e-10`,
    `int Qwind/Qvisc ~= 0.198`.
  - Raw epsilon staging reached `epsilon_w=0.997`, residual `1.21e-09`,
    `int Qwind/Qvisc ~= 0.690`.
- Eta endpoint continuation with `eta=-log(1-epsilon_w)` extended the branch:
  - `eta=6.20`, `epsilon_w=0.997970569`, residual `3.68e-10`,
    `int Qwind/Qvisc ~= 0.783`.
  - `eta=6.30`, `epsilon_w=0.998163695`, residual `1.46e-10`,
    `int Qwind/Qvisc ~= 0.802`.
  - `eta=6.35`, `epsilon_w=0.998253253`, residual `1.59e-10`,
    `int Qwind/Qvisc ~= 0.811`.
- A fixed `eta=6.4` step was too aggressive; a smaller `6.30 -> 6.35` step
  succeeded. This suggests endpoint stiffness / step-size control rather than
  a clean branch loss.
- Exact `epsilon_w=1` cannot be reached at finite eta. Direct exact-endpoint
  attempts from `0.98` and nearby anchors produced poor initial residuals and
  were too costly.

Important output/result files:

- `Note/CODEX_MDOT5_WIND_SPRINT_RESULTS.md`
- `outputs/tables/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_windjac_098_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_windjac_0997_0999_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_eta_590_620_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_eta_630_650_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_eta_635_N896.md`

Verification:

- Full tests pass: `154 passed`.

Questions for you:

1. Does the wind-aware interval-energy Jacobian implementation look physically
   and numerically consistent, or are any derivative terms missing?
2. What is the best next continuation strategy near the endpoint:
   adaptive eta stepping, pseudo-arclength in `(z, eta)`, pseudo-arclength in
   `(z, epsilon_w)`, or a separate endpoint solve with `beta=1-epsilon_w`?
3. What validation should be done before interpreting the high-wind branch as a
   real advective/wind hot branch?
4. The high-eta branch shows `Qwind/Qvisc > 0.8`, `Rson` moving outward, and
   advection diagnostics becoming small/negative. What diagnostics should be
   audited to decide whether this is physically meaningful or a numerical
   closure artifact?
5. Should we next implement adaptive eta stepping, mesh/N validation at
   `eta=6.2-6.35`, or residual/energy-budget localization first?
