# GPT Prompt: Mdot=5 Local-Mdot Eta_E=100 Residual-Floor Review

Please review the latest GitHub state of `huanyang07/IMBH`, especially the
new Mdot=5 local-Mdot eta continuation notes and artifacts:

```text
Note/CODEX_MDOT5_SHEN_DIAGNOSTIC_SPRINT_RESULTS.md
Note/CODEX_MDOT5_MASS_WIND_STATUS_AND_NEXT_PLAN_UPDATED_SHEN2014.md
Note/CODEX_MDOT5_LOCAL_MDOT_ETA_CONTINUATION_RESULTS.md
scripts/run_mdot5_local_mdot_eta_continuation.py
scripts/audit_mdot5_shen_wind_diagnostics.py
outputs/tables/m5_local_mdot_eta_*.md
outputs/tables/m5_local_mdot_eta_*_profiles.json
outputs/checkpoints/m5_local_mdot_eta_*
```

Current scientific target:

```text
Mdot_inner/Edd = 5
stream-fed compact-source/minidisk background
prescribed Shen-style zeta=0.03 mass-wind bridge as anchor
local Mdot(R) BVP with launch energy eta_E = 100
goal: make the weak local mass-loaded wind checkpoint strict before lowering eta_E
```

Main progress:

```text
1. The bad local-Mdot remap/refinement pathology was diagnosed and fixed.
   Old fixed-N residual remaps and ordinary N growth produced artificial
   source-annulus energy defects.  Node-preserving nested refinement with
   targeted insertion in R=100-320 rg avoids that catastrophe.

2. A repaired eta_E=100 ladder now exists:
       N128 -> N136 -> N140 -> N152
       targeted nested insertion band = 100-320 rg

3. The best direct N152 targeted result is:
       outputs/checkpoints/m5_local_mdot_eta_continuation_zeta0p03_N152_strict_eta100_nested_defect_R100_320_picard/stage_00_etaE_100_N152.npz

       final_full = 2.515e-05
       interval_R = 2.515e-05 near R~294.55 rg
       interval_E = 9.530e-06
       mass_residual_max = 1.271e-06
       Mdot_outer/Mdot_inner = 0.232813
       Lrad/LEdd = 0.527388
       Rson = 5.29849 rg

4. A mixed residual pre-polish helped slightly.  The best repaired checkpoint is:
       outputs/checkpoints/m5_local_mdot_eta_polish_N152_integrated_physE_then_differential_resume/stage_00_etaE_100_N152.npz

       full differential residual = 2.075e-05
       interval_R = 2.075e-05 near R~300.49 rg
       interval_E = 6.865e-06
       mass_residual_max = 1.746e-06
       Mdot_outer/Mdot_inner = 0.232809
       f_adv_global = -0.00389086
       Lrad/LEdd = 0.527513
       Rson = 5.29806 rg

5. The local mass equation is no longer the main bottleneck at eta_E=100.
   The residual floor is a localized radial differential/collocation row near
   the outer/source transition around R~295-300 rg.  The physical energy row is
   already below 1e-5 in the differential audit.
```

Failed or non-decisive paths:

```text
1. Full integrated residuals are much worse:
       integrated, no weighting: ~3.600e-02
       integrated, inverse_sqrt_dx: ~2.015e-01

2. integrated_physical_energy is useful only as a conditioning/pre-polish norm.
   It cannot by itself be used as the accepted physical residual, because the
   state audited back in differential form can still worsen.

3. Alternative outer closures do not cure the R~295 radial row:
       pressure_supported_local_energy/full_slope_match: ~5.537e-03
       entropy_slope: ~9.253e-01
       Robin chi=0.5: ~3.070e-04

4. Generic local least-squares relaxers are not the right fix:
       inner-window relaxers damage the source-annulus/mass budget;
       outer-band radial relaxers near 280-305 rg can raise residuals to
       ~2e-3; radial+energy relax is nearly neutral/slightly worse.

5. Continuing the targeted nested ladder past N152 is unsafe with the current
   seed construction:
       N160 from N152, 100-320 rg seed: 4.310e-04
       N168/N176 from N152: O(1) seed defects
```

Question for GPT:

What is the best next numerical move to reduce the remaining eta_E=100
physical differential residual floor from ~2.1e-5 to <=1e-5 while preserving
the actual differential audit?

Please be concrete.  In particular, should we:

```text
1. Implement a higher-order or trapezoid/midpoint radial momentum residual
   audit specifically for the R~300 rg row?

2. Implement a block/Jacobian-aware correction for the coupled local block
   consisting of neighboring radial, energy, and local-Mdot rows, instead of
   a generic local least-squares relaxer?

3. Change row scaling/nondimensionalization for the radial momentum row near
   the source/outer transition?

4. Add an explicit source-transition buffer/matching condition so the radial
   equation is not forced through a single unresolved source-gradient cell?

5. Treat the remaining interval_R floor as a finite-difference representation
   issue and define a stricter integrated-defect acceptance test instead?  If
   so, what exact equations and cross-audits would prevent us from fooling
   ourselves?
```

Please propose equations or algorithmic steps, expected diagnostics, and clear
acceptance criteria.  Do not recommend lowering eta_E below 100, adding wind
angular momentum, or adding stronger physical wind/heating terms until this
weak local mass-loaded wind checkpoint is made mesh-transfer stable or fails
for a clearly identified physical reason.
