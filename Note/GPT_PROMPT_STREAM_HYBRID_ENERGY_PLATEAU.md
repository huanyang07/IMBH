# GPT prompt: stream-fed outer-buffer hybrid-energy plateau

Please review the latest IMBH/QPE repository state and advise on the next numerical move.

Repository: https://github.com/huanyang07/IMBH

Important current files:
- `Note/CODEX_PHYSICAL_GATE_AUTOMATION_RESULTS.md`
- `Note/CODEX_DIFF_CLEANUP_GATE_RESULTS.md`
- `Note/CODEX_TARGETED_REMESH_REFINEMENT_RESULTS.md`
- `Note/CODEX_GRID_HOMOTOPY_HYBRID_ENERGY_RESULTS.md`
- `scripts/run_standard_slim_stream_mass_annulus_scan.py`
- `scripts/run_standard_slim_stream_grid_homotopy.py`
- `src/imri_qpe/layer3_minidisk_1d/transonic_collocation.py`
- `tests/test_transonic_collocation.py`
- `outputs/tables/high_mdot_stream_outer_buffer_phys_gate_hybrid_quartersteps_accept3e5.md`
- `outputs/tables/high_mdot_stream_outer_buffer_phys_gate_hybrid_eighthsteps_0898078125_to089809375.md`

Current physical setup:
- No wind, no stream heating.
- `Mdot_inner/Edd = 2`
- `Rout = 335 rg`
- `Rinj = 240 rg`
- compact C2 mass source
- `torque_delta_l_fraction = +0.005`
- outer-buffer/source-tail formulation
- high-resolution branch at `N = 896`
- source fraction continuation in `f_s`

What changed:
1. Added a stricter physical gate based on the real differential energy residual (`physical_E`), so acceptance is no longer allowed to hide behind integrated/averaged interval residuals.
2. Added a hybrid interval residual mode, `integrated_physical_energy`, in which the solver can keep integrated radial residuals while retaining the physical differential energy residual in the square residual.
3. Aligned forced-hybrid acceptance tolerances with the intended physical gate instead of an artificially tighter default residual.
4. Added grid-homotopy and direct high-N remap tests.
5. Added a regression test for the new residual form. Current test result: `147 passed`.

Main results:
- Direct high-N remap is not enough:
  - N1024 target grid: `physical_E = 2.256e-2`
  - N1024 resample grid: `physical_E = 1.098e-3`
- Grid homotopy from the clean `f_s=0.8980625` anchor toward the gentle target grid failed even for extremely small mesh mixing:
  - smallest tested `eta = 0.000390625`
  - `physical_E = 5.891e-5`
- Hybrid residual full step `f_s=0.8980625 -> 0.898125` improved but did not meet the physical gate:
  - best long run: `physical_E = 3.086e-5`
- Hybrid half step to `f_s=0.89809375` also missed:
  - `physical_E = 3.062e-5`
- Hybrid quarter-step scan with physical-gate-aligned acceptance produced one new strict-ish accepted point:
  - `f_s = 0.898078125`
  - `physical_E = 2.989e-5`
  - `nfev = 74`
  - checkpoint:
    `outputs/checkpoints/high_mdot_stream_outer_buffer_phys_gate_hybrid_quartersteps_accept3e5/phys_gate_hybrid_quartersteps_accept3e5_mass_0p898078125_torque_0p005_mdot_2_N896.npz`
- Continuing from this new clean point still stalls immediately:
  - next quarter step to `f_s=0.89809375`: `physical_E = 3.025e-5`
  - eighth step to `f_s=0.8980859375`: `physical_E = 3.116e-5`

Interpretation so far:
- The previous apparent progress to larger `f_s` was partly due to weak/integrated residual accounting.
- With the honest physical energy residual in the square system, the branch currently has a real numerical plateau just above `f_s ~= 0.898078`.
- The obstruction is still not a sonic failure. It is dominated by the outer/source energy residual and mesh/source/boundary handling.
- The plateau is very close to the adopted `physical_E <= 3e-5` gate, so please distinguish between:
  1. a genuine numerical/physical branch obstruction,
  2. an overly strict mesh-dependent tolerance,
  3. a Newton merit/scaling/Jacobian issue,
  4. a residual-definition issue in the source/outer-buffer energy equation.

Questions for GPT:
1. What is the most principled next solver modification to break the `physical_E ~= 3e-5` plateau?
2. Should we implement an energy-focused block merit or weighted line search, a better scaled Jacobian, an analytic/local Jacobian for interval_E, or pseudo-arclength in `f_s`?
3. Should the source annulus energy equation be reformulated in conservative/integral form while still auditing a physical differential residual separately?
4. Is `physical_E <= 3e-5` too strict for the current `N=896` outer-buffer/source-tail discretization, and what mesh-convergence criterion would be more defensible?
5. What minimal experiment would tell whether this is Newton conditioning versus a real discretization/model closure problem?

Please propose a concrete next sequence of 3-6 implementation/testing steps. Avoid adding wind or stream heating until this no-wind compact-source branch is either robustly continued or cleanly diagnosed.
