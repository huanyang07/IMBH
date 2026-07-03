# GPT Prompt: Outer-Buffer Source-Fraction Continuation

Please review the current GitHub repository status for the IMBH/QPE numerical
minidisk project, focusing on the latest outer-buffer stream-fed branch work.

Key files to read first:

- `Note/CODEX_COMPACT_SOURCE_OUTER_BUFFER_RESULTS.md`
- `scripts/run_standard_slim_stream_mass_annulus_scan.py`
- `scripts/run_standard_slim_stream_interval_profile.py`
- `src/imri_qpe/layer3_minidisk_1d/transonic_collocation.py`
- `outputs/tables/high_mdot_stream_compact_outer_buffer_ladder_validation_profile.md`
- `outputs/tables/high_mdot_stream_compact_outer_buffer_fs08625_to090_N896.md`

Current accepted status:

- No-wind compact stream-fed reservoir-formulation branch.
- `Mdot_inner/Edd = 2`
- `Rout = 335 rg`
- fixed physical stream injection center `Rinj = 240 rg`
- compact C2 source
- `torque_delta_l_fraction = +0.005`
- no wind and no stream heating
- integrated interval residual
- outer-buffer formulation with `R_buffer = 300 rg`
- baseline buffer weights `(R,E,B) = (1e-3, 1e-3, 1e-4)`

What changed:

1. Implemented an opt-in outer-tail/buffer formulation in the collocation
   residual.
2. Added a split residual audit:
   - physical/source domain, `R < R_buffer`;
   - softened outer buffer, `R_buffer < R < Rout`;
   - terminal boundary.
3. Staged radius continuation from `Rout=310` to `335 rg` now succeeds.
4. The `Rout=335 rg`, `f_s=0.8625` reservoir branch passes first-pass
   robustness checks:
   - `N=768`, `896`, `1024`;
   - `R_buffer=295`, `300`, `305 rg`;
   - stricter buffer weights `(3e-3,3e-3,3e-4)`.
5. Source-fraction continuation from `f_s=0.8625` continued to at least
   `f_s=0.8759639587`.

Important numerical results:

- `Rout=335`, `f_s=0.8625`, `N=896`:
  - final weighted residual `2.342e-07`
  - physical raw energy residual max `1.185e-05`
  - `f_adv_global = 0.20415`
  - `f_adv_inner = 0.09622`
  - `Lrad/LEdd = 0.86754`
  - `Rson = 4.65992 rg`

- Highest clean source-fraction anchor from the latest run:
  - `f_s = 0.8759639587`
  - final weighted residual `1.242e-07`
  - physical raw energy residual max `1.473e-05`
  - `f_adv_global = 0.20422`
  - `f_adv_inner = 0.09443`
  - `Lrad/LEdd = 0.86717`
  - `Rson = 4.65992 rg`

Main caveat:

This is a reservoir-formulation branch. The raw unweighted residual inside the
outer buffer is intentionally large and should not be used as the physical
convergence criterion. The new split audit should be used to judge physical
domain convergence.

Current bottleneck:

The old `f_s ~ 0.86` wall is not a physical endpoint. The branch continues
smoothly past it. However, near `f_s ~ 0.876`, continuation becomes expensive:

- adaptive step reaches the minimum `df_s = 2.5e-4`;
- accepted steps often cost `~100-160` function evaluations;
- physical-domain residual can spike to `~1e-3`, then relax back to `~1e-5`
  after small steps and remeshing;
- global physical diagnostics remain smooth.

Questions for GPT:

1. What is the best next numerical improvement to push from
   `f_s=0.8759639587` toward `f_s=0.90`?
2. Should we implement a true source-fraction tangent predictor
   `J_z dz/df_s = -F_f_s`, pseudo-arclength continuation, or a local
   source/interface remesh strategy first?
3. What acceptance criteria should distinguish a real physical source-fraction
   limit from a numerical predictor/remesh bottleneck?
4. Given the split-audit results, is it scientifically acceptable to use this
   outer-buffer formulation as a finite reservoir boundary for the no-wind
   stream-fed branch?
5. Should the next physics step be:
   - continue `f_s` to `0.90` at `Mdot_inner/Edd=2`;
   - retry `Mdot_inner/Edd=3` and `5` finite-Rout no-wind branches;
   - add stream heating;
   - or implement a more explicit two-domain buffer before any physics change?

Please give a concrete implementation plan with the first 3 to 5 tasks, and
prioritize changes that reduce wasted Newton/remesh cost while preserving the
physical-domain split-audit validation.
