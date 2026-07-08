# GPT Prompt: Mdot=5 Eta Continuation Sonic/Inner-Mass Wall

Please review the latest IMBH repository state on GitHub, focusing on the
Mdot_inner/Edd=5 local-Mdot eta continuation work.

Current implementation and results:

- Main script:
  `scripts/run_mdot5_local_mdot_eta_continuation.py`
- Latest summary note:
  `Note/CODEX_MDOT5_ADAPTIVE_ACTIVE_WINDOW_ETA_CONTINUATION_RESULTS.md`
- Important result tables:
  - `outputs/tables/m5_eta_two_pass_sonic12_98p4375_N164.json`
  - `outputs/tables/m5_eta_two_pass_sonic12_from98p4375_N164.json`
  - `outputs/tables/m5_eta_two_pass_sonic12_98p21875_N164.json`
  - `outputs/tables/m5_eta_two_pass_sonic12_anchor1em3_98p21875_N164.json`
  - `outputs/tables/m5_eta_two_pass_sonic12_prepass12_98p21875_N164.json`
  - `outputs/tables/m5_eta_two_pass_sonic12_inner_mass_98p21875_N164.json`
  - `outputs/tables/m5_eta_two_pass_sonic15_inner_mass_98p21875_N164.json`
- Latest strict compatible checkpoint:
  `outputs/checkpoints/m5_eta_two_pass_sonic12_from98p4375_N164/stage_03_etaE_98p25_N164.npz`

Summary:

- A two-pass sonic prepass was added behind
  `IMBH_MDOT5_LOCAL_MDOT_ETA_ACTIVE_MASS_PROFILE_TWO_PASS_SONIC=1`.
- The prepass solves a small dense inner block containing sonic rows, the
  inner Mdot row, inner variables, and global variables, then line-searches by
  the compatible source-band replacement score before the existing active
  source-band corrector.
- This converted the old `eta_E=98.4375` non-strict point into a strict
  compatible source-band checkpoint:
  `source_band_global_replacement_final_score = 9.189e-6`.
- The strict compatible ladder now reaches `eta_E=98.25`:
  `score = 9.857e-6`.
- The next point, `eta_E=98.21875`, is still slightly non-strict:
  `score = 1.0185e-5`.
- Leading rows at the new wall:
  - `old_sonic_pivot` at `R~5.30 rg`, about `1.0185e-5`
  - `old_mass` at `R~5.93 rg`, about `1.0089e-5`
- Mass-increment compatibility remains strict at the wall:
  `active_mass_increment_int/link ~9.41e-6`.
- Quick variants did not fix the wall:
  - softer sonic prepass anchor `1e-3`: unchanged;
  - wider sonic prepass to `R<12 rg`: unchanged;
  - include inner old-mass rows in prepass to `R<8 rg`: worse;
  - include inner old-mass rows in prepass to `R<15 rg`: worse.
- Full regression tests passed locally: `160 passed`.

Important interpretation:

- The compatible source-band replacement score is the relevant current
  strictness metric. The legacy midpoint `final_full` remains large, around
  `1.11`, for these source-band-compatible runs.
- The latest wall is not a finite-volume mass-increment bookkeeping problem.
  It is a coupled inner sonic plus old-mass residual floor.
- The staged two-pass prepass helped, but simple broadening or adding mass rows
  to that prepass makes the line-search direction worse.

Please advise on the next numerical formulation. In particular:

1. Should we replace the staged prepass plus active corrector with one coupled
   inner-window least-squares objective containing sonic rows, nearby old-mass
   rows, and mass-increment compatibility rows together?
2. What row scaling/weighting is physically and numerically appropriate so the
   sonic pivot, old mass rows, and mass-increment rows can all improve without
   one family dominating?
3. Should the inner old-mass rows be represented in the compatible source-band
   formulation differently, rather than reusing the old midpoint mass rows?
4. Is there a better local variable basis near the sonic point, e.g. replacing
   `logR_son`/`lambda0` updates with a sonic tangent or regularity coordinate?
5. What acceptance diagnostics should be required before continuing below
   `eta_E=98.21875` or starting N192/N224 mesh validation?
