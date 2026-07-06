# Source-Plus-Buffer Interface Formulation Results

Date: 2026-07-06

## Goal

Implement a true source-plus-buffer local formulation so the source-interface finite-volume view and the source-element polynomial/Simpson view can be compared and corrected in one compatible system.

## Code Changes

Primary file:

- `scripts/run_mdot5_local_mdot_eta_continuation.py`

New opt-in mode:

- `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_PLUS_BUFFER_CORRECT=1`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_PLUS_BUFFER_PRODUCTION_POLISH=1`

Main implementation pieces:

- Added source-plus-buffer local block variables for all selected source/buffer nodes:
  - `logu_block`
  - `logT_block`
  - `logMdot_block`
  - cumulative endpoint mass increment `mass_cum`
  - cumulative endpoint energy increment `energy_cum`
- Added compatible interval rows:
  - `mass_interface`: cumulative mass increment equals source-interface FV wind/source integral.
  - `mass_endpoint`: actual endpoint `Mdot` jump equals cumulative mass increment.
  - `mass_element`: cumulative mass increment equals source-element polynomial FV integral.
  - `production_mass`: optional original production residual mass row.
  - `energy_interface`: cumulative energy increment equals source-interface FV energy numerator.
  - `energy_element`: cumulative energy increment equals source-element polynomial FV energy numerator.
  - `energy_balance`: cumulative energy increment should close.
  - `energy_compat`: source-interface and source-element energy numerators agree.
- Added source-element mass terms helper:
  - `_source_element_poly_fv_mass_terms`
- Added sparse finite-difference solve support plus optional hybrid Jacobian support:
  - Sparse FD is default.
  - Hybrid mode can be enabled with `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_PLUS_BUFFER_USE_HYBRID_JAC=1`.
  - The hybrid path injects exact analytic entries for the cumulative mass/energy increment columns, but was too slow for interactive validation.
- Added conservative correction guards:
  - `SOURCE_PLUS_BUFFER_FULL_GUARD_REL`
  - `SOURCE_PLUS_BUFFER_FULL_GUARD_ABS`
  - `SOURCE_PLUS_BUFFER_EXTRA_GUARD_REL`
  - `SOURCE_PLUS_BUFFER_EXTRA_GUARD_ABS`
  - `SOURCE_PLUS_BUFFER_PRESERVE_ACCEPTED=1` by default, so a strict input checkpoint cannot be degraded above `ACCEPT_TOL`.
- Added augmented production polish:
  - variable vector is `[selected x columns, mass_cum, energy_cum]`;
  - default `SOURCE_PLUS_BUFFER_PRODUCTION_VARIABLE_MODE=band`;
  - default band mode uses source/buffer state columns plus source-element stencil nodes, not the whole global vector;
  - residual includes the normal production residual plus the same source-plus-buffer compatibility rows;
  - source-row sparsity maps the local endpoint/stencil/increment dependencies into the augmented variable vector.

## Validation Runs

Common physical setup:

- `Mdot_inner/Edd = 5`
- `Rout = 335 rg`
- `Rinj = 240 rg`
- `f_s = 0.80`
- `eta_E = 100`
- compact C2 stream source
- `torque_delta_l_fraction = +0.005`
- checkpoint seed: `outputs/checkpoints/m5_source_interface_energy_audit_eta100_N164_seed/stage_00_etaE_100_N164.npz`

### Full source-plus-buffer smoke

Output stem:

- `outputs/tables/m5_source_plus_buffer_eta100_N164_smoke.*`

Result:

- Input/global residual remained strict: `final_full = 9.056e-6`.
- No correction accepted.
- Local selected source-plus-buffer score: `0.15939`.
- Source-element audit remained large:
  - `poly_E_max = 0.13415`
  - `FV_E_max = 0.02178`

### Source-band-only, state rows disabled

Output stem:

- `outputs/tables/m5_source_plus_buffer_eta100_N164_bandonly_state0.*`

Result:

- No correction accepted.
- Local source-band selected residual: `0.04848`.
- The unconstrained candidate reduced local source rows but badly violated global mass/full residual, so the guard rejected it.

### Production-mass compatible source-band solve

Output stem:

- `outputs/tables/m5_source_plus_buffer_eta100_N164_bandonly_prodM.*`

Result:

- A tiny damped step was accepted under the first loose guard:
  - alpha `4.883e-4`
  - local selected residual improved `0.0484758 -> 0.0484507`
  - global full residual degraded `9.056e-6 -> 5.418e-5`
- This was judged too permissive because it spent strict global residual for negligible source-band improvement.

### Preserve-strict guarded solve

Output stem:

- `outputs/tables/m5_source_plus_buffer_eta100_N164_bandonly_prodM_preserve.*`

Result:

- The strict checkpoint was preserved:
  - `final_full = 9.056e-6`
  - `accepted_exploratory = true`
  - `mass_residual_max = 3.874e-7`
- No source-plus-buffer correction accepted:
  - `source_plus_buffer_correct_applied = false`
  - `source_plus_buffer_correct_alpha = 0`
- Trial diagnostics show why:
  - alpha `1.2207e-4` would still have raised `full` to `1.346e-5`.
  - preserve-strict guard capped the allowed full residual at `1.0e-5`.
  - all larger alphas were much worse globally.

### Augmented source-plus-buffer production polish

Output stem:

- `outputs/tables/m5_source_plus_buffer_production_eta100_N164_bandonly_nfev8.*`

Settings:

- `SOURCE_PLUS_BUFFER_PRODUCTION_POLISH=1`
- `SOURCE_PLUS_BUFFER_PRODUCTION_VARIABLE_MODE=band`
- `SOURCE_PLUS_BUFFER_PRODUCTION_MAX_NFEV=8`
- `SOURCE_PLUS_BUFFER_STATE_WEIGHT=0`
- `SOURCE_PLUS_BUFFER_FRACTIONS=0.5`

Result:

- The augmented production polish is now computationally usable in band mode:
  - active state variables: `42`
  - total augmented variables: `64`
  - source-plus-buffer rows: `124`
- It found an accepted strict-preserving damped step:
  - `source_plus_buffer_production_applied = true`
  - `alpha = 2.4414e-4`
  - `final_full = 9.354e-6`
  - `mass_residual_max = 3.874e-7`
- The source-band improvement is real but tiny:
  - selected source-plus-buffer residual `0.0484758 -> 0.0484633`
  - mass group `0.0120775 -> 0.0120765`
  - energy group `0.00712424 -> 0.00712378`
  - source-band extra `0.1593905 -> 0.1593848`
- The undamped candidate would improve the source-band rows much more:
  - candidate selected residual `0.02295`
  - candidate source-band extra `0.13741`
  - but candidate full residual `1.523e-2`, so it is rejected.
- The line search shows the same structural conflict as the local-only formulation:
  - useful source-band moves rapidly violate the strict global residual;
  - strict-preserving moves are so damped that compatibility barely changes.

## Interpretation

The new formulation is implemented and internally exposes the intended compatibility rows between:

- endpoint mass/energy increments,
- source-interface FV integrals,
- source-element polynomial/Simpson integrals,
- and original production mass rows.

However, the eta=100 N164 validation says the current source-band state cannot be locally or augmented-production repaired into polynomial/interface agreement without immediately increasing the global differential residual above the strict tolerance. The dominant mismatch is still a source-band representation defect, not a sonic or outer-boundary failure.

The conservative guard is important: without it, the local corrector accepts tiny improvements that are not scientifically useful because they degrade the strict checkpoint.

## Tests

Commands run:

```bash
PYTHONPYCACHEPREFIX=/tmp/imbh_pycache python -m py_compile scripts/run_mdot5_local_mdot_eta_continuation.py
PYTHONPATH=src PYTHONPYCACHEPREFIX=/tmp/imbh_pycache python -m pytest tests/test_transonic_local.py tests/test_transonic_collocation.py -q
PYTHONPATH=src PYTHONPYCACHEPREFIX=/tmp/imbh_pycache python -m pytest -q
```

Results:

- Compile passed.
- Focused tests: `73 passed, 2 subtests passed`.
- Full tests: `160 passed, 2 subtests passed`.

## Recommended Next Step

The compatible source-plus-buffer formulation is now available both as a diagnostic/local correction layer and as an augmented production polish. It still does not remove the eta=100 source-band defect. The next useful move is not to loosen the guard. Instead:

1. Add analytic/local derivatives for the expensive source-interface and source-element energy numerator rows with respect to `(logu, logT, logMdot)` inside the source band.
2. Try a row-local production residual option that includes only source-band/base rows during the augmented pre-polish, while continuing to audit full residual globally in the line search.
3. Retry eta=100 with the strict guard still on.
4. Only if eta=100 becomes representation-strict should eta=90 be retried.
