# Scientific Entry Points

The repository still contains legacy research runners because many import
helpers from sibling scripts. Their dependency map is recorded in
`docs/manifests/script_inventory.csv`. Do not remove a wrapper until reusable
logic has moved into `src/` and numerical parity is tested.

## Current P0 Audits

- `run_mdot5_endpoint_validity_audit.py`
- `run_mdot5_angular_momentum_ledger_audit.py`
- `run_mdot5_independent_outer_manifold_search.py`

## Current Phase-DAE Workflows

- `run_mdot5_global_phase_dae_production.py`
- `run_mdot5_phase_dae_exit_refinement.py`
- `run_mdot5_phase_critical_globalization.py`
- `run_mdot5_phase_critical_classification.py`
- `run_mdot5_local_mdot_eta_continuation.py` (legacy monolith and shared helper
  source; retain until refactored)

## Benchmark Workflows

- `run_standard_slim_high_mdot_no_wind_ladder.py`
- `run_standard_slim_stream_anchor_regression.py`

## Repository Maintenance

- `build_repository_cleanup_inventory.py`
- `build_and_verify_legacy_archive.py`
- `build_canonical_results.py`
- `check_repository_hygiene.py`

Older transonic and source-band scripts are retained as legacy numerical
experiments. Current scientific claims and review order are defined in
`docs/PROJECT_STATUS.md`, not inferred from script filenames.
