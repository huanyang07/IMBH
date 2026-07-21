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

## Current Causal Reduction Audits

- `run_causal_moment_sufficiency_audit_wp10c8i.py` constructs the locked
  storage-consistent five-shell moment ladder and performs the offline
  finite-time constraint-null audit. Its moment decision remains conditional
  on tangent certification.
- `run_causal_tangent_certification_wp10c8j.py` independently attempts the
  evolving-anchor smooth-tangent repair and certifies it only when every
  declared gate passes. It also applies the strict
  finite-neighborhood Rusanov contract, and may authorize—but never launches—
  a separate unchanged WP10c8i repeat. It runs no new truth trajectory,
  nonlinear lift, healing burst, closure, or reduced evolution.

## Repository Maintenance

- `build_repository_cleanup_inventory.py`
- `build_and_verify_legacy_archive.py`
- `build_canonical_results.py`
- `check_repository_hygiene.py`

Older transonic and source-band scripts are retained as legacy numerical
experiments. Current scientific claims and review order are defined in
`docs/PROJECT_STATUS.md`, not inferred from script filenames.
