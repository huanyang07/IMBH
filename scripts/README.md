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
- `run_causal_nonlinear_fiber_audit_wp10c8o.py` constructs exact finite-state
  pairs on the richest 34-coordinate fiber, evaluates production observables
  and fresh coarse rates, certifies the decisive pair against independent
  full-DAE/Schur and path-storage checks, and confirms only that N64
  counterexample at N128. It changes no production operator and launches no
  healing trajectory.
- `run_causal_natural_healing_wp10c8p.py` evolves the frozen decisive
  WP10c8o pairs with synchronized fixed-step BDF1-start/BDF2 trajectories.
  It compares complete coarse/fine histories, decomposes the full interface-4
  mass/angular-momentum/Killing-energy flux, reconciles interface impulses
  with adjacent-shell ledgers, and classifies only whether the unresolved
  transport heals within `0.025 s`. It changes no production operator or
  reduced coordinate.
- `run_causal_inner_mode_healing_wp10c8t.py` replays and serializes the exact
  increment-primary BDF history of the binding WP10c8s inner-shell mode,
  continues a refined N64 `h/h/2` pair to `0.125 s` without another BDF1
  startup, and bounds both its complete slow-rate decay and all 34
  accumulated initial-slip components. It changes no production operator,
  reduced coordinate, or physical model.
- `run_causal_inner_mode_n128_confirmation_wp10c8t.py` constructs an exact
  finite-amplitude equal-`q_34` N128 mode-0 pair from the matched complete-rate
  direction and runs only its nested fixed-BDF2 `h/h/2` confirmation through
  `0.125 s`. It compares endpoint temporal control, N64/N128 rate direction,
  and shell-0 localization without changing the production operator or
  authorizing reduced evolution.

## Repository Maintenance

- `build_repository_cleanup_inventory.py`
- `build_and_verify_legacy_archive.py`
- `build_canonical_results.py`
- `check_repository_hygiene.py`

Older transonic and source-band scripts are retained as legacy numerical
experiments. Current scientific claims and review order are defined in
`docs/PROJECT_STATUS.md`, not inferred from script filenames.
