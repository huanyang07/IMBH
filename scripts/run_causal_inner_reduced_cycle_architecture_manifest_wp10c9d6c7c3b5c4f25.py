#!/usr/bin/env python3
"""Freeze the offline/online reduced-cycle architecture and cost contract."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25"
CLASSIFICATION = (
    "reduced_cycle_architecture_manifest_frozen_"
    "evidence_only_identifiability_authorized"
)
ANALYZED_COMMIT = "a1284d398cc83a6efe997a3350a5f635c16fc825"
ANALYZED_PARENT = "8b85e4bdc7de6a2310090472959353ab64bb6bd4"
ANALYZED_TREE = "ab524a0c4c322a8587264958aeb7240b5a27ac9c"

ARTIFACT = "causal_inner_reduced_cycle_architecture_manifest_wp10c9d6c7c3b5c4f25"
ARTIFACT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_reduced_cycle_architecture_manifest_"
    "wp10c9d6c7c3b5c4f25.py"
)
THIS_TEST = (
    "tests/test_causal_inner_reduced_cycle_architecture_manifest_"
    "wp10c9d6c7c3b5c4f25.py"
)
NEXT_RUNNER = (
    "scripts/run_causal_inner_reduced_cycle_identifiability_"
    "wp10c9d6c7c3b5c4f25a.py"
)
NEXT_TEST = (
    "tests/test_causal_inner_reduced_cycle_identifiability_"
    "wp10c9d6c7c3b5c4f25a.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_REDUCED_CYCLE_ARCHITECTURE_"
    "MANIFEST_WP10C9D6C7C3B5C4F25_2026-08-17.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

LATEST_RESULT = ROOT / (
    "results/canonical/causal_inner_face36_fixed_q_operational_timestep_"
    "predictor_rung_wp10c9d6c7c3b5c4f24e14x/summary.json"
)
PRIMARY_WARM = ROOT / (
    "results/canonical/causal_inner_face36_fixed_q_primary_retry_"
    "wp10c9d6c7c3b5c4f24e14l/metrics_warm_1.json"
)
MEMORY_SCREEN = ROOT / (
    "results/canonical/causal_inner_face36_augmented_memory_screen_"
    "wp10c9d6c7c3b5c4f13/summary.json"
)
TWO_MODE_SCREEN = ROOT / (
    "results/canonical/causal_inner_face36_q_plus_a_reaction_coordinate_"
    "preflight_wp10c9d6c7c3b5c4f15/summary.json"
)
SIX_MODE_REPLAY = ROOT / (
    "results/canonical/causal_inner_face36_six_mode_fine_dynamic_coordinate_"
    "replay_wp10c9d6c7c3b5c4f20/summary.json"
)
LEADING_TWO_HMM = ROOT / (
    "results/canonical/causal_inner_face36_leading_two_plus_hmm_manifest_"
    "wp10c9d6c7c3b5c4f21/summary.json"
)
FIBER_REPORT = ROOT / (
    "docs/reports/current/CODEX_CAUSAL_NONLINEAR_FIBER_AUDIT_"
    "WP10C8O_RESULTS_2026-07-22.md"
)
HEALING_REPORT = ROOT / (
    "docs/reports/current/CODEX_CAUSAL_INNER_MODE_HEALING_"
    "WP10C8T_RESULTS_2026-07-24.md"
)
ONE_ZONE_SOURCE = ROOT / "src/imri_qpe/layer3_minidisk_1d/limit_cycle.py"
ONE_ZONE_TEST = ROOT / "tests/test_layer3_minidisk.py"

DAY_SECONDS = 86_400.0
FIDUCIAL_CYCLE_DAYS = 6.7
FIDUCIAL_CYCLE_SECONDS = FIDUCIAL_CYCLE_DAYS * DAY_SECONDS
WALL_BUDGET_DAYS = 3.0
WALL_BUDGET_SECONDS = WALL_BUDGET_DAYS * DAY_SECONDS
CERTIFIED_TRUTH_TIMESTEP_SECONDS = 1.0e-7
REFERENCE_WARM_ROOT_SECONDS = 1289.6064693750814
MAXIMUM_ONLINE_MACROSTEPS = 100_000
MAXIMUM_ONLINE_WALL_SECONDS_PER_STEP = 1.0
PRIMARY_RADIAL_CELLS = 16
VALIDATION_RADIAL_CELLS = (8, 16, 32)
CELL_STORAGE_DIMENSION = 5
EXPLICIT_MEMORY_DIMENSION = 2
KERNEL_MEMORY_CANDIDATES = (0, 2, 4, 6)


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, int):
        return value
    return value


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _tracked_tree_clean() -> bool:
    return not _git("status", "--short", "--untracked-files=no")


def _input_hashes() -> dict[str, str]:
    paths = (
        LATEST_RESULT,
        PRIMARY_WARM,
        MEMORY_SCREEN,
        TWO_MODE_SCREEN,
        SIX_MODE_REPLAY,
        LEADING_TWO_HMM,
        FIBER_REPORT,
        HEALING_REPORT,
        ONE_ZONE_SOURCE,
        ONE_ZONE_TEST,
    )
    return {str(path.relative_to(ROOT)): _sha(path) for path in paths}


def _validate_inputs() -> dict:
    if _git("rev-parse", ANALYZED_COMMIT) != ANALYZED_COMMIT:
        raise RuntimeError("analyzed fixed-Q result commit changed")
    if _git("rev-parse", f"{ANALYZED_COMMIT}^") != ANALYZED_PARENT:
        raise RuntimeError("analyzed fixed-Q result parent changed")
    if _git("rev-parse", f"{ANALYZED_COMMIT}^{{tree}}") != ANALYZED_TREE:
        raise RuntimeError("analyzed fixed-Q result tree changed")

    latest = _read(LATEST_RESULT)
    warm = _read(PRIMARY_WARM)
    memory = _read(MEMORY_SCREEN)
    two_mode = _read(TWO_MODE_SCREEN)
    six_mode = _read(SIX_MODE_REPLAY)
    hybrid = _read(LEADING_TWO_HMM)
    if latest["classification"] != "operational_timestep_rung_2e7_failed":
        raise RuntimeError("latest doubled-step rejection changed")
    if latest["accepted_BDF2_roots"] != 0 or latest["accepted_horizon_seconds"] != 0.0:
        raise RuntimeError("rejected doubled step was incorrectly propagated")
    if not warm["root_passed"] or warm["timestep_seconds"] != CERTIFIED_TRUTH_TIMESTEP_SECONDS:
        raise RuntimeError("certified warm truth root changed")
    if not math.isclose(warm["root_wall_seconds"], REFERENCE_WARM_ROOT_SECONDS):
        raise RuntimeError("reference warm-root cost changed")
    if memory["classification"] != "face36_compact_persistent_observable_memory_detected":
        raise RuntimeError("observable-memory evidence changed")
    if memory["cross_resolution"]["minimum_principal_cosine"] < 0.99998:
        raise RuntimeError("leading observable-memory subspace changed")
    if two_mode["minimum_passing_output_oriented_dimension"] != 6:
        raise RuntimeError("two-mode reconstruction rejection changed")
    if two_mode["two_mode_significant_direction_gate_passed"]:
        raise RuntimeError("two-mode significant-direction rejection changed")
    if six_mode["passed"] or six_mode["cross_resolution"]["minimum_full_subspace_projector_cosine"] >= 0.9:
        raise RuntimeError("six-mode dynamic rejection changed")
    if not hybrid["leading_two_state_coordinate_supported"]:
        raise RuntimeError("leading-two coordinate support changed")
    if hybrid["direct_two_mode_output_closure_supported"]:
        raise RuntimeError("direct two-mode output closure was relabelled")
    return {
        "latest": latest,
        "warm": warm,
        "memory": memory,
        "two_mode": two_mode,
        "six_mode": six_mode,
        "hybrid": hybrid,
    }


def _cost_model() -> dict:
    truth_steps_per_cycle = FIDUCIAL_CYCLE_SECONDS / CERTIFIED_TRUTH_TIMESTEP_SECONDS
    direct_wall_seconds = truth_steps_per_cycle * REFERENCE_WARM_ROOT_SECONDS
    primary_dimension = (
        PRIMARY_RADIAL_CELLS * CELL_STORAGE_DIMENSION
        + EXPLICIT_MEMORY_DIMENSION
        + max(KERNEL_MEMORY_CANDIDATES)
    )
    validation_dimensions = {
        str(cells): (
            cells * CELL_STORAGE_DIMENSION
            + EXPLICIT_MEMORY_DIMENSION
            + max(KERNEL_MEMORY_CANDIDATES)
        )
        for cells in VALIDATION_RADIAL_CELLS
    }
    return {
        "fiducial_cycle_days": FIDUCIAL_CYCLE_DAYS,
        "fiducial_cycle_seconds": FIDUCIAL_CYCLE_SECONDS,
        "cycle_value_role": "phenomenological_runtime_target_not_a_truth_prediction",
        "wall_budget_days": WALL_BUDGET_DAYS,
        "wall_budget_seconds": WALL_BUDGET_SECONDS,
        "certified_truth_timestep_seconds": CERTIFIED_TRUTH_TIMESTEP_SECONDS,
        "reference_warm_root_wall_seconds": REFERENCE_WARM_ROOT_SECONDS,
        "direct_truth_steps_per_cycle": truth_steps_per_cycle,
        "direct_truth_wall_hours_per_microsecond": (
            REFERENCE_WARM_ROOT_SECONDS * 10.0 / 3600.0
        ),
        "direct_truth_wall_days_per_millisecond": (
            REFERENCE_WARM_ROOT_SECONDS * 10_000.0 / DAY_SECONDS
        ),
        "direct_truth_wall_years_per_cycle": direct_wall_seconds / (365.25 * DAY_SECONDS),
        "minimum_required_end_to_end_speedup": direct_wall_seconds / WALL_BUDGET_SECONDS,
        "online_truth_solver_calls_per_macrostep": 0,
        "maximum_online_macrosteps_per_cycle": MAXIMUM_ONLINE_MACROSTEPS,
        "minimum_average_macrostep_seconds": FIDUCIAL_CYCLE_SECONDS / MAXIMUM_ONLINE_MACROSTEPS,
        "maximum_online_wall_seconds_per_step": MAXIMUM_ONLINE_WALL_SECONDS_PER_STEP,
        "primary_continuous_state_dimension": primary_dimension,
        "validation_continuous_state_dimensions": validation_dimensions,
        "maximum_primary_continuous_state_dimension": 96,
        "maximum_validation_continuous_state_dimension": 192,
    }


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "new_truth_trajectory_executed": False,
        "purpose": (
            "replace_online_fixed_Q_microstepping_with_an_offline_identified_"
            "conservative_finite_memory_macro_model"
        ),
        "preserved_evidence": {
            "fixed_Q_local_history_and_multistep_continuation_supported": True,
            "largest_certified_truth_timestep_seconds": CERTIFIED_TRUTH_TIMESTEP_SECONDS,
            "doubled_timestep_rejected_without_accepted_state": True,
            "no_physical_failure_at_an_accepted_state": True,
            "raw_pointwise_horizon_flux_remains_rejected": True,
            "slow_export": "certified_face36_exterior_partition",
        },
        "runtime_contract": _cost_model(),
        "offline_online_split": {
            "offline_truth_roles": (
                "selected_constrained_steady_roots",
                "local_complete_Jacobians_and_transfer_functions",
                "short_fixed_Q_bursts_only_where_memory_or_switching_is_unresolved",
                "middle_layout_training_with_sparse_fine_validation",
            ),
            "online_truth_solver_calls": 0,
            "online_HMM_microbursts": False,
            "online_model_may_use_only": (
                "tabulated_branch_maps",
                "stable_interpolants",
                "finite_memory_auxiliary_ODEs",
                "conservative_hybrid_event_maps",
            ),
        },
        "macro_state": {
            "radial_cell_candidates": VALIDATION_RADIAL_CELLS,
            "primary_radial_cells": PRIMARY_RADIAL_CELLS,
            "cellwise_exact_or_declared_storage": (
                "mapped_mass",
                "mapped_angular_momentum",
                "mapped_killing_energy",
                "column_thermal_content_candidate",
                "relaxing_stress_storage_candidate",
            ),
            "leading_cross_grid_state_amplitudes": 2,
            "finite_memory_dimensions_to_screen": KERNEL_MEMORY_CANDIDATES,
            "hybrid_branch_label": ("cold", "hot", "transition"),
            "responsive_height_one_form_is_not_an_absolute_coordinate": True,
        },
        "conservative_semidiscrete_form": {
            "cell_balance": "dU_i/dt=Phi_(i-1/2)-Phi_(i+1/2)+S_i-W_i",
            "interior_face_flux_is_single_valued": True,
            "interior_fluxes_telescope_exactly": True,
            "binding_global_ledgers": ("mass", "angular_momentum", "killing_energy"),
            "inner_boundary": "certified_face36_extraction_partition",
            "raw_horizon_face_flux_forbidden": True,
        },
        "finite_memory_boundary_closure": {
            "continuous_form": (
                "Phi_inner(t)=Phi_qs(U,b)+integral_0^t_K_b(t-s;U)dU/ds_ds"
            ),
            "rational_approximation": (
                "dm_j/dt=-m_j/tau_j+B_j(U,b)dU/dt"
            ),
            "online_flux_map": (
                "Phi_inner=Phi_qs(U,b)+C_a(U,b)a+C_m(U,b)m+Phi_nl(U,a,m,b)"
            ),
            "stable_poles_required": True,
            "passivity_or_declared_dissipation_gate_required": True,
            "memory_dimension_selected_prospectively": True,
        },
        "hybrid_branch_contract": {
            "branches": ("cold", "hot", "transition"),
            "separate_switch_surfaces": ("g_up(U,a,m)=0", "g_down(U,a,m)=0"),
            "hysteresis_required": True,
            "mass_angular_momentum_energy_continuous_across_switch": True,
            "reset_may_change_only_declared_auxiliary_memory": True,
            "reset_impulse_ledger_must_close": True,
            "phenomenological_switches_are_exploratory_not_predictive": True,
        },
        "cycle_physics_scope": {
            "current_6p7_day_cycle_source": "existing_one_zone_relaxation_oscillator",
            "current_cycle_is_predictive_from_certified_truth": False,
            "exploratory_route": (
                "use_frozen_one_zone_thresholds_only_to_test_runtime_events_and_ledgers"
            ),
            "predictive_route_requires": (
                "certified_cold_branch",
                "certified_hot_branch",
                "fold_or_hysteresis_surfaces",
                "distributed_tide_and_wind_if_physically_relevant",
                "held_out_cycle_validation",
            ),
        },
        "predeclared_identifiability_candidates": {
            "negative_controls": (
                "global_Q3_instantaneous_Markov_closure",
                "global_Q3_plus_two_mode_direct_output_closure",
                "global_Q3_plus_six_explicit_modes",
                "leading_two_plus_online_HMM_truth_microbursts",
            ),
            "preferred": (
                "cellwise_Q5_conservative_FV_plus_two_stable_modes_plus_"
                "finite_memory_plus_hybrid_branch"
            ),
            "fallback": (
                "larger_conservative_coarse_radial_PDE_without_compact_kernel"
            ),
            "kernel_memory_dimensions": KERNEL_MEMORY_CANDIDATES,
            "post_result_candidate_addition_forbidden": True,
        },
        "evidence_only_screen": {
            "new_nonlinear_trajectory_count": 0,
            "new_fixed_Q_root_count": 0,
            "new_tangent_propagation_count": 0,
            "must_hash_validate_all_inputs": True,
            "may_select_architecture_but_not_fit_coefficients": True,
            "may_not_authorize_online_solver_implementation": True,
            "decision_branches": {
                "architecture_supported_coefficients_unidentified": (
                    "authorize_definitions_only_offline_closure_database_manifest"
                ),
                "no_conservative_finite_memory_architecture_supported": (
                    "retain_conservative_coarse_PDE_and_reassess_reduction"
                ),
            },
        },
        "validation_ladder_after_offline_identification": (
            "committed_5_to_20ms_response_replay",
            "held_out_16ms_and_20ms_local_fixed_Q_checks",
            "selected_cold_hot_transition_truth_snippets",
            "exploratory_6p7_day_cycle_runtime_and_ledger_demo",
            "predictive_cycle_only_after_missing_physics_is_certified",
        ),
        "hard_stops": (
            "do_not_optimize_the_truth_residual_as_the_primary_route_to_cycle_time",
            "do_not_run_more_fixed_Q_roots_in_the_evidence_only_screen",
            "do_not_call_online_HMM_computationally_feasible",
            "do_not_relax_the_primitive_change_or_residual_gates",
            "do_not_claim_predictive_cycle_physics_from_the_one_zone_demo",
            "do_not_implement_the_online_reduced_solver_before_coefficients_are_identified",
            "do_not_use_raw_pointwise_horizon_flux",
        ),
        "authorized_next": "WP10c9d6c7c3b5c4f25a_evidence_only_identifiability_screen",
    }


def _update_catalog(summary: dict) -> None:
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(ARTIFACT_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({
                "case": ARTIFACT,
                "path": str(path.relative_to(ROOT)),
                "bytes": str(path.stat().st_size),
                "sha256": _sha(path),
                "scientific_status": "PROSPECTIVE",
            })
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case", "path", "bytes", "sha256", "scientific_status"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(ARTIFACT_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": summary["passed"],
    }
    catalog.update({
        "case_count": len({row["case"] for row in rows}),
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "all_payload_hashes_recorded": True,
        "latest_source_parent_commit": ANALYZED_COMMIT,
        "latest_work_package": WORK_PACKAGE,
    })
    _write(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    inputs = _validate_inputs()
    if not _tracked_tree_clean():
        raise RuntimeError("reduced-cycle architecture manifest requires a clean tracked tree")
    if ARTIFACT_DIRECTORY.exists():
        raise RuntimeError("reduced-cycle architecture manifest is already frozen")
    contract = _contract()
    cost = contract["runtime_contract"]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "new_truth_trajectory_executed": False,
        "direct_truth_cycle_computationally_feasible": False,
        "online_truth_solver_calls_required": 0,
        "primary_continuous_state_dimension": cost["primary_continuous_state_dimension"],
        "minimum_required_end_to_end_speedup": cost["minimum_required_end_to_end_speedup"],
        "conservative_finite_memory_architecture_frozen": True,
        "evidence_only_identifiability_authorized": True,
        "offline_closure_database_manifest_authorized": False,
        "online_reduced_solver_implementation_authorized": False,
        "physical_microburst_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "physical_failure_detected": False,
        "parent_classification_preserved": inputs["latest"]["classification"],
        "authorized_next": contract["authorized_next"],
    }
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write(ARTIFACT_DIRECTORY / "architecture_contract.json", contract)
    _write(ARTIFACT_DIRECTORY / "input_lock.json", {
        "analyzed_commit": ANALYZED_COMMIT,
        "analyzed_parent": ANALYZED_PARENT,
        "analyzed_tree": ANALYZED_TREE,
        "input_hashes": _input_hashes(),
    })
    _write(ARTIFACT_DIRECTORY / "summary.json", summary)
    _write(ARTIFACT_DIRECTORY / "provenance.json", {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "scientific_status": "PROSPECTIVE",
        "definition_commit": _git("rev-parse", "HEAD"),
        "definition_tree": _git("rev-parse", "HEAD^{tree}"),
        "tracked_worktree_clean_at_start": True,
        "runner": THIS_RUNNER,
        "test": THIS_TEST,
        "authorized_next_runner": NEXT_RUNNER,
        "authorized_next_test": NEXT_TEST,
        "report": REPORT_RELATIVE,
        "source_hashes": {
            relative: _sha(ROOT / relative)
            for relative in (THIS_RUNNER, THIS_TEST)
        },
        "python": sys.version,
        "platform": platform.platform(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OMP_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
    })
    REPORT_PATH.write_text(
        "\n".join((
            "# Reduced-cycle architecture manifest WP10c9d6c7c3b5c4f25",
            "",
            "## Classification",
            "",
            f"`{CLASSIFICATION}`",
            "",
            "The certified fixed-Q solver is retained as an offline truth engine, but it is rejected as an online cycle-time integrator. At the certified `1e-7 s` step and measured `1289.606 s` warm-root cost, a fiducial `6.7 day` cycle would require about `5.79e12` roots and `2.37e8` wall-years. Meeting a three-day wall budget therefore requires an end-to-end change of architecture, not an incremental residual optimization.",
            "",
            "The online candidate is a conservative coarse radial finite-volume model with cellwise mapped mass/angular momentum/Killing-energy storage, thermal and stress storage candidates, the two cross-grid-stable amplitudes, a prospectively selected `r=0/2/4/6` stable finite-memory kernel, and a cold/hot/transition branch label. Interior fluxes must telescope exactly. The face-36 exterior partition remains the inner boundary; the raw horizon-face flux remains rejected.",
            "",
            "No truth solve or HMM microburst is permitted online. The existing one-zone `6.7 day` result is only a runtime and event-handling target; it is not a prediction of the certified no-tide/no-wind short-time truth model.",
            "",
            "The next package is evidence-only. It may select an architecture from committed results, but it may not fit coefficients, run a new root, implement the online solver, or authorize predictive cycle evolution.",
            "",
        )),
        encoding="utf-8",
    )
    names = (
        "architecture_contract.json",
        "input_lock.json",
        "provenance.json",
        "summary.json",
    )
    (ARTIFACT_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(ARTIFACT_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> int:
    print(json.dumps(_freeze(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
