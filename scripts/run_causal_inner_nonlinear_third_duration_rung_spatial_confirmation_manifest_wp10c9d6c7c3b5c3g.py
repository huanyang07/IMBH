#!/usr/bin/env python3
"""Freeze the nonlinear third-duration-rung spatial confirmation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import sys

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_nonlinear_profile_breadth_controller_manifest_wp10c9d6c7c3b4a as breadth  # noqa: E402
import run_causal_inner_nonlinear_profile_breadth_spatial_wp10c9d6c7c3b4b3 as short_spatial  # noqa: E402
import run_causal_inner_nonlinear_profile_breadth_temporal_wp10c9d6c7c3b4b2 as short_temporal  # noqa: E402
import run_causal_inner_nonlinear_coarse_third_duration_rung_completion_wp10c9d6c7c3b5c3d as c3d  # noqa: E402
import run_causal_inner_nonlinear_third_duration_rung_breadth_manifest_wp10c9d6c7c3b5c3e as c3e  # noqa: E402
import run_causal_inner_nonlinear_coarse_heldout_third_duration_rung_screen_wp10c9d6c7c3b5c3f as c3f  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_canonical_json_sha256,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c3g"
ANALYZED_BASE_COMMIT = "05ce1e5b63eecdddef06402977d7bb417679e8d9"
ANALYZED_BASE_PARENT = "1c97bea290c0d0e649be0aedf94293074b76013d"
ANALYZED_BASE_TREE = "ce1c7f4ef38c596392c11f45c868fd4ef88a1a80"

GENERIC_PROFILE = c3e.GENERIC_PROFILE
LAYOUTS = tuple(c3e.SPATIAL_LAYOUTS)
COARSE_LAYOUT, MIDDLE_LAYOUT, FINE_LAYOUT = LAYOUTS
NEW_LAYOUTS = (MIDDLE_LAYOUT, FINE_LAYOUT)
LAYOUT_CELL_COUNTS = {
    COARSE_LAYOUT: 64,
    MIDDLE_LAYOUT: 112,
    FINE_LAYOUT: 208,
}
INNER_REFINEMENT_RATIOS = {
    COARSE_LAYOUT: 1,
    MIDDLE_LAYOUT: 2,
    FINE_LAYOUT: 4,
}
ACTIVE_COUPLING_FACE_INDICES = dict(c3e.ACTIVE_COUPLING_FACE_INDICES)

INITIAL_HISTORY_SECONDS = np.asarray((30.0e-6, 40.0e-6))
INITIAL_PREVIOUS_TIMESTEP_SECONDS = 1.0e-5
HORIZON_SECONDS = c3e.HORIZON_SECONDS
MAIN_TARGET_MICROSECONDS = np.asarray(c3e.MAIN_TARGET_MICROSECONDS, dtype=int)
REPLAY_TARGET_MICROSECONDS = np.asarray(c3e.REPLAY_TARGET_MICROSECONDS, dtype=int)
STRICT_TARGET_MICROSECONDS = np.asarray(c3e.STRICT_TARGET_MICROSECONDS, dtype=int)
MAIN_TARGETS_SECONDS = MAIN_TARGET_MICROSECONDS.astype(float) * 1.0e-6
REPLAY_TARGETS_SECONDS = REPLAY_TARGET_MICROSECONDS.astype(float) * 1.0e-6
STRICT_TARGETS_SECONDS = STRICT_TARGET_MICROSECONDS.astype(float) * 1.0e-6

OBSERVABLE_NAMES = (
    "inner_flux_mass",
    "inner_flux_angular_momentum",
    "inner_flux_killing_energy",
    "interface_flux_mass",
    "interface_flux_angular_momentum",
    "interface_flux_killing_energy",
    "net_drive_mass",
    "net_drive_angular_momentum",
    "net_drive_killing_energy",
    "cooling_angular_momentum",
    "cooling_killing_energy",
    "vertical_work_angular_momentum",
    "vertical_work_killing_energy",
)

SPATIAL_GATES = {
    "minimum_rms_order": 0.75,
    "minimum_maximum_order": 0.75,
    "minimum_significant_component_order": 0.75,
    "maximum_fine_normalized_difference": 0.05,
    "minimum_history_cosine": 0.90,
    "minimum_refinement_error_cosine": 0.90,
    "minimum_relative_activity": 1.0e-8,
}
TEMPORAL_UNCERTAINTY_GATES = {
    "conservative_envelope": "sum_of_base_and_perturbed_main_strict_differences",
    "maximum_strict_to_observable_medium_fine_spatial_error_ratio": 0.10,
    "observability_factor": 5.0,
    "unobservable_route": (
        "report_upper_bound_only_and_do_not_use_order_or_direction_as_evidence"
    ),
    "unobservable_upper_bound_must_be_below_fine_difference_gate": True,
}

EXECUTION_ORDER = (
    f"{MIDDLE_LAYOUT}__base_main_replay_strict",
    f"{MIDDLE_LAYOUT}__{GENERIC_PROFILE}_main_replay_strict",
    f"{FINE_LAYOUT}__base_main_replay_strict",
    f"{FINE_LAYOUT}__{GENERIC_PROFILE}_main_replay_strict",
    "common_parent_state_spatial_gate",
    "instantaneous_Tier_I_spatial_gate",
    "windowed_cumulative_Tier_I_spatial_gate",
)

ARTIFACT = (
    "causal_inner_nonlinear_third_duration_rung_spatial_confirmation_"
    "manifest_wp10c9d6c7c3b5c3g"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_third_duration_rung_spatial_"
    "confirmation_manifest_wp10c9d6c7c3b5c3g.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_third_duration_rung_spatial_"
    "confirmation_manifest_wp10c9d6c7c3b5c3g.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_NONLINEAR_THIRD_DURATION_RUNG_SPATIAL_"
    "CONFIRMATION_MANIFEST_WP10C9D6C7C3B5C3G_2026-08-05.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "spatial_confirmation_manifest.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_parent() -> tuple[dict, dict, dict]:
    parent = _read_json(c3f.SUMMARY_PATH)
    scope = _read_json(c3e.SUMMARY_PATH)
    short = _read_json(short_spatial.SUMMARY_PATH)
    if (
        not parent["passed"]
        or parent["classification"]
        != "coarse_heldout_third_rung_duration_breadth_certified_"
        "generic_spatial_confirmation_manifest_authorized"
        or not parent["third_duration_rung_spatial_confirmation_manifest_authorized"]
        or parent["third_duration_rung_spatial_confirmation_propagation_authorized"]
        or parent["authorized_next"]
        != "WP10c9d6c7c3b5c3g_third_duration_rung_spatial_confirmation_manifest"
        or scope["classification"]
        != "third_duration_rung_breadth_manifest_frozen_coarse_heldout_"
        "duration_screen_authorized"
        or short["classification"]
        != "heldout_profile_spatial_confirmation_failed_duration_extension_blocked"
    ):
        raise RuntimeError("c3g authorization or inherited status changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c3g analyzed identity changed")
    return parent, scope, short


def _controller_contracts() -> tuple[dict, dict]:
    main = json.loads(json.dumps(c3e._main_controller()))
    main.update(
        {
            "initial_timestep_seconds": 5.0e-6,
            "minimum_timestep_seconds": 1.25e-6,
            "maximum_timestep_seconds": 4.0e-4,
            "maximum_BDF2_step_ratio": 2.0,
            "initial_previous_timestep_seconds": INITIAL_PREVIOUS_TIMESTEP_SECONDS,
            "exact_output_landing": True,
        }
    )
    strict = json.loads(json.dumps(c3e._manifest()["common_contract"]["strict_controller"]))
    strict.update(
        {
            "maximum_timestep_seconds": 1.0e-4,
            "exact_output_landing": True,
        }
    )
    return main, strict


def _manifest() -> dict:
    main_controller, strict_controller = _controller_contracts()
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            "third_duration_rung_spatial_confirmation_manifest_frozen_"
            "middle_fine_generic_propagation_authorized"
        ),
        "passed": True,
        "propagation_executed": False,
        "operator_changed": False,
        "production_defaults_changed": False,
        "historical_classifications_preserved": {
            "c2d": "second_rung_perturbed_completion_failed_later_duration_blocked",
            "b4b3": "heldout_profile_spatial_confirmation_failed_duration_extension_blocked",
        },
        "experiment": {
            "profile": GENERIC_PROFILE,
            "binding_multiplier": breadth.BINDING_PROPAGATION_MULTIPLIER,
            "horizon_seconds": HORIZON_SECONDS,
            "fraction_of_N128_cell_crossing": HORIZON_SECONDS / 5.54e-3,
            "layouts": LAYOUTS,
            "new_layouts": NEW_LAYOUTS,
            "layout_cell_counts": LAYOUT_CELL_COUNTS,
            "inner_refinement_ratios": INNER_REFINEMENT_RATIOS,
            "active_coupling_face_indices": ACTIVE_COUPLING_FACE_INDICES,
            "canonical_target_source": "c3c_single_integer_100_microsecond_source",
            "main_target_microseconds": MAIN_TARGET_MICROSECONDS,
            "replay_target_microseconds": REPLAY_TARGET_MICROSECONDS,
            "strict_target_microseconds": STRICT_TARGET_MICROSECONDS,
            "independent_target_construction_forbidden": True,
            "coarse_c3d_base_and_perturbed_main_replay_strict_reused_by_hash": True,
            "middle_fine_initial_history_source": (
                "committed_b4b3_layout_native_base_and_generic_states_at_"
                "30_and_40_microseconds"
            ),
            "middle_fine_initial_history_seconds": INITIAL_HISTORY_SECONDS,
            "middle_fine_initial_previous_timestep_seconds": (
                INITIAL_PREVIOUS_TIMESTEP_SECONDS
            ),
            "mapped_and_height_histories_reconstructed_from_primitive_states": True,
            "no_new_BDF1_startup": True,
            "one_tangent_and_one_process_per_layout": True,
            "durable_cache_after_each_complete_trajectory": True,
            "main_controller": main_controller,
            "strict_controller": strict_controller,
            "serialized_replay_start_seconds": float(REPLAY_TARGETS_SECONDS[0]),
            "strict_shadow_start_seconds": float(STRICT_TARGETS_SECONDS[0]),
            "same_target_replay_states_exports_histories_and_restart_bitwise": True,
            "execution_order": EXECUTION_ORDER,
            "stop_on_first_failure": True,
        },
        "state_contract": {
            "response": "perturbed_minus_independently_evolved_base",
            "restriction": (
                "conservative_embedded_cell_averages_to_common_64_cell_parent"
            ),
            "coarse_middle_fine_inner_refinement_ratios": [1, 2, 4],
            "field_scales_source": "committed_b4b2_field_scales",
            "spatial_gates": SPATIAL_GATES,
        },
        "Tier_I_contract": {
            "observable_names": OBSERVABLE_NAMES,
            "fixed_scales_source": "committed_b4b2_fixed_physical_observable_scales",
            "interface_flux_must_use_layout_active_face": ACTIVE_COUPLING_FACE_INDICES,
            "instantaneous_response": True,
            "windowed_cumulative_response": {
                "integration": "trapezoidal",
                "window_seconds": [
                    float(MAIN_TARGETS_SECONDS[0]),
                    float(MAIN_TARGETS_SECONDS[-1]),
                ],
                "not_claimed_as_zero_to_horizon_cumulative_export": True,
            },
            "spatial_gates": SPATIAL_GATES,
        },
        "temporal_uncertainty_contract": TEMPORAL_UNCERTAINTY_GATES,
        "method_gates": {
            "maximum_scaled_residual": 1.0e-10,
            "maximum_discrete_ledger_defect": 1.0e-12,
            "maximum_local_error_estimate": 2.5e-4,
            "maximum_sum_local_error_estimates": 5.0e-3,
            "minimum_reconstruction_factor": 1.0,
            "minimum_scattering_optical_depth": 1.0,
            "maximum_h_over_r": 0.12,
            "maximum_incoming_excision_characteristics": 0,
            "strict_response_maximum_scaled_state_difference": 5.0e-3,
            "strict_response_maximum_scaled_Tier_I_difference": 5.0e-3,
            "strict_response_minimum_history_cosine": 0.90,
        },
        "cost_and_fail_fast": {
            "coarse_reference_wall_seconds": 22224.52,
            "cell_count_scaled_middle_base_plus_perturbed_lower_bound_hours": 21.6,
            "cell_count_scaled_fine_base_plus_perturbed_lower_bound_hours": 40.1,
            "estimate_is_not_a_runtime_gate": True,
            "middle_completes_before_fine_begins": True,
        },
        "positive_branch": {
            "classification": (
                "third_duration_rung_breadth_and_spatial_convergence_"
                "certified_fourth_duration_rung_manifest_authorized"
            ),
            "authorized_next": (
                "WP10c9d6c7c3b5c4a_fourth_duration_rung_manifest"
            ),
            "only_definitions_only_fourth_rung_manifest_authorized": True,
            "fixed_q_and_reduced_evolution_still_blocked": True,
        },
        "negative_branch": {
            "classification": (
                "third_duration_rung_spatial_confirmation_failed_"
                "fourth_duration_rung_blocked"
            ),
            "authorized_next": "failure_localization_only",
            "separate_state_export_and_temporal_uncertainty_failures": True,
            "no_operator_redesign_without_stable_noncontracting_mechanism": True,
        },
        "hard_stops": [
            "do not amend the c2d or b4b3 historical failures",
            "do not rerun or tune the coarse c3d reference",
            "do not generate target arrays independently",
            "do not use the parent face 48 on middle or fine exports",
            "do not relax spatial, temporal-uncertainty, method or replay gates",
            "do not begin the 2e-2 s fourth rung before spatial confirmation passes",
            "do not begin fixed-Q experiments or reduced slow evolution",
            "do not add tide, wind, hot-state, S-curve or QPE-cycle physics",
            "do not use N1024 as a rescue",
        ],
        "authorized_next": (
            "WP10c9d6c7c3b5c3h_third_duration_rung_spatial_confirmation"
        ),
    }


def _update_catalog(summary: dict) -> None:
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha256(path),
                    "scientific_status": "CERTIFIED",
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case", "path", "bytes", "sha256", "scientific_status"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read_json(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": summary["passed"],
    }
    catalog.update(
        {
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": ANALYZED_BASE_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def _report(manifest: dict) -> str:
    return "\n".join(
        [
            "# Third nonlinear duration-rung spatial-confirmation manifest WP10c9d6c7c3b5c3g",
            "",
            "## Classification",
            "",
            f"`{manifest['classification']}`",
            "",
            "This definitions-only package freezes the middle/fine generic-five-field spatial confirmation at `5e-3 s`. It propagates no state and changes no operator or production default.",
            "",
            "The coarse c3d base/perturbed result is reused by hash. New middle and fine base/perturbed trajectories must use their layout-native committed short-horizon histories, the one canonical target source, correct active coupling faces `96/192`, serialized replay, and strict final-interval shadows.",
            "",
            "State responses are conservatively restricted to the common 64-cell parent. Instantaneous and windowed-cumulative 13-component Tier-I responses use the inherited `0.75/0.05/0.90` spatial gates. Strict temporal uncertainty must be no more than ten percent of an observable middle/fine spatial difference.",
            "",
            f"Authorized next: `{manifest['authorized_next']}`.",
            "",
            "The `2e-2 s` fourth rung, fixed-Q experiments, reduced slow evolution, tide, wind, production promotion, and N1024 remain blocked.",
            "",
        ]
    )


def main() -> int:
    parent, scope, short = _validate_parent()
    manifest = _manifest()
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "profile": GENERIC_PROFILE,
        "binding_multiplier": breadth.BINDING_PROPAGATION_MULTIPLIER,
        "horizon_seconds": HORIZON_SECONDS,
        "layouts": LAYOUTS,
        "new_layouts": NEW_LAYOUTS,
        "layout_cell_counts": LAYOUT_CELL_COUNTS,
        "inner_refinement_ratios": INNER_REFINEMENT_RATIOS,
        "active_coupling_face_indices": ACTIVE_COUPLING_FACE_INDICES,
        "initial_history_seconds": INITIAL_HISTORY_SECONDS,
        "initial_previous_timestep_seconds": INITIAL_PREVIOUS_TIMESTEP_SECONDS,
        "main_targets_seconds": MAIN_TARGETS_SECONDS,
        "replay_targets_seconds": REPLAY_TARGETS_SECONDS,
        "strict_targets_seconds": STRICT_TARGETS_SECONDS,
        "observable_names": OBSERVABLE_NAMES,
        "spatial_gates": SPATIAL_GATES,
        "temporal_uncertainty_gates": TEMPORAL_UNCERTAINTY_GATES,
        "execution_order": EXECUTION_ORDER,
        "propagation_executed": False,
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": manifest["classification"],
        "passed": True,
        "propagation_executed": False,
        "operator_changed": False,
        "production_defaults_changed": False,
        "parent_classification_preserved": parent["classification"],
        "scope_classification_preserved": scope["classification"],
        "historical_b4b3_classification_preserved": short["classification"],
        "historical_c2d_classification_preserved": (
            "second_rung_perturbed_completion_failed_later_duration_blocked"
        ),
        "coarse_heldout_duration_breadth_certified": True,
        "middle_fine_generic_spatial_confirmation_authorized": True,
        "middle_fine_generic_spatial_confirmation_executed": False,
        "third_duration_rung_spatial_convergence_certified": False,
        "fourth_duration_rung_manifest_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": manifest["authorized_next"],
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(CONFIG_PATH, config)
    _write_json(MANIFEST_PATH, manifest)
    _write_json(SUMMARY_PATH, summary)
    input_paths = {
        "parent_c3f_summary": c3f.SUMMARY_PATH,
        "parent_c3f_arrays": c3f.DECISIVE_ARRAYS,
        "scope_c3e_manifest": c3e.MANIFEST_PATH,
        "coarse_c3d_summary": c3d.SUMMARY_PATH,
        "coarse_c3d_arrays": c3d.DECISIVE_ARRAYS,
        "short_spatial_b4b3_summary": short_spatial.SUMMARY_PATH,
        "short_spatial_b4b3_arrays": short_spatial.DECISIVE_ARRAYS,
        "short_temporal_b4b2_summary": short_temporal.SUMMARY_PATH,
        "short_temporal_b4b2_arrays": short_temporal.DECISIVE_ARRAYS,
    }
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "CERTIFIED",
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "analyzed_base_parent": ANALYZED_BASE_PARENT,
            "analyzed_base_tree": ANALYZED_BASE_TREE,
            "implementation_commit_before_manifest": _git_value("rev-parse", "HEAD"),
            "implementation_tree_before_manifest": _git_value("rev-parse", "HEAD^{tree}"),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "runner": THIS_RUNNER,
            "runner_sha256": _sha256(ROOT / THIS_RUNNER),
            "test": THIS_TEST,
            "input_hashes": {name: _sha256(path) for name, path in input_paths.items()},
            "config_sha256": causal_canonical_json_sha256(_plain(config)),
            "manifest_sha256": causal_canonical_json_sha256(_plain(manifest)),
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_report(manifest), encoding="utf-8")
    checksum_paths = (CONFIG_PATH, MANIFEST_PATH, PROVENANCE_PATH, SUMMARY_PATH)
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in checksum_paths),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
