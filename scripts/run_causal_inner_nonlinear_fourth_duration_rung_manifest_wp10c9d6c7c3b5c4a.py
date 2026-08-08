#!/usr/bin/env python3
"""Freeze the cost-controlled 10 ms screen for the 20 ms duration rung."""

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

import run_causal_inner_nonlinear_5ms_cumulative_extraction_recovery_audit_wp10c9d6c7c3b5c3h2j1 as h2j1  # noqa: E402
import run_causal_inner_nonlinear_5ms_extraction_surface_certificate_wp10c9d6c7c3b5c3h2i1 as h2i1  # noqa: E402
import run_causal_inner_nonlinear_third_duration_rung_completion_manifest_wp10c9d6c7c3b5c3c as c3c  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4a"
ANALYZED_BASE_COMMIT = "f0c661691e55c70d27bf7825c589671357221872"
ANALYZED_BASE_PARENT = "ef491daea69fa0f92075b5de768e7b84b9de839d"
ANALYZED_BASE_TREE = "dade52f5d6295964a9e977d5f56239ff5c6142f1"

RUNG_START_SECONDS = 5.0e-3
SCREEN_HORIZON_SECONDS = 1.0e-2
FULL_RUNG_HORIZON_SECONDS = 2.0e-2
PILOT_HORIZON_SECONDS = 5.4e-3
MASTER_TARGET_MICROSECONDS = np.asarray(
    (5000, 5400, 6000, 7000, 8000, 9000, 9600, 9800, 10000), dtype=int
)
MAIN_TARGET_INDICES = np.asarray((0, 2, 3, 4, 5, 8), dtype=int)
REPLAY_TARGET_INDICES = np.asarray((5, 6, 7, 8), dtype=int)
STRICT_TARGET_INDICES = np.asarray((6, 7, 8), dtype=int)
PILOT_TARGET_INDICES = np.asarray((0, 1), dtype=int)
MASTER_TARGETS_SECONDS = MASTER_TARGET_MICROSECONDS.astype(float) * 1.0e-6
MAIN_TARGETS_SECONDS = MASTER_TARGETS_SECONDS[MAIN_TARGET_INDICES]
REPLAY_TARGETS_SECONDS = MASTER_TARGETS_SECONDS[REPLAY_TARGET_INDICES]
STRICT_TARGETS_SECONDS = MASTER_TARGETS_SECONDS[STRICT_TARGET_INDICES]
PILOT_TARGETS_SECONDS = MASTER_TARGETS_SECONDS[PILOT_TARGET_INDICES]

SELECTED_EXTRACTION_RADIUS_RG = 1.9531594414758637
SELECTED_EXTRACTION_LAYOUT_FACE_INDICES = (2, 4, 8)

ARTIFACT = "causal_inner_nonlinear_fourth_duration_rung_manifest_wp10c9d6c7c3b5c4a"
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_fourth_duration_rung_manifest_"
    "wp10c9d6c7c3b5c4a.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_fourth_duration_rung_manifest_"
    "wp10c9d6c7c3b5c4a.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_FOURTH_DURATION_"
    "RUNG_MANIFEST_WP10C9D6C7C3B5C4A_2026-08-08.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "fourth_duration_rung_manifest.json"
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
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _validate_parent() -> dict:
    parent = _read_json(h2j1.SUMMARY_PATH)
    analysis = parent["analysis"]
    if (
        not parent["passed"]
        or not parent["fourth_duration_rung_manifest_authorized"]
        or parent["authorized_next"] != f"{WORK_PACKAGE}_fourth_duration_rung_manifest"
        or analysis["selected_coarse_face_index"] != 2
        or tuple(analysis["selected_layout_face_indices"])
        != SELECTED_EXTRACTION_LAYOUT_FACE_INDICES
        or analysis["selected_radius_rg"] != SELECTED_EXTRACTION_RADIUS_RG
        or parent["pointwise_horizon_flux_convergence_certified"]
        or parent["fixed_q_micro_solver_authorized"]
        or parent["reduced_slow_evolution_authorized"]
    ):
        raise RuntimeError("c4a authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c4a analyzed identity changed")
    return parent


def _controller_contracts() -> tuple[dict, dict]:
    main, strict = c3c._controller_contracts()
    main = json.loads(json.dumps(main))
    strict = json.loads(json.dumps(strict))
    main.update(
        {
            "initial_timestep_seconds": 4.0e-4,
            "minimum_timestep_seconds": 2.5e-5,
            "maximum_timestep_seconds": 4.0e-4,
            "maximum_BDF2_step_ratio": 2.0,
            "exact_output_landing": True,
        }
    )
    strict.update(
        {
            "initial_timestep_seconds": 1.0e-4,
            "minimum_timestep_seconds": 2.5e-5,
            "maximum_timestep_seconds": 1.0e-4,
            "maximum_BDF2_step_ratio": 2.0,
            "exact_output_landing": True,
        }
    )
    return main, strict


def _manifest() -> dict:
    main, strict = _controller_contracts()
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            "fourth_duration_rung_manifest_frozen_cost_controlled_"
            "ten_ms_screen_pilot_authorized"
        ),
        "passed": True,
        "definitions_only": True,
        "propagation_executed": False,
        "operator_changed": False,
        "production_defaults_changed": False,
        "rung": {
            "start_seconds": RUNG_START_SECONDS,
            "pilot_horizon_seconds": PILOT_HORIZON_SECONDS,
            "screen_horizon_seconds": SCREEN_HORIZON_SECONDS,
            "full_rung_horizon_seconds": FULL_RUNG_HORIZON_SECONDS,
            "screen_fraction_of_stress_relaxation_time": SCREEN_HORIZON_SECONDS / 0.147,
            "full_fraction_of_stress_relaxation_time": FULL_RUNG_HORIZON_SECONDS / 0.147,
            "layout": "N128-exterior__N128-equivalent-inner",
            "profile": "generic_five_field",
            "trajectories": ("base", "perturbed"),
            "initial_restart_source": (
                "committed_c3d_complete_base_and_perturbed_BDF2_restarts_at_5ms"
            ),
            "no_new_BDF1_startup": True,
        },
        "extraction_partition_contract": {
            "physical_radius_rg": SELECTED_EXTRACTION_RADIUS_RG,
            "coarse_middle_fine_face_indices": SELECTED_EXTRACTION_LAYOUT_FACE_INDICES,
            "observable_names": h2i1.OBSERVABLE_NAMES,
            "extraction_flux_is_not_pointwise_horizon_flux": True,
            "excision_to_extraction_buffer_remains_inside_microdomain": True,
            "buffer_storage_and_sources_remain_explicit": True,
            "raw_inner_face_rejection_preserved": True,
        },
        "canonical_targets": {
            "construction": "one_integer_microsecond_master_table",
            "master_microseconds": MASTER_TARGET_MICROSECONDS,
            "main_indices": MAIN_TARGET_INDICES,
            "replay_indices": REPLAY_TARGET_INDICES,
            "strict_indices": STRICT_TARGET_INDICES,
            "pilot_indices": PILOT_TARGET_INDICES,
            "main_seconds": MAIN_TARGETS_SECONDS,
            "replay_seconds": REPLAY_TARGETS_SECONDS,
            "strict_seconds": STRICT_TARGETS_SECONDS,
            "pilot_seconds": PILOT_TARGETS_SECONDS,
            "independent_target_construction_forbidden": True,
        },
        "main_controller": main,
        "strict_controller": strict,
        "pilot": {
            "base_first_then_perturbed": True,
            "one_accepted_comparison_per_trajectory": True,
            "full_step_two_half_step_estimator_retained": True,
            "method_and_extraction_partition_gates_retained": True,
            "projection_safety_factor": 1.5,
            "projected_wall_hours_at_or_below_24": "continue_automatically",
            "projected_wall_hours_24_to_48": "continue_after_optimization_review",
            "projected_wall_hours_above_48": "stop_and_optimize_before_full_screen",
            "runtime_projection_is_not_a_physical_gate": True,
        },
        "screen_execution": {
            "main_expected_comparisons_per_trajectory": 15,
            "replay_expected_comparisons_per_trajectory": 3,
            "strict_expected_comparisons_per_trajectory": 4,
            "serialized_replay_start_seconds": float(REPLAY_TARGETS_SECONDS[0]),
            "strict_shadow_start_seconds": float(STRICT_TARGETS_SECONDS[0]),
            "same_target_replay_state_export_history_restart_bitwise": True,
            "all_common_strict_outputs_binding": True,
            "durable_checkpoint_after_every_canonical_target": True,
            "stop_on_first_scientific_failure": True,
        },
        "binding_gates": {
            "maximum_scaled_residual": 1.0e-10,
            "maximum_discrete_ledger_defect": 1.0e-12,
            "main_local_error_maximum": 2.5e-4,
            "main_local_error_sum_maximum": 5.0e-3,
            "strict_local_error_maximum": 3.125e-5,
            "minimum_reconstruction_factor": 1.0,
            "minimum_scattering_optical_depth": 1.0,
            "maximum_h_over_r": 0.12,
            "maximum_incoming_excision_characteristics": 0,
            "maximum_exterior_prefix_identity_defect": 1.0e-12,
            "maximum_shared_conservative_face_defect": 1.0e-12,
            "strict_response_maximum_scaled_state_difference": 5.0e-3,
            "strict_response_maximum_scaled_extraction_partition_difference": 5.0e-3,
            "strict_response_minimum_history_cosine": 0.90,
        },
        "positive_branch": {
            "classification": (
                "ten_ms_screen_certified_twenty_ms_completion_manifest_authorized"
            ),
            "authorized_next": (
                "WP10c9d6c7c3b5c4c_twenty_ms_completion_manifest"
            ),
            "twenty_ms_propagation_requires_fresh_manifest": True,
        },
        "negative_branch": {
            "classification": "ten_ms_screen_failed_later_duration_blocked",
            "authorized_next": "failure_localization_only",
            "runtime_projection_alone_is_not_scientific_failure": True,
        },
        "hard_stops": (
            "do_not_use_raw_inner_face_flux_as_the_slow_export",
            "do_not_relabel_extraction_flux_as_pointwise_horizon_flux",
            "do_not_change_operator_profile_or_production_defaults",
            "do_not_run_twenty_ms_before_a_fresh_completion_manifest",
            "do_not_start_fixed_Q_or_reduced_slow_evolution",
            "do_not_add_tide_wind_hot_state_S_curve_or_QPE_cycle_physics",
            "do_not_use_N1024_as_a_rescue",
        ),
        "authorized_next": "WP10c9d6c7c3b5c4b_ten_ms_cost_pilot",
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


def main() -> int:
    parent = _validate_parent()
    manifest = _manifest()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": manifest["classification"],
        "passed": True,
        "definitions_only": True,
        "parent_classification_preserved": parent["classification"],
        "ten_ms_cost_pilot_authorized": True,
        "ten_ms_screen_propagation_authorized": False,
        "twenty_ms_completion_manifest_authorized": False,
        "twenty_ms_propagation_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": manifest["authorized_next"],
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "rung_start_seconds": RUNG_START_SECONDS,
            "pilot_horizon_seconds": PILOT_HORIZON_SECONDS,
            "screen_horizon_seconds": SCREEN_HORIZON_SECONDS,
            "full_rung_horizon_seconds": FULL_RUNG_HORIZON_SECONDS,
            "master_target_microseconds": MASTER_TARGET_MICROSECONDS,
            "selected_extraction_radius_rg": SELECTED_EXTRACTION_RADIUS_RG,
            "selected_extraction_layout_face_indices": (
                SELECTED_EXTRACTION_LAYOUT_FACE_INDICES
            ),
        },
    )
    _write_json(MANIFEST_PATH, manifest)
    _write_json(SUMMARY_PATH, summary)
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "CERTIFIED",
            "source_parent_commit": ANALYZED_BASE_COMMIT,
            "source_parent_tree_sha": ANALYZED_BASE_TREE,
            "working_head": _git_value("rev-parse", "HEAD"),
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "parent_summary_sha256": _sha256(h2j1.SUMMARY_PATH),
            "implementation_source_hashes": {
                path: _sha256(ROOT / path)
                for path in (THIS_RUNNER, THIS_TEST)
                if (ROOT / path).exists()
            },
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "command": f"PYTHONPATH=src:scripts python {THIS_RUNNER}",
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Fourth nonlinear duration-rung manifest WP10c9d6c7c3b5c4a",
                "",
                "## Classification",
                "",
                f"`{manifest['classification']}`",
                "",
                "The certified 5 ms exterior-domain extraction partition is carried forward unchanged. The selected extraction surface is `R=1.9531594414758637 r_g` (coarse/middle/fine faces `2/4/8`); it is not the pointwise horizon flux. The excision-to-extraction buffer remains inside the microdomain with explicit storage and sources.",
                "",
                "This definitions-only package authorizes a one-comparison-per-trajectory coarse cost pilot from 5.0 to 5.4 ms. A satisfactory pilot may authorize the complete 10 ms screen. The 20 ms completion still requires a fresh manifest after the 10 ms screen passes.",
                "",
                "The 24-hour projection is advisory. A 24-48 hour projection triggers optimization review, while a projection above 48 hours stops the full screen until optimized. Runtime classification is never a physical-failure classification.",
                "",
                "Fixed-Q experiments and reduced slow evolution remain blocked.",
                "",
            )
        ),
        encoding="utf-8",
    )
    names = (
        "config.json",
        "fourth_duration_rung_manifest.json",
        "provenance.json",
        "summary.json",
    )
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
