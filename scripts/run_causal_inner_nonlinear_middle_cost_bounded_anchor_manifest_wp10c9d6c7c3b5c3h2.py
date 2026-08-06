#!/usr/bin/env python3
"""Freeze the cost-bounded middle-layout nonlinear anchor campaign."""

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

import run_causal_inner_nonlinear_discrete_bdf_tangent_calibration_wp10c9d6c7c3b5c3h1 as h1  # noqa: E402
import run_causal_inner_nonlinear_third_duration_rung_cost_bounded_spatial_confirmation_manifest_wp10c9d6c7c3b5c3g1 as g1  # noqa: E402
import run_causal_inner_nonlinear_third_duration_rung_spatial_confirmation_manifest_wp10c9d6c7c3b5c3g as g  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c3h2"
ANALYZED_BASE_COMMIT = "71972ba60ec78f6e16eb672bc3815fb8e40980a6"
ANALYZED_BASE_PARENT = "a868dc3c52c860399bd2af36d0f17f1d3fa5cac9"
ANALYZED_BASE_TREE = "fcea3ff4dfa7f5cd96455d20194dfb8fdf1b82ef"

MIDDLE_LAYOUT = g1.MIDDLE_LAYOUT
PROFILES = tuple(g1.PROFILE_NAMES)
GENERIC_PROFILE = g1.c3g.GENERIC_PROFILE
HORIZON_SECONDS = float(g.HORIZON_SECONDS)
ACTIVE_COUPLING_FACE = int(g.ACTIVE_COUPLING_FACE_INDICES[MIDDLE_LAYOUT])
INITIAL_HISTORY_SECONDS = tuple(float(value) for value in g.INITIAL_HISTORY_SECONDS)
MAIN_TARGET_MICROSECONDS = tuple(int(value) for value in g.MAIN_TARGET_MICROSECONDS)
REPLAY_TARGET_MICROSECONDS = tuple(int(value) for value in g.REPLAY_TARGET_MICROSECONDS)
STRICT_TARGET_MICROSECONDS = tuple(int(value) for value in g.STRICT_TARGET_MICROSECONDS)
SPATIAL_GATES = dict(g.SPATIAL_GATES)
TANGENT_GATES = dict(g1.TANGENT_GATES)
METHOD_GATES = dict(g._manifest()["method_gates"])

CALIBRATION_SUMMARY = json.loads(h1.SUMMARY_PATH.read_text(encoding="utf-8"))
MIDDLE_CALIBRATION = CALIBRATION_SUMMARY["calibration"]["short"][MIDDLE_LAYOUT]

COST_GATES = {
    "maximum_projected_total_new_nonlinear_wall_hours": 24.0,
    "maximum_single_unattended_stage_wall_hours": 12.0,
    "maximum_pilot_projection_safety_factor": 2.0,
    "minimum_measured_accepted_steps_for_projection": 2,
    "projection_includes_context_base_anchor_replay_and_sampled_strict_cost": True,
    "projection_failure_action": (
        "stop_before_full_middle_and_optimize_context_history_or_factorization_reuse"
    ),
}

ARTIFACT = (
    "causal_inner_nonlinear_middle_cost_bounded_anchor_manifest_"
    "wp10c9d6c7c3b5c3h2"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_middle_cost_bounded_anchor_"
    "manifest_wp10c9d6c7c3b5c3h2.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_middle_cost_bounded_anchor_"
    "manifest_wp10c9d6c7c3b5c3h2.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_MIDDLE_COST_BOUNDED_"
    "ANCHOR_MANIFEST_WP10C9D6C7C3B5C3H2_2026-08-05.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "middle_anchor_manifest.json"
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


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _validate_parent() -> dict:
    parent = _read_json(h1.SUMMARY_PATH)
    if (
        not parent["passed"]
        or parent["classification"]
        != "complete_discrete_BDF_tangent_calibrated_middle_cost_bounded_"
        "anchor_manifest_authorized"
        or not parent["middle_cost_bounded_anchor_manifest_authorized"]
        or parent["middle_cost_bounded_propagation_authorized"]
    ):
        raise RuntimeError("h2 authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("h2 analyzed identity changed")
    return parent


def _manifest() -> dict:
    main_controller, strict_controller = g._controller_contracts()
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            "middle_cost_bounded_anchor_manifest_frozen_staged_middle_"
            "execution_authorized"
        ),
        "passed": True,
        "definitions_only": True,
        "propagation_executed": False,
        "operator_changed": False,
        "production_defaults_changed": False,
        "scientific_basis": {
            "complete_discrete_BDF_tangent_certified": True,
            "long_tail_profiles": PROFILES,
            "long_tail_maximum_scaled_state_discrepancy": CALIBRATION_SUMMARY[
                "calibration"
            ]["long"]["state"]["maximum_scaled_discrepancy"],
            "long_tail_maximum_scaled_Tier_I_discrepancy": CALIBRATION_SUMMARY[
                "calibration"
            ]["long"]["instantaneous_Tier_I"]["maximum_scaled_discrepancy"],
            "middle_short_matrix_assembly_seconds": MIDDLE_CALIBRATION[
                "matrix_assembly_wall_seconds"
            ],
            "middle_short_block_solve_seconds": MIDDLE_CALIBRATION[
                "block_step_wall_seconds"
            ],
            "binding_export_reference": (
                "WP10c9d6c7c3b4d_corrected_active_coupling_face"
            ),
        },
        "middle_experiment": {
            "layout": MIDDLE_LAYOUT,
            "cell_count": g.LAYOUT_CELL_COUNTS[MIDDLE_LAYOUT],
            "active_coupling_face_index": ACTIVE_COUPLING_FACE,
            "horizon_seconds": HORIZON_SECONDS,
            "initial_history_seconds": INITIAL_HISTORY_SECONDS,
            "initial_previous_timestep_seconds": (
                g.INITIAL_PREVIOUS_TIMESTEP_SECONDS
            ),
            "base_and_generic_history_source": (
                "committed_b4b3_layout_native_30_and_40_microsecond_states"
            ),
            "mapped_and_height_histories_reconstructed_exactly": True,
            "no_new_BDF1_startup": True,
            "main_controller": main_controller,
            "strict_controller": strict_controller,
            "main_target_microseconds": MAIN_TARGET_MICROSECONDS,
            "replay_target_microseconds": REPLAY_TARGET_MICROSECONDS,
            "strict_target_microseconds": STRICT_TARGET_MICROSECONDS,
            "one_context_and_one_base_schedule_reused": True,
            "all_profile_tangents_solved_as_one_block": True,
        },
        "fail_fast_stages": (
            {
                "stage": "h2a_cost_pilot",
                "stop_seconds": 2.0e-4,
                "work": (
                    "construct_context_and_histories",
                    "advance_middle_base",
                    "advance_middle_generic_anchor_on_same_accepted_schedule",
                    "propagate_all_five_discrete_tangents_as_one_block",
                    "sample_step_doubling_on_first_two_accepted_steps",
                ),
                "decision": (
                    "project_total_and_each_stage_cost_with_safety_factor_before_"
                    "continuation"
                ),
            },
            {
                "stage": "h2b_one_millisecond_screen",
                "stop_seconds": 1.0e-3,
                "authorized_only_if": "h2a_cost_and_scientific_gates_pass",
                "durable_checkpoint_required": True,
            },
            {
                "stage": "h2c_two_millisecond_screen",
                "stop_seconds": 2.0e-3,
                "authorized_only_if": "h2b_gates_pass",
                "same_target_bitwise_replay_required": True,
            },
            {
                "stage": "h2d_five_millisecond_completion",
                "stop_seconds": 5.0e-3,
                "authorized_only_if": "h2c_gates_pass_and_projection_remains_bounded",
                "sampled_strict_shadow_required": True,
            },
        ),
        "cost_contract": COST_GATES,
        "tangent_anchor_contract": {
            "profiles": PROFILES,
            "generic_full_nonlinear_anchor_required": True,
            "other_full_nonlinear_perturbed_trajectories_forbidden": True,
            "maximum_scaled_state_discrepancy": TANGENT_GATES[
                "maximum_scaled_state_response_discrepancy"
            ],
            "maximum_scaled_Tier_I_discrepancy": TANGENT_GATES[
                "maximum_scaled_Tier_I_response_discrepancy"
            ],
            "minimum_state_history_cosine": TANGENT_GATES[
                "minimum_state_response_history_cosine"
            ],
            "minimum_Tier_I_history_cosine": TANGENT_GATES[
                "minimum_Tier_I_response_history_cosine"
            ],
            "surrogate_uncertainty_added_to_spatial_and_temporal_budgets": True,
            "stop_on_generic_anchor_failure": True,
        },
        "temporal_contract": {
            "layout_owns_adaptive_schedule": True,
            "coarse_schedule_reuse_forbidden": True,
            "sampled_step_doubling_windows": (
                "first_two_accepted_steps",
                "around_one_millisecond",
                "around_two_milliseconds",
                "final_4p8_to_5p0_millisecond_window",
            ),
            "maximum_sampled_temporal_to_observable_spatial_error_ratio": 0.10,
            "same_target_replay_complete_payload_bitwise": True,
            "all_common_strict_outputs_are_compared": True,
        },
        "method_gates": METHOD_GATES,
        "middle_decision": {
            "pass": (
                "authorize_definitions_only_fine_cost_bounded_confirmation_manifest"
            ),
            "scientific_failure": (
                "stop_before_fine_and_localize_base_anchor_or_tangent_defect"
            ),
            "cost_projection_failure": (
                "stop_without_scientific_rejection_and_optimize_reuse"
            ),
        },
        "downstream_stops": {
            "fine_propagation_authorized": False,
            "third_duration_rung_spatial_convergence_certified": False,
            "fourth_duration_rung_manifest_authorized": False,
            "fixed_q_micro_solver_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
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
        "passed": True,
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


def _report(summary: dict) -> str:
    manifest = summary["manifest"]
    return "\n".join(
        (
            "# Middle cost-bounded anchor manifest WP10c9d6c7c3b5c3h2",
            "",
            "## Classification",
            "",
            f"`{summary['classification']}`",
            "",
            "The complete discrete BDF tangent is certified. This definitions-only package authorizes a staged middle-layout experiment with one nonlinear base, one full generic nonlinear anchor, and five profile responses propagated as one block tangent.",
            "",
            "The first stage stops at 0.2 ms and must project the complete new nonlinear cost below 24 hours, with no unattended stage above 12 hours. A failed cost projection is a scheduling stop, not a physical rejection.",
            "",
            f"The calibration measured `{manifest['scientific_basis']['middle_short_matrix_assembly_seconds']:.1f} s` matrix assembly and `{manifest['scientific_basis']['middle_short_block_solve_seconds']:.3f} s` for the block solve. The correct active face is `{ACTIVE_COUPLING_FACE}`.",
            "",
            "Fine propagation, the fourth duration rung, fixed-Q experiments, and reduced slow evolution remain blocked.",
            "",
        )
    )


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
        "manifest": manifest,
        "middle_staged_execution_authorized": True,
        "fine_cost_bounded_confirmation_manifest_authorized": False,
        "fine_cost_bounded_propagation_authorized": False,
        "third_duration_rung_spatial_convergence_certified": False,
        "fourth_duration_rung_manifest_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": "WP10c9d6c7c3b5c3h2a_middle_cost_pilot",
    }
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "layout": MIDDLE_LAYOUT,
        "profiles": PROFILES,
        "horizon_seconds": HORIZON_SECONDS,
        "cost_gates": COST_GATES,
        "tangent_gates": TANGENT_GATES,
        "spatial_gates": SPATIAL_GATES,
        "method_gates": METHOD_GATES,
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(CONFIG_PATH, config)
    _write_json(MANIFEST_PATH, manifest)
    _write_json(SUMMARY_PATH, summary)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "analyzed_base_parent": ANALYZED_BASE_PARENT,
        "analyzed_base_tree": ANALYZED_BASE_TREE,
        "working_head": _git_value("rev-parse", "HEAD"),
        "runner": THIS_RUNNER,
        "test": THIS_TEST,
        "report": REPORT_RELATIVE,
        "input_hashes": {
            "h1_summary": _sha256(h1.SUMMARY_PATH),
            "h1_arrays": _sha256(h1.DECISIVE_ARRAYS),
            "g1_manifest": _sha256(g1.MANIFEST_PATH),
            "g_manifest": _sha256(g.MANIFEST_PATH),
        },
        "implementation_source_hashes": {
            "runner": _sha256(ROOT / THIS_RUNNER),
            "test": _sha256(ROOT / THIS_TEST),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "command": (
            "PYTHONPATH=src:scripts python scripts/"
            "run_causal_inner_nonlinear_middle_cost_bounded_anchor_"
            "manifest_wp10c9d6c7c3b5c3h2.py"
        ),
    }
    _write_json(PROVENANCE_PATH, provenance)
    REPORT_PATH.write_text(_report(summary), encoding="utf-8")
    names = ("config.json", "middle_anchor_manifest.json", "provenance.json", "summary.json")
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(summary["classification"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
