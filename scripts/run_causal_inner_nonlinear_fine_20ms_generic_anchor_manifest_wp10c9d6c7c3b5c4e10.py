#!/usr/bin/env python3
"""Freeze the targeted fine-layout 5-to-20 ms generic-anchor campaign."""

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

import run_causal_inner_nonlinear_fine_5ms_completion_wp10c9d6c7c3b5c3h2e1 as fine5  # noqa: E402
import run_causal_inner_nonlinear_cost_bounded_fine_20ms_base_tangent_wp10c9d6c7c3b5c4e8 as c4e8  # noqa: E402
import run_causal_inner_nonlinear_three_grid_20ms_spatial_analysis_wp10c9d6c7c3b5c4e9 as c4e9  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4e10"
ANALYZED_BASE_COMMIT = c4e9.ANALYZED_BASE_COMMIT
ANALYZED_BASE_PARENT = c4e9.ANALYZED_BASE_PARENT
ANALYZED_BASE_TREE = c4e9.ANALYZED_BASE_TREE

FINE_LAYOUT = c4e8.FINE_LAYOUT
GENERIC_PROFILE = c4e8.PROFILES[c4e8.GENERIC_INDEX]
COUPLING_FACE = c4e8.COUPLING_FACE
EXTRACTION_FACE = c4e8.EXTRACTION_FACE
EXTRACTION_RADIUS_RG = c4e8.c4e3.c4e1.c4e.EXTRACTION_RADIUS_RG
START_MICROSECONDS = 5_000
STOP_MICROSECONDS = 20_000
TARGET_MICROSECONDS = tuple(c4e8.TARGET_MICROSECONDS)
TEMPORAL_AUDIT_TARGET_MICROSECONDS = (8_000, 14_000, 20_000)
MAXIMUM_TIMESTEP_SECONDS = c4e8.MAXIMUM_TIMESTEP_SECONDS
PROJECTED_ACCEPTED_STEPS = 39

ARTIFACT = (
    "causal_inner_nonlinear_fine_20ms_generic_anchor_manifest_"
    "wp10c9d6c7c3b5c4e10"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_fine_20ms_generic_anchor_manifest_"
    "wp10c9d6c7c3b5c4e10.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_fine_20ms_generic_anchor_manifest_"
    "wp10c9d6c7c3b5c4e10.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_FINE_20MS_GENERIC_"
    "ANCHOR_MANIFEST_WP10C9D6C7C3B5C4E10_2026-08-12.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "anchor_manifest.json"
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


def _validate_parent() -> tuple[dict, dict]:
    analysis = _read_json(c4e9.SUMMARY_PATH)
    fine = _read_json(c4e8.SUMMARY_PATH)
    if (
        not analysis["passed"]
        or not analysis["state_twenty_ms_spatial_contract_certified"]
        or not analysis["full_fine_generic_anchor_required"]
        or not analysis["fine_generic_anchor_manifest_authorized"]
        or analysis["full_fine_generic_anchor_authorized"]
        or analysis["fine_twenty_ms_spatial_certificate_issued"]
        or analysis["physical_failure_detected"]
    ):
        raise RuntimeError("c4e10 three-grid decision changed")
    if (
        not fine["passed"]
        or not fine["fine_twenty_ms_computation_completed"]
        or not fine["serialized_base_replay"]["last_step_replay_bitwise"]
        or fine["physical_failure_detected"]
    ):
        raise RuntimeError("c4e10 fine base/tangent evidence changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c4e10 analyzed identity changed")
    return analysis, fine


def _cost_projection() -> dict:
    fine5_summary = _read_json(fine5.SUMMARY_PATH)
    anchor = fine5_summary["anchor"]
    routine = float(anchor["median_routine_step_wall_seconds"])
    sampled = float(anchor["median_sampled_step_wall_seconds"])
    audit_count = len(TEMPORAL_AUDIT_TARGET_MICROSECONDS)
    routine_count = PROJECTED_ACCEPTED_STEPS - audit_count
    extraction_and_replay = 2400.0
    raw = routine_count * routine + audit_count * sampled + extraction_and_replay
    return {
        "basis": "measured_fine_5ms_generic_anchor_step_costs",
        "accepted_steps": PROJECTED_ACCEPTED_STEPS,
        "routine_steps": routine_count,
        "sampled_temporal_audits": audit_count,
        "routine_step_wall_seconds": routine,
        "sampled_step_wall_seconds": sampled,
        "extraction_reanalysis_replay_allowance_wall_seconds": extraction_and_replay,
        "projected_raw_wall_seconds": raw,
        "projected_raw_wall_hours": raw / 3600.0,
        "safety_factor": 1.25,
        "projected_safe_wall_hours": 1.25 * raw / 3600.0,
        "projection_is_scheduling_only": True,
    }


def _manifest() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            "fine_20ms_generic_anchor_manifest_frozen_targeted_anchor_"
            "propagation_authorized"
        ),
        "definitions_frozen_before_propagation": True,
        "propagation_executed": False,
        "operator_changed": False,
        "production_defaults_changed": False,
        "scientific_reason": {
            "state_spatial_contract_already_certified": True,
            "aggregate_extraction_orders_are_near_second_order": True,
            "fine_tangent_extraction_surrogate_fraction_range": (0.421, 0.616),
            "maximum_allowed_surrogate_fraction": 0.10,
            "decision": (
                "replace_only_the_fine_generic_tangent_response_with_one_"
                "continuous_nonlinear_anchor"
            ),
            "physical_failure_detected": False,
        },
        "scope": {
            "layout": FINE_LAYOUT,
            "profile": GENERIC_PROFILE,
            "start_microseconds": START_MICROSECONDS,
            "stop_microseconds": STOP_MICROSECONDS,
            "target_microseconds": TARGET_MICROSECONDS,
            "temporal_audit_target_microseconds": TEMPORAL_AUDIT_TARGET_MICROSECONDS,
            "coupling_face": COUPLING_FACE,
            "extraction_face": EXTRACTION_FACE,
            "extraction_radius_rg": EXTRACTION_RADIUS_RG,
            "slow_export": "certified_conservative_exterior_partition",
            "raw_inner_face_is_not_a_slow_export": True,
        },
        "reuse_contract": {
            "fine_base": "reuse_c4e8_canonical_states_histories_and_schedule_bitwise",
            "fine_tangent": "reuse_c4e8_generic_direction_as_Newton_predictor",
            "initial_generic_anchor": (
                "reuse_fine5_nonlinear_generic_state_and_complete_BDF2_history"
            ),
            "rerun_fine_base": False,
            "reassemble_five_profile_block_tangent": False,
            "run_other_profiles_nonlinearly": False,
        },
        "anchor_contract": {
            "continuous_nonlinear_generic_anchor_required": True,
            "exact_fine_base_accepted_timestep_schedule_required": True,
            "tangent_predicted_Newton_initial_guess_required": True,
            "maximum_timestep_seconds": MAXIMUM_TIMESTEP_SECONDS,
            "sampled_full_versus_two_half_audits": (
                TEMPORAL_AUDIT_TARGET_MICROSECONDS
            ),
            "maximum_local_error_estimate": 2.5e-4,
            "maximum_scaled_nonlinear_residual": 1.0e-10,
            "maximum_discrete_ledger_defect": 1.0e-12,
            "maximum_mapped_endpoint_path_closure_defect": 1.0e-9,
            "minimum_reconstruction_factor": 1.0,
            "maximum_incoming_excision_characteristics": 0,
            "same_target_last_step_replay_bitwise": True,
        },
        "output_contract": {
            "actual_state_response_at_every_accepted_time": True,
            "actual_extraction_response_at_every_accepted_time": True,
            "instantaneous_cumulative_and_window_mean_responses": True,
            "rerun_c4e9_with_actual_fine_response": True,
            "state_spatial_gate_may_not_be_weakened": True,
            "extraction_component_and_aggregate_gates_may_not_be_weakened": True,
            "temporal_fraction_gate": 0.10,
            "surrogate_fraction_gate_removed_only_for_actual_fine_anchor": True,
        },
        "durability": {
            "atomic_checkpoint_after_every_accepted_step": True,
            "retain_previous_complete_checkpoint": True,
            "bind_c4e8_base_and_tangent_hashes": True,
            "bind_fine5_initial_history_hashes": True,
            "bind_execution_source_hashes": True,
            "canonical_integer_target_IDs_and_exact_float_bits": True,
        },
        "cost_control": _cost_projection(),
        "decision": {
            "anchor_and_reanalysis_pass": (
                "issue_20ms_spatial_certificate_and_authorize_50ms_manifest_only"
            ),
            "anchor_method_or_physics_fails": "stop_and_localize_without_operator_reopen",
            "actual_fine_extraction_spatial_gate_fails": (
                "reject_20ms_spatial_certificate_and_localize_components"
            ),
        },
        "hard_stops": (
            "do_not_rerun_the_fine_base",
            "do_not_rerun_other_profiles_nonlinearly",
            "do_not_use_the_raw_inner_face_flux_as_slow_export",
            "do_not_change_operator_profile_or_spatial_thresholds",
            "do_not_launch_50ms_fixed_Q_or_reduced_evolution",
        ),
        "authorized_next": (
            "WP10c9d6c7c3b5c4e11_fine_5_to_20ms_generic_nonlinear_anchor"
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
                    "scientific_status": "PROSPECTIVE",
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
    analysis, fine = _validate_parent()
    manifest = _manifest()
    cost = manifest["cost_control"]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": manifest["classification"],
        "passed": True,
        "definitions_only": True,
        "parent_classification_preserved": analysis["classification"],
        "fine_base_and_tangent_reused": fine["passed"],
        "fine_generic_anchor_propagation_authorized": True,
        "fine_base_rerun_authorized": False,
        "other_nonlinear_profiles_authorized": False,
        "fine_twenty_ms_spatial_certificate_issued": False,
        "fifty_ms_manifest_authorized": False,
        "fifty_ms_propagation_authorized": False,
        "physical_failure_detected": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "projected_raw_wall_hours": cost["projected_raw_wall_hours"],
        "projected_safe_wall_hours": cost["projected_safe_wall_hours"],
        "authorized_next": manifest["authorized_next"],
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "layout": FINE_LAYOUT,
            "profile": GENERIC_PROFILE,
            "start_microseconds": START_MICROSECONDS,
            "stop_microseconds": STOP_MICROSECONDS,
            "target_microseconds": TARGET_MICROSECONDS,
            "temporal_audit_target_microseconds": (
                TEMPORAL_AUDIT_TARGET_MICROSECONDS
            ),
            "coupling_face": COUPLING_FACE,
            "extraction_face": EXTRACTION_FACE,
            "maximum_timestep_seconds": MAXIMUM_TIMESTEP_SECONDS,
        },
    )
    _write_json(MANIFEST_PATH, manifest)
    _write_json(SUMMARY_PATH, summary)
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "PROSPECTIVE",
            "source_parent_commit": ANALYZED_BASE_COMMIT,
            "source_parent_tree_sha": ANALYZED_BASE_TREE,
            "working_head": _git_value("rev-parse", "HEAD"),
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "command": f"PYTHONPATH=src:scripts python {THIS_RUNNER}",
            "input_hashes": {
                "c4e9_summary": _sha256(c4e9.SUMMARY_PATH),
                "c4e9_arrays": _sha256(c4e9.DECISIVE_ARRAYS),
                "c4e8_summary": _sha256(c4e8.SUMMARY_PATH),
                "c4e8_arrays": _sha256(c4e8.DECISIVE_ARRAYS),
                "fine5_summary": _sha256(fine5.SUMMARY_PATH),
                "fine5_arrays": _sha256(fine5.DECISIVE_ARRAYS),
            },
            "implementation_source_hashes": {
                path: _sha256(ROOT / path)
                for path in (THIS_RUNNER, THIS_TEST)
                if (ROOT / path).exists()
            },
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "platform": platform.platform(),
            },
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Fine 20 ms generic-anchor manifest WP10c9d6c7c3b5c4e10",
                "",
                "## Classification",
                "",
                f"`{summary['classification']}`",
                "",
                "The state response is already certified at nearly second order. The aggregate extraction response also contracts near second order, but the fine tangent uncertainty is 42%--62% of the middle--fine extraction difference, above the frozen 10% gate.",
                "",
                "This definitions-only package therefore authorizes exactly one continuous fine generic nonlinear anchor from 5 to 20 ms. It reuses the completed c4e8 fine base, timestep schedule, and tangent; no base or breadth trajectory may be repeated.",
                "",
                f"The measured-cost estimate is `{summary['projected_raw_wall_hours']:.2f} h` raw and `{summary['projected_safe_wall_hours']:.2f} h` with the scheduling safety factor.",
                "",
                "The anchor result must replace the tangent-only fine response in the unchanged c4e9 analysis. No 20 ms certificate, 50 ms work, fixed-Q experiment, or reduced evolution is authorized here.",
                "",
            )
        ),
        encoding="utf-8",
    )
    names = ("anchor_manifest.json", "config.json", "provenance.json", "summary.json")
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
