#!/usr/bin/env python3
"""Freeze the cost-bounded fine-layout 5-to-20 ms spatial certificate."""

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
import run_causal_inner_nonlinear_discrete_bdf_tangent_calibration_wp10c9d6c7c3b5c3h1 as tangent_calibration  # noqa: E402
import run_causal_inner_nonlinear_middle_20ms_temporal_reference_shadow_wp10c9d6c7c3b5c4e6 as c4e6  # noqa: E402
import run_causal_inner_nonlinear_optimized_middle_20ms_completion_wp10c9d6c7c3b5c4e3 as c4e3  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4e7"
ANALYZED_BASE_COMMIT = c4e6.ANALYZED_BASE_COMMIT
ANALYZED_BASE_PARENT = c4e6.ANALYZED_BASE_PARENT
ANALYZED_BASE_TREE = c4e6.ANALYZED_BASE_TREE

FINE_LAYOUT = fine5.FINE_LAYOUT
PROFILES = tuple(fine5.PROFILES)
GENERIC_PROFILE = fine5.GENERIC_PROFILE
COUPLING_FACE = 192
EXTRACTION_FACE = 8
EXTRACTION_RADIUS_RG = c4e3.c4e1.c4e.EXTRACTION_RADIUS_RG
START_MICROSECONDS = 5_000
PILOT_STOP_MICROSECONDS = 6_000
STOP_MICROSECONDS = 20_000
TARGET_MICROSECONDS = (
    5_000,
    5_400,
    6_000,
    8_000,
    10_000,
    12_000,
    14_000,
    16_000,
    18_000,
    18_800,
    19_600,
    19_800,
    20_000,
)
AUDIT_TARGET_MICROSECONDS = (5_400, 8_000, 12_000, 16_000, 18_000, 20_000)
MAXIMUM_TIMESTEP_SECONDS = 4.0e-4
PROJECTED_ACCEPTED_STEPS = 39
COST_SAFETY_FACTOR = 1.25

ARTIFACT = (
    "causal_inner_nonlinear_cost_bounded_fine_20ms_manifest_"
    "wp10c9d6c7c3b5c4e7"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_cost_bounded_fine_20ms_manifest_"
    "wp10c9d6c7c3b5c4e7.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_cost_bounded_fine_20ms_manifest_"
    "wp10c9d6c7c3b5c4e7.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_COST_BOUNDED_FINE_20MS_"
    "MANIFEST_WP10C9D6C7C3B5C4E7_2026-08-11.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "fine_completion_manifest.json"
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


def _cost_projection() -> dict:
    middle = _read_json(c4e3.SUMMARY_PATH)
    calibration = _read_json(tangent_calibration.SUMMARY_PATH)
    layouts = calibration["calibration"]["short"]
    middle_matrix = float(
        layouts[fine5.h2e0.MIDDLE_LAYOUT]["matrix_assembly_wall_seconds"]
    )
    fine_matrix = float(layouts[FINE_LAYOUT]["matrix_assembly_wall_seconds"])
    scaling = fine_matrix / middle_matrix
    routine = float(middle["base"]["median_routine_step_wall_seconds"]) * scaling
    audit = float(middle["base"]["median_audit_step_wall_seconds"]) * scaling
    audit_count = len(AUDIT_TARGET_MICROSECONDS)
    routine_count = PROJECTED_ACCEPTED_STEPS - audit_count
    tangent = PROJECTED_ACCEPTED_STEPS * fine_matrix
    sampled_anchor_count = len(AUDIT_TARGET_MICROSECONDS)
    sampled_anchor = sampled_anchor_count * float(
        _read_json(fine5.SUMMARY_PATH)["anchor"][
            "median_routine_step_wall_seconds"
        ]
    )
    setup_replay_io = 3600.0
    raw = (
        routine_count * routine
        + audit_count * audit
        + tangent
        + sampled_anchor
        + setup_replay_io
    )
    return {
        "basis": "measured_middle_cost_times_measured_fine_to_middle_matrix_scaling",
        "fine_to_middle_scaling": scaling,
        "projected_accepted_steps": PROJECTED_ACCEPTED_STEPS,
        "projected_routine_steps": routine_count,
        "projected_audit_steps": audit_count,
        "projected_routine_step_wall_seconds": routine,
        "projected_audit_step_wall_seconds": audit,
        "projected_tangent_matrix_wall_seconds": tangent,
        "projected_sampled_anchor_count": sampled_anchor_count,
        "projected_sampled_anchor_wall_seconds": sampled_anchor,
        "projected_setup_replay_io_wall_seconds": setup_replay_io,
        "projected_raw_wall_seconds": raw,
        "projected_raw_wall_hours": raw / 3600.0,
        "safety_factor": COST_SAFETY_FACTOR,
        "projected_safe_wall_hours": COST_SAFETY_FACTOR * raw / 3600.0,
        "projection_is_scheduling_only": True,
    }


def _validate_parent() -> tuple[dict, dict]:
    parent = _read_json(c4e6.SUMMARY_PATH)
    middle = _read_json(c4e3.SUMMARY_PATH)
    if (
        not parent["passed"]
        or not parent["temporal_reference_hardened"]
        or not parent["fine_twenty_ms_manifest_authorized"]
        or parent["fine_twenty_ms_propagation_authorized"]
        or parent["full_fine_generic_anchor_required"]
        or parent["physical_failure_detected"]
        or parent["fixed_q_micro_solver_authorized"]
        or parent["reduced_slow_evolution_authorized"]
    ):
        raise RuntimeError("c4e7 temporal-reference authorization changed")
    if not middle["passed"] or middle["physical_failure_detected"]:
        raise RuntimeError("c4e7 middle completion evidence changed")
    if any(
        item["temporal_to_spatial_fraction"] > 0.10
        for item in parent["analysis"]["observables"].values()
    ):
        raise RuntimeError("c4e7 temporal gate changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c4e7 analyzed identity changed")
    return parent, middle


def _manifest() -> dict:
    projection = _cost_projection()
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            "cost_bounded_fine_20ms_spatial_certificate_manifest_frozen_"
            "fine_base_block_tangent_propagation_authorized"
        ),
        "definitions_frozen_before_propagation": True,
        "propagation_executed": False,
        "operator_changed": False,
        "production_defaults_changed": False,
        "scientific_scope": {
            "layout": FINE_LAYOUT,
            "start_microseconds": START_MICROSECONDS,
            "pilot_stop_microseconds": PILOT_STOP_MICROSECONDS,
            "stop_microseconds": STOP_MICROSECONDS,
            "target_microseconds": TARGET_MICROSECONDS,
            "audit_target_microseconds": AUDIT_TARGET_MICROSECONDS,
            "profiles": PROFILES,
            "generic_profile": GENERIC_PROFILE,
            "coupling_face": COUPLING_FACE,
            "extraction_face": EXTRACTION_FACE,
            "extraction_radius_rg": EXTRACTION_RADIUS_RG,
            "slow_export": "certified_conservative_exterior_partition",
            "raw_inner_face_is_not_a_slow_export": True,
        },
        "minimum_work": {
            "nonlinear_fine_base": "required",
            "five_profile_block_tangent": "required_on_every_accepted_base_step",
            "generic_fine_nonlinear_anchor": "not_initially_required",
            "all_profile_tangents_share_one_matrix_factorization": True,
            "full_fine_anchor_may_not_launch_inside_this_campaign": True,
        },
        "controller": {
            "fine_layout_owns_schedule": True,
            "maximum_timestep_seconds": MAXIMUM_TIMESTEP_SECONDS,
            "larger_timestep_preflight_forbidden_after_middle_0p8ms_failure": True,
            "routine_base_step": "one_full_nonlinear_BDF2_solve",
            "routine_error_bound": "four_times_last_audited_error_scaled_by_dt_cubed",
            "routine_error_bound_safety_factor": 4.0,
            "audit_step": "one_full_step_plus_two_half_steps",
            "audit_target_microseconds": AUDIT_TARGET_MICROSECONDS,
            "maximum_BDF2_step_ratio": 2.0,
        },
        "nonlinear_remainder": {
            "generic_tangent_predicted_state_is_audit_reference": True,
            "one_step_nonlinear_generic_anchor_shadow_at_audit_targets": True,
            "shadow_uses_tangent_predicted_incoming_state_and_BDF_history": True,
            "shadow_uses_exact_accepted_base_timestep": True,
            "continuous_or_full_generic_anchor_forbidden": True,
            "maximum_correction_fraction_of_generic_response": 0.01,
            "maximum_correction_fraction_of_middle_fine_difference": 0.10,
            "failure_action": "stop_and_authorize_separate_fine_generic_anchor_manifest",
            "automatic_full_anchor_forbidden": True,
        },
        "spatial_certificate": {
            "state_restricted_to_common_64_cell_parent": True,
            "instantaneous_extraction_required": True,
            "cumulative_extraction_interval_seconds": (0.005, 0.020),
            "window_mean_intervals_seconds": (
                (0.005, 0.020),
                (0.010, 0.020),
                (0.016, 0.020),
            ),
            "minimum_RMS_order": 0.75,
            "minimum_maximum_order": 0.75,
            "minimum_significant_component_order": 0.75,
            "minimum_refinement_error_cosine": 0.90,
            "maximum_fine_normalized_difference": 0.05,
            "maximum_temporal_fraction_of_middle_fine_difference": 0.10,
            "maximum_surrogate_fraction_of_middle_fine_difference": 0.10,
            "unobservable_difference_yields_upper_bound_not_order": True,
        },
        "method_gates": {
            "maximum_local_error_estimate": 2.5e-4,
            "minimum_audit_error_margin_factor": 10.0,
            "maximum_scaled_nonlinear_residual": 1.0e-10,
            "maximum_discrete_ledger_defect": 1.0e-12,
            "maximum_mapped_endpoint_path_closure_defect": 1.0e-9,
            "maximum_extraction_identity_defect": 1.0e-12,
            "maximum_shared_conservative_face_defect": 1.0e-12,
            "maximum_source_double_count_defect": 1.0e-12,
            "minimum_reconstruction_factor": 1.0,
            "maximum_incoming_excision_characteristics": 0,
            "maximum_local_error_bound": 2.5e-4,
            "maximum_sum_of_local_error_bounds": 5.0e-3,
            "maximum_tangent_linear_solve_relative_defect": 1.0e-10,
            "maximum_tangent_matrix_JVP_relative_defect": 1.0e-8,
            "same_target_final_restart_replay_bitwise": True,
        },
        "cost_control": {
            **projection,
            "pilot_reprojection_required_at_6ms": True,
            "pilot_requires_at_least_two_routine_steps_and_one_audit": True,
            "up_to_30h": "continue_automatically_if_scientific_gates_pass",
            "30_to_40h": "optimization_review_then_continue_if_justified",
            "over_40h": "scheduling_stop_and_redesign_not_scientific_failure",
            "maximum_unattended_segment_hours": 6.0,
        },
        "durability": {
            "atomic_generation_checkpoint_after_every_accepted_step": True,
            "retain_previous_complete_generation": True,
            "complete_controller_and_BDF2_state_required": True,
            "source_dependency_hashes_binding": True,
            "canonical_integer_target_IDs_required": True,
            "exact_float64_target_bits_verified": True,
        },
        "decision_branches": {
            "pilot_or_method_gate_fails": "stop_before_20ms_continuation",
            "cost_only_stop": "preserve_scientific_status_and_optimize",
            "nonlinear_remainder_trigger_fails": (
                "authorize_separate_definitions_only_full_fine_anchor_manifest"
            ),
            "base_tangent_and_spatial_gates_pass": (
                "issue_20ms_spatial_certificate_and_authorize_next_duration_manifest"
            ),
            "spatial_gate_fails": "reject_20ms_spatial_certificate_and_localize",
        },
        "hard_stops": (
            "do_not_run_a_routine_full_fine_generic_anchor",
            "do_not_change_operator_profile_or_thresholds",
            "do_not_raise_the_0p4ms_timestep_cap",
            "do_not_use_raw_inner_face_flux_as_slow_export",
            "do_not_run_50ms_fixed_Q_or_reduced_evolution",
            "do_not_claim_spatial_order_when_middle_fine_difference_is_unobservable",
        ),
        "authorized_next": (
            "WP10c9d6c7c3b5c4e8_cost_bounded_fine_5_to_20ms_base_tangent_campaign"
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
    parent, _middle = _validate_parent()
    manifest = _manifest()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": manifest["classification"],
        "passed": True,
        "definitions_only": True,
        "parent_classification_preserved": parent["classification"],
        "fine_base_block_tangent_propagation_authorized": True,
        "full_fine_generic_anchor_required": False,
        "full_fine_generic_anchor_authorized": False,
        "fine_twenty_ms_spatial_certificate_issued": False,
        "fifty_ms_propagation_authorized": False,
        "physical_failure_detected": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "projected_raw_wall_hours": manifest["cost_control"][
            "projected_raw_wall_hours"
        ],
        "projected_safe_wall_hours": manifest["cost_control"][
            "projected_safe_wall_hours"
        ],
        "authorized_next": manifest["authorized_next"],
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "layout": FINE_LAYOUT,
            "profiles": PROFILES,
            "generic_profile": GENERIC_PROFILE,
            "coupling_face": COUPLING_FACE,
            "extraction_face": EXTRACTION_FACE,
            "extraction_radius_rg": EXTRACTION_RADIUS_RG,
            "target_microseconds": TARGET_MICROSECONDS,
            "audit_target_microseconds": AUDIT_TARGET_MICROSECONDS,
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
            "command": f"PYTHONPATH=src:scripts python {THIS_RUNNER}",
            "source_parent_commit": ANALYZED_BASE_COMMIT,
            "source_parent_tree_sha": ANALYZED_BASE_TREE,
            "working_head": _git_value("rev-parse", "HEAD"),
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "input_hashes": {
                "c4e6_summary": _sha256(c4e6.SUMMARY_PATH),
                "c4e6_arrays": _sha256(c4e6.DECISIVE_ARRAYS),
                "c4e3_summary": _sha256(c4e3.SUMMARY_PATH),
                "fine_5ms_summary": _sha256(fine5.SUMMARY_PATH),
                "fine_5ms_arrays": _sha256(fine5.DECISIVE_ARRAYS),
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
                "# Cost-bounded fine 20 ms manifest WP10c9d6c7c3b5c4e7",
                "",
                "## Classification",
                "",
                f"`{summary['classification']}`",
                "",
                "This definitions-only package freezes one nonlinear fine base and one five-profile block tangent from `5` to `20 ms`.",
                "",
                f"The measured-data projection is `{summary['projected_raw_wall_hours']:.2f} h` raw and `{summary['projected_safe_wall_hours']:.2f} h` with the frozen safety factor.",
                "",
                "The first `5 -> 6 ms` stage is a measured cost and method pilot. A full fine generic nonlinear anchor is neither required nor authorized; a nonlinear-remainder trigger can only authorize a separate definitions-only anchor manifest.",
                "",
                "No propagation occurs here. The 20 ms spatial certificate, 50 ms evolution, fixed-Q experiments, and reduced slow evolution remain blocked.",
                "",
            )
        ),
        encoding="utf-8",
    )
    names = (
        "config.json",
        "fine_completion_manifest.json",
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
