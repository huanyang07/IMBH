#!/usr/bin/env python3
"""Freeze prospective hardening for the cost-bounded middle campaign."""

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

import run_causal_inner_nonlinear_middle_cost_bounded_anchor_manifest_wp10c9d6c7c3b5c3h2 as h2  # noqa: E402
import run_causal_inner_nonlinear_coarse_third_duration_rung_completion_wp10c9d6c7c3b5c3d as c3d  # noqa: E402
import run_causal_inner_nonlinear_coarse_heldout_third_duration_rung_screen_wp10c9d6c7c3b5c3f as c3f  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c3h2a0"
ANALYZED_BASE_COMMIT = "32b679f85bd27fffda7d0e8bef205be48b8795ce"
ANALYZED_BASE_PARENT = "71972ba60ec78f6e16eb672bc3815fb8e40980a6"
ANALYZED_BASE_TREE = "661bbc9fad78215855292f69fe8ac8e99574db89"

PROFILES = tuple(h2.PROFILES)
MIDDLE_LAYOUT = h2.MIDDLE_LAYOUT
RATIO_AUDIT_VALUES = (0.5, 1.0, 2.0)
SURROGATE_GATES = {
    "maximum_absolute_scaled_state_discrepancy": 5.0e-3,
    "maximum_absolute_scaled_Tier_I_discrepancy": 5.0e-3,
    "maximum_discrepancy_fraction_of_observable_response": 1.0e-2,
    "maximum_uncertainty_fraction_of_observable_spatial_difference": 1.0e-1,
    "maximum_internal_discrete_residual_jvp_relative_defect": 1.0e-6,
    "minimum_state_history_cosine": 0.99,
    "minimum_Tier_I_history_cosine": 0.99,
    "unobservable_spatial_difference_action": (
        "report_upper_bound_or_run_full_nonlinear_anchor_do_not_report_order"
    ),
}
RESOURCE_POLICY = {
    "projected_wall_hours_at_most_24": "continue_automatically",
    "projected_wall_hours_24_to_48": (
        "continue_after_factorization_context_and_schedule_reuse_review"
    ),
    "projected_wall_hours_above_48": "explicit_cost_benefit_decision",
    "maximum_unattended_checkpoint_interval_hours": 4.0,
    "scientific_rejection_from_cost_projection_alone_forbidden": True,
    "pilot_minimum_accepted_steps": 5,
    "pilot_must_reach_controller_timestep_plateau_or_step_minimum": True,
    "projection_components": (
        "context_setup",
        "base_accepted_steps",
        "generic_anchor_schedule_replay",
        "tangent_matrix_assembly_and_block_solve",
        "serialized_replay",
        "sampled_strict_shadow",
        "evidence_packaging",
    ),
}

ARTIFACT = (
    "causal_inner_nonlinear_middle_cost_bounded_anchor_hardening_manifest_"
    "wp10c9d6c7c3b5c3h2a0"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_middle_cost_bounded_anchor_"
    "hardening_manifest_wp10c9d6c7c3b5c3h2a0.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_middle_cost_bounded_anchor_"
    "hardening_manifest_wp10c9d6c7c3b5c3h2a0.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_MIDDLE_COST_BOUNDED_"
    "ANCHOR_HARDENING_MANIFEST_WP10C9D6C7C3B5C3H2A0_2026-08-06.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "hardening_manifest.json"
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


def _load_times(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        return np.asarray(payload[key], dtype=float)


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
    parent = _read_json(h2.SUMMARY_PATH)
    if (
        not parent["passed"]
        or not parent["middle_staged_execution_authorized"]
        or parent["fine_cost_bounded_propagation_authorized"]
    ):
        raise RuntimeError("h2 authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("h2a0 analyzed identity changed")
    return parent


def _archive_scope() -> dict:
    generic_times = _load_times(c3d.DECISIVE_ARRAYS, "main_times_seconds")
    heldout_times = _load_times(c3f.DECISIVE_ARRAYS, "main_times_seconds")
    same = np.array_equal(generic_times, heldout_times)
    return {
        "generic_and_heldout_output_times_bitwise_equal": same,
        "stored_output_times_seconds": generic_times,
        "first_stored_long_duration_output_seconds": float(generic_times[0]),
        "five_profile_long_tangent_calibration_start_seconds": float(
            generic_times[1]
        ),
        "last_stored_long_duration_output_seconds": float(generic_times[-1]),
        "accepted_internal_states_stored_for_40us_to_first_output": False,
        "five_profile_40us_to_5ms_tangent_replay_claim_authorized": False,
        "binding_interpretation": (
            "long five-profile tangent calibration covers stored 2.4ms-to-5ms "
            "outputs; the full middle generic nonlinear anchor must certify the "
            "unstored early interval"
        ),
    }


def _manifest() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            "middle_cost_bounded_anchor_hardening_frozen_cheap_audits_and_"
            "0p2ms_pilot_authorized"
        ),
        "passed": True,
        "definitions_only": True,
        "propagation_executed": False,
        "operator_changed": False,
        "production_defaults_changed": False,
        "archive_scope": _archive_scope(),
        "cheap_prerequisite_audits": {
            "analytic_initial_BDF_history_direction_required": True,
            "centered_difference_history_direction_retained_as_audit": True,
            "middle_all_profile_short_step_interval_seconds": (3.0e-5, 4.0e-5),
            "middle_all_profile_short_step_profiles": PROFILES,
            "variable_BDF_step_ratio_audit_values": RATIO_AUDIT_VALUES,
            "variable_ratio_complete_residual_JVP_required": True,
            "no_new_physical_trajectory_required": True,
        },
        "surrogate_contract": SURROGATE_GATES,
        "resource_policy": RESOURCE_POLICY,
        "pilot_contract": {
            "layout": MIDDLE_LAYOUT,
            "stop_seconds": 2.0e-4,
            "one_adaptive_nonlinear_base": True,
            "one_generic_nonlinear_anchor_on_base_schedule": True,
            "five_profile_discrete_tangent_block": True,
            "tangent_prediction_used_as_anchor_Newton_initial_guess": True,
            "complete_residual_JVP_checkpoints": (
                "first_accepted_step",
                "first_nonunit_step_ratio",
                "final_pilot_step",
            ),
            "all_common_strict_outputs_compared": True,
            "durable_accepted_step_artifacts_required": True,
            "continue_to_1ms_without_fresh_decision_forbidden": True,
        },
        "fine_strategy": {
            "fine_work_before_middle_5ms_pass_forbidden": True,
            "minimum_fine_work_after_middle_pass": (
                "fine_nonlinear_base",
                "five_profile_fine_discrete_tangent",
                "sampled_temporal_audits",
            ),
            "full_fine_generic_anchor_is_conditional": True,
            "full_fine_generic_anchor_triggers": (
                "middle_generic_tangent_anchor_failure",
                "surrogate_uncertainty_above_10_percent_of_middle_fine_difference",
                "spatial_result_near_a_binding_gate",
                "observable_nonlinear_remainder",
            ),
        },
        "downstream_stops": {
            "middle_1ms_propagation_authorized": False,
            "middle_5ms_spatial_confirmation_certified": False,
            "fine_propagation_authorized": False,
            "third_duration_rung_spatial_convergence_certified": False,
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
    scope = summary["manifest"]["archive_scope"]
    return "\n".join(
        (
            "# Middle cost-bounded anchor hardening manifest WP10c9d6c7c3b5c3h2a0",
            "",
            "## Classification",
            "",
            f"`{summary['classification']}`",
            "",
            "This prospective addendum preserves the h2 scientific campaign while tightening the tangent, surrogate, runtime, and evidence contracts before the first middle propagation.",
            "",
            f"The stored long-duration five-profile outputs begin at `{scope['first_stored_long_duration_output_seconds']:.4e} s`, while the committed tangent calibration begins at `{scope['five_profile_long_tangent_calibration_start_seconds']:.4e} s`; accepted internal states from 40 microseconds to the first output are not archived. Therefore no full-horizon five-profile tangent replay is claimed. The complete middle generic nonlinear anchor remains binding for that interval.",
            "",
            "The 24-hour projection is now a soft scheduling tier, not a scientific rejection. The pilot must measure at least five accepted steps or reach the controller timestep plateau and must project setup, base, anchor, tangent, replay, strict-shadow, and packaging costs separately.",
            "",
            "Only the 0.2 ms middle pilot is authorized. Fine propagation, fixed-Q experiments, and reduced slow evolution remain blocked.",
            "",
        )
    )


def main() -> int:
    _validate_parent()
    manifest = _manifest()
    if not manifest["archive_scope"]["generic_and_heldout_output_times_bitwise_equal"]:
        raise RuntimeError("coarse archive output schedules changed")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": manifest["classification"],
        "passed": True,
        "definitions_only": True,
        "manifest": manifest,
        "cheap_hardening_audits_authorized": True,
        "middle_0p2ms_pilot_authorized_after_cheap_audits": True,
        "middle_1ms_propagation_authorized": False,
        "fine_cost_bounded_propagation_authorized": False,
        "third_duration_rung_spatial_convergence_certified": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": (
            "WP10c9d6c7c3b5c3h2a_middle_cheap_audits_and_0p2ms_cost_pilot"
        ),
    }
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "profiles": PROFILES,
        "middle_layout": MIDDLE_LAYOUT,
        "ratio_audit_values": RATIO_AUDIT_VALUES,
        "surrogate_gates": SURROGATE_GATES,
        "resource_policy": RESOURCE_POLICY,
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(CONFIG_PATH, config)
    _write_json(MANIFEST_PATH, manifest)
    _write_json(SUMMARY_PATH, summary)
    _write_json(
        PROVENANCE_PATH,
        {
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
                "h2_summary": _sha256(h2.SUMMARY_PATH),
                "h2_manifest": _sha256(h2.MANIFEST_PATH),
                "h1_summary": _sha256(h2.h1.SUMMARY_PATH),
                "generic_duration_arrays": _sha256(c3d.DECISIVE_ARRAYS),
                "heldout_duration_arrays": _sha256(c3f.DECISIVE_ARRAYS),
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
            "command": f"PYTHONPATH=src:scripts python {THIS_RUNNER}",
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_report(summary), encoding="utf-8")
    hash_names = ("config.json", "hardening_manifest.json", "provenance.json", "summary.json")
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in hash_names
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
