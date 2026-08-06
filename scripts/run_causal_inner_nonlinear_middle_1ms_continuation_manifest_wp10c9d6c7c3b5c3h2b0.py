#!/usr/bin/env python3
"""Freeze the measured-cost middle continuation from 0.2 to 1 ms."""

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

import run_causal_inner_nonlinear_middle_cost_bounded_anchor_hardening_manifest_wp10c9d6c7c3b5c3h2a0 as h2a0  # noqa: E402
import run_causal_inner_nonlinear_middle_cost_pilot_wp10c9d6c7c3b5c3h2a2 as h2a2  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c3h2b0"
ANALYZED_BASE_COMMIT = "137c844506f369b0c526953385ed7a08c8faefda"
ANALYZED_BASE_PARENT = "df7d660844061c2064a35fcc16e1a2cf590a3926"
ANALYZED_BASE_TREE = "dee1c6eefb424a46af3918a59f09df0a5c48608a"

START_MICROSECONDS = 200
STOP_MICROSECONDS = 1000
TARGET_MICROSECONDS = (200, 400, 600, 800, 1000)
REPLAY_TARGET_MICROSECONDS = (800, 1000)
PROFILES = tuple(h2a2.PROFILES)
MIDDLE_LAYOUT = h2a2.MIDDLE_LAYOUT
GENERIC_PROFILE = h2a2.GENERIC_PROFILE
COUPLING_FACE = int(h2a2.COUPLING_FACE)

ARTIFACT = (
    "causal_inner_nonlinear_middle_1ms_continuation_manifest_"
    "wp10c9d6c7c3b5c3h2b0"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_middle_1ms_continuation_manifest_"
    "wp10c9d6c7c3b5c3h2b0.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_middle_1ms_continuation_manifest_"
    "wp10c9d6c7c3b5c3h2b0.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_MIDDLE_1MS_"
    "CONTINUATION_MANIFEST_WP10C9D6C7C3B5C3H2B0_2026-08-06.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "continuation_manifest.json"
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
    parent = _read_json(h2a2.SUMMARY_PATH)
    if (
        not parent["passed"]
        or not parent["middle_1ms_continuation_manifest_authorized"]
        or parent["middle_1ms_propagation_authorized"]
        or parent["cost_projection"]["resource_tier"]
        != "automatic_continuation"
    ):
        raise RuntimeError("h2b0 pilot authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("h2b0 analyzed identity changed")
    return parent


def _manifest(parent: dict) -> dict:
    pilot_config = _read_json(h2a2.CONFIG_PATH)
    controller = pilot_config["main_controller"]
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            "middle_1ms_continuation_manifest_frozen_cost_bounded_"
            "propagation_authorized"
        ),
        "passed": True,
        "definitions_only": True,
        "propagation_executed": False,
        "operator_changed": False,
        "production_defaults_changed": False,
        "measured_basis": {
            "pilot_classification": parent["classification"],
            "pilot_stop_seconds": h2a2.STOP_SECONDS,
            "pilot_accepted_steps": parent["base"]["accepted_steps"],
            "pilot_rejected_attempts": parent["base"]["rejected_attempts"],
            "maximum_base_local_error_estimate": parent["base"][
                "maximum_local_error_estimate"
            ],
            "local_error_tolerance": controller["error_estimator"][
                "local_tolerance"
            ],
            "median_full_controller_comparison_seconds": parent["base"][
                "median_accepted_step_wall_seconds"
            ],
            "median_sampled_anchor_comparison_seconds": parent["anchor"][
                "median_sampled_anchor_step_wall_seconds"
            ],
            "median_routine_anchor_step_seconds": parent["anchor"][
                "median_unsampled_anchor_step_wall_seconds"
            ],
            "median_tangent_matrix_assembly_seconds": float(
                np.median(parent["tangent"]["matrix_assembly_wall_seconds"])
            ),
            "routine_five_profile_block_solve_seconds": parent["tangent"][
                "routine_block_step_median_wall_seconds"
            ],
            "projected_5ms_wall_hours_with_factor_two_safety": parent[
                "cost_projection"
            ]["projected_total_wall_hours"],
            "resource_tier": parent["cost_projection"]["resource_tier"],
        },
        "continuation": {
            "layout": MIDDLE_LAYOUT,
            "generic_profile": GENERIC_PROFILE,
            "profiles": PROFILES,
            "active_coupling_face": COUPLING_FACE,
            "start_microseconds": START_MICROSECONDS,
            "stop_microseconds": STOP_MICROSECONDS,
            "canonical_target_microseconds": TARGET_MICROSECONDS,
            "replay_target_microseconds": REPLAY_TARGET_MICROSECONDS,
            "single_integer_target_source_required": True,
            "restart_source": (
                "h2a2_canonical_decisive_arrays_exact_0p2ms_base_anchor_and_"
                "BDF_histories"
            ),
            "restart_roundtrip_bitwise_required": True,
            "no_new_BDF1_startup": True,
            "initial_candidate_timestep_seconds": parent["base"][
                "next_candidate_timestep_seconds"
            ],
            "durable_checkpoint_after_every_declared_target": True,
            "maximum_unattended_checkpoint_interval_hours": 4.0,
            "stop_at_1ms_before_any_2ms_work": True,
        },
        "base_schedule_contract": {
            "layout_owns_schedule": True,
            "coarse_schedule_reuse_forbidden": True,
            "full_step_doubling_on_every_base_accepted_comparison": True,
            "controller": controller,
            "all_method_physics_and_ledger_gates_unchanged": True,
            "accepted_step_artifact_cached_once": True,
            "cached_items": (
                "old_and_new_base_primitives",
                "primitive_mapped_and_height_BDF_histories",
                "accepted_and_previous_timestep",
                "state_and_Tier_I_exports",
                "method_and_physical_audits",
            ),
        },
        "generic_anchor_contract": {
            "replays_exact_base_accepted_schedule": True,
            "tangent_prediction_used_as_Newton_initial_guess": True,
            "step_doubling_audit_locations": (
                "first_continuation_step",
                "first_new_maximum_timestep_or_ratio_transition",
                "final_step_landing_at_1ms",
            ),
            "all_other_steps_use_one_full_nonlinear_solve": True,
            "any_sampled_error_failure_stops_continuation": True,
            "maximum_sampled_state_error": controller["error_estimator"][
                "local_tolerance"
            ],
            "maximum_sampled_Tier_I_error": controller["error_estimator"][
                "local_tolerance"
            ],
        },
        "tangent_contract": {
            "all_profiles_propagated_as_one_block": True,
            "one_step_matrix_and_factorization_per_base_step": True,
            "complete_residual_JVP_audit_locations": (
                "first_continuation_step",
                "first_new_nonunit_step_ratio",
                "final_step_landing_at_1ms",
            ),
            "surrogate_gates": h2a0.SURROGATE_GATES,
            "generic_anchor_closes_full_0p2_to_1ms_interval": True,
            "non_generic_full_nonlinear_trajectories_forbidden": True,
        },
        "replay_and_evidence_contract": {
            "base_and_anchor_last_step_serialized_replay_bitwise": True,
            "accepted_timestep_schedule_bitwise": True,
            "complete_restart_payload_bitwise": True,
            "canonical_states_exports_and_histories_committed": True,
            "source_and_input_hashes_required": True,
            "results_resumable_by_stage_and_declared_target": True,
        },
        "decision": {
            "pass_action": (
                "authorize_only_fresh_definitions_only_middle_2ms_"
                "continuation_manifest"
            ),
            "generic_anchor_failure_action": (
                "stop_and_localize_surrogate_or_temporal_defect"
            ),
            "base_method_or_physics_failure_action": (
                "stop_without_fine_work_or_operator_redesign_unless_localized"
            ),
            "cost_projection_alone_is_not_a_scientific_rejection": True,
        },
        "downstream_stops": {
            "middle_2ms_propagation_authorized": False,
            "middle_5ms_spatial_confirmation_certified": False,
            "fine_cost_bounded_propagation_authorized": False,
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
    manifest = summary["manifest"]
    measured = manifest["measured_basis"]
    return "\n".join(
        (
            "# Middle 1 ms continuation manifest WP10c9d6c7c3b5c3h2b0",
            "",
            "## Classification",
            "",
            f"`{summary['classification']}`",
            "",
            "The executed 0.2 ms pilot passed the nonlinear base, generic anchor, five-profile tangent, method, physical, ledger, and bitwise replay gates. This package freezes—but does not execute—the middle continuation to 1 ms.",
            "",
            f"The factor-two cost model projects `{measured['projected_5ms_wall_hours_with_factor_two_safety']:.2f}` hours through 5 ms, so the campaign remains in the `{measured['resource_tier']}` tier. The middle base retains full step-doubling; cost is controlled by sampling step-doubling only on the generic anchor and by propagating all five profile responses in one tangent block.",
            "",
            "Canonical targets are constructed once from integer microseconds. Base and anchor states, complete BDF histories, timesteps, exports, and restart payloads remain binding. A pass at 1 ms authorizes only a fresh 2 ms continuation manifest.",
            "",
            "Fine propagation, the 5 ms spatial certificate, fixed-Q experiments, and reduced slow evolution remain blocked.",
            "",
        )
    )


def main() -> int:
    parent = _validate_parent()
    manifest = _manifest(parent)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": manifest["classification"],
        "passed": True,
        "definitions_only": True,
        "manifest": manifest,
        "middle_1ms_propagation_authorized": True,
        "middle_2ms_propagation_authorized": False,
        "middle_5ms_spatial_confirmation_certified": False,
        "fine_cost_bounded_propagation_authorized": False,
        "third_duration_rung_spatial_convergence_certified": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": (
            "WP10c9d6c7c3b5c3h2b1_middle_0p2_to_1ms_continuation"
        ),
    }
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "layout": MIDDLE_LAYOUT,
        "profiles": PROFILES,
        "generic_profile": GENERIC_PROFILE,
        "active_coupling_face": COUPLING_FACE,
        "target_microseconds": TARGET_MICROSECONDS,
        "replay_target_microseconds": REPLAY_TARGET_MICROSECONDS,
        "pilot_cost_projection": parent["cost_projection"],
        "surrogate_gates": h2a0.SURROGATE_GATES,
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
                "pilot_config": _sha256(h2a2.CONFIG_PATH),
                "pilot_summary": _sha256(h2a2.SUMMARY_PATH),
                "pilot_decisive_arrays": _sha256(h2a2.DECISIVE_ARRAYS),
                "hardening_manifest": _sha256(h2a0.MANIFEST_PATH),
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
    names = (
        "config.json",
        "continuation_manifest.json",
        "provenance.json",
        "summary.json",
    )
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
