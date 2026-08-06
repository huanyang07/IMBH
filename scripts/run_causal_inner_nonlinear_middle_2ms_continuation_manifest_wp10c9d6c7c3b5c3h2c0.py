#!/usr/bin/env python3
"""Freeze the cost-bounded middle continuation from 1 to 2 ms."""

from __future__ import annotations

import csv
import hashlib
import json
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

import run_causal_inner_nonlinear_middle_1ms_continuation_wp10c9d6c7c3b5c3h2b1 as h2b1  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c3h2c0"
ANALYZED_BASE_COMMIT = "ea3afa95ca57d8dabd6cde8a9c31ff53bf179662"
ANALYZED_BASE_PARENT = "5c84f134f6905215c3088bdfa909e8705548c648"
ANALYZED_BASE_TREE = "98d9e28267191711fedf0de21e1b45c61e0c4f15"

TARGET_MICROSECONDS = (1000, 1200, 1600, 2000)
REPLAY_TARGET_MICROSECONDS = (1600, 2000)
STRICT_SAMPLE_TARGET_MICROSECONDS = (1200, 1600, 2000)
ARTIFACT = (
    "causal_inner_nonlinear_middle_2ms_continuation_manifest_"
    "wp10c9d6c7c3b5c3h2c0"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_middle_2ms_continuation_manifest_"
    "wp10c9d6c7c3b5c3h2c0.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_middle_2ms_continuation_manifest_"
    "wp10c9d6c7c3b5c3h2c0.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_MIDDLE_2MS_"
    "CONTINUATION_MANIFEST_WP10C9D6C7C3B5C3H2C0_2026-08-06.md"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "continuation_manifest.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _validate_parent() -> dict:
    parent = _read_json(h2b1.SUMMARY_PATH)
    if (
        not parent["passed"]
        or not parent["middle_2ms_continuation_manifest_authorized"]
        or parent["middle_2ms_propagation_authorized"]
        or parent["remaining_cost_projection"]["resource_tier"]
        != "automatic_continuation"
    ):
        raise RuntimeError("h2c0 1 ms authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("h2c0 analyzed identity changed")
    return parent


def _manifest(parent: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            "middle_2ms_continuation_manifest_frozen_cost_bounded_"
            "propagation_authorized"
        ),
        "passed": True,
        "definitions_only": True,
        "propagation_executed": False,
        "operator_changed": False,
        "production_defaults_changed": False,
        "continuation": {
            "layout": h2b1.MIDDLE_LAYOUT,
            "profiles": list(h2b1.PROFILES),
            "generic_profile": h2b1.GENERIC_PROFILE,
            "active_coupling_face": h2b1.COUPLING_FACE,
            "canonical_target_microseconds": list(TARGET_MICROSECONDS),
            "replay_target_microseconds": list(REPLAY_TARGET_MICROSECONDS),
            "strict_sample_target_microseconds": list(
                STRICT_SAMPLE_TARGET_MICROSECONDS
            ),
            "target_1ms_inherited_bitwise_from_h2b1": True,
            "targets_after_1ms_constructed_once_from_integer_microseconds": True,
            "initial_candidate_timestep_seconds": parent["base"][
                "next_candidate_timestep_seconds"
            ],
            "maximum_timestep_seconds": 4.0e-4,
            "plateau_schedule_intent": (
                "one 0p2ms landing step followed by two 0p4ms plateau steps"
            ),
            "durable_checkpoint_after_every_declared_target": True,
            "no_new_BDF1_startup": True,
        },
        "base_contract": {
            "layout_owns_schedule": True,
            "full_step_doubling_on_every_accepted_comparison": True,
            "all_method_physics_and_ledger_gates_unchanged": True,
            "maximum_scaled_residual": 1.0e-10,
            "maximum_discrete_ledger_defect": 1.0e-12,
            "maximum_mapped_endpoint_path_closure_defect": 1.0e-9,
            "incoming_excision_characteristics": 0,
            "minimum_path_reconstruction_factor": 1.0,
        },
        "tangent_contract": {
            "all_five_profiles_in_one_block": True,
            "one_matrix_and_factorization_per_base_step": True,
            "independent_JVP_audits": ["first", "first_ratio_transition", "final"],
            "generic_anchor_is_binding_surrogate_calibration": True,
            "non_generic_full_nonlinear_runs_forbidden": True,
        },
        "anchor_contract": {
            "replay_exact_base_schedule": True,
            "tangent_prediction_is_only_Newton_initial_guess": True,
            "sampled_full_vs_two_half_steps_at_declared_targets": True,
            "maximum_sampled_state_error": 2.5e-4,
            "maximum_sampled_Tier_I_error": 2.5e-4,
        },
        "replay_contract": {
            "base_and_anchor_last_step_replay_bitwise": True,
            "checkpoint_roundtrip_bitwise": True,
            "complete_BDF_histories_and_previous_timestep_binding": True,
        },
        "measured_basis": {
            "parent_classification": parent["classification"],
            "base_accepted_steps": parent["base"]["accepted_steps"],
            "base_rejected_attempts": parent["base"]["rejected_attempts"],
            "maximum_base_local_error": parent["base"][
                "maximum_local_error_estimate"
            ],
            "median_full_controller_comparison_seconds": parent["base"][
                "median_accepted_step_wall_seconds"
            ],
            "median_routine_anchor_seconds": parent["anchor"][
                "median_routine_step_wall_seconds"
            ],
            "median_sampled_anchor_seconds": parent["anchor"][
                "median_sampled_step_wall_seconds"
            ],
            "median_tangent_matrix_seconds": parent["tangent"][
                "median_matrix_assembly_wall_seconds"
            ],
            "median_five_profile_block_seconds": parent["tangent"][
                "median_routine_block_step_wall_seconds"
            ],
            "projected_remaining_5ms_wall_hours": parent[
                "remaining_cost_projection"
            ]["projected_remaining_wall_hours"],
            "resource_tier": parent["remaining_cost_projection"][
                "resource_tier"
            ],
        },
        "decision": {
            "pass_action": (
                "authorize_only_fresh_definitions_only_middle_5ms_completion_manifest"
            ),
            "failure_action": "stop_and_localize_before_any_5ms_or_fine_work",
            "cost_projection_is_not_a_scientific_gate": True,
        },
        "downstream_stops": {
            "middle_5ms_propagation_authorized": False,
            "middle_5ms_spatial_confirmation_certified": False,
            "fine_cost_bounded_propagation_authorized": False,
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
        "middle_2ms_propagation_authorized": True,
        "middle_5ms_propagation_authorized": False,
        "middle_5ms_spatial_confirmation_certified": False,
        "fine_cost_bounded_propagation_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": (
            "WP10c9d6c7c3b5c3h2c1_middle_1_to_2ms_continuation"
        ),
    }
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "target_microseconds": list(TARGET_MICROSECONDS),
        "replay_target_microseconds": list(REPLAY_TARGET_MICROSECONDS),
        "strict_sample_target_microseconds": list(
            STRICT_SAMPLE_TARGET_MICROSECONDS
        ),
        "active_coupling_face": h2b1.COUPLING_FACE,
        "projected_remaining_5ms_wall_hours": parent[
            "remaining_cost_projection"
        ]["projected_remaining_wall_hours"],
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
                "parent_config": _sha256(h2b1.CONFIG_PATH),
                "parent_summary": _sha256(h2b1.SUMMARY_PATH),
                "parent_decisive_arrays": _sha256(h2b1.DECISIVE_ARRAYS),
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
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Middle 2 ms continuation manifest WP10c9d6c7c3b5c3h2c0",
                "",
                "## Classification",
                "",
                f"`{summary['classification']}`",
                "",
                "The middle nonlinear base, five-profile tangent, and generic nonlinear anchor passed through 1 ms with bitwise replays. This definitions-only package freezes the continuation to 2 ms.",
                "",
                "Targets at 1.0, 1.2, 1.6, and 2.0 ms permit one 0.2 ms landing followed by two 0.4 ms plateau steps. Base step-doubling, sampled anchor checks, tangent JVP audits, and all physical/ledger gates remain unchanged.",
                "",
                "A pass authorizes only a fresh definitions-only 5 ms completion manifest. Fine propagation, fixed-Q experiments, and reduced slow evolution remain blocked.",
                "",
            )
        ),
        encoding="utf-8",
    )
    names = ("config.json", "continuation_manifest.json", "provenance.json", "summary.json")
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
