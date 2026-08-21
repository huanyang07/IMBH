#!/usr/bin/env python3
"""Freeze the transition phase-collocation vector-field replay."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_cold_phase_collocation_wp10c9d6c7c3b5c4f25e7 as cold  # noqa: E402
import run_causal_inner_transition_tube_geometry_wp10c9d6c7c3b5c4f25ds as geometry  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25e8"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25e9"
PARENT_COMMIT = "e6bfd7ca7a90cae1fa9f8886b67708a0a9686149"
PARENT_TREE = "7e2178a54fbb76c5e64078096cbd163f197bad33"
CLASSIFICATION = "transition_phase_collocation_exact_rate_replay_manifest_frozen"

HELDOUT_INDICES = (1, 3, 5, 7, 9, 11, 13, 15)
FINE_SEGMENT_SPECS = (
    (0, 4, (0, 2, 4)),
    (4, 8, (4, 6, 8)),
    (8, 12, (8, 10, 12)),
    (12, 17, (12, 14, 16, 17)),
)
COARSE_SEGMENT_SPECS = (
    (0, 8, (0, 4, 8)),
    (8, 17, (8, 12, 16, 17)),
)
MAXIMUM_EXACT_CONTINUOUS_RATE_CALLS = len(HELDOUT_INDICES)
MAXIMUM_STATE_ERROR_OVER_PATH = 5.0e-3
MAXIMUM_FULL_RATE_RELATIVE_DEFECT = 1.0e-1
MINIMUM_FULL_RATE_DIRECTION_COSINE = 0.995
MAXIMUM_MACRO_RATE_RELATIVE_DEFECT = 1.0e-1
MAXIMUM_HIDDEN_RATE_RELATIVE_DEFECT = 1.0e-1
MAXIMUM_FINE_COARSE_STATE_DEFECT = 2.0e-3
MAXIMUM_FINE_COARSE_RATE_DEFECT = 2.0e-2
MAXIMUM_INTERFACE_VALUE_DEFECT = 5.0e-12
MAXIMUM_CONSTRAINT_CONDITION_NUMBER = 1.0e4
MAXIMUM_AFFINE_EVENT_DEFECT = 5.0e-12

ARTIFACT = "causal_inner_transition_phase_collocation_manifest_wp10c9d6c7c3b5c4f25e8"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = "scripts/run_causal_inner_transition_phase_collocation_manifest_wp10c9d6c7c3b5c4f25e8.py"
THIS_TEST = "tests/test_causal_inner_transition_phase_collocation_manifest_wp10c9d6c7c3b5c4f25e8.py"
EXECUTION_RUNNER = "scripts/run_causal_inner_transition_phase_collocation_wp10c9d6c7c3b5c4f25e9.py"
EXECUTION_TEST = "tests/test_causal_inner_transition_phase_collocation_wp10c9d6c7c3b5c4f25e9.py"
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_TRANSITION_PHASE_COLLOCATION_"
    "MANIFEST_WP10C9D6C7C3B5C4F25E8_2026-08-21.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE


def _helper():
    return cold._helper()


def _accepted_stage_files() -> list[Path]:
    paths = []
    for directory in geometry.manifest._accepted_stage_directories():
        summary = _helper()._read(directory / "summary.json")
        local = int(summary["step_index"])
        paths.extend(
            (
                directory / f"checkpoint_step_{local:02d}.npz",
                directory / f"result_step_{local:02d}.npz",
                directory / "summary.json",
            )
        )
    return paths


def _validate_parent(*, require_clean: bool) -> dict:
    helper = _helper()
    if helper._git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("transition-collocation parent commit changed")
    if helper._git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("transition-collocation parent tree changed")
    cold_hashes = helper._validate_checksums(cold.CANONICAL_DIRECTORY)
    cold_summary = helper._read(cold.CANONICAL_DIRECTORY / "summary.json")
    if (
        not cold_summary["passed"]
        or cold_summary["authorized_next"] != WORK_PACKAGE
        or not cold_summary["transition_collocation_manifest_authorized"]
    ):
        raise RuntimeError("cold phase-collocation authorization changed")
    geometry.manifest._validate_parents(require_clean=False)
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("transition-collocation manifest requires a clean tracked tree")
    return {"cold_result_hashes": cold_hashes}


def _contract() -> dict:
    helper = _helper()
    seed = geometry.manifest.full_step.manifest.SEED_CHECKPOINT
    decisive = {
        "cold_summary": helper._sha(cold.CANONICAL_DIRECTORY / "summary.json"),
        "cold_metrics": helper._sha(cold.CANONICAL_DIRECTORY / "cold_collocation_metrics.json"),
        "transition_geometry": helper._sha(manifest_path := manifest_geometry_path()),
        "transition_seed_checkpoint": helper._sha(seed),
        "affine_parent_summary": helper._sha(cold.manifest.parent.CANONICAL_DIRECTORY / "summary.json"),
        "affine_parent_arrays": helper._sha(cold.manifest.parent.CANONICAL_DIRECTORY / "affine_engine_model_and_replay.npz"),
    }
    for path in _accepted_stage_files():
        decisive[str(path.relative_to(ROOT))] = helper._sha(path)
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "trajectory": {
            "state_count": geometry.manifest.STATE_COUNT,
            "heldout_indices": HELDOUT_INDICES,
            "fine_segment_specs": FINE_SEGMENT_SPECS,
            "coarse_segment_specs": COARSE_SEGMENT_SPECS,
            "accepted_history_only": True,
            "rejected_full_step_06_excluded": True,
        },
        "vector_field_witness": {
            "operator": "exact_continuous_constrained_fixed_Q_rate_at_saved_accepted_primitive_state",
            "coordinate_map": "exact_geometric_470_coordinate_Jacobian_times_scaled_rate560",
            "maximum_calls": MAXIMUM_EXACT_CONTINUOUS_RATE_CALLS,
            "nonlinear_roots": 0,
            "propagated_states": 0,
            "secants_are_not_truth": True,
        },
        "binding_gates": {
            "maximum_state_error_over_path": MAXIMUM_STATE_ERROR_OVER_PATH,
            "maximum_full_rate_relative_defect": MAXIMUM_FULL_RATE_RELATIVE_DEFECT,
            "minimum_full_rate_direction_cosine": MINIMUM_FULL_RATE_DIRECTION_COSINE,
            "maximum_macro_rate_relative_defect": MAXIMUM_MACRO_RATE_RELATIVE_DEFECT,
            "maximum_hidden_rate_relative_defect": MAXIMUM_HIDDEN_RATE_RELATIVE_DEFECT,
            "maximum_fine_coarse_state_defect": MAXIMUM_FINE_COARSE_STATE_DEFECT,
            "maximum_fine_coarse_rate_defect": MAXIMUM_FINE_COARSE_RATE_DEFECT,
            "maximum_interface_value_defect": MAXIMUM_INTERFACE_VALUE_DEFECT,
            "maximum_constraint_condition_number": MAXIMUM_CONSTRAINT_CONDITION_NUMBER,
            "maximum_affine_event_defect": MAXIMUM_AFFINE_EVENT_DEFECT,
            "all_exact_rate_physical_gates_except_memoryless_hidden_fraction": True,
        },
        "decision": {
            "pass_authorizes_only": "definitions_only_bounded_post_transition_phase_window_manifest",
            "post_transition_execution_not_yet_authorized": True,
            "hot_exit_not_claimed": True,
            "predictive_cycle_authorized": False,
        },
        "decisive_input_hashes": decisive,
        "frozen_source_hashes": {
            name: helper._sha(ROOT / name)
            for name in (THIS_RUNNER, THIS_TEST, EXECUTION_RUNNER, EXECUTION_TEST)
        },
    }


def manifest_geometry_path() -> Path:
    return geometry.CANONICAL_DIRECTORY / "geometry_arrays.npz"


def _update_catalog(summary: dict) -> None:
    helper = _helper()
    with cold.manifest.CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": helper._sha(path), "scientific_status": "DEFINITIONS_ONLY"})
    with cold.manifest.CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    catalog = helper._read(cold.manifest.CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": summary["classification"], "passed": True}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": PARENT_COMMIT, "latest_work_package": WORK_PACKAGE})
    helper._write_json(cold.manifest.CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    helper = _helper()
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("transition phase-collocation manifest already exists")
    locked = _validate_parent(require_clean=True)
    contract = _contract()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "transition_collocation_contract.json", contract)
    helper._write_json(CANONICAL_DIRECTORY / "input_lock.json", locked)
    summary = {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "classification": CLASSIFICATION, "passed": True, "definitions_only": True, "exact_continuous_rate_call_budget": MAXIMUM_EXACT_CONTINUOUS_RATE_CALLS, "transition_execution_authorized": True, "post_transition_execution_authorized": False, "predictive_cycle_authorized": False, "authorized_next": AUTHORIZED_NEXT}
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    helper._write_json(CANONICAL_DIRECTORY / "provenance.json", {"runner": THIS_RUNNER, "test": THIS_TEST, "implementation_commit": helper._git("rev-parse", "HEAD"), "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"), "python": sys.version, "numpy": np.__version__, "platform": platform.platform()})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{helper._sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(("# Transition phase-collocation manifest WP10c9d6c7c3b5c4f25e8", "", f"Classification: `{CLASSIFICATION}`.", "", "Eight held-out accepted transition states will receive new exact continuous fixed-Q rate evaluations. Four fine shooting windows and two coarse windows are frozen before those calls. Secants remain geometric diagnostics only.", "", "A pass authorizes only a definitions-only bounded post-transition window manifest; it does not authorize a hot exit or predictive cycle.", "")), encoding="utf-8")
    _update_catalog(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if not args.freeze: parser.error("use --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
