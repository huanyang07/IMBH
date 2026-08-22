#!/usr/bin/env python3
"""Freeze the first moving exact weighted-arclength truth segment."""

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

import run_causal_inner_arclength_transport_preflight_wp10c9d6c7c3b5c4f25f3 as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25f4"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25f5"
CLASSIFICATION = "moving_exact_arclength_segment_manifest_frozen"
ARTIFACT = (
    "causal_inner_arclength_segment_manifest_"
    "wp10c9d6c7c3b5c4f25f4"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_arclength_segment_manifest_"
    "wp10c9d6c7c3b5c4f25f4.py"
)
THIS_TEST = (
    "tests/test_causal_inner_arclength_segment_manifest_"
    "wp10c9d6c7c3b5c4f25f4.py"
)
EXECUTION_RUNNER = (
    "scripts/run_causal_inner_arclength_segment_"
    "wp10c9d6c7c3b5c4f25f5.py"
)
EXECUTION_TEST = (
    "tests/test_causal_inner_arclength_segment_"
    "wp10c9d6c7c3b5c4f25f5.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ARCLENGTH_SEGMENT_MANIFEST_"
    "WP10C9D6C7C3B5C4F25F4_2026-08-22.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE

NODE_COUNT = 5
ARCLENGTH_SPAN = 2.5e-2
MAXIMUM_UNIQUE_RATE_STATES = 2 * NODE_COUNT - 1
MAXIMUM_PROJECTED_COLLOCATION_DEFECT = 2.5e-2
MAXIMUM_FULL_COLLOCATION_DEFECT = 2.5e-2
MAXIMUM_NORMAL_RATE_DEFECT = 1.0e-2
MINIMUM_RATE_DIRECTION_COSINE = 0.9995
MAXIMUM_TIME_MAPPING_DEFECT = 2.5e-2
MINIMUM_PHASE_SPEED_RATIO = 0.5
MAXIMUM_Q3_RELATIVE_DRIFT = 5.0e-4
MINIMUM_RECONSTRUCTION_FACTOR = 1.0 - 1.0e-12
GROW_MAXIMUM_FULL_COLLOCATION_DEFECT = 5.0e-3
GROW_MAXIMUM_TIME_MAPPING_DEFECT = 5.0e-3
GROW_MAXIMUM_ANCHOR_DEPARTURE = 2.0e-2


def _helper():
    return parent._helper()


def _window_05_directory() -> Path:
    return parent.manifest.parent._window_directories()[-1]


def _decisive_inputs() -> dict[str, Path]:
    return {
        "transport_summary": parent.CANONICAL_DIRECTORY / "summary.json",
        "transport_metrics": parent.CANONICAL_DIRECTORY / "transport_metrics.json",
        "transport_arrays": parent.CANONICAL_DIRECTORY / "transport_arrays.npz",
        "transport_manifest_summary": parent.manifest.CANONICAL_DIRECTORY / "summary.json",
        "diagnosis_summary": parent.manifest.parent.CANONICAL_DIRECTORY / "summary.json",
        "diagnosis_metrics": parent.manifest.parent.CANONICAL_DIRECTORY / "arclength_event_metrics.json",
        "diagnosis_arrays": parent.manifest.parent.CANONICAL_DIRECTORY / "arclength_event_arrays.npz",
        "window_05_summary": _window_05_directory() / "summary.json",
        "window_05_metrics": _window_05_directory() / "phase_window_metrics.json",
        "window_05_arrays": _window_05_directory() / "phase_window_arrays.npz",
        "window_05_checkpoint": _window_05_directory() / "phase_window_checkpoint.npz",
    }


def _validate_parent(*, require_clean: bool) -> dict:
    helper = _helper()
    transport_hashes = helper._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = helper._read(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = helper._read(parent.CANONICAL_DIRECTORY / "transport_metrics.json")
    if (
        not summary["passed"]
        or not summary["Window_05_targets_replayed"]
        or not summary["arclength_execution_manifest_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or not metrics["passed"]
        or int(metrics["new_exact_fixed_Q_rate_calls"]) != 0
    ):
        raise RuntimeError("arclength transport preflight changed")
    window_hashes = helper._validate_checksums(_window_05_directory())
    window_summary = helper._read(_window_05_directory() / "summary.json")
    if not window_summary["passed"] or int(window_summary["window_index"]) != 5:
        raise RuntimeError("arclength segment seed changed")
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("arclength segment manifest requires a clean tracked tree")
    return {"transport_hashes": transport_hashes, "window_05_hashes": window_hashes}


def _contract(parent_lock: dict) -> dict:
    helper = _helper()
    source = parent.manifest.parent.source
    sources = (
        THIS_RUNNER,
        THIS_TEST,
        EXECUTION_RUNNER,
        EXECUTION_TEST,
        parent.manifest.ARCLENGTH_SOURCE,
        parent.THIS_RUNNER,
        parent.manifest.THIS_RUNNER,
        source.THIS_RUNNER,
        source.manifest.EXACT_CHART_SOURCE,
        source.manifest.rejected.manifest.original.FIXED_Q_SOURCE,
        source.manifest.rejected.manifest.original.EXACT_RATE_SOURCE,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "seed": {
            "accepted_window_index": 5,
            "seed_directory": str(_window_05_directory().relative_to(ROOT)),
            "last_accepted_state_only": True,
            "fixed_time_window_06_is_not_executed": True,
        },
        "phase_system": {
            "nu": "||f_Q(y,t)||_2",
            "dy_ds": "f_Q(y,t)/nu",
            "dt_ds": "1/nu",
            "arclength_span": ARCLENGTH_SPAN,
            "node_count": NODE_COUNT,
            "maximum_unique_exact_rate_states": MAXIMUM_UNIQUE_RATE_STATES,
            "one_projected_Picard_update": True,
        },
        "exact_chart": {
            "moving_anchor_is_Window_05_endpoint": True,
            "one_anchor_augmented_Jacobian": True,
            "target_local_Broyden_transport": True,
            "maximum_target_refreshes": parent.manifest.MAXIMUM_TOTAL_TARGET_EXACT_REFRESHES,
            "coordinate_tolerance": source.manifest.COORDINATE_TOLERANCE,
            "gauge_tolerance": source.manifest.GAUGE_TOLERANCE,
            "maximum_condition_number": source.manifest.MAXIMUM_AUGMENTED_CONDITION_NUMBER,
            "maximum_scaled_anchor_departure": source.manifest.MAXIMUM_SCALED_ANCHOR_DEPARTURE,
        },
        "binding_gates": {
            "maximum_training_normal_rate_defect": source.manifest.MAXIMUM_TRAINING_NORMAL_RATE_DEFECT,
            "maximum_projected_collocation_defect": MAXIMUM_PROJECTED_COLLOCATION_DEFECT,
            "maximum_full_collocation_defect": MAXIMUM_FULL_COLLOCATION_DEFECT,
            "maximum_normal_rate_defect": MAXIMUM_NORMAL_RATE_DEFECT,
            "minimum_rate_direction_cosine": MINIMUM_RATE_DIRECTION_COSINE,
            "maximum_time_mapping_defect": MAXIMUM_TIME_MAPPING_DEFECT,
            "minimum_phase_speed_ratio": MINIMUM_PHASE_SPEED_RATIO,
            "maximum_Q3_relative_drift": MAXIMUM_Q3_RELATIVE_DRIFT,
            "minimum_reconstruction_factor": MINIMUM_RECONSTRUCTION_FACTOR,
            "all_existing_exact_rate_physical_gates": True,
            "failed_segment_never_propagates": True,
        },
        "regime_classification": {
            "legacy_exit_remains_a_candidate_surface": True,
            "fast_equilibrium_candidate_requires_speed_ratio_at_most": 1.0e-3,
            "recurrence_candidate_distance_over_local_span_at_most": 0.1,
            "recurrence_candidate_direction_cosine_at_least": 0.99,
            "candidate_persistence_segments": 2,
            "candidate_requires_separate_refinement_before_mode_transition": True,
        },
        "cost_boundary": {
            "new_nonlinear_fixed_Q_roots_equal": 0,
            "new_BDF_microsteps_equal": 0,
            "truth_calls_are_offline_only": True,
            "online_truth_calls_equal": 0,
        },
        "parent_lock": parent_lock,
        "decisive_input_hashes": {
            name: helper._sha(path) for name, path in _decisive_inputs().items()
        },
        "frozen_source_hashes": {
            relative: helper._sha(ROOT / relative) for relative in sources
        },
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }


def _update_catalog(summary: dict) -> None:
    helper = _helper()
    cold = parent.manifest.parent.source._post().manifest.transition.manifest.cold.manifest
    with cold.CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({
                "case": ARTIFACT,
                "path": str(path.relative_to(ROOT)),
                "bytes": str(path.stat().st_size),
                "sha256": helper._sha(path),
                "scientific_status": "DEFINITIONS_ONLY",
            })
    with cold.CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case", "path", "bytes", "sha256", "scientific_status"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = helper._read(cold.CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": True,
    }
    catalog.update({
        "case_count": len({row["case"] for row in rows}),
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "all_payload_hashes_recorded": True,
        "latest_source_parent_commit": helper._git("rev-parse", "HEAD"),
        "latest_work_package": WORK_PACKAGE,
    })
    helper._write_json(cold.CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    helper = _helper()
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("arclength segment manifest already exists")
    parent_lock = _validate_parent(require_clean=True)
    contract = _contract(parent_lock)
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "arclength_segment_contract.json", contract)
    helper._write_json(CANONICAL_DIRECTORY / "input_lock.json", parent_lock)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "arclength_segment_execution_authorized": True,
        "fixed_time_window_06_authorized": False,
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    helper._write_json(CANONICAL_DIRECTORY / "provenance.json", {
        "runner": THIS_RUNNER,
        "test": THIS_TEST,
        "implementation_commit": helper._git("rev-parse", "HEAD"),
        "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"),
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
    })
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{helper._sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join((
            "# First moving exact arclength segment manifest",
            "",
            f"Classification: `{CLASSIFICATION}`.",
            "",
            "One five-node segment of nondimensional coordinate arclength 0.025 is authorized from the accepted Window-5 endpoint. Physical time is an integrated dependent variable; the fixed-time Window-6 contract is superseded.",
            "",
            "The segment retains exact retraction, every fixed-Q physical gate, fail-closed propagation, and zero nonlinear roots or BDF microsteps. A detected exit, equilibrium, or recurrence is only a candidate until separately refined.",
            "",
        )),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if not args.freeze:
        parser.error("use --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
