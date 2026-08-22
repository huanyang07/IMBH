#!/usr/bin/env python3
"""Freeze the truth-free conservative hot-mode engine replay."""

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

import run_causal_inner_hot_mode_off_axis_preflight_wp10c9d6c7c3b5c4f25fa as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25fb"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25fc"
CLASSIFICATION = "truth_free_conservative_hot_mode_engine_manifest_frozen"
ARTIFACT = "causal_inner_truth_free_hot_mode_engine_manifest_wp10c9d6c7c3b5c4f25fb"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_TRUTH_FREE_HOT_MODE_ENGINE_"
    "MANIFEST_WP10C9D6C7C3B5C4F25FB_2026-08-22.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
EXECUTION_RUNNER = (
    "scripts/run_causal_inner_truth_free_hot_mode_engine_"
    "wp10c9d6c7c3b5c4f25fc.py"
)
EXECUTION_TEST = (
    "tests/test_causal_inner_truth_free_hot_mode_engine_"
    "wp10c9d6c7c3b5c4f25fc.py"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_truth_free_hot_mode_engine_manifest_"
    "wp10c9d6c7c3b5c4f25fb.py"
)
THIS_TEST = (
    "tests/test_causal_inner_truth_free_hot_mode_engine_manifest_"
    "wp10c9d6c7c3b5c4f25fb.py"
)

MACRO_STEP_SECONDS = parent.manifest.PHYSICAL_MACRO_STEP_SECONDS
HOT_CENTER_INDEX = parent.manifest.HOT_CENTER_INDEX
MAXIMUM_ABSOLUTE_PATCH_COORDINATE = 1.25
MAXIMUM_EMBEDDED_ERROR_FRACTION = 5.0e-2
MAXIMUM_ENDPOINT_INCREMENT_DEFECT = 1.0e-2
MAXIMUM_MACRO_INCREMENT_DEFECT = 1.0e-4
MAXIMUM_MACRO_LEDGER_DEFECT = 5.0e-13
MAXIMUM_DECODER_MACRO_CLOSURE = 5.0e-12
MAXIMUM_ANCHOR_HIDDEN_RATE_PROJECTION_DEFECT = 5.0e-2
OVERSIZE_REJECTION_FACTOR = 2.0
BENCHMARK_STEPS = 100_000
MAXIMUM_BENCHMARK_WALL_SECONDS = 60.0
REFERENCE_PHASE_PERIOD_SECONDS = 2.0e-2
MODE_SWITCH_MARGIN = 0.1
MODE_SWITCH_PERSISTENCE = 2


def _helper():
    return parent._helper()


def _decisive_inputs() -> dict[str, Path]:
    return {
        "off_axis_summary": parent.CANONICAL_DIRECTORY / "summary.json",
        "off_axis_metrics": parent.CANONICAL_DIRECTORY
        / "hot_mode_off_axis_metrics.json",
        "off_axis_arrays": parent.CANONICAL_DIRECTORY
        / "hot_mode_off_axis_arrays.npz",
        "hot_axis_arrays": parent.hot.CANONICAL_DIRECTORY
        / "hot_free_field_arrays.npz",
    }


def _source_paths() -> tuple[Path, ...]:
    return (
        ROOT / THIS_RUNNER,
        ROOT / THIS_TEST,
        ROOT / EXECUTION_RUNNER,
        ROOT / EXECUTION_TEST,
        ROOT / "src/imri_qpe/layer3_minidisk_1d/conservative_free_field_rom.py",
        ROOT / parent.THIS_RUNNER,
    )


def _validate_parent(*, require_clean: bool) -> dict:
    helper = _helper()
    hashes = helper._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = helper._read(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = helper._read(
        parent.CANONICAL_DIRECTORY / "hot_mode_off_axis_metrics.json"
    )
    if (
        not summary["passed"]
        or summary["classification"] != parent.PASS_CLASSIFICATION
        or not summary["truth_free_hot_mode_engine_manifest_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or not all(metrics["gates"].values())
        or metrics["gate_values"]["selected_hidden_rate_rank"] != 2
    ):
        raise RuntimeError("off-axis hot-mode certificate changed")
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("truth-free hot engine manifest requires a clean tracked tree")
    return {
        "off_axis_hashes": hashes,
        "parent_classification": summary["classification"],
    }


def _contract(locked: dict) -> dict:
    helper = _helper()
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "parent_lock": locked,
        "decisive_input_hashes": {
            name: helper._sha(path) for name, path in _decisive_inputs().items()
        },
        "frozen_source_hashes": {
            str(path.relative_to(ROOT)): helper._sha(path) for path in _source_paths()
        },
        "engine": {
            "state": "q82_plus_hot_hidden_amplitudes2_plus_phase_plus_mode_plus_anchor_id",
            "decoder": "L*q+Z*(h_anchor+V_hot*a)",
            "operator": "anchor_reduced_free_rate_plus_eta_times_physical_rate_delta",
            "integrator": "explicit_Heun_with_Euler_embedded_error",
            "macro_step_seconds": MACRO_STEP_SECONDS,
            "patch_trust_coordinate_maximum": MAXIMUM_ABSOLUTE_PATCH_COORDINATE,
            "mode_switch": "nearest_atlas_with_margin_and_two_step_persistence",
            "macro_ledger": "all_82_coordinates_retained_without_projection",
            "forcing_phase_period_for_replay_seconds": REFERENCE_PHASE_PERIOD_SECONDS,
        },
        "binding_replay": {
            "one_hot_macro_step": True,
            "compare_to_certified_full_coordinate_Heun_endpoint": True,
            "oversize_step_must_reject_without_propagation": True,
            "checkpoint_and_replay_bitwise": True,
            "benchmark_update_plus_full_decode_steps": BENCHMARK_STEPS,
        },
        "gates": {
            "maximum_embedded_error_fraction": MAXIMUM_EMBEDDED_ERROR_FRACTION,
            "maximum_endpoint_increment_defect": MAXIMUM_ENDPOINT_INCREMENT_DEFECT,
            "maximum_macro_increment_defect": MAXIMUM_MACRO_INCREMENT_DEFECT,
            "maximum_macro_ledger_defect": MAXIMUM_MACRO_LEDGER_DEFECT,
            "maximum_decoder_macro_closure": MAXIMUM_DECODER_MACRO_CLOSURE,
            "maximum_anchor_hidden_rate_projection_defect": MAXIMUM_ANCHOR_HIDDEN_RATE_PROJECTION_DEFECT,
            "maximum_benchmark_wall_seconds": MAXIMUM_BENCHMARK_WALL_SECONDS,
            "restart_bitwise": True,
            "oversize_rejection_factor": OVERSIZE_REJECTION_FACTOR,
        },
        "online_forbidden": {
            "truth_calls": 0,
            "fixed_Q_reaction_calls": 0,
            "coordinate_retractions": 0,
            "nonlinear_roots": 0,
            "BDF_microsteps": 0,
        },
        "fixed_Q_physical_phase_authorized": False,
        "complete_cycle_execution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }


def _update_catalog(summary: dict) -> None:
    helper = _helper()
    cold = parent.arclength._source()._post().manifest.transition.manifest.cold.manifest
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
        writer = csv.DictWriter(handle, fieldnames=(
            "case", "path", "bytes", "sha256", "scientific_status"
        ), lineterminator="\n")
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
        raise RuntimeError("truth-free hot engine manifest already exists")
    locked = _validate_parent(require_clean=True)
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "hot_engine_contract.json", _contract(locked))
    helper._write_json(CANONICAL_DIRECTORY / "input_lock.json", locked)
    helper._write_json(CANONICAL_DIRECTORY / "provenance.json", {
        "runner": THIS_RUNNER,
        "test": THIS_TEST,
        "implementation_commit": helper._git("rev-parse", "HEAD"),
        "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"),
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
    })
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "truth_free_hot_mode_engine_replay_authorized": True,
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{helper._sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join((
            "# Truth-free conservative hot-mode engine manifest",
            "",
            f"Classification: `{CLASSIFICATION}`.",
            "",
            "Replay one 0.25 ms hot macro step with q82 plus two hidden amplitudes, explicit Heun, exact macro bookkeeping, local-patch trust, deterministic restart, and hysteretic mode selection.",
            "",
            "The online replay permits no truth call, fixed-Q reaction, coordinate retraction, nonlinear root, or BDF microstep. Complete-cycle execution remains unauthorized.",
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
