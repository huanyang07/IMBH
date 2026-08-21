#!/usr/bin/env python3
"""Freeze the truth-free hybrid phase engine replay and cost certificate."""

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

import run_causal_inner_hybrid_phase_memory_architecture_selection_wp10c9d6c7c3b5c4f25e1 as architecture  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25e2"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25e3"
PARENT_COMMIT = "668f276edaa42ba074799afc48ea5fd12f8f0976"
PARENT_TREE = "23924039fa65ef6d78d1b548230c5fdcbfef2b37"
CLASSIFICATION = "truth_free_hybrid_phase_engine_replay_and_cost_manifest_frozen"

COLD_TRAIN_INDICES = (0, 2, 4, 5)
COLD_HOLDOUT_INDICES = (1, 3)
ENERGY_CAPTURE_TARGET = 0.99999
MAXIMUM_HIDDEN_EMBEDDING_RANK = 4
MAXIMUM_COLD_HOLDOUT_ERROR_OVER_PATH = 0.02
MAXIMUM_COLD_HOLDOUT_ERROR_OVER_LOCAL_CHORD = 0.05
MAXIMUM_COLD_MACRO_LEDGER_ERROR_OVER_PATH = 0.02
MAXIMUM_EVENT_STATE_JUMP_OVER_COLD_PATH = 0.005
MAXIMUM_TRANSITION_ENDPOINT_ERROR_OVER_PATH = 0.005
MAXIMUM_MACRO_CLOSURE = 5.0e-12
MAXIMUM_EVENT_MACRO_DISCONTINUITY = 5.0e-12
MAXIMUM_PROJECTED_100K_STEP_WALL_SECONDS = 86_400.0
ONLINE_MACROSTEPS = 100_000

ARTIFACT = "causal_inner_hybrid_phase_engine_manifest_wp10c9d6c7c3b5c4f25e2"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = "scripts/run_causal_inner_hybrid_phase_engine_manifest_wp10c9d6c7c3b5c4f25e2.py"
THIS_TEST = "tests/test_causal_inner_hybrid_phase_engine_manifest_wp10c9d6c7c3b5c4f25e2.py"
EXECUTION_RUNNER = "scripts/run_causal_inner_hybrid_phase_engine_wp10c9d6c7c3b5c4f25e3.py"
EXECUTION_TEST = "tests/test_causal_inner_hybrid_phase_engine_wp10c9d6c7c3b5c4f25e3.py"
ENGINE_SOURCE = "src/imri_qpe/layer3_minidisk_1d/hybrid_phase_memory.py"
ENGINE_TEST = "tests/test_hybrid_phase_memory.py"
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_HYBRID_PHASE_ENGINE_MANIFEST_"
    "WP10C9D6C7C3B5C4F25E2_2026-08-21.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _validate_parent(*, require_clean: bool) -> dict:
    helper = architecture.manifest.tube.manifest.geometry
    if helper._git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("hybrid engine parent commit changed")
    if helper._git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("hybrid engine parent tree changed")
    hashes = helper._validate_checksums(architecture.CANONICAL_DIRECTORY)
    summary = helper._read(architecture.CANONICAL_DIRECTORY / "summary.json")
    metrics = helper._read(
        architecture.CANONICAL_DIRECTORY / "architecture_metrics.json"
    )
    if (
        not summary["passed"]
        or summary["classification"] != architecture.PASS_CLASSIFICATION
        or summary["authorized_next"] != WORK_PACKAGE
        or not metrics["checks"]["memoryless_cold_graph_rejected"]
        or metrics["complete_cycle_truth_available"]
    ):
        raise RuntimeError("hybrid engine architecture evidence changed")
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("hybrid engine manifest requires a clean tracked tree")
    return {"architecture_hashes": hashes}


def _contract() -> dict:
    helper = architecture.manifest.tube.manifest.geometry
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "fit": {
            "cold_training_indices": COLD_TRAIN_INDICES,
            "cold_held_out_indices": COLD_HOLDOUT_INDICES,
            "cold_phase": "normalized_physical_time_from_2ms_to_20ms",
            "transition_phase": "validated_rank_adaptive_progress_coordinate",
            "energy_capture_target": ENERGY_CAPTURE_TARGET,
            "maximum_hidden_embedding_rank": MAXIMUM_HIDDEN_EMBEDDING_RANK,
            "cold_to_transition_event": "phase_equals_one_at_20ms",
            "event_macro_reset": "zero",
        },
        "engine": {
            "online_state": "q_R82_plus_s_scalar_plus_mode_discrete",
            "macro_update": "q_new=q_old+ell_m(s_new)-ell_m(s_old)+event_reset",
            "decoder": "L*q+Z*(h0+U*a_m(s))",
            "event_integrator": "exact_event_crossing_with_piecewise_constant_phase_speed",
            "restart": "lossless_JSON_float64_roundtrip",
            "online_truth_calls": 0,
            "online_470_roots": 0,
            "online_fixed_Q_microsteps": 0,
        },
        "binding_gates": {
            "cold_hidden_embedding_rank_max": MAXIMUM_HIDDEN_EMBEDDING_RANK,
            "cold_training_energy_capture_min": ENERGY_CAPTURE_TARGET,
            "cold_holdout_error_over_path_max": MAXIMUM_COLD_HOLDOUT_ERROR_OVER_PATH,
            "cold_holdout_error_over_local_chord_max": MAXIMUM_COLD_HOLDOUT_ERROR_OVER_LOCAL_CHORD,
            "cold_macro_ledger_error_over_path_max": MAXIMUM_COLD_MACRO_LEDGER_ERROR_OVER_PATH,
            "event_state_jump_over_cold_path_max": MAXIMUM_EVENT_STATE_JUMP_OVER_COLD_PATH,
            "transition_endpoint_error_over_path_max": MAXIMUM_TRANSITION_ENDPOINT_ERROR_OVER_PATH,
            "macro_decoder_closure_max": MAXIMUM_MACRO_CLOSURE,
            "event_macro_discontinuity_max": MAXIMUM_EVENT_MACRO_DISCONTINUITY,
            "restart_bitwise": True,
            "single_call_equals_staged_advance_bitwise": True,
            "projected_100k_step_wall_seconds_max": MAXIMUM_PROJECTED_100K_STEP_WALL_SECONDS,
        },
        "scope": {
            "cold_observed_segment_only": True,
            "transition_observed_segment_only": True,
            "hot_exit_missing": True,
            "complete_impulse_missing": True,
            "remaining_cycle_modes_missing": True,
            "predictive_cycle_authorized": False,
        },
        "input_hashes": {
            "architecture_summary": helper._sha(
                architecture.CANONICAL_DIRECTORY / "summary.json"
            ),
            "architecture_metrics": helper._sha(
                architecture.CANONICAL_DIRECTORY / "architecture_metrics.json"
            ),
            "architecture_arrays": helper._sha(
                architecture.CANONICAL_DIRECTORY / "architecture_arrays.npz"
            ),
        },
        "frozen_source_hashes": {
            name: helper._sha(ROOT / name)
            for name in (
                THIS_RUNNER,
                THIS_TEST,
                EXECUTION_RUNNER,
                EXECUTION_TEST,
                ENGINE_SOURCE,
                ENGINE_TEST,
            )
        },
    }


def _update_catalog(summary: dict) -> None:
    helper = architecture.manifest.tube.manifest.geometry
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
                    "sha256": helper._sha(path),
                    "scientific_status": "DEFINITIONS_ONLY",
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case", "path", "bytes", "sha256", "scientific_status"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = helper._read(CANONICAL_SUMMARY)
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
            "latest_source_parent_commit": PARENT_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    helper._write_json(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    helper = architecture.manifest.tube.manifest.geometry
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("hybrid engine manifest already exists")
    inputs = _validate_parent(require_clean=True)
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "engine_contract.json", _contract())
    helper._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {"parent_commit": PARENT_COMMIT, "parent_tree": PARENT_TREE, **inputs},
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "new_truth_calls": 0,
        "truth_free_engine_execution_authorized": True,
        "predictive_cycle_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    helper._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "engine_source": ENGINE_SOURCE,
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "implementation_commit": helper._git("rev-parse", "HEAD"),
            "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"),
        },
    )
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{helper._sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Hybrid phase engine manifest WP10c9d6c7c3b5c4f25e2",
                "",
                f"Classification: `{CLASSIFICATION}`.",
                "",
                "This freezes a truth-free event-driven engine for the observed cold and fixed-Q transition segments. It binds held-out reconstruction, exact macro closure, event continuity, restart replay, staged/single-call equality, and projected 100,000-step cost.",
                "",
                "A pass demonstrates a working offline/online numerical architecture on observed modes. It cannot authorize an astrophysical cycle until the hot exit, impulse, and remaining cycle modes are calibrated and held out.",
                "",
            )
        ),
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
