#!/usr/bin/env python3
"""Freeze the affine cold-to-transition chart-gluing correction."""

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

import run_causal_inner_hybrid_phase_engine_wp10c9d6c7c3b5c4f25e3 as rejected  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25e4"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25e5"
PARENT_COMMIT = "8618f9f82dc6e7ccf504c0bec578c662afbd7a91"
PARENT_TREE = "ede3c7c5ecd0bc8869fa593d97fcbaaec77d3c21"
CLASSIFICATION = "affine_cold_to_transition_phase_chart_gluing_manifest_frozen"

LOCAL_SEED_ZERO_GATE = 1.0e-14
AFFINE_ENTRY_CLOSURE_GATE = 5.0e-12

ARTIFACT = "causal_inner_affine_phase_chart_gluing_manifest_wp10c9d6c7c3b5c4f25e4"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = "scripts/run_causal_inner_affine_phase_chart_gluing_manifest_wp10c9d6c7c3b5c4f25e4.py"
THIS_TEST = "tests/test_causal_inner_affine_phase_chart_gluing_manifest_wp10c9d6c7c3b5c4f25e4.py"
EXECUTION_RUNNER = "scripts/run_causal_inner_affine_phase_chart_gluing_wp10c9d6c7c3b5c4f25e5.py"
EXECUTION_TEST = "tests/test_causal_inner_affine_phase_chart_gluing_wp10c9d6c7c3b5c4f25e5.py"
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_AFFINE_PHASE_CHART_GLUING_"
    "MANIFEST_WP10C9D6C7C3B5C4F25E4_2026-08-21.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _validate_parent(*, require_clean: bool) -> dict:
    helper = rejected.manifest.architecture.manifest.tube.manifest.geometry
    if helper._git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("affine chart-gluing parent commit changed")
    if helper._git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("affine chart-gluing parent tree changed")
    hashes = helper._validate_checksums(rejected.CANONICAL_DIRECTORY)
    summary = helper._read(rejected.CANONICAL_DIRECTORY / "summary.json")
    metrics = helper._read(rejected.CANONICAL_DIRECTORY / "engine_metrics.json")
    failed = [name for name, passed in metrics["gates"].items() if not passed]
    if (
        summary["passed"]
        or summary["classification"] != rejected.FAIL_CLASSIFICATION
        or failed != ["event_state"]
        or not summary["online_cost_feasible"]
    ):
        raise RuntimeError("unglued phase-engine rejection changed")
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("affine gluing manifest requires a clean tracked tree")
    return {"rejected_engine_hashes": hashes}


def _contract() -> dict:
    helper = rejected.manifest.architecture.manifest.tube.manifest.geometry
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "diagnosis": {
            "cold_coordinates": "absolute_y470",
            "transition_coordinates": "local_delta_y470_with_seed_zero",
            "rejected_decoder": "L*q_absolute+Z*h_transition_local",
            "omitted_term": "Z*h_entry_absolute",
            "only_failed_parent_gate": "event_state",
        },
        "corrected_gluing": {
            "cold_decoder": "D_C(q,s)=L*q+Z*h_C(s)",
            "transition_decoder": (
                "D_T(q,s)=L*q+Z*(h_entry_absolute+h_T_local(s))"
            ),
            "transition_truth_alignment": "y_T_absolute=y_entry+y_T_local",
            "macro_ledger": "q_T=q_entry+ell_T_local(s)",
            "event_reset": "q_plus=q_minus_and_s_plus=0",
            "no_fitted_event_jump": True,
        },
        "binding_gates": {
            "transition_local_seed_norm_max": LOCAL_SEED_ZERO_GATE,
            "absolute_entry_decoder_closure_max": AFFINE_ENTRY_CLOSURE_GATE,
            "event_state_continuity_max": AFFINE_ENTRY_CLOSURE_GATE,
            "event_macro_continuity_max": rejected.manifest.MAXIMUM_EVENT_MACRO_DISCONTINUITY,
            "all_non_event_parent_gates_unchanged_and_pass": True,
            "projected_100k_step_wall_seconds_max": rejected.manifest.MAXIMUM_PROJECTED_100K_STEP_WALL_SECONDS,
        },
        "scope": {
            "coordinate_origin_repair_only": True,
            "no_gate_relaxed": True,
            "new_truth_calls": 0,
            "complete_cycle_calibration_missing": True,
            "predictive_cycle_authorized": False,
        },
        "input_hashes": {
            "rejected_summary": helper._sha(rejected.CANONICAL_DIRECTORY / "summary.json"),
            "rejected_metrics": helper._sha(
                rejected.CANONICAL_DIRECTORY / "engine_metrics.json"
            ),
            "rejected_arrays": helper._sha(
                rejected.CANONICAL_DIRECTORY / "engine_model_and_replay.npz"
            ),
        },
        "frozen_source_hashes": {
            name: helper._sha(ROOT / name)
            for name in (THIS_RUNNER, THIS_TEST, EXECUTION_RUNNER, EXECUTION_TEST)
        },
    }


def _update_catalog(summary: dict) -> None:
    helper = rejected.manifest.architecture.manifest.tube.manifest.geometry
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
    helper = rejected.manifest.architecture.manifest.tube.manifest.geometry
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("affine phase-gluing manifest already exists")
    inputs = _validate_parent(require_clean=True)
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "affine_gluing_contract.json", _contract())
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
        "parent_rejection_preserved": True,
        "affine_chart_gluing_execution_authorized": True,
        "predictive_cycle_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    helper._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "implementation_commit": helper._git("rev-parse", "HEAD"),
            "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"),
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
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
                "# Affine phase-chart gluing manifest WP10c9d6c7c3b5c4f25e4",
                "",
                f"Classification: `{CLASSIFICATION}`.",
                "",
                "The first online engine failed only because it combined an absolute cold macro state with a local transition hidden coordinate. This package freezes the required affine chart origin. No tolerance is weakened and no fitted event jump is introduced.",
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
