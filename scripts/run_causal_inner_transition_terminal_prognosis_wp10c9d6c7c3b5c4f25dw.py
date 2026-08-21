#!/usr/bin/env python3
"""Classify the terminal transition trend and select the next architecture."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_transition_terminal_prognosis_manifest_wp10c9d6c7c3b5c4f25dv as manifest  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25dw"
BRANCH_CLASSIFICATION = (
    "continued_hot_exit_microstepping_not_cost_justified_"
    "cold_branch_pseudo_arclength_architecture_selected"
)
EXTENSION_CLASSIFICATION = "bounded_hot_exit_extension_supported_within_frozen_budget"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25dx"

ARTIFACT = "causal_inner_transition_terminal_prognosis_wp10c9d6c7c3b5c4f25dw"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = "scripts/run_causal_inner_transition_terminal_prognosis_wp10c9d6c7c3b5c4f25dw.py"
THIS_TEST = "tests/test_causal_inner_transition_terminal_prognosis_wp10c9d6c7c3b5c4f25dw.py"
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_TRANSITION_TERMINAL_PROGNOSIS_"
    "WP10C9D6C7C3B5C4F25DW_2026-08-21.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
HALF_STEP_PREFIX = (
    "causal_inner_hot_exit_half_step_recovery_wp10c9d6c7c3b5c4f25dq"
)


def _validate_lock(*, require_clean: bool) -> dict:
    hashes = manifest.tube.manifest.geometry._validate_checksums(manifest.CANONICAL_DIRECTORY)
    contract = manifest.tube.manifest.geometry._read(
        manifest.CANONICAL_DIRECTORY / "prognosis_contract.json"
    )
    summary = manifest.tube.manifest.geometry._read(
        manifest.CANONICAL_DIRECTORY / "summary.json"
    )
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["work_package"] != manifest.WORK_PACKAGE
    ):
        raise RuntimeError("terminal prognosis manifest classification changed")
    for relative, expected in contract["frozen_source_hashes"].items():
        if manifest.tube.manifest.geometry._sha(ROOT / relative) != expected:
            raise RuntimeError(f"frozen terminal prognosis source changed: {relative}")
    manifest._validate_parent(require_clean=False)
    if require_clean and manifest.tube.manifest.geometry._git(
        "status", "--short", "--untracked-files=no"
    ):
        raise RuntimeError("terminal prognosis execution requires a clean tracked tree")
    return {"manifest_hashes": hashes, "contract": contract}


def _prognosis(hidden_fractions: np.ndarray, root_wall_seconds: np.ndarray) -> dict:
    tail = np.asarray(hidden_fractions[-manifest.TAIL_INTERVALS :], dtype=float)
    tail_differences = np.diff(tail)
    every_tail_fraction_decreases = bool(np.all(tail_differences < 0.0))
    slope_per_root = float(np.polyfit(np.arange(len(tail), dtype=float), tail, 1)[0])
    crossing_roots = None
    if slope_per_root < 0.0 and tail[-1] > manifest.HIDDEN_EXIT_FRACTION:
        crossing_roots = float(
            (tail[-1] - manifest.HIDDEN_EXIT_FRACTION) / (-slope_per_root)
        )
    elif tail[-1] <= manifest.HIDDEN_EXIT_FRACTION:
        crossing_roots = 0.0
    median_wall_seconds = float(np.median(root_wall_seconds))
    projected_wall_hours = (
        None
        if crossing_roots is None
        else float(math.ceil(crossing_roots) * median_wall_seconds / 3600.0)
    )
    crossing_within_root_budget = bool(
        crossing_roots is not None
        and crossing_roots <= manifest.MAXIMUM_ADDITIONAL_HALF_STEP_ROOTS
    )
    crossing_within_wall_budget = bool(
        projected_wall_hours is not None
        and projected_wall_hours <= manifest.MAXIMUM_EXTENSION_WALL_HOURS
    )
    extension_warranted = bool(
        every_tail_fraction_decreases
        and crossing_within_root_budget
        and crossing_within_wall_budget
    )
    return {
        "tail_hidden_fractions": tail,
        "tail_fraction_differences": tail_differences,
        "every_tail_fraction_decreases": every_tail_fraction_decreases,
        "linear_tail_slope_per_root": slope_per_root,
        "forecast_roots_to_hidden_exit": crossing_roots,
        "median_half_step_root_wall_seconds": median_wall_seconds,
        "forecast_wall_hours_to_hidden_exit": projected_wall_hours,
        "crossing_within_root_budget": crossing_within_root_budget,
        "crossing_within_wall_budget": crossing_within_wall_budget,
        "bounded_extension_warranted": extension_warranted,
    }


def _half_step_wall_times() -> np.ndarray:
    values = []
    for index in range(1, 13):
        directory = ROOT / "results/canonical" / f"{HALF_STEP_PREFIX}_step_{index:02d}"
        metrics = manifest.tube.manifest.geometry._read(
            directory / f"metrics_step_{index:02d}.json"
        )
        values.append(float(metrics["root_wall_seconds"]))
    return np.asarray(values)


def _update_catalog(summary: dict) -> None:
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
                    "sha256": manifest.tube.manifest.geometry._sha(path),
                    "scientific_status": "SUPPORTED",
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
    catalog = manifest.tube.manifest.geometry._read(CANONICAL_SUMMARY)
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
            "latest_source_parent_commit": manifest.PARENT_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    manifest.tube.manifest.geometry._write_json(CANONICAL_SUMMARY, catalog)


def _execute() -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("terminal prognosis result already exists")
    lock = _validate_lock(require_clean=True)
    geometry_arrays = manifest.tube.manifest.geometry._load_npz(
        manifest.tube.manifest.geometry.CANONICAL_DIRECTORY / "geometry_arrays.npz"
    )
    prognosis = _prognosis(
        np.asarray(geometry_arrays["hidden_secant_fractions"], dtype=float),
        _half_step_wall_times(),
    )
    cold_metrics = manifest.tube.manifest.geometry._read(
        manifest.COLD_SCREEN_DIRECTORY / "branch_candidate_screen_metrics.json"
    )
    cold_candidate_supported = cold_metrics["selected_cold_candidate"] == "full_model_12ms"
    hot_candidate_supported = cold_metrics["selected_hot_candidate"] is not None
    branch_architecture_selected = bool(
        not prognosis["bounded_extension_warranted"]
        and cold_candidate_supported
        and not hot_candidate_supported
    )
    classification = (
        BRANCH_CLASSIFICATION if branch_architecture_selected else EXTENSION_CLASSIFICATION
    )
    metrics = {
        "classification": classification,
        "passed": True,
        "prognosis": prognosis,
        "cold_candidate_supported": cold_candidate_supported,
        "selected_cold_candidate": cold_metrics["selected_cold_candidate"],
        "hot_candidate_supported": hot_candidate_supported,
        "selected_hot_candidate": cold_metrics["selected_hot_candidate"],
        "branch_pseudo_arclength_architecture_selected": branch_architecture_selected,
        "additional_hot_exit_microsteps_authorized": prognosis[
            "bounded_extension_warranted"
        ],
        "online_cycle_seconds": manifest.ONLINE_CYCLE_SECONDS,
        "maximum_online_macrosteps": manifest.MAXIMUM_ONLINE_MACROSTEPS,
        "minimum_average_macrostep_seconds": manifest.MINIMUM_AVERAGE_MACROSTEP_SECONDS,
        "new_truth_calls": 0,
        "hot_branch_truth_established": False,
        "complete_impulse_fit_authorized": False,
        "reduced_cycle_authorized": False,
    }
    arrays = {
        "tail_hidden_fractions": np.asarray(prognosis["tail_hidden_fractions"]),
        "tail_fraction_differences": np.asarray(
            prognosis["tail_fraction_differences"]
        ),
        "half_step_root_wall_seconds": _half_step_wall_times(),
    }
    CANONICAL_DIRECTORY.mkdir(parents=True)
    manifest.tube.manifest.geometry._write_json(
        CANONICAL_DIRECTORY / "prognosis_metrics.json", metrics
    )
    np.savez(CANONICAL_DIRECTORY / "prognosis_arrays.npz", **arrays)
    manifest.tube.manifest.geometry._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "manifest_directory": str(manifest.CANONICAL_DIRECTORY.relative_to(ROOT)),
            "manifest_hashes": lock["manifest_hashes"],
            "input_hashes": lock["contract"]["input_hashes"],
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": True,
        "additional_hot_exit_microsteps_authorized": metrics[
            "additional_hot_exit_microsteps_authorized"
        ],
        "branch_pseudo_arclength_architecture_selected": branch_architecture_selected,
        "cold_branch_root_manifest_authorized": branch_architecture_selected,
        "hot_branch_truth_established": False,
        "complete_impulse_fit_authorized": False,
        "reduced_cycle_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    manifest.tube.manifest.geometry._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    manifest.tube.manifest.geometry._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "implementation_commit": manifest.tube.manifest.geometry._git(
                "rev-parse", "HEAD"
            ),
            "implementation_tree": manifest.tube.manifest.geometry._git(
                "rev-parse", "HEAD^{tree}"
            ),
            "source_hashes": lock["contract"]["frozen_source_hashes"],
        },
    )
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{manifest.tube.manifest.geometry._sha(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Transition terminal prognosis WP10c9d6c7c3b5c4f25dw",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                f"Tail hidden-fraction slope per root: `{prognosis['linear_tail_slope_per_root']:.6e}`. Every tail interval decreases: `{prognosis['every_tail_fraction_decreases']}`. Forecast roots to the 0.25 exit gate: `{prognosis['forecast_roots_to_hidden_exit']}`.",
                "",
                f"Median half-step root cost is `{prognosis['median_half_step_root_wall_seconds']:.3f}` s. Another bounded extension is authorized: `{prognosis['bounded_extension_warranted']}`.",
                "",
                "The validated online architecture is branch-first: solve G(q,h)=0 offline, continue branch sheets by pseudo-arclength through folds, evolve q online with multi-second macrosteps, and use the scalar conservative tube only for prevalidated transitions. The next package may define the cold 12 ms branch-root pilot.",
                "",
                "No hot branch, complete impulse, or reduced cycle is yet established.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classify", action="store_true")
    args = parser.parse_args()
    if not args.classify:
        parser.error("use --classify")
    print(
        json.dumps(
            manifest.tube.manifest.geometry._plain(_execute()),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
