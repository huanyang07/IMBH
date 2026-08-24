#!/usr/bin/env python3
"""Freeze a nonpropagating local metric-chart radius diagnosis."""

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

import run_causal_inner_metric_chart_wide_continuation_resume_execution_wp10c9d6c7c3b5c4f25fij as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25fik"
CLASSIFICATION = "metric_chart_local_radius_diagnosis_manifest_frozen"
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fil_metric_chart_local_radius_diagnosis_execution"
)
ARTIFACT = (
    "causal_inner_metric_chart_local_radius_diagnosis_manifest_"
    "wp10c9d6c7c3b5c4f25fik"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_METRIC_CHART_LOCAL_RADIUS_"
    "DIAGNOSIS_MANIFEST_WP10C9D6C7C3B5C4F25FIK_2026-08-24.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_metric_chart_local_radius_diagnosis_manifest_"
    "wp10c9d6c7c3b5c4f25fik.py"
)
THIS_TEST = (
    "tests/test_causal_inner_metric_chart_local_radius_diagnosis_manifest_"
    "wp10c9d6c7c3b5c4f25fik.py"
)
STRICT_ATLAS_SOURCE = (
    "src/imri_qpe/layer3_minidisk_1d/conservative_metric_chart_atlas_v2.py"
)
STRICT_ATLAS_TEST = "tests/test_conservative_metric_chart_atlas_v2.py"

INITIAL_ELAPSED_SECONDS = 0.1325000000000001
INITIAL_ACCEPTED_SEGMENTS = 91
INITIAL_WIDE_ACCEPTED_SEGMENTS = 15
INITIAL_WIDE_ATTEMPTS = 16
PREVIOUS_SEGMENT_SECONDS = 2.0e-3
SPAN_LADDER_SECONDS = (2.0e-3, 1.0e-3, 5.0e-4)
MAXIMUM_RETRACTIONS = len(SPAN_LADDER_SECONDS)
MAXIMUM_FRESH_PATCH_AUDITS = 2
MAXIMUM_EXECUTION_WALL_HOURS = 1.0
ORIGINAL_COORDINATE_RESIDUAL_TOLERANCE = (
    parent.manifest.ORIGINAL_COORDINATE_RESIDUAL_TOLERANCE
)
METRIC_COORDINATE_RESIDUAL_TOLERANCE = (
    parent.manifest.METRIC_COORDINATE_RESIDUAL_TOLERANCE
)
GAUGE_RESIDUAL_TOLERANCE = parent.manifest.GAUGE_RESIDUAL_TOLERANCE
MAXIMUM_METRIC_JACOBIAN_CONDITION = (
    parent.manifest.MAXIMUM_METRIC_JACOBIAN_CONDITION
)
MAXIMUM_METRIC_AUGMENTED_CONDITION = (
    parent.manifest.MAXIMUM_METRIC_AUGMENTED_CONDITION
)
MAXIMUM_PATCH_TRANSITION_CONDITION = (
    parent.manifest.MAXIMUM_PATCH_TRANSITION_CONDITION
)


def _helper():
    return parent._helper()


def _validate_parent(*, require_clean: bool) -> dict:
    helper = _helper()
    hashes = helper._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = helper._read(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = helper._read(
        parent.CANONICAL_DIRECTORY / "resume_execution_metrics.json"
    )
    values = metrics["gate_values"]
    if (
        summary["classification"] != parent.NUMERICAL_FAILURE_CLASSIFICATION
        or summary["passed"]
        or summary["new_accepted_segments"] != INITIAL_WIDE_ACCEPTED_SEGMENTS
        or summary["cycle_observed"]
        or summary["equilibrium_candidate_observed"]
        or summary["authorized_next"] is not None
        or metrics["classification"] != parent.NUMERICAL_FAILURE_CLASSIFICATION
        or metrics["passed"]
        or values["terminal_elapsed_seconds"] != INITIAL_ELAPSED_SECONDS
        or values["accepted_segments"] != INITIAL_WIDE_ACCEPTED_SEGMENTS
        or values["attempted_segments"] != INITIAL_WIDE_ATTEMPTS
        or values["rejected_segments"] != 1
        or not values["all_exact_fields_physical_passed"]
        or not values["restart_roundtrip_bitwise"]
        or not values["suffix_history_replay_bitwise"]
        or metrics["fixed_Q_physical_rate_calls"] != 0
        or metrics["fixed_Q_reaction_calls"] != 0
        or metrics["nonlinear_roots"] != 0
        or metrics["BDF_microsteps"] != 0
    ):
        raise RuntimeError("wide-resume numerical-failure evidence changed")
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("local-radius manifest requires a clean tracked tree")
    return {
        "hashes": hashes,
        "classification": summary["classification"],
        "terminal_elapsed_seconds": values["terminal_elapsed_seconds"],
        "accepted_segments": values["accepted_segments"],
    }


def _seed() -> dict[str, np.ndarray]:
    arrays = _helper()._load_npz(
        parent.CANONICAL_DIRECTORY / "resume_execution_arrays.npz"
    )
    names = (
        "previous_coordinate470",
        "current_coordinate470",
        "previous_primitive_state",
        "current_primitive_state",
        "previous_coordinate_rate470_per_s",
        "current_coordinate_rate470_per_s",
        "previous_span_seconds",
        "elapsed_seconds",
        "accepted_segments_total",
        "current_metric_transform470x470",
        "current_metric_augmented560x560",
        "current_gauge_basis560x90",
    )
    seed = {name: np.asarray(arrays[name]) for name in names}
    if (
        seed["previous_coordinate470"].shape != (470,)
        or seed["current_coordinate470"].shape != (470,)
        or seed["previous_primitive_state"].shape != (112, 5)
        or seed["current_primitive_state"].shape != (112, 5)
        or seed["current_metric_transform470x470"].shape != (470, 470)
        or seed["current_metric_augmented560x560"].shape != (560, 560)
        or seed["current_gauge_basis560x90"].shape != (560, 90)
        or float(seed["elapsed_seconds"]) != INITIAL_ELAPSED_SECONDS
        or int(seed["accepted_segments_total"]) != INITIAL_ACCEPTED_SEGMENTS
        or float(seed["previous_span_seconds"]) != PREVIOUS_SEGMENT_SECONDS
    ):
        raise RuntimeError("wide-resume terminal seed changed")
    return seed


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "authorized_execution": AUTHORIZED_NEXT,
        "question": (
            "Does the rejected 2 ms predictor exceed only the current local "
            "metric-patch radius, with strict closure restored at 1 or 0.5 ms?"
        ),
        "truth_system": {
            "state": "hash-validated accepted original primitive state at 132.5 ms",
            "coordinate": "original conservative q=C(u)",
            "predictor": "variable-step AB2 from accepted original-coordinate history",
            "retraction": "frozen metric transport with strict-v2 status semantics",
            "fresh_patch": "exact block whitening at each strictly passed smaller target",
            "fixed_Q_rate_or_reaction": "forbidden",
            "exact_free_field_call": "forbidden",
        },
        "scope": {
            "initial_elapsed_seconds": INITIAL_ELAPSED_SECONDS,
            "span_ladder_seconds": list(SPAN_LADDER_SECONDS),
            "maximum_retractions": MAXIMUM_RETRACTIONS,
            "maximum_fresh_patch_audits": MAXIMUM_FRESH_PATCH_AUDITS,
            "maximum_wall_hours": MAXIMUM_EXECUTION_WALL_HOURS,
            "new_trajectory": False,
            "accepted_history_mutation": False,
            "exact_free_field_calls": 0,
            "fixed_Q_calls": 0,
            "reaction_calls": 0,
            "nonlinear_roots": 0,
            "BDF_microsteps": 0,
        },
        "gates": {
            "strict_status_requires_physical_closure_and_condition": True,
            "original_coordinate_residual_tolerance": ORIGINAL_COORDINATE_RESIDUAL_TOLERANCE,
            "metric_coordinate_residual_tolerance": METRIC_COORDINATE_RESIDUAL_TOLERANCE,
            "gauge_residual_tolerance": GAUGE_RESIDUAL_TOLERANCE,
            "maximum_metric_jacobian_condition": MAXIMUM_METRIC_JACOBIAN_CONDITION,
            "maximum_metric_augmented_condition": MAXIMUM_METRIC_AUGMENTED_CONDITION,
            "maximum_patch_transition_condition": MAXIMUM_PATCH_TRANSITION_CONDITION,
            "minimum_reconstruction_factor": 1.0 - 1.0e-12,
            "maximum_height_ratio": 0.5,
            "minimum_scattering_optical_depth": 1.0,
        },
        "decision": {
            "positive": (
                "2 ms strict failure reproduced and a smaller strictly passed "
                "well-conditioned fresh patch identifies the local radius"
            ),
            "positive_authorizes": (
                "definitions-only adaptive chart-radius recovery manifest"
            ),
            "physical_failure": "local radius diagnosis physical gate failed",
            "negative": "no smaller span restored a strict well-conditioned patch",
        },
        "forbidden": [
            "propagate any diagnosed state",
            "evaluate the slow free field",
            "reinterpret a physical failure as chart-radius failure",
            "relax any residual or conditioning gate",
            "authorize complete-cycle or reduced slow execution directly",
        ],
    }


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def _update_catalog(summary: dict) -> None:
    helper = _helper()
    manifest_path = ROOT / "results/manifests/canonical_artifacts.csv"
    summary_path = ROOT / "results/manifests/canonical_summary.json"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
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
                    "scientific_status": "SUPPORTED",
                }
            )
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case", "path", "bytes", "sha256", "scientific_status"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = helper._read(summary_path)
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
            "latest_source_parent_commit": helper._git("rev-parse", "HEAD"),
            "latest_work_package": WORK_PACKAGE,
        }
    )
    helper._write_json(summary_path, catalog)


def _freeze() -> dict:
    helper = _helper()
    lock = _validate_parent(require_clean=True)
    seed = _seed()
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("local-radius diagnosis manifest already exists")
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "diagnosis_contract.json", _contract())
    _save_npz(CANONICAL_DIRECTORY / "diagnosis_seed.npz", seed)
    helper._write_json(CANONICAL_DIRECTORY / "input_lock.json", lock)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "metric_chart_local_radius_diagnosis_authorized": True,
        "metric_chart_local_radius_diagnosis_executed": False,
        "new_trajectory": False,
        "cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    helper._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "definition_commit": helper._git("rev-parse", "HEAD"),
            "definition_tree": helper._git("rev-parse", "HEAD^{tree}"),
            "source_hashes": {
                THIS_RUNNER: helper._sha(ROOT / THIS_RUNNER),
                THIS_TEST: helper._sha(ROOT / THIS_TEST),
                parent.THIS_RUNNER: helper._sha(ROOT / parent.THIS_RUNNER),
                parent.THIS_TEST: helper._sha(ROOT / parent.THIS_TEST),
                STRICT_ATLAS_SOURCE: helper._sha(ROOT / STRICT_ATLAS_SOURCE),
                STRICT_ATLAS_TEST: helper._sha(ROOT / STRICT_ATLAS_TEST),
                parent.source.parent.ATLAS_SOURCE: helper._sha(
                    ROOT / parent.source.parent.ATLAS_SOURCE
                ),
            },
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    )
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{helper._sha(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Metric-chart local-radius diagnosis manifest",
                "",
                f"Classification: `{CLASSIFICATION}`.",
                "",
                "This definitions-only package preserves the accepted 132.50 ms endpoint and diagnoses the rejected local patch without evaluating the slow free field or advancing the trajectory.",
                "",
                "It prospectively tests the exact AB2 targets at 2.00, 1.00, and 0.50 ms with strict-v2 retraction status, then builds fresh exact metric patches only at smaller targets that pass all retraction gates.",
                "",
                f"Authorized next artifact: `{AUTHORIZED_NEXT}`.",
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
    arguments = parser.parse_args()
    if not arguments.freeze:
        parser.error("--freeze is required")
    print(json.dumps(_freeze(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
