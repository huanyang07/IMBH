#!/usr/bin/env python3
"""Freeze a four-segment conservative metric-chart suffix."""

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

import run_causal_inner_metric_chart_boundary_crossing_execution_wp10c9d6c7c3b5c4f25fif as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25fig"
CLASSIFICATION = "metric_chart_short_suffix_manifest_frozen"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25fih_metric_chart_short_suffix_execution"
ARTIFACT = (
    "causal_inner_metric_chart_short_suffix_manifest_"
    "wp10c9d6c7c3b5c4f25fig"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_METRIC_CHART_SHORT_SUFFIX_"
    "MANIFEST_WP10C9D6C7C3B5C4F25FIG_2026-08-24.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_metric_chart_short_suffix_manifest_"
    "wp10c9d6c7c3b5c4f25fig.py"
)
THIS_TEST = (
    "tests/test_causal_inner_metric_chart_short_suffix_manifest_"
    "wp10c9d6c7c3b5c4f25fig.py"
)

INITIAL_ELAPSED_SECONDS = parent.manifest.ENDPOINT_ELAPSED_SECONDS
SEGMENT_SECONDS = parent.manifest.SEGMENT_SECONDS
NEW_SEGMENTS = 4
NEW_HORIZON_SECONDS = NEW_SEGMENTS * SEGMENT_SECONDS
TERMINAL_ELAPSED_SECONDS = INITIAL_ELAPSED_SECONDS + NEW_HORIZON_SECONDS
INITIAL_ACCEPTED_SEGMENTS = parent.manifest.EXPECTED_PRIOR_ACCEPTED_SEGMENTS + 1
TENTATIVE_SEGMENT_NUMBERS = (73, 74, 75, 76)
BLIND_MIDPOINT_SEGMENT = 76
RESTART_AFTER_NEW_SEGMENTS = 2
MAXIMUM_EXACT_FREE_FIELD_CALLS = NEW_SEGMENTS + 1
MAXIMUM_INDEPENDENT_METRIC_JACOBIAN_AUDITS = NEW_SEGMENTS + 1
MAXIMUM_METRIC_RETRACTIONS = NEW_SEGMENTS + 1
MAXIMUM_METRIC_JACOBIAN_CONDITION = parent.manifest.MAXIMUM_METRIC_JACOBIAN_CONDITION
MAXIMUM_METRIC_AUGMENTED_CONDITION = parent.manifest.MAXIMUM_METRIC_AUGMENTED_CONDITION
MAXIMUM_PATCH_TRANSITION_CONDITION = parent.manifest.MAXIMUM_PATCH_TRANSITION_CONDITION
MAXIMUM_TRANSFORM_INVERSE_CLOSURE = parent.manifest.MAXIMUM_TRANSFORM_INVERSE_CLOSURE
MAXIMUM_COORDINATE_RECONSTRUCTION_DEFECT = parent.manifest.MAXIMUM_COORDINATE_RECONSTRUCTION_DEFECT
MAXIMUM_ENDPOINT_INTEGRAL_DEFECT = parent.manifest.MAXIMUM_ENDPOINT_INTEGRAL_DEFECT
MAXIMUM_BLIND_MIDPOINT_RATE_DEFECT = parent.manifest.MAXIMUM_BLIND_MIDPOINT_RATE_DEFECT
ORIGINAL_COORDINATE_RESIDUAL_TOLERANCE = parent.manifest.ORIGINAL_COORDINATE_RESIDUAL_TOLERANCE
METRIC_COORDINATE_RESIDUAL_TOLERANCE = parent.manifest.METRIC_COORDINATE_RESIDUAL_TOLERANCE
GAUGE_RESIDUAL_TOLERANCE = parent.manifest.GAUGE_RESIDUAL_TOLERANCE
MAXIMUM_EXECUTION_WALL_HOURS = 2.0


def _helper():
    return parent._helper()


def _validate_parent(*, require_clean: bool) -> dict:
    helper = _helper()
    hashes = helper._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = helper._read(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = helper._read(
        parent.CANONICAL_DIRECTORY / "boundary_execution_metrics.json"
    )
    values = metrics["gate_values"]
    if (
        summary["classification"] != parent.PASS_CLASSIFICATION
        or not summary["passed"]
        or not summary["metric_chart_boundary_crossing_passed"]
        or not summary["historical_f25fi_rejection_preserved"]
        or summary["new_accepted_segments"] != 1
        or not summary["short_suffix_manifest_authorized"]
        or summary["authorized_next"] != parent.AUTHORIZED_NEXT
        or metrics["classification"] != parent.PASS_CLASSIFICATION
        or not metrics["passed"]
        or values["terminal_elapsed_seconds"] != INITIAL_ELAPSED_SECONDS
        or values["new_accepted_segments"] != 1
        or not values["endpoint_retraction_replay_bitwise"]
        or not values["checkpoint_roundtrip_bitwise"]
        or not values["all_new_exact_fields_physical_passed"]
        or values["maximum_metric_coordinate_jacobian_condition"]
        > MAXIMUM_METRIC_JACOBIAN_CONDITION
        or metrics["fixed_Q_physical_rate_calls"] != 0
        or metrics["fixed_Q_reaction_calls"] != 0
    ):
        raise RuntimeError("metric boundary crossing evidence changed")
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("short-suffix manifest requires a clean tracked tree")
    return {"hashes": hashes, "classification": summary["classification"]}


def _seed() -> dict[str, np.ndarray]:
    arrays = _helper()._load_npz(
        parent.CANONICAL_DIRECTORY / "boundary_execution_arrays.npz"
    )
    required = {
        "previous_coordinate470": (470,),
        "current_coordinate470": (470,),
        "previous_primitive_state": (112, 5),
        "current_primitive_state": (112, 5),
        "previous_coordinate_rate470_per_s": (470,),
        "current_coordinate_rate470_per_s": (470,),
        "current_metric_transform470x470": (470, 470),
        "current_metric_augmented560x560": (560, 560),
        "current_gauge_basis560x90": (560, 90),
        "section_normal470": (470,),
        "start_coordinate470": (470,),
    }
    seed = {name: np.asarray(arrays[name]) for name in required}
    if any(seed[name].shape != shape for name, shape in required.items()):
        raise RuntimeError("accepted boundary checkpoint shape changed")
    for name in ("previous_span_seconds", "next_span_seconds", "elapsed_seconds"):
        seed[name] = np.asarray(arrays[name])
    seed["accepted_segments_total"] = np.asarray(arrays["accepted_segments_total"])
    if (
        float(seed["previous_span_seconds"]) != SEGMENT_SECONDS
        or float(seed["next_span_seconds"]) != SEGMENT_SECONDS
        or float(seed["elapsed_seconds"]) != INITIAL_ELAPSED_SECONDS
        or int(seed["accepted_segments_total"]) != INITIAL_ACCEPTED_SEGMENTS
    ):
        raise RuntimeError("accepted boundary continuation counters changed")
    return seed


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "authorized_execution": AUTHORIZED_NEXT,
        "truth_system": {
            "field": "autonomous original reaction-free field dy/dt=f_free(y)",
            "state": "original primitive state u",
            "physical_coordinate": "original q=C(u)",
            "metric_chart": "invertible block-whitened numerical residual only",
            "all_physics_and_ledgers_in_original_coordinates": True,
            "fixed_Q_rate_or_reaction": "forbidden",
            "external_clock_or_phase": "forbidden",
        },
        "suffix": {
            "initial_elapsed_seconds": INITIAL_ELAPSED_SECONDS,
            "fixed_segment_seconds": SEGMENT_SECONDS,
            "new_segments": NEW_SEGMENTS,
            "new_horizon_seconds": NEW_HORIZON_SECONDS,
            "terminal_elapsed_seconds": TERMINAL_ELAPSED_SECONDS,
            "tentative_segment_numbers": TENTATIVE_SEGMENT_NUMBERS,
            "blind_midpoint_segment": BLIND_MIDPOINT_SEGMENT,
            "predictor": "equal-step AB2 in original q",
            "interpolant": "cubic Hermite in original q",
            "reanchor_after_every_accepted_endpoint": True,
            "failed_candidate_is_never_propagated": True,
        },
        "restart": {
            "checkpoint_after_new_segments": RESTART_AFTER_NEW_SEGMENTS,
            "checkpoint_roundtrip_bitwise": True,
            "remaining_two_segment_history_replay_bitwise": True,
            "replay_uses_saved_exact_fields_without_new_field_calls": True,
        },
        "scope": {
            "new_exact_free_field_calls": MAXIMUM_EXACT_FREE_FIELD_CALLS,
            "new_independent_metric_jacobian_audits": (
                MAXIMUM_INDEPENDENT_METRIC_JACOBIAN_AUDITS
            ),
            "new_metric_retractions": MAXIMUM_METRIC_RETRACTIONS,
            "new_accepted_segments_required": NEW_SEGMENTS,
            "new_physical_time_seconds": NEW_HORIZON_SECONDS,
            "fixed_Q_calls": 0,
            "reaction_calls": 0,
            "nonlinear_roots": 0,
            "BDF_microsteps": 0,
            "maximum_wall_hours": MAXIMUM_EXECUTION_WALL_HOURS,
        },
        "gates": {
            "original_coordinate_residual_tolerance": ORIGINAL_COORDINATE_RESIDUAL_TOLERANCE,
            "metric_coordinate_residual_tolerance": METRIC_COORDINATE_RESIDUAL_TOLERANCE,
            "gauge_residual_tolerance": GAUGE_RESIDUAL_TOLERANCE,
            "maximum_metric_jacobian_condition": MAXIMUM_METRIC_JACOBIAN_CONDITION,
            "maximum_metric_augmented_condition": MAXIMUM_METRIC_AUGMENTED_CONDITION,
            "maximum_patch_transition_condition": MAXIMUM_PATCH_TRANSITION_CONDITION,
            "maximum_transform_inverse_closure": MAXIMUM_TRANSFORM_INVERSE_CLOSURE,
            "maximum_coordinate_reconstruction_defect": MAXIMUM_COORDINATE_RECONSTRUCTION_DEFECT,
            "maximum_endpoint_integral_defect": MAXIMUM_ENDPOINT_INTEGRAL_DEFECT,
            "maximum_blind_midpoint_rate_defect": MAXIMUM_BLIND_MIDPOINT_RATE_DEFECT,
            "rank": 470,
            "minimum_reconstruction_factor": 1.0 - 1.0e-12,
            "maximum_height_ratio": 0.5,
            "minimum_scattering_optical_depth": 1.0,
            "all_reaction_free_ledgers": True,
        },
        "decision": {
            "pass": "metric_chart_short_suffix_passed_wide_resume_manifest_authorized",
            "physical_failure": "metric_chart_short_suffix_original_physical_gate_failed",
            "numerical_failure": "metric_chart_short_suffix_numerical_or_restart_failed",
        },
        "forbidden": [
            "adapt or grow the timestep in this suffix",
            "evaluate physics in metric coordinates",
            "bind the historical raw condition after the prospective atlas switch",
            "propagate a rejected endpoint",
            "authorize a cycle or reduced slow evolution",
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
            rows.append({
                "case": ARTIFACT,
                "path": str(path.relative_to(ROOT)),
                "bytes": str(path.stat().st_size),
                "sha256": helper._sha(path),
                "scientific_status": "SUPPORTED",
            })
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
    catalog.update({
        "case_count": len({row["case"] for row in rows}),
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "all_payload_hashes_recorded": True,
        "latest_source_parent_commit": helper._git("rev-parse", "HEAD"),
        "latest_work_package": WORK_PACKAGE,
    })
    helper._write_json(summary_path, catalog)


def _freeze() -> dict:
    helper = _helper()
    lock = _validate_parent(require_clean=True)
    seed = _seed()
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("short-suffix manifest already exists")
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "suffix_contract.json", _contract())
    _save_npz(CANONICAL_DIRECTORY / "suffix_seed.npz", seed)
    helper._write_json(CANONICAL_DIRECTORY / "input_lock.json", lock)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "metric_chart_short_suffix_authorized": True,
        "metric_chart_short_suffix_executed": False,
        "new_trajectory": False,
        "cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    helper._write_json(CANONICAL_DIRECTORY / "provenance.json", {
        "runner": THIS_RUNNER,
        "test": THIS_TEST,
        "definition_commit": helper._git("rev-parse", "HEAD"),
        "definition_tree": helper._git("rev-parse", "HEAD^{tree}"),
        "source_hashes": {
            THIS_RUNNER: helper._sha(ROOT / THIS_RUNNER),
            THIS_TEST: helper._sha(ROOT / THIS_TEST),
            parent.THIS_RUNNER: helper._sha(ROOT / parent.THIS_RUNNER),
            parent.ATLAS_SOURCE: helper._sha(ROOT / parent.ATLAS_SOURCE),
            parent.execution.source.THIS_RUNNER: helper._sha(
                ROOT / parent.execution.source.THIS_RUNNER
            ),
        },
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
    })
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
        "\n".join((
            "# Metric-chart short-suffix manifest",
            "",
            f"Classification: `{CLASSIFICATION}`.",
            "",
            "This definitions-only package freezes four consecutive 0.25 ms AB2/Hermite segments after the certified 111.50 ms boundary crossing. Every endpoint receives a fresh local metric chart; tentative segment 76 also receives a blind midpoint field evaluation.",
            "",
            "The suffix is limited to 1 ms and tests repeated chart transition, accepted-history propagation, and restart replay. It does not authorize a cycle run.",
            "",
            f"Authorized next artifact: `{AUTHORIZED_NEXT}`.",
            "",
        )),
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
