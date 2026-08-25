#!/usr/bin/env python3
"""Execute the prospective conservative tangent-phase-atlas holdout."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
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

from imri_qpe.layer3_minidisk_1d.tangent_phase_atlas import (  # noqa: E402
    fit_tangent_phase_chart,
    normalized_metric_tangents,
)
import run_causal_inner_adaptive_metric_chart_continuation_execution_wp10c9d6c7c3b5c4f25fip as engine  # noqa: E402
import run_causal_inner_adaptive_metric_chart_cycle_readiness_phase_atlas_manifest_wp10c9d6c7c3b5c4f25fiu as manifest  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = manifest.AUTHORIZED_NEXT
PASS_CLASSIFICATION = "tangent_phase_atlas_prospective_holdout_passed"
PHASE_FAILURE_CLASSIFICATION = "tangent_phase_atlas_prospective_geometry_failed"
PHYSICAL_FAILURE_CLASSIFICATION = "tangent_phase_holdout_original_physical_gate_failed"
NUMERICAL_FAILURE_CLASSIFICATION = "tangent_phase_holdout_numerical_or_restart_failed"
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fiw_tangent_phase_lap_recurrence_manifest"
)
ARTIFACT = (
    "causal_inner_conservative_tangent_phase_atlas_holdout_execution_"
    "wp10c9d6c7c3b5c4f25fiv"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_CONSERVATIVE_TANGENT_PHASE_"
    "ATLAS_HOLDOUT_EXECUTION_WP10C9D6C7C3B5C4F25FIV_2026-08-24.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_conservative_tangent_phase_atlas_holdout_"
    "execution_wp10c9d6c7c3b5c4f25fiv.py"
)
THIS_TEST = (
    "tests/test_causal_inner_conservative_tangent_phase_atlas_holdout_"
    "execution_wp10c9d6c7c3b5c4f25fiv.py"
)

# Adapter constants consumed by the already certified continuation engine.
_LEGACY = manifest.parent.manifest
_CONTRACT_FILE = manifest.CANONICAL_DIRECTORY / "phase_atlas_contract.json"
INITIAL_ELAPSED_SECONDS = 0.16400000000000012
MINIMUM_SEGMENT_SECONDS = 1.25e-4
MAXIMUM_SEGMENT_SECONDS = 2.5e-4
GROWTH_FACTOR = 2.0
ACCEPTED_SEGMENTS_BEFORE_GROWTH = 4
BLIND_MIDPOINT_FREQUENCY = 4
MAXIMUM_ACCEPTED_SEGMENTS = 16
MAXIMUM_ATTEMPTED_SEGMENTS = 18
MAXIMUM_EXACT_FREE_FIELD_CALLS = 20
MAXIMUM_RETRACTIONS = 20
MAXIMUM_EXECUTION_WALL_HOURS = 2.5
MAXIMUM_ENDPOINT_INTEGRAL_DEFECT = _LEGACY.MAXIMUM_ENDPOINT_INTEGRAL_DEFECT
MAXIMUM_BLIND_MIDPOINT_RATE_DEFECT = _LEGACY.MAXIMUM_BLIND_MIDPOINT_RATE_DEFECT


_BASE_HELPER_MODULE = manifest._helper()
_ORIGINAL_ENGINE_ATTEMPT = engine._attempt


def _helper():
    return manifest._helper()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def _contract() -> dict:
    return _helper()._read(_CONTRACT_FILE)


def _validate_manifest(*, require_clean: bool) -> dict:
    helper = _helper()
    hashes = helper._validate_checksums(manifest.CANONICAL_DIRECTORY)
    summary = helper._read(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = _contract()
    provenance = helper._read(manifest.CANONICAL_DIRECTORY / "provenance.json")
    scope = contract["scope"]
    gates = contract["binding_phase_gates"]
    if (
        summary["classification"] != manifest.CLASSIFICATION
        or not summary["passed"]
        or not summary["definitions_only"]
        or not summary["phase_atlas_holdout_execution_authorized"]
        or summary["phase_atlas_holdout_execution_executed"]
        or summary["complete_cycle_execution_authorized"]
        or summary["reduced_slow_evolution_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["authorized_execution"] != WORK_PACKAGE
        or scope["accepted_segments"] != MAXIMUM_ACCEPTED_SEGMENTS
        or scope["maximum_attempted_segments"] != MAXIMUM_ATTEMPTED_SEGMENTS
        or scope["maximum_exact_free_field_calls"] != MAXIMUM_EXACT_FREE_FIELD_CALLS
        or scope["maximum_retractions"] != MAXIMUM_RETRACTIONS
        or scope["maximum_execution_wall_hours"] != MAXIMUM_EXECUTION_WALL_HOURS
        or gates["minimum_training_two_plane_energy_fraction"] != 0.999
        or gates["maximum_direction_prediction_defect_radians"] != 0.005
    ):
        raise RuntimeError("tangent-phase holdout authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if helper._sha(ROOT / relative) != expected:
            raise RuntimeError(f"frozen tangent-phase source changed: {relative}")
    manifest._validate_parent(require_clean=False)
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("tangent-phase holdout requires a clean tracked tree")
    return {
        "hashes": hashes,
        "summary": summary,
        "contract": contract,
        "provenance": provenance,
    }


def _seed() -> dict[str, np.ndarray]:
    return _load_npz(manifest.CANONICAL_DIRECTORY / "continuation_seed.npz")


def _diagnostic_arrays() -> dict[str, np.ndarray]:
    return _load_npz(
        manifest.CANONICAL_DIRECTORY / "phase_atlas_diagnostic_arrays.npz"
    )


def _initial_progress() -> dict:
    seed = _seed()
    return {
        "previous_coordinate": seed["previous_coordinate470"].copy(),
        "current_coordinate": seed["current_coordinate470"].copy(),
        "previous_state": seed["previous_primitive_state"].copy(),
        "current_state": seed["current_primitive_state"].copy(),
        "previous_rate": seed["previous_coordinate_rate470_per_s"].copy(),
        "current_rate": seed["current_coordinate_rate470_per_s"].copy(),
        "previous_span": float(seed["previous_span_seconds"]),
        "next_span": float(seed["next_span_seconds"]),
        "elapsed_seconds": float(seed["elapsed_seconds"]),
        "accepted_segments_total": int(seed["accepted_segments_total"]),
        "accepted_segments_new": 0,
        "attempts": 0,
        "accepted_since_growth": int(seed["accepted_since_growth"]),
        "metric_transform": seed["metric_transform470x470"].copy(),
        "metric_augmented": seed["metric_augmented560x560"].copy(),
        "gauge_basis": seed["gauge_basis560x90"].copy(),
        "section_normal": seed["section_normal470"].copy(),
        "start_coordinate": seed["start_coordinate470"].copy(),
        "stop_reason": None,
    }


def _source_hashes() -> dict[str, str]:
    helper = _helper()
    paths = (
        THIS_RUNNER,
        THIS_TEST,
        manifest.THIS_RUNNER,
        manifest.THIS_TEST,
        manifest.PHASE_SOURCE,
        manifest.PHASE_TEST,
        engine.THIS_RUNNER,
        engine.THIS_TEST,
        engine.suffix.THIS_RUNNER,
        engine.execution.source.THIS_RUNNER,
    )
    return {name: helper._sha(ROOT / name) for name in paths}


def _identity(lock: dict) -> dict:
    helper = _helper()
    return {
        "work_package": WORK_PACKAGE,
        "implementation_commit": helper._git("rev-parse", "HEAD"),
        "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"),
        "manifest_hashes": lock["hashes"],
        "source_hashes": _source_hashes(),
        "contract": lock["contract"],
    }


def _prepare_scratch(lock: dict) -> dict:
    helper = _helper()
    identity = _identity(lock)
    path = SCRATCH_DIRECTORY / "execution_identity.json"
    if SCRATCH_DIRECTORY.exists():
        if not path.exists() or helper._read(path) != identity:
            raise RuntimeError("tangent-phase holdout scratch mismatch")
    else:
        SCRATCH_DIRECTORY.mkdir(parents=True)
        helper._write_json(path, identity)
    return identity


def _accepted_phase_history() -> np.ndarray:
    training = _diagnostic_arrays()["terminal_training_raw_rates470_per_s"].copy()
    accepted = []
    if SCRATCH_DIRECTORY.exists():
        for directory in sorted(SCRATCH_DIRECTORY.glob("attempt_[0-9][0-9][0-9][0-9]")):
            metrics_path = directory / "attempt.json"
            arrays_path = directory / "attempt.npz"
            if not metrics_path.exists() or not arrays_path.exists():
                continue
            metrics = _helper()._read(metrics_path)
            if metrics["accepted"]:
                accepted.append(
                    _load_npz(arrays_path)["accepted_coordinate_rate470_per_s"]
                )
    if accepted:
        training = np.vstack((training, np.stack(accepted)))
    return training[-manifest.SELECTED_WINDOW :]


def _predicted_unit_tangent(chart, phase_increment: float) -> np.ndarray:
    phase = float(chart.training_phases[-1] + phase_increment)
    oriented_angle = phase + chart.oriented_angle_origin
    raw_angle = chart.orientation_sign * oriented_angle
    plane = chart.circle_center + chart.circle_radius * np.asarray(
        (np.cos(raw_angle), np.sin(raw_angle))
    )
    tangent = chart.mean_tangent + chart.plane_basis @ plane
    return tangent / np.linalg.norm(tangent)


def _prediction(progress: dict) -> tuple[dict, dict[str, np.ndarray], object]:
    history = _accepted_phase_history()
    transform = _diagnostic_arrays()["terminal_metric_transform470x470"]
    unit = normalized_metric_tangents(history, transform)
    chart = fit_tangent_phase_chart(
        unit, predictor_increment_count=manifest.PREDICTOR_INCREMENT_COUNT
    )
    span_ratio = float(progress["next_span"] / progress["previous_span"])
    increment = float(chart.predicted_phase_increment * span_ratio)
    predicted = _predicted_unit_tangent(chart, increment)
    metrics = {
        "attempt_index": int(progress["attempts"]),
        "tentative_segment_number": int(progress["accepted_segments_total"] + 1),
        "history_samples": int(len(history)),
        "span_seconds": float(progress["next_span"]),
        "previous_span_seconds": float(progress["previous_span"]),
        "span_ratio": span_ratio,
        "predicted_phase_increment": increment,
        "training_two_plane_energy_fraction": chart.two_plane_energy_fraction,
        "training_relative_radial_rms": chart.training_relative_radial_rms,
        "circle_solve_condition_number": chart.circle_solve_condition_number,
        "frozen_before_exact_holdout": True,
    }
    arrays = {
        "training_raw_rates470_per_s": history,
        "training_unit_tangents470": unit,
        "mean_tangent470": chart.mean_tangent,
        "plane_basis470x2": chart.plane_basis,
        "circle_center2": chart.circle_center,
        "circle_radius": np.asarray(chart.circle_radius),
        "orientation_sign": np.asarray(chart.orientation_sign),
        "oriented_angle_origin": np.asarray(chart.oriented_angle_origin),
        "training_phases": chart.training_phases,
        "predicted_unit_tangent470": predicted,
        "terminal_metric_transform470x470": transform,
    }
    return metrics, arrays, chart


def _phase_gate(
    prediction: dict,
    prediction_arrays: dict[str, np.ndarray],
    chart,
    exact_raw_rate: np.ndarray,
) -> dict:
    transform = prediction_arrays["terminal_metric_transform470x470"]
    exact = normalized_metric_tangents(exact_raw_rate[None, :], transform)[0]
    geometry = chart.evaluate(exact)
    direction = float(
        np.arccos(
            np.clip(
                prediction_arrays["predicted_unit_tangent470"] @ exact,
                -1.0,
                1.0,
            )
        )
    )
    gates = _contract()["binding_phase_gates"]
    passed = bool(
        geometry["phase_increment"]
        > gates["minimum_phase_increment_strictly_greater_than"]
        and geometry["phase_increment"] <= gates["maximum_phase_increment"]
        and prediction["training_two_plane_energy_fraction"]
        >= gates["minimum_training_two_plane_energy_fraction"]
        and prediction["training_relative_radial_rms"]
        <= gates["maximum_training_relative_radial_rms"]
        and geometry["relative_radial_defect"]
        <= gates["maximum_holdout_relative_radial_defect"]
        and geometry["out_of_plane_defect"]
        <= gates["maximum_holdout_out_of_plane_defect"]
        and direction <= gates["maximum_direction_prediction_defect_radians"]
    )
    return {
        **geometry,
        "direction_prediction_defect_radians": direction,
        "predicted_phase_increment": prediction["predicted_phase_increment"],
        "passed": passed,
    }


def _phase_attempt(*, progress: dict, inputs: dict, exact_chart):
    helper = _helper()
    directory = engine._attempt_directory(int(progress["attempts"]))
    directory.mkdir(exist_ok=True)
    prediction_path = directory / "phase_prediction.json"
    prediction_arrays_path = directory / "phase_prediction.npz"
    if prediction_path.exists() or prediction_arrays_path.exists():
        if not prediction_path.exists() or not prediction_arrays_path.exists():
            raise RuntimeError("incomplete frozen phase prediction")
        prediction = helper._read(prediction_path)
        prediction_arrays = _load_npz(prediction_arrays_path)
        _new_prediction, _new_arrays, chart = _prediction(progress)
        if prediction != _new_prediction:
            raise RuntimeError("frozen phase prediction metadata changed")
        for name, value in _new_arrays.items():
            np.testing.assert_array_equal(prediction_arrays[name], value)
    else:
        prediction, prediction_arrays, chart = _prediction(progress)
        helper._write_json(prediction_path, prediction)
        _save_npz(prediction_arrays_path, prediction_arrays)
    metrics, arrays = _ORIGINAL_ENGINE_ATTEMPT(
        progress=progress, inputs=inputs, exact_chart=exact_chart
    )
    if "phase_geometry" in metrics:
        return metrics, arrays
    if metrics["accepted"]:
        phase = _phase_gate(
            prediction,
            prediction_arrays,
            chart,
            arrays["endpoint_coordinate_rate470_per_s"],
        )
        metrics["phase_geometry"] = phase
        if not phase["passed"]:
            metrics.update(
                {
                    "accepted": False,
                    "numerical_passed": False,
                    "stop_reason": "phase_geometry",
                }
            )
            arrays.update(
                {
                    "accepted_coordinate470": progress["current_coordinate"],
                    "accepted_primitive_state": progress["current_state"],
                    "accepted_coordinate_rate470_per_s": progress["current_rate"],
                    "accepted_metric_transform470x470": progress[
                        "metric_transform"
                    ],
                    "accepted_metric_augmented560x560": progress[
                        "metric_augmented"
                    ],
                    "accepted_gauge_basis560x90": progress["gauge_basis"],
                }
            )
    else:
        metrics["phase_geometry"] = None
    helper._write_json(directory / "attempt.json", metrics)
    _save_npz(directory / "attempt.npz", arrays)
    return metrics, arrays


_ENGINE_NAMES = (
    "manifest",
    "WORK_PACKAGE",
    "PASS_CLASSIFICATION",
    "PHYSICAL_FAILURE_CLASSIFICATION",
    "NUMERICAL_FAILURE_CLASSIFICATION",
    "AUTHORIZED_NEXT",
    "SCRATCH_DIRECTORY",
    "_initial_progress",
    "_helper",
    "_attempt",
)


def _stable_engine_helper():
    return _BASE_HELPER_MODULE


@contextmanager
def _engine_context():
    saved = {name: getattr(engine, name) for name in _ENGINE_NAMES}
    replacements = {
        "manifest": sys.modules[__name__],
        "WORK_PACKAGE": WORK_PACKAGE,
        "PASS_CLASSIFICATION": PASS_CLASSIFICATION,
        "PHYSICAL_FAILURE_CLASSIFICATION": PHYSICAL_FAILURE_CLASSIFICATION,
        "NUMERICAL_FAILURE_CLASSIFICATION": NUMERICAL_FAILURE_CLASSIFICATION,
        "AUTHORIZED_NEXT": AUTHORIZED_NEXT,
        "SCRATCH_DIRECTORY": SCRATCH_DIRECTORY,
        "_initial_progress": _initial_progress,
        "_helper": _stable_engine_helper,
        "_attempt": _phase_attempt,
    }
    try:
        for name, value in replacements.items():
            setattr(engine, name, value)
        yield
    finally:
        for name, value in saved.items():
            setattr(engine, name, value)


def _classify_phase(metrics: dict, arrays: dict[str, np.ndarray]) -> tuple[dict, dict]:
    result_metrics = dict(metrics)
    result_arrays = dict(arrays)
    phase_records = []
    for directory in sorted(SCRATCH_DIRECTORY.glob("attempt_[0-9][0-9][0-9][0-9]")):
        path = directory / "attempt.json"
        if path.exists():
            item = _helper()._read(path)
            if item.get("phase_geometry") is not None:
                phase_records.append(item["phase_geometry"])
    phase_passed = bool(
        len(phase_records) == MAXIMUM_ACCEPTED_SEGMENTS
        and all(item["passed"] for item in phase_records)
    )
    phase_failed = any(not item["passed"] for item in phase_records)
    values = dict(metrics["gate_values"])
    values.update(
        {
            "phase_holdouts_evaluated": len(phase_records),
            "all_phase_holdouts_passed": phase_passed,
            "minimum_phase_increment": min(
                (item["phase_increment"] for item in phase_records), default=None
            ),
            "maximum_phase_increment": max(
                (item["phase_increment"] for item in phase_records), default=None
            ),
            "maximum_phase_radial_defect": max(
                (item["relative_radial_defect"] for item in phase_records),
                default=None,
            ),
            "maximum_phase_out_of_plane_defect": max(
                (item["out_of_plane_defect"] for item in phase_records),
                default=None,
            ),
            "maximum_phase_direction_prediction_defect_radians": max(
                (
                    item["direction_prediction_defect_radians"]
                    for item in phase_records
                ),
                default=None,
            ),
            "phase_lap_observed": False,
            "cycle_observed": False,
        }
    )
    result_metrics["gate_values"] = values
    if metrics["passed"] and phase_passed:
        result_metrics.update(
            {
                "classification": PASS_CLASSIFICATION,
                "passed": True,
                "authorized_next": AUTHORIZED_NEXT,
            }
        )
    elif phase_failed:
        result_metrics.update(
            {
                "classification": PHASE_FAILURE_CLASSIFICATION,
                "passed": False,
                "authorized_next": None,
            }
        )
    result_arrays.update(
        {
            "accepted_phase_increments": np.asarray(
                [item["phase_increment"] for item in phase_records]
            ),
            "accepted_phase_radial_defects": np.asarray(
                [item["relative_radial_defect"] for item in phase_records]
            ),
            "accepted_phase_out_of_plane_defects": np.asarray(
                [item["out_of_plane_defect"] for item in phase_records]
            ),
            "accepted_phase_direction_prediction_defects_radians": np.asarray(
                [
                    item["direction_prediction_defect_radians"]
                    for item in phase_records
                ]
            ),
        }
    )
    return result_metrics, result_arrays


def _execute(lock: dict, identity: dict) -> tuple[dict, dict[str, np.ndarray]]:
    with _engine_context():
        metrics, arrays = engine._execute(lock, identity)
    return _classify_phase(metrics, arrays)


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
                    "scientific_status": (
                        "SUPPORTED" if summary["passed"] else "REJECTED"
                    ),
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
        "passed": summary["passed"],
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


def _canonicalize(metrics: dict, arrays: dict[str, np.ndarray], lock: dict, identity: dict) -> dict:
    helper = _helper()
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("tangent-phase holdout result already exists")
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "holdout_metrics.json", metrics)
    _save_npz(CANONICAL_DIRECTORY / "holdout_arrays.npz", arrays)
    helper._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "manifest_hashes": lock["hashes"],
            "manifest_classification": lock["summary"]["classification"],
            "execution_identity": identity,
        },
    )
    values = metrics["gate_values"]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": metrics["passed"],
        "new_accepted_segments": values["accepted_segments"],
        "terminal_elapsed_seconds": values["terminal_elapsed_seconds"],
        "phase_atlas_prospectively_validated": values[
            "all_phase_holdouts_passed"
        ],
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": metrics["authorized_next"],
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    helper._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "implementation_commit": identity["implementation_commit"],
            "implementation_tree": identity["implementation_tree"],
            "source_hashes": identity["source_hashes"],
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
                "# Conservative tangent-phase-atlas holdout execution",
                "",
                f"Classification: `{metrics['classification']}`.",
                "",
                f"Accepted `{values['accepted_segments']}` of `{values['attempted_segments']}` attempted segments, reaching `{values['terminal_elapsed_seconds']:.6f}` s. All prospective phase holdouts passed: `{values['all_phase_holdouts_passed']}`.",
                "",
                f"Observed phase-increment range: `{values['minimum_phase_increment']}` to `{values['maximum_phase_increment']}`. Maximum radial, out-of-plane, and direction-prediction defects were `{values['maximum_phase_radial_defect']}`, `{values['maximum_phase_out_of_plane_defect']}`, and `{values['maximum_phase_direction_prediction_defect_radians']}` rad.",
                "",
                f"Maximum endpoint/blind defects: `{values['maximum_accepted_endpoint_integral_defect']:.6e}` / `{values['maximum_accepted_blind_midpoint_rate_defect']:.6e}`. Exact fields/retractions/wall seconds: `{values['exact_free_field_calls']}` / `{values['retractions']}` / `{values['execution_wall_seconds']:.3f}`.",
                "",
                "No phase lap or state recurrence was claimed. A phase lap remains insufficient for a cycle without recurrence and transverse return.",
                "",
                f"Authorized next artifact: `{metrics['authorized_next']}`. Complete-cycle execution and reduced slow evolution remain unauthorized.",
                "",
            )
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    arguments = parser.parse_args()
    if not arguments.run:
        parser.error("--run is required")
    lock = _validate_manifest(require_clean=True)
    identity = _prepare_scratch(lock)
    metrics, arrays = _execute(lock, identity)
    summary = _canonicalize(metrics, arrays, lock, identity)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
