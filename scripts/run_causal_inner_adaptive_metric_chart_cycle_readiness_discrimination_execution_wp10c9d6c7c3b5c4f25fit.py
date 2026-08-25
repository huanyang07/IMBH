#!/usr/bin/env python3
"""Execute the bounded orientation-based cycle-readiness discrimination tranche."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_adaptive_metric_chart_continuation_execution_wp10c9d6c7c3b5c4f25fip as engine  # noqa: E402
import run_causal_inner_adaptive_metric_chart_cycle_readiness_execution_wp10c9d6c7c3b5c4f25fir as parent  # noqa: E402
import run_causal_inner_adaptive_metric_chart_cycle_readiness_reforecast_manifest_wp10c9d6c7c3b5c4f25fis as manifest  # noqa: E402


_BASE_HELPER_MODULE = parent._BASE_HELPER_MODULE
SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25fit"
ACQUISITION_COMMIT = "74086051673b111abc284704677947daa6f514e9"
ACQUISITION_TREE = "e82d4e258b42e3de3fc6f89b2e4ebf134027a017"
TURN_CLASSIFICATION = "cycle_readiness_orientation_turn_bracketed"
OPEN_CLASSIFICATION = "cycle_readiness_orientation_discrimination_open"
PHYSICAL_FAILURE_CLASSIFICATION = (
    "cycle_readiness_discrimination_original_physical_gate_failed"
)
NUMERICAL_FAILURE_CLASSIFICATION = (
    "cycle_readiness_discrimination_numerical_radius_or_budget_failed"
)
TURN_AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fiu_adaptive_metric_chart_negative_section_manifest"
)
OPEN_AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fiu_"
    "adaptive_metric_chart_cycle_readiness_reforecast_manifest"
)
ARTIFACT = (
    "causal_inner_adaptive_metric_chart_cycle_readiness_discrimination_"
    "execution_wp10c9d6c7c3b5c4f25fit"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ADAPTIVE_METRIC_CHART_"
    "CYCLE_READINESS_DISCRIMINATION_EXECUTION_"
    "WP10C9D6C7C3B5C4F25FIT_2026-08-24.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_adaptive_metric_chart_cycle_readiness_"
    "discrimination_execution_wp10c9d6c7c3b5c4f25fit.py"
)
THIS_TEST = (
    "tests/test_causal_inner_adaptive_metric_chart_cycle_readiness_"
    "discrimination_execution_wp10c9d6c7c3b5c4f25fit.py"
)


def _helper():
    return manifest._helper()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def _validate_manifest(*, require_clean: bool) -> dict:
    helper = _helper()
    hashes = helper._validate_checksums(manifest.CANONICAL_DIRECTORY)
    summary = helper._read(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = helper._read(
        manifest.CANONICAL_DIRECTORY / "forecast_discrimination_contract.json"
    )
    cost = helper._read(manifest.CANONICAL_DIRECTORY / "cost_projection.json")
    geometry = helper._read(
        manifest.CANONICAL_DIRECTORY / "geometry_reforecast.json"
    )
    provenance = helper._read(manifest.CANONICAL_DIRECTORY / "provenance.json")
    if (
        summary["classification"] != manifest.CLASSIFICATION
        or not summary["passed"]
        or not summary["definitions_only"]
        or not summary["forecast_discrimination_execution_authorized"]
        or summary["forecast_discrimination_execution_executed"]
        or summary["complete_cycle_execution_authorized"]
        or summary["reduced_slow_evolution_authorized"]
        or summary["authorized_next"] != manifest.AUTHORIZED_NEXT
        or contract["authorized_execution"] != manifest.AUTHORIZED_NEXT
        or contract["scope"]["maximum_accepted_segments"] != 48
        or contract["scope"]["maximum_attempted_segments"] != 51
        or contract["scope"]["maximum_exact_free_field_calls"] != 63
        or contract["scope"]["maximum_retractions"] != 63
        or contract["adaptive_policy"]["maximum_segment_seconds"] != 2.5e-4
        or geometry["section_turning_point_observed"]
        or geometry["section_negative_observed"]
        or geometry["terminal_orientation_cosine"] <= 0.0
        or not cost["cost_gate_passed"]
        or cost["maximum_projected_wall_hours"] > 8.0
    ):
        raise RuntimeError("cycle-readiness discrimination authorization changed")
    for relative, frozen_hash in provenance["source_hashes"].items():
        if helper._sha(ROOT / relative) != frozen_hash:
            raise RuntimeError(f"frozen reforecast source changed: {relative}")
    parent_lock = manifest._validate_parent(require_clean=False)
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError(
            "cycle-readiness discrimination execution requires a clean tracked tree"
        )
    return {
        "hashes": hashes,
        "summary": summary,
        "contract": contract,
        "cost": cost,
        "geometry": geometry,
        "parent_lock": parent_lock,
    }


def _seed() -> dict[str, np.ndarray]:
    return _load_npz(manifest.CANONICAL_DIRECTORY / "continuation_seed.npz")


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
)


def _stable_engine_helper():
    return _BASE_HELPER_MODULE


@contextmanager
def _engine_context():
    saved = {name: getattr(engine, name) for name in _ENGINE_NAMES}
    replacements = {
        "manifest": manifest,
        "WORK_PACKAGE": WORK_PACKAGE,
        "PASS_CLASSIFICATION": OPEN_CLASSIFICATION,
        "PHYSICAL_FAILURE_CLASSIFICATION": PHYSICAL_FAILURE_CLASSIFICATION,
        "NUMERICAL_FAILURE_CLASSIFICATION": NUMERICAL_FAILURE_CLASSIFICATION,
        "AUTHORIZED_NEXT": OPEN_AUTHORIZED_NEXT,
        "SCRATCH_DIRECTORY": SCRATCH_DIRECTORY,
        "_initial_progress": _initial_progress,
        "_helper": _stable_engine_helper,
    }
    try:
        for name, value in replacements.items():
            setattr(engine, name, value)
        yield
    finally:
        for name, value in saved.items():
            setattr(engine, name, value)


def _source_hashes() -> dict[str, str]:
    helper = _helper()
    paths = (
        THIS_RUNNER,
        THIS_TEST,
        manifest.THIS_RUNNER,
        manifest.THIS_TEST,
        parent.THIS_RUNNER,
        parent.THIS_TEST,
        engine.THIS_RUNNER,
        engine.THIS_TEST,
        manifest.parent.manifest.parent.manifest.POLICY_SOURCE,
        manifest.parent.manifest.parent.manifest.parent.diagnosis.manifest.STRICT_ATLAS_SOURCE,
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
        "adaptive_policy": lock["contract"]["adaptive_policy"],
    }


def _prepare_scratch(lock: dict) -> dict:
    helper = _helper()
    identity = _identity(lock)
    path = SCRATCH_DIRECTORY / "execution_identity.json"
    if SCRATCH_DIRECTORY.exists():
        if not path.exists() or helper._read(path) != identity:
            raise RuntimeError("cycle-readiness discrimination scratch mismatch")
    else:
        SCRATCH_DIRECTORY.mkdir(parents=True)
        helper._write_json(path, identity)
    return identity


def _git_blob_sha256(commit: str, relative: str) -> str:
    payload = subprocess.run(
        ("git", "show", f"{commit}:{relative}"),
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(payload).hexdigest()


def _validate_acquisition_scratch(lock: dict) -> dict:
    """Validate the completed immutable acquisition before reclassification."""

    helper = _helper()
    path = SCRATCH_DIRECTORY / "execution_identity.json"
    if not path.exists():
        raise RuntimeError("cycle-readiness discrimination scratch is absent")
    identity = helper._read(path)
    if (
        identity.get("work_package") != WORK_PACKAGE
        or identity.get("implementation_commit") != ACQUISITION_COMMIT
        or identity.get("implementation_tree") != ACQUISITION_TREE
        or identity.get("manifest_hashes") != lock["hashes"]
        or identity.get("adaptive_policy") != lock["contract"]["adaptive_policy"]
    ):
        raise RuntimeError("cycle-readiness discrimination acquisition changed")
    if helper._git("rev-parse", f"{ACQUISITION_COMMIT}^{{tree}}") != ACQUISITION_TREE:
        raise RuntimeError("cycle-readiness discrimination acquisition tree changed")
    for relative, expected in identity["source_hashes"].items():
        if _git_blob_sha256(ACQUISITION_COMMIT, relative) != expected:
            raise RuntimeError(f"acquisition source changed: {relative}")
    return identity


def _finite_forecast_range(forecasts: dict) -> tuple[list[float] | None, list[str]]:
    finite = []
    absent = []
    for name, item in forecasts.items():
        value = item["forecast_zero_time_seconds"]
        if value is None or not np.isfinite(value):
            absent.append(name)
        else:
            finite.append(float(value))
    return (
        None if not finite else [float(np.min(finite)), float(np.max(finite))],
        absent,
    )


def _forecast_bundle(
    lock: dict,
    accepted_times: np.ndarray,
    accepted_velocities: np.ndarray,
    accepted_speeds: np.ndarray,
) -> dict:
    prior_times = np.asarray(lock["geometry"]["times_seconds"], dtype=float)
    prior_velocities = np.asarray(
        lock["geometry"]["section_velocities_per_second"], dtype=float
    )
    prior_speeds = np.asarray(lock["geometry"]["speeds_per_second"], dtype=float)
    times = np.concatenate((prior_times, accepted_times))
    velocities = np.concatenate((prior_velocities, accepted_velocities))
    speeds = np.concatenate((prior_speeds, accepted_speeds))
    orientations = velocities / speeds
    raw = manifest._linear_forecasts(times, velocities)
    normalized = manifest._linear_forecasts(times, orientations)
    raw_range, raw_absent = _finite_forecast_range(raw)
    normalized_range, normalized_absent = _finite_forecast_range(normalized)
    secant_acceleration = np.diff(velocities) / np.diff(times)
    acceleration_left = np.concatenate(([-np.inf], secant_acceleration[:-1]))
    reversal_indices = np.flatnonzero(
        (acceleration_left < 0.0) & (secant_acceleration >= 0.0)
    )
    return {
        "sample_count": int(times.size),
        "raw_velocity_linear_forecasts": raw,
        "orientation_linear_forecasts": normalized,
        "raw_velocity_forecast_range_seconds": raw_range,
        "orientation_forecast_range_seconds": normalized_range,
        "raw_velocity_windows_without_forward_zero": raw_absent,
        "orientation_windows_without_forward_zero": normalized_absent,
        "terminal_secant_acceleration_per_second2": float(
            secant_acceleration[-1]
        ),
        "secant_acceleration_reversal_indices": reversal_indices.tolist(),
        "secant_acceleration_reversal_observed": bool(len(reversal_indices)),
        "diagnostic_only": True,
    }


def _classify_geometry(
    metrics: dict,
    arrays: dict[str, np.ndarray],
    lock: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    result_metrics = dict(metrics)
    result_arrays = dict(arrays)
    coordinates = np.asarray(arrays["accepted_endpoint_coordinates470"])
    rates = np.asarray(arrays["accepted_endpoint_coordinate_rates470_per_s"])
    spans = np.asarray(arrays["accepted_segment_seconds"])
    normal = np.asarray(arrays["section_normal470"])
    start = np.asarray(arrays["start_coordinate470"])
    sections = (
        (coordinates - start) @ normal if len(coordinates) else np.empty(0)
    )
    velocities = rates @ normal if len(rates) else np.empty(0)
    speeds = np.linalg.norm(rates, axis=1) if len(rates) else np.empty(0)
    orientations = velocities / speeds if len(speeds) else np.empty(0)
    times = manifest.INITIAL_ELAPSED_SECONDS + np.cumsum(spans)
    initial_velocity = float(
        lock["geometry"]["terminal_section_velocity_per_second"]
    )
    initial_orientation = float(lock["geometry"]["terminal_orientation_cosine"])
    velocity_left = np.concatenate(([initial_velocity], velocities[:-1]))
    orientation_left = np.concatenate(([initial_orientation], orientations[:-1]))
    turn_indices = np.flatnonzero((velocity_left > 0.0) & (velocities <= 0.0))
    orientation_turn_indices = np.flatnonzero(
        (orientation_left > 0.0) & (orientations <= 0.0)
    )
    if not np.array_equal(turn_indices, orientation_turn_indices):
        raise RuntimeError("section velocity/orientation zero equivalence failed")
    negative_indices = np.flatnonzero(sections < 0.0)
    turn_observed = bool(len(turn_indices))
    negative_observed = bool(len(negative_indices))
    forecast = (
        None if not len(times) else _forecast_bundle(lock, times, velocities, speeds)
    )
    result_arrays.update(
        {
            "accepted_endpoint_section_values": sections,
            "accepted_endpoint_section_velocities_per_s": velocities,
            "accepted_endpoint_speeds_per_s": speeds,
            "accepted_endpoint_orientation_cosines": orientations,
            "accepted_endpoint_elapsed_seconds": times,
            "section_turn_indices": turn_indices,
            "orientation_turn_indices": orientation_turn_indices,
            "negative_section_indices": negative_indices,
        }
    )
    gates = dict(metrics["gate_values"])
    gates.update(
        {
            "initial_section_value": float(lock["geometry"]["terminal_section_value"]),
            "initial_section_velocity_per_second": initial_velocity,
            "initial_orientation_cosine": initial_orientation,
            "terminal_section_value": (
                None if not len(sections) else float(sections[-1])
            ),
            "terminal_section_velocity_per_second": (
                None if not len(velocities) else float(velocities[-1])
            ),
            "terminal_speed_per_second": (
                None if not len(speeds) else float(speeds[-1])
            ),
            "terminal_orientation_cosine": (
                None if not len(orientations) else float(orientations[-1])
            ),
            "minimum_section_velocity_per_second": (
                None if not len(velocities) else float(np.min(velocities))
            ),
            "minimum_orientation_cosine": (
                None if not len(orientations) else float(np.min(orientations))
            ),
            "section_turning_point_observed": turn_observed,
            "orientation_turning_point_observed": bool(
                len(orientation_turn_indices)
            ),
            "section_velocity_orientation_zero_equivalence_passed": True,
            "section_negative_observed": negative_observed,
            "cycle_observed": False,
            "updated_forecast_bundle": forecast,
            "secant_acceleration_reversal_observed": bool(
                forecast is not None
                and forecast["secant_acceleration_reversal_observed"]
            ),
        }
    )
    result_metrics["gate_values"] = gates
    if metrics["passed"] and turn_observed:
        result_metrics.update(
            {
                "classification": TURN_CLASSIFICATION,
                "passed": True,
                "authorized_next": TURN_AUTHORIZED_NEXT,
            }
        )
    elif metrics["passed"]:
        result_metrics.update(
            {
                "classification": OPEN_CLASSIFICATION,
                "passed": True,
                "authorized_next": OPEN_AUTHORIZED_NEXT,
            }
        )
    return result_metrics, result_arrays


def _execute(
    lock: dict, identity: dict
) -> tuple[dict, dict[str, np.ndarray]]:
    with _engine_context():
        metrics, arrays = engine._execute(lock, identity)
    return _classify_geometry(metrics, arrays, lock)


def _update_catalog(summary: dict) -> None:
    helper = _helper()
    manifest_path = ROOT / "results/manifests/canonical_artifacts.csv"
    summary_path = ROOT / "results/manifests/canonical_summary.json"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    status = "SUPPORTED" if summary["passed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": helper._sha(path),
                    "scientific_status": status,
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


def _canonicalize(
    metrics: dict,
    arrays: dict[str, np.ndarray],
    lock: dict,
    acquisition_identity: dict,
    classification_identity: dict,
) -> dict:
    helper = _helper()
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("cycle-readiness discrimination result already exists")
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(
        CANONICAL_DIRECTORY / "forecast_discrimination_metrics.json", metrics
    )
    _save_npz(
        CANONICAL_DIRECTORY / "forecast_discrimination_arrays.npz", arrays
    )
    helper._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "manifest_hashes": lock["hashes"],
            "manifest_classification": lock["summary"]["classification"],
            "acquisition_identity": acquisition_identity,
            "classification_identity": classification_identity,
        },
    )
    values = metrics["gate_values"]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": metrics["passed"],
        "cycle_readiness_turn_bracketed": values[
            "section_turning_point_observed"
        ],
        "open_transient_extended": bool(
            metrics["passed"] and not values["section_turning_point_observed"]
        ),
        "secant_acceleration_reversal_observed": values[
            "secant_acceleration_reversal_observed"
        ],
        "new_accepted_segments": values["accepted_segments"],
        "terminal_elapsed_seconds": values["terminal_elapsed_seconds"],
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
            "implementation_commit": classification_identity[
                "implementation_commit"
            ],
            "implementation_tree": classification_identity["implementation_tree"],
            "source_hashes": classification_identity["source_hashes"],
            "acquisition_commit": acquisition_identity["implementation_commit"],
            "acquisition_tree": acquisition_identity["implementation_tree"],
            "acquisition_source_hashes": acquisition_identity["source_hashes"],
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
    forecast = values["updated_forecast_bundle"]
    raw_range = None if forecast is None else forecast["raw_velocity_forecast_range_seconds"]
    orientation_range = (
        None if forecast is None else forecast["orientation_forecast_range_seconds"]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Adaptive metric-chart cycle-readiness discrimination execution",
                "",
                f"Classification: `{metrics['classification']}`.",
                "",
                f"Accepted `{values['accepted_segments']}` of `{values['attempted_segments']}` attempted segments over `{values['new_accepted_horizon_seconds']:.6f}` s, reaching `{values['terminal_elapsed_seconds']:.6f}` s.",
                "",
                f"Terminal section velocity/orientation: `{values['terminal_section_velocity_per_second']}` / `{values['terminal_orientation_cosine']}`. Observed turn: `{values['section_turning_point_observed']}`; negative section: `{values['section_negative_observed']}`.",
                "",
                f"Diagnostic raw-velocity forecast range: `{raw_range}` s. Equivalent orientation forecast range: `{orientation_range}` s. Acceleration reversal observed: `{values['secant_acceleration_reversal_observed']}`.",
                "",
                f"Maximum endpoint/blind defects: `{values['maximum_accepted_endpoint_integral_defect']:.6e}` / `{values['maximum_accepted_blind_midpoint_rate_defect']:.6e}`. Exact fields/retractions/wall seconds: `{values['exact_free_field_calls']}` / `{values['retractions']}` / `{values['execution_wall_seconds']:.3f}`.",
                "",
                f"Checkpoint/suffix replay: `{values['all_accepted_checkpoint_roundtrips_bitwise']}` / `{values['suffix_history_replay_bitwise']}`. Velocity/orientation zero equivalence: `{values['section_velocity_orientation_zero_equivalence_passed']}`.",
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
    parser.add_argument("--finalize-existing", action="store_true")
    arguments = parser.parse_args()
    if arguments.run == arguments.finalize_existing:
        parser.error("choose exactly one of --run or --finalize-existing")
    lock = _validate_manifest(require_clean=True)
    if arguments.finalize_existing:
        acquisition_identity = _validate_acquisition_scratch(lock)
    else:
        acquisition_identity = _prepare_scratch(lock)
    classification_identity = _identity(lock)
    metrics, arrays = _execute(lock, acquisition_identity)
    summary = _canonicalize(
        metrics,
        arrays,
        lock,
        acquisition_identity,
        classification_identity,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
