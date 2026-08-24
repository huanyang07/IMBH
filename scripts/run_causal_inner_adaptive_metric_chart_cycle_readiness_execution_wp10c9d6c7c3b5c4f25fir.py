#!/usr/bin/env python3
"""Execute the bounded exact-field tranche toward cycle readiness."""

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

import run_causal_inner_adaptive_metric_chart_continuation_execution_wp10c9d6c7c3b5c4f25fip as engine  # noqa: E402
import run_causal_inner_adaptive_metric_chart_cycle_readiness_manifest_wp10c9d6c7c3b5c4f25fiq as manifest  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25fir"
TURN_CLASSIFICATION = "cycle_readiness_section_turn_bracketed"
OPEN_CLASSIFICATION = "cycle_readiness_open_transient_extended"
PHYSICAL_FAILURE_CLASSIFICATION = (
    "cycle_readiness_original_physical_gate_failed"
)
NUMERICAL_FAILURE_CLASSIFICATION = (
    "cycle_readiness_numerical_radius_or_restart_failed"
)
TURN_AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fis_adaptive_metric_chart_negative_section_manifest"
)
OPEN_AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fis_adaptive_metric_chart_cycle_readiness_reforecast_manifest"
)
ARTIFACT = (
    "causal_inner_adaptive_metric_chart_cycle_readiness_execution_"
    "wp10c9d6c7c3b5c4f25fir"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ADAPTIVE_METRIC_CHART_"
    "CYCLE_READINESS_EXECUTION_WP10C9D6C7C3B5C4F25FIR_2026-08-24.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_adaptive_metric_chart_cycle_readiness_execution_"
    "wp10c9d6c7c3b5c4f25fir.py"
)
THIS_TEST = (
    "tests/test_causal_inner_adaptive_metric_chart_cycle_readiness_execution_"
    "wp10c9d6c7c3b5c4f25fir.py"
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
        manifest.CANONICAL_DIRECTORY / "cycle_readiness_contract.json"
    )
    cost = helper._read(manifest.CANONICAL_DIRECTORY / "cost_projection.json")
    geometry = helper._read(
        manifest.CANONICAL_DIRECTORY / "geometry_forecast.json"
    )
    provenance = helper._read(manifest.CANONICAL_DIRECTORY / "provenance.json")
    if (
        summary["classification"] != manifest.CLASSIFICATION
        or not summary["passed"]
        or not summary["definitions_only"]
        or not summary["cycle_readiness_execution_authorized"]
        or summary["cycle_readiness_execution_executed"]
        or summary["complete_cycle_execution_authorized"]
        or summary["reduced_slow_evolution_authorized"]
        or summary["authorized_next"] != manifest.AUTHORIZED_NEXT
        or contract["authorized_execution"] != manifest.AUTHORIZED_NEXT
        or contract["scope"]["maximum_accepted_segments"] != 32
        or contract["scope"]["maximum_attempted_segments"] != 40
        or contract["scope"]["maximum_exact_free_field_calls"] != 48
        or contract["scope"]["maximum_retractions"] != 56
        or geometry["cycle_observed"]
        or geometry["section_negative_observed"]
        or geometry["terminal_section_velocity_per_second"] <= 0.0
        or not cost["cost_gate_passed"]
    ):
        raise RuntimeError("cycle-readiness authorization changed")
    for relative, frozen_hash in provenance["source_hashes"].items():
        if helper._sha(ROOT / relative) != frozen_hash:
            raise RuntimeError(f"frozen cycle-readiness source changed: {relative}")
    parent_lock = manifest._validate_parent(require_clean=False)
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("cycle-readiness execution requires a clean tracked tree")
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
)


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
        engine.THIS_RUNNER,
        engine.THIS_TEST,
        manifest.parent.manifest.POLICY_SOURCE,
        manifest.parent.manifest.parent.diagnosis.manifest.STRICT_ATLAS_SOURCE,
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
            raise RuntimeError("cycle-readiness scratch identity mismatch")
    else:
        SCRATCH_DIRECTORY.mkdir(parents=True)
        helper._write_json(path, identity)
    return identity


def _recent_forecast(
    lock: dict,
    accepted_times: np.ndarray,
    accepted_velocities: np.ndarray,
) -> dict:
    prior_velocities = np.asarray(
        lock["geometry"]["recent_section_velocities_per_second"], dtype=float
    )
    prior_spans = np.asarray(
        manifest._parent_arrays()["accepted_segment_seconds"], dtype=float
    )
    prior_times = (
        manifest.INITIAL_ELAPSED_SECONDS
        - float(np.sum(prior_spans))
        + np.cumsum(prior_spans)
    )
    if prior_times.shape != prior_velocities.shape:
        raise RuntimeError("frozen readiness geometry history changed")
    times = np.concatenate((prior_times, accepted_times))
    velocities = np.concatenate((prior_velocities, accepted_velocities))
    count = min(16, len(times))
    slope, intercept = np.polyfit(times[-count:], velocities[-count:], 1)
    zero_time = float(-intercept / slope) if slope < 0.0 else None
    return {
        "sample_count": count,
        "section_acceleration_per_second2": float(slope),
        "forecast_zero_velocity_time_seconds": zero_time,
        "forecast_additional_seconds": (
            None
            if zero_time is None
            else zero_time - float(accepted_times[-1])
        ),
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
    times = manifest.INITIAL_ELAPSED_SECONDS + np.cumsum(spans)
    initial_velocity = float(
        lock["geometry"]["terminal_section_velocity_per_second"]
    )
    left = np.concatenate(([initial_velocity], velocities[:-1]))
    turn_indices = np.flatnonzero((left > 0.0) & (velocities <= 0.0))
    negative_indices = np.flatnonzero(sections < 0.0)
    turn_observed = bool(len(turn_indices))
    negative_observed = bool(len(negative_indices))
    result_arrays.update(
        {
            "accepted_endpoint_section_values": sections,
            "accepted_endpoint_section_velocities_per_s": velocities,
            "accepted_endpoint_speeds_per_s": speeds,
            "accepted_endpoint_elapsed_seconds": times,
            "section_turn_indices": turn_indices,
            "negative_section_indices": negative_indices,
        }
    )
    gates = dict(metrics["gate_values"])
    gates.update(
        {
            "initial_section_value": float(
                lock["geometry"]["terminal_section_value"]
            ),
            "initial_section_velocity_per_second": initial_velocity,
            "terminal_section_value": (
                None if not len(sections) else float(sections[-1])
            ),
            "terminal_section_velocity_per_second": (
                None if not len(velocities) else float(velocities[-1])
            ),
            "terminal_speed_per_second": (
                None if not len(speeds) else float(speeds[-1])
            ),
            "minimum_section_velocity_per_second": (
                None if not len(velocities) else float(np.min(velocities))
            ),
            "section_turning_point_observed": turn_observed,
            "section_negative_observed": negative_observed,
            "cycle_observed": False,
            "updated_turning_forecast": (
                None
                if not len(times)
                else _recent_forecast(lock, times, velocities)
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
    identity: dict,
) -> dict:
    helper = _helper()
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("cycle-readiness result already exists")
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(
        CANONICAL_DIRECTORY / "cycle_readiness_metrics.json", metrics
    )
    _save_npz(CANONICAL_DIRECTORY / "cycle_readiness_arrays.npz", arrays)
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
        "cycle_readiness_turn_bracketed": values[
            "section_turning_point_observed"
        ],
        "open_transient_extended": bool(
            metrics["passed"] and not values["section_turning_point_observed"]
        ),
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
    forecast = values["updated_turning_forecast"]
    terminal_section = values["terminal_section_value"]
    terminal_velocity = values["terminal_section_velocity_per_second"]
    terminal_section_text = (
        "None" if terminal_section is None else f"{terminal_section:.6e}"
    )
    terminal_velocity_text = (
        "None" if terminal_velocity is None else f"{terminal_velocity:.6e}"
    )
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Adaptive metric-chart cycle-readiness execution",
                "",
                f"Classification: `{metrics['classification']}`.",
                "",
                f"Accepted `{values['accepted_segments']}` of `{values['attempted_segments']}` attempted segments over `{values['new_accepted_horizon_seconds']:.6f}` s, reaching `{values['terminal_elapsed_seconds']:.6f}` s.",
                "",
                f"Section value/velocity at the endpoint: `{terminal_section_text}` / `{terminal_velocity_text}` per second. Turning point observed: `{values['section_turning_point_observed']}`; negative section observed: `{values['section_negative_observed']}`.",
                "",
                f"Updated diagnostic zero-velocity forecast: `{None if forecast is None else forecast['forecast_zero_velocity_time_seconds']}` s. This does not establish a cycle.",
                "",
                f"Maximum endpoint/blind defects: `{values['maximum_accepted_endpoint_integral_defect']:.6e}` / `{values['maximum_accepted_blind_midpoint_rate_defect']:.6e}`. Exact fields/retractions/wall seconds: `{values['exact_free_field_calls']}` / `{values['retractions']}` / `{values['execution_wall_seconds']:.3f}`.",
                "",
                f"Checkpoint/suffix replay: `{values['all_accepted_checkpoint_roundtrips_bitwise']}` / `{values['suffix_history_replay_bitwise']}`.",
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
