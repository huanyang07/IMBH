#!/usr/bin/env python3
"""Execute one prospective coupled physical-memory chart boundary step."""

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

import run_causal_inner_coupled_physical_memory_metric_chart_recovery_manifest_wp10c9d6c7c3b5c4f25fiy as manifest  # noqa: E402


parent = manifest.parent
engine = parent.engine
suffix = engine.suffix
SCHEMA_VERSION = 1
WORK_PACKAGE = manifest.AUTHORIZED_NEXT
PASS_CLASSIFICATION = (
    "coupled_physical_memory_metric_chart_boundary_recovery_passed"
)
PHASE_FAILURE_CLASSIFICATION = (
    "coupled_physical_memory_metric_chart_boundary_phase_failed"
)
PHYSICAL_FAILURE_CLASSIFICATION = (
    "coupled_physical_memory_metric_chart_boundary_physical_gate_failed"
)
NUMERICAL_FAILURE_CLASSIFICATION = (
    "coupled_physical_memory_metric_chart_boundary_numerical_or_restart_failed"
)
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fiza_"
    "tangent_phase_lap_stage1_resume_manifest"
)
ARTIFACT = (
    "causal_inner_coupled_physical_memory_metric_chart_boundary_recovery_"
    "execution_wp10c9d6c7c3b5c4f25fiz"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_COUPLED_PHYSICAL_MEMORY_METRIC_"
    "CHART_BOUNDARY_RECOVERY_EXECUTION_WP10C9D6C7C3B5C4F25FIZ_2026-08-25.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_coupled_physical_memory_metric_chart_boundary_"
    "recovery_execution_wp10c9d6c7c3b5c4f25fiz.py"
)
THIS_TEST = (
    "tests/test_causal_inner_coupled_physical_memory_metric_chart_boundary_"
    "recovery_execution_wp10c9d6c7c3b5c4f25fiz.py"
)

# Adapter constants consumed by the certified continuation engine.
INITIAL_ELAPSED_SECONDS = 0.17850000000000013
MINIMUM_SEGMENT_SECONDS = manifest.RECOVERY_SEGMENT_SECONDS
MAXIMUM_SEGMENT_SECONDS = manifest.RECOVERY_SEGMENT_SECONDS
GROWTH_FACTOR = 2.0
ACCEPTED_SEGMENTS_BEFORE_GROWTH = 4
BLIND_MIDPOINT_FREQUENCY = 8
MAXIMUM_ACCEPTED_SEGMENTS = 1
MAXIMUM_ATTEMPTED_SEGMENTS = 1
MAXIMUM_EXACT_FREE_FIELD_CALLS = 1
MAXIMUM_RETRACTIONS = 1
MAXIMUM_EXECUTION_WALL_HOURS = 1.0
MAXIMUM_ENDPOINT_INTEGRAL_DEFECT = parent.MAXIMUM_ENDPOINT_INTEGRAL_DEFECT
MAXIMUM_BLIND_MIDPOINT_RATE_DEFECT = parent.MAXIMUM_BLIND_MIDPOINT_RATE_DEFECT


_BASE_HELPER_MODULE = manifest._helper()
_ORIGINAL_ENGINE_ATTEMPT = parent._ORIGINAL_ENGINE_ATTEMPT


def _helper():
    return manifest._helper()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def _contract() -> dict:
    return _helper()._read(manifest.CANONICAL_DIRECTORY / "recovery_contract.json")


def _seed() -> dict[str, np.ndarray]:
    return _load_npz(manifest.CANONICAL_DIRECTORY / "recovery_seed.npz")


def _source_hashes() -> dict[str, str]:
    helper = _helper()
    paths = (
        THIS_RUNNER,
        THIS_TEST,
        manifest.THIS_RUNNER,
        manifest.THIS_TEST,
        parent.THIS_RUNNER,
        parent.THIS_TEST,
        parent.manifest.THIS_RUNNER,
        parent.manifest.THIS_TEST,
        engine.THIS_RUNNER,
        engine.THIS_TEST,
        suffix.THIS_RUNNER,
        engine.execution.source.THIS_RUNNER,
    )
    return {name: helper._sha(ROOT / name) for name in paths}


def _validate_manifest(*, require_clean: bool) -> dict:
    helper = _helper()
    hashes = helper._validate_checksums(manifest.CANONICAL_DIRECTORY)
    summary = helper._read(manifest.CANONICAL_DIRECTORY / "summary.json")
    metrics = helper._read(
        manifest.CANONICAL_DIRECTORY / "chart_recovery_metrics.json"
    )
    contract = _contract()
    provenance = helper._read(manifest.CANONICAL_DIRECTORY / "provenance.json")
    scope = contract["authorized_scope"]
    gates = contract["binding_recovery_gates"]
    if (
        summary["classification"] != manifest.CLASSIFICATION
        or not summary["passed"]
        or not summary["definitions_only"]
        or not summary["boundary_recovery_execution_authorized"]
        or summary["boundary_recovery_execution_executed"]
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["complete_cycle_execution_authorized"]
        or summary["reduced_slow_evolution_authorized"]
        or metrics["classification"] != manifest.CLASSIFICATION
        or not metrics["passed"]
        or metrics["selected_block_sizes"] != [442, 28]
        or contract["authorized_execution"] != WORK_PACKAGE
        or contract["selected_partition"] != [442, 28]
        or scope["maximum_new_accepted_segments"] != 1
        or scope["maximum_retractions"] != 1
        or scope["maximum_exact_free_field_calls"] != 1
        or scope["segment_seconds"] != manifest.RECOVERY_SEGMENT_SECONDS
        or gates["maximum_metric_jacobian_condition"] != 10.0
        or gates["maximum_metric_augmented_condition"] != 10.0
    ):
        raise RuntimeError("coupled chart recovery authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if helper._sha(ROOT / relative) != expected:
            raise RuntimeError(f"coupled chart recovery source changed: {relative}")
    manifest._validate_parent(require_clean=False)
    if require_clean and helper._git(
        "status", "--short", "--untracked-files=no"
    ):
        raise RuntimeError("boundary recovery requires a clean tracked tree")
    return {
        "hashes": hashes,
        "summary": summary,
        "metrics": metrics,
        "contract": contract,
        "provenance": provenance,
    }


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
            raise RuntimeError("boundary recovery scratch mismatch")
    else:
        SCRATCH_DIRECTORY.mkdir(parents=True)
        helper._write_json(path, identity)
    return identity


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


def _phase_history() -> np.ndarray:
    rates = _seed()["accepted_endpoint_coordinate_rates470_per_s"]
    if rates.shape != (42, 470):
        raise RuntimeError("boundary recovery phase history changed")
    return rates[-parent.holdout.manifest.SELECTED_WINDOW :].copy()


def _prior_accumulation() -> dict:
    seed = _seed()
    return {
        "cumulative_phase_advance_radians": float(
            seed["unwrapped_phase_advance_radians"]
        ),
        "cumulative_metric_path_length": float(
            seed["accumulated_metric_path_length"]
        ),
        "registered_section_value": float(
            seed["accepted_registered_section_values"][-1]
        ),
    }


def _phase_attempt(*, progress: dict, inputs: dict, exact_chart):
    return parent._phase_attempt(
        progress=progress,
        inputs=inputs,
        exact_chart=exact_chart,
    )


def _one_record_replay(
    records: list[tuple[dict, dict[str, np.ndarray]]],
    final_progress: dict,
) -> tuple[bool, int | None]:
    if len(records) != 1 or not records[0][0]["accepted"]:
        return False, None
    metrics, arrays = records[0]
    initial = _initial_progress()
    candidate = engine.execution._variable_step_ab2(
        initial["current_coordinate"],
        initial["current_rate"],
        initial["previous_rate"],
        initial["next_span"],
        initial["previous_span"],
    )
    checkpoint_path = (
        SCRATCH_DIRECTORY
        / f"attempt_{metrics['attempt_index']:04d}"
        / "accepted_checkpoint.npz"
    )
    if not checkpoint_path.exists():
        return False, None
    checkpoint = _load_npz(checkpoint_path)
    expected = engine._checkpoint_arrays(final_progress)
    replay = bool(
        np.array_equal(candidate, arrays["candidate_target470"])
        and set(checkpoint) == set(expected)
        and all(np.array_equal(checkpoint[name], expected[name]) for name in expected)
    )
    return replay, int(metrics["attempt_index"])


def _stable_engine_helper():
    return _BASE_HELPER_MODULE


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
    "_restart_replay",
)


@contextmanager
def _execution_context():
    engine_saved = {name: getattr(engine, name) for name in _ENGINE_NAMES}
    parent_saved = {
        "_seed": parent._seed,
        "_accepted_phase_history": parent._accepted_phase_history,
        "_prior_accumulation": parent._prior_accumulation,
    }
    block_sizes_saved = suffix._block_sizes
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
        "_restart_replay": _one_record_replay,
    }
    try:
        for name, value in replacements.items():
            setattr(engine, name, value)
        parent._seed = _seed
        parent._accepted_phase_history = _phase_history
        parent._prior_accumulation = _prior_accumulation
        suffix._block_sizes = lambda: manifest.COUPLED_PHYSICAL_MEMORY_BLOCKS
        yield
    finally:
        for name, value in engine_saved.items():
            setattr(engine, name, value)
        for name, value in parent_saved.items():
            setattr(parent, name, value)
        suffix._block_sizes = block_sizes_saved


def _records() -> list[dict]:
    records = []
    for directory in sorted(SCRATCH_DIRECTORY.glob("attempt_[0-9][0-9][0-9][0-9]")):
        path = directory / "attempt.json"
        if path.exists():
            records.append(_helper()._read(path))
    return records


def _classify(
    metrics: dict,
    arrays: dict[str, np.ndarray],
) -> tuple[dict, dict[str, np.ndarray]]:
    result_metrics = dict(metrics)
    result_arrays = dict(arrays)
    records = _records()
    record = records[0] if len(records) == 1 else None
    phase = None if record is None else record.get("phase_geometry")
    recurrence = None if record is None else record.get("recurrence_geometry")
    field = None if record is None else record.get("endpoint_field")
    seed = _seed()
    candidate_matches = bool(
        record is not None
        and np.array_equal(
            _load_npz(
                SCRATCH_DIRECTORY
                / f"attempt_{record['attempt_index']:04d}"
                / "attempt.npz"
            )["candidate_target470"],
            seed["next_candidate_target470"],
        )
    )
    selected_blocks = (
        None
        if field is None
        else field["metric_chart"].get("block_sizes")
    )
    existing_pass = bool(metrics["passed"])
    phase_pass = bool(phase is not None and phase["passed"])
    recurrence_pass = recurrence is not None
    field_pass = bool(
        field is not None
        and field["physical_passed"]
        and field["metric_chart"]["metric_jacobian_condition_number"] <= 10.0
        and field["metric_chart"]["metric_augmented_condition_number"] <= 10.0
        and selected_blocks == [442, 28]
    )
    passed = bool(
        existing_pass
        and len(records) == 1
        and record["accepted"]
        and candidate_matches
        and phase_pass
        and recurrence_pass
        and field_pass
        and metrics["gate_values"]["accepted_segments"] == 1
        and metrics["gate_values"]["exact_free_field_calls"] == 1
        and metrics["gate_values"]["retractions"] == 1
        and metrics["gate_values"]["all_accepted_checkpoint_roundtrips_bitwise"]
        and metrics["gate_values"]["suffix_history_replay_bitwise"]
    )
    if passed:
        classification = PASS_CLASSIFICATION
        authorized_next = AUTHORIZED_NEXT
    elif phase is not None and not phase_pass:
        classification = PHASE_FAILURE_CLASSIFICATION
        authorized_next = None
    elif record is not None and record["physical_failure"]:
        classification = PHYSICAL_FAILURE_CLASSIFICATION
        authorized_next = None
    else:
        classification = NUMERICAL_FAILURE_CLASSIFICATION
        authorized_next = None
    values = dict(metrics["gate_values"])
    values.update(
        {
            "candidate_target_matches_frozen_seed_bitwise": candidate_matches,
            "selected_metric_block_sizes": selected_blocks,
            "phase_geometry_passed": phase_pass,
            "phase_increment": None if phase is None else phase["phase_increment"],
            "cumulative_phase_advance_radians": (
                float(seed["unwrapped_phase_advance_radians"])
                if recurrence is None
                else recurrence["cumulative_phase_advance_radians"]
            ),
            "cumulative_metric_path_length": (
                float(seed["accumulated_metric_path_length"])
                if recurrence is None
                else recurrence["cumulative_metric_path_length"]
            ),
            "phase_lap_observed": bool(
                recurrence is not None and recurrence["phase_lap_observed"]
            ),
            "coarse_recurrence_candidate_observed": bool(
                recurrence is not None
                and recurrence["coarse_recurrence_candidate"]
            ),
            "registered_section_value": (
                None
                if recurrence is None
                else recurrence["endpoint_registered_section_value"]
            ),
            "return_distance_over_path_length": (
                None
                if recurrence is None
                else recurrence["endpoint_return_distance_over_path_length"]
            ),
        }
    )
    result_metrics.update(
        {
            "classification": classification,
            "passed": passed,
            "authorized_next": authorized_next,
            "gate_values": values,
        }
    )
    result_arrays.update(
        {
            "prior_accepted_endpoint_coordinates470": seed[
                "accepted_endpoint_coordinates470"
            ],
            "prior_accepted_endpoint_coordinate_rates470_per_s": seed[
                "accepted_endpoint_coordinate_rates470_per_s"
            ],
            "prior_accepted_phase_increments": seed["accepted_phase_increments"],
            "new_phase_increment": np.asarray(
                np.nan if phase is None else phase["phase_increment"]
            ),
            "cumulative_phase_advance_radians": np.asarray(
                values["cumulative_phase_advance_radians"]
            ),
            "cumulative_metric_path_length": np.asarray(
                values["cumulative_metric_path_length"]
            ),
        }
    )
    return result_metrics, result_arrays


def _execute(lock: dict, identity: dict) -> tuple[dict, dict[str, np.ndarray]]:
    with _execution_context():
        metrics, arrays = engine._execute(lock, identity)
    return _classify(metrics, arrays)


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
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("boundary recovery result already exists")
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(
        CANONICAL_DIRECTORY / "boundary_recovery_metrics.json", metrics
    )
    _save_npz(CANONICAL_DIRECTORY / "boundary_recovery_arrays.npz", arrays)
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
        "cumulative_phase_advance_radians": values[
            "cumulative_phase_advance_radians"
        ],
        "phase_lap_observed": values["phase_lap_observed"],
        "coarse_recurrence_candidate_observed": values[
            "coarse_recurrence_candidate_observed"
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
                "# Coupled physical-memory metric-chart boundary recovery",
                "",
                f"Classification: `{metrics['classification']}`.",
                "",
                f"Accepted `{values['accepted_segments']}` of `{values['attempted_segments']}` prospective boundary attempts, reaching `{values['terminal_elapsed_seconds']:.6f}` s.",
                "",
                f"The exact endpoint used metric blocks `{values['selected_metric_block_sizes']}`. Maximum raw/metric conditions were `{values['maximum_raw_coordinate_jacobian_condition']:.6e}` / `{values['maximum_metric_coordinate_jacobian_condition']:.6e}` under the unchanged 10.0 metric gate.",
                "",
                f"Endpoint defect was `{values['maximum_accepted_endpoint_integral_defect']:.6e}`. Reconstruction, height, optical-depth, ledger, checkpoint, suffix-replay, phase, and recurrence gates remained binding and passed: `{metrics['passed']}`.",
                "",
                f"Cumulative registered phase is `{values['cumulative_phase_advance_radians']:.9f}` rad. Phase lap: `{values['phase_lap_observed']}`. Coarse recurrence candidate: `{values['coarse_recurrence_candidate_observed']}`.",
                "",
                "This one endpoint does not retroactively pass the rejected stage-1 run. It only tests the prospectively selected chart recovery.",
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
