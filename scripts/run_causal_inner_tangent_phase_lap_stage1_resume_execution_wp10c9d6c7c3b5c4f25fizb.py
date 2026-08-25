#!/usr/bin/env python3
"""Execute the five-endpoint resume that completes phase-lap stage 1."""

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

import run_causal_inner_tangent_phase_lap_stage1_resume_manifest_wp10c9d6c7c3b5c4f25fiza as manifest  # noqa: E402


recovery = manifest.parent
phase = recovery.parent
engine = phase.engine
suffix = engine.suffix
SCHEMA_VERSION = 1
WORK_PACKAGE = manifest.AUTHORIZED_NEXT
PASS_CLASSIFICATION = (
    "tangent_phase_lap_acquisition_stage1_completed_after_chart_recovery"
)
COARSE_RECURRENCE_CLASSIFICATION = (
    "coarse_tangent_phase_recurrence_candidate_observed_during_stage1_resume"
)
OPEN_CLASSIFICATION = "tangent_phase_lap_without_coarse_state_recurrence"
PHASE_FAILURE_CLASSIFICATION = "tangent_phase_lap_stage1_resume_geometry_failed"
PHYSICAL_FAILURE_CLASSIFICATION = (
    "tangent_phase_lap_stage1_resume_physical_gate_failed"
)
NUMERICAL_FAILURE_CLASSIFICATION = (
    "tangent_phase_lap_stage1_resume_numerical_or_restart_failed"
)
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizc_"
    "tangent_phase_lap_recurrence_stage2_manifest"
)
COARSE_RECURRENCE_AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizc_"
    "registered_tangent_phase_return_refinement_manifest"
)
ARTIFACT = (
    "causal_inner_tangent_phase_lap_stage1_resume_execution_"
    "wp10c9d6c7c3b5c4f25fizb"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_TANGENT_PHASE_LAP_STAGE1_"
    "RESUME_EXECUTION_WP10C9D6C7C3B5C4F25FIZB_2026-08-25.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_tangent_phase_lap_stage1_resume_execution_"
    "wp10c9d6c7c3b5c4f25fizb.py"
)
THIS_TEST = (
    "tests/test_causal_inner_tangent_phase_lap_stage1_resume_execution_"
    "wp10c9d6c7c3b5c4f25fizb.py"
)

# Adapter constants consumed by the certified continuation engine.
INITIAL_ELAPSED_SECONDS = 0.17875000000000013
MINIMUM_SEGMENT_SECONDS = manifest.SEGMENT_SECONDS
MAXIMUM_SEGMENT_SECONDS = manifest.SEGMENT_SECONDS
GROWTH_FACTOR = 2.0
ACCEPTED_SEGMENTS_BEFORE_GROWTH = 4
BLIND_MIDPOINT_FREQUENCY = manifest.BLIND_MIDPOINT_FREQUENCY
MAXIMUM_ACCEPTED_SEGMENTS = manifest.REMAINING_ACCEPTED
MAXIMUM_ATTEMPTED_SEGMENTS = manifest.MAXIMUM_ATTEMPTS
MAXIMUM_EXACT_FREE_FIELD_CALLS = 6
MAXIMUM_RETRACTIONS = 6
MAXIMUM_EXECUTION_WALL_HOURS = manifest.MAXIMUM_WALL_HOURS
MAXIMUM_ENDPOINT_INTEGRAL_DEFECT = phase.MAXIMUM_ENDPOINT_INTEGRAL_DEFECT
MAXIMUM_BLIND_MIDPOINT_RATE_DEFECT = phase.MAXIMUM_BLIND_MIDPOINT_RATE_DEFECT


_BASE_HELPER_MODULE = manifest._helper()


def _helper():
    return manifest._helper()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def _contract() -> dict:
    return _helper()._read(manifest.CANONICAL_DIRECTORY / "resume_contract.json")


def _seed() -> dict[str, np.ndarray]:
    return _load_npz(manifest.CANONICAL_DIRECTORY / "resume_seed.npz")


def _source_hashes() -> dict[str, str]:
    helper = _helper()
    paths = (
        THIS_RUNNER,
        THIS_TEST,
        manifest.THIS_RUNNER,
        manifest.THIS_TEST,
        recovery.THIS_RUNNER,
        recovery.THIS_TEST,
        recovery.manifest.THIS_RUNNER,
        recovery.manifest.THIS_TEST,
        phase.THIS_RUNNER,
        phase.THIS_TEST,
        phase.manifest.THIS_RUNNER,
        phase.manifest.THIS_TEST,
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
    metrics = helper._read(manifest.CANONICAL_DIRECTORY / "resume_metrics.json")
    contract = _contract()
    provenance = helper._read(manifest.CANONICAL_DIRECTORY / "provenance.json")
    scope = contract["scope"]
    gates = contract["binding_completion_gates"]
    if (
        summary["classification"] != manifest.CLASSIFICATION
        or not summary["passed"]
        or not summary["definitions_only"]
        or not summary["stage1_resume_execution_authorized"]
        or summary["stage1_resume_execution_executed"]
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["complete_cycle_execution_authorized"]
        or summary["reduced_slow_evolution_authorized"]
        or metrics["classification"] != manifest.CLASSIFICATION
        or not metrics["passed"]
        or contract["authorized_execution"] != WORK_PACKAGE
        or scope["prior_accepted_endpoints"] != 43
        or scope["new_accepted_endpoints"] != 5
        or scope["maximum_attempted_endpoints"] != 5
        or scope["maximum_exact_free_field_calls"] != 6
        or scope["maximum_retractions"] != 6
        or scope["blind_midpoint_segment_numbers"] != [240]
        or contract["computational_chart"]["block_sizes"] != [442, 28]
        or contract["computational_chart"]
        ["maximum_metric_and_augmented_condition"]
        != 10.0
        or gates["combined_accepted_endpoints"] != 48
    ):
        raise RuntimeError("stage1 resume authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if helper._sha(ROOT / relative) != expected:
            raise RuntimeError(f"stage1 resume source changed: {relative}")
    manifest._validate_parent(require_clean=False)
    if require_clean and helper._git(
        "status", "--short", "--untracked-files=no"
    ):
        raise RuntimeError("stage1 resume requires a clean tracked tree")
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
            raise RuntimeError("stage1 resume scratch mismatch")
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


def _accepted_attempts() -> list[tuple[dict, dict[str, np.ndarray]]]:
    accepted = []
    for directory in sorted(SCRATCH_DIRECTORY.glob("attempt_[0-9][0-9][0-9][0-9]")):
        metrics_path = directory / "attempt.json"
        arrays_path = directory / "attempt.npz"
        if not metrics_path.exists() or not arrays_path.exists():
            continue
        metrics = _helper()._read(metrics_path)
        if (
            metrics.get("accepted")
            and metrics.get("phase_geometry") is not None
            and metrics.get("recurrence_geometry") is not None
        ):
            accepted.append((metrics, _load_npz(arrays_path)))
    return accepted


def _phase_history() -> np.ndarray:
    history = _seed()["accepted_endpoint_coordinate_rates470_per_s"].copy()
    new = [
        arrays["accepted_coordinate_rate470_per_s"]
        for _metrics, arrays in _accepted_attempts()
    ]
    if new:
        history = np.vstack((history, np.stack(new)))
    return history[-phase.holdout.manifest.SELECTED_WINDOW :]


def _prior_accumulation() -> dict:
    seed = _seed()
    prior = {
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
    for metrics, _arrays in _accepted_attempts():
        recurrence = metrics["recurrence_geometry"]
        prior = {
            "cumulative_phase_advance_radians": recurrence[
                "cumulative_phase_advance_radians"
            ],
            "cumulative_metric_path_length": recurrence[
                "cumulative_metric_path_length"
            ],
            "registered_section_value": recurrence[
                "endpoint_registered_section_value"
            ],
        }
    return prior


def _phase_attempt(*, progress: dict, inputs: dict, exact_chart):
    return phase._phase_attempt(
        progress=progress,
        inputs=inputs,
        exact_chart=exact_chart,
    )


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
)


@contextmanager
def _execution_context():
    engine_saved = {name: getattr(engine, name) for name in _ENGINE_NAMES}
    phase_saved = {
        "_seed": phase._seed,
        "_accepted_phase_history": phase._accepted_phase_history,
        "_prior_accumulation": phase._prior_accumulation,
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
    }
    try:
        for name, value in replacements.items():
            setattr(engine, name, value)
        phase._seed = _seed
        phase._accepted_phase_history = _phase_history
        phase._prior_accumulation = _prior_accumulation
        suffix._block_sizes = lambda: manifest.METRIC_BLOCK_SIZES
        yield
    finally:
        for name, value in engine_saved.items():
            setattr(engine, name, value)
        for name, value in phase_saved.items():
            setattr(phase, name, value)
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
    seed = _seed()
    records = _records()
    accepted = [
        item
        for item in records
        if item.get("accepted") and item.get("recurrence_geometry") is not None
    ]
    phase_records = [
        item for item in records if item.get("phase_geometry") is not None
    ]
    recurrence = [item["recurrence_geometry"] for item in accepted]
    final = recurrence[-1] if recurrence else None
    phase_failed = any(not item["phase_geometry"]["passed"] for item in phase_records)
    candidate = any(item["coarse_recurrence_candidate"] for item in recurrence)
    phase_lap = any(item["phase_lap_observed"] for item in recurrence)
    combined = manifest.COMBINED_ACCEPTED + len(accepted)
    cumulative_phase = (
        float(seed["unwrapped_phase_advance_radians"])
        if final is None
        else final["cumulative_phase_advance_radians"]
    )
    all_blocks = bool(
        records
        and all(
            item.get("endpoint_field") is not None
            and item["endpoint_field"]["metric_chart"]["block_sizes"] == [442, 28]
            for item in records
            if item.get("endpoint_field") is not None
        )
    )
    completion = bool(
        metrics["passed"]
        and len(records) == MAXIMUM_ATTEMPTED_SEGMENTS
        and len(accepted) == MAXIMUM_ACCEPTED_SEGMENTS
        and combined == manifest.ORIGINAL_STAGE_TARGET
        and cumulative_phase >= manifest.MINIMUM_FINAL_CUMULATIVE_PHASE
        and phase_records
        and all(item["phase_geometry"]["passed"] for item in phase_records)
        and all_blocks
    )
    if candidate:
        classification = COARSE_RECURRENCE_CLASSIFICATION
        passed = True
        authorized_next = COARSE_RECURRENCE_AUTHORIZED_NEXT
    elif phase_lap:
        classification = OPEN_CLASSIFICATION
        passed = False
        authorized_next = None
    elif completion:
        classification = PASS_CLASSIFICATION
        passed = True
        authorized_next = AUTHORIZED_NEXT
    elif phase_failed:
        classification = PHASE_FAILURE_CLASSIFICATION
        passed = False
        authorized_next = None
    elif any(item.get("physical_failure") for item in records):
        classification = PHYSICAL_FAILURE_CLASSIFICATION
        passed = False
        authorized_next = None
    else:
        classification = NUMERICAL_FAILURE_CLASSIFICATION
        passed = False
        authorized_next = None
    values = dict(metrics["gate_values"])
    values.update(
        {
            "prior_accepted_phase_endpoints": manifest.COMBINED_ACCEPTED,
            "new_accepted_phase_endpoints": len(accepted),
            "combined_accepted_phase_endpoints": combined,
            "phase_predictions_evaluated": len(phase_records),
            "all_phase_geometry_gates_passed": bool(
                phase_records
                and all(item["phase_geometry"]["passed"] for item in phase_records)
            ),
            "all_endpoint_metric_blocks_are_442_plus_28": all_blocks,
            "minimum_new_phase_increment": min(
                (item["phase_geometry"]["phase_increment"] for item in accepted),
                default=None,
            ),
            "maximum_new_phase_increment": max(
                (item["phase_geometry"]["phase_increment"] for item in accepted),
                default=None,
            ),
            "maximum_new_phase_radial_defect": max(
                (
                    item["phase_geometry"]["relative_radial_defect"]
                    for item in accepted
                ),
                default=None,
            ),
            "maximum_new_phase_out_of_plane_defect": max(
                (
                    item["phase_geometry"]["out_of_plane_defect"]
                    for item in accepted
                ),
                default=None,
            ),
            "maximum_new_phase_direction_prediction_defect_radians": max(
                (
                    item["phase_geometry"][
                        "direction_prediction_defect_radians"
                    ]
                    for item in accepted
                ),
                default=None,
            ),
            "cumulative_phase_advance_radians": cumulative_phase,
            "cumulative_metric_path_length": (
                float(seed["accumulated_metric_path_length"])
                if final is None
                else final["cumulative_metric_path_length"]
            ),
            "phase_lap_observed": phase_lap,
            "coarse_recurrence_candidate_observed": candidate,
            "terminal_registered_section_value": (
                None if final is None else final["endpoint_registered_section_value"]
            ),
            "terminal_return_distance_over_path_length": (
                None
                if final is None
                else final["endpoint_return_distance_over_path_length"]
            ),
            "minimum_new_tangent_cosine": min(
                (item["endpoint_metric_tangent_cosine"] for item in recurrence),
                default=None,
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
    new_phase = np.asarray(
        [item["phase_geometry"]["phase_increment"] for item in accepted]
    )
    result_arrays.update(
        {
            "combined_accepted_endpoint_coordinates470": np.vstack(
                (
                    seed["accepted_endpoint_coordinates470"],
                    arrays["accepted_endpoint_coordinates470"],
                )
            ),
            "combined_accepted_endpoint_primitive_states": np.concatenate(
                (
                    seed["accepted_endpoint_primitive_states"],
                    arrays["accepted_endpoint_primitive_states"],
                ),
                axis=0,
            ),
            "combined_accepted_endpoint_coordinate_rates470_per_s": np.vstack(
                (
                    seed["accepted_endpoint_coordinate_rates470_per_s"],
                    arrays["accepted_endpoint_coordinate_rates470_per_s"],
                )
            ),
            "combined_accepted_phase_increments": np.concatenate(
                (seed["accepted_phase_increments"], new_phase)
            ),
            "new_phase_increments": new_phase,
            "new_cumulative_phase_advance_radians": np.asarray(
                [item["cumulative_phase_advance_radians"] for item in recurrence]
            ),
            "new_cumulative_metric_path_lengths": np.asarray(
                [item["cumulative_metric_path_length"] for item in recurrence]
            ),
            "new_registered_section_values": np.asarray(
                [item["endpoint_registered_section_value"] for item in recurrence]
            ),
            "new_return_distance_over_path_lengths": np.asarray(
                [
                    item["endpoint_return_distance_over_path_length"]
                    for item in recurrence
                ]
            ),
            "new_metric_tangent_cosines": np.asarray(
                [item["endpoint_metric_tangent_cosine"] for item in recurrence]
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
        raise RuntimeError("stage1 resume result already exists")
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "stage1_resume_metrics.json", metrics)
    _save_npz(CANONICAL_DIRECTORY / "stage1_resume_arrays.npz", arrays)
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
        "combined_accepted_phase_endpoints": values[
            "combined_accepted_phase_endpoints"
        ],
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
                "# Tangent-phase-lap stage-1 resume execution",
                "",
                f"Classification: `{metrics['classification']}`.",
                "",
                f"Accepted `{values['new_accepted_phase_endpoints']}` of `{values['attempted_segments']}` resume endpoints. The combined accepted stage-1 chain contains `{values['combined_accepted_phase_endpoints']}` endpoints and reaches `{values['terminal_elapsed_seconds']:.6f}` s.",
                "",
                f"Maximum endpoint/blind defects were `{values['maximum_accepted_endpoint_integral_defect']:.6e}` / `{values['maximum_accepted_blind_midpoint_rate_defect']:.6e}`. Maximum raw/metric conditions were `{values['maximum_raw_coordinate_jacobian_condition']:.6e}` / `{values['maximum_metric_coordinate_jacobian_condition']:.6e}`.",
                "",
                f"New phase increments span `{values['minimum_new_phase_increment']}` to `{values['maximum_new_phase_increment']}` rad. Cumulative registered phase is `{values['cumulative_phase_advance_radians']:.9f}` rad. Phase lap: `{values['phase_lap_observed']}`. Coarse recurrence candidate: `{values['coarse_recurrence_candidate_observed']}`.",
                "",
                f"All new endpoints used the 442+28 chart: `{values['all_endpoint_metric_blocks_are_442_plus_28']}`. Checkpoint/suffix replay: `{values['all_accepted_checkpoint_roundtrips_bitwise']}` / `{values['suffix_history_replay_bitwise']}`.",
                "",
                "The original f25fix execution remains a binding rejection at its three-block chart boundary; this package completes the intended stage through a separately prospective chart recovery and resume.",
                "",
                f"Authorized next artifact: `{metrics['authorized_next']}`. Stage-1 completion is not a phase lap, cycle, complete-cycle execution, or reduced slow evolution certificate.",
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
