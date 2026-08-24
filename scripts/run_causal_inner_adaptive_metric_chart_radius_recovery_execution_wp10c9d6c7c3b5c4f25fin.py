#!/usr/bin/env python3
"""Execute one blind-audited recovery segment at the diagnosed chart radius."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_adaptive_metric_chart_radius_recovery_manifest_wp10c9d6c7c3b5c4f25fim as manifest  # noqa: E402
from imri_qpe.layer3_minidisk_1d.conservative_metric_chart_atlas_v2 import (  # noqa: E402
    ConservativeMetricChart,
    metric_transport_retract_strict,
)


diagnosis = manifest.parent
wide = diagnosis.manifest.parent
suffix = wide.source
execution = wide.execution
SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25fin"
PASS_CLASSIFICATION = "adaptive_metric_chart_radius_recovery_passed"
PHYSICAL_FAILURE_CLASSIFICATION = (
    "adaptive_metric_chart_radius_recovery_original_physical_gate_failed"
)
NUMERICAL_FAILURE_CLASSIFICATION = (
    "adaptive_metric_chart_radius_recovery_numerical_or_restart_failed"
)
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fio_adaptive_metric_chart_continuation_manifest"
)
ARTIFACT = (
    "causal_inner_adaptive_metric_chart_radius_recovery_execution_"
    "wp10c9d6c7c3b5c4f25fin"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ADAPTIVE_METRIC_CHART_RADIUS_"
    "RECOVERY_EXECUTION_WP10C9D6C7C3B5C4F25FIN_2026-08-24.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_adaptive_metric_chart_radius_recovery_execution_"
    "wp10c9d6c7c3b5c4f25fin.py"
)
THIS_TEST = (
    "tests/test_causal_inner_adaptive_metric_chart_radius_recovery_execution_"
    "wp10c9d6c7c3b5c4f25fin.py"
)


def _helper():
    return manifest._helper()


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _validate_manifest(*, require_clean: bool) -> dict:
    helper = _helper()
    hashes = helper._validate_checksums(manifest.CANONICAL_DIRECTORY)
    summary = helper._read(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = helper._read(manifest.CANONICAL_DIRECTORY / "recovery_contract.json")
    provenance = helper._read(manifest.CANONICAL_DIRECTORY / "provenance.json")
    if (
        summary["classification"] != manifest.CLASSIFICATION
        or not summary["passed"]
        or not summary["definitions_only"]
        or not summary["adaptive_metric_chart_radius_recovery_authorized"]
        or summary["adaptive_metric_chart_radius_recovery_executed"]
        or summary["new_trajectory"]
        or summary["authorized_next"] != manifest.AUTHORIZED_NEXT
        or contract["authorized_execution"] != manifest.AUTHORIZED_NEXT
        or contract["history"]["recovery_segment_seconds"]
        != manifest.SEGMENT_SECONDS
        or not contract["history"]["blind_midpoint_required"]
        or not contract["history"]["failed_2_ms_candidate_never_propagated"]
        or contract["scope"]["new_exact_free_field_calls"] != 2
        or contract["scope"]["new_accepted_segments_maximum"] != 1
    ):
        raise RuntimeError("adaptive radius-recovery authorization changed")
    for relative, frozen_hash in provenance["source_hashes"].items():
        if helper._sha(ROOT / relative) != frozen_hash:
            raise RuntimeError(f"frozen recovery source changed: {relative}")
    parent_lock = manifest._validate_parent(require_clean=False)
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("radius-recovery execution requires a clean tracked tree")
    return {
        "hashes": hashes,
        "summary": summary,
        "contract": contract,
        "parent_lock": parent_lock,
    }


def _seed() -> dict[str, np.ndarray]:
    return _load_npz(manifest.CANONICAL_DIRECTORY / "recovery_seed.npz")


def _source_hashes() -> dict[str, str]:
    helper = _helper()
    paths = (
        THIS_RUNNER,
        THIS_TEST,
        manifest.THIS_RUNNER,
        manifest.THIS_TEST,
        diagnosis.THIS_RUNNER,
        diagnosis.manifest.STRICT_ATLAS_SOURCE,
        suffix.THIS_RUNNER,
        execution.source.THIS_RUNNER,
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
        "segment_seconds": manifest.SEGMENT_SECONDS,
    }


def _prepare_scratch(lock: dict) -> dict:
    helper = _helper()
    identity = _identity(lock)
    path = SCRATCH_DIRECTORY / "execution_identity.json"
    if SCRATCH_DIRECTORY.exists():
        if not path.exists() or helper._read(path) != identity:
            raise RuntimeError("radius-recovery scratch identity mismatch")
    else:
        SCRATCH_DIRECTORY.mkdir(parents=True)
        helper._write_json(path, identity)
    return identity


def _strict_retraction(
    *,
    stem: str,
    target: np.ndarray,
    seed: dict[str, np.ndarray],
    inputs: dict,
    exact_chart,
    anchor_chart: ConservativeMetricChart,
) -> tuple[dict, dict[str, np.ndarray]]:
    helper = _helper()
    metrics_path = SCRATCH_DIRECTORY / f"{stem}.json"
    arrays_path = SCRATCH_DIRECTORY / f"{stem}.npz"
    if metrics_path.exists() or arrays_path.exists():
        if not metrics_path.exists() or not arrays_path.exists():
            raise RuntimeError("incomplete strict retraction cache")
        metrics = helper._read(metrics_path)
        arrays = _load_npz(arrays_path)
        np.testing.assert_array_equal(arrays["target_original_coordinate470"], target)
        print(f"{stem}: reused strict retraction", flush=True)
        return metrics, arrays
    initial = suffix.parent.parent._initial_state(
        inputs["model"],
        seed["current_primitive_state"],
        seed["current_coordinate470"],
        target,
    )
    state, matrix, metrics = metric_transport_retract_strict(
        exact_chart=exact_chart,
        model=inputs["model"],
        initial_state=initial,
        target_original_coordinate=target,
        gauge_basis=seed["current_gauge_basis560x90"],
        anchor_delta=exact_chart._delta(
            inputs["model"], seed["current_primitive_state"]
        ),
        anchor_metric_augmented=seed["current_metric_augmented560x560"],
        chart=anchor_chart,
        policy=suffix._policy(),
    )
    recovered, factors = inputs["model"].coordinate(state)
    arrays = {
        "target_original_coordinate470": np.asarray(target),
        "recovered_original_coordinate470": np.asarray(recovered),
        "primitive_state": np.asarray(state),
        "final_metric_broyden560x560": np.asarray(matrix),
        "decoder_reconstruction_factors": np.asarray(factors),
    }
    helper._write_json(metrics_path, metrics)
    _save_npz(arrays_path, arrays)
    print(
        f"{stem}: strict={metrics['passed']} "
        f"original={metrics['original_coordinate_residual_infinity']:.3e} "
        f"metric={metrics['metric_coordinate_residual_infinity']:.3e} "
        f"condition={metrics['maximum_metric_augmented_condition_number']:.6g}",
        flush=True,
    )
    return metrics, arrays


def _checkpoint_arrays(
    seed: dict[str, np.ndarray],
    endpoint_coordinate: np.ndarray,
    endpoint_state: np.ndarray,
    endpoint_rate: np.ndarray,
    endpoint_field_arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        "previous_coordinate470": np.asarray(seed["current_coordinate470"]),
        "current_coordinate470": np.asarray(endpoint_coordinate),
        "previous_primitive_state": np.asarray(seed["current_primitive_state"]),
        "current_primitive_state": np.asarray(endpoint_state),
        "previous_coordinate_rate470_per_s": np.asarray(
            seed["current_coordinate_rate470_per_s"]
        ),
        "current_coordinate_rate470_per_s": np.asarray(endpoint_rate),
        "previous_span_seconds": np.asarray(manifest.SEGMENT_SECONDS),
        "next_span_seconds": np.asarray(manifest.SEGMENT_SECONDS),
        "elapsed_seconds": np.asarray(manifest.ENDPOINT_ELAPSED_SECONDS),
        "accepted_segments_total": np.asarray(
            manifest.EXPECTED_NEXT_TENTATIVE_SEGMENT
        ),
        "accepted_since_growth": np.asarray(0),
        "current_metric_transform470x470": np.asarray(
            endpoint_field_arrays["metric_transform470x470"]
        ),
        "current_metric_augmented560x560": np.asarray(
            endpoint_field_arrays["metric_augmented560x560"]
        ),
        "current_gauge_basis560x90": np.asarray(
            endpoint_field_arrays["gauge_basis560x90"]
        ),
        "section_normal470": np.asarray(seed["section_normal470"]),
        "start_coordinate470": np.asarray(seed["start_coordinate470"]),
    }


def _checkpoint_roundtrip(arrays: dict[str, np.ndarray]) -> bool:
    path = SCRATCH_DIRECTORY / "accepted_checkpoint.npz"
    _save_npz(path, arrays)
    replay = _load_npz(path)
    return bool(
        set(arrays) == set(replay)
        and all(np.array_equal(arrays[name], replay[name]) for name in arrays)
    )


def _history_replay(
    seed: dict[str, np.ndarray],
    endpoint_coordinate: np.ndarray,
    endpoint_rate: np.ndarray,
    midpoint_target: np.ndarray,
    midpoint_rate: np.ndarray,
) -> bool:
    candidate = execution._variable_step_ab2(
        seed["current_coordinate470"],
        seed["current_coordinate_rate470_per_s"],
        seed["previous_coordinate_rate470_per_s"],
        manifest.SEGMENT_SECONDS,
        seed["previous_span_seconds"],
    )
    replay_target, replay_rate = execution._hermite(
        seed["current_coordinate470"],
        seed["current_coordinate_rate470_per_s"],
        endpoint_coordinate,
        endpoint_rate,
        manifest.SEGMENT_SECONDS,
        0.5,
    )
    return bool(
        np.array_equal(candidate, seed["candidate_target470"])
        and np.array_equal(replay_target, midpoint_target)
        and np.array_equal(replay_rate, midpoint_rate)
    )


def _execute(lock: dict, identity: dict) -> tuple[dict, dict[str, np.ndarray]]:
    del lock, identity
    began = time.perf_counter()
    seed = _seed()
    inputs = execution.source._initial_inputs()
    exact_chart = execution.source.arclength._exact_chart()
    anchor_chart = ConservativeMetricChart(
        seed["current_coordinate470"],
        seed["current_metric_transform470x470"],
        suffix._block_sizes(),
    )
    endpoint_retraction, endpoint_retraction_arrays = _strict_retraction(
        stem="endpoint_retraction",
        target=seed["candidate_target470"],
        seed=seed,
        inputs=inputs,
        exact_chart=exact_chart,
        anchor_chart=anchor_chart,
    )
    endpoint_coordinate = np.asarray(
        endpoint_retraction_arrays["recovered_original_coordinate470"]
    )
    endpoint_state = np.asarray(endpoint_retraction_arrays["primitive_state"])
    saved_state_defect = manifest._relative(
        endpoint_state, seed["diagnosed_primitive_state"]
    )
    endpoint_field = None
    endpoint_field_arrays = None
    endpoint_rate = np.full(470, np.nan)
    endpoint_defect = float("inf")
    if endpoint_retraction["passed"]:
        endpoint_field, endpoint_field_arrays = suffix._metric_field(
            directory=SCRATCH_DIRECTORY,
            stem="endpoint_field",
            inputs=inputs,
            exact_chart=exact_chart,
            state=endpoint_state,
            coordinate=endpoint_coordinate,
            retraction=endpoint_retraction,
            anchor_chart=anchor_chart,
        )
        endpoint_rate = np.asarray(
            endpoint_field_arrays["coordinate_free_rate470_per_s"]
        )
        endpoint_defect = execution._endpoint_integral_defect(
            seed["current_coordinate470"],
            seed["current_coordinate_rate470_per_s"],
            endpoint_coordinate,
            endpoint_rate,
            manifest.SEGMENT_SECONDS,
        )
    endpoint_passed = bool(
        endpoint_retraction["passed"]
        and endpoint_field is not None
        and endpoint_field["physical_passed"]
        and saved_state_defect <= manifest.MAXIMUM_SAVED_STATE_RELATIVE_DEFECT
        and endpoint_defect <= manifest.MAXIMUM_ENDPOINT_INTEGRAL_DEFECT
    )
    midpoint_target = np.full(470, np.nan)
    midpoint_hermite_rate = np.full(470, np.nan)
    midpoint_retraction = None
    midpoint_retraction_arrays = None
    midpoint_field = None
    midpoint_field_arrays = None
    midpoint_coordinate = np.full(470, np.nan)
    midpoint_state = np.full((112, 5), np.nan)
    midpoint_rate = np.full(470, np.nan)
    midpoint_defect = float("inf")
    if endpoint_passed:
        midpoint_target, midpoint_hermite_rate = execution._hermite(
            seed["current_coordinate470"],
            seed["current_coordinate_rate470_per_s"],
            endpoint_coordinate,
            endpoint_rate,
            manifest.SEGMENT_SECONDS,
            0.5,
        )
        midpoint_retraction, midpoint_retraction_arrays = _strict_retraction(
            stem="midpoint_retraction",
            target=midpoint_target,
            seed=seed,
            inputs=inputs,
            exact_chart=exact_chart,
            anchor_chart=anchor_chart,
        )
        midpoint_coordinate = np.asarray(
            midpoint_retraction_arrays["recovered_original_coordinate470"]
        )
        midpoint_state = np.asarray(midpoint_retraction_arrays["primitive_state"])
        if midpoint_retraction["passed"]:
            midpoint_field, midpoint_field_arrays = suffix._metric_field(
                directory=SCRATCH_DIRECTORY,
                stem="midpoint_field",
                inputs=inputs,
                exact_chart=exact_chart,
                state=midpoint_state,
                coordinate=midpoint_coordinate,
                retraction=midpoint_retraction,
                anchor_chart=anchor_chart,
            )
            midpoint_rate = np.asarray(
                midpoint_field_arrays["coordinate_free_rate470_per_s"]
            )
            midpoint_defect = suffix._relative(midpoint_hermite_rate, midpoint_rate)
    midpoint_passed = bool(
        midpoint_retraction is not None
        and midpoint_retraction["passed"]
        and midpoint_field is not None
        and midpoint_field["physical_passed"]
        and midpoint_defect <= manifest.MAXIMUM_BLIND_MIDPOINT_RATE_DEFECT
    )
    accepted = bool(endpoint_passed and midpoint_passed)
    checkpoint = None
    checkpoint_roundtrip = False
    history_replay = False
    if accepted:
        checkpoint = _checkpoint_arrays(
            seed,
            endpoint_coordinate,
            endpoint_state,
            endpoint_rate,
            endpoint_field_arrays,
        )
        checkpoint_roundtrip = _checkpoint_roundtrip(checkpoint)
        history_replay = _history_replay(
            seed,
            endpoint_coordinate,
            endpoint_rate,
            midpoint_target,
            midpoint_hermite_rate,
        )
    physical_failure = bool(
        (endpoint_field is not None and not endpoint_field["physical_passed"])
        or (midpoint_field is not None and not midpoint_field["physical_passed"])
        or not endpoint_retraction["physical_passed"]
        or (
            midpoint_retraction is not None
            and not midpoint_retraction["physical_passed"]
        )
    )
    wall_seconds = float(time.perf_counter() - began)
    passed = bool(
        accepted
        and checkpoint_roundtrip
        and history_replay
        and wall_seconds <= 3600.0 * manifest.MAXIMUM_EXECUTION_WALL_HOURS
    )
    if passed:
        classification = PASS_CLASSIFICATION
    elif physical_failure:
        classification = PHYSICAL_FAILURE_CLASSIFICATION
    else:
        classification = NUMERICAL_FAILURE_CLASSIFICATION
    exact_fields = [
        value for value in (endpoint_field, midpoint_field) if value is not None
    ]
    ledgers = [
        max(
            value
            for name, value in field["free_field"][
                "reaction_free_ledger_values"
            ].items()
            if name != "incoming_excision_characteristics"
        )
        for field in exact_fields
    ]
    section_before = float(
        seed["section_normal470"]
        @ (seed["current_coordinate470"] - seed["start_coordinate470"])
    )
    section_after = float(
        seed["section_normal470"]
        @ (endpoint_coordinate - seed["start_coordinate470"])
    )
    gates = {
        "parent_elapsed_seconds": manifest.PARENT_ELAPSED_SECONDS,
        "endpoint_elapsed_seconds": (
            manifest.ENDPOINT_ELAPSED_SECONDS if accepted else manifest.PARENT_ELAPSED_SECONDS
        ),
        "segment_seconds": manifest.SEGMENT_SECONDS,
        "tentative_segment_number": manifest.EXPECTED_NEXT_TENTATIVE_SEGMENT,
        "accepted": accepted,
        "saved_endpoint_state_relative_defect": saved_state_defect,
        "endpoint_retraction_passed": endpoint_retraction["passed"],
        "midpoint_retraction_passed": (
            None if midpoint_retraction is None else midpoint_retraction["passed"]
        ),
        "endpoint_physical_passed": (
            False if endpoint_field is None else endpoint_field["physical_passed"]
        ),
        "midpoint_physical_passed": (
            False if midpoint_field is None else midpoint_field["physical_passed"]
        ),
        "endpoint_integral_defect": endpoint_defect,
        "blind_midpoint_rate_defect": midpoint_defect,
        "maximum_raw_coordinate_jacobian_condition": max(
            (
                field["free_field"]["coordinate_jacobian_condition_number"]
                for field in exact_fields
            ),
            default=0.0,
        ),
        "maximum_metric_coordinate_jacobian_condition": max(
            (
                field["metric_chart"]["metric_jacobian_condition_number"]
                for field in exact_fields
            ),
            default=0.0,
        ),
        "maximum_metric_augmented_condition": max(
            (
                field["metric_chart"]["metric_augmented_condition_number"]
                for field in exact_fields
            ),
            default=0.0,
        ),
        "maximum_patch_transition_condition": max(
            (
                field["metric_chart"]["patch_transition_condition_number"]
                for field in exact_fields
            ),
            default=0.0,
        ),
        "minimum_reconstruction_factor": min(
            (
                field["free_field"]["minimum_reconstruction_factor"]
                for field in exact_fields
            ),
            default=1.0,
        ),
        "maximum_height_ratio": max(
            (field["free_field"]["maximum_height_ratio"] for field in exact_fields),
            default=0.0,
        ),
        "minimum_scattering_optical_depth": min(
            (
                field["free_field"]["minimum_scattering_optical_depth"]
                for field in exact_fields
            ),
            default=float("inf"),
        ),
        "maximum_reaction_free_ledger_defect": max(ledgers, default=0.0),
        "section_before": section_before,
        "section_after": section_after,
        "checkpoint_roundtrip_bitwise": checkpoint_roundtrip,
        "history_replay_bitwise": history_replay,
        "new_accepted_segments": int(accepted),
        "exact_free_field_calls": len(exact_fields),
        "fixed_Q_calls": 0,
        "reaction_calls": 0,
        "nonlinear_roots": 0,
        "BDF_microsteps": 0,
        "execution_wall_seconds": wall_seconds,
    }
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "authorized_next": AUTHORIZED_NEXT if passed else None,
        "gate_values": gates,
        "endpoint_field": endpoint_field,
        "midpoint_field": midpoint_field,
    }
    arrays = {
        "candidate_target470": np.asarray(seed["candidate_target470"]),
        "endpoint_coordinate470": endpoint_coordinate,
        "endpoint_primitive_state": endpoint_state,
        "endpoint_coordinate_rate470_per_s": endpoint_rate,
        "midpoint_target470": midpoint_target,
        "midpoint_hermite_rate470_per_s": midpoint_hermite_rate,
        "midpoint_coordinate470": midpoint_coordinate,
        "midpoint_primitive_state": midpoint_state,
        "midpoint_coordinate_rate470_per_s": midpoint_rate,
        **(
            checkpoint
            if checkpoint is not None
            else {
                "previous_coordinate470": seed["previous_coordinate470"],
                "current_coordinate470": seed["current_coordinate470"],
                "previous_primitive_state": seed["previous_primitive_state"],
                "current_primitive_state": seed["current_primitive_state"],
                "previous_coordinate_rate470_per_s": seed[
                    "previous_coordinate_rate470_per_s"
                ],
                "current_coordinate_rate470_per_s": seed[
                    "current_coordinate_rate470_per_s"
                ],
                "previous_span_seconds": seed["previous_span_seconds"],
                "next_span_seconds": np.asarray(manifest.SEGMENT_SECONDS),
                "elapsed_seconds": seed["elapsed_seconds"],
                "accepted_segments_total": seed["accepted_segments_total"],
                "accepted_since_growth": np.asarray(0),
                "current_metric_transform470x470": seed[
                    "current_metric_transform470x470"
                ],
                "current_metric_augmented560x560": seed[
                    "current_metric_augmented560x560"
                ],
                "current_gauge_basis560x90": seed[
                    "current_gauge_basis560x90"
                ],
                "section_normal470": seed["section_normal470"],
                "start_coordinate470": seed["start_coordinate470"],
            }
        ),
    }
    return metrics, arrays


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


def _canonicalize(metrics: dict, arrays: dict[str, np.ndarray], lock: dict, identity: dict) -> dict:
    helper = _helper()
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("radius-recovery result already exists")
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "recovery_metrics.json", metrics)
    _save_npz(CANONICAL_DIRECTORY / "recovery_arrays.npz", arrays)
    helper._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "manifest_hashes": lock["hashes"],
            "manifest_classification": lock["summary"]["classification"],
            "execution_identity": identity,
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": metrics["passed"],
        "new_accepted_segments": metrics["gate_values"]["new_accepted_segments"],
        "endpoint_elapsed_seconds": metrics["gate_values"]["endpoint_elapsed_seconds"],
        "adaptive_metric_chart_radius_recovery_passed": metrics["passed"],
        "cycle_authorized": False,
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
    values = metrics["gate_values"]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Adaptive metric-chart radius recovery execution",
                "",
                f"Classification: `{metrics['classification']}`.",
                "",
                f"Accepted `{values['new_accepted_segments']}` segment of `{values['segment_seconds']:.6f}` s, reaching `{values['endpoint_elapsed_seconds']:.6f}` s.",
                "",
                f"Endpoint/blind defects: `{values['endpoint_integral_defect']:.6e}` / `{values['blind_midpoint_rate_defect']:.6e}`. Maximum raw/metric conditions: `{values['maximum_raw_coordinate_jacobian_condition']:.6e}` / `{values['maximum_metric_coordinate_jacobian_condition']:.6e}`.",
                "",
                f"Checkpoint/history replay: `{values['checkpoint_roundtrip_bitwise']}` / `{values['history_replay_bitwise']}`.",
                "",
                f"Authorized next artifact: `{metrics['authorized_next']}`.",
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
