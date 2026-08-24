#!/usr/bin/env python3
"""Execute one accepted segment through the historical raw chart boundary."""

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

import run_causal_inner_metric_chart_boundary_crossing_manifest_wp10c9d6c7c3b5c4f25fie as manifest  # noqa: E402
from imri_qpe.layer3_minidisk_1d.conservative_metric_chart_atlas import (  # noqa: E402
    ConservativeMetricChart,
    block_whitening_transform,
    metric_augmented_jacobian,
    metric_transport_retract,
)


parent = manifest.parent
execution = manifest.execution
SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25fif"
PASS_CLASSIFICATION = "metric_chart_boundary_crossing_passed"
PHYSICAL_FAILURE_CLASSIFICATION = (
    "metric_chart_boundary_crossing_original_physical_gate_failed"
)
NUMERICAL_FAILURE_CLASSIFICATION = (
    "metric_chart_boundary_crossing_numerical_gate_failed"
)
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fig_metric_chart_short_suffix_manifest"
)
ARTIFACT = (
    "causal_inner_metric_chart_boundary_crossing_execution_"
    "wp10c9d6c7c3b5c4f25fif"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_METRIC_CHART_BOUNDARY_CROSSING_"
    "EXECUTION_WP10C9D6C7C3B5C4F25FIF_2026-08-24.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_metric_chart_boundary_crossing_execution_"
    "wp10c9d6c7c3b5c4f25fif.py"
)
THIS_TEST = (
    "tests/test_causal_inner_metric_chart_boundary_crossing_execution_"
    "wp10c9d6c7c3b5c4f25fif.py"
)
ATLAS_SOURCE = parent.ATLAS_SOURCE


def _helper():
    return manifest._helper()


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    return manifest._relative(left, right)


def _validate_manifest(*, require_clean: bool) -> dict:
    helper = _helper()
    hashes = helper._validate_checksums(manifest.CANONICAL_DIRECTORY)
    summary = helper._read(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = helper._read(
        manifest.CANONICAL_DIRECTORY / "boundary_contract.json"
    )
    provenance = helper._read(manifest.CANONICAL_DIRECTORY / "provenance.json")
    if (
        summary["classification"] != manifest.CLASSIFICATION
        or not summary["passed"]
        or not summary["definitions_only"]
        or not summary["metric_chart_boundary_crossing_authorized"]
        or summary["metric_chart_boundary_crossing_executed"]
        or summary["new_trajectory"]
        or summary["authorized_next"] != manifest.AUTHORIZED_NEXT
        or contract["authorized_execution"] != manifest.AUTHORIZED_NEXT
        or contract["scope"]["new_exact_free_field_calls"] != 2
        or contract["scope"]["new_retractions"] != 3
        or contract["scope"]["new_accepted_segments_maximum"] != 1
        or contract["history"]["tentative_segment_number"] != 72
        or not contract["history"]["blind_midpoint_required"]
    ):
        raise RuntimeError("boundary-crossing authorization changed")
    for relative, frozen_hash in provenance["source_hashes"].items():
        if helper._sha(ROOT / relative) != frozen_hash:
            raise RuntimeError(f"frozen boundary source changed: {relative}")
    parent_lock = manifest._validate_parent(require_clean=False)
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("boundary-crossing execution requires a clean tracked tree")
    return {
        "hashes": hashes,
        "summary": summary,
        "contract": contract,
        "parent_lock": parent_lock,
    }


def _source_hashes() -> dict[str, str]:
    helper = _helper()
    paths = (
        THIS_RUNNER,
        THIS_TEST,
        manifest.THIS_RUNNER,
        parent.THIS_RUNNER,
        ATLAS_SOURCE,
        execution.THIS_RUNNER,
        execution.source.THIS_RUNNER,
        "src/imri_qpe/layer3_minidisk_1d/conservative_free_field_rom.py",
    )
    return {name: helper._sha(ROOT / name) for name in paths}


def _prepare_scratch(lock: dict) -> dict:
    helper = _helper()
    identity = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "implementation_commit": helper._git("rev-parse", "HEAD"),
        "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"),
        "source_hashes": _source_hashes(),
        "manifest_hashes": lock["hashes"],
    }
    path = SCRATCH_DIRECTORY / "execution_identity.json"
    if SCRATCH_DIRECTORY.exists():
        if not path.exists() or helper._read(path) != identity:
            raise RuntimeError("boundary-crossing scratch identity mismatch")
    else:
        SCRATCH_DIRECTORY.mkdir(parents=True)
        helper._write_json(path, identity)
    return identity


def _pair(stem: str) -> tuple[Path, Path]:
    return (
        SCRATCH_DIRECTORY / f"{stem}.json",
        SCRATCH_DIRECTORY / f"{stem}.npz",
    )


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _load_pair(stem: str) -> tuple[dict, dict[str, np.ndarray]] | None:
    metrics_path, arrays_path = _pair(stem)
    if metrics_path.exists() != arrays_path.exists():
        raise RuntimeError(f"partial boundary scratch pair: {stem}")
    if not metrics_path.exists():
        return None
    return _helper()._read(metrics_path), _load_npz(arrays_path)


def _write_pair(
    stem: str, metrics: dict, arrays: dict[str, np.ndarray]
) -> None:
    metrics_path, arrays_path = _pair(stem)
    _helper()._write_json(metrics_path, metrics)
    _save_npz(arrays_path, arrays)


def _seed() -> dict[str, np.ndarray]:
    return _load_npz(manifest.CANONICAL_DIRECTORY / "boundary_seed.npz")


def _block_sizes() -> tuple[int, ...]:
    return (
        parent.diagnosis.manifest.PHYSICAL_ROWS,
        parent.diagnosis.manifest.MEMORY_ROWS,
        parent.diagnosis.manifest.DEPARTURE_ROWS,
    )


def _retraction_policy():
    policy = parent._policy()
    if (
        policy.original_coordinate_tolerance
        != manifest.ORIGINAL_COORDINATE_RESIDUAL_TOLERANCE
        or policy.metric_coordinate_tolerance
        != manifest.METRIC_COORDINATE_RESIDUAL_TOLERANCE
        or policy.gauge_tolerance != manifest.GAUGE_RESIDUAL_TOLERANCE
        or policy.maximum_metric_augmented_condition
        != manifest.MAXIMUM_METRIC_AUGMENTED_CONDITION
    ):
        raise RuntimeError("metric retraction policy changed")
    return policy


def _metric_retraction(
    *,
    stem: str,
    exact_chart,
    model,
    seed: dict[str, np.ndarray],
    target: np.ndarray,
    anchor_chart: ConservativeMetricChart,
) -> tuple[dict, dict[str, np.ndarray]]:
    cached = _load_pair(stem)
    if cached is not None:
        np.testing.assert_array_equal(
            cached[1]["target_original_coordinate470"], target
        )
        print(f"{stem}: reused metric retraction", flush=True)
        return cached
    initial = parent._initial_state(
        model,
        seed["current_primitive_state"],
        seed["current_coordinate470"],
        target,
    )
    state, matrix, metrics = metric_transport_retract(
        exact_chart=exact_chart,
        model=model,
        initial_state=initial,
        target_original_coordinate=target,
        gauge_basis=seed["anchor_gauge_basis560x90"],
        anchor_delta=exact_chart._delta(model, seed["current_primitive_state"]),
        anchor_metric_augmented=seed["anchor_metric_augmented560x560"],
        chart=anchor_chart,
        policy=_retraction_policy(),
    )
    recovered, factors = model.coordinate(state)
    arrays = {
        "target_original_coordinate470": np.asarray(target),
        "recovered_original_coordinate470": np.asarray(recovered),
        "primitive_state": np.asarray(state),
        "final_metric_broyden560x560": np.asarray(matrix),
        "decoder_reconstruction_factors": np.asarray(factors),
    }
    _write_pair(stem, metrics, arrays)
    print(
        f"{stem}: original={metrics['original_coordinate_residual_infinity']:.3e} "
        f"metric={metrics['metric_coordinate_residual_infinity']:.3e} "
        f"gauge={metrics['gauge_residual_infinity']:.3e} "
        f"passed={metrics['passed']}",
        flush=True,
    )
    return metrics, arrays


def _field_physical_passed(
    free: dict,
    retraction: dict,
    metric: dict,
) -> bool:
    return bool(
        retraction["passed"]
        and retraction["original_coordinate_residual_infinity"]
        <= manifest.ORIGINAL_COORDINATE_RESIDUAL_TOLERANCE
        and retraction["metric_coordinate_residual_infinity"]
        <= manifest.METRIC_COORDINATE_RESIDUAL_TOLERANCE
        and retraction["gauge_residual_infinity"]
        <= manifest.GAUGE_RESIDUAL_TOLERANCE
        and retraction["maximum_metric_augmented_condition_number"]
        <= manifest.MAXIMUM_METRIC_AUGMENTED_CONDITION
        and free["coordinate_jacobian_rank"] == 470
        and free["coordinate_reconstruction_relative_defect"]
        <= manifest.MAXIMUM_COORDINATE_RECONSTRUCTION_DEFECT
        and metric["metric_jacobian_condition_number"]
        <= manifest.MAXIMUM_METRIC_JACOBIAN_CONDITION
        and metric["metric_augmented_condition_number"]
        <= manifest.MAXIMUM_METRIC_AUGMENTED_CONDITION
        and metric["patch_transition_condition_number"]
        <= manifest.MAXIMUM_PATCH_TRANSITION_CONDITION
        and metric["transform_inverse_closure_defect"]
        <= manifest.MAXIMUM_TRANSFORM_INVERSE_CLOSURE
        and metric["raw_condition_reproduction_relative_defect"] <= 1.0e-12
        and free["minimum_reconstruction_factor"] >= 1.0 - 1.0e-12
        and free["maximum_height_ratio"] <= 0.5
        and free["minimum_scattering_optical_depth"] >= 1.0
        and free["reaction_free_ledger_passed"]
    )


def _metric_field(
    *,
    stem: str,
    inputs: dict,
    exact_chart,
    state: np.ndarray,
    coordinate: np.ndarray,
    retraction: dict,
    anchor_chart: ConservativeMetricChart,
) -> tuple[dict, dict[str, np.ndarray]]:
    cached = _load_pair(stem)
    if cached is not None:
        np.testing.assert_array_equal(cached[1]["requested_coordinate470"], coordinate)
        print(f"{stem}: reused exact metric-audited field", flush=True)
        return cached
    began = time.perf_counter()
    free_metrics, free_arrays = execution.source._free_field(inputs, state)
    free_wall = float(time.perf_counter() - began)
    began = time.perf_counter()
    jacobian, jacobian_metrics = exact_chart._coordinate_jacobian(
        inputs["model"], state
    )
    transform, transform_metrics = block_whitening_transform(
        jacobian, _block_sizes()
    )
    chart = ConservativeMetricChart(coordinate, transform, _block_sizes())
    gauge = exact_chart._canonical_null_basis(jacobian)
    augmented, augmented_condition = metric_augmented_jacobian(
        jacobian, gauge, chart
    )
    independent_wall = float(time.perf_counter() - began)
    transition = transform @ anchor_chart.inverse_transform
    raw_reproduction = abs(
        float(jacobian_metrics["condition_number"])
        - float(free_metrics["coordinate_jacobian_condition_number"])
    ) / max(
        float(jacobian_metrics["condition_number"]),
        float(free_metrics["coordinate_jacobian_condition_number"]),
        np.finfo(float).tiny,
    )
    metric = {
        **transform_metrics,
        "metric_augmented_condition_number": float(augmented_condition),
        "patch_transition_condition_number": float(np.linalg.cond(transition)),
        "transform_inverse_closure_defect": chart.inverse_closure_defect,
        "raw_condition_reproduction_relative_defect": float(raw_reproduction),
        "independent_metric_jacobian_wall_seconds": independent_wall,
    }
    physical_passed = _field_physical_passed(
        free_metrics, retraction, metric
    )
    metrics = {
        "free_field": free_metrics,
        "retraction": retraction,
        "metric_chart": metric,
        "physical_passed": physical_passed,
        "exact_free_field_call_wall_seconds": free_wall,
        "total_call_wall_seconds": free_wall + independent_wall,
        "historical_raw_condition_exceeded": bool(
            free_metrics["coordinate_jacobian_condition_number"]
            > manifest.HISTORICAL_RAW_CONDITION
        ),
    }
    arrays = {
        **free_arrays,
        "requested_coordinate470": np.asarray(coordinate),
        "coordinate_jacobian470x560": np.asarray(jacobian),
        "metric_transform470x470": np.asarray(transform),
        "metric_augmented560x560": np.asarray(augmented),
        "gauge_basis560x90": np.asarray(gauge),
    }
    _write_pair(stem, metrics, arrays)
    print(
        f"{stem}: |f|={free_metrics['coordinate_free_rate_norm_per_second']:.3e}/s "
        f"raw_kappa={free_metrics['coordinate_jacobian_condition_number']:.3f} "
        f"metric_kappa={metric['metric_jacobian_condition_number']:.3f} "
        f"wall={metrics['total_call_wall_seconds']:.1f}s "
        f"physical={physical_passed}",
        flush=True,
    )
    return metrics, arrays


def _without_wall(metrics: dict) -> dict:
    return {
        name: value
        for name, value in metrics.items()
        if name != "wall_seconds"
    }


def _checkpoint_roundtrip(arrays: dict[str, np.ndarray]) -> bool:
    path = SCRATCH_DIRECTORY / "accepted_boundary_checkpoint.npz"
    if not path.exists():
        _save_npz(path, arrays)
    loaded = _load_npz(path)
    return bool(
        set(loaded) == set(arrays)
        and all(np.array_equal(loaded[name], arrays[name]) for name in arrays)
    )


def _execute(lock: dict, identity: dict) -> tuple[dict, dict[str, np.ndarray]]:
    del identity
    began = time.perf_counter()
    seed = _seed()
    inputs = execution.source._initial_inputs()
    model = inputs["model"]
    exact_chart = execution.source.arclength._exact_chart()
    anchor_chart = ConservativeMetricChart(
        seed["current_coordinate470"],
        seed["anchor_metric_transform470x470"],
        _block_sizes(),
    )

    endpoint_retraction, endpoint_retraction_arrays = _metric_retraction(
        stem="endpoint_retraction",
        exact_chart=exact_chart,
        model=model,
        seed=seed,
        target=seed["candidate_target470"],
        anchor_chart=anchor_chart,
    )
    replay_metrics, replay_arrays = _metric_retraction(
        stem="endpoint_retraction_replay",
        exact_chart=exact_chart,
        model=model,
        seed=seed,
        target=seed["candidate_target470"],
        anchor_chart=anchor_chart,
    )
    endpoint_replay_bitwise = bool(
        np.array_equal(
            endpoint_retraction_arrays["primitive_state"],
            replay_arrays["primitive_state"],
        )
        and np.array_equal(
            endpoint_retraction_arrays["final_metric_broyden560x560"],
            replay_arrays["final_metric_broyden560x560"],
        )
        and _without_wall(endpoint_retraction) == _without_wall(replay_metrics)
    )
    endpoint_coordinate = np.asarray(
        endpoint_retraction_arrays["recovered_original_coordinate470"]
    )
    endpoint_state = np.asarray(endpoint_retraction_arrays["primitive_state"])
    endpoint_metrics, endpoint_arrays = _metric_field(
        stem="endpoint_field",
        inputs=inputs,
        exact_chart=exact_chart,
        state=endpoint_state,
        coordinate=endpoint_coordinate,
        retraction=endpoint_retraction,
        anchor_chart=anchor_chart,
    )
    endpoint_rate = np.asarray(endpoint_arrays["coordinate_free_rate470_per_s"])
    endpoint_integral_defect = execution._endpoint_integral_defect(
        seed["current_coordinate470"],
        seed["current_coordinate_rate470_per_s"],
        endpoint_coordinate,
        endpoint_rate,
        manifest.SEGMENT_SECONDS,
    )
    endpoint_state_defect = _relative(
        endpoint_state, seed["saved_boundary_primitive_state"]
    )
    endpoint_rate_defect = _relative(
        endpoint_rate, seed["saved_boundary_coordinate_rate470_per_s"]
    )
    endpoint_coordinate_defect = _relative(
        endpoint_coordinate, seed["saved_boundary_target470"]
    )
    endpoint_passed = bool(
        endpoint_retraction["passed"]
        and endpoint_metrics["physical_passed"]
        and endpoint_replay_bitwise
        and endpoint_integral_defect <= manifest.MAXIMUM_ENDPOINT_INTEGRAL_DEFECT
        and endpoint_state_defect <= manifest.MAXIMUM_SAVED_STATE_RELATIVE_DEFECT
        and endpoint_rate_defect <= manifest.MAXIMUM_SAVED_RATE_RELATIVE_DEFECT
        and endpoint_coordinate_defect
        <= manifest.MAXIMUM_SAVED_TARGET_RELATIVE_DEFECT
        and endpoint_metrics["historical_raw_condition_exceeded"]
    )

    midpoint_target, midpoint_hermite_rate = execution._hermite(
        seed["current_coordinate470"],
        seed["current_coordinate_rate470_per_s"],
        endpoint_coordinate,
        endpoint_rate,
        manifest.SEGMENT_SECONDS,
        0.5,
    )
    midpoint_retraction = None
    midpoint_retraction_arrays = None
    midpoint_metrics = None
    midpoint_arrays = None
    blind_defect = float("inf")
    if endpoint_passed:
        midpoint_retraction, midpoint_retraction_arrays = _metric_retraction(
            stem="midpoint_retraction",
            exact_chart=exact_chart,
            model=model,
            seed=seed,
            target=midpoint_target,
            anchor_chart=anchor_chart,
        )
        midpoint_coordinate = np.asarray(
            midpoint_retraction_arrays["recovered_original_coordinate470"]
        )
        midpoint_metrics, midpoint_arrays = _metric_field(
            stem="midpoint_field",
            inputs=inputs,
            exact_chart=exact_chart,
            state=midpoint_retraction_arrays["primitive_state"],
            coordinate=midpoint_coordinate,
            retraction=midpoint_retraction,
            anchor_chart=anchor_chart,
        )
        blind_defect = _relative(
            midpoint_hermite_rate,
            midpoint_arrays["coordinate_free_rate470_per_s"],
        )
    midpoint_passed = bool(
        midpoint_retraction is not None
        and midpoint_retraction["passed"]
        and midpoint_metrics is not None
        and midpoint_metrics["physical_passed"]
        and blind_defect <= manifest.MAXIMUM_BLIND_MIDPOINT_RATE_DEFECT
    )

    physical_failure = bool(
        not endpoint_metrics["physical_passed"]
        or (
            midpoint_metrics is not None
            and not midpoint_metrics["physical_passed"]
        )
    )
    provisional_pass = bool(endpoint_passed and midpoint_passed)
    checkpoint_arrays = {
        "previous_coordinate470": seed["current_coordinate470"],
        "current_coordinate470": endpoint_coordinate,
        "previous_primitive_state": seed["current_primitive_state"],
        "current_primitive_state": endpoint_state,
        "previous_coordinate_rate470_per_s": seed[
            "current_coordinate_rate470_per_s"
        ],
        "current_coordinate_rate470_per_s": endpoint_rate,
        "previous_span_seconds": np.asarray(manifest.SEGMENT_SECONDS),
        "next_span_seconds": np.asarray(manifest.SEGMENT_SECONDS),
        "elapsed_seconds": np.asarray(manifest.ENDPOINT_ELAPSED_SECONDS),
        "accepted_segments_total": np.asarray(
            manifest.EXPECTED_PRIOR_ACCEPTED_SEGMENTS + 1
        ),
        "section_normal470": seed["section_normal470"],
        "start_coordinate470": seed["start_coordinate470"],
        "current_metric_transform470x470": endpoint_arrays[
            "metric_transform470x470"
        ],
        "current_metric_augmented560x560": endpoint_arrays[
            "metric_augmented560x560"
        ],
        "current_gauge_basis560x90": endpoint_arrays["gauge_basis560x90"],
    }
    checkpoint_roundtrip = bool(
        provisional_pass and _checkpoint_roundtrip(checkpoint_arrays)
    )
    passed = bool(provisional_pass and checkpoint_roundtrip)
    if passed:
        classification = PASS_CLASSIFICATION
    elif physical_failure:
        classification = PHYSICAL_FAILURE_CLASSIFICATION
    else:
        classification = NUMERICAL_FAILURE_CLASSIFICATION
    raw_conditions = [
        endpoint_metrics["free_field"]["coordinate_jacobian_condition_number"]
    ]
    metric_conditions = [
        endpoint_metrics["metric_chart"]["metric_jacobian_condition_number"]
    ]
    if midpoint_metrics is not None:
        raw_conditions.append(
            midpoint_metrics["free_field"]["coordinate_jacobian_condition_number"]
        )
        metric_conditions.append(
            midpoint_metrics["metric_chart"]["metric_jacobian_condition_number"]
        )
    execution_wall = float(time.perf_counter() - began)
    gates = {
        "parent_elapsed_seconds": manifest.PARENT_ELAPSED_SECONDS,
        "terminal_elapsed_seconds": (
            manifest.ENDPOINT_ELAPSED_SECONDS
            if passed
            else manifest.PARENT_ELAPSED_SECONDS
        ),
        "new_accepted_segments": int(passed),
        "new_accepted_horizon_seconds": (
            manifest.SEGMENT_SECONDS if passed else 0.0
        ),
        "exact_free_field_calls": 1 + int(midpoint_metrics is not None),
        "independent_metric_jacobian_audits": 1 + int(midpoint_metrics is not None),
        "metric_retractions": 2 + int(midpoint_retraction is not None),
        "endpoint_retraction_replay_bitwise": endpoint_replay_bitwise,
        "checkpoint_roundtrip_bitwise": checkpoint_roundtrip,
        "endpoint_original_coordinate_residual_infinity": endpoint_retraction[
            "original_coordinate_residual_infinity"
        ],
        "endpoint_metric_coordinate_residual_infinity": endpoint_retraction[
            "metric_coordinate_residual_infinity"
        ],
        "endpoint_gauge_residual_infinity": endpoint_retraction[
            "gauge_residual_infinity"
        ],
        "endpoint_integral_defect": endpoint_integral_defect,
        "blind_midpoint_rate_defect": blind_defect,
        "endpoint_saved_coordinate_relative_defect": endpoint_coordinate_defect,
        "endpoint_saved_state_relative_defect": endpoint_state_defect,
        "endpoint_saved_rate_relative_defect": endpoint_rate_defect,
        "endpoint_historical_raw_condition_crossed": endpoint_metrics[
            "historical_raw_condition_exceeded"
        ],
        "maximum_raw_coordinate_jacobian_condition": float(max(raw_conditions)),
        "maximum_metric_coordinate_jacobian_condition": float(
            max(metric_conditions)
        ),
        "all_new_exact_fields_physical_passed": bool(
            endpoint_metrics["physical_passed"]
            and midpoint_metrics is not None
            and midpoint_metrics["physical_passed"]
        ),
        "execution_wall_seconds": execution_wall,
        "maximum_execution_wall_seconds": (
            3600.0 * manifest.MAXIMUM_EXECUTION_WALL_HOURS
        ),
    }
    passed = bool(
        passed
        and gates["exact_free_field_calls"]
        <= manifest.MAXIMUM_EXACT_FREE_FIELD_CALLS
        and gates["independent_metric_jacobian_audits"]
        <= manifest.MAXIMUM_INDEPENDENT_METRIC_JACOBIAN_AUDITS
        and gates["metric_retractions"] <= manifest.MAXIMUM_RETRACTIONS
        and execution_wall <= 3600.0 * manifest.MAXIMUM_EXECUTION_WALL_HOURS
    )
    if not passed and classification == PASS_CLASSIFICATION:
        classification = NUMERICAL_FAILURE_CLASSIFICATION
        gates["terminal_elapsed_seconds"] = manifest.PARENT_ELAPSED_SECONDS
        gates["new_accepted_segments"] = 0
        gates["new_accepted_horizon_seconds"] = 0.0
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "gate_values": gates,
        "endpoint_retraction": endpoint_retraction,
        "endpoint_field": endpoint_metrics,
        "midpoint_retraction": midpoint_retraction,
        "midpoint_field": midpoint_metrics,
        "input_lock": {
            "manifest_hashes": lock["hashes"],
            "manifest_classification": lock["summary"]["classification"],
        },
        "fixed_Q_physical_rate_calls": 0,
        "fixed_Q_reaction_calls": 0,
        "nonlinear_roots": 0,
        "BDF_microsteps": 0,
    }
    arrays = {
        **checkpoint_arrays,
        "candidate_target470": seed["candidate_target470"],
        "endpoint_retracted_coordinate470": endpoint_coordinate,
        "endpoint_retracted_primitive_state": endpoint_state,
        "endpoint_coordinate_rate470_per_s": endpoint_rate,
        "midpoint_target470": midpoint_target,
        "midpoint_hermite_rate470_per_s": midpoint_hermite_rate,
        "endpoint_metric_final_broyden560x560": endpoint_retraction_arrays[
            "final_metric_broyden560x560"
        ],
    }
    if midpoint_arrays is not None and midpoint_retraction_arrays is not None:
        arrays.update({
            "midpoint_retracted_coordinate470": midpoint_retraction_arrays[
                "recovered_original_coordinate470"
            ],
            "midpoint_retracted_primitive_state": midpoint_retraction_arrays[
                "primitive_state"
            ],
            "midpoint_coordinate_rate470_per_s": midpoint_arrays[
                "coordinate_free_rate470_per_s"
            ],
        })
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
            rows.append({
                "case": ARTIFACT,
                "path": str(path.relative_to(ROOT)),
                "bytes": str(path.stat().st_size),
                "sha256": helper._sha(path),
                "scientific_status": status,
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
        "passed": summary["passed"],
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


def _canonicalize(
    metrics: dict,
    arrays: dict[str, np.ndarray],
    lock: dict,
    identity: dict,
) -> dict:
    helper = _helper()
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("boundary-crossing result already exists")
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(
        CANONICAL_DIRECTORY / "boundary_execution_metrics.json", metrics
    )
    _save_npz(CANONICAL_DIRECTORY / "boundary_execution_arrays.npz", arrays)
    helper._write_json(CANONICAL_DIRECTORY / "input_lock.json", {
        "manifest_hashes": lock["hashes"],
        "manifest_classification": lock["summary"]["classification"],
        "execution_identity": identity,
    })
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": bool(metrics["passed"]),
        "metric_chart_boundary_crossing_passed": bool(metrics["passed"]),
        "historical_f25fi_rejection_preserved": True,
        "new_accepted_segments": metrics["gate_values"]["new_accepted_segments"],
        "short_suffix_manifest_authorized": bool(metrics["passed"]),
        "cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT if metrics["passed"] else None,
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    helper._write_json(CANONICAL_DIRECTORY / "provenance.json", {
        "runner": THIS_RUNNER,
        "test": THIS_TEST,
        "implementation_commit": identity["implementation_commit"],
        "implementation_tree": identity["implementation_tree"],
        "source_hashes": identity["source_hashes"],
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
    values = metrics["gate_values"]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join((
            "# Metric-chart boundary-crossing execution",
            "",
            f"Classification: `{metrics['classification']}`.",
            "",
            f"The prospective 0.25 ms segment {'was accepted' if metrics['passed'] else 'was not accepted'}. The endpoint raw coordinate condition was above the historical 2500 boundary: `{values['endpoint_historical_raw_condition_crossed']}`; the maximum new metric condition was `{values['maximum_metric_coordinate_jacobian_condition']:.6e}`.",
            "",
            f"Endpoint integral defect: `{values['endpoint_integral_defect']:.6e}`. Blind midpoint rate defect: `{values['blind_midpoint_rate_defect']:.6e}`. Endpoint retraction replay and accepted checkpoint roundtrip were `{values['endpoint_retraction_replay_bitwise']}` and `{values['checkpoint_roundtrip_bitwise']}`.",
            "",
            "All physics, ledgers, observables, and time integration remained in the original primitive/original coordinate representation. The historical f25fi rejection is preserved.",
            "",
            f"Authorized next artifact: `{AUTHORIZED_NEXT if metrics['passed'] else None}`.",
            "",
        )),
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
