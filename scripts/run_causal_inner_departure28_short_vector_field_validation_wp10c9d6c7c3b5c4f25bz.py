#!/usr/bin/env python3
"""Validate one short forecast of the frozen departure-28 vector field."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_departure28_short_vector_field_manifest_wp10c9d6c7c3b5c4f25by as manifest  # noqa: E402
import run_causal_inner_face36_fixed_q_primary_bounded_continuation_wp10c9d6c7c3b5c4f24e14d as continuation_tools  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_fixed_q import (  # noqa: E402
    causal_five_field_fixed_q_continuation_state,
    causal_five_field_fixed_q_continuation_states_equal,
    load_causal_five_field_fixed_q_continuation_state,
    save_causal_five_field_fixed_q_continuation_state,
    solve_causal_five_field_fixed_q_bdf,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25bz"
MANIFEST_COMMIT = "5181c241b36ea7af4372c750e484e97367f61174"
MANIFEST_PARENT = "05727ac26ad9b95515990aaa9ea52f4ebaa0438c"
MANIFEST_TREE = "23bc1e6a30d884cd9d3ba737621b5b924043d90f"
PASS_CLASSIFICATION = "departure28_short_reduced_vector_field_validated"
FAIL_CLASSIFICATION = "departure28_short_reduced_vector_field_validation_failed"
AUTHORIZED_NEXT = (
    "definitions_only_fixed_Q_fast_attractor_and_normal_hyperbolicity_manifest"
)

ARTIFACT = (
    "causal_inner_departure28_short_vector_field_validation_"
    "wp10c9d6c7c3b5c4f25bz"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_departure28_short_vector_field_validation_"
    "wp10c9d6c7c3b5c4f25bz.py"
)
THIS_TEST = (
    "tests/test_causal_inner_departure28_short_vector_field_validation_"
    "wp10c9d6c7c3b5c4f25bz.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_DEPARTURE28_SHORT_VECTOR_FIELD_"
    "VALIDATION_WP10C9D6C7C3B5C4F25BZ_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
FORECAST_ARRAYS = SCRATCH_DIRECTORY / "forecast.npz"
FORECAST_LOCK = SCRATCH_DIRECTORY / "forecast_lock.json"
READINESS_METRICS = SCRATCH_DIRECTORY / "readiness_metrics.json"
TRUTH_RESULT = SCRATCH_DIRECTORY / "result_warm_4.npz"
TRUTH_METRICS = SCRATCH_DIRECTORY / "metrics_warm_4.json"
TRUTH_CHECKPOINT = SCRATCH_DIRECTORY / "checkpoint_warm_4.npz"

THREAD_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
SOURCE_FILES = (
    THIS_RUNNER,
    THIS_TEST,
    manifest.THIS_RUNNER,
    manifest.THIS_TEST,
    manifest.parent.THIS_RUNNER,
    manifest.parent.THIS_TEST,
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_fixed_q.py",
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_bdf.py",
)


_plain = manifest._plain
_read = manifest._read
_write_json = manifest._write_json
_sha = manifest._sha
_checksums = manifest._checksums


def _write_npz(path: Path, **arrays) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _source_hashes() -> dict[str, str]:
    return {relative: _sha(ROOT / relative) for relative in SOURCE_FILES}


def _validate_manifest(*, require_clean: bool) -> dict:
    if _git("rev-parse", MANIFEST_COMMIT) != MANIFEST_COMMIT:
        raise RuntimeError("short-vector-field manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("short-vector-field manifest lineage changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("short-vector-field manifest tree changed")
    hashes = _checksums(manifest.CANONICAL_DIRECTORY)
    summary = _read(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = _read(manifest.CANONICAL_DIRECTORY / "contract.json")
    lock = _read(manifest.CANONICAL_DIRECTORY / "input_lock.json")
    provenance = _read(manifest.CANONICAL_DIRECTORY / "provenance.json")
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or summary["classification"] != manifest.CLASSIFICATION
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["prospective_truth_roots_authorized"] != 1
        or summary["model_470_is_final_cycle_integrator"]
        or contract["decision"]["pass_classification"] != PASS_CLASSIFICATION
        or contract["decision"]["failure_classification"] != FAIL_CLASSIFICATION
        or contract["decision"]["pass_authorizes_only"] != AUTHORIZED_NEXT
    ):
        raise RuntimeError("short-vector-field frozen contract changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"manifest source changed: {relative}")
    inputs = manifest._decisive_inputs()
    for name, path in inputs.items():
        if _sha(path) != lock["decisive_input_hashes"][name]:
            raise RuntimeError(f"decisive input changed: {path}")
    for name, expected in manifest.parent.geometry.chart_tools.coordinate_tools.THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("short-vector-field validation requires a clean tracked tree")
    return {"summary": summary, "contract": contract, "lock": lock, "hashes": hashes}


def _load_coefficients() -> dict[str, np.ndarray]:
    path = manifest.parent.CANONICAL_DIRECTORY / "frozen_coefficients.npz"
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name], dtype=float) for name in source.files}


class ReducedVectorField:
    """Frozen algebraic 470-state model at the accepted warm_3 anchor."""

    def __init__(self):
        parent = manifest.parent
        inputs = parent._load_inputs()
        self.inputs = inputs
        self.coefficients = _load_coefficients()
        self.components = parent.geometry.base.high_chart._prepare_components()
        self.base_state = np.asarray(self.components["state"], dtype=float)
        self.columns = np.asarray(self.components["columns"], dtype=float)
        self.restriction = np.asarray(
            inputs["online_geometry"]["online_coordinate_restriction"], dtype=float
        )
        self.lifting = np.asarray(
            inputs["online_geometry"]["online_coordinate_lifting"], dtype=float
        )
        self.memory_basis = np.asarray(
            inputs["online_geometry"]["stable_memory_coordinate_basis"], dtype=float
        )
        self.departure_basis = np.asarray(
            inputs["online_geometry"]["departure_coordinate_basis"], dtype=float
        )
        self.generator = np.asarray(inputs["generator"], dtype=float)
        self.base_rate = np.asarray(inputs["base_rate"], dtype=float)
        self.curvature_basis = np.asarray(inputs["curvature_basis"], dtype=float)
        self.energy_directions = np.asarray(
            inputs["database"]["energy_directions"], dtype=float
        )
        self.base_coordinate = np.asarray(
            self.components["coordinate_target"], dtype=float
        )
        self._field_cache: dict[bytes, np.ndarray] = {}

    def coordinate(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        physical, factors = (
            manifest.parent.geometry.chart_tools._coordinate_value_with_factors(
                state, self.components
            )
        )
        delta = ((np.asarray(state) - self.base_state) / self.columns).ravel()
        return np.concatenate(
            (
                np.asarray(physical) - self.base_coordinate,
                self.memory_basis.T @ delta,
                self.departure_basis.T @ delta,
            )
        ), np.asarray(factors, dtype=float)

    def nonlinear_departure(self, departure: np.ndarray) -> np.ndarray:
        coordinate = np.asarray(departure, dtype=float)
        if float(np.linalg.norm(coordinate)) <= np.finfo(float).tiny:
            return np.zeros(manifest.DEPARTURE_DIMENSION)
        return manifest.parent._predict_rate(coordinate, self.coefficients)

    def decoded_delta(self, coordinate: np.ndarray) -> np.ndarray:
        y = np.asarray(coordinate, dtype=float)
        departure = y[-manifest.DEPARTURE_DIMENSION :]
        curvature = manifest.parent._predict_curvature(
            self.energy_directions.T @ departure, self.coefficients
        )
        return self.lifting @ y + self.curvature_basis @ curvature

    def decoded_state(self, coordinate: np.ndarray) -> np.ndarray:
        return self.base_state + (
            self.columns.ravel() * self.decoded_delta(coordinate)
        ).reshape(self.base_state.shape)

    def field(self, coordinate: np.ndarray) -> np.ndarray:
        y = np.asarray(coordinate, dtype=float)
        key = y.tobytes()
        if key in self._field_cache:
            return np.array(self._field_cache[key], copy=True)
        delta = self.decoded_delta(y)
        state = self.base_state + (
            self.columns.ravel() * delta
        ).reshape(self.base_state.shape)
        full_rate = self.base_rate + self.generator @ delta
        full_rate = full_rate + self.departure_basis @ self.nonlinear_departure(
            y[-manifest.DEPARTURE_DIMENSION :]
        )
        physical_jacobian, _metrics = (
            manifest.parent.geometry.chart_tools._coordinate_jacobian(
                state, self.components
            )
        )
        result = np.concatenate(
            (
                physical_jacobian @ full_rate,
                self.memory_basis.T @ full_rate,
                self.departure_basis.T @ full_rate,
            )
        )
        self._field_cache[key] = np.array(result, copy=True)
        return result


def _rk4(field, initial: np.ndarray, duration: float, substeps: int) -> np.ndarray:
    state = np.asarray(initial, dtype=float).copy()
    timestep = float(duration) / int(substeps)
    for _ in range(int(substeps)):
        k1 = field(state)
        k2 = field(state + 0.5 * timestep * k1)
        k3 = field(state + 0.5 * timestep * k2)
        k4 = field(state + timestep * k3)
        state += timestep * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return state


def _relative_error(
    predicted: np.ndarray, truth: np.ndarray, reference_change: np.ndarray
) -> float:
    return float(
        np.linalg.norm(np.asarray(predicted) - np.asarray(truth))
        / max(float(np.linalg.norm(reference_change)), np.finfo(float).tiny)
    )


def _endpoint_errors(
    predicted: np.ndarray, truth: np.ndarray, start: np.ndarray
) -> dict:
    slices = {
        "full": slice(None),
        "q162": slice(0, manifest.PHYSICAL_DIMENSION),
        "z280": slice(
            manifest.PHYSICAL_DIMENSION,
            manifest.PHYSICAL_DIMENSION + manifest.MEMORY_DIMENSION,
        ),
        "a28": slice(-manifest.DEPARTURE_DIMENSION, None),
    }
    return {
        name: _relative_error(
            np.asarray(predicted)[selection],
            np.asarray(truth)[selection],
            np.asarray(truth)[selection] - np.asarray(start)[selection],
        )
        for name, selection in slices.items()
    }


def _state_audit(model: ReducedVectorField, coordinate: np.ndarray) -> dict:
    state = model.decoded_state(coordinate)
    physical = manifest.parent.geometry.chart_tools._state_audit(
        model.components["context"], state
    )
    decoded_coordinate, factors = model.coordinate(state)
    mismatch = _endpoint_errors(decoded_coordinate, coordinate, np.zeros_like(coordinate))
    return {
        "minimum_reconstruction_factor": min(
            float(np.min(factors)), physical["minimum_reconstruction_factor"]
        ),
        "maximum_H_over_R": physical["maximum_h_over_r"],
        "minimum_scattering_optical_depth": physical[
            "minimum_scattering_optical_depth"
        ],
        "decoder_coordinate_relative_mismatch": mismatch,
    }


def _load_reference_states(model: ReducedVectorField) -> dict:
    data = continuation_tools.e1._state_data("primary_20ms")
    context = data["context"]
    warm_2 = load_causal_five_field_fixed_q_continuation_state(
        manifest.RETRY_DIRECTORY / "checkpoint_warm_2.npz", context
    )
    warm_3 = load_causal_five_field_fixed_q_continuation_state(
        manifest.RETRY_DIRECTORY / "checkpoint_warm_3.npz", context
    )
    return {"data": data, "warm_2": warm_2, "warm_3": warm_3}


def _structural_metrics(model: ReducedVectorField, references: dict) -> dict:
    with np.load(
        manifest.parent.CANONICAL_DIRECTORY / "departure28_closure.npz",
        allow_pickle=False,
    ) as source:
        validated_base_rate = np.asarray(
            source["base_fixed_Q_rate_per_second"], dtype=float
        )
    anchor_equal = np.array_equal(
        model.base_state, references["warm_3"].current_primitive_charts
    )
    base_rate_equal = np.array_equal(model.base_rate, validated_base_rate)
    lifting_defect = float(
        np.max(
            np.abs(
                model.restriction @ model.lifting
                - np.eye(manifest.ONLINE_DIMENSION)
            )
        )
    )
    field_zero = model.field(np.zeros(manifest.ONLINE_DIMENSION))
    projected_base = model.restriction @ model.base_rate
    base_identity = float(
        np.linalg.norm(field_zero - projected_base)
        / max(float(np.linalg.norm(field_zero)), np.finfo(float).tiny)
    )
    nonlinear_zero = float(
        np.linalg.norm(model.nonlinear_departure(np.zeros(manifest.DEPARTURE_DIMENSION)))
    )
    return {
        "anchor_state_array_equal": anchor_equal,
        "anchor_base_rate_array_equal": base_rate_equal,
        "restriction_lifting_identity_defect": lifting_defect,
        "base_vector_field_relative_identity_defect": base_identity,
        "nonlinear_departure_at_zero_norm": nonlinear_zero,
    }


def _forecast() -> dict:
    frozen = _validate_manifest(require_clean=True)
    if SCRATCH_DIRECTORY.exists() or CANONICAL_DIRECTORY.exists():
        raise RuntimeError("short-vector-field forecast already exists")
    SCRATCH_DIRECTORY.mkdir(parents=True)
    began = time.perf_counter()
    model = ReducedVectorField()
    references = _load_reference_states(model)
    structural = _structural_metrics(model, references)
    warm_2_state = references["warm_2"].current_primitive_charts
    warm_3_state = references["warm_3"].current_primitive_charts
    warm_2_y, warm_2_factors = model.coordinate(warm_2_state)
    warm_3_y, warm_3_factors = model.coordinate(warm_3_state)
    duration = manifest.TIMESTEP_SECONDS
    retro_coarse = _rk4(model.field, warm_2_y, duration, 1)
    retro_refined = _rk4(model.field, warm_2_y, duration, 2)
    retro_errors = _endpoint_errors(retro_refined, warm_3_y, warm_2_y)
    retro_step = _relative_error(
        retro_coarse, retro_refined, retro_refined - warm_2_y
    )
    warm_2_delta = ((warm_2_state - model.base_state) / model.columns).ravel()
    decoded_warm_2 = model.decoded_delta(warm_2_y)
    decoded_start_error = float(
        np.linalg.norm(decoded_warm_2 - warm_2_delta)
        / max(float(np.linalg.norm(warm_2_delta)), np.finfo(float).tiny)
    )
    prospective_start = np.zeros(manifest.ONLINE_DIMENSION)
    forecast_coarse = _rk4(model.field, prospective_start, duration, 1)
    forecast_refined = _rk4(model.field, prospective_start, duration, 2)
    forecast_step = _relative_error(
        forecast_coarse, forecast_refined, forecast_refined
    )
    forecast_state = model.decoded_state(forecast_refined)
    forecast_field = model.field(forecast_refined)
    retrospective_audit = _state_audit(model, retro_refined)
    prospective_audit = _state_audit(model, forecast_refined)
    gates = frozen["contract"]["binding_retrospective_readiness_gates"]
    structural_gates = frozen["contract"]["binding_structural_gates"]
    checks = {
        "anchor_state": structural["anchor_state_array_equal"]
        == structural_gates["anchor_state_array_equal"],
        "anchor_rate": structural["anchor_base_rate_array_equal"]
        == structural_gates["anchor_base_rate_array_equal"],
        "lifting": structural["restriction_lifting_identity_defect"]
        <= structural_gates["restriction_lifting_identity_defect_max"],
        "base_field": structural["base_vector_field_relative_identity_defect"]
        <= structural_gates["base_vector_field_relative_identity_defect_max"],
        "nonlinear_zero": structural["nonlinear_departure_at_zero_norm"]
        <= structural_gates["nonlinear_departure_at_zero_norm_max"],
        "decoded_start": decoded_start_error
        <= gates["decoded_start_full_scaled_state_relative_error_max"],
        "endpoint_full": retro_errors["full"]
        <= gates["refined_endpoint_full_coordinate_relative_error_max"],
        "endpoint_q": retro_errors["q162"]
        <= gates["refined_endpoint_q162_relative_error_max"],
        "endpoint_z": retro_errors["z280"]
        <= gates["refined_endpoint_z280_relative_error_max"],
        "endpoint_a": retro_errors["a28"]
        <= gates["refined_endpoint_a28_relative_error_max"],
        "step_refinement": retro_step
        <= gates["coarse_refined_endpoint_relative_difference_max"],
        "reconstruction": min(
            float(np.min(warm_2_factors)),
            float(np.min(warm_3_factors)),
            retrospective_audit["minimum_reconstruction_factor"],
            prospective_audit["minimum_reconstruction_factor"],
        )
        >= gates["minimum_reconstruction_factor"],
        "height": max(
            retrospective_audit["maximum_H_over_R"],
            prospective_audit["maximum_H_over_R"],
        )
        <= gates["maximum_H_over_R"],
        "optical_depth": min(
            retrospective_audit["minimum_scattering_optical_depth"],
            prospective_audit["minimum_scattering_optical_depth"],
        )
        >= gates["minimum_scattering_optical_depth"],
    }
    passed = all(checks.values())
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "passed": passed,
        "checks": checks,
        "structural": structural,
        "retrospective": {
            "endpoint_relative_errors": retro_errors,
            "coarse_refined_endpoint_relative_difference": retro_step,
            "decoded_start_full_scaled_state_relative_error": decoded_start_error,
            "state_audit": retrospective_audit,
        },
        "prospective_forecast": {
            "coarse_refined_endpoint_relative_difference": forecast_step,
            "state_audit": prospective_audit,
        },
        "new_truth_roots_at_forecast_lock": 0,
        "wall_seconds": time.perf_counter() - began,
    }
    _write_npz(
        FORECAST_ARRAYS,
        warm_2_coordinate=warm_2_y,
        warm_3_coordinate=warm_3_y,
        retrospective_coarse_coordinate=retro_coarse,
        retrospective_refined_coordinate=retro_refined,
        prospective_coarse_coordinate=forecast_coarse,
        prospective_refined_coordinate=forecast_refined,
        prospective_decoded_primitive_state=forecast_state,
        prospective_endpoint_vector_field=forecast_field,
    )
    _write_json(READINESS_METRICS, metrics)
    lock = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "passed": passed,
        "forecast_sha256": _sha(FORECAST_ARRAYS),
        "readiness_metrics_sha256": _sha(READINESS_METRICS),
        "coefficient_sha256": _sha(
            manifest.parent.CANONICAL_DIRECTORY / "frozen_coefficients.npz"
        ),
        "new_truth_roots_before_lock": 0,
        "source_hashes": _source_hashes(),
        "execution_commit": _git("rev-parse", "HEAD"),
        "execution_tree": _git("rev-parse", "HEAD^{tree}"),
        "manifest_hashes": frozen["hashes"],
    }
    _write_json(FORECAST_LOCK, lock)
    print(json.dumps(_plain(metrics), indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)
    return metrics


def _validate_forecast_lock() -> dict:
    if not FORECAST_LOCK.is_file() or not FORECAST_ARRAYS.is_file() or not READINESS_METRICS.is_file():
        raise RuntimeError("forecast lock is incomplete")
    lock = _read(FORECAST_LOCK)
    if (
        not lock["passed"]
        or lock["new_truth_roots_before_lock"] != 0
        or lock["forecast_sha256"] != _sha(FORECAST_ARRAYS)
        or lock["readiness_metrics_sha256"] != _sha(READINESS_METRICS)
        or lock["coefficient_sha256"]
        != _sha(manifest.parent.CANONICAL_DIRECTORY / "frozen_coefficients.npz")
        or lock["source_hashes"] != _source_hashes()
    ):
        raise RuntimeError("forecast lock changed")
    return lock


def _execution_identity(forecast_lock: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "execution_commit": _git("rev-parse", "HEAD"),
        "execution_tree": _git("rev-parse", "HEAD^{tree}"),
        "source_hashes": _source_hashes(),
        "forecast_sha256": forecast_lock["forecast_sha256"],
    }


def _truth_root(data: dict, continuation, identity: dict):
    rate, multiplier = continuation_tools._predictors(
        continuation, data["columns"]
    )
    events = []

    def progress(payload: dict) -> None:
        plain = _plain(payload)
        events.append(plain)
        print(f"f25bz warm_4: {plain}", flush=True)

    began_wall = time.perf_counter()
    began_process = time.process_time()
    result = solve_causal_five_field_fixed_q_bdf(
        data["context"],
        continuation.current_primitive_charts,
        manifest.TIMESTEP_SECONDS,
        rate,
        multiplier,
        None,
        order=2,
        history=continuation.history,
        primitive_column_scales=data["columns"],
        conservation_row_scales=data["rows"],
        parent_cell_indices=data["layout"].parent_cell_indices,
        refinement_ratio=data["layout"].refinement_ratio,
        q3_target=continuation.q3_target,
        constraint_row_scales=continuation.constraint_row_scales,
        reaction_channel_basis="frozen_normalized",
        reaction_channel_transform=continuation.next_reaction_channel_transform,
        residual_tolerance=1.0e-10,
        constraint_tolerance=1.0e-12,
        ledger_tolerance=1.0e-12,
        storage_parity_tolerance=1.0e-9,
        minimum_reconstruction_factor=1.0 - 1.0e-12,
        maximum_schur_condition_number=1.0e8,
        maximum_scaled_primitive_change=5.0e-3,
        maximum_newton_iterations=8,
        maximum_line_search_iterations=12,
        refresh_exact_jacobian=True,
        maximum_exact_jacobian_refreshes=1,
        exact_jacobian_refresh_policy="on_line_search_failure_or_iteration_reserve",
        initial_nonlinear_solver_state=continuation.nonlinear_solver_state,
        initial_exact_jacobian_required=False,
        solver_state_provenance=identity,
        physical_state_audit=continuation_tools.e1._state_audit,
        require_physical_state_audit=True,
        maximum_h_over_r=0.12,
        minimum_scattering_optical_depth=1.0,
        progress_callback=progress,
    )
    metrics = continuation_tools._result_metrics(
        result,
        events,
        time.perf_counter() - began_wall,
        time.process_time() - began_process,
    )
    metrics.update(
        {
            "label": "warm_4",
            "timestep_seconds": manifest.TIMESTEP_SECONDS,
            "policy": {
                "cold": False,
                "initial_exact_jacobian_required": False,
                "maximum_exact_jacobian_refreshes": 1,
                "use_carried_solver_state": True,
                "exact_jacobian_refresh_policy": (
                    "on_line_search_failure_or_iteration_reserve"
                ),
            },
        }
    )
    continuation_tools._save_result(TRUTH_RESULT, result, metrics)
    _write_json(TRUTH_METRICS, metrics)
    return result, metrics


def _roundtrip_truth_checkpoint(result, data: dict, start, identity: dict) -> dict:
    continuation = causal_five_field_fixed_q_continuation_state(
        result,
        data["context"],
        start.current_primitive_charts,
        primitive_column_scales=data["columns"],
        conservation_row_scales=data["rows"],
        parent_cell_indices=data["layout"].parent_cell_indices,
        refinement_ratio=data["layout"].refinement_ratio,
        elapsed_time_seconds=start.elapsed_time_seconds + manifest.TIMESTEP_SECONDS,
        completed_steps=start.completed_steps + 1,
        provenance=identity,
    )
    timings = {}
    save_causal_five_field_fixed_q_continuation_state(
        TRUTH_CHECKPOINT, data["context"], continuation, timing_accumulator=timings
    )
    loaded = load_causal_five_field_fixed_q_continuation_state(
        TRUTH_CHECKPOINT,
        data["context"],
        expected_provenance=identity,
        timing_accumulator=timings,
    )
    return {
        "bitwise_roundtrip": causal_five_field_fixed_q_continuation_states_equal(
            continuation, loaded
        ),
        "sha256": _sha(TRUTH_CHECKPOINT),
        "bytes": TRUTH_CHECKPOINT.stat().st_size,
        **timings,
    }


def _rate_relative(predicted: np.ndarray, truth: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(predicted) - np.asarray(truth))
        / max(
            float(np.linalg.norm(predicted)),
            float(np.linalg.norm(truth)),
            np.finfo(float).tiny,
        )
    )


def _rate_errors(predicted: np.ndarray, truth: np.ndarray) -> dict:
    zeros = np.zeros_like(truth)
    # _endpoint_errors uses the truth-minus-start norm, which is the desired
    # fixed truth norm here.
    return _endpoint_errors(predicted, truth, zeros)


def _truth() -> dict:
    frozen = _validate_manifest(require_clean=True)
    forecast_lock = _validate_forecast_lock()
    if CANONICAL_DIRECTORY.exists() or TRUTH_RESULT.exists():
        raise RuntimeError("short-vector-field truth already exists")
    model = ReducedVectorField()
    references = _load_reference_states(model)
    data = references["data"]
    start = references["warm_3"]
    identity = _execution_identity(forecast_lock)
    result, root_metrics = _truth_root(data, start, identity)
    checkpoint = {"bitwise_roundtrip": False}
    if result.accepted:
        checkpoint = _roundtrip_truth_checkpoint(result, data, start, identity)
    with np.load(FORECAST_ARRAYS, allow_pickle=False) as source:
        forecast_y = np.asarray(source["prospective_refined_coordinate"], dtype=float)
        forecast_coarse_y = np.asarray(source["prospective_coarse_coordinate"], dtype=float)
        forecast_state = np.asarray(
            source["prospective_decoded_primitive_state"], dtype=float
        )
        forecast_field = np.asarray(
            source["prospective_endpoint_vector_field"], dtype=float
        )
    truth_y, truth_factors = model.coordinate(result.primitive_charts)
    start_y = np.zeros(manifest.ONLINE_DIMENSION)
    endpoint_errors = _endpoint_errors(forecast_y, truth_y, start_y)
    truth_scaled_delta = (
        (result.primitive_charts - model.base_state) / model.columns
    ).ravel()
    state_error = float(
        np.linalg.norm(
            ((forecast_state - result.primitive_charts) / model.columns).ravel()
        )
        / max(float(np.linalg.norm(truth_scaled_delta)), np.finfo(float).tiny)
    )
    truth_jacobian, _ = manifest.parent.geometry.chart_tools._coordinate_jacobian(
        result.primitive_charts, model.components
    )
    truth_rate = np.concatenate(
        (
            truth_jacobian @ result.scaled_rate_per_s,
            model.memory_basis.T @ result.scaled_rate_per_s,
            model.departure_basis.T @ result.scaled_rate_per_s,
        )
    )
    rate_errors = _rate_errors(forecast_field, truth_rate)
    step_error = _relative_error(forecast_coarse_y, forecast_y, forecast_y)
    forecast_audit = _state_audit(model, forecast_y)
    gates = frozen["contract"]["binding_prospective_forecast_gates"]
    checks = {
        "root_accepted": result.accepted == gates["truth_root_accepted"],
        "root_residual": root_metrics["maximum_scaled_residual"]
        <= gates["truth_root_maximum_scaled_residual"],
        "root_Q3": root_metrics["maximum_Q3_relative_defect"]
        <= gates["truth_root_maximum_Q3_relative_defect"],
        "root_reconstruction": root_metrics["minimum_path_reconstruction_factor"]
        >= gates["truth_root_minimum_reconstruction_factor"],
        "root_height": root_metrics["maximum_H_over_R"]
        <= gates["truth_root_maximum_H_over_R"],
        "root_optical_depth": root_metrics["minimum_scattering_optical_depth"]
        >= gates["truth_root_minimum_scattering_optical_depth"],
        "root_exact_assemblies": root_metrics["exact_Jacobian_assemblies"]
        <= gates["truth_root_maximum_exact_Jacobian_assemblies"],
        "checkpoint_roundtrip": checkpoint["bitwise_roundtrip"],
        "endpoint_full": endpoint_errors["full"]
        <= gates["forecast_full_coordinate_relative_error_max"],
        "endpoint_q": endpoint_errors["q162"]
        <= gates["forecast_q162_relative_error_max"],
        "endpoint_z": endpoint_errors["z280"]
        <= gates["forecast_z280_relative_error_max"],
        "endpoint_a": endpoint_errors["a28"]
        <= gates["forecast_a28_relative_error_max"],
        "state": state_error
        <= gates["forecast_full_scaled_state_relative_error_max"],
        "rate_full": rate_errors["full"]
        <= gates["endpoint_vector_field_full_relative_error_max"],
        "rate_q": rate_errors["q162"]
        <= gates["endpoint_vector_field_q162_relative_error_max"],
        "rate_z": rate_errors["z280"]
        <= gates["endpoint_vector_field_z280_relative_error_max"],
        "rate_a": rate_errors["a28"]
        <= gates["endpoint_vector_field_a28_relative_error_max"],
        "step_refinement": step_error
        <= gates["coarse_refined_endpoint_relative_difference_max"],
        "truth_factors": float(np.min(truth_factors))
        >= gates["truth_root_minimum_reconstruction_factor"],
    }
    passed = all(checks.values())
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,
        "passed": passed,
        "checks": checks,
        "forecast_lock": forecast_lock,
        "readiness": _read(READINESS_METRICS),
        "truth_root": root_metrics,
        "truth_checkpoint": checkpoint,
        "forecast_endpoint_coordinate_relative_errors": endpoint_errors,
        "forecast_full_scaled_state_relative_error": state_error,
        "endpoint_vector_field_relative_errors": rate_errors,
        "coarse_refined_endpoint_relative_difference": step_error,
        "forecast_state_audit": forecast_audit,
        "truth_minimum_reconstruction_factor": float(np.min(truth_factors)),
        "new_truth_roots": 1,
        "propagated_reference_states": int(result.accepted),
    }
    _write_json(SCRATCH_DIRECTORY / "validation_metrics.json", metrics)
    _write_npz(
        SCRATCH_DIRECTORY / "validation_arrays.npz",
        forecast_coordinate=forecast_y,
        truth_coordinate=truth_y,
        forecast_primitive_state=forecast_state,
        truth_primitive_state=result.primitive_charts,
        forecast_endpoint_vector_field=forecast_field,
        truth_BDF_coordinate_rate=truth_rate,
    )
    _canonicalize(metrics, frozen)
    print(json.dumps(_plain(metrics), indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)
    return metrics


def _canonicalize(metrics: dict, frozen: dict) -> None:
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("short-vector-field result already canonicalized")
    CANONICAL_DIRECTORY.mkdir(parents=True)
    for name in (
        "forecast.npz",
        "forecast_lock.json",
        "readiness_metrics.json",
        "result_warm_4.npz",
        "metrics_warm_4.json",
        "checkpoint_warm_4.npz",
        "validation_metrics.json",
        "validation_arrays.npz",
    ):
        source = SCRATCH_DIRECTORY / name
        if source.is_file():
            shutil.copy2(source, CANONICAL_DIRECTORY / name)
    passed = bool(metrics["passed"])
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": passed,
        "retrospective_readiness_passed": metrics["readiness"]["passed"],
        "prospective_forecast_passed": passed,
        "new_truth_roots": metrics["new_truth_roots"],
        "accepted_truth_roots": int(metrics["truth_root"]["accepted"]),
        "model_470_role": "offline_fast_transient_and_closure_model",
        "fast_attractor_manifest_authorized": passed,
        "physical_microburst_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT if passed else None,
    }
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    _write_json(
        CANONICAL_DIRECTORY / "parent_lock.json",
        {
            "manifest_commit": MANIFEST_COMMIT,
            "manifest_parent": MANIFEST_PARENT,
            "manifest_tree": MANIFEST_TREE,
            "manifest_hashes": frozen["hashes"],
        },
    )
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "CERTIFIED" if passed else "REJECTED",
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "source_hashes": _source_hashes(),
            "thread_environment": THREAD_ENVIRONMENT,
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    )
    names = tuple(sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()))
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    endpoint = metrics["forecast_endpoint_coordinate_relative_errors"]
    rate = metrics["endpoint_vector_field_relative_errors"]
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Departure-28 short-vector-field validation WP10c9d6c7c3b5c4f25bz",
                "",
                "## Classification",
                "",
                f"`{metrics['classification']}`",
                "",
                f"Retrospective readiness passed: `{metrics['readiness']['passed']}`.",
                "",
                f"Prospective endpoint errors (full/q/z/a): `{endpoint['full']:.6e}`, `{endpoint['q162']:.6e}`, `{endpoint['z280']:.6e}`, `{endpoint['a28']:.6e}`.",
                "",
                f"Endpoint vector-field errors (full/q/z/a): `{rate['full']:.6e}`, `{rate['q162']:.6e}`, `{rate['z280']:.6e}`, `{rate['a28']:.6e}`.",
                "",
                "The 470-state model remains an offline fast/transient closure model. It is not a cycle integrator.",
                "",
                (
                    f"Authorized next artifact: `{AUTHORIZED_NEXT}`."
                    if passed
                    else "No further execution is authorized."
                ),
                "",
            )
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)


def _update_catalog(summary: dict) -> None:
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    status = "CERTIFIED" if summary["passed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha(path),
                    "scientific_status": status,
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case", "path", "bytes", "sha256", "scientific_status"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read(CANONICAL_SUMMARY)
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
            "latest_source_parent_commit": MANIFEST_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate", action="store_true")
    group.add_argument("--forecast", action="store_true")
    group.add_argument("--truth", action="store_true")
    args = parser.parse_args()
    if args.validate:
        print(json.dumps(_plain(_validate_manifest(require_clean=False)), indent=2, sort_keys=True))
        return 0
    if args.forecast:
        _forecast()
        return 0
    _truth()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
