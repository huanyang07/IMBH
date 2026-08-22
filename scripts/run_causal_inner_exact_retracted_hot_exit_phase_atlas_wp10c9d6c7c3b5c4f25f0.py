#!/usr/bin/env python3
"""Advance one moving exact-chart phase-atlas window toward hot exit."""

from __future__ import annotations

import argparse
import csv
import hashlib
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

from imri_qpe.layer3_minidisk_1d.causal_inner_fixed_q import causal_five_field_exterior_q3  # noqa: E402
import run_causal_inner_exact_geometric_470_chart_preflight_wp10c9d6c7c3b5c4f25de as exact_chart  # noqa: E402
import run_causal_inner_exact_retracted_hot_exit_phase_atlas_manifest_wp10c9d6c7c3b5c4f25ef as manifest  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25f0"
CONTINUE_CLASSIFICATION = "exact_retracted_hot_exit_phase_window_passed_event_not_yet_observed"
EXIT_CLASSIFICATION = "persistent_hot_exit_candidate_observed_endpoint_refinement_authorized"
BUDGET_CLASSIFICATION = "exact_retracted_hot_exit_phase_atlas_budget_exhausted_without_event"
FAIL_CLASSIFICATION = "exact_retracted_hot_exit_phase_window_rejected_last_accepted_endpoint_preserved"
FINAL_WORK_PACKAGE = "WP10c9d6c7c3b5c4f25f1"
ARTIFACT_PREFIX = "causal_inner_exact_retracted_hot_exit_phase_atlas_wp10c9d6c7c3b5c4f25f0"
SCRATCH_ROOT = ROOT / "outputs/checkpoints" / ARTIFACT_PREFIX
THIS_RUNNER = "scripts/run_causal_inner_exact_retracted_hot_exit_phase_atlas_wp10c9d6c7c3b5c4f25f0.py"
THIS_TEST = "tests/test_causal_inner_exact_retracted_hot_exit_phase_atlas_wp10c9d6c7c3b5c4f25f0.py"


def _helper():
    return manifest._helper()


def _adaptive():
    return manifest.rejected


def _post():
    return _adaptive()._post()


def _stage_directory(index: int) -> Path:
    return ROOT / "results/canonical" / f"{ARTIFACT_PREFIX}_window_{index:02d}"


def _scratch_directory(index: int) -> Path:
    return SCRATCH_ROOT / f"window_{index:02d}"


def _report_path(index: int) -> Path:
    return ROOT / "docs/reports/current" / (
        "CODEX_CAUSAL_INNER_EXACT_RETRACTED_HOT_EXIT_PHASE_ATLAS_WINDOW_"
        f"{index:02d}_WP10C9D6C7C3B5C4F25F0_2026-08-21.md"
    )


def _validate_manifest(*, require_clean: bool) -> dict:
    helper = _helper()
    hashes = helper._validate_checksums(manifest.CANONICAL_DIRECTORY)
    summary = helper._read(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = helper._read(
        manifest.CANONICAL_DIRECTORY / "exact_retracted_hot_exit_phase_atlas_contract.json"
    )
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or not summary["exact_retracted_phase_atlas_execution_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["work_package"] != manifest.WORK_PACKAGE
    ):
        raise RuntimeError("moving exact-chart manifest changed")
    for relative, expected in contract["frozen_source_hashes"].items():
        if helper._sha(ROOT / relative) != expected:
            raise RuntimeError(f"frozen moving-chart source changed: {relative}")
    current = {name: helper._sha(path) for name, path in manifest._decisive_inputs().items()}
    if current != contract["decisive_input_hashes"]:
        raise RuntimeError("moving-chart decisive input changed")
    manifest._validate_parent(require_clean=False)
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("moving exact-chart window requires a clean tracked tree")
    return {"manifest_hashes": hashes, "contract": contract}


def _validate_order(index: int) -> dict:
    helper = _helper()
    if index == manifest.FIRST_WINDOW_INDEX:
        helper._validate_checksums(manifest.ACCEPTED_DIRECTORY)
        helper._validate_checksums(manifest.REJECTED_DIRECTORY)
        accepted = helper._read(manifest.ACCEPTED_DIRECTORY / "summary.json")
        rejected = helper._read(manifest.REJECTED_DIRECTORY / "summary.json")
        if not accepted["passed"] or accepted["window_index"] != 2 or rejected["passed"]:
            raise RuntimeError("recovery seed changed")
        return accepted
    previous = _stage_directory(index - 1)
    helper._validate_checksums(previous)
    summary = helper._read(previous / "summary.json")
    if (
        not summary["passed"]
        or not summary["next_window_authorized"]
        or summary["hot_exit_observed"]
        or int(summary["window_index"]) != index - 1
    ):
        raise RuntimeError("prior moving-chart window does not authorize this window")
    return summary


def _canonical_basis(rates: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, float]:
    values = np.asarray(rates, dtype=float)
    normalized = values / np.linalg.norm(values, axis=1)[:, None]
    _left, singular_values, right = np.linalg.svd(normalized, full_matrices=False)
    selected = None
    selected_rank = 0
    selected_defect = float("inf")
    for rank in manifest.RATE_BASIS_RANKS:
        if rank > right.shape[0]:
            continue
        basis = np.asarray(right[:rank].T, dtype=float)
        defect = float(np.max(np.linalg.norm(normalized - (normalized @ basis) @ basis.T, axis=1)))
        selected, selected_rank, selected_defect = basis, rank, defect
        if defect <= manifest.MAXIMUM_TRAINING_NORMAL_RATE_DEFECT:
            break
    if selected is None:
        raise RuntimeError("no moving-chart rate basis could be constructed")
    for column in range(selected.shape[1]):
        pivot = int(np.argmax(np.abs(selected[:, column])))
        if selected[pivot, column] < 0.0:
            selected[:, column] *= -1.0
    return selected, singular_values, selected_rank, selected_defect


def _next_duration(index: int, previous_metrics: dict | None) -> float:
    if index == manifest.FIRST_WINDOW_INDEX:
        return manifest.INITIAL_DURATION_SECONDS
    if previous_metrics is None:
        raise RuntimeError("missing prior moving-chart metrics")
    duration = float(previous_metrics["duration_seconds"])
    if previous_metrics["growth_margin_passed"]:
        return min(2.0 * duration, manifest.MAXIMUM_DURATION_SECONDS)
    return duration


def _anchor_retraction_metrics() -> dict:
    return {
        "coordinate_residual_infinity": 0.0,
        "gauge_residual_infinity": 0.0,
        "Newton_corrections": 0,
        "accepted_line_factors": [],
        "residual_history": [0.0],
        "maximum_augmented_condition_number": 0.0,
        "maximum_scaled_anchor_departure": 0.0,
        "wall_seconds": 0.0,
        "minimum_reconstruction_factor": 1.0,
        "maximum_height_ratio": 0.0,
        "minimum_scattering_optical_depth": float("inf"),
        "passed": True,
    }


class _ExactRetractedField:
    """Restartable exact field on a recentered, gauge-fixed implicit chart."""

    def __init__(
        self,
        *,
        index: int,
        anchor_state: np.ndarray,
        anchor_coordinate: np.ndarray,
        model,
        gauge_basis: np.ndarray,
        anchor_delta: np.ndarray,
        geometry: dict[str, np.ndarray],
        layout,
        configuration: dict,
        q3_target: np.ndarray,
        identity: dict,
        seed_metrics: dict,
        seed_arrays: dict[str, np.ndarray],
    ) -> None:
        self.index = index
        self.scratch = _scratch_directory(index)
        self.anchor_state = np.asarray(anchor_state, dtype=float)
        self.anchor_coordinate = np.asarray(anchor_coordinate, dtype=float)
        self.model = model
        self.anchor_model_state = np.asarray(model.decoded_state(anchor_coordinate), dtype=float)
        self.gauge_basis = np.asarray(gauge_basis, dtype=float)
        self.anchor_delta = np.asarray(anchor_delta, dtype=float)
        self.geometry = geometry
        self.layout = layout
        self.configuration = configuration
        self.q3_target = np.asarray(q3_target, dtype=float)
        self.face = 36 * int(layout.refinement_ratio)
        self.identity = identity
        self.records: dict[str, dict] = {}
        self.arrays: dict[str, dict[str, np.ndarray]] = {}
        self.new_call_count = 0
        self._insert(self.anchor_coordinate, seed_metrics, seed_arrays)
        self._prepare_scratch()

    @staticmethod
    def _key(coordinate: np.ndarray) -> str:
        value = np.ascontiguousarray(np.asarray(coordinate, dtype=float))
        return hashlib.sha256(value.tobytes()).hexdigest()

    def _insert(self, coordinate: np.ndarray, metrics: dict, arrays: dict[str, np.ndarray]) -> None:
        key = self._key(coordinate)
        self.records[key] = dict(metrics)
        self.arrays[key] = {name: np.asarray(value) for name, value in arrays.items()}

    def _prepare_scratch(self) -> None:
        helper = _helper()
        identity_path = self.scratch / "identity.json"
        index_path = self.scratch / "index.json"
        if self.scratch.exists():
            if not identity_path.exists() or helper._read(identity_path) != self.identity:
                raise RuntimeError("moving-chart scratch identity changed")
            listing = helper._read(index_path) if index_path.exists() else {"records": []}
            for entry in listing["records"]:
                metrics = helper._read(self.scratch / entry["metrics_file"])
                arrays = helper._load_npz(self.scratch / entry["arrays_file"])
                self._insert(arrays["coordinate470"], metrics, arrays)
            return
        self.scratch.mkdir(parents=True)
        helper._write_json(identity_path, self.identity)
        helper._write_json(index_path, {"records": []})

    def _raw_initial_state(self, coordinate: np.ndarray) -> np.ndarray:
        decoded = np.asarray(self.model.decoded_state(coordinate), dtype=float)
        return self.anchor_state + decoded - self.anchor_model_state

    def _retract(self, coordinate: np.ndarray) -> tuple[np.ndarray, dict]:
        value = np.asarray(coordinate, dtype=float)
        if np.array_equal(value, self.anchor_coordinate):
            return np.array(self.anchor_state, copy=True), _anchor_retraction_metrics()
        old_limit = exact_chart.MAXIMUM_SCALED_DEPARTURE
        old_corrections = exact_chart.MAXIMUM_NEWTON_CORRECTIONS
        try:
            exact_chart.MAXIMUM_SCALED_DEPARTURE = manifest.MAXIMUM_SCALED_ANCHOR_DEPARTURE
            exact_chart.MAXIMUM_NEWTON_CORRECTIONS = manifest.MAXIMUM_NEWTON_CORRECTIONS
            return exact_chart._newton_retract(
                self.model,
                self._raw_initial_state(value),
                value,
                self.gauge_basis,
                self.anchor_delta,
            )
        finally:
            exact_chart.MAXIMUM_SCALED_DEPARTURE = old_limit
            exact_chart.MAXIMUM_NEWTON_CORRECTIONS = old_corrections

    def decode(self, coordinate: np.ndarray) -> np.ndarray:
        value = np.asarray(coordinate, dtype=float)
        key = self._key(value)
        if key in self.arrays:
            return np.array(self.arrays[key]["decoded_primitive_state"], copy=True)
        state, _metrics = self._retract(value)
        return state

    def __call__(self, coordinate: np.ndarray, time_seconds: float) -> np.ndarray:
        helper = _helper()
        value = np.asarray(coordinate, dtype=float)
        key = self._key(value)
        if key in self.records:
            np.testing.assert_array_equal(self.arrays[key]["coordinate470"], value)
            return np.array(self.arrays[key]["coordinate_rate470_per_s"], copy=True)
        if len(self.records) >= manifest.MAXIMUM_UNIQUE_RATE_STATES_PER_WINDOW:
            raise RuntimeError("moving-chart exact-rate budget exhausted")
        state, retraction = self._retract(value)
        item, evidence = _post().exact_rate._evaluate_candidate(
            float(time_seconds), state, self.geometry, self.model, self.layout, self.configuration
        )
        recovered, factors = self.model.coordinate(state)
        q3, q3_factors = causal_five_field_exterior_q3(
            self.configuration["context"], state, exterior_face_index=self.face
        )
        metrics = {
            **item,
            "requested_time_seconds": float(time_seconds),
            "retraction": retraction,
            "Q3_relative_drift": float(
                np.linalg.norm(q3 - self.q3_target)
                / max(float(np.linalg.norm(self.q3_target)), np.finfo(float).tiny)
            ),
            "decoder_minimum_reconstruction_factor": float(np.min(factors)),
            "Q3_minimum_reconstruction_factor": float(np.min(q3_factors)),
        }
        arrays = {
            "coordinate470": value,
            "decoded_primitive_state": state,
            "recovered_coordinate470": np.asarray(recovered, dtype=float),
            "Q3": np.asarray(q3, dtype=float),
            "coordinate_rate470_per_s": np.asarray(evidence["coordinate_rate470_per_s"], dtype=float),
            "scaled_fixed_Q_rate560_per_s": np.asarray(evidence["scaled_fixed_Q_rate560_per_s"], dtype=float),
            "scaled_reaction_action560_per_s": np.asarray(evidence["scaled_reaction_action560_per_s"], dtype=float),
        }
        stem = f"exact_rate_{len(self.records):02d}_{key[:12]}"
        metrics_file, arrays_file = stem + ".json", stem + ".npz"
        helper._write_json(self.scratch / metrics_file, metrics)
        with (self.scratch / arrays_file).open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        listing = helper._read(self.scratch / "index.json")
        listing["records"].append({"key": key, "metrics_file": metrics_file, "arrays_file": arrays_file})
        helper._write_json(self.scratch / "index.json", listing)
        self._insert(value, metrics, arrays)
        self.new_call_count += 1
        print(
            f"moving window {self.index:02d} exact rate {len(self.records):02d}/"
            f"{manifest.MAXIMUM_UNIQUE_RATE_STATES_PER_WINDOW}: "
            f"t={time_seconds:.9e}s Q3={metrics['Q3_relative_drift']:.3e} "
            f"coord={retraction['coordinate_residual_infinity']:.3e} "
            f"departure={retraction['maximum_scaled_anchor_departure']:.3e}",
            flush=True,
        )
        return np.array(arrays["coordinate_rate470_per_s"], copy=True)


def _base_inputs() -> dict:
    return _adaptive()._base_inputs()


def _seed_and_training(index: int, base: dict) -> dict:
    helper = _helper()
    post = _post()
    post_arrays = helper._load_npz(post.CANONICAL_DIRECTORY / "post_transition_phase_window_model_and_witnesses.npz")
    training = [
        np.asarray(post_arrays["half_1__final_rates"], dtype=float),
        np.asarray(post_arrays["half_2__final_rates"], dtype=float),
    ]
    for old_index in (1, 2):
        old_arrays = helper._load_npz(_adaptive()._stage_directory(old_index) / "phase_window_arrays.npz")
        training.append(np.asarray(old_arrays["final_rates470_per_s"], dtype=float))
    if index == manifest.FIRST_WINDOW_INDEX:
        previous_directory = manifest.ACCEPTED_DIRECTORY
    else:
        for prior_index in range(manifest.FIRST_WINDOW_INDEX, index):
            prior_arrays = helper._load_npz(_stage_directory(prior_index) / "phase_window_arrays.npz")
            training.append(np.asarray(prior_arrays["final_rates470_per_s"], dtype=float))
        previous_directory = _stage_directory(index - 1)
    previous_metrics = helper._read(previous_directory / "phase_window_metrics.json")
    previous_arrays = helper._load_npz(previous_directory / "phase_window_arrays.npz")
    state = np.asarray(previous_arrays["endpoint_primitive_state"], dtype=float)
    coordinate, factors = base["model"].coordinate(state)
    intended = np.asarray(previous_arrays["endpoint_coordinate470"], dtype=float)
    exact_coordinates = np.asarray(previous_arrays["exact_evaluation_coordinates470"])
    matches = np.flatnonzero(np.all(exact_coordinates == intended, axis=1))
    if len(matches) != 1:
        raise RuntimeError("accepted endpoint exact witness changed")
    witness = int(matches[0])
    seed_metrics = dict(previous_metrics["exact_rate_metrics"][witness])
    seed_metrics.update({
        "requested_time_seconds": float(previous_metrics["end_time_seconds"]),
        "retraction": _anchor_retraction_metrics(),
        "Q3_relative_drift": float(previous_metrics["gate_values"]["maximum_Q3_relative_drift"]),
        "decoder_minimum_reconstruction_factor": float(np.min(factors)),
        "Q3_minimum_reconstruction_factor": 1.0,
    })
    seed_arrays = {
        "coordinate470": np.asarray(coordinate, dtype=float),
        "decoded_primitive_state": state,
        "recovered_coordinate470": np.asarray(coordinate, dtype=float),
        "Q3": np.asarray(previous_arrays["exact_evaluation_Q3"][witness]),
        "coordinate_rate470_per_s": np.asarray(previous_arrays["exact_evaluation_rates470_per_s"][witness]),
    }
    return {
        "coordinate": np.asarray(coordinate, dtype=float),
        "state": state,
        "time_seconds": float(previous_metrics["end_time_seconds"]),
        "training_rates": np.vstack(training),
        "event_persistence": int(previous_metrics["persistent_event_window_run"]),
        "seed_metrics": seed_metrics,
        "seed_arrays": seed_arrays,
        "previous_metrics": None if index == manifest.FIRST_WINDOW_INDEX else previous_metrics,
    }


def _evaluate(index: int, locked: dict) -> tuple[dict, dict[str, np.ndarray]]:
    base = _base_inputs()
    seed = _seed_and_training(index, base)
    _anchor_coordinate, anchor_factors = base["model"].coordinate(seed["state"])
    np.testing.assert_array_equal(_anchor_coordinate, seed["coordinate"])
    anchor_q3, anchor_q3_factors = causal_five_field_exterior_q3(
        base["configuration"]["context"],
        seed["state"],
        exterior_face_index=36 * int(base["layout"].refinement_ratio),
    )
    seed["seed_metrics"]["retraction"].update(
        exact_chart._physical_audit(base["model"], seed["state"], anchor_factors)
    )
    seed["seed_metrics"]["Q3_relative_drift"] = float(
        np.linalg.norm(anchor_q3 - base["q3_target"])
        / max(float(np.linalg.norm(base["q3_target"])), np.finfo(float).tiny)
    )
    seed["seed_metrics"]["Q3_minimum_reconstruction_factor"] = float(
        np.min(anchor_q3_factors)
    )
    seed["seed_arrays"]["Q3"] = np.asarray(anchor_q3, dtype=float)
    basis, singular_values, rank, training_defect = _canonical_basis(seed["training_rates"])
    duration = _next_duration(index, seed["previous_metrics"])
    coordinate_jacobian, coordinate_metrics = exact_chart._coordinate_jacobian(base["model"], seed["state"])
    gauge_basis = exact_chart._canonical_null_basis(coordinate_jacobian)
    augmented, augmented_metrics = exact_chart._augmented_jacobian(base["model"], seed["state"], gauge_basis)
    anchor_delta = exact_chart._delta(base["model"], seed["state"])
    if augmented_metrics["augmented_rank"] != exact_chart.PHYSICAL_DIMENSION:
        raise RuntimeError("moving-chart anchor augmented Jacobian lost rank")
    identity = {
        "work_package": WORK_PACKAGE,
        "window_index": index,
        "manifest_hashes": locked["manifest_hashes"],
        "anchor_state_sha256": hashlib.sha256(np.ascontiguousarray(seed["state"]).tobytes()).hexdigest(),
        "anchor_coordinate_sha256": hashlib.sha256(np.ascontiguousarray(seed["coordinate"]).tobytes()).hexdigest(),
        "duration_seconds": duration,
        "basis_rank": rank,
    }
    field = _ExactRetractedField(
        index=index,
        anchor_state=seed["state"],
        anchor_coordinate=seed["coordinate"],
        model=base["model"],
        gauge_basis=gauge_basis,
        anchor_delta=anchor_delta,
        geometry=base["geometry"],
        layout=base["layout"],
        configuration=base["configuration"],
        q3_target=base["q3_target"],
        identity=identity,
        seed_metrics=seed["seed_metrics"],
        seed_arrays=seed["seed_arrays"],
    )
    began = time.perf_counter()
    window = _post()._picard_window(
        start_coordinate=seed["coordinate"],
        start_time_seconds=seed["time_seconds"],
        duration_seconds=duration,
        basis=basis,
        evaluator=field,
        node_count=manifest.NODE_COUNT,
    )
    wall_seconds = float(time.perf_counter() - began)
    endpoint_coordinate = np.asarray(window["endpoint"], dtype=float)
    endpoint_state = field.decode(endpoint_coordinate)
    event = _adaptive()._event_features(seed["state"], endpoint_state, duration)
    records = list(field.records.values())
    required_physical = (
        "coordinate_decomposition", "coordinate_rank", "coordinate_condition",
        "fixed_Q_tangency", "reaction_ledger", "Schur_rank", "Schur_condition",
        "reconstruction", "height", "optical_depth",
    )
    physical_pass = all(all(item["gates"][name] for name in required_physical) for item in records)
    retractions = [item["retraction"] for item in records]
    maximum_coordinate = max(float(item["coordinate_residual_infinity"]) for item in retractions)
    maximum_gauge = max(float(item["gauge_residual_infinity"]) for item in retractions)
    maximum_condition = max(float(item["maximum_augmented_condition_number"]) for item in retractions)
    maximum_departure = max(float(item["maximum_scaled_anchor_departure"]) for item in retractions)
    maximum_q3 = max(float(item["Q3_relative_drift"]) for item in records)
    minimum_reconstruction = min(
        min(
            float(item["minimum_reconstruction_factor"]),
            float(item["decoder_minimum_reconstruction_factor"]),
            float(item["Q3_minimum_reconstruction_factor"]),
            float(item["retraction"]["minimum_reconstruction_factor"]),
        )
        for item in records
    )
    maximum_projected = float(np.max(window["projected_defects"]))
    maximum_full = float(np.max(window["full_defects"]))
    maximum_normal = float(np.max(window["normal_defects"]))
    minimum_cosine = float(np.min(window["direction_cosines"]))
    gate_values = {
        "maximum_training_normal_rate_defect": training_defect,
        "maximum_projected_collocation_defect": maximum_projected,
        "maximum_full_collocation_defect": maximum_full,
        "maximum_normal_rate_defect": maximum_normal,
        "minimum_rate_direction_cosine": minimum_cosine,
        "maximum_coordinate_residual_infinity": maximum_coordinate,
        "maximum_gauge_residual_infinity": maximum_gauge,
        "maximum_augmented_condition_number": maximum_condition,
        "maximum_scaled_anchor_departure": maximum_departure,
        "maximum_Q3_relative_drift": maximum_q3,
        "minimum_reconstruction_factor": minimum_reconstruction,
        "unique_exact_rate_states": len(field.records),
        "new_exact_rate_calls_this_process": field.new_call_count,
        "execution_wall_seconds_this_process": wall_seconds,
    }
    gates = {
        "training_subspace": training_defect <= manifest.MAXIMUM_TRAINING_NORMAL_RATE_DEFECT,
        "projected_collocation": maximum_projected <= manifest.MAXIMUM_PROJECTED_COLLOCATION_DEFECT,
        "full_collocation": maximum_full <= manifest.MAXIMUM_FULL_COLLOCATION_DEFECT,
        "normal_rate": maximum_normal <= manifest.MAXIMUM_NORMAL_RATE_DEFECT,
        "rate_direction": minimum_cosine >= manifest.MINIMUM_RATE_DIRECTION_COSINE,
        "coordinate_retraction": maximum_coordinate <= manifest.COORDINATE_TOLERANCE,
        "gauge_retraction": maximum_gauge <= manifest.GAUGE_TOLERANCE,
        "retraction_conditioning": maximum_condition <= manifest.MAXIMUM_AUGMENTED_CONDITION_NUMBER,
        "retraction_neighborhood": maximum_departure <= manifest.MAXIMUM_SCALED_ANCHOR_DEPARTURE,
        "retraction_physics": all(bool(item["passed"]) for item in retractions),
        "fixed_Q_state_drift": maximum_q3 <= manifest.MAXIMUM_Q3_RELATIVE_DRIFT,
        "reconstruction": minimum_reconstruction >= manifest.MINIMUM_RECONSTRUCTION_FACTOR,
        "exact_rate_physics": physical_pass,
        "truth_budget": len(field.records) <= manifest.MAXIMUM_UNIQUE_RATE_STATES_PER_WINDOW,
        "no_roots_or_microsteps": True,
    }
    passed = bool(all(gates.values()))
    growth_margin = bool(
        passed
        and maximum_full <= manifest.GROW_MAXIMUM_FULL_COLLOCATION_DEFECT
        and maximum_normal <= manifest.GROW_MAXIMUM_NORMAL_RATE_DEFECT
        and minimum_cosine >= manifest.GROW_MINIMUM_RATE_DIRECTION_COSINE
        and maximum_q3 <= manifest.GROW_MAXIMUM_Q3_RELATIVE_DRIFT
        and maximum_departure <= 0.5 * manifest.MAXIMUM_SCALED_ANCHOR_DEPARTURE
    )
    persistent_run = seed["event_persistence"] + 1 if passed and event["event_gate_passed"] else 0
    hot_exit = bool(passed and persistent_run >= manifest.HIDDEN_EXIT_PERSISTENCE_WINDOWS)
    budget_exhausted = bool(passed and index >= manifest.MAXIMUM_WINDOW_INDEX and not hot_exit)
    if not passed:
        classification = FAIL_CLASSIFICATION
    elif hot_exit:
        classification = EXIT_CLASSIFICATION
    elif budget_exhausted:
        classification = BUDGET_CLASSIFICATION
    else:
        classification = CONTINUE_CLASSIFICATION
    metrics = {
        "classification": classification,
        "passed": passed,
        "window_index": index,
        "basis_rank": rank,
        "node_count": manifest.NODE_COUNT,
        "start_time_seconds": float(seed["time_seconds"]),
        "end_time_seconds": float(seed["time_seconds"] + duration),
        "duration_seconds": duration,
        "anchor_coordinate_metrics": coordinate_metrics,
        "anchor_augmented_metrics": augmented_metrics,
        "gates": gates,
        "gate_values": gate_values,
        "growth_margin_passed": growth_margin,
        "event_gate_passed": bool(event["event_gate_passed"]),
        "event_metrics": event["metrics"],
        "persistent_event_window_run": persistent_run,
        "hot_exit_observed": hot_exit,
        "budget_exhausted": budget_exhausted,
        "new_nonlinear_fixed_Q_roots": 0,
        "new_BDF_microsteps": 0,
        "exact_rate_metrics": records,
    }
    keys = list(field.records)
    arrays = {
        "basis470xr": basis,
        "training_rates470_per_s": np.asarray(seed["training_rates"]),
        "training_singular_values": singular_values,
        "anchor_coordinate470": np.asarray(seed["coordinate"]),
        "anchor_delta560": anchor_delta,
        "anchor_gauge_basis560x90": gauge_basis,
        "start_coordinate470": np.asarray(seed["coordinate"]),
        "start_primitive_state": np.asarray(seed["state"]),
        "endpoint_coordinate470": endpoint_coordinate,
        "endpoint_primitive_state": endpoint_state,
        "Q3_target": base["q3_target"],
        "nodes": np.asarray(window["nodes"]),
        "coordinates470": np.asarray(window["coordinates"]),
        "predictor_coordinates470": np.asarray(window["predictor_coordinates"]),
        "predictor_rates470_per_s": np.asarray(window["predictor_rates"]),
        "final_rates470_per_s": np.asarray(window["final_rates"]),
        "collocation_derivatives470_per_s": np.asarray(window["collocation_derivatives"]),
        "projected_defects": np.asarray(window["projected_defects"]),
        "full_defects": np.asarray(window["full_defects"]),
        "normal_defects": np.asarray(window["normal_defects"]),
        "direction_cosines": np.asarray(window["direction_cosines"]),
        "exact_evaluation_coordinates470": np.stack([field.arrays[key]["coordinate470"] for key in keys]),
        "exact_evaluation_recovered_coordinates470": np.stack([field.arrays[key]["recovered_coordinate470"] for key in keys]),
        "exact_evaluation_primitive_states": np.stack([field.arrays[key]["decoded_primitive_state"] for key in keys]),
        "exact_evaluation_rates470_per_s": np.stack([field.arrays[key]["coordinate_rate470_per_s"] for key in keys]),
        "exact_evaluation_Q3": np.stack([field.arrays[key]["Q3"] for key in keys]),
        "exact_retraction_coordinate_residuals": np.asarray([item["coordinate_residual_infinity"] for item in retractions]),
        "exact_retraction_gauge_residuals": np.asarray([item["gauge_residual_infinity"] for item in retractions]),
        "exact_retraction_condition_numbers": np.asarray([item["maximum_augmented_condition_number"] for item in retractions]),
        "exact_retraction_anchor_departures": np.asarray([item["maximum_scaled_anchor_departure"] for item in retractions]),
        **{f"event__{name}": np.asarray(value) for name, value in event["arrays"].items()},
    }
    return metrics, arrays


def _update_catalog(index: int, summary: dict) -> None:
    helper = _helper()
    cold = _post().manifest.transition.manifest.cold.manifest
    artifact = _stage_directory(index).name
    with cold.CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != artifact]
    status = "SUPPORTED" if summary["passed"] else "REJECTED"
    for path in sorted(_stage_directory(index).iterdir()):
        if path.is_file():
            rows.append({"case": artifact, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": helper._sha(path), "scientific_status": status})
    with cold.CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    catalog = helper._read(cold.CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[artifact] = {"path": str(_stage_directory(index).relative_to(ROOT)), "classification": summary["classification"], "passed": summary["passed"]}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": helper._git("rev-parse", "HEAD"), "latest_work_package": WORK_PACKAGE})
    helper._write_json(cold.CANONICAL_SUMMARY, catalog)


def _run(index: int) -> dict:
    helper = _helper()
    if index < manifest.FIRST_WINDOW_INDEX or index > manifest.MAXIMUM_WINDOW_INDEX:
        raise ValueError("moving-chart window index outside frozen budget")
    destination, report = _stage_directory(index), _report_path(index)
    if destination.exists() or report.exists():
        raise RuntimeError("moving-chart result already exists")
    locked = _validate_manifest(require_clean=True)
    _validate_order(index)
    metrics, arrays = _evaluate(index, locked)
    destination.mkdir(parents=True)
    helper._write_json(destination / "phase_window_metrics.json", metrics)
    with (destination / "phase_window_arrays.npz").open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    checkpoint = {
        "endpoint_coordinate470": arrays["endpoint_coordinate470"],
        "endpoint_primitive_state": arrays["endpoint_primitive_state"],
        "Q3_target": arrays["Q3_target"],
        "end_time_seconds": np.asarray(metrics["end_time_seconds"]),
        "duration_seconds": np.asarray(metrics["duration_seconds"]),
        "persistent_event_window_run": np.asarray(metrics["persistent_event_window_run"], dtype=np.int64),
    }
    with (destination / "phase_window_checkpoint.npz").open("wb") as handle:
        np.savez_compressed(handle, **checkpoint)
    reloaded = helper._load_npz(destination / "phase_window_checkpoint.npz")
    for name, value in checkpoint.items():
        np.testing.assert_array_equal(reloaded[name], value)
    helper._write_json(destination / "input_lock.json", locked)
    next_window = bool(metrics["passed"] and not metrics["hot_exit_observed"] and not metrics["budget_exhausted"])
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": metrics["passed"],
        "window_passed": metrics["passed"],
        "window_index": index,
        "moving_exact_chart": True,
        "checkpoint_roundtrip_bitwise": True,
        "hot_exit_observed": metrics["hot_exit_observed"],
        "persistent_event_window_run": metrics["persistent_event_window_run"],
        "next_window_authorized": next_window,
        "endpoint_refinement_manifest_authorized": metrics["hot_exit_observed"],
        "terminal_prognosis_authorized": metrics["budget_exhausted"],
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": WORK_PACKAGE if next_window else FINAL_WORK_PACKAGE,
    }
    helper._write_json(destination / "summary.json", summary)
    helper._write_json(destination / "provenance.json", {"runner": THIS_RUNNER, "test": THIS_TEST, "implementation_commit": helper._git("rev-parse", "HEAD"), "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"), "python": sys.version, "numpy": np.__version__, "platform": platform.platform()})
    names = sorted(path.name for path in destination.iterdir())
    (destination / "SHA256SUMS.txt").write_text("".join(f"{helper._sha(destination / name)}  {name}\n" for name in names), encoding="utf-8")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join((f"# Moving exact-chart hot-exit phase-atlas window {index:02d}", "", f"Classification: `{metrics['classification']}`.", "", f"Duration: `{metrics['duration_seconds']:.6e}` s; rank: `{metrics['basis_rank']}`; full collocation defect: `{metrics['gate_values']['maximum_full_collocation_defect']:.6e}`; maximum exact coordinate residual: `{metrics['gate_values']['maximum_coordinate_residual_infinity']:.6e}`; maximum local departure: `{metrics['gate_values']['maximum_scaled_anchor_departure']:.6e}`.", "", f"Accepted: `{metrics['passed']}`. Hot-exit persistence: `{metrics['persistent_event_window_run']}` of `{manifest.HIDDEN_EXIT_PERSISTENCE_WINDOWS}`. No nonlinear fixed-Q root or BDF microstep was executed.", "")), encoding="utf-8")
    _update_catalog(index, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--window", type=int)
    args = parser.parse_args()
    if args.window is None:
        parser.error("use --window INDEX")
    print(json.dumps(_run(args.window), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
