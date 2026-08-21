#!/usr/bin/env python3
"""Execute one committed adaptive phase-atlas window toward hot exit."""

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
import run_causal_inner_adaptive_hot_exit_phase_atlas_manifest_v2_wp10c9d6c7c3b5c4f25ed as manifest  # noqa: E402
import run_causal_inner_bounded_hot_exit_acquisition_wp10c9d6c7c3b5c4f25do as legacy_hot  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25ee"
CONTINUE_CLASSIFICATION = "adaptive_hot_exit_phase_window_passed_event_not_yet_observed"
EXIT_CLASSIFICATION = "persistent_hot_exit_candidate_observed_endpoint_refinement_authorized"
BUDGET_CLASSIFICATION = "adaptive_hot_exit_phase_atlas_budget_exhausted_without_event"
FAIL_CLASSIFICATION = "adaptive_hot_exit_phase_window_rejected_last_accepted_endpoint_preserved"
FINAL_WORK_PACKAGE = "WP10c9d6c7c3b5c4f25ef"
ARTIFACT_PREFIX = "causal_inner_adaptive_hot_exit_phase_atlas_wp10c9d6c7c3b5c4f25ee"
SCRATCH_ROOT = ROOT / "outputs/checkpoints" / f"{ARTIFACT_PREFIX}_v2"
THIS_RUNNER = "scripts/run_causal_inner_adaptive_hot_exit_phase_atlas_wp10c9d6c7c3b5c4f25ee.py"
THIS_TEST = "tests/test_causal_inner_adaptive_hot_exit_phase_atlas_wp10c9d6c7c3b5c4f25ee.py"


def _helper():
    return manifest._helper()


def _post():
    return manifest.architecture.rejected.post


def _stage_directory(index: int) -> Path:
    return ROOT / "results/canonical" / f"{ARTIFACT_PREFIX}_window_{index:02d}"


def _scratch_directory(index: int) -> Path:
    return SCRATCH_ROOT / f"window_{index:02d}"


def _report_path(index: int) -> Path:
    return ROOT / "docs/reports/current" / (
        "CODEX_CAUSAL_INNER_ADAPTIVE_HOT_EXIT_PHASE_ATLAS_WINDOW_"
        f"{index:02d}_WP10C9D6C7C3B5C4F25EE_2026-08-21.md"
    )


def _validate_manifest(*, require_clean: bool) -> dict:
    helper = _helper()
    hashes = helper._validate_checksums(manifest.CANONICAL_DIRECTORY)
    summary = helper._read(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = helper._read(
        manifest.CANONICAL_DIRECTORY / "adaptive_hot_exit_phase_atlas_contract.json"
    )
    if (
        not summary["passed"]
        or not (
            summary.get("definitions_only", False)
            or summary.get("definitions_only_with_recovered_truth_cache", False)
        )
        or not summary["adaptive_phase_atlas_execution_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["work_package"] != manifest.WORK_PACKAGE
    ):
        raise RuntimeError("adaptive phase-atlas manifest changed")
    for relative, expected in contract["frozen_source_hashes"].items():
        if helper._sha(ROOT / relative) != expected:
            raise RuntimeError(f"frozen adaptive phase-atlas source changed: {relative}")
    current_inputs = {
        name: helper._sha(path) for name, path in manifest._decisive_inputs().items()
    }
    if current_inputs != contract["decisive_input_hashes"]:
        raise RuntimeError("adaptive phase-atlas decisive input changed")
    manifest._validate_parent(require_clean=False)
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("adaptive phase-atlas window requires a clean tracked tree")
    return {"manifest_hashes": hashes, "contract": contract}


def _validate_order(index: int) -> dict | None:
    helper = _helper()
    if index == 1:
        return None
    previous = _stage_directory(index - 1)
    helper._validate_checksums(previous)
    summary = helper._read(previous / "summary.json")
    if (
        not summary["passed"]
        or not summary["window_passed"]
        or not summary["next_window_authorized"]
        or summary["hot_exit_observed"]
        or int(summary["window_index"]) != index - 1
    ):
        raise RuntimeError("prior adaptive phase window does not authorize this window")
    return summary


def _canonical_basis(rates: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, float]:
    values = np.asarray(rates, dtype=float)
    normalized = values / np.linalg.norm(values, axis=1)[:, None]
    _left, singular_values, right = np.linalg.svd(normalized, full_matrices=False)
    selected_rank = min(manifest.RATE_BASIS_RANKS[-1], right.shape[0])
    selected_defect = float("inf")
    selected_basis = None
    for rank in manifest.RATE_BASIS_RANKS:
        if rank > right.shape[0]:
            continue
        basis = np.asarray(right[:rank].T, dtype=float)
        defects = np.linalg.norm(normalized - (normalized @ basis) @ basis.T, axis=1)
        defect = float(np.max(defects))
        selected_rank = rank
        selected_defect = defect
        selected_basis = basis
        if defect <= manifest.MAXIMUM_TRAINING_NORMAL_RATE_DEFECT:
            break
    if selected_basis is None:
        raise RuntimeError("no adaptive phase basis could be constructed")
    for column in range(selected_basis.shape[1]):
        pivot = int(np.argmax(np.abs(selected_basis[:, column])))
        if selected_basis[pivot, column] < 0.0:
            selected_basis[:, column] *= -1.0
    return selected_basis, singular_values, selected_rank, selected_defect


def _next_duration(previous_metrics: dict | None) -> float:
    if previous_metrics is None:
        return manifest.INITIAL_DURATION_SECONDS
    duration = float(previous_metrics["duration_seconds"])
    if previous_metrics["event_gate_passed"]:
        return duration
    if previous_metrics["growth_margin_passed"]:
        return min(2.0 * duration, manifest.MAXIMUM_DURATION_SECONDS)
    return duration


class _ExactField:
    """Stage-local exact field cache using the certified anchored decoder."""

    def __init__(
        self,
        *,
        index: int,
        anchor_state: np.ndarray,
        anchor_coordinate: np.ndarray,
        transition_path: float,
        model,
        geometry: dict[str, np.ndarray],
        layout,
        configuration: dict,
        q3_target: np.ndarray,
        identity: dict,
        seed_metrics: dict,
        seed_arrays: dict[str, np.ndarray],
        recovered_records: list[tuple[dict, dict[str, np.ndarray]]],
    ) -> None:
        self.index = index
        self.scratch = _scratch_directory(index)
        self.anchor_state = np.asarray(anchor_state, dtype=float)
        self.anchor_coordinate = np.asarray(anchor_coordinate, dtype=float)
        self.transition_path = float(transition_path)
        self.model = model
        self.anchor_model_state = np.asarray(model.decoded_state(anchor_coordinate), dtype=float)
        self.geometry = geometry
        self.layout = layout
        self.configuration = configuration
        self.q3_target = np.asarray(q3_target, dtype=float)
        self.face = 36 * int(layout.refinement_ratio)
        self.identity = identity
        self.records: dict[str, dict] = {}
        self.arrays: dict[str, dict[str, np.ndarray]] = {}
        self.new_call_count = 0
        self._insert(seed_arrays["coordinate470"], seed_metrics, seed_arrays)
        for metrics, arrays in recovered_records:
            self._insert(arrays["coordinate470"], metrics, arrays)
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
                raise RuntimeError("adaptive phase-window scratch identity changed")
            listing = helper._read(index_path) if index_path.exists() else {"records": []}
            for entry in listing["records"]:
                metrics = helper._read(self.scratch / entry["metrics_file"])
                arrays = helper._load_npz(self.scratch / entry["arrays_file"])
                self._insert(arrays["coordinate470"], metrics, arrays)
            return
        self.scratch.mkdir(parents=True)
        helper._write_json(identity_path, self.identity)
        helper._write_json(index_path, {"records": []})

    def decode(self, coordinate: np.ndarray) -> np.ndarray:
        value = np.asarray(coordinate, dtype=float)
        if np.array_equal(value, self.anchor_coordinate):
            return np.array(self.anchor_state, copy=True)
        decoded = np.asarray(self.model.decoded_state(value), dtype=float)
        return self.anchor_state + decoded - self.anchor_model_state

    def __call__(self, coordinate: np.ndarray, time_seconds: float) -> np.ndarray:
        helper = _helper()
        exact_rate = _post().exact_rate
        value = np.asarray(coordinate, dtype=float)
        key = self._key(value)
        if key in self.records:
            np.testing.assert_array_equal(self.arrays[key]["coordinate470"], value)
            return np.array(self.arrays[key]["coordinate_rate470_per_s"], copy=True)
        if len(self.records) >= manifest.MAXIMUM_UNIQUE_RATE_STATES_PER_WINDOW:
            raise RuntimeError("adaptive phase-window exact-rate budget exhausted")
        state = self.decode(value)
        item, evidence = exact_rate._evaluate_candidate(
            float(time_seconds),
            state,
            self.geometry,
            self.model,
            self.layout,
            self.configuration,
        )
        recovered, factors = self.model.coordinate(state)
        q3, q3_factors = causal_five_field_exterior_q3(
            self.configuration["context"], state, exterior_face_index=self.face
        )
        metrics = {
            **item,
            "requested_time_seconds": float(time_seconds),
            "decoder_coordinate_error_over_transition_path": float(
                np.linalg.norm(recovered - value) / self.transition_path
            ),
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
        metrics_file = stem + ".json"
        arrays_file = stem + ".npz"
        helper._write_json(self.scratch / metrics_file, metrics)
        with (self.scratch / arrays_file).open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        listing = helper._read(self.scratch / "index.json")
        listing["records"].append({"key": key, "metrics_file": metrics_file, "arrays_file": arrays_file})
        helper._write_json(self.scratch / "index.json", listing)
        self._insert(value, metrics, arrays)
        self.new_call_count += 1
        print(
            f"window {self.index:02d} exact rate {len(self.records):02d}/"
            f"{manifest.MAXIMUM_UNIQUE_RATE_STATES_PER_WINDOW}: "
            f"t={time_seconds:.9e}s Q3={metrics['Q3_relative_drift']:.3e} "
            f"decoder={metrics['decoder_coordinate_error_over_transition_path']:.3e}",
            flush=True,
        )
        return np.array(arrays["coordinate_rate470_per_s"], copy=True)


def _base_inputs() -> dict:
    helper = _helper()
    post = _post()
    transition_arrays = helper._load_npz(
        post.manifest.transition.manifest.manifest_geometry_path()
    )
    coordinates = np.asarray(transition_arrays["trajectory_coordinates470"], dtype=float)
    times = np.asarray(transition_arrays["trajectory_times_seconds"], dtype=float)
    anchor_coordinate = coordinates[-1]
    anchor_state = np.asarray(post.manifest.transition._states()[-1], dtype=float)
    model, _candidate, _fiber = post.exact_rate.exact_chart._model_and_inputs()
    layout, configuration, _trajectory, *_unused = post.exact_rate.rate_source.c4f24._endpoint_data()
    geometry = post.exact_rate._geometry()
    terminal_checkpoint = helper._load_npz(post.manifest._decisive_inputs()["terminal_checkpoint"])
    return {
        "transition_arrays": transition_arrays,
        "anchor_coordinate": anchor_coordinate,
        "anchor_state": anchor_state,
        "transition_path": float(np.sum(np.linalg.norm(np.diff(coordinates, axis=0), axis=1))),
        "model": model,
        "layout": layout,
        "configuration": configuration,
        "geometry": geometry,
        "q3_target": np.asarray(terminal_checkpoint["q3_target"], dtype=float),
        "terminal_time": float(times[-1]),
    }


def _seed_and_training(index: int, base: dict) -> dict:
    helper = _helper()
    post = _post()
    post_arrays = helper._load_npz(
        post.CANONICAL_DIRECTORY / "post_transition_phase_window_model_and_witnesses.npz"
    )
    post_metrics = helper._read(
        post.CANONICAL_DIRECTORY / "post_transition_phase_window_metrics.json"
    )
    training = [
        np.asarray(post_arrays["half_1__final_rates"], dtype=float),
        np.asarray(post_arrays["half_2__final_rates"], dtype=float),
    ]
    if index == 1:
        coordinate = np.asarray(post_arrays["two_half_endpoint_coordinate470"], dtype=float)
        state = np.asarray(post_arrays["two_half_endpoint_primitive_state"], dtype=float)
        time_seconds = float(base["terminal_time"] + post.manifest.FULL_DURATION_SECONDS)
        exact_coordinates = np.asarray(post_arrays["exact_evaluation_coordinates470"])
        matches = np.flatnonzero(np.all(exact_coordinates == coordinate, axis=1))
        if len(matches) != 1:
            raise RuntimeError("Stage-4 accepted endpoint exact witness changed")
        witness = int(matches[0])
        metrics = dict(post_metrics["exact_rate_metrics"][witness])
        arrays = {
            "coordinate470": coordinate,
            "decoded_primitive_state": state,
            "Q3": np.asarray(post_arrays["exact_evaluation_Q3"][witness]),
            "coordinate_rate470_per_s": np.asarray(post_arrays["exact_evaluation_rates470_per_s"][witness]),
        }
        return {
            "coordinate": coordinate,
            "state": state,
            "time_seconds": time_seconds,
            "training_rates": np.vstack(training),
            "event_persistence": 0,
            "seed_metrics": metrics,
            "seed_arrays": arrays,
            "previous_metrics": None,
        }
    for prior_index in range(1, index):
        prior_arrays = helper._load_npz(_stage_directory(prior_index) / "phase_window_arrays.npz")
        training.append(np.asarray(prior_arrays["final_rates470_per_s"], dtype=float))
    previous = _stage_directory(index - 1)
    previous_metrics = helper._read(previous / "phase_window_metrics.json")
    previous_arrays = helper._load_npz(previous / "phase_window_arrays.npz")
    coordinate = np.asarray(previous_arrays["endpoint_coordinate470"], dtype=float)
    exact_coordinates = np.asarray(previous_arrays["exact_evaluation_coordinates470"])
    matches = np.flatnonzero(np.all(exact_coordinates == coordinate, axis=1))
    if len(matches) != 1:
        raise RuntimeError("prior adaptive endpoint exact witness changed")
    witness = int(matches[0])
    metrics = dict(previous_metrics["exact_rate_metrics"][witness])
    arrays = {
        "coordinate470": coordinate,
        "decoded_primitive_state": np.asarray(previous_arrays["endpoint_primitive_state"]),
        "Q3": np.asarray(previous_arrays["exact_evaluation_Q3"][witness]),
        "coordinate_rate470_per_s": np.asarray(previous_arrays["exact_evaluation_rates470_per_s"][witness]),
    }
    return {
        "coordinate": coordinate,
        "state": np.asarray(previous_arrays["endpoint_primitive_state"], dtype=float),
        "time_seconds": float(previous_metrics["end_time_seconds"]),
        "training_rates": np.vstack(training),
        "event_persistence": int(previous_metrics["persistent_event_window_run"]),
        "seed_metrics": metrics,
        "seed_arrays": arrays,
        "previous_metrics": previous_metrics,
    }


def _static_feature_data() -> dict:
    helper = _helper()
    source = manifest.legacy_exit
    tangent_arrays = helper._load_npz(source.TANGENT_ARRAYS)
    geometry_arrays = helper._load_npz(source.GEOMETRY_ARRAYS)
    screen_arrays = helper._load_npz(source.PARENT_ARRAYS)
    field_arrays = helper._load_npz(legacy_hot.screen.geometry.FIELD_ARRAYS)
    field = legacy_hot.screen.geometry.field_manifest.ForwardQuadraticAuthenticCenterField(
        field_arrays
    )
    labels = helper._read(source.PARENT_METRICS)["candidate_labels"]
    if labels[-1] != "fixed_Q_warm_3":
        raise RuntimeError("saved warm_3 coordinate label changed")
    return {
        "model": field.model,
        "macro_restriction": tangent_arrays["macro_restriction_R82"],
        "hidden_basis": tangent_arrays["hidden_basis_Z388"],
        "hidden_dual": tangent_arrays["hidden_dual_Q388"],
        "rank16_basis": tangent_arrays["selected_hidden_basis388"],
        "anchor_coordinate": geometry_arrays["candidate_absolute_y470_coordinates"][5],
        "seed_coordinate": screen_arrays["candidate_absolute_y470_coordinates"][-1],
    }


def _event_features(
    start_state: np.ndarray,
    endpoint_state: np.ndarray,
    duration_seconds: float,
) -> dict:
    static = _static_feature_data()
    previous_coordinate = np.asarray(static["model"].coordinate(start_state)[0], dtype=float)
    current_coordinate = np.asarray(static["model"].coordinate(endpoint_state)[0], dtype=float)
    secant_rate = (current_coordinate - previous_coordinate) / duration_seconds
    hidden_rate = static["hidden_dual"] @ secant_rate
    hidden_action = static["hidden_basis"] @ hidden_rate
    hidden_fraction = float(
        np.linalg.norm(hidden_action)
        / max(float(np.linalg.norm(secant_rate)), np.finfo(float).tiny)
    )
    rank16_capture = float(
        np.linalg.norm(static["rank16_basis"].T @ hidden_rate)
        / max(float(np.linalg.norm(hidden_rate)), np.finfo(float).tiny)
    )
    hidden_departure = static["hidden_dual"] @ (
        current_coordinate - static["anchor_coordinate"]
    )
    rank16_amplitude = float(
        np.linalg.norm(static["rank16_basis"].T @ hidden_departure)
    )
    macro_drift = float(
        np.linalg.norm(
            static["macro_restriction"]
            @ (current_coordinate - static["seed_coordinate"])
        )
    )
    metrics = {
        "hidden_secant_fraction": hidden_fraction,
        "rank16_secant_capture": rank16_capture,
        "rank16_hidden_amplitude_from_20ms_anchor": rank16_amplitude,
        "macro_drift_from_warm3_seed": macro_drift,
        "hidden_fraction_gate_passed": hidden_fraction <= manifest.HIDDEN_SECANT_FRACTION_MAX,
        "rank16_amplitude_gate_passed": rank16_amplitude >= manifest.RANK16_HIDDEN_AMPLITUDE_MIN,
        "macro_drift_gate_passed": macro_drift <= manifest.MAXIMUM_MACRO_DRIFT_FROM_SEED,
    }
    arrays = {
        "previous_coordinate470": previous_coordinate,
        "current_coordinate470": current_coordinate,
        "coordinate_secant_rate470_per_s": secant_rate,
        "hidden_secant_rate388_per_s": hidden_rate,
        "hidden_secant_action470_per_s": hidden_action,
    }
    event_gate = bool(
        metrics["hidden_fraction_gate_passed"]
        and metrics["rank16_amplitude_gate_passed"]
        and metrics["macro_drift_gate_passed"]
    )
    return {"metrics": metrics, "arrays": arrays, "event_gate_passed": event_gate}


def _recovered_records(index: int) -> list[tuple[dict, dict[str, np.ndarray]]]:
    if index != 1:
        return []
    helper = _helper()
    metrics = helper._read(manifest.RECOVERED_METRICS)["records"]
    arrays = helper._load_npz(manifest.RECOVERED_ARRAYS)
    names = (
        "coordinate470",
        "decoded_primitive_state",
        "recovered_coordinate470",
        "Q3",
        "coordinate_rate470_per_s",
        "scaled_fixed_Q_rate560_per_s",
        "scaled_reaction_action560_per_s",
    )
    return [
        (
            dict(item),
            {name: np.asarray(arrays[name][record_index]) for name in names},
        )
        for record_index, item in enumerate(metrics)
    ]


def _evaluate(index: int, locked: dict) -> tuple[dict, dict[str, np.ndarray]]:
    helper = _helper()
    base = _base_inputs()
    seed = _seed_and_training(index, base)
    basis, singular_values, rank, training_defect = _canonical_basis(seed["training_rates"])
    duration = _next_duration(seed["previous_metrics"])
    identity = {
        "work_package": WORK_PACKAGE,
        "window_index": index,
        "manifest_hashes": locked["manifest_hashes"],
        "start_coordinate_sha256": hashlib.sha256(np.ascontiguousarray(seed["coordinate"]).tobytes()).hexdigest(),
        "duration_seconds": duration,
        "basis_rank": rank,
    }
    field = _ExactField(
        index=index,
        anchor_state=base["anchor_state"],
        anchor_coordinate=base["anchor_coordinate"],
        transition_path=base["transition_path"],
        model=base["model"],
        geometry=base["geometry"],
        layout=base["layout"],
        configuration=base["configuration"],
        q3_target=base["q3_target"],
        identity=identity,
        seed_metrics=seed["seed_metrics"],
        seed_arrays=seed["seed_arrays"],
        recovered_records=_recovered_records(index),
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
    event = _event_features(seed["state"], endpoint_state, duration)
    required_physical = (
        "coordinate_decomposition", "coordinate_rank", "coordinate_condition",
        "fixed_Q_tangency", "reaction_ledger", "Schur_rank", "Schur_condition",
        "reconstruction", "height", "optical_depth",
    )
    records = list(field.records.values())
    physical_pass = all(all(item["gates"][name] for name in required_physical) for item in records)
    maximum_decoder = max(float(item["decoder_coordinate_error_over_transition_path"]) for item in records)
    maximum_q3 = max(float(item["Q3_relative_drift"]) for item in records)
    minimum_reconstruction = min(
        min(float(item["minimum_reconstruction_factor"]), float(item["decoder_minimum_reconstruction_factor"]), float(item["Q3_minimum_reconstruction_factor"]))
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
        "maximum_decoder_coordinate_error_over_transition_path": maximum_decoder,
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
        "decoder_roundtrip": maximum_decoder <= manifest.MAXIMUM_DECODER_COORDINATE_ERROR_OVER_TRANSITION_PATH,
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
        and maximum_decoder <= manifest.GROW_MAXIMUM_DECODER_COORDINATE_ERROR_OVER_TRANSITION_PATH
        and maximum_q3 <= manifest.GROW_MAXIMUM_Q3_RELATIVE_DRIFT
    )
    persistent_run = seed["event_persistence"] + 1 if passed and event["event_gate_passed"] else 0
    hot_exit = bool(passed and persistent_run >= manifest.HIDDEN_EXIT_PERSISTENCE_WINDOWS)
    budget_exhausted = bool(passed and index >= manifest.MAXIMUM_WINDOWS and not hot_exit)
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
        "exact_evaluation_rates470_per_s": np.stack([field.arrays[key]["coordinate_rate470_per_s"] for key in keys]),
        "exact_evaluation_Q3": np.stack([field.arrays[key]["Q3"] for key in keys]),
        **{f"event__{name}": np.asarray(value) for name, value in event["arrays"].items()},
    }
    return metrics, arrays


def _update_catalog(index: int, summary: dict) -> None:
    helper = _helper()
    post = _post()
    manifest_path = post.manifest.transition.manifest.cold.manifest.CANONICAL_MANIFEST
    summary_path = post.manifest.transition.manifest.cold.manifest.CANONICAL_SUMMARY
    artifact = _stage_directory(index).name
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != artifact]
    status = "SUPPORTED" if summary["passed"] else "REJECTED"
    for path in sorted(_stage_directory(index).iterdir()):
        if path.is_file():
            rows.append({"case": artifact, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": helper._sha(path), "scientific_status": status})
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    catalog = helper._read(summary_path)
    catalog.setdefault("artifacts", {})[artifact] = {"path": str(_stage_directory(index).relative_to(ROOT)), "classification": summary["classification"], "passed": summary["passed"]}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": helper._git("rev-parse", "HEAD"), "latest_work_package": WORK_PACKAGE})
    helper._write_json(summary_path, catalog)


def _run(index: int) -> dict:
    helper = _helper()
    if index < 1 or index > manifest.MAXIMUM_WINDOWS:
        raise ValueError("window index outside frozen budget")
    destination = _stage_directory(index)
    report = _report_path(index)
    if destination.exists() or report.exists():
        raise RuntimeError("adaptive phase-window result already exists")
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
    report.write_text("\n".join((f"# Adaptive hot-exit phase-atlas window {index:02d}", "", f"Classification: `{metrics['classification']}`.", "", f"Duration: `{metrics['duration_seconds']:.6e}` s; selected rank: `{metrics['basis_rank']}`; maximum full defect: `{metrics['gate_values']['maximum_full_collocation_defect']:.6e}`; hidden secant fraction: `{metrics['event_metrics']['hidden_secant_fraction']:.6e}`.", "", f"Window accepted: `{metrics['passed']}`. Persistent event run: `{metrics['persistent_event_window_run']}` of `{manifest.HIDDEN_EXIT_PERSISTENCE_WINDOWS}`. No nonlinear fixed-Q root or BDF microstep was executed.", "")), encoding="utf-8")
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
