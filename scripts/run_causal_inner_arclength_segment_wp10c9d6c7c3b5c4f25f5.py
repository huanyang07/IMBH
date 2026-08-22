#!/usr/bin/env python3
"""Execute the first moving exact weighted-arclength hot-mode segment."""

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

from imri_qpe.layer3_minidisk_1d.arclength_phase import (  # noqa: E402
    FastRegimePolicy,
    arclength_picard_window,
    classify_fast_regime,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_fixed_q import (  # noqa: E402
    causal_five_field_exterior_q3,
)
import run_causal_inner_arclength_segment_manifest_wp10c9d6c7c3b5c4f25f4 as manifest  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25f5"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25f6"
PASS_CLASSIFICATION = "moving_exact_arclength_segment_passed_continuing_fast_branch"
CANDIDATE_CLASSIFICATION = "moving_exact_arclength_segment_passed_regime_candidate_requires_refinement"
FAIL_CLASSIFICATION = "moving_exact_arclength_segment_rejected_Window_05_endpoint_preserved"
ARTIFACT = (
    "causal_inner_arclength_segment_"
    "wp10c9d6c7c3b5c4f25f5"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
THIS_RUNNER = manifest.EXECUTION_RUNNER
THIS_TEST = manifest.EXECUTION_TEST
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ARCLENGTH_SEGMENT_"
    "WP10C9D6C7C3B5C4F25F5_2026-08-22.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE


def _helper():
    return manifest._helper()


def _source():
    return manifest.parent.manifest.parent.source


def _transport():
    return manifest.parent


def _exact_chart():
    return _source().exact_chart


def _validate_manifest(*, require_clean: bool) -> dict:
    helper = _helper()
    hashes = helper._validate_checksums(manifest.CANONICAL_DIRECTORY)
    summary = helper._read(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = helper._read(
        manifest.CANONICAL_DIRECTORY / "arclength_segment_contract.json"
    )
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or not summary["arclength_segment_execution_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["work_package"] != manifest.WORK_PACKAGE
    ):
        raise RuntimeError("arclength segment manifest changed")
    for relative, expected in contract["frozen_source_hashes"].items():
        if helper._sha(ROOT / relative) != expected:
            raise RuntimeError(f"frozen arclength segment source changed: {relative}")
    current = {name: helper._sha(path) for name, path in manifest._decisive_inputs().items()}
    if current != contract["decisive_input_hashes"]:
        raise RuntimeError("arclength segment decisive input changed")
    manifest._validate_parent(require_clean=False)
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("arclength segment execution requires a clean tracked tree")
    return {"manifest_hashes": hashes, "contract": contract}


def _anchor_retraction_metrics(anchor_state: np.ndarray, model) -> dict:
    exact_chart = _exact_chart()
    coordinate, factors = model.coordinate(anchor_state)
    physical = exact_chart._physical_audit(model, anchor_state, factors)
    return {
        "passed": bool(physical["passed"]),
        "coordinate_residual_infinity": 0.0,
        "gauge_residual_infinity": 0.0,
        "transport_corrections": 0,
        "target_exact_refreshes": 0,
        "accepted_line_factors": [],
        "residual_history": [0.0],
        "maximum_augmented_condition_number": 0.0,
        "maximum_scaled_anchor_departure": 0.0,
        "wall_seconds": 0.0,
        **physical,
    }


class _TransportedExactField:
    """Restartable exact field with one anchor Jacobian per segment."""

    def __init__(
        self,
        *,
        anchor_state: np.ndarray,
        anchor_coordinate: np.ndarray,
        model,
        gauge_basis: np.ndarray,
        anchor_delta: np.ndarray,
        anchor_augmented: np.ndarray,
        geometry: dict[str, np.ndarray],
        layout,
        configuration: dict,
        q3_target: np.ndarray,
        identity: dict,
        seed_metrics: dict,
        seed_arrays: dict[str, np.ndarray],
    ) -> None:
        self.anchor_state = np.asarray(anchor_state, dtype=float)
        self.anchor_coordinate = np.asarray(anchor_coordinate, dtype=float)
        self.model = model
        self.anchor_model_state = np.asarray(
            model.decoded_state(anchor_coordinate), dtype=float
        )
        self.gauge_basis = np.asarray(gauge_basis, dtype=float)
        self.anchor_delta = np.asarray(anchor_delta, dtype=float)
        self.anchor_augmented = np.asarray(anchor_augmented, dtype=float)
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
        identity_path = SCRATCH_DIRECTORY / "identity.json"
        index_path = SCRATCH_DIRECTORY / "index.json"
        if SCRATCH_DIRECTORY.exists():
            if not identity_path.exists() or helper._read(identity_path) != self.identity:
                raise RuntimeError("arclength segment scratch identity changed")
            listing = helper._read(index_path) if index_path.exists() else {"records": []}
            for entry in listing["records"]:
                metrics = helper._read(SCRATCH_DIRECTORY / entry["metrics_file"])
                arrays = helper._load_npz(SCRATCH_DIRECTORY / entry["arrays_file"])
                self._insert(arrays["coordinate470"], metrics, arrays)
            return
        SCRATCH_DIRECTORY.mkdir(parents=True)
        helper._write_json(identity_path, self.identity)
        helper._write_json(index_path, {"records": []})

    def _raw_initial_state(self, coordinate: np.ndarray) -> np.ndarray:
        decoded = np.asarray(self.model.decoded_state(coordinate), dtype=float)
        return self.anchor_state + decoded - self.anchor_model_state

    def _retract(self, coordinate: np.ndarray) -> tuple[np.ndarray, dict]:
        value = np.asarray(coordinate, dtype=float)
        if np.array_equal(value, self.anchor_coordinate):
            return np.array(self.anchor_state, copy=True), _anchor_retraction_metrics(
                self.anchor_state, self.model
            )
        state, _matrix, metrics = _transport()._transport_retract(
            model=self.model,
            initial_state=self._raw_initial_state(value),
            target=value,
            gauge_basis=self.gauge_basis,
            anchor_delta=self.anchor_delta,
            anchor_augmented=self.anchor_augmented,
        )
        return state, metrics

    def decode(self, coordinate: np.ndarray) -> np.ndarray:
        value = np.asarray(coordinate, dtype=float)
        key = self._key(value)
        if key in self.arrays:
            return np.array(self.arrays[key]["decoded_primitive_state"], copy=True)
        state, _metrics = self._retract(value)
        return state

    def __call__(self, coordinate: np.ndarray, time_seconds: float) -> np.ndarray:
        helper = _helper()
        source = _source()
        value = np.asarray(coordinate, dtype=float)
        key = self._key(value)
        if key in self.records:
            np.testing.assert_array_equal(self.arrays[key]["coordinate470"], value)
            return np.array(self.arrays[key]["coordinate_rate470_per_s"], copy=True)
        if len(self.records) >= manifest.MAXIMUM_UNIQUE_RATE_STATES:
            raise RuntimeError("arclength segment exact-rate budget exhausted")
        state, retraction = self._retract(value)
        item, evidence = source._post().exact_rate._evaluate_candidate(
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
            "coordinate_rate470_per_s": np.asarray(
                evidence["coordinate_rate470_per_s"], dtype=float
            ),
            "scaled_fixed_Q_rate560_per_s": np.asarray(
                evidence["scaled_fixed_Q_rate560_per_s"], dtype=float
            ),
            "scaled_reaction_action560_per_s": np.asarray(
                evidence["scaled_reaction_action560_per_s"], dtype=float
            ),
        }
        stem = f"exact_rate_{len(self.records):02d}_{key[:12]}"
        metrics_file, arrays_file = stem + ".json", stem + ".npz"
        helper._write_json(SCRATCH_DIRECTORY / metrics_file, metrics)
        with (SCRATCH_DIRECTORY / arrays_file).open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        listing = helper._read(SCRATCH_DIRECTORY / "index.json")
        listing["records"].append({
            "key": key,
            "metrics_file": metrics_file,
            "arrays_file": arrays_file,
        })
        helper._write_json(SCRATCH_DIRECTORY / "index.json", listing)
        self._insert(value, metrics, arrays)
        self.new_call_count += 1
        print(
            f"arclength exact rate {len(self.records):02d}/"
            f"{manifest.MAXIMUM_UNIQUE_RATE_STATES}: "
            f"t={time_seconds:.9e}s Q3={metrics['Q3_relative_drift']:.3e} "
            f"coord={retraction['coordinate_residual_infinity']:.3e} "
            f"departure={retraction['maximum_scaled_anchor_departure']:.3e}",
            flush=True,
        )
        return np.array(arrays["coordinate_rate470_per_s"], copy=True)


def _seed(base: dict) -> dict:
    source = _source()
    seed = source._seed_and_training(6, base)
    if float(seed["time_seconds"]) != 2.499999999999999e-6:
        raise RuntimeError("Window-5 endpoint time changed")
    return seed


def _regime_metrics(
    *,
    seed: dict,
    endpoint_coordinate: np.ndarray,
    endpoint_state: np.ndarray,
    endpoint_rate: np.ndarray,
    physical_duration: float,
) -> dict:
    helper = _helper()
    diagnosis_directory = manifest.parent.manifest.parent.CANONICAL_DIRECTORY
    diagnosis = helper._load_npz(diagnosis_directory / "arclength_event_arrays.npz")
    legacy = _source()._adaptive()._event_features(
        seed["state"], endpoint_state, physical_duration
    )
    history_points = np.vstack((
        diagnosis["window_01__start_coordinate470"],
        diagnosis["endpoint_coordinates470"][:-1],
    ))
    history_directions = np.vstack((
        diagnosis["window_01__unit_rates470"][0],
        diagnosis["endpoint_unit_rates470"][:-1],
    ))
    endpoint_speed = float(np.linalg.norm(endpoint_rate))
    endpoint_direction = endpoint_rate / endpoint_speed
    distances = np.linalg.norm(history_points - endpoint_coordinate[None, :], axis=1)
    closest = int(np.argmin(distances))
    closest_ratio = float(distances[closest] / manifest.ARCLENGTH_SPAN)
    closest_cosine = float(history_directions[closest] @ endpoint_direction)
    reference_speed = float(diagnosis["window_01__phase_speeds_per_second"][0])
    speed_ratio = endpoint_speed / reference_speed
    policy = FastRegimePolicy()
    legacy_run = seed["event_persistence"] + 1 if legacy["event_gate_passed"] else 0
    equilibrium_run = 1 if speed_ratio <= policy.equilibrium_speed_ratio_maximum else 0
    recurrence_run = 1 if (
        closest_ratio <= policy.recurrence_distance_over_local_span_maximum
        and closest_cosine >= policy.recurrence_direction_cosine_minimum
    ) else 0
    classification = classify_fast_regime(
        legacy_exit_run=legacy_run,
        equilibrium_run=equilibrium_run,
        recurrence_run=recurrence_run,
        terminal_speed_ratio=speed_ratio,
        closest_return_distance_over_local_span=closest_ratio,
        closest_return_direction_cosine=closest_cosine,
        policy=policy,
    )
    return {
        "classification": classification,
        "legacy_event": legacy,
        "legacy_exit_run": legacy_run,
        "equilibrium_run": equilibrium_run,
        "recurrence_run": recurrence_run,
        "terminal_speed_over_hot_entry_speed": speed_ratio,
        "closest_historical_point_index": closest,
        "closest_return_distance": float(distances[closest]),
        "closest_return_distance_over_local_span": closest_ratio,
        "closest_return_direction_cosine": closest_cosine,
    }


def _evaluate(locked: dict) -> tuple[dict, dict[str, np.ndarray]]:
    source = _source()
    exact_chart = _exact_chart()
    base = source._base_inputs()
    seed = _seed(base)
    anchor_coordinate, anchor_factors = base["model"].coordinate(seed["state"])
    np.testing.assert_array_equal(anchor_coordinate, seed["coordinate"])
    anchor_q3, anchor_q3_factors = causal_five_field_exterior_q3(
        base["configuration"]["context"],
        seed["state"],
        exterior_face_index=36 * int(base["layout"].refinement_ratio),
    )
    anchor_retraction = _anchor_retraction_metrics(seed["state"], base["model"])
    seed["seed_metrics"].update({
        "requested_time_seconds": float(seed["time_seconds"]),
        "retraction": anchor_retraction,
        "Q3_relative_drift": float(
            np.linalg.norm(anchor_q3 - base["q3_target"])
            / max(float(np.linalg.norm(base["q3_target"])), np.finfo(float).tiny)
        ),
        "decoder_minimum_reconstruction_factor": float(np.min(anchor_factors)),
        "Q3_minimum_reconstruction_factor": float(np.min(anchor_q3_factors)),
    })
    seed["seed_arrays"].update({
        "coordinate470": np.asarray(anchor_coordinate),
        "decoded_primitive_state": np.asarray(seed["state"]),
        "recovered_coordinate470": np.asarray(anchor_coordinate),
        "Q3": np.asarray(anchor_q3),
    })
    basis, singular_values, rank, training_defect = source._canonical_basis(
        seed["training_rates"]
    )
    coordinate_jacobian, coordinate_metrics = exact_chart._coordinate_jacobian(
        base["model"], seed["state"]
    )
    gauge_basis = exact_chart._canonical_null_basis(coordinate_jacobian)
    began_anchor = time.perf_counter()
    anchor_augmented, augmented_metrics = exact_chart._augmented_jacobian(
        base["model"], seed["state"], gauge_basis
    )
    anchor_assembly_wall = float(time.perf_counter() - began_anchor)
    anchor_delta = exact_chart._delta(base["model"], seed["state"])
    if augmented_metrics["augmented_rank"] != exact_chart.PHYSICAL_DIMENSION:
        raise RuntimeError("arclength anchor augmented Jacobian lost rank")
    identity = {
        "work_package": WORK_PACKAGE,
        "manifest_hashes": locked["manifest_hashes"],
        "anchor_state_sha256": hashlib.sha256(
            np.ascontiguousarray(seed["state"]).tobytes()
        ).hexdigest(),
        "anchor_coordinate_sha256": hashlib.sha256(
            np.ascontiguousarray(seed["coordinate"]).tobytes()
        ).hexdigest(),
        "arclength_span": manifest.ARCLENGTH_SPAN,
        "basis_rank": rank,
        "node_count": manifest.NODE_COUNT,
    }
    field = _TransportedExactField(
        anchor_state=seed["state"],
        anchor_coordinate=seed["coordinate"],
        model=base["model"],
        gauge_basis=gauge_basis,
        anchor_delta=anchor_delta,
        anchor_augmented=anchor_augmented,
        geometry=base["geometry"],
        layout=base["layout"],
        configuration=base["configuration"],
        q3_target=base["q3_target"],
        identity=identity,
        seed_metrics=seed["seed_metrics"],
        seed_arrays=seed["seed_arrays"],
    )
    began = time.perf_counter()
    window = arclength_picard_window(
        start_coordinate=seed["coordinate"],
        start_time_seconds=seed["time_seconds"],
        arclength_span=manifest.ARCLENGTH_SPAN,
        basis=basis,
        evaluator=field,
        node_count=manifest.NODE_COUNT,
    )
    wall_seconds = float(time.perf_counter() - began)
    endpoint_coordinate = np.asarray(window["endpoint"], dtype=float)
    endpoint_state = field.decode(endpoint_coordinate)
    endpoint_rate = np.asarray(window["final_rates_per_second"][-1], dtype=float)
    regime = _regime_metrics(
        seed=seed,
        endpoint_coordinate=endpoint_coordinate,
        endpoint_state=endpoint_state,
        endpoint_rate=endpoint_rate,
        physical_duration=float(window["physical_duration_seconds"]),
    )
    records = list(field.records.values())
    required_physical = (
        "coordinate_decomposition",
        "coordinate_rank",
        "coordinate_condition",
        "fixed_Q_tangency",
        "reaction_ledger",
        "Schur_rank",
        "Schur_condition",
        "reconstruction",
        "height",
        "optical_depth",
    )
    physical_pass = all(
        all(item["gates"][name] for name in required_physical) for item in records
    )
    retractions = [item["retraction"] for item in records]
    maximum_coordinate = max(item["coordinate_residual_infinity"] for item in retractions)
    maximum_gauge = max(item["gauge_residual_infinity"] for item in retractions)
    maximum_condition = max(
        float(augmented_metrics["augmented_condition_number"]),
        *(float(item["maximum_augmented_condition_number"]) for item in retractions),
    )
    maximum_departure = max(item["maximum_scaled_anchor_departure"] for item in retractions)
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
    maximum_time = float(np.max(window["time_mapping_defects"]))
    speeds = np.asarray(window["final_phase_speeds_per_second"])
    speed_ratio = float(np.min(speeds) / np.max(speeds))
    target_refreshes = sum(int(item["target_exact_refreshes"]) for item in retractions)
    gate_values = {
        "maximum_training_normal_rate_defect": training_defect,
        "maximum_projected_collocation_defect": maximum_projected,
        "maximum_full_collocation_defect": maximum_full,
        "maximum_normal_rate_defect": maximum_normal,
        "minimum_rate_direction_cosine": minimum_cosine,
        "maximum_time_mapping_defect": maximum_time,
        "minimum_phase_speed_ratio": speed_ratio,
        "minimum_phase_speed_per_second": float(np.min(speeds)),
        "maximum_phase_speed_per_second": float(np.max(speeds)),
        "maximum_coordinate_residual_infinity": maximum_coordinate,
        "maximum_gauge_residual_infinity": maximum_gauge,
        "maximum_augmented_condition_number": maximum_condition,
        "maximum_scaled_anchor_departure": maximum_departure,
        "maximum_Q3_relative_drift": maximum_q3,
        "minimum_reconstruction_factor": minimum_reconstruction,
        "unique_exact_rate_states": len(field.records),
        "new_exact_rate_calls_this_process": field.new_call_count,
        "anchor_exact_assemblies": 1,
        "target_exact_refreshes": target_refreshes,
        "anchor_assembly_wall_seconds": anchor_assembly_wall,
        "execution_wall_seconds_this_process": wall_seconds,
        "physical_duration_seconds": float(window["physical_duration_seconds"]),
    }
    gates = {
        "training_subspace": training_defect
        <= _source().manifest.MAXIMUM_TRAINING_NORMAL_RATE_DEFECT,
        "projected_collocation": maximum_projected
        <= manifest.MAXIMUM_PROJECTED_COLLOCATION_DEFECT,
        "full_collocation": maximum_full <= manifest.MAXIMUM_FULL_COLLOCATION_DEFECT,
        "normal_rate": maximum_normal <= manifest.MAXIMUM_NORMAL_RATE_DEFECT,
        "rate_direction": minimum_cosine >= manifest.MINIMUM_RATE_DIRECTION_COSINE,
        "time_mapping": maximum_time <= manifest.MAXIMUM_TIME_MAPPING_DEFECT,
        "phase_speed_conditioning": speed_ratio >= manifest.MINIMUM_PHASE_SPEED_RATIO,
        "coordinate_retraction": maximum_coordinate
        <= _source().manifest.COORDINATE_TOLERANCE,
        "gauge_retraction": maximum_gauge <= _source().manifest.GAUGE_TOLERANCE,
        "retraction_conditioning": maximum_condition
        <= _source().manifest.MAXIMUM_AUGMENTED_CONDITION_NUMBER,
        "retraction_neighborhood": maximum_departure
        <= _source().manifest.MAXIMUM_SCALED_ANCHOR_DEPARTURE,
        "retraction_physics": all(bool(item["passed"]) for item in retractions),
        "fixed_Q_state_drift": maximum_q3 <= manifest.MAXIMUM_Q3_RELATIVE_DRIFT,
        "reconstruction": minimum_reconstruction >= manifest.MINIMUM_RECONSTRUCTION_FACTOR,
        "exact_rate_physics": physical_pass,
        "truth_budget": len(field.records) <= manifest.MAXIMUM_UNIQUE_RATE_STATES,
        "transport_refresh_budget": target_refreshes
        <= _transport().manifest.MAXIMUM_TOTAL_TARGET_EXACT_REFRESHES,
        "positive_physical_time": float(window["physical_duration_seconds"]) > 0.0,
        "no_roots_or_microsteps": True,
    }
    passed = bool(all(gates.values()))
    growth_margin = bool(
        passed
        and maximum_full <= manifest.GROW_MAXIMUM_FULL_COLLOCATION_DEFECT
        and maximum_time <= manifest.GROW_MAXIMUM_TIME_MAPPING_DEFECT
        and maximum_departure <= manifest.GROW_MAXIMUM_ANCHOR_DEPARTURE
    )
    regime_candidate = regime["classification"] != "continuing_fast_branch"
    if not passed:
        classification = FAIL_CLASSIFICATION
    elif regime_candidate:
        classification = CANDIDATE_CLASSIFICATION
    else:
        classification = PASS_CLASSIFICATION
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "basis_rank": rank,
        "node_count": manifest.NODE_COUNT,
        "arclength_span": manifest.ARCLENGTH_SPAN,
        "start_time_seconds": float(seed["time_seconds"]),
        "end_time_seconds": float(window["end_time_seconds"]),
        "physical_duration_seconds": float(window["physical_duration_seconds"]),
        "anchor_coordinate_metrics": coordinate_metrics,
        "anchor_augmented_metrics": augmented_metrics,
        "gates": gates,
        "gate_values": gate_values,
        "growth_margin_passed": growth_margin,
        "regime": regime,
        "regime_candidate_observed": regime_candidate,
        "new_nonlinear_fixed_Q_roots": 0,
        "new_BDF_microsteps": 0,
        "exact_rate_metrics": records,
        "input_lock": locked,
    }
    keys = list(field.records)
    arrays = {
        "basis470xr": basis,
        "training_rates470_per_s": np.asarray(seed["training_rates"]),
        "training_singular_values": singular_values,
        "anchor_coordinate470": np.asarray(seed["coordinate"]),
        "anchor_delta560": anchor_delta,
        "anchor_gauge_basis560x90": gauge_basis,
        "anchor_augmented_jacobian560x560": anchor_augmented,
        "start_coordinate470": np.asarray(seed["coordinate"]),
        "start_primitive_state": np.asarray(seed["state"]),
        "endpoint_coordinate470": endpoint_coordinate,
        "endpoint_primitive_state": endpoint_state,
        "Q3_target": base["q3_target"],
        **{
            name: np.asarray(value)
            for name, value in window.items()
            if isinstance(value, np.ndarray)
        },
        "exact_evaluation_coordinates470": np.stack(
            [field.arrays[key]["coordinate470"] for key in keys]
        ),
        "exact_evaluation_recovered_coordinates470": np.stack(
            [field.arrays[key]["recovered_coordinate470"] for key in keys]
        ),
        "exact_evaluation_primitive_states": np.stack(
            [field.arrays[key]["decoded_primitive_state"] for key in keys]
        ),
        "exact_evaluation_rates470_per_s": np.stack(
            [field.arrays[key]["coordinate_rate470_per_s"] for key in keys]
        ),
        "exact_evaluation_Q3": np.stack([field.arrays[key]["Q3"] for key in keys]),
        "regime__legacy_previous_coordinate470": np.asarray(
            regime["legacy_event"]["arrays"]["previous_coordinate470"]
        ),
        "regime__legacy_current_coordinate470": np.asarray(
            regime["legacy_event"]["arrays"]["current_coordinate470"]
        ),
    }
    return metrics, arrays


def _update_catalog(summary: dict) -> None:
    helper = _helper()
    cold = _source()._post().manifest.transition.manifest.cold.manifest
    with cold.CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
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
    with cold.CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case", "path", "bytes", "sha256", "scientific_status"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = helper._read(cold.CANONICAL_SUMMARY)
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
    helper._write_json(cold.CANONICAL_SUMMARY, catalog)


def _run() -> dict:
    helper = _helper()
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("arclength segment result already exists")
    locked = _validate_manifest(require_clean=True)
    metrics, arrays = _evaluate(locked)
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "arclength_segment_metrics.json", metrics)
    with (CANONICAL_DIRECTORY / "arclength_segment_arrays.npz").open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    checkpoint = {
        "endpoint_coordinate470": arrays["endpoint_coordinate470"],
        "endpoint_primitive_state": arrays["endpoint_primitive_state"],
        "Q3_target": arrays["Q3_target"],
        "end_time_seconds": np.asarray(metrics["end_time_seconds"]),
        "arclength_span": np.asarray(metrics["arclength_span"]),
        "physical_duration_seconds": np.asarray(metrics["physical_duration_seconds"]),
    }
    with (CANONICAL_DIRECTORY / "arclength_segment_checkpoint.npz").open("wb") as handle:
        np.savez_compressed(handle, **checkpoint)
    reloaded = helper._load_npz(CANONICAL_DIRECTORY / "arclength_segment_checkpoint.npz")
    for name, value in checkpoint.items():
        np.testing.assert_array_equal(reloaded[name], value)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": metrics["passed"],
        "checkpoint_roundtrip_bitwise": True,
        "regime_classification": metrics["regime"]["classification"],
        "regime_candidate_observed": metrics["regime_candidate_observed"],
        "adaptive_arclength_atlas_manifest_authorized": metrics["passed"],
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT if metrics["passed"] else None,
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    helper._write_json(CANONICAL_DIRECTORY / "provenance.json", {
        "runner": THIS_RUNNER,
        "test": THIS_TEST,
        "implementation_commit": helper._git("rev-parse", "HEAD"),
        "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"),
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
    })
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{helper._sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    values = metrics["gate_values"]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join((
            "# First moving exact weighted-arclength segment",
            "",
            f"Classification: `{metrics['classification']}`.",
            "",
            f"Arclength `{metrics['arclength_span']:.6e}` produced physical duration `{metrics['physical_duration_seconds']:.6e}` s using `{values['unique_exact_rate_states']}` exact offline rates, one anchor coordinate Jacobian, and `{values['target_exact_refreshes']}` target refreshes.",
            "",
            f"Maximum full phase defect `{values['maximum_full_collocation_defect']:.6e}`; time-map defect `{values['maximum_time_mapping_defect']:.6e}`; coordinate residual `{values['maximum_coordinate_residual_infinity']:.6e}`; local departure `{values['maximum_scaled_anchor_departure']:.6e}`.",
            "",
            f"Fast-regime classification: `{metrics['regime']['classification']}`. No nonlinear fixed-Q root or BDF microstep was executed.",
            "",
        )),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("use --run")
    print(json.dumps(_run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
