#!/usr/bin/env python3
"""Execute the bounded post-transition rank-4 Lobatto phase window."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from imri_qpe.layer3_minidisk_1d.causal_inner_fixed_q import (  # noqa: E402
    causal_five_field_exterior_q3,
)
from imri_qpe.layer3_minidisk_1d.phase_collocation import (  # noqa: E402
    direction_cosine,
    gauss_lobatto_nodes,
    lagrange_differentiation_matrix,
    lagrange_integration_matrix,
)
import run_causal_inner_cold_branch_anchor_preflight_wp10c9d6c7c3b5c4f25dy as exact_rate  # noqa: E402
import run_causal_inner_post_transition_phase_window_manifest_wp10c9d6c7c3b5c4f25ea as manifest  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25eb"
PASS_CLASSIFICATION = "bounded_post_transition_rank4_phase_window_passed"
FAIL_CLASSIFICATION = "bounded_post_transition_rank4_phase_window_rejected"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25ec"
ARTIFACT = "causal_inner_post_transition_phase_window_wp10c9d6c7c3b5c4f25eb"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
THIS_RUNNER = "scripts/run_causal_inner_post_transition_phase_window_wp10c9d6c7c3b5c4f25eb.py"
THIS_TEST = "tests/test_causal_inner_post_transition_phase_window_wp10c9d6c7c3b5c4f25eb.py"
REPORT_RELATIVE = "docs/reports/current/CODEX_CAUSAL_INNER_POST_TRANSITION_PHASE_WINDOW_WP10C9D6C7C3B5C4F25EB_2026-08-21.md"
REPORT_PATH = ROOT / REPORT_RELATIVE


def _helper():
    return manifest._helper()


def _validate_manifest(*, require_clean: bool) -> dict:
    helper = _helper()
    hashes = helper._validate_checksums(manifest.CANONICAL_DIRECTORY)
    summary = helper._read(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = helper._read(
        manifest.CANONICAL_DIRECTORY / "post_transition_phase_window_contract.json"
    )
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or not summary["post_transition_phase_window_execution_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["work_package"] != manifest.WORK_PACKAGE
    ):
        raise RuntimeError("post-transition phase-window manifest changed")
    for relative, expected in contract["frozen_source_hashes"].items():
        if helper._sha(ROOT / relative) != expected:
            raise RuntimeError(f"frozen post-transition source changed: {relative}")
    current_inputs = {
        name: helper._sha(path) for name, path in manifest._decisive_inputs().items()
    }
    if current_inputs != contract["decisive_input_hashes"]:
        raise RuntimeError("post-transition decisive input changed")
    manifest._validate_parent(require_clean=False)
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("post-transition execution requires a clean tracked tree")
    return {"manifest_hashes": hashes, "contract": contract}


def _canonical_basis(rates: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(rates, dtype=float)
    normalized = values / np.linalg.norm(values, axis=1)[:, None]
    _left, singular_values, right = np.linalg.svd(normalized, full_matrices=False)
    basis = np.asarray(right[:rank].T, dtype=float)
    for column in range(basis.shape[1]):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0.0:
            basis[:, column] *= -1.0
    return basis, singular_values


def _picard_window(
    *,
    start_coordinate: np.ndarray,
    start_time_seconds: float,
    duration_seconds: float,
    basis: np.ndarray,
    evaluator: Callable[[np.ndarray, float], np.ndarray],
    node_count: int,
) -> dict[str, np.ndarray | float]:
    """Apply one projected Picard update and audit its nodal derivative."""

    nodes = gauss_lobatto_nodes(node_count)
    integration = lagrange_integration_matrix(nodes)
    differentiation = lagrange_differentiation_matrix(nodes)
    start = np.asarray(start_coordinate, dtype=float)
    projector = basis @ basis.T
    start_rate = np.asarray(evaluator(start, start_time_seconds), dtype=float)
    predictor = start[None, :] + (
        duration_seconds * nodes[:, None] * (projector @ start_rate)[None, :]
    )
    predictor_rates = np.asarray(
        [
            evaluator(coordinate, start_time_seconds + duration_seconds * node)
            for coordinate, node in zip(predictor, nodes, strict=True)
        ]
    )
    reduced_rates = predictor_rates @ basis
    coordinates = start[None, :] + duration_seconds * (integration @ reduced_rates) @ basis.T
    final_rates = np.asarray(
        [
            evaluator(coordinate, start_time_seconds + duration_seconds * node)
            for coordinate, node in zip(coordinates, nodes, strict=True)
        ]
    )
    derivative = differentiation @ coordinates / duration_seconds
    projected_final = (final_rates @ basis) @ basis.T
    tiny = np.finfo(float).tiny
    projected_defects = np.linalg.norm(derivative - projected_final, axis=1) / np.maximum(
        np.linalg.norm(projected_final, axis=1), tiny
    )
    full_defects = np.linalg.norm(derivative - final_rates, axis=1) / np.maximum(
        np.linalg.norm(final_rates, axis=1), tiny
    )
    normal_defects = np.linalg.norm(final_rates - projected_final, axis=1) / np.maximum(
        np.linalg.norm(final_rates, axis=1), tiny
    )
    cosines = np.asarray(
        [direction_cosine(left, right) for left, right in zip(derivative, final_rates, strict=True)]
    )
    return {
        "start_time_seconds": float(start_time_seconds),
        "duration_seconds": float(duration_seconds),
        "nodes": nodes,
        "predictor_coordinates": predictor,
        "predictor_rates": predictor_rates,
        "coordinates": coordinates,
        "final_rates": final_rates,
        "collocation_derivatives": derivative,
        "projected_defects": projected_defects,
        "full_defects": full_defects,
        "normal_defects": normal_defects,
        "direction_cosines": cosines,
        "endpoint": coordinates[-1],
    }


class _ExactField:
    """Restartable exact field evaluator with an anchored local decoder."""

    def __init__(
        self,
        *,
        anchor_state: np.ndarray,
        anchor_coordinate: np.ndarray,
        transition_path: float,
        model,
        geometry: dict[str, np.ndarray],
        layout,
        configuration: dict,
        q3_target: np.ndarray,
        identity: dict,
    ) -> None:
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
        self._prepare_scratch()

    def _prepare_scratch(self) -> None:
        helper = _helper()
        identity_path = SCRATCH_DIRECTORY / "identity.json"
        index_path = SCRATCH_DIRECTORY / "index.json"
        if SCRATCH_DIRECTORY.exists():
            if not identity_path.exists() or helper._read(identity_path) != self.identity:
                raise RuntimeError("post-transition scratch identity changed")
            index = helper._read(index_path) if index_path.exists() else {"records": []}
            for entry in index["records"]:
                key = entry["key"]
                self.records[key] = helper._read(SCRATCH_DIRECTORY / entry["metrics_file"])
                self.arrays[key] = helper._load_npz(SCRATCH_DIRECTORY / entry["arrays_file"])
            return
        SCRATCH_DIRECTORY.mkdir(parents=True)
        helper._write_json(identity_path, self.identity)
        helper._write_json(index_path, {"records": []})

    @staticmethod
    def _key(coordinate: np.ndarray) -> str:
        value = np.ascontiguousarray(np.asarray(coordinate, dtype=float))
        return hashlib.sha256(value.tobytes()).hexdigest()

    def decode(self, coordinate: np.ndarray) -> np.ndarray:
        value = np.asarray(coordinate, dtype=float)
        if np.array_equal(value, self.anchor_coordinate):
            return np.array(self.anchor_state, copy=True)
        decoded = np.asarray(self.model.decoded_state(value), dtype=float)
        return self.anchor_state + decoded - self.anchor_model_state

    def __call__(self, coordinate: np.ndarray, time_seconds: float) -> np.ndarray:
        helper = _helper()
        value = np.asarray(coordinate, dtype=float)
        key = self._key(value)
        if key in self.records:
            np.testing.assert_array_equal(self.arrays[key]["coordinate470"], value)
            return np.array(self.arrays[key]["coordinate_rate470_per_s"], copy=True)
        if len(self.records) >= manifest.MAXIMUM_EXACT_CONTINUOUS_RATE_CALLS:
            raise RuntimeError("exact continuous rate-call budget exhausted")
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
        coordinate_error = float(np.linalg.norm(recovered - value) / self.transition_path)
        q3_drift = float(
            np.linalg.norm(q3 - self.q3_target)
            / max(float(np.linalg.norm(self.q3_target)), np.finfo(float).tiny)
        )
        metrics = {
            **item,
            "requested_time_seconds": float(time_seconds),
            "decoder_coordinate_error_over_transition_path": coordinate_error,
            "Q3_relative_drift": q3_drift,
            "decoder_minimum_reconstruction_factor": float(np.min(factors)),
            "Q3_minimum_reconstruction_factor": float(np.min(q3_factors)),
        }
        arrays = {
            "coordinate470": value,
            "decoded_primitive_state": state,
            "recovered_coordinate470": np.asarray(recovered, dtype=float),
            "Q3": q3,
            "coordinate_rate470_per_s": np.asarray(evidence["coordinate_rate470_per_s"], dtype=float),
            "scaled_fixed_Q_rate560_per_s": np.asarray(evidence["scaled_fixed_Q_rate560_per_s"], dtype=float),
            "scaled_reaction_action560_per_s": np.asarray(evidence["scaled_reaction_action560_per_s"], dtype=float),
        }
        index = len(self.records)
        stem = f"exact_rate_{index:02d}_{key[:12]}"
        metrics_file = stem + ".json"
        arrays_file = stem + ".npz"
        helper._write_json(SCRATCH_DIRECTORY / metrics_file, metrics)
        with (SCRATCH_DIRECTORY / arrays_file).open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        listing = helper._read(SCRATCH_DIRECTORY / "index.json")
        listing["records"].append({"key": key, "metrics_file": metrics_file, "arrays_file": arrays_file})
        helper._write_json(SCRATCH_DIRECTORY / "index.json", listing)
        self.records[key] = metrics
        self.arrays[key] = arrays
        self.new_call_count += 1
        print(
            f"exact rate {len(self.records):02d}/{manifest.MAXIMUM_EXACT_CONTINUOUS_RATE_CALLS}: "
            f"t={time_seconds:.9e}s Q3={q3_drift:.3e} decoder={coordinate_error:.3e}",
            flush=True,
        )
        return np.array(arrays["coordinate_rate470_per_s"], copy=True)


def _window_arrays(prefix: str, window: dict) -> dict[str, np.ndarray]:
    arrays = {}
    for name, value in window.items():
        if isinstance(value, np.ndarray):
            arrays[f"{prefix}__{name}"] = value
    return arrays


def _evaluate(locked: dict) -> tuple[dict, dict[str, np.ndarray]]:
    helper = _helper()
    transition_arrays = helper._load_npz(
        manifest.transition.CANONICAL_DIRECTORY / "transition_collocation_model_and_witnesses.npz"
    )
    exact_transition_rates = np.asarray(
        transition_arrays["heldout_exact_continuous_rates470_per_s"], dtype=float
    )
    basis, singular_values = _canonical_basis(exact_transition_rates, manifest.RATE_BASIS_RANK)
    projected_training = (exact_transition_rates @ basis) @ basis.T
    training_normal = np.linalg.norm(exact_transition_rates - projected_training, axis=1) / np.linalg.norm(exact_transition_rates, axis=1)

    geometry_arrays = helper._load_npz(manifest.transition.manifest.manifest_geometry_path())
    coordinates = np.asarray(geometry_arrays["trajectory_coordinates470"], dtype=float)
    times = np.asarray(geometry_arrays["trajectory_times_seconds"], dtype=float)
    transition_path = float(np.sum(np.linalg.norm(np.diff(coordinates, axis=0), axis=1)))
    anchor_coordinate = np.asarray(coordinates[-1], dtype=float)
    anchor_state = np.asarray(manifest.transition._states()[-1], dtype=float)
    model, _candidate, _fiber = exact_rate.exact_chart._model_and_inputs()
    model_anchor_coordinate, _factors = model.coordinate(anchor_state)
    if np.linalg.norm(model_anchor_coordinate - anchor_coordinate) > 1.0e-12:
        raise RuntimeError("terminal transition coordinate changed")
    layout, configuration, _trajectory, *_unused = exact_rate.rate_source.c4f24._endpoint_data()
    geometry = exact_rate._geometry()
    terminal_checkpoint = helper._load_npz(manifest._decisive_inputs()["terminal_checkpoint"])
    q3_target = np.asarray(terminal_checkpoint["q3_target"], dtype=float)
    identity = {
        "work_package": WORK_PACKAGE,
        "manifest_hashes": locked["manifest_hashes"],
        "frozen_source_hashes": locked["contract"]["frozen_source_hashes"],
        "decisive_input_hashes": locked["contract"]["decisive_input_hashes"],
    }
    field = _ExactField(
        anchor_state=anchor_state,
        anchor_coordinate=anchor_coordinate,
        transition_path=transition_path,
        model=model,
        geometry=geometry,
        layout=layout,
        configuration=configuration,
        q3_target=q3_target,
        identity=identity,
    )
    start_time = float(times[-1])
    began = time.perf_counter()
    full = _picard_window(
        start_coordinate=anchor_coordinate,
        start_time_seconds=start_time,
        duration_seconds=manifest.FULL_DURATION_SECONDS,
        basis=basis,
        evaluator=field,
        node_count=manifest.NODE_COUNT,
    )
    half_1 = _picard_window(
        start_coordinate=anchor_coordinate,
        start_time_seconds=start_time,
        duration_seconds=manifest.HALF_DURATION_SECONDS,
        basis=basis,
        evaluator=field,
        node_count=manifest.NODE_COUNT,
    )
    half_2 = _picard_window(
        start_coordinate=np.asarray(half_1["endpoint"]),
        start_time_seconds=start_time + manifest.HALF_DURATION_SECONDS,
        duration_seconds=manifest.HALF_DURATION_SECONDS,
        basis=basis,
        evaluator=field,
        node_count=manifest.NODE_COUNT,
    )
    execution_wall = float(time.perf_counter() - began)

    all_windows = (full, half_1, half_2)
    all_projected = np.concatenate([np.asarray(item["projected_defects"]) for item in all_windows])
    all_full = np.concatenate([np.asarray(item["full_defects"]) for item in all_windows])
    all_normal = np.concatenate([np.asarray(item["normal_defects"]) for item in all_windows])
    all_cosines = np.concatenate([np.asarray(item["direction_cosines"]) for item in all_windows])
    full_endpoint = np.asarray(full["endpoint"])
    half_endpoint = np.asarray(half_2["endpoint"])
    displacement = max(float(np.linalg.norm(half_endpoint - anchor_coordinate)), np.finfo(float).tiny)
    endpoint_coordinate_defect = float(np.linalg.norm(full_endpoint - half_endpoint) / displacement)
    restriction = np.asarray(geometry["R"], dtype=float)
    macro_displacement = max(float(np.linalg.norm(restriction @ (half_endpoint - anchor_coordinate))), np.finfo(float).tiny)
    endpoint_macro_defect = float(np.linalg.norm(restriction @ (full_endpoint - half_endpoint)) / macro_displacement)
    full_state = field.decode(full_endpoint)
    half_state = field.decode(half_endpoint)
    columns = np.asarray(configuration["columns"], dtype=float).reshape(anchor_state.shape)
    scaled_half_displacement = (half_state - anchor_state) / columns
    endpoint_state_defect = float(
        np.linalg.norm((full_state - half_state) / columns)
        / max(float(np.linalg.norm(scaled_half_displacement)), np.finfo(float).tiny)
    )

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
    records = list(field.records.values())
    physical_pass = all(all(item["gates"][name] for name in required_physical) for item in records)
    maximum_decoder = max(item["decoder_coordinate_error_over_transition_path"] for item in records)
    maximum_q3 = max(item["Q3_relative_drift"] for item in records)
    minimum_reconstruction = min(
        min(item["minimum_reconstruction_factor"], item["decoder_minimum_reconstruction_factor"], item["Q3_minimum_reconstruction_factor"])
        for item in records
    )
    gate_values = {
        "maximum_training_normal_rate_defect": float(np.max(training_normal)),
        "maximum_projected_collocation_defect": float(np.max(all_projected)),
        "maximum_full_collocation_defect": float(np.max(all_full)),
        "maximum_normal_rate_defect": float(np.max(all_normal)),
        "minimum_rate_direction_cosine": float(np.min(all_cosines)),
        "matched_endpoint_coordinate_defect": endpoint_coordinate_defect,
        "matched_endpoint_macro_defect": endpoint_macro_defect,
        "matched_endpoint_state_defect": endpoint_state_defect,
        "maximum_decoder_coordinate_error_over_transition_path": float(maximum_decoder),
        "maximum_Q3_relative_drift": float(maximum_q3),
        "minimum_reconstruction_factor": float(minimum_reconstruction),
        "unique_exact_continuous_rate_calls": len(records),
        "new_exact_continuous_rate_calls_this_process": field.new_call_count,
        "execution_wall_seconds_this_process": execution_wall,
    }
    gates = {
        "training_subspace": gate_values["maximum_training_normal_rate_defect"] <= manifest.MAXIMUM_TRAINING_NORMAL_RATE_DEFECT,
        "projected_collocation": gate_values["maximum_projected_collocation_defect"] <= manifest.MAXIMUM_PROJECTED_COLLOCATION_DEFECT,
        "full_collocation": gate_values["maximum_full_collocation_defect"] <= manifest.MAXIMUM_FULL_COLLOCATION_DEFECT,
        "normal_rate": gate_values["maximum_normal_rate_defect"] <= manifest.MAXIMUM_NORMAL_RATE_DEFECT,
        "rate_direction": gate_values["minimum_rate_direction_cosine"] >= manifest.MINIMUM_RATE_DIRECTION_COSINE,
        "matched_endpoint_coordinate": endpoint_coordinate_defect <= manifest.MAXIMUM_MATCHED_ENDPOINT_COORDINATE_DEFECT,
        "matched_endpoint_macro": endpoint_macro_defect <= manifest.MAXIMUM_MATCHED_ENDPOINT_MACRO_DEFECT,
        "matched_endpoint_state": endpoint_state_defect <= manifest.MAXIMUM_MATCHED_ENDPOINT_STATE_DEFECT,
        "decoder_roundtrip": maximum_decoder <= manifest.MAXIMUM_DECODER_COORDINATE_ERROR_OVER_TRANSITION_PATH,
        "fixed_Q_state_drift": maximum_q3 <= manifest.MAXIMUM_Q3_RELATIVE_DRIFT,
        "reconstruction": minimum_reconstruction >= manifest.MINIMUM_RECONSTRUCTION_FACTOR,
        "exact_rate_physics": physical_pass,
        "truth_budget": len(records) <= manifest.MAXIMUM_EXACT_CONTINUOUS_RATE_CALLS,
        "no_roots_or_microsteps": True,
    }
    passed = bool(all(gates.values()))
    metrics = {
        "classification": PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,
        "passed": passed,
        "gates": gates,
        "gate_values": gate_values,
        "basis_rank": manifest.RATE_BASIS_RANK,
        "node_count_per_window": manifest.NODE_COUNT,
        "full_duration_seconds": manifest.FULL_DURATION_SECONDS,
        "half_duration_seconds": manifest.HALF_DURATION_SECONDS,
        "new_nonlinear_fixed_Q_roots": 0,
        "new_BDF_microsteps": 0,
        "collocation_windows_evaluated": 3,
        "accepted_phase_endpoint": "two_half_window_endpoint" if passed else None,
        "hot_exit_observed": False,
        "predictive_cycle_authorized": False,
        "exact_rate_metrics": records,
    }
    exact_keys = list(field.records)
    arrays = {
        "rate_basis470x4": basis,
        "normalized_training_singular_values": singular_values,
        "training_normal_rate_defects": training_normal,
        "anchor_coordinate470": anchor_coordinate,
        "anchor_primitive_state": anchor_state,
        "Q3_target": q3_target,
        "full_endpoint_coordinate470": full_endpoint,
        "two_half_endpoint_coordinate470": half_endpoint,
        "full_endpoint_primitive_state": full_state,
        "two_half_endpoint_primitive_state": half_state,
        "exact_evaluation_coordinates470": np.stack([field.arrays[key]["coordinate470"] for key in exact_keys]),
        "exact_evaluation_rates470_per_s": np.stack([field.arrays[key]["coordinate_rate470_per_s"] for key in exact_keys]),
        "exact_evaluation_Q3": np.stack([field.arrays[key]["Q3"] for key in exact_keys]),
        **_window_arrays("full", full),
        **_window_arrays("half_1", half_1),
        **_window_arrays("half_2", half_2),
    }
    return metrics, arrays


def _update_catalog(summary: dict) -> None:
    helper = _helper()
    manifest_path = manifest.transition.manifest.cold.manifest.CANONICAL_MANIFEST
    summary_path = manifest.transition.manifest.cold.manifest.CANONICAL_SUMMARY
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    status = "SUPPORTED" if summary["passed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": helper._sha(path), "scientific_status": status})
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    catalog = helper._read(summary_path)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": summary["classification"], "passed": summary["passed"]}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": manifest.PARENT_COMMIT, "latest_work_package": WORK_PACKAGE})
    helper._write_json(summary_path, catalog)


def _run() -> dict:
    helper = _helper()
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("post-transition phase-window result already exists")
    locked = _validate_manifest(require_clean=True)
    metrics, arrays = _evaluate(locked)
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "post_transition_phase_window_metrics.json", metrics)
    with (CANONICAL_DIRECTORY / "post_transition_phase_window_model_and_witnesses.npz").open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    helper._write_json(CANONICAL_DIRECTORY / "input_lock.json", locked)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": metrics["passed"],
        "bounded_post_transition_phase_window_passed": metrics["passed"],
        "architecture_decision_authorized": True,
        "hot_exit_observed": False,
        "hot_exit_execution_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    helper._write_json(CANONICAL_DIRECTORY / "provenance.json", {"runner": THIS_RUNNER, "test": THIS_TEST, "implementation_commit": helper._git("rev-parse", "HEAD"), "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"), "python": sys.version, "numpy": np.__version__, "platform": platform.platform()})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{helper._sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    verdict = "passed" if metrics["passed"] else "was rejected"
    REPORT_PATH.write_text("\n".join(("# Bounded post-transition phase window WP10c9d6c7c3b5c4f25eb", "", f"Classification: `{metrics['classification']}`.", "", f"The rank-4 Lobatto phase window {verdict}. Maximum full collocation defect: `{metrics['gate_values']['maximum_full_collocation_defect']:.6e}`. Matched endpoint coordinate defect: `{metrics['gate_values']['matched_endpoint_coordinate_defect']:.6e}`. Exact rate calls: `{metrics['gate_values']['unique_exact_continuous_rate_calls']}`.", "", "No nonlinear root or BDF microstep was executed. No hot exit is claimed. The next package is a definitions-only cycle-map architecture decision informed by this result.", "")), encoding="utf-8")
    _update_catalog(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("use --run")
    payload = _run()
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
