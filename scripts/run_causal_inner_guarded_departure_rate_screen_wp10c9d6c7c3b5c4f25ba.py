#!/usr/bin/env python3
"""Execute the guarded primary-anchor nonlinear departure-rate screen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_guarded_departure_rate_screen_manifest_wp10c9d6c7c3b5c4f25az as manifest  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_fixed_q import (  # noqa: E402
    causal_five_field_fixed_q_reaction,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_monolithic_dae import (  # noqa: E402
    evaluate_causal_five_field_monolithic_backward_euler,
)
from run_causal_inner_monolithic_bdf_base_preflight_wp10c9d6c7c3b1a import (  # noqa: E402
    _state_audit,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25ba"
MANIFEST_COMMIT = "fd8688f4691a8bb96b14077e703f61d295964abc"
MANIFEST_PARENT = "e6845f094420890c115250679e705e2b4980dcc8"
MANIFEST_TREE = "a7ce3f3638967bd87fe8dc627cfc5d257d2e9613"

NONLINEAR_CLASSIFICATION = (
    "guarded_primary_departure_rate_screen_passed_"
    "nonlinear_signal_resolved_mixed_direction_database_manifest_authorized"
)
UNRESOLVED_CLASSIFICATION = (
    "guarded_primary_departure_rate_screen_passed_"
    "nonlinear_signal_not_resolved_expanded_chart_manifest_authorized"
)
FAIL_CLASSIFICATION = (
    "guarded_primary_departure_rate_screen_failed_"
    "nonlinear_closure_identification_blocked"
)

ARTIFACT = (
    "causal_inner_guarded_departure_rate_screen_"
    "wp10c9d6c7c3b5c4f25ba"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_guarded_departure_rate_screen_"
    "wp10c9d6c7c3b5c4f25ba.py"
)
THIS_TEST = (
    "tests/test_causal_inner_guarded_departure_rate_screen_"
    "wp10c9d6c7c3b5c4f25ba.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_GUARDED_DEPARTURE_RATE_"
    "SCREEN_WP10C9D6C7C3B5C4F25BA_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


class RateEvaluationFailure(RuntimeError):
    """Carry fail-closed diagnostics for one nonbase truth evaluation."""

    def __init__(self, message: str, diagnostics: dict):
        super().__init__(message)
        self.diagnostics = diagnostics


def _plain(value):
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, int):
        return value
    return value


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _tracked_tree_clean() -> bool:
    return not _git("status", "--short", "--untracked-files=no")


def _checksums(directory: Path) -> dict[str, str]:
    recorded = {}
    for line in (directory / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        if _sha(directory / name) != expected:
            raise RuntimeError(f"checksum mismatch: {directory / name}")
        recorded[name] = expected
    return recorded


def _validate_manifest(*, require_clean: bool) -> dict:
    if _git("rev-parse", MANIFEST_COMMIT) != MANIFEST_COMMIT:
        raise RuntimeError("departure-rate screen manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("departure-rate screen manifest lineage changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("departure-rate screen manifest tree changed")
    hashes = _checksums(manifest.ARTIFACT_DIRECTORY)
    summary = _read(manifest.ARTIFACT_DIRECTORY / "summary.json")
    contract = _read(manifest.ARTIFACT_DIRECTORY / "contract.json")
    provenance = _read(manifest.ARTIFACT_DIRECTORY / "provenance.json")
    lock = _read(manifest.ARTIFACT_DIRECTORY / "parent_lock.json")
    if (
        not summary["passed"]
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["planned_nonbase_continuous_rate_evaluations"]
        != manifest.CANDIDATE_COUNT
        or summary["full_closure_database_claimed"]
        or contract["claim_boundary"][
            "48_axial_samples_are_a_full_28D_closure_database"
        ]
    ):
        raise RuntimeError("departure-rate screen authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"manifest source changed: {relative}")
    for name, path in (
        ("geometric_departure_chart", manifest.CHART_PATH),
        ("online_470_geometry", manifest.GEOMETRY_PATH),
        ("complete_primary_generator", manifest.GENERATOR_PATH),
    ):
        if _sha(path) != lock["decisive_input_hashes"][name]:
            raise RuntimeError(f"rate-screen input changed: {path}")
    _checksums(manifest.parent.CANONICAL_DIRECTORY)
    if require_clean and not _tracked_tree_clean():
        raise RuntimeError("departure-rate screen requires a clean tracked tree")
    for name, expected in manifest.parent.coordinate_tools.THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    return {"summary": summary, "contract": contract, "hashes": hashes}


def _relative_error(actual: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(actual) - np.asarray(reference))
        / max(float(np.linalg.norm(reference)), np.finfo(float).tiny)
    )


def _load_inputs() -> dict:
    chart_metrics = _read(manifest.parent.CANONICAL_DIRECTORY / "metrics.json")
    with np.load(manifest.CHART_PATH, allow_pickle=False) as source:
        chart = {name: np.asarray(source[name], dtype=float) for name in source.files}
    with np.load(manifest.GEOMETRY_PATH, allow_pickle=False) as source:
        geometry = {name: np.asarray(source[name], dtype=float) for name in source.files}
    with np.load(manifest.GENERATOR_PATH, allow_pickle=False) as source:
        generator = np.asarray(source["complete_fixed_Q_generator"], dtype=float)
        base_rate = np.asarray(source["fixed_Q_rate"], dtype=float)
    states = chart["candidate_primitive_states"]
    deltas = chart["candidate_scaled_deltas"]
    coordinates = chart["candidate_departure_coordinates"]
    candidates = chart_metrics["candidates"]
    if (
        states.shape[0] != manifest.CANDIDATE_COUNT
        or deltas.shape != (manifest.CANDIDATE_COUNT, 560)
        or coordinates.shape != (manifest.CANDIDATE_COUNT, 28)
        or len(candidates) != manifest.CANDIDATE_COUNT
        or generator.shape != (560, 560)
        or base_rate.shape != (560,)
    ):
        raise RuntimeError("guarded rate-screen input dimensions changed")
    if [item["candidate_index"] for item in candidates] != list(
        range(manifest.CANDIDATE_COUNT)
    ):
        raise RuntimeError("guarded chart candidate ordering changed")
    return {
        "chart": chart,
        "states": states,
        "deltas": deltas,
        "coordinates": coordinates,
        "candidates": candidates,
        "restriction": geometry["online_coordinate_restriction"],
        "memory_basis": geometry["stable_memory_coordinate_basis"],
        "departure_basis": geometry["departure_coordinate_basis"],
        "generator": generator,
        "base_rate": base_rate,
    }


def _continuous_rate(data: dict, state: np.ndarray) -> tuple[dict, dict[str, np.ndarray]]:
    timing: dict[str, float] = {}
    began = time.perf_counter()
    reaction = causal_five_field_fixed_q_reaction(
        data["context"],
        state,
        primitive_column_scales=data["columns"],
        conservation_row_scales=data["rows"],
        parent_cell_indices=data["layout"].parent_cell_indices,
        refinement_ratio=data["layout"].refinement_ratio,
        maximum_schur_condition_number=1.0e8,
        timing_accumulator=timing,
    )
    evaluation = evaluate_causal_five_field_monolithic_backward_euler(
        state, state, 1.0, data["context"], path_quadrature_order=6
    )
    stationary = np.asarray(evaluation.residual_rows, dtype=float).ravel() / data[
        "rows"
    ].ravel()
    free = np.linalg.solve(reaction.descriptor_scaled_matrix, -stationary)
    multiplier = -reaction.q3_scaled_derivative @ free
    action = reaction.reaction_lift @ multiplier
    rate = free + action
    physical = _state_audit(data["context"], state)
    minimum_factor = min(
        reaction.minimum_q3_reconstruction_factor,
        physical["minimum_reconstruction_factor"],
    )
    metrics = {
        "rate_norm_per_second": float(np.linalg.norm(rate)),
        "free_rate_norm_per_second": float(np.linalg.norm(free)),
        "reaction_action_norm_per_second": float(np.linalg.norm(action)),
        "raw_Schur_condition_number": reaction.raw_schur_condition_number,
        "reaction_identity_defect": reaction.maximum_identity_defect,
        "rate_tangency_relative_defect": float(
            np.linalg.norm(reaction.q3_scaled_derivative @ rate)
            / max(float(np.linalg.norm(rate)), np.finfo(float).tiny)
        ),
        "minimum_reconstruction_factor": minimum_factor,
        "maximum_reconstruction_factor": reaction.maximum_q3_reconstruction_factor,
        "maximum_H_over_R": physical["maximum_h_over_r"],
        "minimum_scattering_optical_depth": physical[
            "minimum_scattering_optical_depth"
        ],
        "incoming_excision_characteristics": int(
            evaluation.incoming_excision_characteristics
        ),
        "wall_seconds": time.perf_counter() - began,
        "timing": timing,
    }
    return metrics, {
        "total_rate": rate,
        "free_rate": free,
        "reaction_action": action,
        "multiplier": multiplier,
    }


def _radial_analysis(
    candidates: list[dict],
    departure_coordinates: np.ndarray,
    departure_rate_increments: np.ndarray,
    departure_linear_references: np.ndarray,
    directions: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    direction_count = directions.shape[1]
    amplitude_count = len(manifest.parent.manifest.MAXIMUM_COMPONENT_BOUNDS)
    growth = np.empty((direction_count, amplitude_count), dtype=float)
    linear_growth = np.empty_like(growth)
    nonlinear_fraction = np.empty_like(growth)
    effective_radii = np.empty_like(growth)
    cubic = np.empty(direction_count, dtype=float)
    for direction_index in range(direction_count):
        direction = directions[:, direction_index]
        for amplitude_index in range(amplitude_count):
            indices = [
                index
                for index, item in enumerate(candidates)
                if item["direction_index"] == direction_index
                and item["amplitude_index"] == amplitude_index
            ]
            if len(indices) != 2:
                raise RuntimeError("signed radial pair is incomplete")
            negative, positive = sorted(indices, key=lambda index: candidates[index]["sign"])
            coordinate_odd = 0.5 * (
                departure_coordinates[positive] - departure_coordinates[negative]
            )
            rate_odd = 0.5 * (
                departure_rate_increments[positive]
                - departure_rate_increments[negative]
            )
            linear_odd = 0.5 * (
                departure_linear_references[positive]
                - departure_linear_references[negative]
            )
            radius = float(direction @ coordinate_odd)
            if radius <= np.finfo(float).tiny:
                raise RuntimeError("radial coordinate lost its signed direction")
            effective_radii[direction_index, amplitude_index] = radius
            growth[direction_index, amplitude_index] = float(
                direction @ rate_odd / radius
            )
            linear_growth[direction_index, amplitude_index] = float(
                direction @ linear_odd / radius
            )
            nonlinear_fraction[direction_index, amplitude_index] = (
                _relative_error(rate_odd, linear_odd)
            )
        cubic[direction_index] = float(
            np.polyfit(
                effective_radii[direction_index] ** 2,
                growth[direction_index],
                1,
            )[0]
        )
    largest = nonlinear_fraction[:, -1]
    metrics = {
        "minimum_smallest_radial_growth_per_second": float(np.min(growth[:, 0])),
        "maximum_smallest_radial_growth_per_second": float(np.max(growth[:, 0])),
        "minimum_largest_radial_growth_per_second": float(np.min(growth[:, -1])),
        "maximum_largest_radial_growth_per_second": float(np.max(growth[:, -1])),
        "nonpositive_largest_radial_growth_count": int(
            np.count_nonzero(growth[:, -1] <= 0.0)
        ),
        "negative_fitted_cubic_count": int(np.count_nonzero(cubic < 0.0)),
        "median_largest_departure_nonlinear_relative_defect": float(
            np.median(largest)
        ),
        "minimum_largest_departure_nonlinear_relative_defect": float(
            np.min(largest)
        ),
        "maximum_largest_departure_nonlinear_relative_defect": float(
            np.max(largest)
        ),
    }
    arrays = {
        "effective_departure_radii": effective_radii,
        "central_radial_growth_per_second": growth,
        "central_linear_radial_growth_per_second": linear_growth,
        "central_departure_nonlinear_relative_defects": nonlinear_fraction,
        "fitted_cubic_growth_coefficients": cubic,
    }
    return metrics, arrays


def _evaluate() -> tuple[dict, dict[str, np.ndarray]]:
    inputs = _load_inputs()
    data = manifest.parent.manifest.failed_screen._anchor_data("primary")
    components = manifest.parent.coordinate_tools._coordinate_components()
    candidates = inputs["candidates"]
    total_rates = []
    free_rates = []
    actions = []
    multipliers = []
    online_rates = []
    departure_rates = []
    linear_references = []
    departure_linear_references = []
    evaluation_metrics = []
    failures = []
    began = time.perf_counter()
    for index, state in enumerate(inputs["states"]):
        try:
            item, arrays = _continuous_rate(data, state)
            coordinate_jacobian, coordinate_metrics = (
                manifest.parent._coordinate_jacobian(state, components)
            )
            linear = inputs["generator"] @ inputs["deltas"][index]
            increment = arrays["total_rate"] - inputs["base_rate"]
            departure_increment = inputs["departure_basis"].T @ increment
            departure_linear = inputs["departure_basis"].T @ linear
            online_rate = np.concatenate(
                (
                    coordinate_jacobian @ arrays["total_rate"],
                    inputs["memory_basis"].T @ arrays["total_rate"],
                    inputs["departure_basis"].T @ arrays["total_rate"],
                )
            )
            item.update(
                {
                    "candidate_index": index,
                    "direction_index": candidates[index]["direction_index"],
                    "amplitude_index": candidates[index]["amplitude_index"],
                    "component_bound": candidates[index]["component_bound"],
                    "sign": candidates[index]["sign"],
                    "state_rate_linear_relative_defect": _relative_error(
                        increment, linear
                    ),
                    "departure_rate_linear_relative_defect": _relative_error(
                        departure_increment, departure_linear
                    ),
                    "coordinate_Jacobian_rank": coordinate_metrics["rank"],
                    "coordinate_Jacobian_condition_number": coordinate_metrics[
                        "condition_number"
                    ],
                }
            )
            total_rates.append(arrays["total_rate"])
            free_rates.append(arrays["free_rate"])
            actions.append(arrays["reaction_action"])
            multipliers.append(arrays["multiplier"])
            online_rates.append(online_rate)
            departure_rates.append(departure_increment)
            linear_references.append(linear)
            departure_linear_references.append(departure_linear)
            evaluation_metrics.append(item)
            status = "accepted"
        except Exception as error:  # canonicalize the first fail-closed truth error
            failures.append(
                {
                    "candidate_index": index,
                    "reason": type(error).__name__,
                    "message": str(error),
                }
            )
            status = "failed"
        print(
            json.dumps(
                {
                    "candidate": index + 1,
                    "total": manifest.CANDIDATE_COUNT,
                    "direction": candidates[index]["direction_index"],
                    "component_bound": candidates[index]["component_bound"],
                    "sign": candidates[index]["sign"],
                    "status": status,
                    "elapsed_seconds": time.perf_counter() - began,
                }
            ),
            flush=True,
        )
        if failures:
            break
    total_rates_array = np.asarray(total_rates, dtype=float)
    departure_increments = np.asarray(departure_rates, dtype=float)
    departure_linear_array = np.asarray(departure_linear_references, dtype=float)
    radial = {}
    radial_arrays = {}
    if len(evaluation_metrics) == manifest.CANDIDATE_COUNT:
        radial, radial_arrays = _radial_analysis(
            candidates,
            inputs["coordinates"],
            departure_increments,
            departure_linear_array,
            inputs["chart"]["energy_directions"],
        )

    def maximum(name: str, default=math.inf) -> float:
        values = [item[name] for item in evaluation_metrics]
        return float(max(values)) if values else float(default)

    def minimum(name: str, default=-math.inf) -> float:
        values = [item[name] for item in evaluation_metrics]
        return float(min(values)) if values else float(default)

    smallest = [
        item for item in evaluation_metrics if item["amplitude_index"] == 0
    ]
    metrics = {
        "planned_nonbase_rate_evaluations": manifest.CANDIDATE_COUNT,
        "completed_nonbase_rate_evaluations": len(evaluation_metrics),
        "failed_rate_evaluations": len(failures),
        "failures": failures,
        "maximum_smallest_state_rate_linear_relative_defect": (
            max(item["state_rate_linear_relative_defect"] for item in smallest)
            if smallest
            else math.inf
        ),
        "maximum_smallest_departure_rate_linear_relative_defect": (
            max(item["departure_rate_linear_relative_defect"] for item in smallest)
            if smallest
            else math.inf
        ),
        "minimum_reconstruction_factor": minimum(
            "minimum_reconstruction_factor", math.inf
        ),
        "maximum_reconstruction_factor": maximum("maximum_reconstruction_factor"),
        "maximum_raw_Schur_condition_number": maximum(
            "raw_Schur_condition_number"
        ),
        "maximum_reaction_identity_defect": maximum("reaction_identity_defect"),
        "maximum_rate_tangency_relative_defect": maximum(
            "rate_tangency_relative_defect"
        ),
        "maximum_H_over_R": maximum("maximum_H_over_R"),
        "minimum_scattering_optical_depth": minimum(
            "minimum_scattering_optical_depth"
        ),
        "maximum_incoming_excision_characteristics": maximum(
            "incoming_excision_characteristics"
        ),
        "maximum_coordinate_Jacobian_condition_number": maximum(
            "coordinate_Jacobian_condition_number"
        ),
        "total_truth_wall_seconds": time.perf_counter() - began,
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "radial_nonlinearity": radial,
        "evaluations": evaluation_metrics,
    }
    arrays = {
        "candidate_primitive_states": inputs["states"][: len(evaluation_metrics)],
        "candidate_scaled_deltas": inputs["deltas"][: len(evaluation_metrics)],
        "candidate_departure_coordinates": inputs["coordinates"][
            : len(evaluation_metrics)
        ],
        "total_rates_per_second": total_rates_array,
        "free_rates_per_second": np.asarray(free_rates, dtype=float),
        "physical_reaction_actions_per_second": np.asarray(actions, dtype=float),
        "multiplier_coordinates_per_second": np.asarray(multipliers, dtype=float),
        "online_470_coordinate_rates_per_second": np.asarray(
            online_rates, dtype=float
        ),
        "departure_rate_increments_per_second": departure_increments,
        "linear_rate_references_per_second": np.asarray(
            linear_references, dtype=float
        ),
        "departure_linear_references_per_second": departure_linear_array,
        "base_fixed_Q_rate_per_second": inputs["base_rate"],
        **radial_arrays,
    }
    return metrics, arrays


def _gate_checks(metrics: dict, gates: dict) -> dict:
    return {
        "completed": metrics["completed_nonbase_rate_evaluations"]
        == gates["completed_nonbase_rate_evaluations_equal"],
        "failed": metrics["failed_rate_evaluations"]
        == gates["failed_rate_evaluations_equal"],
        "state_linear_limit": metrics[
            "maximum_smallest_state_rate_linear_relative_defect"
        ]
        <= gates["maximum_smallest_state_rate_linear_relative_defect"],
        "departure_linear_limit": metrics[
            "maximum_smallest_departure_rate_linear_relative_defect"
        ]
        <= gates["maximum_smallest_departure_rate_linear_relative_defect"],
        "reconstruction_minimum": metrics["minimum_reconstruction_factor"]
        >= gates["minimum_reconstruction_factor"],
        "reconstruction_maximum": metrics["maximum_reconstruction_factor"]
        <= gates["maximum_reconstruction_factor"],
        "Schur_condition": metrics["maximum_raw_Schur_condition_number"]
        <= gates["maximum_raw_Schur_condition_number"],
        "reaction_identity": metrics["maximum_reaction_identity_defect"]
        <= gates["maximum_reaction_identity_defect"],
        "rate_tangency": metrics["maximum_rate_tangency_relative_defect"]
        <= gates["maximum_rate_tangency_relative_defect"],
        "height": metrics["maximum_H_over_R"] <= gates["maximum_H_over_R"],
        "optical_depth": metrics["minimum_scattering_optical_depth"]
        >= gates["minimum_scattering_optical_depth"],
        "incoming_excision": metrics[
            "maximum_incoming_excision_characteristics"
        ]
        == gates["maximum_incoming_excision_characteristics_equal"],
        "generator_budget": metrics["new_complete_generator_assemblies"]
        == gates["new_complete_generator_assemblies_equal"],
        "root_budget": metrics["new_nonlinear_roots"]
        == gates["new_nonlinear_roots_equal"],
        "propagation_budget": metrics["propagated_states"]
        == gates["propagated_states_equal"],
    }


def _classify(evaluator_passed: bool, median_signal: float) -> tuple[str, str | None]:
    if not evaluator_passed:
        return FAIL_CLASSIFICATION, None
    if median_signal >= manifest.NONLINEAR_SIGNAL_THRESHOLD:
        return (
            NONLINEAR_CLASSIFICATION,
            "definitions_only_mixed_direction_adaptive_28D_database_manifest",
        )
    return (
        UNRESOLVED_CLASSIFICATION,
        "definitions_only_expanded_safe_departure_chart_manifest",
    )


def _update_catalog(summary: dict) -> None:
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha(path),
                    "scientific_status": (
                        "CERTIFIED" if summary["passed"] else "REJECTED"
                    ),
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


def _run() -> dict:
    frozen = _validate_manifest(require_clean=True)
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("guarded departure-rate screen is already canonicalized")
    metrics, arrays = _evaluate()
    checks = _gate_checks(
        metrics, frozen["contract"]["binding_evaluator_gates"]
    )
    evaluator_passed = all(checks.values())
    median_signal = metrics["radial_nonlinearity"].get(
        "median_largest_departure_nonlinear_relative_defect", -math.inf
    )
    classification, authorized_next = _classify(evaluator_passed, median_signal)
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_json(CANONICAL_DIRECTORY / "metrics.json", {"checks": checks, **metrics})
    np.savez_compressed(CANONICAL_DIRECTORY / "departure_rate_screen.npz", **arrays)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": evaluator_passed,
        "completed_nonbase_rate_evaluations": metrics[
            "completed_nonbase_rate_evaluations"
        ],
        "failed_rate_evaluations": metrics["failed_rate_evaluations"],
        "maximum_smallest_state_rate_linear_relative_defect": metrics[
            "maximum_smallest_state_rate_linear_relative_defect"
        ],
        "maximum_smallest_departure_rate_linear_relative_defect": metrics[
            "maximum_smallest_departure_rate_linear_relative_defect"
        ],
        "median_largest_departure_nonlinear_relative_defect": median_signal,
        "nonlinear_signal_resolved": bool(
            evaluator_passed and median_signal >= manifest.NONLINEAR_SIGNAL_THRESHOLD
        ),
        "full_28D_closure_identified": False,
        "heldout_state_validated": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": authorized_next,
    }
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    _write_json(
        CANONICAL_DIRECTORY / "parent_lock.json",
        {
            "manifest_commit": MANIFEST_COMMIT,
            "manifest_parent": MANIFEST_PARENT,
            "manifest_tree": MANIFEST_TREE,
            "manifest_hashes": _checksums(manifest.ARTIFACT_DIRECTORY),
            "chart_hashes": _checksums(manifest.parent.CANONICAL_DIRECTORY),
        },
    )
    source_files = (THIS_RUNNER, THIS_TEST, manifest.THIS_RUNNER, manifest.THIS_TEST)
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "CERTIFIED" if evaluator_passed else "REJECTED",
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "source_hashes": {
                relative: _sha(ROOT / relative) for relative in source_files
            },
            "python": sys.version,
            "platform": platform.platform(),
            "thread_environment": manifest.parent.coordinate_tools.THREAD_ENVIRONMENT,
        },
    )
    names = tuple(sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()))
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    radial = metrics["radial_nonlinearity"]
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Guarded departure-rate screen WP10c9d6c7c3b5c4f25ba",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                "## Result",
                "",
                f"Completed `{metrics['completed_nonbase_rate_evaluations']}` of `{manifest.CANDIDATE_COUNT}` exact nonbase rate evaluations with `{metrics['failed_rate_evaluations']}` failures.",
                "",
                f"The maximum smallest-amplitude full-state linear defect is `{metrics['maximum_smallest_state_rate_linear_relative_defect']:.6e}` and the 28D departure defect is `{metrics['maximum_smallest_departure_rate_linear_relative_defect']:.6e}`.",
                "",
                f"The median largest-amplitude nonlinear departure fraction is `{median_signal:.6e}`. Largest-amplitude radial growth is nonpositive in `{radial.get('nonpositive_largest_radial_growth_count', 0)}` of 8 directions; this is diagnostic rather than binding.",
                "",
                f"Authorized next artifact: `{authorized_next}`. No full 28D closure, held-out validation, online trajectory, or predictive cycle is claimed.",
                "",
            )
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("pass --run")
    print(json.dumps(_plain(_run()), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
