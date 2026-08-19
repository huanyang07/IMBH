#!/usr/bin/env python3
"""Execute the exact geometric 28D departure-chart preflight."""

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

import run_causal_inner_exact_geometric_departure_chart_manifest_wp10c9d6c7c3b5c4f25ax as manifest  # noqa: E402
import run_causal_inner_first_conditional_branch_seed_preflight_wp10c9d6c7c3b5c4f25aq as coordinate_tools  # noqa: E402
from imri_qpe.constants import C  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_fixed_q import (  # noqa: E402
    causal_five_field_exterior_q3,
)
from run_causal_inner_monolithic_bdf_base_preflight_wp10c9d6c7c3b1a import (  # noqa: E402
    _state_audit,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25ay"
MANIFEST_COMMIT = "359690473ccf956d5f369331c3d71e09f61f80a7"
MANIFEST_PARENT = "6db52b4b1689ccde639eebb70a96b440c1d611d7"
MANIFEST_TREE = "cee3af61e3f2cf99e186db59a4e83418f70b1051"

PASS_CLASSIFICATION = (
    "exact_geometric_28D_departure_chart_preflight_passed_"
    "guarded_nonlinear_rate_database_manifest_authorized"
)
FAIL_CLASSIFICATION = (
    "exact_geometric_28D_departure_chart_preflight_failed_"
    "nonlinear_rate_database_blocked"
)

ARTIFACT = (
    "causal_inner_exact_geometric_departure_chart_preflight_"
    "wp10c9d6c7c3b5c4f25ay"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_exact_geometric_departure_chart_preflight_"
    "wp10c9d6c7c3b5c4f25ay.py"
)
THIS_TEST = (
    "tests/test_causal_inner_exact_geometric_departure_chart_preflight_"
    "wp10c9d6c7c3b5c4f25ay.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_EXACT_GEOMETRIC_DEPARTURE_"
    "CHART_PREFLIGHT_WP10C9D6C7C3B5C4F25AY_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


class ChartRetractionFailure(RuntimeError):
    """Carry fail-closed diagnostics for one finite-amplitude candidate."""

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
        raise RuntimeError("geometric chart manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("geometric chart manifest lineage changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("geometric chart manifest tree changed")
    hashes = _checksums(manifest.ARTIFACT_DIRECTORY)
    summary = _read(manifest.ARTIFACT_DIRECTORY / "summary.json")
    contract = _read(manifest.ARTIFACT_DIRECTORY / "contract.json")
    provenance = _read(manifest.ARTIFACT_DIRECTORY / "provenance.json")
    lock = _read(manifest.ARTIFACT_DIRECTORY / "parent_lock.json")
    if (
        not summary["passed"]
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["planned_candidate_count"] != manifest.PLANNED_CANDIDATES
        or summary["planned_nonbase_continuous_rate_evaluations"] != 0
        or contract["exact_geometric_retraction"]["rate_reaction_lift_used"]
    ):
        raise RuntimeError("geometric chart execution authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"manifest source changed: {relative}")
    for name, path in (
        ("online_470_geometry", manifest.GEOMETRY_PATH),
        ("exact_coordinate_preflight", manifest.PREFLIGHT_PATH),
        ("exact_departure_fiber", manifest.FIBER_PATH),
    ):
        if _sha(path) != lock["decisive_input_hashes"][name]:
            raise RuntimeError(f"decisive chart input changed: {path}")
    _checksums(manifest.parent.CANONICAL_DIRECTORY)
    _checksums(manifest.failed_screen.CANONICAL_DIRECTORY)
    if require_clean and not _tracked_tree_clean():
        raise RuntimeError("geometric chart preflight requires a clean tracked tree")
    for name, expected in coordinate_tools.THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    return {"summary": summary, "contract": contract, "hashes": hashes}


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(left) - np.asarray(right))
        / max(
            float(np.linalg.norm(left)),
            float(np.linalg.norm(right)),
            np.finfo(float).tiny,
        )
    )


def _coordinate_value_with_factors(
    state: np.ndarray, components: dict
) -> tuple[np.ndarray, np.ndarray]:
    spatial_nodes = components.get("spatial_nodes")
    if spatial_nodes is None:
        spatial_nodes = coordinate_tools._spatial_nodes(components["context"])
    integrated, factors, _node_values = coordinate_tools._integrated_mapped_storage(
        components["context"], state, spatial_nodes
    )
    mapped = np.zeros(manifest.PHYSICAL_COORDINATE_DIMENSION, dtype=float)
    for coarse_cell, (start, stop) in enumerate(components["groups"]):
        block = np.sum(integrated[start:stop], axis=0)
        mapped[
            5 * coarse_cell : 5 * (coarse_cell + 1)
        ] = block
    mapped[:160] /= components["mapped_row_scales"]
    scaled_delta = (
        (np.asarray(state) - components["state"]) / components["columns"]
    ).ravel()
    stable = components["stable_dual"] @ scaled_delta
    return np.concatenate((mapped[:160], stable)), np.asarray(factors, dtype=float)


def _coordinate_jacobian(
    state: np.ndarray, components: dict
) -> tuple[np.ndarray, dict]:
    (
        node_weights,
        node_cells,
        node_radii,
        node_measures,
        reconstruction_defect,
        partition_defect,
    ) = coordinate_tools._node_reconstruction_weights(components["context"], state)
    mapped, _height = coordinate_tools._descriptor_matrices(
        components["context"],
        state,
        components["columns"],
        components["rows"],
        node_weights,
        node_cells,
        node_radii,
        node_measures,
    )
    physical_mapped = C * components["rows"].ravel()[:, None] * mapped
    unscaled = np.zeros((160, state.size), dtype=float)
    for coarse_cell, (start, stop) in enumerate(components["groups"]):
        for field in range(5):
            target = 5 * coarse_cell + field
            source_rows = 5 * np.arange(start, stop) + field
            unscaled[target] = np.sum(physical_mapped[source_rows], axis=0)
    jacobian = np.vstack(
        (
            unscaled / components["mapped_row_scales"][:, None],
            components["stable_dual"],
        )
    )
    singular = np.linalg.svd(jacobian, compute_uv=False)
    return jacobian, {
        "rank": int(np.linalg.matrix_rank(jacobian)),
        "condition_number": float(singular[0] / singular[-1]),
        "reconstruction_relative_defect": float(reconstruction_defect),
        "partition_defect": float(partition_defect),
    }


def _minimum_norm_coordinate_correction(
    jacobian: np.ndarray, error: np.ndarray
) -> np.ndarray:
    return jacobian.T @ np.linalg.solve(jacobian @ jacobian.T, error)


def _departure_family() -> tuple[dict, dict[str, np.ndarray]]:
    with np.load(manifest.GEOMETRY_PATH, allow_pickle=False) as source:
        geometry = {name: np.asarray(source[name], dtype=float) for name in source.files}
    with np.load(manifest.FIBER_PATH, allow_pickle=False) as source:
        right = np.asarray(source["primary_right_basis"], dtype=float)
        old_operator = np.asarray(source["primary_unstable_operator"], dtype=float)
    departure = geometry["departure_coordinate_basis"]
    transform = departure.T @ right
    operator = transform @ old_operator @ np.linalg.inv(transform)
    symmetric = 0.5 * (operator + operator.T)
    values, vectors = np.linalg.eigh(symmetric)
    directions = np.asarray(
        vectors[:, -manifest.ENERGY_DIRECTION_COUNT :][:, ::-1], dtype=float
    )
    energy = np.asarray(
        values[-manifest.ENERGY_DIRECTION_COUNT :][::-1], dtype=float
    )
    for column in range(directions.shape[1]):
        pivot = int(np.argmax(np.abs(directions[:, column])))
        if directions[pivot, column] < 0.0:
            directions[:, column] *= -1.0
    with np.load(manifest.PREFLIGHT_PATH, allow_pickle=False) as source:
        physical = np.asarray(source["coordinate_jacobian"], dtype=float)
    metrics = {
        "departure_basis_shape": list(departure.shape),
        "departure_basis_orthogonality_defect": float(
            np.max(np.abs(departure.T @ departure - np.eye(28)))
        ),
        "departure_base_physical_tangency_defect": float(
            np.max(np.abs(physical @ departure))
        ),
        "old_to_new_coordinate_transform_condition_number": float(
            np.linalg.cond(transform)
        ),
        "minimum_selected_energy_growth_per_second": float(np.min(energy)),
        "maximum_selected_energy_growth_per_second": float(np.max(energy)),
    }
    return metrics, {
        "departure_basis": departure,
        "old_to_new_coordinate_transform": transform,
        "departure_operator_per_second": operator,
        "energy_directions": directions,
        "selected_energy_growth_per_second": energy,
        "stable_memory_basis": geometry["stable_memory_coordinate_basis"],
    }


def _newton_retract(
    components: dict,
    seed: np.ndarray,
    target: np.ndarray,
    *,
    maximum_component_bound: float,
    gates: dict,
    retraction_contract: dict,
) -> tuple[np.ndarray, dict]:
    delta = np.asarray(seed, dtype=float).ravel().copy()
    line_factors = tuple(float(value) for value in retraction_contract["line_factors"])
    residual_history = []
    accepted_line_factors = []
    condition_numbers = []
    total_correction_norm = 0.0
    last_factors = None
    for iteration in range(retraction_contract["maximum_Newton_iterations"] + 1):
        state = components["state"] + (
            components["columns"].ravel() * delta
        ).reshape(components["state"].shape)
        value, last_factors = _coordinate_value_with_factors(state, components)
        error = value - target
        residual = float(np.max(np.abs(error)))
        residual_history.append(residual)
        if residual <= gates["maximum_coordinate_residual_infinity"]:
            return delta, {
                "coordinate_residual_infinity": residual,
                "Newton_corrections": iteration,
                "accepted_line_factors": accepted_line_factors,
                "residual_history": residual_history,
                "maximum_coordinate_Jacobian_condition_number": (
                    max(condition_numbers) if condition_numbers else 0.0
                ),
                "total_coordinate_correction_norm": total_correction_norm,
                "minimum_coordinate_reconstruction_factor": float(
                    np.min(last_factors)
                ),
                "maximum_coordinate_reconstruction_factor": float(
                    np.max(last_factors)
                ),
            }
        if iteration == retraction_contract["maximum_Newton_iterations"]:
            break
        jacobian, jacobian_metrics = _coordinate_jacobian(state, components)
        condition_numbers.append(jacobian_metrics["condition_number"])
        if (
            jacobian_metrics["rank"] != manifest.PHYSICAL_COORDINATE_DIMENSION
            or jacobian_metrics["condition_number"]
            > gates["maximum_coordinate_Jacobian_condition_number"]
        ):
            raise ChartRetractionFailure(
                "coordinate Jacobian failed",
                {"jacobian": jacobian_metrics, "residual_history": residual_history},
            )
        correction = _minimum_norm_coordinate_correction(jacobian, error)
        accepted = False
        for factor in line_factors:
            proposed = delta - factor * correction
            if float(np.max(np.abs(proposed))) > 1.25 * maximum_component_bound:
                continue
            proposed_state = components["state"] + (
                components["columns"].ravel() * proposed
            ).reshape(components["state"].shape)
            proposed_value, _factors = _coordinate_value_with_factors(
                proposed_state, components
            )
            proposed_residual = float(np.max(np.abs(proposed_value - target)))
            if proposed_residual < residual:
                delta = proposed
                accepted_line_factors.append(factor)
                total_correction_norm += factor * float(np.linalg.norm(correction))
                accepted = True
                break
        if not accepted:
            raise ChartRetractionFailure(
                "coordinate Newton line search failed",
                {
                    "coordinate_residual_infinity": residual,
                    "residual_history": residual_history,
                    "correction_norm": float(np.linalg.norm(correction)),
                },
            )
    raise ChartRetractionFailure(
        "coordinate Newton iteration budget exhausted",
        {"residual_history": residual_history},
    )


def _retract_candidate(
    components: dict,
    departure_basis: np.ndarray,
    stable_memory_basis: np.ndarray,
    direction: np.ndarray,
    sign: int,
    component_bound: float,
    contract: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    gates = contract["binding_preflight_gates"]
    retraction_contract = contract["exact_geometric_retraction"]
    target = components["coordinate_target"]
    lifted = departure_basis @ direction
    radius = 0.99 * component_bound / float(np.max(np.abs(lifted)))
    rescalings = 0
    began = time.perf_counter()
    while True:
        seed = float(sign) * radius * lifted
        delta, retraction = _newton_retract(
            components,
            seed,
            target,
            maximum_component_bound=component_bound,
            gates=gates,
            retraction_contract=retraction_contract,
        )
        final_component = float(np.max(np.abs(delta)))
        if final_component <= component_bound * (1.0 + 1.0e-12):
            break
        rescalings += 1
        if rescalings > retraction_contract["maximum_radius_rescalings"]:
            raise ChartRetractionFailure(
                "component-bound radius rescaling exhausted",
                {
                    "component_bound": component_bound,
                    "final_component": final_component,
                    "radius": radius,
                },
            )
        radius *= 0.99 * component_bound / final_component
    state = components["state"] + (
        components["columns"].ravel() * delta
    ).reshape(components["state"].shape)
    actual = departure_basis.T @ delta
    signed_direction = float(sign) * direction
    axial = float(signed_direction @ actual)
    actual_norm = float(np.linalg.norm(actual))
    transverse = actual - axial * signed_direction
    face = 36 * int(components["data"]["layout"].refinement_ratio)
    q0 = components["base_q3"]
    q0_factors = components["base_q3_factors"]
    q, q_factors = causal_five_field_exterior_q3(
        components["context"], state, exterior_face_index=face
    )
    physical = _state_audit(components["context"], state)
    minimum_factor = min(
        retraction["minimum_coordinate_reconstruction_factor"],
        float(np.min(q0_factors)),
        float(np.min(q_factors)),
        physical["minimum_reconstruction_factor"],
    )
    maximum_factor = max(
        retraction["maximum_coordinate_reconstruction_factor"],
        float(np.max(q0_factors)),
        float(np.max(q_factors)),
    )
    metrics = {
        **retraction,
        "component_bound": component_bound,
        "sign": int(sign),
        "initial_departure_radius": radius,
        "final_scaled_component": float(np.max(np.abs(delta))),
        "radius_rescalings": rescalings,
        "departure_coordinate_norm": actual_norm,
        "departure_direction_alignment_cosine": axial
        / max(actual_norm, np.finfo(float).tiny),
        "departure_transverse_fraction": float(np.linalg.norm(transverse))
        / max(actual_norm, np.finfo(float).tiny),
        "stable_memory_coordinate_leakage_norm": float(
            np.linalg.norm(stable_memory_basis.T @ delta)
        ),
        "normalized_Q3_defect": _relative(q, q0),
        "minimum_reconstruction_factor": minimum_factor,
        "maximum_reconstruction_factor": maximum_factor,
        "maximum_H_over_R": physical["maximum_h_over_r"],
        "minimum_scattering_optical_depth": physical[
            "minimum_scattering_optical_depth"
        ],
        "wall_seconds": time.perf_counter() - began,
    }
    arrays = {
        "primitive_state": state,
        "scaled_delta": delta,
        "departure_coordinates": actual,
    }
    return metrics, arrays


def _execute_candidates() -> tuple[dict, dict[str, np.ndarray]]:
    components = coordinate_tools._coordinate_components()
    components["spatial_nodes"] = coordinate_tools._spatial_nodes(
        components["context"]
    )
    components["coordinate_target"], _base_coordinate_factors = (
        _coordinate_value_with_factors(components["state"], components)
    )
    face = 36 * int(components["data"]["layout"].refinement_ratio)
    components["base_q3"], components["base_q3_factors"] = (
        causal_five_field_exterior_q3(
            components["context"],
            components["state"],
            exterior_face_index=face,
        )
    )
    family_metrics, family = _departure_family()
    contract = manifest._contract()
    candidate_metrics = []
    states = []
    deltas = []
    coordinates = []
    failed = []
    began = time.perf_counter()
    total = manifest.PLANNED_CANDIDATES
    for direction_index in range(manifest.ENERGY_DIRECTION_COUNT):
        direction = family["energy_directions"][:, direction_index]
        for amplitude_index, component_bound in enumerate(
            manifest.MAXIMUM_COMPONENT_BOUNDS
        ):
            pair = []
            for sign in manifest.SIGNS:
                sequence = len(candidate_metrics) + len(failed) + 1
                try:
                    metrics, arrays = _retract_candidate(
                        components,
                        family["departure_basis"],
                        family["stable_memory_basis"],
                        direction,
                        sign,
                        float(component_bound),
                        contract,
                    )
                    metrics.update(
                        {
                            "candidate_index": sequence - 1,
                            "direction_index": direction_index,
                            "amplitude_index": amplitude_index,
                        }
                    )
                    candidate_metrics.append(metrics)
                    states.append(arrays["primitive_state"])
                    deltas.append(arrays["scaled_delta"])
                    coordinates.append(arrays["departure_coordinates"])
                    pair.append(arrays["departure_coordinates"])
                    status = "accepted"
                except ChartRetractionFailure as error:
                    failed.append(
                        {
                            "candidate_index": sequence - 1,
                            "direction_index": direction_index,
                            "amplitude_index": amplitude_index,
                            "component_bound": float(component_bound),
                            "sign": int(sign),
                            "reason": str(error),
                            "diagnostics": error.diagnostics,
                        }
                    )
                    status = "failed"
                print(
                    json.dumps(
                        {
                            "candidate": sequence,
                            "total": total,
                            "direction": direction_index,
                            "component_bound": component_bound,
                            "sign": sign,
                            "status": status,
                            "elapsed_seconds": time.perf_counter() - began,
                        }
                    ),
                    flush=True,
                )
                if failed:
                    break
            if failed:
                break
            if len(pair) == 2:
                denominator = max(
                    float(np.linalg.norm(pair[0])) + float(np.linalg.norm(pair[1])),
                    np.finfo(float).tiny,
                )
                candidate_metrics[-1]["pair_coordinate_odd_symmetry_defect"] = float(
                    np.linalg.norm(pair[0] + pair[1]) / denominator
                )
                candidate_metrics[-2]["pair_coordinate_odd_symmetry_defect"] = (
                    candidate_metrics[-1]["pair_coordinate_odd_symmetry_defect"]
                )
        if failed:
            break
    def maximum(name: str, default=math.inf) -> float:
        values = [item.get(name, default) for item in candidate_metrics]
        return float(max(values)) if values else float(default)

    def minimum(name: str, default=-math.inf) -> float:
        values = [item.get(name, default) for item in candidate_metrics]
        return float(min(values)) if values else float(default)

    metrics = {
        "departure_family": family_metrics,
        "planned_candidate_count": total,
        "completed_candidate_count": len(candidate_metrics),
        "failed_candidate_count": len(failed),
        "failed_candidates": failed,
        "maximum_coordinate_residual_infinity": maximum(
            "coordinate_residual_infinity"
        ),
        "maximum_normalized_Q3_defect": maximum("normalized_Q3_defect"),
        "maximum_final_scaled_component": maximum("final_scaled_component"),
        "minimum_reconstruction_factor": minimum(
            "minimum_reconstruction_factor", math.inf
        ),
        "maximum_reconstruction_factor": maximum("maximum_reconstruction_factor"),
        "maximum_coordinate_Jacobian_condition_number": maximum(
            "maximum_coordinate_Jacobian_condition_number"
        ),
        "minimum_departure_direction_alignment_cosine": minimum(
            "departure_direction_alignment_cosine"
        ),
        "maximum_departure_transverse_fraction": maximum(
            "departure_transverse_fraction"
        ),
        "maximum_coordinate_odd_symmetry_defect": maximum(
            "pair_coordinate_odd_symmetry_defect"
        ),
        "maximum_stable_memory_coordinate_leakage_norm": maximum(
            "stable_memory_coordinate_leakage_norm"
        ),
        "maximum_H_over_R": maximum("maximum_H_over_R"),
        "minimum_scattering_optical_depth": minimum(
            "minimum_scattering_optical_depth"
        ),
        "maximum_Newton_corrections": maximum("Newton_corrections"),
        "maximum_radius_rescalings": maximum("radius_rescalings"),
        "total_wall_seconds": time.perf_counter() - began,
        "nonbase_continuous_rate_evaluations": 0,
        "new_full_generator_assemblies": 0,
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "candidates": candidate_metrics,
    }
    arrays = {
        **family,
        "candidate_primitive_states": np.asarray(states, dtype=float),
        "candidate_scaled_deltas": np.asarray(deltas, dtype=float),
        "candidate_departure_coordinates": np.asarray(coordinates, dtype=float),
    }
    return metrics, arrays


def _gate_checks(metrics: dict, gates: dict) -> dict:
    return {
        "candidate_count": metrics["completed_candidate_count"]
        == gates["completed_candidate_count_equal"],
        "failure_count": metrics["failed_candidate_count"]
        == gates["failed_candidate_count_equal"],
        "coordinate_closure": metrics["maximum_coordinate_residual_infinity"]
        <= gates["maximum_coordinate_residual_infinity"],
        "Q3_closure": metrics["maximum_normalized_Q3_defect"]
        <= gates["maximum_normalized_Q3_defect"],
        "component_trust": metrics["maximum_final_scaled_component"]
        <= gates["maximum_final_scaled_component"] * (1.0 + 1.0e-12),
        "reconstruction_minimum": metrics["minimum_reconstruction_factor"]
        >= gates["minimum_reconstruction_factor"],
        "reconstruction_maximum": metrics["maximum_reconstruction_factor"]
        <= gates["maximum_reconstruction_factor"],
        "coordinate_condition": metrics[
            "maximum_coordinate_Jacobian_condition_number"
        ]
        <= gates["maximum_coordinate_Jacobian_condition_number"],
        "direction_alignment": metrics[
            "minimum_departure_direction_alignment_cosine"
        ]
        >= gates["minimum_departure_direction_alignment_cosine"],
        "transverse_distortion": metrics["maximum_departure_transverse_fraction"]
        <= gates["maximum_departure_transverse_fraction"],
        "odd_symmetry": metrics["maximum_coordinate_odd_symmetry_defect"]
        <= gates["maximum_coordinate_odd_symmetry_defect"],
        "height_guard": metrics["maximum_H_over_R"] <= gates["maximum_H_over_R"],
        "optical_guard": metrics["minimum_scattering_optical_depth"]
        >= gates["minimum_scattering_optical_depth"],
        "rate_budget": metrics["nonbase_continuous_rate_evaluations"]
        == gates["nonbase_continuous_rate_evaluations_equal"],
        "generator_budget": metrics["new_full_generator_assemblies"]
        == gates["new_full_generator_assemblies_equal"],
        "root_budget": metrics["new_nonlinear_roots"]
        == gates["new_nonlinear_roots_equal"],
        "propagation_budget": metrics["propagated_states"]
        == gates["propagated_states_equal"],
    }


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
        raise RuntimeError("geometric chart preflight is already canonicalized")
    metrics, arrays = _execute_candidates()
    gates = frozen["contract"]["binding_preflight_gates"]
    checks = _gate_checks(metrics, gates)
    passed = all(checks.values())
    classification = PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION
    authorized_next = (
        "definitions_only_guarded_nonlinear_28D_rate_database_manifest"
        if passed
        else None
    )
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_json(CANONICAL_DIRECTORY / "metrics.json", {"checks": checks, **metrics})
    np.savez_compressed(
        CANONICAL_DIRECTORY / "geometric_departure_chart.npz", **arrays
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "completed_candidate_count": metrics["completed_candidate_count"],
        "failed_candidate_count": metrics["failed_candidate_count"],
        "maximum_coordinate_residual_infinity": metrics[
            "maximum_coordinate_residual_infinity"
        ],
        "maximum_normalized_Q3_defect": metrics["maximum_normalized_Q3_defect"],
        "maximum_final_scaled_component": metrics["maximum_final_scaled_component"],
        "nonbase_continuous_rate_evaluations": 0,
        "nonlinear_rate_database_executed": False,
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
            "architecture_hashes": _checksums(manifest.parent.CANONICAL_DIRECTORY),
        },
    )
    source_files = (THIS_RUNNER, THIS_TEST, manifest.THIS_RUNNER, manifest.THIS_TEST)
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "CERTIFIED" if passed else "REJECTED",
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
            "thread_environment": coordinate_tools.THREAD_ENVIRONMENT,
        },
    )
    names = tuple(sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()))
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Exact geometric departure-chart preflight WP10c9d6c7c3b5c4f25ay",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                "## Result",
                "",
                f"Completed `{metrics['completed_candidate_count']}` of `{manifest.PLANNED_CANDIDATES}` candidates with `{metrics['failed_candidate_count']}` failures.",
                "",
                f"The maximum exact C_phys closure defect is `{metrics['maximum_coordinate_residual_infinity']:.6e}` and the maximum normalized Q3 defect is `{metrics['maximum_normalized_Q3_defect']:.6e}`. The maximum final scaled primitive component is `{metrics['maximum_final_scaled_component']:.6e}`.",
                "",
                "No nonbase continuous-rate evaluation, full-generator assembly, nonlinear root, or propagated state was used. Passing this package authorizes only a prospective guarded nonlinear departure-rate database manifest.",
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
