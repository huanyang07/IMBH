"""Test conservative radial-shell coarse closure for WP10c8h."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import run_causal_ledger_equation_free_preflight_wp10c8g as wp10c8g
import run_causal_mixed_mode_reduction_audit_wp10c8d as wp10c8d
import run_causal_region_selective_closure_audit_wp10c8c as wp10c8c
import run_causal_stress_time_audit_wp10c8b as wp10c8b
from imri_qpe.constants import C
from imri_qpe.layer3_minidisk_1d import (
    CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1,
    causal_descriptor_explicit_matrices,
    causal_five_field_reduced_descriptor_matrices,
    causal_projective_ab2_prediction,
    causal_projective_euler_prediction,
    causal_weighted_constraint_null_projection,
    unpack_causal_five_field_state,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "d89663531dbbce97be01d583e364bc3585448e76"
WP10C8G_OUTPUT = (
    ROOT
    / "outputs/tables/causal_ledger_equation_free_preflight_wp10c8g.json"
)
CACHE_DIRECTORY = (
    ROOT / "outputs/checkpoints/causal_five_field_wp10c8h"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/tables/causal_shell_closure_preflight_wp10c8h.json"
)
DEFAULT_ARRAYS = (
    ROOT / "outputs/tables/causal_shell_closure_preflight_wp10c8h_arrays.npz"
)
RESOLUTIONS = wp10c8g.RESOLUTIONS
TRAJECTORY_MODES = wp10c8g.TRAJECTORY_MODES
CHECKPOINTS = wp10c8g.CHECKPOINTS
TANGENT_ANCHOR = wp10c8g.TANGENT_ANCHOR
PROJECTION_WINDOW_SECONDS = wp10c8g.PROJECTION_WINDOW_SECONDS
MAXIMUM_GATE_FRACTION = wp10c8g.MAXIMUM_GATE_FRACTION
MAXIMUM_CONSTRAINT_DEFECT = wp10c8g.MAXIMUM_CONSTRAINT_DEFECT
MAXIMUM_DIRECTION_CORRECTION_FRACTION = (
    wp10c8g.MAXIMUM_DIRECTION_CORRECTION_FRACTION
)
MINIMUM_ELIGIBLE_DIRECTIONS = wp10c8g.MINIMUM_ELIGIBLE_DIRECTIONS
LEDGER_COMPONENTS = wp10c8g.LEDGER_COMPONENTS
SHELL_LAYOUT_TARGETS_RG = {
    "five_shell": (6.0, 60.0, 200.0, 280.0),
    "eight_shell": (3.0, 6.0, 20.0, 60.0, 130.0, 200.0, 280.0),
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-shell-operators", action="store_true")
    parser.add_argument(
        "--operators-only",
        action="store_true",
        help="Build or validate the N64/N128 shell operator caches.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate authorization and all selected checkpoints.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--arrays", type=Path, default=DEFAULT_ARRAYS)
    return parser.parse_args()


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _plain(value):
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(name): _plain(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _plain(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _validate_authorization() -> tuple[dict, str]:
    if not WP10C8G_OUTPUT.exists():
        raise RuntimeError("WP10c8h requires canonical WP10c8g evidence")
    evidence = json.loads(WP10C8G_OUTPUT.read_text(encoding="utf-8"))
    arrays = ROOT / str(evidence.get("artifacts", {}).get("arrays_path", ""))
    if not (
        evidence.get("work_package") == "WP10c8g"
        and evidence.get("decision")
        == "wp10c8g_global_equation_free_closure_not_identifiable"
        and evidence.get("next_authorization")
        == "five_shell_conservative_closure_preflight"
        and not evidence.get("gates", {}).get(
            "nonlinear_microbursts_authorized",
            True,
        )
        and evidence.get("gates", {}).get(
            "five_shell_preflight_authorized",
            False,
        )
        and arrays.exists()
        and _sha256(arrays)
        == evidence.get("artifacts", {}).get("arrays_sha256")
    ):
        raise RuntimeError("WP10c8g did not authorize the shell preflight")
    return evidence, _sha256(WP10C8G_OUTPUT)


def _common_shell_edges(initial: dict) -> dict[str, np.ndarray]:
    coarse = initial[64]["context"].grid
    fine = initial[128]["context"].grid
    coarse_rg = coarse.gravitational_radius
    fine_rg = fine.gravitational_radius
    coarse_edges = np.asarray(coarse.edges, dtype=float) / coarse_rg
    fine_edges = np.asarray(fine.edges, dtype=float) / fine_rg
    layouts = {}
    for name, targets in SHELL_LAYOUT_TARGETS_RG.items():
        indices = [
            0,
            *(
                int(np.argmin(np.abs(coarse_edges - target)))
                for target in targets
            ),
            coarse_edges.size - 1,
        ]
        if len(set(indices)) != len(indices) or indices != sorted(indices):
            raise RuntimeError(f"{name} shell targets do not define cells")
        selected = coarse_edges[np.asarray(indices)]
        for edge in selected:
            if not np.any(np.isclose(fine_edges, edge, rtol=0.0, atol=1.0e-12)):
                raise RuntimeError(f"{name} shell edge is not mesh coincident")
        layouts[name] = selected
    return layouts


def _shell_masks(radius_rg: np.ndarray, edges_rg: np.ndarray) -> list[np.ndarray]:
    masks = []
    for index, (left, right) in enumerate(
        zip(edges_rg[:-1], edges_rg[1:], strict=True)
    ):
        if index + 1 == edges_rg.size - 1:
            mask = (radius_rg >= left) & (radius_rg <= right)
        else:
            mask = (radius_rg >= left) & (radius_rg < right)
        if not np.any(mask):
            raise RuntimeError("a declared coarse shell is empty")
        masks.append(mask)
    return masks


def _shell_operator_cache_path(n_cells: int) -> Path:
    return CACHE_DIRECTORY / f"N{n_cells:03d}_t_0p125_shell_operators.npz"


def _build_shell_operator_cache(
    initial: dict,
    vector: np.ndarray,
    layouts: dict[str, np.ndarray],
) -> tuple[dict, dict]:
    context = initial["context"]
    n_cells = initial["state"].n_cells
    started = time.perf_counter()
    reduced = causal_five_field_reduced_descriptor_matrices(
        context,
        vector,
    )
    dynamic, _inputs, explicit_defect = causal_descriptor_explicit_matrices(
        reduced["descriptor_reduced_scaled_matrix"],
        reduced["stationary_reduced_scaled_jacobian"],
        np.zeros((5 * n_cells, 0), dtype=float),
    )
    outputs = wp10c8d._output_operators(initial, vector, reduced)
    state = unpack_causal_five_field_state(vector, n_cells)
    response_scaled = np.asarray(
        reduced["algebraic_response_scaled"],
        dtype=float,
    )
    algebraic_scale = np.asarray(
        reduced["algebraic_column_scales"],
        dtype=float,
    )
    response_physical = algebraic_scale[:, None] * response_scaled
    conserved_response = response_physical[: 5 * n_cells].reshape(
        n_cells,
        5,
        5 * n_cells,
    )
    face_response = response_physical[5 * n_cells :].reshape(
        n_cells + 1,
        5,
        5 * n_cells,
    )
    measures = np.asarray(context.grid.cell_measures, dtype=float)
    radius_rg = (
        np.asarray(context.grid.centers, dtype=float)
        / context.grid.gravitational_radius
    )
    ledger_gate = CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1[
        "maximum_integrated_conserved_relative"
    ]
    arrays = {
        "dynamic": dynamic,
        "balanced_output_matrix": outputs["balanced_matrix"],
        "log_h_over_r_profile": outputs["log_h_over_r_profile"],
        "primitive_column_scales": reduced["primitive_column_scales"],
        "state_weights": wp10c8d._state_weights(initial),
    }
    metadata = {
        "work_package": "WP10c8h",
        "base_commit": BASE_COMMIT,
        "n_cells": n_cells,
        "anchor": TANGENT_ANCHOR,
        "state_vector_sha256": _array_sha256(vector),
        "output_names": outputs["names"],
        "descriptor_rank": int(
            np.linalg.matrix_rank(
                reduced["descriptor_reduced_scaled_matrix"]
            )
        ),
        "descriptor_explicit_solve_relative_defect": explicit_defect,
        "descriptor_wall_seconds": time.perf_counter() - started,
        "layouts": {},
    }
    for layout_name, edges_rg in layouts.items():
        masks = _shell_masks(radius_rg, edges_rg)
        rows = []
        names = []
        interface_rows = []
        interface_names = []
        for shell_index, mask in enumerate(masks):
            storage = np.sum(
                measures[mask, None] * state.conserved[mask],
                axis=0,
            )
            response = np.sum(
                measures[mask, None, None] * conserved_response[mask],
                axis=0,
            )
            for component, component_name in LEDGER_COMPONENTS:
                rows.append(
                    response[component]
                    / max(abs(float(storage[component])), np.finfo(float).tiny)
                    / ledger_gate
                )
                names.append(f"shell_{shell_index}_{component_name}")
        for boundary_index, edge_rg in enumerate(edges_rg[1:-1], start=1):
            face = int(
                np.argmin(
                    np.abs(
                        np.asarray(context.grid.edges, dtype=float)
                        / context.grid.gravitational_radius
                        - edge_rg
                    )
                )
            )
            for component, component_name in LEDGER_COMPONENTS:
                interface_rows.append(C * face_response[face, component])
                interface_names.append(
                    f"interface_{boundary_index}_{component_name}"
                )
        arrays[f"{layout_name}_constraint_matrix"] = np.asarray(rows)
        arrays[f"{layout_name}_interface_flux_jacobian"] = np.asarray(
            interface_rows
        )
        metadata["layouts"][layout_name] = {
            "edges_rg": edges_rg,
            "constraint_names": names,
            "interface_flux_names": interface_names,
            "shell_count": len(masks),
            "coarse_coordinate_count": len(rows),
        }
    return arrays, metadata


def _load_or_build_shell_operator_cache(
    initial: dict,
    vector: np.ndarray,
    layouts: dict[str, np.ndarray],
    *,
    force: bool,
) -> tuple[dict, dict, dict]:
    n_cells = initial["state"].n_cells
    path = _shell_operator_cache_path(n_cells)
    expected_state = _array_sha256(vector)
    if path.exists() and not force:
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata_json"].item()))
            arrays = {
                name: np.asarray(saved[name], dtype=float)
                for name in saved.files
                if name != "metadata_json"
            }
        expected_layouts = {
            name: list(edges) for name, edges in layouts.items()
        }
        cached_layouts = {
            name: row["edges_rg"]
            for name, row in metadata.get("layouts", {}).items()
        }
        if not (
            metadata.get("work_package") == "WP10c8h"
            and metadata.get("base_commit") == BASE_COMMIT
            and metadata.get("n_cells") == n_cells
            and metadata.get("anchor") == TANGENT_ANCHOR
            and metadata.get("state_vector_sha256") == expected_state
            and metadata.get("descriptor_rank") == 5 * n_cells
            and cached_layouts == expected_layouts
        ):
            raise RuntimeError("WP10c8h shell operator cache differs")
        return arrays, metadata, {
            "path": _relative(path),
            "sha256": _sha256(path),
            "reused": True,
        }
    arrays, metadata = _build_shell_operator_cache(
        initial,
        vector,
        layouts,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **arrays,
        metadata_json=json.dumps(
            _plain(metadata),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
    )
    return arrays, metadata, {
        "path": _relative(path),
        "sha256": _sha256(path),
        "reused": False,
    }


def _shell_values(
    initial: dict,
    state_vector: np.ndarray,
    edges_rg: np.ndarray,
) -> np.ndarray:
    context = initial["context"]
    n_cells = initial["state"].n_cells
    state = unpack_causal_five_field_state(state_vector, n_cells)
    radius_rg = (
        np.asarray(context.grid.centers, dtype=float)
        / context.grid.gravitational_radius
    )
    measures = np.asarray(context.grid.cell_measures, dtype=float)
    values = []
    for mask in _shell_masks(radius_rg, edges_rg):
        storage = np.sum(
            measures[mask, None] * state.conserved[mask],
            axis=0,
        )
        values.extend(
            float(storage[component])
            for component, _name in LEDGER_COMPONENTS
        )
    return np.asarray(values, dtype=float)


def _normalized_shell_residual(
    prediction: np.ndarray,
    truth: np.ndarray,
) -> np.ndarray:
    gate = CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1[
        "maximum_integrated_conserved_relative"
    ]
    return (
        (prediction - truth)
        / np.maximum(np.abs(truth), np.finfo(float).tiny)
        / gate
    )


def _shell_projective_audit(values: np.ndarray) -> tuple[dict, dict]:
    predictions = {
        "euler_to_0p10": (
            causal_projective_euler_prediction(values[0], values[1]),
            values[2],
        ),
        "euler_to_0p125": (
            causal_projective_euler_prediction(values[1], values[2]),
            values[3],
        ),
        "ab2_to_0p125": (
            causal_projective_ab2_prediction(
                values[0],
                values[1],
                values[2],
            ),
            values[3],
        ),
    }
    rows = {}
    traces = {"values": values}
    for label, (prediction, truth) in predictions.items():
        residual = _normalized_shell_residual(prediction, truth)
        maximum = float(np.max(np.abs(residual)))
        rows[label] = {
            "maximum_normalized_error": maximum,
            "normalized_residual": residual,
            "passed": bool(maximum <= MAXIMUM_GATE_FRACTION),
        }
        traces[f"{label}_normalized_residual"] = residual
    return {
        "predictions": rows,
        "passed": bool(all(row["passed"] for row in rows.values())),
    }, traces


def _signed_window(
    radius_rg: np.ndarray,
    lower: float,
    upper: float,
) -> np.ndarray:
    window = wp10c8c._smooth_window(radius_rg, lower, upper)
    center = 0.5 * (np.log(lower) + np.log(upper))
    width = max(np.log(upper / lower), np.finfo(float).tiny)
    return window * np.tanh(6.0 * (np.log(radius_rg) - center) / width)


def _redistribution_directions(
    initial: dict,
    vector: np.ndarray,
    primitive_scales: np.ndarray,
) -> dict[str, np.ndarray]:
    n_cells = initial["state"].n_cells
    _summary, diagnostics = wp10c8b._off_manifold_diagnostics(
        initial,
        vector,
    )
    radius_rg = np.asarray(diagnostics["radius_rg"], dtype=float)

    def signed(component: int, amplitude: float, lower: float, upper: float):
        values = np.zeros((n_cells, 5), dtype=float)
        values[:, component] = (
            amplitude * _signed_window(radius_rg, lower, upper)
        )
        return values

    source = np.zeros((n_cells, 5), dtype=float)
    source_window = _signed_window(radius_rg, 200.0, 280.0)
    source[:, 0] = 1.0e-2 * source_window
    source[:, 2] = 5.0e-3 * source_window
    stress = np.zeros((n_cells, 5), dtype=float)
    stress[:, 4] = (
        np.asarray(diagnostics["target_specific_stress"], dtype=float)
        - np.asarray(diagnostics["specific_stress"], dtype=float)
    )
    physical = {
        "thermal_redistribution_6_to_60rg": signed(
            3,
            1.0e-2,
            6.0,
            60.0,
        ),
        "density_redistribution_6_to_60rg": signed(
            0,
            1.0e-2,
            6.0,
            60.0,
        ),
        "source_redistribution_200_to_280rg": source,
        "stress_target_adjustment": stress,
        "thermal_redistribution_60_to_200rg": signed(
            3,
            1.0e-2,
            60.0,
            200.0,
        ),
        "density_redistribution_20_to_200rg": signed(
            0,
            1.0e-2,
            20.0,
            200.0,
        ),
        "azimuthal_redistribution_6_to_60rg": signed(
            2,
            2.0e-3,
            6.0,
            60.0,
        ),
        "radial_redistribution_6_to_60rg": signed(
            1,
            2.0e-3,
            6.0,
            60.0,
        ),
    }
    return {
        name: values.ravel() / primitive_scales
        for name, values in physical.items()
    }


def _tangent_shell_audit(
    arrays: dict,
    metadata: dict,
    layout: str,
    directions: dict,
) -> tuple[dict, dict]:
    constraints = np.asarray(
        arrays[f"{layout}_constraint_matrix"],
        dtype=float,
    )
    weights = np.asarray(arrays["state_weights"], dtype=float)
    output_matrix = np.asarray(
        arrays["balanced_output_matrix"],
        dtype=float,
    )
    profile_matrix = (
        np.asarray(arrays["log_h_over_r_profile"], dtype=float)
        / CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1[
            "maximum_log_h_over_r_profile"
        ]
    )
    dynamic = np.asarray(arrays["dynamic"], dtype=float)
    rows = {}
    traces = {}
    for name, direction in directions.items():
        projected, defect = causal_weighted_constraint_null_projection(
            direction,
            constraints,
            state_weights=weights,
        )
        correction = wp10c8g._weighted_relative_change(
            direction,
            projected,
            weights,
        )
        eligible = bool(
            correction <= MAXIMUM_DIRECTION_CORRECTION_FRACTION
        )
        immediate_outputs = output_matrix @ projected
        immediate_profile = profile_matrix @ projected
        derivative = dynamic @ projected
        projected_outputs = (
            PROJECTION_WINDOW_SECONDS * (output_matrix @ derivative)
        )
        projected_profile = (
            PROJECTION_WINDOW_SECONDS * (profile_matrix @ derivative)
        )
        projected_constraints = (
            PROJECTION_WINDOW_SECONDS * (constraints @ derivative)
        )
        immediate = max(
            float(np.max(np.abs(immediate_outputs))),
            float(np.max(np.abs(immediate_profile))),
        )
        projected_output = max(
            float(np.max(np.abs(projected_outputs))),
            float(np.max(np.abs(projected_profile))),
        )
        projected_coarse = float(
            np.max(np.abs(projected_constraints))
        )
        passed = bool(
            eligible
            and float(defect) <= MAXIMUM_CONSTRAINT_DEFECT
            and immediate <= MAXIMUM_GATE_FRACTION
            and projected_output <= MAXIMUM_GATE_FRACTION
            and projected_coarse <= MAXIMUM_GATE_FRACTION
        )
        rows[name] = {
            "correction_fraction": correction,
            "direction_eligible": eligible,
            "constraint_defect": float(defect),
            "maximum_immediate_output_gate_fraction": immediate,
            "maximum_projected_output_gate_fraction": projected_output,
            "maximum_projected_coarse_rate_gate_fraction": projected_coarse,
            "passed": passed,
        }
        traces[name] = {
            "original_direction": direction,
            "projected_direction": projected,
            "immediate_outputs": immediate_outputs,
            "immediate_profile": immediate_profile,
            "projected_outputs": projected_outputs,
            "projected_profile": projected_profile,
            "projected_constraints": projected_constraints,
        }
    eligible_count = sum(row["direction_eligible"] for row in rows.values())
    return {
        "layout": metadata["layouts"][layout],
        "directions": rows,
        "eligible_direction_count": eligible_count,
        "passed": bool(
            eligible_count >= MINIMUM_ELIGIBLE_DIRECTIONS
            and all(
                row["passed"]
                for row in rows.values()
                if row["direction_eligible"]
            )
        ),
    }, traces


def _cross_mesh_scalar_impacts(
    coarse: dict,
    fine: dict,
) -> dict:
    rows = {}
    for direction in coarse["directions"]:
        left = coarse["directions"][direction]
        right = fine["directions"][direction]
        differences = {
            "immediate_output": abs(
                left["maximum_immediate_output_gate_fraction"]
                - right["maximum_immediate_output_gate_fraction"]
            ),
            "projected_output": abs(
                left["maximum_projected_output_gate_fraction"]
                - right["maximum_projected_output_gate_fraction"]
            ),
            "projected_coarse_rate": abs(
                left["maximum_projected_coarse_rate_gate_fraction"]
                - right["maximum_projected_coarse_rate_gate_fraction"]
            ),
        }
        maximum = max(differences.values())
        rows[direction] = {
            "differences": differences,
            "maximum_gate_fraction_difference": maximum,
            "passed": bool(maximum <= MAXIMUM_GATE_FRACTION),
        }
    return {
        "directions": rows,
        "passed": bool(all(row["passed"] for row in rows.values())),
    }


def main() -> None:
    args = _arguments()
    output_path = _absolute(args.output)
    arrays_path = _absolute(args.arrays)
    authorization, authorization_sha256 = _validate_authorization()
    initial, vectors, state_provenance = wp10c8d._load_states()
    restarts, checkpoint_provenance = wp10c8g._checkpoint_restarts(initial)
    layouts = _common_shell_edges(initial)
    if args.preflight:
        print(
            json.dumps(
                {
                    "work_package": "WP10c8h",
                    "preflight_passed": True,
                    "wp10c8g_evidence_sha256": authorization_sha256,
                    "layouts": {
                        name: list(edges) for name, edges in layouts.items()
                    },
                    "checkpoint_count": (
                        len(RESOLUTIONS)
                        * len(TRAJECTORY_MODES)
                        * len(CHECKPOINTS)
                    ),
                },
                sort_keys=True,
            )
        )
        return

    operator_arrays = {}
    operator_metadata = {}
    operator_provenance = {}
    for n_cells in RESOLUTIONS:
        arrays, metadata, provenance = _load_or_build_shell_operator_cache(
            initial[n_cells],
            vectors[n_cells][TANGENT_ANCHOR],
            layouts,
            force=args.force_shell_operators,
        )
        operator_arrays[n_cells] = arrays
        operator_metadata[n_cells] = metadata
        operator_provenance[str(n_cells)] = provenance
        print(
            json.dumps(
                {
                    "work_package": "WP10c8h",
                    "phase": "shell_operator",
                    "n_cells": n_cells,
                    "reused": provenance["reused"],
                    "wall_seconds": metadata["descriptor_wall_seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if args.operators_only:
        return

    projective = {str(n_cells): {} for n_cells in RESOLUTIONS}
    projective_traces = {n_cells: {} for n_cells in RESOLUTIONS}
    path_agreement = {str(n_cells): {} for n_cells in RESOLUTIONS}
    for n_cells in RESOLUTIONS:
        for layout, edges_rg in layouts.items():
            projective[str(n_cells)][layout] = {}
            projective_traces[n_cells][layout] = {}
            mode_values = {}
            for mode in TRAJECTORY_MODES:
                values = np.asarray(
                    [
                        _shell_values(
                            initial[n_cells],
                            restarts[n_cells][mode][label].state_vector,
                            edges_rg,
                        )
                        for label, _time_seconds in CHECKPOINTS
                    ]
                )
                audit, traces = _shell_projective_audit(values)
                projective[str(n_cells)][layout][mode] = audit
                projective_traces[n_cells][layout][mode] = traces
                mode_values[mode] = values
            normalized_path = _normalized_shell_residual(
                mode_values["production"],
                mode_values["temporal_control"],
            )
            maximum_path = float(np.max(np.abs(normalized_path)))
            path_agreement[str(n_cells)][layout] = {
                "maximum_normalized_difference": maximum_path,
                "passed": bool(maximum_path <= MAXIMUM_GATE_FRACTION),
            }

    cross_mesh_projective = {}
    for layout in layouts:
        cross_mesh_projective[layout] = {}
        for mode in TRAJECTORY_MODES:
            rows = {}
            for label in (
                "euler_to_0p10",
                "euler_to_0p125",
                "ab2_to_0p125",
            ):
                coarse = projective_traces[64][layout][mode][
                    f"{label}_normalized_residual"
                ]
                fine = projective_traces[128][layout][mode][
                    f"{label}_normalized_residual"
                ]
                maximum = float(np.max(np.abs(coarse - fine)))
                rows[label] = {
                    "maximum_normalized_residual_difference": maximum,
                    "passed": bool(maximum <= MAXIMUM_GATE_FRACTION),
                }
            cross_mesh_projective[layout][mode] = {
                "predictions": rows,
                "passed": bool(
                    all(row["passed"] for row in rows.values())
                ),
            }

    tangent = {str(n_cells): {} for n_cells in RESOLUTIONS}
    tangent_traces = {n_cells: {} for n_cells in RESOLUTIONS}
    for n_cells in RESOLUTIONS:
        arrays = operator_arrays[n_cells]
        directions = _redistribution_directions(
            initial[n_cells],
            vectors[n_cells][TANGENT_ANCHOR],
            np.asarray(arrays["primitive_column_scales"], dtype=float),
        )
        for layout in layouts:
            audit, traces = _tangent_shell_audit(
                arrays,
                operator_metadata[n_cells],
                layout,
                directions,
            )
            tangent[str(n_cells)][layout] = audit
            tangent_traces[n_cells][layout] = traces

    cross_mesh_tangent = {
        layout: _cross_mesh_scalar_impacts(
            tangent["64"][layout],
            tangent["128"][layout],
        )
        for layout in layouts
    }
    layout_decisions = {}
    for layout in layouts:
        checkpoint_passed = bool(
            all(
                projective[str(n_cells)][layout][mode]["passed"]
                for n_cells in RESOLUTIONS
                for mode in TRAJECTORY_MODES
            )
            and all(
                path_agreement[str(n_cells)][layout]["passed"]
                for n_cells in RESOLUTIONS
            )
            and all(
                cross_mesh_projective[layout][mode]["passed"]
                for mode in TRAJECTORY_MODES
            )
        )
        tangent_passed = bool(
            all(
                tangent[str(n_cells)][layout]["passed"]
                for n_cells in RESOLUTIONS
            )
            and cross_mesh_tangent[layout]["passed"]
        )
        layout_decisions[layout] = {
            "checkpoint_projective_contract_passed": checkpoint_passed,
            "ledger_null_tangent_contract_passed": tangent_passed,
            "complete_contract_passed": bool(
                checkpoint_passed and tangent_passed
            ),
        }

    passing_layouts = [
        name
        for name, row in layout_decisions.items()
        if row["complete_contract_passed"]
    ]
    compact_shell_closure_found = bool(passing_layouts)
    decision = (
        "wp10c8h_compact_conservative_shell_closure_found"
        if compact_shell_closure_found
        else "wp10c8h_compact_conservative_shell_closure_not_identifiable"
    )
    next_authorization = (
        "minimal_n64_shell_healing_microbursts"
        if compact_shell_closure_found
        else "retain_full_dae_microbursts_and_reassess_physical_closure"
    )

    array_payload = {}
    for n_cells in RESOLUTIONS:
        for layout in layouts:
            for mode in TRAJECTORY_MODES:
                prefix = f"n{n_cells}_{layout}_{mode}"
                for name, values in projective_traces[n_cells][layout][
                    mode
                ].items():
                    array_payload[f"{prefix}_{name}"] = values
            for direction, traces in tangent_traces[n_cells][layout].items():
                prefix = f"n{n_cells}_{layout}_{direction}"
                for name, values in traces.items():
                    array_payload[f"{prefix}_{name}"] = values
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **array_payload)

    payload = {
        "work_package": "WP10c8h",
        "base_commit": BASE_COMMIT,
        "decision": decision,
        "next_authorization": next_authorization,
        "scope": {
            "description": (
                "Five-shell and one predeclared eight-shell conservative "
                "equation-free closure preflight"
            ),
            "resolutions": RESOLUTIONS,
            "trajectory_modes": TRAJECTORY_MODES,
            "checkpoints_seconds": dict(CHECKPOINTS),
            "projection_window_seconds": PROJECTION_WINDOW_SECONDS,
            "tangent_anchor": TANGENT_ANCHOR,
            "layouts": {
                name: {
                    "edges_rg": edges,
                    "shell_count": edges.size - 1,
                    "coarse_coordinate_count": 3 * (edges.size - 1),
                }
                for name, edges in layouts.items()
            },
            "new_nonlinear_microbursts_run": False,
            "new_full_dae_trajectory_run": False,
        },
        "authorization": {
            "wp10c8g_decision": authorization["decision"],
            "wp10c8g_evidence_sha256": authorization_sha256,
        },
        "state_provenance": state_provenance,
        "checkpoint_provenance": checkpoint_provenance,
        "operator_provenance": operator_provenance,
        "projective_predictions": projective,
        "production_control_agreement": path_agreement,
        "cross_mesh_projective_agreement": cross_mesh_projective,
        "shell_ledger_null_tangent_identifiability": tangent,
        "cross_mesh_tangent_agreement": cross_mesh_tangent,
        "layout_decisions": layout_decisions,
        "passing_layouts": passing_layouts,
        "gates": {
            "maximum_gate_fraction": MAXIMUM_GATE_FRACTION,
            "maximum_constraint_defect": MAXIMUM_CONSTRAINT_DEFECT,
            "maximum_direction_correction_fraction": (
                MAXIMUM_DIRECTION_CORRECTION_FRACTION
            ),
            "minimum_eligible_directions": MINIMUM_ELIGIBLE_DIRECTIONS,
            "compact_shell_closure_found": compact_shell_closure_found,
            "nonlinear_shell_microbursts_authorized": (
                compact_shell_closure_found
            ),
            "nonlinear_macrostep_authorized": False,
        },
        "interpretation": {
            "conservation": (
                "Each constraint is an exact finite-volume cell integral "
                "of mass, angular momentum, or Killing energy over mesh-"
                "coincident physical shells."
            ),
            "redistributions": (
                "Signed perturbations move unresolved structure within "
                "shells before a minimum-weighted correction restores every "
                "declared shell ledger to first order."
            ),
            "stop_rule": (
                "Failure of both the five-shell and the single authorized "
                "eight-shell refinement closes compact shell-only "
                "microbursts under the current observable contract."
            ),
        },
        "artifacts": {
            "arrays_path": _relative(arrays_path),
            "arrays_sha256": _sha256(arrays_path),
        },
    }
    _write_json(output_path, payload)
    print(
        json.dumps(
            {
                "work_package": "WP10c8h",
                "decision": decision,
                "passing_layouts": passing_layouts,
                "next_authorization": next_authorization,
                "output": _relative(output_path),
                "arrays": _relative(arrays_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
