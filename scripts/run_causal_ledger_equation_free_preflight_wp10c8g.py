"""Test ledger-driven equation-free closure within the certified truth window."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import run_causal_characteristic_extension_wp10c7l as wp10c7l
import run_causal_mixed_mode_reduction_audit_wp10c8d as wp10c8d
import run_causal_region_selective_closure_audit_wp10c8c as wp10c8c
import run_causal_stable_observable_reduction_audit_wp10c8f as wp10c8f
import run_causal_stress_time_audit_wp10c8b as wp10c8b
from imri_qpe.layer3_minidisk_1d import (
    CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1,
    causal_five_field_bdf_physical_ledger_from_restart,
    causal_five_field_h_over_r_profile,
    causal_five_field_observable_snapshot,
    causal_projective_ab2_prediction,
    causal_projective_euler_prediction,
    causal_weighted_constraint_null_projection,
    unpack_causal_five_field_state,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "d89663531dbbce97be01d583e364bc3585448e76"
WP10C8F_OUTPUT = (
    ROOT
    / "outputs/tables/causal_stable_observable_reduction_audit_wp10c8f.json"
)
WP10C8D_OUTPUT = (
    ROOT / "outputs/tables/causal_mixed_mode_reduction_audit_wp10c8d.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/tables/causal_ledger_equation_free_preflight_wp10c8g.json"
)
DEFAULT_ARRAYS = (
    ROOT
    / "outputs/tables/causal_ledger_equation_free_preflight_wp10c8g_arrays.npz"
)
RESOLUTIONS = (64, 128)
TRAJECTORY_MODES = ("production", "temporal_control")
CHECKPOINTS = (
    ("t_0p05", 5.0e-2),
    ("t_0p075", 7.5e-2),
    ("t_0p10", 1.0e-1),
    ("t_0p125", 1.25e-1),
)
TANGENT_ANCHOR = "t_0p125"
PROJECTION_WINDOW_SECONDS = 2.5e-2
MAXIMUM_GATE_FRACTION = 2.5e-1
MAXIMUM_RATE_SECANT_RELATIVE_ERROR = 5.0e-2
MAXIMUM_CONSTRAINT_DEFECT = 1.0e-10
MAXIMUM_DIRECTION_CORRECTION_FRACTION = 7.5e-1
MINIMUM_ELIGIBLE_DIRECTIONS = 4
LEDGER_COMPONENTS = (
    (0, "rest_mass"),
    (2, "angular_momentum"),
    (3, "killing_energy"),
)
H_OVER_R_BANDS_RG = (
    ("horizon_to_6rg", 0.0, 6.0),
    ("6_to_60rg", 6.0, 60.0),
    ("60_to_200rg", 60.0, 200.0),
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate authorization, checkpoints, and descriptor caches.",
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


def _validate_authorization() -> tuple[dict, str, dict]:
    if not WP10C8F_OUTPUT.exists() or not WP10C8D_OUTPUT.exists():
        raise RuntimeError("WP10c8g requires canonical WP10c8d/f evidence")
    evidence = json.loads(WP10C8F_OUTPUT.read_text(encoding="utf-8"))
    evidence_arrays = ROOT / str(
        evidence.get("artifacts", {}).get("arrays_path", "")
    )
    descriptors = json.loads(WP10C8D_OUTPUT.read_text(encoding="utf-8"))
    if not (
        evidence.get("work_package") == "WP10c8f"
        and evidence.get("decision")
        == "wp10c8f_stable_cross_mesh_observable_model_not_found"
        and evidence.get("next_authorization")
        == "ledger_driven_equation_free_closure_preflight"
        and not evidence.get("gates", {}).get(
            "nonlinear_rom_authorized",
            True,
        )
        and not evidence.get("gates", {}).get(
            "memory_model_authorized",
            True,
        )
        and evidence_arrays.exists()
        and _sha256(evidence_arrays)
        == evidence.get("artifacts", {}).get("arrays_sha256")
        and descriptors.get("work_package") == "WP10c8d"
        and descriptors.get("decision")
        == "wp10c8d_compact_cross_mesh_markovian_basis_not_found"
    ):
        raise RuntimeError("WP10c8f did not authorize this preflight")
    return evidence, _sha256(WP10C8F_OUTPUT), descriptors


def _checkpoint_restarts(initial: dict) -> tuple[dict, dict]:
    (
        _spectral,
        spectral_sha256,
        reference,
        reference_sha256,
    ) = wp10c8b._validate_authorization()
    (
        _bundles,
        wp10c7k_evidence,
        wp10c7k_sha256,
    ) = wp10c8b._initial_bundles(reference)
    restarts = {n_cells: {} for n_cells in RESOLUTIONS}
    provenance = {str(n_cells): {} for n_cells in RESOLUTIONS}
    for n_cells in RESOLUTIONS:
        for mode in TRAJECTORY_MODES:
            parent, parent_entry = wp10c8b._parent_restart(
                initial[n_cells],
                n_cells,
                mode,
                wp10c7k_evidence,
                wp10c7k_sha256,
                reference,
            )
            if parent.elapsed_time == dict(CHECKPOINTS)["t_0p05"]:
                at_050 = parent
                at_050_path = ROOT / parent_entry["path"]
            else:
                wp10c7k_parent = wp10c7l._parent_checkpoint_entry(
                    wp10c7k_evidence,
                    n_cells,
                )
                at_050 = wp10c7l._load_snapshot(
                    initial[n_cells],
                    wp10c7k_sha256,
                    wp10c7k_parent,
                    mode,
                    "t_0p05",
                )
                at_050_path = wp10c7l._checkpoint_path(
                    n_cells,
                    mode,
                    "t_0p05",
                )
            selected = {"t_0p05": at_050}
            selected_paths = {"t_0p05": at_050_path}
            for label, _time_seconds in CHECKPOINTS[1:]:
                selected[label] = wp10c8b._load_snapshot(
                    initial[n_cells],
                    mode,
                    label,
                    parent_entry,
                    spectral_sha256,
                    reference_sha256,
                )
                selected_paths[label] = wp10c8b._checkpoint_path(
                    n_cells,
                    mode,
                    label,
                )
            restarts[n_cells][mode] = selected
            provenance[str(n_cells)][mode] = {
                label: {
                    "path": _relative(selected_paths[label]),
                    "sha256": _sha256(selected_paths[label]),
                    "state_vector_sha256": _array_sha256(
                        selected[label].state_vector
                    ),
                    "elapsed_time_seconds": selected[label].elapsed_time,
                }
                for label, _time_seconds in CHECKPOINTS
            }
    return restarts, provenance


def _observable_row(initial: dict, restart) -> dict:
    context = initial["context"]
    vector = np.asarray(restart.state_vector, dtype=float)
    radius_rg = (
        np.asarray(context.grid.centers, dtype=float)
        / context.grid.gravitational_radius
    )
    measures = np.asarray(context.grid.cell_measures, dtype=float)
    snapshot = causal_five_field_observable_snapshot(
        context,
        vector,
        cooling_inner_cutoff=6.0 * context.grid.gravitational_radius,
    )
    log_h_over_r = np.log(
        causal_five_field_h_over_r_profile(context, vector)
    )
    moments = {}
    for name, lower, upper in H_OVER_R_BANDS_RG:
        mask = (radius_rg >= lower) & (radius_rg < upper)
        weights = measures[mask] / float(np.sum(measures[mask]))
        moments[name] = float(weights @ log_h_over_r[mask])
    ledger = causal_five_field_bdf_physical_ledger_from_restart(restart)
    integrated = np.asarray(snapshot.integrated_conserved, dtype=float)
    effective = integrated + np.asarray(
        ledger.actual_vertical_storage,
        dtype=float,
    )
    return {
        "global_ledgers": {
            name: float(integrated[component])
            for component, name in LEDGER_COMPONENTS
        },
        "effective_global_ledgers": {
            name: float(effective[component])
            for component, name in LEDGER_COMPONENTS
        },
        "augmented_observables": {
            "cooling_outside_6rg": (
                snapshot.cooling_power_proxy_outside_cutoff_erg_s
            ),
            "inner_accretion_rate": snapshot.inner_accretion_rate_g_s,
            **{
                f"log_h_over_r_moment_{name}": value
                for name, value in moments.items()
            },
        },
        "diagnostic_observables": {
            "total_cooling": snapshot.cooling_power_proxy_erg_s,
            "maximum_h_over_r": snapshot.maximum_h_over_r,
        },
        "state_vector_sha256": _array_sha256(vector),
    }


def _candidate_schema(name: str) -> tuple[tuple[str, str, float], ...]:
    ledger_gate = CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1[
        "maximum_integrated_conserved_relative"
    ]
    rows = tuple(
        (component_name, "relative", ledger_gate)
        for _component, component_name in LEDGER_COMPONENTS
    )
    if name == "global_ledgers":
        return rows
    if name != "observable_augmented_ledgers":
        raise ValueError(f"unknown coarse candidate {name!r}")
    return rows + (
        (
            "cooling_outside_6rg",
            "relative",
            CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1[
                "cooling_power_proxy_outside_cutoff_relative"
            ],
        ),
        (
            "inner_accretion_rate",
            "relative",
            CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1[
                "inner_accretion_rate_relative"
            ],
        ),
        *tuple(
            (
                f"log_h_over_r_moment_{band_name}",
                "absolute",
                CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1[
                    "maximum_log_h_over_r_profile"
                ],
            )
            for band_name, _lower, _upper in H_OVER_R_BANDS_RG
        ),
    )


def _candidate_values(row: dict, name: str) -> np.ndarray:
    values = dict(row["global_ledgers"])
    if name == "observable_augmented_ledgers":
        values.update(row["augmented_observables"])
    return np.asarray(
        [values[item_name] for item_name, _kind, _gate in _candidate_schema(name)],
        dtype=float,
    )


def _normalized_residual(
    prediction: np.ndarray,
    truth: np.ndarray,
    schema: tuple[tuple[str, str, float], ...],
) -> np.ndarray:
    result = np.empty(len(schema), dtype=float)
    for index, (_name, kind, gate) in enumerate(schema):
        difference = float(prediction[index] - truth[index])
        if kind == "relative":
            difference /= max(
                abs(float(truth[index])),
                np.finfo(float).tiny,
            )
        result[index] = difference / gate
    return result


def _projective_audit(rows: dict, candidate: str) -> tuple[dict, dict]:
    schema = _candidate_schema(candidate)
    values = np.asarray(
        [
            _candidate_values(rows[label], candidate)
            for label, _time_seconds in CHECKPOINTS
        ],
        dtype=float,
    )
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
    audits = {}
    traces = {"values": values}
    for label, (prediction, truth) in predictions.items():
        residual = _normalized_residual(prediction, truth, schema)
        maximum = float(np.max(np.abs(residual)))
        audits[label] = {
            "normalized_residuals": residual,
            "maximum_normalized_error": maximum,
            "passed": bool(maximum <= MAXIMUM_GATE_FRACTION),
        }
        traces[f"{label}_prediction"] = prediction
        traces[f"{label}_truth"] = truth
        traces[f"{label}_normalized_residual"] = residual
    return {
        "component_schema": [
            {"name": name, "difference": kind, "gate": gate}
            for name, kind, gate in schema
        ],
        "predictions": audits,
        "passed": bool(all(row["passed"] for row in audits.values())),
    }, traces


def _path_agreement(
    left: dict,
    right: dict,
    candidate: str,
) -> dict:
    schema = _candidate_schema(candidate)
    rows = {}
    for label, _time_seconds in CHECKPOINTS:
        left_values = _candidate_values(left[label], candidate)
        right_values = _candidate_values(right[label], candidate)
        residual = _normalized_residual(left_values, right_values, schema)
        maximum = float(np.max(np.abs(residual)))
        rows[label] = {
            "normalized_difference": residual,
            "maximum_normalized_difference": maximum,
            "passed": bool(maximum <= MAXIMUM_GATE_FRACTION),
        }
    return {
        "checkpoints": rows,
        "passed": bool(all(row["passed"] for row in rows.values())),
    }


def _cross_mesh_projective_agreement(
    traces: dict,
    candidate: str,
    mode: str,
) -> dict:
    rows = {}
    for label in (
        "euler_to_0p10",
        "euler_to_0p125",
        "ab2_to_0p125",
    ):
        coarse = traces[64][mode][candidate][
            f"{label}_normalized_residual"
        ]
        fine = traces[128][mode][candidate][
            f"{label}_normalized_residual"
        ]
        maximum = float(np.max(np.abs(coarse - fine)))
        rows[label] = {
            "maximum_normalized_residual_difference": maximum,
            "passed": bool(maximum <= MAXIMUM_GATE_FRACTION),
        }
    return {
        "predictions": rows,
        "passed": bool(all(row["passed"] for row in rows.values())),
    }


def _ledger_rate_secant_audit(
    initial: dict,
    restarts: dict,
    rows: dict,
) -> dict:
    instantaneous = {}
    for label, _time_seconds in CHECKPOINTS:
        instantaneous[label] = wp10c8f._global_ledger_row(
            initial,
            restarts[label].state_vector,
        )
    intervals = {}
    for (left_label, left_time), (right_label, right_time) in zip(
        CHECKPOINTS[:-1],
        CHECKPOINTS[1:],
        strict=True,
    ):
        timestep = right_time - left_time
        interval = {}
        for _component, name in LEDGER_COMPONENTS:
            left = rows[left_label]["effective_global_ledgers"][name]
            right = rows[right_label]["effective_global_ledgers"][name]
            secant = (right - left) / timestep
            left_rate = instantaneous[left_label]["components"][name][
                "net_storage_plus_vertical_rate"
            ]
            right_rate = instantaneous[right_label]["components"][name][
                "net_storage_plus_vertical_rate"
            ]
            trapezoidal = 0.5 * (left_rate + right_rate)
            difference = abs(secant - trapezoidal) / max(
                abs(secant),
                abs(trapezoidal),
                np.finfo(float).tiny,
            )
            interval[name] = {
                "effective_storage_secant_rate": secant,
                "trapezoidal_instantaneous_rate": trapezoidal,
                "relative_difference": difference,
                "passed": bool(
                    difference <= MAXIMUM_RATE_SECANT_RELATIVE_ERROR
                ),
            }
        intervals[f"{left_label}_to_{right_label}"] = interval
    passed = all(
        row["passed"]
        for interval in intervals.values()
        for row in interval.values()
    )
    return {
        "instantaneous": instantaneous,
        "intervals": intervals,
        "passed": bool(passed),
    }


def _physical_perturbation_directions(
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

    def windowed(component: int, amplitude: float, lower: float, upper: float):
        values = np.zeros((n_cells, 5), dtype=float)
        values[:, component] = amplitude * wp10c8c._smooth_window(
            radius_rg,
            lower,
            upper,
        )
        return values

    source = np.zeros((n_cells, 5), dtype=float)
    source_window = wp10c8c._smooth_window(radius_rg, 200.0, 280.0)
    source[:, 0] = 1.0e-2 * source_window
    source[:, 2] = 5.0e-3 * source_window
    stress = np.zeros((n_cells, 5), dtype=float)
    stress[:, 4] = (
        np.asarray(diagnostics["target_specific_stress"], dtype=float)
        - np.asarray(diagnostics["specific_stress"], dtype=float)
    )
    physical = {
        "thermal_6_to_60rg": windowed(3, 1.0e-2, 6.0, 60.0),
        "surface_density_6_to_60rg": windowed(
            0,
            1.0e-2,
            6.0,
            60.0,
        ),
        "source_band_loading_200_to_280rg": source,
        "stress_target_adjustment": stress,
        "thermal_60_to_200rg": windowed(3, 1.0e-2, 60.0, 200.0),
        "surface_density_20_to_200rg": windowed(
            0,
            1.0e-2,
            20.0,
            200.0,
        ),
        "azimuthal_velocity_6_to_60rg": windowed(
            2,
            2.0e-3,
            6.0,
            60.0,
        ),
        "radial_velocity_6_to_60rg": windowed(
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


def _constraint_rows(arrays: dict, metadata: dict, candidate: str) -> dict:
    ledger_gate = CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1[
        "maximum_integrated_conserved_relative"
    ]
    ledger = np.asarray(arrays["protected_operators"], dtype=float)
    normalized_ledger = ledger / ledger_gate
    if candidate == "global_ledgers":
        return {
            "matrix": normalized_ledger,
            "names": tuple(name for _component, name in LEDGER_COMPONENTS),
            "selected_output_indices": (),
        }
    names = tuple(metadata["output_names"])
    selected_names = (
        "cooling_outside_6rg_relative",
        "inner_accretion_relative",
        "log_h_over_r_moment_horizon_to_6rg",
        "log_h_over_r_moment_6_to_60rg",
        "log_h_over_r_moment_60_to_200rg",
    )
    selected = tuple(names.index(name) for name in selected_names)
    return {
        "matrix": np.vstack(
            (
                normalized_ledger,
                np.asarray(arrays["balanced_output_matrix"])[
                    np.asarray(selected)
                ],
            )
        ),
        "names": (
            *(name for _component, name in LEDGER_COMPONENTS),
            *selected_names,
        ),
        "selected_output_indices": selected,
    }


def _weighted_relative_change(
    original: np.ndarray,
    projected: np.ndarray,
    weights: np.ndarray,
) -> float:
    difference = projected - original
    return float(
        np.sqrt(np.sum(weights * difference**2))
        / max(
            float(np.sqrt(np.sum(weights * original**2))),
            np.finfo(float).tiny,
        )
    )


def _tangent_candidate_audit(
    arrays: dict,
    metadata: dict,
    directions: dict,
    candidate: str,
) -> tuple[dict, dict]:
    constraints = _constraint_rows(arrays, metadata, candidate)
    matrix = np.asarray(constraints["matrix"], dtype=float)
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
    selected = set(constraints["selected_output_indices"])
    held_indices = np.asarray(
        [
            index
            for index in range(output_matrix.shape[0])
            if index not in selected
            and index < output_matrix.shape[0] - len(LEDGER_COMPONENTS)
        ],
        dtype=int,
    )
    rows = {}
    traces = {}
    for name, direction in directions.items():
        projected, defect = causal_weighted_constraint_null_projection(
            direction,
            matrix,
            state_weights=weights,
        )
        correction = _weighted_relative_change(
            direction,
            projected,
            weights,
        )
        eligible = bool(
            correction <= MAXIMUM_DIRECTION_CORRECTION_FRACTION
        )
        immediate_selected = output_matrix @ projected
        immediate_profile = profile_matrix @ projected
        derivative = dynamic @ projected
        projected_selected = (
            PROJECTION_WINDOW_SECONDS * (output_matrix @ derivative)
        )
        projected_profile = (
            PROJECTION_WINDOW_SECONDS * (profile_matrix @ derivative)
        )
        projected_constraints = (
            PROJECTION_WINDOW_SECONDS * (matrix @ derivative)
        )
        immediate_held = max(
            float(np.max(np.abs(immediate_selected[held_indices]))),
            float(np.max(np.abs(immediate_profile))),
        )
        rate_held = max(
            float(np.max(np.abs(projected_selected[held_indices]))),
            float(np.max(np.abs(projected_profile))),
        )
        constraint_rate = float(
            np.max(np.abs(projected_constraints))
        )
        passed = bool(
            eligible
            and float(defect) <= MAXIMUM_CONSTRAINT_DEFECT
            and immediate_held <= MAXIMUM_GATE_FRACTION
            and rate_held <= MAXIMUM_GATE_FRACTION
            and constraint_rate <= MAXIMUM_GATE_FRACTION
        )
        rows[name] = {
            "correction_fraction": correction,
            "direction_eligible": eligible,
            "constraint_defect": float(defect),
            "maximum_immediate_held_output_gate_fraction": immediate_held,
            "maximum_projected_held_output_gate_fraction": rate_held,
            "maximum_projected_constraint_gate_fraction": constraint_rate,
            "passed": passed,
        }
        traces[name] = {
            "original_direction": direction,
            "projected_direction": projected,
            "immediate_selected": immediate_selected,
            "immediate_profile": immediate_profile,
            "projected_selected": projected_selected,
            "projected_profile": projected_profile,
            "projected_constraints": projected_constraints,
        }
    eligible_count = sum(row["direction_eligible"] for row in rows.values())
    return {
        "constraint_names": constraints["names"],
        "directions": rows,
        "eligible_direction_count": eligible_count,
        "minimum_eligible_direction_count": MINIMUM_ELIGIBLE_DIRECTIONS,
        "passed": bool(
            eligible_count >= MINIMUM_ELIGIBLE_DIRECTIONS
            and all(
                row["passed"]
                for row in rows.values()
                if row["direction_eligible"]
            )
        ),
    }, traces


def _load_tangent_cache(
    descriptor_evidence: dict,
    n_cells: int,
) -> tuple[dict, dict, dict]:
    cache = descriptor_evidence["descriptor_caches"][str(n_cells)][
        TANGENT_ANCHOR
    ]
    path = ROOT / cache["path"]
    if not path.exists() or _sha256(path) != cache["sha256"]:
        raise RuntimeError(f"WP10c8g descriptor cache differs: {path}")
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata_json"].item()))
        arrays = {
            name: np.asarray(saved[name], dtype=float)
            for name in saved.files
            if name != "metadata_json"
        }
    if not (
        metadata.get("work_package") == "WP10c8d"
        and metadata.get("n_cells") == n_cells
        and metadata.get("anchor") == TANGENT_ANCHOR
        and metadata.get("descriptor_rank") == 5 * n_cells
    ):
        raise RuntimeError("WP10c8g descriptor metadata differs")
    return arrays, metadata, {
        "path": cache["path"],
        "sha256": cache["sha256"],
        "state_vector_sha256": metadata["state_vector_sha256"],
    }


def main() -> None:
    args = _arguments()
    output_path = _absolute(args.output)
    arrays_path = _absolute(args.arrays)
    authorization, authorization_sha256, descriptor_evidence = (
        _validate_authorization()
    )
    initial, vectors, state_provenance = wp10c8d._load_states()
    restarts, checkpoint_provenance = _checkpoint_restarts(initial)
    tangent_caches = {}
    tangent_cache_provenance = {}
    for n_cells in RESOLUTIONS:
        arrays, metadata, provenance = _load_tangent_cache(
            descriptor_evidence,
            n_cells,
        )
        tangent_caches[n_cells] = (arrays, metadata)
        tangent_cache_provenance[str(n_cells)] = provenance
    if args.preflight:
        print(
            json.dumps(
                {
                    "work_package": "WP10c8g",
                    "preflight_passed": True,
                    "wp10c8f_evidence_sha256": authorization_sha256,
                    "checkpoint_count": (
                        len(RESOLUTIONS)
                        * len(TRAJECTORY_MODES)
                        * len(CHECKPOINTS)
                    ),
                    "descriptor_cache_count": len(RESOLUTIONS),
                },
                sort_keys=True,
            )
        )
        return

    observable_rows = {n_cells: {} for n_cells in RESOLUTIONS}
    for n_cells in RESOLUTIONS:
        for mode in TRAJECTORY_MODES:
            observable_rows[n_cells][mode] = {
                label: _observable_row(
                    initial[n_cells],
                    restarts[n_cells][mode][label],
                )
                for label, _time_seconds in CHECKPOINTS
            }

    candidate_names = (
        "global_ledgers",
        "observable_augmented_ledgers",
    )
    projective = {str(n_cells): {} for n_cells in RESOLUTIONS}
    projective_traces = {n_cells: {} for n_cells in RESOLUTIONS}
    rate_secant = {str(n_cells): {} for n_cells in RESOLUTIONS}
    for n_cells in RESOLUTIONS:
        for mode in TRAJECTORY_MODES:
            projective[str(n_cells)][mode] = {}
            projective_traces[n_cells][mode] = {}
            for candidate in candidate_names:
                audit, traces = _projective_audit(
                    observable_rows[n_cells][mode],
                    candidate,
                )
                projective[str(n_cells)][mode][candidate] = audit
                projective_traces[n_cells][mode][candidate] = traces
            rate_secant[str(n_cells)][mode] = _ledger_rate_secant_audit(
                initial[n_cells],
                restarts[n_cells][mode],
                observable_rows[n_cells][mode],
            )

    path_agreement = {str(n_cells): {} for n_cells in RESOLUTIONS}
    for n_cells in RESOLUTIONS:
        for candidate in candidate_names:
            path_agreement[str(n_cells)][candidate] = _path_agreement(
                observable_rows[n_cells]["production"],
                observable_rows[n_cells]["temporal_control"],
                candidate,
            )
    cross_mesh = {candidate: {} for candidate in candidate_names}
    for candidate in candidate_names:
        for mode in TRAJECTORY_MODES:
            cross_mesh[candidate][mode] = (
                _cross_mesh_projective_agreement(
                    projective_traces,
                    candidate,
                    mode,
                )
            )

    tangent = {str(n_cells): {} for n_cells in RESOLUTIONS}
    tangent_traces = {n_cells: {} for n_cells in RESOLUTIONS}
    for n_cells in RESOLUTIONS:
        arrays, metadata = tangent_caches[n_cells]
        directions = _physical_perturbation_directions(
            initial[n_cells],
            vectors[n_cells][TANGENT_ANCHOR],
            np.asarray(arrays["primitive_column_scales"], dtype=float),
        )
        for candidate in candidate_names:
            audit, traces = _tangent_candidate_audit(
                arrays,
                metadata,
                directions,
                candidate,
            )
            tangent[str(n_cells)][candidate] = audit
            tangent_traces[n_cells][candidate] = traces

    candidates = {}
    for candidate in candidate_names:
        checkpoint_passed = bool(
            all(
                projective[str(n_cells)][mode][candidate]["passed"]
                for n_cells in RESOLUTIONS
                for mode in TRAJECTORY_MODES
            )
            and all(
                path_agreement[str(n_cells)][candidate]["passed"]
                for n_cells in RESOLUTIONS
            )
            and all(
                cross_mesh[candidate][mode]["passed"]
                for mode in TRAJECTORY_MODES
            )
            and all(
                rate_secant[str(n_cells)][mode]["passed"]
                for n_cells in RESOLUTIONS
                for mode in TRAJECTORY_MODES
            )
        )
        tangent_passed = all(
            tangent[str(n_cells)][candidate]["passed"]
            for n_cells in RESOLUTIONS
        )
        candidates[candidate] = {
            "checkpoint_projective_contract_passed": checkpoint_passed,
            "ledger_null_tangent_contract_passed": tangent_passed,
            "complete_contract_passed": bool(
                checkpoint_passed and tangent_passed
            ),
        }

    passing_candidates = [
        name
        for name, row in candidates.items()
        if row["complete_contract_passed"]
    ]
    global_closure_found = bool(passing_candidates)
    decision = (
        "wp10c8g_global_equation_free_closure_candidate_found"
        if global_closure_found
        else "wp10c8g_global_equation_free_closure_not_identifiable"
    )
    next_authorization = (
        "minimal_symmetric_n64_healing_microbursts"
        if global_closure_found
        else "five_shell_conservative_closure_preflight"
    )

    array_payload = {}
    for n_cells in RESOLUTIONS:
        for mode in TRAJECTORY_MODES:
            for candidate in candidate_names:
                prefix = f"n{n_cells}_{mode}_{candidate}"
                for name, values in projective_traces[n_cells][mode][
                    candidate
                ].items():
                    array_payload[f"{prefix}_{name}"] = values
        for candidate in candidate_names:
            for direction, traces in tangent_traces[n_cells][candidate].items():
                prefix = f"n{n_cells}_{candidate}_{direction}"
                for name, values in traces.items():
                    array_payload[f"{prefix}_{name}"] = values
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **array_payload)

    payload = {
        "work_package": "WP10c8g",
        "base_commit": BASE_COMMIT,
        "decision": decision,
        "next_authorization": next_authorization,
        "scope": {
            "description": (
                "Existing-checkpoint projective and DAE-consistent "
                "ledger-null identifiability preflight"
            ),
            "resolutions": RESOLUTIONS,
            "trajectory_modes": TRAJECTORY_MODES,
            "checkpoints_seconds": dict(CHECKPOINTS),
            "projection_window_seconds": PROJECTION_WINDOW_SECONDS,
            "tangent_anchor": TANGENT_ANCHOR,
            "candidate_coarse_states": candidate_names,
            "new_nonlinear_microbursts_run": False,
            "new_full_dae_trajectory_run": False,
        },
        "authorization": {
            "wp10c8f_decision": authorization["decision"],
            "wp10c8f_evidence_sha256": authorization_sha256,
        },
        "state_provenance": state_provenance,
        "checkpoint_provenance": checkpoint_provenance,
        "tangent_cache_provenance": tangent_cache_provenance,
        "observable_rows": observable_rows,
        "ledger_rate_secant": rate_secant,
        "projective_predictions": projective,
        "production_control_agreement": path_agreement,
        "cross_mesh_projective_agreement": cross_mesh,
        "ledger_null_tangent_identifiability": tangent,
        "candidate_decisions": candidates,
        "passing_candidates": passing_candidates,
        "gates": {
            "maximum_gate_fraction": MAXIMUM_GATE_FRACTION,
            "maximum_rate_secant_relative_error": (
                MAXIMUM_RATE_SECANT_RELATIVE_ERROR
            ),
            "maximum_constraint_defect": MAXIMUM_CONSTRAINT_DEFECT,
            "maximum_direction_correction_fraction": (
                MAXIMUM_DIRECTION_CORRECTION_FRACTION
            ),
            "minimum_eligible_directions": MINIMUM_ELIGIBLE_DIRECTIONS,
            "global_equation_free_closure_found": global_closure_found,
            "nonlinear_microbursts_authorized": global_closure_found,
            "nonlinear_macrostep_authorized": False,
            "five_shell_preflight_authorized": not global_closure_found,
        },
        "interpretation": {
            "checkpoint_prediction": (
                "Factor-two Euler and AB2 predictions use only withheld "
                "states inside the certified 0.125 s truth interval."
            ),
            "ledger_null_directions": (
                "Physical primitive perturbations are projected with the "
                "minimum cell-measure-weighted correction into each coarse "
                "constraint null space before any response is measured."
            ),
            "stop_rule": (
                "Failure on either mesh at the latest common descriptor "
                "anchor rejects a universal global closure and avoids new "
                "nonlinear lifting/healing microbursts."
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
                "work_package": "WP10c8g",
                "decision": decision,
                "passing_candidates": passing_candidates,
                "next_authorization": next_authorization,
                "output": _relative(output_path),
                "arrays": _relative(arrays_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
