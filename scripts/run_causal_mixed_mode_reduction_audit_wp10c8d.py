"""Audit conservation-constrained mixed causal modes for WP10c8d."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import eigvals, subspace_angles

import run_causal_region_selective_closure_audit_wp10c8c as wp10c8c
import run_causal_slow_mode_audit_wp10c8a as wp10c8a
import run_causal_stress_time_audit_wp10c8b as wp10c8b
from imri_qpe.layer3_minidisk_1d import (
    CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1,
    audit_causal_five_field_state_gates,
    causal_conservation_constrained_balanced_rom,
    causal_descriptor_explicit_matrices,
    causal_five_field_reduced_descriptor_matrices,
    causal_linear_initial_response,
    causal_restrict_cell_averages,
    causal_rom_initial_response,
    causal_rom_memory_kernel_actions,
    causal_stream_descriptor_inputs,
    causal_truncate_mixed_mode_rom,
    unpack_causal_five_field_state,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "4247696c2c65039fc4c08d6aaca7cbace8be6636"
WP10C8C_OUTPUT = (
    ROOT
    / "outputs/tables/causal_region_selective_closure_audit_wp10c8c.json"
)
CACHE_DIRECTORY = (
    ROOT / "outputs/checkpoints/causal_five_field_wp10c8d"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/tables/causal_mixed_mode_reduction_audit_wp10c8d.json"
)
DEFAULT_ARRAYS = (
    ROOT
    / "outputs/tables/causal_mixed_mode_reduction_audit_wp10c8d_arrays.npz"
)
RESOLUTIONS = (64, 128)
ANCHORS = (
    ("t_0", 0.0),
    ("t_0p05", 5.0e-2),
    ("t_0p125", 1.25e-1),
)
ORDERS = (8, 16, 32, 64, 96, 128)
CERTIFIED_HORIZON_SECONDS = 1.25e-1
RESPONSE_TIMES_SECONDS = (0.0, 1.0e-2, 5.0e-2, 1.25e-1)
MEMORY_TIMES_SECONDS = (
    0.0,
    1.0e-2,
    1.0e-1,
    1.0,
    10.0,
    100.0,
    1000.0,
    2000.0,
)
BALANCED_SNAPSHOT_COUNT = 18
MAXIMUM_TRAINING_RELATIVE_ERROR = 0.10
MAXIMUM_HELD_OUT_RELATIVE_ERROR = 0.25
MAXIMUM_REDUCED_REAL_EIGENVALUE_PER_S = 1.0e-8
MAXIMUM_PROTECTED_DEFECT = 2.0e-9
MAXIMUM_LINEAR_ONLINE_COST_FRACTION = 0.10
PREFERRED_MAXIMUM_ORDER = 64
MAXIMUM_CROSS_MESH_95TH_ANGLE_DEGREES = 45.0
OUTPUT_ACTIVITY_FLOOR = 1.0e-10
H_OVER_R_SAMPLE_RADII_RG = (2.5, 6.0, 10.0, 20.0, 60.0, 240.0)
H_OVER_R_BANDS_RG = (
    ("horizon_to_6rg", 0.0, 6.0),
    ("6_to_60rg", 6.0, 60.0),
    ("60_to_200rg", 60.0, 200.0),
    ("200rg_to_outer", 200.0, np.inf),
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-descriptors", action="store_true")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate authorization and all selected causal states.",
    )
    parser.add_argument(
        "--descriptors-only",
        action="store_true",
        help="Build or validate resumable descriptor caches only.",
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
    array = np.ascontiguousarray(values)
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
    if not WP10C8C_OUTPUT.exists():
        raise RuntimeError("WP10c8d requires canonical WP10c8c evidence")
    evidence = json.loads(WP10C8C_OUTPUT.read_text(encoding="utf-8"))
    arrays = ROOT / str(evidence.get("artifacts", {}).get("arrays_path", ""))
    if not (
        evidence.get("work_package") == "WP10c8c"
        and evidence.get("decision")
        == "wp10c8c_region_selective_reduction_not_authorized"
        and evidence.get("next_authorization")
        == "retain_full_causal_dae_and_design_alternative_secular_coordinates"
        and not evidence.get("gates", {}).get(
            "region_selective_candidate_authorized",
            True,
        )
        and arrays.exists()
        and _sha256(arrays)
        == evidence.get("artifacts", {}).get("arrays_sha256")
    ):
        raise RuntimeError("WP10c8c did not authorize alternative coordinates")
    return evidence, _sha256(WP10C8C_OUTPUT)


def _load_states() -> tuple[dict, dict, dict]:
    (
        spectral,
        spectral_sha256,
        reference,
        reference_sha256,
    ) = wp10c8b._validate_authorization()
    initial, wp10c7k_evidence, wp10c7k_sha256 = (
        wp10c8b._initial_bundles(reference)
    )
    vectors = {}
    provenance = {}
    for n_cells in RESOLUTIONS:
        parent, parent_entry = wp10c8b._parent_restart(
            initial[n_cells],
            n_cells,
            "production",
            wp10c7k_evidence,
            wp10c7k_sha256,
            reference,
        )
        at_125 = wp10c8b._load_snapshot(
            initial[n_cells],
            "production",
            "t_0p125",
            parent_entry,
            spectral_sha256,
            reference_sha256,
        )
        at_100 = wp10c8b._load_snapshot(
            initial[n_cells],
            "production",
            "t_0p10",
            parent_entry,
            spectral_sha256,
            reference_sha256,
        )
        vectors[n_cells] = {
            "t_0": np.asarray(initial[n_cells]["vector"], dtype=float),
            "t_0p05": np.asarray(parent.state_vector, dtype=float),
            "t_0p10": np.asarray(at_100.state_vector, dtype=float),
            "t_0p125": np.asarray(at_125.state_vector, dtype=float),
        }
        provenance[str(n_cells)] = {
            label: {
                "state_vector_sha256": _array_sha256(vector),
                "elapsed_time_seconds": dict(ANCHORS).get(
                    label,
                    1.0e-1,
                ),
            }
            for label, vector in vectors[n_cells].items()
        }
    return initial, vectors, {
        "wp10c8a_evidence_sha256": spectral_sha256,
        "wp10c7n_evidence_sha256": reference_sha256,
        "states": provenance,
    }


def _state_weights(initial: dict) -> np.ndarray:
    measures = np.asarray(
        initial["context"].grid.cell_measures,
        dtype=float,
    )
    normalized = measures / float(np.mean(measures))
    return np.repeat(normalized, 5)


def _normalize_columns(
    values: np.ndarray,
    state_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(values, dtype=float)
    norms = np.sqrt(np.sum(state_weights[:, None] * matrix**2, axis=0))
    if np.any(~np.isfinite(norms)) or np.any(norms <= np.finfo(float).tiny):
        raise RuntimeError("WP10c8d encountered a zero perturbation direction")
    return matrix / norms[None, :], norms


def _perturbation_directions(
    initial: dict,
    before_vector: np.ndarray,
    vector: np.ndarray,
    primitive_scales: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    n_cells = initial["state"].n_cells
    before = unpack_causal_five_field_state(before_vector, n_cells)
    current = unpack_causal_five_field_state(vector, n_cells)
    _, diagnostics = wp10c8b._off_manifold_diagnostics(initial, vector)
    radius_rg = np.asarray(diagnostics["radius_rg"], dtype=float)
    trajectory = (
        np.asarray(current.primitives - before.primitives, dtype=float).ravel()
        / primitive_scales
    )
    thermal = np.zeros((n_cells, 5), dtype=float)
    thermal[:, 3] = 1.0e-2 * wp10c8c._smooth_window(
        radius_rg,
        6.0,
        60.0,
    )
    density = np.zeros((n_cells, 5), dtype=float)
    density[:, 0] = 1.0e-2 * wp10c8c._smooth_window(
        radius_rg,
        6.0,
        60.0,
    )
    source_band = np.zeros((n_cells, 5), dtype=float)
    source_window = wp10c8c._smooth_window(radius_rg, 200.0, 280.0)
    source_band[:, 0] = 1.0e-2 * source_window
    source_band[:, 2] = 5.0e-3 * source_window
    stress = np.zeros((n_cells, 5), dtype=float)
    stress[:, 4] = (
        diagnostics["target_specific_stress"]
        - diagnostics["specific_stress"]
    )
    training = {
        "trajectory_secant": trajectory,
        "thermal_6_to_60rg": thermal.ravel() / primitive_scales,
        "surface_density_6_to_60rg": (
            density.ravel() / primitive_scales
        ),
        "source_band_loading_200_to_280rg": (
            source_band.ravel() / primitive_scales
        ),
        "stress_target_adjustment": stress.ravel() / primitive_scales,
    }

    outer_thermal = np.zeros((n_cells, 5), dtype=float)
    outer_thermal[:, 3] = 1.0e-2 * wp10c8c._smooth_window(
        radius_rg,
        60.0,
        200.0,
    )
    broad_density = np.zeros((n_cells, 5), dtype=float)
    broad_density[:, 0] = 1.0e-2 * wp10c8c._smooth_window(
        radius_rg,
        20.0,
        200.0,
    )
    azimuthal = np.zeros((n_cells, 5), dtype=float)
    azimuthal[:, 2] = 2.0e-3 * wp10c8c._smooth_window(
        radius_rg,
        6.0,
        60.0,
    )
    held_out = {
        "thermal_60_to_200rg": (
            outer_thermal.ravel() / primitive_scales
        ),
        "surface_density_20_to_200rg": (
            broad_density.ravel() / primitive_scales
        ),
        "azimuthal_velocity_6_to_60rg": (
            azimuthal.ravel() / primitive_scales
        ),
    }
    weights = _state_weights(initial)
    training_values, _ = _normalize_columns(
        np.column_stack(tuple(training.values())),
        weights,
    )
    held_values, _ = _normalize_columns(
        np.column_stack(tuple(held_out.values())),
        weights,
    )
    return (
        {
            name: training_values[:, index]
            for index, name in enumerate(training)
        },
        {
            name: held_values[:, index]
            for index, name in enumerate(held_out)
        },
    )


def _output_operators(
    initial: dict,
    vector: np.ndarray,
    reduced: dict,
) -> dict:
    operators = wp10c8a._observable_operators(initial, vector, reduced)
    context = initial["context"]
    radius_rg = (
        np.asarray(context.grid.centers, dtype=float)
        / context.grid.gravitational_radius
    )
    measures = np.asarray(context.grid.cell_measures, dtype=float)
    log_h = np.asarray(operators["log_h_over_r_profile"], dtype=float)
    rows = [
        operators["cooling_relative"],
        operators["cooling_outside_6rg_relative"],
        operators["inner_accretion_relative"],
    ]
    names = [
        "cooling_relative",
        "cooling_outside_6rg_relative",
        "inner_accretion_relative",
    ]
    gates = [
        CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1[
            "cooling_power_proxy_relative"
        ],
        CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1[
            "cooling_power_proxy_outside_cutoff_relative"
        ],
        CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1[
            "inner_accretion_rate_relative"
        ],
    ]
    for target in H_OVER_R_SAMPLE_RADII_RG:
        index = int(np.argmin(np.abs(radius_rg - target)))
        rows.append(log_h[index])
        names.append(f"log_h_over_r_at_{target:g}rg")
        gates.append(
            CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1[
                "maximum_log_h_over_r_profile"
            ]
        )
    for name, lower, upper in H_OVER_R_BANDS_RG:
        mask = (radius_rg >= lower) & (radius_rg < upper)
        local_weights = measures[mask] / float(np.sum(measures[mask]))
        rows.append(np.sum(local_weights[:, None] * log_h[mask], axis=0))
        names.append(f"log_h_over_r_moment_{name}")
        gates.append(
            CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1[
                "maximum_log_h_over_r_profile"
            ]
        )
    ledger = np.vstack(
        (
            operators["integrated_mass_relative"],
            operators["integrated_angular_momentum_relative"],
            operators["integrated_killing_energy_relative"],
        )
    )
    for index, name in enumerate(
        (
            "integrated_mass_relative",
            "integrated_angular_momentum_relative",
            "integrated_killing_energy_relative",
        )
    ):
        rows.append(ledger[index])
        names.append(name)
        gates.append(
            CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1[
                "maximum_integrated_conserved_relative"
            ]
        )
    output = np.vstack(rows)
    gate_values = np.asarray(gates, dtype=float)
    return {
        "names": tuple(names),
        "matrix": output,
        "balanced_matrix": output / gate_values[:, None],
        "gates": gate_values,
        "protected": ledger,
        "log_h_over_r_profile": log_h,
    }


def _cache_path(n_cells: int, label: str) -> Path:
    return CACHE_DIRECTORY / f"N{n_cells:03d}_{label}_descriptor.npz"


def _descriptor_payload(
    initial: dict,
    before_vector: np.ndarray,
    vector: np.ndarray,
) -> tuple[dict, dict]:
    context = initial["context"]
    started = time.perf_counter()
    reduced = causal_five_field_reduced_descriptor_matrices(
        context,
        vector,
    )
    descriptor_wall = time.perf_counter() - started
    source = context.stream_sources
    if source is None:
        raise RuntimeError("WP10c8d requires the exact linked stream source")
    row_inputs, input_names = causal_stream_descriptor_inputs(
        source.weighted_killing_source_per_ct,
        reduced["conservation_row_scales"],
    )
    dynamic, physical_inputs, solve_defect = (
        causal_descriptor_explicit_matrices(
            reduced["descriptor_reduced_scaled_matrix"],
            reduced["stationary_reduced_scaled_jacobian"],
            row_inputs,
        )
    )
    weights = _state_weights(initial)
    basis_inputs, input_norms = _normalize_columns(physical_inputs, weights)
    outputs = _output_operators(initial, vector, reduced)
    training, held_out = _perturbation_directions(
        initial,
        before_vector,
        vector,
        np.asarray(reduced["primitive_column_scales"], dtype=float),
    )
    arrays = {
        "descriptor": reduced["descriptor_reduced_scaled_matrix"],
        "stationary": reduced["stationary_reduced_scaled_jacobian"],
        "dynamic": dynamic,
        "basis_inputs": basis_inputs,
        "physical_inputs": physical_inputs,
        "input_norms": input_norms,
        "output_matrix": outputs["matrix"],
        "balanced_output_matrix": outputs["balanced_matrix"],
        "output_gates": outputs["gates"],
        "protected_operators": outputs["protected"],
        "log_h_over_r_profile": outputs["log_h_over_r_profile"],
        "primitive_column_scales": reduced["primitive_column_scales"],
        "state_weights": weights,
        "training_directions": np.column_stack(tuple(training.values())),
        "held_out_directions": np.column_stack(tuple(held_out.values())),
    }
    metadata = {
        "state_vector_sha256": _array_sha256(vector),
        "descriptor_dimensions": reduced["dimensions"],
        "descriptor_rank": int(
            np.linalg.matrix_rank(
                reduced["descriptor_reduced_scaled_matrix"]
            )
        ),
        "descriptor_condition_estimate": float(
            np.linalg.cond(reduced["descriptor_reduced_scaled_matrix"])
        ),
        "explicit_solve_relative_defect": solve_defect,
        "algebraic_solve_relative_defect": reduced[
            "algebraic_solve_relative_defect"
        ],
        "maximum_scaled_algebraic_reconstruction_defect": reduced[
            "maximum_scaled_algebraic_reconstruction_defect"
        ],
        "descriptor_wall_seconds": descriptor_wall,
        "input_names": input_names,
        "training_direction_names": tuple(training),
        "held_out_direction_names": tuple(held_out),
        "output_names": outputs["names"],
    }
    return arrays, metadata


def _load_or_build_descriptor(
    initial: dict,
    before_vector: np.ndarray,
    vector: np.ndarray,
    label: str,
    *,
    force: bool,
) -> tuple[dict, dict, dict]:
    n_cells = initial["state"].n_cells
    path = _cache_path(n_cells, label)
    expected_state = _array_sha256(vector)
    if path.exists() and not force:
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata_json"].item()))
            if (
                metadata.get("work_package") == "WP10c8d"
                and metadata.get("base_commit") == BASE_COMMIT
                and metadata.get("state_vector_sha256") == expected_state
            ):
                arrays = {
                    name: np.asarray(saved[name], dtype=float)
                    for name in saved.files
                    if name != "metadata_json"
                }
                return arrays, metadata, {
                    "path": _relative(path),
                    "sha256": _sha256(path),
                    "reused": True,
                }
    arrays, metadata = _descriptor_payload(
        initial,
        before_vector,
        vector,
    )
    metadata.update(
        {
            "work_package": "WP10c8d",
            "base_commit": BASE_COMMIT,
            "n_cells": n_cells,
            "anchor": label,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        metadata_json=np.asarray(json.dumps(_plain(metadata), sort_keys=True)),
        **arrays,
    )
    return arrays, metadata, {
        "path": _relative(path),
        "sha256": _sha256(path),
        "reused": False,
    }


def _relative_response_errors(
    output_matrix: np.ndarray,
    log_h_profile: np.ndarray,
    full: np.ndarray,
    reduced: np.ndarray,
) -> dict:
    full_output = np.einsum(
        "on,tnd->tod",
        output_matrix,
        full,
        optimize=True,
    )
    reduced_output = np.einsum(
        "on,tnd->tod",
        output_matrix,
        reduced,
        optimize=True,
    )
    denominator = np.max(np.abs(full_output), axis=(0, 2))
    active = denominator >= OUTPUT_ACTIVITY_FLOOR
    output_errors = np.zeros(denominator.shape, dtype=float)
    output_errors[active] = (
        np.max(
            np.abs(reduced_output - full_output),
            axis=(0, 2),
        )[active]
        / denominator[active]
    )
    full_h = np.einsum(
        "hn,tnd->thd",
        log_h_profile,
        full,
        optimize=True,
    )
    reduced_h = np.einsum(
        "hn,tnd->thd",
        log_h_profile,
        reduced,
        optimize=True,
    )
    h_denominator = max(
        float(np.max(np.abs(full_h))),
        OUTPUT_ACTIVITY_FLOOR,
    )
    return {
        "maximum_output_relative_error": float(np.max(output_errors)),
        "per_output_relative_error": output_errors,
        "active_outputs": active,
        "active_output_count": int(np.count_nonzero(active)),
        "maximum_log_h_over_r_profile_relative_error": float(
            np.max(np.abs(reduced_h - full_h)) / h_denominator
        ),
    }


def _state_response_error(
    state_weights: np.ndarray,
    full: np.ndarray,
    reduced: np.ndarray,
) -> float:
    full_norm = np.sqrt(
        np.sum(state_weights[None, :, None] * full**2, axis=1)
    )
    error_norm = np.sqrt(
        np.sum(
            state_weights[None, :, None] * (reduced - full) ** 2,
            axis=1,
        )
    )
    return float(
        np.max(error_norm)
        / max(float(np.max(full_norm)), OUTPUT_ACTIVITY_FLOOR)
    )


def _audit_rom_ladder(arrays: dict) -> tuple[dict, dict]:
    dynamic = arrays["dynamic"]
    basis_inputs = arrays["basis_inputs"]
    outputs = arrays["output_matrix"]
    balanced_outputs = arrays["balanced_output_matrix"]
    protected = arrays["protected_operators"]
    state_weights = arrays["state_weights"]
    training = arrays["training_directions"]
    held_out = arrays["held_out_directions"]
    maximum_order = max(ORDERS)
    started = time.perf_counter()
    full_rom = causal_conservation_constrained_balanced_rom(
        dynamic,
        basis_inputs,
        balanced_outputs,
        protected,
        order=maximum_order,
        horizon_seconds=CERTIFIED_HORIZON_SECONDS,
        state_weights=state_weights,
        initial_directions=training,
        sample_count=BALANCED_SNAPSHOT_COUNT,
        singular_value_tolerance=1.0e-14,
        allow_rank_truncation=True,
    )
    basis_wall = time.perf_counter() - started
    response_times = np.asarray(RESPONSE_TIMES_SECONDS, dtype=float)
    full_training = causal_linear_initial_response(
        dynamic,
        training,
        response_times,
    )
    full_held_out = causal_linear_initial_response(
        dynamic,
        held_out,
        response_times,
    )
    rows = {}
    roms = {}
    for order in ORDERS:
        if order > full_rom.order:
            rows[str(order)] = {
                "order": order,
                "available": False,
                "numerically_resolved_maximum_order": full_rom.order,
                "local_gate_passed": False,
            }
            continue
        rom = (
            full_rom
            if order == full_rom.order
            else causal_truncate_mixed_mode_rom(
                full_rom,
                dynamic,
                basis_inputs,
                balanced_outputs,
                order=order,
            )
        )
        roms[order] = rom
        reduced_training = causal_rom_initial_response(
            rom,
            training,
            response_times,
        )
        reduced_held_out = causal_rom_initial_response(
            rom,
            held_out,
            response_times,
        )
        training_errors = _relative_response_errors(
            outputs,
            arrays["log_h_over_r_profile"],
            full_training,
            reduced_training,
        )
        held_errors = _relative_response_errors(
            outputs,
            arrays["log_h_over_r_profile"],
            full_held_out,
            reduced_held_out,
        )
        maximum_real = float(np.max(np.real(eigvals(rom.dynamic_matrix))))
        linear_cost_fraction = float((order / dynamic.shape[0]) ** 2)
        passed = bool(
            maximum_real <= MAXIMUM_REDUCED_REAL_EIGENVALUE_PER_S
            and rom.biorthogonality_defect <= MAXIMUM_PROTECTED_DEFECT
            and rom.protected_value_defect <= MAXIMUM_PROTECTED_DEFECT
            and rom.protected_dynamics_defect <= MAXIMUM_PROTECTED_DEFECT
            and max(
                training_errors["maximum_output_relative_error"],
                training_errors[
                    "maximum_log_h_over_r_profile_relative_error"
                ],
            )
            <= MAXIMUM_TRAINING_RELATIVE_ERROR
            and max(
                held_errors["maximum_output_relative_error"],
                held_errors["maximum_log_h_over_r_profile_relative_error"],
            )
            <= MAXIMUM_HELD_OUT_RELATIVE_ERROR
            and linear_cost_fraction <= MAXIMUM_LINEAR_ONLINE_COST_FRACTION
        )
        rows[str(order)] = {
            "order": order,
            "available": True,
            "maximum_real_eigenvalue_per_s": maximum_real,
            "stable": bool(
                maximum_real <= MAXIMUM_REDUCED_REAL_EIGENVALUE_PER_S
            ),
            "training": {
                **training_errors,
                "state_metric_relative_error": _state_response_error(
                    state_weights,
                    full_training,
                    reduced_training,
                ),
            },
            "held_out": {
                **held_errors,
                "state_metric_relative_error": _state_response_error(
                    state_weights,
                    full_held_out,
                    reduced_held_out,
                ),
            },
            "biorthogonality_defect": rom.biorthogonality_defect,
            "protected_value_defect": rom.protected_value_defect,
            "protected_dynamics_defect": rom.protected_dynamics_defect,
            "linear_online_cost_fraction_estimate": linear_cost_fraction,
            "local_gate_passed": passed,
        }
    return {
        "basis_wall_seconds": basis_wall,
        "requested_maximum_order": maximum_order,
        "numerically_resolved_maximum_order": full_rom.order,
        "hankel_singular_values": full_rom.hankel_singular_values,
        "orders": rows,
    }, roms


def _restricted_fine_basis(
    coarse_initial: dict,
    fine_initial: dict,
    coarse_arrays: dict,
    fine_arrays: dict,
    fine_basis: np.ndarray,
) -> np.ndarray:
    fine_physical = (
        fine_arrays["primitive_column_scales"][:, None] * fine_basis
    ).reshape(fine_initial["state"].n_cells, 5, fine_basis.shape[1])
    restricted = causal_restrict_cell_averages(
        coarse_initial["context"].grid,
        fine_initial["context"].grid,
        fine_physical,
    )
    return (
        restricted.ravel().reshape(
            5 * coarse_initial["state"].n_cells,
            fine_basis.shape[1],
        )
        / coarse_arrays["primitive_column_scales"][:, None]
    )


def _cross_mesh_rows(
    initial: dict,
    descriptor_arrays: dict,
    roms: dict,
) -> dict:
    rows = {}
    coarse_weights = np.sqrt(descriptor_arrays[64]["state_weights"])
    for order in ORDERS:
        if (
            order not in roms[64]
            or order not in roms[128]
        ):
            rows[str(order)] = {
                "available": False,
                "passed": False,
            }
            continue
        coarse = coarse_weights[:, None] * roms[64][order].trial_basis
        fine_restricted = _restricted_fine_basis(
            initial[64],
            initial[128],
            descriptor_arrays[64],
            descriptor_arrays[128],
            roms[128][order].trial_basis,
        )
        fine = coarse_weights[:, None] * fine_restricted
        angles = np.degrees(subspace_angles(coarse, fine))
        rows[str(order)] = {
            "maximum_principal_angle_degrees": float(np.max(angles)),
            "median_principal_angle_degrees": float(np.median(angles)),
            "p95_principal_angle_degrees": float(
                np.percentile(angles, 95.0)
            ),
            "passed": bool(
                np.percentile(angles, 95.0)
                <= MAXIMUM_CROSS_MESH_95TH_ANGLE_DEGREES
            ),
        }
    return rows


def _memory_audit(
    descriptor_arrays: dict,
    roms: dict,
    order: int,
) -> tuple[dict, dict]:
    rows = {}
    arrays = {}
    times = np.asarray(MEMORY_TIMES_SECONDS, dtype=float)
    for n_cells in RESOLUTIONS:
        started = time.perf_counter()
        system = descriptor_arrays[n_cells]["dynamic"]
        rom = roms[n_cells][order]
        projection = rom.trial_basis @ rom.test_basis.T
        complement = np.eye(system.shape[0]) - projection
        unresolved_dynamic = complement @ system @ complement
        unresolved_eigenvalues = eigvals(unresolved_dynamic)
        nonzero = np.abs(unresolved_eigenvalues) > 1.0e-7
        maximum_real = float(
            np.max(np.real(unresolved_eigenvalues[nonzero]))
            if np.any(nonzero)
            else 0.0
        )
        if maximum_real > MAXIMUM_REDUCED_REAL_EIGENVALUE_PER_S:
            safe_limit = 20.0 / maximum_real
            selected = times <= safe_limit
        else:
            safe_limit = float(times[-1])
            selected = np.ones(times.shape, dtype=bool)
        evaluated_times = times[selected]
        try:
            kernel = causal_rom_memory_kernel_actions(
                system,
                rom,
                evaluated_times,
            )
            finite = bool(np.all(np.isfinite(kernel)))
            norms = np.asarray(
                [np.linalg.norm(item, ord=2) for item in kernel],
                dtype=float,
            )
            initial = max(float(norms[0]), OUTPUT_ACTIVITY_FLOOR)
            relative = norms / initial
            error = None
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            kernel = np.empty((0, 0, 0))
            norms = np.empty(0)
            relative = np.empty(0)
            finite = False
            error = str(exc)
        rows[str(n_cells)] = {
            "order": order,
            "finite": finite,
            "unresolved_maximum_real_eigenvalue_per_s": maximum_real,
            "unresolved_dynamics_stable": bool(
                maximum_real <= MAXIMUM_REDUCED_REAL_EIGENVALUE_PER_S
            ),
            "requested_times_seconds": times,
            "evaluated_times_seconds": evaluated_times,
            "safe_exponential_horizon_seconds": safe_limit,
            "long_horizon_evaluation_skipped": bool(
                evaluated_times.size < times.size
            ),
            "spectral_norms": norms,
            "relative_to_initial": relative,
            "wall_seconds": time.perf_counter() - started,
            "error": error,
        }
        if finite:
            arrays[f"n{n_cells}_memory_kernel"] = kernel
    return rows, arrays


def main() -> None:
    args = _arguments()
    output_path = _absolute(args.output)
    arrays_path = _absolute(args.arrays)
    authorization, authorization_sha256 = _validate_authorization()
    initial, vectors, provenance = _load_states()
    state_rows = {
        str(n_cells): {
            label: audit_causal_five_field_state_gates(
                initial[n_cells]["context"],
                vectors[n_cells][label],
            )["passed"]
            for label, _elapsed in ANCHORS
        }
        for n_cells in RESOLUTIONS
    }
    if args.preflight:
        print(
            json.dumps(
                {
                    "work_package": "WP10c8d",
                    "preflight_passed": all(
                        passed
                        for mesh in state_rows.values()
                        for passed in mesh.values()
                    ),
                    "wp10c8c_evidence_sha256": authorization_sha256,
                    "selected_state_gates": state_rows,
                },
                sort_keys=True,
            )
        )
        return

    descriptor_arrays = {n_cells: {} for n_cells in RESOLUTIONS}
    descriptor_rows = {str(n_cells): {} for n_cells in RESOLUTIONS}
    cache_rows = {str(n_cells): {} for n_cells in RESOLUTIONS}
    for n_cells in RESOLUTIONS:
        for label, _elapsed in ANCHORS:
            before_label = (
                "t_0"
                if label in ("t_0", "t_0p05")
                else "t_0p10"
            )
            before = vectors[n_cells][before_label]
            if label == "t_0":
                before = vectors[n_cells]["t_0p05"]
            arrays, metadata, cache = _load_or_build_descriptor(
                initial[n_cells],
                before,
                vectors[n_cells][label],
                label,
                force=args.force_descriptors,
            )
            descriptor_arrays[n_cells][label] = arrays
            descriptor_rows[str(n_cells)][label] = metadata
            cache_rows[str(n_cells)][label] = cache
            print(
                json.dumps(
                    {
                        "work_package": "WP10c8d",
                        "phase": "descriptor",
                        "n_cells": n_cells,
                        "anchor": label,
                        "cache_reused": cache["reused"],
                        "rank": metadata["descriptor_rank"],
                    },
                    sort_keys=True,
                )
            )
    if args.descriptors_only:
        print(
            json.dumps(
                {
                    "work_package": "WP10c8d",
                    "descriptors_ready": True,
                    "caches": cache_rows,
                },
                sort_keys=True,
            )
        )
        return

    reductions = {str(n_cells): {} for n_cells in RESOLUTIONS}
    rom_ladders = {n_cells: {} for n_cells in RESOLUTIONS}
    array_payload = {}
    for label, _elapsed in ANCHORS:
        for n_cells in RESOLUTIONS:
            summary, roms = _audit_rom_ladder(
                descriptor_arrays[n_cells][label]
            )
            reductions[str(n_cells)][label] = summary
            rom_ladders[n_cells][label] = roms
            array_payload[
                f"n{n_cells}_{label}_hankel_singular_values"
            ] = summary["hankel_singular_values"]
            print(
                json.dumps(
                    {
                        "work_package": "WP10c8d",
                        "phase": "mixed_mode_ladder",
                        "n_cells": n_cells,
                        "anchor": label,
                        "passing_orders": [
                            int(order)
                            for order, row in summary["orders"].items()
                            if row["local_gate_passed"]
                        ],
                    },
                    sort_keys=True,
                )
            )

    cross_mesh = {}
    compact_orders = []
    for label, _elapsed in ANCHORS:
        cross_mesh[label] = _cross_mesh_rows(
            initial,
            {
                n_cells: descriptor_arrays[n_cells][label]
                for n_cells in RESOLUTIONS
            },
            {
                n_cells: rom_ladders[n_cells][label]
                for n_cells in RESOLUTIONS
            },
        )
    for order in ORDERS:
        if (
            order <= PREFERRED_MAXIMUM_ORDER
            and all(
                reductions[str(n_cells)][label]["orders"][str(order)][
                    "local_gate_passed"
                ]
                for n_cells in RESOLUTIONS
                for label, _elapsed in ANCHORS
            )
            and all(
                cross_mesh[label][str(order)]["passed"]
                for label, _elapsed in ANCHORS
            )
        ):
            compact_orders.append(order)

    common_memory_orders = [
        order
        for order in ORDERS
        if order <= PREFERRED_MAXIMUM_ORDER
        and order in rom_ladders[64]["t_0p125"]
        and order in rom_ladders[128]["t_0p125"]
    ]
    if not common_memory_orders:
        raise RuntimeError("WP10c8d found no common mixed-mode memory order")
    memory_order = (
        min(compact_orders)
        if compact_orders
        else max(common_memory_orders)
    )
    memory, memory_arrays = _memory_audit(
        {
            n_cells: descriptor_arrays[n_cells]["t_0p125"]
            for n_cells in RESOLUTIONS
        },
        {
            n_cells: rom_ladders[n_cells]["t_0p125"]
            for n_cells in RESOLUTIONS
        },
        memory_order,
    )
    array_payload.update(memory_arrays)

    compact_basis_found = bool(compact_orders)
    decision = (
        "wp10c8d_compact_cross_mesh_markovian_basis_found"
        if compact_basis_found
        else "wp10c8d_compact_cross_mesh_markovian_basis_not_found"
    )
    next_authorization = (
        "linear_markovian_prototype_with_measured_memory_gate"
        if compact_basis_found
        else "stationary_branch_preflight_or_narrower_observable_audit"
    )
    payload = {
        "work_package": "WP10c8d",
        "base_commit": BASE_COMMIT,
        "decision": decision,
        "next_authorization": next_authorization,
        "scope": {
            "description": (
                "Conservation-constrained finite-horizon mixed-mode "
                "compressibility and memory-necessity audit"
            ),
            "resolutions": RESOLUTIONS,
            "anchors": dict(ANCHORS),
            "orders": ORDERS,
            "certified_horizon_seconds": CERTIFIED_HORIZON_SECONDS,
            "response_times_seconds": RESPONSE_TIMES_SECONDS,
            "memory_times_seconds": MEMORY_TIMES_SECONDS,
            "nonlinear_rom_implemented": False,
            "stationary_branch_solve_included": False,
        },
        "authorization": {
            "wp10c8c_decision": authorization["decision"],
            "wp10c8c_evidence_sha256": authorization_sha256,
        },
        "provenance": provenance,
        "selected_state_gates": state_rows,
        "descriptor_caches": cache_rows,
        "descriptors": descriptor_rows,
        "reductions": reductions,
        "cross_mesh_subspaces": cross_mesh,
        "compact_passing_orders": compact_orders,
        "memory_necessity_audit": {
            "selected_order": memory_order,
            "resolutions": memory,
            "interpretation": (
                "Frozen-state linear diagnostic only; not a nonlinear "
                "prediction over the long sampled horizons."
            ),
        },
        "online_complexity": {
            "linear_cost_fraction_gate": (
                MAXIMUM_LINEAR_ONLINE_COST_FRACTION
            ),
            "nonlinear_hyper_reduction_demonstrated": False,
            "exact_structural_ledger_hyper_reduction_demonstrated": False,
            "eventual_total_speedup_target": 1.0e5,
            "nonlinear_production_rom_authorized": False,
        },
        "gates": {
            "maximum_training_relative_error": (
                MAXIMUM_TRAINING_RELATIVE_ERROR
            ),
            "maximum_held_out_relative_error": (
                MAXIMUM_HELD_OUT_RELATIVE_ERROR
            ),
            "maximum_reduced_real_eigenvalue_per_s": (
                MAXIMUM_REDUCED_REAL_EIGENVALUE_PER_S
            ),
            "maximum_protected_defect": MAXIMUM_PROTECTED_DEFECT,
            "preferred_maximum_order": PREFERRED_MAXIMUM_ORDER,
            "maximum_cross_mesh_p95_angle_degrees": (
                MAXIMUM_CROSS_MESH_95TH_ANGLE_DEGREES
            ),
            "compact_cross_mesh_basis_found": compact_basis_found,
            "nonlinear_rom_authorized": False,
        },
        "artifacts": {
            "arrays_path": _relative(arrays_path),
        },
    }
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **array_payload)
    payload["artifacts"]["arrays_sha256"] = _sha256(arrays_path)
    _write_json(output_path, payload)
    print(
        json.dumps(
            {
                "work_package": "WP10c8d",
                "decision": decision,
                "compact_passing_orders": compact_orders,
                "memory_order": memory_order,
                "output": _relative(output_path),
                "arrays": _relative(arrays_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
