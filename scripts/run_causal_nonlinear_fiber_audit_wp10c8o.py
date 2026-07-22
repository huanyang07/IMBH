"""Run the WP10c8o exact nonlinear equal-coordinate fiber audit.

The package is deliberately fail-fast.  It first tests the strongest saved
N64 richest-coordinate direction and the independent face-58 Rusanov switch
witness.  One admissible exact-coordinate pair above the locked 0.25
half-spread gate is a binding counterexample to an instantaneous deterministic
34-coordinate Markov closure.  Only that physical counterexample is then
prolonged to N128; no new N128 direction is optimized.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np
import scipy

import run_causal_mixed_mode_reduction_audit_wp10c8d as wp10c8d
import run_causal_moment_sufficiency_audit_wp10c8i as wp10c8i
import run_causal_shell_closure_preflight_wp10c8h as wp10c8h
from imri_qpe.constants import C
from imri_qpe.layer3_minidisk_1d import (
    CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1,
    audit_causal_five_field_state_gates,
    causal_exact_equal_coordinate_lift_pair,
    causal_five_field_moment_coordinate_values,
    causal_five_field_observable_snapshot,
    causal_five_field_reconstruct_face_charts,
    causal_five_field_reduced_descriptor_matrices,
    causal_five_field_reduced_storage_action,
    causal_five_field_scaled_primitive_vector_field,
    causal_five_field_state_from_primitives,
    causal_gate_normalized_pair_half_spread,
    causal_rescale_descriptor_matrix,
    pack_causal_five_field_state,
    unpack_causal_five_field_state,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_dae_system import (
    causal_five_field_rusanov_control_diagnostics,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "4dc5cea0342d35135e31078669e7e71ba7d16cf9"
WORK_PACKAGE = "WP10c8o"
SCHEMA_VERSION = 1
THIS_RUNNER = "scripts/run_causal_nonlinear_fiber_audit_wp10c8o.py"
PARENT_C8N = (
    ROOT / "outputs/tables/causal_rusanov_candidate_screen_wp10c8n.json"
)
PARENT_C8N_ARRAYS = (
    ROOT
    / "outputs/tables/causal_rusanov_candidate_screen_wp10c8n_arrays.npz"
)
PARENT_C8I = (
    ROOT / "outputs/tables/causal_moment_sufficiency_audit_wp10c8i.json"
)
PARENT_C8I_ARRAYS = (
    ROOT / "outputs/tables/causal_moment_sufficiency_audit_wp10c8i_arrays.npz"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/tables/causal_nonlinear_fiber_audit_wp10c8o.json"
)
DEFAULT_ARRAYS = (
    ROOT / "outputs/tables/causal_nonlinear_fiber_audit_wp10c8o_arrays.npz"
)

LEVEL_INDEX = 4
LEVEL_NAME = "plus_targeted_shape_moments"
PRIMARY_ANCHOR = "t_0p025"
PRIMARY_ROLE = "construction"
LEADING_SEED_KEY = (
    "n64_t_0p025_plus_targeted_shape_moments_"
    "endpoint_h_0_leading_state"
)
WITNESS_SEED_KEY = "n64_t_0p025_nonlinear_witness_direction"
LEADING_MULTIPLIERS = (2.5e-4, 5.0e-4, 1.0e-3, 2.0e-3)
WITNESS_RADII = (
    0.005532380178387262,
    0.0058177345244303956,
    0.00582938164059542,
    0.006114735986638554,
)
COORDINATE_DEFECT_GATE = 1.0e-10
PAIR_COORDINATE_DEFECT_GATE = 2.0e-10
MAXIMUM_WEIGHTED_RADIUS = 1.0
MAXIMUM_POINTWISE_AMPLITUDE_RATIO = 1.0
MAXIMUM_CORRECTION_FRACTION = 0.25
MINIMUM_DIRECTION_COSINE = 0.99
MAXIMUM_CONSTRAINT_CONDITION_ESTIMATE = 1.0e10
INSTANTANEOUS_SCREEN_GATE = 0.25
CROSS_MESH_SPREAD_DISAGREEMENT_GATE = 0.10
INTERFACE_FLUX_RELATIVE_GATE = 1.0e-3
COORDINATE_RATE_WINDOW_SECONDS = 2.5e-2
COORDINATE_RATE_GATE = 1.0
RATE_DIRECTIONAL_SCALED_STEPS = (5.0e-5, 1.0e-4, 2.0e-4)
MAXIMUM_RATE_DIRECTIONAL_STABILITY_DEFECT = 5.0e-3
BRANCH_FROZEN_LOCAL_STEP = 1.0e-3
BRANCH_FROZEN_COARSE_STEP = 2.0e-3
MAXIMUM_VECTOR_FIELD_STEP_DEFECT = 5.0e-3
MAXIMUM_STORAGE_COMPONENT_DEFECT = 5.0e-10
MAXIMUM_DESCRIPTOR_SOLVE_DEFECT = 1.0e-10
MAXIMUM_FULL_SCHUR_DESCRIPTOR_DEFECT = 1.0e-8
MAXIMUM_FULL_SCHUR_ALGEBRAIC_ROW = 1.0e-9
MAXIMUM_FULL_SCHUR_SOLVE_DEFECT = 1.0e-10
MAXIMUM_FULL_SCHUR_RATE_DEFECT = 5.0e-3
MAXIMUM_DESCRIPTOR_CONDITION_ESTIMATE = 1.0e12
MAXIMUM_STORAGE_ACTION_DEFECT = 5.0e-5
MAXIMUM_STORAGE_ACTION_STEP_DEFECT = 5.0e-3
MAXIMUM_HEIGHT_FORBIDDEN_COMPONENT_DEFECT = 1.0e-12
MAXIMUM_ANCHOR_INTERFACE_SCALE_DEFECT = 1.0e-12
STORAGE_ACTION_DIFFERENCE_STEPS = (5.0e-5, 1.0e-4, 2.0e-4)
FACE58 = 58


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    """Return the canonical WP10c8i float-array digest."""

    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _relative_defect(left: np.ndarray, right: np.ndarray) -> float:
    first = np.asarray(left, dtype=float)
    second = np.asarray(right, dtype=float)
    scale = max(
        float(np.max(np.abs(first))),
        float(np.max(np.abs(second))),
        np.finfo(float).tiny,
    )
    return float(np.max(np.abs(first - second)) / scale)


def _load_parent_contract() -> tuple[dict, dict]:
    c8n = json.loads(PARENT_C8N.read_text(encoding="utf-8"))
    c8i = json.loads(PARENT_C8I.read_text(encoding="utf-8"))
    if not (
        c8n.get("decision")
        == "wp10c8n_possible_winner_screen_rejected_by_nonlinear_witness"
        and c8n.get("work_package") == "WP10c8n"
        and c8i.get("work_package") == "WP10c8i"
        and PARENT_C8N_ARRAYS.exists()
        and PARENT_C8I_ARRAYS.exists()
        and _sha256(PARENT_C8N_ARRAYS)
        == c8n.get("artifacts", {}).get("arrays_sha256")
        and _sha256(PARENT_C8I_ARRAYS)
        == c8i.get("artifacts", {}).get("arrays_sha256")
    ):
        raise RuntimeError("WP10c8i/WP10c8n parent evidence differs")
    return c8n, c8i


def _load_anchor_cache(
    n_cells: int,
    label: str,
    vector: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict, Path]:
    path = (
        ROOT
        / "outputs/checkpoints/causal_five_field_wp10c8i"
        / f"N{n_cells:03d}_{label}_moment_operators.npz"
    )
    with np.load(path, allow_pickle=False) as source:
        metadata = json.loads(str(source["metadata_json"].item()))
        arrays = {
            name: np.asarray(source[name], dtype=float)
            for name in source.files
            if name != "metadata_json"
        }
    if not (
        metadata.get("work_package") == "WP10c8i"
        and metadata.get("n_cells") == n_cells
        and metadata.get("anchor_label") == label
        and metadata.get("state_vector_sha256") == _array_sha256(vector)
        and arrays[f"level_{LEVEL_INDEX}_constraints"].shape
        == (34, 5 * n_cells)
    ):
        raise RuntimeError(f"N{n_cells} {label} parent operator differs")
    return arrays, metadata, path


def _rebuilt_state(context, primitives: np.ndarray):
    charts = np.asarray(primitives, dtype=float).reshape(-1, 5)
    state = causal_five_field_state_from_primitives(context, charts)
    return state, pack_causal_five_field_state(state)


def _coordinate_evaluator(context, shell_edges_rg: np.ndarray):
    def evaluate(primitives: np.ndarray) -> np.ndarray:
        _state, vector = _rebuilt_state(context, primitives)
        values = causal_five_field_moment_coordinate_values(
            context,
            vector,
            shell_edges_rg,
        )
        return np.asarray(
            values.level(LEVEL_NAME).coordinate_values,
            dtype=float,
        )

    return evaluate


def _scientific_output_values(
    *,
    context,
    vector: np.ndarray,
    baseline,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], np.ndarray]:
    cutoff = 6.0 * context.grid.gravitational_radius
    snapshot = causal_five_field_observable_snapshot(
        context,
        vector,
        cooling_inner_cutoff=cutoff,
    )
    log_h = np.log(np.asarray(snapshot.h_over_r, dtype=float))
    radius_rg = (
        np.asarray(context.grid.centers, dtype=float)
        / context.grid.gravitational_radius
    )
    measures = np.asarray(context.grid.cell_measures, dtype=float)
    values = [
        snapshot.cooling_power_proxy_erg_s
        / max(
            abs(baseline.cooling_power_proxy_erg_s),
            np.finfo(float).tiny,
        ),
        snapshot.cooling_power_proxy_outside_cutoff_erg_s
        / max(
            abs(baseline.cooling_power_proxy_outside_cutoff_erg_s),
            np.finfo(float).tiny,
        ),
        snapshot.inner_accretion_rate_g_s
        / max(
            abs(baseline.inner_accretion_rate_g_s),
            np.finfo(float).tiny,
        ),
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
    h_gate = CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1[
        "maximum_log_h_over_r_profile"
    ]
    for target in wp10c8d.H_OVER_R_SAMPLE_RADII_RG:
        index = int(np.argmin(np.abs(radius_rg - target)))
        values.append(float(log_h[index]))
        names.append(f"log_h_over_r_at_{target:g}rg")
        gates.append(h_gate)
    for name, lower, upper in wp10c8d.H_OVER_R_BANDS_RG:
        mask = (radius_rg >= lower) & (radius_rg < upper)
        local_weights = measures[mask] / float(np.sum(measures[mask]))
        values.append(float(local_weights @ log_h[mask]))
        names.append(f"log_h_over_r_moment_{name}")
        gates.append(h_gate)
    integrated = np.asarray(snapshot.integrated_conserved, dtype=float)
    baseline_integrated = np.asarray(
        baseline.integrated_conserved,
        dtype=float,
    )
    integrated_gate = CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1[
        "maximum_integrated_conserved_relative"
    ]
    for component, name in (
        (0, "integrated_mass_relative"),
        (2, "integrated_angular_momentum_relative"),
        (3, "integrated_killing_energy_relative"),
    ):
        values.append(
            float(
                integrated[component]
                / max(
                    abs(baseline_integrated[component]),
                    np.finfo(float).tiny,
                )
            )
        )
        names.append(name)
        gates.append(integrated_gate)
    return (
        np.asarray(values, dtype=float),
        np.asarray(gates, dtype=float),
        tuple(names),
        log_h,
    )


def _static_output_stack(
    *,
    context,
    vector: np.ndarray,
    baseline_snapshot,
    anchor_interface_scales: np.ndarray,
    shell_edges_rg: np.ndarray,
    common_interpolation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], dict[str, np.ndarray]]:
    scientific, scientific_gates, scientific_names, log_h = (
        _scientific_output_values(
            context=context,
            vector=vector,
            baseline=baseline_snapshot,
        )
    )
    h_gate = CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1[
        "maximum_log_h_over_r_profile"
    ]
    common_log_h = common_interpolation @ log_h
    coordinates = causal_five_field_moment_coordinate_values(
        context,
        vector,
        shell_edges_rg,
    )
    interface = (
        np.asarray(coordinates.interface_flux_values, dtype=float)
        / np.asarray(anchor_interface_scales, dtype=float)
    )
    values = np.concatenate((scientific, log_h, common_log_h, interface))
    gates = np.concatenate(
        (
            scientific_gates,
            np.full(log_h.size, h_gate),
            np.full(common_log_h.size, h_gate),
            np.full(interface.size, INTERFACE_FLUX_RELATIVE_GATE),
        )
    )
    names = (
        *scientific_names,
        *(f"native_log_h_over_r_cell_{index}" for index in range(log_h.size)),
        *(
            f"common_log_h_over_r_sample_{index}"
            for index in range(common_log_h.size)
        ),
        *coordinates.interface_flux_names,
    )
    return values, gates, tuple(names), {
        "scientific": scientific,
        "native_log_h_over_r": log_h,
        "common_log_h_over_r": common_log_h,
        "macro_interface_flux": interface,
    }


def _descriptor_rank_summary(matrix: np.ndarray) -> tuple[dict, np.ndarray]:
    values = np.asarray(matrix, dtype=float)
    singular = np.linalg.svd(values, compute_uv=False)
    largest = float(singular[0])
    smallest = float(singular[-1])
    tolerance = float(
        max(values.shape) * np.finfo(float).eps * largest
    )
    rank = int(np.count_nonzero(singular > tolerance))
    condition = float(largest / max(smallest, np.finfo(float).tiny))
    return {
        "rank": rank,
        "expected_rank": int(values.shape[0]),
        "rank_tolerance": tolerance,
        "largest_singular_value": largest,
        "smallest_singular_value": smallest,
        "condition_estimate": condition,
        "maximum_condition_estimate": (
            MAXIMUM_DESCRIPTOR_CONDITION_ESTIMATE
        ),
        "passed": bool(
            rank == values.shape[0]
            and condition <= MAXIMUM_DESCRIPTOR_CONDITION_ESTIMATE
        ),
    }, singular


def _binding_dae_storage_audit(
    *,
    context,
    primitives: np.ndarray,
    primitive_scales: np.ndarray,
    conservation_scales: np.ndarray,
    primary: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Certify the descriptor and vector storage on one decisive lift."""

    _state, exact_vector = _rebuilt_state(context, primitives)
    track_mass = np.asarray(
        primary["descriptor_reduced_scaled_matrix"],
        dtype=float,
    )
    track_mapped = np.asarray(
        primary["conserved_descriptor_reduced_scaled_matrix"],
        dtype=float,
    )
    track_height = np.asarray(
        primary["vertical_descriptor_reduced_scaled_matrix"],
        dtype=float,
    )
    track_rate = np.asarray(
        primary["scaled_primitive_rate_per_s"],
        dtype=float,
    ).ravel()
    physical_rate = np.asarray(
        primary["primitive_rate_per_s"],
        dtype=float,
    ).ravel()
    stationary = np.asarray(
        primary["scaled_stationary_residual"],
        dtype=float,
    ).ravel()
    balance = track_mass @ track_rate + stationary
    balance_scale = max(
        float(np.max(np.abs(track_mass @ track_rate))),
        float(np.max(np.abs(stationary))),
        np.finfo(float).tiny,
    )
    solve_defect = float(np.max(np.abs(balance)) / balance_scale)

    full_schur = causal_five_field_reduced_descriptor_matrices(
        context,
        exact_vector,
        finite_difference_step=2.0e-6,
        descriptor_timestep_seconds=1.0,
    )
    local_mass = np.asarray(
        full_schur["descriptor_reduced_scaled_matrix"],
        dtype=float,
    )
    local_primitive_scales = np.asarray(
        full_schur["primitive_column_scales"],
        dtype=float,
    )
    local_conservation_scales = np.asarray(
        full_schur["conservation_row_scales"],
        dtype=float,
    )
    schur_mass_anchor_scaled = causal_rescale_descriptor_matrix(
        local_mass,
        source_primitive_scales=local_primitive_scales,
        source_conservation_scales=local_conservation_scales,
        target_primitive_scales=primitive_scales,
        target_conservation_scales=conservation_scales,
    )
    schur_parity = _relative_defect(
        track_mass,
        schur_mass_anchor_scaled,
    )
    schur_rate = np.linalg.solve(schur_mass_anchor_scaled, -stationary)
    schur_rate_defect = _relative_defect(track_rate, schur_rate)

    track_rank, track_singular = _descriptor_rank_summary(track_mass)
    schur_rank, schur_singular = _descriptor_rank_summary(
        schur_mass_anchor_scaled
    )

    action_rows = []
    for difference_step in STORAGE_ACTION_DIFFERENCE_STEPS:
        action_rows.append(
            causal_five_field_reduced_storage_action(
                context,
                np.asarray(primitives, dtype=float),
                physical_rate,
                storage_difference_step=difference_step,
                storage_quadrature_order=4,
                storage_directional_step=1.0e-3,
                conserved_difference_order=2,
            )
        )
    selected_action = action_rows[1]
    path_total = np.asarray(
        selected_action["total_conservation_storage_per_ct"],
        dtype=float,
    ).ravel() / conservation_scales
    path_mapped = np.asarray(
        selected_action["conserved_storage_per_ct"],
        dtype=float,
    ).ravel() / conservation_scales
    path_height = np.asarray(
        selected_action["vertical_storage_per_ct"],
        dtype=float,
    ).ravel() / conservation_scales
    matrix_total = track_mass @ track_rate
    matrix_mapped = track_mapped @ track_rate
    matrix_height = track_height @ track_rate
    total_action_defect = _relative_defect(path_total, matrix_total)
    mapped_action_defect = _relative_defect(path_mapped, matrix_mapped)
    height_action_defect = _relative_defect(path_height, matrix_height)
    component_defect = _relative_defect(
        path_total,
        path_mapped + path_height,
    )
    action_step_defect = max(
        _relative_defect(
            np.asarray(
                action_rows[index]["total_conservation_storage_per_ct"],
                dtype=float,
            ).ravel()
            / conservation_scales,
            path_total,
        )
        for index in (0, 2)
    )
    vertical_physical = np.asarray(
        selected_action["vertical_storage_per_ct"],
        dtype=float,
    )
    reference = max(
        float(
            np.max(
                np.abs(
                    selected_action["total_conservation_storage_per_ct"]
                )
            )
        ),
        np.finfo(float).tiny,
    )
    forbidden_height_defect = float(
        max(
            np.max(np.abs(vertical_physical[:, 0])),
            np.max(np.abs(vertical_physical[:, 4])),
        )
        / reference
    )
    allowed_height_maxima = np.max(
        np.abs(vertical_physical[:, 1:4]),
        axis=0,
    )
    height_structure_passed = bool(
        forbidden_height_defect
        <= MAXIMUM_HEIGHT_FORBIDDEN_COMPONENT_DEFECT
        and np.all(allowed_height_maxima > 0.0)
    )
    schur_algebraic_passed = bool(
        full_schur["maximum_scaled_descriptor_algebraic_row"]
        <= MAXIMUM_FULL_SCHUR_ALGEBRAIC_ROW
        and full_schur["algebraic_solve_relative_defect"]
        <= MAXIMUM_FULL_SCHUR_SOLVE_DEFECT
        and full_schur["maximum_scaled_algebraic_reconstruction_defect"]
        <= MAXIMUM_FULL_SCHUR_SOLVE_DEFECT
    )
    passed = bool(
        solve_defect <= MAXIMUM_DESCRIPTOR_SOLVE_DEFECT
        and schur_parity <= MAXIMUM_FULL_SCHUR_DESCRIPTOR_DEFECT
        and schur_rate_defect <= MAXIMUM_FULL_SCHUR_RATE_DEFECT
        and schur_algebraic_passed
        and track_rank["passed"]
        and schur_rank["passed"]
        and total_action_defect <= MAXIMUM_STORAGE_ACTION_DEFECT
        and mapped_action_defect <= MAXIMUM_STORAGE_ACTION_DEFECT
        and height_action_defect <= MAXIMUM_STORAGE_ACTION_DEFECT
        and component_defect <= MAXIMUM_STORAGE_COMPONENT_DEFECT
        and action_step_defect <= MAXIMUM_STORAGE_ACTION_STEP_DEFECT
        and height_structure_passed
    )
    return {
        "descriptor_solve_relative_defect": solve_defect,
        "maximum_descriptor_solve_relative_defect": (
            MAXIMUM_DESCRIPTOR_SOLVE_DEFECT
        ),
        "descriptor_solve_maximum_absolute_residual": float(
            np.max(np.abs(balance))
        ),
        "full_schur_descriptor_relative_defect": schur_parity,
        "maximum_full_schur_descriptor_relative_defect": (
            MAXIMUM_FULL_SCHUR_DESCRIPTOR_DEFECT
        ),
        "full_schur_rate_relative_defect": schur_rate_defect,
        "maximum_full_schur_rate_relative_defect": (
            MAXIMUM_FULL_SCHUR_RATE_DEFECT
        ),
        "full_schur_algebraic_solve_relative_defect": float(
            full_schur["algebraic_solve_relative_defect"]
        ),
        "full_schur_algebraic_reconstruction_defect": float(
            full_schur["maximum_scaled_algebraic_reconstruction_defect"]
        ),
        "full_schur_descriptor_algebraic_row": float(
            full_schur["maximum_scaled_descriptor_algebraic_row"]
        ),
        "full_schur_algebraic_passed": schur_algebraic_passed,
        "track_descriptor": track_rank,
        "full_schur_descriptor_anchor_scaled": schur_rank,
        "total_storage_action_relative_defect": total_action_defect,
        "mapped_storage_action_relative_defect": mapped_action_defect,
        "responsive_height_action_relative_defect": height_action_defect,
        "maximum_storage_action_relative_defect": (
            MAXIMUM_STORAGE_ACTION_DEFECT
        ),
        "storage_component_reconstruction_defect": component_defect,
        "storage_action_step_defect": action_step_defect,
        "maximum_storage_action_step_defect": (
            MAXIMUM_STORAGE_ACTION_STEP_DEFECT
        ),
        "responsive_height_forbidden_component_defect": (
            forbidden_height_defect
        ),
        "maximum_responsive_height_forbidden_component_defect": (
            MAXIMUM_HEIGHT_FORBIDDEN_COMPONENT_DEFECT
        ),
        "responsive_height_allowed_component_maxima": (
            allowed_height_maxima
        ),
        "responsive_height_structure_passed": height_structure_passed,
        "passed": passed,
    }, {
        "track_descriptor_singular_values": track_singular,
        "full_schur_descriptor_singular_values": schur_singular,
        "track_scaled_primitive_rate_per_s": track_rate,
        "full_schur_scaled_primitive_rate_per_s": schur_rate,
        "path_total_storage_action": path_total,
        "path_mapped_storage_action": path_mapped,
        "path_responsive_height_storage_action": path_height,
        "matrix_total_storage_action": matrix_total,
        "matrix_mapped_storage_action": matrix_mapped,
        "matrix_responsive_height_storage_action": matrix_height,
    }


def _fresh_coordinate_rate(
    *,
    context,
    primitives: np.ndarray,
    coordinate_evaluator,
    primitive_scales: np.ndarray,
    conservation_scales: np.ndarray,
    coordinate_scales: np.ndarray,
    binding_dae_storage_audit: bool,
) -> tuple[np.ndarray, dict, dict[str, np.ndarray]]:
    arguments = {
        "primitive_column_scales": primitive_scales,
        "conservation_row_scales": conservation_scales,
        "mapped_storage_backend": "branch_frozen_local",
    }
    primary = causal_five_field_scaled_primitive_vector_field(
        context,
        primitives,
        branch_frozen_local_difference_step=BRANCH_FROZEN_LOCAL_STEP,
        **arguments,
    )
    coarse = causal_five_field_scaled_primitive_vector_field(
        context,
        primitives,
        branch_frozen_local_difference_step=BRANCH_FROZEN_COARSE_STEP,
        **arguments,
    )
    scaled_rate = np.asarray(
        primary["scaled_primitive_rate_per_s"],
        dtype=float,
    ).ravel()
    physical_rate = np.asarray(
        primary["primitive_rate_per_s"],
        dtype=float,
    ).ravel()
    coarse_scaled_rate = np.asarray(
        coarse["scaled_primitive_rate_per_s"],
        dtype=float,
    ).ravel()
    vector_field_step_defect = _relative_defect(
        scaled_rate,
        coarse_scaled_rate,
    )
    maximum_rate = max(
        float(np.max(np.abs(scaled_rate))),
        np.finfo(float).tiny,
    )
    normalized_rows = []
    physical_rows = []
    time_steps = []
    for scaled_step in RATE_DIRECTIONAL_SCALED_STEPS:
        dt = float(scaled_step / maximum_rate)
        plus = coordinate_evaluator(primitives + dt * physical_rate)
        minus = coordinate_evaluator(primitives - dt * physical_rate)
        rate = (plus - minus) / (2.0 * dt)
        physical_rows.append(rate)
        normalized_rows.append(
            COORDINATE_RATE_WINDOW_SECONDS * rate / coordinate_scales
        )
        time_steps.append(dt)
    normalized = np.asarray(normalized_rows, dtype=float)
    selected = normalized[1]
    rate_stability = max(
        _relative_defect(normalized[0], selected),
        _relative_defect(normalized[2], selected),
    )
    total = np.asarray(
        primary["descriptor_reduced_scaled_matrix"], dtype=float
    )
    components = (
        np.asarray(
            primary["conserved_descriptor_reduced_scaled_matrix"],
            dtype=float,
        )
        + np.asarray(
            primary["vertical_descriptor_reduced_scaled_matrix"],
            dtype=float,
        )
    )
    storage_defect = _relative_defect(total, components)
    dae_storage_audit = None
    dae_storage_arrays: dict[str, np.ndarray] = {}
    if binding_dae_storage_audit:
        dae_storage_audit, dae_storage_arrays = (
            _binding_dae_storage_audit(
                context=context,
                primitives=primitives,
                primitive_scales=primitive_scales,
                conservation_scales=conservation_scales,
                primary=primary,
            )
        )
    audit = {
        "vector_field_step_defect": vector_field_step_defect,
        "maximum_vector_field_step_defect": (
            MAXIMUM_VECTOR_FIELD_STEP_DEFECT
        ),
        "coordinate_rate_directional_stability_defect": rate_stability,
        "maximum_coordinate_rate_directional_stability_defect": (
            MAXIMUM_RATE_DIRECTIONAL_STABILITY_DEFECT
        ),
        "storage_component_reconstruction_defect": storage_defect,
        "maximum_storage_component_reconstruction_defect": (
            MAXIMUM_STORAGE_COMPONENT_DEFECT
        ),
        "directional_scaled_steps": list(RATE_DIRECTIONAL_SCALED_STEPS),
        "directional_time_steps_seconds": time_steps,
        "binding_dae_storage_audit_evaluated": bool(
            binding_dae_storage_audit
        ),
        "dae_storage_audit": dae_storage_audit,
        "passed": bool(
            vector_field_step_defect <= MAXIMUM_VECTOR_FIELD_STEP_DEFECT
            and rate_stability
            <= MAXIMUM_RATE_DIRECTIONAL_STABILITY_DEFECT
            and storage_defect <= MAXIMUM_STORAGE_COMPONENT_DEFECT
            and (
                not binding_dae_storage_audit
                or (
                    dae_storage_audit is not None
                    and dae_storage_audit["passed"]
                )
            )
        ),
    }
    return selected, audit, {
        "scaled_primitive_rate_per_s": scaled_rate,
        "physical_primitive_rate_per_s": physical_rate,
        "coordinate_rates_by_step": np.asarray(physical_rows),
        "normalized_coordinate_rates_by_step": normalized,
        **dae_storage_arrays,
    }


def _lift_row(lift) -> dict:
    return {
        "optimizer_success": lift.optimizer_success,
        "optimizer_status": lift.optimizer_status,
        "optimizer_message": lift.optimizer_message,
        "function_evaluations": lift.function_evaluations,
        "jacobian_evaluations": lift.jacobian_evaluations,
        "maximum_coordinate_defect": lift.maximum_coordinate_defect,
        "weighted_radius": lift.weighted_radius,
        "provisional_weighted_radius": lift.provisional_weighted_radius,
        "maximum_pointwise_amplitude_ratio": (
            lift.maximum_pointwise_amplitude_ratio
        ),
        "provisional_maximum_pointwise_amplitude_ratio": (
            lift.provisional_maximum_pointwise_amplitude_ratio
        ),
        "correction_fraction": lift.correction_fraction,
        "retained_seed_multiplier": lift.retained_seed_multiplier,
        "retained_seed_multiplier_defect": (
            lift.retained_seed_multiplier_defect
        ),
        "weighted_direction_cosine": lift.weighted_direction_cosine,
    }


def _state_audit(context, primitives: np.ndarray) -> tuple[dict, dict]:
    state, vector = _rebuilt_state(context, primitives)
    gates = audit_causal_five_field_state_gates(context, vector)
    reconstruction = causal_five_field_reconstruct_face_charts(
        context,
        np.asarray(primitives, dtype=float).reshape(-1, 5),
    )
    rusanov = causal_five_field_rusanov_control_diagnostics(
        context,
        np.asarray(primitives, dtype=float).reshape(-1, 5),
    )
    factors = np.asarray(reconstruction.admissibility_factors, dtype=float)
    controls = np.asarray(rusanov["control_codes"], dtype=int)
    speeds = np.asarray(
        rusanov["candidate_absolute_speeds_over_c"],
        dtype=float,
    )
    row = {
        "state_gates": gates,
        "minimum_reconstruction_admissibility_factor": float(
            np.min(factors)
        ),
        "maximum_reconstruction_admissibility_departure_from_unity": float(
            np.max(np.abs(factors - 1.0))
        ),
        "face58_control_code": int(controls[FACE58 - 1]),
        "face58_candidate_absolute_speeds_over_c": speeds[FACE58 - 1],
        "passed": bool(
            gates["passed"]
            and np.max(np.abs(factors - 1.0)) <= 1.0e-12
        ),
    }
    return row, {
        "state_vector": vector,
        "control_codes": controls,
        "candidate_absolute_speeds_over_c": speeds,
        "admissibility_factors": factors,
        "conserved": np.asarray(state.conserved, dtype=float),
        "weighted_face_fluxes_over_c": np.asarray(
            state.weighted_face_fluxes_over_c,
            dtype=float,
        ),
    }


def _build_pair(
    *,
    case_id: str,
    seed_name: str,
    seed_origin: str,
    seed_direction: np.ndarray,
    seed_multiplier: float,
    initial: dict,
    vector: np.ndarray,
    cache: dict[str, np.ndarray],
    shell_edges_rg: np.ndarray,
    require_face58_switch: bool,
) -> tuple[dict, dict[str, np.ndarray], dict]:
    context = initial["context"]
    n_cells = initial["state"].n_cells
    stored = unpack_causal_five_field_state(vector, n_cells)
    base_state, base_vector = _rebuilt_state(
        context,
        np.asarray(stored.primitives, dtype=float).ravel(),
    )
    coordinate_map = _coordinate_evaluator(context, shell_edges_rg)
    target_snapshot = causal_five_field_moment_coordinate_values(
        context,
        base_vector,
        shell_edges_rg,
    )
    target_level = target_snapshot.level(LEVEL_NAME)
    target = np.asarray(target_level.coordinate_values, dtype=float)
    coordinate_scales = np.asarray(
        cache[f"level_{LEVEL_INDEX}_coordinate_scales"],
        dtype=float,
    )
    cached_interface_scales = np.asarray(
        cache["interface_flux_scales"],
        dtype=float,
    )
    rebuilt_interface_scale_defect = _relative_defect(
        cached_interface_scales,
        np.asarray(target_snapshot.interface_flux_scales, dtype=float),
    )
    cached_target = np.asarray(
        cache[f"level_{LEVEL_INDEX}_coordinate_values"],
        dtype=float,
    )
    cached_target_defect = float(
        np.max(np.abs((target - cached_target) / coordinate_scales))
    )
    primitive_scales = np.asarray(
        cache["primitive_column_scales"], dtype=float
    )
    weights = np.asarray(cache["state_weights"], dtype=float)
    amplitudes = np.asarray(cache["physical_input_amplitudes"], dtype=float)
    constraints = np.asarray(
        cache[f"level_{LEVEL_INDEX}_constraints"], dtype=float
    )
    started = time.perf_counter()
    pair = causal_exact_equal_coordinate_lift_pair(
        base_primitive_vector=np.asarray(base_state.primitives).ravel(),
        primitive_column_scales=primitive_scales,
        state_weights=weights,
        physical_input_amplitudes=amplitudes,
        target_coordinate_values=target,
        target_coordinate_scales=coordinate_scales,
        constraint_matrix=constraints,
        seed_direction=seed_direction,
        seed_multiplier=seed_multiplier,
        coordinate_evaluator=coordinate_map,
    )
    minus_audit, minus_arrays = _state_audit(
        context, pair.minus.primitive_vector
    )
    plus_audit, plus_arrays = _state_audit(
        context, pair.plus.primitive_vector
    )
    baseline_snapshot = causal_five_field_observable_snapshot(
        context,
        base_vector,
        cooling_inner_cutoff=6.0 * context.grid.gravitational_radius,
    )
    radius_rg = context.grid.centers / context.grid.gravitational_radius
    grid_edges_rg = context.grid.edges / context.grid.gravitational_radius
    common_radius, common_interpolation = (
        wp10c8i._common_log_h_interpolation(radius_rg, grid_edges_rg)
    )
    minus_static, static_gates, static_names, minus_blocks = (
        _static_output_stack(
            context=context,
            vector=minus_arrays["state_vector"],
            baseline_snapshot=baseline_snapshot,
            anchor_interface_scales=cached_interface_scales,
            shell_edges_rg=shell_edges_rg,
            common_interpolation=common_interpolation,
        )
    )
    plus_static, plus_gates, plus_names, plus_blocks = _static_output_stack(
        context=context,
        vector=plus_arrays["state_vector"],
        baseline_snapshot=baseline_snapshot,
        anchor_interface_scales=cached_interface_scales,
        shell_edges_rg=shell_edges_rg,
        common_interpolation=common_interpolation,
    )
    if not (
        np.array_equal(static_gates, plus_gates)
        and static_names == plus_names
    ):
        raise RuntimeError("pair output schemas differ")
    static_spread = causal_gate_normalized_pair_half_spread(
        minus_static,
        plus_static,
        static_gates,
    )
    static_control = int(np.argmax(static_spread))
    intended_switch = bool(
        minus_audit["face58_control_code"]
        != plus_audit["face58_control_code"]
    )
    lift_valid = bool(
        pair.minus.optimizer_success
        and pair.plus.optimizer_success
        and pair.minus.maximum_coordinate_defect <= COORDINATE_DEFECT_GATE
        and pair.plus.maximum_coordinate_defect <= COORDINATE_DEFECT_GATE
        and pair.maximum_pairwise_coordinate_defect
        <= PAIR_COORDINATE_DEFECT_GATE
        and pair.minus.weighted_radius <= MAXIMUM_WEIGHTED_RADIUS
        and pair.plus.weighted_radius <= MAXIMUM_WEIGHTED_RADIUS
        and pair.minus.maximum_pointwise_amplitude_ratio
        <= MAXIMUM_POINTWISE_AMPLITUDE_RATIO
        and pair.plus.maximum_pointwise_amplitude_ratio
        <= MAXIMUM_POINTWISE_AMPLITUDE_RATIO
        and pair.minus.correction_fraction <= MAXIMUM_CORRECTION_FRACTION
        and pair.plus.correction_fraction <= MAXIMUM_CORRECTION_FRACTION
        and pair.minus.weighted_direction_cosine >= MINIMUM_DIRECTION_COSINE
        and pair.plus.weighted_direction_cosine >= MINIMUM_DIRECTION_COSINE
        and pair.normal_basis.condition_estimate
        <= MAXIMUM_CONSTRAINT_CONDITION_ESTIMATE
        and rebuilt_interface_scale_defect
        <= MAXIMUM_ANCHOR_INTERFACE_SCALE_DEFECT
        and minus_audit["passed"]
        and plus_audit["passed"]
        and (not require_face58_switch or intended_switch)
    )
    row = {
        "case_id": case_id,
        "n_cells": n_cells,
        "anchor_label": PRIMARY_ANCHOR,
        "anchor_role": PRIMARY_ROLE,
        "seed_name": seed_name,
        "seed_origin": seed_origin,
        "seed_multiplier": float(seed_multiplier),
        "seed_sha256": _array_sha256(seed_direction),
        "cached_anchor_coordinate_defect": cached_target_defect,
        "rebuilt_anchor_interface_scale_defect": (
            rebuilt_interface_scale_defect
        ),
        "maximum_anchor_interface_scale_defect": (
            MAXIMUM_ANCHOR_INTERFACE_SCALE_DEFECT
        ),
        "normal_basis": {
            "numerical_rank": pair.normal_basis.numerical_rank,
            "condition_estimate": pair.normal_basis.condition_estimate,
            "weighted_orthogonality_defect": (
                pair.normal_basis.weighted_orthogonality_defect
            ),
            "row_space_reconstruction_defect": (
                pair.normal_basis.row_space_reconstruction_defect
            ),
        },
        "minus": _lift_row(pair.minus),
        "plus": _lift_row(pair.plus),
        "maximum_pairwise_coordinate_defect": (
            pair.maximum_pairwise_coordinate_defect
        ),
        "minus_state_audit": minus_audit,
        "plus_state_audit": plus_audit,
        "intended_face58_switch_required": require_face58_switch,
        "face58_opposite_controllers": intended_switch,
        "lift_valid": lift_valid,
        "static_output": {
            "output_count": len(static_names),
            "maximum_half_spread": float(np.max(static_spread)),
            "controlling_output_index": static_control,
            "controlling_output": static_names[static_control],
            "counterexample": bool(
                lift_valid
                and float(np.max(static_spread))
                > INSTANTANEOUS_SCREEN_GATE
            ),
        },
        "fresh_rate_output_evaluated": False,
        "full_output": None,
        "wall_seconds": time.perf_counter() - started,
    }
    arrays = {
        "seed_direction": np.asarray(seed_direction, dtype=float),
        "projected_seed_direction": pair.projected_seed_direction,
        "coordinate_names": np.asarray(
            target_level.coordinate_names,
            dtype="U",
        ),
        "target_coordinates": target,
        "coordinate_scales": coordinate_scales,
        "interface_flux_scales": cached_interface_scales,
        "minus_coordinates": pair.minus.coordinate_values,
        "plus_coordinates": pair.plus.coordinate_values,
        "minus_scaled_increment": pair.minus.scaled_increment,
        "plus_scaled_increment": pair.plus.scaled_increment,
        "minus_primitive_vector": pair.minus.primitive_vector,
        "plus_primitive_vector": pair.plus.primitive_vector,
        "minus_state_vector": minus_arrays["state_vector"],
        "plus_state_vector": plus_arrays["state_vector"],
        "minus_static_outputs": minus_static,
        "plus_static_outputs": plus_static,
        "static_output_names": np.asarray(static_names, dtype="U"),
        "static_output_gates": static_gates,
        "static_output_half_spreads": static_spread,
        "common_log_h_radius_rg": common_radius,
        "minus_native_log_h_over_r": minus_blocks[
            "native_log_h_over_r"
        ],
        "plus_native_log_h_over_r": plus_blocks["native_log_h_over_r"],
        "minus_common_log_h_over_r": minus_blocks[
            "common_log_h_over_r"
        ],
        "plus_common_log_h_over_r": plus_blocks[
            "common_log_h_over_r"
        ],
        "minus_interface_flux": minus_blocks["macro_interface_flux"],
        "plus_interface_flux": plus_blocks["macro_interface_flux"],
        "minus_control_codes": minus_arrays["control_codes"],
        "plus_control_codes": plus_arrays["control_codes"],
    }
    runtime = {
        "context": context,
        "coordinate_map": coordinate_map,
        "primitive_scales": primitive_scales,
        "conservation_scales": np.asarray(
            cache["conservation_row_scales"], dtype=float
        ),
        "coordinate_scales": coordinate_scales,
        "static_names": static_names,
        "static_gates": static_gates,
        "pair": pair,
        "arrays": arrays,
    }
    return row, arrays, runtime


def _complete_pair_rates(
    row: dict,
    runtime: dict,
    *,
    binding_dae_storage_audit: bool,
) -> None:
    pair = runtime["pair"]
    rate_rows = []
    rate_audits = []
    rate_arrays = []
    for lift in (pair.minus, pair.plus):
        rate, audit, arrays = _fresh_coordinate_rate(
            context=runtime["context"],
            primitives=lift.primitive_vector,
            coordinate_evaluator=runtime["coordinate_map"],
            primitive_scales=runtime["primitive_scales"],
            conservation_scales=runtime["conservation_scales"],
            coordinate_scales=runtime["coordinate_scales"],
            binding_dae_storage_audit=binding_dae_storage_audit,
        )
        rate_rows.append(rate)
        rate_audits.append(audit)
        rate_arrays.append(arrays)
    rate_gates = np.full(34, COORDINATE_RATE_GATE, dtype=float)
    minus_full = np.concatenate(
        (runtime["arrays"]["minus_static_outputs"], rate_rows[0])
    )
    plus_full = np.concatenate(
        (runtime["arrays"]["plus_static_outputs"], rate_rows[1])
    )
    full_gates = np.concatenate((runtime["static_gates"], rate_gates))
    full_names = (
        *runtime["static_names"],
        *(
            f"coordinate_rate_{index}"
            for index in range(rate_gates.size)
        ),
    )
    spread = causal_gate_normalized_pair_half_spread(
        minus_full,
        plus_full,
        full_gates,
    )
    control = int(np.argmax(spread))
    all_rates_pass = all(audit["passed"] for audit in rate_audits)
    all_binding_dae_storage_audits_pass = bool(
        binding_dae_storage_audit
        and all(
            audit["dae_storage_audit"] is not None
            and audit["dae_storage_audit"]["passed"]
            for audit in rate_audits
        )
    )
    row["fresh_rate_output_evaluated"] = True
    row["fresh_rate_audits"] = {
        "minus": rate_audits[0],
        "plus": rate_audits[1],
    }
    row["full_output"] = {
        "output_count": len(full_names),
        "maximum_half_spread": float(np.max(spread)),
        "controlling_output_index": control,
        "controlling_output": full_names[control],
        "all_fresh_rate_gates_passed": all_rates_pass,
        "binding_dae_storage_audits_evaluated": bool(
            binding_dae_storage_audit
        ),
        "all_binding_dae_storage_audits_passed": (
            all_binding_dae_storage_audits_pass
        ),
        "counterexample": bool(
            row["lift_valid"]
            and all_rates_pass
            and all_binding_dae_storage_audits_pass
            and float(np.max(spread)) > INSTANTANEOUS_SCREEN_GATE
        ),
    }
    runtime["arrays"].update(
        {
            "minus_coordinate_rate_output": rate_rows[0],
            "plus_coordinate_rate_output": rate_rows[1],
            "full_output_names": np.asarray(full_names, dtype="U"),
            "full_output_gates": full_gates,
            "full_output_half_spreads": spread,
            "minus_scaled_primitive_rate_per_s": rate_arrays[0][
                "scaled_primitive_rate_per_s"
            ],
            "plus_scaled_primitive_rate_per_s": rate_arrays[1][
                "scaled_primitive_rate_per_s"
            ],
            "minus_coordinate_rates_by_step": rate_arrays[0][
                "coordinate_rates_by_step"
            ],
            "plus_coordinate_rates_by_step": rate_arrays[1][
                "coordinate_rates_by_step"
            ],
            **{
                f"minus_{name}": value
                for name, value in rate_arrays[0].items()
                if name.startswith(("track_", "full_schur_", "path_", "matrix_"))
            },
            **{
                f"plus_{name}": value
                for name, value in rate_arrays[1].items()
                if name.startswith(("track_", "full_schur_", "path_", "matrix_"))
            },
        }
    )


def _prolong_decisive_physical_direction(
    runtime_n64: dict,
    multiplier: float,
    primitive_scales_n128: np.ndarray,
) -> np.ndarray:
    pair = runtime_n64["pair"]
    physical_direction_n64 = (
        pair.plus.primitive_vector - pair.minus.primitive_vector
    ) / (2.0 * multiplier)
    reshaped = physical_direction_n64.reshape(64, 5)
    prolonged = np.repeat(reshaped, 2, axis=0)
    if prolonged.shape != (128, 5):
        raise RuntimeError("N64-to-N128 direction prolongation failed")
    return prolonged.ravel() / primitive_scales_n128


def main() -> None:
    started = time.perf_counter()
    c8n, c8i = _load_parent_contract()
    initial_by_mesh, vectors_by_mesh, state_provenance = (
        wp10c8i._load_states()
    )
    shell_edges_rg = np.asarray(
        wp10c8h._common_shell_edges(initial_by_mesh)["five_shell"],
        dtype=float,
    )
    cache64, metadata64, cache64_path = _load_anchor_cache(
        64,
        PRIMARY_ANCHOR,
        vectors_by_mesh[64][PRIMARY_ANCHOR],
    )
    cache128, metadata128, cache128_path = _load_anchor_cache(
        128,
        PRIMARY_ANCHOR,
        vectors_by_mesh[128][PRIMARY_ANCHOR],
    )
    with np.load(PARENT_C8I_ARRAYS, allow_pickle=False) as parent_i_arrays:
        leading_seed = np.asarray(
            parent_i_arrays[LEADING_SEED_KEY], dtype=float
        )
    with np.load(PARENT_C8N_ARRAYS, allow_pickle=False) as parent_n_arrays:
        witness_seed = np.asarray(
            parent_n_arrays[WITNESS_SEED_KEY], dtype=float
        )

    rows: dict[str, dict] = {}
    runtimes: dict[str, dict] = {}
    all_arrays: dict[str, np.ndarray] = {}
    tier1 = (
        *(
            (
                f"n64_{PRIMARY_ANCHOR}_leading_alpha_{alpha:.4e}",
                "wp10c8i_richest_endpoint_h_0_leading_state",
                "WP10c8i conditional leading state used only as a predeclared seed",
                leading_seed,
                alpha,
                False,
            )
            for alpha in LEADING_MULTIPLIERS
        ),
        *(
            (
                f"n64_{PRIMARY_ANCHOR}_face58_radius_{radius:.9e}",
                "wp10c8n_face58_switch_witness",
                "WP10c8n exact anchor-null face-58 switch direction",
                witness_seed,
                radius,
                radius >= WITNESS_RADII[2],
            )
            for radius in WITNESS_RADII
        ),
    )
    for (
        case_id,
        seed_name,
        seed_origin,
        direction,
        multiplier,
        require_switch,
    ) in tier1:
        row, arrays, runtime = _build_pair(
            case_id=case_id,
            seed_name=seed_name,
            seed_origin=seed_origin,
            seed_direction=direction,
            seed_multiplier=multiplier,
            initial=initial_by_mesh[64],
            vector=vectors_by_mesh[64][PRIMARY_ANCHOR],
            cache=cache64,
            shell_edges_rg=shell_edges_rg,
            require_face58_switch=require_switch,
        )
        rows[case_id] = row
        runtimes[case_id] = runtime
        all_arrays.update(
            {f"{case_id}_{name}": value for name, value in arrays.items()}
        )

    leading_counterexamples = [
        case_id
        for case_id in rows
        if "leading_alpha" in case_id
        and rows[case_id]["static_output"]["counterexample"]
    ]
    if leading_counterexamples:
        decisive_id = min(
            leading_counterexamples,
            key=lambda key: rows[key]["seed_multiplier"],
        )
    else:
        any_counterexamples = [
            case_id
            for case_id, row in rows.items()
            if row["static_output"]["counterexample"]
        ]
        decisive_id = (
            max(
                any_counterexamples,
                key=lambda key: rows[key]["static_output"][
                    "maximum_half_spread"
                ],
            )
            if any_counterexamples
            else None
        )

    n128_id = None
    if decisive_id is None:
        # A static-output pass is not a complete instantaneous-fiber pass:
        # fresh coordinate-rate rows are part of the locked output contract.
        # Evaluate them for every Tier-1 pair before declaring the finite
        # matrix free of a counterexample.
        for case_id in rows:
            _complete_pair_rates(
                rows[case_id],
                runtimes[case_id],
                binding_dae_storage_audit=True,
            )
            all_arrays.update(
                {
                    f"{case_id}_{name}": value
                    for name, value in runtimes[case_id]["arrays"].items()
                }
            )
        full_counterexamples = [
            case_id
            for case_id, row in rows.items()
            if row["full_output"] is not None
            and row["full_output"]["counterexample"]
        ]
        decisive_id = (
            max(
                full_counterexamples,
                key=lambda key: rows[key]["full_output"][
                    "maximum_half_spread"
                ],
            )
            if full_counterexamples
            else None
        )

    if decisive_id is not None:
        if rows[decisive_id]["full_output"] is None:
            _complete_pair_rates(
                rows[decisive_id],
                runtimes[decisive_id],
                binding_dae_storage_audit=True,
            )
        all_arrays.update(
            {
                f"{decisive_id}_{name}": value
                for name, value in runtimes[decisive_id]["arrays"].items()
            }
        )
        multiplier = float(rows[decisive_id]["seed_multiplier"])
        prolonged = _prolong_decisive_physical_direction(
            runtimes[decisive_id],
            multiplier,
            np.asarray(cache128["primitive_column_scales"], dtype=float),
        )
        n128_id = f"n128_{PRIMARY_ANCHOR}_prolonged_decisive"
        row128, arrays128, runtime128 = _build_pair(
            case_id=n128_id,
            seed_name="prolonged_n64_decisive_physical_direction",
            seed_origin=(
                "piecewise-constant N64 physical corrected-pair half-difference "
                f"from {decisive_id}; no N128 output optimization"
            ),
            seed_direction=prolonged,
            seed_multiplier=multiplier,
            initial=initial_by_mesh[128],
            vector=vectors_by_mesh[128][PRIMARY_ANCHOR],
            cache=cache128,
            shell_edges_rg=shell_edges_rg,
            require_face58_switch=False,
        )
        _complete_pair_rates(
            row128,
            runtime128,
            binding_dae_storage_audit=True,
        )
        rows[n128_id] = row128
        runtimes[n128_id] = runtime128
        all_arrays.update(
            {
                f"{n128_id}_{name}": value
                for name, value in runtime128["arrays"].items()
            }
        )

    decisive_full_counterexample = bool(
        decisive_id is not None
        and rows[decisive_id]["full_output"] is not None
        and rows[decisive_id]["full_output"]["counterexample"]
    )
    n128_confirmation = bool(
        n128_id is not None
        and rows[n128_id]["full_output"] is not None
        and rows[n128_id]["full_output"]["counterexample"]
    )
    controlling_outputs_compatible = bool(
        decisive_full_counterexample
        and n128_confirmation
        and rows[decisive_id]["full_output"]["controlling_output"]
        == rows[n128_id]["full_output"]["controlling_output"]
    )
    cross_mesh_spread_disagreement = (
        abs(
            rows[decisive_id]["full_output"]["maximum_half_spread"]
            - rows[n128_id]["full_output"]["maximum_half_spread"]
        )
        if controlling_outputs_compatible
        else None
    )
    cross_mesh_spreads_compatible = bool(
        cross_mesh_spread_disagreement is not None
        and cross_mesh_spread_disagreement
        <= CROSS_MESH_SPREAD_DISAGREEMENT_GATE
    )
    counterexample_confirmed = bool(
        decisive_full_counterexample
        and n128_confirmation
        and controlling_outputs_compatible
        and cross_mesh_spreads_compatible
    )
    binding_dae_storage_audit_failed = bool(
        any(
            row["full_output"] is not None
            and row["full_output"][
                "binding_dae_storage_audits_evaluated"
            ]
            and not row["full_output"][
                "all_binding_dae_storage_audits_passed"
            ]
            for row in rows.values()
        )
    )
    tested_matrix_passed = bool(
        all(row["lift_valid"] for row in rows.values())
        and all(row["full_output"] is not None for row in rows.values())
        and all(
            row["full_output"][
                "all_binding_dae_storage_audits_passed"
            ]
            for row in rows.values()
        )
        and not any(
            row["full_output"]["counterexample"] for row in rows.values()
        )
    )
    decision = (
        "wp10c8o_exact_nonlinear_fiber_counterexample_confirmed_n64_n128"
        if counterexample_confirmed
        else (
            "wp10c8o_inconclusive_due_to_dae_storage_audit"
            if binding_dae_storage_audit_failed
            else (
                "wp10c8o_n64_counterexample_not_confirmed_at_n128"
                if decisive_full_counterexample
                else (
                    "wp10c8o_tier1_matrix_has_no_instantaneous_counterexample"
                    if tested_matrix_passed
                    else "wp10c8o_audit_incomplete_or_invalid"
                )
            )
        )
    )
    source_paths = (
        ROOT / THIS_RUNNER,
        ROOT / "src/imri_qpe/layer3_minidisk_1d/__init__.py",
        ROOT
        / "src/imri_qpe/layer3_minidisk_1d/causal_inner_dae_system.py",
        ROOT
        / "src/imri_qpe/layer3_minidisk_1d/causal_inner_moment_audit.py",
        ROOT
        / "src/imri_qpe/layer3_minidisk_1d/causal_inner_nonlinear_fiber.py",
        ROOT
        / "src/imri_qpe/layer3_minidisk_1d/causal_inner_spatial_audit.py",
        ROOT / "scripts/run_causal_mixed_mode_reduction_audit_wp10c8d.py",
        ROOT / "scripts/run_causal_shell_closure_preflight_wp10c8h.py",
        ROOT / "scripts/run_causal_moment_sufficiency_audit_wp10c8i.py",
        PARENT_C8I,
        PARENT_C8I_ARRAYS,
        PARENT_C8N,
        PARENT_C8N_ARRAYS,
        cache64_path,
        cache128_path,
    )
    output = {
        "schema_version": SCHEMA_VERSION,
        "base_commit": BASE_COMMIT,
        "work_package": WORK_PACKAGE,
        "scope": {
            "production_flux_changed": False,
            "production_descriptor_changed": False,
            "production_time_integrator_changed": False,
            "moment_ladder_changed": False,
            "new_truth_trajectory_run": False,
            "instantaneous_markov_closure_tested": True,
            "constrained_healing_run": False,
            "global_fiber_sufficiency_proven": False,
        },
        "authorization": {
            "wp10c8i": {
                "path": _relative(PARENT_C8I),
                "sha256": _sha256(PARENT_C8I),
                "arrays_path": _relative(PARENT_C8I_ARRAYS),
                "arrays_sha256": _sha256(PARENT_C8I_ARRAYS),
            },
            "wp10c8n": {
                "path": _relative(PARENT_C8N),
                "sha256": _sha256(PARENT_C8N),
                "arrays_path": _relative(PARENT_C8N_ARRAYS),
                "arrays_sha256": _sha256(PARENT_C8N_ARRAYS),
            },
        },
        "frozen_contract": {
            "coordinate_level": LEVEL_NAME,
            "coordinate_count": 34,
            "shell_edges_rg": shell_edges_rg,
            "leading_multipliers": LEADING_MULTIPLIERS,
            "face58_witness_radii": WITNESS_RADII,
            "output_semantics": "exact pair half-spread abs(Oplus-Ominus)/(2*gate)",
            "fresh_rate_semantics": (
                "0.025 s times centered derivative of the exact nonlinear "
                "coordinate-value map along the fresh branch-frozen Track-A "
                "primitive rate, divided by fixed anchor coordinate scales"
            ),
        },
        "gates": {
            "maximum_coordinate_defect": COORDINATE_DEFECT_GATE,
            "maximum_pair_coordinate_defect": PAIR_COORDINATE_DEFECT_GATE,
            "maximum_weighted_radius": MAXIMUM_WEIGHTED_RADIUS,
            "maximum_pointwise_amplitude_ratio": (
                MAXIMUM_POINTWISE_AMPLITUDE_RATIO
            ),
            "maximum_correction_fraction": MAXIMUM_CORRECTION_FRACTION,
            "minimum_direction_cosine": MINIMUM_DIRECTION_COSINE,
            "maximum_constraint_condition_estimate": (
                MAXIMUM_CONSTRAINT_CONDITION_ESTIMATE
            ),
            "instantaneous_screen_half_spread": INSTANTANEOUS_SCREEN_GATE,
            "maximum_cross_mesh_spread_disagreement": (
                CROSS_MESH_SPREAD_DISAGREEMENT_GATE
            ),
            "interface_flux_relative_gate": INTERFACE_FLUX_RELATIVE_GATE,
            "coordinate_rate_window_seconds": (
                COORDINATE_RATE_WINDOW_SECONDS
            ),
            "coordinate_rate_gate": COORDINATE_RATE_GATE,
        },
        "state_provenance": {
            "64": state_provenance["64"][PRIMARY_ANCHOR],
            "128": state_provenance["128"][PRIMARY_ANCHOR],
        },
        "operator_cache_provenance": {
            "64": {
                "path": _relative(cache64_path),
                "sha256": _sha256(cache64_path),
                "metadata_state_vector_sha256": metadata64[
                    "state_vector_sha256"
                ],
            },
            "128": {
                "path": _relative(cache128_path),
                "sha256": _sha256(cache128_path),
                "metadata_state_vector_sha256": metadata128[
                    "state_vector_sha256"
                ],
            },
        },
        "pairs": rows,
        "decisive_n64_pair": decisive_id,
        "n128_confirmation_pair": n128_id,
        "counterexample_found_n64": decisive_full_counterexample,
        "counterexample_confirmed_n128": n128_confirmation,
        "controlling_outputs_cross_mesh_compatible": (
            controlling_outputs_compatible
        ),
        "cross_mesh_spread_disagreement": cross_mesh_spread_disagreement,
        "cross_mesh_spreads_compatible": cross_mesh_spreads_compatible,
        "tested_matrix_passed": tested_matrix_passed,
        "binding_dae_storage_audit_failed": (
            binding_dae_storage_audit_failed
        ),
        "global_fiber_sufficiency_proven": False,
        "decision": decision,
        "next_action": (
            "close_raw_34_coordinate_markov_closure_and_classify_the_measured_fiber_direction_for_memory_or_one_targeted_coordinate"
            if counterexample_confirmed
            else (
                "diagnose_decisive_pair_dae_storage_audit_before_any_closure_claim"
                if binding_dae_storage_audit_failed
                else "do_not_claim_global_sufficiency_complete_the_predeclared_n64_fiber_matrix_or_diagnose_cross_mesh_failure"
            )
        ),
        "semantics": (
            "A valid exact-coordinate pair above 0.25 is a one-sided binding "
            "counterexample to instantaneous deterministic closure.  A finite "
            "matrix pass would authorize healing only and would not prove "
            "global fiber sufficiency.  Constrained healing was not run because "
            "the current production BDF API has no exact-coordinate constraint "
            "hook and any such burst is an augmented audit DAE with explicit "
            "constraint forcing, not the unmodified physical DAE."
        ),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "wall_seconds": time.perf_counter() - started,
        },
        "source_hashes": {
            _relative(path): _sha256(path) for path in source_paths
        },
    }
    DEFAULT_ARRAYS.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DEFAULT_ARRAYS, **all_arrays)
    output["artifacts"] = {
        "arrays_path": _relative(DEFAULT_ARRAYS),
        "arrays_sha256": _sha256(DEFAULT_ARRAYS),
    }
    DEFAULT_OUTPUT.write_text(
        json.dumps(_plain(output), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_plain(output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
