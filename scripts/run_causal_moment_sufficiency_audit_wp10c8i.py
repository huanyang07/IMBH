"""Audit storage-consistent internal moment sufficiency for WP10c8i.

This work package is deliberately an operator-level audit.  It does not
construct a reduced nonlinear evolution model, run new full-DAE trajectories,
or call a full N128 residual/Jacobian during any claimed ordinary reduced
evaluation.  The unresolved coordinate fiber is screened with exact frozen
finite-time propagation and a worst-case, gate-normalized singular-vector
search.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np
import scipy
from scipy.linalg import subspace_angles
from scipy.sparse.linalg import expm_multiply

import run_causal_ledger_equation_free_preflight_wp10c8g as wp10c8g
import run_causal_mixed_mode_reduction_audit_wp10c8d as wp10c8d
import run_causal_shell_closure_preflight_wp10c8h as wp10c8h
import run_causal_stress_time_audit_wp10c8b as wp10c8b
from imri_qpe.constants import C
from imri_qpe.layer3_minidisk_1d import (
    CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1,
    audit_causal_five_field_state_gates,
    causal_finite_time_output_operator,
    causal_five_field_evolving_tangent_matrices,
    causal_five_field_moment_coordinate_ladder,
    causal_five_field_reconstruct_face_charts,
    causal_five_field_reduced_descriptor_matrices,
    causal_five_field_scaled_primitive_vector_field,
    causal_five_field_state_from_primitives,
    causal_gate_normalized_finite_time_null_gain,
    causal_restrict_cell_averages,
    causal_weighted_constraint_null_basis,
    evaluate_causal_five_field_dae,
    load_causal_five_field_adaptive_bdf2_restart,
    pack_causal_five_field_state,
    unpack_causal_five_field_state,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_dae_system import (
    causal_five_field_rusanov_control_diagnostics,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "3e204d173a71f5c2ad02228e7c673601a7316e11"
WP10C8H_OUTPUT = (
    ROOT / "outputs/tables/causal_shell_closure_preflight_wp10c8h.json"
)
CACHE_DIRECTORY = (
    ROOT / "outputs/checkpoints/causal_five_field_wp10c8i"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/tables/causal_moment_sufficiency_audit_wp10c8i.json"
)
DEFAULT_ARRAYS = (
    ROOT
    / "outputs/tables/causal_moment_sufficiency_audit_wp10c8i_arrays.npz"
)
RESOLUTIONS = (64, 128)
ANCHORS = (
    ("t_0", 0.0, "construction"),
    ("t_0p025", 2.5e-2, "construction"),
    ("t_0p05", 5.0e-2, "construction"),
    ("t_0p075", 7.5e-2, "held_out"),
    ("t_0p10", 1.0e-1, "held_out"),
    ("t_0p125", 1.25e-1, "construction"),
)
FINITE_TIME_HORIZONS_SECONDS = (0.0, 1.0e-2, 2.5e-2)
RESPONSE_KINDS = ("endpoint", "increment")
CACHE_SCHEMA_VERSION = 9
FINITE_DIFFERENCE_STEP = 2.0e-6
DESCRIPTOR_TIMESTEP_SECONDS = 1.0
STORAGE_DIFFERENCE_STEP = 1.0e-4
STORAGE_RATE_DERIVATIVE_STEP = 2.0e-6
STORAGE_QUADRATURE_ORDER = 4
STORAGE_DIRECTIONAL_STEP = 1.0e-3
INNER_GENERATOR_STABILITY_STEPS = (1.0e-6, 2.0e-6, 4.0e-6)
OUTER_GENERATOR_STABILITY_STEPS = (1.0e-6, 2.0e-6, 4.0e-6)
STORAGE_ACTION_STABILITY_STEPS = (5.0e-5, 1.0e-4, 2.0e-4)
MAXIMUM_GENERATOR_STABILITY_RELATIVE_DEFECT = 5.0e-3
INDEPENDENT_VECTOR_FIELD_JVP_STEP = 3.0e-4
INDEPENDENT_VECTOR_FIELD_JVP_STABILITY_STEPS = (
    1.0e-4,
    3.0e-4,
    1.0e-3,
)
MAXIMUM_INDEPENDENT_VECTOR_FIELD_JVP_RELATIVE_DEFECT = 1.0e-2
MAXIMUM_INDEPENDENT_VECTOR_FIELD_JVP_ABSOLUTE_DEFECT = 1.0e-8
MAXIMUM_FORWARD_BACKWARD_JVP_RELATIVE_DEFECT = 2.0e-2
MAXIMUM_JVP_ADDITIVITY_RELATIVE_DEFECT = 2.0e-2
MINIMUM_RUSANOV_CONTROL_RELATIVE_MARGIN = 1.0e-8
RUSANOV_NUMERICAL_TIE_RELATIVE_MARGIN = 1.0e-14
MAXIMUM_SUPPRESSED_RUSANOV_SCALED_RELATIVE_JUMP = 1.0e-4
MAXIMUM_RUSANOV_GENERATOR_KINK_RELATIVE_DIAMETER = 5.0e-3
MAXIMUM_RUSANOV_FINITE_TIME_GATE_FRACTION = 1.0e-2
RUSANOV_FRECHET_QUADRATURE_ORDERS = (4, 8)
MAXIMUM_RUSANOV_FRECHET_QUADRATURE_RELATIVE_DEFECT = 5.0e-3
RUSANOV_SWITCHING_NORMAL_DIFFERENCE_STEP = 2.0e-5
JVP_ACTIVITY_FLOOR_PER_S = 1.0e-10
FULL_GENERATOR_STABILITY_ANCHORS = ("t_0", "t_0p10")
CROSS_MESH_GAIN_ACTIVITY_FLOOR = 1.0e-10
COMMON_LOG_H_CROSS_MESH_SAMPLE_COUNT = 129
OPERATOR_SOURCE_PATHS = tuple(
    dict.fromkeys(
        (
            Path(__file__).resolve(),
            *sorted(
                path
                for path in (ROOT / "src/imri_qpe").glob("**/*.py")
                if path.is_file()
            ),
            *sorted(
                (ROOT / "scripts").glob(
                    "run_causal_*wp10c8*.py"
                )
            ),
        )
    )
)

# The continuum-L2 input metric is cross-mesh normalized.  Separate
# pointwise box bounds below prevent inadmissible concentrated directions
# from deciding the audit.
PRIMITIVE_AMPLITUDE_POLICY = {
    "log_surface_density": 1.0e-2,
    "radial_three_velocity_over_c": 2.0e-3,
    "azimuthal_three_velocity_over_c": 2.0e-3,
    "log_temperature": 1.0e-2,
    "specific_causal_stress": (
        "one_percent_of_maximum_absolute_equilibrium_target_stress_"
        "with_robust_median_floor"
    ),
}

SCREENING_MAXIMUM_GATE_FRACTION = 0.25
PRE_MICROBURST_MAXIMUM_GATE_FRACTION = 0.10
MAXIMUM_ONLINE_COST_FRACTION = 0.10
MAXIMUM_STORAGE_ACTION_RELATIVE_DEFECT = 5.0e-5
MAXIMUM_GENERATOR_FACTORIZATION_DEFECT = 1.0e-8
MAXIMUM_STORAGE_COMPONENT_RECONSTRUCTION_DEFECT = 1.0e-10
MAXIMUM_DIRECT_TO_HISTORICAL_STORAGE_CHANGE = 1.0e-8
MAXIMUM_DIRECT_STORAGE_CONDITION_ESTIMATE = 1.0e12
MAXIMUM_COORDINATE_STORAGE_ROW_RELATIVE_DEFECT = 5.0e-5
MAXIMUM_NAMED_DIRECTION_CORRECTION_FRACTION = 0.25
ADMISSIBILITY_FACTOR_INACTIVE_ABSOLUTE_TOLERANCE = 1.0e-12
MAXIMUM_CONSTRAINT_CONDITION_ESTIMATE = 1.0e10
MAXIMUM_CROSS_MESH_GAIN_RELATIVE_DIFFERENCE = 0.25
MINIMUM_CROSS_MESH_LEADING_DIRECTION_COSINE = 0.50
CROSS_MESH_SUBSPACE_ANGLE_GATE_DEGREES = 45.0
INTERFACE_FLUX_RELATIVE_GATE = 1.0e-3
COORDINATE_RATE_WINDOW_SECONDS = 2.5e-2
COORDINATE_RATE_WINDOW_GATE = 1.0


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate authorization and every selected production state.",
    )
    parser.add_argument(
        "--operators-only",
        action="store_true",
        help="Build or validate all resumable WP10c8i operator caches.",
    )
    parser.add_argument(
        "--force-operators",
        action="store_true",
        help="Rebuild operator caches even when their provenance is valid.",
    )
    parser.add_argument(
        "--operator-resolution",
        action="append",
        type=int,
        choices=RESOLUTIONS,
        help=(
            "With --operators-only, build only this resolution. Repeat for "
            "multiple resolutions."
        ),
    )
    parser.add_argument(
        "--operator-anchor",
        action="append",
        choices=tuple(label for label, _time, _role in ANCHORS),
        help=(
            "With --operators-only, build only this anchor. Repeat for "
            "multiple anchors."
        ),
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


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stable_object_value(value):
    if isinstance(value, np.ndarray):
        return {
            "shape": value.shape,
            "sha256": _array_sha256(value),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_stable_object_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(name): _stable_object_value(item)
            for name, item in sorted(value.items())
        }
    if is_dataclass(value):
        contents = {
            field.name: _stable_object_value(getattr(value, field.name))
            for field in fields(value)
        }
    elif hasattr(value, "__dict__"):
        contents = {
            str(name): _stable_object_value(item)
            for name, item in sorted(vars(value).items())
        }
    else:
        raise TypeError(
            f"cannot form a stable WP10c8i contract for {type(value)!r}"
        )
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "contents": contents,
    }


def _operator_contract(context, shell_edges_rg: np.ndarray) -> dict:
    stream = context.stream_sources
    if stream is None:
        raise RuntimeError("WP10c8i requires the exact linked stream source")
    contract = {
        "code_sha256": {
            _relative(path): _sha256(path)
            for path in OPERATOR_SOURCE_PATHS
        },
        "runtime": {
            "python_version": sys.version,
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
        },
        "context": {
            "grid_edges_sha256": _array_sha256(context.grid.edges),
            "cell_measures_sha256": _array_sha256(
                context.grid.cell_measures
            ),
            "face_measures_sha256": _array_sha256(
                context.grid.face_measures
            ),
            "stream_source_sha256": _array_sha256(stream.matrix),
            "alpha": context.alpha,
            "stress_factor": context.stress_factor,
            "kappa": context.kappa,
            "include_radiative_cooling": (
                context.include_radiative_cooling
            ),
            "spatial_reconstruction": context.spatial_reconstruction,
            "boundary_trace_reconstruction": (
                context.boundary_trace_reconstruction
            ),
            "cell_rate_scheme": context.cell_rate_scheme,
            "cell_source_quadrature": context.cell_source_quadrature,
            "cell_storage_quadrature": context.cell_storage_quadrature,
            "vertical_frequency_provider": _stable_object_value(
                context.vertical_frequency
            ),
            "outer_boundary_provider": _stable_object_value(
                context.outer_boundary_provider
            ),
        },
        "finite_difference": {
            "finite_difference_step": FINITE_DIFFERENCE_STEP,
            "descriptor_timestep_seconds": DESCRIPTOR_TIMESTEP_SECONDS,
            "storage_difference_step": STORAGE_DIFFERENCE_STEP,
            "storage_rate_derivative_step": (
                STORAGE_RATE_DERIVATIVE_STEP
            ),
            "storage_quadrature_order": STORAGE_QUADRATURE_ORDER,
            "storage_directional_step": STORAGE_DIRECTIONAL_STEP,
        },
        "input": {
            "primitive_amplitude_policy": PRIMITIVE_AMPLITUDE_POLICY,
            "metric": (
                "cell_measure_normalized_continuum_l2_with_separate_"
                "pointwise_box_bounds"
            ),
        },
        "outputs": {
            "finite_time_horizons_seconds": (
                FINITE_TIME_HORIZONS_SECONDS
            ),
            "response_kinds": RESPONSE_KINDS,
            "interface_flux_relative_gate": (
                INTERFACE_FLUX_RELATIVE_GATE
            ),
            "coordinate_rate_window_seconds": (
                COORDINATE_RATE_WINDOW_SECONDS
            ),
            "coordinate_rate_window_gate": (
                COORDINATE_RATE_WINDOW_GATE
            ),
            "scientific_gates": (
                CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1
            ),
            "binding_norm": (
                "per_output_continuum_l2_with_pointwise_box_bounds"
            ),
            "screening_maximum_gate_fraction": (
                SCREENING_MAXIMUM_GATE_FRACTION
            ),
            "pre_microburst_maximum_gate_fraction": (
                PRE_MICROBURST_MAXIMUM_GATE_FRACTION
            ),
            "pointwise_amplitude_contract": (
                "continuum_l2_plus_rigorous_component_box_bounds"
            ),
        },
        "method_gates": {
            "maximum_storage_action_relative_defect": (
                MAXIMUM_STORAGE_ACTION_RELATIVE_DEFECT
            ),
            "maximum_storage_component_reconstruction_defect": (
                MAXIMUM_STORAGE_COMPONENT_RECONSTRUCTION_DEFECT
            ),
            "maximum_direct_to_historical_storage_change": (
                MAXIMUM_DIRECT_TO_HISTORICAL_STORAGE_CHANGE
            ),
            "maximum_direct_storage_condition_estimate": (
                MAXIMUM_DIRECT_STORAGE_CONDITION_ESTIMATE
            ),
            "maximum_generator_factorization_defect": (
                MAXIMUM_GENERATOR_FACTORIZATION_DEFECT
            ),
            "maximum_coordinate_storage_row_relative_defect": (
                MAXIMUM_COORDINATE_STORAGE_ROW_RELATIVE_DEFECT
            ),
            "admissibility_factor_inactive_absolute_tolerance": (
                ADMISSIBILITY_FACTOR_INACTIVE_ABSOLUTE_TOLERANCE
            ),
            "maximum_generator_stability_relative_defect": (
                MAXIMUM_GENERATOR_STABILITY_RELATIVE_DEFECT
            ),
            "inner_generator_stability_steps": (
                INNER_GENERATOR_STABILITY_STEPS
            ),
            "outer_generator_stability_steps": (
                OUTER_GENERATOR_STABILITY_STEPS
            ),
            "vertical_action_stability_steps": (
                STORAGE_ACTION_STABILITY_STEPS
            ),
            "full_generator_stability_anchors": (
                FULL_GENERATOR_STABILITY_ANCHORS
            ),
            "independent_vector_field_jvp_step": (
                INDEPENDENT_VECTOR_FIELD_JVP_STEP
            ),
            "maximum_independent_vector_field_jvp_relative_defect": (
                MAXIMUM_INDEPENDENT_VECTOR_FIELD_JVP_RELATIVE_DEFECT
            ),
            "maximum_independent_vector_field_jvp_absolute_defect": (
                MAXIMUM_INDEPENDENT_VECTOR_FIELD_JVP_ABSOLUTE_DEFECT
            ),
            "maximum_forward_backward_jvp_relative_defect": (
                MAXIMUM_FORWARD_BACKWARD_JVP_RELATIVE_DEFECT
            ),
            "maximum_jvp_additivity_relative_defect": (
                MAXIMUM_JVP_ADDITIVITY_RELATIVE_DEFECT
            ),
            "minimum_rusanov_control_relative_margin": (
                MINIMUM_RUSANOV_CONTROL_RELATIVE_MARGIN
            ),
            "rusanov_numerical_tie_relative_margin": (
                RUSANOV_NUMERICAL_TIE_RELATIVE_MARGIN
            ),
            "maximum_suppressed_rusanov_scaled_relative_jump": (
                MAXIMUM_SUPPRESSED_RUSANOV_SCALED_RELATIVE_JUMP
            ),
            "maximum_rusanov_generator_kink_relative_diameter": (
                MAXIMUM_RUSANOV_GENERATOR_KINK_RELATIVE_DIAMETER
            ),
            "maximum_rusanov_finite_time_gate_fraction": (
                MAXIMUM_RUSANOV_FINITE_TIME_GATE_FRACTION
            ),
            "rusanov_frechet_quadrature_orders": (
                RUSANOV_FRECHET_QUADRATURE_ORDERS
            ),
            "maximum_rusanov_frechet_quadrature_relative_defect": (
                MAXIMUM_RUSANOV_FRECHET_QUADRATURE_RELATIVE_DEFECT
            ),
            "jvp_activity_floor_per_s": JVP_ACTIVITY_FLOOR_PER_S,
            "maximum_constraint_condition_estimate": (
                MAXIMUM_CONSTRAINT_CONDITION_ESTIMATE
            ),
            "maximum_cross_mesh_gain_relative_difference": (
                MAXIMUM_CROSS_MESH_GAIN_RELATIVE_DIFFERENCE
            ),
            "cross_mesh_gain_activity_floor": (
                CROSS_MESH_GAIN_ACTIVITY_FLOOR
            ),
            "common_log_h_cross_mesh_sample_count": (
                COMMON_LOG_H_CROSS_MESH_SAMPLE_COUNT
            ),
            "minimum_cross_mesh_leading_direction_cosine": (
                MINIMUM_CROSS_MESH_LEADING_DIRECTION_COSINE
            ),
            "maximum_cross_mesh_admissible_subspace_angle_degrees": (
                CROSS_MESH_SUBSPACE_ANGLE_GATE_DEGREES
            ),
        },
        "moments": {
            "shell_edges_rg": np.asarray(
                shell_edges_rg,
                dtype=float,
            ),
            "incremental_order": (
                "instantaneous_shell_mje",
                "plus_shell_mean_log_temperature",
                "plus_shell_radial_momentum",
                "plus_shell_stress_storage",
                "plus_targeted_shape_moments",
            ),
        },
    }
    return _plain(contract)


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
    if not WP10C8H_OUTPUT.exists():
        raise RuntimeError("WP10c8i requires canonical WP10c8h evidence")
    evidence = json.loads(WP10C8H_OUTPUT.read_text(encoding="utf-8"))
    arrays = ROOT / str(evidence.get("artifacts", {}).get("arrays_path", ""))
    gates = evidence.get("gates", {})
    if not (
        evidence.get("work_package") == "WP10c8h"
        and evidence.get("decision")
        == "wp10c8h_compact_conservative_shell_closure_not_identifiable"
        and evidence.get("next_authorization")
        == "retain_full_dae_microbursts_and_reassess_physical_closure"
        and not gates.get("compact_shell_closure_found", True)
        and not gates.get("nonlinear_shell_microbursts_authorized", True)
        and not gates.get("nonlinear_macrostep_authorized", True)
        and arrays.exists()
        and _sha256(arrays)
        == evidence.get("artifacts", {}).get("arrays_sha256")
    ):
        raise RuntimeError(
            "WP10c8h does not authorize a bounded moment-sufficiency audit"
        )
    return evidence, _sha256(WP10C8H_OUTPUT)


def _t_0p025_path(n_cells: int) -> Path:
    package = "wp10c7l" if n_cells == 64 else "wp10c7n"
    return (
        ROOT
        / f"outputs/checkpoints/causal_five_field_{package}"
        / (
            f"causal_{package}_N{n_cells:03d}_production_"
            "t_0p025.npz"
        )
    )


def _load_states() -> tuple[dict, dict, dict]:
    initial, existing_vectors, existing_provenance = wp10c8d._load_states()
    restarts, restart_provenance = wp10c8g._checkpoint_restarts(initial)
    vectors: dict[int, dict[str, np.ndarray]] = {}
    provenance: dict[str, dict] = {}
    expected_times = {label: seconds for label, seconds, _ in ANCHORS}
    for n_cells in RESOLUTIONS:
        context = initial[n_cells]["context"]
        path_025 = _t_0p025_path(n_cells)
        restart_025 = load_causal_five_field_adaptive_bdf2_restart(
            path_025,
            context,
        )
        expected_package = "WP10c7l" if n_cells == 64 else "WP10c7n"
        if not (
            restart_025.elapsed_time == expected_times["t_0p025"]
            and restart_025.provenance.get("work_package")
            == expected_package
            and restart_025.provenance.get("trajectory_mode")
            == "production"
            and restart_025.provenance.get("n_cells") == n_cells
            and [row.get("label") for row in restart_025.provenance.get(
                "segments",
                [],
            )]
            == ["t_0p025"]
        ):
            raise RuntimeError(f"N{n_cells} t=0.025 checkpoint time differs")
        selected = {
            "t_0": np.asarray(
                existing_vectors[n_cells]["t_0"],
                dtype=float,
            ),
            "t_0p025": np.asarray(restart_025.state_vector, dtype=float),
            "t_0p05": np.asarray(
                restarts[n_cells]["production"]["t_0p05"].state_vector,
                dtype=float,
            ),
            "t_0p075": np.asarray(
                restarts[n_cells]["production"]["t_0p075"].state_vector,
                dtype=float,
            ),
            "t_0p10": np.asarray(
                restarts[n_cells]["production"]["t_0p10"].state_vector,
                dtype=float,
            ),
            "t_0p125": np.asarray(
                restarts[n_cells]["production"]["t_0p125"].state_vector,
                dtype=float,
            ),
        }
        mesh_provenance = {
            "t_0": dict(
                existing_provenance["states"][str(n_cells)]["t_0"]
            ),
            "t_0p025": {
                "path": _relative(path_025),
                "sha256": _sha256(path_025),
                "elapsed_time_seconds": restart_025.elapsed_time,
                "state_vector_sha256": _array_sha256(
                    restart_025.state_vector
                ),
            },
        }
        for label in ("t_0p05", "t_0p075", "t_0p10", "t_0p125"):
            restart = restarts[n_cells]["production"][label]
            if restart.elapsed_time != expected_times[label]:
                raise RuntimeError(
                    f"N{n_cells} {label} checkpoint time differs"
                )
            mesh_provenance[label] = dict(
                restart_provenance[str(n_cells)]["production"][label]
            )
        for label, vector in selected.items():
            if (
                _array_sha256(vector)
                != mesh_provenance[label]["state_vector_sha256"]
                or not audit_causal_five_field_state_gates(
                    context,
                    vector,
                )["passed"]
            ):
                raise RuntimeError(
                    f"N{n_cells} {label} state provenance or gates differ"
                )
        vectors[n_cells] = selected
        provenance[str(n_cells)] = mesh_provenance
    return initial, vectors, provenance


def _cache_path(n_cells: int, label: str) -> Path:
    return CACHE_DIRECTORY / f"N{n_cells:03d}_{label}_moment_operators.npz"


def _state_metric(
    initial: dict,
    vector: np.ndarray,
    primitive_scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    n_cells = initial["state"].n_cells
    scales = np.asarray(primitive_scales, dtype=float).reshape(n_cells, 5)
    amplitudes = np.empty_like(scales)
    amplitudes[:, 0] = float(
        PRIMITIVE_AMPLITUDE_POLICY["log_surface_density"]
    )
    amplitudes[:, 1] = float(
        PRIMITIVE_AMPLITUDE_POLICY["radial_three_velocity_over_c"]
    )
    amplitudes[:, 2] = float(
        PRIMITIVE_AMPLITUDE_POLICY["azimuthal_three_velocity_over_c"]
    )
    amplitudes[:, 3] = float(
        PRIMITIVE_AMPLITUDE_POLICY["log_temperature"]
    )
    _summary, diagnostics = wp10c8b._off_manifold_diagnostics(
        initial,
        vector,
    )
    target_stress = np.abs(
        np.asarray(
            diagnostics["target_specific_stress"],
            dtype=float,
        )
    )
    robust_floor = max(
        float(np.median(target_stress)),
        np.finfo(float).tiny,
    )
    target_reference = max(
        float(np.max(target_stress)),
        robust_floor,
    )
    stress_amplitude = 1.0e-2 * target_reference
    amplitudes[:, 4] = stress_amplitude
    measures = np.asarray(
        initial["context"].grid.cell_measures,
        dtype=float,
    )
    normalized_measures = measures / float(np.sum(measures))
    weights = (
        normalized_measures[:, None]
        * np.square(scales / amplitudes)
    )
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise RuntimeError("WP10c8i input metric is invalid")
    return weights.ravel(), amplitudes.ravel(), {
        "target_stress_maximum_absolute": float(
            np.max(target_stress)
        ),
        "target_stress_robust_median_floor": robust_floor,
        "specific_stress_input_amplitude": stress_amplitude,
    }


def _storage_audit(evolving: dict) -> dict:
    action = evolving["base_storage_action"]
    vertical = np.asarray(
        action["vertical_storage_per_ct"],
        dtype=float,
    )
    total = np.asarray(
        action["total_conservation_storage_per_ct"],
        dtype=float,
    )
    component_maxima = np.max(np.abs(vertical), axis=0)
    reference = max(
        float(np.max(np.abs(total))),
        np.finfo(float).tiny,
    )
    zero_component_defect = float(
        max(component_maxima[0], component_maxima[4]) / reference
    )
    total_mass = np.asarray(
        evolving["descriptor_reduced_scaled_matrix"],
        dtype=float,
    )
    conserved_mass = np.asarray(
        evolving["conserved_descriptor_reduced_scaled_matrix"],
        dtype=float,
    )
    vertical_mass = np.asarray(
        evolving["vertical_descriptor_reduced_scaled_matrix"],
        dtype=float,
    )
    total_dm = np.asarray(
        evolving["storage_rate_derivative_scaled_matrix"],
        dtype=float,
    )
    conserved_dm = np.asarray(
        evolving[
            "conserved_storage_rate_derivative_scaled_matrix"
        ],
        dtype=float,
    )
    vertical_dm = np.asarray(
        evolving["vertical_storage_rate_derivative_scaled_matrix"],
        dtype=float,
    )
    vertical_scale = max(
        float(np.max(np.abs(vertical_mass))),
        np.finfo(float).tiny,
    )
    forbidden_vertical_rows = np.concatenate(
        (vertical_mass[0::5].ravel(), vertical_mass[4::5].ravel())
    )
    allowed_vertical_row_maxima = np.asarray(
        [
            np.max(np.abs(vertical_mass[component::5]))
            for component in (1, 2, 3)
        ],
        dtype=float,
    )
    vertical_structure_defect = float(
        np.max(np.abs(forbidden_vertical_rows)) / vertical_scale
    )
    complete_vector = bool(
        vertical.shape[1] == 5
        and zero_component_defect <= 1.0e-12
        and vertical_structure_defect <= 1.0e-12
        and np.all(allowed_vertical_row_maxima > 0.0)
    )
    direct_rank = int(np.linalg.matrix_rank(total_mass))
    direct_condition = float(np.linalg.cond(total_mass))
    return {
        "semantics": (
            "vector_storage_one_form_with_path_ledger_not_state_function"
        ),
        "component_order": (
            "mass",
            "radial_momentum",
            "angular_momentum",
            "killing_energy",
            "stress_storage",
        ),
        "maximum_absolute_components_per_ct": component_maxima,
        "mass_and_stress_zero_component_relative_defect": (
            zero_component_defect
        ),
        "complete_vector_one_form_present": complete_vector,
        "vertical_descriptor_allowed_row_maxima": (
            allowed_vertical_row_maxima
        ),
        "vertical_descriptor_forbidden_row_relative_defect": (
            vertical_structure_defect
        ),
        "maximum_scaled_descriptor_component_reconstruction_defect": (
            np.max(np.abs(total_mass - conserved_mass - vertical_mass))
        ),
        "maximum_scaled_storage_rate_component_reconstruction_defect": (
            np.max(np.abs(total_dm - conserved_dm - vertical_dm))
        ),
        "direct_storage_rank": direct_rank,
        "direct_storage_dimension": total_mass.shape[0],
        "direct_storage_condition_estimate": direct_condition,
        "maximum_relative_storage_action_defect": evolving[
            "maximum_relative_storage_action_defect"
        ],
        "maximum_relative_historical_storage_action_defect": evolving[
            "maximum_relative_frozen_storage_action_defect"
        ],
        "maximum_relative_direct_to_historical_storage_matrix_change": (
            evolving["maximum_relative_storage_matrix_change"]
        ),
        "maximum_absolute_direct_off_cell_storage_entry": evolving[
            "maximum_absolute_direct_off_cell_storage_entry"
        ],
        "maximum_absolute_historical_off_cell_storage_entry": evolving[
            "maximum_absolute_frozen_off_cell_storage_entry"
        ],
        "direct_off_cell_storage_nonzero_count": evolving[
            "direct_off_cell_storage_nonzero_count"
        ],
        "storage_color_count": evolving["storage_component_colors"],
        "storage_rate_derivative_source": evolving[
            "storage_rate_derivative_source"
        ],
        "storage_rate_derivative_step": evolving[
            "storage_rate_derivative_step"
        ],
        "storage_rate_derivative_outer_component_colors": evolving[
            "storage_rate_derivative_component_colors"
        ],
        "storage_rate_derivative_inner_component_colors": evolving[
            "storage_rate_derivative_inner_component_colors"
        ],
        "storage_rate_derivative_outer_component_evaluations": evolving[
            "storage_rate_derivative_outer_component_evaluations"
        ],
        "storage_rate_derivative_nested_component_evaluations": evolving[
            "storage_rate_derivative_nested_component_evaluations"
        ],
        "vertical_storage_rate_derivative_outer_action_evaluations": (
            evolving[
                "vertical_storage_rate_derivative_outer_action_evaluations"
            ]
        ),
        "vertical_storage_rate_derivative_path_evaluations": evolving[
            "vertical_storage_rate_derivative_path_evaluations"
        ],
        "direct_storage_matrix_source": evolving["mass_matrix_source"],
        "maximum_scaled_generator_factorization_defect": evolving[
            "maximum_scaled_generator_factorization_defect"
        ],
        "passed": bool(
            complete_vector
            and evolving["maximum_relative_storage_action_defect"]
            <= MAXIMUM_STORAGE_ACTION_RELATIVE_DEFECT
            and evolving["maximum_relative_storage_matrix_change"]
            <= MAXIMUM_DIRECT_TO_HISTORICAL_STORAGE_CHANGE
            and evolving[
                "maximum_scaled_descriptor_component_reconstruction_defect"
            ]
            <= MAXIMUM_STORAGE_COMPONENT_RECONSTRUCTION_DEFECT
            and evolving[
                "maximum_scaled_storage_rate_component_reconstruction_defect"
            ]
            <= MAXIMUM_STORAGE_COMPONENT_RECONSTRUCTION_DEFECT
            and direct_rank == total_mass.shape[0]
            and direct_condition
            <= MAXIMUM_DIRECT_STORAGE_CONDITION_ESTIMATE
            and evolving["direct_off_cell_storage_nonzero_count"] > 0
            and evolving["maximum_scaled_generator_factorization_defect"]
            <= MAXIMUM_GENERATOR_FACTORIZATION_DEFECT
            and evolving["storage_rate_derivative_source"]
            == (
                "nested_colored_conserved_matrix_plus_"
                "vertical_rate_action"
            )
        ),
    }


def _jvp_defect(
    predicted: np.ndarray,
    observed: np.ndarray,
    *,
    relative_tolerance: float,
) -> dict:
    left = np.asarray(predicted, dtype=float).ravel()
    right = np.asarray(observed, dtype=float).ravel()
    difference = left - right
    infinity_activity = max(
        float(np.max(np.abs(left))),
        float(np.max(np.abs(right))),
    )
    l2_activity = max(
        float(np.linalg.norm(left)),
        float(np.linalg.norm(right)),
    )
    infinity_absolute = float(np.max(np.abs(difference)))
    l2_absolute = float(np.linalg.norm(difference))
    infinity_relative = (
        infinity_absolute / infinity_activity
        if infinity_activity > JVP_ACTIVITY_FLOOR_PER_S
        else 0.0
    )
    l2_relative = (
        l2_absolute / l2_activity
        if l2_activity > JVP_ACTIVITY_FLOOR_PER_S
        else 0.0
    )
    absolute_tolerance = (
        MAXIMUM_INDEPENDENT_VECTOR_FIELD_JVP_ABSOLUTE_DEFECT
    )
    return {
        "maximum_absolute_defect_per_s": infinity_absolute,
        "l2_absolute_defect_per_s": l2_absolute,
        "infinity_activity_per_s": infinity_activity,
        "l2_activity_per_s": l2_activity,
        "relative_infinity_defect": infinity_relative,
        "relative_l2_defect": l2_relative,
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance_per_s": absolute_tolerance,
        "passed": bool(
            infinity_absolute
            <= absolute_tolerance
            + relative_tolerance * infinity_activity
            and l2_absolute
            <= np.sqrt(left.size) * absolute_tolerance
            + relative_tolerance * l2_activity
        ),
    }


def _normalized_jvp_direction(
    values: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    direction = np.asarray(values, dtype=float).ravel()
    norm = float(np.linalg.norm(direction))
    if (
        direction.ndim != 1
        or np.any(~np.isfinite(direction))
        or norm <= np.finfo(float).tiny
    ):
        raise RuntimeError(f"JVP direction {name!r} is invalid")
    return direction / norm


def _smooth_jvp_direction(initial: dict) -> np.ndarray:
    n_cells = initial["state"].n_cells
    log_radius = np.log(
        np.asarray(initial["context"].grid.centers, dtype=float)
    )
    coordinate = (
        (log_radius - log_radius[0])
        / max(log_radius[-1] - log_radius[0], np.finfo(float).tiny)
    )
    direction = np.empty((n_cells, 5), dtype=float)
    direction[:, 0] = 0.35 * np.sin(np.pi * coordinate)
    direction[:, 1] = 0.25 * np.sin(2.0 * np.pi * coordinate)
    direction[:, 2] = 0.15 * np.cos(np.pi * coordinate)
    direction[:, 3] = 0.45 * np.cos(2.0 * np.pi * coordinate)
    direction[:, 4] = 0.20 * np.sin(3.0 * np.pi * coordinate)
    return _normalized_jvp_direction(direction, name="smooth_mixed")


def _vector_field_branch_state(
    initial: dict,
    primitive_vector: np.ndarray,
) -> tuple[dict, np.ndarray, dict]:
    context = initial["context"]
    n_cells = initial["state"].n_cells
    charts = np.asarray(primitive_vector, dtype=float).reshape(n_cells, 5)
    reconstruction = causal_five_field_reconstruct_face_charts(
        context,
        charts,
    )
    evaluation = evaluate_causal_five_field_dae(
        pack_causal_five_field_state(
            causal_five_field_state_from_primitives(context, charts)
        ),
        context,
    )
    active_set = {
        "outer_boundary_choked": bool(
            evaluation.outer_boundary_choked
        ),
        "outer_incoming_characteristics": int(
            evaluation.outer_incoming_characteristics
        ),
    }
    rusanov = causal_five_field_rusanov_control_diagnostics(
        context,
        charts,
    )
    return (
        active_set,
        np.asarray(reconstruction.admissibility_factors, dtype=float),
        rusanov,
    )


def _rusanov_branch_resolution(
    *audits: dict,
) -> dict:
    """Classify max-speed branches without rejecting a zero-jump tie.

    A controller unique above the numerical tie floor is resolved directly.
    A tied or switching controller is accepted only when the component-scaled
    conserved jump multiplying the Rusanov speed is below the declared
    suppression bound.  Near-tie branch sensitivity is then reserved through
    the generator-diameter and frozen finite-time generalized-Jacobian
    contracts; it is not mislabeled as a controller with the larger screening
    margin.
    """

    if not audits:
        raise ValueError("at least one Rusanov audit is required")
    margins = np.stack(
        [
            np.asarray(
                audit["relative_control_margins"],
                dtype=float,
            )
            for audit in audits
        ]
    )
    jumps = np.stack(
        [
            np.asarray(
                audit["relative_scaled_conserved_jump_maximum"],
                dtype=float,
            )
            for audit in audits
        ]
    )
    if margins.shape != jumps.shape:
        raise ValueError("Rusanov margin and jump audits differ in shape")
    unique = margins > RUSANOV_NUMERICAL_TIE_RELATIVE_MARGIN
    declared_unique = (
        margins >= MINIMUM_RUSANOV_CONTROL_RELATIVE_MARGIN
    )
    suppressed = (
        jumps <= MAXIMUM_SUPPRESSED_RUSANOV_SCALED_RELATIVE_JUMP
    )
    resolved = unique | suppressed
    finite_branch_screen_resolved = declared_unique | suppressed
    base_codes = np.asarray(audits[0]["control_codes"], dtype=int)
    changed = np.zeros(base_codes.shape, dtype=bool)
    for audit in audits[1:]:
        changed |= (
            np.asarray(audit["control_codes"], dtype=int) != base_codes
        )
    changed_suppressed = bool(
        not np.any(changed)
        or np.all(suppressed[:, changed])
    )
    exact_zero = np.stack(
        [
            np.asarray(
                audit["exact_zero_conserved_jump"],
                dtype=bool,
            )
            for audit in audits
        ]
    )
    return {
        "unique_face_state_count": int(np.count_nonzero(np.all(unique, axis=0))),
        "suppressed_face_state_count": int(
            np.count_nonzero(~unique & suppressed)
        ),
        "exact_zero_jump_face_state_count": int(
            np.count_nonzero(exact_zero)
        ),
        "control_changed_face_count": int(np.count_nonzero(changed)),
        "changed_faces_jump_suppressed": changed_suppressed,
        "minimum_relative_control_margin": float(np.min(margins)),
        "maximum_relative_jump_on_unresolved_margin": (
            float(np.max(jumps[~unique]))
            if np.any(~unique)
            else 0.0
        ),
        "maximum_allowed_suppressed_relative_jump": (
            MAXIMUM_SUPPRESSED_RUSANOV_SCALED_RELATIVE_JUMP
        ),
        "unresolved_face_state_count": int(np.count_nonzero(~resolved)),
        "declared_finite_branch_screen_unresolved_face_state_count": int(
            np.count_nonzero(~finite_branch_screen_resolved)
        ),
        "declared_finite_branch_screen_passed": bool(
            np.all(finite_branch_screen_resolved)
        ),
        "passed": bool(np.all(resolved) and changed_suppressed),
    }


def _rusanov_switching_normal_directions(
    initial: dict,
    primitives: np.ndarray,
    primitive_scales: np.ndarray,
    base_audit: dict,
    evolving: dict,
) -> tuple[dict[str, np.ndarray], dict, dict[str, np.ndarray]]:
    """Bound and probe the descriptor-propagated Rusanov branch kink."""

    margins = np.asarray(
        base_audit["relative_control_margins"],
        dtype=float,
    )
    scaled_jumps = np.asarray(
        base_audit["relative_scaled_conserved_jump_maximum"],
        dtype=float,
    )
    exact_zero = np.asarray(
        base_audit["exact_zero_conserved_jump"],
        dtype=bool,
    )
    context = initial["context"]
    n_cells = initial["state"].n_cells
    base_codes = np.asarray(base_audit["control_codes"], dtype=int)
    candidate_speeds = np.asarray(
        base_audit["candidate_absolute_speeds_over_c"],
        dtype=float,
    )
    targeted: list[tuple[int, int]] = []
    for output_index in np.flatnonzero(
        (margins < MINIMUM_RUSANOV_CONTROL_RELATIVE_MARGIN)
        & ~exact_zero
    ):
        controlling_code = int(base_codes[output_index])
        maximum = float(candidate_speeds[output_index, controlling_code])
        relative_candidate_gaps = (
            maximum - candidate_speeds[output_index]
        ) / max(maximum, np.finfo(float).tiny)
        for competitor_code in np.flatnonzero(
            relative_candidate_gaps
            < MINIMUM_RUSANOV_CONTROL_RELATIVE_MARGIN
        ):
            if int(competitor_code) != controlling_code:
                targeted.append(
                    (int(output_index), int(competitor_code))
                )
    step = RUSANOV_SWITCHING_NORMAL_DIFFERENCE_STEP
    raw_gradients = []
    generator_left_factors = []
    generator_right_factors = []
    physical_flux_left_factors = []
    kink_face_indices = []
    generator_relative_diameters = []
    rows = {}
    for output_index, competitor_code in targeted:
        face_index = int(base_audit["face_indices"][output_index])
        active_cells = range(
            max(0, face_index - 2),
            min(n_cells, face_index + 2),
        )
        active_columns = [
            5 * cell + component
            for cell in active_cells
            for component in range(5)
        ]
        gradient = np.zeros(5 * n_cells, dtype=float)
        controlling_code = int(base_codes[output_index])
        for column in active_columns:
            increment = np.zeros(5 * n_cells, dtype=float)
            increment[column] = step * primitive_scales[column]
            plus = causal_five_field_rusanov_control_diagnostics(
                context,
                (primitives + increment).reshape(n_cells, 5),
            )
            minus = causal_five_field_rusanov_control_diagnostics(
                context,
                (primitives - increment).reshape(n_cells, 5),
            )
            plus_candidates = np.asarray(
                plus["candidate_absolute_speeds_over_c"],
                dtype=float,
            )
            minus_candidates = np.asarray(
                minus["candidate_absolute_speeds_over_c"],
                dtype=float,
            )
            plus_gap = (
                plus_candidates[output_index, controlling_code]
                - plus_candidates[output_index, competitor_code]
            )
            minus_gap = (
                minus_candidates[output_index, controlling_code]
                - minus_candidates[output_index, competitor_code]
            )
            gradient[column] = (plus_gap - minus_gap) / (2.0 * step)
        norm = float(np.linalg.norm(gradient))
        if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
            raise RuntimeError(
                "a consequential Rusanov near tie has no resolved "
                "switching-normal gradient"
            )
        jump = np.asarray(
            base_audit["conserved_jumps"][output_index],
            dtype=float,
        )
        measure = float(context.grid.face_measures[face_index])
        delta_stationary_left = np.zeros(5 * n_cells, dtype=float)
        left_rows = slice(
            5 * (face_index - 1),
            5 * face_index,
        )
        right_rows = slice(
            5 * face_index,
            5 * (face_index + 1),
        )
        row_scales = np.asarray(
            evolving["conservation_row_scales"],
            dtype=float,
        )
        delta_stationary_left[left_rows] += (
            -0.5 * measure * jump / row_scales[left_rows]
        )
        delta_stationary_left[right_rows] -= (
            -0.5 * measure * jump / row_scales[right_rows]
        )
        mass = np.asarray(
            evolving["descriptor_reduced_scaled_matrix"],
            dtype=float,
        )
        generator_left = -np.linalg.solve(
            mass,
            delta_stationary_left,
        )
        generator_right = gradient
        generator = np.asarray(
            evolving["evolving_scaled_generator_per_s"],
            dtype=float,
        )
        relative_diameter = float(
            np.linalg.norm(generator_left)
            * np.linalg.norm(generator_right)
            / max(
                float(np.linalg.norm(generator)),
                np.finfo(float).tiny,
            )
        )
        raw_gradients.append(gradient)
        generator_left_factors.append(generator_left)
        generator_right_factors.append(generator_right)
        physical_flux_left_factors.append(
            C * (-0.5 * measure * jump)
        )
        kink_face_indices.append(face_index)
        generator_relative_diameters.append(relative_diameter)
        row_name = f"face_{face_index}_competitor_{competitor_code}"
        rows[row_name] = {
            "face_index": face_index,
            "base_relative_control_margin": float(margins[output_index]),
            "base_relative_scaled_conserved_jump_maximum": float(
                scaled_jumps[output_index]
            ),
            "controlling_code": controlling_code,
            "competitor_code": competitor_code,
            "active_scaled_column_count": len(active_columns),
            "scaled_gap_gradient_norm": norm,
            "scaled_generator_relative_diameter": relative_diameter,
            "difference_step": step,
        }
    directions = {}
    if raw_gradients:
        controlling = int(np.argmax(generator_relative_diameters))
        output_index, competitor_code = targeted[controlling]
        physical_face = int(
            base_audit["face_indices"][output_index]
        )
        directions[
            "rusanov_switching_normal_"
            f"face_{physical_face}_competitor_{competitor_code}"
        ] = raw_gradients[controlling] / np.linalg.norm(
            raw_gradients[controlling]
        )
    aggregate_relative_diameter_bound = float(
        np.sum(generator_relative_diameters)
    )
    kink_arrays = {
        "rusanov_kink_generator_left_factors": (
            np.column_stack(generator_left_factors)
            if generator_left_factors
            else np.empty((5 * n_cells, 0), dtype=float)
        ),
        "rusanov_kink_generator_right_factors": (
            np.column_stack(generator_right_factors)
            if generator_right_factors
            else np.empty((5 * n_cells, 0), dtype=float)
        ),
        "rusanov_kink_generator_relative_diameters": np.asarray(
            generator_relative_diameters,
            dtype=float,
        ),
        "rusanov_kink_face_indices": np.asarray(
            kink_face_indices,
            dtype=float,
        ),
        "rusanov_kink_physical_flux_left_factors": (
            np.column_stack(physical_flux_left_factors)
            if physical_flux_left_factors
            else np.empty((5, 0), dtype=float)
        ),
    }
    return directions, {
        "targeted_branch_count": len(targeted),
        "targeted_face_count": len(
            {output_index for output_index, _code in targeted}
        ),
        "targeted_branches": rows,
        "candidate_policy": (
            "every noncontrolling one of the ten side/family speed "
            "candidates within the declared near-tie margin"
        ),
        "generator_kink_relative_diameter_bound": (
            aggregate_relative_diameter_bound
        ),
        "maximum_generator_kink_relative_diameter": (
            MAXIMUM_RUSANOV_GENERATOR_KINK_RELATIVE_DIAMETER
        ),
        "probe_direction_count": len(directions),
        "bound_semantics": (
            "triangle bound over descriptor-propagated fixed-candidate "
            "generalized-Jacobian differences"
        ),
        "passed": bool(
            len(rows) == len(targeted)
            and aggregate_relative_diameter_bound
            <= MAXIMUM_RUSANOV_GENERATOR_KINK_RELATIVE_DIAMETER
        ),
    }, kink_arrays


def _independent_vector_field_jvp_audit(
    initial: dict,
    vector: np.ndarray,
    evolving: dict,
    direction: np.ndarray,
    *,
    direction_name: str,
    centered_difference_step: float,
    inner_storage_matrix_difference_step: float,
    base_vector_field: dict | None = None,
    base_branch_state: tuple[dict, np.ndarray, dict] | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    context = initial["context"]
    n_cells = initial["state"].n_cells
    state = unpack_causal_five_field_state(vector, n_cells)
    primitives = np.asarray(state.primitives, dtype=float).ravel()
    primitive_scales = np.asarray(
        evolving["primitive_column_scales"],
        dtype=float,
    )
    conservation_scales = np.asarray(
        evolving["conservation_row_scales"],
        dtype=float,
    )
    direction = _normalized_jvp_direction(
        direction,
        name=direction_name,
    )
    step = float(centered_difference_step)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("JVP centered-difference step is invalid")
    if base_vector_field is None:
        base_vector_field = causal_five_field_scaled_primitive_vector_field(
            context,
            primitives,
            primitive_column_scales=primitive_scales,
            conservation_row_scales=conservation_scales,
            finite_difference_step=inner_storage_matrix_difference_step,
            storage_quadrature_order=STORAGE_QUADRATURE_ORDER,
            storage_directional_step=STORAGE_DIRECTIONAL_STEP,
        )
    if base_branch_state is None:
        base_branch_state = _vector_field_branch_state(
            initial,
            primitives,
        )
    plus_primitives = primitives + step * primitive_scales * direction
    minus_primitives = primitives - step * primitive_scales * direction
    plus = causal_five_field_scaled_primitive_vector_field(
        context,
        plus_primitives,
        primitive_column_scales=primitive_scales,
        conservation_row_scales=conservation_scales,
        finite_difference_step=inner_storage_matrix_difference_step,
        storage_quadrature_order=STORAGE_QUADRATURE_ORDER,
        storage_directional_step=STORAGE_DIRECTIONAL_STEP,
    )
    minus = causal_five_field_scaled_primitive_vector_field(
        context,
        minus_primitives,
        primitive_column_scales=primitive_scales,
        conservation_row_scales=conservation_scales,
        finite_difference_step=inner_storage_matrix_difference_step,
        storage_quadrature_order=STORAGE_QUADRATURE_ORDER,
        storage_directional_step=STORAGE_DIRECTIONAL_STEP,
    )
    base_rate = np.asarray(
        base_vector_field["scaled_primitive_rate_per_s"],
        dtype=float,
    ).ravel()
    plus_rate = np.asarray(
        plus["scaled_primitive_rate_per_s"],
        dtype=float,
    ).ravel()
    minus_rate = np.asarray(
        minus["scaled_primitive_rate_per_s"],
        dtype=float,
    ).ravel()
    direct = (
        plus_rate - minus_rate
    ) / (2.0 * step)
    forward = (plus_rate - base_rate) / step
    backward = (base_rate - minus_rate) / step
    predicted = (
        np.asarray(
            evolving["evolving_scaled_generator_per_s"],
            dtype=float,
        )
        @ direction
    )
    central_defect = _jvp_defect(
        predicted,
        direct,
        relative_tolerance=(
            MAXIMUM_INDEPENDENT_VECTOR_FIELD_JVP_RELATIVE_DEFECT
        ),
    )
    forward_defect = _jvp_defect(
        predicted,
        forward,
        relative_tolerance=(
            MAXIMUM_FORWARD_BACKWARD_JVP_RELATIVE_DEFECT
        ),
    )
    backward_defect = _jvp_defect(
        predicted,
        backward,
        relative_tolerance=(
            MAXIMUM_FORWARD_BACKWARD_JVP_RELATIVE_DEFECT
        ),
    )
    base_active_set, base_factors, base_rusanov = base_branch_state
    plus_active_set, plus_factors, plus_rusanov = (
        _vector_field_branch_state(initial, plus_primitives)
    )
    minus_active_set, minus_factors, minus_rusanov = (
        _vector_field_branch_state(initial, minus_primitives)
    )
    active_set_unchanged = bool(
        plus_active_set == base_active_set
        and minus_active_set == base_active_set
    )
    perturbations_differentiable = bool(
        np.all(
            np.abs(base_factors - 1.0)
            <= ADMISSIBILITY_FACTOR_INACTIVE_ABSOLUTE_TOLERANCE
        )
        and np.all(
            np.abs(plus_factors - 1.0)
            <= ADMISSIBILITY_FACTOR_INACTIVE_ABSOLUTE_TOLERANCE
        )
        and np.all(
            np.abs(minus_factors - 1.0)
            <= ADMISSIBILITY_FACTOR_INACTIVE_ABSOLUTE_TOLERANCE
        )
    )
    rusanov_controls_unchanged = bool(
        np.array_equal(
            plus_rusanov["control_codes"],
            base_rusanov["control_codes"],
        )
        and np.array_equal(
            minus_rusanov["control_codes"],
            base_rusanov["control_codes"],
        )
    )
    minimum_rusanov_margin = min(
        float(base_rusanov["minimum_relative_control_margin"]),
        float(plus_rusanov["minimum_relative_control_margin"]),
        float(minus_rusanov["minimum_relative_control_margin"]),
    )
    rusanov_resolution = _rusanov_branch_resolution(
        base_rusanov,
        plus_rusanov,
        minus_rusanov,
    )
    rusanov_branch_differentiable = bool(rusanov_resolution["passed"])
    return {
        "independent_vector_field_jvp_direction": direction,
        "independent_vector_field_jvp_direct": direct,
        "independent_vector_field_jvp_forward": forward,
        "independent_vector_field_jvp_backward": backward,
        "independent_vector_field_jvp_predicted": predicted,
        "independent_vector_field_jvp_base_rate": base_rate,
        "independent_vector_field_jvp_base_admissibility_factors": (
            base_factors
        ),
        "independent_vector_field_jvp_plus_admissibility_factors": (
            plus_factors
        ),
        "independent_vector_field_jvp_minus_admissibility_factors": (
            minus_factors
        ),
        "independent_vector_field_jvp_base_rusanov_control_codes": (
            base_rusanov["control_codes"]
        ),
        "independent_vector_field_jvp_plus_rusanov_control_codes": (
            plus_rusanov["control_codes"]
        ),
        "independent_vector_field_jvp_minus_rusanov_control_codes": (
            minus_rusanov["control_codes"]
        ),
        "independent_vector_field_jvp_base_rusanov_relative_margins": (
            base_rusanov["relative_control_margins"]
        ),
        "independent_vector_field_jvp_plus_rusanov_relative_margins": (
            plus_rusanov["relative_control_margins"]
        ),
        "independent_vector_field_jvp_minus_rusanov_relative_margins": (
            minus_rusanov["relative_control_margins"]
        ),
        "independent_vector_field_jvp_base_rusanov_relative_jumps": (
            base_rusanov["relative_conserved_jump_l2"]
        ),
        "independent_vector_field_jvp_plus_rusanov_relative_jumps": (
            plus_rusanov["relative_conserved_jump_l2"]
        ),
        "independent_vector_field_jvp_minus_rusanov_relative_jumps": (
            minus_rusanov["relative_conserved_jump_l2"]
        ),
        "independent_vector_field_jvp_base_rusanov_scaled_relative_jumps": (
            base_rusanov["relative_scaled_conserved_jump_maximum"]
        ),
        "independent_vector_field_jvp_plus_rusanov_scaled_relative_jumps": (
            plus_rusanov["relative_scaled_conserved_jump_maximum"]
        ),
        "independent_vector_field_jvp_minus_rusanov_scaled_relative_jumps": (
            minus_rusanov["relative_scaled_conserved_jump_maximum"]
        ),
    }, {
        "scope": "production_gauss_plm_directional_frechet_screen",
        "direction_name": direction_name,
        "direction_component_labels": (
            "log_surface_density",
            "radial_three_velocity_over_c",
            "azimuthal_three_velocity_over_c",
            "log_temperature",
            "specific_causal_stress",
        ),
        "direction_scaled_euclidean_norm": float(
            np.linalg.norm(direction)
        ),
        "centered_difference_step": step,
        "storage_matrix_inner_difference_step": (
            inner_storage_matrix_difference_step
        ),
        "central_jvp_defect": central_defect,
        "forward_jvp_defect": forward_defect,
        "backward_jvp_defect": backward_defect,
        "maximum_relative_defect": (
            MAXIMUM_INDEPENDENT_VECTOR_FIELD_JVP_RELATIVE_DEFECT
        ),
        "plus_minus_reconstruction_differentiable": (
            perturbations_differentiable
        ),
        "base_outer_active_set": base_active_set,
        "plus_outer_active_set": plus_active_set,
        "minus_outer_active_set": minus_active_set,
        "plus_minus_outer_active_set_unchanged": active_set_unchanged,
        "base_rusanov_control_labels": base_rusanov["control_labels"],
        "plus_rusanov_control_labels": plus_rusanov["control_labels"],
        "minus_rusanov_control_labels": minus_rusanov["control_labels"],
        "rusanov_controls_unchanged": rusanov_controls_unchanged,
        "minimum_rusanov_control_relative_margin": (
            minimum_rusanov_margin
        ),
        "minimum_required_rusanov_control_relative_margin": (
            MINIMUM_RUSANOV_CONTROL_RELATIVE_MARGIN
        ),
        "rusanov_branch_resolution": rusanov_resolution,
        "rusanov_branch_differentiable": (
            rusanov_branch_differentiable
        ),
        "plus_storage_component_colors": plus[
            "storage_component_colors"
        ],
        "minus_storage_component_colors": minus[
            "storage_component_colors"
        ],
        "passed": bool(
            perturbations_differentiable
            and active_set_unchanged
            and rusanov_branch_differentiable
            and central_defect["passed"]
            and forward_defect["passed"]
            and backward_defect["passed"]
        ),
    }


def _production_vector_field_jvp_suite(
    initial: dict,
    vector: np.ndarray,
    evolving: dict,
    named_directions: dict[str, np.ndarray],
    *,
    include_extended_directions: bool,
) -> tuple[dict[str, np.ndarray], dict]:
    required_names = (
        "thermal_redistribution_6_to_60rg",
        "radial_redistribution_6_to_60rg",
    )
    if any(name not in named_directions for name in required_names):
        raise RuntimeError("required localized JVP directions are absent")
    directions = {
        name: _normalized_jvp_direction(
            named_directions[name],
            name=name,
        )
        for name in required_names
    }
    if include_extended_directions:
        directions["smooth_mixed"] = _smooth_jvp_direction(initial)

    context = initial["context"]
    n_cells = initial["state"].n_cells
    state = unpack_causal_five_field_state(vector, n_cells)
    primitives = np.asarray(state.primitives, dtype=float).ravel()
    primitive_scales = np.asarray(
        evolving["primitive_column_scales"],
        dtype=float,
    )
    conservation_scales = np.asarray(
        evolving["conservation_row_scales"],
        dtype=float,
    )
    base_vector_field = causal_five_field_scaled_primitive_vector_field(
        context,
        primitives,
        primitive_column_scales=primitive_scales,
        conservation_row_scales=conservation_scales,
        finite_difference_step=FINITE_DIFFERENCE_STEP,
        storage_quadrature_order=STORAGE_QUADRATURE_ORDER,
        storage_directional_step=STORAGE_DIRECTIONAL_STEP,
    )
    base_branch_state = _vector_field_branch_state(initial, primitives)
    (
        switching_directions,
        switching_metadata,
        switching_arrays,
    ) = (
        _rusanov_switching_normal_directions(
            initial,
            primitives,
            primitive_scales,
            base_branch_state[2],
            evolving,
        )
    )
    directions.update(switching_directions)
    arrays: dict[str, np.ndarray] = {}
    arrays.update(switching_arrays)
    direction_metadata = {}
    direction_arrays = {}
    for name, direction in directions.items():
        jvp_arrays, jvp_metadata = _independent_vector_field_jvp_audit(
            initial,
            vector,
            evolving,
            direction,
            direction_name=name,
            centered_difference_step=INDEPENDENT_VECTOR_FIELD_JVP_STEP,
            inner_storage_matrix_difference_step=FINITE_DIFFERENCE_STEP,
            base_vector_field=base_vector_field,
            base_branch_state=base_branch_state,
        )
        direction_arrays[name] = jvp_arrays
        direction_metadata[name] = jvp_metadata
        arrays.update(
            {
                f"{name}_{array_name}": values
                for array_name, values in jvp_arrays.items()
            }
        )

    independent_step_stability = {
        "evaluated": False,
        "steps": INDEPENDENT_VECTOR_FIELD_JVP_STABILITY_STEPS,
        "directions": {},
        "passed": True,
    }
    if include_extended_directions:
        smooth = directions["smooth_mixed"]
        stability_rows = {}
        for step in INDEPENDENT_VECTOR_FIELD_JVP_STABILITY_STEPS:
            if step == INDEPENDENT_VECTOR_FIELD_JVP_STEP:
                row_arrays = direction_arrays["smooth_mixed"]
                row_metadata = direction_metadata["smooth_mixed"]
            else:
                row_arrays, row_metadata = (
                    _independent_vector_field_jvp_audit(
                        initial,
                        vector,
                        evolving,
                        smooth,
                        direction_name=(
                            f"smooth_mixed_independent_step_{step:.0e}"
                        ),
                        centered_difference_step=step,
                        inner_storage_matrix_difference_step=(
                            FINITE_DIFFERENCE_STEP
                        ),
                        base_vector_field=base_vector_field,
                        base_branch_state=base_branch_state,
                    )
                )
                arrays.update(
                    {
                        (
                            f"smooth_mixed_step_{step:.0e}_"
                            f"{array_name}"
                        ): values
                        for array_name, values in row_arrays.items()
                    }
                )
            stability_rows[f"{step:.0e}"] = row_metadata
        independent_step_stability = {
            "evaluated": True,
            "steps": INDEPENDENT_VECTOR_FIELD_JVP_STABILITY_STEPS,
            "directions": stability_rows,
            "passed": bool(
                all(row["passed"] for row in stability_rows.values())
            ),
        }

    additivity = {
        "evaluated": False,
        "passed": True,
    }
    if include_extended_directions:
        left_name, right_name = required_names
        unnormalized = directions[left_name] + directions[right_name]
        normalization = float(np.linalg.norm(unnormalized))
        combined = unnormalized / normalization
        combined_arrays, combined_metadata = (
            _independent_vector_field_jvp_audit(
                initial,
                vector,
                evolving,
                combined,
                direction_name=(
                    f"normalized_{left_name}_plus_{right_name}"
                ),
                centered_difference_step=INDEPENDENT_VECTOR_FIELD_JVP_STEP,
                inner_storage_matrix_difference_step=(
                    FINITE_DIFFERENCE_STEP
                ),
                base_vector_field=base_vector_field,
                base_branch_state=base_branch_state,
            )
        )
        expected = (
            direction_arrays[left_name][
                "independent_vector_field_jvp_direct"
            ]
            + direction_arrays[right_name][
                "independent_vector_field_jvp_direct"
            ]
        ) / normalization
        observed = combined_arrays[
            "independent_vector_field_jvp_direct"
        ]
        additivity_defect = _jvp_defect(
            expected,
            observed,
            relative_tolerance=MAXIMUM_JVP_ADDITIVITY_RELATIVE_DEFECT,
        )
        additivity = {
            "evaluated": True,
            "left_direction": left_name,
            "right_direction": right_name,
            "combined_direction_normalization": normalization,
            "defect": additivity_defect,
            "combined_direction_jvp": combined_metadata,
            "passed": bool(
                combined_metadata["passed"]
                and additivity_defect["passed"]
            ),
        }
        arrays.update(
            {
                f"additivity_combined_{array_name}": values
                for array_name, values in combined_arrays.items()
            }
        )
        arrays["additivity_expected_direct_jvp"] = expected
        arrays["additivity_observed_direct_jvp"] = observed

    passed = bool(
        all(row["passed"] for row in direction_metadata.values())
        and additivity["passed"]
        and independent_step_stability["passed"]
        and switching_metadata["passed"]
    )
    return arrays, {
        "scope": (
            "all_anchor_localized_thermal_radial_jvps"
            + (
                "_plus_smooth_and_additivity"
                if include_extended_directions
                else ""
            )
        ),
        "direction_names": tuple(directions),
        "directions": direction_metadata,
        "rusanov_switching_normal_directions": switching_metadata,
        "independent_secant_step_stability": (
            independent_step_stability
        ),
        "additivity": additivity,
        "nonlinear_vector_field_build_count": (
            1 + 2 * len(directions)
            + (2 if include_extended_directions else 0)
            + (
                2
                * (
                    len(INDEPENDENT_VECTOR_FIELD_JVP_STABILITY_STEPS)
                    - 1
                )
                if include_extended_directions
                else 0
            )
        ),
        "passed": passed,
    }


def _generator_stability_audit(
    initial: dict,
    vector: np.ndarray,
    named_directions: dict[str, np.ndarray],
    *,
    base_reduced: dict,
    base_evolving: dict,
    full_scan: bool,
) -> tuple[dict[str, np.ndarray], dict]:
    def compare_variants(
        variants: dict[str, np.ndarray],
        base_key: str,
    ) -> tuple[dict, bool]:
        base = variants[base_key]
        comparisons = {}
        passed = True
        for key, dynamic in variants.items():
            if key == base_key:
                continue
            frobenius = float(
                np.linalg.norm(dynamic - base)
                / max(
                    float(np.linalg.norm(dynamic)),
                    float(np.linalg.norm(base)),
                    np.finfo(float).tiny,
                )
            )
            jvp_rows = {}
            maximum_jvp = 0.0
            for name, direction in named_directions.items():
                left = dynamic @ direction
                right = base @ direction
                relative = float(
                    np.linalg.norm(left - right)
                    / max(
                        float(np.linalg.norm(left)),
                        float(np.linalg.norm(right)),
                        np.finfo(float).tiny,
                    )
                )
                jvp_rows[name] = relative
                maximum_jvp = max(maximum_jvp, relative)
            comparison_passed = bool(
                frobenius <= MAXIMUM_GENERATOR_STABILITY_RELATIVE_DEFECT
                and maximum_jvp
                <= MAXIMUM_GENERATOR_STABILITY_RELATIVE_DEFECT
            )
            comparisons[f"{key}_versus_{base_key}"] = {
                "relative_frobenius_defect": frobenius,
                "deterministic_physical_jvp_relative_defects": jvp_rows,
                "maximum_deterministic_physical_jvp_relative_defect": (
                    maximum_jvp
                ),
                "passed": comparison_passed,
            }
            passed = passed and comparison_passed
        return comparisons, bool(passed)

    production_arrays, production_metadata = (
        _production_vector_field_jvp_suite(
            initial,
            vector,
            base_evolving,
            named_directions,
            include_extended_directions=full_scan,
        )
    )
    if not full_scan:
        return {
            f"production_{name}": values
            for name, values in production_arrays.items()
        }, {
            "scope": "all_anchor_production_jvp_without_full_fd_scan",
            "full_finite_difference_scan_evaluated": False,
            "production_vector_field_jvp": production_metadata,
            "passed": bool(production_metadata["passed"]),
        }

    context = initial["context"]
    inner_variants = {}
    inner_metadata = {}
    for step in INNER_GENERATOR_STABILITY_STEPS:
        if step == FINITE_DIFFERENCE_STEP:
            reduced = base_reduced
            evolving = base_evolving
        else:
            reduced = causal_five_field_reduced_descriptor_matrices(
                context,
                vector,
                finite_difference_step=step,
                descriptor_timestep_seconds=DESCRIPTOR_TIMESTEP_SECONDS,
            )
            evolving = causal_five_field_evolving_tangent_matrices(
                context,
                vector,
                reduced_descriptor=reduced,
                finite_difference_step=step,
                descriptor_timestep_seconds=DESCRIPTOR_TIMESTEP_SECONDS,
                storage_rate_derivative_step=(
                    STORAGE_RATE_DERIVATIVE_STEP
                ),
                storage_difference_step=STORAGE_DIFFERENCE_STEP,
                storage_quadrature_order=STORAGE_QUADRATURE_ORDER,
                storage_directional_step=STORAGE_DIRECTIONAL_STEP,
            )
        key = f"{step:.0e}"
        inner_variants[key] = np.asarray(
            evolving["evolving_scaled_generator_per_s"],
            dtype=float,
        )
        inner_metadata[key] = {
            "finite_difference_step": step,
            "storage_rate_derivative_step": (
                STORAGE_RATE_DERIVATIVE_STEP
            ),
            "storage_difference_step": STORAGE_DIFFERENCE_STEP,
            "maximum_relative_storage_action_defect": evolving[
                "maximum_relative_storage_action_defect"
            ],
            "maximum_scaled_generator_factorization_defect": evolving[
                "maximum_scaled_generator_factorization_defect"
            ],
        }
    base_key = f"{FINITE_DIFFERENCE_STEP:.0e}"
    inner_comparisons, inner_passed = compare_variants(
        inner_variants,
        base_key,
    )

    outer_variants = {}
    outer_evolving = {}
    outer_metadata = {}
    for step in OUTER_GENERATOR_STABILITY_STEPS:
        if step == STORAGE_RATE_DERIVATIVE_STEP:
            evolving = base_evolving
        else:
            evolving = causal_five_field_evolving_tangent_matrices(
                context,
                vector,
                reduced_descriptor=base_reduced,
                finite_difference_step=FINITE_DIFFERENCE_STEP,
                descriptor_timestep_seconds=DESCRIPTOR_TIMESTEP_SECONDS,
                storage_rate_derivative_step=step,
                storage_difference_step=STORAGE_DIFFERENCE_STEP,
                storage_quadrature_order=STORAGE_QUADRATURE_ORDER,
                storage_directional_step=STORAGE_DIRECTIONAL_STEP,
            )
        key = f"{step:.0e}"
        outer_evolving[key] = evolving
        outer_variants[key] = np.asarray(
            evolving["evolving_scaled_generator_per_s"],
            dtype=float,
        )
        outer_metadata[key] = {
            "finite_difference_step": FINITE_DIFFERENCE_STEP,
            "storage_rate_derivative_step": step,
            "storage_difference_step": STORAGE_DIFFERENCE_STEP,
            "maximum_relative_storage_action_defect": evolving[
                "maximum_relative_storage_action_defect"
            ],
            "maximum_scaled_generator_factorization_defect": evolving[
                "maximum_scaled_generator_factorization_defect"
            ],
        }
    outer_base_key = f"{STORAGE_RATE_DERIVATIVE_STEP:.0e}"
    outer_comparisons, outer_passed = compare_variants(
        outer_variants,
        outer_base_key,
    )

    action_variants = {}
    action_metadata = {}
    for step in STORAGE_ACTION_STABILITY_STEPS:
        if step == STORAGE_DIFFERENCE_STEP:
            evolving = base_evolving
        else:
            evolving = causal_five_field_evolving_tangent_matrices(
                context,
                vector,
                reduced_descriptor=base_reduced,
                finite_difference_step=FINITE_DIFFERENCE_STEP,
                descriptor_timestep_seconds=DESCRIPTOR_TIMESTEP_SECONDS,
                storage_rate_derivative_step=(
                    STORAGE_RATE_DERIVATIVE_STEP
                ),
                storage_difference_step=step,
                storage_quadrature_order=STORAGE_QUADRATURE_ORDER,
                storage_directional_step=STORAGE_DIRECTIONAL_STEP,
            )
        key = f"{step:.0e}"
        action_variants[key] = np.asarray(
            evolving["evolving_scaled_generator_per_s"],
            dtype=float,
        )
        action_metadata[key] = {
            "finite_difference_step": FINITE_DIFFERENCE_STEP,
            "storage_rate_derivative_step": (
                STORAGE_RATE_DERIVATIVE_STEP
            ),
            "storage_difference_step": step,
            "maximum_relative_storage_action_defect": evolving[
                "maximum_relative_storage_action_defect"
            ],
            "maximum_scaled_generator_factorization_defect": evolving[
                "maximum_scaled_generator_factorization_defect"
            ],
        }
    action_base_key = f"{STORAGE_DIFFERENCE_STEP:.0e}"
    action_comparisons, action_passed = compare_variants(
        action_variants,
        action_base_key,
    )

    independent_arrays = {}
    independent_metadata = {}
    for key, evolving in outer_evolving.items():
        if key == outer_base_key:
            continue
        jvp_arrays, jvp_metadata = _independent_vector_field_jvp_audit(
            initial,
            vector,
            evolving,
            _smooth_jvp_direction(initial),
            direction_name=f"smooth_mixed_outer_step_{key}",
            centered_difference_step=INDEPENDENT_VECTOR_FIELD_JVP_STEP,
            inner_storage_matrix_difference_step=(
                FINITE_DIFFERENCE_STEP
            ),
        )
        independent_metadata[key] = jvp_metadata
        independent_arrays.update(
            {
                f"outer_{key}_{name}": values
                for name, values in jvp_arrays.items()
            }
        )
    nonproduction_independent_passed = bool(
        all(row["passed"] for row in independent_metadata.values())
    )
    selected_outer_step_independently_certified = bool(
        production_metadata["passed"]
        and (
            not full_scan
            or production_metadata[
                "independent_secant_step_stability"
            ]["passed"]
        )
    )

    arrays = {
        **{
            f"production_{name}": values
            for name, values in production_arrays.items()
        },
        **{
            f"generator_inner_storage_fd_dynamic_{key}": value
            for key, value in inner_variants.items()
        },
        **{
            f"generator_outer_storage_rate_fd_dynamic_{key}": value
            for key, value in outer_variants.items()
        },
        **{
            f"generator_vertical_action_fd_dynamic_{key}": value
            for key, value in action_variants.items()
        },
        **independent_arrays,
    }
    return arrays, {
        "scope": "declared_full_scan_anchor",
        "full_finite_difference_scan_evaluated": True,
        "production_vector_field_jvp": production_metadata,
        "inner_storage_matrix_and_stationary_difference_scan": {
            "steps": INNER_GENERATOR_STABILITY_STEPS,
            "outer_storage_rate_derivative_step_fixed": (
                STORAGE_RATE_DERIVATIVE_STEP
            ),
            "vertical_action_difference_step_fixed": (
                STORAGE_DIFFERENCE_STEP
            ),
            "variants": inner_metadata,
            "comparisons": inner_comparisons,
            "passed": inner_passed,
        },
        "outer_conserved_dm_difference_scan": {
            "steps": OUTER_GENERATOR_STABILITY_STEPS,
            "inner_storage_matrix_difference_step_fixed": (
                FINITE_DIFFERENCE_STEP
            ),
            "vertical_action_difference_step_fixed": (
                STORAGE_DIFFERENCE_STEP
            ),
            "variants": outer_metadata,
            "comparisons": outer_comparisons,
            "neighboring_step_plateau_passed": outer_passed,
            "selected_step_independent_nonlinear_jvp_passed": (
                selected_outer_step_independently_certified
            ),
            "contract": (
                "Neighboring nested second-difference steps are retained as "
                "a cancellation/truncation diagnostic.  Selection is binding "
                "only through direct nonlinear vector-field JVP agreement "
                "over the independent secant-step ladder."
            ),
            "passed": selected_outer_step_independently_certified,
        },
        "vertical_action_difference_scan": {
            "steps": STORAGE_ACTION_STABILITY_STEPS,
            "inner_storage_matrix_difference_step_fixed": (
                FINITE_DIFFERENCE_STEP
            ),
            "outer_storage_rate_derivative_step_fixed": (
                STORAGE_RATE_DERIVATIVE_STEP
            ),
            "variants": action_metadata,
            "comparisons": action_comparisons,
            "passed": action_passed,
        },
        "independent_nonlinear_vector_field_jvp": {
            "production_outer_step_key": outer_base_key,
            "production_suite": production_metadata,
            "nonproduction_outer_step_smooth_variants": (
                independent_metadata
            ),
            "nonproduction_outer_step_variants_passed": (
                nonproduction_independent_passed
            ),
            "passed": bool(
                selected_outer_step_independently_certified
            ),
        },
        "maximum_relative_defect": (
            MAXIMUM_GENERATOR_STABILITY_RELATIVE_DEFECT
        ),
        "passed": bool(
            inner_passed
            and action_passed
            and selected_outer_step_independently_certified
        ),
    }


def _tangent_differentiability_audit(
    initial: dict,
    vector: np.ndarray,
) -> tuple[np.ndarray, dict]:
    state = unpack_causal_five_field_state(
        vector,
        initial["state"].n_cells,
    )
    reconstruction = causal_five_field_reconstruct_face_charts(
        initial["context"],
        state.primitives,
    )
    rusanov = causal_five_field_rusanov_control_diagnostics(
        initial["context"],
        state.primitives,
    )
    evaluation = evaluate_causal_five_field_dae(vector, initial["context"])
    factors = np.asarray(
        reconstruction.admissibility_factors,
        dtype=float,
    )
    finite = bool(np.all(np.isfinite(factors)))
    inactive = np.abs(factors - 1.0) <= (
        ADMISSIBILITY_FACTOR_INACTIVE_ABSOLUTE_TOLERANCE
    )
    active = factors < (
        1.0 - ADMISSIBILITY_FACTOR_INACTIVE_ABSOLUTE_TOLERANCE
    )
    above_one = factors > (
        1.0 + ADMISSIBILITY_FACTOR_INACTIVE_ABSOLUTE_TOLERANCE
    )
    active_margins = 1.0 - factors[active]
    rusanov_resolution = _rusanov_branch_resolution(rusanov)
    passed = bool(
        finite
        and not np.any(above_one)
        and np.all(inactive)
        and rusanov_resolution["passed"]
    )
    return factors, {
        "reconstruction_mode": reconstruction.mode,
        "factor_count": factors.size,
        "minimum_admissibility_factor": float(np.min(factors)),
        "maximum_admissibility_factor": float(np.max(factors)),
        "maximum_absolute_departure_from_unity": float(
            np.max(np.abs(factors - 1.0))
        ),
        "inactive_unity_factor_count": int(np.count_nonzero(inactive)),
        "active_admissibility_factor_count": int(
            np.count_nonzero(active)
        ),
        "above_unity_factor_count": int(np.count_nonzero(above_one)),
        "minimum_active_branch_margin_from_unity": (
            float(np.min(active_margins))
            if active_margins.size
            else None
        ),
        "outer_boundary_choked": bool(
            evaluation.outer_boundary_choked
        ),
        "outer_incoming_characteristics": int(
            evaluation.outer_incoming_characteristics
        ),
        "rusanov_control_labels": rusanov["control_labels"],
        "minimum_rusanov_control_relative_margin": (
            rusanov["minimum_relative_control_margin"]
        ),
        "minimum_required_rusanov_control_relative_margin": (
            MINIMUM_RUSANOV_CONTROL_RELATIVE_MARGIN
        ),
        "rusanov_branch_resolution": rusanov_resolution,
        "contract": (
            "all production admissibility factors must be inactive unity "
            "factors; local tangent construction accepts a numerically unique "
            "Rusanov controller or a component-scaled jump-suppressed tie; "
            "finite-time binding separately requires the declared margin, "
            "suppression, or an exact finite-branch contract"
        ),
        "finite_time_preflight_screen_passed": bool(
            passed
            and rusanov_resolution[
                "declared_finite_branch_screen_passed"
            ]
        ),
        "passed": passed,
    }


def _build_operator_cache(
    initial: dict,
    vector: np.ndarray,
    label: str,
    anchor_role: str,
    shell_edges_rg: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict]:
    n_cells = initial["state"].n_cells
    context = initial["context"]
    operator_contract = _operator_contract(context, shell_edges_rg)
    started = time.perf_counter()
    admissibility_factors, differentiability_audit = (
        _tangent_differentiability_audit(initial, vector)
    )
    if not differentiability_audit["passed"]:
        raise RuntimeError(
            f"N{n_cells} {label} tangent branch is not differentiable"
        )
    reduced = causal_five_field_reduced_descriptor_matrices(
        context,
        vector,
        finite_difference_step=FINITE_DIFFERENCE_STEP,
        descriptor_timestep_seconds=DESCRIPTOR_TIMESTEP_SECONDS,
    )
    evolving = causal_five_field_evolving_tangent_matrices(
        context,
        vector,
        reduced_descriptor=reduced,
        finite_difference_step=FINITE_DIFFERENCE_STEP,
        descriptor_timestep_seconds=DESCRIPTOR_TIMESTEP_SECONDS,
        storage_rate_derivative_step=STORAGE_RATE_DERIVATIVE_STEP,
        storage_difference_step=STORAGE_DIFFERENCE_STEP,
        storage_quadrature_order=STORAGE_QUADRATURE_ORDER,
        storage_directional_step=STORAGE_DIRECTIONAL_STEP,
    )
    outputs = wp10c8d._output_operators(initial, vector, reduced)
    ladder = causal_five_field_moment_coordinate_ladder(
        context,
        vector,
        reduced,
        shell_edges_rg,
    )
    weights, physical_amplitudes, stress_metric = _state_metric(
        initial,
        vector,
        reduced["primitive_column_scales"],
    )
    named_directions = wp10c8h._redistribution_directions(
        initial,
        vector,
        np.asarray(reduced["primitive_column_scales"], dtype=float),
    )
    stability_arrays, stability_metadata = (
        _generator_stability_audit(
            initial,
            vector,
            named_directions,
            base_reduced=reduced,
            base_evolving=evolving,
            full_scan=label in FULL_GENERATOR_STABILITY_ANCHORS,
        )
    )
    arrays: dict[str, np.ndarray] = {
        "historical_backward_difference_descriptor": np.asarray(
            reduced["descriptor_reduced_scaled_matrix"],
            dtype=float,
        ),
        "direct_vector_storage_descriptor": np.asarray(
            evolving["descriptor_reduced_scaled_matrix"],
            dtype=float,
        ),
        "conserved_vector_storage_descriptor": np.asarray(
            evolving["conserved_descriptor_reduced_scaled_matrix"],
            dtype=float,
        ),
        "vertical_vector_storage_descriptor": np.asarray(
            evolving["vertical_descriptor_reduced_scaled_matrix"],
            dtype=float,
        ),
        "stationary_jacobian": np.asarray(
            reduced["stationary_reduced_scaled_jacobian"],
            dtype=float,
        ),
        "storage_rate_derivative": np.asarray(
            evolving["storage_rate_derivative_scaled_matrix"],
            dtype=float,
        ),
        "conserved_storage_rate_derivative": np.asarray(
            evolving[
                "conserved_storage_rate_derivative_scaled_matrix"
            ],
            dtype=float,
        ),
        "vertical_storage_rate_derivative": np.asarray(
            evolving[
                "vertical_storage_rate_derivative_scaled_matrix"
            ],
            dtype=float,
        ),
        "evolving_jacobian": np.asarray(
            evolving["evolving_reduced_scaled_jacobian"],
            dtype=float,
        ),
        "dynamic": np.asarray(
            evolving["evolving_scaled_generator_per_s"],
            dtype=float,
        ),
        "scaled_primitive_rate": np.asarray(
            evolving["scaled_primitive_rate_per_s"],
            dtype=float,
        ).ravel(),
        "primitive_column_scales": np.asarray(
            reduced["primitive_column_scales"],
            dtype=float,
        ),
        "conservation_row_scales": np.asarray(
            evolving["conservation_row_scales"],
            dtype=float,
        ),
        "state_weights": weights,
        "physical_input_amplitudes": physical_amplitudes,
        "named_regression_directions": np.column_stack(
            tuple(named_directions.values())
        ),
        "scientific_output_matrix": np.asarray(
            outputs["matrix"],
            dtype=float,
        ),
        "scientific_output_gates": np.asarray(
            outputs["gates"],
            dtype=float,
        ),
        "log_h_over_r_profile": np.asarray(
            outputs["log_h_over_r_profile"],
            dtype=float,
        ),
        "interface_flux_jacobian": np.asarray(
            ladder.interface_flux_jacobian,
            dtype=float,
        ),
        "interface_flux_values": np.asarray(
            ladder.interface_flux_values,
            dtype=float,
        ),
        "interface_flux_scales": np.asarray(
            ladder.interface_flux_scales,
            dtype=float,
        ),
        "shell_edges_rg": np.asarray(ladder.geometry.edges_rg, dtype=float),
        "radius_rg": np.asarray(
            context.grid.centers / context.grid.gravitational_radius,
            dtype=float,
        ),
        "grid_edges_rg": np.asarray(
            context.grid.edges / context.grid.gravitational_radius,
            dtype=float,
        ),
        "reconstruction_admissibility_factors": (
            admissibility_factors
        ),
        "shell_edge_indices": np.asarray(
            ladder.geometry.edge_indices,
            dtype=float,
        ),
        "vertical_storage_per_ct": np.asarray(
            evolving["base_storage_action"]["vertical_storage_per_ct"],
            dtype=float,
        ),
        "total_storage_per_ct": np.asarray(
            evolving["base_storage_action"][
                "total_conservation_storage_per_ct"
            ],
            dtype=float,
        ),
        **stability_arrays,
    }
    level_metadata = []
    for index, level in enumerate(ladder.levels):
        prefix = f"level_{index}"
        arrays[f"{prefix}_constraints"] = np.asarray(
            level.constraint_matrix,
            dtype=float,
        )
        arrays[f"{prefix}_raw_constraints"] = np.asarray(
            level.raw_constraint_matrix,
            dtype=float,
        )
        arrays[f"{prefix}_conditioned_constraints"] = np.asarray(
            level.conditioned_constraint_matrix,
            dtype=float,
        )
        arrays[f"{prefix}_coordinate_values"] = np.asarray(
            level.coordinate_values,
            dtype=float,
        )
        arrays[f"{prefix}_coordinate_scales"] = np.asarray(
            level.coordinate_scales,
            dtype=float,
        )
        level_metadata.append(
            {
                "name": level.name,
                "coordinate_count": level.coordinate_count,
                "coordinate_names": level.coordinate_names,
                "coordinate_families": level.coordinate_families,
            }
        )
    storage = _storage_audit(evolving)
    if _operator_contract(context, shell_edges_rg) != operator_contract:
        raise RuntimeError(
            "WP10c8i operator source changed during cache construction"
        )
    metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "work_package": "WP10c8i",
        "base_commit": BASE_COMMIT,
        "n_cells": n_cells,
        "anchor_label": label,
        "anchor_role": anchor_role,
        "state_vector_sha256": _array_sha256(vector),
        "operator_contract": operator_contract,
        "operator_contract_sha256": _text_sha256(
            json.dumps(operator_contract, sort_keys=True)
        ),
        "shell_edges_rg": ladder.geometry.edges_rg,
        "storage_semantics": ladder.storage_semantics,
        "levels": level_metadata,
        "scientific_output_names": outputs["names"],
        "interface_flux_names": ladder.interface_flux_names,
        "named_regression_direction_names": tuple(named_directions),
        "operator_wall_seconds": time.perf_counter() - started,
        "historical_descriptor_rank": int(
            np.linalg.matrix_rank(
                arrays["historical_backward_difference_descriptor"]
            )
        ),
        "historical_descriptor_condition_estimate": float(
            np.linalg.cond(
                arrays["historical_backward_difference_descriptor"]
            )
        ),
        "direct_vector_storage_descriptor_rank": int(
            np.linalg.matrix_rank(
                arrays["direct_vector_storage_descriptor"]
            )
        ),
        "direct_vector_storage_descriptor_condition_estimate": float(
            np.linalg.cond(
                arrays["direct_vector_storage_descriptor"]
            )
        ),
        "algebraic_solve_relative_defect": reduced[
            "algebraic_solve_relative_defect"
        ],
        "maximum_scaled_algebraic_reconstruction_defect": reduced[
            "maximum_scaled_algebraic_reconstruction_defect"
        ],
        "rate_source": evolving["rate_source"],
        "stress_input_metric": stress_metric,
        "storage_audit": storage,
        "tangent_differentiability_audit": differentiability_audit,
        "generator_stability_audit": stability_metadata,
    }
    return arrays, metadata


def _write_cache(
    path: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **arrays,
        metadata_json=np.asarray(
            json.dumps(
                _plain(metadata),
                sort_keys=True,
                allow_nan=False,
            )
        ),
    )


def _load_cache(
    path: Path,
    *,
    context,
    n_cells: int,
    label: str,
    vector: np.ndarray,
    shell_edges_rg: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as source:
        metadata = json.loads(str(source["metadata_json"].item()))
        arrays = {
            name: np.asarray(source[name], dtype=float)
            for name in source.files
            if name != "metadata_json"
        }
    expected_contract = _operator_contract(context, shell_edges_rg)
    if not (
        metadata.get("schema_version") == CACHE_SCHEMA_VERSION
        and metadata.get("work_package") == "WP10c8i"
        and metadata.get("base_commit") == BASE_COMMIT
        and metadata.get("n_cells") == n_cells
        and metadata.get("anchor_label") == label
        and metadata.get("state_vector_sha256") == _array_sha256(vector)
        and metadata.get("operator_contract") == expected_contract
        and metadata.get("operator_contract_sha256")
        == _text_sha256(json.dumps(expected_contract, sort_keys=True))
        and np.array_equal(
            np.asarray(metadata.get("shell_edges_rg"), dtype=float),
            np.asarray(shell_edges_rg, dtype=float),
        )
        and arrays["dynamic"].shape == (5 * n_cells, 5 * n_cells)
        and np.all(np.isfinite(arrays["dynamic"]))
    ):
        raise RuntimeError(f"WP10c8i cache {path.name} differs")
    return arrays, metadata


def _operator_cache(
    initial: dict,
    vector: np.ndarray,
    label: str,
    anchor_role: str,
    shell_edges_rg: np.ndarray,
    *,
    force: bool,
) -> tuple[dict[str, np.ndarray], dict, dict]:
    n_cells = initial["state"].n_cells
    path = _cache_path(n_cells, label)
    rebuild = bool(force or not path.exists())
    if not rebuild:
        try:
            arrays, metadata = _load_cache(
                path,
                context=initial["context"],
                n_cells=n_cells,
                label=label,
                vector=vector,
                shell_edges_rg=shell_edges_rg,
            )
        except (KeyError, RuntimeError, ValueError, json.JSONDecodeError):
            rebuild = True
    if rebuild:
        arrays, metadata = _build_operator_cache(
            initial,
            vector,
            label,
            anchor_role,
            shell_edges_rg,
        )
        _write_cache(path, arrays, metadata)
        arrays, metadata = _load_cache(
            path,
            context=initial["context"],
            n_cells=n_cells,
            label=label,
            vector=vector,
            shell_edges_rg=shell_edges_rg,
        )
    return arrays, metadata, {
        "path": _relative(path),
        "sha256": _sha256(path),
        "state_vector_sha256": _array_sha256(vector),
    }


def _instantaneous_coordinate_shell_component(
    name: str,
) -> tuple[int, int] | None:
    components = {
        "rest_mass": 0,
        "radial_momentum": 1,
        "angular_momentum": 2,
        "killing_energy": 3,
        "stress_storage": 4,
    }
    for suffix, component in components.items():
        marker = f"_{suffix}"
        if name.startswith("shell_") and name.endswith(marker):
            return int(name.split("_", 2)[1]), component
    return None


def _rate_output_rows(
    arrays: dict[str, np.ndarray],
    metadata: dict,
    level_index: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return the WP10c8h-compatible natural-coordinate rate response.

    ``constraint_matrix`` is already divided by each coordinate's natural
    scale.  Multiplying ``C L`` by the declared 0.025-s screening window
    therefore measures the fraction of one natural coordinate scale changed
    by an unresolved unit-amplitude perturbation during that window.  A gate
    of one makes the global 0.25 screening contract literal rather than
    normalizing each row by its own unrestricted worst-case gain.
    """

    level = metadata["levels"][level_index]
    constraints = arrays[f"level_{level_index}_constraints"]
    raw_constraints = arrays[f"level_{level_index}_raw_constraints"]
    coordinate_scales = arrays[
        f"level_{level_index}_coordinate_scales"
    ]
    moving = np.zeros_like(constraints)
    maximum_validation_defect = 0.0
    shell_edges = np.asarray(
        arrays["shell_edge_indices"],
        dtype=int,
    )
    row_scales = np.asarray(
        arrays["conservation_row_scales"],
        dtype=float,
    )
    conserved_mass = arrays["conserved_vector_storage_descriptor"]
    conserved_dm = arrays["conserved_storage_rate_derivative"]
    for coordinate_index, coordinate_name in enumerate(
        level["coordinate_names"]
    ):
        shell_component = _instantaneous_coordinate_shell_component(
            coordinate_name
        )
        if shell_component is None:
            continue
        shell, component = shell_component
        cells = np.arange(shell_edges[shell], shell_edges[shell + 1])
        rows = 5 * cells + component
        weighted_mass = (
            C
            * np.sum(
                row_scales[rows, None] * conserved_mass[rows],
                axis=0,
            )
        )
        raw = raw_constraints[coordinate_index]
        scale = max(
            float(np.max(np.abs(raw))),
            float(np.max(np.abs(weighted_mass))),
            np.finfo(float).tiny,
        )
        maximum_validation_defect = max(
            maximum_validation_defect,
            float(np.max(np.abs(raw - weighted_mass))) / scale,
        )
        moving[coordinate_index] = (
            C
            * np.sum(
                row_scales[rows, None] * conserved_dm[rows],
                axis=0,
            )
            / coordinate_scales[coordinate_index]
        )
    if (
        maximum_validation_defect
        > MAXIMUM_COORDINATE_STORAGE_ROW_RELATIVE_DEFECT
    ):
        raise RuntimeError(
            "instantaneous coordinate row does not match conserved storage"
        )
    rate_rows = (
        COORDINATE_RATE_WINDOW_SECONDS
        * (
            np.asarray(constraints, dtype=float)
            @ np.asarray(arrays["dynamic"], dtype=float)
            + moving
        )
    )
    gates = np.full(
        rate_rows.shape[0],
        COORDINATE_RATE_WINDOW_GATE,
        dtype=float,
    )
    return rate_rows, gates, {
        "moving_coordinate_term_included": True,
        "moving_coordinate_term_semantics": (
            "DC[delta_scaled_primitive] times base_primitive_rate"
        ),
        "maximum_conserved_storage_row_relative_defect": (
            maximum_validation_defect
        ),
        "maximum_absolute_moving_coordinate_rate_entry_per_s": float(
            np.max(np.abs(moving))
        ),
    }


def _response_stack(
    arrays: dict[str, np.ndarray],
    metadata: dict,
    level_index: int,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], dict]:
    level = metadata["levels"][level_index]
    constraints = arrays[f"level_{level_index}_constraints"]
    scientific = arrays["scientific_output_matrix"]
    scientific_gates = arrays["scientific_output_gates"]
    log_h = arrays["log_h_over_r_profile"]
    h_gate = CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1[
        "maximum_log_h_over_r_profile"
    ]
    interface = arrays["interface_flux_jacobian"]
    rate_rows, rate_gates, rate_diagnostics = _rate_output_rows(
        arrays,
        metadata,
        level_index,
    )
    matrix = np.vstack((scientific, log_h, interface, rate_rows))
    gates = np.concatenate(
        (
            scientific_gates,
            np.full(log_h.shape[0], h_gate, dtype=float),
            np.full(
                interface.shape[0],
                INTERFACE_FLUX_RELATIVE_GATE,
                dtype=float,
            ),
            rate_gates,
        )
    )
    names = (
        *metadata["scientific_output_names"],
        *(
            f"log_h_over_r_cell_{index}"
            for index in range(log_h.shape[0])
        ),
        *metadata["interface_flux_names"],
        *(
            f"coordinate_rate_{name}"
            for name in level["coordinate_names"]
        ),
    )
    blocks = {
        "scientific": (0, scientific.shape[0]),
        "full_log_h_over_r": (
            scientific.shape[0],
            scientific.shape[0] + log_h.shape[0],
        ),
        "macro_interface_flux": (
            scientific.shape[0] + log_h.shape[0],
            scientific.shape[0] + log_h.shape[0] + interface.shape[0],
        ),
        "coarse_coordinate_rate": (
            scientific.shape[0] + log_h.shape[0] + interface.shape[0],
            matrix.shape[0],
        ),
        "coordinate_rate_window_seconds": (
            COORDINATE_RATE_WINDOW_SECONDS
        ),
        "coordinate_rate_window_gate": COORDINATE_RATE_WINDOW_GATE,
        "coordinate_rate_diagnostics": rate_diagnostics,
    }
    return matrix, gates, tuple(names), blocks


def _audit_summary(
    audit,
    output_names: tuple[str, ...],
    physical_scale_over_amplitude: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Return rigorous L2-plus-box bounds and nonbinding diagnostics.

    The continuum-L2 maximum is supplemented with declared pointwise
    primitive-amplitude bounds.  The admissible lower gain is achieved by a
    rescaled null direction and proves failure when above a gate.  The
    admissible upper gain is rigorous and proves a pass when below a gate.
    Values between those bounds are explicitly inconclusive.
    """

    row_gains = np.asarray(
        audit.per_output_maximum_gains,
        dtype=float,
    )
    lower_gains = np.asarray(
        audit.per_output_admissible_lower_gains,
        dtype=float,
    )
    upper_gains = np.asarray(
        audit.per_output_admissible_upper_gains,
        dtype=float,
    )
    pointwise_ratios = np.asarray(
        audit.per_output_l2_maximum_pointwise_ratios,
        dtype=float,
    )
    controlling = int(audit.controlling_admissible_output_index)
    upper_controlling = int(np.argmax(upper_gains))
    maximum = float(audit.maximum_per_output_gain)
    leading_state = np.asarray(
        audit.controlling_admissible_state_direction,
        dtype=float,
    )
    leading_output = np.asarray(
        audit.controlling_admissible_gate_normalized_output_response,
        dtype=float,
    )
    amplitude_ratios = np.abs(
        np.asarray(physical_scale_over_amplitude, dtype=float)
        * leading_state
    ).reshape(-1, 5)
    component_amplitude_ratios = np.max(amplitude_ratios, axis=0)
    return {
        "maximum_gate_normalized_gain": maximum,
        "maximum_admissible_lower_gain": (
            audit.maximum_admissible_lower_gain
        ),
        "maximum_admissible_upper_gain": (
            audit.maximum_admissible_upper_gain
        ),
        "binding_contract": (
            "fail_if_lower_exceeds_gate_pass_only_if_upper_within_gate"
        ),
        "stacked_spectral_gain_diagnostic": audit.maximum_gain,
        "constraint_rank": audit.null_basis_audit.constraint_rank,
        "constraint_active_row_count": (
            audit.null_basis_audit.active_row_count
        ),
        "constraint_nullity": audit.null_basis_audit.nullity,
        "constraint_condition_estimate": (
            audit.null_basis_audit.condition_estimate
        ),
        "raw_constraint_defect": (
            audit.null_basis_audit.raw_constraint_defect
        ),
        "weighted_orthogonality_defect": (
            audit.null_basis_audit.weighted_orthogonality_defect
        ),
        "controlling_output": output_names[controlling],
        "controlling_admissible_lower_output": output_names[controlling],
        "controlling_admissible_upper_output": (
            output_names[upper_controlling]
        ),
        "controlling_admissible_upper_gain": float(
            upper_gains[upper_controlling]
        ),
        "controlling_gate_normalized_response": (
            leading_output[controlling]
        ),
        "controlling_l2_maximum_pointwise_ratio": (
            pointwise_ratios[controlling]
        ),
        "controlling_direction_maximum_declared_amplitude_ratio": float(
            np.max(component_amplitude_ratios)
        ),
        "controlling_direction_component_amplitude_ratios": {
            name: component_amplitude_ratios[index]
            for index, name in enumerate(
                (
                    "log_surface_density",
                    "radial_three_velocity_over_c",
                    "azimuthal_three_velocity_over_c",
                    "log_temperature",
                    "specific_causal_stress",
                )
            )
        },
    }, {
        "binding_leading_state": leading_state,
        "binding_leading_gate_normalized_output": leading_output,
        "binding_output_row_gains": row_gains,
        "admissible_lower_row_gains": lower_gains,
        "admissible_upper_row_gains": upper_gains,
        "l2_maximum_pointwise_ratios": pointwise_ratios,
        "admissible_leading_state_subspace": (
            audit.admissible_leading_state_subspace
        ),
        "admissible_leading_output_indices": (
            audit.admissible_leading_output_indices
        ),
        "spectral_singular_values": audit.singular_values,
    }


def _named_regression_audit(
    response: np.ndarray,
    gates: np.ndarray,
    constraints: np.ndarray,
    state_weights: np.ndarray,
    directions: np.ndarray,
    names: tuple[str, ...],
    finite_time_operators: dict[str, np.ndarray],
) -> tuple[dict, np.ndarray]:
    basis = causal_weighted_constraint_null_basis(
        constraints,
        state_weights=state_weights,
    )
    values = np.asarray(directions, dtype=float)
    metric = np.asarray(state_weights, dtype=float)
    weighted_values = metric[:, None] * values
    projected = basis.basis @ (basis.basis.T @ weighted_values)
    rows = {}
    amplitude_preserving_projected = np.asarray(projected, dtype=float)
    for index, name in enumerate(names):
        original_norm = float(
            np.sqrt(np.sum(metric * np.square(values[:, index])))
        )
        projected_norm = float(
            np.sqrt(np.sum(metric * np.square(projected[:, index])))
        )
        correction_norm = float(
            np.sqrt(
                np.sum(
                    metric
                    * np.square(values[:, index] - projected[:, index])
                )
            )
        )
        eligible = bool(
            original_norm > np.finfo(float).tiny
            and projected_norm > np.finfo(float).tiny
            and correction_norm / original_norm
            <= MAXIMUM_NAMED_DIRECTION_CORRECTION_FRACTION
        )
        responses = {}
        maximum = 0.0
        for kind in RESPONSE_KINDS:
            normalized = (
                finite_time_operators[kind]
                @ amplitude_preserving_projected[:, index]
            ) / gates
            value = float(np.max(np.abs(normalized)))
            responses[kind] = value
            maximum = max(maximum, value)
        rows[name] = {
            "original_weighted_norm": original_norm,
            "projected_weighted_norm": projected_norm,
            "physical_amplitude_preserved": True,
            "correction_fraction": (
                correction_norm
                / max(original_norm, np.finfo(float).tiny)
            ),
            "eligible": eligible,
            "maximum_gate_normalized_response": maximum,
            "response_at_0p025_seconds": responses,
            "passed": bool(
                eligible
                and maximum <= SCREENING_MAXIMUM_GATE_FRACTION
            ),
        }
    return rows, amplitude_preserving_projected


def _rusanov_kink_frechet_output_operators(
    dynamic: np.ndarray,
    outputs: np.ndarray,
    left_factors: np.ndarray,
    right_factors: np.ndarray,
    horizon_seconds: float,
    *,
    quadrature_order: int,
) -> np.ndarray:
    """Return ``O Dexp[L h](u_i v_i^T h)`` for every branch kink.

    Rank-one generator differences allow the Fréchet integral to be evaluated
    with batched vector exponential actions rather than dense doubled block
    exponentials.
    """

    system = np.asarray(dynamic, dtype=float)
    output_matrix = np.asarray(outputs, dtype=float)
    left = np.asarray(left_factors, dtype=float)
    right = np.asarray(right_factors, dtype=float)
    horizon = float(horizon_seconds)
    kink_count = int(left.shape[1])
    if (
        system.ndim != 2
        or system.shape[0] != system.shape[1]
        or output_matrix.shape[1] != system.shape[0]
        or left.shape != right.shape
        or left.shape[0] != system.shape[0]
    ):
        raise ValueError("Rusanov Fréchet factors have incompatible shapes")
    result = np.zeros(
        (kink_count, output_matrix.shape[0], system.shape[0]),
        dtype=float,
    )
    if horizon == 0.0 or kink_count == 0:
        return result
    nodes, weights = np.polynomial.legendre.leggauss(
        int(quadrature_order)
    )
    times = 0.5 * horizon * (nodes + 1.0)
    integration_weights = 0.5 * horizon * weights
    trace = float(np.trace(system))
    for sample_time, integration_weight in zip(
        times,
        integration_weights,
    ):
        propagated_left = expm_multiply(
            (horizon - sample_time) * system,
            left,
            traceA=(horizon - sample_time) * trace,
        )
        propagated_right = expm_multiply(
            sample_time * system.T,
            right,
            traceA=sample_time * trace,
        )
        observed_left = output_matrix @ propagated_left
        result += (
            integration_weight
            * np.einsum(
                "oi,ni->ion",
                observed_left,
                propagated_right,
                optimize=True,
            )
        )
    return result


def _rusanov_kink_null_upper_bound(
    operators: np.ndarray,
    constraints: np.ndarray,
    gates: np.ndarray,
    state_weights: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return a triangle upper bound over all generalized branch choices."""

    response = np.asarray(operators, dtype=float)
    if response.shape[0] == 0:
        zeros = np.zeros(np.asarray(gates).shape, dtype=float)
        return zeros, 0.0
    basis = causal_weighted_constraint_null_basis(
        constraints,
        state_weights=state_weights,
    ).basis
    normalized = response / np.asarray(gates, dtype=float)[None, :, None]
    projected = np.einsum(
        "ion,nk->iok",
        normalized,
        basis,
        optimize=True,
    )
    per_output = np.sum(
        np.linalg.norm(projected, axis=2),
        axis=0,
    )
    return per_output, float(np.max(per_output, initial=0.0))


def _rusanov_kink_instantaneous_output_deltas(
    operator_arrays: dict,
    operator_metadata: dict,
    level_index: int,
) -> np.ndarray:
    """Return branch-induced changes in the instantaneous output rows."""

    left = np.asarray(
        operator_arrays[
            "production_rusanov_kink_physical_flux_left_factors"
        ],
        dtype=float,
    )
    right = np.asarray(
        operator_arrays[
            "production_rusanov_kink_generator_right_factors"
        ],
        dtype=float,
    )
    faces = np.asarray(
        operator_arrays["production_rusanov_kink_face_indices"],
        dtype=int,
    )
    kink_count = int(right.shape[1])
    response, _gates, _names, blocks = _response_stack(
        operator_arrays,
        operator_metadata,
        level_index,
    )
    deltas = np.zeros(
        (kink_count, response.shape[0], response.shape[1]),
        dtype=float,
    )
    if kink_count == 0:
        return deltas

    interface_start, interface_end = blocks["macro_interface_flux"]
    interface_scales = np.asarray(
        operator_arrays["interface_flux_scales"],
        dtype=float,
    )
    shell_faces = np.asarray(
        operator_arrays["shell_edge_indices"],
        dtype=int,
    )
    component_by_name = {
        "rest_mass": 0,
        "angular_momentum": 2,
        "killing_energy": 3,
    }
    for local_row, name in enumerate(
        operator_metadata["interface_flux_names"]
    ):
        parts = str(name).split("_", 2)
        boundary_index = int(parts[1])
        component = component_by_name[parts[2]]
        physical_face = int(shell_faces[boundary_index])
        for kink_index in np.flatnonzero(faces == physical_face):
            deltas[
                kink_index,
                interface_start + local_row,
            ] = (
                left[component, kink_index]
                * right[:, kink_index]
                / interface_scales[local_row]
            )
    if interface_end - interface_start != len(
        operator_metadata["interface_flux_names"]
    ):
        raise RuntimeError("interface response block metadata differ")

    rate_start, rate_end = blocks["coarse_coordinate_rate"]
    constraints = np.asarray(
        operator_arrays[f"level_{level_index}_constraints"],
        dtype=float,
    )
    generator_left = np.asarray(
        operator_arrays[
            "production_rusanov_kink_generator_left_factors"
        ],
        dtype=float,
    )
    for kink_index in range(kink_count):
        deltas[kink_index, rate_start:rate_end] = (
            COORDINATE_RATE_WINDOW_SECONDS
            * np.outer(
                constraints @ generator_left[:, kink_index],
                right[:, kink_index],
            )
        )
    return deltas


def _common_log_h_interpolation(
    radius_rg: np.ndarray,
    grid_edges_rg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a fixed common-radius linear output reconstruction."""

    radius = np.asarray(radius_rg, dtype=float)
    edges = np.asarray(grid_edges_rg, dtype=float)
    if (
        radius.ndim != 1
        or radius.size < 2
        or np.any(np.diff(radius) <= 0.0)
        or edges.ndim != 1
        or edges.size < 2
    ):
        raise ValueError("invalid radial geometry for common H/R outputs")
    common = np.geomspace(
        float(edges[0]),
        float(edges[-1]),
        COMMON_LOG_H_CROSS_MESH_SAMPLE_COUNT,
    )
    interpolation = np.zeros((common.size, radius.size), dtype=float)
    log_radius = np.log(radius)
    for row, target in enumerate(common):
        if target <= radius[0]:
            interpolation[row, 0] = 1.0
            continue
        if target >= radius[-1]:
            interpolation[row, -1] = 1.0
            continue
        right = int(np.searchsorted(radius, target, side="right"))
        left = right - 1
        fraction = float(
            (np.log(target) - log_radius[left])
            / (log_radius[right] - log_radius[left])
        )
        interpolation[row, left] = 1.0 - fraction
        interpolation[row, right] = fraction
    return common, interpolation


def _run_null_audits(
    operator_arrays: dict,
    operator_metadata: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    rows = {}
    traces: dict[str, np.ndarray] = {}
    dynamic = operator_arrays["dynamic"]
    weights = operator_arrays["state_weights"]
    physical_scale_over_amplitude = (
        operator_arrays["primitive_column_scales"]
        / operator_arrays["physical_input_amplitudes"]
    )
    final_level_index = len(operator_metadata["levels"]) - 1
    (
        final_response,
        final_gates,
        final_names,
        final_blocks,
    ) = _response_stack(
        operator_arrays,
        operator_metadata,
        final_level_index,
    )
    common_output_count = int(
        final_blocks["coarse_coordinate_rate"][0]
    )
    finite_time_operators = {}
    for horizon in FINITE_TIME_HORIZONS_SECONDS:
        endpoint = causal_finite_time_output_operator(
            dynamic,
            final_response,
            horizon,
            response_kind="endpoint",
        )
        finite_time_operators[("endpoint", horizon)] = endpoint
        finite_time_operators[("increment", horizon)] = (
            endpoint - final_response
        )
    kink_left = np.asarray(
        operator_arrays[
            "production_rusanov_kink_generator_left_factors"
        ],
        dtype=float,
    )
    kink_right = np.asarray(
        operator_arrays[
            "production_rusanov_kink_generator_right_factors"
        ],
        dtype=float,
    )
    has_consequential_kink = bool(kink_left.shape[1] > 0)
    kink_frechet_operators = {}
    for order in RUSANOV_FRECHET_QUADRATURE_ORDERS:
        for horizon in FINITE_TIME_HORIZONS_SECONDS:
            kink_frechet_operators[(order, horizon)] = (
                _rusanov_kink_frechet_output_operators(
                    dynamic,
                    final_response,
                    kink_left,
                    kink_right,
                    horizon,
                    quadrature_order=order,
                )
            )
    common_log_h_radius, common_log_h_interpolation = (
        _common_log_h_interpolation(
            operator_arrays["radius_rg"],
            operator_arrays["grid_edges_rg"],
        )
    )
    for level_index, level in enumerate(operator_metadata["levels"]):
        (
            level_response,
            level_gates,
            level_names,
            blocks,
        ) = _response_stack(
            operator_arrays,
            operator_metadata,
            level_index,
        )
        output_count = common_output_count + level["coordinate_count"]
        response = final_response[:output_count]
        gates = final_gates[:output_count]
        names = final_names[:output_count]
        if not (
            response.shape == level_response.shape
            and np.allclose(response, level_response, rtol=0.0, atol=0.0)
            and np.array_equal(gates, level_gates)
            and names == level_names
        ):
            raise RuntimeError(
                "cumulative moment response stack is not a strict prefix"
            )
        constraints = operator_arrays[
            f"level_{level_index}_constraints"
        ]
        level_rows = {
            "coordinate_count": level["coordinate_count"],
            "coordinate_names": level["coordinate_names"],
            "coordinate_families": level["coordinate_families"],
            "output_count": response.shape[0],
            "response_blocks": blocks,
            "finite_time": {},
        }
        named_rows, named_projected = _named_regression_audit(
            response,
            gates,
            constraints,
            weights,
            operator_arrays["named_regression_directions"],
            tuple(operator_metadata["named_regression_direction_names"]),
            {
                kind: finite_time_operators[
                    (kind, COORDINATE_RATE_WINDOW_SECONDS)
                ][:output_count]
                for kind in RESPONSE_KINDS
            },
        )
        level_rows["named_regression_directions"] = named_rows
        traces[f"{level['name']}_named_projected_directions"] = (
            named_projected
        )
        instantaneous_kink_deltas = (
            _rusanov_kink_instantaneous_output_deltas(
                operator_arrays,
                operator_metadata,
                level_index,
            )
        )
        maximum_l2 = 0.0
        maximum_raw_lower = 0.0
        maximum_raw_upper = 0.0
        maximum_lower = 0.0
        maximum_upper = 0.0
        maximum_kink_upper = 0.0
        # The Fréchet calculation below is a first-order generalized-Jacobian
        # sensitivity diagnostic.  Its quadrature defect does not bound the
        # nonlinear exponential remainder or simultaneous branch
        # interactions.  Therefore it may reserve engineering headroom but
        # cannot make an affected finite-time decision binding.
        kink_contract_passed = not has_consequential_kink
        for kind in RESPONSE_KINDS:
            kind_rows = {}
            for horizon in FINITE_TIME_HORIZONS_SECONDS:
                operator = finite_time_operators[
                    (kind, horizon)
                ][:output_count]
                audit = causal_gate_normalized_finite_time_null_gain(
                    operator,
                    constraints,
                    gates,
                    state_weights=weights,
                    state_amplitudes_scaled=(
                        operator_arrays["physical_input_amplitudes"]
                        / operator_arrays["primitive_column_scales"]
                    ),
                )
                key = f"{horizon:.8g}"
                summary, binding_traces = _audit_summary(
                    audit,
                    names,
                    physical_scale_over_amplitude,
                )
                h_start, h_end = blocks["full_log_h_over_r"]
                native_h_gates = np.asarray(
                    gates[h_start:h_end],
                    dtype=float,
                )
                if (
                    native_h_gates.size
                    != common_log_h_interpolation.shape[1]
                    or not np.allclose(
                        native_h_gates,
                        native_h_gates[0],
                        rtol=0.0,
                        atol=0.0,
                    )
                ):
                    raise RuntimeError(
                        "native log-H gate block is not a uniform profile"
                    )
                common_log_h_operator = (
                    common_log_h_interpolation
                    @ operator[h_start:h_end]
                )
                common_log_h_gates = np.full(
                    common_log_h_radius.shape,
                    native_h_gates[0],
                    dtype=float,
                )
                common_log_h_audit = (
                    causal_gate_normalized_finite_time_null_gain(
                        common_log_h_operator,
                        constraints,
                        common_log_h_gates,
                        state_weights=weights,
                        state_amplitudes_scaled=(
                            operator_arrays[
                                "physical_input_amplitudes"
                            ]
                            / operator_arrays[
                                "primitive_column_scales"
                            ]
                        ),
                    )
                )
                (
                    common_log_h_summary,
                    common_log_h_traces,
                ) = _audit_summary(
                    common_log_h_audit,
                    tuple(
                        f"common_log_h_over_r_at_{radius:.9g}_rg"
                        for radius in common_log_h_radius
                    ),
                    physical_scale_over_amplitude,
                )
                summary["common_log_h_profile"] = {
                    "sample_count": common_log_h_radius.size,
                    "maximum_admissible_lower_gain": (
                        common_log_h_summary[
                            "maximum_admissible_lower_gain"
                        ]
                    ),
                    "maximum_admissible_upper_gain": (
                        common_log_h_summary[
                            "maximum_admissible_upper_gain"
                        ]
                    ),
                    "binding_cross_mesh_localization_operator": True,
                }
                if instantaneous_kink_deltas.shape[0]:
                    propagated_delta_output = (
                        causal_finite_time_output_operator(
                            dynamic,
                            instantaneous_kink_deltas.reshape(
                                -1,
                                dynamic.shape[0],
                            ),
                            horizon,
                            response_kind=kind,
                        ).reshape(instantaneous_kink_deltas.shape)
                    )
                else:
                    propagated_delta_output = instantaneous_kink_deltas
                kink_bounds = {}
                for order in RUSANOV_FRECHET_QUADRATURE_ORDERS:
                    total_kink_operator = (
                        propagated_delta_output
                        + kink_frechet_operators[
                            (order, horizon)
                        ][:, :output_count]
                    )
                    per_output_kink, maximum_kink = (
                        _rusanov_kink_null_upper_bound(
                            total_kink_operator,
                            constraints,
                            gates,
                            weights,
                        )
                    )
                    kink_bounds[order] = (
                        per_output_kink,
                        maximum_kink,
                    )
                coarse_order, fine_order = (
                    RUSANOV_FRECHET_QUADRATURE_ORDERS
                )
                coarse_kink = kink_bounds[coarse_order][0]
                fine_kink = kink_bounds[fine_order][0]
                quadrature_difference = np.abs(
                    fine_kink - coarse_kink
                )
                conservative_kink = fine_kink + quadrature_difference
                maximum_conservative_kink = float(
                    np.max(conservative_kink, initial=0.0)
                )
                quadrature_relative_defect = float(
                    np.max(quadrature_difference, initial=0.0)
                    / max(
                        float(np.max(fine_kink, initial=0.0)),
                        np.finfo(float).tiny,
                    )
                )
                kink_passed = bool(
                    maximum_conservative_kink
                    <= MAXIMUM_RUSANOV_FINITE_TIME_GATE_FRACTION
                    and quadrature_relative_defect
                    <= (
                        MAXIMUM_RUSANOV_FRECHET_QUADRATURE_RELATIVE_DEFECT
                    )
                )
                raw_lower = float(
                    summary["maximum_admissible_lower_gain"]
                )
                raw_upper = float(
                    summary["maximum_admissible_upper_gain"]
                )
                raw_lower_rows = np.asarray(
                    binding_traces["admissible_lower_row_gains"],
                    dtype=float,
                )
                raw_upper_rows = np.asarray(
                    binding_traces["admissible_upper_row_gains"],
                    dtype=float,
                )
                robust_lower_rows = raw_lower_rows
                robust_upper_rows = raw_upper_rows
                robust_lower = float(
                    np.max(robust_lower_rows, initial=0.0)
                )
                robust_upper = float(
                    np.max(robust_upper_rows, initial=0.0)
                )
                summary.update(
                    {
                        "raw_maximum_admissible_lower_gain": raw_lower,
                        "raw_maximum_admissible_upper_gain": raw_upper,
                        "rusanov_kink_linearized_reserve_gain": (
                            maximum_conservative_kink
                        ),
                        "rusanov_kink_linearized_quadrature_relative_defect": (
                            quadrature_relative_defect
                        ),
                        "rusanov_kink_linearized_screening_passed": (
                            kink_passed
                        ),
                        "rusanov_finite_branch_contract_passed": (
                            not has_consequential_kink
                        ),
                        "maximum_admissible_lower_gain": robust_lower,
                        "maximum_admissible_upper_gain": robust_upper,
                    }
                )
                kind_rows[key] = summary
                trace_prefix = (
                    f"{level['name']}_{kind}_h_{horizon:.8g}"
                    .replace(".", "p")
                )
                traces[f"{trace_prefix}_leading_state"] = (
                    binding_traces["binding_leading_state"]
                )
                traces[f"{trace_prefix}_binding_output"] = (
                    binding_traces[
                        "binding_leading_gate_normalized_output"
                    ]
                )
                traces[f"{trace_prefix}_binding_row_gains"] = (
                    binding_traces["binding_output_row_gains"]
                )
                traces[f"{trace_prefix}_admissible_lower_row_gains"] = (
                    robust_lower_rows
                )
                traces[f"{trace_prefix}_admissible_upper_row_gains"] = (
                    robust_upper_rows
                )
                traces[
                    f"{trace_prefix}_raw_admissible_lower_row_gains"
                ] = (
                    raw_lower_rows
                )
                traces[
                    f"{trace_prefix}_raw_admissible_upper_row_gains"
                ] = (
                    raw_upper_rows
                )
                traces[
                    f"{trace_prefix}_rusanov_kink_row_reserve"
                ] = (
                    conservative_kink
                )
                traces[
                    f"{trace_prefix}_common_log_h_radius_rg"
                ] = common_log_h_radius
                traces[
                    f"{trace_prefix}_common_log_h_admissible_lower_row_gains"
                ] = common_log_h_traces[
                    "admissible_lower_row_gains"
                ]
                traces[
                    f"{trace_prefix}_common_log_h_admissible_upper_row_gains"
                ] = common_log_h_traces[
                    "admissible_upper_row_gains"
                ]
                traces[f"{trace_prefix}_l2_pointwise_ratios"] = (
                    binding_traces["l2_maximum_pointwise_ratios"]
                )
                traces[f"{trace_prefix}_leading_state_subspace"] = (
                    binding_traces[
                        "admissible_leading_state_subspace"
                    ]
                )
                traces[
                    f"{trace_prefix}_admissible_leading_output_indices"
                ] = (
                    binding_traces[
                        "admissible_leading_output_indices"
                    ]
                )
                traces[f"{trace_prefix}_singular_values"] = (
                    binding_traces["spectral_singular_values"]
                )
                maximum_l2 = max(
                    maximum_l2,
                    summary["maximum_gate_normalized_gain"],
                )
                maximum_lower = max(
                    maximum_lower,
                    summary["maximum_admissible_lower_gain"],
                )
                maximum_upper = max(
                    maximum_upper,
                    summary["maximum_admissible_upper_gain"],
                )
                maximum_raw_lower = max(
                    maximum_raw_lower,
                    raw_lower,
                )
                maximum_raw_upper = max(
                    maximum_raw_upper,
                    raw_upper,
                )
                maximum_kink_upper = max(
                    maximum_kink_upper,
                    maximum_conservative_kink,
                )
                # Deliberately do not promote the first-order Fréchet
                # diagnostic to an exact finite-branch contract.
            level_rows["finite_time"][kind] = kind_rows
        level_rows["maximum_gate_normalized_gain"] = maximum_l2
        level_rows["raw_maximum_admissible_lower_gain"] = (
            maximum_raw_lower
        )
        level_rows["raw_maximum_admissible_upper_gain"] = (
            maximum_raw_upper
        )
        level_rows["maximum_admissible_lower_gain"] = maximum_lower
        level_rows["maximum_admissible_upper_gain"] = maximum_upper
        level_rows["maximum_rusanov_kink_linearized_reserve_gain"] = (
            maximum_kink_upper
        )
        level_rows["rusanov_finite_branch_contract_passed"] = (
            kink_contract_passed
        )
        local_rank_and_conditioning_passed = bool(
            all(
                summary["constraint_rank"]
                == level_rows["coordinate_count"]
                and summary["constraint_active_row_count"]
                == level_rows["coordinate_count"]
                and summary["constraint_condition_estimate"]
                <= MAXIMUM_CONSTRAINT_CONDITION_ESTIMATE
                for kind_rows in level_rows["finite_time"].values()
                for summary in kind_rows.values()
            )
        )
        local_results_numerically_binding = bool(
            local_rank_and_conditioning_passed
            and operator_metadata["storage_audit"]["passed"]
            and operator_metadata["generator_stability_audit"]["passed"]
            and operator_metadata["tangent_differentiability_audit"][
                "passed"
            ]
            and operator_metadata["tangent_differentiability_audit"][
                "finite_time_preflight_screen_passed"
            ]
            and kink_contract_passed
        )
        raw_screening_passed = bool(
            maximum_upper <= SCREENING_MAXIMUM_GATE_FRACTION
        )
        raw_screening_failed_proven = bool(
            maximum_lower > SCREENING_MAXIMUM_GATE_FRACTION
        )
        raw_pre_microburst_passed = bool(
            maximum_upper <= PRE_MICROBURST_MAXIMUM_GATE_FRACTION
        )
        raw_pre_microburst_failed_proven = bool(
            maximum_lower > PRE_MICROBURST_MAXIMUM_GATE_FRACTION
        )
        level_rows["results_numerically_binding"] = (
            local_results_numerically_binding
        )
        level_rows["constraint_rank_and_conditioning_passed"] = (
            local_rank_and_conditioning_passed
        )
        level_rows["raw_screening_upper_bound_passed"] = (
            raw_screening_passed
        )
        level_rows["raw_screening_lower_bound_failed"] = (
            raw_screening_failed_proven
        )
        level_rows["raw_screening_bounds_inconclusive"] = bool(
            maximum_lower <= SCREENING_MAXIMUM_GATE_FRACTION
            < maximum_upper
        )
        level_rows["raw_pre_microburst_upper_bound_passed"] = (
            raw_pre_microburst_passed
        )
        level_rows["raw_pre_microburst_lower_bound_failed"] = (
            raw_pre_microburst_failed_proven
        )
        level_rows["screening_passed"] = bool(
            raw_screening_passed and local_results_numerically_binding
        )
        level_rows["screening_failed_proven"] = bool(
            raw_screening_failed_proven
            and local_results_numerically_binding
        )
        level_rows["screening_inconclusive"] = bool(
            not level_rows["screening_passed"]
            and not level_rows["screening_failed_proven"]
        )
        level_rows["pre_microburst_passed"] = bool(
            raw_pre_microburst_passed
            and local_results_numerically_binding
        )
        level_rows["pre_microburst_failed_proven"] = bool(
            raw_pre_microburst_failed_proven
            and local_results_numerically_binding
        )
        rows[level["name"]] = level_rows
    return rows, traces


def _weighted_cosine(
    left: np.ndarray,
    right: np.ndarray,
    weights: np.ndarray,
) -> float:
    lhs = np.asarray(left, dtype=float)
    rhs = np.asarray(right, dtype=float)
    metric = np.asarray(weights, dtype=float)
    denominator = np.sqrt(
        float(np.sum(metric * lhs**2))
        * float(np.sum(metric * rhs**2))
    )
    if denominator <= np.finfo(float).tiny:
        return 0.0
    return float(abs(np.sum(metric * lhs * rhs)) / denominator)


def _matched_output_rows(
    coarse_arrays: dict[str, np.ndarray],
    coarse_metadata: dict,
    fine_arrays: dict[str, np.ndarray],
    fine_metadata: dict,
    level_index: int,
) -> list[dict]:
    _, _, coarse_names, coarse_blocks = _response_stack(
        coarse_arrays,
        coarse_metadata,
        level_index,
    )
    _, _, fine_names, fine_blocks = _response_stack(
        fine_arrays,
        fine_metadata,
        level_index,
    )
    rows = []
    for block_name in (
        "scientific",
        "macro_interface_flux",
        "coarse_coordinate_rate",
    ):
        coarse_start, coarse_stop = coarse_blocks[block_name]
        fine_start, fine_stop = fine_blocks[block_name]
        coarse_block_names = coarse_names[coarse_start:coarse_stop]
        fine_block_names = fine_names[fine_start:fine_stop]
        if coarse_block_names != fine_block_names:
            raise RuntimeError(
                f"cross-mesh {block_name} output names do not agree"
            )
        for offset, output_name in enumerate(coarse_block_names):
            rows.append(
                {
                    "block": block_name,
                    "output": output_name,
                    "binding_cross_mesh": True,
                    "coarse_indices": (coarse_start + offset,),
                    "coarse_weights": (1.0,),
                    "fine_indices": (fine_start + offset,),
                    "fine_weights": (1.0,),
                    "coarse_radius_rg": None,
                    "fine_radius_rg": None,
                    "common_radius_rg": None,
                }
            )

    coarse_start, coarse_stop = coarse_blocks["full_log_h_over_r"]
    fine_start, fine_stop = fine_blocks["full_log_h_over_r"]
    coarse_radius = np.asarray(coarse_arrays["radius_rg"], dtype=float)
    fine_radius = np.asarray(fine_arrays["radius_rg"], dtype=float)
    if (
        coarse_stop - coarse_start != coarse_radius.size
        or fine_stop - fine_start != fine_radius.size
    ):
        raise RuntimeError("log-H output block does not match mesh radii")

    coarse_edges = np.asarray(
        coarse_arrays.get("grid_edges_rg", coarse_radius[[0, -1]]),
        dtype=float,
    )
    fine_edges = np.asarray(
        fine_arrays.get("grid_edges_rg", fine_radius[[0, -1]]),
        dtype=float,
    )
    if not np.allclose(
        coarse_edges[[0, -1]],
        fine_edges[[0, -1]],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("cross-mesh physical radial domains disagree")
    base_common_radius = np.geomspace(
        coarse_edges[0],
        coarse_edges[-1],
        COMMON_LOG_H_CROSS_MESH_SAMPLE_COUNT,
    )
    # Include every native center on both meshes.  In particular, an N128-only
    # peak cannot disappear merely because no N64 center lies nearby.
    common_radius = np.unique(
        np.concatenate(
            (base_common_radius, coarse_radius, fine_radius)
        )
    )

    def interpolation_stencil(
        source_radius: np.ndarray,
        target_radius: float,
        block_start: int,
    ) -> tuple[tuple[int, ...], tuple[float, ...]]:
        source = np.asarray(source_radius, dtype=float)
        if source.size < 2 or np.any(np.diff(source) <= 0.0):
            raise RuntimeError("log-H radii must be strictly increasing")
        exact = np.flatnonzero(source == target_radius)
        if exact.size:
            return (block_start + int(exact[0]),), (1.0,)
        # Use constant endpoint extension.  Interior values are interpolated
        # linearly in log radius, matching the logarithmic mesh geometry.
        if target_radius <= source[0]:
            return (block_start,), (1.0,)
        if target_radius >= source[-1]:
            return (block_start + source.size - 1,), (1.0,)
        right = int(np.searchsorted(source, target_radius, side="right"))
        left = right - 1
        log_source = np.log(source[[left, right]])
        fraction = float(
            (np.log(target_radius) - log_source[0])
            / (log_source[1] - log_source[0])
        )
        return (
            (block_start + left, block_start + right),
            (1.0 - fraction, fraction),
        )

    for radius in common_radius:
        coarse_indices, coarse_weights = interpolation_stencil(
            coarse_radius,
            float(radius),
            coarse_start,
        )
        fine_indices, fine_weights = interpolation_stencil(
            fine_radius,
            float(radius),
            fine_start,
        )
        rows.append(
            {
                "block": "full_log_h_over_r",
                "output": f"log_h_over_r_at_{radius:.9g}_rg",
                # Interpolating already-optimized rowwise gains does not
                # preserve a mathematical lower-bound interpretation.  These
                # rows therefore diagnose radial localization only.  The
                # binding full-profile information enters through the native
                # per-mesh maximum admissible lower/upper gains below.
                "binding_cross_mesh": False,
                "coarse_indices": coarse_indices,
                "coarse_weights": coarse_weights,
                "fine_indices": fine_indices,
                "fine_weights": fine_weights,
                "coarse_radius_rg": float(radius),
                "fine_radius_rg": float(radius),
                "common_radius_rg": float(radius),
            }
        )
    return rows


def _scalar_cross_mesh_agreement(
    coarse: float,
    fine: float,
) -> dict:
    coarse_value = float(coarse)
    fine_value = float(fine)
    absolute_difference = abs(coarse_value - fine_value)
    activity = max(abs(coarse_value), abs(fine_value))
    active = bool(activity > CROSS_MESH_GAIN_ACTIVITY_FLOOR)
    relative_difference = (
        absolute_difference / activity if active else 0.0
    )
    return {
        "coarse_value": coarse_value,
        "fine_value": fine_value,
        "absolute_difference": absolute_difference,
        "activity_scale": activity,
        "activity_floor": CROSS_MESH_GAIN_ACTIVITY_FLOOR,
        "comparison_active": active,
        "relative_difference": relative_difference,
        "passed": bool(
            not active
            or relative_difference
            <= MAXIMUM_CROSS_MESH_GAIN_RELATIVE_DIFFERENCE
        ),
    }


def _matched_bound_agreement(
    matched_rows: list[dict],
    coarse_values: np.ndarray,
    fine_values: np.ndarray,
) -> dict:
    differences = []
    for row in matched_rows:
        coarse = float(
            np.dot(
                np.asarray(row["coarse_weights"], dtype=float),
                np.asarray(coarse_values, dtype=float)[
                    np.asarray(row["coarse_indices"], dtype=int)
                ],
            )
        )
        fine = float(
            np.dot(
                np.asarray(row["fine_weights"], dtype=float),
                np.asarray(fine_values, dtype=float)[
                    np.asarray(row["fine_indices"], dtype=int)
                ],
            )
        )
        agreement = _scalar_cross_mesh_agreement(coarse, fine)
        differences.append(
            {
                **row,
                "coarse_bound": coarse,
                "fine_bound": fine,
                **agreement,
            }
        )
    if not differences:
        raise RuntimeError("cross-mesh output matching produced no rows")
    binding_differences = [
        row
        for row in differences
        if bool(row.get("binding_cross_mesh", True))
    ]
    if not binding_differences:
        raise RuntimeError(
            "cross-mesh output matching produced no binding rows"
        )

    def compact(row: dict) -> dict:
        return {
            "block": row["block"],
            "output": row["output"],
            "binding_cross_mesh": bool(
                row.get("binding_cross_mesh", True)
            ),
            "coarse_radius_rg": row["coarse_radius_rg"],
            "fine_radius_rg": row["fine_radius_rg"],
            "common_radius_rg": row["common_radius_rg"],
            "coarse_bound": row["coarse_bound"],
            "fine_bound": row["fine_bound"],
            "absolute_difference": row["absolute_difference"],
            "comparison_active": row["comparison_active"],
            "relative_difference": row["relative_difference"],
        }

    controlling = max(
        binding_differences,
        key=lambda row: row["relative_difference"],
    )
    diagnostic_controlling = max(
        differences,
        key=lambda row: row["relative_difference"],
    )
    per_block = {}
    for block_name in (
        "scientific",
        "full_log_h_over_r",
        "macro_interface_flux",
        "coarse_coordinate_rate",
    ):
        block_rows = [
            row for row in differences if row["block"] == block_name
        ]
        if not block_rows:
            continue
        block_controlling = max(
            block_rows,
            key=lambda row: row["relative_difference"],
        )
        per_block[block_name] = {
            "matched_output_count": len(block_rows),
            "binding_output_count": int(
                sum(
                    bool(row.get("binding_cross_mesh", True))
                    for row in block_rows
                )
            ),
            "active_output_count": int(
                sum(row["comparison_active"] for row in block_rows)
            ),
            "maximum_relative_difference": block_controlling[
                "relative_difference"
            ],
            "controlling_output": compact(block_controlling),
            "passed": bool(
                block_controlling["relative_difference"]
                <= MAXIMUM_CROSS_MESH_GAIN_RELATIVE_DIFFERENCE
            ),
            "binding": bool(
                any(
                    bool(row.get("binding_cross_mesh", True))
                    for row in block_rows
                )
            ),
        }
    return {
        "matching_policy": (
            "same named scalar rows bind; full log-H row gains are "
            "interpolated diagnostically "
            "in log radius onto a common base grid augmented by every "
            "N64 and N128 native center, with constant endpoint extension; "
            "native per-mesh maximum admissible bounds remain binding"
        ),
        "matched_output_count": len(differences),
        "binding_output_count": len(binding_differences),
        "active_output_count": int(
            sum(row["comparison_active"] for row in differences)
        ),
        "maximum_relative_difference": controlling["relative_difference"],
        "controlling_output": compact(controlling),
        "diagnostic_maximum_relative_difference": (
            diagnostic_controlling["relative_difference"]
        ),
        "diagnostic_controlling_output": compact(
            diagnostic_controlling
        ),
        "per_block": per_block,
        "passed": bool(
            controlling["relative_difference"]
            <= MAXIMUM_CROSS_MESH_GAIN_RELATIVE_DIFFERENCE
        ),
    }


def _cross_mesh_agreement(
    initial: dict,
    caches: dict,
    results: dict,
    traces: dict,
) -> dict:
    rows = {}
    for label, _seconds, _role in ANCHORS:
        level_rows = {}
        coarse_arrays = caches[64][label][0]
        fine_arrays = caches[128][label][0]
        for level_index, level in enumerate(
            caches[64][label][1]["levels"]
        ):
            name = level["name"]
            comparisons = {}
            matched_rows = _matched_output_rows(
                coarse_arrays,
                caches[64][label][1],
                fine_arrays,
                caches[128][label][1],
                level_index,
            )
            for kind in RESPONSE_KINDS:
                for horizon in FINITE_TIME_HORIZONS_SECONDS:
                    time_key = f"{horizon:.8g}"
                    coarse_result = results[64][label][name][
                        "finite_time"
                    ][kind][time_key]
                    fine_result = results[128][label][name][
                        "finite_time"
                    ][kind][time_key]
                    comparison_key = f"{kind}_h_{time_key}"
                    if kind == "increment" and horizon == 0.0:
                        comparisons[comparison_key] = {
                            "comparison_skipped": True,
                            "skip_reason": (
                                "the h=0 increment operator is analytically "
                                "zero on both meshes"
                            ),
                            "passed": True,
                        }
                        continue
                    trace_key = (
                        f"{name}_{kind}_h_{horizon:.8g}_leading_state"
                        .replace(".", "p")
                    )
                    subspace_key = (
                        f"{name}_{kind}_h_{horizon:.8g}_"
                        "leading_state_subspace"
                    ).replace(".", "p")
                    fine_scaled = traces[128][label][trace_key]
                    fine_physical = (
                        fine_arrays["primitive_column_scales"]
                        * fine_scaled
                    ).reshape(128, 5)
                    restricted = causal_restrict_cell_averages(
                        initial[64]["context"].grid,
                        initial[128]["context"].grid,
                        fine_physical,
                    ).ravel()
                    restricted_scaled = (
                        restricted
                        / coarse_arrays["primitive_column_scales"]
                    )
                    coarse_scaled = traces[64][label][trace_key]
                    fine_subspace = traces[128][label][subspace_key]
                    coarse_subspace = traces[64][label][subspace_key]
                    fine_dimension = int(fine_subspace.shape[1])
                    coarse_dimension = int(coarse_subspace.shape[1])
                    if fine_dimension > 0 and coarse_dimension > 0:
                        fine_physical_subspace = (
                            fine_arrays["primitive_column_scales"][:, None]
                            * fine_subspace
                        ).reshape(128, 5, fine_subspace.shape[1])
                        restricted_physical_subspace = (
                            causal_restrict_cell_averages(
                                initial[64]["context"].grid,
                                initial[128]["context"].grid,
                                fine_physical_subspace,
                            ).reshape(
                                5 * initial[64]["state"].n_cells,
                                fine_subspace.shape[1],
                            )
                        )
                        restricted_scaled_subspace = (
                            restricted_physical_subspace
                            / coarse_arrays[
                                "primitive_column_scales"
                            ][:, None]
                        )
                        root_weight = np.sqrt(
                            coarse_arrays["state_weights"]
                        )[:, None]
                        coarse_weighted_subspace = (
                            root_weight * coarse_subspace
                        )
                        fine_weighted_subspace = (
                            root_weight * restricted_scaled_subspace
                        )
                        coarse_subspace_rank = int(
                            np.linalg.matrix_rank(
                                coarse_weighted_subspace
                            )
                        )
                        fine_subspace_rank = int(
                            np.linalg.matrix_rank(
                                fine_weighted_subspace
                            )
                        )
                        angles_degrees = np.degrees(
                            subspace_angles(
                                coarse_weighted_subspace,
                                fine_weighted_subspace,
                            )
                        )
                        angle_gate = (
                            CROSS_MESH_SUBSPACE_ANGLE_GATE_DEGREES
                        )
                        angle_summary = {
                            "available": True,
                            "coarse_dimension": coarse_dimension,
                            "fine_dimension": fine_dimension,
                            "coarse_weighted_rank": (
                                coarse_subspace_rank
                            ),
                            "restricted_fine_weighted_rank": (
                                fine_subspace_rank
                            ),
                            "dimension_match": bool(
                                coarse_dimension == fine_dimension
                            ),
                            "angle_count": angles_degrees.size,
                            "maximum_principal_angle_degrees": float(
                                np.max(angles_degrees)
                            ),
                            "median_principal_angle_degrees": float(
                                np.median(angles_degrees)
                            ),
                            "principal_angles_degrees": angles_degrees,
                            "passed": bool(
                                coarse_dimension == fine_dimension
                                and coarse_subspace_rank
                                == coarse_dimension
                                and fine_subspace_rank == fine_dimension
                                and
                                float(np.max(angles_degrees))
                                <= angle_gate
                            ),
                        }
                    else:
                        angle_summary = {
                            "available": False,
                            "coarse_dimension": coarse_dimension,
                            "fine_dimension": fine_dimension,
                            "dimension_match": bool(
                                coarse_dimension == fine_dimension
                            ),
                            "angle_count": 0,
                            "passed": bool(
                                coarse_dimension == fine_dimension == 0
                            ),
                        }
                    coarse_gain = coarse_result[
                        "maximum_gate_normalized_gain"
                    ]
                    fine_gain = fine_result[
                        "maximum_gate_normalized_gain"
                    ]
                    trace_prefix = (
                        f"{name}_{kind}_h_{horizon:.8g}".replace(".", "p")
                    )
                    matched_bound_rows = {}
                    for bound_name, trace_suffix in (
                        (
                            "admissible_lower",
                            "admissible_lower_row_gains",
                        ),
                        (
                            "admissible_upper",
                            "admissible_upper_row_gains",
                        ),
                    ):
                        matched_bound_rows[bound_name] = (
                            _matched_bound_agreement(
                                matched_rows,
                                traces[64][label][
                                    f"{trace_prefix}_{trace_suffix}"
                                ],
                                traces[128][label][
                                    f"{trace_prefix}_{trace_suffix}"
                                ],
                            )
                        )
                    coarse_common_radius = traces[64][label][
                        f"{trace_prefix}_common_log_h_radius_rg"
                    ]
                    fine_common_radius = traces[128][label][
                        f"{trace_prefix}_common_log_h_radius_rg"
                    ]
                    if not np.array_equal(
                        coarse_common_radius,
                        fine_common_radius,
                    ):
                        raise RuntimeError(
                            "common log-H operator radii differ across meshes"
                        )
                    common_log_h_rows = [
                        {
                            "block": "common_log_h_over_r_operator",
                            "output": (
                                f"log_h_over_r_at_{radius:.9g}_rg"
                            ),
                            "binding_cross_mesh": True,
                            "coarse_indices": (index,),
                            "coarse_weights": (1.0,),
                            "fine_indices": (index,),
                            "fine_weights": (1.0,),
                            "coarse_radius_rg": float(radius),
                            "fine_radius_rg": float(radius),
                            "common_radius_rg": float(radius),
                        }
                        for index, radius in enumerate(
                            coarse_common_radius
                        )
                    ]
                    common_log_h_bound_rows = {}
                    for bound_name in ("lower", "upper"):
                        common_log_h_bound_rows[bound_name] = (
                            _matched_bound_agreement(
                                common_log_h_rows,
                                traces[64][label][
                                    f"{trace_prefix}_common_log_h_"
                                    f"admissible_{bound_name}_row_gains"
                                ],
                                traces[128][label][
                                    f"{trace_prefix}_common_log_h_"
                                    f"admissible_{bound_name}_row_gains"
                                ],
                            )
                        )
                    gain_scale = max(
                        abs(coarse_gain),
                        abs(fine_gain),
                        np.finfo(float).tiny,
                    )
                    gain_difference = abs(
                        coarse_gain - fine_gain
                    ) / gain_scale
                    maximum_admissible_agreement = {
                        bound_name: _scalar_cross_mesh_agreement(
                            coarse_result[
                                f"maximum_admissible_{bound_name}_gain"
                            ],
                            fine_result[
                                f"maximum_admissible_{bound_name}_gain"
                            ],
                        )
                        for bound_name in ("lower", "upper")
                    }
                    maximum_admissible_agreement_passed = bool(
                        all(
                            row["passed"]
                            for row in maximum_admissible_agreement.values()
                        )
                    )
                    bound_agreement = bool(
                        all(
                            row["passed"]
                            for row in matched_bound_rows.values()
                        )
                        and all(
                            row["passed"]
                            for row in common_log_h_bound_rows.values()
                        )
                    )
                    direction_is_resolved = bool(
                        max(abs(coarse_gain), abs(fine_gain)) > 1.0e-12
                    )
                    direction_is_one_dimensional = bool(
                        coarse_dimension == fine_dimension == 1
                    )
                    direction_applicable = bool(
                        direction_is_resolved
                        and direction_is_one_dimensional
                    )
                    leading_cosine = (
                        _weighted_cosine(
                            coarse_scaled,
                            restricted_scaled,
                            coarse_arrays["state_weights"],
                        )
                        if direction_applicable
                        else None
                    )
                    direction_passed = bool(
                        not direction_applicable
                        or leading_cosine
                        >= MINIMUM_CROSS_MESH_LEADING_DIRECTION_COSINE
                    )
                    comparisons[comparison_key] = {
                        "maximum_gain_relative_difference": gain_difference,
                        "maximum_gain_relative_difference_binding": False,
                        "maximum_gain_agreement_passed": bool(
                            maximum_admissible_agreement_passed
                        ),
                        "maximum_admissible_lower_upper_agreement": (
                            maximum_admissible_agreement
                        ),
                        "matched_per_output_admissible_bound_agreement": (
                            matched_bound_rows
                        ),
                        "binding_common_log_h_operator_bound_agreement": (
                            common_log_h_bound_rows
                        ),
                        "restricted_leading_direction_absolute_cosine": (
                            leading_cosine
                        ),
                        "leading_direction_comparison_applicable": (
                            direction_applicable
                        ),
                        "leading_direction_comparison_reason": (
                            "isolated_one_dimensional_admissible_subspace"
                            if direction_applicable
                            else
                            "principal_angles_are_diagnostic_for_"
                            "multidimensional_or_unresolved_subspace"
                        ),
                        "leading_direction_agreement_passed": (
                            direction_passed
                        ),
                        "leading_subspace_principal_angles": angle_summary,
                        "direction_and_subspace_diagnostic_passed": bool(
                            direction_passed and angle_summary["passed"]
                        ),
                        "direction_and_subspace_are_binding": False,
                        "passed": bool(
                            bound_agreement
                            and maximum_admissible_agreement_passed
                        ),
                    }
            level_rows[name] = comparisons
        rows[label] = level_rows
    return rows


def main() -> None:
    arguments = _arguments()
    if (
        arguments.operator_resolution or arguments.operator_anchor
    ) and not arguments.operators_only:
        raise ValueError(
            "operator resolution/anchor selection requires --operators-only"
        )
    output_path = _absolute(arguments.output)
    arrays_path = _absolute(arguments.arrays)
    authorization, authorization_sha256 = _validate_authorization()
    initial, vectors, state_provenance = _load_states()
    layouts = wp10c8h._common_shell_edges(initial)
    shell_edges_rg = np.asarray(layouts["five_shell"], dtype=float)
    if arguments.preflight:
        differentiability = {
            str(n_cells): {
                label: _tangent_differentiability_audit(
                    initial[n_cells],
                    vectors[n_cells][label],
                )[1]
                for label, _seconds, _role in ANCHORS
            }
            for n_cells in RESOLUTIONS
        }
        all_preflight_passed = all(
            row["passed"]
            for mesh_rows in differentiability.values()
            for row in mesh_rows.values()
        )
        print(
            json.dumps(
                {
                    "work_package": "WP10c8i",
                    "preflight": (
                        "passed" if all_preflight_passed else "blocked"
                    ),
                    "authorization_decision": authorization["decision"],
                    "resolutions": RESOLUTIONS,
                    "anchors": [label for label, _time, _role in ANCHORS],
                    "five_shell_edges_rg": shell_edges_rg.tolist(),
                    "tangent_differentiability": differentiability,
                    "all_tangent_differentiability_passed": (
                        all_preflight_passed
                    ),
                    "new_full_dae_trajectory_run": False,
                    "new_nonlinear_microburst_run": False,
                },
                sort_keys=True,
            )
        )
        if not all_preflight_passed:
            raise RuntimeError(
                "WP10c8i preflight found a nondifferentiable production "
                "branch; operator construction is blocked"
            )
        return

    selected_resolutions = (
        tuple(dict.fromkeys(arguments.operator_resolution))
        if arguments.operator_resolution
        else RESOLUTIONS
    )
    selected_anchor_labels = (
        set(arguments.operator_anchor)
        if arguments.operator_anchor
        else {label for label, _time, _role in ANCHORS}
    )
    selected_anchors = tuple(
        row for row in ANCHORS if row[0] in selected_anchor_labels
    )
    caches: dict[int, dict] = {
        n_cells: {} for n_cells in selected_resolutions
    }
    cache_provenance: dict[str, dict] = {
        str(n_cells): {} for n_cells in selected_resolutions
    }
    for n_cells in selected_resolutions:
        for label, _time_seconds, role in selected_anchors:
            arrays, metadata, provenance = _operator_cache(
                initial[n_cells],
                vectors[n_cells][label],
                label,
                role,
                shell_edges_rg,
                force=arguments.force_operators,
            )
            if not metadata["storage_audit"]["passed"]:
                raise RuntimeError(
                    f"N{n_cells} {label} vector storage audit failed"
                )
            if not metadata["tangent_differentiability_audit"]["passed"]:
                raise RuntimeError(
                    f"N{n_cells} {label} tangent differentiability failed"
                )
            caches[n_cells][label] = (arrays, metadata)
            cache_provenance[str(n_cells)][label] = provenance
            print(
                json.dumps(
                    {
                        "work_package": "WP10c8i",
                        "mode": "operator_cache",
                        "n_cells": n_cells,
                        "anchor": label,
                        "storage_passed": metadata["storage_audit"][
                            "passed"
                        ],
                        "tangent_differentiability_passed": metadata[
                            "tangent_differentiability_audit"
                        ]["passed"],
                        "generator_stability_passed": metadata[
                            "generator_stability_audit"
                        ]["passed"],
                        "cache": provenance["path"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if arguments.operators_only:
        selected_contract_rows = [
            caches[n_cells][label][1]
            for n_cells in selected_resolutions
            for label, _time, _role in selected_anchors
        ]
        failed_generator_contracts = sum(
            not row["generator_stability_audit"]["passed"]
            for row in selected_contract_rows
        )
        print(
            json.dumps(
                {
                    "work_package": "WP10c8i",
                    "operators_only": "completed",
                    "operator_cache_count": (
                        len(selected_resolutions)
                        * len(selected_anchors)
                    ),
                    "all_selected_numerical_contracts_passed": bool(
                        all(
                            row["storage_audit"]["passed"]
                            and row[
                                "tangent_differentiability_audit"
                            ]["passed"]
                            and row[
                                "tangent_differentiability_audit"
                            ][
                                "finite_time_preflight_screen_passed"
                            ]
                            and row["generator_stability_audit"][
                                "passed"
                            ]
                            for row in selected_contract_rows
                        )
                    ),
                    "failed_generator_stability_contract_count": int(
                        failed_generator_contracts
                    ),
                    "selected_resolutions": selected_resolutions,
                    "selected_anchors": tuple(
                        label
                        for label, _time, _role in selected_anchors
                    ),
                    "cache_directory": _relative(CACHE_DIRECTORY),
                },
                sort_keys=True,
            )
        )
        return

    results: dict[int, dict] = {n_cells: {} for n_cells in RESOLUTIONS}
    traces: dict[int, dict] = {n_cells: {} for n_cells in RESOLUTIONS}
    for n_cells in RESOLUTIONS:
        for label, _time_seconds, _role in ANCHORS:
            rows, anchor_traces = _run_null_audits(
                *caches[n_cells][label]
            )
            results[n_cells][label] = rows
            traces[n_cells][label] = anchor_traces
            print(
                json.dumps(
                    {
                        "work_package": "WP10c8i",
                        "mode": "null_gain",
                        "n_cells": n_cells,
                        "anchor": label,
                        "minimum_level_admissible_lower_gain": min(
                            row["maximum_admissible_lower_gain"]
                            for row in rows.values()
                        ),
                        "minimum_level_admissible_upper_gain": min(
                            row["maximum_admissible_upper_gain"]
                            for row in rows.values()
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    cross_mesh = _cross_mesh_agreement(
        initial,
        caches,
        results,
        traces,
    )
    level_names = [
        level["name"] for level in caches[64]["t_0"][1]["levels"]
    ]
    level_decisions = {}
    all_storage_passed = all(
        caches[n_cells][label][1]["storage_audit"]["passed"]
        for n_cells in RESOLUTIONS
        for label, _time, _role in ANCHORS
    )
    all_generator_stability_passed = all(
        caches[n_cells][label][1]["generator_stability_audit"]["passed"]
        for n_cells in RESOLUTIONS
        for label, _time, _role in ANCHORS
    )
    all_tangent_differentiability_passed = all(
        caches[n_cells][label][1]["tangent_differentiability_audit"][
            "passed"
        ]
        for n_cells in RESOLUTIONS
        for label, _time, _role in ANCHORS
    )
    for level_name in level_names:
        construction_rows = [
            results[n_cells][label][level_name]
            for n_cells in RESOLUTIONS
            for label, _time, role in ANCHORS
            if role == "construction"
        ]
        held_out_rows = [
            results[n_cells][label][level_name]
            for n_cells in RESOLUTIONS
            for label, _time, role in ANCHORS
            if role == "held_out"
        ]
        screening = bool(
            all(
                row["raw_screening_upper_bound_passed"]
                for row in construction_rows
            )
            and all(
                row["raw_screening_upper_bound_passed"]
                for row in held_out_rows
            )
        )
        screening_failed_proven = bool(
            any(
                row["raw_screening_lower_bound_failed"]
                for row in construction_rows + held_out_rows
            )
        )
        pre_microburst = bool(
            all(
                row["raw_pre_microburst_upper_bound_passed"]
                for row in construction_rows
            )
            and all(
                row["raw_pre_microburst_upper_bound_passed"]
                for row in held_out_rows
            )
        )
        numerical_rows = construction_rows + held_out_rows
        rank_conditioning = bool(
            all(
                summary["constraint_rank"] == row["coordinate_count"]
                and summary["constraint_active_row_count"]
                == row["coordinate_count"]
                and summary["constraint_condition_estimate"]
                <= MAXIMUM_CONSTRAINT_CONDITION_ESTIMATE
                for row in numerical_rows
                for kind_rows in row["finite_time"].values()
                for summary in kind_rows.values()
            )
        )
        cross_mesh_passed = bool(
            all(
                comparison["passed"]
                for label_rows in cross_mesh.values()
                for comparison in label_rows[level_name].values()
            )
        )
        cross_mesh_direction_diagnostic_passed = bool(
            all(
                comparison.get(
                    "direction_and_subspace_diagnostic_passed",
                    True,
                )
                for label_rows in cross_mesh.values()
                for comparison in label_rows[level_name].values()
            )
        )
        rusanov_finite_time_passed = bool(
            all(
                row["rusanov_finite_branch_contract_passed"]
                for row in numerical_rows
            )
        )
        numerical_contract = bool(
            rank_conditioning
            and all_storage_passed
            and all_generator_stability_passed
            and all_tangent_differentiability_passed
            and rusanov_finite_time_passed
            and cross_mesh_passed
        )
        identifiability = bool(screening and numerical_contract)
        proven_not_identifiable = bool(
            screening_failed_proven and numerical_contract
        )
        inconclusive = bool(
            not identifiability and not proven_not_identifiable
        )
        raw_construction_passed = bool(
            all(
                row["raw_screening_upper_bound_passed"]
                for row in construction_rows
            )
        )
        raw_construction_failed_proven = bool(
            any(
                row["raw_screening_lower_bound_failed"]
                for row in construction_rows
            )
        )
        raw_held_out_passed = bool(
            all(
                row["raw_screening_upper_bound_passed"]
                for row in held_out_rows
            )
        )
        raw_held_out_failed_proven = bool(
            any(
                row["raw_screening_lower_bound_failed"]
                for row in held_out_rows
            )
        )
        # No candidate may pass the complete contract until a cheap online
        # closure exists and is measured.  Operator construction above is an
        # offline full-DAE audit and is not counted as an ordinary reduced
        # evaluation.
        level_decisions[level_name] = {
            "raw_construction_screening_upper_bound_passed": (
                raw_construction_passed
            ),
            "raw_construction_screening_lower_bound_failed": (
                raw_construction_failed_proven
            ),
            "raw_held_out_screening_upper_bound_passed": (
                raw_held_out_passed
            ),
            "raw_held_out_screening_lower_bound_failed": (
                raw_held_out_failed_proven
            ),
            "raw_screening_upper_bound_passed": screening,
            "raw_screening_lower_bound_failed": screening_failed_proven,
            "screening_results_numerically_binding": numerical_contract,
            "construction_screening_passed": bool(
                raw_construction_passed and numerical_contract
            ),
            "construction_screening_failed_proven": bool(
                raw_construction_failed_proven and numerical_contract
            ),
            "held_out_screening_passed": bool(
                raw_held_out_passed and numerical_contract
            ),
            "held_out_screening_failed_proven": bool(
                raw_held_out_failed_proven and numerical_contract
            ),
            "screening_passed": identifiability,
            "screening_failed_proven": proven_not_identifiable,
            "screening_inconclusive": bool(
                not identifiability and not proven_not_identifiable
            ),
            "raw_pre_microburst_gain_passed": pre_microburst,
            "pre_microburst_gain_passed": bool(
                pre_microburst and numerical_contract
            ),
            "constraint_rank_and_conditioning_passed": (
                rank_conditioning
            ),
            "complete_vector_storage_passed": all_storage_passed,
            "generator_fd_stability_passed": (
                all_generator_stability_passed
            ),
            "tangent_differentiability_passed": (
                all_tangent_differentiability_passed
            ),
            "finite_time_null_results_binding": (
                all_tangent_differentiability_passed
                and all_generator_stability_passed
                and rusanov_finite_time_passed
            ),
            "rusanov_exact_finite_branch_contract_passed": (
                rusanov_finite_time_passed
            ),
            "cross_mesh_admissible_bound_agreement_passed": (
                cross_mesh_passed
            ),
            "cross_mesh_direction_subspace_diagnostic_passed": (
                cross_mesh_direction_diagnostic_passed
            ),
            "cross_mesh_direction_subspace_is_binding": False,
            "numerical_contract_passed": numerical_contract,
            "identifiability_contract_passed": identifiability,
            "proven_not_identifiable": proven_not_identifiable,
            "identifiability_inconclusive": inconclusive,
            "online_cost_gate_evaluated": False,
            "ordinary_reduced_evaluation_calls_full_n128": False,
            "online_cost_passed": False,
            "complete_contract_passed": False,
        }
    identifiable_levels = [
        name
        for name, row in level_decisions.items()
        if row["identifiability_contract_passed"]
    ]
    proven_failed_levels = [
        name
        for name, row in level_decisions.items()
        if row["proven_not_identifiable"]
    ]
    inconclusive_levels = [
        name
        for name, row in level_decisions.items()
        if row["identifiability_inconclusive"]
    ]
    all_finite_time_null_results_binding = bool(
        all(
            row["finite_time_null_results_binding"]
            for row in level_decisions.values()
        )
    )
    if identifiable_levels:
        decision = (
            "wp10c8i_storage_consistent_candidate_requires_"
            "online_closure_audit"
        )
        next_authorization = (
            "implement_cheap_operator_only_closure_for_"
            "identifiable_candidate"
        )
    elif len(proven_failed_levels) == len(level_names):
        decision = (
            "wp10c8i_storage_consistent_moment_ladder_"
            "proven_not_identifiable"
        )
        next_authorization = (
            "retain_full_dae_and_reassess_dynamic_internal_coordinates"
        )
    else:
        decision = "wp10c8i_moment_sufficiency_inconclusive"
        next_authorization = (
            "repair_generator_tangent_contract_then_repeat_wp10c8i"
        )

    array_payload = {}
    for n_cells in RESOLUTIONS:
        for label, anchor_traces in traces[n_cells].items():
            for name, values in anchor_traces.items():
                array_payload[f"n{n_cells}_{label}_{name}"] = values
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **array_payload)
    payload = {
        "work_package": "WP10c8i",
        "base_commit": BASE_COMMIT,
        "decision": decision,
        "next_authorization": next_authorization,
        "scope": {
            "description": (
                "Incremental five-shell moment sufficiency and complete "
                "vector storage-one-form audit"
            ),
            "resolutions": RESOLUTIONS,
            "anchors_seconds": {
                label: seconds for label, seconds, _role in ANCHORS
            },
            "construction_anchors": [
                label for label, _seconds, role in ANCHORS
                if role == "construction"
            ],
            "held_out_anchors": [
                label for label, _seconds, role in ANCHORS
                if role == "held_out"
            ],
            "finite_time_horizons_seconds": (
                FINITE_TIME_HORIZONS_SECONDS
            ),
            "response_kinds": RESPONSE_KINDS,
            "five_shell_edges_rg": shell_edges_rg,
            "new_full_dae_trajectory_run": False,
            "new_nonlinear_microburst_run": False,
            "reduced_nonlinear_evolution_constructed": False,
        },
        "authorization": {
            "wp10c8h_decision": authorization["decision"],
            "wp10c8h_evidence_sha256": authorization_sha256,
        },
        "state_provenance": state_provenance,
        "operator_provenance": cache_provenance,
        "input_metric": {
            "definition": (
                "cell_measure_normalized_continuum_L2_norm_of_each_"
                "physical_primitive_perturbation_divided_by_its_declared_"
                "component_amplitude"
            ),
            "primitive_amplitude_policy": PRIMITIVE_AMPLITUDE_POLICY,
            "unit_direction_semantics": (
                "cross-mesh continuum-L2 unit direction; separate "
                "pointwise box bounds are binding for pass/fail"
            ),
        },
        "storage_contract": {
            "semantics": (
                "complete vector one-form; cumulative height work remains "
                "a path ledger and is not an instantaneous coordinate"
            ),
            "component_order": (
                "mass",
                "radial_momentum",
                "angular_momentum",
                "killing_energy",
                "stress_storage",
            ),
            "per_anchor": {
                str(n_cells): {
                    label: caches[n_cells][label][1]["storage_audit"]
                    for label, _seconds, _role in ANCHORS
                }
                for n_cells in RESOLUTIONS
            },
        },
        "generator_stability_contract": {
            "scope": (
                "production directional JVP checks at every anchor; full "
                "separated FD scans at t_0 and held-out t_0p10, on N64 "
                "and N128"
            ),
            "full_scan_anchors": FULL_GENERATOR_STABILITY_ANCHORS,
            "all_anchors_passed": all_generator_stability_passed,
            "per_anchor": {
                str(n_cells): {
                    label: caches[n_cells][label][1][
                        "generator_stability_audit"
                    ]
                    for label, _seconds, _role in ANCHORS
                }
                for n_cells in RESOLUTIONS
            },
        },
        "tangent_differentiability_contract": {
            "semantics": (
                "local tangent construction requires inactive production "
                "face-reconstruction factors, a numerically unique or "
                "jump-suppressed Rusanov controller, and a fixed outer active "
                "set. Finite-time decisions additionally require exact "
                "absence of every consequential declared-margin Rusanov kink. "
                "The generalized-Jacobian Fréchet calculation is diagnostic "
                "only and cannot restore binding status."
            ),
            "all_anchors_passed": all_tangent_differentiability_passed,
            "finite_time_null_results_binding": (
                all_finite_time_null_results_binding
            ),
            "per_anchor": {
                str(n_cells): {
                    label: caches[n_cells][label][1][
                        "tangent_differentiability_audit"
                    ]
                    for label, _seconds, _role in ANCHORS
                }
                for n_cells in RESOLUTIONS
            },
        },
        "moment_levels": {
            str(n_cells): results[n_cells]
            for n_cells in RESOLUTIONS
        },
        "cross_mesh_agreement": cross_mesh,
        "level_decisions": level_decisions,
        "identifiable_levels": identifiable_levels,
        "proven_not_identifiable_levels": proven_failed_levels,
        "inconclusive_levels": inconclusive_levels,
        "gates": {
            "maximum_screening_gate_fraction": (
                SCREENING_MAXIMUM_GATE_FRACTION
            ),
            "maximum_pre_microburst_gate_fraction": (
                PRE_MICROBURST_MAXIMUM_GATE_FRACTION
            ),
            "maximum_online_cost_fraction": (
                MAXIMUM_ONLINE_COST_FRACTION
            ),
            "interface_flux_relative_gate": (
                INTERFACE_FLUX_RELATIVE_GATE
            ),
            "coordinate_rate_window_seconds": (
                COORDINATE_RATE_WINDOW_SECONDS
            ),
            "coordinate_rate_window_gate": (
                COORDINATE_RATE_WINDOW_GATE
            ),
            "maximum_storage_action_relative_defect": (
                MAXIMUM_STORAGE_ACTION_RELATIVE_DEFECT
            ),
            "maximum_storage_component_reconstruction_defect": (
                MAXIMUM_STORAGE_COMPONENT_RECONSTRUCTION_DEFECT
            ),
            "maximum_direct_to_historical_storage_change": (
                MAXIMUM_DIRECT_TO_HISTORICAL_STORAGE_CHANGE
            ),
            "maximum_direct_storage_condition_estimate": (
                MAXIMUM_DIRECT_STORAGE_CONDITION_ESTIMATE
            ),
            "maximum_generator_factorization_defect": (
                MAXIMUM_GENERATOR_FACTORIZATION_DEFECT
            ),
            "maximum_generator_stability_relative_defect": (
                MAXIMUM_GENERATOR_STABILITY_RELATIVE_DEFECT
            ),
            "maximum_coordinate_storage_row_relative_defect": (
                MAXIMUM_COORDINATE_STORAGE_ROW_RELATIVE_DEFECT
            ),
            "maximum_named_direction_correction_fraction": (
                MAXIMUM_NAMED_DIRECTION_CORRECTION_FRACTION
            ),
            "admissibility_factor_inactive_absolute_tolerance": (
                ADMISSIBILITY_FACTOR_INACTIVE_ABSOLUTE_TOLERANCE
            ),
            "maximum_constraint_condition_estimate": (
                MAXIMUM_CONSTRAINT_CONDITION_ESTIMATE
            ),
            "maximum_cross_mesh_gain_relative_difference": (
                MAXIMUM_CROSS_MESH_GAIN_RELATIVE_DIFFERENCE
            ),
            "minimum_cross_mesh_leading_direction_cosine": (
                MINIMUM_CROSS_MESH_LEADING_DIRECTION_COSINE
            ),
            "maximum_cross_mesh_admissible_subspace_angle_degrees": (
                CROSS_MESH_SUBSPACE_ANGLE_GATE_DEGREES
            ),
            "nonlinear_microbursts_authorized": False,
            "nonlinear_macrosteps_authorized": False,
        },
        "online_cost_contract": {
            "ordinary_reduced_evaluation_implemented": False,
            "ordinary_reduced_evaluation_calls_full_n128": False,
            "full_n128_residual_or_jacobian_allowed_online": False,
            "offline_full_operator_calls_used_for_this_audit": True,
            "binding_maximum_fraction_of_full_operator": (
                MAXIMUM_ONLINE_COST_FRACTION
            ),
            "status": (
                "not_evaluated_until_a_candidate_passes_identifiability"
            ),
        },
        "interpretation": {
            "unresolved_state": (
                "The constraint-null complement is unresolved, not assumed "
                "fast or Markovian."
            ),
            "finite_time": (
                "For the selected frozen linearized generator, endpoint "
                "uses O exp(L h); increment uses "
                "O(exp(L h)-I), with the evolving-anchor generator "
                "including the DM[delta x, xdot] storage term. These are "
                "not exact nonlinear finite-time responses."
            ),
            "binding_output_norm": (
                "Each row uses a continuum-L2 null gain plus declared "
                "pointwise amplitude bounds. A lower bound above the gate "
                "proves failure; only an upper bound below the gate proves "
                "a pass. Intermediate cases are inconclusive. The stacked "
                "spectral norm is nonbinding."
            ),
            "three_way_decision": (
                "A level is identifiable only when every admissible upper "
                "bound and every numerical contract pass; it is proven not "
                "identifiable only when an admissible lower bound fails "
                "under those numerical contracts; all other cases remain "
                "inconclusive."
            ),
            "coordinate_rates": (
                "Each candidate level includes 0.025 s times C L as an "
                "explicit output, where C uses natural coordinate scales. "
                "A unit output gate makes 0.25 mean a 25 percent natural-"
                "scale coordinate change over the declared window."
            ),
            "stop_rule": (
                "No nonlinear lift, healing burst, or macrostep is "
                "authorized until gain is below 0.10 on construction and "
                "held-out anchors and a no-full-N128 online cost audit "
                "passes."
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
                "work_package": "WP10c8i",
                "decision": decision,
                "identifiable_levels": identifiable_levels,
                "next_authorization": next_authorization,
                "output": _relative(output_path),
                "arrays": _relative(arrays_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
