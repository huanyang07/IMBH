"""Certify the WP10c8i evolving tangent before repeating its moment audit.

WP10c8j is deliberately a numerical-certification package.  It keeps the
WP10c8i meshes, anchors, moment ladder, horizons, inputs, and scientific gates
locked.  The script first separates the stationary, storage-matrix, and
storage-rate derivative finite-difference scans, then requires a binding
finite-neighborhood certificate for every consequential Rusanov branch.  It
only authorizes a separate unchanged WP10c8i repetition after every selected
numerical contract passes.  This certification runner never launches that
scientific campaign itself.

No truth trajectory, reduced lift, healing burst, closure, or nonlinear
reduced evolution is constructed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import run_causal_moment_sufficiency_audit_wp10c8i as wp10c8i
import run_causal_shell_closure_preflight_wp10c8h as wp10c8h
from imri_qpe.layer3_minidisk_1d import (
    causal_five_field_assemble_evolving_tangent,
    causal_five_field_reduced_stationary_jacobian,
    causal_five_field_reduced_storage_matrices,
    causal_five_field_reduced_storage_rate_derivatives,
    causal_five_field_scaled_primitive_vector_field,
    unpack_causal_five_field_state,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_rusanov_certification import (
    RusanovCandidateCoverage,
    certify_cached_rusanov_finite_neighborhood,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "6233914eab6d9b719b90602243e59c7f09de525d"
WORK_PACKAGE = "WP10c8j"
CACHE_SCHEMA_VERSION = 3

# These are literals on purpose.  Importing a changed WP10c8i constant and
# silently treating it as the locked contract would defeat the audit.
LOCKED_RESOLUTIONS = (64, 128)
LOCKED_ANCHORS = (
    ("t_0", 0.0, "construction"),
    ("t_0p025", 2.5e-2, "construction"),
    ("t_0p05", 5.0e-2, "construction"),
    ("t_0p075", 7.5e-2, "held_out"),
    ("t_0p10", 1.0e-1, "held_out"),
    ("t_0p125", 1.25e-1, "construction"),
)
LOCKED_FINITE_TIME_HORIZONS_SECONDS = (0.0, 1.0e-2, 2.5e-2)
LOCKED_RESPONSE_KINDS = ("endpoint", "increment")
FULL_SCAN_ANCHORS = ("t_0", "t_0p10")

INNER_DIFFERENCE_STEPS = (1.0e-6, 2.0e-6, 4.0e-6)
OUTER_DIFFERENCE_STEPS = (7.0e-6, 8.0e-6, 9.0e-6)
VERTICAL_ACTION_DIFFERENCE_STEPS = (3.2e-3, 6.4e-3, 1.28e-2)
LEGACY_OUTER_DIFFERENCE_STEPS = (1.0e-6, 2.0e-6, 4.0e-6)
LEGACY_VERTICAL_ACTION_DIFFERENCE_STEPS = (5.0e-5, 1.0e-4, 2.0e-4)
BASE_INNER_DIFFERENCE_STEP = 2.0e-6
BASE_OUTER_DIFFERENCE_STEP = 8.0e-6
BASE_VERTICAL_ACTION_DIFFERENCE_STEP = 6.4e-3
LEGACY_BASE_OUTER_DIFFERENCE_STEP = 2.0e-6
LEGACY_BASE_VERTICAL_ACTION_DIFFERENCE_STEP = 1.0e-4
# A pilot at N64, t=0.10 s showed that 3e-4 lies below the repeatable
# production-vector-field secant floor in two low-activity outer directions:
# 5e-4, 1e-3, and 3e-3 form the first resolved three-point plateau.  The
# rejected 3e-4 pilot remains recorded in the WP10c8j report rather than being
# silently omitted from the evidence.
INDEPENDENT_VECTOR_FIELD_JVP_STEPS = (5.0e-4, 1.0e-3, 3.0e-3)
BASE_INDEPENDENT_VECTOR_FIELD_JVP_STEP = 1.0e-3
STORAGE_QUADRATURE_ORDER = 4
STORAGE_DIRECTIONAL_STEP = 1.0e-3

MAXIMUM_GENERATOR_RELATIVE_DEFECT = 5.0e-3
MAXIMUM_INDEPENDENT_VECTOR_FIELD_JVP_RELATIVE_DEFECT = 1.0e-2
MAXIMUM_FORWARD_BACKWARD_JVP_RELATIVE_DEFECT = 2.0e-2
MAXIMUM_JVP_ADDITIVITY_RELATIVE_DEFECT = 2.0e-2
MAXIMUM_GENERATOR_FACTORIZATION_DEFECT = 1.0e-8
MAXIMUM_STORAGE_ACTION_RELATIVE_DEFECT = 5.0e-5
MAXIMUM_STORAGE_COMPONENT_RECONSTRUCTION_DEFECT = 1.0e-10
MAXIMUM_RUSANOV_GENERATOR_KINK_RELATIVE_DIAMETER = 5.0e-3
MAXIMUM_RUSANOV_FINITE_TIME_GATE_FRACTION = 1.0e-2
JVP_ACTIVITY_FLOOR_PER_S = 1.0e-10

CACHE_DIRECTORY = ROOT / "outputs/checkpoints/causal_five_field_wp10c8j"
OPERATOR_SOURCE_DIRECTORY = CACHE_DIRECTORY / "operator_sources"
DEFAULT_OUTPUT = (
    ROOT / "outputs/tables/causal_tangent_certification_wp10c8j.json"
)
DEFAULT_ARRAYS = (
    ROOT / "outputs/tables/causal_tangent_certification_wp10c8j_arrays.npz"
)
WP10C8I_OUTPUT = (
    ROOT / "outputs/tables/causal_moment_sufficiency_audit_wp10c8i.json"
)
WP10C8I_BASE_COMMIT = "3e204d173a71f5c2ad02228e7c673601a7316e11"
WP10C8I_REQUIRED_DECISION = "wp10c8i_moment_sufficiency_inconclusive"
WP10C8I_REQUIRED_NEXT_AUTHORIZATION = (
    "repair_generator_tangent_contract_then_repeat_wp10c8i"
)

PRIMITIVE_COMPONENTS = (
    "log_surface_density",
    "radial_three_velocity_over_c",
    "azimuthal_three_velocity_over_c",
    "log_temperature",
    "specific_causal_stress",
)
CONSERVATION_COMPONENTS = (
    "rest_mass",
    "radial_momentum",
    "angular_momentum",
    "killing_energy",
    "stress_storage",
)
LOCKED_PHYSICAL_DIRECTION_NAMES = frozenset(
    {
        "azimuthal_redistribution_6_to_60rg",
        "density_redistribution_20_to_200rg",
        "density_redistribution_6_to_60rg",
        "radial_redistribution_6_to_60rg",
        "smooth_mixed",
        "source_redistribution_200_to_280rg",
        "stress_target_adjustment",
        "thermal_redistribution_60_to_200rg",
        "thermal_redistribution_6_to_60rg",
    }
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate the locked WP10c8i contract and selected states.",
    )
    parser.add_argument(
        "--certification-only",
        action="store_true",
        help=(
            "Permit a partial diagnostic selection; WP10c8j always stops "
            "after tangent and Rusanov certification."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild selected WP10c8j certification caches.",
    )
    parser.add_argument(
        "--rebuild-operator-source",
        action="store_true",
        help=(
            "Explicitly build a WP10c8j-owned operator-source artifact when "
            "the immutable WP10c8i cache hash is unavailable.  The canonical "
            "WP10c8i path is never overwritten."
        ),
    )
    parser.add_argument(
        "--resolution",
        action="append",
        type=int,
        choices=LOCKED_RESOLUTIONS,
        help="Select a certification resolution; repeat as needed.",
    )
    parser.add_argument(
        "--anchor",
        action="append",
        choices=tuple(row[0] for row in LOCKED_ANCHORS),
        help="Select a certification anchor; repeat as needed.",
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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _locked_contract() -> dict:
    """Return the literal contract after checking the imported WP10c8i."""

    imported = {
        "resolutions": tuple(wp10c8i.RESOLUTIONS),
        "anchors": tuple(wp10c8i.ANCHORS),
        "finite_time_horizons_seconds": tuple(
            wp10c8i.FINITE_TIME_HORIZONS_SECONDS
        ),
        "response_kinds": tuple(wp10c8i.RESPONSE_KINDS),
        "maximum_screening_gate_fraction": float(
            wp10c8i.SCREENING_MAXIMUM_GATE_FRACTION
        ),
        "maximum_pre_microburst_gate_fraction": float(
            wp10c8i.PRE_MICROBURST_MAXIMUM_GATE_FRACTION
        ),
        "maximum_cross_mesh_gain_relative_difference": float(
            wp10c8i.MAXIMUM_CROSS_MESH_GAIN_RELATIVE_DIFFERENCE
        ),
        "minimum_cross_mesh_leading_direction_cosine": float(
            wp10c8i.MINIMUM_CROSS_MESH_LEADING_DIRECTION_COSINE
        ),
        "maximum_cross_mesh_subspace_angle_degrees": float(
            wp10c8i.CROSS_MESH_SUBSPACE_ANGLE_GATE_DEGREES
        ),
    }
    expected = {
        "resolutions": LOCKED_RESOLUTIONS,
        "anchors": LOCKED_ANCHORS,
        "finite_time_horizons_seconds": (
            LOCKED_FINITE_TIME_HORIZONS_SECONDS
        ),
        "response_kinds": LOCKED_RESPONSE_KINDS,
        "maximum_screening_gate_fraction": 0.25,
        "maximum_pre_microburst_gate_fraction": 0.10,
        "maximum_cross_mesh_gain_relative_difference": 0.25,
        "minimum_cross_mesh_leading_direction_cosine": 0.50,
        "maximum_cross_mesh_subspace_angle_degrees": 45.0,
    }
    if imported != expected:
        raise RuntimeError(
            "WP10c8i contract changed; WP10c8j requires the locked campaign"
        )
    return expected


def _validate_authorization() -> tuple[dict, str]:
    """Require the committed WP10c8i result and its complete evidence chain."""

    # Preserve WP10c8i's own WP10c8h provenance check first.
    wp10c8i._validate_authorization()
    if not WP10C8I_OUTPUT.exists():
        raise RuntimeError("WP10c8j requires canonical WP10c8i evidence")
    evidence = json.loads(WP10C8I_OUTPUT.read_text(encoding="utf-8"))
    artifacts = evidence.get("artifacts", {})
    arrays = ROOT / str(artifacts.get("arrays_path", ""))
    scope = evidence.get("scope", {})
    if not (
        evidence.get("work_package") == "WP10c8i"
        and evidence.get("base_commit") == WP10C8I_BASE_COMMIT
        and evidence.get("decision") == WP10C8I_REQUIRED_DECISION
        and evidence.get("next_authorization")
        == WP10C8I_REQUIRED_NEXT_AUTHORIZATION
        and tuple(scope.get("resolutions", ())) == LOCKED_RESOLUTIONS
        and tuple(scope.get("finite_time_horizons_seconds", ()))
        == LOCKED_FINITE_TIME_HORIZONS_SECONDS
        and tuple(scope.get("response_kinds", ()))
        == LOCKED_RESPONSE_KINDS
        and scope.get("new_full_dae_trajectory_run") is False
        and scope.get("new_nonlinear_microburst_run") is False
        and scope.get("reduced_nonlinear_evolution_constructed") is False
        and arrays.exists()
        and _sha256(arrays) == artifacts.get("arrays_sha256")
    ):
        raise RuntimeError(
            "WP10c8i evidence does not authorize the locked tangent repair"
        )
    return evidence, _sha256(WP10C8I_OUTPUT)


def _matrix_from_result(result: dict, *names: str) -> np.ndarray:
    for name in names:
        if name in result:
            values = np.asarray(result[name], dtype=float)
            if values.ndim != 2 or np.any(~np.isfinite(values)):
                raise ValueError(f"matrix {name} is invalid")
            return values
    raise KeyError(f"none of the matrix keys are present: {names}")


def _assemble_generator(
    mass: np.ndarray,
    stationary: np.ndarray,
    storage_rate_derivative: np.ndarray,
) -> np.ndarray:
    result = causal_five_field_assemble_evolving_tangent(
        mass,
        stationary,
        storage_rate_derivative,
    )
    return _matrix_from_result(
        result,
        "evolving_scaled_generator_per_s",
        "generator",
    )


def _normalized_directions(
    initial: dict,
    vector: np.ndarray,
    primitive_scales: np.ndarray,
) -> dict[str, np.ndarray]:
    directions = wp10c8h._redistribution_directions(
        initial,
        vector,
        primitive_scales,
    )
    directions = dict(directions)
    directions["smooth_mixed"] = wp10c8i._smooth_jvp_direction(initial)
    normalized = {}
    for name, direction in directions.items():
        values = np.asarray(direction, dtype=float).ravel()
        norm = float(np.linalg.norm(values))
        if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
            raise ValueError(f"tangent direction {name} is invalid")
        normalized[name] = values / norm
    return normalized


def _component_location(index: int, *, row_kind: str) -> dict:
    names = (
        CONSERVATION_COMPONENTS
        if row_kind == "conservation"
        else PRIMITIVE_COMPONENTS
    )
    return {
        "flat_index": int(index),
        "cell_index": int(index // 5),
        "component": names[index % 5],
    }


def _matrix_comparison(
    candidate: np.ndarray,
    reference: np.ndarray,
    directions: dict[str, np.ndarray],
    *,
    row_kind: str,
    relative_gate: float = MAXIMUM_GENERATOR_RELATIVE_DEFECT,
) -> dict:
    """Compare matrices and their deterministic physical JVPs."""

    left = np.asarray(candidate, dtype=float)
    right = np.asarray(reference, dtype=float)
    if (
        left.shape != right.shape
        or left.ndim != 2
        or left.shape[0] != left.shape[1]
        or np.any(~np.isfinite(left))
        or np.any(~np.isfinite(right))
    ):
        raise ValueError("matrix comparison inputs are incompatible")
    delta = left - right
    tiny = np.finfo(float).tiny
    relative_frobenius = float(
        np.linalg.norm(delta)
        / max(float(np.linalg.norm(left)), float(np.linalg.norm(right)), tiny)
    )
    maximum_absolute = float(np.max(np.abs(delta)))
    matrix_index = np.unravel_index(int(np.argmax(np.abs(delta))), delta.shape)
    jvp_rows = {}
    maximum_relative_jvp = 0.0
    maximum_activity_scaled_jvp = 0.0
    controlling_name = None
    controlling_location = None
    for name, direction in directions.items():
        values = np.asarray(direction, dtype=float).ravel()
        if values.shape != (left.shape[1],):
            raise ValueError(f"direction {name} has the wrong shape")
        candidate_jvp = left @ values
        reference_jvp = right @ values
        difference = candidate_jvp - reference_jvp
        candidate_norm = float(np.linalg.norm(candidate_jvp))
        reference_norm = float(np.linalg.norm(reference_jvp))
        difference_norm = float(np.linalg.norm(difference))
        relative = float(
            difference_norm / max(candidate_norm, reference_norm, tiny)
        )
        activity_scaled = float(
            difference_norm
            / max(
                candidate_norm,
                reference_norm,
                JVP_ACTIVITY_FLOOR_PER_S,
            )
        )
        row_index = int(np.argmax(np.abs(difference)))
        jvp_rows[name] = {
            "relative_l2_defect": relative,
            "activity_scaled_l2_defect": activity_scaled,
            "absolute_l2_defect": difference_norm,
            "maximum_absolute_component_defect": float(
                np.max(np.abs(difference))
            ),
            "controlling_row": _component_location(
                row_index,
                row_kind=row_kind,
            ),
        }
        if relative > maximum_relative_jvp:
            maximum_relative_jvp = relative
            controlling_name = name
            controlling_location = jvp_rows[name]["controlling_row"]
        maximum_activity_scaled_jvp = max(
            maximum_activity_scaled_jvp,
            activity_scaled,
        )
    frobenius_passed = bool(relative_frobenius <= relative_gate)
    deterministic_jvp_passed = bool(
        maximum_relative_jvp <= relative_gate
    )
    passed = bool(frobenius_passed and deterministic_jvp_passed)
    return {
        "relative_frobenius_defect": relative_frobenius,
        "maximum_absolute_matrix_defect": maximum_absolute,
        "controlling_matrix_row": _component_location(
            int(matrix_index[0]),
            row_kind=row_kind,
        ),
        "controlling_matrix_column": _component_location(
            int(matrix_index[1]),
            row_kind="primitive",
        ),
        "deterministic_physical_jvps": jvp_rows,
        "maximum_deterministic_physical_jvp_relative_defect": (
            maximum_relative_jvp
        ),
        "maximum_activity_scaled_jvp_relative_defect": (
            maximum_activity_scaled_jvp
        ),
        "controlling_jvp_direction": controlling_name,
        "controlling_jvp_row": controlling_location,
        "maximum_relative_defect": relative_gate,
        "frobenius_passed": frobenius_passed,
        "deterministic_physical_jvps_passed": (
            deterministic_jvp_passed
        ),
        "passed": passed,
    }


def _scan_variants(
    variants: dict[str, np.ndarray],
    *,
    base_key: str,
    directions: dict[str, np.ndarray],
    row_kind: str,
) -> dict:
    if base_key not in variants:
        raise KeyError("scan base key is absent")
    reference = variants[base_key]
    comparisons = {
        f"{key}_versus_{base_key}": _matrix_comparison(
            values,
            reference,
            directions,
            row_kind=row_kind,
        )
        for key, values in variants.items()
        if key != base_key
    }
    return {
        "base_key": base_key,
        "variant_keys": tuple(variants),
        "comparisons": comparisons,
        "passed": bool(
            comparisons and all(row["passed"] for row in comparisons.values())
        ),
    }


def _result_components(result: dict) -> dict[str, np.ndarray]:
    """Normalize public storage helper schemas for runner provenance."""

    return {
        "total": _matrix_from_result(
            result,
            "total",
            "descriptor_reduced_scaled_matrix",
            "storage_rate_derivative_scaled_matrix",
        ),
        "conserved": _matrix_from_result(
            result,
            "conserved",
            "conserved_descriptor_reduced_scaled_matrix",
            "conserved_storage_rate_derivative_scaled_matrix",
        ),
        "vertical": _matrix_from_result(
            result,
            "vertical",
            "vertical_descriptor_reduced_scaled_matrix",
            "vertical_storage_rate_derivative_scaled_matrix",
        ),
    }


def _direct_action_storage_rate_result(
    initial: dict,
    vector: np.ndarray,
    operator_arrays: dict[str, np.ndarray],
    *,
    outer_step: float = BASE_OUTER_DIFFERENCE_STEP,
    action_step: float = BASE_VERTICAL_ACTION_DIFFERENCE_STEP,
    physical_rate_per_s: np.ndarray | None = None,
) -> dict:
    """Build repaired ``DM[., p_dot]`` from the complete storage action."""

    state = unpack_causal_five_field_state(
        vector,
        initial["state"].n_cells,
    )
    primitive_scales = np.asarray(
        operator_arrays["primitive_column_scales"],
        dtype=float,
    )
    if physical_rate_per_s is None:
        scaled_rate = np.asarray(
            operator_arrays["scaled_primitive_rate"],
            dtype=float,
        )
        physical_rate = scaled_rate * primitive_scales
        rate_source = "authorized_operator_cache"
    else:
        physical_rate = np.asarray(physical_rate_per_s, dtype=float).ravel()
        if physical_rate.shape != primitive_scales.shape:
            raise ValueError("fresh physical primitive rate has wrong shape")
        rate_source = "fresh_production_vector_field"
    return causal_five_field_reduced_storage_rate_derivatives(
        initial["context"],
        np.asarray(state.primitives, dtype=float).ravel(),
        physical_rate,
        primitive_column_scales=primitive_scales,
        conservation_row_scales=np.asarray(
            operator_arrays["conservation_row_scales"],
            dtype=float,
        ),
        storage_matrix_difference_step=BASE_INNER_DIFFERENCE_STEP,
        storage_rate_derivative_step=outer_step,
        storage_difference_step=action_step,
        storage_quadrature_order=STORAGE_QUADRATURE_ORDER,
        storage_directional_step=STORAGE_DIRECTIONAL_STEP,
        backend="direct_action",
    ) | {"wp10c8j_rate_source": rate_source}


def _fresh_base_vector_field(
    initial: dict,
    vector: np.ndarray,
    operator_arrays: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict, tuple[dict, np.ndarray, dict], dict]:
    """Evaluate the exact base vector field used by all WP10c8j secants."""

    state = unpack_causal_five_field_state(
        vector,
        initial["state"].n_cells,
    )
    primitives = np.asarray(state.primitives, dtype=float).ravel()
    primitive_scales = np.asarray(
        operator_arrays["primitive_column_scales"],
        dtype=float,
    )
    conservation_scales = np.asarray(
        operator_arrays["conservation_row_scales"],
        dtype=float,
    )
    result = causal_five_field_scaled_primitive_vector_field(
        initial["context"],
        primitives,
        primitive_column_scales=primitive_scales,
        conservation_row_scales=conservation_scales,
        finite_difference_step=BASE_INNER_DIFFERENCE_STEP,
        storage_quadrature_order=STORAGE_QUADRATURE_ORDER,
        storage_directional_step=STORAGE_DIRECTIONAL_STEP,
    )
    fresh_scaled_rate = np.asarray(
        result["scaled_primitive_rate_per_s"],
        dtype=float,
    ).ravel()
    cached_scaled_rate = np.asarray(
        operator_arrays["scaled_primitive_rate"],
        dtype=float,
    ).ravel()
    rate_difference = fresh_scaled_rate - cached_scaled_rate
    rate_scale = max(
        float(np.linalg.norm(fresh_scaled_rate)),
        float(np.linalg.norm(cached_scaled_rate)),
        np.finfo(float).tiny,
    )
    relative_rate_defect = float(
        np.linalg.norm(rate_difference) / rate_scale
    )
    rate_contract = {
        "relative_l2_defect": relative_rate_defect,
        "maximum_absolute_defect_per_s": float(
            np.max(np.abs(rate_difference))
        ),
        "maximum_relative_defect": 1.0e-12,
        "passed": bool(relative_rate_defect <= 1.0e-12),
    }
    return (
        primitives,
        result,
        wp10c8i._vector_field_branch_state(initial, primitives),
        rate_contract,
    )


def _cached_independent_jvp_row(
    generator: np.ndarray,
    operator_arrays: dict[str, np.ndarray],
    metadata_row: dict,
    *,
    prefix: str,
) -> tuple[dict[str, np.ndarray], dict]:
    direction = np.asarray(
        operator_arrays[f"{prefix}_independent_vector_field_jvp_direction"],
        dtype=float,
    )
    predicted = np.asarray(generator, dtype=float) @ direction
    direct = np.asarray(
        operator_arrays[f"{prefix}_independent_vector_field_jvp_direct"],
        dtype=float,
    )
    forward = np.asarray(
        operator_arrays[f"{prefix}_independent_vector_field_jvp_forward"],
        dtype=float,
    )
    backward = np.asarray(
        operator_arrays[f"{prefix}_independent_vector_field_jvp_backward"],
        dtype=float,
    )
    central_defect = wp10c8i._jvp_defect(
        predicted,
        direct,
        relative_tolerance=(
            MAXIMUM_INDEPENDENT_VECTOR_FIELD_JVP_RELATIVE_DEFECT
        ),
    )
    forward_defect = wp10c8i._jvp_defect(
        predicted,
        forward,
        relative_tolerance=MAXIMUM_FORWARD_BACKWARD_JVP_RELATIVE_DEFECT,
    )
    backward_defect = wp10c8i._jvp_defect(
        predicted,
        backward,
        relative_tolerance=MAXIMUM_FORWARD_BACKWARD_JVP_RELATIVE_DEFECT,
    )
    smooth_contract = bool(
        metadata_row.get("plus_minus_reconstruction_differentiable", False)
        and metadata_row.get(
            "plus_minus_outer_active_set_unchanged",
            False,
        )
        and metadata_row.get("rusanov_branch_differentiable", False)
    )
    return {
        "direction": direction,
        "predicted": predicted,
        "direct": direct,
        "forward": forward,
        "backward": backward,
    }, {
        "source_direction_name": metadata_row.get("direction_name"),
        "centered_difference_step": metadata_row.get(
            "centered_difference_step"
        ),
        "smooth_secant_contract_passed": smooth_contract,
        "central_jvp_defect": central_defect,
        "forward_jvp_defect": forward_defect,
        "backward_jvp_defect": backward_defect,
        "passed": bool(
            smooth_contract
            and central_defect["passed"]
            and forward_defect["passed"]
            and backward_defect["passed"]
        ),
    }


def _repaired_generator_jvp_contract(
    generator: np.ndarray,
    operator_arrays: dict[str, np.ndarray],
    operator_metadata: dict,
) -> tuple[dict[str, np.ndarray], dict]:
    """Re-score cached independent secants against the repaired tangent.

    Switching-normal directions are excluded from the smooth decision and
    remain the responsibility of the separate finite-branch certificate.
    """

    production = operator_metadata["generator_stability_audit"][
        "production_vector_field_jvp"
    ]
    arrays: dict[str, np.ndarray] = {}
    directions = {}
    for name in production["direction_names"]:
        if str(name).startswith("rusanov_switching_normal_"):
            continue
        row_arrays, row = _cached_independent_jvp_row(
            generator,
            operator_arrays,
            production["directions"][name],
            prefix=f"production_{name}",
        )
        directions[name] = row
        arrays.update(
            {
                f"repaired_{name}_{array_name}": values
                for array_name, values in row_arrays.items()
            }
        )

    stability_source = production["independent_secant_step_stability"]
    stability_rows = {}
    if stability_source.get("evaluated", False):
        for key, source_row in stability_source["directions"].items():
            prefix = f"production_smooth_mixed_step_{key}"
            if (
                f"{prefix}_independent_vector_field_jvp_direction"
                not in operator_arrays
            ):
                prefix = "production_smooth_mixed"
            row_arrays, row = _cached_independent_jvp_row(
                generator,
                operator_arrays,
                source_row,
                prefix=prefix,
            )
            stability_rows[key] = row
            arrays.update(
                {
                    f"repaired_smooth_mixed_step_{key}_{array_name}": values
                    for array_name, values in row_arrays.items()
                }
            )
    stability = {
        "evaluated": bool(stability_source.get("evaluated", False)),
        "directions": stability_rows,
        "passed": bool(
            not stability_source.get("evaluated", False)
            or (
                stability_rows
                and all(row["passed"] for row in stability_rows.values())
            )
        ),
    }

    additivity_source = production["additivity"]
    additivity = {
        "evaluated": bool(additivity_source.get("evaluated", False)),
        "cached_independent_secant_additivity": (
            additivity_source.get("defect")
        ),
        "passed": bool(
            not additivity_source.get("evaluated", False)
            or (
                additivity_source.get("defect", {}).get("passed", False)
            )
        ),
    }
    passed = bool(
        directions
        and all(row["passed"] for row in directions.values())
        and stability["passed"]
        and additivity["passed"]
    )
    return arrays, {
        "scope": (
            "cached independent nonlinear-vector-field secants rescored "
            "against the direct-storage-action generator; Rusanov switching "
            "normals are reserved for the finite-branch contract"
        ),
        "directions": directions,
        "excluded_rusanov_switching_direction_count": int(
            sum(
                str(name).startswith("rusanov_switching_normal_")
                for name in production["direction_names"]
            )
        ),
        "independent_secant_step_stability": stability,
        "additivity": additivity,
        "passed": passed,
    }


def _fresh_independent_vector_field_jvp_contract(
    initial: dict,
    vector: np.ndarray,
    generator: np.ndarray,
    operator_arrays: dict[str, np.ndarray],
    directions: dict[str, np.ndarray],
    *,
    base_vector_field: dict | None = None,
    base_branch_state: tuple[dict, np.ndarray, dict] | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Test the repaired generator against fresh nonlinear secants.

    The cached WP10c8i secants do not span every deterministic physical
    direction used by the WP10c8j step scans.  In particular, a nearly
    inactive outer thermal direction can make a matrix-versus-matrix JVP
    ratio ill conditioned without saying whether the selected generator is
    the derivative of the production vector field.  This contract evaluates
    that production vector field afresh for every declared direction.

    A direction that is smooth in reconstruction and the outer active set,
    but crosses a Rusanov max-speed branch, is reserved for the separate
    finite-neighborhood branch certificate.  Other nonsmoothness is a hard
    failure of the smooth contract.
    """

    state = unpack_causal_five_field_state(
        vector,
        initial["state"].n_cells,
    )
    primitives = np.asarray(state.primitives, dtype=float).ravel()
    primitive_scales = np.asarray(
        operator_arrays["primitive_column_scales"],
        dtype=float,
    )
    conservation_scales = np.asarray(
        operator_arrays["conservation_row_scales"],
        dtype=float,
    )
    evolving = {
        "evolving_scaled_generator_per_s": np.asarray(
            generator,
            dtype=float,
        ),
        "primitive_column_scales": primitive_scales,
        "conservation_row_scales": conservation_scales,
    }
    if base_vector_field is None or base_branch_state is None:
        (
            _base_primitives,
            evaluated_vector_field,
            evaluated_branch_state,
            _rate_contract,
        ) = _fresh_base_vector_field(initial, vector, operator_arrays)
        if base_vector_field is None:
            base_vector_field = evaluated_vector_field
        if base_branch_state is None:
            base_branch_state = evaluated_branch_state

    arrays: dict[str, np.ndarray] = {}
    rows = {}
    binding_rows = {}
    rusanov_reserved_rows = {}
    hard_nonsmooth_rows = {}
    base_step_key = f"{BASE_INDEPENDENT_VECTOR_FIELD_JVP_STEP:.0e}"
    for name, direction in directions.items():
        step_rows = {}
        step_arrays = {}
        for step in INDEPENDENT_VECTOR_FIELD_JVP_STEPS:
            row_arrays, row = wp10c8i._independent_vector_field_jvp_audit(
                initial,
                vector,
                evolving,
                np.asarray(direction, dtype=float),
                direction_name=name,
                centered_difference_step=step,
                inner_storage_matrix_difference_step=(
                    BASE_INNER_DIFFERENCE_STEP
                ),
                base_vector_field=base_vector_field,
                base_branch_state=base_branch_state,
            )
            row = {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "base_rusanov_control_labels",
                    "plus_rusanov_control_labels",
                    "minus_rusanov_control_labels",
                }
            }
            key = f"{step:.0e}"
            step_rows[key] = row
            step_arrays[key] = row_arrays
            arrays.update(
                {
                    f"fresh_{name}_step_{key}_{array_name}": values
                    for array_name, values in row_arrays.items()
                }
            )

        base_direct = np.asarray(
            step_arrays[base_step_key][
                "independent_vector_field_jvp_direct"
            ],
            dtype=float,
        )
        secant_stability = {}
        for key, row_arrays in step_arrays.items():
            if key == base_step_key:
                continue
            secant_stability[f"{key}_versus_{base_step_key}"] = (
                wp10c8i._jvp_defect(
                    np.asarray(
                        row_arrays[
                            "independent_vector_field_jvp_direct"
                        ],
                        dtype=float,
                    ),
                    base_direct,
                    relative_tolerance=(
                        MAXIMUM_INDEPENDENT_VECTOR_FIELD_JVP_RELATIVE_DEFECT
                    ),
                )
            )

        reconstruction_smooth = all(
            row.get("plus_minus_reconstruction_differentiable", False)
            for row in step_rows.values()
        )
        active_set_smooth = all(
            row.get("plus_minus_outer_active_set_unchanged", False)
            for row in step_rows.values()
        )
        strict_rusanov_rows = {}
        for key, row in step_rows.items():
            row_arrays = step_arrays[key]
            exact_zero_ties = True
            for sample in ("base", "plus", "minus"):
                margins = np.asarray(
                    row_arrays[
                        "independent_vector_field_jvp_"
                        f"{sample}_rusanov_relative_margins"
                    ],
                    dtype=float,
                )
                jumps = np.asarray(
                    row_arrays[
                        "independent_vector_field_jvp_"
                        f"{sample}_rusanov_scaled_relative_jumps"
                    ],
                    dtype=float,
                )
                tied = margins < float(
                    row["minimum_required_rusanov_control_relative_margin"]
                )
                exact_zero_ties = bool(
                    exact_zero_ties
                    and np.all(jumps[tied] == 0.0)
                )
            strict_rusanov_rows[key] = {
                "controls_unchanged": bool(
                    row.get("rusanov_controls_unchanged", False)
                ),
                "every_under_margin_jump_exactly_zero": exact_zero_ties,
                "passed": bool(
                    row.get("rusanov_controls_unchanged", False)
                    and exact_zero_ties
                ),
            }
        rusanov_smooth = all(
            row["passed"] for row in strict_rusanov_rows.values()
        )
        numerical_jvp_passed = all(
            row["central_jvp_defect"]["passed"]
            and row["forward_jvp_defect"]["passed"]
            and row["backward_jvp_defect"]["passed"]
            for row in step_rows.values()
        )
        secant_stability_passed = bool(
            secant_stability
            and all(row["passed"] for row in secant_stability.values())
        )
        direction_row = {
            "steps": step_rows,
            "strict_rusanov_branch_contract": strict_rusanov_rows,
            "central_secant_step_stability": secant_stability,
            "reconstruction_smooth_at_every_step": (
                reconstruction_smooth
            ),
            "outer_active_set_smooth_at_every_step": active_set_smooth,
            "rusanov_smooth_or_exact_zero_at_every_step": rusanov_smooth,
            "numerical_jvp_passed_at_every_step": numerical_jvp_passed,
            "central_secant_step_stability_passed": (
                secant_stability_passed
            ),
            "passed": bool(
                reconstruction_smooth
                and active_set_smooth
                and rusanov_smooth
                and numerical_jvp_passed
                and secant_stability_passed
            ),
        }
        rows[name] = direction_row
        if not (reconstruction_smooth and active_set_smooth):
            hard_nonsmooth_rows[name] = direction_row
        elif not rusanov_smooth:
            rusanov_reserved_rows[name] = direction_row
        else:
            binding_rows[name] = direction_row

    classified_count = (
        len(binding_rows)
        + len(rusanov_reserved_rows)
        + len(hard_nonsmooth_rows)
    )
    passed = bool(
        directions
        and classified_count == len(directions)
        and binding_rows
        and not hard_nonsmooth_rows
        and all(row.get("passed", False) for row in binding_rows.values())
    )
    return arrays, {
        "scope": (
            "fresh production nonlinear-vector-field secants for every "
            "declared WP10c8j physical direction"
        ),
        "centered_difference_steps": INDEPENDENT_VECTOR_FIELD_JVP_STEPS,
        "directions": rows,
        "binding_smooth_direction_names": tuple(binding_rows),
        "rusanov_reserved_direction_names": tuple(rusanov_reserved_rows),
        "hard_nonsmooth_direction_names": tuple(hard_nonsmooth_rows),
        "declared_direction_count": len(directions),
        "classified_direction_count": classified_count,
        "passed": passed,
    }


def _cached_operator_numerical_contract(operator_metadata: dict) -> dict:
    """Require every cached smooth-operator contract used by WP10c8j.

    The separated scans are an additional diagnostic at the two full-scan
    anchors.  They do not replace the independent storage, differentiability,
    or nonlinear-vector-field secants already stored by WP10c8i.  The old
    nested-DM prediction is not required to pass because WP10c8j rescored
    those independent secants against the repaired generator.
    """

    try:
        storage = operator_metadata["storage_audit"]
        differentiability = operator_metadata[
            "tangent_differentiability_audit"
        ]
        production_jvp = operator_metadata["generator_stability_audit"][
            "production_vector_field_jvp"
        ]
        action_defect = float(
            storage["maximum_relative_storage_action_defect"]
        )
        historical_action_defect = float(
            storage["maximum_relative_historical_storage_action_defect"]
        )
        descriptor_component_defect = float(
            storage[
                "maximum_scaled_descriptor_component_reconstruction_defect"
            ]
        )
        rate_component_defect = float(
            storage[
                "maximum_scaled_storage_rate_component_reconstruction_defect"
            ]
        )
        factorization_defect = float(
            storage["maximum_scaled_generator_factorization_defect"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            "WP10c8i cached numerical contract is incomplete"
        ) from error

    scalar_values = np.asarray(
        (
            action_defect,
            historical_action_defect,
            descriptor_component_defect,
            rate_component_defect,
            factorization_defect,
        ),
        dtype=float,
    )
    if np.any(~np.isfinite(scalar_values)):
        raise RuntimeError("WP10c8i cached numerical contract is nonfinite")

    checks = {
        "storage_audit_passed": bool(storage.get("passed", False)),
        "complete_vector_storage_one_form_present": bool(
            storage.get("complete_vector_one_form_present", False)
        ),
        "storage_action_gate_passed": bool(
            max(action_defect, historical_action_defect)
            <= MAXIMUM_STORAGE_ACTION_RELATIVE_DEFECT
        ),
        "storage_component_reconstruction_gate_passed": bool(
            max(descriptor_component_defect, rate_component_defect)
            <= MAXIMUM_STORAGE_COMPONENT_RECONSTRUCTION_DEFECT
        ),
        "generator_factorization_gate_passed": bool(
            factorization_defect
            <= MAXIMUM_GENERATOR_FACTORIZATION_DEFECT
        ),
        "tangent_differentiability_passed": bool(
            differentiability.get("passed", False)
        ),
        "independent_vector_field_secants_present": bool(
            production_jvp.get("direction_names")
            and production_jvp.get("directions")
        ),
    }
    return {
        "source": "authorized_wp10c8i_operator_cache",
        "checks": checks,
        "maximum_relative_storage_action_defect": action_defect,
        "maximum_relative_historical_storage_action_defect": (
            historical_action_defect
        ),
        "maximum_scaled_descriptor_component_reconstruction_defect": (
            descriptor_component_defect
        ),
        "maximum_scaled_storage_rate_component_reconstruction_defect": (
            rate_component_defect
        ),
        "maximum_scaled_generator_factorization_defect": (
            factorization_defect
        ),
        "production_vector_field_jvp_scope": production_jvp.get("scope"),
        "passed": bool(all(checks.values())),
    }


def _separated_tangent_scan(
    initial: dict,
    vector: np.ndarray,
    operator_arrays: dict[str, np.ndarray],
    operator_metadata: dict,
) -> tuple[dict[str, np.ndarray], dict]:
    """Vary smooth blocks and certify the direct-action DM repair."""

    context = initial["context"]
    state = unpack_causal_five_field_state(
        vector,
        initial["state"].n_cells,
    )
    primitives = np.asarray(state.primitives, dtype=float).ravel()
    primitive_scales = np.asarray(
        operator_arrays["primitive_column_scales"],
        dtype=float,
    )
    conservation_scales = np.asarray(
        operator_arrays["conservation_row_scales"],
        dtype=float,
    )
    directions = _normalized_directions(
        initial,
        vector,
        primitive_scales,
    )
    (
        _base_primitives,
        base_vector_field,
        base_branch_state,
        base_rate_contract,
    ) = _fresh_base_vector_field(initial, vector, operator_arrays)
    fresh_physical_rate = (
        np.asarray(
            base_vector_field["scaled_primitive_rate_per_s"],
            dtype=float,
        ).ravel()
        * primitive_scales
    )

    stationary_results = {
        f"{step:.0e}": causal_five_field_reduced_stationary_jacobian(
            context,
            vector,
            finite_difference_step=step,
        )
        for step in INNER_DIFFERENCE_STEPS
    }
    stationary_variants = {
        key: _matrix_from_result(
            value,
            "stationary_reduced_scaled_jacobian",
        )
        for key, value in stationary_results.items()
    }

    mass_results = {
        f"{step:.0e}": causal_five_field_reduced_storage_matrices(
            context,
            primitives,
            primitive_column_scales=primitive_scales,
            conservation_row_scales=conservation_scales,
            finite_difference_step=step,
            storage_quadrature_order=STORAGE_QUADRATURE_ORDER,
            storage_directional_step=STORAGE_DIRECTIONAL_STEP,
        )
        for step in INNER_DIFFERENCE_STEPS
    }
    mass_components = {
        key: _result_components(value)
        for key, value in mass_results.items()
    }
    mass_component_reconstruction_defects = {
        key: float(value["maximum_scaled_component_reconstruction_defect"])
        for key, value in mass_results.items()
    }
    base_inner_key = f"{BASE_INNER_DIFFERENCE_STEP:.0e}"
    base_mass = mass_components[base_inner_key]["total"]
    base_stationary = stationary_variants[base_inner_key]

    base_outer_key = f"{BASE_OUTER_DIFFERENCE_STEP:.0e}"
    base_action_key = f"{BASE_VERTICAL_ACTION_DIFFERENCE_STEP:.0e}"

    direct_outer_results = {
        f"{step:.0e}": _direct_action_storage_rate_result(
            initial,
            vector,
            operator_arrays,
            outer_step=step,
            action_step=BASE_VERTICAL_ACTION_DIFFERENCE_STEP,
            physical_rate_per_s=fresh_physical_rate,
        )
        for step in OUTER_DIFFERENCE_STEPS
    }
    direct_action_results = {
        f"{step:.0e}": _direct_action_storage_rate_result(
            initial,
            vector,
            operator_arrays,
            outer_step=BASE_OUTER_DIFFERENCE_STEP,
            action_step=step,
            physical_rate_per_s=fresh_physical_rate,
        )
        for step in VERTICAL_ACTION_DIFFERENCE_STEPS
    }
    direct_outer_components = {
        key: _result_components(value)
        for key, value in direct_outer_results.items()
    }
    direct_action_components = {
        key: _result_components(value)
        for key, value in direct_action_results.items()
    }
    base_dm = direct_outer_components[base_outer_key]["total"]

    # WP10c8i already paid for the nine nested DM variants at each declared
    # full-scan anchor.  Recover each K=DM[.,p_dot] without another nested
    # finite difference from M L + J + K = 0.  For the inner scan M and J use
    # their matching steps; outer and vertical scans held both at baseline.
    inner_dynamic = {
        f"{step:.0e}": np.asarray(
            operator_arrays[
                f"generator_inner_storage_fd_dynamic_{step:.0e}"
            ],
            dtype=float,
        )
        for step in INNER_DIFFERENCE_STEPS
    }
    outer_dynamic = {
        f"{step:.0e}": np.asarray(
            operator_arrays[
                f"generator_outer_storage_rate_fd_dynamic_{step:.0e}"
            ],
            dtype=float,
        )
        for step in LEGACY_OUTER_DIFFERENCE_STEPS
    }
    vertical_dynamic = {
        f"{step:.0e}": np.asarray(
            operator_arrays[
                f"generator_vertical_action_fd_dynamic_{step:.0e}"
            ],
            dtype=float,
        )
        for step in LEGACY_VERTICAL_ACTION_DIFFERENCE_STEPS
    }
    legacy_dm_inner = {
        key: -mass_components[key]["total"] @ dynamic
        - stationary_variants[key]
        for key, dynamic in inner_dynamic.items()
    }
    legacy_dm_outer = {
        key: -base_mass @ dynamic - base_stationary
        for key, dynamic in outer_dynamic.items()
    }
    legacy_dm_vertical = {
        key: -base_mass @ dynamic - base_stationary
        for key, dynamic in vertical_dynamic.items()
    }

    block_variants = {
        "stationary_jacobian": stationary_variants,
        "total_storage_matrix": {
            key: value["total"] for key, value in mass_components.items()
        },
        "conserved_storage_matrix": {
            key: value["conserved"] for key, value in mass_components.items()
        },
        "vertical_storage_matrix": {
            key: value["vertical"] for key, value in mass_components.items()
        },
        "direct_total_dm_outer": {
            key: value["total"]
            for key, value in direct_outer_components.items()
        },
        "direct_conserved_dm_outer": {
            key: value["conserved"]
            for key, value in direct_outer_components.items()
        },
        "direct_vertical_dm_outer": {
            key: value["vertical"]
            for key, value in direct_outer_components.items()
        },
        "direct_total_dm_action": {
            key: value["total"]
            for key, value in direct_action_components.items()
        },
        "direct_conserved_dm_action": {
            key: value["conserved"]
            for key, value in direct_action_components.items()
        },
        "direct_vertical_dm_action": {
            key: value["vertical"]
            for key, value in direct_action_components.items()
        },
    }
    base_keys = {
        "stationary_jacobian": base_inner_key,
        "total_storage_matrix": base_inner_key,
        "conserved_storage_matrix": base_inner_key,
        "vertical_storage_matrix": base_inner_key,
        "direct_total_dm_outer": base_outer_key,
        "direct_conserved_dm_outer": base_outer_key,
        "direct_vertical_dm_outer": base_outer_key,
        "direct_total_dm_action": base_action_key,
        "direct_conserved_dm_action": base_action_key,
        "direct_vertical_dm_action": base_action_key,
    }
    row_kinds = {
        name: "conservation" for name in block_variants
    }
    block_scans = {
        name: _scan_variants(
            variants,
            base_key=base_keys[name],
            directions=directions,
            row_kind=row_kinds[name],
        )
        for name, variants in block_variants.items()
    }

    generator_variants = {
        "stationary_jacobian": {
            key: _assemble_generator(base_mass, value, base_dm)
            for key, value in stationary_variants.items()
        },
        "storage_matrix": {
            key: _assemble_generator(
                value["total"],
                base_stationary,
                base_dm,
            )
            for key, value in mass_components.items()
        },
        "direct_dm_outer": {
            key: _assemble_generator(
                base_mass,
                base_stationary,
                value["total"],
            )
            for key, value in direct_outer_components.items()
        },
        "direct_dm_action": {
            key: _assemble_generator(
                base_mass,
                base_stationary,
                value["total"],
            )
            for key, value in direct_action_components.items()
        },
    }
    generator_base_keys = {
        "stationary_jacobian": base_inner_key,
        "storage_matrix": base_inner_key,
        "direct_dm_outer": base_outer_key,
        "direct_dm_action": base_action_key,
    }
    generator_scans = {
        name: _scan_variants(
            variants,
            base_key=generator_base_keys[name],
            directions=directions,
            row_kind="primitive",
        )
        for name, variants in generator_variants.items()
    }

    base_generator = generator_variants["direct_dm_outer"][base_outer_key]
    production_generator = np.asarray(operator_arrays["dynamic"], dtype=float)
    legacy_production_difference = _matrix_comparison(
        base_generator,
        production_generator,
        directions,
        row_kind="primitive",
    )
    repaired_jvp_arrays, repaired_jvp = _repaired_generator_jvp_contract(
        base_generator,
        operator_arrays,
        operator_metadata,
    )
    fresh_jvp_arrays, fresh_jvp = (
        _fresh_independent_vector_field_jvp_contract(
            initial,
            vector,
            base_generator,
            operator_arrays,
            directions,
            base_vector_field=base_vector_field,
            base_branch_state=base_branch_state,
        )
    )
    factorization_defect = float(
        causal_five_field_assemble_evolving_tangent(
            base_mass,
            base_stationary,
            base_dm,
        )["maximum_scaled_generator_factorization_defect"]
    )
    maximum_mass_component_reconstruction_defect = max(
        mass_component_reconstruction_defects.values()
    )
    dm_component_reconstruction_defects = {
        f"outer_{key}": float(
            value["maximum_scaled_component_reconstruction_defect"]
        )
        for key, value in direct_outer_results.items()
    }
    dm_component_reconstruction_defects.update(
        {
            f"action_{key}": float(
                value["maximum_scaled_component_reconstruction_defect"]
            )
            for key, value in direct_action_results.items()
        }
    )
    maximum_diagnostic_dm_component_reconstruction_defect = max(
        dm_component_reconstruction_defects.values()
    )
    selected_dm_component_reconstruction_defect = float(
        direct_outer_results[base_outer_key][
            "maximum_scaled_component_reconstruction_defect"
        ]
    )
    factorization_passed = bool(
        factorization_defect <= MAXIMUM_GENERATOR_FACTORIZATION_DEFECT
    )
    mass_component_reconstruction_passed = bool(
        maximum_mass_component_reconstruction_defect
        <= MAXIMUM_STORAGE_COMPONENT_RECONSTRUCTION_DEFECT
    )
    dm_component_reconstruction_passed = bool(
        selected_dm_component_reconstruction_defect
        <= MAXIMUM_STORAGE_COMPONENT_RECONSTRUCTION_DEFECT
    )
    binding_block_scan_names = (
        "stationary_jacobian",
        "total_storage_matrix",
        "conserved_storage_matrix",
        "vertical_storage_matrix",
    )
    diagnostic_block_scan_names = tuple(
        name for name in block_scans if name not in binding_block_scan_names
    )
    binding_generator_scan_names = (
        "stationary_jacobian",
        "storage_matrix",
    )
    diagnostic_generator_scan_names = tuple(
        name
        for name in generator_scans
        if name not in binding_generator_scan_names
    )
    direct_full_matrix_block_scan_names = (
        "direct_total_dm_outer",
        "direct_total_dm_action",
    )
    direct_full_matrix_generator_scan_names = (
        "direct_dm_outer",
        "direct_dm_action",
    )
    direct_raw_dm_full_matrix_step_scans_passed = bool(
        all(
            comparison["frobenius_passed"]
            for name in direct_full_matrix_block_scan_names
            for comparison in block_scans[name]["comparisons"].values()
        )
    )
    direct_generator_full_matrix_step_scans_passed = bool(
        all(
            comparison["frobenius_passed"]
            for name in direct_full_matrix_generator_scan_names
            for comparison in generator_scans[name]["comparisons"].values()
        )
    )
    direct_full_matrix_step_scans_passed = (
        direct_generator_full_matrix_step_scans_passed
    )
    binding_step_scans_passed = bool(
        all(block_scans[name]["passed"] for name in binding_block_scan_names)
        and all(
            generator_scans[name]["passed"]
            for name in binding_generator_scan_names
        )
        and direct_full_matrix_step_scans_passed
    )
    legacy_block_variants = {
        "nested_dm_inner": legacy_dm_inner,
        "nested_dm_outer": legacy_dm_outer,
        "nested_vertical_action": legacy_dm_vertical,
    }
    legacy_base_keys = {
        "nested_dm_inner": base_inner_key,
        "nested_dm_outer": (
            f"{LEGACY_BASE_OUTER_DIFFERENCE_STEP:.0e}"
        ),
        "nested_vertical_action": (
            f"{LEGACY_BASE_VERTICAL_ACTION_DIFFERENCE_STEP:.0e}"
        ),
    }
    legacy_scans = {
        name: _scan_variants(
            variants,
            base_key=legacy_base_keys[name],
            directions=directions,
            row_kind="conservation",
        )
        for name, variants in legacy_block_variants.items()
    }
    arrays = {
        **{
            f"block_{name}_{key}": values
            for name, variants in block_variants.items()
            for key, values in variants.items()
        },
        **{
            f"generator_{name}_{key}": values
            for name, variants in generator_variants.items()
            for key, values in variants.items()
        },
        **{
            f"legacy_block_{name}_{key}": values
            for name, variants in legacy_block_variants.items()
            for key, values in variants.items()
        },
        **repaired_jvp_arrays,
        **fresh_jvp_arrays,
        "repaired_dynamic": base_generator,
    }
    metadata = {
        "directions": tuple(directions),
        "block_scans": block_scans,
        "generator_scans": generator_scans,
        "binding_block_scan_names": binding_block_scan_names,
        "diagnostic_block_scan_names": diagnostic_block_scan_names,
        "binding_generator_scan_names": binding_generator_scan_names,
        "diagnostic_generator_scan_names": (
            diagnostic_generator_scan_names
        ),
        "binding_step_scans_passed": binding_step_scans_passed,
        "direct_full_matrix_block_scan_names": (
            direct_full_matrix_block_scan_names
        ),
        "direct_full_matrix_generator_scan_names": (
            direct_full_matrix_generator_scan_names
        ),
        "direct_full_matrix_step_scans_passed": (
            direct_full_matrix_step_scans_passed
        ),
        "direct_raw_dm_full_matrix_step_scans_passed": (
            direct_raw_dm_full_matrix_step_scans_passed
        ),
        "direct_generator_full_matrix_step_scans_passed": (
            direct_generator_full_matrix_step_scans_passed
        ),
        "direct_dm_step_scan_semantics": (
            "full-matrix relative-Frobenius local step stability of the assembled "
            "scaled generator is binding; the raw un-inverted DM block and "
            "per-direction matrix-versus-matrix JVP ratios are diagnostic "
            "because their fixed row scaling is ill conditioned; the "
            "selected generator is independently bound by fresh nonlinear "
            "vector-field secants"
        ),
        "legacy_nested_generator_difference": (
            legacy_production_difference
        ),
        "legacy_nested_dm_step_scans_nonbinding": legacy_scans,
        "repaired_independent_vector_field_jvp": repaired_jvp,
        "cached_repaired_jvp_semantics": (
            "diagnostic only; the cached suite uses the older WP10c8i "
            "branch classification and cannot veto the fresh strict "
            "all-direction contract"
        ),
        "fresh_full_direction_independent_vector_field_jvp": fresh_jvp,
        "fresh_base_vector_field_rate_contract": base_rate_contract,
        "storage_matrix_component_reconstruction_defects": (
            mass_component_reconstruction_defects
        ),
        "maximum_storage_matrix_component_reconstruction_defect": (
            maximum_mass_component_reconstruction_defect
        ),
        "storage_matrix_component_reconstruction_passed": (
            mass_component_reconstruction_passed
        ),
        "storage_rate_component_reconstruction_defects": (
            dm_component_reconstruction_defects
        ),
        "maximum_storage_rate_component_reconstruction_defect": (
            maximum_diagnostic_dm_component_reconstruction_defect
        ),
        "selected_storage_rate_component_reconstruction_defect": (
            selected_dm_component_reconstruction_defect
        ),
        "storage_rate_component_reconstruction_passed": (
            dm_component_reconstruction_passed
        ),
        "storage_rate_derivative_backend": "direct_action",
        "maximum_generator_factorization_defect": factorization_defect,
        "generator_factorization_passed": factorization_passed,
        "passed": bool(
            binding_step_scans_passed
            and fresh_jvp["passed"]
            and base_rate_contract["passed"]
            and factorization_passed
            and mass_component_reconstruction_passed
            and dm_component_reconstruction_passed
        ),
    }
    return arrays, metadata


def _base_repaired_tangent(
    initial: dict,
    vector: np.ndarray,
    operator_arrays: dict[str, np.ndarray],
    operator_metadata: dict,
) -> tuple[dict[str, np.ndarray], dict]:
    """Certify one base direct-action generator without a step ladder."""

    primitive_scales = np.asarray(
        operator_arrays["primitive_column_scales"],
        dtype=float,
    )
    directions = _normalized_directions(
        initial,
        vector,
        primitive_scales,
    )
    (
        _base_primitives,
        base_vector_field,
        base_branch_state,
        base_rate_contract,
    ) = _fresh_base_vector_field(initial, vector, operator_arrays)
    fresh_physical_rate = (
        np.asarray(
            base_vector_field["scaled_primitive_rate_per_s"],
            dtype=float,
        ).ravel()
        * primitive_scales
    )
    result = _direct_action_storage_rate_result(
        initial,
        vector,
        operator_arrays,
        physical_rate_per_s=fresh_physical_rate,
    )
    dm = _result_components(result)
    mass = np.asarray(
        operator_arrays["direct_vector_storage_descriptor"],
        dtype=float,
    )
    stationary = np.asarray(
        operator_arrays["stationary_jacobian"],
        dtype=float,
    )
    assembled = causal_five_field_assemble_evolving_tangent(
        mass,
        stationary,
        dm["total"],
    )
    generator = np.asarray(
        assembled["evolving_scaled_generator_per_s"],
        dtype=float,
    )
    jvp_arrays, jvp = _repaired_generator_jvp_contract(
        generator,
        operator_arrays,
        operator_metadata,
    )
    fresh_jvp_arrays, fresh_jvp = (
        _fresh_independent_vector_field_jvp_contract(
            initial,
            vector,
            generator,
            operator_arrays,
            directions,
            base_vector_field=base_vector_field,
            base_branch_state=base_branch_state,
        )
    )
    component_defect = float(
        result["maximum_scaled_component_reconstruction_defect"]
    )
    factorization_defect = float(
        assembled["maximum_scaled_generator_factorization_defect"]
    )
    component_passed = bool(
        component_defect
        <= MAXIMUM_STORAGE_COMPONENT_RECONSTRUCTION_DEFECT
    )
    factorization_passed = bool(
        factorization_defect <= MAXIMUM_GENERATOR_FACTORIZATION_DEFECT
    )
    arrays = {
        "repaired_dynamic": generator,
        "repaired_storage_rate_derivative": dm["total"],
        "repaired_conserved_storage_rate_derivative": dm["conserved"],
        "repaired_vertical_storage_rate_derivative": dm["vertical"],
        **jvp_arrays,
        **fresh_jvp_arrays,
    }
    return arrays, {
        "scope": (
            "base direct-storage-action DM and repaired generator; step "
            "ladders remain locked to t_0 and t_0p10"
        ),
        "separated_scan_evaluated": False,
        "storage_rate_derivative_backend": "direct_action",
        "maximum_storage_rate_component_reconstruction_defect": (
            component_defect
        ),
        "storage_rate_component_reconstruction_passed": component_passed,
        "maximum_generator_factorization_defect": factorization_defect,
        "generator_factorization_passed": factorization_passed,
        "repaired_independent_vector_field_jvp": jvp,
        "cached_repaired_jvp_semantics": (
            "diagnostic only; fresh strict all-direction secants are binding"
        ),
        "fresh_full_direction_independent_vector_field_jvp": fresh_jvp,
        "fresh_base_vector_field_rate_contract": base_rate_contract,
        "passed": bool(
            component_passed
            and factorization_passed
            and fresh_jvp["passed"]
            and base_rate_contract["passed"]
        ),
    }


def _branch_certification(
    operator_arrays: dict[str, np.ndarray],
    operator_metadata: dict,
    *,
    candidate_coverages: tuple[RusanovCandidateCoverage, ...] = (),
    nonlinear_remainder_rate: float | None = None,
    nonlinear_output_remainder_bounds: dict[
        tuple[str, float], np.ndarray
    ] | None = None,
    nonlinear_remainder_certified: bool = False,
    certified_neighborhood_radius: float | None = None,
) -> dict:
    """Certify simultaneous finite-time switching for every moment level."""

    left = np.asarray(
        operator_arrays[
            "production_rusanov_kink_generator_left_factors"
        ],
        dtype=float,
    )
    right = np.asarray(
        operator_arrays[
            "production_rusanov_kink_generator_right_factors"
        ],
        dtype=float,
    )
    branch_count = int(left.shape[1])
    if right.shape != left.shape:
        raise ValueError("Rusanov branch factor shapes differ")
    nonlinear_output_remainder_bounds = (
        {}
        if nonlinear_output_remainder_bounds is None
        else nonlinear_output_remainder_bounds
    )
    level_rows = {}
    for level_index, level in enumerate(operator_metadata["levels"]):
        output, gates, names, _blocks = wp10c8i._response_stack(
            operator_arrays,
            operator_metadata,
            level_index,
        )
        direct = wp10c8i._rusanov_kink_instantaneous_output_deltas(
            operator_arrays,
            operator_metadata,
            level_index,
        )
        horizon_rows = {}
        for horizon in LOCKED_FINITE_TIME_HORIZONS_SECONDS:
            certificate = certify_cached_rusanov_finite_neighborhood(
                operator_arrays,
                output_operator=output,
                output_gates=gates,
                horizon_seconds=horizon,
                direct_output_deltas=direct,
                candidate_coverages=candidate_coverages,
                nonlinear_remainder_rate=nonlinear_remainder_rate,
                nonlinear_output_remainder_bounds=(
                    nonlinear_output_remainder_bounds.get(
                        (level["name"], horizon)
                    )
                ),
                nonlinear_remainder_certified=(
                    nonlinear_remainder_certified
                ),
                coefficient_bounds=np.ones(branch_count, dtype=float),
                initial_state_radius=1.0,
                certified_neighborhood_radius=(
                    certified_neighborhood_radius
                ),
                maximum_gate_fraction=(
                    MAXIMUM_RUSANOV_FINITE_TIME_GATE_FRACTION
                ),
            )
            row = certificate.as_dict()
            row["consequential_branch_count"] = branch_count
            row["controlling_output"] = (
                names[
                    int(
                        np.argmax(
                            np.asarray(
                                row.get("per_output_gate_fractions", [0.0]),
                                dtype=float,
                            )
                        )
                    )
                ]
                if names and row.get("per_output_gate_fractions")
                else None
            )
            horizon_rows[f"{horizon:.6g}"] = row
        level_rows[level["name"]] = horizon_rows
    all_rows_binding = bool(
        all(
            row["binding"]
            for horizons in level_rows.values()
            for row in horizons.values()
        )
    )
    return {
        "consequential_branch_count": branch_count,
        "interior_face_count": max(
            np.asarray(operator_arrays["dynamic"]).shape[0] // 5 - 1,
            0,
        ),
        "candidate_coverage_count": len(candidate_coverages),
        "nonlinear_remainder_supplied": (
            nonlinear_remainder_rate is not None
        ),
        "certified_neighborhood_radius": certified_neighborhood_radius,
        "nonlinear_output_remainder_set_count": len(
            nonlinear_output_remainder_bounds
        ),
        "levels": level_rows,
        "all_rows_binding": all_rows_binding,
        "passed": bool(
            all_rows_binding
            and all(
                row["passed"]
                for horizons in level_rows.values()
                for row in horizons.values()
            )
        ),
    }


def _cache_path(n_cells: int, label: str) -> Path:
    return CACHE_DIRECTORY / f"wp10c8j_n{n_cells}_{label}.npz"


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
            json.dumps(metadata, sort_keys=True, allow_nan=False)
        ),
    )


def _validate_certification_cache_payload(
    arrays: dict[str, np.ndarray],
    metadata: dict,
    *,
    n_cells: int,
    label: str,
) -> None:
    """Reject metadata-only or internally inconsistent certification caches."""

    width = 5 * int(n_cells)
    tangent = metadata.get("separated_tangent", {})
    branch = metadata.get("rusanov_finite_neighborhood", {})
    fresh = tangent.get(
        "fresh_full_direction_independent_vector_field_jvp",
        {},
    )
    direction_rows = fresh.get("directions", {})
    if set(direction_rows) != LOCKED_PHYSICAL_DIRECTION_NAMES:
        raise RuntimeError("WP10c8j cache has incomplete physical directions")
    if tuple(fresh.get("centered_difference_steps", ())) != tuple(
        INDEPENDENT_VECTOR_FIELD_JVP_STEPS
    ):
        raise RuntimeError("WP10c8j cache uses a different secant ladder")

    expected_shapes: dict[str, tuple[int, ...]] = {
        "repaired_dynamic": (width, width),
        "repaired_storage_rate_derivative": (width, width),
        "repaired_conserved_storage_rate_derivative": (width, width),
        "repaired_vertical_storage_rate_derivative": (width, width),
    }
    vector_suffixes = (
        "direction",
        "direct",
        "forward",
        "backward",
        "predicted",
        "base_rate",
    )
    for direction_name in LOCKED_PHYSICAL_DIRECTION_NAMES:
        for step in INDEPENDENT_VECTOR_FIELD_JVP_STEPS:
            step_key = f"{step:.0e}"
            prefix = f"fresh_{direction_name}_step_{step_key}_"
            for suffix in vector_suffixes:
                expected_shapes[
                    prefix + "independent_vector_field_jvp_" + suffix
                ] = (width,)

    if label in FULL_SCAN_ANCHORS:
        for step in INNER_DIFFERENCE_STEPS:
            step_key = f"{step:.0e}"
            expected_shapes[
                f"generator_stationary_jacobian_{step_key}"
            ] = (width, width)
            expected_shapes[
                f"generator_storage_matrix_{step_key}"
            ] = (width, width)
        for step in OUTER_DIFFERENCE_STEPS:
            expected_shapes[
                f"generator_direct_dm_outer_{step:.0e}"
            ] = (width, width)
        for step in VERTICAL_ACTION_DIFFERENCE_STEPS:
            expected_shapes[
                f"generator_direct_dm_action_{step:.0e}"
            ] = (width, width)

    for name, shape in expected_shapes.items():
        if name not in arrays or np.asarray(arrays[name]).shape != shape:
            raise RuntimeError(
                f"WP10c8j cache array {name} is absent or malformed"
            )

    if tangent.get("separated_scan_evaluated", False):
        expected_tangent_pass = bool(
            tangent.get("binding_step_scans_passed", False)
            and fresh.get("passed", False)
            and tangent.get("fresh_base_vector_field_rate_contract", {}).get(
                "passed", False
            )
            and tangent.get("generator_factorization_passed", False)
            and tangent.get(
                "storage_matrix_component_reconstruction_passed", False
            )
            and tangent.get(
                "storage_rate_component_reconstruction_passed", False
            )
        )
    else:
        expected_tangent_pass = bool(
            fresh.get("passed", False)
            and tangent.get("fresh_base_vector_field_rate_contract", {}).get(
                "passed", False
            )
            and tangent.get("generator_factorization_passed", False)
            and tangent.get(
                "storage_rate_component_reconstruction_passed", False
            )
        )
    if bool(tangent.get("passed", False)) != expected_tangent_pass:
        raise RuntimeError("WP10c8j cache tangent decision is inconsistent")

    level_rows = [
        row
        for horizons in branch.get("levels", {}).values()
        for row in horizons.values()
    ]
    expected_branch_pass = bool(
        level_rows
        and branch.get("all_rows_binding", False)
        and all(row.get("binding", False) for row in level_rows)
        and all(row.get("passed", False) for row in level_rows)
    )
    if bool(branch.get("passed", False)) != expected_branch_pass:
        raise RuntimeError("WP10c8j cache Rusanov decision is inconsistent")
    if bool(metadata.get("passed", False)) != bool(
        expected_tangent_pass and expected_branch_pass
    ):
        raise RuntimeError("WP10c8j cache top-level decision is inconsistent")


def _operator_source_path(n_cells: int, label: str) -> Path:
    return (
        OPERATOR_SOURCE_DIRECTORY
        / f"wp10c8j_N{n_cells:03d}_{label}_moment_operators.npz"
    )


def _load_npz_payload(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as source:
        metadata = json.loads(str(source["metadata_json"].item()))
        arrays = {
            name: np.asarray(source[name], dtype=float)
            for name in source.files
            if name != "metadata_json"
        }
    return arrays, metadata


def _operator_array_hashes(
    arrays: dict[str, np.ndarray],
) -> dict[str, str]:
    return {
        name: wp10c8i._array_sha256(np.asarray(values, dtype=float))
        for name, values in sorted(arrays.items())
    }


def _validate_operator_payload(
    arrays: dict[str, np.ndarray],
    metadata: dict,
    *,
    initial: dict,
    vector: np.ndarray,
    label: str,
    shell_edges_rg: np.ndarray,
    require_current_contract: bool,
) -> None:
    n_cells = int(initial["state"].n_cells)
    common = bool(
        metadata.get("schema_version") == wp10c8i.CACHE_SCHEMA_VERSION
        and metadata.get("work_package") == "WP10c8i"
        and metadata.get("base_commit") == WP10C8I_BASE_COMMIT
        and metadata.get("n_cells") == n_cells
        and metadata.get("anchor_label") == label
        and metadata.get("state_vector_sha256")
        == wp10c8i._array_sha256(vector)
        and np.array_equal(
            np.asarray(metadata.get("shell_edges_rg"), dtype=float),
            np.asarray(shell_edges_rg, dtype=float),
        )
        and arrays.get("dynamic", np.empty((0, 0))).shape
        == (5 * n_cells, 5 * n_cells)
        and all(np.all(np.isfinite(value)) for value in arrays.values())
    )
    if require_current_contract:
        current_contract = wp10c8i._operator_contract(
            initial["context"],
            shell_edges_rg,
        )
        source = metadata.get("wp10c8j_operator_source", {})
        common = bool(
            common
            and metadata.get("operator_contract") == current_contract
            and metadata.get("operator_contract_sha256")
            == wp10c8i._text_sha256(
                json.dumps(current_contract, sort_keys=True)
            )
            and source.get("schema_version") == 1
            and source.get("work_package") == WORK_PACKAGE
            and source.get("base_commit") == BASE_COMMIT
            and source.get("array_sha256")
            == _operator_array_hashes(arrays)
        )
    if not common:
        raise RuntimeError("operator-source payload is not authorized")


def _build_wp10c8j_operator_source(
    path: Path,
    *,
    initial: dict,
    vector: np.ndarray,
    label: str,
    role: str,
    shell_edges_rg: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict]:
    """Build a versioned source artifact without touching WP10c8i files."""

    arrays, metadata = wp10c8i._build_operator_cache(
        initial,
        vector,
        label,
        role,
        shell_edges_rg,
    )
    metadata = dict(metadata)
    metadata["wp10c8j_operator_source"] = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "base_commit": BASE_COMMIT,
        "array_sha256": _operator_array_hashes(arrays),
        "semantics": (
            "explicit WP10c8j-owned rebuild; canonical WP10c8i artifact "
            "was unavailable at its immutable hash"
        ),
    }
    wp10c8i._write_cache(path, arrays, metadata)
    arrays, metadata = _load_npz_payload(path)
    _validate_operator_payload(
        arrays,
        metadata,
        initial=initial,
        vector=vector,
        label=label,
        shell_edges_rg=shell_edges_rg,
        require_current_contract=True,
    )
    return arrays, metadata


def _authorized_wp10c8i_operator_cache(
    initial: dict,
    vector: np.ndarray,
    label: str,
    role: str,
    shell_edges_rg: np.ndarray,
    *,
    rebuild_operator_source: bool = False,
) -> tuple[dict[str, np.ndarray], dict, dict]:
    """Load immutable WP10c8i evidence or an explicit WP10c8j rebuild.

    A canonical hash mismatch is never accepted from self-reported metadata.
    When the ignored canonical runtime artifact is unavailable, the caller
    must explicitly authorize a fresh build at a separate WP10c8j-owned path.
    """

    n_cells = int(initial["state"].n_cells)
    evidence = json.loads(WP10C8I_OUTPUT.read_text(encoding="utf-8"))
    canonical = evidence["operator_provenance"][str(n_cells)][label]
    canonical_path = ROOT / str(canonical["path"])
    canonical_match = bool(
        canonical_path.exists()
        and _sha256(canonical_path) == canonical["sha256"]
    )
    if canonical_match:
        path = canonical_path
        arrays, metadata = _load_npz_payload(path)
        _validate_operator_payload(
            arrays,
            metadata,
            initial=initial,
            vector=vector,
            label=label,
            shell_edges_rg=shell_edges_rg,
            require_current_contract=False,
        )
        source_kind = "canonical_wp10c8i_artifact"
    else:
        path = _operator_source_path(n_cells, label)
        if rebuild_operator_source:
            arrays, metadata = _build_wp10c8j_operator_source(
                path,
                initial=initial,
                vector=vector,
                label=label,
                role=role,
                shell_edges_rg=shell_edges_rg,
            )
        elif path.exists():
            arrays, metadata = _load_npz_payload(path)
            _validate_operator_payload(
                arrays,
                metadata,
                initial=initial,
                vector=vector,
                label=label,
                shell_edges_rg=shell_edges_rg,
                require_current_contract=True,
            )
        else:
            raise RuntimeError(
                f"canonical WP10c8i cache {canonical_path.name} is not at "
                "its immutable hash and no WP10c8j operator source exists; "
                "rerun with --rebuild-operator-source"
            )
        source_kind = "versioned_wp10c8j_operator_source"
    current_sha256 = _sha256(path)
    return arrays, metadata, {
        "path": _relative(path),
        "sha256": current_sha256,
        "canonical_wp10c8i_sha256": canonical["sha256"],
        "canonical_hash_matched": canonical_match,
        "state_vector_sha256": canonical["state_vector_sha256"],
        "source_kind": source_kind,
        "explicit_rebuild_performed": bool(
            source_kind == "versioned_wp10c8j_operator_source"
            and rebuild_operator_source
        ),
        "canonical_path_left_unmodified": True,
    }


def _build_certification_cache(
    initial: dict,
    vector: np.ndarray,
    label: str,
    role: str,
    shell_edges_rg: np.ndarray,
    *,
    rebuild_operator_source: bool = False,
) -> tuple[dict[str, np.ndarray], dict, dict]:
    operator_arrays, operator_metadata, operator_provenance = (
        _authorized_wp10c8i_operator_cache(
            initial,
            vector,
            label,
            role,
            shell_edges_rg,
            rebuild_operator_source=rebuild_operator_source,
        )
    )
    started = time.perf_counter()
    cached_numerical_contract = _cached_operator_numerical_contract(
        operator_metadata
    )
    if label in FULL_SCAN_ANCHORS:
        arrays, separated = _separated_tangent_scan(
            initial,
            vector,
            operator_arrays,
            operator_metadata,
        )
        tangent = {
            **separated,
            "separated_scan_evaluated": True,
            "cached_operator_numerical_contract": (
                cached_numerical_contract
            ),
            "passed": bool(
                separated["passed"]
                and cached_numerical_contract["passed"]
            ),
        }
    else:
        arrays, repaired = _base_repaired_tangent(
            initial,
            vector,
            operator_arrays,
            operator_metadata,
        )
        tangent = {
            **repaired,
            "cached_operator_numerical_contract": (
                cached_numerical_contract
            ),
            "passed": bool(
                repaired["passed"]
                and cached_numerical_contract["passed"]
            ),
        }
    repaired_operator_arrays = dict(operator_arrays)
    repaired_operator_arrays["dynamic"] = np.asarray(
        arrays["repaired_dynamic"],
        dtype=float,
    )
    branch = _branch_certification(
        repaired_operator_arrays,
        operator_metadata,
    )
    metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "base_commit": BASE_COMMIT,
        "n_cells": int(initial["state"].n_cells),
        "anchor_label": label,
        "anchor_role": role,
        "state_vector_sha256": wp10c8i._array_sha256(vector),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "operator_contract_sha256": wp10c8i._text_sha256(
            json.dumps(
                wp10c8i._operator_contract(
                    initial["context"],
                    shell_edges_rg,
                ),
                sort_keys=True,
            )
        ),
        "wp10c8i_operator_cache": operator_provenance,
        "separated_tangent": tangent,
        "rusanov_finite_neighborhood": branch,
        "passed": bool(tangent["passed"] and branch["passed"]),
        "wall_seconds": time.perf_counter() - started,
    }
    path = _cache_path(int(initial["state"].n_cells), label)
    _write_cache(path, arrays, metadata)
    return arrays, metadata, {
        "path": _relative(path),
        "sha256": _sha256(path),
        "wp10c8i_operator_cache": operator_provenance,
    }


def _load_certification_cache(
    path: Path,
    *,
    initial: dict,
    vector: np.ndarray,
    label: str,
    shell_edges_rg: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as source:
        metadata = json.loads(str(source["metadata_json"].item()))
        arrays = {
            name: np.asarray(source[name], dtype=float)
            for name in source.files
            if name != "metadata_json"
        }
    if not (
        metadata.get("schema_version") == CACHE_SCHEMA_VERSION
        and metadata.get("work_package") == WORK_PACKAGE
        and metadata.get("base_commit") == BASE_COMMIT
        and metadata.get("n_cells") == int(initial["state"].n_cells)
        and metadata.get("anchor_label") == label
        and metadata.get("state_vector_sha256")
        == wp10c8i._array_sha256(vector)
        and metadata.get("runner_sha256")
        == _sha256(Path(__file__).resolve())
        and metadata.get("operator_contract_sha256")
        == wp10c8i._text_sha256(
            json.dumps(
                wp10c8i._operator_contract(
                    initial["context"],
                    shell_edges_rg,
                ),
                sort_keys=True,
            )
        )
        and arrays
        and all(np.all(np.isfinite(value)) for value in arrays.values())
    ):
        raise RuntimeError(f"WP10c8j cache {path.name} differs")
    _validate_certification_cache_payload(
        arrays,
        metadata,
        n_cells=int(initial["state"].n_cells),
        label=label,
    )
    operator_path = ROOT / metadata["wp10c8i_operator_cache"]["path"]
    if (
        not operator_path.exists()
        or _sha256(operator_path)
        != metadata["wp10c8i_operator_cache"]["sha256"]
    ):
        raise RuntimeError("WP10c8i operator cache provenance changed")
    return arrays, metadata


def _certification_cache(
    initial: dict,
    vector: np.ndarray,
    label: str,
    role: str,
    shell_edges_rg: np.ndarray,
    *,
    force: bool,
    rebuild_operator_source: bool = False,
) -> tuple[dict[str, np.ndarray], dict, dict]:
    path = _cache_path(int(initial["state"].n_cells), label)
    rebuild = bool(force or rebuild_operator_source or not path.exists())
    if not rebuild:
        try:
            arrays, metadata = _load_certification_cache(
                path,
                initial=initial,
                vector=vector,
                label=label,
                shell_edges_rg=shell_edges_rg,
            )
        except (KeyError, ValueError, RuntimeError, json.JSONDecodeError):
            rebuild = True
    if rebuild:
        return _build_certification_cache(
            initial,
            vector,
            label,
            role,
            shell_edges_rg,
            rebuild_operator_source=rebuild_operator_source,
        )
    return arrays, metadata, {
        "path": _relative(path),
        "sha256": _sha256(path),
        "wp10c8i_operator_cache": metadata["wp10c8i_operator_cache"],
    }


def _campaign_decision(
    certifications: dict[str, dict],
    *,
    complete_selection: bool,
) -> tuple[str, str]:
    tangent_passed = bool(
        certifications
        and all(
            row["separated_tangent"]["passed"]
            for row in certifications.values()
        )
    )
    branch_binding = bool(
        certifications
        and all(
            row["rusanov_finite_neighborhood"]["all_rows_binding"]
            for row in certifications.values()
        )
    )
    branch_passed = bool(
        branch_binding
        and all(
            row["rusanov_finite_neighborhood"]["passed"]
            for row in certifications.values()
        )
    )
    if not complete_selection:
        return (
            "wp10c8j_partial_certification_completed",
            "complete_all_locked_n64_n128_anchor_certifications",
        )
    if not tangent_passed and not branch_binding:
        return (
            "wp10c8j_smooth_tangent_failed_rusanov_certificate_absent",
            "repair_the_smooth_tangent_and_supply_the_finite_branch_contract",
        )
    if tangent_passed and not branch_binding:
        return (
            "wp10c8j_smooth_tangent_certified_rusanov_certificate_absent",
            "supply_the_binding_finite_neighborhood_rusanov_contract",
        )
    if not tangent_passed:
        return (
            "wp10c8j_smooth_tangent_contract_failed",
            "repair_the_controlling_smooth_derivative_then_repeat_wp10c8j",
        )
    if not branch_passed:
        return (
            "wp10c8j_binding_rusanov_contract_failed",
            "tighten_or_repair_the_measured_finite_branch_contract",
        )
    return (
        "wp10c8j_tangent_and_finite_branch_contract_passed",
        "repeat_the_unchanged_wp10c8i_moment_audit_in_a_separate_package",
    )


def main() -> None:
    arguments = _arguments()
    contract = _locked_contract()
    selected_resolutions = (
        tuple(dict.fromkeys(arguments.resolution))
        if arguments.resolution
        else LOCKED_RESOLUTIONS
    )
    selected_labels = (
        set(arguments.anchor)
        if arguments.anchor
        else {row[0] for row in LOCKED_ANCHORS}
    )
    selected_anchors = tuple(
        row for row in LOCKED_ANCHORS if row[0] in selected_labels
    )
    complete_selection = bool(
        selected_resolutions == LOCKED_RESOLUTIONS
        and selected_anchors == LOCKED_ANCHORS
    )
    if not complete_selection and not (
        arguments.preflight or arguments.certification_only
    ):
        raise ValueError(
            "partial selection requires --preflight or --certification-only"
        )

    authorization, authorization_sha256 = _validate_authorization()
    if arguments.preflight:
        initial, vectors, _state_provenance = wp10c8i._load_states()
        shell_edges_rg = np.asarray(
            wp10c8h._common_shell_edges(initial)["five_shell"],
            dtype=float,
        )
        prior_state_provenance = authorization["state_provenance"]
        selected_prior_caches = {}
        for n_cells in selected_resolutions:
            for label, _seconds, role in selected_anchors:
                key = f"n{n_cells}_{label}"
                state_row = prior_state_provenance[str(n_cells)][label]
                _arrays, _metadata, provenance = (
                    _authorized_wp10c8i_operator_cache(
                        initial[n_cells],
                        vectors[n_cells][label],
                        label,
                        role,
                        shell_edges_rg,
                        rebuild_operator_source=False,
                    )
                )
                if provenance["state_vector_sha256"] != state_row[
                    "state_vector_sha256"
                ]:
                    raise RuntimeError(
                        f"operator/state provenance differs for {key}"
                    )
                selected_prior_caches[key] = provenance
        print(
            json.dumps(
                {
                    "work_package": WORK_PACKAGE,
                    "preflight": "passed",
                    "preflight_scope": (
                        "locked evidence, selected state, and immutable or "
                        "versioned operator provenance only; no tangent was "
                        "rebuilt"
                    ),
                    "locked_contract": contract,
                    "selected_resolutions": selected_resolutions,
                    "selected_anchors": tuple(row[0] for row in selected_anchors),
                    "five_shell_edges_rg": authorization["scope"][
                        "five_shell_edges_rg"
                    ],
                    "wp10c8i_authorization_decision": authorization["decision"],
                    "wp10c8i_next_authorization": authorization[
                        "next_authorization"
                    ],
                    "selected_prior_cache_provenance": (
                        selected_prior_caches
                    ),
                    "wp10c8j_certification_cache_rebuild_or_validation_"
                    "required": True,
                    "new_full_dae_trajectory_run": False,
                    "new_nonlinear_microburst_run": False,
                },
                sort_keys=True,
            )
        )
        return

    initial, vectors, state_provenance = wp10c8i._load_states()
    shell_edges_rg = np.asarray(
        wp10c8h._common_shell_edges(initial)["five_shell"],
        dtype=float,
    )

    certification_rows = {}
    cache_provenance = {}
    array_payload = {}
    for n_cells in selected_resolutions:
        for label, _seconds, role in selected_anchors:
            arrays, metadata, provenance = _certification_cache(
                initial[n_cells],
                vectors[n_cells][label],
                label,
                role,
                shell_edges_rg,
                force=arguments.force,
                rebuild_operator_source=(
                    arguments.rebuild_operator_source
                ),
            )
            key = f"n{n_cells}_{label}"
            certification_rows[key] = metadata
            cache_provenance[key] = provenance
            array_payload.update(
                {f"{key}_{name}": values for name, values in arrays.items()}
            )
            print(
                json.dumps(
                    {
                        "work_package": WORK_PACKAGE,
                        "phase": "tangent_certification",
                        "n_cells": n_cells,
                        "anchor": label,
                        "separated_tangent_passed": metadata[
                            "separated_tangent"
                        ]["passed"],
                        "rusanov_finite_neighborhood_passed": metadata[
                            "rusanov_finite_neighborhood"
                        ]["passed"],
                        "passed": metadata["passed"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    all_certifications_passed = bool(
        complete_selection
        and len(certification_rows)
        == len(LOCKED_RESOLUTIONS) * len(LOCKED_ANCHORS)
        and all(row["passed"] for row in certification_rows.values())
    )
    all_smooth_tangent_contracts_passed = bool(
        complete_selection
        and certification_rows
        and all(
            row["separated_tangent"]["passed"]
            for row in certification_rows.values()
        )
    )
    all_rusanov_rows_binding = bool(
        complete_selection
        and certification_rows
        and all(
            row["rusanov_finite_neighborhood"]["all_rows_binding"]
            for row in certification_rows.values()
        )
    )
    decision, next_authorization = _campaign_decision(
        certification_rows,
        complete_selection=complete_selection,
    )
    arrays_path = _absolute(arguments.arrays)
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **array_payload)
    payload = {
        "work_package": WORK_PACKAGE,
        "base_commit": BASE_COMMIT,
        "decision": decision,
        "next_authorization": next_authorization,
        "scope": {
            "description": (
                "Separated evolving-tangent and finite-neighborhood Rusanov "
                "certification before unchanged WP10c8i repetition"
            ),
            "locked_contract": contract,
            "selected_resolutions": selected_resolutions,
            "selected_anchors": tuple(row[0] for row in selected_anchors),
            "complete_locked_selection": complete_selection,
            "five_shell_edges_rg": shell_edges_rg.tolist(),
            "new_full_dae_trajectory_run": False,
            "new_nonlinear_microburst_run": False,
            "reduced_nonlinear_evolution_constructed": False,
            "moment_families_changed": False,
            "rusanov_production_operator_changed": False,
            "unchanged_wp10c8i_repeat_launched": False,
        },
        "authorization": {
            "wp10c8i_decision": authorization["decision"],
            "wp10c8i_evidence_sha256": authorization_sha256,
        },
        "state_provenance": state_provenance,
        "certifications": certification_rows,
        "cache_provenance": cache_provenance,
        "all_locked_certifications_passed": all_certifications_passed,
        "all_smooth_tangent_contracts_passed": (
            all_smooth_tangent_contracts_passed
        ),
        "all_rusanov_finite_neighborhood_rows_binding": (
            all_rusanov_rows_binding
        ),
        "unchanged_wp10c8i_repeat": {
            "performed": False,
            "policy": (
                "a passing WP10c8j result authorizes, but never launches, "
                "a separate unchanged WP10c8i repetition"
            ),
        },
        "gates": {
            "maximum_generator_relative_defect": (
                MAXIMUM_GENERATOR_RELATIVE_DEFECT
            ),
            "maximum_independent_vector_field_jvp_relative_defect": (
                MAXIMUM_INDEPENDENT_VECTOR_FIELD_JVP_RELATIVE_DEFECT
            ),
            "maximum_forward_backward_jvp_relative_defect": (
                MAXIMUM_FORWARD_BACKWARD_JVP_RELATIVE_DEFECT
            ),
            "maximum_jvp_additivity_relative_defect": (
                MAXIMUM_JVP_ADDITIVITY_RELATIVE_DEFECT
            ),
            "maximum_generator_factorization_defect": (
                MAXIMUM_GENERATOR_FACTORIZATION_DEFECT
            ),
            "maximum_storage_action_relative_defect": (
                MAXIMUM_STORAGE_ACTION_RELATIVE_DEFECT
            ),
            "maximum_storage_component_reconstruction_defect": (
                MAXIMUM_STORAGE_COMPONENT_RECONSTRUCTION_DEFECT
            ),
            "maximum_rusanov_generator_kink_relative_diameter": (
                MAXIMUM_RUSANOV_GENERATOR_KINK_RELATIVE_DIAMETER
            ),
            "maximum_rusanov_finite_time_gate_fraction": (
                MAXIMUM_RUSANOV_FINITE_TIME_GATE_FRACTION
            ),
        },
        "artifacts": {
            "arrays_path": _relative(arrays_path),
            "arrays_sha256": _sha256(arrays_path),
        },
    }
    output_path = _absolute(arguments.output)
    _write_json(output_path, payload)
    print(
        json.dumps(
            {
                "work_package": WORK_PACKAGE,
                "decision": decision,
                "next_authorization": next_authorization,
                "all_locked_certifications_passed": (
                    all_certifications_passed
                ),
                "output": _relative(output_path),
                "arrays": _relative(arrays_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
