"""Run the WP10c8w independent-anchor and excision-sensitivity audit.

WP10c8v localized the unresolved fast phase to the near-horizon transport
stencil, but its N256-equivalent state and rate were prolongated from N128.
This package first projects the fine local anchor onto the same instantaneous
coarse moment fiber, solves its descriptor rate independently, and rebuilds
an exact equal-coordinate inner-mode pair.  It then varies only the inner
trace and the safely interior excision edge, retaining the frozen exterior
buffer and production physics.

This remains a local frozen-linear audit.  A successful result can authorize
one higher local refinement and nonlinear truth test; it cannot certify a
formal fast average or a reduced architecture.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_phase_spatial_preflight_wp10c8v as wp10c8v

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    audit_causal_five_field_state_gates,
    causal_exact_coordinate_projection,
    causal_exact_equal_coordinate_lift_pair,
    causal_five_field_evolving_tangent_matrices,
    causal_five_field_moment_coordinate_ladder,
    causal_five_field_moment_coordinate_values,
    causal_five_field_reduced_descriptor_matrices,
    causal_five_field_state_from_primitives,
    make_kerr_schild_column_grid,
    pack_causal_five_field_state,
)


BASE_COMMIT = "6764fc117ce453b4deb5c6b1c275a19c7352b4be"
WORK_PACKAGE = "WP10c8w"
SCHEMA_VERSION = 1
THIS_RUNNER = (
    "scripts/run_causal_inner_anchor_excision_audit_wp10c8w.py"
)
CORE_DAE_FILE = (
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_dae_system.py"
)
CORE_FIBER_FILE = (
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_nonlinear_fiber.py"
)
CORE_TANGENT_FILE = (
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_spatial_audit.py"
)

MESHES = (64, 128, 256)
PARENT_MESHES = (64, 128)
TRACE_MODES = ("inherit", "cell_centered", "linear_outgoing")
PLACEMENT_OFFSETS = {
    "original": {128: 0, 256: 0},
    "one_n128_cell": {128: 1, 256: 2},
    "two_n128_cells": {128: 2, 256: 4},
}
COORDINATE_LEVEL = "plus_shell_stress_storage"
TARGET_SECONDS = wp10c8v.TARGET_SECONDS
TIME_SAMPLES = wp10c8v.TIME_SAMPLES
COMMON_EXTERIOR_INNER_RG = 2.2
BUFFER_WEIGHT_MULTIPLIER = 1.0e12

MAXIMUM_ANCHOR_COORDINATE_DEFECT = 1.0e-10
MAXIMUM_PAIR_COORDINATE_DEFECT = 2.0e-10
MAXIMUM_ACTIVE_SCALED_ANCHOR_CORRECTION = 1.0e-2
MAXIMUM_BUFFER_SCALED_ANCHOR_CORRECTION = 1.0e-8
MAXIMUM_CONSTRAINT_CONDITION = 1.0e10
MAXIMUM_STORAGE_ACTION_DEFECT = 5.0e-5
MAXIMUM_GENERATOR_FACTORIZATION_DEFECT = 1.0e-8
MAXIMUM_PROPAGATION_GROWTH_EXPONENT = 10.0
MINIMUM_SPATIAL_ORDER = 0.75
MINIMUM_SIGNED_COSINE = 0.90
MAXIMUM_ZERO_CROSSING_DEFECT = 0.10
MAXIMUM_FREQUENCY_DEFECT = 0.10
MAXIMUM_DAMPING_DEFECT = 0.25
MAXIMUM_EXTERIOR_PLACEMENT_HISTORY_DEFECT = 0.10

CHECKPOINT_DIRECTORY = (
    ROOT / "outputs/checkpoints/causal_five_field_wp10c8w"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/tables/"
    "causal_inner_anchor_excision_audit_wp10c8w.json"
)
DEFAULT_ARRAYS = (
    ROOT
    / "outputs/tables/"
    "causal_inner_anchor_excision_audit_wp10c8w_arrays.npz"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _zero_sources(n_cells: int):
    return wp10c8v._zero_sources(n_cells)


def _audit_local_state_gates(context, vector: np.ndarray) -> dict:
    """Apply physical gates appropriate to the audit-only local buffer."""

    audit = audit_causal_five_field_state_gates(context, vector)
    measured = audit["measured"]
    gates = audit["gates"]
    passed = bool(
        measured["maximum_h_over_r"] <= gates["maximum_h_over_r"]
        and measured["minimum_scattering_optical_depth"]
        > gates["minimum_scattering_optical_depth"]
        and measured["inner_incoming_characteristics"]
        == gates["inner_incoming_characteristics"]
        and measured["maximum_inner_light_cone_excess"]
        <= gates["maximum_inner_light_cone_excess"]
        and measured["maximum_scaled_algebraic_residual"]
        <= gates["maximum_scaled_algebraic_residual"]
    )
    return {
        **audit,
        "schema_version": "wp10c8w-local-frozen-exterior-state-gates-v1",
        "production_roche_outer_gate_applicable": False,
        "passed": passed,
    }


def _operator_paths(label: str) -> tuple[Path, Path]:
    return (
        CHECKPOINT_DIRECTORY / f"{label}.json",
        CHECKPOINT_DIRECTORY / f"{label}_arrays.npz",
    )


def _continuum_weights(
    context,
    *,
    active_outer_rg: float,
) -> np.ndarray:
    measures = np.asarray(context.grid.cell_measures, dtype=float)
    weights = np.repeat(
        measures / (5.0 * np.sum(measures)),
        5,
    )
    radius = (
        np.asarray(context.grid.centers, dtype=float)
        / context.grid.gravitational_radius
    )
    weights.reshape(-1, 5)[radius > active_outer_rg] *= (
        BUFFER_WEIGHT_MULTIPLIER
    )
    return weights


def _local_shell_edges_rg(
    context,
    shell_zero_outer_rg: float,
) -> np.ndarray:
    edges = np.asarray(
        (
            context.grid.edges[0] / context.grid.gravitational_radius,
            shell_zero_outer_rg,
            context.grid.edges[-1] / context.grid.gravitational_radius,
        ),
        dtype=float,
    )
    if not edges[0] < edges[1] < edges[2]:
        raise RuntimeError("WP10c8w local shell edges are not ordered")
    return edges


def _moment_values(
    context,
    primitives: np.ndarray,
    shell_edges_rg: np.ndarray,
):
    state = causal_five_field_state_from_primitives(context, primitives)
    values = causal_five_field_moment_coordinate_values(
        context,
        pack_causal_five_field_state(state),
        shell_edges_rg,
        shape_bands_rg=(),
    )
    return values.level(COORDINATE_LEVEL)


def _moment_evaluator(context, shell_edges_rg: np.ndarray):
    def evaluate(flat_primitives: np.ndarray) -> np.ndarray:
        level = _moment_values(
            context,
            np.asarray(flat_primitives, dtype=float).reshape(-1, 5),
            shell_edges_rg,
        )
        return np.asarray(level.coordinate_values, dtype=float)

    return evaluate


def _reduced_and_ladder(
    context,
    primitives: np.ndarray,
    shell_edges_rg: np.ndarray,
):
    state = causal_five_field_state_from_primitives(context, primitives)
    vector = pack_causal_five_field_state(state)
    reduced = causal_five_field_reduced_descriptor_matrices(
        context,
        vector,
        finite_difference_step=wp10c8v.FINITE_DIFFERENCE_STEP,
        descriptor_timestep_seconds=(
            wp10c8v.DESCRIPTOR_TIMESTEP_SECONDS
        ),
    )
    ladder = causal_five_field_moment_coordinate_ladder(
        context,
        vector,
        reduced,
        shell_edges_rg,
        shape_bands_rg=(),
    )
    return state, vector, reduced, ladder.level(COORDINATE_LEVEL)


def _project_fine_anchor(
    *,
    context,
    profiles: dict,
    shell_edges_rg: np.ndarray,
    target_values: np.ndarray,
    target_scales: np.ndarray,
    active_outer_rg: float,
) -> tuple[np.ndarray, dict]:
    provisional = np.asarray(profiles["primitives"], dtype=float)
    _state, _vector, _reduced, level = _reduced_and_ladder(
        context,
        provisional,
        shell_edges_rg,
    )
    constraint = (
        np.asarray(level.raw_constraint_matrix, dtype=float)
        / np.asarray(target_scales, dtype=float)[:, None]
    )
    projection = causal_exact_coordinate_projection(
        base_primitive_vector=provisional.ravel(),
        primitive_column_scales=np.asarray(
            _reduced["primitive_column_scales"],
            dtype=float,
        ),
        state_weights=_continuum_weights(
            context,
            active_outer_rg=active_outer_rg,
        ),
        physical_input_amplitudes=np.asarray(
            profiles["physical_input_amplitudes"],
            dtype=float,
        ).ravel(),
        target_coordinate_values=target_values,
        target_coordinate_scales=target_scales,
        constraint_matrix=constraint,
        coordinate_evaluator=_moment_evaluator(
            context,
            shell_edges_rg,
        ),
    )
    corrected = np.asarray(
        projection.primitive_vector,
        dtype=float,
    ).reshape(provisional.shape)
    radius = (
        np.asarray(context.grid.centers, dtype=float)
        / context.grid.gravitational_radius
    )
    scaled_increment = np.asarray(
        projection.scaled_increment,
        dtype=float,
    ).reshape(provisional.shape)
    active = radius <= active_outer_rg * (1.0 + 2.0e-14)
    active_norm = wp10c8v._continuum_norm(
        scaled_increment[active],
        np.asarray(context.grid.cell_measures, dtype=float)[active],
    )
    buffer_norm = wp10c8v._continuum_norm(
        scaled_increment[~active],
        np.asarray(context.grid.cell_measures, dtype=float)[~active],
    )
    maximum_active_scaled_correction = float(
        np.max(np.abs(scaled_increment[active]))
    )
    maximum_buffer_scaled_correction = float(
        np.max(np.abs(scaled_increment[~active]))
    )
    report = {
        "optimizer_success": projection.optimizer_success,
        "optimizer_status": projection.optimizer_status,
        "optimizer_message": projection.optimizer_message,
        "function_evaluations": projection.function_evaluations,
        "jacobian_evaluations": projection.jacobian_evaluations,
        "maximum_coordinate_defect": (
            projection.maximum_coordinate_defect
        ),
        "weighted_radius": projection.weighted_radius,
        "maximum_pointwise_amplitude_ratio": (
            projection.maximum_pointwise_amplitude_ratio
        ),
        "constraint_rank": projection.normal_basis.numerical_rank,
        "constraint_condition": (
            projection.normal_basis.condition_estimate
        ),
        "constraint_weighted_orthogonality_defect": (
            projection.normal_basis.weighted_orthogonality_defect
        ),
        "active_scaled_correction_norm": active_norm,
        "buffer_scaled_correction_norm": buffer_norm,
        "maximum_active_scaled_correction": (
            maximum_active_scaled_correction
        ),
        "maximum_buffer_scaled_correction": (
            maximum_buffer_scaled_correction
        ),
        "corrected_primitives_sha256": _array_sha256(corrected),
    }
    report["passed"] = bool(
        report["optimizer_success"]
        and report["maximum_coordinate_defect"]
        <= MAXIMUM_ANCHOR_COORDINATE_DEFECT
        and report["maximum_active_scaled_correction"]
        <= MAXIMUM_ACTIVE_SCALED_ANCHOR_CORRECTION
        and report["maximum_buffer_scaled_correction"]
        <= MAXIMUM_BUFFER_SCALED_ANCHOR_CORRECTION
        and report["constraint_condition"]
        <= MAXIMUM_CONSTRAINT_CONDITION
    )
    return corrected, report


def _exact_local_pair(
    *,
    context,
    primitives: np.ndarray,
    profiles: dict,
    reduced: dict,
    level,
    shell_edges_rg: np.ndarray,
    active_outer_rg: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    scales = np.asarray(
        reduced["primitive_column_scales"],
        dtype=float,
    )
    seed_physical = np.asarray(
        profiles["matched_half_difference"],
        dtype=float,
    ).copy()
    radius = (
        np.asarray(context.grid.centers, dtype=float)
        / context.grid.gravitational_radius
    )
    discarded = radius > active_outer_rg * (1.0 + 2.0e-14)
    original_norm = wp10c8v._continuum_norm(
        seed_physical,
        context.grid.cell_measures,
    )
    seed_physical[discarded] = 0.0
    retained_norm = wp10c8v._continuum_norm(
        seed_physical,
        context.grid.cell_measures,
    )
    pair = causal_exact_equal_coordinate_lift_pair(
        base_primitive_vector=np.asarray(primitives, dtype=float).ravel(),
        primitive_column_scales=scales,
        state_weights=_continuum_weights(
            context,
            active_outer_rg=active_outer_rg,
        ),
        physical_input_amplitudes=np.asarray(
            profiles["physical_input_amplitudes"],
            dtype=float,
        ).ravel(),
        target_coordinate_values=np.asarray(
            level.coordinate_values,
            dtype=float,
        ),
        target_coordinate_scales=np.asarray(
            level.coordinate_scales,
            dtype=float,
        ),
        constraint_matrix=np.asarray(
            level.constraint_matrix,
            dtype=float,
        ),
        seed_direction=seed_physical.ravel() / scales,
        seed_multiplier=1.0,
        coordinate_evaluator=_moment_evaluator(
            context,
            shell_edges_rg,
        ),
    )
    minus = np.asarray(pair.minus.primitive_vector).reshape(-1, 5)
    plus = np.asarray(pair.plus.primitive_vector).reshape(-1, 5)
    minus_state = causal_five_field_state_from_primitives(context, minus)
    plus_state = causal_five_field_state_from_primitives(context, plus)
    minus_gates = _audit_local_state_gates(
        context,
        pack_causal_five_field_state(minus_state),
    )
    plus_gates = _audit_local_state_gates(
        context,
        pack_causal_five_field_state(plus_state),
    )
    report = {
        "maximum_pairwise_coordinate_defect": (
            pair.maximum_pairwise_coordinate_defect
        ),
        "constraint_rank": pair.normal_basis.numerical_rank,
        "constraint_condition": pair.normal_basis.condition_estimate,
        "seed_active_norm_fraction": retained_norm
        / max(original_norm, np.finfo(float).tiny),
        "minus": {
            "optimizer_success": pair.minus.optimizer_success,
            "maximum_coordinate_defect": (
                pair.minus.maximum_coordinate_defect
            ),
            "correction_fraction": pair.minus.correction_fraction,
            "weighted_direction_cosine": (
                pair.minus.weighted_direction_cosine
            ),
            "maximum_pointwise_amplitude_ratio": (
                pair.minus.maximum_pointwise_amplitude_ratio
            ),
            "state_gates_passed": minus_gates["passed"],
            "state_gates": minus_gates,
        },
        "plus": {
            "optimizer_success": pair.plus.optimizer_success,
            "maximum_coordinate_defect": (
                pair.plus.maximum_coordinate_defect
            ),
            "correction_fraction": pair.plus.correction_fraction,
            "weighted_direction_cosine": (
                pair.plus.weighted_direction_cosine
            ),
            "maximum_pointwise_amplitude_ratio": (
                pair.plus.maximum_pointwise_amplitude_ratio
            ),
            "state_gates_passed": plus_gates["passed"],
            "state_gates": plus_gates,
        },
    }
    report["passed"] = bool(
        report["maximum_pairwise_coordinate_defect"]
        <= MAXIMUM_PAIR_COORDINATE_DEFECT
        and report["constraint_condition"]
        <= MAXIMUM_CONSTRAINT_CONDITION
        and all(
            report[side]["optimizer_success"]
            and report[side]["maximum_coordinate_defect"]
            <= MAXIMUM_ANCHOR_COORDINATE_DEFECT
            and report[side]["state_gates_passed"]
            for side in ("minus", "plus")
        )
    )
    return minus, plus, report


def _operator_contract(
    label: str,
    *,
    context,
    primitives: np.ndarray,
    minus: np.ndarray,
    plus: np.ndarray,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "base_commit": BASE_COMMIT,
        "label": label,
        "n_cells": int(context.grid.centers.size),
        "inner_edge_rg": float(
            context.grid.edges[0] / context.grid.gravitational_radius
        ),
        "outer_edge_rg": float(
            context.grid.edges[-1] / context.grid.gravitational_radius
        ),
        "inner_boundary_trace_override": (
            context.inner_boundary_trace_override
        ),
        "primitives_sha256": _array_sha256(primitives),
        "minus_sha256": _array_sha256(minus),
        "plus_sha256": _array_sha256(plus),
        "core_dae_sha256": _sha256(ROOT / CORE_DAE_FILE),
        "core_fiber_sha256": _sha256(ROOT / CORE_FIBER_FILE),
        "core_tangent_sha256": _sha256(ROOT / CORE_TANGENT_FILE),
    }


def _build_or_load_operator(
    label: str,
    *,
    context,
    primitives: np.ndarray,
    minus: np.ndarray,
    plus: np.ndarray,
    amplitudes: np.ndarray,
    normalization: float | None,
    reduced: dict | None = None,
    force: bool,
) -> tuple[dict, dict[str, np.ndarray]]:
    contract = _operator_contract(
        label,
        context=context,
        primitives=primitives,
        minus=minus,
        plus=plus,
    )
    json_path, arrays_path = _operator_paths(label)
    if not force and json_path.exists() and arrays_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if (
            all(payload.get(key) == value for key, value in contract.items())
            and payload.get("arrays_sha256") == _sha256(arrays_path)
        ):
            cached_state = causal_five_field_state_from_primitives(
                context,
                primitives,
            )
            gates = _audit_local_state_gates(
                context,
                pack_causal_five_field_state(cached_state),
            )
            payload["state_gates"] = gates
            payload["state_gates_passed"] = gates["passed"]
            payload["passed"] = bool(
                payload["rate_source"] == "descriptor_balance"
                and payload["state_gates_passed"]
                and payload["inner_incoming_characteristics"] == 0
                and payload[
                    "maximum_scaled_generator_factorization_defect"
                ]
                <= MAXIMUM_GENERATOR_FACTORIZATION_DEFECT
                and payload["maximum_relative_storage_action_defect"]
                <= MAXIMUM_STORAGE_ACTION_DEFECT
            )
            payload["producer_runner_sha256"] = _sha256(
                ROOT / THIS_RUNNER
            )
            json_path.write_text(
                json.dumps(
                    _plain(payload),
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            return payload, wp10c8v._load_npz(arrays_path)

    started = time.perf_counter()
    state = causal_five_field_state_from_primitives(context, primitives)
    vector = pack_causal_five_field_state(state)
    if reduced is None:
        reduced = causal_five_field_reduced_descriptor_matrices(
            context,
            vector,
            finite_difference_step=wp10c8v.FINITE_DIFFERENCE_STEP,
            descriptor_timestep_seconds=(
                wp10c8v.DESCRIPTOR_TIMESTEP_SECONDS
            ),
        )
    evolving = causal_five_field_evolving_tangent_matrices(
        context,
        vector,
        primitive_rate_per_s=None,
        reduced_descriptor=reduced,
        finite_difference_step=wp10c8v.FINITE_DIFFERENCE_STEP,
        descriptor_timestep_seconds=(
            wp10c8v.DESCRIPTOR_TIMESTEP_SECONDS
        ),
        storage_difference_step=wp10c8v.STORAGE_DIFFERENCE_STEP,
        storage_rate_derivative_step=(
            wp10c8v.STORAGE_RATE_DERIVATIVE_STEP
        ),
        storage_quadrature_order=wp10c8v.STORAGE_QUADRATURE_ORDER,
        storage_directional_step=wp10c8v.STORAGE_DIRECTIONAL_STEP,
    )
    half_difference = 0.5 * (
        np.asarray(plus, dtype=float) - np.asarray(minus, dtype=float)
    )
    scales = np.asarray(
        evolving["primitive_column_scales"],
        dtype=float,
    ).reshape(-1, 5)
    initial = half_difference / np.asarray(amplitudes, dtype=float)
    if normalization is None:
        normalization = wp10c8v._continuum_norm(
            initial,
            context.grid.cell_measures,
        )
    if not np.isfinite(normalization) or normalization <= 0.0:
        raise RuntimeError("WP10c8w initial normalization is invalid")
    arrays = {
        "generator": np.asarray(
            evolving["evolving_scaled_generator_per_s"],
            dtype=float,
        ),
        "descriptor": np.asarray(
            evolving["descriptor_reduced_scaled_matrix"],
            dtype=float,
        ),
        "stationary_jacobian": np.asarray(
            evolving["stationary_reduced_scaled_jacobian"],
            dtype=float,
        ),
        "storage_rate_derivative": np.asarray(
            evolving["storage_rate_derivative_scaled_matrix"],
            dtype=float,
        ),
        "primitive_column_scales": np.asarray(scales, dtype=float),
        "conservation_row_scales": np.asarray(
            evolving["conservation_row_scales"],
            dtype=float,
        ),
        "base_primitives": np.asarray(primitives, dtype=float),
        "base_physical_rate_per_s": np.asarray(
            evolving["primitive_rate_per_s"],
            dtype=float,
        ),
        "matched_half_difference": half_difference,
        "physical_input_amplitudes": np.asarray(amplitudes, dtype=float),
        "frozen_exterior_chart": np.asarray(
            context.outer_boundary_frozen_exterior_chart,
            dtype=float,
        ),
        "radius_rg": np.asarray(
            context.grid.centers / context.grid.gravitational_radius,
            dtype=float,
        ),
        "grid_edges_rg": np.asarray(
            context.grid.edges / context.grid.gravitational_radius,
            dtype=float,
        ),
        "cell_measures": np.asarray(
            context.grid.cell_measures,
            dtype=float,
        ),
        "initial_normalization": np.asarray(normalization, dtype=float),
    }
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **arrays)
    gates = _audit_local_state_gates(context, vector)
    payload = {
        **contract,
        "producer_runner": THIS_RUNNER,
        "producer_runner_sha256": _sha256(ROOT / THIS_RUNNER),
        "arrays_path": _relative(arrays_path),
        "arrays_sha256": _sha256(arrays_path),
        "rate_source": evolving["rate_source"],
        "state_gates_passed": gates["passed"],
        "state_gates": gates,
        "inner_incoming_characteristics": gates["measured"][
            "inner_incoming_characteristics"
        ],
        "maximum_scaled_generator_factorization_defect": float(
            evolving["maximum_scaled_generator_factorization_defect"]
        ),
        "maximum_relative_storage_action_defect": float(
            evolving["maximum_relative_storage_action_defect"]
        ),
        "wall_seconds": time.perf_counter() - started,
    }
    payload["passed"] = bool(
        payload["rate_source"] == "descriptor_balance"
        and payload["state_gates_passed"]
        and payload["inner_incoming_characteristics"] == 0
        and payload["maximum_scaled_generator_factorization_defect"]
        <= MAXIMUM_GENERATOR_FACTORIZATION_DEFECT
        and payload["maximum_relative_storage_action_defect"]
        <= MAXIMUM_STORAGE_ACTION_DEFECT
    )
    json_path.write_text(
        json.dumps(
            _plain(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return payload, arrays


def _propagate(
    arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    scales = np.asarray(
        arrays["primitive_column_scales"],
        dtype=float,
    ).reshape(-1, 5)
    amplitudes = np.asarray(
        arrays["physical_input_amplitudes"],
        dtype=float,
    )
    generator = wp10c8v._similarity_rescale_generator(
        arrays["generator"],
        scales,
        amplitudes,
    )
    initial = (
        np.asarray(arrays["matched_half_difference"], dtype=float)
        / amplitudes
        / float(arrays["initial_normalization"])
    )
    times = np.linspace(0.0, TARGET_SECONDS, TIME_SAMPLES)
    state = np.asarray(
        scipy.sparse.linalg.expm_multiply(
            generator,
            initial.ravel(),
            start=0.0,
            stop=TARGET_SECONDS,
            num=TIME_SAMPLES,
            endpoint=True,
        ),
        dtype=float,
    ).reshape(TIME_SAMPLES, -1, 5)
    rate = np.asarray(
        [generator @ row.ravel() for row in state],
        dtype=float,
    ).reshape(state.shape)
    radius = np.asarray(arrays["radius_rg"], dtype=float)
    active_outer = int(
        np.count_nonzero(
            radius <= _ACTIVE_OUTER_RG * (1.0 + 2.0e-14)
        )
    )
    weights = np.asarray(arrays["cell_measures"][:active_outer], dtype=float)
    weights /= np.sum(weights)
    stress_rate = np.sum(
        weights[None, :] * rate[:, :active_outer, 4],
        axis=1,
    )
    return {
        "times": times,
        "state": state,
        "rate": rate,
        "stress_rate_signal": stress_rate,
    }


def _generator_propagation_safety(
    arrays: dict[str, np.ndarray],
) -> dict:
    """Reject explosive audit operators before a matrix exponential is run."""

    scales = np.asarray(
        arrays["primitive_column_scales"],
        dtype=float,
    ).reshape(-1, 5)
    amplitudes = np.asarray(
        arrays["physical_input_amplitudes"],
        dtype=float,
    )
    generator = wp10c8v._similarity_rescale_generator(
        arrays["generator"],
        scales,
        amplitudes,
    )
    eigenvalues = np.linalg.eigvals(generator)
    spectral_abscissa = float(np.max(np.real(eigenvalues)))
    growth_exponent = float(spectral_abscissa * TARGET_SECONDS)
    passed = bool(
        np.isfinite(growth_exponent)
        and growth_exponent <= MAXIMUM_PROPAGATION_GROWTH_EXPONENT
    )
    return {
        "passed": passed,
        "spectral_abscissa_per_s": spectral_abscissa,
        "target_growth_exponent": growth_exponent,
        "maximum_eigenvalue_magnitude_per_s": float(
            np.max(np.abs(eigenvalues))
        ),
        "generator_infinity_norm_per_s": float(
            np.linalg.norm(generator, ord=np.inf)
        ),
        "maximum_target_growth_exponent": (
            MAXIMUM_PROPAGATION_GROWTH_EXPONENT
        ),
    }


def _pair_metrics(
    coarse: dict[str, np.ndarray],
    fine: dict[str, np.ndarray],
    coarse_arrays: dict[str, np.ndarray],
    fine_arrays: dict[str, np.ndarray],
    *,
    lower_rg: float | None = None,
) -> dict:
    coarse_edges = np.asarray(coarse_arrays["grid_edges_rg"], dtype=float)
    fine_edges = np.asarray(fine_arrays["grid_edges_rg"], dtype=float)
    if (
        fine_edges.size != 2 * (coarse_edges.size - 1) + 1
        or not np.allclose(
            fine_edges[::2],
            coarse_edges,
            rtol=0.0,
            atol=2.0e-12,
        )
    ):
        raise RuntimeError("WP10c8w comparison grids are not nested")
    restricted_state = wp10c8v._restrict_pairwise(
        fine["state"],
        fine_arrays["cell_measures"],
    )
    restricted_rate = wp10c8v._restrict_pairwise(
        fine["rate"],
        fine_arrays["cell_measures"],
    )
    radius = np.asarray(coarse_arrays["radius_rg"], dtype=float)
    mask = np.ones(radius.size, dtype=bool)
    if lower_rg is not None:
        mask &= radius >= float(lower_rg)
    mask &= radius <= _ACTIVE_OUTER_RG * (1.0 + 2.0e-14)
    weights = np.asarray(
        coarse_arrays["cell_measures"],
        dtype=float,
    )[mask]
    weights /= np.sum(weights)

    def metrics(first: np.ndarray, second: np.ndarray) -> dict:
        first = np.asarray(first, dtype=float)[:, mask]
        second = np.asarray(second, dtype=float)[:, mask]
        difference = second - first
        first_norm = np.sqrt(
            np.sum(weights[None, :, None] * first**2, axis=(1, 2))
        )
        second_norm = np.sqrt(
            np.sum(weights[None, :, None] * second**2, axis=(1, 2))
        )
        difference_norm = np.sqrt(
            np.sum(
                weights[None, :, None] * difference**2,
                axis=(1, 2),
            )
        )
        cosine = np.sum(
            weights[None, :, None] * first * second,
            axis=(1, 2),
        ) / np.maximum(
            first_norm * second_norm,
            np.finfo(float).tiny,
        )
        relative = difference_norm / np.maximum(
            first_norm,
            np.finfo(float).tiny,
        )
        return {
            "maximum_relative_l2_difference": float(np.max(relative)),
            "final_relative_l2_difference": float(relative[-1]),
            "minimum_signed_cosine": float(np.min(cosine)),
            "final_signed_cosine": float(cosine[-1]),
            "final_amplitude_ratio": float(
                second_norm[-1]
                / max(first_norm[-1], np.finfo(float).tiny)
            ),
        }

    return {
        "state": metrics(coarse["state"], restricted_state),
        "rate": metrics(coarse["rate"], restricted_rate),
    }


def _same_mesh_placement_metrics(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    reference_arrays: dict[str, np.ndarray],
    candidate_arrays: dict[str, np.ndarray],
) -> dict:
    reference_radius = np.asarray(
        reference_arrays["radius_rg"],
        dtype=float,
    )
    candidate_radius = np.asarray(
        candidate_arrays["radius_rg"],
        dtype=float,
    )
    offset = reference_radius.size - candidate_radius.size
    if (
        offset < 0
        or not np.allclose(
            reference_radius[offset:],
            candidate_radius,
            rtol=0.0,
            atol=2.0e-12,
        )
    ):
        raise RuntimeError("WP10c8w placement grids do not share a lattice")
    mask = (
        candidate_radius >= COMMON_EXTERIOR_INNER_RG
    ) & (
        candidate_radius
        <= float(
            _ACTIVE_OUTER_RG
        )
        * (1.0 + 2.0e-14)
    )
    weights = np.asarray(
        candidate_arrays["cell_measures"],
        dtype=float,
    )[mask]
    weights /= np.sum(weights)

    def metrics(name: str) -> dict:
        first = np.asarray(reference[name], dtype=float)[:, offset:][:, mask]
        second = np.asarray(candidate[name], dtype=float)[:, mask]
        difference = second - first
        first_norm = np.sqrt(
            np.sum(weights[None, :, None] * first**2, axis=(1, 2))
        )
        second_norm = np.sqrt(
            np.sum(weights[None, :, None] * second**2, axis=(1, 2))
        )
        difference_norm = np.sqrt(
            np.sum(
                weights[None, :, None] * difference**2,
                axis=(1, 2),
            )
        )
        cosine = np.sum(
            weights[None, :, None] * first * second,
            axis=(1, 2),
        ) / np.maximum(
            first_norm * second_norm,
            np.finfo(float).tiny,
        )
        relative = difference_norm / np.maximum(
            first_norm,
            np.finfo(float).tiny,
        )
        return {
            "maximum_relative_l2_defect": float(np.max(relative)),
            "minimum_signed_cosine": float(np.min(cosine)),
            "final_relative_l2_defect": float(relative[-1]),
            "final_signed_cosine": float(cosine[-1]),
        }

    return {"state": metrics("state"), "rate": metrics("rate")}


def _slice_context_and_state(
    *,
    base_context,
    base_arrays: dict[str, np.ndarray],
    minus: np.ndarray,
    plus: np.ndarray,
    offset: int,
    trace_mode: str,
) -> tuple[object, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = int(base_context.grid.centers.size)
    offset = int(offset)
    if not 0 <= offset < count - 3:
        raise ValueError("WP10c8w excision offset is invalid")
    new_count = count - offset
    grid = make_kerr_schild_column_grid(
        float(base_context.grid.edges[offset]),
        float(base_context.grid.edges[-1]),
        new_count,
        float(base_context.grid.gravitational_radius),
    )
    if not np.allclose(
        grid.edges,
        base_context.grid.edges[offset:],
        rtol=2.0e-14,
        atol=0.0,
    ):
        raise RuntimeError("WP10c8w shifted grid lost its common lattice")
    context = replace(
        base_context,
        grid=grid,
        stream_sources=_zero_sources(new_count),
        inner_boundary_trace_override=trace_mode,
    ).validated()
    return (
        context,
        np.asarray(base_arrays["base_primitives"], dtype=float)[offset:],
        np.asarray(minus, dtype=float)[offset:],
        np.asarray(plus, dtype=float)[offset:],
        np.asarray(
            base_arrays["physical_input_amplitudes"],
            dtype=float,
        )[offset:],
    )


_ACTIVE_OUTER_RG = float("nan")


def run(*, force: bool = False) -> tuple[dict, dict[str, np.ndarray]]:
    global _ACTIVE_OUTER_RG
    started = time.perf_counter()
    parents = {
        mesh: wp10c8v._parent_bundle(mesh)
        for mesh in PARENT_MESHES
    }
    base_profiles = {
        mesh: wp10c8v._base_profiles(mesh, parents)
        for mesh in MESHES
    }
    contexts = {
        mesh: wp10c8v._local_context(
            parents[128]["context"],
            base_profiles[mesh],
        )
        for mesh in MESHES
    }
    _ACTIVE_OUTER_RG = float(
        base_profiles[64]["grid"].edges[
            wp10c8v._active_cell_count(64)
        ]
        / contexts[64].grid.gravitational_radius
    )
    parent_operator = wp10c8v._load_npz(parents[128]["operator_path"])
    shell_zero_outer_rg = float(parent_operator["shell_edges_rg"][1])

    target_shell_edges = _local_shell_edges_rg(
        contexts[128],
        shell_zero_outer_rg,
    )
    target_level = _moment_values(
        contexts[128],
        base_profiles[128]["primitives"],
        target_shell_edges,
    )
    target_values = np.asarray(
        target_level.coordinate_values,
        dtype=float,
    )
    target_scales = np.asarray(
        target_level.coordinate_scales,
        dtype=float,
    )

    anchor_primitives = {
        64: np.asarray(base_profiles[64]["primitives"], dtype=float),
        128: np.asarray(base_profiles[128]["primitives"], dtype=float),
    }
    projection_report = {
        "N64": {"required": False, "passed": True},
        "N128": {"required": False, "passed": True},
    }
    fine_edges = _local_shell_edges_rg(
        contexts[256],
        shell_zero_outer_rg,
    )
    print("WP10c8w: projecting the N256 local anchor", flush=True)
    anchor_primitives[256], projection_report["N256"] = (
        _project_fine_anchor(
            context=contexts[256],
            profiles=base_profiles[256],
            shell_edges_rg=fine_edges,
            target_values=target_values,
            target_scales=target_scales,
            active_outer_rg=_ACTIVE_OUTER_RG,
        )
    )

    anchor_rows = {}
    anchor_arrays = {}
    pair_primitives = {}
    for mesh in MESHES:
        print(
            f"WP10c8w: building exact local N{mesh} pair",
            flush=True,
        )
        shell_edges = _local_shell_edges_rg(
            contexts[mesh],
            shell_zero_outer_rg,
        )
        (
            _state,
            _vector,
            reduced,
            level,
        ) = _reduced_and_ladder(
            contexts[mesh],
            anchor_primitives[mesh],
            shell_edges,
        )
        minus, plus, pair_report = _exact_local_pair(
            context=contexts[mesh],
            primitives=anchor_primitives[mesh],
            profiles=base_profiles[mesh],
            reduced=reduced,
            level=level,
            shell_edges_rg=shell_edges,
            active_outer_rg=_ACTIVE_OUTER_RG,
        )
        pair_primitives[mesh] = (minus, plus)
        label = f"N{mesh:03d}_anchor_inherit"
        print(f"WP10c8w: building/loading {label}", flush=True)
        operator_row, arrays = _build_or_load_operator(
            label,
            context=contexts[mesh],
            primitives=anchor_primitives[mesh],
            minus=minus,
            plus=plus,
            amplitudes=base_profiles[mesh][
                "physical_input_amplitudes"
            ],
            normalization=None,
            reduced=reduced,
            force=force,
        )
        anchor_rows[mesh] = {
            "projection": projection_report[f"N{mesh}"],
            "pair": pair_report,
            "operator": operator_row,
            "propagation_safety": _generator_propagation_safety(arrays),
        }
        anchor_arrays[mesh] = arrays

    if not all(
        row["propagation_safety"]["passed"]
        for row in anchor_rows.values()
    ):
        raise RuntimeError(
            "WP10c8w inherited anchor operator failed propagation safety"
        )
    propagated_anchor = {
        mesh: _propagate(anchor_arrays[mesh]) for mesh in MESHES
    }
    anchor_pair_metrics = {
        "N64_N128": _pair_metrics(
            propagated_anchor[64],
            propagated_anchor[128],
            anchor_arrays[64],
            anchor_arrays[128],
        ),
        "N128_N256": _pair_metrics(
            propagated_anchor[128],
            propagated_anchor[256],
            anchor_arrays[128],
            anchor_arrays[256],
        ),
    }

    trace_rows = {"inherit": anchor_rows}
    trace_arrays = {"inherit": anchor_arrays}
    trace_histories = {"inherit": propagated_anchor}
    trace_metrics = {
        "inherit": {
            "available": True,
            "pair_metrics": _pair_metrics(
                propagated_anchor[64],
                propagated_anchor[128],
                anchor_arrays[64],
                anchor_arrays[128],
                lower_rg=COMMON_EXTERIOR_INNER_RG,
            ),
        }
    }
    for trace_mode in TRACE_MODES[1:]:
        rows = {}
        arrays_by_mesh = {}
        histories = {}
        for mesh in PARENT_MESHES:
            context = replace(
                contexts[mesh],
                inner_boundary_trace_override=trace_mode,
            ).validated()
            minus, plus = pair_primitives[mesh]
            label = f"N{mesh:03d}_anchor_{trace_mode}"
            print(f"WP10c8w: building/loading {label}", flush=True)
            row, arrays = _build_or_load_operator(
                label,
                context=context,
                primitives=anchor_primitives[mesh],
                minus=minus,
                plus=plus,
                amplitudes=base_profiles[mesh][
                    "physical_input_amplitudes"
                ],
                normalization=float(
                    anchor_arrays[mesh]["initial_normalization"]
                ),
                force=force,
            )
            safety = _generator_propagation_safety(arrays)
            rows[mesh] = {
                "operator": row,
                "propagation_safety": safety,
            }
            arrays_by_mesh[mesh] = arrays
            if not safety["passed"]:
                print(
                    "WP10c8w: rejecting "
                    f"{trace_mode} at N{mesh} before propagation "
                    f"(growth exponent={safety['target_growth_exponent']:.6e})",
                    flush=True,
                )
                break
            histories[mesh] = _propagate(arrays)
        trace_rows[trace_mode] = rows
        trace_arrays[trace_mode] = arrays_by_mesh
        trace_histories[trace_mode] = histories
        if set(histories) == set(PARENT_MESHES):
            trace_metrics[trace_mode] = {
                "available": True,
                "pair_metrics": _pair_metrics(
                    histories[64],
                    histories[128],
                    arrays_by_mesh[64],
                    arrays_by_mesh[128],
                    lower_rg=COMMON_EXTERIOR_INNER_RG,
                ),
            }
        else:
            trace_metrics[trace_mode] = {
                "available": False,
                "reason": "propagation_safety_failed",
            }

    def trace_score(mode: str) -> float:
        if not trace_metrics[mode]["available"]:
            return float(np.finfo(float).max)
        metrics = trace_metrics[mode]["pair_metrics"]
        relative = max(
            metrics["state"]["maximum_relative_l2_difference"],
            metrics["rate"]["maximum_relative_l2_difference"],
        )
        cosine = min(
            metrics["state"]["minimum_signed_cosine"],
            metrics["rate"]["minimum_signed_cosine"],
        )
        return float(relative + 10.0 * max(0.0, MINIMUM_SIGNED_COSINE - cosine))

    selected_trace = min(TRACE_MODES, key=trace_score)
    if selected_trace != "inherit":
        context = replace(
            contexts[256],
            inner_boundary_trace_override=selected_trace,
        ).validated()
        minus, plus = pair_primitives[256]
        label = f"N256_anchor_{selected_trace}"
        print(f"WP10c8w: building/loading {label}", flush=True)
        row, arrays = _build_or_load_operator(
            label,
            context=context,
            primitives=anchor_primitives[256],
            minus=minus,
            plus=plus,
            amplitudes=base_profiles[256]["physical_input_amplitudes"],
            normalization=float(
                anchor_arrays[256]["initial_normalization"]
            ),
            force=force,
        )
        safety = _generator_propagation_safety(arrays)
        trace_rows[selected_trace][256] = {
            "operator": row,
            "propagation_safety": safety,
        }
        trace_arrays[selected_trace][256] = arrays
        if safety["passed"]:
            trace_histories[selected_trace][256] = _propagate(arrays)
        else:
            print(
                "WP10c8w: selected alternate trace failed N256 "
                "propagation safety; falling back to inherit",
                flush=True,
            )
            selected_trace = "inherit"
    selected_histories = trace_histories[selected_trace]
    selected_arrays = trace_arrays[selected_trace]
    selected_cross_metrics = {
        "N64_N128": _pair_metrics(
            selected_histories[64],
            selected_histories[128],
            selected_arrays[64],
            selected_arrays[128],
            lower_rg=COMMON_EXTERIOR_INNER_RG,
        ),
        "N128_N256": _pair_metrics(
            selected_histories[128],
            selected_histories[256],
            selected_arrays[128],
            selected_arrays[256],
            lower_rg=COMMON_EXTERIOR_INNER_RG,
        ),
    }
    state_coarse = selected_cross_metrics["N64_N128"]["state"][
        "maximum_relative_l2_difference"
    ]
    state_fine = selected_cross_metrics["N128_N256"]["state"][
        "maximum_relative_l2_difference"
    ]
    rate_coarse = selected_cross_metrics["N64_N128"]["rate"][
        "maximum_relative_l2_difference"
    ]
    rate_fine = selected_cross_metrics["N128_N256"]["rate"][
        "maximum_relative_l2_difference"
    ]
    state_order = float(
        np.log2(
            max(state_coarse, np.finfo(float).tiny)
            / max(state_fine, np.finfo(float).tiny)
        )
    )
    rate_order = float(
        np.log2(
            max(rate_coarse, np.finfo(float).tiny)
            / max(rate_fine, np.finfo(float).tiny)
        )
    )
    selected_signals = {
        mesh: wp10c8v._signal_diagnostics(
            selected_histories[mesh]["times"],
            selected_histories[mesh]["stress_rate_signal"],
        )
        for mesh in MESHES
    }
    selected_signal_pairs = {
        "N64_N128": wp10c8v._signal_pair_metrics(
            selected_signals[64],
            selected_signals[128],
        ),
        "N128_N256": wp10c8v._signal_pair_metrics(
            selected_signals[128],
            selected_signals[256],
        ),
    }

    placement_rows = {}
    placement_arrays = {}
    placement_histories = {}
    placement_cross_metrics = {}
    placement_sensitivity = {}
    for placement, offsets in PLACEMENT_OFFSETS.items():
        if placement == "original":
            rows = {
                mesh: {
                    "operator": trace_rows[selected_trace][mesh][
                        "operator"
                    ],
                    "propagation_safety": (
                        trace_rows[selected_trace][mesh][
                            "propagation_safety"
                        ]
                    ),
                }
                for mesh in (128, 256)
            }
            arrays_by_mesh = {
                mesh: selected_arrays[mesh] for mesh in (128, 256)
            }
            histories = {
                mesh: selected_histories[mesh] for mesh in (128, 256)
            }
        else:
            rows = {}
            arrays_by_mesh = {}
            histories = {}
            for mesh in (128, 256):
                base_context = replace(
                    contexts[mesh],
                    inner_boundary_trace_override=selected_trace,
                ).validated()
                minus, plus = pair_primitives[mesh]
                (
                    context,
                    primitives,
                    shifted_minus,
                    shifted_plus,
                    amplitudes,
                ) = _slice_context_and_state(
                    base_context=base_context,
                    base_arrays=selected_arrays[mesh],
                    minus=minus,
                    plus=plus,
                    offset=offsets[mesh],
                    trace_mode=selected_trace,
                )
                label = (
                    f"N{mesh:03d}_{selected_trace}_"
                    f"edge_offset_{offsets[mesh]:02d}"
                )
                print(f"WP10c8w: building/loading {label}", flush=True)
                row, arrays = _build_or_load_operator(
                    label,
                    context=context,
                    primitives=primitives,
                    minus=shifted_minus,
                    plus=shifted_plus,
                    amplitudes=amplitudes,
                    normalization=float(
                        selected_arrays[mesh]["initial_normalization"]
                    ),
                    force=force,
                )
                safety = _generator_propagation_safety(arrays)
                rows[mesh] = {
                    "operator": row,
                    "propagation_safety": safety,
                }
                arrays_by_mesh[mesh] = arrays
                if not safety["passed"]:
                    print(
                        "WP10c8w: rejecting excision placement "
                        f"{placement} at N{mesh} before propagation "
                        "(growth exponent="
                        f"{safety['target_growth_exponent']:.6e})",
                        flush=True,
                    )
                    break
                histories[mesh] = _propagate(arrays)
        placement_rows[placement] = rows
        placement_arrays[placement] = arrays_by_mesh
        placement_histories[placement] = histories
        available = set(histories) == {128, 256}
        if available:
            placement_cross_metrics[placement] = {
                "available": True,
                "pair_metrics": _pair_metrics(
                    histories[128],
                    histories[256],
                    arrays_by_mesh[128],
                    arrays_by_mesh[256],
                    lower_rg=COMMON_EXTERIOR_INNER_RG,
                ),
            }
        else:
            placement_cross_metrics[placement] = {
                "available": False,
                "reason": "propagation_safety_failed",
            }
        if available and set(placement_histories["original"]) == {128, 256}:
            placement_sensitivity[placement] = {
                "available": True,
                "by_mesh": {
                    f"N{mesh}": _same_mesh_placement_metrics(
                        placement_histories["original"][mesh],
                        histories[mesh],
                        placement_arrays["original"][mesh],
                        arrays_by_mesh[mesh],
                    )
                    for mesh in (128, 256)
                },
            }
        else:
            placement_sensitivity[placement] = {
                "available": False,
                "reason": "propagation_safety_failed",
            }

    fine_signal = selected_signal_pairs["N128_N256"]
    signal_passed = bool(
        fine_signal["maximum_zero_crossing_relative_defect"] is not None
        and fine_signal["frequency_relative_defect"] is not None
        and fine_signal["damping_relative_defect"] is not None
        and fine_signal["maximum_zero_crossing_relative_defect"]
        <= MAXIMUM_ZERO_CROSSING_DEFECT
        and fine_signal["frequency_relative_defect"]
        <= MAXIMUM_FREQUENCY_DEFECT
        and fine_signal["damping_relative_defect"]
        <= MAXIMUM_DAMPING_DEFECT
    )
    selected_minimum_cosine = min(
        selected_cross_metrics["N128_N256"]["state"][
            "minimum_signed_cosine"
        ],
        selected_cross_metrics["N128_N256"]["rate"][
            "minimum_signed_cosine"
        ],
    )
    refinement_passed = bool(
        state_order >= MINIMUM_SPATIAL_ORDER
        and rate_order >= MINIMUM_SPATIAL_ORDER
        and selected_minimum_cosine >= MINIMUM_SIGNED_COSINE
        and signal_passed
    )
    placement_passed = all(
        result["available"]
        and all(
            max(
                metrics[kind]["maximum_relative_l2_defect"]
                for kind in ("state", "rate")
            )
            <= MAXIMUM_EXTERIOR_PLACEMENT_HISTORY_DEFECT
            and min(
                metrics[kind]["minimum_signed_cosine"]
                for kind in ("state", "rate")
            )
            >= MINIMUM_SIGNED_COSINE
            for metrics in result["by_mesh"].values()
        )
        for placement, result in placement_sensitivity.items()
        if placement != "original"
    )
    anchor_passed = bool(
        all(row["projection"]["passed"] for row in anchor_rows.values())
        and all(row["pair"]["passed"] for row in anchor_rows.values())
        and all(row["operator"]["passed"] for row in anchor_rows.values())
        and all(
            row["propagation_safety"]["passed"]
            for row in anchor_rows.values()
        )
    )
    n512_authorized = bool(
        anchor_passed and refinement_passed and placement_passed
    )
    classification = (
        "independent_anchor_and_excision_phase_preflight_passed"
        if n512_authorized
        else (
            "independent_anchor_passed_excision_or_phase_unresolved"
            if anchor_passed
            else "independent_anchor_consistency_failed"
        )
    )

    arrays_out = {
        "times": propagated_anchor[64]["times"],
    }
    for mesh in MESHES:
        arrays_out[f"N{mesh}_anchor_state"] = propagated_anchor[mesh][
            "state"
        ]
        arrays_out[f"N{mesh}_anchor_rate"] = propagated_anchor[mesh][
            "rate"
        ]
        arrays_out[f"N{mesh}_anchor_primitives"] = anchor_primitives[mesh]
        arrays_out[f"N{mesh}_pair_minus"] = pair_primitives[mesh][0]
        arrays_out[f"N{mesh}_pair_plus"] = pair_primitives[mesh][1]
    for placement, histories in placement_histories.items():
        for mesh in sorted(histories):
            prefix = f"{placement}_N{mesh}"
            arrays_out[f"{prefix}_radius_rg"] = placement_arrays[
                placement
            ][mesh]["radius_rg"]
            arrays_out[f"{prefix}_state"] = histories[mesh]["state"]
            arrays_out[f"{prefix}_rate"] = histories[mesh]["rate"]

    DEFAULT_ARRAYS.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DEFAULT_ARRAYS, **arrays_out)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "base_commit": BASE_COMMIT,
        "purpose": (
            "separate fine-anchor inconsistency from near-horizon "
            "trace and excision-placement phase error"
        ),
        "classification": classification,
        "scope": {
            "local_frozen_linear_audit": True,
            "formal_fast_average_certified": False,
            "nonlinear_n256_truth_certified": False,
            "production_inner_boundary_changed": False,
            "reduced_architecture_selected": False,
            "common_exterior_inner_rg": COMMON_EXTERIOR_INNER_RG,
            "active_outer_rg": _ACTIVE_OUTER_RG,
        },
        "anchor_consistency": {
            f"N{mesh}": anchor_rows[mesh] for mesh in MESHES
        },
        "anchor_pairwise_history": anchor_pair_metrics,
        "trace_screen": {
            "selected_trace": selected_trace,
            "scores": {
                mode: trace_score(mode) for mode in TRACE_MODES
            },
            "cross_mesh_exterior_metrics": trace_metrics,
            "operator_rows": trace_rows,
        },
        "selected_trace_refinement": {
            "pairwise_history": selected_cross_metrics,
            "state_observed_order": state_order,
            "rate_observed_order": rate_order,
            "signal_diagnostics": {
                f"N{mesh}": selected_signals[mesh] for mesh in MESHES
            },
            "signal_pair_diagnostics": selected_signal_pairs,
            "signal_passed": signal_passed,
            "passed": refinement_passed,
        },
        "excision_placement": {
            "offsets": PLACEMENT_OFFSETS,
            "operator_rows": placement_rows,
            "cross_mesh_exterior_metrics": placement_cross_metrics,
            "same_mesh_exterior_sensitivity": placement_sensitivity,
            "passed": placement_passed,
        },
        "gates": {
            "maximum_anchor_coordinate_defect": (
                MAXIMUM_ANCHOR_COORDINATE_DEFECT
            ),
            "maximum_pair_coordinate_defect": (
                MAXIMUM_PAIR_COORDINATE_DEFECT
            ),
            "maximum_active_scaled_anchor_correction": (
                MAXIMUM_ACTIVE_SCALED_ANCHOR_CORRECTION
            ),
            "maximum_buffer_scaled_anchor_correction": (
                MAXIMUM_BUFFER_SCALED_ANCHOR_CORRECTION
            ),
            "minimum_spatial_order": MINIMUM_SPATIAL_ORDER,
            "minimum_signed_cosine": MINIMUM_SIGNED_COSINE,
            "maximum_zero_crossing_defect": (
                MAXIMUM_ZERO_CROSSING_DEFECT
            ),
            "maximum_frequency_defect": MAXIMUM_FREQUENCY_DEFECT,
            "maximum_damping_defect": MAXIMUM_DAMPING_DEFECT,
            "maximum_exterior_placement_history_defect": (
                MAXIMUM_EXTERIOR_PLACEMENT_HISTORY_DEFECT
            ),
            "maximum_propagation_growth_exponent": (
                MAXIMUM_PROPAGATION_GROWTH_EXPONENT
            ),
        },
        "decision": {
            "anchor_consistency_passed": anchor_passed,
            "selected_trace_refinement_passed": refinement_passed,
            "excision_placement_insensitivity_passed": placement_passed,
            "n512_or_embedded_patch_authorized": n512_authorized,
            "fixed_q_averaging_authorized": False,
            "production_boundary_replacement_authorized": False,
        },
        "artifacts": {
            "arrays_path": _relative(DEFAULT_ARRAYS),
            "arrays_sha256": _sha256(DEFAULT_ARRAYS),
            "runner": THIS_RUNNER,
            "runner_sha256": _sha256(ROOT / THIS_RUNNER),
        },
        "wall_seconds": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    DEFAULT_OUTPUT.write_text(
        json.dumps(
            _plain(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return payload, arrays_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    payload, _arrays = run(force=args.force)
    print(json.dumps(_plain(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
