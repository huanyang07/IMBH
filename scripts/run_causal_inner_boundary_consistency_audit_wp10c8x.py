"""Run the WP10c8x inner boundary flux/storage consistency audit.

WP10c8w showed that the audit-only inner trace changed both the physical
excision flux and the first-cell Gauss reconstruction.  This package first
separates those two roles and measures their smooth-profile consistency
before any additional frozen-linear history is authorized.
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
from numpy.polynomial.legendre import leggauss
import scipy
from numpy.polynomial import Chebyshev

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_anchor_excision_audit_wp10c8w as wp10c8w
import run_causal_inner_phase_spatial_preflight_wp10c8v as wp10c8v

from imri_qpe.constants import C  # noqa: E402
from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    causal_five_field_evolving_tangent_matrices,
    causal_five_field_reconstruct_face_charts,
    causal_five_field_reduced_descriptor_matrices,
    causal_five_field_state_from_primitives,
    evaluate_causal_five_field_dae,
    kerr_schild_column_geometry,
    make_kerr_schild_column_grid,
    pack_causal_five_field_state,
)
from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    causal_inner_dae_system as dae,
)


BASE_COMMIT = "6764fc117ce453b4deb5c6b1c275a19c7352b4be"
WORK_PACKAGE = "WP10c8x"
SCHEMA_VERSION = 1
THIS_RUNNER = (
    "scripts/run_causal_inner_boundary_consistency_audit_wp10c8x.py"
)
CORE_DAE_FILE = (
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_dae_system.py"
)
CORE_SPATIAL_FILE = (
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_spatial_audit.py"
)

MESHES = (64, 128, 256, 512)
ORDER_MESHES = (128, 256, 512)
HISTORY_MESHES = (64, 128, 256)
MANUFACTURED_QUADRATURE_ORDER = 32
MANUFACTURED_DERIVATIVE_STEP = 2.0e-6
MANUFACTURED_POLYNOMIAL_DEGREE = 5
MINIMUM_STATIC_BOUNDARY_ORDER = 1.5
MAXIMUM_FINE_RELATIVE_ERROR = 2.5e-3
MAXIMUM_INNER_SPEED_OVER_C = 0.0
MAXIMUM_STORAGE_ACTION_DEFECT = 5.0e-5
MAXIMUM_GENERATOR_FACTORIZATION_DEFECT = 1.0e-8
MAXIMUM_PROPAGATION_GROWTH_EXPONENT = 10.0
MINIMUM_HISTORY_SPATIAL_ORDER = 0.75
MINIMUM_HISTORY_SIGNED_COSINE = 0.90
MAXIMUM_ZERO_CROSSING_DEFECT = 0.10
MAXIMUM_FREQUENCY_DEFECT = 0.10
MAXIMUM_DAMPING_DEFECT = 0.25
MINIMUM_INITIAL_SIGNED_COSINE = 0.99
MAXIMUM_INITIAL_AMPLITUDE_DEFECT = 0.05
MAXIMUM_INITIAL_RELATIVE_L2_DEFECT = 0.10

CANDIDATES = {
    "production": ("inherit", "inherit"),
    "flux_linear": ("linear_outgoing", "inherit"),
    "storage_linear": ("inherit", "linear_outgoing"),
    "both_linear": ("linear_outgoing", "linear_outgoing"),
    "flux_cell_centered": ("cell_centered", "inherit"),
    "storage_cell_centered": ("inherit", "cell_centered"),
}

WP10C8W_ANCHOR = (
    ROOT
    / "outputs/checkpoints/causal_five_field_wp10c8w/"
    "N256_anchor_inherit_arrays.npz"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/tables/"
    "causal_inner_boundary_consistency_audit_wp10c8x.json"
)
DEFAULT_ARRAYS = (
    ROOT
    / "outputs/tables/"
    "causal_inner_boundary_consistency_audit_wp10c8x_arrays.npz"
)
CHECKPOINT_DIRECTORY = (
    ROOT / "outputs/checkpoints/causal_five_field_wp10c8x"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
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


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _component_error(
    numerical: np.ndarray,
    exact: np.ndarray,
    scale: np.ndarray,
) -> float:
    difference = (
        np.asarray(numerical, dtype=float)
        - np.asarray(exact, dtype=float)
    )
    denominator = np.maximum(
        np.asarray(scale, dtype=float),
        np.finfo(float).tiny,
    )
    return float(np.sqrt(np.mean(np.square(difference / denominator))))


def _observed_order(coarse: float, fine: float) -> float | None:
    first = float(coarse)
    second = float(fine)
    if not (
        np.isfinite(first)
        and np.isfinite(second)
        and first > 0.0
        and second > 0.0
    ):
        return None
    return float(np.log2(first / second))


def _high_order_cell_nodes_and_weights(context, cell: int):
    nodes, base_weights = leggauss(MANUFACTURED_QUADRATURE_ORDER)
    lower = float(np.log(context.grid.edges[cell]))
    upper = float(np.log(context.grid.edges[cell + 1]))
    midpoint = 0.5 * (lower + upper)
    half_width = 0.5 * (upper - lower)
    log_nodes = midpoint + half_width * nodes
    radii = np.exp(log_nodes)
    raw = np.asarray(
        [
            half_width
            * weight
            * radius
            * kerr_schild_column_geometry(
                float(radius),
                context.grid.gravitational_radius,
            ).face_measure
            for radius, weight in zip(
                radii,
                base_weights,
                strict=True,
            )
        ],
        dtype=float,
    )
    weights = (
        raw
        * float(context.grid.cell_measures[cell])
        / float(np.sum(raw))
    )
    return radii, weights


def _manufactured_chart_function(anchor: dict[str, np.ndarray]):
    radii = np.asarray(anchor["radius_rg"], dtype=float)
    primitives = np.asarray(anchor["base_primitives"], dtype=float)
    log_radii = np.log(radii)
    fits = tuple(
        Chebyshev.fit(
            log_radii,
            primitives[:, field],
            deg=MANUFACTURED_POLYNOMIAL_DEGREE,
        )
        for field in range(primitives.shape[1])
    )

    def evaluate(radius_rg):
        argument = np.log(np.asarray(radius_rg))
        return np.stack(
            [fit(argument) for fit in fits],
            axis=-1,
        ).astype(float)

    return evaluate


def _local_context(reference_context, chart_function, mesh: int):
    n_cells = 3 * int(mesh) // 8
    anchor = _load_npz(WP10C8W_ANCHOR)
    lower_rg = float(anchor["grid_edges_rg"][0])
    upper_rg = float(anchor["grid_edges_rg"][-1])
    gravitational_radius = float(
        reference_context.grid.gravitational_radius
    )
    grid = make_kerr_schild_column_grid(
        lower_rg * gravitational_radius,
        upper_rg * gravitational_radius,
        n_cells,
        gravitational_radius,
    )
    frozen = chart_function(upper_rg)
    return replace(
        reference_context,
        grid=grid,
        stream_sources=wp10c8v._zero_sources(n_cells),
        outer_boundary_flux_mode="frozen_exterior_rusanov",
        outer_boundary_frozen_exterior_chart=np.asarray(
            frozen,
            dtype=float,
        ),
        inner_boundary_trace_override="inherit",
        inner_flux_trace_override="inherit",
        inner_storage_trace_override="inherit",
    ).validated()


def _local_derivatives(context, chart_function, radius: float):
    gravitational_radius = float(context.grid.gravitational_radius)
    radius_rg = float(radius / gravitational_radius)
    log_radius = float(np.log(radius_rg))
    step = MANUFACTURED_DERIVATIVE_STEP

    def local_values(offset: float):
        trial_rg = float(np.exp(log_radius + offset))
        trial_radius = trial_rg * gravitational_radius
        chart = chart_function(trial_rg)
        state = dae._cell_state(context, trial_radius, chart)
        four_velocity = dae.kerr_schild_column_four_velocity(
            state.geometry,
            state.primitive,
        )
        lower = state.geometry.spacetime_metric @ four_velocity
        return (
            np.asarray(lower, dtype=float),
            float(np.log(state.thermodynamics.proper_half_thickness)),
        )

    lower_minus, height_minus = local_values(-step)
    lower_plus, height_plus = local_values(step)
    radial_width = float(
        radius * (np.exp(step) - np.exp(-step))
    )
    lower_derivative = (lower_plus - lower_minus) / radial_width
    height_derivative = (height_plus - height_minus) / radial_width
    state = dae._cell_state(
        context,
        radius,
        chart_function(radius_rg),
    )
    shear = dae.causal_rest_frame_shear_rate(
        state.geometry,
        state.primitive,
        radial_lower_four_velocity_derivative=lower_derivative,
    )
    four_velocity = dae.kerr_schild_column_four_velocity(
        state.geometry,
        state.primitive,
    )
    height_rate = C * four_velocity[1] * height_derivative
    return float(shear), float(height_rate)


def _inner_flux_pieces(context, chart: np.ndarray) -> dict[str, np.ndarray]:
    radius = float(context.grid.edges[0])
    measure = float(context.grid.face_measures[0])
    state = dae._cell_state(context, radius, chart)
    total = measure * np.asarray(state.flux_over_c, dtype=float)
    stress = np.zeros(5, dtype=float)
    stress[:4] = (
        measure
        * np.asarray(
            state.stress.stress_killing_flux_increment_over_c,
            dtype=float,
        )
    )
    return {
        "total": total,
        "perfect_fluid": total - stress,
        "stress": stress,
    }


def _exact_first_cell_terms(context, chart_function):
    gravitational_radius = float(context.grid.gravitational_radius)
    inner_rg = float(context.grid.edges[0] / gravitational_radius)
    outer_rg = float(context.grid.edges[1] / gravitational_radius)
    inner_chart = chart_function(inner_rg)
    outer_chart = chart_function(outer_rg)
    inner_pieces = _inner_flux_pieces(context, inner_chart)
    inner_flux = inner_pieces["total"]
    central, dissipative = dae._interior_rusanov_flux_components(
        context,
        1,
        outer_chart,
        outer_chart,
    )
    if np.any(dissipative != 0.0):
        raise RuntimeError("equal-state manufactured Rusanov jump is nonzero")
    outer_flux = central
    storage = np.zeros(5, dtype=float)
    source = np.zeros(5, dtype=float)
    radii, weights = _high_order_cell_nodes_and_weights(context, 0)
    for radius, weight in zip(radii, weights, strict=True):
        chart = chart_function(radius / gravitational_radius)
        state = dae._cell_state(context, float(radius), chart)
        storage += weight * state.conserved
        shear, height_rate = _local_derivatives(
            context,
            chart_function,
            float(radius),
        )
        density, _depth, _components = dae._local_cell_source_density(
            context,
            state,
            shear_rate=shear,
            height_rate=height_rate,
        )
        source += weight * density
    storage /= float(context.grid.cell_measures[0])
    residual = outer_flux - inner_flux - source
    return {
        "inner_chart": inner_chart,
        "inner_flux": inner_flux,
        "inner_perfect_fluid_flux": inner_pieces["perfect_fluid"],
        "inner_stress_flux": inner_pieces["stress"],
        "outer_flux": outer_flux,
        "storage": storage,
        "source": source,
        "residual": residual,
    }


def _candidate_static_row(
    *,
    base_context,
    chart_function,
    mesh: int,
    flux_mode: str,
    storage_mode: str,
    primitive_scale: np.ndarray,
):
    context = replace(
        base_context,
        inner_flux_trace_override=flux_mode,
        inner_storage_trace_override=storage_mode,
    ).validated()
    gravitational_radius = float(context.grid.gravitational_radius)
    radius_rg = np.asarray(
        context.grid.centers / gravitational_radius,
        dtype=float,
    )
    primitives = chart_function(radius_rg)
    state = causal_five_field_state_from_primitives(context, primitives)
    vector = pack_causal_five_field_state(state)
    evaluation = evaluate_causal_five_field_dae(vector, context)
    flux_reconstruction = causal_five_field_reconstruct_face_charts(
        context,
        primitives,
        purpose="flux",
    )
    storage_reconstruction = causal_five_field_reconstruct_face_charts(
        context,
        primitives,
        purpose="storage",
    )
    exact = _exact_first_cell_terms(context, chart_function)
    numerical_inner_pieces = _inner_flux_pieces(
        context,
        flux_reconstruction.right_face_charts[0],
    )
    numerical_inner = np.asarray(
        evaluation.numerical_weighted_face_fluxes_over_c[0],
        dtype=float,
    )
    numerical_outer = np.asarray(
        evaluation.numerical_weighted_face_fluxes_over_c[1],
        dtype=float,
    )
    numerical_source = np.asarray(
        evaluation.integrated_sources_per_ct[0],
        dtype=float,
    )
    numerical_residual = numerical_outer - numerical_inner - numerical_source
    flux_scale = np.maximum(
        np.abs(exact["inner_flux"]) + np.abs(exact["outer_flux"]),
        np.finfo(float).tiny,
    )
    storage_scale = np.maximum(
        np.abs(exact["storage"]),
        np.finfo(float).tiny,
    )
    source_scale = np.maximum(
        np.abs(exact["source"])
        + np.abs(exact["inner_flux"])
        + np.abs(exact["outer_flux"]),
        np.finfo(float).tiny,
    )
    residual_scale = source_scale
    perfect_scale = np.maximum(
        np.abs(exact["inner_perfect_fluid_flux"]),
        flux_scale,
    )
    stress_scale = np.maximum(
        np.abs(exact["inner_stress_flux"]),
        flux_scale,
    )
    maximum_speed = dae._maximum_inner_trace_speed_over_c(
        context,
        flux_reconstruction.right_face_charts[0],
    )
    central, rusanov = dae._interior_rusanov_flux_components(
        context,
        1,
        flux_reconstruction.left_face_charts[1],
        flux_reconstruction.right_face_charts[1],
    )
    return {
        "mesh": mesh,
        "n_cells": int(context.grid.centers.size),
        "inner_flux_trace_error": _component_error(
            flux_reconstruction.right_face_charts[0],
            exact["inner_chart"],
            primitive_scale,
        ),
        "inner_storage_trace_error": _component_error(
            storage_reconstruction.right_face_charts[0],
            exact["inner_chart"],
            primitive_scale,
        ),
        "inner_flux_error": _component_error(
            numerical_inner,
            exact["inner_flux"],
            flux_scale,
        ),
        "inner_perfect_fluid_flux_error": _component_error(
            numerical_inner_pieces["perfect_fluid"],
            exact["inner_perfect_fluid_flux"],
            perfect_scale,
        ),
        "inner_stress_flux_error": _component_error(
            numerical_inner_pieces["stress"],
            exact["inner_stress_flux"],
            stress_scale,
        ),
        "first_cell_storage_error": _component_error(
            evaluation.mapped_conserved[0],
            exact["storage"],
            storage_scale,
        ),
        "first_cell_source_error": _component_error(
            numerical_source,
            exact["source"],
            source_scale,
        ),
        "boundary_transport_error": _component_error(
            numerical_outer - numerical_inner,
            exact["outer_flux"] - exact["inner_flux"],
            flux_scale,
        ),
        "complete_boundary_row_error": _component_error(
            numerical_residual,
            exact["residual"],
            residual_scale,
        ),
        "interior_rusanov_fraction": float(
            np.linalg.norm(rusanov / flux_scale)
            / max(
                np.linalg.norm((central + rusanov) / flux_scale),
                np.finfo(float).tiny,
            )
        ),
        "maximum_inner_trace_speed_over_c": maximum_speed,
        "incoming_inner_characteristic": bool(maximum_speed > 0.0),
        "state_is_finite": bool(
            np.all(np.isfinite(vector))
            and np.all(np.isfinite(evaluation.residual))
        ),
    }


def _candidate_summary(rows: dict[int, dict]) -> dict:
    metrics = (
        "inner_flux_trace_error",
        "inner_storage_trace_error",
        "inner_flux_error",
        "inner_perfect_fluid_flux_error",
        "inner_stress_flux_error",
        "first_cell_storage_error",
        "first_cell_source_error",
        "boundary_transport_error",
        "complete_boundary_row_error",
    )
    orders = {}
    for metric in metrics:
        values = [rows[mesh][metric] for mesh in ORDER_MESHES]
        orders[metric] = {
            "N128_N256": _observed_order(values[0], values[1]),
            "N256_N512": _observed_order(values[1], values[2]),
        }
    binding_metrics = (
        "inner_flux_error",
        "first_cell_storage_error",
        "boundary_transport_error",
        "complete_boundary_row_error",
    )
    order_passed = all(
        (
            orders[metric]["N256_N512"] is not None
            and orders[metric]["N256_N512"]
            >= MINIMUM_STATIC_BOUNDARY_ORDER
        )
        or rows[512][metric] <= 1.0e-10
        for metric in binding_metrics
    )
    fine_error_passed = all(
        rows[512][metric] <= MAXIMUM_FINE_RELATIVE_ERROR
        for metric in binding_metrics
    )
    characteristic_passed = all(
        not row["incoming_inner_characteristic"]
        for row in rows.values()
    )
    passed = bool(
        order_passed
        and fine_error_passed
        and characteristic_passed
        and all(row["state_is_finite"] for row in rows.values())
    )
    return {
        "orders": orders,
        "order_passed": order_passed,
        "fine_error_passed": fine_error_passed,
        "characteristic_passed": characteristic_passed,
        "passed": passed,
    }


def _history_operator_paths(label: str) -> tuple[Path, Path]:
    return (
        CHECKPOINT_DIRECTORY / f"{label}.json",
        CHECKPOINT_DIRECTORY / f"{label}_arrays.npz",
    )


def _history_operator_contract(
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
        "inner_flux_trace_override": context.inner_flux_trace_override,
        "inner_storage_trace_override": (
            context.inner_storage_trace_override
        ),
        "primitives_sha256": _array_sha256(primitives),
        "minus_sha256": _array_sha256(minus),
        "plus_sha256": _array_sha256(plus),
        "core_dae_sha256": _sha256(ROOT / CORE_DAE_FILE),
        "core_spatial_sha256": _sha256(ROOT / CORE_SPATIAL_FILE),
    }


def _build_or_load_history_operator(
    label: str,
    *,
    context,
    primitives: np.ndarray,
    minus: np.ndarray,
    plus: np.ndarray,
    amplitudes: np.ndarray,
    normalization: float,
    force: bool,
) -> tuple[dict, dict[str, np.ndarray]]:
    contract = _history_operator_contract(
        label,
        context=context,
        primitives=primitives,
        minus=minus,
        plus=plus,
    )
    json_path, arrays_path = _history_operator_paths(label)
    if not force and json_path.exists() and arrays_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if (
            all(payload.get(key) == value for key, value in contract.items())
            and payload.get("arrays_sha256") == _sha256(arrays_path)
            and payload.get("passed") is True
        ):
            return payload, _load_npz(arrays_path)

    started = time.perf_counter()
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
        "primitive_column_scales": scales,
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
        "physical_input_amplitudes": np.asarray(
            amplitudes,
            dtype=float,
        ),
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
        "initial_normalization": np.asarray(
            normalization,
            dtype=float,
        ),
    }
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **arrays)
    gates = wp10c8w._audit_local_state_gates(context, vector)
    payload = {
        **contract,
        "producer_runner": THIS_RUNNER,
        "producer_runner_sha256": _sha256(ROOT / THIS_RUNNER),
        "arrays_path": _relative(arrays_path),
        "arrays_sha256": _sha256(arrays_path),
        "rate_source": evolving["rate_source"],
        "state_gates": gates,
        "state_gates_passed": gates["passed"],
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


def _history_inputs(
    mesh: int,
    *,
    context,
    flux_mode: str,
    storage_mode: str,
) -> tuple[object, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    parent_path = (
        ROOT
        / "outputs/checkpoints/causal_five_field_wp10c8w/"
        f"N{mesh:03d}_anchor_inherit_arrays.npz"
    )
    parent = _load_npz(parent_path)
    base = np.asarray(parent["base_primitives"], dtype=float)
    half = np.asarray(parent["matched_half_difference"], dtype=float)
    candidate_context = replace(
        context,
        outer_boundary_frozen_exterior_chart=np.asarray(
            parent["frozen_exterior_chart"],
            dtype=float,
        ),
        inner_boundary_trace_override="inherit",
        inner_flux_trace_override=flux_mode,
        inner_storage_trace_override=storage_mode,
    ).validated()
    return candidate_context, parent, base - half, base + half


def _history_candidate(
    name: str,
    *,
    flux_mode: str,
    storage_mode: str,
    contexts: dict[int, object],
    force: bool,
) -> tuple[dict, dict[str, np.ndarray]]:
    rows = {}
    operator_arrays = {}
    histories = {}
    saved_arrays = {}
    for mesh in HISTORY_MESHES:
        context, parent, minus, plus = _history_inputs(
            mesh,
            context=contexts[mesh],
            flux_mode=flux_mode,
            storage_mode=storage_mode,
        )
        label = f"N{mesh:03d}_{name}"
        print(f"WP10c8x: building/loading {label}", flush=True)
        row, arrays = _build_or_load_history_operator(
            label,
            context=context,
            primitives=np.asarray(parent["base_primitives"], dtype=float),
            minus=minus,
            plus=plus,
            amplitudes=np.asarray(
                parent["physical_input_amplitudes"],
                dtype=float,
            ),
            normalization=float(parent["initial_normalization"]),
            force=force,
        )
        safety = wp10c8w._generator_propagation_safety(arrays)
        rows[mesh] = {
            "operator": row,
            "propagation_safety": safety,
        }
        operator_arrays[mesh] = arrays
        if not row["passed"] or not safety["passed"]:
            return (
                {
                    "available": False,
                    "reason": (
                        "operator_contract_failed"
                        if not row["passed"]
                        else "propagation_safety_failed"
                    ),
                    "rows": rows,
                },
                saved_arrays,
            )
        histories[mesh] = wp10c8w._propagate(arrays)
        saved_arrays[f"{name}_N{mesh}_times"] = histories[mesh]["times"]
        saved_arrays[f"{name}_N{mesh}_stress_rate"] = histories[mesh][
            "stress_rate_signal"
        ]

    def initial_metrics(
        coarse: dict[str, np.ndarray],
        fine: dict[str, np.ndarray],
        coarse_arrays: dict[str, np.ndarray],
        fine_arrays: dict[str, np.ndarray],
        field: str,
    ) -> dict:
        coarse_values = np.asarray(coarse[field][0], dtype=float)
        fine_values = wp10c8v._restrict_pairwise(
            np.asarray(fine[field][0:1], dtype=float),
            fine_arrays["cell_measures"],
        )[0]
        radius = np.asarray(coarse_arrays["radius_rg"], dtype=float)
        mask = (
            radius >= wp10c8w.COMMON_EXTERIOR_INNER_RG
        ) & (
            radius
            <= wp10c8w._ACTIVE_OUTER_RG * (1.0 + 2.0e-14)
        )
        weights = np.asarray(
            coarse_arrays["cell_measures"],
            dtype=float,
        )[mask]
        weights /= np.sum(weights)
        first = coarse_values[mask]
        second = fine_values[mask]
        first_norm = float(
            np.sqrt(np.sum(weights[:, None] * first**2))
        )
        second_norm = float(
            np.sqrt(np.sum(weights[:, None] * second**2))
        )
        difference_norm = float(
            np.sqrt(
                np.sum(weights[:, None] * (second - first) ** 2)
            )
        )
        return {
            "signed_cosine": float(
                np.sum(weights[:, None] * first * second)
                / max(
                    first_norm * second_norm,
                    np.finfo(float).tiny,
                )
            ),
            "amplitude_ratio": float(
                second_norm / max(first_norm, np.finfo(float).tiny)
            ),
            "relative_l2_difference": float(
                difference_norm
                / max(first_norm, np.finfo(float).tiny)
            ),
        }

    initial_pair_metrics = {
        "N64_N128": {
            field: initial_metrics(
                histories[64],
                histories[128],
                operator_arrays[64],
                operator_arrays[128],
                field,
            )
            for field in ("state", "rate")
        },
        "N128_N256": {
            field: initial_metrics(
                histories[128],
                histories[256],
                operator_arrays[128],
                operator_arrays[256],
                field,
            )
            for field in ("state", "rate")
        },
    }
    fine_initial = initial_pair_metrics["N128_N256"]
    initial_match_passed = all(
        values["signed_cosine"] >= MINIMUM_INITIAL_SIGNED_COSINE
        and abs(values["amplitude_ratio"] - 1.0)
        <= MAXIMUM_INITIAL_AMPLITUDE_DEFECT
        and values["relative_l2_difference"]
        <= MAXIMUM_INITIAL_RELATIVE_L2_DEFECT
        for values in fine_initial.values()
    )
    pair_metrics = {
        "N64_N128": wp10c8w._pair_metrics(
            histories[64],
            histories[128],
            operator_arrays[64],
            operator_arrays[128],
            lower_rg=wp10c8w.COMMON_EXTERIOR_INNER_RG,
        ),
        "N128_N256": wp10c8w._pair_metrics(
            histories[128],
            histories[256],
            operator_arrays[128],
            operator_arrays[256],
            lower_rg=wp10c8w.COMMON_EXTERIOR_INNER_RG,
        ),
    }
    state_coarse = pair_metrics["N64_N128"]["state"][
        "maximum_relative_l2_difference"
    ]
    state_fine = pair_metrics["N128_N256"]["state"][
        "maximum_relative_l2_difference"
    ]
    rate_coarse = pair_metrics["N64_N128"]["rate"][
        "maximum_relative_l2_difference"
    ]
    rate_fine = pair_metrics["N128_N256"]["rate"][
        "maximum_relative_l2_difference"
    ]
    state_order = _observed_order(state_coarse, state_fine)
    rate_order = _observed_order(rate_coarse, rate_fine)
    signals = {
        mesh: wp10c8v._signal_diagnostics(
            histories[mesh]["times"],
            histories[mesh]["stress_rate_signal"],
        )
        for mesh in HISTORY_MESHES
    }
    signal_pairs = {
        "N64_N128": wp10c8v._signal_pair_metrics(
            signals[64],
            signals[128],
        ),
        "N128_N256": wp10c8v._signal_pair_metrics(
            signals[128],
            signals[256],
        ),
    }
    fine_signal = signal_pairs["N128_N256"]
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
    minimum_cosine = min(
        pair_metrics["N128_N256"]["state"]["minimum_signed_cosine"],
        pair_metrics["N128_N256"]["rate"]["minimum_signed_cosine"],
    )
    passed = bool(
        state_order is not None
        and rate_order is not None
        and state_order >= MINIMUM_HISTORY_SPATIAL_ORDER
        and rate_order >= MINIMUM_HISTORY_SPATIAL_ORDER
        and minimum_cosine >= MINIMUM_HISTORY_SIGNED_COSINE
        and signal_passed
        and initial_match_passed
    )
    return (
        {
            "available": True,
            "rows": rows,
            "pairwise_history": pair_metrics,
            "initial_pair_metrics": initial_pair_metrics,
            "initial_match_passed": initial_match_passed,
            "state_observed_order": state_order,
            "rate_observed_order": rate_order,
            "minimum_fine_signed_cosine": minimum_cosine,
            "signals": signals,
            "signal_pairs": signal_pairs,
            "signal_passed": signal_passed,
            "passed": passed,
        },
        saved_arrays,
    )


def _history_operator_equivalence() -> dict:
    comparisons = {
        "production_vs_storage_linear": (
            "production",
            "storage_linear",
        ),
        "flux_linear_vs_both_linear": (
            "flux_linear",
            "both_linear",
        ),
    }
    fields = (
        "descriptor",
        "stationary_jacobian",
        "storage_rate_derivative",
        "generator",
        "base_physical_rate_per_s",
    )
    result = {}
    for label, (first_name, second_name) in comparisons.items():
        by_mesh = {}
        for mesh in HISTORY_MESHES:
            first_path = _history_operator_paths(
                f"N{mesh:03d}_{first_name}"
            )[1]
            second_path = _history_operator_paths(
                f"N{mesh:03d}_{second_name}"
            )[1]
            if not first_path.exists() or not second_path.exists():
                by_mesh[str(mesh)] = {"available": False}
                continue
            first = _load_npz(first_path)
            second = _load_npz(second_path)
            defects = {
                field: float(
                    np.max(
                        np.abs(
                            np.asarray(first[field], dtype=float)
                            - np.asarray(second[field], dtype=float)
                        )
                    )
                )
                for field in fields
            }
            by_mesh[str(mesh)] = {
                "available": True,
                "maximum_absolute_defects": defects,
                "bitwise_equal": all(
                    np.array_equal(first[field], second[field])
                    for field in fields
                ),
            }
        result[label] = {
            "by_mesh": by_mesh,
            "all_available": all(
                row["available"] for row in by_mesh.values()
            ),
            "all_bitwise_equal": all(
                row.get("bitwise_equal", False)
                for row in by_mesh.values()
            ),
        }
    return result


def run(*, force: bool = False) -> tuple[dict, dict[str, np.ndarray]]:
    started = time.perf_counter()
    if not WP10C8W_ANCHOR.exists():
        raise FileNotFoundError(
            "WP10c8x requires the committed WP10c8w local anchor evidence"
        )
    anchor = _load_npz(WP10C8W_ANCHOR)
    chart_function = _manufactured_chart_function(anchor)
    parents = {
        mesh: wp10c8v._parent_bundle(mesh)
        for mesh in wp10c8v.PARENT_MESHES
    }
    reference_context = parents[128]["context"]
    base_contexts = {
        mesh: _local_context(reference_context, chart_function, mesh)
        for mesh in MESHES
    }
    primitive_scale = np.asarray(
        anchor["primitive_column_scales"][0],
        dtype=float,
    )

    candidates = {}
    arrays = {}
    for name, (flux_mode, storage_mode) in CANDIDATES.items():
        print(f"WP10c8x: static candidate {name}", flush=True)
        rows = {
            mesh: _candidate_static_row(
                base_context=base_contexts[mesh],
                chart_function=chart_function,
                mesh=mesh,
                flux_mode=flux_mode,
                storage_mode=storage_mode,
                primitive_scale=primitive_scale,
            )
            for mesh in MESHES
        }
        summary = _candidate_summary(rows)
        candidates[name] = {
            "flux_trace_override": flux_mode,
            "storage_trace_override": storage_mode,
            "rows": rows,
            **summary,
        }
        for mesh, row in rows.items():
            for metric, value in row.items():
                if isinstance(value, (float, int)) and not isinstance(
                    value,
                    bool,
                ):
                    arrays[f"{name}_N{mesh}_{metric}"] = np.asarray(value)

    passed_candidates = [
        name for name, result in candidates.items() if result["passed"]
    ]
    history_results = {}
    if passed_candidates:
        base_profiles = {
            mesh: wp10c8v._base_profiles(mesh, parents)
            for mesh in HISTORY_MESHES
        }
        history_contexts = {
            mesh: wp10c8v._local_context(
                parents[128]["context"],
                base_profiles[mesh],
            )
            for mesh in HISTORY_MESHES
        }
        wp10c8w._ACTIVE_OUTER_RG = float(
            base_profiles[64]["grid"].edges[
                wp10c8v._active_cell_count(64)
            ]
            / history_contexts[64].grid.gravitational_radius
        )
        for name in passed_candidates:
            flux_mode, storage_mode = CANDIDATES[name]
            result, history_arrays = _history_candidate(
                name,
                flux_mode=flux_mode,
                storage_mode=storage_mode,
                contexts=history_contexts,
                force=force,
            )
            history_results[name] = result
            arrays.update(history_arrays)

    passed_history_candidates = [
        name
        for name, result in history_results.items()
        if result.get("available") and result.get("passed")
    ]
    initially_matched_candidates = [
        name
        for name, result in history_results.items()
        if result.get("available") and result.get("initial_match_passed")
    ]
    operator_equivalence = (
        _history_operator_equivalence() if history_results else {}
    )
    if not passed_candidates:
        classification = "static_boundary_consistency_failed"
    elif not initially_matched_candidates:
        classification = "static_pass_but_common_initial_mode_unresolved"
    elif not passed_history_candidates:
        classification = "static_pass_but_inner_phase_unresolved"
    else:
        classification = "bounded_boundary_candidate_converged"
    DEFAULT_ARRAYS.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DEFAULT_ARRAYS, **arrays)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "base_commit": BASE_COMMIT,
        "classification": classification,
        "scope": {
            "production_boundary_changed": False,
            "new_truth_evolution_run": False,
            "frozen_linear_history_run": bool(history_results),
            "formal_fast_average_certified": False,
            "reduced_architecture_selected": False,
        },
        "manufactured_profile": {
            "source_anchor": _relative(WP10C8W_ANCHOR),
            "source_anchor_sha256": _sha256(WP10C8W_ANCHOR),
            "quadrature_order": MANUFACTURED_QUADRATURE_ORDER,
            "derivative_log_radius_step": (
                MANUFACTURED_DERIVATIVE_STEP
            ),
            "polynomial_degree": MANUFACTURED_POLYNOMIAL_DEGREE,
        },
        "candidates": candidates,
        "passed_static_candidates": passed_candidates,
        "history_candidates": history_results,
        "initially_matched_candidates": initially_matched_candidates,
        "passed_history_candidates": passed_history_candidates,
        "history_operator_equivalence": operator_equivalence,
        "gates": {
            "minimum_static_boundary_order": (
                MINIMUM_STATIC_BOUNDARY_ORDER
            ),
            "maximum_fine_relative_error": MAXIMUM_FINE_RELATIVE_ERROR,
            "maximum_inner_speed_over_c": MAXIMUM_INNER_SPEED_OVER_C,
            "maximum_storage_action_defect": (
                MAXIMUM_STORAGE_ACTION_DEFECT
            ),
            "maximum_generator_factorization_defect": (
                MAXIMUM_GENERATOR_FACTORIZATION_DEFECT
            ),
            "maximum_propagation_growth_exponent": (
                MAXIMUM_PROPAGATION_GROWTH_EXPONENT
            ),
            "minimum_history_spatial_order": (
                MINIMUM_HISTORY_SPATIAL_ORDER
            ),
            "minimum_history_signed_cosine": (
                MINIMUM_HISTORY_SIGNED_COSINE
            ),
            "maximum_zero_crossing_defect": (
                MAXIMUM_ZERO_CROSSING_DEFECT
            ),
            "maximum_frequency_defect": MAXIMUM_FREQUENCY_DEFECT,
            "maximum_damping_defect": MAXIMUM_DAMPING_DEFECT,
            "minimum_initial_signed_cosine": (
                MINIMUM_INITIAL_SIGNED_COSINE
            ),
            "maximum_initial_amplitude_defect": (
                MAXIMUM_INITIAL_AMPLITUDE_DEFECT
            ),
            "maximum_initial_relative_l2_defect": (
                MAXIMUM_INITIAL_RELATIVE_L2_DEFECT
            ),
        },
        "decision": {
            "static_boundary_candidate_available": bool(
                passed_candidates
            ),
            "bounded_history_completed": bool(history_results),
            "bounded_history_candidate_passed": bool(
                passed_history_candidates
            ),
            "n512_history_authorized": bool(
                passed_history_candidates
            ),
            "production_boundary_replacement_authorized": False,
            "fixed_q_averaging_authorized": False,
        },
        "artifacts": {
            "arrays_path": _relative(DEFAULT_ARRAYS),
            "arrays_sha256": _sha256(DEFAULT_ARRAYS),
            "runner": THIS_RUNNER,
            "runner_sha256": _sha256(ROOT / THIS_RUNNER),
            "core_dae_sha256": _sha256(ROOT / CORE_DAE_FILE),
            "core_spatial_sha256": _sha256(ROOT / CORE_SPATIAL_FILE),
        },
        "wall_seconds": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
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
    return payload, arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild cached frozen-linear operators",
    )
    arguments = parser.parse_args()
    payload, _arrays = run(force=arguments.force)
    print(json.dumps(_plain(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
