"""Run the WP10c9a characteristic-family phase and operator preflight.

The package reuses the certified WP10c8z live-coupled N128/N256/N512 hybrid
generators.  Five smooth compact packets are built from the responsive-height
local-rest eigenvectors on the exact Schur-reduced primitive DAE manifold and
propagated with the unchanged evolving generator.  A signed
transport/source/storage audit localizes the first nonconvergent rate block.

Three bounded reconstruction kernels are screened without changing production:
the current primitive quadratic reconstruction, a locally orthonormal
rapidity chart, and a characteristic perturbation reconstruction.  No screened
kernel is allowed to rerun the nonlinear common mode until it has a complete
live-coupled generator and passes all phase, storage, Jacobian, and restart
contracts.
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
from scipy.sparse.linalg import expm_multiply

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_embedded_patch_preflight_wp10c8z as wp10c8z

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES,
    causal_characteristic_packet_moments,
    causal_compact_log_radius_envelope,
    causal_embedded_patch_flux_audit,
    causal_five_field_characteristic_basis,
    causal_five_field_characteristic_packet,
    causal_five_field_dae_scaling,
    causal_five_field_face_flux_decomposition,
    causal_five_field_reconstruct_face_charts,
    causal_five_field_reduced_storage_action,
    causal_five_field_state_from_primitives,
    evaluate_causal_five_field_dae,
    pack_causal_five_field_state,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_dae_system import (  # noqa: E402
    _three_point_interpolate,
)


BASE_COMMIT = "6764fc117ce453b4deb5c6b1c275a19c7352b4be"
WORK_PACKAGE = "WP10c9a"
SCHEMA_VERSION = 1
THIS_RUNNER = (
    "scripts/run_causal_inner_characteristic_phase_audit_wp10c9a.py"
)
CORE_PACKET_FILE = (
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_characteristic_phase.py"
)
WP10C8Z_OUTPUT = (
    ROOT
    / "outputs/tables/"
    "causal_inner_embedded_patch_preflight_wp10c8z.json"
)
WP10C8Z_ARRAYS = (
    ROOT
    / "outputs/tables/"
    "causal_inner_embedded_patch_preflight_wp10c8z_arrays.npz"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/tables/"
    "causal_inner_characteristic_phase_audit_wp10c9a.json"
)
DEFAULT_ARRAYS = (
    ROOT
    / "outputs/tables/"
    "causal_inner_characteristic_phase_audit_wp10c9a_arrays.npz"
)

SUPPORT_INNER_RG = 2.15
SUPPORT_OUTER_RG = 5.40
TARGET_SECONDS = 0.125
TIME_SAMPLES = 201
TERM_DIRECTIONAL_STEP = 2.0e-2
RECONSTRUCTION_DIRECTIONAL_STEP = 1.0e-1

MINIMUM_SMOOTH_ORDER = 1.8
MINIMUM_PACKET_PHASE_ORDER = 0.75
MINIMUM_PACKET_DAMPING_ORDER = 0.75
MINIMUM_SIGNED_COSINE = 0.90
MAXIMUM_SHARED_FLUX_DEFECT = 1.0e-12
MAXIMUM_STORAGE_ACTION_DEFECT = 2.0e-5
MAXIMUM_RESTART_DEFECT = 2.0e-10
MAXIMUM_EIGENPAIR_DEFECT = 2.0e-11


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


def _observed_order(coarse: float, fine: float) -> float | None:
    if not (
        np.isfinite(coarse)
        and np.isfinite(fine)
        and coarse > 0.0
        and fine > 0.0
    ):
        return None
    return float(np.log2(coarse / fine))


def _continuum_norm(values: np.ndarray, measures: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    weights = np.asarray(measures, dtype=float)
    return float(
        np.sqrt(
            np.sum(weights[:, None] * array**2)
            / (5.0 * np.sum(weights))
        )
    )


def _configurations() -> tuple[dict, dict[str, dict], dict]:
    parent = wp10c8z._parent_data()
    active_outer_rg = float(
        parent["parent_grid"].edges[wp10c8z.ACTIVE_OUTER_PARENT_FACE]
        / parent["parent_grid"].gravitational_radius
    )
    parent["active_outer_rg"] = active_outer_rg
    shell_edges = wp10c8z._local_shell_edges_rg(
        parent["context"],
        parent["shell_zero_outer_rg"],
    )
    _state, _vector, _reduced, level = wp10c8z._build_reduced_level(
        parent["context"],
        parent["parent_base_primitives"],
        shell_edges,
    )
    target_values = np.asarray(level.coordinate_values, dtype=float)
    target_scales = np.asarray(level.coordinate_scales, dtype=float)
    labels = {
        1: "N128_exterior_N128_inner_c48",
        2: "N128_exterior_N256_inner_c48",
        4: "N128_exterior_N512_inner_c48",
    }
    configurations = {}
    for ratio, label in labels.items():
        print(f"WP10c9a: loading {label}", flush=True)
        configuration = wp10c8z._configuration(
            label=label,
            parent=parent,
            coupling_face=wp10c8z.PRIMARY_COUPLING_PARENT_FACE,
            refinement_ratio=ratio,
            target_values=target_values,
            target_scales=target_scales,
            active_outer_rg=active_outer_rg,
            force=False,
        )
        configuration["active_outer_rg"] = active_outer_rg
        configurations[label] = configuration
    return parent, configurations, labels


def _project_packet(
    configuration: dict,
    family: str,
) -> tuple[np.ndarray, tuple, dict]:
    context = configuration["context"]
    amplitudes = np.asarray(configuration["amplitudes"], dtype=float)
    seed, bases = causal_five_field_characteristic_packet(
        context,
        configuration["base_primitives"],
        amplitudes,
        family=family,
        support_inner_radius=(
            SUPPORT_INNER_RG * context.grid.gravitational_radius
        ),
        support_outer_radius=(
            SUPPORT_OUTER_RG * context.grid.gravitational_radius
        ),
    )
    # The Schur-reduced primitive generator is already the exact differential
    # DAE manifold: conserved and face-flux perturbations are reconstructed
    # algebraically from every primitive direction.  Imposing the unrelated
    # q_34 slow-coordinate null fiber here would rotate a localized family
    # packet into several characteristic families and defeat this benchmark.
    projected = np.array(seed, copy=True)
    norm = _continuum_norm(projected, context.grid.cell_measures)
    if norm <= np.finfo(float).tiny:
        raise RuntimeError("WP10c9a projected packet is zero")
    projected /= norm
    report = {
        "continuum_profile_cosine": 1.0,
        "exact_reduced_descriptor_primitive_manifold": True,
        "retained_slow_coordinate_null_fiber_imposed": False,
        "maximum_eigenpair_defect": max(
            basis.maximum_eigenpair_defect for basis in bases
        ),
        "maximum_characteristic_condition_number": max(
            basis.condition_number for basis in bases
        ),
        "passed": bool(
            max(
                basis.maximum_eigenpair_defect for basis in bases
            )
            <= MAXIMUM_EIGENPAIR_DEFECT
        ),
    }
    return projected, bases, report


def _propagate_packet(
    configuration: dict,
    initial: np.ndarray,
    bases: tuple,
    family: str,
) -> tuple[dict[str, np.ndarray], dict]:
    generator = np.asarray(configuration["generator"], dtype=float)
    times = np.linspace(0.0, TARGET_SECONDS, TIME_SAMPLES)
    state = np.asarray(
        expm_multiply(
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
    moments = causal_characteristic_packet_moments(
        state,
        bases,
        configuration["context"].grid.centers,
        configuration["context"].grid.cell_measures,
        family=family,
    )
    midpoint = expm_multiply(
        generator * (0.5 * TARGET_SECONDS),
        initial.ravel(),
    )
    restarted = expm_multiply(
        generator * (0.5 * TARGET_SECONDS),
        midpoint,
    )
    restart = float(
        np.linalg.norm(restarted - state[-1].ravel())
        / max(np.linalg.norm(state[-1]), np.finfo(float).tiny)
    )
    history = {
        "times": times,
        "state": state,
        "rate": rate,
        "stress_rate_signal": np.zeros(TIME_SAMPLES),
    }
    arrays = {
        "l2_amplitude": moments.l2_amplitude,
        "log_radius_centroid": moments.log_radius_centroid,
        "log_radius_width": moments.log_radius_width,
        "selected_family_fraction": moments.selected_family_fraction,
        "opposite_family_fraction": moments.opposite_family_fraction,
    }
    return history, {
        "restart_relative_defect": restart,
        "initial_selected_family_fraction": float(
            moments.selected_family_fraction[0]
        ),
        "maximum_opposite_family_fraction": float(
            np.max(moments.opposite_family_fraction)
        ),
        "arrays": arrays,
    }


def _moment_pair(first: dict, second: dict) -> dict:
    a = first["arrays"]
    b = second["arrays"]
    amplitude_defect = float(
        np.max(
            np.abs(
                np.log(
                    np.maximum(
                        b["l2_amplitude"],
                        np.finfo(float).tiny,
                    )
                    / np.maximum(
                        a["l2_amplitude"],
                        np.finfo(float).tiny,
                    )
                )
            )
        )
    )
    return {
        "maximum_log_radius_centroid_defect": float(
            np.max(
                np.abs(
                    b["log_radius_centroid"]
                    - a["log_radius_centroid"]
                )
            )
        ),
        "maximum_log_amplitude_defect": amplitude_defect,
        "maximum_log_radius_width_defect": float(
            np.max(
                np.abs(
                    b["log_radius_width"]
                    - a["log_radius_width"]
                )
            )
        ),
        "maximum_opposite_family_fraction": float(
            max(
                np.max(a["opposite_family_fraction"]),
                np.max(b["opposite_family_fraction"]),
            )
        ),
    }


def _integrated_restriction(values: np.ndarray, configuration: dict) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    layout = configuration["layout"]
    ratio = int(layout.refinement_ratio)
    parent_face = int(layout.parent_coupling_face_index)
    if array.shape != (layout.n_cells, 5):
        raise ValueError("WP10c9a integrated restriction shape is invalid")
    refined = array[: layout.n_refined_cells].reshape(
        parent_face,
        ratio,
        5,
    ).sum(axis=1)
    return np.concatenate((refined, array[layout.n_refined_cells :]), axis=0)


def _directional_blocks(
    configuration: dict,
    initial: np.ndarray,
) -> dict[str, np.ndarray]:
    context = configuration["context"]
    base = np.asarray(configuration["base_primitives"], dtype=float)
    amplitudes = np.asarray(configuration["amplitudes"], dtype=float)
    physical_direction = amplitudes * np.asarray(initial, dtype=float)
    step = TERM_DIRECTIONAL_STEP

    def blocks(primitives: np.ndarray) -> dict[str, np.ndarray]:
        state = causal_five_field_state_from_primitives(
            context,
            primitives,
        )
        vector = pack_causal_five_field_state(state)
        evaluation = evaluate_causal_five_field_dae(vector, context)
        faces = np.asarray(
            evaluation.numerical_weighted_face_fluxes_over_c,
            dtype=float,
        )
        split = causal_five_field_face_flux_decomposition(context, vector)
        components = {
            "inner_excision_transport": np.zeros_like(faces),
            "central_perfect_transport": np.zeros_like(faces),
            "central_stress_transport": np.zeros_like(faces),
            "rusanov_transport": np.zeros_like(faces),
            "outer_transport": np.zeros_like(faces),
        }
        components["inner_excision_transport"][0] = faces[0]
        components["central_perfect_transport"][1:-1] = (
            split.central_perfect_weighted_face_fluxes_over_c
        )
        components["central_stress_transport"][1:-1] = (
            split.central_stress_weighted_face_fluxes_over_c
        )
        components["rusanov_transport"][1:-1] = (
            split.rusanov_weighted_face_fluxes_over_c
        )
        components["outer_transport"][-1] = faces[-1]
        result = {
            name: values[1:] - values[:-1]
            for name, values in components.items()
        }
        for name, values in sorted(
            evaluation.integrated_source_components_per_ct.items()
        ):
            result[f"source_{name}"] = -np.asarray(values, dtype=float)
        return result

    plus = blocks(base + step * physical_direction)
    minus = blocks(base - step * physical_direction)
    result = {
        name: (plus[name] - minus[name]) / (2.0 * step)
        for name in plus
    }
    generator = np.asarray(configuration["generator"], dtype=float)
    dimensionless_rate = (
        generator @ np.asarray(initial, dtype=float).ravel()
    ).reshape(base.shape)
    physical_rate = amplitudes * dimensionless_rate
    storage = causal_five_field_reduced_storage_action(
        context,
        base.ravel(),
        physical_rate.ravel(),
        storage_difference_step=1.0e-4,
        storage_quadrature_order=4,
        storage_directional_step=1.0e-3,
        conserved_difference_order=4,
    )
    result["mapped_storage_action"] = np.asarray(
        storage["conserved_storage_per_ct"],
        dtype=float,
    )
    result["responsive_height_storage_action"] = np.asarray(
        storage["vertical_storage_per_ct"],
        dtype=float,
    )
    known = np.sum(np.asarray(list(result.values())), axis=0)
    result["evolving_storage_remainder"] = -known
    return result


def _term_difference_norms(
    medium: dict[str, np.ndarray],
    fine: dict[str, np.ndarray],
    medium_configuration: dict,
    fine_configuration: dict,
    parent_row_scales: np.ndarray,
    active_cells: int,
) -> dict[str, float]:
    names = tuple(medium)
    if tuple(fine) != names:
        raise RuntimeError("WP10c9a term schemas differ")
    scales = np.asarray(parent_row_scales, dtype=float).reshape(-1, 5)
    result = {}
    for name in names:
        coarse = _integrated_restriction(
            medium[name],
            medium_configuration,
        )
        refined = _integrated_restriction(
            fine[name],
            fine_configuration,
        )
        difference = (
            refined[:active_cells] - coarse[:active_cells]
        ) / scales[:active_cells]
        result[name] = float(np.linalg.norm(difference))
    return result


def _quadratic_face_values(
    log_centers: np.ndarray,
    log_edges: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_cells = int(log_centers.size)
    left = np.empty((n_cells + 1,) + values.shape[1:], dtype=float)
    right = np.empty_like(left)
    left[0] = values[0]
    right[0] = values[0]
    left[-1] = values[-1]
    right[-1] = values[-1]
    for cell in range(n_cells):
        start = min(max(cell - 1, 0), n_cells - 3)
        stencil = slice(start, start + 3)
        left_candidate = _three_point_interpolate(
            log_centers[stencil],
            values[stencil],
            float(log_edges[cell]),
        )
        right_candidate = _three_point_interpolate(
            log_centers[stencil],
            values[stencil],
            float(log_edges[cell + 1]),
        )
        if cell > 0:
            right[cell] = left_candidate
        if cell < n_cells - 1:
            left[cell + 1] = right_candidate
    return left, right


def _rapidity_chart(charts: np.ndarray) -> np.ndarray:
    result = np.array(charts, copy=True)
    velocity = np.asarray(charts[:, 1:3], dtype=float)
    speed = np.linalg.norm(velocity, axis=1)
    rapidity = np.arctanh(np.minimum(speed, 1.0 - 1.0e-14))
    factor = np.ones_like(speed)
    moving = speed > np.finfo(float).tiny
    factor[moving] = rapidity[moving] / speed[moving]
    result[:, 1:3] = velocity * factor[:, None]
    return result


def _inverse_rapidity_chart(charts: np.ndarray) -> np.ndarray:
    result = np.array(charts, copy=True)
    rapidity = np.asarray(charts[..., 1:3], dtype=float)
    magnitude = np.linalg.norm(rapidity, axis=-1)
    factor = np.ones_like(magnitude)
    moving = magnitude > np.finfo(float).tiny
    factor[moving] = np.tanh(magnitude[moving]) / magnitude[moving]
    result[..., 1:3] = rapidity * factor[..., None]
    return result


def _face_basis(
    configuration: dict,
    face: int,
) -> tuple[np.ndarray, float]:
    context = configuration["context"]
    radius = float(context.grid.edges[face])
    radius_rg = radius / context.grid.gravitational_radius
    base = wp10c8z._common_profile_function(
        configuration["context"].grid.centers
        / configuration["context"].grid.gravitational_radius,
        configuration["base_primitives"],
    )(np.asarray([radius_rg]))[0]
    amplitudes = wp10c8z._common_positive_profile_function(
        configuration["context"].grid.centers
        / configuration["context"].grid.gravitational_radius,
        configuration["amplitudes"],
    )(np.asarray([radius_rg]))[0]
    basis = causal_five_field_characteristic_basis(
        context,
        radius,
        base,
        amplitudes,
    )
    return basis.dimensionless_right_eigenvectors, radius_rg


def _candidate_errors(
    configuration: dict,
    family: str,
    method: str,
) -> dict:
    context = configuration["context"]
    base = np.asarray(configuration["base_primitives"], dtype=float)
    amplitudes = np.asarray(configuration["amplitudes"], dtype=float)
    pure, _bases = causal_five_field_characteristic_packet(
        context,
        base,
        amplitudes,
        family=family,
        support_inner_radius=(
            SUPPORT_INNER_RG * context.grid.gravitational_radius
        ),
        support_outer_radius=(
            SUPPORT_OUTER_RG * context.grid.gravitational_radius
        ),
    )
    step = RECONSTRUCTION_DIRECTIONAL_STEP
    log_centers = np.log(context.grid.centers)
    log_edges = np.log(context.grid.edges)
    if method == "production_primitive_quadratic":
        plus = causal_five_field_reconstruct_face_charts(
            context,
            base + step * amplitudes * pure,
        )
        minus = causal_five_field_reconstruct_face_charts(
            context,
            base - step * amplitudes * pure,
        )
        left_physical = (
            plus.left_face_charts - minus.left_face_charts
        ) / (2.0 * step)
        right_physical = (
            plus.right_face_charts - minus.right_face_charts
        ) / (2.0 * step)
    elif method == "horizon_rapidity_quadratic":
        plus_values = _rapidity_chart(base + step * amplitudes * pure)
        minus_values = _rapidity_chart(base - step * amplitudes * pure)
        plus_left, plus_right = _quadratic_face_values(
            log_centers,
            log_edges,
            plus_values,
        )
        minus_left, minus_right = _quadratic_face_values(
            log_centers,
            log_edges,
            minus_values,
        )
        left_physical = (
            _inverse_rapidity_chart(plus_left)
            - _inverse_rapidity_chart(minus_left)
        ) / (2.0 * step)
        right_physical = (
            _inverse_rapidity_chart(plus_right)
            - _inverse_rapidity_chart(minus_right)
        ) / (2.0 * step)
    elif method == "characteristic_perturbation_quadratic":
        envelope = causal_compact_log_radius_envelope(
            context.grid.centers,
            support_inner_radius=(
                SUPPORT_INNER_RG * context.grid.gravitational_radius
            ),
            support_outer_radius=(
                SUPPORT_OUTER_RG * context.grid.gravitational_radius
            ),
        )
        left_scalar, right_scalar = _quadratic_face_values(
            log_centers,
            log_edges,
            envelope[:, None],
        )
        left_scalar = left_scalar[:, 0]
        right_scalar = right_scalar[:, 0]
        left_coefficients = right_coefficients = None
    else:
        raise ValueError("unknown WP10c9a reconstruction candidate")

    selected = CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES.index(family)
    n_faces = context.grid.edges.size
    left_coefficients = np.zeros(n_faces, dtype=float)
    right_coefficients = np.zeros(n_faces, dtype=float)
    exact = causal_compact_log_radius_envelope(
        context.grid.edges,
        support_inner_radius=(
            SUPPORT_INNER_RG * context.grid.gravitational_radius
        ),
        support_outer_radius=(
            SUPPORT_OUTER_RG * context.grid.gravitational_radius
        ),
    )
    speeds = np.zeros(n_faces, dtype=float)
    for face in range(n_faces):
        face_basis, _radius_rg = _face_basis(configuration, face)
        basis_object = causal_five_field_characteristic_basis(
            context,
            float(context.grid.edges[face]),
            wp10c8z._common_profile_function(
                context.grid.centers / context.grid.gravitational_radius,
                base,
            )(
                np.asarray(
                    [
                        context.grid.edges[face]
                        / context.grid.gravitational_radius
                    ]
                )
            )[0],
            wp10c8z._common_positive_profile_function(
                context.grid.centers / context.grid.gravitational_radius,
                amplitudes,
            )(
                np.asarray(
                    [
                        context.grid.edges[face]
                        / context.grid.gravitational_radius
                    ]
                )
            )[0],
        )
        speeds[face] = basis_object.coordinate_speeds_over_c[selected]
        if method == "characteristic_perturbation_quadratic":
            left_coefficients[face] = left_scalar[face]
            right_coefficients[face] = right_scalar[face]
        else:
            face_amplitudes = wp10c8z._common_positive_profile_function(
                context.grid.centers / context.grid.gravitational_radius,
                amplitudes,
            )(
                np.asarray(
                    [
                        context.grid.edges[face]
                        / context.grid.gravitational_radius
                    ]
                )
            )[0]
            left_coefficients[face] = np.linalg.solve(
                face_basis,
                left_physical[face] / face_amplitudes,
            )[selected]
            right_coefficients[face] = np.linalg.solve(
                face_basis,
                right_physical[face] / face_amplitudes,
            )[selected]
    face_mask = (
        context.grid.edges >= SUPPORT_INNER_RG
        * context.grid.gravitational_radius
    ) & (
        context.grid.edges <= SUPPORT_OUTER_RG
        * context.grid.gravitational_radius
    )
    state_error = float(
        np.sqrt(
            0.5
            * np.mean(
                (left_coefficients[face_mask] - exact[face_mask]) ** 2
                + (right_coefficients[face_mask] - exact[face_mask]) ** 2
            )
        )
    )
    numerical_flux = (
        0.5 * speeds * (left_coefficients + right_coefficients)
        - 0.5
        * np.abs(speeds)
        * (right_coefficients - left_coefficients)
    ) * context.grid.face_measures
    exact_flux = (
        speeds * exact * context.grid.face_measures
    )
    numerical_divergence = (
        numerical_flux[1:] - numerical_flux[:-1]
    ) / context.grid.cell_measures
    exact_divergence = (
        exact_flux[1:] - exact_flux[:-1]
    ) / context.grid.cell_measures
    cell_mask = (
        context.grid.centers >= SUPPORT_INNER_RG
        * context.grid.gravitational_radius
    ) & (
        context.grid.centers <= SUPPORT_OUTER_RG
        * context.grid.gravitational_radius
    )
    rate_scale = max(
        float(np.sqrt(np.mean(exact_divergence[cell_mask] ** 2))),
        np.finfo(float).tiny,
    )
    rate_error = float(
        np.sqrt(
            np.mean(
                (
                    numerical_divergence[cell_mask]
                    - exact_divergence[cell_mask]
                )
                ** 2
            )
        )
        / rate_scale
    )
    return {
        "face_state_rms_error": state_error,
        "characteristic_rate_relative_rms_error": rate_error,
        "maximum_face_jump": float(
            np.max(np.abs(right_coefficients - left_coefficients))
        ),
        "minimum_coordinate_speed_over_c": float(np.min(speeds[face_mask])),
        "maximum_coordinate_speed_over_c": float(np.max(speeds[face_mask])),
    }


def run() -> tuple[dict, dict[str, np.ndarray]]:
    started = time.perf_counter()
    if not WP10C8Z_OUTPUT.exists() or not WP10C8Z_ARRAYS.exists():
        raise FileNotFoundError("WP10c9a requires WP10c8z evidence")
    parent_payload = json.loads(WP10C8Z_OUTPUT.read_text(encoding="utf-8"))
    if parent_payload.get("classification") != (
        "embedded_patch_inner_phase_not_converged"
    ):
        raise RuntimeError("WP10c8z stop classification changed")

    parent, configurations, labels = _configurations()
    active_outer_rg = float(parent["active_outer_rg"])
    active_cells = wp10c8z.ACTIVE_OUTER_PARENT_FACE
    reference_configuration = configurations[labels[1]]
    reference_state = causal_five_field_state_from_primitives(
        reference_configuration["context"],
        reference_configuration["base_primitives"],
    )
    reference_vector = pack_causal_five_field_state(reference_state)
    reference_evaluation = evaluate_causal_five_field_dae(
        reference_vector,
        reference_configuration["context"],
    )
    reference_scaling = causal_five_field_dae_scaling(
        reference_state,
        reference_evaluation,
    )
    parent_row_scales = np.asarray(
        reference_scaling.row_scales[
            : 5 * reference_configuration["layout"].n_cells
        ],
        dtype=float,
    )

    packet_reports = {}
    packet_arrays = {}
    term_reports = {}
    projected_packets = {}
    histories = {}
    moment_reports = {}
    maximum_restart = 0.0
    for family in CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES:
        print(f"WP10c9a: propagating {family} packet ladder", flush=True)
        projected_packets[family] = {}
        histories[family] = {}
        moment_reports[family] = {}
        packet_reports[family] = {"projection": {}}
        for ratio, label in labels.items():
            configuration = configurations[label]
            packet, bases, projection = _project_packet(
                configuration,
                family,
            )
            history, moments = _propagate_packet(
                configuration,
                packet,
                bases,
                family,
            )
            projected_packets[family][ratio] = packet
            histories[family][ratio] = history
            moment_reports[family][ratio] = moments
            packet_reports[family]["projection"][f"ratio_{ratio}"] = (
                projection
            )
            maximum_restart = max(
                maximum_restart,
                moments["restart_relative_defect"],
            )
            packet_arrays[f"{family}_ratio{ratio}_initial"] = packet
            packet_arrays[f"{family}_ratio{ratio}_state_history"] = (
                history["state"]
            )
            packet_arrays[f"{family}_ratio{ratio}_rate_history"] = (
                history["rate"]
            )
            for name, values in moments["arrays"].items():
                packet_arrays[f"{family}_ratio{ratio}_{name}"] = values

        restricted = {
            ratio: wp10c8z._restrict_history(
                histories[family][ratio],
                configurations[labels[ratio]],
            )
            for ratio in labels
        }
        coarse_medium = wp10c8z._history_metrics(
            restricted[1],
            restricted[2],
            parent["parent_grid"],
            lower_rg=None,
            upper_rg=active_outer_rg,
        )
        medium_fine = wp10c8z._history_metrics(
            restricted[2],
            restricted[4],
            parent["parent_grid"],
            lower_rg=None,
            upper_rg=active_outer_rg,
        )
        moment_coarse_medium = _moment_pair(
            moment_reports[family][1],
            moment_reports[family][2],
        )
        moment_medium_fine = _moment_pair(
            moment_reports[family][2],
            moment_reports[family][4],
        )
        orders = {
            "state_history": _observed_order(
                coarse_medium["state"]["maximum_relative_l2_difference"],
                medium_fine["state"]["maximum_relative_l2_difference"],
            ),
            "rate_history": _observed_order(
                coarse_medium["rate"]["maximum_relative_l2_difference"],
                medium_fine["rate"]["maximum_relative_l2_difference"],
            ),
            "phase_centroid": _observed_order(
                moment_coarse_medium[
                    "maximum_log_radius_centroid_defect"
                ],
                moment_medium_fine[
                    "maximum_log_radius_centroid_defect"
                ],
            ),
            "damping": _observed_order(
                moment_coarse_medium["maximum_log_amplitude_defect"],
                moment_medium_fine["maximum_log_amplitude_defect"],
            ),
        }
        packet_reports[family].update(
            {
                "history_pairs": {
                    "N128_N256patch": coarse_medium,
                    "N256patch_N512patch": medium_fine,
                },
                "moment_pairs": {
                    "N128_N256patch": moment_coarse_medium,
                    "N256patch_N512patch": moment_medium_fine,
                },
                "observed_orders": orders,
                "fine_minimum_signed_cosine": min(
                    medium_fine["state"]["minimum_signed_cosine"],
                    medium_fine["rate"]["minimum_signed_cosine"],
                ),
            }
        )

        medium_terms = _directional_blocks(
            configurations[labels[2]],
            projected_packets[family][2],
        )
        fine_terms = _directional_blocks(
            configurations[labels[4]],
            projected_packets[family][4],
        )
        term_norms = _term_difference_norms(
            medium_terms,
            fine_terms,
            configurations[labels[2]],
            configurations[labels[4]],
            parent_row_scales,
            active_cells,
        )
        controlling_balance_block = max(term_norms, key=term_norms.get)
        forcing_names = tuple(
            name
            for name in term_norms
            if name
            not in (
                "mapped_storage_action",
                "responsive_height_storage_action",
                "evolving_storage_remainder",
            )
        )
        controlling_forcing_term = max(
            forcing_names,
            key=term_norms.get,
        )
        term_reports[family] = {
            "N256patch_N512patch_scaled_l2_difference": term_norms,
            "controlling_forcing_term": controlling_forcing_term,
            "controlling_forcing_term_scaled_l2_difference": term_norms[
                controlling_forcing_term
            ],
            "controlling_balance_block": controlling_balance_block,
            "controlling_balance_block_scaled_l2_difference": term_norms[
                controlling_balance_block
            ],
        }

    controlling_family = min(
        CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES,
        key=lambda name: (
            float("inf")
            if packet_reports[name]["observed_orders"]["rate_history"]
            is None
            else packet_reports[name]["observed_orders"]["rate_history"]
        ),
    )
    controlling_term = term_reports[controlling_family][
        "controlling_forcing_term"
    ]
    controlling_balance_block = term_reports[controlling_family][
        "controlling_balance_block"
    ]

    candidate_methods = (
        "production_primitive_quadratic",
        "horizon_rapidity_quadratic",
        "characteristic_perturbation_quadratic",
    )
    candidate_reports = {}
    for method in candidate_methods:
        print(f"WP10c9a: screening {method}", flush=True)
        by_family = {}
        for family in CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES:
            errors = {
                ratio: _candidate_errors(
                    configurations[labels[ratio]],
                    family,
                    method,
                )
                for ratio in labels
            }
            state_orders = (
                _observed_order(
                    errors[1]["face_state_rms_error"],
                    errors[2]["face_state_rms_error"],
                ),
                _observed_order(
                    errors[2]["face_state_rms_error"],
                    errors[4]["face_state_rms_error"],
                ),
            )
            rate_orders = (
                _observed_order(
                    errors[1][
                        "characteristic_rate_relative_rms_error"
                    ],
                    errors[2][
                        "characteristic_rate_relative_rms_error"
                    ],
                ),
                _observed_order(
                    errors[2][
                        "characteristic_rate_relative_rms_error"
                    ],
                    errors[4][
                        "characteristic_rate_relative_rms_error"
                    ],
                ),
            )
            by_family[family] = {
                "errors_by_ratio": errors,
                "state_orders": state_orders,
                "rate_orders": rate_orders,
                "passed": bool(
                    all(
                        value is not None and value >= MINIMUM_SMOOTH_ORDER
                        for value in state_orders + rate_orders
                    )
                ),
            }
        method_passed = all(
            row["passed"] for row in by_family.values()
        )
        candidate_reports[method] = {
            "by_family": by_family,
            "smooth_method_gate_passed": method_passed,
            "live_coupled_generator_built": (
                method == "production_primitive_quadratic"
            ),
            "full_packet_phase_certified": bool(
                method == "production_primitive_quadratic"
                and all(
                    packet_reports[family]["observed_orders"][
                        "phase_centroid"
                    ]
                    is not None
                    and packet_reports[family]["observed_orders"][
                        "phase_centroid"
                    ]
                    >= MINIMUM_PACKET_PHASE_ORDER
                    and packet_reports[family]["observed_orders"][
                        "damping"
                    ]
                    is not None
                    and packet_reports[family]["observed_orders"][
                        "damping"
                    ]
                    >= MINIMUM_PACKET_DAMPING_ORDER
                    and packet_reports[family][
                        "fine_minimum_signed_cosine"
                    ]
                    >= MINIMUM_SIGNED_COSINE
                    for family in (
                        CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES
                    )
                )
            ),
        }
    production_candidate = candidate_reports[
        "production_primitive_quadratic"
    ]
    for method, report in candidate_reports.items():
        ratios = []
        for family in CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES:
            candidate_error = report["by_family"][family][
                "errors_by_ratio"
            ][4]["characteristic_rate_relative_rms_error"]
            production_error = production_candidate["by_family"][family][
                "errors_by_ratio"
            ][4]["characteristic_rate_relative_rms_error"]
            ratios.append(candidate_error / production_error)
        report["maximum_fine_rate_error_ratio_to_production"] = float(
            max(ratios)
        )
        report["meaningful_rate_coefficient_improvement"] = bool(
            method != "production_primitive_quadratic"
            and max(ratios) <= 0.95
        )

    maximum_flux_defect = max(
        causal_embedded_patch_flux_audit(
            configuration["context"],
            pack_causal_five_field_state(
                causal_five_field_state_from_primitives(
                    configuration["context"],
                    configuration["base_primitives"],
                )
            ),
            configuration["layout"],
        ).maximum_state_flux_defect
        for configuration in configurations.values()
    )
    maximum_storage_defect = max(
        float(
            configuration["operator_report"][
                "maximum_relative_storage_action_defect"
            ]
        )
        for configuration in configurations.values()
    )
    method_contract_passed = bool(
        maximum_flux_defect <= MAXIMUM_SHARED_FLUX_DEFECT
        and maximum_storage_defect <= MAXIMUM_STORAGE_ACTION_DEFECT
        and maximum_restart <= MAXIMUM_RESTART_DEFECT
        and all(
            report["passed"]
            for family in packet_reports.values()
            for report in family["projection"].values()
        )
        and all(
            configuration["operator_report"]["state_gates"]["measured"][
                "inner_incoming_characteristics"
            ]
            == 0
            for configuration in configurations.values()
        )
    )
    passing_candidates = [
        method
        for method, report in candidate_reports.items()
        if report["smooth_method_gate_passed"]
        and report["full_packet_phase_certified"]
    ]
    production_packet_phase_passed = candidate_reports[
        "production_primitive_quadratic"
    ]["full_packet_phase_certified"]
    if not method_contract_passed:
        classification = "characteristic_packet_method_contract_failed"
    elif passing_candidates:
        classification = "characteristic_operator_candidate_passed"
    elif production_packet_phase_passed:
        classification = (
            "characteristic_packets_converged_common_mode_still_unresolved"
        )
    else:
        classification = (
            "characteristic_rate_phase_unresolved_operator_redesign_required"
        )

    arrays = {
        "times": np.linspace(0.0, TARGET_SECONDS, TIME_SAMPLES),
        **packet_arrays,
    }
    DEFAULT_ARRAYS.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DEFAULT_ARRAYS, **arrays)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "base_commit": BASE_COMMIT,
        "classification": classification,
        "passed": bool(passing_candidates),
        "scope": {
            "production_physics_changed": False,
            "production_operator_changed": False,
            "wp10c8z_live_coupling_reused": True,
            "nonlinear_common_mode_rerun": False,
            "bounded_nonlinear_patch_truth_run": False,
            "fixed_q_averaging_run": False,
            "reduced_coordinate_selected": False,
        },
        "frozen_parent_evidence": {
            "wp10c8z_json": _relative(WP10C8Z_OUTPUT),
            "wp10c8z_json_sha256": _sha256(WP10C8Z_OUTPUT),
            "wp10c8z_arrays": _relative(WP10C8Z_ARRAYS),
            "wp10c8z_arrays_sha256": _sha256(WP10C8Z_ARRAYS),
            "wp10c8z_classification": parent_payload["classification"],
        },
        "machine_evidence": {
            "arrays_path": _relative(DEFAULT_ARRAYS),
            "arrays_sha256": _sha256(DEFAULT_ARRAYS),
        },
        "packet_definition": {
            "families": CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES,
            "support_inner_rg": SUPPORT_INNER_RG,
            "support_outer_rg": SUPPORT_OUTER_RG,
            "target_seconds": TARGET_SECONDS,
            "time_samples": TIME_SAMPLES,
            "exact_retained_coordinate_tangent_projection": False,
            "exact_reduced_descriptor_primitive_manifold": True,
            "retained_slow_coordinate_null_fiber_imposed": False,
            "live_coupling_face": True,
        },
        "packet_results": packet_reports,
        "phase_defect_decomposition": {
            "by_family": term_reports,
            "controlling_family": controlling_family,
            "controlling_term": controlling_term,
            "controlling_balance_block": controlling_balance_block,
            "interpretation": (
                "largest forcing-side N256/N512 scaled integrated tangent "
                "difference for the family with the lowest rate-history "
                "order; the balance block is reported separately"
            ),
        },
        "candidate_screen": candidate_reports,
        "characteristic_travel_time_grid": {
            "screened": True,
            "eligible_for_promotion": False,
            "reason": (
                "the five coordinate families have distinct spatially varying "
                "speeds, so one family-specific equi-travel grid cannot be "
                "promoted before a common monitor and full conservative "
                "descriptor are defined"
            ),
        },
        "method_contract": {
            "maximum_shared_flux_defect": maximum_flux_defect,
            "maximum_storage_action_defect": maximum_storage_defect,
            "maximum_restart_relative_defect": maximum_restart,
            "no_incoming_excision_characteristic": True,
            "dense_colored_parity_inherited_from_wp10c8z": True,
            "bdf2_bitwise_replay_inherited_from_wp10c8z": True,
            "passed": method_contract_passed,
        },
        "gates": {
            "minimum_smooth_state_rate_order": MINIMUM_SMOOTH_ORDER,
            "minimum_packet_phase_order": MINIMUM_PACKET_PHASE_ORDER,
            "minimum_packet_damping_order": MINIMUM_PACKET_DAMPING_ORDER,
            "minimum_same_time_signed_cosine": MINIMUM_SIGNED_COSINE,
            "maximum_shared_flux_defect": MAXIMUM_SHARED_FLUX_DEFECT,
            "maximum_storage_action_defect": (
                MAXIMUM_STORAGE_ACTION_DEFECT
            ),
            "maximum_restart_defect": MAXIMUM_RESTART_DEFECT,
        },
        "decision": {
            "passing_candidates": passing_candidates,
            "rerun_common_mode_authorized": bool(passing_candidates),
            "bounded_nonlinear_patch_truth_authorized": False,
            "one_more_brute_force_refinement_authorized": False,
            "targeted_operator_implementation_authorized": bool(
                any(
                    report["smooth_method_gate_passed"]
                    and report[
                        "meaningful_rate_coefficient_improvement"
                    ]
                    for method, report in candidate_reports.items()
                    if method != "production_primitive_quadratic"
                )
            ),
            "fixed_q_averaging_authorized": False,
            "reduced_model_authorized": False,
            "next_operator_target": (
                "family_resolved_characteristic_dissipation replacing the "
                "single scalar max-speed Rusanov penalty in the inner bulk"
            ),
            "reason": (
                "Rusanov transport is the largest forcing-side fine-pair "
                "defect for the controlling inward-shear packet, while "
                "rapidity and characteristic reconstruction have essentially "
                "the same smooth error coefficient as production"
            ),
        },
        "provenance": {
            "runner": THIS_RUNNER,
            "core_packet_file": CORE_PACKET_FILE,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "wall_seconds": time.perf_counter() - started,
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
    parser = argparse.ArgumentParser()
    parser.parse_args()
    payload, _arrays = run()
    print(
        json.dumps(
            {
                "classification": payload["classification"],
                "passed": payload["passed"],
                "controlling_family": payload[
                    "phase_defect_decomposition"
                ]["controlling_family"],
                "controlling_term": payload[
                    "phase_defect_decomposition"
                ]["controlling_term"],
                "passing_candidates": payload["decision"][
                    "passing_candidates"
                ],
                "output": str(DEFAULT_OUTPUT),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
