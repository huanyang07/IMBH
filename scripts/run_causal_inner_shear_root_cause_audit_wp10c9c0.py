"""Run the WP10c9c0 causal-shear root-cause audit.

This package is deliberately diagnostic.  It derives the implemented
sign-explicit principal pencil, certifies the two-family shear energy, compares
the current split and a monolithic complete-principal reference in Fourier and
manufactured-wave tests, and finally applies only the measured monolithic
principal correction to the unchanged full frozen generator.  It does not
implement a nonlinear path flux or change a production default.
"""

from __future__ import annotations

import argparse
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

import run_causal_inner_characteristic_phase_audit_wp10c9a as wp10c9a
import run_causal_inner_embedded_patch_preflight_wp10c8z as wp10c8z

from imri_qpe.constants import C  # noqa: E402
from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    causal_five_field_coordinate_principal_basis,
    causal_five_field_coordinate_principal_components,
    causal_five_field_frozen_principal_generator,
    causal_five_field_lower_stress_relaxation_matrix,
    causal_five_field_manufactured_principal_wave,
    causal_five_field_principal_step_defects,
    causal_five_field_shear_fourier_symbols,
    causal_five_field_shear_invariant_subspace,
    causal_five_field_straight_principal_path_jump,
)


BASE_COMMIT = "90f82c238e802abe22aa15b42f62b7d929048a60"
WORK_PACKAGE = "WP10c9c0"
SCHEMA_VERSION = 1
THIS_RUNNER = (
    "scripts/run_causal_inner_shear_root_cause_audit_wp10c9c0.py"
)
CORE_FILES = (
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_characteristic_dissipation.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_shear_root_cause.py",
)
WP10C9A_OUTPUT = (
    ROOT
    / "outputs/tables/"
    "causal_inner_characteristic_phase_audit_wp10c9a.json"
)
WP10C9A_ARRAYS = (
    ROOT
    / "outputs/tables/"
    "causal_inner_characteristic_phase_audit_wp10c9a_arrays.npz"
)
WP10C9B_OUTPUT = (
    ROOT
    / "outputs/tables/"
    "causal_inner_characteristic_dissipation_audit_wp10c9b.json"
)
WP10C9B_ARRAYS = (
    ROOT
    / "outputs/tables/"
    "causal_inner_characteristic_dissipation_audit_wp10c9b_arrays.npz"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/tables/"
    "causal_inner_shear_root_cause_audit_wp10c9c0.json"
)
DEFAULT_ARRAYS = (
    ROOT
    / "outputs/tables/"
    "causal_inner_shear_root_cause_audit_wp10c9c0_arrays.npz"
)

REPRESENTATIVE_RADII_RG = (1.90, 2.20, 3.00, 5.00, 6.40)
DERIVATIVE_RELATIVE_STEPS = (5.0e-5, 1.0e-4, 2.0e-4, 4.0e-4)
FOURIER_THETA_COARSE = (0.05, 0.10, 0.20, 0.40, 0.80)
FOURIER_BINDING_MAXIMUM_THETA = 0.40
SHEAR_FAMILIES = ("inward_shear", "outward_shear")
MANUFACTURED_SUPPORT_RG = (2.15, 5.40)

MAXIMUM_PATH_LINEARIZATION_DEFECT = 2.0e-8
MAXIMUM_PATH_REVERSAL_DEFECT = 1.0e-12
MAXIMUM_DERIVATIVE_PLATEAU_DEFECT = 2.0e-5
MAXIMUM_PROJECTOR_DEFECT = 2.0e-10
MAXIMUM_SYMMETRY_DEFECT = 1.0e-12
MINIMUM_LOCAL_ORDER = 1.8
MINIMUM_PACKET_PHASE_ORDER = 0.75
MINIMUM_PACKET_DAMPING_ORDER = 0.75
MINIMUM_SIGNED_COSINE = 0.90


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


def _common_chart(configuration: dict, radius_rg: float) -> np.ndarray:
    context = configuration["context"]
    function = wp10c8z._common_profile_function(
        context.grid.centers / context.grid.gravitational_radius,
        configuration["base_primitives"],
    )
    return np.asarray(
        function(np.asarray([float(radius_rg)], dtype=float))[0],
        dtype=float,
    )


def _sign_step_and_energy_audit(reference: dict) -> tuple[dict, dict]:
    context = reference["context"]
    gravitational_radius = context.grid.gravitational_radius
    direction_weights = np.asarray([0.31, -0.23, 0.17, 0.41, -0.29])
    by_radius = {}
    arrays = {}
    maximum_path = 0.0
    maximum_reversal = 0.0
    maximum_step = 0.0
    maximum_speed_step = 0.0
    maximum_projector_step = 0.0
    maximum_projector = 0.0
    maximum_symmetry = 0.0
    maximum_analytic_projector = 0.0
    maximum_analytic_eigenpair = 0.0
    minimum_energy = float("inf")
    maximum_energy_condition = 0.0
    for radius_rg in REPRESENTATIVE_RADII_RG:
        radius = radius_rg * gravitational_radius
        chart = _common_chart(reference, radius_rg)
        components = tuple(
            causal_five_field_coordinate_principal_components(
                context,
                radius,
                chart,
                relative_step=step,
            )
            for step in DERIVATIVE_RELATIVE_STEPS
        )
        plateau = causal_five_field_principal_step_defects(components)
        step_defect = max(plateau.values())
        maximum_step = max(maximum_step, step_defect)
        central = components[2]
        direction = (
            direction_weights * central.primitive_column_scales
        )
        direction /= np.linalg.norm(direction_weights)
        epsilon = 1.0e-6
        left = chart - 0.5 * epsilon * direction
        right = chart + 0.5 * epsilon * direction
        forward = causal_five_field_straight_principal_path_jump(
            context,
            radius,
            left,
            right,
        )
        reverse = causal_five_field_straight_principal_path_jump(
            context,
            radius,
            right,
            left,
        )
        linear = (
            epsilon * central.spatial_principal_matrix @ direction
        )
        scale = max(
            float(np.max(np.abs(forward))),
            float(np.max(np.abs(linear))),
            np.finfo(float).tiny,
        )
        path_defect = float(np.max(np.abs(forward - linear)) / scale)
        reversal_defect = float(
            np.max(np.abs(forward + reverse)) / scale
        )
        maximum_path = max(maximum_path, path_defect)
        maximum_reversal = max(maximum_reversal, reversal_defect)
        shear_steps = tuple(
            causal_five_field_shear_invariant_subspace(
                context,
                radius,
                chart,
                relative_step=step,
            )
            for step in DERIVATIVE_RELATIVE_STEPS
        )
        shear = shear_steps[2]
        speed_step_defect = max(
            float(
                np.max(
                    np.abs(
                        first.coordinate_speeds_over_c
                        - second.coordinate_speeds_over_c
                    )
                )
            )
            for first, second in zip(
                shear_steps[:-1],
                shear_steps[1:],
                strict=True,
            )
        )
        projector_step_defect = max(
            float(
                np.max(
                    np.abs(
                        first.primitive_projector
                        - second.primitive_projector
                    )
                )
                / max(
                    np.max(np.abs(first.primitive_projector)),
                    np.max(np.abs(second.primitive_projector)),
                    np.finfo(float).tiny,
                )
            )
            for first, second in zip(
                shear_steps[:-1],
                shear_steps[1:],
                strict=True,
            )
        )
        maximum_speed_step = max(
            maximum_speed_step,
            speed_step_defect,
        )
        maximum_projector_step = max(
            maximum_projector_step,
            projector_step_defect,
        )
        projector = max(
            shear.maximum_projector_idempotence_defect,
            shear.maximum_projector_complement_defect,
        )
        minimum = min(
            shear.minimum_local_rest_energy_eigenvalue,
            shear.minimum_coordinate_energy_eigenvalue,
        )
        maximum_projector = max(maximum_projector, projector)
        maximum_symmetry = max(
            maximum_symmetry,
            shear.maximum_local_rest_symmetry_defect,
        )
        maximum_analytic_projector = max(
            maximum_analytic_projector,
            shear.maximum_analytic_local_projector_defect,
        )
        maximum_analytic_eigenpair = max(
            maximum_analytic_eigenpair,
            shear.maximum_analytic_local_eigenpair_defect,
        )
        minimum_energy = min(minimum_energy, minimum)
        maximum_energy_condition = max(
            maximum_energy_condition,
            shear.coordinate_energy_condition_number,
        )
        key = f"radius_{radius_rg:.2f}rg"
        by_radius[key] = {
            "radius_rg": radius_rg,
            "implemented_sign_convention": (
                "B_equals_F_p_minus_C_pr"
            ),
            "maximum_path_small_jump_defect": path_defect,
            "path_reversal_defect": reversal_defect,
            "derivative_step_defects": plateau,
            "maximum_derivative_step_defect": step_defect,
            "maximum_shear_speed_step_defect_over_c": (
                speed_step_defect
            ),
            "maximum_shear_projector_step_defect": (
                projector_step_defect
            ),
            "coordinate_shear_speeds_over_c": (
                shear.coordinate_speeds_over_c
            ),
            "maximum_projector_defect": projector,
            "maximum_symmetrizer_defect": (
                shear.maximum_local_rest_symmetry_defect
            ),
            "maximum_analytic_local_projector_defect": (
                shear.maximum_analytic_local_projector_defect
            ),
            "maximum_analytic_local_eigenpair_defect": (
                shear.maximum_analytic_local_eigenpair_defect
            ),
            "analytic_local_rest_speeds_over_c": (
                shear.analytic_local_rest_speeds_over_c
            ),
            "minimum_local_rest_energy_eigenvalue": (
                shear.minimum_local_rest_energy_eigenvalue
            ),
            "minimum_coordinate_energy_eigenvalue": (
                shear.minimum_coordinate_energy_eigenvalue
            ),
            "coordinate_energy_condition_number": (
                shear.coordinate_energy_condition_number
            ),
            "descriptor_condition_number": (
                shear.coordinate_basis.descriptor_condition_number
            ),
            "shear_spectral_gap_over_c": float(
                np.diff(shear.coordinate_speeds_over_c)[0]
            ),
        }
        arrays[f"{key}_local_rest_symmetrizer"] = (
            shear.local_rest_symmetrizer
        )
        arrays[f"{key}_analytic_local_rest_projectors"] = (
            shear.analytic_local_rest_projectors
        )
        arrays[f"{key}_coordinate_energy_gram"] = (
            shear.coordinate_energy_gram
        )
        arrays[f"{key}_primitive_projector"] = (
            shear.primitive_projector
        )
    passed = bool(
        maximum_path <= MAXIMUM_PATH_LINEARIZATION_DEFECT
        and maximum_reversal <= MAXIMUM_PATH_REVERSAL_DEFECT
        and maximum_step <= MAXIMUM_DERIVATIVE_PLATEAU_DEFECT
        and maximum_speed_step <= MAXIMUM_DERIVATIVE_PLATEAU_DEFECT
        and maximum_projector_step <= MAXIMUM_DERIVATIVE_PLATEAU_DEFECT
        and maximum_projector <= MAXIMUM_PROJECTOR_DEFECT
        and maximum_symmetry <= MAXIMUM_SYMMETRY_DEFECT
        and maximum_analytic_projector <= MAXIMUM_PROJECTOR_DEFECT
        and maximum_analytic_eigenpair <= MAXIMUM_PROJECTOR_DEFECT
        and minimum_energy > 0.0
        and np.isfinite(maximum_energy_condition)
    )
    return {
        "by_radius": by_radius,
        "maximum_path_small_jump_defect": maximum_path,
        "maximum_path_reversal_defect": maximum_reversal,
        "maximum_derivative_step_defect": maximum_step,
        "maximum_shear_speed_step_defect_over_c": maximum_speed_step,
        "maximum_shear_projector_step_defect": maximum_projector_step,
        "maximum_projector_defect": maximum_projector,
        "maximum_symmetrizer_defect": maximum_symmetry,
        "maximum_analytic_local_projector_defect": (
            maximum_analytic_projector
        ),
        "maximum_analytic_local_eigenpair_defect": (
            maximum_analytic_eigenpair
        ),
        "minimum_energy_eigenvalue": minimum_energy,
        "maximum_coordinate_energy_condition_number": (
            maximum_energy_condition
        ),
        "passed": passed,
    }, arrays


def _nearest_unique(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    remaining = list(range(values.size))
    selected = []
    for target in targets:
        index = min(remaining, key=lambda item: abs(values[item] - target))
        selected.append(index)
        remaining.remove(index)
    return np.asarray(selected, dtype=int)


def _shear_eigenvalues(
    matrix: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    values = np.linalg.eigvals(np.asarray(matrix))
    return values[_nearest_unique(values, targets)]


def _fourier_audit(reference: dict) -> tuple[dict, dict]:
    context = reference["context"]
    rg = context.grid.gravitational_radius
    arrays = {}
    by_radius = {}
    minimum_orders = {
        "current_split_phase": float("inf"),
        "current_split_damping": float("inf"),
        "monolithic_phase": float("inf"),
        "monolithic_damping": float("inf"),
        "current_split_relaxing": float("inf"),
        "monolithic_relaxing": float("inf"),
    }
    for radius_rg in REPRESENTATIVE_RADII_RG:
        radius = radius_rg * rg
        chart = _common_chart(reference, radius_rg)
        components = causal_five_field_coordinate_principal_components(
            context,
            radius,
            chart,
        )
        basis = causal_five_field_coordinate_principal_basis(
            context,
            radius,
            chart,
        )
        lower = causal_five_field_lower_stress_relaxation_matrix(
            context,
            radius,
            chart,
        )
        cell = int(np.argmin(np.abs(context.grid.centers - radius)))
        coarse_spacing = float(np.diff(context.grid.edges)[cell])
        speed = basis.numerical_speeds_over_c[[1, 3]]
        rows = []
        for theta_coarse in FOURIER_THETA_COARSE:
            wavenumber = theta_coarse / coarse_spacing
            continuum_targets = -1.0j * C * wavenumber * speed
            level = {}
            for ratio in (1, 2, 4):
                symbols = causal_five_field_shear_fourier_symbols(
                    components,
                    basis,
                    lower,
                    theta=theta_coarse / ratio,
                    spacing=coarse_spacing / ratio,
                )
                continuum_principal = _shear_eigenvalues(
                    symbols.continuum_principal_per_s,
                    continuum_targets,
                )
                continuum_relaxing = _shear_eigenvalues(
                    symbols.continuum_relaxing_per_s,
                    continuum_principal,
                )
                row = {}
                for name, matrix in (
                    (
                        "current_split",
                        symbols.current_split_principal_per_s,
                    ),
                    (
                        "monolithic",
                        symbols.monolithic_principal_per_s,
                    ),
                ):
                    values = _shear_eigenvalues(
                        matrix,
                        continuum_principal,
                    )
                    row[f"{name}_phase_error"] = float(
                        np.max(
                            np.abs(
                                -np.imag(values) / (C * wavenumber)
                                - speed
                            )
                        )
                    )
                    row[f"{name}_damping_error_per_s"] = float(
                        np.max(np.abs(np.real(values)))
                    )
                for name, matrix in (
                    (
                        "current_split",
                        symbols.current_split_relaxing_per_s,
                    ),
                    (
                        "monolithic",
                        symbols.monolithic_relaxing_per_s,
                    ),
                ):
                    values = _shear_eigenvalues(
                        matrix,
                        continuum_relaxing,
                    )
                    row[f"{name}_relaxing_eigenvalue_error_per_s"] = (
                        float(np.max(np.abs(values - continuum_relaxing)))
                    )
                row["physical_flux_only_symbol_norm_per_s"] = float(
                    np.linalg.norm(
                        symbols.physical_flux_only_principal_per_s
                    )
                )
                row["principal_source_only_symbol_norm_per_s"] = float(
                    np.linalg.norm(
                        symbols.principal_source_only_per_s
                    )
                )
                row["current_split_characteristic_penalty_norm_per_s"] = (
                    float(
                        np.linalg.norm(
                            symbols.current_split_principal_per_s
                            - (
                                symbols.physical_flux_only_principal_per_s
                                + symbols.principal_source_only_per_s
                            )
                        )
                    )
                )
                if ratio == 1:
                    prefix = (
                        f"radius_{radius_rg:.2f}rg_"
                        f"theta_{theta_coarse:.2f}"
                    )
                    arrays[f"{prefix}_continuum_principal"] = (
                        symbols.continuum_principal_per_s
                    )
                    arrays[f"{prefix}_current_split_principal"] = (
                        symbols.current_split_principal_per_s
                    )
                    arrays[f"{prefix}_monolithic_principal"] = (
                        symbols.monolithic_principal_per_s
                    )
                    arrays[f"{prefix}_physical_flux_only"] = (
                        symbols.physical_flux_only_principal_per_s
                    )
                    arrays[f"{prefix}_principal_source_only"] = (
                        symbols.principal_source_only_per_s
                    )
                level[ratio] = row
            orders = {}
            for name in ("current_split", "monolithic"):
                for measure, suffix in (
                    ("phase", "phase_error"),
                    ("damping", "damping_error_per_s"),
                    (
                        "relaxing",
                        "relaxing_eigenvalue_error_per_s",
                    ),
                ):
                    key = f"{name}_{measure}"
                    field = f"{name}_{suffix}"
                    pair_orders = (
                        _observed_order(
                            level[1][field],
                            level[2][field],
                        ),
                        _observed_order(
                            level[2][field],
                            level[4][field],
                        ),
                    )
                    valid = tuple(
                        value for value in pair_orders if value is not None
                    )
                    orders[key] = min(valid) if valid else None
                    if (
                        valid
                        and theta_coarse
                        <= FOURIER_BINDING_MAXIMUM_THETA
                    ):
                        minimum_orders[key] = min(
                            minimum_orders[key],
                            min(valid),
                        )
            rows.append(
                {
                    "theta_on_coarse_grid": theta_coarse,
                    "physical_wavenumber": wavenumber,
                    "by_ratio": level,
                    "minimum_pair_orders": orders,
                }
            )
        key = f"radius_{radius_rg:.2f}rg"
        by_radius[key] = {
            "radius_rg": radius_rg,
            "coordinate_shear_speeds_over_c": speed,
            "coarse_spacing": coarse_spacing,
            "wavenumber_rows": rows,
        }
        arrays[f"{key}_lower_relaxation_matrix"] = lower
    passed = bool(
        all(value >= MINIMUM_LOCAL_ORDER for value in minimum_orders.values())
    )
    return {
        "by_radius": by_radius,
        "minimum_observed_orders": minimum_orders,
        "principal_only_and_physical_relaxation_separated": True,
        "conservative_and_principal_source_ablations_available": True,
        "binding_maximum_theta_on_coarse_grid": (
            FOURIER_BINDING_MAXIMUM_THETA
        ),
        "theta_0p8_is_high_wavenumber_diagnostic_only": True,
        "passed": passed,
    }, arrays


def _manufactured_audit(
    configurations: dict[int, dict],
) -> tuple[dict, dict]:
    reference = configurations[1]
    context = reference["context"]
    chart = _common_chart(reference, 3.0)
    components = causal_five_field_coordinate_principal_components(
        context,
        3.0 * context.grid.gravitational_radius,
        chart,
    )
    weights = np.asarray([0.31, -0.23, 0.17, 0.41, -0.29])
    direction = weights * components.primitive_column_scales
    direction /= np.linalg.norm(weights)
    by_ratio = {}
    arrays = {}
    for ratio, configuration in configurations.items():
        local = configuration["context"]
        audit = causal_five_field_manufactured_principal_wave(
            local,
            configuration["base_primitives"],
            direction,
            support_inner_radius=(
                MANUFACTURED_SUPPORT_RG[0]
                * local.grid.gravitational_radius
            ),
            support_outer_radius=(
                MANUFACTURED_SUPPORT_RG[1]
                * local.grid.gravitational_radius
            ),
        )
        by_ratio[ratio] = {
            "current_split_relative_l2_error": (
                audit.current_split_relative_l2_error
            ),
            "monolithic_relative_l2_error": (
                audit.monolithic_relative_l2_error
            ),
        }
        arrays[f"ratio{ratio}_manufactured_exact"] = (
            audit.exact_complete_principal
        )
        arrays[f"ratio{ratio}_manufactured_split"] = (
            audit.current_split_principal
        )
        arrays[f"ratio{ratio}_manufactured_monolithic"] = (
            audit.monolithic_principal
        )
    orders = {}
    for name in ("current_split", "monolithic"):
        field = f"{name}_relative_l2_error"
        orders[name] = {
            "ratio1_to_ratio2": _observed_order(
                by_ratio[1][field],
                by_ratio[2][field],
            ),
            "ratio2_to_ratio4": _observed_order(
                by_ratio[2][field],
                by_ratio[4][field],
            ),
        }
    minimum = min(
        value
        for row in orders.values()
        for value in row.values()
        if value is not None
    )
    return {
        "support_rg": MANUFACTURED_SUPPORT_RG,
        "by_ratio": by_ratio,
        "observed_orders": orders,
        "minimum_observed_order": minimum,
        "passed": bool(minimum >= MINIMUM_LOCAL_ORDER),
    }, arrays


def _shear_energy_history(
    configuration: dict,
    state_history: np.ndarray,
    family: str,
) -> dict[str, np.ndarray]:
    context = configuration["context"]
    amplitudes = np.asarray(configuration["amplitudes"], dtype=float)
    physical = np.asarray(state_history, dtype=float) * amplitudes[None, :, :]
    selected = 0 if family == "inward_shear" else 1
    grams = []
    lefts = []
    rights = []
    temporal = []
    row_scales = []
    for radius, chart in zip(
        context.grid.centers,
        configuration["base_primitives"],
        strict=True,
    ):
        shear = causal_five_field_shear_invariant_subspace(
            context,
            float(radius),
            chart,
        )
        grams.append(shear.coordinate_energy_gram)
        lefts.append(shear.primitive_left_eigenvectors)
        rights.append(shear.primitive_right_eigenvectors)
        temporal.append(shear.coordinate_basis.temporal_storage_matrix)
        row_scales.append(shear.coordinate_basis.descriptor_row_scales)
    grams = np.asarray(grams, dtype=float)
    lefts = np.asarray(lefts, dtype=float)
    rights = np.asarray(rights, dtype=float)
    temporal = np.asarray(temporal, dtype=float)
    row_scales = np.asarray(row_scales, dtype=float)
    coefficients = np.einsum("cij,tcj->tci", lefts, physical)
    total_density = 0.5 * np.einsum(
        "tci,cij,tcj->tc",
        coefficients,
        grams,
        coefficients,
    )
    selected_coefficients = np.zeros_like(coefficients)
    selected_coefficients[:, :, selected] = coefficients[:, :, selected]
    selected_density = 0.5 * np.einsum(
        "tci,cij,tcj->tc",
        selected_coefficients,
        grams,
        selected_coefficients,
    )
    measures = np.asarray(context.grid.cell_measures, dtype=float)
    total = total_density @ measures
    selected_energy = selected_density @ measures
    opposite_coefficients = np.zeros_like(coefficients)
    opposite_coefficients[:, :, 1 - selected] = (
        coefficients[:, :, 1 - selected]
    )
    opposite_density = 0.5 * np.einsum(
        "tci,cij,tcj->tc",
        opposite_coefficients,
        grams,
        opposite_coefficients,
    )
    opposite_energy = opposite_density @ measures
    cross_energy = (
        total_density - selected_density - opposite_density
    ) @ measures
    shear_primitive = np.einsum(
        "cij,tcj->tci",
        rights,
        coefficients,
    )
    descriptor = np.einsum(
        "cij,tcj->tci",
        temporal,
        shear_primitive,
    ) / row_scales[None, :, :]
    descriptor_squared = np.sum(descriptor**2, axis=2) @ measures
    branch_sum = np.maximum(
        selected_energy + opposite_energy,
        np.finfo(float).tiny,
    )
    selected_fraction = selected_energy / branch_sum
    opposite_fraction = opposite_energy / branch_sum
    support = (
        context.grid.centers
        >= wp10c9a.SUPPORT_INNER_RG * context.grid.gravitational_radius
    ) & (
        context.grid.centers
        <= wp10c9a.SUPPORT_OUTER_RG * context.grid.gravitational_radius
    )
    support_energy = total_density[:, support] @ measures[support]
    total /= max(float(total[0]), np.finfo(float).tiny)
    selected_energy /= max(
        float(selected_energy[0]),
        np.finfo(float).tiny,
    )
    descriptor_squared /= max(
        float(descriptor_squared[0]),
        np.finfo(float).tiny,
    )
    return {
        "normalized_total_shear_energy": total,
        "normalized_selected_branch_self_energy": selected_energy,
        "normalized_scaled_descriptor_shear_norm_squared": (
            descriptor_squared
        ),
        "selected_branch_self_energy_fraction_of_branch_sum": (
            selected_fraction
        ),
        "opposite_branch_self_energy_fraction_of_branch_sum": (
            opposite_fraction
        ),
        "branch_cross_energy_fraction": (
            cross_energy
            / np.maximum(
                total_density @ measures,
                np.finfo(float).tiny,
            )
        ),
        "measurement_window_energy_fraction": (
            support_energy
            / np.maximum(
                total_density @ measures,
                np.finfo(float).tiny,
            )
        ),
    }


def _energy_pair(first: np.ndarray, second: np.ndarray) -> float:
    return float(
        np.max(
            np.abs(
                np.log(
                    np.maximum(second, np.finfo(float).tiny)
                    / np.maximum(first, np.finfo(float).tiny)
                )
            )
        )
    )


def _energy_ladder_summary(
    energies: dict[int, dict[str, np.ndarray]],
) -> dict:
    defects = {}
    orders = {}
    for name in (
        "normalized_total_shear_energy",
        "normalized_selected_branch_self_energy",
        "normalized_scaled_descriptor_shear_norm_squared",
    ):
        coarse = _energy_pair(energies[1][name], energies[2][name])
        fine = _energy_pair(energies[2][name], energies[4][name])
        defects[name] = {
            "ratio1_ratio2": coarse,
            "ratio2_ratio4": fine,
        }
        orders[name] = _observed_order(coarse, fine)
    return {
        "defects": defects,
        "observed_orders": orders,
        "maximum_opposite_branch_self_energy_fraction_by_ratio": {
            ratio: float(
                np.max(
                    values[
                        "opposite_branch_self_energy_fraction_of_branch_sum"
                    ]
                )
            )
            for ratio, values in energies.items()
        },
        "minimum_measurement_window_energy_fraction_by_ratio": {
            ratio: float(
                np.min(values["measurement_window_energy_fraction"])
            )
            for ratio, values in energies.items()
        },
        "maximum_absolute_branch_cross_energy_fraction_by_ratio": {
            ratio: float(
                np.max(np.abs(values["branch_cross_energy_fraction"]))
            )
            for ratio, values in energies.items()
        },
    }


def _parent_physical_energy_audit(
    configurations: dict[int, dict],
) -> tuple[dict, dict]:
    reports = {}
    arrays = {}
    for label, path in (
        ("production_scalar_rusanov", WP10C9A_ARRAYS),
        ("wp10c9b_characteristic_matrix", WP10C9B_ARRAYS),
    ):
        with np.load(path) as payload:
            operator = {}
            for family in SHEAR_FAMILIES:
                energies = {}
                for ratio, configuration in configurations.items():
                    history = np.asarray(
                        payload[
                            f"{family}_ratio{ratio}_state_history"
                        ],
                        dtype=float,
                    )
                    energies[ratio] = _shear_energy_history(
                        configuration,
                        history,
                        family,
                    )
                    for name, values in energies[ratio].items():
                        arrays[
                            f"{label}_{family}_ratio{ratio}_{name}"
                        ] = values
                operator[family] = _energy_ladder_summary(energies)
            reports[label] = operator
    return reports, arrays


def _corrected_packet_audit(
    parent: dict,
    configurations: dict[int, dict],
) -> tuple[dict, dict]:
    corrected = {}
    correction_norms = {}
    for ratio, configuration in configurations.items():
        split = causal_five_field_frozen_principal_generator(
            configuration["context"],
            configuration["base_primitives"],
            configuration["amplitudes"],
            operator="current_split",
            include_characteristic_dissipation=False,
        )
        monolithic = causal_five_field_frozen_principal_generator(
            configuration["context"],
            configuration["base_primitives"],
            configuration["amplitudes"],
            operator="monolithic",
            include_characteristic_dissipation=False,
        )
        candidate = dict(configuration)
        production = np.asarray(configuration["generator"], dtype=float)
        correction = monolithic - split
        candidate["generator"] = production + correction
        corrected[ratio] = candidate
        correction_norms[ratio] = float(
            np.linalg.norm(correction)
            / max(np.linalg.norm(production), np.finfo(float).tiny)
        )
    reports = {}
    arrays = {}
    for family in SHEAR_FAMILIES:
        histories = {}
        moments = {}
        energies = {}
        for ratio, configuration in corrected.items():
            packet, bases, projection = wp10c9a._project_packet(
                configuration,
                family,
            )
            history, moment = wp10c9a._propagate_packet(
                configuration,
                packet,
                bases,
                family,
            )
            histories[ratio] = history
            moments[ratio] = moment
            energies[ratio] = _shear_energy_history(
                configuration,
                history["state"],
                family,
            )
            arrays[f"{family}_ratio{ratio}_state_history"] = (
                history["state"]
            )
            arrays[f"{family}_ratio{ratio}_rate_history"] = (
                history["rate"]
            )
            arrays[f"{family}_ratio{ratio}_amplitude"] = (
                moment["arrays"]["l2_amplitude"]
            )
            for name, values in energies[ratio].items():
                arrays[f"{family}_ratio{ratio}_{name}"] = values
        restricted = {
            ratio: wp10c8z._restrict_history(
                histories[ratio],
                corrected[ratio],
            )
            for ratio in corrected
        }
        coarse_medium = wp10c8z._history_metrics(
            restricted[1],
            restricted[2],
            parent["parent_grid"],
            lower_rg=None,
            upper_rg=parent["active_outer_rg"],
        )
        medium_fine = wp10c8z._history_metrics(
            restricted[2],
            restricted[4],
            parent["parent_grid"],
            lower_rg=None,
            upper_rg=parent["active_outer_rg"],
        )
        moment_coarse = wp10c9a._moment_pair(moments[1], moments[2])
        moment_fine = wp10c9a._moment_pair(moments[2], moments[4])
        energy_summary = _energy_ladder_summary(energies)
        energy_orders = energy_summary["observed_orders"]
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
                moment_coarse["maximum_log_radius_centroid_defect"],
                moment_fine["maximum_log_radius_centroid_defect"],
            ),
            "characteristic_amplitude_damping": _observed_order(
                moment_coarse["maximum_log_amplitude_defect"],
                moment_fine["maximum_log_amplitude_defect"],
            ),
            "physical_total_shear_energy_damping": energy_orders[
                "normalized_total_shear_energy"
            ],
            "selected_branch_self_energy_damping": energy_orders[
                "normalized_selected_branch_self_energy"
            ],
            "scaled_descriptor_shear_norm_damping": energy_orders[
                "normalized_scaled_descriptor_shear_norm_squared"
            ],
        }
        signed_cosine = min(
            medium_fine["state"]["minimum_signed_cosine"],
            medium_fine["rate"]["minimum_signed_cosine"],
        )
        passed = bool(
            orders["phase_centroid"] is not None
            and orders["phase_centroid"] >= MINIMUM_PACKET_PHASE_ORDER
            and orders["characteristic_amplitude_damping"] is not None
            and orders["characteristic_amplitude_damping"]
            >= MINIMUM_PACKET_DAMPING_ORDER
            and orders["physical_total_shear_energy_damping"] is not None
            and orders["physical_total_shear_energy_damping"]
            >= MINIMUM_PACKET_DAMPING_ORDER
            and signed_cosine >= MINIMUM_SIGNED_COSINE
        )
        reports[family] = {
            "history_pairs": {
                "ratio1_ratio2": coarse_medium,
                "ratio2_ratio4": medium_fine,
            },
            "moment_pairs": {
                "ratio1_ratio2": moment_coarse,
                "ratio2_ratio4": moment_fine,
            },
            "physical_energy": energy_summary,
            "observed_orders": orders,
            "fine_minimum_signed_cosine": signed_cosine,
            "passed": passed,
        }
    return {
        "correction": (
            "unchanged_full_generator_plus_monolithic_minus_split_"
            "complete_principal_blocks"
        ),
        "correction_relative_frobenius_norm_by_ratio": correction_norms,
        "by_family": reports,
        "passed": all(row["passed"] for row in reports.values()),
    }, arrays


def run() -> tuple[dict, dict[str, np.ndarray]]:
    started = time.perf_counter()
    required = (
        WP10C9A_OUTPUT,
        WP10C9A_ARRAYS,
        WP10C9B_OUTPUT,
        WP10C9B_ARRAYS,
    )
    if any(not path.exists() for path in required):
        raise FileNotFoundError("WP10c9c0 requires WP10c9a/b evidence")
    wp10c9a_payload = json.loads(WP10C9A_OUTPUT.read_text(encoding="utf-8"))
    wp10c9b_payload = json.loads(WP10C9B_OUTPUT.read_text(encoding="utf-8"))
    if wp10c9a_payload["classification"] != (
        "characteristic_rate_phase_unresolved_operator_redesign_required"
    ):
        raise RuntimeError("WP10c9a stop classification changed")
    if wp10c9b_payload["classification"] != (
        "characteristic_matrix_rejected_bdf_noise_and_"
        "inward_shear_damping_unresolved"
    ):
        raise RuntimeError("WP10c9b stop classification changed")

    print("WP10c9c0: loading frozen N128/N256/N512 anchors", flush=True)
    parent, by_label, labels = wp10c9a._configurations()
    configurations = {
        ratio: by_label[label] for ratio, label in labels.items()
    }
    reference = configurations[4]
    arrays = {}

    print("WP10c9c0: auditing parent physical shear energies", flush=True)
    parent_energy, parent_energy_arrays = _parent_physical_energy_audit(
        configurations
    )
    arrays.update(parent_energy_arrays)
    print("WP10c9c0: auditing sign, derivative plateau, and energy", flush=True)
    algebra, local_arrays = _sign_step_and_energy_audit(reference)
    arrays.update(local_arrays)
    print("WP10c9c0: running constant-coefficient Fourier symbols", flush=True)
    fourier, fourier_arrays = _fourier_audit(reference)
    arrays.update(fourier_arrays)
    print("WP10c9c0: running variable-coefficient manufactured wave", flush=True)
    manufactured, manufactured_arrays = _manufactured_audit(
        configurations
    )
    arrays.update(manufactured_arrays)
    local_passed = bool(
        algebra["passed"] and fourier["passed"] and manufactured["passed"]
    )
    corrected_packets = None
    if local_passed:
        print(
            "WP10c9c0: local gates passed; running corrected shear packets",
            flush=True,
        )
        corrected_packets, packet_arrays = _corrected_packet_audit(
            parent,
            configurations,
        )
        arrays.update(packet_arrays)

    production_inward = wp10c9a_payload["packet_results"][
        "inward_shear"
    ]["observed_orders"]
    matrix_inward = wp10c9b_payload["packet_results"][
        "inward_shear"
    ]["observed_orders"]
    corrected_passed = bool(
        corrected_packets is not None and corrected_packets["passed"]
    )
    current_split_locally_failed = not (
        fourier["minimum_observed_orders"]["current_split_phase"]
        >= MINIMUM_LOCAL_ORDER
        and fourier["minimum_observed_orders"]["current_split_damping"]
        >= MINIMUM_LOCAL_ORDER
        and manufactured["observed_orders"]["current_split"][
            "ratio2_to_ratio4"
        ]
        >= MINIMUM_LOCAL_ORDER
    )
    monolithic_locally_passed = bool(
        fourier["minimum_observed_orders"]["monolithic_phase"]
        >= MINIMUM_LOCAL_ORDER
        and fourier["minimum_observed_orders"]["monolithic_damping"]
        >= MINIMUM_LOCAL_ORDER
        and manufactured["observed_orders"]["monolithic"][
            "ratio2_to_ratio4"
        ]
        >= MINIMUM_LOCAL_ORDER
    )
    path_inconsistency_proved = bool(
        monolithic_locally_passed
        and (
            current_split_locally_failed
            or corrected_passed
        )
    )
    if not local_passed:
        classification = "principal_or_energy_root_cause_audit_failed"
    elif path_inconsistency_proved:
        classification = (
            "principal_split_defect_proved_path_candidate_authorized"
        )
    else:
        classification = (
            "path_inconsistency_not_proved_selected_shear_damping_"
            "persists"
        )

    arrays["times"] = np.linspace(
        0.0,
        wp10c9a.TARGET_SECONDS,
        wp10c9a.TIME_SAMPLES,
    )
    DEFAULT_ARRAYS.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DEFAULT_ARRAYS, **arrays)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "base_commit": BASE_COMMIT,
        "classification": classification,
        "passed": path_inconsistency_proved,
        "scope": {
            "production_operator_changed": False,
            "nonlinear_path_flux_implemented": False,
            "constant_coefficient_fourier_run": True,
            "variable_coefficient_manufactured_wave_run": True,
            "corrected_full_shear_packet_ladder_run": (
                corrected_packets is not None
            ),
            "nonlinear_truth_run": False,
            "fixed_q_averaging_run": False,
            "reduced_model_run": False,
        },
        "frozen_parent_evidence": {
            "wp10c9a_json": _relative(WP10C9A_OUTPUT),
            "wp10c9a_json_sha256": _sha256(WP10C9A_OUTPUT),
            "wp10c9a_arrays": _relative(WP10C9A_ARRAYS),
            "wp10c9a_arrays_sha256": _sha256(WP10C9A_ARRAYS),
            "wp10c9b_json": _relative(WP10C9B_OUTPUT),
            "wp10c9b_json_sha256": _sha256(WP10C9B_OUTPUT),
            "wp10c9b_arrays": _relative(WP10C9B_ARRAYS),
            "wp10c9b_arrays_sha256": _sha256(WP10C9B_ARRAYS),
        },
        "sign_derivative_and_shear_energy": algebra,
        "parent_physical_shear_energy": parent_energy,
        "constant_coefficient_fourier": fourier,
        "variable_coefficient_manufactured_wave": manufactured,
        "corrected_full_operator_shear_packets": corrected_packets,
        "parent_inward_shear_orders": {
            "production_scalar_rusanov": production_inward,
            "wp10c9b_characteristic_matrix": matrix_inward,
        },
        "root_cause_decision": {
            "current_split_locally_failed": current_split_locally_failed,
            "monolithic_locally_passed": monolithic_locally_passed,
            "corrected_full_packet_passed": corrected_passed,
            "path_inconsistency_proved": path_inconsistency_proved,
            "wp10c9c1_path_candidate_authorized": (
                path_inconsistency_proved
            ),
            "interpretation": (
                "A path candidate is authorized only when the complete-"
                "principal monolithic reference passes and the matched "
                "current split fails, or when its isolated correction "
                "repairs the unchanged full packet ladder."
            ),
            "next_step": (
                "implement one sign-explicit path-conservative candidate"
                if path_inconsistency_proved
                else (
                    "stop before WP10c9c1 and localize the full-generator "
                    "selected-family damping defect among lower-order "
                    "source, non-normal family transfer, projector "
                    "rotation, and packet-energy transport blocks; retain "
                    "the converging total symmetrizer-based shear-subspace "
                    "energy as the basis-invariant control"
                )
            ),
        },
        "gates": {
            "maximum_path_linearization_defect": (
                MAXIMUM_PATH_LINEARIZATION_DEFECT
            ),
            "maximum_path_reversal_defect": (
                MAXIMUM_PATH_REVERSAL_DEFECT
            ),
            "maximum_derivative_plateau_defect": (
                MAXIMUM_DERIVATIVE_PLATEAU_DEFECT
            ),
            "maximum_projector_defect": MAXIMUM_PROJECTOR_DEFECT,
            "maximum_symmetry_defect": MAXIMUM_SYMMETRY_DEFECT,
            "minimum_local_order": MINIMUM_LOCAL_ORDER,
            "fourier_binding_maximum_theta": (
                FOURIER_BINDING_MAXIMUM_THETA
            ),
            "minimum_packet_phase_order": MINIMUM_PACKET_PHASE_ORDER,
            "minimum_packet_damping_order": (
                MINIMUM_PACKET_DAMPING_ORDER
            ),
            "minimum_signed_cosine": MINIMUM_SIGNED_COSINE,
        },
        "machine_evidence": {
            "arrays_path": _relative(DEFAULT_ARRAYS),
            "arrays_sha256": _sha256(DEFAULT_ARRAYS),
        },
        "provenance": {
            "runner": THIS_RUNNER,
            "core_files": {
                path: _sha256(ROOT / path) for path in CORE_FILES
            },
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
                "path_inconsistency_proved": payload[
                    "root_cause_decision"
                ]["path_inconsistency_proved"],
                "wp10c9c1_authorized": payload[
                    "root_cause_decision"
                ]["wp10c9c1_path_candidate_authorized"],
                "output": str(DEFAULT_OUTPUT),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
