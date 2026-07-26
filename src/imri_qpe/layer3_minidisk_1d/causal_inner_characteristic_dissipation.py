"""Audit-only characteristic basis and family-resolved dissipation.

The production causal five-field flux deliberately remains the scalar
maximum-speed Rusanov flux.  This module supplies the bounded WP10c9b audit
candidate.  It constructs the complete local coordinate principal pencil

``A(p) p_ct + B(p) p_R = lower-order terms``

in the physical primitive chart
``(ln Sigma, beta_R, beta_phi, ln T, chi)``.  ``A`` includes mapped Killing
storage and responsive-height temporal storage.  ``B`` includes the physical
flux derivative, vertical-work principal source, and the nonconservative
causal-shear gradient source.

The family-resolved penalty acts on the path-integrated descriptor jump, not
on a separately fitted flux vector.  It therefore preserves one shared face
flux while respecting the five physical acoustic/contact/shear speeds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eig

from imri_qpe.constants import C

from .causal_inner_characteristic_phase import (
    CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES,
)
from .causal_inner_dae import audit_causal_five_field_principal
from .causal_inner_dae_system import (
    CausalFiveFieldDAEContext,
    _cell_state,
)
from .causal_inner_stress import (
    causal_rest_frame_shear_rate,
    causal_stress_relaxation_source,
)
from .causal_inner_thermal import (
    causal_temporal_vertical_work_storage,
    kerr_schild_column_four_velocity,
)


_FIVE_POINT_MULTIPLIERS = np.asarray(
    [-2.0, -1.0, 1.0, 2.0],
    dtype=float,
)
_FIVE_POINT_WEIGHTS = np.asarray(
    [1.0, -8.0, 8.0, -1.0],
    dtype=float,
) / 12.0


@dataclass(frozen=True)
class CausalFiveFieldCoordinatePrincipalBasis:
    """Complete coordinate principal basis at one face state."""

    family_labels: tuple[str, ...]
    primitive_chart: np.ndarray
    primitive_column_scales: np.ndarray
    descriptor_row_scales: np.ndarray
    temporal_storage_matrix: np.ndarray
    spatial_principal_matrix: np.ndarray
    analytic_speeds_over_c: np.ndarray
    numerical_speeds_over_c: np.ndarray
    primitive_right_eigenvectors: np.ndarray
    descriptor_right_eigenvectors: np.ndarray
    descriptor_left_eigenvectors: np.ndarray
    maximum_analytic_speed_defect: float
    maximum_eigenpair_defect: float
    maximum_biorthogonality_defect: float
    maximum_imaginary_part: float
    descriptor_condition_number: float
    incoming_inner_characteristics: int

    @property
    def passed(self) -> bool:
        return bool(
            self.maximum_analytic_speed_defect <= 2.5e-3
            and self.maximum_eigenpair_defect <= 1.0e-10
            and self.maximum_biorthogonality_defect <= 1.0e-10
            and self.maximum_imaginary_part <= 1.0e-10
            and np.isfinite(self.descriptor_condition_number)
            and self.descriptor_condition_number <= 1.0e10
        )


@dataclass(frozen=True)
class CausalFiveFieldCharacteristicDissipation:
    """One audit-only matrix penalty evaluated at a face."""

    basis: CausalFiveFieldCoordinatePrincipalBasis
    descriptor_jump: np.ndarray
    characteristic_jump: np.ndarray
    dissipative_flux_over_c: np.ndarray
    quadratic_dissipation: float
    scalar_equal_speed_defect: float


def _primitive_column_scales(
    context: CausalFiveFieldDAEContext,
    radius: float,
    chart: np.ndarray,
) -> np.ndarray:
    state = _cell_state(context, radius, chart)
    stress_scale = max(
        abs(float(chart[4])),
        abs(float(state.closure.equilibrium_specific_stress)),
        1.0e-14,
    )
    return np.asarray([1.0, 0.1, 0.1, 1.0, stress_scale], dtype=float)


def _column_steps(scales: np.ndarray) -> np.ndarray:
    # A five-point derivative at these relative displacements is below the
    # eigensystem gate on the certified inner anchors while remaining clear
    # of roundoff in the very small causal-stress chart.
    return 2.0e-4 * np.asarray(scales, dtype=float)


def _differentiate_state_maps(
    context: CausalFiveFieldDAEContext,
    radius: float,
    chart: np.ndarray,
    steps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Differentiate U, F, lower-u, and log-height at frozen geometry."""

    conserved = np.empty((5, 5), dtype=float)
    flux = np.empty((5, 5), dtype=float)
    lower_velocity = np.empty((3, 5), dtype=float)
    log_height = np.empty(5, dtype=float)
    for column, step in enumerate(steps):
        conserved_samples = []
        flux_samples = []
        lower_samples = []
        height_samples = []
        for multiplier in _FIVE_POINT_MULTIPLIERS:
            candidate = np.array(chart, copy=True)
            candidate[column] += multiplier * step
            state = _cell_state(context, radius, candidate)
            conserved_samples.append(state.conserved)
            flux_samples.append(state.flux_over_c)
            lower_samples.append(
                state.geometry.spacetime_metric
                @ kerr_schild_column_four_velocity(
                    state.geometry,
                    state.primitive,
                )
            )
            height_samples.append(
                np.log(state.thermodynamics.proper_half_thickness)
            )
        denominator = step
        conserved[:, column] = (
            _FIVE_POINT_WEIGHTS
            @ np.asarray(conserved_samples, dtype=float)
        ) / denominator
        flux[:, column] = (
            _FIVE_POINT_WEIGHTS
            @ np.asarray(flux_samples, dtype=float)
        ) / denominator
        lower_velocity[:, column] = (
            _FIVE_POINT_WEIGHTS
            @ np.asarray(lower_samples, dtype=float)
        ) / denominator
        log_height[column] = float(
            _FIVE_POINT_WEIGHTS
            @ np.asarray(height_samples, dtype=float)
            / denominator
        )
    return conserved, flux, lower_velocity, log_height


def _vertical_temporal_storage_matrix(
    context: CausalFiveFieldDAEContext,
    radius: float,
    chart: np.ndarray,
    steps: np.ndarray,
) -> np.ndarray:
    """Return the responsive-height temporal one-form in five rows."""

    base = _cell_state(context, radius, chart)
    result = np.zeros((5, 5), dtype=float)
    for column, step in enumerate(steps):
        minus_chart = np.array(chart, copy=True)
        plus_chart = np.array(chart, copy=True)
        minus_chart[column] -= step
        plus_chart[column] += step
        minus = _cell_state(context, radius, minus_chart)
        plus = _cell_state(context, radius, plus_chart)
        storage = causal_temporal_vertical_work_storage(
            base.geometry,
            base.primitive,
            minus.thermodynamics,
            plus.thermodynamics,
        )
        result[:4, column] = (
            np.asarray(storage.killing_storage_increment, dtype=float)
            / (2.0 * step)
        )
    return result


def _principal_source_matrix(
    context: CausalFiveFieldDAEContext,
    radius: float,
    chart: np.ndarray,
    lower_velocity_derivative: np.ndarray,
    log_height_derivative: np.ndarray,
) -> np.ndarray:
    """Return dS/d(p_R) for height work and causal shear."""

    state = _cell_state(context, radius, chart)
    result = np.zeros((5, 5), dtype=float)
    zero_shear = causal_rest_frame_shear_rate(
        state.geometry,
        state.primitive,
        radial_lower_four_velocity_derivative=np.zeros(3, dtype=float),
    )
    zero_stress_source = causal_stress_relaxation_source(
        state.geometry,
        state.stress,
        state.closure,
        positive_shear_rate=zero_shear,
    )
    for column in range(5):
        shear = causal_rest_frame_shear_rate(
            state.geometry,
            state.primitive,
            radial_lower_four_velocity_derivative=(
                lower_velocity_derivative[:, column]
            ),
        )
        result[4, column] = (
            causal_stress_relaxation_source(
                state.geometry,
                state.stress,
                state.closure,
                positive_shear_rate=shear,
            )
            - zero_stress_source
        )
        height_rate = (
            C
            * kerr_schild_column_four_velocity(
                state.geometry,
                state.primitive,
            )[1]
            * log_height_derivative[column]
        )
        if height_rate != 0.0:
            from .causal_inner_thermal import causal_comoving_energy_source

            vertical = causal_comoving_energy_source(
                state.geometry,
                state.primitive,
                comoving_energy_rate=(
                    -state.thermodynamics.integrated_pressure * height_rate
                ),
            )
            result[:4, column] = vertical.killing_source_per_ct
    return result


def _ordered_real_generalized_basis(
    temporal: np.ndarray,
    spatial: np.ndarray,
    column_scales: np.ndarray,
    analytic_speeds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve and deterministically order the scaled generalized pencil."""

    row_scales = np.maximum(
        np.max(np.abs(temporal), axis=1),
        np.max(np.abs(spatial), axis=1),
    )
    row_scales = np.maximum(
        row_scales,
        max(float(np.max(row_scales)), 1.0) * 1.0e-14,
    )
    scaled_temporal = (
        temporal * column_scales[None, :] / row_scales[:, None]
    )
    scaled_spatial = (
        spatial * column_scales[None, :] / row_scales[:, None]
    )
    values, vectors = eig(scaled_spatial, scaled_temporal)
    remaining = list(range(5))
    order = []
    for target in analytic_speeds:
        selected = min(
            remaining,
            key=lambda index: abs(values[index] - target),
        )
        order.append(selected)
        remaining.remove(selected)
    values = values[np.asarray(order, dtype=int)]
    vectors = vectors[:, np.asarray(order, dtype=int)]
    primitive = column_scales[:, None] * vectors
    if np.max(np.abs(np.imag(primitive))) <= 1.0e-9:
        primitive = np.real(primitive)
    for column in range(5):
        norm = float(np.linalg.norm(primitive[:, column]))
        if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
            raise RuntimeError("coordinate characteristic vector is singular")
        primitive[:, column] /= norm
        pivot = int(np.argmax(np.abs(primitive[:, column])))
        if np.real(primitive[pivot, column]) < 0.0:
            primitive[:, column] *= -1.0
    return values, primitive, row_scales


def causal_five_field_coordinate_principal_basis(
    context: CausalFiveFieldDAEContext,
    radius: float,
    primitive_chart: np.ndarray,
) -> CausalFiveFieldCoordinatePrincipalBasis:
    """Build the complete five-family coordinate principal eigensystem."""

    context = context.validated()
    radius = float(radius)
    chart = np.asarray(primitive_chart, dtype=float)
    if (
        not np.isfinite(radius)
        or radius <= 0.0
        or chart.shape != (5,)
        or np.any(~np.isfinite(chart))
    ):
        raise ValueError("coordinate-principal inputs are invalid")
    state = _cell_state(context, radius, chart)
    scales = _primitive_column_scales(context, radius, chart)
    steps = _column_steps(scales)
    conserved, flux, lower, log_height = _differentiate_state_maps(
        context,
        radius,
        chart,
        steps,
    )
    vertical = _vertical_temporal_storage_matrix(
        context,
        radius,
        chart,
        steps,
    )
    source = _principal_source_matrix(
        context,
        radius,
        chart,
        lower,
        log_height,
    )
    temporal = conserved + vertical
    spatial = flux - source
    local = audit_causal_five_field_principal(
        state.geometry,
        context.vertical_frequency.eos(radius),
        state.closure,
        surface_density=state.primitive.surface_density,
        radial_velocity_over_c=state.primitive.radial_velocity_over_c,
        azimuthal_velocity_over_c=state.primitive.azimuthal_velocity_over_c,
        temperature=state.thermodynamics.temperature,
    )
    analytic = np.asarray(local.coordinate_speeds_over_c, dtype=float)
    values, primitive, row_scales = _ordered_real_generalized_basis(
        temporal,
        spatial,
        scales,
        analytic,
    )
    descriptor = temporal @ primitive
    scaled_descriptor = descriptor / row_scales[:, None]
    column_norms = np.linalg.norm(scaled_descriptor, axis=0)
    if np.any(column_norms <= np.finfo(float).tiny):
        raise RuntimeError("coordinate descriptor basis is singular")
    scaled_descriptor = scaled_descriptor / column_norms[None, :]
    primitive = primitive / column_norms[None, :]
    left = np.linalg.inv(scaled_descriptor)
    real_values = np.real(values)
    residual = (
        spatial @ primitive
        - temporal @ (primitive * real_values[None, :])
    )
    residual_scale = max(
        float(np.max(np.abs(spatial @ primitive))),
        float(np.max(np.abs(temporal @ primitive))),
        np.finfo(float).tiny,
    )
    return CausalFiveFieldCoordinatePrincipalBasis(
        family_labels=CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES,
        primitive_chart=np.array(chart, copy=True),
        primitive_column_scales=scales,
        descriptor_row_scales=np.asarray(row_scales, dtype=float),
        temporal_storage_matrix=temporal,
        spatial_principal_matrix=spatial,
        analytic_speeds_over_c=analytic,
        numerical_speeds_over_c=np.asarray(real_values, dtype=float),
        primitive_right_eigenvectors=np.asarray(primitive, dtype=float),
        descriptor_right_eigenvectors=np.asarray(
            scaled_descriptor,
            dtype=float,
        ),
        descriptor_left_eigenvectors=np.asarray(left, dtype=float),
        maximum_analytic_speed_defect=float(
            np.max(np.abs(real_values - analytic))
        ),
        maximum_eigenpair_defect=float(
            np.max(np.abs(residual)) / residual_scale
        ),
        maximum_biorthogonality_defect=float(
            np.max(np.abs(left @ scaled_descriptor - np.eye(5)))
        ),
        maximum_imaginary_part=float(np.max(np.abs(np.imag(values)))),
        descriptor_condition_number=float(np.linalg.cond(scaled_descriptor)),
        incoming_inner_characteristics=int(np.sum(real_values > 0.0)),
    )


def causal_five_field_descriptor_jump(
    context: CausalFiveFieldDAEContext,
    radius: float,
    left_chart: np.ndarray,
    right_chart: np.ndarray,
) -> np.ndarray:
    """Return mapped plus path-integrated responsive-height face jump."""

    left_chart = np.asarray(left_chart, dtype=float)
    right_chart = np.asarray(right_chart, dtype=float)
    if left_chart.shape != (5,) or right_chart.shape != (5,):
        raise ValueError("descriptor-jump charts must have length five")
    left = _cell_state(context, radius, left_chart)
    right = _cell_state(context, radius, right_chart)
    midpoint_chart = 0.5 * (left_chart + right_chart)
    midpoint = _cell_state(context, radius, midpoint_chart)
    vertical = causal_temporal_vertical_work_storage(
        midpoint.geometry,
        midpoint.primitive,
        left.thermodynamics,
        right.thermodynamics,
    )
    result = np.asarray(right.conserved - left.conserved, dtype=float)
    result[:4] += np.asarray(
        vertical.killing_storage_increment,
        dtype=float,
    )
    return result


def causal_five_field_characteristic_dissipation(
    context: CausalFiveFieldDAEContext,
    radius: float,
    left_chart: np.ndarray,
    right_chart: np.ndarray,
    *,
    face_measure: float = 1.0,
) -> CausalFiveFieldCharacteristicDissipation:
    """Return the audit-only family-resolved dissipative face flux."""

    context = context.validated()
    radius = float(radius)
    measure = float(face_measure)
    left_chart = np.asarray(left_chart, dtype=float)
    right_chart = np.asarray(right_chart, dtype=float)
    if (
        not np.isfinite(measure)
        or measure <= 0.0
        or left_chart.shape != (5,)
        or right_chart.shape != (5,)
        or np.any(~np.isfinite(left_chart))
        or np.any(~np.isfinite(right_chart))
    ):
        raise ValueError("characteristic-dissipation inputs are invalid")
    midpoint = 0.5 * (left_chart + right_chart)
    basis = causal_five_field_coordinate_principal_basis(
        context,
        radius,
        midpoint,
    )
    jump = causal_five_field_descriptor_jump(
        context,
        radius,
        left_chart,
        right_chart,
    )
    scaled_jump = jump / basis.descriptor_row_scales
    characteristic = basis.descriptor_left_eigenvectors @ scaled_jump
    absolute_speeds = np.abs(basis.numerical_speeds_over_c)
    scaled_penalty = (
        basis.descriptor_right_eigenvectors
        @ (absolute_speeds * characteristic)
    )
    penalty = basis.descriptor_row_scales * scaled_penalty
    dissipation = -0.5 * measure * penalty
    equal_speed = float(np.max(absolute_speeds))
    equal_penalty = basis.descriptor_row_scales * (
        basis.descriptor_right_eigenvectors
        @ (equal_speed * characteristic)
    )
    scalar_defect = float(
        np.max(np.abs(equal_penalty - equal_speed * jump))
        / max(
            float(np.max(np.abs(equal_speed * jump))),
            np.finfo(float).tiny,
        )
    )
    return CausalFiveFieldCharacteristicDissipation(
        basis=basis,
        descriptor_jump=jump,
        characteristic_jump=characteristic,
        dissipative_flux_over_c=np.asarray(dissipation, dtype=float),
        quadratic_dissipation=float(
            np.sum(absolute_speeds * characteristic**2)
        ),
        scalar_equal_speed_defect=scalar_defect,
    )
