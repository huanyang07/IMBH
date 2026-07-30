"""Audit-only invariant energy tools for five-field interface scattering.

The production DAE has no dependency on this module.  The routines here
construct a smooth manufactured primitive background and an energy metric for
the complete frozen coordinate pencil

``A p_t + B p_x = sum_k C_k p + A f``.

The energy metric is built from spectral projectors of ``A^{-1} B`` rather
than from normalized eigenvector coefficients.  It is therefore invariant
under independent rescaling or sign changes of the characteristic vectors.
For separated real families the metric is positive, makes the projectors
mutually orthogonal, and symmetrizes the complete coordinate principal
operator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import make_interp_spline
from scipy.linalg import eig

from .causal_inner_characteristic_phase import (
    CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES,
)


_N_FIELDS = 5
_C4_SMOOTHERSTEP = Polynomial(
    (0.0, 0.0, 0.0, 0.0, 0.0, 126.0, -420.0, 540.0, -315.0, 70.0)
)


@dataclass(frozen=True)
class CausalC4ManufacturedPrimitiveState:
    """One C4 primitive-chart extension evaluated at declared coordinates."""

    log_radii: np.ndarray
    primitive_charts: np.ndarray
    core_log_radii: np.ndarray
    core_primitive_charts: np.ndarray
    left_far_primitive_chart: np.ndarray
    right_far_primitive_chart: np.ndarray
    transition_log_width: float
    maximum_core_replay_defect: float
    maximum_scaled_C4_join_defect: float
    maximum_scaled_C4_far_defect: float


@dataclass(frozen=True)
class CausalInvariantScatteringEnergy:
    """Normalization-invariant energy data for one complete 5x5 pencil."""

    family_labels: tuple[str, ...]
    characteristic_speeds: np.ndarray
    evolution_matrix: np.ndarray
    primitive_projectors: np.ndarray
    primitive_energy_metric: np.ndarray
    primitive_energy_flux_metric: np.ndarray
    minimum_energy_eigenvalue: float
    maximum_energy_eigenvalue: float
    maximum_projector_identity_defect: float
    maximum_projector_idempotence_defect: float
    maximum_cross_projector_defect: float
    maximum_energy_orthogonality_defect: float
    maximum_symmetrizer_defect: float
    maximum_eigenpair_defect: float
    maximum_rescaling_invariance_defect: float
    eigenvector_condition_number: float
    maximum_imaginary_part: float


@dataclass(frozen=True)
class CausalManufacturedEnergyLedger:
    """Pointwise complete energy identity for one manufactured perturbation."""

    stored_energy_rate: np.ndarray
    principal_flux_divergence: np.ndarray
    lower_source_work_by_block: dict[str, np.ndarray]
    manufactured_forcing_work: np.ndarray
    background_gradient_work: np.ndarray
    reconstructed_right_hand_side: np.ndarray
    residual: np.ndarray
    maximum_relative_closure_defect: float


def _validated_field_scales(field_scales: np.ndarray) -> np.ndarray:
    scales = np.asarray(field_scales, dtype=float)
    if (
        scales.shape != (_N_FIELDS,)
        or np.any(~np.isfinite(scales))
        or np.any(scales <= 0.0)
    ):
        raise ValueError("five positive primitive field scales are required")
    return scales


def _longdouble_polynomial_derivative_value(
    polynomial: Polynomial,
    coordinate: float,
    order: int,
) -> float:
    """Evaluate one endpoint derivative without float64 cancellation."""

    coefficients = np.asarray(polynomial.coef, dtype=np.longdouble)
    for _unused in range(int(order)):
        coefficients = np.asarray(
            [
                np.longdouble(index) * coefficients[index]
                for index in range(1, coefficients.size)
            ],
            dtype=np.longdouble,
        )
    value = np.longdouble(0.0)
    point = np.longdouble(coordinate)
    for coefficient in coefficients[::-1]:
        value = value * point + coefficient
    return float(value)


def causal_c4_manufactured_primitive_state(
    log_radii: np.ndarray,
    core_log_radii: np.ndarray,
    core_primitive_charts: np.ndarray,
    left_far_primitive_chart: np.ndarray,
    right_far_primitive_chart: np.ndarray,
    *,
    transition_log_width: float,
    field_scales: np.ndarray,
) -> CausalC4ManufacturedPrimitiveState:
    """Interpolate an exact discrete core and attach C4 constant far states.

    A quintic B-spline gives a C4 interpolant through the twelve declared
    physical-core cell values.  Its fourth-order endpoint Taylor jets are
    blended to independently frozen constant far states by the degree-nine
    C4 smootherstep.  No physical matrix is interpolated.
    """

    target = np.asarray(log_radii, dtype=float)
    core_x = np.asarray(core_log_radii, dtype=float)
    core = np.asarray(core_primitive_charts, dtype=float)
    left_far = np.asarray(left_far_primitive_chart, dtype=float)
    right_far = np.asarray(right_far_primitive_chart, dtype=float)
    scales = _validated_field_scales(field_scales)
    width = float(transition_log_width)
    if (
        target.ndim != 1
        or core_x.ndim != 1
        or core.shape != (core_x.size, _N_FIELDS)
        or core_x.size < 8
        or left_far.shape != (_N_FIELDS,)
        or right_far.shape != (_N_FIELDS,)
        or np.any(~np.isfinite(target))
        or np.any(~np.isfinite(core_x))
        or np.any(~np.isfinite(core))
        or np.any(~np.isfinite(left_far))
        or np.any(~np.isfinite(right_far))
        or np.any(np.diff(target) <= 0.0)
        or np.any(np.diff(core_x) <= 0.0)
        or not np.isfinite(width)
        or width <= 0.0
    ):
        raise ValueError("C4 manufactured primitive-state inputs are invalid")

    left_join = float(core_x[0])
    right_join = float(core_x[-1])
    left_constant = left_join - width
    right_constant = right_join + width
    splines = tuple(
        make_interp_spline(core_x, core[:, field], k=5)
        for field in range(_N_FIELDS)
    )
    charts = np.empty((target.size, _N_FIELDS), dtype=float)
    left_constant_mask = target <= left_constant
    left_transition_mask = (target > left_constant) & (target < left_join)
    core_mask = (target >= left_join) & (target <= right_join)
    right_transition_mask = (target > right_join) & (
        target < right_constant
    )
    right_constant_mask = target >= right_constant
    charts[left_constant_mask] = left_far
    charts[right_constant_mask] = right_far
    for field in range(_N_FIELDS):
        left_delta = target[left_transition_mask] - left_join
        left_jet = np.zeros_like(left_delta)
        right_delta = target[right_transition_mask] - right_join
        right_jet = np.zeros_like(right_delta)
        for order in range(5):
            left_jet += (
                float(splines[field](left_join, nu=order))
                * left_delta**order
                / math.factorial(order)
            )
            right_jet += (
                float(splines[field](right_join, nu=order))
                * right_delta**order
                / math.factorial(order)
            )
        left_taper = 1.0 - _C4_SMOOTHERSTEP(
            -left_delta / width
        )
        right_taper = 1.0 - _C4_SMOOTHERSTEP(
            right_delta / width
        )
        charts[left_transition_mask, field] = left_far[field] + (
            left_taper * (left_jet - left_far[field])
        )
        charts[core_mask, field] = splines[field](target[core_mask])
        charts[right_transition_mask, field] = right_far[field] + (
            right_taper * (right_jet - right_far[field])
        )
    if np.any(~np.isfinite(charts)):
        raise RuntimeError("C4 manufactured primitive state is non-finite")

    replay = np.asarray(
        [
            [float(spline(value)) for spline in splines]
            for value in core_x
        ],
        dtype=float,
    )
    core_replay = float(
        np.max(np.abs((replay - core) / scales[None, :]))
    )
    # The unexpanded product is used above so that exact endpoint constants
    # are not lost to cancellation.  These defects certify the only extra
    # endpoint factors: S(0)=0, S(1)=1, and derivatives one through four
    # vanish at both endpoints.
    join_defect = abs(
        _longdouble_polynomial_derivative_value(
            _C4_SMOOTHERSTEP,
            0.0,
            0,
        )
    )
    far_defect = abs(
        _longdouble_polynomial_derivative_value(
            _C4_SMOOTHERSTEP,
            1.0,
            0,
        )
        - 1.0
    )
    for order in range(1, 5):
        join_defect = max(
            join_defect,
            abs(
                _longdouble_polynomial_derivative_value(
                    _C4_SMOOTHERSTEP,
                    0.0,
                    order,
                )
            ),
        )
        far_defect = max(
            far_defect,
            abs(
                _longdouble_polynomial_derivative_value(
                    _C4_SMOOTHERSTEP,
                    1.0,
                    order,
                )
            ),
        )

    return CausalC4ManufacturedPrimitiveState(
        log_radii=np.array(target, copy=True),
        primitive_charts=charts,
        core_log_radii=np.array(core_x, copy=True),
        core_primitive_charts=np.array(core, copy=True),
        left_far_primitive_chart=np.array(left_far, copy=True),
        right_far_primitive_chart=np.array(right_far, copy=True),
        transition_log_width=width,
        maximum_core_replay_defect=core_replay,
        maximum_scaled_C4_join_defect=float(join_defect),
        maximum_scaled_C4_far_defect=float(far_defect),
    )


def causal_normalization_invariant_scattering_energy(
    temporal_storage_matrix: np.ndarray,
    spatial_principal_matrix: np.ndarray,
    field_scales: np.ndarray,
) -> CausalInvariantScatteringEnergy:
    """Return a positive projector-built symmetrizer of ``A^{-1} B``."""

    temporal = np.asarray(temporal_storage_matrix, dtype=float)
    spatial = np.asarray(spatial_principal_matrix, dtype=float)
    scales = _validated_field_scales(field_scales)
    if (
        temporal.shape != (_N_FIELDS, _N_FIELDS)
        or spatial.shape != temporal.shape
        or np.any(~np.isfinite(temporal))
        or np.any(~np.isfinite(spatial))
    ):
        raise ValueError("complete coordinate-pencil matrices are invalid")
    evolution = np.linalg.solve(temporal, spatial)
    scale_matrix = np.diag(scales)
    inverse_scale = np.diag(1.0 / scales)
    dimensionless_evolution = (
        inverse_scale @ evolution @ scale_matrix
    )
    values, vectors = eig(dimensionless_evolution)
    order = np.lexsort((np.imag(values), np.real(values)))
    values = values[order]
    vectors = vectors[:, order]
    maximum_imaginary = max(
        float(np.max(np.abs(np.imag(values)))),
        float(np.max(np.abs(np.imag(vectors)))),
    )
    if maximum_imaginary > 1.0e-10:
        raise RuntimeError(
            "complete scattering-energy eigensystem is not real"
        )
    speeds = np.real(values)
    right = np.real(vectors)
    inverse = np.linalg.inv(right)
    dimensionless_projectors = np.asarray(
        [
            np.outer(right[:, family], inverse[family])
            for family in range(_N_FIELDS)
        ],
        dtype=float,
    )
    projectors = np.einsum(
        "ij,fjk,kl->fil",
        scale_matrix,
        dimensionless_projectors,
        inverse_scale,
        optimize=True,
    )
    dimensionless_energy = np.sum(
        np.einsum(
            "fji,fjk->fik",
            dimensionless_projectors,
            dimensionless_projectors,
            optimize=True,
        ),
        axis=0,
    )
    energy = inverse_scale @ dimensionless_energy @ inverse_scale
    flux = energy @ evolution
    energy_eigenvalues = np.linalg.eigvalsh(
        0.5 * (energy + energy.T)
    )

    identity = np.eye(_N_FIELDS)
    # Algebraic projector gates are evaluated in the declared dimensionless
    # primitive chart.  Absolute entries in the physical chart can otherwise
    # be amplified solely by the fixed stress-unit conversion.
    identity_defect = float(
        np.max(
            np.abs(
                np.sum(dimensionless_projectors, axis=0) - identity
            )
        )
    )
    idempotence = 0.0
    cross = 0.0
    orthogonality = 0.0
    for first in range(_N_FIELDS):
        idempotence = max(
            idempotence,
            float(
                np.max(
                    np.abs(
                        dimensionless_projectors[first]
                        @ dimensionless_projectors[first]
                        - dimensionless_projectors[first]
                    )
                )
            ),
        )
        for second in range(_N_FIELDS):
            if first == second:
                continue
            cross = max(
                cross,
                float(
                    np.max(
                        np.abs(
                            dimensionless_projectors[first]
                            @ dimensionless_projectors[second]
                        )
                    )
                ),
            )
            block = (
                projectors[first].T
                @ energy
                @ projectors[second]
            )
            orthogonality = max(
                orthogonality,
                float(np.max(np.abs(block)))
                / max(float(np.max(np.abs(energy))), np.finfo(float).tiny),
            )

    eigenpair = 0.0
    for family, speed in enumerate(speeds):
        residual = (
            evolution @ projectors[family]
            - speed * projectors[family]
        )
        scale = max(
            float(np.max(np.abs(evolution @ projectors[family]))),
            float(np.max(np.abs(speed * projectors[family]))),
            np.finfo(float).tiny,
        )
        eigenpair = max(
            eigenpair,
            float(np.max(np.abs(residual)) / scale),
        )

    # This explicitly checks invariance under arbitrary family sign/scale
    # choices.  The factors are fixed audit constants, not fitted values.
    factors = np.asarray((2.0, -3.0, 0.5, -0.75, 4.0), dtype=float)
    changed_right = right * factors[None, :]
    changed_inverse = np.linalg.inv(changed_right)
    changed_projectors = np.asarray(
        [
            np.outer(
                changed_right[:, family],
                changed_inverse[family],
            )
            for family in range(_N_FIELDS)
        ]
    )
    rescaling = float(
        np.max(
            np.abs(
                changed_projectors - dimensionless_projectors
            )
        )
    )
    flux_scale = max(
        float(np.max(np.abs(flux))),
        np.finfo(float).tiny,
    )
    symmetrizer = float(
        np.max(np.abs(flux - flux.T)) / flux_scale
    )
    return CausalInvariantScatteringEnergy(
        family_labels=CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES,
        characteristic_speeds=np.asarray(speeds, dtype=float),
        evolution_matrix=np.asarray(evolution, dtype=float),
        primitive_projectors=np.asarray(projectors, dtype=float),
        primitive_energy_metric=np.asarray(energy, dtype=float),
        primitive_energy_flux_metric=np.asarray(flux, dtype=float),
        minimum_energy_eigenvalue=float(np.min(energy_eigenvalues)),
        maximum_energy_eigenvalue=float(np.max(energy_eigenvalues)),
        maximum_projector_identity_defect=identity_defect,
        maximum_projector_idempotence_defect=float(idempotence),
        maximum_cross_projector_defect=float(cross),
        maximum_energy_orthogonality_defect=float(orthogonality),
        maximum_symmetrizer_defect=symmetrizer,
        maximum_eigenpair_defect=float(eigenpair),
        maximum_rescaling_invariance_defect=rescaling,
        eigenvector_condition_number=float(np.linalg.cond(right)),
        maximum_imaginary_part=maximum_imaginary,
    )


def causal_manufactured_energy_ledger(
    perturbation: np.ndarray,
    perturbation_time_derivative: np.ndarray,
    perturbation_spatial_derivative: np.ndarray,
    forcing: np.ndarray,
    energy_metric: np.ndarray,
    evolution_matrix: np.ndarray,
    energy_flux_metric_derivative: np.ndarray,
    lower_evolution_blocks: dict[str, np.ndarray],
) -> CausalManufacturedEnergyLedger:
    """Close the complete variable-coefficient energy identity pointwise."""

    state = np.asarray(perturbation, dtype=float)
    time_derivative = np.asarray(
        perturbation_time_derivative,
        dtype=float,
    )
    spatial_derivative = np.asarray(
        perturbation_spatial_derivative,
        dtype=float,
    )
    force = np.asarray(forcing, dtype=float)
    grams = np.asarray(energy_metric, dtype=float)
    evolution = np.asarray(evolution_matrix, dtype=float)
    flux_derivative = np.asarray(
        energy_flux_metric_derivative,
        dtype=float,
    )
    n_points = state.shape[0]
    expected_matrix_shape = (n_points, _N_FIELDS, _N_FIELDS)
    if (
        state.shape != (n_points, _N_FIELDS)
        or time_derivative.shape != state.shape
        or spatial_derivative.shape != state.shape
        or force.shape != state.shape
        or grams.shape != expected_matrix_shape
        or evolution.shape != expected_matrix_shape
        or flux_derivative.shape != expected_matrix_shape
        or not lower_evolution_blocks
        or np.any(~np.isfinite(state))
        or np.any(~np.isfinite(time_derivative))
        or np.any(~np.isfinite(spatial_derivative))
        or np.any(~np.isfinite(force))
    ):
        raise ValueError("manufactured energy-ledger inputs are invalid")
    for name, block in lower_evolution_blocks.items():
        values = np.asarray(block, dtype=float)
        if values.shape != expected_matrix_shape:
            raise ValueError(f"lower energy block {name!r} is invalid")

    flux_metric = np.einsum(
        "nij,njk->nik",
        grams,
        evolution,
        optimize=True,
    )
    stored = np.einsum(
        "ni,nij,nj->n",
        state,
        grams,
        time_derivative,
        optimize=True,
    )
    principal = np.einsum(
        "ni,nij,nj->n",
        state,
        flux_metric,
        spatial_derivative,
        optimize=True,
    ) + 0.5 * np.einsum(
        "ni,nij,nj->n",
        state,
        flux_derivative,
        state,
        optimize=True,
    )
    lower = {
        name: np.einsum(
            "ni,nij,njk,nk->n",
            state,
            grams,
            np.asarray(block, dtype=float),
            state,
            optimize=True,
        )
        for name, block in lower_evolution_blocks.items()
    }
    forcing_work = np.einsum(
        "ni,nij,nj->n",
        state,
        grams,
        force,
        optimize=True,
    )
    background = 0.5 * np.einsum(
        "ni,nij,nj->n",
        state,
        flux_derivative,
        state,
        optimize=True,
    )
    reconstructed = (
        np.sum(np.asarray(tuple(lower.values())), axis=0)
        + forcing_work
        + background
    )
    residual = stored + principal - reconstructed
    scale = max(
        float(np.max(np.abs(stored))),
        float(np.max(np.abs(principal))),
        float(np.max(np.abs(reconstructed))),
        np.finfo(float).tiny,
    )
    return CausalManufacturedEnergyLedger(
        stored_energy_rate=stored,
        principal_flux_divergence=principal,
        lower_source_work_by_block=lower,
        manufactured_forcing_work=forcing_work,
        background_gradient_work=background,
        reconstructed_right_hand_side=reconstructed,
        residual=residual,
        maximum_relative_closure_defect=float(
            np.max(np.abs(residual)) / scale
        ),
    )


def causal_fourth_order_centered_derivative(
    values: np.ndarray,
    spacing: float,
) -> np.ndarray:
    """Return a fourth-order centered derivative on the interior points."""

    data = np.asarray(values, dtype=float)
    step = float(spacing)
    if (
        data.ndim < 1
        or data.shape[0] < 5
        or np.any(~np.isfinite(data))
        or not np.isfinite(step)
        or step <= 0.0
    ):
        raise ValueError("fourth-order derivative inputs are invalid")
    return (
        data[:-4]
        - 8.0 * data[1:-3]
        + 8.0 * data[3:-1]
        - data[4:]
    ) / (12.0 * step)
