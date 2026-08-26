"""Path-conservative interfaces for the seven-field causal inner model.

The local generalized Maxwell--Cattaneo equations have six exact-flux rows
and one projected Israel--Stewart shear row.  This module therefore exposes a
Dal Maso--LeFloch--Murat (DLM) path jump and a complete-characteristic signed
split.  It deliberately contains no mesh assembly, boundary condition, or
time advance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .causal_inner_generalized_maxwell_cattaneo import (
    GeneralizedMaxwellCattaneoPrincipal,
    generalized_maxwell_cattaneo_local_state,
    generalized_maxwell_cattaneo_principal,
)
from .causal_inner_geometry import KerrSchildColumnGeometry


_N_FIELDS = 7
_EXACT_FLUX_ROWS = np.asarray((0, 1, 2, 3, 5, 6), dtype=int)


@dataclass(frozen=True)
class GeneralizedMaxwellCattaneoPathJump:
    """Complete radial-principal integral along one straight chart path."""

    left_chart: np.ndarray
    right_chart: np.ndarray
    quadrature_order: int
    left_exact_flux_over_c: np.ndarray
    right_exact_flux_over_c: np.ndarray
    exact_flux_jump_over_c: np.ndarray
    total_principal_jump_over_c: np.ndarray
    projected_shear_path_integral_over_c: float
    exact_flux_parity_relative_defect: float


@dataclass(frozen=True)
class GeneralizedMaxwellCattaneoSignedFluctuations:
    """Midpoint complete-eigenbasis split of a seven-field DLM jump."""

    path_jump: GeneralizedMaxwellCattaneoPathJump
    midpoint_principal: GeneralizedMaxwellCattaneoPrincipal
    eigenvalues_over_c: np.ndarray
    characteristic_jump_coefficients: np.ndarray
    characteristic_quadratic_dissipation: float
    dissipation_over_c: np.ndarray
    negative_fluctuation_over_c: np.ndarray
    positive_fluctuation_over_c: np.ndarray
    left_shared_exact_flux_over_c: np.ndarray
    right_shared_exact_flux_over_c: np.ndarray
    split_closure_relative_defect: float
    shared_exact_flux_relative_defect: float


def _chart(values) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (_N_FIELDS,) or np.any(~np.isfinite(result)):
        raise ValueError("seven-field interface chart must be finite and length seven")
    if float(result[1] ** 2 + result[2] ** 2) >= 1.0:
        raise ValueError("horizontal interface velocity must be subluminal")
    if abs(float(result[6])) >= 1.0:
        raise ValueError("vertical interface velocity must be subluminal")
    return result


def _exact_flux_vector(local_state) -> np.ndarray:
    """Embed the six exact physical/material fluxes in seven-row order."""

    flux6 = np.asarray(local_state.conservative_flux6_over_c, dtype=float)
    result = np.zeros(_N_FIELDS, dtype=float)
    result[:4] = flux6[:4]
    result[5:] = flux6[4:]
    return result


def _relative_maximum(defect: np.ndarray, *references: np.ndarray) -> float:
    scale = max(
        *(float(np.max(np.abs(np.asarray(item)))) for item in references),
        np.finfo(float).tiny,
    )
    return float(np.max(np.abs(np.asarray(defect))) / scale)


def generalized_maxwell_cattaneo_path_jump(
    geometry: KerrSchildColumnGeometry,
    left_chart,
    right_chart,
    *,
    proper_vertical_frequency: float,
    alpha: float,
    stress_factor: float = 1.0,
    quadrature_order: int = 8,
    derivative_step_factor: float = 1.0,
) -> GeneralizedMaxwellCattaneoPathJump:
    """Integrate ``M_R(Psi) Psi_s`` along the declared straight DLM path."""

    left = _chart(left_chart)
    right = _chart(right_chart)
    order = int(quadrature_order)
    if order < 2:
        raise ValueError("quadrature_order must be at least two")
    derivative_step_factor = float(derivative_step_factor)
    if not np.isfinite(derivative_step_factor) or derivative_step_factor <= 0.0:
        raise ValueError("derivative_step_factor must be finite and positive")

    delta = right - left
    nodes, weights = np.polynomial.legendre.leggauss(order)
    total = np.zeros(_N_FIELDS, dtype=float)
    for node, weight in zip(nodes, weights, strict=True):
        fraction = 0.5 * (float(node) + 1.0)
        principal = generalized_maxwell_cattaneo_principal(
            geometry,
            left + fraction * delta,
            proper_vertical_frequency=proper_vertical_frequency,
            alpha=alpha,
            stress_factor=stress_factor,
            derivative_step_factor=derivative_step_factor,
        )
        total += 0.5 * float(weight) * (principal.radial_matrix @ delta)

    left_state = generalized_maxwell_cattaneo_local_state(
        geometry,
        left,
        proper_vertical_frequency=proper_vertical_frequency,
        alpha=alpha,
        stress_factor=stress_factor,
    )
    right_state = generalized_maxwell_cattaneo_local_state(
        geometry,
        right,
        proper_vertical_frequency=proper_vertical_frequency,
        alpha=alpha,
        stress_factor=stress_factor,
    )
    left_flux = _exact_flux_vector(left_state)
    right_flux = _exact_flux_vector(right_state)
    exact_jump = right_flux - left_flux
    exact_defect = total[_EXACT_FLUX_ROWS] - exact_jump[_EXACT_FLUX_ROWS]
    parity = _relative_maximum(
        exact_defect,
        total[_EXACT_FLUX_ROWS],
        exact_jump[_EXACT_FLUX_ROWS],
    )
    return GeneralizedMaxwellCattaneoPathJump(
        left_chart=np.array(left, copy=True),
        right_chart=np.array(right, copy=True),
        quadrature_order=order,
        left_exact_flux_over_c=left_flux,
        right_exact_flux_over_c=right_flux,
        exact_flux_jump_over_c=exact_jump,
        total_principal_jump_over_c=np.asarray(total, dtype=float),
        projected_shear_path_integral_over_c=float(total[4]),
        exact_flux_parity_relative_defect=parity,
    )


def _real_array(values: np.ndarray, *, name: str, tolerance: float) -> np.ndarray:
    candidate = np.asarray(values)
    imaginary = float(np.max(np.abs(np.imag(candidate))))
    scale = max(float(np.max(np.abs(candidate))), 1.0)
    if imaginary > tolerance * scale:
        raise ValueError(f"{name} is not real within the declared tolerance")
    return np.asarray(np.real(candidate), dtype=float)


def generalized_maxwell_cattaneo_signed_fluctuations(
    geometry: KerrSchildColumnGeometry,
    left_chart,
    right_chart,
    *,
    proper_vertical_frequency: float,
    alpha: float,
    stress_factor: float = 1.0,
    quadrature_order: int = 8,
    derivative_step_factor: float = 1.0,
    imaginary_tolerance: float = 1.0e-10,
) -> GeneralizedMaxwellCattaneoSignedFluctuations:
    """Split a DLM jump with the complete midpoint generalized eigensystem.

    The scaled generalized pencil satisfies ``B_s R = A_s R Lambda``.
    If ``z = delta_q / column_scale``, the dimensional equation-space
    dissipation is ``row_scale * A_s R |Lambda| R^-1 z``.
    """

    jump = generalized_maxwell_cattaneo_path_jump(
        geometry,
        left_chart,
        right_chart,
        proper_vertical_frequency=proper_vertical_frequency,
        alpha=alpha,
        stress_factor=stress_factor,
        quadrature_order=quadrature_order,
        derivative_step_factor=derivative_step_factor,
    )
    midpoint = 0.5 * (jump.left_chart + jump.right_chart)
    principal = generalized_maxwell_cattaneo_principal(
        geometry,
        midpoint,
        proper_vertical_frequency=proper_vertical_frequency,
        alpha=alpha,
        stress_factor=stress_factor,
        derivative_step_factor=derivative_step_factor,
    )
    eigenvalues = _real_array(
        principal.eigenvalues_over_c,
        name="generalized eigenvalues",
        tolerance=imaginary_tolerance,
    )
    eigenvectors = _real_array(
        principal.right_eigenvectors_scaled,
        name="generalized right eigenvectors",
        tolerance=imaginary_tolerance,
    )
    scaled_delta = (
        jump.right_chart - jump.left_chart
    ) / principal.primitive_column_scales
    coefficients = np.linalg.solve(eigenvectors, scaled_delta)
    scaled_dissipation = principal.scaled_temporal_matrix @ (
        eigenvectors @ (np.abs(eigenvalues) * coefficients)
    )
    dissipation = principal.equation_row_scales * scaled_dissipation
    negative = 0.5 * (jump.total_principal_jump_over_c - dissipation)
    positive = 0.5 * (jump.total_principal_jump_over_c + dissipation)
    closure = negative + positive - jump.total_principal_jump_over_c
    split_defect = _relative_maximum(
        closure,
        negative,
        positive,
        jump.total_principal_jump_over_c,
    )
    left_shared = jump.left_exact_flux_over_c[_EXACT_FLUX_ROWS] + negative[
        _EXACT_FLUX_ROWS
    ]
    right_shared = jump.right_exact_flux_over_c[_EXACT_FLUX_ROWS] - positive[
        _EXACT_FLUX_ROWS
    ]
    shared_defect = _relative_maximum(
        left_shared - right_shared,
        left_shared,
        right_shared,
    )
    quadratic = float(np.sum(np.abs(eigenvalues) * np.abs(coefficients) ** 2))
    return GeneralizedMaxwellCattaneoSignedFluctuations(
        path_jump=jump,
        midpoint_principal=principal,
        eigenvalues_over_c=eigenvalues,
        characteristic_jump_coefficients=np.asarray(coefficients, dtype=float),
        characteristic_quadratic_dissipation=quadratic,
        dissipation_over_c=np.asarray(dissipation, dtype=float),
        negative_fluctuation_over_c=np.asarray(negative, dtype=float),
        positive_fluctuation_over_c=np.asarray(positive, dtype=float),
        left_shared_exact_flux_over_c=np.asarray(left_shared, dtype=float),
        right_shared_exact_flux_over_c=np.asarray(right_shared, dtype=float),
        split_closure_relative_defect=split_defect,
        shared_exact_flux_relative_defect=shared_defect,
    )


__all__ = (
    "GeneralizedMaxwellCattaneoPathJump",
    "GeneralizedMaxwellCattaneoSignedFluctuations",
    "generalized_maxwell_cattaneo_path_jump",
    "generalized_maxwell_cattaneo_signed_fluctuations",
)
