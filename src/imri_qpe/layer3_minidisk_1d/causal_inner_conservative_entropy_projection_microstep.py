"""Conservative entropy-projected RK2 microstep for the equilibrium core.

The spatial proposal is a periodic explicit-midpoint step written in the
four temporal currents.  A single scalar projection then restores the exact
periodic mathematical entropy without changing any of the four conserved
totals.  The projection direction is built from centered entropy variables,
so its entropy derivative is strictly negative unless the patch is uniform.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext

import numpy as np
from scipy.optimize import root_scalar

from .causal_inner_geometry import KerrSchildColumnGeometry
from .causal_inner_nonlinear_port_atlas import (
    EquilibriumEntropyPoint,
    conditioned_discrete_gradient_radial_flux,
    equilibrium_entropy_variables_decimal,
    equilibrium_mathematical_entropy_decimal,
    equilibrium_temporal_conserved,
    recover_equilibrium_point_from_temporal_conserved,
)


@dataclass(frozen=True)
class EquilibriumPrimitiveSeed:
    density: float
    temperature: float
    radial_velocity_over_c: float
    azimuthal_velocity_over_c: float


@dataclass(frozen=True)
class ConservativeEntropyProjectionResult:
    points: tuple[EquilibriumEntropyPoint, ...]
    seeds: tuple[EquilibriumPrimitiveSeed, ...]
    projection_theta: float
    correction_relative_norm: float
    proposal_entropy_relative_defect: float
    projection_entropy_slope: float
    maximum_recovery_residual: float
    conservation_relative_defect: float
    entropy_relative_defect: float
    projection_converged: bool

    @property
    def passed(self) -> bool:
        return bool(
            self.projection_converged
            and abs(self.projection_theta) <= 1.0
            and self.correction_relative_norm <= 0.05
            and self.maximum_recovery_residual <= 1.0e-11
            and self.conservation_relative_defect <= 2.0e-12
            and self.entropy_relative_defect <= 2.0e-11
        )


def _flux_residual(
    points: tuple[EquilibriumEntropyPoint, ...],
) -> np.ndarray:
    count = len(points)
    if count < 3:
        raise ValueError("periodic microstep needs at least three cells")
    fluxes = tuple(
        conditioned_discrete_gradient_radial_flux(
            points[index], points[(index + 1) % count]
        ).as_decimal()
        for index in range(count)
    )
    with localcontext() as context:
        context.prec = 50
        exact = tuple(
            tuple(
                fluxes[(index - 1) % count][component]
                - fluxes[index][component]
                for component in range(4)
            )
            for index in range(count)
        )
    residual = np.asarray(
        [[float(value) for value in row] for row in exact], dtype=float
    )
    # Preserve the periodic telescoping identity in the binary64 proposal.
    residual[-1] = -np.sum(residual[:-1], axis=0)
    return residual


def _recover_all(
    geometry: KerrSchildColumnGeometry,
    proper_half_thickness: float,
    targets: np.ndarray,
    seeds: tuple[EquilibriumPrimitiveSeed, ...],
) -> tuple[
    tuple[EquilibriumEntropyPoint, ...],
    tuple[EquilibriumPrimitiveSeed, ...],
    float,
]:
    recoveries = []
    for target, seed in zip(targets, seeds, strict=True):
        result = recover_equilibrium_point_from_temporal_conserved(
            geometry,
            proper_half_thickness=proper_half_thickness,
            target_conserved=target,
            initial_density=seed.density,
            initial_temperature=seed.temperature,
            initial_radial_velocity_over_c=seed.radial_velocity_over_c,
            initial_azimuthal_velocity_over_c=seed.azimuthal_velocity_over_c,
        )
        if not result.converged:
            raise RuntimeError("temporal-current recovery failed")
        recoveries.append(result)
    points = tuple(result.point for result in recoveries)
    new_seeds = tuple(
        EquilibriumPrimitiveSeed(
            result.density,
            result.temperature,
            result.radial_velocity_over_c,
            result.azimuthal_velocity_over_c,
        )
        for result in recoveries
    )
    return (
        points,
        new_seeds,
        max(result.scaled_residual_norm for result in recoveries),
    )


def _total_entropy(points: tuple[EquilibriumEntropyPoint, ...]) -> Decimal:
    with localcontext() as context:
        context.prec = 50
        return sum(
            (equilibrium_mathematical_entropy_decimal(point) for point in points),
            Decimal(0),
        )


def _projection_direction(
    proposal: np.ndarray,
    points: tuple[EquilibriumEntropyPoint, ...],
) -> tuple[np.ndarray, np.ndarray, Decimal]:
    """Return a unit-scaled zero-sum direction and its entropy slope."""

    count = len(points)
    scales = np.maximum(np.max(np.abs(proposal), axis=0), 1.0)
    with localcontext() as context:
        context.prec = 50
        variables = tuple(equilibrium_entropy_variables_decimal(point) for point in points)
        means = tuple(
            sum((variables[cell][component] for cell in range(count)), Decimal(0))
            / Decimal(count)
            for component in range(4)
        )
        deviations = tuple(
            tuple(
                variables[cell][component] - means[component]
                for component in range(4)
            )
            for cell in range(count)
        )
        variable_scales = tuple(
            max(abs(deviations[cell][component]) for cell in range(count))
            for component in range(4)
        )
        raw = []
        for cell in range(count):
            row = []
            for component in range(4):
                variable_scale = variable_scales[component]
                if variable_scale == 0:
                    row.append(Decimal(0))
                else:
                    row.append(
                        -Decimal.from_float(float(scales[component]))
                        * deviations[cell][component]
                        / variable_scale
                    )
            raw.append(row)
        raw_norm = max(
            abs(raw[cell][component]) / Decimal.from_float(float(scales[component]))
            for cell in range(count)
            for component in range(4)
        )
        if raw_norm == 0:
            return np.zeros_like(proposal), scales, Decimal(0)
        raw = tuple(
            tuple(value / raw_norm for value in row)
            for row in raw
        )
        slope = sum(
            (
                deviations[cell][component] * raw[cell][component]
                for cell in range(count)
                for component in range(4)
            ),
            Decimal(0),
        )
    direction = np.asarray(
        [[float(value) for value in row] for row in raw], dtype=float
    )
    # Make the last row the exact binary64 negative sum of the preceding rows.
    direction[-1] = -np.sum(direction[:-1], axis=0)
    return direction, scales, slope


def conservative_entropy_projected_midpoint_microstep(
    *,
    geometry: KerrSchildColumnGeometry,
    proper_half_thickness: float,
    points,
    seeds,
    courant_factor: float,
) -> ConservativeEntropyProjectionResult:
    """Advance one bounded periodic equilibrium-core microstep."""

    points = tuple(points)
    seeds = tuple(seeds)
    courant = float(courant_factor)
    if len(points) != len(seeds) or not 0.0 < courant <= 0.05:
        raise ValueError("microstep inputs violate the frozen bound")

    initial = np.asarray(
        [equilibrium_temporal_conserved(point) for point in points], dtype=float
    )
    first_residual = _flux_residual(points)
    midpoint_targets = initial + 0.5 * courant * first_residual
    midpoint_points, midpoint_seeds, midpoint_defect = _recover_all(
        geometry, proper_half_thickness, midpoint_targets, seeds
    )
    second_residual = _flux_residual(midpoint_points)
    proposal = initial + courant * second_residual
    proposal_points, proposal_seeds, proposal_recovery = _recover_all(
        geometry, proper_half_thickness, proposal, midpoint_seeds
    )

    initial_entropy = _total_entropy(points)
    entropy_scale = max(abs(initial_entropy), Decimal(1))
    proposal_entropy = _total_entropy(proposal_points)
    proposal_entropy_defect = float(
        abs(proposal_entropy - initial_entropy) / entropy_scale
    )
    direction, conserved_scales, slope = _projection_direction(
        proposal, proposal_points
    )
    normalized_slope = float(slope / entropy_scale)

    cache: dict[float, tuple[
        tuple[EquilibriumEntropyPoint, ...],
        tuple[EquilibriumPrimitiveSeed, ...],
        float,
        Decimal,
    ]] = {
        0.0: (
            proposal_points,
            proposal_seeds,
            proposal_recovery,
            proposal_entropy,
        )
    }

    def evaluate(theta: float) -> float:
        key = float(theta)
        if key not in cache:
            recovered = _recover_all(
                geometry,
                proper_half_thickness,
                proposal + key * direction,
                proposal_seeds,
            )
            cache[key] = (*recovered, _total_entropy(recovered[0]))
        return float((cache[key][3] - initial_entropy) / entropy_scale)

    value_zero = evaluate(0.0)
    theta = 0.0
    # The entropy tolerance is an acceptance gate, not a reason to turn the
    # projection off.  Skipping a small but resolvable defect at refined
    # substeps introduces a tolerance-dependent branch and can destroy the
    # matched-endpoint RK2 order.
    converged = value_zero == 0.0
    bracket: tuple[float, float] | None = None
    if not converged and normalized_slope < 0.0:
        estimate = -value_zero / normalized_slope
        sign = 1.0 if estimate >= 0.0 else -1.0
        magnitude = max(min(abs(estimate) * 1.25, 1.0), 1.0e-12)
        candidates = []
        while magnitude < 1.0:
            candidates.append(sign * magnitude)
            magnitude *= 2.0
        candidates.append(sign)
        for candidate in candidates:
            try:
                candidate_value = evaluate(candidate)
            except (RuntimeError, ValueError, FloatingPointError):
                continue
            if value_zero * candidate_value <= 0.0:
                bracket = (min(0.0, candidate), max(0.0, candidate))
                break
    if bracket is not None:
        solution = root_scalar(
            evaluate,
            bracket=bracket,
            method="brentq",
            xtol=5.0e-15,
            rtol=1.0e-14,
        )
        theta = float(solution.root)
        converged = bool(solution.converged)
    elif abs(value_zero) <= 2.0e-11:
        # At the arithmetic/recovery floor a sign-changing bracket may not be
        # representable.  Such a point is still admissible under the frozen
        # entropy gate, while the attempted projection remains recorded.
        converged = True

    final_points, final_seeds, final_recovery, final_entropy = cache.get(
        theta,
        (*_recover_all(
            geometry,
            proper_half_thickness,
            proposal + theta * direction,
            proposal_seeds,
        ), None),
    )
    if final_entropy is None:
        final_entropy = _total_entropy(final_points)
    final = np.asarray(
        [equilibrium_temporal_conserved(point) for point in final_points], dtype=float
    )
    conservation_scale = max(float(np.max(np.abs(initial))), 1.0)
    conservation = float(
        np.linalg.norm(np.sum(final - initial, axis=0), ord=np.inf)
        / conservation_scale
    )
    entropy_defect = float(abs(final_entropy - initial_entropy) / entropy_scale)
    correction = float(
        np.max(np.abs(theta * direction) / conserved_scales)
    )
    return ConservativeEntropyProjectionResult(
        final_points,
        final_seeds,
        theta,
        correction,
        proposal_entropy_defect,
        normalized_slope,
        max(midpoint_defect, proposal_recovery, final_recovery),
        conservation,
        entropy_defect,
        bool(converged and entropy_defect <= 2.0e-11),
    )


__all__ = [
    "ConservativeEntropyProjectionResult",
    "EquilibriumPrimitiveSeed",
    "conservative_entropy_projected_midpoint_microstep",
]
