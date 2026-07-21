"""Finite-neighborhood certification tools for Rusanov branch switches.

The production causal flux uses an exact ``max(abs(characteristic speed))``.
At a tie, or in a finite neighborhood that contains a tie, its tangent is a
set of fixed-branch generators rather than one ordinary Jacobian.  This
module does not alter or smooth that maximum.  It supplies conservative
bounds for the resulting switched linear system and for a separately
certified nonlinear finite-amplitude remainder.

All norms below are Euclidean after the optional diagonal state-metric
similarity transform.  If ``W = diag(state_metric_diagonal)``, the state norm
is ``sqrt(x.T @ W @ x)``.  Generator rank-one factors ``u_i v_i.T`` are
transformed consistently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RusanovCandidateCoverage:
    """Finite-neighborhood coverage of one face's speed candidates."""

    face_index: int
    controlling_candidate: int
    neighborhood_radius: float
    base_speed_gaps: np.ndarray
    gap_variation_bounds: np.ndarray
    possible_candidates: np.ndarray
    represented_candidates: np.ndarray
    suppressed_candidates: np.ndarray
    suppressed_candidate_effect_bounds: np.ndarray
    suppression_remainder_rate_required: float
    suppression_remainder_rate_reserved: float
    unresolved_candidates: np.ndarray
    variation_bounds_certified: bool
    suppression_certified: bool
    binding: bool
    passed: bool

    def as_dict(self) -> dict:
        """Return a JSON-compatible audit payload."""

        return {
            "face_index": self.face_index,
            "controlling_candidate": self.controlling_candidate,
            "neighborhood_radius": self.neighborhood_radius,
            "base_speed_gaps": self.base_speed_gaps.tolist(),
            "gap_variation_bounds": self.gap_variation_bounds.tolist(),
            "possible_candidates": self.possible_candidates.tolist(),
            "represented_candidates": self.represented_candidates.tolist(),
            "suppressed_candidates": self.suppressed_candidates.tolist(),
            "suppressed_candidate_effect_bounds": (
                self.suppressed_candidate_effect_bounds.tolist()
            ),
            "suppression_remainder_rate_required": (
                self.suppression_remainder_rate_required
            ),
            "suppression_remainder_rate_reserved": (
                self.suppression_remainder_rate_reserved
            ),
            "unresolved_candidates": self.unresolved_candidates.tolist(),
            "variation_bounds_certified": self.variation_bounds_certified,
            "suppression_certified": self.suppression_certified,
            "binding": self.binding,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class RusanovFiniteNeighborhoodBound:
    """Variation-of-constants enclosure for finite branch switching."""

    horizon_seconds: float
    certified_neighborhood_radius: float | None
    neighborhood_bounds_global: bool
    neighborhood_containment_passed: bool
    logarithmic_norm_per_s: float
    aggregate_switching_radius_per_s: float
    switched_growth_rate_per_s: float
    initial_state_radius: float
    nonlinear_remainder_rate: float
    nominal_state_radius_bound: float
    switched_state_radius_bound: float
    branch_state_deviation_bound: float
    nonlinear_state_deviation_bound: float
    total_state_deviation_bound: float
    per_output_dynamic_bounds: np.ndarray
    per_output_direct_bounds: np.ndarray
    per_output_nonlinear_bounds: np.ndarray
    per_output_total_bounds: np.ndarray
    per_output_gate_fractions: np.ndarray | None
    maximum_gate_fraction: float | None
    allowed_maximum_gate_fraction: float
    candidate_coverage_certified: bool
    nonlinear_remainder_certified: bool
    binding: bool
    passed: bool
    semantics: str

    def as_dict(self) -> dict:
        """Return a JSON-compatible audit payload."""

        return {
            "horizon_seconds": self.horizon_seconds,
            "certified_neighborhood_radius": (
                self.certified_neighborhood_radius
            ),
            "neighborhood_bounds_global": self.neighborhood_bounds_global,
            "neighborhood_containment_passed": (
                self.neighborhood_containment_passed
            ),
            "logarithmic_norm_per_s": self.logarithmic_norm_per_s,
            "aggregate_switching_radius_per_s": (
                self.aggregate_switching_radius_per_s
            ),
            "switched_growth_rate_per_s": self.switched_growth_rate_per_s,
            "initial_state_radius": self.initial_state_radius,
            "nonlinear_remainder_rate": self.nonlinear_remainder_rate,
            "nominal_state_radius_bound": self.nominal_state_radius_bound,
            "switched_state_radius_bound": self.switched_state_radius_bound,
            "branch_state_deviation_bound": (
                self.branch_state_deviation_bound
            ),
            "nonlinear_state_deviation_bound": (
                self.nonlinear_state_deviation_bound
            ),
            "total_state_deviation_bound": self.total_state_deviation_bound,
            "per_output_dynamic_bounds": (
                self.per_output_dynamic_bounds.tolist()
            ),
            "per_output_direct_bounds": self.per_output_direct_bounds.tolist(),
            "per_output_nonlinear_bounds": (
                self.per_output_nonlinear_bounds.tolist()
            ),
            "per_output_total_bounds": self.per_output_total_bounds.tolist(),
            "per_output_gate_fractions": (
                None
                if self.per_output_gate_fractions is None
                else self.per_output_gate_fractions.tolist()
            ),
            "maximum_gate_fraction": self.maximum_gate_fraction,
            "allowed_maximum_gate_fraction": (
                self.allowed_maximum_gate_fraction
            ),
            "candidate_coverage_certified": (
                self.candidate_coverage_certified
            ),
            "nonlinear_remainder_certified": (
                self.nonlinear_remainder_certified
            ),
            "binding": self.binding,
            "passed": self.passed,
            "semantics": self.semantics,
        }


def rusanov_gap_variation_bounds(
    neighborhood_radius: float,
    gap_gradient_norm_bounds: np.ndarray,
    *,
    gap_hessian_norm_bounds: np.ndarray | None = None,
) -> np.ndarray:
    """Bound speed-gap changes on a state ball.

    The caller must establish that the supplied gradient and Hessian norms are
    valid throughout the complete finite neighborhood.  This helper only
    evaluates

    ``radius * gradient_bound + 0.5 * radius**2 * hessian_bound``.
    """

    radius = float(neighborhood_radius)
    gradients = np.asarray(gap_gradient_norm_bounds, dtype=float)
    if radius < 0.0 or not np.isfinite(radius):
        raise ValueError(
            "the Rusanov neighborhood radius must be finite and nonnegative"
        )
    if (
        gradients.ndim != 1
        or np.any(~np.isfinite(gradients))
        or np.any(gradients < 0.0)
    ):
        raise ValueError("Rusanov gap-gradient bounds must be finite and nonnegative")
    if gap_hessian_norm_bounds is None:
        hessians = np.zeros_like(gradients)
    else:
        hessians = np.asarray(gap_hessian_norm_bounds, dtype=float)
        if (
            hessians.shape != gradients.shape
            or np.any(~np.isfinite(hessians))
            or np.any(hessians < 0.0)
        ):
            raise ValueError("Rusanov gap-Hessian bounds are invalid")
    return radius * gradients + 0.5 * radius * radius * hessians


def quadratic_taylor_remainder_bound(
    neighborhood_radius: float,
    hessian_operator_norm_bound: np.ndarray | float,
) -> np.ndarray:
    """Return ``0.5 * H * radius**2`` from a certified Hessian bound.

    This is a calculation helper, not a Hessian-certification procedure.  A
    sampled secant or an anchor Hessian does not by itself establish the
    required uniform finite-neighborhood operator-norm bound.
    """

    radius = float(neighborhood_radius)
    hessian = np.asarray(hessian_operator_norm_bound, dtype=float)
    if radius < 0.0 or not np.isfinite(radius):
        raise ValueError(
            "the Taylor neighborhood radius must be finite and nonnegative"
        )
    if np.any(~np.isfinite(hessian)) or np.any(hessian < 0.0):
        raise ValueError("Taylor Hessian bounds must be finite and nonnegative")
    return np.asarray(0.5 * radius * radius * hessian, dtype=float)


def certify_rusanov_candidate_coverage(
    *,
    face_index: int,
    candidate_absolute_speeds: np.ndarray,
    gap_variation_bounds: np.ndarray,
    neighborhood_radius: float,
    represented_candidates: np.ndarray | tuple[int, ...] | list[int],
    suppressed_candidates: np.ndarray | tuple[int, ...] | list[int] | None = None,
    suppressed_candidate_effect_bounds: np.ndarray | None = None,
    suppression_remainder_rate_reserved: float = 0.0,
    variation_bounds_certified: bool,
    suppression_certified: bool = False,
) -> RusanovCandidateCoverage:
    """Certify that every candidate able to win is represented or suppressed.

    Let ``c`` be the controlling candidate at the anchor and
    ``g_j = a_c - a_j``.  A candidate is excluded from the finite
    neighborhood only when a certified bound on ``|g_j(x)-g_j(x0)|`` is
    strictly smaller than its anchor gap.  This remains valid even if another
    branch becomes the controller.  The nominal controller is represented
    automatically.

    ``represented_candidates`` and ``suppressed_candidates`` may be integer
    index sequences or Boolean masks.  A represented possible noncontroller
    is not sufficient by itself: the cached wrapper later requires the pair
    ``(face_index, candidate)`` to identify an actual rank-one branch factor.

    Suppression is intentionally strict.  The caller must supply a certified
    uniform vector-field effect bound for every suppressed candidate.  An
    exact-zero bound can be removed directly.  Positive bounds are removable
    only when their sum is explicitly reserved in
    ``suppression_remainder_rate_reserved``; the cached wrapper then verifies
    that the declared nonlinear vector-field remainder includes the sum of
    those per-face reserves.  Merely setting ``suppression_certified`` is not
    enough to omit a nonzero branch.
    """

    if isinstance(face_index, (bool, np.bool_)):
        raise ValueError("a Rusanov face index must be a positive integer")
    face = int(face_index)
    if face < 1 or float(face) != float(face_index):
        raise ValueError("a Rusanov face index must be a positive integer")
    speeds = np.asarray(candidate_absolute_speeds, dtype=float)
    variation = np.asarray(gap_variation_bounds, dtype=float)
    radius = float(neighborhood_radius)
    suppression_reserve = float(suppression_remainder_rate_reserved)
    if radius < 0.0 or not np.isfinite(radius):
        raise ValueError(
            "the candidate-coverage radius must be finite and nonnegative"
        )
    if (
        speeds.ndim != 1
        or speeds.size == 0
        or np.any(~np.isfinite(speeds))
        or np.any(speeds < 0.0)
    ):
        raise ValueError("Rusanov candidate speeds must be a finite nonnegative vector")
    if (
        variation.shape != speeds.shape
        or np.any(~np.isfinite(variation))
        or np.any(variation < 0.0)
    ):
        raise ValueError("Rusanov speed-gap variation bounds are invalid")
    if suppression_reserve < 0.0 or not np.isfinite(suppression_reserve):
        raise ValueError(
            "the Rusanov suppression-remainder reserve must be finite and "
            "nonnegative"
        )
    controlling = int(np.argmax(speeds))
    gaps = float(speeds[controlling]) - speeds
    possible = gaps <= variation
    possible[controlling] = True
    represented = _candidate_mask(represented_candidates, speeds.size)
    represented[controlling] = True
    suppressed = (
        np.zeros(speeds.size, dtype=bool)
        if suppressed_candidates is None
        else _candidate_mask(suppressed_candidates, speeds.size)
    )
    if suppressed_candidate_effect_bounds is None:
        if np.any(suppressed):
            raise ValueError(
                "suppressed Rusanov candidates require certified effect bounds"
            )
        suppression_effects = np.zeros(speeds.size, dtype=float)
    else:
        suppression_effects = np.asarray(
            suppressed_candidate_effect_bounds,
            dtype=float,
        )
        if (
            suppression_effects.shape != speeds.shape
            or np.any(~np.isfinite(suppression_effects))
            or np.any(suppression_effects < 0.0)
        ):
            raise ValueError(
                "suppressed Rusanov candidate-effect bounds are invalid"
            )
    possible_suppressed = possible & suppressed
    positive_possible_suppressed = possible_suppressed & (
        suppression_effects > 0.0
    )
    required_suppression_remainder = float(
        np.sum(suppression_effects[positive_possible_suppressed])
    )
    suppression_reserve_covers = bool(
        suppression_reserve >= required_suppression_remainder
    )
    exact_zero_suppressed = possible_suppressed & (
        suppression_effects == 0.0
    )
    remainder_reserved_suppressed = (
        positive_possible_suppressed & suppression_reserve_covers
    )
    effective_suppressed = (
        exact_zero_suppressed | remainder_reserved_suppressed
    ) & bool(suppression_certified)
    unresolved = possible & ~(represented | effective_suppressed)
    binding = bool(
        variation_bounds_certified
        and (
            not np.any(possible_suppressed)
            or (
                suppression_certified
                and suppression_reserve_covers
            )
        )
    )
    return RusanovCandidateCoverage(
        face_index=face,
        controlling_candidate=controlling,
        neighborhood_radius=radius,
        base_speed_gaps=np.asarray(gaps, dtype=float),
        gap_variation_bounds=np.asarray(variation, dtype=float),
        possible_candidates=np.flatnonzero(possible),
        represented_candidates=np.flatnonzero(represented),
        suppressed_candidates=np.flatnonzero(suppressed),
        suppressed_candidate_effect_bounds=np.asarray(
            suppression_effects,
            dtype=float,
        ),
        suppression_remainder_rate_required=(
            required_suppression_remainder
        ),
        suppression_remainder_rate_reserved=suppression_reserve,
        unresolved_candidates=np.flatnonzero(unresolved),
        variation_bounds_certified=bool(variation_bounds_certified),
        suppression_certified=bool(suppression_certified),
        binding=binding,
        passed=bool(binding and not np.any(unresolved)),
    )


def certify_rusanov_finite_neighborhood(
    *,
    base_generator_per_s: np.ndarray,
    output_operator: np.ndarray,
    generator_left_factors: np.ndarray,
    generator_right_factors: np.ndarray,
    horizon_seconds: float,
    output_gates: np.ndarray | None = None,
    direct_output_deltas: np.ndarray | None = None,
    coefficient_bounds: np.ndarray | None = None,
    state_metric_diagonal: np.ndarray | None = None,
    initial_state_radius: float = 1.0,
    certified_neighborhood_radius: float | None = None,
    neighborhood_bounds_global: bool = False,
    nonlinear_remainder_rate: float = 0.0,
    nonlinear_output_remainder_bounds: np.ndarray | None = None,
    candidate_coverage_certified: bool,
    nonlinear_remainder_certified: bool,
    maximum_gate_fraction: float = 1.0e-2,
) -> RusanovFiniteNeighborhoodBound:
    """Bound arbitrary finite-neighborhood Rusanov switching.

    In metric-scaled coordinates, consider

    ``xdot = (L + sum(theta_i(t) u_i v_i.T)) x + r(t)``,

    with measurable coefficients ``|theta_i(t)| <= coefficient_bounds[i]``
    and a certified uniform remainder ``||r(t)|| <= nonlinear_remainder_rate``.
    No assumption is made about the number, ordering, or frequency of branch
    switches.  The aggregate perturbation radius is bounded by

    ``rho = sum b_i ||u_i|| ||v_i||``.

    If ``mu`` is the Euclidean logarithmic norm of ``L``, then

    ``||x(h)|| <= exp((mu + rho) h) ||x(0)|| + eta*phi(mu+rho,h)``

    and variation of constants gives

    ``||x(h)-exp(Lh)x(0)|| <=``
    ``exp(mu h)(exp(rho h)-1)||x(0)|| + eta*phi(mu+rho,h)``.

    ``direct_output_deltas[i]`` represents a branch-specific output operator
    difference.  A triangle bound covers simultaneous final-time output
    branch changes.  ``nonlinear_output_remainder_bounds`` must separately
    enclose nonlinear observable remainders over the neighborhood.

    A result is binding only when candidate coverage and both vector-field and
    output remainders have been independently certified by the caller, and
    the resulting state-radius bound stays inside the neighborhood where those
    contracts hold.  Set ``neighborhood_bounds_global`` only for genuinely
    global bounds (normally just manufactured linear tests).  The nonlinear
    remainder must include finite-amplitude variation of the nominal vector
    field and of the frozen rank-one branch factors.  The output remainder
    must likewise include variation of the direct branch-output operators.
    The helper deliberately refuses to infer these contracts from sampled
    finite differences.
    """

    generator = np.asarray(base_generator_per_s, dtype=float)
    outputs = np.asarray(output_operator, dtype=float)
    left = np.asarray(generator_left_factors, dtype=float)
    right = np.asarray(generator_right_factors, dtype=float)
    if (
        generator.ndim != 2
        or generator.shape[0] != generator.shape[1]
        or np.any(~np.isfinite(generator))
    ):
        raise ValueError("the base Rusanov generator must be finite and square")
    state_count = generator.shape[0]
    if (
        outputs.ndim != 2
        or outputs.shape[1] != state_count
        or np.any(~np.isfinite(outputs))
    ):
        raise ValueError("the Rusanov output operator has an incompatible shape")
    if (
        left.ndim != 2
        or left.shape[0] != state_count
        or right.shape != left.shape
        or np.any(~np.isfinite(left))
        or np.any(~np.isfinite(right))
    ):
        raise ValueError("the Rusanov rank-one generator factors are invalid")
    branch_count = left.shape[1]
    horizon = float(horizon_seconds)
    radius = float(initial_state_radius)
    remainder_rate = float(nonlinear_remainder_rate)
    allowed = float(maximum_gate_fraction)
    neighborhood_radius = (
        None
        if certified_neighborhood_radius is None
        else float(certified_neighborhood_radius)
    )
    if horizon < 0.0 or not np.isfinite(horizon):
        raise ValueError("the Rusanov horizon must be finite and nonnegative")
    if radius < 0.0 or not np.isfinite(radius):
        raise ValueError("the initial-state radius must be finite and nonnegative")
    if remainder_rate < 0.0 or not np.isfinite(remainder_rate):
        raise ValueError("the nonlinear remainder rate must be finite and nonnegative")
    if allowed < 0.0 or not np.isfinite(allowed):
        raise ValueError("the finite-branch gate must be finite and nonnegative")
    if neighborhood_radius is not None and (
        neighborhood_radius < 0.0 or not np.isfinite(neighborhood_radius)
    ):
        raise ValueError(
            "the certified Rusanov neighborhood radius must be finite and "
            "nonnegative"
        )

    coefficients = (
        np.ones(branch_count, dtype=float)
        if coefficient_bounds is None
        else np.asarray(coefficient_bounds, dtype=float)
    )
    if (
        coefficients.shape != (branch_count,)
        or np.any(~np.isfinite(coefficients))
        or np.any(coefficients < 0.0)
    ):
        raise ValueError("Rusanov branch coefficient bounds are invalid")
    metric = (
        np.ones(state_count, dtype=float)
        if state_metric_diagonal is None
        else np.asarray(state_metric_diagonal, dtype=float)
    )
    if (
        metric.shape != (state_count,)
        or np.any(~np.isfinite(metric))
        or np.any(metric <= 0.0)
    ):
        raise ValueError("the Rusanov state metric must be finite and positive")
    square_root_metric = np.sqrt(metric)
    inverse_square_root_metric = 1.0 / square_root_metric
    transformed_generator = (
        square_root_metric[:, None]
        * generator
        * inverse_square_root_metric[None, :]
    )
    transformed_outputs = outputs * inverse_square_root_metric[None, :]
    transformed_left = square_root_metric[:, None] * left
    transformed_right = inverse_square_root_metric[:, None] * right

    direct = _direct_output_array(
        direct_output_deltas,
        branch_count=branch_count,
        output_count=outputs.shape[0],
        state_count=state_count,
    )
    transformed_direct = direct * inverse_square_root_metric[None, None, :]
    nonlinear_output = (
        np.zeros(outputs.shape[0], dtype=float)
        if nonlinear_output_remainder_bounds is None
        else np.asarray(nonlinear_output_remainder_bounds, dtype=float)
    )
    if (
        nonlinear_output.shape != (outputs.shape[0],)
        or np.any(~np.isfinite(nonlinear_output))
        or np.any(nonlinear_output < 0.0)
    ):
        raise ValueError("nonlinear Rusanov output remainders are invalid")
    gates = None if output_gates is None else np.asarray(output_gates, dtype=float)
    if gates is not None and (
        gates.shape != (outputs.shape[0],)
        or np.any(~np.isfinite(gates))
        or np.any(gates <= 0.0)
    ):
        raise ValueError("Rusanov output gates must be finite and positive")

    symmetric = 0.5 * (
        transformed_generator + transformed_generator.T
    )
    logarithmic_norm = float(np.linalg.eigvalsh(symmetric)[-1])
    factor_norm_products = (
        np.linalg.norm(transformed_left, axis=0)
        * np.linalg.norm(transformed_right, axis=0)
    )
    switching_radius = float(np.dot(coefficients, factor_norm_products))
    switched_growth_rate = logarithmic_norm + switching_radius
    nominal_radius_bound = _nonnegative_product(
        radius,
        _safe_exponential(logarithmic_norm * horizon),
    )
    switched_homogeneous_radius = _nonnegative_product(
        radius,
        _safe_exponential(switched_growth_rate * horizon),
    )
    nonlinear_state_bound = _nonnegative_product(
        remainder_rate,
        _exponential_integral(switched_growth_rate, horizon),
    )
    switched_state_bound = switched_homogeneous_radius + nonlinear_state_bound
    branch_state_bound = _nonnegative_product(
        radius,
        _exponential_difference(
            logarithmic_norm,
            switching_radius,
            horizon,
        ),
    )
    total_state_bound = branch_state_bound + nonlinear_state_bound

    output_row_norms = np.linalg.norm(transformed_outputs, axis=1)
    dynamic_bounds = _nonnegative_array_product(
        output_row_norms,
        total_state_bound,
    )
    direct_row_bounds = np.einsum(
        "i,io->o",
        coefficients,
        np.linalg.norm(transformed_direct, axis=2),
        optimize=True,
    )
    direct_bounds = _nonnegative_array_product(
        direct_row_bounds,
        switched_state_bound,
    )
    total_output_bounds = dynamic_bounds + direct_bounds + nonlinear_output
    if gates is None:
        gate_fractions = None
        maximum_fraction = None
        within_gate = True
    else:
        gate_fractions = total_output_bounds / gates
        maximum_fraction = float(np.max(gate_fractions, initial=0.0))
        within_gate = bool(maximum_fraction <= allowed)
    containment_passed = bool(
        neighborhood_bounds_global
        or (
            neighborhood_radius is not None
            and switched_state_bound <= neighborhood_radius
        )
    )
    binding = bool(
        candidate_coverage_certified
        and nonlinear_remainder_certified
        and containment_passed
    )
    return RusanovFiniteNeighborhoodBound(
        horizon_seconds=horizon,
        certified_neighborhood_radius=neighborhood_radius,
        neighborhood_bounds_global=bool(neighborhood_bounds_global),
        neighborhood_containment_passed=containment_passed,
        logarithmic_norm_per_s=logarithmic_norm,
        aggregate_switching_radius_per_s=switching_radius,
        switched_growth_rate_per_s=switched_growth_rate,
        initial_state_radius=radius,
        nonlinear_remainder_rate=remainder_rate,
        nominal_state_radius_bound=nominal_radius_bound,
        switched_state_radius_bound=switched_state_bound,
        branch_state_deviation_bound=branch_state_bound,
        nonlinear_state_deviation_bound=nonlinear_state_bound,
        total_state_deviation_bound=total_state_bound,
        per_output_dynamic_bounds=np.asarray(dynamic_bounds, dtype=float),
        per_output_direct_bounds=np.asarray(direct_bounds, dtype=float),
        per_output_nonlinear_bounds=np.asarray(nonlinear_output, dtype=float),
        per_output_total_bounds=np.asarray(total_output_bounds, dtype=float),
        per_output_gate_fractions=(
            None
            if gate_fractions is None
            else np.asarray(gate_fractions, dtype=float)
        ),
        maximum_gate_fraction=maximum_fraction,
        allowed_maximum_gate_fraction=allowed,
        candidate_coverage_certified=bool(candidate_coverage_certified),
        nonlinear_remainder_certified=bool(nonlinear_remainder_certified),
        binding=binding,
        passed=bool(binding and within_gate),
        semantics=(
            "Euclidean logarithmic-norm and variation-of-constants bound "
            "after the declared diagonal state-metric transform; arbitrary "
            "measurable simultaneous rank-one switching is included; the "
            "finite-neighborhood candidate set and nonlinear vector-field/"
            "output remainder must be certified independently, and the "
            "enclosed trajectory must remain inside that neighborhood"
        ),
    )


def certify_cached_rusanov_finite_neighborhood(
    operator_arrays: dict,
    *,
    output_operator: np.ndarray,
    output_gates: np.ndarray,
    horizon_seconds: float,
    direct_output_deltas: np.ndarray | None = None,
    candidate_coverages: tuple[RusanovCandidateCoverage, ...] = (),
    factor_face_indices: np.ndarray | None = None,
    factor_candidate_indices: np.ndarray | None = None,
    nonlinear_remainder_rate: float | None = None,
    nonlinear_output_remainder_bounds: np.ndarray | None = None,
    nonlinear_remainder_certified: bool = False,
    coefficient_bounds: np.ndarray | None = None,
    initial_state_radius: float = 1.0,
    certified_neighborhood_radius: float | None = None,
    maximum_gate_fraction: float = 1.0e-2,
) -> RusanovFiniteNeighborhoodBound:
    """Apply the enclosure to one WP10c8i-style operator cache.

    Candidate coverage is required exactly once for every interior face,
    including faces with no kink factor at the anchor: a finite perturbation
    can make a currently distant branch competitive.  The expected physical
    face identities are ``1, ..., N-1``, inferred from the five-field
    generator dimension.  Every coverage must be binding and pass.

    Every represented possible noncontroller must map to a real cached
    rank-one factor through explicit parallel face/candidate metadata.  The
    metadata can be supplied directly or stored as
    ``production_rusanov_kink_face_indices`` and
    ``production_rusanov_kink_competitor_codes``.  Missing, duplicate, or
    invalid factor identities make the result nonbinding.  This prevents one
    face's coverage or one cached factor from being reused implicitly for a
    different face/candidate pair.

    The cache cannot establish a nonlinear remainder.  If
    ``nonlinear_remainder_rate`` is omitted, zero is used diagnostically while
    ``nonlinear_remainder_certified`` remains false, so the result cannot pass.
    """

    dynamic = np.asarray(operator_arrays["dynamic"], dtype=float)
    if dynamic.ndim != 2 or dynamic.shape[0] != dynamic.shape[1]:
        raise ValueError("the cached Rusanov dynamic matrix must be square")
    if dynamic.shape[0] % 5 != 0:
        raise ValueError("the cached Rusanov state does not have five fields per cell")
    expected_face_count = max(dynamic.shape[0] // 5 - 1, 0)
    expected_faces = set(range(1, expected_face_count + 1))
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
    if (
        left.ndim != 2
        or left.shape[0] != dynamic.shape[0]
        or right.shape != left.shape
    ):
        raise ValueError("the cached Rusanov rank-one factors are invalid")
    factor_count = int(left.shape[1])
    factor_faces_source = (
        factor_face_indices
        if factor_face_indices is not None
        else operator_arrays.get("production_rusanov_kink_face_indices")
    )
    factor_candidates_source = (
        factor_candidate_indices
        if factor_candidate_indices is not None
        else operator_arrays.get(
            "production_rusanov_kink_competitor_codes"
        )
    )
    factor_faces = _integer_metadata_vector(
        factor_faces_source,
        expected_size=factor_count,
    )
    factor_candidates = _integer_metadata_vector(
        factor_candidates_source,
        expected_size=factor_count,
    )
    factor_metadata_certified = bool(
        factor_faces is not None
        and factor_candidates is not None
        and np.all(factor_faces >= 1)
        and np.all(factor_faces <= expected_face_count)
        and np.all(factor_candidates >= 0)
    )
    factor_pairs: set[tuple[int, int]] = set()
    if factor_metadata_certified:
        factor_pairs = set(
            zip(
                np.asarray(factor_faces, dtype=int).tolist(),
                np.asarray(factor_candidates, dtype=int).tolist(),
                strict=True,
            )
        )
        nonzero_factors = (
            np.linalg.norm(left, axis=0) * np.linalg.norm(right, axis=0)
        ) > 0.0
        factor_metadata_certified = bool(
            len(factor_pairs) == factor_count
            and np.all(nonzero_factors)
        )

    coverage_faces = [int(row.face_index) for row in candidate_coverages]
    complete_unique_face_coverage = bool(
        len(coverage_faces) == expected_face_count
        and len(set(coverage_faces)) == len(coverage_faces)
        and set(coverage_faces) == expected_faces
    )
    coverage_by_face = {
        int(row.face_index): row for row in candidate_coverages
    }
    factor_pairs_have_valid_candidates = bool(
        complete_unique_face_coverage
        and all(
            candidate
            < coverage_by_face[face].base_speed_gaps.size
            and candidate
            != int(coverage_by_face[face].controlling_candidate)
            for face, candidate in factor_pairs
        )
    )
    represented_possible_pairs: set[tuple[int, int]] = set()
    for row in candidate_coverages:
        possible = set(
            np.asarray(row.possible_candidates, dtype=int).tolist()
        )
        represented = set(
            np.asarray(row.represented_candidates, dtype=int).tolist()
        )
        for candidate in (possible & represented) - {
            int(row.controlling_candidate)
        }:
            represented_possible_pairs.add(
                (int(row.face_index), int(candidate))
            )
    represented_candidates_have_factors = bool(
        factor_metadata_certified
        and factor_pairs_have_valid_candidates
        and represented_possible_pairs.issubset(factor_pairs)
    )
    required_suppression_remainder = float(
        sum(
            row.suppression_remainder_rate_required
            for row in candidate_coverages
        )
    )
    remainder_rate = (
        0.0
        if nonlinear_remainder_rate is None
        else float(nonlinear_remainder_rate)
    )
    suppression_remainder_included = bool(
        nonlinear_remainder_rate is not None
        and remainder_rate >= required_suppression_remainder
    )
    coverage_certified = bool(
        complete_unique_face_coverage
        and all(row.binding and row.passed for row in candidate_coverages)
        and certified_neighborhood_radius is not None
        and all(
            row.neighborhood_radius >= certified_neighborhood_radius
            for row in candidate_coverages
        )
        and represented_candidates_have_factors
        and suppression_remainder_included
    )
    return certify_rusanov_finite_neighborhood(
        base_generator_per_s=dynamic,
        output_operator=output_operator,
        generator_left_factors=left,
        generator_right_factors=right,
        horizon_seconds=horizon_seconds,
        output_gates=output_gates,
        direct_output_deltas=direct_output_deltas,
        coefficient_bounds=coefficient_bounds,
        state_metric_diagonal=np.asarray(
            operator_arrays["state_weights"],
            dtype=float,
        ),
        initial_state_radius=initial_state_radius,
        certified_neighborhood_radius=certified_neighborhood_radius,
        nonlinear_remainder_rate=remainder_rate,
        nonlinear_output_remainder_bounds=(
            nonlinear_output_remainder_bounds
        ),
        candidate_coverage_certified=coverage_certified,
        nonlinear_remainder_certified=bool(
            nonlinear_remainder_certified
            and nonlinear_remainder_rate is not None
            and nonlinear_output_remainder_bounds is not None
        ),
        maximum_gate_fraction=maximum_gate_fraction,
    )


def _candidate_mask(
    candidates: np.ndarray | tuple[int, ...] | list[int],
    size: int,
) -> np.ndarray:
    values = np.asarray(candidates)
    if values.dtype == np.bool_:
        if values.shape != (size,):
            raise ValueError("a Rusanov candidate mask has the wrong shape")
        return np.asarray(values, dtype=bool).copy()
    if values.ndim != 1:
        raise ValueError("Rusanov candidate indices must form a vector")
    if not np.issubdtype(values.dtype, np.integer):
        if np.any(~np.isfinite(values)) or np.any(values != np.floor(values)):
            raise ValueError("Rusanov candidate indices must be integers")
    indices = np.asarray(values, dtype=int)
    if np.any(indices < 0) or np.any(indices >= size):
        raise ValueError("a Rusanov candidate index is out of range")
    mask = np.zeros(size, dtype=bool)
    mask[indices] = True
    return mask


def _integer_metadata_vector(
    values: np.ndarray | None,
    *,
    expected_size: int,
) -> np.ndarray | None:
    """Return exact integer metadata, or ``None`` when it is uncertified."""

    if values is None:
        return (
            np.empty(0, dtype=int)
            if expected_size == 0
            else None
        )
    raw = np.asarray(values)
    if raw.shape != (expected_size,):
        return None
    try:
        finite = np.asarray(raw, dtype=float)
    except (TypeError, ValueError):
        return None
    if np.any(~np.isfinite(finite)) or np.any(finite != np.floor(finite)):
        return None
    return np.asarray(finite, dtype=int)


def _direct_output_array(
    values: np.ndarray | None,
    *,
    branch_count: int,
    output_count: int,
    state_count: int,
) -> np.ndarray:
    if values is None:
        return np.zeros(
            (branch_count, output_count, state_count),
            dtype=float,
        )
    result = np.asarray(values, dtype=float)
    if (
        result.shape != (branch_count, output_count, state_count)
        or np.any(~np.isfinite(result))
    ):
        raise ValueError("direct Rusanov output deltas have incompatible shapes")
    return result


def _safe_exponential(exponent: float) -> float:
    if exponent > float(np.log(np.finfo(float).max)):
        return float("inf")
    smallest = float(np.nextafter(0.0, 1.0))
    if exponent < float(np.log(smallest)):
        # A positive mathematical upper bound must not silently become zero.
        return smallest
    return float(np.exp(exponent))


def _nonnegative_product(left: float, right: float) -> float:
    if left == 0.0 or right == 0.0:
        return 0.0
    return float(left * right)


def _nonnegative_array_product(
    values: np.ndarray,
    factor: float,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if factor == 0.0:
        return np.zeros_like(array)
    if np.isinf(factor):
        return np.where(array == 0.0, 0.0, float("inf"))
    return array * factor


def _exponential_integral(rate: float, horizon: float) -> float:
    if horizon == 0.0:
        return 0.0
    product = rate * horizon
    if abs(product) <= 1.0e-8:
        return float(
            horizon
            * (
                1.0
                + 0.5 * product
                + product * product / 6.0
            )
        )
    if product > float(np.log(np.finfo(float).max)):
        return float("inf")
    return float(np.expm1(product) / rate)


def _exponential_difference(
    base_rate: float,
    added_rate: float,
    horizon: float,
) -> float:
    if horizon == 0.0 or added_rate == 0.0:
        return 0.0
    upper_exponent = (base_rate + added_rate) * horizon
    if upper_exponent > float(np.log(np.finfo(float).max)):
        return float("inf")
    base_exponent = base_rate * horizon
    added_exponent = added_rate * horizon
    if added_exponent < 50.0:
        if added_exponent == 0.0:
            return float(np.nextafter(0.0, 1.0))
        log_value = base_exponent + float(np.log(np.expm1(added_exponent)))
        smallest = float(np.nextafter(0.0, 1.0))
        if log_value < float(np.log(smallest)):
            return smallest
        return _safe_exponential(log_value)
    return max(
        _safe_exponential(upper_exponent)
        - _safe_exponential(base_exponent),
        0.0,
    )
