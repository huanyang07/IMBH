"""Local seven-field generalized Maxwell--Cattaneo principal symbol.

The conservative rows are the Kerr--Schild rest-mass/stress-energy balances.
The shear row is the projected covariant transient equation, so the complete
system is quasilinear rather than a seven-row conservation law.  Height and
vertical momentum are finite-inertia material currents.

This module builds local matrices only.  It contains no numerical flux,
boundary condition, or trajectory advancement.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np
from scipy.linalg import eig

from imri_qpe.constants import C, DEFAULT_MU_MOL

from .causal_inner_geometry import (
    KerrSchildColumnGeometry,
    ValenciaPerfectFluidPrimitive,
)
from .causal_inner_recovery import FixedHeightGasRadiationColumnEOS
from .causal_inner_stress import (
    _fluid_rest_tetrad,
    calibrate_causal_alpha_shear,
    causal_stress_column_state,
)


GENERALIZED_MAXWELL_CATTANEO_PRIMITIVE_NAMES = (
    "log_surface_density",
    "radial_velocity_over_c",
    "azimuthal_velocity_over_c",
    "log_temperature",
    "specific_shear_stress",
    "log_proper_half_thickness",
    "vertical_velocity_over_c",
)


@dataclass(frozen=True)
class GeneralizedMaxwellCattaneoLocalState:
    """Physical maps and transport coefficients at one seven-field state."""

    chart: np.ndarray
    surface_density: float
    temperature: float
    proper_half_thickness: float
    vertical_velocity_cm_per_s: float
    integrated_pressure: float
    conservative_state6: np.ndarray
    conservative_flux6_over_c: np.ndarray
    four_velocity: np.ndarray
    lower_four_velocity: np.ndarray
    specific_enthalpy_over_c2: float
    sound_speed_over_c: float
    specific_viscosity_seconds: float
    relaxation_time_seconds: float
    entropy_current_coefficient: float
    shear_ratio: float
    equilibrium_specific_stress: float
    proper_vertical_frequency: float
    four_velocity_normalization_relative_defect: float
    shear_tensor_trace_relative_defect: float
    shear_tensor_orthogonality_relative_defect: float
    shear_radial_work_relative_defect: float


@dataclass(frozen=True)
class GeneralizedMaxwellCattaneoPrincipal:
    """Complete local quasilinear radial pencil in primitive coordinates."""

    temporal_matrix: np.ndarray
    radial_matrix: np.ndarray
    primitive_column_scales: np.ndarray
    equation_row_scales: np.ndarray
    scaled_temporal_matrix: np.ndarray
    scaled_radial_matrix: np.ndarray
    eigenvalues_over_c: np.ndarray
    right_eigenvectors_scaled: np.ndarray
    maximum_imaginary_speed_over_c: float
    maximum_eigenpair_relative_defect: float
    eigenvector_condition_number: float
    maximum_biorthogonality_defect: float
    maximum_projector_idempotence_defect: float
    scaled_temporal_condition_number: float
    maximum_light_cone_excess_over_c: float
    local_state: GeneralizedMaxwellCattaneoLocalState


@dataclass(frozen=True)
class SpecializedNonlinearCausalityAudit:
    """Frozen-coefficient full-tensor causality reference screen.

    The Cordeiro et al. inequalities apply to the full shear tensor.  They
    are a deliberately stronger reference screen at the embedded one-shear
    background, not a theorem-equivalent certificate for this projected
    seven-field disk model.  The complete reduced radial pencil is binding.
    """

    shear_eigenvalues_over_enthalpy: tuple[float, float, float]
    shear_signal_ratio: float
    sound_speed_squared_over_c2: float
    E_plus_Lambda_minimum: float
    inequality_margins: tuple[float, ...]

    @property
    def minimum_margin(self) -> float:
        return float(min(self.inequality_margins))


@dataclass(frozen=True)
class GeneralizedMaxwellCattaneoSourceLedger:
    """Local zero-gradient relaxation and vertical-work identities."""

    vertical_total_energy_relative_defect: float
    vertical_reversible_exchange_relative_defect: float
    shear_extended_entropy_production_rate: float
    vertical_entropy_production_rate: float

    @property
    def minimum_entropy_production_rate(self) -> float:
        return float(
            min(
                self.shear_extended_entropy_production_rate,
                self.vertical_entropy_production_rate,
            )
        )


def _require_chart(chart) -> np.ndarray:
    values = np.asarray(chart, dtype=float)
    if values.shape != (7,) or np.any(~np.isfinite(values)):
        raise ValueError("generalized MC chart must be finite and length seven")
    if values[1] ** 2 + values[2] ** 2 >= 1.0:
        raise ValueError("horizontal velocity must be subluminal")
    if abs(float(values[6])) >= 1.0:
        raise ValueError("vertical velocity must be subluminal")
    return values


def _sixth_order_centered_jacobian(
    function,
    chart: np.ndarray,
    steps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(function(chart), dtype=float)
    jacobian = np.empty((center.size, chart.size), dtype=float)
    for column, step in enumerate(steps):
        direction = np.zeros_like(chart)
        direction[column] = step
        jacobian[:, column] = (
            -np.asarray(function(chart - 3.0 * direction))
            + 9.0 * np.asarray(function(chart - 2.0 * direction))
            - 45.0 * np.asarray(function(chart - direction))
            + 45.0 * np.asarray(function(chart + direction))
            - 9.0 * np.asarray(function(chart + 2.0 * direction))
            + np.asarray(function(chart + 3.0 * direction))
        ) / (60.0 * step)
    return center, jacobian


def generalized_maxwell_cattaneo_local_state(
    geometry: KerrSchildColumnGeometry,
    chart,
    *,
    proper_vertical_frequency: float,
    alpha: float,
    stress_factor: float = 1.0,
    mu_mol: float = DEFAULT_MU_MOL,
    gamma_gas: float = 5.0 / 3.0,
) -> GeneralizedMaxwellCattaneoLocalState:
    """Return the exact physical maps used by the quasilinear audit."""

    values = _require_chart(chart)
    omega = float(proper_vertical_frequency)
    alpha = float(alpha)
    if not np.isfinite(omega) or omega <= 0.0:
        raise ValueError("proper vertical frequency must be positive")
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly in (0,1)")
    sigma = float(np.exp(values[0]))
    beta_r = float(values[1])
    beta_phi = float(values[2])
    temperature = float(np.exp(values[3]))
    chi = float(values[4])
    height = float(np.exp(values[5]))
    beta_h = float(values[6])
    eos = FixedHeightGasRadiationColumnEOS(
        proper_half_thickness=height,
        mu_mol=mu_mol,
        gamma_gas=gamma_gas,
    )
    thermodynamics = eos.from_surface_density_temperature(sigma, temperature)
    vertical_energy = 0.5 * C**2 * (
        beta_h**2 + (omega * height / C) ** 2
    )
    internal_energy = thermodynamics.specific_internal_energy + vertical_energy
    primitive = ValenciaPerfectFluidPrimitive(
        surface_density=sigma,
        radial_velocity_over_c=beta_r,
        azimuthal_velocity_over_c=beta_phi,
        specific_internal_energy=float(internal_energy),
        integrated_pressure=thermodynamics.integrated_pressure,
    )
    signal_speed = np.sqrt(alpha) * thermodynamics.sound_speed / C
    closure = calibrate_causal_alpha_shear(
        primitive,
        alpha=alpha,
        stress_factor=stress_factor,
        reference_positive_shear_rate=1.5 * omega,
        viscous_signal_speed_over_c=float(signal_speed),
    )
    stress = causal_stress_column_state(
        geometry,
        primitive,
        specific_stress=chi,
    )
    four_velocity, _rest_radial, _rest_azimuthal = _fluid_rest_tetrad(
        geometry,
        primitive,
    )
    lower_velocity = geometry.spacetime_metric @ four_velocity
    velocity_norm_defect = abs(
        float(four_velocity @ geometry.spacetime_metric @ four_velocity) + 1.0
    )
    rest_mass = float(stress.killing_conserved[0])
    transport = (
        geometry.base.lapse * beta_r / np.sqrt(geometry.base.gamma_rr)
        - geometry.base.radial_shift_over_c
    )
    vertical_velocity = C * beta_h
    state6 = np.concatenate(
        (
            stress.killing_conserved,
            [rest_mass * height, rest_mass * vertical_velocity],
        )
    )
    flux6 = np.concatenate(
        (
            stress.killing_flux_over_c,
            [
                rest_mass * height * transport,
                rest_mass * vertical_velocity * transport,
            ],
        )
    )
    enthalpy = closure.specific_enthalpy_over_c2
    shear_ratio = (
        closure.specific_shear_viscosity_seconds
        / (closure.relaxation_time * enthalpy)
    )
    return GeneralizedMaxwellCattaneoLocalState(
        chart=np.array(values, copy=True),
        surface_density=sigma,
        temperature=temperature,
        proper_half_thickness=height,
        vertical_velocity_cm_per_s=vertical_velocity,
        integrated_pressure=float(thermodynamics.integrated_pressure),
        conservative_state6=np.asarray(state6, dtype=float),
        conservative_flux6_over_c=np.asarray(flux6, dtype=float),
        four_velocity=np.asarray(four_velocity, dtype=float),
        lower_four_velocity=np.asarray(lower_velocity, dtype=float),
        specific_enthalpy_over_c2=float(enthalpy),
        sound_speed_over_c=float(thermodynamics.sound_speed / C),
        specific_viscosity_seconds=float(
            closure.specific_shear_viscosity_seconds
        ),
        relaxation_time_seconds=float(closure.relaxation_time),
        entropy_current_coefficient=float(
            closure.relaxation_time
            / (
                closure.specific_shear_viscosity_seconds
                * temperature
            )
        ),
        shear_ratio=float(shear_ratio),
        equilibrium_specific_stress=float(
            closure.equilibrium_specific_stress
        ),
        proper_vertical_frequency=omega,
        four_velocity_normalization_relative_defect=float(
            velocity_norm_defect
        ),
        shear_tensor_trace_relative_defect=float(
            stress.tensor_trace_relative_defect
        ),
        shear_tensor_orthogonality_relative_defect=float(
            stress.tensor_orthogonality_relative_defect
        ),
        shear_radial_work_relative_defect=float(
            stress.radial_work_relative_defect
        ),
    )


def default_primitive_steps(
    chart,
    *,
    equilibrium_specific_stress: float,
) -> np.ndarray:
    """Return deterministic derivative steps in the seven primitive chart."""

    values = _require_chart(chart)
    return np.asarray(
        [
            2.0e-5,
            2.0e-6,
            2.0e-6,
            2.0e-5,
            2.0e-4
            * max(
                abs(float(values[4])),
                abs(float(equilibrium_specific_stress)),
                1.0e-6,
            ),
            2.0e-5,
            2.0e-6,
        ],
        dtype=float,
    )


def _projected_shear_principal_coefficients(
    geometry: KerrSchildColumnGeometry,
    chart: np.ndarray,
    lower_velocity_jacobian: np.ndarray,
    *,
    derivative_index: int,
) -> np.ndarray:
    """Return the coefficients of ``-2 c sigma_(R)(phi)``.

    ``causal_rest_frame_shear_rate`` is deliberately a stationary radial
    diagnostic.  A time-dependent principal symbol must also retain the
    time-derivative part of the projected shear tensor.  Connection terms
    are lower order, so the principal coefficient follows directly by
    contracting the symmetric partial derivative of ``u_mu`` with the
    comoving radial/azimuthal tetrad.
    """

    if derivative_index not in (0, 1):
        raise ValueError("only time and radial shear derivatives are supported")
    jacobian = np.asarray(lower_velocity_jacobian, dtype=float)
    if jacobian.shape != (3, 7) or np.any(~np.isfinite(jacobian)):
        raise ValueError("lower-velocity Jacobian must be finite and 3 by 7")
    primitive = ValenciaPerfectFluidPrimitive(
        surface_density=float(np.exp(chart[0])),
        radial_velocity_over_c=float(chart[1]),
        azimuthal_velocity_over_c=float(chart[2]),
        specific_internal_energy=0.0,
        integrated_pressure=0.0,
    )
    _four_velocity, rest_radial, rest_azimuthal = _fluid_rest_tetrad(
        geometry,
        primitive,
    )
    coefficients = np.empty(7, dtype=float)
    for column in range(7):
        partial = np.zeros((3, 3), dtype=float)
        partial[derivative_index] = jacobian[:, column]
        symmetric = partial + partial.T
        coefficients[column] = -C * float(
            np.einsum(
                "i,j,ij->",
                rest_radial,
                rest_azimuthal,
                symmetric,
            )
        )
    return coefficients


def generalized_maxwell_cattaneo_principal(
    geometry: KerrSchildColumnGeometry,
    chart,
    *,
    proper_vertical_frequency: float,
    alpha: float,
    stress_factor: float = 1.0,
    derivative_step_factor: float = 1.0,
) -> GeneralizedMaxwellCattaneoPrincipal:
    """Build and diagonalize the complete local radial quasilinear pencil."""

    values = _require_chart(chart)

    def evaluate(candidate):
        return generalized_maxwell_cattaneo_local_state(
            geometry,
            candidate,
            proper_vertical_frequency=proper_vertical_frequency,
            alpha=alpha,
            stress_factor=stress_factor,
        )

    base = evaluate(values)
    steps = derivative_step_factor * default_primitive_steps(
        values,
        equilibrium_specific_stress=base.equilibrium_specific_stress,
    )
    _state, state_jacobian = _sixth_order_centered_jacobian(
        lambda candidate: evaluate(candidate).conservative_state6,
        values,
        steps,
    )
    _flux, flux_jacobian = _sixth_order_centered_jacobian(
        lambda candidate: evaluate(candidate).conservative_flux6_over_c,
        values,
        steps,
    )
    _lower, lower_velocity_jacobian = _sixth_order_centered_jacobian(
        lambda candidate: evaluate(candidate).lower_four_velocity,
        values,
        steps,
    )
    _entropy_coefficient, log_entropy_coefficient_gradient = (
        _sixth_order_centered_jacobian(
            lambda candidate: np.atleast_1d(
                np.log(evaluate(candidate).entropy_current_coefficient)
            ),
            values,
            steps,
        )
    )
    log_entropy_coefficient_gradient = (
        log_entropy_coefficient_gradient.ravel()
    )

    temporal = np.zeros((7, 7), dtype=float)
    radial = np.zeros((7, 7), dtype=float)
    temporal[:4] = state_jacobian[:4]
    radial[:4] = flux_jacobian[:4]
    temporal[5:] = state_jacobian[4:]
    radial[5:] = flux_jacobian[4:]

    temporal_shear_coefficients = _projected_shear_principal_coefficients(
        geometry,
        values,
        lower_velocity_jacobian,
        derivative_index=0,
    )
    radial_shear_coefficients = _projected_shear_principal_coefficients(
        geometry,
        values,
        lower_velocity_jacobian,
        derivative_index=1,
    )
    # The original Israel--Stewart quadratic entropy current selects the
    # specific-stress equation
    #
    #   tau D chi + chi = nu gamma
    #       - 0.5 tau chi D log(tau/(nu T)).
    #
    # This is the entropy-complete form after writing the physical shear
    # tensor as Sigma*c^2*chi*B^{ab}.  The expansion terms produced by
    # differentiating Sigma cancel those in the full entropy-current term;
    # retaining only one side would be the truncated stress-density law.
    temporal[4, 4] = float(base.four_velocity[0])
    temporal[4] += (
        0.5
        * float(values[4])
        * float(base.four_velocity[0])
        * log_entropy_coefficient_gradient
    )
    temporal[4] -= (
        base.specific_viscosity_seconds
        / (base.relaxation_time_seconds * C)
    ) * temporal_shear_coefficients
    radial[4, 4] = float(base.four_velocity[1])
    radial[4] += (
        0.5
        * float(values[4])
        * float(base.four_velocity[1])
        * log_entropy_coefficient_gradient
    )
    radial[4] -= (
        base.specific_viscosity_seconds
        / (base.relaxation_time_seconds * C)
    ) * radial_shear_coefficients

    stress_scale = max(
        abs(float(values[4])),
        abs(base.equilibrium_specific_stress),
        1.0e-6,
    )
    column_scales = np.asarray(
        [1.0, 0.1, 0.1, 1.0, stress_scale, 1.0, 0.03],
        dtype=float,
    )
    row_scales = np.maximum(
        np.max(np.abs(temporal * column_scales[None, :]), axis=1),
        np.max(np.abs(radial * column_scales[None, :]), axis=1),
    )
    row_scales = np.maximum(
        row_scales,
        max(float(np.max(row_scales)), 1.0) * 1.0e-14,
    )
    scaled_temporal = (
        temporal * column_scales[None, :] / row_scales[:, None]
    )
    scaled_radial = radial * column_scales[None, :] / row_scales[:, None]
    eigenvalues, vectors = eig(scaled_radial, scaled_temporal)
    order = np.lexsort((np.imag(eigenvalues), np.real(eigenvalues)))
    eigenvalues = eigenvalues[order]
    vectors = vectors[:, order]
    residual = scaled_radial @ vectors - scaled_temporal @ (
        vectors * eigenvalues[None, :]
    )
    residual_scale = max(
        float(np.max(np.abs(scaled_radial @ vectors))),
        float(np.max(np.abs(scaled_temporal @ vectors))),
        np.finfo(float).tiny,
    )
    maximum_imaginary = float(np.max(np.abs(np.imag(eigenvalues))))
    condition = float(np.linalg.cond(vectors))
    inverse_vectors = np.linalg.inv(vectors)
    biorthogonality = inverse_vectors @ vectors - np.eye(7)
    maximum_projector_defect = 0.0
    for mode in range(7):
        projector = np.outer(vectors[:, mode], inverse_vectors[mode])
        projector_scale = max(
            float(np.linalg.norm(projector, ord=2)),
            np.finfo(float).tiny,
        )
        maximum_projector_defect = max(
            maximum_projector_defect,
            float(
                np.linalg.norm(projector @ projector - projector, ord=2)
                / projector_scale
            ),
        )
    light_min = geometry.base.ingoing_light_speed_over_c
    light_max = geometry.base.outgoing_light_speed_over_c
    light_excess = max(
        float(light_min - np.min(np.real(eigenvalues))),
        float(np.max(np.real(eigenvalues)) - light_max),
        0.0,
    )
    return GeneralizedMaxwellCattaneoPrincipal(
        temporal_matrix=temporal,
        radial_matrix=radial,
        primitive_column_scales=column_scales,
        equation_row_scales=row_scales,
        scaled_temporal_matrix=scaled_temporal,
        scaled_radial_matrix=scaled_radial,
        eigenvalues_over_c=eigenvalues,
        right_eigenvectors_scaled=vectors,
        maximum_imaginary_speed_over_c=maximum_imaginary,
        maximum_eigenpair_relative_defect=float(
            np.max(np.abs(residual)) / residual_scale
        ),
        eigenvector_condition_number=condition,
        maximum_biorthogonality_defect=float(
            np.max(np.abs(biorthogonality))
        ),
        maximum_projector_idempotence_defect=float(
            maximum_projector_defect
        ),
        scaled_temporal_condition_number=float(
            np.linalg.cond(scaled_temporal)
        ),
        maximum_light_cone_excess_over_c=float(light_excess),
        local_state=base,
    )


def audit_specialized_nonlinear_causality(
    local_state: GeneralizedMaxwellCattaneoLocalState,
) -> SpecializedNonlinearCausalityAudit:
    """Evaluate a frozen-coefficient zero-bulk causality reference.

    Energy is normalized by the positive equilibrium enthalpy density.  The
    off-diagonal R-phi stress has normalized principal values ``(-ell,0,ell)``.
    With all second-order DNMR coefficients set to zero, the effective sound
    speeds in Corollary 5 reduce to the equilibrium adiabatic sound speed.
    This check does not replace direct diagonalization of the entropy-current
    complete reduced principal symbol.
    """

    h = float(local_state.specific_enthalpy_over_c2)
    chi = abs(float(local_state.chart[4]))
    ell = chi / h
    lambdas = np.asarray([-ell, 0.0, ell], dtype=float)
    denominators = 1.0 + lambdas
    r = float(local_state.shear_ratio)
    sound_squared = float(local_state.sound_speed_over_c**2)
    product = float(np.prod(denominators))
    reciprocal_sum = float(np.sum(1.0 / denominators))
    margins = [
        r,
        1.0 - r,
        float(np.min(denominators)),
        sound_squared + r / denominators[0],
    ]
    for value, denominator in zip(lambdas, denominators, strict=True):
        margins.append(
            (1.0 - value / 2.0) * denominator * sound_squared
            + 1.5 * r
        )
    margins.extend(
        (
            sound_squared + r * reciprocal_sum,
            3.0 - sound_squared - r * reciprocal_sum,
        )
    )
    for value, denominator in zip(lambdas, denominators, strict=True):
        bracket = (
            (1.0 - value / 3.0)
            * denominator
            * sound_squared
            + (8.0 / 9.0) * r
        ) / product
        margins.append(
            1.0
            - (2.0 / 3.0) * (sound_squared + r * reciprocal_sum)
            + r * bracket
        )
    for first, second, third in permutations(range(3)):
        margins.append(
            (1.0 - r / denominators[first])
            * (1.0 - r / denominators[second])
            * (
                1.0
                - sound_squared
                - r / denominators[third]
            )
        )
    return SpecializedNonlinearCausalityAudit(
        shear_eigenvalues_over_enthalpy=tuple(float(item) for item in lambdas),
        shear_signal_ratio=r,
        sound_speed_squared_over_c2=sound_squared,
        E_plus_Lambda_minimum=float(np.min(denominators)),
        inequality_margins=tuple(float(item) for item in margins),
    )


def audit_generalized_maxwell_cattaneo_source_ledger(
    local_state: GeneralizedMaxwellCattaneoLocalState,
    *,
    alpha: float,
) -> GeneralizedMaxwellCattaneoSourceLedger:
    """Audit the algebraic source energy and entropy signs.

    The vertical pressure work is exchanged exactly between the mechanical
    oscillator and the gas/radiation internal energy.  Vertical damping is
    returned as heat.  For the entropy-current-complete shear equation, the
    remaining extended-entropy production is proportional to ``chi^2/nu``.
    Rates are reported in positive common units up to an irrelevant entropy
    normalization, since only sign and cancellation are used by this local
    structural audit.
    """

    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be positive and finite")
    sigma = float(local_state.surface_density)
    height = float(local_state.proper_half_thickness)
    velocity = float(local_state.vertical_velocity_cm_per_s)
    pressure = float(local_state.integrated_pressure)
    omega = float(local_state.proper_vertical_frequency)
    damping = alpha * omega
    pressure_work = pressure * velocity / height
    gravity_work = sigma * omega**2 * height * velocity
    damping_work = damping * sigma * velocity**2
    acceleration_work = pressure_work - gravity_work - damping_work
    potential_work = gravity_work
    mechanical_work = acceleration_work + potential_work
    thermal_work = -pressure_work + damping_work
    total = mechanical_work + thermal_work
    reversible = pressure_work - gravity_work + gravity_work - pressure_work
    scale = max(
        abs(mechanical_work),
        abs(thermal_work),
        abs(pressure_work),
        abs(gravity_work),
        abs(damping_work),
        np.finfo(float).tiny,
    )
    temperature = float(local_state.temperature)
    shear_entropy = (
        sigma
        * C**2
        * float(local_state.chart[4]) ** 2
        / (
            local_state.specific_viscosity_seconds
            * temperature
        )
    )
    vertical_entropy = damping_work / temperature
    return GeneralizedMaxwellCattaneoSourceLedger(
        vertical_total_energy_relative_defect=float(abs(total) / scale),
        vertical_reversible_exchange_relative_defect=float(
            abs(reversible) / scale
        ),
        shear_extended_entropy_production_rate=float(shear_entropy),
        vertical_entropy_production_rate=float(vertical_entropy),
    )
