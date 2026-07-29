"""Analytic forward-mode tangent for the audit-only radial candidate.

The nonlinear radial fluctuation candidate contains several nested numerical
derivatives and a state-dependent characteristic decomposition.  Directly
finite-differencing that complete residual does not define an additive map at
the precision required by the frozen-generator audits.

This module instead linearizes the declared local physical maps with a small
second-order forward-mode automatic-differentiation kernel.  The second
derivatives are needed because the nonconservative principal matrices already
contain first derivatives of the primitive state maps.  Characteristic
positive/negative subspaces and reconstruction admissibility branches are
frozen at the supplied base state; they are never reclassified while applying
the tangent.

The result is production neutral.  It is a frozen linear audit operator, not a
finite-amplitude Riemann solver and not a change to the production DAE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.linalg import eig
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from imri_qpe.constants import A_RAD, C, SIGMA_SB
from imri_qpe.scales import gas_constant_per_gram

from .causal_inner_characteristic_dissipation import (
    CausalFiveFieldCoordinatePrincipalBasis,
)
from .causal_inner_characteristic_phase import (
    CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES,
)
from .causal_inner_dae import (
    audit_causal_five_field_principal,
)
from .causal_inner_dae_system import (
    CausalFiveFieldDAEContext,
    _gauss_legendre_cell_nodes_and_measures,
    _cell_state,
    causal_five_field_reconstruct_face_charts,
)
from .causal_inner_geometry import kerr_schild_column_geometry


_N_FIELDS = 5
_EXPLICIT_GEOMETRY_LOG_RADIUS_STEP = 2.0e-5


class _Jet2:
    """Scalar value with first and second derivatives in one local chart."""

    __slots__ = ("value", "gradient", "hessian")
    __array_priority__ = 1000

    def __init__(
        self,
        value: float,
        gradient: np.ndarray,
        hessian: np.ndarray,
    ) -> None:
        self.value = float(value)
        self.gradient = np.asarray(gradient, dtype=float)
        self.hessian = np.asarray(hessian, dtype=float)
        dimensions = int(self.gradient.size)
        if (
            self.gradient.shape != (dimensions,)
            or self.hessian.shape != (dimensions, dimensions)
            or np.any(~np.isfinite(self.gradient))
            or np.any(~np.isfinite(self.hessian))
            or not np.isfinite(self.value)
        ):
            raise ValueError("invalid second-order jet")

    @property
    def dimensions(self) -> int:
        return int(self.gradient.size)

    @classmethod
    def constant(cls, value: float, dimensions: int) -> _Jet2:
        return cls(
            value,
            np.zeros(dimensions, dtype=float),
            np.zeros((dimensions, dimensions), dtype=float),
        )

    @classmethod
    def variable(
        cls,
        value: float,
        dimensions: int,
        index: int,
    ) -> _Jet2:
        gradient = np.zeros(dimensions, dtype=float)
        gradient[int(index)] = 1.0
        return cls(
            value,
            gradient,
            np.zeros((dimensions, dimensions), dtype=float),
        )

    def _coerce(self, other: object) -> _Jet2:
        if isinstance(other, _Jet2):
            if other.dimensions != self.dimensions:
                raise ValueError("jet dimensions differ")
            return other
        return _Jet2.constant(float(other), self.dimensions)

    def __add__(self, other: object) -> _Jet2:
        right = self._coerce(other)
        return _Jet2(
            self.value + right.value,
            self.gradient + right.gradient,
            self.hessian + right.hessian,
        )

    def __radd__(self, other: object) -> _Jet2:
        return self.__add__(other)

    def __neg__(self) -> _Jet2:
        return _Jet2(-self.value, -self.gradient, -self.hessian)

    def __sub__(self, other: object) -> _Jet2:
        return self.__add__(-self._coerce(other))

    def __rsub__(self, other: object) -> _Jet2:
        return self._coerce(other).__sub__(self)

    def __mul__(self, other: object) -> _Jet2:
        right = self._coerce(other)
        return _Jet2(
            self.value * right.value,
            self.gradient * right.value
            + right.gradient * self.value,
            self.hessian * right.value
            + right.hessian * self.value
            + np.outer(self.gradient, right.gradient)
            + np.outer(right.gradient, self.gradient),
        )

    def __rmul__(self, other: object) -> _Jet2:
        return self.__mul__(other)

    def _unary(
        self,
        value: float,
        first: float,
        second: float,
    ) -> _Jet2:
        return _Jet2(
            value,
            first * self.gradient,
            first * self.hessian
            + second * np.outer(self.gradient, self.gradient),
        )

    def reciprocal(self) -> _Jet2:
        inverse = 1.0 / self.value
        return self._unary(
            inverse,
            -(inverse**2),
            2.0 * inverse**3,
        )

    def __truediv__(self, other: object) -> _Jet2:
        return self * self._coerce(other).reciprocal()

    def __rtruediv__(self, other: object) -> _Jet2:
        return self._coerce(other) * self.reciprocal()

    def __pow__(self, exponent: float) -> _Jet2:
        power = float(exponent)
        value = self.value**power
        first = power * self.value ** (power - 1.0)
        second = (
            power
            * (power - 1.0)
            * self.value ** (power - 2.0)
        )
        return self._unary(value, first, second)


def _jexp(value: _Jet2) -> _Jet2:
    result = float(np.exp(value.value))
    return value._unary(result, result, result)


def _jlog(value: _Jet2) -> _Jet2:
    return value._unary(
        float(np.log(value.value)),
        1.0 / value.value,
        -1.0 / value.value**2,
    )


def _jsqrt(value: _Jet2) -> _Jet2:
    root = float(np.sqrt(value.value))
    return value._unary(
        root,
        0.5 / root,
        -0.25 / root**3,
    )


def _jconstant(value: float, dimensions: int = _N_FIELDS) -> _Jet2:
    return _Jet2.constant(value, dimensions)


def _jvariables(chart: np.ndarray) -> tuple[_Jet2, ...]:
    values = np.asarray(chart, dtype=float)
    if values.shape != (_N_FIELDS,) or np.any(~np.isfinite(values)):
        raise ValueError("local tangent chart is invalid")
    return tuple(
        _Jet2.variable(value, _N_FIELDS, index)
        for index, value in enumerate(values)
    )


def _jsum(values: Iterable[_Jet2]) -> _Jet2:
    result = _jconstant(0.0)
    for value in values:
        result = result + value
    return result


def _jmatvec(matrix: np.ndarray, vector: Iterable[_Jet2]) -> tuple[_Jet2, ...]:
    coefficients = np.asarray(matrix, dtype=float)
    values = tuple(vector)
    if coefficients.ndim != 2 or coefficients.shape[1] != len(values):
        raise ValueError("jet matrix-vector dimensions differ")
    return tuple(
        _jsum(
            coefficients[row, column] * values[column]
            for column in range(coefficients.shape[1])
        )
        for row in range(coefficients.shape[0])
    )


def _jdot(left: Iterable[_Jet2], right: Iterable[_Jet2]) -> _Jet2:
    left_values = tuple(left)
    right_values = tuple(right)
    if len(left_values) != len(right_values):
        raise ValueError("jet dot-product dimensions differ")
    return _jsum(
        a * b for a, b in zip(left_values, right_values, strict=True)
    )


def _jouter(
    left: Iterable[_Jet2],
    right: Iterable[_Jet2],
) -> tuple[tuple[_Jet2, ...], ...]:
    left_values = tuple(left)
    right_values = tuple(right)
    return tuple(
        tuple(a * b for b in right_values) for a in left_values
    )


def _jmatrix_add(
    left: tuple[tuple[_Jet2, ...], ...],
    right: tuple[tuple[_Jet2, ...], ...],
) -> tuple[tuple[_Jet2, ...], ...]:
    if len(left) != len(right) or any(
        len(a) != len(b) for a, b in zip(left, right, strict=True)
    ):
        raise ValueError("jet matrix dimensions differ")
    return tuple(
        tuple(a + b for a, b in zip(left_row, right_row, strict=True))
        for left_row, right_row in zip(left, right, strict=True)
    )


def _jmatrix_scale(
    scalar: _Jet2,
    matrix: np.ndarray | tuple[tuple[_Jet2, ...], ...],
) -> tuple[tuple[_Jet2, ...], ...]:
    rows = tuple(tuple(row) for row in matrix)
    return tuple(
        tuple(scalar * value for value in row) for row in rows
    )


def _jmatrix_contract(
    left: tuple[tuple[_Jet2, ...], ...],
    right: np.ndarray,
) -> _Jet2:
    coefficients = np.asarray(right, dtype=float)
    if (
        len(left) != coefficients.shape[0]
        or any(len(row) != coefficients.shape[1] for row in left)
    ):
        raise ValueError("jet contraction dimensions differ")
    return _jsum(
        left[row][column] * coefficients[row, column]
        for row in range(coefficients.shape[0])
        for column in range(coefficients.shape[1])
    )


def _lift_first_derivative(value: _Jet2, column: int) -> _Jet2:
    """Treat one first derivative as a first-order jet in the base chart."""

    return _Jet2(
        value.gradient[int(column)],
        value.hessian[int(column)],
        np.zeros_like(value.hessian),
    )


def _extract_vector(
    vector: Iterable[_Jet2],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = tuple(vector)
    return (
        np.asarray([value.value for value in values], dtype=float),
        np.asarray([value.gradient for value in values], dtype=float),
        np.asarray([value.hessian for value in values], dtype=float),
    )


def _extract_matrix(
    matrix: tuple[tuple[_Jet2, ...], ...],
) -> tuple[np.ndarray, np.ndarray]:
    rows = len(matrix)
    columns = len(matrix[0]) if rows else 0
    values = np.empty((rows, columns), dtype=float)
    gradients = np.empty((rows, columns, _N_FIELDS), dtype=float)
    for row in range(rows):
        for column in range(columns):
            values[row, column] = matrix[row][column].value
            gradients[row, column] = matrix[row][column].gradient
    return values, gradients


@dataclass(frozen=True)
class _JetPhysicalState:
    """Local physical maps evaluated with second-order jets."""

    surface_density: _Jet2
    temperature: _Jet2
    height: _Jet2
    density: _Jet2
    pressure: _Jet2
    internal_energy: _Jet2
    enthalpy_over_c2: _Jet2
    sound_squared_over_c2: _Jet2
    beta_r: _Jet2
    beta_phi: _Jet2
    specific_stress: _Jet2
    lorentz: _Jet2
    transport_velocity: _Jet2
    four_velocity: tuple[_Jet2, ...]
    lower_four_velocity: tuple[_Jet2, ...]
    rest_radial: tuple[_Jet2, ...]
    rest_azimuthal: tuple[_Jet2, ...]
    killing_conserved: tuple[_Jet2, ...]
    flux_over_c: tuple[_Jet2, ...]
    perfect_geometry_source: _Jet2
    stress_geometry_source: _Jet2
    equilibrium_specific_stress: _Jet2
    specific_viscosity: _Jet2
    relaxation_time: _Jet2


def _jet_physical_state(
    context: CausalFiveFieldDAEContext,
    radius: float,
    chart: np.ndarray,
) -> _JetPhysicalState:
    """Evaluate the exact smooth local state map in the five-field chart."""

    geometry = kerr_schild_column_geometry(
        float(radius),
        context.grid.gravitational_radius,
    )
    return _jet_physical_state_body(
        context,
        float(radius),
        np.asarray(chart, dtype=float),
        geometry,
    )


def _jet_connection(
    geometry,
) -> np.ndarray:
    """Return the frozen Kerr--Schild connection used by the shear map."""

    metric_derivative = geometry.radial_spacetime_metric_derivative
    inverse_metric = geometry.inverse_spacetime_metric
    radial_index = 1
    connection = np.zeros((3, 3, 3), dtype=float)
    for upper in range(3):
        for first in range(3):
            for second in range(3):
                value = 0.0
                for contracted in range(3):
                    first_term = (
                        metric_derivative[contracted, second]
                        if first == radial_index
                        else 0.0
                    )
                    second_term = (
                        metric_derivative[contracted, first]
                        if second == radial_index
                        else 0.0
                    )
                    third_term = (
                        metric_derivative[first, second]
                        if contracted == radial_index
                        else 0.0
                    )
                    value += inverse_metric[upper, contracted] * (
                        first_term + second_term - third_term
                    )
                connection[upper, first, second] = 0.5 * value
    return connection


def _jet_shear_rate(
    geometry,
    state: _JetPhysicalState,
    radial_lower_velocity_derivative: Iterable[_Jet2],
) -> _Jet2:
    """Evaluate the exact rest-frame shear map on jet-valued inputs."""

    derivative = tuple(radial_lower_velocity_derivative)
    if len(derivative) != 3:
        raise ValueError("jet shear derivative must have length three")
    connection = _jet_connection(geometry)
    covariant = [
        [_jconstant(0.0) for _second in range(3)]
        for _first in range(3)
    ]
    for second in range(3):
        covariant[1][second] = derivative[second]
    for first in range(3):
        for second in range(3):
            covariant[first][second] = (
                covariant[first][second]
                - _jsum(
                    connection[upper, first, second]
                    * state.lower_four_velocity[upper]
                    for upper in range(3)
                )
            )
    symmetric = tuple(
        tuple(
            covariant[first][second]
            + covariant[second][first]
            for second in range(3)
        )
        for first in range(3)
    )
    contraction = _jsum(
        state.rest_radial[first]
        * state.rest_azimuthal[second]
        * symmetric[first][second]
        for first in range(3)
        for second in range(3)
    )
    return -C * contraction


def _jet_comoving_energy_source(
    geometry,
    state: _JetPhysicalState,
    rate: _Jet2,
) -> tuple[_Jet2, ...]:
    """Transform one jet-valued comoving power into Killing sources."""

    four_force = tuple(
        rate * state.four_velocity[index] / C**3
        for index in range(3)
    )
    lower_force = _jmatvec(geometry.spacetime_metric, four_force)
    alpha = geometry.base.lapse
    return (
        _jconstant(0.0),
        alpha * lower_force[1],
        alpha * lower_force[2],
        -alpha * lower_force[0],
    )


def _jet_stress_relaxation_source(
    geometry,
    state: _JetPhysicalState,
    shear_rate: _Jet2,
) -> _Jet2:
    """Evaluate the local Maxwell--Cattaneo source on jets."""

    target = state.specific_viscosity * shear_rate
    rest_mass = state.killing_conserved[0]
    return (
        geometry.base.lapse
        * rest_mass
        / state.lorentz
        * (target - state.specific_stress)
        / (C * state.relaxation_time)
    )


@dataclass(frozen=True)
class CausalFiveFieldAnalyticLocalMaps:
    """Value and exact chart derivatives of the local candidate maps."""

    radius: float
    primitive_chart: np.ndarray
    mapped_conserved: np.ndarray
    mapped_conserved_jacobian: np.ndarray
    vertical_storage_matrix: np.ndarray
    temporal_storage_matrix: np.ndarray
    physical_flux_over_c: np.ndarray
    physical_flux_jacobian: np.ndarray
    shear_principal_source_matrix: np.ndarray
    shear_principal_source_derivative: np.ndarray
    vertical_principal_source_matrix: np.ndarray
    vertical_principal_source_derivative: np.ndarray
    lower_source_values: dict[str, np.ndarray]
    lower_source_jacobians: dict[str, np.ndarray]


def _jet_principal_source_matrices(
    context: CausalFiveFieldDAEContext,
    radius: float,
    state: _JetPhysicalState,
) -> tuple[
    tuple[tuple[_Jet2, ...], ...],
    tuple[tuple[_Jet2, ...], ...],
]:
    """Return shear and height principal matrices as first-order jets."""

    geometry = kerr_schild_column_geometry(
        float(radius),
        context.grid.gravitational_radius,
    )
    zero_derivative = tuple(_jconstant(0.0) for _index in range(3))
    zero_shear = _jet_shear_rate(
        geometry,
        state,
        zero_derivative,
    )
    zero_stress = _jet_stress_relaxation_source(
        geometry,
        state,
        zero_shear,
    )
    shear = [
        [_jconstant(0.0) for _column in range(_N_FIELDS)]
        for _row in range(_N_FIELDS)
    ]
    vertical = [
        [_jconstant(0.0) for _column in range(_N_FIELDS)]
        for _row in range(_N_FIELDS)
    ]
    for column in range(_N_FIELDS):
        lower_derivative = tuple(
            _lift_first_derivative(value, column)
            for value in state.lower_four_velocity
        )
        column_shear = _jet_shear_rate(
            geometry,
            state,
            lower_derivative,
        )
        shear[4][column] = (
            _jet_stress_relaxation_source(
                geometry,
                state,
                column_shear,
            )
            - zero_stress
        )
        log_height_derivative = _lift_first_derivative(
            _jlog(state.height),
            column,
        )
        height_rate = (
            C * state.four_velocity[1] * log_height_derivative
        )
        source = _jet_comoving_energy_source(
            geometry,
            state,
            -state.pressure * height_rate,
        )
        for row in range(4):
            vertical[row][column] = source[row]
    return (
        tuple(tuple(row) for row in shear),
        tuple(tuple(row) for row in vertical),
    )


def _jet_lower_sources(
    context: CausalFiveFieldDAEContext,
    radius: float,
    chart: np.ndarray,
    *,
    explicit_geometry_log_radius_step: float = (
        _EXPLICIT_GEOMETRY_LOG_RADIUS_STEP
    ),
) -> dict[str, tuple[_Jet2, ...]]:
    """Return all local non-principal source components as jets."""

    center_geometry = kerr_schild_column_geometry(
        float(radius),
        context.grid.gravitational_radius,
    )
    center = _jet_physical_state(context, radius, chart)
    step = float(explicit_geometry_log_radius_step)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError(
            "explicit_geometry_log_radius_step must be positive"
        )
    minus_radius = float(radius) * np.exp(-step)
    plus_radius = float(radius) * np.exp(step)
    minus = _jet_physical_state(context, minus_radius, chart)
    plus = _jet_physical_state(context, plus_radius, chart)
    radial_width = plus_radius - minus_radius
    lower_derivative = tuple(
        (
            plus.lower_four_velocity[index]
            - minus.lower_four_velocity[index]
        )
        / radial_width
        for index in range(3)
    )
    shear_rate = _jet_shear_rate(
        center_geometry,
        center,
        lower_derivative,
    )
    height_rate = (
        C
        * center.four_velocity[1]
        * (
            _jlog(plus.height) - _jlog(minus.height)
        )
        / radial_width
    )
    zero = _jconstant(0.0)
    perfect_geometry = (
        zero,
        center.perfect_geometry_source,
        zero,
        zero,
        zero,
    )
    stress_geometry = (
        zero,
        center.stress_geometry_source,
        zero,
        zero,
        zero,
    )
    if context.include_radiative_cooling:
        cooling_rate = (
            16.0
            * SIGMA_SB
            * center.temperature**4
            / (
                3.0
                * context.kappa
                * center.surface_density
            )
        )
        cooling = _jet_comoving_energy_source(
            center_geometry,
            center,
            -cooling_rate,
        ) + (zero,)
    else:
        cooling = (zero, zero, zero, zero, zero)
    vertical = _jet_comoving_energy_source(
        center_geometry,
        center,
        -center.pressure * height_rate,
    ) + (zero,)
    stress = (
        zero,
        zero,
        zero,
        zero,
        _jet_stress_relaxation_source(
            center_geometry,
            center,
            shear_rate,
        ),
    )
    return {
        "perfect_fluid_geometry": perfect_geometry,
        "stress_geometry": stress_geometry,
        "radiative_cooling": cooling,
        "vertical_work": vertical,
        "stress_relaxation": stress,
    }


def causal_five_field_analytic_local_maps(
    context: CausalFiveFieldDAEContext,
    radius: float,
    primitive_chart: np.ndarray,
    *,
    explicit_geometry_log_radius_step: float = (
        _EXPLICIT_GEOMETRY_LOG_RADIUS_STEP
    ),
) -> CausalFiveFieldAnalyticLocalMaps:
    """Return exact forward-AD derivatives of all local radial maps."""

    context = context.validated()
    radius = float(radius)
    chart = np.asarray(primitive_chart, dtype=float)
    if (
        not np.isfinite(radius)
        or radius <= 0.0
        or chart.shape != (_N_FIELDS,)
        or np.any(~np.isfinite(chart))
    ):
        raise ValueError("analytic local-map inputs are invalid")
    state = _jet_physical_state(context, radius, chart)
    conserved, conserved_jacobian, _conserved_hessian = _extract_vector(
        state.killing_conserved
    )
    flux, flux_jacobian, _flux_hessian = _extract_vector(
        state.flux_over_c
    )
    shear_jets, vertical_jets = _jet_principal_source_matrices(
        context,
        radius,
        state,
    )
    shear, shear_derivative = _extract_matrix(shear_jets)
    vertical, vertical_derivative = _extract_matrix(vertical_jets)
    geometry = kerr_schild_column_geometry(
        radius,
        context.grid.gravitational_radius,
    )
    four_velocity = np.asarray(
        [value.value for value in state.four_velocity],
        dtype=float,
    )
    lower_velocity = np.asarray(
        [value.value for value in state.lower_four_velocity],
        dtype=float,
    )
    pressure = float(state.pressure.value)
    log_height_gradient = _jlog(state.height).gradient
    vertical_storage = np.zeros((_N_FIELDS, _N_FIELDS), dtype=float)
    for column in range(_N_FIELDS):
        coefficient = (
            geometry.base.lapse
            * pressure
            * log_height_gradient[column]
            * four_velocity[0]
            / C**2
        )
        vertical_storage[:4, column] = np.asarray(
            [
                0.0,
                coefficient * lower_velocity[1],
                coefficient * lower_velocity[2],
                -coefficient * lower_velocity[0],
            ],
            dtype=float,
        )
    lower_jets = _jet_lower_sources(
        context,
        radius,
        chart,
        explicit_geometry_log_radius_step=(
            explicit_geometry_log_radius_step
        ),
    )
    lower_values: dict[str, np.ndarray] = {}
    lower_jacobians: dict[str, np.ndarray] = {}
    for name, vector in lower_jets.items():
        values, jacobian, _hessian = _extract_vector(vector)
        lower_values[name] = values
        lower_jacobians[name] = jacobian
    return CausalFiveFieldAnalyticLocalMaps(
        radius=radius,
        primitive_chart=np.array(chart, copy=True),
        mapped_conserved=conserved,
        mapped_conserved_jacobian=conserved_jacobian,
        vertical_storage_matrix=vertical_storage,
        temporal_storage_matrix=conserved_jacobian + vertical_storage,
        physical_flux_over_c=flux,
        physical_flux_jacobian=flux_jacobian,
        shear_principal_source_matrix=shear,
        shear_principal_source_derivative=shear_derivative,
        vertical_principal_source_matrix=vertical,
        vertical_principal_source_derivative=vertical_derivative,
        lower_source_values=lower_values,
        lower_source_jacobians=lower_jacobians,
    )


def _jet_physical_state_body(
    context: CausalFiveFieldDAEContext,
    radius: float,
    chart: np.ndarray,
    geometry,
) -> _JetPhysicalState:
    """Implementation body kept separate to make the local map reusable."""

    eos = context.vertical_frequency.eos(float(radius))
    log_sigma, beta_r, beta_phi, log_temperature, chi = _jvariables(chart)
    sigma = _jexp(log_sigma)
    temperature = _jexp(log_temperature)
    gas_constant = gas_constant_per_gram(eos.mu_mol)
    omega = float(eos.proper_vertical_frequency)
    radiation_energy_density = A_RAD * temperature**4
    radiation_term = (
        2.0 * radiation_energy_density / (3.0 * sigma)
    )
    height = (
        radiation_term
        + _jsqrt(
            radiation_term**2
            + 4.0 * omega**2 * gas_constant * temperature
        )
    ) / (2.0 * omega**2)
    density = sigma / (2.0 * height)
    gas_pressure = density * gas_constant * temperature
    radiation_pressure = radiation_energy_density / 3.0
    pressure_density = gas_pressure + radiation_pressure
    integrated_pressure = 2.0 * height * pressure_density
    internal_energy = (
        gas_constant * temperature / (eos.gamma_gas - 1.0)
        + radiation_energy_density / density
    )
    enthalpy = (
        1.0
        + internal_energy / C**2
        + integrated_pressure / (sigma * C**2)
    )

    radiation_height_term = (
        2.0 * radiation_energy_density * height / (3.0 * sigma)
    )
    orbital_height_term = omega**2 * height**2
    derivative_denominator = (
        2.0 * orbital_height_term - radiation_height_term
    )
    height_sigma = -radiation_height_term / derivative_denominator
    height_temperature = (
        4.0 * radiation_height_term + gas_constant * temperature
    ) / derivative_denominator
    density_sigma = 1.0 - height_sigma
    density_temperature = -height_temperature
    gas_internal = gas_constant * temperature / (eos.gamma_gas - 1.0)
    radiation_internal = radiation_energy_density / density
    internal_sigma = -radiation_internal * density_sigma
    internal_temperature = (
        gas_internal
        + radiation_internal * (4.0 - density_temperature)
    )
    gas_integrated_pressure = sigma * gas_constant * temperature
    radiation_integrated_pressure = (
        2.0 * height * radiation_energy_density / 3.0
    )
    pressure_sigma = (
        gas_integrated_pressure
        + radiation_integrated_pressure * height_sigma
    )
    pressure_temperature = (
        gas_integrated_pressure
        + radiation_integrated_pressure
        * (4.0 + height_temperature)
    )
    pressure_over_density = pressure_density / density
    entropy_sigma = (
        internal_sigma - pressure_over_density * density_sigma
    )
    entropy_temperature = (
        internal_temperature
        - pressure_over_density * density_temperature
    )
    adiabatic_temperature = -entropy_sigma / entropy_temperature
    adiabatic_pressure_log = (
        pressure_sigma
        + pressure_temperature * adiabatic_temperature
    )
    adiabatic_pressure_derivative = adiabatic_pressure_log / sigma
    sound_squared_over_c2 = (
        adiabatic_pressure_derivative / (enthalpy * C**2)
    )

    speed_squared = beta_r**2 + beta_phi**2
    lorentz = 1.0 / _jsqrt(1.0 - speed_squared)
    alpha = geometry.base.lapse
    shift = geometry.base.radial_shift_over_c
    gamma_rr = geometry.base.gamma_rr
    coordinate_v_r = beta_r / np.sqrt(gamma_rr)
    covariant_v_r = np.sqrt(gamma_rr) * beta_r
    covariant_v_phi = float(radius) * beta_phi
    transport = alpha * coordinate_v_r - shift
    common = sigma * enthalpy * lorentz**2
    rest_mass = sigma * lorentz
    radial_momentum = common * covariant_v_r
    angular_momentum = common * covariant_v_phi
    thermal_enthalpy = (
        internal_energy / C**2
        + integrated_pressure / (sigma * C**2)
    )
    energy = (
        rest_mass
        * (lorentz - 1.0 + thermal_enthalpy * lorentz)
        - integrated_pressure / C**2
    )
    valencia_conserved = (
        rest_mass,
        radial_momentum,
        angular_momentum,
        energy,
    )
    valencia_flux = (
        rest_mass * transport,
        radial_momentum * transport
        + alpha * integrated_pressure / C**2,
        angular_momentum * transport,
        energy * transport
        + alpha * integrated_pressure / C**2 * coordinate_v_r,
    )
    killing_density = (
        alpha * (energy + rest_mass) - shift * radial_momentum
    )
    killing_flux = (
        alpha * (valencia_flux[3] + valencia_flux[0])
        - shift * valencia_flux[1]
    )
    perfect_conserved = (
        rest_mass,
        radial_momentum,
        angular_momentum,
        killing_density,
    )
    perfect_flux = (
        valencia_flux[0],
        valencia_flux[1],
        valencia_flux[2],
        killing_flux,
    )

    normal = tuple(
        _jconstant(value)
        for value in (
            1.0 / alpha,
            -shift / alpha,
            0.0,
        )
    )
    radial = tuple(
        _jconstant(value)
        for value in (
            0.0,
            1.0 / np.sqrt(gamma_rr),
            0.0,
        )
    )
    azimuthal = tuple(
        _jconstant(value)
        for value in (0.0, 0.0, 1.0 / float(radius))
    )
    gamma_phi = 1.0 / _jsqrt(1.0 - beta_phi**2)
    corotating_time = tuple(
        gamma_phi * (normal[index] + beta_phi * azimuthal[index])
        for index in range(3)
    )
    corotating_phi = tuple(
        gamma_phi * (beta_phi * normal[index] + azimuthal[index])
        for index in range(3)
    )
    corotating_radial_speed = gamma_phi * beta_r
    gamma_radial = 1.0 / _jsqrt(
        1.0 - corotating_radial_speed**2
    )
    four_velocity = tuple(
        gamma_radial
        * (
            corotating_time[index]
            + corotating_radial_speed * radial[index]
        )
        for index in range(3)
    )
    rest_radial = tuple(
        gamma_radial
        * (
            corotating_radial_speed * corotating_time[index]
            + radial[index]
        )
        for index in range(3)
    )
    rest_azimuthal = corotating_phi
    lower_velocity = _jmatvec(
        geometry.spacetime_metric,
        four_velocity,
    )

    stress_mass = sigma * chi
    stress_tensor = _jmatrix_scale(
        stress_mass,
        _jmatrix_add(
            _jouter(rest_radial, rest_azimuthal),
            _jouter(rest_azimuthal, rest_radial),
        ),
    )
    mixed_time = tuple(
        _jsum(
            stress_tensor[0][contracted]
            * geometry.spacetime_metric[contracted, column]
            for contracted in range(3)
        )
        for column in range(3)
    )
    mixed_radial = tuple(
        _jsum(
            stress_tensor[1][contracted]
            * geometry.spacetime_metric[contracted, column]
            for contracted in range(3)
        )
        for column in range(3)
    )
    stress_conserved = (
        _jconstant(0.0),
        alpha * mixed_time[1],
        alpha * mixed_time[2],
        -alpha * mixed_time[0],
    )
    stress_flux = (
        _jconstant(0.0),
        alpha * mixed_radial[1],
        alpha * mixed_radial[2],
        -alpha * mixed_radial[0],
    )
    killing_conserved = tuple(
        perfect_conserved[index] + stress_conserved[index]
        for index in range(4)
    ) + (rest_mass * chi,)
    total_flux = tuple(
        perfect_flux[index] + stress_flux[index]
        for index in range(4)
    ) + (rest_mass * chi * transport,)

    perfect_stress_energy = _jmatrix_add(
        _jmatrix_scale(
            sigma * enthalpy,
            _jouter(four_velocity, four_velocity),
        ),
        _jmatrix_scale(
            integrated_pressure / C**2,
            geometry.inverse_spacetime_metric,
        ),
    )
    perfect_geometry_source = (
        0.5
        * alpha
        * _jmatrix_contract(
            perfect_stress_energy,
            geometry.radial_spacetime_metric_derivative,
        )
    )
    stress_geometry_source = (
        0.5
        * alpha
        * _jmatrix_contract(
            stress_tensor,
            geometry.radial_spacetime_metric_derivative,
        )
    )

    equilibrium = (
        context.stress_factor
        * context.alpha
        * integrated_pressure
        / (sigma * C**2)
    )
    reference_shear = (
        1.5 * context.vertical_frequency.frequency(float(radius))
    )
    specific_viscosity = equilibrium / reference_shear
    signal_squared = context.alpha * sound_squared_over_c2
    relaxation_time = (
        specific_viscosity / (enthalpy * signal_squared)
    )
    return _JetPhysicalState(
        surface_density=sigma,
        temperature=temperature,
        height=height,
        density=density,
        pressure=integrated_pressure,
        internal_energy=internal_energy,
        enthalpy_over_c2=enthalpy,
        sound_squared_over_c2=sound_squared_over_c2,
        beta_r=beta_r,
        beta_phi=beta_phi,
        specific_stress=chi,
        lorentz=lorentz,
        transport_velocity=transport,
        four_velocity=four_velocity,
        lower_four_velocity=lower_velocity,
        rest_radial=rest_radial,
        rest_azimuthal=rest_azimuthal,
        killing_conserved=killing_conserved,
        flux_over_c=total_flux,
        perfect_geometry_source=perfect_geometry_source,
        stress_geometry_source=stress_geometry_source,
        equilibrium_specific_stress=equilibrium,
        specific_viscosity=specific_viscosity,
        relaxation_time=relaxation_time,
    )


_RADIAL_TANGENT_BLOCK_NAMES = (
    "candidate_conservative_transport",
    "candidate_shear_principal",
    "candidate_height_principal",
    "candidate_local_stress_relaxation",
    "candidate_geometry",
    "candidate_cooling",
    "candidate_stream",
    "candidate_lower_height_work",
)


@dataclass(frozen=True)
class CausalFiveFieldRadialAnalyticTangent:
    """One explicitly linear candidate stationary-residual tangent."""

    base_primitives: np.ndarray
    primitive_column_scales: np.ndarray
    conservation_row_scales: np.ndarray
    left_reconstruction_weights: np.ndarray
    right_reconstruction_weights: np.ndarray
    block_scaled_jacobians: dict[str, np.ndarray]
    candidate_stationary_scaled_jacobian: np.ndarray
    path_quadrature_order: int
    characteristic_subspaces_frozen: bool
    principal_matrix_derivatives_included: bool
    explicit_geometry_log_radius_step: float
    characteristic_face_radii: np.ndarray
    characteristic_face_speeds_over_c: np.ndarray
    characteristic_face_analytic_speeds_over_c: np.ndarray
    characteristic_face_descriptor_condition_numbers: np.ndarray
    characteristic_face_eigenpair_defects: np.ndarray
    characteristic_face_biorthogonality_defects: np.ndarray
    characteristic_face_imaginary_parts: np.ndarray
    incoming_inner_characteristics: int
    minimum_absolute_characteristic_speed: float
    minimum_characteristic_spectral_gap: float
    minimum_neighboring_negative_subspace_cosine: float
    minimum_neighboring_positive_subspace_cosine: float
    neighboring_negative_subspace_rank_changes: int
    neighboring_positive_subspace_rank_changes: int
    maximum_characteristic_analytic_speed_defect: float
    maximum_characteristic_eigenpair_defect: float
    maximum_characteristic_biorthogonality_defect: float
    maximum_characteristic_imaginary_part: float
    maximum_characteristic_descriptor_condition_number: float
    maximum_base_reconstruction_relative_defect: float
    maximum_projector_closure_defect: float
    maximum_block_ledger_relative_defect: float

    def apply(
        self,
        scaled_direction: np.ndarray,
        *,
        block: str | None = None,
    ) -> np.ndarray:
        """Apply the stored linear tangent or one declared physical block."""

        direction = np.asarray(scaled_direction, dtype=float).ravel()
        dimensions = int(self.base_primitives.size)
        if direction.shape != (dimensions,) or np.any(~np.isfinite(direction)):
            raise ValueError("analytic tangent direction is invalid")
        matrix = (
            self.candidate_stationary_scaled_jacobian
            if block is None
            else self.block_scaled_jacobians[str(block)]
        )
        return np.asarray(matrix @ direction, dtype=float)


@dataclass(frozen=True)
class CausalFiveFieldFrozenAnalyticTangent:
    """Candidate frozen generator built from one analytic spatial tangent."""

    candidate_spatial_tangent: CausalFiveFieldRadialAnalyticTangent
    production_scaled_generator_per_s: np.ndarray
    candidate_scaled_generator_per_s: np.ndarray
    descriptor_reduced_scaled_matrix: np.ndarray
    production_anchor_storage_derivative: np.ndarray
    production_stationary_scaled_jacobian: np.ndarray
    stationary_delta_scaled_jacobian: np.ndarray
    descriptor_solve_scaled_correction: np.ndarray
    maximum_production_identity_relative_defect: float
    maximum_descriptor_solve_relative_defect: float
    same_temporal_descriptor: bool
    same_base_rate_storage_derivative: bool


def causal_five_field_frozen_analytic_tangent(
    candidate_spatial_tangent: CausalFiveFieldRadialAnalyticTangent,
    production_scaled_generator_per_s: np.ndarray,
    descriptor_reduced_scaled_matrix: np.ndarray,
    production_anchor_storage_derivative: np.ndarray,
) -> CausalFiveFieldFrozenAnalyticTangent:
    """Combine the forward-AD spatial tangent with the certified DAE identity.

    For the unchanged production frozen generator,

    ``M G_prod + J_prod + D_anchor = 0``.

    Recovering ``J_prod`` from this identity avoids subtracting two separately
    finite-differenced nonlinear residuals.  The candidate retains the same
    descriptor and production-anchor storage derivative, exactly as in the
    preceding frozen A/B discrimination.
    """

    dimensions = int(candidate_spatial_tangent.base_primitives.size)
    production = np.asarray(
        production_scaled_generator_per_s,
        dtype=float,
    )
    descriptor = np.asarray(
        descriptor_reduced_scaled_matrix,
        dtype=float,
    )
    anchor = np.asarray(
        production_anchor_storage_derivative,
        dtype=float,
    )
    shape = (dimensions, dimensions)
    if (
        production.shape != shape
        or descriptor.shape != shape
        or anchor.shape != shape
        or np.any(~np.isfinite(production))
        or np.any(~np.isfinite(descriptor))
        or np.any(~np.isfinite(anchor))
    ):
        raise ValueError("frozen analytic tangent matrices are invalid")
    production_stationary = -(descriptor @ production + anchor)
    production_identity = (
        descriptor @ production + production_stationary + anchor
    )
    identity_scale = max(
        float(np.max(np.abs(descriptor @ production))),
        float(np.max(np.abs(production_stationary))),
        float(np.max(np.abs(anchor))),
        np.finfo(float).tiny,
    )
    delta = (
        candidate_spatial_tangent.candidate_stationary_scaled_jacobian
        - production_stationary
    )
    factor = splu(csc_matrix(descriptor), permc_spec="COLAMD")
    correction = np.asarray(factor.solve(delta), dtype=float)
    solve_residual = descriptor @ correction - delta
    solve_scale = max(
        float(np.max(np.abs(delta))),
        np.finfo(float).tiny,
    )
    candidate_generator = production - correction
    return CausalFiveFieldFrozenAnalyticTangent(
        candidate_spatial_tangent=candidate_spatial_tangent,
        production_scaled_generator_per_s=np.array(production, copy=True),
        candidate_scaled_generator_per_s=np.asarray(
            candidate_generator,
            dtype=float,
        ),
        descriptor_reduced_scaled_matrix=np.array(descriptor, copy=True),
        production_anchor_storage_derivative=np.array(anchor, copy=True),
        production_stationary_scaled_jacobian=np.asarray(
            production_stationary,
            dtype=float,
        ),
        stationary_delta_scaled_jacobian=np.asarray(delta, dtype=float),
        descriptor_solve_scaled_correction=correction,
        maximum_production_identity_relative_defect=float(
            np.max(np.abs(production_identity)) / identity_scale
        ),
        maximum_descriptor_solve_relative_defect=float(
            np.max(np.abs(solve_residual)) / solve_scale
        ),
        same_temporal_descriptor=True,
        same_base_rate_storage_derivative=True,
    )


def _interpolation_weights(
    coordinates: np.ndarray,
    evaluation_coordinate: float,
) -> np.ndarray:
    points = np.asarray(coordinates, dtype=float)
    location = float(evaluation_coordinate)
    if (
        points.ndim != 1
        or points.size not in (3, 4)
        or np.any(~np.isfinite(points))
        or not np.isfinite(location)
        or np.unique(points).size != points.size
    ):
        raise ValueError("frozen reconstruction stencil is invalid")
    offsets = points - location
    moment = np.vstack(
        tuple(offsets**power for power in range(points.size))
    )
    right = np.zeros(points.size, dtype=float)
    right[0] = 1.0
    return np.asarray(np.linalg.solve(moment, right), dtype=float)


def _frozen_quadratic_reconstruction_weights(
    context: CausalFiveFieldDAEContext,
    base_primitives: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the exact affine reconstruction on the active base branch."""

    charts = np.asarray(base_primitives, dtype=float)
    n_cells = int(context.grid.centers.size)
    if (
        charts.shape != (n_cells, _N_FIELDS)
        or context.spatial_reconstruction != "quadratic_admissible"
        or context.boundary_trace_reconstruction != "plm_one_sided"
        or context.inner_boundary_trace_override != "inherit"
        or context.inner_flux_trace_override != "inherit"
    ):
        raise ValueError(
            "analytic tangent currently requires the certified "
            "quadratic one-sided reconstruction branch"
        )
    reconstruction = causal_five_field_reconstruct_face_charts(
        context,
        charts,
        purpose="flux",
    )
    factors = np.asarray(
        reconstruction.admissibility_factors,
        dtype=float,
    )
    if not np.array_equal(factors, np.ones_like(factors)):
        raise ValueError(
            "analytic tangent requires inactive admissibility scaling"
        )
    log_centers = np.log(np.asarray(context.grid.centers, dtype=float))
    log_edges = np.log(np.asarray(context.grid.edges, dtype=float))
    left = np.zeros((n_cells + 1, n_cells), dtype=float)
    right = np.zeros_like(left)
    for cell in range(n_cells):
        start = min(max(cell - 1, 0), n_cells - 3)
        indices = np.arange(start, start + 3, dtype=int)
        left_indices = indices
        right_indices = indices
        left_weights = _interpolation_weights(
            log_centers[indices],
            log_edges[cell],
        )
        right_weights = _interpolation_weights(
            log_centers[indices],
            log_edges[cell + 1],
        )
        if cell == 0 and n_cells >= 4:
            left_indices = np.arange(4, dtype=int)
            left_weights = _interpolation_weights(
                log_centers[:4],
                log_edges[0],
            )
        if cell == n_cells - 1 and n_cells >= 4:
            right_indices = np.arange(n_cells - 4, n_cells, dtype=int)
            right_weights = _interpolation_weights(
                log_centers[-4:],
                log_edges[-1],
            )
        if cell > 0:
            right[cell, left_indices] = left_weights
        else:
            left[0, left_indices] = left_weights
            right[0, left_indices] = left_weights
        if cell < n_cells - 1:
            left[cell + 1, right_indices] = right_weights
        else:
            left[-1, right_indices] = right_weights
            right[-1, right_indices] = right_weights
    predicted_left = left @ charts
    predicted_right = right @ charts
    reference_left = np.asarray(
        reconstruction.left_face_charts,
        dtype=float,
    )
    reference_right = np.asarray(
        reconstruction.right_face_charts,
        dtype=float,
    )
    scale = max(
        float(np.max(np.abs(reference_left))),
        float(np.max(np.abs(reference_right))),
        np.finfo(float).tiny,
    )
    defect = float(
        max(
            np.max(np.abs(predicted_left - reference_left)),
            np.max(np.abs(predicted_right - reference_right)),
        )
        / scale
    )
    return left, right, defect


def _signed_projector_matrices(
    basis: CausalFiveFieldCoordinatePrincipalBasis,
    *,
    stationary_speed_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return frozen left/right fluctuation projectors in row coordinates."""

    tolerance = float(stationary_speed_tolerance)
    speeds = np.asarray(basis.numerical_speeds_over_c, dtype=float)
    scales = np.asarray(basis.descriptor_row_scales, dtype=float)
    right = np.asarray(basis.descriptor_right_eigenvectors, dtype=float)
    left = np.asarray(basis.descriptor_left_eigenvectors, dtype=float)
    stationary = np.abs(speeds) <= tolerance
    negative = (
        (speeds < -tolerance).astype(float)
        + 0.5 * stationary.astype(float)
    )
    positive = (
        (speeds > tolerance).astype(float)
        + 0.5 * stationary.astype(float)
    )
    scale = np.diag(scales)
    inverse_scale = np.diag(1.0 / scales)
    left_projector = (
        scale @ right @ np.diag(negative) @ left @ inverse_scale
    )
    right_projector = (
        scale @ right @ np.diag(positive) @ left @ inverse_scale
    )
    # Measure closure in the equilibrated descriptor coordinates.  Conjugating
    # back to dimensionful residual rows can magnify a roundoff residual by
    # many orders without changing the represented projector action.
    defect = float(
        np.max(
            np.abs(
                right
                @ np.diag(negative + positive)
                @ left
                - np.eye(_N_FIELDS)
            )
        )
    )
    return left_projector, right_projector, defect


def _neighboring_subspace_cosine(
    previous: np.ndarray,
    current: np.ndarray,
) -> float:
    """Return the smallest principal-angle cosine for two equal-rank spaces."""

    left = np.asarray(previous, dtype=float)
    right = np.asarray(current, dtype=float)
    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
        raise ValueError("neighboring characteristic subspaces are invalid")
    if left.shape[1] == 0:
        return 1.0
    left_q = np.linalg.qr(left, mode="reduced")[0]
    right_q = np.linalg.qr(right, mode="reduced")[0]
    singular_values = np.linalg.svd(
        left_q.T @ right_q,
        compute_uv=False,
    )
    return float(np.clip(np.min(singular_values), 0.0, 1.0))


def _analytic_flux_jacobian(
    context: CausalFiveFieldDAEContext,
    radius: float,
    chart: np.ndarray,
) -> np.ndarray:
    state = _jet_physical_state(context, radius, chart)
    _value, jacobian, _hessian = _extract_vector(state.flux_over_c)
    return jacobian


def _analytic_principal_maps(
    context: CausalFiveFieldDAEContext,
    radius: float,
    chart: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    state = _jet_physical_state(context, radius, chart)
    shear_jets, height_jets = _jet_principal_source_matrices(
        context,
        radius,
        state,
    )
    shear, shear_derivative = _extract_matrix(shear_jets)
    height, height_derivative = _extract_matrix(height_jets)
    return shear, shear_derivative, height, height_derivative


def _analytic_coordinate_principal_basis(
    context: CausalFiveFieldDAEContext,
    radius: float,
    chart: np.ndarray,
) -> CausalFiveFieldCoordinatePrincipalBasis:
    """Build the frozen characteristic basis from analytic local matrices."""

    local_maps = causal_five_field_analytic_local_maps(
        context,
        radius,
        chart,
    )
    state = _cell_state(context, radius, chart)
    local_audit = audit_causal_five_field_principal(
        state.geometry,
        context.vertical_frequency.eos(radius),
        state.closure,
        surface_density=state.primitive.surface_density,
        radial_velocity_over_c=state.primitive.radial_velocity_over_c,
        azimuthal_velocity_over_c=state.primitive.azimuthal_velocity_over_c,
        temperature=state.thermodynamics.temperature,
    )
    analytic_speeds = np.asarray(
        local_audit.coordinate_speeds_over_c,
        dtype=float,
    )
    stress_scale = max(
        abs(float(chart[4])),
        abs(float(state.closure.equilibrium_specific_stress)),
        1.0e-14,
    )
    column_scales = np.asarray(
        [1.0, 0.1, 0.1, 1.0, stress_scale],
        dtype=float,
    )
    temporal = np.asarray(local_maps.temporal_storage_matrix, dtype=float)
    spatial = (
        local_maps.physical_flux_jacobian
        - local_maps.shear_principal_source_matrix
        - local_maps.vertical_principal_source_matrix
    )
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
    remaining = list(range(_N_FIELDS))
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
    maximum_imaginary = max(
        float(np.max(np.abs(np.imag(values)))),
        float(np.max(np.abs(np.imag(primitive)))),
    )
    if maximum_imaginary > 1.0e-10:
        raise RuntimeError(
            "analytic coordinate eigensystem is not real"
        )
    primitive = np.real(primitive)
    for column in range(_N_FIELDS):
        norm = float(np.linalg.norm(primitive[:, column]))
        if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
            raise RuntimeError("analytic characteristic vector is singular")
        primitive[:, column] /= norm
        pivot = int(np.argmax(np.abs(primitive[:, column])))
        if primitive[pivot, column] < 0.0:
            primitive[:, column] *= -1.0
    descriptor = temporal @ primitive
    scaled_descriptor = descriptor / row_scales[:, None]
    norms = np.linalg.norm(scaled_descriptor, axis=0)
    if np.any(norms <= np.finfo(float).tiny):
        raise RuntimeError("analytic descriptor basis is singular")
    scaled_descriptor = scaled_descriptor / norms[None, :]
    primitive = primitive / norms[None, :]
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
        primitive_column_scales=column_scales,
        descriptor_row_scales=row_scales,
        temporal_storage_matrix=temporal,
        spatial_principal_matrix=spatial,
        analytic_speeds_over_c=analytic_speeds,
        numerical_speeds_over_c=np.asarray(real_values, dtype=float),
        primitive_right_eigenvectors=np.asarray(primitive, dtype=float),
        descriptor_right_eigenvectors=np.asarray(
            scaled_descriptor,
            dtype=float,
        ),
        descriptor_left_eigenvectors=np.asarray(left, dtype=float),
        maximum_analytic_speed_defect=float(
            np.max(np.abs(real_values - analytic_speeds))
        ),
        maximum_eigenpair_defect=float(
            np.max(np.abs(residual)) / residual_scale
        ),
        maximum_biorthogonality_defect=float(
            np.max(
                np.abs(
                    left @ scaled_descriptor - np.eye(_N_FIELDS)
                )
            )
        ),
        maximum_imaginary_part=float(maximum_imaginary),
        descriptor_condition_number=float(
            np.linalg.cond(scaled_descriptor)
        ),
        incoming_inner_characteristics=int(
            np.sum(real_values > 0.0)
        ),
    )


def _path_source_endpoint_tangents(
    context: CausalFiveFieldDAEContext,
    lower_radius: float,
    upper_radius: float,
    left_chart: np.ndarray,
    right_chart: np.ndarray,
    *,
    quadrature_order: int,
    fixed_radius: bool,
    fixed_measure: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Linearize path-integrated shear/height sources at both endpoints."""

    left = np.asarray(left_chart, dtype=float)
    right = np.asarray(right_chart, dtype=float)
    delta = right - left
    nodes, weights = np.polynomial.legendre.leggauss(
        int(quadrature_order)
    )
    shear_left = np.zeros((_N_FIELDS, _N_FIELDS), dtype=float)
    shear_right = np.zeros_like(shear_left)
    height_left = np.zeros_like(shear_left)
    height_right = np.zeros_like(shear_left)
    log_lower = float(np.log(lower_radius))
    log_upper = float(np.log(upper_radius))
    for node, weight in zip(nodes, weights, strict=True):
        fraction = 0.5 * (float(node) + 1.0)
        radius = (
            float(lower_radius)
            if fixed_radius
            else float(
                np.exp(
                    log_lower
                    + fraction * (log_upper - log_lower)
                )
            )
        )
        chart = left + fraction * delta
        (
            shear,
            shear_derivative,
            height,
            height_derivative,
        ) = _analytic_principal_maps(context, radius, chart)
        measure = (
            float(fixed_measure)
            if fixed_radius
            else float(
                kerr_schild_column_geometry(
                    radius,
                    context.grid.gravitational_radius,
                ).face_measure
            )
        )
        quadrature_weight = 0.5 * float(weight) * measure
        shear_coefficient = np.einsum(
            "ijk,j->ik",
            shear_derivative,
            delta,
        )
        height_coefficient = np.einsum(
            "ijk,j->ik",
            height_derivative,
            delta,
        )
        shear_left += quadrature_weight * (
            -shear + (1.0 - fraction) * shear_coefficient
        )
        shear_right += quadrature_weight * (
            shear + fraction * shear_coefficient
        )
        height_left += quadrature_weight * (
            -height + (1.0 - fraction) * height_coefficient
        )
        height_right += quadrature_weight * (
            height + fraction * height_coefficient
        )
    return shear_left, shear_right, height_left, height_right


def _lower_source_endpoint_tangents(
    context: CausalFiveFieldDAEContext,
    cell: int,
    left_chart: np.ndarray,
    right_chart: np.ndarray,
    *,
    explicit_geometry_log_radius_step: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Linearize every lower-source cell integral at its two traces."""

    names = (
        "perfect_fluid_geometry",
        "stress_geometry",
        "radiative_cooling",
        "vertical_work",
        "stress_relaxation",
    )
    left_result = {
        name: np.zeros((_N_FIELDS, _N_FIELDS), dtype=float)
        for name in names
    }
    right_result = {
        name: np.zeros((_N_FIELDS, _N_FIELDS), dtype=float)
        for name in names
    }
    left = np.asarray(left_chart, dtype=float)
    right = np.asarray(right_chart, dtype=float)
    lower_log = float(np.log(context.grid.edges[cell]))
    upper_log = float(np.log(context.grid.edges[cell + 1]))
    radii, weights = _gauss_legendre_cell_nodes_and_measures(context, cell)
    for radius, weight in zip(radii, weights, strict=True):
        fraction = (
            float(np.log(radius)) - lower_log
        ) / (upper_log - lower_log)
        chart = left + fraction * (right - left)
        sources = _jet_lower_sources(
            context,
            float(radius),
            chart,
            explicit_geometry_log_radius_step=(
                explicit_geometry_log_radius_step
            ),
        )
        for name in names:
            _values, jacobian, _hessian = _extract_vector(sources[name])
            left_result[name] += (
                float(weight) * (1.0 - fraction) * jacobian
            )
            right_result[name] += (
                float(weight) * fraction * jacobian
            )
    return left_result, right_result


def _trace_operator(
    left_matrix: np.ndarray,
    left_weights: np.ndarray,
    right_matrix: np.ndarray | None,
    right_weights: np.ndarray | None,
) -> np.ndarray:
    """Lift one two-trace local derivative into all cell primitives."""

    n_cells = int(np.asarray(left_weights).size)
    result = np.zeros((_N_FIELDS, _N_FIELDS * n_cells), dtype=float)
    for cell, weight in enumerate(np.asarray(left_weights, dtype=float)):
        if weight != 0.0:
            columns = slice(_N_FIELDS * cell, _N_FIELDS * (cell + 1))
            result[:, columns] += float(weight) * left_matrix
    if right_matrix is not None and right_weights is not None:
        for cell, weight in enumerate(
            np.asarray(right_weights, dtype=float)
        ):
            if weight != 0.0:
                columns = slice(
                    _N_FIELDS * cell,
                    _N_FIELDS * (cell + 1),
                )
                result[:, columns] += float(weight) * right_matrix
    return result


def causal_five_field_radial_analytic_tangent(
    context: CausalFiveFieldDAEContext,
    base_primitives: np.ndarray,
    *,
    primitive_column_scales: np.ndarray,
    conservation_row_scales: np.ndarray,
    path_quadrature_order: int = 6,
    stationary_speed_tolerance: float = 1.0e-12,
    explicit_geometry_log_radius_step: float = (
        _EXPLICIT_GEOMETRY_LOG_RADIUS_STEP
    ),
) -> CausalFiveFieldRadialAnalyticTangent:
    """Assemble the frozen-subspace forward-AD candidate tangent."""

    context = context.validated()
    base = np.asarray(base_primitives, dtype=float)
    n_cells = int(context.grid.centers.size)
    dimensions = _N_FIELDS * n_cells
    columns = np.asarray(primitive_column_scales, dtype=float).ravel()
    rows = np.asarray(conservation_row_scales, dtype=float).ravel()
    order = int(path_quadrature_order)
    geometry_step = float(explicit_geometry_log_radius_step)
    if (
        base.shape != (n_cells, _N_FIELDS)
        or columns.shape != (dimensions,)
        or rows.shape != (dimensions,)
        or np.any(~np.isfinite(base))
        or np.any(~np.isfinite(columns))
        or np.any(~np.isfinite(rows))
        or np.any(columns <= 0.0)
        or np.any(rows <= 0.0)
        or order < 2
        or not np.isfinite(geometry_step)
        or geometry_step <= 0.0
    ):
        raise ValueError("analytic radial tangent inputs are invalid")
    left_weights, right_weights, reconstruction_defect = (
        _frozen_quadratic_reconstruction_weights(context, base)
    )
    left_faces = left_weights @ base
    right_faces = right_weights @ base
    face_flux = [
        np.zeros((_N_FIELDS, dimensions), dtype=float)
        for _face in range(n_cells + 1)
    ]
    shear_left_face = [
        np.zeros((_N_FIELDS, dimensions), dtype=float)
        for _face in range(n_cells + 1)
    ]
    shear_right_face = [
        np.zeros((_N_FIELDS, dimensions), dtype=float)
        for _face in range(n_cells + 1)
    ]
    height_left_face = [
        np.zeros((_N_FIELDS, dimensions), dtype=float)
        for _face in range(n_cells + 1)
    ]
    height_right_face = [
        np.zeros((_N_FIELDS, dimensions), dtype=float)
        for _face in range(n_cells + 1)
    ]
    projector_defects = []
    characteristic_bases: list[
        CausalFiveFieldCoordinatePrincipalBasis
    ] = []
    negative_subspace_cosines: list[float] = []
    positive_subspace_cosines: list[float] = []
    negative_subspace_rank_changes = 0
    positive_subspace_rank_changes = 0

    def record_basis(
        basis: CausalFiveFieldCoordinatePrincipalBasis,
    ) -> None:
        nonlocal negative_subspace_rank_changes
        nonlocal positive_subspace_rank_changes
        if characteristic_bases:
            previous = characteristic_bases[-1]
            previous_speeds = np.asarray(
                previous.numerical_speeds_over_c,
                dtype=float,
            )
            current_speeds = np.asarray(
                basis.numerical_speeds_over_c,
                dtype=float,
            )
            previous_negative = (
                previous_speeds < -stationary_speed_tolerance
            )
            current_negative = (
                current_speeds < -stationary_speed_tolerance
            )
            previous_positive = (
                previous_speeds > stationary_speed_tolerance
            )
            current_positive = (
                current_speeds > stationary_speed_tolerance
            )
            if np.sum(previous_negative) == np.sum(current_negative):
                negative_subspace_cosines.append(
                    _neighboring_subspace_cosine(
                        previous.primitive_right_eigenvectors[
                            :,
                            previous_negative,
                        ],
                        basis.primitive_right_eigenvectors[
                            :,
                            current_negative,
                        ],
                    )
                )
            else:
                negative_subspace_rank_changes += 1
            if np.sum(previous_positive) == np.sum(current_positive):
                positive_subspace_cosines.append(
                    _neighboring_subspace_cosine(
                        previous.primitive_right_eigenvectors[
                            :,
                            previous_positive,
                        ],
                        basis.primitive_right_eigenvectors[
                            :,
                            current_positive,
                        ],
                    )
                )
            else:
                positive_subspace_rank_changes += 1
        characteristic_bases.append(basis)

    inner_radius = float(context.grid.edges[0])
    inner_measure = float(context.grid.face_measures[0])
    inner_flux_jacobian = (
        inner_measure
        * _analytic_flux_jacobian(
            context,
            inner_radius,
            right_faces[0],
        )
    )
    face_flux[0] = _trace_operator(
        inner_flux_jacobian,
        right_weights[0],
        None,
        None,
    )
    inner_basis = _analytic_coordinate_principal_basis(
        context,
        inner_radius,
        right_faces[0],
    )
    record_basis(inner_basis)
    _inner_negative, _inner_positive, inner_projector_defect = (
        _signed_projector_matrices(
            inner_basis,
            stationary_speed_tolerance=stationary_speed_tolerance,
        )
    )
    projector_defects.append(inner_projector_defect)

    for face in range(1, n_cells + 1):
        radius = float(context.grid.edges[face])
        measure = float(context.grid.face_measures[face])
        left_chart = left_faces[face]
        exterior = face == n_cells
        right_chart = (
            np.asarray(
                context.outer_boundary_frozen_exterior_chart,
                dtype=float,
            )
            if exterior
            else right_faces[face]
        )
        left_flux_jacobian = (
            measure
            * _analytic_flux_jacobian(context, radius, left_chart)
        )
        right_flux_jacobian = (
            measure
            * _analytic_flux_jacobian(context, radius, right_chart)
        )
        midpoint = 0.5 * (left_chart + right_chart)
        basis = _analytic_coordinate_principal_basis(
            context,
            radius,
            midpoint,
        )
        negative, positive, projector_defect = (
            _signed_projector_matrices(
                basis,
                stationary_speed_tolerance=stationary_speed_tolerance,
            )
        )
        projector_defects.append(projector_defect)
        record_basis(basis)
        candidate_left = (
            np.eye(_N_FIELDS) - negative
        ) @ left_flux_jacobian
        candidate_right = negative @ right_flux_jacobian
        face_flux[face] = _trace_operator(
            candidate_left,
            left_weights[face],
            None if exterior else candidate_right,
            None if exterior else right_weights[face],
        )
        (
            shear_path_left,
            shear_path_right,
            height_path_left,
            height_path_right,
        ) = _path_source_endpoint_tangents(
            context,
            radius,
            radius,
            left_chart,
            right_chart,
            quadrature_order=order,
            fixed_radius=True,
            fixed_measure=measure,
        )
        shear_left_face[face] = _trace_operator(
            -negative @ shear_path_left,
            left_weights[face],
            None if exterior else -negative @ shear_path_right,
            None if exterior else right_weights[face],
        )
        shear_right_face[face] = _trace_operator(
            -positive @ shear_path_left,
            left_weights[face],
            None if exterior else -positive @ shear_path_right,
            None if exterior else right_weights[face],
        )
        height_left_face[face] = _trace_operator(
            -negative @ height_path_left,
            left_weights[face],
            None if exterior else -negative @ height_path_right,
            None if exterior else right_weights[face],
        )
        height_right_face[face] = _trace_operator(
            -positive @ height_path_left,
            left_weights[face],
            None if exterior else -positive @ height_path_right,
            None if exterior else right_weights[face],
        )

    physical_blocks = {
        name: np.zeros((dimensions, dimensions), dtype=float)
        for name in _RADIAL_TANGENT_BLOCK_NAMES
    }
    for cell in range(n_cells):
        row_slice = slice(
            _N_FIELDS * cell,
            _N_FIELDS * (cell + 1),
        )
        physical_blocks["candidate_conservative_transport"][
            row_slice
        ] = face_flux[cell + 1] - face_flux[cell]
        (
            shear_path_left,
            shear_path_right,
            height_path_left,
            height_path_right,
        ) = _path_source_endpoint_tangents(
            context,
            float(context.grid.edges[cell]),
            float(context.grid.edges[cell + 1]),
            right_faces[cell],
            left_faces[cell + 1],
            quadrature_order=order,
            fixed_radius=False,
        )
        shear_within = _trace_operator(
            shear_path_left,
            right_weights[cell],
            shear_path_right,
            left_weights[cell + 1],
        )
        height_within = _trace_operator(
            height_path_left,
            right_weights[cell],
            height_path_right,
            left_weights[cell + 1],
        )
        physical_blocks["candidate_shear_principal"][row_slice] = (
            -shear_within
            + shear_right_face[cell]
            + shear_left_face[cell + 1]
        )
        physical_blocks["candidate_height_principal"][row_slice] = (
            -height_within
            + height_right_face[cell]
            + height_left_face[cell + 1]
        )
        lower_left, lower_right = _lower_source_endpoint_tangents(
            context,
            cell,
            right_faces[cell],
            left_faces[cell + 1],
            explicit_geometry_log_radius_step=geometry_step,
        )

        def lower_operator(name: str) -> np.ndarray:
            return _trace_operator(
                lower_left[name],
                right_weights[cell],
                lower_right[name],
                left_weights[cell + 1],
            )

        physical_blocks["candidate_local_stress_relaxation"][
            row_slice
        ] = -lower_operator("stress_relaxation")
        physical_blocks["candidate_geometry"][row_slice] = -(
            lower_operator("perfect_fluid_geometry")
            + lower_operator("stress_geometry")
        )
        physical_blocks["candidate_cooling"][row_slice] = (
            -lower_operator("radiative_cooling")
        )
        physical_blocks["candidate_lower_height_work"][row_slice] = (
            -lower_operator("vertical_work")
        )

    scaled_blocks = {
        name: (
            matrix
            * columns[None, :]
            / rows[:, None]
        )
        for name, matrix in physical_blocks.items()
    }
    candidate = sum(
        scaled_blocks.values(),
        start=np.zeros((dimensions, dimensions), dtype=float),
    )
    reconstructed = sum(
        (
            scaled_blocks[name]
            for name in _RADIAL_TANGENT_BLOCK_NAMES
        ),
        start=np.zeros_like(candidate),
    )
    scale = max(
        float(np.max(np.abs(candidate))),
        np.finfo(float).tiny,
    )
    ledger_defect = float(
        np.max(np.abs(candidate - reconstructed)) / scale
    )
    characteristic_speeds = np.asarray(
        [
            basis.numerical_speeds_over_c
            for basis in characteristic_bases
        ],
        dtype=float,
    )
    characteristic_analytic_speeds = np.asarray(
        [
            basis.analytic_speeds_over_c
            for basis in characteristic_bases
        ],
        dtype=float,
    )
    sorted_speeds = np.sort(characteristic_speeds, axis=1)
    spectral_gaps = np.diff(sorted_speeds, axis=1)
    descriptor_conditions = np.asarray(
        [
            basis.descriptor_condition_number
            for basis in characteristic_bases
        ],
        dtype=float,
    )
    eigenpair_defects = np.asarray(
        [
            basis.maximum_eigenpair_defect
            for basis in characteristic_bases
        ],
        dtype=float,
    )
    biorthogonality_defects = np.asarray(
        [
            basis.maximum_biorthogonality_defect
            for basis in characteristic_bases
        ],
        dtype=float,
    )
    imaginary_parts = np.asarray(
        [
            basis.maximum_imaginary_part
            for basis in characteristic_bases
        ],
        dtype=float,
    )
    return CausalFiveFieldRadialAnalyticTangent(
        base_primitives=np.array(base, copy=True),
        primitive_column_scales=np.array(columns, copy=True),
        conservation_row_scales=np.array(rows, copy=True),
        left_reconstruction_weights=left_weights,
        right_reconstruction_weights=right_weights,
        block_scaled_jacobians=scaled_blocks,
        candidate_stationary_scaled_jacobian=candidate,
        path_quadrature_order=order,
        characteristic_subspaces_frozen=True,
        principal_matrix_derivatives_included=True,
        explicit_geometry_log_radius_step=geometry_step,
        characteristic_face_radii=np.asarray(
            context.grid.edges,
            dtype=float,
        ),
        characteristic_face_speeds_over_c=characteristic_speeds,
        characteristic_face_analytic_speeds_over_c=(
            characteristic_analytic_speeds
        ),
        characteristic_face_descriptor_condition_numbers=(
            descriptor_conditions
        ),
        characteristic_face_eigenpair_defects=eigenpair_defects,
        characteristic_face_biorthogonality_defects=(
            biorthogonality_defects
        ),
        characteristic_face_imaginary_parts=imaginary_parts,
        incoming_inner_characteristics=(
            inner_basis.incoming_inner_characteristics
        ),
        minimum_absolute_characteristic_speed=float(
            np.min(np.abs(characteristic_speeds))
        ),
        minimum_characteristic_spectral_gap=float(
            np.min(spectral_gaps)
        ),
        minimum_neighboring_negative_subspace_cosine=(
            min(negative_subspace_cosines)
            if negative_subspace_cosines
            else 1.0
        ),
        minimum_neighboring_positive_subspace_cosine=(
            min(positive_subspace_cosines)
            if positive_subspace_cosines
            else 1.0
        ),
        neighboring_negative_subspace_rank_changes=(
            negative_subspace_rank_changes
        ),
        neighboring_positive_subspace_rank_changes=(
            positive_subspace_rank_changes
        ),
        maximum_characteristic_analytic_speed_defect=float(
            max(
                basis.maximum_analytic_speed_defect
                for basis in characteristic_bases
            )
        ),
        maximum_characteristic_eigenpair_defect=float(
            np.max(eigenpair_defects)
        ),
        maximum_characteristic_biorthogonality_defect=float(
            np.max(biorthogonality_defects)
        ),
        maximum_characteristic_imaginary_part=float(
            np.max(imaginary_parts)
        ),
        maximum_characteristic_descriptor_condition_number=float(
            np.max(descriptor_conditions)
        ),
        maximum_base_reconstruction_relative_defect=(
            reconstruction_defect
        ),
        maximum_projector_closure_defect=(
            max(projector_defects) if projector_defects else 0.0
        ),
        maximum_block_ledger_relative_defect=ledger_defect,
    )
