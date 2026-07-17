"""Stream and Hill/Roche adapters for the causal Kerr-Schild column."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from imri_qpe.constants import C, G

from .causal_inner_geometry import (
    KerrSchildColumnGeometry,
    KerrSchildColumnGrid,
    ValenciaPerfectFluidPrimitive,
)
from .causal_inner_recovery import GasRadiationColumnThermodynamics
from .causal_inner_thermal import (
    QuasiHydrostaticGasRadiationColumnEOS,
    kerr_schild_column_four_velocity,
)
from .causal_inner_valencia import (
    valencia_flux_primary_count,
    valencia_radial_characteristic_speeds_over_c,
)
from .hill_roche_nozzle import (
    HillRocheNozzleGate,
    HillRocheNozzleReservoir,
    OverflowBoundaryProvider,
)
from .transonic_local import stream_annulus_shape_and_derivative


KERR_SCHILD_HILL_ENERGY_ZERO = (
    "kerr-schild-killing-matched-at-reservoir-edge"
)


@runtime_checkable
class ProperVerticalFrequencyProvider(Protocol):
    """Explicit radial provider for the quasi-hydrostatic column frequency."""

    gravitational_radius: float

    def frequency(self, radius: float) -> float:
        """Return positive proper vertical frequency in inverse seconds."""

    def logarithmic_radial_derivative(self, radius: float) -> float:
        """Return ``dln(Omega_perp)/dlnR``."""


@dataclass(frozen=True)
class SchwarzschildCurvatureVerticalFrequency:
    """Smooth Schwarzschild curvature-scale vertical-frequency provider.

    This is the positive ``sqrt(GM/R^3)`` tidal scale written as
    ``c sqrt(rg/R^3)``. It is horizon penetrating and has the correct
    weak-field orbital limit, but it is not a resolved vertical dynamics law.
    """

    gravitational_radius: float

    def __post_init__(self) -> None:
        value = float(self.gravitational_radius)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("gravitational radius must be positive and finite")

    def frequency(self, radius: float) -> float:
        radius = float(radius)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius must be positive and finite")
        return float(
            C * np.sqrt(self.gravitational_radius / radius**3)
        )

    def logarithmic_radial_derivative(self, radius: float) -> float:
        self.frequency(radius)
        return -1.5

    def eos(
        self,
        radius: float,
    ) -> QuasiHydrostaticGasRadiationColumnEOS:
        """Return the responsive column EOS at one radius."""

        return QuasiHydrostaticGasRadiationColumnEOS(
            proper_vertical_frequency=self.frequency(radius)
        )


@dataclass(frozen=True)
class KerrSchildSpecificInjectionMoments:
    """Specific covariant moments carried by one injected rest mass."""

    transport_radial_velocity: float
    kinematic_specific_angular_momentum: float
    radial_momentum_over_c: float
    angular_momentum_over_c: float
    killing_energy_over_c2: float

    def validated(self) -> KerrSchildSpecificInjectionMoments:
        values = (
            self.transport_radial_velocity,
            self.kinematic_specific_angular_momentum,
            self.radial_momentum_over_c,
            self.angular_momentum_over_c,
            self.killing_energy_over_c2,
        )
        if any(not np.isfinite(value) for value in values):
            raise ValueError("Kerr-Schild injection moments must be finite")
        if self.killing_energy_over_c2 <= 0.0:
            raise ValueError("Killing energy must include positive rest energy")
        return self

    @property
    def specific_radial_momentum(self) -> float:
        return float(C * self.radial_momentum_over_c)

    @property
    def specific_angular_momentum(self) -> float:
        return float(C * self.angular_momentum_over_c)

    @property
    def specific_killing_energy(self) -> float:
        return float(C**2 * self.killing_energy_over_c2)


@dataclass(frozen=True)
class KerrSchildStreamInjection:
    """One immutable physical stream rate and its four-state moments."""

    rest_mass_rate: float
    moments: KerrSchildSpecificInjectionMoments

    def validated(self) -> KerrSchildStreamInjection:
        moments = self.moments.validated()
        values = (
            self.rest_mass_rate,
            moments.transport_radial_velocity,
            moments.kinematic_specific_angular_momentum,
            moments.radial_momentum_over_c,
            moments.angular_momentum_over_c,
            moments.killing_energy_over_c2,
        )
        if any(not np.isfinite(value) for value in values):
            raise ValueError("stream rate and moments must be finite")
        if self.rest_mass_rate < 0.0:
            raise ValueError("stream rest-mass rate cannot be negative")
        if self.moments.killing_energy_over_c2 <= 0.0:
            raise ValueError("stream Killing energy must include positive rest energy")
        return self


@dataclass(frozen=True)
class KerrSchildCellSourceRates:
    """Exact cell-integrated source rates in the Killing-equivalent chart."""

    rest_mass: np.ndarray
    radial_momentum_over_c: np.ndarray
    angular_momentum_over_c: np.ndarray
    killing_energy_over_c2: np.ndarray

    def validated_for(
        self,
        n_cells: int,
    ) -> KerrSchildCellSourceRates:
        if int(n_cells) != n_cells or n_cells < 1:
            raise ValueError("source validation requires a positive cell count")
        arrays = (
            self.rest_mass,
            self.radial_momentum_over_c,
            self.angular_momentum_over_c,
            self.killing_energy_over_c2,
        )
        if any(np.asarray(array).shape != (int(n_cells),) for array in arrays):
            raise ValueError("Kerr-Schild source arrays have the wrong shape")
        if any(
            not np.all(np.isfinite(np.asarray(array, dtype=float)))
            for array in arrays
        ):
            raise ValueError("Kerr-Schild source arrays must be finite")
        if np.any(np.asarray(self.rest_mass) < 0.0):
            raise ValueError("cell-integrated source mass cannot be negative")
        return self

    @property
    def matrix(self) -> np.ndarray:
        """Return physical mass-equivalent rates per coordinate second."""

        return np.column_stack(
            (
                self.rest_mass,
                self.radial_momentum_over_c,
                self.angular_momentum_over_c,
                self.killing_energy_over_c2,
            )
        )

    @property
    def weighted_killing_source_per_ct(self) -> np.ndarray:
        """Return the source consumed by the ``x^0=ct`` finite-volume DAE."""

        return self.matrix / C


@dataclass(frozen=True)
class KerrSchildRocheEdgeState:
    """One responsive column and its relativistic edge invariants."""

    geometry: KerrSchildColumnGeometry
    thermodynamics: GasRadiationColumnThermodynamics
    temperature: float
    moments: KerrSchildSpecificInjectionMoments


@dataclass(frozen=True)
class KerrSchildRocheBoundaryAudit:
    """Closed/choked outer flux and its Killing/Jacobi ledgers."""

    edge_state: KerrSchildRocheEdgeState
    gate: HillRocheNozzleGate
    weighted_killing_flux_over_c: np.ndarray
    rest_mass_rate: float
    radial_momentum_rate: float
    angular_momentum_rate: float
    killing_energy_rate: float
    pressure_traction: float
    incoming_outer_characteristics: int
    no_inward_mass: bool
    zero_outer_stress: bool
    angular_momentum_relative_defect: float
    killing_energy_relative_defect: float
    binary_pattern_power_relative_defect: float


@dataclass(frozen=True)
class KerrSchildMigrationRankAudit:
    """Exact count and outer face-row rank after source/boundary migration."""

    n_cells: int
    total_unknowns: int
    total_rows: int
    source_unknowns: int
    source_rows: int
    boundary_face_rows: int
    boundary_face_jacobian_rank: int
    physical_outer_boundary_conditions: int
    square: bool


def kerr_schild_specific_injection_moments(
    geometry: KerrSchildColumnGeometry,
    primitive: ValenciaPerfectFluidPrimitive,
) -> KerrSchildSpecificInjectionMoments:
    """Return covariant stream moments from one local fluid four-state."""

    sigma = float(primitive.surface_density)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("stream surface density must be positive and finite")
    four_velocity = kerr_schild_column_four_velocity(
        geometry,
        primitive,
    )
    lower_velocity = geometry.spacetime_metric @ four_velocity
    enthalpy_over_c2 = (
        1.0
        + float(primitive.specific_internal_energy) / C**2
        + float(primitive.integrated_pressure) / (sigma * C**2)
    )
    transport_velocity = C * (
        geometry.base.lapse
        * primitive.radial_velocity_over_c
        / np.sqrt(geometry.base.gamma_rr)
        - geometry.base.radial_shift_over_c
    )
    moments = KerrSchildSpecificInjectionMoments(
        transport_radial_velocity=float(transport_velocity),
        kinematic_specific_angular_momentum=float(
            C * lower_velocity[2]
        ),
        radial_momentum_over_c=float(
            enthalpy_over_c2 * lower_velocity[1]
        ),
        angular_momentum_over_c=float(
            enthalpy_over_c2 * lower_velocity[2]
        ),
        killing_energy_over_c2=float(
            -enthalpy_over_c2 * lower_velocity[0]
        ),
    )
    values = (
        moments.transport_radial_velocity,
        moments.kinematic_specific_angular_momentum,
        moments.radial_momentum_over_c,
        moments.angular_momentum_over_c,
        moments.killing_energy_over_c2,
    )
    if any(not np.isfinite(value) for value in values):
        raise ValueError("derived stream moments are not finite")
    if moments.killing_energy_over_c2 <= 0.0:
        raise ValueError("derived stream Killing energy is not future directed")
    return moments


def kerr_schild_stream_injection(
    geometry: KerrSchildColumnGeometry,
    primitive: ValenciaPerfectFluidPrimitive,
    *,
    rest_mass_rate: float,
) -> KerrSchildStreamInjection:
    """Bind one absolute rest-mass rate to one injected four-state."""

    return KerrSchildStreamInjection(
        rest_mass_rate=float(rest_mass_rate),
        moments=kerr_schild_specific_injection_moments(
            geometry,
            primitive,
        ),
    ).validated()


def exact_kerr_schild_compact_stream_sources(
    grid: KerrSchildColumnGrid,
    injection: KerrSchildStreamInjection,
    *,
    center: float,
    log_width: float,
    shape: str = "compact_c2",
) -> KerrSchildCellSourceRates:
    """Return exact compact cell moments without source quadrature."""

    injection = injection.validated()
    center = float(center)
    width = float(log_width)
    if (
        not np.isfinite(center)
        or not np.isfinite(width)
        or center <= 0.0
        or width <= 0.0
    ):
        raise ValueError("stream center and log width must be positive")
    shape_name = str(shape).strip().lower()
    if shape_name not in {"compact_c2", "c2", "compact_c4", "c4"}:
        raise ValueError("stream shape must be compact_c2 or compact_c4")
    outer_radius = float(grid.edges[-1])
    cumulative = np.asarray(
        [
            stream_annulus_shape_and_derivative(
                float(log_radius),
                center / outer_radius,
                width,
                outer_radius,
                shape=shape_name,
            )[0]
            for log_radius in np.log(grid.edges)
        ],
        dtype=float,
    )
    tolerance = 2.0e-13
    if cumulative[0] > tolerance or cumulative[-1] < 1.0 - tolerance:
        raise ValueError("compact stream support must lie inside the grid")
    weights = np.diff(cumulative)
    if np.any(weights < -tolerance):
        raise ValueError("compact stream cumulative profile is not monotone")
    weights = np.maximum(weights, 0.0)
    if not np.isclose(np.sum(weights), 1.0, rtol=0.0, atol=tolerance):
        raise ValueError("compact stream weights do not sum to unity")
    mass = injection.rest_mass_rate * weights
    moments = injection.moments
    return KerrSchildCellSourceRates(
        rest_mass=mass,
        radial_momentum_over_c=mass * moments.radial_momentum_over_c,
        angular_momentum_over_c=mass * moments.angular_momentum_over_c,
        killing_energy_over_c2=mass * moments.killing_energy_over_c2,
    ).validated_for(grid.centers.size)


def kerr_schild_hill_roche_reservoir(
    geometry: KerrSchildColumnGeometry,
    eos: QuasiHydrostaticGasRadiationColumnEOS,
    primitive: ValenciaPerfectFluidPrimitive,
    *,
    temperature: float,
) -> tuple[KerrSchildRocheEdgeState, HillRocheNozzleReservoir]:
    """Map one responsive Kerr-Schild column into a Hill reservoir state."""

    thermodynamics = eos.from_surface_density_temperature(
        primitive.surface_density,
        temperature,
    )
    pressure_scale = max(
        abs(thermodynamics.integrated_pressure),
        1.0,
    )
    energy_scale = max(
        abs(thermodynamics.specific_internal_energy),
        1.0,
    )
    if (
        abs(
            primitive.integrated_pressure
            - thermodynamics.integrated_pressure
        )
        > 2.0e-12 * pressure_scale
        or abs(
            primitive.specific_internal_energy
            - thermodynamics.specific_internal_energy
        )
        > 2.0e-12 * energy_scale
    ):
        raise ValueError("Roche primitive is inconsistent with the column EOS")

    moments = kerr_schild_specific_injection_moments(
        geometry,
        primitive,
    )
    pressure = (
        thermodynamics.integrated_pressure
        / (2.0 * thermodynamics.proper_half_thickness)
    )
    reservoir = HillRocheNozzleReservoir(
        radius=geometry.radius,
        density=thermodynamics.density,
        pressure=pressure,
        radial_velocity=moments.transport_radial_velocity,
        specific_angular_momentum=(
            moments.kinematic_specific_angular_momentum
        ),
        temperature=float(temperature),
        specific_inertial_bernoulli=(
            moments.specific_killing_energy - C**2
        ),
        specific_flux_angular_momentum=(
            moments.specific_angular_momentum
        ),
    ).validated()
    return (
        KerrSchildRocheEdgeState(
            geometry=geometry,
            thermodynamics=thermodynamics,
            temperature=float(temperature),
            moments=moments,
        ),
        reservoir,
    )


def apply_kerr_schild_hill_roche_boundary(
    geometry: KerrSchildColumnGeometry,
    eos: QuasiHydrostaticGasRadiationColumnEOS,
    primitive: ValenciaPerfectFluidPrimitive,
    *,
    temperature: float,
    provider: OverflowBoundaryProvider,
    outer_specific_stress: float = 0.0,
    ledger_tolerance: float = 2.0e-9,
) -> KerrSchildRocheBoundaryAudit:
    """Map one physical closed/choked Roche state into Killing face fluxes."""

    if not isinstance(provider, OverflowBoundaryProvider):
        raise TypeError("provider does not implement OverflowBoundaryProvider")
    if provider.geometry.energy_zero != KERR_SCHILD_HILL_ENERGY_ZERO:
        raise ValueError(
            "Roche provider must declare the Kerr-Schild Killing-energy zero"
        )
    if not np.isfinite(ledger_tolerance) or ledger_tolerance <= 0.0:
        raise ValueError("ledger tolerance must be positive and finite")
    stress = float(outer_specific_stress)
    if not np.isfinite(stress) or abs(stress) > 1.0e-15:
        raise ValueError("physical Roche edge requires zero outer shear stress")
    expected_gravitational_radius = (
        G * provider.geometry.secondary_mass / C**2
    )
    if not np.isclose(
        geometry.gravitational_radius,
        expected_gravitational_radius,
        rtol=2.0e-14,
        atol=0.0,
    ):
        raise ValueError("Roche provider and Kerr-Schild metric masses differ")
    edge_state, reservoir = kerr_schild_hill_roche_reservoir(
        geometry,
        eos,
        primitive,
        temperature=temperature,
    )
    thermodynamics = edge_state.thermodynamics
    moments = edge_state.moments
    gate = provider.evaluate(reservoir)
    pressure_traction = float(
        geometry.face_measure
        * geometry.base.lapse
        * thermodynamics.integrated_pressure
    )
    if gate.solution is None:
        mass_rate = 0.0
        nozzle_force = 0.0
        angular_rate = 0.0
        energy_rate = 0.0
        angular_defect = 0.0
        energy_defect = 0.0
        pattern_defect = 0.0
    else:
        solution = gate.solution
        mass_rate = float(solution.saddle_flux.mass)
        nozzle_force = float(solution.saddle_flux.radial_momentum)
        angular_rate = float(solution.edge_angular_momentum_flux)
        energy_rate = float(
            solution.edge_total_energy_flux + mass_rate * C**2
        )
        expected_angular = (
            mass_rate * moments.specific_angular_momentum
        )
        expected_energy = mass_rate * moments.specific_killing_energy
        angular_scale = max(
            abs(angular_rate),
            abs(expected_angular),
            1.0,
        )
        killing_scale = max(
            abs(energy_rate),
            abs(expected_energy),
            1.0,
        )
        angular_defect = (
            angular_rate - expected_angular
        ) / angular_scale
        energy_defect = (
            energy_rate - expected_energy
        ) / killing_scale
        paired_power = provider.geometry.pattern_omega * (
            angular_rate - solution.saddle_flux.angular_momentum
        )
        power_scale = max(
            abs(solution.binary_power_gain),
            abs(paired_power),
            1.0,
        )
        pattern_defect = (
            solution.binary_power_gain - paired_power
        ) / power_scale
        if max(
            abs(angular_defect),
            abs(energy_defect),
            abs(pattern_defect),
        ) > ledger_tolerance:
            raise ValueError("Roche and Kerr-Schild ledgers do not close")

    radial_momentum_rate = pressure_traction + nozzle_force
    weighted_flux = np.asarray(
        [
            mass_rate / C,
            radial_momentum_rate / C**2,
            angular_rate / C**2,
            energy_rate / C**3,
        ],
        dtype=float,
    )
    characteristic_speeds = (
        valencia_radial_characteristic_speeds_over_c(
            geometry.base,
            radial_velocity_over_c=primitive.radial_velocity_over_c,
            azimuthal_velocity_over_c=primitive.azimuthal_velocity_over_c,
            sound_speed_over_c=thermodynamics.sound_speed / C,
        )
    )
    incoming = int(
        np.sum(np.asarray(characteristic_speeds) < -1.0e-12)
    )
    if incoming != 1:
        raise ValueError("Roche edge requires one incoming acoustic mode")
    return KerrSchildRocheBoundaryAudit(
        edge_state=edge_state,
        gate=gate,
        weighted_killing_flux_over_c=weighted_flux,
        rest_mass_rate=mass_rate,
        radial_momentum_rate=float(radial_momentum_rate),
        angular_momentum_rate=angular_rate,
        killing_energy_rate=energy_rate,
        pressure_traction=pressure_traction,
        incoming_outer_characteristics=incoming,
        no_inward_mass=mass_rate >= 0.0,
        zero_outer_stress=True,
        angular_momentum_relative_defect=float(angular_defect),
        killing_energy_relative_defect=float(energy_defect),
        binary_pattern_power_relative_defect=float(pattern_defect),
    )


def audit_kerr_schild_migration_rank(
    n_cells: int,
    boundary: KerrSchildRocheBoundaryAudit,
) -> KerrSchildMigrationRankAudit:
    """Return the unchanged square DAE count and outer face-row rank."""

    count = valencia_flux_primary_count(n_cells)
    face_jacobian = np.eye(4)
    rank = int(np.linalg.matrix_rank(face_jacobian))
    return KerrSchildMigrationRankAudit(
        n_cells=int(n_cells),
        total_unknowns=count.total_unknowns,
        total_rows=count.total_rows,
        source_unknowns=0,
        source_rows=0,
        boundary_face_rows=4,
        boundary_face_jacobian_rank=rank,
        physical_outer_boundary_conditions=(
            boundary.incoming_outer_characteristics
        ),
        square=count.total_unknowns == count.total_rows,
    )
