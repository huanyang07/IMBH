"""Adiabatic Hill/Roche overflow nozzle for a truncated minidisk."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.optimize import brentq, least_squares

from imri_qpe.constants import A_RAD, DEFAULT_MU_MOL, G
from imri_qpe.scales import gas_constant_per_gram

from .entropy_advection import (
    gas_radiation_adiabatic_sound_speed_squared,
    gas_radiation_specific_enthalpy,
    gas_radiation_specific_entropy,
    total_pressure,
)
from .tidal_exchange import HillTidalGeometry, fiducial_hill_tidal_geometry
from .transonic_potential import PaczynskiWiitaPotential


@dataclass(frozen=True)
class HillRocheNozzleGeometry:
    """Secondary-centered rotating Hill geometry for one overflow channel."""

    secondary_mass: float
    pattern_omega: float
    nominal_hill_radius: float
    saddle_radius: float
    transverse_curvature_y: float
    transverse_curvature_z: float
    channel_count: int = 2
    filling_factor: float = 1.0
    frame: str = "secondary-centered-binary-corotating-hill-pw"
    energy_zero: str = "pw-disk-matched-at-reservoir-edge"

    def validated(self) -> HillRocheNozzleGeometry:
        scalar_positive = {
            "secondary_mass": self.secondary_mass,
            "pattern_omega": self.pattern_omega,
            "nominal_hill_radius": self.nominal_hill_radius,
            "saddle_radius": self.saddle_radius,
            "transverse_curvature_y": self.transverse_curvature_y,
            "transverse_curvature_z": self.transverse_curvature_z,
        }
        if any(
            not np.isfinite(value) or value <= 0.0
            for value in scalar_positive.values()
        ):
            raise ValueError("Hill/Roche geometry values must be positive and finite")
        potential = PaczynskiWiitaPotential(float(self.secondary_mass))
        if self.saddle_radius <= potential.r_pw:
            raise ValueError("Hill/Roche saddle must lie outside the pseudo-horizon")
        force_scale = G * self.secondary_mass / self.saddle_radius**2
        force_residual = float(
            hill_roche_midplane_force_derivative(
                self.saddle_radius,
                self.secondary_mass,
                self.pattern_omega,
            )
        )
        if abs(force_residual) > 1.0e-10 * force_scale:
            raise ValueError("Hill/Roche saddle does not satisfy force balance")
        expected_y = G * self.secondary_mass / (
            self.saddle_radius
            * (self.saddle_radius - potential.r_pw) ** 2
        )
        expected_z = expected_y + self.pattern_omega**2
        if not np.isclose(
            self.transverse_curvature_y, expected_y, rtol=1.0e-12
        ) or not np.isclose(
            self.transverse_curvature_z, expected_z, rtol=1.0e-12
        ):
            raise ValueError("Hill/Roche transverse curvatures are inconsistent")
        if int(self.channel_count) != self.channel_count or not 1 <= int(
            self.channel_count
        ) <= 2:
            raise ValueError("channel_count must be one or two")
        if not np.isfinite(self.filling_factor) or not 0.0 < self.filling_factor <= 1.0:
            raise ValueError("filling_factor must lie in (0,1]")
        if not self.frame or not self.energy_zero:
            raise ValueError("frame and energy_zero must be declared")
        return self


@dataclass(frozen=True)
class HillRocheNozzleReservoir:
    """Thermodynamic/contact state feeding the effective overflow channel."""

    radius: float
    density: float
    pressure: float
    radial_velocity: float
    specific_angular_momentum: float
    temperature: float | None = None
    specific_inertial_bernoulli: float | None = None
    specific_flux_angular_momentum: float | None = None

    def validated(self) -> HillRocheNozzleReservoir:
        values = (
            self.radius,
            self.density,
            self.pressure,
            self.radial_velocity,
            self.specific_angular_momentum,
        )
        if any(not np.isfinite(value) for value in values):
            raise ValueError("nozzle reservoir values must be finite")
        if self.radius <= 0.0 or self.density <= 0.0 or self.pressure <= 0.0:
            raise ValueError("nozzle radius, density, and pressure must be positive")
        if self.temperature is not None and (
            not np.isfinite(self.temperature) or self.temperature <= 0.0
        ):
            raise ValueError("nozzle temperature must be positive and finite")
        optional_values = (
            self.specific_inertial_bernoulli,
            self.specific_flux_angular_momentum,
        )
        if any(
            value is not None and not np.isfinite(value)
            for value in optional_values
        ):
            raise ValueError("optional nozzle flux moments must be finite")
        return self


@dataclass(frozen=True)
class OverflowFluxState:
    """One shared conservative flux state evaluated at the sonic saddle."""

    mass: float
    radial_momentum: float
    angular_momentum: float
    total_energy: float
    rotating_energy: float

    def validated(self) -> OverflowFluxState:
        values = (
            self.mass,
            self.radial_momentum,
            self.angular_momentum,
            self.total_energy,
            self.rotating_energy,
        )
        if any(not np.isfinite(value) for value in values):
            raise ValueError("overflow flux state must be finite")
        if self.mass < 0.0:
            raise ValueError("overflow mass flux must be outward or zero")
        return self


@dataclass(frozen=True)
class HillRocheNozzleSolution:
    """Choked polytropic solution and its rotating/inertial flux ledger."""

    geometry: HillRocheNozzleGeometry
    reservoir: HillRocheNozzleReservoir
    gamma: float
    rotating_bernoulli: float
    available_specific_energy: float
    sonic_sound_speed: float
    sonic_density: float
    sonic_pressure: float
    density_weighted_throat_area: float
    integrated_throat_pressure: float
    saddle_specific_angular_momentum: float
    saddle_flux: OverflowFluxState
    edge_angular_momentum_flux: float
    edge_total_energy_flux: float
    binary_angular_momentum_gain: float
    binary_power_gain: float
    sonic_residual: float
    jacobi_residual: float
    energy_pairing_residual: float
    thermal_model: str = "fixed_gamma_polytrope"
    sonic_temperature: float | None = None
    entropy_residual: float = 0.0
    transverse_quadrature_zones: int = 0

    @property
    def choked(self) -> bool:
        return self.saddle_flux.mass > 0.0


@dataclass(frozen=True)
class HillRocheNozzleQuadratureAudit:
    """Numerical transverse integration compared with the analytic throat."""

    radial_zones: int
    mass_flux: float
    integrated_pressure: float
    mass_relative_error: float
    pressure_relative_error: float


@dataclass(frozen=True)
class HillRocheNozzleGate:
    """Energetic decision for a closed or regular choked overflow channel."""

    choked: bool
    rotating_bernoulli: float
    saddle_potential: float
    available_specific_energy: float
    reservoir_enthalpy: float
    required_enthalpy_multiplier: float
    solution: HillRocheNozzleSolution | None


@runtime_checkable
class OverflowBoundaryProvider(Protocol):
    """Boundary-physics interface consumed by future disk-edge coupling."""

    geometry: HillRocheNozzleGeometry

    def solve(
        self, reservoir: HillRocheNozzleReservoir
    ) -> HillRocheNozzleSolution:
        """Return one conservative regular overflow solution."""

    def evaluate(
        self, reservoir: HillRocheNozzleReservoir
    ) -> HillRocheNozzleGate:
        """Return a closed-channel gate or its regular overflow solution."""


def hill_roche_midplane_potential(
    radius,
    secondary_mass: float,
    pattern_omega: float,
):
    """Return the unshifted PW-secondary plus local Hill tidal potential."""

    potential = PaczynskiWiitaPotential(float(secondary_mass))
    radius = np.asarray(radius, dtype=float)
    return potential.phi(radius) - 1.5 * float(pattern_omega) ** 2 * radius**2


def hill_roche_disk_matched_potential(
    radius,
    secondary_mass: float,
    pattern_omega: float,
    reservoir_radius: float,
):
    """Return the Hill effective potential on the disk's PW energy zero.

    The constant ``Omega_p^2 R_edge^2`` makes ``B_J + Omega_p l`` at the
    reservoir edge equal the PW disk Bernoulli.  It changes no force or nozzle
    availability because the same constant applies at the saddle.
    """

    if not np.isfinite(reservoir_radius) or reservoir_radius <= 0.0:
        raise ValueError("reservoir_radius must be positive and finite")
    return (
        hill_roche_midplane_potential(
            radius, secondary_mass, pattern_omega
        )
        + float(pattern_omega) ** 2 * float(reservoir_radius) ** 2
    )


def _hill_roche_budget_potentials(
    reservoir: HillRocheNozzleReservoir,
    geometry: HillRocheNozzleGeometry,
    *,
    enthalpy: float,
    kinetic: float,
) -> tuple[float, float]:
    """Return edge/saddle potentials on the declared inertial-energy zero."""

    if reservoir.specific_inertial_bernoulli is None:
        edge_potential = float(
            hill_roche_disk_matched_potential(
                reservoir.radius,
                geometry.secondary_mass,
                geometry.pattern_omega,
                reservoir.radius,
            )
        )
        saddle_potential = float(
            hill_roche_disk_matched_potential(
                geometry.saddle_radius,
                geometry.secondary_mass,
                geometry.pattern_omega,
                reservoir.radius,
            )
        )
        return edge_potential, saddle_potential

    flux_angular_momentum = (
        reservoir.specific_angular_momentum
        if reservoir.specific_flux_angular_momentum is None
        else reservoir.specific_flux_angular_momentum
    )
    rotating_bernoulli = (
        reservoir.specific_inertial_bernoulli
        - geometry.pattern_omega * flux_angular_momentum
    )
    edge_potential = rotating_bernoulli - enthalpy - kinetic
    unshifted_edge = float(
        hill_roche_midplane_potential(
            reservoir.radius,
            geometry.secondary_mass,
            geometry.pattern_omega,
        )
    )
    unshifted_saddle = float(
        hill_roche_midplane_potential(
            geometry.saddle_radius,
            geometry.secondary_mass,
            geometry.pattern_omega,
        )
    )
    saddle_potential = (
        edge_potential + unshifted_saddle - unshifted_edge
    )
    return float(edge_potential), float(saddle_potential)


def hill_roche_midplane_force_derivative(
    radius,
    secondary_mass: float,
    pattern_omega: float,
):
    """Return d(Phi_H)/dR along either symmetric Hill escape axis."""

    potential = PaczynskiWiitaPotential(float(secondary_mass))
    radius = np.asarray(radius, dtype=float)
    return potential.dphi_dR(radius) - 3.0 * float(pattern_omega) ** 2 * radius


def make_hill_roche_nozzle_geometry(
    secondary_mass: float,
    tidal_geometry: HillTidalGeometry,
    *,
    channel_count: int = 2,
    filling_factor: float = 1.0,
) -> HillRocheNozzleGeometry:
    """Locate the PW-corrected Hill saddle and its transverse curvatures."""

    potential = PaczynskiWiitaPotential(float(secondary_mass))
    nominal = float(tidal_geometry.hill_radius)
    omega = float(tidal_geometry.pattern_omega)
    if nominal <= potential.r_pw or omega <= 0.0:
        raise ValueError("nominal Hill geometry is not physical")
    lower = max(0.5 * nominal, 1.01 * potential.r_pw)
    upper = 1.5 * nominal
    saddle = float(
        brentq(
            lambda radius: float(
                hill_roche_midplane_force_derivative(
                    radius, secondary_mass, omega
                )
            ),
            lower,
            upper,
            xtol=1.0e-12 * nominal,
            rtol=1.0e-13,
        )
    )
    gravity_transverse = G * float(secondary_mass) / (
        saddle * (saddle - potential.r_pw) ** 2
    )
    return HillRocheNozzleGeometry(
        secondary_mass=float(secondary_mass),
        pattern_omega=omega,
        nominal_hill_radius=nominal,
        saddle_radius=saddle,
        transverse_curvature_y=float(gravity_transverse),
        transverse_curvature_z=float(gravity_transverse + omega**2),
        channel_count=int(channel_count),
        filling_factor=float(filling_factor),
    ).validated()


def fiducial_hill_roche_nozzle_geometry(
    *,
    channel_count: int = 2,
    filling_factor: float = 1.0,
) -> HillRocheNozzleGeometry:
    """Return the repository fiducial secondary/Hill nozzle geometry."""

    from imri_qpe.parameters import FiducialParams

    params = FiducialParams()
    return make_hill_roche_nozzle_geometry(
        params.M2_g,
        fiducial_hill_tidal_geometry(params),
        channel_count=channel_count,
        filling_factor=filling_factor,
    )


class HillRocheNozzleProvider:
    """Algebraic choked-flow provider for an adiabatic effective side channel."""

    def __init__(
        self,
        geometry: HillRocheNozzleGeometry,
        *,
        gamma: float,
    ) -> None:
        self.geometry = geometry.validated()
        self.gamma = float(gamma)
        if not np.isfinite(self.gamma) or not 1.0 < self.gamma < 2.0:
            raise ValueError("nozzle gamma must lie in (1,2)")

    def solve(
        self, reservoir: HillRocheNozzleReservoir
    ) -> HillRocheNozzleSolution:
        """Return the regular sonic overflow selected by reservoir entropy."""

        reservoir = reservoir.validated()
        geometry = self.geometry
        if reservoir.radius >= geometry.saddle_radius:
            raise ValueError("nozzle reservoir must lie inside the Hill saddle")
        gamma = self.gamma
        entropy_constant = reservoir.pressure / reservoir.density**gamma
        (
            rotating_bernoulli,
            saddle_potential,
            enthalpy,
            _required_enthalpy_multiplier,
        ) = self._reservoir_budget(reservoir)
        available = rotating_bernoulli - saddle_potential
        if available <= 0.0:
            raise ValueError(
                "reservoir Jacobi Bernoulli does not reach the Hill saddle"
            )
        sound_speed_squared = (
            2.0 * (gamma - 1.0) / (gamma + 1.0) * available
        )
        sound_speed = float(np.sqrt(sound_speed_squared))
        sonic_density = float(
            (sound_speed_squared / (gamma * entropy_constant))
            ** (1.0 / (gamma - 1.0))
        )
        sonic_pressure = float(
            entropy_constant * sonic_density**gamma
        )
        curvature_product = float(
            np.sqrt(
                geometry.transverse_curvature_y
                * geometry.transverse_curvature_z
            )
        )
        channel_factor = float(
            geometry.channel_count * geometry.filling_factor
        )
        density_weighted_area = float(
            channel_factor
            * 2.0
            * np.pi
            * sound_speed_squared
            / (gamma * curvature_product)
        )
        mass_flux = float(sonic_density * sound_speed * density_weighted_area)
        integrated_pressure = float(
            channel_factor
            * 2.0
            * np.pi
            * sonic_pressure
            * sound_speed_squared
            / ((2.0 * gamma - 1.0) * curvature_product)
        )
        radial_momentum_flux = float(
            mass_flux * sound_speed + integrated_pressure
        )
        saddle_l = float(
            geometry.pattern_omega * geometry.saddle_radius**2
        )
        rotating_energy_flux = float(mass_flux * rotating_bernoulli)
        angular_flux = float(mass_flux * saddle_l)
        total_energy_flux = float(
            rotating_energy_flux + geometry.pattern_omega * angular_flux
        )
        saddle_flux = OverflowFluxState(
            mass=mass_flux,
            radial_momentum=radial_momentum_flux,
            angular_momentum=angular_flux,
            total_energy=total_energy_flux,
            rotating_energy=rotating_energy_flux,
        ).validated()
        edge_specific_l = (
            reservoir.specific_angular_momentum
            if reservoir.specific_flux_angular_momentum is None
            else reservoir.specific_flux_angular_momentum
        )
        edge_angular_flux = float(mass_flux * edge_specific_l)
        edge_total_energy_flux = float(
            rotating_energy_flux
            + geometry.pattern_omega * edge_angular_flux
        )
        binary_angular_gain = edge_angular_flux - angular_flux
        binary_power_gain = geometry.pattern_omega * binary_angular_gain
        sonic_enthalpy = sound_speed_squared / (gamma - 1.0)
        jacobi_scale = max(abs(rotating_bernoulli), abs(saddle_potential), 1.0)
        jacobi_residual = float(
            (
                rotating_bernoulli
                - saddle_potential
                - 0.5 * sound_speed_squared
                - sonic_enthalpy
            )
            / jacobi_scale
        )
        force_scale = max(
            abs(G * geometry.secondary_mass / geometry.saddle_radius**2),
            1.0,
        )
        sonic_residual = float(
            hill_roche_midplane_force_derivative(
                geometry.saddle_radius,
                geometry.secondary_mass,
                geometry.pattern_omega,
            )
            / force_scale
        )
        energy_scale = max(abs(edge_total_energy_flux), 1.0)
        energy_pairing_residual = float(
            (
                edge_total_energy_flux
                - saddle_flux.total_energy
                - binary_power_gain
            )
            / energy_scale
        )
        return HillRocheNozzleSolution(
            geometry=geometry,
            reservoir=reservoir,
            gamma=gamma,
            rotating_bernoulli=rotating_bernoulli,
            available_specific_energy=float(available),
            sonic_sound_speed=sound_speed,
            sonic_density=sonic_density,
            sonic_pressure=sonic_pressure,
            density_weighted_throat_area=density_weighted_area,
            integrated_throat_pressure=integrated_pressure,
            saddle_specific_angular_momentum=saddle_l,
            saddle_flux=saddle_flux,
            edge_angular_momentum_flux=edge_angular_flux,
            edge_total_energy_flux=edge_total_energy_flux,
            binary_angular_momentum_gain=float(binary_angular_gain),
            binary_power_gain=float(binary_power_gain),
            sonic_residual=sonic_residual,
            jacobi_residual=jacobi_residual,
            energy_pairing_residual=energy_pairing_residual,
        )

    def evaluate(
        self, reservoir: HillRocheNozzleReservoir
    ) -> HillRocheNozzleGate:
        """Return a closed-channel gate or its regular choked solution."""

        reservoir = reservoir.validated()
        if reservoir.radius >= self.geometry.saddle_radius:
            raise ValueError("nozzle reservoir must lie inside the Hill saddle")
        (
            rotating_bernoulli,
            saddle_potential,
            enthalpy,
            required_multiplier,
        ) = self._reservoir_budget(reservoir)
        available = rotating_bernoulli - saddle_potential
        solution = self.solve(reservoir) if available > 0.0 else None
        return HillRocheNozzleGate(
            choked=solution is not None,
            rotating_bernoulli=rotating_bernoulli,
            saddle_potential=saddle_potential,
            available_specific_energy=float(available),
            reservoir_enthalpy=enthalpy,
            required_enthalpy_multiplier=required_multiplier,
            solution=solution,
        )

    def available_specific_energy(
        self, reservoir: HillRocheNozzleReservoir
    ) -> float:
        """Return the energetic gate without launching the sonic solve."""

        reservoir = reservoir.validated()
        rotating_bernoulli, saddle_potential, _enthalpy, _multiplier = (
            self._reservoir_budget(reservoir)
        )
        return float(rotating_bernoulli - saddle_potential)

    def _reservoir_budget(
        self, reservoir: HillRocheNozzleReservoir
    ) -> tuple[float, float, float, float]:
        geometry = self.geometry
        gamma = self.gamma
        enthalpy = float(
            gamma
            / (gamma - 1.0)
            * reservoir.pressure
            / reservoir.density
        )
        rotating_tangential_velocity = (
            reservoir.specific_angular_momentum / reservoir.radius
            - geometry.pattern_omega * reservoir.radius
        )
        kinetic = float(
            0.5 * reservoir.radial_velocity**2
            + 0.5 * rotating_tangential_velocity**2
        )
        edge_potential, saddle_potential = (
            _hill_roche_budget_potentials(
                reservoir,
                geometry,
                enthalpy=enthalpy,
                kinetic=kinetic,
            )
        )
        rotating_bernoulli = edge_potential + enthalpy + kinetic
        required_enthalpy = max(saddle_potential - edge_potential - kinetic, 0.0)
        required_multiplier = required_enthalpy / enthalpy
        return (
            float(rotating_bernoulli),
            saddle_potential,
            enthalpy,
            float(required_multiplier),
        )


class GasRadiationHillRocheNozzleProvider:
    """Adiabatic Hill nozzle using the project's exact gas+radiation EOS."""

    def __init__(
        self,
        geometry: HillRocheNozzleGeometry,
        *,
        mu_mol: float = DEFAULT_MU_MOL,
        gamma_gas: float = 5.0 / 3.0,
        transverse_quadrature_zones: int = 48,
    ) -> None:
        self.geometry = geometry.validated()
        self.mu_mol = float(mu_mol)
        self.gamma_gas = float(gamma_gas)
        self.transverse_quadrature_zones = int(transverse_quadrature_zones)
        if not np.isfinite(self.mu_mol) or self.mu_mol <= 0.0:
            raise ValueError("mu_mol must be positive and finite")
        if not np.isfinite(self.gamma_gas) or self.gamma_gas <= 1.0:
            raise ValueError("gamma_gas must exceed one")
        if (
            self.transverse_quadrature_zones != transverse_quadrature_zones
            or self.transverse_quadrature_zones < 8
        ):
            raise ValueError("transverse_quadrature_zones must be an integer >= 8")

    def _validated_thermal_reservoir(
        self, reservoir: HillRocheNozzleReservoir
    ) -> HillRocheNozzleReservoir:
        reservoir = reservoir.validated()
        if reservoir.temperature is None:
            raise ValueError("gas+radiation nozzle requires reservoir temperature")
        expected_pressure = float(
            total_pressure(
                reservoir.density,
                reservoir.temperature,
                mu_mol=self.mu_mol,
            )
        )
        if not np.isclose(
            reservoir.pressure, expected_pressure, rtol=2.0e-10, atol=0.0
        ):
            raise ValueError(
                "gas+radiation reservoir pressure is inconsistent with rho and T"
            )
        if reservoir.radius >= self.geometry.saddle_radius:
            raise ValueError("nozzle reservoir must lie inside the Hill saddle")
        return reservoir

    def _entropy(self, density: float, temperature: float) -> float:
        return float(
            gas_radiation_specific_entropy(
                density,
                temperature,
                mu_mol=self.mu_mol,
                gamma_gas=self.gamma_gas,
            )
        )

    def _enthalpy(self, density: float, temperature: float) -> float:
        return float(
            gas_radiation_specific_enthalpy(
                density,
                temperature,
                mu_mol=self.mu_mol,
                gamma_gas=self.gamma_gas,
            )
        )

    def _sound_speed_squared(
        self, density: float, temperature: float
    ) -> float:
        return float(
            gas_radiation_adiabatic_sound_speed_squared(
                density,
                temperature,
                mu_mol=self.mu_mol,
                gamma_gas=self.gamma_gas,
            )
        )

    def _solve_isentropic_enthalpy_state(
        self,
        entropy: float,
        target_enthalpy: float,
        reservoir: HillRocheNozzleReservoir,
        seed: tuple[float, float],
    ) -> tuple[float, float, float]:
        if target_enthalpy <= 0.0:
            raise ValueError("target isentropic enthalpy must be positive")
        assert reservoir.temperature is not None
        R_gas = gas_constant_per_gram(self.mu_mol)
        entropy_scale = max(abs(entropy), R_gas, 1.0)
        enthalpy_scale = max(target_enthalpy, 1.0)
        lower_density = max(reservoir.density * 1.0e-24, 1.0e-60)
        lower_temperature = max(reservoir.temperature * 1.0e-10, 1.0e-2)
        upper_density = reservoir.density * (1.0 + 1.0e-9)
        upper_temperature = reservoir.temperature * (1.0 + 1.0e-9)
        initial = np.log(
            np.clip(
                np.asarray(seed, dtype=float),
                [lower_density, lower_temperature],
                [upper_density, upper_temperature],
            )
        )

        def residual(log_state: np.ndarray) -> np.ndarray:
            density, temperature = np.exp(log_state)
            return np.asarray(
                [
                    (self._entropy(density, temperature) - entropy)
                    / entropy_scale,
                    (self._enthalpy(density, temperature) - target_enthalpy)
                    / enthalpy_scale,
                ],
                dtype=float,
            )

        result = least_squares(
            residual,
            initial,
            bounds=(
                np.log([lower_density, lower_temperature]),
                np.log([upper_density, upper_temperature]),
            ),
            xtol=2.0e-13,
            ftol=2.0e-13,
            gtol=2.0e-13,
            max_nfev=300,
        )
        maximum_residual = float(np.max(np.abs(result.fun)))
        if not result.success or maximum_residual > 2.0e-8:
            raise ValueError("gas+radiation isentropic state solve did not converge")
        density, temperature = map(float, np.exp(result.x))
        return density, temperature, maximum_residual

    def _reservoir_budget(
        self, reservoir: HillRocheNozzleReservoir
    ) -> tuple[float, float, float, float]:
        assert reservoir.temperature is not None
        geometry = self.geometry
        enthalpy = self._enthalpy(reservoir.density, reservoir.temperature)
        rotating_tangential_velocity = (
            reservoir.specific_angular_momentum / reservoir.radius
            - geometry.pattern_omega * reservoir.radius
        )
        kinetic = float(
            0.5 * reservoir.radial_velocity**2
            + 0.5 * rotating_tangential_velocity**2
        )
        edge_potential, saddle_potential = (
            _hill_roche_budget_potentials(
                reservoir,
                geometry,
                enthalpy=enthalpy,
                kinetic=kinetic,
            )
        )
        rotating_bernoulli = edge_potential + enthalpy + kinetic
        required_enthalpy = max(
            saddle_potential - edge_potential - kinetic, 0.0
        )
        return (
            float(rotating_bernoulli),
            saddle_potential,
            enthalpy,
            float(required_enthalpy / enthalpy),
        )

    def evaluate(
        self, reservoir: HillRocheNozzleReservoir
    ) -> HillRocheNozzleGate:
        """Return the exact-EOS closed gate or regular choked solution."""

        reservoir = self._validated_thermal_reservoir(reservoir)
        rotating_bernoulli, saddle_potential, enthalpy, multiplier = (
            self._reservoir_budget(reservoir)
        )
        available = rotating_bernoulli - saddle_potential
        solution = self.solve(reservoir) if available > 0.0 else None
        return HillRocheNozzleGate(
            choked=solution is not None,
            rotating_bernoulli=rotating_bernoulli,
            saddle_potential=saddle_potential,
            available_specific_energy=float(available),
            reservoir_enthalpy=enthalpy,
            required_enthalpy_multiplier=multiplier,
            solution=solution,
        )

    def available_specific_energy(
        self, reservoir: HillRocheNozzleReservoir
    ) -> float:
        """Return the exact-EOS energetic gate without solving the throat."""

        reservoir = self._validated_thermal_reservoir(reservoir)
        rotating_bernoulli, saddle_potential, _enthalpy, _multiplier = (
            self._reservoir_budget(reservoir)
        )
        return float(rotating_bernoulli - saddle_potential)

    def solve(
        self, reservoir: HillRocheNozzleReservoir
    ) -> HillRocheNozzleSolution:
        """Solve entropy conservation and sonic regularity at the saddle."""

        reservoir = self._validated_thermal_reservoir(reservoir)
        assert reservoir.temperature is not None
        geometry = self.geometry
        rotating_bernoulli, saddle_potential, reservoir_enthalpy, _ = (
            self._reservoir_budget(reservoir)
        )
        available = float(rotating_bernoulli - saddle_potential)
        if available <= 0.0:
            raise ValueError(
                "reservoir Jacobi Bernoulli does not reach the Hill saddle"
            )
        entropy = self._entropy(reservoir.density, reservoir.temperature)
        reservoir_sound_squared = self._sound_speed_squared(
            reservoir.density, reservoir.temperature
        )
        reservoir_gamma_one = (
            reservoir_sound_squared
            * reservoir.density
            / reservoir.pressure
        )
        enthalpy_fraction = np.clip(
            available / max(reservoir_enthalpy, 1.0), 1.0e-10, 0.999
        )
        exponent = 1.0 / max(reservoir_gamma_one - 1.0, 1.0e-3)
        initial = np.log(
            [
                reservoir.density * enthalpy_fraction**exponent,
                reservoir.temperature * enthalpy_fraction,
            ]
        )
        R_gas = gas_constant_per_gram(self.mu_mol)
        entropy_scale = max(abs(entropy), R_gas, 1.0)
        energy_scale = max(available, 1.0)
        lower = np.log(
            [
                max(reservoir.density * 1.0e-24, 1.0e-60),
                max(reservoir.temperature * 1.0e-10, 1.0e-2),
            ]
        )
        upper = np.log(
            [
                reservoir.density * (1.0 + 1.0e-9),
                reservoir.temperature * (1.0 + 1.0e-9),
            ]
        )

        def sonic_residual(log_state: np.ndarray) -> np.ndarray:
            density, temperature = np.exp(log_state)
            sound_squared = self._sound_speed_squared(density, temperature)
            return np.asarray(
                [
                    (self._entropy(density, temperature) - entropy)
                    / entropy_scale,
                    (
                        self._enthalpy(density, temperature)
                        + 0.5 * sound_squared
                        - available
                    )
                    / energy_scale,
                ],
                dtype=float,
            )

        root = least_squares(
            sonic_residual,
            np.clip(initial, lower, upper),
            bounds=(lower, upper),
            xtol=2.0e-13,
            ftol=2.0e-13,
            gtol=2.0e-13,
            max_nfev=500,
        )
        maximum_sonic_residual = float(np.max(np.abs(root.fun)))
        if not root.success or maximum_sonic_residual > 2.0e-8:
            raise ValueError("gas+radiation sonic state solve did not converge")
        sonic_density, sonic_temperature = map(float, np.exp(root.x))
        sonic_pressure = float(
            total_pressure(
                sonic_density, sonic_temperature, mu_mol=self.mu_mol
            )
        )
        sound_speed_squared = self._sound_speed_squared(
            sonic_density, sonic_temperature
        )
        sound_speed = float(np.sqrt(sound_speed_squared))
        sonic_enthalpy = self._enthalpy(sonic_density, sonic_temperature)

        nodes, weights = np.polynomial.legendre.leggauss(
            self.transverse_quadrature_zones
        )
        potential_offsets = 0.5 * sonic_enthalpy * (nodes + 1.0)
        integration_weights = 0.5 * sonic_enthalpy * weights
        transverse_density = np.empty(nodes.size, dtype=float)
        transverse_pressure = np.empty(nodes.size, dtype=float)
        entropy_error = maximum_sonic_residual
        seed = (sonic_density, sonic_temperature)
        for index, offset in enumerate(potential_offsets):
            target_enthalpy = sonic_enthalpy - float(offset)
            density, temperature, local_error = (
                self._solve_isentropic_enthalpy_state(
                    entropy,
                    target_enthalpy,
                    reservoir,
                    seed,
                )
            )
            transverse_density[index] = density
            transverse_pressure[index] = float(
                total_pressure(density, temperature, mu_mol=self.mu_mol)
            )
            entropy_error = max(entropy_error, local_error)
            seed = (density, temperature)
        curvature_product = float(
            np.sqrt(
                geometry.transverse_curvature_y
                * geometry.transverse_curvature_z
            )
        )
        area_jacobian = float(
            geometry.channel_count
            * geometry.filling_factor
            * 2.0
            * np.pi
            / curvature_product
        )
        density_integral = float(
            area_jacobian
            * np.sum(integration_weights * transverse_density)
        )
        pressure_integral = float(
            area_jacobian
            * np.sum(integration_weights * transverse_pressure)
        )
        mass_flux = float(sound_speed * density_integral)
        radial_momentum_flux = float(
            mass_flux * sound_speed + pressure_integral
        )
        saddle_l = float(
            geometry.pattern_omega * geometry.saddle_radius**2
        )
        rotating_energy_flux = float(mass_flux * rotating_bernoulli)
        saddle_angular_flux = float(mass_flux * saddle_l)
        saddle_total_energy_flux = float(
            rotating_energy_flux
            + geometry.pattern_omega * saddle_angular_flux
        )
        saddle_flux = OverflowFluxState(
            mass=mass_flux,
            radial_momentum=radial_momentum_flux,
            angular_momentum=saddle_angular_flux,
            total_energy=saddle_total_energy_flux,
            rotating_energy=rotating_energy_flux,
        ).validated()
        edge_specific_l = (
            reservoir.specific_angular_momentum
            if reservoir.specific_flux_angular_momentum is None
            else reservoir.specific_flux_angular_momentum
        )
        edge_angular_flux = float(mass_flux * edge_specific_l)
        edge_total_energy_flux = float(
            rotating_energy_flux
            + geometry.pattern_omega * edge_angular_flux
        )
        binary_angular_gain = edge_angular_flux - saddle_angular_flux
        binary_power_gain = geometry.pattern_omega * binary_angular_gain
        jacobi_scale = max(
            abs(rotating_bernoulli), abs(saddle_potential), 1.0
        )
        jacobi_residual = float(
            (
                rotating_bernoulli
                - saddle_potential
                - 0.5 * sound_speed_squared
                - sonic_enthalpy
            )
            / jacobi_scale
        )
        force_scale = max(
            abs(G * geometry.secondary_mass / geometry.saddle_radius**2),
            1.0,
        )
        regularity_residual = float(
            hill_roche_midplane_force_derivative(
                geometry.saddle_radius,
                geometry.secondary_mass,
                geometry.pattern_omega,
            )
            / force_scale
        )
        energy_scale = max(abs(edge_total_energy_flux), 1.0)
        energy_pairing_residual = float(
            (
                edge_total_energy_flux
                - saddle_total_energy_flux
                - binary_power_gain
            )
            / energy_scale
        )
        gamma_one = float(
            sound_speed_squared * sonic_density / sonic_pressure
        )
        return HillRocheNozzleSolution(
            geometry=geometry,
            reservoir=reservoir,
            gamma=gamma_one,
            rotating_bernoulli=rotating_bernoulli,
            available_specific_energy=available,
            sonic_sound_speed=sound_speed,
            sonic_density=sonic_density,
            sonic_pressure=sonic_pressure,
            density_weighted_throat_area=float(
                density_integral / sonic_density
            ),
            integrated_throat_pressure=pressure_integral,
            saddle_specific_angular_momentum=saddle_l,
            saddle_flux=saddle_flux,
            edge_angular_momentum_flux=edge_angular_flux,
            edge_total_energy_flux=edge_total_energy_flux,
            binary_angular_momentum_gain=float(binary_angular_gain),
            binary_power_gain=float(binary_power_gain),
            sonic_residual=regularity_residual,
            jacobi_residual=jacobi_residual,
            energy_pairing_residual=energy_pairing_residual,
            thermal_model="gas_radiation_eos",
            sonic_temperature=sonic_temperature,
            entropy_residual=float(entropy_error),
            transverse_quadrature_zones=self.transverse_quadrature_zones,
        )


def audit_hill_roche_nozzle_transverse_quadrature(
    solution: HillRocheNozzleSolution,
    radial_zones: int,
) -> HillRocheNozzleQuadratureAudit:
    """Integrate the polytropic sonic throat on a uniform potential mesh."""

    if solution.thermal_model != "fixed_gamma_polytrope":
        raise ValueError("polytropic quadrature audit requires fixed-gamma solution")
    if int(radial_zones) != radial_zones or radial_zones < 2:
        raise ValueError("radial_zones must be an integer of at least two")
    radial_zones = int(radial_zones)
    gamma = float(solution.gamma)
    coordinate = (np.arange(radial_zones, dtype=float) + 0.5) / radial_zones
    density = solution.sonic_density * (
        1.0 - coordinate
    ) ** (1.0 / (gamma - 1.0))
    pressure = solution.sonic_pressure * (
        1.0 - coordinate
    ) ** (gamma / (gamma - 1.0))
    sound_speed_squared = solution.sonic_sound_speed**2
    curvature_product = float(
        np.sqrt(
            solution.geometry.transverse_curvature_y
            * solution.geometry.transverse_curvature_z
        )
    )
    channel_factor = float(
        solution.geometry.channel_count * solution.geometry.filling_factor
    )
    central_enthalpy = sound_speed_squared / (gamma - 1.0)
    area_per_coordinate = (
        channel_factor
        * 2.0
        * np.pi
        * central_enthalpy
        / curvature_product
    )
    cell_area = area_per_coordinate / radial_zones
    mass_flux = float(
        solution.sonic_sound_speed * cell_area * np.sum(density)
    )
    integrated_pressure = float(cell_area * np.sum(pressure))
    return HillRocheNozzleQuadratureAudit(
        radial_zones=radial_zones,
        mass_flux=mass_flux,
        integrated_pressure=integrated_pressure,
        mass_relative_error=float(
            abs(mass_flux / solution.saddle_flux.mass - 1.0)
        ),
        pressure_relative_error=float(
            abs(integrated_pressure / solution.integrated_throat_pressure - 1.0)
        ),
    )
