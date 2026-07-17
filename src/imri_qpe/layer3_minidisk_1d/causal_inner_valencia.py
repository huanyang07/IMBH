"""Horizon-penetrating conservative prototype for the causal inner flow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from imri_qpe.constants import C, G


@dataclass(frozen=True)
class SchwarzschildKerrSchildGeometry:
    """Equatorial ingoing Kerr-Schild 3+1 geometry."""

    radius: float
    gravitational_radius: float
    lapse: float
    radial_shift_over_c: float
    gamma_rr: float
    gamma_phiphi: float
    inverse_gamma_rr: float
    sqrt_spatial_metric: float

    @property
    def horizon_radius(self) -> float:
        return 2.0 * self.gravitational_radius

    @property
    def proper_column_jacobian(self) -> float:
        """Return ``2 pi R sqrt(gamma_rr)`` for a vertical column."""

        return 2.0 * np.pi * self.radius * np.sqrt(self.gamma_rr)

    @property
    def ingoing_light_speed_over_c(self) -> float:
        return (
            -self.lapse / np.sqrt(self.gamma_rr)
            - self.radial_shift_over_c
        )

    @property
    def outgoing_light_speed_over_c(self) -> float:
        return (
            self.lapse / np.sqrt(self.gamma_rr)
            - self.radial_shift_over_c
        )


def schwarzschild_kerr_schild_geometry(
    radius: float,
    M_g: float,
) -> SchwarzschildKerrSchildGeometry:
    """Return the horizon-penetrating Schwarzschild metric at the equator."""

    radius = float(radius)
    mass = float(M_g)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be positive and finite")
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("black-hole mass must be positive and finite")
    gravitational_radius = G * mass / C**2
    metric_ratio = 2.0 * gravitational_radius / radius
    gamma_rr = 1.0 + metric_ratio
    lapse = 1.0 / np.sqrt(gamma_rr)
    radial_shift = metric_ratio / gamma_rr
    return SchwarzschildKerrSchildGeometry(
        radius=radius,
        gravitational_radius=gravitational_radius,
        lapse=float(lapse),
        radial_shift_over_c=float(radial_shift),
        gamma_rr=float(gamma_rr),
        gamma_phiphi=float(radius**2),
        inverse_gamma_rr=float(1.0 / gamma_rr),
        sqrt_spatial_metric=float(np.sqrt(gamma_rr) * radius**2),
    )


@dataclass(frozen=True)
class ValenciaColumnState:
    """Mass-equivalent Valencia state and radial coordinate flux.

    The conserved order is ``(D, S_r, S_phi, tau)``. Energy is divided by
    ``c^2`` and momentum by ``c`` so the first, second, and fourth entries
    share surface-mass units. ``S_phi`` is the covariant angular-momentum
    density divided by ``c``. The returned flux is divided by ``c``.
    """

    conserved: np.ndarray
    flux_over_c: np.ndarray
    lorentz_factor: float
    specific_enthalpy_over_c2: float
    transport_velocity_over_c: float


def valencia_column_state(
    geometry: SchwarzschildKerrSchildGeometry,
    *,
    surface_density: float,
    radial_velocity_over_c: float,
    azimuthal_velocity_over_c: float,
    specific_internal_energy: float,
    integrated_pressure: float,
) -> ValenciaColumnState:
    """Map one vertically integrated primitive state to Valencia variables."""

    sigma = float(surface_density)
    beta_r = float(radial_velocity_over_c)
    beta_phi = float(azimuthal_velocity_over_c)
    internal_energy = float(specific_internal_energy)
    pressure = float(integrated_pressure)
    values = (
        sigma,
        beta_r,
        beta_phi,
        internal_energy,
        pressure,
    )
    if any(not np.isfinite(value) for value in values):
        raise ValueError("Valencia column primitives must be finite")
    if sigma <= 0.0:
        raise ValueError("surface density must be positive")
    if internal_energy < 0.0 or pressure < 0.0:
        raise ValueError("internal energy and integrated pressure cannot be negative")
    speed_squared = beta_r**2 + beta_phi**2
    if speed_squared >= 1.0:
        raise ValueError("Eulerian three-velocity must be subluminal")

    lorentz_factor = 1.0 / np.sqrt(1.0 - speed_squared)
    pressure_mass = pressure / C**2
    enthalpy = 1.0 + internal_energy / C**2 + pressure_mass / sigma
    coordinate_v_r = beta_r / np.sqrt(geometry.gamma_rr)
    covariant_v_r = np.sqrt(geometry.gamma_rr) * beta_r
    covariant_v_phi = geometry.radius * beta_phi
    transport_velocity = (
        geometry.lapse * coordinate_v_r
        - geometry.radial_shift_over_c
    )

    common = sigma * enthalpy * lorentz_factor**2
    rest_mass = sigma * lorentz_factor
    radial_momentum = common * covariant_v_r
    angular_momentum = common * covariant_v_phi
    thermal_enthalpy = (
        internal_energy / C**2 + pressure_mass / sigma
    )
    energy = rest_mass * (
        lorentz_factor - 1.0
        + thermal_enthalpy * lorentz_factor
    ) - pressure_mass
    conserved = np.asarray(
        [rest_mass, radial_momentum, angular_momentum, energy],
        dtype=float,
    )
    flux = np.asarray(
        [
            rest_mass * transport_velocity,
            radial_momentum * transport_velocity
            + geometry.lapse * pressure_mass,
            angular_momentum * transport_velocity,
            energy * transport_velocity
            + geometry.lapse * pressure_mass * coordinate_v_r,
        ],
        dtype=float,
    )
    if np.any(~np.isfinite(conserved)) or np.any(~np.isfinite(flux)):
        raise ValueError("Valencia state or flux is not finite")
    return ValenciaColumnState(
        conserved=conserved,
        flux_over_c=flux,
        lorentz_factor=float(lorentz_factor),
        specific_enthalpy_over_c2=float(enthalpy),
        transport_velocity_over_c=float(transport_velocity),
    )


def valencia_radial_characteristic_speeds_over_c(
    geometry: SchwarzschildKerrSchildGeometry,
    *,
    radial_velocity_over_c: float,
    azimuthal_velocity_over_c: float,
    sound_speed_over_c: float,
) -> tuple[float, float, float, float]:
    """Return Valencia radial speeds including transverse rotation."""

    beta_r = float(radial_velocity_over_c)
    beta_phi = float(azimuthal_velocity_over_c)
    sound = float(sound_speed_over_c)
    if any(not np.isfinite(value) for value in (beta_r, beta_phi, sound)):
        raise ValueError("characteristic inputs must be finite")
    speed_squared = beta_r**2 + beta_phi**2
    if speed_squared >= 1.0:
        raise ValueError("Eulerian three-velocity must be subluminal")
    if not 0.0 < sound < 1.0:
        raise ValueError("sound speed must lie strictly between zero and c")

    coordinate_v_r = beta_r / np.sqrt(geometry.gamma_rr)
    sound_squared = sound**2
    denominator = 1.0 - speed_squared * sound_squared
    radicand = (1.0 - speed_squared) * (
        geometry.inverse_gamma_rr
        * (1.0 - speed_squared * sound_squared)
        - coordinate_v_r**2 * (1.0 - sound_squared)
    )
    if radicand < -1.0e-14:
        raise ValueError("Valencia acoustic radicand is negative")
    root = np.sqrt(max(radicand, 0.0))
    advective = (
        geometry.lapse * coordinate_v_r
        - geometry.radial_shift_over_c
    )
    acoustic_minus = (
        geometry.lapse
        * (
            coordinate_v_r * (1.0 - sound_squared)
            - sound * root
        )
        / denominator
        - geometry.radial_shift_over_c
    )
    acoustic_plus = (
        geometry.lapse
        * (
            coordinate_v_r * (1.0 - sound_squared)
            + sound * root
        )
        / denominator
        - geometry.radial_shift_over_c
    )
    speeds = (
        float(acoustic_minus),
        float(advective),
        float(advective),
        float(acoustic_plus),
    )
    light_min = geometry.ingoing_light_speed_over_c
    light_max = geometry.outgoing_light_speed_over_c
    tolerance = 2.0e-13
    if any(
        value < light_min - tolerance or value > light_max + tolerance
        for value in speeds
    ):
        raise ValueError("fluid characteristic lies outside the local light cone")
    return speeds


@dataclass(frozen=True)
class ValenciaCharacteristicAudit:
    """Characteristic and stationary-flux rank audit at one radial state."""

    analytic_speeds_over_c: tuple[float, float, float, float]
    numerical_speeds_over_c: tuple[float, float, float, float]
    incoming_inner_characteristics: int
    stationary_flux_rank: int
    smallest_stationary_singular_value: float
    maximum_eigenvalue_defect: float

    @property
    def causally_outgoing_inner_edge(self) -> bool:
        return self.incoming_inner_characteristics == 0


@dataclass(frozen=True)
class ValenciaFluxPrimaryCount:
    """Exact unknown and residual count for the flux-primary DAE."""

    n_cells: int
    conserved_unknowns: int
    primitive_unknowns: int
    face_flux_unknowns: int
    total_unknowns: int
    conservation_rows: int
    primitive_map_rows: int
    interior_flux_rows: int
    inner_flux_rows: int
    outer_flux_rows: int
    total_rows: int
    physical_inner_boundary_rows: int


def valencia_flux_primary_count(n_cells: int) -> ValenciaFluxPrimaryCount:
    """Return the square WP10b DAE count for ``n_cells``."""

    if int(n_cells) != n_cells or n_cells < 1:
        raise ValueError("Valencia DAE requires a positive integer cell count")
    n_cells = int(n_cells)
    conserved = 4 * n_cells
    primitives = 4 * n_cells
    face_fluxes = 4 * (n_cells + 1)
    conservation_rows = 4 * n_cells
    primitive_rows = 4 * n_cells
    interior_rows = 4 * (n_cells - 1)
    inner_rows = 4
    outer_rows = 4
    return ValenciaFluxPrimaryCount(
        n_cells=n_cells,
        conserved_unknowns=conserved,
        primitive_unknowns=primitives,
        face_flux_unknowns=face_fluxes,
        total_unknowns=conserved + primitives + face_fluxes,
        conservation_rows=conservation_rows,
        primitive_map_rows=primitive_rows,
        interior_flux_rows=interior_rows,
        inner_flux_rows=inner_rows,
        outer_flux_rows=outer_rows,
        total_rows=(
            conservation_rows
            + primitive_rows
            + interior_rows
            + inner_rows
            + outer_rows
        ),
        physical_inner_boundary_rows=0,
    )


def _ideal_gas_state_from_chart(
    geometry: SchwarzschildKerrSchildGeometry,
    chart: np.ndarray,
    gamma_gas: float,
) -> ValenciaColumnState:
    log_sigma, beta_r, beta_phi, log_pressure_mass = map(float, chart)
    sigma = np.exp(log_sigma)
    pressure_mass = np.exp(log_pressure_mass)
    pressure = pressure_mass * C**2
    internal_energy = pressure / ((gamma_gas - 1.0) * sigma)
    return valencia_column_state(
        geometry,
        surface_density=sigma,
        radial_velocity_over_c=beta_r,
        azimuthal_velocity_over_c=beta_phi,
        specific_internal_energy=internal_energy,
        integrated_pressure=pressure,
    )


def audit_ideal_gas_valencia_eigensystem(
    geometry: SchwarzschildKerrSchildGeometry,
    *,
    surface_density: float,
    radial_velocity_over_c: float,
    azimuthal_velocity_over_c: float,
    integrated_pressure: float,
    gamma_gas: float = 4.0 / 3.0,
    finite_difference_step: float = 2.0e-6,
) -> ValenciaCharacteristicAudit:
    """Compare analytic Valencia speeds with the conservative flux Jacobian."""

    sigma = float(surface_density)
    pressure = float(integrated_pressure)
    if sigma <= 0.0 or pressure <= 0.0:
        raise ValueError("ideal-gas density and pressure must be positive")
    if not 1.0 < gamma_gas <= 2.0:
        raise ValueError("ideal-gas gamma must lie in (1, 2]")
    if not 0.0 < finite_difference_step < 1.0e-2:
        raise ValueError("finite-difference step must be positive and small")
    chart = np.asarray(
        [
            np.log(sigma),
            radial_velocity_over_c,
            azimuthal_velocity_over_c,
            np.log(pressure / C**2),
        ],
        dtype=float,
    )
    conserved_jacobian = np.empty((4, 4), dtype=float)
    flux_jacobian = np.empty((4, 4), dtype=float)
    for index in range(4):
        step = finite_difference_step
        plus = np.array(chart, copy=True)
        minus = np.array(chart, copy=True)
        plus[index] += step
        minus[index] -= step
        plus_state = _ideal_gas_state_from_chart(
            geometry, plus, gamma_gas
        )
        minus_state = _ideal_gas_state_from_chart(
            geometry, minus, gamma_gas
        )
        conserved_jacobian[:, index] = (
            plus_state.conserved - minus_state.conserved
        ) / (2.0 * step)
        flux_jacobian[:, index] = (
            plus_state.flux_over_c - minus_state.flux_over_c
        ) / (2.0 * step)

    conservative_flux_jacobian = np.linalg.solve(
        conserved_jacobian.T,
        flux_jacobian.T,
    ).T
    numerical = np.linalg.eigvals(conservative_flux_jacobian)
    if np.max(np.abs(np.imag(numerical))) > 1.0e-8:
        raise ValueError("Valencia flux Jacobian has non-real eigenvalues")
    numerical = np.sort(np.real(numerical))

    pressure_mass = pressure / C**2
    specific_internal = pressure_mass / ((gamma_gas - 1.0) * sigma)
    enthalpy = 1.0 + specific_internal + pressure_mass / sigma
    sound = np.sqrt(gamma_gas * pressure_mass / (sigma * enthalpy))
    analytic = np.sort(
        np.asarray(
            valencia_radial_characteristic_speeds_over_c(
                geometry,
                radial_velocity_over_c=radial_velocity_over_c,
                azimuthal_velocity_over_c=azimuthal_velocity_over_c,
                sound_speed_over_c=float(sound),
            ),
            dtype=float,
        )
    )
    reference = _ideal_gas_state_from_chart(
        geometry, chart, gamma_gas
    )
    component_scale = np.maximum(
        np.abs(reference.conserved),
        np.max(np.abs(reference.conserved)) * 1.0e-12,
    )
    scaled_flux_jacobian = (
        conservative_flux_jacobian
        * component_scale[np.newaxis, :]
        / component_scale[:, np.newaxis]
    )
    singular_values = np.linalg.svd(
        scaled_flux_jacobian, compute_uv=False
    )
    rank_threshold = max(
        float(np.max(np.abs(numerical))) * 1.0e-8,
        1.0e-10,
    )
    return ValenciaCharacteristicAudit(
        analytic_speeds_over_c=tuple(float(value) for value in analytic),
        numerical_speeds_over_c=tuple(float(value) for value in numerical),
        incoming_inner_characteristics=int(np.sum(analytic > 0.0)),
        stationary_flux_rank=int(
            np.sum(np.abs(numerical) > rank_threshold)
        ),
        smallest_stationary_singular_value=float(singular_values[-1]),
        maximum_eigenvalue_defect=float(np.max(np.abs(analytic - numerical))),
    )
