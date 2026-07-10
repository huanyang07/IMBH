"""Unified conservative transport for stream-fed, mass-losing slim disks.

The legacy transonic solver eliminates angular momentum algebraically and
evolves a local entropy equation.  This module keeps those equations intact
while providing the production variables and ledgers needed by the next
solver: inward mass flux, inward angular-momentum flux, and inward total-energy
flux.

The sign convention is outward-increasing ``x = ln R`` with inward-positive
``Mdot``.  Positive ``stream_prime`` adds matter to the disk and positive
``wind_prime`` removes it::

    dMdot/dx = wind_prime - stream_prime

For ``J = Mdot*l - G`` and ``E = Mdot*B - Omega*G`` the matching source
conventions are::

    dJ/dx = wind_prime*l_w - stream_prime*l_s + tau_ext
    dE/dx = L_rad_prime + wind_prime*B_w - stream_prime*B_s + power_ext

``tau_ext`` and ``power_ext`` are positive losses from the radial disk flux.
They are deliberately independent: a physical torque model must specify its
associated power rather than relying on an implicit stream-angular offset.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from imri_qpe.constants import C

from .transonic_local import algebraic_state, entropy_gradient_log, state_partials
from .transonic_potential import PaczynskiWiitaPotential
from .transonic_thermo import integrated_stress, radiative_cooling, surface_density, vertical_state
from .winds import wind_energy_per_mass


@dataclass(frozen=True)
class ConservativeScales:
    """Reference scales used to condition conservative variables."""

    mdot: float
    specific_angular_momentum: float
    specific_energy: float = C**2

    def __post_init__(self) -> None:
        if not np.isfinite(self.mdot) or self.mdot <= 0.0:
            raise ValueError("mdot scale must be positive and finite")
        if not np.isfinite(self.specific_angular_momentum) or self.specific_angular_momentum <= 0.0:
            raise ValueError("specific-angular-momentum scale must be positive and finite")
        if not np.isfinite(self.specific_energy) or self.specific_energy <= 0.0:
            raise ValueError("specific-energy scale must be positive and finite")

    @property
    def angular_flux(self) -> float:
        return float(self.mdot * self.specific_angular_momentum)

    @property
    def energy_flux(self) -> float:
        return float(self.mdot * self.specific_energy)


@dataclass(frozen=True)
class PhysicalTransportClosure:
    """Explicit stream, wind, and external transport prescription.

    ``stream_specific_angular_momentum`` and ``stream_specific_energy`` must be
    supplied by a capture/ballistic model.  A circularization radius is a
    supported deterministic fallback for the initial implementation.

    ``wind_angular_momentum_factor=1`` is the baseline thermal/radiative wind.
    Values above one represent a lever arm.  The associated torque work is
    included in the carried wind energy, preventing an angular/energy mismatch.
    """

    stream_specific_angular_momentum: float | None = None
    stream_specific_energy: float | None = None
    stream_circularization_radius: float | None = None
    wind_angular_momentum_factor: float = 1.0
    wind_launch_energy_multiplier: float = 1.0
    external_torque_prime: float = 0.0
    external_power_prime: float = 0.0

    def __post_init__(self) -> None:
        for name, value in {
            "stream_specific_angular_momentum": self.stream_specific_angular_momentum,
            "stream_specific_energy": self.stream_specific_energy,
            "stream_circularization_radius": self.stream_circularization_radius,
        }.items():
            if value is not None and not np.isfinite(float(value)):
                raise ValueError(f"{name} must be finite when supplied")
        if self.stream_circularization_radius is not None and self.stream_circularization_radius <= 0.0:
            raise ValueError("stream_circularization_radius must be positive")
        if not np.isfinite(self.wind_angular_momentum_factor) or self.wind_angular_momentum_factor < 1.0:
            raise ValueError("wind_angular_momentum_factor must be finite and at least one")
        if not np.isfinite(self.wind_launch_energy_multiplier) or self.wind_launch_energy_multiplier < 0.0:
            raise ValueError("wind_launch_energy_multiplier must be finite and non-negative")
        if not np.isfinite(self.external_torque_prime):
            raise ValueError("external_torque_prime must be finite")
        if not np.isfinite(self.external_power_prime):
            raise ValueError("external_power_prime must be finite")


@dataclass(frozen=True)
class ConservativeNodeState:
    """Local disk state reconstructed from conservative primary variables."""

    logR: float
    R: float
    u: float
    T: float
    F: float
    mdot: float
    Sigma: float
    H: float
    rho: float
    P: float
    Pi: float
    e: float
    W: float
    G: float
    J: float
    l: float
    Omega: float
    Omega_K: float
    l_K: float
    Q_rad: float
    enthalpy: float
    bernoulli: float
    mechanical_energy_flux: float


@dataclass(frozen=True)
class CarriedTransport:
    """Specific angular momentum and energy carried by source and wind."""

    l_stream: float
    l_wind: float
    B_stream: float
    B_wind: float
    wind_launch_energy: float
    wind_torque_work: float


@dataclass(frozen=True)
class ConservativeSourceTerms:
    """Physical source terms per unit ``dlnR``."""

    stream_prime: float
    wind_prime: float
    radiative_loss_prime: float
    mass_rhs: float
    angular_rhs: float
    energy_rhs: float
    carried: CarriedTransport


@dataclass(frozen=True)
class ConservativeIntervalResidual:
    """Dimensionless finite-volume residuals for one interval."""

    mass: float
    angular_momentum: float
    energy: float

    def as_array(self) -> np.ndarray:
        return np.asarray([self.mass, self.angular_momentum, self.energy], dtype=float)


@dataclass(frozen=True)
class EnergyIdentityAudit:
    """Pointwise relation between legacy entropy and conservative energy."""

    mechanical_flux_derivative: float
    entropy_expected_derivative: float
    raw_identity_defect: float
    vertical_work_derivative: float
    corrected_identity_defect: float
    normalized_raw_defect: float
    normalized_corrected_defect: float


def default_conservative_scales(params) -> ConservativeScales:
    """Return stable scales based on the inner accretion rate and ISCO orbit."""

    potential = PaczynskiWiitaPotential(params.M2_g)
    return ConservativeScales(
        mdot=float(params.Mdot_g_s),
        specific_angular_momentum=float(potential.l_k(potential.r_isco)),
    )


def reconstruct_conservative_state(
    logR: float,
    logu: float,
    logT: float,
    F: float,
    j: float,
    params,
    scales: ConservativeScales | None = None,
) -> ConservativeNodeState:
    """Reconstruct a local state from ``(logu, logT, F, j)``.

    The angular flux ``j`` is primary.  The disk angular momentum follows from
    ``l = (J + G)/Mdot`` rather than from the legacy cumulative stream offset.
    """

    scales = default_conservative_scales(params) if scales is None else scales
    if not np.isfinite(F) or F <= 0.0:
        raise ValueError("F must be positive and finite")
    potential = PaczynskiWiitaPotential(params.M2_g)
    R = float(np.exp(logR))
    u = float(np.exp(logu))
    T = float(np.exp(logT))
    mdot = float(scales.mdot * F)
    Sigma = float(surface_density(mdot, R, u))
    vertical = vertical_state(
        Sigma,
        T,
        R,
        potential,
        mu_mol=params.mu_mol,
        kappa=params.kappa,
        gamma_gas=params.gamma_gas,
    )
    stress = float(
        integrated_stress(
            vertical,
            params.alpha,
            mu_stress=params.mu_stress,
            stress_factor=params.stress_factor,
        )
    )
    torque = float(2.0 * np.pi * R**2 * stress)
    angular_flux = float(j * scales.angular_flux)
    specific_l = float((angular_flux + torque) / mdot)
    omega = float(specific_l / R**2)
    enthalpy = float(vertical.e + vertical.Pi / Sigma)
    bernoulli = float(
        0.5 * u**2
        + 0.5 * (specific_l / R) ** 2
        + potential.phi(R)
        + enthalpy
    )
    energy_flux = float(mdot * bernoulli - omega * torque)
    return ConservativeNodeState(
        logR=float(logR),
        R=R,
        u=u,
        T=T,
        F=float(F),
        mdot=mdot,
        Sigma=Sigma,
        H=float(vertical.H),
        rho=float(vertical.rho),
        P=float(vertical.P_tot),
        Pi=float(vertical.Pi),
        e=float(vertical.e),
        W=stress,
        G=torque,
        J=angular_flux,
        l=specific_l,
        Omega=omega,
        Omega_K=float(vertical.Omega_K),
        l_K=float(potential.l_k(R)),
        Q_rad=float(radiative_cooling(vertical, kappa=params.kappa)),
        enthalpy=enthalpy,
        bernoulli=bernoulli,
        mechanical_energy_flux=energy_flux,
    )


def _circular_orbit_energy(potential: PaczynskiWiitaPotential, radius: float) -> float:
    l_k = float(potential.l_k(radius))
    return float(potential.phi(radius) + 0.5 * (l_k / radius) ** 2)


def carried_transport(
    state: ConservativeNodeState,
    closure: PhysicalTransportClosure,
    params,
) -> CarriedTransport:
    """Evaluate explicit stream and wind carried quantities."""

    potential = PaczynskiWiitaPotential(params.M2_g)
    injection_radius = closure.stream_circularization_radius
    if closure.stream_specific_angular_momentum is not None:
        l_stream = float(closure.stream_specific_angular_momentum)
    elif injection_radius is not None:
        l_stream = float(potential.l_k(injection_radius))
    else:
        raise ValueError(
            "physical stream closure requires stream_specific_angular_momentum "
            "or stream_circularization_radius"
        )

    if closure.stream_specific_energy is not None:
        B_stream = float(closure.stream_specific_energy)
    elif injection_radius is not None:
        B_stream = _circular_orbit_energy(potential, float(injection_radius))
    else:
        raise ValueError(
            "physical stream closure requires stream_specific_energy or "
            "stream_circularization_radius"
        )

    l_wind = float(closure.wind_angular_momentum_factor * state.l)
    torque_work = float(state.Omega * (l_wind - state.l))
    launch = float(
        closure.wind_launch_energy_multiplier
        * wind_energy_per_mass(params.M2_g, state.R)
    )
    B_wind = float(state.bernoulli + launch + torque_work)
    return CarriedTransport(
        l_stream=l_stream,
        l_wind=l_wind,
        B_stream=B_stream,
        B_wind=B_wind,
        wind_launch_energy=launch,
        wind_torque_work=torque_work,
    )


def conservative_source_terms(
    state: ConservativeNodeState,
    *,
    stream_prime: float,
    wind_prime: float,
    closure: PhysicalTransportClosure,
    params,
    radiative_loss_prime: float | None = None,
) -> ConservativeSourceTerms:
    """Return mass, angular, and energy source terms with one sign convention."""

    if not np.isfinite(stream_prime) or stream_prime < 0.0:
        raise ValueError("stream_prime must be finite and non-negative")
    if not np.isfinite(wind_prime) or wind_prime < 0.0:
        raise ValueError("wind_prime must be finite and non-negative")
    carried = carried_transport(state, closure, params)
    if radiative_loss_prime is None:
        radiative_loss_prime = float(2.0 * np.pi * state.R**2 * state.Q_rad)
    if not np.isfinite(radiative_loss_prime):
        raise ValueError("radiative_loss_prime must be finite")
    mass_rhs = float(wind_prime - stream_prime)
    angular_rhs = float(
        wind_prime * carried.l_wind
        - stream_prime * carried.l_stream
        + closure.external_torque_prime
    )
    energy_rhs = float(
        radiative_loss_prime
        + wind_prime * carried.B_wind
        - stream_prime * carried.B_stream
        + closure.external_power_prime
    )
    return ConservativeSourceTerms(
        stream_prime=float(stream_prime),
        wind_prime=float(wind_prime),
        radiative_loss_prime=float(radiative_loss_prime),
        mass_rhs=mass_rhs,
        angular_rhs=angular_rhs,
        energy_rhs=energy_rhs,
        carried=carried,
    )


def simpson_interval_residual(
    dx: float,
    left: ConservativeNodeState,
    midpoint: ConservativeNodeState,
    right: ConservativeNodeState,
    source_left: ConservativeSourceTerms,
    source_midpoint: ConservativeSourceTerms,
    source_right: ConservativeSourceTerms,
    scales: ConservativeScales,
    *,
    energy_flux_left: float | None = None,
    energy_flux_right: float | None = None,
) -> ConservativeIntervalResidual:
    """Return finite-volume Simpson rows for one radial interval."""

    if not np.isfinite(dx) or dx <= 0.0:
        raise ValueError("dx must be positive and finite")
    mass_integral = (dx / 6.0) * (
        source_left.mass_rhs + 4.0 * source_midpoint.mass_rhs + source_right.mass_rhs
    )
    angular_integral = (dx / 6.0) * (
        source_left.angular_rhs
        + 4.0 * source_midpoint.angular_rhs
        + source_right.angular_rhs
    )
    energy_integral = (dx / 6.0) * (
        source_left.energy_rhs
        + 4.0 * source_midpoint.energy_rhs
        + source_right.energy_rhs
    )
    E_left = left.mechanical_energy_flux if energy_flux_left is None else float(energy_flux_left)
    E_right = right.mechanical_energy_flux if energy_flux_right is None else float(energy_flux_right)
    return ConservativeIntervalResidual(
        mass=float((right.mdot - left.mdot - mass_integral) / scales.mdot),
        angular_momentum=float((right.J - left.J - angular_integral) / scales.angular_flux),
        energy=float((E_right - E_left - energy_integral) / scales.energy_flux),
    )


def legacy_energy_identity_audit(
    logR: float,
    y,
    g,
    lambda0: float,
    params,
) -> EnergyIdentityAudit:
    """Compare the legacy entropy row with a total-energy flux derivative.

    The legacy radial equation uses ``dPi/Sigma`` while its entropy equation
    uses ``P drho/rho**2``.  Their difference is the work associated with the
    changing one-zone vertical column.  Adding the reported vertical-work
    derivative makes the conservative identity exact without altering the
    legacy equations.
    """

    y = np.asarray(y, dtype=float)
    g = np.asarray(g, dtype=float)
    if y.shape != (2,) or g.shape != (2,):
        raise ValueError("y and g must each have shape (2,)")
    state = algebraic_state(logR, float(y[0]), float(y[1]), lambda0, params)
    partials = state_partials(
        logR,
        y,
        lambda0,
        params,
        eps_x=params.partial_eps,
        eps_y=params.partial_eps,
    )
    dPi_dx = float(partials.x["Pi"] + np.dot(partials.y["Pi"], g))
    drho_dx = float(partials.x["rho"] + np.dot(partials.y["rho"], g))
    de_dx = float(partials.x["e"] + np.dot(partials.y["e"], g))
    dOmega_dx = float(partials.x["Omega"] + np.dot(partials.y["Omega"], g))
    mdot = float(2.0 * np.pi * state.R * state.Sigma * state.u)
    torque = float(2.0 * np.pi * state.R**2 * state.W)
    area = float(2.0 * np.pi * state.R**2)
    Tdsdx = float(entropy_gradient_log(logR, y, g, lambda0, params))

    mechanical_derivative = float(
        mdot * (de_dx - dPi_dx / state.Sigma) - torque * dOmega_dx
    )
    q_visc = float(-state.W * dOmega_dx)
    q_adv = float(-(state.Sigma * state.u / state.R) * Tdsdx)
    local_energy_residual = float(q_visc - state.Q_rad - q_adv)
    expected_derivative = float(area * state.Q_rad + area * local_energy_residual)
    raw_defect = float(mechanical_derivative - expected_derivative)
    vertical_work = float(
        mdot * (dPi_dx / state.Sigma - state.P * drho_dx / state.rho**2)
    )
    corrected = float(mechanical_derivative + vertical_work - expected_derivative)
    scale = max(
        abs(mechanical_derivative),
        abs(expected_derivative),
        abs(vertical_work),
        abs(area * state.Q_rad),
        1.0,
    )
    return EnergyIdentityAudit(
        mechanical_flux_derivative=mechanical_derivative,
        entropy_expected_derivative=expected_derivative,
        raw_identity_defect=raw_defect,
        vertical_work_derivative=vertical_work,
        corrected_identity_defect=corrected,
        normalized_raw_defect=float(raw_defect / scale),
        normalized_corrected_defect=float(corrected / scale),
    )


def normalized_energy_flux(value: float, scales: ConservativeScales) -> float:
    """Normalize a dimensional total-energy flux."""

    if not math.isfinite(value):
        raise ValueError("energy flux must be finite")
    return float(value / scales.energy_flux)
