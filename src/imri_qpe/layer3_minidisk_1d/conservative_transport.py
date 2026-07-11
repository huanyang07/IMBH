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
    wind_launch_mode: str = "eta"
    wind_terminal_bernoulli: float = 0.0
    wind_mass_loading_cap_per_log_radius: float | None = None
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
        if self.wind_launch_mode not in {"eta", "terminal_bernoulli"}:
            raise ValueError("wind_launch_mode must be 'eta' or 'terminal_bernoulli'")
        if not np.isfinite(self.wind_terminal_bernoulli):
            raise ValueError("wind_terminal_bernoulli must be finite")
        if self.wind_mass_loading_cap_per_log_radius is not None and (
            not np.isfinite(self.wind_mass_loading_cap_per_log_radius)
            or self.wind_mass_loading_cap_per_log_radius <= 0.0
        ):
            raise ValueError(
                "wind_mass_loading_cap_per_log_radius must be positive when supplied"
            )
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
    wind_base_energy_prime: float
    wind_launch_power_prime: float
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
class ConservativeIntervalTransport:
    """Decomposed physical transport integrated across one radial interval."""

    stream_mass: float
    stream_mass_quadrature: float
    wind_mass: float
    stream_angular_momentum: float
    wind_angular_momentum: float
    external_angular_momentum: float
    stream_energy: float
    wind_base_energy: float
    wind_launch_energy: float
    wind_energy: float
    radiative_energy: float
    external_energy: float

    @property
    def mass_rhs(self) -> float:
        return float(self.wind_mass - self.stream_mass)

    @property
    def angular_rhs(self) -> float:
        return float(
            self.wind_angular_momentum
            - self.stream_angular_momentum
            + self.external_angular_momentum
        )

    @property
    def energy_rhs(self) -> float:
        return float(
            self.radiative_energy
            + self.wind_energy
            - self.stream_energy
            + self.external_energy
        )

    @property
    def stream_mass_quadrature_error(self) -> float:
        return float(self.stream_mass_quadrature - self.stream_mass)


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


@dataclass(frozen=True)
class WindEscapeDiagnostics:
    """Pointwise Bernoulli audit for the prescribed wind launch energy."""

    disk_bernoulli: float
    target_terminal_bernoulli: float
    required_launch_energy: float
    prescribed_launch_energy: float
    wind_bernoulli: float
    terminal_margin: float
    terminal_speed: float
    escaping: bool


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
    launch = wind_launch_energy(state, closure, params)
    B_wind = float(state.bernoulli + launch + torque_work)
    return CarriedTransport(
        l_stream=l_stream,
        l_wind=l_wind,
        B_stream=B_stream,
        B_wind=B_wind,
        wind_launch_energy=launch,
        wind_torque_work=torque_work,
    )


def wind_launch_energy(
    state: ConservativeNodeState,
    closure: PhysicalTransportClosure,
    params,
) -> float:
    """Return the launch energy selected by the physical closure."""

    if closure.wind_launch_mode == "eta":
        launch = float(
            closure.wind_launch_energy_multiplier
            * wind_energy_per_mass(params.M2_g, state.R)
        )
    else:
        l_wind = float(closure.wind_angular_momentum_factor * state.l)
        torque_work = float(state.Omega * (l_wind - state.l))
        launch = float(
            closure.wind_terminal_bernoulli - state.bernoulli - torque_work
        )
        if launch <= 0.0:
            raise ValueError(
                "terminal-Bernoulli wind requires positive launch energy; "
                "the requested target is already met by the local disk state"
            )
    if not np.isfinite(launch) or launch <= 0.0:
        raise ValueError("wind launch energy must be positive and finite")
    return launch


def conservative_source_terms(
    state: ConservativeNodeState,
    *,
    stream_prime: float,
    wind_prime: float,
    closure: PhysicalTransportClosure,
    params,
    radiative_loss_prime: float | None = None,
    wind_launch_power_prime: float | None = None,
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
    wind_base_energy_prime = float(
        wind_prime * (carried.B_wind - carried.wind_launch_energy)
    )
    if wind_launch_power_prime is None:
        wind_launch_power_prime = float(wind_prime * carried.wind_launch_energy)
    if not np.isfinite(wind_launch_power_prime) or wind_launch_power_prime < 0.0:
        raise ValueError("wind_launch_power_prime must be finite and non-negative")
    mass_rhs = float(wind_prime - stream_prime)
    angular_rhs = float(
        wind_prime * carried.l_wind
        - stream_prime * carried.l_stream
        + closure.external_torque_prime
    )
    energy_rhs = float(
        radiative_loss_prime
        + wind_base_energy_prime
        + wind_launch_power_prime
        - stream_prime * carried.B_stream
        + closure.external_power_prime
    )
    return ConservativeSourceTerms(
        stream_prime=float(stream_prime),
        wind_prime=float(wind_prime),
        radiative_loss_prime=float(radiative_loss_prime),
        wind_base_energy_prime=wind_base_energy_prime,
        wind_launch_power_prime=float(wind_launch_power_prime),
        mass_rhs=mass_rhs,
        angular_rhs=angular_rhs,
        energy_rhs=energy_rhs,
        carried=carried,
    )


def integrate_sampled_interval_transport(
    dx: float,
    sources,
    weights,
    *,
    exact_stream_mass: float | None = None,
    exact_stream_angular_momentum: float | None = None,
    exact_stream_energy: float | None = None,
) -> ConservativeIntervalTransport:
    """Integrate conservative source samples with normalized quadrature weights.

    Stream mass may use an analytic cell integral while state-dependent wind,
    radiation, and external terms retain Simpson quadrature.  If only the exact
    stream mass is supplied, the stream specific angular momentum and energy
    must be constant across the interval, as in the current capture closure.
    """

    if not np.isfinite(dx) or dx <= 0.0:
        raise ValueError("dx must be positive and finite")
    sources = tuple(sources)
    weights = np.asarray(tuple(weights), dtype=float)
    if len(sources) == 0 or weights.shape != (len(sources),):
        raise ValueError("source samples and quadrature weights must have equal nonzero length")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("quadrature weights must be finite and non-negative")
    if not np.isclose(np.sum(weights), 1.0, rtol=0.0, atol=1.0e-13):
        raise ValueError("quadrature weights must sum to one")

    def quadrature(values) -> float:
        array = np.asarray(tuple(values), dtype=float)
        if array.shape != weights.shape or np.any(~np.isfinite(array)):
            raise ValueError("interval transport values must match the quadrature rule")
        return float(dx * np.dot(weights, array))

    stream_mass_quadrature = quadrature(source.stream_prime for source in sources)
    if exact_stream_mass is None:
        stream_mass = stream_mass_quadrature
    else:
        stream_mass = float(exact_stream_mass)
        if not np.isfinite(stream_mass) or stream_mass < 0.0:
            raise ValueError("exact stream mass must be finite and non-negative")

    l_stream = np.asarray([source.carried.l_stream for source in sources], dtype=float)
    B_stream = np.asarray([source.carried.B_stream for source in sources], dtype=float)
    if exact_stream_angular_momentum is None:
        if exact_stream_mass is None:
            stream_angular = quadrature(
                source.stream_prime * source.carried.l_stream for source in sources
            )
        else:
            if not np.allclose(l_stream, l_stream[1], rtol=1.0e-12, atol=0.0):
                raise ValueError("varying stream angular momentum requires an exact moment")
            stream_angular = float(stream_mass * l_stream[1])
    else:
        stream_angular = float(exact_stream_angular_momentum)

    if exact_stream_energy is None:
        if exact_stream_mass is None:
            stream_energy = quadrature(
                source.stream_prime * source.carried.B_stream for source in sources
            )
        else:
            if not np.allclose(B_stream, B_stream[1], rtol=1.0e-12, atol=0.0):
                raise ValueError("varying stream energy requires an exact moment")
            stream_energy = float(stream_mass * B_stream[1])
    else:
        stream_energy = float(exact_stream_energy)

    wind_mass = quadrature(source.wind_prime for source in sources)
    wind_angular = quadrature(
        source.wind_prime * source.carried.l_wind for source in sources
    )
    wind_base_energy = quadrature(source.wind_base_energy_prime for source in sources)
    wind_launch_energy = quadrature(source.wind_launch_power_prime for source in sources)
    wind_energy = float(wind_base_energy + wind_launch_energy)
    radiative_energy = quadrature(source.radiative_loss_prime for source in sources)
    external_angular = quadrature(
        source.angular_rhs
        - source.wind_prime * source.carried.l_wind
        + source.stream_prime * source.carried.l_stream
        for source in sources
    )
    external_energy = quadrature(
        source.energy_rhs
        - source.radiative_loss_prime
        - source.wind_base_energy_prime
        - source.wind_launch_power_prime
        + source.stream_prime * source.carried.B_stream
        for source in sources
    )
    values = (
        stream_angular,
        stream_energy,
        wind_mass,
        wind_angular,
        wind_base_energy,
        wind_launch_energy,
        wind_energy,
        radiative_energy,
        external_angular,
        external_energy,
    )
    if any(not np.isfinite(value) for value in values):
        raise ValueError("integrated transport moments must be finite")
    return ConservativeIntervalTransport(
        stream_mass=stream_mass,
        stream_mass_quadrature=stream_mass_quadrature,
        wind_mass=wind_mass,
        stream_angular_momentum=stream_angular,
        wind_angular_momentum=wind_angular,
        external_angular_momentum=external_angular,
        stream_energy=stream_energy,
        wind_base_energy=wind_base_energy,
        wind_launch_energy=wind_launch_energy,
        wind_energy=wind_energy,
        radiative_energy=radiative_energy,
        external_energy=external_energy,
    )


def wind_escape_diagnostics(
    state: ConservativeNodeState,
    closure: PhysicalTransportClosure,
    params,
    *,
    target_terminal_bernoulli: float = 0.0,
) -> WindEscapeDiagnostics:
    """Compare the prescribed launch energy with an unbound-wind target."""

    if not np.isfinite(target_terminal_bernoulli):
        raise ValueError("target_terminal_bernoulli must be finite")
    carried = carried_transport(state, closure, params)
    required = float(
        target_terminal_bernoulli - state.bernoulli - carried.wind_torque_work
    )
    margin = float(carried.B_wind - target_terminal_bernoulli)
    terminal_speed = float(math.sqrt(2.0 * max(carried.B_wind, 0.0)))
    return WindEscapeDiagnostics(
        disk_bernoulli=float(state.bernoulli),
        target_terminal_bernoulli=float(target_terminal_bernoulli),
        required_launch_energy=required,
        prescribed_launch_energy=float(carried.wind_launch_energy),
        wind_bernoulli=float(carried.B_wind),
        terminal_margin=margin,
        terminal_speed=terminal_speed,
        escaping=bool(margin >= 0.0),
    )


def integrate_interval_transport(
    dx: float,
    source_left: ConservativeSourceTerms,
    source_midpoint: ConservativeSourceTerms,
    source_right: ConservativeSourceTerms,
    *,
    exact_stream_mass: float | None = None,
    exact_stream_angular_momentum: float | None = None,
    exact_stream_energy: float | None = None,
) -> ConservativeIntervalTransport:
    """Integrate one production interval with the shared Simpson operator."""

    return integrate_sampled_interval_transport(
        dx,
        (source_left, source_midpoint, source_right),
        (1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0),
        exact_stream_mass=exact_stream_mass,
        exact_stream_angular_momentum=exact_stream_angular_momentum,
        exact_stream_energy=exact_stream_energy,
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
    exact_stream_mass: float | None = None,
    exact_stream_angular_momentum: float | None = None,
    exact_stream_energy: float | None = None,
) -> ConservativeIntervalResidual:
    """Return finite-volume rows from the shared production integrator."""

    transport = integrate_interval_transport(
        dx,
        source_left,
        source_midpoint,
        source_right,
        exact_stream_mass=exact_stream_mass,
        exact_stream_angular_momentum=exact_stream_angular_momentum,
        exact_stream_energy=exact_stream_energy,
    )
    E_left = left.mechanical_energy_flux if energy_flux_left is None else float(energy_flux_left)
    E_right = right.mechanical_energy_flux if energy_flux_right is None else float(energy_flux_right)
    return ConservativeIntervalResidual(
        mass=float((right.mdot - left.mdot - transport.mass_rhs) / scales.mdot),
        angular_momentum=float((right.J - left.J - transport.angular_rhs) / scales.angular_flux),
        energy=float((E_right - E_left - transport.energy_rhs) / scales.energy_flux),
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
