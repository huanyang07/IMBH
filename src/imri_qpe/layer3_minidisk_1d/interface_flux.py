"""Shared conserved-flux definitions for inner/outer disk interfaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .transonic_potential import PaczynskiWiitaPotential


@dataclass(frozen=True)
class ConservedInterfaceFlux:
    """Inward-positive mass, angular-momentum, and total-energy fluxes."""

    mdot: float
    angular_momentum: float
    total_energy: float

    def __post_init__(self) -> None:
        for name in ("mdot", "angular_momentum", "total_energy"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} flux must be finite")
            object.__setattr__(self, name, value)


def conserved_interface_flux(
    mdot: float,
    specific_angular_momentum: float,
    viscous_torque: float,
    omega: float,
    bernoulli: float,
) -> ConservedInterfaceFlux:
    """Construct ``(Mdot, J, F_E)`` using the canonical sign convention."""

    values = np.asarray(
        [mdot, specific_angular_momentum, viscous_torque, omega, bernoulli],
        dtype=float,
    )
    if np.any(~np.isfinite(values)):
        raise ValueError("interface primitive flux inputs must be finite")
    return ConservedInterfaceFlux(
        mdot=float(mdot),
        angular_momentum=float(mdot * specific_angular_momentum - viscous_torque),
        total_energy=float(mdot * bernoulli - omega * viscous_torque),
    )


def transonic_profile_interface_flux(
    profile,
    M_g: float,
    mdot: float,
    index: int,
) -> ConservedInterfaceFlux:
    """Extract conserved fluxes from a transonic profile node."""

    index = int(index)
    radius = float(profile.R[index])
    potential = PaczynskiWiitaPotential(float(M_g))
    torque = float(2.0 * np.pi * radius**2 * profile.W[index])
    bernoulli = float(
        0.5 * profile.u[index] ** 2
        + 0.5 * (radius * profile.Omega[index]) ** 2
        + potential.phi(radius)
        + profile.e[index]
        + profile.Pi[index] / profile.Sigma[index]
    )
    return conserved_interface_flux(
        mdot,
        float(profile.l[index]),
        torque,
        float(profile.Omega[index]),
        bernoulli,
    )


def signed_inner_interface_flux(transport, energy_profile) -> ConservedInterfaceFlux:
    """Extract the inner face fluxes from a signed total-energy state."""

    return ConservedInterfaceFlux(
        mdot=float(transport.mdot_faces[0]),
        angular_momentum=float(transport.angular_flux_faces[0]),
        total_energy=float(energy_profile.total_energy_flux_faces[0]),
    )
