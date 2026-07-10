"""Conservative angular-momentum bookkeeping for stream-fed wind disks.

The inward-positive mass convention is

``dMdot/dlnR = Mdot_wind_prime - Mdot_stream_prime``.

For net inward angular-momentum flux ``J = Mdot*l - G``, where
``G = 2*pi*R**2*W`` is the outward viscous torque, the matching convention is

``dJ/dlnR = Mdot_wind_prime*l_w - Mdot_stream_prime*l_s + tau_ext``.

Here ``tau_ext`` is defined as a positive contribution to ``dJ/dlnR``.  Mass
carried by a source or sink and an external torque are separate ledger entries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .transonic_local import stream_torque_specific_l_and_derivative
from .transonic_potential import PaczynskiWiitaPotential


@dataclass(frozen=True)
class AngularMomentumLedger:
    """One-point conservative angular-momentum ledger."""

    angular_flux: float
    angular_flux_prime: float
    wind_carried: float
    stream_carried: float
    external_torque: float
    physical_rhs: float
    residual: float
    flux_specific_l: float
    l_stream: float
    l_wind: float


def angular_flux(mdot: float, specific_l: float, viscous_torque: float) -> float:
    """Return net inward angular-momentum flux ``Mdot*l-G``."""

    return float(mdot * specific_l - viscous_torque)


def angular_flux_prime(
    mdot: float,
    mdot_prime: float,
    flux_specific_l: float,
    flux_specific_l_prime: float,
) -> float:
    """Differentiate ``J=Mdot*flux_specific_l`` with respect to ``lnR``."""

    return float(mdot_prime * flux_specific_l + mdot * flux_specific_l_prime)


def evaluate_angular_momentum_ledger(
    *,
    mdot: float,
    mdot_prime: float,
    specific_l: float,
    viscous_torque: float,
    flux_specific_l_prime: float,
    wind_prime: float,
    stream_prime: float,
    l_wind: float,
    l_stream: float,
    external_torque: float,
) -> AngularMomentumLedger:
    """Evaluate the flux derivative and independently specified source terms."""

    if mdot <= 0.0 or not np.isfinite(mdot):
        raise ValueError("mdot must be positive and finite")
    flux = angular_flux(mdot, specific_l, viscous_torque)
    flux_specific = float(flux / mdot)
    flux_prime = angular_flux_prime(
        mdot, mdot_prime, flux_specific, flux_specific_l_prime
    )
    wind_carried = float(wind_prime * l_wind)
    stream_carried = float(stream_prime * l_stream)
    physical_rhs = float(wind_carried - stream_carried + external_torque)
    return AngularMomentumLedger(
        angular_flux=flux,
        angular_flux_prime=flux_prime,
        wind_carried=wind_carried,
        stream_carried=stream_carried,
        external_torque=float(external_torque),
        physical_rhs=physical_rhs,
        residual=float(flux_prime - physical_rhs),
        flux_specific_l=flux_specific,
        l_stream=float(l_stream),
        l_wind=float(l_wind),
    )


def algebraic_flux_ledger(
    logR: float,
    state,
    params,
    *,
    mdot: float,
    mdot_prime: float,
    wind_prime: float,
    stream_prime: float,
    closure: str,
) -> AngularMomentumLedger:
    """Evaluate named closures for the current algebraic transonic state.

    ``representation`` is the exact interpretation of the existing algebraic
    ``stream_l`` offset.  Other closures expose the torque required to retain
    that same state when source/wind material is assigned a different angular
    momentum.
    """

    viscous_torque = float(2.0 * np.pi * state.R**2 * state.W)
    _stream_offset, stream_offset_prime = stream_torque_specific_l_and_derivative(
        float(logR), params
    )
    flux_specific = float(state.l - viscous_torque / mdot)
    prescribed_torque = float(mdot * stream_offset_prime)
    name = str(closure).strip().lower()

    if name == "representation":
        l_stream = flux_specific
        l_wind = flux_specific
        external_torque = prescribed_torque
    elif name == "local_disk_prescribed":
        l_stream = float(state.l)
        l_wind = float(state.l)
        external_torque = prescribed_torque
    elif name == "keplerian_local_prescribed":
        l_stream = float(state.l_K)
        l_wind = float(state.l_K)
        external_torque = prescribed_torque
    elif name in {"keplerian_injection_prescribed", "keplerian_injection_required"}:
        potential = PaczynskiWiitaPotential(params.M2_g)
        center_fraction = float(getattr(params, "stream_source_center_fraction", 0.8))
        l_stream = float(potential.l_k(center_fraction * params.R_out))
        l_wind = float(state.l)
        external_torque = prescribed_torque
    elif name == "local_disk_required":
        l_stream = float(state.l)
        l_wind = float(state.l)
        external_torque = prescribed_torque
    else:
        raise ValueError(f"unknown angular-momentum closure: {closure}")

    trial = evaluate_angular_momentum_ledger(
        mdot=mdot,
        mdot_prime=mdot_prime,
        specific_l=float(state.l),
        viscous_torque=viscous_torque,
        flux_specific_l_prime=float(stream_offset_prime),
        wind_prime=wind_prime,
        stream_prime=stream_prime,
        l_wind=l_wind,
        l_stream=l_stream,
        external_torque=external_torque,
    )
    if name not in {"local_disk_required", "keplerian_injection_required"}:
        return trial

    required_torque = float(
        trial.angular_flux_prime - trial.wind_carried + trial.stream_carried
    )
    return evaluate_angular_momentum_ledger(
        mdot=mdot,
        mdot_prime=mdot_prime,
        specific_l=float(state.l),
        viscous_torque=viscous_torque,
        flux_specific_l_prime=float(stream_offset_prime),
        wind_prime=wind_prime,
        stream_prime=stream_prime,
        l_wind=l_wind,
        l_stream=l_stream,
        external_torque=required_torque,
    )
