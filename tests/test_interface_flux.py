from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from imri_qpe.layer3_minidisk_1d import (
    ConservedInterfaceFlux,
    PaczynskiWiitaPotential,
    SignedFluxBoundary,
    conserved_interface_flux,
    make_log_grid,
    normalized_stream_injection_state,
    solve_signed_flux_steady,
    transonic_profile_interface_flux,
)
from imri_qpe.units import solar_masses_to_g


def _steady_case(outer_mode: str):
    mass = solar_masses_to_g(1.0e4)
    potential = PaczynskiWiitaPotential(mass)
    grid = make_log_grid(10.0 * potential.r_g, 335.0 * potential.r_g, 96)
    stream_rate = 5.0e22
    stream_l = float(potential.l_k(100.0 * potential.r_g))
    stream = normalized_stream_injection_state(
        grid,
        stream_rate,
        center=92.0 * potential.r_g,
        log_width=0.08,
        specific_angular_momentum=stream_l,
        specific_total_energy=-1.0e19,
    )
    result = solve_signed_flux_steady(
        grid,
        1.0e14,
        mass,
        boundary=SignedFluxBoundary(outer_mode=outer_mode),
        stream_state=stream,
    )
    return mass, grid, stream, result


def test_shared_flux_constructor_uses_canonical_signs() -> None:
    flux = conserved_interface_flux(
        mdot=3.0,
        specific_angular_momentum=5.0,
        viscous_torque=7.0,
        omega=0.2,
        bernoulli=-11.0,
    )

    assert flux.mdot == 3.0
    assert flux.angular_momentum == 8.0
    assert flux.total_energy == pytest.approx(-34.4)


def test_transonic_extractor_uses_same_flux_constructor() -> None:
    mass = solar_masses_to_g(1.0e4)
    potential = PaczynskiWiitaPotential(mass)
    radius = 20.0 * potential.r_g
    profile = SimpleNamespace(
        R=np.asarray([radius]),
        u=np.asarray([1.0e6]),
        Omega=np.asarray([0.8 * potential.omega_k(radius)]),
        l=np.asarray([0.8 * potential.l_k(radius)]),
        W=np.asarray([2.0e16]),
        e=np.asarray([3.0e17]),
        Pi=np.asarray([4.0e20]),
        Sigma=np.asarray([2.0e3]),
    )
    extracted = transonic_profile_interface_flux(profile, mass, 5.0e22, 0)
    torque = 2.0 * np.pi * radius**2 * profile.W[0]
    bernoulli = (
        0.5 * profile.u[0] ** 2
        + 0.5 * (radius * profile.Omega[0]) ** 2
        + potential.phi(radius)
        + profile.e[0]
        + profile.Pi[0] / profile.Sigma[0]
    )
    expected = conserved_interface_flux(
        5.0e22, profile.l[0], torque, profile.Omega[0], bernoulli
    )

    assert extracted == expected


@pytest.mark.parametrize("outer_mode", ["tidal_wall", "zero_torque"])
def test_prescribed_inner_flux_round_trips_steady_transport(outer_mode: str) -> None:
    mass, grid, stream, baseline = _steady_case(outer_mode)
    inner = ConservedInterfaceFlux(
        mdot=baseline.mdot_faces[0],
        angular_momentum=baseline.angular_flux_faces[0],
        total_energy=0.0,
    )
    prescribed = solve_signed_flux_steady(
        grid,
        baseline.viscosity,
        mass,
        boundary=SignedFluxBoundary(
            inner_mode="prescribed_flux", outer_mode=outer_mode
        ),
        stream_state=stream,
        prescribed_inner_flux=inner,
    )

    assert np.allclose(prescribed.mdot_faces, baseline.mdot_faces)
    assert np.allclose(prescribed.angular_flux_faces, baseline.angular_flux_faces)
    assert np.allclose(
        prescribed.viscous_torque_faces, baseline.viscous_torque_faces
    )
    assert np.allclose(prescribed.surface_density, baseline.surface_density)


def test_prescribed_inner_mass_rejects_incompatible_outer_wall() -> None:
    mass, grid, stream, baseline = _steady_case("tidal_wall")
    incompatible = ConservedInterfaceFlux(
        mdot=0.9 * baseline.mdot_faces[0],
        angular_momentum=baseline.angular_flux_faces[0],
        total_energy=0.0,
    )

    with pytest.raises(ValueError, match="incompatible with the outer tidal wall"):
        solve_signed_flux_steady(
            grid,
            baseline.viscosity,
            mass,
            boundary=SignedFluxBoundary(
                inner_mode="prescribed_flux", outer_mode="tidal_wall"
            ),
            stream_state=stream,
            prescribed_inner_flux=incompatible,
        )
