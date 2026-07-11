from __future__ import annotations

import numpy as np
import pytest

from imri_qpe.layer3_minidisk_1d import (
    enthalpy_vertical_work,
    internal_energy_vertical_work,
    make_log_grid,
    signed_vertical_work_rate_cells,
)


def test_enthalpy_work_matches_source_free_entropy_identity() -> None:
    mdot = 3.0
    sigma = 5.0
    pi = 35.0
    dsigma = 0.4
    pressure = 11.0
    rho = 2.0
    drho = -0.3
    omega = 0.7
    domega = -0.08
    torque = 13.0
    dl = 0.2
    de = 1.1
    dpi = 2.3
    dh = dpi / sigma - pi * dsigma / sigma**2
    dq = omega * dl - dpi / sigma
    dtorque = mdot * dl
    dflux = mdot * (dq + de + dh) - domega * torque - omega * dtorque
    work = enthalpy_vertical_work(
        mdot, sigma, pi, dsigma, pressure, rho, drho
    )
    entropy_derivative = mdot * (de - pressure * drho / rho**2) - torque * domega

    assert dflux + work == pytest.approx(entropy_derivative)


def test_source_bearing_internal_and_enthalpy_forms_need_mass_source_term() -> None:
    mdot = 3.0
    dmdot = -0.6
    sigma = 5.0
    pi = 35.0
    dsigma = 0.4
    pressure = 11.0
    rho = 2.0
    drho = -0.3
    dpi = 2.3
    enthalpy = pi / sigma
    denthalpy = dpi / sigma - pi * dsigma / sigma**2
    dflux_internal = 17.0
    dflux_enthalpy = dflux_internal + enthalpy * dmdot + mdot * denthalpy
    work_enthalpy = enthalpy_vertical_work(
        mdot, sigma, pi, dsigma, pressure, rho, drho
    )
    work_internal = internal_energy_vertical_work(
        mdot, sigma, dpi, pressure, rho, drho
    )
    source_energy = 9.0

    enthalpy_residual = dflux_enthalpy + work_enthalpy + source_energy
    internal_residual = (
        dflux_internal
        + work_internal
        + source_energy
        + enthalpy * dmdot
    )

    assert enthalpy_residual == pytest.approx(internal_residual)


def test_source_bearing_manufactured_ledgers_converge_together() -> None:
    errors = []
    for n in (64, 128, 256, 512):
        grid = make_log_grid(1.0, np.e, n)
        x = np.log(grid.centers)
        xe = np.log(grid.edges)
        mdot = 3.0 - 0.4 * x
        mdot_edges = 3.0 - 0.4 * xe
        rho = np.exp(0.2 * x)
        sigma = np.exp(0.3 * x)
        enthalpy = np.exp(0.4 * x)
        pressure = rho * enthalpy
        integrated_pressure = sigma * enthalpy
        vertical = signed_vertical_work_rate_cells(
            grid,
            mdot,
            sigma,
            pressure,
            integrated_pressure,
            rho,
        )
        vertical_primitive = np.exp(0.4 * xe) * (1.0 - 0.1 * xe)
        vertical_exact = np.diff(vertical_primitive)

        specific_l_edges = 2.0 + 0.3 * xe
        torque_edges = 0.5 + 0.1 * xe
        angular_flux_edges = mdot_edges * specific_l_edges - torque_edges
        omega_edges = specific_l_edges / np.exp(2.0 * xe)
        base_specific_energy_edges = 1.0 + 0.2 * xe
        energy_flux_edges = (
            mdot_edges
            * (base_specific_energy_edges + np.exp(0.4 * xe))
            - omega_edges * torque_edges
        )

        source_mass = mdot_edges[:-1] - mdot_edges[1:]
        source_angular = angular_flux_edges[:-1] - angular_flux_edges[1:]
        source_energy = (
            energy_flux_edges[:-1]
            - energy_flux_edges[1:]
            - vertical_exact
        )
        mass_residual = np.diff(mdot_edges) + source_mass
        angular_residual = np.diff(angular_flux_edges) + source_angular
        energy_residual = np.diff(energy_flux_edges) + vertical + source_energy
        scale = np.maximum(
            np.abs(np.diff(energy_flux_edges))
            + np.abs(vertical_exact)
            + np.abs(source_energy),
            1.0,
        )

        assert np.max(np.abs(mass_residual)) < 1.0e-14
        assert np.max(np.abs(angular_residual)) < 1.0e-14
        errors.append(float(np.max(np.abs(energy_residual) / scale)))

    assert all(fine < 0.3 * coarse for coarse, fine in zip(errors, errors[1:]))
