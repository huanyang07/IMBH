"""Audit Kerr-Schild geometric sources and finite-volume measures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from imri_qpe.constants import C, G
from imri_qpe.layer3_minidisk_1d import (
    FixedHeightGasRadiationColumnEOS,
    ValenciaPerfectFluidPrimitive,
    audit_kerr_schild_column_sources,
    audit_stationary_kerr_schild_finite_volume_profile,
    kerr_schild_column_geometry,
    make_kerr_schild_column_grid,
)
from imri_qpe.parameters import FiducialParams


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/tables/causal_inner_geometry_wp10c2.json"
RADII_RG = (20.0, 4.5, 2.0, 1.8)
THERMODYNAMIC_STATES = (
    (1.0e7, 1.0e7),
    (1.0e5, 3.0e7),
    (1.0e3, 3.0e8),
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _kinematics(radius_rg: float) -> tuple[float, float]:
    if radius_rg >= 10.0:
        return -0.01, 0.20
    if radius_rg >= 2.0:
        return -0.20, 0.55
    return -0.40, 0.60


def _source_identity_rows(
    gravitational_radius: float,
) -> list[dict]:
    eos = FixedHeightGasRadiationColumnEOS(
        proper_half_thickness=1.0e7
    )
    rows = []
    for radius_rg in RADII_RG:
        geometry = kerr_schild_column_geometry(
            radius_rg * gravitational_radius,
            gravitational_radius,
        )
        radial_velocity, azimuthal_velocity = _kinematics(radius_rg)
        for surface_density, temperature in THERMODYNAMIC_STATES:
            thermodynamics = eos.from_surface_density_temperature(
                surface_density,
                temperature,
            )
            primitive = ValenciaPerfectFluidPrimitive(
                surface_density=surface_density,
                radial_velocity_over_c=radial_velocity,
                azimuthal_velocity_over_c=azimuthal_velocity,
                specific_internal_energy=(
                    thermodynamics.specific_internal_energy
                ),
                integrated_pressure=thermodynamics.integrated_pressure,
            )
            audit = audit_kerr_schild_column_sources(
                geometry,
                primitive,
            )
            source_scale = max(
                abs(audit.radial_momentum_source),
                abs(audit.tau_source),
                1.0 / gravitational_radius,
            )
            state_scale = max(
                np.max(np.abs(audit.killing_conserved)),
                1.0,
            )
            flux_scale = max(
                np.max(np.abs(audit.killing_flux_over_c)),
                1.0,
            )
            rows.append(
                {
                    "radius_rg": radius_rg,
                    "surface_density": surface_density,
                    "temperature": temperature,
                    "momentum_source_relative_identity_defect": (
                        abs(audit.momentum_source_identity_defect)
                        / source_scale
                    ),
                    "tau_source_relative_identity_defect": (
                        abs(audit.tau_source_identity_defect)
                        / source_scale
                    ),
                    "killing_density_relative_identity_defect": (
                        abs(audit.killing_density_identity_defect)
                        / state_scale
                    ),
                    "killing_flux_relative_identity_defect": (
                        abs(audit.killing_flux_identity_defect)
                        / flux_scale
                    ),
                }
            )
    return rows


def _circular_dust_primitive(
    radius: float,
    gravitational_radius: float,
) -> ValenciaPerfectFluidPrimitive:
    geometry = kerr_schild_column_geometry(
        radius,
        gravitational_radius,
    )
    radial_velocity = 2.0 * gravitational_radius / radius
    azimuthal_velocity = (
        np.sqrt(gravitational_radius / radius) / geometry.base.lapse
    )
    return ValenciaPerfectFluidPrimitive(
        surface_density=1.0,
        radial_velocity_over_c=float(radial_velocity),
        azimuthal_velocity_over_c=float(azimuthal_velocity),
        specific_internal_energy=0.0,
        integrated_pressure=0.0,
    )


def _circular_rows(gravitational_radius: float) -> list[dict]:
    rows = []
    for radius_rg in (6.1, 10.0, 20.0):
        radius = radius_rg * gravitational_radius
        geometry = kerr_schild_column_geometry(
            radius,
            gravitational_radius,
        )
        audit = audit_kerr_schild_column_sources(
            geometry,
            _circular_dust_primitive(radius, gravitational_radius),
        )
        rows.append(
            {
                "radius_rg": radius_rg,
                "radial_momentum_source_times_rg": (
                    audit.radial_momentum_source * gravitational_radius
                ),
                "transport_velocity_over_c": (
                    audit.valencia_state.transport_velocity_over_c
                ),
                "local_mass_flux_over_c": audit.killing_flux_over_c[0],
                "local_killing_energy_flux_over_c": (
                    audit.killing_flux_over_c[3]
                ),
            }
        )
    return rows


def _radial_dust_primitive(
    radius: float,
    gravitational_radius: float,
) -> ValenciaPerfectFluidPrimitive:
    geometry = kerr_schild_column_geometry(
        radius,
        gravitational_radius,
    )
    free_fall_speed = np.sqrt(2.0 * gravitational_radius / radius)
    coordinate_time_velocity = (
        1.0 + free_fall_speed + free_fall_speed**2
    ) / (1.0 + free_fall_speed)
    lorentz_factor = geometry.base.lapse * coordinate_time_velocity
    coordinate_velocity = (
        -free_fall_speed / lorentz_factor
        + geometry.base.radial_shift_over_c / geometry.base.lapse
    )
    radial_velocity = (
        np.sqrt(geometry.base.gamma_rr) * coordinate_velocity
    )
    return ValenciaPerfectFluidPrimitive(
        surface_density=float(
            np.sqrt(10.0 * gravitational_radius / radius)
        ),
        radial_velocity_over_c=float(radial_velocity),
        azimuthal_velocity_over_c=0.0,
        specific_internal_energy=0.0,
        integrated_pressure=0.0,
    )


def _radial_free_fall_rows(
    gravitational_radius: float,
) -> tuple[list[dict], dict]:
    rows = []
    for n_cells in (16, 32, 64, 128):
        grid = make_kerr_schild_column_grid(
            1.5 * gravitational_radius,
            20.0 * gravitational_radius,
            n_cells,
            gravitational_radius,
        )
        audit = audit_stationary_kerr_schild_finite_volume_profile(
            grid,
            lambda radius: _radial_dust_primitive(
                radius,
                gravitational_radius,
            ),
            quadrature_order=1,
        )
        momentum_flux_differences = np.diff(
            audit.weighted_face_fluxes_over_c[:, 1]
        )
        momentum_scale = np.max(np.abs(momentum_flux_differences))
        rows.append(
            {
                "n_cells": n_cells,
                "maximum_normalized_momentum_residual": (
                    np.max(np.abs(audit.integrated_residuals[:, 1]))
                    / momentum_scale
                ),
            }
        )

    grid = make_kerr_schild_column_grid(
        1.5 * gravitational_radius,
        20.0 * gravitational_radius,
        128,
        gravitational_radius,
    )
    high_order = audit_stationary_kerr_schild_finite_volume_profile(
        grid,
        lambda radius: _radial_dust_primitive(
            radius,
            gravitational_radius,
        ),
        quadrature_order=8,
    )
    fluxes = high_order.weighted_face_fluxes_over_c
    mass_scale = abs(fluxes[0, 0])
    momentum_scale = np.max(np.abs(np.diff(fluxes[:, 1])))
    summary = {
        "n_cells": 128,
        "quadrature_order": 8,
        "relative_mass_flux_spread": (
            np.ptp(fluxes[:, 0]) / mass_scale
        ),
        "relative_killing_energy_flux_spread": (
            np.ptp(fluxes[:, 3]) / mass_scale
        ),
        "maximum_normalized_momentum_residual": (
            np.max(np.abs(high_order.integrated_residuals[:, 1]))
            / momentum_scale
        ),
        "maximum_relative_telescoping_defect": (
            np.max(np.abs(high_order.telescoping_defect)) / mass_scale
        ),
    }
    return rows, summary


def _flat_pressure_audit() -> dict:
    grid = make_kerr_schild_column_grid(2.0, 20.0, 32, 0.0)
    pressure = 0.04 * C**2

    def primitive(radius: float) -> ValenciaPerfectFluidPrimitive:
        del radius
        return ValenciaPerfectFluidPrimitive(
            surface_density=1.0,
            radial_velocity_over_c=0.0,
            azimuthal_velocity_over_c=0.0,
            specific_internal_energy=0.0,
            integrated_pressure=pressure,
        )

    audit = audit_stationary_kerr_schild_finite_volume_profile(
        grid,
        primitive,
        quadrature_order=2,
    )
    source_scale = np.max(
        np.abs(audit.integrated_geometric_sources[:, 1])
    )
    return {
        "n_cells": 32,
        "maximum_normalized_radial_residual": (
            np.max(np.abs(audit.integrated_residuals[:, 1]))
            / source_scale
        ),
        "maximum_absolute_telescoping_defect": float(
            np.max(np.abs(audit.telescoping_defect))
        ),
    }


def main() -> None:
    arguments = _arguments()
    gravitational_radius = G * FiducialParams().M2_g / C**2
    source_rows = _source_identity_rows(gravitational_radius)
    free_fall_rows, free_fall_summary = _radial_free_fall_rows(
        gravitational_radius
    )
    convergence_orders = [
        np.log2(
            free_fall_rows[index][
                "maximum_normalized_momentum_residual"
            ]
            / free_fall_rows[index + 1][
                "maximum_normalized_momentum_residual"
            ]
        )
        for index in range(len(free_fall_rows) - 1)
    ]
    maximum_source_identity_defect = max(
        max(
            row["momentum_source_relative_identity_defect"],
            row["tau_source_relative_identity_defect"],
        )
        for row in source_rows
    )
    maximum_killing_identity_defect = max(
        max(
            row["killing_density_relative_identity_defect"],
            row["killing_flux_relative_identity_defect"],
        )
        for row in source_rows
    )
    output = {
        "selected_column_reduction": (
            "equatorial 2+1 stationary Kerr-Schild column"
        ),
        "evolved_conserved_chart": ["D", "S_R", "S_phi", "E_K"],
        "killing_energy_definition": (
            "E_K = alpha (tau + D) - beta^R S_R"
        ),
        "geometric_source_contract": {
            "mass": "zero",
            "radial_momentum": "alpha T^munu d_R g_munu / 2",
            "angular_momentum": "zero",
            "killing_energy": "zero",
        },
        "source_identity_rows": source_rows,
        "maximum_source_identity_defect": (
            maximum_source_identity_defect
        ),
        "maximum_killing_identity_defect": (
            maximum_killing_identity_defect
        ),
        "flat_constant_pressure": _flat_pressure_audit(),
        "circular_geodesics": _circular_rows(gravitational_radius),
        "radial_free_fall_midpoint": free_fall_rows,
        "radial_free_fall_observed_orders": convergence_orders,
        "radial_free_fall_high_order": free_fall_summary,
        "passed": bool(
            maximum_source_identity_defect < 1.0e-13
            and maximum_killing_identity_defect < 1.0e-13
            and min(convergence_orders) > 1.8
            and free_fall_summary[
                "maximum_normalized_momentum_residual"
            ]
            < 1.0e-11
            and free_fall_summary["relative_mass_flux_spread"]
            < 1.0e-12
            and free_fall_summary[
                "relative_killing_energy_flux_spread"
            ]
            < 1.0e-12
        ),
        "production_ready": False,
        "blocking_work": [
            "relativistic stress and torque-work closure",
            "radiation and vertical-work source contract",
            "stream and Hill/Roche boundary migration",
            "stationary root and implicit timestep certification",
        ],
    }
    destination = _absolute(arguments.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )
    print(destination)


if __name__ == "__main__":
    main()
