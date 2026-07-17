"""Audit dynamic height, cooling, and stress work for WP10c3b."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad

from imri_qpe.constants import C, G
from imri_qpe.layer3_minidisk_1d import (
    QuasiHydrostaticGasRadiationColumnEOS,
    ValenciaPerfectFluidPrimitive,
    audit_causal_stress_characteristics,
    audit_quasi_hydrostatic_characteristics,
    calibrate_causal_alpha_shear,
    causal_diffusion_cooling_rate,
    causal_stress_work_partition,
    causal_thermal_column_source,
    hydrostatic_vertical_work_identity_defect,
    kerr_schild_column_geometry,
    make_kerr_schild_column_grid,
    recover_valencia_gas_radiation_primitives,
    temporal_vertical_work_per_area,
    valencia_gas_radiation_column_state,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import gas_constant_per_gram


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "outputs/tables/causal_inner_thermal_wp10c3b.json"
)
STATE_GRID = {
    20.0: (
        (2.0e5, 1.0e7),
        (1.0e7, 1.0e7),
        (1.0e9, 1.0e7),
    ),
    4.5: (
        (1.0e6, 3.0e7),
        (1.0e7, 3.0e7),
        (1.0e9, 3.0e7),
    ),
    1.8: (
        (1.0e9, 3.0e8),
        (1.0e10, 3.0e8),
        (1.0e12, 3.0e8),
    ),
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _eos(
    mass: float,
    radius: float,
) -> QuasiHydrostaticGasRadiationColumnEOS:
    return QuasiHydrostaticGasRadiationColumnEOS(
        proper_vertical_frequency=np.sqrt(G * mass / radius**3)
    )


def _kinematics(radius_rg: float) -> tuple[float, float]:
    if radius_rg >= 10.0:
        return -0.01, 0.20
    if radius_rg >= 2.0:
        return -0.20, 0.55
    return -0.40, 0.60


def _state_rows(
    mass: float,
    gravitational_radius: float,
) -> list[dict]:
    rows = []
    gas_constant = gas_constant_per_gram()
    for radius_rg, state_grid in STATE_GRID.items():
        radius = radius_rg * gravitational_radius
        geometry = kerr_schild_column_geometry(
            radius,
            gravitational_radius,
        )
        eos = _eos(mass, radius)
        radial_velocity, azimuthal_velocity = _kinematics(radius_rg)
        for surface_density, temperature in state_grid:
            state, thermodynamics = valencia_gas_radiation_column_state(
                geometry.base,
                eos,
                surface_density=surface_density,
                radial_velocity_over_c=radial_velocity,
                azimuthal_velocity_over_c=azimuthal_velocity,
                temperature=temperature,
            )
            recovered = recover_valencia_gas_radiation_primitives(
                geometry.base,
                eos,
                state.conserved,
            )
            derivatives = eos.derivatives(
                surface_density,
                temperature,
            )
            acoustic = audit_quasi_hydrostatic_characteristics(
                eos,
                surface_density=surface_density,
                temperature=temperature,
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
            orbital_frequency = np.sqrt(G * mass / radius**3)
            closure = calibrate_causal_alpha_shear(
                primitive,
                alpha=0.1,
                reference_positive_shear_rate=1.5 * orbital_frequency,
                viscous_signal_speed_over_c=(
                    np.sqrt(0.1) * thermodynamics.sound_speed / C
                ),
            )
            shear = audit_causal_stress_characteristics(
                geometry,
                eos,
                closure,
                surface_density=surface_density,
                radial_velocity_over_c=radial_velocity,
                azimuthal_velocity_over_c=azimuthal_velocity,
                temperature=temperature,
            )
            cooling, optical_depth = causal_diffusion_cooling_rate(
                thermodynamics
            )
            equilibrium_height_rate = (
                -cooling / thermodynamics.integrated_pressure
            )
            equilibrium_source = causal_thermal_column_source(
                geometry,
                eos,
                surface_density=surface_density,
                radial_velocity_over_c=radial_velocity,
                azimuthal_velocity_over_c=azimuthal_velocity,
                temperature=temperature,
                proper_log_height_rate=equilibrium_height_rate,
            )
            source_scale = max(
                float(
                    np.max(
                        np.abs(
                            equilibrium_source.cooling_source
                            .killing_source_per_ct
                        )
                    )
                ),
                np.finfo(float).tiny,
            )
            pressure = (
                thermodynamics.integrated_pressure
                / (2.0 * thermodynamics.proper_half_thickness)
            )
            gas_pressure = (
                thermodynamics.density * gas_constant * temperature
            )
            rows.append(
                {
                    "radius_rg": radius_rg,
                    "surface_density": surface_density,
                    "temperature": temperature,
                    "height_over_radius": (
                        thermodynamics.proper_half_thickness / radius
                    ),
                    "gas_pressure_fraction": gas_pressure / pressure,
                    "scattering_optical_depth": optical_depth,
                    "sound_speed_over_c": (
                        thermodynamics.sound_speed / C
                    ),
                    "height_log_surface_density": (
                        derivatives.height_log_surface_density
                    ),
                    "height_log_temperature": (
                        derivatives.height_log_temperature
                    ),
                    "height_log_vertical_frequency": (
                        derivatives.height_log_vertical_frequency
                    ),
                    "maximum_recovery_defect": (
                        recovered.maximum_relative_conserved_defect
                    ),
                    "acoustic_eigenvalue_defect": (
                        acoustic.maximum_eigenvalue_defect
                    ),
                    "acoustic_maximum_imaginary_eigenvalue": (
                        acoustic.maximum_imaginary_eigenvalue
                    ),
                    "shear_eigenvalue_defect": (
                        shear.shear_principal_eigenvalue_defect
                    ),
                    "shear_maximum_light_cone_excess": (
                        shear.maximum_light_cone_excess
                    ),
                    "incoming_inner_characteristics": (
                        shear.incoming_inner_characteristics
                    ),
                    "cooling_rate": cooling,
                    "equilibrium_log_height_rate": (
                        equilibrium_height_rate
                    ),
                    "equilibrium_killing_source_relative_defect": (
                        float(
                            np.max(
                                np.abs(
                                    equilibrium_source
                                    .total_killing_source_per_ct
                                )
                            )
                            / source_scale
                        )
                    ),
                    "cooling_comoving_identity_defect": (
                        equilibrium_source.cooling_source
                        .relative_identity_defect
                    ),
                    "cooling_comoving_momentum_defect": (
                        equilibrium_source.cooling_source
                        .comoving_momentum_relative_defect
                    ),
                    "vertical_work_identity_defect": (
                        hydrostatic_vertical_work_identity_defect(
                            thermodynamics,
                            surface_density_derivative=(
                                2.3e-4 * surface_density
                            ),
                            height_derivative=(
                                -1.7e-4
                                * thermodynamics.proper_half_thickness
                            ),
                        )
                    ),
                }
            )
    return rows


def _source_integral_rows(
    mass: float,
    gravitational_radius: float,
) -> list[dict]:
    left = 10.0 * gravitational_radius
    right = 30.0 * gravitational_radius
    reference_radius = 20.0 * gravitational_radius

    def weighted_source(radius: float) -> np.ndarray:
        ratio = radius / reference_radius
        geometry = kerr_schild_column_geometry(
            radius,
            gravitational_radius,
        )
        source = causal_thermal_column_source(
            geometry,
            _eos(mass, radius),
            surface_density=1.0e8 * ratio ** (-0.5),
            radial_velocity_over_c=-0.03,
            azimuthal_velocity_over_c=0.20,
            temperature=1.0e7 * ratio ** (-0.2),
            proper_log_height_rate=(
                2.0e-3 * np.sqrt(G * mass / radius**3)
            ),
        )
        return (
            geometry.face_measure
            * source.total_killing_source_per_ct
        )

    reference = np.asarray(
        [
            quad(
                lambda radius: weighted_source(radius)[component],
                left,
                right,
                epsabs=0.0,
                epsrel=2.0e-11,
                limit=200,
            )[0]
            for component in (1, 2, 3)
        ],
        dtype=float,
    )
    rows = []
    for cells in (16, 32, 64, 128):
        grid = make_kerr_schild_column_grid(
            left,
            right,
            cells,
            gravitational_radius,
        )
        midpoint = np.sum(
            [
                weighted_source(radius)[1:]
                / kerr_schild_column_geometry(
                    radius,
                    gravitational_radius,
                ).face_measure
                * measure
                for radius, measure in zip(
                    grid.centers,
                    grid.cell_measures,
                    strict=True,
                )
            ],
            axis=0,
        )
        component_errors = np.abs(midpoint / reference - 1.0)
        rows.append(
            {
                "cells": cells,
                "component_relative_errors": component_errors.tolist(),
                "maximum_component_relative_error": float(
                    np.max(component_errors)
                ),
            }
        )
    for previous, current in zip(rows[:-1], rows[1:], strict=True):
        current["observed_order_from_previous"] = float(
            np.log2(
                previous["maximum_component_relative_error"]
                / current["maximum_component_relative_error"]
            )
        )
    rows[0]["observed_order_from_previous"] = None
    return rows


def _work_identities(
    mass: float,
    gravitational_radius: float,
) -> dict:
    radius = 20.0 * gravitational_radius
    eos = _eos(mass, radius)
    old = eos.from_surface_density_temperature(1.0e7, 1.0e7)
    new = eos.from_surface_density_temperature(1.1e7, 1.2e7)
    forward = temporal_vertical_work_per_area(old, new)
    reverse = temporal_vertical_work_per_area(new, old)
    partition = causal_stress_work_partition(
        left_angular_velocity=4.0,
        right_angular_velocity=3.0,
        left_torque=7.0,
        right_torque=9.0,
    )
    return {
        "temporal_vertical_work_antisymmetry_defect": abs(
            forward + reverse
        )
        / max(abs(forward), abs(reverse), 1.0),
        "torque_work_product_rule_defect": (
            partition.product_rule_defect
        ),
        "explicit_total_energy_viscous_source": (
            partition.explicit_total_energy_heating_source
        ),
    }


def main() -> None:
    arguments = _arguments()
    output = _absolute(arguments.output)
    parameters = FiducialParams()
    mass = parameters.M2_g
    gravitational_radius = G * mass / C**2
    states = _state_rows(mass, gravitational_radius)
    source_rows = _source_integral_rows(mass, gravitational_radius)
    work = _work_identities(mass, gravitational_radius)
    summary = {
        "state_count": len(states),
        "maximum_height_over_radius": max(
            row["height_over_radius"] for row in states
        ),
        "maximum_recovery_defect": max(
            row["maximum_recovery_defect"] for row in states
        ),
        "maximum_acoustic_eigenvalue_defect": max(
            row["acoustic_eigenvalue_defect"] for row in states
        ),
        "maximum_shear_eigenvalue_defect": max(
            row["shear_eigenvalue_defect"] for row in states
        ),
        "maximum_light_cone_excess": max(
            row["shear_maximum_light_cone_excess"] for row in states
        ),
        "inside_horizon_maximum_incoming_characteristics": max(
            row["incoming_inner_characteristics"]
            for row in states
            if row["radius_rg"] < 2.0
        ),
        "maximum_comoving_source_identity_defect": max(
            max(
                row["cooling_comoving_identity_defect"],
                row["cooling_comoving_momentum_defect"],
            )
            for row in states
        ),
        "maximum_vertical_work_identity_defect": max(
            row["vertical_work_identity_defect"] for row in states
        ),
        "maximum_local_equilibrium_source_defect": max(
            row["equilibrium_killing_source_relative_defect"]
            for row in states
        ),
        "minimum_source_integral_order": min(
            row["observed_order_from_previous"]
            for row in source_rows[1:]
        ),
        "finest_source_integral_error": (
            source_rows[-1]["maximum_component_relative_error"]
        ),
        **work,
    }
    payload = {
        "work_package": "WP10c3b",
        "model": (
            "quasi_hydrostatic_gas_radiation_column_with_causal_sources"
        ),
        "vertical_frequency_scope": (
            "provisional supplied proper frequency; Newtonian orbital "
            "frequency used only for bounded audit states"
        ),
        "source_sign": "positive comoving rate adds energy to the gas",
        "stress_work_rule": (
            "torque work stays in tensor flux; no local total-energy "
            "viscous source"
        ),
        "states": states,
        "finite_volume_source_integration": source_rows,
        "work_identities": work,
        "summary": summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
