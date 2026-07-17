"""Audit the causal alpha-shear and paired Killing-flux prototype."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from imri_qpe.constants import C, G
from imri_qpe.layer3_minidisk_1d import (
    FixedHeightGasRadiationColumnEOS,
    ValenciaPerfectFluidPrimitive,
    audit_advected_stress_flux_eigensystem,
    audit_causal_stress_characteristics,
    calibrate_causal_alpha_shear,
    causal_stress_column_state,
    causal_stress_relaxation_source,
    causal_stress_torque_and_power,
    equilibrium_alpha_specific_stress,
    kerr_schild_column_geometry,
)
from imri_qpe.parameters import FiducialParams


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/tables/causal_inner_stress_wp10c3.json"
RADII_RG = (20.0, 4.5, 1.8)
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


def _primitive(
    eos: FixedHeightGasRadiationColumnEOS,
    surface_density: float,
    temperature: float,
    radial_velocity: float,
    azimuthal_velocity: float,
) -> tuple[ValenciaPerfectFluidPrimitive, float]:
    thermodynamics = eos.from_surface_density_temperature(
        surface_density,
        temperature,
    )
    primitive = ValenciaPerfectFluidPrimitive(
        surface_density=surface_density,
        radial_velocity_over_c=radial_velocity,
        azimuthal_velocity_over_c=azimuthal_velocity,
        specific_internal_energy=thermodynamics.specific_internal_energy,
        integrated_pressure=thermodynamics.integrated_pressure,
    )
    return primitive, float(thermodynamics.sound_speed / C)


def _closure(
    primitive: ValenciaPerfectFluidPrimitive,
    *,
    sound_speed_over_c: float,
    radius: float,
    mass: float,
):
    orbital_frequency = np.sqrt(G * mass / radius**3)
    return calibrate_causal_alpha_shear(
        primitive,
        alpha=0.1,
        reference_positive_shear_rate=1.5 * orbital_frequency,
        viscous_signal_speed_over_c=(
            np.sqrt(0.1) * sound_speed_over_c
        ),
    )


def _state_rows(
    eos: FixedHeightGasRadiationColumnEOS,
    mass: float,
    gravitational_radius: float,
) -> list[dict]:
    rows = []
    for radius_rg in RADII_RG:
        radius = radius_rg * gravitational_radius
        geometry = kerr_schild_column_geometry(
            radius,
            gravitational_radius,
        )
        radial_velocity, azimuthal_velocity = _kinematics(radius_rg)
        for surface_density, temperature in THERMODYNAMIC_STATES:
            primitive, sound_speed = _primitive(
                eos,
                surface_density,
                temperature,
                radial_velocity,
                azimuthal_velocity,
            )
            closure = _closure(
                primitive,
                sound_speed_over_c=sound_speed,
                radius=radius,
                mass=mass,
            )
            state = causal_stress_column_state(
                geometry,
                primitive,
                specific_stress=closure.equilibrium_specific_stress,
            )
            characteristics = audit_causal_stress_characteristics(
                geometry,
                eos,
                closure,
                surface_density=surface_density,
                radial_velocity_over_c=radial_velocity,
                azimuthal_velocity_over_c=azimuthal_velocity,
                temperature=temperature,
            )
            source = causal_stress_relaxation_source(
                geometry,
                state,
                closure,
                positive_shear_rate=(
                    closure.reference_positive_shear_rate
                ),
            )
            rows.append(
                {
                    "radius_rg": radius_rg,
                    "surface_density": surface_density,
                    "temperature": temperature,
                    "sound_speed_over_c": sound_speed,
                    "viscous_signal_speed_over_c": (
                        closure.viscous_signal_speed_over_c
                    ),
                    "relaxation_time_seconds": closure.relaxation_time,
                    "equilibrium_specific_stress": (
                        closure.equilibrium_specific_stress
                    ),
                    "stress_trace_relative_defect": (
                        state.tensor_trace_relative_defect
                    ),
                    "stress_orthogonality_relative_defect": (
                        state.tensor_orthogonality_relative_defect
                    ),
                    "stress_work_relative_defect": (
                        state.radial_work_relative_defect
                    ),
                    "equilibrium_relaxation_source": source,
                    "characteristic_speeds_over_c": (
                        characteristics.speeds_over_c
                    ),
                    "incoming_inner_characteristics": (
                        characteristics.incoming_inner_characteristics
                    ),
                    "stationary_flux_rank": (
                        characteristics.stationary_flux_rank
                    ),
                    "shear_principal_eigenvalue_defect": (
                        characteristics.shear_principal_eigenvalue_defect
                    ),
                    "maximum_imaginary_eigenvalue": (
                        characteristics.maximum_imaginary_eigenvalue
                    ),
                    "maximum_light_cone_excess": (
                        characteristics.maximum_light_cone_excess
                    ),
                    "causal_and_hyperbolic": (
                        characteristics.causal_and_hyperbolic
                    ),
                }
            )
    return rows


def _rejected_control_rows(
    eos: FixedHeightGasRadiationColumnEOS,
    mass: float,
    gravitational_radius: float,
) -> list[dict]:
    radius = 20.0 * gravitational_radius
    geometry = kerr_schild_column_geometry(
        radius,
        gravitational_radius,
    )
    primitive, sound_speed = _primitive(
        eos,
        1.0e7,
        1.0e7,
        -0.01,
        0.20,
    )
    closure = _closure(
        primitive,
        sound_speed_over_c=sound_speed,
        radius=radius,
        mass=mass,
    )
    rows = []
    for step in (1.0e-3, 2.0e-4, 1.0e-4, 5.0e-5):
        audit = audit_advected_stress_flux_eigensystem(
            geometry,
            eos,
            closure,
            surface_density=1.0e7,
            radial_velocity_over_c=-0.01,
            azimuthal_velocity_over_c=0.20,
            temperature=1.0e7,
            finite_difference_step=step,
        )
        rows.append(
            {
                "finite_difference_step": step,
                "eigenvalues": [
                    {"real": value.real, "imaginary": value.imag}
                    for value in audit.eigenvalues
                ],
                "maximum_imaginary_eigenvalue": (
                    audit.maximum_imaginary_eigenvalue
                ),
                "maximum_light_cone_excess": (
                    audit.maximum_light_cone_excess
                ),
                "hyperbolic": audit.hyperbolic,
            }
        )
    return rows


def _torque_rows(
    eos: FixedHeightGasRadiationColumnEOS,
    gravitational_radius: float,
) -> list[dict]:
    rows = []
    for radius_rg in (20.0, 100.0, 1000.0, 10000.0):
        radius = radius_rg * gravitational_radius
        geometry = kerr_schild_column_geometry(
            radius,
            gravitational_radius,
        )
        thermodynamics = eos.from_surface_density_temperature(
            1.0e5,
            3.0e7,
        )
        primitive = ValenciaPerfectFluidPrimitive(
            surface_density=1.0e5,
            radial_velocity_over_c=2.0 / radius_rg,
            azimuthal_velocity_over_c=(
                np.sqrt(gravitational_radius / radius)
                / geometry.base.lapse
            ),
            specific_internal_energy=(
                thermodynamics.specific_internal_energy
            ),
            integrated_pressure=thermodynamics.integrated_pressure,
        )
        state = causal_stress_column_state(
            geometry,
            primitive,
            specific_stress=equilibrium_alpha_specific_stress(
                primitive,
                alpha=0.1,
            ),
        )
        torque, power = causal_stress_torque_and_power(geometry, state)
        common_torque = (
            2.0
            * np.pi
            * radius**2
            * 0.1
            * thermodynamics.integrated_pressure
        )
        rows.append(
            {
                "radius_rg": radius_rg,
                "relativistic_torque": torque,
                "common_weak_field_torque": common_torque,
                "relative_common_torque_defect": abs(
                    torque / common_torque - 1.0
                ),
                "killing_power": power,
                "omega_times_torque": (
                    state.coordinate_angular_velocity * torque
                ),
                "relative_torque_work_defect": abs(
                    power
                    / (state.coordinate_angular_velocity * torque)
                    - 1.0
                ),
            }
        )
    return rows


def main() -> None:
    arguments = _arguments()
    output = _absolute(arguments.output)
    parameters = FiducialParams()
    mass = parameters.M2_g
    gravitational_radius = G * mass / C**2
    eos = FixedHeightGasRadiationColumnEOS(
        proper_half_thickness=1.0e7
    )
    states = _state_rows(eos, mass, gravitational_radius)
    rejected = _rejected_control_rows(
        eos,
        mass,
        gravitational_radius,
    )
    torque = _torque_rows(eos, gravitational_radius)
    summary = {
        "state_count": len(states),
        "all_causal_and_hyperbolic": all(
            row["causal_and_hyperbolic"] for row in states
        ),
        "maximum_tensor_identity_defect": max(
            max(
                row["stress_trace_relative_defect"],
                row["stress_orthogonality_relative_defect"],
                row["stress_work_relative_defect"],
            )
            for row in states
        ),
        "maximum_shear_eigenvalue_defect": max(
            row["shear_principal_eigenvalue_defect"]
            for row in states
        ),
        "maximum_light_cone_excess": max(
            row["maximum_light_cone_excess"] for row in states
        ),
        "inside_horizon_maximum_incoming_characteristics": max(
            row["incoming_inner_characteristics"]
            for row in states
            if row["radius_rg"] < 2.0
        ),
        "rejected_control_minimum_imaginary_eigenvalue": min(
            row["maximum_imaginary_eigenvalue"] for row in rejected
        ),
        "weak_field_relative_torque_defect": (
            torque[-1]["relative_common_torque_defect"]
        ),
        "maximum_relative_torque_work_defect": max(
            row["relative_torque_work_defect"] for row in torque
        ),
    }
    payload = {
        "work_package": "WP10c3a",
        "model": (
            "covariant_rphi_stress_with_maxwell_cattaneo_shear"
        ),
        "stress_sign": "positive stress transports angular momentum outward",
        "viscous_signal_choice": "c_nu=sqrt(alpha)*a, alpha=0.1",
        "states": states,
        "rejected_advected_pressure_stress_control": rejected,
        "weak_field_torque_and_work": torque,
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
