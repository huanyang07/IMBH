"""Audit gas+radiation primitive recovery for the Valencia column chart."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from imri_qpe.constants import C
from imri_qpe.layer3_minidisk_1d import (
    FixedHeightGasRadiationColumnEOS,
    audit_gas_radiation_valencia_eigensystem,
    recover_valencia_gas_radiation_primitives,
    schwarzschild_kerr_schild_geometry,
    valencia_gas_radiation_column_state,
)
from imri_qpe.parameters import FiducialParams


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "outputs/tables/causal_inner_primitive_recovery_wp10c1.json"
)
THERMODYNAMIC_STATES = (
    ("gas_dominated", 1.0e7, 1.0e7),
    ("gas_radiation_transition", 1.0e5, 3.0e7),
    ("radiation_dominated", 1.0e3, 3.0e8),
)
KINEMATIC_STATES = (
    ("weak_field", 20.0, -0.01, 0.20),
    ("inner_rotating", 4.5, -0.20, 0.55),
    ("inside_horizon", 1.8, -0.40, 0.60),
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _relative_defect(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1.0e-30)


def main() -> None:
    arguments = _arguments()
    parameters = FiducialParams()
    reference_geometry = schwarzschild_kerr_schild_geometry(
        1.0,
        parameters.M2_g,
    )
    gravitational_radius = reference_geometry.gravitational_radius
    eos = FixedHeightGasRadiationColumnEOS(
        proper_half_thickness=1.0e7,
    )

    rows = []
    for (
        kinematic_name,
        radius_rg,
        radial_velocity,
        azimuthal_velocity,
    ) in KINEMATIC_STATES:
        geometry = schwarzschild_kerr_schild_geometry(
            radius_rg * gravitational_radius,
            parameters.M2_g,
        )
        for (
            thermodynamic_name,
            surface_density,
            temperature,
        ) in THERMODYNAMIC_STATES:
            state, thermodynamics = valencia_gas_radiation_column_state(
                geometry,
                eos,
                surface_density=surface_density,
                radial_velocity_over_c=radial_velocity,
                azimuthal_velocity_over_c=azimuthal_velocity,
                temperature=temperature,
            )
            recovery = recover_valencia_gas_radiation_primitives(
                geometry,
                eos,
                state.conserved,
            )
            characteristics = audit_gas_radiation_valencia_eigensystem(
                geometry,
                eos,
                surface_density=surface_density,
                radial_velocity_over_c=radial_velocity,
                azimuthal_velocity_over_c=azimuthal_velocity,
                temperature=temperature,
            )
            primitive = recovery.primitive
            primitive_defects = {
                "surface_density": _relative_defect(
                    primitive.surface_density,
                    surface_density,
                ),
                "radial_velocity": _relative_defect(
                    primitive.radial_velocity_over_c,
                    radial_velocity,
                ),
                "azimuthal_velocity": _relative_defect(
                    primitive.azimuthal_velocity_over_c,
                    azimuthal_velocity,
                ),
                "temperature": _relative_defect(
                    primitive.temperature,
                    temperature,
                ),
            }
            rows.append(
                {
                    "kinematic_regime": kinematic_name,
                    "thermodynamic_regime": thermodynamic_name,
                    "radius_rg": radius_rg,
                    "surface_density_g_cm2": surface_density,
                    "temperature_K": temperature,
                    "radial_velocity_over_c": radial_velocity,
                    "azimuthal_velocity_over_c": azimuthal_velocity,
                    "sound_speed_over_c": thermodynamics.sound_speed / C,
                    "primitive_relative_defects": primitive_defects,
                    "maximum_relative_primitive_defect": max(
                        primitive_defects.values()
                    ),
                    "maximum_relative_conserved_defect": (
                        recovery.maximum_relative_conserved_defect
                    ),
                    "pressure_root_iterations": (
                        recovery.pressure_root_iterations
                    ),
                    "pressure_residual_evaluations": (
                        recovery.pressure_root_function_calls
                    ),
                    "analytic_speeds_over_c": list(
                        characteristics.analytic_speeds_over_c
                    ),
                    "numerical_speeds_over_c": list(
                        characteristics.numerical_speeds_over_c
                    ),
                    "maximum_eigenvalue_defect": (
                        characteristics.maximum_eigenvalue_defect
                    ),
                    "incoming_inner_characteristics": (
                        characteristics.incoming_inner_characteristics
                    ),
                }
            )

    maximum_primitive_defect = max(
        row["maximum_relative_primitive_defect"] for row in rows
    )
    maximum_conserved_defect = max(
        row["maximum_relative_conserved_defect"] for row in rows
    )
    maximum_eigenvalue_defect = max(
        row["maximum_eigenvalue_defect"] for row in rows
    )
    maximum_sound_speed = max(row["sound_speed_over_c"] for row in rows)
    inside_horizon_rows = [
        row for row in rows if row["radius_rg"] < 2.0
    ]

    invalid_state_rejected = False
    try:
        recover_valencia_gas_radiation_primitives(
            schwarzschild_kerr_schild_geometry(
                4.5 * gravitational_radius,
                parameters.M2_g,
            ),
            eos,
            np.asarray([1.0, 100.0, 0.0, 0.0]),
        )
    except ValueError:
        invalid_state_rejected = True

    passed = bool(
        maximum_primitive_defect <= 1.0e-10
        and maximum_conserved_defect <= 1.0e-10
        and maximum_eigenvalue_defect <= 1.0e-7
        and 0.0 < maximum_sound_speed < 1.0
        and all(
            row["incoming_inner_characteristics"] == 0
            for row in inside_horizon_rows
        )
        and invalid_state_rejected
    )
    output = {
        "scope": (
            "local fixed-proper-height gas+radiation Valencia primitive map"
        ),
        "proper_half_thickness_cm": eos.proper_half_thickness,
        "rows": rows,
        "summary": {
            "states_evaluated": len(rows),
            "maximum_relative_primitive_defect": maximum_primitive_defect,
            "maximum_relative_conserved_defect": maximum_conserved_defect,
            "maximum_eigenvalue_defect": maximum_eigenvalue_defect,
            "maximum_sound_speed_over_c": maximum_sound_speed,
            "inside_horizon_states": len(inside_horizon_rows),
            "inside_horizon_maximum_incoming_modes": max(
                row["incoming_inner_characteristics"]
                for row in inside_horizon_rows
            ),
            "invalid_state_rejected": invalid_state_rejected,
        },
        "wp10c1_passed": passed,
        "production_ready": False,
        "blocking_work": [
            "covariant Kerr-Schild geometric finite-volume sources",
            "independent cell and face conservation audit",
            "relativistic common-stress and thermal ledger",
            "stream and Hill/Roche contract migration",
            "stationary and tiny-step mesh certification",
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
