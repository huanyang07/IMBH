"""Run the bounded five-field causal-DAE count and rank preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from imri_qpe.constants import C, G
from imri_qpe.layer3_minidisk_1d import (
    SchwarzschildCurvatureVerticalFrequency,
    ValenciaPerfectFluidPrimitive,
    audit_causal_five_field_boundaries,
    audit_causal_five_field_principal,
    calibrate_causal_alpha_shear,
    causal_comoving_energy_source,
    causal_five_field_dae_count,
    causal_rest_frame_shear_rate,
    causal_temporal_vertical_work_storage,
    kerr_schild_column_four_velocity,
    kerr_schild_column_geometry,
)
from imri_qpe.parameters import FiducialParams


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "outputs/tables/causal_inner_dae_preflight_wp10c5.json"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _column(
    radius: float,
    gravitational_radius: float,
    *,
    surface_density: float,
    temperature: float,
    radial_velocity_over_c: float,
    azimuthal_velocity_over_c: float,
):
    geometry = kerr_schild_column_geometry(
        radius,
        gravitational_radius,
    )
    eos = SchwarzschildCurvatureVerticalFrequency(
        gravitational_radius
    ).eos(radius)
    thermodynamics = eos.from_surface_density_temperature(
        surface_density,
        temperature,
    )
    primitive = ValenciaPerfectFluidPrimitive(
        surface_density=surface_density,
        radial_velocity_over_c=radial_velocity_over_c,
        azimuthal_velocity_over_c=azimuthal_velocity_over_c,
        specific_internal_energy=(
            thermodynamics.specific_internal_energy
        ),
        integrated_pressure=thermodynamics.integrated_pressure,
    )
    closure = calibrate_causal_alpha_shear(
        primitive,
        alpha=0.1,
        reference_positive_shear_rate=(
            1.5 * C * np.sqrt(gravitational_radius / radius**3)
        ),
        viscous_signal_speed_over_c=(
            np.sqrt(0.1) * thermodynamics.sound_speed / C
        ),
    )
    return geometry, eos, thermodynamics, primitive, closure


def _principal_row(
    radius_over_rg: float,
    gravitational_radius: float,
    *,
    surface_density: float,
    temperature: float,
    radial_velocity_over_c: float,
    azimuthal_velocity_over_c: float,
) -> tuple[dict, object]:
    radius = radius_over_rg * gravitational_radius
    geometry, eos, _thermodynamics, _primitive, closure = _column(
        radius,
        gravitational_radius,
        surface_density=surface_density,
        temperature=temperature,
        radial_velocity_over_c=radial_velocity_over_c,
        azimuthal_velocity_over_c=azimuthal_velocity_over_c,
    )
    audit = audit_causal_five_field_principal(
        geometry,
        eos,
        closure,
        surface_density=surface_density,
        radial_velocity_over_c=radial_velocity_over_c,
        azimuthal_velocity_over_c=azimuthal_velocity_over_c,
        temperature=temperature,
    )
    return (
        {
            "radius_rg": radius_over_rg,
            "analytic_local_rest_speeds_over_c": list(
                audit.analytic_local_rest_speeds_over_c
            ),
            "numerical_local_rest_speeds_over_c": list(
                audit.numerical_local_rest_speeds_over_c
            ),
            "coordinate_speeds_over_c": list(
                audit.coordinate_speeds_over_c
            ),
            "incoming_inner_characteristics": (
                audit.incoming_inner_characteristics
            ),
            "incoming_outer_characteristics": (
                audit.incoming_outer_characteristics
            ),
            "stationary_coordinate_rank": (
                audit.stationary_coordinate_rank
            ),
            "incoming_mode_response_rank": (
                audit.incoming_mode_response_rank
            ),
            "incoming_mode_response_smallest_singular_value": (
                audit.incoming_mode_response_smallest_singular_value
            ),
            "maximum_local_rest_eigenvalue_defect": (
                audit.maximum_local_rest_eigenvalue_defect
            ),
            "maximum_imaginary_eigenvalue": (
                audit.maximum_imaginary_eigenvalue
            ),
            "maximum_light_cone_excess": (
                audit.maximum_light_cone_excess
            ),
            "causal_and_hyperbolic": audit.causal_and_hyperbolic,
        },
        audit,
    )


def _shear_rows(
    mass: float,
    gravitational_radius: float,
) -> list[dict]:
    rows = []
    for radius_over_rg in (1.0e3, 1.0e4, 1.0e5):
        radius = radius_over_rg * gravitational_radius
        relative_step = 2.0e-5

        def lower_velocity(local_radius: float) -> np.ndarray:
            geometry = kerr_schild_column_geometry(
                local_radius,
                gravitational_radius,
            )
            _geometry, _eos, _thermodynamics, primitive, _closure = (
                _column(
                    local_radius,
                    gravitational_radius,
                    surface_density=1.0e3,
                    temperature=1.0e4,
                    radial_velocity_over_c=(
                        2.0 * gravitational_radius / local_radius
                    ),
                    azimuthal_velocity_over_c=(
                        np.sqrt(
                            gravitational_radius / local_radius
                        )
                        / geometry.base.lapse
                    ),
                )
            )
            return (
                geometry.spacetime_metric
                @ kerr_schild_column_four_velocity(
                    geometry,
                    primitive,
                )
            )

        left = radius * (1.0 - relative_step)
        right = radius * (1.0 + relative_step)
        derivative = (
            lower_velocity(right) - lower_velocity(left)
        ) / (right - left)
        geometry = kerr_schild_column_geometry(
            radius,
            gravitational_radius,
        )
        _geometry, _eos, _thermodynamics, primitive, _closure = _column(
            radius,
            gravitational_radius,
            surface_density=1.0e3,
            temperature=1.0e4,
            radial_velocity_over_c=(
                2.0 * gravitational_radius / radius
            ),
            azimuthal_velocity_over_c=(
                np.sqrt(gravitational_radius / radius)
                / geometry.base.lapse
            ),
        )
        shear = causal_rest_frame_shear_rate(
            geometry,
            primitive,
            radial_lower_four_velocity_derivative=derivative,
        )
        newtonian = 1.5 * np.sqrt(G * mass / radius**3)
        rows.append(
            {
                "radius_rg": radius_over_rg,
                "covariant_shear_rate_per_s": shear,
                "newtonian_minus_r_domega_dr_per_s": newtonian,
                "relative_defect": shear / newtonian - 1.0,
            }
        )
    return rows


def _temporal_storage_audit(
    gravitational_radius: float,
) -> dict:
    radius = 20.0 * gravitational_radius
    geometry, eos, _thermodynamics, primitive, _closure = _column(
        radius,
        gravitational_radius,
        surface_density=1.0e6,
        temperature=2.0e7,
        radial_velocity_over_c=-0.08,
        azimuthal_velocity_over_c=0.35,
    )
    old = eos.from_surface_density_temperature(1.0e6, 1.9e7)
    new = eos.from_surface_density_temperature(1.0e6, 2.1e7)
    storage = causal_temporal_vertical_work_storage(
        geometry,
        primitive,
        old,
        new,
    )
    coordinate_time = 3.25
    four_velocity = kerr_schild_column_four_velocity(
        geometry,
        primitive,
    )
    proper_rate = (
        -storage.work_per_area
        * four_velocity[0]
        / coordinate_time
    )
    source = causal_comoving_energy_source(
        geometry,
        primitive,
        comoving_energy_rate=proper_rate,
    )
    expected = (
        -source.killing_source_per_ct * C * coordinate_time
    )
    scale = np.maximum(
        np.maximum(
            np.abs(storage.killing_storage_increment),
            np.abs(expected),
        ),
        1.0e-30,
    )
    return {
        "work_per_area": storage.work_per_area,
        "killing_storage_increment": (
            storage.killing_storage_increment.tolist()
        ),
        "integrated_four_force_increment": expected.tolist(),
        "maximum_component_relative_defect": float(
            np.max(
                np.abs(
                    storage.killing_storage_increment - expected
                )
                / scale
            )
        ),
    }


def main() -> None:
    arguments = _arguments()
    parameters = FiducialParams()
    mass = parameters.M2_g
    gravitational_radius = G * mass / C**2

    count_rows = []
    for n_cells in (16, 64, 96):
        count = causal_five_field_dae_count(n_cells)
        count_rows.append(
            {
                "n_cells": n_cells,
                "conserved_unknowns": count.conserved_unknowns,
                "primitive_unknowns": count.primitive_unknowns,
                "face_flux_unknowns": count.face_flux_unknowns,
                "total_unknowns": count.total_unknowns,
                "total_rows": count.total_rows,
                "nonconservative_shear_rows": (
                    count.nonconservative_shear_rows
                ),
                "physical_inner_boundary_conditions": (
                    count.physical_inner_boundary_conditions
                ),
                "physical_outer_boundary_conditions": (
                    count.physical_outer_boundary_conditions
                ),
                "square": count.square,
            }
        )

    inner_row, inner = _principal_row(
        1.8,
        gravitational_radius,
        surface_density=1.0e5,
        temperature=3.0e7,
        radial_velocity_over_c=-0.40,
        azimuthal_velocity_over_c=0.60,
    )
    representative_row, _representative = _principal_row(
        20.0,
        gravitational_radius,
        surface_density=1.0e7,
        temperature=1.0e7,
        radial_velocity_over_c=-0.01,
        azimuthal_velocity_over_c=0.20,
    )
    outer_radius = 335.0 * gravitational_radius
    outer_geometry = kerr_schild_column_geometry(
        outer_radius,
        gravitational_radius,
    )
    outer_row, outer = _principal_row(
        335.0,
        gravitational_radius,
        surface_density=1.0e4,
        temperature=8.0e5,
        radial_velocity_over_c=(
            2.0 * gravitational_radius / outer_radius
        ),
        azimuthal_velocity_over_c=(
            np.sqrt(gravitational_radius / outer_radius)
            / outer_geometry.base.lapse
        ),
    )
    boundary = audit_causal_five_field_boundaries(inner, outer)
    shear_rows = _shear_rows(mass, gravitational_radius)
    temporal_storage = _temporal_storage_audit(
        gravitational_radius
    )
    local_gate = bool(
        all(row["square"] for row in count_rows)
        and inner.causal_and_hyperbolic
        and outer.causal_and_hyperbolic
        and boundary.passed
        and abs(shear_rows[-1]["relative_defect"]) < 1.0e-4
        and temporal_storage[
            "maximum_component_relative_defect"
        ]
        < 5.0e-15
    )
    output = {
        "work_package": "WP10c5 bounded five-field DAE preflight",
        "count_formula": "15 N + 5",
        "count_rows": count_rows,
        "principal_rows": [
            inner_row,
            representative_row,
            outer_row,
        ],
        "boundary_rank": {
            "inner_incoming_characteristics": (
                boundary.inner_incoming_characteristics
            ),
            "outer_incoming_characteristics": (
                boundary.outer_incoming_characteristics
            ),
            "outer_face_rows": boundary.outer_face_rows,
            "outer_face_jacobian_rank": (
                boundary.outer_face_jacobian_rank
            ),
            "outer_physical_boundary_conditions": (
                boundary.outer_physical_boundary_conditions
            ),
            "outer_incoming_response_rank": (
                boundary.outer_incoming_response_rank
            ),
            "outer_incoming_response_smallest_singular_value": (
                boundary.outer_incoming_response_smallest_singular_value
            ),
            "passed": boundary.passed,
        },
        "covariant_shear_rows": shear_rows,
        "temporal_vertical_work_storage": temporal_storage,
        "local_preflight_passed": local_gate,
        "assembly_gate": {
            "transformed_global_mass_matrix_assembled": False,
            "path_conservative_shear_gradient_assembled": False,
            "five_component_roche_face_assembled": False,
            "stationary_roots_authorized": False,
            "tiny_implicit_step_authorized": False,
        },
        "blocking_work": [
            (
                "insert the covariant shear gradient into a declared "
                "path-conservative fifth finite-volume row"
            ),
            (
                "insert the temporal Killing-storage increment into the "
                "full backward-Euler primitive map"
            ),
            (
                "extend the four-component Roche provider to the fifth "
                "zero-stress face contract"
            ),
            (
                "differentiate the assembled nonlinear residual and repeat "
                "the global boundary-rank audit"
            ),
        ],
        "classification": {
            "numerical_status": "supported local/count preflight",
            "physical_status": "diagnostic only",
            "production_status": "blocked before roots",
        },
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
