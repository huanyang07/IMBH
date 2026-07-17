"""Audit stream and Roche migration into the causal Kerr-Schild column."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from imri_qpe.constants import C, G
from imri_qpe.layer3_minidisk_1d import (
    KERR_SCHILD_HILL_ENERGY_ZERO,
    GasRadiationHillRocheNozzleProvider,
    SchwarzschildCurvatureVerticalFrequency,
    ValenciaPerfectFluidPrimitive,
    apply_kerr_schild_hill_roche_boundary,
    audit_kerr_schild_migration_rank,
    exact_kerr_schild_compact_stream_sources,
    fiducial_hill_roche_nozzle_geometry,
    kerr_schild_column_four_velocity,
    kerr_schild_column_geometry,
    kerr_schild_hill_roche_reservoir,
    kerr_schild_specific_injection_moments,
    kerr_schild_stream_injection,
    make_kerr_schild_column_grid,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "outputs/tables/causal_inner_migration_wp10c4.json"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _circular_column(
    radius: float,
    gravitational_radius: float,
    *,
    surface_density: float,
    temperature: float,
):
    geometry = kerr_schild_column_geometry(
        radius,
        gravitational_radius,
    )
    frequency = SchwarzschildCurvatureVerticalFrequency(
        gravitational_radius
    )
    eos = frequency.eos(radius)
    thermodynamics = eos.from_surface_density_temperature(
        surface_density,
        temperature,
    )
    primitive = ValenciaPerfectFluidPrimitive(
        surface_density=surface_density,
        radial_velocity_over_c=(
            2.0 * gravitational_radius / radius
        ),
        azimuthal_velocity_over_c=float(
            np.sqrt(gravitational_radius / radius)
            / geometry.base.lapse
        ),
        specific_internal_energy=(
            thermodynamics.specific_internal_energy
        ),
        integrated_pressure=thermodynamics.integrated_pressure,
    )
    return geometry, eos, thermodynamics, primitive


def _provider() -> GasRadiationHillRocheNozzleProvider:
    geometry = replace(
        fiducial_hill_roche_nozzle_geometry(),
        energy_zero=KERR_SCHILD_HILL_ENERGY_ZERO,
    )
    return GasRadiationHillRocheNozzleProvider(
        geometry,
        transverse_quadrature_zones=24,
    )


def _boundary(
    temperature: float,
    gravitational_radius: float,
    provider: GasRadiationHillRocheNozzleProvider,
):
    radius = 335.0 * gravitational_radius
    geometry, eos, _thermodynamics, primitive = _circular_column(
        radius,
        gravitational_radius,
        surface_density=1.0e4,
        temperature=temperature,
    )
    return apply_kerr_schild_hill_roche_boundary(
        geometry,
        eos,
        primitive,
        temperature=temperature,
        provider=provider,
    )


def _vertical_rows(gravitational_radius: float) -> list[dict[str, float]]:
    provider = SchwarzschildCurvatureVerticalFrequency(
        gravitational_radius
    )
    rows = []
    step = 1.0e-5
    for radius_rg in (1.5, 2.0, 20.0, 240.0, 335.0):
        radius = radius_rg * gravitational_radius
        numerical_slope = (
            np.log(provider.frequency(radius * np.exp(step)))
            - np.log(provider.frequency(radius * np.exp(-step)))
        ) / (2.0 * step)
        rows.append(
            {
                "radius_rg": radius_rg,
                "frequency_s_inv": provider.frequency(radius),
                "declared_logarithmic_slope": (
                    provider.logarithmic_radial_derivative(radius)
                ),
                "numerical_logarithmic_slope": float(numerical_slope),
                "slope_defect": float(abs(numerical_slope + 1.5)),
            }
        )
    return rows


def _moment_row(
    gravitational_radius: float,
) -> tuple[dict[str, float], object]:
    radius = 240.0 * gravitational_radius
    geometry, _eos, thermodynamics, primitive = _circular_column(
        radius,
        gravitational_radius,
        surface_density=1.0e5,
        temperature=1.0e6,
    )
    moments = kerr_schild_specific_injection_moments(
        geometry,
        primitive,
    )
    four_velocity = kerr_schild_column_four_velocity(
        geometry,
        primitive,
    )
    lower_velocity = geometry.spacetime_metric @ four_velocity
    enthalpy_over_c2 = (
        1.0 + thermodynamics.specific_enthalpy / C**2
    )
    expected = np.asarray(
        [
            enthalpy_over_c2 * lower_velocity[1],
            enthalpy_over_c2 * lower_velocity[2],
            -enthalpy_over_c2 * lower_velocity[0],
        ]
    )
    observed = np.asarray(
        [
            moments.radial_momentum_over_c,
            moments.angular_momentum_over_c,
            moments.killing_energy_over_c2,
        ]
    )
    scale = np.maximum(np.maximum(np.abs(expected), np.abs(observed)), 1.0)
    return (
        {
            "radius_rg": 240.0,
            "transport_radial_velocity_cm_s": (
                moments.transport_radial_velocity
            ),
            "kinematic_specific_angular_momentum_cm2_s": (
                moments.kinematic_specific_angular_momentum
            ),
            "specific_radial_momentum_cm_s": (
                moments.specific_radial_momentum
            ),
            "specific_flux_angular_momentum_cm2_s": (
                moments.specific_angular_momentum
            ),
            "specific_killing_energy_erg_g": (
                moments.specific_killing_energy
            ),
            "maximum_covariant_moment_relative_defect": float(
                np.max(np.abs(observed - expected) / scale)
            ),
        },
        (geometry, primitive),
    )


def _weak_field_row(
    gravitational_radius: float,
) -> dict[str, float]:
    radius = 1.0e4 * gravitational_radius
    geometry = kerr_schild_column_geometry(
        radius,
        gravitational_radius,
    )
    primitive = ValenciaPerfectFluidPrimitive(
        surface_density=1.0,
        radial_velocity_over_c=2.0 * gravitational_radius / radius,
        azimuthal_velocity_over_c=float(
            np.sqrt(gravitational_radius / radius)
            / geometry.base.lapse
        ),
        specific_internal_energy=0.0,
        integrated_pressure=0.0,
    )
    moments = kerr_schild_specific_injection_moments(
        geometry,
        primitive,
    )
    newtonian_angular_momentum = (
        C * np.sqrt(gravitational_radius * radius)
    )
    newtonian_binding_energy = (
        -0.5 * C**2 * gravitational_radius / radius
    )
    return {
        "radius_rg": 1.0e4,
        "transport_radial_velocity_over_c": (
            moments.transport_radial_velocity / C
        ),
        "angular_momentum_relative_defect": (
            moments.kinematic_specific_angular_momentum
            / newtonian_angular_momentum
            - 1.0
        ),
        "binding_energy_relative_defect": (
            (moments.specific_killing_energy - C**2)
            / newtonian_binding_energy
            - 1.0
        ),
    }


def _source_rows(
    mass: float,
    gravitational_radius: float,
    injection_state,
) -> list[dict[str, float | int | str]]:
    geometry, primitive = injection_state
    injection = kerr_schild_stream_injection(
        geometry,
        primitive,
        rest_mass_rate=5.0 * eddington_mdot(mass),
    )
    expected = np.asarray(
        [
            injection.rest_mass_rate,
            injection.rest_mass_rate
            * injection.moments.radial_momentum_over_c,
            injection.rest_mass_rate
            * injection.moments.angular_momentum_over_c,
            injection.rest_mass_rate
            * injection.moments.killing_energy_over_c2,
        ]
    )
    rows = []
    for shape in ("compact_c2", "compact_c4"):
        for n_cells in (32, 64, 128):
            grid = make_kerr_schild_column_grid(
                1.8 * gravitational_radius,
                335.0 * gravitational_radius,
                n_cells,
                gravitational_radius,
            )
            source = exact_kerr_schild_compact_stream_sources(
                grid,
                injection,
                center=240.0 * gravitational_radius,
                log_width=0.08,
                shape=shape,
            )
            observed = np.sum(source.matrix, axis=0)
            weighted = np.sum(
                source.weighted_killing_source_per_ct,
                axis=0,
            )
            scale = np.maximum(
                np.maximum(np.abs(expected), np.abs(observed)),
                1.0,
            )
            rows.append(
                {
                    "shape": shape,
                    "n_cells": n_cells,
                    "active_source_cells": int(
                        np.count_nonzero(source.rest_mass)
                    ),
                    "rest_mass_rate_g_s": float(observed[0]),
                    "maximum_moment_relative_defect": float(
                        np.max(np.abs(observed - expected) / scale)
                    ),
                    "maximum_ct_conversion_relative_defect": float(
                        np.max(
                            np.abs(weighted - observed / C)
                            / np.maximum(
                                np.maximum(
                                    np.abs(weighted),
                                    np.abs(observed / C),
                                ),
                                1.0,
                            )
                        )
                    ),
                }
            )
    return rows


def _boundary_rows(
    gravitational_radius: float,
) -> tuple[list[dict[str, float | int | bool | str]], float]:
    provider = _provider()
    radius = 335.0 * gravitational_radius

    def available(temperature: float) -> float:
        geometry, eos, _thermodynamics, primitive = _circular_column(
            radius,
            gravitational_radius,
            surface_density=1.0e4,
            temperature=temperature,
        )
        _edge, reservoir = kerr_schild_hill_roche_reservoir(
            geometry,
            eos,
            primitive,
            temperature=temperature,
        )
        return provider.available_specific_energy(reservoir)

    threshold = brentq(
        available,
        8.0e5,
        1.0e6,
        xtol=1.0e-7,
        rtol=1.0e-13,
    )
    temperatures = (
        ("closed", 8.0e5),
        ("threshold_below", threshold * (1.0 - 1.0e-6)),
        ("threshold_above", threshold * (1.0 + 1.0e-6)),
        ("choked", 1.0e6),
    )
    rows = []
    for label, temperature in temperatures:
        audit = _boundary(
            temperature,
            gravitational_radius,
            provider,
        )
        rows.append(
            {
                "label": label,
                "temperature_K": float(temperature),
                "available_specific_energy_erg_g": (
                    audit.gate.available_specific_energy
                ),
                "choked": audit.gate.choked,
                "rest_mass_rate_g_s": audit.rest_mass_rate,
                "radial_momentum_rate_dyn": (
                    audit.radial_momentum_rate
                ),
                "angular_momentum_rate_erg": (
                    audit.angular_momentum_rate
                ),
                "killing_energy_rate_erg_s": (
                    audit.killing_energy_rate
                ),
                "height_over_radius": (
                    audit.edge_state.thermodynamics.proper_half_thickness
                    / radius
                ),
                "incoming_outer_characteristics": (
                    audit.incoming_outer_characteristics
                ),
                "no_inward_mass": audit.no_inward_mass,
                "zero_outer_stress": audit.zero_outer_stress,
                "angular_momentum_relative_defect": (
                    audit.angular_momentum_relative_defect
                ),
                "killing_energy_relative_defect": (
                    audit.killing_energy_relative_defect
                ),
                "binary_pattern_power_relative_defect": (
                    audit.binary_pattern_power_relative_defect
                ),
            }
        )
    return rows, float(threshold)


def _rank_rows(
    gravitational_radius: float,
) -> list[dict[str, int | bool]]:
    boundary = _boundary(
        1.0e6,
        gravitational_radius,
        _provider(),
    )
    rows = []
    for n_cells in (16, 64, 128):
        audit = audit_kerr_schild_migration_rank(
            n_cells,
            boundary,
        )
        rows.append(
            {
                "n_cells": audit.n_cells,
                "total_unknowns": audit.total_unknowns,
                "total_rows": audit.total_rows,
                "source_unknowns": audit.source_unknowns,
                "source_rows": audit.source_rows,
                "boundary_face_rows": audit.boundary_face_rows,
                "boundary_face_jacobian_rank": (
                    audit.boundary_face_jacobian_rank
                ),
                "physical_outer_boundary_conditions": (
                    audit.physical_outer_boundary_conditions
                ),
                "square": audit.square,
            }
        )
    return rows


def main() -> None:
    arguments = _arguments()
    output = _absolute(arguments.output)
    parameters = FiducialParams()
    mass = parameters.M2_g
    gravitational_radius = G * mass / C**2

    vertical_rows = _vertical_rows(gravitational_radius)
    moment_row, injection_state = _moment_row(gravitational_radius)
    weak_field_row = _weak_field_row(gravitational_radius)
    source_rows = _source_rows(
        mass,
        gravitational_radius,
        injection_state,
    )
    boundary_rows, opening_temperature = _boundary_rows(
        gravitational_radius
    )
    rank_rows = _rank_rows(gravitational_radius)
    maximum_boundary_defect = max(
        max(
            abs(float(row["angular_momentum_relative_defect"])),
            abs(float(row["killing_energy_relative_defect"])),
            abs(float(row["binary_pattern_power_relative_defect"])),
        )
        for row in boundary_rows
    )
    summary = {
        "work_package": "WP10c4",
        "scope": (
            "bounded stream/Roche migration audit; no stationary root, "
            "long evolution, tide continuation, or wind"
        ),
        "vertical_frequency_provider": (
            "Schwarzschild curvature scale c*sqrt(rg/R^3)"
        ),
        "source_contract": (
            "one immutable four-state with exact compact cell moments"
        ),
        "outer_boundary_contract": (
            "closed pressure traction or outward choked Hill/Roche nozzle"
        ),
        "maximum_vertical_frequency_slope_defect": max(
            row["slope_defect"] for row in vertical_rows
        ),
        "maximum_covariant_moment_relative_defect": (
            moment_row["maximum_covariant_moment_relative_defect"]
        ),
        "weak_field_angular_momentum_relative_defect": (
            weak_field_row["angular_momentum_relative_defect"]
        ),
        "weak_field_binding_energy_relative_defect": (
            weak_field_row["binding_energy_relative_defect"]
        ),
        "maximum_source_moment_relative_defect": max(
            float(row["maximum_moment_relative_defect"])
            for row in source_rows
        ),
        "maximum_source_ct_conversion_relative_defect": max(
            float(row["maximum_ct_conversion_relative_defect"])
            for row in source_rows
        ),
        "opening_temperature_K": opening_temperature,
        "maximum_boundary_ledger_relative_defect": (
            maximum_boundary_defect
        ),
        "outer_incoming_characteristics": sorted(
            {
                int(row["incoming_outer_characteristics"])
                for row in boundary_rows
            }
        ),
        "all_rank_counts_square": all(
            bool(row["square"]) for row in rank_rows
        ),
        "all_boundary_face_jacobians_full_rank": all(
            int(row["boundary_face_jacobian_rank"])
            == int(row["boundary_face_rows"])
            for row in rank_rows
        ),
    }
    payload = {
        "summary": summary,
        "vertical_frequency": vertical_rows,
        "stream_four_state": moment_row,
        "weak_field_stream_moments": weak_field_row,
        "exact_stream_sources": source_rows,
        "roche_boundary": boundary_rows,
        "rank": rank_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
