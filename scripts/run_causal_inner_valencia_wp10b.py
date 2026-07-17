"""Audit the horizon-penetrating Valencia inner-core architecture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from imri_qpe.constants import C
from imri_qpe.layer3_minidisk_1d import (
    audit_ideal_gas_valencia_eigensystem,
    schwarzschild_kerr_schild_geometry,
    valencia_flux_primary_count,
    valencia_radial_characteristic_speeds_over_c,
)
from imri_qpe.parameters import FiducialParams


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs/tables/causal_inner_thermodynamics_wp10a.json"
DEFAULT_OUTPUT = ROOT / "outputs/tables/causal_inner_valencia_wp10b.json"
RADII_RG = (4.5, 2.1, 2.0, 1.9, 1.5)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _metric_rows(mass: float, gravitational_radius: float) -> list[dict]:
    rows = []
    for radius_rg in RADII_RG:
        geometry = schwarzschild_kerr_schild_geometry(
            radius_rg * gravitational_radius,
            mass,
        )
        rows.append(
            {
                "radius_rg": radius_rg,
                "lapse": geometry.lapse,
                "radial_shift_over_c": geometry.radial_shift_over_c,
                "ingoing_light_speed_over_c": (
                    geometry.ingoing_light_speed_over_c
                ),
                "outgoing_light_speed_over_c": (
                    geometry.outgoing_light_speed_over_c
                ),
            }
        )
    return rows


def _inside_horizon_scan(
    mass: float,
    gravitational_radius: float,
) -> list[dict]:
    rows = []
    for radius_rg in (1.9, 1.5):
        geometry = schwarzschild_kerr_schild_geometry(
            radius_rg * gravitational_radius,
            mass,
        )
        maximum_speed = -np.inf
        evaluated = 0
        for beta_r in np.linspace(-0.9, 0.9, 19):
            maximum_phi = np.sqrt(max(1.0 - beta_r**2, 0.0))
            for fraction in (0.0, 0.4, 0.8):
                beta_phi = fraction * maximum_phi
                if beta_r**2 + beta_phi**2 >= 1.0:
                    continue
                for sound in (0.01, 0.2, 1.0 / np.sqrt(3.0)):
                    speeds = valencia_radial_characteristic_speeds_over_c(
                        geometry,
                        radial_velocity_over_c=float(beta_r),
                        azimuthal_velocity_over_c=float(beta_phi),
                        sound_speed_over_c=float(sound),
                    )
                    maximum_speed = max(maximum_speed, max(speeds))
                    evaluated += 1
        rows.append(
            {
                "radius_rg": radius_rg,
                "states_evaluated": evaluated,
                "maximum_characteristic_speed_over_c": float(maximum_speed),
                "all_characteristics_leave_inner_domain": bool(
                    maximum_speed < 0.0
                ),
            }
        )
    return rows


def _critical_rank_audit(
    mass: float,
    gravitational_radius: float,
) -> dict:
    geometry = schwarzschild_kerr_schild_geometry(
        4.5 * gravitational_radius,
        mass,
    )
    sigma = 2.0
    pressure = 0.03 * sigma * C**2
    beta_phi = 0.25

    def outgoing_speed(beta_r: float) -> float:
        audit = audit_ideal_gas_valencia_eigensystem(
            geometry,
            surface_density=sigma,
            radial_velocity_over_c=beta_r,
            azimuthal_velocity_over_c=beta_phi,
            integrated_pressure=pressure,
        )
        return audit.analytic_speeds_over_c[-1]

    lower = -0.999 * np.sqrt(1.0 - beta_phi**2)
    critical_beta_r = brentq(outgoing_speed, lower, 0.5)
    critical = audit_ideal_gas_valencia_eigensystem(
        geometry,
        surface_density=sigma,
        radial_velocity_over_c=critical_beta_r,
        azimuthal_velocity_over_c=beta_phi,
        integrated_pressure=pressure,
    )
    return {
        "radius_rg": 4.5,
        "critical_radial_velocity_over_c": float(critical_beta_r),
        "azimuthal_velocity_over_c": beta_phi,
        "analytic_speeds_over_c": list(critical.analytic_speeds_over_c),
        "stationary_flux_rank": critical.stationary_flux_rank,
        "smallest_scaled_stationary_singular_value": (
            critical.smallest_stationary_singular_value
        ),
        "maximum_eigenvalue_defect": critical.maximum_eigenvalue_defect,
    }


def main() -> None:
    arguments = _arguments()
    with _absolute(arguments.input).open(encoding="utf-8") as stream:
        wp10a = json.load(stream)

    fiducial = FiducialParams()
    reference = schwarzschild_kerr_schild_geometry(1.0, fiducial.M2_g)
    gravitational_radius = reference.gravitational_radius
    representative_geometry = schwarzschild_kerr_schild_geometry(
        4.5 * gravitational_radius,
        fiducial.M2_g,
    )
    representative = audit_ideal_gas_valencia_eigensystem(
        representative_geometry,
        surface_density=2.0,
        radial_velocity_over_c=-0.2,
        azimuthal_velocity_over_c=0.55,
        integrated_pressure=0.06 * C**2,
    )
    interior_scan = _inside_horizon_scan(
        fiducial.M2_g,
        gravitational_radius,
    )
    old_rows = wp10a["rows"]
    first_superluminal = next(
        (
            row["radius_rg"]
            for row in old_rows
            if not row["full_velocity_subluminal"]
        ),
        None,
    )
    reference_count = valencia_flux_primary_count(16)
    output = {
        "selected_architecture": (
            "one-domain ingoing-Kerr-Schild Schwarzschild Valencia column"
        ),
        "old_pw_profile": {
            "first_audited_superluminal_total_velocity_radius_rg": (
                first_superluminal
            ),
            "maximum_azimuthal_speed_over_c": max(
                row["azimuthal_speed_over_c"] for row in old_rows
            ),
            "full_state_excision_candidate_exists": wp10a[
                "full_state_excision_candidate_exists"
            ],
        },
        "metric_rows": _metric_rows(
            fiducial.M2_g,
            gravitational_radius,
        ),
        "representative_eigensystem": {
            "analytic_speeds_over_c": list(
                representative.analytic_speeds_over_c
            ),
            "numerical_speeds_over_c": list(
                representative.numerical_speeds_over_c
            ),
            "maximum_eigenvalue_defect": (
                representative.maximum_eigenvalue_defect
            ),
            "stationary_flux_rank": representative.stationary_flux_rank,
        },
        "inside_horizon_scan": interior_scan,
        "stationary_critical_rank": _critical_rank_audit(
            fiducial.M2_g,
            gravitational_radius,
        ),
        "flux_primary_count": {
            "reference_cells": reference_count.n_cells,
            "reference_unknowns": reference_count.total_unknowns,
            "reference_rows": reference_count.total_rows,
            "unknowns": "12 N + 4",
            "backward_euler_conservation_rows": "4 N",
            "primitive_map_rows": "4 N",
            "interior_face_flux_rows": "4 (N - 1)",
            "inner_one_sided_flux_rows": 4,
            "outer_provider_flux_rows": 4,
            "physical_inner_boundary_rows": (
                reference_count.physical_inner_boundary_rows
            ),
        },
        "local_prototype_passed": bool(
            representative.maximum_eigenvalue_defect < 2.0e-8
            and representative.stationary_flux_rank == 4
            and all(
                row["all_characteristics_leave_inner_domain"]
                for row in interior_scan
            )
        ),
        "production_ready": False,
        "blocking_work": [
            "gas-radiation column primitive recovery",
            "Kerr-Schild geometric and radiation source discretization",
            "causal stress transport consistent with the stationary branch",
            "full-domain mapping and outer Hill/Roche contract migration",
            "N64/N96 stationary and tiny-step certification",
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
