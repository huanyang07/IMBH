"""Run the bounded WP10c5b assembled-residual and Jacobian gate."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from imri_qpe.constants import C, G
from imri_qpe.layer3_minidisk_1d import (
    KERR_SCHILD_HILL_ENERGY_ZERO,
    CausalFiveFieldDAEContext,
    GasRadiationHillRocheNozzleProvider,
    SchwarzschildCurvatureVerticalFrequency,
    audit_causal_five_field_dae_jacobian,
    causal_five_field_dae_count,
    causal_five_field_dae_scaling,
    evaluate_causal_five_field_dae,
    fiducial_hill_roche_nozzle_geometry,
    make_causal_five_field_seed,
    make_kerr_schild_column_grid,
    pack_causal_five_field_state,
)
from imri_qpe.parameters import FiducialParams


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "outputs/tables/causal_five_field_dae_assembly_wp10c5b.json"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _context(n_cells: int) -> CausalFiveFieldDAEContext:
    mass = FiducialParams().M2_g
    gravitational_radius = G * mass / C**2
    grid = make_kerr_schild_column_grid(
        1.8 * gravitational_radius,
        335.0 * gravitational_radius,
        n_cells,
        gravitational_radius,
    )
    geometry = replace(
        fiducial_hill_roche_nozzle_geometry(),
        energy_zero=KERR_SCHILD_HILL_ENERGY_ZERO,
    )
    provider = GasRadiationHillRocheNozzleProvider(
        geometry,
        transverse_quadrature_zones=24,
    )
    return CausalFiveFieldDAEContext(
        grid=grid,
        vertical_frequency=SchwarzschildCurvatureVerticalFrequency(
            gravitational_radius
        ),
        outer_boundary_provider=provider,
        include_radiative_cooling=True,
    ).validated()


def _right_block_norms(vector: np.ndarray, n_cells: int) -> dict:
    conserved_end = 5 * n_cells
    primitive_end = 10 * n_cells
    primitive = vector[conserved_end:primitive_end].reshape(n_cells, 5)
    cell_norms = np.linalg.norm(primitive, axis=1)
    return {
        "conserved": float(np.linalg.norm(vector[:conserved_end])),
        "primitives": float(
            np.linalg.norm(vector[conserved_end:primitive_end])
        ),
        "face_fluxes": float(np.linalg.norm(vector[primitive_end:])),
        "primitive_field_norms": [
            float(value)
            for value in np.linalg.norm(primitive, axis=0)
        ],
        "primitive_maximum_cell_index": int(np.argmax(cell_norms)),
        "primitive_maximum_cell_fraction": float(
            np.max(cell_norms) / max(np.linalg.norm(primitive), 1.0e-300)
        ),
    }


def _left_block_norms(vector: np.ndarray, n_cells: int) -> dict:
    conservation_end = 5 * n_cells
    primitive_end = 10 * n_cells
    interior_end = primitive_end + 5 * (n_cells - 1)
    inner_end = interior_end + 5
    return {
        "conservation": float(
            np.linalg.norm(vector[:conservation_end])
        ),
        "primitive_map": float(
            np.linalg.norm(vector[conservation_end:primitive_end])
        ),
        "interior_flux": float(
            np.linalg.norm(vector[primitive_end:interior_end])
        ),
        "inner_flux": float(
            np.linalg.norm(vector[interior_end:inner_end])
        ),
        "outer_flux": float(np.linalg.norm(vector[inner_end:])),
    }


def _audit_row(audit, n_cells: int) -> dict:
    return {
        "dimensions": list(audit.dimensions),
        "numerical_rank": audit.numerical_rank,
        "full_rank": audit.full_rank,
        "smallest_singular_value": audit.smallest_singular_value,
        "largest_singular_value": audit.largest_singular_value,
        "condition_estimate": audit.condition_estimate,
        "finite_difference_step": audit.finite_difference_step,
        "smallest_eight_singular_values": [
            float(value) for value in audit.singular_values[-8:]
        ],
        "weakest_right_block_norms": _right_block_norms(
            audit.weakest_right_singular_vector,
            n_cells,
        ),
        "weakest_left_block_norms": _left_block_norms(
            audit.weakest_left_singular_vector,
            n_cells,
        ),
    }


def _matrix_rank(values: np.ndarray, relative_threshold: float) -> dict:
    singular = np.linalg.svd(values, compute_uv=False)
    threshold = max(
        relative_threshold * singular[0],
        np.finfo(float).eps * max(values.shape) * singular[0],
    )
    rank = int(np.sum(singular > threshold))
    return {
        "dimensions": list(values.shape),
        "numerical_rank": rank,
        "expected_rank": 80,
        "rank_threshold": float(threshold),
        "smallest_nonzero_singular_value": float(
            singular[rank - 1] if rank else 0.0
        ),
        "largest_singular_value": float(singular[0]),
    }


def main() -> None:
    args = _arguments()
    n_cells = 16
    context = _context(n_cells)
    state = make_causal_five_field_seed(context)
    vector = pack_causal_five_field_state(state)
    stationary = evaluate_causal_five_field_dae(vector, context)
    scaling = causal_five_field_dae_scaling(state, stationary)
    count = causal_five_field_dae_count(n_cells)

    finite_difference_steps = (1.0e-6, 2.0e-6, 5.0e-6)
    stationary_audits = []
    stationary_objects = []
    for step in finite_difference_steps:
        audit = audit_causal_five_field_dae_jacobian(
            lambda trial: evaluate_causal_five_field_dae(
                trial,
                context,
            ).residual,
            vector,
            scaling,
            finite_difference_step=step,
            rank_relative_threshold=1.0e-11,
        )
        stationary_objects.append(audit)
        stationary_audits.append(_audit_row(audit, n_cells))

    timestep_audits = []
    backward_euler_objects = {}
    for timestep in (0.1, 1.0, 10.0):
        audit = audit_causal_five_field_dae_jacobian(
            lambda trial, dt=timestep: evaluate_causal_five_field_dae(
                trial,
                context,
                old_vector=vector,
                timestep_seconds=dt,
            ).residual,
            vector,
            scaling,
            finite_difference_step=2.0e-6,
            rank_relative_threshold=1.0e-11,
        )
        backward_euler_objects[timestep] = audit
        row = _audit_row(audit, n_cells)
        row["timestep_seconds"] = timestep
        timestep_audits.append(row)

    representative_stationary = stationary_objects[1]
    representative_backward_euler = backward_euler_objects[1.0]
    descriptor = (
        representative_backward_euler.scaled_jacobian
        - representative_stationary.scaled_jacobian
    )
    descriptor_rank = _matrix_rank(descriptor, 1.0e-11)

    telescoped = np.sum(stationary.conservation_rows, axis=0)
    expected_telescoped = (
        state.weighted_face_fluxes_over_c[-1]
        - state.weighted_face_fluxes_over_c[0]
        - np.sum(stationary.integrated_sources_per_ct, axis=0)
    )
    telescope_scale = np.maximum(
        np.maximum(np.abs(telescoped), np.abs(expected_telescoped)),
        1.0,
    )
    telescope_defect = float(
        np.max(np.abs(telescoped - expected_telescoped) / telescope_scale)
    )
    stationary_rank_passed = all(
        row["full_rank"] for row in stationary_audits
    )
    backward_euler_rank_passed = all(
        row["full_rank"] for row in timestep_audits
    )
    descriptor_rank_passed = (
        descriptor_rank["numerical_rank"]
        == descriptor_rank["expected_rank"]
    )
    roots_authorized = (
        stationary_rank_passed
        and backward_euler_rank_passed
        and descriptor_rank_passed
    )
    tiny_step_authorized = roots_authorized

    output = {
        "work_package": "WP10c5b",
        "model": {
            "n_cells": n_cells,
            "inner_radius_rg": 1.8,
            "outer_radius_rg": 335.0,
            "primitive_order": [
                "lnSigma",
                "beta_R",
                "beta_phi",
                "lnT",
                "specific_stress",
            ],
            "conserved_order": [
                "rest_mass",
                "radial_momentum_over_c",
                "angular_momentum_over_c",
                "killing_energy_over_c2",
                "relaxing_stress_density",
            ],
            "path_contract": (
                "straight arithmetic face path in lower four-velocity "
                "and lnH"
            ),
            "interior_flux": "five-field local-Lax-Friedrichs/Rusanov",
            "inner_boundary": "one-sided excision flux, zero physical BC",
            "outer_boundary": (
                "physical Hill/Roche acoustic flux plus zero shear stress"
            ),
            "radiative_cooling": True,
            "stream_source": False,
        },
        "count": {
            "unknowns": count.total_unknowns,
            "rows": count.total_rows,
            "square": count.square,
        },
        "seed": {
            "outer_boundary_choked": stationary.outer_boundary_choked,
            "outer_incoming_characteristics": (
                stationary.outer_incoming_characteristics
            ),
            "maximum_absolute_physical_residual": (
                stationary.maximum_absolute_residual
            ),
            "maximum_scaled_residual": float(
                np.max(np.abs(stationary.residual / scaling.row_scales))
            ),
            "minimum_scattering_optical_depth": float(
                np.min(stationary.scattering_optical_depths)
            ),
            "minimum_proper_shear_rate_per_s": float(
                np.min(stationary.proper_shear_rates)
            ),
            "maximum_proper_shear_rate_per_s": float(
                np.max(stationary.proper_shear_rates)
            ),
            "conservation_telescoping_relative_defect": telescope_defect,
            "maximum_primitive_map_residual": float(
                np.max(np.abs(stationary.primitive_map_rows))
            ),
            "maximum_face_map_residual": float(
                max(
                    np.max(np.abs(stationary.interior_flux_rows)),
                    np.max(np.abs(stationary.inner_flux_rows)),
                    np.max(np.abs(stationary.outer_flux_rows)),
                )
            ),
        },
        "stationary_jacobian_audits": stationary_audits,
        "backward_euler_jacobian_audits": timestep_audits,
        "descriptor_mass_rank": descriptor_rank,
        "gates": {
            "stationary_rank_passed": stationary_rank_passed,
            "backward_euler_rank_passed": backward_euler_rank_passed,
            "descriptor_rank_passed": descriptor_rank_passed,
            "n64_n96_stationary_roots_authorized": roots_authorized,
            "tiny_implicit_step_authorized": tiny_step_authorized,
        },
        "decision": (
            "stop_before_roots"
            if not roots_authorized
            else "proceed_to_n64_n96_roots"
        ),
    }
    output_path = _absolute(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
