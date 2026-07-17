"""Run the bounded WP10c5c primitive Schur and null-mode audit."""

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
    audit_causal_five_field_reduced_stationary_response,
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
    ROOT
    / "outputs/tables/causal_five_field_reduced_null_audit_wp10c5c.json"
)
RANK_THRESHOLD = 1.0e-11


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


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


def _field_norms(vector: np.ndarray, n_cells: int) -> dict:
    fields = np.asarray(vector, dtype=float).reshape(n_cells, 5)
    cell_norms = np.linalg.norm(fields, axis=1)
    total = max(float(np.linalg.norm(fields)), np.finfo(float).tiny)
    return {
        "field_norms": [
            float(value) for value in np.linalg.norm(fields, axis=0)
        ],
        "maximum_cell_index": int(np.argmax(cell_norms)),
        "maximum_cell_fraction": float(np.max(cell_norms) / total),
        "outermost_cell_fraction": float(cell_norms[-1] / total),
    }


def _full_row(full, n_cells: int) -> dict:
    return {
        "dimensions": list(full.dimensions),
        "numerical_rank": full.numerical_rank,
        "smallest_singular_value": full.smallest_singular_value,
        "largest_singular_value": full.largest_singular_value,
        "condition_estimate": full.condition_estimate,
        "smallest_eight_singular_values": [
            float(value) for value in full.singular_values[-8:]
        ],
        "weakest_right": {
            "conserved_norm": float(
                np.linalg.norm(full.weakest_right_singular_vector[: 5 * n_cells])
            ),
            "primitive": _field_norms(
                full.weakest_right_singular_vector[
                    5 * n_cells : 10 * n_cells
                ],
                n_cells,
            ),
            "face_flux_norm": float(
                np.linalg.norm(
                    full.weakest_right_singular_vector[10 * n_cells :]
                )
            ),
        },
    }


def _reduced_row(reduced, n_cells: int) -> dict:
    outer = reduced.outer_thermal_stress
    return {
        "dimensions": list(reduced.dimensions),
        "numerical_rank": reduced.numerical_rank,
        "smallest_singular_value": reduced.smallest_singular_value,
        "largest_singular_value": reduced.largest_singular_value,
        "condition_estimate": reduced.condition_estimate,
        "schur_numerical_rank": reduced.schur_numerical_rank,
        "schur_condition_estimate": reduced.schur_condition_estimate,
        "schur_smallest_singular_value": float(
            reduced.schur_singular_values[-1]
        ),
        "smallest_eight_singular_values": [
            float(value) for value in reduced.singular_values[-8:]
        ],
        "outer_boundary_choked": reduced.outer_boundary_choked,
        "direct_schur_equivalence": {
            "maximum_absolute_matrix_defect": (
                reduced.maximum_absolute_matrix_defect
            ),
            "relative_frobenius_matrix_defect": (
                reduced.relative_frobenius_matrix_defect
            ),
            "maximum_directional_relative_defect": (
                reduced.maximum_directional_relative_defect
            ),
            "maximum_directional_operator_scaled_defect": (
                reduced.maximum_directional_operator_scaled_defect
            ),
        },
        "algebraic_elimination": {
            "dimensions": list(reduced.algebraic_dimensions),
            "numerical_rank": reduced.algebraic_numerical_rank,
            "full_rank": reduced.algebraic_full_rank,
            "condition_estimate": reduced.algebraic_condition_estimate,
            "reconstructed_algebraic_residual_norm": (
                reduced.reconstructed_algebraic_residual_norm
            ),
            "reconstructed_full_residual_norm": (
                reduced.reconstructed_full_residual_norm
            ),
            "full_weakest_vector_alignment": (
                reduced.full_weakest_vector_alignment
            ),
        },
        "weakest_right": _field_norms(
            reduced.weakest_right_singular_vector,
            n_cells,
        ),
        "weakest_left": _field_norms(
            reduced.weakest_left_singular_vector,
            n_cells,
        ),
        "outer_thermal_stress_response": {
            "interior_dimensions": list(outer.interior_dimensions),
            "interior_numerical_rank": outer.interior_numerical_rank,
            "interior_full_rank": outer.interior_full_rank,
            "interior_condition_estimate": (
                outer.interior_condition_estimate
            ),
            "matrix": [
                [float(value) for value in row]
                for row in outer.response_matrix
            ],
            "singular_values": [
                float(value) for value in outer.singular_values
            ],
            "numerical_rank": outer.numerical_rank,
            "condition_estimate": outer.condition_estimate,
            "determinant": outer.determinant,
        },
    }


def _audit_case(
    context: CausalFiveFieldDAEContext,
    *,
    name: str,
    outer_surface_density: float,
    outer_temperature: float,
    finite_difference_step: float,
) -> tuple[dict, object]:
    state = make_causal_five_field_seed(
        context,
        outer_surface_density=outer_surface_density,
        outer_temperature=outer_temperature,
    )
    vector = pack_causal_five_field_state(state)
    evaluation = evaluate_causal_five_field_dae(vector, context)
    scaling = causal_five_field_dae_scaling(state, evaluation)
    full = audit_causal_five_field_dae_jacobian(
        lambda trial: evaluate_causal_five_field_dae(
            trial,
            context,
        ).residual,
        vector,
        scaling,
        finite_difference_step=finite_difference_step,
        rank_relative_threshold=RANK_THRESHOLD,
    )
    reduced = audit_causal_five_field_reduced_stationary_response(
        context,
        state,
        full,
        scaling=scaling,
        finite_difference_step=finite_difference_step,
        rank_relative_threshold=RANK_THRESHOLD,
    )
    return (
        {
            "name": name,
            "seed_parameters": {
                "outer_surface_density": outer_surface_density,
                "outer_temperature": outer_temperature,
                "finite_difference_step": finite_difference_step,
            },
            "seed_is_stationary_root": False,
            "maximum_scaled_conservation_residual": float(
                np.max(
                    np.abs(
                        evaluation.conservation_rows.ravel()
                        / scaling.row_scales[: 5 * state.n_cells]
                    )
                )
            ),
            "maximum_primitive_map_residual": float(
                np.max(np.abs(evaluation.primitive_map_rows))
            ),
            "maximum_face_map_residual": float(
                max(
                    np.max(np.abs(evaluation.interior_flux_rows)),
                    np.max(np.abs(evaluation.inner_flux_rows)),
                    np.max(np.abs(evaluation.outer_flux_rows)),
                )
            ),
            "outer_boundary_choked": evaluation.outer_boundary_choked,
            "full": _full_row(full, state.n_cells),
            "reduced": _reduced_row(reduced, state.n_cells),
        },
        reduced,
    )


def main() -> None:
    args = _arguments()
    n_cells = 16
    context = _context(n_cells)
    reference_rows = []
    reference_objects = []
    for step in (1.0e-6, 2.0e-6, 5.0e-6):
        row, reduced = _audit_case(
            context,
            name="published_reference_closed",
            outer_surface_density=1.0e5,
            outer_temperature=8.0e5,
            finite_difference_step=step,
        )
        reference_rows.append(row)
        reference_objects.append(reduced)

    active_set_rows = []
    active_set_objects = []
    for name, temperature in (
        ("controlled_pair_closed", 8.0e5),
        ("controlled_pair_open", 1.0e6),
    ):
        row, reduced = _audit_case(
            context,
            name=name,
            outer_surface_density=1.0e4,
            outer_temperature=temperature,
            finite_difference_step=2.0e-6,
        )
        active_set_rows.append(row)
        active_set_objects.append(reduced)

    all_objects = reference_objects + active_set_objects
    equivalence_passed = all(
        audit.algebraic_full_rank
        and audit.relative_frobenius_matrix_defect <= 2.0e-7
        and audit.maximum_directional_operator_scaled_defect <= 2.0e-8
        and audit.reconstructed_algebraic_residual_norm <= 2.0e-11
        for audit in all_objects
    )
    reference_ranks = {
        audit.numerical_rank for audit in reference_objects
    }
    reference_smallest = np.asarray(
        [audit.smallest_singular_value for audit in reference_objects],
        dtype=float,
    )
    finite_difference_stable = (
        len(reference_ranks) == 1
        and (
            np.max(reference_smallest) - np.min(reference_smallest)
        )
        / max(np.mean(reference_smallest), np.finfo(float).tiny)
        <= 1.0e-3
    )
    closed, opened = active_set_objects
    if not equivalence_passed:
        classification = "reduction_implementation_not_certified"
        next_action = "repair_reduced_operator_before_any_physics"
    elif any(not audit.algebraic_full_rank for audit in all_objects):
        classification = "algebraic_identity_block_defect"
        next_action = "repair_exact_algebraic_dependency_and_repeat_n16"
    elif (
        finite_difference_stable
        and all(audit.full_rank for audit in reference_objects)
        and all(
            audit.outer_thermal_stress.numerical_rank == 2
            for audit in all_objects
        )
    ):
        classification = (
            "full_primitive_response_with_flux_primary_embedding_conditioning"
        )
        next_action = (
            "retain_operator_and_request_separate_consistent_initial_data_gate"
        )
    else:
        classification = "unresolved_finite_difference_sensitive_response"
        next_action = "stop_and_revisit_scaling_or_active_set"

    output = {
        "work_package": "WP10c5c",
        "scope": (
            "bounded stationary primitive Schur/null audit at nonroot seeds"
        ),
        "n_cells": n_cells,
        "rank_relative_threshold": RANK_THRESHOLD,
        "reference_closed_finite_difference_audits": reference_rows,
        "controlled_closed_open_pair": active_set_rows,
        "comparison": {
            "direct_schur_equivalence_passed": equivalence_passed,
            "reference_finite_difference_stable": finite_difference_stable,
            "reference_reduced_ranks": sorted(reference_ranks),
            "reference_smallest_singular_value_relative_spread": float(
                (
                    np.max(reference_smallest)
                    - np.min(reference_smallest)
                )
                / max(
                    np.mean(reference_smallest),
                    np.finfo(float).tiny,
                )
            ),
            "controlled_closed_reduced_rank": closed.numerical_rank,
            "controlled_open_reduced_rank": opened.numerical_rank,
            "controlled_closed_outer_response_rank": (
                closed.outer_thermal_stress.numerical_rank
            ),
            "controlled_open_outer_response_rank": (
                opened.outer_thermal_stress.numerical_rank
            ),
            "controlled_closed_smallest_singular_value": (
                closed.smallest_singular_value
            ),
            "controlled_open_smallest_singular_value": (
                opened.smallest_singular_value
            ),
        },
        "classification": classification,
        "interpretation_limits": {
            "stationary_root_available": False,
            "exact_nullspace_demonstrated": False,
            "physical_marginality_demonstrated": False,
            "evolution_or_stability_demonstrated": False,
        },
        "gates": {
            "n64_n96_stationary_roots_authorized": False,
            "tiny_implicit_step_authorized": False,
            "tide_authorized": False,
            "wind_authorized": False,
        },
        "next_action": next_action,
    }
    output_path = _absolute(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            output,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            output,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
    )


if __name__ == "__main__":
    main()
