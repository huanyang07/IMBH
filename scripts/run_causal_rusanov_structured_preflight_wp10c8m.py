"""Run the final-generator cached-branch Rusanov preflight for WP10c8m-B."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import run_causal_moment_sufficiency_audit_wp10c8i as wp10c8i
import run_causal_shell_closure_preflight_wp10c8h as wp10c8h
import run_causal_tangent_certification_wp10c8j as wp10c8j
import run_causal_tangent_descriptor_wp10c8l as wp10c8l_a
from imri_qpe.constants import C
from imri_qpe.layer3_minidisk_1d import (
    causal_five_field_assemble_evolving_tangent,
    causal_five_field_branch_frozen_mapped_storage_derivatives,
    causal_five_field_reduced_storage_rate_derivatives,
    causal_five_field_scaled_primitive_vector_field,
    unpack_causal_five_field_state,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_dae_system import (
    causal_five_field_rusanov_control_diagnostics,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_mixed_reduction import (
    causal_weighted_constraint_null_basis,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_rusanov_certification import (
    rusanov_structured_zero_remainder_preflight,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "4dc5cea0342d35135e31078669e7e71ba7d16cf9"
WORK_PACKAGE = "WP10c8m-B"
THIS_RUNNER = "scripts/run_causal_rusanov_structured_preflight_wp10c8m.py"
LOCKED_CASES = ((64, "t_0"), (64, "t_0p025"))
TRACK_A_RESULTS = (
    ROOT / "outputs/tables/causal_tangent_branch_frozen_wp10c8m.json",
    ROOT / "outputs/tables/causal_tangent_branch_frozen_wp10c8m_n128.json",
)
LEVEL_INDEX = 4
LOCKED_HORIZONS_SECONDS = (1.0e-2, 2.5e-2)
LOCKED_TIME_PANELS = (64, 128)
ALLOWED_MAXIMUM_GATE_FRACTION = 1.0e-2
LOCAL_DIFFERENCE_STEP = 1.0e-3
MAXIMUM_FACTOR_IDENTITY_RELATIVE_DEFECT = 1.0e-10
MAXIMUM_CANDIDATE_GRADIENT_RELATIVE_DEFECT = 5.0e-3
DEFAULT_OUTPUT = (
    ROOT / "outputs/tables/causal_rusanov_structured_preflight_wp10c8m.json"
)
DEFAULT_ARRAYS = (
    ROOT
    / "outputs/tables/causal_rusanov_structured_preflight_wp10c8m_arrays.npz"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--arrays", type=Path, default=DEFAULT_ARRAYS)
    return parser.parse_args()


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_defect(left: np.ndarray, right: np.ndarray) -> float:
    first = np.asarray(left, dtype=float)
    second = np.asarray(right, dtype=float)
    scale = max(
        float(np.max(np.abs(first), initial=0.0)),
        float(np.max(np.abs(second), initial=0.0)),
        np.finfo(float).tiny,
    )
    return float(np.max(np.abs(first - second), initial=0.0) / scale)


def _targeted_candidate_identities(
    initial: dict,
    primitives: np.ndarray,
) -> tuple[dict, list[tuple[int, int, int]]]:
    audit = causal_five_field_rusanov_control_diagnostics(
        initial["context"],
        primitives.reshape(initial["state"].n_cells, 5),
    )
    margins = np.asarray(audit["relative_control_margins"], dtype=float)
    exact_zero = np.asarray(audit["exact_zero_conserved_jump"], dtype=bool)
    codes = np.asarray(audit["control_codes"], dtype=int)
    speeds = np.asarray(audit["candidate_absolute_speeds_over_c"], dtype=float)
    identities: list[tuple[int, int, int]] = []
    for output_index in np.flatnonzero(
        (margins < wp10c8i.MINIMUM_RUSANOV_CONTROL_RELATIVE_MARGIN)
        & ~exact_zero
    ):
        control = int(codes[output_index])
        maximum = float(speeds[output_index, control])
        gaps = (maximum - speeds[output_index]) / max(
            maximum, np.finfo(float).tiny
        )
        for competitor in np.flatnonzero(
            gaps < wp10c8i.MINIMUM_RUSANOV_CONTROL_RELATIVE_MARGIN
        ):
            if int(competitor) != control:
                identities.append(
                    (
                        int(output_index),
                        int(audit["face_indices"][output_index]),
                        int(competitor),
                    )
                )
    return audit, identities


def _candidate_gradient_defect(
    initial: dict,
    primitives: np.ndarray,
    primitive_scales: np.ndarray,
    audit: dict,
    identities: list[tuple[int, int, int]],
    cached_right: np.ndarray,
) -> float:
    n_cells = initial["state"].n_cells
    step = wp10c8i.RUSANOV_SWITCHING_NORMAL_DIFFERENCE_STEP
    base_codes = np.asarray(audit["control_codes"], dtype=int)
    maximum_defect = 0.0
    for branch, (output_index, face_index, competitor) in enumerate(identities):
        control = int(base_codes[output_index])
        gradient = np.zeros(5 * n_cells, dtype=float)
        for cell in range(max(0, face_index - 2), min(n_cells, face_index + 2)):
            for component in range(5):
                column = 5 * cell + component
                increment = np.zeros(5 * n_cells, dtype=float)
                increment[column] = step * primitive_scales[column]
                plus = causal_five_field_rusanov_control_diagnostics(
                    initial["context"],
                    (primitives + increment).reshape(n_cells, 5),
                )
                minus = causal_five_field_rusanov_control_diagnostics(
                    initial["context"],
                    (primitives - increment).reshape(n_cells, 5),
                )
                plus_speeds = np.asarray(
                    plus["candidate_absolute_speeds_over_c"], dtype=float
                )
                minus_speeds = np.asarray(
                    minus["candidate_absolute_speeds_over_c"], dtype=float
                )
                plus_gap = (
                    plus_speeds[output_index, control]
                    - plus_speeds[output_index, competitor]
                )
                minus_gap = (
                    minus_speeds[output_index, control]
                    - minus_speeds[output_index, competitor]
                )
                gradient[column] = (plus_gap - minus_gap) / (2.0 * step)
        maximum_defect = max(
            maximum_defect,
            _relative_defect(gradient, cached_right[:, branch]),
        )
    return maximum_defect


def _final_generator_case(
    initial: dict,
    vector: np.ndarray,
    operator_arrays: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict]:
    n_cells = initial["state"].n_cells
    state = unpack_causal_five_field_state(vector, n_cells)
    primitives = np.asarray(state.primitives, dtype=float).ravel()
    primitive_scales = np.asarray(
        operator_arrays["primitive_column_scales"], dtype=float
    )
    row_scales = np.asarray(
        operator_arrays["conservation_row_scales"], dtype=float
    )
    backend = {
        "mapped_storage_backend": "branch_frozen_local",
        "branch_frozen_local_difference_step": LOCAL_DIFFERENCE_STEP,
    }
    base = causal_five_field_scaled_primitive_vector_field(
        initial["context"],
        primitives,
        primitive_column_scales=primitive_scales,
        conservation_row_scales=row_scales,
        finite_difference_step=wp10c8j.BASE_INNER_DIFFERENCE_STEP,
        storage_quadrature_order=wp10c8j.STORAGE_QUADRATURE_ORDER,
        storage_directional_step=wp10c8j.STORAGE_DIRECTIONAL_STEP,
        **backend,
    )
    physical_rate = primitive_scales * np.asarray(
        base["scaled_primitive_rate_per_s"], dtype=float
    ).ravel()
    mapped = causal_five_field_branch_frozen_mapped_storage_derivatives(
        initial["context"],
        primitives,
        physical_rate,
        primitive_column_scales=primitive_scales,
        conservation_row_scales=row_scales,
        local_difference_step=LOCAL_DIFFERENCE_STEP,
    )
    height = causal_five_field_reduced_storage_rate_derivatives(
        initial["context"],
        primitives,
        physical_rate,
        primitive_column_scales=primitive_scales,
        conservation_row_scales=row_scales,
        storage_matrix_difference_step=wp10c8j.BASE_INNER_DIFFERENCE_STEP,
        storage_rate_derivative_step=wp10c8l_a.STORAGE_RATE_DERIVATIVE_STEP,
        storage_difference_step=wp10c8j.BASE_VERTICAL_ACTION_DIFFERENCE_STEP,
        storage_quadrature_order=wp10c8j.STORAGE_QUADRATURE_ORDER,
        storage_directional_step=wp10c8j.STORAGE_DIRECTIONAL_STEP,
        backend="direct_action",
    )
    mass = np.asarray(
        mapped["conserved_descriptor_reduced_scaled_matrix"], dtype=float
    ) + np.asarray(
        base["vertical_descriptor_reduced_scaled_matrix"], dtype=float
    )
    storage_rate = np.asarray(
        mapped["conserved_storage_rate_derivative_scaled_matrix"], dtype=float
    ) + np.asarray(
        height["vertical_storage_rate_derivative_scaled_matrix"], dtype=float
    )
    assembled = causal_five_field_assemble_evolving_tangent(
        mass,
        np.asarray(operator_arrays["stationary_jacobian"], dtype=float),
        storage_rate,
    )
    generator = np.asarray(
        assembled["evolving_scaled_generator_per_s"], dtype=float
    )

    old_mass = np.asarray(
        operator_arrays["direct_vector_storage_descriptor"], dtype=float
    )
    old_left = np.asarray(
        operator_arrays["production_rusanov_kink_generator_left_factors"],
        dtype=float,
    )
    right = np.asarray(
        operator_arrays["production_rusanov_kink_generator_right_factors"],
        dtype=float,
    )
    faces = np.asarray(
        operator_arrays["production_rusanov_kink_face_indices"], dtype=int
    )
    stationary_left = -(old_mass @ old_left)
    new_left = -np.linalg.solve(mass, stationary_left)
    generator_factor_defect = _relative_defect(
        mass @ new_left,
        -stationary_left,
    )

    physical_left = np.asarray(
        operator_arrays["production_rusanov_kink_physical_flux_left_factors"],
        dtype=float,
    )
    reconstructed_stationary = np.zeros_like(stationary_left)
    for branch, face in enumerate(faces):
        unscaled = physical_left[:, branch] / C
        left_rows = slice(5 * (face - 1), 5 * face)
        right_rows = slice(5 * face, 5 * (face + 1))
        reconstructed_stationary[left_rows, branch] += (
            unscaled / row_scales[left_rows]
        )
        reconstructed_stationary[right_rows, branch] -= (
            unscaled / row_scales[right_rows]
        )
    flux_identity_defect = _relative_defect(
        stationary_left,
        reconstructed_stationary,
    )
    audit, identities = _targeted_candidate_identities(initial, primitives)
    identity_faces = np.asarray([item[1] for item in identities], dtype=int)
    candidate_identity_passed = bool(
        len(identities) == right.shape[1]
        and np.array_equal(identity_faces, faces)
    )
    candidate_gradient_defect = (
        _candidate_gradient_defect(
            initial,
            primitives,
            primitive_scales,
            audit,
            identities,
            right,
        )
        if candidate_identity_passed
        else float("inf")
    )
    factors_passed = bool(
        generator_factor_defect <= MAXIMUM_FACTOR_IDENTITY_RELATIVE_DEFECT
        and flux_identity_defect <= MAXIMUM_FACTOR_IDENTITY_RELATIVE_DEFECT
        and candidate_identity_passed
        and candidate_gradient_defect
        <= MAXIMUM_CANDIDATE_GRADIENT_RELATIVE_DEFECT
    )
    updated = dict(operator_arrays)
    updated["dynamic"] = generator
    updated["wp10c8m_descriptor_reduced_scaled_matrix"] = mass
    updated["production_rusanov_kink_generator_left_factors"] = new_left
    return updated, {
        "cached_branch_count": int(right.shape[1]),
        "cached_face_count": int(np.unique(faces).size),
        "candidate_identities": [
            {
                "diagnostic_row": row,
                "face_index": face,
                "competitor_code": competitor,
            }
            for row, face, competitor in identities
        ],
        "candidate_identity_order_matches_parent": candidate_identity_passed,
        "maximum_candidate_gradient_relative_defect": (
            candidate_gradient_defect
        ),
        "maximum_generator_factor_identity_relative_defect": (
            generator_factor_defect
        ),
        "maximum_flux_factor_identity_relative_defect": flux_identity_defect,
        "maximum_generator_factorization_defect": float(
            assembled["maximum_scaled_generator_factorization_defect"]
        ),
        "factor_contract_passed": factors_passed,
    }


def _preflight_case(
    case_id: str,
    initial: dict,
    vector: np.ndarray,
    operator_arrays: dict[str, np.ndarray],
    operator_metadata: dict,
    operator_provenance: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    arrays, factor_audit = _final_generator_case(
        initial, vector, operator_arrays
    )
    response, gates, names, blocks = wp10c8i._response_stack(
        arrays, operator_metadata, LEVEL_INDEX
    )
    constraints = np.asarray(arrays[f"level_{LEVEL_INDEX}_constraints"])
    basis_audit = causal_weighted_constraint_null_basis(
        constraints,
        state_weights=np.asarray(arrays["state_weights"]),
    )
    basis = np.asarray(basis_audit.basis, dtype=float)
    left = np.asarray(
        arrays["production_rusanov_kink_generator_left_factors"], dtype=float
    )
    right = np.asarray(
        arrays["production_rusanov_kink_generator_right_factors"], dtype=float
    )
    faces = np.asarray(
        arrays["production_rusanov_kink_face_indices"], dtype=int
    )
    direct = wp10c8i._rusanov_kink_instantaneous_output_deltas(
        arrays, operator_metadata, LEVEL_INDEX
    )
    rows: dict[str, dict] = {}
    output_arrays: dict[str, np.ndarray] = {
        f"{case_id}_dynamic": np.asarray(arrays["dynamic"], dtype=float),
        f"{case_id}_constraint_null_basis": basis,
        f"{case_id}_generator_left_factors": left,
        f"{case_id}_generator_right_factors": right,
        f"{case_id}_branch_face_indices": faces,
    }
    case_feasible = bool(factor_audit["factor_contract_passed"])
    for horizon in LOCKED_HORIZONS_SECONDS:
        key = f"{horizon:.3e}"
        rows[key] = {}
        for panels in LOCKED_TIME_PANELS:
            result = rusanov_structured_zero_remainder_preflight(
                base_generator_per_s=np.asarray(arrays["dynamic"]),
                output_operator=response,
                generator_left_factors=left,
                generator_right_factors=right,
                branch_face_indices=faces,
                initial_basis=basis,
                horizon_seconds=horizon,
                output_gates=gates,
                direct_output_deltas=direct,
                time_steps=panels,
                maximum_gate_fraction=ALLOWED_MAXIMUM_GATE_FRACTION,
            )
            fractions = np.asarray(result.per_output_gate_fractions)
            row = result.as_dict()
            control = int(np.argmax(fractions))
            row["controlling_output"] = names[control]
            row["controlling_output_index"] = control
            row.pop("per_output_dynamic_bounds")
            row.pop("per_output_direct_bounds")
            row.pop("per_output_total_bounds")
            row.pop("per_output_gate_fractions")
            rows[key][str(panels)] = row
            prefix = f"{case_id}_h_{key}_p_{panels}"
            output_arrays[f"{prefix}_dynamic_bounds"] = (
                result.per_output_dynamic_bounds
            )
            output_arrays[f"{prefix}_direct_bounds"] = (
                result.per_output_direct_bounds
            )
            output_arrays[f"{prefix}_gate_fractions"] = fractions
        case_feasible = bool(
            case_feasible
            and rows[key]["128"]["maximum_gate_fraction"]
            <= ALLOWED_MAXIMUM_GATE_FRACTION
        )
    return {
        "operator_provenance": operator_provenance,
        "nominal_generator_source": "wp10c8m_branch_frozen_mapped_storage",
        "candidate_scope": "cached_consequential_branches_only",
        "coordinate_level_index": LEVEL_INDEX,
        "coordinate_count": int(constraints.shape[0]),
        "constraint_rank": int(basis_audit.constraint_rank),
        "constraint_null_dimension": int(basis.shape[1]),
        "output_blocks": blocks,
        "factor_audit": factor_audit,
        "rows": rows,
        "feasible_under_locked_preflight": case_feasible,
    }, output_arrays


def main() -> None:
    arguments = _arguments()
    output_path = _absolute(arguments.output)
    arrays_path = _absolute(arguments.arrays)
    track_a = []
    expected = (
        "wp10c8m_a_locked_n64_passed",
        "wp10c8m_a_locked_n128_passed",
    )
    for path, decision in zip(TRACK_A_RESULTS, expected, strict=True):
        evidence = json.loads(path.read_text(encoding="utf-8"))
        if evidence.get("decision") != decision:
            raise RuntimeError(f"Track A evidence did not pass: {path}")
        track_a.append(
            {"path": _relative(path), "sha256": _sha256(path), "decision": decision}
        )

    initial_by_mesh, vectors_by_mesh, state_provenance = wp10c8i._load_states()
    shell_edges = np.asarray(
        wp10c8h._common_shell_edges(initial_by_mesh)["five_shell"], dtype=float
    )
    cases: dict[str, dict] = {}
    all_arrays: dict[str, np.ndarray] = {}
    original_runner = wp10c8l_a.THIS_RUNNER
    try:
        wp10c8l_a.THIS_RUNNER = THIS_RUNNER
        for n_cells, label in LOCKED_CASES:
            initial = initial_by_mesh[n_cells]
            vector = vectors_by_mesh[n_cells][label]
            operator_arrays, metadata, provenance = wp10c8l_a._load_parent_operator(
                initial,
                vector,
                shell_edges,
                n_cells=n_cells,
                label=label,
            )
            case_id = f"n{n_cells}_{label}"
            row, arrays = _preflight_case(
                case_id,
                initial,
                vector,
                operator_arrays,
                metadata,
                provenance,
            )
            row["state_provenance"] = state_provenance[str(n_cells)][label]
            cases[case_id] = row
            all_arrays.update(arrays)
    finally:
        wp10c8l_a.THIS_RUNNER = original_runner

    passed = bool(
        all(row["feasible_under_locked_preflight"] for row in cases.values())
    )
    output = {
        "base_commit": BASE_COMMIT,
        "work_package": WORK_PACKAGE,
        "scope": {
            "binding": False,
            "zero_nonlinear_remainder": True,
            "nominal_generator_is_final_track_a_generator": True,
            "weighted_constraint_null_initial_space": True,
            "direct_output_deltas_recomputed": True,
            "per_face_mutual_exclusivity": True,
            "simultaneous_switching_across_faces": True,
            "candidate_scope": "cached_consequential_branches_only",
            "complete_all_face_possible_winner_scope_run": False,
            "finite_neighborhood_contract_run": False,
        },
        "gates": {
            "allowed_maximum_gate_fraction": ALLOWED_MAXIMUM_GATE_FRACTION,
            "maximum_factor_identity_relative_defect": (
                MAXIMUM_FACTOR_IDENTITY_RELATIVE_DEFECT
            ),
            "maximum_candidate_gradient_relative_defect": (
                MAXIMUM_CANDIDATE_GRADIENT_RELATIVE_DEFECT
            ),
            "locked_horizons_seconds": list(LOCKED_HORIZONS_SECONDS),
            "locked_time_panels": list(LOCKED_TIME_PANELS),
        },
        "track_a": track_a,
        "cases": cases,
        "decision": (
            "wp10c8m_b_final_generator_cached_scope_feasible"
            if passed
            else "wp10c8m_b_final_generator_cached_scope_infeasible"
        ),
        "next_action": (
            "build_complete_all_face_possible_winner_zero_remainder_preflight"
            if passed
            else "stop_before_all_face_and_finite_neighborhood_work"
        ),
        "semantics": (
            "Nonbinding zero-remainder feasibility preflight using the final "
            "WP10c8m Track-A generator. Cached branch identities and factors "
            "are independently reconstructed, but all-face coverage and the "
            "finite-neighborhood theorem remain open."
        ),
    }
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **all_arrays)
    output["artifacts"] = {
        "arrays_path": _relative(arrays_path),
        "arrays_sha256": _sha256(arrays_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
