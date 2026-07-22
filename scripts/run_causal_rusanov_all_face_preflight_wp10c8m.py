"""Run the all-face zero-remainder Rusanov feasibility preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import run_causal_moment_sufficiency_audit_wp10c8i as wp10c8i
import run_causal_rusanov_structured_preflight_wp10c8m as wp10c8m_b
import run_causal_shell_closure_preflight_wp10c8h as wp10c8h
import run_causal_tangent_descriptor_wp10c8l as wp10c8l_a
from imri_qpe.constants import C
from imri_qpe.layer3_minidisk_1d import (
    causal_five_field_reconstruct_face_charts,
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
WORK_PACKAGE = "WP10c8m-B2"
THIS_RUNNER = "scripts/run_causal_rusanov_all_face_preflight_wp10c8m.py"
PARENT_RESULT = (
    ROOT / "outputs/tables/causal_rusanov_structured_preflight_wp10c8m.json"
)
LOCKED_CASES = ((64, "t_0"), (64, "t_0p025"))
LEVEL_INDEX = 4
LOCKED_HORIZONS_SECONDS = (1.0e-2, 2.5e-2)
LOCKED_TIME_PANELS = (64, 128)
ALLOWED_MAXIMUM_GATE_FRACTION = 1.0e-2
DEFAULT_OUTPUT = (
    ROOT / "outputs/tables/causal_rusanov_all_face_preflight_wp10c8m.json"
)
DEFAULT_ARRAYS = (
    ROOT / "outputs/tables/causal_rusanov_all_face_preflight_wp10c8m_arrays.npz"
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


def _all_face_branch_factors(
    initial: dict,
    vector: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict]:
    n_cells = initial["state"].n_cells
    state = unpack_causal_five_field_state(vector, n_cells)
    primitives = np.asarray(state.primitives, dtype=float).ravel()
    primitive_scales = np.asarray(arrays["primitive_column_scales"], dtype=float)
    row_scales = np.asarray(arrays["conservation_row_scales"], dtype=float)
    mass = np.asarray(
        arrays["wp10c8m_descriptor_reduced_scaled_matrix"], dtype=float
    )
    base = causal_five_field_rusanov_control_diagnostics(
        initial["context"], primitives.reshape(n_cells, 5)
    )
    faces = np.asarray(base["face_indices"], dtype=int)
    controls = np.asarray(base["control_codes"], dtype=int)
    speeds = np.asarray(base["candidate_absolute_speeds_over_c"], dtype=float)
    jumps = np.asarray(base["conserved_jumps"], dtype=float)
    exact_zero = np.asarray(base["exact_zero_conserved_jump"], dtype=bool)
    expected_faces = np.arange(1, n_cells, dtype=int)
    if not np.array_equal(faces, expected_faces):
        raise RuntimeError("Rusanov diagnostics do not cover each interior face once")
    if speeds.ndim != 2 or speeds.shape[0] != faces.size:
        raise RuntimeError("Rusanov candidate table is invalid")

    step = wp10c8i.RUSANOV_SWITCHING_NORMAL_DIFFERENCE_STEP
    candidate_derivatives = np.zeros(
        (faces.size, speeds.shape[1], 5 * n_cells), dtype=float
    )
    selected_steps = np.empty(5 * n_cells, dtype=float)
    for column in range(5 * n_cells):
        trial_step = step
        for _attempt in range(9):
            increment = np.zeros(5 * n_cells, dtype=float)
            increment[column] = trial_step * primitive_scales[column]
            try:
                plus = causal_five_field_rusanov_control_diagnostics(
                    initial["context"],
                    (primitives + increment).reshape(n_cells, 5),
                )
                minus = causal_five_field_rusanov_control_diagnostics(
                    initial["context"],
                    (primitives - increment).reshape(n_cells, 5),
                )
            except ValueError:
                trial_step *= 0.25
                continue
            plus_reconstruction = causal_five_field_reconstruct_face_charts(
                initial["context"],
                (primitives + increment).reshape(n_cells, 5),
            )
            minus_reconstruction = causal_five_field_reconstruct_face_charts(
                initial["context"],
                (primitives - increment).reshape(n_cells, 5),
            )
            if not (
                np.all(plus_reconstruction.admissibility_factors == 1.0)
                and np.all(minus_reconstruction.admissibility_factors == 1.0)
            ):
                trial_step *= 0.25
                continue
            break
        else:
            raise RuntimeError(
                f"no admissible Rusanov candidate step for column {column}"
            )
        selected_steps[column] = trial_step
        candidate_derivatives[:, :, column] = (
            np.asarray(plus["candidate_absolute_speeds_over_c"], dtype=float)
            - np.asarray(minus["candidate_absolute_speeds_over_c"], dtype=float)
        ) / (2.0 * trial_step)

    left_by_face = np.zeros((5 * n_cells, faces.size), dtype=float)
    physical_by_face = np.zeros((5, faces.size), dtype=float)
    for row, face in enumerate(faces):
        measure = float(initial["context"].grid.face_measures[face])
        physical = C * (-0.5 * measure * jumps[row])
        physical_by_face[:, row] = physical
        stationary = np.zeros(5 * n_cells, dtype=float)
        left_rows = slice(5 * (face - 1), 5 * face)
        right_rows = slice(5 * face, 5 * (face + 1))
        stationary[left_rows] += physical / C / row_scales[left_rows]
        stationary[right_rows] -= physical / C / row_scales[right_rows]
        left_by_face[:, row] = -np.linalg.solve(mass, stationary)

    left_factors: list[np.ndarray] = []
    right_factors: list[np.ndarray] = []
    physical_factors: list[np.ndarray] = []
    factor_faces: list[int] = []
    factor_candidates: list[int] = []
    for row, face in enumerate(faces):
        control = int(controls[row])
        for candidate in range(speeds.shape[1]):
            if candidate == control:
                continue
            left_factors.append(left_by_face[:, row])
            right_factors.append(
                candidate_derivatives[row, control]
                - candidate_derivatives[row, candidate]
            )
            physical_factors.append(physical_by_face[:, row])
            factor_faces.append(int(face))
            factor_candidates.append(candidate)
    left = np.column_stack(left_factors)
    right = np.column_stack(right_factors)
    physical = np.column_stack(physical_factors)
    factor_faces_array = np.asarray(factor_faces, dtype=int)
    factor_candidates_array = np.asarray(factor_candidates, dtype=int)
    per_face_counts = np.asarray(
        [np.count_nonzero(factor_faces_array == face) for face in faces],
        dtype=int,
    )
    factorization_residual = mass @ left
    stationary_reconstructed = np.zeros_like(factorization_residual)
    for column, face in enumerate(factor_faces_array):
        left_rows = slice(5 * (face - 1), 5 * face)
        right_rows = slice(5 * face, 5 * (face + 1))
        stationary_reconstructed[left_rows, column] += (
            physical[:, column] / C / row_scales[left_rows]
        )
        stationary_reconstructed[right_rows, column] -= (
            physical[:, column] / C / row_scales[right_rows]
        )
    factor_scale = max(
        float(np.max(np.abs(stationary_reconstructed), initial=0.0)),
        np.finfo(float).tiny,
    )
    factor_defect = float(
        np.max(
            np.abs(factorization_residual + stationary_reconstructed),
            initial=0.0,
        )
        / factor_scale
    )
    result = dict(arrays)
    result["production_rusanov_kink_generator_left_factors"] = left
    result["production_rusanov_kink_generator_right_factors"] = right
    result["production_rusanov_kink_physical_flux_left_factors"] = physical
    result["production_rusanov_kink_face_indices"] = factor_faces_array
    result["production_rusanov_kink_competitor_codes"] = factor_candidates_array
    return result, {
        "interior_face_count": int(faces.size),
        "candidate_count_per_face": int(speeds.shape[1]),
        "alternative_factor_count": int(left.shape[1]),
        "minimum_alternative_count_per_face": int(np.min(per_face_counts)),
        "maximum_alternative_count_per_face": int(np.max(per_face_counts)),
        "exact_zero_jump_face_count": int(np.count_nonzero(exact_zero)),
        "zero_generator_left_factor_count": int(
            np.count_nonzero(np.linalg.norm(left, axis=0) == 0.0)
        ),
        "zero_generator_right_factor_count": int(
            np.count_nonzero(np.linalg.norm(right, axis=0) == 0.0)
        ),
        "base_candidate_difference_step": step,
        "minimum_selected_candidate_difference_step": float(
            np.min(selected_steps)
        ),
        "reduced_step_column_count": int(
            np.count_nonzero(selected_steps < step)
        ),
        "maximum_generator_factorization_relative_defect": factor_defect,
        "factorization_passed": bool(factor_defect <= 1.0e-10),
    }


def _case(
    case_id: str,
    initial: dict,
    vector: np.ndarray,
    parent_arrays: dict[str, np.ndarray],
    metadata: dict,
    provenance: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    final_arrays, cached_audit = wp10c8m_b._final_generator_case(
        initial, vector, parent_arrays
    )
    arrays, all_face_audit = _all_face_branch_factors(
        initial, vector, final_arrays
    )
    response, gates, names, blocks = wp10c8i._response_stack(
        arrays, metadata, LEVEL_INDEX
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
        arrays, metadata, LEVEL_INDEX
    )
    rows: dict[str, dict] = {}
    result_arrays: dict[str, np.ndarray] = {
        f"{case_id}_dynamic": np.asarray(arrays["dynamic"], dtype=float),
        f"{case_id}_constraint_null_basis": basis,
        f"{case_id}_generator_left_factors": left,
        f"{case_id}_generator_right_factors": right,
        f"{case_id}_branch_face_indices": faces,
        f"{case_id}_branch_candidate_indices": np.asarray(
            arrays["production_rusanov_kink_competitor_codes"], dtype=int
        ),
    }
    feasible = bool(
        cached_audit["factor_contract_passed"]
        and all_face_audit["factorization_passed"]
    )
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
            result_arrays[f"{prefix}_gate_fractions"] = fractions
        feasible = bool(
            feasible
            and rows[key]["128"]["maximum_gate_fraction"]
            <= ALLOWED_MAXIMUM_GATE_FRACTION
        )
    return {
        "operator_provenance": provenance,
        "coordinate_count": int(constraints.shape[0]),
        "constraint_rank": int(basis_audit.constraint_rank),
        "constraint_null_dimension": int(basis.shape[1]),
        "cached_factor_audit": cached_audit,
        "all_face_factor_audit": all_face_audit,
        "output_blocks": blocks,
        "rows": rows,
        "feasible_under_locked_preflight": feasible,
    }, result_arrays


def main() -> None:
    arguments = _arguments()
    output_path = _absolute(arguments.output)
    arrays_path = _absolute(arguments.arrays)
    parent = json.loads(PARENT_RESULT.read_text(encoding="utf-8"))
    if parent.get("decision") != "wp10c8m_b_final_generator_cached_scope_feasible":
        raise RuntimeError("cached-scope WP10c8m-B did not authorize all-face work")
    initial_by_mesh, vectors_by_mesh, state_provenance = wp10c8i._load_states()
    shell_edges = np.asarray(
        wp10c8h._common_shell_edges(initial_by_mesh)["five_shell"], dtype=float
    )
    cases: dict[str, dict] = {}
    all_arrays: dict[str, np.ndarray] = {}
    original_runner = wp10c8l_a.THIS_RUNNER
    original_children = wp10c8l_a.WP10C8M_RUNNERS
    try:
        wp10c8l_a.THIS_RUNNER = THIS_RUNNER
        wp10c8l_a.WP10C8M_RUNNERS = tuple(
            dict.fromkeys((*original_children, THIS_RUNNER))
        )
        for n_cells, label in LOCKED_CASES:
            initial = initial_by_mesh[n_cells]
            vector = vectors_by_mesh[n_cells][label]
            parent_arrays, metadata, provenance = wp10c8l_a._load_parent_operator(
                initial,
                vector,
                shell_edges,
                n_cells=n_cells,
                label=label,
            )
            case_id = f"n{n_cells}_{label}"
            row, arrays = _case(
                case_id,
                initial,
                vector,
                parent_arrays,
                metadata,
                provenance,
            )
            row["state_provenance"] = state_provenance[str(n_cells)][label]
            cases[case_id] = row
            all_arrays.update(arrays)
    finally:
        wp10c8l_a.THIS_RUNNER = original_runner
        wp10c8l_a.WP10C8M_RUNNERS = original_children
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
            "candidate_scope": "all_noncontrolling_candidates_all_interior_faces",
            "complete_all_face_possible_winner_scope_run": True,
            "finite_neighborhood_contract_run": False,
        },
        "gates": {
            "allowed_maximum_gate_fraction": ALLOWED_MAXIMUM_GATE_FRACTION,
            "locked_horizons_seconds": list(LOCKED_HORIZONS_SECONDS),
            "locked_time_panels": list(LOCKED_TIME_PANELS),
        },
        "parent": {
            "path": _relative(PARENT_RESULT),
            "sha256": _sha256(PARENT_RESULT),
            "decision": parent["decision"],
        },
        "cases": cases,
        "decision": (
            "wp10c8m_b2_all_face_zero_remainder_feasible"
            if passed
            else "wp10c8m_b2_all_face_zero_remainder_infeasible"
        ),
        "next_action": (
            "build_finite_neighborhood_gap_remainder_and_containment_contract"
            if passed
            else "reject_current_structured_certificate_architecture_not_flux"
        ),
        "semantics": (
            "Nonbinding feasibility preflight that includes every "
            "noncontrolling characteristic candidate at every N64 interior "
            "face. Exact-zero jump factors are zero only at the anchor; their "
            "finite-amplitude variation remains part of the open nonlinear "
            "remainder contract."
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
