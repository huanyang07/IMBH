"""Recertify the locked WP10c8j smooth tangent with the WP10c8k repair.

The selected diagnostic candidate changes only the mapped-conserved direct
storage-action path to a centered derivative at a 0.0128 scaled primitive
increment.  The responsive-height storage derivative, stationary Jacobian,
nonlinear vector-field secants, truth states, Rusanov operator, moments, and
gates remain immutable parent evidence.  No trajectory or reduced evolution
is run here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import run_causal_moment_sufficiency_audit_wp10c8i as wp10c8i
import run_causal_shell_closure_preflight_wp10c8h as wp10c8h
import run_causal_tangent_certification_wp10c8j as wp10c8j
import run_causal_tangent_localization_wp10c8k as wp10c8k
from imri_qpe.layer3_minidisk_1d import (
    causal_five_field_assemble_evolving_tangent,
    causal_five_field_reduced_storage_rate_derivatives,
    unpack_causal_five_field_state,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_rusanov_certification import (
    certify_rusanov_finite_neighborhood,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = wp10c8k.BASE_COMMIT
WORK_PACKAGE = "WP10c8k"
SELECTED_CASES = (
    (64, "t_0p05", "construction"),
    (128, "t_0p10", "held_out"),
    (64, "t_0", "construction"),
    (64, "t_0p025", "construction"),
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/tables/causal_tangent_recertification_wp10c8k.json"
)
DEFAULT_ARRAYS = (
    ROOT / "outputs/tables/causal_tangent_recertification_wp10c8k_arrays.npz"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--arrays", type=Path, default=DEFAULT_ARRAYS)
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(f"n{n}_{label}" for n, label, _ in SELECTED_CASES),
        help="Select one locked failed-anchor case; repeat as needed.",
    )
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


def _rescore_parent_secants(
    generator: np.ndarray,
    arrays: dict[str, np.ndarray],
    metadata: dict,
) -> dict:
    parent = metadata["separated_tangent"][
        "fresh_full_direction_independent_vector_field_jvp"
    ]
    direction_rows = {}
    maximum_centered = 0.0
    maximum_centered_infinity = 0.0
    for name in parent["binding_smooth_direction_names"]:
        parent_row = parent["directions"][name]
        steps = {}
        for step_key, parent_step in parent_row["steps"].items():
            prefix = f"fresh_{name}_step_{step_key}"
            direction = np.asarray(
                arrays[f"{prefix}_independent_vector_field_jvp_direction"],
                dtype=float,
            )
            predicted = generator @ direction
            direct = np.asarray(
                arrays[f"{prefix}_independent_vector_field_jvp_direct"],
                dtype=float,
            )
            forward = np.asarray(
                arrays[f"{prefix}_independent_vector_field_jvp_forward"],
                dtype=float,
            )
            backward = np.asarray(
                arrays[f"{prefix}_independent_vector_field_jvp_backward"],
                dtype=float,
            )
            centered = wp10c8i._jvp_defect(
                predicted,
                direct,
                relative_tolerance=(
                    wp10c8j.MAXIMUM_INDEPENDENT_VECTOR_FIELD_JVP_RELATIVE_DEFECT
                ),
            )
            forward_defect = wp10c8i._jvp_defect(
                predicted,
                forward,
                relative_tolerance=(
                    wp10c8j.MAXIMUM_FORWARD_BACKWARD_JVP_RELATIVE_DEFECT
                ),
            )
            backward_defect = wp10c8i._jvp_defect(
                predicted,
                backward,
                relative_tolerance=(
                    wp10c8j.MAXIMUM_FORWARD_BACKWARD_JVP_RELATIVE_DEFECT
                ),
            )
            maximum_centered = max(
                maximum_centered,
                float(centered["relative_l2_defect"]),
            )
            maximum_centered_infinity = max(
                maximum_centered_infinity,
                float(centered["relative_infinity_defect"]),
            )
            steps[step_key] = {
                "parent_smooth_contract_passed": bool(
                    parent_step[
                        "plus_minus_reconstruction_differentiable"
                    ]
                    and parent_step[
                        "plus_minus_outer_active_set_unchanged"
                    ]
                ),
                "centered_jvp_defect": centered,
                "forward_jvp_defect": forward_defect,
                "backward_jvp_defect": backward_defect,
                "passed": bool(
                    parent_step[
                        "plus_minus_reconstruction_differentiable"
                    ]
                    and parent_step[
                        "plus_minus_outer_active_set_unchanged"
                    ]
                    and centered["passed"]
                    and forward_defect["passed"]
                    and backward_defect["passed"]
                ),
            }
        direction_rows[name] = {
            "steps": steps,
            "parent_secant_step_stability_passed": bool(
                parent_row["central_secant_step_stability_passed"]
            ),
            "passed": bool(
                parent_row["central_secant_step_stability_passed"]
                and all(row["passed"] for row in steps.values())
            ),
        }
    passed = bool(
        not parent["hard_nonsmooth_direction_names"]
        and direction_rows
        and all(row["passed"] for row in direction_rows.values())
    )
    return {
        "binding_smooth_directions": direction_rows,
        "rusanov_reserved_direction_names": parent[
            "rusanov_reserved_direction_names"
        ],
        "hard_nonsmooth_direction_names": parent[
            "hard_nonsmooth_direction_names"
        ],
        "maximum_centered_jvp_relative_defect": maximum_centered,
        "maximum_centered_jvp_relative_infinity_defect": (
            maximum_centered_infinity
        ),
        "passed": passed,
    }


def _selected_storage_rate(
    initial: dict,
    vector: np.ndarray,
    operator_arrays: dict[str, np.ndarray],
    certification_arrays: dict[str, np.ndarray],
    *,
    n_cells: int,
    label: str,
) -> dict:
    if (
        n_cells == wp10c8k.LOCKED_RESOLUTION
        and label == wp10c8k.LOCKED_ANCHOR
        and wp10c8k.DEFAULT_OUTPUT.exists()
        and wp10c8k.DEFAULT_ARRAYS.exists()
    ):
        localization = json.loads(
            wp10c8k.DEFAULT_OUTPUT.read_text(encoding="utf-8")
        )
        scope = localization.get("scope", {})
        if (
            scope.get("selected_repaired_storage_difference_step")
            == wp10c8k.SELECTED_REPAIRED_STORAGE_DIFFERENCE_STEP
            and scope.get("mapped_conserved_difference_order")
            == wp10c8k.SELECTED_REPAIRED_CONSERVED_DIFFERENCE_ORDER
            and localization.get("artifacts", {}).get("arrays_sha256")
            == _sha256(wp10c8k.DEFAULT_ARRAYS)
        ):
            cached = np.load(wp10c8k.DEFAULT_ARRAYS, allow_pickle=False)
            conserved = np.asarray(
                cached["selected_conserved_storage_rate_derivative"],
                dtype=float,
            )
            vertical = np.asarray(
                cached["selected_vertical_storage_rate_derivative"],
                dtype=float,
            )
            return {
                "total": conserved + vertical,
                "conserved": conserved,
                "vertical": vertical,
                "candidate_raw_vertical": vertical,
                "source": "validated_wp10c8k_localization_cache",
            }
    state = unpack_causal_five_field_state(
        vector, int(initial["state"].n_cells)
    )
    primitive_scales = np.asarray(
        operator_arrays["primitive_column_scales"], dtype=float
    )
    physical_rate = (
        primitive_scales
        * np.asarray(operator_arrays["scaled_primitive_rate"], dtype=float)
    )
    candidate = causal_five_field_reduced_storage_rate_derivatives(
        initial["context"],
        np.asarray(state.primitives, dtype=float).ravel(),
        physical_rate,
        primitive_column_scales=primitive_scales,
        conservation_row_scales=np.asarray(
            operator_arrays["conservation_row_scales"], dtype=float
        ),
        storage_matrix_difference_step=wp10c8j.BASE_INNER_DIFFERENCE_STEP,
        storage_rate_derivative_step=wp10c8j.BASE_OUTER_DIFFERENCE_STEP,
        storage_difference_step=(
            wp10c8k.SELECTED_REPAIRED_STORAGE_DIFFERENCE_STEP
        ),
        storage_quadrature_order=wp10c8j.STORAGE_QUADRATURE_ORDER,
        storage_directional_step=wp10c8j.STORAGE_DIRECTIONAL_STEP,
        conserved_difference_order=(
            wp10c8k.SELECTED_REPAIRED_CONSERVED_DIFFERENCE_ORDER
        ),
        backend="direct_action",
    )
    conserved = np.asarray(
        candidate["conserved_storage_rate_derivative_scaled_matrix"],
        dtype=float,
    )
    vertical = np.asarray(
        certification_arrays["repaired_vertical_storage_rate_derivative"],
        dtype=float,
    )
    return {
        "total": conserved + vertical,
        "conserved": conserved,
        "vertical": vertical,
        "candidate_raw_vertical": np.asarray(
            candidate["vertical_storage_rate_derivative_scaled_matrix"],
            dtype=float,
        ),
        "source": "fresh_wp10c8k_direct_action",
    }


def _optimistic_rusanov_feasibility(
    generator: np.ndarray,
    operator_arrays: dict[str, np.ndarray],
    operator_metadata: dict,
) -> dict:
    """Evaluate the existing enclosure with perfect missing certificates.

    Candidate coverage, neighborhood validity, and nonlinear remainders are
    set to their mathematically best possible values.  The resulting bound is
    a lower bound for this *particular triangle/logarithmic-norm enclosure*:
    real candidate additions and nonzero remainders can only increase it.
    This does not certify the Rusanov operator; it decides whether completing
    the current certificate inputs could possibly meet the locked gate.
    """

    left = np.asarray(
        operator_arrays[
            "production_rusanov_kink_generator_left_factors"
        ],
        dtype=float,
    )
    right = np.asarray(
        operator_arrays[
            "production_rusanov_kink_generator_right_factors"
        ],
        dtype=float,
    )
    branch_count = int(left.shape[1])
    levels = {}
    maximum_fraction = 0.0
    for level_index, level in enumerate(operator_metadata["levels"]):
        output, gates, names, _blocks = wp10c8i._response_stack(
            operator_arrays,
            operator_metadata,
            level_index,
        )
        direct = wp10c8i._rusanov_kink_instantaneous_output_deltas(
            operator_arrays,
            operator_metadata,
            level_index,
        )
        horizon_rows = {}
        for horizon in wp10c8j.LOCKED_FINITE_TIME_HORIZONS_SECONDS:
            certificate = certify_rusanov_finite_neighborhood(
                base_generator_per_s=generator,
                output_operator=output,
                generator_left_factors=left,
                generator_right_factors=right,
                horizon_seconds=horizon,
                output_gates=gates,
                direct_output_deltas=direct,
                coefficient_bounds=np.ones(branch_count, dtype=float),
                state_metric_diagonal=np.asarray(
                    operator_arrays["state_weights"], dtype=float
                ),
                initial_state_radius=1.0,
                neighborhood_bounds_global=True,
                nonlinear_remainder_rate=0.0,
                nonlinear_output_remainder_bounds=np.zeros_like(gates),
                candidate_coverage_certified=True,
                nonlinear_remainder_certified=True,
                maximum_gate_fraction=(
                    wp10c8j.MAXIMUM_RUSANOV_FINITE_TIME_GATE_FRACTION
                ),
            )
            row = certificate.as_dict()
            fractions = np.asarray(
                row["per_output_gate_fractions"], dtype=float
            )
            controlling = int(np.argmax(fractions)) if fractions.size else 0
            row["controlling_output"] = (
                names[controlling] if names else None
            )
            maximum_fraction = max(
                maximum_fraction,
                float(row["maximum_gate_fraction"] or 0.0),
            )
            horizon_rows[f"{horizon:.6g}"] = row
        levels[level["name"]] = horizon_rows
    return {
        "semantics": (
            "optimistic zero-remainder/global-neighborhood lower bound for "
            "the existing aggregate logarithmic-norm enclosure; not a "
            "Rusanov certificate"
        ),
        "consequential_cached_branch_count": branch_count,
        "levels": levels,
        "maximum_gate_fraction": maximum_fraction,
        "allowed_maximum_gate_fraction": (
            wp10c8j.MAXIMUM_RUSANOV_FINITE_TIME_GATE_FRACTION
        ),
        "current_enclosure_feasible": bool(
            maximum_fraction
            <= wp10c8j.MAXIMUM_RUSANOV_FINITE_TIME_GATE_FRACTION
        ),
    }


def main() -> None:
    arguments = _arguments()
    selected_names = set(
        arguments.case
        or tuple(f"n{n}_{label}" for n, label, _ in SELECTED_CASES)
    )
    selected_cases = tuple(
        row
        for row in SELECTED_CASES
        if f"n{row[0]}_{row[1]}" in selected_names
    )
    wp10c8j._locked_contract()
    wp10c8j._validate_authorization()
    initial_by_mesh, vectors_by_mesh, state_provenance = wp10c8i._load_states()
    shell_edges_rg = np.asarray(
        wp10c8h._common_shell_edges(initial_by_mesh)["five_shell"],
        dtype=float,
    )

    rows = {}
    array_payload = {}
    provenance = {}
    for n_cells, label, _role in selected_cases:
        key = f"n{n_cells}_{label}"
        initial = initial_by_mesh[n_cells]
        vector = vectors_by_mesh[n_cells][label]
        operator_arrays, _operator_metadata, operator_provenance = (
            wp10c8k._load_parent_operator_evidence(
                initial,
                vector,
                shell_edges_rg,
                n_cells=n_cells,
                label=label,
            )
        )
        certification_path = wp10c8j._cache_path(n_cells, label)
        certification_arrays, certification_metadata = (
            wp10c8k._load_parent_certification_evidence(
                certification_path,
                vector=vector,
                operator_provenance=operator_provenance,
                n_cells=n_cells,
                label=label,
            )
        )
        storage = _selected_storage_rate(
            initial,
            vector,
            operator_arrays,
            certification_arrays,
            n_cells=n_cells,
            label=label,
        )
        mass = np.asarray(
            operator_arrays["direct_vector_storage_descriptor"], dtype=float
        )
        stationary = np.asarray(
            operator_arrays["stationary_jacobian"], dtype=float
        )
        assembled = causal_five_field_assemble_evolving_tangent(
            mass,
            stationary,
            storage["total"],
        )
        generator = np.asarray(
            assembled["evolving_scaled_generator_per_s"], dtype=float
        )
        smooth = _rescore_parent_secants(
            generator,
            certification_arrays,
            certification_metadata,
        )
        optimistic_rusanov = _optimistic_rusanov_feasibility(
            generator,
            operator_arrays,
            _operator_metadata,
        )
        component_defect = storage["total"] - (
            storage["conserved"] + storage["vertical"]
        )
        rows[key] = {
            "n_cells": n_cells,
            "anchor": label,
            "selected_mapped_storage_difference_step": (
                wp10c8k.SELECTED_REPAIRED_STORAGE_DIFFERENCE_STEP
            ),
            "selected_mapped_conserved_difference_order": (
                wp10c8k.SELECTED_REPAIRED_CONSERVED_DIFFERENCE_ORDER
            ),
            "responsive_height_derivative_source": (
                "immutable_wp10c8j_selected_vertical_derivative"
            ),
            "selected_storage_rate_source": storage["source"],
            "maximum_storage_component_reconstruction_defect": float(
                np.max(np.abs(component_defect))
            ),
            "maximum_generator_factorization_defect": float(
                assembled["maximum_scaled_generator_factorization_defect"]
            ),
            "smooth_vector_field_contract": smooth,
            "optimistic_rusanov_enclosure_feasibility": (
                optimistic_rusanov
            ),
            "passed": bool(
                np.max(np.abs(component_defect))
                <= wp10c8j.MAXIMUM_STORAGE_COMPONENT_RECONSTRUCTION_DEFECT
                and assembled["maximum_scaled_generator_factorization_defect"]
                <= wp10c8j.MAXIMUM_GENERATOR_FACTORIZATION_DEFECT
                and smooth["passed"]
            ),
        }
        provenance[key] = {
            "state": state_provenance[str(n_cells)][label],
            "operator": operator_provenance,
            "certification_path": _relative(certification_path),
            "certification_sha256": _sha256(certification_path),
        }
        for name, values in {
            "selected_storage_rate_derivative": storage["total"],
            "selected_conserved_storage_rate_derivative": storage[
                "conserved"
            ],
            "preserved_vertical_storage_rate_derivative": storage[
                "vertical"
            ],
            "raw_candidate_vertical_storage_rate_derivative": storage[
                "candidate_raw_vertical"
            ],
            "selected_dynamic": generator,
        }.items():
            array_payload[f"{key}_{name}"] = values
        print(
            json.dumps(
                {
                    "work_package": WORK_PACKAGE,
                    "phase": "smooth_tangent_recertification",
                    "case": key,
                    "maximum_centered_jvp_relative_defect": smooth[
                        "maximum_centered_jvp_relative_defect"
                    ],
                    "passed": rows[key]["passed"],
                    "optimistic_rusanov_maximum_gate_fraction": (
                        optimistic_rusanov["maximum_gate_fraction"]
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    complete = len(selected_cases) == len(SELECTED_CASES)
    all_passed = bool(
        complete and rows and all(row["passed"] for row in rows.values())
    )
    rusanov_enclosure_feasible = bool(
        complete
        and rows
        and all(
            row["optimistic_rusanov_enclosure_feasibility"][
                "current_enclosure_feasible"
            ]
            for row in rows.values()
        )
    )
    arrays_path = _absolute(arguments.arrays)
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **array_payload)
    output = {
        "work_package": WORK_PACKAGE,
        "base_commit": BASE_COMMIT,
        "decision": (
            "wp10c8k_smooth_tangent_recertified"
            if all_passed
            else (
                "wp10c8k_partial_smooth_recertification_completed"
                if not complete
                else "wp10c8k_smooth_tangent_recertification_failed"
            )
        ),
        "next_authorization": (
            "populate_binding_rusanov_candidate_and_remainder_contracts"
            if all_passed and rusanov_enclosure_feasible
            else (
                "replace_the_overconservative_aggregate_rusanov_enclosure"
                if all_passed
                else "repair_only_the_remaining_failed_smooth_case"
            )
        ),
        "scope": {
            "selected_cases": [
                f"n{n}_{label}" for n, label, _ in selected_cases
            ],
            "complete_locked_failed_anchor_sequence": complete,
            "mapped_storage_difference_step": (
                wp10c8k.SELECTED_REPAIRED_STORAGE_DIFFERENCE_STEP
            ),
            "mapped_conserved_difference_order": (
                wp10c8k.SELECTED_REPAIRED_CONSERVED_DIFFERENCE_ORDER
            ),
            "responsive_height_derivative_changed": False,
            "stationary_jacobian_changed": False,
            "production_vector_field_changed": False,
            "rusanov_operator_changed": False,
            "truth_trajectory_run": False,
            "moment_ladder_changed": False,
        },
        "cases": rows,
        "all_locked_smooth_cases_passed": all_passed,
        "current_aggregate_rusanov_enclosure_feasible": (
            rusanov_enclosure_feasible
        ),
        "provenance": provenance,
        "gates": {
            "maximum_centered_jvp_relative_defect": (
                wp10c8j.MAXIMUM_INDEPENDENT_VECTOR_FIELD_JVP_RELATIVE_DEFECT
            ),
            "maximum_one_sided_jvp_relative_defect": (
                wp10c8j.MAXIMUM_FORWARD_BACKWARD_JVP_RELATIVE_DEFECT
            ),
            "maximum_generator_factorization_defect": (
                wp10c8j.MAXIMUM_GENERATOR_FACTORIZATION_DEFECT
            ),
            "maximum_storage_component_reconstruction_defect": (
                wp10c8j.MAXIMUM_STORAGE_COMPONENT_RECONSTRUCTION_DEFECT
            ),
        },
        "artifacts": {
            "arrays_path": _relative(arrays_path),
            "arrays_sha256": _sha256(arrays_path),
        },
    }
    output_path = _absolute(arguments.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "work_package": WORK_PACKAGE,
                "decision": output["decision"],
                "all_locked_smooth_cases_passed": all_passed,
                "output": _relative(output_path),
                "arrays": _relative(arrays_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
