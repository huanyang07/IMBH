"""Certify the audit-only unified mapped-storage derivative for WP10c8l-A.

The runner changes neither the production BDF storage operator nor the
stationary residual.  It rebuilds the nonlinear primitive vector field and
its mapped-storage tangent from one finite-difference derivative of the same
instantaneous mapped-storage map.  Responsive-height storage remains the
separate path-dependent vector one-form used by WP10c8j/WP10c8k.
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
    causal_five_field_scaled_primitive_vector_field,
    causal_five_field_unified_mapped_storage_derivatives,
    unpack_causal_five_field_state,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "4dc5cea0342d35135e31078669e7e71ba7d16cf9"
WORK_PACKAGE = "WP10c8l-A"
LOCKED_CASES = (
    (64, "t_0p05", "construction"),
    (128, "t_0p10", "held_out"),
)
LOCKED_DIRECTIONS = (
    "density_redistribution_20_to_200rg",
    "thermal_redistribution_60_to_200rg",
)
LOCKED_SECANT_STEPS = (5.0e-4, 1.0e-3, 3.0e-3)
MAPPED_STORAGE_DIFFERENCE_STEP = 1.0e-3
MAPPED_STORAGE_DIFFERENCE_ORDER = 4
STORAGE_RATE_DERIVATIVE_STEP = 1.0e-6
STORAGE_RATE_DERIVATIVE_ORDER = 2
DEFAULT_OUTPUT = (
    ROOT / "outputs/tables/causal_tangent_descriptor_wp10c8l.json"
)
DEFAULT_ARRAYS = (
    ROOT / "outputs/tables/causal_tangent_descriptor_wp10c8l_arrays.npz"
)
THIS_RUNNER = "scripts/run_causal_tangent_descriptor_wp10c8l.py"
WP10C8L_RUNNER = "scripts/run_causal_tangent_descriptor_wp10c8l.py"
WP10C8L_RUSANOV_RUNNER = (
    "scripts/run_causal_rusanov_structured_preflight_wp10c8l.py"
)
WP10C8M_RUNNERS = (
    "scripts/run_causal_tangent_branch_frozen_wp10c8m.py",
    "scripts/run_causal_rusanov_structured_preflight_wp10c8m.py",
    "scripts/run_causal_rusanov_all_face_preflight_wp10c8m.py",
)
RUSANOV_CERTIFICATION_PATH = (
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_rusanov_certification.py"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--arrays", type=Path, default=DEFAULT_ARRAYS)
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(f"n{n}_{label}" for n, label, _ in LOCKED_CASES),
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


def _load_parent_operator(
    initial: dict,
    vector: np.ndarray,
    shell_edges_rg: np.ndarray,
    *,
    n_cells: int,
    label: str,
) -> tuple[dict[str, np.ndarray], dict, dict]:
    """Reuse the immutable parent evidence with explicit child exceptions."""

    original_runners = wp10c8k.WP10C8K_NEW_RUNNER_PATHS
    original_changed = wp10c8k.AUTHORIZED_WP10C8K_CHANGED_PARENT_PATHS
    versioned_source_exists = wp10c8j._operator_source_path(
        n_cells, label
    ).exists()
    try:
        wp10c8k.WP10C8K_NEW_RUNNER_PATHS = tuple(
            dict.fromkeys(
                (*original_runners, WP10C8L_RUNNER, THIS_RUNNER)
                + (WP10C8L_RUSANOV_RUNNER,)
                + WP10C8M_RUNNERS
                + (
                    ()
                    if versioned_source_exists
                    else (RUSANOV_CERTIFICATION_PATH,)
                )
            )
        )
        wp10c8k.AUTHORIZED_WP10C8K_CHANGED_PARENT_PATHS = tuple(
            dict.fromkeys(
                (*original_changed,)
                + (
                    (RUSANOV_CERTIFICATION_PATH,)
                    if versioned_source_exists
                    else ()
                )
            )
        )
        return wp10c8k._load_parent_operator_evidence(
            initial,
            vector,
            shell_edges_rg,
            n_cells=n_cells,
            label=label,
        )
    finally:
        wp10c8k.WP10C8K_NEW_RUNNER_PATHS = original_runners
        wp10c8k.AUTHORIZED_WP10C8K_CHANGED_PARENT_PATHS = original_changed


def _direction_source(
    n_cells: int,
    operator_arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if n_cells == 64:
        evidence = json.loads(
            wp10c8k.DEFAULT_OUTPUT.read_text(encoding="utf-8")
        )
        if evidence["artifacts"]["arrays_sha256"] != _sha256(
            wp10c8k.DEFAULT_ARRAYS
        ):
            raise RuntimeError("WP10c8k localization arrays changed")
        arrays = np.load(wp10c8k.DEFAULT_ARRAYS, allow_pickle=False)
        return {
            name: np.asarray(arrays[f"{name}_direction"], dtype=float)
            for name in LOCKED_DIRECTIONS
        }
    raise ValueError("the WP10c8k direction cache is available only at N64")


def _jvp_rows(
    initial: dict,
    primitives: np.ndarray,
    primitive_scales: np.ndarray,
    row_scales: np.ndarray,
    generator: np.ndarray,
    directions: dict[str, np.ndarray],
    mapped_storage_column_steps: np.ndarray | None,
    *,
    vector_field_kwargs: dict | None = None,
) -> tuple[dict, dict[str, np.ndarray]]:
    backend_kwargs = (
        {
            "mapped_storage_backend": "unified_audit",
            "mapped_storage_difference_step": MAPPED_STORAGE_DIFFERENCE_STEP,
            "mapped_storage_difference_order": MAPPED_STORAGE_DIFFERENCE_ORDER,
            "mapped_storage_column_steps": mapped_storage_column_steps,
        }
        if vector_field_kwargs is None
        else dict(vector_field_kwargs)
    )
    base = causal_five_field_scaled_primitive_vector_field(
        initial["context"],
        primitives,
        primitive_column_scales=primitive_scales,
        conservation_row_scales=row_scales,
        finite_difference_step=wp10c8j.BASE_INNER_DIFFERENCE_STEP,
        storage_quadrature_order=wp10c8j.STORAGE_QUADRATURE_ORDER,
        storage_directional_step=wp10c8j.STORAGE_DIRECTIONAL_STEP,
        **backend_kwargs,
    )
    base_rate = np.asarray(base["scaled_primitive_rate_per_s"], dtype=float).ravel()
    base_branch = wp10c8i._vector_field_branch_state(initial, primitives)
    rows = {}
    arrays = {}
    for name, raw_direction in directions.items():
        direction = wp10c8i._normalized_jvp_direction(
            raw_direction,
            name=name,
        )
        predicted = generator @ direction
        step_rows = {}
        direct_by_step = {}
        for step in LOCKED_SECANT_STEPS:
            plus_primitives = primitives + step * primitive_scales * direction
            minus_primitives = primitives - step * primitive_scales * direction
            plus = causal_five_field_scaled_primitive_vector_field(
                initial["context"],
                plus_primitives,
                primitive_column_scales=primitive_scales,
                conservation_row_scales=row_scales,
                finite_difference_step=wp10c8j.BASE_INNER_DIFFERENCE_STEP,
                storage_quadrature_order=wp10c8j.STORAGE_QUADRATURE_ORDER,
                storage_directional_step=wp10c8j.STORAGE_DIRECTIONAL_STEP,
                **backend_kwargs,
            )
            minus = causal_five_field_scaled_primitive_vector_field(
                initial["context"],
                minus_primitives,
                primitive_column_scales=primitive_scales,
                conservation_row_scales=row_scales,
                finite_difference_step=wp10c8j.BASE_INNER_DIFFERENCE_STEP,
                storage_quadrature_order=wp10c8j.STORAGE_QUADRATURE_ORDER,
                storage_directional_step=wp10c8j.STORAGE_DIRECTIONAL_STEP,
                **backend_kwargs,
            )
            plus_rate = np.asarray(
                plus["scaled_primitive_rate_per_s"], dtype=float
            ).ravel()
            minus_rate = np.asarray(
                minus["scaled_primitive_rate_per_s"], dtype=float
            ).ravel()
            direct = (plus_rate - minus_rate) / (2.0 * step)
            forward = (plus_rate - base_rate) / step
            backward = (base_rate - minus_rate) / step
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
            plus_branch = wp10c8i._vector_field_branch_state(
                initial, plus_primitives
            )
            minus_branch = wp10c8i._vector_field_branch_state(
                initial, minus_primitives
            )
            reconstruction_smooth = bool(
                np.array_equal(plus_branch[1], base_branch[1])
                and np.array_equal(minus_branch[1], base_branch[1])
            )
            outer_active_set_smooth = bool(
                plus_branch[0] == base_branch[0]
                and minus_branch[0] == base_branch[0]
            )
            rusanov = wp10c8i._rusanov_branch_resolution(
                base_branch[2], plus_branch[2], minus_branch[2]
            )
            key = f"{step:.0e}"
            direct_by_step[key] = direct
            step_rows[key] = {
                "centered_jvp_defect": centered,
                "forward_jvp_defect": forward_defect,
                "backward_jvp_defect": backward_defect,
                "reconstruction_branch_unchanged": reconstruction_smooth,
                "outer_active_set_unchanged": outer_active_set_smooth,
                "rusanov_branch_resolution": rusanov,
                "passed": bool(
                    centered["passed"]
                    and forward_defect["passed"]
                    and backward_defect["passed"]
                    and reconstruction_smooth
                    and outer_active_set_smooth
                    and rusanov["passed"]
                ),
            }
            arrays[f"{name}_step_{key}_direct"] = direct
        reference = direct_by_step["1e-03"]
        stability = {
            key: wp10c8i._jvp_defect(
                value,
                reference,
                relative_tolerance=(
                    wp10c8j.MAXIMUM_INDEPENDENT_VECTOR_FIELD_JVP_RELATIVE_DEFECT
                ),
            )
            for key, value in direct_by_step.items()
            if key != "1e-03"
        }
        rows[name] = {
            "steps": step_rows,
            "secant_step_stability": stability,
            "passed": bool(
                all(row["passed"] for row in step_rows.values())
                and all(row["passed"] for row in stability.values())
            ),
        }
        arrays[f"{name}_direction"] = direction
        arrays[f"{name}_predicted"] = predicted
    return rows, arrays


def _run_case(
    initial: dict,
    vector: np.ndarray,
    operator_arrays: dict[str, np.ndarray],
    *,
    n_cells: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    state = unpack_causal_five_field_state(vector, n_cells)
    primitives = np.asarray(state.primitives, dtype=float).ravel()
    primitive_scales = np.asarray(
        operator_arrays["primitive_column_scales"], dtype=float
    )
    row_scales = np.asarray(
        operator_arrays["conservation_row_scales"], dtype=float
    )
    base_vector_field = causal_five_field_scaled_primitive_vector_field(
        initial["context"],
        primitives,
        primitive_column_scales=primitive_scales,
        conservation_row_scales=row_scales,
        finite_difference_step=wp10c8j.BASE_INNER_DIFFERENCE_STEP,
        storage_quadrature_order=wp10c8j.STORAGE_QUADRATURE_ORDER,
        storage_directional_step=wp10c8j.STORAGE_DIRECTIONAL_STEP,
        mapped_storage_backend="unified_audit",
        mapped_storage_difference_step=MAPPED_STORAGE_DIFFERENCE_STEP,
        mapped_storage_difference_order=MAPPED_STORAGE_DIFFERENCE_ORDER,
    )
    scaled_rate = np.asarray(
        base_vector_field["scaled_primitive_rate_per_s"], dtype=float
    ).ravel()
    mapped_storage_column_steps = np.asarray(
        base_vector_field["mapped_storage_column_steps"], dtype=float
    )
    physical_rate = primitive_scales * scaled_rate
    unified = causal_five_field_unified_mapped_storage_derivatives(
        initial["context"],
        primitives,
        physical_rate,
        primitive_column_scales=primitive_scales,
        conservation_row_scales=row_scales,
        mapped_storage_difference_step=MAPPED_STORAGE_DIFFERENCE_STEP,
        mapped_storage_difference_order=MAPPED_STORAGE_DIFFERENCE_ORDER,
        storage_rate_derivative_step=STORAGE_RATE_DERIVATIVE_STEP,
        storage_rate_derivative_order=STORAGE_RATE_DERIVATIVE_ORDER,
        mapped_storage_column_steps=mapped_storage_column_steps,
    )
    height = causal_five_field_reduced_storage_rate_derivatives(
        initial["context"],
        primitives,
        physical_rate,
        primitive_column_scales=primitive_scales,
        conservation_row_scales=row_scales,
        storage_matrix_difference_step=wp10c8j.BASE_INNER_DIFFERENCE_STEP,
        storage_rate_derivative_step=STORAGE_RATE_DERIVATIVE_STEP,
        storage_difference_step=wp10c8j.BASE_VERTICAL_ACTION_DIFFERENCE_STEP,
        storage_quadrature_order=wp10c8j.STORAGE_QUADRATURE_ORDER,
        storage_directional_step=wp10c8j.STORAGE_DIRECTIONAL_STEP,
        backend="direct_action",
    )
    mapped_descriptor = np.asarray(
        unified["conserved_descriptor_reduced_scaled_matrix"], dtype=float
    )
    vertical_descriptor = np.asarray(
        base_vector_field["vertical_descriptor_reduced_scaled_matrix"],
        dtype=float,
    )
    mass = mapped_descriptor + vertical_descriptor
    mapped_rate_derivative = np.asarray(
        unified["conserved_storage_rate_derivative_scaled_matrix"],
        dtype=float,
    )
    vertical_rate_derivative = np.asarray(
        height["vertical_storage_rate_derivative_scaled_matrix"], dtype=float
    )
    stationary = np.asarray(operator_arrays["stationary_jacobian"], dtype=float)
    assembled = causal_five_field_assemble_evolving_tangent(
        mass,
        stationary,
        mapped_rate_derivative + vertical_rate_derivative,
    )
    generator = np.asarray(
        assembled["evolving_scaled_generator_per_s"], dtype=float
    )
    if n_cells == 64:
        directions = _direction_source(n_cells, operator_arrays)
    else:
        normalized = wp10c8j._normalized_directions(
            initial,
            vector,
            primitive_scales,
        )
        directions = {name: normalized[name] for name in LOCKED_DIRECTIONS}
    jvp, jvp_arrays = _jvp_rows(
        initial,
        primitives,
        primitive_scales,
        row_scales,
        generator,
        directions,
        mapped_storage_column_steps,
    )
    base_mass_defect = float(
        np.max(
            np.abs(
                mass
                - np.asarray(
                    base_vector_field["descriptor_reduced_scaled_matrix"],
                    dtype=float,
                )
            )
        )
    )
    row = {
        "n_cells": n_cells,
        "mapped_storage_difference_step": MAPPED_STORAGE_DIFFERENCE_STEP,
        "mapped_storage_difference_order": MAPPED_STORAGE_DIFFERENCE_ORDER,
        "storage_rate_derivative_step": STORAGE_RATE_DERIVATIVE_STEP,
        "storage_rate_derivative_order": STORAGE_RATE_DERIVATIVE_ORDER,
        "mapped_storage_source": unified["source"],
        "minimum_mapped_storage_column_step": unified[
            "minimum_mapped_storage_column_step"
        ],
        "maximum_mapped_storage_column_step": unified[
            "maximum_mapped_storage_column_step"
        ],
        "reduced_mapped_storage_column_count": unified[
            "reduced_mapped_storage_column_count"
        ],
        "responsive_height_source": "unchanged_direct_path_one_form",
        "maximum_base_mass_reconstruction_defect": base_mass_defect,
        "maximum_generator_factorization_defect": float(
            assembled["maximum_scaled_generator_factorization_defect"]
        ),
        "directions": jvp,
        "passed": bool(
            base_mass_defect
            <= wp10c8j.MAXIMUM_STORAGE_COMPONENT_RECONSTRUCTION_DEFECT
            and assembled["maximum_scaled_generator_factorization_defect"]
            <= wp10c8j.MAXIMUM_GENERATOR_FACTORIZATION_DEFECT
            and all(value["passed"] for value in jvp.values())
        ),
    }
    arrays = {
        "mapped_descriptor": mapped_descriptor,
        "vertical_descriptor": vertical_descriptor,
        "mapped_storage_rate_derivative": mapped_rate_derivative,
        "vertical_storage_rate_derivative": vertical_rate_derivative,
        "dynamic": generator,
        "scaled_primitive_rate": scaled_rate,
        "mapped_storage_column_steps": mapped_storage_column_steps,
        **jvp_arrays,
    }
    return row, arrays


def main() -> None:
    arguments = _arguments()
    selected_names = set(
        arguments.case
        or (f"n{LOCKED_CASES[0][0]}_{LOCKED_CASES[0][1]}",)
    )
    selected_cases = tuple(
        case
        for case in LOCKED_CASES
        if f"n{case[0]}_{case[1]}" in selected_names
    )
    initial_by_mesh, vectors_by_mesh, state_provenance = wp10c8i._load_states()
    shell_edges_rg = np.asarray(
        wp10c8h._common_shell_edges(initial_by_mesh)["five_shell"],
        dtype=float,
    )
    rows = {}
    arrays = {}
    provenance = {}
    for n_cells, label, _role in selected_cases:
        key = f"n{n_cells}_{label}"
        initial = initial_by_mesh[n_cells]
        vector = vectors_by_mesh[n_cells][label]
        operator_arrays, _metadata, operator_provenance = _load_parent_operator(
            initial,
            vector,
            shell_edges_rg,
            n_cells=n_cells,
            label=label,
        )
        row, case_arrays = _run_case(
            initial,
            vector,
            operator_arrays,
            n_cells=n_cells,
        )
        rows[key] = row
        provenance[key] = {
            "state": state_provenance[str(n_cells)][label],
            "operator": operator_provenance,
        }
        arrays.update(
            {f"{key}_{name}": value for name, value in case_arrays.items()}
        )
        print(
            json.dumps(
                {"work_package": WORK_PACKAGE, "case": key, "passed": row["passed"]},
                sort_keys=True,
            ),
            flush=True,
        )
        if not row["passed"]:
            break

    arrays_path = _absolute(arguments.arrays)
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **arrays)
    first_passed = bool(
        rows.get("n64_t_0p05", {}).get("passed", False)
    )
    n128_requested = "n128_t_0p10" in selected_names
    n128_passed = bool(
        rows.get("n128_t_0p10", {}).get("passed", False)
    )
    output = {
        "work_package": WORK_PACKAGE,
        "base_commit": BASE_COMMIT,
        "decision": (
            "wp10c8l_a_locked_n64_failed"
            if not first_passed
            else (
                "wp10c8l_a_locked_n64_passed_n128_not_run"
                if not n128_requested
                else (
                    "wp10c8l_a_locked_n64_n128_passed"
                    if n128_passed
                    else "wp10c8l_a_locked_n128_failed"
                )
            )
        ),
        "scope": {
            "audit_only": True,
            "production_bdf_storage_changed": False,
            "stationary_residual_changed": False,
            "responsive_height_storage_changed": False,
            "truth_trajectory_run": False,
            "moment_ladder_changed": False,
            "exact_rusanov_operator_changed": False,
            "selected_cases": [
                f"n{n}_{label}" for n, label, _ in selected_cases
            ],
        },
        "cases": rows,
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
                "output": _relative(output_path),
                "arrays": _relative(arrays_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
