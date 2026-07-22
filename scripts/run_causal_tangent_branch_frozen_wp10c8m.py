"""Run the locked branch-frozen mapped-storage tangent gate for WP10c8m-A."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import run_causal_moment_sufficiency_audit_wp10c8i as wp10c8i
import run_causal_shell_closure_preflight_wp10c8h as wp10c8h
import run_causal_tangent_certification_wp10c8j as wp10c8j
import run_causal_tangent_descriptor_wp10c8l as wp10c8l
from imri_qpe.layer3_minidisk_1d import (
    causal_five_field_assemble_evolving_tangent,
    causal_five_field_branch_frozen_mapped_storage_derivatives,
    causal_five_field_reduced_storage_rate_derivatives,
    causal_five_field_scaled_primitive_vector_field,
    unpack_causal_five_field_state,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "4dc5cea0342d35135e31078669e7e71ba7d16cf9"
WORK_PACKAGE = "WP10c8m-A"
LOCKED_CASES = (
    (64, "t_0p05", "construction"),
    (128, "t_0p10", "held_out"),
)
LOCAL_DIFFERENCE_STEP = 1.0e-3
LOCAL_CONVERGENCE_COARSE_STEP = 2.0e-3
MAXIMUM_LOCAL_DERIVATIVE_STEP_DEFECT = 5.0e-3
THIS_RUNNER = "scripts/run_causal_tangent_branch_frozen_wp10c8m.py"
DEFAULT_OUTPUT = (
    ROOT / "outputs/tables/causal_tangent_branch_frozen_wp10c8m.json"
)
DEFAULT_ARRAYS = (
    ROOT / "outputs/tables/causal_tangent_branch_frozen_wp10c8m_arrays.npz"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=tuple(f"n{n}_{label}" for n, label, _ in LOCKED_CASES),
        default="n64_t_0p05",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--arrays", type=Path)
    return parser.parse_args()


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


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
        float(np.max(np.abs(first))),
        float(np.max(np.abs(second))),
        np.finfo(float).tiny,
    )
    return float(np.max(np.abs(first - second)) / scale)


def main() -> None:
    arguments = _arguments()
    case_by_key = {
        f"n{n}_{label}": (n, label, role)
        for n, label, role in LOCKED_CASES
    }
    n_cells, label, role = case_by_key[arguments.case]
    output_path = _absolute(arguments.output or DEFAULT_OUTPUT)
    arrays_path = _absolute(arguments.arrays or DEFAULT_ARRAYS)
    initial_by_mesh, vectors_by_mesh, state_provenance = wp10c8i._load_states()
    shell_edges_rg = np.asarray(
        wp10c8h._common_shell_edges(initial_by_mesh)["five_shell"],
        dtype=float,
    )
    initial = initial_by_mesh[n_cells]
    vector = vectors_by_mesh[n_cells][label]
    original_runner = wp10c8l.THIS_RUNNER
    try:
        wp10c8l.THIS_RUNNER = THIS_RUNNER
        operator_arrays, _metadata, operator_provenance = (
            wp10c8l._load_parent_operator(
                initial,
                vector,
                shell_edges_rg,
                n_cells=n_cells,
                label=label,
            )
        )
    finally:
        wp10c8l.THIS_RUNNER = original_runner

    state = unpack_causal_five_field_state(vector, n_cells)
    primitives = np.asarray(state.primitives, dtype=float).ravel()
    primitive_scales = np.asarray(
        operator_arrays["primitive_column_scales"], dtype=float
    )
    row_scales = np.asarray(
        operator_arrays["conservation_row_scales"], dtype=float
    )
    vector_field_kwargs = {
        "mapped_storage_backend": "branch_frozen_local",
        "branch_frozen_local_difference_step": LOCAL_DIFFERENCE_STEP,
    }
    base_vector_field = causal_five_field_scaled_primitive_vector_field(
        initial["context"],
        primitives,
        primitive_column_scales=primitive_scales,
        conservation_row_scales=row_scales,
        finite_difference_step=wp10c8j.BASE_INNER_DIFFERENCE_STEP,
        storage_quadrature_order=wp10c8j.STORAGE_QUADRATURE_ORDER,
        storage_directional_step=wp10c8j.STORAGE_DIRECTIONAL_STEP,
        **vector_field_kwargs,
    )
    scaled_rate = np.asarray(
        base_vector_field["scaled_primitive_rate_per_s"], dtype=float
    ).ravel()
    physical_rate = primitive_scales * scaled_rate
    selected = causal_five_field_branch_frozen_mapped_storage_derivatives(
        initial["context"],
        primitives,
        physical_rate,
        primitive_column_scales=primitive_scales,
        conservation_row_scales=row_scales,
        local_difference_step=LOCAL_DIFFERENCE_STEP,
    )
    coarse = causal_five_field_branch_frozen_mapped_storage_derivatives(
        initial["context"],
        primitives,
        physical_rate,
        primitive_column_scales=primitive_scales,
        conservation_row_scales=row_scales,
        local_difference_step=LOCAL_CONVERGENCE_COARSE_STEP,
    )
    descriptor_step_defect = _relative_defect(
        selected["conserved_descriptor_reduced_scaled_matrix"],
        coarse["conserved_descriptor_reduced_scaled_matrix"],
    )
    rate_step_defect = _relative_defect(
        selected["conserved_storage_rate_derivative_scaled_matrix"],
        coarse["conserved_storage_rate_derivative_scaled_matrix"],
    )
    local_step_passed = bool(
        descriptor_step_defect <= MAXIMUM_LOCAL_DERIVATIVE_STEP_DEFECT
        and rate_step_defect <= MAXIMUM_LOCAL_DERIVATIVE_STEP_DEFECT
    )

    height = causal_five_field_reduced_storage_rate_derivatives(
        initial["context"],
        primitives,
        physical_rate,
        primitive_column_scales=primitive_scales,
        conservation_row_scales=row_scales,
        storage_matrix_difference_step=wp10c8j.BASE_INNER_DIFFERENCE_STEP,
        storage_rate_derivative_step=wp10c8l.STORAGE_RATE_DERIVATIVE_STEP,
        storage_difference_step=wp10c8j.BASE_VERTICAL_ACTION_DIFFERENCE_STEP,
        storage_quadrature_order=wp10c8j.STORAGE_QUADRATURE_ORDER,
        storage_directional_step=wp10c8j.STORAGE_DIRECTIONAL_STEP,
        backend="direct_action",
    )
    mapped_descriptor = np.asarray(
        selected["conserved_descriptor_reduced_scaled_matrix"], dtype=float
    )
    vertical_descriptor = np.asarray(
        base_vector_field["vertical_descriptor_reduced_scaled_matrix"],
        dtype=float,
    )
    mass = mapped_descriptor + vertical_descriptor
    mapped_rate_derivative = np.asarray(
        selected["conserved_storage_rate_derivative_scaled_matrix"],
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
        directions = wp10c8l._direction_source(n_cells, operator_arrays)
    else:
        directions = {
            name: direction
            for name, direction in wp10c8j._normalized_directions(
                initial,
                vector,
                primitive_scales,
            ).items()
            if name in wp10c8l.LOCKED_DIRECTIONS
        }
        missing = set(wp10c8l.LOCKED_DIRECTIONS).difference(directions)
        if missing:
            raise RuntimeError(
                "missing locked N128 tangent directions: "
                + ", ".join(sorted(missing))
            )
    jvp, jvp_arrays = wp10c8l._jvp_rows(
        initial,
        primitives,
        primitive_scales,
        row_scales,
        generator,
        directions,
        None,
        vector_field_kwargs=vector_field_kwargs,
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
    factors = np.asarray(
        selected["base_reconstruction_admissibility_factors"], dtype=float
    )
    reconstruction_branch_is_unlimited = bool(np.all(factors == 1.0))
    case_passed = bool(
        local_step_passed
        and reconstruction_branch_is_unlimited
        and base_mass_defect
        <= wp10c8j.MAXIMUM_STORAGE_COMPONENT_RECONSTRUCTION_DEFECT
        and assembled["maximum_scaled_generator_factorization_defect"]
        <= wp10c8j.MAXIMUM_GENERATOR_FACTORIZATION_DEFECT
        and all(value["passed"] for value in jvp.values())
    )
    output = {
        "base_commit": BASE_COMMIT,
        "work_package": WORK_PACKAGE,
        "case": {
            "n_cells": n_cells,
            "anchor": label,
            "role": role,
            "local_difference_step": LOCAL_DIFFERENCE_STEP,
            "local_convergence_coarse_step": LOCAL_CONVERGENCE_COARSE_STEP,
            "maximum_local_derivative_step_defect": (
                MAXIMUM_LOCAL_DERIVATIVE_STEP_DEFECT
            ),
            "descriptor_step_defect": descriptor_step_defect,
            "storage_rate_step_defect": rate_step_defect,
            "local_step_passed": local_step_passed,
            "minimum_reconstruction_admissibility_factor": float(
                np.min(factors)
            ),
            "reconstruction_branch_is_unlimited": (
                reconstruction_branch_is_unlimited
            ),
            "maximum_base_mass_reconstruction_defect": base_mass_defect,
            "maximum_generator_factorization_defect": float(
                assembled["maximum_scaled_generator_factorization_defect"]
            ),
            "directions": jvp,
            "passed": case_passed,
        },
        "scope": {
            "audit_only": True,
            "production_bdf_storage_changed": False,
            "responsive_height_storage_changed": False,
            "stationary_residual_changed": False,
            "moment_ladder_changed": False,
        },
        "decision": (
            f"wp10c8m_a_locked_n{n_cells}_passed"
            if case_passed
            else f"wp10c8m_a_locked_n{n_cells}_failed"
        ),
        "next_action": (
            (
                "authorize_n128_t_0p10_replay"
                if n_cells == 64
                else "authorize_remaining_smooth_anchor_campaign"
            )
            if case_passed
            else "stop_before_remaining_smooth_and_binding_rusanov_certification"
        ),
        "provenance": {
            "state": state_provenance[str(n_cells)][label],
            "operator": operator_provenance,
        },
    }
    arrays = {
        "mapped_descriptor": mapped_descriptor,
        "vertical_descriptor": vertical_descriptor,
        "mapped_storage_rate_derivative": mapped_rate_derivative,
        "vertical_storage_rate_derivative": vertical_rate_derivative,
        "dynamic": generator,
        "scaled_primitive_rate": scaled_rate,
        "reconstruction_admissibility_factors": factors,
        **jvp_arrays,
    }
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **arrays)
    output["artifacts"] = {
        "arrays_path": str(arrays_path.relative_to(ROOT)),
        "arrays_sha256": _sha256(arrays_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
