"""Run the WP10c9d2 complete five-field path/fluctuation method contract.

WP10c9d1 rejects a single-family repair.  This package therefore audits the
sign-explicit complete principal jump

    Delta F - integral C_pr(Psi) Psi_s ds

and its negative/stationary/positive characteristic split on the existing
N128-exterior N128/N256/N512 inner-patch backgrounds.  The calculation is
production neutral: it neither assembles a cell residual nor selects a
finite-amplitude equilibrium path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_micro_export_preflight_wp10c9d0 as wp10c9d0

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    causal_five_field_complete_principal_path_jump,
    causal_five_field_coordinate_principal_components,
    causal_five_field_signed_principal_fluctuations,
    causal_five_field_straight_principal_path_jump,
)


BASE_COMMIT = "90f82c238e802abe22aa15b42f62b7d929048a60"
WORK_PACKAGE = "WP10c9d2"
SCHEMA_VERSION = 1
THIS_RUNNER = (
    "scripts/run_causal_inner_full_fluctuation_contract_wp10c9d2.py"
)
WP10C9D1_OUTPUT = (
    ROOT
    / "outputs/tables/"
    "causal_inner_micro_export_family_audit_wp10c9d1.json"
)
WP10C9D1_ARRAYS = (
    ROOT
    / "outputs/tables/"
    "causal_inner_micro_export_family_audit_wp10c9d1_arrays.npz"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/tables/"
    "causal_inner_full_fluctuation_contract_wp10c9d2.json"
)
DEFAULT_ARRAYS = (
    ROOT
    / "outputs/tables/"
    "causal_inner_full_fluctuation_contract_wp10c9d2_arrays.npz"
)

TARGET_RADII_RG = (1.90, 2.20, 3.00, 5.00, 6.00)
SMALL_JUMP_EPSILONS = (4.0e-4, 2.0e-4, 1.0e-4, 5.0e-5)
SMALL_JUMP_DIRECTIONS = np.asarray(
    [
        [0.25, -0.20, 0.15, 0.30, -0.10],
        [-0.10, 0.30, -0.25, 0.15, 0.20],
        [0.20, 0.10, 0.30, -0.15, -0.25],
    ],
    dtype=float,
)

MAXIMUM_IDENTITY_DEFECT = 1.0e-10
MAXIMUM_PATH_CONSISTENCY_DEFECT = 1.0e-8
MAXIMUM_FINE_SMALL_JUMP_DEFECT = 1.0e-7
MINIMUM_SMALL_JUMP_ORDER = 1.8
MINIMUM_SPEED_GAP_OVER_C = 1.0e-4
MAXIMUM_DESCRIPTOR_CONDITION = 1.0e6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _relative_defect(left: np.ndarray, right: np.ndarray) -> float:
    scale = max(
        float(np.max(np.abs(left))),
        float(np.max(np.abs(right))),
        np.finfo(float).tiny,
    )
    return float(np.max(np.abs(left - right)) / scale)


def _observed_orders(defects: np.ndarray) -> np.ndarray:
    coarse = np.asarray(defects[:-1], dtype=float)
    fine = np.asarray(defects[1:], dtype=float)
    return np.log2(
        np.maximum(coarse, np.finfo(float).tiny)
        / np.maximum(fine, np.finfo(float).tiny)
    )


def _configuration_audit(label: str, configuration: dict) -> tuple[dict, dict]:
    context = configuration["context"]
    primitives = np.asarray(configuration["base_primitives"], dtype=float)
    radius_rg = (
        np.asarray(context.grid.edges, dtype=float)
        / context.grid.gravitational_radius
    )
    faces = []
    for target in TARGET_RADII_RG:
        face = int(np.argmin(np.abs(radius_rg - target)))
        face = max(1, min(face, primitives.shape[0] - 1))
        if face not in faces:
            faces.append(face)

    per_face = []
    array_records = {
        "radii_rg": [],
        "speeds_over_c": [],
        "path_jumps": [],
        "characteristic_coefficients": [],
        "small_jump_defects": [],
    }
    for face in faces:
        radius = float(context.grid.edges[face])
        measure = float(context.grid.face_measures[face])
        left = np.asarray(primitives[face - 1], dtype=float)
        right = np.asarray(primitives[face], dtype=float)
        midpoint = 0.5 * (left + right)
        forward = causal_five_field_signed_principal_fluctuations(
            context,
            radius,
            left,
            right,
            face_measure=measure,
        )
        reverse = causal_five_field_complete_principal_path_jump(
            context,
            radius,
            right,
            left,
            face_measure=measure,
        )
        constant = causal_five_field_complete_principal_path_jump(
            context,
            radius,
            left,
            left,
            face_measure=measure,
        )
        existing = causal_five_field_straight_principal_path_jump(
            context,
            radius,
            left,
            right,
            face_measure=measure,
        )
        half_left = causal_five_field_complete_principal_path_jump(
            context,
            radius,
            left,
            midpoint,
            face_measure=measure,
        )
        half_right = causal_five_field_complete_principal_path_jump(
            context,
            radius,
            midpoint,
            right,
            face_measure=measure,
        )
        order_four = causal_five_field_complete_principal_path_jump(
            context,
            radius,
            left,
            right,
            quadrature_order=4,
            face_measure=measure,
        )

        total = forward.path_jump.total_principal_jump_over_c
        reversal_defect = _relative_defect(
            reverse.total_principal_jump_over_c,
            -total,
        )
        constant_defect = float(
            np.max(np.abs(constant.total_principal_jump_over_c))
        )
        existing_parity_defect = _relative_defect(existing, total)
        additivity_defect = _relative_defect(
            half_left.total_principal_jump_over_c
            + half_right.total_principal_jump_over_c,
            total,
        )
        quadrature_defect = _relative_defect(
            order_four.total_principal_jump_over_c,
            total,
        )

        components = causal_five_field_coordinate_principal_components(
            context,
            radius,
            midpoint,
        )
        direction_defects = []
        direction_orders = []
        for raw_direction in SMALL_JUMP_DIRECTIONS:
            direction = np.asarray(raw_direction, dtype=float)
            direction = (
                direction
                * components.primitive_column_scales
                / np.linalg.norm(direction)
            )
            defects = []
            for epsilon in SMALL_JUMP_EPSILONS:
                jump = causal_five_field_complete_principal_path_jump(
                    context,
                    radius,
                    midpoint - 0.5 * epsilon * direction,
                    midpoint + 0.5 * epsilon * direction,
                    face_measure=measure,
                )
                expected = (
                    measure
                    * epsilon
                    * (components.spatial_principal_matrix @ direction)
                )
                defects.append(
                    _relative_defect(
                        jump.total_principal_jump_over_c,
                        expected,
                    )
                )
            defects_array = np.asarray(defects, dtype=float)
            direction_defects.append(defects_array)
            direction_orders.append(_observed_orders(defects_array))
        small_defects = np.asarray(direction_defects, dtype=float)
        small_orders = np.asarray(direction_orders, dtype=float)
        # The weakest directions reach the 1e-11--1e-12 differentiation
        # floor before the smallest jump.  Bind the order of the worst-case
        # defect envelope, while retaining every raw direction order as a
        # diagnostic.  This prevents a roundoff-floor ratio from rejecting a
        # uniformly smaller error.
        small_envelope = np.max(small_defects, axis=0)
        small_envelope_orders = _observed_orders(small_envelope)
        per_face.append(
            {
                "face": face,
                "radius_rg": radius / context.grid.gravitational_radius,
                "constant_state_defect": constant_defect,
                "path_reversal_defect": reversal_defect,
                "existing_sign_explicit_path_parity_defect": (
                    existing_parity_defect
                ),
                "source_partition_defect": (
                    forward.path_jump.source_partition_defect
                ),
                "principal_closure_defect": (
                    forward.path_jump.principal_closure_defect
                ),
                "signed_split_closure_defect": (
                    forward.split_closure_defect
                ),
                "straight_path_additivity_defect": additivity_defect,
                "quadrature_4_to_8_defect": quadrature_defect,
                "maximum_fine_small_jump_defect": float(
                    np.max(small_defects[:, -1])
                ),
                "minimum_small_jump_envelope_order": float(
                    np.min(small_envelope_orders)
                ),
                "minimum_raw_small_jump_order": float(
                    np.min(small_orders)
                ),
                "minimum_speed_gap_over_c": (
                    forward.minimum_speed_gap_over_c
                ),
                "descriptor_condition_number": (
                    forward.midpoint_basis.descriptor_condition_number
                ),
                "incoming_characteristics": (
                    forward.midpoint_basis.incoming_inner_characteristics
                ),
            }
        )
        array_records["radii_rg"].append(
            radius / context.grid.gravitational_radius
        )
        array_records["speeds_over_c"].append(
            forward.midpoint_basis.numerical_speeds_over_c
        )
        array_records["path_jumps"].append(total)
        array_records["characteristic_coefficients"].append(
            forward.characteristic_jump_coefficients
        )
        array_records["small_jump_defects"].append(small_defects)

    maxima = {
        "maximum_constant_state_defect": max(
            item["constant_state_defect"] for item in per_face
        ),
        "maximum_path_reversal_defect": max(
            item["path_reversal_defect"] for item in per_face
        ),
        "maximum_existing_path_parity_defect": max(
            item["existing_sign_explicit_path_parity_defect"]
            for item in per_face
        ),
        "maximum_source_partition_defect": max(
            item["source_partition_defect"] for item in per_face
        ),
        "maximum_principal_closure_defect": max(
            item["principal_closure_defect"] for item in per_face
        ),
        "maximum_signed_split_closure_defect": max(
            item["signed_split_closure_defect"] for item in per_face
        ),
        "maximum_straight_path_additivity_defect": max(
            item["straight_path_additivity_defect"] for item in per_face
        ),
        "maximum_quadrature_4_to_8_defect": max(
            item["quadrature_4_to_8_defect"] for item in per_face
        ),
        "maximum_fine_small_jump_defect": max(
            item["maximum_fine_small_jump_defect"] for item in per_face
        ),
        "minimum_small_jump_envelope_order": min(
            item["minimum_small_jump_envelope_order"]
            for item in per_face
        ),
        "minimum_raw_small_jump_order": min(
            item["minimum_raw_small_jump_order"] for item in per_face
        ),
        "minimum_speed_gap_over_c": min(
            item["minimum_speed_gap_over_c"] for item in per_face
        ),
        "maximum_descriptor_condition_number": max(
            item["descriptor_condition_number"] for item in per_face
        ),
        "inner_incoming_characteristics": per_face[0][
            "incoming_characteristics"
        ],
    }
    passed = bool(
        maxima["maximum_constant_state_defect"] == 0.0
        and maxima["maximum_path_reversal_defect"]
        <= MAXIMUM_IDENTITY_DEFECT
        and maxima["maximum_existing_path_parity_defect"]
        <= MAXIMUM_IDENTITY_DEFECT
        and maxima["maximum_source_partition_defect"]
        <= MAXIMUM_IDENTITY_DEFECT
        and maxima["maximum_principal_closure_defect"]
        <= MAXIMUM_IDENTITY_DEFECT
        and maxima["maximum_signed_split_closure_defect"]
        <= MAXIMUM_IDENTITY_DEFECT
        and maxima["maximum_straight_path_additivity_defect"]
        <= MAXIMUM_PATH_CONSISTENCY_DEFECT
        and maxima["maximum_quadrature_4_to_8_defect"]
        <= MAXIMUM_PATH_CONSISTENCY_DEFECT
        and maxima["maximum_fine_small_jump_defect"]
        <= MAXIMUM_FINE_SMALL_JUMP_DEFECT
        and maxima["minimum_small_jump_envelope_order"]
        >= MINIMUM_SMALL_JUMP_ORDER
        and maxima["minimum_speed_gap_over_c"]
        >= MINIMUM_SPEED_GAP_OVER_C
        and maxima["maximum_descriptor_condition_number"]
        <= MAXIMUM_DESCRIPTOR_CONDITION
        and maxima["inner_incoming_characteristics"] == 0
    )
    return (
        {
            "label": label,
            "n_cells": int(np.asarray(context.grid.centers).size),
            "per_face": per_face,
            **maxima,
            "passed": passed,
        },
        {
            key: np.asarray(value)
            for key, value in array_records.items()
        },
    )


def run() -> tuple[dict, dict[str, np.ndarray]]:
    started = time.perf_counter()
    required = (
        WP10C9D1_OUTPUT,
        WP10C9D1_ARRAYS,
        wp10c9d0.WP10C8Z_OUTPUT,
        wp10c9d0.WP10C8Z_ARRAYS,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "WP10c9d2 requires prior evidence: " + ", ".join(missing)
        )
    patch_arrays = wp10c9d0._load_npz(wp10c9d0.WP10C8Z_ARRAYS)
    configurations = wp10c9d0._patch_configurations(patch_arrays)
    summaries = {}
    arrays = {}
    for label, configuration in configurations.items():
        summary, configuration_arrays = _configuration_audit(
            label,
            configuration,
        )
        summaries[label] = summary
        for key, value in configuration_arrays.items():
            arrays[f"{label}_{key}"] = value
    method_passed = all(
        summary["passed"] for summary in summaries.values()
    )
    classification = (
        "full_principal_path_contract_passed_cell_assembly_is_next_gate"
        if method_passed
        else "full_principal_path_contract_failed"
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "base_commit": BASE_COMMIT,
        "runner": THIS_RUNNER,
        "classification": classification,
        "method_contract_passed": method_passed,
        "well_balanced_cell_assembly_audit_authorized": method_passed,
        "production_operator_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "straight_path_is_physical_finite_amplitude_path": False,
        "gates": {
            "maximum_identity_defect": MAXIMUM_IDENTITY_DEFECT,
            "maximum_path_consistency_defect": (
                MAXIMUM_PATH_CONSISTENCY_DEFECT
            ),
            "maximum_fine_small_jump_defect": (
                MAXIMUM_FINE_SMALL_JUMP_DEFECT
            ),
            "minimum_small_jump_order": MINIMUM_SMALL_JUMP_ORDER,
            "minimum_speed_gap_over_c": MINIMUM_SPEED_GAP_OVER_C,
            "maximum_descriptor_condition": MAXIMUM_DESCRIPTOR_CONDITION,
        },
        "configurations": summaries,
        "input_hashes": {
            _relative(path): _sha256(path) for path in required
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    return payload, arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--arrays", type=Path, default=DEFAULT_ARRAYS)
    arguments = parser.parse_args()
    payload, arrays = run()
    arguments.arrays.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.arrays, **arrays)
    payload["arrays_path"] = _relative(arguments.arrays)
    payload["arrays_sha256"] = _sha256(arguments.arrays)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "classification": payload["classification"],
                "method_contract_passed": payload[
                    "method_contract_passed"
                ],
                "runtime_seconds": payload["runtime_seconds"],
                "output": _relative(arguments.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
