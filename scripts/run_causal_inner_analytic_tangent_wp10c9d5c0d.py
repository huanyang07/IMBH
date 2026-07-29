#!/usr/bin/env python3
"""Certify the N128 forward-AD frozen radial tangent (WP10c9d5c0d)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_causal_inner_derivative_repair_wp10c9d5c0a as wp10c9d5c0a
import run_causal_inner_direct_delta_repair_wp10c9d5c0b as wp10c9d5c0b
import run_causal_inner_dynamic_localization_wp10c9d5b as wp10c9d5b

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    causal_five_field_frozen_analytic_tangent,
    causal_five_field_radial_analytic_tangent,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d5c0d"
ANALYZED_BASE_COMMIT = "e492299df5668b49412f033e33df3d42e92f512e"
ANALYZED_BASE_PARENT = "508e5c284c2eaf7305efb45ae30a437b29dabb33"
ANALYZED_BASE_TREE = "8ee7f16a299ef0f1d0a22093cff2a9b4c4d983ec"
THIS_RUNNER = (
    "scripts/run_causal_inner_analytic_tangent_wp10c9d5c0d.py"
)

LABEL = "N128_exterior_N128_inner_c48"
DIRECTION_NAMES = (
    "common_mode",
    "calibration_global_inner_0",
    "heldout_global_0",
    "heldout_near_excision_0",
)
DERIVATIVE_ORDERS = (4, 6)
TARGET_RADII_OVER_RG = (5.0, 8.0, 12.0)
PATH_QUADRATURE_ORDER = 6
HOMOGENEITY_FACTORS = (-0.371, 2.125)

MAXIMUM_LINEARITY_DEFECT = 1.0e-10
MAXIMUM_BLOCK_LEDGER_DEFECT = 1.0e-12
MAXIMUM_RECONSTRUCTION_DEFECT = 1.0e-12
MAXIMUM_PROJECTOR_CLOSURE_DEFECT = 1.0e-10
MAXIMUM_PRODUCTION_IDENTITY_DEFECT = 1.0e-12
MAXIMUM_DESCRIPTOR_SOLVE_DEFECT = 1.0e-12
MAXIMUM_INDEPENDENT_BLOCK_DEFECT = 2.0e-8
MAXIMUM_DIRECT_ACTION_DEFECT = 5.0e-5

C0A_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_derivative_repair_wp10c9d5c0a"
)
C0A_SUMMARY = C0A_DIRECTORY / "summary.json"
C0A_ARRAYS = C0A_DIRECTORY / "decisive_arrays.npz"
C0C_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_uncolored_additivity_wp10c9d5c0c"
)
C0C_SUMMARY = C0C_DIRECTORY / "summary.json"
C0A_CACHE = (
    ROOT
    / "outputs/checkpoints/"
    "causal_inner_derivative_repair_wp10c9d5c0a/"
    f"{LABEL}_arrays.npz"
)
D5B_CACHE = (
    ROOT
    / "outputs/checkpoints/"
    "causal_inner_dynamic_localization_wp10c9d5b/"
    f"{LABEL}_arrays.npz"
)
CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_analytic_tangent_wp10c9d5c0d"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"

IMPLEMENTATION_SOURCES = (
    THIS_RUNNER,
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_radial_linear_tangent.py",
    "src/imri_qpe/layer3_minidisk_1d/__init__.py",
    "tests/test_causal_inner_radial_linear_tangent.py",
    "tests/test_causal_inner_analytic_tangent_wp10c9d5c0d.py",
)

BLOCK_NAMES = tuple(wp10c9d5b.BLOCK_NAMES)


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def _refresh_sha256s(directory: Path) -> None:
    entries = []
    for path in sorted(directory.iterdir()):
        if path.name == "SHA256SUMS.txt" or not path.is_file():
            continue
        entries.append(f"{_sha256(path)}  {path.name}")
    (directory / "SHA256SUMS.txt").write_text(
        "\n".join(entries) + "\n",
        encoding="utf-8",
    )


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_analyzed_git_identity() -> dict:
    commit = _git("rev-parse", ANALYZED_BASE_COMMIT)
    parent = _git("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
    tree = _git("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
    if (commit, parent, tree) != (
        ANALYZED_BASE_COMMIT,
        ANALYZED_BASE_PARENT,
        ANALYZED_BASE_TREE,
    ):
        raise RuntimeError("WP10c9d5c0d analyzed Git identity changed")
    return {
        "analyzed_base_commit": commit,
        "analyzed_base_parent_commit": parent,
        "analyzed_base_tree_sha": tree,
    }


def _source_manifest() -> tuple[dict[str, str], str]:
    hashes = {
        path: _sha256(ROOT / path)
        for path in IMPLEMENTATION_SOURCES
        if (ROOT / path).exists()
    }
    digest = hashlib.sha256()
    for path, value in sorted(hashes.items()):
        digest.update(path.encode("utf-8"))
        digest.update(value.encode("ascii"))
    return hashes, digest.hexdigest()


def _relative_difference(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    scale = max(
        float(np.linalg.norm(left)),
        float(np.linalg.norm(right)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(left - right) / scale)


def _configuration() -> dict:
    return wp10c9d5c0b._configurations()[LABEL]


def _load_parent_evidence() -> tuple[dict, dict[str, np.ndarray]]:
    c0c = json.loads(C0C_SUMMARY.read_text(encoding="utf-8"))
    if (
        c0c["classification"]
        != "independent_cell_additivity_failed_"
        "finite_difference_linear_tangent_blocked"
        or not c0c["analytic_or_ad_linear_tangent_work_authorized"]
        or c0c["wp10c9d5c1_extended_localization_authorized"]
    ):
        raise RuntimeError("WP10c9d5c0c binding status changed")
    c0a = json.loads(C0A_SUMMARY.read_text(encoding="utf-8"))
    with np.load(C0A_ARRAYS, allow_pickle=False) as source:
        arrays = {
            name: np.asarray(source[name])
            for name in source.files
        }
    for name, expected in c0a["decisive_array_hashes"].items():
        if _array_sha256(arrays[name]) != expected:
            raise RuntimeError(f"WP10c9d5c0a array changed: {name}")
    return c0a, arrays


def _load_high_order_matrices() -> dict[int, dict[str, np.ndarray]]:
    if not C0A_CACHE.exists():
        raise RuntimeError("WP10c9d5c0a high-order cache is missing")
    with np.load(C0A_CACHE, allow_pickle=False) as source:
        packed = {
            name: np.asarray(source[name])
            for name in source.files
        }
    sparse = wp10c9d5c0a._unpack_family(packed)
    return {
        order: {
            name: np.asarray(matrix.toarray(), dtype=float)
            for name, matrix in family.items()
        }
        for order, family in sparse.items()
    }


def _load_anchor(configuration: dict) -> np.ndarray:
    if not D5B_CACHE.exists():
        raise RuntimeError("WP10c9d5b N128 block cache is missing")
    _metadata, _matrices, dense = wp10c9d5b._build_or_load_blocks(
        configuration,
        force=False,
    )
    return np.asarray(dense["anchor_storage_derivative"], dtype=float)


def _regions(configuration: dict) -> dict[str, np.ndarray]:
    faces = wp10c9d5c0a.wp10c9d5c0._actual_faces(configuration)
    return {
        f"through_{target:g}rg_plus_halo": (
            wp10c9d5c0a.wp10c9d5c0._region_rows(
                configuration,
                faces[target],
                halo=True,
            )
        )
        for target in TARGET_RADII_OVER_RG
    }


def _independent_matrix_report(
    analytic,
    high_order: dict[int, dict[str, np.ndarray]],
) -> dict:
    reports = {}
    passed = True
    for order in DERIVATIVE_ORDERS:
        block_reports = {}
        for name in BLOCK_NAMES:
            reference = high_order[order][name]
            candidate = analytic.block_scaled_jacobians[
                f"candidate_{name}"
            ]
            defect = _relative_difference(candidate, reference)
            block_reports[name] = {
                "relative_frobenius_defect": defect,
                "passed": bool(
                    defect <= MAXIMUM_INDEPENDENT_BLOCK_DEFECT
                ),
            }
            passed = bool(
                passed
                and defect <= MAXIMUM_INDEPENDENT_BLOCK_DEFECT
            )
        reference_total = sum(
            (
                high_order[order][name]
                for name in BLOCK_NAMES
            ),
            start=np.zeros_like(
                analytic.candidate_stationary_scaled_jacobian
            ),
        )
        aggregate_defect = _relative_difference(
            analytic.candidate_stationary_scaled_jacobian,
            reference_total,
        )
        reports[str(order)] = {
            "blocks": block_reports,
            "aggregate_relative_frobenius_defect": aggregate_defect,
            "aggregate_passed": bool(
                aggregate_defect <= MAXIMUM_INDEPENDENT_BLOCK_DEFECT
            ),
        }
        passed = bool(
            passed
            and aggregate_defect <= MAXIMUM_INDEPENDENT_BLOCK_DEFECT
        )
    return {
        "reference": (
            "independent fourth/sixth-order local-block matrices; "
            "diagnostic only for the old stationary subtraction"
        ),
        "orders": reports,
        "passed": passed,
    }


def _linearity_report(
    matrix: np.ndarray,
    directions: dict[str, np.ndarray],
) -> tuple[dict, dict[str, np.ndarray]]:
    arrays = {}
    additivity = {}
    names = tuple(directions)
    maximum = 0.0
    for left_name, right_name in zip(names[:-1], names[1:], strict=True):
        left = directions[left_name]
        right = directions[right_name]
        combined = matrix @ (left + right)
        summed = matrix @ left + matrix @ right
        defect = _relative_difference(combined, summed)
        label = f"{left_name}__plus__{right_name}"
        additivity[label] = defect
        arrays[f"linearity__{label}__combined"] = combined
        arrays[f"linearity__{label}__summed"] = summed
        maximum = max(maximum, defect)
    homogeneity = {}
    for name, direction in directions.items():
        for factor in HOMOGENEITY_FACTORS:
            scaled = matrix @ (factor * direction)
            expected = factor * (matrix @ direction)
            defect = _relative_difference(scaled, expected)
            label = f"{name}__factor_{factor:g}"
            homogeneity[label] = defect
            maximum = max(maximum, defect)
    return {
        "maximum_relative_defect": maximum,
        "additivity_relative_defects": additivity,
        "homogeneity_relative_defects": homogeneity,
        "passed": bool(maximum <= MAXIMUM_LINEARITY_DEFECT),
    }, arrays


def _direct_action_report(
    matrix: np.ndarray,
    directions: dict[str, np.ndarray],
    parent_arrays: dict[str, np.ndarray],
    regions: dict[str, np.ndarray],
) -> tuple[dict, dict[str, np.ndarray]]:
    reports = {}
    arrays = {}
    maximum = 0.0
    for name, direction in directions.items():
        analytic_action = np.asarray(matrix @ direction, dtype=float)
        arrays[f"direction__{name}"] = direction
        arrays[f"analytic_action__{name}"] = analytic_action
        order_reports = {}
        for order, action_name in (
            (4, "fourth_order_action"),
            (6, "sixth_order_action"),
        ):
            key = (
                f"{LABEL}__{name}__direct__{action_name}"
            )
            direct = np.asarray(parent_arrays[key], dtype=float)
            arrays[f"direct_order{order}_action__{name}"] = direct
            region_reports = {}
            for region_name, selected_rows in regions.items():
                defect = _relative_difference(
                    analytic_action[selected_rows],
                    direct[selected_rows],
                )
                maximum = max(maximum, defect)
                region_reports[region_name] = defect
            order_reports[str(order)] = region_reports
        reports[name] = order_reports
    return {
        "reference": (
            "nonadditive finite-step full-direction JVP retained only as "
            "an independent accuracy diagnostic"
        ),
        "maximum_relative_defect": maximum,
        "directions": reports,
        "passed": bool(maximum <= MAXIMUM_DIRECT_ACTION_DEFECT),
    }, arrays


def run() -> dict:
    started = time.perf_counter()
    identity = _validate_analyzed_git_identity()
    _c0a_summary, parent_arrays = _load_parent_evidence()
    configuration = _configuration()
    native = configuration["candidate_native"]
    high_order = _load_high_order_matrices()
    anchor = _load_anchor(configuration)

    analytic = causal_five_field_radial_analytic_tangent(
        configuration["context"],
        configuration["base_primitives"],
        primitive_column_scales=native["primitive_column_scales"],
        conservation_row_scales=native["conservation_row_scales"],
        path_quadrature_order=PATH_QUADRATURE_ORDER,
    )
    frozen = causal_five_field_frozen_analytic_tangent(
        analytic,
        native["production_generator"],
        native["descriptor"],
        anchor,
    )
    directions = {
        name: np.asarray(
            parent_arrays[
                f"{LABEL}__{name}__direct__direction"
            ],
            dtype=float,
        )
        for name in DIRECTION_NAMES
    }
    independent_report = _independent_matrix_report(
        analytic,
        high_order,
    )
    linearity_report, linearity_arrays = _linearity_report(
        frozen.stationary_delta_scaled_jacobian,
        directions,
    )
    direct_report, direct_arrays = _direct_action_report(
        frozen.stationary_delta_scaled_jacobian,
        directions,
        parent_arrays,
        _regions(configuration),
    )
    binding_method_gates = {
        "reconstruction": bool(
            analytic.maximum_base_reconstruction_relative_defect
            <= MAXIMUM_RECONSTRUCTION_DEFECT
        ),
        "projector_closure": bool(
            analytic.maximum_projector_closure_defect
            <= MAXIMUM_PROJECTOR_CLOSURE_DEFECT
        ),
        "block_ledger": bool(
            analytic.maximum_block_ledger_relative_defect
            <= MAXIMUM_BLOCK_LEDGER_DEFECT
        ),
        "production_identity": bool(
            frozen.maximum_production_identity_relative_defect
            <= MAXIMUM_PRODUCTION_IDENTITY_DEFECT
        ),
        "descriptor_solve": bool(
            frozen.maximum_descriptor_solve_relative_defect
            <= MAXIMUM_DESCRIPTOR_SOLVE_DEFECT
        ),
        "independent_blocks": bool(independent_report["passed"]),
        "linearity": bool(linearity_report["passed"]),
    }
    diagnostic_gates = {
        # WP10c9d5c0c proved that this finite-step construction is not
        # additive, so it is retained verbatim but cannot define acceptance
        # of the explicitly linear tangent.
        "nonadditive_direct_action_reference": bool(
            direct_report["passed"]
        ),
    }
    passed = bool(all(binding_method_gates.values()))
    classification = (
        "n128_analytic_forward_tangent_certified_"
        "cross_grid_tangent_authorized"
        if passed
        else "n128_analytic_forward_tangent_failed_"
        "cross_grid_tangent_blocked"
    )

    decisive = {
        "base_primitives": np.asarray(
            configuration["base_primitives"],
            dtype=float,
        ),
        "primitive_column_scales": np.asarray(
            native["primitive_column_scales"],
            dtype=float,
        ),
        "conservation_row_scales": np.asarray(
            native["conservation_row_scales"],
            dtype=float,
        ),
        "left_reconstruction_weights": (
            analytic.left_reconstruction_weights
        ),
        "right_reconstruction_weights": (
            analytic.right_reconstruction_weights
        ),
        "descriptor": frozen.descriptor_reduced_scaled_matrix,
        "production_anchor_storage_derivative": (
            frozen.production_anchor_storage_derivative
        ),
        "production_stationary_jacobian": (
            frozen.production_stationary_scaled_jacobian
        ),
        "candidate_stationary_jacobian": (
            analytic.candidate_stationary_scaled_jacobian
        ),
        "analytic_stationary_delta": (
            frozen.stationary_delta_scaled_jacobian
        ),
        "analytic_candidate_generator": (
            frozen.candidate_scaled_generator_per_s
        ),
        **{
            f"block__{name}": matrix
            for name, matrix in analytic.block_scaled_jacobians.items()
        },
        **linearity_arrays,
        **direct_arrays,
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)
    source_hashes, source_manifest = _source_manifest()
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "label": LABEL,
        "direction_names": DIRECTION_NAMES,
        "derivative_orders": DERIVATIVE_ORDERS,
        "target_radii_over_rg": TARGET_RADII_OVER_RG,
        "path_quadrature_order": PATH_QUADRATURE_ORDER,
        "homogeneity_factors": HOMOGENEITY_FACTORS,
        "gates": {
            "maximum_linearity_defect": MAXIMUM_LINEARITY_DEFECT,
            "maximum_block_ledger_defect": (
                MAXIMUM_BLOCK_LEDGER_DEFECT
            ),
            "maximum_reconstruction_defect": (
                MAXIMUM_RECONSTRUCTION_DEFECT
            ),
            "maximum_projector_closure_defect": (
                MAXIMUM_PROJECTOR_CLOSURE_DEFECT
            ),
            "maximum_production_identity_defect": (
                MAXIMUM_PRODUCTION_IDENTITY_DEFECT
            ),
            "maximum_descriptor_solve_defect": (
                MAXIMUM_DESCRIPTOR_SOLVE_DEFECT
            ),
            "maximum_independent_block_defect": (
                MAXIMUM_INDEPENDENT_BLOCK_DEFECT
            ),
            "maximum_direct_action_defect": (
                MAXIMUM_DIRECT_ACTION_DEFECT
            ),
        },
    }
    _write_json(CONFIG_PATH, config)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        **identity,
        "parent_wp10c9d5c0c_summary_path": _relative(C0C_SUMMARY),
        "parent_wp10c9d5c0c_summary_sha256": _sha256(C0C_SUMMARY),
        "parent_wp10c9d5c0a_arrays_path": _relative(C0A_ARRAYS),
        "parent_wp10c9d5c0a_arrays_sha256": _sha256(C0A_ARRAYS),
        "replay_cache_hashes": {
            _relative(C0A_CACHE): _sha256(C0A_CACHE),
            _relative(D5B_CACHE): _sha256(D5B_CACHE),
        },
        "method": {
            "local_derivative": (
                "second_order_forward_mode_automatic_differentiation"
            ),
            "reconstruction": (
                "exact_affine_map_on_inactive_quadratic_"
                "admissibility_branch"
            ),
            "signed_split": (
                "analytic_base_descriptor_subspaces_"
                "frozen_during_tangent"
            ),
            "principal_matrix_derivatives_included": True,
            "production_stationary_tangent": (
                "recovered_from_M_G_plus_J_plus_D_anchor_equals_zero"
            ),
        },
        "measured_defects": {
            "base_reconstruction": (
                analytic.maximum_base_reconstruction_relative_defect
            ),
            "projector_closure": (
                analytic.maximum_projector_closure_defect
            ),
            "block_ledger": (
                analytic.maximum_block_ledger_relative_defect
            ),
            "production_identity": (
                frozen.maximum_production_identity_relative_defect
            ),
            "descriptor_solve": (
                frozen.maximum_descriptor_solve_relative_defect
            ),
            "analytic_to_stored_delta": _relative_difference(
                frozen.stationary_delta_scaled_jacobian,
                native["stationary_delta"],
            ),
            "analytic_to_stored_candidate_generator": (
                _relative_difference(
                    frozen.candidate_scaled_generator_per_s,
                    native["candidate_generator"],
                )
            ),
        },
        "binding_method_gates": binding_method_gates,
        "diagnostic_gates": diagnostic_gates,
        "independent_matrix_reference": independent_report,
        "linearity": linearity_report,
        "direct_action_reference": direct_report,
        "parent_wp10c9d5c0c_remains_rejected": True,
        "parent_wp10c9d5_candidate_remains_rejected": True,
        "parent_wp10c9d5b_branch_d_preserved": True,
        "cross_grid_analytic_tangent_work_authorized": passed,
        "derivative_choice_physical_sensitivity_authorized": False,
        "wp10c9d5c1_extended_localization_authorized": False,
        "self_consistent_tangent_authorized": False,
        "frozen_candidate_recertification_authorized": False,
        "production_operator_authorized": False,
        "nonlinear_candidate_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "decisive_arrays_path": _relative(DECISIVE_ARRAYS),
        "decisive_arrays_sha256": _sha256(DECISIVE_ARRAYS),
        "decisive_array_hashes": {
            name: _array_sha256(values)
            for name, values in decisive.items()
        },
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "environment": wp10c9d5c0b._environment(),
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_json(SUMMARY_PATH, summary)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        **identity,
        "generation_command": (
            "PYTHONPATH=src python3 "
            "scripts/run_causal_inner_analytic_tangent_wp10c9d5c0d.py"
        ),
        "method_scope": (
            "N128 ANALYTIC/AD-COMPATIBLE FROZEN LINEAR TANGENT; "
            "PRODUCTION NEUTRAL"
        ),
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        "scientific_status": "CERTIFIED" if passed else "REJECTED",
        "authorization_status": (
            "CROSS-GRID TANGENT CERTIFICATION ONLY"
            if passed
            else "DERIVATIVE METHOD WORK ONLY"
        ),
        "source_input_hashes": {
            _relative(C0C_SUMMARY): _sha256(C0C_SUMMARY),
            _relative(C0A_SUMMARY): _sha256(C0A_SUMMARY),
            _relative(C0A_ARRAYS): _sha256(C0A_ARRAYS),
            _relative(C0A_CACHE): _sha256(C0A_CACHE),
            _relative(D5B_CACHE): _sha256(D5B_CACHE),
        },
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "establishes": (
            "Whether one explicitly linear forward-AD candidate tangent, "
            "with frozen characteristic subspaces and an algebraically "
            "recovered production stationary tangent, passes the N128 "
            "linearity and independent derivative gates."
        ),
        "does_not_establish": (
            "Cross-grid tangent certification, derivative-choice physical "
            "sensitivity, a recovery radius, repaired physical operator, "
            "nonlinear convergence, fixed-Q closure, or reduced evolution."
        ),
    }
    _write_json(PROVENANCE_PATH, provenance)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(_plain(run()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
