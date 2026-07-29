#!/usr/bin/env python3
"""Certify the analytic frozen-subspace tangent on all embedded grids.

WP10c9d5c0e extends the N128 method certification from WP10c9d5c0d to the
N128-, N256-, and N512-equivalent inner grids.  It remains production neutral:
the package constructs audit tangents, checks them against independent
moving-projector high-order references, and does not propagate a physical
history.
"""

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
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import splu


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_causal_inner_analytic_tangent_wp10c9d5c0d as wp10c9d5c0d
import run_causal_inner_cross_grid_hardening_wp10c9d5c0 as wp10c9d5c0
import run_causal_inner_derivative_repair_wp10c9d5c0a as wp10c9d5c0a
import run_causal_inner_dynamic_localization_wp10c9d5b as wp10c9d5b
import run_causal_inner_frozen_hardening_wp10c9d5a as wp10c9d5a

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    causal_five_field_frozen_analytic_tangent,
    causal_five_field_radial_analytic_tangent,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d5c0e"
ANALYZED_BASE_COMMIT = "d57bcc3e63bcd778823736a795a9311592173bd9"
ANALYZED_BASE_PARENT = "e492299df5668b49412f033e33df3d42e92f512e"
ANALYZED_BASE_TREE = "1048352852cb195abb6a99f7b822e6c8a2cab419"
THIS_RUNNER = (
    "scripts/"
    "run_causal_inner_cross_grid_analytic_tangent_wp10c9d5c0e.py"
)

LABELS = tuple(wp10c9d5c0a.LABELS)
BLOCK_NAMES = tuple(wp10c9d5b.BLOCK_NAMES)
DIRECTION_NAMES = (
    "common_mode",
    "global_inner_0",
    "global_inner_1",
    "near_excision_0",
    "near_excision_1",
    "first_cell_field_0",
    "first_cell_field_1",
    "first_cell_field_2",
    "first_cell_field_3",
    "first_cell_field_4",
)
DERIVATIVE_ORDERS = (4, 6)
PATH_QUADRATURE_ORDER = 6
TARGET_RADII_OVER_RG = (5.0, 8.0, 12.0)
HOMOGENEITY_FACTORS = (-0.371, 2.125)
GEOMETRY_LOG_RADIUS_STEPS = (1.0e-5, 2.0e-5, 4.0e-5)
DEFAULT_GEOMETRY_LOG_RADIUS_STEP = 2.0e-5

MAXIMUM_LINEARITY_DEFECT = 1.0e-10
MAXIMUM_BLOCK_LEDGER_DEFECT = 1.0e-12
MAXIMUM_RECONSTRUCTION_DEFECT = 1.0e-12
MAXIMUM_PROJECTOR_CLOSURE_DEFECT = 1.0e-10
MAXIMUM_PRODUCTION_IDENTITY_DEFECT = 1.0e-12
MAXIMUM_DESCRIPTOR_SOLVE_DEFECT = 1.0e-12
MAXIMUM_INDEPENDENT_BLOCK_DEFECT = 2.0e-8
MAXIMUM_PRODUCTION_JVP_DEFECT = 2.0e-6
MAXIMUM_CHARACTERISTIC_EIGENPAIR_DEFECT = 1.0e-10
MAXIMUM_CHARACTERISTIC_BIORTHOGONALITY_DEFECT = 1.0e-10
MAXIMUM_CHARACTERISTIC_IMAGINARY_PART = 1.0e-10
MAXIMUM_CHARACTERISTIC_DESCRIPTOR_CONDITION = 1.0e10
MINIMUM_ABSOLUTE_CHARACTERISTIC_SPEED = 1.0e-6
MINIMUM_CHARACTERISTIC_SPECTRAL_GAP = 1.0e-6
MINIMUM_NEIGHBORING_SUBSPACE_COSINE = 0.90
MAXIMUM_GEOMETRY_STEP_TANGENT_DEFECT = 1.0e-6

PARENT_SUMMARY = (
    ROOT
    / "results/canonical/"
    "causal_inner_analytic_tangent_wp10c9d5c0d/summary.json"
)
CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_cross_grid_analytic_tangent_wp10c9d5c0e"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
REPLAY_CONTEXTS = CANONICAL_DIRECTORY / "replay_contexts.json"
REPLAY_INPUTS = CANONICAL_DIRECTORY / "replay_inputs.npz"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"

IMPLEMENTATION_SOURCES = (
    THIS_RUNNER,
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_radial_linear_tangent.py",
    "src/imri_qpe/layer3_minidisk_1d/__init__.py",
    "tests/test_causal_inner_radial_linear_tangent.py",
    "tests/"
    "test_causal_inner_cross_grid_analytic_tangent_wp10c9d5c0e.py",
)


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


def _relative_difference(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    scale = max(
        float(np.linalg.norm(left)),
        float(np.linalg.norm(right)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(left - right) / scale)


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
        raise RuntimeError("WP10c9d5c0e analyzed Git identity changed")
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


def _pack_sparse(
    prefix: str,
    matrix,
    arrays: dict[str, np.ndarray],
) -> None:
    packed = wp10c9d5b._pack_sparse(prefix, csr_matrix(matrix))
    arrays.update(packed)


def _unpack_sparse(
    prefix: str,
    arrays: dict[str, np.ndarray],
):
    return wp10c9d5b._unpack_sparse(prefix, arrays)


def _prepare_replay_inputs() -> dict:
    """Commit the minimal cross-grid inputs and independent references."""

    configurations = wp10c9d5c0a._configurations()
    arrays: dict[str, np.ndarray] = {}
    contexts = {}
    source_hashes = {}
    for label in LABELS:
        configuration = configurations[label]
        context = configuration["context"]
        native = configuration["candidate_native"]
        prefix = f"{label}__"
        directions = wp10c9d5c0._directions(configuration)
        _metadata, family = wp10c9d5c0a._build_or_load_high_order_blocks(
            configuration,
            force=False,
        )
        _block_metadata, _block_matrices, dense = (
            wp10c9d5b._build_or_load_blocks(
                configuration,
                force=False,
            )
        )
        arrays[prefix + "base_primitives"] = np.asarray(
            configuration["base_primitives"],
            dtype=float,
        )
        arrays[prefix + "primitive_column_scales"] = np.asarray(
            native["primitive_column_scales"],
            dtype=float,
        )
        arrays[prefix + "conservation_row_scales"] = np.asarray(
            native["conservation_row_scales"],
            dtype=float,
        )
        _pack_sparse(
            prefix + "production_generator",
            native["production_generator"],
            arrays,
        )
        _pack_sparse(
            prefix + "descriptor",
            native["descriptor"],
            arrays,
        )
        _pack_sparse(
            prefix + "production_anchor_storage_derivative",
            dense["anchor_storage_derivative"],
            arrays,
        )
        _pack_sparse(
            prefix + "stored_stationary_delta",
            native["stationary_delta"],
            arrays,
        )
        for order, matrices in family.items():
            for name, matrix in matrices.items():
                _pack_sparse(
                    prefix + f"reference_order{order}__{name}",
                    matrix,
                    arrays,
                )
        for name, direction in directions.items():
            arrays[prefix + f"direction__{name}"] = np.asarray(
                direction,
                dtype=float,
            )
        arrays[prefix + "initial"] = np.asarray(
            configuration["initial"],
            dtype=float,
        )
        arrays[prefix + "times"] = np.asarray(
            configuration["times"],
            dtype=float,
        )
        arrays[prefix + "amplitudes"] = np.asarray(
            configuration["amplitudes"],
            dtype=float,
        )
        contexts[label] = {
            **wp10c9d5a._context_payload(
                context,
                label=label,
                arrays=arrays,
            ),
            "interface_face": int(configuration["interface_face"]),
            "active_cells": int(configuration["active_cells"]),
        }
        for path in (
            *wp10c9d5c0a._cache_paths(label),
            *wp10c9d5b._cache_paths(label),
        ):
            if path.exists():
                source_hashes[_relative(path)] = _sha256(path)
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(REPLAY_INPUTS, **arrays)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "labels": LABELS,
        "contexts": contexts,
        "source_input_hashes": source_hashes,
        "replay_inputs_path": _relative(REPLAY_INPUTS),
        "replay_inputs_sha256": _sha256(REPLAY_INPUTS),
        "replay_array_hashes": {
            name: _array_sha256(values)
            for name, values in arrays.items()
        },
    }
    _write_json(REPLAY_CONTEXTS, payload)
    return payload


def _load_replay_inputs() -> tuple[dict, dict[str, np.ndarray]]:
    if not REPLAY_CONTEXTS.exists() or not REPLAY_INPUTS.exists():
        raise FileNotFoundError(
            "WP10c9d5c0e replay inputs are absent; prepare them once "
            "from the scientific worktree"
        )
    payload = json.loads(REPLAY_CONTEXTS.read_text(encoding="utf-8"))
    with np.load(REPLAY_INPUTS, allow_pickle=False) as source:
        arrays = {
            name: np.asarray(source[name])
            for name in source.files
        }
    if set(arrays) != set(payload["replay_array_hashes"]):
        raise RuntimeError("WP10c9d5c0e replay array set changed")
    for name, expected in payload["replay_array_hashes"].items():
        if _array_sha256(arrays[name]) != expected:
            raise RuntimeError(
                f"WP10c9d5c0e replay array changed: {name}"
            )
    return payload, arrays


def _configurations(
    replay_payload: dict,
    replay_arrays: dict[str, np.ndarray],
) -> dict:
    result = {}
    for label in LABELS:
        prefix = f"{label}__"
        payload = replay_payload["contexts"][label]
        production_generator = _unpack_sparse(
            prefix + "production_generator",
            replay_arrays,
        ).toarray()
        descriptor = _unpack_sparse(
            prefix + "descriptor",
            replay_arrays,
        ).toarray()
        stored_stationary_delta = _unpack_sparse(
            prefix + "stored_stationary_delta",
            replay_arrays,
        ).toarray()
        stored_candidate_generator = production_generator - splu(
            csc_matrix(descriptor),
            permc_spec="COLAMD",
        ).solve(stored_stationary_delta)
        result[label] = {
            "label": label,
            "context": wp10c9d5a._context_from_payload(
                payload,
                replay_arrays,
            ),
            "base_primitives": np.asarray(
                replay_arrays[prefix + "base_primitives"],
                dtype=float,
            ),
            "initial": np.asarray(
                replay_arrays[prefix + "initial"],
                dtype=float,
            ),
            "times": np.asarray(
                replay_arrays[prefix + "times"],
                dtype=float,
            ),
            "amplitudes": np.asarray(
                replay_arrays[prefix + "amplitudes"],
                dtype=float,
            ),
            "interface_face": int(payload["interface_face"]),
            "active_cells": int(payload["active_cells"]),
            "candidate_native": {
                "primitive_column_scales": np.asarray(
                    replay_arrays[
                        prefix + "primitive_column_scales"
                    ],
                    dtype=float,
                ),
                "conservation_row_scales": np.asarray(
                    replay_arrays[
                        prefix + "conservation_row_scales"
                    ],
                    dtype=float,
                ),
                "production_generator": production_generator,
                "descriptor": descriptor,
                "stationary_delta": stored_stationary_delta,
                "candidate_generator": stored_candidate_generator,
            },
            "anchor_storage_derivative": _unpack_sparse(
                prefix + "production_anchor_storage_derivative",
                replay_arrays,
            ).toarray(),
            "directions": {
                name: np.asarray(
                    replay_arrays[prefix + f"direction__{name}"],
                    dtype=float,
                )
                for name in DIRECTION_NAMES
            },
            "references": {
                order: {
                    name: _unpack_sparse(
                        prefix + f"reference_order{order}__{name}",
                        replay_arrays,
                    )
                    for name in (*BLOCK_NAMES, "production")
                }
                for order in DERIVATIVE_ORDERS
            },
        }
    return result


def _regions(configuration: dict) -> dict[str, np.ndarray]:
    faces = wp10c9d5c0._actual_faces(configuration)
    return {
        f"through_{target:g}rg_plus_halo": (
            wp10c9d5c0._region_rows(
                configuration,
                faces[target],
                halo=True,
            )
        )
        for target in TARGET_RADII_OVER_RG
    }


def _independent_reference_report(
    analytic,
    frozen,
    configuration: dict,
) -> dict:
    orders = {}
    passed = True
    regions = _regions(configuration)
    for order in DERIVATIVE_ORDERS:
        block_reports = {}
        references = configuration["references"][order]
        for name in BLOCK_NAMES:
            candidate = analytic.block_scaled_jacobians[
                f"candidate_{name}"
            ]
            reference = references[name].toarray()
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
        production_reports = {}
        for direction_name, direction in configuration[
            "directions"
        ].items():
            candidate_action = (
                frozen.production_stationary_scaled_jacobian @ direction
            )
            reference_action = np.asarray(
                references["production"] @ direction,
                dtype=float,
            )
            region_reports = {}
            for region_name, rows in regions.items():
                defect = _relative_difference(
                    candidate_action[rows],
                    reference_action[rows],
                )
                region_reports[region_name] = defect
                passed = bool(
                    passed
                    and defect <= MAXIMUM_PRODUCTION_JVP_DEFECT
                )
            production_reports[direction_name] = region_reports
        orders[str(order)] = {
            "candidate_blocks": block_reports,
            "production_stationary_jvp": production_reports,
            "maximum_candidate_block_defect": max(
                report["relative_frobenius_defect"]
                for report in block_reports.values()
            ),
            "maximum_production_jvp_defect": max(
                defect
                for report in production_reports.values()
                for defect in report.values()
            ),
        }
    return {
        "reference": (
            "independent moving-projector fourth/sixth-order block "
            "matrices and production stationary JVPs"
        ),
        "orders": orders,
        "passed": passed,
    }


def _linearity_report(
    matrix: np.ndarray,
    directions: dict[str, np.ndarray],
) -> dict:
    names = tuple(directions)
    defects = []
    for left_name, right_name in zip(
        names[:-1],
        names[1:],
        strict=True,
    ):
        left = directions[left_name]
        right = directions[right_name]
        defects.append(
            _relative_difference(
                matrix @ (left + right),
                matrix @ left + matrix @ right,
            )
        )
    for name, direction in directions.items():
        for factor in HOMOGENEITY_FACTORS:
            defects.append(
                _relative_difference(
                    matrix @ (factor * direction),
                    factor * (matrix @ direction),
                )
            )
    maximum = max(defects) if defects else 0.0
    return {
        "maximum_relative_defect": maximum,
        "passed": bool(maximum <= MAXIMUM_LINEARITY_DEFECT),
    }


def _spectral_report(analytic) -> dict:
    gates = {
        "eigenpair": bool(
            analytic.maximum_characteristic_eigenpair_defect
            <= MAXIMUM_CHARACTERISTIC_EIGENPAIR_DEFECT
        ),
        "biorthogonality": bool(
            analytic.maximum_characteristic_biorthogonality_defect
            <= MAXIMUM_CHARACTERISTIC_BIORTHOGONALITY_DEFECT
        ),
        "real_spectrum": bool(
            analytic.maximum_characteristic_imaginary_part
            <= MAXIMUM_CHARACTERISTIC_IMAGINARY_PART
        ),
        "descriptor_condition": bool(
            analytic.maximum_characteristic_descriptor_condition_number
            <= MAXIMUM_CHARACTERISTIC_DESCRIPTOR_CONDITION
        ),
        "nonstationary": bool(
            analytic.minimum_absolute_characteristic_speed
            >= MINIMUM_ABSOLUTE_CHARACTERISTIC_SPEED
        ),
        "separated": bool(
            analytic.minimum_characteristic_spectral_gap
            >= MINIMUM_CHARACTERISTIC_SPECTRAL_GAP
        ),
        "negative_subspace_continuity": bool(
            analytic.minimum_neighboring_negative_subspace_cosine
            >= MINIMUM_NEIGHBORING_SUBSPACE_COSINE
        ),
        "positive_subspace_continuity": bool(
            analytic.minimum_neighboring_positive_subspace_cosine
            >= MINIMUM_NEIGHBORING_SUBSPACE_COSINE
        ),
        "constant_signed_subspace_ranks": bool(
            analytic.neighboring_negative_subspace_rank_changes == 0
            and analytic.neighboring_positive_subspace_rank_changes == 0
        ),
        "outgoing_excision": bool(
            analytic.incoming_inner_characteristics == 0
        ),
    }
    return {
        "minimum_absolute_speed_over_c": (
            analytic.minimum_absolute_characteristic_speed
        ),
        "minimum_spectral_gap_over_c": (
            analytic.minimum_characteristic_spectral_gap
        ),
        "minimum_neighboring_negative_subspace_cosine": (
            analytic.minimum_neighboring_negative_subspace_cosine
        ),
        "minimum_neighboring_positive_subspace_cosine": (
            analytic.minimum_neighboring_positive_subspace_cosine
        ),
        "neighboring_negative_subspace_rank_changes": (
            analytic.neighboring_negative_subspace_rank_changes
        ),
        "neighboring_positive_subspace_rank_changes": (
            analytic.neighboring_positive_subspace_rank_changes
        ),
        "maximum_analytic_speed_defect": (
            analytic.maximum_characteristic_analytic_speed_defect
        ),
        "maximum_eigenpair_defect": (
            analytic.maximum_characteristic_eigenpair_defect
        ),
        "maximum_biorthogonality_defect": (
            analytic.maximum_characteristic_biorthogonality_defect
        ),
        "maximum_imaginary_part": (
            analytic.maximum_characteristic_imaginary_part
        ),
        "maximum_descriptor_condition_number": (
            analytic.maximum_characteristic_descriptor_condition_number
        ),
        "incoming_inner_characteristics": (
            analytic.incoming_inner_characteristics
        ),
        "gates": gates,
        "passed": bool(all(gates.values())),
    }


def _geometry_step_report(
    configuration: dict,
    default_analytic,
) -> dict:
    native = configuration["candidate_native"]
    reports = {}
    passed = True
    for step in GEOMETRY_LOG_RADIUS_STEPS:
        if step == DEFAULT_GEOMETRY_LOG_RADIUS_STEP:
            candidate = default_analytic
        else:
            candidate = causal_five_field_radial_analytic_tangent(
                configuration["context"],
                configuration["base_primitives"],
                primitive_column_scales=(
                    native["primitive_column_scales"]
                ),
                conservation_row_scales=(
                    native["conservation_row_scales"]
                ),
                path_quadrature_order=PATH_QUADRATURE_ORDER,
                explicit_geometry_log_radius_step=step,
            )
        direction_defects = {}
        for name, direction in configuration["directions"].items():
            defect = _relative_difference(
                candidate.candidate_stationary_scaled_jacobian @ direction,
                (
                    default_analytic.candidate_stationary_scaled_jacobian
                    @ direction
                ),
            )
            direction_defects[name] = defect
            passed = bool(
                passed
                and defect <= MAXIMUM_GEOMETRY_STEP_TANGENT_DEFECT
            )
        label = f"{step:.0e}"
        reports[label] = {
            "direction_relative_defects": direction_defects,
            "maximum_direction_relative_defect": max(
                direction_defects.values()
            ),
        }
    return {
        "default_step": DEFAULT_GEOMETRY_LOG_RADIUS_STEP,
        "steps": reports,
        "maximum_direction_relative_defect": max(
            report["maximum_direction_relative_defect"]
            for report in reports.values()
        ),
        "passed": passed,
    }


def _pack_decisive_sparse(
    prefix: str,
    matrix: np.ndarray,
    decisive: dict[str, np.ndarray],
) -> None:
    _pack_sparse(prefix, csr_matrix(matrix), decisive)


def run(*, prepare_replay_inputs: bool = False) -> dict:
    started = time.perf_counter()
    identity = _validate_analyzed_git_identity()
    parent = json.loads(PARENT_SUMMARY.read_text(encoding="utf-8"))
    if (
        not parent["cross_grid_analytic_tangent_work_authorized"]
        or parent["derivative_choice_physical_sensitivity_authorized"]
        or not parent["parent_wp10c9d5_candidate_remains_rejected"]
    ):
        raise RuntimeError("WP10c9d5c0d binding status changed")
    if prepare_replay_inputs:
        _prepare_replay_inputs()
    replay_payload, replay_arrays = _load_replay_inputs()
    configurations = _configurations(replay_payload, replay_arrays)
    decisive: dict[str, np.ndarray] = {}
    grid_reports = {}
    passed = True

    for label in LABELS:
        print(f"WP10c9d5c0e: build {label}", flush=True)
        configuration = configurations[label]
        native = configuration["candidate_native"]
        analytic = causal_five_field_radial_analytic_tangent(
            configuration["context"],
            configuration["base_primitives"],
            primitive_column_scales=(
                native["primitive_column_scales"]
            ),
            conservation_row_scales=(
                native["conservation_row_scales"]
            ),
            path_quadrature_order=PATH_QUADRATURE_ORDER,
            explicit_geometry_log_radius_step=(
                DEFAULT_GEOMETRY_LOG_RADIUS_STEP
            ),
        )
        frozen = causal_five_field_frozen_analytic_tangent(
            analytic,
            native["production_generator"],
            native["descriptor"],
            configuration["anchor_storage_derivative"],
        )
        independent = _independent_reference_report(
            analytic,
            frozen,
            configuration,
        )
        linearity = _linearity_report(
            frozen.stationary_delta_scaled_jacobian,
            configuration["directions"],
        )
        spectral = _spectral_report(analytic)
        geometry_step = _geometry_step_report(
            configuration,
            analytic,
        )
        method_gates = {
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
            "independent_references": bool(independent["passed"]),
            "linearity": bool(linearity["passed"]),
            "spectral_conditioning": bool(spectral["passed"]),
            "geometry_step": bool(geometry_step["passed"]),
        }
        grid_passed = bool(all(method_gates.values()))
        passed = bool(passed and grid_passed)
        grid_reports[label] = {
            "n_cells": int(
                np.asarray(configuration["base_primitives"]).shape[0]
            ),
            "passed": grid_passed,
            "method_gates": method_gates,
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
            "independent_reference": independent,
            "linearity": linearity,
            "spectral": spectral,
            "geometry_step_sensitivity": geometry_step,
        }
        prefix = f"{label}__"
        decisive[prefix + "base_primitives"] = np.asarray(
            configuration["base_primitives"],
            dtype=float,
        )
        decisive[prefix + "characteristic_face_radii"] = np.asarray(
            analytic.characteristic_face_radii,
            dtype=float,
        )
        decisive[prefix + "characteristic_face_speeds_over_c"] = (
            np.asarray(
                analytic.characteristic_face_speeds_over_c,
                dtype=float,
            )
        )
        decisive[
            prefix + "characteristic_face_analytic_speeds_over_c"
        ] = np.asarray(
            analytic.characteristic_face_analytic_speeds_over_c,
            dtype=float,
        )
        decisive[prefix + "descriptor_condition_numbers"] = np.asarray(
            analytic.characteristic_face_descriptor_condition_numbers,
            dtype=float,
        )
        for name, direction in configuration["directions"].items():
            decisive[prefix + f"direction__{name}"] = np.asarray(
                direction,
                dtype=float,
            )
            decisive[prefix + f"stationary_delta_action__{name}"] = (
                np.asarray(
                    frozen.stationary_delta_scaled_jacobian @ direction,
                    dtype=float,
                )
            )
            decisive[prefix + f"candidate_generator_action__{name}"] = (
                np.asarray(
                    frozen.candidate_scaled_generator_per_s @ direction,
                    dtype=float,
                )
            )
        _pack_decisive_sparse(
            prefix + "analytic_stationary_delta",
            frozen.stationary_delta_scaled_jacobian,
            decisive,
        )
        _pack_decisive_sparse(
            prefix + "analytic_candidate_generator",
            frozen.candidate_scaled_generator_per_s,
            decisive,
        )
    classification = (
        "cross_grid_analytic_frozen_tangent_certified_"
        "derivative_choice_physical_sensitivity_authorized"
        if passed
        else "cross_grid_analytic_frozen_tangent_failed_"
        "physical_sensitivity_blocked"
    )
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)
    source_hashes, source_manifest = _source_manifest()
    gates = {
        "maximum_linearity_defect": MAXIMUM_LINEARITY_DEFECT,
        "maximum_block_ledger_defect": MAXIMUM_BLOCK_LEDGER_DEFECT,
        "maximum_reconstruction_defect": MAXIMUM_RECONSTRUCTION_DEFECT,
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
        "maximum_production_jvp_defect": (
            MAXIMUM_PRODUCTION_JVP_DEFECT
        ),
        "maximum_characteristic_eigenpair_defect": (
            MAXIMUM_CHARACTERISTIC_EIGENPAIR_DEFECT
        ),
        "maximum_characteristic_biorthogonality_defect": (
            MAXIMUM_CHARACTERISTIC_BIORTHOGONALITY_DEFECT
        ),
        "maximum_characteristic_imaginary_part": (
            MAXIMUM_CHARACTERISTIC_IMAGINARY_PART
        ),
        "maximum_characteristic_descriptor_condition": (
            MAXIMUM_CHARACTERISTIC_DESCRIPTOR_CONDITION
        ),
        "minimum_absolute_characteristic_speed": (
            MINIMUM_ABSOLUTE_CHARACTERISTIC_SPEED
        ),
        "minimum_characteristic_spectral_gap": (
            MINIMUM_CHARACTERISTIC_SPECTRAL_GAP
        ),
        "minimum_neighboring_subspace_cosine": (
            MINIMUM_NEIGHBORING_SUBSPACE_COSINE
        ),
        "maximum_geometry_step_tangent_defect": (
            MAXIMUM_GEOMETRY_STEP_TANGENT_DEFECT
        ),
    }
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "labels": LABELS,
        "direction_names": DIRECTION_NAMES,
        "derivative_orders": DERIVATIVE_ORDERS,
        "path_quadrature_order": PATH_QUADRATURE_ORDER,
        "target_radii_over_rg": TARGET_RADII_OVER_RG,
        "geometry_log_radius_steps": GEOMETRY_LOG_RADIUS_STEPS,
        "gates": gates,
    }
    _write_json(CONFIG_PATH, config)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        **identity,
        "parent_wp10c9d5c0d_summary_path": _relative(PARENT_SUMMARY),
        "parent_wp10c9d5c0d_summary_sha256": _sha256(PARENT_SUMMARY),
        "replay_inputs_path": _relative(REPLAY_INPUTS),
        "replay_inputs_sha256": _sha256(REPLAY_INPUTS),
        "replay_contexts_path": _relative(REPLAY_CONTEXTS),
        "replay_contexts_sha256": _sha256(REPLAY_CONTEXTS),
        "grids": grid_reports,
        "parent_wp10c9d5c0c_remains_rejected": True,
        "parent_wp10c9d5_candidate_remains_rejected": True,
        "parent_wp10c9d5b_branch_d_preserved": True,
        "cross_grid_analytic_tangent_certified": passed,
        "derivative_choice_physical_sensitivity_authorized": passed,
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
        "environment": wp10c9d5c0._environment(),
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_json(SUMMARY_PATH, summary)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        **identity,
        "generation_command": (
            "PYTHONPATH=src python3 "
            "scripts/"
            "run_causal_inner_cross_grid_analytic_tangent_wp10c9d5c0e.py"
        ),
        "method_scope": (
            "THREE-GRID ANALYTIC/AD-COMPATIBLE FROZEN-SUBSPACE "
            "TANGENT CERTIFICATION; PRODUCTION NEUTRAL"
        ),
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        "source_input_hashes": replay_payload["source_input_hashes"],
        "replay_inputs_sha256": _sha256(REPLAY_INPUTS),
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "scientific_status": "CERTIFIED" if passed else "REJECTED",
        "authorization_status": (
            "DERIVATIVE-CHOICE PHYSICAL SENSITIVITY ONLY"
            if passed
            else "DERIVATIVE METHOD WORK ONLY"
        ),
        "establishes": (
            "Whether the explicitly linear frozen-subspace analytic tangent "
            "passes the unchanged method, moving-projector reference, "
            "spectral-conditioning, and geometry-step gates on all three "
            "embedded grids."
        ),
        "does_not_establish": (
            "Physical export convergence, a recovery radius, a repaired "
            "operator, nonlinear convergence, fixed-Q closure, or reduced "
            "evolution."
        ),
    }
    _write_json(PROVENANCE_PATH, provenance)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-replay-inputs", action="store_true")
    arguments = parser.parse_args()
    print(
        json.dumps(
            _plain(
                run(
                    prepare_replay_inputs=(
                        arguments.prepare_replay_inputs
                    )
                )
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
