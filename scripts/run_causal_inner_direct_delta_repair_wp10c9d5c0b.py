#!/usr/bin/env python3
"""Run WP10c9d5c0b direct stationary-correction derivative audit.

WP10c9d5c0a showed that individually accurate candidate and production
matrices lose accuracy when subtracted to form a much smaller stationary
correction. This production-neutral package instead forms that correction at
each residual sample and differentiates it directly. All c0a directions,
stencils, regions, and gates remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import scipy
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import norm as sparse_norm
from scipy.sparse.linalg import splu


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_causal_inner_cross_grid_hardening_wp10c9d5c0 as wp10c9d5c0
import run_causal_inner_derivative_repair_wp10c9d5c0a as wp10c9d5c0a
import run_causal_inner_dynamic_localization_wp10c9d5b as wp10c9d5b

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    causal_five_field_radial_reduced_jacobian_pattern,
    causal_radial_colored_block_jacobian_family,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d5c0b"
ANALYZED_BASE_COMMIT = "5c88fa02f8f25fa62e9e0fdb648e66974bca38d3"
ANALYZED_BASE_PARENT = "f9d21e7bd8ede7c0548c93fc0b18021c30fde7fa"
ANALYZED_BASE_TREE = "ddcc241bc0a6bac03ba00c42fbcbb5b2056b3787"
THIS_RUNNER = (
    "scripts/run_causal_inner_direct_delta_repair_wp10c9d5c0b.py"
)

CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_direct_delta_repair_wp10c9d5c0b"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
CACHE_DIRECTORY = (
    ROOT
    / "outputs/checkpoints/"
    "causal_inner_direct_delta_repair_wp10c9d5c0b"
)
C0A_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_derivative_repair_wp10c9d5c0a"
)
C0A_SUMMARY = C0A_DIRECTORY / "summary.json"
C0A_ARRAYS = C0A_DIRECTORY / "decisive_arrays.npz"

LABELS = wp10c9d5c0a.LABELS
BLOCK_NAMES = wp10c9d5c0a.BLOCK_NAMES
DERIVATIVE_ORDERS = wp10c9d5c0a.DERIVATIVE_ORDERS
METHOD_NAMES = wp10c9d5c0a.METHOD_NAMES
HIGH_ORDER_STEP = wp10c9d5c0a.HIGH_ORDER_STEP
PATH_QUADRATURE_ORDER = wp10c9d5c0a.PATH_QUADRATURE_ORDER

MAXIMUM_MATRIX_ACTION_DEFECT = (
    wp10c9d5c0a.MAXIMUM_MATRIX_ACTION_DEFECT
)
MAXIMUM_MATRIX_ORDER_DIFFERENCE = (
    wp10c9d5c0a.MAXIMUM_MATRIX_ORDER_DIFFERENCE
)
MAXIMUM_DERIVATIVE_EXPORT_DIFFERENCE = (
    wp10c9d5c0a.MAXIMUM_DERIVATIVE_EXPORT_DIFFERENCE
)
MAXIMUM_DERIVATIVE_TO_SPATIAL_RATIO = (
    wp10c9d5c0a.MAXIMUM_DERIVATIVE_TO_SPATIAL_RATIO
)
MAXIMUM_DIRECT_FACE_PARITY_DEFECT = (
    wp10c9d5c0a.MAXIMUM_DIRECT_FACE_PARITY_DEFECT
)
MAXIMUM_STRIDE_DEFECT = wp10c9d5c0a.MAXIMUM_STRIDE_DEFECT
FAIL_FAST_ON_FIRST_FAILED_GRID = True

IMPLEMENTATION_SOURCES = (
    THIS_RUNNER,
    "scripts/run_causal_inner_derivative_repair_wp10c9d5c0a.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_radial_localization.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_radial_fluctuation.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_radial_frozen.py",
    "src/imri_qpe/layer3_minidisk_1d/__init__.py",
    "tests/test_causal_inner_direct_delta_repair_wp10c9d5c0b.py",
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
        raise RuntimeError("WP10c9d5c0b analyzed Git identity changed")
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


def _environment() -> dict:
    blas = np.__config__.CONFIG.get("Build Dependencies", {}).get("blas", {})
    lapack = np.__config__.CONFIG.get("Build Dependencies", {}).get(
        "lapack",
        {},
    )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "blas": blas,
        "lapack": lapack,
    }


def _configurations() -> dict:
    return wp10c9d5c0a._configurations()


def _stationary_delta_blocks(
    configuration: dict,
    increment: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "stationary_delta": wp10c9d5c0._scaled_delta(
            configuration,
            increment,
        )
    }


def _cache_paths(label: str) -> tuple[Path, Path]:
    return (
        CACHE_DIRECTORY / f"{label}.json",
        CACHE_DIRECTORY / f"{label}_arrays.npz",
    )


def _pack_family(
    family: dict[int, csr_matrix],
) -> dict[str, np.ndarray]:
    packed = {}
    for order, matrix in family.items():
        packed.update(
            wp10c9d5b._pack_sparse(
                f"order{order}__stationary_delta",
                matrix,
            )
        )
    return packed


def _unpack_family(
    arrays: dict[str, np.ndarray],
) -> dict[int, csr_matrix]:
    return {
        order: wp10c9d5b._unpack_sparse(
            f"order{order}__stationary_delta",
            arrays,
        )
        for order in DERIVATIVE_ORDERS
    }


def _build_or_load_direct_delta(
    configuration: dict,
    *,
    force: bool,
) -> tuple[dict, dict[int, csr_matrix]]:
    label = str(configuration["label"])
    json_path, arrays_path = _cache_paths(label)
    base = np.asarray(configuration["base_primitives"], dtype=float)
    native = configuration["candidate_native"]
    contract = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "label": label,
        "base_primitives_sha256": _array_sha256(base),
        "grid_edges_sha256": _array_sha256(
            configuration["context"].grid.edges
        ),
        "stored_stationary_delta_sha256": _array_sha256(
            native["stationary_delta"]
        ),
        "finite_difference_step": HIGH_ORDER_STEP,
        "derivative_orders": DERIVATIVE_ORDERS,
        "path_quadrature_order": PATH_QUADRATURE_ORDER,
        "assembly": "differentiate_samplewise_stationary_delta_directly",
    }
    if not force and json_path.exists() and arrays_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if (
            all(
                payload.get(key) == _plain(value)
                for key, value in contract.items()
            )
            and payload.get("arrays_sha256") == _sha256(arrays_path)
        ):
            with np.load(arrays_path, allow_pickle=False) as source:
                packed = {
                    name: np.asarray(source[name])
                    for name in source.files
                }
            return payload, _unpack_family(packed)

    print(
        f"WP10c9d5c0b: building direct delta for {label}",
        flush=True,
    )
    started = time.perf_counter()
    pattern = causal_five_field_radial_reduced_jacobian_pattern(
        int(base.shape[0])
    )
    block_family = causal_radial_colored_block_jacobian_family(
        lambda values: _stationary_delta_blocks(
            configuration,
            values,
        ),
        np.zeros(base.size, dtype=float),
        pattern,
        finite_difference_step=HIGH_ORDER_STEP,
        derivative_orders=DERIVATIVE_ORDERS,
    )
    family = {
        order: blocks["stationary_delta"]
        for order, blocks in block_family.items()
    }
    packed = _pack_family(family)
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **packed)
    payload = {
        **contract,
        "arrays_path": _relative(arrays_path),
        "arrays_sha256": _sha256(arrays_path),
        "n_cells": int(base.shape[0]),
        "wall_seconds": time.perf_counter() - started,
    }
    _write_json(json_path, payload)
    return payload, family


def _load_direct_evidence() -> tuple[dict, dict]:
    summary = json.loads(C0A_SUMMARY.read_text(encoding="utf-8"))
    if (
        not summary["direct_high_order_passed"]
        or summary["matrix_high_order_passed"]
        or summary["physical_sensitivity"]["executed"]
        or not summary["cancellation_attribution"][
            "direct_stationary_delta_assembly_selected"
        ]
    ):
        raise RuntimeError("WP10c9d5c0a binding evidence changed")
    direct_arrays = {}
    decisive = {}
    with np.load(C0A_ARRAYS, allow_pickle=False) as source:
        for label in LABELS:
            label_arrays = {}
            for name in summary["direct_reports"][label]:
                arrays = {}
                for array_name in (
                    "direction",
                    "fourth_order_action",
                    "sixth_order_action",
                    "coarse_fourth_order_action",
                ):
                    key = (
                        f"{label}__{name}__direct__{array_name}"
                    )
                    values = np.asarray(source[key], dtype=float)
                    if (
                        _array_sha256(values)
                        != summary["decisive_array_hashes"][key]
                    ):
                        raise RuntimeError(
                            f"WP10c9d5c0a direct evidence changed: {key}"
                        )
                    arrays[array_name] = values
                    decisive[f"parent_direct__{key}"] = values
                label_arrays[name] = arrays
            direct_arrays[label] = label_arrays
    return direct_arrays, decisive


def _matrix_action_report(
    configuration: dict,
    family: dict[int, csr_matrix],
    direct_arrays: dict[str, dict[str, np.ndarray]],
) -> tuple[dict, dict[str, np.ndarray]]:
    reports = {}
    decisive = {}
    passed = True
    faces = wp10c9d5c0._actual_faces(configuration)
    for name, arrays in direct_arrays.items():
        direction = arrays["direction"]
        actions = {
            order: np.asarray(family[order] @ direction, dtype=float)
            for order in DERIVATIVE_ORDERS
        }
        for order, action in actions.items():
            decisive[f"{name}__order{order}__matrix_action"] = action
        direction_reports = {}
        for target, face in faces.items():
            for halo in (False, True):
                region_name = (
                    f"through_{target:g}rg"
                    + ("_plus_halo" if halo else "")
                )
                rows = wp10c9d5c0._region_rows(
                    configuration,
                    face,
                    halo=halo,
                )
                order4_defect = wp10c9d5c0a._relative_difference(
                    actions[4][rows],
                    arrays["fourth_order_action"][rows],
                )
                order6_defect = wp10c9d5c0a._relative_difference(
                    actions[6][rows],
                    arrays["sixth_order_action"][rows],
                )
                order_difference = wp10c9d5c0a._relative_difference(
                    actions[4][rows],
                    actions[6][rows],
                )
                region_passed = bool(
                    max(order4_defect, order6_defect)
                    <= MAXIMUM_MATRIX_ACTION_DEFECT
                    and order_difference
                    <= MAXIMUM_MATRIX_ORDER_DIFFERENCE
                )
                passed = bool(passed and region_passed)
                direction_reports[region_name] = {
                    "fourth_order_matrix_action_defect": order4_defect,
                    "sixth_order_matrix_action_defect": order6_defect,
                    "matrix_order_difference": order_difference,
                    "passed": region_passed,
                }
        reports[name] = direction_reports
    return {
        "directions": reports,
        "passed": passed,
    }, decisive


def _worst_matrix_defects(report: dict) -> dict:
    worst_action = {
        "value": 0.0,
        "direction": None,
        "region": None,
    }
    worst_order = {
        "value": 0.0,
        "direction": None,
        "region": None,
    }
    for direction, regions in report["directions"].items():
        for region, values in regions.items():
            action = max(
                values["fourth_order_matrix_action_defect"],
                values["sixth_order_matrix_action_defect"],
            )
            if action > worst_action["value"]:
                worst_action = {
                    "value": action,
                    "direction": direction,
                    "region": region,
                }
            order = values["matrix_order_difference"]
            if order > worst_order["value"]:
                worst_order = {
                    "value": order,
                    "direction": direction,
                    "region": region,
                }
    return {
        "worst_matrix_action_defect": worst_action,
        "worst_matrix_order_difference": worst_order,
    }


def _linearity_equivalence_report(
    configuration: dict,
    direct_delta_family: dict[int, csr_matrix],
) -> tuple[dict, dict[str, np.ndarray]]:
    cache_report, block_family = (
        wp10c9d5c0a._build_or_load_high_order_blocks(
            configuration,
            force=False,
        )
    )
    reports = {}
    decisive = {}
    maximum_relative_difference = 0.0
    for order in DERIVATIVE_ORDERS:
        old = wp10c9d5c0a._delta_matrix(block_family[order]).tocsr()
        new = direct_delta_family[order].tocsr()
        difference = (new - old).tocsr()
        old_norm = float(sparse_norm(old))
        new_norm = float(sparse_norm(new))
        difference_norm = float(sparse_norm(difference))
        relative = difference_norm / max(
            old_norm,
            new_norm,
            np.finfo(float).tiny,
        )
        maximum_relative_difference = max(
            maximum_relative_difference,
            relative,
        )
        maximum_entry = (
            float(np.max(np.abs(difference.data)))
            if difference.nnz
            else 0.0
        )
        reports[str(order)] = {
            "separately_differentiated_matrix_norm": old_norm,
            "samplewise_delta_matrix_norm": new_norm,
            "difference_norm": difference_norm,
            "relative_difference": relative,
            "maximum_absolute_entry_difference": maximum_entry,
            "difference_nnz": int(difference.nnz),
        }
        for prefix, matrix in (
            ("separately_differentiated", old),
            ("samplewise_delta", new),
            ("difference", difference),
        ):
            for name, values in wp10c9d5b._pack_sparse(
                f"order{order}__{prefix}",
                matrix,
            ).items():
                decisive[name] = values
    return {
        "label": str(configuration["label"]),
        "old_high_order_cache_path": cache_report["arrays_path"],
        "old_high_order_cache_sha256": cache_report["arrays_sha256"],
        "orders": reports,
        "maximum_relative_difference": maximum_relative_difference,
        "interpretation": (
            "Finite-difference differentiation is linear to roundoff; "
            "samplewise candidate-minus-production subtraction reproduces "
            "the separately differentiated matrix subtraction."
        ),
    }, decisive


def _direct_delta_generators(
    configuration: dict,
    family: dict[int, csr_matrix],
) -> dict[str, np.ndarray]:
    native = configuration["candidate_native"]
    production = np.asarray(native["production_generator"], dtype=float)
    descriptor = np.asarray(native["descriptor"], dtype=float)
    factor = splu(csc_matrix(descriptor), permc_spec="COLAMD")
    return {
        method: production - factor.solve(family[order].toarray())
        for order, method in zip(
            DERIVATIVE_ORDERS,
            METHOD_NAMES,
            strict=True,
        )
    }


def _physical_sensitivity(
    configurations: dict,
    direct_delta_families: dict[str, dict[int, csr_matrix]],
    block_families: dict[
        str,
        dict[int, dict[str, csr_matrix]],
    ],
) -> tuple[dict, dict[str, np.ndarray]]:
    observable_scales = wp10c9d5c0._fixed_observable_scales(configurations)
    face_scales = wp10c9d5c0._fixed_face_scales(configurations)
    histories = {method: {} for method in METHOD_NAMES}
    face_reports = {}
    decisive = {
        "fixed_observable_scales": observable_scales,
        "fixed_face_scales": face_scales,
    }
    for label in LABELS:
        configuration = configurations[label]
        generators = _direct_delta_generators(
            configuration,
            direct_delta_families[label],
        )
        inner_matrices = wp10c9d5c0a._inner_flux_matrices(configuration)
        for order, method in zip(
            DERIVATIVE_ORDERS,
            METHOD_NAMES,
            strict=True,
        ):
            print(f"WP10c9d5c0b: propagate {label} {method}", flush=True)
            history = wp10c9d5c0._observable_history(
                configuration,
                generators[method],
                block_families[label][order],
                inner_matrices[method],
            )
            histories[method][label] = history
            for name in (
                "times",
                "signals",
                "cumulative_signals",
                "face_fluxes",
            ):
                decisive[f"{method}__{label}__{name}"] = np.asarray(
                    history[name],
                    dtype=float,
                )
            decisive[f"{method}__{label}__first_cell_state"] = np.asarray(
                history["state"],
                dtype=float,
            )[:, 0, :]
        face_reports[label] = wp10c9d5c0a._face_parity_report(
            configuration,
            {
                method: histories[method][label]
                for method in METHOD_NAMES
            },
        )

    method_differences = {}
    maximum_difference = 0.0
    for label in LABELS:
        times = np.asarray(histories[METHOD_NAMES[1]][label]["times"])
        duration = max(float(times[-1]), np.finfo(float).tiny)
        signal = wp10c9d5c0._maximum_component_rms_difference(
            histories[METHOD_NAMES[0]][label]["signals"],
            histories[METHOD_NAMES[1]][label]["signals"],
            observable_scales,
        )
        cumulative = wp10c9d5c0._maximum_component_rms_difference(
            histories[METHOD_NAMES[0]][label]["cumulative_signals"],
            histories[METHOD_NAMES[1]][label]["cumulative_signals"],
            observable_scales * duration,
        )
        first_cell = wp10c9d5c0a._relative_difference(
            histories[METHOD_NAMES[0]][label]["state"][:, 0, :],
            histories[METHOD_NAMES[1]][label]["state"][:, 0, :],
        )
        maximum = max(signal, cumulative)
        maximum_difference = max(maximum_difference, maximum)
        method_differences[label] = {
            "signal_difference": signal,
            "cumulative_difference": cumulative,
            "first_cell_state_difference": first_cell,
            "maximum_export_difference": maximum,
        }

    medium = histories[METHOD_NAMES[1]][LABELS[1]]
    fine = histories[METHOD_NAMES[1]][LABELS[2]]
    duration = max(
        float(np.asarray(fine["times"])[-1]),
        np.finfo(float).tiny,
    )
    spatial_signal = wp10c9d5c0._maximum_component_rms_difference(
        medium["signals"],
        fine["signals"],
        observable_scales,
    )
    spatial_cumulative = wp10c9d5c0._maximum_component_rms_difference(
        medium["cumulative_signals"],
        fine["cumulative_signals"],
        observable_scales * duration,
    )
    binding_spatial = max(spatial_signal, spatial_cumulative)
    derivative_ratio = maximum_difference / max(
        binding_spatial,
        np.finfo(float).tiny,
    )

    common_radii, face_maps = wp10c9d5c0._common_face_maps(configurations)
    decisive["common_face_radii_over_rg"] = common_radii
    recovery = {
        method: wp10c9d5c0._recovery_report(
            histories[method],
            common_radii,
            face_maps,
            face_scales,
        )
        for method in METHOD_NAMES
    }
    recovery_stable = wp10c9d5c0a._recovery_is_stable(recovery)
    stride = wp10c9d5c0._stride_report(
        histories[METHOD_NAMES[1]],
        observable_scales,
        face_scales,
    )
    face_passed = bool(
        all(report["passed"] for report in face_reports.values())
    )
    passed = bool(
        maximum_difference <= MAXIMUM_DERIVATIVE_EXPORT_DIFFERENCE
        and derivative_ratio <= MAXIMUM_DERIVATIVE_TO_SPATIAL_RATIO
        and recovery_stable
        and face_passed
        and stride["passed"]
    )
    return {
        "executed": True,
        "method_differences": method_differences,
        "maximum_derivative_export_difference": maximum_difference,
        "binding_medium_fine_signal_difference": spatial_signal,
        "binding_medium_fine_cumulative_difference": spatial_cumulative,
        "binding_medium_fine_spatial_difference": binding_spatial,
        "derivative_to_spatial_ratio": derivative_ratio,
        "recovery_reports": recovery,
        "recovery_location_stable": recovery_stable,
        "face_parity_reports": face_reports,
        "face_parity_passed": face_passed,
        "stride_report": stride,
        "stride_passed": stride["passed"],
        "passed": passed,
    }, decisive


def build_label(label: str, *, force: bool) -> dict:
    _validate_analyzed_git_identity()
    configurations = _configurations()
    if label not in configurations:
        raise ValueError(f"unknown embedded label: {label}")
    report, _family = _build_or_load_direct_delta(
        configurations[label],
        force=force,
    )
    return report


def run(*, force: bool = False) -> dict:
    started = time.perf_counter()
    identity = _validate_analyzed_git_identity()
    parent = json.loads(C0A_SUMMARY.read_text(encoding="utf-8"))
    if (
        parent["derivative_repair_passed"]
        or parent["wp10c9d5c1_extended_localization_authorized"]
        or not parent["cancellation_attribution"][
            "direct_stationary_delta_assembly_selected"
        ]
    ):
        raise RuntimeError("WP10c9d5c0a binding decision changed")
    direct_arrays, decisive = _load_direct_evidence()
    configurations = _configurations()

    cache_reports = {}
    direct_delta_families = {}
    matrix_reports = {}
    matrix_passed = True
    attempted_labels = []
    fail_fast_trigger = None
    for label in LABELS:
        report, family = _build_or_load_direct_delta(
            configurations[label],
            force=force,
        )
        attempted_labels.append(label)
        cache_reports[label] = report
        direct_delta_families[label] = family
        matrix_report, action_arrays = _matrix_action_report(
            configurations[label],
            family,
            direct_arrays[label],
        )
        matrix_reports[label] = matrix_report
        matrix_passed = bool(matrix_passed and matrix_report["passed"])
        for name, values in action_arrays.items():
            decisive[f"{label}__{name}"] = values
        if FAIL_FAST_ON_FIRST_FAILED_GRID and not matrix_report["passed"]:
            fail_fast_trigger = {
                "label": label,
                "reason": "direct_delta_matrix_action_gate_failed",
                **_worst_matrix_defects(matrix_report),
            }
            break

    unattempted_labels = [
        label for label in LABELS if label not in attempted_labels
    ]
    all_three_grid_matrix_gate_executed = not unattempted_labels
    matrix_passed = bool(
        matrix_passed and all_three_grid_matrix_gate_executed
    )
    linearity_equivalence = {
        "executed": False,
    }
    if fail_fast_trigger is not None:
        failed_label = str(fail_fast_trigger["label"])
        linearity_equivalence, equivalence_arrays = (
            _linearity_equivalence_report(
                configurations[failed_label],
                direct_delta_families[failed_label],
            )
        )
        linearity_equivalence["executed"] = True
        for name, values in equivalence_arrays.items():
            decisive[f"linearity_equivalence__{name}"] = values

    physical = {
        "executed": False,
        "passed": False,
    }
    block_cache_reports = {}
    if matrix_passed:
        block_families = {}
        for label in LABELS:
            report, family = (
                wp10c9d5c0a._build_or_load_high_order_blocks(
                    configurations[label],
                    force=False,
                )
            )
            block_cache_reports[label] = report
            block_families[label] = family
        physical, physical_arrays = _physical_sensitivity(
            configurations,
            direct_delta_families,
            block_families,
        )
        decisive.update(physical_arrays)

    repair_passed = bool(matrix_passed and physical["passed"])
    classification = (
        "direct_stationary_delta_derivative_passed_"
        "extended_localization_authorized"
        if repair_passed
        else
        "direct_stationary_delta_derivative_failed_"
        "extended_localization_blocked"
    )

    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)
    source_hashes, source_manifest = _source_manifest()
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "labels": LABELS,
        "assembly": "differentiate_samplewise_stationary_delta_directly",
        "high_order_step": HIGH_ORDER_STEP,
        "derivative_orders": DERIVATIVE_ORDERS,
        "method_names": METHOD_NAMES,
        "inherited_direct_evidence_path": _relative(C0A_ARRAYS),
        "fail_fast_on_first_failed_grid": (
            FAIL_FAST_ON_FIRST_FAILED_GRID
        ),
        "gates": {
            "maximum_matrix_action_defect": (
                MAXIMUM_MATRIX_ACTION_DEFECT
            ),
            "maximum_matrix_order_difference": (
                MAXIMUM_MATRIX_ORDER_DIFFERENCE
            ),
            "maximum_derivative_export_difference": (
                MAXIMUM_DERIVATIVE_EXPORT_DIFFERENCE
            ),
            "maximum_derivative_to_spatial_ratio": (
                MAXIMUM_DERIVATIVE_TO_SPATIAL_RATIO
            ),
            "maximum_direct_face_parity_defect": (
                MAXIMUM_DIRECT_FACE_PARITY_DEFECT
            ),
            "maximum_stride_defect": MAXIMUM_STRIDE_DEFECT,
        },
    }
    _write_json(CONFIG_PATH, config)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        **identity,
        "parent_wp10c9d5c0a_summary_path": _relative(C0A_SUMMARY),
        "parent_wp10c9d5c0a_summary_sha256": _sha256(C0A_SUMMARY),
        "parent_wp10c9d5c0a_arrays_path": _relative(C0A_ARRAYS),
        "parent_wp10c9d5c0a_arrays_sha256": _sha256(C0A_ARRAYS),
        "parent_wp10c9d5c0a_remains_rejected": True,
        "parent_wp10c9d5_candidate_remains_rejected": True,
        "parent_wp10c9d5b_branch_d_preserved": True,
        "direct_high_order_evidence_inherited_and_verified": True,
        "cache_reports": cache_reports,
        "attempted_labels": attempted_labels,
        "unattempted_labels": unattempted_labels,
        "fail_fast_trigger": fail_fast_trigger,
        "all_three_grid_matrix_gate_executed": (
            all_three_grid_matrix_gate_executed
        ),
        "matrix_reports": matrix_reports,
        "direct_delta_matrix_passed": matrix_passed,
        "linearity_equivalence": linearity_equivalence,
        "block_cache_reports": block_cache_reports,
        "physical_sensitivity": physical,
        "direct_delta_repair_passed": repair_passed,
        "wp10c9d5c1_extended_localization_authorized": repair_passed,
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
        "environment": _environment(),
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_json(SUMMARY_PATH, summary)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        **identity,
        "generation_command": (
            "PYTHONPATH=src python3 "
            "scripts/run_causal_inner_direct_delta_repair_"
            "wp10c9d5c0b.py"
        ),
        "method_scope": (
            "DIRECT STATIONARY-CORRECTION FROZEN DERIVATIVE; "
            "PRODUCTION NEUTRAL"
        ),
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        "scientific_status": (
            "DIAGNOSTIC ONLY" if repair_passed else "REJECTED"
        ),
        "authorization_status": (
            "EXTENDED LOCALIZATION ONLY"
            if repair_passed
            else "EXTENDED LOCALIZATION BLOCKED"
        ),
        "source_input_hashes": {
            _relative(C0A_SUMMARY): _sha256(C0A_SUMMARY),
            _relative(C0A_ARRAYS): _sha256(C0A_ARRAYS),
        },
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "establishes": (
            "Whether differentiating the samplewise candidate-minus-"
            "production stationary correction avoids the cancellation "
            "failure of separately differentiated matrices."
        ),
        "does_not_establish": (
            "A repaired physical operator, recovery radius, dominant "
            "physical mechanism, nonlinear convergence, fixed-Q closure, "
            "or reduced slow evolution."
        ),
    }
    _write_json(PROVENANCE_PATH, provenance)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild direct stationary-correction caches",
    )
    parser.add_argument(
        "--build-label",
        choices=LABELS,
        help="build only one direct stationary-correction cache",
    )
    arguments = parser.parse_args()
    if arguments.build_label is not None:
        print(
            json.dumps(
                _plain(
                    build_label(arguments.build_label, force=arguments.force)
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    print(
        json.dumps(
            _plain(run(force=arguments.force)),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
