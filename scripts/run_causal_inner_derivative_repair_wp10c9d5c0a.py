#!/usr/bin/env python3
"""Run WP10c9d5c0a frozen-derivative mechanism and repair audit.

WP10c9d5c0 showed cancellation-amplified, approximately inverse-step
contamination in matched smooth directions on the medium and fine embedded
grids. This production-neutral package localizes that contamination by
physical block, constructs fourth- and sixth-order sparse derivatives from
one shared colored stencil, and propagates physical exports only when both
independent actions certify all three grids.
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
from scipy.sparse.linalg import splu


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_causal_inner_cross_grid_hardening_wp10c9d5c0 as wp10c9d5c0
import run_causal_inner_dynamic_localization_wp10c9d5b as wp10c9d5b
import run_causal_inner_frozen_discrimination_wp10c9d5 as wp10c9d5

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    causal_five_field_radial_candidate_face_flux,
    causal_five_field_radial_reduced_jacobian_pattern,
    causal_radial_colored_block_jacobian_family,
    causal_radial_high_order_directional_derivatives,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d5c0a"
ANALYZED_BASE_COMMIT = "f9d21e7bd8ede7c0548c93fc0b18021c30fde7fa"
ANALYZED_BASE_PARENT = "9c2a4ac6fa464a43fbaed3318cf5e1233a70fe55"
ANALYZED_BASE_TREE = "947a085c5476c8859765f54711ad1c86c9a22156"
THIS_RUNNER = (
    "scripts/run_causal_inner_derivative_repair_wp10c9d5c0a.py"
)

CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_derivative_repair_wp10c9d5c0a"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
CACHE_DIRECTORY = (
    ROOT
    / "outputs/checkpoints/"
    "causal_inner_derivative_repair_wp10c9d5c0a"
)
C0_SUMMARY = (
    ROOT
    / "results/canonical/"
    "causal_inner_cross_grid_hardening_wp10c9d5c0/summary.json"
)

LABELS = wp10c9d5c0.LABELS
N_FIELDS = 5
CONSERVATIVE_FIELDS = np.asarray((0, 2, 3), dtype=int)
BLOCK_NAMES = wp10c9d5c0.BLOCK_NAMES
PATH_QUADRATURE_ORDER = wp10c9d5c0.PATH_QUADRATURE_ORDER
TARGET_RADII_OVER_RG = wp10c9d5c0.TARGET_RADII_OVER_RG
STENCIL_HALO_CELLS = wp10c9d5c0.STENCIL_HALO_CELLS

HIGH_ORDER_STEP = 2.0e-4
COARSE_HIGH_ORDER_STEP = 4.0e-4
DERIVATIVE_ORDERS = (4, 6)
METHOD_NAMES = ("fourth_order_h2e4", "sixth_order_h2e4")
HELD_OUT_DIRECTION_SEEDS = (93061, 93062)
MECHANISM_STEPS = (1.0e-5, 2.0e-5, 4.0e-5)
MECHANISM_LABEL = LABELS[-1]
MECHANISM_DIRECTION = "global_inner_0"
CANCELLATION_ATTRIBUTION_DIRECTION = "heldout_near_excision_0"

MAXIMUM_DIRECT_ORDER_DIFFERENCE = 2.0e-5
MAXIMUM_DIRECT_SCALE_DIFFERENCE = 2.0e-5
MAXIMUM_MATRIX_ACTION_DEFECT = 5.0e-5
MAXIMUM_MATRIX_ORDER_DIFFERENCE = 2.0e-5
MAXIMUM_DERIVATIVE_EXPORT_DIFFERENCE = 5.0e-3
MAXIMUM_DERIVATIVE_TO_SPATIAL_RATIO = 0.10
MAXIMUM_DIRECT_FACE_PARITY_DEFECT = 5.0e-5
MAXIMUM_STRIDE_DEFECT = wp10c9d5c0.MAXIMUM_STRIDE_DEFECT

IMPLEMENTATION_SOURCES = (
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_radial_localization.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_radial_fluctuation.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_radial_frozen.py",
    "src/imri_qpe/layer3_minidisk_1d/__init__.py",
    THIS_RUNNER,
    "tests/test_causal_inner_radial_localization.py",
    "tests/test_causal_inner_derivative_repair_wp10c9d5c0a.py",
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
        raise RuntimeError("WP10c9d5c0a analyzed Git identity changed")
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


def _relative_difference(first: np.ndarray, second: np.ndarray) -> float:
    return wp10c9d5c0._relative_difference(first, second)


def _directions(configuration: dict) -> dict[str, np.ndarray]:
    parent = wp10c9d5c0._directions(configuration)
    return {
        "common_mode": parent["common_mode"],
        "calibration_global_inner_0": parent["global_inner_0"],
        "heldout_global_0": wp10c9d5c0._continuum_direction(
            configuration,
            seed=HELD_OUT_DIRECTION_SEEDS[0],
            lower_radius_over_rg=1.8,
            upper_radius_over_rg=11.75,
        ),
        "heldout_near_excision_0": wp10c9d5c0._continuum_direction(
            configuration,
            seed=HELD_OUT_DIRECTION_SEEDS[1],
            lower_radius_over_rg=1.8,
            upper_radius_over_rg=3.5,
        ),
    }


def _direct_high_order_report(
    configuration: dict,
    direction: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    zero = np.zeros(np.asarray(direction).size, dtype=float)
    fine = causal_radial_high_order_directional_derivatives(
        lambda values: wp10c9d5c0._scaled_delta(
            configuration,
            values,
        ),
        zero,
        direction,
        finite_difference_step=HIGH_ORDER_STEP,
        derivative_orders=DERIVATIVE_ORDERS,
    )
    coarse = causal_radial_high_order_directional_derivatives(
        lambda values: wp10c9d5c0._scaled_delta(
            configuration,
            values,
        ),
        zero,
        direction,
        finite_difference_step=COARSE_HIGH_ORDER_STEP,
        derivative_orders=(4,),
    )
    order_difference = _relative_difference(fine[4], fine[6])
    scale_difference = _relative_difference(fine[4], coarse[4])
    passed = bool(
        order_difference <= MAXIMUM_DIRECT_ORDER_DIFFERENCE
        and scale_difference <= MAXIMUM_DIRECT_SCALE_DIFFERENCE
    )
    return {
        "fourth_sixth_order_difference": order_difference,
        "fourth_order_scale_difference": scale_difference,
        "passed": passed,
    }, {
        "direction": np.asarray(direction, dtype=float),
        "fourth_order_action": np.asarray(fine[4], dtype=float),
        "sixth_order_action": np.asarray(fine[6], dtype=float),
        "coarse_fourth_order_action": np.asarray(coarse[4], dtype=float),
    }


def _mechanism_report(
    configuration: dict,
    direction: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    actions = {
        name: []
        for name in (*BLOCK_NAMES, "production")
    }
    for step in MECHANISM_STEPS:
        plus = wp10c9d5b._scaled_block_residuals(
            configuration,
            step * direction,
        )
        minus = wp10c9d5b._scaled_block_residuals(
            configuration,
            -step * direction,
        )
        for name in actions:
            actions[name].append(
                (plus[name] - minus[name]) / (2.0 * step)
            )
    arrays = {
        f"{name}__actions": np.asarray(values, dtype=float)
        for name, values in actions.items()
    }
    signs = {
        **{name: 1.0 for name in BLOCK_NAMES},
        "production": -1.0,
    }
    signed_changes = {
        name: signs[name] * (
            arrays[f"{name}__actions"][2]
            - arrays[f"{name}__actions"][1]
        )
        for name in actions
    }
    total_change = sum(
        signed_changes.values(),
        start=np.zeros_like(next(iter(signed_changes.values()))),
    )
    total_scale = max(
        float(np.linalg.norm(total_change)),
        np.finfo(float).tiny,
    )
    squared_sum = max(
        sum(float(np.dot(values, values)) for values in signed_changes.values()),
        np.finfo(float).tiny,
    )
    block_reports = {}
    for name, values in signed_changes.items():
        norm = float(np.linalg.norm(values))
        block_reports[name] = {
            "signed_change_norm": norm,
            "individual_squared_norm_fraction": (
                norm * norm / squared_sum
            ),
            "projection_onto_total": float(
                np.dot(values, total_change) / (total_scale * total_scale)
            ),
            "cosine_with_total": float(
                np.dot(values, total_change)
                / max(norm * total_scale, np.finfo(float).tiny)
            ),
        }
        arrays[f"{name}__signed_change"] = values
    total_by_cell = total_change.reshape(-1, N_FIELDS)
    cell_squared = np.sum(total_by_cell * total_by_cell, axis=1)
    cell_fraction = cell_squared / max(
        float(np.sum(cell_squared)),
        np.finfo(float).tiny,
    )
    edges = (
        np.asarray(configuration["context"].grid.edges, dtype=float)
        / float(configuration["context"].grid.gravitational_radius)
    )
    through = {}
    for radius in TARGET_RADII_OVER_RG:
        stop = int(np.flatnonzero(edges <= radius)[-1])
        through[str(radius)] = float(np.sum(cell_fraction[:stop]))
    arrays["total_signed_change"] = total_change
    arrays["cell_squared_fraction"] = cell_fraction
    return {
        "label": str(configuration["label"]),
        "direction": MECHANISM_DIRECTION,
        "steps": MECHANISM_STEPS,
        "total_change_norm": total_scale,
        "block_reports": block_reports,
        "dominant_cell": int(np.argmax(cell_fraction)),
        "dominant_cell_center_over_rg": float(
            configuration["context"].grid.centers[np.argmax(cell_fraction)]
            / configuration["context"].grid.gravitational_radius
        ),
        "squared_fraction_through_radius": through,
    }, arrays


def _block_cancellation_report(
    configuration: dict,
    family: dict[int, dict[str, csr_matrix]],
    direction: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    weights = {
        4: {
            -2: 1.0 / 12.0,
            -1: -2.0 / 3.0,
            1: 2.0 / 3.0,
            2: -1.0 / 12.0,
        },
        6: {
            -3: -1.0 / 60.0,
            -2: 3.0 / 20.0,
            -1: -3.0 / 4.0,
            1: 3.0 / 4.0,
            2: -3.0 / 20.0,
            3: 1.0 / 60.0,
        },
    }
    samples = {}
    for multiplier in (-3, -2, -1, 1, 2, 3):
        blocks = wp10c9d5b._scaled_block_residuals(
            configuration,
            multiplier * HIGH_ORDER_STEP * direction,
        )
        blocks["stationary_delta"] = sum(
            (
                np.asarray(blocks[name], dtype=float)
                for name in BLOCK_NAMES
            ),
            start=np.zeros_like(
                np.asarray(blocks["production"], dtype=float)
            ),
        ) - np.asarray(blocks["production"], dtype=float)
        samples[multiplier] = blocks

    face = wp10c9d5c0._actual_faces(configuration)[5.0]
    rows = wp10c9d5c0._region_rows(
        configuration,
        face,
        halo=True,
    )
    matrix_families = {
        order: {
            **family[order],
            "stationary_delta": _delta_matrix(family[order]),
        }
        for order in DERIVATIVE_ORDERS
    }
    arrays = {
        "direction": np.asarray(direction, dtype=float),
        "region_rows": rows,
    }
    reports = {}
    for name in (*BLOCK_NAMES, "production", "stationary_delta"):
        report = {}
        for order in DERIVATIVE_ORDERS:
            direct = sum(
                (
                    weight
                    * np.asarray(samples[multiplier][name], dtype=float)
                    for multiplier, weight in weights[order].items()
                ),
                start=np.zeros_like(
                    np.asarray(samples[-1][name], dtype=float)
                ),
            ) / HIGH_ORDER_STEP
            matrix = np.asarray(
                matrix_families[order][name] @ direction,
                dtype=float,
            )
            difference = matrix[rows] - direct[rows]
            arrays[f"{name}__order{order}__direct"] = direct[rows]
            arrays[f"{name}__order{order}__matrix"] = matrix[rows]
            arrays[f"{name}__order{order}__difference"] = difference
            report[f"order{order}"] = {
                "direct_action_norm": float(
                    np.linalg.norm(direct[rows])
                ),
                "absolute_matrix_action_defect": float(
                    np.linalg.norm(difference)
                ),
                "relative_matrix_action_defect": _relative_difference(
                    matrix[rows],
                    direct[rows],
                ),
            }
        reports[name] = report

    maximum_individual_defect = max(
        reports[name][f"order{order}"][
            "relative_matrix_action_defect"
        ]
        for name in (*BLOCK_NAMES, "production")
        for order in DERIVATIVE_ORDERS
    )
    maximum_delta_defect = max(
        reports["stationary_delta"][f"order{order}"][
            "relative_matrix_action_defect"
        ]
        for order in DERIVATIVE_ORDERS
    )
    return {
        "executed": True,
        "label": str(configuration["label"]),
        "direction": CANCELLATION_ATTRIBUTION_DIRECTION,
        "region": "through_5rg_plus_halo",
        "block_reports": reports,
        "maximum_individual_block_relative_defect": (
            maximum_individual_defect
        ),
        "maximum_stationary_delta_relative_defect": (
            maximum_delta_defect
        ),
        "cancellation_amplification": (
            maximum_delta_defect
            / max(maximum_individual_defect, np.finfo(float).tiny)
        ),
        "direct_stationary_delta_assembly_selected": True,
    }, arrays


def _cache_paths(label: str) -> tuple[Path, Path]:
    return (
        CACHE_DIRECTORY / f"{label}.json",
        CACHE_DIRECTORY / f"{label}_arrays.npz",
    )


def _pack_family(
    family: dict[int, dict[str, csr_matrix]],
) -> dict[str, np.ndarray]:
    packed = {}
    for order, matrices in family.items():
        for name, matrix in matrices.items():
            packed.update(
                wp10c9d5b._pack_sparse(f"order{order}__{name}", matrix)
            )
    return packed


def _unpack_family(
    arrays: dict[str, np.ndarray],
) -> dict[int, dict[str, csr_matrix]]:
    return {
        order: {
            name: wp10c9d5b._unpack_sparse(
                f"order{order}__{name}",
                arrays,
            )
            for name in (*BLOCK_NAMES, "production")
        }
        for order in DERIVATIVE_ORDERS
    }


def _build_or_load_high_order_blocks(
    configuration: dict,
    *,
    force: bool,
) -> tuple[dict, dict[int, dict[str, csr_matrix]]]:
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
        f"WP10c9d5c0a: building high-order blocks for {label}",
        flush=True,
    )
    started = time.perf_counter()
    pattern = causal_five_field_radial_reduced_jacobian_pattern(
        int(base.shape[0])
    )
    family = causal_radial_colored_block_jacobian_family(
        lambda values: wp10c9d5b._scaled_block_residuals(
            configuration,
            values,
        ),
        np.zeros(base.size, dtype=float),
        pattern,
        finite_difference_step=HIGH_ORDER_STEP,
        derivative_orders=DERIVATIVE_ORDERS,
    )
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


def _candidate_stationary(
    matrices: dict[str, csr_matrix],
) -> csr_matrix:
    shape = matrices[BLOCK_NAMES[0]].shape
    return sum(
        (matrices[name] for name in BLOCK_NAMES),
        start=csr_matrix(shape, dtype=float),
    )


def _delta_matrix(matrices: dict[str, csr_matrix]) -> csr_matrix:
    return _candidate_stationary(matrices) - matrices["production"]


def _matrix_action_report(
    configuration: dict,
    family: dict[int, dict[str, csr_matrix]],
    directions: dict[str, np.ndarray],
    direct_arrays: dict[str, dict[str, np.ndarray]],
) -> dict:
    reports = {}
    passed = True
    deltas = {
        order: _delta_matrix(family[order])
        for order in DERIVATIVE_ORDERS
    }
    faces = wp10c9d5c0._actual_faces(configuration)
    for name, direction in directions.items():
        direction_reports = {}
        actions = {
            order: np.asarray(deltas[order] @ direction, dtype=float)
            for order in DERIVATIVE_ORDERS
        }
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
                order4_defect = _relative_difference(
                    actions[4][rows],
                    direct_arrays[name]["fourth_order_action"][rows],
                )
                order6_defect = _relative_difference(
                    actions[6][rows],
                    direct_arrays[name]["sixth_order_action"][rows],
                )
                order_difference = _relative_difference(
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
    }


def _inner_flux_function(
    configuration: dict,
    increment: np.ndarray,
) -> np.ndarray:
    base = np.asarray(configuration["base_primitives"], dtype=float)
    columns = np.asarray(
        configuration["candidate_native"]["primitive_column_scales"],
        dtype=float,
    )
    charts = (
        base.ravel() + columns * np.asarray(increment, dtype=float)
    ).reshape(base.shape)
    _production, candidate = causal_five_field_radial_candidate_face_flux(
        configuration["context"],
        charts,
        0,
        quadrature_order=PATH_QUADRATURE_ORDER,
    )
    return np.asarray(candidate[CONSERVATIVE_FIELDS], dtype=float)


def _inner_flux_matrices(configuration: dict) -> dict[str, np.ndarray]:
    base = np.asarray(configuration["base_primitives"], dtype=float)
    zero = np.zeros(base.size, dtype=float)
    matrices = {
        method: np.zeros((CONSERVATIVE_FIELDS.size, base.size), dtype=float)
        for method in METHOD_NAMES
    }
    stop = min(base.size, N_FIELDS * wp10c9d5c0.INNER_FLUX_STENCIL_CELLS)
    for column in range(stop):
        direction = np.zeros_like(zero)
        direction[column] = 1.0
        actions = causal_radial_high_order_directional_derivatives(
            lambda values: _inner_flux_function(
                configuration,
                values,
            ),
            zero,
            direction,
            finite_difference_step=HIGH_ORDER_STEP,
            derivative_orders=DERIVATIVE_ORDERS,
        )
        matrices[METHOD_NAMES[0]][:, column] = actions[4]
        matrices[METHOD_NAMES[1]][:, column] = actions[6]
    return matrices


def _method_generators(
    configuration: dict,
    family: dict[int, dict[str, csr_matrix]],
) -> dict[str, np.ndarray]:
    native = configuration["candidate_native"]
    production = np.asarray(native["production_generator"], dtype=float)
    descriptor = np.asarray(native["descriptor"], dtype=float)
    factor = splu(csc_matrix(descriptor), permc_spec="COLAMD")
    return {
        METHOD_NAMES[0]: production - factor.solve(
            _delta_matrix(family[4]).toarray()
        ),
        METHOD_NAMES[1]: production - factor.solve(
            _delta_matrix(family[6]).toarray()
        ),
    }


def _direct_selected_face_actions(
    configuration: dict,
    direction: np.ndarray,
    faces: np.ndarray,
) -> dict[int, np.ndarray]:
    base = np.asarray(configuration["base_primitives"], dtype=float)
    columns = np.asarray(
        configuration["candidate_native"]["primitive_column_scales"],
        dtype=float,
    )
    zero = np.zeros(base.size, dtype=float)

    def fluxes(increment: np.ndarray) -> np.ndarray:
        charts = (
            base.ravel() + columns * np.asarray(increment, dtype=float)
        ).reshape(base.shape)
        values = []
        for face in faces:
            _production, candidate = (
                causal_five_field_radial_candidate_face_flux(
                    configuration["context"],
                    charts,
                    int(face),
                    quadrature_order=PATH_QUADRATURE_ORDER,
                )
            )
            values.append(candidate[CONSERVATIVE_FIELDS])
        return np.asarray(values, dtype=float).ravel()

    return causal_radial_high_order_directional_derivatives(
        fluxes,
        zero,
        direction,
        finite_difference_step=HIGH_ORDER_STEP,
        derivative_orders=DERIVATIVE_ORDERS,
    )


def _face_parity_report(
    configuration: dict,
    histories: dict[str, dict],
) -> dict:
    times = np.asarray(configuration["times"], dtype=float)
    time_indices = np.asarray(
        [
            int(round(fraction * (times.size - 1)))
            for fraction in wp10c9d5c0.DIRECT_FACE_TIME_FRACTIONS
        ],
        dtype=int,
    )
    faces = wp10c9d5c0._selected_direct_faces(configuration)
    defects = {method: [] for method in METHOD_NAMES}
    for order, method in zip(
        DERIVATIVE_ORDERS,
        METHOD_NAMES,
        strict=True,
    ):
        for time_index in time_indices:
            direction = np.asarray(
                histories[method]["scaled_state"],
                dtype=float,
            )[time_index]
            direct = _direct_selected_face_actions(
                configuration,
                direction,
                faces,
            )
            assembled = np.asarray(
                histories[method]["face_fluxes"],
                dtype=float,
            )[time_index, faces, :].ravel()
            defects[method].append(
                _relative_difference(direct[order], assembled)
            )
    reports = {
        method: {
            "relative_defects": values,
            "maximum_relative_defect": max(values),
            "passed": bool(
                max(values) <= MAXIMUM_DIRECT_FACE_PARITY_DEFECT
            ),
        }
        for method, values in defects.items()
    }
    return {
        "time_indices": time_indices,
        "face_indices": faces,
        "methods": reports,
        "maximum_relative_defect": max(
            report["maximum_relative_defect"]
            for report in reports.values()
        ),
        "passed": bool(all(report["passed"] for report in reports.values())),
    }


def _recovery_is_stable(reports: dict[str, dict]) -> bool:
    indices = [
        reports[method]["recovery_surface_index"]
        for method in METHOD_NAMES
    ]
    if all(index is None for index in indices):
        return True
    if any(index is None for index in indices):
        return False
    return max(int(index) for index in indices) - min(
        int(index) for index in indices
    ) <= 1


def _physical_sensitivity(
    configurations: dict,
    families: dict,
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
        generators = _method_generators(configuration, families[label])
        inner_matrices = _inner_flux_matrices(configuration)
        for order, method in zip(
            DERIVATIVE_ORDERS,
            METHOD_NAMES,
            strict=True,
        ):
            print(f"WP10c9d5c0a: propagate {label} {method}", flush=True)
            history = wp10c9d5c0._observable_history(
                configuration,
                generators[method],
                families[label][order],
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
        face_reports[label] = _face_parity_report(
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
        first_cell = _relative_difference(
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
    recovery_stable = _recovery_is_stable(recovery)
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


def _configurations() -> dict:
    return {
        label: configuration
        for label, configuration
        in wp10c9d5._common_configurations(False).items()
        if label in LABELS
    }


def build_label(label: str, *, force: bool) -> dict:
    _validate_analyzed_git_identity()
    configurations = _configurations()
    if label not in configurations:
        raise ValueError(f"unknown embedded label: {label}")
    report, _family = _build_or_load_high_order_blocks(
        configurations[label],
        force=force,
    )
    return report


def run(*, force: bool = False) -> dict:
    started = time.perf_counter()
    identity = _validate_analyzed_git_identity()
    parent = json.loads(C0_SUMMARY.read_text(encoding="utf-8"))
    if (
        parent["directional_derivative_passed"]
        or parent["wp10c9d5c1_extended_localization_authorized"]
    ):
        raise RuntimeError("WP10c9d5c0 binding stop changed")
    configurations = _configurations()
    decisive: dict[str, np.ndarray] = {}

    print("WP10c9d5c0a: localize cancellation mechanism", flush=True)
    mechanism_configuration = configurations[MECHANISM_LABEL]
    mechanism_direction = wp10c9d5c0._directions(
        mechanism_configuration
    )[MECHANISM_DIRECTION]
    mechanism, mechanism_arrays = _mechanism_report(
        mechanism_configuration,
        mechanism_direction,
    )
    for name, values in mechanism_arrays.items():
        decisive[f"mechanism__{name}"] = values

    direct_reports = {}
    direct_arrays = {}
    direct_passed = True
    for label in LABELS:
        print(f"WP10c9d5c0a: high-order JVP audit {label}", flush=True)
        configuration = configurations[label]
        directions = _directions(configuration)
        label_reports = {}
        label_arrays = {}
        for name, direction in directions.items():
            print(f"  direction {name}", flush=True)
            report, arrays = _direct_high_order_report(
                configuration,
                direction,
            )
            label_reports[name] = report
            label_arrays[name] = arrays
            direct_passed = bool(direct_passed and report["passed"])
            for array_name, values in arrays.items():
                decisive[
                    f"{label}__{name}__direct__{array_name}"
                ] = values
        direct_reports[label] = label_reports
        direct_arrays[label] = label_arrays

    cache_reports = {}
    families = {}
    matrix_reports = {}
    matrix_passed = False
    if direct_passed:
        matrix_passed = True
        for label in LABELS:
            report, family = _build_or_load_high_order_blocks(
                configurations[label],
                force=force,
            )
            cache_reports[label] = report
            families[label] = family
            matrix_report = _matrix_action_report(
                configurations[label],
                family,
                _directions(configurations[label]),
                direct_arrays[label],
            )
            matrix_reports[label] = matrix_report
            matrix_passed = bool(
                matrix_passed and matrix_report["passed"]
            )

    cancellation_attribution = {
        "executed": False,
        "direct_stationary_delta_assembly_selected": False,
    }
    if direct_passed and not matrix_passed:
        configuration = configurations[MECHANISM_LABEL]
        cancellation_attribution, attribution_arrays = (
            _block_cancellation_report(
                configuration,
                families[MECHANISM_LABEL],
                _directions(configuration)[
                    CANCELLATION_ATTRIBUTION_DIRECTION
                ],
            )
        )
        for name, values in attribution_arrays.items():
            decisive[f"cancellation_attribution__{name}"] = values

    physical = {
        "executed": False,
        "passed": False,
    }
    if direct_passed and matrix_passed:
        physical, physical_arrays = _physical_sensitivity(
            configurations,
            families,
        )
        decisive.update(physical_arrays)

    repair_passed = bool(
        direct_passed and matrix_passed and physical["passed"]
    )
    classification = (
        "high_order_frozen_derivative_repair_passed_"
        "extended_localization_authorized"
        if repair_passed
        else
        "high_order_frozen_derivative_repair_failed_"
        "extended_localization_blocked"
    )

    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)
    source_hashes, source_manifest = _source_manifest()
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "labels": LABELS,
        "high_order_step": HIGH_ORDER_STEP,
        "coarse_high_order_step": COARSE_HIGH_ORDER_STEP,
        "derivative_orders": DERIVATIVE_ORDERS,
        "method_names": METHOD_NAMES,
        "held_out_direction_seeds": HELD_OUT_DIRECTION_SEEDS,
        "mechanism_steps": MECHANISM_STEPS,
        "gates": {
            "maximum_direct_order_difference": (
                MAXIMUM_DIRECT_ORDER_DIFFERENCE
            ),
            "maximum_direct_scale_difference": (
                MAXIMUM_DIRECT_SCALE_DIFFERENCE
            ),
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
        "parent_wp10c9d5c0_summary_path": _relative(C0_SUMMARY),
        "parent_wp10c9d5c0_summary_sha256": _sha256(C0_SUMMARY),
        "parent_wp10c9d5_candidate_remains_rejected": True,
        "parent_wp10c9d5b_branch_d_preserved": True,
        "mechanism_report": mechanism,
        "cancellation_attribution": cancellation_attribution,
        "direct_reports": direct_reports,
        "direct_high_order_passed": direct_passed,
        "cache_reports": cache_reports,
        "matrix_reports": matrix_reports,
        "matrix_high_order_passed": matrix_passed,
        "physical_sensitivity": physical,
        "derivative_repair_passed": repair_passed,
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
            "scripts/run_causal_inner_derivative_repair_wp10c9d5c0a.py"
        ),
        "method_scope": (
            "HIGH-ORDER FROZEN DERIVATIVE REPAIR; PRODUCTION NEUTRAL"
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
            _relative(C0_SUMMARY): _sha256(C0_SUMMARY),
        },
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "establishes": (
            "Whether cancellation-amplified finite-difference noise can be "
            "replaced by mutually consistent fourth- and sixth-order sparse "
            "frozen derivatives on all three embedded grids."
        ),
        "does_not_establish": (
            "A repaired physical operator, nonlinear convergence, fixed-Q "
            "closure, or reduced slow evolution."
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
        help="rebuild high-order derivative caches",
    )
    parser.add_argument(
        "--build-label",
        choices=LABELS,
        help="build only one high-order block cache",
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
