#!/usr/bin/env python3
"""Run WP10c9d5c0c N128 uncolored/additivity discrimination.

The rejected c0b matrix is first compared with a sum of independently
evaluated one-cell directional derivatives. Only if that additivity gate
passes are selected high-impact columns recomputed without coloring.
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
from scipy.sparse import csr_matrix


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
    causal_radial_high_order_directional_derivatives,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d5c0c"
ANALYZED_BASE_COMMIT = "508e5c284c2eaf7305efb45ae30a437b29dabb33"
ANALYZED_BASE_PARENT = "5c88fa02f8f25fa62e9e0fdb648e66974bca38d3"
ANALYZED_BASE_TREE = "6fd889e2f6acc27304f1b243011c8bf255ae5d0b"
THIS_RUNNER = (
    "scripts/run_causal_inner_uncolored_additivity_wp10c9d5c0c.py"
)

LABEL = "N128_exterior_N128_inner_c48"
DIRECTION_NAME = "calibration_global_inner_0"
DERIVATIVE_ORDERS = (4, 6)
FINITE_DIFFERENCE_STEP = 2.0e-4
SELECTED_COLUMN_COUNT = 12
SELECTION_RADIUS_OVER_RG = 8.0
MAXIMUM_ACTION_DEFECT = 5.0e-5
FAIL_FAST_AFTER_CELL_ADDITIVITY = True

C0B_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_direct_delta_repair_wp10c9d5c0b"
)
C0B_SUMMARY = C0B_DIRECTORY / "summary.json"
C0B_ARRAYS = C0B_DIRECTORY / "decisive_arrays.npz"
CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_uncolored_additivity_wp10c9d5c0c"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
CACHE_DIRECTORY = (
    ROOT
    / "outputs/checkpoints/"
    "causal_inner_uncolored_additivity_wp10c9d5c0c"
)
CELL_CACHE_JSON = CACHE_DIRECTORY / "cell_partitions.json"
CELL_CACHE_ARRAYS = CACHE_DIRECTORY / "cell_partitions_arrays.npz"
COLUMN_CACHE_JSON = CACHE_DIRECTORY / "selected_columns.json"
COLUMN_CACHE_ARRAYS = CACHE_DIRECTORY / "selected_columns_arrays.npz"

IMPLEMENTATION_SOURCES = (
    THIS_RUNNER,
    "scripts/run_causal_inner_direct_delta_repair_wp10c9d5c0b.py",
    "scripts/run_causal_inner_derivative_repair_wp10c9d5c0a.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_radial_localization.py",
    "tests/test_causal_inner_uncolored_additivity_wp10c9d5c0c.py",
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
        raise RuntimeError("WP10c9d5c0c analyzed Git identity changed")
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


def _configuration() -> dict:
    return wp10c9d5c0b._configurations()[LABEL]


def _load_parent_evidence() -> tuple[dict, dict, dict[int, csr_matrix]]:
    summary = json.loads(C0B_SUMMARY.read_text(encoding="utf-8"))
    if (
        summary["direct_delta_repair_passed"]
        or summary["wp10c9d5c1_extended_localization_authorized"]
        or summary["fail_fast_trigger"]["label"] != LABEL
        or not summary["linearity_equivalence"]["executed"]
    ):
        raise RuntimeError("WP10c9d5c0b binding evidence changed")
    with np.load(C0B_ARRAYS, allow_pickle=False) as source:
        arrays = {
            name: np.asarray(source[name])
            for name in source.files
        }
    for name, expected in summary["decisive_array_hashes"].items():
        if _array_sha256(arrays[name]) != expected:
            raise RuntimeError(f"WP10c9d5c0b array changed: {name}")
    prefix = f"parent_direct__{LABEL}__{DIRECTION_NAME}__direct__"
    direct = {
        "direction": np.asarray(arrays[f"{prefix}direction"], dtype=float),
        "fourth_order_action": np.asarray(
            arrays[f"{prefix}fourth_order_action"],
            dtype=float,
        ),
        "sixth_order_action": np.asarray(
            arrays[f"{prefix}sixth_order_action"],
            dtype=float,
        ),
    }
    matrices = {
        order: wp10c9d5b._unpack_sparse(
            (
                f"linearity_equivalence__order{order}"
                "__samplewise_delta"
            ),
            arrays,
        )
        for order in DERIVATIVE_ORDERS
    }
    return summary, direct, matrices


def _direct_actions(
    configuration: dict,
    direction: np.ndarray,
) -> dict[int, np.ndarray]:
    zero = np.zeros_like(np.asarray(direction, dtype=float))
    return causal_radial_high_order_directional_derivatives(
        lambda values: wp10c9d5c0b._stationary_delta_blocks(
            configuration,
            values,
        )["stationary_delta"],
        zero,
        np.asarray(direction, dtype=float),
        finite_difference_step=FINITE_DIFFERENCE_STEP,
        derivative_orders=DERIVATIVE_ORDERS,
    )


def _cache_contract(
    configuration: dict,
    direction: np.ndarray,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "label": LABEL,
        "direction_name": DIRECTION_NAME,
        "direction_sha256": _array_sha256(direction),
        "base_primitives_sha256": _array_sha256(
            configuration["base_primitives"]
        ),
        "grid_edges_sha256": _array_sha256(
            configuration["context"].grid.edges
        ),
        "finite_difference_step": FINITE_DIFFERENCE_STEP,
        "derivative_orders": DERIVATIVE_ORDERS,
    }


def _load_valid_cache(
    json_path: Path,
    arrays_path: Path,
    contract: dict,
) -> tuple[dict, dict[str, np.ndarray]] | None:
    if not json_path.exists() or not arrays_path.exists():
        return None
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if (
        not all(
            payload.get(key) == _plain(value)
            for key, value in contract.items()
        )
        or payload.get("arrays_sha256") != _sha256(arrays_path)
    ):
        return None
    with np.load(arrays_path, allow_pickle=False) as source:
        arrays = {
            name: np.asarray(source[name])
            for name in source.files
        }
    return payload, arrays


def _build_or_load_cell_partitions(
    configuration: dict,
    direction: np.ndarray,
    *,
    force: bool,
) -> tuple[dict, dict[str, np.ndarray]]:
    contract = {
        **_cache_contract(configuration, direction),
        "assembly": "independent_one_cell_directional_derivatives",
    }
    if not force:
        cached = _load_valid_cache(
            CELL_CACHE_JSON,
            CELL_CACHE_ARRAYS,
            contract,
        )
        if cached is not None:
            return cached
    by_cell = np.asarray(direction, dtype=float).reshape(-1, 5)
    cells = np.flatnonzero(np.any(by_cell != 0.0, axis=1))
    actions = {
        order: np.zeros((cells.size, direction.size), dtype=float)
        for order in DERIVATIVE_ORDERS
    }
    started = time.perf_counter()
    for index, cell in enumerate(cells):
        part = np.zeros_like(by_cell)
        part[cell] = by_cell[cell]
        evaluated = _direct_actions(configuration, part.ravel())
        for order in DERIVATIVE_ORDERS:
            actions[order][index] = evaluated[order]
        print(
            "WP10c9d5c0c: cell "
            f"{index + 1}/{cells.size} (native cell {cell})",
            flush=True,
        )
    arrays = {
        "cells": cells,
        **{
            f"order{order}_cell_actions": actions[order]
            for order in DERIVATIVE_ORDERS
        },
    }
    CELL_CACHE_ARRAYS.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CELL_CACHE_ARRAYS, **arrays)
    payload = {
        **contract,
        "cell_count": int(cells.size),
        "arrays_path": _relative(CELL_CACHE_ARRAYS),
        "arrays_sha256": _sha256(CELL_CACHE_ARRAYS),
        "wall_seconds": time.perf_counter() - started,
    }
    _write_json(CELL_CACHE_JSON, payload)
    return payload, arrays


def _region_rows(configuration: dict) -> dict[str, np.ndarray]:
    return {
        f"through_{target:g}rg_plus_halo": (
            wp10c9d5c0b.wp10c9d5c0._region_rows(
                configuration,
                face,
                halo=True,
            )
        )
        for target, face
        in wp10c9d5c0b.wp10c9d5c0._actual_faces(
            configuration
        ).items()
    }


def _cell_additivity_report(
    configuration: dict,
    direct: dict,
    matrices: dict[int, csr_matrix],
    arrays: dict[str, np.ndarray],
) -> tuple[dict, dict[str, np.ndarray]]:
    reports = {}
    decisive = {
        "cells": np.asarray(arrays["cells"], dtype=int),
    }
    passed = True
    for order, direct_name in (
        (4, "fourth_order_action"),
        (6, "sixth_order_action"),
    ):
        cell_actions = np.asarray(
            arrays[f"order{order}_cell_actions"],
            dtype=float,
        )
        summed = np.sum(cell_actions, axis=0)
        colored = np.asarray(
            matrices[order] @ direct["direction"],
            dtype=float,
        )
        decisive[f"order{order}_cell_actions"] = cell_actions
        decisive[f"order{order}_summed_action"] = summed
        decisive[f"order{order}_colored_action"] = colored
        order_report = {}
        for region, rows in _region_rows(configuration).items():
            defect = wp10c9d5c0a._relative_difference(
                summed[rows],
                direct[direct_name][rows],
            )
            colored_defect = wp10c9d5c0a._relative_difference(
                colored[rows],
                direct[direct_name][rows],
            )
            cell_to_colored = wp10c9d5c0a._relative_difference(
                summed[rows],
                colored[rows],
            )
            region_passed = bool(defect <= MAXIMUM_ACTION_DEFECT)
            passed = bool(passed and region_passed)
            order_report[region] = {
                "relative_defect": defect,
                "colored_to_direct_relative_defect": colored_defect,
                "cell_sum_to_colored_relative_defect": cell_to_colored,
                "passed": region_passed,
            }
        reports[str(order)] = order_report
    return {
        "assembly": "sum_of_independent_one_cell_jvps",
        "cell_count": int(np.asarray(arrays["cells"]).size),
        "orders": reports,
        "passed": passed,
    }, decisive


def _selected_columns(
    configuration: dict,
    direction: np.ndarray,
    matrix: csr_matrix,
) -> tuple[np.ndarray, np.ndarray]:
    face = wp10c9d5c0b.wp10c9d5c0._actual_faces(
        configuration
    )[SELECTION_RADIUS_OVER_RG]
    rows = wp10c9d5c0b.wp10c9d5c0._region_rows(
        configuration,
        face,
        halo=True,
    )
    candidates = np.flatnonzero(np.asarray(direction) != 0.0)
    weights = np.asarray(
        [
            abs(float(direction[column]))
            * float(np.linalg.norm(matrix[rows, column].toarray()))
            for column in candidates
        ],
        dtype=float,
    )
    ordering = np.lexsort((candidates, -weights))
    selected = candidates[ordering[:SELECTED_COLUMN_COUNT]]
    return selected.astype(int), weights[ordering[:SELECTED_COLUMN_COUNT]]


def _build_or_load_selected_columns(
    configuration: dict,
    direction: np.ndarray,
    matrix: csr_matrix,
    *,
    force: bool,
) -> tuple[dict, dict[str, np.ndarray]]:
    columns, weights = _selected_columns(
        configuration,
        direction,
        matrix,
    )
    contract = {
        **_cache_contract(configuration, direction),
        "assembly": "independent_uncolored_selected_columns",
        "selected_columns": columns,
        "selection_weights": weights,
        "selection_radius_over_rg": SELECTION_RADIUS_OVER_RG,
    }
    if not force:
        cached = _load_valid_cache(
            COLUMN_CACHE_JSON,
            COLUMN_CACHE_ARRAYS,
            contract,
        )
        if cached is not None:
            return cached
    actions = {
        order: np.zeros((columns.size, direction.size), dtype=float)
        for order in DERIVATIVE_ORDERS
    }
    started = time.perf_counter()
    for index, column in enumerate(columns):
        basis = np.zeros_like(direction)
        basis[column] = 1.0
        evaluated = _direct_actions(configuration, basis)
        for order in DERIVATIVE_ORDERS:
            actions[order][index] = evaluated[order]
        print(
            "WP10c9d5c0c: column "
            f"{index + 1}/{columns.size} (native column {column})",
            flush=True,
        )
    arrays = {
        "selected_columns": columns,
        "selection_weights": weights,
        **{
            f"order{order}_column_actions": actions[order]
            for order in DERIVATIVE_ORDERS
        },
    }
    COLUMN_CACHE_ARRAYS.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(COLUMN_CACHE_ARRAYS, **arrays)
    payload = {
        **contract,
        "arrays_path": _relative(COLUMN_CACHE_ARRAYS),
        "arrays_sha256": _sha256(COLUMN_CACHE_ARRAYS),
        "wall_seconds": time.perf_counter() - started,
    }
    _write_json(COLUMN_CACHE_JSON, payload)
    return payload, arrays


def _selected_column_report(
    configuration: dict,
    matrices: dict[int, csr_matrix],
    arrays: dict[str, np.ndarray],
) -> tuple[dict, dict[str, np.ndarray]]:
    columns = np.asarray(arrays["selected_columns"], dtype=int)
    reports = {}
    decisive = {
        "selected_columns": columns,
        "selection_weights": np.asarray(
            arrays["selection_weights"],
            dtype=float,
        ),
    }
    passed = True
    for order in DERIVATIVE_ORDERS:
        direct_actions = np.asarray(
            arrays[f"order{order}_column_actions"],
            dtype=float,
        )
        decisive[f"order{order}_direct_column_actions"] = direct_actions
        order_reports = {}
        for region, rows in _region_rows(configuration).items():
            colored_actions = np.asarray(
                [
                    matrices[order][:, column].toarray().ravel()
                    for column in columns
                ],
                dtype=float,
            )
            defects = np.asarray(
                [
                    wp10c9d5c0a._relative_difference(
                        direct_actions[index, rows],
                        colored_actions[index, rows],
                    )
                    for index in range(columns.size)
                ],
                dtype=float,
            )
            decisive[
                f"order{order}_{region}_colored_column_actions"
            ] = colored_actions[:, rows]
            decisive[f"order{order}_{region}_relative_defects"] = defects
            maximum = float(np.max(defects))
            region_passed = bool(maximum <= MAXIMUM_ACTION_DEFECT)
            passed = bool(passed and region_passed)
            order_reports[region] = {
                "maximum_relative_defect": maximum,
                "per_column_relative_defects": defects,
                "passed": region_passed,
            }
        reports[str(order)] = order_reports
    return {
        "assembly": "independent_selected_column_jvps",
        "selected_column_count": int(columns.size),
        "orders": reports,
        "passed": passed,
    }, decisive


def run(*, force: bool = False) -> dict:
    started = time.perf_counter()
    identity = _validate_analyzed_git_identity()
    parent, direct, matrices = _load_parent_evidence()
    configuration = _configuration()

    cell_cache, cell_arrays = _build_or_load_cell_partitions(
        configuration,
        direct["direction"],
        force=force,
    )
    cell_report, decisive = _cell_additivity_report(
        configuration,
        direct,
        matrices,
        cell_arrays,
    )
    decisive["direction"] = np.asarray(direct["direction"], dtype=float)
    decisive["parent_fourth_order_action"] = np.asarray(
        direct["fourth_order_action"],
        dtype=float,
    )
    decisive["parent_sixth_order_action"] = np.asarray(
        direct["sixth_order_action"],
        dtype=float,
    )

    column_cache = {}
    column_report = {
        "executed": False,
        "passed": False,
    }
    if cell_report["passed"]:
        column_cache, column_arrays = _build_or_load_selected_columns(
            configuration,
            direct["direction"],
            matrices[4],
            force=force,
        )
        column_report, column_decisive = _selected_column_report(
            configuration,
            matrices,
            column_arrays,
        )
        column_report["executed"] = True
        decisive.update(
            {
                f"selected_columns__{name}": values
                for name, values in column_decisive.items()
            }
        )

    if not cell_report["passed"]:
        classification = (
            "independent_cell_additivity_failed_"
            "finite_difference_linear_tangent_blocked"
        )
        next_route = "analytic_or_ad_compatible_linear_tangent"
    elif not column_report["passed"]:
        classification = (
            "selected_uncolored_columns_differ_from_colored_"
            "sparse_assembly_repair_authorized"
        )
        next_route = "repair_coloring_or_sparsity_then_retest"
    else:
        classification = (
            "selected_uncolored_columns_match_colored_"
            "basis_accumulation_remains_blocked"
        )
        next_route = "analytic_or_ad_compatible_linear_tangent"

    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)
    source_hashes, source_manifest = _source_manifest()
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "label": LABEL,
        "direction_name": DIRECTION_NAME,
        "finite_difference_step": FINITE_DIFFERENCE_STEP,
        "derivative_orders": DERIVATIVE_ORDERS,
        "selected_column_count": SELECTED_COLUMN_COUNT,
        "selection_radius_over_rg": SELECTION_RADIUS_OVER_RG,
        "fail_fast_after_cell_additivity": (
            FAIL_FAST_AFTER_CELL_ADDITIVITY
        ),
        "gates": {
            "maximum_action_defect": MAXIMUM_ACTION_DEFECT,
        },
    }
    _write_json(CONFIG_PATH, config)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "selected_next_route": next_route,
        **identity,
        "parent_summary_path": _relative(C0B_SUMMARY),
        "parent_summary_sha256": _sha256(C0B_SUMMARY),
        "parent_arrays_path": _relative(C0B_ARRAYS),
        "parent_arrays_sha256": _sha256(C0B_ARRAYS),
        "parent_wp10c9d5c0b_remains_rejected": True,
        "parent_wp10c9d5_candidate_remains_rejected": True,
        "parent_wp10c9d5b_branch_d_preserved": True,
        "cell_partition_cache": cell_cache,
        "cell_additivity_report": cell_report,
        "selected_column_cache": column_cache,
        "selected_column_report": column_report,
        "explicit_finite_difference_matrix_authorized": False,
        "uncolored_full_matrix_followup_authorized": bool(
            cell_report["passed"] and not column_report["passed"]
        ),
        "matrix_free_finite_difference_jvp_authorized": False,
        "analytic_or_ad_linear_tangent_work_authorized": bool(
            next_route == "analytic_or_ad_compatible_linear_tangent"
        ),
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
            "scripts/run_causal_inner_uncolored_additivity_"
            "wp10c9d5c0c.py"
        ),
        "method_scope": (
            "N128 FROZEN DERIVATIVE ASSEMBLY DISCRIMINATION; "
            "PRODUCTION NEUTRAL"
        ),
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        "scientific_status": "REJECTED",
        "authorization_status": (
            "DERIVATIVE METHOD WORK ONLY; PHYSICAL LOCALIZATION BLOCKED"
        ),
        "source_input_hashes": {
            _relative(C0B_SUMMARY): _sha256(C0B_SUMMARY),
            _relative(C0B_ARRAYS): _sha256(C0B_ARRAYS),
        },
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "establishes": (
            "Whether the c0b direct-JVP mismatch is already present when "
            "the failed direction is assembled from independent one-cell "
            "JVPs, and conditionally whether selected uncolored columns "
            "differ from the colored sparse matrix."
        ),
        "does_not_establish": (
            "A cross-grid tangent, physical recovery radius, dominant "
            "physical mechanism, repaired operator, nonlinear convergence, "
            "fixed-Q closure, or reduced slow evolution."
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
        help="rebuild the independent directional caches",
    )
    arguments = parser.parse_args()
    print(
        json.dumps(
            _plain(run(force=arguments.force)),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
