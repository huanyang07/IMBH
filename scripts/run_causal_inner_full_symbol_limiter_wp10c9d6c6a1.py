#!/usr/bin/env python3
"""Attribute the complete-symbol limiter left by WP10c9d6c6a.

This package changes no operator and does not relax the failed c6a gates.
It allocates the complete finite-time propagator difference over the exact
DAE components, checks the allocation across N128/N256/N512, and performs a
bounded variable-radius ray-ordered accumulation preflight.  The ray audit
can authorize a separate windowed-continuum contract audit; it cannot itself
certify packets.
"""

from __future__ import annotations

import argparse
import csv
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
from scipy.linalg import expm


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_causal_inner_packet_resolution_wp10c9d6c6a as c6a

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    causal_five_field_continuum_local_symbol,
    causal_five_field_local_symbol_stencil,
    causal_five_field_matched_principal_eigenvalues,
    causal_five_field_monolithic_frozen_tangent,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_continuum_truncation import (  # noqa: E402
    build_causal_five_field_continuum_background,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_full_symbol_limiter import (  # noqa: E402
    CausalNormalizedLocalDAESymbol,
    causal_match_symbol_eigenvalues_to_tracked_branches,
    causal_continuum_normalized_local_dae,
    causal_local_dae_component_stencil,
    causal_symbol_shapley_attribution,
    causal_track_symbol_eigenbranches,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c6a1"
ANALYZED_BASE_COMMIT = "76e34e8ef0b4688b0c371c27cb2288c880419961"
ANALYZED_BASE_PARENT = "14bc3e753c2530ef8799d5ad092854156a6c6551"
ANALYZED_BASE_TREE = "5b4b37e5e2c2d0affaeed90bf547892f5a7808ed"
THIS_RUNNER = (
    "scripts/run_causal_inner_full_symbol_limiter_wp10c9d6c6a1.py"
)

LABELS = c6a.LABELS
REFERENCE_LABEL = c6a.REFERENCE_LABEL
AUDIT_RADII_OVER_RG = c6a.SYMBOL_RADII_OVER_RG
AUDIT_THETA_VALUES = (0.10, 0.17, 0.18, 0.20, 0.30)
AUDIT_TIMES_S = (0.015625, 0.03125, 0.0625, 0.125)
CROSS_GRID_REFERENCE_THETAS = (0.18, 0.20)
LIMITER_RADIUS_OVER_RG = 8.0
LIMITER_THETA = 0.18

# Prospectively frozen diagnostic gates.  None modifies the failed c6a
# 0.025 semigroup budget or theta >= 0.20 usable-range requirement.
MAXIMUM_COMPONENT_CLOSURE_DEFECT = 1.0e-11
MAXIMUM_GENERATOR_PARITY_DEFECT = 1.0e-11
MAXIMUM_SHAPLEY_CLOSURE_DEFECT = 1.0e-11
SIGNIFICANT_PROPAGATOR_CONTRIBUTION = 2.5e-4
MINIMUM_SIGNIFICANT_COMPONENT_ORDER = 1.25
MINIMUM_TIME_ACCUMULATION_EXPONENT = 0.75
MAXIMUM_TIME_ACCUMULATION_EXPONENT = 1.25
MAXIMUM_NUMERICAL_CONTINUUM_PROPAGATOR_NORM_RATIO = 1.25

# The ray calculation is an architecture preflight, not a replacement
# packet contract.  A result below the unchanged 0.025 budget is reported;
# a result below the existing total 0.05 export budget may authorize a
# separate, independent windowed-continuum contract audit.
RAY_START_RADII_OVER_RG = (5.0, 8.0, 11.0)
RAY_THETA_VALUES = (0.18, 0.20)
RAY_HORIZON_S = 0.125
RAY_PRIMARY_STEP_S = 0.00125
RAY_SECONDARY_STEP_S = 0.0025
RAY_GROUP_DERIVATIVE_STEP = 1.0e-4
RAY_BRANCH_TABLE_BASE_NODES = 257
MINIMUM_RAY_BRANCH_OVERLAP = 0.90
MAXIMUM_RAY_INTEGRATOR_TO_ERROR_RATIO = 0.10
MAXIMUM_RAY_REFERENCE_TO_ERROR_RATIO = 0.10
MAXIMUM_RAY_PREFLIGHT_ERROR = 0.05
PACKET_CONTRACT_ERROR = c6a.MAXIMUM_COMPLETE_SEMIGROUP_ERROR
MINIMUM_USABLE_THETA = c6a.MINIMUM_CERTIFIED_THETA

PARENT_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_packet_resolution_wp10c9d6c6a"
)
PARENT_CONFIG = PARENT_DIRECTORY / "config.json"
PARENT_SUMMARY = PARENT_DIRECTORY / "summary.json"
PARENT_ARRAYS = PARENT_DIRECTORY / "decisive_arrays.npz"
PARENT_PROVENANCE = PARENT_DIRECTORY / "provenance.json"
CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_full_symbol_limiter_wp10c9d6c6a1"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
MIDPOINT_SUMMARY_PATH = (
    CANONICAL_DIRECTORY / "midpoint_preflight_summary.json"
)
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

IMPLEMENTATION_SOURCES = (
    THIS_RUNNER,
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_full_symbol_limiter.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_packet_resolution.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_monolithic_tangent.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_continuum_truncation.py",
    "tests/test_causal_inner_full_symbol_limiter.py",
    "tests/test_causal_inner_full_symbol_limiter_wp10c9d6c6a1.py",
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
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
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


def _refresh_sha256s(directory: Path) -> None:
    entries = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.name == "SHA256SUMS.txt":
            continue
        entries.append(f"{_sha256(path)}  {path.name}")
    (directory / "SHA256SUMS.txt").write_text(
        "\n".join(entries) + "\n",
        encoding="utf-8",
    )


def _refresh_canonical_catalog() -> None:
    rows: list[dict[str, str | int]] = []
    for case in sorted(CANONICAL_DIRECTORY.parent.iterdir()):
        if not case.is_dir():
            continue
        provenance_path = case / "provenance.json"
        if not provenance_path.is_file():
            continue
        provenance = json.loads(
            provenance_path.read_text(encoding="utf-8")
        )
        status = provenance.get(
            "scientific_status",
            provenance.get("numerical_status", "DIAGNOSTIC ONLY"),
        )
        for path in sorted(case.iterdir()):
            if not path.is_file():
                continue
            rows.append(
                {
                    "case": case.name,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "scientific_status": status,
                }
            )
    CANONICAL_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with CANONICAL_MANIFEST.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "case",
                "path",
                "bytes",
                "sha256",
                "scientific_status",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    summary = json.loads(CANONICAL_SUMMARY.read_text(encoding="utf-8"))
    summary.update(
        {
            "case_count": len({str(row["case"]) for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": ANALYZED_BASE_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, summary)


def _git_value(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_analyzed_git_identity() -> dict:
    resolved = _git_value("rev-parse", ANALYZED_BASE_COMMIT)
    parent = _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
    tree = _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
    if (
        resolved != ANALYZED_BASE_COMMIT
        or parent != ANALYZED_BASE_PARENT
        or tree != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("WP10c9d6c6a1 analyzed git identity changed")
    return {
        "analyzed_base_commit": resolved,
        "analyzed_base_parent": parent,
        "analyzed_base_tree": tree,
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
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
    }


def _load_parent() -> tuple[dict, dict[str, np.ndarray]]:
    summary = json.loads(PARENT_SUMMARY.read_text(encoding="utf-8"))
    if (
        summary["classification"]
        != "symbol_derived_packet_resolution_contract_failed"
        or summary["passed"]
        or summary["authorized_next"] != "none"
        or summary["operator_changed"]
    ):
        raise RuntimeError("WP10c9d6c6a binding status changed")
    with np.load(PARENT_ARRAYS, allow_pickle=False) as source:
        arrays = {
            name: np.array(source[name], copy=True)
            for name in source.files
        }
    return summary, arrays


def _relative_propagator_error(
    numerical: np.ndarray,
    continuum: np.ndarray,
    interval: float,
) -> tuple[float, float]:
    numerical_step = expm(float(interval) * numerical)
    continuum_step = expm(float(interval) * continuum)
    scale = max(
        float(np.linalg.norm(numerical_step)),
        float(np.linalg.norm(continuum_step)),
        np.finfo(float).tiny,
    )
    error = float(
        np.linalg.norm(numerical_step - continuum_step) / scale
    )
    ratio = max(
        float(np.linalg.norm(numerical_step))
        / max(float(np.linalg.norm(continuum_step)), np.finfo(float).tiny),
        float(np.linalg.norm(continuum_step))
        / max(float(np.linalg.norm(numerical_step)), np.finfo(float).tiny),
    )
    return error, ratio


def _local_attribution(
    tangent,
    grid,
    background,
    field_scales: np.ndarray,
    target_radius_over_rg: float,
    log_spacing: float,
) -> tuple[dict, dict[str, np.ndarray]]:
    rg = float(grid.gravitational_radius)
    row = int(
        np.argmin(
            np.abs(grid.centers / rg - target_radius_over_rg)
        )
    )
    components = causal_local_dae_component_stencil(
        tangent,
        row,
        field_scales,
    )
    exact = causal_five_field_local_symbol_stencil(
        tangent,
        row,
        field_scales,
    )
    player_names = None
    theta_count = len(AUDIT_THETA_VALUES)
    time_count = len(AUDIT_TIMES_S)
    arrays: dict[str, np.ndarray] = {
        "theta_values": np.asarray(AUDIT_THETA_VALUES, dtype=float),
        "times_s": np.asarray(AUDIT_TIMES_S, dtype=float),
    }
    total_errors = np.empty((theta_count, time_count))
    norm_ratios = np.empty_like(total_errors)
    closure = np.empty_like(total_errors)
    contributions = None
    contribution_total = None
    contribution_scale = None
    cosines = None
    grams = None
    only = None
    leave = None
    generator_parity = 0.0
    continuum_parity = 0.0
    for theta_index, theta in enumerate(AUDIT_THETA_VALUES):
        numerical = components.symbol(theta)
        continuum = causal_continuum_normalized_local_dae(
            background,
            components.radius,
            theta,
            log_spacing,
            field_scales,
        )
        exact_numerical = exact.generators(theta)[0]
        exact_continuum = causal_five_field_continuum_local_symbol(
            background,
            components.radius,
            theta,
            log_spacing,
            field_scales,
        ).complete_generator_per_s
        generator_parity = max(
            generator_parity,
            c6a._semigroup_relative_difference(
                numerical.generator_per_s,
                exact_numerical,
            ),
        )
        continuum_parity = max(
            continuum_parity,
            c6a._semigroup_relative_difference(
                continuum.generator_per_s,
                exact_continuum,
            ),
        )
        for time_index, interval in enumerate(AUDIT_TIMES_S):
            audit = causal_symbol_shapley_attribution(
                numerical,
                continuum,
                interval,
            )
            if player_names is None:
                player_names = audit.player_names
                shape = (
                    theta_count,
                    time_count,
                    len(player_names),
                )
                contributions = np.empty(
                    (*shape, 5, 5),
                    dtype=complex,
                )
                contribution_total = np.empty(shape)
                contribution_scale = np.empty(shape)
                cosines = np.empty(shape)
                grams = np.empty(
                    (
                        theta_count,
                        time_count,
                        len(player_names),
                        len(player_names),
                    )
                )
                only = np.empty(shape)
                leave = np.empty(shape)
            error, ratio = _relative_propagator_error(
                numerical.generator_per_s,
                continuum.generator_per_s,
                interval,
            )
            total_errors[theta_index, time_index] = error
            norm_ratios[theta_index, time_index] = ratio
            closure[theta_index, time_index] = (
                audit.maximum_closure_defect
            )
            contributions[theta_index, time_index] = audit.contributions
            contribution_total[theta_index, time_index] = (
                audit.contribution_norms_relative_to_total
            )
            contribution_scale[theta_index, time_index] = (
                audit.contribution_norms_relative_to_propagator
            )
            cosines[theta_index, time_index] = (
                audit.contribution_cosines_with_total
            )
            grams[theta_index, time_index] = (
                audit.gram_matrix_relative_to_total
            )
            only[theta_index, time_index] = (
                audit.only_player_errors_relative_to_propagator
            )
            leave[theta_index, time_index] = (
                audit.leave_one_out_errors_relative_to_propagator
            )
    arrays.update(
        {
            "total_semigroup_errors": total_errors,
            "propagator_norm_ratios": norm_ratios,
            "shapley_closure_defects": closure,
            "shapley_contributions": contributions,
            "contribution_norms_relative_to_total": contribution_total,
            "contribution_norms_relative_to_propagator": contribution_scale,
            "contribution_cosines_with_total": cosines,
            "contribution_gram_matrices": grams,
            "only_player_errors_relative_to_propagator": only,
            "leave_one_out_errors_relative_to_propagator": leave,
        }
    )
    return {
        "target_radius_over_rg": target_radius_over_rg,
        "radius_over_rg": components.radius / rg,
        "cell_index": row,
        "offsets": components.offsets,
        "player_names": player_names,
        "touches_boundary": components.touches_boundary,
        "maximum_component_closure_defect": (
            components.maximum_component_closure_defect
        ),
        "maximum_omitted_fraction": components.maximum_omitted_fraction,
        "maximum_generator_parity_defect": generator_parity,
        "maximum_continuum_generator_parity_defect": continuum_parity,
        "maximum_shapley_closure_defect": float(np.max(closure)),
        "maximum_propagator_norm_ratio": float(np.max(norm_ratios)),
    }, arrays


def _cross_grid_attribution(
    configurations: dict,
    tangents: dict,
    background,
    field_scales: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    reference_grid = configurations[REFERENCE_LABEL]["context"].grid
    reference_spacing = float(
        np.diff(np.log(reference_grid.edges))[0]
    )
    reports = {}
    arrays: dict[str, np.ndarray] = {}
    minimum_significant_order = float("inf")
    significant_count = 0
    maximum_closure = 0.0
    player_names = None
    for target in AUDIT_RADII_OVER_RG:
        target_reports = {}
        for reference_theta in CROSS_GRID_REFERENCE_THETAS:
            level_contributions = []
            level_total_errors = []
            for label in LABELS:
                configuration = configurations[label]
                grid = configuration["context"].grid
                spacing = float(np.diff(np.log(grid.edges))[0])
                rg = float(grid.gravitational_radius)
                row = int(
                    np.argmin(np.abs(grid.centers / rg - target))
                )
                components = causal_local_dae_component_stencil(
                    tangents[label],
                    row,
                    field_scales,
                )
                theta = reference_theta * spacing / reference_spacing
                numerical = components.symbol(theta)
                continuum = causal_continuum_normalized_local_dae(
                    background,
                    components.radius,
                    theta,
                    spacing,
                    field_scales,
                )
                audit = causal_symbol_shapley_attribution(
                    numerical,
                    continuum,
                    AUDIT_TIMES_S[-1],
                )
                player_names = audit.player_names
                level_contributions.append(
                    audit.contribution_norms_relative_to_propagator
                )
                total_error, _ratio = _relative_propagator_error(
                    numerical.generator_per_s,
                    continuum.generator_per_s,
                    AUDIT_TIMES_S[-1],
                )
                level_total_errors.append(total_error)
                maximum_closure = max(
                    maximum_closure,
                    audit.maximum_closure_defect,
                )
            contribution_array = np.asarray(
                level_contributions,
                dtype=float,
            )
            total_array = np.asarray(level_total_errors, dtype=float)
            contribution_orders = np.log2(
                np.maximum(
                    contribution_array[:-1],
                    np.finfo(float).tiny,
                )
                / np.maximum(
                    contribution_array[1:],
                    np.finfo(float).tiny,
                )
            )
            total_orders = np.log2(
                np.maximum(total_array[:-1], np.finfo(float).tiny)
                / np.maximum(total_array[1:], np.finfo(float).tiny)
            )
            significant = np.maximum(
                contribution_array[0],
                contribution_array[1],
            ) >= SIGNIFICANT_PROPAGATOR_CONTRIBUTION
            if np.any(significant):
                significant_count += int(np.count_nonzero(significant))
                minimum_significant_order = min(
                    minimum_significant_order,
                    float(
                        np.min(contribution_orders[:, significant])
                    ),
                )
            key = f"theta{reference_theta:.2f}"
            target_reports[key] = {
                "contribution_errors": contribution_array,
                "contribution_orders": contribution_orders,
                "significant_players": significant,
                "total_errors": total_array,
                "total_orders": total_orders,
            }
            prefix = f"r{target:.2f}__{key}"
            arrays[f"{prefix}__contribution_errors"] = contribution_array
            arrays[f"{prefix}__contribution_orders"] = contribution_orders
            arrays[f"{prefix}__significant_players"] = (
                significant.astype(np.int8)
            )
            arrays[f"{prefix}__total_errors"] = total_array
            arrays[f"{prefix}__total_orders"] = total_orders
        reports[f"r{target:.2f}"] = target_reports
    passed = bool(
        significant_count > 0
        and minimum_significant_order
        >= MINIMUM_SIGNIFICANT_COMPONENT_ORDER
        and maximum_closure <= MAXIMUM_SHAPLEY_CLOSURE_DEFECT
    )
    return {
        "player_names": player_names,
        "radius_reports": reports,
        "significant_player_instances": significant_count,
        "minimum_significant_component_order": (
            minimum_significant_order
        ),
        "maximum_shapley_closure_defect": maximum_closure,
        "passed": passed,
    }, arrays


def _matched_group_velocities(
    background,
    radius: float,
    theta: float,
    spacing: float,
    field_scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    step = RAY_GROUP_DERIVATIVE_STEP
    center = causal_continuum_normalized_local_dae(
        background,
        radius,
        theta,
        spacing,
        field_scales,
    )
    lower = causal_continuum_normalized_local_dae(
        background,
        radius,
        theta - step,
        spacing,
        field_scales,
    )
    upper = causal_continuum_normalized_local_dae(
        background,
        radius,
        theta + step,
        spacing,
        field_scales,
    )
    center_principal = -center.operator("principal")
    lower_principal = -lower.operator("principal")
    upper_principal = -upper.operator("principal")
    center_values, lower_values = (
        causal_five_field_matched_principal_eigenvalues(
            lower_principal,
            center_principal,
        )
    )
    check_values, upper_values = (
        causal_five_field_matched_principal_eigenvalues(
            upper_principal,
            center_principal,
        )
    )
    if np.max(np.abs(check_values - center_values)) > 1.0e-7 * max(
        float(np.max(np.abs(center_values))),
        1.0,
    ):
        raise RuntimeError("ray family matching changed center ordering")
    order = np.argsort(np.imag(center_values))
    # For q_t = lambda(k) q and exp(i k x), physical x-speed is
    # -d Im(lambda)/dk.
    velocities = (
        -spacing
        * (
            np.imag(upper_values[order])
            - np.imag(lower_values[order])
        )
        / (2.0 * step)
    )
    values, vectors = np.linalg.eig(center_principal)
    vector_order = []
    for target in center_values[order]:
        available = [
            index
            for index in range(values.size)
            if index not in vector_order
        ]
        selected = min(
            available,
            key=lambda index: abs(values[index] - target),
        )
        vector_order.append(selected)
    family_vectors = vectors[:, vector_order]
    family_vectors /= np.linalg.norm(
        family_vectors,
        axis=0,
        keepdims=True,
    )
    return np.asarray(velocities, dtype=float), family_vectors


def _interpolated_numerical_symbol(
    stencils: dict[int, object],
    grid,
    log_radius: float,
    theta: float,
) -> CausalNormalizedLocalDAESymbol:
    centers = np.log(np.asarray(grid.centers, dtype=float))
    upper = int(np.searchsorted(centers, log_radius, side="right"))
    lower = upper - 1
    available = sorted(stencils)
    lower = max(available[0], min(lower, available[-1] - 1))
    upper = lower + 1
    if upper not in stencils:
        upper = available[-1]
        lower = upper - 1
    weight = (
        (float(log_radius) - centers[lower])
        / (centers[upper] - centers[lower])
    )
    weight = min(max(float(weight), 0.0), 1.0)
    left = stencils[lower].symbol(theta)
    right = stencils[upper].symbol(theta)
    return CausalNormalizedLocalDAESymbol(
        descriptor=(
            (1.0 - weight) * left.descriptor
            + weight * right.descriptor
        ),
        operators={
            name: (
                (1.0 - weight) * left.operators[name]
                + weight * right.operators[name]
            )
            for name in left.operators
        },
    )


def _interpolate_table(
    log_radii: np.ndarray,
    values: np.ndarray,
    log_radius: float,
) -> np.ndarray:
    upper = int(np.searchsorted(log_radii, log_radius, side="right"))
    upper = max(1, min(upper, log_radii.size - 1))
    lower = upper - 1
    weight = (
        (float(log_radius) - float(log_radii[lower]))
        / float(log_radii[upper] - log_radii[lower])
    )
    weight = min(max(float(weight), 0.0), 1.0)
    return (
        (1.0 - weight) * values[lower]
        + weight * values[upper]
    )


def _build_ray_coefficient_table(
    stencils: dict[int, object],
    grid,
    primary_background,
    secondary_background,
    field_scales: np.ndarray,
    theta: float,
) -> dict[str, np.ndarray | float]:
    spacing = float(np.diff(np.log(grid.edges))[0])
    cell_logs = np.log(np.asarray(grid.centers, dtype=float))
    lower = float(cell_logs[min(stencils)])
    upper = float(cell_logs[max(stencils)])
    base_logs = np.linspace(
        lower,
        upper,
        RAY_BRANCH_TABLE_BASE_NODES,
    )
    log_radii = np.unique(
        np.concatenate(
            (
                base_logs,
                cell_logs[min(stencils) : max(stencils) + 1],
            )
        )
    )
    count = int(log_radii.size)
    numerical = np.empty((count, 5, 5), dtype=complex)
    continuum = np.empty_like(numerical)
    secondary = np.empty_like(numerical)
    center_principal = np.empty_like(numerical)
    lower_principal = np.empty_like(numerical)
    upper_principal = np.empty_like(numerical)
    for index, log_radius in enumerate(log_radii):
        radius = float(np.exp(log_radius))
        numerical[index] = _interpolated_numerical_symbol(
            stencils,
            grid,
            float(log_radius),
            theta,
        ).generator_per_s
        center = causal_continuum_normalized_local_dae(
            primary_background,
            radius,
            theta,
            spacing,
            field_scales,
        )
        lower_symbol = causal_continuum_normalized_local_dae(
            primary_background,
            radius,
            theta - RAY_GROUP_DERIVATIVE_STEP,
            spacing,
            field_scales,
        )
        upper_symbol = causal_continuum_normalized_local_dae(
            primary_background,
            radius,
            theta + RAY_GROUP_DERIVATIVE_STEP,
            spacing,
            field_scales,
        )
        continuum[index] = center.generator_per_s
        secondary[index] = causal_continuum_normalized_local_dae(
            secondary_background,
            radius,
            theta,
            spacing,
            field_scales,
        ).generator_per_s
        center_principal[index] = -center.operator("principal")
        lower_principal[index] = -lower_symbol.operator("principal")
        upper_principal[index] = -upper_symbol.operator("principal")
    tracked = causal_track_symbol_eigenbranches(center_principal)
    lower_values = causal_match_symbol_eigenvalues_to_tracked_branches(
        lower_principal,
        tracked,
    )
    upper_values = causal_match_symbol_eigenvalues_to_tracked_branches(
        upper_principal,
        tracked,
    )
    velocities = (
        -spacing
        * (np.imag(upper_values) - np.imag(lower_values))
        / (2.0 * RAY_GROUP_DERIVATIVE_STEP)
    )
    return {
        "log_radii": log_radii,
        "numerical_generators": numerical,
        "continuum_generators": continuum,
        "secondary_generators": secondary,
        "tracked_eigenvalues": tracked.eigenvalues,
        "tracked_eigenvectors": tracked.eigenvectors,
        "branch_overlaps": tracked.consecutive_overlaps,
        "velocities": np.asarray(velocities, dtype=float),
        "minimum_branch_overlap": tracked.minimum_consecutive_overlap,
    }


def _one_ray(
    table: dict[str, np.ndarray | float],
    start_radius: float,
    family: int,
    step_size: float,
) -> dict:
    step_count = int(round(RAY_HORIZON_S / step_size))
    if not np.isclose(step_count * step_size, RAY_HORIZON_S):
        raise RuntimeError("ray step does not divide the horizon")
    x = float(np.log(start_radius))
    log_radii = np.asarray(table["log_radii"], dtype=float)
    velocities = np.asarray(table["velocities"], dtype=float)
    vectors = np.asarray(table["tracked_eigenvectors"], dtype=complex)
    start_index = int(np.argmin(np.abs(log_radii - x)))
    initial = np.asarray(
        vectors[start_index, :, family],
        dtype=complex,
    )
    numerical_state = np.array(initial, copy=True)
    continuum_state = np.array(initial, copy=True)
    secondary_state = np.array(initial, copy=True)
    maximum_error = 0.0
    maximum_reference_error = 0.0
    minimum_radius = start_radius
    maximum_radius = start_radius

    def derivative(
        log_radius: float,
        numerical_values: np.ndarray,
        continuum_values: np.ndarray,
        secondary_values: np.ndarray,
    ) -> tuple[
        float,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        local_velocities = _interpolate_table(
            log_radii,
            velocities,
            log_radius,
        )
        numerical_generator = _interpolate_table(
            log_radii,
            np.asarray(table["numerical_generators"]),
            log_radius,
        )
        continuum_generator = _interpolate_table(
            log_radii,
            np.asarray(table["continuum_generators"]),
            log_radius,
        )
        secondary_generator = _interpolate_table(
            log_radii,
            np.asarray(table["secondary_generators"]),
            log_radius,
        )
        return (
            float(local_velocities[family]),
            numerical_generator @ numerical_values,
            continuum_generator @ continuum_values,
            secondary_generator @ secondary_values,
        )

    for _index in range(step_count):
        first = derivative(
            x,
            numerical_state,
            continuum_state,
            secondary_state,
        )
        second = derivative(
            x + 0.5 * step_size * first[0],
            numerical_state + 0.5 * step_size * first[1],
            continuum_state + 0.5 * step_size * first[2],
            secondary_state + 0.5 * step_size * first[3],
        )
        third = derivative(
            x + 0.5 * step_size * second[0],
            numerical_state + 0.5 * step_size * second[1],
            continuum_state + 0.5 * step_size * second[2],
            secondary_state + 0.5 * step_size * second[3],
        )
        fourth = derivative(
            x + step_size * third[0],
            numerical_state + step_size * third[1],
            continuum_state + step_size * third[2],
            secondary_state + step_size * third[3],
        )
        x += step_size / 6.0 * (
            first[0] + 2.0 * second[0] + 2.0 * third[0] + fourth[0]
        )
        numerical_state += step_size / 6.0 * (
            first[1] + 2.0 * second[1] + 2.0 * third[1] + fourth[1]
        )
        continuum_state += step_size / 6.0 * (
            first[2] + 2.0 * second[2] + 2.0 * third[2] + fourth[2]
        )
        secondary_state += step_size / 6.0 * (
            first[3] + 2.0 * second[3] + 2.0 * third[3] + fourth[3]
        )
        scale = max(
            float(np.linalg.norm(numerical_state)),
            float(np.linalg.norm(continuum_state)),
            np.finfo(float).tiny,
        )
        maximum_error = max(
            maximum_error,
            float(
                np.linalg.norm(numerical_state - continuum_state)
                / scale
            ),
        )
        reference_scale = max(
            float(np.linalg.norm(continuum_state)),
            float(np.linalg.norm(secondary_state)),
            np.finfo(float).tiny,
        )
        maximum_reference_error = max(
            maximum_reference_error,
            float(
                np.linalg.norm(continuum_state - secondary_state)
                / reference_scale
            ),
        )
        radius = float(np.exp(x))
        minimum_radius = min(minimum_radius, radius)
        maximum_radius = max(maximum_radius, radius)
        if (
            x <= float(log_radii[0])
            or x >= float(log_radii[-1])
        ):
            raise RuntimeError("ray left the certified interior stencil band")
    final_scale = max(
        float(np.linalg.norm(numerical_state)),
        float(np.linalg.norm(continuum_state)),
        np.finfo(float).tiny,
    )
    return {
        "initial_group_velocity_log_radius_per_s": float(
            _interpolate_table(
                log_radii,
                velocities,
                float(np.log(start_radius)),
            )[family]
        ),
        "final_radius": float(np.exp(x)),
        "minimum_radius": minimum_radius,
        "maximum_radius": maximum_radius,
        "final_error": float(
            np.linalg.norm(numerical_state - continuum_state)
            / final_scale
        ),
        "maximum_error": maximum_error,
        "maximum_continuum_reference_error": maximum_reference_error,
        "final_numerical_state": numerical_state,
        "final_continuum_state": continuum_state,
    }


def _ray_report(
    configurations: dict,
    tangent,
    primary_background,
    secondary_background,
    field_scales: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    grid = configurations[REFERENCE_LABEL]["context"].grid
    rg = float(grid.gravitational_radius)
    # Two halo cells are excluded on each side.  Every interpolation stencil
    # is therefore strictly interior and uses the same exact component map.
    stencils = {
        row: causal_local_dae_component_stencil(
            tangent,
            row,
            field_scales,
        )
        for row in range(2, grid.centers.size - 2)
    }
    reports = {}
    arrays: dict[str, np.ndarray] = {}
    maximum_theta20_error = 0.0
    maximum_integrator_ratio = 0.0
    maximum_reference_ratio = 0.0
    minimum_branch_overlap = 1.0
    for theta in RAY_THETA_VALUES:
        print(
            f"WP10c9d6c6a1: build tracked ray table theta={theta:.2f}",
            flush=True,
        )
        table = _build_ray_coefficient_table(
            stencils,
            grid,
            primary_background,
            secondary_background,
            field_scales,
            theta,
        )
        minimum_branch_overlap = min(
            minimum_branch_overlap,
            float(table["minimum_branch_overlap"]),
        )
        table_key = f"theta{theta:.2f}"
        arrays[f"{table_key}__log_radii"] = np.asarray(
            table["log_radii"]
        )
        arrays[f"{table_key}__tracked_eigenvalues"] = np.asarray(
            table["tracked_eigenvalues"]
        )
        arrays[f"{table_key}__branch_overlaps"] = np.asarray(
            table["branch_overlaps"]
        )
        arrays[f"{table_key}__group_velocities"] = np.asarray(
            table["velocities"]
        )
        for target in RAY_START_RADII_OVER_RG:
            start_radius = target * rg
            family_reports = []
            primary_states = []
            secondary_states = []
            for family in range(5):
                fine = _one_ray(
                    table,
                    start_radius,
                    family,
                    RAY_PRIMARY_STEP_S,
                )
                coarse = _one_ray(
                    table,
                    start_radius,
                    family,
                    RAY_SECONDARY_STEP_S,
                )
                error_scale = max(
                    fine["maximum_error"],
                    np.finfo(float).tiny,
                )
                integrator_difference = float(
                    np.linalg.norm(
                        fine["final_numerical_state"]
                        - coarse["final_numerical_state"]
                    )
                    / max(
                        float(
                            np.linalg.norm(fine["final_numerical_state"])
                        ),
                        float(
                            np.linalg.norm(coarse["final_numerical_state"])
                        ),
                        np.finfo(float).tiny,
                    )
                )
                integrator_ratio = integrator_difference / error_scale
                reference_ratio = (
                    fine["maximum_continuum_reference_error"]
                    / error_scale
                )
                maximum_integrator_ratio = max(
                    maximum_integrator_ratio,
                    integrator_ratio,
                )
                maximum_reference_ratio = max(
                    maximum_reference_ratio,
                    reference_ratio,
                )
                if np.isclose(theta, MINIMUM_USABLE_THETA):
                    maximum_theta20_error = max(
                        maximum_theta20_error,
                        fine["maximum_error"],
                    )
                family_reports.append(
                    {
                        "family_index": family,
                        "initial_group_velocity_log_radius_per_s": (
                            fine[
                                "initial_group_velocity_log_radius_per_s"
                            ]
                        ),
                        "final_radius_over_rg": fine["final_radius"] / rg,
                        "minimum_radius_over_rg": (
                            fine["minimum_radius"] / rg
                        ),
                        "maximum_radius_over_rg": (
                            fine["maximum_radius"] / rg
                        ),
                        "final_error": fine["final_error"],
                        "maximum_error": fine["maximum_error"],
                        "integrator_difference": integrator_difference,
                        "integrator_to_error_ratio": integrator_ratio,
                        "maximum_continuum_reference_error": (
                            fine["maximum_continuum_reference_error"]
                        ),
                        "continuum_reference_to_error_ratio": (
                            reference_ratio
                        ),
                    }
                )
                primary_states.append(fine["final_numerical_state"])
                secondary_states.append(coarse["final_numerical_state"])
            key = f"r{target:.2f}__theta{theta:.2f}"
            reports[key] = {
                "start_radius_over_rg": target,
                "theta": theta,
                "families": family_reports,
                "maximum_error": max(
                    row["maximum_error"] for row in family_reports
                ),
                "maximum_integrator_to_error_ratio": max(
                    row["integrator_to_error_ratio"]
                    for row in family_reports
                ),
                "maximum_continuum_reference_to_error_ratio": max(
                    row["continuum_reference_to_error_ratio"]
                    for row in family_reports
                ),
                "minimum_branch_overlap": float(
                    table["minimum_branch_overlap"]
                ),
            }
            arrays[f"{key}__primary_final_states"] = np.asarray(
                primary_states
            )
            arrays[f"{key}__secondary_final_states"] = np.asarray(
                secondary_states
            )
    method_passed = bool(
        maximum_integrator_ratio
        <= MAXIMUM_RAY_INTEGRATOR_TO_ERROR_RATIO
        and maximum_reference_ratio
        <= MAXIMUM_RAY_REFERENCE_TO_ERROR_RATIO
        and minimum_branch_overlap >= MINIMUM_RAY_BRANCH_OVERLAP
    )
    return {
        "reports": reports,
        "maximum_theta20_error": maximum_theta20_error,
        "theta20_passes_original_packet_budget": (
            maximum_theta20_error <= PACKET_CONTRACT_ERROR
        ),
        "theta20_passes_preflight_ceiling": (
            maximum_theta20_error <= MAXIMUM_RAY_PREFLIGHT_ERROR
        ),
        "maximum_integrator_to_error_ratio": maximum_integrator_ratio,
        "maximum_continuum_reference_to_error_ratio": (
            maximum_reference_ratio
        ),
        "minimum_branch_overlap": minimum_branch_overlap,
        "method_passed": method_passed,
    }, arrays


def _config() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "operator_change": False,
        "production_defaults_changed": False,
        "parent_contract_preserved": True,
        "labels": LABELS,
        "audit_radii_over_rg": AUDIT_RADII_OVER_RG,
        "audit_theta_values": AUDIT_THETA_VALUES,
        "audit_times_s": AUDIT_TIMES_S,
        "cross_grid_reference_thetas": CROSS_GRID_REFERENCE_THETAS,
        "limiter_radius_over_rg": LIMITER_RADIUS_OVER_RG,
        "limiter_theta": LIMITER_THETA,
        "ray_start_radii_over_rg": RAY_START_RADII_OVER_RG,
        "ray_theta_values": RAY_THETA_VALUES,
        "ray_horizon_s": RAY_HORIZON_S,
        "ray_integrator": (
            "overlap_tracked_coupled_classical_rk4"
        ),
        "ray_branch_table_base_nodes": RAY_BRANCH_TABLE_BASE_NODES,
        "ray_primary_step_s": RAY_PRIMARY_STEP_S,
        "ray_secondary_step_s": RAY_SECONDARY_STEP_S,
        "gates": {
            "maximum_component_closure_defect": (
                MAXIMUM_COMPONENT_CLOSURE_DEFECT
            ),
            "maximum_generator_parity_defect": (
                MAXIMUM_GENERATOR_PARITY_DEFECT
            ),
            "maximum_shapley_closure_defect": (
                MAXIMUM_SHAPLEY_CLOSURE_DEFECT
            ),
            "significant_propagator_contribution": (
                SIGNIFICANT_PROPAGATOR_CONTRIBUTION
            ),
            "minimum_significant_component_order": (
                MINIMUM_SIGNIFICANT_COMPONENT_ORDER
            ),
            "minimum_time_accumulation_exponent": (
                MINIMUM_TIME_ACCUMULATION_EXPONENT
            ),
            "maximum_time_accumulation_exponent": (
                MAXIMUM_TIME_ACCUMULATION_EXPONENT
            ),
            "maximum_propagator_norm_ratio": (
                MAXIMUM_NUMERICAL_CONTINUUM_PROPAGATOR_NORM_RATIO
            ),
            "maximum_ray_integrator_to_error_ratio": (
                MAXIMUM_RAY_INTEGRATOR_TO_ERROR_RATIO
            ),
            "minimum_ray_branch_overlap": MINIMUM_RAY_BRANCH_OVERLAP,
            "maximum_ray_reference_to_error_ratio": (
                MAXIMUM_RAY_REFERENCE_TO_ERROR_RATIO
            ),
            "maximum_ray_preflight_error": MAXIMUM_RAY_PREFLIGHT_ERROR,
            "unchanged_packet_contract_error": PACKET_CONTRACT_ERROR,
            "unchanged_minimum_usable_theta": MINIMUM_USABLE_THETA,
        },
    }


def run() -> dict:
    started = time.perf_counter()
    identity = _validate_analyzed_git_identity()
    parent_summary, _parent_arrays = _load_parent()
    (
        configurations,
        construction_arrays,
        construction,
        background_profile,
        field_scales,
    ) = c6a._build_construction()
    if not construction["passed"]:
        raise RuntimeError("c6a1 continuum construction failed")
    tangents = c6a._build_tangents(configurations)
    context = configurations[REFERENCE_LABEL]["context"]
    print("WP10c9d6c6a1: build continuum references", flush=True)
    primary_background = build_causal_five_field_continuum_background(
        context,
        background_profile.evaluate,
        node_count=c6a.PRIMARY_CONTINUUM_NODES,
    )
    secondary_background = build_causal_five_field_continuum_background(
        context,
        background_profile.evaluate,
        node_count=c6a.SECONDARY_CONTINUUM_NODES,
    )
    grid = context.grid
    spacing = float(np.diff(np.log(grid.edges))[0])
    local_reports = {}
    decisive: dict[str, np.ndarray] = {
        "continuum_background_knots": np.asarray(
            construction_arrays["continuum_background_knots"],
            dtype=float,
        ),
        "continuum_background_coefficients": np.asarray(
            construction_arrays["continuum_background_coefficients"],
            dtype=float,
        ),
        "field_scales": np.asarray(field_scales, dtype=float),
    }
    maximum_component_closure = 0.0
    maximum_generator_parity = 0.0
    maximum_shapley_closure = 0.0
    maximum_norm_ratio = 0.0
    for target in AUDIT_RADII_OVER_RG:
        print(
            f"WP10c9d6c6a1: local attribution r={target:.2f} rg",
            flush=True,
        )
        report, arrays = _local_attribution(
            tangents[REFERENCE_LABEL],
            grid,
            primary_background,
            field_scales,
            target,
            spacing,
        )
        key = f"r{target:.2f}"
        local_reports[key] = report
        for name, values in arrays.items():
            decisive[f"{key}__{name}"] = np.asarray(values)
        maximum_component_closure = max(
            maximum_component_closure,
            report["maximum_component_closure_defect"],
        )
        maximum_generator_parity = max(
            maximum_generator_parity,
            report["maximum_generator_parity_defect"],
            report["maximum_continuum_generator_parity_defect"],
        )
        maximum_shapley_closure = max(
            maximum_shapley_closure,
            report["maximum_shapley_closure_defect"],
        )
        maximum_norm_ratio = max(
            maximum_norm_ratio,
            report["maximum_propagator_norm_ratio"],
        )

    cross_grid, cross_arrays = _cross_grid_attribution(
        configurations,
        tangents,
        primary_background,
        field_scales,
    )
    decisive.update(
        {
            f"cross_grid__{name}": np.asarray(values)
            for name, values in cross_arrays.items()
        }
    )
    limiter_key = f"r{LIMITER_RADIUS_OVER_RG:.2f}"
    limiter_errors = decisive[
        f"{limiter_key}__total_semigroup_errors"
    ]
    theta_index = AUDIT_THETA_VALUES.index(LIMITER_THETA)
    time_exponents = np.log2(
        np.maximum(
            limiter_errors[theta_index, 1:],
            np.finfo(float).tiny,
        )
        / np.maximum(
            limiter_errors[theta_index, :-1],
            np.finfo(float).tiny,
        )
    )
    decisive["limiter_time_accumulation_exponents"] = time_exponents
    time_accumulation_passed = bool(
        np.min(time_exponents)
        >= MINIMUM_TIME_ACCUMULATION_EXPONENT
        and np.max(time_exponents)
        <= MAXIMUM_TIME_ACCUMULATION_EXPONENT
    )
    method_passed = bool(
        maximum_component_closure <= MAXIMUM_COMPONENT_CLOSURE_DEFECT
        and maximum_generator_parity <= MAXIMUM_GENERATOR_PARITY_DEFECT
        and maximum_shapley_closure <= MAXIMUM_SHAPLEY_CLOSURE_DEFECT
        and maximum_norm_ratio
        <= MAXIMUM_NUMERICAL_CONTINUUM_PROPAGATOR_NORM_RATIO
        and cross_grid["passed"]
        and time_accumulation_passed
    )

    print("WP10c9d6c6a1: variable-radius ray preflight", flush=True)
    ray_report, ray_arrays = _ray_report(
        configurations,
        tangents[REFERENCE_LABEL],
        primary_background,
        secondary_background,
        field_scales,
    )
    decisive.update(
        {
            f"ray__{name}": np.asarray(values)
            for name, values in ray_arrays.items()
        }
    )
    if not method_passed:
        classification = "full_symbol_limiter_method_or_order_failed"
        authorized_next = "none"
    elif not ray_report["method_passed"]:
        classification = "full_symbol_ray_preflight_unresolved"
        authorized_next = "none"
    elif ray_report["theta20_passes_preflight_ceiling"]:
        classification = (
            "full_symbol_limiter_convergent_accumulation_"
            "windowed_contract_audit_authorized"
        )
        authorized_next = (
            "WP10c9d6c6a2_variable_coefficient_windowed_contract"
        )
    else:
        classification = (
            "full_symbol_limiter_convergent_but_no_usable_ray_range"
        )
        authorized_next = "none"
    passed = bool(authorized_next != "none")

    source_hashes, source_manifest = _source_manifest()
    config = _config()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "authorized_next": authorized_next,
        **identity,
        "parent_classification": parent_summary["classification"],
        "parent_classification_preserved": True,
        "parent_failed_certified_theta": parent_summary[
            "symbol_contract"
        ]["certified_theta"],
        "parent_packet_contract_error_preserved": True,
        "parent_minimum_usable_theta_preserved": True,
        "operator_changed": False,
        "production_defaults_changed": False,
        "method_report": {
            "maximum_component_closure_defect": (
                maximum_component_closure
            ),
            "maximum_generator_parity_defect": maximum_generator_parity,
            "maximum_shapley_closure_defect": maximum_shapley_closure,
            "maximum_propagator_norm_ratio": maximum_norm_ratio,
            "limiter_time_accumulation_exponents": time_exponents,
            "time_accumulation_passed": time_accumulation_passed,
            "passed": method_passed,
        },
        "local_attribution_reports": local_reports,
        "cross_grid_attribution_report": cross_grid,
        "ray_preflight_report": ray_report,
        "configuration": config,
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "packet_resolution_contract_certified": False,
        "prospective_packet_manifest_authorized": False,
        "uniform_packet_propagation_authorized": False,
        "embedded_export_discrimination_authorized": False,
        "nonlinear_physical_trajectory_authorized": False,
        "production_operator_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)
    summary["decisive_arrays_sha256"] = _sha256(DECISIVE_ARRAYS)
    summary["decisive_array_hashes"] = {
        name: _array_sha256(values)
        for name, values in sorted(decisive.items())
    }
    _write_json(SUMMARY_PATH, summary)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "scientific_status": (
            "DIAGNOSTIC ONLY" if passed else "REJECTED"
        ),
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        **identity,
        "implementation_base_tree": _git_value("rev-parse", "HEAD^{tree}"),
        "working_tree_status": _git_value("status", "--short"),
        "command": (
            "PYTHONPATH=src:scripts python "
            "scripts/run_causal_inner_full_symbol_limiter_"
            "wp10c9d6c6a1.py"
        ),
        "environment": _environment(),
        "parent_canonical_hashes": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (
                PARENT_CONFIG,
                PARENT_SUMMARY,
                PARENT_ARRAYS,
                PARENT_PROVENANCE,
            )
        },
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
    }
    _write_json(CONFIG_PATH, config)
    _write_json(PROVENANCE_PATH, provenance)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    _refresh_canonical_catalog()
    print(
        json.dumps(
            _plain(
                {
                    "classification": classification,
                    "authorized_next": authorized_next,
                    "minimum_significant_component_order": cross_grid[
                        "minimum_significant_component_order"
                    ],
                    "limiter_time_exponents": time_exponents,
                    "maximum_theta20_ray_error": ray_report[
                        "maximum_theta20_error"
                    ],
                }
            ),
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return summary


def rerun_ray_with_rk4() -> dict:
    """Correct only the unresolved ray association/integration evidence."""

    started = time.perf_counter()
    _validate_analyzed_git_identity()
    if not SUMMARY_PATH.exists() or not DECISIVE_ARRAYS.exists():
        raise RuntimeError("the initial c6a1 midpoint preflight is absent")
    current_summary = json.loads(
        SUMMARY_PATH.read_text(encoding="utf-8")
    )
    if MIDPOINT_SUMMARY_PATH.exists():
        midpoint_summary = json.loads(
            MIDPOINT_SUMMARY_PATH.read_text(encoding="utf-8")
        )
    else:
        if (
            current_summary["classification"]
            != "full_symbol_ray_preflight_unresolved"
        ):
            raise RuntimeError("c6a1 midpoint classification changed")
        midpoint_summary = current_summary
        _write_json(MIDPOINT_SUMMARY_PATH, midpoint_summary)
    with np.load(DECISIVE_ARRAYS, allow_pickle=False) as source:
        arrays = {
            name: np.array(source[name], copy=True)
            for name in source.files
        }
    prior_correction = current_summary.get("ray_method_correction", {})
    prior_is_speed_sorted_rk4 = (
        prior_correction.get("corrected_integrator")
        == "coupled_classical_rk4"
    )
    historical_prefix = (
        "speed_sorted_rk4"
        if prior_is_speed_sorted_rk4
        else "midpoint"
    )
    for name in tuple(arrays):
        if (
            name.startswith("ray__")
            and not name.startswith("ray__midpoint__")
            and not name.startswith("ray__speed_sorted_rk4__")
        ):
            historical_name = (
                f"ray__{historical_prefix}__"
                + name.removeprefix("ray__")
            )
            if historical_name not in arrays:
                arrays[historical_name] = arrays[name]
            del arrays[name]

    (
        configurations,
        _construction_arrays,
        construction,
        background_profile,
        field_scales,
    ) = c6a._build_construction()
    if not construction["passed"]:
        raise RuntimeError("c6a1 RK4 continuum construction failed")
    configuration = configurations[REFERENCE_LABEL]
    print(
        "WP10c9d6c6a1: rebuild exact N128 tangent for tracked RK4",
        flush=True,
    )
    tangent = causal_five_field_monolithic_frozen_tangent(
        configuration["context"],
        configuration["base_primitives"],
        primitive_column_scales=configuration[
            "primitive_column_scales"
        ],
        conservation_row_scales=configuration[
            "conservation_row_scales"
        ],
        path_quadrature_order=c6a.wp10c9d6c.PATH_QUADRATURE_ORDER,
    )
    print(
        "WP10c9d6c6a1: rebuild continuum references for tracked RK4",
        flush=True,
    )
    primary_background = build_causal_five_field_continuum_background(
        configuration["context"],
        background_profile.evaluate,
        node_count=c6a.PRIMARY_CONTINUUM_NODES,
    )
    secondary_background = build_causal_five_field_continuum_background(
        configuration["context"],
        background_profile.evaluate,
        node_count=c6a.SECONDARY_CONTINUUM_NODES,
    )
    print(
        "WP10c9d6c6a1: overlap-tracked coupled RK4 ray correction",
        flush=True,
    )
    ray_report, ray_arrays = _ray_report(
        configurations,
        tangent,
        primary_background,
        secondary_background,
        field_scales,
    )
    arrays.update(
        {
            f"ray__{name}": np.asarray(values)
            for name, values in ray_arrays.items()
        }
    )
    method_passed = bool(current_summary["method_report"]["passed"])
    if not method_passed:
        classification = "full_symbol_limiter_method_or_order_failed"
        authorized_next = "none"
    elif not ray_report["method_passed"]:
        classification = "full_symbol_ray_preflight_unresolved"
        authorized_next = "none"
    elif ray_report["theta20_passes_preflight_ceiling"]:
        classification = (
            "full_symbol_limiter_convergent_accumulation_"
            "windowed_contract_audit_authorized"
        )
        authorized_next = (
            "WP10c9d6c6a2_variable_coefficient_windowed_contract"
        )
    else:
        classification = (
            "full_symbol_limiter_convergent_but_no_usable_ray_range"
        )
        authorized_next = "none"
    passed = bool(authorized_next != "none")

    source_hashes, source_manifest = _source_manifest()
    summary = dict(current_summary)
    summary.update(
        {
            "classification": classification,
            "passed": passed,
            "authorized_next": authorized_next,
            "configuration": _config(),
            "ray_preflight_report": ray_report,
            "initial_midpoint_ray_preflight_report": midpoint_summary[
                "ray_preflight_report"
            ],
            "speed_sorted_rk4_ray_preflight_report": (
                current_summary["ray_preflight_report"]
                if prior_is_speed_sorted_rk4
                else current_summary.get(
                    "speed_sorted_rk4_ray_preflight_report"
                )
            ),
            "ray_method_correction": {
                "initial_classification": midpoint_summary[
                    "classification"
                ],
                "initial_integrator": (
                    "midpoint_exponential_with_midpoint_ray_step"
                ),
                "speed_sorted_rk4_classification": (
                    current_summary["classification"]
                    if prior_is_speed_sorted_rk4
                    else None
                ),
                "corrected_integrator": (
                    "overlap_tracked_coupled_classical_rk4"
                ),
                "branch_association_changed": True,
                "physical_rays_changed": False,
                "step_sizes_changed": False,
                "scientific_gates_changed": False,
                "initial_summary_path": str(
                    MIDPOINT_SUMMARY_PATH.relative_to(ROOT)
                ),
                "initial_summary_sha256": _sha256(
                    MIDPOINT_SUMMARY_PATH
                ),
            },
            "implementation_source_hashes": source_hashes,
            "implementation_source_manifest_sha256": source_manifest,
            "runtime_seconds": (
                float(current_summary["runtime_seconds"])
                + float(time.perf_counter() - started)
            ),
        }
    )
    np.savez_compressed(DECISIVE_ARRAYS, **arrays)
    summary["decisive_arrays_sha256"] = _sha256(DECISIVE_ARRAYS)
    summary["decisive_array_hashes"] = {
        name: _array_sha256(values)
        for name, values in sorted(arrays.items())
    }
    provenance = json.loads(
        PROVENANCE_PATH.read_text(encoding="utf-8")
    )
    provenance.update(
        {
            "scientific_status": (
                "DIAGNOSTIC ONLY" if passed else "REJECTED"
            ),
            "working_tree_status": _git_value("status", "--short"),
            "environment": _environment(),
            "implementation_source_hashes": source_hashes,
            "implementation_source_manifest_sha256": source_manifest,
            "ray_method_correction_command": (
                "PYTHONPATH=src:scripts python "
                "scripts/run_causal_inner_full_symbol_limiter_"
                "wp10c9d6c6a1.py --rerun-ray-rk4"
            ),
        }
    )
    _write_json(CONFIG_PATH, _config())
    _write_json(SUMMARY_PATH, summary)
    _write_json(PROVENANCE_PATH, provenance)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    _refresh_canonical_catalog()
    print(
        json.dumps(
            _plain(
                {
                    "classification": classification,
                    "authorized_next": authorized_next,
                    "maximum_theta20_ray_error": ray_report[
                        "maximum_theta20_error"
                    ],
                    "maximum_integrator_to_error_ratio": ray_report[
                        "maximum_integrator_to_error_ratio"
                    ],
                }
            ),
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return summary


def refresh_metadata_only() -> dict:
    if not SUMMARY_PATH.exists() or not DECISIVE_ARRAYS.exists():
        raise RuntimeError("c6a1 canonical evidence is unavailable")
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    with np.load(DECISIVE_ARRAYS, allow_pickle=False) as source:
        arrays = {
            name: np.array(source[name], copy=True)
            for name in source.files
        }
    source_hashes, source_manifest = _source_manifest()
    summary["configuration"] = _config()
    summary["implementation_source_hashes"] = source_hashes
    summary["implementation_source_manifest_sha256"] = source_manifest
    summary["decisive_arrays_sha256"] = _sha256(DECISIVE_ARRAYS)
    summary["decisive_array_hashes"] = {
        name: _array_sha256(values)
        for name, values in sorted(arrays.items())
    }
    provenance = (
        json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
        if PROVENANCE_PATH.exists()
        else {}
    )
    provenance.update(
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": (
                "DIAGNOSTIC ONLY" if summary["passed"] else "REJECTED"
            ),
            "source_parent_commit": ANALYZED_BASE_COMMIT,
            **_validate_analyzed_git_identity(),
            "implementation_base_tree": _git_value(
                "rev-parse",
                "HEAD^{tree}",
            ),
            "working_tree_status": _git_value("status", "--short"),
            "environment": _environment(),
            "implementation_source_hashes": source_hashes,
            "implementation_source_manifest_sha256": source_manifest,
        }
    )
    _write_json(CONFIG_PATH, _config())
    _write_json(PROVENANCE_PATH, provenance)
    _write_json(SUMMARY_PATH, summary)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    _refresh_canonical_catalog()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-metadata-only",
        action="store_true",
    )
    parser.add_argument(
        "--rerun-ray-rk4",
        action="store_true",
    )
    arguments = parser.parse_args()
    if arguments.refresh_metadata_only and arguments.rerun_ray_rk4:
        raise ValueError("select only one metadata/ray action")
    if arguments.refresh_metadata_only:
        refresh_metadata_only()
    elif arguments.rerun_ray_rk4:
        rerun_ray_with_rk4()
    else:
        run()


if __name__ == "__main__":
    main()
