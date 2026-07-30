#!/usr/bin/env python3
"""Derive a prospective packet-resolution contract for the monolithic DAE.

WP10c9d6c5 showed that two narrow packets fail the strict refinement-error
direction gate while doubled-width controls pass, without selecting an
operator defect.  This package changes no operator.  It builds the exact
local generalized symbol of the self-consistent monolithic tangent, compares
it with an independent smooth continuum symbol, and freezes a usable
wavenumber interval before any new packet propagation.

Interior packets are eligible through the spectral contract.  A packet that
overlaps the one-sided excision boundary additionally needs the certified
one-sided DAE-truncation contract inherited from WP10c9d6c5.
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
from scipy.special import erf


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_causal_inner_continuum_lift_wp10c9d6c3 as wp10c9d6c3
import run_causal_inner_local_truncation_wp10c9d6c5 as wp10c9d6c5
import run_causal_inner_monolithic_uniform_exports_wp10c9d6c as wp10c9d6c

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    causal_five_field_continuum_local_symbol,
    causal_five_field_local_symbol_stencil,
    causal_five_field_matched_principal_eigenvalues,
    causal_five_field_monolithic_frozen_tangent,
    causal_five_field_symbol_error,
    causal_packet_spectrum,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_continuum_truncation import (  # noqa: E402
    build_causal_five_field_continuum_background,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c6a"
ANALYZED_BASE_COMMIT = "14bc3e753c2530ef8799d5ad092854156a6c6551"
ANALYZED_BASE_PARENT = "c082f62f62f9c5c9f28e61c7f25f4d353a5f7a09"
ANALYZED_BASE_TREE = "43cc19475e8e80a0ec2c7ce649a4561e732670e8"
THIS_RUNNER = (
    "scripts/run_causal_inner_packet_resolution_wp10c9d6c6a.py"
)

LABELS = ("uniform_N128", "uniform_N256", "uniform_N512")
REFERENCE_LABEL = LABELS[0]
SYMBOL_RADII_OVER_RG = (2.20, 3.00, 5.00, 8.00, 11.00)
THETA_VALUES = np.linspace(0.02, 0.60, 59)
SYMBOL_TIMES_S = (0.03125, 0.0625, 0.125)
FIXED_PHYSICAL_THETA_ON_N128 = (0.10, 0.20)
PRIMARY_CONTINUUM_NODES = 769
SECONDARY_CONTINUUM_NODES = 513
SPECTRAL_ENERGY_QUANTILE = 0.99
PACKET_WINDOW_SIGMAS = 8.0

# These prospective gates consume at most half of the existing five-percent
# physical-export difference budget, except for the separately reported
# family-leakage allowance.
MAXIMUM_COMPLETE_SEMIGROUP_ERROR = 0.025
MAXIMUM_PRINCIPAL_SEMIGROUP_ERROR = 0.025
MAXIMUM_PHASE_ERROR_RADIANS = 0.025
MAXIMUM_LOG_AMPLITUDE_ERROR = 0.025
MAXIMUM_GROUP_SPEED_RELATIVE_ERROR = 0.025
MAXIMUM_FAMILY_LEAKAGE = 0.010
MAXIMUM_PRINCIPAL_BASIS_CONDITION = 1.0e4
MAXIMUM_ROW_SYMBOL_PARITY_DEFECT = 1.0e-11
MAXIMUM_OMITTED_STENCIL_FRACTION = 1.0e-11
MAXIMUM_CONTINUUM_REFERENCE_SEMIGROUP_ERROR = 2.5e-3
MAXIMUM_CONTINUUM_REFERENCE_TO_DISCRETE_RATIO = 0.10
MAXIMUM_ALIAS_FRACTION = 1.0e-3
MINIMUM_CERTIFIED_THETA = 0.20
MINIMUM_CROSS_GRID_SYMBOL_ORDER = 1.50

PARENT_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_local_truncation_wp10c9d6c5"
)
PARENT_SUMMARY = PARENT_DIRECTORY / "summary.json"
PARENT_ARRAYS = PARENT_DIRECTORY / "decisive_arrays.npz"
CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_packet_resolution_wp10c9d6c6a"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
CANONICAL_MANIFEST = (
    ROOT / "results/manifests/canonical_artifacts.csv"
)
CANONICAL_SUMMARY = (
    ROOT / "results/manifests/canonical_summary.json"
)

IMPLEMENTATION_SOURCES = (
    THIS_RUNNER,
    "src/imri_qpe/layer3_minidisk_1d/__init__.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_packet_resolution.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_monolithic_tangent.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_continuum_truncation.py",
    "tests/test_causal_inner_packet_resolution.py",
    "tests/test_causal_inner_packet_resolution_wp10c9d6c6a.py",
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
    """Rebuild the compact catalog for every canonical evidence case."""

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

    catalog_summary = json.loads(
        CANONICAL_SUMMARY.read_text(encoding="utf-8")
    )
    catalog_summary.update(
        {
            "case_count": len({str(row["case"]) for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": ANALYZED_BASE_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog_summary)


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
        raise RuntimeError("WP10c9d6c6a analyzed git identity changed")
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


def _load_parent() -> tuple[dict, dict[str, np.ndarray]]:
    summary = json.loads(PARENT_SUMMARY.read_text(encoding="utf-8"))
    if (
        summary["classification"]
        != "narrow_profile_preasymptotic_width_crossover_no_redesign"
        or not summary["passed"]
        or summary["operator_changed"]
    ):
        raise RuntimeError("WP10c9d6c5 binding parent status changed")
    with np.load(PARENT_ARRAYS, allow_pickle=False) as source:
        arrays = {name: np.array(source[name], copy=True) for name in source.files}
    return summary, arrays


def _build_construction():
    configurations, decisive, construction = (
        wp10c9d6c3._build_continuum_configurations()
    )
    if not construction["passed"]:
        raise RuntimeError("smooth continuum construction failed")
    reference = configurations[REFERENCE_LABEL]
    profile = wp10c9d6c3.SmoothCellAverageProfile(
        knots=np.asarray(
            decisive["continuum_background_knots"],
            dtype=float,
        ),
        coefficients=np.asarray(
            decisive["continuum_background_coefficients"],
            dtype=float,
        ),
        degree=wp10c9d6c3.PRIMARY_BACKGROUND_DEGREE,
        gravitational_radius=float(
            reference["context"].grid.gravitational_radius
        ),
    )
    field_scales = np.asarray(
        decisive["continuum_perturbation_field_scales"],
        dtype=float,
    )
    return configurations, decisive, construction, profile, field_scales


def _build_tangents(configurations: dict) -> dict:
    result = {}
    for label in LABELS:
        print(f"WP10c9d6c6a: build exact tangent {label}", flush=True)
        configuration = configurations[label]
        result[label] = causal_five_field_monolithic_frozen_tangent(
            configuration["context"],
            configuration["base_primitives"],
            primitive_column_scales=(
                configuration["primitive_column_scales"]
            ),
            conservation_row_scales=(
                configuration["conservation_row_scales"]
            ),
            path_quadrature_order=wp10c9d6c.PATH_QUADRATURE_ORDER,
        )
    return result


def _semigroup_relative_difference(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    maximum = 0.0
    for interval in SYMBOL_TIMES_S:
        left = expm(float(interval) * np.asarray(first, dtype=complex))
        right = expm(float(interval) * np.asarray(second, dtype=complex))
        scale = max(
            float(np.linalg.norm(left)),
            float(np.linalg.norm(right)),
            np.finfo(float).tiny,
        )
        maximum = max(
            maximum,
            float(np.linalg.norm(left - right) / scale),
        )
    return maximum


def _sorted_eigenvalues(
    numerical: np.ndarray,
    continuum: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    continuum_values, numerical_values = (
        causal_five_field_matched_principal_eigenvalues(
            numerical,
            continuum,
        )
    )
    order = np.argsort(np.imag(continuum_values))
    return continuum_values[order], numerical_values[order]


def _radius_scan(
    stencil,
    primary_background,
    secondary_background,
    log_spacing: float,
) -> tuple[dict, dict[str, np.ndarray]]:
    count = THETA_VALUES.size
    fields = 5
    metrics = {
        "complete_semigroup_error": np.empty(count),
        "principal_semigroup_error": np.empty(count),
        "phase_error_radians": np.empty(count),
        "log_amplitude_error": np.empty(count),
        "family_leakage": np.empty(count),
        "continuum_basis_condition": np.empty(count),
        "numerical_basis_condition": np.empty(count),
        "continuum_reference_error": np.empty(count),
        "continuum_reference_to_discrete_ratio": np.empty(count),
    }
    continuum_eigenvalues = np.empty((count, fields), dtype=complex)
    numerical_eigenvalues = np.empty((count, fields), dtype=complex)
    for index, theta in enumerate(THETA_VALUES):
        numerical_complete, numerical_principal = stencil.generators(theta)
        primary = causal_five_field_continuum_local_symbol(
            primary_background,
            stencil.radius,
            theta,
            log_spacing,
            stencil.field_scales,
        )
        secondary = causal_five_field_continuum_local_symbol(
            secondary_background,
            stencil.radius,
            theta,
            log_spacing,
            stencil.field_scales,
        )
        error = causal_five_field_symbol_error(
            numerical_complete,
            numerical_principal,
            primary.complete_generator_per_s,
            primary.principal_generator_per_s,
            times=SYMBOL_TIMES_S,
        )
        reference_error = max(
            _semigroup_relative_difference(
                primary.complete_generator_per_s,
                secondary.complete_generator_per_s,
            ),
            _semigroup_relative_difference(
                primary.principal_generator_per_s,
                secondary.principal_generator_per_s,
            ),
        )
        discrete_error = max(
            error.maximum_complete_semigroup_relative_error,
            error.maximum_principal_semigroup_relative_error,
            np.finfo(float).tiny,
        )
        metrics["complete_semigroup_error"][index] = (
            error.maximum_complete_semigroup_relative_error
        )
        metrics["principal_semigroup_error"][index] = (
            error.maximum_principal_semigroup_relative_error
        )
        metrics["phase_error_radians"][index] = (
            error.maximum_principal_phase_error_radians
        )
        metrics["log_amplitude_error"][index] = (
            error.maximum_principal_log_amplitude_error
        )
        metrics["family_leakage"][index] = (
            error.maximum_principal_family_leakage
        )
        metrics["continuum_basis_condition"][index] = (
            error.continuum_principal_basis_condition_number
        )
        metrics["numerical_basis_condition"][index] = (
            error.numerical_principal_basis_condition_number
        )
        metrics["continuum_reference_error"][index] = reference_error
        metrics["continuum_reference_to_discrete_ratio"][index] = (
            reference_error / discrete_error
        )
        continuum_values, numerical_values = _sorted_eigenvalues(
            numerical_principal,
            primary.principal_generator_per_s,
        )
        continuum_eigenvalues[index] = continuum_values
        numerical_eigenvalues[index] = numerical_values

    wavenumbers = THETA_VALUES / float(log_spacing)
    continuum_group = np.gradient(
        np.imag(continuum_eigenvalues),
        wavenumbers,
        axis=0,
        edge_order=2,
    )
    numerical_group = np.gradient(
        np.imag(numerical_eigenvalues),
        wavenumbers,
        axis=0,
        edge_order=2,
    )
    group_scale = max(
        float(np.max(np.abs(continuum_group))),
        np.finfo(float).tiny,
    )
    metrics["group_speed_relative_error"] = np.max(
        np.abs(numerical_group - continuum_group),
        axis=1,
    ) / group_scale
    passed = (
        (metrics["complete_semigroup_error"]
         <= MAXIMUM_COMPLETE_SEMIGROUP_ERROR)
        & (metrics["principal_semigroup_error"]
           <= MAXIMUM_PRINCIPAL_SEMIGROUP_ERROR)
        & (metrics["phase_error_radians"]
           <= MAXIMUM_PHASE_ERROR_RADIANS)
        & (metrics["log_amplitude_error"]
           <= MAXIMUM_LOG_AMPLITUDE_ERROR)
        & (metrics["group_speed_relative_error"]
           <= MAXIMUM_GROUP_SPEED_RELATIVE_ERROR)
        & (metrics["family_leakage"] <= MAXIMUM_FAMILY_LEAKAGE)
        & (metrics["continuum_basis_condition"]
           <= MAXIMUM_PRINCIPAL_BASIS_CONDITION)
        & (metrics["numerical_basis_condition"]
           <= MAXIMUM_PRINCIPAL_BASIS_CONDITION)
        & (metrics["continuum_reference_error"]
           <= MAXIMUM_CONTINUUM_REFERENCE_SEMIGROUP_ERROR)
        & (metrics["continuum_reference_to_discrete_ratio"]
           <= MAXIMUM_CONTINUUM_REFERENCE_TO_DISCRETE_RATIO)
    )
    arrays = {
        **metrics,
        "continuum_principal_eigenvalues": continuum_eigenvalues,
        "numerical_principal_eigenvalues": numerical_eigenvalues,
        "point_passed": passed.astype(np.int8),
    }
    report = {
        "radius_over_rg": None,
        "cell_index": stencil.cell_index,
        "offsets": stencil.offsets,
        "touches_boundary": stencil.touches_boundary,
        "maximum_row_symbol_parity_defect": (
            stencil.maximum_row_symbol_parity_defect
        ),
        "maximum_omitted_stencil_fraction": max(
            stencil.maximum_descriptor_omitted_fraction,
            stencil.maximum_evolving_omitted_fraction,
            stencil.maximum_principal_omitted_fraction,
        ),
        "maximum_metrics": {
            name: float(np.max(values))
            for name, values in metrics.items()
        },
        "point_pass_fraction": float(np.mean(passed)),
    }
    return report, arrays


def _symbol_contract(
    configurations: dict,
    tangents: dict,
    primary_background,
    secondary_background,
    field_scales: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    reference = configurations[REFERENCE_LABEL]
    grid = reference["context"].grid
    rg = float(grid.gravitational_radius)
    spacing = float(np.diff(np.log(grid.edges))[0])
    tangent = tangents[REFERENCE_LABEL]
    decisive: dict[str, np.ndarray] = {
        "theta_values": np.asarray(THETA_VALUES, dtype=float),
        "symbol_times_s": np.asarray(SYMBOL_TIMES_S, dtype=float),
    }
    reports = {}
    point_masks = []
    stencil_passed = True
    for target in SYMBOL_RADII_OVER_RG:
        row = int(np.argmin(np.abs(grid.centers / rg - target)))
        stencil = causal_five_field_local_symbol_stencil(
            tangent,
            row,
            field_scales,
        )
        report, arrays = _radius_scan(
            stencil,
            primary_background,
            secondary_background,
            spacing,
        )
        report["radius_over_rg"] = float(stencil.radius / rg)
        report["target_radius_over_rg"] = target
        key = f"r{target:.2f}"
        reports[key] = report
        point_masks.append(np.asarray(arrays["point_passed"], dtype=bool))
        for name, values in arrays.items():
            decisive[f"{key}__{name}"] = values
        stencil_passed = bool(
            stencil_passed
            and not stencil.touches_boundary
            and stencil.maximum_row_symbol_parity_defect
            <= MAXIMUM_ROW_SYMBOL_PARITY_DEFECT
            and report["maximum_omitted_stencil_fraction"]
            <= MAXIMUM_OMITTED_STENCIL_FRACTION
        )

    global_mask = np.logical_and.reduce(point_masks)
    contiguous = np.cumprod(global_mask.astype(np.int8)).astype(bool)
    certified_indices = np.flatnonzero(contiguous)
    certified_theta = (
        float(THETA_VALUES[certified_indices[-1]])
        if certified_indices.size
        else 0.0
    )
    decisive["global_point_passed"] = global_mask.astype(np.int8)
    decisive["global_contiguous_passed"] = contiguous.astype(np.int8)
    passed = bool(
        stencil_passed
        and certified_theta >= MINIMUM_CERTIFIED_THETA
    )
    return {
        "radius_reports": reports,
        "stencil_contract_passed": stencil_passed,
        "certified_theta": certified_theta,
        "minimum_usable_theta": MINIMUM_CERTIFIED_THETA,
        "certified_range_usable": (
            certified_theta >= MINIMUM_CERTIFIED_THETA
        ),
        "passed": passed,
    }, decisive


def _cross_grid_report(
    configurations: dict,
    tangents: dict,
    primary_background,
    field_scales: np.ndarray,
    reference_spacing: float,
) -> tuple[dict, dict[str, np.ndarray]]:
    arrays: dict[str, np.ndarray] = {}
    reports = {}
    minimum_order = float("inf")
    maximum_parity = 0.0
    maximum_omitted = 0.0
    any_boundary = False
    for target in SYMBOL_RADII_OVER_RG:
        radius_errors = {}
        for reference_theta in FIXED_PHYSICAL_THETA_ON_N128:
            errors = []
            for label in LABELS:
                configuration = configurations[label]
                grid = configuration["context"].grid
                rg = float(grid.gravitational_radius)
                spacing = float(np.diff(np.log(grid.edges))[0])
                row = int(
                    np.argmin(np.abs(grid.centers / rg - target))
                )
                stencil = causal_five_field_local_symbol_stencil(
                    tangents[label],
                    row,
                    field_scales,
                )
                theta = reference_theta * spacing / reference_spacing
                numerical_complete, numerical_principal = (
                    stencil.generators(theta)
                )
                continuum = causal_five_field_continuum_local_symbol(
                    primary_background,
                    stencil.radius,
                    theta,
                    spacing,
                    field_scales,
                )
                error = causal_five_field_symbol_error(
                    numerical_complete,
                    numerical_principal,
                    continuum.complete_generator_per_s,
                    continuum.principal_generator_per_s,
                    times=SYMBOL_TIMES_S,
                )
                errors.append(
                    max(
                        error.maximum_complete_semigroup_relative_error,
                        error.maximum_principal_semigroup_relative_error,
                    )
                )
                maximum_parity = max(
                    maximum_parity,
                    stencil.maximum_row_symbol_parity_defect,
                )
                maximum_omitted = max(
                    maximum_omitted,
                    stencil.maximum_descriptor_omitted_fraction,
                    stencil.maximum_evolving_omitted_fraction,
                    stencil.maximum_principal_omitted_fraction,
                )
                any_boundary = any_boundary or stencil.touches_boundary
            errors_array = np.asarray(errors, dtype=float)
            orders = np.log2(
                np.maximum(errors_array[:-1], np.finfo(float).tiny)
                / np.maximum(errors_array[1:], np.finfo(float).tiny)
            )
            minimum_order = min(minimum_order, float(np.min(orders)))
            theta_key = f"theta{reference_theta:.2f}"
            radius_errors[theta_key] = {
                "errors": errors_array,
                "orders": orders,
                "minimum_order": float(np.min(orders)),
            }
            arrays[
                f"r{target:.2f}__{theta_key}__cross_grid_errors"
            ] = errors_array
            arrays[
                f"r{target:.2f}__{theta_key}__cross_grid_orders"
            ] = orders
        reports[f"r{target:.2f}"] = radius_errors
    passed = bool(
        minimum_order >= MINIMUM_CROSS_GRID_SYMBOL_ORDER
        and maximum_parity <= MAXIMUM_ROW_SYMBOL_PARITY_DEFECT
        and maximum_omitted <= MAXIMUM_OMITTED_STENCIL_FRACTION
        and not any_boundary
    )
    return {
        "radius_reports": reports,
        "minimum_observed_order": minimum_order,
        "maximum_row_symbol_parity_defect": maximum_parity,
        "maximum_omitted_stencil_fraction": maximum_omitted,
        "any_selected_stencil_touches_boundary": any_boundary,
        "passed": passed,
    }, arrays


def _gaussian_cell_averages(
    sigma: float,
    spacing: float,
) -> np.ndarray:
    half_width = PACKET_WINDOW_SIGMAS * float(sigma)
    count = int(np.ceil(2.0 * half_width / spacing))
    count = max(count, 32)
    if count % 2:
        count += 1
    edges = spacing * (np.arange(count + 1) - 0.5 * count)
    factor = np.sqrt(np.pi / 2.0) * sigma / spacing
    return factor * (
        erf(edges[1:] / (np.sqrt(2.0) * sigma))
        - erf(edges[:-1] / (np.sqrt(2.0) * sigma))
    )


def _packet_resolution_report(
    parent_summary: dict,
    parent_arrays: dict[str, np.ndarray],
    certified_theta: float,
    reference_spacing: float,
    field_scales: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    vector = np.asarray(
        parent_arrays["first_cell_outgoing_physical_vector"],
        dtype=float,
    )
    normalized_vector = vector / field_scales
    profiles = {}
    arrays = {}
    definitions = wp10c9d6c5.BOUNDARY_PROFILE_DEFINITIONS
    truncation_profiles = parent_summary["band_attribution"][
        "profile_reports"
    ]
    for name, definition in definitions.items():
        sigma = float(definition["log_width"])
        envelope = _gaussian_cell_averages(sigma, reference_spacing)
        normalized = envelope[:, None] * normalized_vector[None, :]
        spectrum = causal_packet_spectrum(
            normalized,
            reference_spacing,
            quantile=SPECTRAL_ENERGY_QUANTILE,
        )
        theta_quantile = (
            spectrum.quantile_angular_wavenumber * reference_spacing
        )
        spectral_eligible = bool(
            theta_quantile <= certified_theta
            and spectrum.nyquist_alias_fraction <= MAXIMUM_ALIAS_FRACTION
        )
        boundary_dae_eligible = bool(
            all(
                band["cleanly_contracting"]
                for band in truncation_profiles[name].values()
            )
        )
        empirical = parent_summary["boundary_profile_reports"][name][
            "historical"
        ]["primary_fine"]["instantaneous"]["passed"]
        profiles[name] = {
            "center_over_rg": float(definition["center_over_rg"]),
            "log_width": sigma,
            "role": definition["role"],
            "theta_quantile": theta_quantile,
            "cells_per_sigma_on_N128": sigma / reference_spacing,
            "nyquist_alias_fraction": (
                spectrum.nyquist_alias_fraction
            ),
            "spectral_eligible": spectral_eligible,
            "boundary_dae_eligible": boundary_dae_eligible,
            "combined_boundary_packet_eligible": (
                spectral_eligible and boundary_dae_eligible
            ),
            "historical_instantaneous_gate_passed": bool(empirical),
            "historical_result_is_nonbinding_for_contract": True,
        }
        arrays[f"{name}__spectral_wavenumbers"] = (
            spectrum.angular_wavenumbers
        )
        arrays[f"{name}__spectral_energy"] = spectrum.spectral_energy
        arrays[f"{name}__spectral_cumulative_fraction"] = (
            spectrum.cumulative_energy_fraction
        )
    return {
        "spectral_energy_quantile": SPECTRAL_ENERGY_QUANTILE,
        "profiles": profiles,
        "boundary_packets_require_spectral_and_dae_contracts": True,
        "historical_results_used_to_set_threshold": False,
    }, arrays


def _config() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "operator_change": False,
        "production_defaults_changed": False,
        "labels": LABELS,
        "symbol_radii_over_rg": SYMBOL_RADII_OVER_RG,
        "theta_values": THETA_VALUES,
        "symbol_times_s": SYMBOL_TIMES_S,
        "fixed_physical_theta_on_N128": FIXED_PHYSICAL_THETA_ON_N128,
        "primary_continuum_nodes": PRIMARY_CONTINUUM_NODES,
        "secondary_continuum_nodes": SECONDARY_CONTINUUM_NODES,
        "spectral_energy_quantile": SPECTRAL_ENERGY_QUANTILE,
        "packet_window_sigmas": PACKET_WINDOW_SIGMAS,
        "gates": {
            "maximum_complete_semigroup_error": (
                MAXIMUM_COMPLETE_SEMIGROUP_ERROR
            ),
            "maximum_principal_semigroup_error": (
                MAXIMUM_PRINCIPAL_SEMIGROUP_ERROR
            ),
            "maximum_phase_error_radians": MAXIMUM_PHASE_ERROR_RADIANS,
            "maximum_log_amplitude_error": (
                MAXIMUM_LOG_AMPLITUDE_ERROR
            ),
            "maximum_group_speed_relative_error": (
                MAXIMUM_GROUP_SPEED_RELATIVE_ERROR
            ),
            "maximum_family_leakage": MAXIMUM_FAMILY_LEAKAGE,
            "maximum_principal_basis_condition": (
                MAXIMUM_PRINCIPAL_BASIS_CONDITION
            ),
            "maximum_row_symbol_parity_defect": (
                MAXIMUM_ROW_SYMBOL_PARITY_DEFECT
            ),
            "maximum_omitted_stencil_fraction": (
                MAXIMUM_OMITTED_STENCIL_FRACTION
            ),
            "maximum_continuum_reference_semigroup_error": (
                MAXIMUM_CONTINUUM_REFERENCE_SEMIGROUP_ERROR
            ),
            "maximum_continuum_reference_to_discrete_ratio": (
                MAXIMUM_CONTINUUM_REFERENCE_TO_DISCRETE_RATIO
            ),
            "maximum_alias_fraction": MAXIMUM_ALIAS_FRACTION,
            "minimum_certified_theta": MINIMUM_CERTIFIED_THETA,
            "minimum_cross_grid_symbol_order": (
                MINIMUM_CROSS_GRID_SYMBOL_ORDER
            ),
        },
        "eligibility_classes": {
            "interior": "spectral_contract",
            "boundary_overlapping": (
                "spectral_contract_plus_one_sided_DAE_truncation"
            ),
        },
    }


def _environment() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
    }


def run() -> dict:
    started = time.perf_counter()
    identity = _validate_analyzed_git_identity()
    parent_summary, parent_arrays = _load_parent()
    (
        configurations,
        construction_arrays,
        construction,
        background_profile,
        field_scales,
    ) = _build_construction()
    tangents = _build_tangents(configurations)
    reference_context = configurations[REFERENCE_LABEL]["context"]
    print("WP10c9d6c6a: build primary continuum background", flush=True)
    primary_background = build_causal_five_field_continuum_background(
        reference_context,
        background_profile.evaluate,
        node_count=PRIMARY_CONTINUUM_NODES,
    )
    print("WP10c9d6c6a: build secondary continuum background", flush=True)
    secondary_background = build_causal_five_field_continuum_background(
        reference_context,
        background_profile.evaluate,
        node_count=SECONDARY_CONTINUUM_NODES,
    )
    symbol_contract, decisive = _symbol_contract(
        configurations,
        tangents,
        primary_background,
        secondary_background,
        field_scales,
    )
    reference_spacing = float(
        np.diff(
            np.log(configurations[REFERENCE_LABEL]["context"].grid.edges)
        )[0]
    )
    cross_grid, cross_arrays = _cross_grid_report(
        configurations,
        tangents,
        primary_background,
        field_scales,
        reference_spacing,
    )
    decisive.update(cross_arrays)
    packet_report, packet_arrays = _packet_resolution_report(
        parent_summary,
        parent_arrays,
        symbol_contract["certified_theta"],
        reference_spacing,
        field_scales,
    )
    decisive.update(packet_arrays)
    decisive["continuum_perturbation_field_scales"] = field_scales
    decisive["continuum_background_knots"] = np.asarray(
        construction_arrays["continuum_background_knots"],
        dtype=float,
    )
    decisive["continuum_background_coefficients"] = np.asarray(
        construction_arrays["continuum_background_coefficients"],
        dtype=float,
    )

    passed = bool(symbol_contract["passed"] and cross_grid["passed"])
    classification = (
        "symbol_derived_packet_resolution_contract_certified"
        if passed
        else "symbol_derived_packet_resolution_contract_failed"
    )
    authorized_next = (
        "prospective_packet_suite_manifest"
        if passed
        else "none"
    )
    source_hashes, source_manifest = _source_manifest()
    config = _config()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "authorized_next": authorized_next,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "analyzed_base_parent": ANALYZED_BASE_PARENT,
        "analyzed_base_tree": ANALYZED_BASE_TREE,
        "parent_classification": parent_summary["classification"],
        "parent_classification_preserved": True,
        "operator_changed": False,
        "production_defaults_changed": False,
        "symbol_contract": symbol_contract,
        "cross_grid_symbol_report": cross_grid,
        "packet_resolution_report": packet_report,
        "continuum_construction_passed": construction["passed"],
        "configuration": config,
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "interior_packet_spectral_contract_certified": passed,
        "boundary_packet_requires_additional_DAE_contract": True,
        "prospective_packet_manifest_authorized": passed,
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
    # Write the scientific result before optional provenance packaging.  This
    # checkpoint prevents a metadata-only failure from forcing reconstruction
    # of all three expensive analytic tangents.
    _write_json(SUMMARY_PATH, summary)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "scientific_status": "REJECTED",
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        **identity,
        "implementation_base_tree": _git_value("rev-parse", "HEAD^{tree}"),
        "working_tree_status": _git_value("status", "--short"),
        "runner": THIS_RUNNER,
        "command": (
            "PYTHONPATH=src:scripts python "
            "scripts/run_causal_inner_packet_resolution_wp10c9d6c6a.py"
        ),
        "environment": _environment(),
        "parent_summary_sha256": _sha256(PARENT_SUMMARY),
        "parent_arrays_sha256": _sha256(PARENT_ARRAYS),
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
    }
    _write_json(CONFIG_PATH, config)
    _write_json(PROVENANCE_PATH, provenance)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    _refresh_canonical_catalog()
    print(json.dumps(_plain({
        "classification": classification,
        "certified_theta": symbol_contract["certified_theta"],
        "minimum_cross_grid_order": cross_grid["minimum_observed_order"],
        "authorized_next": authorized_next,
    }), indent=2, sort_keys=True))
    return summary


def refresh_metadata_only() -> dict:
    """Refresh source/provenance hashes without rerunning the symbol audit."""

    if not SUMMARY_PATH.exists() or not DECISIVE_ARRAYS.exists():
        raise RuntimeError("c6a canonical evidence is not available")
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    with np.load(DECISIVE_ARRAYS, allow_pickle=False) as source:
        arrays = {name: np.array(source[name], copy=True) for name in source.files}
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
            "scientific_status": "REJECTED",
            "source_parent_commit": ANALYZED_BASE_COMMIT,
            **_validate_analyzed_git_identity(),
            "implementation_base_tree": _git_value(
                "rev-parse",
                "HEAD^{tree}",
            ),
            "working_tree_status": _git_value("status", "--short"),
            "runner": THIS_RUNNER,
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
        help="refresh hashes without recomputing the scientific arrays",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional additional summary JSON path",
    )
    arguments = parser.parse_args()
    summary = (
        refresh_metadata_only()
        if arguments.refresh_metadata_only
        else run()
    )
    if arguments.output is not None:
        _write_json(arguments.output, summary)


if __name__ == "__main__":
    main()
