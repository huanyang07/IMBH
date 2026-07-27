"""Run the WP10c9d4a interface-inclusive fixed-geometry audit.

WP10c9d3 supplied exact continuous edge values, so its smooth manufactured
waves had zero interface jumps.  This package starts from exact cell averages,
uses the periodic form of the production unlimited quadratic stencil, checks
that the production admissibility limiter is inactive, and exercises the
complete signed interface plus within-cell fluctuation residual.

Geometry remains frozen.  This package is a binding gate before any radial
well-balance implementation and does not change the production DAE.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
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

from imri_qpe.constants import C  # noqa: E402
from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    causal_five_field_coordinate_principal_basis,
    causal_five_field_coordinate_principal_components,
    causal_five_field_lower_stress_relaxation_matrix,
    causal_five_field_periodic_cell_fluctuation_ledger,
    causal_five_field_periodic_quadratic_reconstruction,
    causal_five_field_reconstruct_face_charts,
    causal_five_field_reconstructed_fluctuation_symbol,
    make_kerr_schild_column_grid,
)


ANALYZED_BASE_COMMIT = "f0b4dcc1715647fb7300c3840546cc61ef4482b7"
WORK_PACKAGE = "WP10c9d4a"
SCHEMA_VERSION = 1
THIS_RUNNER = (
    "scripts/run_causal_inner_interface_fluctuation_audit_wp10c9d4a.py"
)
IMPLEMENTATION_SOURCES = (
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_full_fluctuation.py",
    "src/imri_qpe/layer3_minidisk_1d/__init__.py",
    THIS_RUNNER,
    "tests/test_causal_inner_full_fluctuation.py",
    "tests/test_causal_inner_interface_fluctuation_wp10c9d4a.py",
)
WP10C9D2_OUTPUT = (
    ROOT
    / "outputs/tables/"
    "causal_inner_full_fluctuation_contract_wp10c9d2.json"
)
WP10C9D2_ARRAYS = (
    ROOT
    / "outputs/tables/"
    "causal_inner_full_fluctuation_contract_wp10c9d2_arrays.npz"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/tables/"
    "causal_inner_interface_fluctuation_audit_wp10c9d4a.json"
)
DEFAULT_ARRAYS = (
    ROOT
    / "outputs/tables/"
    "causal_inner_interface_fluctuation_audit_wp10c9d4a_arrays.npz"
)
DEFAULT_CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_interface_fluctuation_wp10c9d4a"
)

TARGET_RADII_RG = (2.20, 5.00)
GRID_SIZES = (16, 32, 64)
DIRECTION_NAMES = ("mixed_transport", "thermal_material", "stress_acoustic")
RAW_DIRECTIONS = np.asarray(
    [
        [0.25, -0.20, 0.15, 0.30, -0.10],
        [0.30, 0.05, -0.10, 0.35, 0.15],
        [-0.10, 0.30, 0.20, -0.05, 0.35],
    ],
    dtype=float,
)
AMPLITUDE = 1.0e-5
WAVENUMBER = 1
REFERENCE_QUADRATURE_ORDER = 8
PATH_QUADRATURE_ORDER = 6
LOCAL_LOG_RADIUS_SPAN = 0.08
FOURIER_THETA_COARSE = (0.10, 0.20, 0.40)
SYMBOL_DIRECTIONAL_N_CELLS = 16
SYMBOL_DIRECTIONAL_EPSILON = 1.0e-6

MAXIMUM_LEDGER_DEFECT = 1.0e-10
MAXIMUM_RECONSTRUCTION_PARITY_DEFECT = 1.0e-11
MINIMUM_ADMISSIBILITY_FACTOR = 1.0 - 1.0e-14
MINIMUM_INTERFACE_JUMP_ACTIVITY = 1.0e-5
MINIMUM_MANUFACTURED_ORDER = 1.8
MAXIMUM_FINE_MANUFACTURED_ERROR = 2.0e-3
MINIMUM_SYMBOL_ORDER = 1.8
MAXIMUM_SYMBOL_DIRECTIONAL_DEFECT = 2.0e-5
MAXIMUM_PRINCIPAL_SPLIT_DEFECT = 1.0e-10


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
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    return value


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _orders(errors: np.ndarray) -> np.ndarray:
    values = np.asarray(errors, dtype=float)
    return np.log2(values[:-1] / values[1:])


def _observed_order(coarse: float, fine: float) -> float | None:
    coarse = float(coarse)
    fine = float(fine)
    if (
        not np.isfinite(coarse)
        or not np.isfinite(fine)
        or coarse <= 0.0
        or fine <= 0.0
    ):
        return None
    return float(np.log2(coarse / fine))


def _nearest_unique(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    candidates = np.asarray(values, dtype=complex)
    remaining = list(range(candidates.size))
    selected = []
    for target in np.asarray(targets, dtype=complex):
        index = min(
            remaining,
            key=lambda item: abs(candidates[item] - target),
        )
        selected.append(index)
        remaining.remove(index)
    return np.asarray(selected, dtype=int)


def _matched_eigenvalues(matrix: np.ndarray, targets: np.ndarray) -> np.ndarray:
    values = np.linalg.eigvals(np.asarray(matrix, dtype=complex))
    return values[_nearest_unique(values, targets)]


def _cell_average_wave(
    base_chart: np.ndarray,
    direction: np.ndarray,
    n_cells: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    edges = np.linspace(0.0, 2.0 * np.pi, int(n_cells) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    spacing = float(edges[1] - edges[0])
    half_phase = 0.5 * WAVENUMBER * spacing
    average_factor = float(np.sin(half_phase) / half_phase)
    averages = (
        np.asarray(base_chart, dtype=float)[None, :]
        + AMPLITUDE
        * average_factor
        * np.sin(WAVENUMBER * centers)[:, None]
        * np.asarray(direction, dtype=float)[None, :]
    )
    return averages, edges, centers, spacing


def _independent_cell_integrated_reference(
    context,
    radius: float,
    base_chart: np.ndarray,
    direction: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    """Integrate ``B(p(x)) p_x`` without calling the path-jump routine."""

    nodes, weights = np.polynomial.legendre.leggauss(
        REFERENCE_QUADRATURE_ORDER
    )
    result = np.zeros((edges.size - 1, 5), dtype=float)
    for cell, (left, right) in enumerate(
        zip(edges[:-1], edges[1:], strict=True)
    ):
        midpoint = 0.5 * (float(left) + float(right))
        half_width = 0.5 * (float(right) - float(left))
        integral = np.zeros(5, dtype=float)
        for node, weight in zip(nodes, weights, strict=True):
            coordinate = midpoint + half_width * float(node)
            chart = (
                base_chart
                + AMPLITUDE
                * np.sin(WAVENUMBER * coordinate)
                * direction
            )
            derivative = (
                AMPLITUDE
                * WAVENUMBER
                * np.cos(WAVENUMBER * coordinate)
                * direction
            )
            components = causal_five_field_coordinate_principal_components(
                context,
                radius,
                chart,
            )
            integral += (
                float(weight)
                * components.spatial_principal_matrix
                @ derivative
            )
        result[cell] = half_width * integral
    return result


def _local_production_reconstruction_parity(
    context,
    radius: float,
    averages: np.ndarray,
    periodic,
    column_scales: np.ndarray,
) -> tuple[float, float]:
    """Compare periodic traces with unchanged interior production traces."""

    n_cells = averages.shape[0]
    half_span = 0.5 * LOCAL_LOG_RADIUS_SPAN
    local_grid = make_kerr_schild_column_grid(
        radius * np.exp(-half_span),
        radius * np.exp(half_span),
        n_cells,
        context.grid.gravitational_radius,
    )
    local_context = replace(
        context,
        grid=local_grid,
        stream_sources=None,
        boundary_trace_reconstruction="cell_centered",
    ).validated()
    production = causal_five_field_reconstruct_face_charts(
        local_context,
        averages,
    )
    active_faces = slice(2, n_cells - 1)
    left_difference = (
        production.left_face_charts[active_faces]
        - periodic.cell_right_charts[1 : n_cells - 2]
    ) / column_scales[None, :]
    right_difference = (
        production.right_face_charts[active_faces]
        - periodic.cell_left_charts[2 : n_cells - 1]
    ) / column_scales[None, :]
    parity = float(
        max(
            np.max(np.abs(left_difference)),
            np.max(np.abs(right_difference)),
        )
    )
    minimum_factor = float(
        np.min(production.admissibility_factors[1:-1])
    )
    return parity, minimum_factor


def _manufactured_case(
    context,
    radius: float,
    base_chart: np.ndarray,
    direction: np.ndarray,
    column_scales: np.ndarray,
    n_cells: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    averages, edges, _centers, spacing = _cell_average_wave(
        base_chart,
        direction,
        n_cells,
    )
    reconstruction = causal_five_field_periodic_quadratic_reconstruction(
        averages
    )
    ledger = causal_five_field_periodic_cell_fluctuation_ledger(
        context,
        radius,
        reconstruction.cell_left_charts,
        reconstruction.cell_right_charts,
        quadrature_order=PATH_QUADRATURE_ORDER,
    )
    exact = _independent_cell_integrated_reference(
        context,
        radius,
        base_chart,
        direction,
        edges,
    )
    numerical = np.asarray(
        ledger.cell_principal_residuals_over_c,
        dtype=float,
    )
    error = float(
        np.linalg.norm(numerical - exact)
        / max(np.linalg.norm(exact), np.finfo(float).tiny)
    )
    interface_residual = (
        numerical - ledger.within_cell_total_jumps_over_c
    )
    interface_residual_fraction = float(
        np.linalg.norm(interface_residual)
        / max(np.linalg.norm(numerical), np.finfo(float).tiny)
    )
    parity, minimum_factor = _local_production_reconstruction_parity(
        context,
        radius,
        averages,
        reconstruction,
        column_scales,
    )
    summary = {
        "n_cells": int(n_cells),
        "spacing": spacing,
        "relative_l2_error": error,
        "maximum_absolute_interface_jump": (
            reconstruction.maximum_absolute_interface_jump
        ),
        "relative_interface_jump_activity": (
            reconstruction.relative_interface_jump_activity
        ),
        "interface_residual_fraction": interface_residual_fraction,
        "production_reconstruction_parity_defect": parity,
        "minimum_production_admissibility_factor": minimum_factor,
        "global_conservative_cycle_defect": (
            ledger.global_conservative_cycle_defect
        ),
        "global_fluctuation_assembly_defect": (
            ledger.global_fluctuation_assembly_defect
        ),
        "maximum_interface_split_defect": (
            ledger.maximum_interface_split_defect
        ),
        "global_interface_split_defect": (
            ledger.global_interface_split_defect
        ),
        "maximum_absolute_interface_split_defect": (
            ledger.maximum_absolute_interface_split_defect
        ),
    }
    arrays = {
        "cell_averages": averages,
        "cell_left_traces": reconstruction.cell_left_charts,
        "cell_right_traces": reconstruction.cell_right_charts,
        "interface_jumps": reconstruction.interface_jumps,
        "numerical_integrated_residual": numerical,
        "independent_integrated_reference": exact,
        "interface_residual": interface_residual,
    }
    return summary, arrays


def _directional_symbol_defect(
    context,
    radius: float,
    base_chart: np.ndarray,
    direction: np.ndarray,
    symbol: np.ndarray,
    *,
    theta: float,
    spacing: float,
) -> float:
    n_cells = SYMBOL_DIRECTIONAL_N_CELLS
    indices = np.arange(n_cells, dtype=float)

    def derivative(pattern: np.ndarray) -> np.ndarray:
        delta = pattern[:, None] * direction[None, :]

        def residual(sign: float) -> np.ndarray:
            charts = (
                base_chart[None, :]
                + sign * SYMBOL_DIRECTIONAL_EPSILON * delta
            )
            reconstruction = (
                causal_five_field_periodic_quadratic_reconstruction(charts)
            )
            ledger = causal_five_field_periodic_cell_fluctuation_ledger(
                context,
                radius,
                reconstruction.cell_left_charts,
                reconstruction.cell_right_charts,
                quadrature_order=PATH_QUADRATURE_ORDER,
            )
            return ledger.cell_principal_residuals_over_c / spacing

        return (
            residual(1.0) - residual(-1.0)
        ) / (2.0 * SYMBOL_DIRECTIONAL_EPSILON)

    cosine = derivative(np.cos(theta * indices))
    sine = derivative(np.sin(theta * indices))
    demodulation = np.exp(-1.0j * theta * indices)
    numerical = np.mean(
        (cosine + 1.0j * sine) * demodulation[:, None],
        axis=0,
    )
    expected = np.asarray(symbol, dtype=complex) @ direction
    return float(
        np.max(np.abs(numerical - expected))
        / max(
            float(np.max(np.abs(expected))),
            float(np.max(np.abs(numerical))),
            np.finfo(float).tiny,
        )
    )


def _symbol_audit(
    context,
    radius: float,
    base_chart: np.ndarray,
    directions: dict[str, np.ndarray],
    coarse_spacing: float,
) -> tuple[dict, dict[str, np.ndarray]]:
    components = causal_five_field_coordinate_principal_components(
        context,
        radius,
        base_chart,
    )
    basis = causal_five_field_coordinate_principal_basis(
        context,
        radius,
        base_chart,
    )
    lower = causal_five_field_lower_stress_relaxation_matrix(
        context,
        radius,
        base_chart,
    )
    temporal = np.asarray(components.temporal_storage_matrix, dtype=float)
    zero = np.zeros((5, 5), dtype=complex)

    def generator(spatial: np.ndarray, source: np.ndarray) -> np.ndarray:
        return C * np.linalg.solve(
            temporal,
            -np.asarray(spatial, dtype=complex)
            + np.asarray(source, dtype=complex),
        )

    rows = []
    arrays = {}
    minima = {
        "principal_phase": float("inf"),
        "principal_damping": float("inf"),
        "physical_relaxation": float("inf"),
    }
    maximum_split_defect = 0.0
    for theta_coarse in FOURIER_THETA_COARSE:
        level = {}
        for ratio in (1, 2, 4):
            theta = theta_coarse / ratio
            spacing = coarse_spacing / ratio
            symbol = causal_five_field_reconstructed_fluctuation_symbol(
                components,
                basis,
                theta=theta,
                spacing=spacing,
            )
            continuum_principal = generator(
                symbol.continuum_spatial_symbol,
                zero,
            )
            continuum_relaxing = generator(
                symbol.continuum_spatial_symbol,
                lower,
            )
            reconstructed_principal = generator(
                symbol.reconstructed_spatial_symbol,
                zero,
            )
            reconstructed_relaxing = generator(
                symbol.reconstructed_spatial_symbol,
                lower,
            )
            continuum_principal_values = np.linalg.eigvals(
                continuum_principal
            )
            numerical_principal_values = _matched_eigenvalues(
                reconstructed_principal,
                continuum_principal_values,
            )
            continuum_relaxing_values = np.linalg.eigvals(
                continuum_relaxing
            )
            numerical_relaxing_values = _matched_eigenvalues(
                reconstructed_relaxing,
                continuum_relaxing_values,
            )
            phase_error = float(
                np.max(
                    np.abs(
                        np.imag(numerical_principal_values)
                        - np.imag(continuum_principal_values)
                    )
                )
            )
            damping_error = float(
                np.max(
                    np.abs(
                        np.real(numerical_principal_values)
                        - np.real(continuum_principal_values)
                    )
                )
            )
            relaxing_error = float(
                np.max(
                    np.abs(
                        numerical_relaxing_values
                        - continuum_relaxing_values
                    )
                )
            )
            maximum_split_defect = max(
                maximum_split_defect,
                symbol.principal_split_closure_defect,
            )
            level[ratio] = {
                "theta": theta,
                "spacing": spacing,
                "principal_phase_error_per_s": phase_error,
                "principal_damping_error_per_s": damping_error,
                "physical_relaxation_eigenvalue_error_per_s": (
                    relaxing_error
                ),
            }
            if ratio == 1:
                key = f"theta_{theta_coarse:.2f}"
                arrays[f"{key}_continuum_principal"] = continuum_principal
                arrays[f"{key}_reconstructed_principal"] = (
                    reconstructed_principal
                )
                arrays[f"{key}_continuum_relaxing"] = continuum_relaxing
                arrays[f"{key}_reconstructed_relaxing"] = (
                    reconstructed_relaxing
                )
        orders = {}
        for name, field in (
            ("principal_phase", "principal_phase_error_per_s"),
            ("principal_damping", "principal_damping_error_per_s"),
            (
                "physical_relaxation",
                "physical_relaxation_eigenvalue_error_per_s",
            ),
        ):
            pair = (
                _observed_order(level[1][field], level[2][field]),
                _observed_order(level[2][field], level[4][field]),
            )
            valid = tuple(item for item in pair if item is not None)
            minimum = min(valid) if valid else None
            orders[name] = {
                "pair_orders": pair,
                "minimum_order": minimum,
            }
            if minimum is not None:
                minima[name] = min(minima[name], minimum)
        rows.append(
            {
                "theta_on_coarse_grid": theta_coarse,
                "by_refinement_ratio": level,
                "observed_orders": orders,
            }
        )

    parity_theta = 2.0 * np.pi / SYMBOL_DIRECTIONAL_N_CELLS
    parity_symbol = causal_five_field_reconstructed_fluctuation_symbol(
        components,
        basis,
        theta=parity_theta,
        spacing=coarse_spacing,
    )
    directional_defects = {
        name: _directional_symbol_defect(
            context,
            radius,
            base_chart,
            direction,
            parity_symbol.reconstructed_spatial_symbol,
            theta=parity_theta,
            spacing=coarse_spacing,
        )
        for name, direction in directions.items()
    }
    maximum_directional = max(directional_defects.values())
    passed = bool(
        all(value >= MINIMUM_SYMBOL_ORDER for value in minima.values())
        and maximum_directional <= MAXIMUM_SYMBOL_DIRECTIONAL_DEFECT
        and maximum_split_defect <= MAXIMUM_PRINCIPAL_SPLIT_DEFECT
    )
    return {
        "rows": rows,
        "minimum_observed_orders": minima,
        "directional_ledger_symbol_defects": directional_defects,
        "maximum_directional_ledger_symbol_defect": maximum_directional,
        "maximum_principal_split_defect": maximum_split_defect,
        "all_five_families_included": True,
        "principal_only_and_physical_relaxation_separated": True,
        "passed": passed,
    }, arrays


def _source_manifest() -> tuple[dict[str, str], str]:
    hashes = {
        path: _sha256(ROOT / path)
        for path in IMPLEMENTATION_SOURCES
    }
    digest = hashlib.sha256()
    for path, value in sorted(hashes.items()):
        digest.update(path.encode("utf-8"))
        digest.update(value.encode("ascii"))
    return hashes, digest.hexdigest()


def run() -> tuple[dict, dict[str, np.ndarray]]:
    started = time.perf_counter()
    required = (
        WP10C9D2_OUTPUT,
        WP10C9D2_ARRAYS,
        wp10c9d0.WP10C8Z_OUTPUT,
        wp10c9d0.WP10C8Z_ARRAYS,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "WP10c9d4a requires prior evidence: " + ", ".join(missing)
        )
    patch_arrays = wp10c9d0._load_npz(wp10c9d0.WP10C8Z_ARRAYS)
    configurations = wp10c9d0._patch_configurations(patch_arrays)
    configuration = configurations["N128_exterior_N256_inner_c48"]
    context = configuration["context"]
    if context.spatial_reconstruction != "quadratic_admissible":
        raise RuntimeError("WP10c9d4a requires the production quadratic mode")
    primitives = np.asarray(configuration["base_primitives"], dtype=float)
    center_radii_rg = (
        np.asarray(context.grid.centers, dtype=float)
        / context.grid.gravitational_radius
    )

    cases = {}
    arrays = {}
    all_passed = True
    for target_radius in TARGET_RADII_RG:
        cell = int(np.argmin(np.abs(center_radii_rg - target_radius)))
        radius = float(context.grid.centers[cell])
        base_chart = np.asarray(primitives[cell], dtype=float)
        components = causal_five_field_coordinate_principal_components(
            context,
            radius,
            base_chart,
        )
        directions = {
            name: (
                raw
                * components.primitive_column_scales
                / np.linalg.norm(raw)
            )
            for name, raw in zip(
                DIRECTION_NAMES,
                RAW_DIRECTIONS,
                strict=True,
            )
        }
        radius_case = {
            "radius_rg": radius / context.grid.gravitational_radius,
            "directions": {},
        }
        for name, direction in directions.items():
            ladder = []
            errors = []
            for n_cells in GRID_SIZES:
                summary, case_arrays = _manufactured_case(
                    context,
                    radius,
                    base_chart,
                    direction,
                    components.primitive_column_scales,
                    n_cells,
                )
                ladder.append(summary)
                errors.append(summary["relative_l2_error"])
                prefix = (
                    f"r{target_radius:.2f}_{name}_N{n_cells}"
                )
                for array_name, values in case_arrays.items():
                    arrays[f"{prefix}_{array_name}"] = values
            observed_orders = _orders(np.asarray(errors, dtype=float))
            maximum_ledger = max(
                max(
                    item["global_conservative_cycle_defect"],
                    item["global_fluctuation_assembly_defect"],
                    item["global_interface_split_defect"],
                )
                for item in ladder
            )
            maximum_parity = max(
                item["production_reconstruction_parity_defect"]
                for item in ladder
            )
            minimum_admissibility = min(
                item["minimum_production_admissibility_factor"]
                for item in ladder
            )
            minimum_activity = min(
                item["relative_interface_jump_activity"]
                for item in ladder
            )
            minimum_residual_fraction = min(
                item["interface_residual_fraction"]
                for item in ladder
            )
            passed = bool(
                maximum_ledger <= MAXIMUM_LEDGER_DEFECT
                and maximum_parity
                <= MAXIMUM_RECONSTRUCTION_PARITY_DEFECT
                and minimum_admissibility
                >= MINIMUM_ADMISSIBILITY_FACTOR
                and minimum_activity >= MINIMUM_INTERFACE_JUMP_ACTIVITY
                and minimum_residual_fraction
                >= MINIMUM_INTERFACE_JUMP_ACTIVITY
                and float(np.min(observed_orders))
                >= MINIMUM_MANUFACTURED_ORDER
                and float(errors[-1])
                <= MAXIMUM_FINE_MANUFACTURED_ERROR
            )
            radius_case["directions"][name] = {
                "smooth_ladder": ladder,
                "observed_orders": observed_orders,
                "minimum_observed_order": float(np.min(observed_orders)),
                "fine_relative_l2_error": float(errors[-1]),
                "maximum_ledger_defect": maximum_ledger,
                "maximum_production_reconstruction_parity_defect": (
                    maximum_parity
                ),
                "minimum_production_admissibility_factor": (
                    minimum_admissibility
                ),
                "minimum_interface_jump_activity": minimum_activity,
                "minimum_interface_residual_fraction": (
                    minimum_residual_fraction
                ),
                "passed": passed,
            }
            all_passed = all_passed and passed

        coarse_spacing = float(np.diff(context.grid.edges)[cell])
        symbol, symbol_arrays = _symbol_audit(
            context,
            radius,
            base_chart,
            directions,
            coarse_spacing,
        )
        for name, values in symbol_arrays.items():
            arrays[f"r{target_radius:.2f}_symbol_{name}"] = values
        radius_case["fourier_symbol"] = symbol
        radius_case["passed"] = bool(
            symbol["passed"]
            and all(
                item["passed"]
                for item in radius_case["directions"].values()
            )
        )
        all_passed = all_passed and radius_case["passed"]
        cases[f"{target_radius:.2f}rg"] = radius_case

    source_hashes, source_manifest = _source_manifest()
    classification = (
        "interface_inclusive_fixed_geometry_gate_passed_"
        "radial_well_balance_authorized"
        if all_passed
        else "interface_inclusive_fixed_geometry_gate_failed_"
        "radial_work_blocked"
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "implementation_parent_commit": ANALYZED_BASE_COMMIT,
        "runner": THIS_RUNNER,
        "classification": classification,
        "interface_inclusive_gate_passed": all_passed,
        "radial_well_balance_audit_authorized": all_passed,
        "production_operator_authorized": False,
        "nonlinear_candidate_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "gates": {
            "maximum_ledger_defect": MAXIMUM_LEDGER_DEFECT,
            "maximum_reconstruction_parity_defect": (
                MAXIMUM_RECONSTRUCTION_PARITY_DEFECT
            ),
            "minimum_admissibility_factor": (
                MINIMUM_ADMISSIBILITY_FACTOR
            ),
            "minimum_interface_jump_activity": (
                MINIMUM_INTERFACE_JUMP_ACTIVITY
            ),
            "minimum_manufactured_order": MINIMUM_MANUFACTURED_ORDER,
            "maximum_fine_manufactured_error": (
                MAXIMUM_FINE_MANUFACTURED_ERROR
            ),
            "minimum_symbol_order": MINIMUM_SYMBOL_ORDER,
            "maximum_symbol_directional_defect": (
                MAXIMUM_SYMBOL_DIRECTIONAL_DEFECT
            ),
            "maximum_principal_split_defect": (
                MAXIMUM_PRINCIPAL_SPLIT_DEFECT
            ),
        },
        "audit_configuration": "N128_exterior_N256_inner_c48",
        "production_reconstruction_mode": context.spatial_reconstruction,
        "periodic_boundary_treatment": (
            "wrapped production unlimited quadratic interior stencil; "
            "not a radial boundary condition"
        ),
        "reference_contract": (
            "independent Gauss integration of B(p(x)) p_x; "
            "does not call the complete path-jump routine"
        ),
        "cases": cases,
        "input_hashes": {
            _relative(path): _sha256(path) for path in required
        },
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    return payload, arrays


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_canonical(
    directory: Path,
    payload: dict,
    arrays: dict[str, np.ndarray],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    decisive_arrays = {
        name: values
        for name, values in arrays.items()
        if (
            name.endswith("_interface_jumps")
            or name.endswith("_numerical_integrated_residual")
            or name.endswith("_independent_integrated_reference")
            or "_symbol_" in name
        )
    }
    arrays_path = directory / "decisive_arrays.npz"
    np.savez_compressed(arrays_path, **decisive_arrays)
    summary = dict(payload)
    summary.pop("runtime_seconds", None)
    summary["decisive_arrays_sha256"] = _sha256(arrays_path)
    summary["decisive_array_hashes"] = {
        name: _array_sha256(values)
        for name, values in sorted(decisive_arrays.items())
    }
    _write_json(directory / "summary.json", summary)
    _write_json(
        directory / "config.json",
        {
            "target_radii_rg": TARGET_RADII_RG,
            "grid_sizes": GRID_SIZES,
            "direction_names": DIRECTION_NAMES,
            "raw_directions": RAW_DIRECTIONS,
            "amplitude": AMPLITUDE,
            "wavenumber": WAVENUMBER,
            "reference_quadrature_order": REFERENCE_QUADRATURE_ORDER,
            "path_quadrature_order": PATH_QUADRATURE_ORDER,
            "fourier_theta_coarse": FOURIER_THETA_COARSE,
            "gates": payload["gates"],
        },
    )
    _write_json(
        directory / "provenance.json",
        {
            "work_package": WORK_PACKAGE,
            "scientific_status": "CERTIFIED",
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "source_parent_commit": ANALYZED_BASE_COMMIT,
            "implementation_parent_commit": ANALYZED_BASE_COMMIT,
            "implementation_source_manifest_sha256": payload[
                "implementation_source_manifest_sha256"
            ],
            "implementation_source_hashes": payload[
                "implementation_source_hashes"
            ],
            "input_hashes": payload["input_hashes"],
            "generation_command": (
                "PYTHONPATH=src python3 "
                "scripts/run_causal_inner_interface_fluctuation_audit_"
                "wp10c9d4a.py"
            ),
            "establishes": (
                "The reconstructed signed interface plus within-cell "
                "fixed-geometry gate stated by the summary classification."
            ),
            "does_not_establish": (
                "A radial well-balanced operator, nonlinear path, production "
                "promotion, fixed-Q average, or reduced slow evolution."
            ),
        },
    )
    checksum_paths = (
        directory / "config.json",
        directory / "decisive_arrays.npz",
        directory / "provenance.json",
        directory / "summary.json",
    )
    lines = [
        f"{_sha256(path)}  {path.name}"
        for path in checksum_paths
    ]
    (directory / "SHA256SUMS.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--arrays", type=Path, default=DEFAULT_ARRAYS)
    parser.add_argument(
        "--canonical-directory",
        type=Path,
        default=DEFAULT_CANONICAL_DIRECTORY,
    )
    arguments = parser.parse_args()
    payload, arrays = run()
    arguments.arrays.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.arrays, **arrays)
    payload["arrays_path"] = _relative(arguments.arrays)
    payload["arrays_sha256"] = _sha256(arguments.arrays)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(arguments.output, payload)
    _write_canonical(arguments.canonical_directory, payload, arrays)
    print(
        json.dumps(
            {
                "classification": payload["classification"],
                "interface_inclusive_gate_passed": payload[
                    "interface_inclusive_gate_passed"
                ],
                "runtime_seconds": payload["runtime_seconds"],
                "output": _relative(arguments.output),
                "canonical_directory": _relative(
                    arguments.canonical_directory
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
