"""Run the WP10c9d3 fixed-geometry complete fluctuation cell ledger.

This package assembles the WP10c9d2 complete path jumps in the standard
wave-propagation balance: positive fluctuation from the left interface,
complete within-cell path, and negative fluctuation from the right interface.
It tests constant states, discontinuous interface traces, and smooth periodic
manufactured waves.  Geometry is frozen at each audit radius, so radial
well-balancing remains a later gate.
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
    causal_five_field_coordinate_principal_components,
    causal_five_field_periodic_cell_fluctuation_ledger,
)


BASE_COMMIT = "90f82c238e802abe22aa15b42f62b7d929048a60"
WORK_PACKAGE = "WP10c9d3"
SCHEMA_VERSION = 1
THIS_RUNNER = (
    "scripts/run_causal_inner_cell_fluctuation_ledger_wp10c9d3.py"
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
    "causal_inner_cell_fluctuation_ledger_wp10c9d3.json"
)
DEFAULT_ARRAYS = (
    ROOT
    / "outputs/tables/"
    "causal_inner_cell_fluctuation_ledger_wp10c9d3_arrays.npz"
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

MAXIMUM_LEDGER_DEFECT = 1.0e-10
MINIMUM_MANUFACTURED_ORDER = 1.8
MAXIMUM_FINE_MANUFACTURED_ERROR = 1.0e-3


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


def _orders(errors: np.ndarray) -> np.ndarray:
    errors = np.asarray(errors, dtype=float)
    return np.log2(errors[:-1] / errors[1:])


def _smooth_wave_case(
    context,
    radius: float,
    base_chart: np.ndarray,
    direction: np.ndarray,
    n_cells: int,
) -> tuple[dict, dict]:
    edges = np.linspace(0.0, 2.0 * np.pi, n_cells + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dx = float(edges[1] - edges[0])
    edge_values = (
        base_chart[None, :]
        + AMPLITUDE
        * np.sin(WAVENUMBER * edges)[:, None]
        * direction[None, :]
    )
    edge_values[-1] = edge_values[0]
    ledger = causal_five_field_periodic_cell_fluctuation_ledger(
        context,
        radius,
        edge_values[:-1],
        edge_values[1:],
    )
    exact = []
    for coordinate in centers:
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
        exact.append(components.spatial_principal_matrix @ derivative)
    exact_array = np.asarray(exact, dtype=float)
    numerical = ledger.cell_principal_residuals_over_c / dx
    error = float(
        np.linalg.norm(numerical - exact_array)
        / max(np.linalg.norm(exact_array), np.finfo(float).tiny)
    )
    return (
        {
            "n_cells": n_cells,
            "relative_l2_error": error,
            "global_conservative_cycle_defect": (
                ledger.global_conservative_cycle_defect
            ),
            "global_fluctuation_assembly_defect": (
                ledger.global_fluctuation_assembly_defect
            ),
            "maximum_interface_split_defect": (
                ledger.maximum_interface_split_defect
            ),
        },
        {
            "numerical": numerical,
            "exact": exact_array,
        },
    )


def _piecewise_constant_case(
    context,
    radius: float,
    base_chart: np.ndarray,
    direction: np.ndarray,
) -> dict:
    n_cells = 32
    centers = (np.arange(n_cells, dtype=float) + 0.5) * (
        2.0 * np.pi / n_cells
    )
    values = (
        base_chart[None, :]
        + AMPLITUDE
        * np.sin(WAVENUMBER * centers)[:, None]
        * direction[None, :]
    )
    ledger = causal_five_field_periodic_cell_fluctuation_ledger(
        context,
        radius,
        values,
        values,
    )
    return {
        "global_conservative_cycle_defect": (
            ledger.global_conservative_cycle_defect
        ),
        "global_fluctuation_assembly_defect": (
            ledger.global_fluctuation_assembly_defect
        ),
        "maximum_interface_split_defect": (
            ledger.maximum_interface_split_defect
        ),
        "maximum_within_cell_jump": float(
            np.max(np.abs(ledger.within_cell_total_jumps_over_c))
        ),
    }


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
            "WP10c9d3 requires prior evidence: " + ", ".join(missing)
        )
    patch_arrays = wp10c9d0._load_npz(wp10c9d0.WP10C8Z_ARRAYS)
    configurations = wp10c9d0._patch_configurations(patch_arrays)
    configuration = configurations["N128_exterior_N256_inner_c48"]
    context = configuration["context"]
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
        constant = np.repeat(base_chart[None, :], 8, axis=0)
        constant_ledger = (
            causal_five_field_periodic_cell_fluctuation_ledger(
                context,
                radius,
                constant,
                constant,
            )
        )
        radius_cases = {
            "radius_rg": radius / context.grid.gravitational_radius,
            "maximum_constant_residual": float(
                np.max(
                    np.abs(
                        constant_ledger.cell_principal_residuals_over_c
                    )
                )
            ),
            "directions": {},
        }
        for direction_name, raw_direction in zip(
            DIRECTION_NAMES,
            RAW_DIRECTIONS,
            strict=True,
        ):
            direction = (
                raw_direction
                * components.primitive_column_scales
                / np.linalg.norm(raw_direction)
            )
            smooth_summaries = []
            errors = []
            for n_cells in GRID_SIZES:
                summary, case_arrays = _smooth_wave_case(
                    context,
                    radius,
                    base_chart,
                    direction,
                    n_cells,
                )
                smooth_summaries.append(summary)
                errors.append(summary["relative_l2_error"])
                prefix = (
                    f"r{target_radius:.2f}_{direction_name}_N{n_cells}"
                )
                arrays[f"{prefix}_numerical"] = case_arrays["numerical"]
                arrays[f"{prefix}_exact"] = case_arrays["exact"]
            errors_array = np.asarray(errors, dtype=float)
            observed_orders = _orders(errors_array)
            piecewise = _piecewise_constant_case(
                context,
                radius,
                base_chart,
                direction,
            )
            ledger_maximum = max(
                *(
                    item["global_conservative_cycle_defect"]
                    for item in smooth_summaries
                ),
                *(
                    item["global_fluctuation_assembly_defect"]
                    for item in smooth_summaries
                ),
                *(
                    item["maximum_interface_split_defect"]
                    for item in smooth_summaries
                ),
                piecewise["global_conservative_cycle_defect"],
                piecewise["global_fluctuation_assembly_defect"],
                piecewise["maximum_interface_split_defect"],
            )
            passed = bool(
                ledger_maximum <= MAXIMUM_LEDGER_DEFECT
                and float(np.min(observed_orders))
                >= MINIMUM_MANUFACTURED_ORDER
                and float(errors_array[-1])
                <= MAXIMUM_FINE_MANUFACTURED_ERROR
                and piecewise["maximum_within_cell_jump"] == 0.0
            )
            radius_cases["directions"][direction_name] = {
                "smooth_ladder": smooth_summaries,
                "observed_orders": observed_orders,
                "minimum_observed_order": float(
                    np.min(observed_orders)
                ),
                "fine_relative_l2_error": float(errors_array[-1]),
                "piecewise_constant_interface_case": piecewise,
                "maximum_ledger_defect": ledger_maximum,
                "passed": passed,
            }
            all_passed = all_passed and passed
        radius_cases["passed"] = bool(
            radius_cases["maximum_constant_residual"] == 0.0
            and all(
                item["passed"]
                for item in radius_cases["directions"].values()
            )
        )
        all_passed = all_passed and radius_cases["passed"]
        cases[f"{target_radius:.2f}rg"] = radius_cases

    classification = (
        "fixed_geometry_full_fluctuation_assembly_passed_"
        "radial_well_balance_is_next_gate"
        if all_passed
        else "fixed_geometry_full_fluctuation_assembly_failed"
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "base_commit": BASE_COMMIT,
        "runner": THIS_RUNNER,
        "classification": classification,
        "method_contract_passed": all_passed,
        "radial_well_balance_audit_authorized": all_passed,
        "production_operator_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "gates": {
            "maximum_ledger_defect": MAXIMUM_LEDGER_DEFECT,
            "minimum_manufactured_order": MINIMUM_MANUFACTURED_ORDER,
            "maximum_fine_manufactured_error": (
                MAXIMUM_FINE_MANUFACTURED_ERROR
            ),
        },
        "audit_configuration": "N128_exterior_N256_inner_c48",
        "cases": cases,
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
