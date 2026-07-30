#!/usr/bin/env python3
"""Test a variable-coefficient windowed contract after c6a1.

WP10c9d6c6a rejected the fixed-radius 0.125 s complete-symbol contract at
theta=0.18.  WP10c9d6c6a1 showed that the exact full-symbol contributions
converge and that overlap-tracked variable-radius rays remain well below the
unchanged 0.025 budget at theta=0.20.  Rays do not contain finite windows or
the complete variable-coefficient spatial coupling.

This package therefore propagates prospectively frozen analytic window
probes with the unchanged full N128/N256/N512 monolithic tangents.  It uses
proper-measure restriction and a controlled three-level Richardson
reference.  A pass may authorize a packet-definition manifest only.  It
cannot authorize a physical packet campaign, embedded coupling, nonlinear
evolution, production promotion, fixed-Q averaging, or slow reduction.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
from scipy.interpolate import CubicSpline
from scipy.sparse.linalg import expm_multiply


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_causal_inner_continuum_lift_wp10c9d6c3 as c3
import run_causal_inner_full_symbol_limiter_wp10c9d6c6a1 as c6a1
import run_causal_inner_packet_resolution_wp10c9d6c6a as c6a

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    causal_five_field_characteristic_basis,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_characteristic_phase import (  # noqa: E402
    CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_resolution import (  # noqa: E402
    causal_packet_spectrum,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_windowed_contract import (  # noqa: E402
    causal_align_characteristic_field,
    causal_field_history_inner_product,
    causal_field_history_norm,
    causal_restrict_proper_cell_averages,
    causal_sine_power_window,
    causal_trapezoid_weights,
    causal_windowed_richardson_reference,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c6a2"
ANALYZED_BASE_COMMIT = "8d7f4ebcf5ab3fe97dfdc54abf2eb82c5ffb0858"
ANALYZED_BASE_PARENT = "76e34e8ef0b4688b0c371c27cb2288c880419961"
ANALYZED_BASE_TREE = "f72a165f5be910170e5d920038f05a4de104ecf4"
THIS_RUNNER = (
    "scripts/run_causal_inner_windowed_contract_wp10c9d6c6a2.py"
)

LABELS = ("uniform_N128", "uniform_N256", "uniform_N512")
REFERENCE_LABEL = LABELS[0]
TIME_HORIZON_S = 0.125
TIME_SAMPLE_COUNT = 65
PRIMARY_PROJECTION_ORDER = 24
SECONDARY_PROJECTION_ORDER = 12
CHARACTERISTIC_FIELD_NODES = 513
WINDOW_POWERS = (2, 4)
LOW_CONTROL_POWER = 2
BINDING_POWER = 4
MIXED_COEFFICIENTS = (0.35, -0.40, 0.50, -0.45, 0.30)
PROBE_AMPLITUDE = 1.0e-2
SPECTRAL_ENERGY_QUANTILE = 0.99

# The c6a physical gates are unchanged.
MAXIMUM_COMPLETE_SEMIGROUP_ERROR = (
    c6a.MAXIMUM_COMPLETE_SEMIGROUP_ERROR
)
MINIMUM_USABLE_THETA = c6a.MINIMUM_CERTIFIED_THETA
MAXIMUM_ALIAS_FRACTION = c6a.MAXIMUM_ALIAS_FRACTION
MINIMUM_CROSS_GRID_ORDER = c6a.MINIMUM_CROSS_GRID_SYMBOL_ORDER

# Prospectively frozen window/reference gates.
MAXIMUM_BINDING_THETA = 0.30
MINIMUM_COMPONENT_ORDER = 1.25
MINIMUM_REFINEMENT_ERROR_COSINE = 0.90
MAXIMUM_UNCERTAINTY_TO_FINE_DIFFERENCE_RATIO = 0.10
MAXIMUM_ENDPOINT_CELL_FRACTION = 5.0e-3
MINIMUM_CHARACTERISTIC_FIELD_OVERLAP = 0.995
MAXIMUM_CHARACTERISTIC_NORM_DEFECT = 1.0e-12
MAXIMUM_RESTART_TO_FINE_DIFFERENCE_RATIO = 0.10
MAXIMUM_BOUNDARY_INTEGRAL_UNCERTAINTY_TO_FINE_DIFFERENCE_RATIO = 0.10

PARENT_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_full_symbol_limiter_wp10c9d6c6a1"
)
PARENT_CONFIG = PARENT_DIRECTORY / "config.json"
PARENT_SUMMARY = PARENT_DIRECTORY / "summary.json"
PARENT_ARRAYS = PARENT_DIRECTORY / "decisive_arrays.npz"
PARENT_PROVENANCE = PARENT_DIRECTORY / "provenance.json"
C6A_SUMMARY = (
    ROOT
    / "results/canonical/"
    "causal_inner_packet_resolution_wp10c9d6c6a/summary.json"
)
C3_ARRAYS = (
    ROOT
    / "results/canonical/"
    "causal_inner_continuum_lift_wp10c9d6c3/decisive_arrays.npz"
)
CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_windowed_contract_wp10c9d6c6a2"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
TRAPEZOID_PREFLIGHT_SUMMARY_PATH = (
    CANONICAL_DIRECTORY / "trapezoid_preflight_summary.json"
)
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

IMPLEMENTATION_SOURCES = (
    THIS_RUNNER,
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_windowed_contract.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_monolithic_tangent.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_packet_resolution.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_characteristic_phase.py",
    "tests/test_causal_inner_windowed_contract.py",
    "tests/test_causal_inner_windowed_contract_wp10c9d6c6a2.py",
)


def _probe_definitions() -> dict[str, dict]:
    definitions: dict[str, dict] = {}
    for power in WINDOW_POWERS:
        role = (
            "binding_theta20_window"
            if power == BINDING_POWER
            else "low_wavenumber_control"
        )
        for family in CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES:
            definitions[f"p{power}__{family}"] = {
                "window_power": power,
                "role": role,
                "family": family,
                "mixed_coefficients": None,
            }
    definitions[f"p{BINDING_POWER}__mixed"] = {
        "window_power": BINDING_POWER,
        "role": "binding_theta20_window",
        "family": "mixed",
        "mixed_coefficients": MIXED_COEFFICIENTS,
    }
    return definitions


PROBE_DEFINITIONS = _probe_definitions()
PROBE_NAMES = tuple(PROBE_DEFINITIONS)


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
        raise RuntimeError("WP10c9d6c6a2 analyzed git identity changed")
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


def _load_parent() -> tuple[dict, dict]:
    parent = json.loads(PARENT_SUMMARY.read_text(encoding="utf-8"))
    failed_c6a = json.loads(C6A_SUMMARY.read_text(encoding="utf-8"))
    if (
        parent["classification"]
        != (
            "full_symbol_limiter_convergent_accumulation_"
            "windowed_contract_audit_authorized"
        )
        or parent["authorized_next"]
        != "WP10c9d6c6a2_variable_coefficient_windowed_contract"
        or not parent["passed"]
        or parent["operator_changed"]
    ):
        raise RuntimeError("c6a1 authorization changed")
    if (
        failed_c6a["classification"]
        != "symbol_derived_packet_resolution_contract_failed"
        or failed_c6a["passed"]
    ):
        raise RuntimeError("c6a rejection was not preserved")
    return parent, failed_c6a


@dataclass(frozen=True)
class _TrackedCharacteristicInterpolator:
    spline: CubicSpline
    field_scales: np.ndarray
    family_labels: tuple[str, ...]

    def evaluate(self, radii: np.ndarray) -> np.ndarray:
        values = np.asarray(
            self.spline(np.log(np.asarray(radii, dtype=float))),
            dtype=float,
        )
        dimensionless = values / self.field_scales[None, :, None]
        norms = np.linalg.norm(dimensionless, axis=1)
        if np.any(norms <= np.finfo(float).tiny):
            raise RuntimeError("interpolated characteristic field is singular")
        return values / norms[:, None, :]


def _build_characteristic_interpolator(
    configurations: dict,
    construction_arrays: dict[str, np.ndarray],
) -> tuple[_TrackedCharacteristicInterpolator, dict, dict[str, np.ndarray]]:
    reference = configurations[REFERENCE_LABEL]
    grid = reference["context"].grid
    field_scales = np.asarray(
        construction_arrays["continuum_perturbation_field_scales"],
        dtype=float,
    )
    background = c3.SmoothCellAverageProfile(
        knots=np.asarray(
            construction_arrays["continuum_background_knots"],
            dtype=float,
        ),
        coefficients=np.asarray(
            construction_arrays["continuum_background_coefficients"],
            dtype=float,
        ),
        degree=c3.PRIMARY_BACKGROUND_DEGREE,
        gravitational_radius=float(grid.gravitational_radius),
    )
    radii = np.geomspace(
        float(grid.edges[0]),
        float(grid.edges[-1]),
        CHARACTERISTIC_FIELD_NODES,
    )
    physical = np.empty(
        (radii.size, 5, 5),
        dtype=float,
    )
    maximum_eigenpair = 0.0
    maximum_condition = 0.0
    minimum_speed_gap = float("inf")
    for index, radius in enumerate(radii):
        basis = causal_five_field_characteristic_basis(
            reference["context"],
            float(radius),
            background.evaluate(np.asarray([radius]))[0],
            field_scales,
        )
        if tuple(basis.family_labels) != tuple(
            CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES
        ):
            raise RuntimeError("characteristic family order changed")
        physical[index] = basis.physical_right_eigenvectors
        maximum_eigenpair = max(
            maximum_eigenpair,
            float(basis.maximum_eigenpair_defect),
        )
        maximum_condition = max(
            maximum_condition,
            float(basis.condition_number),
        )
        speeds = np.sort(
            np.asarray(basis.coordinate_speeds_over_c, dtype=float)
        )
        minimum_speed_gap = min(
            minimum_speed_gap,
            float(np.min(np.diff(speeds))),
        )
    aligned = causal_align_characteristic_field(
        physical,
        field_scales,
    )
    interpolator = _TrackedCharacteristicInterpolator(
        spline=CubicSpline(
            np.log(radii),
            aligned.physical_right_eigenvectors,
            axis=0,
        ),
        field_scales=field_scales,
        family_labels=tuple(
            CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES
        ),
    )
    report = {
        "node_count": int(radii.size),
        "minimum_adjacent_overlap": (
            aligned.minimum_adjacent_overlap
        ),
        "maximum_dimensionless_norm_defect": (
            aligned.maximum_dimensionless_norm_defect
        ),
        "maximum_eigenpair_defect": maximum_eigenpair,
        "maximum_basis_condition_number": maximum_condition,
        "minimum_coordinate_speed_gap_over_c": minimum_speed_gap,
        "passed": bool(
            aligned.minimum_adjacent_overlap
            >= MINIMUM_CHARACTERISTIC_FIELD_OVERLAP
            and aligned.maximum_dimensionless_norm_defect
            <= MAXIMUM_CHARACTERISTIC_NORM_DEFECT
            and minimum_speed_gap > 0.0
        ),
    }
    arrays = {
        "characteristic_field_radii": radii,
        "characteristic_field_physical_vectors": (
            aligned.physical_right_eigenvectors
        ),
    }
    return interpolator, report, arrays


def _probe_evaluator(
    definition: dict,
    interpolator: _TrackedCharacteristicInterpolator,
    *,
    lower_radius: float,
    upper_radius: float,
):
    lower_log = float(np.log(lower_radius))
    upper_log = float(np.log(upper_radius))
    power = int(definition["window_power"])
    family = str(definition["family"])
    if family == "mixed":
        coefficients = np.asarray(
            definition["mixed_coefficients"],
            dtype=float,
        )
    else:
        family_index = interpolator.family_labels.index(family)

    def evaluate(radii: np.ndarray) -> np.ndarray:
        physical_radii = np.asarray(radii, dtype=float)
        window = causal_sine_power_window(
            np.log(physical_radii),
            lower_log_radius=lower_log,
            upper_log_radius=upper_log,
            power=power,
        )
        bases = interpolator.evaluate(physical_radii)
        if family == "mixed":
            vector = np.einsum("rij,j->ri", bases, coefficients)
            dimensionless = (
                vector / interpolator.field_scales[None, :]
            )
            norms = np.linalg.norm(dimensionless, axis=1)
            vector = vector / norms[:, None]
        else:
            vector = bases[:, :, family_index]
        return PROBE_AMPLITUDE * window[:, None] * vector

    return evaluate


def _build_probes(
    configurations: dict,
    construction_arrays: dict[str, np.ndarray],
    interpolator: _TrackedCharacteristicInterpolator,
) -> tuple[dict, dict, dict[str, np.ndarray]]:
    field_scales = np.asarray(
        construction_arrays["continuum_perturbation_field_scales"],
        dtype=float,
    )
    directions: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    arrays: dict[str, np.ndarray] = {}
    evaluators = {}
    reference_grid = configurations[REFERENCE_LABEL]["context"].grid
    lower_radius = float(reference_grid.edges[0])
    upper_radius = float(reference_grid.edges[-1])
    for name, definition in PROBE_DEFINITIONS.items():
        evaluators[name] = _probe_evaluator(
            definition,
            interpolator,
            lower_radius=lower_radius,
            upper_radius=upper_radius,
        )

    maximum_endpoint = 0.0
    for label in LABELS:
        configuration = configurations[label]
        grid = configuration["context"].grid
        columns = np.asarray(
            configuration["primitive_column_scales"],
            dtype=float,
        ).reshape(-1, 5)
        directions[label] = {}
        for name in PROBE_NAMES:
            primary = c3._project_callable_to_cells(
                grid,
                evaluators[name],
                quadrature_order=PRIMARY_PROJECTION_ORDER,
            )
            secondary = c3._project_callable_to_cells(
                grid,
                evaluators[name],
                quadrature_order=SECONDARY_PROJECTION_ORDER,
            )
            directions[label][name] = {
                "primary_physical": primary,
                "secondary_physical": secondary,
                "primary_scaled": (primary / columns).ravel(),
                "secondary_scaled": (secondary / columns).ravel(),
            }
            normalized = primary / field_scales[None, :]
            cell_norms = np.linalg.norm(normalized, axis=1)
            endpoint = float(
                max(cell_norms[0], cell_norms[-1])
                / max(float(np.max(cell_norms)), np.finfo(float).tiny)
            )
            maximum_endpoint = max(maximum_endpoint, endpoint)
            if label == REFERENCE_LABEL:
                arrays[f"{name}__N128_primary_physical"] = primary
                arrays[f"{name}__N128_secondary_physical"] = secondary

    spacing = float(
        np.mean(np.diff(np.log(reference_grid.edges)))
    )
    spectrum_reports = {}
    for name, definition in PROBE_DEFINITIONS.items():
        values = directions[REFERENCE_LABEL][name]["primary_physical"]
        secondary_values = directions[REFERENCE_LABEL][name][
            "secondary_physical"
        ]
        primary_spectrum = causal_packet_spectrum(
            values / field_scales[None, :],
            spacing,
            quantile=SPECTRAL_ENERGY_QUANTILE,
        )
        secondary_spectrum = causal_packet_spectrum(
            secondary_values / field_scales[None, :],
            spacing,
            quantile=SPECTRAL_ENERGY_QUANTILE,
        )
        theta = (
            primary_spectrum.quantile_angular_wavenumber * spacing
        )
        theta_secondary = (
            secondary_spectrum.quantile_angular_wavenumber * spacing
        )
        binding = definition["role"] == "binding_theta20_window"
        spectral_pass = bool(
            primary_spectrum.nyquist_alias_fraction
            <= MAXIMUM_ALIAS_FRACTION
            and (
                (
                    MINIMUM_USABLE_THETA
                    <= theta
                    <= MAXIMUM_BINDING_THETA
                )
                if binding
                else theta < MINIMUM_USABLE_THETA
            )
        )
        spectrum_reports[name] = {
            "role": definition["role"],
            "window_power": int(definition["window_power"]),
            "family": definition["family"],
            "theta_99": theta,
            "secondary_projection_theta_99": theta_secondary,
            "theta_projection_difference": abs(
                theta - theta_secondary
            ),
            "nyquist_alias_fraction": (
                primary_spectrum.nyquist_alias_fraction
            ),
            "spectral_passed": spectral_pass,
        }
        arrays[f"{name}__spectrum_theta"] = (
            primary_spectrum.angular_wavenumbers * spacing
        )
        arrays[f"{name}__spectrum_energy"] = (
            primary_spectrum.spectral_energy
        )
        arrays[f"{name}__spectrum_cumulative"] = (
            primary_spectrum.cumulative_energy_fraction
        )
    report = {
        "definitions": PROBE_DEFINITIONS,
        "spectrum_reports": spectrum_reports,
        "maximum_endpoint_cell_fraction": maximum_endpoint,
        "all_spectra_passed": all(
            item["spectral_passed"]
            for item in spectrum_reports.values()
        ),
        "passed": bool(
            maximum_endpoint <= MAXIMUM_ENDPOINT_CELL_FRACTION
            and all(
                item["spectral_passed"]
                for item in spectrum_reports.values()
            )
        ),
    }
    return directions, report, arrays


def _propagate(
    configurations: dict,
    tangents: dict,
    observable_maps: dict,
    directions: dict,
) -> tuple[dict, dict]:
    times = np.linspace(0.0, TIME_HORIZON_S, TIME_SAMPLE_COUNT)
    propagated = {}
    report = {}
    for label in LABELS:
        print(f"WP10c9d6c6a2: propagate {label}", flush=True)
        configuration = configurations[label]
        tangent = tangents[label]
        columns = np.asarray(
            configuration["primitive_column_scales"],
            dtype=float,
        ).ravel()
        primary = np.column_stack(
            [
                directions[label][name]["primary_scaled"]
                for name in PROBE_NAMES
            ]
        )
        secondary = np.column_stack(
            [
                directions[label][name]["secondary_scaled"]
                for name in PROBE_NAMES
            ]
        )
        initial = np.column_stack((primary, secondary))
        generator = np.asarray(
            tangent.scaled_generator_per_s,
            dtype=float,
        )
        trace = float(np.trace(generator))
        scaled = np.asarray(
            expm_multiply(
                generator,
                initial,
                start=0.0,
                stop=TIME_HORIZON_S,
                num=TIME_SAMPLE_COUNT,
                endpoint=True,
                traceA=trace,
            ),
            dtype=float,
        )
        half = np.asarray(
            expm_multiply(
                0.5 * TIME_HORIZON_S * generator,
                primary,
                traceA=0.5 * TIME_HORIZON_S * trace,
            ),
            dtype=float,
        )
        restarted = np.asarray(
            expm_multiply(
                0.5 * TIME_HORIZON_S * generator,
                half,
                traceA=0.5 * TIME_HORIZON_S * trace,
            ),
            dtype=float,
        )
        primary_scaled = scaled[:, :, : len(PROBE_NAMES)]
        secondary_scaled = scaled[:, :, len(PROBE_NAMES) :]
        cells = configuration["context"].grid.centers.size
        primary_physical = np.transpose(
            primary_scaled * columns[None, :, None],
            (2, 0, 1),
        ).reshape(len(PROBE_NAMES), TIME_SAMPLE_COUNT, cells, 5)
        secondary_physical = np.transpose(
            secondary_scaled * columns[None, :, None],
            (2, 0, 1),
        ).reshape(len(PROBE_NAMES), TIME_SAMPLE_COUNT, cells, 5)
        restart_physical = np.transpose(
            restarted * columns[:, None],
            (1, 0),
        ).reshape(len(PROBE_NAMES), cells, 5)
        observable = np.asarray(observable_maps[label], dtype=float)
        signals = np.einsum(
            "tnp,on->pto",
            primary_scaled,
            observable,
        )
        delta = primary_scaled[-1] - primary
        integrated = np.linalg.solve(generator, delta)
        solve_residual = generator @ integrated - delta
        correction = np.linalg.solve(generator, solve_residual)
        refined_integrated = integrated - correction
        boundary_integrals = np.transpose(
            observable @ refined_integrated,
            (1, 0),
        )
        boundary_integral_corrections = np.transpose(
            observable @ (refined_integrated - integrated),
            (1, 0),
        )
        residual_scale = np.maximum(
            np.linalg.norm(delta, axis=0),
            np.finfo(float).tiny,
        )
        solve_residuals = (
            np.linalg.norm(
                generator @ refined_integrated - delta,
                axis=0,
            )
            / residual_scale
        )
        propagated[label] = {
            "times": times,
            "primary_physical": primary_physical,
            "secondary_physical": secondary_physical,
            "restart_physical": restart_physical,
            "signals": signals,
            "boundary_integrals": boundary_integrals,
            "boundary_integral_corrections": (
                boundary_integral_corrections
            ),
            "boundary_integral_solve_residuals": solve_residuals,
        }
        report[label] = {
            "cell_count": int(cells),
            "probe_count": len(PROBE_NAMES),
            "maximum_boundary_integral_solve_residual": float(
                np.max(solve_residuals)
            ),
        }
    return propagated, report


def _history_norm(
    values: np.ndarray,
    *,
    measures: np.ndarray,
    field_scales: np.ndarray,
    times: np.ndarray,
) -> float:
    return causal_field_history_norm(
        values,
        cell_measures=measures,
        field_scales=field_scales,
        time_weights=causal_trapezoid_weights(times),
    )


def _cumulative(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    result = np.zeros_like(data)
    increments = np.diff(np.asarray(times, dtype=float))
    result[1:] = np.cumsum(
        0.5 * increments[:, None] * (data[1:] + data[:-1]),
        axis=0,
    )
    return result


def _observable_history_norm(
    values: np.ndarray,
    scales: np.ndarray,
    times: np.ndarray,
) -> float:
    normalized = np.asarray(values, dtype=float) / np.asarray(
        scales,
        dtype=float,
    )[None, :]
    weights = causal_trapezoid_weights(times)
    return float(
        np.sqrt(
            np.einsum("to,to,t->", normalized, normalized, weights)
        )
    )


def _comparison_report(
    configurations: dict,
    construction_arrays: dict[str, np.ndarray],
    directions: dict,
    probe_construction: dict,
    propagated: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    times = np.asarray(propagated[REFERENCE_LABEL]["times"], dtype=float)
    field_scales = np.asarray(
        construction_arrays["continuum_perturbation_field_scales"],
        dtype=float,
    )
    with np.load(C3_ARRAYS, allow_pickle=False) as source:
        observable_scales = np.asarray(
            source["fixed_physical_observable_scales"],
            dtype=float,
        )
    coarse_grid = configurations[LABELS[0]]["context"].grid
    medium_grid = configurations[LABELS[1]]["context"].grid
    fine_grid = configurations[LABELS[2]]["context"].grid
    coarse_measures = np.asarray(coarse_grid.cell_measures, dtype=float)
    medium_measures = np.asarray(medium_grid.cell_measures, dtype=float)
    fine_measures = np.asarray(fine_grid.cell_measures, dtype=float)

    coarse = propagated[LABELS[0]]["primary_physical"]
    medium = causal_restrict_proper_cell_averages(
        propagated[LABELS[1]]["primary_physical"],
        medium_measures,
        refinement_factor=2,
    )
    fine = causal_restrict_proper_cell_averages(
        propagated[LABELS[2]]["primary_physical"],
        fine_measures,
        refinement_factor=4,
    )
    fine_secondary = causal_restrict_proper_cell_averages(
        propagated[LABELS[2]]["secondary_physical"],
        fine_measures,
        refinement_factor=4,
    )
    fine_restart = causal_restrict_proper_cell_averages(
        propagated[LABELS[2]]["restart_physical"],
        fine_measures,
        refinement_factor=4,
    )

    reports = {}
    arrays: dict[str, np.ndarray] = {}
    minimum_binding_theta = float("inf")
    maximum_binding_theta = 0.0
    for probe_index, name in enumerate(PROBE_NAMES):
        richardson = causal_windowed_richardson_reference(
            coarse[probe_index],
            medium[probe_index],
            fine[probe_index],
            times=times,
            coarse_cell_measures=coarse_measures,
            field_scales=field_scales,
        )
        fine_difference = max(
            richardson.medium_fine_history_norm,
            np.finfo(float).tiny,
        )
        projection_ratio = (
            _history_norm(
                fine[probe_index] - fine_secondary[probe_index],
                measures=coarse_measures,
                field_scales=field_scales,
                times=times,
            )
            / fine_difference
        )
        restart_history = np.stack(
            (
                fine_restart[probe_index],
                fine_restart[probe_index],
            ),
            axis=0,
        )
        direct_history = np.stack(
            (
                fine[probe_index, -1],
                fine[probe_index, -1],
            ),
            axis=0,
        )
        restart_ratio = (
            causal_field_history_norm(
                restart_history - direct_history,
                cell_measures=coarse_measures,
                field_scales=field_scales,
                time_weights=np.ones(2),
            )
            / fine_difference
        )

        medium_signal = propagated[LABELS[1]]["signals"][
            probe_index, :, :6
        ]
        fine_signal = propagated[LABELS[2]]["signals"][
            probe_index, :, :6
        ]
        boundary_scales = observable_scales[:6]
        medium_boundary_integral = propagated[LABELS[1]][
            "boundary_integrals"
        ][probe_index, :6]
        fine_boundary_integral = propagated[LABELS[2]][
            "boundary_integrals"
        ][probe_index, :6]
        fine_boundary_difference = max(
            float(
                np.linalg.norm(
                    (medium_boundary_integral - fine_boundary_integral)
                    / boundary_scales
                )
            ),
            np.finfo(float).tiny,
        )
        boundary_integral_uncertainty = float(
            np.linalg.norm(
                propagated[LABELS[2]][
                    "boundary_integral_corrections"
                ][probe_index, :6]
                / boundary_scales
            )
        )
        boundary_ratio = (
            boundary_integral_uncertainty / fine_boundary_difference
        )
        stride_times = times[::2]
        boundary_quadrature = float(
            np.linalg.norm(
                (
                    _cumulative(times, fine_signal)[-1]
                    - _cumulative(
                        stride_times,
                        fine_signal[::2],
                    )[-1]
                )
                / boundary_scales
            )
        )
        trapezoid_boundary_ratio = (
            boundary_quadrature / fine_boundary_difference
        )
        boundary_history_difference = _observable_history_norm(
            medium_signal - fine_signal,
            boundary_scales,
            times,
        )

        spectrum = probe_construction["spectrum_reports"][name]
        binding = (
            PROBE_DEFINITIONS[name]["role"]
            == "binding_theta20_window"
        )
        if binding:
            minimum_binding_theta = min(
                minimum_binding_theta,
                float(spectrum["theta_99"]),
            )
            maximum_binding_theta = max(
                maximum_binding_theta,
                float(spectrum["theta_99"]),
            )
        passed = bool(
            spectrum["spectral_passed"]
            and richardson.observed_order >= MINIMUM_CROSS_GRID_ORDER
            and richardson.minimum_significant_component_order
            >= MINIMUM_COMPONENT_ORDER
            and richardson.refinement_error_cosine
            >= MINIMUM_REFINEMENT_ERROR_COSINE
            and richardson.maximum_coarse_reference_relative_error
            <= MAXIMUM_COMPLETE_SEMIGROUP_ERROR
            and richardson.reference_choice_to_fine_difference_ratio
            <= MAXIMUM_UNCERTAINTY_TO_FINE_DIFFERENCE_RATIO
            and projection_ratio
            <= MAXIMUM_UNCERTAINTY_TO_FINE_DIFFERENCE_RATIO
            and restart_ratio
            <= MAXIMUM_RESTART_TO_FINE_DIFFERENCE_RATIO
            and boundary_ratio
            <= (
                MAXIMUM_BOUNDARY_INTEGRAL_UNCERTAINTY_TO_FINE_DIFFERENCE_RATIO
            )
        )
        reports[name] = {
            "role": PROBE_DEFINITIONS[name]["role"],
            "family": PROBE_DEFINITIONS[name]["family"],
            "theta_99": spectrum["theta_99"],
            "nyquist_alias_fraction": (
                spectrum["nyquist_alias_fraction"]
            ),
            "observed_order": richardson.observed_order,
            "minimum_significant_component_order": (
                richardson.minimum_significant_component_order
            ),
            "refinement_error_cosine": (
                richardson.refinement_error_cosine
            ),
            "coarse_medium_history_norm": (
                richardson.coarse_medium_history_norm
            ),
            "medium_fine_history_norm": (
                richardson.medium_fine_history_norm
            ),
            "maximum_coarse_reference_relative_error": (
                richardson.maximum_coarse_reference_relative_error
            ),
            "history_coarse_reference_relative_error": (
                richardson.history_coarse_reference_relative_error
            ),
            "reference_choice_to_fine_difference_ratio": (
                richardson.reference_choice_to_fine_difference_ratio
            ),
            "window_projection_to_fine_difference_ratio": (
                projection_ratio
            ),
            "restart_to_fine_difference_ratio": restart_ratio,
            "boundary_integral_uncertainty_to_fine_difference_ratio": (
                boundary_ratio
            ),
            "diagnostic_trapezoid_to_fine_difference_ratio": (
                trapezoid_boundary_ratio
            ),
            "boundary_integral_solve_residual": float(
                propagated[LABELS[2]][
                    "boundary_integral_solve_residuals"
                ][probe_index]
            ),
            "boundary_history_fine_difference": (
                boundary_history_difference
            ),
            "passed": passed,
        }
        initial_norm = max(
            _history_norm(
                np.broadcast_to(
                    fine[probe_index, 0:1],
                    fine[probe_index].shape,
                ),
                measures=coarse_measures,
                field_scales=field_scales,
                times=times,
            ),
            np.finfo(float).tiny,
        )
        time_errors = np.empty(times.size, dtype=float)
        for time_index in range(times.size):
            duplicated = np.stack(
                (
                    coarse[probe_index, time_index]
                    - richardson.observed_reference[time_index],
                )
                * 2,
                axis=0,
            )
            time_errors[time_index] = (
                causal_field_history_norm(
                    duplicated,
                    cell_measures=coarse_measures,
                    field_scales=field_scales,
                    time_weights=np.ones(2),
                )
                / initial_norm
            )
        arrays[f"{name}__time_relative_error"] = time_errors
        arrays[f"{name}__coarse_medium_time_norm"] = np.asarray(
            [
                _history_norm(
                    np.stack((value, value), axis=0),
                    measures=coarse_measures,
                    field_scales=field_scales,
                    times=np.asarray((0.0, 1.0)),
                )
                for value in (
                    coarse[probe_index] - medium[probe_index]
                )
            ]
        )
        arrays[f"{name}__medium_fine_time_norm"] = np.asarray(
            [
                _history_norm(
                    np.stack((value, value), axis=0),
                    measures=coarse_measures,
                    field_scales=field_scales,
                    times=np.asarray((0.0, 1.0)),
                )
                for value in (
                    medium[probe_index] - fine[probe_index]
                )
            ]
        )
        arrays[f"{name}__final_observed_reference"] = (
            richardson.observed_reference[-1]
        )
    report = {
        "probe_reports": reports,
        "minimum_binding_theta_99": minimum_binding_theta,
        "maximum_binding_theta_99": maximum_binding_theta,
        "all_probes_passed": all(
            item["passed"] for item in reports.values()
        ),
        "binding_range_reaches_theta20": bool(
            minimum_binding_theta >= MINIMUM_USABLE_THETA
        ),
        "passed": bool(
            all(item["passed"] for item in reports.values())
            and minimum_binding_theta >= MINIMUM_USABLE_THETA
        ),
    }
    arrays["times"] = times
    return report, arrays


def _config() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "operator_change": False,
        "production_defaults_changed": False,
        "labels": LABELS,
        "time_horizon_s": TIME_HORIZON_S,
        "time_sample_count": TIME_SAMPLE_COUNT,
        "projection_orders": (
            PRIMARY_PROJECTION_ORDER,
            SECONDARY_PROJECTION_ORDER,
        ),
        "characteristic_field_nodes": CHARACTERISTIC_FIELD_NODES,
        "probe_definitions": PROBE_DEFINITIONS,
        "probe_amplitude": PROBE_AMPLITUDE,
        "spectral_energy_quantile": SPECTRAL_ENERGY_QUANTILE,
        "gates": {
            "maximum_complete_semigroup_error": (
                MAXIMUM_COMPLETE_SEMIGROUP_ERROR
            ),
            "minimum_usable_theta": MINIMUM_USABLE_THETA,
            "maximum_binding_theta": MAXIMUM_BINDING_THETA,
            "maximum_alias_fraction": MAXIMUM_ALIAS_FRACTION,
            "minimum_cross_grid_order": MINIMUM_CROSS_GRID_ORDER,
            "minimum_component_order": MINIMUM_COMPONENT_ORDER,
            "minimum_refinement_error_cosine": (
                MINIMUM_REFINEMENT_ERROR_COSINE
            ),
            "maximum_uncertainty_to_fine_difference_ratio": (
                MAXIMUM_UNCERTAINTY_TO_FINE_DIFFERENCE_RATIO
            ),
            "maximum_endpoint_cell_fraction": (
                MAXIMUM_ENDPOINT_CELL_FRACTION
            ),
            "minimum_characteristic_field_overlap": (
                MINIMUM_CHARACTERISTIC_FIELD_OVERLAP
            ),
            "maximum_characteristic_norm_defect": (
                MAXIMUM_CHARACTERISTIC_NORM_DEFECT
            ),
            "maximum_restart_to_fine_difference_ratio": (
                MAXIMUM_RESTART_TO_FINE_DIFFERENCE_RATIO
            ),
            "maximum_boundary_integral_uncertainty_to_fine_"
            "difference_ratio": (
                MAXIMUM_BOUNDARY_INTEGRAL_UNCERTAINTY_TO_FINE_DIFFERENCE_RATIO
            ),
        },
    }


def run(*, probe_preflight_only: bool = False) -> dict:
    started = time.perf_counter()
    identity = _validate_analyzed_git_identity()
    parent, failed_c6a = _load_parent()
    initial_trapezoid_summary = None
    if TRAPEZOID_PREFLIGHT_SUMMARY_PATH.exists():
        initial_trapezoid_summary = json.loads(
            TRAPEZOID_PREFLIGHT_SUMMARY_PATH.read_text(
                encoding="utf-8"
            )
        )
    elif SUMMARY_PATH.exists():
        candidate = json.loads(
            SUMMARY_PATH.read_text(encoding="utf-8")
        )
        if candidate.get("classification") == (
            "variable_coefficient_windowed_contract_failed_"
            "packet_manifest_blocked"
        ):
            initial_trapezoid_summary = candidate
            _write_json(
                TRAPEZOID_PREFLIGHT_SUMMARY_PATH,
                candidate,
            )
    configurations, construction_arrays, construction = (
        c3._build_continuum_configurations()
    )
    if not construction["passed"]:
        raise RuntimeError("c6a2 continuum construction failed")
    interpolator, characteristic_report, characteristic_arrays = (
        _build_characteristic_interpolator(
            configurations,
            construction_arrays,
        )
    )
    directions, probe_construction, probe_arrays = _build_probes(
        configurations,
        construction_arrays,
        interpolator,
    )
    if probe_preflight_only:
        payload = {
            "characteristic_report": characteristic_report,
            "probe_construction": probe_construction,
        }
        print(json.dumps(_plain(payload), indent=2, sort_keys=True))
        return payload

    print("WP10c9d6c6a2: build unchanged monolithic tangents", flush=True)
    tangents, observable_maps, method_reports, _baselines = (
        c3._build_tangents(configurations, construction_arrays)
    )
    method_passed = bool(
        characteristic_report["passed"]
        and probe_construction["passed"]
        and all(
            method_reports[label]["passed"]
            for label in LABELS
        )
    )
    propagated, propagation_report = _propagate(
        configurations,
        tangents,
        observable_maps,
        directions,
    )
    comparison, comparison_arrays = _comparison_report(
        configurations,
        construction_arrays,
        directions,
        probe_construction,
        propagated,
    )
    if not method_passed:
        classification = "variable_coefficient_windowed_contract_method_failed"
        authorized_next = "none"
    elif not comparison["passed"]:
        classification = (
            "variable_coefficient_windowed_contract_failed_"
            "packet_manifest_blocked"
        )
        authorized_next = "none"
    else:
        classification = (
            "variable_coefficient_windowed_contract_certified_"
            "packet_manifest_authorized"
        )
        authorized_next = "WP10c9d6c6b_packet_definition_manifest_only"
    passed = bool(authorized_next != "none")

    arrays = {
        **characteristic_arrays,
        **probe_arrays,
        **comparison_arrays,
        "field_scales": np.asarray(
            construction_arrays[
                "continuum_perturbation_field_scales"
            ],
            dtype=float,
        ),
    }
    source_hashes, source_manifest = _source_manifest()
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DECISIVE_ARRAYS, **arrays)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        **identity,
        "classification": classification,
        "passed": passed,
        "authorized_next": authorized_next,
        "audit_executed": True,
        "operator_changed": False,
        "parent_classification": parent["classification"],
        "parent_classification_preserved": True,
        "c6a_classification": failed_c6a["classification"],
        "c6a_rejection_preserved": True,
        "parent_packet_contract_error_preserved": (
            MAXIMUM_COMPLETE_SEMIGROUP_ERROR
            == c6a.MAXIMUM_COMPLETE_SEMIGROUP_ERROR
        ),
        "parent_minimum_usable_theta_preserved": (
            MINIMUM_USABLE_THETA == c6a.MINIMUM_CERTIFIED_THETA
        ),
        "configuration": _config(),
        "continuum_construction": construction,
        "characteristic_field_report": characteristic_report,
        "probe_construction_report": probe_construction,
        "method_reports": {
            label: method_reports[label] for label in LABELS
        },
        "method_passed": method_passed,
        "propagation_report": propagation_report,
        "comparison_report": comparison,
        "initial_trapezoid_preflight_report": (
            initial_trapezoid_summary
        ),
        "boundary_integration_method_correction": {
            "initial_method": "65_point_composite_trapezoid",
            "corrected_method": (
                "exact_linear_semigroup_integral_with_"
                "one_step_iterative_refinement"
            ),
            "physical_probes_changed": False,
            "window_definitions_changed": False,
            "tangents_changed": False,
            "scientific_gates_changed": False,
            "initial_summary_path": (
                str(TRAPEZOID_PREFLIGHT_SUMMARY_PATH.relative_to(ROOT))
                if initial_trapezoid_summary is not None
                else None
            ),
        },
        "windowed_contract_certified": comparison["passed"],
        "prospective_packet_manifest_authorized": passed,
        "uniform_packet_propagation_authorized": False,
        "embedded_export_discrimination_authorized": False,
        "nonlinear_physical_trajectory_authorized": False,
        "production_operator_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "decisive_arrays_path": str(
            DECISIVE_ARRAYS.relative_to(ROOT)
        ),
        "decisive_arrays_sha256": _sha256(DECISIVE_ARRAYS),
        "decisive_array_hashes": {
            name: _array_sha256(values)
            for name, values in sorted(arrays.items())
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "scientific_status": "DIAGNOSTIC ONLY" if passed else "REJECTED",
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        **identity,
        "implementation_base_tree": _git_value(
            "rev-parse",
            "HEAD^{tree}",
        ),
        "working_tree_status": _git_value("status", "--short"),
        "command": (
            "PYTHONPATH=src:scripts python "
            "scripts/run_causal_inner_windowed_contract_"
            "wp10c9d6c6a2.py"
        ),
        "environment": _environment(),
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "parent_canonical_hashes": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (
                PARENT_CONFIG,
                PARENT_SUMMARY,
                PARENT_ARRAYS,
                PARENT_PROVENANCE,
            )
        },
    }
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
                    "minimum_binding_theta_99": comparison[
                        "minimum_binding_theta_99"
                    ],
                    "maximum_binding_theta_99": comparison[
                        "maximum_binding_theta_99"
                    ],
                    "failed_probes": [
                        name
                        for name, report in comparison[
                            "probe_reports"
                        ].items()
                        if not report["passed"]
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
        raise RuntimeError("c6a2 canonical evidence is unavailable")
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
    provenance = json.loads(
        PROVENANCE_PATH.read_text(encoding="utf-8")
    )
    provenance.update(
        {
            "scientific_status": (
                "DIAGNOSTIC ONLY" if summary["passed"] else "REJECTED"
            ),
            "working_tree_status": _git_value("status", "--short"),
            "environment": _environment(),
            "implementation_source_hashes": source_hashes,
            "implementation_source_manifest_sha256": source_manifest,
        }
    )
    _write_json(CONFIG_PATH, _config())
    _write_json(SUMMARY_PATH, summary)
    _write_json(PROVENANCE_PATH, provenance)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    _refresh_canonical_catalog()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-preflight-only", action="store_true")
    parser.add_argument("--refresh-metadata-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.probe_preflight_only and arguments.refresh_metadata_only:
        raise ValueError("select only one c6a2 runner action")
    if arguments.refresh_metadata_only:
        refresh_metadata_only()
    else:
        run(probe_preflight_only=arguments.probe_preflight_only)


if __name__ == "__main__":
    main()
