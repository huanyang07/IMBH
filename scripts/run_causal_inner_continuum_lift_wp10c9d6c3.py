#!/usr/bin/env python3
"""Audit continuum-lift and metric conditioning without changing the operator.

WP10c9d6c2 rejected the strict four-level asymptotic-direction contract, but
its two N512 continuations differed by an amount comparable with the binding
N256/N512 maximum export difference.  This package preserves that rejection
and changes no physical or numerical operator.  It replaces discrete-grid
continuation by:

1. one C4 continuum background fitted to the N128 finite-volume cell
   averages and constrained exactly by the existing physical boundary
   anchors;
2. analytic mixed-field perturbations projected as exact Kerr--Schild
   measure-weighted cell averages on N64/N128/N256/N512; and
3. an independent projection-order uncertainty check.

The historical convergence gates remain binding and are reported unchanged.
Additional time-weighted norms, peak locations, error-history SVDs, and fixed
power continuum fits diagnose whether the strict direction failure is a
physical nonconvergence or a small multi-mode/peak-switching effect.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
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
from scipy.interpolate import BSpline


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_causal_inner_monolithic_anchor_audit_wp10c9d6c1 as wp10c9d6c1
import run_causal_inner_monolithic_four_level_wp10c9d6c2 as wp10c9d6c2
import run_causal_inner_monolithic_uniform_exports_wp10c9d6c as wp10c9d6c

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    causal_five_field_monolithic_frozen_tangent,
    causal_five_field_reconstruct_face_charts,
    evaluate_causal_five_field_monolithic_backward_euler,
    kerr_schild_column_geometry,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c3"
ANALYZED_BASE_COMMIT = "da2d7612cc9a2fff7093bee705f3f5fbe2d2101d"
ANALYZED_BASE_PARENT = "c28dd65708ee817fd23d0c619dbb0afd5f991178"
ANALYZED_BASE_TREE = "7259a115a04458e745f00db77dd2500f0ffb7f55"
THIS_RUNNER = (
    "scripts/run_causal_inner_continuum_lift_wp10c9d6c3.py"
)

MESHES = tuple(wp10c9d6c2.MESHES)
LABELS = tuple(wp10c9d6c2.LABELS)
ACTIVE_CELLS = (24, 48, 96, 192)
REFERENCE_LABEL = "uniform_N128"
PRIMARY_PROJECTION_ORDER = 24
SECONDARY_PROJECTION_ORDER = 12
BACKGROUND_DEGREES = (5, 7)
PRIMARY_BACKGROUND_DEGREE = BACKGROUND_DEGREES[0]
BACKGROUND_COEFFICIENT_COUNT = 24
PERTURBATIONS = (
    "calibration_mixed",
    "heldout_near_excision",
)
PRIMARY_STRIDE = 2
STRIDE_AUDITS = (1, 2, 4)

MINIMUM_EXPORT_ORDER = wp10c9d6c2.MINIMUM_EXPORT_ORDER
MAXIMUM_FINE_PHYSICAL_DIFFERENCE = (
    wp10c9d6c2.MAXIMUM_FINE_PHYSICAL_DIFFERENCE
)
MINIMUM_HISTORY_COSINE = wp10c9d6c2.MINIMUM_HISTORY_COSINE
MINIMUM_ERROR_COSINE = wp10c9d6c2.MINIMUM_ERROR_COSINE
MINIMUM_RELATIVE_ACTIVITY = wp10c9d6c2.MINIMUM_RELATIVE_ACTIVITY
MAXIMUM_RESTART_DEFECT = wp10c9d6c2.MAXIMUM_RESTART_DEFECT

MAXIMUM_BACKGROUND_CONSTRAINT_DEFECT = 1.0e-4
MAXIMUM_BACKGROUND_BOUNDARY_DEFECT = 1.0e-12
MAXIMUM_BACKGROUND_REPRESENTATION_DIFFERENCE = 1.0e-3
MAXIMUM_RECONSTRUCTION_FACTOR_CHANGE = 0.0
MAXIMUM_LIFT_STATE_RELATIVE_DIFFERENCE = 1.0e-9
MAXIMUM_LIFT_RATE_RELATIVE_DIFFERENCE = 1.0e-8
MAXIMUM_LIFT_TO_FINE_EXPORT_RATIO = 0.10
LIFT_EXPORT_COMPARISON_FLOOR = 1.0e-10
MINIMUM_TWO_MODE_EXPLAINED_FRACTION = 0.95
MINIMUM_SECOND_MODE_FRACTION = 0.05
MAXIMUM_CONTINUUM_MODEL_DISAGREEMENT = 0.05

PARENT_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_monolithic_four_level_wp10c9d6c2"
)
PARENT_SUMMARY = PARENT_DIRECTORY / "summary.json"
PARENT_ARRAYS = PARENT_DIRECTORY / "decisive_arrays.npz"
UNIFORM_REPLAY_CONTEXTS = (
    ROOT
    / "results/canonical/"
    "causal_inner_monolithic_uniform_exports_wp10c9d6c/"
    "replay_contexts.json"
)
UNIFORM_REPLAY_INPUTS = (
    ROOT
    / "results/canonical/"
    "causal_inner_monolithic_uniform_exports_wp10c9d6c/"
    "replay_inputs.npz"
)
CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_continuum_lift_wp10c9d6c3"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"

IMPLEMENTATION_SOURCES = (
    THIS_RUNNER,
    "scripts/run_causal_inner_monolithic_four_level_wp10c9d6c2.py",
    "scripts/run_causal_inner_monolithic_anchor_audit_wp10c9d6c1.py",
    "scripts/run_causal_inner_monolithic_uniform_exports_wp10c9d6c.py",
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_dae.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_monolithic_tangent.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_radial_linear_tangent.py",
    "tests/test_causal_inner_continuum_lift_wp10c9d6c3.py",
)

OBSERVABLE_NAMES = tuple(wp10c9d6c2.OBSERVABLE_NAMES)
CONSERVATIVE_OBSERVABLES = tuple(range(9))
DISTRIBUTED_OBSERVABLES = tuple(range(9, 13))


@dataclass(frozen=True)
class SmoothCellAverageProfile:
    """One vector-valued B-spline continuum profile in log radius."""

    knots: np.ndarray
    coefficients: np.ndarray
    degree: int
    gravitational_radius: float

    def evaluate(self, radii: np.ndarray) -> np.ndarray:
        values = np.asarray(radii, dtype=float)
        if values.ndim != 1 or np.any(values <= 0.0):
            raise ValueError("profile radii must be a positive vector")
        spline = BSpline(
            np.asarray(self.knots, dtype=float),
            np.asarray(self.coefficients, dtype=float),
            int(self.degree),
            extrapolate=False,
        )
        result = np.asarray(spline(np.log(values)), dtype=float)
        if result.shape != (values.size, 5) or np.any(~np.isfinite(result)):
            raise ValueError("continuum profile evaluation failed")
        return result


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


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    return wp10c9d6c2._cosine(first, second)


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
        raise RuntimeError("WP10c9d6c3 analyzed git identity changed")
    return {
        "analyzed_base_commit": resolved,
        "analyzed_base_parent_commit": parent,
        "analyzed_base_tree_sha": tree,
    }


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _load_parent() -> tuple[dict, dict[str, np.ndarray]]:
    summary = json.loads(PARENT_SUMMARY.read_text(encoding="utf-8"))
    if not (
        summary["work_package"] == "WP10c9d6c2"
        and summary["classification"]
        == "four_level_uniform_asymptotic_direction_rejected"
        and summary["method_passed"]
        and not summary["passed"]
        and not summary["embedded_export_discrimination_authorized"]
        and not summary["reduced_slow_evolution_authorized"]
    ):
        raise RuntimeError("WP10c9d6c2 binding classification changed")
    return summary, _load_npz(PARENT_ARRAYS)


def _open_uniform_knots(
    lower: float,
    upper: float,
    n_coefficients: int,
    degree: int,
) -> np.ndarray:
    interior_count = int(n_coefficients) - int(degree) - 1
    if interior_count < 1:
        raise ValueError("spline requires at least one interior knot")
    interior = np.linspace(
        float(lower),
        float(upper),
        interior_count + 2,
        dtype=float,
    )[1:-1]
    return np.concatenate(
        (
            np.full(int(degree) + 1, float(lower)),
            interior,
            np.full(int(degree) + 1, float(upper)),
        )
    )


def _cell_projection_matrix(
    grid,
    knots: np.ndarray,
    degree: int,
    *,
    quadrature_order: int,
) -> np.ndarray:
    """Map spline coefficients to proper-measure cell averages."""

    nodes, weights = np.polynomial.legendre.leggauss(
        int(quadrature_order)
    )
    result = np.empty(
        (grid.centers.size, len(knots) - int(degree) - 1),
        dtype=float,
    )
    for cell, (left, right) in enumerate(
        zip(grid.edges[:-1], grid.edges[1:], strict=True)
    ):
        lower = float(np.log(left))
        upper = float(np.log(right))
        midpoint = 0.5 * (lower + upper)
        half_width = 0.5 * (upper - lower)
        log_radii = midpoint + half_width * nodes
        radii = np.exp(log_radii)
        raw_weights = np.asarray(
            [
                half_width
                * weight
                * radius
                * kerr_schild_column_geometry(
                    float(radius),
                    grid.gravitational_radius,
                ).face_measure
                for radius, weight in zip(
                    radii,
                    weights,
                    strict=True,
                )
            ],
            dtype=float,
        )
        raw_weights *= (
            float(grid.cell_measures[cell])
            / float(np.sum(raw_weights))
        )
        basis = BSpline.design_matrix(
            log_radii,
            knots,
            int(degree),
            extrapolate=False,
        ).toarray()
        result[cell] = (
            raw_weights @ basis
            / float(grid.cell_measures[cell])
        )
    return result


def _fit_cell_average_profile(
    grid,
    cell_averages: np.ndarray,
    inner_anchor: np.ndarray,
    outer_anchor: np.ndarray,
    *,
    degree: int,
    quadrature_order: int = PRIMARY_PROJECTION_ORDER,
) -> tuple[SmoothCellAverageProfile, dict]:
    """Fit one C^(degree-1) spline to averages and boundary anchors."""

    values = np.asarray(cell_averages, dtype=float)
    inner = np.asarray(inner_anchor, dtype=float)
    outer = np.asarray(outer_anchor, dtype=float)
    if values.shape != (grid.centers.size, 5):
        raise ValueError("background cell averages have wrong shape")
    if inner.shape != (5,) or outer.shape != (5,):
        raise ValueError("background boundary anchors have wrong shape")
    n_coefficients = int(BACKGROUND_COEFFICIENT_COUNT)
    lower = float(np.log(grid.edges[0]))
    upper = float(np.log(grid.edges[-1]))
    knots = _open_uniform_knots(
        lower,
        upper,
        n_coefficients,
        int(degree),
    )
    cell_matrix = _cell_projection_matrix(
        grid,
        knots,
        int(degree),
        quadrature_order=int(quadrature_order),
    )
    boundary = BSpline.design_matrix(
        np.asarray([lower, upper]),
        knots,
        int(degree),
        extrapolate=False,
    ).toarray()
    targets = np.vstack((values, inner, outer))
    # Enforce both boundary anchors exactly and solve only the cell-average
    # rows in least squares.  The reduced representation avoids fitting a
    # grid-scale oscillation while preserving the physical edge data.
    _u, singular, right = np.linalg.svd(
        boundary,
        full_matrices=True,
    )
    rank = int(np.sum(singular > 1.0e-13 * singular[0]))
    if rank != 2:
        raise RuntimeError("continuum boundary constraint lost rank")
    null_basis = right[rank:].T
    boundary_gram = boundary @ boundary.T
    particular = boundary.T @ np.linalg.solve(
        boundary_gram,
        np.vstack((inner, outer)),
    )
    reduced = cell_matrix @ null_basis
    correction = np.linalg.lstsq(
        reduced,
        values - cell_matrix @ particular,
        rcond=None,
    )[0]
    coefficients = particular + null_basis @ correction
    profile = SmoothCellAverageProfile(
        knots=np.asarray(knots, dtype=float),
        coefficients=np.asarray(coefficients, dtype=float),
        degree=int(degree),
        gravitational_radius=float(grid.gravitational_radius),
    )
    cell_closure = cell_matrix @ coefficients - values
    boundary_closure = boundary @ coefficients - np.vstack(
        (inner, outer)
    )
    scales = np.maximum(
        np.max(np.abs(targets), axis=0),
        np.finfo(float).tiny,
    )
    return profile, {
        "degree": int(degree),
        "continuity_order": int(degree) - 1,
        "coefficient_count": n_coefficients,
        "cell_fit_matrix_condition": float(np.linalg.cond(reduced)),
        "maximum_scaled_constraint_defect": float(
            np.max(np.abs(cell_closure) / scales)
        ),
        "maximum_boundary_defect": float(
            np.max(np.abs(boundary_closure))
        ),
    }


def _project_callable_to_cells(
    grid,
    evaluator,
    *,
    quadrature_order: int,
) -> np.ndarray:
    """Project a pointwise five-field function to proper cell averages."""

    nodes, weights = np.polynomial.legendre.leggauss(
        int(quadrature_order)
    )
    result = np.empty((grid.centers.size, 5), dtype=float)
    for cell, (left, right) in enumerate(
        zip(grid.edges[:-1], grid.edges[1:], strict=True)
    ):
        lower = float(np.log(left))
        upper = float(np.log(right))
        midpoint = 0.5 * (lower + upper)
        half_width = 0.5 * (upper - lower)
        log_radii = midpoint + half_width * nodes
        radii = np.exp(log_radii)
        measure_weights = np.asarray(
            [
                half_width
                * weight
                * radius
                * kerr_schild_column_geometry(
                    float(radius),
                    grid.gravitational_radius,
                ).face_measure
                for radius, weight in zip(
                    radii,
                    weights,
                    strict=True,
                )
            ],
            dtype=float,
        )
        measure_weights *= (
            float(grid.cell_measures[cell])
            / float(np.sum(measure_weights))
        )
        point_values = np.asarray(evaluator(radii), dtype=float)
        if point_values.shape != (radii.size, 5):
            raise ValueError("continuum evaluator returned wrong shape")
        result[cell] = (
            measure_weights @ point_values
            / float(grid.cell_measures[cell])
        )
    return result


def _smooth_cutoff(log_radius: np.ndarray, outer_log_radius: float):
    """Return a C-infinity cutoff equal to one well inside the domain."""

    values = np.asarray(log_radius, dtype=float)
    start = float(np.log(7.0))
    stop = float(outer_log_radius)
    result = np.ones_like(values)
    result[values >= stop] = 0.0
    active = (values > start) & (values < stop)
    if np.any(active):
        coordinate = (values[active] - start) / (stop - start)
        left = np.exp(-1.0 / np.maximum(coordinate, 1.0e-300))
        right = np.exp(
            -1.0 / np.maximum(1.0 - coordinate, 1.0e-300)
        )
        result[active] = right / (left + right)
    return result


def _analytic_perturbation(
    name: str,
    radii: np.ndarray,
    *,
    gravitational_radius: float,
    field_scales: np.ndarray,
    outer_radius: float,
) -> np.ndarray:
    """Evaluate one declared grid-independent physical perturbation."""

    radius_over_rg = (
        np.asarray(radii, dtype=float) / float(gravitational_radius)
    )
    log_radius = np.log(radius_over_rg)
    cutoff = _smooth_cutoff(
        log_radius,
        float(np.log(outer_radius / gravitational_radius)),
    )
    scales = np.asarray(field_scales, dtype=float)
    if name == "calibration_mixed":
        inner = np.exp(
            -0.5
            * (
                np.log(radius_over_rg / 1.82)
                / 0.48
            )
            ** 2
        )
        middle = np.exp(
            -0.5
            * (
                np.log(radius_over_rg / 3.05)
                / 0.24
            )
            ** 2
        )
        dimensionless = (
            inner[:, None]
            * np.asarray(
                (0.015, -0.003, 0.010, 0.0010, 0.055),
                dtype=float,
            )[None, :]
            + middle[:, None]
            * np.asarray(
                (0.003, -0.008, 0.002, -0.0005, 0.040),
                dtype=float,
            )[None, :]
        )
    elif name == "heldout_near_excision":
        envelope = np.exp(
            -0.5
            * (
                np.log(radius_over_rg / 2.20)
                / 0.13
            )
            ** 2
        )
        dimensionless = (
            envelope[:, None]
            * np.asarray(
                (0.008, -0.012, 0.006, 0.002, 0.030),
                dtype=float,
            )[None, :]
        )
    else:
        raise KeyError(f"unknown analytic perturbation {name!r}")
    return cutoff[:, None] * dimensionless * scales[None, :]


def _build_continuum_configurations() -> tuple[
    dict,
    dict[str, np.ndarray],
    dict,
]:
    """Build all four grids from one smooth finite-volume continuum lift."""

    legacy, parent_decisive, parent_construction = (
        wp10c9d6c2._build_four_configurations()
    )
    if not parent_construction["passed"]:
        raise RuntimeError("WP10c9d6c2 configuration replay failed")
    reference = legacy[REFERENCE_LABEL]
    reference_grid = reference["context"].grid
    inner_anchor = np.asarray(
        parent_decisive["common_inner_boundary_anchor"],
        dtype=float,
    )
    outer_anchor = np.asarray(
        parent_decisive["common_outer_boundary_anchor"],
        dtype=float,
    )
    profiles = {}
    profile_reports = {}
    for degree in BACKGROUND_DEGREES:
        profile, report = _fit_cell_average_profile(
            reference_grid,
            reference["base_primitives"],
            inner_anchor,
            outer_anchor,
            degree=int(degree),
        )
        profiles[int(degree)] = profile
        profile_reports[str(degree)] = report

    primary_profile = profiles[PRIMARY_BACKGROUND_DEGREE]
    field_scales = wp10c9d6c1._field_scales(
        {label: legacy[label] for label in wp10c9d6c1.LABELS}
    )
    result = {}
    decisive = {
        "continuum_background_knots": primary_profile.knots,
        "continuum_background_coefficients": (
            primary_profile.coefficients
        ),
        "continuum_background_reference_cell_averages": np.asarray(
            reference["base_primitives"],
            dtype=float,
        ),
        "continuum_background_inner_anchor": inner_anchor,
        "continuum_background_outer_anchor": outer_anchor,
        "continuum_perturbation_field_scales": field_scales,
    }
    maximum_factor_change = 0.0
    representation_differences = {}
    for label in LABELS:
        legacy_configuration = legacy[label]
        grid = legacy_configuration["context"].grid
        base = _project_callable_to_cells(
            grid,
            primary_profile.evaluate,
            quadrature_order=PRIMARY_PROJECTION_ORDER,
        )
        secondary_base = _project_callable_to_cells(
            grid,
            profiles[BACKGROUND_DEGREES[1]].evaluate,
            quadrature_order=PRIMARY_PROJECTION_ORDER,
        )
        base_scale = np.maximum(
            np.max(np.abs(base), axis=0),
            np.finfo(float).tiny,
        )
        representation_differences[label] = float(
            np.max(np.abs(base - secondary_base) / base_scale)
        )
        boundary = primary_profile.evaluate(
            np.asarray([grid.edges[-1]], dtype=float)
        )[0]
        context = replace(
            legacy_configuration["context"],
            outer_boundary_frozen_exterior_chart=np.array(
                boundary,
                copy=True,
            ),
        ).validated()
        columns, rows = wp10c9d6c2._scales_for(context, base)
        reconstruction = causal_five_field_reconstruct_face_charts(
            context,
            base,
            purpose="flux",
        )
        maximum_factor_change = max(
            maximum_factor_change,
            float(
                np.max(
                    np.abs(
                        reconstruction.admissibility_factors - 1.0
                    )
                )
            ),
        )
        directions = {}
        physical_directions = {}
        for name in PERTURBATIONS:
            evaluator = lambda radii, profile_name=name: (
                _analytic_perturbation(
                    profile_name,
                    radii,
                    gravitational_radius=grid.gravitational_radius,
                    field_scales=field_scales,
                    outer_radius=float(grid.edges[-1]),
                )
            )
            primary_physical = _project_callable_to_cells(
                grid,
                evaluator,
                quadrature_order=PRIMARY_PROJECTION_ORDER,
            )
            secondary_physical = _project_callable_to_cells(
                grid,
                evaluator,
                quadrature_order=SECONDARY_PROJECTION_ORDER,
            )
            directions[name] = (
                primary_physical / columns.reshape(-1, 5)
            ).ravel()
            directions[name + "__projection_order_12"] = (
                secondary_physical / columns.reshape(-1, 5)
            ).ravel()
            physical_directions[name] = primary_physical
            physical_directions[
                name + "__projection_order_12"
            ] = secondary_physical
            decisive[f"{label}__{name}__physical_direction"] = (
                primary_physical
            )
            decisive[
                f"{label}__{name}__physical_direction_order_12"
            ] = secondary_physical
        directions["common_mode"] = np.array(
            directions["calibration_mixed"],
            copy=True,
        )
        result[label] = {
            **legacy_configuration,
            "context": context,
            "base_primitives": base,
            "primitive_column_scales": columns,
            "conservation_row_scales": rows,
            "initial_directions": directions,
            "physical_directions": physical_directions,
        }
        decisive[f"{label}__base_primitives"] = base
        decisive[f"{label}__primitive_column_scales"] = columns
        decisive[f"{label}__conservation_row_scales"] = rows

    reference_projection = result[REFERENCE_LABEL]["base_primitives"]
    reference_defect = _relative_difference(
        reference_projection,
        reference["base_primitives"],
    )
    maximum_constraint = max(
        report["maximum_scaled_constraint_defect"]
        for report in profile_reports.values()
    )
    maximum_boundary = max(
        report["maximum_boundary_defect"]
        for report in profile_reports.values()
    )
    maximum_representation = max(representation_differences.values())
    report = {
        "background_definition": (
            "C4 quintic B-spline in log radius least-squares fitted to "
            "the N128 proper-measure primitive cell averages and exactly "
            "constrained by the existing inner/outer boundary anchors"
        ),
        "state_semantics": "proper-measure finite-volume cell averages",
        "projection_orders": (
            PRIMARY_PROJECTION_ORDER,
            SECONDARY_PROJECTION_ORDER,
        ),
        "spline_reports": profile_reports,
        "representation_differences": representation_differences,
        "maximum_scaled_constraint_defect": maximum_constraint,
        "maximum_boundary_defect": maximum_boundary,
        "maximum_background_representation_difference": (
            maximum_representation
        ),
        "reference_cell_average_defect": reference_defect,
        "maximum_reconstruction_factor_change": maximum_factor_change,
        "passed": bool(
            maximum_constraint
            <= MAXIMUM_BACKGROUND_CONSTRAINT_DEFECT
            and maximum_boundary <= MAXIMUM_BACKGROUND_BOUNDARY_DEFECT
            and maximum_representation
            <= MAXIMUM_BACKGROUND_REPRESENTATION_DIFFERENCE
            and reference_defect <= MAXIMUM_BACKGROUND_CONSTRAINT_DEFECT
            and maximum_factor_change
            <= MAXIMUM_RECONSTRUCTION_FACTOR_CHANGE
        ),
    }
    return result, decisive, report


def _time_weights(times: np.ndarray) -> np.ndarray:
    values = np.asarray(times, dtype=float)
    if (
        values.ndim != 1
        or values.size < 2
        or np.any(np.diff(values) <= 0.0)
    ):
        raise ValueError("time samples must be strictly increasing")
    increments = np.diff(values)
    weights = np.empty_like(values)
    weights[0] = 0.5 * increments[0]
    weights[-1] = 0.5 * increments[-1]
    weights[1:-1] = 0.5 * (increments[:-1] + increments[1:])
    weights /= float(np.sum(weights))
    return weights


def _weighted_norm(values: np.ndarray, weights: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    time_weights = np.asarray(weights, dtype=float)
    if array.shape[0] != time_weights.size:
        raise ValueError("weighted history has incompatible time axis")
    return float(
        np.sqrt(
            np.sum(
                time_weights
                * np.sum(array.reshape(array.shape[0], -1) ** 2, axis=1)
            )
        )
    )


def _weighted_cosine(
    first: np.ndarray,
    second: np.ndarray,
    weights: np.ndarray,
) -> float:
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    time_weights = np.asarray(weights, dtype=float)
    if left.shape != right.shape or left.shape[0] != time_weights.size:
        raise ValueError("weighted cosine inputs are incompatible")
    product = float(
        np.sum(
            time_weights
            * np.sum(
                left.reshape(left.shape[0], -1)
                * right.reshape(right.shape[0], -1),
                axis=1,
            )
        )
    )
    denominator = _weighted_norm(left, time_weights) * _weighted_norm(
        right,
        time_weights,
    )
    if denominator <= np.finfo(float).tiny:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(product / denominator)


def _argmax_report(
    difference: np.ndarray,
    times: np.ndarray,
    active_indices: np.ndarray,
) -> dict:
    values = np.asarray(difference, dtype=float)
    position = np.unravel_index(
        int(np.argmax(np.abs(values))),
        values.shape,
    )
    component = int(active_indices[int(position[1])])
    return {
        "time_index": int(position[0]),
        "time_s": float(times[int(position[0])]),
        "component_index": component,
        "component": OBSERVABLE_NAMES[component],
        "signed_value": float(values[position]),
        "absolute_value": float(abs(values[position])),
    }


def _conditioned_metrics(
    histories: dict,
    scales: np.ndarray,
    *,
    labels: tuple[str, str, str],
    stride: int = PRIMARY_STRIDE,
) -> dict:
    indices = wp10c9d6c2._selected_indices(
        histories[labels[0]]["times"].size,
        int(stride),
    )
    times = np.asarray(
        histories[labels[0]]["times"],
        dtype=float,
    )[indices]
    normalized = {
        label: np.asarray(
            histories[label]["signals"],
            dtype=float,
        )[indices]
        / scales
        for label in labels
    }
    response = np.max(
        np.abs(np.asarray([normalized[label] for label in labels])),
        axis=(0, 1),
    )
    significant = response >= MINIMUM_RELATIVE_ACTIVITY
    if not np.any(significant):
        return {
            "passed": False,
            "reason": "no physically significant component",
        }
    active = np.flatnonzero(significant)
    coarse = normalized[labels[0]][:, significant]
    medium = normalized[labels[1]][:, significant]
    fine = normalized[labels[2]][:, significant]
    first = medium - coarse
    second = fine - medium
    weights = _time_weights(times)
    first_l2 = _weighted_norm(first, weights)
    second_l2 = _weighted_norm(second, weights)
    first_linf = float(np.max(np.abs(first)))
    second_linf = float(np.max(np.abs(second)))
    component_first_l2 = np.asarray(
        [
            _weighted_norm(first[:, position : position + 1], weights)
            for position in range(first.shape[1])
        ],
        dtype=float,
    )
    component_second_l2 = np.asarray(
        [
            _weighted_norm(second[:, position : position + 1], weights)
            for position in range(second.shape[1])
        ],
        dtype=float,
    )
    component_first_linf = np.max(np.abs(first), axis=0)
    component_second_linf = np.max(np.abs(second), axis=0)
    endpoint_first = np.abs(first[-1])
    endpoint_second = np.abs(second[-1])
    floor = np.finfo(float).tiny
    component_l2_orders = np.log2(
        np.maximum(component_first_l2, floor)
        / np.maximum(component_second_l2, floor)
    )
    component_linf_orders = np.log2(
        np.maximum(component_first_linf, floor)
        / np.maximum(component_second_linf, floor)
    )
    component_endpoint_orders = np.log2(
        np.maximum(endpoint_first, floor)
        / np.maximum(endpoint_second, floor)
    )
    first_peak = _argmax_report(first, times, active)
    second_peak = _argmax_report(second, times, active)
    peak_migrated = bool(
        first_peak["time_index"] != second_peak["time_index"]
        or first_peak["component_index"]
        != second_peak["component_index"]
    )
    active_lookup = {
        int(component): position
        for position, component in enumerate(active)
    }

    def sector_cosine(indices_to_use: tuple[int, ...]) -> float | None:
        positions = [
            active_lookup[index]
            for index in indices_to_use
            if index in active_lookup
        ]
        if not positions:
            return None
        return _weighted_cosine(
            first[:, positions],
            second[:, positions],
            weights,
        )

    l2_order = float(
        np.log2(
            max(first_l2, floor) / max(second_l2, floor)
        )
    )
    linf_order = float(
        np.log2(
            max(first_linf, floor) / max(second_linf, floor)
        )
    )
    norm_passed = bool(
        l2_order >= MINIMUM_EXPORT_ORDER
        and np.all(component_l2_orders >= MINIMUM_EXPORT_ORDER)
        and second_linf <= MAXIMUM_FINE_PHYSICAL_DIFFERENCE
        and _weighted_cosine(medium, fine, weights)
        >= MINIMUM_HISTORY_COSINE
    )
    return {
        "passed": norm_passed,
        "labels": labels,
        "stride": int(stride),
        "significant_components": [
            OBSERVABLE_NAMES[index] for index in active
        ],
        "weighted_l2_order": l2_order,
        "linfinity_order": linf_order,
        "component_weighted_l2_orders": {
            OBSERVABLE_NAMES[index]: float(component_l2_orders[position])
            for position, index in enumerate(active)
        },
        "component_linfinity_orders": {
            OBSERVABLE_NAMES[index]: float(
                component_linf_orders[position]
            )
            for position, index in enumerate(active)
        },
        "component_endpoint_orders": {
            OBSERVABLE_NAMES[index]: float(
                component_endpoint_orders[position]
            )
            for position, index in enumerate(active)
        },
        "minimum_component_weighted_l2_order": float(
            np.min(component_l2_orders)
        ),
        "minimum_component_linfinity_order": float(
            np.min(component_linf_orders)
        ),
        "fine_weighted_l2_difference": second_l2,
        "fine_linfinity_difference": second_linf,
        "weighted_history_cosine": _weighted_cosine(
            medium,
            fine,
            weights,
        ),
        "weighted_refinement_error_cosine": _weighted_cosine(
            first,
            second,
            weights,
        ),
        "conservative_error_cosine": sector_cosine(
            CONSERVATIVE_OBSERVABLES
        ),
        "distributed_error_cosine": sector_cosine(
            DISTRIBUTED_OBSERVABLES
        ),
        "coarse_medium_argmax": first_peak,
        "medium_fine_argmax": second_peak,
        "peak_migrated": peak_migrated,
    }


def _error_history_svd(
    histories: dict,
    scales: np.ndarray,
    *,
    stride: int = PRIMARY_STRIDE,
) -> tuple[dict, np.ndarray]:
    indices = wp10c9d6c2._selected_indices(
        histories[LABELS[0]]["times"].size,
        int(stride),
    )
    times = np.asarray(
        histories[LABELS[0]]["times"],
        dtype=float,
    )[indices]
    weights = _time_weights(times)
    scaled = {
        label: np.asarray(
            histories[label]["signals"],
            dtype=float,
        )[indices]
        / scales
        for label in LABELS
    }
    differences = np.asarray(
        [
            scaled[LABELS[index + 1]] - scaled[LABELS[index]]
            for index in range(3)
        ],
        dtype=float,
    )
    weighted = (
        differences
        * np.sqrt(weights)[None, :, None]
    ).reshape(3, -1)
    _left, singular, _right = np.linalg.svd(
        weighted,
        full_matrices=False,
    )
    energy = singular**2
    fractions = energy / max(float(np.sum(energy)), np.finfo(float).tiny)
    cosine_matrix = np.eye(3)
    for first in range(3):
        for second in range(first + 1, 3):
            value = _cosine(weighted[first], weighted[second])
            cosine_matrix[first, second] = value
            cosine_matrix[second, first] = value
    two_mode = bool(
        float(np.sum(fractions[:2]))
        >= MINIMUM_TWO_MODE_EXPLAINED_FRACTION
        and float(fractions[1]) >= MINIMUM_SECOND_MODE_FRACTION
    )
    return {
        "singular_values": singular,
        "explained_energy_fractions": fractions,
        "first_two_explained_fraction": float(np.sum(fractions[:2])),
        "second_mode_fraction": float(fractions[1]),
        "pairwise_error_cosines": cosine_matrix,
        "two_mode_candidate": two_mode,
    }, differences


def _continuum_fit_report(
    histories: dict,
    scales: np.ndarray,
    *,
    stride: int = PRIMARY_STRIDE,
) -> tuple[dict, dict[str, np.ndarray]]:
    indices = wp10c9d6c2._selected_indices(
        histories[LABELS[0]]["times"].size,
        int(stride),
    )
    times = np.asarray(
        histories[LABELS[0]]["times"],
        dtype=float,
    )[indices]
    weights = _time_weights(times)
    values = np.asarray(
        [
            np.asarray(histories[label]["signals"], dtype=float)[
                indices
            ]
            / scales
            for label in LABELS
        ],
        dtype=float,
    )
    h = np.asarray([1.0, 0.5, 0.25, 0.125], dtype=float)
    designs = {
        "h2": np.column_stack((np.ones_like(h), h**2)),
        "h2_h3": np.column_stack((np.ones_like(h), h**2, h**3)),
        "h2_h4": np.column_stack((np.ones_like(h), h**2, h**4)),
    }
    flattened = values.reshape(4, -1)
    reports = {}
    estimates = {}
    for name, design in designs.items():
        coefficients = np.linalg.lstsq(
            design,
            flattened,
            rcond=None,
        )[0]
        fit = (design @ coefficients).reshape(values.shape)
        residual = fit - values
        weighted_rms = float(
            np.sqrt(
                np.mean(
                    np.sum(
                        weights[None, :, None] * residual**2,
                        axis=1,
                    )
                )
            )
        )
        estimates[name] = coefficients[0].reshape(values.shape[1:])
        reports[name] = {
            "weighted_rms_residual": weighted_rms,
            "maximum_absolute_residual": float(
                np.max(np.abs(residual))
            ),
        }
    disagreement_h3_h4 = float(
        np.max(np.abs(estimates["h2_h3"] - estimates["h2_h4"]))
    )
    disagreement_h2_h4 = float(
        np.max(np.abs(estimates["h2"] - estimates["h2_h4"]))
    )
    stable = bool(
        disagreement_h3_h4 <= MAXIMUM_CONTINUUM_MODEL_DISAGREEMENT
    )
    return {
        "models": reports,
        "maximum_h2_h3_vs_h2_h4_continuum_difference": (
            disagreement_h3_h4
        ),
        "maximum_h2_vs_h2_h4_continuum_difference": (
            disagreement_h2_h4
        ),
        "stable": stable,
    }, estimates


def _build_tangents(
    configurations: dict,
    decisive: dict[str, np.ndarray],
) -> tuple[dict, dict, dict, dict]:
    tangents = {}
    observable_maps = {}
    method_reports = {}
    baselines = {}
    for label in LABELS:
        print(
            f"WP10c9d6c3: build monolithic tangent {label}",
            flush=True,
        )
        configuration = configurations[label]
        tangent = causal_five_field_monolithic_frozen_tangent(
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
        tangents[label] = tangent
        observable_maps[label] = wp10c9d6c._observable_map(tangent)
        method_reports[label] = wp10c9d6c._method_report(
            configuration,
            tangent,
        )
        evaluation = evaluate_causal_five_field_monolithic_backward_euler(
            configuration["base_primitives"],
            configuration["base_primitives"],
            1.0,
            configuration["context"],
            path_quadrature_order=wp10c9d6c.PATH_QUADRATURE_ORDER,
        )
        baselines[label] = wp10c9d6c._direct_observables(evaluation)
        decisive[f"{label}__baseline_observables"] = baselines[label]
        decisive[f"{label}__scaled_base_rate"] = (
            tangent.scaled_base_rate_per_s
        )
    return tangents, observable_maps, method_reports, baselines


def _propagate_profiles(
    configurations: dict,
    tangents: dict,
    observable_maps: dict,
    decisive: dict[str, np.ndarray],
) -> tuple[dict, dict]:
    all_histories = {}
    restart_defects = {}
    for profile in PERTURBATIONS:
        all_histories[profile] = {}
        restart_defects[profile] = {}
        for variant, suffix in (
            ("primary", ""),
            ("projection_order_12", "__projection_order_12"),
        ):
            histories = {}
            restarts = {}
            for label in LABELS:
                print(
                    "WP10c9d6c3: propagate "
                    f"{profile} {variant} on {label}",
                    flush=True,
                )
                configuration = configurations[label]
                initial = configuration["initial_directions"][
                    profile + suffix
                ]
                state, restart = wp10c9d6c._propagate(
                    tangents[label].scaled_generator_per_s,
                    initial,
                    configuration["times"],
                )
                signals = state @ observable_maps[label].T
                histories[label] = {
                    "times": np.asarray(
                        configuration["times"],
                        dtype=float,
                    ),
                    "signals": signals,
                    "final_scaled_state": state[-1],
                }
                restarts[label] = restart
                prefix = f"{profile}__{variant}__{label}__"
                decisive[prefix + "times"] = configuration["times"]
                decisive[prefix + "signals"] = signals
                decisive[prefix + "cumulative"] = wp10c9d6c._cumulative(
                    configuration["times"],
                    signals,
                )
                decisive[prefix + "final_scaled_state"] = state[-1]
            all_histories[profile][variant] = histories
            restart_defects[profile][variant] = restarts
    return all_histories, restart_defects


def _lift_uncertainty(
    name: str,
    configurations: dict,
    tangents: dict,
    observable_maps: dict,
    histories: dict,
    scales: np.ndarray,
) -> dict:
    state_defects = {}
    rate_defects = {}
    export_defects = {}
    for label in LABELS:
        primary = configurations[label]["initial_directions"][name]
        secondary = configurations[label]["initial_directions"][
            name + "__projection_order_12"
        ]
        state_defects[label] = _relative_difference(primary, secondary)
        rate_defects[label] = _relative_difference(
            tangents[label].scaled_generator_per_s @ primary,
            tangents[label].scaled_generator_per_s @ secondary,
        )
        export_defects[label] = float(
            np.max(
                np.abs(
                    observable_maps[label] @ (primary - secondary)
                    / scales
                )
            )
        )
    primary_histories = histories["primary"]
    secondary_histories = histories["projection_order_12"]
    fine_lift = float(
        np.max(
            np.abs(
                (
                    primary_histories[LABELS[-1]]["signals"]
                    - secondary_histories[LABELS[-1]]["signals"]
                )
                / scales
            )
        )
    )
    fine_spatial = float(
        np.max(
            np.abs(
                (
                    primary_histories[LABELS[-1]]["signals"]
                    - primary_histories[LABELS[-2]]["signals"]
                )
                / scales
            )
        )
    )
    primary_cumulative = wp10c9d6c._cumulative(
        primary_histories[LABELS[-1]]["times"],
        primary_histories[LABELS[-1]]["signals"],
    )
    secondary_cumulative = wp10c9d6c._cumulative(
        secondary_histories[LABELS[-1]]["times"],
        secondary_histories[LABELS[-1]]["signals"],
    )
    medium_cumulative = wp10c9d6c._cumulative(
        primary_histories[LABELS[-2]]["times"],
        primary_histories[LABELS[-2]]["signals"],
    )
    duration = max(
        float(primary_histories[LABELS[-1]]["times"][-1]),
        np.finfo(float).tiny,
    )
    cumulative_lift = float(
        np.max(
            np.abs(
                (primary_cumulative - secondary_cumulative)
                / (scales * duration)
            )
        )
    )
    cumulative_spatial = float(
        np.max(
            np.abs(
                (primary_cumulative - medium_cumulative)
                / (scales * duration)
            )
        )
    )
    history_ratio = fine_lift / max(
        fine_spatial,
        LIFT_EXPORT_COMPARISON_FLOOR,
    )
    cumulative_ratio = cumulative_lift / max(
        cumulative_spatial,
        LIFT_EXPORT_COMPARISON_FLOOR,
    )
    maximum_state = max(state_defects.values())
    maximum_rate = max(rate_defects.values())
    maximum_export = max(export_defects.values())
    passed = bool(
        maximum_state <= MAXIMUM_LIFT_STATE_RELATIVE_DIFFERENCE
        and maximum_rate <= MAXIMUM_LIFT_RATE_RELATIVE_DIFFERENCE
        and history_ratio <= MAXIMUM_LIFT_TO_FINE_EXPORT_RATIO
        and cumulative_ratio <= MAXIMUM_LIFT_TO_FINE_EXPORT_RATIO
    )
    return {
        "passed": passed,
        "projection_orders": (
            PRIMARY_PROJECTION_ORDER,
            SECONDARY_PROJECTION_ORDER,
        ),
        "state_relative_differences": state_defects,
        "rate_relative_differences": rate_defects,
        "initial_export_fixed_physical_differences": export_defects,
        "maximum_state_relative_difference": maximum_state,
        "maximum_rate_relative_difference": maximum_rate,
        "maximum_initial_export_fixed_physical_difference": (
            maximum_export
        ),
        "fine_history_lift_difference": fine_lift,
        "fine_history_spatial_difference": fine_spatial,
        "history_lift_to_spatial_ratio": history_ratio,
        "fine_cumulative_lift_difference": cumulative_lift,
        "fine_cumulative_spatial_difference": cumulative_spatial,
        "cumulative_lift_to_spatial_ratio": cumulative_ratio,
    }


def _profile_report(
    histories: dict,
    scales: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    primary = histories["primary"]
    historical = wp10c9d6c2._stride_report(primary, scales)
    conditioned = _conditioned_metrics(
        primary,
        scales,
        labels=LABELS[1:],
    )
    svd, differences = _error_history_svd(primary, scales)
    fits, estimates = _continuum_fit_report(primary, scales)
    metric_conditioning_candidate = bool(
        conditioned["passed"]
        and (conditioned["peak_migrated"] or svd["two_mode_candidate"])
        and fits["stable"]
    )
    arrays = {
        "error_history_differences": differences,
        **{
            f"continuum_estimate_{name}": values
            for name, values in estimates.items()
        },
    }
    return {
        "historical": historical,
        "conditioned": conditioned,
        "error_history_svd": svd,
        "continuum_fits": fits,
        "metric_conditioning_candidate": (
            metric_conditioning_candidate
        ),
    }, arrays


def _environment() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def run() -> dict:
    started = time.perf_counter()
    identity = _validate_analyzed_git_identity()
    parent_summary, parent_arrays = _load_parent()
    configurations, decisive, construction = (
        _build_continuum_configurations()
    )
    if not construction["passed"]:
        raise RuntimeError("WP10c9d6c3 continuum construction failed")

    tangents, observable_maps, method_reports, baselines = (
        _build_tangents(configurations, decisive)
    )
    physical_scales = np.asarray(
        parent_arrays["fixed_physical_observable_scales"],
        dtype=float,
    )
    decisive["fixed_physical_observable_scales"] = physical_scales
    histories, restart_defects = _propagate_profiles(
        configurations,
        tangents,
        observable_maps,
        decisive,
    )

    lift_reports = {
        name: _lift_uncertainty(
            name,
            configurations,
            tangents,
            observable_maps,
            histories[name],
            physical_scales,
        )
        for name in PERTURBATIONS
    }
    profile_reports = {}
    for name in PERTURBATIONS:
        report, arrays = _profile_report(
            histories[name],
            physical_scales,
        )
        profile_reports[name] = report
        for array_name, values in arrays.items():
            decisive[f"{name}__{array_name}"] = values

    maximum_restart = max(
        value
        for profile in restart_defects.values()
        for variant in profile.values()
        for value in variant.values()
    )
    method_passed = bool(
        construction["passed"]
        and all(
            report["passed"] for report in method_reports.values()
        )
        and maximum_restart <= MAXIMUM_RESTART_DEFECT
    )
    lift_passed = bool(
        all(report["passed"] for report in lift_reports.values())
    )
    strict_passed = bool(
        all(
            report["historical"]["passed"]
            for report in profile_reports.values()
        )
    )
    conditioned_norm_passed = bool(
        all(
            report["conditioned"]["passed"]
            for report in profile_reports.values()
        )
    )
    metric_conditioning_candidate = bool(
        all(
            report["metric_conditioning_candidate"]
            for report in profile_reports.values()
        )
    )

    if not method_passed:
        classification = "smooth_continuum_lift_method_failed"
        authorized_next = "none"
    elif not lift_passed:
        classification = "smooth_continuum_lift_uncertainty_unresolved"
        authorized_next = "improve_continuum_projection"
    elif strict_passed:
        classification = (
            "smooth_continuum_four_level_export_direction_certified"
        )
        authorized_next = "prospective_heldout_uniform_validation"
    elif conditioned_norm_passed and metric_conditioning_candidate:
        classification = (
            "smooth_continuum_norm_convergent_"
            "strict_direction_unresolved"
        )
        authorized_next = "prospective_uniform_validation_contract"
    else:
        classification = (
            "smooth_continuum_near_excision_"
            "truncation_audit_authorized"
        )
        authorized_next = "near_excision_local_truncation_audit"

    source_hashes, source_manifest = _source_manifest()
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "meshes": MESHES,
        "labels": LABELS,
        "active_cells": dict(zip(LABELS, ACTIVE_CELLS, strict=True)),
        "reference_label": REFERENCE_LABEL,
        "background_degrees": BACKGROUND_DEGREES,
        "background_coefficient_count": BACKGROUND_COEFFICIENT_COUNT,
        "primary_projection_order": PRIMARY_PROJECTION_ORDER,
        "secondary_projection_order": SECONDARY_PROJECTION_ORDER,
        "perturbations": PERTURBATIONS,
        "primary_stride": PRIMARY_STRIDE,
        "stride_audits": STRIDE_AUDITS,
        "gates": {
            "minimum_export_order": MINIMUM_EXPORT_ORDER,
            "maximum_fine_physical_difference": (
                MAXIMUM_FINE_PHYSICAL_DIFFERENCE
            ),
            "minimum_history_cosine": MINIMUM_HISTORY_COSINE,
            "minimum_error_cosine": MINIMUM_ERROR_COSINE,
            "maximum_background_constraint_defect": (
                MAXIMUM_BACKGROUND_CONSTRAINT_DEFECT
            ),
            "maximum_background_boundary_defect": (
                MAXIMUM_BACKGROUND_BOUNDARY_DEFECT
            ),
            "maximum_background_representation_difference": (
                MAXIMUM_BACKGROUND_REPRESENTATION_DIFFERENCE
            ),
            "maximum_lift_state_relative_difference": (
                MAXIMUM_LIFT_STATE_RELATIVE_DIFFERENCE
            ),
            "maximum_lift_rate_relative_difference": (
                MAXIMUM_LIFT_RATE_RELATIVE_DIFFERENCE
            ),
            "maximum_lift_to_fine_export_ratio": (
                MAXIMUM_LIFT_TO_FINE_EXPORT_RATIO
            ),
            "minimum_two_mode_explained_fraction": (
                MINIMUM_TWO_MODE_EXPLAINED_FRACTION
            ),
            "minimum_second_mode_fraction": (
                MINIMUM_SECOND_MODE_FRACTION
            ),
            "maximum_continuum_model_disagreement": (
                MAXIMUM_CONTINUUM_MODEL_DISAGREEMENT
            ),
        },
        "operator_change": False,
        "production_defaults_changed": False,
    }
    _write_json(CONFIG_PATH, config)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        **identity,
        "classification": classification,
        "authorized_next": authorized_next,
        "passed": strict_passed,
        "audit_executed": True,
        "method_passed": method_passed,
        "lift_uncertainty_passed": lift_passed,
        "strict_four_level_export_direction_passed": strict_passed,
        "conditioned_norm_passed": conditioned_norm_passed,
        "metric_conditioning_candidate": metric_conditioning_candidate,
        "parent_wp10c9d6c2_classification_preserved": True,
        "parent_classification": parent_summary["classification"],
        "continuum_construction": construction,
        "method_reports": method_reports,
        "baseline_observables": baselines,
        "lift_reports": lift_reports,
        "profile_reports": profile_reports,
        "restart_defects": restart_defects,
        "maximum_restart_defect": maximum_restart,
        "near_excision_local_truncation_audit_authorized": bool(
            authorized_next == "near_excision_local_truncation_audit"
        ),
        "prospective_uniform_validation_authorized": bool(
            authorized_next
            in {
                "prospective_heldout_uniform_validation",
                "prospective_uniform_validation_contract",
            }
        ),
        "direct_operator_redesign_authorized": False,
        "embedded_export_discrimination_authorized": False,
        "nonlinear_physical_trajectory_authorized": False,
        "production_operator_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "uses_production_generator": False,
        "uses_production_anchor_storage_derivative": False,
        "operator_changed": False,
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
        "scientific_status": (
            "CERTIFIED"
            if strict_passed
            else "DIAGNOSTIC COMPLETE"
            if method_passed and lift_passed
            else "UNRESOLVED"
        ),
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        **identity,
        "generation_command": (
            "PYTHONPATH=src:scripts python3 "
            "scripts/run_causal_inner_continuum_lift_wp10c9d6c3.py"
        ),
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "parent_canonical_hashes": {
            _relative(PARENT_SUMMARY): _sha256(PARENT_SUMMARY),
            _relative(PARENT_ARRAYS): _sha256(PARENT_ARRAYS),
            _relative(UNIFORM_REPLAY_CONTEXTS): _sha256(
                UNIFORM_REPLAY_CONTEXTS
            ),
            _relative(UNIFORM_REPLAY_INPUTS): _sha256(
                UNIFORM_REPLAY_INPUTS
            ),
        },
        "establishes": (
            "whether a smooth, proper-measure common continuum lift "
            "reproduces the strict four-level export-direction failure "
            "and whether that failure is conditioned by peak or error-mode "
            "migration"
        ),
        "does_not_establish": (
            "an operator defect, embedded convergence, nonlinear "
            "convergence, production eligibility, fixed-Q closure, or "
            "reduced slow-time evolution"
        ),
    }
    _write_json(PROVENANCE_PATH, provenance)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    print(
        f"WP10c9d6c3: classification={classification}",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
