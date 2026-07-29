#!/usr/bin/env python3
"""Run monolithic manufactured balance and wave gates (WP10c9d6b)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from dataclasses import replace
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from imri_qpe.constants import C  # noqa: E402
from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    causal_five_field_coordinate_principal_basis,
    evaluate_causal_five_field_monolithic_backward_euler,
    make_causal_five_field_regression_context,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_geometry import (  # noqa: E402
    kerr_schild_column_geometry,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_radial_linear_tangent import (  # noqa: E402
    causal_five_field_analytic_local_maps,
)


def _load_d4b_runner():
    path = (
        ROOT
        / "scripts/run_causal_inner_radial_fluctuation_audit_wp10c9d4b.py"
    )
    spec = importlib.util.spec_from_file_location("wp10c9d4b_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import the WP10c9d4b reference runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


D4B = _load_d4b_runner()

SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6b"
ANALYZED_BASE_COMMIT = "4140ffeb58ce791425219b88209a8f20e0e2a70d"
ANALYZED_BASE_PARENT = "e836df2e2c0e1d180f3a8c56383498578434762e"
ANALYZED_BASE_TREE = "21b77ef20f847f65bfebc7fca05c3ac445b30daa"
THIS_RUNNER = (
    "scripts/run_causal_inner_monolithic_manufactured_wp10c9d6b.py"
)

GRID_SIZES = (12, 24, 48)
ACTIVE_GUARD_CELLS = 2
INNER_RADIUS_OVER_RG = 1.8
OUTER_RADIUS_OVER_RG = 6.648
TIMESTEP_SECONDS = 1.0e-4
TEMPORAL_QUADRATURE_ORDER = 6
REFERENCE_SPATIAL_QUADRATURE_ORDER = 6
REFERENCE_TEMPORAL_QUADRATURE_ORDER = 8
PATH_QUADRATURE_ORDER = 6
WAVE_FAMILY = "inward_shear"
WAVE_AMPLITUDE = 1.0e-3
TEMPORAL_FREQUENCY_PER_S = 500.0
TEMPORAL_TIMESTEPS_SECONDS = (
    4.0e-4,
    2.0e-4,
    1.0e-4,
    5.0e-5,
    2.5e-5,
)

MINIMUM_SPATIAL_ORDER = 1.8
MAXIMUM_FINE_STATIONARY_ERROR = 2.0e-3
MAXIMUM_FINE_WAVE_ERROR = 1.0e-2
MINIMUM_WAVE_TEMPORAL_ACTIVITY = 0.1
MINIMUM_TEMPORAL_ORDER = 0.9
MAXIMUM_FINE_TEMPORAL_ERROR = 1.0e-2
MAXIMUM_BLOCK_LEDGER_DEFECT = 1.0e-12
MAXIMUM_SHARED_FLUX_TELESCOPE_DEFECT = 1.0e-12
MAXIMUM_MAPPED_PATH_CLOSURE_DEFECT = 5.0e-7
MAXIMUM_AFFINE_RECONSTRUCTION_PATH_DEFECT = 1.0e-12
MAXIMUM_RECONSTRUCTION_FACTOR_CHANGE = 1.0e-12

PARENT_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_monolithic_dae_preflight_wp10c9d6a"
)
PARENT_SUMMARY = PARENT_DIRECTORY / "summary.json"
PARENT_ARRAYS = PARENT_DIRECTORY / "decisive_arrays.npz"
D4B_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_radial_fluctuation_wp10c9d4b"
)
D4B_SUMMARY = D4B_DIRECTORY / "summary.json"
D4B_ARRAYS = D4B_DIRECTORY / "decisive_arrays.npz"
CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_monolithic_manufactured_wp10c9d6b"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"

IMPLEMENTATION_SOURCES = (
    THIS_RUNNER,
    "scripts/run_causal_inner_radial_fluctuation_audit_wp10c9d4b.py",
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_dae.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_radial_fluctuation.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_radial_linear_tangent.py",
    "src/imri_qpe/layer3_minidisk_1d/__init__.py",
    "tests/test_causal_inner_monolithic_manufactured_wp10c9d6b.py",
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
        raise RuntimeError("WP10c9d6b analyzed Git identity changed")
    return {
        "analyzed_base_commit": commit,
        "analyzed_base_parent_commit": parent,
        "analyzed_base_tree_sha": tree,
    }


def _load_parent_evidence() -> tuple[dict, dict]:
    parent = json.loads(PARENT_SUMMARY.read_text(encoding="utf-8"))
    if (
        not parent["passed"]
        or not parent[
            "manufactured_equilibrium_and_wave_preflight_authorized"
        ]
        or parent["physical_export_discrimination_authorized"]
        or parent["nonlinear_physical_trajectory_authorized"]
        or not parent["declared_temporal_path_product_required"]
        or parent["strict_endpoint_storage_potential_authorized"]
    ):
        raise RuntimeError("WP10c9d6a binding status changed")
    if _sha256(PARENT_ARRAYS) != parent["decisive_arrays_sha256"]:
        raise RuntimeError("WP10c9d6a decisive archive changed")
    d4b = json.loads(D4B_SUMMARY.read_text(encoding="utf-8"))
    if not d4b["radial_candidate_gate_passed"]:
        raise RuntimeError("WP10c9d4b radial reference status changed")
    if _sha256(D4B_ARRAYS) != d4b["decisive_arrays_sha256"]:
        raise RuntimeError("WP10c9d4b decisive archive changed")
    return parent, d4b


def _orders(errors: list[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(errors, dtype=float)
    return np.log2(values[:-1] / values[1:])


def _relative_l2(
    numerical: np.ndarray,
    reference: np.ndarray,
    selected: slice,
) -> float:
    difference = np.asarray(numerical, dtype=float)[selected] - np.asarray(
        reference,
        dtype=float,
    )[selected]
    scale = max(
        float(np.linalg.norm(np.asarray(reference, dtype=float)[selected])),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(difference) / scale)


def _base_profile_and_context():
    with np.load(D4B_ARRAYS, allow_pickle=False) as source:
        inner = np.asarray(source["manufactured_inner_chart"], dtype=float)
        outer = np.asarray(source["manufactured_outer_chart"], dtype=float)
    context = make_causal_five_field_regression_context(4)
    gravitational_radius = context.grid.gravitational_radius
    profile = D4B._ManufacturedProfile(
        INNER_RADIUS_OVER_RG * gravitational_radius,
        OUTER_RADIUS_OVER_RG * gravitational_radius,
        inner,
        outer,
    )
    context = replace(
        context,
        stream_sources=None,
        spatial_reconstruction="quadratic_admissible",
        boundary_trace_reconstruction="plm_one_sided",
        cell_source_quadrature="gauss_legendre_4_local_rates",
        cell_storage_quadrature="gauss_legendre_4",
        outer_boundary_flux_mode="frozen_exterior_rusanov",
        outer_boundary_frozen_exterior_chart=np.asarray(
            profile.chart(profile.outer_radius),
            dtype=float,
        ),
    ).validated()
    return profile, context


class _OutgoingWaveProfile:
    """Boundary-active negative-speed shear-family manufactured wave."""

    def __init__(
        self,
        base_profile,
        time_seconds: float,
        direction: np.ndarray,
        speed_over_c: float,
    ) -> None:
        self.base_profile = base_profile
        self.time_seconds = float(time_seconds)
        self.direction = np.asarray(direction, dtype=float)
        self.speed_over_c = float(speed_over_c)
        self.inner_radius = float(base_profile.inner_radius)
        self.outer_radius = float(base_profile.outer_radius)

    def coordinate(self, radius):
        return self.base_profile.coordinate(radius)

    def _phase(self, coordinate):
        coordinate_time = C * self.time_seconds
        scaled_time = (
            coordinate_time
            / (self.inner_radius * self.base_profile.log_span)
        )
        return 2.0 * np.pi * (
            coordinate - self.speed_over_c * scaled_time
        )

    def chart(self, radius):
        coordinate = self.coordinate(radius)
        envelope = (1.0 - coordinate) ** 4
        return (
            self.base_profile.chart(radius)
            + WAVE_AMPLITUDE
            * envelope[..., None]
            * np.cos(self._phase(coordinate))[..., None]
            * self.direction
        )

    def derivative(self, radius: float) -> np.ndarray:
        selected_radius = float(radius)
        coordinate = float(self.coordinate(selected_radius))
        envelope = (1.0 - coordinate) ** 4
        phase = float(self._phase(coordinate))
        derivative_coordinate = (
            -4.0 * (1.0 - coordinate) ** 3 * np.cos(phase)
            - 2.0 * np.pi * envelope * np.sin(phase)
        )
        return (
            self.base_profile.derivative(selected_radius)
            + WAVE_AMPLITUDE
            * self.direction
            * derivative_coordinate
            / (selected_radius * self.base_profile.log_span)
        )


def _outgoing_wave(base_profile, context):
    basis = causal_five_field_coordinate_principal_basis(
        context,
        base_profile.inner_radius,
        base_profile.chart(base_profile.inner_radius),
    )
    family = basis.family_labels.index(WAVE_FAMILY)
    raw = np.asarray(
        basis.primitive_right_eigenvectors[:, family],
        dtype=float,
    )
    direction = raw / max(
        float(np.max(np.abs(raw / basis.primitive_column_scales))),
        np.finfo(float).tiny,
    )
    return {
        "direction": direction,
        "speed_over_c": float(basis.numerical_speeds_over_c[family]),
        "incoming_inner_characteristics": int(
            basis.incoming_inner_characteristics
        ),
        "minimum_speed_over_c": float(
            np.min(basis.numerical_speeds_over_c)
        ),
        "maximum_speed_over_c": float(
            np.max(basis.numerical_speeds_over_c)
        ),
    }


def _reference_spatial_nodes_and_weights(context, cell: int):
    nodes, weights = np.polynomial.legendre.leggauss(
        REFERENCE_SPATIAL_QUADRATURE_ORDER
    )
    lower = float(np.log(context.grid.edges[cell]))
    upper = float(np.log(context.grid.edges[cell + 1]))
    midpoint = 0.5 * (lower + upper)
    half_width = 0.5 * (upper - lower)
    radii = np.exp(midpoint + half_width * nodes)
    raw_weights = np.asarray(
        [
            half_width
            * float(weight)
            * float(radius)
            * kerr_schild_column_geometry(
                float(radius),
                context.grid.gravitational_radius,
            ).face_measure
            for radius, weight in zip(radii, weights, strict=True)
        ],
        dtype=float,
    )
    physical_weights = (
        raw_weights
        * float(context.grid.cell_measures[cell])
        / float(np.sum(raw_weights))
    )
    return radii, physical_weights


def _independent_temporal_reference(
    context,
    old_profile,
    new_profile,
    timestep_seconds: float,
) -> np.ndarray:
    """Integrate exact profiles, not reconstructed cell-center samples."""

    temporal_nodes, temporal_weights = np.polynomial.legendre.leggauss(
        REFERENCE_TEMPORAL_QUADRATURE_ORDER
    )
    fractions = 0.5 * (temporal_nodes + 1.0)
    temporal_weights = 0.5 * temporal_weights
    result = np.zeros((context.grid.centers.size, 5), dtype=float)
    for cell in range(int(context.grid.centers.size)):
        radii, spatial_weights = _reference_spatial_nodes_and_weights(
            context,
            cell,
        )
        for radius, spatial_weight in zip(
            radii,
            spatial_weights,
            strict=True,
        ):
            old = np.asarray(old_profile.chart(radius), dtype=float)
            new = np.asarray(new_profile.chart(radius), dtype=float)
            direction = new - old
            old_local = causal_five_field_analytic_local_maps(
                context,
                float(radius),
                old,
            )
            new_local = causal_five_field_analytic_local_maps(
                context,
                float(radius),
                new,
            )
            increment = (
                new_local.mapped_conserved - old_local.mapped_conserved
            )
            for fraction, temporal_weight in zip(
                fractions,
                temporal_weights,
                strict=True,
            ):
                local = causal_five_field_analytic_local_maps(
                    context,
                    float(radius),
                    old + float(fraction) * direction,
                )
                increment += (
                    float(temporal_weight)
                    * local.vertical_storage_matrix
                    @ direction
                )
            result[cell] += (
                float(spatial_weight)
                * increment
                / (C * float(timestep_seconds))
            )
    return result


def _shared_flux_telescope(evaluation) -> float:
    faces = np.asarray(
        evaluation.stationary_ledger.interfaces
        .candidate_shared_face_fluxes_over_c,
        dtype=float,
    )
    difference = (
        evaluation.conservative_transport_rows
        - (faces[1:] - faces[:-1])
    )
    scale = max(
        float(np.max(np.abs(evaluation.conservative_transport_rows))),
        float(np.max(np.abs(faces))),
        np.finfo(float).tiny,
    )
    return float(np.max(np.abs(difference)) / scale)


def _spatial_wave_ladders(base_profile, base_context, wave) -> tuple[
    dict,
    dict[str, np.ndarray],
]:
    old_wave = _OutgoingWaveProfile(
        base_profile,
        0.0,
        wave["direction"],
        wave["speed_over_c"],
    )
    new_wave = _OutgoingWaveProfile(
        base_profile,
        TIMESTEP_SECONDS,
        wave["direction"],
        wave["speed_over_c"],
    )
    stationary_errors = []
    wave_interior_errors = []
    wave_boundary_errors = []
    temporal_activity = []
    ladder = []
    arrays: dict[str, np.ndarray] = {
        "wave_direction": wave["direction"],
    }
    maximum_ledger = 0.0
    maximum_telescope = 0.0
    maximum_mapped_closure = 0.0
    maximum_affine_reconstruction_defect = 0.0
    maximum_factor_change = 0.0
    exact_affine_reconstruction_derivative_used = True
    incoming = 0

    for n_cells in GRID_SIZES:
        print(f"WP10c9d6b: grid N{n_cells}", flush=True)
        context = D4B._make_context(base_context, new_wave, n_cells)
        context = replace(
            context,
            outer_boundary_frozen_exterior_chart=np.asarray(
                new_wave.chart(new_wave.outer_radius),
                dtype=float,
            ),
        ).validated()
        base_charts = np.asarray(
            base_profile.chart(context.grid.centers),
            dtype=float,
        )
        old_charts = np.asarray(
            old_wave.chart(context.grid.centers),
            dtype=float,
        )
        new_charts = np.asarray(
            new_wave.chart(context.grid.centers),
            dtype=float,
        )
        base_evaluation = (
            evaluate_causal_five_field_monolithic_backward_euler(
                base_charts,
                base_charts,
                TIMESTEP_SECONDS,
                context,
                temporal_quadrature_order=TEMPORAL_QUADRATURE_ORDER,
                path_quadrature_order=PATH_QUADRATURE_ORDER,
            )
        )
        wave_evaluation = (
            evaluate_causal_five_field_monolithic_backward_euler(
                old_charts,
                new_charts,
                TIMESTEP_SECONDS,
                context,
                temporal_quadrature_order=TEMPORAL_QUADRATURE_ORDER,
                path_quadrature_order=PATH_QUADRATURE_ORDER,
            )
        )
        base_reference = D4B._independent_radial_reference(
            context,
            base_profile,
        )
        wave_spatial_reference = D4B._independent_radial_reference(
            context,
            new_wave,
        )
        temporal_reference = _independent_temporal_reference(
            context,
            old_wave,
            new_wave,
            TIMESTEP_SECONDS,
        )
        wave_reference = temporal_reference + wave_spatial_reference
        numerical_delta = (
            wave_evaluation.residual_rows
            - base_evaluation.residual_rows
        )
        reference_delta = (
            wave_reference - base_reference
        )
        interior = slice(ACTIVE_GUARD_CELLS, -ACTIVE_GUARD_CELLS)
        boundary_inclusive = slice(0, -ACTIVE_GUARD_CELLS)
        stationary_error = _relative_l2(
            base_evaluation.residual_rows,
            base_reference,
            interior,
        )
        wave_interior_error = _relative_l2(
            numerical_delta,
            reference_delta,
            interior,
        )
        wave_boundary_error = _relative_l2(
            numerical_delta,
            reference_delta,
            boundary_inclusive,
        )
        activity = float(
            np.linalg.norm(temporal_reference[interior])
            / max(
                float(np.linalg.norm(reference_delta[interior])),
                np.finfo(float).tiny,
            )
        )
        stationary_errors.append(stationary_error)
        wave_interior_errors.append(wave_interior_error)
        wave_boundary_errors.append(wave_boundary_error)
        temporal_activity.append(activity)
        maximum_ledger = max(
            maximum_ledger,
            base_evaluation.maximum_block_ledger_defect,
            wave_evaluation.maximum_block_ledger_defect,
        )
        maximum_telescope = max(
            maximum_telescope,
            _shared_flux_telescope(base_evaluation),
            _shared_flux_telescope(wave_evaluation),
        )
        maximum_mapped_closure = max(
            maximum_mapped_closure,
            base_evaluation.storage_increment
            .maximum_mapped_path_closure_defect,
            wave_evaluation.storage_increment
            .maximum_mapped_path_closure_defect,
        )
        maximum_affine_reconstruction_defect = max(
            maximum_affine_reconstruction_defect,
            base_evaluation.storage_increment
            .maximum_affine_reconstruction_path_defect,
            wave_evaluation.storage_increment
            .maximum_affine_reconstruction_path_defect,
        )
        exact_affine_reconstruction_derivative_used = bool(
            exact_affine_reconstruction_derivative_used
            and base_evaluation.storage_increment
            .uses_exact_affine_reconstruction_path_derivative
            and wave_evaluation.storage_increment
            .uses_exact_affine_reconstruction_path_derivative
        )
        maximum_factor_change = max(
            maximum_factor_change,
            base_evaluation.storage_increment
            .maximum_path_reconstruction_factor_change,
            wave_evaluation.storage_increment
            .maximum_path_reconstruction_factor_change,
        )
        incoming = max(
            incoming,
            base_evaluation.incoming_excision_characteristics,
            wave_evaluation.incoming_excision_characteristics,
        )
        ladder.append(
            {
                "n_cells": n_cells,
                "stationary_manufactured_relative_l2_error": (
                    stationary_error
                ),
                "outgoing_wave_interior_relative_l2_error": (
                    wave_interior_error
                ),
                "outgoing_wave_boundary_relative_l2_error": (
                    wave_boundary_error
                ),
                "wave_temporal_activity_fraction": activity,
                "base_block_ledger_defect": (
                    base_evaluation.maximum_block_ledger_defect
                ),
                "wave_block_ledger_defect": (
                    wave_evaluation.maximum_block_ledger_defect
                ),
                "incoming_excision_characteristics": (
                    wave_evaluation.incoming_excision_characteristics
                ),
                "base_uses_exact_affine_reconstruction_path_derivative": (
                    base_evaluation.storage_increment
                    .uses_exact_affine_reconstruction_path_derivative
                ),
                "wave_uses_exact_affine_reconstruction_path_derivative": (
                    wave_evaluation.storage_increment
                    .uses_exact_affine_reconstruction_path_derivative
                ),
            }
        )
        prefix = f"N{n_cells}_"
        arrays[f"{prefix}centers"] = np.asarray(
            context.grid.centers,
            dtype=float,
        )
        arrays[f"{prefix}base_charts"] = base_charts
        arrays[f"{prefix}old_wave_charts"] = old_charts
        arrays[f"{prefix}new_wave_charts"] = new_charts
        arrays[f"{prefix}base_numerical_residual"] = (
            base_evaluation.residual_rows
        )
        arrays[f"{prefix}base_reference_residual"] = base_reference
        arrays[f"{prefix}wave_numerical_delta"] = numerical_delta
        arrays[f"{prefix}wave_reference_delta"] = reference_delta
        arrays[f"{prefix}temporal_reference"] = temporal_reference

    stationary_orders = _orders(stationary_errors)
    wave_interior_orders = _orders(wave_interior_errors)
    wave_boundary_orders = _orders(wave_boundary_errors)
    report = {
        "grid_sizes": GRID_SIZES,
        "active_guard_cells": ACTIVE_GUARD_CELLS,
        "ladder": ladder,
        "stationary_errors": stationary_errors,
        "stationary_orders": stationary_orders,
        "outgoing_wave_interior_errors": wave_interior_errors,
        "outgoing_wave_interior_orders": wave_interior_orders,
        "outgoing_wave_boundary_errors": wave_boundary_errors,
        "outgoing_wave_boundary_orders": wave_boundary_orders,
        "minimum_temporal_activity_fraction": min(temporal_activity),
        "maximum_block_ledger_defect": maximum_ledger,
        "maximum_shared_flux_telescope_defect": maximum_telescope,
        "maximum_mapped_path_closure_defect": maximum_mapped_closure,
        "maximum_affine_reconstruction_path_defect": (
            maximum_affine_reconstruction_defect
        ),
        "maximum_reconstruction_factor_change": maximum_factor_change,
        "exact_affine_reconstruction_path_derivative_used": (
            exact_affine_reconstruction_derivative_used
        ),
        "incoming_excision_characteristics": incoming,
        "stationary_passed": bool(
            np.min(stationary_orders) >= MINIMUM_SPATIAL_ORDER
            and stationary_errors[-1] <= MAXIMUM_FINE_STATIONARY_ERROR
        ),
        "outgoing_wave_passed": bool(
            np.min(wave_interior_orders) >= MINIMUM_SPATIAL_ORDER
            and np.min(wave_boundary_orders) >= MINIMUM_SPATIAL_ORDER
            and wave_interior_errors[-1] <= MAXIMUM_FINE_WAVE_ERROR
            and wave_boundary_errors[-1] <= MAXIMUM_FINE_WAVE_ERROR
            and min(temporal_activity)
            >= MINIMUM_WAVE_TEMPORAL_ACTIVITY
            and incoming == 0
        ),
    }
    return report, arrays


def _integrate_local_vertical_path(
    context,
    radius: float,
    old: np.ndarray,
    new: np.ndarray,
) -> np.ndarray:
    nodes, weights = np.polynomial.legendre.leggauss(
        REFERENCE_TEMPORAL_QUADRATURE_ORDER
    )
    direction = np.asarray(new, dtype=float) - np.asarray(old, dtype=float)
    result = np.zeros(5, dtype=float)
    for fraction, weight in zip(
        0.5 * (nodes + 1.0),
        0.5 * weights,
        strict=True,
    ):
        local = causal_five_field_analytic_local_maps(
            context,
            radius,
            np.asarray(old, dtype=float) + float(fraction) * direction,
        )
        result += (
            float(weight) * local.vertical_storage_matrix @ direction
        )
    return result


def _temporal_refinement(base_profile, context) -> tuple[
    dict,
    dict[str, np.ndarray],
]:
    radius = float(base_profile.inner_radius)
    base = np.asarray(base_profile.chart(radius), dtype=float)
    direction = np.zeros(5, dtype=float)
    direction[3] = 1.0e-3
    errors = []
    numerical_rates = []
    reference_rates = []

    for timestep in TEMPORAL_TIMESTEPS_SECONDS:
        old = np.array(base, copy=True)
        exponential = float(
            np.exp(TEMPORAL_FREQUENCY_PER_S * timestep)
        )
        new = base + (exponential - 1.0) * direction
        old_local = causal_five_field_analytic_local_maps(
            context,
            radius,
            old,
        )
        new_local = causal_five_field_analytic_local_maps(
            context,
            radius,
            new,
        )
        increment = (
            new_local.mapped_conserved
            - old_local.mapped_conserved
            + _integrate_local_vertical_path(
                context,
                radius,
                old,
                new,
            )
        )
        numerical = increment / (C * timestep)
        endpoint_rate = (
            TEMPORAL_FREQUENCY_PER_S
            * exponential
            * direction
        )
        reference = (
            new_local.temporal_storage_matrix @ endpoint_rate / C
        )
        error = float(
            np.linalg.norm(numerical - reference)
            / max(
                float(np.linalg.norm(reference)),
                np.finfo(float).tiny,
            )
        )
        errors.append(error)
        numerical_rates.append(numerical)
        reference_rates.append(reference)

    orders = _orders(errors)
    report = {
        "timesteps_seconds": TEMPORAL_TIMESTEPS_SECONDS,
        "relative_errors": errors,
        "observed_orders": orders,
        "minimum_observed_order": float(np.min(orders)),
        "fine_relative_error": errors[-1],
        "passed": bool(
            np.min(orders) >= MINIMUM_TEMPORAL_ORDER
            and errors[-1] <= MAXIMUM_FINE_TEMPORAL_ERROR
        ),
    }
    arrays = {
        "temporal_refinement_direction": direction,
        "temporal_refinement_numerical_rates": np.asarray(
            numerical_rates,
            dtype=float,
        ),
        "temporal_refinement_reference_rates": np.asarray(
            reference_rates,
            dtype=float,
        ),
        "temporal_refinement_errors": np.asarray(errors, dtype=float),
    }
    return report, arrays


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
    parent, d4b = _load_parent_evidence()
    base_profile, base_context = _base_profile_and_context()
    wave = _outgoing_wave(base_profile, base_context)
    spatial_wave, spatial_arrays = _spatial_wave_ladders(
        base_profile,
        base_context,
        wave,
    )
    temporal, temporal_arrays = _temporal_refinement(
        base_profile,
        base_context,
    )

    method_passed = bool(
        spatial_wave["stationary_passed"]
        and spatial_wave["outgoing_wave_passed"]
        and temporal["passed"]
        and spatial_wave["maximum_block_ledger_defect"]
        <= MAXIMUM_BLOCK_LEDGER_DEFECT
        and spatial_wave["maximum_shared_flux_telescope_defect"]
        <= MAXIMUM_SHARED_FLUX_TELESCOPE_DEFECT
        and spatial_wave["maximum_mapped_path_closure_defect"]
        <= MAXIMUM_MAPPED_PATH_CLOSURE_DEFECT
        and spatial_wave["maximum_affine_reconstruction_path_defect"]
        <= MAXIMUM_AFFINE_RECONSTRUCTION_PATH_DEFECT
        and spatial_wave["maximum_reconstruction_factor_change"]
        <= MAXIMUM_RECONSTRUCTION_FACTOR_CHANGE
        and spatial_wave[
            "exact_affine_reconstruction_path_derivative_used"
        ]
        and wave["speed_over_c"] < 0.0
        and wave["incoming_inner_characteristics"] == 0
    )
    classification = (
        "monolithic_manufactured_balance_and_outgoing_wave_passed_"
        "uniform_export_preflight_authorized"
        if method_passed
        else "monolithic_manufactured_method_gate_failed"
    )
    decisive = {
        "base_profile_inner_chart": np.asarray(
            base_profile.inner_chart,
            dtype=float,
        ),
        "base_profile_outer_chart": np.asarray(
            base_profile.outer_chart,
            dtype=float,
        ),
        **spatial_arrays,
        **temporal_arrays,
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)
    source_hashes, source_manifest = _source_manifest()
    gates = {
        "minimum_spatial_order": MINIMUM_SPATIAL_ORDER,
        "maximum_fine_stationary_error": (
            MAXIMUM_FINE_STATIONARY_ERROR
        ),
        "maximum_fine_wave_error": MAXIMUM_FINE_WAVE_ERROR,
        "minimum_wave_temporal_activity": (
            MINIMUM_WAVE_TEMPORAL_ACTIVITY
        ),
        "minimum_temporal_order": MINIMUM_TEMPORAL_ORDER,
        "maximum_fine_temporal_error": MAXIMUM_FINE_TEMPORAL_ERROR,
        "maximum_block_ledger_defect": MAXIMUM_BLOCK_LEDGER_DEFECT,
        "maximum_shared_flux_telescope_defect": (
            MAXIMUM_SHARED_FLUX_TELESCOPE_DEFECT
        ),
        "maximum_mapped_path_closure_defect": (
            MAXIMUM_MAPPED_PATH_CLOSURE_DEFECT
        ),
        "maximum_affine_reconstruction_path_defect": (
            MAXIMUM_AFFINE_RECONSTRUCTION_PATH_DEFECT
        ),
        "maximum_reconstruction_factor_change": (
            MAXIMUM_RECONSTRUCTION_FACTOR_CHANGE
        ),
    }
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "grid_sizes": GRID_SIZES,
        "active_guard_cells": ACTIVE_GUARD_CELLS,
        "inner_radius_over_rg": INNER_RADIUS_OVER_RG,
        "outer_radius_over_rg": OUTER_RADIUS_OVER_RG,
        "timestep_seconds": TIMESTEP_SECONDS,
        "temporal_quadrature_order": TEMPORAL_QUADRATURE_ORDER,
        "reference_spatial_quadrature_order": (
            REFERENCE_SPATIAL_QUADRATURE_ORDER
        ),
        "reference_temporal_quadrature_order": (
            REFERENCE_TEMPORAL_QUADRATURE_ORDER
        ),
        "path_quadrature_order": PATH_QUADRATURE_ORDER,
        "wave_family": WAVE_FAMILY,
        "wave_amplitude": WAVE_AMPLITUDE,
        "temporal_frequency_per_s": TEMPORAL_FREQUENCY_PER_S,
        "temporal_timesteps_seconds": TEMPORAL_TIMESTEPS_SECONDS,
        "gates": gates,
    }
    _write_json(CONFIG_PATH, config)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        **identity,
        "classification": classification,
        "method_passed": method_passed,
        "parent_wp10c9d6a_summary_path": _relative(PARENT_SUMMARY),
        "parent_wp10c9d6a_summary_sha256": _sha256(PARENT_SUMMARY),
        "parent_descriptor_path_preflight_remains_passed": bool(
            parent["passed"]
        ),
        "wp10c9d4b_reference_summary_path": _relative(D4B_SUMMARY),
        "wp10c9d4b_reference_summary_sha256": _sha256(D4B_SUMMARY),
        "wp10c9d4b_radial_reference_remains_passed": bool(
            d4b["radial_candidate_gate_passed"]
        ),
        "manufactured_balance": {
            "forcing_inserted_into_operator": False,
            "residual_subtraction_used_in_operator": False,
            "independent_continuum_forcing_reference": True,
            "errors": spatial_wave["stationary_errors"],
            "orders": spatial_wave["stationary_orders"],
            "passed": spatial_wave["stationary_passed"],
        },
        "outgoing_near_horizon_wave": {
            "family": WAVE_FAMILY,
            "speed_over_c": wave["speed_over_c"],
            "direction": wave["direction"],
            "base_incoming_inner_characteristics": (
                wave["incoming_inner_characteristics"]
            ),
            "interior_errors": (
                spatial_wave["outgoing_wave_interior_errors"]
            ),
            "interior_orders": (
                spatial_wave["outgoing_wave_interior_orders"]
            ),
            "boundary_inclusive_errors": (
                spatial_wave["outgoing_wave_boundary_errors"]
            ),
            "boundary_inclusive_orders": (
                spatial_wave["outgoing_wave_boundary_orders"]
            ),
            "minimum_temporal_activity_fraction": (
                spatial_wave["minimum_temporal_activity_fraction"]
            ),
            "passed": spatial_wave["outgoing_wave_passed"],
        },
        "temporal_refinement": temporal,
        "method_ledger": {
            "maximum_block_ledger_defect": (
                spatial_wave["maximum_block_ledger_defect"]
            ),
            "maximum_shared_flux_telescope_defect": (
                spatial_wave["maximum_shared_flux_telescope_defect"]
            ),
            "maximum_mapped_path_closure_defect": (
                spatial_wave["maximum_mapped_path_closure_defect"]
            ),
            "maximum_affine_reconstruction_path_defect": (
                spatial_wave[
                    "maximum_affine_reconstruction_path_defect"
                ]
            ),
            "maximum_reconstruction_factor_change": (
                spatial_wave["maximum_reconstruction_factor_change"]
            ),
            "exact_affine_reconstruction_path_derivative_used": (
                spatial_wave[
                    "exact_affine_reconstruction_path_derivative_used"
                ]
            ),
            "incoming_excision_characteristics": (
                spatial_wave["incoming_excision_characteristics"]
            ),
        },
        "strict_endpoint_storage_potential_authorized": False,
        "declared_temporal_path_product_required": True,
        "uniform_grid_physical_export_preflight_authorized": (
            method_passed
        ),
        "embedded_export_discrimination_authorized": False,
        "nonlinear_physical_trajectory_authorized": False,
        "production_operator_authorized": False,
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
            "PYTHONPATH=src:scripts python3 "
            "scripts/run_causal_inner_monolithic_manufactured_wp10c9d6b.py"
        ),
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        "parent_canonical_hashes": {
            _relative(path): _sha256(path)
            for path in (
                PARENT_SUMMARY,
                PARENT_ARRAYS,
                D4B_SUMMARY,
                D4B_ARRAYS,
            )
        },
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "scientific_status": (
            "CERTIFIED" if method_passed else "REJECTED"
        ),
        "authorization_status": (
            "uniform_grid_physical_export_preflight"
            if method_passed
            else "none"
        ),
        "establishes": (
            "Spatial convergence of an independent manufactured balance, "
            "boundary-inclusive convergence of a negative-speed near-horizon "
            "wave, first-order temporal consistency, and complete method "
            "ledgers for the monolithic descriptor-path residual."
        ),
        "does_not_establish": (
            "Uniform or embedded physical-export convergence, a nonlinear "
            "physical trajectory, production readiness, fixed-Q closure, "
            "or reduced slow evolution."
        ),
    }
    _write_json(PROVENANCE_PATH, provenance)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    return summary


def main() -> None:
    print(json.dumps(_plain(run()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
