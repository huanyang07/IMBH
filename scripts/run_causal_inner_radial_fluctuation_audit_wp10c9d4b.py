"""Run the WP10c9d4b radial complete-fluctuation balance audit.

The audit uses the real nonuniform Kerr--Schild grid, exact face/cell
measures, the unchanged production reconstruction and boundary states, and a
production-neutral complete five-field fluctuation residual.  It compares
that candidate with an independently evaluated smooth radial manufactured
family and verifies candidate finite-difference versus block-assembled
stationary Jacobians.  The old production Jacobian is retained as a distinct
baseline, not as a candidate closure target.
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
    causal_five_field_coordinate_principal_components,
    causal_five_field_radial_candidate_ledger,
    causal_five_field_reduced_stationary_residual,
    make_kerr_schild_column_grid,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_dae_system import (  # noqa: E402
    _cell_state,
    _local_cell_source_density,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_radial_fluctuation import (  # noqa: E402
    _explicit_geometry_rates,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_stress import (  # noqa: E402
    causal_rest_frame_shear_rate,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_thermal import (  # noqa: E402
    kerr_schild_column_four_velocity,
)


ANALYZED_BASE_COMMIT = "10546da78561ccb4a5f60a203b8b80a47fa26be3"
WORK_PACKAGE = "WP10c9d4b"
SCHEMA_VERSION = 1
THIS_RUNNER = (
    "scripts/run_causal_inner_radial_fluctuation_audit_wp10c9d4b.py"
)
IMPLEMENTATION_SOURCES = (
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_radial_fluctuation.py",
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_full_fluctuation.py",
    "src/imri_qpe/layer3_minidisk_1d/__init__.py",
    THIS_RUNNER,
    "tests/test_causal_inner_radial_fluctuation.py",
    "tests/test_causal_inner_radial_fluctuation_wp10c9d4b.py",
)
WP10C9D4A_CANONICAL = (
    ROOT
    / "results/canonical/"
    "causal_inner_interface_fluctuation_wp10c9d4a/summary.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/tables/"
    "causal_inner_radial_fluctuation_audit_wp10c9d4b.json"
)
DEFAULT_ARRAYS = (
    ROOT
    / "outputs/tables/"
    "causal_inner_radial_fluctuation_audit_wp10c9d4b_arrays.npz"
)
DEFAULT_CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_radial_fluctuation_wp10c9d4b"
)

INNER_RADIUS_RG = 1.8
OUTER_RADIUS_RG = 6.648
GRID_SIZES = (12, 24, 48)
ACTIVE_GUARD_CELLS = 2
MANUFACTURED_AMPLITUDE = 2.0e-3
MANUFACTURED_RAW_DIRECTION = np.asarray(
    [0.15, -0.10, 0.08, 0.12, -0.15],
    dtype=float,
)
REFERENCE_QUADRATURE_ORDER = 12
PATH_QUADRATURE_ORDER = 6
JACOBIAN_N_CELLS = 6
JACOBIAN_RELATIVE_STEPS = (5.0e-6, 1.0e-5, 2.0e-5, 4.0e-5)
JACOBIAN_RELATIVE_STEP = JACOBIAN_RELATIVE_STEPS[-1]
SOURCE_PARTITION_SAMPLES = 25

MAXIMUM_SHARED_CONSERVATIVE_FACE_DEFECT = 1.0e-12
MAXIMUM_LOCAL_BLOCK_LEDGER_DEFECT = 1.0e-12
MAXIMUM_SOURCE_DOUBLE_COUNT_DEFECT = 1.0e-12
MAXIMUM_PATH_PARTITION_DEFECT = 1.0e-12
MAXIMUM_PHYSICAL_SOURCE_PARTITION_DEFECT = 1.0e-7
MAXIMUM_CANDIDATE_JACOBIAN_ACTION_DEFECT = 1.0e-9
MINIMUM_SMOOTH_RADIAL_ORDER = 1.8
MAXIMUM_FINE_RADIAL_RELATIVE_ERROR = 2.0e-3
MINIMUM_PRODUCTION_CANDIDATE_DIFFERENCE = 1.0e-8


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
    return value


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _orders(errors: np.ndarray) -> np.ndarray:
    values = np.asarray(errors, dtype=float)
    return np.log2(values[:-1] / values[1:])


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


class _ManufacturedProfile:
    """Explicit C2 near-horizon chart used by every radial grid."""

    def __init__(
        self,
        inner_radius: float,
        outer_radius: float,
        inner_chart: np.ndarray,
        outer_chart: np.ndarray,
    ) -> None:
        self.inner_radius = float(inner_radius)
        self.outer_radius = float(outer_radius)
        self.inner_chart = np.asarray(inner_chart, dtype=float)
        self.outer_chart = np.asarray(outer_chart, dtype=float)
        stress_scale = max(
            abs(float(self.inner_chart[4])),
            abs(float(self.outer_chart[4])),
            1.0e-14,
        )
        self.direction = MANUFACTURED_RAW_DIRECTION * np.asarray(
            [1.0, 0.1, 0.1, 1.0, stress_scale],
            dtype=float,
        )
        self.log_span = float(
            np.log(self.outer_radius / self.inner_radius)
        )

    def coordinate(self, radius: np.ndarray | float) -> np.ndarray:
        return (
            np.log(np.asarray(radius, dtype=float) / self.inner_radius)
            / self.log_span
        )

    def chart(self, radius: np.ndarray | float) -> np.ndarray:
        coordinate = self.coordinate(radius)
        smoother = coordinate**3 * (
            10.0 - 15.0 * coordinate + 6.0 * coordinate**2
        )
        return (
            self.inner_chart
            + smoother[..., None] * (
                self.outer_chart - self.inner_chart
            )
            + MANUFACTURED_AMPLITUDE
            * np.sin(2.0 * np.pi * coordinate)[..., None]
            * self.direction
        )

    def derivative(self, radius: float) -> np.ndarray:
        radius = float(radius)
        coordinate = float(self.coordinate(radius))
        smoother_derivative = (
            30.0 * coordinate**2 * (1.0 - coordinate) ** 2
        )
        derivative_coordinate = (
            smoother_derivative * (self.outer_chart - self.inner_chart)
            + MANUFACTURED_AMPLITUDE
            * 2.0
            * np.pi
            * np.cos(2.0 * np.pi * coordinate)
            * self.direction
        )
        return derivative_coordinate / (radius * self.log_span)


def _make_context(base_context, profile: _ManufacturedProfile, n_cells: int):
    grid = make_kerr_schild_column_grid(
        profile.inner_radius,
        profile.outer_radius,
        int(n_cells),
        base_context.grid.gravitational_radius,
    )
    return replace(
        base_context,
        grid=grid,
        stream_sources=None,
        outer_boundary_frozen_exterior_chart=np.asarray(
            profile.chart(profile.outer_radius),
            dtype=float,
        ),
    ).validated()


def _reference_lower_source(
    context,
    radius: float,
    chart: np.ndarray,
) -> np.ndarray:
    state = _cell_state(context, radius, chart)
    shear_rate, height_rate = _explicit_geometry_rates(
        context,
        radius,
        chart,
    )
    total, _optical_depth, _components = _local_cell_source_density(
        context,
        state,
        shear_rate=shear_rate,
        height_rate=height_rate,
    )
    return np.asarray(total, dtype=float)


def _independent_radial_reference(
    context,
    profile: _ManufacturedProfile,
) -> np.ndarray:
    """Return high-order cell-integrated continuum residuals."""

    nodes, weights = np.polynomial.legendre.leggauss(
        REFERENCE_QUADRATURE_ORDER
    )
    residual = np.zeros((context.grid.centers.size, 5), dtype=float)
    for cell, (lower_radius, upper_radius) in enumerate(
        zip(context.grid.edges[:-1], context.grid.edges[1:], strict=True)
    ):
        left_chart = np.asarray(profile.chart(lower_radius), dtype=float)
        right_chart = np.asarray(profile.chart(upper_radius), dtype=float)
        left_state = _cell_state(context, float(lower_radius), left_chart)
        right_state = _cell_state(context, float(upper_radius), right_chart)
        residual[cell] = (
            right_state.geometry.face_measure * right_state.flux_over_c
            - left_state.geometry.face_measure * left_state.flux_over_c
        )
        lower_log = float(np.log(lower_radius))
        upper_log = float(np.log(upper_radius))
        midpoint = 0.5 * (lower_log + upper_log)
        half_width = 0.5 * (upper_log - lower_log)
        integrated_source = np.zeros(5, dtype=float)
        for node, weight in zip(nodes, weights, strict=True):
            log_radius = midpoint + half_width * float(node)
            radius = float(np.exp(log_radius))
            chart = np.asarray(profile.chart(radius), dtype=float)
            state = _cell_state(context, radius, chart)
            principal = causal_five_field_coordinate_principal_components(
                context,
                radius,
                chart,
            ).principal_source_matrix @ profile.derivative(radius)
            lower_source = _reference_lower_source(
                context,
                radius,
                chart,
            )
            integrated_source += (
                float(weight)
                * half_width
                * radius
                * state.geometry.face_measure
                * (principal + lower_source)
            )
        residual[cell] -= integrated_source
    return residual


def _total_profile_rates(
    context,
    profile: _ManufacturedProfile,
    radius: float,
) -> tuple[float, float]:
    """Differentiate the full manufactured profile independently."""

    step = 1.0e-5
    minus_radius = radius * np.exp(-step)
    plus_radius = radius * np.exp(step)
    center = _cell_state(context, radius, profile.chart(radius))
    minus = _cell_state(
        context,
        minus_radius,
        profile.chart(minus_radius),
    )
    plus = _cell_state(
        context,
        plus_radius,
        profile.chart(plus_radius),
    )
    radial_width = plus_radius - minus_radius
    lower_minus = (
        minus.geometry.spacetime_metric
        @ kerr_schild_column_four_velocity(
            minus.geometry,
            minus.primitive,
        )
    )
    lower_plus = (
        plus.geometry.spacetime_metric
        @ kerr_schild_column_four_velocity(
            plus.geometry,
            plus.primitive,
        )
    )
    shear = causal_rest_frame_shear_rate(
        center.geometry,
        center.primitive,
        radial_lower_four_velocity_derivative=(
            (lower_plus - lower_minus) / radial_width
        ),
    )
    height_derivative = (
        np.log(plus.thermodynamics.proper_half_thickness)
        - np.log(minus.thermodynamics.proper_half_thickness)
    ) / radial_width
    velocity = kerr_schild_column_four_velocity(
        center.geometry,
        center.primitive,
    )
    return float(shear), float(C * velocity[1] * height_derivative)


def _physical_source_partition_defect(
    context,
    profile: _ManufacturedProfile,
) -> float:
    """Compare total physical rates with principal plus explicit geometry."""

    maximum = 0.0
    radii = np.geomspace(
        profile.inner_radius * 1.001,
        profile.outer_radius / 1.001,
        SOURCE_PARTITION_SAMPLES,
    )
    for radius in radii:
        chart = np.asarray(profile.chart(radius), dtype=float)
        state = _cell_state(context, float(radius), chart)
        explicit_shear, explicit_height = _explicit_geometry_rates(
            context,
            float(radius),
            chart,
        )
        lower, _depth, _components = _local_cell_source_density(
            context,
            state,
            shear_rate=explicit_shear,
            height_rate=explicit_height,
        )
        total_shear, total_height = _total_profile_rates(
            context,
            profile,
            float(radius),
        )
        direct, _depth, _components = _local_cell_source_density(
            context,
            state,
            shear_rate=total_shear,
            height_rate=total_height,
        )
        principal = (
            causal_five_field_coordinate_principal_components(
                context,
                float(radius),
                chart,
            ).principal_source_matrix
            @ profile.derivative(float(radius))
        )
        scale = max(
            float(np.max(np.abs(direct))),
            float(np.max(np.abs(lower + principal))),
            np.finfo(float).tiny,
        )
        maximum = max(
            maximum,
            float(np.max(np.abs(direct - lower - principal)) / scale),
        )
    return maximum


_BLOCK_FIELDS = (
    "conservative_transport_rows",
    "shear_principal_rows",
    "height_principal_rows",
    "local_stress_relaxation_rows",
    "geometry_rows",
    "cooling_rows",
    "stream_rows",
    "lower_height_work_rows",
)


def _candidate_blocks(context, charts: np.ndarray) -> dict[str, np.ndarray]:
    ledger = causal_five_field_radial_candidate_ledger(
        context,
        charts,
        quadrature_order=PATH_QUADRATURE_ORDER,
    )
    return {
        name: np.asarray(getattr(ledger, name), dtype=float).ravel()
        for name in _BLOCK_FIELDS
    }


def _jacobian_audit(
    base_context,
    profile: _ManufacturedProfile,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Build distinct production, candidate-FD, and block-assembled matrices."""

    context = _make_context(base_context, profile, JACOBIAN_N_CELLS)
    charts = np.asarray(profile.chart(context.grid.centers), dtype=float)
    n_unknowns = charts.size
    column_scales = np.tile(
        np.asarray(
            [
                1.0,
                0.1,
                0.1,
                1.0,
                max(float(np.max(np.abs(charts[:, 4]))), 1.0e-14),
            ],
            dtype=float,
        ),
        JACOBIAN_N_CELLS,
    )
    flattened = charts.ravel()

    def candidate_matrices(relative_step: float):
        candidate = np.zeros((n_unknowns, n_unknowns), dtype=float)
        block_values = {
            name: np.zeros_like(candidate) for name in _BLOCK_FIELDS
        }
        for column in range(n_unknowns):
            step = relative_step * column_scales[column]
            plus = np.array(flattened, copy=True)
            minus = np.array(flattened, copy=True)
            plus[column] += step
            minus[column] -= step
            plus_ledger = causal_five_field_radial_candidate_ledger(
                context,
                plus.reshape(JACOBIAN_N_CELLS, 5),
                quadrature_order=PATH_QUADRATURE_ORDER,
            )
            minus_ledger = causal_five_field_radial_candidate_ledger(
                context,
                minus.reshape(JACOBIAN_N_CELLS, 5),
                quadrature_order=PATH_QUADRATURE_ORDER,
            )
            candidate[:, column] = (
                plus_ledger.residual_rows.ravel()
                - minus_ledger.residual_rows.ravel()
            ) / (2.0 * step)
            for name in _BLOCK_FIELDS:
                block_values[name][:, column] = (
                    np.asarray(
                        getattr(plus_ledger, name),
                        dtype=float,
                    ).ravel()
                    - np.asarray(
                        getattr(minus_ledger, name),
                        dtype=float,
                    ).ravel()
                ) / (2.0 * step)
        assembled_values = np.sum(
            np.asarray(list(block_values.values())),
            axis=0,
        )
        scale = max(
            float(np.max(np.abs(candidate))),
            float(np.max(np.abs(assembled_values))),
            np.finfo(float).tiny,
        )
        defect = float(
            np.max(np.abs(candidate - assembled_values)) / scale
        )
        return candidate, assembled_values, block_values, defect

    sweep = []
    selected = None
    previous_candidate = None
    for relative_step in JACOBIAN_RELATIVE_STEPS:
        candidate, assembled, blocks, defect = candidate_matrices(
            relative_step
        )
        change = (
            None
            if previous_candidate is None
            else float(
                np.max(np.abs(candidate - previous_candidate))
                / max(
                    float(np.max(np.abs(candidate))),
                    float(np.max(np.abs(previous_candidate))),
                    np.finfo(float).tiny,
                )
            )
        )
        sweep.append(
            {
                "relative_step": relative_step,
                "candidate_fd_assembled_defect": defect,
                "candidate_change_from_previous_step": change,
            }
        )
        previous_candidate = candidate
        selected = (candidate, assembled, blocks, defect)
    assert selected is not None
    candidate_fd, assembled, block_matrices, closure = selected

    production = np.zeros_like(candidate_fd)
    for column in range(n_unknowns):
        step = JACOBIAN_RELATIVE_STEP * column_scales[column]
        plus = np.array(flattened, copy=True)
        minus = np.array(flattened, copy=True)
        plus[column] += step
        minus[column] -= step
        production[:, column] = (
            causal_five_field_reduced_stationary_residual(plus, context)
            - causal_five_field_reduced_stationary_residual(minus, context)
        ) / (2.0 * step)
    comparison_scale = max(
        float(np.max(np.abs(candidate_fd))),
        float(np.max(np.abs(production))),
        np.finfo(float).tiny,
    )
    production_difference = float(
        np.max(np.abs(candidate_fd - production)) / comparison_scale
    )
    return {
        "n_cells": JACOBIAN_N_CELLS,
        "relative_step_sweep": sweep,
        "selected_relative_step": JACOBIAN_RELATIVE_STEP,
        "candidate_fd_assembled_defect": closure,
        "production_candidate_relative_difference": production_difference,
        "candidate_internal_closure_passed": (
            closure <= MAXIMUM_CANDIDATE_JACOBIAN_ACTION_DEFECT
        ),
        "production_is_distinct_baseline": (
            production_difference
            >= MINIMUM_PRODUCTION_CANDIDATE_DIFFERENCE
        ),
        "production_equality_required": False,
        "assembled_from_explicit_physical_blocks": list(_BLOCK_FIELDS),
    }, {
        "production_stationary_jacobian": production,
        "candidate_fd_stationary_jacobian": candidate_fd,
        "candidate_assembled_stationary_jacobian": assembled,
        **{
            f"candidate_block_jacobian_{name}": matrix
            for name, matrix in block_matrices.items()
        },
    }


def run() -> tuple[dict, dict[str, np.ndarray]]:
    started = time.perf_counter()
    required = (
        WP10C9D4A_CANONICAL,
        wp10c9d0.WP10C8Z_OUTPUT,
        wp10c9d0.WP10C8Z_ARRAYS,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "WP10c9d4b requires prior evidence: " + ", ".join(missing)
        )
    d4a_summary = json.loads(
        WP10C9D4A_CANONICAL.read_text(encoding="utf-8")
    )
    if not d4a_summary["radial_well_balance_audit_authorized"]:
        raise RuntimeError("WP10c9d4a did not authorize radial work")

    patch_arrays = wp10c9d0._load_npz(wp10c9d0.WP10C8Z_ARRAYS)
    configuration = wp10c9d0._patch_configurations(patch_arrays)[
        "N128_exterior_N256_inner_c48"
    ]
    base_context = configuration["context"]
    base_primitives = np.asarray(
        configuration["base_primitives"],
        dtype=float,
    )
    gravitational_radius = base_context.grid.gravitational_radius
    inner_radius = INNER_RADIUS_RG * gravitational_radius
    outer_radius = OUTER_RADIUS_RG * gravitational_radius
    inner_cell = int(
        np.argmin(np.abs(base_context.grid.centers - inner_radius))
    )
    outer_cell = int(
        np.argmin(np.abs(base_context.grid.centers - outer_radius))
    )
    profile = _ManufacturedProfile(
        inner_radius,
        outer_radius,
        base_primitives[inner_cell],
        base_primitives[outer_cell],
    )

    ladder = []
    arrays = {
        "manufactured_inner_chart": profile.inner_chart,
        "manufactured_outer_chart": profile.outer_chart,
        "manufactured_direction": profile.direction,
    }
    errors = []
    maximum_shared = 0.0
    maximum_ledger = 0.0
    maximum_double_count = 0.0
    maximum_path_partition = 0.0
    incoming_excision = 0
    for n_cells in GRID_SIZES:
        context = _make_context(base_context, profile, n_cells)
        charts = np.asarray(profile.chart(context.grid.centers), dtype=float)
        candidate = causal_five_field_radial_candidate_ledger(
            context,
            charts,
            quadrature_order=PATH_QUADRATURE_ORDER,
        )
        reference = _independent_radial_reference(context, profile)
        active = slice(ACTIVE_GUARD_CELLS, -ACTIVE_GUARD_CELLS)
        error_rows = candidate.residual_rows - reference
        relative_error = float(
            np.linalg.norm(error_rows[active])
            / max(
                float(np.linalg.norm(reference[active])),
                np.finfo(float).tiny,
            )
        )
        errors.append(relative_error)
        conservative_telescope = (
            np.sum(candidate.conservative_transport_rows, axis=0)
            - (
                candidate.interfaces.candidate_shared_face_fluxes_over_c[-1]
                - candidate.interfaces.candidate_shared_face_fluxes_over_c[0]
            )
        )
        conservative_scale = max(
            float(
                np.max(
                    np.abs(
                        candidate.interfaces
                        .candidate_shared_face_fluxes_over_c
                    )
                )
            ),
            np.finfo(float).tiny,
        )
        telescope_defect = float(
            np.max(np.abs(conservative_telescope)) / conservative_scale
        )
        path_partition = max(
            max(
                path.source_partition_defect,
                path.principal_closure_defect,
            )
            for path in candidate.within_cell_paths
        )
        maximum_shared = max(
            maximum_shared,
            candidate.interfaces.shared_conservative_face_defect,
            telescope_defect,
        )
        maximum_ledger = max(
            maximum_ledger,
            candidate.local_block_ledger_defect,
        )
        maximum_double_count = max(
            maximum_double_count,
            candidate.source_double_count_defect,
        )
        maximum_path_partition = max(
            maximum_path_partition,
            path_partition,
            candidate.interfaces.maximum_split_closure_defect,
        )
        incoming_excision = max(
            incoming_excision,
            candidate.interfaces.incoming_excision_characteristics,
        )
        ladder.append(
            {
                "n_cells": n_cells,
                "relative_l2_error_active_cells": relative_error,
                "maximum_absolute_error_active_cells": float(
                    np.max(np.abs(error_rows[active]))
                ),
                "shared_conservative_face_defect": (
                    candidate.interfaces.shared_conservative_face_defect
                ),
                "conservative_telescope_defect": telescope_defect,
                "maximum_interface_split_defect": (
                    candidate.interfaces.maximum_split_closure_defect
                ),
                "maximum_within_path_partition_defect": path_partition,
                "local_block_ledger_defect": (
                    candidate.local_block_ledger_defect
                ),
                "source_double_count_defect": (
                    candidate.source_double_count_defect
                ),
                "incoming_excision_characteristics": (
                    candidate.interfaces.incoming_excision_characteristics
                ),
                "minimum_reconstruction_admissibility_factor": float(
                    np.min(candidate.reconstruction.admissibility_factors)
                ),
                "outer_candidate_adjustment_relative_norm": float(
                    np.linalg.norm(
                        candidate.interfaces
                        .conservative_face_adjustments_over_c[-1]
                    )
                    / max(
                        float(
                            np.linalg.norm(
                                candidate.interfaces
                                .production_shared_face_fluxes_over_c[-1]
                            )
                        ),
                        np.finfo(float).tiny,
                    )
                ),
            }
        )
        if n_cells == GRID_SIZES[-1]:
            arrays.update(
                {
                    "fine_grid_edges": context.grid.edges,
                    "fine_grid_centers": context.grid.centers,
                    "fine_primitive_charts": charts,
                    "fine_candidate_residual": candidate.residual_rows,
                    "fine_independent_reference": reference,
                    "fine_residual_error": error_rows,
                    "fine_candidate_shared_fluxes": (
                        candidate.interfaces
                        .candidate_shared_face_fluxes_over_c
                    ),
                    "fine_production_shared_fluxes": (
                        candidate.interfaces
                        .production_shared_face_fluxes_over_c
                    ),
                }
            )

    observed_orders = _orders(np.asarray(errors, dtype=float))
    partition_context = _make_context(
        base_context,
        profile,
        GRID_SIZES[1],
    )
    physical_partition = _physical_source_partition_defect(
        partition_context,
        profile,
    )
    jacobian, jacobian_arrays = _jacobian_audit(base_context, profile)
    arrays.update(jacobian_arrays)
    radial_passed = bool(
        maximum_shared <= MAXIMUM_SHARED_CONSERVATIVE_FACE_DEFECT
        and maximum_ledger <= MAXIMUM_LOCAL_BLOCK_LEDGER_DEFECT
        and maximum_double_count <= MAXIMUM_SOURCE_DOUBLE_COUNT_DEFECT
        and maximum_path_partition <= MAXIMUM_PATH_PARTITION_DEFECT
        and physical_partition
        <= MAXIMUM_PHYSICAL_SOURCE_PARTITION_DEFECT
        and incoming_excision == 0
        and float(np.min(observed_orders)) >= MINIMUM_SMOOTH_RADIAL_ORDER
        and float(errors[-1]) <= MAXIMUM_FINE_RADIAL_RELATIVE_ERROR
        and jacobian["candidate_internal_closure_passed"]
        and jacobian["production_is_distinct_baseline"]
    )
    source_hashes, source_manifest = _source_manifest()
    classification = (
        "radial_five_field_candidate_gate_passed_"
        "frozen_linear_discrimination_authorized"
        if radial_passed
        else "radial_five_field_candidate_gate_failed_"
        "frozen_linear_work_blocked"
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "implementation_parent_commit": ANALYZED_BASE_COMMIT,
        "runner": THIS_RUNNER,
        "classification": classification,
        "radial_candidate_gate_passed": radial_passed,
        "wp10c9d5_frozen_linear_discrimination_authorized": radial_passed,
        "production_operator_authorized": False,
        "nonlinear_candidate_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "gates": {
            "maximum_shared_conservative_face_defect": (
                MAXIMUM_SHARED_CONSERVATIVE_FACE_DEFECT
            ),
            "maximum_local_block_ledger_defect": (
                MAXIMUM_LOCAL_BLOCK_LEDGER_DEFECT
            ),
            "maximum_source_double_count_defect": (
                MAXIMUM_SOURCE_DOUBLE_COUNT_DEFECT
            ),
            "maximum_path_partition_defect": (
                MAXIMUM_PATH_PARTITION_DEFECT
            ),
            "maximum_physical_source_partition_defect": (
                MAXIMUM_PHYSICAL_SOURCE_PARTITION_DEFECT
            ),
            "maximum_candidate_jacobian_action_defect": (
                MAXIMUM_CANDIDATE_JACOBIAN_ACTION_DEFECT
            ),
            "minimum_smooth_radial_order": MINIMUM_SMOOTH_RADIAL_ORDER,
            "maximum_fine_radial_relative_error": (
                MAXIMUM_FINE_RADIAL_RELATIVE_ERROR
            ),
            "minimum_production_candidate_difference": (
                MINIMUM_PRODUCTION_CANDIDATE_DIFFERENCE
            ),
            "incoming_excision_characteristics": 0,
        },
        "manufactured_family": {
            "type": (
                "explicit C2 nonequilibrium primitive-chart profile; "
                "independent high-order physical residual"
            ),
            "exact_equilibrium_claimed": False,
            "residual_subtraction_used": False,
            "inner_radius_rg": INNER_RADIUS_RG,
            "outer_radius_rg": OUTER_RADIUS_RG,
            "amplitude": MANUFACTURED_AMPLITUDE,
            "raw_direction": MANUFACTURED_RAW_DIRECTION,
            "reference_quadrature_order": REFERENCE_QUADRATURE_ORDER,
            "active_guard_cells": ACTIVE_GUARD_CELLS,
        },
        "radial_ladder": ladder,
        "observed_orders": observed_orders,
        "minimum_observed_order": float(np.min(observed_orders)),
        "fine_relative_l2_error": float(errors[-1]),
        "maximum_shared_conservative_or_telescope_defect": maximum_shared,
        "maximum_local_block_ledger_defect": maximum_ledger,
        "maximum_source_double_count_defect": maximum_double_count,
        "maximum_path_or_interface_partition_defect": (
            maximum_path_partition
        ),
        "physical_source_partition_defect": physical_partition,
        "maximum_incoming_excision_characteristics": incoming_excision,
        "jacobian_audit": jacobian,
        "one_shared_conservative_face_flux": True,
        "actual_nonuniform_radial_measures": True,
        "shear_and_height_principal_terms_separate": True,
        "local_stress_relaxation_separate": True,
        "geometry_cooling_stream_and_lower_height_separate": True,
        "production_default_changed": False,
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
    decisive_names = {
        "manufactured_inner_chart",
        "manufactured_outer_chart",
        "manufactured_direction",
        "fine_grid_edges",
        "fine_grid_centers",
        "fine_primitive_charts",
        "fine_candidate_residual",
        "fine_independent_reference",
        "fine_residual_error",
        "fine_candidate_shared_fluxes",
        "fine_production_shared_fluxes",
        "production_stationary_jacobian",
        "candidate_fd_stationary_jacobian",
        "candidate_assembled_stationary_jacobian",
    }
    decisive_arrays = {
        name: values
        for name, values in arrays.items()
        if name in decisive_names
        or name.startswith("candidate_block_jacobian_")
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
            "inner_radius_rg": INNER_RADIUS_RG,
            "outer_radius_rg": OUTER_RADIUS_RG,
            "grid_sizes": GRID_SIZES,
            "active_guard_cells": ACTIVE_GUARD_CELLS,
            "manufactured_amplitude": MANUFACTURED_AMPLITUDE,
            "manufactured_raw_direction": MANUFACTURED_RAW_DIRECTION,
            "reference_quadrature_order": REFERENCE_QUADRATURE_ORDER,
            "path_quadrature_order": PATH_QUADRATURE_ORDER,
            "jacobian_n_cells": JACOBIAN_N_CELLS,
            "jacobian_relative_step": JACOBIAN_RELATIVE_STEP,
            "jacobian_relative_step_sweep": JACOBIAN_RELATIVE_STEPS,
            "gates": payload["gates"],
        },
    )
    _write_json(
        directory / "provenance.json",
        {
            "work_package": WORK_PACKAGE,
            "scientific_status": (
                "CERTIFIED"
                if payload["radial_candidate_gate_passed"]
                else "FAILED"
            ),
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
                "scripts/run_causal_inner_radial_fluctuation_audit_"
                "wp10c9d4b.py"
            ),
            "establishes": (
                "The radial production-neutral five-field candidate, "
                "physical block ledger, manufactured balance, boundary "
                "causality, and candidate FD/assembled Jacobian gates stated "
                "by the summary classification."
            ),
            "does_not_establish": (
                "Frozen-linear export convergence, a nonlinear path solver, "
                "production promotion, fixed-Q averaging, or reduced slow "
                "evolution."
            ),
        },
    )
    checksum_paths = (
        directory / "config.json",
        directory / "decisive_arrays.npz",
        directory / "provenance.json",
        directory / "summary.json",
    )
    (directory / "SHA256SUMS.txt").write_text(
        "\n".join(
            f"{_sha256(path)}  {path.name}" for path in checksum_paths
        )
        + "\n",
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
                "radial_candidate_gate_passed": payload[
                    "radial_candidate_gate_passed"
                ],
                "minimum_observed_order": payload[
                    "minimum_observed_order"
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
