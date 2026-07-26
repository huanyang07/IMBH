"""Run the WP10c8v localized inner-phase spatial preflight.

WP10c8u showed that the shell-0 hidden-mode response is localized but that
its late instantaneous phase and amplitude are not N64/N128 converged.  This
package asks whether that failure is ordinary inner-grid phase dispersion.
It builds buffered local copies of the production causal operator on the
first three eighths of the logarithmic domain, validates their active
first-quarter blocks and histories against the committed full N64/N128
tangent operators, and adds one N256-equivalent factor-two local refinement.

The artificial local outer boundary uses one frozen exterior Rusanov trace.
It is audit-only.  Results are binding only inside the nested active core,
which excludes eight N64 cells (and the corresponding sixteen/thirty-two
fine cells) next to that boundary.  The package propagates the linearized
evolving
descriptor, not another nonlinear truth trajectory.  Consequently a passing
result can authorize a future independently equilibrated local/fixed-Q truth
test; it cannot certify a formal fast average or select a reduced
architecture.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np
import scipy
from scipy.interpolate import PchipInterpolator
from scipy.signal import hilbert
from scipy.sparse.linalg import expm_multiply

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_mode_phase_average_audit_wp10c8u as wp10c8u
import run_causal_moment_sufficiency_audit_wp10c8i as wp10c8i
import run_causal_spatial_balance_adaptive_wp10c7k as wp10c7k

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    KerrSchildCellSourceRates,
    causal_five_field_evolving_tangent_matrices,
    causal_five_field_face_flux_decomposition,
    causal_five_field_reconstruct_face_charts,
    causal_five_field_state_from_primitives,
    evaluate_causal_five_field_dae,
    load_causal_five_field_adaptive_bdf2_restart,
    make_kerr_schild_column_grid,
    pack_causal_five_field_state,
    unpack_causal_five_field_state,
)


BASE_COMMIT = "3ccdb9532359acbaa197e066a800a9119dfe60ef"
WORK_PACKAGE = "WP10c8v"
SCHEMA_VERSION = 1
THIS_RUNNER = (
    "scripts/run_causal_inner_phase_spatial_preflight_wp10c8v.py"
)
CORE_DAE_FILE = (
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_dae_system.py"
)
CORE_TANGENT_FILE = (
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_spatial_audit.py"
)

GLOBAL_EQUIVALENT_MESHES = (64, 128, 256)
PARENT_MESHES = (64, 128)
LOCAL_DOMAIN_FRACTION = 0.375
ACTIVE_DOMAIN_FRACTION = 2.0 / 3.0
TARGET_SECONDS = 0.125
TIME_SAMPLES = 201
REFINED_TIME_SAMPLES = 401

FINITE_DIFFERENCE_STEP = 2.0e-6
DESCRIPTOR_TIMESTEP_SECONDS = 1.0
STORAGE_DIFFERENCE_STEP = 1.0e-4
STORAGE_RATE_DERIVATIVE_STEP = 2.0e-6
STORAGE_QUADRATURE_ORDER = 4
STORAGE_DIRECTIONAL_STEP = 1.0e-3
TERM_DIRECTIONAL_STEP = 2.0e-5

MAXIMUM_ACTIVE_OPERATOR_RELATIVE_DEFECT = 1.0e-4
MAXIMUM_FULL_LOCAL_HISTORY_RELATIVE_DEFECT = 2.0e-2
MINIMUM_FULL_LOCAL_HISTORY_SIGNED_COSINE = 0.995
MAXIMUM_TEMPORAL_REFINEMENT_DEFECT = 5.0e-10
MINIMUM_SPATIAL_CONTRACTION_ORDER = 0.75
MINIMUM_SAME_TIME_SIGNED_COSINE = 0.90
MAXIMUM_ZERO_CROSSING_RELATIVE_DEFECT = 0.10
MAXIMUM_FREQUENCY_RELATIVE_DEFECT = 0.10
MAXIMUM_DAMPING_RELATIVE_DEFECT = 0.25
MINIMUM_SIGNIFICANT_SIGNAL = 1.0e-8

CHECKPOINT_DIRECTORY = (
    ROOT / "outputs/checkpoints/causal_five_field_wp10c8v"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/tables/"
    "causal_inner_phase_spatial_preflight_wp10c8v.json"
)
DEFAULT_ARRAYS = (
    ROOT
    / "outputs/tables/"
    "causal_inner_phase_spatial_preflight_wp10c8v_arrays.npz"
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
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _operator_path(n_cells: int) -> Path:
    return (
        ROOT
        / "outputs/checkpoints/causal_five_field_wp10c8i/"
        f"N{n_cells:03d}_t_0p025_moment_operators.npz"
    )


def _dense_path(n_cells: int, side: str) -> Path:
    return wp10c8u._dense_cache_paths(
        n_cells,
        "fine",
        side,
    )[1]


def _local_operator_paths(
    global_equivalent_cells: int,
) -> tuple[Path, Path]:
    stem = f"N{global_equivalent_cells:03d}_local_evolving_operator"
    return (
        CHECKPOINT_DIRECTORY / f"{stem}.json",
        CHECKPOINT_DIRECTORY / f"{stem}_arrays.npz",
    )


def _local_cell_count(global_equivalent_cells: int) -> int:
    cells = int(round(global_equivalent_cells * LOCAL_DOMAIN_FRACTION))
    if cells * 8 != 3 * global_equivalent_cells:
        raise ValueError(
            "WP10c8v requires a three-eighth-domain nested mesh"
        )
    return cells


def _active_cell_count(global_equivalent_cells: int) -> int:
    active = int(round(global_equivalent_cells * 0.25))
    if active * 4 != global_equivalent_cells:
        raise ValueError("WP10c8v active core is not nested")
    return active


def _zero_sources(n_cells: int) -> KerrSchildCellSourceRates:
    zeros = np.zeros(int(n_cells), dtype=float)
    return KerrSchildCellSourceRates(
        rest_mass=np.array(zeros, copy=True),
        radial_momentum_over_c=np.array(zeros, copy=True),
        angular_momentum_over_c=np.array(zeros, copy=True),
        killing_energy_over_c2=np.array(zeros, copy=True),
    )


def _resample_columns(
    source_radius: np.ndarray,
    source_values: np.ndarray,
    target_radius: np.ndarray,
) -> np.ndarray:
    source_r = np.asarray(source_radius, dtype=float)
    target_r = np.asarray(target_radius, dtype=float)
    values = np.asarray(source_values, dtype=float)
    if (
        source_r.ndim != 1
        or target_r.ndim != 1
        or values.shape[0] != source_r.size
        or source_r.size < 4
        or np.any(source_r <= 0.0)
        or np.any(target_r <= 0.0)
        or np.any(~np.isfinite(values))
    ):
        raise ValueError("invalid WP10c8v resampling inputs")
    flat = values.reshape(source_r.size, -1)
    result = np.empty((target_r.size, flat.shape[1]), dtype=float)
    source_x = np.log(source_r)
    target_x = np.log(target_r)
    for column in range(flat.shape[1]):
        spline = PchipInterpolator(
            source_x,
            flat[:, column],
            extrapolate=True,
        )
        result[:, column] = spline(target_x)
    return result.reshape((target_r.size,) + values.shape[1:])


def _parent_bundle(n_cells: int) -> dict:
    context = wp10c7k._context(n_cells)
    restart_path = wp10c8i._t_0p025_path(n_cells)
    restart = load_causal_five_field_adaptive_bdf2_restart(
        restart_path,
        context,
    )
    operator_path = _operator_path(n_cells)
    operator = _load_npz(operator_path)
    minus_path = _dense_path(n_cells, "minus")
    plus_path = _dense_path(n_cells, "plus")
    minus = _load_npz(minus_path)
    plus = _load_npz(plus_path)
    if not (
        restart.elapsed_time == 0.025
        and np.array_equal(minus["times"], plus["times"])
        and minus["times"][0] == 0.0
    ):
        raise RuntimeError("WP10c8v parent anchor contract differs")
    state = unpack_causal_five_field_state(
        restart.state_vector,
        n_cells,
    )
    half_difference = 0.5 * (
        np.asarray(plus["primitives"][0], dtype=float)
        - np.asarray(minus["primitives"][0], dtype=float)
    )
    physical_rate = (
        np.asarray(operator["scaled_primitive_rate"], dtype=float)
        * np.asarray(operator["primitive_column_scales"], dtype=float)
    ).reshape(n_cells, 5)
    return {
        "context": context,
        "state_vector": np.asarray(restart.state_vector, dtype=float),
        "primitives": np.asarray(state.primitives, dtype=float),
        "physical_rate_per_s": physical_rate,
        "primitive_column_scales": np.asarray(
            operator["primitive_column_scales"],
            dtype=float,
        ).reshape(n_cells, 5),
        "physical_input_amplitudes": np.asarray(
            operator["physical_input_amplitudes"],
            dtype=float,
        ).reshape(n_cells, 5),
        "half_difference": half_difference,
        "dynamic": np.asarray(operator["dynamic"], dtype=float),
        "restart_path": restart_path,
        "operator_path": operator_path,
        "minus_path": minus_path,
        "plus_path": plus_path,
    }


def _nested_local_grid(
    reference_context,
    global_equivalent_cells: int,
):
    local_cells = _local_cell_count(global_equivalent_cells)
    full_ratio = (
        float(reference_context.grid.edges[-1])
        / float(reference_context.grid.edges[0])
    )
    outer_radius = float(reference_context.grid.edges[0]) * (
        full_ratio**LOCAL_DOMAIN_FRACTION
    )
    return make_kerr_schild_column_grid(
        float(reference_context.grid.edges[0]),
        outer_radius,
        local_cells,
        float(reference_context.grid.gravitational_radius),
    )


def _extended_grid(
    reference_context,
    global_equivalent_cells: int,
    extra_cells: int = 3,
):
    local_cells = _local_cell_count(global_equivalent_cells)
    full_ratio = (
        float(reference_context.grid.edges[-1])
        / float(reference_context.grid.edges[0])
    )
    full_log_step = np.log(full_ratio) / global_equivalent_cells
    outer_radius = float(reference_context.grid.edges[0]) * np.exp(
        (local_cells + extra_cells) * full_log_step
    )
    return make_kerr_schild_column_grid(
        float(reference_context.grid.edges[0]),
        outer_radius,
        local_cells + extra_cells,
        float(reference_context.grid.gravitational_radius),
    )


def _base_profiles(
    global_equivalent_cells: int,
    parents: dict[int, dict],
) -> dict:
    reference = parents[128]
    reference_context = reference["context"]
    grid = _nested_local_grid(
        reference_context,
        global_equivalent_cells,
    )
    local_cells = _local_cell_count(global_equivalent_cells)
    if global_equivalent_cells in PARENT_MESHES:
        parent = parents[global_equivalent_cells]
        if not np.allclose(
            parent["context"].grid.edges[: local_cells + 1],
            grid.edges,
            rtol=2.0e-14,
            atol=0.0,
        ):
            raise RuntimeError("WP10c8v nested parent grid differs")
        primitives = np.asarray(
            parent["primitives"][:local_cells],
            dtype=float,
        )
        physical_rate = np.asarray(
            parent["physical_rate_per_s"][:local_cells],
            dtype=float,
        )
        native_half_difference = np.asarray(
            parent["half_difference"][:local_cells],
            dtype=float,
        )
        input_amplitudes = np.asarray(
            parent["physical_input_amplitudes"][:local_cells],
            dtype=float,
        )
        reconstruction = causal_five_field_reconstruct_face_charts(
            parent["context"],
            parent["primitives"],
        )
        frozen_exterior = np.asarray(
            reconstruction.right_face_charts[local_cells],
            dtype=float,
        )
    else:
        source_radius = np.asarray(
            reference_context.grid.centers,
            dtype=float,
        )
        primitives = _resample_columns(
            source_radius,
            reference["primitives"],
            grid.centers,
        )
        physical_rate = _resample_columns(
            source_radius,
            reference["physical_rate_per_s"],
            grid.centers,
        )
        native_half_difference = _resample_columns(
            source_radius,
            reference["half_difference"],
            grid.centers,
        )
        input_amplitudes = np.abs(
            _resample_columns(
                source_radius,
                reference["physical_input_amplitudes"],
                grid.centers,
            )
        )
        extended_grid = _extended_grid(
            reference_context,
            global_equivalent_cells,
        )
        extended_primitives = _resample_columns(
            source_radius,
            reference["primitives"],
            extended_grid.centers,
        )
        extended_context = replace(
            reference_context,
            grid=extended_grid,
            stream_sources=_zero_sources(extended_grid.centers.size),
            outer_boundary_flux_mode="frozen_exterior_rusanov",
            outer_boundary_frozen_exterior_chart=np.array(
                extended_primitives[-1],
                copy=True,
            ),
        ).validated()
        reconstruction = causal_five_field_reconstruct_face_charts(
            extended_context,
            extended_primitives,
        )
        frozen_exterior = np.asarray(
            reconstruction.right_face_charts[local_cells],
            dtype=float,
        )
    if np.any(input_amplitudes <= 0.0):
        raise RuntimeError("WP10c8v input amplitudes are not positive")
    matched_half_difference = _resample_columns(
        np.asarray(reference_context.grid.centers, dtype=float),
        reference["half_difference"],
        grid.centers,
    )
    return {
        "grid": grid,
        "primitives": primitives,
        "physical_rate_per_s": physical_rate,
        "native_half_difference": native_half_difference,
        "matched_half_difference": matched_half_difference,
        "physical_input_amplitudes": input_amplitudes,
        "frozen_exterior_chart": frozen_exterior,
    }


def _local_context(reference_context, profiles: dict):
    full_sources = reference_context.stream_sources
    local_cells = int(profiles["grid"].centers.size)
    if full_sources is not None:
        source_matrix = np.asarray(full_sources.matrix, dtype=float)
        if np.any(source_matrix[: min(local_cells, source_matrix.shape[0])]):
            raise RuntimeError("WP10c8v inner domain contains stream source")
    return replace(
        reference_context,
        grid=profiles["grid"],
        stream_sources=_zero_sources(local_cells),
        outer_boundary_flux_mode="frozen_exterior_rusanov",
        outer_boundary_frozen_exterior_chart=np.array(
            profiles["frozen_exterior_chart"],
            copy=True,
        ),
    ).validated()


def _operator_cache_contract(
    global_equivalent_cells: int,
    parents: dict[int, dict],
) -> dict:
    parent_mesh = min(global_equivalent_cells, 128)
    parent = parents[parent_mesh]
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "base_commit": BASE_COMMIT,
        "global_equivalent_cells": global_equivalent_cells,
        "local_cells": _local_cell_count(global_equivalent_cells),
        "active_cells": _active_cell_count(global_equivalent_cells),
        "parent_restart_sha256": _sha256(parent["restart_path"]),
        "parent_operator_sha256": _sha256(parent["operator_path"]),
        "parent_minus_dense_sha256": _sha256(parent["minus_path"]),
        "parent_plus_dense_sha256": _sha256(parent["plus_path"]),
        "core_dae_sha256": _sha256(ROOT / CORE_DAE_FILE),
        "core_tangent_sha256": _sha256(ROOT / CORE_TANGENT_FILE),
        "outer_boundary_flux_mode": "frozen_exterior_rusanov",
        "finite_difference_step": FINITE_DIFFERENCE_STEP,
        "descriptor_timestep_seconds": DESCRIPTOR_TIMESTEP_SECONDS,
        "storage_difference_step": STORAGE_DIFFERENCE_STEP,
        "storage_rate_derivative_step": (
            STORAGE_RATE_DERIVATIVE_STEP
        ),
        "storage_quadrature_order": STORAGE_QUADRATURE_ORDER,
        "storage_directional_step": STORAGE_DIRECTIONAL_STEP,
    }


def _load_cached_local_operator(
    global_equivalent_cells: int,
    parents: dict[int, dict],
) -> tuple[dict, dict[str, np.ndarray]] | None:
    json_path, arrays_path = _local_operator_paths(
        global_equivalent_cells
    )
    if not (json_path.exists() and arrays_path.exists()):
        return None
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    expected = _operator_cache_contract(
        global_equivalent_cells,
        parents,
    )
    if not all(payload.get(key) == value for key, value in expected.items()):
        return None
    if payload.get("arrays_sha256") != _sha256(arrays_path):
        return None
    return payload, _load_npz(arrays_path)


def _build_local_operator(
    global_equivalent_cells: int,
    parents: dict[int, dict],
    *,
    force: bool,
) -> tuple[dict, dict[str, np.ndarray], dict]:
    cached = (
        None
        if force
        else _load_cached_local_operator(
            global_equivalent_cells,
            parents,
        )
    )
    profiles = _base_profiles(global_equivalent_cells, parents)
    context = _local_context(parents[128]["context"], profiles)
    if cached is not None:
        payload, arrays = cached
        return payload, arrays, {
            "context": context,
            "profiles": profiles,
        }

    started = time.perf_counter()
    state = causal_five_field_state_from_primitives(
        context,
        profiles["primitives"],
    )
    vector = pack_causal_five_field_state(state)
    evolving = causal_five_field_evolving_tangent_matrices(
        context,
        vector,
        primitive_rate_per_s=profiles["physical_rate_per_s"],
        finite_difference_step=FINITE_DIFFERENCE_STEP,
        descriptor_timestep_seconds=DESCRIPTOR_TIMESTEP_SECONDS,
        storage_difference_step=STORAGE_DIFFERENCE_STEP,
        storage_rate_derivative_step=STORAGE_RATE_DERIVATIVE_STEP,
        storage_quadrature_order=STORAGE_QUADRATURE_ORDER,
        storage_directional_step=STORAGE_DIRECTIONAL_STEP,
    )
    arrays = {
        "generator": np.asarray(
            evolving["evolving_scaled_generator_per_s"],
            dtype=float,
        ),
        "descriptor": np.asarray(
            evolving["descriptor_reduced_scaled_matrix"],
            dtype=float,
        ),
        "stationary_jacobian": np.asarray(
            evolving["stationary_reduced_scaled_jacobian"],
            dtype=float,
        ),
        "storage_rate_derivative": np.asarray(
            evolving["storage_rate_derivative_scaled_matrix"],
            dtype=float,
        ),
        "primitive_column_scales": np.asarray(
            evolving["primitive_column_scales"],
            dtype=float,
        ),
        "conservation_row_scales": np.asarray(
            evolving["conservation_row_scales"],
            dtype=float,
        ),
        "base_primitives": np.asarray(
            profiles["primitives"],
            dtype=float,
        ),
        "base_physical_rate_per_s": np.asarray(
            profiles["physical_rate_per_s"],
            dtype=float,
        ),
        "native_half_difference": np.asarray(
            profiles["native_half_difference"],
            dtype=float,
        ),
        "matched_half_difference": np.asarray(
            profiles["matched_half_difference"],
            dtype=float,
        ),
        "physical_input_amplitudes": np.asarray(
            profiles["physical_input_amplitudes"],
            dtype=float,
        ),
        "frozen_exterior_chart": np.asarray(
            profiles["frozen_exterior_chart"],
            dtype=float,
        ),
        "radius_rg": np.asarray(
            context.grid.centers / context.grid.gravitational_radius,
            dtype=float,
        ),
        "grid_edges_rg": np.asarray(
            context.grid.edges / context.grid.gravitational_radius,
            dtype=float,
        ),
        "cell_measures": np.asarray(
            context.grid.cell_measures,
            dtype=float,
        ),
    }
    contract = _operator_cache_contract(
        global_equivalent_cells,
        parents,
    )
    json_path, arrays_path = _local_operator_paths(
        global_equivalent_cells
    )
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **arrays)
    payload = {
        **contract,
        "producer_runner": THIS_RUNNER,
        "producer_runner_sha256": _sha256(ROOT / THIS_RUNNER),
        "arrays_path": _relative(arrays_path),
        "arrays_sha256": _sha256(arrays_path),
        "maximum_scaled_generator_factorization_defect": float(
            evolving["maximum_scaled_generator_factorization_defect"]
        ),
        "maximum_relative_storage_action_defect": float(
            evolving["maximum_relative_storage_action_defect"]
        ),
        "wall_seconds": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    json_path.write_text(
        json.dumps(
            _plain(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return payload, arrays, {
        "context": context,
        "profiles": profiles,
    }


def _similarity_rescale_generator(
    generator: np.ndarray,
    source_scales: np.ndarray,
    target_scales: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(generator, dtype=float)
    source = np.asarray(source_scales, dtype=float).ravel()
    target = np.asarray(target_scales, dtype=float).ravel()
    if (
        matrix.shape != (source.size, source.size)
        or target.shape != source.shape
        or np.any(source <= 0.0)
        or np.any(target <= 0.0)
    ):
        raise ValueError("invalid generator rescaling inputs")
    return (
        (source / target)[:, None]
        * matrix
        * (target / source)[None, :]
    )


def _active_operator_reproduction(
    global_equivalent_cells: int,
    parent: dict,
    local: dict[str, np.ndarray],
) -> dict:
    active_size = 5 * _active_cell_count(global_equivalent_cells)
    local_size = 5 * _local_cell_count(global_equivalent_cells)
    full_scales = np.asarray(
        parent["primitive_column_scales"],
        dtype=float,
    ).ravel()[:local_size]
    local_in_full_scale = _similarity_rescale_generator(
        local["generator"],
        local["primitive_column_scales"],
        full_scales,
    )
    reference = np.asarray(parent["dynamic"], dtype=float)[
        :active_size,
        :active_size,
    ]
    candidate = local_in_full_scale[:active_size, :active_size]
    difference = candidate - reference
    denominator = max(
        float(np.linalg.norm(reference)),
        np.finfo(float).tiny,
    )
    maximum_scale = max(
        float(np.max(np.abs(reference))),
        np.finfo(float).tiny,
    )
    relative = float(np.linalg.norm(difference) / denominator)
    maximum_relative = float(
        np.max(np.abs(difference)) / maximum_scale
    )
    return {
        "relative_frobenius_defect": relative,
        "maximum_relative_entry_defect": maximum_relative,
        "maximum_absolute_entry_defect": float(
            np.max(np.abs(difference))
        ),
        "passed": (
            relative <= MAXIMUM_ACTIVE_OPERATOR_RELATIVE_DEFECT
            and maximum_relative
            <= MAXIMUM_ACTIVE_OPERATOR_RELATIVE_DEFECT
        ),
    }


def _continuum_norm(
    values: np.ndarray,
    cell_measures: np.ndarray,
) -> float:
    fields = np.asarray(values, dtype=float)
    weights = np.asarray(cell_measures, dtype=float)
    if fields.shape != (weights.size, 5):
        raise ValueError("invalid continuum norm inputs")
    normalized_weights = weights / np.sum(weights)
    return float(
        np.sqrt(
            np.sum(normalized_weights[:, None] * fields**2) / 5.0
        )
    )


def _propagate(
    arrays: dict[str, np.ndarray],
    *,
    time_samples: int,
) -> dict[str, np.ndarray]:
    scales = np.asarray(
        arrays["primitive_column_scales"],
        dtype=float,
    ).reshape(-1, 5)
    amplitudes = np.asarray(
        arrays["physical_input_amplitudes"],
        dtype=float,
    )
    generator = _similarity_rescale_generator(
        arrays["generator"],
        scales,
        amplitudes,
    )
    initial = (
        np.asarray(arrays["matched_half_difference"], dtype=float)
        / amplitudes
    )
    norm = _continuum_norm(initial, arrays["cell_measures"])
    if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
        raise RuntimeError("WP10c8v matched initial mode is zero")
    initial = initial / norm
    times = np.linspace(0.0, TARGET_SECONDS, time_samples)
    state = np.asarray(
        expm_multiply(
            generator,
            initial.ravel(),
            start=0.0,
            stop=TARGET_SECONDS,
            num=time_samples,
            endpoint=True,
        ),
        dtype=float,
    ).reshape(time_samples, -1, 5)
    rate = np.asarray(
        [generator @ row.ravel() for row in state],
        dtype=float,
    ).reshape(state.shape)
    active = _active_cell_count(
        int(round(arrays["radius_rg"].size / LOCAL_DOMAIN_FRACTION))
    )
    weights = np.asarray(arrays["cell_measures"][:active], dtype=float)
    normalized_weights = weights / np.sum(weights)
    stress_state = np.sum(
        normalized_weights[None, :] * state[:, :active, 4],
        axis=1,
    )
    stress_rate = np.sum(
        normalized_weights[None, :] * rate[:, :active, 4],
        axis=1,
    )
    radius = np.asarray(arrays["radius_rg"][:active], dtype=float)
    rate_activity = np.linalg.norm(rate[:, :active], axis=2)
    activity_weights = (
        weights[None, :] * rate_activity
    )
    activity_denominator = np.maximum(
        np.sum(activity_weights, axis=1),
        np.finfo(float).tiny,
    )
    radial_centroid = (
        np.sum(activity_weights * radius[None, :], axis=1)
        / activity_denominator
    )
    radial_width = np.sqrt(
        np.sum(
            activity_weights
            * (radius[None, :] - radial_centroid[:, None]) ** 2,
            axis=1,
        )
        / activity_denominator
    )
    return {
        "times": times,
        "state": state,
        "rate": rate,
        "stress_state_signal": stress_state,
        "stress_rate_signal": stress_rate,
        "rate_activity_radial_centroid_rg": radial_centroid,
        "rate_activity_radial_width_rg": radial_width,
        "generator_in_amplitude_scale": generator,
        "initial_continuum_norm": np.asarray(norm),
    }


def _propagate_full_parent(
    n_cells: int,
    parent: dict,
    reference: dict,
) -> dict[str, np.ndarray]:
    amplitudes = np.asarray(
        parent["physical_input_amplitudes"],
        dtype=float,
    )
    generator = _similarity_rescale_generator(
        parent["dynamic"],
        parent["primitive_column_scales"],
        amplitudes,
    )
    matched = _resample_columns(
        reference["context"].grid.centers,
        reference["half_difference"],
        parent["context"].grid.centers,
    )
    initial = matched / amplitudes
    local_cells = _local_cell_count(n_cells)
    norm = _continuum_norm(
        initial[:local_cells],
        parent["context"].grid.cell_measures[:local_cells],
    )
    if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
        raise RuntimeError("WP10c8v full-parent mode is zero")
    initial /= norm
    times = np.linspace(0.0, TARGET_SECONDS, TIME_SAMPLES)
    state = np.asarray(
        expm_multiply(
            generator,
            initial.ravel(),
            start=0.0,
            stop=TARGET_SECONDS,
            num=TIME_SAMPLES,
            endpoint=True,
        ),
        dtype=float,
    ).reshape(TIME_SAMPLES, n_cells, 5)
    rate = np.asarray(
        [generator @ row.ravel() for row in state],
        dtype=float,
    ).reshape(state.shape)
    return {
        "times": times,
        "state": state,
        "rate": rate,
    }


def _same_grid_history_reproduction(
    full: dict[str, np.ndarray],
    local: dict[str, np.ndarray],
    cell_measures: np.ndarray,
    active_cells: int,
) -> dict:
    weights = np.asarray(cell_measures[:active_cells], dtype=float)
    weights /= np.sum(weights)

    def metrics(name: str) -> dict:
        reference = np.asarray(full[name], dtype=float)[:, :active_cells]
        candidate = np.asarray(local[name], dtype=float)[:, :active_cells]
        difference = candidate - reference
        reference_norm = np.sqrt(
            np.sum(
                weights[None, :, None] * reference**2,
                axis=(1, 2),
            )
        )
        candidate_norm = np.sqrt(
            np.sum(
                weights[None, :, None] * candidate**2,
                axis=(1, 2),
            )
        )
        difference_norm = np.sqrt(
            np.sum(
                weights[None, :, None] * difference**2,
                axis=(1, 2),
            )
        )
        cosine = np.sum(
            weights[None, :, None] * reference * candidate,
            axis=(1, 2),
        ) / np.maximum(
            reference_norm * candidate_norm,
            np.finfo(float).tiny,
        )
        relative = difference_norm / np.maximum(
            reference_norm,
            np.finfo(float).tiny,
        )
        return {
            "maximum_relative_l2_defect": float(np.max(relative)),
            "final_relative_l2_defect": float(relative[-1]),
            "minimum_signed_cosine": float(np.min(cosine)),
            "final_signed_cosine": float(cosine[-1]),
        }

    state = metrics("state")
    rate = metrics("rate")
    passed = bool(
        max(
            state["maximum_relative_l2_defect"],
            rate["maximum_relative_l2_defect"],
        )
        <= MAXIMUM_FULL_LOCAL_HISTORY_RELATIVE_DEFECT
        and min(
            state["minimum_signed_cosine"],
            rate["minimum_signed_cosine"],
        )
        >= MINIMUM_FULL_LOCAL_HISTORY_SIGNED_COSINE
    )
    return {
        "state": state,
        "rate": rate,
        "passed": passed,
    }


def _temporal_refinement(
    coarse: dict[str, np.ndarray],
    refined: dict[str, np.ndarray],
) -> dict:
    if not np.array_equal(coarse["times"], refined["times"][::2]):
        raise RuntimeError("WP10c8v temporal grids are not nested")
    state_difference = (
        np.asarray(coarse["state"], dtype=float)
        - np.asarray(refined["state"], dtype=float)[::2]
    )
    rate_difference = (
        np.asarray(coarse["rate"], dtype=float)
        - np.asarray(refined["rate"], dtype=float)[::2]
    )
    state_scale = max(
        float(np.max(np.abs(refined["state"]))),
        np.finfo(float).tiny,
    )
    rate_scale = max(
        float(np.max(np.abs(refined["rate"]))),
        np.finfo(float).tiny,
    )
    defect = max(
        float(np.max(np.abs(state_difference)) / state_scale),
        float(np.max(np.abs(rate_difference)) / rate_scale),
    )
    return {
        "maximum_relative_state_defect": float(
            np.max(np.abs(state_difference)) / state_scale
        ),
        "maximum_relative_rate_defect": float(
            np.max(np.abs(rate_difference)) / rate_scale
        ),
        "maximum_relative_defect": defect,
        "passed": defect <= MAXIMUM_TEMPORAL_REFINEMENT_DEFECT,
    }


def _restrict_pairwise(
    fine: np.ndarray,
    fine_measures: np.ndarray,
) -> np.ndarray:
    values = np.asarray(fine, dtype=float)
    measures = np.asarray(fine_measures, dtype=float)
    if values.shape[-2] != measures.size or measures.size % 2:
        raise ValueError("WP10c8v restriction inputs are not nested")
    leading = values.shape[:-2]
    fields = values.shape[-1]
    reshaped = values.reshape(leading + (measures.size // 2, 2, fields))
    weights = measures.reshape(measures.size // 2, 2)
    numerator = np.sum(
        reshaped * weights.reshape((1,) * len(leading) + weights.shape + (1,)),
        axis=-2,
    )
    denominator = np.sum(weights, axis=1)
    return numerator / denominator.reshape(
        (1,) * len(leading) + (denominator.size, 1)
    )


def _weighted_history_metrics(
    coarse: dict[str, np.ndarray],
    fine: dict[str, np.ndarray],
    coarse_arrays: dict[str, np.ndarray],
    fine_arrays: dict[str, np.ndarray],
) -> dict:
    coarse_active = _active_cell_count(
        int(
            round(
                coarse_arrays["radius_rg"].size
                / LOCAL_DOMAIN_FRACTION
            )
        )
    )
    fine_active = _active_cell_count(
        int(
            round(
                fine_arrays["radius_rg"].size
                / LOCAL_DOMAIN_FRACTION
            )
        )
    )
    if fine_active != 2 * coarse_active:
        raise RuntimeError("WP10c8v active grids are not nested")
    fine_state = _restrict_pairwise(
        fine["state"][:, :fine_active],
        fine_arrays["cell_measures"][:fine_active],
    )
    fine_rate = _restrict_pairwise(
        fine["rate"][:, :fine_active],
        fine_arrays["cell_measures"][:fine_active],
    )
    coarse_state = np.asarray(
        coarse["state"][:, :coarse_active],
        dtype=float,
    )
    coarse_rate = np.asarray(
        coarse["rate"][:, :coarse_active],
        dtype=float,
    )
    weights = np.asarray(
        coarse_arrays["cell_measures"][:coarse_active],
        dtype=float,
    )
    weights = weights / np.sum(weights)

    def metrics(first: np.ndarray, second: np.ndarray) -> dict:
        difference = second - first
        first_norm = np.sqrt(
            np.sum(weights[None, :, None] * first**2, axis=(1, 2))
        )
        second_norm = np.sqrt(
            np.sum(weights[None, :, None] * second**2, axis=(1, 2))
        )
        difference_norm = np.sqrt(
            np.sum(
                weights[None, :, None] * difference**2,
                axis=(1, 2),
            )
        )
        dot = np.sum(
            weights[None, :, None] * first * second,
            axis=(1, 2),
        )
        cosine = dot / np.maximum(
            first_norm * second_norm,
            np.finfo(float).tiny,
        )
        amplitude_ratio = second_norm / np.maximum(
            first_norm,
            np.finfo(float).tiny,
        )
        return {
            "signed_cosine": cosine,
            "amplitude_ratio": amplitude_ratio,
            "relative_l2_difference": difference_norm
            / np.maximum(first_norm, np.finfo(float).tiny),
            "maximum_relative_l2_difference": float(
                np.max(
                    difference_norm
                    / np.maximum(first_norm, np.finfo(float).tiny)
                )
            ),
            "minimum_signed_cosine": float(np.min(cosine)),
            "final_signed_cosine": float(cosine[-1]),
            "final_amplitude_ratio": float(amplitude_ratio[-1]),
        }

    return {
        "state": metrics(coarse_state, fine_state),
        "rate": metrics(coarse_rate, fine_rate),
    }


def _zero_crossings(times: np.ndarray, signal: np.ndarray) -> np.ndarray:
    time_values = np.asarray(times, dtype=float)
    values = np.asarray(signal, dtype=float)
    crossings = []
    for index in range(values.size - 1):
        left = float(values[index])
        right = float(values[index + 1])
        if left == 0.0:
            crossings.append(float(time_values[index]))
        elif left * right < 0.0:
            fraction = abs(left) / (abs(left) + abs(right))
            crossings.append(
                float(
                    time_values[index]
                    + fraction
                    * (time_values[index + 1] - time_values[index])
                )
            )
    return np.asarray(crossings, dtype=float)


def _signal_diagnostics(
    times: np.ndarray,
    signal: np.ndarray,
) -> dict:
    values = np.asarray(signal, dtype=float)
    crossings = _zero_crossings(times, values)
    frequency = None
    if crossings.size >= 3:
        half_period = float(np.mean(np.diff(crossings)))
        if half_period > 0.0:
            frequency = 1.0 / (2.0 * half_period)
    envelope = np.maximum(
        np.abs(hilbert(values - np.mean(values))),
        np.finfo(float).tiny,
    )
    mask = envelope >= 0.1 * float(np.max(envelope))
    damping = None
    if np.count_nonzero(mask) >= 4:
        damping = float(
            np.polyfit(
                np.asarray(times, dtype=float)[mask],
                np.log(envelope[mask]),
                1,
            )[0]
        )
    return {
        "zero_crossings_seconds": crossings,
        "frequency_hz": frequency,
        "envelope_log_slope_per_s": damping,
        "maximum_absolute_signal": float(np.max(np.abs(values))),
        "final_signal": float(values[-1]),
    }


def _relative_scalar_defect(
    first: float | None,
    second: float | None,
) -> float | None:
    if (
        first is None
        or second is None
        or not (np.isfinite(first) and np.isfinite(second))
    ):
        return None
    return float(
        abs(second - first)
        / max(abs(first), abs(second), np.finfo(float).tiny)
    )


def _signal_pair_metrics(
    coarse: dict,
    fine: dict,
) -> dict:
    coarse_cross = np.asarray(
        coarse["zero_crossings_seconds"],
        dtype=float,
    )
    fine_cross = np.asarray(
        fine["zero_crossings_seconds"],
        dtype=float,
    )
    count = min(coarse_cross.size, fine_cross.size, 4)
    crossing_defect = None
    if count:
        crossing_defect = float(
            np.max(
                np.abs(fine_cross[:count] - coarse_cross[:count])
                / np.maximum(
                    np.maximum(
                        np.abs(fine_cross[:count]),
                        np.abs(coarse_cross[:count]),
                    ),
                    np.finfo(float).tiny,
                )
            )
        )
    return {
        "matched_zero_crossing_count": count,
        "maximum_zero_crossing_relative_defect": crossing_defect,
        "frequency_relative_defect": _relative_scalar_defect(
            coarse["frequency_hz"],
            fine["frequency_hz"],
        ),
        "damping_relative_defect": _relative_scalar_defect(
            coarse["envelope_log_slope_per_s"],
            fine["envelope_log_slope_per_s"],
        ),
    }


def _directional_term_contributions(
    context,
    arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    base = np.asarray(arrays["base_primitives"], dtype=float)
    scales = np.asarray(
        arrays["primitive_column_scales"],
        dtype=float,
    ).reshape(base.shape)
    row_scales = np.asarray(
        arrays["conservation_row_scales"],
        dtype=float,
    ).ravel()
    amplitudes = np.asarray(
        arrays["physical_input_amplitudes"],
        dtype=float,
    )
    direction = np.asarray(
        arrays["matched_half_difference"],
        dtype=float,
    ) / amplitudes
    direction /= _continuum_norm(direction, arrays["cell_measures"])
    physical_direction = amplitudes * direction
    step = TERM_DIRECTIONAL_STEP

    def blocks(primitives: np.ndarray) -> dict[str, np.ndarray]:
        state = causal_five_field_state_from_primitives(
            context,
            primitives,
        )
        evaluation = evaluate_causal_five_field_dae(
            pack_causal_five_field_state(state),
            context,
        )
        faces = np.asarray(
            evaluation.numerical_weighted_face_fluxes_over_c,
            dtype=float,
        )
        decomposition = causal_five_field_face_flux_decomposition(
            context,
            pack_causal_five_field_state(state),
        )
        component_faces = {
            "transport_inner_boundary": np.zeros_like(faces),
            "transport_central_perfect": np.zeros_like(faces),
            "transport_central_stress": np.zeros_like(faces),
            "transport_rusanov": np.zeros_like(faces),
            "transport_outer_boundary": np.zeros_like(faces),
        }
        component_faces["transport_inner_boundary"][0] = faces[0]
        component_faces["transport_central_perfect"][1:-1] = (
            decomposition.central_perfect_weighted_face_fluxes_over_c
        )
        component_faces["transport_central_stress"][1:-1] = (
            decomposition.central_stress_weighted_face_fluxes_over_c
        )
        component_faces["transport_rusanov"][1:-1] = (
            decomposition.rusanov_weighted_face_fluxes_over_c
        )
        component_faces["transport_outer_boundary"][-1] = faces[-1]
        result = {
            name: values[1:] - values[:-1]
            for name, values in component_faces.items()
        }
        for name, values in (
            evaluation.integrated_source_components_per_ct.items()
        ):
            result[f"source_{name}"] = -np.asarray(values, dtype=float)
        return result

    plus = blocks(base + step * physical_direction)
    minus = blocks(base - step * physical_direction)
    mass = np.asarray(arrays["descriptor"], dtype=float)
    input_scaled = physical_direction.ravel() / scales.ravel()
    result = {}
    reconstructed = np.zeros_like(input_scaled)
    for name in sorted(plus):
        residual_direction = (
            np.asarray(plus[name], dtype=float)
            - np.asarray(minus[name], dtype=float)
        ).ravel() / (2.0 * step * row_scales)
        rate = -np.linalg.solve(mass, residual_direction)
        result[name] = rate.reshape(base.shape)
        reconstructed += rate
    target = np.linalg.solve(
        mass,
        -np.asarray(arrays["stationary_jacobian"], dtype=float)
        @ input_scaled,
    )
    scale = max(
        float(np.linalg.norm(target)),
        np.finfo(float).tiny,
    )
    result["stationary_reconstruction_relative_defect"] = np.asarray(
        np.linalg.norm(reconstructed - target) / scale
    )
    storage_rate = -np.linalg.solve(
        mass,
        np.asarray(arrays["storage_rate_derivative"], dtype=float)
        @ input_scaled,
    )
    result["descriptor_rate_dependence"] = storage_rate.reshape(
        base.shape
    )
    return result


def _term_summary(
    term_arrays: dict[int, dict[str, np.ndarray]],
    operator_arrays: dict[int, dict[str, np.ndarray]],
) -> dict:
    names = sorted(
        name
        for name in term_arrays[64]
        if not name.endswith("relative_defect")
    )
    summary = {
        "component_names": names,
        "stationary_reconstruction_relative_defects": {
            f"N{mesh}": float(
                term_arrays[mesh][
                    "stationary_reconstruction_relative_defect"
                ]
            )
            for mesh in GLOBAL_EQUIVALENT_MESHES
        },
        "component_active_norms": {},
        "n128_n256_restricted_absolute_defects": {},
        "n128_n256_restricted_relative_defects": {},
        "controlling_component_by_mesh": {},
        "controlling_peak_by_mesh": {},
    }
    for name in names:
        norms = {}
        for mesh in GLOBAL_EQUIVALENT_MESHES:
            active = _active_cell_count(mesh)
            values = np.asarray(term_arrays[mesh][name], dtype=float)[
                :active
            ]
            norms[f"N{mesh}"] = _continuum_norm(
                values,
                operator_arrays[mesh]["cell_measures"][:active],
            )
        summary["component_active_norms"][name] = norms
        coarse = np.asarray(term_arrays[128][name], dtype=float)[
            : _active_cell_count(128)
        ]
        restricted = _restrict_pairwise(
            np.asarray(term_arrays[256][name], dtype=float)[
                : _active_cell_count(256)
            ][None, ...],
            operator_arrays[256]["cell_measures"][
                : _active_cell_count(256)
            ],
        )[0]
        scale = max(
            _continuum_norm(
                coarse,
                operator_arrays[128]["cell_measures"][
                    : _active_cell_count(128)
                ],
            ),
            np.finfo(float).tiny,
        )
        absolute_defect = _continuum_norm(
            restricted - coarse,
            operator_arrays[128]["cell_measures"][
                : _active_cell_count(128)
            ],
        )
        summary["n128_n256_restricted_absolute_defects"][name] = (
            absolute_defect
        )
        summary["n128_n256_restricted_relative_defects"][name] = (
            None
            if scale <= MINIMUM_SIGNIFICANT_SIGNAL
            else absolute_defect / scale
        )
    for mesh in GLOBAL_EQUIVALENT_MESHES:
        label = f"N{mesh}"
        controlling = max(
            names,
            key=lambda name: summary["component_active_norms"][name][
                label
            ],
        )
        active = _active_cell_count(mesh)
        values = np.asarray(
            term_arrays[mesh][controlling],
            dtype=float,
        )[:active]
        peak = np.unravel_index(
            int(np.argmax(np.abs(values))),
            values.shape,
        )
        summary["controlling_component_by_mesh"][label] = controlling
        summary["controlling_peak_by_mesh"][label] = {
            "cell_index": int(peak[0]),
            "field_index": int(peak[1]),
            "radius_rg": float(
                operator_arrays[mesh]["radius_rg"][peak[0]]
            ),
            "absolute_scaled_rate": float(abs(values[peak])),
        }
    summary["controlling_n128_n256_absolute_defect_component"] = max(
        names,
        key=lambda name: summary[
            "n128_n256_restricted_absolute_defects"
        ][name],
    )
    return summary


def run(*, force_operators: bool = False) -> tuple[dict, dict]:
    started = time.perf_counter()
    parents = {mesh: _parent_bundle(mesh) for mesh in PARENT_MESHES}
    operator_payloads = {}
    operator_arrays = {}
    runtime = {}
    for mesh in GLOBAL_EQUIVALENT_MESHES:
        print(f"WP10c8v: building/loading N{mesh} local operator", flush=True)
        payload, arrays, local_runtime = _build_local_operator(
            mesh,
            parents,
            force=force_operators,
        )
        operator_payloads[mesh] = payload
        operator_arrays[mesh] = arrays
        runtime[mesh] = local_runtime

    reproduction = {
        f"N{mesh}": _active_operator_reproduction(
            mesh,
            parents[mesh],
            operator_arrays[mesh],
        )
        for mesh in PARENT_MESHES
    }
    boundary_gate_passed = all(
        item["passed"] for item in reproduction.values()
    )

    propagated = {}
    temporal = {}
    signals = {}
    for mesh in GLOBAL_EQUIVALENT_MESHES:
        print(f"WP10c8v: propagating N{mesh} local tangent", flush=True)
        propagated[mesh] = _propagate(
            operator_arrays[mesh],
            time_samples=TIME_SAMPLES,
        )
        refined = _propagate(
            operator_arrays[mesh],
            time_samples=REFINED_TIME_SAMPLES,
        )
        temporal[f"N{mesh}"] = _temporal_refinement(
            propagated[mesh],
            refined,
        )
        signals[mesh] = _signal_diagnostics(
            propagated[mesh]["times"],
            propagated[mesh]["stress_rate_signal"],
        )
        signals[mesh]["initial_rate_activity_centroid_rg"] = float(
            propagated[mesh]["rate_activity_radial_centroid_rg"][0]
        )
        signals[mesh]["final_rate_activity_centroid_rg"] = float(
            propagated[mesh]["rate_activity_radial_centroid_rg"][-1]
        )
        signals[mesh]["initial_rate_activity_width_rg"] = float(
            propagated[mesh]["rate_activity_radial_width_rg"][0]
        )
        signals[mesh]["final_rate_activity_width_rg"] = float(
            propagated[mesh]["rate_activity_radial_width_rg"][-1]
        )

    full_parent_histories = {
        mesh: _propagate_full_parent(
            mesh,
            parents[mesh],
            parents[128],
        )
        for mesh in PARENT_MESHES
    }
    full_local_history_reproduction = {
        f"N{mesh}": _same_grid_history_reproduction(
            full_parent_histories[mesh],
            propagated[mesh],
            parents[mesh]["context"].grid.cell_measures,
            _active_cell_count(mesh),
        )
        for mesh in PARENT_MESHES
    }
    boundary_history_gate_passed = all(
        item["passed"]
        for item in full_local_history_reproduction.values()
    )
    boundary_gate_passed = bool(
        boundary_gate_passed and boundary_history_gate_passed
    )

    pair_histories = {
        "N64_N128": _weighted_history_metrics(
            propagated[64],
            propagated[128],
            operator_arrays[64],
            operator_arrays[128],
        ),
        "N128_N256": _weighted_history_metrics(
            propagated[128],
            propagated[256],
            operator_arrays[128],
            operator_arrays[256],
        ),
    }
    signal_pairs = {
        "N64_N128": _signal_pair_metrics(signals[64], signals[128]),
        "N128_N256": _signal_pair_metrics(signals[128], signals[256]),
    }
    state_coarse = pair_histories["N64_N128"]["state"][
        "maximum_relative_l2_difference"
    ]
    state_fine = pair_histories["N128_N256"]["state"][
        "maximum_relative_l2_difference"
    ]
    rate_coarse = pair_histories["N64_N128"]["rate"][
        "maximum_relative_l2_difference"
    ]
    rate_fine = pair_histories["N128_N256"]["rate"][
        "maximum_relative_l2_difference"
    ]
    state_order = float(
        np.log2(
            max(state_coarse, np.finfo(float).tiny)
            / max(state_fine, np.finfo(float).tiny)
        )
    )
    rate_order = float(
        np.log2(
            max(rate_coarse, np.finfo(float).tiny)
            / max(rate_fine, np.finfo(float).tiny)
        )
    )

    term_arrays = {
        mesh: _directional_term_contributions(
            runtime[mesh]["context"],
            operator_arrays[mesh],
        )
        for mesh in GLOBAL_EQUIVALENT_MESHES
    }
    terms = _term_summary(term_arrays, operator_arrays)

    fine_signal = signal_pairs["N128_N256"]
    fine_signal_gate_passed = bool(
        fine_signal["maximum_zero_crossing_relative_defect"] is not None
        and fine_signal["frequency_relative_defect"] is not None
        and fine_signal["damping_relative_defect"] is not None
        and fine_signal["maximum_zero_crossing_relative_defect"]
        <= MAXIMUM_ZERO_CROSSING_RELATIVE_DEFECT
        and fine_signal["frequency_relative_defect"]
        <= MAXIMUM_FREQUENCY_RELATIVE_DEFECT
        and fine_signal["damping_relative_defect"]
        <= MAXIMUM_DAMPING_RELATIVE_DEFECT
    )
    refinement_gate_passed = bool(
        state_order >= MINIMUM_SPATIAL_CONTRACTION_ORDER
        and rate_order >= MINIMUM_SPATIAL_CONTRACTION_ORDER
        and pair_histories["N128_N256"]["state"][
            "minimum_signed_cosine"
        ]
        >= MINIMUM_SAME_TIME_SIGNED_COSINE
        and pair_histories["N128_N256"]["rate"][
            "minimum_signed_cosine"
        ]
        >= MINIMUM_SAME_TIME_SIGNED_COSINE
        and fine_signal_gate_passed
    )
    temporal_gate_passed = all(
        item["passed"] for item in temporal.values()
    )
    authorized_next = bool(
        boundary_gate_passed
        and temporal_gate_passed
        and refinement_gate_passed
    )
    classification = (
        "local_linear_phase_spatially_convergent_preflight"
        if authorized_next
        else "inner_fast_phase_spatially_unresolved_local_preflight"
    )

    arrays = {
        "times": propagated[64]["times"],
    }
    for mesh in GLOBAL_EQUIVALENT_MESHES:
        prefix = f"N{mesh}_"
        arrays[f"{prefix}radius_rg"] = operator_arrays[mesh]["radius_rg"]
        arrays[f"{prefix}cell_measures"] = operator_arrays[mesh][
            "cell_measures"
        ]
        arrays[f"{prefix}state_history"] = propagated[mesh]["state"]
        arrays[f"{prefix}rate_history"] = propagated[mesh]["rate"]
        arrays[f"{prefix}stress_state_signal"] = propagated[mesh][
            "stress_state_signal"
        ]
        arrays[f"{prefix}stress_rate_signal"] = propagated[mesh][
            "stress_rate_signal"
        ]
        arrays[f"{prefix}rate_activity_radial_centroid_rg"] = (
            propagated[mesh]["rate_activity_radial_centroid_rg"]
        )
        arrays[f"{prefix}rate_activity_radial_width_rg"] = (
            propagated[mesh]["rate_activity_radial_width_rg"]
        )
        arrays[f"{prefix}zero_crossings"] = signals[mesh][
            "zero_crossings_seconds"
        ]
        for name, values in term_arrays[mesh].items():
            arrays[f"{prefix}term_{name}"] = np.asarray(values)
    for label, comparison in pair_histories.items():
        for kind in ("state", "rate"):
            for name in (
                "signed_cosine",
                "amplitude_ratio",
                "relative_l2_difference",
            ):
                arrays[
                    f"{label}_{kind}_{name}"
                ] = comparison[kind][name]

    DEFAULT_ARRAYS.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DEFAULT_ARRAYS, **arrays)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "base_commit": BASE_COMMIT,
        "purpose": (
            "test whether the WP10c8u inner-mode phase failure contracts "
            "under one localized factor-two spatial refinement"
        ),
        "classification": classification,
        "authorized_next_step": (
            "independently_equilibrated_local_or_fixed_Q_truth_test"
            if authorized_next
            else "no_formal_average_or_reduced_architecture"
        ),
        "scope": {
            "global_equivalent_meshes": GLOBAL_EQUIVALENT_MESHES,
            "local_cell_counts": {
                f"N{mesh}": _local_cell_count(mesh)
                for mesh in GLOBAL_EQUIVALENT_MESHES
            },
            "active_cell_counts": {
                f"N{mesh}": _active_cell_count(mesh)
                for mesh in GLOBAL_EQUIVALENT_MESHES
            },
            "local_outer_radius_rg": float(
                operator_arrays[64]["grid_edges_rg"][-1]
            ),
            "active_outer_radius_rg": float(
                operator_arrays[64]["grid_edges_rg"][
                    _active_cell_count(64)
                ]
            ),
            "target_seconds": TARGET_SECONDS,
            "linearized_evolving_descriptor_only": True,
            "n256_is_prolongated_from_n128": True,
            "formal_fast_average_certified": False,
            "reduced_architecture_selected": False,
        },
        "gates": {
            "maximum_active_operator_relative_defect": (
                MAXIMUM_ACTIVE_OPERATOR_RELATIVE_DEFECT
            ),
            "maximum_full_local_history_relative_defect": (
                MAXIMUM_FULL_LOCAL_HISTORY_RELATIVE_DEFECT
            ),
            "minimum_full_local_history_signed_cosine": (
                MINIMUM_FULL_LOCAL_HISTORY_SIGNED_COSINE
            ),
            "maximum_temporal_refinement_defect": (
                MAXIMUM_TEMPORAL_REFINEMENT_DEFECT
            ),
            "minimum_spatial_contraction_order": (
                MINIMUM_SPATIAL_CONTRACTION_ORDER
            ),
            "minimum_same_time_signed_cosine": (
                MINIMUM_SAME_TIME_SIGNED_COSINE
            ),
            "maximum_zero_crossing_relative_defect": (
                MAXIMUM_ZERO_CROSSING_RELATIVE_DEFECT
            ),
            "maximum_frequency_relative_defect": (
                MAXIMUM_FREQUENCY_RELATIVE_DEFECT
            ),
            "maximum_damping_relative_defect": (
                MAXIMUM_DAMPING_RELATIVE_DEFECT
            ),
        },
        "active_operator_reproduction": reproduction,
        "full_local_history_reproduction": (
            full_local_history_reproduction
        ),
        "temporal_refinement": temporal,
        "signal_diagnostics": {
            f"N{mesh}": signals[mesh]
            for mesh in GLOBAL_EQUIVALENT_MESHES
        },
        "pairwise_signal_diagnostics": signal_pairs,
        "pairwise_history_summary": {
            label: {
                kind: {
                    key: value
                    for key, value in metrics.items()
                    if np.isscalar(value)
                }
                for kind, metrics in comparison.items()
            }
            for label, comparison in pair_histories.items()
        },
        "spatial_contraction": {
            "state_observed_order": state_order,
            "rate_observed_order": rate_order,
            "passed": refinement_gate_passed,
        },
        "term_attribution": terms,
        "decision_gates": {
            "buffered_boundary_reproduction_passed": (
                boundary_gate_passed
            ),
            "buffered_boundary_history_reproduction_passed": (
                boundary_history_gate_passed
            ),
            "temporal_refinement_passed": temporal_gate_passed,
            "spatial_phase_refinement_passed": (
                refinement_gate_passed
            ),
            "authorized_next": authorized_next,
        },
        "operator_caches": {
            f"N{mesh}": {
                "arrays_path": operator_payloads[mesh]["arrays_path"],
                "arrays_sha256": operator_payloads[mesh]["arrays_sha256"],
                "wall_seconds": operator_payloads[mesh]["wall_seconds"],
            }
            for mesh in GLOBAL_EQUIVALENT_MESHES
        },
        "artifacts": {
            "arrays_path": _relative(DEFAULT_ARRAYS),
            "arrays_sha256": _sha256(DEFAULT_ARRAYS),
            "runner": THIS_RUNNER,
            "runner_sha256": _sha256(ROOT / THIS_RUNNER),
        },
        "wall_seconds": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    DEFAULT_OUTPUT.write_text(
        json.dumps(
            _plain(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return payload, arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force-operators",
        action="store_true",
        help="rebuild all local evolving-tangent operator caches",
    )
    args = parser.parse_args()
    payload, _arrays = run(force_operators=args.force_operators)
    summary = {
        "work_package": payload["work_package"],
        "classification": payload["classification"],
        "decision_gates": payload["decision_gates"],
        "spatial_contraction": payload["spatial_contraction"],
        "output": _relative(DEFAULT_OUTPUT),
        "arrays": _relative(DEFAULT_ARRAYS),
    }
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
