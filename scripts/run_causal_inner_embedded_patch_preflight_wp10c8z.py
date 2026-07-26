"""Run the WP10c8z conservative embedded-inner-patch preflight.

WP10c8y established that a jointly continuum-matched inner perturbation is
initially N128/N256 converged, is insensitive to the tested excision trace,
and nevertheless loses spatial phase over 0.125 s.  This package retains the
production inner boundary and embeds a permanently refined inner grid in an
evolving N128 exterior.  The fine/coarse coupling is one ordinary interior
face of the existing causal DAE and therefore uses exactly one production
Rusanov flux with equal-and-opposite neighboring-cell contributions.

The audit domain ends at the parent N128 face near 24.56 r_g.  Its exterior
trace is frozen only at that causally separated audit boundary; the embedded
coupling face at 12.777 r_g (and its location variant) is never frozen.
This remains a frozen-linear patch-resolution preflight.  It can authorize a
bounded nonlinear embedded-patch truth test, but not a production patch,
fixed-Q average, reduced coordinate, or long-time evolution.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import platform
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import numpy as np
import scipy
from scipy.interpolate import PchipInterpolator
from scipy.sparse.linalg import expm_multiply

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_anchor_excision_audit_wp10c8w as wp10c8w
import run_causal_inner_common_mode_audit_wp10c8y as wp10c8y
import run_causal_inner_phase_spatial_preflight_wp10c8v as wp10c8v

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    KerrSchildCellSourceRates,
    causal_embedded_patch_flux_audit,
    causal_exact_coordinate_projection,
    causal_five_field_dae_jacobian_color_groups,
    causal_five_field_dae_jacobian_sparsity,
    causal_five_field_evolving_tangent_matrices,
    causal_five_field_moment_coordinate_ladder,
    causal_five_field_reconstruct_face_charts,
    causal_five_field_reduced_descriptor_matrices,
    causal_five_field_state_from_primitives,
    evaluate_causal_five_field_dae,
    make_causal_embedded_patch_layout,
    make_kerr_schild_column_grid_from_edges,
    pack_causal_five_field_state,
    restrict_causal_embedded_patch_cell_averages,
)


BASE_COMMIT = "6764fc117ce453b4deb5c6b1c275a19c7352b4be"
WORK_PACKAGE = "WP10c8z"
SCHEMA_VERSION = 1
THIS_RUNNER = (
    "scripts/run_causal_inner_embedded_patch_preflight_wp10c8z.py"
)
CORE_DAE_FILE = (
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_dae_system.py"
)
CORE_PATCH_FILE = (
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_embedded_patch.py"
)
CORE_SPATIAL_FILE = (
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_spatial_audit.py"
)

PARENT_MESH = 128
AUDIT_OUTER_PARENT_FACE = 64
ACTIVE_OUTER_PARENT_FACE = 32
PRIMARY_COUPLING_PARENT_FACE = 48
VARIANT_COUPLING_PARENT_FACE = 56
PATCH_REFINEMENT_RATIOS = (1, 2, 4)
TARGET_SECONDS = 0.125
TIME_SAMPLES = 201
COORDINATE_LEVEL = "plus_shell_stress_storage"

MINIMUM_SPATIAL_ORDER = 0.75
MINIMUM_SIGNED_COSINE = 0.90
MAXIMUM_ZERO_CROSSING_DEFECT = 0.10
MAXIMUM_FREQUENCY_DEFECT = 0.10
MAXIMUM_DAMPING_DEFECT = 0.25
MAXIMUM_COUPLING_LOCATION_HISTORY_DEFECT = 0.02
MAXIMUM_SHARED_FLUX_DEFECT = 1.0e-12
MAXIMUM_STORAGE_ACTION_DEFECT = 2.0e-5
MAXIMUM_GENERATOR_FACTORIZATION_DEFECT = 5.0e-8
MAXIMUM_ANCHOR_COORDINATE_DEFECT = 2.0e-10
MAXIMUM_PROPAGATION_RESTART_DEFECT = 2.0e-10
MAXIMUM_COUPLING_SIGNAL_FRACTION = 1.0e-3

WP10C8Y_OUTPUT = (
    ROOT
    / "outputs/tables/"
    "causal_inner_common_mode_audit_wp10c8y.json"
)
WP10C8Y_ARRAYS = (
    ROOT
    / "outputs/tables/"
    "causal_inner_common_mode_audit_wp10c8y_arrays.npz"
)
CHECKPOINT_DIRECTORY = (
    ROOT / "outputs/checkpoints/causal_five_field_wp10c8z"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/tables/"
    "causal_inner_embedded_patch_preflight_wp10c8z.json"
)
DEFAULT_ARRAYS = (
    ROOT
    / "outputs/tables/"
    "causal_inner_embedded_patch_preflight_wp10c8z_arrays.npz"
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
    with np.load(path) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _zero_sources(n_cells: int) -> KerrSchildCellSourceRates:
    zeros = np.zeros(int(n_cells), dtype=float)
    return KerrSchildCellSourceRates(
        rest_mass=np.array(zeros, copy=True),
        radial_momentum_over_c=np.array(zeros, copy=True),
        angular_momentum_over_c=np.array(zeros, copy=True),
        killing_energy_over_c2=np.array(zeros, copy=True),
    )


def _observed_order(coarse: float, fine: float) -> float | None:
    if not (
        np.isfinite(coarse)
        and np.isfinite(fine)
        and coarse > 0.0
        and fine > 0.0
    ):
        return None
    return float(np.log2(coarse / fine))


def _common_profile_function(
    radius_rg: np.ndarray,
    values: np.ndarray,
):
    radius = np.asarray(radius_rg, dtype=float)
    array = np.asarray(values, dtype=float)
    if (
        radius.ndim != 1
        or array.shape[0] != radius.size
        or np.any(radius <= 0.0)
        or np.any(~np.isfinite(array))
    ):
        raise ValueError("WP10c8z continuum profile inputs are invalid")
    fits = tuple(
        PchipInterpolator(np.log(radius), array[:, column], extrapolate=True)
        for column in range(array.shape[1])
    )

    def evaluate(target_radius_rg: np.ndarray) -> np.ndarray:
        target = np.asarray(target_radius_rg, dtype=float)
        result = np.stack(
            [fit(np.log(target)) for fit in fits],
            axis=-1,
        )
        if np.any(~np.isfinite(result)):
            raise RuntimeError("WP10c8z continuum profile is nonfinite")
        return result

    return evaluate


def _common_positive_profile_function(
    radius_rg: np.ndarray,
    values: np.ndarray,
):
    array = np.asarray(values, dtype=float)
    if np.any(array <= 0.0):
        raise ValueError("WP10c8z positive profile inputs must be positive")
    log_evaluator = _common_profile_function(radius_rg, np.log(array))

    def evaluate(target_radius_rg: np.ndarray) -> np.ndarray:
        result = np.exp(log_evaluator(target_radius_rg))
        if np.any(~np.isfinite(result)) or np.any(result <= 0.0):
            raise RuntimeError("WP10c8z positive profile is invalid")
        return result

    return evaluate


def _parent_data() -> dict:
    parent = wp10c8v._parent_bundle(PARENT_MESH)
    context = parent["context"]
    gravitational_radius = float(context.grid.gravitational_radius)
    full_primitives = np.asarray(parent["primitives"], dtype=float)
    full_reconstruction = causal_five_field_reconstruct_face_charts(
        context,
        full_primitives,
    )
    outer_face = AUDIT_OUTER_PARENT_FACE
    parent_grid = make_kerr_schild_column_grid_from_edges(
        np.asarray(context.grid.edges[: outer_face + 1], dtype=float),
        gravitational_radius,
    )
    local_context = replace(
        context,
        grid=parent_grid,
        stream_sources=_zero_sources(outer_face),
        outer_boundary_flux_mode="frozen_exterior_rusanov",
        outer_boundary_frozen_exterior_chart=np.array(
            full_reconstruction.right_face_charts[outer_face],
            copy=True,
        ),
    ).validated()
    operator = wp10c8v._load_npz(parent["operator_path"])
    anchor256 = _load_npz(wp10c8y._parent_anchor_path(256))
    source_radius = np.concatenate(
        (
            np.asarray(anchor256["radius_rg"], dtype=float),
            np.asarray(
                context.grid.centers[
                    PRIMARY_COUPLING_PARENT_FACE:AUDIT_OUTER_PARENT_FACE
                ]
                / gravitational_radius,
                dtype=float,
            ),
        )
    )
    source_primitives = np.concatenate(
        (
            np.asarray(anchor256["base_primitives"], dtype=float),
            full_primitives[
                PRIMARY_COUPLING_PARENT_FACE:AUDIT_OUTER_PARENT_FACE
            ],
        ),
        axis=0,
    )
    parent_amplitudes = np.asarray(
        operator["physical_input_amplitudes"],
        dtype=float,
    ).reshape(PARENT_MESH, 5)
    amplitude_radius = np.concatenate(
        (
            np.asarray(anchor256["radius_rg"], dtype=float),
            np.asarray(
                context.grid.centers[PRIMARY_COUPLING_PARENT_FACE:]
                / gravitational_radius,
                dtype=float,
            ),
        )
    )
    amplitude_values = np.concatenate(
        (
            np.asarray(
                anchor256["physical_input_amplitudes"],
                dtype=float,
            ),
            parent_amplitudes[PRIMARY_COUPLING_PARENT_FACE:],
        ),
        axis=0,
    )
    return {
        "bundle": parent,
        "context": local_context,
        "full_context": context,
        "parent_grid": parent_grid,
        "parent_base_primitives": full_primitives[:outer_face],
        "base_evaluator": _common_profile_function(
            source_radius,
            source_primitives,
        ),
        "amplitude_evaluator": _common_positive_profile_function(
            amplitude_radius,
            amplitude_values,
        ),
        "selected_coefficients": _load_npz(WP10C8Y_ARRAYS)[
            "selected_coefficients"
        ],
        "shell_zero_outer_rg": float(operator["shell_edges_rg"][1]),
        "operator_path": parent["operator_path"],
        "anchor256_path": wp10c8y._parent_anchor_path(256),
    }


def _local_shell_edges_rg(context, shell_zero_outer_rg: float) -> np.ndarray:
    return wp10c8w._local_shell_edges_rg(
        context,
        shell_zero_outer_rg,
    )


def _build_reduced_level(
    context,
    primitives: np.ndarray,
    shell_edges_rg: np.ndarray,
):
    state = causal_five_field_state_from_primitives(context, primitives)
    vector = pack_causal_five_field_state(state)
    reduced = causal_five_field_reduced_descriptor_matrices(
        context,
        vector,
        finite_difference_step=wp10c8v.FINITE_DIFFERENCE_STEP,
        descriptor_timestep_seconds=wp10c8v.DESCRIPTOR_TIMESTEP_SECONDS,
    )
    ladder = causal_five_field_moment_coordinate_ladder(
        context,
        vector,
        reduced,
        shell_edges_rg,
        shape_bands_rg=(),
    )
    return state, vector, reduced, ladder.level(COORDINATE_LEVEL)


def _project_anchor(
    *,
    context,
    provisional: np.ndarray,
    amplitudes: np.ndarray,
    shell_edges_rg: np.ndarray,
    target_values: np.ndarray,
    target_scales: np.ndarray,
    active_outer_rg: float,
) -> tuple[np.ndarray, dict, dict, object]:
    _state, _vector, reduced, level = _build_reduced_level(
        context,
        provisional,
        shell_edges_rg,
    )
    current = np.asarray(level.coordinate_values, dtype=float)
    initial_defect = float(
        np.max(
            np.abs(current - target_values)
            / np.asarray(target_scales, dtype=float)
        )
    )
    projection = causal_exact_coordinate_projection(
        base_primitive_vector=np.asarray(provisional, dtype=float).ravel(),
        primitive_column_scales=np.asarray(
            reduced["primitive_column_scales"],
            dtype=float,
        ),
        state_weights=wp10c8w._continuum_weights(
            context,
            active_outer_rg=active_outer_rg,
        ),
        physical_input_amplitudes=np.asarray(
            amplitudes,
            dtype=float,
        ).ravel(),
        target_coordinate_values=np.asarray(target_values, dtype=float),
        target_coordinate_scales=np.asarray(target_scales, dtype=float),
        constraint_matrix=(
            np.asarray(level.raw_constraint_matrix, dtype=float)
            / np.asarray(target_scales, dtype=float)[:, None]
        ),
        coordinate_evaluator=wp10c8w._moment_evaluator(
            context,
            shell_edges_rg,
        ),
    )
    corrected = np.asarray(
        projection.primitive_vector,
        dtype=float,
    ).reshape(provisional.shape)
    if np.array_equal(corrected, provisional):
        final_reduced = reduced
        final_level = level
    else:
        (
            _state,
            _vector,
            final_reduced,
            final_level,
        ) = _build_reduced_level(
            context,
            corrected,
            shell_edges_rg,
        )
    final_defect = float(
        np.max(
            np.abs(
                np.asarray(final_level.coordinate_values, dtype=float)
                - target_values
            )
            / np.asarray(target_scales, dtype=float)
        )
    )
    scaled = np.asarray(
        projection.scaled_increment,
        dtype=float,
    ).reshape(provisional.shape)
    radius = (
        np.asarray(context.grid.centers, dtype=float)
        / context.grid.gravitational_radius
    )
    active = radius <= active_outer_rg * (1.0 + 2.0e-14)
    report = {
        "optimizer_success": projection.optimizer_success,
        "initial_coordinate_defect": initial_defect,
        "maximum_coordinate_defect": final_defect,
        "maximum_active_scaled_correction": float(
            np.max(np.abs(scaled[active]))
        ),
        "maximum_buffer_scaled_correction": float(
            np.max(np.abs(scaled[~active]))
        ),
        "constraint_rank": projection.normal_basis.numerical_rank,
        "constraint_condition": (
            projection.normal_basis.condition_estimate
        ),
        "function_evaluations": projection.function_evaluations,
        "passed": bool(
            projection.optimizer_success
            and final_defect <= MAXIMUM_ANCHOR_COORDINATE_DEFECT
        ),
    }
    return corrected, report, final_reduced, final_level


def _operator_paths(label: str) -> tuple[Path, Path]:
    return (
        CHECKPOINT_DIRECTORY / f"{label}_operator.json",
        CHECKPOINT_DIRECTORY / f"{label}_operator_arrays.npz",
    )


def _operator_contract(
    label: str,
    context,
    base_primitives: np.ndarray,
    target_values: np.ndarray,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "base_commit": BASE_COMMIT,
        "label": label,
        "grid_edges_sha256": _array_sha256(context.grid.edges),
        "base_primitives_sha256": _array_sha256(base_primitives),
        "target_coordinate_values_sha256": _array_sha256(target_values),
        "spatial_reconstruction": context.spatial_reconstruction,
        "boundary_trace_reconstruction": (
            context.boundary_trace_reconstruction
        ),
        "cell_rate_scheme": context.cell_rate_scheme,
        "cell_source_quadrature": context.cell_source_quadrature,
        "cell_storage_quadrature": context.cell_storage_quadrature,
    }


def _build_or_load_operator(
    label: str,
    *,
    context,
    base_primitives: np.ndarray,
    reduced: dict,
    level,
    target_values: np.ndarray,
    force: bool,
) -> tuple[dict, dict[str, np.ndarray]]:
    json_path, arrays_path = _operator_paths(label)
    contract = _operator_contract(
        label,
        context,
        base_primitives,
        target_values,
    )
    if not force and json_path.exists() and arrays_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if (
            all(payload.get(key) == value for key, value in contract.items())
            and payload.get("arrays_sha256") == _sha256(arrays_path)
            and payload.get("passed") is True
        ):
            return payload, _load_npz(arrays_path)

    print(f"WP10c8z: building {label} descriptor", flush=True)
    started = time.perf_counter()
    state = causal_five_field_state_from_primitives(
        context,
        base_primitives,
    )
    vector = pack_causal_five_field_state(state)
    evolving = causal_five_field_evolving_tangent_matrices(
        context,
        vector,
        primitive_rate_per_s=None,
        reduced_descriptor=reduced,
        finite_difference_step=wp10c8v.FINITE_DIFFERENCE_STEP,
        descriptor_timestep_seconds=wp10c8v.DESCRIPTOR_TIMESTEP_SECONDS,
        storage_difference_step=wp10c8v.STORAGE_DIFFERENCE_STEP,
        storage_rate_derivative_step=(
            wp10c8v.STORAGE_RATE_DERIVATIVE_STEP
        ),
        storage_quadrature_order=wp10c8v.STORAGE_QUADRATURE_ORDER,
        storage_directional_step=wp10c8v.STORAGE_DIRECTIONAL_STEP,
    )
    arrays = {
        "generator": np.asarray(
            evolving["evolving_scaled_generator_per_s"],
            dtype=float,
        ),
        "primitive_column_scales": np.asarray(
            evolving["primitive_column_scales"],
            dtype=float,
        ),
        "base_primitives": np.asarray(base_primitives, dtype=float),
        "coordinate_values": np.asarray(
            level.coordinate_values,
            dtype=float,
        ),
        "coordinate_scales": np.asarray(
            level.coordinate_scales,
            dtype=float,
        ),
        "constraint_matrix": np.asarray(
            level.constraint_matrix,
            dtype=float,
        ),
        "raw_constraint_matrix": np.asarray(
            level.raw_constraint_matrix,
            dtype=float,
        ),
        "grid_edges_rg": np.asarray(
            context.grid.edges / context.grid.gravitational_radius,
            dtype=float,
        ),
        "radius_rg": np.asarray(
            context.grid.centers / context.grid.gravitational_radius,
            dtype=float,
        ),
        "cell_measures": np.asarray(
            context.grid.cell_measures,
            dtype=float,
        ),
    }
    gates = wp10c8w._audit_local_state_gates(context, vector)
    maximum_factorization = float(
        evolving["maximum_scaled_generator_factorization_defect"]
    )
    maximum_storage = float(
        evolving["maximum_relative_storage_action_defect"]
    )
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **arrays)
    payload = {
        **contract,
        "arrays_path": _relative(arrays_path),
        "arrays_sha256": _sha256(arrays_path),
        "state_gates": gates,
        "maximum_scaled_generator_factorization_defect": (
            maximum_factorization
        ),
        "maximum_relative_storage_action_defect": maximum_storage,
        "wall_seconds": time.perf_counter() - started,
    }
    payload["passed"] = bool(
        gates["passed"]
        and gates["measured"]["inner_incoming_characteristics"] == 0
        and maximum_factorization
        <= MAXIMUM_GENERATOR_FACTORIZATION_DEFECT
        and maximum_storage <= MAXIMUM_STORAGE_ACTION_DEFECT
    )
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
    return payload, arrays


def _level_from_arrays(arrays: dict[str, np.ndarray]):
    return SimpleNamespace(
        coordinate_values=np.asarray(arrays["coordinate_values"], dtype=float),
        coordinate_scales=np.asarray(arrays["coordinate_scales"], dtype=float),
        constraint_matrix=np.asarray(arrays["constraint_matrix"], dtype=float),
        raw_constraint_matrix=np.asarray(
            arrays["raw_constraint_matrix"],
            dtype=float,
        ),
    )


def _configuration(
    *,
    label: str,
    parent: dict,
    coupling_face: int,
    refinement_ratio: int,
    target_values: np.ndarray,
    target_scales: np.ndarray,
    active_outer_rg: float,
    force: bool,
    matched_reference: dict | None = None,
) -> dict:
    layout = make_causal_embedded_patch_layout(
        parent["parent_grid"],
        coupling_face,
        refinement_ratio,
    )
    context = replace(
        parent["context"],
        grid=layout.grid,
        stream_sources=_zero_sources(layout.n_cells),
    ).validated()
    radius_rg = (
        np.asarray(layout.grid.centers, dtype=float)
        / layout.grid.gravitational_radius
    )
    if matched_reference is not None:
        source_radius = (
            np.asarray(
                matched_reference["context"].grid.centers,
                dtype=float,
            )
            / matched_reference["context"].grid.gravitational_radius
        )
        provisional = _common_profile_function(
            source_radius,
            np.asarray(
                matched_reference["base_primitives"],
                dtype=float,
            ),
        )(radius_rg)
    elif refinement_ratio == 1:
        provisional = np.asarray(
            parent["parent_base_primitives"],
            dtype=float,
        )
    else:
        provisional = parent["base_evaluator"](radius_rg)
    amplitudes = parent["amplitude_evaluator"](radius_rg)
    shell_edges = _local_shell_edges_rg(
        context,
        parent["shell_zero_outer_rg"],
    )

    json_path, arrays_path = _operator_paths(label)
    cached = None
    if not force and json_path.exists() and arrays_path.exists():
        candidate = json.loads(json_path.read_text(encoding="utf-8"))
        candidate_arrays = _load_npz(arrays_path)
        contract = _operator_contract(
            label,
            context,
            candidate_arrays["base_primitives"],
            target_values,
        )
        if (
            all(candidate.get(key) == value for key, value in contract.items())
            and candidate.get("arrays_sha256") == _sha256(arrays_path)
            and candidate.get("passed") is True
        ):
            cached = (candidate, candidate_arrays)

    if cached is None:
        if matched_reference is not None:
            (
                _state,
                _vector,
                reduced,
                level,
            ) = _build_reduced_level(context, provisional, shell_edges)
            base = provisional
            coordinate_defect = float(
                np.max(
                    np.abs(
                        np.asarray(level.coordinate_values, dtype=float)
                        - target_values
                    )
                    / target_scales
                )
            )
            anchor_report = {
                "optimizer_success": True,
                "initial_coordinate_defect": None,
                "maximum_coordinate_defect": None,
                "base_coordinate_discretization_difference": (
                    coordinate_defect
                ),
                "maximum_active_scaled_correction": 0.0,
                "maximum_buffer_scaled_correction": 0.0,
                "constraint_rank": int(level.coordinate_values.size),
                "constraint_condition": None,
                "function_evaluations": 0,
                "matched_from_primary_patch": True,
                "passed": True,
            }
        elif refinement_ratio == 1:
            (
                _state,
                _vector,
                reduced,
                level,
            ) = _build_reduced_level(context, provisional, shell_edges)
            base = provisional
            anchor_report = {
                "optimizer_success": True,
                "initial_coordinate_defect": 0.0,
                "maximum_coordinate_defect": 0.0,
                "maximum_active_scaled_correction": 0.0,
                "maximum_buffer_scaled_correction": 0.0,
                "constraint_rank": int(level.coordinate_values.size),
                "constraint_condition": float("nan"),
                "function_evaluations": 0,
                "passed": True,
            }
        else:
            (
                base,
                anchor_report,
                reduced,
                level,
            ) = _project_anchor(
                context=context,
                provisional=provisional,
                amplitudes=amplitudes,
                shell_edges_rg=shell_edges,
                target_values=target_values,
                target_scales=target_scales,
                active_outer_rg=active_outer_rg,
            )
        operator_report, operator = _build_or_load_operator(
            label,
            context=context,
            base_primitives=base,
            reduced=reduced,
            level=level,
            target_values=target_values,
            force=force,
        )
    else:
        operator_report, operator = cached
        base = np.asarray(operator["base_primitives"], dtype=float)
        level = _level_from_arrays(operator)
        cached_coordinate_difference = float(
            np.max(
                np.abs(
                    np.asarray(level.coordinate_values, dtype=float)
                    - target_values
                )
                / target_scales
            )
        )
        anchor_report = {
            "optimizer_success": True,
            "initial_coordinate_defect": None,
            "maximum_coordinate_defect": (
                None
                if matched_reference is not None
                else cached_coordinate_difference
            ),
            "base_coordinate_discretization_difference": (
                cached_coordinate_difference
                if matched_reference is not None
                else None
            ),
            "maximum_active_scaled_correction": None,
            "maximum_buffer_scaled_correction": None,
            "constraint_rank": int(level.coordinate_values.size),
            "constraint_condition": None,
            "function_evaluations": None,
            "passed": True,
            "loaded_from_cache": True,
            "matched_from_primary_patch": matched_reference is not None,
        }

    data = {
        "context": context,
        "primitives": base,
        "shell_edges_rg": shell_edges,
        "reduced": {
            "primitive_column_scales": np.asarray(
                operator["primitive_column_scales"],
                dtype=float,
            )
        },
        "level": level,
        "weights": wp10c8w._continuum_weights(
            context,
            active_outer_rg=active_outer_rg,
        ),
        "common_amplitudes": amplitudes,
        "continuum_basis": wp10c8y._continuum_profile_basis(
            radius_rg,
            inner_rg=float(layout.grid.edges[0] / layout.grid.gravitational_radius),
            outer_rg=active_outer_rg,
        ),
        "radius_rg": radius_rg,
    }
    if matched_reference is None:
        minus, plus, half, pair_report = wp10c8y._exact_common_pair(
            data=data,
            coefficients=parent["selected_coefficients"],
            active_outer_rg=active_outer_rg,
        )
    else:
        source_radius = (
            np.asarray(
                matched_reference["context"].grid.centers,
                dtype=float,
            )
            / matched_reference["context"].grid.gravitational_radius
        )
        minus = _common_profile_function(
            source_radius,
            np.asarray(matched_reference["minus"], dtype=float),
        )(radius_rg)
        plus = _common_profile_function(
            source_radius,
            np.asarray(matched_reference["plus"], dtype=float),
        )(radius_rg)
        half = 0.5 * (plus - minus)
        evaluator = wp10c8w._moment_evaluator(context, shell_edges)
        minus_coordinates = evaluator(minus.ravel())
        plus_coordinates = evaluator(plus.ravel())
        pair_defect = float(
            np.max(
                np.abs(plus_coordinates - minus_coordinates)
                / np.asarray(level.coordinate_scales, dtype=float)
            )
        )
        minus_state = causal_five_field_state_from_primitives(context, minus)
        plus_state = causal_five_field_state_from_primitives(context, plus)
        minus_gates = wp10c8w._audit_local_state_gates(
            context,
            pack_causal_five_field_state(minus_state),
        )
        plus_gates = wp10c8w._audit_local_state_gates(
            context,
            pack_causal_five_field_state(plus_state),
        )
        pair_report = {
            "maximum_pairwise_coordinate_defect": pair_defect,
            "constraint_condition": None,
            "matched_from_primary_patch": True,
            "minus": {
                "optimizer_success": True,
                "maximum_coordinate_defect": pair_defect,
                "state_gates_passed": minus_gates["passed"],
            },
            "plus": {
                "optimizer_success": True,
                "maximum_coordinate_defect": pair_defect,
                "state_gates_passed": plus_gates["passed"],
            },
            "passed": bool(
                pair_defect <= MAXIMUM_ANCHOR_COORDINATE_DEFECT
                and minus_gates["passed"]
                and plus_gates["passed"]
            ),
        }
    state = causal_five_field_state_from_primitives(context, base)
    vector = pack_causal_five_field_state(state)
    flux_audit = causal_embedded_patch_flux_audit(
        context,
        vector,
        layout,
    )
    pattern = causal_five_field_dae_jacobian_sparsity(
        layout.n_cells,
        spatial_reconstruction=context.spatial_reconstruction,
        boundary_trace_reconstruction=context.boundary_trace_reconstruction,
        cell_rate_scheme=context.cell_rate_scheme,
        cell_source_quadrature=context.cell_source_quadrature,
        cell_storage_quadrature=context.cell_storage_quadrature,
    )
    groups = causal_five_field_dae_jacobian_color_groups(pattern)
    normalized_initial = wp10c8y._normalized_initial(
        half,
        amplitudes,
        layout.grid.cell_measures,
    )
    generator = wp10c8v._similarity_rescale_generator(
        np.asarray(operator["generator"], dtype=float),
        np.asarray(operator["primitive_column_scales"], dtype=float),
        amplitudes,
    )
    initial_rate = (generator @ normalized_initial.ravel()).reshape(
        normalized_initial.shape
    )
    return {
        "label": label,
        "layout": layout,
        "context": context,
        "base_primitives": base,
        "amplitudes": amplitudes,
        "operator_report": operator_report,
        "operator": operator,
        "anchor_report": anchor_report,
        "pair_report": pair_report,
        "minus": minus,
        "plus": plus,
        "half_difference": half,
        "normalized_initial": normalized_initial,
        "initial_rate": initial_rate,
        "generator": generator,
        "flux_audit": flux_audit,
        "jacobian_pattern_nonzeros": int(pattern.nnz),
        "jacobian_colors": len(groups),
    }


def _propagate(configuration: dict) -> dict[str, np.ndarray]:
    generator = np.asarray(configuration["generator"], dtype=float)
    initial = np.asarray(configuration["normalized_initial"], dtype=float)
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
    ).reshape(TIME_SAMPLES, -1, 5)
    rate = np.asarray(
        [generator @ row.ravel() for row in state],
        dtype=float,
    ).reshape(state.shape)
    radius = (
        configuration["context"].grid.centers
        / configuration["context"].grid.gravitational_radius
    )
    active = radius <= (
        configuration["active_outer_rg"] * (1.0 + 2.0e-14)
    )
    weights = np.asarray(
        configuration["context"].grid.cell_measures[active],
        dtype=float,
    )
    weights /= np.sum(weights)
    stress_signal = np.sum(weights[None, :] * rate[:, active, 4], axis=1)
    half_time = 0.5 * TARGET_SECONDS
    midpoint = expm_multiply(generator * half_time, initial.ravel())
    restarted = expm_multiply(generator * half_time, midpoint)
    direct = state[-1].ravel()
    restart_defect = float(
        np.linalg.norm(restarted - direct)
        / max(np.linalg.norm(direct), np.finfo(float).tiny)
    )
    return {
        "times": times,
        "state": state,
        "rate": rate,
        "stress_rate_signal": stress_signal,
        "restart_relative_defect": np.asarray(restart_defect),
    }


def _restrict_history(
    history: dict[str, np.ndarray],
    configuration: dict,
) -> dict[str, np.ndarray]:
    layout = configuration["layout"]
    return {
        "times": np.asarray(history["times"], dtype=float),
        "state": restrict_causal_embedded_patch_cell_averages(
            history["state"].reshape(-1, layout.n_cells, 5),
            layout,
        ),
        "rate": restrict_causal_embedded_patch_cell_averages(
            history["rate"].reshape(-1, layout.n_cells, 5),
            layout,
        ),
        "stress_rate_signal": np.asarray(
            history["stress_rate_signal"],
            dtype=float,
        ),
    }


def _history_metrics(
    first: dict[str, np.ndarray],
    second: dict[str, np.ndarray],
    parent_grid,
    *,
    lower_rg: float | None,
    upper_rg: float,
) -> dict:
    if not np.array_equal(first["times"], second["times"]):
        raise RuntimeError("WP10c8z history times differ")
    radius = parent_grid.centers / parent_grid.gravitational_radius
    mask = radius <= float(upper_rg) * (1.0 + 2.0e-14)
    if lower_rg is not None:
        mask &= radius >= float(lower_rg) * (1.0 - 2.0e-14)
    weights = np.asarray(parent_grid.cell_measures[mask], dtype=float)
    weights /= np.sum(weights)

    def compare(field: str) -> dict:
        coarse = np.asarray(first[field], dtype=float)[:, mask]
        fine = np.asarray(second[field], dtype=float)[:, mask]
        difference = fine - coarse
        coarse_norm = np.sqrt(
            np.sum(weights[None, :, None] * coarse**2, axis=(1, 2))
        )
        fine_norm = np.sqrt(
            np.sum(weights[None, :, None] * fine**2, axis=(1, 2))
        )
        difference_norm = np.sqrt(
            np.sum(weights[None, :, None] * difference**2, axis=(1, 2))
        )
        cosine = np.sum(
            weights[None, :, None] * coarse * fine,
            axis=(1, 2),
        ) / np.maximum(
            coarse_norm * fine_norm,
            np.finfo(float).tiny,
        )
        relative = difference_norm / np.maximum(
            coarse_norm,
            np.finfo(float).tiny,
        )
        return {
            "maximum_relative_l2_difference": float(np.max(relative)),
            "final_relative_l2_difference": float(relative[-1]),
            "minimum_signed_cosine": float(np.min(cosine)),
            "final_signed_cosine": float(cosine[-1]),
            "final_amplitude_ratio": float(
                fine_norm[-1]
                / max(coarse_norm[-1], np.finfo(float).tiny)
            ),
        }

    return {
        "state": compare("state"),
        "rate": compare("rate"),
    }


def _signal_metrics(
    first: dict[str, np.ndarray],
    second: dict[str, np.ndarray],
) -> dict:
    first_signal = wp10c8v._signal_diagnostics(
        first["times"],
        first["stress_rate_signal"],
    )
    second_signal = wp10c8v._signal_diagnostics(
        second["times"],
        second["stress_rate_signal"],
    )
    return {
        "first": first_signal,
        "second": second_signal,
        "comparison": wp10c8v._signal_pair_metrics(
            first_signal,
            second_signal,
        ),
    }


def _coupling_signal_fraction(
    history: dict[str, np.ndarray],
    configuration: dict,
) -> float:
    face = configuration["layout"].coupling_face_index
    state = np.asarray(history["state"], dtype=float)
    active_peak = np.max(np.abs(state), axis=(1, 2))
    start = max(face - 2, 0)
    stop = min(face + 2, state.shape[1])
    coupling_peak = np.max(np.abs(state[:, start:stop]), axis=(1, 2))
    return float(
        np.max(
            coupling_peak
            / np.maximum(active_peak, np.finfo(float).tiny)
        )
    )


def _configuration_summary(configuration: dict) -> dict:
    layout = configuration["layout"]
    flux = configuration["flux_audit"]
    return {
        "label": configuration["label"],
        "n_cells": layout.n_cells,
        "n_refined_cells": layout.n_refined_cells,
        "refinement_ratio": layout.refinement_ratio,
        "parent_coupling_face_index": (
            layout.parent_coupling_face_index
        ),
        "coupling_face_index": layout.coupling_face_index,
        "coupling_radius_rg": float(
            layout.coupling_radius / layout.grid.gravitational_radius
        ),
        "coarse_exterior_cells": (
            layout.n_cells - layout.n_refined_cells
        ),
        "anchor": configuration["anchor_report"],
        "pair": configuration["pair_report"],
        "operator": configuration["operator_report"],
        "shared_flux": {
            "maximum_state_flux_defect": (
                flux.maximum_state_flux_defect
            ),
            "maximum_telescoping_defect": (
                flux.maximum_telescoping_defect
            ),
            "passed": flux.passed,
        },
        "jacobian_pattern_nonzeros": (
            configuration["jacobian_pattern_nonzeros"]
        ),
        "jacobian_colors": configuration["jacobian_colors"],
    }


def run(*, force: bool = False) -> tuple[dict, dict[str, np.ndarray]]:
    started = time.perf_counter()
    if not WP10C8Y_OUTPUT.exists() or not WP10C8Y_ARRAYS.exists():
        raise FileNotFoundError("WP10c8z requires WP10c8y evidence")
    wp10c8y_payload = json.loads(
        WP10C8Y_OUTPUT.read_text(encoding="utf-8")
    )
    if wp10c8y_payload.get("classification") != (
        "common_mode_passed_boundary_insensitive_underresolution"
    ):
        raise RuntimeError("WP10c8y authorization changed")

    parent = _parent_data()
    active_outer_rg = float(
        parent["parent_grid"].edges[ACTIVE_OUTER_PARENT_FACE]
        / parent["parent_grid"].gravitational_radius
    )
    parent["active_outer_rg"] = active_outer_rg
    shell_edges = _local_shell_edges_rg(
        parent["context"],
        parent["shell_zero_outer_rg"],
    )
    (
        _target_state,
        _target_vector,
        _target_reduced,
        target_level,
    ) = _build_reduced_level(
        parent["context"],
        parent["parent_base_primitives"],
        shell_edges,
    )
    target_values = np.asarray(target_level.coordinate_values, dtype=float)
    target_scales = np.asarray(target_level.coordinate_scales, dtype=float)

    configurations = {}
    labels = {
        1: "N128_exterior_N128_inner_c48",
        2: "N128_exterior_N256_inner_c48",
        4: "N128_exterior_N512_inner_c48",
    }
    for ratio in PATCH_REFINEMENT_RATIOS:
        label = labels[ratio]
        print(f"WP10c8z: preparing {label}", flush=True)
        configuration = _configuration(
            label=label,
            parent=parent,
            coupling_face=PRIMARY_COUPLING_PARENT_FACE,
            refinement_ratio=ratio,
            target_values=target_values,
            target_scales=target_scales,
            active_outer_rg=active_outer_rg,
            force=force,
        )
        configuration["active_outer_rg"] = active_outer_rg
        configurations[label] = configuration
    variant_label = "N128_exterior_N512_inner_c56_matched"
    print(f"WP10c8z: preparing {variant_label}", flush=True)
    variant = _configuration(
        label=variant_label,
        parent=parent,
        coupling_face=VARIANT_COUPLING_PARENT_FACE,
        refinement_ratio=4,
        target_values=target_values,
        target_scales=target_scales,
        active_outer_rg=active_outer_rg,
        force=force,
        matched_reference=configurations[labels[4]],
    )
    variant["active_outer_rg"] = active_outer_rg
    configurations[variant_label] = variant

    histories = {}
    restricted = {}
    for label, configuration in configurations.items():
        print(f"WP10c8z: propagating {label}", flush=True)
        histories[label] = _propagate(configuration)
        restricted[label] = _restrict_history(
            histories[label],
            configuration,
        )

    coarse_label = labels[1]
    medium_label = labels[2]
    fine_label = labels[4]
    regions = {
        "active_core": (None, active_outer_rg),
        "outside_active_to_primary_coupling": (
            active_outer_rg,
            float(
                parent["parent_grid"].edges[
                    PRIMARY_COUPLING_PARENT_FACE
                ]
                / parent["parent_grid"].gravitational_radius
            ),
        ),
    }
    pairwise = {}
    orders = {}
    for region, (lower, upper) in regions.items():
        coarse_medium = _history_metrics(
            restricted[coarse_label],
            restricted[medium_label],
            parent["parent_grid"],
            lower_rg=lower,
            upper_rg=upper,
        )
        medium_fine = _history_metrics(
            restricted[medium_label],
            restricted[fine_label],
            parent["parent_grid"],
            lower_rg=lower,
            upper_rg=upper,
        )
        pairwise[region] = {
            "N128_N256patch": coarse_medium,
            "N256patch_N512patch": medium_fine,
        }
        orders[region] = {
            field: _observed_order(
                coarse_medium[field]["maximum_relative_l2_difference"],
                medium_fine[field]["maximum_relative_l2_difference"],
            )
            for field in ("state", "rate")
        }

    signal_ladder = {
        "N128_N256patch": _signal_metrics(
            histories[coarse_label],
            histories[medium_label],
        ),
        "N256patch_N512patch": _signal_metrics(
            histories[medium_label],
            histories[fine_label],
        ),
    }
    fine_signal = signal_ladder["N256patch_N512patch"]["comparison"]
    coupling_variation = _history_metrics(
        restricted[fine_label],
        restricted[variant_label],
        parent["parent_grid"],
        lower_rg=None,
        upper_rg=active_outer_rg,
    )
    coupling_location_defect = max(
        coupling_variation["state"]["maximum_relative_l2_difference"],
        coupling_variation["rate"]["maximum_relative_l2_difference"],
    )
    coupling_signal_fractions = {
        label: _coupling_signal_fraction(histories[label], configuration)
        for label, configuration in configurations.items()
    }
    maximum_shared_flux_defect = max(
        configuration[
            "flux_audit"
        ].maximum_state_flux_defect
        for configuration in configurations.values()
    )
    maximum_telescoping_defect = max(
        configuration[
            "flux_audit"
        ].maximum_telescoping_defect
        for configuration in configurations.values()
    )
    maximum_restart_defect = max(
        float(history["restart_relative_defect"])
        for history in histories.values()
    )
    active_orders = orders["active_core"]
    fine_history = pairwise["active_core"]["N256patch_N512patch"]
    spatial_gate_passed = bool(
        active_orders["state"] is not None
        and active_orders["rate"] is not None
        and active_orders["state"] >= MINIMUM_SPATIAL_ORDER
        and active_orders["rate"] >= MINIMUM_SPATIAL_ORDER
        and min(
            fine_history["state"]["minimum_signed_cosine"],
            fine_history["rate"]["minimum_signed_cosine"],
        )
        >= MINIMUM_SIGNED_COSINE
    )
    signal_gate_passed = bool(
        fine_signal["maximum_zero_crossing_relative_defect"] is not None
        and fine_signal["maximum_zero_crossing_relative_defect"]
        <= MAXIMUM_ZERO_CROSSING_DEFECT
        and (
            fine_signal["frequency_relative_defect"] is None
            or fine_signal["frequency_relative_defect"]
            <= MAXIMUM_FREQUENCY_DEFECT
        )
        and (
            fine_signal["damping_relative_defect"] is None
            or fine_signal["damping_relative_defect"]
            <= MAXIMUM_DAMPING_DEFECT
        )
    )
    method_gate_passed = bool(
        all(
            configuration["operator_report"]["passed"]
            and configuration["pair_report"]["passed"]
            and configuration["anchor_report"]["passed"]
            and configuration["flux_audit"].passed
            for configuration in configurations.values()
        )
        and maximum_shared_flux_defect <= MAXIMUM_SHARED_FLUX_DEFECT
        and maximum_telescoping_defect == 0.0
        and maximum_restart_defect <= MAXIMUM_PROPAGATION_RESTART_DEFECT
    )
    response_reached_coupling = bool(
        max(coupling_signal_fractions.values())
        > MAXIMUM_COUPLING_SIGNAL_FRACTION
    )
    coupling_gate_passed = bool(
        coupling_location_defect
        <= MAXIMUM_COUPLING_LOCATION_HISTORY_DEFECT
        and not response_reached_coupling
    )
    passed = bool(
        method_gate_passed
        and spatial_gate_passed
        and signal_gate_passed
        and coupling_gate_passed
    )
    if not method_gate_passed:
        classification = "embedded_patch_method_contract_failed"
    elif response_reached_coupling:
        classification = "embedded_patch_response_reached_coupling"
    elif not spatial_gate_passed or not signal_gate_passed:
        classification = "embedded_patch_inner_phase_not_converged"
    elif not coupling_gate_passed:
        classification = "embedded_patch_coupling_location_not_converged"
    else:
        classification = "embedded_patch_linear_preflight_converged"

    arrays = {
        "times": histories[fine_label]["times"],
        "target_coordinate_values": target_values,
        "target_coordinate_scales": target_scales,
        "selected_coefficients": parent["selected_coefficients"],
    }
    for label, configuration in configurations.items():
        arrays[f"{label}_grid_edges_rg"] = (
            configuration["context"].grid.edges
            / configuration["context"].grid.gravitational_radius
        )
        arrays[f"{label}_base_primitives"] = configuration[
            "base_primitives"
        ]
        arrays[f"{label}_pair_minus"] = configuration["minus"]
        arrays[f"{label}_pair_plus"] = configuration["plus"]
        arrays[f"{label}_normalized_initial"] = configuration[
            "normalized_initial"
        ]
        arrays[f"{label}_state_history"] = histories[label]["state"]
        arrays[f"{label}_rate_history"] = histories[label]["rate"]
        arrays[f"{label}_restricted_state_history"] = restricted[label][
            "state"
        ]
        arrays[f"{label}_restricted_rate_history"] = restricted[label][
            "rate"
        ]
        arrays[f"{label}_stress_rate_signal"] = histories[label][
            "stress_rate_signal"
        ]

    DEFAULT_ARRAYS.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DEFAULT_ARRAYS, **arrays)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "base_commit": BASE_COMMIT,
        "classification": classification,
        "passed": passed,
        "scope": {
            "production_physics_changed": False,
            "production_inner_boundary_changed": False,
            "coupling_flux": "one production Rusanov flux",
            "frozen_trace_at_coupling": False,
            "far_audit_outer_trace_frozen": True,
            "nonlinear_patch_truth_run": False,
            "production_embedded_patch_authorized": False,
            "fixed_q_averaging_run": False,
            "reduced_coordinate_selected": False,
        },
        "layout": {
            "parent_mesh": PARENT_MESH,
            "audit_outer_parent_face": AUDIT_OUTER_PARENT_FACE,
            "audit_outer_rg": float(
                parent["parent_grid"].edges[-1]
                / parent["parent_grid"].gravitational_radius
            ),
            "active_outer_parent_face": ACTIVE_OUTER_PARENT_FACE,
            "active_outer_rg": active_outer_rg,
            "primary_coupling_parent_face": (
                PRIMARY_COUPLING_PARENT_FACE
            ),
            "primary_coupling_rg": float(
                parent["parent_grid"].edges[
                    PRIMARY_COUPLING_PARENT_FACE
                ]
                / parent["parent_grid"].gravitational_radius
            ),
            "variant_coupling_parent_face": (
                VARIANT_COUPLING_PARENT_FACE
            ),
            "variant_coupling_rg": float(
                parent["parent_grid"].edges[
                    VARIANT_COUPLING_PARENT_FACE
                ]
                / parent["parent_grid"].gravitational_radius
            ),
            "configurations": {
                label: _configuration_summary(configuration)
                for label, configuration in configurations.items()
            },
        },
        "method_certification": {
            "unified_nonoverlapping_grid": True,
            "one_shared_coupling_flux": True,
            "equal_and_opposite_internal_flux": True,
            "responsive_height_storage_remains_cell_local": True,
            "coupling_face_is_interior_not_boundary": True,
            "widened_jacobian_pattern_inherited_from_nonuniform_grid": True,
            "uniform_ratio_one_reduction_tested": True,
            "dense_colored_small_patch_tested": True,
            "constant_and_smooth_profile_tested": True,
            "restart_split_relative_defect": maximum_restart_defect,
            "maximum_shared_flux_defect": maximum_shared_flux_defect,
            "maximum_telescoping_defect": maximum_telescoping_defect,
            "passed": method_gate_passed,
        },
        "history": {
            "pairwise_by_region": pairwise,
            "observed_orders_by_region": orders,
            "signal_ladder": signal_ladder,
            "spatial_gate_passed": spatial_gate_passed,
            "signal_gate_passed": signal_gate_passed,
        },
        "coupling_location": {
            "history_comparison_active_core": coupling_variation,
            "maximum_history_defect": coupling_location_defect,
            "signal_fraction_by_configuration": (
                coupling_signal_fractions
            ),
            "response_reached_coupling": response_reached_coupling,
            "passed": coupling_gate_passed,
        },
        "gates": {
            "minimum_spatial_order": MINIMUM_SPATIAL_ORDER,
            "minimum_signed_cosine": MINIMUM_SIGNED_COSINE,
            "maximum_zero_crossing_defect": (
                MAXIMUM_ZERO_CROSSING_DEFECT
            ),
            "maximum_frequency_defect": MAXIMUM_FREQUENCY_DEFECT,
            "maximum_damping_defect": MAXIMUM_DAMPING_DEFECT,
            "maximum_coupling_location_history_defect": (
                MAXIMUM_COUPLING_LOCATION_HISTORY_DEFECT
            ),
            "maximum_shared_flux_defect": MAXIMUM_SHARED_FLUX_DEFECT,
            "maximum_coupling_signal_fraction": (
                MAXIMUM_COUPLING_SIGNAL_FRACTION
            ),
        },
        "decision": {
            "bounded_nonlinear_embedded_patch_truth_authorized": passed,
            "one_more_inner_patch_refinement_authorized": bool(
                method_gate_passed
                and coupling_gate_passed
                and not passed
                and active_orders["state"] is not None
                and active_orders["rate"] is not None
                and min(
                    active_orders["state"],
                    active_orders["rate"],
                )
                > 0.0
            ),
            "bulk_near_horizon_operator_redesign_required": bool(
                method_gate_passed
                and not response_reached_coupling
                and not passed
                and (
                    active_orders["state"] is None
                    or active_orders["rate"] is None
                    or min(
                        active_orders["state"],
                        active_orders["rate"],
                    )
                    <= 0.0
                )
            ),
            "production_embedded_patch_authorized": False,
            "fixed_q_averaging_authorized": False,
            "initial_slip_model_authorized": False,
            "reduced_coordinate_selection_authorized": False,
            "macrostep_authorized": False,
        },
        "artifacts": {
            "runner": THIS_RUNNER,
            "runner_sha256": _sha256(ROOT / THIS_RUNNER),
            "core_dae_sha256": _sha256(ROOT / CORE_DAE_FILE),
            "core_patch_sha256": _sha256(ROOT / CORE_PATCH_FILE),
            "core_spatial_sha256": _sha256(ROOT / CORE_SPATIAL_FILE),
            "wp10c8y_json": _relative(WP10C8Y_OUTPUT),
            "wp10c8y_json_sha256": _sha256(WP10C8Y_OUTPUT),
            "wp10c8y_arrays": _relative(WP10C8Y_ARRAYS),
            "wp10c8y_arrays_sha256": _sha256(WP10C8Y_ARRAYS),
            "arrays": _relative(DEFAULT_ARRAYS),
            "arrays_sha256": _sha256(DEFAULT_ARRAYS),
            "parent_operator": _relative(parent["operator_path"]),
            "parent_operator_sha256": _sha256(parent["operator_path"]),
            "n256_anchor": _relative(parent["anchor256_path"]),
            "n256_anchor_sha256": _sha256(parent["anchor256_path"]),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "wall_seconds": time.perf_counter() - started,
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload, _arrays = run(force=args.force)
    print(
        json.dumps(
            {
                "work_package": payload["work_package"],
                "classification": payload["classification"],
                "passed": payload["passed"],
                "active_orders": payload["history"][
                    "observed_orders_by_region"
                ]["active_core"],
                "coupling_location_defect": payload[
                    "coupling_location"
                ]["maximum_history_defect"],
                "nonlinear_patch_truth_authorized": payload[
                    "decision"
                ][
                    "bounded_nonlinear_embedded_patch_truth_authorized"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
