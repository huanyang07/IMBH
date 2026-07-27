"""Run the WP10c9d0 conservative inner-micro-solver export preflight.

This package is deliberately cache first and production neutral.  It does not
construct a constrained fast problem.  Instead it asks the preceding question:
do the already cached common-mode and conservative embedded-patch tangent
histories produce mesh-convergent conservative quantities that a future inner
micro-solver would export?

The binding observables are the inner and coupling M/J/E fluxes, the integrated
M/J/E source, cooling and responsive-height work, and their conservative net
drive.  Instantaneous and time-integrated histories are assessed separately.
For the uniform N64/N128/N256 ladder, the cached complete descriptor,
rate-dependent descriptor derivative, and stationary Jacobian provide an
independent responsive-height ledger closure.
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

import run_causal_inner_common_mode_audit_wp10c8y as wp10c8y
import run_causal_inner_embedded_patch_preflight_wp10c8z as wp10c8z
import run_causal_inner_family_transfer_audit_wp10c9c0c as wp10c9c0c

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    causal_five_field_state_from_primitives,
    evaluate_causal_five_field_dae,
    pack_causal_five_field_state,
)


BASE_COMMIT = "90f82c238e802abe22aa15b42f62b7d929048a60"
WORK_PACKAGE = "WP10c9d0"
SCHEMA_VERSION = 1
THIS_RUNNER = (
    "scripts/run_causal_inner_micro_export_preflight_wp10c9d0.py"
)
WP10C8Y_ARRAYS = (
    ROOT
    / "outputs/tables/causal_inner_common_mode_audit_wp10c8y_arrays.npz"
)
WP10C8Z_OUTPUT = (
    ROOT
    / "outputs/tables/causal_inner_embedded_patch_preflight_wp10c8z.json"
)
WP10C8Z_ARRAYS = (
    ROOT
    / "outputs/tables/"
    "causal_inner_embedded_patch_preflight_wp10c8z_arrays.npz"
)
WP10C9C0D_OUTPUT = (
    ROOT
    / "outputs/tables/"
    "causal_inner_common_block_family_audit_wp10c9c0d.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/tables/"
    "causal_inner_micro_export_preflight_wp10c9d0.json"
)
DEFAULT_ARRAYS = (
    ROOT
    / "outputs/tables/"
    "causal_inner_micro_export_preflight_wp10c9d0_arrays.npz"
)
CACHE_ROOT = (
    ROOT
    / "outputs/checkpoints/"
    "causal_inner_micro_export_preflight_wp10c9d0"
)

CONSERVATIVE_FIELDS = np.asarray((0, 2, 3), dtype=int)
FIELD_NAMES = ("mass", "angular_momentum", "killing_energy")
FINITE_DIFFERENCE_STEP = 2.0e-4
FINITE_DIFFERENCE_STEPS = (1.0e-4, 2.0e-4, 4.0e-4)
SAMPLE_STRIDE = 2
MINIMUM_SPATIAL_ORDER = 0.75
MINIMUM_SIGNED_COSINE = 0.90
MAXIMUM_FINE_NORMALIZED_DIFFERENCE = 0.10
# This reserves one percent of the 0.10 scientific observable gate for the
# centered directional-map uncertainty.  The independent stationary-matrix
# comparison below remains the stronger ledger-specific check.
MAXIMUM_DIRECTIONAL_STEP_DEFECT = 1.0e-3
MAXIMUM_TEMPORAL_SAMPLING_DEFECT = 5.0e-3
MAXIMUM_DIRECT_MATRIX_DRIVE_DEFECT = 2.0e-3
MAXIMUM_GENERATOR_LEDGER_DEFECT = 1.0e-8
MINIMUM_RELATIVE_ACTIVITY = 1.0e-8
MINIMUM_VECTOR_NORM = 1.0e-10

OBSERVABLE_NAMES = tuple(
    [
        *(f"inner_flux_{name}" for name in FIELD_NAMES),
        *(f"interface_flux_{name}" for name in FIELD_NAMES),
        *(f"net_drive_{name}" for name in FIELD_NAMES),
        "cooling_angular_momentum",
        "cooling_killing_energy",
        "vertical_work_angular_momentum",
        "vertical_work_killing_energy",
    ]
)
GROUPS = {
    "boundary_flux": np.arange(0, 6, dtype=int),
    "net_drive": np.arange(6, 9, dtype=int),
    "cooling_and_height": np.arange(9, 13, dtype=int),
    "exported": np.arange(0, 13, dtype=int),
}


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
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            return None
        return number
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: np.asarray(source[key]) for key in source.files}


def _sample_indices(size: int) -> np.ndarray:
    indices = np.arange(0, int(size), SAMPLE_STRIDE, dtype=int)
    if indices[-1] != size - 1:
        indices = np.concatenate((indices, np.asarray([size - 1])))
    return indices


def _cumulative_trapezoid(
    times: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    t = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)
    if t.ndim != 1 or y.shape[0] != t.size or t.size < 2:
        raise ValueError("cumulative-trapezoid inputs are invalid")
    result = np.zeros_like(y)
    increments = (
        0.5
        * np.diff(t)[:, None]
        * (y[1:] + y[:-1])
    )
    result[1:] = np.cumsum(increments, axis=0)
    return result


def _stationary_observables(
    context,
    primitives: np.ndarray,
    *,
    interface_face: int,
    active_cells: int,
) -> np.ndarray:
    state = causal_five_field_state_from_primitives(context, primitives)
    evaluation = evaluate_causal_five_field_dae(
        pack_causal_five_field_state(state),
        context,
    )
    fluxes = np.asarray(
        evaluation.numerical_weighted_face_fluxes_over_c,
        dtype=float,
    )
    sources = np.sum(
        np.asarray(
            evaluation.integrated_sources_per_ct[:active_cells],
            dtype=float,
        ),
        axis=0,
    )
    cooling = np.sum(
        np.asarray(
            evaluation.integrated_source_components_per_ct[
                "radiative_cooling"
            ][:active_cells],
            dtype=float,
        ),
        axis=0,
    )
    vertical = np.sum(
        np.asarray(
            evaluation.integrated_source_components_per_ct[
                "vertical_work"
            ][:active_cells],
            dtype=float,
        ),
        axis=0,
    )
    inner = fluxes[0, CONSERVATIVE_FIELDS]
    interface = fluxes[int(interface_face), CONSERVATIVE_FIELDS]
    net = inner - interface + sources[CONSERVATIVE_FIELDS]
    return np.concatenate(
        (
            inner,
            interface,
            net,
            cooling[CONSERVATIVE_FIELDS[1:]],
            vertical[CONSERVATIVE_FIELDS[1:]],
        )
    )


def _directional_observables(
    context,
    base_primitives: np.ndarray,
    physical_direction: np.ndarray,
    *,
    interface_face: int,
    active_cells: int,
    step: float,
) -> np.ndarray:
    delta = float(step) * np.asarray(physical_direction, dtype=float)
    plus = _stationary_observables(
        context,
        np.asarray(base_primitives, dtype=float) + delta,
        interface_face=interface_face,
        active_cells=active_cells,
    )
    minus = _stationary_observables(
        context,
        np.asarray(base_primitives, dtype=float) - delta,
        interface_face=interface_face,
        active_cells=active_cells,
    )
    return (plus - minus) / (2.0 * float(step))


def _directional_step_defect(
    responses: dict[float, np.ndarray],
    base_observable: np.ndarray,
) -> float:
    reference = np.asarray(
        responses[FINITE_DIFFERENCE_STEP],
        dtype=float,
    )
    baseline = np.asarray(base_observable, dtype=float)
    activity = np.max(
        np.abs(np.asarray(list(responses.values()), dtype=float)),
        axis=0,
    )
    significant = activity >= (
        MINIMUM_RELATIVE_ACTIVITY * np.maximum(np.abs(baseline), 1.0)
    )
    if not np.any(significant):
        return 0.0
    scale = np.maximum(activity[significant], np.finfo(float).tiny)
    return float(
        max(
            np.max(
                np.abs(values[significant] - reference[significant])
                / scale
            )
            for step, values in responses.items()
            if step != FINITE_DIFFERENCE_STEP
        )
    )


def _signal_cache_contract(
    *,
    label: str,
    base_primitives: np.ndarray,
    amplitudes: np.ndarray,
    state_history: np.ndarray,
    times: np.ndarray,
    interface_face: int,
    active_cells: int,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "label": label,
        "finite_difference_step": FINITE_DIFFERENCE_STEP,
        "sample_stride": SAMPLE_STRIDE,
        "base_sha256": _array_sha256(base_primitives),
        "amplitudes_sha256": _array_sha256(amplitudes),
        "state_history_sha256": _array_sha256(state_history),
        "times_sha256": _array_sha256(times),
        "interface_face": int(interface_face),
        "active_cells": int(active_cells),
    }


def _build_or_load_signal_history(
    *,
    label: str,
    context,
    base_primitives: np.ndarray,
    amplitudes: np.ndarray,
    state_history: np.ndarray,
    times: np.ndarray,
    interface_face: int,
    active_cells: int,
    force: bool,
) -> tuple[dict, dict[str, np.ndarray]]:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = CACHE_ROOT / f"{label}.json"
    arrays_path = CACHE_ROOT / f"{label}_arrays.npz"
    contract = _signal_cache_contract(
        label=label,
        base_primitives=base_primitives,
        amplitudes=amplitudes,
        state_history=state_history,
        times=times,
        interface_face=interface_face,
        active_cells=active_cells,
    )
    if not force and json_path.exists() and arrays_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if (
            all(payload.get(key) == value for key, value in contract.items())
            and payload.get("arrays_sha256") == _sha256(arrays_path)
        ):
            payload["passed"] = bool(
                payload["maximum_directional_step_defect"]
                <= MAXIMUM_DIRECTIONAL_STEP_DEFECT
                and payload["temporal_sampling_defect"]
                <= MAXIMUM_TEMPORAL_SAMPLING_DEFECT
            )
            return payload, _load_npz(arrays_path)

    started = time.perf_counter()
    indices = _sample_indices(state_history.shape[0])
    sampled_times = np.asarray(times[indices], dtype=float)
    base_observable = _stationary_observables(
        context,
        base_primitives,
        interface_face=interface_face,
        active_cells=active_cells,
    )
    signals = []
    for output_index, history_index in enumerate(indices):
        if output_index % 10 == 0 or output_index + 1 == indices.size:
            print(
                f"WP10c9d0: {label} observable "
                f"{output_index + 1}/{indices.size}",
                flush=True,
            )
        physical_direction = (
            np.asarray(amplitudes, dtype=float)
            * np.asarray(state_history[history_index], dtype=float)
        )
        signals.append(
            _directional_observables(
                context,
                base_primitives,
                physical_direction,
                interface_face=interface_face,
                active_cells=active_cells,
                step=FINITE_DIFFERENCE_STEP,
            )
        )
    signals_array = np.asarray(signals, dtype=float)
    plateau_indices = np.asarray(
        sorted(
            {
                0,
                int(np.argmin(np.abs(sampled_times - 0.04125))),
                sampled_times.size - 1,
            }
        ),
        dtype=int,
    )
    plateau_defects = []
    for sampled_index in plateau_indices:
        direction = (
            np.asarray(amplitudes, dtype=float)
            * np.asarray(
                state_history[indices[sampled_index]],
                dtype=float,
            )
        )
        responses = {
            step: _directional_observables(
                context,
                base_primitives,
                direction,
                interface_face=interface_face,
                active_cells=active_cells,
                step=step,
            )
            for step in FINITE_DIFFERENCE_STEPS
        }
        plateau_defects.append(
            _directional_step_defect(responses, base_observable)
        )
    cumulative = _cumulative_trapezoid(sampled_times, signals_array)
    coarse_cumulative = _cumulative_trapezoid(
        sampled_times[::2],
        signals_array[::2],
    )
    cumulative_scale = np.maximum(
        np.max(np.abs(cumulative), axis=0),
        np.finfo(float).tiny,
    )
    active_cumulative = cumulative_scale >= (
        MINIMUM_RELATIVE_ACTIVITY
        * np.maximum(np.abs(base_observable), 1.0)
        * max(float(sampled_times[-1]), np.finfo(float).tiny)
    )
    if np.any(active_cumulative):
        temporal_sampling_defect = float(
            np.max(
                np.abs(
                    coarse_cumulative[-1, active_cumulative]
                    - cumulative[-1, active_cumulative]
                )
                / cumulative_scale[active_cumulative]
            )
        )
    else:
        temporal_sampling_defect = 0.0
    arrays = {
        "indices": indices,
        "times": sampled_times,
        "base_observable": base_observable,
        "signals": signals_array,
        "cumulative_signals": cumulative,
        "plateau_indices": plateau_indices,
        "plateau_defects": np.asarray(plateau_defects, dtype=float),
    }
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **arrays)
    maximum_plateau = float(np.max(plateau_defects))
    payload = {
        **contract,
        "arrays_path": _relative(arrays_path),
        "arrays_sha256": _sha256(arrays_path),
        "sample_count": int(indices.size),
        "maximum_directional_step_defect": maximum_plateau,
        "temporal_sampling_defect": temporal_sampling_defect,
        "wall_seconds": time.perf_counter() - started,
    }
    payload["passed"] = bool(
        maximum_plateau <= MAXIMUM_DIRECTIONAL_STEP_DEFECT
        and temporal_sampling_defect <= MAXIMUM_TEMPORAL_SAMPLING_DEFECT
    )
    json_path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload, arrays


def _pair_metrics(
    first: np.ndarray,
    second: np.ndarray,
    scales: np.ndarray,
    significant: np.ndarray,
) -> dict:
    if not np.any(significant):
        return {
            "available": False,
            "reason": "no_significant_components",
        }
    a = np.asarray(first, dtype=float)[:, significant] / scales[significant]
    b = np.asarray(second, dtype=float)[:, significant] / scales[significant]
    difference = b - a
    difference_norm = np.sqrt(np.mean(difference**2, axis=1))
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)
    active = (a_norm >= MINIMUM_VECTOR_NORM) & (
        b_norm >= MINIMUM_VECTOR_NORM
    )
    cosines = np.full(a_norm.shape, np.nan, dtype=float)
    cosines[active] = np.sum(a[active] * b[active], axis=1) / (
        a_norm[active] * b_norm[active]
    )
    return {
        "available": True,
        "rms_history_difference": float(
            np.sqrt(np.mean(difference**2))
        ),
        "maximum_time_rms_difference": float(
            np.max(difference_norm)
        ),
        "maximum_component_difference": float(
            np.max(np.abs(difference))
        ),
        "final_time_rms_difference": float(difference_norm[-1]),
        "minimum_signed_cosine": (
            float(np.nanmin(cosines)) if np.any(active) else None
        ),
        "final_signed_cosine": (
            float(cosines[-1]) if active[-1] else None
        ),
        "fine_final_norm": float(b_norm[-1]),
    }


def _observed_order(coarse: float, fine: float) -> float | None:
    if (
        not np.isfinite(coarse)
        or not np.isfinite(fine)
        or coarse <= 0.0
        or fine <= 0.0
    ):
        return None
    return float(np.log2(coarse / fine))


def _ladder_metrics(
    labels: tuple[str, str, str],
    histories: dict[str, np.ndarray],
    baselines: dict[str, np.ndarray],
    *,
    indices: np.ndarray,
) -> dict:
    selected = np.asarray(indices, dtype=int)
    response_scale = np.max(
        np.abs(
            np.asarray(
                [histories[label][:, selected] for label in labels],
                dtype=float,
            )
        ),
        axis=(0, 1),
    )
    baseline_scale = np.max(
        np.abs(
            np.asarray(
                [baselines[label][selected] for label in labels],
                dtype=float,
            )
        ),
        axis=0,
    )
    significant = response_scale >= (
        MINIMUM_RELATIVE_ACTIVITY * np.maximum(baseline_scale, 1.0)
    )
    scales = np.maximum(response_scale, np.finfo(float).tiny)
    first = _pair_metrics(
        histories[labels[0]][:, selected],
        histories[labels[1]][:, selected],
        scales,
        significant,
    )
    second = _pair_metrics(
        histories[labels[1]][:, selected],
        histories[labels[2]][:, selected],
        scales,
        significant,
    )
    if not first["available"] or not second["available"]:
        return {
            "significant_components": [],
            "coarse_medium": first,
            "medium_fine": second,
            "passed": False,
        }
    order_rms = _observed_order(
        first["rms_history_difference"],
        second["rms_history_difference"],
    )
    order_maximum = _observed_order(
        first["maximum_component_difference"],
        second["maximum_component_difference"],
    )
    minimum_cosine = second["minimum_signed_cosine"]
    passed = bool(
        order_rms is not None
        and order_rms >= MINIMUM_SPATIAL_ORDER
        and order_maximum is not None
        and order_maximum >= MINIMUM_SPATIAL_ORDER
        and second["maximum_component_difference"]
        <= MAXIMUM_FINE_NORMALIZED_DIFFERENCE
        and minimum_cosine is not None
        and minimum_cosine >= MINIMUM_SIGNED_COSINE
    )
    return {
        "significant_components": [
            OBSERVABLE_NAMES[int(selected[index])]
            for index in np.flatnonzero(significant)
        ],
        "component_scales": scales[significant],
        "coarse_medium": first,
        "medium_fine": second,
        "observed_order_rms": order_rms,
        "observed_order_maximum": order_maximum,
        "passed": passed,
    }


def _uniform_ledger(
    *,
    mesh: int,
    operator: dict[str, np.ndarray],
    amplitudes: np.ndarray,
    state_history: np.ndarray,
    rate_history: np.ndarray,
    signal_history: np.ndarray,
    sample_indices: np.ndarray,
    times: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    native_scales = np.asarray(
        operator["primitive_column_scales"],
        dtype=float,
    ).ravel()
    common_scales = np.asarray(amplitudes, dtype=float).ravel()
    ratio = common_scales / native_scales
    descriptor = np.asarray(operator["descriptor"], dtype=float)
    storage_rate = np.asarray(
        operator["storage_rate_derivative"],
        dtype=float,
    )
    stationary = np.asarray(
        operator["stationary_jacobian"],
        dtype=float,
    )
    row_scales = np.asarray(
        operator["conservation_row_scales"],
        dtype=float,
    ).reshape(mesh // 64 * 24, 5)

    storage_states = []
    total_storage_rates = []
    matrix_net_drives = []
    for index in sample_indices:
        scaled_state = (
            np.asarray(state_history[index], dtype=float).ravel() * ratio
        )
        scaled_rate = (
            np.asarray(rate_history[index], dtype=float).ravel() * ratio
        )
        storage_state_rows = (
            row_scales
            * (descriptor @ scaled_state).reshape(row_scales.shape)
        )
        total_storage_rate_rows = (
            row_scales
            * (
                descriptor @ scaled_rate
                + storage_rate @ scaled_state
            ).reshape(row_scales.shape)
        )
        matrix_net_rows = (
            -row_scales
            * (stationary @ scaled_state).reshape(row_scales.shape)
        )
        storage_states.append(
            np.sum(storage_state_rows[:, CONSERVATIVE_FIELDS], axis=0)
        )
        total_storage_rates.append(
            np.sum(
                total_storage_rate_rows[:, CONSERVATIVE_FIELDS],
                axis=0,
            )
        )
        matrix_net_drives.append(
            np.sum(matrix_net_rows[:, CONSERVATIVE_FIELDS], axis=0)
        )
    storage_states_array = np.asarray(storage_states, dtype=float)
    total_rates_array = np.asarray(total_storage_rates, dtype=float)
    matrix_net_array = np.asarray(matrix_net_drives, dtype=float)
    direct_net = np.asarray(signal_history[:, 6:9], dtype=float)
    scale = np.maximum(
        np.max(
            np.abs(
                np.concatenate(
                    (total_rates_array, matrix_net_array, direct_net),
                    axis=0,
                )
            ),
            axis=0,
        ),
        np.finfo(float).tiny,
    )
    direct_matrix_defect = float(
        np.max(np.abs(direct_net - matrix_net_array) / scale)
    )
    generator_defect = float(
        np.max(np.abs(total_rates_array - matrix_net_array) / scale)
    )
    cumulative = _cumulative_trapezoid(times, matrix_net_array)
    storage_change = storage_states_array - storage_states_array[0]
    storage_scale = np.maximum(
        np.max(
            np.abs(np.concatenate((cumulative, storage_change), axis=0)),
            axis=0,
        ),
        np.finfo(float).tiny,
    )
    cumulative_defect = float(
        np.max(np.abs(cumulative - storage_change) / storage_scale)
    )
    payload = {
        "mesh": int(mesh),
        "maximum_direct_matrix_drive_defect": direct_matrix_defect,
        "maximum_generator_ledger_defect": generator_defect,
        "maximum_cumulative_ledger_defect": cumulative_defect,
    }
    # The frozen evolving tangent contains the base-rate derivative DM[p_dot].
    # Its instantaneous balance is exact, but the frozen system does not
    # evolve the background descriptor M(p).  Therefore integral(net_drive)
    # need not equal M(base) delta-p(t)-M(base) delta-p(0).  Retain that
    # non-integrability measurement as a diagnostic and bind only the
    # instantaneous physical-map and generator ledger identities.
    payload["passed"] = bool(
        direct_matrix_defect <= MAXIMUM_DIRECT_MATRIX_DRIVE_DEFECT
        and generator_defect <= MAXIMUM_GENERATOR_LEDGER_DEFECT
    )
    return payload, {
        "storage_state_per_c": storage_states_array,
        "total_storage_rate_per_ct": total_rates_array,
        "matrix_net_drive_per_ct": matrix_net_array,
        "direct_net_drive_per_ct": direct_net,
        "cumulative_matrix_net_drive": cumulative,
        "storage_state_change_per_c": storage_change,
    }


def _patch_configurations(
    patch_arrays: dict[str, np.ndarray],
) -> dict[str, dict]:
    parent = wp10c8z._parent_data()
    active_outer_rg = float(
        parent["parent_grid"].edges[
            wp10c8z.ACTIVE_OUTER_PARENT_FACE
        ]
        / parent["parent_grid"].gravitational_radius
    )
    parent["active_outer_rg"] = active_outer_rg
    labels = {
        1: "N128_exterior_N128_inner_c48",
        2: "N128_exterior_N256_inner_c48",
        4: "N128_exterior_N512_inner_c48",
    }
    result = {}
    for ratio, label in labels.items():
        configuration = wp10c8z._configuration(
            label=label,
            parent=parent,
            coupling_face=wp10c8z.PRIMARY_COUPLING_PARENT_FACE,
            refinement_ratio=ratio,
            target_values=np.asarray(
                patch_arrays["target_coordinate_values"],
                dtype=float,
            ),
            target_scales=np.asarray(
                patch_arrays["target_coordinate_scales"],
                dtype=float,
            ),
            active_outer_rg=active_outer_rg,
            force=False,
        )
        configuration["active_outer_rg"] = active_outer_rg
        result[label] = configuration
    return result


def run(*, force: bool = False) -> tuple[dict, dict[str, np.ndarray]]:
    started = time.perf_counter()
    required = (
        WP10C8Y_ARRAYS,
        WP10C8Z_OUTPUT,
        WP10C8Z_ARRAYS,
        WP10C9C0D_OUTPUT,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "WP10c9d0 requires prior cache evidence: " + ", ".join(missing)
        )
    c0d = json.loads(WP10C9C0D_OUTPUT.read_text(encoding="utf-8"))
    if c0d.get("classification") != (
        "common_mode_defect_remains_multiblock_after_direct_ledger"
    ):
        raise RuntimeError("WP10c9c0d authorization changed")
    c8z_payload = json.loads(WP10C8Z_OUTPUT.read_text(encoding="utf-8"))
    if not c8z_payload.get("method_certification", {}).get(
        "passed",
        False,
    ):
        raise RuntimeError("WP10c8z conservative patch contract changed")

    c8y = _load_npz(WP10C8Y_ARRAYS)
    c8z = _load_npz(WP10C8Z_ARRAYS)
    contexts, profiles = wp10c9c0c._common_contexts()
    operators = wp10c8y._load_family_operators()["production"]
    arrays: dict[str, np.ndarray] = {}
    signal_reports = {}
    uniform_signals = {}
    uniform_cumulative = {}
    uniform_baselines = {}
    uniform_ledgers = {}

    for mesh in wp10c8y.MESHES:
        label = f"uniform_N{mesh}"
        print(f"WP10c9d0: preparing {label}", flush=True)
        full_times = np.asarray(
            c8y[f"production_N{mesh}_times"],
            dtype=float,
        )
        state_history = np.asarray(
            c8y[f"production_N{mesh}_state"],
            dtype=float,
        )
        rate_history = np.asarray(
            c8y[f"production_N{mesh}_rate"],
            dtype=float,
        )
        amplitudes = np.asarray(
            c8y[f"N{mesh}_common_amplitudes"],
            dtype=float,
        )
        context = contexts[mesh]
        active_cells = int(context.grid.centers.size)
        report, cached = _build_or_load_signal_history(
            label=label,
            context=context,
            base_primitives=np.asarray(
                profiles[mesh]["primitives"],
                dtype=float,
            ),
            amplitudes=amplitudes,
            state_history=state_history,
            times=full_times,
            interface_face=active_cells,
            active_cells=active_cells,
            force=force,
        )
        signal_reports[label] = report
        uniform_signals[label] = cached["signals"]
        uniform_cumulative[label] = cached["cumulative_signals"]
        uniform_baselines[label] = cached["base_observable"]
        for key, values in cached.items():
            arrays[f"{label}_{key}"] = values
        ledger, ledger_arrays = _uniform_ledger(
            mesh=mesh,
            operator=operators[mesh],
            amplitudes=amplitudes,
            state_history=state_history,
            rate_history=rate_history,
            signal_history=cached["signals"],
            sample_indices=np.asarray(cached["indices"], dtype=int),
            times=np.asarray(cached["times"], dtype=float),
        )
        uniform_ledgers[label] = ledger
        for key, values in ledger_arrays.items():
            arrays[f"{label}_{key}"] = values

    patch_configurations = _patch_configurations(c8z)
    patch_labels = (
        "N128_exterior_N128_inner_c48",
        "N128_exterior_N256_inner_c48",
        "N128_exterior_N512_inner_c48",
    )
    patch_signals = {}
    patch_cumulative = {}
    patch_baselines = {}
    patch_shared_flux = {}
    patch_times = np.asarray(c8z["times"], dtype=float)
    for label in patch_labels:
        print(f"WP10c9d0: preparing {label}", flush=True)
        configuration = patch_configurations[label]
        layout = configuration["layout"]
        cached_base = np.asarray(
            c8z[f"{label}_base_primitives"],
            dtype=float,
        )
        configuration_base = np.asarray(
            configuration["base_primitives"],
            dtype=float,
        )
        base_defect = float(np.max(np.abs(cached_base - configuration_base)))
        if base_defect != 0.0:
            raise RuntimeError(
                f"{label} reconstructed base differs from cached history"
            )
        report, cached = _build_or_load_signal_history(
            label=label,
            context=configuration["context"],
            base_primitives=cached_base,
            amplitudes=np.asarray(
                configuration["amplitudes"],
                dtype=float,
            ),
            state_history=np.asarray(
                c8z[f"{label}_state_history"],
                dtype=float,
            ),
            times=patch_times,
            interface_face=int(layout.coupling_face_index),
            active_cells=int(layout.coupling_face_index),
            force=force,
        )
        signal_reports[label] = report
        patch_signals[label] = cached["signals"]
        patch_cumulative[label] = cached["cumulative_signals"]
        patch_baselines[label] = cached["base_observable"]
        patch_shared_flux[label] = {
            "maximum_state_flux_defect": float(
                configuration["flux_audit"].maximum_state_flux_defect
            ),
            "maximum_telescoping_defect": float(
                configuration["flux_audit"].maximum_telescoping_defect
            ),
            "passed": bool(configuration["flux_audit"].passed),
        }
        for key, values in cached.items():
            arrays[f"patch_{label}_{key}"] = values

    uniform_labels = tuple(
        f"uniform_N{mesh}" for mesh in wp10c8y.MESHES
    )
    ladders = {"uniform": {}, "embedded_patch": {}}
    for group, indices in GROUPS.items():
        ladders["uniform"][group] = {
            "instantaneous": _ladder_metrics(
                uniform_labels,
                uniform_signals,
                uniform_baselines,
                indices=indices,
            ),
            "cumulative": _ladder_metrics(
                uniform_labels,
                uniform_cumulative,
                {
                    label: np.zeros_like(uniform_baselines[label])
                    for label in uniform_labels
                },
                indices=indices,
            ),
        }
        ladders["embedded_patch"][group] = {
            "instantaneous": _ladder_metrics(
                patch_labels,
                patch_signals,
                patch_baselines,
                indices=indices,
            ),
            "cumulative": _ladder_metrics(
                patch_labels,
                patch_cumulative,
                {
                    label: np.zeros_like(patch_baselines[label])
                    for label in patch_labels
                },
                indices=indices,
            ),
        }

    method_contract_passed = bool(
        all(report["passed"] for report in signal_reports.values())
        and all(item["passed"] for item in uniform_ledgers.values())
        and all(item["passed"] for item in patch_shared_flux.values())
    )
    uniform_instantaneous = bool(
        ladders["uniform"]["exported"]["instantaneous"]["passed"]
    )
    uniform_cumulative_passed = bool(
        ladders["uniform"]["exported"]["cumulative"]["passed"]
    )
    patch_instantaneous = bool(
        ladders["embedded_patch"]["exported"]["instantaneous"]["passed"]
    )
    patch_cumulative_passed = bool(
        ladders["embedded_patch"]["exported"]["cumulative"]["passed"]
    )
    cumulative_export_passed = bool(
        uniform_cumulative_passed and patch_cumulative_passed
    )
    instantaneous_export_passed = bool(
        uniform_instantaneous and patch_instantaneous
    )
    fixed_q_micro_solver_authorized = bool(
        method_contract_passed and cumulative_export_passed
    )
    if fixed_q_micro_solver_authorized:
        classification = (
            "conservative_cumulative_exports_authorize_constrained_"
            "micro_solver_feasibility"
        )
        next_action = (
            "construct a constraint-consistent, ledger-complete fixed-Q "
            "micro-solver feasibility experiment with multiple equal-Q lifts"
        )
    else:
        classification = (
            "conservative_micro_exports_fail_spatial_gate"
        )
        next_action = (
            "redesign the complete coupled near-horizon spatial operator "
            "before any fixed-Q averaging or reduced slow evolution"
        )

    arrays_path = DEFAULT_ARRAYS
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **arrays)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "base_commit": BASE_COMMIT,
        "runner": THIS_RUNNER,
        "classification": classification,
        "method_contract_passed": method_contract_passed,
        "instantaneous_export_passed": instantaneous_export_passed,
        "cumulative_export_passed": cumulative_export_passed,
        "fixed_q_micro_solver_authorized": (
            fixed_q_micro_solver_authorized
        ),
        "next_action": next_action,
        "observable_names": OBSERVABLE_NAMES,
        "gates": {
            "minimum_spatial_order": MINIMUM_SPATIAL_ORDER,
            "minimum_signed_cosine": MINIMUM_SIGNED_COSINE,
            "maximum_fine_normalized_difference": (
                MAXIMUM_FINE_NORMALIZED_DIFFERENCE
            ),
            "maximum_directional_step_defect": (
                MAXIMUM_DIRECTIONAL_STEP_DEFECT
            ),
            "maximum_temporal_sampling_defect": (
                MAXIMUM_TEMPORAL_SAMPLING_DEFECT
            ),
            "maximum_direct_matrix_drive_defect": (
                MAXIMUM_DIRECT_MATRIX_DRIVE_DEFECT
            ),
            "maximum_generator_ledger_defect": (
                MAXIMUM_GENERATOR_LEDGER_DEFECT
            ),
        },
        "diagnostics": {
            "frozen_tangent_cumulative_storage_is_nonbinding": True,
            "reason": (
                "the frozen evolving tangent retains the base-rate "
                "descriptor derivative without evolving the background "
                "descriptor, so it has no exact finite-time M(base) state "
                "integral"
            ),
        },
        "signal_contracts": signal_reports,
        "uniform_ledgers": uniform_ledgers,
        "embedded_patch_shared_flux": patch_shared_flux,
        "ladders": ladders,
        "input_evidence": {
            _relative(path): _sha256(path) for path in required
        },
        "arrays_path": _relative(arrays_path),
        "arrays_sha256": _sha256(arrays_path),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "wall_seconds": time.perf_counter() - started,
    }
    DEFAULT_OUTPUT.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload, arrays


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    payload, _arrays = run(force=arguments.force)
    print(json.dumps(_plain(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
