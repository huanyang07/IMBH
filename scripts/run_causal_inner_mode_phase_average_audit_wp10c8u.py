"""Run the WP10c8u cached inner-mode phase and activity audit.

WP10c8t establishes that the exact equal-q34 shell-0 mode remains persistent
and localized through 0.125 s on N64 and N128, while the late instantaneous
rate direction and amplitude fail their same-time cross-mesh gate.  This
package does not evolve another truth trajectory.  It consumes every stored
coarse/fine plus/minus WP10c8t state and:

* deterministically recomputes the fresh primitive and 34-coordinate rates;
* reproduces the committed sparse rate outputs before adding dense evidence;
* measures the signed cross-mesh phase/amplitude history on one common scale;
* separates signed slip from absolute impulse and RMS fast activity;
* evaluates sliding-window mean rates without calling them a formal fast
  average;
* conservatively restricts N128 shell-0 profiles onto N64;
* performs significance-filtered weighted subspace diagnostics; and
* decomposes selected rate differences into descriptor, flux, and source
  terms.

No time shift is used for a binding comparison.  No reduced coordinate,
relaxation law, embedded patch, macrostep, tide, wind, hot-state, stability,
or cycle claim is authorized here.  A symmetric even/quadratic response is
also not claimed because no matching unperturbed 0.125 s center trajectory is
present in the committed WP10c8t cache set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_mode_healing_wp10c8t as wp10c8t
import run_causal_inner_mode_n128_confirmation_wp10c8t as wp10c8t_n128
import run_causal_natural_healing_wp10c8p as wp10c8p
import run_causal_nonlinear_fiber_audit_wp10c8o as wp10c8o

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    audit_causal_five_field_state_gates,
    causal_cumulative_trapezoid,
    causal_five_field_face_flux_decomposition,
    causal_five_field_scaled_primitive_vector_field,
    causal_restrict_cell_averages,
    evaluate_causal_five_field_dae,
    unpack_causal_five_field_state,
)


BASE_COMMIT = "3ccdb9532359acbaa197e066a800a9119dfe60ef"
WORK_PACKAGE = "WP10c8u"
SCHEMA_VERSION = 1
THIS_RUNNER = (
    "scripts/run_causal_inner_mode_phase_average_audit_wp10c8u.py"
)

N64_PARENT_JSON = wp10c8t.DEFAULT_OUTPUT
N64_PARENT_ARRAYS = wp10c8t.DEFAULT_ARRAYS
N128_PARENT_JSON = wp10c8t_n128.DEFAULT_OUTPUT
N128_PARENT_ARRAYS = wp10c8t_n128.DEFAULT_ARRAYS

CHECKPOINT_DIRECTORY = (
    ROOT / "outputs/checkpoints/causal_five_field_wp10c8u"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/tables/"
    "causal_inner_mode_phase_average_audit_wp10c8u.json"
)
DEFAULT_ARRAYS = (
    ROOT
    / "outputs/tables/"
    "causal_inner_mode_phase_average_audit_wp10c8u_arrays.npz"
)

MESHES = (64, 128)
RESOLUTIONS = ("coarse", "fine")
SIDES = ("minus", "plus")
WINDOW_SECONDS = (0.01, 0.025, 0.05, 0.10)
COMMON_SCALE_MESH = 64
MINIMUM_SIGNED_COSINE = 0.90
MAXIMUM_AMPLITUDE_RATIO_DEFECT = 0.50
RATE_SIGNIFICANCE_GATE = 0.10
AVERAGING_PLAUSIBILITY_RESERVE = 0.10
MAXIMUM_SPARSE_RATE_REPRODUCTION_DEFECT = 1.0e-12
MAXIMUM_RATE_RECONSTRUCTION_DEFECT = 1.0e-10
MAXIMUM_SUBSPACE_DIMENSION = 6
ATTRIBUTION_BASE_TIMES = (0.0, 0.075, 0.10, 0.125)
STRESS_COORDINATE_NAME = "shell_0_stress_storage"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    return wp10c8t._array_sha256(values)


def _plain(value):
    return wp10c8t._plain(value)


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _trajectory_label(
    n_cells: int,
    resolution: str,
    side: str,
) -> str:
    if n_cells not in MESHES:
        raise ValueError("unsupported WP10c8u mesh")
    if resolution not in RESOLUTIONS or side not in SIDES:
        raise ValueError("unsupported WP10c8u trajectory label")
    return f"N{n_cells:03d}_{resolution}_{side}"


def _dense_cache_paths(
    n_cells: int,
    resolution: str,
    side: str,
) -> tuple[Path, Path]:
    label = _trajectory_label(n_cells, resolution, side)
    return (
        CHECKPOINT_DIRECTORY / f"{label}_dense_rates.json",
        CHECKPOINT_DIRECTORY / f"{label}_dense_rates_arrays.npz",
    )


def _load_contract(
    n_cells: int,
) -> tuple[dict, dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    if n_cells == 64:
        _parent, contract, case, operator = wp10c8t._load_contract()
        return contract, case, operator["arrays"], operator["metadata"]
    if n_cells == 128:
        contract, case, _pair, operator = wp10c8t_n128._load_contract(
            force_pair=False,
        )
        return contract, case, operator["arrays"], operator["metadata"]
    raise ValueError("unsupported WP10c8u mesh")


def _load_trajectory(
    *,
    contract: dict,
    case: dict[str, np.ndarray],
    resolution: str,
    side: str,
) -> dict:
    return wp10c8t._run_or_load_trajectory(
        contract=contract,
        case=case,
        resolution=resolution,
        side=side,
        force=False,
    )


def _all_output_times(
    resolution: str,
) -> tuple[float, ...]:
    subdivisions = wp10c8t.TOTAL_SUBDIVISIONS[resolution]
    timestep = wp10c8t.TIMESTEP_SECONDS[resolution]
    return tuple(
        float(index * timestep) for index in range(subdivisions + 1)
    )


def _parent_evidence(
    n_cells: int,
) -> tuple[Path, Path, dict[str, np.ndarray]]:
    if n_cells == 64:
        return (
            N64_PARENT_JSON,
            N64_PARENT_ARRAYS,
            _load_npz(N64_PARENT_ARRAYS),
        )
    if n_cells == 128:
        return (
            N128_PARENT_JSON,
            N128_PARENT_ARRAYS,
            _load_npz(N128_PARENT_ARRAYS),
        )
    raise ValueError("unsupported WP10c8u mesh")


def _sparse_reproduction(
    *,
    n_cells: int,
    resolution: str,
    side: str,
    dense: dict[str, np.ndarray],
    parent: dict[str, np.ndarray],
) -> dict:
    prefix = f"{resolution}_{side}_"
    parent_times = np.asarray(
        parent[f"{prefix}output_times"],
        dtype=float,
    )
    dense_times = np.asarray(dense["times"], dtype=float)
    indices = []
    for value in parent_times:
        rows = np.flatnonzero(
            np.isclose(
                dense_times,
                float(value),
                rtol=0.0,
                atol=32.0 * np.finfo(float).eps,
            )
        )
        if rows.size != 1:
            raise RuntimeError("sparse output time is not unique")
        indices.append(int(rows[0]))
    selected = np.asarray(indices, dtype=int)
    comparisons = {
        "coordinates": (
            np.asarray(dense["coordinates"], dtype=float)[selected],
            np.asarray(parent[f"{prefix}coordinates"], dtype=float),
        ),
        "normalized_coordinate_rates": (
            np.asarray(
                dense["normalized_coordinate_rates"],
                dtype=float,
            )[selected],
            np.asarray(
                parent[f"{prefix}normalized_coordinate_rates"],
                dtype=float,
            ),
        ),
        "scaled_primitive_rates_per_s": (
            np.asarray(
                dense["scaled_primitive_rates_per_s"],
                dtype=float,
            )[selected],
            np.asarray(
                parent[f"{prefix}scaled_primitive_rates_per_s"],
                dtype=float,
            ),
        ),
        "primitives": (
            np.asarray(dense["primitives"], dtype=float)[selected],
            np.asarray(
                parent[f"{prefix}output_primitives"],
                dtype=float,
            ),
        ),
    }
    defects = {}
    bitwise = {}
    for name, (actual, expected) in comparisons.items():
        if actual.shape != expected.shape:
            raise RuntimeError(f"sparse {name} schema differs")
        scale = np.maximum(np.abs(expected), 1.0)
        defects[name] = float(np.max(np.abs(actual - expected) / scale))
        bitwise[name] = bool(np.array_equal(actual, expected))
    maximum = max(defects.values(), default=0.0)
    return {
        "n_cells": n_cells,
        "resolution": resolution,
        "side": side,
        "sample_count": int(parent_times.size),
        "maximum_relative_defect": maximum,
        "maximum_allowed_relative_defect": (
            MAXIMUM_SPARSE_RATE_REPRODUCTION_DEFECT
        ),
        "relative_defects": defects,
        "bitwise_equal": bitwise,
        "passed": bool(
            maximum <= MAXIMUM_SPARSE_RATE_REPRODUCTION_DEFECT
        ),
    }


def _run_or_load_dense_rate_cache(
    *,
    n_cells: int,
    resolution: str,
    side: str,
    force: bool,
) -> tuple[dict, dict[str, np.ndarray]]:
    json_path, arrays_path = _dense_cache_paths(
        n_cells,
        resolution,
        side,
    )
    parent_json_path, parent_arrays_path, parent_arrays = _parent_evidence(
        n_cells
    )
    contract, case, operator_arrays, operator_metadata = _load_contract(
        n_cells
    )
    trajectory = _load_trajectory(
        contract=contract,
        case=case,
        resolution=resolution,
        side=side,
    )
    expected = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "base_commit": BASE_COMMIT,
        "n_cells": n_cells,
        "resolution": resolution,
        "side": side,
        "trajectory_sha256": trajectory["sha256"],
        "trajectory_state_sha256": _array_sha256(trajectory["states"]),
        "parent_json_sha256": _sha256(parent_json_path),
        "parent_arrays_sha256": _sha256(parent_arrays_path),
        "operator_metadata": _plain(operator_metadata),
        "rate_backend": "wp10c8t_branch_frozen_local_fresh_rate",
    }
    if json_path.exists() and arrays_path.exists() and not force:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if not all(payload.get(key) == value for key, value in expected.items()):
            raise RuntimeError(f"stale WP10c8u dense cache: {json_path}")
        if payload.get("arrays_sha256") != _sha256(arrays_path):
            raise RuntimeError("WP10c8u dense arrays checksum differs")
        arrays = _load_npz(arrays_path)
        return payload, arrays

    started = time.perf_counter()
    output_times = _all_output_times(resolution)
    rate_cache: dict[
        str,
        tuple[np.ndarray, dict, dict[str, np.ndarray]],
    ] = {}
    summary, dense = wp10c8t._trajectory_diagnostics_with_rates(
        contract=contract,
        case=case,
        operator_arrays=operator_arrays,
        states=np.asarray(trajectory["states"], dtype=float),
        subdivisions=wp10c8t.TOTAL_SUBDIVISIONS[resolution],
        rate_cache=rate_cache,
        compute_fresh_rates=True,
        duration_seconds=wp10c8t.TARGET_DURATION_SECONDS,
        output_offsets_seconds=output_times,
    )
    arrays = {
        "times": np.asarray(dense["output_times"], dtype=float),
        "coordinate_names": np.asarray(
            dense["coordinate_names"],
            dtype="U",
        ),
        "coordinate_scales": np.asarray(
            dense["coordinate_scales"],
            dtype=float,
        ),
        "coordinates": np.asarray(dense["coordinates"], dtype=float),
        "normalized_coordinate_rates": np.asarray(
            dense["normalized_coordinate_rates"],
            dtype=float,
        ),
        "scaled_primitive_rates_per_s": np.asarray(
            dense["scaled_primitive_rates_per_s"],
            dtype=float,
        ),
        "primitives": np.asarray(
            dense["output_primitives"],
            dtype=float,
        ),
        "conserved": np.asarray(
            dense["output_conserved"],
            dtype=float,
        ),
        "primitive_column_scales": np.asarray(
            operator_arrays["primitive_column_scales"],
            dtype=float,
        ),
        "physical_input_amplitudes": np.asarray(
            operator_arrays["physical_input_amplitudes"],
            dtype=float,
        ),
        "conservation_row_scales": np.asarray(
            operator_arrays["conservation_row_scales"],
            dtype=float,
        ),
        "shell_edge_indices": np.asarray(
            operator_arrays["shell_edge_indices"],
            dtype=int,
        ),
        "radius_rg": np.asarray(
            contract["context"].grid.centers
            / contract["context"].grid.gravitational_radius,
            dtype=float,
        ),
        "cell_measures": np.asarray(
            contract["context"].grid.cell_measures,
            dtype=float,
        ),
        "trajectory_states": np.asarray(
            trajectory["states"],
            dtype=float,
        ),
    }
    reproduction = _sparse_reproduction(
        n_cells=n_cells,
        resolution=resolution,
        side=side,
        dense=arrays,
        parent=parent_arrays,
    )
    if not (
        summary["all_fresh_rate_audits_passed"]
        and summary["all_output_state_gates_passed"]
        and reproduction["passed"]
    ):
        raise RuntimeError("WP10c8u dense rate contract failed")
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **arrays)
    payload = {
        **expected,
        "producer_runner_sha256": _sha256(ROOT / THIS_RUNNER),
        "arrays_path": _relative(arrays_path),
        "arrays_sha256": _sha256(arrays_path),
        "state_count": int(arrays["times"].size),
        "fresh_rate_audits_passed": True,
        "state_gates_passed": True,
        "sparse_reproduction": reproduction,
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
    return payload, arrays


def _load_existing_dense_rate_cache(
    *,
    n_cells: int,
    resolution: str,
    side: str,
) -> tuple[dict, dict[str, np.ndarray]]:
    json_path, arrays_path = _dense_cache_paths(
        n_cells,
        resolution,
        side,
    )
    parent_json_path = (
        N64_PARENT_JSON if n_cells == 64 else N128_PARENT_JSON
    )
    parent_arrays_path = (
        N64_PARENT_ARRAYS if n_cells == 64 else N128_PARENT_ARRAYS
    )
    if not (json_path.exists() and arrays_path.exists()):
        raise RuntimeError(
            f"missing WP10c8u dense cache: {json_path}"
        )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "base_commit": BASE_COMMIT,
        "n_cells": n_cells,
        "resolution": resolution,
        "side": side,
        "parent_json_sha256": _sha256(parent_json_path),
        "parent_arrays_sha256": _sha256(parent_arrays_path),
        "rate_backend": "wp10c8t_branch_frozen_local_fresh_rate",
    }
    if not all(payload.get(key) == value for key, value in expected.items()):
        raise RuntimeError(f"stale WP10c8u dense cache: {json_path}")
    if payload.get("arrays_sha256") != _sha256(arrays_path):
        raise RuntimeError("WP10c8u dense arrays checksum differs")
    if not payload.get("sparse_reproduction", {}).get("passed", False):
        raise RuntimeError("WP10c8u sparse reproduction did not pass")
    return payload, _load_npz(arrays_path)


def _physical_coordinate_rates(
    arrays: dict[str, np.ndarray],
) -> np.ndarray:
    return (
        np.asarray(
            arrays["normalized_coordinate_rates"],
            dtype=float,
        )
        * np.asarray(arrays["coordinate_scales"], dtype=float)[None, :]
        / wp10c8o.COORDINATE_RATE_WINDOW_SECONDS
    )


def _pair_history(
    *,
    minus: dict[str, np.ndarray],
    plus: dict[str, np.ndarray],
    common_coordinate_scales: np.ndarray,
    loading_time_seconds: float,
) -> dict[str, np.ndarray]:
    times = np.asarray(minus["times"], dtype=float)
    if not (
        np.array_equal(times, np.asarray(plus["times"], dtype=float))
        and np.array_equal(
            minus["coordinate_names"],
            plus["coordinate_names"],
        )
    ):
        raise RuntimeError("WP10c8u pair schema differs")
    rates_minus = _physical_coordinate_rates(minus)
    rates_plus = _physical_coordinate_rates(plus)
    slow_rate = (
        0.5
        * (rates_plus - rates_minus)
        * float(loading_time_seconds)
        / common_coordinate_scales[None, :]
    )
    slip = (
        0.5
        * (
            np.asarray(plus["coordinates"], dtype=float)
            - np.asarray(minus["coordinates"], dtype=float)
        )
        / common_coordinate_scales[None, :]
    )
    slow_times = times / float(loading_time_seconds)
    integrated = causal_cumulative_trapezoid(slow_times, slow_rate)
    activity = causal_cumulative_trapezoid(
        slow_times,
        np.linalg.norm(slow_rate, axis=1),
    )
    primitive_shape = np.asarray(
        minus["primitives"],
        dtype=float,
    ).shape
    if primitive_shape != np.asarray(
        plus["primitives"],
        dtype=float,
    ).shape:
        raise RuntimeError("WP10c8u primitive histories differ")
    primitive_minus = (
        np.asarray(minus["scaled_primitive_rates_per_s"], dtype=float)
        * np.asarray(
            minus["primitive_column_scales"],
            dtype=float,
        )[None, :]
    ).reshape(primitive_shape)
    primitive_plus = (
        np.asarray(plus["scaled_primitive_rates_per_s"], dtype=float)
        * np.asarray(
            plus["primitive_column_scales"],
            dtype=float,
        )[None, :]
    ).reshape(primitive_shape)
    return {
        "times": times,
        "signed_slow_rate_half_difference": slow_rate,
        "signed_coordinate_slip": slip,
        "rate_integrated_slip": integrated,
        "absolute_impulse": activity,
        "physical_primitive_rate_half_difference_per_s": (
            0.5 * (primitive_plus - primitive_minus)
        ),
        "primitive_state_half_difference": 0.5
        * (
            np.asarray(plus["primitives"], dtype=float)
            - np.asarray(minus["primitives"], dtype=float)
        ),
    }


def _direction_metrics(
    left: np.ndarray,
    right: np.ndarray,
) -> dict[str, np.ndarray]:
    first = np.asarray(left, dtype=float)
    second = np.asarray(right, dtype=float)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("direction histories differ")
    first_norm = np.linalg.norm(first, axis=1)
    second_norm = np.linalg.norm(second, axis=1)
    denominator = np.maximum(
        first_norm * second_norm,
        np.finfo(float).tiny,
    )
    signed_cosine = np.sum(first * second, axis=1) / denominator
    maximum_first = np.max(np.abs(first), axis=1)
    maximum_second = np.max(np.abs(second), axis=1)
    maximum_ratio = maximum_second / np.maximum(
        maximum_first,
        np.finfo(float).tiny,
    )
    norm_ratio = second_norm / np.maximum(
        first_norm,
        np.finfo(float).tiny,
    )
    return {
        "signed_cosine": signed_cosine,
        "absolute_cosine": np.abs(signed_cosine),
        "left_l2_norm": first_norm,
        "right_l2_norm": second_norm,
        "l2_norm_ratio": norm_ratio,
        "left_maximum": maximum_first,
        "right_maximum": maximum_second,
        "maximum_ratio": maximum_ratio,
        "maximum_ratio_defect": np.abs(maximum_ratio - 1.0),
        "same_time_gate_passed": (
            (signed_cosine >= MINIMUM_SIGNED_COSINE)
            & (
                np.abs(maximum_ratio - 1.0)
                <= MAXIMUM_AMPLITUDE_RATIO_DEFECT
            )
        ),
    }


def _first_index(mask: np.ndarray) -> int | None:
    indices = np.flatnonzero(np.asarray(mask, dtype=bool))
    return None if indices.size == 0 else int(indices[0])


def _component_zero_crossings(
    times: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    time_values = np.asarray(times, dtype=float)
    rows = np.asarray(values, dtype=float)
    if time_values.ndim != 1 or rows.shape != time_values.shape:
        raise ValueError("zero-crossing history has the wrong shape")
    indices = np.flatnonzero(
        (rows[:-1] == 0.0)
        | (rows[1:] == 0.0)
        | (np.signbit(rows[:-1]) != np.signbit(rows[1:]))
    )
    crossings = []
    for index in indices:
        left_time = time_values[index]
        right_time = time_values[index + 1]
        left = rows[index]
        right = rows[index + 1]
        denominator = right - left
        fraction = (
            0.0
            if denominator == 0.0
            else float(np.clip(-left / denominator, 0.0, 1.0))
        )
        crossings.append(left_time + fraction * (right_time - left_time))
    return np.asarray(crossings, dtype=float)


def _window_statistics(
    times: np.ndarray,
    rate: np.ndarray,
    duration_seconds: float,
) -> dict[str, np.ndarray]:
    time_values = np.asarray(times, dtype=float)
    rows = np.asarray(rate, dtype=float)
    if (
        time_values.ndim != 1
        or rows.ndim != 2
        or rows.shape[0] != time_values.size
        or time_values.size < 2
    ):
        raise ValueError("window history has the wrong shape")
    timestep = float(np.median(np.diff(time_values)))
    steps = int(round(float(duration_seconds) / timestep))
    if (
        steps < 1
        or not np.isclose(
            steps * timestep,
            duration_seconds,
            rtol=0.0,
            atol=64.0 * np.finfo(float).eps,
        )
    ):
        raise ValueError("window is not commensurate with the time grid")
    starts = np.arange(0, time_values.size - steps, dtype=int)
    means = []
    rms = []
    signed_changes = []
    for start in starts:
        stop = start + steps
        window_times = time_values[start : stop + 1]
        window_rows = rows[start : stop + 1]
        integral = np.trapezoid(window_rows, window_times, axis=0)
        means.append(integral / float(duration_seconds))
        rms.append(
            float(
                np.sqrt(
                    np.trapezoid(
                        np.sum(window_rows**2, axis=1),
                        window_times,
                    )
                    / float(duration_seconds)
                )
            )
        )
        signed_changes.append(integral)
    return {
        "start_indices": starts,
        "start_times": time_values[starts],
        "mean_vectors": np.asarray(means, dtype=float),
        "mean_norms": np.linalg.norm(
            np.asarray(means, dtype=float),
            axis=1,
        ),
        "rms_amplitudes": np.asarray(rms, dtype=float),
        "physical_time_integrals": np.asarray(
            signed_changes,
            dtype=float,
        ),
    }


def _restrict_profile(
    *,
    coarse_context,
    fine_context,
    fine_values: np.ndarray,
) -> np.ndarray:
    values = np.asarray(fine_values, dtype=float)
    restricted_rows = [
        causal_restrict_cell_averages(
            coarse_context.grid,
            fine_context.grid,
            row,
        )
        for row in values
    ]
    return np.asarray(restricted_rows, dtype=float)


def _profile_support(
    *,
    values: np.ndarray,
    radius_rg: np.ndarray,
    cell_measures: np.ndarray,
    shell_stop: int,
) -> dict[str, np.ndarray]:
    rows = np.asarray(values, dtype=float)
    radius = np.asarray(radius_rg, dtype=float)
    measures = np.asarray(cell_measures, dtype=float)
    if rows.ndim != 3 or rows.shape[1:] != (radius.size, 5):
        raise ValueError("profile history has the wrong shape")
    weights = np.sum(np.abs(rows), axis=2) * measures[None, :]
    total = np.sum(weights, axis=1)
    shell = np.sum(weights[:, :shell_stop], axis=1)
    centroid = np.sum(weights * radius[None, :], axis=1) / np.maximum(
        total,
        np.finfo(float).tiny,
    )
    variance = np.sum(
        weights * (radius[None, :] - centroid[:, None]) ** 2,
        axis=1,
    ) / np.maximum(total, np.finfo(float).tiny)
    controlling_cell = np.argmax(weights, axis=1)
    return {
        "shell_0_l1_fraction": shell
        / np.maximum(total, np.finfo(float).tiny),
        "centroid_rg": centroid,
        "width_rg": np.sqrt(np.maximum(variance, 0.0)),
        "controlling_cell": controlling_cell,
        "controlling_radius_rg": radius[controlling_cell],
    }


def _weighted_profile_direction_metrics(
    *,
    left: np.ndarray,
    right: np.ndarray,
    cell_measures: np.ndarray,
    shell_stop: int,
) -> dict[str, np.ndarray]:
    first = np.asarray(left, dtype=float)[:, :shell_stop]
    second = np.asarray(right, dtype=float)[:, :shell_stop]
    weights = np.sqrt(
        np.asarray(cell_measures[:shell_stop], dtype=float)
        / np.sum(cell_measures[:shell_stop])
    )
    first_weighted = (first * weights[None, :, None]).reshape(
        first.shape[0],
        -1,
    )
    second_weighted = (second * weights[None, :, None]).reshape(
        second.shape[0],
        -1,
    )
    return _direction_metrics(first_weighted, second_weighted)


def _weighted_pod(
    snapshots: np.ndarray,
    *,
    cell_measures: np.ndarray,
    shell_stop: int,
    significant_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    rows = np.asarray(snapshots, dtype=float)
    selected = np.asarray(significant_mask, dtype=bool)
    if rows.ndim != 3 or selected.shape != (rows.shape[0],):
        raise ValueError("POD inputs have the wrong shape")
    weights = np.sqrt(
        np.asarray(cell_measures[:shell_stop], dtype=float)
        / np.sum(cell_measures[:shell_stop])
    )
    matrix = (
        rows[selected, :shell_stop]
        * weights[None, :, None]
    ).reshape(np.count_nonzero(selected), -1)
    if matrix.shape[0] == 0:
        return {
            "singular_values": np.zeros(0),
            "right_vectors": np.zeros((0, matrix.shape[1])),
            "selected_indices": np.flatnonzero(selected),
        }
    _left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    return {
        "singular_values": singular,
        "right_vectors": right,
        "selected_indices": np.flatnonzero(selected),
    }


def _principal_cosines(
    left_vectors: np.ndarray,
    right_vectors: np.ndarray,
    maximum_dimension: int = MAXIMUM_SUBSPACE_DIMENSION,
) -> np.ndarray:
    first = np.asarray(left_vectors, dtype=float)
    second = np.asarray(right_vectors, dtype=float)
    dimension = min(
        int(maximum_dimension),
        first.shape[0],
        second.shape[0],
    )
    if dimension == 0:
        return np.zeros(0, dtype=float)
    return np.linalg.svd(
        first[:dimension] @ second[:dimension].T,
        compute_uv=False,
    )


def _term_rate_decomposition(
    *,
    context,
    vector: np.ndarray,
    operator_arrays: dict[str, np.ndarray],
) -> tuple[dict, dict[str, np.ndarray]]:
    state = unpack_causal_five_field_state(
        np.asarray(vector, dtype=float),
        int(context.grid.centers.size),
    )
    primitive_scales = np.asarray(
        operator_arrays["primitive_column_scales"],
        dtype=float,
    )
    conservation_scales = np.asarray(
        operator_arrays["conservation_row_scales"],
        dtype=float,
    )
    primary = causal_five_field_scaled_primitive_vector_field(
        context,
        np.asarray(state.primitives, dtype=float).ravel(),
        primitive_column_scales=primitive_scales,
        conservation_row_scales=conservation_scales,
        mapped_storage_backend="branch_frozen_local",
        branch_frozen_local_difference_step=(
            wp10c8o.BRANCH_FROZEN_LOCAL_STEP
        ),
    )
    evaluation = evaluate_causal_five_field_dae(vector, context)
    split = causal_five_field_face_flux_decomposition(context, vector)
    n_cells = int(context.grid.centers.size)
    face_terms = {
        "perfect_fluid_transport": np.zeros((n_cells + 1, 5)),
        "stress_transport": np.zeros((n_cells + 1, 5)),
        "rusanov_transport": np.zeros((n_cells + 1, 5)),
        "boundary_characteristic_transport": np.zeros((n_cells + 1, 5)),
    }
    face_terms["perfect_fluid_transport"][1:-1] = (
        split.central_perfect_weighted_face_fluxes_over_c
    )
    face_terms["stress_transport"][1:-1] = (
        split.central_stress_weighted_face_fluxes_over_c
    )
    face_terms["rusanov_transport"][1:-1] = (
        split.rusanov_weighted_face_fluxes_over_c
    )
    face_terms["boundary_characteristic_transport"][0] = (
        state.weighted_face_fluxes_over_c[0]
    )
    face_terms["boundary_characteristic_transport"][-1] = (
        state.weighted_face_fluxes_over_c[-1]
    )
    mass = np.asarray(
        primary["descriptor_reduced_scaled_matrix"],
        dtype=float,
    )
    residual_terms = {}
    rate_terms = {}
    for name, faces in face_terms.items():
        residual = faces[1:] - faces[:-1]
        scaled_residual = residual.ravel() / conservation_scales
        residual_terms[name] = scaled_residual.reshape(n_cells, 5)
        rate_terms[name] = (
            -np.linalg.solve(mass, scaled_residual)
        ).reshape(n_cells, 5)
    for name, source in sorted(
        evaluation.integrated_source_components_per_ct.items()
    ):
        scaled_residual = (
            -np.asarray(source, dtype=float).ravel()
            / conservation_scales
        )
        label = f"source_{name}"
        residual_terms[label] = scaled_residual.reshape(n_cells, 5)
        rate_terms[label] = (
            -np.linalg.solve(mass, scaled_residual)
        ).reshape(n_cells, 5)
    total_rate = np.asarray(
        primary["scaled_primitive_rate_per_s"],
        dtype=float,
    )
    reconstructed = np.sum(
        np.asarray(list(rate_terms.values()), dtype=float),
        axis=0,
    )
    rate_scale = max(
        float(np.max(np.abs(total_rate))),
        np.finfo(float).tiny,
    )
    rate_defect = float(
        np.max(np.abs(reconstructed - total_rate)) / rate_scale
    )
    mapped_action = (
        np.asarray(
            primary["conserved_descriptor_reduced_scaled_matrix"],
            dtype=float,
        )
        @ total_rate.ravel()
    ).reshape(n_cells, 5)
    height_action = (
        np.asarray(
            primary["vertical_descriptor_reduced_scaled_matrix"],
            dtype=float,
        )
        @ total_rate.ravel()
    ).reshape(n_cells, 5)
    total_action = (
        np.asarray(
            primary["descriptor_reduced_scaled_matrix"],
            dtype=float,
        )
        @ total_rate.ravel()
    ).reshape(n_cells, 5)
    storage_scale = max(
        float(np.max(np.abs(total_action))),
        np.finfo(float).tiny,
    )
    storage_defect = float(
        np.max(np.abs(total_action - mapped_action - height_action))
        / storage_scale
    )
    passed = bool(
        rate_defect <= MAXIMUM_RATE_RECONSTRUCTION_DEFECT
        and storage_defect <= MAXIMUM_RATE_RECONSTRUCTION_DEFECT
        and split.maximum_production_reconstruction_defect
        <= MAXIMUM_RATE_RECONSTRUCTION_DEFECT
        and audit_causal_five_field_state_gates(context, vector)["passed"]
    )
    arrays = {
        "total_scaled_primitive_rate_per_s": total_rate,
        "mapped_storage_action": mapped_action,
        "responsive_height_storage_action": height_action,
        "total_storage_action": total_action,
        **{
            f"scaled_rate_term_{name}": values
            for name, values in rate_terms.items()
        },
        **{
            f"scaled_residual_term_{name}": values
            for name, values in residual_terms.items()
        },
    }
    return {
        "rate_term_names": tuple(rate_terms),
        "rate_reconstruction_relative_defect": rate_defect,
        "storage_reconstruction_relative_defect": storage_defect,
        "flux_reconstruction_relative_defect": (
            split.maximum_production_reconstruction_defect
        ),
        "passed": passed,
    }, arrays


def _analysis(
    *,
    dense_by_label: dict[str, dict[str, np.ndarray]],
    contracts: dict[int, dict],
    operators: dict[int, dict[str, np.ndarray]],
    loading_times: dict[int, float],
) -> tuple[dict, dict[str, np.ndarray]]:
    common_scales = np.asarray(
        dense_by_label["N064_fine_minus"]["coordinate_scales"],
        dtype=float,
    )
    common_names = np.asarray(
        dense_by_label["N064_fine_minus"]["coordinate_names"],
        dtype="U",
    )
    stress_rows = np.flatnonzero(common_names == STRESS_COORDINATE_NAME)
    if stress_rows.size != 1:
        raise RuntimeError("shell-0 stress coordinate is not unique")
    stress_index = int(stress_rows[0])

    pairs = {}
    output_arrays: dict[str, np.ndarray] = {
        "common_coordinate_names": common_names,
        "common_coordinate_scales": common_scales,
    }
    for n_cells in MESHES:
        for resolution in RESOLUTIONS:
            prefix = f"N{n_cells:03d}_{resolution}"
            pair = _pair_history(
                minus=dense_by_label[f"{prefix}_minus"],
                plus=dense_by_label[f"{prefix}_plus"],
                common_coordinate_scales=common_scales,
                loading_time_seconds=loading_times[n_cells],
            )
            pairs[(n_cells, resolution)] = pair
            output_arrays.update(
                {
                    f"{prefix}_{name}": values
                    for name, values in pair.items()
                }
            )

    n64 = pairs[(64, "fine")]
    n128 = pairs[(128, "fine")]
    orientation = (
        1.0
        if np.dot(
            n64["signed_slow_rate_half_difference"][0],
            n128["signed_slow_rate_half_difference"][0],
        )
        >= 0.0
        else -1.0
    )
    n128_oriented_rate = (
        orientation * n128["signed_slow_rate_half_difference"]
    )
    direction = _direction_metrics(
        n64["signed_slow_rate_half_difference"],
        n128_oriented_rate,
    )
    output_arrays.update(
        {
            f"cross_mesh_rate_{name}": values
            for name, values in direction.items()
        }
    )
    first_signed_failure = _first_index(
        direction["signed_cosine"] < MINIMUM_SIGNED_COSINE
    )
    first_amplitude_failure = _first_index(
        direction["maximum_ratio_defect"]
        > MAXIMUM_AMPLITUDE_RATIO_DEFECT
    )
    first_same_time_failure = _first_index(
        ~direction["same_time_gate_passed"]
    )
    n64_stress = n64["signed_slow_rate_half_difference"][:, stress_index]
    n128_stress = n128_oriented_rate[:, stress_index]
    stress_sign_disagreement = _first_index(
        (np.signbit(n64_stress) != np.signbit(n128_stress))
        & (np.abs(n64_stress) >= RATE_SIGNIFICANCE_GATE)
        & (np.abs(n128_stress) >= RATE_SIGNIFICANCE_GATE)
    )
    output_arrays["n64_stress_zero_crossings_seconds"] = (
        _component_zero_crossings(n64["times"], n64_stress)
    )
    output_arrays["n128_stress_zero_crossings_seconds"] = (
        _component_zero_crossings(n128["times"], n128_stress)
    )

    window_summary = {}
    for duration in WINDOW_SECONDS:
        label = str(duration).replace(".", "p")
        n64_window = _window_statistics(
            n64["times"],
            n64["signed_slow_rate_half_difference"],
            duration,
        )
        n128_window = _window_statistics(
            n128["times"],
            n128_oriented_rate,
            duration,
        )
        if not np.array_equal(
            n64_window["start_times"],
            n128_window["start_times"],
        ):
            raise RuntimeError("cross-mesh window starts differ")
        window_direction = _direction_metrics(
            n64_window["mean_vectors"],
            n128_window["mean_vectors"],
        )
        for prefix, values in (
            ("n64", n64_window),
            ("n128", n128_window),
            ("cross_mesh", window_direction),
        ):
            output_arrays.update(
                {
                    f"window_{label}_{prefix}_{name}": array
                    for name, array in values.items()
                }
            )
        window_summary[str(duration)] = {
            "sample_count": int(n64_window["start_times"].size),
            "n64_maximum_mean_norm": float(
                np.max(n64_window["mean_norms"])
            ),
            "n128_maximum_mean_norm": float(
                np.max(n128_window["mean_norms"])
            ),
            "minimum_signed_cross_mesh_cosine": float(
                np.min(window_direction["signed_cosine"])
            ),
            "maximum_cross_mesh_amplitude_ratio_defect": float(
                np.max(window_direction["maximum_ratio_defect"])
            ),
            "all_same_time_gates_passed": bool(
                np.all(window_direction["same_time_gate_passed"])
            ),
        }

    n64_dense = dense_by_label["N064_fine_minus"]
    n128_dense = dense_by_label["N128_fine_minus"]
    n64_primitive_scales = np.asarray(
        n64_dense["primitive_column_scales"],
        dtype=float,
    ).reshape(64, 5)
    n64_input_amplitudes = np.asarray(
        n64_dense["physical_input_amplitudes"],
        dtype=float,
    ).reshape(64, 5)
    n64_state_profile = (
        n64["primitive_state_half_difference"] / n64_input_amplitudes[None]
    )
    n64_rate_profile = (
        n64["physical_primitive_rate_half_difference_per_s"]
        / n64_primitive_scales[None]
    )
    restricted_n128_state_physical = _restrict_profile(
        coarse_context=contracts[64]["context"],
        fine_context=contracts[128]["context"],
        fine_values=n128["primitive_state_half_difference"],
    )
    restricted_n128_rate_physical = _restrict_profile(
        coarse_context=contracts[64]["context"],
        fine_context=contracts[128]["context"],
        fine_values=n128[
            "physical_primitive_rate_half_difference_per_s"
        ],
    )
    n128_state_profile = (
        orientation
        * restricted_n128_state_physical
        / n64_input_amplitudes[None]
    )
    n128_rate_profile = (
        orientation
        * restricted_n128_rate_physical
        / n64_primitive_scales[None]
    )
    output_arrays.update(
        {
            "n64_normalized_state_half_difference": n64_state_profile,
            "n128_restricted_normalized_state_half_difference": (
                n128_state_profile
            ),
            "n64_normalized_primitive_rate_half_difference": (
                n64_rate_profile
            ),
            "n128_restricted_normalized_primitive_rate_half_difference": (
                n128_rate_profile
            ),
        }
    )
    shell_stop = int(
        np.asarray(n64_dense["shell_edge_indices"], dtype=int)[1]
    )
    radius_rg = np.asarray(n64_dense["radius_rg"], dtype=float)
    cell_measures = np.asarray(n64_dense["cell_measures"], dtype=float)
    profile_support = {}
    for label, values in (
        ("n64_state", n64_state_profile),
        ("n128_state", n128_state_profile),
        ("n64_rate", n64_rate_profile),
        ("n128_rate", n128_rate_profile),
    ):
        support = _profile_support(
            values=values,
            radius_rg=radius_rg,
            cell_measures=cell_measures,
            shell_stop=shell_stop,
        )
        profile_support[label] = {
            key: _plain(value) for key, value in support.items()
        }
        output_arrays.update(
            {
                f"{label}_support_{name}": values
                for name, values in support.items()
            }
        )
    state_profile_direction = _weighted_profile_direction_metrics(
        left=n64_state_profile,
        right=n128_state_profile,
        cell_measures=cell_measures,
        shell_stop=shell_stop,
    )
    rate_profile_direction = _weighted_profile_direction_metrics(
        left=n64_rate_profile,
        right=n128_rate_profile,
        cell_measures=cell_measures,
        shell_stop=shell_stop,
    )
    output_arrays.update(
        {
            f"cross_mesh_state_profile_{name}": values
            for name, values in state_profile_direction.items()
        }
    )
    output_arrays.update(
        {
            f"cross_mesh_rate_profile_{name}": values
            for name, values in rate_profile_direction.items()
        }
    )

    significant = (
        np.maximum(
            np.max(
                np.abs(n64["signed_slow_rate_half_difference"]),
                axis=1,
            ),
            np.max(np.abs(n128_oriented_rate), axis=1),
        )
        >= RATE_SIGNIFICANCE_GATE
    )
    pods = {}
    for kind, left_values, right_values in (
        ("state", n64_state_profile, n128_state_profile),
        ("rate", n64_rate_profile, n128_rate_profile),
    ):
        left_pod = _weighted_pod(
            left_values,
            cell_measures=cell_measures,
            shell_stop=shell_stop,
            significant_mask=significant,
        )
        right_pod = _weighted_pod(
            right_values,
            cell_measures=cell_measures,
            shell_stop=shell_stop,
            significant_mask=significant,
        )
        principal = _principal_cosines(
            left_pod["right_vectors"],
            right_pod["right_vectors"],
        )
        output_arrays[f"pod_{kind}_n64_singular_values"] = left_pod[
            "singular_values"
        ]
        output_arrays[f"pod_{kind}_n128_singular_values"] = right_pod[
            "singular_values"
        ]
        output_arrays[f"pod_{kind}_principal_cosines"] = principal
        pods[kind] = {
            "significant_snapshot_count": int(np.count_nonzero(significant)),
            "n64_singular_values": _plain(left_pod["singular_values"]),
            "n128_singular_values": _plain(right_pod["singular_values"]),
            "principal_cosines": _plain(principal),
            "diagnostic_only": True,
        }

    event_indices = {0, n64["times"].size - 1}
    for value in ATTRIBUTION_BASE_TIMES:
        rows = np.flatnonzero(
            np.isclose(
                n64["times"],
                value,
                rtol=0.0,
                atol=32.0 * np.finfo(float).eps,
            )
        )
        if rows.size == 1:
            event_indices.add(int(rows[0]))
    for index in (
        first_signed_failure,
        first_amplitude_failure,
        first_same_time_failure,
        stress_sign_disagreement,
    ):
        if index is not None:
            event_indices.add(index)
            if index > 0:
                event_indices.add(index - 1)
    event_indices_array = np.asarray(sorted(event_indices), dtype=int)
    output_arrays["attribution_event_indices"] = event_indices_array
    output_arrays["attribution_event_times_seconds"] = n64["times"][
        event_indices_array
    ]

    attribution_summary = {}
    for n_cells in MESHES:
        prefix = f"N{n_cells:03d}_fine"
        minus_dense = dense_by_label[f"{prefix}_minus"]
        plus_dense = dense_by_label[f"{prefix}_plus"]
        term_rows = {}
        audit_rows = []
        for index in event_indices_array:
            side_arrays = {}
            side_audits = {}
            for side, dense in (
                ("minus", minus_dense),
                ("plus", plus_dense),
            ):
                audit, arrays = _term_rate_decomposition(
                    context=contracts[n_cells]["context"],
                    vector=np.asarray(
                        dense["trajectory_states"][index],
                        dtype=float,
                    ),
                    operator_arrays=operators[n_cells],
                )
                if not audit["passed"]:
                    raise RuntimeError(
                        "WP10c8u physical term decomposition failed"
                    )
                side_arrays[side] = arrays
                side_audits[side] = audit
            names = side_audits["minus"]["rate_term_names"]
            if names != side_audits["plus"]["rate_term_names"]:
                raise RuntimeError("plus/minus rate-term schemas differ")
            for name in names:
                key = f"scaled_rate_term_{name}"
                half = 0.5 * (
                    side_arrays["plus"][key]
                    - side_arrays["minus"][key]
                )
                term_rows.setdefault(name, []).append(half)
            for name in (
                "mapped_storage_action",
                "responsive_height_storage_action",
                "total_storage_action",
            ):
                half = 0.5 * (
                    side_arrays["plus"][name]
                    - side_arrays["minus"][name]
                )
                term_rows.setdefault(name, []).append(half)
            audit_rows.append(side_audits)
        for name, values in term_rows.items():
            output_arrays[
                f"attribution_N{n_cells:03d}_{name}"
            ] = np.asarray(values, dtype=float)
        attribution_summary[str(n_cells)] = {
            "event_count": int(event_indices_array.size),
            "all_decompositions_passed": True,
            "maximum_rate_reconstruction_relative_defect": float(
                max(
                    row[side]["rate_reconstruction_relative_defect"]
                    for row in audit_rows
                    for side in SIDES
                )
            ),
            "maximum_storage_reconstruction_relative_defect": float(
                max(
                    row[side][
                        "storage_reconstruction_relative_defect"
                    ]
                    for row in audit_rows
                    for side in SIDES
                )
            ),
        }

    temporal = {}
    for n_cells in MESHES:
        coarse = pairs[(n_cells, "coarse")]
        fine = pairs[(n_cells, "fine")]
        if not np.array_equal(coarse["times"], fine["times"][::2]):
            raise RuntimeError("nested coarse/fine dense times differ")
        uncertainty = np.abs(
            coarse["signed_slow_rate_half_difference"]
            - fine["signed_slow_rate_half_difference"][::2]
        )
        output_arrays[
            f"N{n_cells:03d}_coarse_fine_rate_uncertainty"
        ] = uncertainty
        temporal[str(n_cells)] = {
            "maximum_absolute_rate_uncertainty": float(
                np.max(uncertainty)
            ),
            "final_maximum_absolute_rate_uncertainty": float(
                np.max(uncertainty[-1])
            ),
        }

    longest = window_summary[str(max(WINDOW_SECONDS))]
    small_signed_slip = bool(
        max(
            float(np.max(np.abs(n64["rate_integrated_slip"][-1]))),
            float(np.max(np.abs(n128["rate_integrated_slip"][-1]))),
        )
        <= AVERAGING_PLAUSIBILITY_RESERVE
    )
    mean_mesh_controlled = bool(
        longest["all_same_time_gates_passed"]
    )
    longest_window_mean_small = bool(
        max(
            longest["n64_maximum_mean_norm"],
            longest["n128_maximum_mean_norm"],
        )
        <= AVERAGING_PLAUSIBILITY_RESERVE
    )
    if (
        small_signed_slip
        and mean_mesh_controlled
        and longest_window_mean_small
    ):
        classification = (
            "cache_evidence_supports_averaging_plausibility_only"
        )
    elif not mean_mesh_controlled:
        classification = "localized_inner_phase_spatially_unresolved"
    else:
        classification = "cache_evidence_rejects_averaging_plausibility"
    summary = {
        "classification": classification,
        "formal_fast_average_certified": False,
        "architecture_change_authorized": False,
        "quadratic_even_response_measured": False,
        "quadratic_even_response_blocker": (
            "no_matching_unperturbed_center_trajectory_through_0p125s"
        ),
        "common_scale_mesh": COMMON_SCALE_MESH,
        "orientation_factor_applied_to_n128": orientation,
        "stress_coordinate_index": stress_index,
        "first_signed_cosine_failure_index": first_signed_failure,
        "first_signed_cosine_failure_time_seconds": (
            None
            if first_signed_failure is None
            else float(n64["times"][first_signed_failure])
        ),
        "first_amplitude_failure_index": first_amplitude_failure,
        "first_amplitude_failure_time_seconds": (
            None
            if first_amplitude_failure is None
            else float(n64["times"][first_amplitude_failure])
        ),
        "first_same_time_gate_failure_index": first_same_time_failure,
        "first_same_time_gate_failure_time_seconds": (
            None
            if first_same_time_failure is None
            else float(n64["times"][first_same_time_failure])
        ),
        "first_significant_stress_sign_disagreement_index": (
            stress_sign_disagreement
        ),
        "first_significant_stress_sign_disagreement_time_seconds": (
            None
            if stress_sign_disagreement is None
            else float(n64["times"][stress_sign_disagreement])
        ),
        "minimum_signed_rate_cosine": float(
            np.min(direction["signed_cosine"])
        ),
        "maximum_rate_amplitude_ratio_defect": float(
            np.max(direction["maximum_ratio_defect"])
        ),
        "n64_final_signed_slip_maximum": float(
            np.max(np.abs(n64["rate_integrated_slip"][-1]))
        ),
        "n128_final_signed_slip_maximum": float(
            np.max(np.abs(n128["rate_integrated_slip"][-1]))
        ),
        "n64_final_absolute_impulse": float(
            n64["absolute_impulse"][-1]
        ),
        "n128_final_absolute_impulse": float(
            n128["absolute_impulse"][-1]
        ),
        "small_signed_slip_passed": small_signed_slip,
        "longest_window_cross_mesh_controlled": mean_mesh_controlled,
        "longest_window_mean_below_reserve": (
            longest_window_mean_small
        ),
        "window_statistics": window_summary,
        "temporal_uncertainty": temporal,
        "profile_support": profile_support,
        "subspace_diagnostics": pods,
        "attribution": attribution_summary,
        "hard_stops": (
            "no_formal_average",
            "no_new_reduced_coordinate",
            "no_relaxation_law",
            "no_embedded_patch_selection",
            "no_reduced_evolution",
            "no_macrostep",
            "no_tide_or_wind",
            "no_hot_state_stability_or_cycle_claim",
        ),
    }
    return summary, output_arrays


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dense-cache-only",
        choices=tuple(
            _trajectory_label(n_cells, resolution, side)
            for n_cells in MESHES
            for resolution in RESOLUTIONS
            for side in SIDES
        ),
        help="Populate one dense fresh-rate cache and exit.",
    )
    parser.add_argument(
        "--force-dense-cache",
        action="store_true",
        help="Recompute the selected dense rate cache.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--arrays",
        type=Path,
        default=DEFAULT_ARRAYS,
    )
    return parser.parse_args()


def _parse_trajectory_label(label: str) -> tuple[int, str, str]:
    mesh, resolution, side = label.split("_", maxsplit=2)
    return int(mesh.removeprefix("N")), resolution, side


def main() -> None:
    args = _arguments()
    if args.dense_cache_only is not None:
        n_cells, resolution, side = _parse_trajectory_label(
            args.dense_cache_only
        )
        payload, _arrays = _run_or_load_dense_rate_cache(
            n_cells=n_cells,
            resolution=resolution,
            side=side,
            force=args.force_dense_cache,
        )
        print(
            json.dumps(
                {
                    "work_package": WORK_PACKAGE,
                    "dense_cache_only": args.dense_cache_only,
                    "arrays_path": payload["arrays_path"],
                    "arrays_sha256": payload["arrays_sha256"],
                    "state_count": payload["state_count"],
                    "wall_seconds": payload["wall_seconds"],
                    "sparse_reproduction": payload[
                        "sparse_reproduction"
                    ],
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return

    started = time.perf_counter()
    dense_by_label = {}
    dense_provenance = {}
    contracts = {}
    operators = {}
    loading_times = {}
    for n_cells in MESHES:
        contract, _case, operator_arrays, _metadata = _load_contract(
            n_cells
        )
        contracts[n_cells] = contract
        operators[n_cells] = operator_arrays
        parent_payload = json.loads(
            (
                N64_PARENT_JSON
                if n_cells == 64
                else N128_PARENT_JSON
            ).read_text(encoding="utf-8")
        )
        loading_times[n_cells] = float(
            parent_payload["diagnostics"]["loading_time_seconds"]
        )
        for resolution in RESOLUTIONS:
            for side in SIDES:
                label = _trajectory_label(n_cells, resolution, side)
                payload, arrays = _load_existing_dense_rate_cache(
                    n_cells=n_cells,
                    resolution=resolution,
                    side=side,
                )
                dense_by_label[label] = arrays
                dense_provenance[label] = {
                    "arrays_path": payload["arrays_path"],
                    "arrays_sha256": payload["arrays_sha256"],
                    "state_count": payload["state_count"],
                    "sparse_reproduction": payload[
                        "sparse_reproduction"
                    ],
                }

    summary, arrays = _analysis(
        dense_by_label=dense_by_label,
        contracts=contracts,
        operators=operators,
        loading_times=loading_times,
    )
    args.arrays.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.arrays, **arrays)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "base_commit": BASE_COMMIT,
        "runner": THIS_RUNNER,
        "runner_sha256": _sha256(ROOT / THIS_RUNNER),
        "parent_evidence": {
            "n64_json": _relative(N64_PARENT_JSON),
            "n64_json_sha256": _sha256(N64_PARENT_JSON),
            "n64_arrays": _relative(N64_PARENT_ARRAYS),
            "n64_arrays_sha256": _sha256(N64_PARENT_ARRAYS),
            "n128_json": _relative(N128_PARENT_JSON),
            "n128_json_sha256": _sha256(N128_PARENT_JSON),
            "n128_arrays": _relative(N128_PARENT_ARRAYS),
            "n128_arrays_sha256": _sha256(N128_PARENT_ARRAYS),
        },
        "dense_rate_caches": dense_provenance,
        "summary": _plain(summary),
        "arrays_path": _relative(args.arrays),
        "arrays_sha256": _sha256(args.arrays),
        "wall_seconds": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    args.output.write_text(
        json.dumps(
            _plain(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "work_package": WORK_PACKAGE,
                "classification": summary["classification"],
                "formal_fast_average_certified": summary[
                    "formal_fast_average_certified"
                ],
                "architecture_change_authorized": summary[
                    "architecture_change_authorized"
                ],
                "arrays_path": _relative(args.arrays),
                "arrays_sha256": payload["arrays_sha256"],
                "wall_seconds": payload["wall_seconds"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
