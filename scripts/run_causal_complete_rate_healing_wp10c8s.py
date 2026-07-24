"""Run the WP10c8s complete slow-rate localization and healing gate.

WP10c8r showed that the scientifically significant ambiguity in the complete
34-coordinate slow-rate field is not represented by the negligible
interface-4 vectors used in the superseded WP10c8q rank-two interpretation.
This package therefore starts from the complete rate operator, constructs
exact finite-amplitude equal-coordinate pairs for independent stress,
thermal, radial-momentum, and sub-shell stress directions, and localizes the
result before authorizing expensive natural-healing trajectories.

The package does not modify production physics, the moment definition, the
spatial flux, the descriptor, or BDF2.  It is deliberately gated:

* the exact nonlinear pair must pass every WP10c8o lift/rate/storage gate;
* its complete slow-rate half-spread must exceed the locked 0.25 screen;
* the matched N64/N128 tangent response must agree;
* only independent, localized, architecture-controlling families may enter
  the natural-healing campaign.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

import run_causal_extended_healing_wp10c8q as wp10c8q
import run_causal_interface_state_sufficiency_wp10c8r as wp10c8r
import run_causal_moment_sufficiency_audit_wp10c8i as wp10c8i
import run_causal_natural_healing_wp10c8p as wp10c8p
import run_causal_nonlinear_fiber_audit_wp10c8o as wp10c8o

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    causal_five_field_observable_snapshot,
    causal_refined_spread_upper_bound,
    evolve_causal_five_field_fixed_bdf2,
    unpack_causal_five_field_state,
)


BASE_COMMIT = "4a209ef0cdec7e835e5e61e0a518eb348989a65c"
WORK_PACKAGE = "WP10c8s"
SCHEMA_VERSION = 1
THIS_RUNNER = "scripts/run_causal_complete_rate_healing_wp10c8s.py"

PARENT_JSON = (
    ROOT
    / "outputs/tables/causal_interface_state_sufficiency_wp10c8r.json"
)
PARENT_ARRAYS = (
    ROOT
    / "outputs/tables/causal_interface_state_sufficiency_wp10c8r_arrays.npz"
)
RATE_PARENT_JSON = (
    ROOT
    / "outputs/checkpoints/causal_five_field_wp10c8q"
    / "slow_rate_fiber_audit.json"
)
RATE_PARENT_ARRAYS = (
    ROOT
    / "outputs/checkpoints/causal_five_field_wp10c8q"
    / "slow_rate_fiber_audit_arrays.npz"
)
OPERATOR_N64 = (
    ROOT
    / "outputs/checkpoints/causal_five_field_wp10c8i"
    / "N064_t_0p025_moment_operators.npz"
)
CHECKPOINT_DIRECTORY = (
    ROOT / "outputs/checkpoints/causal_five_field_wp10c8s"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/tables/causal_complete_rate_healing_wp10c8s.json"
)
DEFAULT_ARRAYS = (
    ROOT / "outputs/tables/causal_complete_rate_healing_wp10c8s_arrays.npz"
)

SEED_MULTIPLIER = 1.0e-3
NONLINEAR_RATE_GATE = wp10c8o.INSTANTANEOUS_SCREEN_GATE
TANGENT_CROSS_MESH_COSINE_GATE = 0.95
TANGENT_CROSS_MESH_AMPLITUDE_RATIO_GATE = 0.50
LOCALIZATION_FRACTION_GATE = 0.80
LINEAR_HEALING_SCREEN_SECONDS = (0.0, 0.025, 0.05, 0.10, 0.125)
LINEAR_RAPID_HEALING_RATIO = 0.10
RAPID_DURATION_SECONDS = 0.025
PERSISTENT_DURATION_SECONDS = 0.125
COARSE_TIMESTEP_SECONDS = 0.0025
FINE_TIMESTEP_SECONDS = 0.00125
RAPID_OUTPUT_OFFSETS_SECONDS = (0.0, 0.0025, 0.005, 0.01, 0.025)
PERSISTENT_OUTPUT_OFFSETS_SECONDS = (0.0, 0.025, 0.05, 0.10, 0.125)
TEMPORAL_UNCERTAINTY_GATE = 0.025
TEMPORAL_RELATIVE_UNCERTAINTY_GATE = 0.10
TEMPORAL_RELATIVE_SPREAD_FLOOR = 0.10
HEALING_FINAL_SPREAD_GATE = 0.10
MINIMUM_HEALING_FACTOR = 2.0
MAXIMUM_SHELL_LEDGER_RELATIVE_DEFECT = 1.0e-3

# Modes zero and four already have exact WP10c8q pairs.  The four remaining
# cases complete the predeclared leading stress/thermal/radial-momentum
# matrix without repeating those expensive nonlinear solves.
CASE_SPECS = (
    {
        "case_id": "mode_0_inner_stress_existing",
        "mode_index": 0,
        "family": "inner_stress",
        "parent_case_id": "n64_slow_rate_held_out_direction",
    },
    {
        "case_id": "mode_1_inner_stress_independent",
        "mode_index": 1,
        "family": "inner_stress_independent",
        "parent_case_id": None,
    },
    {
        "case_id": "mode_2_inner_thermal",
        "mode_index": 2,
        "family": "inner_thermal",
        "parent_case_id": None,
    },
    {
        "case_id": "mode_3_inner_radial_momentum",
        "mode_index": 3,
        "family": "inner_radial_momentum",
        "parent_case_id": None,
    },
    {
        "case_id": "mode_4_middle_stress_existing",
        "mode_index": 4,
        "family": "middle_stress",
        "parent_case_id": "n64_slow_rate_alpha_1.0000e-03",
    },
    {
        "case_id": "mode_7_source_shell_stress",
        "mode_index": 7,
        "family": "source_shell_stress",
        "parent_case_id": None,
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _case_cache_paths(case_id: str) -> tuple[Path, Path]:
    return (
        CHECKPOINT_DIRECTORY / f"{case_id}.json",
        CHECKPOINT_DIRECTORY / f"{case_id}_arrays.npz",
    )


def _slice_parent_case_arrays(
    arrays: dict[str, np.ndarray],
    parent_case_id: str,
) -> dict[str, np.ndarray]:
    prefix = f"{parent_case_id}_"
    result = {
        name.removeprefix(prefix): np.asarray(value)
        for name, value in arrays.items()
        if name.startswith(prefix)
    }
    if not result:
        raise RuntimeError(f"missing WP10c8q parent case {parent_case_id}")
    return result


def _cross_mesh_tangent_row(
    parent_arrays: dict[str, np.ndarray],
    mode_index: int,
) -> dict:
    n64 = np.asarray(
        parent_arrays["n64_t_0p025_top_slow_rate_responses"],
        dtype=float,
    )[mode_index]
    n128 = np.asarray(
        parent_arrays["n128_t_0p025_top_slow_rate_responses"],
        dtype=float,
    )[mode_index]
    denominator = max(
        float(np.linalg.norm(n64) * np.linalg.norm(n128)),
        np.finfo(float).tiny,
    )
    cosine = float(np.dot(n64, n128) / denominator)
    maximum64 = float(np.max(np.abs(n64)))
    maximum128 = float(np.max(np.abs(n128)))
    amplitude_ratio = maximum128 / max(maximum64, np.finfo(float).tiny)
    amplitude_defect = abs(amplitude_ratio - 1.0)
    passed = bool(
        abs(cosine) >= TANGENT_CROSS_MESH_COSINE_GATE
        and amplitude_defect <= TANGENT_CROSS_MESH_AMPLITUDE_RATIO_GATE
    )
    return {
        "signed_response_cosine": cosine,
        "absolute_response_cosine": abs(cosine),
        "n64_maximum_response": maximum64,
        "n128_maximum_response": maximum128,
        "n128_to_n64_maximum_ratio": amplitude_ratio,
        "amplitude_ratio_defect": amplitude_defect,
        "minimum_absolute_cosine": TANGENT_CROSS_MESH_COSINE_GATE,
        "maximum_amplitude_ratio_defect": (
            TANGENT_CROSS_MESH_AMPLITUDE_RATIO_GATE
        ),
        "passed": passed,
    }


def _dominant_support(
    values: np.ndarray,
    *,
    radius_rg: np.ndarray,
    shell_edge_indices: np.ndarray,
) -> dict:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] != radius_rg.size:
        raise ValueError("support array must be cell-by-field")
    absolute = np.abs(array)
    flat = absolute.ravel()
    control = int(np.argmax(flat))
    cell, field = np.unravel_index(control, absolute.shape)
    cell_l1 = np.sum(absolute, axis=1)
    total = max(float(np.sum(cell_l1)), np.finfo(float).tiny)
    order = np.argsort(cell_l1)[::-1]
    cumulative = np.cumsum(cell_l1[order]) / total
    cells_for_90 = int(np.searchsorted(cumulative, 0.90) + 1)
    shell_l1 = []
    for left, right in zip(
        shell_edge_indices[:-1],
        shell_edge_indices[1:],
        strict=True,
    ):
        shell_l1.append(float(np.sum(cell_l1[int(left) : int(right)])))
    shell_l1 = np.asarray(shell_l1, dtype=float)
    shell_fraction = shell_l1 / total
    controlling_shell = int(np.argmax(shell_fraction))
    return {
        "maximum_absolute_value": float(flat[control]),
        "controlling_cell": int(cell),
        "controlling_field": int(field),
        "controlling_radius_rg": float(radius_rg[cell]),
        "cells_containing_90_percent_l1": cells_for_90,
        "shell_l1_fractions": shell_fraction,
        "controlling_shell": controlling_shell,
        "controlling_shell_l1_fraction": float(
            shell_fraction[controlling_shell]
        ),
        "localized_in_one_shell": bool(
            shell_fraction[controlling_shell] >= LOCALIZATION_FRACTION_GATE
        ),
    }


def _pair_localization(
    *,
    context,
    arrays: dict[str, np.ndarray],
    operator_arrays: dict[str, np.ndarray],
) -> tuple[dict, dict[str, np.ndarray]]:
    n_cells = int(context.grid.centers.size)
    radius_rg = (
        np.asarray(context.grid.centers, dtype=float)
        / context.grid.gravitational_radius
    )
    shell_edges = np.asarray(
        operator_arrays["shell_edge_indices"],
        dtype=int,
    )
    primitive_amplitudes = np.asarray(
        operator_arrays["physical_input_amplitudes"],
        dtype=float,
    ).reshape(n_cells, 5)
    conservation_scales = np.asarray(
        operator_arrays["conservation_row_scales"],
        dtype=float,
    ).reshape(n_cells, 5)

    minus_primitive = np.asarray(
        arrays["minus_primitive_vector"], dtype=float
    ).reshape(n_cells, 5)
    plus_primitive = np.asarray(
        arrays["plus_primitive_vector"], dtype=float
    ).reshape(n_cells, 5)
    state_half = 0.5 * (plus_primitive - minus_primitive)
    state_normalized = state_half / primitive_amplitudes

    minus_rate = np.asarray(
        arrays["minus_scaled_primitive_rate_per_s"], dtype=float
    ).reshape(n_cells, 5)
    plus_rate = np.asarray(
        arrays["plus_scaled_primitive_rate_per_s"], dtype=float
    ).reshape(n_cells, 5)
    primitive_rate_half = 0.5 * (plus_rate - minus_rate)

    storage_components = {}
    for label, key in (
        ("total", "path_total_storage_action"),
        ("mapped", "path_mapped_storage_action"),
        ("responsive_height", "path_responsive_height_storage_action"),
    ):
        minus = np.asarray(arrays[f"minus_{key}"], dtype=float).reshape(
            n_cells, 5
        )
        plus = np.asarray(arrays[f"plus_{key}"], dtype=float).reshape(
            n_cells, 5
        )
        storage_components[label] = 0.5 * (plus - minus)

    minus_state = unpack_causal_five_field_state(
        np.asarray(arrays["minus_state_vector"], dtype=float),
        n_cells,
    )
    plus_state = unpack_causal_five_field_state(
        np.asarray(arrays["plus_state_vector"], dtype=float),
        n_cells,
    )
    minus_flux = np.asarray(
        minus_state.weighted_face_fluxes_over_c, dtype=float
    )
    plus_flux = np.asarray(
        plus_state.weighted_face_fluxes_over_c, dtype=float
    )
    flux_divergence_half = 0.5 * (
        (plus_flux[1:] - plus_flux[:-1])
        - (minus_flux[1:] - minus_flux[:-1])
    ) / conservation_scales
    # The exact rate solve obeys total_storage + stationary = 0, while the
    # stationary conservation residual is flux_divergence - physical_source.
    stationary_half = -storage_components["total"]
    inferred_source_half = flux_divergence_half - stationary_half
    balance_defect = float(
        np.max(np.abs(storage_components["total"] + stationary_half))
    )
    storage_component_defect = float(
        np.max(
            np.abs(
                storage_components["total"]
                - storage_components["mapped"]
                - storage_components["responsive_height"]
            )
        )
        / max(
            float(np.max(np.abs(storage_components["total"]))),
            np.finfo(float).tiny,
        )
    )

    blocks = {
        "state": state_normalized,
        "primitive_rate": primitive_rate_half,
        "total_storage": storage_components["total"],
        "mapped_storage": storage_components["mapped"],
        "responsive_height_storage": storage_components[
            "responsive_height"
        ],
        "flux_divergence": flux_divergence_half,
        "inferred_physical_source": inferred_source_half,
    }
    supports = {
        label: _dominant_support(
            values,
            radius_rg=radius_rg,
            shell_edge_indices=shell_edges,
        )
        for label, values in blocks.items()
    }
    term_maxima = {
        label: float(np.max(np.abs(values)))
        for label, values in blocks.items()
    }
    dynamic_terms = (
        "mapped_storage",
        "responsive_height_storage",
        "flux_divergence",
        "inferred_physical_source",
    )
    controlling_term = max(dynamic_terms, key=term_maxima.__getitem__)
    return {
        "state_support": supports["state"],
        "primitive_rate_support": supports["primitive_rate"],
        "term_support": {
            label: supports[label] for label in dynamic_terms
        },
        "term_maximum_absolute_values": {
            label: term_maxima[label] for label in dynamic_terms
        },
        "controlling_dynamic_term": controlling_term,
        "controlling_dynamic_term_support": supports[controlling_term],
        "storage_balance_maximum_absolute_defect": balance_defect,
        "storage_component_relative_defect": storage_component_defect,
    }, {
        "radius_rg": radius_rg,
        "shell_edge_indices": shell_edges,
        "state_half_difference_over_amplitude": state_normalized,
        "scaled_primitive_rate_half_difference_per_s": primitive_rate_half,
        "total_storage_action_half_difference": storage_components["total"],
        "mapped_storage_action_half_difference": storage_components["mapped"],
        "responsive_height_storage_action_half_difference": (
            storage_components["responsive_height"]
        ),
        "flux_divergence_half_difference": flux_divergence_half,
        "inferred_physical_source_half_difference": inferred_source_half,
    }


def _linear_healing_preflight(
    *,
    mode_direction: np.ndarray,
    slow_rate_rows: np.ndarray,
    dynamic: np.ndarray,
) -> tuple[dict, np.ndarray]:
    from scipy.linalg import expm

    direction = np.asarray(mode_direction, dtype=float)
    rows = []
    for time_seconds in LINEAR_HEALING_SCREEN_SECONDS:
        evolved = expm(float(time_seconds) * dynamic) @ direction
        rows.append(slow_rate_rows @ evolved)
    responses = np.asarray(rows, dtype=float)
    maxima = np.max(np.abs(responses), axis=1)
    initial = max(float(maxima[0]), np.finfo(float).tiny)
    ratios = maxima / initial
    final_ratio = float(ratios[-1])
    return {
        "times_seconds": LINEAR_HEALING_SCREEN_SECONDS,
        "maximum_slow_rate_responses": maxima,
        "ratios_to_initial": ratios,
        "final_ratio": final_ratio,
        "rapid_linear_healing_predicted": bool(
            final_ratio <= LINEAR_RAPID_HEALING_RATIO
        ),
        "semantics": (
            "Frozen-tangent preflight only; it does not replace a nonlinear "
            "natural-healing trajectory."
        ),
    }, responses


def _load_or_build_new_case(
    *,
    spec: dict,
    seed_direction: np.ndarray,
    initial_by_mesh: dict,
    vectors_by_mesh: dict,
    contracts: dict,
    operator_arrays: dict[str, np.ndarray],
    force: bool,
) -> tuple[dict, dict[str, np.ndarray]]:
    json_path, arrays_path = _case_cache_paths(spec["case_id"])
    expected = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "base_commit": BASE_COMMIT,
        "case_id": spec["case_id"],
        "mode_index": spec["mode_index"],
        "seed_sha256": _array_sha256(seed_direction),
        "seed_multiplier": SEED_MULTIPLIER,
        "parent_arrays_sha256": _sha256(PARENT_ARRAYS),
    }
    if json_path.exists() and arrays_path.exists() and not force:
        cached = _load_json(json_path)
        if (
            all(cached.get(key) == value for key, value in expected.items())
            and cached.get("arrays_sha256") == _sha256(arrays_path)
        ):
            return cached["row"], _load_npz(arrays_path)
        raise RuntimeError(f"stale WP10c8s case cache: {json_path}")

    started = time.perf_counter()
    row, _pair_arrays, runtime = wp10c8o._build_pair(
        case_id=spec["case_id"],
        seed_name=f"wp10c8r_complete_rate_mode_{spec['mode_index']}",
        seed_origin=(
            "significance-gated complete 34-coordinate slow-rate singular "
            f"direction {spec['mode_index']} at N64 t=0.025 s"
        ),
        seed_direction=seed_direction,
        seed_multiplier=SEED_MULTIPLIER,
        initial=initial_by_mesh[64],
        vector=vectors_by_mesh[64][wp10c8o.PRIMARY_ANCHOR],
        cache=operator_arrays,
        shell_edges_rg=contracts[64]["shell_edges_rg"],
        require_face58_switch=False,
    )
    wp10c8o._complete_pair_rates(
        row,
        runtime,
        binding_dae_storage_audit=True,
    )
    loading = float(
        _load_json(PARENT_JSON)["loading_time_inference"][
            "n64_primary_seconds"
        ]
    )
    wp10c8q._actual_slow_rate_row(row, runtime, loading)
    row["mode_index"] = spec["mode_index"]
    row["family"] = spec["family"]
    row["parent_case_reused"] = False
    row["total_case_wall_seconds"] = time.perf_counter() - started
    arrays = {
        name: np.asarray(value)
        for name, value in runtime["arrays"].items()
    }
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **arrays)
    payload = {
        **expected,
        "row": _plain(row),
        "arrays_path": _relative(arrays_path),
        "arrays_sha256": _sha256(arrays_path),
    }
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return row, arrays


def _case_spec(case_id: str) -> dict:
    try:
        return next(row for row in CASE_SPECS if row["case_id"] == case_id)
    except StopIteration as exc:
        raise KeyError(f"unknown WP10c8s case {case_id}") from exc


def _load_case_arrays(
    *,
    spec: dict,
    rate_parent_arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if spec["parent_case_id"] is not None:
        return _slice_parent_case_arrays(
            rate_parent_arrays,
            spec["parent_case_id"],
        )
    json_path, arrays_path = _case_cache_paths(spec["case_id"])
    if not json_path.exists() or not arrays_path.exists():
        raise FileNotFoundError(
            f"WP10c8s nonlinear pair is missing: {spec['case_id']}"
        )
    cached = _load_json(json_path)
    if cached.get("arrays_sha256") != _sha256(arrays_path):
        raise RuntimeError(f"WP10c8s pair cache differs: {arrays_path}")
    return _load_npz(arrays_path)


def _healing_contract(case_id: str) -> dict:
    if case_id in {
        "mode_4_middle_stress_existing",
        "mode_7_source_shell_stress",
    }:
        duration = PERSISTENT_DURATION_SECONDS
        outputs = PERSISTENT_OUTPUT_OFFSETS_SECONDS
    else:
        duration = RAPID_DURATION_SECONDS
        outputs = RAPID_OUTPUT_OFFSETS_SECONDS
    coarse = int(round(duration / COARSE_TIMESTEP_SECONDS))
    fine = int(round(duration / FINE_TIMESTEP_SECONDS))
    if not (
        np.isclose(
            coarse * COARSE_TIMESTEP_SECONDS,
            duration,
            rtol=0.0,
            atol=1.0e-14,
        )
        and np.isclose(
            fine * FINE_TIMESTEP_SECONDS,
            duration,
            rtol=0.0,
            atol=1.0e-14,
        )
    ):
        raise RuntimeError("WP10c8s duration is not commensurate with timestep")
    return {
        "duration_seconds": duration,
        "output_offsets_seconds": outputs,
        "coarse_subdivisions": coarse,
        "fine_subdivisions": fine,
        "coarse_timestep_seconds": COARSE_TIMESTEP_SECONDS,
        "fine_timestep_seconds": FINE_TIMESTEP_SECONDS,
    }


def _trajectory_path(
    case_id: str,
    resolution: str,
    side: str,
) -> Path:
    return (
        CHECKPOINT_DIRECTORY
        / "trajectories"
        / f"{case_id}_{resolution}_{side}.npz"
    )


def _run_or_load_trajectory(
    *,
    case_id: str,
    resolution: str,
    side: str,
    context,
    initial_vector: np.ndarray,
    force: bool,
) -> dict:
    contract = _healing_contract(case_id)
    subdivisions = int(contract[f"{resolution}_subdivisions"])
    duration = float(contract["duration_seconds"])
    timestep = duration / subdivisions
    path = _trajectory_path(case_id, resolution, side)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "base_commit": BASE_COMMIT,
        "case_id": case_id,
        "resolution": resolution,
        "side": side,
        "subdivisions": subdivisions,
        "duration_seconds": duration,
        "timestep_seconds": timestep,
        "initial_state_sha256": _array_sha256(initial_vector),
        "startup": "one_bdf1_with_zero_predictor_then_fixed_bdf2",
    }
    if path.exists() and not force:
        with np.load(path, allow_pickle=False) as source:
            metadata = json.loads(str(source["metadata_json"].item()))
            states = np.asarray(source["states"], dtype=float)
        if all(
            metadata.get(key) == value
            for key, value in expected.items()
        ):
            return {
                "path": path,
                "sha256": _sha256(path),
                "metadata": metadata,
                "summary": metadata["summary"],
                "states": states,
                "cached": True,
            }
        raise RuntimeError(f"stale WP10c8s trajectory: {path}")

    snapshots = [np.asarray(initial_vector, dtype=float).copy()]

    def progress(completed, total, state, _history) -> None:
        snapshots.append(np.asarray(state, dtype=float).copy())
        print(
            f"WP10c8s {case_id} {resolution} {side}: "
            f"step {completed}/{total}",
            flush=True,
        )

    started = time.perf_counter()
    result = evolve_causal_five_field_fixed_bdf2(
        context,
        np.asarray(initial_vector, dtype=float),
        np.zeros_like(initial_vector, dtype=float),
        timestep,
        duration,
        subdivisions,
        wp10c8p._step_config(),
        startup_with_bdf1=True,
        progress=progress,
    )
    wall_seconds = time.perf_counter() - started
    states = np.asarray(snapshots, dtype=float)
    if states.shape[0] != result.completed_steps + 1:
        raise RuntimeError("WP10c8s trajectory snapshot count is inconsistent")
    summary = wp10c8p._result_row(result, wall_seconds)
    metadata = {**expected, "summary": _plain(summary)}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        states=states,
        metadata_json=np.asarray(
            json.dumps(metadata, sort_keys=True, allow_nan=False)
        ),
    )
    return {
        "path": path,
        "sha256": _sha256(path),
        "metadata": metadata,
        "summary": summary,
        "states": states,
        "cached": False,
    }


def _diagnostic_paths(case_id: str) -> tuple[Path, Path]:
    return (
        CHECKPOINT_DIRECTORY / "diagnostics" / f"{case_id}.json",
        CHECKPOINT_DIRECTORY / "diagnostics" / f"{case_id}_arrays.npz",
    )


def _pair_spread_arrays(
    *,
    minus: dict[str, np.ndarray],
    plus: dict[str, np.ndarray],
    coordinate_scales: np.ndarray,
    loading_time_seconds: float,
) -> dict[str, np.ndarray]:
    if not np.array_equal(minus["output_times"], plus["output_times"]):
        raise RuntimeError("WP10c8s plus/minus output times differ")
    static_gates = np.asarray(minus["static_output_gates"], dtype=float)
    static_spreads = 0.5 * np.abs(
        plus["static_outputs"] - minus["static_outputs"]
    ) / static_gates[None, :]
    slow_rate_spreads = (
        0.5
        * np.abs(
            plus["normalized_coordinate_rates"]
            - minus["normalized_coordinate_rates"]
        )
        * float(loading_time_seconds)
        / wp10c8o.COORDINATE_RATE_WINDOW_SECONDS
    )
    coordinate_spreads = 0.5 * np.abs(
        plus["coordinates"] - minus["coordinates"]
    ) / coordinate_scales[None, :]
    return {
        "times": np.asarray(minus["output_times"], dtype=float),
        "static_spreads": static_spreads,
        "slow_rate_spreads": slow_rate_spreads,
        "full_spreads": np.concatenate(
            (static_spreads, slow_rate_spreads),
            axis=1,
        ),
        "coordinate_spreads": coordinate_spreads,
        "full_names": np.concatenate(
            (
                np.asarray(minus["static_output_names"], dtype="U"),
                np.asarray(
                    [
                        f"slow_rate_{value}"
                        for value in minus["coordinate_names"]
                    ],
                    dtype="U",
                ),
            )
        ),
    }


def _healing_decision(
    *,
    coarse: dict[str, np.ndarray],
    fine: dict[str, np.ndarray],
) -> tuple[dict, dict[str, np.ndarray]]:
    if not (
        np.array_equal(coarse["times"], fine["times"])
        and np.array_equal(coarse["full_names"], fine["full_names"])
    ):
        raise RuntimeError("WP10c8s coarse/fine schemas differ")
    uncertainty, upper = causal_refined_spread_upper_bound(
        coarse["full_spreads"],
        fine["full_spreads"],
    )
    coordinate_uncertainty, coordinate_upper = (
        causal_refined_spread_upper_bound(
            coarse["coordinate_spreads"],
            fine["coordinate_spreads"],
        )
    )
    relative_mask = (
        fine["full_spreads"] >= TEMPORAL_RELATIVE_SPREAD_FLOOR
    )
    relative_passed = bool(
        np.all(
            uncertainty[relative_mask]
            <= TEMPORAL_RELATIVE_UNCERTAINTY_GATE
            * fine["full_spreads"][relative_mask]
        )
    )
    temporal_passed = bool(
        float(np.max(uncertainty)) <= TEMPORAL_UNCERTAINTY_GATE
        and relative_passed
    )
    initial = upper[0]
    final = upper[-1]
    significant = initial >= NONLINEAR_RATE_GATE
    factor_two = bool(
        np.all(
            final[significant]
            <= initial[significant] / MINIMUM_HEALING_FACTOR
        )
    )
    no_regrowth = bool(
        np.all(
            fine["full_spreads"][-1, significant]
            <= fine["full_spreads"][-2, significant]
            + uncertainty[-1, significant]
            + uncertainty[-2, significant]
        )
    )
    final_gate = bool(
        np.all(final[significant] <= HEALING_FINAL_SPREAD_GATE)
    )
    lower = np.maximum(
        fine["full_spreads"][-1] - uncertainty[-1],
        0.0,
    )
    persistent_mask = significant & (
        lower > HEALING_FINAL_SPREAD_GATE
    )
    persistence_separated = bool(np.any(persistent_mask))
    healed = bool(
        temporal_passed
        and np.any(significant)
        and factor_two
        and no_regrowth
        and final_gate
    )
    initial_maximum = max(
        float(np.max(initial[significant])),
        np.finfo(float).tiny,
    )
    final_maximum = max(
        float(np.max(final[significant])),
        np.finfo(float).tiny,
    )
    e_folds = float(np.log(initial_maximum / final_maximum))
    control_initial = int(np.argmax(initial))
    control_final = int(np.argmax(final))
    return {
        "maximum_temporal_uncertainty": float(np.max(uncertainty)),
        "maximum_relative_temporal_uncertainty": float(
            np.max(
                uncertainty[relative_mask]
                / fine["full_spreads"][relative_mask]
            )
            if np.any(relative_mask)
            else 0.0
        ),
        "temporal_uncertainty_passed": temporal_passed,
        "initial_maximum_upper_spread": float(np.max(initial)),
        "initial_controlling_output": str(
            coarse["full_names"][control_initial]
        ),
        "final_maximum_upper_spread": float(np.max(final)),
        "final_controlling_output": str(
            coarse["full_names"][control_final]
        ),
        "significant_initial_output_count": int(np.count_nonzero(significant)),
        "factor_two_decay_passed": factor_two,
        "no_late_regrowth_passed": no_regrowth,
        "final_healing_gate_passed": final_gate,
        "final_maximum_lower_spread": float(np.max(lower)),
        "persistence_separated_from_healing_gate": persistence_separated,
        "persistent_output_count": int(np.count_nonzero(persistent_mask)),
        "measured_minimum_controlling_e_folds": e_folds,
        "final_maximum_coordinate_upper_spread": float(
            np.max(coordinate_upper[-1])
        ),
        "natural_healing_passed": healed,
    }, {
        "temporal_uncertainty": uncertainty,
        "upper_spreads": upper,
        "coordinate_temporal_uncertainty": coordinate_uncertainty,
        "coordinate_upper_spreads": coordinate_upper,
        "significant_initial_output_mask": significant,
        "final_lower_spreads": lower,
        "persistent_output_mask": persistent_mask,
    }


def _run_or_load_healing_diagnostic(
    *,
    case_id: str,
    context,
    anchor_vector: np.ndarray,
    case_arrays: dict[str, np.ndarray],
    operator_arrays: dict[str, np.ndarray],
    loading_time_seconds: float,
    shell_edges_rg: np.ndarray,
    force: bool,
) -> tuple[dict, dict[str, np.ndarray]]:
    json_path, arrays_path = _diagnostic_paths(case_id)
    trajectory_rows = {}
    for resolution in ("coarse", "fine"):
        for side in ("minus", "plus"):
            trajectory_rows[f"{resolution}_{side}"] = (
                _run_or_load_trajectory(
                    case_id=case_id,
                    resolution=resolution,
                    side=side,
                    context=context,
                    initial_vector=np.asarray(
                        case_arrays[f"{side}_state_vector"],
                        dtype=float,
                    ),
                    force=False,
                )
            )
    expected = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "base_commit": BASE_COMMIT,
        "case_id": case_id,
        "trajectory_hashes": {
            key: value["sha256"] for key, value in trajectory_rows.items()
        },
        "loading_time_seconds": float(loading_time_seconds),
    }
    if json_path.exists() and arrays_path.exists() and not force:
        cached = _load_json(json_path)
        if (
            all(cached.get(key) == value for key, value in expected.items())
            and cached.get("arrays_sha256") == _sha256(arrays_path)
        ):
            return cached["summary"], _load_npz(arrays_path)
        raise RuntimeError(f"stale WP10c8s healing diagnostic: {json_path}")

    baseline = causal_five_field_observable_snapshot(
        context,
        anchor_vector,
        cooling_inner_cutoff=6.0 * context.grid.gravitational_radius,
    )
    radius_rg = (
        context.grid.centers / context.grid.gravitational_radius
    )
    grid_edges_rg = (
        context.grid.edges / context.grid.gravitational_radius
    )
    _common_radius, common_interpolation = (
        wp10c8i._common_log_h_interpolation(radius_rg, grid_edges_rg)
    )
    coordinate_names = tuple(
        str(value) for value in case_arrays["coordinate_names"]
    )
    coordinate_scales = np.asarray(
        case_arrays["coordinate_scales"], dtype=float
    )
    contract = _healing_contract(case_id)
    rate_cache: dict[str, tuple[np.ndarray, dict]] = {}
    diagnostics = {}
    all_arrays: dict[str, np.ndarray] = {}
    started = time.perf_counter()
    for resolution in ("coarse", "fine"):
        for side in ("minus", "plus"):
            label = f"{resolution}_{side}"
            summary, arrays = wp10c8p._trajectory_diagnostics(
                context=context,
                states=trajectory_rows[label]["states"],
                subdivisions=int(contract[f"{resolution}_subdivisions"]),
                shell_edges_rg=shell_edges_rg,
                baseline_snapshot=baseline,
                anchor_interface_scales=np.asarray(
                    case_arrays["interface_flux_scales"],
                    dtype=float,
                ),
                coordinate_names=coordinate_names,
                coordinate_scales=coordinate_scales,
                primitive_scales=np.asarray(
                    operator_arrays["primitive_column_scales"],
                    dtype=float,
                ),
                conservation_scales=np.asarray(
                    operator_arrays["conservation_row_scales"],
                    dtype=float,
                ),
                common_interpolation=common_interpolation,
                compute_fresh_rates=True,
                rate_cache=rate_cache,
                duration_seconds=float(contract["duration_seconds"]),
                output_offsets_seconds=tuple(
                    contract["output_offsets_seconds"]
                ),
            )
            diagnostics[label] = summary
            all_arrays.update(
                {f"{label}_{name}": value for name, value in arrays.items()}
            )

    pair_arrays = {}
    for resolution in ("coarse", "fine"):
        minus = {
            name.removeprefix(f"{resolution}_minus_"): value
            for name, value in all_arrays.items()
            if name.startswith(f"{resolution}_minus_")
        }
        plus = {
            name.removeprefix(f"{resolution}_plus_"): value
            for name, value in all_arrays.items()
            if name.startswith(f"{resolution}_plus_")
        }
        spread = _pair_spread_arrays(
            minus=minus,
            plus=plus,
            coordinate_scales=coordinate_scales,
            loading_time_seconds=loading_time_seconds,
        )
        pair_arrays[resolution] = spread
        all_arrays.update(
            {
                f"{resolution}_pair_{name}": value
                for name, value in spread.items()
            }
        )
    decision, decision_arrays = _healing_decision(
        coarse=pair_arrays["coarse"],
        fine=pair_arrays["fine"],
    )
    all_arrays.update(
        {f"decision_{name}": value for name, value in decision_arrays.items()}
    )
    trajectories_passed = bool(
        all(row["summary"]["passed"] for row in trajectory_rows.values())
    )
    diagnostics_passed = bool(
        all(
            row["maximum_physical_mje_shell_ledger_relative_defect"]
            <= (
                2.0 * MAXIMUM_SHELL_LEDGER_RELATIVE_DEFECT
                if label.startswith("coarse_")
                else MAXIMUM_SHELL_LEDGER_RELATIVE_DEFECT
            )
            and row["maximum_flux_reconstruction_defect"]
            <= wp10c8p.MAXIMUM_FLUX_RECONSTRUCTION_DEFECT
            and row["all_output_state_gates_passed"]
            and row["all_fresh_rate_audits_passed"]
            for label, row in diagnostics.items()
        )
    )
    numerically_interpretable = bool(
        trajectories_passed
        and diagnostics_passed
        and (
            decision["temporal_uncertainty_passed"]
            or decision["persistence_separated_from_healing_gate"]
        )
    )
    if not numerically_interpretable:
        classification = "numerically_inconclusive"
    elif decision["natural_healing_passed"]:
        classification = (
            f"natural_healing_supported_through_"
            f"{contract['duration_seconds']:.3f}s"
        )
    elif decision["persistence_separated_from_healing_gate"]:
        classification = (
            f"natural_healing_rejected_with_resolved_lower_bound_through_"
            f"{contract['duration_seconds']:.3f}s"
        )
    else:
        classification = (
            f"natural_healing_not_observed_through_"
            f"{contract['duration_seconds']:.3f}s"
        )
    summary = {
        "case_id": case_id,
        "contract": contract,
        "trajectory_provenance": {
            key: {
                "path": _relative(value["path"]),
                "sha256": value["sha256"],
                "cached": value["cached"],
                "summary": value["summary"],
            }
            for key, value in trajectory_rows.items()
        },
        "trajectory_diagnostics": diagnostics,
        "healing_decision": decision,
        "trajectory_contracts_passed": trajectories_passed,
        "diagnostic_contracts_passed": diagnostics_passed,
        "numerically_interpretable": numerically_interpretable,
        "classification": classification,
        "wall_seconds": time.perf_counter() - started,
    }
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **all_arrays)
    payload = {
        **expected,
        "summary": _plain(summary),
        "arrays_path": _relative(arrays_path),
        "arrays_sha256": _sha256(arrays_path),
    }
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary, all_arrays


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case-only",
        choices=tuple(
            spec["case_id"]
            for spec in CASE_SPECS
            if spec["parent_case_id"] is None
        ),
        default=None,
        help="Populate one new nonlinear-pair cache and exit.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute selected nonlinear-pair caches.",
    )
    parser.add_argument(
        "--trajectory-case",
        choices=tuple(spec["case_id"] for spec in CASE_SPECS),
        default=None,
    )
    parser.add_argument(
        "--trajectory-resolution",
        choices=("coarse", "fine"),
        default=None,
    )
    parser.add_argument(
        "--trajectory-side",
        choices=("minus", "plus"),
        default=None,
    )
    parser.add_argument(
        "--diagnostic-case",
        choices=tuple(spec["case_id"] for spec in CASE_SPECS),
        default=None,
    )
    parser.add_argument(
        "--skip-healing-assembly",
        action="store_true",
        help="Assemble only the static localization evidence.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--arrays", type=Path, default=DEFAULT_ARRAYS)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    required = (
        PARENT_JSON,
        PARENT_ARRAYS,
        RATE_PARENT_JSON,
        RATE_PARENT_ARRAYS,
        OPERATOR_N64,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"required WP10c8s parent is missing: {path}")
    parent = _load_json(PARENT_JSON)
    parent_rate = _load_json(RATE_PARENT_JSON)
    if not (
        parent.get("work_package") == "WP10c8r"
        and parent.get("artifacts", {}).get("arrays_sha256")
        == _sha256(PARENT_ARRAYS)
        and parent_rate.get("work_package") == "WP10c8q"
        and parent_rate.get("arrays_sha256") == _sha256(RATE_PARENT_ARRAYS)
    ):
        raise RuntimeError("WP10c8s parent provenance failed")

    parent_arrays = _load_npz(PARENT_ARRAYS)
    rate_parent_arrays = _load_npz(RATE_PARENT_ARRAYS)
    operator_arrays, operator_metadata = wp10c8r._load_operator_cache(
        OPERATOR_N64
    )
    initial_by_mesh, vectors_by_mesh, state_provenance, contracts = (
        wp10c8q._runtime_contracts()
    )

    top_directions = np.asarray(
        parent_arrays["n64_t_0p025_top_tested_state_directions"],
        dtype=float,
    )
    # WP10c8r stored already-tested directions at a 1e-3 box amplitude.
    unit_seed_directions = top_directions / wp10c8r.AUDIT_SEED_MULTIPLIER
    loading = float(parent["loading_time_inference"]["n64_primary_seconds"])
    rate_rows, _gates, _diagnostics = wp10c8r.wp10c8i._rate_output_rows(
        operator_arrays,
        operator_metadata,
        wp10c8o.LEVEL_INDEX,
    )
    slow_rate_rows = (
        loading / wp10c8o.COORDINATE_RATE_WINDOW_SECONDS * rate_rows
    )
    dynamic = np.asarray(operator_arrays["dynamic"], dtype=float)

    if args.trajectory_case is not None:
        if (
            args.trajectory_resolution is None
            or args.trajectory_side is None
        ):
            raise ValueError(
                "--trajectory-case requires --trajectory-resolution and "
                "--trajectory-side"
            )
        spec = _case_spec(args.trajectory_case)
        arrays = _load_case_arrays(
            spec=spec,
            rate_parent_arrays=rate_parent_arrays,
        )
        result = _run_or_load_trajectory(
            case_id=args.trajectory_case,
            resolution=args.trajectory_resolution,
            side=args.trajectory_side,
            context=contracts[64]["context"],
            initial_vector=np.asarray(
                arrays[f"{args.trajectory_side}_state_vector"],
                dtype=float,
            ),
            force=args.force,
        )
        print(
            json.dumps(
                _plain(
                    {
                        "case_id": args.trajectory_case,
                        "resolution": args.trajectory_resolution,
                        "side": args.trajectory_side,
                        "path": _relative(result["path"]),
                        "sha256": result["sha256"],
                        "cached": result["cached"],
                        "summary": result["summary"],
                    }
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.diagnostic_case is not None:
        spec = _case_spec(args.diagnostic_case)
        arrays = _load_case_arrays(
            spec=spec,
            rate_parent_arrays=rate_parent_arrays,
        )
        summary, _values = _run_or_load_healing_diagnostic(
            case_id=args.diagnostic_case,
            context=contracts[64]["context"],
            anchor_vector=vectors_by_mesh[64][wp10c8o.PRIMARY_ANCHOR],
            case_arrays=arrays,
            operator_arrays=operator_arrays,
            loading_time_seconds=loading,
            shell_edges_rg=contracts[64]["shell_edges_rg"],
            force=args.force,
        )
        print(json.dumps(_plain(summary), indent=2, sort_keys=True))
        return

    if args.case_only is not None:
        spec = next(
            row for row in CASE_SPECS if row["case_id"] == args.case_only
        )
        row, _arrays = _load_or_build_new_case(
            spec=spec,
            seed_direction=unit_seed_directions[spec["mode_index"]],
            initial_by_mesh=initial_by_mesh,
            vectors_by_mesh=vectors_by_mesh,
            contracts=contracts,
            operator_arrays=operator_arrays,
            force=args.force,
        )
        print(json.dumps(_plain(row), indent=2, sort_keys=True))
        return

    rows = {}
    output_arrays: dict[str, np.ndarray] = {}
    tangent_confirmations = {}
    linear_preflights = {}
    localizations = {}
    for spec in CASE_SPECS:
        case_id = spec["case_id"]
        if spec["parent_case_id"] is not None:
            parent_case = spec["parent_case_id"]
            row = dict(parent_rate["summary"]["cases"][parent_case])
            arrays = _slice_parent_case_arrays(
                rate_parent_arrays,
                parent_case,
            )
            row.update(
                {
                    "case_id": case_id,
                    "mode_index": spec["mode_index"],
                    "family": spec["family"],
                    "parent_case_reused": True,
                    "parent_case_id": parent_case,
                }
            )
        else:
            row, arrays = _load_or_build_new_case(
                spec=spec,
                seed_direction=unit_seed_directions[spec["mode_index"]],
                initial_by_mesh=initial_by_mesh,
                vectors_by_mesh=vectors_by_mesh,
                contracts=contracts,
                operator_arrays=operator_arrays,
                force=args.force,
            )
        tangent = _cross_mesh_tangent_row(
            parent_arrays,
            spec["mode_index"],
        )
        localization, localization_arrays = _pair_localization(
            context=contracts[64]["context"],
            arrays=arrays,
            operator_arrays=operator_arrays,
        )
        linear, linear_arrays = _linear_healing_preflight(
            mode_direction=top_directions[spec["mode_index"]],
            slow_rate_rows=slow_rate_rows,
            dynamic=dynamic,
        )
        slow_rate = row["slow_rate_audit"]
        nonlinear_significant = bool(
            row["lift_valid"]
            and row["full_output"] is not None
            and row["full_output"]["all_fresh_rate_gates_passed"]
            and row["full_output"][
                "all_binding_dae_storage_audits_passed"
            ]
            and slow_rate[
                "maximum_absolute_half_difference_per_unit_slow_time"
            ]
            >= NONLINEAR_RATE_GATE
        )
        row["wp10c8s_nonlinear_significant"] = nonlinear_significant
        row["wp10c8s_cross_mesh_tangent_passed"] = tangent["passed"]
        rows[case_id] = row
        tangent_confirmations[case_id] = tangent
        localizations[case_id] = localization
        linear_preflights[case_id] = linear
        output_arrays.update(
            {
                f"{case_id}_{name}": np.asarray(value)
                for name, value in arrays.items()
            }
        )
        output_arrays.update(
            {
                f"{case_id}_localization_{name}": value
                for name, value in localization_arrays.items()
            }
        )
        output_arrays[
            f"{case_id}_linear_healing_slow_rate_responses"
        ] = linear_arrays

    authorized = [
        case_id
        for case_id, row in rows.items()
        if row["wp10c8s_nonlinear_significant"]
        and row["wp10c8s_cross_mesh_tangent_passed"]
    ]
    persistent_preflight = [
        case_id
        for case_id in authorized
        if not linear_preflights[case_id][
            "rapid_linear_healing_predicted"
        ]
    ]
    rapidly_healing_preflight = [
        case_id
        for case_id in authorized
        if linear_preflights[case_id][
            "rapid_linear_healing_predicted"
        ]
    ]
    healing = {}
    pending_healing = []
    if authorized and not args.skip_healing_assembly:
        for case_id in authorized:
            paths = [
                _trajectory_path(case_id, resolution, side)
                for resolution in ("coarse", "fine")
                for side in ("minus", "plus")
            ]
            if not all(path.exists() for path in paths):
                pending_healing.append(case_id)
                continue
            spec = _case_spec(case_id)
            arrays = _load_case_arrays(
                spec=spec,
                rate_parent_arrays=rate_parent_arrays,
            )
            summary, diagnostic_arrays = _run_or_load_healing_diagnostic(
                case_id=case_id,
                context=contracts[64]["context"],
                anchor_vector=vectors_by_mesh[64][wp10c8o.PRIMARY_ANCHOR],
                case_arrays=arrays,
                operator_arrays=operator_arrays,
                loading_time_seconds=loading,
                shell_edges_rg=contracts[64]["shell_edges_rg"],
                force=False,
            )
            healing[case_id] = summary
            output_arrays.update(
                {
                    f"{case_id}_healing_{name}": value
                    for name, value in diagnostic_arrays.items()
                }
            )
    all_healing_complete = bool(
        authorized
        and not pending_healing
        and len(healing) == len(authorized)
    )
    healed_cases = [
        case_id
        for case_id, row in healing.items()
        if row["numerically_interpretable"]
        and row["healing_decision"]["natural_healing_passed"]
    ]
    persistent_cases = [
        case_id
        for case_id, row in healing.items()
        if row["numerically_interpretable"]
        and not row["healing_decision"]["natural_healing_passed"]
    ]
    inconclusive_cases = [
        case_id
        for case_id, row in healing.items()
        if not row["numerically_interpretable"]
    ]
    single_interface_route_rejected = bool(persistent_cases)
    if not authorized:
        decision = "wp10c8s_no_nonlinear_complete_rate_case_authorized"
        next_action = "stop_complete_rate_healing_campaign"
    elif single_interface_route_rejected:
        decision = (
            "wp10c8s_single_interface4_route_rejected_by_independent_"
            "nonhealing_complete_rate_mode"
        )
        next_action = (
            "extend_and_confirm_the_binding_inner_mode_before_selecting_"
            "localized_multistate_or_staggered_coarse_architecture"
        )
    elif not all_healing_complete:
        decision = (
            "wp10c8s_static_localization_complete_healing_cases_authorized"
        )
        next_action = (
            "run_synchronized_n64_natural_healing_for_pending_cases"
        )
    elif inconclusive_cases:
        decision = "wp10c8s_healing_numerically_inconclusive"
        next_action = "repair_only_the_failed_temporal_or_physical_gate"
    elif len(persistent_cases) >= 2:
        decision = (
            "wp10c8s_multiple_persistent_complete_rate_modes_detected_n64"
        )
        next_action = (
            "confirm_architecture_controlling_persistent_modes_at_n128_then_"
            "design_conservative_staggered_coarse_model"
        )
    elif len(persistent_cases) == 1:
        decision = "wp10c8s_one_persistent_complete_rate_mode_detected_n64"
        next_action = (
            "confirm_the_single_persistent_mode_at_n128_then_test_one_"
            "measured_physical_coordinate"
        )
    else:
        decision = "wp10c8s_all_tested_complete_rate_modes_heal_at_n64"
        next_action = (
            "retain_healed_q34_closure_hypothesis_and_run_worst_case_"
            "post_healing_fiber_search"
        )

    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    arrays_path = args.arrays if args.arrays.is_absolute() else ROOT / args.arrays
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **output_arrays)
    source_paths = (
        ROOT / THIS_RUNNER,
        ROOT / "scripts/run_causal_interface_state_sufficiency_wp10c8r.py",
        ROOT / "scripts/run_causal_extended_healing_wp10c8q.py",
        ROOT / "scripts/run_causal_nonlinear_fiber_audit_wp10c8o.py",
    )
    output = {
        "schema_version": SCHEMA_VERSION,
        "base_commit": BASE_COMMIT,
        "work_package": WORK_PACKAGE,
        "scope": {
            "production_physics_changed": False,
            "production_flux_changed": False,
            "production_descriptor_changed": False,
            "production_time_integrator_changed": False,
            "moment_ladder_changed": False,
            "new_exact_nonlinear_pairs_constructed": True,
            "natural_healing_run": bool(healing),
            "natural_healing_matrix_complete": all_healing_complete,
            "n128_natural_healing_confirmation_run": False,
            "linear_healing_preflight_binding": False,
        },
        "authorization": {
            "wp10c8r_json_path": _relative(PARENT_JSON),
            "wp10c8r_json_sha256": _sha256(PARENT_JSON),
            "wp10c8r_arrays_path": _relative(PARENT_ARRAYS),
            "wp10c8r_arrays_sha256": _sha256(PARENT_ARRAYS),
            "wp10c8q_rate_json_path": _relative(RATE_PARENT_JSON),
            "wp10c8q_rate_json_sha256": _sha256(RATE_PARENT_JSON),
            "wp10c8q_rate_arrays_path": _relative(RATE_PARENT_ARRAYS),
            "wp10c8q_rate_arrays_sha256": _sha256(RATE_PARENT_ARRAYS),
        },
        "gates": {
            "minimum_nonlinear_slow_rate_half_spread": NONLINEAR_RATE_GATE,
            "minimum_cross_mesh_tangent_response_cosine": (
                TANGENT_CROSS_MESH_COSINE_GATE
            ),
            "maximum_cross_mesh_tangent_amplitude_ratio_defect": (
                TANGENT_CROSS_MESH_AMPLITUDE_RATIO_GATE
            ),
            "one_shell_localization_fraction": LOCALIZATION_FRACTION_GATE,
            "linear_rapid_healing_final_ratio": (
                LINEAR_RAPID_HEALING_RATIO
            ),
        },
        "case_specs": CASE_SPECS,
        "cases": rows,
        "cross_mesh_tangent_confirmation": tangent_confirmations,
        "localization": localizations,
        "frozen_tangent_healing_preflight": linear_preflights,
        "authorized_nonlinear_healing_cases": authorized,
        "persistent_frozen_tangent_cases": persistent_preflight,
        "rapidly_healing_frozen_tangent_cases": rapidly_healing_preflight,
        "natural_healing": healing,
        "pending_natural_healing_cases": pending_healing,
        "natural_healing_matrix_complete": all_healing_complete,
        "fail_fast_single_interface_route_rejected": (
            single_interface_route_rejected
        ),
        "naturally_healed_cases": healed_cases,
        "naturally_persistent_cases": persistent_cases,
        "numerically_inconclusive_healing_cases": inconclusive_cases,
        "state_provenance": state_provenance,
        "decision": decision,
        "next_action": next_action,
        "interpretation": (
            "The frozen-tangent healing calculation is a cost-selection "
            "preflight only.  Architecture decisions require synchronized "
            "finite-amplitude N64 trajectories and selected N128 confirmation."
        ),
        "source_hashes": {
            _relative(path): _sha256(path) for path in source_paths
        },
        "artifacts": {
            "arrays_path": _relative(arrays_path),
            "arrays_sha256": _sha256(arrays_path),
        },
    }
    output_path.write_text(
        json.dumps(_plain(output), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "work_package": WORK_PACKAGE,
                "decision": decision,
                "authorized_cases": authorized,
                "persistent_frozen_tangent_cases": persistent_preflight,
                "rapidly_healing_frozen_tangent_cases": (
                    rapidly_healing_preflight
                ),
                "pending_natural_healing_cases": pending_healing,
                "naturally_healed_cases": healed_cases,
                "naturally_persistent_cases": persistent_cases,
                "fail_fast_single_interface_route_rejected": (
                    single_interface_route_rejected
                ),
                "output": _relative(output_path),
                "arrays": _relative(arrays_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
