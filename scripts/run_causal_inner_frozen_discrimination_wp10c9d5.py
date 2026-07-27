"""Run the WP10c9d5 frozen-linear radial-candidate discrimination.

The package keeps the certified temporal descriptor and base-rate storage
derivative fixed and replaces only the stationary radial Jacobian.  Production
and candidate histories use identical anchors, perturbations, grids, output
times, and physical normalization.  The binding first stage is the unchanged
common-mode physical-export ladder.  Pure-family and held-out mixed packets
are authorized only when that stage contracts.
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
from scipy.sparse.linalg import expm_multiply

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_characteristic_phase_audit_wp10c9a as wp10c9a
import run_causal_inner_common_mode_audit_wp10c8y as wp10c8y
import run_causal_inner_embedded_patch_preflight_wp10c8z as wp10c8z
import run_causal_inner_family_transfer_audit_wp10c9c0c as wp10c9c0c
import run_causal_inner_micro_export_preflight_wp10c9d0 as wp10c9d0
import run_causal_inner_phase_spatial_preflight_wp10c8v as wp10c8v

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES,
    causal_five_field_radial_candidate_face_flux,
    causal_five_field_radial_candidate_ledger,
    causal_five_field_radial_candidate_lower_source_totals,
    causal_five_field_radial_frozen_candidate,
    causal_five_field_reduced_stationary_residual,
)


ANALYZED_BASE_COMMIT = "42dd7f16446eaac33f4e2f5c0e90d6e81866733b"
WORK_PACKAGE = "WP10c9d5"
SCHEMA_VERSION = 1
THIS_RUNNER = (
    "scripts/run_causal_inner_frozen_discrimination_wp10c9d5.py"
)
D4B_CANONICAL = (
    ROOT
    / "results/canonical/"
    "causal_inner_radial_fluctuation_wp10c9d4b/summary.json"
)
D0_OUTPUT = wp10c9d0.DEFAULT_OUTPUT
D0_ARRAYS = wp10c9d0.DEFAULT_ARRAYS
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/tables/"
    "causal_inner_frozen_discrimination_wp10c9d5.json"
)
DEFAULT_ARRAYS = (
    ROOT
    / "outputs/tables/"
    "causal_inner_frozen_discrimination_wp10c9d5_arrays.npz"
)
DEFAULT_CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_frozen_discrimination_wp10c9d5"
)
CACHE_DIRECTORY = (
    ROOT
    / "outputs/checkpoints/"
    "causal_inner_frozen_discrimination_wp10c9d5"
)

TARGET_SECONDS = 0.125
TIME_SAMPLES = 201
SAMPLE_STRIDE = 2
GENERATOR_RELATIVE_STEP = 4.0e-5
GENERATOR_JVP_STEPS = (2.0e-5, 4.0e-5, 8.0e-5)
PATH_QUADRATURE_ORDER = 6
DIRECTIONAL_STEP = 2.0e-4
MINIMUM_POSITIVE_EXPORT_ORDER = 0.0
PREFERRED_EXPORT_ORDER = 0.75
MAXIMUM_FINE_NORMALIZED_EXPORT_DIFFERENCE = 0.05
MINIMUM_FINE_SIGNED_COSINE = 0.90
MAXIMUM_DESCRIPTOR_SOLVE_DEFECT = 1.0e-10
MAXIMUM_MASS_OFF_PATTERN_ENTRY = 2.0e-2
MAXIMUM_EXPORT_MAP_PARITY_DEFECT = 5.0e-6
MAXIMUM_GENERATOR_JVP_DEFECT = 2.0e-5
MINIMUM_MATERIAL_IMPROVEMENT = 0.25
PACKET_ORDER_GATE = 0.75

PATCH_LABELS = (
    "N128_exterior_N128_inner_c48",
    "N128_exterior_N256_inner_c48",
    "N128_exterior_N512_inner_c48",
)
UNIFORM_LABELS = ("uniform_N64", "uniform_N128", "uniform_N256")
HELD_OUT_COEFFICIENTS = {
    "heldout_shear_acoustic": {
        "inward_shear": 1.0,
        "outward_acoustic": 0.35,
    },
    "heldout_material_shear": {
        "material": 1.0,
        "outward_shear": -0.40,
    },
    "heldout_five_family": {
        "inward_acoustic": 0.30,
        "inward_shear": -0.20,
        "material": 0.40,
        "outward_shear": 0.25,
        "outward_acoustic": 0.50,
    },
}

IMPLEMENTATION_SOURCES = (
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_radial_fluctuation.py",
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_radial_frozen.py",
    "src/imri_qpe/layer3_minidisk_1d/__init__.py",
    THIS_RUNNER,
    "tests/test_causal_inner_radial_frozen.py",
    "tests/test_causal_inner_frozen_discrimination_wp10c9d5.py",
)


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
    return str(path.relative_to(ROOT))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
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


def _generator_paths(label: str) -> tuple[Path, Path]:
    return (
        CACHE_DIRECTORY / f"{label}.json",
        CACHE_DIRECTORY / f"{label}_arrays.npz",
    )


def _build_or_load_generator(
    label: str,
    *,
    context,
    base_primitives: np.ndarray,
    production_generator: np.ndarray,
    primitive_scales: np.ndarray,
    force: bool,
) -> tuple[dict, dict[str, np.ndarray]]:
    json_path, arrays_path = _generator_paths(label)
    contract = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "label": label,
        "grid_edges_sha256": _array_sha256(context.grid.edges),
        "base_primitives_sha256": _array_sha256(base_primitives),
        "production_generator_sha256": _array_sha256(production_generator),
        "primitive_scales_sha256": _array_sha256(primitive_scales),
        "finite_difference_step": GENERATOR_RELATIVE_STEP,
        "path_quadrature_order": PATH_QUADRATURE_ORDER,
    }
    if not force and json_path.exists() and arrays_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if (
            all(payload.get(key) == value for key, value in contract.items())
            and payload.get("arrays_sha256") == _sha256(arrays_path)
        ):
            with np.load(arrays_path, allow_pickle=False) as source:
                return payload, {
                    name: np.asarray(source[name]) for name in source.files
                }

    print(f"WP10c9d5: building {label} candidate generator", flush=True)
    started = time.perf_counter()
    frozen = causal_five_field_radial_frozen_candidate(
        context,
        base_primitives,
        production_generator,
        primitive_column_scales=primitive_scales,
        finite_difference_step=GENERATOR_RELATIVE_STEP,
        path_quadrature_order=PATH_QUADRATURE_ORDER,
    )
    arrays = {
        "candidate_generator": frozen.candidate_scaled_generator_per_s,
        "production_generator": frozen.production_scaled_generator_per_s,
        "descriptor": frozen.descriptor_reduced_scaled_matrix,
        "stationary_delta": frozen.stationary_delta_scaled_jacobian,
        "primitive_column_scales": frozen.primitive_column_scales,
        "conservation_row_scales": frozen.conservation_row_scales,
    }
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **arrays)
    relative_correction = float(
        np.linalg.norm(
            frozen.candidate_scaled_generator_per_s
            - frozen.production_scaled_generator_per_s
        )
        / max(
            float(
                np.linalg.norm(
                    frozen.production_scaled_generator_per_s
                )
            ),
            np.finfo(float).tiny,
        )
    )
    payload = {
        **contract,
        "arrays_path": _relative(arrays_path),
        "arrays_sha256": _sha256(arrays_path),
        "n_cells": int(context.grid.centers.size),
        "color_count": frozen.color_count,
        "maximum_descriptor_solve_relative_defect": (
            frozen.maximum_descriptor_solve_relative_defect
        ),
        "maximum_mass_off_pattern_relative_entry": (
            frozen.maximum_mass_off_pattern_relative_entry
        ),
        "relative_generator_correction_norm": relative_correction,
        "same_temporal_descriptor": frozen.same_temporal_descriptor,
        "same_base_rate_storage_derivative": (
            frozen.same_base_rate_storage_derivative
        ),
        "wall_seconds": time.perf_counter() - started,
    }
    payload["passed"] = bool(
        payload["maximum_descriptor_solve_relative_defect"]
        <= MAXIMUM_DESCRIPTOR_SOLVE_DEFECT
        and payload["maximum_mass_off_pattern_relative_entry"]
        <= MAXIMUM_MASS_OFF_PATTERN_ENTRY
        and payload["same_temporal_descriptor"]
        and payload["same_base_rate_storage_derivative"]
    )
    _write_json(json_path, payload)
    return payload, arrays


def _common_configurations(force: bool):
    c8y = wp10c9d0._load_npz(wp10c9d0.WP10C8Y_ARRAYS)
    c8z = wp10c9d0._load_npz(wp10c9d0.WP10C8Z_ARRAYS)
    contexts, profiles = wp10c9c0c._common_contexts()
    production_operators = wp10c8y._load_family_operators()["production"]
    result = {}
    for mesh in wp10c8y.MESHES:
        label = f"uniform_N{mesh}"
        context = contexts[mesh]
        operator = production_operators[mesh]
        amplitudes = np.asarray(
            c8y[f"N{mesh}_common_amplitudes"],
            dtype=float,
        )
        # The cached evolving generator is binding at its own exact anchor.
        # N64/N128 happen to match the reconstructed profile bitwise; the
        # independently prolonged N256 audit anchor does not.  Reusing the
        # generator therefore requires its stored base rather than the later
        # convenience reconstruction.
        base = np.asarray(operator["base_primitives"], dtype=float)
        report, candidate = _build_or_load_generator(
            label,
            context=context,
            base_primitives=base,
            production_generator=np.asarray(
                operator["generator"],
                dtype=float,
            ),
            primitive_scales=np.asarray(
                operator["primitive_column_scales"],
                dtype=float,
            ),
            force=force,
        )
        generator = wp10c8v._similarity_rescale_generator(
            candidate["candidate_generator"],
            candidate["primitive_column_scales"],
            amplitudes,
        )
        result[label] = {
            "label": label,
            "context": context,
            "base_primitives": base,
            "amplitudes": amplitudes,
            "initial": np.asarray(
                c8y[f"production_N{mesh}_state"],
                dtype=float,
            )[0],
            "times": np.asarray(
                c8y[f"production_N{mesh}_times"],
                dtype=float,
            ),
            "generator": generator,
            "candidate_native": candidate,
            "generator_report": report,
            "interface_face": int(context.grid.centers.size),
            "active_cells": int(context.grid.centers.size),
        }

    patch_configurations = wp10c9d0._patch_configurations(c8z)
    patch_times = np.asarray(c8z["times"], dtype=float)
    for label in PATCH_LABELS:
        configuration = patch_configurations[label]
        operator = configuration["operator"]
        amplitudes = np.asarray(configuration["amplitudes"], dtype=float)
        base = np.asarray(configuration["base_primitives"], dtype=float)
        report, candidate = _build_or_load_generator(
            label,
            context=configuration["context"],
            base_primitives=base,
            production_generator=np.asarray(
                operator["generator"],
                dtype=float,
            ),
            primitive_scales=np.asarray(
                operator["primitive_column_scales"],
                dtype=float,
            ),
            force=force,
        )
        generator = wp10c8v._similarity_rescale_generator(
            candidate["candidate_generator"],
            candidate["primitive_column_scales"],
            amplitudes,
        )
        result[label] = {
            **configuration,
            "initial": np.asarray(
                c8z[f"{label}_state_history"],
                dtype=float,
            )[0],
            "times": patch_times,
            "generator": generator,
            "candidate_native": candidate,
            "generator_report": report,
            "interface_face": int(
                configuration["layout"].coupling_face_index
            ),
            "active_cells": int(
                configuration["layout"].coupling_face_index
            ),
        }
    return result


def _propagate(generator: np.ndarray, initial: np.ndarray, times: np.ndarray):
    state = np.asarray(
        expm_multiply(
            generator,
            np.asarray(initial, dtype=float).ravel(),
            start=float(times[0]),
            stop=float(times[-1]),
            num=int(times.size),
            endpoint=True,
        ),
        dtype=float,
    ).reshape(times.size, -1, 5)
    rate = np.asarray(
        [generator @ row.ravel() for row in state],
        dtype=float,
    ).reshape(state.shape)
    midpoint = expm_multiply(
        generator * (0.5 * float(times[-1])),
        np.asarray(initial, dtype=float).ravel(),
    )
    restarted = expm_multiply(
        generator * (0.5 * float(times[-1])),
        midpoint,
    )
    restart = float(
        np.linalg.norm(restarted - state[-1].ravel())
        / max(float(np.linalg.norm(state[-1])), np.finfo(float).tiny)
    )
    return state, rate, restart


def _generator_jvp_audit(configuration: dict) -> dict:
    context = configuration["context"]
    base = np.asarray(configuration["base_primitives"], dtype=float)
    direction = (
        np.asarray(configuration["amplitudes"], dtype=float)
        * np.asarray(configuration["initial"], dtype=float)
    )
    native = configuration["candidate_native"]
    row_scales = np.asarray(
        native["conservation_row_scales"],
        dtype=float,
    )
    scaled_direction = (
        direction.ravel()
        / np.asarray(native["primitive_column_scales"], dtype=float)
    )
    matrix_action = (
        np.asarray(native["stationary_delta"], dtype=float)
        @ scaled_direction
    )

    def scaled_delta(charts: np.ndarray) -> np.ndarray:
        candidate = causal_five_field_radial_candidate_ledger(
            context,
            charts,
            quadrature_order=PATH_QUADRATURE_ORDER,
        ).residual_rows.ravel()
        production = causal_five_field_reduced_stationary_residual(
            np.asarray(charts, dtype=float).ravel(),
            context,
        )
        return (
            np.asarray(candidate, dtype=float)
            - np.asarray(production, dtype=float)
        ) / row_scales

    defects = {}
    direct_actions = {}
    for step in GENERATOR_JVP_STEPS:
        direct = (
            scaled_delta(base + step * direction)
            - scaled_delta(base - step * direction)
        ) / (2.0 * step)
        scale = max(
            float(np.linalg.norm(direct)),
            float(np.linalg.norm(matrix_action)),
            np.finfo(float).tiny,
        )
        defect = float(np.linalg.norm(direct - matrix_action) / scale)
        defects[f"{step:.1e}"] = defect
        direct_actions[f"{step:.1e}"] = direct
    selected = defects[f"{GENERATOR_RELATIVE_STEP:.1e}"]
    endpoint_change = float(
        np.linalg.norm(
            direct_actions[f"{GENERATOR_JVP_STEPS[0]:.1e}"]
            - direct_actions[f"{GENERATOR_JVP_STEPS[-1]:.1e}"]
        )
        / max(
            float(
                np.linalg.norm(
                    direct_actions[
                        f"{GENERATOR_JVP_STEPS[0]:.1e}"
                    ]
                )
            ),
            float(
                np.linalg.norm(
                    direct_actions[
                        f"{GENERATOR_JVP_STEPS[-1]:.1e}"
                    ]
                )
            ),
            np.finfo(float).tiny,
        )
    )
    return {
        "relative_step_defects": defects,
        "selected_step": GENERATOR_RELATIVE_STEP,
        "selected_relative_defect": selected,
        "endpoint_direct_action_change": endpoint_change,
        "passed": bool(selected <= MAXIMUM_GENERATOR_JVP_DEFECT),
    }


def _direct_candidate_observables(
    configuration: dict,
    primitives: np.ndarray,
) -> np.ndarray:
    context = configuration["context"]
    charts = np.asarray(primitives, dtype=float)
    face = int(configuration["interface_face"])
    active = int(configuration["active_cells"])
    ledger = causal_five_field_radial_candidate_ledger(
        context,
        charts,
        quadrature_order=PATH_QUADRATURE_ORDER,
    )
    fluxes = ledger.interfaces.candidate_shared_face_fluxes_over_c
    inner = fluxes[0, wp10c9d0.CONSERVATIVE_FIELDS]
    interface = fluxes[face, wp10c9d0.CONSERVATIVE_FIELDS]
    net = -np.sum(
        ledger.residual_rows[:active, wp10c9d0.CONSERVATIVE_FIELDS],
        axis=0,
    )
    lower = ledger.integrated_lower_source_components_per_ct
    cooling = np.sum(lower["radiative_cooling"][:active], axis=0)
    vertical = np.sum(lower["vertical_work"][:active], axis=0)
    return np.concatenate(
        (
            inner,
            interface,
            net,
            cooling[wp10c9d0.CONSERVATIVE_FIELDS[1:]],
            vertical[wp10c9d0.CONSERVATIVE_FIELDS[1:]],
        )
    )


def _candidate_baseline(configuration: dict) -> np.ndarray:
    return _direct_candidate_observables(
        configuration,
        np.asarray(configuration["base_primitives"], dtype=float),
    )


def _export_map_parity(configuration: dict) -> dict:
    base = np.asarray(configuration["base_primitives"], dtype=float)
    direction = (
        np.asarray(configuration["amplitudes"], dtype=float)
        * np.asarray(configuration["initial"], dtype=float)
    )
    step = DIRECTIONAL_STEP
    accelerated = _candidate_directional_observables(
        configuration,
        direction,
        step=step,
    )
    direct = (
        _direct_candidate_observables(
            configuration,
            base + step * direction,
        )
        - _direct_candidate_observables(
            configuration,
            base - step * direction,
        )
    ) / (2.0 * step)
    baseline = _candidate_baseline(configuration)
    activity = np.maximum(np.abs(accelerated), np.abs(direct))
    significant = activity >= (
        wp10c9d0.MINIMUM_RELATIVE_ACTIVITY
        * np.maximum(np.abs(baseline), 1.0)
    )
    scale = np.maximum(activity, np.finfo(float).tiny)
    defect = (
        float(
            np.max(
                np.abs(accelerated[significant] - direct[significant])
                / scale[significant]
            )
        )
        if np.any(significant)
        else 0.0
    )
    return {
        "maximum_significant_relative_defect": defect,
        "significant_components": [
            wp10c9d0.OBSERVABLE_NAMES[index]
            for index in np.flatnonzero(significant)
        ],
        "passed": bool(defect <= MAXIMUM_EXPORT_MAP_PARITY_DEFECT),
    }


def _candidate_directional_observables(
    configuration: dict,
    physical_direction: np.ndarray,
    *,
    step: float = DIRECTIONAL_STEP,
) -> np.ndarray:
    context = configuration["context"]
    base = np.asarray(configuration["base_primitives"], dtype=float)
    direction = np.asarray(physical_direction, dtype=float)
    face = int(configuration["interface_face"])
    active = int(configuration["active_cells"])
    production = wp10c9d0._directional_observables(
        context,
        base,
        direction,
        interface_face=face,
        active_cells=active,
        step=step,
    )
    delta = float(step) * direction
    production_plus, candidate_plus = (
        causal_five_field_radial_candidate_face_flux(
            context,
            base + delta,
            face,
            quadrature_order=PATH_QUADRATURE_ORDER,
        )
    )
    production_minus, candidate_minus = (
        causal_five_field_radial_candidate_face_flux(
            context,
            base - delta,
            face,
            quadrature_order=PATH_QUADRATURE_ORDER,
        )
    )
    adjustment = (
        (candidate_plus - production_plus)
        - (candidate_minus - production_minus)
    ) / (2.0 * float(step))
    result = np.array(production, copy=True)
    result[3:6] += adjustment[wp10c9d0.CONSERVATIVE_FIELDS]

    native = configuration["candidate_native"]
    scaled_direction = (
        direction.ravel()
        / np.asarray(native["primitive_column_scales"], dtype=float)
    )
    delta_rows = (
        np.asarray(native["conservation_row_scales"], dtype=float)
        * (
            np.asarray(native["stationary_delta"], dtype=float)
            @ scaled_direction
        )
    ).reshape(-1, 5)
    result[6:9] -= np.sum(
        delta_rows[:active, wp10c9d0.CONSERVATIVE_FIELDS],
        axis=0,
    )
    lower_plus = causal_five_field_radial_candidate_lower_source_totals(
        context,
        base + delta,
        active_cells=active,
    )
    lower_minus = causal_five_field_radial_candidate_lower_source_totals(
        context,
        base - delta,
        active_cells=active,
    )
    cooling = (
        lower_plus["radiative_cooling"]
        - lower_minus["radiative_cooling"]
    ) / (2.0 * float(step))
    vertical = (
        lower_plus["vertical_work"]
        - lower_minus["vertical_work"]
    ) / (2.0 * float(step))
    result[9:11] = cooling[wp10c9d0.CONSERVATIVE_FIELDS[1:]]
    result[11:13] = vertical[wp10c9d0.CONSERVATIVE_FIELDS[1:]]
    return result


def _signal_history(configuration: dict, state_history: np.ndarray):
    indices = np.arange(
        0,
        int(state_history.shape[0]),
        SAMPLE_STRIDE,
        dtype=int,
    )
    if indices[-1] != state_history.shape[0] - 1:
        indices = np.append(indices, state_history.shape[0] - 1)
    times = np.asarray(configuration["times"], dtype=float)[indices]
    baseline = _candidate_baseline(configuration)
    signals = []
    for output, index in enumerate(indices):
        if output % 20 == 0 or output + 1 == indices.size:
            print(
                f"WP10c9d5: {configuration['label']} export "
                f"{output + 1}/{indices.size}",
                flush=True,
            )
        physical = (
            np.asarray(configuration["amplitudes"], dtype=float)
            * np.asarray(state_history[index], dtype=float)
        )
        signals.append(
            _candidate_directional_observables(
                configuration,
                physical,
            )
        )
    signal_array = np.asarray(signals, dtype=float)
    cumulative = wp10c9d0._cumulative_trapezoid(times, signal_array)
    return {
        "indices": indices,
        "times": times,
        "baseline": baseline,
        "signals": signal_array,
        "cumulative": cumulative,
    }


def _component_orders(
    labels: tuple[str, str, str],
    histories: dict[str, np.ndarray],
    baselines: dict[str, np.ndarray],
) -> dict:
    values = np.asarray([histories[label] for label in labels], dtype=float)
    response_scale = np.max(np.abs(values), axis=(0, 1))
    baseline_scale = np.max(
        np.abs(np.asarray([baselines[label] for label in labels])),
        axis=0,
    )
    significant = response_scale >= (
        wp10c9d0.MINIMUM_RELATIVE_ACTIVITY
        * np.maximum(baseline_scale, 1.0)
    )
    scale = np.maximum(response_scale, np.finfo(float).tiny)
    first = np.sqrt(
        np.mean(
            (
                (histories[labels[1]] - histories[labels[0]])
                / scale
            )
            ** 2,
            axis=0,
        )
    )
    second = np.sqrt(
        np.mean(
            (
                (histories[labels[2]] - histories[labels[1]])
                / scale
            )
            ** 2,
            axis=0,
        )
    )
    orders = np.full(first.shape, np.nan, dtype=float)
    valid = significant & (first > 0.0) & (second > 0.0)
    orders[valid] = np.log2(first[valid] / second[valid])
    active_orders = orders[valid]
    return {
        "significant_components": [
            wp10c9d0.OBSERVABLE_NAMES[index]
            for index in np.flatnonzero(significant)
        ],
        "orders": {
            wp10c9d0.OBSERVABLE_NAMES[index]: orders[index]
            for index in np.flatnonzero(significant)
        },
        "minimum_order": (
            float(np.min(active_orders)) if active_orders.size else None
        ),
        "all_positive": bool(
            active_orders.size and np.all(active_orders > 0.0)
        ),
        "all_preferred": bool(
            active_orders.size
            and np.all(active_orders >= PREFERRED_EXPORT_ORDER)
        ),
    }


def _export_ladder(
    labels: tuple[str, str, str],
    signals: dict[str, dict[str, np.ndarray]],
) -> dict:
    histories = {
        label: signals[label]["cumulative"] for label in labels
    }
    baselines = {
        label: np.zeros_like(signals[label]["baseline"]) for label in labels
    }
    aggregate = wp10c9d0._ladder_metrics(
        labels,
        histories,
        baselines,
        indices=wp10c9d0.GROUPS["exported"],
    )
    components = _component_orders(labels, histories, baselines)
    fine = aggregate.get("medium_fine", {})
    fine_difference = fine.get("maximum_component_difference")
    fine_cosine = fine.get("minimum_signed_cosine")
    positive = bool(
        components["all_positive"]
        and fine_difference is not None
        and fine_difference <= MAXIMUM_FINE_NORMALIZED_EXPORT_DIFFERENCE
        and fine_cosine is not None
        and fine_cosine >= MINIMUM_FINE_SIGNED_COSINE
    )
    preferred = bool(positive and components["all_preferred"])
    return {
        "aggregate": aggregate,
        "component_contraction": components,
        "minimum_positive_gate_passed": positive,
        "preferred_order_gate_passed": preferred,
    }


def _packet_stage(configurations: dict, arrays: dict[str, np.ndarray]):
    """Run pure/held-out embedded packets after the export stop gate passes."""

    ratios = dict(zip((1, 2, 4), PATCH_LABELS, strict=True))
    initial_by_family = {}
    reports = {}
    for family in CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES:
        initial_by_family[family] = {}
        histories = {}
        for ratio, label in ratios.items():
            configuration = configurations[label]
            packet, bases, projection = wp10c9a._project_packet(
                configuration,
                family,
            )
            state, rate, restart = _propagate(
                configuration["generator"],
                packet,
                np.asarray(configuration["times"], dtype=float),
            )
            initial_by_family[family][ratio] = packet
            histories[ratio] = {
                "times": configuration["times"],
                "state": state,
                "rate": rate,
                "stress_rate_signal": np.zeros(state.shape[0]),
            }
            arrays[f"{family}_ratio{ratio}_initial"] = packet
            arrays[f"{family}_ratio{ratio}_final_state"] = state[-1]
            arrays[f"{family}_ratio{ratio}_final_rate"] = rate[-1]
            reports.setdefault(family, {}).setdefault("projection", {})[
                f"ratio_{ratio}"
            ] = projection
            reports[family].setdefault("restart_defects", {})[
                f"ratio_{ratio}"
            ] = restart
        restricted = {
            ratio: wp10c8z._restrict_history(
                histories[ratio],
                configurations[ratios[ratio]],
            )
            for ratio in ratios
        }
        parent = wp10c8z._parent_data()
        active_outer = configurations[PATCH_LABELS[0]][
            "active_outer_rg"
        ]
        first = wp10c8z._history_metrics(
            restricted[1],
            restricted[2],
            parent["parent_grid"],
            lower_rg=None,
            upper_rg=active_outer,
        )
        second = wp10c8z._history_metrics(
            restricted[2],
            restricted[4],
            parent["parent_grid"],
            lower_rg=None,
            upper_rg=active_outer,
        )
        state_order = wp10c9a._observed_order(
            first["state"]["maximum_relative_l2_difference"],
            second["state"]["maximum_relative_l2_difference"],
        )
        rate_order = wp10c9a._observed_order(
            first["rate"]["maximum_relative_l2_difference"],
            second["rate"]["maximum_relative_l2_difference"],
        )
        reports[family].update(
            {
                "state_order": state_order,
                "rate_order": rate_order,
                "fine_state_cosine": second["state"][
                    "minimum_signed_cosine"
                ],
                "fine_rate_cosine": second["rate"][
                    "minimum_signed_cosine"
                ],
                "passed": bool(
                    state_order is not None
                    and state_order >= PACKET_ORDER_GATE
                    and rate_order is not None
                    and rate_order >= PACKET_ORDER_GATE
                    and second["state"]["minimum_signed_cosine"]
                    >= MINIMUM_FINE_SIGNED_COSINE
                    and second["rate"]["minimum_signed_cosine"]
                    >= MINIMUM_FINE_SIGNED_COSINE
                ),
            }
        )

    held_out = {}
    for name, coefficients in HELD_OUT_COEFFICIENTS.items():
        histories = {}
        for ratio, label in ratios.items():
            initial = np.zeros_like(
                next(iter(initial_by_family.values()))[ratio]
            )
            for family, coefficient in coefficients.items():
                initial += (
                    float(coefficient)
                    * initial_by_family[family][ratio]
                )
            weights = np.asarray(
                configurations[label]["context"].grid.cell_measures,
                dtype=float,
            )
            norm = float(
                np.sqrt(
                    np.sum(weights[:, None] * initial**2)
                    / (5.0 * np.sum(weights))
                )
            )
            initial /= max(norm, np.finfo(float).tiny)
            state, rate, restart = _propagate(
                configurations[label]["generator"],
                initial,
                np.asarray(configurations[label]["times"], dtype=float),
            )
            histories[ratio] = {
                "times": configurations[label]["times"],
                "state": state,
                "rate": rate,
                "stress_rate_signal": np.zeros(state.shape[0]),
            }
            arrays[f"{name}_ratio{ratio}_initial"] = initial
            arrays[f"{name}_ratio{ratio}_final_state"] = state[-1]
            arrays[f"{name}_ratio{ratio}_final_rate"] = rate[-1]
            held_out.setdefault(name, {}).setdefault(
                "restart_defects",
                {},
            )[f"ratio_{ratio}"] = restart
        restricted = {
            ratio: wp10c8z._restrict_history(
                histories[ratio],
                configurations[ratios[ratio]],
            )
            for ratio in ratios
        }
        parent = wp10c8z._parent_data()
        active_outer = configurations[PATCH_LABELS[0]][
            "active_outer_rg"
        ]
        first = wp10c8z._history_metrics(
            restricted[1],
            restricted[2],
            parent["parent_grid"],
            lower_rg=None,
            upper_rg=active_outer,
        )
        second = wp10c8z._history_metrics(
            restricted[2],
            restricted[4],
            parent["parent_grid"],
            lower_rg=None,
            upper_rg=active_outer,
        )
        state_order = wp10c9a._observed_order(
            first["state"]["maximum_relative_l2_difference"],
            second["state"]["maximum_relative_l2_difference"],
        )
        rate_order = wp10c9a._observed_order(
            first["rate"]["maximum_relative_l2_difference"],
            second["rate"]["maximum_relative_l2_difference"],
        )
        held_out[name].update(
            {
                "coefficients": coefficients,
                "state_order": state_order,
                "rate_order": rate_order,
                "fine_state_cosine": second["state"][
                    "minimum_signed_cosine"
                ],
                "fine_rate_cosine": second["rate"][
                    "minimum_signed_cosine"
                ],
                "passed": bool(
                    state_order is not None
                    and state_order >= PACKET_ORDER_GATE
                    and rate_order is not None
                    and rate_order >= PACKET_ORDER_GATE
                    and second["state"]["minimum_signed_cosine"]
                    >= MINIMUM_FINE_SIGNED_COSINE
                    and second["rate"]["minimum_signed_cosine"]
                    >= MINIMUM_FINE_SIGNED_COSINE
                ),
            }
        )
    passed = bool(
        all(item["passed"] for item in reports.values())
        and all(item["passed"] for item in held_out.values())
    )
    return {
        "executed": True,
        "pure_families": reports,
        "held_out_mixed_packets": held_out,
        "passed": passed,
    }


def run(*, force: bool = False):
    started = time.perf_counter()
    required = (
        D4B_CANONICAL,
        D0_OUTPUT,
        D0_ARRAYS,
        wp10c9d0.WP10C8Y_ARRAYS,
        wp10c9d0.WP10C8Z_OUTPUT,
        wp10c9d0.WP10C8Z_ARRAYS,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "WP10c9d5 requires prior evidence: " + ", ".join(missing)
        )
    d4b = json.loads(D4B_CANONICAL.read_text(encoding="utf-8"))
    if not d4b["wp10c9d5_frozen_linear_discrimination_authorized"]:
        raise RuntimeError("WP10c9d4b did not authorize WP10c9d5")
    d0 = json.loads(D0_OUTPUT.read_text(encoding="utf-8"))
    configurations = _common_configurations(force)
    arrays = {}
    generator_reports = {
        label: configuration["generator_report"]
        for label, configuration in configurations.items()
    }
    export_map_parity = {}
    generator_jvp_audits = {}
    for label, configuration in configurations.items():
        print(f"WP10c9d5: auditing {label} generator JVP", flush=True)
        generator_jvp_audits[label] = _generator_jvp_audit(configuration)
        print(f"WP10c9d5: auditing {label} export-map parity", flush=True)
        export_map_parity[label] = _export_map_parity(configuration)
    common = {}
    for label, configuration in configurations.items():
        print(f"WP10c9d5: propagating {label} common mode", flush=True)
        state, rate, restart = _propagate(
            configuration["generator"],
            configuration["initial"],
            np.asarray(configuration["times"], dtype=float),
        )
        signal = _signal_history(configuration, state)
        common[label] = {
            "restart_relative_defect": restart,
            **signal,
        }
        arrays[f"{label}_final_state"] = state[-1]
        arrays[f"{label}_final_rate"] = rate[-1]
        arrays[f"{label}_signal_times"] = signal["times"]
        arrays[f"{label}_signals"] = signal["signals"]
        arrays[f"{label}_cumulative"] = signal["cumulative"]

    uniform_ladder = _export_ladder(UNIFORM_LABELS, common)
    patch_ladder = _export_ladder(PATCH_LABELS, common)
    previous_patch = d0["ladders"]["embedded_patch"]["exported"][
        "cumulative"
    ]
    old_rms = float(previous_patch["observed_order_rms"])
    old_maximum = float(previous_patch["observed_order_maximum"])
    new_rms = patch_ladder["aggregate"]["observed_order_rms"]
    new_maximum = patch_ladder["aggregate"]["observed_order_maximum"]
    materially_improved = bool(
        new_rms is not None
        and new_rms >= old_rms + MINIMUM_MATERIAL_IMPROVEMENT
        and new_maximum is not None
        and new_maximum >= old_maximum + MINIMUM_MATERIAL_IMPROVEMENT
    )
    common_export_passed = bool(
        all(report["passed"] for report in generator_reports.values())
        and all(report["passed"] for report in generator_jvp_audits.values())
        and all(report["passed"] for report in export_map_parity.values())
        and uniform_ladder["preferred_order_gate_passed"]
        and patch_ladder["preferred_order_gate_passed"]
        and materially_improved
    )
    if common_export_passed:
        packet_stage = _packet_stage(configurations, arrays)
    else:
        packet_stage = {
            "executed": False,
            "passed": False,
            "reason": (
                "binding common-mode physical-export ladder did not pass; "
                "pure-family and held-out histories remain blocked by the "
                "predeclared stop gate"
            ),
            "predeclared_held_out_coefficients": HELD_OUT_COEFFICIENTS,
        }
    candidate_passed = bool(common_export_passed and packet_stage["passed"])
    if candidate_passed:
        classification = (
            "radial_candidate_frozen_linear_exports_and_packets_pass_"
            "nonlinear_preflight_authorized"
        )
        next_action = (
            "proceed to WP10c9d6 analytic/AD/QZ nonlinear residual "
            "preflight without changing production defaults"
        )
    else:
        classification = (
            "radial_candidate_frozen_linear_discrimination_failed_"
            "candidate_rejected"
        )
        next_action = (
            "retain production, reject this complete-fluctuation candidate, "
            "and localize the failed radial export before any nonlinear or "
            "fixed-Q work"
        )
    source_hashes, source_manifest = _source_manifest()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "runner": THIS_RUNNER,
        "classification": classification,
        "candidate_frozen_linear_gate_passed": candidate_passed,
        "common_export_gate_passed": common_export_passed,
        "packet_stage": packet_stage,
        "wp10c9d6_nonlinear_preflight_authorized": candidate_passed,
        "production_operator_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "next_action": next_action,
        "frozen_discrimination_contract": {
            "same_temporal_descriptor": True,
            "same_base_rate_storage_derivative": True,
            "stationary_radial_jacobian_only_changed": True,
            "candidate_equals_production_required": False,
            "each_cached_generator_uses_its_exact_stored_anchor": True,
            "uniform_n256_convenience_profile_used_as_anchor": False,
            "production_default_changed": False,
        },
        "gates": {
            "minimum_positive_export_order": (
                MINIMUM_POSITIVE_EXPORT_ORDER
            ),
            "preferred_export_order": PREFERRED_EXPORT_ORDER,
            "maximum_fine_normalized_export_difference": (
                MAXIMUM_FINE_NORMALIZED_EXPORT_DIFFERENCE
            ),
            "minimum_fine_signed_cosine": MINIMUM_FINE_SIGNED_COSINE,
            "maximum_descriptor_solve_defect": (
                MAXIMUM_DESCRIPTOR_SOLVE_DEFECT
            ),
            "maximum_mass_off_pattern_entry": (
                MAXIMUM_MASS_OFF_PATTERN_ENTRY
            ),
            "maximum_export_map_parity_defect": (
                MAXIMUM_EXPORT_MAP_PARITY_DEFECT
            ),
            "maximum_generator_jvp_defect": (
                MAXIMUM_GENERATOR_JVP_DEFECT
            ),
            "minimum_material_improvement": MINIMUM_MATERIAL_IMPROVEMENT,
            "packet_order_gate": PACKET_ORDER_GATE,
        },
        "generator_reports": generator_reports,
        "generator_jvp_audits": generator_jvp_audits,
        "export_map_parity": export_map_parity,
        "common_mode_exports": {
            "uniform": uniform_ladder,
            "embedded_patch": patch_ladder,
            "production_embedded_reference": {
                "observed_order_rms": old_rms,
                "observed_order_maximum": old_maximum,
            },
            "candidate_materially_improved": materially_improved,
        },
        "observable_names": wp10c9d0.OBSERVABLE_NAMES,
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


def _write_canonical(directory: Path, payload: dict, arrays: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    decisive = {
        name: values
        for name, values in arrays.items()
        if (
            name.endswith("_signals")
            or name.endswith("_cumulative")
            or name.endswith("_final_state")
            or name.endswith("_final_rate")
        )
    }
    arrays_path = directory / "decisive_arrays.npz"
    np.savez_compressed(arrays_path, **decisive)
    summary = dict(payload)
    summary.pop("runtime_seconds", None)
    summary["decisive_arrays_sha256"] = _sha256(arrays_path)
    summary["decisive_array_hashes"] = {
        name: _array_sha256(values)
        for name, values in sorted(decisive.items())
    }
    _write_json(directory / "summary.json", summary)
    _write_json(
        directory / "config.json",
        {
            "target_seconds": TARGET_SECONDS,
            "time_samples": TIME_SAMPLES,
            "sample_stride": SAMPLE_STRIDE,
            "generator_relative_step": GENERATOR_RELATIVE_STEP,
            "generator_jvp_steps": GENERATOR_JVP_STEPS,
            "path_quadrature_order": PATH_QUADRATURE_ORDER,
            "directional_step": DIRECTIONAL_STEP,
            "held_out_coefficients": HELD_OUT_COEFFICIENTS,
            "gates": payload["gates"],
        },
    )
    _write_json(
        directory / "provenance.json",
        {
            "work_package": WORK_PACKAGE,
            "scientific_status": (
                "CERTIFIED"
                if payload["candidate_frozen_linear_gate_passed"]
                else "REJECTED"
            ),
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "source_parent_commit": ANALYZED_BASE_COMMIT,
            "implementation_source_manifest_sha256": payload[
                "implementation_source_manifest_sha256"
            ],
            "implementation_source_hashes": payload[
                "implementation_source_hashes"
            ],
            "input_hashes": payload["input_hashes"],
            "generation_command": (
                "PYTHONPATH=src python3 "
                "scripts/run_causal_inner_frozen_discrimination_wp10c9d5.py"
            ),
            "establishes": (
                "The frozen-background production/candidate A/B comparison, "
                "common-mode physical-export refinement, and conditional "
                "packet gates stated by the summary."
            ),
            "does_not_establish": (
                "A nonlinear path solver, production promotion, fixed-Q "
                "averaging, or reduced slow evolution."
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
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--arrays", type=Path, default=DEFAULT_ARRAYS)
    parser.add_argument(
        "--canonical-directory",
        type=Path,
        default=DEFAULT_CANONICAL_DIRECTORY,
    )
    arguments = parser.parse_args()
    payload, arrays = run(force=arguments.force)
    arguments.arrays.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.arrays, **arrays)
    payload["arrays_path"] = _relative(arguments.arrays)
    payload["arrays_sha256"] = _sha256(arguments.arrays)
    _write_json(arguments.output, payload)
    _write_canonical(arguments.canonical_directory, payload, arrays)
    print(
        json.dumps(
            {
                "classification": payload["classification"],
                "candidate_frozen_linear_gate_passed": payload[
                    "candidate_frozen_linear_gate_passed"
                ],
                "common_export_gate_passed": payload[
                    "common_export_gate_passed"
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
