#!/usr/bin/env python3
"""Audit the four-level uniform monolithic export-error direction.

WP10c9d6c rejected the N64/N128/N256 physical-export ladder because the
already-small inner M/J/E refinement errors did not share one direction.
WP10c9d6c1 showed that an N128-defined common background did not cure that
rotation and selected no first-cell intervention.

This package changes no operator.  It adds one deterministic
N512-equivalent (192 active-cell) uniform level, keeps the N128-defined
common background and the fixed physical export scales, and asks whether
the N128/N256/N512 triplet reaches the unchanged asymptotic-direction gate.
Two independent continuations of the common perturbation to N512 are
propagated so the conclusion cannot be selected by one interpolation.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
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


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_causal_inner_monolithic_anchor_audit_wp10c9d6c1 as wp10c9d6c1
import run_causal_inner_monolithic_uniform_exports_wp10c9d6c as wp10c9d6c

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    KerrSchildCellSourceRates,
    causal_five_field_dae_scaling,
    causal_five_field_monolithic_frozen_tangent,
    causal_five_field_reconstruct_face_charts,
    causal_five_field_state_from_primitives,
    evaluate_causal_five_field_dae,
    evaluate_causal_five_field_monolithic_backward_euler,
    make_kerr_schild_column_grid,
    pack_causal_five_field_state,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c2"
ANALYZED_BASE_COMMIT = "c28dd65708ee817fd23d0c619dbb0afd5f991178"
ANALYZED_BASE_PARENT = "db6625f397141083d359505d84d072d4381ce92a"
ANALYZED_BASE_TREE = "84ea835a99157616030ee6504864578f85acd0f5"
THIS_RUNNER = (
    "scripts/run_causal_inner_monolithic_four_level_wp10c9d6c2.py"
)

MESHES = (64, 128, 256, 512)
LABELS = tuple(f"uniform_N{mesh}" for mesh in MESHES)
PARENT_LABELS = LABELS[:3]
FINE_LABEL = LABELS[-1]
FINE_ACTIVE_CELLS = 192
TRIPLETS = {
    "legacy_N64_N128_N256": LABELS[:3],
    "fine_N128_N256_N512": LABELS[1:],
}
PRIMARY_STRIDE = 2
STRIDE_AUDITS = (1, 2, 4)
PRIMARY_CONTINUATION = "from_N256"
SECONDARY_CONTINUATION = "from_N128"
CONTINUATIONS = (PRIMARY_CONTINUATION, SECONDARY_CONTINUATION)

MAXIMUM_PARENT_REPLAY_DEFECT = 1.0e-12
MAXIMUM_GRID_NESTING_DEFECT = 1.0e-14
MAXIMUM_REFERENCE_BACKGROUND_DEFECT = 1.0e-14
MAXIMUM_RECONSTRUCTION_FACTOR_CHANGE = 0.0
MAXIMUM_STRIDE_DEFECT = wp10c9d6c1.MAXIMUM_STRIDE_DEFECT
MAXIMUM_CONTINUATION_EXPORT_DIFFERENCE = 5.0e-3
MINIMUM_CONTINUATION_HISTORY_COSINE = 0.90

MINIMUM_EXPORT_ORDER = wp10c9d6c.MINIMUM_EXPORT_ORDER
MAXIMUM_FINE_PHYSICAL_DIFFERENCE = (
    wp10c9d6c.MAXIMUM_FINE_PHYSICAL_DIFFERENCE
)
MINIMUM_HISTORY_COSINE = wp10c9d6c.MINIMUM_HISTORY_COSINE
MINIMUM_ERROR_COSINE = wp10c9d6c.MINIMUM_ERROR_COSINE
MINIMUM_RELATIVE_ACTIVITY = wp10c9d6c.MINIMUM_RELATIVE_ACTIVITY
MAXIMUM_RESTART_DEFECT = wp10c9d6c.MAXIMUM_RESTART_DEFECT

PARENT_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_monolithic_anchor_audit_wp10c9d6c1"
)
PARENT_SUMMARY = PARENT_DIRECTORY / "summary.json"
PARENT_ARRAYS = PARENT_DIRECTORY / "decisive_arrays.npz"
UNIFORM_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_monolithic_uniform_exports_wp10c9d6c"
)
UNIFORM_REPLAY_CONTEXTS = UNIFORM_DIRECTORY / "replay_contexts.json"
UNIFORM_REPLAY_INPUTS = UNIFORM_DIRECTORY / "replay_inputs.npz"
CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_monolithic_four_level_wp10c9d6c2"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"

IMPLEMENTATION_SOURCES = (
    THIS_RUNNER,
    "scripts/run_causal_inner_monolithic_anchor_audit_wp10c9d6c1.py",
    "scripts/run_causal_inner_monolithic_uniform_exports_wp10c9d6c.py",
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_dae.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_monolithic_tangent.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_radial_linear_tangent.py",
    "tests/test_causal_inner_monolithic_four_level_wp10c9d6c2.py",
)

OBSERVABLE_NAMES = tuple(wp10c9d6c.OBSERVABLE_NAMES)


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
    return wp10c9d6c1._cosine(first, second)


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
        raise RuntimeError("WP10c9d6c2 analyzed git identity changed")
    return {
        "analyzed_base_commit": resolved,
        "analyzed_base_parent_commit": parent,
        "analyzed_base_tree_sha": tree,
    }


def _load_parent() -> tuple[dict, dict[str, np.ndarray]]:
    summary = json.loads(PARENT_SUMMARY.read_text(encoding="utf-8"))
    if (
        summary["classification"]
        != "uniform_inner_export_error_direction_unresolved"
        or not summary["method_passed"]
        or summary["passed"]
        or summary["authorized_next"] != "none"
    ):
        raise RuntimeError("WP10c9d6c1 binding classification changed")
    with np.load(PARENT_ARRAYS, allow_pickle=False) as source:
        arrays = {
            name: np.asarray(source[name])
            for name in source.files
        }
    if set(arrays) != set(summary["decisive_array_hashes"]):
        raise RuntimeError("WP10c9d6c1 decisive array set changed")
    for name, expected in summary["decisive_array_hashes"].items():
        if _array_sha256(arrays[name]) != expected:
            raise RuntimeError(
                f"WP10c9d6c1 decisive array changed: {name}"
            )
    return summary, arrays


def _zero_sources(n_cells: int) -> KerrSchildCellSourceRates:
    zero = np.zeros(int(n_cells), dtype=float)
    return KerrSchildCellSourceRates(
        rest_mass=zero,
        radial_momentum_over_c=np.array(zero, copy=True),
        angular_momentum_over_c=np.array(zero, copy=True),
        killing_energy_over_c2=np.array(zero, copy=True),
    ).validated_for(int(n_cells))


def _scales_for(
    context,
    base_primitives: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    state = causal_five_field_state_from_primitives(
        context,
        base_primitives,
    )
    evaluation = evaluate_causal_five_field_dae(
        pack_causal_five_field_state(state),
        context,
    )
    scaling = causal_five_field_dae_scaling(state, evaluation)
    n_primitive = int(np.asarray(base_primitives).size)
    return (
        np.asarray(
            scaling.column_scales[n_primitive : 2 * n_primitive],
            dtype=float,
        ),
        np.asarray(
            scaling.row_scales[:n_primitive],
            dtype=float,
        ),
    )


def _physical_direction(configuration: dict) -> np.ndarray:
    return (
        np.asarray(
            configuration["initial_directions"]["common_mode"],
            dtype=float,
        ).reshape(-1, 5)
        * np.asarray(
            configuration["primitive_column_scales"],
            dtype=float,
        ).reshape(-1, 5)
    )


def _continued_direction(
    source: dict,
    target: dict,
) -> np.ndarray:
    physical = wp10c9d6c1._pchip_values(
        source["context"].grid.centers,
        _physical_direction(source),
        target["context"].grid.centers,
    )
    columns = np.asarray(
        target["primitive_column_scales"],
        dtype=float,
    ).reshape(-1, 5)
    return (physical / columns).ravel()


def _heldout_near_excision(configuration: dict) -> np.ndarray:
    definition = wp10c9d6c.HELD_OUT_DEFINITIONS[
        "heldout_near_excision"
    ]
    context = configuration["context"]
    radii_over_rg = (
        np.asarray(context.grid.centers, dtype=float)
        / context.grid.gravitational_radius
    )
    envelope = np.exp(
        -0.5
        * (
            np.log(
                radii_over_rg
                / float(definition["center_over_rg"])
            )
            / float(definition["log_width"])
        )
        ** 2
    )
    coefficients = np.asarray(
        definition["scaled_coefficients"],
        dtype=float,
    )
    return (
        float(definition["amplitude"])
        * envelope[:, None]
        * coefficients[None, :]
    ).ravel()


def _build_four_configurations() -> tuple[dict, dict, dict]:
    replay_payload, replay_arrays = wp10c9d6c._load_replay_inputs()
    native = wp10c9d6c._configurations(
        replay_payload,
        replay_arrays,
    )
    common, common_decisive, common_report = (
        wp10c9d6c1._common_continuum_configurations(native)
    )
    if not common_report["passed"]:
        raise RuntimeError("WP10c9d6c1 common background changed")

    source = common["uniform_N256"]
    source_grid = source["context"].grid
    fine_grid = make_kerr_schild_column_grid(
        float(source_grid.edges[0]),
        float(source_grid.edges[-1]),
        FINE_ACTIVE_CELLS,
        float(source_grid.gravitational_radius),
    )
    fine_context = replace(
        source["context"],
        grid=fine_grid,
        stream_sources=_zero_sources(FINE_ACTIVE_CELLS),
    ).validated()
    fine_base = wp10c9d6c1._pchip_values(
        common_decisive["common_continuum_source_radii"],
        common_decisive["common_continuum_source_values"],
        fine_grid.centers,
    )
    fine_columns, fine_rows = _scales_for(fine_context, fine_base)
    fine_configuration = {
        "label": FINE_LABEL,
        "context": fine_context,
        "base_primitives": fine_base,
        "primitive_column_scales": fine_columns,
        "conservation_row_scales": fine_rows,
        "times": np.array(
            common["uniform_N256"]["times"],
            copy=True,
        ),
        "initial_directions": {},
    }
    fine_configuration["initial_directions"][
        "heldout_near_excision"
    ] = _heldout_near_excision(fine_configuration)
    fine_configuration["initial_directions"]["common_mode"] = (
        _continued_direction(
            common["uniform_N256"],
            fine_configuration,
        )
    )
    fine_configuration["initial_directions"][
        "common_mode_from_N128"
    ] = _continued_direction(
        common["uniform_N128"],
        fine_configuration,
    )
    result = {**common, FINE_LABEL: fine_configuration}

    nested = float(
        np.max(
            np.abs(
                fine_grid.edges[::2] - source_grid.edges
            )
            / np.maximum(
                np.abs(source_grid.edges),
                np.finfo(float).tiny,
            )
        )
    )
    reconstruction = causal_five_field_reconstruct_face_charts(
        fine_context,
        fine_base,
        purpose="flux",
    )
    factor_change = float(
        np.max(
            np.abs(
                reconstruction.admissibility_factors - 1.0
            )
        )
    )
    reference_background_defect = wp10c9d6c1._relative_difference(
        common["uniform_N128"]["base_primitives"],
        native["uniform_N128"]["base_primitives"],
    )
    direction_primary = fine_configuration["initial_directions"][
        "common_mode"
    ]
    direction_secondary = fine_configuration["initial_directions"][
        "common_mode_from_N128"
    ]
    report = {
        "fine_active_cells": FINE_ACTIVE_CELLS,
        "grid_nesting_defect": nested,
        "reference_background_defect": reference_background_defect,
        "maximum_reconstruction_factor_change": factor_change,
        "continuation_scaled_relative_difference": (
            _relative_difference(direction_primary, direction_secondary)
        ),
        "continuation_scaled_cosine": _cosine(
            direction_primary,
            direction_secondary,
        ),
        "passed": bool(
            fine_grid.centers.size == FINE_ACTIVE_CELLS
            and nested <= MAXIMUM_GRID_NESTING_DEFECT
            and reference_background_defect
            <= MAXIMUM_REFERENCE_BACKGROUND_DEFECT
            and factor_change
            <= MAXIMUM_RECONSTRUCTION_FACTOR_CHANGE
        ),
    }
    decisive = {
        **common_decisive,
        "uniform_N512__grid_edges": fine_grid.edges,
        "uniform_N512__grid_centers": fine_grid.centers,
        "uniform_N512__grid_cell_measures": fine_grid.cell_measures,
        "uniform_N512__grid_face_measures": fine_grid.face_measures,
        "uniform_N512__base_primitives": fine_base,
        "uniform_N512__primitive_column_scales": fine_columns,
        "uniform_N512__conservation_row_scales": fine_rows,
        "uniform_N512__direction_from_N256": direction_primary,
        "uniform_N512__direction_from_N128": direction_secondary,
    }
    return result, decisive, report


def _parent_histories(
    parent_arrays: dict[str, np.ndarray],
) -> dict:
    result = {}
    for label in PARENT_LABELS:
        result[label] = {
            "times": np.asarray(
                parent_arrays[f"common__{label}__times"],
                dtype=float,
            ),
            "signals": np.asarray(
                parent_arrays[f"common__{label}__signals"],
                dtype=float,
            ),
        }
    return result


def _selected_indices(size: int, stride: int) -> np.ndarray:
    indices = np.arange(0, int(size), int(stride), dtype=int)
    if indices[-1] != int(size) - 1:
        indices = np.append(indices, int(size) - 1)
    return indices


def _history_metrics(
    histories: dict[str, np.ndarray],
    labels: tuple[str, str, str],
    scales: np.ndarray,
) -> dict:
    normalized = {
        label: np.asarray(histories[label], dtype=float) / scales
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
            "reason": "no physically significant export component",
        }
    coarse = normalized[labels[0]][:, significant]
    medium = normalized[labels[1]][:, significant]
    fine = normalized[labels[2]][:, significant]
    first = medium - coarse
    second = fine - medium
    first_rms = float(np.sqrt(np.mean(first**2)))
    second_rms = float(np.sqrt(np.mean(second**2)))
    first_maximum = float(np.max(np.abs(first)))
    second_maximum = float(np.max(np.abs(second)))
    rms_order = float(
        np.log2(
            max(first_rms, np.finfo(float).tiny)
            / max(second_rms, np.finfo(float).tiny)
        )
    )
    maximum_order = float(
        np.log2(
            max(first_maximum, np.finfo(float).tiny)
            / max(second_maximum, np.finfo(float).tiny)
        )
    )
    component_first = np.sqrt(np.mean(first**2, axis=0))
    component_second = np.sqrt(np.mean(second**2, axis=0))
    component_orders = np.log2(
        np.maximum(component_first, np.finfo(float).tiny)
        / np.maximum(component_second, np.finfo(float).tiny)
    )
    active_indices = np.flatnonzero(significant)
    order_map = {
        OBSERVABLE_NAMES[index]: float(component_orders[position])
        for position, index in enumerate(active_indices)
    }
    component_cosines = {
        OBSERVABLE_NAMES[index]: _cosine(
            first[:, position],
            second[:, position],
        )
        for position, index in enumerate(active_indices)
    }
    history_cosine = _cosine(medium, fine)
    error_cosine = _cosine(first, second)
    passed = bool(
        rms_order >= MINIMUM_EXPORT_ORDER
        and maximum_order >= MINIMUM_EXPORT_ORDER
        and np.all(component_orders >= MINIMUM_EXPORT_ORDER)
        and second_maximum <= MAXIMUM_FINE_PHYSICAL_DIFFERENCE
        and history_cosine >= MINIMUM_HISTORY_COSINE
        and error_cosine >= MINIMUM_ERROR_COSINE
    )
    return {
        "passed": passed,
        "labels": labels,
        "significant_components": [
            OBSERVABLE_NAMES[index] for index in active_indices
        ],
        "observed_rms_order": rms_order,
        "observed_maximum_order": maximum_order,
        "component_orders": order_map,
        "component_error_cosines": component_cosines,
        "minimum_component_order": float(np.min(component_orders)),
        "coarse_medium_rms_difference": first_rms,
        "medium_fine_rms_difference": second_rms,
        "fine_rms_physical_difference": second_rms,
        "fine_maximum_physical_difference": second_maximum,
        "history_cosine": history_cosine,
        "refinement_error_cosine": error_cosine,
    }


def _stride_report(
    histories: dict,
    physical_scales: np.ndarray,
) -> dict:
    reports = {}
    reference_endpoints = {}
    for label in LABELS:
        history = histories[label]
        reference_endpoints[label] = wp10c9d6c._cumulative(
            history["times"],
            history["signals"],
        )[-1]
    maximum_endpoint_defect = 0.0
    duration = max(
        float(histories[LABELS[0]]["times"][-1]),
        np.finfo(float).tiny,
    )
    for stride in STRIDE_AUDITS:
        signals = {}
        cumulative = {}
        for label in LABELS:
            history = histories[label]
            indices = _selected_indices(
                history["times"].size,
                stride,
            )
            selected_times = history["times"][indices]
            selected_signals = history["signals"][indices]
            signals[label] = selected_signals
            cumulative[label] = wp10c9d6c._cumulative(
                selected_times,
                selected_signals,
            )
            endpoint_defect = float(
                np.max(
                    np.abs(
                        cumulative[label][-1]
                        - reference_endpoints[label]
                    )
                    / (physical_scales * duration)
                )
            )
            maximum_endpoint_defect = max(
                maximum_endpoint_defect,
                endpoint_defect,
            )
        triplet_reports = {}
        for name, labels in TRIPLETS.items():
            triplet_reports[name] = {
                "instantaneous": _history_metrics(
                    signals,
                    labels,
                    physical_scales,
                ),
                "cumulative": _history_metrics(
                    cumulative,
                    labels,
                    physical_scales * duration,
                ),
            }
        reports[str(stride)] = triplet_reports
    primary = reports[str(PRIMARY_STRIDE)]
    fine = primary["fine_N128_N256_N512"]
    return {
        "strides": reports,
        "primary_stride": PRIMARY_STRIDE,
        "primary_legacy": primary["legacy_N64_N128_N256"],
        "primary_fine": fine,
        "maximum_cumulative_endpoint_defect": maximum_endpoint_defect,
        "stride_passed": bool(
            maximum_endpoint_defect <= MAXIMUM_STRIDE_DEFECT
        ),
        "passed": bool(
            fine["instantaneous"]["passed"]
            and fine["cumulative"]["passed"]
            and maximum_endpoint_defect <= MAXIMUM_STRIDE_DEFECT
        ),
    }


def _continuation_sensitivity(
    primary: dict,
    secondary: dict,
    parent_histories: dict,
    physical_scales: np.ndarray,
) -> dict:
    duration = max(
        float(primary["times"][-1]),
        np.finfo(float).tiny,
    )
    primary_signals = np.asarray(primary["signals"], dtype=float)
    secondary_signals = np.asarray(secondary["signals"], dtype=float)
    difference = (
        primary_signals - secondary_signals
    ) / physical_scales
    parent_difference = (
        parent_histories["uniform_N256"]["signals"]
        - parent_histories["uniform_N128"]["signals"]
    ) / physical_scales
    primary_cumulative = wp10c9d6c._cumulative(
        primary["times"],
        primary_signals,
    )
    secondary_cumulative = wp10c9d6c._cumulative(
        secondary["times"],
        secondary_signals,
    )
    cumulative_difference = (
        primary_cumulative - secondary_cumulative
    ) / (physical_scales * duration)
    maximum = max(
        float(np.max(np.abs(difference))),
        float(np.max(np.abs(cumulative_difference))),
    )
    relative_to_parent_pair = float(
        np.linalg.norm(difference)
        / max(
            np.linalg.norm(parent_difference),
            np.finfo(float).tiny,
        )
    )
    signal_cosine = _cosine(
        primary_signals / physical_scales,
        secondary_signals / physical_scales,
    )
    cumulative_cosine = _cosine(
        primary_cumulative / (physical_scales * duration),
        secondary_cumulative / (physical_scales * duration),
    )
    return {
        "passed": bool(
            maximum <= MAXIMUM_CONTINUATION_EXPORT_DIFFERENCE
            and min(signal_cosine, cumulative_cosine)
            >= MINIMUM_CONTINUATION_HISTORY_COSINE
        ),
        "maximum_fixed_physical_difference": maximum,
        "relative_to_parent_medium_fine_difference": (
            relative_to_parent_pair
        ),
        "instantaneous_history_cosine": signal_cosine,
        "cumulative_history_cosine": cumulative_cosine,
    }


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
        _build_four_configurations()
    )
    if not construction["passed"]:
        raise RuntimeError("WP10c9d6c2 fine configuration failed")

    print(
        "WP10c9d6c2: build monolithic tangent uniform_N512",
        flush=True,
    )
    fine_configuration = configurations[FINE_LABEL]
    tangent = causal_five_field_monolithic_frozen_tangent(
        fine_configuration["context"],
        fine_configuration["base_primitives"],
        primitive_column_scales=(
            fine_configuration["primitive_column_scales"]
        ),
        conservation_row_scales=(
            fine_configuration["conservation_row_scales"]
        ),
        path_quadrature_order=wp10c9d6c.PATH_QUADRATURE_ORDER,
    )
    method_report = wp10c9d6c._method_report(
        fine_configuration,
        tangent,
    )
    observable_map = wp10c9d6c._observable_map(tangent)
    base_evaluation = (
        evaluate_causal_five_field_monolithic_backward_euler(
            fine_configuration["base_primitives"],
            fine_configuration["base_primitives"],
            1.0,
            fine_configuration["context"],
            path_quadrature_order=wp10c9d6c.PATH_QUADRATURE_ORDER,
        )
    )
    fine_baseline = wp10c9d6c._direct_observables(base_evaluation)

    parent_histories = _parent_histories(parent_arrays)
    physical_scales = np.asarray(
        parent_arrays["fixed_physical_observable_scales"],
        dtype=float,
    )
    decisive["fixed_physical_observable_scales"] = physical_scales
    decisive["uniform_N512__baseline_observables"] = fine_baseline
    decisive["uniform_N512__descriptor"] = (
        tangent.descriptor_scaled_matrix
    )
    decisive["uniform_N512__stationary_jacobian"] = (
        tangent.stationary_scaled_jacobian
    )
    decisive["uniform_N512__storage_rate_derivative"] = (
        tangent.storage_rate_derivative_scaled_matrix
    )
    decisive["uniform_N512__generator"] = tangent.scaled_generator_per_s
    decisive["uniform_N512__scaled_base_rate"] = (
        tangent.scaled_base_rate_per_s
    )
    decisive["uniform_N512__observable_map"] = observable_map

    variant_histories = {}
    restart_defects = {}
    for continuation in CONTINUATIONS:
        print(
            f"WP10c9d6c2: propagate uniform_N512 {continuation}",
            flush=True,
        )
        initial = (
            fine_configuration["initial_directions"]["common_mode"]
            if continuation == PRIMARY_CONTINUATION
            else fine_configuration["initial_directions"][
                "common_mode_from_N128"
            ]
        )
        state, restart = wp10c9d6c._propagate(
            tangent.scaled_generator_per_s,
            initial,
            fine_configuration["times"],
        )
        signals = state @ observable_map.T
        variant_histories[continuation] = {
            "times": np.asarray(
                fine_configuration["times"],
                dtype=float,
            ),
            "signals": signals,
            "final_scaled_state": state[-1],
        }
        restart_defects[continuation] = restart
        prefix = f"uniform_N512__{continuation}__"
        decisive[prefix + "times"] = fine_configuration["times"]
        decisive[prefix + "signals"] = signals
        decisive[prefix + "cumulative"] = wp10c9d6c._cumulative(
            fine_configuration["times"],
            signals,
        )
        decisive[prefix + "final_scaled_state"] = state[-1]

    reports = {}
    for continuation in CONTINUATIONS:
        histories = {
            **parent_histories,
            FINE_LABEL: variant_histories[continuation],
        }
        reports[continuation] = _stride_report(
            histories,
            physical_scales,
        )
    sensitivity = _continuation_sensitivity(
        variant_histories[PRIMARY_CONTINUATION],
        variant_histories[SECONDARY_CONTINUATION],
        parent_histories,
        physical_scales,
    )
    restart_passed = bool(
        max(restart_defects.values()) <= MAXIMUM_RESTART_DEFECT
    )
    classification_agrees = bool(
        reports[PRIMARY_CONTINUATION]["passed"]
        == reports[SECONDARY_CONTINUATION]["passed"]
    )
    method_passed = bool(method_report["passed"])
    fine_passed = bool(
        method_passed
        and restart_passed
        and sensitivity["passed"]
        and classification_agrees
        and reports[PRIMARY_CONTINUATION]["passed"]
        and reports[SECONDARY_CONTINUATION]["passed"]
    )
    robust_rejection = bool(
        method_passed
        and restart_passed
        and sensitivity["passed"]
        and classification_agrees
        and not reports[PRIMARY_CONTINUATION]["passed"]
        and not reports[SECONDARY_CONTINUATION]["passed"]
    )
    if not method_passed:
        classification = "four_level_uniform_method_failed"
        authorized_next = "none"
    elif not restart_passed or not sensitivity["passed"] or not classification_agrees:
        classification = (
            "four_level_uniform_direction_continuation_unresolved"
        )
        authorized_next = "none"
    elif fine_passed:
        classification = (
            "fine_uniform_asymptotic_direction_certified"
        )
        authorized_next = "heldout_uniform_physical_exports"
    else:
        classification = (
            "four_level_uniform_asymptotic_direction_rejected"
        )
        authorized_next = "uniform_near_excision_redesign"

    decisive["uniform_N64__parent_signals"] = parent_histories[
        "uniform_N64"
    ]["signals"]
    decisive["uniform_N128__parent_signals"] = parent_histories[
        "uniform_N128"
    ]["signals"]
    decisive["uniform_N256__parent_signals"] = parent_histories[
        "uniform_N256"
    ]["signals"]
    decisive["parent_times"] = parent_histories["uniform_N128"]["times"]

    source_hashes, source_manifest = _source_manifest()
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "meshes": MESHES,
        "labels": LABELS,
        "active_cells": {
            "uniform_N64": 24,
            "uniform_N128": 48,
            "uniform_N256": 96,
            "uniform_N512": FINE_ACTIVE_CELLS,
        },
        "triplets": TRIPLETS,
        "continuations": CONTINUATIONS,
        "primary_stride": PRIMARY_STRIDE,
        "stride_audits": STRIDE_AUDITS,
        "gates": {
            "minimum_export_order": MINIMUM_EXPORT_ORDER,
            "maximum_fine_physical_difference": (
                MAXIMUM_FINE_PHYSICAL_DIFFERENCE
            ),
            "minimum_history_cosine": MINIMUM_HISTORY_COSINE,
            "minimum_error_cosine": MINIMUM_ERROR_COSINE,
            "maximum_continuation_export_difference": (
                MAXIMUM_CONTINUATION_EXPORT_DIFFERENCE
            ),
            "minimum_continuation_history_cosine": (
                MINIMUM_CONTINUATION_HISTORY_COSINE
            ),
            "maximum_stride_defect": MAXIMUM_STRIDE_DEFECT,
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
        "passed": fine_passed,
        "method_passed": method_passed,
        "robust_rejection": robust_rejection,
        "authorized_next": authorized_next,
        "parent_wp10c9d6c1_classification_preserved": True,
        "fine_configuration": construction,
        "fine_method_report": method_report,
        "continuation_reports": reports,
        "continuation_sensitivity": sensitivity,
        "continuation_classification_agrees": classification_agrees,
        "restart_defects": restart_defects,
        "restart_passed": restart_passed,
        "heldout_uniform_exports_authorized": bool(fine_passed),
        "uniform_near_excision_redesign_authorized": robust_rejection,
        "embedded_export_discrimination_authorized": False,
        "nonlinear_physical_trajectory_authorized": False,
        "production_operator_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "uses_production_generator": False,
        "uses_production_anchor_storage_derivative": False,
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
    scientific_status = (
        "CERTIFIED"
        if fine_passed
        else "REJECTED"
        if robust_rejection
        else "DIAGNOSTIC ONLY"
    )
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "scientific_status": scientific_status,
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        **identity,
        "generation_command": (
            "PYTHONPATH=src:scripts python3 "
            "scripts/run_causal_inner_monolithic_four_level_"
            "wp10c9d6c2.py"
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
            "whether the N128/N256/N512 common-mode physical-export "
            "errors reach the unchanged asymptotic-direction gate"
        ),
        "does_not_establish": (
            "embedded, nonlinear, production, fixed-Q, or reduced "
            "slow-time certification"
        ),
    }
    _write_json(PROVENANCE_PATH, provenance)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    print(
        f"WP10c9d6c2: classification={classification}",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
