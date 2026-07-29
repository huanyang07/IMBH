#!/usr/bin/env python3
"""Run the uniform physical-export preflight of the monolithic inner DAE.

The package constructs a self-consistent frozen tangent on the uniform
N64/N128/N256 common-mode grids.  Its base rate comes from the monolithic
stationary residual and reconstructed temporal descriptor, and its evolving
tangent includes the derivative of that same descriptor acting on that rate.
No production generator or production-anchor storage derivative is reused.

The common mode is binding.  Two predeclared held-out perturbations run only
when the common-mode instantaneous and cumulative physical exports pass.
Embedded and nonlinear work remain blocked unless the full uniform package
passes.
"""

from __future__ import annotations

import argparse
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
from scipy.sparse.linalg import expm_multiply


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_causal_inner_common_mode_audit_wp10c8y as wp10c8y
import run_causal_inner_family_transfer_audit_wp10c9c0c as wp10c9c0c
import run_causal_inner_frozen_hardening_wp10c9d5a as wp10c9d5a
import run_causal_inner_micro_export_preflight_wp10c9d0 as wp10c9d0

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    causal_five_field_monolithic_frozen_tangent,
    causal_five_field_monolithic_storage_rate_action,
    evaluate_causal_five_field_monolithic_backward_euler,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c"
ANALYZED_BASE_COMMIT = "5884b307a3245e6f1c948d5147b5c2a1c70a509a"
ANALYZED_BASE_PARENT = "4140ffeb58ce791425219b88209a8f20e0e2a70d"
ANALYZED_BASE_TREE = "642ce35037d49da876f80a842d22aa5b7527f2cc"
THIS_RUNNER = (
    "scripts/run_causal_inner_monolithic_uniform_exports_wp10c9d6c.py"
)

MESHES = (64, 128, 256)
LABELS = tuple(f"uniform_N{mesh}" for mesh in MESHES)
PERTURBATIONS = (
    "common_mode",
    "heldout_near_excision",
    "heldout_mid_inner",
)
HELD_OUT_DEFINITIONS = {
    "heldout_near_excision": {
        "center_over_rg": 2.4,
        "log_width": 0.18,
        "scaled_coefficients": (0.20, -0.40, 0.30, 0.10, 0.50),
        "amplitude": 1.0e-3,
    },
    "heldout_mid_inner": {
        "center_over_rg": 6.0,
        "log_width": 0.24,
        "scaled_coefficients": (-0.25, 0.20, -0.35, 0.30, -0.20),
        "amplitude": 1.0e-3,
    },
}

PATH_QUADRATURE_ORDER = 6
DIRECTIONAL_STEP = 2.0e-4
TIME_SAMPLE_STRIDE = 2
MINIMUM_EXPORT_ORDER = 0.75
MAXIMUM_FINE_PHYSICAL_DIFFERENCE = 0.05
MINIMUM_HISTORY_COSINE = 0.90
MINIMUM_ERROR_COSINE = 0.90
MINIMUM_RELATIVE_ACTIVITY = 1.0e-8
MAXIMUM_RECONSTRUCTION_DEFECT = 1.0e-12
MAXIMUM_PARTITION_DEFECT = 1.0e-12
MAXIMUM_COMPONENT_DEFECT = 1.0e-12
MAXIMUM_BALANCE_DEFECT = 1.0e-12
MAXIMUM_FACTORIZATION_DEFECT = 1.0e-12
MAXIMUM_CENTERED_STORAGE_ACTION_DEFECT = 2.0e-7
MAXIMUM_DIRECTIONAL_STATIONARY_DEFECT = 2.0e-6
MAXIMUM_DIRECTIONAL_STORAGE_RATE_DEFECT = 2.0e-6
MAXIMUM_DIRECTIONAL_EXPORT_DEFECT = 2.0e-6
MAXIMUM_RESTART_DEFECT = 1.0e-12
MAXIMUM_DESCRIPTOR_CONDITION = 1.0e12
MINIMUM_CHARACTERISTIC_SPEED = 1.0e-6
MINIMUM_CHARACTERISTIC_GAP = 1.0e-6
MAXIMUM_CHARACTERISTIC_CONDITION = 1.0e10

PARENT_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_monolithic_manufactured_wp10c9d6b"
)
PARENT_SUMMARY = PARENT_DIRECTORY / "summary.json"
PARENT_ARRAYS = PARENT_DIRECTORY / "decisive_arrays.npz"
CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_monolithic_uniform_exports_wp10c9d6c"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
REPLAY_CONTEXTS = CANONICAL_DIRECTORY / "replay_contexts.json"
REPLAY_INPUTS = CANONICAL_DIRECTORY / "replay_inputs.npz"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"

IMPLEMENTATION_SOURCES = (
    THIS_RUNNER,
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_dae.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_monolithic_tangent.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_radial_linear_tangent.py",
    "src/imri_qpe/layer3_minidisk_1d/__init__.py",
    "tests/test_causal_inner_monolithic_tangent.py",
    "tests/"
    "test_causal_inner_monolithic_uniform_exports_wp10c9d6c.py",
)

CONSERVATIVE_FIELDS = np.asarray((0, 2, 3), dtype=int)
OBSERVABLE_NAMES = tuple(wp10c9d0.OBSERVABLE_NAMES)


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
        raise RuntimeError("WP10c9d6c analyzed git identity changed")
    return {
        "analyzed_base_commit": resolved,
        "analyzed_base_parent_commit": parent,
        "analyzed_base_tree_sha": tree,
    }


def _prepare_replay_inputs() -> dict:
    """Promote the minimal uniform physical inputs into canonical storage."""

    common = wp10c9d0._load_npz(wp10c9d0.WP10C8Y_ARRAYS)
    contexts, _profiles = wp10c9c0c._common_contexts()
    operators = wp10c8y._load_family_operators()["production"]
    arrays: dict[str, np.ndarray] = {}
    contexts_payload = {}
    source_hashes = {
        _relative(wp10c9d0.WP10C8Y_ARRAYS): _sha256(
            wp10c9d0.WP10C8Y_ARRAYS
        )
    }
    for mesh, label in zip(MESHES, LABELS, strict=True):
        context = contexts[mesh]
        operator = operators[mesh]
        prefix = f"{label}__"
        arrays[prefix + "base_primitives"] = np.asarray(
            operator["base_primitives"],
            dtype=float,
        )
        arrays[prefix + "primitive_column_scales"] = np.asarray(
            operator["primitive_column_scales"],
            dtype=float,
        )
        arrays[prefix + "conservation_row_scales"] = np.asarray(
            operator["conservation_row_scales"],
            dtype=float,
        )
        arrays[prefix + "common_amplitudes"] = np.asarray(
            common[f"N{mesh}_common_amplitudes"],
            dtype=float,
        )
        arrays[prefix + "common_initial"] = np.asarray(
            common[f"production_N{mesh}_state"],
            dtype=float,
        )[0]
        arrays[prefix + "times"] = np.asarray(
            common[f"production_N{mesh}_times"],
            dtype=float,
        )
        contexts_payload[label] = wp10c9d5a._context_payload(
            context,
            label=label,
            arrays=arrays,
        )
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(REPLAY_INPUTS, **arrays)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "labels": LABELS,
        "contexts": contexts_payload,
        "source_input_hashes": source_hashes,
        "replay_inputs_path": _relative(REPLAY_INPUTS),
        "replay_inputs_sha256": _sha256(REPLAY_INPUTS),
        "replay_array_hashes": {
            name: _array_sha256(values)
            for name, values in arrays.items()
        },
    }
    _write_json(REPLAY_CONTEXTS, payload)
    return payload


def _load_replay_inputs() -> tuple[dict, dict[str, np.ndarray]]:
    if not REPLAY_CONTEXTS.exists() or not REPLAY_INPUTS.exists():
        raise FileNotFoundError(
            "WP10c9d6c replay inputs are absent; run once with "
            "--prepare-replay-inputs"
        )
    payload = json.loads(REPLAY_CONTEXTS.read_text(encoding="utf-8"))
    with np.load(REPLAY_INPUTS, allow_pickle=False) as source:
        arrays = {
            name: np.asarray(source[name])
            for name in source.files
        }
    if set(arrays) != set(payload["replay_array_hashes"]):
        raise RuntimeError("WP10c9d6c replay array set changed")
    for name, expected in payload["replay_array_hashes"].items():
        if _array_sha256(arrays[name]) != expected:
            raise RuntimeError(f"WP10c9d6c replay array changed: {name}")
    return payload, arrays


def _configurations(
    replay_payload: dict,
    replay_arrays: dict[str, np.ndarray],
) -> dict:
    result = {}
    for label in LABELS:
        prefix = f"{label}__"
        base = np.asarray(
            replay_arrays[prefix + "base_primitives"],
            dtype=float,
        )
        amplitudes = np.asarray(
            replay_arrays[prefix + "common_amplitudes"],
            dtype=float,
        )
        common_initial = np.asarray(
            replay_arrays[prefix + "common_initial"],
            dtype=float,
        )
        columns = np.asarray(
            replay_arrays[prefix + "primitive_column_scales"],
            dtype=float,
        ).ravel()
        context = wp10c9d5a._context_from_payload(
            replay_payload["contexts"][label],
            replay_arrays,
        )
        common_physical = amplitudes * common_initial
        result[label] = {
            "label": label,
            "context": context,
            "base_primitives": base,
            "primitive_column_scales": columns,
            "conservation_row_scales": np.asarray(
                replay_arrays[prefix + "conservation_row_scales"],
                dtype=float,
            ).ravel(),
            "times": np.asarray(
                replay_arrays[prefix + "times"],
                dtype=float,
            ),
            "initial_directions": {
                "common_mode": common_physical.ravel() / columns,
            },
        }
        radii_over_rg = (
            np.asarray(context.grid.centers, dtype=float)
            / context.grid.gravitational_radius
        )
        for name, definition in HELD_OUT_DEFINITIONS.items():
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
            result[label]["initial_directions"][name] = (
                float(definition["amplitude"])
                * envelope[:, None]
                * coefficients[None, :]
            ).ravel()
    return result


def _direct_observables(evaluation) -> np.ndarray:
    fluxes = np.asarray(
        evaluation.stationary_ledger.interfaces
        .candidate_shared_face_fluxes_over_c,
        dtype=float,
    )
    inner = fluxes[0, CONSERVATIVE_FIELDS]
    outer = fluxes[-1, CONSERVATIVE_FIELDS]
    net = -np.sum(
        evaluation.residual_rows[:, CONSERVATIVE_FIELDS],
        axis=0,
    )
    cooling = -np.sum(
        evaluation.cooling_rows[:, CONSERVATIVE_FIELDS[1:]],
        axis=0,
    )
    height = -np.sum(
        evaluation.lower_height_work_rows[
            :,
            CONSERVATIVE_FIELDS[1:],
        ],
        axis=0,
    )
    return np.concatenate((inner, outer, net, cooling, height))


def _observable_map(tangent) -> np.ndarray:
    n_cells = int(tangent.base_primitives.shape[0])
    rows = tangent.conservation_row_scales.reshape(n_cells, 5)
    spatial = tangent.spatial_tangent
    face = np.asarray(
        spatial.shared_face_flux_scaled_jacobians,
        dtype=float,
    )
    stationary_physical = (
        tangent.stationary_scaled_jacobian
        * tangent.conservation_row_scales[:, None]
    ).reshape(n_cells, 5, -1)
    cooling_physical = (
        spatial.block_scaled_jacobians["candidate_cooling"]
        * tangent.conservation_row_scales[:, None]
    ).reshape(n_cells, 5, -1)
    height_physical = (
        spatial.block_scaled_jacobians[
            "candidate_lower_height_work"
        ]
        * tangent.conservation_row_scales[:, None]
    ).reshape(n_cells, 5, -1)
    del rows
    return np.concatenate(
        (
            face[0, CONSERVATIVE_FIELDS],
            face[-1, CONSERVATIVE_FIELDS],
            -np.sum(
                stationary_physical[:, CONSERVATIVE_FIELDS],
                axis=0,
            ),
            -np.sum(
                cooling_physical[:, CONSERVATIVE_FIELDS[1:]],
                axis=0,
            ),
            -np.sum(
                height_physical[:, CONSERVATIVE_FIELDS[1:]],
                axis=0,
            ),
        ),
        axis=0,
    )


def _fourth_order_action(values: dict[float, np.ndarray], step: float):
    return (
        -values[2.0 * step]
        + 8.0 * values[step]
        - 8.0 * values[-step]
        + values[-2.0 * step]
    ) / (12.0 * step)


def _directional_audit(configuration: dict, tangent, name: str) -> dict:
    base = np.asarray(configuration["base_primitives"], dtype=float)
    columns = np.asarray(
        configuration["primitive_column_scales"],
        dtype=float,
    )
    rows = np.asarray(
        configuration["conservation_row_scales"],
        dtype=float,
    )
    scaled_direction = np.asarray(
        configuration["initial_directions"][name],
        dtype=float,
    ).ravel()
    normalization = max(
        float(np.max(np.abs(scaled_direction))),
        np.finfo(float).tiny,
    )
    scaled_direction = scaled_direction / normalization
    physical_direction = (columns * scaled_direction).reshape(base.shape)
    residuals = {}
    observables = {}
    for multiplier in (-2.0, -1.0, 1.0, 2.0):
        offset = multiplier * DIRECTIONAL_STEP
        charts = base + offset * physical_direction
        evaluation = (
            evaluate_causal_five_field_monolithic_backward_euler(
                charts,
                charts,
                1.0,
                configuration["context"],
                path_quadrature_order=PATH_QUADRATURE_ORDER,
            )
        )
        residuals[offset] = evaluation.residual_rows.ravel() / rows
        observables[offset] = _direct_observables(evaluation)
    direct_stationary = _fourth_order_action(
        residuals,
        DIRECTIONAL_STEP,
    )
    direct_export = _fourth_order_action(
        observables,
        DIRECTIONAL_STEP,
    )
    matrix_stationary = (
        tangent.stationary_scaled_jacobian @ scaled_direction
    )
    matrix_export = _observable_map(tangent) @ scaled_direction

    rate = tangent.physical_base_rate_per_s.reshape(base.shape)
    storage_actions = {}
    for multiplier in (-2.0, -1.0, 1.0, 2.0):
        offset = multiplier * DIRECTIONAL_STEP
        storage_actions[offset] = (
            causal_five_field_monolithic_storage_rate_action(
                configuration["context"],
                base + offset * physical_direction,
                rate,
                conservation_row_scales=rows,
            )
        )
    direct_storage_rate = _fourth_order_action(
        storage_actions,
        DIRECTIONAL_STEP,
    )
    matrix_storage_rate = (
        tangent.storage_rate_derivative_scaled_matrix
        @ scaled_direction
    )
    return {
        "stationary_relative_defect": _relative_difference(
            direct_stationary,
            matrix_stationary,
        ),
        "storage_rate_relative_defect": _relative_difference(
            direct_storage_rate,
            matrix_storage_rate,
        ),
        "export_relative_defect": _relative_difference(
            direct_export,
            matrix_export,
        ),
        "scaled_direction_normalization": normalization,
    }


def _method_report(configuration: dict, tangent) -> dict:
    audits = {
        name: _directional_audit(configuration, tangent, name)
        for name in ("common_mode", "heldout_near_excision")
    }
    stationary_defect = max(
        item["stationary_relative_defect"] for item in audits.values()
    )
    storage_rate_defect = max(
        item["storage_rate_relative_defect"] for item in audits.values()
    )
    export_defect = max(
        item["export_relative_defect"] for item in audits.values()
    )
    descriptor_condition = float(
        np.linalg.cond(tangent.descriptor_scaled_matrix)
    )
    spatial = tangent.spatial_tangent
    gates = {
        "node_reconstruction": bool(
            tangent.maximum_node_reconstruction_relative_defect
            <= MAXIMUM_RECONSTRUCTION_DEFECT
        ),
        "node_partition": bool(
            tangent.maximum_node_partition_of_unity_defect
            <= MAXIMUM_PARTITION_DEFECT
        ),
        "descriptor_components": bool(
            tangent.maximum_descriptor_component_defect
            <= MAXIMUM_COMPONENT_DEFECT
        ),
        "storage_rate_components": bool(
            tangent.maximum_storage_rate_component_defect
            <= MAXIMUM_COMPONENT_DEFECT
        ),
        "base_rate_balance": bool(
            tangent.maximum_base_rate_balance_defect
            <= MAXIMUM_BALANCE_DEFECT
        ),
        "generator_factorization": bool(
            tangent.maximum_generator_factorization_defect
            <= MAXIMUM_FACTORIZATION_DEFECT
        ),
        "centered_storage_action": bool(
            tangent.maximum_centered_storage_action_relative_defect
            <= MAXIMUM_CENTERED_STORAGE_ACTION_DEFECT
        ),
        "directional_stationary": bool(
            stationary_defect <= MAXIMUM_DIRECTIONAL_STATIONARY_DEFECT
        ),
        "directional_storage_rate": bool(
            storage_rate_defect
            <= MAXIMUM_DIRECTIONAL_STORAGE_RATE_DEFECT
        ),
        "directional_export": bool(
            export_defect <= MAXIMUM_DIRECTIONAL_EXPORT_DEFECT
        ),
        "descriptor_condition": bool(
            descriptor_condition <= MAXIMUM_DESCRIPTOR_CONDITION
        ),
        "characteristics": bool(
            spatial.incoming_inner_characteristics == 0
            and spatial.minimum_absolute_characteristic_speed
            >= MINIMUM_CHARACTERISTIC_SPEED
            and spatial.minimum_characteristic_spectral_gap
            >= MINIMUM_CHARACTERISTIC_GAP
            and spatial.maximum_characteristic_descriptor_condition_number
            <= MAXIMUM_CHARACTERISTIC_CONDITION
        ),
        "center_broken_paths": bool(
            tangent.uses_center_broken_within_cell_paths
            and spatial.center_broken_within_cell_paths
        ),
        "production_neutral": bool(
            not tangent.uses_production_generator
            and not tangent.uses_production_anchor_storage_derivative
        ),
    }
    return {
        "passed": bool(all(gates.values())),
        "gates": gates,
        "directional_audits": audits,
        "maximum_directional_stationary_defect": stationary_defect,
        "maximum_directional_storage_rate_defect": storage_rate_defect,
        "maximum_directional_export_defect": export_defect,
        "descriptor_condition_number": descriptor_condition,
        "maximum_node_reconstruction_defect": (
            tangent.maximum_node_reconstruction_relative_defect
        ),
        "maximum_node_partition_defect": (
            tangent.maximum_node_partition_of_unity_defect
        ),
        "maximum_descriptor_component_defect": (
            tangent.maximum_descriptor_component_defect
        ),
        "maximum_storage_rate_component_defect": (
            tangent.maximum_storage_rate_component_defect
        ),
        "maximum_base_rate_balance_defect": (
            tangent.maximum_base_rate_balance_defect
        ),
        "maximum_generator_factorization_defect": (
            tangent.maximum_generator_factorization_defect
        ),
        "maximum_centered_storage_action_defect": (
            tangent.maximum_centered_storage_action_relative_defect
        ),
        "minimum_absolute_characteristic_speed": (
            spatial.minimum_absolute_characteristic_speed
        ),
        "minimum_characteristic_spectral_gap": (
            spatial.minimum_characteristic_spectral_gap
        ),
        "maximum_characteristic_descriptor_condition": (
            spatial.maximum_characteristic_descriptor_condition_number
        ),
        "incoming_excision_characteristics": (
            spatial.incoming_inner_characteristics
        ),
    }


def _propagate(
    generator: np.ndarray,
    initial: np.ndarray,
    times: np.ndarray,
) -> tuple[np.ndarray, float]:
    trace = float(np.trace(generator))
    state = np.asarray(
        expm_multiply(
            generator,
            np.asarray(initial, dtype=float).ravel(),
            start=float(times[0]),
            stop=float(times[-1]),
            num=int(times.size),
            endpoint=True,
            traceA=trace,
        ),
        dtype=float,
    )
    half = expm_multiply(
        generator * (0.5 * float(times[-1])),
        np.asarray(initial, dtype=float).ravel(),
        traceA=0.5 * float(times[-1]) * trace,
    )
    restarted = expm_multiply(
        generator * (0.5 * float(times[-1])),
        half,
        traceA=0.5 * float(times[-1]) * trace,
    )
    defect = _relative_difference(restarted, state[-1])
    return state, defect


def _cumulative(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values)
    increments = np.diff(times)
    result[1:] = np.cumsum(
        0.5
        * increments[:, None]
        * (values[1:] + values[:-1]),
        axis=0,
    )
    return result


def _fixed_physical_scales(
    baselines: dict[str, np.ndarray],
) -> np.ndarray:
    values = np.asarray([baselines[label] for label in LABELS])
    mass = max(
        float(np.max(np.abs(values[:, (0, 3, 6)]))),
        np.finfo(float).tiny,
    )
    angular = max(
        float(np.max(np.abs(values[:, (1, 4, 7, 9, 11)]))),
        np.finfo(float).tiny,
    )
    energy = max(
        float(np.max(np.abs(values[:, (2, 5, 8, 10, 12)]))),
        np.finfo(float).tiny,
    )
    return np.asarray(
        [
            mass,
            angular,
            energy,
            mass,
            angular,
            energy,
            mass,
            angular,
            energy,
            angular,
            energy,
            angular,
            energy,
        ],
        dtype=float,
    )


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=float).ravel()
    right = np.asarray(second, dtype=float).ravel()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= np.finfo(float).tiny:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(np.dot(left, right) / denominator)


def _history_metrics(
    histories: dict[str, np.ndarray],
    scales: np.ndarray,
) -> dict:
    normalized = {
        label: np.asarray(histories[label], dtype=float) / scales
        for label in LABELS
    }
    response = np.max(
        np.abs(np.asarray([normalized[label] for label in LABELS])),
        axis=(0, 1),
    )
    significant = response >= MINIMUM_RELATIVE_ACTIVITY
    if not np.any(significant):
        return {
            "passed": False,
            "reason": "no physically significant export component",
        }
    coarse = normalized[LABELS[0]][:, significant]
    medium = normalized[LABELS[1]][:, significant]
    fine = normalized[LABELS[2]][:, significant]
    first = medium - coarse
    second = fine - medium
    first_rms = float(np.sqrt(np.mean(first**2)))
    second_rms = float(np.sqrt(np.mean(second**2)))
    first_maximum = float(np.max(np.abs(first)))
    second_maximum = float(np.max(np.abs(second)))
    rms_order = float(np.log2(first_rms / second_rms))
    maximum_order = float(
        np.log2(first_maximum / second_maximum)
    )
    component_first = np.sqrt(np.mean(first**2, axis=0))
    component_second = np.sqrt(np.mean(second**2, axis=0))
    component_orders = np.log2(component_first / component_second)
    active_indices = np.flatnonzero(significant)
    order_map = {
        OBSERVABLE_NAMES[index]: float(component_orders[position])
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
        "significant_components": [
            OBSERVABLE_NAMES[index] for index in active_indices
        ],
        "observed_rms_order": rms_order,
        "observed_maximum_order": maximum_order,
        "component_orders": order_map,
        "minimum_component_order": float(np.min(component_orders)),
        "fine_rms_physical_difference": second_rms,
        "fine_maximum_physical_difference": second_maximum,
        "history_cosine": history_cosine,
        "refinement_error_cosine": error_cosine,
    }


def _perturbation_ladder(
    name: str,
    configurations: dict,
    tangents: dict,
    observable_maps: dict,
    physical_scales: np.ndarray,
    decisive: dict[str, np.ndarray],
) -> dict:
    signals = {}
    cumulative = {}
    restart_defects = {}
    state_norms = {}
    for label in LABELS:
        configuration = configurations[label]
        print(f"WP10c9d6c: propagate {name} on {label}", flush=True)
        times = np.asarray(configuration["times"], dtype=float)
        state, restart = _propagate(
            tangents[label].scaled_generator_per_s,
            configuration["initial_directions"][name],
            times,
        )
        indices = np.arange(
            0,
            times.size,
            TIME_SAMPLE_STRIDE,
            dtype=int,
        )
        if indices[-1] != times.size - 1:
            indices = np.append(indices, times.size - 1)
        selected_times = times[indices]
        selected_state = state[indices]
        selected_signals = (
            selected_state @ observable_maps[label].T
        )
        signals[label] = selected_signals
        cumulative[label] = _cumulative(
            selected_times,
            selected_signals,
        )
        restart_defects[label] = restart
        state_norms[label] = {
            "initial": float(np.linalg.norm(state[0])),
            "final": float(np.linalg.norm(state[-1])),
            "maximum": float(
                np.max(np.linalg.norm(state, axis=1))
            ),
        }
        prefix = f"{name}__{label}__"
        decisive[prefix + "times"] = selected_times
        decisive[prefix + "signals"] = selected_signals
        decisive[prefix + "cumulative"] = cumulative[label]
        decisive[prefix + "final_scaled_state"] = state[-1]
    final_time = max(
        float(configurations[label]["times"][-1]) for label in LABELS
    )
    instantaneous = _history_metrics(signals, physical_scales)
    integrated = _history_metrics(
        cumulative,
        physical_scales * final_time,
    )
    restart_passed = bool(
        max(restart_defects.values()) <= MAXIMUM_RESTART_DEFECT
    )
    return {
        "executed": True,
        "passed": bool(
            instantaneous["passed"]
            and integrated["passed"]
            and restart_passed
        ),
        "instantaneous": instantaneous,
        "cumulative": integrated,
        "restart_defects": restart_defects,
        "state_norms": state_norms,
        "restart_passed": restart_passed,
    }


def _environment() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def run(*, prepare_replay_inputs: bool = False) -> dict:
    started = time.perf_counter()
    identity = _validate_analyzed_git_identity()
    parent = json.loads(PARENT_SUMMARY.read_text(encoding="utf-8"))
    if (
        not parent["method_passed"]
        or not parent[
            "uniform_grid_physical_export_preflight_authorized"
        ]
        or parent["embedded_export_discrimination_authorized"]
    ):
        raise RuntimeError("WP10c9d6b binding authorization changed")
    if prepare_replay_inputs:
        _prepare_replay_inputs()
    replay_payload, replay_arrays = _load_replay_inputs()
    configurations = _configurations(replay_payload, replay_arrays)
    decisive: dict[str, np.ndarray] = {}
    tangents = {}
    observable_maps = {}
    baselines = {}
    method_reports = {}
    method_passed = True

    for label in LABELS:
        print(f"WP10c9d6c: build monolithic tangent {label}", flush=True)
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
            path_quadrature_order=PATH_QUADRATURE_ORDER,
        )
        report = _method_report(configuration, tangent)
        method_passed = bool(method_passed and report["passed"])
        tangents[label] = tangent
        method_reports[label] = report
        observable_maps[label] = _observable_map(tangent)
        base_evaluation = (
            evaluate_causal_five_field_monolithic_backward_euler(
                configuration["base_primitives"],
                configuration["base_primitives"],
                1.0,
                configuration["context"],
                path_quadrature_order=PATH_QUADRATURE_ORDER,
            )
        )
        baselines[label] = _direct_observables(base_evaluation)
        prefix = f"{label}__"
        decisive[prefix + "descriptor"] = (
            tangent.descriptor_scaled_matrix
        )
        decisive[prefix + "storage_rate_derivative"] = (
            tangent.storage_rate_derivative_scaled_matrix
        )
        decisive[prefix + "stationary_jacobian"] = (
            tangent.stationary_scaled_jacobian
        )
        decisive[prefix + "generator"] = (
            tangent.scaled_generator_per_s
        )
        decisive[prefix + "scaled_base_rate"] = (
            tangent.scaled_base_rate_per_s
        )
        decisive[prefix + "observable_map"] = observable_maps[label]
        decisive[prefix + "baseline_observables"] = baselines[label]

    physical_scales = _fixed_physical_scales(baselines)
    decisive["fixed_physical_observable_scales"] = physical_scales
    ladders = {}
    if method_passed:
        ladders["common_mode"] = _perturbation_ladder(
            "common_mode",
            configurations,
            tangents,
            observable_maps,
            physical_scales,
            decisive,
        )
    else:
        ladders["common_mode"] = {
            "executed": False,
            "passed": False,
            "reason": "monolithic tangent method gate failed",
        }
    common_passed = bool(ladders["common_mode"]["passed"])

    if common_passed:
        for name in PERTURBATIONS[1:]:
            ladders[name] = _perturbation_ladder(
                name,
                configurations,
                tangents,
                observable_maps,
                physical_scales,
                decisive,
            )
    else:
        for name in PERTURBATIONS[1:]:
            ladders[name] = {
                "executed": False,
                "passed": False,
                "reason": (
                    "binding common-mode uniform physical-export gate "
                    "did not pass"
                ),
            }
    all_ladders_passed = bool(
        common_passed
        and all(ladders[name]["passed"] for name in PERTURBATIONS[1:])
    )
    passed = bool(method_passed and all_ladders_passed)
    classification = (
        "monolithic_uniform_physical_exports_passed_"
        "embedded_export_discrimination_authorized"
        if passed
        else (
            "monolithic_uniform_tangent_method_gate_failed"
            if not method_passed
            else "monolithic_uniform_physical_exports_rejected"
        )
    )

    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)
    source_hashes, source_manifest = _source_manifest()
    gates = {
        "minimum_export_order": MINIMUM_EXPORT_ORDER,
        "maximum_fine_physical_difference": (
            MAXIMUM_FINE_PHYSICAL_DIFFERENCE
        ),
        "minimum_history_cosine": MINIMUM_HISTORY_COSINE,
        "minimum_refinement_error_cosine": MINIMUM_ERROR_COSINE,
        "minimum_relative_activity": MINIMUM_RELATIVE_ACTIVITY,
        "maximum_reconstruction_defect": MAXIMUM_RECONSTRUCTION_DEFECT,
        "maximum_partition_defect": MAXIMUM_PARTITION_DEFECT,
        "maximum_component_defect": MAXIMUM_COMPONENT_DEFECT,
        "maximum_balance_defect": MAXIMUM_BALANCE_DEFECT,
        "maximum_factorization_defect": MAXIMUM_FACTORIZATION_DEFECT,
        "maximum_centered_storage_action_defect": (
            MAXIMUM_CENTERED_STORAGE_ACTION_DEFECT
        ),
        "maximum_directional_stationary_defect": (
            MAXIMUM_DIRECTIONAL_STATIONARY_DEFECT
        ),
        "maximum_directional_storage_rate_defect": (
            MAXIMUM_DIRECTIONAL_STORAGE_RATE_DEFECT
        ),
        "maximum_directional_export_defect": (
            MAXIMUM_DIRECTIONAL_EXPORT_DEFECT
        ),
        "maximum_restart_defect": MAXIMUM_RESTART_DEFECT,
    }
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "labels": LABELS,
        "perturbations": PERTURBATIONS,
        "held_out_definitions": HELD_OUT_DEFINITIONS,
        "path_quadrature_order": PATH_QUADRATURE_ORDER,
        "directional_step": DIRECTIONAL_STEP,
        "time_sample_stride": TIME_SAMPLE_STRIDE,
        "observable_names": OBSERVABLE_NAMES,
        "fixed_physical_scale_rule": (
            "maximum absolute monolithic base mass/angular/energy "
            "observable across all three grids, assigned by physical field"
        ),
        "gates": gates,
    }
    _write_json(CONFIG_PATH, config)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        **identity,
        "classification": classification,
        "passed": passed,
        "method_passed": method_passed,
        "parent_wp10c9d6b_summary_path": _relative(PARENT_SUMMARY),
        "parent_wp10c9d6b_summary_sha256": _sha256(PARENT_SUMMARY),
        "parent_manufactured_preflight_remains_passed": bool(
            parent["method_passed"]
        ),
        "method_reports": method_reports,
        "observable_names": OBSERVABLE_NAMES,
        "fixed_physical_observable_scales": physical_scales,
        "ladders": ladders,
        "common_mode_passed": common_passed,
        "held_out_ladders_passed": bool(
            all_ladders_passed and common_passed
        ),
        "uses_production_generator": False,
        "uses_production_anchor_storage_derivative": False,
        "candidate_base_rate_is_self_consistent": True,
        "center_broken_within_cell_paths": True,
        "embedded_export_discrimination_authorized": passed,
        "nonlinear_physical_trajectory_authorized": False,
        "production_operator_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "replay_contexts_path": _relative(REPLAY_CONTEXTS),
        "replay_contexts_sha256": _sha256(REPLAY_CONTEXTS),
        "replay_inputs_path": _relative(REPLAY_INPUTS),
        "replay_inputs_sha256": _sha256(REPLAY_INPUTS),
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
            "scripts/run_causal_inner_monolithic_uniform_exports_"
            "wp10c9d6c.py"
        ),
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        "parent_canonical_hashes": {
            _relative(path): _sha256(path)
            for path in (PARENT_SUMMARY, PARENT_ARRAYS)
        },
        "replay_source_hashes": replay_payload["source_input_hashes"],
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "scientific_status": "CERTIFIED" if passed else "REJECTED",
        "authorization_status": (
            "embedded_export_discrimination" if passed else "none"
        ),
        "establishes": (
            "A self-consistent production-neutral monolithic frozen tangent "
            "and, conditionally, uniform-grid instantaneous and cumulative "
            "physical-export convergence."
        ),
        "does_not_establish": (
            "Embedded coupling convergence, a nonlinear physical trajectory, "
            "production readiness, fixed-Q closure, or reduced slow "
            "evolution."
        ),
    }
    _write_json(PROVENANCE_PATH, provenance)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepare-replay-inputs",
        action="store_true",
        help="promote the uniform inputs before running the self-contained audit",
    )
    arguments = parser.parse_args()
    print(
        json.dumps(
            _plain(
                run(
                    prepare_replay_inputs=(
                        arguments.prepare_replay_inputs
                    )
                )
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
