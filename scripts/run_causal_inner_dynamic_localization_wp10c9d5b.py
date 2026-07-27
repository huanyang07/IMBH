"""Run the WP10c9d5b nested inner control-volume localization.

This audit reuses the rejected WP10c9d5 candidate histories.  It does not
change the physical operator.  The only authorized question is whether the
mesh defect in the conservative M/J/E exports is confined to a compact inner
control volume, and which exact frozen ledger blocks carry the defect.
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
import scipy
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_frozen_discrimination_wp10c9d5 as wp10c9d5
import run_causal_inner_micro_export_preflight_wp10c9d0 as wp10c9d0

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    causal_five_field_radial_candidate_ledger,
    causal_five_field_radial_reduced_jacobian_pattern,
    causal_five_field_reduced_stationary_residual,
    causal_five_field_reduced_storage_matrices,
    causal_radial_colored_block_jacobians,
    causal_radial_first_consecutive_recovery,
    causal_radial_history_convergence,
    causal_radial_prefix_face_fluxes,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d5b"
ANALYZED_BASE_COMMIT = "cb10412aef66ff5e1e2724f8bd702b2c17a5f734"
ANALYZED_BASE_PARENT = "155e18339076fd2b27d419173b92e1d5d608963b"
ANALYZED_BASE_TREE = "6e41143fd363ecb204dce1f78343535f90ff6898"
THIS_RUNNER = "scripts/run_causal_inner_dynamic_localization_wp10c9d5b.py"

CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_dynamic_localization_wp10c9d5b"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
CACHE_DIRECTORY = (
    ROOT
    / "outputs/checkpoints/"
    "causal_inner_dynamic_localization_wp10c9d5b"
)
D5_CANONICAL_ARRAYS = (
    ROOT
    / "results/canonical/"
    "causal_inner_frozen_discrimination_wp10c9d5/decisive_arrays.npz"
)
D5A1_SUMMARY = (
    ROOT
    / "results/canonical/"
    "causal_inner_frozen_domain_hardening_wp10c9d5a1/summary.json"
)

LABELS = wp10c9d5.PATCH_LABELS
N_FIELDS = 5
CONSERVATIVE_FIELDS = np.asarray((0, 2, 3), dtype=int)
FIELD_NAMES = ("mass", "angular_momentum", "killing_energy")
PATH_QUADRATURE_ORDER = 6
FINITE_DIFFERENCE_STEP = 4.0e-5
SAMPLE_STRIDE = 2
MAXIMUM_BLOCK_CLOSURE_DEFECT = 2.0e-9
MAXIMUM_DESCRIPTOR_CLOSURE_DEFECT = 1.0e-10
MAXIMUM_EXPORT_REPLAY_DEFECT = 5.0e-6
MAXIMUM_CONTROL_VOLUME_CLOSURE_DEFECT = 1.0e-10
MINIMUM_RECOVERY_ORDER = 0.75
MAXIMUM_FINE_NORMALIZED_DIFFERENCE = 0.05
MINIMUM_FINE_SIGNED_COSINE = 0.90
REQUIRED_CONSECUTIVE_RECOVERY_SURFACES = 2
MAXIMUM_RECOVERY_RADIUS_OVER_RG = 5.0
MINIMUM_DOMINANT_BLOCK_FRACTION = 0.50
MINIMUM_DOMINANT_BLOCK_COSINE = 0.90

BLOCK_NAMES = (
    "conservative_transport",
    "shear_principal",
    "height_principal",
    "local_stress_relaxation",
    "geometry",
    "cooling",
    "stream",
    "lower_height_work",
)
BALANCE_BLOCK_NAMES = (
    "inner_shared_face",
    "outer_shared_face",
    "shear_principal",
    "height_principal",
    "local_stress_relaxation",
    "geometry",
    "cooling",
    "stream",
    "lower_height_work",
    "mapped_storage_rate",
    "responsive_height_storage_rate",
    "production_anchor_storage_derivative",
)
IMPLEMENTATION_SOURCES = (
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_radial_localization.py",
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_radial_fluctuation.py",
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_radial_frozen.py",
    "src/imri_qpe/layer3_minidisk_1d/__init__.py",
    THIS_RUNNER,
    "tests/test_causal_inner_radial_localization.py",
    "tests/test_causal_inner_dynamic_localization_wp10c9d5b.py",
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
    return str(path.relative_to(ROOT))


def _git(*args: str) -> str:
    import subprocess

    return subprocess.run(
        ("git", *args),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_git_identity() -> dict:
    commit = _git("rev-parse", ANALYZED_BASE_COMMIT)
    parent = _git("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
    tree = _git("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
    if (commit, parent, tree) != (
        ANALYZED_BASE_COMMIT,
        ANALYZED_BASE_PARENT,
        ANALYZED_BASE_TREE,
    ):
        raise RuntimeError("WP10c9d5b analyzed Git identity changed")
    return {
        "analyzed_base_commit": commit,
        "analyzed_base_parent_commit": parent,
        "analyzed_base_tree_sha": tree,
    }


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


def _refresh_sha256s(directory: Path) -> None:
    members = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.name != "SHA256SUMS.txt"
    )
    (directory / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in members),
        encoding="utf-8",
    )


def _environment() -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "blas_lapack": np.__config__.show(mode="dicts"),
        "float64_epsilon": np.finfo(np.float64).eps,
    }


def _cache_paths(label: str) -> tuple[Path, Path]:
    return (
        CACHE_DIRECTORY / f"{label}.json",
        CACHE_DIRECTORY / f"{label}_arrays.npz",
    )


def _pack_sparse(prefix: str, matrix: csr_matrix) -> dict[str, np.ndarray]:
    value = matrix.tocsr()
    return {
        f"{prefix}_data": np.asarray(value.data, dtype=float),
        f"{prefix}_indices": np.asarray(value.indices, dtype=np.int64),
        f"{prefix}_indptr": np.asarray(value.indptr, dtype=np.int64),
        f"{prefix}_shape": np.asarray(value.shape, dtype=np.int64),
    }


def _unpack_sparse(prefix: str, arrays: dict[str, np.ndarray]) -> csr_matrix:
    shape = tuple(int(value) for value in arrays[f"{prefix}_shape"])
    return csr_matrix(
        (
            arrays[f"{prefix}_data"],
            arrays[f"{prefix}_indices"],
            arrays[f"{prefix}_indptr"],
        ),
        shape=shape,
    )


def _scaled_block_residuals(
    configuration: dict,
    scaled_increment: np.ndarray,
) -> dict[str, np.ndarray]:
    base = np.asarray(configuration["base_primitives"], dtype=float)
    native = configuration["candidate_native"]
    columns = np.asarray(native["primitive_column_scales"], dtype=float)
    rows = np.asarray(native["conservation_row_scales"], dtype=float)
    charts = base.ravel() + columns * np.asarray(
        scaled_increment,
        dtype=float,
    )
    ledger = causal_five_field_radial_candidate_ledger(
        configuration["context"],
        charts.reshape(base.shape),
        quadrature_order=PATH_QUADRATURE_ORDER,
    )
    result = {
        "conservative_transport": ledger.conservative_transport_rows,
        "shear_principal": ledger.shear_principal_rows,
        "height_principal": ledger.height_principal_rows,
        "local_stress_relaxation": (
            ledger.local_stress_relaxation_rows
        ),
        "geometry": ledger.geometry_rows,
        "cooling": ledger.cooling_rows,
        "stream": ledger.stream_rows,
        "lower_height_work": ledger.lower_height_work_rows,
        "production": causal_five_field_reduced_stationary_residual(
            charts,
            configuration["context"],
        ).reshape(base.shape),
    }
    return {
        name: np.asarray(values, dtype=float).ravel() / rows
        for name, values in result.items()
    }


def _build_or_load_blocks(
    configuration: dict,
    *,
    force: bool,
) -> tuple[dict, dict[str, csr_matrix], dict[str, np.ndarray]]:
    label = str(configuration["label"])
    json_path, arrays_path = _cache_paths(label)
    base = np.asarray(configuration["base_primitives"], dtype=float)
    native = configuration["candidate_native"]
    contract = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "label": label,
        "base_primitives_sha256": _array_sha256(base),
        "grid_edges_sha256": _array_sha256(
            configuration["context"].grid.edges
        ),
        "candidate_generator_sha256": _array_sha256(
            native["candidate_generator"]
        ),
        "stationary_delta_sha256": _array_sha256(
            native["stationary_delta"]
        ),
        "finite_difference_step": FINITE_DIFFERENCE_STEP,
        "path_quadrature_order": PATH_QUADRATURE_ORDER,
    }
    if not force and json_path.exists() and arrays_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if (
            all(payload.get(key) == value for key, value in contract.items())
            and payload.get("arrays_sha256") == _sha256(arrays_path)
        ):
            with np.load(arrays_path, allow_pickle=False) as source:
                packed = {
                    name: np.asarray(source[name]) for name in source.files
                }
            matrices = {
                name: _unpack_sparse(name, packed)
                for name in (*BLOCK_NAMES, "production")
            }
            dense = {
                name: np.asarray(packed[name], dtype=float)
                for name in (
                    "mapped_descriptor",
                    "vertical_descriptor",
                    "anchor_storage_derivative",
                )
            }
            return payload, matrices, dense

    print(f"WP10c9d5b: building block Jacobians for {label}", flush=True)
    started = time.perf_counter()
    n_cells = int(base.shape[0])
    pattern = causal_five_field_radial_reduced_jacobian_pattern(n_cells)
    zero = np.zeros(base.size, dtype=float)
    matrices = causal_radial_colored_block_jacobians(
        lambda values: _scaled_block_residuals(
            configuration,
            values,
        ),
        zero,
        pattern,
        finite_difference_step=FINITE_DIFFERENCE_STEP,
    )
    candidate_stationary = sum(
        (matrices[name] for name in BLOCK_NAMES),
        start=csr_matrix(pattern.shape, dtype=float),
    )
    stationary_delta = (
        candidate_stationary - matrices["production"]
    ).toarray()
    expected_delta = np.asarray(native["stationary_delta"], dtype=float)
    delta_scale = max(
        float(np.max(np.abs(expected_delta))),
        np.finfo(float).tiny,
    )
    block_closure = float(
        np.max(np.abs(stationary_delta - expected_delta)) / delta_scale
    )
    storage = causal_five_field_reduced_storage_matrices(
        configuration["context"],
        base.ravel(),
        primitive_column_scales=np.asarray(
            native["primitive_column_scales"],
            dtype=float,
        ),
        conservation_row_scales=np.asarray(
            native["conservation_row_scales"],
            dtype=float,
        ),
    )
    mapped = np.asarray(
        storage["conserved_descriptor_reduced_scaled_matrix"],
        dtype=float,
    )
    vertical = np.asarray(
        storage["vertical_descriptor_reduced_scaled_matrix"],
        dtype=float,
    )
    descriptor = np.asarray(native["descriptor"], dtype=float)
    descriptor_scale = max(
        float(np.max(np.abs(descriptor))),
        np.finfo(float).tiny,
    )
    descriptor_closure = float(
        np.max(np.abs(mapped + vertical - descriptor))
        / descriptor_scale
    )
    candidate_generator = np.asarray(
        native["candidate_generator"],
        dtype=float,
    )
    anchor = -(
        descriptor @ candidate_generator
        + candidate_stationary.toarray()
    )
    packed: dict[str, np.ndarray] = {}
    for name, matrix in matrices.items():
        packed.update(_pack_sparse(name, matrix))
    packed.update(
        {
            "mapped_descriptor": mapped,
            "vertical_descriptor": vertical,
            "anchor_storage_derivative": anchor,
        }
    )
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **packed)
    payload = {
        **contract,
        "arrays_path": _relative(arrays_path),
        "arrays_sha256": _sha256(arrays_path),
        "n_cells": n_cells,
        "block_names": BLOCK_NAMES,
        "maximum_stationary_block_closure_defect": block_closure,
        "maximum_descriptor_component_closure_defect": descriptor_closure,
        "wall_seconds": time.perf_counter() - started,
        "passed": bool(
            block_closure <= MAXIMUM_BLOCK_CLOSURE_DEFECT
            and descriptor_closure <= MAXIMUM_DESCRIPTOR_CLOSURE_DEFECT
        ),
    }
    _write_json(json_path, payload)
    return payload, matrices, {
        "mapped_descriptor": mapped,
        "vertical_descriptor": vertical,
        "anchor_storage_derivative": anchor,
    }


def _sampled_history(
    configuration: dict,
    matrices: dict[str, csr_matrix],
    dense: dict[str, np.ndarray],
    d5_signals: np.ndarray,
) -> dict[str, np.ndarray]:
    state, rate, _restart = wp10c9d5._propagate(
        configuration["generator"],
        configuration["initial"],
        configuration["times"],
    )
    indices = np.arange(0, state.shape[0], SAMPLE_STRIDE, dtype=int)
    if indices[-1] != state.shape[0] - 1:
        indices = np.append(indices, state.shape[0] - 1)
    state = np.asarray(state[indices], dtype=float)
    rate = np.asarray(rate[indices], dtype=float)
    times = np.asarray(configuration["times"], dtype=float)[indices]
    amplitudes = np.asarray(configuration["amplitudes"], dtype=float)
    native = configuration["candidate_native"]
    column_scales = np.asarray(
        native["primitive_column_scales"],
        dtype=float,
    )
    row_scales = np.asarray(
        native["conservation_row_scales"],
        dtype=float,
    )
    scaled_state = (
        state * amplitudes[None, :, :]
    ).reshape(state.shape[0], -1) / column_scales[None, :]
    scaled_rate = (
        rate * amplitudes[None, :, :]
    ).reshape(rate.shape[0], -1) / column_scales[None, :]
    n_cells = int(state.shape[1])

    def physical_rows(matrix, values: np.ndarray) -> np.ndarray:
        return (
            np.asarray([matrix @ vector for vector in values], dtype=float)
            * row_scales[None, :]
        ).reshape(values.shape[0], n_cells, N_FIELDS)

    block_rows = {
        name: physical_rows(matrices[name], scaled_state)
        for name in BLOCK_NAMES
    }
    mapped_storage_state = physical_rows(
        dense["mapped_descriptor"],
        scaled_state,
    )
    vertical_storage_state = physical_rows(
        dense["vertical_descriptor"],
        scaled_state,
    )
    mapped_storage_rate = physical_rows(
        dense["mapped_descriptor"],
        scaled_rate,
    )
    vertical_storage_rate = physical_rows(
        dense["vertical_descriptor"],
        scaled_rate,
    )
    anchor = physical_rows(
        dense["anchor_storage_derivative"],
        scaled_state,
    )
    inner = np.asarray(d5_signals[:, :3], dtype=float)
    face_fluxes = causal_radial_prefix_face_fluxes(
        inner,
        block_rows["conservative_transport"][
            :, :, CONSERVATIVE_FIELDS
        ],
    )
    active = int(configuration["active_cells"])
    coupling_reference_scale = float(
        np.max(np.abs(d5_signals[:, 3:6]))
    )
    inner_reference_scale = max(
        float(np.max(np.abs(inner))),
        np.finfo(float).tiny,
    )
    coupling_replay_applicable = bool(
        coupling_reference_scale
        > 1.0e-12 * inner_reference_scale
    )
    coupling_replay = (
        float(
            np.max(
                np.abs(face_fluxes[:, active] - d5_signals[:, 3:6])
            )
            / coupling_reference_scale
        )
        if coupling_replay_applicable
        else 0.0
    )
    candidate_stationary = sum(block_rows.values())
    net_replay = float(
        np.max(
            np.abs(
                -np.sum(
                    candidate_stationary[
                        :, :active, :
                    ][:, :, CONSERVATIVE_FIELDS],
                    axis=1,
                )
                - d5_signals[:, 6:9]
            )
        )
        / max(
            float(np.max(np.abs(d5_signals[:, 6:9]))),
            np.finfo(float).tiny,
        )
    )
    full_balance = (
        mapped_storage_rate
        + vertical_storage_rate
        + anchor
        + candidate_stationary
    )
    balance_scale = max(
        float(
            np.max(
                np.abs(
                    mapped_storage_rate
                    + vertical_storage_rate
                    + anchor
                )
            )
        ),
        float(np.max(np.abs(candidate_stationary))),
        np.finfo(float).tiny,
    )
    closure = float(np.max(np.abs(full_balance)) / balance_scale)
    return {
        "times": times,
        "state": state,
        "rate": rate,
        "face_fluxes": face_fluxes,
        "mapped_storage_state": mapped_storage_state,
        "vertical_storage_state": vertical_storage_state,
        "mapped_storage_rate": mapped_storage_rate,
        "vertical_storage_rate": vertical_storage_rate,
        "production_anchor_storage_derivative": anchor,
        "candidate_stationary": candidate_stationary,
        "coupling_export_replay_defect": coupling_replay,
        "coupling_export_replay_applicable": (
            coupling_replay_applicable
        ),
        "net_export_replay_defect": net_replay,
        "control_volume_closure_defect": closure,
        **{f"{name}_rows": values for name, values in block_rows.items()},
    }


def _cumulative(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values)
    increments = 0.5 * np.diff(times)[:, None] * (
        values[:-1] + values[1:]
    )
    result[1:] = np.cumsum(increments, axis=0)
    return result


def _metrics_payload(metrics) -> dict:
    return {
        "component_scales": metrics.component_scales,
        "significant_components": metrics.significant_components,
        "component_coarse_medium_differences": (
            metrics.component_coarse_medium_differences
        ),
        "component_medium_fine_differences": (
            metrics.component_medium_fine_differences
        ),
        "component_observed_orders": metrics.component_observed_orders,
        "component_fine_signed_cosines": (
            metrics.component_fine_signed_cosines
        ),
        "component_passed": metrics.component_passed,
        "coarse_medium_difference": metrics.coarse_medium_difference,
        "medium_fine_difference": metrics.medium_fine_difference,
        "observed_order": metrics.observed_order,
        "fine_signed_cosine": metrics.fine_signed_cosine,
        "passed": metrics.passed,
    }


def _common_face_indices(configurations: dict) -> tuple[np.ndarray, dict]:
    coarse = configurations[LABELS[0]]
    edges_over_rg = np.asarray(
        coarse["operator"]["grid_edges_rg"],
        dtype=float,
    )
    last = int(
        np.flatnonzero(
            edges_over_rg <= MAXIMUM_RECOVERY_RADIUS_OVER_RG
        )[-1]
    )
    coarse_faces = np.arange(last + 1, dtype=int)
    maps = {
        LABELS[0]: coarse_faces,
        LABELS[1]: 2 * coarse_faces,
        LABELS[2]: 4 * coarse_faces,
    }
    for label, indices in maps.items():
        label_edges_over_rg = np.asarray(
            configurations[label]["operator"]["grid_edges_rg"],
            dtype=float,
        )
        if not np.allclose(
            label_edges_over_rg[indices],
            edges_over_rg[coarse_faces],
            rtol=5.0e-14,
            atol=0.0,
        ):
            raise RuntimeError("embedded grids do not share declared faces")
    return edges_over_rg[coarse_faces], maps


def _prefix(values: np.ndarray, face: int) -> np.ndarray:
    return np.sum(
        np.asarray(values, dtype=float)[
            :, :int(face), :
        ][:, :, CONSERVATIVE_FIELDS],
        axis=1,
    )


def _balance_histories(history: dict, face: int) -> dict[str, np.ndarray]:
    return {
        "inner_shared_face": -history["face_fluxes"][:, 0],
        "outer_shared_face": history["face_fluxes"][:, face],
        "shear_principal": _prefix(
            history["shear_principal_rows"],
            face,
        ),
        "height_principal": _prefix(
            history["height_principal_rows"],
            face,
        ),
        "local_stress_relaxation": _prefix(
            history["local_stress_relaxation_rows"],
            face,
        ),
        "geometry": _prefix(history["geometry_rows"], face),
        "cooling": _prefix(history["cooling_rows"], face),
        "stream": _prefix(history["stream_rows"], face),
        "lower_height_work": _prefix(
            history["lower_height_work_rows"],
            face,
        ),
        "mapped_storage_rate": _prefix(
            history["mapped_storage_rate"],
            face,
        ),
        "responsive_height_storage_rate": _prefix(
            history["vertical_storage_rate"],
            face,
        ),
        "production_anchor_storage_derivative": _prefix(
            history["production_anchor_storage_derivative"],
            face,
        ),
    }


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(
        np.linalg.norm(first) * np.linalg.norm(second)
    )
    if denominator <= np.finfo(float).tiny:
        return 1.0
    return float(np.dot(first.ravel(), second.ravel()) / denominator)


def _dominance_at_face(
    histories: dict,
    face_map: dict[str, int],
) -> dict:
    blocks = {
        label: _balance_histories(histories[label], face_map[label])
        for label in LABELS
    }
    pair_names = ("coarse_medium", "medium_fine")
    pairs = ((LABELS[0], LABELS[1]), (LABELS[1], LABELS[2]))
    norms: dict[str, dict[str, float]] = {}
    differences: dict[str, dict[str, np.ndarray]] = {}
    for pair_name, (left, right) in zip(pair_names, pairs, strict=True):
        differences[pair_name] = {
            name: blocks[right][name] - blocks[left][name]
            for name in BALANCE_BLOCK_NAMES
        }
        squared = {
            name: float(np.linalg.norm(values) ** 2)
            for name, values in differences[pair_name].items()
        }
        total = max(sum(squared.values()), np.finfo(float).tiny)
        norms[pair_name] = {
            name: value / total for name, value in squared.items()
        }
    selected = {
        pair: max(values, key=values.get)
        for pair, values in norms.items()
    }
    same = selected[pair_names[0]] == selected[pair_names[1]]
    name = selected[pair_names[0]] if same else None
    cosine = (
        _cosine(
            differences[pair_names[0]][name],
            differences[pair_names[1]][name],
        )
        if name is not None
        else 0.0
    )
    stable = bool(
        name is not None
        and all(
            norms[pair][name] >= MINIMUM_DOMINANT_BLOCK_FRACTION
            for pair in pair_names
        )
        and abs(cosine) >= MINIMUM_DOMINANT_BLOCK_COSINE
    )
    return {
        "fractions": norms,
        "selected_blocks": selected,
        "stable_selected_block": name if stable else None,
        "cross_pair_signed_cosine": cosine,
        "passed": stable,
    }


def run(*, force_blocks: bool = False) -> dict:
    started = time.perf_counter()
    identity = _validate_git_identity()
    d5a1 = json.loads(D5A1_SUMMARY.read_text(encoding="utf-8"))
    if not d5a1["wp10c9d5b_inner_localization_authorized"]:
        raise RuntimeError("WP10c9d5a1 did not authorize localization")
    configurations = {
        label: configuration
        for label, configuration
        in wp10c9d5._common_configurations(False).items()
        if label in LABELS
    }
    with np.load(D5_CANONICAL_ARRAYS, allow_pickle=False) as source:
        d5_arrays = {
            name: np.asarray(source[name]) for name in source.files
        }

    cache_reports = {}
    histories = {}
    for label in LABELS:
        report, matrices, dense = _build_or_load_blocks(
            configurations[label],
            force=force_blocks,
        )
        cache_reports[label] = report
        histories[label] = _sampled_history(
            configurations[label],
            matrices,
            dense,
            d5_arrays[f"{label}_signals"],
        )

    common_radii, face_maps = _common_face_indices(configurations)
    surface_reports = []
    surface_passes = []
    dominance_reports = []
    decisive: dict[str, np.ndarray] = {
        "common_face_radii_over_rg": common_radii,
    }
    for label in LABELS:
        history = histories[label]
        decisive[f"{label}_times"] = history["times"]
        decisive[f"{label}_face_fluxes"] = history["face_fluxes"][
            :, face_maps[label], :
        ]
        decisive[f"{label}_mapped_storage_rate"] = (
            history["mapped_storage_rate"][
                :, : int(face_maps[label][-1]), :
            ][:, :, CONSERVATIVE_FIELDS]
        )
        decisive[f"{label}_vertical_storage_rate"] = (
            history["vertical_storage_rate"][
                :, : int(face_maps[label][-1]), :
            ][:, :, CONSERVATIVE_FIELDS]
        )
        decisive[f"{label}_anchor_storage_derivative"] = (
            history["production_anchor_storage_derivative"][
                :, : int(face_maps[label][-1]), :
            ][:, :, CONSERVATIVE_FIELDS]
        )
        for name in BALANCE_BLOCK_NAMES:
            decisive[f"{label}_prefix_{name}"] = np.asarray(
                [
                    _balance_histories(history, int(face))[name]
                    for face in face_maps[label]
                ],
                dtype=float,
            )
    for surface, radius in enumerate(common_radii):
        face_map = {
            label: int(face_maps[label][surface])
            for label in LABELS
        }
        instant = {
            label: histories[label]["face_fluxes"][:, face_map[label], :]
            for label in LABELS
        }
        cumulative = {
            label: _cumulative(histories[label]["times"], instant[label])
            for label in LABELS
        }
        instant_metrics = causal_radial_history_convergence(
            *(instant[label] for label in LABELS),
            minimum_order=MINIMUM_RECOVERY_ORDER,
            maximum_fine_normalized_difference=(
                MAXIMUM_FINE_NORMALIZED_DIFFERENCE
            ),
            minimum_fine_signed_cosine=MINIMUM_FINE_SIGNED_COSINE,
        )
        cumulative_metrics = causal_radial_history_convergence(
            *(cumulative[label] for label in LABELS),
            minimum_order=MINIMUM_RECOVERY_ORDER,
            maximum_fine_normalized_difference=(
                MAXIMUM_FINE_NORMALIZED_DIFFERENCE
            ),
            minimum_fine_signed_cosine=MINIMUM_FINE_SIGNED_COSINE,
        )
        passed = bool(instant_metrics.passed and cumulative_metrics.passed)
        surface_passes.append(passed)
        surface_reports.append(
            {
                "surface": surface,
                "radius_over_rg": float(radius),
                "face_indices": face_map,
                "instantaneous": _metrics_payload(instant_metrics),
                "cumulative": _metrics_payload(cumulative_metrics),
                "passed": passed,
            }
        )
        dominance_reports.append(
            {
                "surface": surface,
                "radius_over_rg": float(radius),
                **_dominance_at_face(histories, face_map),
            }
        )

    recovery_index = causal_radial_first_consecutive_recovery(
        np.asarray(surface_passes, dtype=bool),
        required_consecutive=REQUIRED_CONSECUTIVE_RECOVERY_SURFACES,
    )
    recovery_radius = (
        float(common_radii[recovery_index])
        if recovery_index is not None
        else None
    )
    stable_dominant_block = None
    stable_dominance_start = None
    for start in range(1, len(dominance_reports) - 1):
        first = dominance_reports[start]
        second = dominance_reports[start + 1]
        if (
            first["passed"]
            and second["passed"]
            and first["stable_selected_block"]
            == second["stable_selected_block"]
        ):
            stable_dominant_block = first["stable_selected_block"]
            stable_dominance_start = start
            break
    if recovery_index is not None:
        branch = "A_compact_recovery_radius"
        authorized_next = "conservative_extraction_surface_audit"
    elif stable_dominant_block in {
        "inner_shared_face",
        "outer_shared_face",
        "conservative_transport",
    }:
        branch = "B_first_face_or_first_cell_dominance"
        authorized_next = "boundary_compatible_half_cell_audit"
    elif stable_dominant_block in {
        "mapped_storage_rate",
        "responsive_height_storage_rate",
        "production_anchor_storage_derivative",
    }:
        branch = "C_descriptor_dominance"
        authorized_next = "self_consistent_space_storage_tangent_audit"
    else:
        branch = "D_no_compact_recovery_or_stable_dominant_term"
        authorized_next = "reject_current_embedded_microclosure_discretization"

    maximum_block_closure = max(
        float(report["maximum_stationary_block_closure_defect"])
        for report in cache_reports.values()
    )
    maximum_descriptor_closure = max(
        float(report["maximum_descriptor_component_closure_defect"])
        for report in cache_reports.values()
    )
    maximum_export_replay = max(
        max(
            float(history["coupling_export_replay_defect"]),
            float(history["net_export_replay_defect"]),
        )
        for history in histories.values()
    )
    maximum_control_volume_closure = max(
        float(history["control_volume_closure_defect"])
        for history in histories.values()
    )
    method_passed = bool(
        all(report["passed"] for report in cache_reports.values())
        and maximum_block_closure <= MAXIMUM_BLOCK_CLOSURE_DEFECT
        and maximum_descriptor_closure
        <= MAXIMUM_DESCRIPTOR_CLOSURE_DEFECT
        and maximum_export_replay <= MAXIMUM_EXPORT_REPLAY_DEFECT
        and maximum_control_volume_closure
        <= MAXIMUM_CONTROL_VOLUME_CLOSURE_DEFECT
    )
    if not method_passed:
        branch = "METHOD_GATE_FAILED"
        authorized_next = "none"

    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)
    source_hashes, source_manifest = _source_manifest()
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "labels": LABELS,
        "finite_difference_step": FINITE_DIFFERENCE_STEP,
        "path_quadrature_order": PATH_QUADRATURE_ORDER,
        "sample_stride": SAMPLE_STRIDE,
        "maximum_recovery_radius_over_rg": (
            MAXIMUM_RECOVERY_RADIUS_OVER_RG
        ),
        "minimum_recovery_order": MINIMUM_RECOVERY_ORDER,
        "maximum_fine_normalized_difference": (
            MAXIMUM_FINE_NORMALIZED_DIFFERENCE
        ),
        "minimum_fine_signed_cosine": MINIMUM_FINE_SIGNED_COSINE,
        "required_consecutive_recovery_surfaces": (
            REQUIRED_CONSECUTIVE_RECOVERY_SURFACES
        ),
    }
    _write_json(CONFIG_PATH, config)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        **identity,
        "method_scope": (
            "CACHE-FIRST FROZEN NESTED CONTROL-VOLUME LOCALIZATION / "
            "PRODUCTION NEUTRAL"
        ),
        "parent_wp10c9d5_candidate_remains_rejected": True,
        "parent_wp10c9d5a_global_hardening_remains_rejected": True,
        "parent_wp10c9d5a1_inner_authorization_preserved": True,
        "cache_reports": cache_reports,
        "maximum_stationary_block_closure_defect": maximum_block_closure,
        "maximum_descriptor_component_closure_defect": (
            maximum_descriptor_closure
        ),
        "maximum_export_replay_defect": maximum_export_replay,
        "maximum_control_volume_closure_defect": (
            maximum_control_volume_closure
        ),
        "method_passed": method_passed,
        "surface_reports": surface_reports,
        "surface_passes": surface_passes,
        "recovery_surface_index": recovery_index,
        "recovery_radius_over_rg": recovery_radius,
        "dominance_reports": dominance_reports,
        "stable_dominant_block": stable_dominant_block,
        "stable_dominance_start_surface": stable_dominance_start,
        "coupling_export_replay_applicable": {
            label: bool(
                histories[label]["coupling_export_replay_applicable"]
            )
            for label in LABELS
        },
        "binding_branch": branch,
        "authorized_next_work": authorized_next,
        "frozen_candidate_recertification_authorized": False,
        "nonlinear_candidate_authorized": False,
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
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        "generation_command": (
            "PYTHONPATH=src python3 "
            "scripts/run_causal_inner_dynamic_localization_wp10c9d5b.py"
        ),
        "method_scope": summary["method_scope"],
        "scientific_status": (
            "DIAGNOSTIC ONLY" if method_passed else "REJECTED"
        ),
        "authorization_status": authorized_next,
        "source_input_hashes": {
            _relative(D5_CANONICAL_ARRAYS): _sha256(
                D5_CANONICAL_ARRAYS
            ),
            _relative(D5A1_SUMMARY): _sha256(D5A1_SUMMARY),
        },
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "establishes": (
            "Whether the frozen M/J/E face histories recover spatial "
            "contraction outside a compact inner layer and which exact "
            "control-volume ledger blocks carry the refinement defect."
        ),
        "does_not_establish": (
            "A repaired boundary operator, frozen recertification, nonlinear "
            "convergence, fixed-Q closure, or reduced slow evolution."
        ),
    }
    _write_json(PROVENANCE_PATH, provenance)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-blocks", action="store_true")
    args = parser.parse_args()
    result = run(force_blocks=args.force_blocks)
    print(json.dumps(_plain(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
