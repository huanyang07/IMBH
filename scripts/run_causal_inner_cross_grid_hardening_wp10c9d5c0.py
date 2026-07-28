#!/usr/bin/env python3
"""Run WP10c9d5c0 cross-grid derivative and metric hardening.

This production-neutral audit preserves the rejected WP10c9d5/WP10c9d5b
evidence.  It asks whether the frozen candidate derivative is robust on all
three embedded grids through the still-refined domain, whether the physical
export conclusion is insensitive to the accepted derivative construction,
and whether the reconstructed face histories agree with direct face-flux
directional derivatives.
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
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import splu


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_causal_inner_dynamic_localization_wp10c9d5b as wp10c9d5b
import run_causal_inner_frozen_discrimination_wp10c9d5 as wp10c9d5
import run_causal_inner_micro_export_preflight_wp10c9d0 as wp10c9d0

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    causal_five_field_radial_candidate_ledger,
    causal_five_field_radial_reduced_jacobian_pattern,
    causal_radial_colored_block_jacobians,
    causal_radial_first_consecutive_recovery,
    causal_radial_history_convergence,
    causal_radial_jvp_step_sweep,
    causal_radial_prefix_face_fluxes,
    causal_radial_volume_weighted_scaled_direction,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d5c0"
ANALYZED_BASE_COMMIT = "9c2a4ac6fa464a43fbaed3318cf5e1233a70fe55"
ANALYZED_BASE_PARENT = "cb10412aef66ff5e1e2724f8bd702b2c17a5f734"
ANALYZED_BASE_TREE = "aa50e90606e7224dc57849834584f4b6a1d06fb1"
THIS_RUNNER = (
    "scripts/run_causal_inner_cross_grid_hardening_wp10c9d5c0.py"
)

CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_cross_grid_hardening_wp10c9d5c0"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
CACHE_DIRECTORY = (
    ROOT
    / "outputs/checkpoints/"
    "causal_inner_cross_grid_hardening_wp10c9d5c0"
)
D5B_SUMMARY = (
    ROOT
    / "results/canonical/"
    "causal_inner_dynamic_localization_wp10c9d5b/summary.json"
)

LABELS = wp10c9d5.PATCH_LABELS
N_FIELDS = 5
CONSERVATIVE_FIELDS = np.asarray((0, 2, 3), dtype=int)
FIELD_NAMES = ("mass", "angular_momentum", "killing_energy")
BLOCK_NAMES = wp10c9d5b.BLOCK_NAMES
PATH_QUADRATURE_ORDER = 6
STENCIL_HALO_CELLS = 3
TARGET_RADII_OVER_RG = (5.0, 8.0, 12.0)
DIRECTIONAL_STEPS = (5.0e-6, 1.0e-5, 2.0e-5, 4.0e-5, 8.0e-5)
FINE_EXTRAPOLATION_STEP = 1.0e-5
COARSE_EXTRAPOLATION_STEP = 2.0e-5
STORED_MATRIX_STEP = 4.0e-5
ALTERNATIVE_MATRIX_STEP = 2.0e-5
DIRECTION_SEEDS = (92051, 92052, 92053, 92054)
INNER_FLUX_STENCIL_CELLS = 4
SENSITIVITY_SAMPLE_STRIDE = 2
STRIDE_AUDITS = (1, 2, 4)
DIRECT_FACE_TARGETS_OVER_RG = (3.0, 5.0, 8.0, 12.0)
DIRECT_FACE_TIME_FRACTIONS = (0.0, 0.5, 1.0)

MAXIMUM_SELECTED_MATRIX_DEFECT = 5.0e-5
MAXIMUM_BRACKETING_CENTRAL_CHANGE = 2.0e-5
MAXIMUM_EXTRAPOLATED_JVP_DIFFERENCE = 2.0e-5
MAXIMUM_DERIVATIVE_EXPORT_DIFFERENCE = 5.0e-3
MAXIMUM_DERIVATIVE_TO_SPATIAL_RATIO = 0.10
MAXIMUM_DIRECT_FACE_PARITY_DEFECT = 5.0e-5
MAXIMUM_STRIDE_DEFECT = 5.0e-3
MINIMUM_RELATIVE_ACTIVITY = 1.0e-8
MINIMUM_RECOVERY_ORDER = 0.75
MAXIMUM_FINE_NORMALIZED_DIFFERENCE = 0.05
MINIMUM_HISTORY_COSINE = 0.90
MINIMUM_ERROR_COSINE = 0.90
REQUIRED_CONSECUTIVE_RECOVERY_SURFACES = 2

METHOD_NAMES = ("stored_4e5", "central_2e5", "richardson_2e5_4e5")
IMPLEMENTATION_SOURCES = (
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_radial_localization.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_radial_hardening.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_radial_domain_hardening.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_radial_fluctuation.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_radial_frozen.py",
    "src/imri_qpe/layer3_minidisk_1d/__init__.py",
    THIS_RUNNER,
    "tests/test_causal_inner_radial_localization.py",
    "tests/test_causal_inner_cross_grid_hardening_wp10c9d5c0.py",
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
    return str(path.resolve().relative_to(ROOT.resolve()))


def _refresh_sha256s(directory: Path) -> None:
    entries = []
    for path in sorted(directory.iterdir()):
        if path.name == "SHA256SUMS.txt" or not path.is_file():
            continue
        entries.append(f"{_sha256(path)}  {path.name}")
    (directory / "SHA256SUMS.txt").write_text(
        "\n".join(entries) + "\n",
        encoding="utf-8",
    )


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_analyzed_git_identity() -> dict:
    commit = _git("rev-parse", ANALYZED_BASE_COMMIT)
    parent = _git("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
    tree = _git("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
    if (commit, parent, tree) != (
        ANALYZED_BASE_COMMIT,
        ANALYZED_BASE_PARENT,
        ANALYZED_BASE_TREE,
    ):
        raise RuntimeError("WP10c9d5c0 analyzed Git identity changed")
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


def _support_bump(coordinate: np.ndarray) -> np.ndarray:
    values = np.asarray(coordinate, dtype=float)
    clipped = np.clip(values, 0.0, 1.0)
    bump = np.sin(np.pi * clipped) ** 4
    return np.where((values > 0.0) & (values < 1.0), bump, 0.0)


def _continuum_direction(
    configuration: dict,
    *,
    seed: int,
    lower_radius_over_rg: float,
    upper_radius_over_rg: float,
) -> np.ndarray:
    context = configuration["context"]
    radius = (
        np.asarray(context.grid.centers, dtype=float)
        / float(context.grid.gravitational_radius)
    )
    coordinate = (
        np.log(radius) - np.log(float(lower_radius_over_rg))
    ) / (
        np.log(float(upper_radius_over_rg))
        - np.log(float(lower_radius_over_rg))
    )
    bump = _support_bump(coordinate)
    rng = np.random.default_rng(int(seed))
    values = np.zeros((radius.size, N_FIELDS), dtype=float)
    for field in range(N_FIELDS):
        coefficients = rng.standard_normal((2, 4))
        for mode in range(1, 5):
            values[:, field] += bump * (
                coefficients[0, mode - 1]
                * np.sin(mode * np.pi * coordinate)
                + coefficients[1, mode - 1]
                * np.cos(mode * np.pi * coordinate)
            )
    return causal_radial_volume_weighted_scaled_direction(
        values,
        np.asarray(context.grid.cell_measures, dtype=float),
    ).ravel()


def _common_direction(configuration: dict) -> np.ndarray:
    native = configuration["candidate_native"]
    direction = (
        np.asarray(configuration["amplitudes"], dtype=float)
        * np.asarray(configuration["initial"], dtype=float)
    ).reshape(-1) / np.asarray(
        native["primitive_column_scales"],
        dtype=float,
    )
    return causal_radial_volume_weighted_scaled_direction(
        direction.reshape(-1, N_FIELDS),
        np.asarray(
            configuration["context"].grid.cell_measures,
            dtype=float,
        ),
    ).ravel()


def _directions(configuration: dict) -> dict[str, np.ndarray]:
    result = {
        "common_mode": _common_direction(configuration),
        "global_inner_0": _continuum_direction(
            configuration,
            seed=DIRECTION_SEEDS[0],
            lower_radius_over_rg=1.8,
            upper_radius_over_rg=11.75,
        ),
        "global_inner_1": _continuum_direction(
            configuration,
            seed=DIRECTION_SEEDS[1],
            lower_radius_over_rg=1.8,
            upper_radius_over_rg=11.75,
        ),
        "near_excision_0": _continuum_direction(
            configuration,
            seed=DIRECTION_SEEDS[2],
            lower_radius_over_rg=1.8,
            upper_radius_over_rg=3.5,
        ),
        "near_excision_1": _continuum_direction(
            configuration,
            seed=DIRECTION_SEEDS[3],
            lower_radius_over_rg=1.8,
            upper_radius_over_rg=3.5,
        ),
    }
    for field in range(N_FIELDS):
        direction = np.zeros(
            np.asarray(configuration["base_primitives"]).size,
            dtype=float,
        )
        direction[field] = 1.0
        result[f"first_cell_field_{field}"] = direction
    return result


def _scaled_delta(configuration: dict, increment: np.ndarray) -> np.ndarray:
    blocks = wp10c9d5b._scaled_block_residuals(
        configuration,
        increment,
    )
    candidate = sum(
        (np.asarray(blocks[name], dtype=float) for name in BLOCK_NAMES),
        start=np.zeros_like(np.asarray(blocks["production"], dtype=float)),
    )
    return candidate - np.asarray(blocks["production"], dtype=float)


def _actual_faces(configuration: dict) -> dict[float, int]:
    context = configuration["context"]
    edges = (
        np.asarray(context.grid.edges, dtype=float)
        / float(context.grid.gravitational_radius)
    )
    return {
        target: int(np.flatnonzero(edges <= float(target))[-1])
        for target in TARGET_RADII_OVER_RG
    }


def _region_rows(
    configuration: dict,
    face: int,
    *,
    halo: bool,
) -> np.ndarray:
    n_cells = int(np.asarray(configuration["base_primitives"]).shape[0])
    stop = int(face) + (STENCIL_HALO_CELLS if halo else 0)
    stop = min(stop, n_cells)
    return np.arange(N_FIELDS * stop, dtype=int)


def _relative_difference(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    scale = max(
        float(np.linalg.norm(left)),
        float(np.linalg.norm(right)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(left - right) / scale)


def _directional_report(
    configuration: dict,
    direction: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    native = configuration["candidate_native"]
    matrix = np.asarray(native["stationary_delta"], dtype=float)
    zero = np.zeros(matrix.shape[1], dtype=float)
    sweep = causal_radial_jvp_step_sweep(
        lambda values: _scaled_delta(configuration, values),
        zero,
        matrix,
        direction,
        DIRECTIONAL_STEPS,
        selected_step=STORED_MATRIX_STEP,
    )
    steps = np.asarray(DIRECTIONAL_STEPS, dtype=float)
    indices = {
        float(step): int(np.flatnonzero(steps == float(step))[0])
        for step in steps
    }
    reports = {}
    arrays: dict[str, np.ndarray] = {
        "direction": np.asarray(direction, dtype=float),
        "direct_actions": sweep.direct_actions,
        "stored_matrix_action": sweep.matrix_action,
    }
    for target, face in _actual_faces(configuration).items():
        for halo in (False, True):
            region_name = (
                f"through_{target:g}rg"
                + ("_plus_halo" if halo else "")
            )
            rows = _region_rows(configuration, face, halo=halo)
            direct = sweep.direct_actions[:, rows]
            matrix_action = sweep.matrix_action[rows]
            d1 = direct[indices[FINE_EXTRAPOLATION_STEP]]
            d2 = direct[indices[COARSE_EXTRAPOLATION_STEP]]
            d4 = direct[indices[STORED_MATRIX_STEP]]
            richardson_fine = (4.0 * d1 - d2) / 3.0
            richardson_coarse = (4.0 * d2 - d4) / 3.0
            matrix_defect = _relative_difference(matrix_action, d4)
            bracketing_change = _relative_difference(d2, d4)
            extrapolated_difference = _relative_difference(
                richardson_fine,
                richardson_coarse,
            )
            passed = bool(
                matrix_defect <= MAXIMUM_SELECTED_MATRIX_DEFECT
                and bracketing_change
                <= MAXIMUM_BRACKETING_CENTRAL_CHANGE
                and extrapolated_difference
                <= MAXIMUM_EXTRAPOLATED_JVP_DIFFERENCE
            )
            reports[region_name] = {
                "target_radius_over_rg": target,
                "actual_face_index": face,
                "row_count": int(rows.size),
                "selected_matrix_defect": matrix_defect,
                "central_2e5_4e5_difference": bracketing_change,
                "fine_coarse_extrapolated_difference": (
                    extrapolated_difference
                ),
                "passed": passed,
            }
            arrays[f"{region_name}__rows"] = rows
            arrays[f"{region_name}__richardson_fine"] = richardson_fine
            arrays[f"{region_name}__richardson_coarse"] = (
                richardson_coarse
            )
    return {
        "regions": reports,
        "passed": bool(all(item["passed"] for item in reports.values())),
    }, arrays


def _step_cache_paths(label: str) -> tuple[Path, Path]:
    return (
        CACHE_DIRECTORY / f"{label}_step2e5.json",
        CACHE_DIRECTORY / f"{label}_step2e5_arrays.npz",
    )


def _build_or_load_step2_blocks(
    configuration: dict,
    *,
    force: bool,
) -> tuple[dict, dict[str, csr_matrix]]:
    label = str(configuration["label"])
    json_path, arrays_path = _step_cache_paths(label)
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
        "stored_stationary_delta_sha256": _array_sha256(
            native["stationary_delta"]
        ),
        "finite_difference_step": ALTERNATIVE_MATRIX_STEP,
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
                    name: np.asarray(source[name])
                    for name in source.files
                }
            matrices = {
                name: wp10c9d5b._unpack_sparse(name, packed)
                for name in (*BLOCK_NAMES, "production")
            }
            return payload, matrices

    print(f"WP10c9d5c0: building 2e-5 blocks for {label}", flush=True)
    started = time.perf_counter()
    pattern = causal_five_field_radial_reduced_jacobian_pattern(
        int(base.shape[0])
    )
    matrices = causal_radial_colored_block_jacobians(
        lambda values: wp10c9d5b._scaled_block_residuals(
            configuration,
            values,
        ),
        np.zeros(base.size, dtype=float),
        pattern,
        finite_difference_step=ALTERNATIVE_MATRIX_STEP,
    )
    packed: dict[str, np.ndarray] = {}
    for name, matrix in matrices.items():
        packed.update(wp10c9d5b._pack_sparse(name, matrix))
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **packed)
    payload = {
        **contract,
        "arrays_path": _relative(arrays_path),
        "arrays_sha256": _sha256(arrays_path),
        "wall_seconds": time.perf_counter() - started,
    }
    _write_json(json_path, payload)
    return payload, matrices


def _combine_blocks(
    step2: dict[str, csr_matrix],
    step4: dict[str, csr_matrix],
) -> dict[str, dict[str, csr_matrix]]:
    return {
        "stored_4e5": {
            name: step4[name].tocsr()
            for name in (*BLOCK_NAMES, "production")
        },
        "central_2e5": {
            name: step2[name].tocsr()
            for name in (*BLOCK_NAMES, "production")
        },
        "richardson_2e5_4e5": {
            name: ((4.0 * step2[name] - step4[name]) / 3.0).tocsr()
            for name in (*BLOCK_NAMES, "production")
        },
    }


def _inner_flux_function(
    configuration: dict,
    increment: np.ndarray,
) -> np.ndarray:
    base = np.asarray(configuration["base_primitives"], dtype=float)
    columns = np.asarray(
        configuration["candidate_native"]["primitive_column_scales"],
        dtype=float,
    )
    charts = (
        base.ravel() + columns * np.asarray(increment, dtype=float)
    ).reshape(base.shape)
    ledger = causal_five_field_radial_candidate_ledger(
        configuration["context"],
        charts,
        quadrature_order=PATH_QUADRATURE_ORDER,
    )
    return np.asarray(
        ledger.interfaces.candidate_shared_face_fluxes_over_c[
            0,
            CONSERVATIVE_FIELDS,
        ],
        dtype=float,
    )


def _partial_inner_flux_jacobian(
    configuration: dict,
    step: float,
) -> np.ndarray:
    base = np.asarray(configuration["base_primitives"], dtype=float)
    zero = np.zeros(base.size, dtype=float)
    matrix = np.zeros((CONSERVATIVE_FIELDS.size, base.size), dtype=float)
    stop = min(
        base.size,
        N_FIELDS * INNER_FLUX_STENCIL_CELLS,
    )
    for column in range(stop):
        perturbation = np.zeros_like(zero)
        perturbation[column] = float(step)
        matrix[:, column] = (
            _inner_flux_function(configuration, perturbation)
            - _inner_flux_function(configuration, -perturbation)
        ) / (2.0 * float(step))
    return matrix


def _inner_flux_matrices(configuration: dict) -> dict[str, np.ndarray]:
    step2 = _partial_inner_flux_jacobian(
        configuration,
        ALTERNATIVE_MATRIX_STEP,
    )
    step4 = _partial_inner_flux_jacobian(
        configuration,
        STORED_MATRIX_STEP,
    )
    return {
        "stored_4e5": step4,
        "central_2e5": step2,
        "richardson_2e5_4e5": (4.0 * step2 - step4) / 3.0,
    }


def _candidate_stationary(
    blocks: dict[str, csr_matrix],
) -> csr_matrix:
    shape = blocks[BLOCK_NAMES[0]].shape
    return sum(
        (blocks[name] for name in BLOCK_NAMES),
        start=csr_matrix(shape, dtype=float),
    )


def _native_generators(
    configuration: dict,
    method_blocks: dict[str, dict[str, csr_matrix]],
) -> dict[str, np.ndarray]:
    native = configuration["candidate_native"]
    production = np.asarray(native["production_generator"], dtype=float)
    descriptor = np.asarray(native["descriptor"], dtype=float)
    factor = splu(csc_matrix(descriptor), permc_spec="COLAMD")
    stored_delta = np.asarray(native["stationary_delta"], dtype=float)
    step2_delta = (
        _candidate_stationary(method_blocks["central_2e5"]).toarray()
        - method_blocks["central_2e5"]["production"].toarray()
    )
    richardson_delta = (4.0 * step2_delta - stored_delta) / 3.0
    return {
        "stored_4e5": np.asarray(
            native["candidate_generator"],
            dtype=float,
        ),
        "central_2e5": production - factor.solve(step2_delta),
        "richardson_2e5_4e5": (
            production - factor.solve(richardson_delta)
        ),
    }


def _cumulative(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    return wp10c9d0._cumulative_trapezoid(
        np.asarray(times, dtype=float),
        np.asarray(values, dtype=float),
    )


def _observable_history(
    configuration: dict,
    native_generator: np.ndarray,
    blocks: dict[str, csr_matrix],
    inner_flux_matrix: np.ndarray,
) -> dict[str, np.ndarray | float]:
    amplitudes = np.asarray(configuration["amplitudes"], dtype=float)
    columns = np.asarray(
        configuration["candidate_native"]["primitive_column_scales"],
        dtype=float,
    )
    rows = np.asarray(
        configuration["candidate_native"]["conservation_row_scales"],
        dtype=float,
    )
    generator = wp10c9d5.wp10c8v._similarity_rescale_generator(
        native_generator,
        columns,
        amplitudes,
    )
    state, rate, restart = wp10c9d5._propagate(
        generator,
        np.asarray(configuration["initial"], dtype=float),
        np.asarray(configuration["times"], dtype=float),
    )
    scaled_state = (
        state * amplitudes[None, :, :]
    ).reshape(state.shape[0], -1) / columns[None, :]
    n_cells = int(state.shape[1])

    def physical_rows(matrix: csr_matrix) -> np.ndarray:
        return (
            np.asarray(
                [matrix @ vector for vector in scaled_state],
                dtype=float,
            )
            * rows[None, :]
        ).reshape(state.shape[0], n_cells, N_FIELDS)

    block_rows = {
        name: physical_rows(blocks[name])
        for name in BLOCK_NAMES
    }
    inner = np.asarray(
        [inner_flux_matrix @ vector for vector in scaled_state],
        dtype=float,
    )
    face_fluxes = causal_radial_prefix_face_fluxes(
        inner,
        block_rows["conservative_transport"][:, :, CONSERVATIVE_FIELDS],
    )
    active = int(configuration["active_cells"])
    stationary_rows = sum(
        (block_rows[name] for name in BLOCK_NAMES),
        start=np.zeros_like(block_rows[BLOCK_NAMES[0]]),
    )
    net = -np.sum(
        stationary_rows[:, :active, :][:, :, CONSERVATIVE_FIELDS],
        axis=1,
    )
    cooling = np.sum(
        block_rows["cooling"][:, :active, :],
        axis=1,
    )[:, CONSERVATIVE_FIELDS[1:]]
    height = np.sum(
        block_rows["lower_height_work"][:, :active, :],
        axis=1,
    )[:, CONSERVATIVE_FIELDS[1:]]
    signals = np.concatenate(
        (
            inner,
            face_fluxes[:, active, :],
            net,
            cooling,
            height,
        ),
        axis=1,
    )
    return {
        "times": np.asarray(configuration["times"], dtype=float),
        "state": state,
        "rate": rate,
        "scaled_state": scaled_state,
        "signals": signals,
        "cumulative_signals": _cumulative(
            np.asarray(configuration["times"], dtype=float),
            signals,
        ),
        "face_fluxes": face_fluxes,
        "restart_defect": restart,
    }


def _fixed_observable_scales(configurations: dict) -> np.ndarray:
    baselines = np.asarray(
        [
            wp10c9d5._candidate_baseline(configurations[label])
            for label in LABELS
        ],
        dtype=float,
    )
    return np.maximum(
        np.max(np.abs(baselines), axis=0),
        np.finfo(float).tiny,
    )


def _fixed_face_scales(configurations: dict) -> np.ndarray:
    values = []
    for label in LABELS:
        configuration = configurations[label]
        ledger = causal_five_field_radial_candidate_ledger(
            configuration["context"],
            np.asarray(configuration["base_primitives"], dtype=float),
            quadrature_order=PATH_QUADRATURE_ORDER,
        )
        active = int(configuration["active_cells"])
        values.append(
            np.asarray(
                ledger.interfaces.candidate_shared_face_fluxes_over_c[
                    : active + 1,
                    :,
                ][:, CONSERVATIVE_FIELDS],
                dtype=float,
            )
        )
    return np.maximum(
        np.max(
            np.abs(np.concatenate(values, axis=0)),
            axis=0,
        ),
        np.finfo(float).tiny,
    )


def _maximum_component_rms_difference(
    first: np.ndarray,
    second: np.ndarray,
    scales: np.ndarray,
) -> float:
    difference = (
        np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    ) / np.asarray(scales, dtype=float)[None, :]
    return float(np.max(np.sqrt(np.mean(difference * difference, axis=0))))


def _common_face_maps(configurations: dict) -> tuple[np.ndarray, dict]:
    coarse = configurations[LABELS[0]]
    edges = (
        np.asarray(coarse["context"].grid.edges, dtype=float)
        / float(coarse["context"].grid.gravitational_radius)
    )
    last = int(
        np.flatnonzero(edges <= max(TARGET_RADII_OVER_RG))[-1]
    )
    coarse_faces = np.arange(last + 1, dtype=int)
    maps = {
        LABELS[0]: coarse_faces,
        LABELS[1]: 2 * coarse_faces,
        LABELS[2]: 4 * coarse_faces,
    }
    for label, indices in maps.items():
        configuration = configurations[label]
        label_edges = (
            np.asarray(configuration["context"].grid.edges, dtype=float)
            / float(configuration["context"].grid.gravitational_radius)
        )
        if not np.allclose(
            label_edges[indices],
            edges[coarse_faces],
            rtol=5.0e-14,
            atol=0.0,
        ):
            raise RuntimeError("embedded grids do not share c0 faces")
    return edges[coarse_faces], maps


def _metrics_payload(metrics) -> dict:
    return {
        "component_scales": metrics.component_scales,
        "component_normalization_scales": (
            metrics.component_normalization_scales
        ),
        "component_activity_thresholds": (
            metrics.component_activity_thresholds
        ),
        "significant_components": metrics.significant_components,
        "component_coarse_medium_differences": (
            metrics.component_coarse_medium_differences
        ),
        "component_medium_fine_differences": (
            metrics.component_medium_fine_differences
        ),
        "component_observed_orders": metrics.component_observed_orders,
        "component_history_cosines": metrics.component_history_cosines,
        "component_error_cosines": metrics.component_error_cosines,
        "component_passed": metrics.component_passed,
        "coarse_medium_difference": metrics.coarse_medium_difference,
        "medium_fine_difference": metrics.medium_fine_difference,
        "observed_order": metrics.observed_order,
        "history_cosine": metrics.history_cosine,
        "error_cosine": metrics.error_cosine,
        "passed": metrics.passed,
    }


def _recovery_report(
    histories: dict,
    common_radii: np.ndarray,
    face_maps: dict,
    face_scales: np.ndarray,
) -> dict:
    surfaces = []
    passes = []
    stride = int(SENSITIVITY_SAMPLE_STRIDE)
    for surface, radius in enumerate(common_radii):
        instant = {
            label: np.asarray(
                histories[label]["face_fluxes"],
                dtype=float,
            )[::stride, face_maps[label][surface], :]
            for label in LABELS
        }
        sampled_times = {
            label: np.asarray(
                histories[label]["times"],
                dtype=float,
            )[::stride]
            for label in LABELS
        }
        cumulative = {
            label: _cumulative(sampled_times[label], instant[label])
            for label in LABELS
        }
        instantaneous_metrics = causal_radial_history_convergence(
            *(instant[label] for label in LABELS),
            minimum_order=MINIMUM_RECOVERY_ORDER,
            maximum_fine_normalized_difference=(
                MAXIMUM_FINE_NORMALIZED_DIFFERENCE
            ),
            minimum_fine_signed_cosine=MINIMUM_HISTORY_COSINE,
            minimum_relative_activity=MINIMUM_RELATIVE_ACTIVITY,
            component_reference_scales=face_scales,
            minimum_error_cosine=MINIMUM_ERROR_COSINE,
        )
        cumulative_metrics = causal_radial_history_convergence(
            *(cumulative[label] for label in LABELS),
            minimum_order=MINIMUM_RECOVERY_ORDER,
            maximum_fine_normalized_difference=(
                MAXIMUM_FINE_NORMALIZED_DIFFERENCE
            ),
            minimum_fine_signed_cosine=MINIMUM_HISTORY_COSINE,
            minimum_relative_activity=MINIMUM_RELATIVE_ACTIVITY,
            component_reference_scales=(
                face_scales
                * max(
                    float(next(iter(sampled_times.values()))[-1]),
                    np.finfo(float).tiny,
                )
            ),
            minimum_error_cosine=MINIMUM_ERROR_COSINE,
        )
        passed = bool(
            instantaneous_metrics.passed and cumulative_metrics.passed
        )
        passes.append(passed)
        surfaces.append(
            {
                "surface": surface,
                "radius_over_rg": float(radius),
                "instantaneous": _metrics_payload(
                    instantaneous_metrics
                ),
                "cumulative": _metrics_payload(cumulative_metrics),
                "passed": passed,
            }
        )
    recovery_index = causal_radial_first_consecutive_recovery(
        np.asarray(passes, dtype=bool),
        required_consecutive=REQUIRED_CONSECUTIVE_RECOVERY_SURFACES,
    )
    return {
        "surface_reports": surfaces,
        "surface_passes": passes,
        "recovery_surface_index": recovery_index,
        "recovery_radius_over_rg": (
            None
            if recovery_index is None
            else float(common_radii[recovery_index])
        ),
    }


def _recovery_is_stable(reports: dict[str, dict]) -> bool:
    indices = [
        reports[name]["recovery_surface_index"] for name in METHOD_NAMES
    ]
    if all(index is None for index in indices):
        return True
    if any(index is None for index in indices):
        return False
    return max(int(index) for index in indices) - min(
        int(index) for index in indices
    ) <= 1


def _direct_face_action(
    configuration: dict,
    direction: np.ndarray,
    step: float,
) -> np.ndarray:
    base = np.asarray(configuration["base_primitives"], dtype=float)
    columns = np.asarray(
        configuration["candidate_native"]["primitive_column_scales"],
        dtype=float,
    )
    perturbation = columns * (float(step) * np.asarray(direction, dtype=float))

    def flux(charts: np.ndarray) -> np.ndarray:
        ledger = causal_five_field_radial_candidate_ledger(
            configuration["context"],
            charts.reshape(base.shape),
            quadrature_order=PATH_QUADRATURE_ORDER,
        )
        return np.asarray(
            ledger.interfaces.candidate_shared_face_fluxes_over_c[
                :,
                CONSERVATIVE_FIELDS,
            ],
            dtype=float,
        )

    plus = flux(base.ravel() + perturbation)
    minus = flux(base.ravel() - perturbation)
    return (plus - minus) / (2.0 * float(step))


def _selected_direct_faces(configuration: dict) -> np.ndarray:
    edges = (
        np.asarray(configuration["context"].grid.edges, dtype=float)
        / float(configuration["context"].grid.gravitational_radius)
    )
    return np.asarray(
        [
            int(np.flatnonzero(edges <= target)[-1])
            for target in DIRECT_FACE_TARGETS_OVER_RG
        ],
        dtype=int,
    )


def _face_parity_report(
    configuration: dict,
    histories: dict[str, dict],
) -> tuple[dict, dict[str, np.ndarray]]:
    times = np.asarray(configuration["times"], dtype=float)
    time_indices = np.asarray(
        [
            int(round(fraction * (times.size - 1)))
            for fraction in DIRECT_FACE_TIME_FRACTIONS
        ],
        dtype=int,
    )
    faces = _selected_direct_faces(configuration)
    reports = {}
    arrays = {
        "time_indices": time_indices,
        "face_indices": faces,
    }
    maximum = 0.0
    for method in ("stored_4e5", "richardson_2e5_4e5"):
        method_defects = []
        for time_index in time_indices:
            direction = np.asarray(
                histories[method]["scaled_state"],
                dtype=float,
            )[time_index]
            d2 = _direct_face_action(
                configuration,
                direction,
                ALTERNATIVE_MATRIX_STEP,
            )
            d4 = _direct_face_action(
                configuration,
                direction,
                STORED_MATRIX_STEP,
            )
            direct = (
                d4
                if method == "stored_4e5"
                else (4.0 * d2 - d4) / 3.0
            )
            assembled = np.asarray(
                histories[method]["face_fluxes"],
                dtype=float,
            )[time_index]
            defect = _relative_difference(
                direct[faces],
                assembled[faces],
            )
            method_defects.append(defect)
            maximum = max(maximum, defect)
        reports[method] = {
            "relative_defects": method_defects,
            "maximum_relative_defect": max(method_defects),
            "passed": bool(
                max(method_defects) <= MAXIMUM_DIRECT_FACE_PARITY_DEFECT
            ),
        }
        arrays[f"{method}__relative_defects"] = np.asarray(
            method_defects,
            dtype=float,
        )
    return {
        "methods": reports,
        "maximum_relative_defect": maximum,
        "passed": bool(
            maximum <= MAXIMUM_DIRECT_FACE_PARITY_DEFECT
        ),
    }, arrays


def _stride_report(
    histories: dict,
    observable_scales: np.ndarray,
    face_scales: np.ndarray,
) -> dict:
    reports = {}
    maximum = 0.0
    for label in LABELS:
        times = np.asarray(histories[label]["times"], dtype=float)
        signals = np.asarray(histories[label]["signals"], dtype=float)
        faces = np.asarray(histories[label]["face_fluxes"], dtype=float)
        endpoints = {}
        face_endpoints = {}
        for stride in STRIDE_AUDITS:
            indices = np.arange(0, times.size, int(stride), dtype=int)
            if indices[-1] != times.size - 1:
                indices = np.append(indices, times.size - 1)
            endpoints[str(stride)] = _cumulative(
                times[indices],
                signals[indices],
            )[-1]
            face_endpoints[str(stride)] = _cumulative(
                times[indices],
                faces[indices].reshape(indices.size, -1),
            )[-1].reshape(faces.shape[1:])
        duration = max(float(times[-1]), np.finfo(float).tiny)
        observable_scale = observable_scales * duration
        face_scale = face_scales * duration
        observable_defects = {}
        face_defects = {}
        for stride in STRIDE_AUDITS[1:]:
            key = str(stride)
            observable_defects[key] = float(
                np.max(
                    np.abs(endpoints[key] - endpoints["1"])
                    / observable_scale
                )
            )
            face_defects[key] = float(
                np.max(
                    np.abs(face_endpoints[key] - face_endpoints["1"])
                    / face_scale[None, :]
                )
            )
            maximum = max(
                maximum,
                observable_defects[key],
                face_defects[key],
            )
        reports[label] = {
            "observable_endpoint_defects": observable_defects,
            "face_endpoint_defects": face_defects,
            "passed": bool(
                max(
                    (*observable_defects.values(), *face_defects.values())
                )
                <= MAXIMUM_STRIDE_DEFECT
            ),
        }
    return {
        "configurations": reports,
        "maximum_relative_defect": maximum,
        "passed": bool(maximum <= MAXIMUM_STRIDE_DEFECT),
    }


def run(*, force_step2: bool = False) -> dict:
    started = time.perf_counter()
    identity = _validate_analyzed_git_identity()
    d5b_summary = json.loads(D5B_SUMMARY.read_text(encoding="utf-8"))
    if d5b_summary["binding_branch"] != (
        "D_no_compact_recovery_or_stable_dominant_term"
    ):
        raise RuntimeError("WP10c9d5b binding classification changed")
    configurations = {
        label: configuration
        for label, configuration
        in wp10c9d5._common_configurations(False).items()
        if label in LABELS
    }
    decisive: dict[str, np.ndarray] = {
        "directional_steps": np.asarray(DIRECTIONAL_STEPS, dtype=float),
    }
    directional_reports = {}
    directional_passed = True
    for label in LABELS:
        print(f"WP10c9d5c0: directional audit {label}", flush=True)
        configuration = configurations[label]
        label_reports = {}
        for name, direction in _directions(configuration).items():
            print(f"  direction {name}", flush=True)
            report, arrays = _directional_report(
                configuration,
                direction,
            )
            label_reports[name] = report
            directional_passed = bool(
                directional_passed and report["passed"]
            )
            for array_name, values in arrays.items():
                decisive[f"{label}__{name}__{array_name}"] = values
        directional_reports[label] = label_reports

    step2_cache_reports = {}
    physical_sensitivity = {
        "executed": False,
        "passed": False,
    }
    recovery_reports = {}
    face_parity_reports = {}
    stride_report = {
        "executed": False,
        "passed": False,
    }
    if directional_passed:
        observable_scales = _fixed_observable_scales(configurations)
        face_scales = _fixed_face_scales(configurations)
        decisive["fixed_observable_scales"] = observable_scales
        decisive["fixed_face_scales"] = face_scales
        all_histories: dict[str, dict[str, dict]] = {
            method: {} for method in METHOD_NAMES
        }
        for label in LABELS:
            configuration = configurations[label]
            _report4, blocks4, _dense4 = (
                wp10c9d5b._build_or_load_blocks(
                    configuration,
                    force=False,
                )
            )
            report2, blocks2 = _build_or_load_step2_blocks(
                configuration,
                force=force_step2,
            )
            step2_cache_reports[label] = report2
            method_blocks = _combine_blocks(blocks2, blocks4)
            inner_matrices = _inner_flux_matrices(configuration)
            generators = _native_generators(
                configuration,
                method_blocks,
            )
            for method in METHOD_NAMES:
                print(
                    f"WP10c9d5c0: propagate {label} {method}",
                    flush=True,
                )
                history = _observable_history(
                    configuration,
                    generators[method],
                    method_blocks[method],
                    inner_matrices[method],
                )
                all_histories[method][label] = history
                for name in (
                    "times",
                    "signals",
                    "cumulative_signals",
                    "face_fluxes",
                ):
                    decisive[f"{method}__{label}__{name}"] = np.asarray(
                        history[name],
                        dtype=float,
                    )
                decisive[
                    f"{method}__{label}__first_cell_state"
                ] = np.asarray(history["state"], dtype=float)[:, 0, :]

            parity, parity_arrays = _face_parity_report(
                configuration,
                {
                    method: all_histories[method][label]
                    for method in METHOD_NAMES
                },
            )
            face_parity_reports[label] = parity
            for name, values in parity_arrays.items():
                decisive[f"{label}__face_parity__{name}"] = values

        derivative_differences = {}
        maximum_derivative_difference = 0.0
        for label in LABELS:
            times = np.asarray(
                all_histories["stored_4e5"][label]["times"],
                dtype=float,
            )
            duration = max(float(times[-1]), np.finfo(float).tiny)
            label_report = {}
            for method in METHOD_NAMES[1:]:
                signal_difference = _maximum_component_rms_difference(
                    all_histories[method][label]["signals"],
                    all_histories["stored_4e5"][label]["signals"],
                    observable_scales,
                )
                cumulative_difference = _maximum_component_rms_difference(
                    all_histories[method][label]["cumulative_signals"],
                    all_histories["stored_4e5"][label][
                        "cumulative_signals"
                    ],
                    observable_scales * duration,
                )
                first_cell_difference = _relative_difference(
                    all_histories[method][label]["state"][:, 0, :],
                    all_histories["stored_4e5"][label]["state"][:, 0, :],
                )
                maximum = max(signal_difference, cumulative_difference)
                maximum_derivative_difference = max(
                    maximum_derivative_difference,
                    maximum,
                )
                label_report[method] = {
                    "signal_difference": signal_difference,
                    "cumulative_difference": cumulative_difference,
                    "first_cell_state_difference": first_cell_difference,
                    "maximum_export_difference": maximum,
                }
            derivative_differences[label] = label_report

        spatial_signal_difference = _maximum_component_rms_difference(
            all_histories["stored_4e5"][LABELS[1]]["signals"],
            all_histories["stored_4e5"][LABELS[2]]["signals"],
            observable_scales,
        )
        duration = max(
            float(
                np.asarray(
                    all_histories["stored_4e5"][LABELS[2]]["times"],
                    dtype=float,
                )[-1]
            ),
            np.finfo(float).tiny,
        )
        spatial_cumulative_difference = _maximum_component_rms_difference(
            all_histories["stored_4e5"][LABELS[1]][
                "cumulative_signals"
            ],
            all_histories["stored_4e5"][LABELS[2]][
                "cumulative_signals"
            ],
            observable_scales * duration,
        )
        binding_spatial_difference = max(
            spatial_signal_difference,
            spatial_cumulative_difference,
        )
        derivative_to_spatial_ratio = (
            maximum_derivative_difference
            / max(binding_spatial_difference, np.finfo(float).tiny)
        )

        common_radii, face_maps = _common_face_maps(configurations)
        decisive["common_face_radii_over_rg"] = common_radii
        for method in METHOD_NAMES:
            recovery_reports[method] = _recovery_report(
                all_histories[method],
                common_radii,
                face_maps,
                face_scales,
            )
        recovery_stable = _recovery_is_stable(recovery_reports)
        face_parity_passed = bool(
            all(report["passed"] for report in face_parity_reports.values())
        )
        stride_report = {
            "executed": True,
            **_stride_report(
                all_histories["stored_4e5"],
                observable_scales,
                face_scales,
            ),
        }
        physical_passed = bool(
            maximum_derivative_difference
            <= MAXIMUM_DERIVATIVE_EXPORT_DIFFERENCE
            and derivative_to_spatial_ratio
            <= MAXIMUM_DERIVATIVE_TO_SPATIAL_RATIO
            and recovery_stable
            and face_parity_passed
            and stride_report["passed"]
        )
        physical_sensitivity = {
            "executed": True,
            "derivative_differences": derivative_differences,
            "maximum_derivative_export_difference": (
                maximum_derivative_difference
            ),
            "binding_medium_fine_signal_difference": (
                spatial_signal_difference
            ),
            "binding_medium_fine_cumulative_difference": (
                spatial_cumulative_difference
            ),
            "binding_medium_fine_spatial_difference": (
                binding_spatial_difference
            ),
            "derivative_to_spatial_ratio": derivative_to_spatial_ratio,
            "recovery_location_stable": recovery_stable,
            "face_parity_passed": face_parity_passed,
            "stride_passed": stride_report["passed"],
            "passed": physical_passed,
        }

    c0_passed = bool(
        directional_passed and physical_sensitivity["passed"]
    )
    classification = (
        "cross_grid_derivative_and_metric_hardening_passed_"
        "extended_localization_authorized"
        if c0_passed
        else
        "cross_grid_derivative_or_physical_sensitivity_failed_"
        "extended_localization_blocked"
    )

    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)
    source_hashes, source_manifest = _source_manifest()
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "labels": LABELS,
        "target_radii_over_rg": TARGET_RADII_OVER_RG,
        "stencil_halo_cells": STENCIL_HALO_CELLS,
        "directional_steps": DIRECTIONAL_STEPS,
        "direction_seeds": DIRECTION_SEEDS,
        "methods": METHOD_NAMES,
        "sensitivity_sample_stride": SENSITIVITY_SAMPLE_STRIDE,
        "stride_audits": STRIDE_AUDITS,
        "direct_face_targets_over_rg": DIRECT_FACE_TARGETS_OVER_RG,
        "gates": {
            "maximum_selected_matrix_defect": (
                MAXIMUM_SELECTED_MATRIX_DEFECT
            ),
            "maximum_bracketing_central_change": (
                MAXIMUM_BRACKETING_CENTRAL_CHANGE
            ),
            "maximum_extrapolated_jvp_difference": (
                MAXIMUM_EXTRAPOLATED_JVP_DIFFERENCE
            ),
            "maximum_derivative_export_difference": (
                MAXIMUM_DERIVATIVE_EXPORT_DIFFERENCE
            ),
            "maximum_derivative_to_spatial_ratio": (
                MAXIMUM_DERIVATIVE_TO_SPATIAL_RATIO
            ),
            "maximum_direct_face_parity_defect": (
                MAXIMUM_DIRECT_FACE_PARITY_DEFECT
            ),
            "maximum_stride_defect": MAXIMUM_STRIDE_DEFECT,
            "minimum_relative_activity": MINIMUM_RELATIVE_ACTIVITY,
            "minimum_recovery_order": MINIMUM_RECOVERY_ORDER,
            "maximum_fine_normalized_difference": (
                MAXIMUM_FINE_NORMALIZED_DIFFERENCE
            ),
            "minimum_history_cosine": MINIMUM_HISTORY_COSINE,
            "minimum_error_cosine": MINIMUM_ERROR_COSINE,
        },
    }
    _write_json(CONFIG_PATH, config)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        **identity,
        "parent_wp10c9d5_candidate_remains_rejected": True,
        "parent_wp10c9d5b_branch_d_preserved": True,
        "parent_wp10c9d5b_summary_path": _relative(D5B_SUMMARY),
        "parent_wp10c9d5b_summary_sha256": _sha256(D5B_SUMMARY),
        "directional_reports": directional_reports,
        "directional_derivative_passed": directional_passed,
        "step2_cache_reports": step2_cache_reports,
        "physical_sensitivity": physical_sensitivity,
        "recovery_reports": recovery_reports,
        "face_parity_reports": face_parity_reports,
        "stride_report": stride_report,
        "cross_grid_hardening_passed": c0_passed,
        "wp10c9d5c1_extended_localization_authorized": c0_passed,
        "self_consistent_tangent_authorized": False,
        "frozen_candidate_recertification_authorized": False,
        "production_operator_authorized": False,
        "nonlinear_candidate_authorized": False,
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
        "generation_command": (
            "PYTHONPATH=src python3 "
            "scripts/run_causal_inner_cross_grid_hardening_"
            "wp10c9d5c0.py"
        ),
        "method_scope": (
            "CROSS-GRID FROZEN DERIVATIVE / METRIC HARDENING; "
            "PRODUCTION NEUTRAL"
        ),
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        "scientific_status": (
            "DIAGNOSTIC ONLY" if c0_passed else "REJECTED"
        ),
        "authorization_status": (
            "EXTENDED LOCALIZATION ONLY"
            if c0_passed
            else "EXTENDED LOCALIZATION BLOCKED"
        ),
        "source_input_hashes": {
            _relative(D5B_SUMMARY): _sha256(D5B_SUMMARY),
        },
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "establishes": (
            "Whether derivative choice is stable on all embedded grids "
            "through the still-refined 12-rg domain, whether direct face "
            "JVPs agree with prefix reconstruction, and whether the rejected "
            "physical-export classification is derivative robust."
        ),
        "does_not_establish": (
            "A repaired physical operator, a recovery radius, a dominant "
            "mechanism, nonlinear convergence, fixed-Q closure, or reduced "
            "evolution."
        ),
    }
    _write_json(PROVENANCE_PATH, provenance)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force-step2",
        action="store_true",
        help="rebuild the 2e-5 block-Jacobian caches",
    )
    arguments = parser.parse_args()
    run(force_step2=arguments.force_step2)


if __name__ == "__main__":
    main()
