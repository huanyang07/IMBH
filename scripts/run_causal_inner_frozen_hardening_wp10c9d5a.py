"""Run the WP10c9d5a provenance and frozen-Jacobian hardening audit.

This package changes no scientific operator and no decisive WP10c9d5 array.
It corrects the exact Git lineage, creates a self-contained replay bundle for
two binding configurations, compares dense and colored finite differences,
audits the declared sparsity, and measures directional step sensitivity.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    causal_five_field_radial_candidate_ledger,
    causal_five_field_radial_reduced_jacobian_pattern,
    causal_five_field_reduced_stationary_residual,
    causal_radial_dense_colored_audit,
    causal_radial_jvp_step_sweep,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_dae_system import (  # noqa: E402
    CausalFiveFieldDAEContext,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_geometry import (  # noqa: E402
    KerrSchildColumnGrid,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_migration import (  # noqa: E402
    KerrSchildCellSourceRates,
    SchwarzschildCurvatureVerticalFrequency,
)
from imri_qpe.layer3_minidisk_1d.hill_roche_nozzle import (  # noqa: E402
    GasRadiationHillRocheNozzleProvider,
    HillRocheNozzleGeometry,
)


WORK_PACKAGE = "WP10c9d5a"
SCHEMA_VERSION = 1
SCIENTIFIC_IMPLEMENTATION_COMMIT = (
    "038ba35659e76aff0605fffa5fb457e99362063d"
)
SCIENTIFIC_PARENT_COMMIT = "42dd7f1d4ca048fcbd2faa02e71e0a66db300891"
SCIENTIFIC_TREE_SHA = "a1e4e33378154d91d17afe001479b063b74ca27f"
WP10C9D5_DECISIVE_ARRAYS_SHA256 = (
    "384c17dc99c3a739015d5298b8a06c416b134d0b1591a9e4c1c2360aa9dbee8b"
)
THIS_RUNNER = "scripts/run_causal_inner_frozen_hardening_wp10c9d5a.py"
CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_frozen_hardening_wp10c9d5a"
)
REPLAY_INPUTS = CANONICAL_DIRECTORY / "replay_inputs.npz"
REPLAY_CONTEXTS = CANONICAL_DIRECTORY / "replay_contexts.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
D5_CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_frozen_discrimination_wp10c9d5"
)
D5_SUMMARY = D5_CANONICAL_DIRECTORY / "summary.json"
D5_PROVENANCE = D5_CANONICAL_DIRECTORY / "provenance.json"
D5_DECISIVE_ARRAYS = D5_CANONICAL_DIRECTORY / "decisive_arrays.npz"

REPLAY_LABELS = (
    "uniform_N64",
    "N128_exterior_N128_inner_c48",
)
GENERATOR_RELATIVE_STEP = 4.0e-5
JVP_STEPS = (
    5.0e-6,
    1.0e-5,
    2.0e-5,
    4.0e-5,
    8.0e-5,
    1.6e-4,
    3.2e-4,
)
PATH_QUADRATURE_ORDER = 6
RANDOM_SEED = 9105
MAXIMUM_DENSE_COLORED_DEFECT = 1.0e-10
MAXIMUM_OFF_PATTERN_ENTRY = 1.0e-10
MAXIMUM_SELECTED_JVP_DEFECT = 5.0e-5
MAXIMUM_PLATEAU_ADJACENT_CHANGE = 2.0e-5
MINIMUM_CONSECUTIVE_PLATEAU_CHANGES = 2
MAXIMUM_REPLAY_BASE_DEFECT = 1.0e-13

IMPLEMENTATION_SOURCES = (
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_radial_hardening.py",
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_radial_frozen.py",
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_radial_fluctuation.py",
    "src/imri_qpe/layer3_minidisk_1d/__init__.py",
    THIS_RUNNER,
    "scripts/run_causal_inner_frozen_discrimination_wp10c9d5.py",
    "tests/test_causal_inner_radial_hardening.py",
    "tests/test_causal_inner_frozen_hardening_wp10c9d5a.py",
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


def _git_revision(revision: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", revision],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _validate_scientific_git_identity() -> dict[str, str]:
    identity = {
        "scientific_implementation_commit": _git_revision(
            SCIENTIFIC_IMPLEMENTATION_COMMIT
        ),
        "scientific_implementation_parent_commit": _git_revision(
            f"{SCIENTIFIC_IMPLEMENTATION_COMMIT}^"
        ),
        "scientific_implementation_tree_sha": _git_revision(
            f"{SCIENTIFIC_IMPLEMENTATION_COMMIT}^{{tree}}"
        ),
    }
    expected = {
        "scientific_implementation_commit": (
            SCIENTIFIC_IMPLEMENTATION_COMMIT
        ),
        "scientific_implementation_parent_commit": SCIENTIFIC_PARENT_COMMIT,
        "scientific_implementation_tree_sha": SCIENTIFIC_TREE_SHA,
    }
    if identity != expected:
        raise RuntimeError(
            "WP10c9d5a scientific Git identity differs from its declaration"
        )
    return identity


def _refresh_sha256s(directory: Path) -> None:
    names = tuple(
        name
        for name in (
            "config.json",
            "decisive_arrays.npz",
            "provenance.json",
            "replay_contexts.json",
            "replay_inputs.npz",
            "summary.json",
        )
        if (directory / name).exists()
    )
    lines = [f"{_sha256(directory / name)}  {name}" for name in names]
    (directory / "SHA256SUMS.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _correct_d5_provenance(identity: dict[str, str]) -> dict:
    if _sha256(D5_DECISIVE_ARRAYS) != WP10C9D5_DECISIVE_ARRAYS_SHA256:
        raise RuntimeError("WP10c9d5 decisive scientific arrays changed")
    summary = json.loads(D5_SUMMARY.read_text(encoding="utf-8"))
    provenance = json.loads(D5_PROVENANCE.read_text(encoding="utf-8"))
    correction = {
        **identity,
        "metadata_correction_work_package": WORK_PACKAGE,
        "metadata_correction_kind": "provenance_only",
        "scientific_arrays_unchanged": True,
        "scientific_decisive_arrays_sha256": (
            WP10C9D5_DECISIVE_ARRAYS_SHA256
        ),
    }
    summary["analyzed_base_commit"] = SCIENTIFIC_PARENT_COMMIT
    summary.update(correction)
    provenance["analyzed_base_commit"] = SCIENTIFIC_PARENT_COMMIT
    provenance["source_parent_commit"] = SCIENTIFIC_PARENT_COMMIT
    provenance.update(correction)
    _write_json(D5_SUMMARY, summary)
    _write_json(D5_PROVENANCE, provenance)
    d5_names = (
        "config.json",
        "decisive_arrays.npz",
        "provenance.json",
        "summary.json",
    )
    lines = [
        f"{_sha256(D5_CANONICAL_DIRECTORY / name)}  {name}"
        for name in d5_names
    ]
    (D5_CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return correction


def _context_payload(
    context: CausalFiveFieldDAEContext,
    *,
    label: str,
    arrays: dict[str, np.ndarray],
) -> dict:
    context = context.validated()
    prefix = f"{label}__"
    arrays[prefix + "grid_edges"] = np.asarray(context.grid.edges)
    arrays[prefix + "grid_centers"] = np.asarray(context.grid.centers)
    arrays[prefix + "grid_cell_measures"] = np.asarray(
        context.grid.cell_measures
    )
    arrays[prefix + "grid_face_measures"] = np.asarray(
        context.grid.face_measures
    )
    if context.stream_sources is None:
        stream_present = False
    else:
        stream_present = True
        arrays[prefix + "stream_rest_mass"] = np.asarray(
            context.stream_sources.rest_mass
        )
        arrays[prefix + "stream_radial_momentum_over_c"] = np.asarray(
            context.stream_sources.radial_momentum_over_c
        )
        arrays[prefix + "stream_angular_momentum_over_c"] = np.asarray(
            context.stream_sources.angular_momentum_over_c
        )
        arrays[prefix + "stream_killing_energy_over_c2"] = np.asarray(
            context.stream_sources.killing_energy_over_c2
        )
    arrays[prefix + "outer_frozen_chart"] = np.asarray(
        context.outer_boundary_frozen_exterior_chart,
        dtype=float,
    )
    provider = context.outer_boundary_provider
    if not isinstance(provider, GasRadiationHillRocheNozzleProvider):
        raise TypeError("WP10c9d5a supports the committed nozzle provider")
    if not isinstance(
        context.vertical_frequency,
        SchwarzschildCurvatureVerticalFrequency,
    ):
        raise TypeError("WP10c9d5a supports the committed vertical provider")
    return {
        "label": label,
        "grid_gravitational_radius": float(
            context.grid.gravitational_radius
        ),
        "vertical_frequency": {
            "type": "SchwarzschildCurvatureVerticalFrequency",
            "gravitational_radius": float(
                context.vertical_frequency.gravitational_radius
            ),
        },
        "outer_boundary_provider": {
            "type": "GasRadiationHillRocheNozzleProvider",
            "geometry": asdict(provider.geometry),
            "mu_mol": provider.mu_mol,
            "gamma_gas": provider.gamma_gas,
            "transverse_quadrature_zones": (
                provider.transverse_quadrature_zones
            ),
        },
        "stream_sources_present": stream_present,
        "alpha": context.alpha,
        "stress_factor": context.stress_factor,
        "kappa": context.kappa,
        "include_radiative_cooling": context.include_radiative_cooling,
        "spatial_reconstruction": context.spatial_reconstruction,
        "boundary_trace_reconstruction": (
            context.boundary_trace_reconstruction
        ),
        "cell_rate_scheme": context.cell_rate_scheme,
        "cell_source_quadrature": context.cell_source_quadrature,
        "cell_storage_quadrature": context.cell_storage_quadrature,
        "inner_boundary_trace_override": (
            context.inner_boundary_trace_override
        ),
        "inner_flux_trace_override": context.inner_flux_trace_override,
        "inner_storage_trace_override": (
            context.inner_storage_trace_override
        ),
        "outer_boundary_flux_mode": context.outer_boundary_flux_mode,
        "interior_dissipation_mode": context.interior_dissipation_mode,
    }


def _context_from_payload(
    payload: dict,
    arrays: dict[str, np.ndarray],
) -> CausalFiveFieldDAEContext:
    label = str(payload["label"])
    prefix = f"{label}__"
    grid = KerrSchildColumnGrid(
        edges=np.asarray(arrays[prefix + "grid_edges"], dtype=float),
        centers=np.asarray(arrays[prefix + "grid_centers"], dtype=float),
        cell_measures=np.asarray(
            arrays[prefix + "grid_cell_measures"],
            dtype=float,
        ),
        face_measures=np.asarray(
            arrays[prefix + "grid_face_measures"],
            dtype=float,
        ),
        gravitational_radius=float(payload["grid_gravitational_radius"]),
    )
    vertical_payload = payload["vertical_frequency"]
    if vertical_payload["type"] != "SchwarzschildCurvatureVerticalFrequency":
        raise ValueError("unsupported replay vertical-frequency provider")
    vertical = SchwarzschildCurvatureVerticalFrequency(
        float(vertical_payload["gravitational_radius"])
    )
    provider_payload = payload["outer_boundary_provider"]
    if provider_payload["type"] != "GasRadiationHillRocheNozzleProvider":
        raise ValueError("unsupported replay outer-boundary provider")
    provider = GasRadiationHillRocheNozzleProvider(
        HillRocheNozzleGeometry(
            **provider_payload["geometry"],
        ),
        mu_mol=float(provider_payload["mu_mol"]),
        gamma_gas=float(provider_payload["gamma_gas"]),
        transverse_quadrature_zones=int(
            provider_payload["transverse_quadrature_zones"]
        ),
    )
    stream = None
    if payload["stream_sources_present"]:
        stream = KerrSchildCellSourceRates(
            rest_mass=np.asarray(
                arrays[prefix + "stream_rest_mass"],
                dtype=float,
            ),
            radial_momentum_over_c=np.asarray(
                arrays[prefix + "stream_radial_momentum_over_c"],
                dtype=float,
            ),
            angular_momentum_over_c=np.asarray(
                arrays[prefix + "stream_angular_momentum_over_c"],
                dtype=float,
            ),
            killing_energy_over_c2=np.asarray(
                arrays[prefix + "stream_killing_energy_over_c2"],
                dtype=float,
            ),
        )
    return CausalFiveFieldDAEContext(
        grid=grid,
        vertical_frequency=vertical,
        outer_boundary_provider=provider,
        stream_sources=stream,
        alpha=float(payload["alpha"]),
        stress_factor=float(payload["stress_factor"]),
        kappa=float(payload["kappa"]),
        include_radiative_cooling=bool(
            payload["include_radiative_cooling"]
        ),
        spatial_reconstruction=str(payload["spatial_reconstruction"]),
        boundary_trace_reconstruction=str(
            payload["boundary_trace_reconstruction"]
        ),
        cell_rate_scheme=str(payload["cell_rate_scheme"]),
        cell_source_quadrature=str(payload["cell_source_quadrature"]),
        cell_storage_quadrature=str(payload["cell_storage_quadrature"]),
        inner_boundary_trace_override=str(
            payload["inner_boundary_trace_override"]
        ),
        inner_flux_trace_override=str(
            payload["inner_flux_trace_override"]
        ),
        inner_storage_trace_override=str(
            payload["inner_storage_trace_override"]
        ),
        outer_boundary_flux_mode=str(payload["outer_boundary_flux_mode"]),
        outer_boundary_frozen_exterior_chart=np.asarray(
            arrays[prefix + "outer_frozen_chart"],
            dtype=float,
        ),
        interior_dissipation_mode=str(
            payload["interior_dissipation_mode"]
        ),
    ).validated()


def _scaled_delta_function(
    context: CausalFiveFieldDAEContext,
    base_primitives: np.ndarray,
    primitive_column_scales: np.ndarray,
    conservation_row_scales: np.ndarray,
):
    base = np.asarray(base_primitives, dtype=float)
    column_scales = np.asarray(primitive_column_scales, dtype=float).ravel()
    row_scales = np.asarray(conservation_row_scales, dtype=float).ravel()

    def residual(scaled_increment: np.ndarray) -> np.ndarray:
        charts = (
            base.ravel()
            + column_scales
            * np.asarray(scaled_increment, dtype=float).ravel()
        ).reshape(base.shape)
        candidate = causal_five_field_radial_candidate_ledger(
            context,
            charts,
            quadrature_order=PATH_QUADRATURE_ORDER,
        ).residual_rows.ravel()
        production = causal_five_field_reduced_stationary_residual(
            charts.ravel(),
            context,
        )
        return (
            np.asarray(candidate, dtype=float)
            - np.asarray(production, dtype=float)
        ) / row_scales

    return residual


def _prepare_replay_inputs() -> dict:
    import run_causal_inner_frozen_discrimination_wp10c9d5 as wp10c9d5

    configurations = wp10c9d5._common_configurations(False)
    arrays: dict[str, np.ndarray] = {}
    contexts = {}
    source_paths = (
        wp10c9d5.wp10c9d0.WP10C8Y_ARRAYS,
        wp10c9d5.wp10c9d0.WP10C8Z_OUTPUT,
        wp10c9d5.wp10c9d0.WP10C8Z_ARRAYS,
        wp10c9d5.D0_OUTPUT,
        wp10c9d5.D0_ARRAYS,
    )
    source_hashes = {
        _relative(path): _sha256(path)
        for path in source_paths
        if path.exists()
    }
    for label in REPLAY_LABELS:
        configuration = configurations[label]
        context = configuration["context"]
        native = configuration["candidate_native"]
        prefix = f"{label}__"
        base = np.asarray(configuration["base_primitives"], dtype=float)
        column_scales = np.asarray(
            native["primitive_column_scales"],
            dtype=float,
        )
        row_scales = np.asarray(
            native["conservation_row_scales"],
            dtype=float,
        )
        common_physical = (
            np.asarray(configuration["amplitudes"], dtype=float)
            * np.asarray(configuration["initial"], dtype=float)
        )
        common_scaled = common_physical.ravel() / column_scales
        residual = _scaled_delta_function(
            context,
            base,
            column_scales,
            row_scales,
        )
        arrays[prefix + "base_primitives"] = base
        arrays[prefix + "primitive_column_scales"] = column_scales
        arrays[prefix + "conservation_row_scales"] = row_scales
        arrays[prefix + "colored_stationary_delta"] = np.asarray(
            native["stationary_delta"],
            dtype=float,
        )
        arrays[prefix + "common_scaled_direction"] = common_scaled
        arrays[prefix + "base_scaled_delta"] = residual(
            np.zeros(base.size, dtype=float)
        )
        contexts[label] = _context_payload(
            context,
            label=label,
            arrays=arrays,
        )
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(REPLAY_INPUTS, **arrays)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "scientific_implementation_commit": (
            SCIENTIFIC_IMPLEMENTATION_COMMIT
        ),
        "source_input_hashes": source_hashes,
        "contexts": contexts,
        "replay_array_hashes": {
            name: _array_sha256(values)
            for name, values in arrays.items()
        },
    }
    _write_json(REPLAY_CONTEXTS, payload)
    return payload


def _load_replay_inputs() -> tuple[dict, dict[str, np.ndarray]]:
    if not REPLAY_INPUTS.exists() or not REPLAY_CONTEXTS.exists():
        raise FileNotFoundError(
            "WP10c9d5a replay inputs are absent; run once with "
            "--prepare-replay-inputs from the scientific worktree"
        )
    payload = json.loads(REPLAY_CONTEXTS.read_text(encoding="utf-8"))
    with np.load(REPLAY_INPUTS, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    if set(arrays) != set(payload["replay_array_hashes"]):
        raise RuntimeError("WP10c9d5a replay array set changed")
    for name, expected in payload["replay_array_hashes"].items():
        if _array_sha256(arrays[name]) != expected:
            raise RuntimeError(f"WP10c9d5a replay array changed: {name}")
    return payload, arrays


def _normalized_direction(values: np.ndarray) -> np.ndarray:
    direction = np.asarray(values, dtype=float).ravel()
    scale = float(np.max(np.abs(direction)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("hardening direction is degenerate")
    return direction / scale


def _directions(
    label: str,
    common_scaled_direction: np.ndarray,
    n_reduced: int,
) -> dict[str, np.ndarray]:
    common = _normalized_direction(common_scaled_direction)
    inner = np.zeros(n_reduced, dtype=float)
    inner[:15] = common[:15]
    result = {
        "common": common,
        "inner_three_cell_common": _normalized_direction(inner),
    }
    rng = np.random.default_rng(
        RANDOM_SEED + (0 if label == "uniform_N64" else 1)
    )
    for index in range(2):
        result[f"random_{index}"] = _normalized_direction(
            rng.standard_normal(n_reduced)
        )
    for field in range(5):
        basis = np.zeros(n_reduced, dtype=float)
        basis[field] = 1.0
        result[f"first_cell_field_{field}"] = basis
    return result


def _plateau_intervals(changes: np.ndarray) -> list[int]:
    flags = np.asarray(changes, dtype=float) <= (
        MAXIMUM_PLATEAU_ADJACENT_CHANGE
    )
    return [
        index
        for index in range(
            max(
                0,
                flags.size - MINIMUM_CONSECUTIVE_PLATEAU_CHANGES + 1,
            )
        )
        if np.all(
            flags[
                index : index + MINIMUM_CONSECUTIVE_PLATEAU_CHANGES
            ]
        )
    ]


def _audit_configuration(
    label: str,
    context_payload: dict,
    arrays: dict[str, np.ndarray],
) -> tuple[dict, dict[str, np.ndarray]]:
    print(f"WP10c9d5a: auditing {label}", flush=True)
    started = time.perf_counter()
    prefix = f"{label}__"
    context = _context_from_payload(context_payload, arrays)
    base = np.asarray(arrays[prefix + "base_primitives"], dtype=float)
    column_scales = np.asarray(
        arrays[prefix + "primitive_column_scales"],
        dtype=float,
    )
    row_scales = np.asarray(
        arrays[prefix + "conservation_row_scales"],
        dtype=float,
    )
    colored = np.asarray(
        arrays[prefix + "colored_stationary_delta"],
        dtype=float,
    )
    function = _scaled_delta_function(
        context,
        base,
        column_scales,
        row_scales,
    )
    base_delta = function(np.zeros(base.size, dtype=float))
    expected_base = np.asarray(
        arrays[prefix + "base_scaled_delta"],
        dtype=float,
    )
    base_scale = max(
        float(np.max(np.abs(base_delta))),
        float(np.max(np.abs(expected_base))),
        np.finfo(float).tiny,
    )
    replay_base_defect = float(
        np.max(np.abs(base_delta - expected_base)) / base_scale
    )
    pattern = causal_five_field_radial_reduced_jacobian_pattern(
        int(context.grid.centers.size)
    )
    selected_columns = (
        np.arange(base.size, dtype=int)
        if label == "uniform_N64"
        else np.arange(15, dtype=int)
    )
    dense = causal_radial_dense_colored_audit(
        function,
        np.zeros(base.size, dtype=float),
        colored,
        pattern,
        selected_columns,
        finite_difference_step=GENERATOR_RELATIVE_STEP,
    )
    sweeps = {}
    output_arrays = {
        f"{label}__dense_selected_columns": dense.selected_columns,
        f"{label}__dense_columns": dense.dense_columns,
        f"{label}__colored_columns": dense.colored_columns,
        f"{label}__dense_per_column_defects": (
            dense.per_column_relative_defects
        ),
    }
    for name, direction in _directions(
        label,
        arrays[prefix + "common_scaled_direction"],
        base.size,
    ).items():
        print(f"WP10c9d5a: {label} JVP {name}", flush=True)
        sweep = causal_radial_jvp_step_sweep(
            function,
            np.zeros(base.size, dtype=float),
            colored,
            direction,
            JVP_STEPS,
            selected_step=GENERATOR_RELATIVE_STEP,
        )
        intervals = _plateau_intervals(
            sweep.adjacent_relative_changes
        )
        selected_on_plateau = any(
            index
            <= sweep.selected_step_index
            <= index + MINIMUM_CONSECUTIVE_PLATEAU_CHANGES
            for index in intervals
        )
        passed = bool(
            sweep.selected_matrix_relative_defect
            <= MAXIMUM_SELECTED_JVP_DEFECT
            and len(intervals) >= 1
            and selected_on_plateau
        )
        sweeps[name] = {
            "selected_matrix_relative_defect": (
                sweep.selected_matrix_relative_defect
            ),
            "minimum_adjacent_relative_change": (
                sweep.minimum_adjacent_relative_change
            ),
            "plateau_interval_starts": intervals,
            "selected_step_on_plateau": selected_on_plateau,
            "passed": passed,
        }
        output_arrays[f"{label}__{name}__direction"] = direction
        output_arrays[f"{label}__{name}__direct_actions"] = (
            sweep.direct_actions
        )
        output_arrays[f"{label}__{name}__matrix_action"] = (
            sweep.matrix_action
        )
        output_arrays[f"{label}__{name}__matrix_defects"] = (
            sweep.matrix_relative_defects
        )
        output_arrays[f"{label}__{name}__adjacent_changes"] = (
            sweep.adjacent_relative_changes
        )
    dense_passed = bool(
        dense.maximum_relative_defect <= MAXIMUM_DENSE_COLORED_DEFECT
        and dense.maximum_off_pattern_relative_entry
        <= MAXIMUM_OFF_PATTERN_ENTRY
    )
    jvp_passed = bool(all(item["passed"] for item in sweeps.values()))
    payload = {
        "label": label,
        "n_cells": int(context.grid.centers.size),
        "selected_dense_column_count": int(selected_columns.size),
        "replay_base_relative_defect": replay_base_defect,
        "maximum_dense_colored_relative_defect": (
            dense.maximum_relative_defect
        ),
        "maximum_off_pattern_relative_entry": (
            dense.maximum_off_pattern_relative_entry
        ),
        "maximum_per_column_relative_defect": float(
            np.max(dense.per_column_relative_defects)
        ),
        "dense_colored_passed": dense_passed,
        "jvp_sweeps": sweeps,
        "jvp_plateau_passed": jvp_passed,
        "passed": bool(
            replay_base_defect <= MAXIMUM_REPLAY_BASE_DEFECT
            and dense_passed
            and jvp_passed
        ),
        "wall_seconds": time.perf_counter() - started,
    }
    return payload, output_arrays


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


def run(*, prepare_replay_inputs: bool = False) -> dict:
    started = time.perf_counter()
    identity = _validate_scientific_git_identity()
    correction = _correct_d5_provenance(identity)
    if prepare_replay_inputs:
        _prepare_replay_inputs()
    replay_payload, replay_arrays = _load_replay_inputs()
    reports = {}
    decisive: dict[str, np.ndarray] = {
        "jvp_steps": np.asarray(JVP_STEPS, dtype=float),
    }
    for label in REPLAY_LABELS:
        report, arrays = _audit_configuration(
            label,
            replay_payload["contexts"][label],
            replay_arrays,
        )
        reports[label] = report
        decisive.update(arrays)
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)
    source_hashes, source_manifest = _source_manifest()
    numerical_hardening_passed = bool(
        all(report["passed"] for report in reports.values())
    )
    classification = (
        "frozen_jacobian_hardening_passed_dynamic_localization_authorized"
        if numerical_hardening_passed
        else "frozen_jacobian_hardening_failed_dynamic_localization_blocked"
    )
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "replay_labels": REPLAY_LABELS,
        "generator_relative_step": GENERATOR_RELATIVE_STEP,
        "jvp_steps": JVP_STEPS,
        "random_seed": RANDOM_SEED,
        "gates": {
            "maximum_dense_colored_defect": (
                MAXIMUM_DENSE_COLORED_DEFECT
            ),
            "maximum_off_pattern_entry": MAXIMUM_OFF_PATTERN_ENTRY,
            "maximum_selected_jvp_defect": (
                MAXIMUM_SELECTED_JVP_DEFECT
            ),
            "maximum_plateau_adjacent_change": (
                MAXIMUM_PLATEAU_ADJACENT_CHANGE
            ),
            "minimum_consecutive_plateau_changes": (
                MINIMUM_CONSECUTIVE_PLATEAU_CHANGES
            ),
            "maximum_replay_base_defect": MAXIMUM_REPLAY_BASE_DEFECT,
        },
    }
    _write_json(CONFIG_PATH, config)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        **identity,
        "metadata_correction": correction,
        "wp10c9d5_decisive_arrays_unchanged": True,
        "wp10c9d5_decisive_arrays_sha256": _sha256(
            D5_DECISIVE_ARRAYS
        ),
        "replay_inputs_path": _relative(REPLAY_INPUTS),
        "replay_inputs_sha256": _sha256(REPLAY_INPUTS),
        "replay_contexts_path": _relative(REPLAY_CONTEXTS),
        "replay_contexts_sha256": _sha256(REPLAY_CONTEXTS),
        "configuration_reports": reports,
        "numerical_hardening_passed": numerical_hardening_passed,
        "wp10c9d5b_dynamic_localization_authorized": (
            numerical_hardening_passed
        ),
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
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_json(SUMMARY_PATH, summary)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        **identity,
        "source_parent_commit": SCIENTIFIC_PARENT_COMMIT,
        "generation_command": (
            "PYTHONPATH=src python3 "
            "scripts/run_causal_inner_frozen_hardening_wp10c9d5a.py"
        ),
        "scientific_status": "REJECTED",
        "method_scope": "NUMERICAL HARDENING / PRODUCTION NEUTRAL",
        "source_input_hashes": replay_payload["source_input_hashes"],
        "replay_inputs_sha256": _sha256(REPLAY_INPUTS),
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "establishes": (
            "Exact WP10c9d5 Git lineage, compact replay inputs, "
            "dense/colored parity, actual sparsity, and JVP step sensitivity."
        ),
        "does_not_establish": (
            "A repaired boundary operator, nonlinear candidate, fixed-Q "
            "closure, or reduced slow evolution."
        ),
    }
    _write_json(PROVENANCE_PATH, provenance)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-replay-inputs", action="store_true")
    args = parser.parse_args()
    result = run(prepare_replay_inputs=args.prepare_replay_inputs)
    print(
        json.dumps(
            {
                "classification": result["classification"],
                "numerical_hardening_passed": (
                    result["numerical_hardening_passed"]
                ),
                "wp10c9d5b_dynamic_localization_authorized": (
                    result[
                        "wp10c9d5b_dynamic_localization_authorized"
                    ]
                ),
                "runtime_seconds": result["runtime_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
