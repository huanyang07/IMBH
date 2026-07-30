#!/usr/bin/env python3
"""Localize the c6c lower-height-work shear-component failure.

This package changes no operator.  It propagates the two independent failed
``sin^2`` shear bases and three passing controls, reconstructs the complete
cell-integrated lower-height-work JVP, and compares its radial error with an
independent 769/513-node continuum reference.
"""

from __future__ import annotations

import argparse
import csv
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

import run_causal_inner_continuum_lift_wp10c9d6c3 as c3
import run_causal_inner_monolithic_uniform_exports_wp10c9d6c as c6
import run_causal_inner_windowed_contract_wp10c9d6c6a2 as c6a2

from imri_qpe.layer3_minidisk_1d.causal_inner_continuum_truncation import (  # noqa: E402
    build_causal_five_field_continuum_background,
    linearize_causal_five_field_continuum_reference,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_height_localization import (  # noqa: E402
    causal_partition_cell_integrals,
    causal_prefix_suffix_histories,
    causal_restrict_cell_integrals,
    causal_signed_band_gram_matrix,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_validation import (  # noqa: E402
    causal_exact_semigroup_integral_history,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_windowed_contract import (  # noqa: E402
    causal_trapezoid_weights,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c6d"
ANALYZED_BASE_COMMIT = "80bdb60674d8a3afaf3e35a61edcae5934bc1a1f"
ANALYZED_BASE_PARENT = "2593204ee4b0a7116157b7fda0b619cf5fd0bab7"
ANALYZED_BASE_TREE = "e03fe4cb1af7b34838e6943c6431f308124d9438"
THIS_RUNNER = (
    "scripts/run_causal_inner_height_localization_wp10c9d6c6d.py"
)

LABELS = ("uniform_N128", "uniform_N256", "uniform_N512")
PROFILE_NAMES = (
    "p2__inward_shear",
    "p2__outward_shear",
    "p4__inward_shear",
    "p4__outward_shear",
    "p2__material",
)
FAILED_PROFILES = PROFILE_NAMES[:2]
PASSING_CONTROLS = PROFILE_NAMES[2:]
TIME_HORIZON_S = 0.125
TIME_SAMPLE_COUNT = 65
PRIMARY_CONTINUUM_NODES = 769
SECONDARY_CONTINUUM_NODES = 513
ANGULAR_FIELD = 2
KILLING_FIELD = 3
SOURCE_CHANNELS = {
    "radial_momentum": 1,
    "angular_momentum": ANGULAR_FIELD,
    "killing_energy": KILLING_FIELD,
}
OBSERVABLE_NAMES = tuple(c3.OBSERVABLE_NAMES)
ANGULAR_OBSERVABLE_INDEX = 11
KILLING_OBSERVABLE_INDEX = 12
FIXED_BAND_TARGETS_OVER_RG = (1.8, 3.0, 5.0, 8.0, 10.5)
MINIMUM_COMPONENT_ORDER = 0.75
MAXIMUM_DIRECT_PARITY_DEFECT = 1.0e-12
MAXIMUM_PARENT_REPLAY_DEFECT = 1.0e-10
MAXIMUM_CONTINUUM_REFERENCE_TO_FINE_RATIO = 0.10
MAXIMUM_CONTINUUM_LEDGER_DEFECT = 1.0e-10

PARENT_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_packet_validation_wp10c9d6c6c"
)
PARENT_SUMMARY = PARENT_DIRECTORY / "summary.json"
PARENT_ARRAYS = PARENT_DIRECTORY / "decisive_arrays.npz"
PARENT_CONFIG = PARENT_DIRECTORY / "config.json"
PARENT_PROVENANCE = PARENT_DIRECTORY / "provenance.json"
CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_height_localization_wp10c9d6c6d"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

IMPLEMENTATION_SOURCES = (
    THIS_RUNNER,
    "scripts/run_causal_inner_packet_validation_wp10c9d6c6c.py",
    "scripts/run_causal_inner_windowed_contract_wp10c9d6c6a2.py",
    "scripts/run_causal_inner_continuum_lift_wp10c9d6c3.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_height_localization.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_continuum_truncation.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_monolithic_tangent.py",
    "tests/test_causal_inner_height_localization.py",
    "tests/test_causal_inner_height_localization_wp10c9d6c6d.py",
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


def _refresh_canonical_catalog() -> None:
    rows: list[dict[str, str | int]] = []
    for case in sorted(CANONICAL_DIRECTORY.parent.iterdir()):
        if not case.is_dir():
            continue
        provenance_path = case / "provenance.json"
        if not provenance_path.is_file():
            continue
        provenance = json.loads(
            provenance_path.read_text(encoding="utf-8")
        )
        status = provenance.get(
            "scientific_status",
            provenance.get("numerical_status", "DIAGNOSTIC ONLY"),
        )
        for path in sorted(case.iterdir()):
            if path.is_file():
                rows.append(
                    {
                        "case": case.name,
                        "path": str(path.relative_to(ROOT)),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                        "scientific_status": status,
                    }
                )
    with CANONICAL_MANIFEST.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "case",
                "path",
                "bytes",
                "sha256",
                "scientific_status",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    summary = json.loads(CANONICAL_SUMMARY.read_text(encoding="utf-8"))
    summary.update(
        {
            "case_count": len({str(row["case"]) for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": ANALYZED_BASE_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, summary)


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
        raise RuntimeError("WP10c9d6c6d analyzed git identity changed")
    return {
        "analyzed_base_commit": resolved,
        "analyzed_base_parent": parent,
        "analyzed_base_tree": tree,
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
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
    }


def _load_parent() -> tuple[dict, dict[str, np.ndarray]]:
    summary = json.loads(PARENT_SUMMARY.read_text(encoding="utf-8"))
    expected_failed = {
        f"p2__{family}::a{amplitude:.2f}::{sign}"
        for family in ("inward_shear", "outward_shear")
        for amplitude in (0.5, 1.0)
        for sign in ("minus", "plus")
    }
    if (
        summary["classification"]
        != "prospective_uniform_packet_validation_failed"
        or summary["authorized_next"]
        != "freeze_failed_variant_and_localize"
        or set(summary["comparison_report"]["failed_packets"])
        != expected_failed
    ):
        raise RuntimeError("c6c localization authorization changed")
    with np.load(PARENT_ARRAYS, allow_pickle=False) as source:
        arrays = {
            name: np.array(source[name], copy=True)
            for name in source.files
        }
    return summary, arrays


def _build_inputs():
    configurations, construction_arrays, construction = (
        c3._build_continuum_configurations()
    )
    interpolator, characteristic_report, characteristic_arrays = (
        c6a2._build_characteristic_interpolator(
            configurations,
            construction_arrays,
        )
    )
    base_directions, probe_report, probe_arrays = c6a2._build_probes(
        configurations,
        construction_arrays,
        interpolator,
    )
    selected = {}
    for label in LABELS:
        selected[label] = np.column_stack(
            [
                base_directions[label][name]["primary_scaled"]
                for name in PROFILE_NAMES
            ]
        )
    reference_grid = configurations[LABELS[0]]["context"].grid
    evaluators = {
        name: c6a2._probe_evaluator(
            c6a2.PROBE_DEFINITIONS[name],
            interpolator,
            lower_radius=float(reference_grid.edges[0]),
            upper_radius=float(reference_grid.edges[-1]),
        )
        for name in PROFILE_NAMES
    }
    report = {
        "continuum_construction_passed": construction["passed"],
        "characteristic_field_passed": characteristic_report["passed"],
        "probe_replay_passed": probe_report["passed"],
        "passed": bool(
            construction["passed"]
            and characteristic_report["passed"]
            and probe_report["passed"]
        ),
    }
    arrays = {**characteristic_arrays, **probe_arrays}
    return (
        configurations,
        construction_arrays,
        selected,
        evaluators,
        report,
        arrays,
    )


def _lower_height_cell_map(tangent) -> np.ndarray:
    cells = int(tangent.base_primitives.shape[0])
    rows = np.asarray(
        tangent.conservation_row_scales,
        dtype=float,
    )
    block = np.asarray(
        tangent.spatial_tangent.block_scaled_jacobians[
            "candidate_lower_height_work"
        ],
        dtype=float,
    )
    return (
        -block * rows[:, None]
    ).reshape(cells, 5, block.shape[1])


def _propagate(
    configurations: dict,
    tangents: dict,
    observable_maps: dict,
    directions: dict,
) -> tuple[dict, dict]:
    times = np.linspace(0.0, TIME_HORIZON_S, TIME_SAMPLE_COUNT)
    propagated = {}
    report = {}
    for label in LABELS:
        print(f"WP10c9d6c6d: propagate five profiles on {label}", flush=True)
        generator = np.asarray(
            tangents[label].scaled_generator_per_s,
            dtype=float,
        )
        initial = np.asarray(directions[label], dtype=float)
        trace = float(np.trace(generator))
        states = np.asarray(
            expm_multiply(
                generator,
                initial,
                start=0.0,
                stop=TIME_HORIZON_S,
                num=TIME_SAMPLE_COUNT,
                endpoint=True,
                traceA=trace,
            ),
            dtype=float,
        )
        exact = causal_exact_semigroup_integral_history(
            generator,
            states,
            initial,
        )
        cell_map = _lower_height_cell_map(tangents[label])
        cell_actions = np.transpose(
            np.einsum("cfn,tnp->tpcf", cell_map, states),
            (1, 0, 2, 3),
        )
        cumulative_cell_actions = np.transpose(
            np.einsum(
                "cfn,tnp->tpcf",
                cell_map,
                exact.integrated_states,
            ),
            (1, 0, 2, 3),
        )
        signals = np.einsum(
            "tnp,on->pto",
            states,
            np.asarray(observable_maps[label], dtype=float),
        )
        parity = max(
            _relative_defect(
                np.sum(cell_actions[..., ANGULAR_FIELD], axis=-1),
                signals[..., ANGULAR_OBSERVABLE_INDEX],
            ),
            _relative_defect(
                np.sum(cell_actions[..., KILLING_FIELD], axis=-1),
                signals[..., KILLING_OBSERVABLE_INDEX],
            ),
        )
        propagated[label] = {
            "times": times,
            "states": states,
            "signals": signals,
            "cell_actions": cell_actions,
            "cumulative_cell_actions": cumulative_cell_actions,
            "integral_residuals": exact.relative_solve_residuals.T,
        }
        report[label] = {
            "cell_count": int(cell_actions.shape[2]),
            "maximum_direct_cell_sum_parity_defect": parity,
            "maximum_exact_integral_relative_solve_residual": (
                exact.maximum_relative_solve_residual
            ),
        }
    return propagated, report


def _relative_defect(left: np.ndarray, right: np.ndarray) -> float:
    first = np.asarray(left, dtype=float)
    second = np.asarray(right, dtype=float)
    scale = max(
        float(np.linalg.norm(first)),
        float(np.linalg.norm(second)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(first - second) / scale)


def _scalar_metrics(
    coarse: np.ndarray,
    medium: np.ndarray,
    fine: np.ndarray,
    *,
    physical_scale: float,
) -> dict:
    first = (
        np.asarray(medium, dtype=float) - np.asarray(coarse, dtype=float)
    ) / float(physical_scale)
    second = (
        np.asarray(fine, dtype=float) - np.asarray(medium, dtype=float)
    ) / float(physical_scale)
    tiny = np.finfo(float).tiny
    first_rms = float(np.sqrt(np.mean(first**2)))
    second_rms = float(np.sqrt(np.mean(second**2)))
    first_max = float(np.max(np.abs(first)))
    second_max = float(np.max(np.abs(second)))
    denominator = max(
        float(np.linalg.norm(first) * np.linalg.norm(second)),
        tiny,
    )
    cosine = float(np.dot(first.ravel(), second.ravel()) / denominator)
    return {
        "observed_rms_order": float(
            np.log2(max(first_rms, tiny) / max(second_rms, tiny))
        ),
        "observed_maximum_order": float(
            np.log2(max(first_max, tiny) / max(second_max, tiny))
        ),
        "coarse_medium_rms_difference": first_rms,
        "medium_fine_rms_difference": second_rms,
        "maximum_fine_normalized_difference": second_max,
        "refinement_error_cosine": cosine,
    }


def _band_edges(grid) -> tuple[np.ndarray, np.ndarray]:
    edges_rg = np.asarray(grid.edges, dtype=float) / float(
        grid.gravitational_radius
    )
    indices = [0]
    for target in FIXED_BAND_TARGETS_OVER_RG[1:]:
        index = int(np.argmin(np.abs(edges_rg - float(target))))
        if index > indices[-1] and index < edges_rg.size - 1:
            indices.append(index)
    indices.append(edges_rg.size - 1)
    unique = np.asarray(sorted(set(indices)), dtype=int)
    return unique, edges_rg[unique]


def _radial_report(
    configurations: dict,
    propagated: dict,
    observable_scales: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    coarse_grid = configurations[LABELS[0]]["context"].grid
    band_indices, band_edges_rg = _band_edges(coarse_grid)
    times = np.asarray(propagated[LABELS[0]]["times"], dtype=float)
    time_weights = causal_trapezoid_weights(times)
    reports = {}
    arrays: dict[str, np.ndarray] = {
        "times": times,
        "fixed_band_edge_indices": band_indices,
        "fixed_band_edges_over_rg": band_edges_rg,
    }
    for profile_index, profile in enumerate(PROFILE_NAMES):
        reports[profile] = {}
        for channel_name, field in SOURCE_CHANNELS.items():
            scale_index = (
                ANGULAR_OBSERVABLE_INDEX
                if field == ANGULAR_FIELD
                else KILLING_OBSERVABLE_INDEX
                if field == KILLING_FIELD
                else ANGULAR_OBSERVABLE_INDEX
            )
            physical_scale = float(observable_scales[scale_index])
            reports[profile][channel_name] = {}
            for history_name, source_key in (
                ("instantaneous", "cell_actions"),
                ("cumulative", "cumulative_cell_actions"),
            ):
                coarse = propagated[LABELS[0]][source_key][
                    profile_index, :, :, field
                ]
                medium = causal_restrict_cell_integrals(
                    propagated[LABELS[1]][source_key][
                        profile_index, :, :, field
                    ],
                    refinement_factor=2,
                )
                fine = causal_restrict_cell_integrals(
                    propagated[LABELS[2]][source_key][
                        profile_index, :, :, field
                    ],
                    refinement_factor=4,
                )
                prefix = tuple(
                    causal_prefix_suffix_histories(values)
                    for values in (coarse, medium, fine)
                )
                bands = tuple(
                    causal_partition_cell_integrals(
                        values,
                        band_indices,
                    )
                    for values in (coarse, medium, fine)
                )
                global_metrics = _scalar_metrics(
                    np.sum(coarse, axis=-1),
                    np.sum(medium, axis=-1),
                    np.sum(fine, axis=-1),
                    physical_scale=physical_scale,
                )
                cell_metrics = [
                    _scalar_metrics(
                        coarse[:, cell],
                        medium[:, cell],
                        fine[:, cell],
                        physical_scale=physical_scale,
                    )
                    for cell in range(coarse.shape[1])
                ]
                prefix_metrics = [
                    _scalar_metrics(
                        prefix[0][0][:, cell],
                        prefix[1][0][:, cell],
                        prefix[2][0][:, cell],
                        physical_scale=physical_scale,
                    )
                    for cell in range(coarse.shape[1])
                ]
                suffix_metrics = [
                    _scalar_metrics(
                        prefix[0][1][:, cell],
                        prefix[1][1][:, cell],
                        prefix[2][1][:, cell],
                        physical_scale=physical_scale,
                    )
                    for cell in range(coarse.shape[1])
                ]
                band_metrics = [
                    _scalar_metrics(
                        bands[0][:, band],
                        bands[1][:, band],
                        bands[2][:, band],
                        physical_scale=physical_scale,
                    )
                    for band in range(bands[0].shape[1])
                ]
                coarse_medium_bands = bands[1] - bands[0]
                medium_fine_bands = bands[2] - bands[1]
                gram_coarse = causal_signed_band_gram_matrix(
                    coarse_medium_bands,
                    physical_scale=physical_scale,
                    time_weights=time_weights,
                )
                gram_fine = causal_signed_band_gram_matrix(
                    medium_fine_bands,
                    physical_scale=physical_scale,
                    time_weights=time_weights,
                )

                def cancellation_ratio(gram: np.ndarray) -> float:
                    diagonal = np.maximum(np.diag(gram), 0.0)
                    return float(
                        np.sqrt(max(float(np.sum(gram)), 0.0))
                        / max(
                            float(np.sum(np.sqrt(diagonal))),
                            np.finfo(float).tiny,
                        )
                    )

                report = {
                    "global": global_metrics,
                    "minimum_cell_rms_order": min(
                        item["observed_rms_order"]
                        for item in cell_metrics
                    ),
                    "minimum_prefix_rms_order": min(
                        item["observed_rms_order"]
                        for item in prefix_metrics
                    ),
                    "minimum_suffix_rms_order": min(
                        item["observed_rms_order"]
                        for item in suffix_metrics
                    ),
                    "band_metrics": {
                        (
                            f"{band_edges_rg[index]:.6f}:"
                            f"{band_edges_rg[index + 1]:.6f}"
                        ): band_metrics[index]
                        for index in range(len(band_metrics))
                    },
                    "failing_cell_indices": [
                        index
                        for index, item in enumerate(cell_metrics)
                        if item["observed_rms_order"]
                        < MINIMUM_COMPONENT_ORDER
                    ],
                    "failing_prefix_indices": [
                        index
                        for index, item in enumerate(prefix_metrics)
                        if item["observed_rms_order"]
                        < MINIMUM_COMPONENT_ORDER
                    ],
                    "failing_suffix_indices": [
                        index
                        for index, item in enumerate(suffix_metrics)
                        if item["observed_rms_order"]
                        < MINIMUM_COMPONENT_ORDER
                    ],
                    "coarse_medium_band_cancellation_ratio": (
                        cancellation_ratio(gram_coarse)
                    ),
                    "medium_fine_band_cancellation_ratio": (
                        cancellation_ratio(gram_fine)
                    ),
                }
                reports[profile][channel_name][history_name] = report
                prefix_name = (
                    f"{profile}__{channel_name}__{history_name}__"
                )
                arrays[prefix_name + "coarse_cells"] = coarse
                arrays[prefix_name + "medium_on_coarse_cells"] = medium
                arrays[prefix_name + "fine_on_coarse_cells"] = fine
                arrays[prefix_name + "coarse_medium_band_gram"] = (
                    gram_coarse
                )
                arrays[prefix_name + "medium_fine_band_gram"] = gram_fine
                arrays[prefix_name + "cell_rms_orders"] = np.asarray(
                    [
                        item["observed_rms_order"]
                        for item in cell_metrics
                    ]
                )
                arrays[prefix_name + "prefix_rms_orders"] = np.asarray(
                    [
                        item["observed_rms_order"]
                        for item in prefix_metrics
                    ]
                )
                arrays[prefix_name + "suffix_rms_orders"] = np.asarray(
                    [
                        item["observed_rms_order"]
                        for item in suffix_metrics
                    ]
                )
    return {
        "fixed_band_edges_over_rg": band_edges_rg,
        "profile_reports": reports,
    }, arrays


def _continuum_report(
    configurations: dict,
    construction_arrays: dict[str, np.ndarray],
    evaluators: dict,
    propagated: dict,
    observable_scales: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    finest = configurations[LABELS[-1]]
    grid = finest["context"].grid
    background_profile = c3.SmoothCellAverageProfile(
        knots=np.asarray(
            construction_arrays["continuum_background_knots"],
            dtype=float,
        ),
        coefficients=np.asarray(
            construction_arrays["continuum_background_coefficients"],
            dtype=float,
        ),
        degree=c3.PRIMARY_BACKGROUND_DEGREE,
        gravitational_radius=float(grid.gravitational_radius),
    )
    print("WP10c9d6c6d: build 769-node continuum background", flush=True)
    primary_background = build_causal_five_field_continuum_background(
        finest["context"],
        background_profile.evaluate,
        node_count=PRIMARY_CONTINUUM_NODES,
    )
    print("WP10c9d6c6d: build 513-node continuum background", flush=True)
    secondary_background = build_causal_five_field_continuum_background(
        finest["context"],
        background_profile.evaluate,
        node_count=SECONDARY_CONTINUUM_NODES,
    )
    reports = {}
    arrays = {}
    maximum_ledger = 0.0
    for profile_index, profile in enumerate(PROFILE_NAMES):
        print(f"WP10c9d6c6d: continuum action {profile}", flush=True)
        primary = linearize_causal_five_field_continuum_reference(
            primary_background,
            evaluators[profile],
        )
        secondary = linearize_causal_five_field_continuum_reference(
            secondary_background,
            evaluators[profile],
        )
        maximum_ledger = max(
            maximum_ledger,
            primary.maximum_pointwise_ledger_relative_defect,
            secondary.maximum_pointwise_ledger_relative_defect,
        )
        discrete_global = []
        primary_global = []
        secondary_global = []
        for label in LABELS:
            local_grid = configurations[label]["context"].grid
            primary_rows = primary.integrate_blocks(local_grid.edges)[
                "candidate_lower_height_work"
            ]
            secondary_rows = secondary.integrate_blocks(local_grid.edges)[
                "candidate_lower_height_work"
            ]
            discrete = propagated[label]["cell_actions"][
                profile_index, 0
            ]
            discrete_global.append(
                -float(np.sum(discrete[:, ANGULAR_FIELD]))
            )
            primary_global.append(
                float(np.sum(primary_rows[:, ANGULAR_FIELD]))
            )
            secondary_global.append(
                float(np.sum(secondary_rows[:, ANGULAR_FIELD]))
            )
            arrays[
                f"{profile}__{label}__continuum_primary_height_rows"
            ] = primary_rows
            arrays[
                f"{profile}__{label}__continuum_secondary_height_rows"
            ] = secondary_rows
        # Both ``discrete`` and continuum block rows carry residual signs.
        # The leading minus above only converts the stored observable action
        # back to the residual block sign for a like-for-like comparison.
        discrete_global = np.asarray(discrete_global)
        primary_global = np.asarray(primary_global)
        secondary_global = np.asarray(secondary_global)
        scale = float(observable_scales[ANGULAR_OBSERVABLE_INDEX])
        errors = np.abs(discrete_global - primary_global) / scale
        fine_spatial = max(
            abs(discrete_global[2] - discrete_global[1]) / scale,
            np.finfo(float).tiny,
        )
        reference_uncertainty = (
            abs(primary_global[2] - secondary_global[2]) / scale
        )
        reports[profile] = {
            "discrete_global_residual_block": discrete_global,
            "primary_continuum_global_residual_block": primary_global,
            "secondary_continuum_global_residual_block": secondary_global,
            "fixed_scale_discrete_continuum_errors": errors,
            "coarse_medium_error_order": float(
                np.log2(
                    max(errors[0], np.finfo(float).tiny)
                    / max(errors[1], np.finfo(float).tiny)
                )
            ),
            "medium_fine_error_order": float(
                np.log2(
                    max(errors[1], np.finfo(float).tiny)
                    / max(errors[2], np.finfo(float).tiny)
                )
            ),
            "reference_uncertainty_to_fine_difference": (
                reference_uncertainty / fine_spatial
            ),
            "maximum_continuum_ledger_relative_defect": max(
                primary.maximum_pointwise_ledger_relative_defect,
                secondary.maximum_pointwise_ledger_relative_defect,
            ),
        }
        arrays[f"{profile}__initial_discrete_global"] = discrete_global
        arrays[f"{profile}__initial_primary_continuum_global"] = (
            primary_global
        )
        arrays[f"{profile}__initial_secondary_continuum_global"] = (
            secondary_global
        )
    passed = bool(
        maximum_ledger <= MAXIMUM_CONTINUUM_LEDGER_DEFECT
        and all(
            report["reference_uncertainty_to_fine_difference"]
            <= MAXIMUM_CONTINUUM_REFERENCE_TO_FINE_RATIO
            for report in reports.values()
        )
    )
    return {
        "profile_reports": reports,
        "maximum_continuum_ledger_relative_defect": maximum_ledger,
        "passed": passed,
    }, arrays


def _mechanism_selection(radial: dict, continuum: dict) -> dict:
    if not continuum["passed"]:
        return {
            "selected_mechanism": "continuum_reference_unresolved",
            "authorized_next": "repair_continuum_reference",
        }
    profile_reports = radial["profile_reports"]
    failed_band_sets = []
    for profile in FAILED_PROFILES:
        band_reports = profile_reports[profile]["angular_momentum"][
            "instantaneous"
        ]["band_metrics"]
        failed_band_sets.append(
            {
                name
                for name, report in band_reports.items()
                if report["observed_rms_order"] < MINIMUM_COMPONENT_ORDER
            }
        )
    common_failed_bands = set.intersection(*failed_band_sets)
    control_failed_bands = set()
    for profile in PASSING_CONTROLS:
        control_failed_bands.update(
            name
            for name, report in profile_reports[profile][
                "angular_momentum"
            ]["instantaneous"]["band_metrics"].items()
            if report["observed_rms_order"] < MINIMUM_COMPONENT_ORDER
        )
    selected_bands = sorted(common_failed_bands - control_failed_bands)
    failed_global = all(
        profile_reports[profile]["angular_momentum"]["instantaneous"][
            "global"
        ]["observed_rms_order"]
        < MINIMUM_COMPONENT_ORDER
        for profile in FAILED_PROFILES
    )
    every_failed_band_converges = all(
        not failed_band_sets[index]
        for index in range(len(failed_band_sets))
    )
    if selected_bands:
        mechanism = "stable_lower_height_work_band_hypothesis"
        authorized_next = "single_lower_height_work_band_audit"
    elif failed_global and every_failed_band_converges:
        mechanism = "convergent_bands_noncontracting_cancellation_remainder"
        authorized_next = "prospective_integral_conditioning_audit"
    else:
        mechanism = "no_stable_lower_height_work_mechanism"
        authorized_next = "reconsider_prospective_export_contract"
    return {
        "selected_mechanism": mechanism,
        "authorized_next": authorized_next,
        "common_failed_bands": sorted(common_failed_bands),
        "control_failed_bands": sorted(control_failed_bands),
        "selected_bands": selected_bands,
        "all_failed_profile_bands_converge": every_failed_band_converges,
    }


def _config() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "operator_change": False,
        "profiles": PROFILE_NAMES,
        "failed_profiles": FAILED_PROFILES,
        "passing_controls": PASSING_CONTROLS,
        "time_horizon_s": TIME_HORIZON_S,
        "time_sample_count": TIME_SAMPLE_COUNT,
        "primary_continuum_nodes": PRIMARY_CONTINUUM_NODES,
        "secondary_continuum_nodes": SECONDARY_CONTINUUM_NODES,
        "source_channels": SOURCE_CHANNELS,
        "fixed_band_targets_over_rg": FIXED_BAND_TARGETS_OVER_RG,
        "gates": {
            "minimum_component_order": MINIMUM_COMPONENT_ORDER,
            "maximum_direct_parity_defect": (
                MAXIMUM_DIRECT_PARITY_DEFECT
            ),
            "maximum_parent_replay_defect": (
                MAXIMUM_PARENT_REPLAY_DEFECT
            ),
            "maximum_continuum_reference_to_fine_ratio": (
                MAXIMUM_CONTINUUM_REFERENCE_TO_FINE_RATIO
            ),
            "maximum_continuum_ledger_defect": (
                MAXIMUM_CONTINUUM_LEDGER_DEFECT
            ),
        },
    }


def run() -> dict:
    started = time.perf_counter()
    identity = _validate_analyzed_git_identity()
    parent_summary, parent_arrays = _load_parent()
    (
        configurations,
        construction_arrays,
        directions,
        evaluators,
        construction,
        arrays,
    ) = _build_inputs()
    if not construction["passed"]:
        raise RuntimeError("c6d inherited construction failed")
    print("WP10c9d6c6d: build unchanged monolithic tangents", flush=True)
    tangents, observable_maps, method_reports, _baselines = (
        c3._build_tangents(configurations, construction_arrays)
    )
    method_passed = all(
        method_reports[label]["passed"] for label in LABELS
    )
    propagated, propagation_report = _propagate(
        configurations,
        tangents,
        observable_maps,
        directions,
    )
    maximum_direct_parity = max(
        report["maximum_direct_cell_sum_parity_defect"]
        for report in propagation_report.values()
    )
    base_lookup = {
        name: index for index, name in enumerate(c6a2.PROBE_NAMES)
    }
    replay_defects = {}
    for label in LABELS:
        parent = np.asarray(
            parent_arrays[f"{label}__base_instantaneous_exports"],
            dtype=float,
        )
        current = propagated[label]["signals"]
        replay_defects[label] = max(
            _relative_defect(
                current[index],
                parent[base_lookup[profile]],
            )
            for index, profile in enumerate(PROFILE_NAMES)
        )
    maximum_parent_replay = max(replay_defects.values())
    observable_scales = np.asarray(
        parent_arrays["fixed_physical_observable_scales"],
        dtype=float,
    )
    radial, radial_arrays = _radial_report(
        configurations,
        propagated,
        observable_scales,
    )
    continuum, continuum_arrays = _continuum_report(
        configurations,
        construction_arrays,
        evaluators,
        propagated,
        observable_scales,
    )
    mechanism = _mechanism_selection(radial, continuum)
    audit_passed = bool(
        method_passed
        and maximum_direct_parity <= MAXIMUM_DIRECT_PARITY_DEFECT
        and maximum_parent_replay <= MAXIMUM_PARENT_REPLAY_DEFECT
        and continuum["passed"]
    )
    if not audit_passed:
        classification = "lower_height_work_localization_method_unresolved"
        authorized_next = "none"
    else:
        classification = mechanism["selected_mechanism"]
        authorized_next = mechanism["authorized_next"]

    decisive = {
        **arrays,
        **radial_arrays,
        **continuum_arrays,
        "fixed_physical_observable_scales": observable_scales,
    }
    for label in LABELS:
        decisive[f"{label}__selected_cell_actions"] = propagated[label][
            "cell_actions"
        ]
        decisive[f"{label}__selected_cumulative_cell_actions"] = (
            propagated[label]["cumulative_cell_actions"]
        )
    source_hashes, source_manifest = _source_manifest()
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        **identity,
        "classification": classification,
        "authorized_next": authorized_next,
        "passed": audit_passed,
        "audit_executed": True,
        "operator_changed": False,
        "parent_classification": parent_summary["classification"],
        "parent_classification_preserved": True,
        "c6c_failure_preserved": True,
        "configuration": _config(),
        "construction_report": construction,
        "method_reports": {
            label: method_reports[label] for label in LABELS
        },
        "method_passed": method_passed,
        "propagation_report": propagation_report,
        "direct_cell_sum_parity_defect": maximum_direct_parity,
        "parent_history_replay_defects": replay_defects,
        "maximum_parent_history_replay_defect": maximum_parent_replay,
        "radial_localization_report": radial,
        "continuum_reference_report": continuum,
        "mechanism_selection": mechanism,
        "targeted_operator_intervention_authorized": bool(
            authorized_next == "single_lower_height_work_band_audit"
        ),
        "embedded_export_discrimination_authorized": False,
        "nonlinear_physical_trajectory_authorized": False,
        "production_operator_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "decisive_arrays_path": str(
            DECISIVE_ARRAYS.relative_to(ROOT)
        ),
        "decisive_arrays_sha256": _sha256(DECISIVE_ARRAYS),
        "decisive_array_hashes": {
            name: _array_sha256(values)
            for name, values in sorted(decisive.items())
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "scientific_status": "DIAGNOSTIC ONLY",
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        **identity,
        "implementation_base_tree": _git_value(
            "rev-parse",
            "HEAD^{tree}",
        ),
        "working_tree_status": _git_value("status", "--short"),
        "command": (
            "PYTHONPATH=src:scripts python "
            "scripts/run_causal_inner_height_localization_wp10c9d6c6d.py"
        ),
        "environment": _environment(),
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "parent_canonical_hashes": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (
                PARENT_CONFIG,
                PARENT_SUMMARY,
                PARENT_ARRAYS,
                PARENT_PROVENANCE,
            )
        },
    }
    _write_json(CONFIG_PATH, _config())
    _write_json(SUMMARY_PATH, summary)
    _write_json(PROVENANCE_PATH, provenance)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    _refresh_canonical_catalog()
    print(
        json.dumps(
            _plain(
                {
                    "classification": classification,
                    "authorized_next": authorized_next,
                    "direct_cell_sum_parity_defect": (
                        maximum_direct_parity
                    ),
                    "maximum_parent_history_replay_defect": (
                        maximum_parent_replay
                    ),
                    "continuum_reference_passed": continuum["passed"],
                    "selected_bands": mechanism.get("selected_bands", []),
                }
            ),
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return summary


def refresh_metadata_only() -> dict:
    if not SUMMARY_PATH.exists() or not DECISIVE_ARRAYS.exists():
        raise RuntimeError("c6d canonical evidence is unavailable")
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    with np.load(DECISIVE_ARRAYS, allow_pickle=False) as source:
        arrays = {
            name: np.array(source[name], copy=True)
            for name in source.files
        }
    source_hashes, source_manifest = _source_manifest()
    summary["configuration"] = _config()
    summary["implementation_source_hashes"] = source_hashes
    summary["implementation_source_manifest_sha256"] = source_manifest
    summary["decisive_arrays_sha256"] = _sha256(DECISIVE_ARRAYS)
    summary["decisive_array_hashes"] = {
        name: _array_sha256(values)
        for name, values in sorted(arrays.items())
    }
    provenance = json.loads(
        PROVENANCE_PATH.read_text(encoding="utf-8")
    )
    provenance.update(
        {
            "implementation_base_tree": _git_value(
                "rev-parse",
                "HEAD^{tree}",
            ),
            "working_tree_status": _git_value("status", "--short"),
            "environment": _environment(),
            "implementation_source_hashes": source_hashes,
            "implementation_source_manifest_sha256": source_manifest,
        }
    )
    _write_json(CONFIG_PATH, _config())
    _write_json(SUMMARY_PATH, summary)
    _write_json(PROVENANCE_PATH, provenance)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    _refresh_canonical_catalog()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-metadata-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.refresh_metadata_only:
        refresh_metadata_only()
    else:
        run()


if __name__ == "__main__":
    main()
