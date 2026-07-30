#!/usr/bin/env python3
"""Propagate the frozen c6b packet manifest on three uniform grids.

WP10c9d6c6b froze 44 sign/amplitude variants of eleven spectrally
qualified analytic profiles.  This package propagates every hashed variant
with the unchanged N128/N256/N512 monolithic tangents.  It evaluates the
frozen instantaneous, exact cumulative-export, and state-reference gates.

A pass authorizes embedded discrimination only.  It does not authorize a
nonlinear trajectory, production promotion, fixed-Q averaging, or reduced
slow-time evolution.
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
import run_causal_inner_packet_manifest_wp10c9d6c6b as c6b
import run_causal_inner_windowed_contract_wp10c9d6c6a2 as c6a2

from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_canonical_json_sha256,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_validation import (  # noqa: E402
    causal_exact_semigroup_integral_history,
    causal_packet_history_metrics,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_windowed_contract import (  # noqa: E402
    causal_field_history_norm,
    causal_restrict_proper_cell_averages,
    causal_trapezoid_weights,
    causal_windowed_richardson_reference,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c6c"
ANALYZED_BASE_COMMIT = "2593204ee4b0a7116157b7fda0b619cf5fd0bab7"
ANALYZED_BASE_PARENT = "4fd671c10809fb015476549a7afb5fc56f0e3d0a"
ANALYZED_BASE_TREE = "5e9c272a93421fbf84bc8aec221830608a9b417e"
FROZEN_MANIFEST_SHA256 = (
    "c908494d0886e126c4c8f4a6ef80e872e7df6161cf8937bc39cfbbe0a65811fc"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_packet_validation_wp10c9d6c6c.py"
)

LABELS = ("uniform_N128", "uniform_N256", "uniform_N512")
TIME_HORIZON_S = 0.125
TIME_SAMPLE_COUNT = 65
OBSERVABLE_NAMES = tuple(c3.OBSERVABLE_NAMES)
BOUNDARY_OBSERVABLE_COUNT = 6
MINIMUM_RELATIVE_ACTIVITY = 1.0e-8
MAXIMUM_PROPAGATION_SCALING_DEFECT = 1.0e-10

PARENT_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_packet_manifest_wp10c9d6c6b"
)
PARENT_CONFIG = PARENT_DIRECTORY / "config.json"
PARENT_SUMMARY = PARENT_DIRECTORY / "summary.json"
PARENT_ARRAYS = PARENT_DIRECTORY / "decisive_arrays.npz"
PARENT_PROVENANCE = PARENT_DIRECTORY / "provenance.json"
MANIFEST_PATH = PARENT_DIRECTORY / "packet_manifest.json"
C3_ARRAYS = (
    ROOT
    / "results/canonical/"
    "causal_inner_continuum_lift_wp10c9d6c3/decisive_arrays.npz"
)
CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_packet_validation_wp10c9d6c6c"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

IMPLEMENTATION_SOURCES = (
    THIS_RUNNER,
    "scripts/run_causal_inner_packet_manifest_wp10c9d6c6b.py",
    "scripts/run_causal_inner_windowed_contract_wp10c9d6c6a2.py",
    "scripts/run_causal_inner_continuum_lift_wp10c9d6c3.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_packet_validation.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_packet_manifest.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_windowed_contract.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_monolithic_tangent.py",
    "tests/test_causal_inner_packet_validation.py",
    "tests/test_causal_inner_packet_validation_wp10c9d6c6c.py",
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
            if not path.is_file():
                continue
            rows.append(
                {
                    "case": case.name,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "scientific_status": status,
                }
            )
    CANONICAL_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
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
        raise RuntimeError("WP10c9d6c6c analyzed git identity changed")
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


def _load_and_verify_manifest() -> tuple[dict, dict]:
    parent = json.loads(PARENT_SUMMARY.read_text(encoding="utf-8"))
    if (
        parent["classification"]
        != (
            "packet_definition_manifest_frozen_"
            "uniform_propagation_authorized"
        )
        or parent["authorized_next"]
        != "WP10c9d6c6c_prospective_uniform_packet_propagation"
        or not parent["passed"]
    ):
        raise RuntimeError("c6b propagation authorization changed")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    declared_hash = manifest["manifest_sha256"]
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256")
    calculated_hash = causal_canonical_json_sha256(unhashed)
    contract = manifest["prospective_propagation_contract"]
    verified = bool(
        declared_hash == calculated_hash == FROZEN_MANIFEST_SHA256
        and len(manifest["base_profiles"]) == 11
        and len(manifest["packet_variants"]) == 44
        and all(
            entry["propagate_in_prospective_uniform_suite"]
            for entry in manifest["packet_variants"]
        )
        and all(
            not entry["propagate_in_prospective_uniform_suite"]
            for entry in manifest["nonbinding_stress_controls"]
        )
        and tuple(contract["binding_grids"]) == LABELS
        and float(contract["time_horizon_s"]) == TIME_HORIZON_S
        and contract["all_manifest_variants_binding"]
        and contract["exact_boundary_semigroup_integral_required"]
        and contract["physical_export_vector_required"]
        and contract["no_threshold_changes_after_propagation"]
    )
    if not verified:
        raise RuntimeError("frozen c6b manifest verification failed")
    return manifest, {
        "declared_sha256": declared_hash,
        "calculated_sha256": calculated_hash,
        "packet_variant_count": len(manifest["packet_variants"]),
        "base_profile_count": len(manifest["base_profiles"]),
        "stress_control_count": len(
            manifest["nonbinding_stress_controls"]
        ),
        "verified": verified,
    }


def _build_variant_directions(
    manifest: dict,
) -> tuple[dict, dict, dict, dict, dict]:
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
    base_names = tuple(
        entry["source_probe"] for entry in manifest["base_profiles"]
    )
    if base_names != tuple(c6a2.PROBE_NAMES):
        raise RuntimeError("manifest base ordering changed")
    base_lookup = {name: index for index, name in enumerate(base_names)}
    variant_entries = manifest["packet_variants"]
    multipliers = np.asarray(
        [
            float(entry["sign"]) * float(entry["amplitude_factor"])
            for entry in variant_entries
        ],
        dtype=float,
    )
    base_indices = np.asarray(
        [
            base_lookup[entry["base_id"].removeprefix("base::")]
            for entry in variant_entries
        ],
        dtype=int,
    )
    variants: dict[str, dict[str, np.ndarray]] = {}
    maximum_projection_hash_defect = 0.0
    for label in LABELS:
        columns = np.asarray(
            configurations[label]["primitive_column_scales"],
            dtype=float,
        ).ravel()
        primary_base_physical = [
            np.asarray(
                base_directions[label][name]["primary_physical"],
                dtype=float,
            )
            for name in base_names
        ]
        secondary_base = np.column_stack(
            [
                base_directions[label][name]["secondary_scaled"]
                for name in base_names
            ]
        )
        primary_variant_physical = [
            multipliers[index]
            * primary_base_physical[base_indices[index]]
            for index in range(len(multipliers))
        ]
        primary_variants = np.column_stack(
            [
                (physical.ravel() / columns)
                for physical in primary_variant_physical
            ]
        )
        variants[label] = {
            "primary_scaled": primary_variants,
            "secondary_base_scaled": secondary_base,
        }
        if label == LABELS[0]:
            for index, entry in enumerate(variant_entries):
                physical = primary_variant_physical[index]
                matches = (
                    causal_array_sha256(physical)
                    == entry["projection_sha256"]
                )
                maximum_projection_hash_defect = max(
                    maximum_projection_hash_defect,
                    0.0 if matches else 1.0,
                )
    report = {
        "continuum_construction_passed": construction["passed"],
        "characteristic_field_passed": characteristic_report["passed"],
        "probe_replay_passed": probe_report["passed"],
        "maximum_N128_projection_hash_defect": (
            maximum_projection_hash_defect
        ),
        "passed": bool(
            construction["passed"]
            and characteristic_report["passed"]
            and probe_report["passed"]
            and maximum_projection_hash_defect == 0.0
        ),
    }
    arrays = {
        **characteristic_arrays,
        **probe_arrays,
        "variant_multipliers": multipliers,
        "variant_base_indices": base_indices,
    }
    variant_metadata = {
        "base_names": base_names,
        "base_indices": base_indices,
        "multipliers": multipliers,
        "packet_ids": tuple(
            entry["packet_id"] for entry in variant_entries
        ),
    }
    return (
        configurations,
        construction_arrays,
        variants,
        variant_metadata,
        {"report": report, "arrays": arrays},
    )


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
        print(f"WP10c9d6c6c: propagate 44 variants on {label}", flush=True)
        configuration = configurations[label]
        generator = np.asarray(
            tangents[label].scaled_generator_per_s,
            dtype=float,
        )
        primary = np.asarray(
            directions[label]["primary_scaled"],
            dtype=float,
        )
        secondary = np.asarray(
            directions[label]["secondary_base_scaled"],
            dtype=float,
        )
        initial = np.column_stack((primary, secondary))
        trace = float(np.trace(generator))
        scaled = np.asarray(
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
        primary_scaled = scaled[:, :, : primary.shape[1]]
        secondary_scaled = scaled[:, :, primary.shape[1] :]
        half = np.asarray(
            expm_multiply(
                0.5 * TIME_HORIZON_S * generator,
                primary,
                traceA=0.5 * TIME_HORIZON_S * trace,
            ),
            dtype=float,
        )
        restarted = np.asarray(
            expm_multiply(
                0.5 * TIME_HORIZON_S * generator,
                half,
                traceA=0.5 * TIME_HORIZON_S * trace,
            ),
            dtype=float,
        )
        exact = causal_exact_semigroup_integral_history(
            generator,
            primary_scaled,
            primary,
        )
        observable = np.asarray(observable_maps[label], dtype=float)
        signals = np.einsum("tnp,on->pto", primary_scaled, observable)
        cumulative = np.einsum(
            "tnp,on->pto",
            exact.integrated_states,
            observable,
        )
        cumulative_corrections = np.einsum(
            "tnp,on->pto",
            exact.correction_states,
            observable,
        )
        columns = np.asarray(
            configuration["primitive_column_scales"],
            dtype=float,
        ).ravel()
        cells = configuration["context"].grid.centers.size
        primary_physical = np.transpose(
            primary_scaled * columns[None, :, None],
            (2, 0, 1),
        ).reshape(primary.shape[1], TIME_SAMPLE_COUNT, cells, 5)
        secondary_physical = np.transpose(
            secondary_scaled * columns[None, :, None],
            (2, 0, 1),
        ).reshape(secondary.shape[1], TIME_SAMPLE_COUNT, cells, 5)
        restart_physical = np.transpose(
            restarted * columns[:, None],
            (1, 0),
        ).reshape(primary.shape[1], cells, 5)
        propagated[label] = {
            "times": times,
            "primary_physical": primary_physical,
            "secondary_base_physical": secondary_physical,
            "restart_physical": restart_physical,
            "signals": signals,
            "cumulative_signals": cumulative,
            "cumulative_corrections": cumulative_corrections,
            "integral_relative_solve_residuals": (
                exact.relative_solve_residuals.T
            ),
        }
        report[label] = {
            "cell_count": int(cells),
            "variant_count": int(primary.shape[1]),
            "secondary_base_count": int(secondary.shape[1]),
            "maximum_exact_integral_relative_solve_residual": (
                exact.maximum_relative_solve_residual
            ),
        }
    return propagated, report


def _history_norm(
    values: np.ndarray,
    scales: np.ndarray,
    times: np.ndarray,
) -> float:
    normalized = np.asarray(values, dtype=float) / np.asarray(
        scales,
        dtype=float,
    )[None, :]
    weights = causal_trapezoid_weights(times)
    return float(
        np.sqrt(
            np.einsum("to,to,t->", normalized, normalized, weights)
        )
    )


def _metric_payload(metrics) -> dict:
    indices = np.asarray(metrics.significant_components, dtype=int)
    return {
        "passed": metrics.passed,
        "significant_components": [
            OBSERVABLE_NAMES[index] for index in indices
        ],
        "observed_rms_order": metrics.observed_rms_order,
        "observed_maximum_order": metrics.observed_maximum_order,
        "component_orders": {
            OBSERVABLE_NAMES[index]: float(metrics.component_orders[position])
            for position, index in enumerate(indices)
        },
        "minimum_significant_component_order": (
            metrics.minimum_significant_component_order
        ),
        "coarse_medium_rms_difference": (
            metrics.coarse_medium_rms_difference
        ),
        "medium_fine_rms_difference": (
            metrics.medium_fine_rms_difference
        ),
        "maximum_fine_normalized_difference": (
            metrics.maximum_fine_normalized_difference
        ),
        "history_cosine": metrics.history_cosine,
        "refinement_error_cosine": metrics.refinement_error_cosine,
    }


def _relative_defect(candidate: np.ndarray, reference: np.ndarray) -> float:
    left = np.asarray(candidate, dtype=float)
    right = np.asarray(reference, dtype=float)
    scale = max(
        float(np.linalg.norm(left)),
        float(np.linalg.norm(right)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(left - right) / scale)


def _comparison_report(
    manifest: dict,
    configurations: dict,
    construction_arrays: dict[str, np.ndarray],
    metadata: dict,
    propagated: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    times = np.asarray(propagated[LABELS[0]]["times"], dtype=float)
    contract = manifest["prospective_propagation_contract"]
    export_gates = contract["instantaneous_and_cumulative_gates"]
    state_gates = contract["state_reference_gates"]
    with np.load(C3_ARRAYS, allow_pickle=False) as source:
        observable_scales = np.asarray(
            source["fixed_physical_observable_scales"],
            dtype=float,
        )
    field_scales = np.asarray(
        construction_arrays["continuum_perturbation_field_scales"],
        dtype=float,
    )
    grids = {
        label: configurations[label]["context"].grid for label in LABELS
    }
    measures = {
        label: np.asarray(grids[label].cell_measures, dtype=float)
        for label in LABELS
    }
    coarse = propagated[LABELS[0]]["primary_physical"]
    medium = causal_restrict_proper_cell_averages(
        propagated[LABELS[1]]["primary_physical"],
        measures[LABELS[1]],
        refinement_factor=2,
    )
    fine = causal_restrict_proper_cell_averages(
        propagated[LABELS[2]]["primary_physical"],
        measures[LABELS[2]],
        refinement_factor=4,
    )
    fine_secondary_base = causal_restrict_proper_cell_averages(
        propagated[LABELS[2]]["secondary_base_physical"],
        measures[LABELS[2]],
        refinement_factor=4,
    )
    fine_restart = causal_restrict_proper_cell_averages(
        propagated[LABELS[2]]["restart_physical"],
        measures[LABELS[2]],
        refinement_factor=4,
    )

    packet_ids = metadata["packet_ids"]
    base_indices = metadata["base_indices"]
    multipliers = metadata["multipliers"]
    reports = {}
    maximum_scaling_defect = 0.0
    arrays: dict[str, np.ndarray] = {
        "times": times,
        "fixed_physical_observable_scales": observable_scales,
        "field_scales": field_scales,
    }
    instantaneous_metric_matrix = np.empty((len(packet_ids), 6))
    cumulative_metric_matrix = np.empty((len(packet_ids), 6))
    state_metric_matrix = np.empty((len(packet_ids), 8))
    base_variant_indices = {}
    for base_index, base_name in enumerate(metadata["base_names"]):
        matches = np.flatnonzero(
            (base_indices == base_index) & (multipliers == 1.0)
        )
        if matches.size != 1:
            raise RuntimeError(f"missing +1 base variant for {base_name}")
        base_variant_indices[base_index] = int(matches[0])

    for packet_index, packet_id in enumerate(packet_ids):
        instantaneous = causal_packet_history_metrics(
            propagated[LABELS[0]]["signals"][packet_index],
            propagated[LABELS[1]]["signals"][packet_index],
            propagated[LABELS[2]]["signals"][packet_index],
            physical_scales=observable_scales,
            relative_activity=MINIMUM_RELATIVE_ACTIVITY,
            minimum_rms_order=export_gates["minimum_rms_order"],
            minimum_maximum_order=export_gates[
                "minimum_maximum_order"
            ],
            minimum_significant_component_order=export_gates[
                "minimum_significant_component_order"
            ],
            maximum_fine_normalized_difference=export_gates[
                "maximum_fine_normalized_difference"
            ],
            minimum_history_cosine=export_gates[
                "minimum_history_cosine"
            ],
            minimum_refinement_error_cosine=export_gates[
                "minimum_refinement_error_cosine"
            ],
        )
        cumulative = causal_packet_history_metrics(
            propagated[LABELS[0]]["cumulative_signals"][packet_index],
            propagated[LABELS[1]]["cumulative_signals"][packet_index],
            propagated[LABELS[2]]["cumulative_signals"][packet_index],
            physical_scales=observable_scales,
            relative_activity=MINIMUM_RELATIVE_ACTIVITY,
            minimum_rms_order=export_gates["minimum_rms_order"],
            minimum_maximum_order=export_gates[
                "minimum_maximum_order"
            ],
            minimum_significant_component_order=export_gates[
                "minimum_significant_component_order"
            ],
            maximum_fine_normalized_difference=export_gates[
                "maximum_fine_normalized_difference"
            ],
            minimum_history_cosine=export_gates[
                "minimum_history_cosine"
            ],
            minimum_refinement_error_cosine=export_gates[
                "minimum_refinement_error_cosine"
            ],
        )
        richardson = causal_windowed_richardson_reference(
            coarse[packet_index],
            medium[packet_index],
            fine[packet_index],
            times=times,
            coarse_cell_measures=measures[LABELS[0]],
            field_scales=field_scales,
        )
        fine_state_difference = max(
            richardson.medium_fine_history_norm,
            np.finfo(float).tiny,
        )
        secondary = (
            multipliers[packet_index]
            * fine_secondary_base[base_indices[packet_index]]
        )
        projection_ratio = (
            causal_field_history_norm(
                fine[packet_index] - secondary,
                cell_measures=measures[LABELS[0]],
                field_scales=field_scales,
                time_weights=causal_trapezoid_weights(times),
            )
            / fine_state_difference
        )
        restart_ratio = (
            causal_field_history_norm(
                np.stack(
                    (
                        fine_restart[packet_index]
                        - fine[packet_index, -1],
                    )
                    * 2,
                    axis=0,
                ),
                cell_measures=measures[LABELS[0]],
                field_scales=field_scales,
                time_weights=np.ones(2),
            )
            / fine_state_difference
        )

        medium_boundary = propagated[LABELS[1]][
            "cumulative_signals"
        ][packet_index, :, :BOUNDARY_OBSERVABLE_COUNT]
        fine_boundary = propagated[LABELS[2]][
            "cumulative_signals"
        ][packet_index, :, :BOUNDARY_OBSERVABLE_COUNT]
        fine_boundary_difference = max(
            _history_norm(
                medium_boundary - fine_boundary,
                observable_scales[:BOUNDARY_OBSERVABLE_COUNT],
                times,
            ),
            np.finfo(float).tiny,
        )
        boundary_correction = propagated[LABELS[2]][
            "cumulative_corrections"
        ][packet_index, :, :BOUNDARY_OBSERVABLE_COUNT]
        boundary_ratio = (
            _history_norm(
                boundary_correction,
                observable_scales[:BOUNDARY_OBSERVABLE_COUNT],
                times,
            )
            / fine_boundary_difference
        )
        state_parent_replay = bool(
            richardson.observed_order
            >= c6a2.MINIMUM_CROSS_GRID_ORDER
            and richardson.minimum_significant_component_order
            >= c6a2.MINIMUM_COMPONENT_ORDER
            and richardson.refinement_error_cosine
            >= c6a2.MINIMUM_REFINEMENT_ERROR_COSINE
        )
        state_passed = bool(
            state_parent_replay
            and richardson.maximum_coarse_reference_relative_error
            <= state_gates["maximum_N128_Richardson_error"]
            and richardson.reference_choice_to_fine_difference_ratio
            <= state_gates[
                "maximum_reference_uncertainty_to_fine_difference"
            ]
            and projection_ratio
            <= state_gates[
                "maximum_projection_uncertainty_to_fine_difference"
            ]
            and restart_ratio
            <= state_gates[
                "maximum_restart_uncertainty_to_fine_difference"
            ]
            and boundary_ratio
            <= state_gates[
                "maximum_boundary_integral_uncertainty_to_fine_difference"
            ]
        )
        base_variant = base_variant_indices[base_indices[packet_index]]
        expected_factor = multipliers[packet_index]
        scaling_defects = {}
        for label in LABELS:
            for quantity in (
                "primary_physical",
                "signals",
                "cumulative_signals",
            ):
                defect = _relative_defect(
                    propagated[label][quantity][packet_index],
                    expected_factor
                    * propagated[label][quantity][base_variant],
                )
                scaling_defects[f"{label}::{quantity}"] = defect
                maximum_scaling_defect = max(
                    maximum_scaling_defect,
                    defect,
                )
        packet_passed = bool(
            instantaneous.passed
            and cumulative.passed
            and state_passed
            and max(scaling_defects.values())
            <= MAXIMUM_PROPAGATION_SCALING_DEFECT
        )
        reports[packet_id] = {
            "manifest_variant": manifest["packet_variants"][packet_index],
            "instantaneous_exports": _metric_payload(instantaneous),
            "cumulative_exports": _metric_payload(cumulative),
            "state_reference": {
                "parent_state_convergence_replayed": state_parent_replay,
                "observed_order": richardson.observed_order,
                "minimum_significant_component_order": (
                    richardson.minimum_significant_component_order
                ),
                "refinement_error_cosine": (
                    richardson.refinement_error_cosine
                ),
                "maximum_N128_Richardson_error": (
                    richardson.maximum_coarse_reference_relative_error
                ),
                "reference_uncertainty_to_fine_difference": (
                    richardson.reference_choice_to_fine_difference_ratio
                ),
                "projection_uncertainty_to_fine_difference": (
                    projection_ratio
                ),
                "restart_uncertainty_to_fine_difference": restart_ratio,
                "boundary_integral_uncertainty_to_fine_difference": (
                    boundary_ratio
                ),
                "passed": state_passed,
            },
            "propagation_scaling_defects": scaling_defects,
            "passed": packet_passed,
        }
        instantaneous_metric_matrix[packet_index] = (
            instantaneous.observed_rms_order,
            instantaneous.observed_maximum_order,
            instantaneous.minimum_significant_component_order,
            instantaneous.maximum_fine_normalized_difference,
            instantaneous.history_cosine,
            instantaneous.refinement_error_cosine,
        )
        cumulative_metric_matrix[packet_index] = (
            cumulative.observed_rms_order,
            cumulative.observed_maximum_order,
            cumulative.minimum_significant_component_order,
            cumulative.maximum_fine_normalized_difference,
            cumulative.history_cosine,
            cumulative.refinement_error_cosine,
        )
        state_metric_matrix[packet_index] = (
            richardson.observed_order,
            richardson.minimum_significant_component_order,
            richardson.refinement_error_cosine,
            richardson.maximum_coarse_reference_relative_error,
            richardson.reference_choice_to_fine_difference_ratio,
            projection_ratio,
            restart_ratio,
            boundary_ratio,
        )

    base_indices_to_store = np.asarray(
        [base_variant_indices[index] for index in range(len(base_variant_indices))],
        dtype=int,
    )
    for label in LABELS:
        arrays[f"{label}__base_instantaneous_exports"] = propagated[label][
            "signals"
        ][base_indices_to_store]
        arrays[f"{label}__base_cumulative_exports"] = propagated[label][
            "cumulative_signals"
        ][base_indices_to_store]
        arrays[f"{label}__base_integral_solve_residuals"] = propagated[
            label
        ]["integral_relative_solve_residuals"][base_indices_to_store]
    arrays["instantaneous_metric_matrix"] = instantaneous_metric_matrix
    arrays["cumulative_metric_matrix"] = cumulative_metric_matrix
    arrays["state_metric_matrix"] = state_metric_matrix
    arrays["packet_pass_flags"] = np.asarray(
        [reports[name]["passed"] for name in packet_ids],
        dtype=np.int8,
    )
    report = {
        "packet_reports": reports,
        "packet_count": len(packet_ids),
        "maximum_propagation_scaling_defect": maximum_scaling_defect,
        "failed_packets": [
            packet_id
            for packet_id in packet_ids
            if not reports[packet_id]["passed"]
        ],
        "all_packets_passed": all(
            reports[packet_id]["passed"] for packet_id in packet_ids
        ),
        "passed": bool(
            all(reports[packet_id]["passed"] for packet_id in packet_ids)
            and maximum_scaling_defect
            <= MAXIMUM_PROPAGATION_SCALING_DEFECT
        ),
    }
    return report, arrays


def _config(manifest: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "operator_change": False,
        "manifest_sha256": FROZEN_MANIFEST_SHA256,
        "time_horizon_s": TIME_HORIZON_S,
        "time_sample_count": TIME_SAMPLE_COUNT,
        "labels": LABELS,
        "observable_names": OBSERVABLE_NAMES,
        "minimum_relative_activity": MINIMUM_RELATIVE_ACTIVITY,
        "maximum_propagation_scaling_defect": (
            MAXIMUM_PROPAGATION_SCALING_DEFECT
        ),
        "frozen_contract": manifest[
            "prospective_propagation_contract"
        ],
        "exact_cumulative_method": (
            "G_inverse_expm_tG_minus_identity_with_"
            "one_step_iterative_refinement"
        ),
    }


def run() -> dict:
    started = time.perf_counter()
    identity = _validate_analyzed_git_identity()
    manifest, manifest_report = _load_and_verify_manifest()
    (
        configurations,
        construction_arrays,
        directions,
        metadata,
        construction_bundle,
    ) = _build_variant_directions(manifest)
    if not construction_bundle["report"]["passed"]:
        raise RuntimeError("c6c packet reconstruction failed")
    print("WP10c9d6c6c: build unchanged monolithic tangents", flush=True)
    tangents, observable_maps, method_reports, _baselines = (
        c3._build_tangents(configurations, construction_arrays)
    )
    method_passed = bool(
        all(method_reports[label]["passed"] for label in LABELS)
        and construction_bundle["report"]["passed"]
    )
    propagated, propagation_report = _propagate(
        configurations,
        tangents,
        observable_maps,
        directions,
    )
    comparison, comparison_arrays = _comparison_report(
        manifest,
        configurations,
        construction_arrays,
        metadata,
        propagated,
    )
    if not method_passed:
        classification = "prospective_uniform_packet_method_failed"
        authorized_next = "none"
    elif not comparison["passed"]:
        classification = "prospective_uniform_packet_validation_failed"
        authorized_next = "freeze_failed_variant_and_localize"
    else:
        classification = (
            "prospective_uniform_resolved_packet_class_certified_"
            "embedded_discrimination_authorized"
        )
        authorized_next = "WP10c9d6c7_embedded_packet_discrimination"
    passed = bool(
        method_passed
        and comparison["passed"]
        and authorized_next
        == "WP10c9d6c7_embedded_packet_discrimination"
    )

    arrays = {
        **construction_bundle["arrays"],
        **comparison_arrays,
    }
    source_hashes, source_manifest = _source_manifest()
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DECISIVE_ARRAYS, **arrays)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        **identity,
        "classification": classification,
        "passed": passed,
        "authorized_next": authorized_next,
        "audit_executed": True,
        "operator_changed": False,
        "parent_classification": (
            "packet_definition_manifest_frozen_"
            "uniform_propagation_authorized"
        ),
        "parent_classification_preserved": True,
        "manifest_report": manifest_report,
        "configuration": _config(manifest),
        "construction_report": construction_bundle["report"],
        "method_reports": {
            label: method_reports[label] for label in LABELS
        },
        "method_passed": method_passed,
        "propagation_report": propagation_report,
        "comparison_report": comparison,
        "prospective_uniform_packet_class_certified": passed,
        "embedded_export_discrimination_authorized": passed,
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
            for name, values in sorted(arrays.items())
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "scientific_status": "DIAGNOSTIC ONLY" if passed else "REJECTED",
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        **identity,
        "implementation_base_tree": _git_value(
            "rev-parse",
            "HEAD^{tree}",
        ),
        "working_tree_status": _git_value("status", "--short"),
        "command": (
            "PYTHONPATH=src:scripts python "
            "scripts/run_causal_inner_packet_validation_wp10c9d6c6c.py"
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
                MANIFEST_PATH,
            )
        },
    }
    _write_json(CONFIG_PATH, _config(manifest))
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
                    "packet_count": comparison["packet_count"],
                    "failed_packets": comparison["failed_packets"],
                    "maximum_propagation_scaling_defect": comparison[
                        "maximum_propagation_scaling_defect"
                    ],
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
        raise RuntimeError("c6c canonical evidence is unavailable")
    manifest, _report = _load_and_verify_manifest()
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    with np.load(DECISIVE_ARRAYS, allow_pickle=False) as source:
        arrays = {
            name: np.array(source[name], copy=True)
            for name in source.files
        }
    source_hashes, source_manifest = _source_manifest()
    summary["configuration"] = _config(manifest)
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
    _write_json(CONFIG_PATH, _config(manifest))
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
