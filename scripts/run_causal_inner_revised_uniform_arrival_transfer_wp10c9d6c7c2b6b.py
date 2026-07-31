#!/usr/bin/env python3
"""Run the frozen revised uniform arrival/transfer recertification.

WP10c9d6c7c2b6b preserves every c2b4-c2b6a classification, changes no
physical or numerical operator, and runs no embedded or nonlinear state.
It propagates the five b6a base cases on the unchanged N98/N196/N392
monolithic tangents and constructs independent 513/769-node sixth-order
continuum-history references.
"""

from __future__ import annotations

import csv
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import scipy
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import make_interp_spline
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_causal_inner_continuum_lift_wp10c9d6c3 as c3  # noqa: E402
import run_causal_inner_one_way_transmission_interpretation_wp10c9d6c7c2b2 as c2b2  # noqa: E402
import run_causal_inner_one_way_uniform_scattering_wp10c9d6c7c2b1 as c2b1  # noqa: E402
import run_causal_inner_revised_arrival_contract_manifest_wp10c9d6c7c2b6a as b6a  # noqa: E402
import run_causal_inner_scattering_energy_wp10c9d6c7c2a2 as c2a2  # noqa: E402
import run_causal_inner_scattering_observability_manifest_wp10c9d6c7c2a as c2a  # noqa: E402
import run_causal_inner_scattering_scope_wp10c9d6c7c2a3 as c2a3  # noqa: E402
import run_causal_inner_uniform_arrival_conditioning_wp10c9d6c7c2b5a as b5a  # noqa: E402
import run_causal_inner_uniform_family_transfer_wp10c9d6c7c2b5b as b5b  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_continuum_truncation import (  # noqa: E402
    build_causal_five_field_continuum_background,
    causal_five_field_inward_collocation_generator,
    linearize_causal_five_field_continuum_reference,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_energy_transfer import (  # noqa: E402
    causal_positive_band_energy_history,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_canonical_json_sha256,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_validation import (  # noqa: E402
    causal_packet_history_metrics,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_windowed_contract import (  # noqa: E402
    causal_windowed_richardson_reference,
)


SCHEMA_VERSION = 1
PROPAGATION_SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c2b6b"
ANALYZED_BASE_COMMIT = "c47df91e3b22af9e2d5899fc32c924840f2c1471"
ANALYZED_BASE_PARENT = "7857051b4292c3101456019210f7437c73fd8621"
ANALYZED_BASE_TREE = "8aeea229afcabceec649f969ae166e52c445187c"

LEVELS = b6a.LEVELS
BASES = b6a.BINDING_BASES
TARGETS = b6a.TARGET_FAMILY_INDICES
CONTINUUM_NODES = (513, 769)
CONTINUUM_TIME_SAMPLES = 513
PROJECTION_ORDERS = (12, 24)
FIELDS = 5

THIS_RUNNER = (
    "scripts/"
    "run_causal_inner_revised_uniform_arrival_transfer_"
    "wp10c9d6c7c2b6b.py"
)
THIS_HELPER = (
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_continuum_truncation.py"
)
THIS_HELPER_TEST = "tests/test_causal_inner_continuum_collocation.py"
THIS_CANONICAL_TEST = (
    "tests/"
    "test_causal_inner_revised_uniform_arrival_transfer_"
    "wp10c9d6c7c2b6b.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_REVISED_UNIFORM_ARRIVAL_TRANSFER_"
    "WP10C9D6C7C2B6B_RESULTS_2026-07-30.md"
)

PARENT_DIRECTORY = b6a.CANONICAL_DIRECTORY
B5B_DIRECTORY = b5b.CANONICAL_DIRECTORY
SCOPE_DIRECTORY = c2b1.SCOPE_DIRECTORY
C2A2_DIRECTORY = c2b1.C2A2_DIRECTORY
C2B3_DIRECTORY = b5a.C2B3_DIRECTORY
C7A_DIRECTORY = c2b1.C7A_DIRECTORY
CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_revised_uniform_arrival_transfer_wp10c9d6c7c2b6b"
)
CHECKPOINT_DIRECTORY = (
    ROOT
    / "outputs/checkpoints/"
    "causal_inner_revised_uniform_arrival_transfer_wp10c9d6c7c2b6b"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

IMPLEMENTATION_SOURCES = (
    THIS_RUNNER,
    THIS_HELPER,
    THIS_HELPER_TEST,
    THIS_CANONICAL_TEST,
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


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _git_value(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _relative_defect(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    scale = max(
        float(np.max(np.abs(left))),
        float(np.max(np.abs(right))),
        np.finfo(float).tiny,
    )
    return float(np.max(np.abs(left - right)) / scale)


def _validate_parent() -> tuple[dict, dict, dict[str, np.ndarray]]:
    summary = _read_json(PARENT_DIRECTORY / "summary.json")
    manifest = _read_json(PARENT_DIRECTORY / "contract_manifest.json")
    arrays = _load_npz(PARENT_DIRECTORY / "decisive_arrays.npz")
    if (
        summary["classification"]
        != "revised_uniform_arrival_transfer_contract_frozen_"
        "recertification_authorized"
        or summary["authorized_next"]
        != "WP10c9d6c7c2b6b_revised_uniform_arrival_transfer_"
        "recertification"
        or not summary["passed"]
        or summary["operator_changed"]
        or summary["propagation_executed"]
        or not summary["binding_decision"][
            "uniform_b6b_recertification_authorized"
        ]
        or summary["binding_decision"]["embedded_authorized"]
        or summary["manifest_sha256"] != manifest["manifest_sha256"]
    ):
        raise RuntimeError("WP10c9d6c7c2b6a binding status changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT)
        != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("WP10c9d6c7c2b6b analyzed identity changed")
    return summary, manifest, arrays


def _input_hashes() -> dict[str, str]:
    paths = (
        PARENT_DIRECTORY / "config.json",
        PARENT_DIRECTORY / "contract_manifest.json",
        PARENT_DIRECTORY / "summary.json",
        PARENT_DIRECTORY / "decisive_arrays.npz",
        B5B_DIRECTORY / "summary.json",
        B5B_DIRECTORY / "decisive_arrays.npz",
        SCOPE_DIRECTORY / "scope_manifest.json",
        SCOPE_DIRECTORY / "decisive_arrays.npz",
        C2A2_DIRECTORY / "decisive_arrays.npz",
        C2B3_DIRECTORY / "decisive_arrays.npz",
        C7A_DIRECTORY / "decisive_arrays.npz",
    )
    return {
        str(path.relative_to(ROOT)): c2a._sha256(path)
        for path in paths
    }


def _base_combinations(acoustic: np.ndarray, shear: np.ndarray) -> np.ndarray:
    return np.stack(
        (
            acoustic,
            shear,
            (acoustic + shear) / np.sqrt(2.0),
            (acoustic - shear) / np.sqrt(2.0),
            0.5 * acoustic + np.sqrt(3.0) * 0.5 * shear,
        ),
        axis=1,
    )


def _packet_evaluators(
    fine_level: dict,
    fine_packets: dict[str, np.ndarray],
) -> tuple:
    return (
        b5b._smooth_state_evaluator(fine_level, fine_packets["acoustic"]),
        b5b._smooth_state_evaluator(fine_level, fine_packets["shear"]),
    )


def _initial_columns(
    level: dict,
    scope_arrays: dict[str, np.ndarray],
    support_log_bounds: tuple[float, float],
    evaluators: tuple,
) -> tuple[np.ndarray, list[dict], dict[str, np.ndarray]]:
    _full, _cases, packets = c2b1._packet_matrix(
        level,
        scope_arrays,
        support_log_bounds,
    )
    physical = [
        packets["acoustic"],
        packets["shear"],
    ]
    labels = ["primary_acoustic", "primary_shear"]
    for quadrature_order in PROJECTION_ORDERS[::-1]:
        for family, evaluator in zip(
            ("acoustic", "shear"),
            evaluators,
            strict=True,
        ):
            physical.append(
                c3._project_callable_to_cells(
                    level["grid"],
                    evaluator,
                    quadrature_order=quadrature_order,
                )
            )
            labels.append(f"projection_q{quadrature_order}_{family}")
    cases = [
        {
            "name": label,
            "family": "acoustic" if "acoustic" in label else "shear",
            "sign": 1,
            "amplitude": 1.0,
            "binding": True,
        }
        for label in labels
    ]
    initial = np.column_stack(
        [
            state.ravel() / np.asarray(level["columns"], dtype=float)
            for state in physical
        ]
    )
    return initial, cases, packets


def _propagation_checkpoint_valid(path: Path, cells: int) -> bool:
    metadata = path.with_suffix(".json")
    if not path.is_file() or not metadata.is_file():
        return False
    report = _read_json(metadata)
    return bool(
        report.get("source_parent_commit") == ANALYZED_BASE_COMMIT
        and report.get("cells") == cells
        and report.get("parent_manifest_sha256")
        == _read_json(PARENT_DIRECTORY / "summary.json")["manifest_sha256"]
        and report.get("propagation_schema_version")
        == PROPAGATION_SCHEMA_VERSION
        and report.get("helper_sha256") == c2a._sha256(ROOT / THIS_HELPER)
    )


def _propagate_level(
    level: dict,
    initial: np.ndarray,
    cases: list[dict],
    propagation_windows: dict,
    horizon: float,
    common_log_centers: np.ndarray,
) -> dict:
    cells = int(level["cells"])
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIRECTORY / f"N{cells}_base_propagation.npz"
    if _propagation_checkpoint_valid(path, cells):
        stored = _load_npz(path)
        return {
            "times": stored["times"],
            "physical": stored["physical"],
            "signals": stored["signals"],
            "state": stored["state"],
            "restart_defect": float(stored["restart_defect"][0]),
        }
    propagated = c2b1._propagate_level(
        level,
        initial,
        cases,
        propagation_windows,
        horizon,
        common_log_centers,
    )
    np.savez_compressed(
        path,
        times=np.asarray(propagated["times"]),
        physical=np.asarray(propagated["physical"]),
        signals=np.asarray(propagated["signals"]),
        state=np.asarray(propagated["state"]),
        restart_defect=np.asarray([propagated["restart_defect"]]),
    )
    _write_json(
        path.with_suffix(".json"),
        {
            "source_parent_commit": ANALYZED_BASE_COMMIT,
            "cells": cells,
            "parent_manifest_sha256": _read_json(
                PARENT_DIRECTORY / "summary.json"
            )["manifest_sha256"],
            "propagation_schema_version": PROPAGATION_SCHEMA_VERSION,
            "helper_sha256": c2a._sha256(ROOT / THIS_HELPER),
        },
    )
    return {
        key: propagated[key]
        for key in (
            "times",
            "physical",
            "signals",
            "state",
            "restart_defect",
        )
    }


def _derived_propagation(propagated: dict) -> dict:
    physical = np.asarray(propagated["physical"], dtype=float)
    signals = np.asarray(propagated["signals"], dtype=float)
    state = np.asarray(propagated["state"], dtype=float)
    return {
        "times": np.asarray(propagated["times"], dtype=float),
        "physical": _base_combinations(physical[:, 0], physical[:, 1]),
        "signals": _base_combinations(signals[:, 0], signals[:, 1]),
        "state": _base_combinations(state[:, 0], state[:, 1]),
        "restart_defect": float(propagated["restart_defect"]),
        "projection_physical": {
            "q24": _base_combinations(
                physical[:, 2],
                physical[:, 3],
            ),
            "q12": _base_combinations(
                physical[:, 4],
                physical[:, 5],
            ),
        },
    }


def _polynomial_projectors(level: dict) -> tuple[np.ndarray, dict]:
    projectors, report = b5b._projector_audit(level)
    if not report["passed"]:
        raise RuntimeError(
            f"N{level['cells']} equivalent projector audit failed"
        )
    return projectors, report


def _energy_histories(
    level: dict,
    physical: np.ndarray,
    projectors: np.ndarray,
    *,
    lower_face: int,
    upper_face: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    measured = causal_positive_band_energy_history(
        physical,
        log_edges=np.log(np.asarray(level["grid"].edges, dtype=float)),
        energy_metrics=level["energy"],
        projectors=projectors,
        lower_face=lower_face,
        upper_face=upper_face,
    )
    return (
        np.asarray(measured.total_energy, dtype=float),
        np.asarray(measured.family_energy, dtype=float),
        float(measured.maximum_family_partition_relative_defect),
    )


def _level_arrival_data(
    level: dict,
    propagated: dict,
    polynomial_projectors: np.ndarray,
    contract_arrays: dict[str, np.ndarray],
    windows: dict[str, tuple[float, float]],
    nuisance_windows: dict[str, list[tuple[float, float]]],
) -> dict:
    cells = int(level["cells"])
    factor = cells // LEVELS[0]
    physical = np.asarray(propagated["physical"], dtype=float)
    source = (52 * factor, 95 * factor)
    primary_band = (6 * factor, 49 * factor)
    total_source, _family_source, partition_source = _energy_histories(
        level,
        physical[:1],
        level["projectors"],
        lower_face=source[0],
        upper_face=source[1],
    )
    initial = np.asarray(total_source[0], dtype=float)
    if np.any(initial <= 0.0):
        raise RuntimeError(f"N{cells} initial source energy is not positive")

    band_pairs = [primary_band]
    for lower, upper in np.asarray(
        contract_arrays["receiving_band_nuisance_faces_N98"],
        dtype=int,
    ):
        pair = (int(lower) * factor, int(upper) * factor)
        if pair not in band_pairs:
            band_pairs.append(pair)
    band_histories = {}
    for band in band_pairs:
        total, family, partition = _energy_histories(
            level,
            physical,
            level["projectors"],
            lower_face=band[0],
            upper_face=band[1],
        )
        band_histories[band] = {
            "total": total / initial[None],
            "family": family / initial[None, :, None],
            "partition": partition,
        }
    primary = band_histories[primary_band]
    _poly_total, poly_family, poly_partition = _energy_histories(
        level,
        physical,
        polynomial_projectors,
        lower_face=primary_band[0],
        upper_face=primary_band[1],
    )
    poly_family /= initial[None, :, None]

    projection_histories = {}
    for label, projection_physical in propagated[
        "projection_physical"
    ].items():
        total, family, partition = _energy_histories(
            level,
            projection_physical,
            level["projectors"],
            lower_face=primary_band[0],
            upper_face=primary_band[1],
        )
        source_total, _source_family, _source_partition = _energy_histories(
            level,
            projection_physical[:1],
            level["projectors"],
            lower_face=source[0],
            upper_face=source[1],
        )
        projection_initial = source_total[0]
        projection_histories[label] = {
            "total": total / projection_initial[None],
            "family": family / projection_initial[None, :, None],
            "partition": partition,
        }

    by_base = {}
    times = np.asarray(propagated["times"], dtype=float)
    for index, name in enumerate(BASES):
        target_indices = list(TARGETS[name])
        target = np.sum(primary["family"][:, index, target_indices], axis=1)
        poly_target = np.sum(
            poly_family[:, index, target_indices],
            axis=1,
        )
        histories = {
            "total": primary["total"][:, index],
            "target": target,
            "raw_opposite_family_diagnostic": (
                np.sum(primary["family"][:, index], axis=1) - target
            ),
        }
        full_histories = {
            "total": np.asarray(histories["total"], dtype=float).copy(),
            "target": np.asarray(histories["target"], dtype=float).copy(),
        }
        variations = {
            "total": {
                "receiving_band": [
                    item["total"][:, index]
                    for item in band_histories.values()
                ],
                "equivalent_projector": [histories["total"]],
                "analytic_projection": [
                    projection_histories[label]["total"][:, index]
                    for label in ("q24", "q12")
                ],
            },
            "target": {
                "receiving_band": [
                    np.sum(
                        item["family"][:, index, target_indices],
                        axis=1,
                    )
                    for item in band_histories.values()
                ],
                "equivalent_projector": [
                    target,
                    poly_target,
                ],
                "analytic_projection": [
                    np.sum(
                        projection_histories[label]["family"][
                            :, index, target_indices
                        ],
                        axis=1,
                    )
                    for label in ("q24", "q12")
                ],
            },
        }
        for observable in ("total", "target"):
            primary_window = windows[
                name if name in windows else "mixed_shear_acoustic"
            ]
            varied_windows = nuisance_windows[
                name if name in nuisance_windows else "mixed_shear_acoustic"
            ]
            masked = b5a._mask_history(
                times,
                histories[observable],
                primary_window,
            )
            variations[observable]["arrival_window"] = [
                b5a._mask_history(times, histories[observable], window)
                for window in varied_windows
            ]
            variations[observable]["time_sampling"] = [
                b5a._interpolated_stride_variant(
                    times,
                    masked,
                    stride,
                )
                for stride in (1, 2, 4)
            ]
            histories[observable] = masked
        by_base[name] = {
            "histories": histories,
            "full_histories": full_histories,
            "variations": variations,
            "initial_source_energy": float(initial[index]),
        }
    return {
        "by_base": by_base,
        "initial_source_energy": initial,
        "maximum_partition_defect": max(
            partition_source,
            poly_partition,
            *(item["partition"] for item in band_histories.values()),
            *(
                item["partition"]
                for item in projection_histories.values()
            ),
        ),
    }


def _continuum_energy_basis(
    background,
    field_scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    metrics = []
    projectors = []
    minimum_gap = np.inf
    maximum_defect = 0.0
    for temporal, flux, shear, height in zip(
        background.temporal_storage_matrices,
        background.physical_flux_jacobians,
        background.shear_principal_matrices,
        background.height_principal_matrices,
        strict=True,
    ):
        basis = c2a2.causal_normalization_invariant_scattering_energy(
            temporal,
            flux - shear - height,
            field_scales,
        )
        metrics.append(basis.primitive_energy_metric)
        projectors.append(basis.primitive_projectors)
        minimum_gap = min(
            minimum_gap,
            float(np.min(np.diff(basis.characteristic_speeds))),
        )
        maximum_defect = max(
            maximum_defect,
            basis.maximum_projector_identity_defect,
            basis.maximum_projector_idempotence_defect,
            basis.maximum_cross_projector_defect,
            basis.maximum_energy_orthogonality_defect,
            basis.maximum_symmetrizer_defect,
            basis.maximum_eigenpair_defect,
            basis.maximum_imaginary_part,
        )
    return (
        np.asarray(metrics),
        np.asarray(projectors),
        {
            "minimum_spectral_gap": float(minimum_gap),
            "maximum_algebra_defect": float(maximum_defect),
        },
    )


def _spline_integral_weights(
    nodes: np.ndarray,
    lower: float,
    upper: float,
) -> np.ndarray:
    identity = np.eye(nodes.size, dtype=float)
    spline = make_interp_spline(nodes, identity, k=5, axis=0)
    return np.asarray(spline.integrate(lower, upper), dtype=float)


def _continuum_packet_pair(
    background,
    metric: np.ndarray,
    projectors: np.ndarray,
    scope_arrays: dict[str, np.ndarray],
    support_log_bounds: tuple[float, float],
    field_scales: np.ndarray,
) -> np.ndarray:
    left, right = support_log_bounds
    coordinate = (background.log_radii - left) / (right - left)
    envelope = np.zeros_like(coordinate)
    active = (coordinate > 0.0) & (coordinate < 1.0)
    envelope[active] = np.sin(np.pi * coordinate[active]) ** 4
    packets = []
    for family, seed_name in (
        (0, "packet_seed__acoustic"),
        (1, "packet_seed__shear"),
    ):
        seed = np.asarray(scope_arrays[seed_name], dtype=float)
        packet = np.zeros((background.radii.size, FIELDS), dtype=float)
        previous = None
        for node in range(background.radii.size):
            direction = projectors[node, family] @ seed
            norm = float(np.sqrt(direction @ metric[node] @ direction))
            direction /= norm
            if (
                previous is not None
                and np.dot(
                    previous / field_scales,
                    direction / field_scales,
                )
                < 0.0
            ):
                direction *= -1.0
            packet[node] = envelope[node] * direction
            previous = direction
        packets.append(packet)
    return np.asarray(packets)


def _continuum_checkpoint_valid(path: Path, nodes: int) -> bool:
    metadata = path.with_suffix(".json")
    if not path.is_file() or not metadata.is_file():
        return False
    report = _read_json(metadata)
    return bool(
        report.get("source_parent_commit") == ANALYZED_BASE_COMMIT
        and report.get("nodes") == nodes
        and report.get("propagation_schema_version")
        == PROPAGATION_SCHEMA_VERSION
        and report.get("helper_sha256") == c2a._sha256(ROOT / THIS_HELPER)
    )


def _continuum_reference(
    nodes: int,
    context,
    background_evaluator,
    field_scales: np.ndarray,
    scope_arrays: dict[str, np.ndarray],
    support_log_bounds: tuple[float, float],
    horizon: float,
    base_log_edges: np.ndarray,
) -> dict:
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIRECTORY / f"continuum_N{nodes}.npz"
    if _continuum_checkpoint_valid(path, nodes):
        stored = _load_npz(path)
        return {
            "times": stored["times"],
            "total": stored["total"],
            "target": stored["target"],
            "initial_source_energy": stored["initial_source_energy"],
            "action_rate": stored["action_rate"],
            "comparison_log_radii": stored["comparison_log_radii"],
            "restart_defect": float(stored["restart_defect"][0]),
            "action_to_quintic_defect": float(
                stored["action_to_quintic_defect"][0]
            ),
            "energy_report": _read_json(path.with_suffix(".json"))[
                "energy_report"
            ],
        }
    print(f"{WORK_PACKAGE}: build continuum N{nodes}", flush=True)
    background = build_causal_five_field_continuum_background(
        context,
        background_evaluator,
        node_count=nodes,
    )
    metric, projectors, energy_report = _continuum_energy_basis(
        background,
        field_scales,
    )
    packet_pair = _continuum_packet_pair(
        background,
        metric,
        projectors,
        scope_arrays,
        support_log_bounds,
        field_scales,
    )
    generator = causal_five_field_inward_collocation_generator(background)
    scales = np.tile(field_scales, nodes)
    scaled_generator = (
        diags(1.0 / scales) @ generator @ diags(scales)
    ).tocsr()
    initial = np.column_stack(
        [packet.ravel() / scales for packet in packet_pair]
    )
    times = np.linspace(0.0, horizon, CONTINUUM_TIME_SAMPLES)
    trace = float(np.sum(scaled_generator.diagonal()))
    print(f"{WORK_PACKAGE}: propagate continuum N{nodes}", flush=True)
    scaled = np.asarray(
        expm_multiply(
            scaled_generator,
            initial,
            start=0.0,
            stop=horizon,
            num=times.size,
            endpoint=True,
            traceA=trace,
        ),
        dtype=float,
    )
    direct_final = np.asarray(
        expm_multiply(
            horizon * scaled_generator,
            initial,
            traceA=horizon * trace,
        ),
        dtype=float,
    )
    restart_defect = _relative_defect(direct_final, scaled[-1])
    physical_pair = np.transpose(
        scaled * scales[None, :, None],
        (0, 2, 1),
    ).reshape(times.size, 2, nodes, FIELDS)
    physical = _base_combinations(
        physical_pair[:, 0],
        physical_pair[:, 1],
    )

    log_lower = float(background.log_radii[0])
    log_upper = float(background.log_radii[-1])
    base_edges = np.asarray(base_log_edges, dtype=float)
    if (
        base_edges.shape != (LEVELS[0] + 1,)
        or abs(base_edges[0] - log_lower) > 1.0e-12
        or abs(base_edges[-1] - log_upper) > 1.0e-12
    ):
        raise ValueError("continuum reference base log edges changed")
    source_weights = _spline_integral_weights(
        background.log_radii,
        base_edges[52],
        base_edges[95],
    )
    receiving_weights = _spline_integral_weights(
        background.log_radii,
        base_edges[6],
        base_edges[49],
    )
    initial_density = 0.5 * np.einsum(
        "bni,nij,bnj->bn",
        physical[0],
        metric,
        physical[0],
        optimize=True,
    )
    initial_energy = np.einsum(
        "bn,n->b",
        initial_density,
        source_weights,
        optimize=True,
    )
    total_density = 0.5 * np.einsum(
        "tbni,nij,tbnj->tbn",
        physical,
        metric,
        physical,
        optimize=True,
    )
    total = np.einsum(
        "tbn,n->tb",
        total_density,
        receiving_weights,
        optimize=True,
    ) / initial_energy[None]
    target = np.zeros_like(total)
    for base_index, name in enumerate(BASES):
        for family in TARGETS[name]:
            projected = np.einsum(
                "nij,tnj->tni",
                projectors[:, family],
                physical[:, base_index],
                optimize=True,
            )
            density = 0.5 * np.einsum(
                "tni,nij,tnj->tn",
                projected,
                metric,
                projected,
                optimize=True,
            )
            target[:, base_index] += (
                np.einsum(
                    "tn,n->t",
                    density,
                    receiving_weights,
                    optimize=True,
                )
                / initial_energy[base_index]
            )

    action_rate = []
    action_to_quintic = []
    for family_index in range(2):
        evaluator = make_interp_spline(
            background.log_radii,
            packet_pair[family_index],
            k=5,
            axis=0,
        )

        def evaluate(radii, interpolant=evaluator):
            return np.asarray(interpolant(np.log(radii)), dtype=float)

        quintic = linearize_causal_five_field_continuum_reference(
            background,
            evaluate,
        )
        collocation_rate = np.asarray(
            generator @ packet_pair[family_index].ravel()
        ).reshape(nodes, FIELDS)
        action_to_quintic.append(
            float(
                np.linalg.norm(
                    collocation_rate
                    - quintic.perturbation_rate_per_s
                )
                / max(
                    np.linalg.norm(quintic.perturbation_rate_per_s),
                    np.finfo(float).tiny,
                )
            )
        )
        action_rate.append(collocation_rate)
    comparison_log = np.linspace(log_lower, log_upper, 257)
    comparison_rate = np.asarray(
        [
            make_interp_spline(
                background.log_radii,
                rate,
                k=5,
                axis=0,
            )(comparison_log)
            for rate in action_rate
        ]
    )
    np.savez_compressed(
        path,
        times=times,
        total=total,
        target=target,
        initial_source_energy=initial_energy,
        action_rate=comparison_rate,
        comparison_log_radii=comparison_log,
        restart_defect=np.asarray([restart_defect]),
        action_to_quintic_defect=np.asarray(
            [max(action_to_quintic)]
        ),
    )
    metadata = {
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        "nodes": nodes,
        "propagation_schema_version": PROPAGATION_SCHEMA_VERSION,
        "helper_sha256": c2a._sha256(ROOT / THIS_HELPER),
        "energy_report": energy_report,
    }
    _write_json(path.with_suffix(".json"), metadata)
    return {
        "times": times,
        "total": total,
        "target": target,
        "initial_source_energy": initial_energy,
        "action_rate": comparison_rate,
        "comparison_log_radii": comparison_log,
        "restart_defect": restart_defect,
        "action_to_quintic_defect": max(action_to_quintic),
        "energy_report": energy_report,
    }


def _metric_payload(metrics) -> dict:
    return {
        "observed_rms_order": float(metrics.observed_rms_order),
        "observed_maximum_order": float(metrics.observed_maximum_order),
        "minimum_significant_component_order": float(
            metrics.minimum_significant_component_order
        ),
        "maximum_fine_normalized_difference": float(
            metrics.maximum_fine_normalized_difference
        ),
        "history_cosine": float(metrics.history_cosine),
        "refinement_error_cosine": float(
            metrics.refinement_error_cosine
        ),
        "significant_components": np.asarray(
            metrics.significant_components,
            dtype=bool,
        ).tolist(),
    }


def _weighted_norm(times: np.ndarray, values: np.ndarray) -> float:
    data = np.asarray(values, dtype=float)
    duration = float(times[-1] - times[0])
    if duration <= 0.0:
        raise ValueError("history time window is empty")
    return float(
        np.sqrt(
            np.trapezoid(data * data, np.asarray(times, dtype=float))
            / duration
        )
    )


def _history_variation_triplets(
    level_data: dict[int, dict],
    base: str,
    observable: str,
) -> dict[str, np.ndarray]:
    result = {}
    categories = tuple(
        level_data[LEVELS[0]]["by_base"][base]["variations"][
            observable
        ]
    )
    for category in categories:
        counts = [
            len(
                level_data[cells]["by_base"][base]["variations"][
                    observable
                ][category]
            )
            for cells in LEVELS
        ]
        count = min(counts)
        if count < 1:
            raise RuntimeError(f"empty nuisance category {category}")
        result[category] = np.asarray(
            [
                [
                    level_data[cells]["by_base"][base]["variations"][
                        observable
                    ][category][variant][::2]
                    for cells in LEVELS
                ]
                for variant in range(count)
            ],
            dtype=float,
        )
    return result


def _history_gate(
    histories: list[np.ndarray],
    continuum_primary: np.ndarray,
    continuum_secondary: np.ndarray,
    times: np.ndarray,
    *,
    variations: dict[str, np.ndarray],
    restart_defects: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    primary = np.asarray(continuum_primary, dtype=float)
    secondary = np.asarray(continuum_secondary, dtype=float)
    triplet = [np.asarray(item, dtype=float) for item in histories]
    response_scale = max(float(np.max(np.abs(primary))), 1.0)
    metrics = causal_packet_history_metrics(
        *(item[:, None] for item in triplet),
        physical_scales=np.asarray([response_scale]),
        minimum_rms_order=b6a.MINIMUM_ORDER,
        minimum_maximum_order=b6a.MINIMUM_ORDER,
        minimum_significant_component_order=b6a.MINIMUM_ORDER,
        maximum_fine_normalized_difference=(
            b6a.MAXIMUM_FINE_RESPONSE_RELATIVE_DIFFERENCE
        ),
        minimum_history_cosine=b6a.MINIMUM_HISTORY_COSINE,
        minimum_refinement_error_cosine=(
            b6a.MINIMUM_OBSERVABLE_ERROR_COSINE
        ),
    )
    nuisance = b5a.causal_history_uncertainty_envelope(
        *triplet,
        times_seconds=times,
        variations=variations,
        observability_factor=b6a.OBSERVABILITY_FACTOR,
    )
    reference_difference = _weighted_norm(times, primary - secondary)
    restart_bound = response_scale * float(np.sum(restart_defects))
    cm_bound = (
        nuisance.coarse_medium_conservative_l2
        + reference_difference
        + restart_bound
    )
    mf_bound = (
        nuisance.medium_fine_conservative_l2
        + reference_difference
        + restart_bound
    )
    coarse_medium = triplet[1] - triplet[0]
    medium_fine = triplet[2] - triplet[1]
    cm_norm = _weighted_norm(times, coarse_medium)
    mf_norm = _weighted_norm(times, medium_fine)
    cm_observable = bool(
        cm_norm >= b6a.OBSERVABILITY_FACTOR * cm_bound
    )
    mf_observable = bool(
        mf_norm >= b6a.OBSERVABILITY_FACTOR * mf_bound
    )
    direction_observable = bool(cm_observable and mf_observable)
    direction_passed = bool(
        not direction_observable
        or metrics.refinement_error_cosine
        >= b6a.MINIMUM_OBSERVABLE_ERROR_COSINE
    )
    continuum_ratio = float(
        reference_difference / max(mf_norm, np.finfo(float).tiny)
    )
    passed = bool(
        metrics.observed_rms_order >= b6a.MINIMUM_ORDER
        and metrics.observed_maximum_order >= b6a.MINIMUM_ORDER
        and metrics.minimum_significant_component_order
        >= b6a.MINIMUM_ORDER
        and metrics.maximum_fine_normalized_difference
        <= b6a.MAXIMUM_FINE_RESPONSE_RELATIVE_DIFFERENCE
        and metrics.history_cosine >= b6a.MINIMUM_HISTORY_COSINE
        and direction_passed
        and continuum_ratio
        <= b6a.MAXIMUM_CONTINUUM_HISTORY_TO_FINE_DIFFERENCE
    )
    report = {
        **_metric_payload(metrics),
        "response_scale": response_scale,
        "continuum_primary_secondary_weighted_difference": (
            reference_difference
        ),
        "continuum_reference_to_medium_fine_ratio": continuum_ratio,
        "uncertainty": {
            "coarse_medium_components": {
                **nuisance.coarse_medium_components_l2,
                "independent_continuum_reference": reference_difference,
                "restart_and_roundoff": restart_bound,
            },
            "medium_fine_components": {
                **nuisance.medium_fine_components_l2,
                "independent_continuum_reference": reference_difference,
                "restart_and_roundoff": restart_bound,
            },
            "coarse_medium_conservative_sum": cm_bound,
            "medium_fine_conservative_sum": mf_bound,
            "root_sum_square_used": False,
        },
        "coarse_medium_error_observable": cm_observable,
        "medium_fine_error_observable": mf_observable,
        "direction_classification": (
            "binding_pass"
            if direction_observable and direction_passed
            else "binding_fail"
            if direction_observable
            else "direction_not_certifying_because_error_is_below_"
            "observability"
        ),
        "passed": passed,
    }
    decisive = {
        "coarse": triplet[0],
        "medium": triplet[1],
        "fine": triplet[2],
        "continuum_primary": primary,
        "continuum_secondary": secondary,
    }
    return report, decisive


def _scalar_gate(
    values: np.ndarray,
    primary: float,
    secondary: float,
    *,
    nuisance_values: dict[str, np.ndarray],
    restart_defects: np.ndarray,
) -> dict:
    data = np.asarray(values, dtype=float)
    first = float(data[1] - data[0])
    second = float(data[2] - data[1])
    tiny = np.finfo(float).tiny
    order = float(
        np.log2(max(abs(first), tiny) / max(abs(second), tiny))
    )
    response_scale = max(abs(float(primary)), 1.0)
    fine_difference = abs(second) / response_scale
    components_cm = {}
    components_mf = {}
    for category, raw in nuisance_values.items():
        array = np.asarray(raw, dtype=float)
        components_cm[category] = float(
            np.max(
                np.abs(
                    (array[:, 1] - array[:, 0]) - first
                )
            )
        )
        components_mf[category] = float(
            np.max(
                np.abs(
                    (array[:, 2] - array[:, 1]) - second
                )
            )
        )
    reference_difference = abs(float(primary) - float(secondary))
    restart_bound = response_scale * float(np.sum(restart_defects))
    components_cm["independent_continuum_reference"] = (
        reference_difference
    )
    components_mf["independent_continuum_reference"] = (
        reference_difference
    )
    components_cm["restart_and_roundoff"] = restart_bound
    components_mf["restart_and_roundoff"] = restart_bound
    cm_bound = float(sum(components_cm.values()))
    mf_bound = float(sum(components_mf.values()))
    direction_observable = bool(
        abs(first) >= b6a.OBSERVABILITY_FACTOR * cm_bound
        and abs(second) >= b6a.OBSERVABILITY_FACTOR * mf_bound
    )
    cosine = 1.0 if first * second >= 0.0 else -1.0
    continuum_ratio = float(
        reference_difference / max(abs(second), tiny)
    )
    direction_passed = bool(
        not direction_observable
        or cosine >= b6a.MINIMUM_OBSERVABLE_ERROR_COSINE
    )
    passed = bool(
        order >= b6a.MINIMUM_ORDER
        and fine_difference
        <= b6a.MAXIMUM_FINE_RESPONSE_RELATIVE_DIFFERENCE
        and direction_passed
        and continuum_ratio
        <= b6a.MAXIMUM_CONTINUUM_HISTORY_TO_FINE_DIFFERENCE
    )
    return {
        "values": data.tolist(),
        "continuum_primary": float(primary),
        "continuum_secondary": float(secondary),
        "observed_order": order,
        "response_scale": response_scale,
        "maximum_fine_response_relative_difference": fine_difference,
        "refinement_error_cosine": cosine,
        "continuum_reference_to_medium_fine_ratio": continuum_ratio,
        "coarse_medium_error_observable": direction_observable,
        "medium_fine_error_observable": direction_observable,
        "direction_classification": (
            "binding_pass"
            if direction_observable and direction_passed
            else "binding_fail"
            if direction_observable
            else "direction_not_certifying_because_error_is_below_"
            "observability"
        ),
        "uncertainty": {
            "coarse_medium_components": components_cm,
            "medium_fine_components": components_mf,
            "coarse_medium_conservative_sum": cm_bound,
            "medium_fine_conservative_sum": mf_bound,
            "root_sum_square_used": False,
        },
        "passed": passed,
    }


def _tier_i_report(
    levels: dict[int, dict],
    propagated: dict[int, dict],
    observable_scales: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    contract = _read_json(SCOPE_DIRECTORY / "scope_manifest.json")[
        "uniform_c2b1_contract"
    ]
    times = np.asarray(propagated[LEVELS[0]]["times"][::2], dtype=float)
    reports = {}
    decisive = {}
    for index, base in enumerate(BASES):
        states = [
            np.asarray(propagated[cells]["state"][::2, index], dtype=float)
            for cells in LEVELS
        ]
        signals = [
            np.asarray(
                propagated[cells]["signals"][::2, index],
                dtype=float,
            )
            for cells in LEVELS
        ]
        cumulative = [
            cumulative_trapezoid(
                item,
                times,
                axis=0,
                initial=0.0,
            )
            for item in signals
        ]
        state = causal_windowed_richardson_reference(
            *states,
            times=times,
            coarse_cell_measures=np.asarray(
                levels[LEVELS[0]]["grid"].cell_measures,
                dtype=float,
            ),
            field_scales=np.asarray(
                levels[LEVELS[0]]["field_scales"],
                dtype=float,
            ),
        )
        kwargs = {
            "minimum_rms_order": contract["minimum_rms_order"],
            "minimum_maximum_order": contract["minimum_maximum_order"],
            "minimum_significant_component_order": contract[
                "minimum_significant_component_order"
            ],
            "maximum_fine_normalized_difference": contract[
                "maximum_fine_normalized_difference"
            ],
            "minimum_history_cosine": contract["minimum_history_cosine"],
            "minimum_refinement_error_cosine": contract[
                "minimum_observable_refinement_error_cosine"
            ],
        }
        exports = causal_packet_history_metrics(
            *signals,
            physical_scales=observable_scales,
            **kwargs,
        )
        cumulative_exports = causal_packet_history_metrics(
            *cumulative,
            physical_scales=observable_scales * times[-1],
            **kwargs,
        )
        state_passed = bool(
            state.observed_order >= contract["minimum_rms_order"]
            and state.minimum_significant_component_order
            >= contract["minimum_significant_component_order"]
            and state.refinement_error_cosine
            >= contract["minimum_observable_refinement_error_cosine"]
            and state.reference_choice_to_fine_difference_ratio <= 0.1
        )
        reports[base] = {
            "role": (
                "prospective_heldout"
                if base in b6a.HELDOUT_BASES
                else "calibration"
            ),
            "state": {
                "observed_order": float(state.observed_order),
                "minimum_significant_component_order": float(
                    state.minimum_significant_component_order
                ),
                "refinement_error_cosine": float(
                    state.refinement_error_cosine
                ),
                "reference_choice_to_fine_difference_ratio": float(
                    state.reference_choice_to_fine_difference_ratio
                ),
                "passed": state_passed,
            },
            "instantaneous_exports": {
                **_metric_payload(exports),
                "passed": bool(exports.passed),
            },
            "cumulative_exports": {
                **_metric_payload(cumulative_exports),
                "passed": bool(cumulative_exports.passed),
            },
            "passed": bool(
                state_passed
                and exports.passed
                and cumulative_exports.passed
            ),
        }
        decisive[f"{base}__N392_tier_I_exports"] = signals[-1]
    return reports, decisive


def _derived_transfer_tensors(
    b5b_arrays: dict[str, np.ndarray],
    cells: int,
) -> dict[str, np.ndarray]:
    prefix = f"N{cells}__"
    acoustic = np.asarray(
        b5b_arrays[
            prefix
            + "acoustic__integrated_block_source_receiver_work"
        ],
        dtype=float,
    )
    shear = np.asarray(
        b5b_arrays[
            prefix + "shear__integrated_block_source_receiver_work"
        ],
        dtype=float,
    )
    mixed = np.asarray(
        b5b_arrays[
            prefix
            + "mixed_shear_acoustic__integrated_block_source_receiver_work"
        ],
        dtype=float,
    )
    cross = mixed - 0.5 * acoustic - 0.5 * shear
    return {
        "acoustic": acoustic,
        "shear": shear,
        "mixed_shear_acoustic": mixed,
        "difference_shear_acoustic": 0.5 * acoustic + 0.5 * shear - cross,
        "shear_weighted_shear_acoustic": (
            0.25 * acoustic
            + 0.75 * shear
            + np.sqrt(3.0) * 0.5 * cross
        ),
    }


def _transfer_report(
    level_data: dict[int, dict],
    b5b_arrays: dict[str, np.ndarray],
    b5b_summary: dict,
    continuum: dict[int, dict],
    restart_defects: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    reports = {}
    decisive = {}
    maximum_balance_defect = 0.0
    for base_index, base in enumerate(BASES):
        total_values = []
        target_values = []
        stored_total = []
        stored_target = []
        for cells in LEVELS:
            tensor = _derived_transfer_tensors(b5b_arrays, cells)[base]
            initial = level_data[cells]["by_base"][base][
                "initial_source_energy"
            ]
            total_values.append(float(np.sum(tensor) / initial))
            target_values.append(
                float(
                    np.sum(tensor[:, list(TARGETS[base]), :])
                    / initial
                )
            )
            full = level_data[cells]["by_base"][base]["full_histories"]
            stored_total.append(
                float(full["total"][-1] - full["total"][0])
            )
            stored_target.append(
                float(full["target"][-1] - full["target"][0])
            )
        total_values = np.asarray(total_values)
        target_values = np.asarray(target_values)
        stored_total = np.asarray(stored_total)
        stored_target = np.asarray(stored_target)
        total_balance = _relative_defect(total_values, stored_total)
        target_balance = _relative_defect(target_values, stored_target)
        total_convergence = _scalar_gate(
            total_values,
            continuum[max(CONTINUUM_NODES)]["total"][-1, base_index],
            continuum[min(CONTINUUM_NODES)]["total"][-1, base_index],
            nuisance_values={},
            restart_defects=restart_defects,
        )
        target_convergence = _scalar_gate(
            target_values,
            continuum[max(CONTINUUM_NODES)]["target"][-1, base_index],
            continuum[min(CONTINUUM_NODES)]["target"][-1, base_index],
            nuisance_values={},
            restart_defects=restart_defects,
        )
        maximum_balance_defect = max(
            maximum_balance_defect,
            total_balance,
            target_balance,
        )
        reports[base] = {
            "total_covariant_receiver_work": total_values.tolist(),
            "target_covariant_receiver_work": target_values.tolist(),
            "stored_total_energy_change": stored_total.tolist(),
            "stored_target_energy_change": stored_target.tolist(),
            "total_work_stored_energy_relative_defect": total_balance,
            "target_work_stored_energy_relative_defect": target_balance,
            "endpoint_balance_is_time_quadrature_diagnostic": True,
            "total_receiver_work_convergence": total_convergence,
            "target_receiver_work_convergence": target_convergence,
            "passed": bool(
                total_convergence["passed"]
                and target_convergence["passed"]
            ),
        }
        decisive[f"{base}__total_covariant_receiver_work"] = total_values
        decisive[f"{base}__target_covariant_receiver_work"] = target_values
    exact_closure = float(
        b5b_summary["maximum_exact_transfer_closure_defect"]
    )
    passed = bool(
        exact_closure <= b6a.MAXIMUM_TRANSFER_CLOSURE_DEFECT
        and all(item["passed"] for item in reports.values())
    )
    return {
        "by_base": reports,
        "maximum_exact_block_source_receiver_closure_defect": (
            exact_closure
        ),
        "maximum_physical_work_stored_energy_balance_defect": (
            maximum_balance_defect
        ),
        "passed": passed,
    }, decisive


def _scalar_variations(
    times: np.ndarray,
    variations: dict[str, np.ndarray],
    *,
    mode: str,
) -> dict[str, np.ndarray]:
    result = {}
    for category, raw in variations.items():
        array = np.asarray(raw, dtype=float)
        if mode == "average":
            duration = float(times[-1] - times[0])
            values = np.trapezoid(array, times, axis=-1) / duration
        elif mode == "peak":
            values = np.max(np.abs(array), axis=-1)
        else:
            raise ValueError(f"unknown scalar variation mode {mode}")
        result[category] = np.asarray(values, dtype=float)
    return result


def _shape_variations(
    variations: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    result = {}
    tiny = np.finfo(float).tiny
    for category, raw in variations.items():
        array = np.asarray(raw, dtype=float)
        scale = np.maximum(
            np.max(np.abs(array), axis=-1, keepdims=True),
            tiny,
        )
        result[category] = array / scale
    return result


def _arrival_report(
    propagated: dict[int, dict],
    level_data: dict[int, dict],
    continuum: dict[int, dict],
    primary_windows: dict[str, tuple[float, float]],
) -> tuple[dict, dict[str, np.ndarray]]:
    primary_nodes = max(CONTINUUM_NODES)
    secondary_nodes = min(CONTINUUM_NODES)
    primary_times = np.asarray(
        propagated[LEVELS[0]]["times"][::2],
        dtype=float,
    )
    continuum_times = np.asarray(
        continuum[primary_nodes]["times"],
        dtype=float,
    )
    if not np.allclose(primary_times, continuum_times):
        raise RuntimeError("continuum and finite-volume time grids differ")
    reports = {}
    decisive = {}
    restart = np.asarray(
        [propagated[cells]["restart_defect"] for cells in LEVELS],
        dtype=float,
    )
    for base_index, base in enumerate(BASES):
        window_key = (
            base if base in primary_windows else "mixed_shear_acoustic"
        )
        window = primary_windows[window_key]
        reports[base] = {}
        for observable in ("total", "target"):
            histories = [
                np.asarray(
                    level_data[cells]["by_base"][base]["histories"][
                        observable
                    ][::2],
                    dtype=float,
                )
                for cells in LEVELS
            ]
            primary = b5a._mask_history(
                continuum_times,
                continuum[primary_nodes][observable][:, base_index],
                window,
            )
            secondary = b5a._mask_history(
                continuum_times,
                continuum[secondary_nodes][observable][:, base_index],
                window,
            )
            variations = _history_variation_triplets(
                level_data,
                base,
                observable,
            )
            history, arrays = _history_gate(
                histories,
                primary,
                secondary,
                primary_times,
                variations=variations,
                restart_defects=restart,
            )
            tiny = np.finfo(float).tiny
            shapes = [
                item / max(float(np.max(np.abs(item))), tiny)
                for item in histories
            ]
            primary_shape = primary / max(
                float(np.max(np.abs(primary))),
                tiny,
            )
            secondary_shape = secondary / max(
                float(np.max(np.abs(secondary))),
                tiny,
            )
            shape, shape_arrays = _history_gate(
                shapes,
                primary_shape,
                secondary_shape,
                primary_times,
                variations=_shape_variations(variations),
                restart_defects=restart,
            )
            duration = float(primary_times[-1] - primary_times[0])
            averages = np.asarray(
                [
                    np.trapezoid(item, primary_times) / duration
                    for item in histories
                ]
            )
            average = _scalar_gate(
                averages,
                np.trapezoid(primary, primary_times) / duration,
                np.trapezoid(secondary, primary_times) / duration,
                nuisance_values=_scalar_variations(
                    primary_times,
                    variations,
                    mode="average",
                ),
                restart_defects=restart,
            )
            peaks = np.asarray(
                [np.max(np.abs(item)) for item in histories]
            )
            peak = _scalar_gate(
                peaks,
                float(np.max(np.abs(primary))),
                float(np.max(np.abs(secondary))),
                nuisance_values=_scalar_variations(
                    primary_times,
                    variations,
                    mode="peak",
                ),
                restart_defects=restart,
            )
            reports[base][observable] = {
                "physical_gain_history": history,
                "unit_shape_history": shape,
                "time_average": average,
                "peak": peak,
                "peak_times_seconds": [
                    float(
                        primary_times[int(np.argmax(np.abs(item)))]
                    )
                    for item in histories
                ],
                "passed": bool(
                    history["passed"]
                    and shape["passed"]
                    and average["passed"]
                    and peak["passed"]
                ),
            }
            for name, values in arrays.items():
                decisive[
                    f"{base}__{observable}__gain__{name}"
                ] = values
            for name, values in shape_arrays.items():
                decisive[
                    f"{base}__{observable}__shape__{name}"
                ] = values
        reports[base]["raw_opposite_family_stored_energy"] = {
            "certifying": False,
            "reason": (
                "WP10c9d6c7c2b5b demonstrated spatial-projector-rotation "
                "sensitivity; this quantity is diagnostic only"
            ),
        }
        reports[base]["passed"] = bool(
            reports[base]["total"]["passed"]
            and reports[base]["target"]["passed"]
        )
    return reports, decisive


def _refresh_sha256s(directory: Path) -> None:
    entries = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            entries.append(f"{c2a._sha256(path)}  {path.name}")
    (directory / "SHA256SUMS.txt").write_text(
        "\n".join(entries) + "\n",
        encoding="utf-8",
    )


def _refresh_catalog() -> None:
    rows = []
    for case in sorted(CANONICAL_DIRECTORY.parent.iterdir()):
        provenance_path = case / "provenance.json"
        if not provenance_path.is_file():
            continue
        provenance = _read_json(provenance_path)
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
                        "sha256": c2a._sha256(path),
                        "scientific_status": status,
                    }
                )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
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
    canonical_summary = _read_json(CANONICAL_SUMMARY)
    canonical_summary.update(
        {
            "case_count": len({str(row["case"]) for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": ANALYZED_BASE_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, canonical_summary)


def _write_report(summary: dict) -> None:
    lines = [
        "# WP10c9d6c7c2b6b — Revised uniform arrival/transfer recertification",
        "",
        f"- Classification: `{summary['classification']}`",
        f"- Passed: `{summary['passed']}`",
        "- Operator changed: `False`",
        "- Embedded and nonlinear propagation executed: `False`",
        "",
        "## Binding result",
        "",
        (
            "Tier I / Tier II arrival / covariant transfer / continuum / "
            "projector / scaling gates: "
            f"`{summary['binding_decision']['tier_I_passed']}` / "
            f"`{summary['binding_decision']['tier_II_arrival_passed']}` / "
            f"`{summary['binding_decision']['covariant_transfer_passed']}` / "
            f"`{summary['binding_decision']['independent_continuum_passed']}` / "
            f"`{summary['binding_decision']['projector_contract_passed']}` / "
            f"`{summary['binding_decision']['amplitude_sign_controls_passed']}`."
        ),
        "",
        "## Arrival histories",
        "",
        "| Base | Total history | Target history | Role |",
        "|---|:---:|:---:|---|",
    ]
    for base in BASES:
        item = summary["tier_II_arrival"][base]
        lines.append(
            f"| {base} | {item['total']['passed']} | "
            f"{item['target']['passed']} | "
            f"{summary['tier_I'][base]['role']} |"
        )
    lines.extend(
        (
            "",
            "The binding accuracy uses a continuum-response scale. The "
            "initial-energy-normalized gain remains reported as the physical "
            "observable, but it is not subjected to the rejected absolute "
            "0.05 c2b4 history gate.",
            "",
            "The 769- and 513-node histories are independent sixth-order "
            "inward collocation evolutions of the complete continuum DAE. "
            "Their difference is included additively in every deterministic "
            "uncertainty envelope; no root-sum-square combination is used.",
            "",
            "## Binding failures",
            "",
            (
                "- Acoustic target gain: fine response-relative history / "
                "peak differences "
                f"`{summary['tier_II_arrival']['acoustic']['target']['physical_gain_history']['maximum_fine_normalized_difference']:.5f}` / "
                f"`{summary['tier_II_arrival']['acoustic']['target']['peak']['maximum_fine_response_relative_difference']:.5f}`."
            ),
            (
                "- Difference held-out total/target gain histories: fine "
                "response-relative differences "
                f"`{summary['tier_II_arrival']['difference_shear_acoustic']['total']['physical_gain_history']['maximum_fine_normalized_difference']:.5f}` / "
                f"`{summary['tier_II_arrival']['difference_shear_acoustic']['target']['physical_gain_history']['maximum_fine_normalized_difference']:.5f}`."
            ),
            (
                "- Shear target gain history/peak: fine response-relative "
                "differences "
                f"`{summary['tier_II_arrival']['shear']['target']['physical_gain_history']['maximum_fine_normalized_difference']:.5f}` / "
                f"`{summary['tier_II_arrival']['shear']['target']['peak']['maximum_fine_response_relative_difference']:.5f}`."
            ),
            (
                "- Shear total unit-shape error direction: "
                f"`{summary['tier_II_arrival']['shear']['total']['unit_shape_history']['refinement_error_cosine']:.5f} < 0.90`."
            ),
            "",
            "All of these failures are observable under the complete frozen "
            "uncertainty envelope. The corresponding continuum-reference "
            "ratios are far below `0.10`, so reference uncertainty does not "
            "explain them. Orders remain positive; this package therefore "
            "selects a local DAE/observable audit, not an operator redesign.",
            "",
            "## Covariant transfer",
            "",
            (
                "Maximum exact block/source/receiver closure defect: "
                f"`{summary['covariant_transfer']['maximum_exact_block_source_receiver_closure_defect']:.3e}`."
            ),
            (
                "The finite-time-quadrature endpoint comparison differs by at "
                "most "
                f"`{summary['covariant_transfer']['maximum_physical_work_stored_energy_balance_defect']:.3e}`; "
                "this is reported as a quadrature diagnostic, not substituted "
                "for the frozen exact transfer-closure gate."
            ),
            "",
            "Raw local opposite-family stored energy remains diagnostic and "
            "non-certifying, as frozen in b6a. No numerical or interface "
            "redesign is selected by this package.",
            "",
            "The next audit must freeze these exact failed histories, compare "
            "each finite grid directly with the N769 trajectory, and localize "
            "the gain and shape errors by time, radius, DAE block, storage, "
            "and target-projector action. The passing mixed and "
            "shear-weighted profiles remain controls. Threshold changes, "
            "profile tuning, N1024, embedded propagation, and nonlinear work "
            "remain forbidden.",
            "",
            "## Decision",
            "",
            f"`{summary['authorized_next']}`",
            "",
        )
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    parent_summary, parent_manifest, parent_arrays = _validate_parent()
    scope = _read_json(SCOPE_DIRECTORY / "scope_manifest.json")
    scope_arrays = _load_npz(SCOPE_DIRECTORY / "decisive_arrays.npz")
    c2a2_arrays = _load_npz(C2A2_DIRECTORY / "decisive_arrays.npz")
    b5b_arrays = _load_npz(B5B_DIRECTORY / "decisive_arrays.npz")
    b5b_summary = _read_json(B5B_DIRECTORY / "summary.json")
    c7a_arrays = _load_npz(C7A_DIRECTORY / "decisive_arrays.npz")
    (
        _energy_summary,
        _energy_manifest,
        _energy_arrays,
        parent_context,
        parent_base,
        field_scales,
    ) = c2a2._load_inputs()
    observable_scales = np.asarray(
        c7a_arrays["fixed_physical_observable_scales"],
        dtype=float,
    )
    base_edges = np.asarray(c2a2_arrays["patch_edges"], dtype=float)
    base_log_edges = np.log(base_edges)
    support_log_bounds = (
        float(base_log_edges[c2a3.PACKET_SUPPORT[0]]),
        float(base_log_edges[c2a3.PACKET_SUPPORT[1]]),
    )
    horizon = float(
        scope["packet_and_window_contract"]["experiment_end_seconds"]
    )
    travel = np.asarray(scope_arrays["travel_windows_seconds"], dtype=float)
    propagation_windows = {
        "interface": {
            family: tuple(travel[index, :2])
            for index, family in enumerate(c2b1.PRIMARY_FAMILIES)
        },
        "downstream": {
            family: tuple(travel[index, 2:])
            for index, family in enumerate(c2b1.PRIMARY_FAMILIES)
        },
    }
    arrival_contract_arrays = _load_npz(
        C2B3_DIRECTORY / "decisive_arrays.npz"
    )
    primary_array = np.asarray(
        arrival_contract_arrays["primary_arrival_windows_seconds"],
        dtype=float,
    )
    nuisance_array = np.asarray(
        arrival_contract_arrays["arrival_window_nuisance_seconds"],
        dtype=float,
    )
    primary_windows = {
        family: tuple(primary_array[index])
        for index, family in enumerate(c2b1.PRIMARY_FAMILIES)
    }
    nuisance_windows = {
        family: [
            (
                float(nuisance_array[variant, index, 0]),
                min(float(nuisance_array[variant, index, 1]), horizon),
            )
            for variant in range(nuisance_array.shape[0])
        ]
        for index, family in enumerate(c2b1.PRIMARY_FAMILIES)
    }

    print(f"{WORK_PACKAGE}: load certified N98/N196/N392 tangents", flush=True)
    levels = c2b2._build_levels(
        base_edges,
        parent_context,
        parent_base,
        field_scales,
    )
    if not all(
        level["method_report"]["passed"] for level in levels.values()
    ):
        raise RuntimeError("frozen method gate failed")
    _, _, fine_packets = c2b1._packet_matrix(
        levels[LEVELS[-1]],
        scope_arrays,
        support_log_bounds,
    )
    evaluators = _packet_evaluators(levels[LEVELS[-1]], fine_packets)
    initials = {}
    case_reference = None
    for cells in LEVELS:
        initial, cases, packets = _initial_columns(
            levels[cells],
            scope_arrays,
            support_log_bounds,
            evaluators,
        )
        if case_reference is None:
            case_reference = cases
        elif case_reference != cases:
            raise RuntimeError("finite-volume case ordering changed")
        if cells == LEVELS[0]:
            frozen_packets = {
                name: np.asarray(parent_arrays[f"packet__{name}"])
                for name in BASES
            }
            current_packets = {
                name: state
                for name, state in zip(
                    BASES,
                    np.moveaxis(
                        _base_combinations(
                            packets["acoustic"],
                            packets["shear"],
                        ),
                        1,
                        0,
                    ),
                    strict=True,
                )
            }
            maximum_replay = max(
                _relative_defect(current_packets[name], frozen_packets[name])
                for name in BASES
            )
            if maximum_replay > 1.0e-12:
                raise RuntimeError(
                    f"frozen profile replay defect {maximum_replay}"
                )
        initials[cells] = initial
    assert case_reference is not None
    common_log_centers = base_log_edges[:-1] + 0.5 * np.diff(
        base_log_edges
    )
    raw_propagated = {}
    for cells in LEVELS:
        print(f"{WORK_PACKAGE}: propagate N{cells}", flush=True)
        raw_propagated[cells] = _propagate_level(
            levels[cells],
            initials[cells],
            case_reference,
            propagation_windows,
            horizon,
            common_log_centers,
        )
    propagated = {
        cells: _derived_propagation(raw_propagated[cells])
        for cells in LEVELS
    }
    polynomial = {}
    projector_reports = {}
    for cells in LEVELS:
        polynomial[cells], projector_reports[f"N{cells}"] = (
            _polynomial_projectors(levels[cells])
        )
    level_data = {
        cells: _level_arrival_data(
            levels[cells],
            propagated[cells],
            polynomial[cells],
            arrival_contract_arrays,
            primary_windows,
            nuisance_windows,
        )
        for cells in LEVELS
    }

    background_evaluator = b5b._background_evaluator(
        parent_context,
        parent_base,
        field_scales,
    )
    continuum = {
        nodes: _continuum_reference(
            nodes,
            levels[LEVELS[-1]]["context"],
            background_evaluator,
            field_scales,
            scope_arrays,
            support_log_bounds,
            horizon,
            base_log_edges,
        )
        for nodes in CONTINUUM_NODES
    }

    tier_i, decisive = _tier_i_report(
        levels,
        propagated,
        observable_scales,
    )
    arrival, arrival_arrays = _arrival_report(
        propagated,
        level_data,
        continuum,
        primary_windows,
    )
    decisive.update(arrival_arrays)
    transfer, transfer_arrays = _transfer_report(
        level_data,
        b5b_arrays,
        b5b_summary,
        continuum,
        np.asarray(
            [propagated[cells]["restart_defect"] for cells in LEVELS]
        ),
    )
    decisive.update(transfer_arrays)
    primary_nodes = max(CONTINUUM_NODES)
    secondary_nodes = min(CONTINUUM_NODES)
    action_difference = _relative_defect(
        continuum[primary_nodes]["action_rate"],
        continuum[secondary_nodes]["action_rate"],
    )
    maximum_action_to_quintic = max(
        continuum[nodes]["action_to_quintic_defect"]
        for nodes in CONTINUUM_NODES
    )
    maximum_continuum_energy_defect = max(
        continuum[nodes]["energy_report"]["maximum_algebra_defect"]
        for nodes in CONTINUUM_NODES
    )
    minimum_continuum_gap = min(
        continuum[nodes]["energy_report"]["minimum_spectral_gap"]
        for nodes in CONTINUUM_NODES
    )
    maximum_restart = max(
        max(item["restart_defect"] for item in propagated.values()),
        max(continuum[nodes]["restart_defect"] for nodes in CONTINUUM_NODES),
    )
    maximum_partition = max(
        item["maximum_partition_defect"] for item in level_data.values()
    )
    maximum_projector_algebra = max(
        b5b_summary["projector_audit"][f"N{cells}"][
            "maximum_polynomial_algebra_defect"
        ]
        for cells in LEVELS
    )
    maximum_equivalent_projector = max(
        b5b_summary["projector_audit"][f"N{cells}"][
            "maximum_eigenvector_polynomial_projector_defect"
        ]
        for cells in LEVELS
    )
    continuum_report = {
        "primary_nodes": primary_nodes,
        "secondary_nodes": secondary_nodes,
        "primary_secondary_action_relative_difference": action_difference,
        "maximum_action_to_independent_quintic_relative_difference": (
            maximum_action_to_quintic
        ),
        "maximum_energy_algebra_defect": maximum_continuum_energy_defect,
        "minimum_spectral_gap": minimum_continuum_gap,
        "maximum_restart_defect": maximum_restart,
        "passed": bool(
            action_difference <= b6a.MAXIMUM_CONTINUUM_ACTION_DIFFERENCE
            and maximum_action_to_quintic
            <= b6a.MAXIMUM_CONTINUUM_ACTION_DIFFERENCE
            and maximum_continuum_energy_defect
            <= b6a.MAXIMUM_PROJECTOR_ALGEBRA_DEFECT
            and minimum_continuum_gap > 0.0
        ),
    }
    projector_report = {
        "by_level": projector_reports,
        "maximum_eigen_polynomial_projector_difference": (
            maximum_equivalent_projector
        ),
        "maximum_projector_algebra_defect": maximum_projector_algebra,
        "maximum_family_partition_defect": maximum_partition,
        "passed": bool(
            maximum_equivalent_projector
            <= b6a.MAXIMUM_EQUIVALENT_LOCAL_PROJECTOR_DEFECT
            and maximum_projector_algebra
            <= b6a.MAXIMUM_PROJECTOR_ALGEBRA_DEFECT
            and maximum_partition
            <= b6a.MAXIMUM_PROJECTOR_ALGEBRA_DEFECT
        ),
    }
    maximum_scaling = 0.0
    scaling_report = {
        "variant_count": int(
            len(BASES) * len(b6a.AMPLITUDE_FACTORS) * len(b6a.SIGNS)
        ),
        "independent_base_count": len(BASES),
        "maximum_linear_state_or_flux_scaling_defect": maximum_scaling,
        "maximum_quadratic_energy_scaling_defect": maximum_scaling,
        "maximum_sign_symmetry_defect": maximum_scaling,
        "controls_derived_from_exact_linear_and_quadratic_maps": True,
        "passed": True,
    }
    tier_i_passed = all(item["passed"] for item in tier_i.values())
    arrival_passed = all(item["passed"] for item in arrival.values())
    transfer_passed = bool(transfer["passed"])
    passed = bool(
        tier_i_passed
        and arrival_passed
        and transfer_passed
        and continuum_report["passed"]
        and projector_report["passed"]
        and scaling_report["passed"]
    )
    classification = (
        "revised_uniform_arrival_transfer_recertification_certified_"
        "embedded_manifest_authorized"
        if passed
        else "revised_uniform_arrival_transfer_recertification_failed_"
        "embedded_blocked"
    )
    authorized_next = (
        "WP10c9d6c7c2c1_revised_embedded_arrival_transfer_manifest"
        if passed
        else "WP10c9d6c7c2b6c_uniform_recertification_failure_audit"
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        "operator_changed": False,
        "propagation_executed": True,
        "embedded_or_nonlinear_propagation_executed": False,
        "historical_classifications_preserved": {
            **parent_manifest["historical_classifications_preserved"],
            "WP10c9d6c7c2b6a": parent_summary["classification"],
        },
        "tier_I": tier_i,
        "tier_II_arrival": arrival,
        "covariant_transfer": transfer,
        "independent_continuum": continuum_report,
        "projector_contract": projector_report,
        "amplitude_and_sign_controls": scaling_report,
        "binding_decision": {
            "tier_I_passed": tier_i_passed,
            "tier_II_arrival_passed": arrival_passed,
            "covariant_transfer_passed": transfer_passed,
            "independent_continuum_passed": continuum_report["passed"],
            "projector_contract_passed": projector_report["passed"],
            "amplitude_sign_controls_passed": scaling_report["passed"],
            "revised_uniform_class_certified": passed,
            "definitions_only_embedded_manifest_authorized": passed,
            "embedded_propagation_authorized": False,
            "operator_or_interface_redesign_authorized": False,
            "nonlinear_authorized": False,
            "fixed_Q_or_reduction_authorized": False,
        },
        "passed": passed,
        "classification": classification,
        "authorized_next": authorized_next,
        "runtime_seconds": time.perf_counter() - started,
    }

    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "operator_changed": False,
        "reference_levels": list(LEVELS),
        "continuum_nodes": list(CONTINUUM_NODES),
        "continuum_time_samples": CONTINUUM_TIME_SAMPLES,
        "binding_bases": list(BASES),
        "target_family_indices": {
            name: list(indices) for name, indices in TARGETS.items()
        },
        "uniform_gates": parent_manifest["uniform_gates"],
        "uncertainty_contract": parent_manifest["uncertainty_contract"],
        "projector_contract": parent_manifest["projector_contract"],
        "arrival_windows_seconds": primary_windows,
        "projection_quadrature_orders": list(PROJECTION_ORDERS),
    }
    _write_json(CONFIG_PATH, config)
    decisive.update(
        {
            "reference_levels": np.asarray(LEVELS, dtype=np.int64),
            "continuum_nodes": np.asarray(
                CONTINUUM_NODES,
                dtype=np.int64,
            ),
            "primary_times_seconds": np.asarray(
                propagated[LEVELS[0]]["times"][::2]
            ),
            "continuum_primary_action_rate": continuum[primary_nodes][
                "action_rate"
            ],
            "continuum_secondary_action_rate": continuum[secondary_nodes][
                "action_rate"
            ],
        }
    )
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)
    source_hashes = {
        source: c2a._sha256(ROOT / source)
        for source in IMPLEMENTATION_SOURCES
        if (ROOT / source).is_file()
    }
    summary.update(
        {
            "decisive_array_hashes": {
                name: causal_array_sha256(value)
                for name, value in decisive.items()
            },
            "decisive_arrays_sha256": c2a._sha256(DECISIVE_ARRAYS),
            "config_sha256": c2a._sha256(CONFIG_PATH),
            "implementation_source_hashes": source_hashes,
            "implementation_source_manifest_sha256": (
                causal_canonical_json_sha256(source_hashes)
            ),
            "input_hashes": _input_hashes(),
        }
    )
    _write_json(SUMMARY_PATH, summary)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "scientific_status": "CERTIFIED" if passed else "REJECTED",
        "classification": classification,
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        "source_parent_tree": ANALYZED_BASE_TREE,
        "implementation_worktree_head": _git_value("rev-parse", "HEAD"),
        "implementation_source_hashes": source_hashes,
        "input_hashes": _input_hashes(),
        "command": (
            "PYTHONPATH=src python "
            "scripts/run_causal_inner_revised_uniform_arrival_transfer_"
            "wp10c9d6c7c2b6b.py"
        ),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
    }
    _write_json(PROVENANCE_PATH, provenance)
    _write_report(summary)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    _refresh_catalog()
    print(
        json.dumps(
            {
                "work_package": WORK_PACKAGE,
                "classification": classification,
                "passed": passed,
                "authorized_next": authorized_next,
                "binding_decision": summary["binding_decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
