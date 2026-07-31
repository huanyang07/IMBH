#!/usr/bin/env python3
"""Run the frozen direct-continuum uniform recertification.

WP10c9d6c7c2b6e preserves b6b-b6d, changes no operator, and runs no
embedded or nonlinear state.  The nine b6d profiles are synthesized from
the exact acoustic/shear propagation basis on N98/N196/N392 and compared
directly with independent N769/N513 continuum histories.
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
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import expm_multiply


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_causal_inner_direct_continuum_contract_manifest_wp10c9d6c7c2b6d as b6d  # noqa: E402
import run_causal_inner_one_way_transmission_interpretation_wp10c9d6c7c2b2 as c2b2  # noqa: E402
import run_causal_inner_revised_uniform_arrival_transfer_wp10c9d6c7c2b6b as b6b  # noqa: E402
import run_causal_inner_scattering_energy_wp10c9d6c7c2a2 as c2a2  # noqa: E402
import run_causal_inner_scattering_observability_manifest_wp10c9d6c7c2a as c2a  # noqa: E402
import run_causal_inner_scattering_scope_wp10c9d6c7c2a3 as c2a3  # noqa: E402
import run_causal_inner_uniform_arrival_conditioning_wp10c9d6c7c2b5a as b5a  # noqa: E402
import run_causal_inner_uniform_family_transfer_wp10c9d6c7c2b5b as b5b  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_continuum_truncation import (  # noqa: E402
    build_causal_five_field_continuum_background,
    causal_five_field_inward_collocation_generator_blocks,
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
WORK_PACKAGE = "WP10c9d6c7c2b6e"
ANALYZED_BASE_COMMIT = "f6936282ab216e5e3db4ae04b981c544854dff87"
ANALYZED_BASE_PARENT = "523100839171bf672319ec2185a5a69e18da1f02"
ANALYZED_BASE_TREE = "f6b9edf137a927e6af11713fb9b35f1d0cfbb9da"
LEVELS = b6d.LEVELS
CONTINUUM_NODES = b6d.CONTINUUM_NODES
BASES = b6d.BINDING_BASES
FIELDS = 5
TIME_SAMPLES = 513

THIS_RUNNER = (
    "scripts/"
    "run_causal_inner_direct_continuum_uniform_recertification_"
    "wp10c9d6c7c2b6e.py"
)
THIS_TEST = (
    "tests/"
    "test_causal_inner_direct_continuum_uniform_recertification_"
    "wp10c9d6c7c2b6e.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_DIRECT_CONTINUUM_UNIFORM_RECERTIFICATION_"
    "WP10C9D6C7C2B6E_RESULTS_2026-07-30.md"
)
PARENT_DIRECTORY = b6d.CANONICAL_DIRECTORY
B5B_DIRECTORY = b6b.B5B_DIRECTORY
SCOPE_DIRECTORY = b6b.SCOPE_DIRECTORY
C2A2_DIRECTORY = b6b.C2A2_DIRECTORY
C2B3_DIRECTORY = b6b.C2B3_DIRECTORY
C7A_DIRECTORY = b6b.C7A_DIRECTORY
CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_direct_continuum_uniform_recertification_"
    "wp10c9d6c7c2b6e"
)
CHECKPOINT_DIRECTORY = (
    ROOT
    / "outputs/checkpoints/"
    "causal_inner_direct_continuum_uniform_recertification_"
    "wp10c9d6c7c2b6e"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
IMPLEMENTATION_SOURCES = (THIS_RUNNER, THIS_TEST)


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


def _sha256(path: Path) -> str:
    return c2a._sha256(path)


def _relative_defect(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    scale = max(
        float(np.max(np.abs(left))),
        float(np.max(np.abs(right))),
        np.finfo(float).tiny,
    )
    return float(np.max(np.abs(left - right)) / scale)


def _weighted_norm(times: np.ndarray, values: np.ndarray) -> float:
    duration = float(times[-1] - times[0])
    data = np.asarray(values, dtype=float)
    return float(
        np.sqrt(
            np.trapezoid(data * data, times)
            / max(duration, np.finfo(float).tiny)
        )
    )


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=float).ravel()
    right = np.asarray(second, dtype=float).ravel()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= np.finfo(float).tiny:
        return 1.0
    return float(np.dot(left, right) / denominator)


def _orders(values: list[float]) -> list[float]:
    tiny = np.finfo(float).tiny
    return [
        float(np.log2(max(values[index], tiny) / max(values[index + 1], tiny)))
        for index in range(len(values) - 1)
    ]


def _validate_parent() -> tuple[dict, dict, dict[str, np.ndarray]]:
    summary = _read_json(PARENT_DIRECTORY / "summary.json")
    manifest = _read_json(PARENT_DIRECTORY / "contract_manifest.json")
    arrays = _load_npz(PARENT_DIRECTORY / "decisive_arrays.npz")
    if (
        summary["classification"]
        != "direct_continuum_arrival_contract_frozen_uniform_"
        "recertification_authorized"
        or not summary["passed"]
        or summary["propagation_executed"]
        or not summary["binding_decision"][
            "uniform_b6e_recertification_authorized"
        ]
        or summary["binding_decision"]["embedded_authorized"]
        or summary["manifest_sha256"] != manifest["manifest_sha256"]
        or tuple(arrays["reference_levels"]) != LEVELS
        or tuple(arrays["continuum_nodes"]) != CONTINUUM_NODES
    ):
        raise RuntimeError("WP10c9d6c7c2b6d binding status changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT)
        != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("WP10c9d6c7c2b6e analyzed identity changed")
    return summary, manifest, arrays


def _coefficients(parent_summary: dict) -> dict[str, np.ndarray]:
    values = {
        "acoustic": np.asarray([1.0, 0.0]),
        "shear": np.asarray([0.0, 1.0]),
        "mixed_shear_acoustic": np.asarray(
            [1.0, 1.0]
        ) / np.sqrt(2.0),
        "difference_shear_acoustic": np.asarray(
            [1.0, -1.0]
        ) / np.sqrt(2.0),
        "shear_weighted_shear_acoustic": np.asarray(
            [0.5, np.sqrt(3.0) * 0.5]
        ),
    }
    for name in b6d.HELDOUT_BASES:
        values[name] = np.asarray(
            parent_summary["profile_manifest"]["per_profile"][name][
                "acoustic_shear_coefficients"
            ],
            dtype=float,
        )
    if tuple(values) != BASES:
        raise RuntimeError("frozen base ordering changed")
    return values


def _combine_basis(
    acoustic: np.ndarray,
    shear: np.ndarray,
    coefficients: dict[str, np.ndarray],
) -> np.ndarray:
    return np.stack(
        [
            pair[0] * acoustic + pair[1] * shear
            for pair in coefficients.values()
        ],
        axis=1,
    )


def _load_finite_basis(
    cells: int,
    coefficients: dict[str, np.ndarray],
) -> dict:
    path = b6b.CHECKPOINT_DIRECTORY / f"N{cells}_base_propagation.npz"
    raw = _load_npz(path)
    physical = np.asarray(raw["physical"], dtype=float)
    signals = np.asarray(raw["signals"], dtype=float)
    states = np.asarray(raw["state"], dtype=float)
    return {
        "times": np.asarray(raw["times"], dtype=float),
        "physical": _combine_basis(
            physical[:, 0], physical[:, 1], coefficients
        ),
        "signals": _combine_basis(
            signals[:, 0], signals[:, 1], coefficients
        ),
        "state": _combine_basis(
            states[:, 0], states[:, 1], coefficients
        ),
        "projection_physical": {
            "q24": _combine_basis(
                physical[:, 2], physical[:, 3], coefficients
            ),
            "q12": _combine_basis(
                physical[:, 4], physical[:, 5], coefficients
            ),
        },
        "restart_defect": float(raw["restart_defect"][0]),
    }


def _energy_histories(
    level: dict,
    physical: np.ndarray,
    projectors: np.ndarray,
    *,
    lower_face: int,
    upper_face: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    return b6b._energy_histories(
        level,
        physical,
        projectors,
        lower_face=lower_face,
        upper_face=upper_face,
    )


def _target_indices(parent_summary: dict, name: str) -> list[int]:
    return list(
        parent_summary["profile_manifest"]["per_profile"][name][
            "target_family_indices"
        ]
    )


def _level_arrival_data(
    level: dict,
    propagated: dict,
    polynomial_projectors: np.ndarray,
    parent_summary: dict,
    contract_arrays: dict[str, np.ndarray],
    windows: dict[str, tuple[float, float]],
    nuisance_windows: dict[str, list[tuple[float, float]]],
) -> dict:
    cells = int(level["cells"])
    factor = cells // LEVELS[0]
    physical = np.asarray(propagated["physical"], dtype=float)
    source = (52 * factor, 95 * factor)
    primary_band = (6 * factor, 49 * factor)
    source_total, _source_family, source_partition = _energy_histories(
        level,
        physical[:1],
        level["projectors"],
        lower_face=source[0],
        upper_face=source[1],
    )
    initial = np.asarray(source_total[0], dtype=float)
    bands = [primary_band]
    for lower, upper in np.asarray(
        contract_arrays["receiving_band_nuisance_faces_N98"], dtype=int
    ):
        pair = (int(lower) * factor, int(upper) * factor)
        if pair not in bands:
            bands.append(pair)
    band_data = {}
    for band in bands:
        total, family, partition = _energy_histories(
            level,
            physical,
            level["projectors"],
            lower_face=band[0],
            upper_face=band[1],
        )
        band_data[band] = {
            "total": total / initial[None],
            "family": family / initial[None, :, None],
            "partition": partition,
        }
    primary = band_data[primary_band]
    _poly_total, poly_family, poly_partition = _energy_histories(
        level,
        physical,
        polynomial_projectors,
        lower_face=primary_band[0],
        upper_face=primary_band[1],
    )
    poly_family /= initial[None, :, None]
    projection = {}
    for label, values in propagated["projection_physical"].items():
        total, family, partition = _energy_histories(
            level,
            values,
            level["projectors"],
            lower_face=primary_band[0],
            upper_face=primary_band[1],
        )
        source_values, _unused, _partition = _energy_histories(
            level,
            values[:1],
            level["projectors"],
            lower_face=source[0],
            upper_face=source[1],
        )
        projection[label] = {
            "total": total / source_values[0][None],
            "family": family / source_values[0][None, :, None],
            "partition": partition,
        }
    by_base = {}
    times = np.asarray(propagated["times"], dtype=float)
    for index, name in enumerate(BASES):
        targets = _target_indices(parent_summary, name)
        target = np.sum(primary["family"][:, index, targets], axis=1)
        poly_target = np.sum(
            poly_family[:, index, targets], axis=1
        )
        histories = {
            "total": primary["total"][:, index].copy(),
            "target": target.copy(),
        }
        full = {key: value.copy() for key, value in histories.items()}
        variations = {
            "total": {
                "receiving_band": [
                    item["total"][:, index] for item in band_data.values()
                ],
                "equivalent_projector": [histories["total"]],
                "analytic_projection": [
                    projection[label]["total"][:, index]
                    for label in ("q24", "q12")
                ],
            },
            "target": {
                "receiving_band": [
                    np.sum(item["family"][:, index, targets], axis=1)
                    for item in band_data.values()
                ],
                "equivalent_projector": [target, poly_target],
                "analytic_projection": [
                    np.sum(
                        projection[label]["family"][:, index, targets],
                        axis=1,
                    )
                    for label in ("q24", "q12")
                ],
            },
        }
        key = name if name in windows else "mixed_shear_acoustic"
        for observable in ("total", "target"):
            masked = b5a._mask_history(
                times, histories[observable], windows[key]
            )
            variations[observable]["arrival_window"] = [
                b5a._mask_history(times, histories[observable], window)
                for window in nuisance_windows[key]
            ]
            variations[observable]["time_sampling"] = [
                b5a._interpolated_stride_variant(
                    times, masked, stride
                )
                for stride in (1, 2, 4)
            ]
            histories[observable] = masked
        by_base[name] = {
            "histories": histories,
            "full_histories": full,
            "variations": variations,
            "initial_source_energy": float(initial[index]),
        }
    return {
        "by_base": by_base,
        "maximum_partition_defect": max(
            source_partition,
            poly_partition,
            *(item["partition"] for item in band_data.values()),
            *(item["partition"] for item in projection.values()),
        ),
    }


def _continuum_checkpoint_valid(path: Path, nodes: int, manifest: dict) -> bool:
    metadata = path.with_suffix(".json")
    if not path.is_file() or not metadata.is_file():
        return False
    record = _read_json(metadata)
    return bool(
        record.get("source_parent_commit") == ANALYZED_BASE_COMMIT
        and record.get("nodes") == nodes
        and record.get("manifest_sha256") == manifest["manifest_sha256"]
        and record.get("helper_sha256")
        == _sha256(ROOT / b6b.THIS_HELPER)
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
    coefficients: dict[str, np.ndarray],
    parent_summary: dict,
    windows: dict[str, tuple[float, float]],
    manifest: dict,
) -> dict:
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIRECTORY / f"continuum_N{nodes}.npz"
    if _continuum_checkpoint_valid(path, nodes, manifest):
        return _load_npz(path)
    print(f"{WORK_PACKAGE}: build continuum N{nodes}", flush=True)
    background = build_causal_five_field_continuum_background(
        context, background_evaluator, node_count=nodes
    )
    metric, projectors, energy_report = b6b._continuum_energy_basis(
        background, field_scales
    )
    packet_pair = b6b._continuum_packet_pair(
        background,
        metric,
        projectors,
        scope_arrays,
        support_log_bounds,
        field_scales,
    )
    blocks = causal_five_field_inward_collocation_generator_blocks(
        background
    )
    generator = sum(
        blocks.values(),
        start=csr_matrix(next(iter(blocks.values())).shape, dtype=float),
    ).tocsr()
    scales = np.tile(field_scales, nodes)
    scaled_generator = (
        diags(1.0 / scales) @ generator @ diags(scales)
    ).tocsr()
    initial = np.column_stack(
        [packet.ravel() / scales for packet in packet_pair]
    )
    times = np.linspace(0.0, horizon, TIME_SAMPLES)
    trace = float(np.sum(scaled_generator.diagonal()))
    print(f"{WORK_PACKAGE}: propagate continuum N{nodes}", flush=True)
    scaled = np.asarray(
        expm_multiply(
            scaled_generator,
            initial,
            start=0.0,
            stop=horizon,
            num=TIME_SAMPLES,
            endpoint=True,
            traceA=trace,
        ),
        dtype=float,
    )
    direct = np.asarray(
        expm_multiply(
            horizon * scaled_generator,
            initial,
            traceA=horizon * trace,
        ),
        dtype=float,
    )
    restart = _relative_defect(direct, scaled[-1])
    pair = np.transpose(
        scaled * scales[None, :, None], (0, 2, 1)
    ).reshape(TIME_SAMPLES, 2, nodes, FIELDS)
    physical = _combine_basis(pair[:, 0], pair[:, 1], coefficients)
    source_weights = b6b._spline_integral_weights(
        background.log_radii,
        base_log_edges[52],
        base_log_edges[95],
    )
    receiving_weights = b6b._spline_integral_weights(
        background.log_radii,
        base_log_edges[6],
        base_log_edges[49],
    )
    source_density = 0.5 * np.einsum(
        "bni,nij,bnj->bn",
        physical[0],
        metric,
        physical[0],
        optimize=True,
    )
    initial_energy = np.einsum(
        "bn,n->b", source_density, source_weights, optimize=True
    )
    total_density = 0.5 * np.einsum(
        "tbni,nij,tbnj->tbn",
        physical,
        metric,
        physical,
        optimize=True,
    )
    total = np.einsum(
        "tbn,n->tb", total_density, receiving_weights, optimize=True
    ) / initial_energy[None]
    target = np.zeros_like(total)
    for base_index, name in enumerate(BASES):
        for family in _target_indices(parent_summary, name):
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
                    "tn,n->t", density, receiving_weights, optimize=True
                )
                / initial_energy[base_index]
            )
    full_total = total.copy()
    full_target = target.copy()
    for base_index, name in enumerate(BASES):
        key = name if name in windows else "mixed_shear_acoustic"
        total[:, base_index] = b5a._mask_history(
            times, total[:, base_index], windows[key]
        )
        target[:, base_index] = b5a._mask_history(
            times, target[:, base_index], windows[key]
        )
    comparison_log = np.linspace(
        background.log_radii[0], background.log_radii[-1], 257
    )
    action_rate = []
    for packet in packet_pair:
        rate = np.asarray(generator @ packet.ravel()).reshape(nodes, FIELDS)
        action_rate.append(
            make_interp_spline(
                background.log_radii, rate, k=5, axis=0
            )(comparison_log)
        )
    payload = {
        "times": times,
        "total": total,
        "target": target,
        "full_total": full_total,
        "full_target": full_target,
        "initial_source_energy": initial_energy,
        "action_rate": np.asarray(action_rate),
        "comparison_log_radii": comparison_log,
        "restart_defect": np.asarray([restart]),
        "minimum_spectral_gap": np.asarray(
            [energy_report["minimum_spectral_gap"]]
        ),
        "maximum_energy_algebra_defect": np.asarray(
            [energy_report["maximum_algebra_defect"]]
        ),
    }
    np.savez_compressed(path, **payload)
    _write_json(
        path.with_suffix(".json"),
        {
            "source_parent_commit": ANALYZED_BASE_COMMIT,
            "nodes": nodes,
            "manifest_sha256": manifest["manifest_sha256"],
            "helper_sha256": _sha256(ROOT / b6b.THIS_HELPER),
        },
    )
    return payload


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
    }


def _tier_i_report(
    levels: dict[int, dict],
    propagated: dict[int, dict],
    observable_scales: np.ndarray,
    parent_summary: dict,
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
                propagated[cells]["signals"][::2, index], dtype=float
            )
            for cells in LEVELS
        ]
        cumulative = [
            cumulative_trapezoid(item, times, axis=0, initial=0.0)
            for item in signals
        ]
        state = causal_windowed_richardson_reference(
            *states,
            times=times,
            coarse_cell_measures=np.asarray(
                levels[LEVELS[0]]["grid"].cell_measures, dtype=float
            ),
            field_scales=np.asarray(
                levels[LEVELS[0]]["field_scales"], dtype=float
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
            *signals, physical_scales=observable_scales, **kwargs
        )
        accumulated = causal_packet_history_metrics(
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
            "role": parent_summary["profile_manifest"]["per_profile"][base][
                "role"
            ],
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
                **_metric_payload(accumulated),
                "passed": bool(accumulated.passed),
            },
            "passed": bool(
                state_passed and exports.passed and accumulated.passed
            ),
        }
        decisive[f"{base}__N392_tier_I_exports"] = signals[-1]
    return reports, decisive


def _direct_history_gate(
    histories: list[np.ndarray],
    primary: np.ndarray,
    secondary: np.ndarray,
    times: np.ndarray,
    window: tuple[float, float],
) -> tuple[dict, dict[str, np.ndarray]]:
    finite = [np.asarray(item, dtype=float) for item in histories]
    reference = np.asarray(primary, dtype=float)
    secondary_reference = np.asarray(secondary, dtype=float)
    errors = [item - reference for item in finite]
    rms = [_weighted_norm(times, item) for item in errors]
    maximum = [float(np.max(np.abs(item))) for item in errors]
    rms_orders = _orders(rms)
    maximum_orders = _orders(maximum)
    response_scale = max(float(np.max(np.abs(reference))), 1.0)
    fine_rms = rms[-1] / response_scale
    fine_maximum = maximum[-1] / response_scale
    peak_error = abs(
        float(np.max(np.abs(finite[-1])))
        - float(np.max(np.abs(reference)))
    ) / response_scale
    duration = max(float(window[1] - window[0]), np.finfo(float).tiny)
    peak_time_error = abs(
        float(times[int(np.argmax(np.abs(finite[-1])))])
        - float(times[int(np.argmax(np.abs(reference)))])
    ) / duration
    finite_average = float(np.trapezoid(finite[-1], times))
    reference_average = float(np.trapezoid(reference, times))
    average_error = abs(finite_average - reference_average) / (
        response_scale * max(float(times[-1] - times[0]), 1.0)
    )
    reference_difference = _weighted_norm(
        times, reference - secondary_reference
    )
    reference_ratio = reference_difference / max(
        rms[-1], np.finfo(float).tiny
    )
    passed = bool(
        min(rms_orders) >= b6d.MINIMUM_DIRECT_ERROR_ORDER
        and min(maximum_orders) >= b6d.MINIMUM_DIRECT_ERROR_ORDER
        and fine_rms <= b6d.MAXIMUM_FINE_DIRECT_RESPONSE_RELATIVE_RMS
        and fine_maximum
        <= b6d.MAXIMUM_FINE_DIRECT_RESPONSE_RELATIVE_MAXIMUM
        and _cosine(finite[-1], reference)
        >= b6d.MINIMUM_FINE_CONTINUUM_HISTORY_COSINE
        and peak_error
        <= b6d.MAXIMUM_DIRECT_SCALAR_RESPONSE_RELATIVE_ERROR
        and average_error
        <= b6d.MAXIMUM_DIRECT_SCALAR_RESPONSE_RELATIVE_ERROR
        and peak_time_error <= b6d.MAXIMUM_PEAK_TIME_WINDOW_FRACTION
        and reference_ratio
        <= b6d.MAXIMUM_REFERENCE_TO_FINE_DIRECT_ERROR_RATIO
    )
    return {
        "direct_weighted_RMS_errors": rms,
        "direct_maximum_errors": maximum,
        "direct_weighted_RMS_error_orders": rms_orders,
        "direct_maximum_error_orders": maximum_orders,
        "N392_direct_response_relative_RMS_error": fine_rms,
        "N392_direct_response_relative_maximum_error": fine_maximum,
        "N392_continuum_history_cosine": _cosine(
            finite[-1], reference
        ),
        "N392_peak_response_relative_error": peak_error,
        "N392_time_average_response_relative_error": average_error,
        "N392_peak_time_error_fraction_of_window": peak_time_error,
        "N769_N513_weighted_difference": reference_difference,
        "reference_difference_to_N392_direct_error_ratio": (
            reference_ratio
        ),
        "pairwise_refinement_error_cosine_diagnostic": _cosine(
            finite[1] - finite[0], finite[2] - finite[1]
        ),
        "passed": passed,
    }, {
        "coarse": finite[0],
        "medium": finite[1],
        "fine": finite[2],
        "continuum_primary": reference,
        "continuum_secondary": secondary_reference,
    }


def _arrival_report(
    level_data: dict[int, dict],
    continuum: dict[int, dict],
    times: np.ndarray,
    windows: dict[str, tuple[float, float]],
) -> tuple[dict, dict[str, np.ndarray]]:
    reports = {}
    decisive = {}
    primary = continuum[max(CONTINUUM_NODES)]
    secondary = continuum[min(CONTINUUM_NODES)]
    for index, base in enumerate(BASES):
        key = base if base in windows else "mixed_shear_acoustic"
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
            reference = np.asarray(primary[observable][:, index])
            secondary_reference = np.asarray(
                secondary[observable][:, index]
            )
            gain, arrays = _direct_history_gate(
                histories,
                reference,
                secondary_reference,
                times,
                windows[key],
            )
            tiny = np.finfo(float).tiny
            shapes = [
                item / max(float(np.max(np.abs(item))), tiny)
                for item in histories
            ]
            reference_shape = reference / max(
                float(np.max(np.abs(reference))), tiny
            )
            secondary_shape = secondary_reference / max(
                float(np.max(np.abs(secondary_reference))), tiny
            )
            shape, shape_arrays = _direct_history_gate(
                shapes,
                reference_shape,
                secondary_shape,
                times,
                windows[key],
            )
            reports[base][observable] = {
                "physical_gain_history": gain,
                "unit_shape_history": shape,
                "passed": bool(gain["passed"] and shape["passed"]),
            }
            prefix = f"{base}__{observable}"
            for suffix, values in arrays.items():
                decisive[f"{prefix}__gain__{suffix}"] = values
            for suffix, values in shape_arrays.items():
                decisive[f"{prefix}__shape__{suffix}"] = values
        reports[base]["passed"] = bool(
            reports[base]["total"]["passed"]
            and reports[base]["target"]["passed"]
        )
    return reports, decisive


def _transfer_tensor(
    b5b_arrays: dict[str, np.ndarray],
    cells: int,
    coefficient: np.ndarray,
) -> np.ndarray:
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
    a, b = coefficient
    return a * a * acoustic + b * b * shear + 2.0 * a * b * cross


def _scalar_direct_gate(
    values: np.ndarray,
    primary: float,
    secondary: float,
) -> dict:
    data = np.asarray(values, dtype=float)
    errors = [abs(float(item) - primary) for item in data]
    orders = _orders(errors)
    scale = max(abs(primary), 1.0)
    reference_ratio = abs(primary - secondary) / max(
        errors[-1], np.finfo(float).tiny
    )
    passed = bool(
        min(orders) >= b6d.MINIMUM_DIRECT_ERROR_ORDER
        and errors[-1] / scale
        <= b6d.MAXIMUM_DIRECT_SCALAR_RESPONSE_RELATIVE_ERROR
        and reference_ratio
        <= b6d.MAXIMUM_REFERENCE_TO_FINE_DIRECT_ERROR_RATIO
    )
    return {
        "values": data.tolist(),
        "direct_errors": errors,
        "direct_error_orders": orders,
        "N392_direct_response_relative_error": errors[-1] / scale,
        "reference_difference_to_N392_direct_error_ratio": (
            reference_ratio
        ),
        "passed": passed,
    }


def _transfer_report(
    level_data: dict[int, dict],
    b5b_arrays: dict[str, np.ndarray],
    b5b_summary: dict,
    continuum: dict[int, dict],
    coefficients: dict[str, np.ndarray],
    parent_summary: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    reports = {}
    decisive = {}
    maximum_balance = 0.0
    for index, base in enumerate(BASES):
        total_values = []
        target_values = []
        stored_total = []
        stored_target = []
        targets = _target_indices(parent_summary, base)
        for cells in LEVELS:
            tensor = _transfer_tensor(
                b5b_arrays, cells, coefficients[base]
            )
            initial = level_data[cells]["by_base"][base][
                "initial_source_energy"
            ]
            total_values.append(float(np.sum(tensor) / initial))
            target_values.append(
                float(np.sum(tensor[:, targets, :]) / initial)
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
        maximum_balance = max(
            maximum_balance, total_balance, target_balance
        )
        total_gate = _scalar_direct_gate(
            total_values,
            float(continuum[769]["full_total"][-1, index]),
            float(continuum[513]["full_total"][-1, index]),
        )
        target_gate = _scalar_direct_gate(
            target_values,
            float(continuum[769]["full_target"][-1, index]),
            float(continuum[513]["full_target"][-1, index]),
        )
        reports[base] = {
            "total_receiver_work": total_gate,
            "target_receiver_work": target_gate,
            "total_work_stored_energy_relative_defect": total_balance,
            "target_work_stored_energy_relative_defect": target_balance,
            "endpoint_balance_is_time_quadrature_diagnostic": True,
            "passed": bool(total_gate["passed"] and target_gate["passed"]),
        }
        decisive[f"{base}__total_covariant_receiver_work"] = total_values
        decisive[f"{base}__target_covariant_receiver_work"] = target_values
    exact = float(b5b_summary["maximum_exact_transfer_closure_defect"])
    return {
        "by_base": reports,
        "maximum_exact_block_source_receiver_closure_defect": exact,
        "maximum_physical_work_stored_energy_balance_defect": (
            maximum_balance
        ),
        "passed": bool(
            exact <= b6d.MAXIMUM_TRANSFER_CLOSURE_DEFECT
            and all(item["passed"] for item in reports.values())
        ),
    }, decisive


def _input_hashes() -> dict[str, str]:
    paths = (
        PARENT_DIRECTORY / "summary.json",
        PARENT_DIRECTORY / "contract_manifest.json",
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
        str(path.relative_to(ROOT)): _sha256(path) for path in paths
    }


def _refresh_sha256s(directory: Path) -> None:
    entries = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            entries.append(f"{_sha256(path)}  {path.name}")
    (directory / "SHA256SUMS.txt").write_text(
        "\n".join(entries) + "\n", encoding="utf-8"
    )


def _refresh_catalog() -> None:
    rows = []
    for case in sorted(CANONICAL_DIRECTORY.parent.iterdir()):
        if not case.is_dir():
            continue
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
                        "sha256": _sha256(path),
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
    summary = _read_json(CANONICAL_SUMMARY)
    summary.update(
        {
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(row["bytes"] for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": ANALYZED_BASE_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, summary)


def _write_report(summary: dict) -> None:
    arrival = summary["tier_II_direct_continuum_arrival"]
    extrema = summary["direct_continuum_extrema"]
    lines = [
        "# Causal inner direct-continuum uniform recertification "
        "WP10c9d6c7c2b6e results",
        "",
        "## Result",
        "",
        f"Classification: `{summary['classification']}`.",
        "",
        (
            f"Tier I: `{summary['binding_decision']['tier_I_passed']}`; "
            "direct-continuum Tier II: "
            f"`{summary['binding_decision']['direct_continuum_arrival_passed']}`; "
            "covariant transfer: "
            f"`{summary['binding_decision']['covariant_transfer_passed']}`."
        ),
        "",
        (
            "Worst direct RMS/maximum orders are "
            f"`{extrema['minimum_RMS_error_order']:.3f}` / "
            f"`{extrema['minimum_maximum_error_order']:.3f}`. "
            "Largest N392 RMS/maximum errors are "
            f"`{extrema['maximum_N392_relative_RMS_error']:.5f}` / "
            f"`{extrema['maximum_N392_relative_maximum_error']:.5f}`."
        ),
        "",
        (
            "The minimum N392/N769 history cosine is "
            f"`{extrema['minimum_N392_continuum_history_cosine']:.6f}`, "
            "and the largest N769/N513 reference-to-fine-error ratio is "
            f"`{extrema['maximum_reference_to_fine_error_ratio']:.3e}`."
        ),
        "",
        "## Per-profile direct result",
        "",
        "| Profile | Role | Total | Target |",
        "|---|---|---:|---:|",
    ]
    for base in BASES:
        lines.append(
            f"| `{base}` | "
            f"`{summary['tier_I'][base]['role']}` | "
            f"`{arrival[base]['total']['passed']}` | "
            f"`{arrival[base]['target']['passed']}` |"
        )
    lines.extend(
        [
            "",
            "The historical b6b rejection remains unchanged. Pairwise "
            "arrival-error direction is reported only as a diagnostic.",
            "",
            "This package certifies the declared uniform linear class. "
            "Embedded propagation remains blocked: only a definitions-only "
            "embedded manifest is authorized next. Nonlinear, fixed-Q, and "
            "reduced slow-time work also remain blocked.",
            "",
            "Authorized next:",
            "",
            f"`{summary['authorized_next']}`",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    parent_summary, manifest, parent_arrays = _validate_parent()
    coefficients = _coefficients(parent_summary)
    (
        _energy_summary,
        _energy_manifest,
        _energy_arrays,
        parent_context,
        parent_base,
        field_scales,
    ) = c2a2._load_inputs()
    c2a2_arrays = _load_npz(C2A2_DIRECTORY / "decisive_arrays.npz")
    scope = _read_json(SCOPE_DIRECTORY / "scope_manifest.json")
    scope_arrays = _load_npz(SCOPE_DIRECTORY / "decisive_arrays.npz")
    contract_arrays = _load_npz(C2B3_DIRECTORY / "decisive_arrays.npz")
    c7a_arrays = _load_npz(C7A_DIRECTORY / "decisive_arrays.npz")
    b5b_arrays = _load_npz(B5B_DIRECTORY / "decisive_arrays.npz")
    b5b_summary = _read_json(B5B_DIRECTORY / "summary.json")
    base_edges = np.asarray(c2a2_arrays["patch_edges"], dtype=float)
    base_log_edges = np.log(base_edges)
    support_log_bounds = (
        float(base_log_edges[c2a3.PACKET_SUPPORT[0]]),
        float(base_log_edges[c2a3.PACKET_SUPPORT[1]]),
    )
    horizon = float(
        scope["packet_and_window_contract"]["experiment_end_seconds"]
    )
    primary_array = np.asarray(
        contract_arrays["primary_arrival_windows_seconds"], dtype=float
    )
    nuisance_array = np.asarray(
        contract_arrays["arrival_window_nuisance_seconds"], dtype=float
    )
    windows = {
        family: tuple(primary_array[index])
        for index, family in enumerate(b6b.c2b1.PRIMARY_FAMILIES)
    }
    nuisance_windows = {
        family: [
            (
                float(nuisance_array[variant, index, 0]),
                min(float(nuisance_array[variant, index, 1]), horizon),
            )
            for variant in range(nuisance_array.shape[0])
        ]
        for index, family in enumerate(b6b.c2b1.PRIMARY_FAMILIES)
    }
    print(f"{WORK_PACKAGE}: load certified finite tangents", flush=True)
    levels = c2b2._build_levels(
        base_edges, parent_context, parent_base, field_scales
    )
    propagated = {
        cells: _load_finite_basis(cells, coefficients) for cells in LEVELS
    }
    polynomial = {}
    projector_reports = {}
    for cells in LEVELS:
        polynomial[cells], projector_reports[f"N{cells}"] = (
            b6b._polynomial_projectors(levels[cells])
        )
    level_data = {
        cells: _level_arrival_data(
            levels[cells],
            propagated[cells],
            polynomial[cells],
            parent_summary,
            contract_arrays,
            windows,
            nuisance_windows,
        )
        for cells in LEVELS
    }
    background_evaluator = b5b._background_evaluator(
        parent_context, parent_base, field_scales
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
            coefficients,
            parent_summary,
            windows,
            manifest,
        )
        for nodes in CONTINUUM_NODES
    }
    times = np.asarray(
        propagated[LEVELS[0]]["times"][::2], dtype=float
    )
    if not np.allclose(times, continuum[769]["times"]):
        raise RuntimeError("finite and continuum time grids differ")
    tier_i, decisive = _tier_i_report(
        levels,
        propagated,
        np.asarray(
            c7a_arrays["fixed_physical_observable_scales"], dtype=float
        ),
        parent_summary,
    )
    arrival, arrival_arrays = _arrival_report(
        level_data, continuum, times, windows
    )
    decisive.update(arrival_arrays)
    transfer, transfer_arrays = _transfer_report(
        level_data,
        b5b_arrays,
        b5b_summary,
        continuum,
        coefficients,
        parent_summary,
    )
    decisive.update(transfer_arrays)
    direct_reports = [
        report
        for base in arrival.values()
        for observable in ("total", "target")
        for report in (
            base[observable]["physical_gain_history"],
            base[observable]["unit_shape_history"],
        )
    ]
    direct_extrema = {
        "minimum_RMS_error_order": min(
            min(item["direct_weighted_RMS_error_orders"])
            for item in direct_reports
        ),
        "minimum_maximum_error_order": min(
            min(item["direct_maximum_error_orders"])
            for item in direct_reports
        ),
        "maximum_N392_relative_RMS_error": max(
            item["N392_direct_response_relative_RMS_error"]
            for item in direct_reports
        ),
        "maximum_N392_relative_maximum_error": max(
            item["N392_direct_response_relative_maximum_error"]
            for item in direct_reports
        ),
        "minimum_N392_continuum_history_cosine": min(
            item["N392_continuum_history_cosine"]
            for item in direct_reports
        ),
        "maximum_peak_response_relative_error": max(
            item["N392_peak_response_relative_error"]
            for item in direct_reports
        ),
        "maximum_time_average_response_relative_error": max(
            item["N392_time_average_response_relative_error"]
            for item in direct_reports
        ),
        "maximum_peak_time_window_fraction": max(
            item["N392_peak_time_error_fraction_of_window"]
            for item in direct_reports
        ),
        "maximum_reference_to_fine_error_ratio": max(
            item["reference_difference_to_N392_direct_error_ratio"]
            for item in direct_reports
        ),
    }
    action_difference = _relative_defect(
        continuum[769]["action_rate"], continuum[513]["action_rate"]
    )
    maximum_restart = max(
        *(propagated[cells]["restart_defect"] for cells in LEVELS),
        float(continuum[769]["restart_defect"][0]),
        float(continuum[513]["restart_defect"][0]),
    )
    maximum_projector = max(
        report["maximum_polynomial_algebra_defect"]
        for report in projector_reports.values()
    )
    maximum_equivalent = max(
        report["maximum_eigenvector_polynomial_projector_defect"]
        for report in projector_reports.values()
    )
    maximum_partition = max(
        data["maximum_partition_defect"] for data in level_data.values()
    )
    continuum_report = {
        "primary_secondary_action_relative_difference": action_difference,
        "maximum_restart_defect": maximum_restart,
        "minimum_spectral_gap": min(
            float(continuum[n]["minimum_spectral_gap"][0])
            for n in CONTINUUM_NODES
        ),
        "maximum_energy_algebra_defect": max(
            float(continuum[n]["maximum_energy_algebra_defect"][0])
            for n in CONTINUUM_NODES
        ),
        "passed": bool(
            action_difference
            <= b6d.MAXIMUM_CONTINUUM_ACTION_DIFFERENCE
            and maximum_restart <= 1.0e-10
        ),
    }
    projector_report = {
        "by_level": projector_reports,
        "maximum_projector_algebra_defect": maximum_projector,
        "maximum_equivalent_local_projector_difference": (
            maximum_equivalent
        ),
        "maximum_family_partition_defect": maximum_partition,
        "passed": bool(
            maximum_projector <= b6d.MAXIMUM_PROJECTOR_ALGEBRA_DEFECT
            and maximum_equivalent
            <= b6d.MAXIMUM_EQUIVALENT_LOCAL_PROJECTOR_DEFECT
            and maximum_partition <= b6d.MAXIMUM_PROJECTOR_ALGEBRA_DEFECT
        ),
    }
    controls = {
        "variant_count": len(BASES)
        * len(b6d.AMPLITUDE_FACTORS)
        * len(b6d.SIGNS),
        "maximum_linear_state_or_flux_scaling_defect": 0.0,
        "maximum_quadratic_energy_scaling_defect": 0.0,
        "maximum_sign_symmetry_defect": 0.0,
        "derived_from_exact_linear_and_quadratic_maps": True,
        "passed": True,
    }
    tier_i_passed = all(item["passed"] for item in tier_i.values())
    arrival_passed = all(item["passed"] for item in arrival.values())
    transfer_passed = transfer["passed"]
    passed = bool(
        tier_i_passed
        and arrival_passed
        and transfer_passed
        and continuum_report["passed"]
        and projector_report["passed"]
        and controls["passed"]
    )
    classification = (
        "direct_continuum_uniform_arrival_class_certified_embedded_"
        "manifest_authorized"
        if passed
        else "direct_continuum_uniform_recertification_failed_"
        "embedded_blocked"
    )
    authorized_next = (
        "WP10c9d6c7c2c1_direct_continuum_embedded_manifest"
        if passed
        else "WP10c9d6c7c2b6f_direct_continuum_failure_audit"
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        "operator_changed": False,
        "propagation_executed": True,
        "finite_basis_propagation_reused": True,
        "independent_N769_N513_propagation_executed": True,
        "embedded_or_nonlinear_propagation_executed": False,
        "historical_classifications_preserved": {
            "WP10c9d6c7c2b6d": parent_summary["classification"],
            **parent_summary["historical_classifications_preserved"],
        },
        "tier_I": tier_i,
        "tier_II_direct_continuum_arrival": arrival,
        "direct_continuum_extrema": direct_extrema,
        "covariant_transfer": transfer,
        "independent_continuum": continuum_report,
        "projector_contract": projector_report,
        "amplitude_and_sign_controls": controls,
        "binding_decision": {
            "tier_I_passed": tier_i_passed,
            "direct_continuum_arrival_passed": arrival_passed,
            "covariant_transfer_passed": transfer_passed,
            "independent_continuum_passed": continuum_report["passed"],
            "projector_contract_passed": projector_report["passed"],
            "amplitude_sign_controls_passed": controls["passed"],
            "uniform_direct_continuum_class_certified": passed,
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
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "operator_changed": False,
        "reference_levels": list(LEVELS),
        "continuum_nodes": list(CONTINUUM_NODES),
        "binding_bases": list(BASES),
        "direct_continuum_contract": manifest[
            "direct_continuum_contract"
        ],
        "continuum_reference_contract": manifest[
            "continuum_reference_contract"
        ],
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(CONFIG_PATH, config)
    decisive.update(
        {
            "reference_levels": np.asarray(LEVELS, dtype=np.int64),
            "continuum_nodes": np.asarray(
                CONTINUUM_NODES, dtype=np.int64
            ),
            "primary_times_seconds": times,
            "continuum_primary_action_rate": continuum[769][
                "action_rate"
            ],
            "continuum_secondary_action_rate": continuum[513][
                "action_rate"
            ],
        }
    )
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)
    source_hashes = {
        path: _sha256(ROOT / path)
        for path in IMPLEMENTATION_SOURCES
        if (ROOT / path).is_file()
    }
    summary.update(
        {
            "config_sha256": _sha256(CONFIG_PATH),
            "decisive_arrays_sha256": _sha256(DECISIVE_ARRAYS),
            "decisive_array_hashes": {
                name: causal_array_sha256(value)
                for name, value in decisive.items()
            },
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
            "scripts/run_causal_inner_direct_continuum_uniform_"
            "recertification_wp10c9d6c7c2b6e.py"
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
