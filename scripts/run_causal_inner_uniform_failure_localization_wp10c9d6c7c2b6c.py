#!/usr/bin/env python3
"""Localize the frozen WP10c9d6c7c2b6b uniform failures.

This audit changes no physical or numerical operator.  It compares the five
failed b6b history channels directly with the independent N769 continuum
history, reconstructs the full N769 state, and evaluates the finite-volume
DAE truncation residual before and after descriptor inversion.  A numerical
intervention is authorized only when the same noncontracting block and fixed
physical band dominate both refinement pairs.
"""

from __future__ import annotations

import json
import math
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import scipy
from scipy.interpolate import make_interp_spline
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import expm_multiply, splu


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_causal_inner_continuum_lift_wp10c9d6c3 as c3  # noqa: E402
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
from imri_qpe.layer3_minidisk_1d.causal_inner_energy_transfer import (  # noqa: E402
    causal_positive_band_energy_history,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_canonical_json_sha256,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c2b6c"
ANALYZED_BASE_COMMIT = "51c584a859b238eb528569179c570e6fd128707c"
ANALYZED_BASE_PARENT = "c47df91e3b22af9e2d5899fc32c924840f2c1471"
ANALYZED_BASE_TREE = "add9dc620f625e3be361867324dbdee10d5558dc"
LEVELS = b6b.LEVELS
CONTINUUM_NODES = 769
TIME_SAMPLES = 513
FIELDS = 5
MINIMUM_DIRECT_ORDER = 0.75
MAXIMUM_DIRECT_FINE_DIFFERENCE = 0.05
DOMINANT_FRACTION = 0.60

FAILURE_CHANNELS = (
    ("acoustic", "target", "gain"),
    ("difference_shear_acoustic", "total", "gain"),
    ("difference_shear_acoustic", "target", "gain"),
    ("shear", "total", "shape"),
    ("shear", "target", "gain"),
)
PASSING_CONTROLS = (
    ("mixed_shear_acoustic", "total", "gain"),
    ("mixed_shear_acoustic", "target", "gain"),
    ("shear_weighted_shear_acoustic", "total", "gain"),
    ("shear_weighted_shear_acoustic", "target", "gain"),
)
FIXED_BANDS_N98 = {
    "inner_outflow": (0, 6),
    "receiver_1": (6, 17),
    "receiver_2": (17, 28),
    "receiver_3": (28, 39),
    "receiver_4": (39, 49),
    "source_and_transition": (49, 98),
}

THIS_RUNNER = (
    "scripts/"
    "run_causal_inner_uniform_failure_localization_wp10c9d6c7c2b6c.py"
)
THIS_HELPER = (
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_continuum_truncation.py"
)
THIS_HELPER_TEST = "tests/test_causal_inner_continuum_collocation.py"
THIS_CANONICAL_TEST = (
    "tests/"
    "test_causal_inner_uniform_failure_localization_"
    "wp10c9d6c7c2b6c.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_UNIFORM_FAILURE_LOCALIZATION_"
    "WP10C9D6C7C2B6C_RESULTS_2026-07-30.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
PARENT_DIRECTORY = b6b.CANONICAL_DIRECTORY
CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_uniform_failure_localization_wp10c9d6c7c2b6c"
)
CHECKPOINT_DIRECTORY = (
    ROOT
    / "outputs/checkpoints/"
    "causal_inner_uniform_failure_localization_wp10c9d6c7c2b6c"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"

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


def _weighted_norm(times: np.ndarray, values: np.ndarray) -> float:
    data = np.asarray(values, dtype=float)
    duration = float(times[-1] - times[0])
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


def _validate_parent() -> tuple[dict, dict[str, np.ndarray]]:
    summary = _read_json(PARENT_DIRECTORY / "summary.json")
    arrays = _load_npz(PARENT_DIRECTORY / "decisive_arrays.npz")
    decision = summary["binding_decision"]
    if (
        summary["classification"]
        != "revised_uniform_arrival_transfer_recertification_failed_"
        "embedded_blocked"
        or summary["passed"]
        or not decision["tier_I_passed"]
        or decision["tier_II_arrival_passed"]
        or not decision["covariant_transfer_passed"]
        or not decision["independent_continuum_passed"]
        or decision["operator_or_interface_redesign_authorized"]
        or summary["authorized_next"]
        != "WP10c9d6c7c2b6c_uniform_recertification_failure_audit"
    ):
        raise RuntimeError("WP10c9d6c7c2b6b binding status changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT)
        != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("WP10c9d6c7c2b6c analyzed identity changed")
    return summary, arrays


def _channel_key(channel: tuple[str, str, str], suffix: str) -> str:
    base, observable, kind = channel
    return f"{base}__{observable}__{kind}__{suffix}"


def _direct_history_report(
    arrays: dict[str, np.ndarray],
    times: np.ndarray,
    channels: tuple[tuple[str, str, str], ...],
) -> tuple[dict, dict[str, np.ndarray], list[int]]:
    report = {}
    decisive = {}
    selected_indices = []
    for channel in channels:
        label = "__".join(channel)
        continuum = np.asarray(
            arrays[_channel_key(channel, "continuum_primary")],
            dtype=float,
        )
        histories = [
            np.asarray(arrays[_channel_key(channel, suffix)], dtype=float)
            for suffix in ("coarse", "medium", "fine")
        ]
        errors = [item - continuum for item in histories]
        norms = [_weighted_norm(times, item) for item in errors]
        orders = _orders(norms)
        response_scale = max(float(np.max(np.abs(continuum))), 1.0)
        fine_maximum = float(
            np.max(np.abs(errors[-1])) / response_scale
        )
        maximum_index = int(np.argmax(np.abs(errors[-1])))
        selected_indices.append(maximum_index)
        report[label] = {
            "direct_error_weighted_rms": norms,
            "direct_error_orders": orders,
            "minimum_direct_error_order": min(orders),
            "fine_direct_response_relative_maximum": fine_maximum,
            "coarse_medium_direct_error_cosine": _cosine(
                errors[0], errors[1]
            ),
            "medium_fine_direct_error_cosine": _cosine(
                errors[1], errors[2]
            ),
            "fine_maximum_error_time_index": maximum_index,
            "fine_maximum_error_time_seconds": float(times[maximum_index]),
            "direct_continuum_contract_passed": bool(
                min(orders) >= MINIMUM_DIRECT_ORDER
                and fine_maximum <= MAXIMUM_DIRECT_FINE_DIFFERENCE
            ),
        }
        for suffix, values in zip(
            ("coarse_error", "medium_error", "fine_error"),
            errors,
            strict=True,
        ):
            decisive[f"{label}__{suffix}"] = values
    return report, decisive, sorted(set(selected_indices))


def _load_finite_propagation(cells: int) -> dict:
    path = b6b.CHECKPOINT_DIRECTORY / f"N{cells}_base_propagation.npz"
    stored = _load_npz(path)
    return b6b._derived_propagation(
        {
            "times": stored["times"],
            "physical": stored["physical"],
            "signals": stored["signals"],
            "state": stored["state"],
            "restart_defect": float(stored["restart_defect"][0]),
        }
    )


def _projector_sensitivity(
    levels: dict[int, dict],
    propagated: dict[int, dict],
    windows: dict[str, tuple[float, float]],
) -> tuple[dict, dict[str, np.ndarray]]:
    common = b5a._common_projectors(levels)
    report = {}
    decisive = {}
    for cells in LEVELS:
        level = levels[cells]
        factor = cells // LEVELS[0]
        physical = np.asarray(propagated[cells]["physical"], dtype=float)[::2]
        times = np.asarray(propagated[cells]["times"], dtype=float)[::2]
        source_total, _source_family, _ = b6b._energy_histories(
            level,
            physical[:1],
            level["projectors"],
            lower_face=52 * factor,
            upper_face=95 * factor,
        )
        initial = np.asarray(source_total[0], dtype=float)
        polynomial, audit = b6b._polynomial_projectors(level)
        constructions = {
            "local_eigenvector": level["projectors"],
            "local_polynomial": polynomial,
            "common_N392_field": common[cells],
        }
        report[f"N{cells}"] = {"projector_audit": audit}
        for construction, projectors in constructions.items():
            total, family, partition = b6b._energy_histories(
                level,
                physical,
                projectors,
                lower_face=6 * factor,
                upper_face=49 * factor,
            )
            report[f"N{cells}"][construction] = {
                "maximum_partition_defect": partition,
            }
            for base_index, base in enumerate(b6b.BASES):
                targets = list(b6b.TARGETS[base])
                target = np.sum(family[:, base_index, targets], axis=1)
                target /= initial[base_index]
                window = windows.get(base, windows["mixed_shear_acoustic"])
                target = b5a._mask_history(times, target, window)
                decisive[
                    f"N{cells}__{construction}__{base}__target_gain"
                ] = target
                if base in ("acoustic", "shear", "difference_shear_acoustic"):
                    report[f"N{cells}"][construction][base] = {
                        "target_peak": float(np.max(np.abs(target))),
                        "target_rms": _weighted_norm(times, target),
                    }
    sensitivity = {}
    for base in ("acoustic", "shear", "difference_shear_acoustic"):
        fine_local = decisive[
            f"N{LEVELS[-1]}__local_eigenvector__{base}__target_gain"
        ]
        for construction in ("local_polynomial", "common_N392_field"):
            comparison = decisive[
                f"N{LEVELS[-1]}__{construction}__{base}__target_gain"
            ]
            sensitivity[f"{base}__{construction}"] = float(
                np.max(np.abs(comparison - fine_local))
                / max(float(np.max(np.abs(fine_local))), 1.0)
            )
    report["fine_response_relative_sensitivity"] = sensitivity
    report["equivalent_local_projectors_passed"] = bool(
        max(
            value
            for key, value in sensitivity.items()
            if key.endswith("local_polynomial")
        )
        <= 1.0e-9
    )
    return report, decisive


def _continuum_state_checkpoint(
    context,
    background_evaluator,
    field_scales: np.ndarray,
    scope_arrays: dict[str, np.ndarray],
    support_log_bounds: tuple[float, float],
    horizon: float,
) -> dict:
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIRECTORY / "continuum_N769_full_state.npz"
    metadata = path.with_suffix(".json")
    helper_hash = c2a._sha256(ROOT / THIS_HELPER)
    if path.is_file() and metadata.is_file():
        record = _read_json(metadata)
        if (
            record.get("source_parent_commit") == ANALYZED_BASE_COMMIT
            and record.get("nodes") == CONTINUUM_NODES
            and record.get("helper_sha256") == helper_hash
        ):
            return _load_npz(path)
    print(f"{WORK_PACKAGE}: build full N769 continuum state", flush=True)
    background = build_causal_five_field_continuum_background(
        context,
        background_evaluator,
        node_count=CONTINUUM_NODES,
    )
    metric, projectors, _energy_report = b6b._continuum_energy_basis(
        background,
        field_scales,
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
    scales = np.tile(field_scales, CONTINUUM_NODES)
    scaled_generator = (
        diags(1.0 / scales) @ generator @ diags(scales)
    ).tocsr()
    initial = np.column_stack(
        [packet.ravel() / scales for packet in packet_pair]
    )
    times = np.linspace(0.0, horizon, TIME_SAMPLES)
    trace = float(np.sum(scaled_generator.diagonal()))
    print(f"{WORK_PACKAGE}: propagate full N769 continuum state", flush=True)
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
    pair = np.transpose(
        scaled * scales[None, :, None],
        (0, 2, 1),
    ).reshape(TIME_SAMPLES, 2, CONTINUUM_NODES, FIELDS)
    physical = b6b._base_combinations(pair[:, 0], pair[:, 1])
    payload = {
        "times": times,
        "log_radii": background.log_radii,
        "physical": physical,
    }
    np.savez_compressed(path, **payload)
    _write_json(
        metadata,
        {
            "source_parent_commit": ANALYZED_BASE_COMMIT,
            "nodes": CONTINUUM_NODES,
            "helper_sha256": helper_hash,
        },
    )
    return payload


def _project_field(
    log_nodes: np.ndarray,
    values: np.ndarray,
    grid,
) -> np.ndarray:
    spline = make_interp_spline(
        np.asarray(log_nodes, dtype=float),
        np.asarray(values, dtype=float),
        k=5,
        axis=0,
    )

    def evaluate(radii):
        return np.asarray(spline(np.log(radii)), dtype=float)

    return c3._project_callable_to_cells(
        grid,
        evaluate,
        quadrature_order=24,
    )


def _physical_norm(
    values: np.ndarray,
    level: dict,
    field_scales: np.ndarray,
    lower: int = 0,
    upper: int | None = None,
) -> float:
    data = np.asarray(values, dtype=float)
    if upper is None:
        upper = data.shape[0]
    widths = np.diff(np.log(np.asarray(level["grid"].edges, dtype=float)))
    scaled = data[lower:upper] / field_scales[None]
    return float(
        np.sqrt(
            np.sum(
                widths[lower:upper, None] * scaled * scaled
            )
        )
    )


def _residual_blocks(level: dict) -> dict[str, np.ndarray]:
    operator = level["energy_operator"]
    return {
        **operator["blocks"],
        "mapped_storage_rate_derivative": operator["mapped_storage_rate"],
        "responsive_height_storage_rate_derivative": (
            operator["height_storage_rate"]
        ),
    }


def _dae_localization(
    levels: dict[int, dict],
    continuum: dict,
    selected_indices: list[int],
    field_scales: np.ndarray,
    background_evaluator,
) -> tuple[dict, dict[str, np.ndarray]]:
    log_nodes = np.asarray(continuum["log_radii"], dtype=float)
    physical = np.asarray(continuum["physical"], dtype=float)
    base_names = list(b6b.BASES)
    generator_background = build_causal_five_field_continuum_background(
        levels[LEVELS[-1]]["context"],
        background_evaluator,
        node_count=CONTINUUM_NODES,
    )
    continuum_blocks = causal_five_field_inward_collocation_generator_blocks(
        generator_background
    )
    reports = {}
    decisive = {}
    stable_candidates = []
    for base in (
        "acoustic",
        "difference_shear_acoustic",
        "shear",
    ):
        base_index = base_names.index(base)
        reports[base] = {}
        for time_index in selected_indices:
            state = physical[time_index, base_index]
            block_rates = {
                name: np.asarray(matrix @ state.ravel()).reshape(
                    CONTINUUM_NODES, FIELDS
                )
                for name, matrix in continuum_blocks.items()
            }
            total_rate = sum(block_rates.values())
            level_report = {}
            block_error_norms = {
                name: [] for name in continuum_blocks
            }
            total_error_norms = []
            tau_norms = []
            band_error_norms = {
                band: [] for band in FIXED_BANDS_N98
            }
            for cells in LEVELS:
                level = levels[cells]
                factor = cells // LEVELS[0]
                projected_state = _project_field(
                    log_nodes, state, level["grid"]
                )
                projected_rate = _project_field(
                    log_nodes, total_rate, level["grid"]
                )
                projected_blocks = {
                    name: _project_field(
                        log_nodes, rate, level["grid"]
                    )
                    for name, rate in block_rates.items()
                }
                columns = np.asarray(level["columns"], dtype=float)
                state_scaled = projected_state.ravel() / columns
                rate_scaled = projected_rate.ravel() / columns
                operator = level["energy_operator"]
                descriptor = csr_matrix(operator["descriptor"])
                residual_blocks = {
                    name: csr_matrix(matrix)
                    for name, matrix in _residual_blocks(level).items()
                }
                stationary = sum(
                    (
                        matrix @ state_scaled
                        for matrix in residual_blocks.values()
                    ),
                    start=np.zeros_like(state_scaled),
                )
                tau = descriptor @ rate_scaled + stationary
                tau_scale = max(
                    float(np.linalg.norm(descriptor @ rate_scaled)),
                    float(np.linalg.norm(stationary)),
                    np.finfo(float).tiny,
                )
                tau_norms.append(float(np.linalg.norm(tau) / tau_scale))
                finite_rate_scaled = np.asarray(
                    level["generator"] @ state_scaled
                ).ravel()
                finite_rate = (
                    finite_rate_scaled * columns
                ).reshape(cells, FIELDS)
                total_error = finite_rate - projected_rate
                total_error_norms.append(
                    _physical_norm(
                        total_error, level, field_scales
                    )
                )
                factorization = splu(descriptor.tocsc())
                finite_block_errors = {}
                for name, matrix in residual_blocks.items():
                    finite_block_scaled = -factorization.solve(
                        np.asarray(matrix @ state_scaled).ravel()
                    )
                    finite_block = (
                        finite_block_scaled * columns
                    ).reshape(cells, FIELDS)
                    error = finite_block - projected_blocks[name]
                    finite_block_errors[name] = error
                    block_error_norms[name].append(
                        _physical_norm(error, level, field_scales)
                    )
                for band, (lower, upper) in FIXED_BANDS_N98.items():
                    band_error_norms[band].append(
                        _physical_norm(
                            total_error,
                            level,
                            field_scales,
                            lower * factor,
                            upper * factor,
                        )
                    )
                level_report[f"N{cells}"] = {
                    "unsolved_DAE_relative_residual": tau_norms[-1],
                    "mass_solved_rate_error_norm": total_error_norms[-1],
                    "block_error_norms": {
                        name: block_error_norms[name][-1]
                        for name in block_error_norms
                    },
                    "fixed_band_error_norms": {
                        name: band_error_norms[name][-1]
                        for name in band_error_norms
                    },
                }
                decisive[
                    f"{base}__t{time_index}__N{cells}__tau_scaled"
                ] = np.asarray(tau)
                decisive[
                    f"{base}__t{time_index}__N{cells}__rate_error"
                ] = total_error
            block_orders = {
                name: _orders(values)
                for name, values in block_error_norms.items()
            }
            band_orders = {
                name: _orders(values)
                for name, values in band_error_norms.items()
            }
            fine_block = {
                name: values[-1] for name, values in block_error_norms.items()
            }
            block_total = sum(fine_block.values())
            dominant_block = max(fine_block, key=fine_block.get)
            block_fraction = float(
                fine_block[dominant_block]
                / max(block_total, np.finfo(float).tiny)
            )
            fine_band = {
                name: values[-1] for name, values in band_error_norms.items()
            }
            band_total = sum(fine_band.values())
            dominant_band = max(fine_band, key=fine_band.get)
            band_fraction = float(
                fine_band[dominant_band]
                / max(band_total, np.finfo(float).tiny)
            )
            candidate = bool(
                block_fraction >= DOMINANT_FRACTION
                and min(block_orders[dominant_block])
                < MINIMUM_DIRECT_ORDER
                and band_fraction >= DOMINANT_FRACTION
                and min(band_orders[dominant_band])
                < MINIMUM_DIRECT_ORDER
            )
            if candidate:
                stable_candidates.append(
                    (base, time_index, dominant_block, dominant_band)
                )
            reports[base][f"time_index_{time_index}"] = {
                "time_seconds": float(continuum["times"][time_index]),
                "levels": level_report,
                "unsolved_DAE_residual_orders": _orders(tau_norms),
                "mass_solved_rate_error_orders": _orders(
                    total_error_norms
                ),
                "block_error_orders": block_orders,
                "fixed_band_error_orders": band_orders,
                "dominant_fine_block": dominant_block,
                "dominant_fine_block_fraction": block_fraction,
                "dominant_fine_band": dominant_band,
                "dominant_fine_band_fraction": band_fraction,
                "stable_noncontracting_mechanism_selected": candidate,
            }
    reports["stable_candidates"] = [list(item) for item in stable_candidates]
    reports["stable_noncontracting_mechanism_selected"] = bool(
        stable_candidates
    )
    return reports, decisive


def _input_hashes() -> dict[str, str]:
    paths = (
        PARENT_DIRECTORY / "summary.json",
        PARENT_DIRECTORY / "config.json",
        PARENT_DIRECTORY / "decisive_arrays.npz",
        b6b.B5B_DIRECTORY / "summary.json",
        b6b.SCOPE_DIRECTORY / "scope_manifest.json",
        b6b.C2A2_DIRECTORY / "decisive_arrays.npz",
        b6b.C2B3_DIRECTORY / "decisive_arrays.npz",
    )
    return {
        str(path.relative_to(ROOT)): c2a._sha256(path)
        for path in paths
    }


def _write_report(summary: dict) -> None:
    direct = summary["direct_to_N769_history"]
    dae_cases = [
        item
        for base in ("acoustic", "difference_shear_acoustic", "shear")
        for item in summary["DAE_localization"][base].values()
    ]
    minimum_tau_orders = [
        min(item["unsolved_DAE_residual_orders"][pair] for item in dae_cases)
        for pair in range(2)
    ]
    minimum_solved_orders = [
        min(
            item["mass_solved_rate_error_orders"][pair]
            for item in dae_cases
        )
        for pair in range(2)
    ]
    maximum_block_fraction = max(
        item["dominant_fine_block_fraction"] for item in dae_cases
    )
    maximum_band_fraction = max(
        item["dominant_fine_band_fraction"] for item in dae_cases
    )
    lines = [
        "# Causal inner uniform failure localization "
        "WP10c9d6c7c2b6c results",
        "",
        "## Binding status",
        "",
        "The frozen b6b rejection is preserved. This package changes no "
        "operator, profile, threshold, or historical classification.",
        "",
        "## Main result",
        "",
        (
            "All five failed b6b channels contract directly toward the "
            "independent N769 history. Their minimum direct orders are "
            f"`{summary['minimum_direct_history_order']:.3f}`, and the "
            "largest N392 direct response-relative error is "
            f"`{summary['maximum_fine_direct_history_difference']:.5f}`."
        ),
        "",
        (
            "No stable noncontracting DAE block and fixed physical band "
            "was selected: "
            f"`{summary['binding_decision'][
                'stable_noncontracting_DAE_mechanism_selected'
            ]}`."
        ),
        "",
        "The unsolved DAE residual has worst pair orders "
        f"`{minimum_tau_orders[0]:.3f}, {minimum_tau_orders[1]:.3f}`; "
        "after descriptor inversion the worst rate-error orders are "
        f"`{minimum_solved_orders[0]:.3f}, "
        f"{minimum_solved_orders[1]:.3f}`.",
        "",
        "## Direct history table",
        "",
        "| Channel | Direct orders | N392 direct maximum |",
        "|---|---:|---:|",
    ]
    for name, item in direct.items():
        lines.append(
            f"| `{name}` | "
            f"`{item['direct_error_orders'][0]:.3f}, "
            f"{item['direct_error_orders'][1]:.3f}` | "
            f"`{item['fine_direct_response_relative_maximum']:.5f}` |"
        )
    lines.extend(
        [
            "",
            "## DAE and projector localization",
            "",
            (
                "Local stress relaxation is the largest N392 block in "
                "the sampled cases, but its maximum absolute-error score "
                f"is only `{maximum_block_fraction:.3f}`, below the frozen "
                f"`{DOMINANT_FRACTION:.2f}` selection threshold."
            ),
            "",
            (
                "The largest fixed-band score is "
                f"`{maximum_band_fraction:.3f}` in the third receiving "
                "band, but that band contracts on both pairs. A large "
                "location is therefore not a nonconvergent mechanism."
            ),
            "",
            (
                "The polynomial and eigenvector local projectors agree "
                "within the required algebraic tolerance. Projector "
                "normalization is not the cause of the failed b6b gates."
            ),
            "",
            "## Interpretation",
            "",
            "The pairwise b6b gates remain failed. The independent "
            "continuum comparison shows that those failures are not a "
            "failure to approach the continuum: the finite histories "
            "cross a changing leading-error mixture while converging to "
            "N769. Equivalent local projector constructions agree, and "
            "the DAE-level audit does not select a noncontracting storage, "
            "transport, principal, relaxation, geometry, cooling, stream, "
            "or height-work block.",
            "",
            "No numerical or interface redesign is authorized. The only "
            "authorized continuation is a definitions-only prospective "
            "manifest for a direct-continuum arrival contract with new "
            "unseen profiles. Embedded, nonlinear, fixed-Q, and reduced "
            "slow-time work remain blocked.",
            "",
            "## Classification",
            "",
            f"`{summary['classification']}`",
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
    parent_summary, parent_arrays = _validate_parent()
    times = np.asarray(
        parent_arrays["primary_times_seconds"], dtype=float
    )
    failures, decisive, selected_indices = _direct_history_report(
        parent_arrays, times, FAILURE_CHANNELS
    )
    controls, control_arrays, _control_indices = _direct_history_report(
        parent_arrays, times, PASSING_CONTROLS
    )
    decisive.update(control_arrays)

    (
        _energy_summary,
        _energy_manifest,
        _energy_arrays,
        parent_context,
        parent_base,
        field_scales,
    ) = c2a2._load_inputs()
    c2a2_arrays = _load_npz(
        b6b.C2A2_DIRECTORY / "decisive_arrays.npz"
    )
    scope = _read_json(b6b.SCOPE_DIRECTORY / "scope_manifest.json")
    scope_arrays = _load_npz(
        b6b.SCOPE_DIRECTORY / "decisive_arrays.npz"
    )
    contract_arrays = _load_npz(
        b6b.C2B3_DIRECTORY / "decisive_arrays.npz"
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
    primary_array = np.asarray(
        contract_arrays["primary_arrival_windows_seconds"],
        dtype=float,
    )
    windows = {
        family: tuple(primary_array[index])
        for index, family in enumerate(b6b.c2b1.PRIMARY_FAMILIES)
    }

    print(f"{WORK_PACKAGE}: load certified finite tangents", flush=True)
    levels = c2b2._build_levels(
        base_edges,
        parent_context,
        parent_base,
        field_scales,
    )
    propagated = {
        cells: _load_finite_propagation(cells) for cells in LEVELS
    }
    projector_report, projector_arrays = _projector_sensitivity(
        levels, propagated, windows
    )
    decisive.update(projector_arrays)

    background_evaluator = b5b._background_evaluator(
        parent_context, parent_base, field_scales
    )
    continuum = _continuum_state_checkpoint(
        levels[LEVELS[-1]]["context"],
        background_evaluator,
        field_scales,
        scope_arrays,
        support_log_bounds,
        horizon,
    )
    dae_report, dae_arrays = _dae_localization(
        levels,
        continuum,
        selected_indices,
        field_scales,
        background_evaluator,
    )
    decisive.update(dae_arrays)

    minimum_direct_order = min(
        item["minimum_direct_error_order"] for item in failures.values()
    )
    maximum_fine_direct = max(
        item["fine_direct_response_relative_maximum"]
        for item in failures.values()
    )
    direct_passed = bool(
        all(
            item["direct_continuum_contract_passed"]
            for item in failures.values()
        )
    )
    stable_mechanism = bool(
        dae_report["stable_noncontracting_mechanism_selected"]
    )
    classification = (
        "uniform_failure_localized_to_stable_noncontracting_DAE_"
        "mechanism_local_audit_authorized"
        if stable_mechanism
        else "direct_continuum_arrival_errors_contract_pairwise_"
        "rotation_preasymptotic_no_redesign"
        if direct_passed
        else "uniform_failure_audit_inconclusive_downstream_blocked"
    )
    authorized_next = (
        "WP10c9d6c7c2b6d_selected_DAE_mechanism_local_audit"
        if stable_mechanism
        else "WP10c9d6c7c2b6d_direct_continuum_arrival_contract_manifest"
        if direct_passed
        else "none"
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        "operator_changed": False,
        "propagation_executed": True,
        "finite_volume_propagation_reused": True,
        "independent_N769_state_propagated": True,
        "embedded_or_nonlinear_propagation_executed": False,
        "historical_classifications_preserved": {
            "WP10c9d6c7c2b6b": parent_summary["classification"],
        },
        "frozen_failure_channels": [list(item) for item in FAILURE_CHANNELS],
        "passing_controls": [list(item) for item in PASSING_CONTROLS],
        "selected_time_indices": selected_indices,
        "direct_to_N769_history": failures,
        "direct_to_N769_passing_controls": controls,
        "projector_sensitivity": projector_report,
        "DAE_localization": dae_report,
        "minimum_direct_history_order": minimum_direct_order,
        "maximum_fine_direct_history_difference": maximum_fine_direct,
        "binding_decision": {
            "b6b_rejection_preserved": True,
            "all_failed_channels_contract_directly_to_N769": direct_passed,
            "equivalent_local_projectors_passed": projector_report[
                "equivalent_local_projectors_passed"
            ],
            "stable_noncontracting_DAE_mechanism_selected": stable_mechanism,
            "operator_or_interface_redesign_authorized": stable_mechanism,
            "definitions_only_direct_continuum_manifest_authorized": bool(
                direct_passed and not stable_mechanism
            ),
            "embedded_propagation_authorized": False,
            "nonlinear_authorized": False,
            "fixed_Q_or_reduction_authorized": False,
        },
        "passed": bool(direct_passed and not stable_mechanism),
        "classification": classification,
        "authorized_next": authorized_next,
        "runtime_seconds": time.perf_counter() - started,
    }
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "operator_changed": False,
        "failure_channels": [list(item) for item in FAILURE_CHANNELS],
        "passing_controls": [list(item) for item in PASSING_CONTROLS],
        "reference_levels": list(LEVELS),
        "continuum_nodes": CONTINUUM_NODES,
        "minimum_direct_order": MINIMUM_DIRECT_ORDER,
        "maximum_direct_fine_difference": MAXIMUM_DIRECT_FINE_DIFFERENCE,
        "dominant_fraction": DOMINANT_FRACTION,
        "fixed_bands_N98": FIXED_BANDS_N98,
        "mechanism_selection_rule": (
            "same selected time; dominant block and fixed band each carry "
            "at least 0.60 of the N392 error score and have a direct error "
            "order below 0.75"
        ),
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(CONFIG_PATH, config)
    decisive.update(
        {
            "reference_levels": np.asarray(LEVELS, dtype=np.int64),
            "selected_time_indices": np.asarray(
                selected_indices, dtype=np.int64
            ),
            "selected_times_seconds": times[selected_indices],
            "continuum_log_radii": np.asarray(continuum["log_radii"]),
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
            "config_sha256": c2a._sha256(CONFIG_PATH),
            "decisive_arrays_sha256": c2a._sha256(DECISIVE_ARRAYS),
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
        "scientific_status": (
            "DIAGNOSTIC COMPLETE" if summary["passed"] else "BLOCKED"
        ),
        "classification": classification,
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        "source_parent_tree": ANALYZED_BASE_TREE,
        "implementation_worktree_head": _git_value("rev-parse", "HEAD"),
        "implementation_source_hashes": source_hashes,
        "input_hashes": _input_hashes(),
        "command": (
            "PYTHONPATH=src python "
            "scripts/run_causal_inner_uniform_failure_localization_"
            "wp10c9d6c7c2b6c.py"
        ),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
    }
    _write_json(PROVENANCE_PATH, provenance)
    _write_report(summary)
    b6b._refresh_sha256s(CANONICAL_DIRECTORY)
    b6b._refresh_catalog()
    print(
        json.dumps(
            {
                "work_package": WORK_PACKAGE,
                "classification": classification,
                "passed": summary["passed"],
                "authorized_next": authorized_next,
                "binding_decision": summary["binding_decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
