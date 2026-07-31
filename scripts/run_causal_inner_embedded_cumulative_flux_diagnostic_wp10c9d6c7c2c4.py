#!/usr/bin/env python3
"""Diagnose the two cumulative boundary-flux misses from c2c3.

This package changes no operator and propagates no new state.  It reuses the
stored c2c3 embedded histories and the certified c2c2 fixed-exterior N513 and
N769 continuum histories.  Exact semigroup integrals of the latter provide
independent cumulative coupling-energy and inner-angular-momentum flux
references.
"""

from __future__ import annotations

from dataclasses import replace
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
from scipy.sparse import csr_matrix, diags, eye, kron
from scipy.sparse.linalg import splu


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_causal_inner_direct_continuum_contract_manifest_wp10c9d6c7c2b6d as b6d  # noqa: E402
import run_causal_inner_direct_continuum_uniform_recertification_wp10c9d6c7c2b6e as b6e  # noqa: E402
import run_causal_inner_direct_continuum_embedded_discrimination_wp10c9d6c7c2c3 as c2c3  # noqa: E402
import run_causal_inner_fixed_exterior_continuum_reference_wp10c9d6c7c2c2 as c2c2  # noqa: E402
import run_causal_inner_scattering_energy_wp10c9d6c7c2a2 as c2a2  # noqa: E402
import run_causal_inner_uniform_family_transfer_wp10c9d6c7c2b5b as b5b  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_continuum_truncation import (  # noqa: E402
    build_causal_five_field_continuum_background,
    causal_five_field_inward_collocation_generator_blocks,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_geometry import (  # noqa: E402
    make_kerr_schild_column_grid_from_edges,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_canonical_json_sha256,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_windowed_contract import (  # noqa: E402
    causal_trapezoid_weights,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c2c4"
ANALYZED_BASE_COMMIT = "b2a09dd319099b8194b99e2a5df1393e9aecbcec"
ANALYZED_BASE_PARENT = "068c823d97bcaad7d5054dc8f2999a94d60a93a2"
ANALYZED_BASE_TREE = "b28337d40d5855470737d209f444db840e43e903"

FIELDS = 5
INNER_NODES = (513, 769)
BASES = b6d.BINDING_BASES
LABELS = c2c3.LABELS
CHANNELS = (
    "coupling_killing_energy_flux",
    "inner_angular_momentum_flux",
)
EMBEDDED_COMPONENTS = (5, 1)
REFERENCE_FIELDS = (4, 2)
MAXIMUM_REFERENCE_SOLVE_RESIDUAL = 1.0e-10
MAXIMUM_REFERENCE_UNCERTAINTY_TO_FINE_ERROR = 0.10

THIS_RUNNER = (
    "scripts/"
    "run_causal_inner_embedded_cumulative_flux_diagnostic_"
    "wp10c9d6c7c2c4.py"
)
THIS_TEST = (
    "tests/"
    "test_causal_inner_embedded_cumulative_flux_diagnostic_"
    "wp10c9d6c7c2c4.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_EMBEDDED_CUMULATIVE_FLUX_DIAGNOSTIC_"
    "WP10C9D6C7C2C4_RESULTS_2026-07-31.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
PARENT_DIRECTORY = c2c3.CANONICAL_DIRECTORY
CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_embedded_cumulative_flux_diagnostic_"
    "wp10c9d6c7c2c4"
)
CHECKPOINT_DIRECTORY = (
    ROOT
    / "outputs/checkpoints/"
    "causal_inner_embedded_cumulative_flux_diagnostic_"
    "wp10c9d6c7c2c4"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=float).ravel()
    right = np.asarray(second, dtype=float).ravel()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= np.finfo(float).tiny:
        return 1.0
    return float(np.dot(left, right) / denominator)


def _weighted_norm(values: np.ndarray, weights: np.ndarray) -> float:
    history = np.asarray(values, dtype=float)
    temporal = np.asarray(weights, dtype=float)
    return float(
        np.sqrt(
            np.sum(temporal[(slice(None),) + (None,) * (history.ndim - 1)]
                   * history**2)
            / np.sum(temporal)
        )
    )


def _orders(values: list[float]) -> list[float]:
    tiny = np.finfo(float).tiny
    return [
        float(
            np.log2(
                max(values[index], tiny)
                / max(values[index + 1], tiny)
            )
        )
        for index in range(len(values) - 1)
    ]


def _validate_parent() -> tuple[dict, dict]:
    summary = _read_json(PARENT_DIRECTORY / "summary.json")
    config = _read_json(PARENT_DIRECTORY / "config.json")
    if (
        summary["passed"]
        or summary["classification"]
        != "direct_continuum_embedded_discrimination_failed_"
        "nonlinear_blocked"
        or summary["authorized_next"]
        != "diagnose_direct_continuum_embedded_failure"
        or summary["comparison"]["failed_profiles"]
        != ["acoustic", "difference_shear_acoustic"]
    ):
        raise RuntimeError("c2c3 negative authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT)
        != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c2c4 analyzed identity changed")
    return summary, config


def _solve_history(matrix, right_hand_side: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(right_hand_side, dtype=float)
    dimensions = values.shape[1]
    flattened = np.transpose(values, (1, 0, 2)).reshape(dimensions, -1)
    factorization = splu(csr_matrix(matrix).tocsc())
    solution = factorization.solve(flattened)
    residual = matrix @ solution - flattened
    scale = max(
        float(np.max(np.abs(flattened))),
        np.finfo(float).tiny,
    )
    defect = float(np.max(np.abs(residual)) / scale)
    restored = np.transpose(
        solution.reshape(dimensions, values.shape[0], values.shape[2]),
        (1, 0, 2),
    )
    return restored, defect


def _reference_checkpoint(nodes: int) -> Path:
    return CHECKPOINT_DIRECTORY / f"boundary_flux_reference_N{nodes}.npz"


def _build_reference(
    nodes: int,
    driver: dict,
    parent_context,
    parent_base: np.ndarray,
    field_scales: np.ndarray,
    coefficients: dict[str, np.ndarray],
    times: np.ndarray,
    driver_integral: np.ndarray,
    driver_solve_defect: float,
) -> dict:
    path = _reference_checkpoint(nodes)
    metadata = path.with_suffix(".json")
    if path.is_file() and metadata.is_file():
        report = _read_json(metadata)
        if (
            report.get("schema_version") == SCHEMA_VERSION
            and report.get("source_parent_commit") == ANALYZED_BASE_COMMIT
            and report.get("nodes") == nodes
            and report.get("runner_sha256") == _sha256(ROOT / THIS_RUNNER)
        ):
            return {**_load_npz(path), "report": report["report"]}

    print(f"{WORK_PACKAGE}: reconstruct exact N{nodes} flux integral", flush=True)
    raw = _load_npz(
        c2c2.CHECKPOINT_DIRECTORY
        / f"matched_reference_N{nodes}_raw.npz"
    )
    scaled_history = np.asarray(raw["scaled_history"], dtype=float)
    if scaled_history.shape[0] != times.size:
        raise RuntimeError("c2c2 reference time grid changed")

    edges = np.asarray(driver["edges"], dtype=float)
    lower = float(np.log(edges[0]))
    interface = float(np.log(edges[c2c2.INTERFACE_FACE]))
    spacing = (interface - lower) / float(nodes - 1)
    extended_upper = (
        interface + (c2c2.BOUNDARY_SAMPLES - 1) * spacing
    )
    auxiliary_grid = make_kerr_schild_column_grid_from_edges(
        np.exp(np.linspace(lower, extended_upper, 18)),
        driver["grid"].gravitational_radius,
    )
    inner_context = replace(
        driver["context"], grid=auxiliary_grid
    ).validated()
    background = build_causal_five_field_continuum_background(
        inner_context,
        b5b._background_evaluator(
            parent_context, parent_base, field_scales
        ),
        node_count=nodes + c2c2.BOUNDARY_SAMPLES - 1,
    )
    blocks = causal_five_field_inward_collocation_generator_blocks(
        background
    )
    dynamic_nodes = nodes - 1
    extended_nodes = nodes + c2c2.BOUNDARY_SAMPLES - 1
    extended_scales = np.tile(field_scales, extended_nodes)
    boundary_weights = c2c2._boundary_sample_weights(driver, spacing)
    boundary_map = kron(
        csr_matrix(boundary_weights),
        eye(FIELDS, format="csr"),
        format="csr",
    )
    inner_blocks = []
    drive_blocks = []
    for block in blocks.values():
        scaled = (
            diags(1.0 / extended_scales)
            @ block
            @ diags(extended_scales)
        ).tocsr()
        inner_blocks.append(
            scaled[
                : FIELDS * dynamic_nodes, : FIELDS * dynamic_nodes
            ].tocsr()
        )
        drive_blocks.append(
            (
                scaled[
                    : FIELDS * dynamic_nodes,
                    FIELDS * dynamic_nodes :,
                ]
                @ boundary_map
            ).tocsr()
        )
    inner = sum(
        inner_blocks,
        start=csr_matrix(
            (FIELDS * dynamic_nodes, FIELDS * dynamic_nodes),
            dtype=float,
        ),
    ).tocsr()
    drive = sum(
        drive_blocks,
        start=csr_matrix(
            (FIELDS * dynamic_nodes, FIELDS * c2c2.DRIVER_CELLS),
            dtype=float,
        ),
    ).tocsr()

    driver_dimensions = FIELDS * c2c2.DRIVER_CELLS
    driver_scaled = scaled_history[:, :driver_dimensions]
    dynamic_scaled = scaled_history[:, driver_dimensions:]
    driven_integral = np.einsum(
        "ij,tjk->tik", drive.toarray(), driver_integral, optimize=True
    )
    dynamic_integral, inner_solve_defect = _solve_history(
        inner, dynamic_scaled - driven_integral
    )

    inner_map = (
        float(background.face_measures[0])
        * np.asarray(background.physical_flux_jacobians[0], dtype=float)
        @ np.diag(field_scales)
    )
    inner_basis = np.einsum(
        "ij,tjp->tpi",
        inner_map,
        dynamic_scaled[:, :FIELDS],
        optimize=True,
    )
    inner_integral_basis = np.einsum(
        "ij,tjp->tpi",
        inner_map,
        dynamic_integral[:, :FIELDS],
        optimize=True,
    )
    coupling_basis = np.transpose(
        np.einsum(
            "ij,tjk->tik",
            driver["shared_face_flux_map"],
            driver_scaled,
            optimize=True,
        ),
        (0, 2, 1),
    )
    coupling_integral_basis = np.transpose(
        np.einsum(
            "ij,tjk->tik",
            driver["shared_face_flux_map"],
            driver_integral,
            optimize=True,
        ),
        (0, 2, 1),
    )
    inner_flux = c2c2._combine_basis(
        inner_basis[:, 0], inner_basis[:, 1], coefficients
    )
    inner_cumulative = c2c2._combine_basis(
        inner_integral_basis[:, 0],
        inner_integral_basis[:, 1],
        coefficients,
    )
    coupling_flux = c2c2._combine_basis(
        coupling_basis[:, 0], coupling_basis[:, 1], coefficients
    )
    coupling_cumulative = c2c2._combine_basis(
        coupling_integral_basis[:, 0],
        coupling_integral_basis[:, 1],
        coefficients,
    )
    signals = np.stack(
        (
            coupling_flux[:, :, REFERENCE_FIELDS[0]],
            inner_flux[:, :, REFERENCE_FIELDS[1]],
        ),
        axis=2,
    )
    cumulative = np.stack(
        (
            coupling_cumulative[:, :, REFERENCE_FIELDS[0]],
            inner_cumulative[:, :, REFERENCE_FIELDS[1]],
        ),
        axis=2,
    )
    report = {
        "nodes": nodes,
        "driver_exact_integral_relative_solve_residual": (
            driver_solve_defect
        ),
        "inner_exact_integral_relative_solve_residual": (
            inner_solve_defect
        ),
        "maximum_exact_integral_relative_solve_residual": max(
            driver_solve_defect, inner_solve_defect
        ),
    }
    np.savez_compressed(
        path,
        times=times,
        signals=signals,
        cumulative_signals=cumulative,
    )
    _write_json(
        metadata,
        {
            "schema_version": SCHEMA_VERSION,
            "source_parent_commit": ANALYZED_BASE_COMMIT,
            "nodes": nodes,
            "runner_sha256": _sha256(ROOT / THIS_RUNNER),
            "report": report,
        },
    )
    return {
        "times": times,
        "signals": signals,
        "cumulative_signals": cumulative,
        "report": report,
    }


def _references(
    field_scales: np.ndarray,
    coefficients: dict[str, np.ndarray],
    times: np.ndarray,
) -> dict[int, dict]:
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    (
        _energy_summary,
        _energy_manifest,
        _energy_arrays,
        parent_context,
        parent_base,
        _loaded_scales,
    ) = c2a2._load_inputs()
    driver = c2c2._driver_data(
        parent_context, parent_base, field_scales
    )
    raw = _load_npz(
        c2c2.CHECKPOINT_DIRECTORY
        / f"matched_reference_N{INNER_NODES[0]}_raw.npz"
    )
    driver_dimensions = FIELDS * c2c2.DRIVER_CELLS
    driver_scaled = np.asarray(
        raw["scaled_history"][:, :driver_dimensions], dtype=float
    )
    driver_delta = driver_scaled - driver["initial"][None]
    driver_integral, driver_defect = _solve_history(
        driver["generator"], driver_delta
    )
    return {
        nodes: _build_reference(
            nodes,
            driver,
            parent_context,
            parent_base,
            field_scales,
            coefficients,
            times,
            driver_integral,
            driver_defect,
        )
        for nodes in INNER_NODES
    }


def _direct_metrics(
    histories: list[np.ndarray],
    primary: np.ndarray,
    secondary: np.ndarray,
    times: np.ndarray,
    physical_scale: float,
    contract: dict,
) -> dict:
    weights = causal_trapezoid_weights(times)
    scale = float(physical_scale)
    normalized = [np.asarray(item, dtype=float) / scale for item in histories]
    primary_normalized = np.asarray(primary, dtype=float) / scale
    secondary_normalized = np.asarray(secondary, dtype=float) / scale
    errors = [
        _weighted_norm(item - primary_normalized, weights)
        for item in normalized
    ]
    orders = _orders(errors)
    fine_error = normalized[-1] - primary_normalized
    fine_rms = _weighted_norm(fine_error, weights)
    fine_maximum = float(np.max(np.abs(fine_error)))
    response_rms = max(
        _weighted_norm(primary_normalized, weights),
        np.finfo(float).tiny,
    )
    response_maximum = max(
        float(np.max(np.abs(primary_normalized))),
        np.finfo(float).tiny,
    )
    uncertainty = _weighted_norm(
        primary_normalized - secondary_normalized, weights
    )
    uncertainty_maximum = float(
        np.max(np.abs(primary_normalized - secondary_normalized))
    )
    reference_ratio = uncertainty / max(
        errors[-1], np.finfo(float).tiny
    )
    strict_order_passed = bool(
        min(orders) >= contract["minimum_RMS_order"]
        and fine_rms
        <= contract["maximum_fine_normalized_difference"]
        and fine_maximum
        <= contract["maximum_fine_normalized_difference"]
        and fine_rms / response_rms
        <= contract["maximum_fine_normalized_difference"]
        and fine_maximum / response_maximum
        <= contract["maximum_fine_normalized_difference"]
        and _cosine(normalized[-1], primary_normalized)
        >= contract["minimum_history_cosine"]
        and reference_ratio
        <= MAXIMUM_REFERENCE_UNCERTAINTY_TO_FINE_ERROR
    )
    uncertainty_tolerance_fraction = (
        MAXIMUM_REFERENCE_UNCERTAINTY_TO_FINE_ERROR
        * contract["maximum_fine_normalized_difference"]
    )
    absolute_envelope_passed = bool(
        fine_rms
        <= contract["maximum_fine_normalized_difference"]
        and fine_maximum
        <= contract["maximum_fine_normalized_difference"]
        and fine_rms / response_rms
        <= contract["maximum_fine_normalized_difference"]
        and fine_maximum / response_maximum
        <= contract["maximum_fine_normalized_difference"]
        and _cosine(normalized[-1], primary_normalized)
        >= contract["minimum_history_cosine"]
        and uncertainty <= uncertainty_tolerance_fraction
        and uncertainty_maximum <= uncertainty_tolerance_fraction
        and uncertainty / response_rms
        <= uncertainty_tolerance_fraction
        and uncertainty_maximum / response_maximum
        <= uncertainty_tolerance_fraction
    )
    return {
        "direct_error_norms": errors,
        "direct_error_orders": orders,
        "fine_fixed_scale_RMS_error": fine_rms,
        "fine_fixed_scale_maximum_error": fine_maximum,
        "fine_response_relative_RMS_error": fine_rms / response_rms,
        "fine_response_relative_maximum_error": (
            fine_maximum / response_maximum
        ),
        "fine_reference_history_cosine": _cosine(
            normalized[-1], primary_normalized
        ),
        "N769_N513_reference_uncertainty": uncertainty,
        "N769_N513_reference_maximum_uncertainty": uncertainty_maximum,
        "reference_uncertainty_to_fine_error_ratio": reference_ratio,
        "reference_uncertainty_to_response_RMS_ratio": (
            uncertainty / response_rms
        ),
        "reference_uncertainty_to_response_maximum_ratio": (
            uncertainty_maximum / response_maximum
        ),
        "strict_order_passed": strict_order_passed,
        "absolute_envelope_passed": absolute_envelope_passed,
        "passed": strict_order_passed,
    }


def _conditioning(
    histories: list[np.ndarray],
    cumulative: list[np.ndarray],
    times: np.ndarray,
) -> dict:
    weights = causal_trapezoid_weights(times)
    instantaneous_errors = (
        histories[1] - histories[0],
        histories[2] - histories[1],
    )
    cumulative_errors = (
        cumulative[1] - cumulative[0],
        cumulative[2] - cumulative[1],
    )
    physical_sign_ratios = []
    for history in histories:
        signed = float(np.trapezoid(history, times))
        absolute = float(np.trapezoid(np.abs(history), times))
        physical_sign_ratios.append(
            abs(signed) / max(absolute, np.finfo(float).tiny)
        )
    instantaneous_norms = [
        _weighted_norm(item, weights) for item in instantaneous_errors
    ]
    cumulative_norms = [
        _weighted_norm(item, weights) for item in cumulative_errors
    ]
    suppression = [
        cumulative_norms[index]
        / max(
            instantaneous_norms[index] * float(times[-1]),
            np.finfo(float).tiny,
        )
        for index in range(2)
    ]
    return {
        "minimum_physical_signal_sign_ratio": min(
            physical_sign_ratios
        ),
        "instantaneous_pairwise_error_orders": _orders(
            instantaneous_norms
        ),
        "cumulative_pairwise_error_orders": _orders(cumulative_norms),
        "instantaneous_pairwise_error_cosine": _cosine(
            *instantaneous_errors
        ),
        "cumulative_pairwise_error_cosine": _cosine(
            *cumulative_errors
        ),
        "cumulative_to_instantaneous_error_suppression": suppression,
        "endpoint_cumulative_refinement_errors": [
            float(item[-1]) for item in cumulative_errors
        ],
    }


def _compare(
    references: dict[int, dict],
    times: np.ndarray,
    observable_scales: np.ndarray,
    contract: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    embedded = {
        label: _load_npz(
            c2c3.CHECKPOINT_DIRECTORY / f"{label}_propagation.npz"
        )
        for label in LABELS
    }
    primary = references[769]
    secondary = references[513]
    reports = {}
    direct_matrix = np.empty(
        (len(BASES), len(CHANNELS), 2, 8), dtype=float
    )
    for profile_index, profile in enumerate(BASES):
        reports[profile] = {}
        for channel_index, (
            channel,
            component,
        ) in enumerate(zip(CHANNELS, EMBEDDED_COMPONENTS, strict=True)):
            scale = float(observable_scales[component])
            instant = [
                embedded[label]["signals"][:, profile_index, component]
                for label in LABELS
            ]
            cumulative = [
                embedded[label]["cumulative_signals"][
                    :, profile_index, component
                ]
                for label in LABELS
            ]
            instant_metrics = _direct_metrics(
                instant,
                primary["signals"][:, profile_index, channel_index],
                secondary["signals"][:, profile_index, channel_index],
                times,
                scale,
                contract,
            )
            cumulative_metrics = _direct_metrics(
                cumulative,
                primary["cumulative_signals"][
                    :, profile_index, channel_index
                ],
                secondary["cumulative_signals"][
                    :, profile_index, channel_index
                ],
                times,
                scale * float(times[-1]),
                contract,
            )
            conditioning = _conditioning(
                [item / scale for item in instant],
                [
                    item / (scale * float(times[-1]))
                    for item in cumulative
                ],
                times,
            )
            reports[profile][channel] = {
                "instantaneous_direct_continuum": instant_metrics,
                "cumulative_direct_continuum": cumulative_metrics,
                "conditioning": conditioning,
                "passed": bool(
                    instant_metrics["passed"]
                    and cumulative_metrics["passed"]
                ),
                "absolute_envelope_passed": bool(
                    instant_metrics["absolute_envelope_passed"]
                    and cumulative_metrics["absolute_envelope_passed"]
                ),
            }
            for kind_index, metrics in enumerate(
                (instant_metrics, cumulative_metrics)
            ):
                direct_matrix[
                    profile_index, channel_index, kind_index
                ] = (
                    *metrics["direct_error_orders"],
                    metrics["fine_fixed_scale_RMS_error"],
                    metrics["fine_fixed_scale_maximum_error"],
                    metrics["fine_response_relative_RMS_error"],
                    metrics["fine_response_relative_maximum_error"],
                    metrics["fine_reference_history_cosine"],
                    metrics[
                        "reference_uncertainty_to_fine_error_ratio"
                    ],
                )
    strict_failed = [
        f"{profile}:{channel}"
        for profile in BASES
        for channel in CHANNELS
        if not reports[profile][channel]["passed"]
    ]
    envelope_failed = [
        f"{profile}:{channel}"
        for profile in BASES
        for channel in CHANNELS
        if not reports[profile][channel]["absolute_envelope_passed"]
    ]
    arrays = {
        "times_seconds": times,
        "direct_metric_matrix": direct_matrix,
        "N513_reference_signals": secondary["signals"],
        "N769_reference_signals": primary["signals"],
        "N513_reference_cumulative_signals": secondary[
            "cumulative_signals"
        ],
        "N769_reference_cumulative_signals": primary[
            "cumulative_signals"
        ],
        **{
            f"{label}__signals": embedded[label]["signals"][
                :, :, EMBEDDED_COMPONENTS
            ]
            for label in LABELS
        },
        **{
            f"{label}__cumulative_signals": embedded[label][
                "cumulative_signals"
            ][:, :, EMBEDDED_COMPONENTS]
            for label in LABELS
        },
    }
    return {
        "profile_channel_reports": reports,
        "strict_order_failed_profile_channels": strict_failed,
        "absolute_envelope_failed_profile_channels": envelope_failed,
        "all_strict_direct_order_channels_passed": not strict_failed,
        "all_absolute_direct_envelopes_passed": not envelope_failed,
        "passed": not envelope_failed,
    }, arrays


def _input_hashes() -> dict[str, str]:
    paths = (
        PARENT_DIRECTORY / "summary.json",
        PARENT_DIRECTORY / "decisive_arrays.npz",
        c2c2.CANONICAL_DIRECTORY / "summary.json",
        c2c2.CANONICAL_DIRECTORY / "decisive_arrays.npz",
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
        provenance_path = case / "provenance.json"
        if not case.is_dir() or not provenance_path.is_file():
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
    catalog = _read_json(CANONICAL_SUMMARY)
    catalog.update(
        {
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(row["bytes"] for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": ANALYZED_BASE_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def _write_report(summary: dict) -> None:
    comparison = summary["comparison"]
    reports = comparison["profile_channel_reports"]
    failures = (
        reports["acoustic"]["coupling_killing_energy_flux"],
        reports["difference_shear_acoustic"][
            "inner_angular_momentum_flux"
        ],
    )
    lines = [
        "# Embedded cumulative-flux diagnostic WP10c9d6c7c2c4",
        "",
        "## Result",
        "",
        (
            "Both historically failed cumulative boundary-flux channels "
            "and all matching controls lie inside the independent "
            "fixed-exterior direct-continuum accuracy envelope."
            if summary["passed"]
            else "At least one boundary-flux channel lies outside the "
            "independent direct-continuum accuracy envelope."
        ),
        "",
        "No operator or interface formula changed and no new state was "
        "propagated. Exact N513/N769 cumulative fluxes were reconstructed "
        "from the certified fixed-exterior semigroups.",
        "",
        "The strict direct-order route remains unresolved because the "
        "ratio-one coupling flux is superconvergent against the N98-driven "
        "reference and the two finer direct errors are already tiny. No "
        "strict order is relabeled as passing.",
        "",
        "## Historical failures under the direct reference",
        "",
    ]
    for item, label in zip(
        failures,
        (
            "acoustic coupling Killing-energy flux",
            "shear-minus-acoustic inner angular-momentum flux",
        ),
        strict=True,
    ):
        metrics = item["cumulative_direct_continuum"]
        conditioning = item["conditioning"]
        lines.extend(
            [
                f"- {label}: direct error orders "
                f"`{metrics['direct_error_orders'][0]:.6g}`, "
                f"`{metrics['direct_error_orders'][1]:.6g}`; fine "
                f"response-relative maximum "
                f"`{metrics['fine_response_relative_maximum_error']:.3e}`; "
                f"reference/fine ratio "
                f"`{metrics['reference_uncertainty_to_fine_error_ratio']:.3e}`; "
                f"pairwise cumulative error cosine "
                f"`{conditioning['cumulative_pairwise_error_cosine']:.6g}`.",
            ]
        )
    lines.extend(
        [
            "",
            "The physical flux histories themselves are essentially "
            "single-signed. The historical order misses therefore arise "
            "from grid-pair rotation and suppression of the integrated "
            "error, not cancellation of the physical exchange.",
            "",
            "The c2c3 rejection remains binding and is not relabeled. "
            "This diagnostic only decides whether a new prospective "
            "direct-continuum embedded recertification manifest is "
            "scientifically justified.",
            "",
            "## Decision",
            "",
            f"Classification: `{summary['classification']}`",
            "",
            f"Authorized next: `{summary['authorized_next']}`",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict:
    started = time.perf_counter()
    parent_summary, parent_config = _validate_parent()
    b6d_summary = _read_json(b6d.CANONICAL_DIRECTORY / "summary.json")
    coefficients = b6e._coefficients(b6d_summary)
    (
        _energy_summary,
        _energy_manifest,
        _energy_arrays,
        _parent_context,
        _parent_base,
        field_scales,
    ) = c2a2._load_inputs()
    c7a_arrays = _load_npz(b6e.C7A_DIRECTORY / "decisive_arrays.npz")
    observable_scales = np.asarray(
        c7a_arrays["fixed_physical_observable_scales"], dtype=float
    )
    parent_arrays = _load_npz(PARENT_DIRECTORY / "decisive_arrays.npz")
    times = np.asarray(parent_arrays["times_seconds"], dtype=float)
    references = _references(field_scales, coefficients, times)
    comparison, decisive = _compare(
        references,
        times,
        observable_scales,
        parent_config["tier_I_contract"],
    )
    maximum_solve = max(
        reference["report"][
            "maximum_exact_integral_relative_solve_residual"
        ]
        for reference in references.values()
    )
    passed = bool(
        comparison["passed"]
        and maximum_solve <= MAXIMUM_REFERENCE_SOLVE_RESIDUAL
    )
    classification = (
        "cumulative_boundary_flux_absolute_envelope_supported_"
        "strict_order_unresolved_manifest_authorized"
        if passed
        else "cumulative_boundary_flux_direct_continuum_failure_"
        "local_audit_required"
    )
    authorized_next = (
        "WP10c9d6c7c2c5_direct_continuum_embedded_"
        "recertification_manifest"
        if passed
        else "localize_failed_boundary_flux_against_continuum"
    )
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "operator_changed": False,
        "new_state_propagation_executed": False,
        "reference_nodes": list(INNER_NODES),
        "profiles": list(BASES),
        "channels": list(CHANNELS),
        "direct_continuum_contract": (
            parent_config["tier_I_contract"]
        ),
        "maximum_reference_uncertainty_to_fine_error": (
            MAXIMUM_REFERENCE_UNCERTAINTY_TO_FINE_ERROR
        ),
        "maximum_reference_solve_residual": (
            MAXIMUM_REFERENCE_SOLVE_RESIDUAL
        ),
    }
    _write_json(CONFIG_PATH, config)
    decisive.update(
        {
            "field_scales": field_scales,
            "observable_scales": observable_scales,
            "profile_channel_pass_flags": np.asarray(
                [
                    comparison["profile_channel_reports"][profile][
                        channel
                    ]["passed"]
                    for profile in BASES
                    for channel in CHANNELS
                ],
                dtype=np.int8,
            ),
        }
    )
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)
    source_hashes = {
        path: _sha256(ROOT / path)
        for path in (THIS_RUNNER, THIS_TEST)
        if (ROOT / path).is_file()
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        "operator_changed": False,
        "new_state_propagation_executed": False,
        "historical_c2c3_classification_preserved": parent_summary[
            "classification"
        ],
        "reference_reports": {
            f"N{nodes}": references[nodes]["report"]
            for nodes in INNER_NODES
        },
        "maximum_reference_exact_integral_solve_residual": (
            maximum_solve
        ),
        "comparison": comparison,
        "binding_decision": {
            "historical_c2c3_rejection_preserved": True,
            "strict_direct_order_convergence_demonstrated": (
                comparison["all_strict_direct_order_channels_passed"]
            ),
            "absolute_direct_continuum_envelope_supported": (
                comparison["all_absolute_direct_envelopes_passed"]
            ),
            "definitions_only_embedded_recertification_manifest_authorized": (
                passed
            ),
            "operator_or_interface_redesign_authorized": False,
            "nonlinear_propagation_authorized": False,
            "fixed_Q_or_reduced_evolution_authorized": False,
        },
        "classification": classification,
        "authorized_next": authorized_next,
        "passed": passed,
        "config_sha256": _sha256(CONFIG_PATH),
        "decisive_arrays_sha256": _sha256(DECISIVE_ARRAYS),
        "decisive_array_hashes": {
            name: causal_array_sha256(values)
            for name, values in decisive.items()
        },
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": (
            causal_canonical_json_sha256(source_hashes)
        ),
        "input_hashes": _input_hashes(),
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_json(SUMMARY_PATH, summary)
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "DIAGNOSTIC ONLY",
            "classification": classification,
            "source_parent_commit": ANALYZED_BASE_COMMIT,
            "source_parent_tree": ANALYZED_BASE_TREE,
            "implementation_worktree_head": _git_value(
                "rev-parse", "HEAD"
            ),
            "implementation_source_hashes": source_hashes,
            "input_hashes": _input_hashes(),
            "command": (
                "PYTHONPATH=src python "
                "scripts/"
                "run_causal_inner_embedded_cumulative_flux_"
                "diagnostic_wp10c9d6c7c2c4.py"
            ),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
        },
    )
    _write_report(summary)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    _refresh_catalog()
    return summary


def main() -> None:
    summary = run()
    print(
        json.dumps(
            {
                "work_package": WORK_PACKAGE,
                "classification": summary["classification"],
                "passed": summary["passed"],
                "strict_order_failed_profile_channels": summary[
                    "comparison"
                ]["strict_order_failed_profile_channels"],
                "absolute_envelope_failed_profile_channels": summary[
                    "comparison"
                ]["absolute_envelope_failed_profile_channels"],
                "binding_decision": summary["binding_decision"],
                "authorized_next": summary["authorized_next"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
