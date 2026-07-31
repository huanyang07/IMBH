#!/usr/bin/env python3
"""Run the frozen c2c6 two-route embedded recertification."""

from __future__ import annotations

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


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_causal_inner_direct_continuum_contract_manifest_wp10c9d6c7c2b6d as b6d  # noqa: E402
import run_causal_inner_direct_continuum_embedded_discrimination_wp10c9d6c7c2c3 as c2c3  # noqa: E402
import run_causal_inner_direct_continuum_embedded_recertification_manifest_wp10c9d6c7c2c5 as c2c5  # noqa: E402
import run_causal_inner_embedded_cumulative_flux_diagnostic_wp10c9d6c7c2c4 as c2c4  # noqa: E402
import run_causal_inner_direct_continuum_uniform_recertification_wp10c9d6c7c2b6e as b6e  # noqa: E402
import run_causal_inner_scattering_energy_wp10c9d6c7c2a2 as c2a2  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_canonical_json_sha256,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_validation import (  # noqa: E402
    causal_packet_history_metrics,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_windowed_contract import (  # noqa: E402
    causal_trapezoid_weights,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c2c6"
ANALYZED_BASE_COMMIT = "9ec5997536a5c497b95a15500bd2f8f533234031"
ANALYZED_BASE_PARENT = "f2e90efbf3c5a2293d7ed7455609fc1bdbba4c31"
ANALYZED_BASE_TREE = "6e29245474c27fe64a627476b0e5cff0ea77a2e6"

PROFILES = c2c5.PROFILES
LABELS = c2c3.LABELS
OBSERVABLE_NAMES = c2c3.OBSERVABLE_NAMES
BOUNDARY_COMPONENTS = c2c5.BOUNDARY_FLUX_COMPONENTS
RELATIVE_ACTIVITY = c2c3.RELATIVE_ACTIVITY

THIS_RUNNER = (
    "scripts/"
    "run_causal_inner_direct_continuum_embedded_"
    "recertification_wp10c9d6c7c2c6.py"
)
THIS_TEST = (
    "tests/"
    "test_causal_inner_direct_continuum_embedded_"
    "recertification_wp10c9d6c7c2c6.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_DIRECT_CONTINUUM_EMBEDDED_RECERTIFICATION_"
    "WP10C9D6C7C2C6_RESULTS_2026-07-31.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
PARENT_DIRECTORY = c2c5.CANONICAL_DIRECTORY
CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_direct_continuum_embedded_recertification_"
    "wp10c9d6c7c2c6"
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
    return float(np.sqrt(np.sum(temporal * history**2) / np.sum(temporal)))


def _validate_parent() -> tuple[dict, dict]:
    summary = _read_json(PARENT_DIRECTORY / "summary.json")
    manifest = _read_json(PARENT_DIRECTORY / "recertification_manifest.json")
    if (
        not summary["passed"]
        or summary["classification"]
        != "direct_continuum_embedded_two_route_contract_frozen_"
        "recertification_authorized"
        or summary["authorized_next"]
        != "WP10c9d6c7c2c6_direct_continuum_embedded_recertification"
        or summary["manifest_sha256"] != manifest["manifest_sha256"]
    ):
        raise RuntimeError("c2c5 authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT)
        != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c2c6 analyzed identity changed")
    return summary, manifest


def _coefficients(manifest: dict) -> dict[str, np.ndarray]:
    values = {
        item["name"]: np.asarray(
            item["acoustic_shear_coefficients"], dtype=float
        )
        for item in manifest["binding_profiles"]
    }
    if tuple(values) != PROFILES:
        raise RuntimeError("c2c6 profile order changed")
    return values


def _combine(
    acoustic: np.ndarray,
    shear: np.ndarray,
    coefficients: dict[str, np.ndarray],
    *,
    axis: int,
) -> np.ndarray:
    return np.stack(
        [
            pair[0] * acoustic + pair[1] * shear
            for pair in coefficients.values()
        ],
        axis=axis,
    )


def _cached_histories(
    coefficients: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    result = {}
    for label in LABELS:
        source = _load_npz(
            c2c3.CHECKPOINT_DIRECTORY / f"{label}_propagation.npz"
        )
        result[label] = {
            "times": source["times"],
            "parent_state": _combine(
                source["parent_state"][:, 0],
                source["parent_state"][:, 1],
                coefficients,
                axis=1,
            ),
            "signals": _combine(
                source["signals"][:, 0],
                source["signals"][:, 1],
                coefficients,
                axis=1,
            ),
            "cumulative_signals": _combine(
                source["cumulative_signals"][:, 0],
                source["cumulative_signals"][:, 1],
                coefficients,
                axis=1,
            ),
            "restart_parent_state": _combine(
                source["restart_parent_state"][0],
                source["restart_parent_state"][1],
                coefficients,
                axis=0,
            ),
        }
    return result


def _state_references(
    coefficients: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    b6d_summary = _read_json(b6d.CANONICAL_DIRECTORY / "summary.json")
    historical_coefficients = b6e._coefficients(b6d_summary)
    primary, secondary = c2c3._fixed_reference(
        historical_coefficients
    )
    return (
        _combine(primary[:, 0], primary[:, 1], coefficients, axis=1),
        _combine(
            secondary[:, 0], secondary[:, 1], coefficients, axis=1
        ),
    )


def _boundary_references(
    coefficients: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    arrays = _load_npz(c2c4.DECISIVE_ARRAYS)
    channel_index = {
        name: index for index, name in enumerate(c2c4.CHANNELS)
    }
    result = {}
    for channel in c2c4.CHANNELS:
        index = channel_index[channel]
        result[channel] = {
            "instantaneous_N513": _combine(
                arrays["N513_reference_signals"][:, 0, index],
                arrays["N513_reference_signals"][:, 1, index],
                coefficients,
                axis=1,
            ),
            "instantaneous_N769": _combine(
                arrays["N769_reference_signals"][:, 0, index],
                arrays["N769_reference_signals"][:, 1, index],
                coefficients,
                axis=1,
            ),
            "cumulative_N513": _combine(
                arrays["N513_reference_cumulative_signals"][:, 0, index],
                arrays["N513_reference_cumulative_signals"][:, 1, index],
                coefficients,
                axis=1,
            ),
            "cumulative_N769": _combine(
                arrays["N769_reference_cumulative_signals"][:, 0, index],
                arrays["N769_reference_cumulative_signals"][:, 1, index],
                coefficients,
                axis=1,
            ),
        }
    return result


def _metric_payload(metrics) -> dict:
    significant = np.asarray(metrics.significant_components, dtype=int)
    return {
        "significant_components": [
            OBSERVABLE_NAMES[index] for index in significant
        ],
        "component_orders": {
            OBSERVABLE_NAMES[index]: float(order)
            for index, order in zip(
                significant, metrics.component_orders, strict=True
            )
        },
        "observed_rms_order": metrics.observed_rms_order,
        "observed_maximum_order": metrics.observed_maximum_order,
        "minimum_significant_component_order": (
            metrics.minimum_significant_component_order
        ),
        "maximum_fine_normalized_difference": (
            metrics.maximum_fine_normalized_difference
        ),
        "history_cosine": metrics.history_cosine,
        "refinement_error_cosine": metrics.refinement_error_cosine,
        "strict_passed": bool(metrics.passed),
    }


def _global_gates_pass(metrics, contract: dict) -> bool:
    return bool(
        metrics.observed_rms_order >= contract["minimum_RMS_order"]
        and metrics.observed_maximum_order
        >= contract["minimum_maximum_order"]
        and metrics.maximum_fine_normalized_difference
        <= contract["maximum_fine_normalized_difference"]
        and metrics.history_cosine >= contract["minimum_history_cosine"]
        and metrics.refinement_error_cosine
        >= contract["minimum_refinement_error_cosine"]
    )


def _absolute_envelope(
    histories: list[np.ndarray],
    primary: np.ndarray,
    secondary: np.ndarray,
    times: np.ndarray,
    scale: float,
    route: dict,
) -> dict:
    weights = causal_trapezoid_weights(times)
    normalized = [
        np.asarray(history, dtype=float) / float(scale)
        for history in histories
    ]
    truth = np.asarray(primary, dtype=float) / float(scale)
    reference = np.asarray(secondary, dtype=float) / float(scale)
    response_rms = max(
        _weighted_norm(truth, weights), np.finfo(float).tiny
    )
    response_maximum = max(
        float(np.max(np.abs(truth))), np.finfo(float).tiny
    )
    per_level = []
    for history in normalized:
        error = history - truth
        rms = _weighted_norm(error, weights)
        maximum = float(np.max(np.abs(error)))
        per_level.append(
            {
                "fixed_scale_RMS_error": rms,
                "fixed_scale_maximum_error": maximum,
                "response_relative_RMS_error": rms / response_rms,
                "response_relative_maximum_error": (
                    maximum / response_maximum
                ),
                "continuum_history_cosine": _cosine(history, truth),
            }
        )
    uncertainty = truth - reference
    uncertainty_rms = _weighted_norm(uncertainty, weights)
    uncertainty_maximum = float(np.max(np.abs(uncertainty)))
    passed = bool(
        all(
            item["fixed_scale_RMS_error"]
            <= route["maximum_fixed_scale_RMS_error"]
            and item["fixed_scale_maximum_error"]
            <= route["maximum_fixed_scale_maximum_error"]
            and item["response_relative_RMS_error"]
            <= route["maximum_response_relative_RMS_error"]
            and item["response_relative_maximum_error"]
            <= route["maximum_response_relative_maximum_error"]
            and item["continuum_history_cosine"]
            >= route["minimum_continuum_history_cosine"]
            for item in per_level
        )
        and uncertainty_rms
        <= route["maximum_reference_uncertainty_fixed_scale"]
        and uncertainty_maximum
        <= route["maximum_reference_uncertainty_fixed_scale"]
        and uncertainty_rms / response_rms
        <= route["maximum_reference_uncertainty_response_relative"]
        and uncertainty_maximum / response_maximum
        <= route["maximum_reference_uncertainty_response_relative"]
    )
    return {
        "per_level": per_level,
        "reference_uncertainty_fixed_scale_RMS": uncertainty_rms,
        "reference_uncertainty_fixed_scale_maximum": uncertainty_maximum,
        "reference_uncertainty_response_relative_RMS": (
            uncertainty_rms / response_rms
        ),
        "reference_uncertainty_response_relative_maximum": (
            uncertainty_maximum / response_maximum
        ),
        "passed": passed,
    }


def _export_report(
    histories: dict[str, dict[str, np.ndarray]],
    boundary_references: dict[str, dict[str, np.ndarray]],
    profile_index: int,
    kind: str,
    scales: np.ndarray,
    times: np.ndarray,
    manifest: dict,
) -> dict:
    key = "signals" if kind == "instantaneous" else "cumulative_signals"
    scale_multiplier = 1.0 if kind == "instantaneous" else float(times[-1])
    values = [
        histories[label][key][:, profile_index]
        for label in LABELS
    ]
    contract = manifest["tier_I_global_contract"]
    metrics = causal_packet_history_metrics(
        *values,
        physical_scales=scales * scale_multiplier,
        relative_activity=RELATIVE_ACTIVITY,
        minimum_rms_order=contract["minimum_RMS_order"],
        minimum_maximum_order=contract["minimum_maximum_order"],
        minimum_significant_component_order=contract[
            "minimum_significant_component_order"
        ],
        maximum_fine_normalized_difference=contract[
            "maximum_fine_normalized_difference"
        ],
        minimum_history_cosine=contract["minimum_history_cosine"],
        minimum_refinement_error_cosine=contract[
            "minimum_refinement_error_cosine"
        ],
    )
    payload = _metric_payload(metrics)
    failed_components = [
        OBSERVABLE_NAMES[index]
        for index, order in zip(
            metrics.significant_components,
            metrics.component_orders,
            strict=True,
        )
        if order < contract["minimum_significant_component_order"]
    ]
    global_passed = _global_gates_pass(metrics, contract)
    alternate = {}
    route = manifest["component_order_routes"]["alternate_route"]
    for component in failed_components:
        if (
            component not in BOUNDARY_COMPONENTS
            or component not in boundary_references
        ):
            alternate[component] = {
                "passed": False,
                "reason": "component_not_eligible_or_reference_unavailable",
            }
            continue
        component_index = OBSERVABLE_NAMES.index(component)
        reference = boundary_references[component]
        alternate[component] = _absolute_envelope(
            [
                histories[label][key][:, profile_index, component_index]
                for label in LABELS
            ],
            reference[f"{kind}_N769"][:, profile_index],
            reference[f"{kind}_N513"][:, profile_index],
            times,
            scales[component_index] * scale_multiplier,
            route,
        )
    component_passed = bool(
        not failed_components
        or all(alternate[item]["passed"] for item in failed_components)
    )
    payload.update(
        {
            "failed_component_orders": failed_components,
            "alternate_route": alternate,
            "global_gates_passed": global_passed,
            "component_gate_passed": component_passed,
            "route_used": (
                "strict_pairwise_component_order"
                if not failed_components
                else "direct_continuum_absolute_envelope"
            ),
            "passed": bool(global_passed and component_passed),
        }
    )
    return payload


def _comparison(
    histories: dict[str, dict[str, np.ndarray]],
    coefficients: dict[str, np.ndarray],
    manifest: dict,
    observable_scales: np.ndarray,
    field_scales: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    times = np.asarray(histories[LABELS[0]]["times"], dtype=float)
    primary_state, secondary_state = _state_references(coefficients)
    boundary_references = _boundary_references(coefficients)
    c2c1_arrays = _load_npz(c2c3.C2C1_DIRECTORY / "decisive_arrays.npz")
    parent_edges = np.asarray(
        c2c1_arrays["parent_patch_edges"], dtype=float
    )
    (
        _energy_summary,
        _energy_manifest,
        _energy_arrays,
        parent_context,
        _parent_base,
        _loaded_scales,
    ) = c2a2._load_inputs()
    parent_grid = c2c3.make_kerr_schild_column_grid_from_edges(
        parent_edges, parent_context.grid.gravitational_radius
    )
    reports = {}
    for profile_index, profile in enumerate(PROFILES):
        instantaneous = _export_report(
            histories,
            boundary_references,
            profile_index,
            "instantaneous",
            observable_scales,
            times,
            manifest,
        )
        cumulative = _export_report(
            histories,
            boundary_references,
            profile_index,
            "cumulative",
            observable_scales,
            times,
            manifest,
        )
        state = c2c3._state_report(
            [
                histories[label]["parent_state"][:, profile_index]
                for label in LABELS
            ],
            primary_state[:, profile_index],
            secondary_state[:, profile_index],
            times,
            np.asarray(parent_grid.cell_measures, dtype=float),
            field_scales,
            manifest["tier_I_global_contract"],
        )
        reports[profile] = {
            "instantaneous_exports": instantaneous,
            "cumulative_exports": cumulative,
            "state": state,
            "passed": bool(
                instantaneous["passed"]
                and cumulative["passed"]
                and state["passed"]
            ),
        }
    failed = [name for name in PROFILES if not reports[name]["passed"]]
    decisive = {
        "times_seconds": times,
        "profile_pass_flags": np.asarray(
            [reports[name]["passed"] for name in PROFILES], dtype=np.int8
        ),
        "N769_state_endpoint": primary_state[-1],
        "N513_state_endpoint": secondary_state[-1],
        **{
            f"{label}__instantaneous_exports": histories[label]["signals"]
            for label in LABELS
        },
        **{
            f"{label}__cumulative_exports": histories[label][
                "cumulative_signals"
            ]
            for label in LABELS
        },
        **{
            f"{label}__state_endpoint": histories[label][
                "parent_state"
            ][-1]
            for label in LABELS
        },
    }
    return {
        "profile_reports": reports,
        "failed_profiles": failed,
        "all_profiles_passed": not failed,
        "passed": not failed,
    }, decisive


def _input_hashes() -> dict[str, str]:
    paths = (
        PARENT_DIRECTORY / "summary.json",
        PARENT_DIRECTORY / "recertification_manifest.json",
        PARENT_DIRECTORY / "decisive_arrays.npz",
        c2c3.CANONICAL_DIRECTORY / "summary.json",
        c2c4.CANONICAL_DIRECTORY / "summary.json",
        c2c4.DECISIVE_ARRAYS,
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
    reports = summary["comparison"]["profile_reports"]
    alternate = [
        (profile, kind, component, result)
        for profile, report in reports.items()
        for kind in ("instantaneous_exports", "cumulative_exports")
        for component, result in report[kind][
            "alternate_route"
        ].items()
    ]
    lines = [
        "# Direct-continuum embedded recertification WP10c9d6c7c2c6",
        "",
        "## Result",
        "",
        (
            "All eight unseen profiles pass the frozen two-route embedded "
            "contract."
            if summary["passed"]
            else "At least one unseen profile fails the frozen two-route "
            "embedded contract."
        ),
        "",
        f"Alternate-route uses: `{len(alternate)}`.",
        "",
    ]
    for profile, kind, component, result in alternate:
        lines.append(
            f"- `{profile}` / `{kind}` / `{component}`: "
            f"`{'PASS' if result['passed'] else 'FAIL'}`."
        )
    lines.extend(
        [
            "",
            "The c2c3 rejection remains historical and unchanged. This "
            "package certifies only the declared frozen-linear embedded "
            "class and, on a complete pass, authorizes a definitions-only "
            "bounded nonlinear manifest.",
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
    parent_summary, manifest = _validate_parent()
    coefficients = _coefficients(manifest)
    histories = _cached_histories(coefficients)
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
    comparison, decisive = _comparison(
        histories,
        coefficients,
        manifest,
        observable_scales,
        field_scales,
    )
    c2c4_summary = _read_json(c2c4.SUMMARY_PATH)
    reference_solve_passed = bool(
        c2c4_summary[
            "maximum_reference_exact_integral_solve_residual"
        ]
        <= manifest["component_order_routes"]["alternate_route"][
            "maximum_exact_integral_solve_residual"
        ]
    )
    passed = bool(comparison["passed"] and reference_solve_passed)
    classification = (
        "direct_continuum_embedded_linear_class_certified_"
        "bounded_nonlinear_manifest_authorized"
        if passed
        else "direct_continuum_embedded_recertification_failed_"
        "nonlinear_blocked"
    )
    authorized_next = (
        "WP10c9d6c7c3a_bounded_nonlinear_contract_manifest"
        if passed
        else "freeze_and_diagnose_failed_c2c6_profile"
    )
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "operator_changed": False,
        "new_state_propagation_executed": False,
        "cached_basis_recombination_executed": True,
        "profiles": list(PROFILES),
        "variant_count": parent_summary["variant_count"],
        "manifest_sha256": manifest["manifest_sha256"],
        "tier_I_global_contract": manifest["tier_I_global_contract"],
        "component_order_routes": manifest["component_order_routes"],
    }
    _write_json(CONFIG_PATH, config)
    decisive.update(
        {
            "field_scales": field_scales,
            "observable_scales": observable_scales,
            "acoustic_shear_coefficients": np.stack(
                tuple(coefficients.values())
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
        "cached_basis_recombination_executed": True,
        "historical_c2c3_rejection_preserved": (
            parent_summary[
                "historical_c2c3_classification_preserved"
            ]
        ),
        "comparison": comparison,
        "reference_exact_integral_solve_passed": (
            reference_solve_passed
        ),
        "maximum_reference_exact_integral_solve_residual": (
            c2c4_summary[
                "maximum_reference_exact_integral_solve_residual"
            ]
        ),
        "maximum_sign_amplitude_scaling_defect": 0.0,
        "binding_decision": {
            "declared_frozen_linear_embedded_class_certified": passed,
            "definitions_only_bounded_nonlinear_manifest_authorized": (
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
            "scientific_status": (
                "SUPPORTED BUT NOT FULLY CERTIFIED"
                if passed
                else "DIAGNOSTIC ONLY"
            ),
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
                "run_causal_inner_direct_continuum_embedded_"
                "recertification_wp10c9d6c7c2c6.py"
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
                "failed_profiles": summary["comparison"][
                    "failed_profiles"
                ],
                "binding_decision": summary["binding_decision"],
                "authorized_next": summary["authorized_next"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
