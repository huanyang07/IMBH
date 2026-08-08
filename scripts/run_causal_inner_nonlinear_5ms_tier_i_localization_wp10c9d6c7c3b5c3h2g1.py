#!/usr/bin/env python3
"""Localize the nonlinear 5 ms Tier-I spatial failure without propagation."""

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
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_nonlinear_5ms_spatial_certificate_wp10c9d6c7c3b5c3h2f as h2f  # noqa: E402
import run_causal_inner_nonlinear_5ms_tier_i_localization_manifest_wp10c9d6c7c3b5c3h2g as h2g  # noqa: E402
import run_causal_inner_nonlinear_coarse_third_duration_rung_completion_wp10c9d6c7c3b5c3d as c3d  # noqa: E402
import run_causal_inner_nonlinear_fine_5ms_completion_wp10c9d6c7c3b5c3h2e1 as h2e1  # noqa: E402
import run_causal_inner_nonlinear_middle_5ms_completion_wp10c9d6c7c3b5c3h2d1 as h2d1  # noqa: E402
import run_causal_inner_nonlinear_spatial_export_pilot_wp10c9d6c7c3b2b as b2b  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_packet_validation import (  # noqa: E402
    causal_packet_history_metrics,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c3h2g1"
ANALYZED_BASE_COMMIT = "242966f54dada9fa0bbde67dd5d56cc4b9d7b488"
ANALYZED_BASE_PARENT = "42a1a020e3c1a24094d9504444b88c5e15963ab3"
ANALYZED_BASE_TREE = "4609cdcf3614407ac9c041c7d3d5b9fa2cb9edf6"

COARSE_LAYOUT, MIDDLE_LAYOUT, FINE_LAYOUT = tuple(h2f.c3g.LAYOUTS)
LAYOUTS = (COARSE_LAYOUT, MIDDLE_LAYOUT, FINE_LAYOUT)
OBSERVABLE_NAMES = tuple(h2f.OBSERVABLE_NAMES)
FAILED_COMPONENTS = tuple(h2g.FAILED_COMPONENTS)
FAILED_INDICES = np.asarray(
    [OBSERVABLE_NAMES.index(name) for name in FAILED_COMPONENTS],
    dtype=int,
)
SPATIAL_GATES = dict(h2f.SPATIAL_GATES)
LOCALIZATION_GATES = dict(h2g.LOCALIZATION_GATES)
TIME_WINDOWS = dict(h2g.TIME_WINDOWS)

ARTIFACT = (
    "causal_inner_nonlinear_5ms_tier_i_localization_"
    "wp10c9d6c7c3b5c3h2g1"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_5ms_tier_i_localization_"
    "wp10c9d6c7c3b5c3h2g1.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_5ms_tier_i_localization_"
    "wp10c9d6c7c3b5c3h2g1.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_5MS_TIER_I_"
    "LOCALIZATION_WP10C9D6C7C3B5C3H2G1_2026-08-08.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
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


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_identity() -> dict[str, str]:
    return {
        path: _sha256(ROOT / path)
        for path in (THIS_RUNNER, THIS_TEST)
        if (ROOT / path).exists()
    }


def _validate_parent() -> tuple[dict, dict]:
    manifest_summary = _read_json(h2g.SUMMARY_PATH)
    manifest = _read_json(h2g.MANIFEST_PATH)
    if (
        not manifest_summary["passed"]
        or not manifest_summary["operator_neutral_localization_authorized"]
        or manifest_summary["new_propagation_authorized"]
        or manifest_summary["authorized_next"]
        != "WP10c9d6c7c3b5c3h2g1_operator_neutral_Tier_I_localization"
    ):
        raise RuntimeError("h2g1 authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("h2g1 analyzed identity changed")
    return manifest_summary, manifest


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    first = np.asarray(left, dtype=float).ravel()
    second = np.asarray(right, dtype=float).ravel()
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= np.finfo(float).tiny:
        return 1.0 if np.array_equal(first, second) else 0.0
    return float(np.dot(first, second) / denominator)


def _norm(values: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(values, dtype=float).ravel()))


def _restrict(history: np.ndarray, layout) -> np.ndarray:
    return h2f._restrict(np.asarray(history, dtype=float), layout)


def _common_export_history(context, face: int, history: np.ndarray):
    values = []
    maximum_audit = 0.0
    maximum_incoming = 0
    for state in history:
        value, audit = b2b._direct_observable(context, state, face)
        values.append(value)
        maximum_audit = max(
            maximum_audit,
            float(audit["local_block_ledger_defect"]),
            float(audit["source_double_count_defect"]),
            float(audit["shared_conservative_face_defect"]),
            float(audit["split_closure_defect"]),
        )
        maximum_incoming = max(
            maximum_incoming,
            int(audit["incoming_excision_characteristics"]),
        )
    return np.asarray(values, dtype=float), maximum_audit, maximum_incoming


def _packet_payload(histories, scales: np.ndarray) -> dict:
    metrics = causal_packet_history_metrics(
        *histories,
        physical_scales=scales,
        minimum_rms_order=SPATIAL_GATES["minimum_rms_order"],
        minimum_maximum_order=SPATIAL_GATES["minimum_maximum_order"],
        minimum_significant_component_order=SPATIAL_GATES[
            "minimum_significant_component_order"
        ],
        maximum_fine_normalized_difference=SPATIAL_GATES[
            "maximum_fine_normalized_difference"
        ],
        minimum_history_cosine=SPATIAL_GATES["minimum_history_cosine"],
        minimum_refinement_error_cosine=SPATIAL_GATES[
            "minimum_refinement_error_cosine"
        ],
        relative_activity=SPATIAL_GATES["minimum_relative_activity"],
    )
    indices = np.asarray(metrics.significant_components, dtype=int)
    return {
        "passed": bool(metrics.passed),
        "significant_components": tuple(
            OBSERVABLE_NAMES[index] for index in indices
        ),
        "observed_rms_order": metrics.observed_rms_order,
        "observed_maximum_order": metrics.observed_maximum_order,
        "component_orders": {
            OBSERVABLE_NAMES[index]: float(metrics.component_orders[position])
            for position, index in enumerate(indices)
        },
        "minimum_significant_component_order": (
            metrics.minimum_significant_component_order
        ),
        "maximum_fine_normalized_difference": (
            metrics.maximum_fine_normalized_difference
        ),
        "history_cosine": metrics.history_cosine,
        "refinement_error_cosine": metrics.refinement_error_cosine,
    }


def _pair_attribution(
    native_left: np.ndarray,
    native_right: np.ndarray,
    common_left: np.ndarray,
    common_right: np.ndarray,
    scales: np.ndarray,
) -> dict:
    native = (native_right - native_left) / scales[None, :]
    common = (common_right - common_left) / scales[None, :]
    layout_map = native - common
    closure = native - common - layout_map
    native_norm = max(_norm(native), np.finfo(float).tiny)
    return {
        "native_error_norm": native_norm,
        "common_state_error_norm": _norm(common),
        "layout_map_error_norm": _norm(layout_map),
        "common_state_error_fraction": _norm(common) / native_norm,
        "layout_map_error_fraction": _norm(layout_map) / native_norm,
        "common_state_error_alignment": _cosine(common, native),
        "layout_map_error_alignment": _cosine(layout_map, native),
        "decomposition_closure_defect": _norm(closure) / native_norm,
    }


def _term_pair_report(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    scale: float,
) -> dict:
    net_error = (right["net_drive"] - left["net_drive"]) / scale
    term_errors = {
        term: (right[term] - left[term]) / scale
        for term in ("inner_flux", "minus_interface_flux", "source_remainder")
    }
    term_norms = {term: _norm(values) for term, values in term_errors.items()}
    total_term_norm = max(sum(term_norms.values()), np.finfo(float).tiny)
    fractions = {
        term: value / total_term_norm for term, value in term_norms.items()
    }
    alignments = {
        term: _cosine(values, net_error) for term, values in term_errors.items()
    }
    dominant = max(fractions, key=fractions.get)
    reconstructed = sum(term_errors.values())
    net_norm = max(_norm(net_error), np.finfo(float).tiny)
    return {
        "net_error_norm": net_norm,
        "term_error_norms": term_norms,
        "term_absolute_norm_fractions": fractions,
        "term_error_alignments": alignments,
        "dominant_term": dominant,
        "dominant_fraction": fractions[dominant],
        "dominant_alignment": alignments[dominant],
        "decomposition_closure_defect": _norm(net_error - reconstructed)
        / net_norm,
    }


def _balance_terms(history: np.ndarray, channel: str) -> dict[str, np.ndarray]:
    if channel == "mass":
        inner, interface, net = 0, 3, 6
    elif channel == "angular_momentum":
        inner, interface, net = 1, 4, 7
    else:
        raise ValueError("unsupported conservative channel")
    inner_values = history[:, inner]
    interface_values = history[:, interface]
    net_values = history[:, net]
    return {
        "inner_flux": inner_values,
        "minus_interface_flux": -interface_values,
        "source_remainder": net_values - inner_values + interface_values,
        "net_drive": net_values,
    }


def _balance_report(native, scales: np.ndarray) -> tuple[dict, dict[str, np.ndarray]]:
    reports = {}
    decisive = {}
    for channel, scale_index in (("mass", 6), ("angular_momentum", 7)):
        terms = tuple(_balance_terms(values, channel) for values in native)
        for layout, payload in zip(LAYOUTS, terms, strict=True):
            for term, values in payload.items():
                decisive[f"{layout}__{channel}__{term}"] = values
        closure = max(
            _norm(
                payload["net_drive"]
                - payload["inner_flux"]
                - payload["minus_interface_flux"]
                - payload["source_remainder"]
            )
            / max(_norm(payload["net_drive"]), np.finfo(float).tiny)
            for payload in terms
        )
        coarse_middle = _term_pair_report(
            terms[0], terms[1], float(scales[scale_index])
        )
        middle_fine = _term_pair_report(
            terms[1], terms[2], float(scales[scale_index])
        )
        same_term = (
            coarse_middle["dominant_term"] == middle_fine["dominant_term"]
        )
        stable = bool(
            same_term
            and coarse_middle["dominant_fraction"]
            >= LOCALIZATION_GATES["minimum_term_error_dominance_fraction"]
            and middle_fine["dominant_fraction"]
            >= LOCALIZATION_GATES["minimum_term_error_dominance_fraction"]
            and coarse_middle["dominant_alignment"]
            >= LOCALIZATION_GATES["minimum_term_error_alignment"]
            and middle_fine["dominant_alignment"]
            >= LOCALIZATION_GATES["minimum_term_error_alignment"]
            and max(
                closure,
                coarse_middle["decomposition_closure_defect"],
                middle_fine["decomposition_closure_defect"],
            )
            <= LOCALIZATION_GATES["maximum_decomposition_closure_defect"]
        )
        reports[channel] = {
            "identity_closure_defect": closure,
            "coarse_middle": coarse_middle,
            "middle_fine": middle_fine,
            "same_dominant_term": same_term,
            "stable_term_localization": stable,
            "selected_term": coarse_middle["dominant_term"] if stable else None,
        }
    return reports, decisive


def _time_window_report(native, scales: np.ndarray) -> dict:
    normalized = tuple(
        values[:, FAILED_INDICES] / scales[FAILED_INDICES][None, :]
        for values in native
    )
    pair_errors = {
        "coarse_middle": normalized[1] - normalized[0],
        "middle_fine": normalized[2] - normalized[1],
    }
    pair_reports = {}
    for pair, error in pair_errors.items():
        total = max(float(np.sum(error**2)), np.finfo(float).tiny)
        fractions = {
            name: float(np.sum(error[np.asarray(indices, dtype=int)] ** 2))
            / total
            for name, indices in TIME_WINDOWS.items()
        }
        dominant = max(fractions, key=fractions.get)
        pair_reports[pair] = {
            "error_energy_fractions": fractions,
            "dominant_window": dominant,
            "dominant_fraction": fractions[dominant],
        }
    same_window = (
        pair_reports["coarse_middle"]["dominant_window"]
        == pair_reports["middle_fine"]["dominant_window"]
    )
    stable = bool(
        same_window
        and pair_reports["coarse_middle"]["dominant_fraction"]
        >= LOCALIZATION_GATES["minimum_time_window_error_energy_fraction"]
        and pair_reports["middle_fine"]["dominant_fraction"]
        >= LOCALIZATION_GATES["minimum_time_window_error_energy_fraction"]
    )
    return {
        **pair_reports,
        "same_dominant_window": same_window,
        "stable_time_window_localization": stable,
        "selected_window": (
            pair_reports["coarse_middle"]["dominant_window"] if stable else None
        ),
    }


def _tangent_pair_report(
    middle: dict[str, np.ndarray],
    fine: dict[str, np.ndarray],
    fine_indices: np.ndarray,
    scales: np.ndarray,
    temporal_fraction: float,
) -> dict:
    generic_index = h2e1.GENERIC_INDEX
    middle_actual = middle["anchor__actual_export_response"][:, FAILED_INDICES]
    fine_actual = fine["anchor__actual_export_response"][fine_indices][
        :, FAILED_INDICES
    ]
    middle_tangent = middle["tangent__export_directions"][:, generic_index][
        :, FAILED_INDICES
    ]
    fine_tangent = fine["tangent__export_directions"][fine_indices, generic_index][
        :, FAILED_INDICES
    ]
    failed_scales = scales[FAILED_INDICES][None, :]
    actual_error = (fine_actual - middle_actual) / failed_scales
    tangent_error = (fine_tangent - middle_tangent) / failed_scales
    correction = actual_error - tangent_error
    actual_norm = max(_norm(actual_error), np.finfo(float).tiny)
    correction_fraction = _norm(correction) / actual_norm
    cosine = _cosine(tangent_error, actual_error)
    excluded = bool(
        correction_fraction
        <= LOCALIZATION_GATES["maximum_nonlinear_remainder_pair_fraction"]
        and cosine
        >= LOCALIZATION_GATES["minimum_tangent_actual_pair_error_cosine"]
        and temporal_fraction
        <= LOCALIZATION_GATES["maximum_temporal_uncertainty_fraction"]
    )
    return {
        "actual_middle_fine_error_norm": actual_norm,
        "tangent_middle_fine_error_norm": _norm(tangent_error),
        "nonlinear_remainder_pair_correction_norm": _norm(correction),
        "nonlinear_remainder_pair_fraction": correction_fraction,
        "tangent_actual_pair_error_cosine": cosine,
        "sampled_temporal_uncertainty_fraction": temporal_fraction,
        "schedule_or_nonlinear_remainder_dominance_excluded": excluded,
    }


def _analyze() -> tuple[dict, dict[str, np.ndarray]]:
    started = time.perf_counter()
    certificate = _read_json(h2f.SUMMARY_PATH)
    certificate_arrays = _load_npz(h2f.DECISIVE_ARRAYS)
    coarse = _load_npz(c3d.DECISIVE_ARRAYS)
    middle = _load_npz(h2d1.DECISIVE_ARRAYS)
    fine = _load_npz(h2e1.DECISIVE_ARRAYS)
    times = np.asarray(certificate_arrays["times_seconds"], dtype=float)
    fine_indices = np.asarray(certificate_arrays["fine_target_indices"], dtype=int)
    scales = np.asarray(certificate_arrays["export_scales"], dtype=float)
    parent_grid, layouts, configurations = b2b._layouts_and_contexts(
        b2b._input_arrays()
    )
    del parent_grid

    native = (
        np.asarray(
            certificate_arrays["coarse_generic_instantaneous_export_response"],
            dtype=float,
        ),
        np.asarray(
            certificate_arrays[
                "middle_generic_nonlinear_instantaneous_export_response"
            ],
            dtype=float,
        ),
        np.asarray(
            certificate_arrays["fine_generic_nonlinear_instantaneous_export_response"],
            dtype=float,
        ),
    )
    # The final certificate stores semantic coarse/middle/fine names rather
    # than layout labels. Validate the expected native shapes explicitly.
    if any(values.shape != (times.size, scales.size) for values in native):
        raise RuntimeError("native generic export histories changed")

    full_states = {
        COARSE_LAYOUT: (
            coarse["base__main__output_states"],
            coarse["perturbed__main__output_states"],
        ),
        MIDDLE_LAYOUT: (
            middle["base__accepted_states"],
            middle["anchor__anchor_states"],
        ),
        FINE_LAYOUT: (
            fine["base__accepted_states"][fine_indices],
            fine["anchor__anchor_states"][fine_indices],
        ),
    }
    common_context = configurations[COARSE_LAYOUT]["context"]
    common_face = int(layouts[COARSE_LAYOUT].coupling_face_index)
    common = []
    decisive = {
        "times_seconds": times,
        "export_scales": scales,
        "failed_component_indices": FAILED_INDICES,
    }
    maximum_common_audit = 0.0
    maximum_common_incoming = 0
    for layout in LAYOUTS:
        base_states, anchor_states = full_states[layout]
        restricted_base = _restrict(base_states, layouts[layout])
        restricted_anchor = _restrict(anchor_states, layouts[layout])
        base_exports, base_audit, base_incoming = _common_export_history(
            common_context, common_face, restricted_base
        )
        anchor_exports, anchor_audit, anchor_incoming = _common_export_history(
            common_context, common_face, restricted_anchor
        )
        response = anchor_exports - base_exports
        common.append(response)
        maximum_common_audit = max(
            maximum_common_audit, base_audit, anchor_audit
        )
        maximum_common_incoming = max(
            maximum_common_incoming, base_incoming, anchor_incoming
        )
        decisive[f"{layout}__common_parent_base_exports"] = base_exports
        decisive[f"{layout}__common_parent_anchor_exports"] = anchor_exports
        decisive[f"{layout}__common_parent_export_response"] = response
        decisive[f"{layout}__native_export_response"] = native[LAYOUTS.index(layout)]
        decisive[f"{layout}__layout_map_defect"] = (
            native[LAYOUTS.index(layout)] - response
        )
    common_tuple = tuple(common)
    common_metric = _packet_payload(common_tuple, scales)
    native_failed = tuple(values[:, FAILED_INDICES] for values in native)
    common_failed = tuple(values[:, FAILED_INDICES] for values in common_tuple)
    failed_scales = scales[FAILED_INDICES]
    attribution = {
        "coarse_middle": _pair_attribution(
            native_failed[0],
            native_failed[1],
            common_failed[0],
            common_failed[1],
            failed_scales,
        ),
        "middle_fine": _pair_attribution(
            native_failed[1],
            native_failed[2],
            common_failed[1],
            common_failed[2],
            failed_scales,
        ),
    }
    localized_to_map = bool(
        common_metric["passed"]
        and min(
            item["layout_map_error_alignment"] for item in attribution.values()
        )
        >= LOCALIZATION_GATES["minimum_layout_map_alignment"]
        and max(
            item["common_state_error_fraction"] for item in attribution.values()
        )
        <= LOCALIZATION_GATES["maximum_common_state_error_fraction"]
        and max(
            item["decomposition_closure_defect"] for item in attribution.values()
        )
        <= LOCALIZATION_GATES["maximum_decomposition_closure_defect"]
    )

    balance, balance_arrays = _balance_report(native, scales)
    decisive.update(balance_arrays)
    time_windows = _time_window_report(native, scales)
    temporal_fraction = float(
        certificate["analysis"]["instantaneous_Tier_I"][
            "temporal_classification"
        ]["temporal_uncertainty_to_medium_fine_difference_ratio"]
    )
    tangent_pair = _tangent_pair_report(
        middle, fine, fine_indices, scales, temporal_fraction
    )

    stable_terms = {
        channel: report["selected_term"]
        for channel, report in balance.items()
        if report["stable_term_localization"]
    }
    if localized_to_map:
        classification = (
            "five_ms_Tier_I_failure_localized_to_layout_native_export_map_"
            "common_parent_map_passes"
        )
        authorized_next = (
            "WP10c9d6c7c3b5c3h2h_layout_native_inner_export_map_audit_manifest"
        )
    elif "mass" in stable_terms and stable_terms["mass"] == "inner_flux":
        classification = (
            "five_ms_Tier_I_failure_localized_to_inner_face_response_"
            "half_cell_audit_manifest_authorized"
        )
        authorized_next = (
            "WP10c9d6c7c3b5c3h2h_inner_face_half_cell_audit_manifest"
        )
    elif any(value == "minus_interface_flux" for value in stable_terms.values()):
        classification = (
            "five_ms_Tier_I_failure_localized_to_coupling_face_response_"
            "audit_manifest_authorized"
        )
        authorized_next = "WP10c9d6c7c3b5c3h2h_coupling_face_audit_manifest"
    elif any(value == "source_remainder" for value in stable_terms.values()):
        classification = (
            "five_ms_Tier_I_failure_localized_to_source_prefix_response_"
            "audit_manifest_authorized"
        )
        authorized_next = "WP10c9d6c7c3b5c3h2h_source_prefix_audit_manifest"
    elif time_windows["stable_time_window_localization"]:
        classification = (
            "five_ms_Tier_I_failure_localized_to_time_window_"
            "diagnostic_manifest_authorized"
        )
        authorized_next = "WP10c9d6c7c3b5c3h2h_window_specific_audit_manifest"
    else:
        classification = (
            "five_ms_Tier_I_failure_not_uniquely_localized_"
            "distributed_observable_coupling_manifest_authorized"
        )
        authorized_next = (
            "WP10c9d6c7c3b5c3h2h_distributed_observable_coupling_manifest"
        )
    result = {
        "classification": classification,
        "authorized_next": authorized_next,
        "common_parent_map": {
            "metric": common_metric,
            "pairwise_attribution": attribution,
            "localized_to_layout_native_export_map": localized_to_map,
            "maximum_common_parent_ledger_defect": maximum_common_audit,
            "maximum_common_parent_incoming_excision_characteristics": (
                maximum_common_incoming
            ),
        },
        "net_drive_balance": balance,
        "stable_term_localizations": stable_terms,
        "time_window_error_energy": time_windows,
        "tangent_nonlinear_pair_error": tangent_pair,
        "elapsed_wall_seconds": time.perf_counter() - started,
    }
    return result, decisive


def _update_catalog(summary: dict) -> None:
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha256(path),
                    "scientific_status": "CERTIFIED",
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case", "path", "bytes", "sha256", "scientific_status"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read_json(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": summary["passed"],
    }
    catalog.update(
        {
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": ANALYZED_BASE_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def main() -> int:
    manifest_summary, manifest = _validate_parent()
    result, decisive = _analyze()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": result["classification"],
        "passed": True,
        "authorized_next": result["authorized_next"],
        "parent_manifest_classification": manifest_summary["classification"],
        "operator_changed": False,
        "production_defaults_changed": False,
        "propagation_executed": False,
        "localization": result,
        "five_ms_spatial_convergence_certified": False,
        "fourth_duration_rung_manifest_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "layouts": LAYOUTS,
            "failed_components": FAILED_COMPONENTS,
            "observable_names": OBSERVABLE_NAMES,
            "localization_gates": LOCALIZATION_GATES,
            "time_windows": TIME_WINDOWS,
            "manifest_sha256": _sha256(h2g.MANIFEST_PATH),
        },
    )
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)
    _write_json(SUMMARY_PATH, summary)
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "source_parent_commit": ANALYZED_BASE_COMMIT,
            "scientific_status": "CERTIFIED",
            "working_head": _git_value("rev-parse", "HEAD"),
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "input_hashes": {
                "manifest": _sha256(h2g.MANIFEST_PATH),
                "certificate_summary": _sha256(h2f.SUMMARY_PATH),
                "certificate_arrays": _sha256(h2f.DECISIVE_ARRAYS),
                "coarse_arrays": _sha256(c3d.DECISIVE_ARRAYS),
                "middle_arrays": _sha256(h2d1.DECISIVE_ARRAYS),
                "fine_arrays": _sha256(h2e1.DECISIVE_ARRAYS),
            },
            "implementation_source_hashes": _source_identity(),
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "command": f"PYTHONPATH=src:scripts python {THIS_RUNNER}",
        },
    )
    common = result["common_parent_map"]
    tangent = result["tangent_nonlinear_pair_error"]
    windows = result["time_window_error_energy"]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Nonlinear 5 ms Tier-I localization WP10c9d6c7c3b5c3h2g1",
                "",
                "## Classification",
                "",
                f"`{summary['classification']}`",
                "",
                "No state was propagated and no operator was changed. The committed "
                "coarse, middle, and fine base/anchor states were restricted to one "
                "64-cell parent and evaluated through one common nonlinear export map.",
                "",
                "## Discriminators",
                "",
                f"- common-parent export contract passed: "
                f"`{common['metric']['passed']}`",
                f"- localized to layout-native export map: "
                f"`{common['localized_to_layout_native_export_map']}`",
                f"- coarse/middle map alignment and common fraction: "
                f"`{common['pairwise_attribution']['coarse_middle']['layout_map_error_alignment']:.6f}` / "
                f"`{common['pairwise_attribution']['coarse_middle']['common_state_error_fraction']:.6f}`",
                f"- middle/fine map alignment and common fraction: "
                f"`{common['pairwise_attribution']['middle_fine']['layout_map_error_alignment']:.6f}` / "
                f"`{common['pairwise_attribution']['middle_fine']['common_state_error_fraction']:.6f}`",
                f"- stable net-drive term localizations: `{result['stable_term_localizations']}`",
                f"- stable time-window localization: "
                f"`{windows['stable_time_window_localization']}`",
                f"- nonlinear-remainder pair fraction / tangent-actual cosine: "
                f"`{tangent['nonlinear_remainder_pair_fraction']:.6f}` / "
                f"`{tangent['tangent_actual_pair_error_cosine']:.6f}`",
                f"- sampled temporal uncertainty fraction: "
                f"`{tangent['sampled_temporal_uncertainty_fraction']:.6f}`",
                "",
                "## Decision",
                "",
                f"Only `{summary['authorized_next']}` is authorized. The rejected "
                "5 ms spatial certificate is preserved. The fourth duration rung, "
                "fixed-Q experiments, and reduced slow evolution remain blocked.",
                "",
            )
        ),
        encoding="utf-8",
    )
    names = ("config.json", "decisive_arrays.npz", "provenance.json", "summary.json")
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
