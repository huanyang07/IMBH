#!/usr/bin/env python3
"""Analyze the committed coarse/middle 20 ms checkpoint without propagation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import sys

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_nonlinear_ten_ms_screen_wp10c9d6c7c3b5c4b2 as c4b2  # noqa: E402
import run_causal_inner_nonlinear_twenty_ms_completion_wp10c9d6c7c3b5c4c1 as c4c1  # noqa: E402
import run_causal_inner_nonlinear_twenty_ms_spatial_checkpoint_manifest_wp10c9d6c7c3b5c4e as c4e  # noqa: E402
import run_causal_inner_nonlinear_middle_6ms_spatial_cost_pilot_wp10c9d6c7c3b5c4e1 as c4e1  # noqa: E402
import run_causal_inner_nonlinear_optimized_middle_20ms_completion_wp10c9d6c7c3b5c4e3 as c4e3  # noqa: E402
import run_causal_inner_nonlinear_spatial_export_pilot_wp10c9d6c7c3b2b as b2b  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_embedded_patch import (  # noqa: E402
    restrict_causal_embedded_patch_cell_averages,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4e4"
ANALYZED_BASE_COMMIT = "132857f4acb0c71185bd2096fc6a4bbd87ac0675"
ANALYZED_BASE_PARENT = "e0154f7c4f97be51f6b8efce507bdb8b1a14f9d0"
ANALYZED_BASE_TREE = "91f3cc17443e4cf25e652a5159b251420b764ba8"

ARTIFACT = (
    "causal_inner_nonlinear_coarse_middle_20ms_checkpoint_analysis_"
    "wp10c9d6c7c3b5c4e4"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_coarse_middle_20ms_checkpoint_"
    "analysis_wp10c9d6c7c3b5c4e4.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_coarse_middle_20ms_checkpoint_"
    "analysis_wp10c9d6c7c3b5c4e4.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_COARSE_MIDDLE_20MS_"
    "CHECKPOINT_ANALYSIS_WP10C9D6C7C3B5C4E4_2026-08-11.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
CONTRACT_PATH = CANONICAL_DIRECTORY / "analysis_contract.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

COARSE_TEN_ARRAYS = c4b2.DECISIVE_ARRAYS
COARSE_TWENTY_ARRAYS = c4c1.DECISIVE_ARRAYS
MIDDLE_SIX_ARRAYS = c4e1.DECISIVE_ARRAYS
MIDDLE_TWENTY_ARRAYS = c4e3.DECISIVE_ARRAYS
MIDDLE_LAYOUT = c4e3.h2b1.MIDDLE_LAYOUT
GENERIC_INDEX = c4e3.GENERIC_INDEX
EXTRACTION_RELATIVE_STEP = c4e3.EXTRACTION_JVP_RELATIVE_STEP

MAXIMUM_NORMALIZED_DIFFERENCE = 0.05
MINIMUM_CROSS_GRID_RESPONSE_COSINE = 0.90
MAXIMUM_TEMPORAL_UNCERTAINTY_FRACTION = 0.10
MAXIMUM_SURROGATE_UNCERTAINTY_FRACTION = 0.10
NEAR_GATE_FRACTION = 0.80
WINDOWS_SECONDS = ((0.005, 0.020), (0.010, 0.020), (0.016, 0.020))


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


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": "coarse_middle_20ms_checkpoint_analysis_contract_frozen",
        "definitions_frozen_before_analysis": True,
        "propagation_authorized": False,
        "operator_change_authorized": False,
        "scope": {
            "generic_nonlinear_response_only": True,
            "state_restricted_to_common_64_cell_parent": True,
            "slow_export": "certified_exterior_extraction_partition",
            "raw_inner_face_is_not_a_slow_export": True,
            "instantaneous_interval_seconds": [0.005, 0.020],
            "cumulative_interval_seconds": [0.005, 0.020],
            "window_mean_intervals_seconds": WINDOWS_SECONDS,
            "cumulative_uses_one_common_coarse_target_quadrature": True,
        },
        "screening_gates": {
            "maximum_normalized_state_or_extraction_difference": (
                MAXIMUM_NORMALIZED_DIFFERENCE
            ),
            "minimum_cross_grid_response_history_cosine": (
                MINIMUM_CROSS_GRID_RESPONSE_COSINE
            ),
            "maximum_temporal_uncertainty_fraction_of_spatial_difference": (
                MAXIMUM_TEMPORAL_UNCERTAINTY_FRACTION
            ),
            "maximum_surrogate_uncertainty_fraction_for_tangent_only_fine_anchor": (
                MAXIMUM_SURROGATE_UNCERTAINTY_FRACTION
            ),
            "near_gate_fraction_triggering_full_fine_anchor": NEAR_GATE_FRACTION,
        },
        "uncertainty": {
            "coarse": (
                "maximum of strict response discrepancy and summed base-plus-"
                "perturbed local estimator"
            ),
            "middle_base": (
                "observable-specific cubic envelope between declared full-step-"
                "versus-two-half audits"
            ),
            "middle_anchor": (
                "observable-specific cubic envelope between declared sampled "
                "nonlinear-anchor audits"
            ),
            "surrogate": "nonlinear middle anchor minus complete discrete BDF tangent",
        },
        "decision": {
            "screen_passes": (
                "definitions-only cost-bounded fine manifest authorized; no fine "
                "propagation directly"
            ),
            "screen_fails_spatial_or_direction_gate": "stop_before_fine_and_localize",
            "screen_fails_temporal_gate": "strengthen_middle_temporal_reference_before_fine",
            "fine_level_required_for_measured_order": True,
            "fine_base_and_five_profile_block_tangent_are_minimum_work": True,
            "full_fine_generic_anchor_is_conditional": True,
        },
        "hard_stops": (
            "do_not_issue_a_spatial_order_from_two_grids",
            "do_not_run_fine_in_this_package",
            "do_not_run_50ms_fixed_Q_or_reduced_evolution",
            "do_not_use_the_raw_inner_face_flux_as_slow_export",
            "do_not_change_operator_profile_or_thresholds",
        ),
    }


def _validate_inputs() -> None:
    parent = _read_json(c4e3.SUMMARY_PATH)
    coarse = _read_json(c4c1.SUMMARY_PATH)
    campaign = _read_json(c4e.SUMMARY_PATH)
    if (
        not parent["passed"]
        or not parent["coarse_middle_twenty_ms_checkpoint_analysis_authorized"]
        or parent["authorized_next"]
        != f"{WORK_PACKAGE}_coarse_middle_20ms_checkpoint_analysis"
        or parent["fine_twenty_ms_propagation_authorized"]
        or not coarse["passed"]
        or not campaign["passed"]
    ):
        raise RuntimeError("c4e4 input authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c4e4 analyzed identity changed")


def _concat(left_times, left_values, right_times, right_values):
    if abs(float(left_times[-1] - right_times[0])) > 1.0e-15:
        raise RuntimeError("c4e4 continuation boundary changed")
    if not np.array_equal(left_values[-1], right_values[0]):
        raise RuntimeError("c4e4 continuation value is not bitwise common")
    return (
        np.concatenate((left_times, right_times[1:])),
        np.concatenate((left_values, right_values[1:]), axis=0),
    )


def _indices(source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    source_ids = np.rint(source_times * 1.0e6).astype(int)
    target_ids = np.rint(target_times * 1.0e6).astype(int)
    result = []
    for target in target_ids:
        matches = np.flatnonzero(source_ids == target)
        if matches.size != 1:
            raise RuntimeError(f"c4e4 target {target} us is not unique")
        result.append(int(matches[0]))
    return np.asarray(result, dtype=int)


def _interpolate(times, values, target_times):
    flat = values.reshape(values.shape[0], -1)
    result = np.column_stack(
        [np.interp(target_times, times, flat[:, index]) for index in range(flat.shape[1])]
    )
    return result.reshape((target_times.size,) + values.shape[1:])


def _restrict(history: np.ndarray, layout) -> np.ndarray:
    return np.asarray(
        [
            restrict_causal_embedded_patch_cell_averages(state, layout)
            for state in history
        ],
        dtype=float,
    )


def _cumulative(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values)
    result[1:] = np.cumsum(
        0.5 * (values[1:] + values[:-1]) * np.diff(times)[:, None], axis=0
    )
    return result


def _window_means(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    means = []
    for start, stop in WINDOWS_SECONDS:
        selected = (times >= start - 1.0e-15) & (times <= stop + 1.0e-15)
        selected_times = times[selected]
        selected_values = values[selected]
        if (
            selected_times.size < 2
            or abs(float(selected_times[0] - start)) > 1.0e-15
            or abs(float(selected_times[-1] - stop)) > 1.0e-15
        ):
            raise RuntimeError("c4e4 mean window lacks exact endpoints")
        means.append(
            np.trapezoid(selected_values, selected_times, axis=0) / (stop - start)
        )
    return np.asarray(means)


def _scaled(values: np.ndarray, scales: np.ndarray) -> np.ndarray:
    shape = (1,) * (values.ndim - 1) + (scales.size,)
    return values / scales.reshape(shape)


def _metrics(coarse, middle, scales) -> dict:
    coarse_scaled = _scaled(coarse, scales)
    middle_scaled = _scaled(middle, scales)
    difference = middle_scaled - coarse_scaled
    left = coarse_scaled.ravel()
    right = middle_scaled.ravel()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    cosine = float(np.dot(left, right) / denominator) if denominator else 1.0
    axes = tuple(range(difference.ndim - 1))
    return {
        "maximum_normalized_difference": float(np.max(np.abs(difference))),
        "rms_normalized_difference": float(np.sqrt(np.mean(difference**2))),
        "cross_grid_response_history_cosine": cosine,
        "maximum_component_normalized_difference": np.max(
            np.abs(difference), axis=axes
        ),
        "maximum_coarse_response": float(np.max(np.abs(coarse_scaled))),
        "maximum_middle_response": float(np.max(np.abs(middle_scaled))),
    }


def _difference(left, right, scales) -> float:
    return float(np.max(np.abs(_scaled(left - right, scales))))


def _component_envelope(
    timesteps: np.ndarray,
    audit_flags: np.ndarray,
    audit_values: np.ndarray,
) -> float:
    last_dt = None
    last_value = None
    maximum = 0.0
    audit_index = 0
    compact = audit_values.size != audit_flags.size
    for index, (dt, audited) in enumerate(zip(timesteps, audit_flags, strict=True)):
        if audited:
            value = float(audit_values[audit_index] if compact else audit_values[index])
            audit_index += int(compact)
            last_dt = float(dt)
            last_value = value
        else:
            if last_dt is None or last_value is None:
                raise RuntimeError("c4e4 component envelope starts without audit")
            value = 4.0 * last_value * (float(dt) / last_dt) ** 3
        maximum = max(maximum, value)
    if compact and audit_index != audit_values.size:
        raise RuntimeError("c4e4 compact audit values were not consumed")
    return maximum


def _coarse_strict_uncertainty(arrays, value_name, scales) -> float:
    main_times = arrays["base_main__output_times"]
    strict_times = arrays["base_strict__output_times"]
    indices = _indices(main_times, strict_times)
    main = (
        arrays[f"perturbed_main__{value_name}"][indices]
        - arrays[f"base_main__{value_name}"][indices]
    )
    strict = (
        arrays[f"perturbed_strict__{value_name}"]
        - arrays[f"base_strict__{value_name}"]
    )
    strict_difference = _difference(main, strict, scales)
    local = float(
        np.max(arrays["base_main__local_error_estimates"])
        + np.max(arrays["perturbed_main__local_error_estimates"])
    )
    return max(strict_difference, local)


def _middle_early_extraction_tangent(
    middle_six: dict[str, np.ndarray],
) -> tuple[np.ndarray, float, np.ndarray]:
    configuration = c4e3.h2b1._configuration()
    context = configuration["context"]
    directions = []
    identity = 0.0
    audit = np.zeros(4, dtype=float)
    for state, direction in zip(
        middle_six["base__accepted_states"],
        middle_six["tangent__state_directions"][:, GENERIC_INDEX],
        strict=True,
    ):
        value, defect, local_audit = c4e3._extraction_direction(
            context, state, direction, EXTRACTION_RELATIVE_STEP
        )
        directions.append(value)
        identity = max(identity, defect)
        audit = np.maximum(audit, local_audit)
    return np.asarray(directions), identity, audit


def _analyze(contract: dict) -> tuple[dict, dict[str, np.ndarray]]:
    coarse_ten = _load_npz(COARSE_TEN_ARRAYS)
    coarse_twenty = _load_npz(COARSE_TWENTY_ARRAYS)
    middle_six = _load_npz(MIDDLE_SIX_ARRAYS)
    middle_twenty = _load_npz(MIDDLE_TWENTY_ARRAYS)
    field_scales = np.asarray(middle_twenty["tangent__field_scales"], dtype=float)
    extraction_scales = np.asarray(
        coarse_twenty["extraction_partition_scales"], dtype=float
    )
    if not np.array_equal(field_scales, coarse_ten["field_scales"]):
        raise RuntimeError("c4e4 field scales changed")
    if not np.array_equal(
        extraction_scales, coarse_ten["extraction_partition_scales"]
    ):
        raise RuntimeError("c4e4 extraction scales changed")

    coarse_ten_response = (
        coarse_ten["perturbed_main__output_extraction_partition"]
        - coarse_ten["base_main__output_extraction_partition"]
    )
    coarse_twenty_response = (
        coarse_twenty["perturbed_main__output_extraction_partition"]
        - coarse_twenty["base_main__output_extraction_partition"]
    )
    coarse_times, coarse_extraction = _concat(
        coarse_ten["base_main__output_times"],
        coarse_ten_response,
        coarse_twenty["base_main__output_times"],
        coarse_twenty_response,
    )

    early_actual_extraction = middle_six["extraction__response"]
    late_actual_extraction = middle_twenty["extraction__actual_generic_response"]
    middle_times, middle_extraction = _concat(
        middle_six["extraction__accepted_times"],
        early_actual_extraction,
        middle_twenty["extraction__accepted_times"],
        late_actual_extraction,
    )
    early_predicted_extraction, early_identity, early_audit = (
        _middle_early_extraction_tangent(middle_six)
    )
    _, middle_predicted_extraction = _concat(
        middle_six["extraction__accepted_times"],
        early_predicted_extraction,
        middle_twenty["extraction__accepted_times"],
        middle_twenty["extraction__predicted_generic_response"],
    )

    middle_state_times, middle_actual_state = _concat(
        middle_six["base__accepted_times"],
        middle_six["anchor__actual_state_response"],
        middle_twenty["base__accepted_times"],
        middle_twenty["anchor__actual_state_response"],
    )
    _, middle_predicted_state = _concat(
        middle_six["base__accepted_times"],
        middle_six["tangent__state_directions"][:, GENERIC_INDEX],
        middle_twenty["base__accepted_times"],
        middle_twenty["tangent__state_directions"][:, GENERIC_INDEX],
    )
    coarse_state_times, coarse_state = _concat(
        coarse_ten["base_main__output_times"],
        coarse_ten["perturbed_main__output_states"]
        - coarse_ten["base_main__output_states"],
        coarse_twenty["base_main__output_times"],
        coarse_twenty["perturbed_main__output_states"]
        - coarse_twenty["base_main__output_states"],
    )

    common_times = np.asarray(
        sorted(set(coarse_state_times).intersection(set(middle_state_times))),
        dtype=float,
    )
    coarse_state_common = coarse_state[_indices(coarse_state_times, common_times)]
    middle_state_common_native = middle_actual_state[
        _indices(middle_state_times, common_times)
    ]
    middle_tangent_common_native = middle_predicted_state[
        _indices(middle_state_times, common_times)
    ]
    _parent, layouts, _contexts = b2b._layouts_and_contexts(b2b._input_arrays())
    layout = layouts[MIDDLE_LAYOUT]
    middle_state_common = _restrict(middle_state_common_native, layout)
    middle_tangent_common = _restrict(middle_tangent_common_native, layout)
    state_metrics = _metrics(
        coarse_state_common, middle_state_common, field_scales
    )
    state_surrogate = _difference(
        middle_state_common, middle_tangent_common, field_scales
    )

    coarse_extraction_common = coarse_extraction[
        _indices(coarse_times, common_times)
    ]
    middle_extraction_common = middle_extraction[
        _indices(middle_times, common_times)
    ]
    middle_predicted_common = middle_predicted_extraction[
        _indices(middle_times, common_times)
    ]
    instantaneous_metrics = _metrics(
        coarse_extraction_common, middle_extraction_common, extraction_scales
    )
    instantaneous_surrogate = _difference(
        middle_extraction_common, middle_predicted_common, extraction_scales
    )

    quadrature_times = coarse_times
    middle_quadrature = _interpolate(
        middle_times, middle_extraction, quadrature_times
    )
    middle_predicted_quadrature = _interpolate(
        middle_times, middle_predicted_extraction, quadrature_times
    )
    coarse_cumulative = _cumulative(coarse_extraction, quadrature_times)
    middle_cumulative = _cumulative(middle_quadrature, quadrature_times)
    predicted_cumulative = _cumulative(
        middle_predicted_quadrature, quadrature_times
    )
    duration = float(quadrature_times[-1] - quadrature_times[0])
    cumulative_scales = extraction_scales * duration
    cumulative_metrics = _metrics(
        coarse_cumulative, middle_cumulative, cumulative_scales
    )
    cumulative_surrogate = _difference(
        middle_cumulative, predicted_cumulative, cumulative_scales
    )
    coarse_means = _window_means(coarse_extraction, quadrature_times)
    middle_means = _window_means(middle_quadrature, quadrature_times)
    predicted_means = _window_means(
        middle_predicted_quadrature, quadrature_times
    )
    mean_metrics = _metrics(coarse_means, middle_means, extraction_scales)
    mean_surrogate = _difference(
        middle_means, predicted_means, extraction_scales
    )

    coarse_state_temporal = max(
        _coarse_strict_uncertainty(
            coarse_ten, "output_states", field_scales
        ),
        _coarse_strict_uncertainty(
            coarse_twenty, "output_states", field_scales
        ),
    )
    coarse_extraction_temporal = max(
        _coarse_strict_uncertainty(
            coarse_ten, "output_extraction_partition", extraction_scales
        ),
        _coarse_strict_uncertainty(
            coarse_twenty, "output_extraction_partition", extraction_scales
        ),
    )
    middle_early_state_temporal = float(
        np.max(middle_six["base__local_state_estimates"])
        + np.max(middle_six["anchor__sampled_state_error_estimates"])
    )
    middle_early_extraction_temporal = float(
        np.max(middle_six["base__local_export_estimates"])
        + np.max(middle_six["anchor__sampled_export_error_estimates"])
    )
    middle_late_state_temporal = _component_envelope(
        middle_twenty["base__accepted_timesteps"],
        middle_twenty["base__audit_flags"],
        middle_twenty["base__local_state_estimates"],
    ) + _component_envelope(
        middle_twenty["base__accepted_timesteps"],
        middle_twenty["anchor__sampled_flags"],
        middle_twenty["anchor__sampled_state_error_estimates"],
    )
    middle_late_extraction_temporal = _component_envelope(
        middle_twenty["base__accepted_timesteps"],
        middle_twenty["base__audit_flags"],
        middle_twenty["base__local_extraction_estimates"],
    ) + _component_envelope(
        middle_twenty["base__accepted_timesteps"],
        middle_twenty["anchor__sampled_flags"],
        middle_twenty["anchor__sampled_export_error_estimates"],
    )
    state_temporal = coarse_state_temporal + max(
        middle_early_state_temporal, middle_late_state_temporal
    )
    extraction_temporal = coarse_extraction_temporal + max(
        middle_early_extraction_temporal, middle_late_extraction_temporal
    )

    def finalize(metrics, temporal, surrogate):
        spatial = max(
            float(metrics["maximum_normalized_difference"]),
            np.finfo(float).tiny,
        )
        metrics.update(
            {
                "temporal_uncertainty": temporal,
                "temporal_uncertainty_fraction_of_spatial_difference": (
                    temporal / spatial
                ),
                "surrogate_uncertainty": surrogate,
                "surrogate_uncertainty_fraction_of_spatial_difference": (
                    surrogate / spatial
                ),
            }
        )
        metrics["screen_passed"] = bool(
            metrics["maximum_normalized_difference"]
            <= MAXIMUM_NORMALIZED_DIFFERENCE
            and metrics["cross_grid_response_history_cosine"]
            >= MINIMUM_CROSS_GRID_RESPONSE_COSINE
            and metrics["temporal_uncertainty_fraction_of_spatial_difference"]
            <= MAXIMUM_TEMPORAL_UNCERTAINTY_FRACTION
        )
        return metrics

    state_metrics = finalize(state_metrics, state_temporal, state_surrogate)
    instantaneous_metrics = finalize(
        instantaneous_metrics, extraction_temporal, instantaneous_surrogate
    )
    cumulative_metrics = finalize(
        cumulative_metrics, extraction_temporal, cumulative_surrogate
    )
    mean_metrics = finalize(mean_metrics, extraction_temporal, mean_surrogate)
    checkpoint_passed = bool(
        state_metrics["screen_passed"]
        and instantaneous_metrics["screen_passed"]
        and cumulative_metrics["screen_passed"]
        and mean_metrics["screen_passed"]
    )
    surrogate_ratios = {
        "state": state_metrics[
            "surrogate_uncertainty_fraction_of_spatial_difference"
        ],
        "instantaneous_extraction": instantaneous_metrics[
            "surrogate_uncertainty_fraction_of_spatial_difference"
        ],
        "cumulative_extraction": cumulative_metrics[
            "surrogate_uncertainty_fraction_of_spatial_difference"
        ],
        "window_mean_extraction": mean_metrics[
            "surrogate_uncertainty_fraction_of_spatial_difference"
        ],
    }
    near_gate = max(
        state_metrics["maximum_normalized_difference"],
        instantaneous_metrics["maximum_normalized_difference"],
        cumulative_metrics["maximum_normalized_difference"],
        mean_metrics["maximum_normalized_difference"],
    ) >= NEAR_GATE_FRACTION * MAXIMUM_NORMALIZED_DIFFERENCE
    full_fine_anchor = bool(
        max(surrogate_ratios.values()) > MAXIMUM_SURROGATE_UNCERTAINTY_FRACTION
        or near_gate
    )
    maximum_audit = np.maximum(
        early_audit,
        np.max(middle_twenty["extraction__maximum_ledger_audits"], axis=0),
    )
    analysis = {
        "checkpoint_screen_passed": checkpoint_passed,
        "state": state_metrics,
        "instantaneous_extraction": instantaneous_metrics,
        "cumulative_extraction": cumulative_metrics,
        "window_mean_extraction": mean_metrics,
        "surrogate_to_spatial_difference_ratios": surrogate_ratios,
        "result_near_normalized_difference_gate": near_gate,
        "full_fine_generic_anchor_required": full_fine_anchor,
        "minimum_fine_work": (
            "one_fine_nonlinear_base",
            "one_five_profile_block_tangent",
            "certified_extraction_partition_tangents",
            "sampled_temporal_audits",
        ),
        "maximum_early_extraction_identity_defect": early_identity,
        "maximum_shared_conservative_face_defect": float(maximum_audit[0]),
        "maximum_local_block_ledger_defect": float(maximum_audit[1]),
        "maximum_source_double_count_defect": float(maximum_audit[2]),
        "maximum_incoming_excision_characteristics": int(maximum_audit[3]),
        "analysis_contract": contract,
    }
    decisive = {
        "common_times_seconds": common_times,
        "quadrature_times_seconds": quadrature_times,
        "field_scales": field_scales,
        "extraction_scales": extraction_scales,
        "coarse_state_response": coarse_state_common,
        "middle_state_response": middle_state_common,
        "middle_tangent_state_response": middle_tangent_common,
        "coarse_extraction_response": coarse_extraction_common,
        "middle_extraction_response": middle_extraction_common,
        "middle_tangent_extraction_response": middle_predicted_common,
        "coarse_cumulative_extraction_response": coarse_cumulative,
        "middle_cumulative_extraction_response": middle_cumulative,
        "middle_tangent_cumulative_extraction_response": predicted_cumulative,
        "coarse_window_mean_extraction_response": coarse_means,
        "middle_window_mean_extraction_response": middle_means,
        "middle_tangent_window_mean_extraction_response": predicted_means,
    }
    return analysis, decisive


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
    _validate_inputs()
    contract = _contract()
    analysis, decisive = _analyze(contract)
    screen_passed = analysis["checkpoint_screen_passed"]
    if screen_passed:
        classification = (
            "coarse_middle_20ms_checkpoint_screen_passed_cost_bounded_fine_"
            "manifest_authorized"
        )
        authorized_next = (
            "WP10c9d6c7c3b5c4e5_cost_bounded_fine_20ms_completion_manifest"
        )
    else:
        temporal_failed = any(
            analysis[name]["temporal_uncertainty_fraction_of_spatial_difference"]
            > MAXIMUM_TEMPORAL_UNCERTAINTY_FRACTION
            for name in (
                "state",
                "instantaneous_extraction",
                "cumulative_extraction",
                "window_mean_extraction",
            )
        )
        classification = (
            "coarse_middle_20ms_checkpoint_temporal_reference_insufficient_"
            "fine_blocked"
            if temporal_failed
            else "coarse_middle_20ms_checkpoint_screen_failed_fine_blocked"
        )
        authorized_next = (
            "middle_20ms_temporal_reference_hardening_only"
            if temporal_failed
            else "coarse_middle_20ms_failure_localization_only"
        )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": True,
        "analysis_completed": True,
        "analysis": analysis,
        "coarse_middle_twenty_ms_checkpoint_screen_passed": screen_passed,
        "twenty_ms_spatial_checkpoint_certified": False,
        "fine_completion_manifest_authorized": screen_passed,
        "fine_twenty_ms_propagation_authorized": False,
        "full_fine_generic_anchor_required": analysis[
            "full_fine_generic_anchor_required"
        ],
        "fifty_ms_propagation_authorized": False,
        "physical_failure_detected": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": authorized_next,
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "coarse_layout": c4e.LAYOUTS[0],
            "middle_layout": MIDDLE_LAYOUT,
            "coarse_extraction_face": c4e.EXTRACTION_FACE_INDICES[0],
            "middle_extraction_face": c4e.EXTRACTION_FACE_INDICES[1],
            "extraction_radius_rg": c4e.EXTRACTION_RADIUS_RG,
            "generic_profile": c4e.GENERIC_PROFILE,
            "windows_seconds": WINDOWS_SECONDS,
        },
    )
    _write_json(CONTRACT_PATH, contract)
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)
    _write_json(SUMMARY_PATH, summary)
    implementation = {
        relative: _sha256(ROOT / relative)
        for relative in (
            THIS_RUNNER,
            THIS_TEST,
            c4e3.THIS_RUNNER,
            c4e3.THIS_TEST,
            "src/imri_qpe/layer3_minidisk_1d/causal_inner_embedded_patch.py",
        )
    }
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "CERTIFIED",
            "source_parent_commit": ANALYZED_BASE_COMMIT,
            "source_parent_tree_sha": ANALYZED_BASE_TREE,
            "working_head": _git_value("rev-parse", "HEAD"),
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "input_hashes": {
                "coarse_10ms_arrays": _sha256(COARSE_TEN_ARRAYS),
                "coarse_20ms_arrays": _sha256(COARSE_TWENTY_ARRAYS),
                "middle_6ms_arrays": _sha256(MIDDLE_SIX_ARRAYS),
                "middle_20ms_arrays": _sha256(MIDDLE_TWENTY_ARRAYS),
                "middle_20ms_summary": _sha256(c4e3.SUMMARY_PATH),
                "spatial_checkpoint_manifest": _sha256(c4e.MANIFEST_PATH),
            },
            "implementation_source_hashes": implementation,
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "command": f"PYTHONPATH=src:scripts python {THIS_RUNNER}",
        },
    )
    state = analysis["state"]
    instant = analysis["instantaneous_extraction"]
    cumulative = analysis["cumulative_extraction"]
    means = analysis["window_mean_extraction"]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Coarse-middle 20 ms checkpoint analysis WP10c9d6c7c3b5c4e4",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                "This package executes no trajectory. It compares the committed "
                "coarse and middle generic nonlinear responses on one common parent "
                "state grid and at the certified exterior extraction partition.",
                "",
                "## Two-grid checkpoint screen",
                "",
                f"State maximum/RMS differences are `{state['maximum_normalized_difference']:.6e}` / `{state['rms_normalized_difference']:.6e}`, with response cosine `{state['cross_grid_response_history_cosine']:.9f}` and temporal/spatial ratio `{state['temporal_uncertainty_fraction_of_spatial_difference']:.6e}`.",
                "",
                f"Instantaneous extraction maximum/RMS differences are `{instant['maximum_normalized_difference']:.6e}` / `{instant['rms_normalized_difference']:.6e}`, with response cosine `{instant['cross_grid_response_history_cosine']:.9f}` and temporal/spatial ratio `{instant['temporal_uncertainty_fraction_of_spatial_difference']:.6e}`.",
                "",
                f"Cumulative extraction maximum/RMS differences are `{cumulative['maximum_normalized_difference']:.6e}` / `{cumulative['rms_normalized_difference']:.6e}`, with response cosine `{cumulative['cross_grid_response_history_cosine']:.9f}` and temporal/spatial ratio `{cumulative['temporal_uncertainty_fraction_of_spatial_difference']:.6e}`.",
                "",
                f"Window-mean extraction maximum/RMS differences are `{means['maximum_normalized_difference']:.6e}` / `{means['rms_normalized_difference']:.6e}`, with response cosine `{means['cross_grid_response_history_cosine']:.9f}` and temporal/spatial ratio `{means['temporal_uncertainty_fraction_of_spatial_difference']:.6e}`.",
                "",
                "## Cost decision",
                "",
                f"The checkpoint screen passes: `{screen_passed}`. A full fine generic nonlinear anchor is required: `{analysis['full_fine_generic_anchor_required']}`.",
                "",
                f"Authorized next: `{authorized_next}`.",
                "",
                "A fine level is still required before any measured spatial order or 20 ms spatial certificate. Fine propagation, 50 ms evolution, fixed-Q experiments, and reduced slow evolution remain unauthorized in this package.",
                "",
            )
        ),
        encoding="utf-8",
    )
    names = (
        "analysis_contract.json",
        "config.json",
        "decisive_arrays.npz",
        "provenance.json",
        "summary.json",
    )
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
