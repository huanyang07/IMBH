"""Run the conditional fresh N128 no-tide reference for WP10c7n."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import run_causal_characteristic_extension_wp10c7l as wp10c7l
import run_causal_evolved_spatial_order_wp10c7m as wp10c7m
import run_causal_spatial_balance_adaptive_wp10c7k as wp10c7k
from imri_qpe.layer3_minidisk_1d import (
    CausalFiveFieldAdaptiveBDF2Restart,
    audit_causal_five_field_state_gates,
    causal_five_field_adaptive_bdf2_restarts_equal,
    causal_five_field_bdf_physical_ledger_from_restart,
    causal_five_field_bdf_physical_ledger_relative_defects,
    causal_five_field_profile_fields,
    causal_five_field_regression_seed_parameters,
    causal_restrict_cell_averages,
    causal_spatial_contraction_order,
    evolve_causal_five_field_adaptive_bdf2_campaign,
    load_causal_five_field_adaptive_bdf2_restart,
    make_causal_five_field_regression_context,
    save_causal_five_field_adaptive_bdf2_restart,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "f6d1e296bf4dc8a446e1a967ae85720d45cd4161"
WP10C7M_OUTPUT = (
    ROOT / "outputs/tables/causal_evolved_spatial_order_wp10c7m.json"
)
OUTPUT_CHECKPOINT_DIRECTORY = (
    ROOT / "outputs/checkpoints/causal_five_field_wp10c7n"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/tables/causal_n128_reference_wp10c7n.json"
)
DEFAULT_ARRAYS = (
    ROOT / "outputs/tables/causal_n128_reference_wp10c7n_arrays.npz"
)
N_CELLS = 128
TRAJECTORY_MODES = ("production", "temporal_control")
COMMON_OUTPUTS = wp10c7l.COMMON_OUTPUTS
TARGET_TIME_SECONDS = wp10c7l.TARGET_TIME_SECONDS
CONTROL_MAXIMUM_TIMESTEP_SECONDS = (
    0.5 * wp10c7k.SHARED_PASSING_CEILING_SECONDS
)
TARGET_TIME_RELATIVE_TOLERANCE = wp10c7l.TARGET_TIME_RELATIVE_TOLERANCE
MAXIMUM_PHYSICAL_LEDGER_RELATIVE_DEFECT = (
    wp10c7l.MAXIMUM_PHYSICAL_LEDGER_RELATIVE_DEFECT
)
MAXIMUM_PRODUCTION_TO_CONTROL_JACOBIAN_FRACTION = 0.75
SPATIAL_RESPONSE_GATE = 5.0e-3
PREFERRED_SPATIAL_TOTAL = 2.5e-3
MAXIMUM_RICHARDSON_REMAINDER = 1.25e-3
MINIMUM_SPATIAL_ORDER = 1.8
REPLAY_FROM_LABEL = "t_0p0375"
REPLAY_TO_LABEL = "t_0p05"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Load completed N128 checkpoints and rewrite evidence.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate authorization and construct the fresh N128 seed.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--arrays", type=Path, default=DEFAULT_ARRAYS)
    return parser.parse_args()


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            wp10c7k._plain(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _validate_authorization() -> tuple[dict, str]:
    if not WP10C7M_OUTPUT.exists():
        raise RuntimeError("WP10c7n requires canonical WP10c7m evidence")
    evidence = json.loads(WP10C7M_OUTPUT.read_text(encoding="utf-8"))
    arrays = ROOT / str(evidence.get("artifacts", {}).get("arrays_path", ""))
    if not (
        evidence.get("work_package") == "WP10c7m"
        and evidence.get("decision")
        == "wp10c7m_n128_campaign_authorized"
        and evidence.get("next_authorization")
        == "one_fresh_n128_0p05_campaign_with_temporal_control"
        and evidence.get("authorization", {}).get(
            "n128_campaign_authorized",
            False,
        )
        and evidence.get("authorization", {}).get(
            "conservative_authorization_total",
            np.inf,
        )
        <= wp10c7m.MAXIMUM_AUTHORIZATION_BUDGET
        and arrays.exists()
        and _sha256(arrays)
        == evidence.get("artifacts", {}).get("arrays_sha256")
    ):
        raise RuntimeError("WP10c7m did not authorize WP10c7n")
    return evidence, _sha256(WP10C7M_OUTPUT)


def _initial_bundle() -> dict:
    baseline = make_causal_five_field_regression_context(
        16,
        spatial_reconstruction="plm_smooth",
    )
    seed_parameters = causal_five_field_regression_seed_parameters(
        baseline
    )
    return wp10c7k._initial_bundle(N_CELLS, seed_parameters)


def _controller_config(context, mode: str):
    production = wp10c7k._controller_config(context)
    if mode == "production":
        return production
    if mode == "temporal_control":
        return replace(
            production,
            maximum_dt=CONTROL_MAXIMUM_TIMESTEP_SECONDS,
        ).validated()
    raise ValueError(f"unknown WP10c7n trajectory mode {mode!r}")


def _controller_contract(mode: str) -> dict:
    contract = dict(wp10c7k._controller_contract())
    contract["role"] = mode
    if mode == "temporal_control":
        contract["maximum_timestep_seconds"] = (
            CONTROL_MAXIMUM_TIMESTEP_SECONDS
        )
        contract["reference_policy"] = (
            "same controller with only the maximum timestep halved"
        )
    else:
        contract["reference_policy"] = "unchanged WP10c7k controller"
    return contract


def _initial_restart(
    initial: dict,
    evidence_sha256: str,
    mode: str,
) -> CausalFiveFieldAdaptiveBDF2Restart:
    restart = wp10c7k._initial_adaptive_restart(
        initial,
        evidence_sha256,
    )
    return replace(
        restart,
        provenance={
            "work_package": "WP10c7n",
            "role": "fresh_n128_no_tide_spatial_reference",
            "trajectory_mode": mode,
            "base_commit": BASE_COMMIT,
            "wp10c7m_evidence_sha256": evidence_sha256,
            "n_cells": N_CELLS,
            "target_elapsed_time_seconds": TARGET_TIME_SECONDS,
            "common_output_times_seconds": dict(COMMON_OUTPUTS),
            "spatial_options": dict(wp10c7l.SPATIAL_OPTIONS),
            "initial_state_sha256": initial["vector_sha256"],
            "controller": _controller_contract(mode),
            "segments": [],
        },
    )


def _checkpoint_path(mode: str, label: str) -> Path:
    return (
        OUTPUT_CHECKPOINT_DIRECTORY
        / f"causal_wp10c7n_N128_{mode}_{label}.npz"
    )


def _replay_path() -> Path:
    return (
        OUTPUT_CHECKPOINT_DIRECTORY
        / "causal_wp10c7n_N128_production_replay_t_0p05.npz"
    )


def _output_target(label: str) -> float:
    return dict(COMMON_OUTPUTS)[label]


def _expected_labels(label: str) -> list[str]:
    labels = [name for name, _ in COMMON_OUTPUTS]
    return labels[: labels.index(label) + 1]


def _validate_restart(
    restart: CausalFiveFieldAdaptiveBDF2Restart,
    initial: dict,
    evidence_sha256: str,
    mode: str,
    label: str,
) -> None:
    provenance = restart.provenance
    segments = provenance.get("segments", [])
    if not (
        provenance.get("work_package") == "WP10c7n"
        and provenance.get("role")
        == "fresh_n128_no_tide_spatial_reference"
        and provenance.get("trajectory_mode") == mode
        and provenance.get("wp10c7m_evidence_sha256")
        == evidence_sha256
        and provenance.get("n_cells") == N_CELLS
        and provenance.get("target_elapsed_time_seconds")
        == TARGET_TIME_SECONDS
        and provenance.get("spatial_options")
        == wp10c7l.SPATIAL_OPTIONS
        and provenance.get("initial_state_sha256")
        == initial["vector_sha256"]
        and provenance.get("controller") == _controller_contract(mode)
        and [row.get("label") for row in segments]
        == _expected_labels(label)
        and all(row.get("passed", False) for row in segments)
        and restart.elapsed_time == _output_target(label)
        and restart.next_order == 2
        and audit_causal_five_field_state_gates(
            initial["context"],
            restart.state_vector,
        )["passed"]
    ):
        raise RuntimeError(f"WP10c7n N128 {mode} {label} differs")


def _load_snapshot(
    initial: dict,
    evidence_sha256: str,
    mode: str,
    label: str,
    *,
    path: Path | None = None,
) -> CausalFiveFieldAdaptiveBDF2Restart:
    restart = load_causal_five_field_adaptive_bdf2_restart(
        _checkpoint_path(mode, label) if path is None else path,
        initial["context"],
    )
    _validate_restart(
        restart,
        initial,
        evidence_sha256,
        mode,
        label,
    )
    return restart


def _progress(mode: str, label: str):
    def progress(relative_step, restart, result) -> None:
        print(
            json.dumps(
                {
                    "mode": f"n128_wp10c7n_{mode}_{label}",
                    "relative_accepted_step": int(relative_step),
                    "accepted_steps": int(restart.accepted_steps),
                    "order": int(result.order),
                    "dt_used": float(result.dt_used),
                    "dt_next": float(result.dt_next),
                    "elapsed_time": float(restart.elapsed_time),
                    "audits": int(restart.audit_count),
                    "rejected_attempts": int(
                        restart.rejected_attempts
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    return progress


def _run_segment(
    initial: dict,
    start: CausalFiveFieldAdaptiveBDF2Restart,
    mode: str,
    label: str,
) -> CausalFiveFieldAdaptiveBDF2Restart:
    started = time.perf_counter()
    campaign = evolve_causal_five_field_adaptive_bdf2_campaign(
        initial["context"],
        start,
        _output_target(label),
        _controller_config(initial["context"], mode),
        target_time_relative_tolerance=TARGET_TIME_RELATIVE_TOLERANCE,
        progress=_progress(mode, label),
    )
    wall_seconds = time.perf_counter() - started
    if not campaign.passed:
        raise RuntimeError(
            f"WP10c7n N128 {mode} {label} failed: "
            f"{campaign.message}"
        )
    summary = wp10c7k._segment_summary(label, start, campaign)
    summary["wall_seconds"] = wall_seconds
    return wp10c7k._append_segment(campaign.restart, summary)


def _run_or_load_segment(
    initial: dict,
    evidence_sha256: str,
    start: CausalFiveFieldAdaptiveBDF2Restart,
    mode: str,
    label: str,
    *,
    force: bool,
) -> tuple[CausalFiveFieldAdaptiveBDF2Restart, bool]:
    path = _checkpoint_path(mode, label)
    if path.exists() and not force:
        return (
            _load_snapshot(
                initial,
                evidence_sha256,
                mode,
                label,
            ),
            True,
        )
    state = _run_segment(initial, start, mode, label)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_causal_five_field_adaptive_bdf2_restart(
        path,
        initial["context"],
        state,
    )
    restored = _load_snapshot(
        initial,
        evidence_sha256,
        mode,
        label,
    )
    bitwise = causal_five_field_adaptive_bdf2_restarts_equal(
        state,
        restored,
    )
    if not bitwise:
        raise RuntimeError(f"WP10c7n N128 {mode} {label} is not bitwise")
    return restored, bitwise


def _load_wp10c7l_snapshots(
    initial: dict,
) -> tuple[dict, dict, str]:
    wp10c7k_evidence, wp10c7k_sha256 = wp10c7l._validate_wp10c7k()
    parent = wp10c7l._parent_checkpoint_entry(wp10c7k_evidence, 64)
    snapshots = {
        mode: {
            label: wp10c7l._load_snapshot(
                initial,
                wp10c7k_sha256,
                parent,
                mode,
                label,
            )
            for label, _ in COMMON_OUTPUTS
        }
        for mode in TRAJECTORY_MODES
    }
    evidence = json.loads(
        wp10c7l.WP10C7K_OUTPUT.read_text(encoding="utf-8")
    )
    return snapshots, evidence, wp10c7k_sha256


def _profile_response(initial: dict, vector: np.ndarray) -> dict:
    baseline = causal_five_field_profile_fields(
        initial["context"],
        initial["vector"],
    )
    current = causal_five_field_profile_fields(
        initial["context"],
        vector,
    )
    return {
        name: np.asarray(current[name] - baseline[name], dtype=float)
        for name in baseline
    }


def _n64_temporal_uncertainty(
    wp10c7l_evidence: dict,
    label: str,
) -> float:
    row = wp10c7l_evidence["common_time_contract"][label]
    return float(
        row["inherited_wp10c7k_temporal_uncertainty"]["64"]
        + row["extension_temporal_uncertainty"]["64"]
    )


def _common_time_row(
    label: str,
    wp10c7l_evidence: dict,
    initial64: dict,
    initial128: dict,
    n64_snapshots: dict,
    n128_snapshots: dict,
) -> tuple[dict, dict]:
    temporal, temporal_arrays = wp10c7l._accumulated_temporal_row(
        initial128,
        n128_snapshots["production"][label],
        n128_snapshots["temporal_control"][label],
    )
    coarse = _profile_response(
        initial64,
        n64_snapshots["production"][label].state_vector,
    )
    fine = _profile_response(
        initial128,
        n128_snapshots["production"][label].state_vector,
    )
    differences = {}
    arrays = {
        f"{label}_n128_production_control_{name}": values
        for name, values in temporal_arrays.items()
    }
    for name, coarse_values in coarse.items():
        restricted = causal_restrict_cell_averages(
            initial64["context"].grid,
            initial128["context"].grid,
            fine[name],
        )
        differences[name] = coarse_values - restricted
        arrays[f"{label}_n64_{name}"] = coarse_values
        arrays[f"{label}_restricted_n128_{name}"] = restricted
        arrays[f"{label}_spatial_difference_{name}"] = differences[name]
    profiles = wp10c7k._profile_difference_rows(
        initial64["context"],
        differences,
    )
    raw = float(
        profiles["log_h_over_r"]["full_domain"][
            "maximum_absolute_difference"
        ]
    )
    n64_uncertainty = _n64_temporal_uncertainty(
        wp10c7l_evidence,
        label,
    )
    n128_uncertainty = float(
        temporal["estimated_production_temporal_error"][
            "maximum_log_h_over_r_profile"
        ]
    )
    conservative = raw + n64_uncertainty + n128_uncertainty
    coarse_difference = float(
        wp10c7l_evidence["common_time_contract"][label][
            "raw_n32_n64_log_h_over_r_difference"
        ]
    )
    order = causal_spatial_contraction_order(coarse_difference, raw)
    richardson = raw / (2.0**order - 1.0)
    row = {
        "label": label,
        "elapsed_time_seconds": _output_target(label),
        "n128_temporal_control": temporal,
        "spatial_profile_differences": profiles,
        "raw_n32_n64_log_h_over_r_difference": coarse_difference,
        "raw_n64_n128_log_h_over_r_difference": raw,
        "observed_spatial_order": order,
        "n64_temporal_uncertainty": n64_uncertainty,
        "n128_temporal_uncertainty": n128_uncertainty,
        "conservative_n64_n128_log_h_over_r_total": conservative,
        "richardson_n128_to_continuum_remainder": richardson,
        "spatial_gate": SPATIAL_RESPONSE_GATE,
        "preferred_half_gate": PREFERRED_SPATIAL_TOTAL,
        "maximum_richardson_remainder": MAXIMUM_RICHARDSON_REMAINDER,
        "minimum_spatial_order": MINIMUM_SPATIAL_ORDER,
        "temporal_passed": temporal["gate_audit"]["passed"],
        "original_spatial_contract_passed": bool(
            conservative <= SPATIAL_RESPONSE_GATE
        ),
        "preferred_half_gate_passed": bool(
            conservative <= PREFERRED_SPATIAL_TOTAL
        ),
        "order_passed": bool(order >= MINIMUM_SPATIAL_ORDER),
        "richardson_passed": bool(
            richardson <= MAXIMUM_RICHARDSON_REMAINDER
        ),
    }
    row["passed"] = bool(
        row["temporal_passed"]
        and row["original_spatial_contract_passed"]
        and row["order_passed"]
        and row["richardson_passed"]
    )
    return row, arrays


def _campaign_summary(
    initial: dict,
    start: CausalFiveFieldAdaptiveBDF2Restart,
    final: CausalFiveFieldAdaptiveBDF2Restart,
    mode: str,
    roundtrips: dict,
) -> dict:
    segments = list(final.provenance["segments"])
    attempts = [
        attempt
        for segment in segments
        for record in segment["records"]
        for attempt in record["attempts"]
    ]
    accepted = [row for row in attempts if row["accepted"]]
    audits = [
        row["independent_audit"]
        for row in attempts
        if row["independent_audit"] is not None
    ]
    local_passed = all(
        row["local_gate_audit"] is None
        or row["local_gate_audit"]["passed"]
        for row in accepted
    )
    audit_passed = bool(audits and all(row["passed"] for row in audits))
    ledger = causal_five_field_bdf_physical_ledger_from_restart(final)
    defects = causal_five_field_bdf_physical_ledger_relative_defects(
        ledger
    )
    maximum_ledger = float(np.max(defects))
    state_gates = audit_causal_five_field_state_gates(
        initial["context"],
        final.state_vector,
    )
    return {
        "segments": segments,
        "accepted_steps": int(final.accepted_steps - start.accepted_steps),
        "accepted_bdf2_steps": int(
            final.accepted_bdf2_steps - start.accepted_bdf2_steps
        ),
        "rejected_attempts": int(
            final.rejected_attempts - start.rejected_attempts
        ),
        "audit_count": int(final.audit_count - start.audit_count),
        "minimum_dt_used": min(
            segment["minimum_dt_used"] for segment in segments
        ),
        "maximum_dt_used": max(
            segment["maximum_dt_used"] for segment in segments
        ),
        "wall_seconds": float(
            sum(segment.get("wall_seconds", 0.0) for segment in segments)
        ),
        "work": wp10c7k._sum_work(segments),
        "local_estimator_passed": local_passed,
        "independent_audits": {
            "count": len(audits),
            "maximum_normalized_error": max(
                float(
                    row["temporal_gate_audit"][
                        "maximum_normalized_error"
                    ]
                )
                for row in audits
            ),
            "passed": audit_passed,
        },
        "physical_ledger": {
            "component_relative_defects": wp10c7k._plain(defects),
            "maximum_relative_defect": maximum_ledger,
            "gate": MAXIMUM_PHYSICAL_LEDGER_RELATIVE_DEFECT,
            "passed": bool(
                maximum_ledger
                <= MAXIMUM_PHYSICAL_LEDGER_RELATIVE_DEFECT
            ),
        },
        "state_gates": state_gates,
        "clock_summary": {
            "initial": wp10c7l._clock_summary(
                initial,
                start.state_vector,
            ),
            "endpoint": wp10c7l._clock_summary(
                initial,
                final.state_vector,
            ),
        },
        "snapshot_limiters": {
            label: wp10c7l._limiter_summary(
                initial,
                _load_snapshot(
                    initial,
                    final.provenance["wp10c7m_evidence_sha256"],
                    mode,
                    label,
                ).state_vector,
            )
            for label, _ in COMMON_OUTPUTS
        },
        "snapshot_checkpoints": {
            label: {
                "path": _relative(_checkpoint_path(mode, label)),
                "sha256": _sha256(_checkpoint_path(mode, label)),
                "roundtrip_bitwise": roundtrips[label],
            }
            for label, _ in COMMON_OUTPUTS
        },
        "passed": bool(
            all(segment["passed"] for segment in segments)
            and local_passed
            and audit_passed
            and maximum_ledger
            <= MAXIMUM_PHYSICAL_LEDGER_RELATIVE_DEFECT
            and state_gates["passed"]
            and all(roundtrips.values())
        ),
    }


def _run_replay(
    initial: dict,
    evidence_sha256: str,
    snapshots: dict,
    *,
    force: bool,
) -> dict:
    path = _replay_path()
    if path.exists() and not force:
        replay = _load_snapshot(
            initial,
            evidence_sha256,
            "production",
            REPLAY_TO_LABEL,
            path=path,
        )
    else:
        replay = _run_segment(
            initial,
            snapshots[REPLAY_FROM_LABEL],
            "production",
            REPLAY_TO_LABEL,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        save_causal_five_field_adaptive_bdf2_restart(
            path,
            initial["context"],
            replay,
        )
        replay = _load_snapshot(
            initial,
            evidence_sha256,
            "production",
            REPLAY_TO_LABEL,
            path=path,
        )

    def deterministic_restart(
        restart: CausalFiveFieldAdaptiveBDF2Restart,
    ) -> CausalFiveFieldAdaptiveBDF2Restart:
        provenance = dict(restart.provenance)
        segments = []
        for row in provenance.get("segments", []):
            segment = dict(row)
            segment.pop("wall_seconds", None)
            segments.append(segment)
        provenance["segments"] = segments
        return replace(restart, provenance=provenance)

    bitwise = causal_five_field_adaptive_bdf2_restarts_equal(
        deterministic_restart(snapshots[REPLAY_TO_LABEL]),
        deterministic_restart(replay),
    )
    if not bitwise:
        raise RuntimeError("WP10c7n N128 replay differs")
    return {
        "path": _relative(path),
        "sha256": _sha256(path),
        "from_label": REPLAY_FROM_LABEL,
        "to_label": REPLAY_TO_LABEL,
        "endpoint_replay_bitwise": bitwise,
        "comparison_policy": (
            "bitwise equality of state, BDF history, counters, ledgers, "
            "controller state, and deterministic provenance after removing "
            "nondeterministic segment wall_seconds telemetry"
        ),
    }


def _aggregate(
    output_path: Path,
    arrays_path: Path,
    authorization: dict,
    authorization_sha256: str,
    wp10c7l_evidence: dict,
    initial64: dict,
    initial128: dict,
    starts: dict,
    snapshots: dict,
    roundtrips: dict,
    replay: dict,
) -> dict:
    n64_snapshots, _, _ = _load_wp10c7l_snapshots(initial64)
    common_rows = {}
    arrays = {}
    for label, _ in COMMON_OUTPUTS:
        row, row_arrays = _common_time_row(
            label,
            wp10c7l_evidence,
            initial64,
            initial128,
            n64_snapshots,
            snapshots,
        )
        common_rows[label] = row
        arrays.update(row_arrays)

    campaigns = {
        mode: _campaign_summary(
            initial128,
            starts[mode],
            snapshots[mode][COMMON_OUTPUTS[-1][0]],
            mode,
            roundtrips[mode],
        )
        for mode in TRAJECTORY_MODES
    }
    production_work = campaigns["production"]["work"]
    control_work = campaigns["temporal_control"]["work"]
    jacobian_fraction = (
        production_work["jacobian_evaluations"]
        / control_work["jacobian_evaluations"]
    )
    work_audit = {
        "production": production_work,
        "temporal_control": control_work,
        "production_to_control_jacobian_fraction": jacobian_fraction,
        "production_to_control_function_fraction": (
            production_work["function_evaluations"]
            / control_work["function_evaluations"]
        ),
        "maximum_jacobian_fraction": (
            MAXIMUM_PRODUCTION_TO_CONTROL_JACOBIAN_FRACTION
        ),
        "passed": bool(
            jacobian_fraction
            <= MAXIMUM_PRODUCTION_TO_CONTROL_JACOBIAN_FRACTION
        ),
    }
    source_audit = wp10c7k._source_restriction_audit(
        initial64["context"],
        initial128["context"],
    )
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **arrays)

    all_common_passed = all(row["passed"] for row in common_rows.values())
    campaign_passed = all(row["passed"] for row in campaigns.values())
    passed = bool(
        source_audit["passed"]
        and all_common_passed
        and campaign_passed
        and work_audit["passed"]
        and replay["endpoint_replay_bitwise"]
    )
    payload = {
        "work_package": "WP10c7n",
        "base_commit": BASE_COMMIT,
        "scope": (
            "fresh continuum-seeded N128 no-tide production and "
            "half-ceiling temporal-control trajectories to 0.05 s, "
            "measured against the certified N64 trajectory"
        ),
        "spatial_options": dict(wp10c7l.SPATIAL_OPTIONS),
        "physics": {
            "stream": "exact circularized source",
            "tide": "off",
            "wind": "off",
        },
        "target_elapsed_time_seconds": TARGET_TIME_SECONDS,
        "common_output_times_seconds": dict(COMMON_OUTPUTS),
        "wp10c7m_authorization": {
            "path": _relative(WP10C7M_OUTPUT),
            "sha256": authorization_sha256,
            "decision": authorization["decision"],
            "conservative_projected_total": authorization[
                "authorization"
            ]["conservative_authorization_total"],
        },
        "initialization": {
            "n128_initial_state_sha256": initial128["vector_sha256"],
            "n128_state_gates": initial128["state_gates"],
            "n128_state_summary": initial128["state_summary"],
            "n128_throughput_ratio": initial128["throughput_ratio"],
            "n128_tangent_defects": initial128["tangent_defects"],
            "policy": (
                "fresh deterministic physical seed and N128 consistent "
                "tangent; no remapped N64 evolved state or BDF history"
            ),
        },
        "source_restriction_audit": source_audit,
        "temporal_reference_contract": {
            "production_controller": _controller_contract("production"),
            "temporal_control_controller": _controller_contract(
                "temporal_control"
            ),
            "second_order_error_safety_factor": (
                wp10c7l.SECOND_ORDER_ERROR_SAFETY_FACTOR
            ),
            "reserved_temporal_gate_fraction": (
                wp10c7l.ACCUMULATED_TEMPORAL_GATE_FRACTION
            ),
        },
        "campaigns": campaigns,
        "common_time_contract": common_rows,
        "work_audit": work_audit,
        "restart_replay": replay,
        "primary_log_h_over_r_contract": {
            "maximum_raw_n64_n128_difference": max(
                row["raw_n64_n128_log_h_over_r_difference"]
                for row in common_rows.values()
            ),
            "maximum_conservative_n64_n128_total": max(
                row["conservative_n64_n128_log_h_over_r_total"]
                for row in common_rows.values()
            ),
            "minimum_observed_spatial_order": min(
                row["observed_spatial_order"]
                for row in common_rows.values()
            ),
            "maximum_richardson_n128_remainder": max(
                row["richardson_n128_to_continuum_remainder"]
                for row in common_rows.values()
            ),
            "spatial_gate": SPATIAL_RESPONSE_GATE,
            "preferred_half_gate": PREFERRED_SPATIAL_TOTAL,
            "all_common_times_passed": all_common_passed,
        },
        "gates": {
            "source_restriction_passed": source_audit["passed"],
            "both_n128_temporal_modes_passed": campaign_passed,
            "all_common_time_contracts_passed": all_common_passed,
            "production_work_advantage_passed": work_audit["passed"],
            "production_replay_bitwise": replay[
                "endpoint_replay_bitwise"
            ],
            "wp10c7n_passed": passed,
        },
        "decision": (
            "wp10c7n_n128_0p05_reference_certified"
            if passed
            else "wp10c7n_n128_0p05_reference_failed"
        ),
        "next_authorization": (
            "selected_state_slow_mode_audit"
            if passed
            else "diagnose_measured_n64_n128_failure"
        ),
        "artifacts": {
            "arrays_path": _relative(arrays_path),
            "arrays_sha256": _sha256(arrays_path),
        },
    }
    _write_json(output_path, payload)
    return payload


def main() -> None:
    args = _arguments()
    output_path = _absolute(args.output)
    arrays_path = _absolute(args.arrays)
    authorization, authorization_sha256 = _validate_authorization()
    wp10c7l_evidence, _, initial, _ = wp10c7m._load_inputs()
    initial64 = initial[64]
    initial128 = _initial_bundle()
    source_audit = wp10c7k._source_restriction_audit(
        initial64["context"],
        initial128["context"],
    )
    starts = {
        mode: _initial_restart(
            initial128,
            authorization_sha256,
            mode,
        )
        for mode in TRAJECTORY_MODES
    }
    if args.preflight:
        print(
            json.dumps(
                {
                    "work_package": "WP10c7n",
                    "preflight_passed": bool(
                        source_audit["passed"]
                        and initial128["state_gates"]["passed"]
                        and all(
                            start.elapsed_time == 0.0
                            and start.next_order == 1
                            for start in starts.values()
                        )
                    ),
                    "wp10c7m_evidence_sha256": authorization_sha256,
                    "n128_initial_state_sha256": initial128[
                        "vector_sha256"
                    ],
                    "source_restriction_audit": source_audit,
                    "target_elapsed_time_seconds": TARGET_TIME_SECONDS,
                },
                sort_keys=True,
            )
        )
        return

    snapshots = {mode: {} for mode in TRAJECTORY_MODES}
    roundtrips = {mode: {} for mode in TRAJECTORY_MODES}
    states = dict(starts)
    for label, _ in COMMON_OUTPUTS:
        for mode in TRAJECTORY_MODES:
            path = _checkpoint_path(mode, label)
            if args.aggregate_only:
                if not path.exists():
                    raise RuntimeError(
                        f"missing WP10c7n N128 {mode} {label}"
                    )
                state = _load_snapshot(
                    initial128,
                    authorization_sha256,
                    mode,
                    label,
                )
                roundtrip = True
            else:
                state, roundtrip = _run_or_load_segment(
                    initial128,
                    authorization_sha256,
                    states[mode],
                    mode,
                    label,
                    force=args.force,
                )
            states[mode] = state
            snapshots[mode][label] = state
            roundtrips[mode][label] = roundtrip
        print(
            json.dumps(
                {
                    "work_package": "WP10c7n",
                    "completed_output": label,
                    "elapsed_time_seconds": _output_target(label),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    replay = _run_replay(
        initial128,
        authorization_sha256,
        snapshots["production"],
        force=args.force,
    )
    payload = _aggregate(
        output_path,
        arrays_path,
        authorization,
        authorization_sha256,
        wp10c7l_evidence,
        initial64,
        initial128,
        starts,
        snapshots,
        roundtrips,
        replay,
    )
    final = payload["common_time_contract"]["t_0p05"]
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "raw_n64_n128_log_h_over_r_difference": final[
                    "raw_n64_n128_log_h_over_r_difference"
                ],
                "conservative_n64_n128_total": final[
                    "conservative_n64_n128_log_h_over_r_total"
                ],
                "observed_spatial_order": final[
                    "observed_spatial_order"
                ],
                "richardson_n128_remainder": final[
                    "richardson_n128_to_continuum_remainder"
                ],
                "output": _relative(output_path),
                "arrays": _relative(arrays_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
