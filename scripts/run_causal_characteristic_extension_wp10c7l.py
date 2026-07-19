"""Run the matched N32/N64 no-tide WP10c7l extension to 0.05 s."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

import run_causal_spatial_balance_adaptive_wp10c7k as wp10c7k
from imri_qpe.layer3_minidisk_1d import (
    CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1,
    CausalFiveFieldAdaptiveBDF2Restart,
    audit_causal_five_field_state_gates,
    causal_five_field_adaptive_bdf2_restarts_equal,
    causal_five_field_bdf_physical_ledger_from_restart,
    causal_five_field_bdf_physical_ledger_relative_defects,
    causal_five_field_local_timescale_audit,
    causal_five_field_profile_fields,
    causal_five_field_reconstruct_face_charts,
    causal_five_field_regression_seed_parameters,
    causal_five_field_state_summary,
    causal_five_field_temporal_error_ratio,
    causal_restrict_cell_averages,
    compare_causal_five_field_endpoint_vectors,
    evolve_causal_five_field_adaptive_bdf2_campaign,
    load_causal_five_field_adaptive_bdf2_restart,
    make_causal_five_field_regression_context,
    save_causal_five_field_adaptive_bdf2_restart,
    unpack_causal_five_field_state,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "9952692e49795a0c2a75558b26f4c260117b41a4"
WP10C7K_OUTPUT = (
    ROOT
    / "outputs/tables/causal_spatial_balance_adaptive_wp10c7k.json"
)
OUTPUT_CHECKPOINT_DIRECTORY = (
    ROOT / "outputs/checkpoints/causal_five_field_wp10c7l"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/tables/causal_characteristic_extension_wp10c7l.json"
)
DEFAULT_ARRAYS = (
    ROOT
    / "outputs/tables/causal_characteristic_extension_wp10c7l_arrays.npz"
)
SPATIAL_OPTIONS = dict(wp10c7k.SPATIAL_OPTIONS)
RESOLUTIONS = (32, 64)
TRAJECTORY_MODES = ("production", "temporal_control")
COMMON_OUTPUTS = (
    ("t_0p025", 2.5e-2),
    ("t_0p0375", 3.75e-2),
    ("t_0p05", 5.0e-2),
)
START_TIME_SECONDS = wp10c7k.TARGET_DURATION_SECONDS
TARGET_TIME_SECONDS = 5.0e-2
CONTROL_MAXIMUM_TIMESTEP_SECONDS = (
    0.5 * wp10c7k.SHARED_PASSING_CEILING_SECONDS
)
TARGET_TIME_RELATIVE_TOLERANCE = 5.0e-14
ACCUMULATED_TEMPORAL_GATE_FRACTION = 0.25
SECOND_ORDER_ERROR_SAFETY_FACTOR = 4.0 / 3.0
MAXIMUM_PRODUCTION_TO_CONTROL_JACOBIAN_FRACTION = 0.75
MAXIMUM_PHYSICAL_LEDGER_RELATIVE_DEFECT = 1.0e-3
SPATIAL_RESPONSE_GATE = 5.0e-3
REPLAY_MODE = "production"
COOLING_INNER_CUTOFF_RG = 6.0


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Load all completed checkpoints and rewrite canonical evidence.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate WP10c7k evidence and restart inputs without evolving.",
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


def _controller_config(context, mode: str):
    production = wp10c7k._controller_config(context)
    if mode == "production":
        return production
    if mode == "temporal_control":
        return replace(
            production,
            maximum_dt=CONTROL_MAXIMUM_TIMESTEP_SECONDS,
        ).validated()
    raise ValueError(f"unknown WP10c7l trajectory mode {mode!r}")


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
    elif mode == "production":
        contract["reference_policy"] = "unchanged WP10c7k controller"
    else:
        raise ValueError(f"unknown WP10c7l trajectory mode {mode!r}")
    return contract


def _validate_wp10c7k() -> tuple[dict, str]:
    if not WP10C7K_OUTPUT.exists():
        raise RuntimeError("WP10c7l requires canonical WP10c7k evidence")
    evidence = json.loads(WP10C7K_OUTPUT.read_text(encoding="utf-8"))
    artifact = evidence.get("artifacts", {})
    arrays_path = ROOT / str(artifact.get("arrays_path", ""))
    if not (
        evidence.get("work_package") == "WP10c7k"
        and evidence.get("decision")
        == "wp10c7k_matched_adaptive_bdf2_certified"
        and evidence.get("next_authorization")
        == "no_tide_duration_ladder_characteristic_clock"
        and evidence.get("spatial_options") == SPATIAL_OPTIONS
        and evidence.get("gates", {}).get("wp10c7k_passed", False)
        and evidence.get("target_duration_seconds")
        == START_TIME_SECONDS
        and arrays_path.exists()
        and _sha256(arrays_path) == artifact.get("arrays_sha256")
    ):
        raise RuntimeError("WP10c7k did not authorize WP10c7l")
    return evidence, _sha256(WP10C7K_OUTPUT)


def _initial_bundles(evidence: dict) -> dict[int, dict]:
    baseline = make_causal_five_field_regression_context(
        16,
        spatial_reconstruction="plm_smooth",
    )
    seed_parameters = causal_five_field_regression_seed_parameters(
        baseline
    )
    bundles = {
        n_cells: wp10c7k._initial_bundle(n_cells, seed_parameters)
        for n_cells in RESOLUTIONS
    }
    expected = evidence["initialization"]["meshes"]
    if not all(
        bundles[n_cells]["vector_sha256"]
        == expected[str(n_cells)]["state_vector_sha256"]
        for n_cells in RESOLUTIONS
    ):
        raise RuntimeError("WP10c7l fresh initial vectors differ")
    return bundles


def _parent_checkpoint_entry(evidence: dict, n_cells: int) -> dict:
    return evidence["adaptive_campaigns"][str(n_cells)][
        "restart_replay"
    ]["snapshot_checkpoints"]["t_1"]


def _load_parent_restart(
    evidence: dict,
    initial: dict,
) -> tuple[CausalFiveFieldAdaptiveBDF2Restart, dict]:
    n_cells = initial["state"].n_cells
    checkpoint = _parent_checkpoint_entry(evidence, n_cells)
    path = ROOT / checkpoint["path"]
    if not (
        path.exists()
        and _sha256(path) == checkpoint["sha256"]
        and checkpoint["roundtrip_bitwise"]
    ):
        raise RuntimeError(f"WP10c7k N{n_cells} parent checkpoint differs")
    restart = load_causal_five_field_adaptive_bdf2_restart(
        path,
        initial["context"],
    )
    provenance = restart.provenance
    if not (
        provenance.get("work_package") == "WP10c7k"
        and provenance.get("role")
        == "matched_spatial_balance_adaptive_bdf2"
        and provenance.get("spatial_options") == SPATIAL_OPTIONS
        and provenance.get("initial_state_sha256")
        == initial["vector_sha256"]
        and restart.elapsed_time == START_TIME_SECONDS
        and restart.next_order == 2
        and audit_causal_five_field_state_gates(
            initial["context"],
            restart.state_vector,
        )["passed"]
    ):
        raise RuntimeError(f"WP10c7k N{n_cells} parent restart differs")
    return restart, checkpoint


def _extension_start(
    parent: CausalFiveFieldAdaptiveBDF2Restart,
    initial: dict,
    evidence_sha256: str,
    checkpoint: dict,
    mode: str,
) -> CausalFiveFieldAdaptiveBDF2Restart:
    n_cells = initial["state"].n_cells
    maximum_dt = _controller_config(
        initial["context"],
        mode,
    ).maximum_dt
    return replace(
        parent,
        dt_next=min(parent.dt_next, maximum_dt),
        provenance={
            "work_package": "WP10c7l",
            "role": "matched_no_tide_characteristic_extension",
            "trajectory_mode": mode,
            "base_commit": BASE_COMMIT,
            "wp10c7k_evidence_sha256": evidence_sha256,
            "parent_checkpoint": dict(checkpoint),
            "parent_elapsed_time_seconds": START_TIME_SECONDS,
            "n_cells": n_cells,
            "target_elapsed_time_seconds": TARGET_TIME_SECONDS,
            "common_output_times_seconds": {
                label: target for label, target in COMMON_OUTPUTS
            },
            "spatial_options": dict(SPATIAL_OPTIONS),
            "initial_state_sha256": initial["vector_sha256"],
            "controller": _controller_contract(mode),
            "segments": [],
        },
    )


def _checkpoint_path(n_cells: int, mode: str, label: str) -> Path:
    return (
        OUTPUT_CHECKPOINT_DIRECTORY
        / (
            f"causal_wp10c7l_N{n_cells:03d}_{mode}_"
            f"{label}.npz"
        )
    )


def _replay_path(n_cells: int, label: str) -> Path:
    return (
        OUTPUT_CHECKPOINT_DIRECTORY
        / f"causal_wp10c7l_N{n_cells:03d}_production_replay_{label}.npz"
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
    checkpoint: dict,
    mode: str,
    label: str,
) -> None:
    provenance = restart.provenance
    segments = provenance.get("segments", [])
    if not (
        provenance.get("work_package") == "WP10c7l"
        and provenance.get("role")
        == "matched_no_tide_characteristic_extension"
        and provenance.get("trajectory_mode") == mode
        and provenance.get("wp10c7k_evidence_sha256")
        == evidence_sha256
        and provenance.get("parent_checkpoint") == checkpoint
        and provenance.get("parent_elapsed_time_seconds")
        == START_TIME_SECONDS
        and provenance.get("n_cells") == initial["state"].n_cells
        and provenance.get("target_elapsed_time_seconds")
        == TARGET_TIME_SECONDS
        and provenance.get("spatial_options") == SPATIAL_OPTIONS
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
        raise RuntimeError(
            f"WP10c7l N{initial['state'].n_cells} {mode} {label} differs"
        )


def _load_snapshot(
    initial: dict,
    evidence_sha256: str,
    checkpoint: dict,
    mode: str,
    label: str,
) -> CausalFiveFieldAdaptiveBDF2Restart:
    restart = load_causal_five_field_adaptive_bdf2_restart(
        _checkpoint_path(initial["state"].n_cells, mode, label),
        initial["context"],
    )
    _validate_restart(
        restart,
        initial,
        evidence_sha256,
        checkpoint,
        mode,
        label,
    )
    return restart


def _progress(n_cells: int, mode: str, label: str):
    def progress(relative_step, restart, result) -> None:
        print(
            json.dumps(
                {
                    "mode": (
                        f"n{n_cells}_wp10c7l_{mode}_{label}"
                    ),
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
    campaign = evolve_causal_five_field_adaptive_bdf2_campaign(
        initial["context"],
        start,
        _output_target(label),
        _controller_config(initial["context"], mode),
        target_time_relative_tolerance=TARGET_TIME_RELATIVE_TOLERANCE,
        progress=_progress(initial["state"].n_cells, mode, label),
    )
    if not campaign.passed:
        raise RuntimeError(
            f"WP10c7l N{initial['state'].n_cells} {mode} {label} "
            f"failed: {campaign.message}"
        )
    return wp10c7k._append_segment(
        campaign.restart,
        wp10c7k._segment_summary(label, start, campaign),
    )


def _run_or_load_segment(
    initial: dict,
    evidence_sha256: str,
    checkpoint: dict,
    start: CausalFiveFieldAdaptiveBDF2Restart,
    mode: str,
    label: str,
    *,
    force: bool,
) -> tuple[CausalFiveFieldAdaptiveBDF2Restart, bool]:
    n_cells = initial["state"].n_cells
    path = _checkpoint_path(n_cells, mode, label)
    if path.exists() and not force:
        return (
            _load_snapshot(
                initial,
                evidence_sha256,
                checkpoint,
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
        checkpoint,
        mode,
        label,
    )
    roundtrip = causal_five_field_adaptive_bdf2_restarts_equal(
        state,
        restored,
    )
    if not roundtrip:
        raise RuntimeError(
            f"WP10c7l N{n_cells} {mode} {label} is not bitwise"
        )
    return restored, roundtrip


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


def _accumulated_temporal_row(
    initial: dict,
    production: CausalFiveFieldAdaptiveBDF2Restart,
    control: CausalFiveFieldAdaptiveBDF2Restart,
) -> tuple[dict, dict[str, np.ndarray]]:
    cutoff = (
        COOLING_INNER_CUTOFF_RG
        * initial["context"].grid.gravitational_radius
    )
    raw = compare_causal_five_field_endpoint_vectors(
        initial["context"],
        initial["vector"],
        production.state_vector,
        control.state_vector,
        cooling_inner_cutoff=cutoff,
    )
    estimated = {
        name: SECOND_ORDER_ERROR_SAFETY_FACTOR * float(raw[name])
        for name in CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1
    }
    gates = {
        name: (
            ACCUMULATED_TEMPORAL_GATE_FRACTION * float(value)
        )
        for name, value in (
            CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1.items()
        )
    }
    audit = causal_five_field_temporal_error_ratio(estimated, gates)
    production_profiles = _profile_response(
        initial,
        production.state_vector,
    )
    control_profiles = _profile_response(
        initial,
        control.state_vector,
    )
    differences = {
        name: production_profiles[name] - control_profiles[name]
        for name in production_profiles
    }
    return (
        {
            "raw_production_to_half_ceiling_control": raw,
            "second_order_safety_factor": (
                SECOND_ORDER_ERROR_SAFETY_FACTOR
            ),
            "estimated_production_temporal_error": estimated,
            "reserved_gate_fraction": (
                ACCUMULATED_TEMPORAL_GATE_FRACTION
            ),
            "gate_audit": audit,
            "profile_differences": wp10c7k._profile_difference_rows(
                initial["context"],
                differences,
            ),
        },
        differences,
    )


def _inherited_temporal_uncertainty(
    evidence: dict,
    n_cells: int,
) -> float:
    row = evidence["primary_log_h_over_r_contract"][
        "snapshot_rows"
    ]["t_1"]
    return float(
        row[f"n{n_cells}_adaptive_to_fixed_s64"]
        + row[f"n{n_cells}_fixed_reference_uncertainty"]
    )


def _common_time_row(
    label: str,
    evidence: dict,
    initial: dict,
    snapshots: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    temporal = {}
    arrays = {}
    for n_cells in RESOLUTIONS:
        row, differences = _accumulated_temporal_row(
            initial[n_cells],
            snapshots[(n_cells, "production")][label],
            snapshots[(n_cells, "temporal_control")][label],
        )
        temporal[n_cells] = row
        arrays.update(
            {
                f"{label}_n{n_cells}_production_control_{name}": values
                for name, values in differences.items()
            }
        )

    coarse = _profile_response(
        initial[32],
        snapshots[(32, "production")][label].state_vector,
    )
    fine = _profile_response(
        initial[64],
        snapshots[(64, "production")][label].state_vector,
    )
    spatial_differences = {}
    for name, coarse_values in coarse.items():
        restricted = causal_restrict_cell_averages(
            initial[32]["context"].grid,
            initial[64]["context"].grid,
            fine[name],
        )
        spatial_differences[name] = coarse_values - restricted
        arrays[f"{label}_n32_{name}"] = coarse_values
        arrays[f"{label}_restricted_n64_{name}"] = restricted
        arrays[f"{label}_spatial_difference_{name}"] = (
            spatial_differences[name]
        )
    profiles = wp10c7k._profile_difference_rows(
        initial[32]["context"],
        spatial_differences,
    )
    raw_spatial = profiles["log_h_over_r"]["full_domain"][
        "maximum_absolute_difference"
    ]
    inherited = {
        n_cells: _inherited_temporal_uncertainty(
            evidence,
            n_cells,
        )
        for n_cells in RESOLUTIONS
    }
    extension_temporal = {
        n_cells: temporal[n_cells][
            "estimated_production_temporal_error"
        ]["maximum_log_h_over_r_profile"]
        for n_cells in RESOLUTIONS
    }
    conservative = float(
        raw_spatial
        + sum(inherited.values())
        + sum(extension_temporal.values())
    )
    temporal_passed = all(
        temporal[n_cells]["gate_audit"]["passed"]
        for n_cells in RESOLUTIONS
    )
    row = {
        "label": label,
        "elapsed_time_seconds": _output_target(label),
        "extension_duration_seconds": (
            _output_target(label) - START_TIME_SECONDS
        ),
        "temporal_control": {
            str(n_cells): temporal[n_cells]
            for n_cells in RESOLUTIONS
        },
        "spatial_profile_differences": profiles,
        "raw_n32_n64_log_h_over_r_difference": raw_spatial,
        "inherited_wp10c7k_temporal_uncertainty": {
            str(n_cells): inherited[n_cells]
            for n_cells in RESOLUTIONS
        },
        "extension_temporal_uncertainty": {
            str(n_cells): extension_temporal[n_cells]
            for n_cells in RESOLUTIONS
        },
        "conservative_log_h_over_r_total": conservative,
        "spatial_gate": SPATIAL_RESPONSE_GATE,
        "temporal_passed": temporal_passed,
        "raw_spatial_passed": bool(
            raw_spatial <= SPATIAL_RESPONSE_GATE
        ),
        "conservative_spatial_passed": bool(
            conservative <= SPATIAL_RESPONSE_GATE
        ),
    }
    row["passed"] = bool(
        row["temporal_passed"]
        and row["raw_spatial_passed"]
        and row["conservative_spatial_passed"]
    )
    return row, arrays


def _limiter_summary(initial: dict, vector: np.ndarray) -> dict:
    state = unpack_causal_five_field_state(
        vector,
        initial["state"].n_cells,
    )
    reconstruction = causal_five_field_reconstruct_face_charts(
        initial["context"],
        state.primitives,
    )
    factors = reconstruction.admissibility_factors
    return {
        "minimum_admissibility_factor": float(np.min(factors)),
        "admissibility_limited_cell_count": int(
            np.count_nonzero(factors < 1.0 - 1.0e-12)
        ),
    }


def _clock_summary(initial: dict, vector: np.ndarray) -> dict:
    clocks = causal_five_field_local_timescale_audit(
        initial["context"],
        vector,
    )
    radius_rg = (
        initial["context"].grid.centers
        / initial["context"].grid.gravitational_radius
    )
    names = (
        "characteristic_crossing_seconds",
        "stress_relaxation_seconds",
        "thermal_response_seconds",
        "luminosity_response_seconds",
        "radial_advection_seconds",
        "local_loading_seconds",
    )
    minima = {}
    for name in names:
        values = np.asarray(getattr(clocks, name), dtype=float)
        index = int(np.argmin(values))
        minima[name] = {
            "seconds": float(values[index]),
            "cell_index": index,
            "radius_rg": float(radius_rg[index]),
        }
    finite = {
        name: row
        for name, row in minima.items()
        if name != "local_loading_seconds"
    }
    shortest = min(finite, key=lambda name: finite[name]["seconds"])
    return {
        "minima": minima,
        "global_loading_seconds": float(clocks.global_loading_seconds),
        "shortest_physical_clock": {
            "name": shortest,
            **finite[shortest],
        },
    }


def _campaign_summary(
    initial: dict,
    start: CausalFiveFieldAdaptiveBDF2Restart,
    final: CausalFiveFieldAdaptiveBDF2Restart,
    snapshot_rows: dict,
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
    independent = [
        row["independent_audit"]
        for row in attempts
        if row["independent_audit"] is not None
    ]
    local_passed = all(
        row["local_gate_audit"] is None
        or row["local_gate_audit"]["passed"]
        for row in accepted
    )
    audits_passed = bool(
        independent and all(row["passed"] for row in independent)
    )
    ledger = causal_five_field_bdf_physical_ledger_from_restart(final)
    relative = causal_five_field_bdf_physical_ledger_relative_defects(
        ledger
    )
    maximum_ledger = float(np.max(relative))
    state_gates = audit_causal_five_field_state_gates(
        initial["context"],
        final.state_vector,
    )
    return {
        "segments": segments,
        "extension_accepted_steps": int(
            final.accepted_steps - start.accepted_steps
        ),
        "extension_accepted_bdf2_steps": int(
            final.accepted_bdf2_steps - start.accepted_bdf2_steps
        ),
        "extension_rejected_attempts": int(
            final.rejected_attempts - start.rejected_attempts
        ),
        "extension_audit_count": int(
            final.audit_count - start.audit_count
        ),
        "minimum_dt_used": min(
            segment["minimum_dt_used"] for segment in segments
        ),
        "maximum_dt_used": max(
            segment["maximum_dt_used"] for segment in segments
        ),
        "work": wp10c7k._sum_work(segments),
        "all_segments_passed": all(
            segment["passed"] for segment in segments
        ),
        "local_estimator_passed": local_passed,
        "independent_audits": {
            "count": len(independent),
            "maximum_normalized_error": max(
                float(
                    row["temporal_gate_audit"][
                        "maximum_normalized_error"
                    ]
                )
                for row in independent
            ),
            "passed": audits_passed,
        },
        "physical_ledger": {
            "component_relative_defects": wp10c7k._plain(relative),
            "maximum_relative_defect": maximum_ledger,
            "gate": MAXIMUM_PHYSICAL_LEDGER_RELATIVE_DEFECT,
            "passed": bool(
                maximum_ledger
                <= MAXIMUM_PHYSICAL_LEDGER_RELATIVE_DEFECT
            ),
        },
        "state_gates": state_gates,
        "state_summary": causal_five_field_state_summary(
            initial["context"],
            final.state_vector,
        ),
        "clock_summary": {
            "wp10c7k_start": _clock_summary(
                initial,
                start.state_vector,
            ),
            "extension_endpoint": _clock_summary(
                initial,
                final.state_vector,
            ),
        },
        "snapshot_limiters": {
            label: _limiter_summary(
                initial,
                snapshot_rows[label].state_vector,
            )
            for label in snapshot_rows
        },
        "snapshot_checkpoints": {
            label: {
                "path": _relative(
                    _checkpoint_path(
                        initial["state"].n_cells,
                        final.provenance["trajectory_mode"],
                        label,
                    )
                ),
                "sha256": _sha256(
                    _checkpoint_path(
                        initial["state"].n_cells,
                        final.provenance["trajectory_mode"],
                        label,
                    )
                ),
                "roundtrip_bitwise": roundtrips[label],
            }
            for label in snapshot_rows
        },
        "all_snapshot_roundtrips_bitwise": all(
            roundtrips.values()
        ),
        "passed": bool(
            all(segment["passed"] for segment in segments)
            and local_passed
            and audits_passed
            and maximum_ledger
            <= MAXIMUM_PHYSICAL_LEDGER_RELATIVE_DEFECT
            and state_gates["passed"]
            and all(roundtrips.values())
        ),
    }


def _spatial_growth_summary(common_rows: dict) -> dict:
    labels = list(common_rows)
    times = np.asarray(
        [common_rows[label]["elapsed_time_seconds"] for label in labels],
        dtype=float,
    )
    raw = np.asarray(
        [
            common_rows[label][
                "raw_n32_n64_log_h_over_r_difference"
            ]
            for label in labels
        ],
        dtype=float,
    )
    conservative = np.asarray(
        [
            common_rows[label]["conservative_log_h_over_r_total"]
            for label in labels
        ],
        dtype=float,
    )
    raw_slope = float(np.dot(times, raw) / np.dot(times, times))
    raw_fit = raw_slope * times
    crossing = None
    for index in range(1, len(labels)):
        lower = conservative[index - 1]
        upper = conservative[index]
        if lower <= SPATIAL_RESPONSE_GATE < upper:
            fraction = (
                (SPATIAL_RESPONSE_GATE - lower) / (upper - lower)
            )
            crossing = float(
                times[index - 1]
                + fraction * (times[index] - times[index - 1])
            )
            break
    return {
        "raw_difference_per_elapsed_second": {
            label: float(raw[index] / times[index])
            for index, label in enumerate(labels)
        },
        "through_origin_raw_slope_per_second": raw_slope,
        "maximum_raw_linear_fit_absolute_residual": float(
            np.max(np.abs(raw - raw_fit))
        ),
        "raw_gate_crossing_projection_seconds": float(
            SPATIAL_RESPONSE_GATE / raw_slope
        ),
        "conservative_gate_crossing_linear_interpolation_seconds": (
            crossing
        ),
    }


def _run_replay(
    initial: dict,
    evidence_sha256: str,
    checkpoint: dict,
    snapshots: dict,
    reached_labels: list[str],
    *,
    force: bool,
) -> dict:
    final_label = reached_labels[-1]
    if len(reached_labels) == 1:
        parent, _ = _load_parent_restart(
            json.loads(WP10C7K_OUTPUT.read_text(encoding="utf-8")),
            initial,
        )
        start = _extension_start(
            parent,
            initial,
            evidence_sha256,
            checkpoint,
            REPLAY_MODE,
        )
    else:
        start = snapshots[reached_labels[-2]]
    path = _replay_path(initial["state"].n_cells, final_label)
    if path.exists() and not force:
        replay = load_causal_five_field_adaptive_bdf2_restart(
            path,
            initial["context"],
        )
        _validate_restart(
            replay,
            initial,
            evidence_sha256,
            checkpoint,
            REPLAY_MODE,
            final_label,
        )
    else:
        replay = _run_segment(
            initial,
            start,
            REPLAY_MODE,
            final_label,
        )
        save_causal_five_field_adaptive_bdf2_restart(
            path,
            initial["context"],
            replay,
        )
        replay = load_causal_five_field_adaptive_bdf2_restart(
            path,
            initial["context"],
        )
        _validate_restart(
            replay,
            initial,
            evidence_sha256,
            checkpoint,
            REPLAY_MODE,
            final_label,
        )
    bitwise = causal_five_field_adaptive_bdf2_restarts_equal(
        snapshots[final_label],
        replay,
    )
    if not bitwise:
        raise RuntimeError(
            f"WP10c7l N{initial['state'].n_cells} replay differs"
        )
    return {
        "path": _relative(path),
        "sha256": _sha256(path),
        "from_label": (
            reached_labels[-2]
            if len(reached_labels) > 1
            else "wp10c7k_parent"
        ),
        "to_label": final_label,
        "endpoint_replay_bitwise": bitwise,
    }


def _load_completed(
    initial: dict,
    evidence_sha256: str,
    checkpoint: dict,
    mode: str,
) -> tuple[dict, dict]:
    snapshots = {}
    roundtrips = {}
    for label, _ in COMMON_OUTPUTS:
        path = _checkpoint_path(
            initial["state"].n_cells,
            mode,
            label,
        )
        if not path.exists():
            break
        snapshots[label] = _load_snapshot(
            initial,
            evidence_sha256,
            checkpoint,
            mode,
            label,
        )
        roundtrips[label] = True
    return snapshots, roundtrips


def _aggregate(
    output_path: Path,
    arrays_path: Path,
    evidence: dict,
    evidence_sha256: str,
    initial: dict,
    starts: dict,
    snapshots: dict,
    roundtrips: dict,
    common_rows: dict,
    common_arrays: dict,
    replay: dict,
) -> dict:
    reached_labels = list(common_rows)
    final_label = reached_labels[-1]
    source_audit = wp10c7k._source_restriction_audit(
        initial[32]["context"],
        initial[64]["context"],
    )
    campaigns = {}
    for n_cells in RESOLUTIONS:
        campaigns[str(n_cells)] = {}
        for mode in TRAJECTORY_MODES:
            campaigns[str(n_cells)][mode] = _campaign_summary(
                initial[n_cells],
                starts[(n_cells, mode)],
                snapshots[(n_cells, mode)][final_label],
                {
                    label: snapshots[(n_cells, mode)][label]
                    for label in reached_labels
                },
                {
                    label: roundtrips[(n_cells, mode)][label]
                    for label in reached_labels
                },
            )

    work = {}
    work_passed = True
    for n_cells in RESOLUTIONS:
        production = campaigns[str(n_cells)]["production"]["work"]
        control = campaigns[str(n_cells)]["temporal_control"]["work"]
        jacobian_fraction = (
            production["jacobian_evaluations"]
            / control["jacobian_evaluations"]
        )
        function_fraction = (
            production["function_evaluations"]
            / control["function_evaluations"]
        )
        row_passed = bool(
            jacobian_fraction
            <= MAXIMUM_PRODUCTION_TO_CONTROL_JACOBIAN_FRACTION
        )
        work_passed = work_passed and row_passed
        work[str(n_cells)] = {
            "production": production,
            "temporal_control": control,
            "production_to_control_jacobian_fraction": (
                jacobian_fraction
            ),
            "production_to_control_function_fraction": (
                function_fraction
            ),
            "maximum_jacobian_fraction": (
                MAXIMUM_PRODUCTION_TO_CONTROL_JACOBIAN_FRACTION
            ),
            "passed": row_passed,
        }

    campaign_passed = all(
        campaigns[str(n_cells)][mode]["passed"]
        for n_cells in RESOLUTIONS
        for mode in TRAJECTORY_MODES
    )
    common_passed = all(row["passed"] for row in common_rows.values())
    target_reached = final_label == COMMON_OUTPUTS[-1][0]
    replay_passed = all(
        replay[str(n_cells)]["endpoint_replay_bitwise"]
        for n_cells in RESOLUTIONS
    )
    endpoint_terms, endpoint_term_arrays = wp10c7k._endpoint_term_rows(
        initial,
        {
            n_cells: {
                "final": snapshots[(n_cells, "production")][final_label]
            }
            for n_cells in RESOLUTIONS
        },
    )
    common_arrays.update(endpoint_term_arrays)
    growth = _spatial_growth_summary(common_rows)
    passed = bool(
        target_reached
        and source_audit["passed"]
        and campaign_passed
        and common_passed
        and work_passed
        and replay_passed
    )
    if passed:
        decision = "wp10c7l_characteristic_crossing_extension_certified"
        next_authorization = (
            "no_tide_duration_ladder_stress_relaxation_clock"
        )
    elif (
        not source_audit["passed"]
        or not campaign_passed
        or not replay_passed
    ):
        decision = "wp10c7l_numerical_or_physical_gate_failed"
        next_authorization = "diagnose_extension_solver_or_state_failure"
    elif not common_rows[final_label]["temporal_passed"]:
        decision = "wp10c7l_accumulated_temporal_gate_failed"
        next_authorization = "tighten_or_upgrade_temporal_reference"
    elif not common_rows[final_label]["conservative_spatial_passed"]:
        decision = "wp10c7l_characteristic_rung_spatial_stop"
        next_authorization = "diagnose_spatial_error_growth_before_extension"
    elif not work_passed:
        decision = "wp10c7l_extension_work_gate_failed"
        next_authorization = "profile_adaptive_extension_cost"
    else:
        decision = "wp10c7l_stopped_before_target"
        next_authorization = "diagnose_first_failed_common_output"

    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **common_arrays)
    payload = {
        "work_package": "WP10c7l",
        "base_commit": BASE_COMMIT,
        "scope": (
            "matched N32/N64 no-tide adaptive-BDF2 extension from "
            "the certified WP10c7k endpoint to the first declared "
            "characteristic-crossing rung"
        ),
        "spatial_options": dict(SPATIAL_OPTIONS),
        "physics": {
            "stream": "exact circularized source",
            "tide": "off",
            "wind": "off",
        },
        "start_elapsed_time_seconds": START_TIME_SECONDS,
        "target_elapsed_time_seconds": TARGET_TIME_SECONDS,
        "common_output_times_seconds": {
            label: target for label, target in COMMON_OUTPUTS
        },
        "reached_labels": reached_labels,
        "temporal_reference_contract": {
            "production_controller": _controller_contract("production"),
            "temporal_control_controller": _controller_contract(
                "temporal_control"
            ),
            "second_order_error_safety_factor": (
                SECOND_ORDER_ERROR_SAFETY_FACTOR
            ),
            "reserved_temporal_gate_fraction": (
                ACCUMULATED_TEMPORAL_GATE_FRACTION
            ),
            "policy": (
                "compare the unchanged production controller with a "
                "same-method half-ceiling control, multiply the raw "
                "difference by 4/3, and retain the complete v1 "
                "observable schema"
            ),
        },
        "wp10c7k_evidence": {
            "path": _relative(WP10C7K_OUTPUT),
            "sha256": evidence_sha256,
            "decision": evidence["decision"],
            "parent_conservative_spatial_error": evidence[
                "primary_log_h_over_r_contract"
            ]["maximum_conservative_spatial_error"],
            "parent_checkpoints": {
                str(n_cells): _parent_checkpoint_entry(
                    evidence,
                    n_cells,
                )
                for n_cells in RESOLUTIONS
            },
        },
        "initialization": {
            str(n_cells): {
                "initial_state_sha256": initial[n_cells][
                    "vector_sha256"
                ],
                "wp10c7k_start_state_summary": (
                    causal_five_field_state_summary(
                        initial[n_cells]["context"],
                        starts[(n_cells, "production")].state_vector,
                    )
                ),
            }
            for n_cells in RESOLUTIONS
        },
        "source_restriction_audit": source_audit,
        "campaigns": campaigns,
        "common_time_contract": common_rows,
        "spatial_growth_audit": growth,
        "endpoint_term_response_comparison": endpoint_terms,
        "work_audit": work,
        "restart_replay": replay,
        "primary_log_h_over_r_contract": {
            "maximum_raw_spatial_difference": max(
                row["raw_n32_n64_log_h_over_r_difference"]
                for row in common_rows.values()
            ),
            "maximum_conservative_spatial_error": max(
                row["conservative_log_h_over_r_total"]
                for row in common_rows.values()
            ),
            "gate": SPATIAL_RESPONSE_GATE,
            "all_common_times_passed": common_passed,
        },
        "gates": {
            "source_restriction_passed": source_audit["passed"],
            "both_meshes_and_both_temporal_modes_passed": (
                campaign_passed
            ),
            "all_common_time_temporal_and_spatial_budgets_passed": (
                common_passed
            ),
            "target_time_reached": target_reached,
            "production_work_advantage_passed": work_passed,
            "production_replays_bitwise": replay_passed,
            "wp10c7l_passed": passed,
        },
        "decision": decision,
        "next_authorization": next_authorization,
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
    evidence, evidence_sha256 = _validate_wp10c7k()
    initial = _initial_bundles(evidence)
    parents = {}
    parent_checkpoints = {}
    for n_cells in RESOLUTIONS:
        parent, checkpoint = _load_parent_restart(
            evidence,
            initial[n_cells],
        )
        parents[n_cells] = parent
        parent_checkpoints[n_cells] = checkpoint

    starts = {
        (n_cells, mode): _extension_start(
            parents[n_cells],
            initial[n_cells],
            evidence_sha256,
            parent_checkpoints[n_cells],
            mode,
        )
        for n_cells in RESOLUTIONS
        for mode in TRAJECTORY_MODES
    }
    if args.preflight:
        source = wp10c7k._source_restriction_audit(
            initial[32]["context"],
            initial[64]["context"],
        )
        print(
            json.dumps(
                {
                    "work_package": "WP10c7l",
                    "preflight_passed": bool(
                        source["passed"]
                        and all(
                            start.elapsed_time == START_TIME_SECONDS
                            and start.next_order == 2
                            for start in starts.values()
                        )
                    ),
                    "wp10c7k_evidence_sha256": evidence_sha256,
                    "parent_checkpoint_sha256": {
                        str(n_cells): parent_checkpoints[n_cells][
                            "sha256"
                        ]
                        for n_cells in RESOLUTIONS
                    },
                    "source_restriction_audit": source,
                    "target_elapsed_time_seconds": TARGET_TIME_SECONDS,
                },
                sort_keys=True,
            )
        )
        return

    snapshots = {
        key: {} for key in starts
    }
    roundtrips = {
        key: {} for key in starts
    }
    states = dict(starts)
    common_rows = {}
    common_arrays = {}

    if args.aggregate_only:
        for n_cells in RESOLUTIONS:
            for mode in TRAJECTORY_MODES:
                loaded, loaded_roundtrips = _load_completed(
                    initial[n_cells],
                    evidence_sha256,
                    parent_checkpoints[n_cells],
                    mode,
                )
                snapshots[(n_cells, mode)] = loaded
                roundtrips[(n_cells, mode)] = loaded_roundtrips
        common_labels = [
            label
            for label, _ in COMMON_OUTPUTS
            if all(
                label in snapshots[(n_cells, mode)]
                for n_cells in RESOLUTIONS
                for mode in TRAJECTORY_MODES
            )
        ]
        if not common_labels:
            raise RuntimeError("WP10c7l has no complete common output")
        for label in common_labels:
            row, arrays = _common_time_row(
                label,
                evidence,
                initial,
                snapshots,
            )
            common_rows[label] = row
            common_arrays.update(arrays)
            if not row["passed"]:
                break
    else:
        for label, _ in COMMON_OUTPUTS:
            for n_cells in RESOLUTIONS:
                for mode in TRAJECTORY_MODES:
                    state, roundtrip = _run_or_load_segment(
                        initial[n_cells],
                        evidence_sha256,
                        parent_checkpoints[n_cells],
                        states[(n_cells, mode)],
                        mode,
                        label,
                        force=args.force,
                    )
                    states[(n_cells, mode)] = state
                    snapshots[(n_cells, mode)][label] = state
                    roundtrips[(n_cells, mode)][label] = roundtrip
            row, arrays = _common_time_row(
                label,
                evidence,
                initial,
                snapshots,
            )
            common_rows[label] = row
            common_arrays.update(arrays)
            print(
                json.dumps(
                    {
                        "work_package": "WP10c7l",
                        "common_output": label,
                        "elapsed_time_seconds": row[
                            "elapsed_time_seconds"
                        ],
                        "raw_spatial_error": row[
                            "raw_n32_n64_log_h_over_r_difference"
                        ],
                        "conservative_spatial_error": row[
                            "conservative_log_h_over_r_total"
                        ],
                        "passed": row["passed"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if not row["passed"]:
                break

    reached_labels = list(common_rows)
    replay = {
        str(n_cells): _run_replay(
            initial[n_cells],
            evidence_sha256,
            parent_checkpoints[n_cells],
            snapshots[(n_cells, REPLAY_MODE)],
            reached_labels,
            force=args.force,
        )
        for n_cells in RESOLUTIONS
    }
    payload = _aggregate(
        output_path,
        arrays_path,
        evidence,
        evidence_sha256,
        initial,
        starts,
        snapshots,
        roundtrips,
        common_rows,
        common_arrays,
        replay,
    )
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "reached_labels": payload["reached_labels"],
                "maximum_conservative_log_h_over_r_error": payload[
                    "primary_log_h_over_r_contract"
                ]["maximum_conservative_spatial_error"],
                "output": _relative(output_path),
                "arrays": _relative(arrays_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
