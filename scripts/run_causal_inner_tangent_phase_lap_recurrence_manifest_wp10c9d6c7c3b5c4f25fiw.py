#!/usr/bin/env python3
"""Freeze a staged tangent-phase-lap and state-recurrence acquisition."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from imri_qpe.layer3_minidisk_1d.tangent_phase_atlas import (  # noqa: E402
    normalized_metric_tangents,
)
import run_causal_inner_conservative_tangent_phase_atlas_holdout_execution_wp10c9d6c7c3b5c4f25fiv as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = parent.AUTHORIZED_NEXT
CLASSIFICATION = (
    "bounded_tangent_phase_lap_recurrence_acquisition_selected_"
    "stage1_required"
)
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fix_"
    "tangent_phase_lap_recurrence_stage1_execution"
)
ARTIFACT = (
    "causal_inner_tangent_phase_lap_recurrence_manifest_"
    "wp10c9d6c7c3b5c4f25fiw"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_TANGENT_PHASE_LAP_RECURRENCE_"
    "MANIFEST_WP10C9D6C7C3B5C4F25FIW_2026-08-24.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_tangent_phase_lap_recurrence_manifest_"
    "wp10c9d6c7c3b5c4f25fiw.py"
)
THIS_TEST = (
    "tests/test_causal_inner_tangent_phase_lap_recurrence_manifest_"
    "wp10c9d6c7c3b5c4f25fiw.py"
)

INITIAL_ELAPSED_SECONDS = 0.16800000000000012
SEGMENT_SECONDS = 2.5e-4
STAGE_ACCEPTED_SEGMENTS = 48
PLANNED_STAGES = 3
PLANNED_ACCEPTED_SEGMENTS = STAGE_ACCEPTED_SEGMENTS * PLANNED_STAGES
MAXIMUM_STAGE_ATTEMPTS = 52
BLIND_MIDPOINT_FREQUENCY = 8
STAGE_EXACT_FIELD_BUDGET = (
    STAGE_ACCEPTED_SEGMENTS
    + STAGE_ACCEPTED_SEGMENTS // BLIND_MIDPOINT_FREQUENCY
)
STAGE_RETRACTION_BUDGET = STAGE_EXACT_FIELD_BUDGET
STAGE_WALL_HOURS = 7.0
MINIMUM_STAGE_PHASE_ADVANCE = 1.5
MAXIMUM_LOCAL_PHASE_INCREMENT = 0.08
MINIMUM_LOCAL_PHASE_INCREMENT = 0.0
PHASE_LAP_RADIANS = 2.0 * math.pi
PHASE_LAP_REGISTRATION_TOLERANCE_RADIANS = 0.25
MAXIMUM_RETURN_DISTANCE_OVER_PATH_LENGTH = 0.10
MINIMUM_RETURN_TANGENT_COSINE = 0.99
MINIMUM_SECTION_DERIVATIVE_FRACTION = 0.50


def _helper():
    return parent._helper()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def _source_hashes() -> dict[str, str]:
    helper = _helper()
    paths = (
        THIS_RUNNER,
        THIS_TEST,
        parent.THIS_RUNNER,
        parent.THIS_TEST,
        parent.manifest.THIS_RUNNER,
        parent.manifest.THIS_TEST,
        parent.manifest.PHASE_SOURCE,
        parent.manifest.PHASE_TEST,
        parent.engine.THIS_RUNNER,
        parent.engine.THIS_TEST,
    )
    return {name: helper._sha(ROOT / name) for name in paths}


def _validate_parent(*, require_clean: bool) -> dict:
    helper = _helper()
    hashes = helper._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = helper._read(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = helper._read(
        parent.CANONICAL_DIRECTORY / "holdout_metrics.json"
    )
    provenance = helper._read(parent.CANONICAL_DIRECTORY / "provenance.json")
    values = metrics["gate_values"]
    if (
        summary["classification"] != parent.PASS_CLASSIFICATION
        or not summary["passed"]
        or not summary["phase_atlas_prospectively_validated"]
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["complete_cycle_execution_authorized"]
        or summary["reduced_slow_evolution_authorized"]
        or metrics["classification"] != parent.PASS_CLASSIFICATION
        or not metrics["passed"]
        or values["accepted_segments"] != 16
        or values["attempted_segments"] != 16
        or values["phase_holdouts_evaluated"] != 16
        or not values["all_phase_holdouts_passed"]
        or values["phase_lap_observed"]
        or values["cycle_observed"]
        or values["terminal_elapsed_seconds"] != INITIAL_ELAPSED_SECONDS
        or not values["all_accepted_checkpoint_roundtrips_bitwise"]
        or not values["suffix_history_replay_bitwise"]
    ):
        raise RuntimeError("prospective tangent-phase holdout changed")
    for relative, expected in provenance["source_hashes"].items():
        if helper._sha(ROOT / relative) != expected:
            raise RuntimeError(f"tangent-phase holdout source changed: {relative}")
    parent._validate_manifest(require_clean=False)
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("phase-lap recurrence manifest requires a clean tracked tree")
    return {
        "hashes": hashes,
        "summary": summary,
        "metrics": metrics,
        "provenance": provenance,
    }


def _observations(parent_lock: dict) -> dict:
    arrays = _load_npz(parent.CANONICAL_DIRECTORY / "holdout_arrays.npz")
    phase = np.asarray(arrays["accepted_phase_increments"], dtype=float)
    values = parent_lock["metrics"]["gate_values"]
    if (
        phase.shape != (16,)
        or not np.all(phase > 0.0)
        or arrays["accepted_endpoint_coordinate_rates470_per_s"].shape
        != (16, 470)
        or arrays["current_coordinate470"].shape != (470,)
        or arrays["current_primitive_state"].shape != (112, 5)
    ):
        raise RuntimeError("phase-lap input arrays changed")
    minimum = float(np.min(phase))
    mean = float(np.mean(phase))
    segments_at_minimum = int(math.ceil(PHASE_LAP_RADIANS / minimum))
    segments_at_mean = int(math.ceil(PHASE_LAP_RADIANS / mean))
    exact_unit_wall_seconds = float(
        values["execution_wall_seconds"] / values["exact_free_field_calls"]
    )
    total_blind_calls = (
        PLANNED_ACCEPTED_SEGMENTS // BLIND_MIDPOINT_FREQUENCY
    )
    planned_exact_calls = PLANNED_ACCEPTED_SEGMENTS + total_blind_calls
    return {
        "minimum_observed_phase_increment": minimum,
        "mean_observed_phase_increment": mean,
        "median_observed_phase_increment": float(np.median(phase)),
        "maximum_observed_phase_increment": float(np.max(phase)),
        "observed_holdout_phase_advance": float(np.sum(phase)),
        "segments_to_lap_at_observed_minimum": segments_at_minimum,
        "segments_to_lap_at_observed_mean": segments_at_mean,
        "planned_accepted_segments": PLANNED_ACCEPTED_SEGMENTS,
        "planned_exact_field_calls": planned_exact_calls,
        "measured_wall_seconds_per_exact_field_and_retraction_unit": (
            exact_unit_wall_seconds
        ),
        "projected_stage1_wall_hours": (
            STAGE_EXACT_FIELD_BUDGET * exact_unit_wall_seconds / 3600.0
        ),
        "projected_full_acquisition_wall_hours": (
            planned_exact_calls * exact_unit_wall_seconds / 3600.0
        ),
        "stage1_phase_advance_at_observed_minimum": (
            STAGE_ACCEPTED_SEGMENTS * minimum
        ),
    }


def _continuation_seed() -> dict[str, np.ndarray]:
    arrays = _load_npz(parent.CANONICAL_DIRECTORY / "holdout_arrays.npz")
    diagnostics = parent._diagnostic_arrays()
    observer_transform = diagnostics["terminal_metric_transform470x470"]
    current_rate = arrays["current_coordinate_rate470_per_s"]
    reference_tangent = normalized_metric_tangents(
        current_rate[None, :], observer_transform
    )[0]
    section_covector = observer_transform.T @ reference_tangent
    training_rates = arrays[
        "accepted_endpoint_coordinate_rates470_per_s"
    ][-parent.manifest.SELECTED_WINDOW :]
    seed_names = (
        "previous_coordinate470",
        "current_coordinate470",
        "previous_primitive_state",
        "current_primitive_state",
        "previous_coordinate_rate470_per_s",
        "current_coordinate_rate470_per_s",
        "previous_span_seconds",
        "next_span_seconds",
        "elapsed_seconds",
        "accepted_segments_total",
        "accepted_since_growth",
        "metric_transform470x470",
        "metric_augmented560x560",
        "gauge_basis560x90",
        "section_normal470",
        "start_coordinate470",
    )
    seed = {name: np.asarray(arrays[name]) for name in seed_names}
    seed.update(
        {
            "phase_training_raw_rates470_per_s": training_rates,
            "phase_observer_metric_transform470x470": observer_transform,
            "phase_lap_reference_coordinate470": arrays[
                "current_coordinate470"
            ].copy(),
            "phase_lap_reference_primitive_state": arrays[
                "current_primitive_state"
            ].copy(),
            "phase_lap_reference_unit_tangent470": reference_tangent,
            "registered_section_covector470": section_covector,
            "registered_section_reference_value": np.asarray(0.0),
            "reference_metric_speed_per_s": np.asarray(
                np.linalg.norm(observer_transform @ current_rate)
            ),
            "unwrapped_phase_advance_radians": np.asarray(0.0),
            "accumulated_metric_path_length": np.asarray(0.0),
            "acquisition_stage": np.asarray(1),
        }
    )
    return seed


def _definitions(observations: dict) -> dict:
    stages = [
        {
            "stage": index,
            "accepted_segments": STAGE_ACCEPTED_SEGMENTS,
            "physical_horizon_seconds": (
                STAGE_ACCEPTED_SEGMENTS * SEGMENT_SECONDS
            ),
            "terminal_elapsed_seconds_if_complete": (
                INITIAL_ELAPSED_SECONDS
                + index * STAGE_ACCEPTED_SEGMENTS * SEGMENT_SECONDS
            ),
        }
        for index in range(1, PLANNED_STAGES + 1)
    ]
    contract = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "authorized_execution": AUTHORIZED_NEXT,
        "truth_dynamics": "dq/dt=f_free(q) on the certified conservative atlas",
        "phase_registration": {
            "phase_zero_elapsed_seconds": INITIAL_ELAPSED_SECONDS,
            "phase_zero_is_prospective": True,
            "unwrapped_phase_is_sum_of_accepted_exact_local_increments": True,
            "rejected_candidates_never_advance_phase_or_history": True,
            "phase_lap_radians": PHASE_LAP_RADIANS,
            "local_observer": (
                "trailing 12-tangent oriented affine two-plane circle in frozen W_164ms"
            ),
        },
        "staged_scope": {
            "only_stage1_is_authorized_now": True,
            "stages": stages,
            "stage1": {
                "accepted_segments": STAGE_ACCEPTED_SEGMENTS,
                "maximum_attempted_segments": MAXIMUM_STAGE_ATTEMPTS,
                "segment_seconds": SEGMENT_SECONDS,
                "blind_midpoint_frequency": BLIND_MIDPOINT_FREQUENCY,
                "maximum_exact_free_field_calls": STAGE_EXACT_FIELD_BUDGET,
                "maximum_retractions": STAGE_RETRACTION_BUDGET,
                "maximum_wall_hours": STAGE_WALL_HOURS,
            },
            "stop_immediately_on_first_binding_failure": True,
            "stop_at_first_phase_lap_or_registered_return_candidate": True,
        },
        "binding_stage1_gates": {
            "minimum_accepted_segments": STAGE_ACCEPTED_SEGMENTS,
            "minimum_cumulative_phase_advance_radians": (
                MINIMUM_STAGE_PHASE_ADVANCE
            ),
            "minimum_local_phase_increment_strictly_greater_than": (
                MINIMUM_LOCAL_PHASE_INCREMENT
            ),
            "maximum_local_phase_increment": MAXIMUM_LOCAL_PHASE_INCREMENT,
            "maximum_holdout_relative_radial_defect": 0.002,
            "maximum_holdout_out_of_plane_defect": 0.005,
            "maximum_direction_prediction_defect_radians": 0.005,
            "minimum_training_two_plane_energy_fraction": 0.999,
            "maximum_training_relative_radial_rms": 0.001,
            "all_original_physical_and_retraction_gates_unchanged": True,
            "checkpoint_roundtrip_bitwise": True,
            "suffix_history_replay_bitwise": True,
        },
        "coarse_recurrence_candidate_requires": {
            "unwrapped_phase_at_crossing_within_radians_of_2pi": (
                PHASE_LAP_REGISTRATION_TOLERANCE_RADIANS
            ),
            "registered_section_bracket": "g_previous < 0 <= g_current",
            "registered_section_definition": (
                "g(q)=tau_0^T W_phase (q-q_0)"
            ),
            "minimum_positive_section_derivative_fraction_of_reference_speed": (
                MINIMUM_SECTION_DERIVATIVE_FRACTION
            ),
            "maximum_metric_state_return_distance_over_accumulated_path_length": (
                MAXIMUM_RETURN_DISTANCE_OVER_PATH_LENGTH
            ),
            "minimum_metric_tangent_cosine": MINIMUM_RETURN_TANGENT_COSINE,
            "all_original_physical_and_replay_gates": True,
        },
        "classification_branches": {
            "stage1_passed_no_lap": (
                "authorize only a definitions-only stage2 resume manifest"
            ),
            "coarse_recurrence_candidate_observed": (
                "authorize only exact registered-section return refinement"
            ),
            "phase_lap_without_coarse_recurrence": (
                "reject periodic averaging and classify the trajectory as open/nonperiodic"
            ),
            "phase_geometry_failed": "stop without propagating the rejected endpoint",
            "physical_or_numerical_failed": (
                "stop and preserve the original physical/numerical classification"
            ),
        },
        "authorization_boundary": (
            "neither a stage pass nor a coarse recurrence candidate is a cycle certificate; "
            "complete-cycle execution and reduced slow evolution remain unauthorized"
        ),
    }
    architecture = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "offline_truth_layer": {
            "dynamics": "autonomous exact reaction-free conservative field",
            "purpose": "one-time orbit or nonrecurrence identification",
            "phase": "overlapping tangent-circle atlas",
            "recurrence": "metric state return plus registered transverse section",
        },
        "phase_lap_is_not_a_cycle": True,
        "cycle_refinement_if_coarse_candidate_passes": {
            "method": "exact retracted registered-section root refinement",
            "then": "phase-conditioned multiple shooting/collocation",
            "unknowns": "orbit nodes, segment durations, and period",
            "equations": "exact free-field matching and periodic closure",
        },
        "slow_architecture_only_after_periodic_family_is_certified": {
            "offline": (
                "continue periodic orbit over slow Q anchors and compute averaged drift plus bordered periodic adjoint"
            ),
            "online_state": "slow Q, mode label, and event state only",
            "online_rhs": "interpolated certified averaged drift Fbar(Q)",
            "online_forbidden": (
                "truth integration, fixed-Q reaction solves, nonlinear roots, micro-BDF steps, and phase-lap acquisition"
            ),
            "computational_intent": (
                "the 15-17 hour acquisition is offline once; cycle-scale slow evolution must run from the tabulated macro model within days"
            ),
        },
        "no_recurrence_branch": (
            "do not force cycle averaging; test equilibrium, transient, or nonperiodic-attractor closure"
        ),
        "measured_feasibility": observations,
    }
    return {"contract": contract, "architecture": architecture}


def _evaluate(parent_lock: dict) -> tuple[dict, dict[str, np.ndarray], dict]:
    observations = _observations(parent_lock)
    supported = bool(
        observations["segments_to_lap_at_observed_minimum"]
        <= PLANNED_ACCEPTED_SEGMENTS
        and observations["projected_stage1_wall_hours"] <= STAGE_WALL_HOURS
        and observations["minimum_observed_phase_increment"] > 0.0
        and observations["maximum_observed_phase_increment"]
        <= MAXIMUM_LOCAL_PHASE_INCREMENT
    )
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            CLASSIFICATION
            if supported
            else "tangent_phase_lap_recurrence_acquisition_not_feasible"
        ),
        "passed": supported,
        "definitions_only": True,
        "new_truth_evaluations": 0,
        "observations": observations,
        "phase_lap_observed": False,
        "state_recurrence_observed": False,
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT if supported else None,
        "input_lock": {
            "parent_hashes": parent_lock["hashes"],
            "parent_classification": parent_lock["summary"]["classification"],
        },
    }
    return metrics, _continuation_seed(), _definitions(observations)


def _update_catalog(summary: dict) -> None:
    helper = _helper()
    manifest_path = ROOT / "results/manifests/canonical_artifacts.csv"
    summary_path = ROOT / "results/manifests/canonical_summary.json"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": helper._sha(path),
                    "scientific_status": (
                        "SUPPORTED" if summary["passed"] else "REJECTED"
                    ),
                }
            )
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case", "path", "bytes", "sha256", "scientific_status"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = helper._read(summary_path)
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
            "latest_source_parent_commit": helper._git("rev-parse", "HEAD"),
            "latest_work_package": WORK_PACKAGE,
        }
    )
    helper._write_json(summary_path, catalog)


def _canonicalize(
    metrics: dict,
    seed: dict[str, np.ndarray],
    definitions: dict,
    parent_lock: dict,
) -> dict:
    helper = _helper()
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("phase-lap recurrence manifest result already exists")
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(
        CANONICAL_DIRECTORY / "feasibility_estimate.json", metrics
    )
    helper._write_json(
        CANONICAL_DIRECTORY / "acquisition_contract.json",
        definitions["contract"],
    )
    helper._write_json(
        CANONICAL_DIRECTORY / "mathematical_architecture.json",
        definitions["architecture"],
    )
    _save_npz(CANONICAL_DIRECTORY / "continuation_seed.npz", seed)
    helper._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_hashes": parent_lock["hashes"],
            "parent_classification": parent_lock["summary"]["classification"],
            "parent_implementation_commit": parent_lock["provenance"][
                "implementation_commit"
            ],
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": metrics["passed"],
        "definitions_only": True,
        "stage1_execution_authorized": metrics["passed"],
        "stage1_execution_executed": False,
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": metrics["authorized_next"],
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    helper._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "implementation_commit": helper._git("rev-parse", "HEAD"),
            "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"),
            "source_hashes": _source_hashes(),
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    )
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{helper._sha(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
    observations = metrics["observations"]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Tangent-phase-lap and state-recurrence acquisition manifest",
                "",
                f"Classification: `{metrics['classification']}`.",
                "",
                f"The 16-point prospective phase holdout passed. Its exact local phase increments span `{observations['minimum_observed_phase_increment']:.6e}` to `{observations['maximum_observed_phase_increment']:.6e}` rad, implying `{observations['segments_to_lap_at_observed_minimum']}` accepted 0.25 ms segments for a conservative 2*pi estimate.",
                "",
                f"The acquisition is split into three 48-endpoint tranches. Only stage 1 is authorized now. Measured-cost projections are `{observations['projected_stage1_wall_hours']:.2f}` hours for stage 1 and `{observations['projected_full_acquisition_wall_hours']:.2f}` hours for the full 144-endpoint acquisition.",
                "",
                "A phase lap is only a candidate. A cycle additionally requires metric state return/path length <= 0.10, tangent cosine >= 0.99, a positive transverse registered-section crossing near 2*pi, all physical gates, and exact restart/replay. Any candidate must then undergo exact registered-section refinement and periodic multiple shooting/collocation.",
                "",
                "The expensive tangent acquisition is an offline one-time task. A future online reduced solver may use only precomputed averaged slow drift over certified periodic Q anchors; it may not call the truth integrator, fixed-Q reactions, nonlinear roots, or micro-BDF steps.",
                "",
                f"Authorized next artifact: `{metrics['authorized_next']}`. Complete-cycle execution and reduced slow evolution remain unauthorized.",
                "",
            )
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    arguments = parser.parse_args()
    if not arguments.run:
        parser.error("--run is required")
    parent_lock = _validate_parent(require_clean=True)
    metrics, seed, definitions = _evaluate(parent_lock)
    summary = _canonicalize(metrics, seed, definitions, parent_lock)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
