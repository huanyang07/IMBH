#!/usr/bin/env python3
"""Freeze the second bounded tranche of tangent-phase-lap acquisition."""

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

import run_causal_inner_tangent_phase_lap_stage1_resume_execution_wp10c9d6c7c3b5c4f25fizb as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = parent.AUTHORIZED_NEXT
CLASSIFICATION = (
    "tangent_phase_lap_recurrence_stage2_selected_definitions_only"
)
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizd_"
    "tangent_phase_lap_recurrence_stage2_execution"
)
ARTIFACT = (
    "causal_inner_tangent_phase_lap_recurrence_stage2_manifest_"
    "wp10c9d6c7c3b5c4f25fizc"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_TANGENT_PHASE_LAP_"
    "RECURRENCE_STAGE2_MANIFEST_WP10C9D6C7C3B5C4F25FIZC_2026-08-25.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_tangent_phase_lap_recurrence_stage2_"
    "manifest_wp10c9d6c7c3b5c4f25fizc.py"
)
THIS_TEST = (
    "tests/test_causal_inner_tangent_phase_lap_recurrence_stage2_"
    "manifest_wp10c9d6c7c3b5c4f25fizc.py"
)

PRIOR_ACCEPTED_ENDPOINTS = 48
STAGE_ACCEPTED_ENDPOINTS = 48
COMBINED_ACCEPTED_ENDPOINTS = 96
MAXIMUM_ATTEMPTED_ENDPOINTS = 52
SEGMENT_SECONDS = 2.5e-4
BLIND_MIDPOINT_FREQUENCY = 8
MAXIMUM_EXACT_FREE_FIELD_CALLS = 54
MAXIMUM_RETRACTIONS = 54
MAXIMUM_WALL_HOURS = 7.0
METRIC_BLOCK_SIZES = (442, 28)
MAXIMUM_METRIC_CONDITION = 10.0
MINIMUM_NEW_PHASE_ADVANCE = 1.5
MAXIMUM_LOCAL_PHASE_INCREMENT = 0.08
PHASE_LAP_RADIANS = 2.0 * np.pi


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
        parent.recovery.THIS_RUNNER,
        parent.recovery.THIS_TEST,
        parent.phase.THIS_RUNNER,
        parent.phase.THIS_TEST,
        parent.engine.THIS_RUNNER,
        parent.engine.THIS_TEST,
        parent.suffix.THIS_RUNNER,
    )
    return {name: helper._sha(ROOT / name) for name in paths}


def _validate_parent(*, require_clean: bool) -> dict:
    helper = _helper()
    hashes = helper._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = helper._read(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = helper._read(
        parent.CANONICAL_DIRECTORY / "stage1_resume_metrics.json"
    )
    provenance = helper._read(parent.CANONICAL_DIRECTORY / "provenance.json")
    values = metrics["gate_values"]
    if (
        summary["classification"] != parent.PASS_CLASSIFICATION
        or not summary["passed"]
        or summary["combined_accepted_phase_endpoints"]
        != PRIOR_ACCEPTED_ENDPOINTS
        or summary["phase_lap_observed"]
        or summary["coarse_recurrence_candidate_observed"]
        or summary["complete_cycle_execution_authorized"]
        or summary["reduced_slow_evolution_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or metrics["classification"] != parent.PASS_CLASSIFICATION
        or not metrics["passed"]
        or values["combined_accepted_phase_endpoints"]
        != PRIOR_ACCEPTED_ENDPOINTS
        or values["new_accepted_phase_endpoints"] != 5
        or values["attempted_segments"] != 5
        or values["phase_lap_observed"]
        or values["coarse_recurrence_candidate_observed"]
        or not values["all_phase_geometry_gates_passed"]
        or not values["all_endpoint_metric_blocks_are_442_plus_28"]
        or not values["all_accepted_checkpoint_roundtrips_bitwise"]
        or not values["suffix_history_replay_bitwise"]
        or values["maximum_metric_coordinate_jacobian_condition"]
        > MAXIMUM_METRIC_CONDITION
    ):
        raise RuntimeError("stage1 completion certificate changed")
    for relative, expected in provenance["source_hashes"].items():
        if helper._sha(ROOT / relative) != expected:
            raise RuntimeError(f"stage1 completion source changed: {relative}")
    parent._validate_manifest(require_clean=False)
    if require_clean and helper._git(
        "status", "--short", "--untracked-files=no"
    ):
        raise RuntimeError("stage2 manifest requires a clean tracked tree")
    return {
        "hashes": hashes,
        "summary": summary,
        "metrics": metrics,
        "provenance": provenance,
    }


def _stage2_seed() -> dict[str, np.ndarray]:
    base = _load_npz(
        parent.manifest.CANONICAL_DIRECTORY / "resume_seed.npz"
    )
    stage1 = _load_npz(
        parent.CANONICAL_DIRECTORY / "stage1_resume_arrays.npz"
    )
    values = _helper()._read(
        parent.CANONICAL_DIRECTORY / "stage1_resume_metrics.json"
    )["gate_values"]
    progress_names = (
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
    seed = {name: np.asarray(stage1[name]) for name in progress_names}
    for name in (
        "phase_training_raw_rates470_per_s",
        "phase_observer_metric_transform470x470",
        "phase_lap_reference_coordinate470",
        "phase_lap_reference_primitive_state",
        "phase_lap_reference_unit_tangent470",
        "registered_section_covector470",
        "registered_section_reference_value",
        "reference_metric_speed_per_s",
    ):
        seed[name] = base[name]
    seed.update(
        {
            "accepted_endpoint_coordinates470": stage1[
                "combined_accepted_endpoint_coordinates470"
            ],
            "accepted_endpoint_primitive_states": stage1[
                "combined_accepted_endpoint_primitive_states"
            ],
            "accepted_endpoint_coordinate_rates470_per_s": stage1[
                "combined_accepted_endpoint_coordinate_rates470_per_s"
            ],
            "accepted_phase_increments": stage1[
                "combined_accepted_phase_increments"
            ],
            "accepted_cumulative_phase_advance_radians": np.concatenate(
                (
                    base["accepted_cumulative_phase_advance_radians"],
                    stage1["new_cumulative_phase_advance_radians"],
                )
            ),
            "accepted_cumulative_metric_path_lengths": np.concatenate(
                (
                    base["accepted_cumulative_metric_path_lengths"],
                    stage1["new_cumulative_metric_path_lengths"],
                )
            ),
            "accepted_registered_section_values": np.concatenate(
                (
                    base["accepted_registered_section_values"],
                    stage1["new_registered_section_values"],
                )
            ),
            "unwrapped_phase_advance_radians": np.asarray(
                values["cumulative_phase_advance_radians"]
            ),
            "accumulated_metric_path_length": np.asarray(
                values["cumulative_metric_path_length"]
            ),
            "selected_metric_block_sizes": np.asarray(
                METRIC_BLOCK_SIZES, dtype=np.int64
            ),
            "accepted_segments_new": np.asarray(0),
            "attempts": np.asarray(0),
            "acquisition_stage": np.asarray(2),
            "metric_chart_generation": np.asarray(1),
        }
    )
    if (
        seed["accepted_endpoint_coordinates470"].shape != (48, 470)
        or seed["accepted_endpoint_primitive_states"].shape != (48, 112, 5)
        or seed["accepted_endpoint_coordinate_rates470_per_s"].shape
        != (48, 470)
        or seed["accepted_phase_increments"].shape != (48,)
        or seed["accepted_cumulative_phase_advance_radians"].shape != (48,)
        or seed["accepted_cumulative_metric_path_lengths"].shape != (48,)
        or seed["accepted_registered_section_values"].shape != (48,)
        or int(seed["accepted_segments_total"]) != 244
        or float(seed["elapsed_seconds"]) != 0.18000000000000013
        or float(seed["unwrapped_phase_advance_radians"])
        != values["cumulative_phase_advance_radians"]
    ):
        raise RuntimeError("stage2 seed changed")
    return seed


def _observations(parent_lock: dict) -> dict:
    values = parent_lock["metrics"]["gate_values"]
    seed = _stage2_seed()
    first_segment = int(seed["accepted_segments_total"]) + 1
    segment_numbers = list(
        range(first_segment, first_segment + STAGE_ACCEPTED_ENDPOINTS)
    )
    blind_numbers = [
        value
        for value in segment_numbers
        if value % BLIND_MIDPOINT_FREQUENCY == 0
    ]
    wall_per_unit = float(
        values["execution_wall_seconds"] / values["exact_free_field_calls"]
    )
    current_phase = float(seed["unwrapped_phase_advance_radians"])
    observed = seed["accepted_phase_increments"][-12:]
    remaining = float(PHASE_LAP_RADIANS - current_phase)
    return {
        "prior_accepted_phase_endpoints": PRIOR_ACCEPTED_ENDPOINTS,
        "new_stage2_accepted_endpoints": STAGE_ACCEPTED_ENDPOINTS,
        "combined_if_complete": COMBINED_ACCEPTED_ENDPOINTS,
        "stage2_segment_numbers": segment_numbers,
        "blind_midpoint_segment_numbers": blind_numbers,
        "maximum_exact_field_and_retraction_units": (
            MAXIMUM_EXACT_FREE_FIELD_CALLS
        ),
        "prior_cumulative_phase_advance_radians": current_phase,
        "prior_cumulative_metric_path_length": float(
            seed["accumulated_metric_path_length"]
        ),
        "remaining_phase_to_2pi_radians": remaining,
        "minimum_recent_phase_increment": float(np.min(observed)),
        "median_recent_phase_increment": float(np.median(observed)),
        "maximum_recent_phase_increment": float(np.max(observed)),
        "projected_stage2_terminal_phase_at_recent_median": float(
            current_phase
            + STAGE_ACCEPTED_ENDPOINTS * float(np.median(observed))
        ),
        "maximum_stage2_terminal_phase_under_binding_increment_gate": float(
            current_phase
            + STAGE_ACCEPTED_ENDPOINTS * MAXIMUM_LOCAL_PHASE_INCREMENT
        ),
        "phase_lap_reachable_within_stage2_under_binding_increment_gate": bool(
            current_phase
            + STAGE_ACCEPTED_ENDPOINTS * MAXIMUM_LOCAL_PHASE_INCREMENT
            >= PHASE_LAP_RADIANS
        ),
        "segments_to_phase_lap_at_recent_minimum": int(
            math.ceil(remaining / float(np.min(observed)))
        ),
        "segments_to_phase_lap_at_recent_median": int(
            math.ceil(remaining / float(np.median(observed)))
        ),
        "measured_stage1_resume_wall_seconds_per_exact_unit": wall_per_unit,
        "projected_stage2_wall_hours": (
            MAXIMUM_EXACT_FREE_FIELD_CALLS * wall_per_unit / 3600.0
        ),
        "terminal_elapsed_seconds_if_complete": float(
            seed["elapsed_seconds"]
            + STAGE_ACCEPTED_ENDPOINTS * SEGMENT_SECONDS
        ),
    }


def _definitions(observations: dict) -> dict:
    minimum_terminal_phase = (
        observations["prior_cumulative_phase_advance_radians"]
        + MINIMUM_NEW_PHASE_ADVANCE
    )
    contract = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "authorized_execution": AUTHORIZED_NEXT,
        "preserved_classifications": {
            "original_stage1_chart_boundary": (
                parent.manifest.parent.manifest.parent.PHYSICAL_FAILURE_CLASSIFICATION
            ),
            "coupled_chart_boundary_recovery": parent.recovery.PASS_CLASSIFICATION,
            "stage1_completion": parent.PASS_CLASSIFICATION,
        },
        "scope": {
            "stage": 2,
            "prior_accepted_phase_endpoints": PRIOR_ACCEPTED_ENDPOINTS,
            "new_accepted_phase_endpoints": STAGE_ACCEPTED_ENDPOINTS,
            "combined_accepted_phase_endpoints_if_complete": (
                COMBINED_ACCEPTED_ENDPOINTS
            ),
            "maximum_attempted_endpoints": MAXIMUM_ATTEMPTED_ENDPOINTS,
            "segment_seconds": SEGMENT_SECONDS,
            "stage2_segment_numbers": observations["stage2_segment_numbers"],
            "blind_midpoint_frequency": BLIND_MIDPOINT_FREQUENCY,
            "blind_midpoint_segment_numbers": observations[
                "blind_midpoint_segment_numbers"
            ],
            "maximum_exact_free_field_calls": MAXIMUM_EXACT_FREE_FIELD_CALLS,
            "maximum_retractions": MAXIMUM_RETRACTIONS,
            "maximum_wall_hours": MAXIMUM_WALL_HOURS,
            "stop_immediately_on_first_binding_failure": True,
            "stop_at_first_phase_lap_or_registered_return_candidate": True,
        },
        "computational_chart": {
            "block_sizes": list(METRIC_BLOCK_SIZES),
            "maximum_metric_and_augmented_condition": (
                MAXIMUM_METRIC_CONDITION
            ),
            "same_coupled_physical_memory_chart_policy_as_recovery": True,
            "primitive_state_and_original_coordinate_unchanged": True,
            "all_physics_and_ledgers_in_original_coordinates": True,
        },
        "binding_stage2_gates": {
            "new_accepted_phase_endpoints": STAGE_ACCEPTED_ENDPOINTS,
            "combined_accepted_phase_endpoints": COMBINED_ACCEPTED_ENDPOINTS,
            "minimum_new_cumulative_phase_advance_radians": (
                MINIMUM_NEW_PHASE_ADVANCE
            ),
            "minimum_terminal_cumulative_phase_advance_radians": (
                minimum_terminal_phase
            ),
            "all_new_local_phase_increments_strictly_positive": True,
            "maximum_local_phase_increment": MAXIMUM_LOCAL_PHASE_INCREMENT,
            "maximum_phase_radial_defect": 0.002,
            "maximum_phase_out_of_plane_defect": 0.005,
            "maximum_phase_direction_prediction_defect_radians": 0.005,
            "all_original_retraction_and_physical_gates_unchanged": True,
            "all_checkpoint_roundtrips_bitwise": True,
            "suffix_history_replay_bitwise": True,
            "accepted_history_only_propagation": True,
        },
        "phase_lap_scope_proof": {
            "prior_phase_radians": observations[
                "prior_cumulative_phase_advance_radians"
            ],
            "maximum_terminal_phase_under_binding_increment_gate": (
                observations[
                    "maximum_stage2_terminal_phase_under_binding_increment_gate"
                ]
            ),
            "two_pi_radians": float(PHASE_LAP_RADIANS),
            "phase_lap_reachable_within_stage2": observations[
                "phase_lap_reachable_within_stage2_under_binding_increment_gate"
            ],
        },
        "classification_branches": {
            "all_48_endpoints_and_all_gates_pass": (
                "stage2 complete; authorize only a definitions-only event-limited stage3 manifest"
            ),
            "coarse_recurrence_candidate": (
                "authorize only exact registered-section return refinement"
            ),
            "phase_lap_without_recurrence": (
                "classify open/nonperiodic and stop acquisition"
            ),
            "any_physical_phase_numerical_or_replay_failure": (
                "stop without propagating the failed endpoint"
            ),
        },
        "forbidden": [
            "retroactively pass the rejected original stage1 boundary",
            "change the 10.0 metric condition gate",
            "change the physical equations, phase observer, or section",
            "execute more than the 48 stage2 endpoints",
            "treat stage2 as a phase lap or cycle certificate",
            "authorize complete-cycle execution or reduced slow evolution",
        ],
    }
    architecture = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "offline_truth_acquisition": (
            "continue the autonomous conservative field for exactly one "
            "bounded 48-endpoint tranche in the coupled physical-memory chart"
        ),
        "stage2_is_not_a_complete_cycle_execution": True,
        "stage2_phase_lap_is_impossible_under_its_binding_increment_ceiling": (
            not observations[
                "phase_lap_reachable_within_stage2_under_binding_increment_gate"
            ]
        ),
        "next_if_passed": (
            "a separately frozen event-limited stage3 acquisition sized from "
            "the remaining registered phase, with immediate stop at the first lap"
        ),
        "periodic_architecture_if_a_coarse_return_is_later_observed": {
            "refine": "exact registered-section root",
            "certify": "phase-conditioned multiple shooting or collocation",
            "continue": "periodic family across slow Q anchors",
            "average": "physical slow drift and bordered periodic adjoint",
        },
        "online_reduced_architecture_unchanged": (
            "evolve only slow Q, mode, and event state with a certified "
            "tabulated averaged drift; perform no truth integration, fixed-Q "
            "reaction solve, nonlinear root, or micro-BDF online"
        ),
        "no_recurrence_branch": (
            "do not force periodic averaging; test equilibrium, transient, "
            "or nonperiodic-attractor closure"
        ),
    }
    return {"contract": contract, "architecture": architecture}


def _evaluate(parent_lock: dict) -> tuple[dict, dict[str, np.ndarray], dict]:
    observations = _observations(parent_lock)
    seed = _stage2_seed()
    supported = bool(
        observations["prior_accepted_phase_endpoints"] == 48
        and observations["stage2_segment_numbers"] == list(range(245, 293))
        and observations["blind_midpoint_segment_numbers"]
        == [248, 256, 264, 272, 280, 288]
        and observations["maximum_exact_field_and_retraction_units"] == 54
        and observations["projected_stage2_wall_hours"] < MAXIMUM_WALL_HOURS
        and not observations[
            "phase_lap_reachable_within_stage2_under_binding_increment_gate"
        ]
        and seed["selected_metric_block_sizes"].tolist() == [442, 28]
    )
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            CLASSIFICATION
            if supported
            else "tangent_phase_lap_recurrence_stage2_not_supported"
        ),
        "passed": supported,
        "definitions_only": True,
        "new_truth_evaluations": 0,
        "new_retractions": 0,
        "new_accepted_segments": 0,
        "observations": observations,
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT if supported else None,
        "input_lock": {
            "parent_hashes": parent_lock["hashes"],
            "parent_classification": parent_lock["summary"]["classification"],
        },
    }
    return metrics, seed, _definitions(observations)


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
        raise RuntimeError("stage2 manifest result already exists")
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "stage2_metrics.json", metrics)
    helper._write_json(
        CANONICAL_DIRECTORY / "stage2_contract.json", definitions["contract"]
    )
    helper._write_json(
        CANONICAL_DIRECTORY / "mathematical_architecture.json",
        definitions["architecture"],
    )
    _save_npz(CANONICAL_DIRECTORY / "stage2_seed.npz", seed)
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
        "stage2_execution_authorized": metrics["passed"],
        "stage2_execution_executed": False,
        "phase_lap_execution_authorized": False,
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
                "# Tangent-phase-lap recurrence stage-2 manifest",
                "",
                f"Classification: `{metrics['classification']}`.",
                "",
                f"Stage 1 completed with 48 accepted phase endpoints at `{float(_stage2_seed()['elapsed_seconds']):.6f}` s and cumulative phase `{observations['prior_cumulative_phase_advance_radians']:.9f}` rad.",
                "",
                f"This package authorizes exactly 48 additional 0.25 ms endpoints, total segments 245--292, with blind midpoints at `{observations['blind_midpoint_segment_numbers']}`. The measured-cost projection is `{observations['projected_stage2_wall_hours']:.2f}` hours under a 7-hour cap.",
                "",
                f"The binding 0.08-rad local-increment ceiling bounds the stage-2 terminal phase by `{observations['maximum_stage2_terminal_phase_under_binding_increment_gate']:.9f}` rad, below 2*pi. Stage 2 therefore cannot itself be a phase-lap or cycle execution.",
                "",
                "The 442+28 coupled physical-memory chart, 10.0 condition gate, primitive-space physics, phase observer, recurrence section, checkpoint, suffix replay, and accepted-history-only rules remain unchanged.",
                "",
                f"Authorized next artifact: `{metrics['authorized_next']}`. A stage-2 pass may authorize only a separately frozen event-limited stage-3 manifest; complete-cycle execution and reduced slow evolution remain unauthorized.",
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
