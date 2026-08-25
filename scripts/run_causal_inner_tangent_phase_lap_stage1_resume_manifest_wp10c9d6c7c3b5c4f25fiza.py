#!/usr/bin/env python3
"""Freeze the five-endpoint resume needed to complete phase-lap stage 1."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_coupled_physical_memory_metric_chart_boundary_recovery_execution_wp10c9d6c7c3b5c4f25fiz as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = parent.AUTHORIZED_NEXT
CLASSIFICATION = "tangent_phase_lap_stage1_resume_selected_definitions_only"
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizb_"
    "tangent_phase_lap_stage1_resume_execution"
)
ARTIFACT = (
    "causal_inner_tangent_phase_lap_stage1_resume_manifest_"
    "wp10c9d6c7c3b5c4f25fiza"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_TANGENT_PHASE_LAP_STAGE1_"
    "RESUME_MANIFEST_WP10C9D6C7C3B5C4F25FIZA_2026-08-25.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_tangent_phase_lap_stage1_resume_manifest_"
    "wp10c9d6c7c3b5c4f25fiza.py"
)
THIS_TEST = (
    "tests/test_causal_inner_tangent_phase_lap_stage1_resume_manifest_"
    "wp10c9d6c7c3b5c4f25fiza.py"
)

ORIGINAL_STAGE_TARGET = 48
PRIOR_STAGE_ACCEPTED = 42
RECOVERY_ACCEPTED = 1
COMBINED_ACCEPTED = PRIOR_STAGE_ACCEPTED + RECOVERY_ACCEPTED
REMAINING_ACCEPTED = ORIGINAL_STAGE_TARGET - COMBINED_ACCEPTED
MAXIMUM_ATTEMPTS = REMAINING_ACCEPTED
SEGMENT_SECONDS = 2.5e-4
BLIND_MIDPOINT_FREQUENCY = 8
MAXIMUM_WALL_HOURS = 1.5
METRIC_BLOCK_SIZES = (442, 28)
MAXIMUM_METRIC_CONDITION = 10.0
MINIMUM_FINAL_CUMULATIVE_PHASE = 1.5


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
        parent.manifest.parent.THIS_RUNNER,
        parent.manifest.parent.THIS_TEST,
        parent.parent.manifest.THIS_RUNNER,
        parent.parent.manifest.THIS_TEST,
        parent.engine.THIS_RUNNER,
        parent.suffix.THIS_RUNNER,
    )
    return {name: helper._sha(ROOT / name) for name in paths}


def _validate_parent(*, require_clean: bool) -> dict:
    helper = _helper()
    hashes = helper._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = helper._read(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = helper._read(
        parent.CANONICAL_DIRECTORY / "boundary_recovery_metrics.json"
    )
    provenance = helper._read(parent.CANONICAL_DIRECTORY / "provenance.json")
    values = metrics["gate_values"]
    rejected_parent_summary = helper._read(
        parent.manifest.parent.CANONICAL_DIRECTORY / "summary.json"
    )
    if (
        summary["classification"] != parent.PASS_CLASSIFICATION
        or not summary["passed"]
        or summary["new_accepted_segments"] != RECOVERY_ACCEPTED
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["phase_lap_observed"]
        or summary["coarse_recurrence_candidate_observed"]
        or summary["complete_cycle_execution_authorized"]
        or summary["reduced_slow_evolution_authorized"]
        or metrics["classification"] != parent.PASS_CLASSIFICATION
        or not metrics["passed"]
        or values["accepted_segments"] != RECOVERY_ACCEPTED
        or values["attempted_segments"] != 1
        or values["selected_metric_block_sizes"] != [442, 28]
        or values["maximum_metric_coordinate_jacobian_condition"] > 10.0
        or not values["candidate_target_matches_frozen_seed_bitwise"]
        or not values["phase_geometry_passed"]
        or not values["all_accepted_checkpoint_roundtrips_bitwise"]
        or not values["suffix_history_replay_bitwise"]
        or rejected_parent_summary["classification"]
        != parent.manifest.parent.PHYSICAL_FAILURE_CLASSIFICATION
        or rejected_parent_summary["passed"]
    ):
        raise RuntimeError("boundary recovery certificate changed")
    for relative, expected in provenance["source_hashes"].items():
        if helper._sha(ROOT / relative) != expected:
            raise RuntimeError(f"boundary recovery source changed: {relative}")
    parent._validate_manifest(require_clean=False)
    if require_clean and helper._git(
        "status", "--short", "--untracked-files=no"
    ):
        raise RuntimeError("stage1 resume manifest requires a clean tracked tree")
    return {
        "hashes": hashes,
        "summary": summary,
        "metrics": metrics,
        "provenance": provenance,
        "rejected_parent_summary": rejected_parent_summary,
    }


def _resume_seed() -> dict[str, np.ndarray]:
    base = _load_npz(
        parent.manifest.CANONICAL_DIRECTORY / "recovery_seed.npz"
    )
    recovery = _load_npz(
        parent.CANONICAL_DIRECTORY / "boundary_recovery_arrays.npz"
    )
    metrics = _helper()._read(
        parent.CANONICAL_DIRECTORY / "boundary_recovery_metrics.json"
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
    seed = {name: np.asarray(recovery[name]) for name in progress_names}
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
            "accepted_endpoint_coordinates470": np.vstack(
                (
                    base["accepted_endpoint_coordinates470"],
                    recovery["accepted_endpoint_coordinates470"],
                )
            ),
            "accepted_endpoint_primitive_states": np.concatenate(
                (
                    base["accepted_endpoint_primitive_states"],
                    recovery["accepted_endpoint_primitive_states"],
                ),
                axis=0,
            ),
            "accepted_endpoint_coordinate_rates470_per_s": np.vstack(
                (
                    base["accepted_endpoint_coordinate_rates470_per_s"],
                    recovery["accepted_endpoint_coordinate_rates470_per_s"],
                )
            ),
            "accepted_phase_increments": np.concatenate(
                (
                    base["accepted_phase_increments"],
                    np.atleast_1d(recovery["new_phase_increment"]),
                )
            ),
            "accepted_cumulative_phase_advance_radians": np.concatenate(
                (
                    base["accepted_cumulative_phase_advance_radians"],
                    np.atleast_1d(
                        recovery["cumulative_phase_advance_radians"]
                    ),
                )
            ),
            "accepted_cumulative_metric_path_lengths": np.concatenate(
                (
                    base["accepted_cumulative_metric_path_lengths"],
                    np.atleast_1d(recovery["cumulative_metric_path_length"]),
                )
            ),
            "accepted_registered_section_values": np.concatenate(
                (
                    base["accepted_registered_section_values"],
                    np.asarray([metrics["registered_section_value"]]),
                )
            ),
            "unwrapped_phase_advance_radians": recovery[
                "cumulative_phase_advance_radians"
            ],
            "accumulated_metric_path_length": recovery[
                "cumulative_metric_path_length"
            ],
            "selected_metric_block_sizes": np.asarray(
                METRIC_BLOCK_SIZES, dtype=np.int64
            ),
            "accepted_segments_new": np.asarray(0),
            "attempts": np.asarray(0),
            "acquisition_stage": np.asarray(1),
            "metric_chart_generation": np.asarray(1),
        }
    )
    if (
        seed["accepted_endpoint_coordinates470"].shape != (COMBINED_ACCEPTED, 470)
        or seed["accepted_endpoint_primitive_states"].shape
        != (COMBINED_ACCEPTED, 112, 5)
        or seed["accepted_endpoint_coordinate_rates470_per_s"].shape
        != (COMBINED_ACCEPTED, 470)
        or seed["accepted_phase_increments"].shape != (COMBINED_ACCEPTED,)
        or int(seed["accepted_segments_total"]) != 239
        or float(seed["elapsed_seconds"]) != 0.17875000000000013
    ):
        raise RuntimeError("stage1 resume seed changed")
    return seed


def _observations(parent_lock: dict) -> dict:
    values = parent_lock["metrics"]["gate_values"]
    seed = _resume_seed()
    segment_numbers = [
        int(seed["accepted_segments_total"]) + index
        for index in range(1, REMAINING_ACCEPTED + 1)
    ]
    blind_numbers = [
        value
        for value in segment_numbers
        if value % BLIND_MIDPOINT_FREQUENCY == 0
    ]
    exact_units = REMAINING_ACCEPTED + len(blind_numbers)
    wall_per_unit = float(
        values["execution_wall_seconds"]
        / values["exact_free_field_calls"]
    )
    return {
        "prior_stage_accepted": PRIOR_STAGE_ACCEPTED,
        "boundary_recovery_accepted": RECOVERY_ACCEPTED,
        "combined_accepted": COMBINED_ACCEPTED,
        "remaining_accepted": REMAINING_ACCEPTED,
        "resume_segment_numbers": segment_numbers,
        "blind_midpoint_segment_numbers": blind_numbers,
        "maximum_exact_field_and_retraction_units": exact_units,
        "prior_cumulative_phase_advance_radians": float(
            seed["unwrapped_phase_advance_radians"]
        ),
        "prior_cumulative_metric_path_length": float(
            seed["accumulated_metric_path_length"]
        ),
        "measured_recovery_wall_seconds_per_unit": wall_per_unit,
        "projected_resume_wall_hours": exact_units * wall_per_unit / 3600.0,
        "terminal_elapsed_seconds_if_complete": float(
            seed["elapsed_seconds"] + REMAINING_ACCEPTED * SEGMENT_SECONDS
        ),
    }


def _definitions(observations: dict) -> dict:
    contract = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "authorized_execution": AUTHORIZED_NEXT,
        "preserved_classifications": {
            "original_stage1_boundary": (
                parent.manifest.parent.PHYSICAL_FAILURE_CLASSIFICATION
            ),
            "prospective_chart_recovery": parent.PASS_CLASSIFICATION,
        },
        "scope": {
            "complete_only_the_original_48_endpoint_stage1": True,
            "prior_accepted_endpoints": COMBINED_ACCEPTED,
            "new_accepted_endpoints": REMAINING_ACCEPTED,
            "maximum_attempted_endpoints": MAXIMUM_ATTEMPTS,
            "segment_seconds": SEGMENT_SECONDS,
            "blind_midpoint_frequency": BLIND_MIDPOINT_FREQUENCY,
            "blind_midpoint_segment_numbers": observations[
                "blind_midpoint_segment_numbers"
            ],
            "maximum_exact_free_field_calls": observations[
                "maximum_exact_field_and_retraction_units"
            ],
            "maximum_retractions": observations[
                "maximum_exact_field_and_retraction_units"
            ],
            "maximum_wall_hours": MAXIMUM_WALL_HOURS,
            "stop_immediately_on_first_binding_failure": True,
        },
        "computational_chart": {
            "block_sizes": list(METRIC_BLOCK_SIZES),
            "maximum_metric_and_augmented_condition": (
                MAXIMUM_METRIC_CONDITION
            ),
            "primitive_state_and_original_coordinate_unchanged": True,
            "all_physics_and_ledgers_in_original_coordinates": True,
        },
        "binding_completion_gates": {
            "combined_accepted_endpoints": ORIGINAL_STAGE_TARGET,
            "minimum_final_cumulative_phase_advance_radians": (
                MINIMUM_FINAL_CUMULATIVE_PHASE
            ),
            "all_new_local_phase_increments_strictly_positive": True,
            "maximum_local_phase_increment": 0.08,
            "maximum_phase_radial_defect": 0.002,
            "maximum_phase_out_of_plane_defect": 0.005,
            "maximum_phase_direction_prediction_defect_radians": 0.005,
            "all_original_retraction_and_physical_gates_unchanged": True,
            "all_checkpoint_roundtrips_bitwise": True,
            "suffix_history_replay_bitwise": True,
            "accepted_history_only_propagation": True,
        },
        "classification_branches": {
            "five_endpoints_and_all_gates_pass": (
                "stage1 complete; authorize only a definitions-only stage2 resume manifest"
            ),
            "coarse_recurrence_candidate": (
                "authorize only exact registered-section return refinement"
            ),
            "phase_lap_without_recurrence": (
                "classify open/nonperiodic and do not continue stage acquisition"
            ),
            "any_physical_phase_numerical_or_replay_failure": (
                "stop without propagating the failed endpoint"
            ),
        },
        "forbidden": [
            "retroactively pass the rejected f25fix execution",
            "change the 10.0 metric condition gate",
            "change the physical equations, phase observer, or section",
            "execute more than the five remaining endpoints",
            "authorize a complete cycle or reduced slow evolution",
        ],
    }
    architecture = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "offline_truth_acquisition": (
            "continue the autonomous conservative field in the coupled "
            "physical-memory numerical chart while measuring phase and "
            "recurrence in the frozen 168 ms observer metric"
        ),
        "chart_is_not_the_reduced_model": True,
        "stage1_completion_is_not_a_phase_lap_or_cycle": True,
        "next_if_passed": (
            "a separately frozen 48-endpoint stage2 tranche carrying the "
            "same accepted history and adaptive nested chart policy"
        ),
        "online_reduced_architecture_unchanged": (
            "after a periodic family is certified, evolve only slow Q, mode, "
            "and event state with tabulated averaged drift; perform no truth "
            "integration, reaction solve, nonlinear root, or micro-BDF online"
        ),
    }
    return {"contract": contract, "architecture": architecture}


def _evaluate(parent_lock: dict) -> tuple[dict, dict[str, np.ndarray], dict]:
    observations = _observations(parent_lock)
    seed = _resume_seed()
    supported = bool(
        observations["remaining_accepted"] == 5
        and observations["blind_midpoint_segment_numbers"] == [240]
        and observations["maximum_exact_field_and_retraction_units"] == 6
        and observations["projected_resume_wall_hours"] < MAXIMUM_WALL_HOURS
        and observations["prior_cumulative_phase_advance_radians"]
        >= MINIMUM_FINAL_CUMULATIVE_PHASE
        and seed["selected_metric_block_sizes"].tolist() == [442, 28]
    )
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            CLASSIFICATION
            if supported
            else "tangent_phase_lap_stage1_resume_not_supported"
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
        raise RuntimeError("stage1 resume manifest result already exists")
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "resume_metrics.json", metrics)
    helper._write_json(
        CANONICAL_DIRECTORY / "resume_contract.json", definitions["contract"]
    )
    helper._write_json(
        CANONICAL_DIRECTORY / "mathematical_architecture.json",
        definitions["architecture"],
    )
    _save_npz(CANONICAL_DIRECTORY / "resume_seed.npz", seed)
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
        "stage1_resume_execution_authorized": metrics["passed"],
        "stage1_resume_execution_executed": False,
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
                "# Tangent-phase-lap stage-1 resume manifest",
                "",
                f"Classification: `{metrics['classification']}`.",
                "",
                "The original stage-1 execution remains rejected at its three-block chart boundary. The separately prospective coupled physical-memory recovery accepted exactly one replacement endpoint under every unchanged gate.",
                "",
                f"The accepted chain now contains `{observations['combined_accepted']}` of the original 48 endpoints and cumulative phase `{observations['prior_cumulative_phase_advance_radians']:.9f}` rad. This package authorizes only the remaining `{observations['remaining_accepted']}` endpoints, total segments `{observations['resume_segment_numbers']}`.",
                "",
                f"One blind midpoint is scheduled at total segment `{observations['blind_midpoint_segment_numbers'][0]}`, for six exact field/retraction units and a measured-cost projection of `{observations['projected_resume_wall_hours']:.2f}` hours.",
                "",
                "The 442+28 chart, 10.0 condition gate, primitive-space physics, phase observer, recurrence section, checkpoint, and accepted-history-only rules remain binding. Any failed endpoint stops without propagation.",
                "",
                f"Authorized next artifact: `{metrics['authorized_next']}`. Stage-1 completion would still not be a phase lap, cycle, or reduced slow evolution certificate.",
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
