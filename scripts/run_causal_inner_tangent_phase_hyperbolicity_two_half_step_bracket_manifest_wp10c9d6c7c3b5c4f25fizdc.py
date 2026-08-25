#!/usr/bin/env python3
"""Freeze two authentic half steps at the stage-2 hyperbolicity boundary."""

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

import run_causal_inner_tangent_phase_lap_stage2_hyperbolicity_boundary_diagnostic_wp10c9d6c7c3b5c4f25fizdb as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = parent.AUTHORIZED_NEXT
CLASSIFICATION = (
    "tangent_phase_hyperbolicity_two_half_step_bracket_"
    "selected_definitions_only"
)
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizdd_"
    "tangent_phase_hyperbolicity_two_half_step_bracket_execution"
)
ARTIFACT = (
    "causal_inner_tangent_phase_hyperbolicity_two_half_step_bracket_manifest_"
    "wp10c9d6c7c3b5c4f25fizdc"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_TANGENT_PHASE_"
    "HYPERBOLICITY_TWO_HALF_STEP_BRACKET_MANIFEST_"
    "WP10C9D6C7C3B5C4F25FIZDC_2026-08-25.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_tangent_phase_hyperbolicity_"
    "two_half_step_bracket_manifest_wp10c9d6c7c3b5c4f25fizdc.py"
)
THIS_TEST = (
    "tests/test_causal_inner_tangent_phase_hyperbolicity_"
    "two_half_step_bracket_manifest_wp10c9d6c7c3b5c4f25fizdc.py"
)

FULL_STEP_SECONDS = 2.5e-4
HALF_STEP_SECONDS = 1.25e-4
MAXIMUM_ACCEPTED_HALF_STEPS = 2
MAXIMUM_ATTEMPTED_HALF_STEPS = 2
MAXIMUM_EXACT_FREE_FIELD_CALLS = 2
MAXIMUM_RETRACTIONS = 2
MAXIMUM_WALL_HOURS = 1.0
FIRST_TENTATIVE_SEGMENT = 268
TENTATIVE_SEGMENT_NUMBERS = (268, 269)
BLIND_MIDPOINT_SEGMENT_NUMBERS: tuple[int, ...] = ()
MAXIMUM_REAL_SPECTRUM_IMAGINARY_SPEED = 1.0e-10
COMPLEX_SPECTRUM_IMAGINARY_SPEED = 1.0e-8


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
    boundary = parent.manifest
    stage2 = boundary.parent
    paths = (
        THIS_RUNNER,
        THIS_TEST,
        parent.THIS_RUNNER,
        parent.THIS_TEST,
        boundary.THIS_RUNNER,
        boundary.THIS_TEST,
        stage2.THIS_RUNNER,
        stage2.THIS_TEST,
        stage2.manifest.THIS_RUNNER,
        stage2.manifest.THIS_TEST,
        stage2.phase.THIS_RUNNER,
        stage2.phase.THIS_TEST,
        stage2.engine.THIS_RUNNER,
        stage2.engine.THIS_TEST,
        stage2.suffix.THIS_RUNNER,
        "src/imri_qpe/layer3_minidisk_1d/causal_inner_radial_linear_tangent.py",
        "src/imri_qpe/layer3_minidisk_1d/causal_inner_characteristic_dissipation.py",
    )
    return {name: helper._sha(ROOT / name) for name in paths}


def _validate_parent(*, require_clean: bool) -> dict:
    helper = _helper()
    hashes = helper._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = helper._read(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = helper._read(
        parent.CANONICAL_DIRECTORY / "diagnostic_metrics.json"
    )
    provenance = helper._read(parent.CANONICAL_DIRECTORY / "provenance.json")
    boundary_seed = _load_npz(
        parent.manifest.CANONICAL_DIRECTORY / "boundary_seed.npz"
    )
    if (
        summary["classification"] != parent.PASS_CLASSIFICATION
        or not summary["passed"]
        or not summary["nonpropagating"]
        or summary["failed_candidate_propagated"]
        or summary["first_complex_face"] != 3
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["complete_cycle_execution_authorized"]
        or summary["reduced_slow_evolution_authorized"]
        or metrics["classification"] != parent.PASS_CLASSIFICATION
        or not metrics["passed"]
        or not metrics["nonpropagating"]
        or metrics["new_free_field_calls"] != 0
        or metrics["new_retractions"] != 0
        or metrics["analytic_maximum_imaginary_speed"] < 1.0e-8
        or metrics["last_scanned_real_fraction"] < 0.95
        or metrics["first_scanned_complex_fraction"] != 1.0
        or boundary_seed["combined_accepted_endpoint_coordinates470"].shape
        != (71, 470)
        or int(boundary_seed["accepted_segments_total"]) != 267
        or float(boundary_seed["elapsed_seconds"])
        != 0.18575000000000014
    ):
        raise RuntimeError("hyperbolicity boundary diagnosis changed")
    for relative, expected in provenance["source_hashes"].items():
        if helper._sha(ROOT / relative) != expected:
            raise RuntimeError(f"boundary diagnostic source changed: {relative}")
    parent._validate_manifest(require_clean=False)
    if require_clean and helper._git(
        "status", "--short", "--untracked-files=no"
    ):
        raise RuntimeError("two-half-step manifest requires a clean tracked tree")
    return {
        "hashes": hashes,
        "summary": summary,
        "metrics": metrics,
        "provenance": provenance,
    }


def _half_step_seed() -> dict[str, np.ndarray]:
    boundary = _load_npz(
        parent.manifest.CANONICAL_DIRECTORY / "boundary_seed.npz"
    )
    stage2_seed = parent.manifest.parent._seed()
    seed = {name: np.asarray(value) for name, value in boundary.items()}
    aliases = {
        "accepted_endpoint_coordinates470": (
            "combined_accepted_endpoint_coordinates470"
        ),
        "accepted_endpoint_primitive_states": (
            "combined_accepted_endpoint_primitive_states"
        ),
        "accepted_endpoint_coordinate_rates470_per_s": (
            "combined_accepted_endpoint_coordinate_rates470_per_s"
        ),
        "accepted_phase_increments": "combined_accepted_phase_increments",
        "accepted_cumulative_phase_advance_radians": (
            "combined_cumulative_phase_advance_radians"
        ),
        "accepted_cumulative_metric_path_lengths": (
            "combined_cumulative_metric_path_lengths"
        ),
        "accepted_registered_section_values": (
            "combined_registered_section_values"
        ),
    }
    for target, source in aliases.items():
        seed[target] = boundary[source].copy()
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
        seed[name] = stage2_seed[name].copy()
    seed.update(
        {
            "next_span_seconds": np.asarray(HALF_STEP_SECONDS),
            "accepted_segments_new": np.asarray(0, dtype=np.int64),
            "attempts": np.asarray(0, dtype=np.int64),
            "accepted_since_growth": np.asarray(0, dtype=np.int64),
            "unwrapped_phase_advance_radians": boundary[
                "cumulative_phase_advance_radians"
            ].copy(),
            "accumulated_metric_path_length": boundary[
                "cumulative_metric_path_length"
            ].copy(),
            "acquisition_stage": np.asarray(2, dtype=np.int64),
            "metric_chart_generation": np.asarray(1, dtype=np.int64),
        }
    )
    if (
        seed["accepted_endpoint_coordinates470"].shape != (71, 470)
        or seed["accepted_endpoint_primitive_states"].shape != (71, 112, 5)
        or seed["accepted_endpoint_coordinate_rates470_per_s"].shape
        != (71, 470)
        or float(seed["previous_span_seconds"]) != FULL_STEP_SECONDS
        or float(seed["next_span_seconds"]) != HALF_STEP_SECONDS
        or int(seed["accepted_segments_total"]) != 267
        or int(seed["accepted_segments_new"]) != 0
    ):
        raise RuntimeError("two-half-step seed changed")
    return seed


def _contract() -> dict:
    phase_contract = parent.manifest.parent.phase._contract()
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "authorized_execution": AUTHORIZED_NEXT,
        "preserved_boundary": {
            "diagnostic_classification": parent.PASS_CLASSIFICATION,
            "accepted_endpoint_count": 71,
            "accepted_terminal_elapsed_seconds": 0.18575000000000014,
            "accepted_total_segments": 267,
            "failed_full_step_seconds": FULL_STEP_SECONDS,
            "failed_full_step_total_segment": 268,
            "failed_full_step_propagated": False,
            "failed_full_step_remains_rejected": True,
        },
        "scope": {
            "truth_dynamics": "dq/dt=f_free(q) on the certified conservative atlas",
            "half_step_seconds": HALF_STEP_SECONDS,
            "maximum_accepted_half_steps": MAXIMUM_ACCEPTED_HALF_STEPS,
            "maximum_attempted_half_steps": MAXIMUM_ATTEMPTED_HALF_STEPS,
            "tentative_segment_numbers": list(TENTATIVE_SEGMENT_NUMBERS),
            "blind_midpoint_segment_numbers": list(
                BLIND_MIDPOINT_SEGMENT_NUMBERS
            ),
            "maximum_exact_free_field_calls": MAXIMUM_EXACT_FREE_FIELD_CALLS,
            "maximum_retractions": MAXIMUM_RETRACTIONS,
            "maximum_wall_hours": MAXIMUM_WALL_HOURS,
            "first_step_uses_authentic_full_step_history": True,
            "second_step_uses_only_the_accepted_first_half_step_history": True,
            "failed_full_step_is_diagnostic_only": True,
        },
        "binding_endpoint_sequence": [
            "variable-step AB2 prediction from accepted history",
            "strict metric-chart retraction",
            "analytic all-face real-spectrum preflight",
            "exact conservative free field",
            "endpoint integral and all original physical gates",
            "phase and recurrence geometry",
            "accepted checkpoint roundtrip",
        ],
        "binding_hyperbolicity_gate": {
            "all_113_face_generalized_pencils_checked": True,
            "maximum_imaginary_coordinate_speed": (
                MAXIMUM_REAL_SPECTRUM_IMAGINARY_SPEED
            ),
            "complex_boundary_diagnostic_threshold": (
                COMPLEX_SPECTRUM_IMAGINARY_SPEED
            ),
            "checked_after_retraction_before_exact_field": True,
            "no_complex_flux_split": True,
            "no_real_part_coercion": True,
        },
        "unchanged_original_gates": phase_contract["binding_stage1_gates"],
        "classification_branches": {
            "both_authentic_half_steps_pass": (
                "full-step predictor overshoot supported; authorize only a "
                "definitions-only halved-step stage2 continuation manifest"
            ),
            "first_half_step_is_complex": (
                "hyperbolicity boundary lies before the first half step; "
                "propagate nothing and stop"
            ),
            "first_passes_second_is_complex": (
                "accept only the first half step, bracket the trajectory "
                "hyperbolicity boundary, and stop"
            ),
            "physical_or_numerical_gate_fails": (
                "preserve the original failure class and stop without "
                "propagating the failed candidate"
            ),
        },
        "comparison_if_both_pass": {
            "saved_failed_full_step_state_is_diagnostic_only": True,
            "compare_final_coordinate_and_primitive_state": True,
            "comparison_is_not_an_acceptance_gate": True,
            "old_failed_state_is_never_reclassified_or_propagated": True,
        },
        "forbidden": [
            "propagate the saved failed full-step candidate",
            "propagate any endpoint with a complex characteristic pencil",
            "replace a complex eigenvalue by its real part",
            "relax any original physical, retraction, phase, or replay gate",
            "add a blind midpoint not prospectively listed",
            "authorize a phase lap, complete cycle, or reduced slow evolution",
        ],
    }


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
                    "scientific_status": "DEFINITIONS_ONLY",
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


def _freeze() -> dict:
    helper = _helper()
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("two-half-step bracket manifest already exists")
    validated = _validate_parent(require_clean=True)
    contract = _contract()
    seed = _half_step_seed()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(
        CANONICAL_DIRECTORY / "two_half_step_contract.json", contract
    )
    _save_npz(CANONICAL_DIRECTORY / "two_half_step_seed.npz", seed)
    helper._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "boundary_diagnostic_hashes": validated["hashes"],
            "boundary_classification": validated["summary"]["classification"],
            "boundary_seed_sha256": helper._sha(
                parent.manifest.CANONICAL_DIRECTORY / "boundary_seed.npz"
            ),
            "diagnostic_arrays_sha256": helper._sha(
                parent.CANONICAL_DIRECTORY / "diagnostic_arrays.npz"
            ),
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "half_step_seconds": HALF_STEP_SECONDS,
        "maximum_accepted_half_steps": MAXIMUM_ACCEPTED_HALF_STEPS,
        "new_free_field_calls": 0,
        "new_retractions": 0,
        "failed_full_step_propagated": False,
        "half_step_execution_authorized": True,
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
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
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Tangent-phase hyperbolicity two-half-step bracket manifest",
                "",
                f"Classification: `{CLASSIFICATION}`.",
                "",
                "The genuine complex pencil at the rejected 0.25 ms predictor is preserved. The saved failed state remains diagnostic-only and is never propagated.",
                "",
                "The next package may attempt at most two authentic 0.125 ms steps. Each candidate is strictly retracted, checked for a real generalized characteristic spectrum at every face, and only then passed to the unchanged exact field, physical, phase, and checkpoint gates.",
                "",
                "If the first half step is complex, nothing advances. If only the second is complex, exactly one accepted half step may remain in the bracket evidence. Only two accepted hyperbolic half steps can support the predictor-overshoot interpretation.",
                "",
                f"Authorized next artifact: `{AUTHORIZED_NEXT}`. Complete-cycle execution and reduced slow evolution remain unauthorized.",
                "",
            )
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    arguments = parser.parse_args()
    if not arguments.freeze:
        parser.error("--freeze is required")
    print(json.dumps(_freeze(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
