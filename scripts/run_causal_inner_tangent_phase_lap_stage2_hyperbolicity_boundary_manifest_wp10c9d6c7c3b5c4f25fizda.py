#!/usr/bin/env python3
"""Freeze the interrupted stage-2 prefix and its hyperbolicity diagnosis."""

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

import run_causal_inner_tangent_phase_lap_recurrence_stage2_execution_wp10c9d6c7c3b5c4f25fizd as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fizda_"
    "tangent_phase_lap_stage2_hyperbolicity_boundary_manifest"
)
CLASSIFICATION = (
    "tangent_phase_lap_stage2_hyperbolicity_boundary_diagnosis_"
    "selected_definitions_only"
)
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizdb_"
    "tangent_phase_lap_stage2_hyperbolicity_boundary_diagnostic"
)
ARTIFACT = (
    "causal_inner_tangent_phase_lap_stage2_hyperbolicity_boundary_manifest_"
    "wp10c9d6c7c3b5c4f25fizda"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_TANGENT_PHASE_LAP_STAGE2_"
    "HYPERBOLICITY_BOUNDARY_MANIFEST_WP10C9D6C7C3B5C4F25FIZDA_"
    "2026-08-25.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_tangent_phase_lap_stage2_"
    "hyperbolicity_boundary_manifest_wp10c9d6c7c3b5c4f25fizda.py"
)
THIS_TEST = (
    "tests/test_causal_inner_tangent_phase_lap_stage2_"
    "hyperbolicity_boundary_manifest_wp10c9d6c7c3b5c4f25fizda.py"
)

ACCEPTED_STAGE2_ENDPOINTS = 23
PRIOR_STAGE1_ENDPOINTS = 48
COMBINED_ACCEPTED_ENDPOINTS = 71
FAILED_ATTEMPT_INDEX = 23
FAILED_TOTAL_SEGMENT = 268
MAXIMUM_DIAGNOSTIC_WALL_HOURS = 0.25
ANALYTIC_IMAGINARY_LOWER_GATE = 1.0e-8
FINITE_DIFFERENCE_STEPS = (1.0e-3, 2.0e-4, 2.0e-5)
FINITE_DIFFERENCE_IMAGINARY_LOWER_GATE = 1.0e-8
FINITE_DIFFERENCE_TO_ANALYTIC_RELATIVE_GATE = 0.02
INTERPOLATION_SCAN_FRACTIONS = tuple(np.linspace(0.0, 1.0, 21))


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
        parent.phase.THIS_RUNNER,
        parent.phase.THIS_TEST,
        parent.engine.THIS_RUNNER,
        parent.engine.THIS_TEST,
        parent.suffix.THIS_RUNNER,
        "src/imri_qpe/layer3_minidisk_1d/causal_inner_radial_linear_tangent.py",
        "src/imri_qpe/layer3_minidisk_1d/causal_inner_characteristic_dissipation.py",
    )
    return {name: helper._sha(ROOT / name) for name in paths}


def _attempt_directory(index: int) -> Path:
    return parent.SCRATCH_DIRECTORY / f"attempt_{index:04d}"


def _decisive_scratch_paths() -> list[Path]:
    paths = [
        parent.SCRATCH_DIRECTORY / "execution_identity.json",
        parent.SCRATCH_DIRECTORY / "cumulative_wall_seconds.json",
    ]
    accepted_names = (
        "attempt.json",
        "attempt.npz",
        "accepted_checkpoint.npz",
        "endpoint_retraction.json",
        "endpoint_retraction.npz",
        "endpoint_field.json",
        "endpoint_field.npz",
        "phase_prediction.json",
        "phase_prediction.npz",
    )
    for index in range(ACCEPTED_STAGE2_ENDPOINTS):
        directory = _attempt_directory(index)
        paths.extend(directory / name for name in accepted_names)
        if (directory / "midpoint_field.json").exists():
            paths.extend(
                directory / name
                for name in (
                    "midpoint_retraction.json",
                    "midpoint_retraction.npz",
                    "midpoint_field.json",
                    "midpoint_field.npz",
                )
            )
    failure = _attempt_directory(FAILED_ATTEMPT_INDEX)
    paths.extend(
        failure / name
        for name in (
            "phase_prediction.json",
            "phase_prediction.npz",
            "endpoint_retraction.json",
            "endpoint_retraction.npz",
        )
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"stage2 boundary scratch is incomplete: {missing[0]}")
    return paths


def _scratch_hashes() -> dict[str, str]:
    helper = _helper()
    return {
        str(path.relative_to(ROOT)): helper._sha(path)
        for path in _decisive_scratch_paths()
    }


def _validate_execution_contract(*, require_clean: bool) -> dict:
    helper = _helper()
    lock = parent._validate_manifest(require_clean=False)
    identity = helper._read(parent.SCRATCH_DIRECTORY / "execution_identity.json")
    expected_identity = parent._identity(lock)
    if identity != expected_identity:
        raise RuntimeError("stage2 execution identity changed")
    if (parent.CANONICAL_DIRECTORY.exists() or parent.REPORT_PATH.exists()):
        raise RuntimeError("interrupted stage2 unexpectedly has a canonical result")
    records = []
    for index in range(ACCEPTED_STAGE2_ENDPOINTS):
        directory = _attempt_directory(index)
        metrics = helper._read(directory / "attempt.json")
        if (
            metrics["attempt_index"] != index
            or not metrics["accepted"]
            or metrics["physical_failure"]
            or metrics["retryable_chart_failure"]
            or not metrics["phase_geometry"]["passed"]
            or metrics["recurrence_geometry"]["phase_lap_observed"]
            or metrics["recurrence_geometry"]["coarse_recurrence_candidate"]
            or not metrics["endpoint_field"]["physical_passed"]
            or metrics["endpoint_field"]["metric_chart"]["block_sizes"]
            != [442, 28]
        ):
            raise RuntimeError(f"accepted stage2 prefix changed at {index}")
        records.append(metrics)
    failed = _attempt_directory(FAILED_ATTEMPT_INDEX)
    retraction = helper._read(failed / "endpoint_retraction.json")
    prediction = helper._read(failed / "phase_prediction.json")
    forbidden = (
        "attempt.json",
        "attempt.npz",
        "endpoint_field.json",
        "endpoint_field.npz",
        "accepted_checkpoint.npz",
    )
    if (
        prediction["attempt_index"] != FAILED_ATTEMPT_INDEX
        or prediction["tentative_segment_number"] != FAILED_TOTAL_SEGMENT
        or not retraction["passed"]
        or not retraction["physical_passed"]
        or not retraction["chart_condition_passed"]
        or retraction["maximum_metric_augmented_condition_number"] > 10.0
        or any((failed / name).exists() for name in forbidden)
    ):
        raise RuntimeError("nonpropagated stage2 boundary changed")
    terminal = _load_npz(
        _attempt_directory(ACCEPTED_STAGE2_ENDPOINTS - 1)
        / "accepted_checkpoint.npz"
    )
    if (
        int(terminal["accepted_segments_new"]) != ACCEPTED_STAGE2_ENDPOINTS
        or int(terminal["accepted_segments_total"]) != 267
        or int(terminal["attempts"]) != ACCEPTED_STAGE2_ENDPOINTS
        or float(terminal["elapsed_seconds"]) != 0.18575000000000014
    ):
        raise RuntimeError("accepted stage2 terminal checkpoint changed")
    if require_clean and helper._git(
        "status", "--short", "--untracked-files=no"
    ):
        raise RuntimeError("hyperbolicity manifest requires a clean tracked tree")
    return {
        "manifest_lock": lock,
        "identity": identity,
        "records": records,
        "retraction": retraction,
        "prediction": prediction,
        "terminal": terminal,
        "scratch_hashes": _scratch_hashes(),
    }


def _boundary_seed(validated: dict) -> dict[str, np.ndarray]:
    stage2_seed = parent._seed()
    terminal = validated["terminal"]
    failed = _load_npz(
        _attempt_directory(FAILED_ATTEMPT_INDEX) / "endpoint_retraction.npz"
    )
    prediction = _load_npz(
        _attempt_directory(FAILED_ATTEMPT_INDEX) / "phase_prediction.npz"
    )
    accepted_arrays = [
        _load_npz(_attempt_directory(index) / "attempt.npz")
        for index in range(ACCEPTED_STAGE2_ENDPOINTS)
    ]
    records = validated["records"]
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
        "accepted_segments_new",
        "attempts",
        "accepted_since_growth",
        "metric_transform470x470",
        "metric_augmented560x560",
        "gauge_basis560x90",
        "section_normal470",
        "start_coordinate470",
    )
    seed = {name: terminal[name] for name in progress_names}
    new_coordinates = np.stack(
        [item["accepted_coordinate470"] for item in accepted_arrays]
    )
    new_states = np.stack(
        [item["accepted_primitive_state"] for item in accepted_arrays]
    )
    new_rates = np.stack(
        [item["accepted_coordinate_rate470_per_s"] for item in accepted_arrays]
    )
    new_phase = np.asarray(
        [item["phase_geometry"]["phase_increment"] for item in records]
    )
    recurrence = [item["recurrence_geometry"] for item in records]
    seed.update(
        {
            "combined_accepted_endpoint_coordinates470": np.vstack(
                (stage2_seed["accepted_endpoint_coordinates470"], new_coordinates)
            ),
            "combined_accepted_endpoint_primitive_states": np.concatenate(
                (stage2_seed["accepted_endpoint_primitive_states"], new_states),
                axis=0,
            ),
            "combined_accepted_endpoint_coordinate_rates470_per_s": np.vstack(
                (
                    stage2_seed["accepted_endpoint_coordinate_rates470_per_s"],
                    new_rates,
                )
            ),
            "combined_accepted_phase_increments": np.concatenate(
                (stage2_seed["accepted_phase_increments"], new_phase)
            ),
            "combined_cumulative_phase_advance_radians": np.concatenate(
                (
                    stage2_seed["accepted_cumulative_phase_advance_radians"],
                    np.asarray(
                        [item["cumulative_phase_advance_radians"] for item in recurrence]
                    ),
                )
            ),
            "combined_cumulative_metric_path_lengths": np.concatenate(
                (
                    stage2_seed["accepted_cumulative_metric_path_lengths"],
                    np.asarray(
                        [item["cumulative_metric_path_length"] for item in recurrence]
                    ),
                )
            ),
            "combined_registered_section_values": np.concatenate(
                (
                    stage2_seed["accepted_registered_section_values"],
                    np.asarray(
                        [item["endpoint_registered_section_value"] for item in recurrence]
                    ),
                )
            ),
            "cumulative_phase_advance_radians": np.asarray(
                recurrence[-1]["cumulative_phase_advance_radians"]
            ),
            "cumulative_metric_path_length": np.asarray(
                recurrence[-1]["cumulative_metric_path_length"]
            ),
            "failed_target_original_coordinate470": failed[
                "target_original_coordinate470"
            ],
            "failed_recovered_original_coordinate470": failed[
                "recovered_original_coordinate470"
            ],
            "failed_retracted_primitive_state": failed["primitive_state"],
            "failed_retraction_metric_broyden560x560": failed[
                "final_metric_broyden560x560"
            ],
            "failed_decoder_reconstruction_factors": failed[
                "decoder_reconstruction_factors"
            ],
            "failed_phase_training_raw_rates470_per_s": prediction[
                "training_raw_rates470_per_s"
            ],
            "failed_phase_predicted_unit_tangent470": prediction[
                "predicted_unit_tangent470"
            ],
            "selected_metric_block_sizes": np.asarray([442, 28], dtype=np.int64),
        }
    )
    if (
        seed["combined_accepted_endpoint_coordinates470"].shape != (71, 470)
        or seed["combined_accepted_endpoint_primitive_states"].shape
        != (71, 112, 5)
        or seed["combined_accepted_endpoint_coordinate_rates470_per_s"].shape
        != (71, 470)
        or seed["combined_accepted_phase_increments"].shape != (71,)
    ):
        raise RuntimeError("hyperbolicity boundary seed changed")
    return seed


def _definitions() -> dict:
    contract = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "authorized_diagnostic": AUTHORIZED_NEXT,
        "preserved_execution": {
            "implementation_commit": "215c618f",
            "accepted_stage2_endpoints": ACCEPTED_STAGE2_ENDPOINTS,
            "combined_accepted_phase_endpoints": COMBINED_ACCEPTED_ENDPOINTS,
            "failed_attempt_index": FAILED_ATTEMPT_INDEX,
            "failed_total_segment": FAILED_TOTAL_SEGMENT,
            "failed_candidate_was_not_propagated": True,
        },
        "diagnostic_scope": {
            "reconstruct_only_the_saved_failed_retracted_state": True,
            "locate_the_first_failing_characteristic_face": True,
            "analytic_generalized_pencil": True,
            "independent_five_point_relative_steps": list(
                FINITE_DIFFERENCE_STEPS
            ),
            "accepted_to_failed_state_chord_scan_fractions": list(
                INTERPOLATION_SCAN_FRACTIONS
            ),
            "maximum_new_free_field_calls": 0,
            "maximum_new_retractions": 0,
            "maximum_wall_hours": MAXIMUM_DIAGNOSTIC_WALL_HOURS,
            "nonpropagating": True,
        },
        "genuine_hyperbolicity_loss_requires": {
            "analytic_maximum_imaginary_speed_at_least": (
                ANALYTIC_IMAGINARY_LOWER_GATE
            ),
            "each_independent_finite_difference_maximum_imaginary_speed_at_least": (
                FINITE_DIFFERENCE_IMAGINARY_LOWER_GATE
            ),
            "smallest_step_to_analytic_imaginary_relative_defect_at_most": (
                FINITE_DIFFERENCE_TO_ANALYTIC_RELATIVE_GATE
            ),
            "last_accepted_chord_endpoint_real": True,
            "failed_chord_endpoint_complex": True,
            "saved_retraction_and_physical_gates_pass": True,
        },
        "classification_branches": {
            "genuine_local_complex_pair": (
                "authorize only a definitions-only two-half-step boundary bracket manifest"
            ),
            "analytic_tangent_only_defect": (
                "authorize only an analytic-tangent repair manifest"
            ),
            "not_reproducible": "stop with no continuation authorization",
        },
        "forbidden": [
            "propagate the saved failed candidate",
            "replace the complex pair by its real parts",
            "relax the 1e-10 real-eigensystem tolerance",
            "use a complex invariant subspace as a hyperbolic flux split",
            "resume stage2 before a prospective boundary experiment",
            "authorize a phase lap, complete cycle, or reduced slow evolution",
        ],
    }
    architecture = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "mathematical_question": (
            "does the accepted conservative trajectory encounter a genuine "
            "loss of strong hyperbolicity, or did the full 0.25 ms predictor "
            "overshoot a boundary that an authentic smaller step avoids"
        ),
        "why_smaller_steps_are_not_yet_a_solution": (
            "a smaller step can localize or avoid predictor overshoot but cannot "
            "legitimately cross a genuine complex-characteristic region"
        ),
        "next_if_genuine": (
            "two authentic 0.125 ms steps with fail-closed characteristic "
            "checks and no propagation past the first complex pencil"
        ),
        "periodic_reduced_architecture_status": (
            "blocked unless the authentic trajectory remains hyperbolic through "
            "a registered return and a periodic orbit is independently certified"
        ),
    }
    return {"contract": contract, "architecture": architecture}


def _evaluate(validated: dict) -> tuple[dict, dict[str, np.ndarray], dict]:
    records = validated["records"]
    final = records[-1]["recurrence_geometry"]
    supported = bool(
        len(records) == ACCEPTED_STAGE2_ENDPOINTS
        and final["cumulative_phase_advance_radians"] == 3.4852236702137773
        and validated["prediction"]["tentative_segment_number"]
        == FAILED_TOTAL_SEGMENT
        and validated["retraction"]["passed"]
        and len(validated["scratch_hashes"]) >= 200
    )
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            CLASSIFICATION
            if supported
            else "stage2_hyperbolicity_boundary_diagnosis_not_supported"
        ),
        "passed": supported,
        "definitions_only": True,
        "new_truth_evaluations": 0,
        "new_retractions": 0,
        "accepted_stage2_endpoints": len(records),
        "combined_accepted_phase_endpoints": (
            PRIOR_STAGE1_ENDPOINTS + len(records)
        ),
        "accepted_terminal_elapsed_seconds": float(
            validated["terminal"]["elapsed_seconds"]
        ),
        "accepted_cumulative_phase_advance_radians": final[
            "cumulative_phase_advance_radians"
        ],
        "failed_attempt_index": FAILED_ATTEMPT_INDEX,
        "failed_total_segment": FAILED_TOTAL_SEGMENT,
        "failed_candidate_propagated": False,
        "failed_retraction_passed": validated["retraction"]["passed"],
        "failed_retraction_metric_condition": validated["retraction"][
            "maximum_metric_augmented_condition_number"
        ],
        "scratch_hash_count": len(validated["scratch_hashes"]),
        "authorized_next": AUTHORIZED_NEXT if supported else None,
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    return metrics, _boundary_seed(validated), _definitions()


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
    validated: dict,
) -> dict:
    helper = _helper()
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("hyperbolicity boundary manifest already exists")
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "boundary_metrics.json", metrics)
    helper._write_json(
        CANONICAL_DIRECTORY / "diagnostic_contract.json", definitions["contract"]
    )
    helper._write_json(
        CANONICAL_DIRECTORY / "mathematical_architecture.json",
        definitions["architecture"],
    )
    _save_npz(CANONICAL_DIRECTORY / "boundary_seed.npz", seed)
    helper._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "stage2_manifest_hashes": validated["manifest_lock"]["hashes"],
            "stage2_execution_identity": validated["identity"],
            "decisive_scratch_hashes": validated["scratch_hashes"],
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": metrics["passed"],
        "definitions_only": True,
        "accepted_stage2_endpoints": metrics["accepted_stage2_endpoints"],
        "failed_candidate_propagated": False,
        "hyperbolicity_diagnostic_authorized": metrics["passed"],
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
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Stage-2 hyperbolicity-boundary diagnosis manifest",
                "",
                f"Classification: `{metrics['classification']}`.",
                "",
                f"The interrupted stage-2 execution accepted `{metrics['accepted_stage2_endpoints']}` endpoints and reaches `{metrics['accepted_terminal_elapsed_seconds']:.6f}` s with cumulative phase `{metrics['accepted_cumulative_phase_advance_radians']:.9f}` rad.",
                "",
                f"Attempt `{FAILED_ATTEMPT_INDEX}` (total segment `{FAILED_TOTAL_SEGMENT}`) completed a physical, well-conditioned retraction with metric condition `{metrics['failed_retraction_metric_condition']:.6f}`, then stopped during analytic characteristic construction. It produced no field, accepted checkpoint, or propagated history.",
                "",
                "The authorized diagnostic reconstructs only that saved state, compares the analytic generalized pencil with three independent five-point pencils, and scans the accepted-to-failed chord. It performs no free-field call and cannot advance the trajectory.",
                "",
                f"Authorized next artifact: `{metrics['authorized_next']}`. No phase lap, complete cycle, or reduced slow evolution is authorized.",
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
    validated = _validate_execution_contract(require_clean=True)
    metrics, seed, definitions = _evaluate(validated)
    summary = _canonicalize(metrics, seed, definitions, validated)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
