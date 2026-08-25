#!/usr/bin/env python3
"""Freeze a prospective conservative tangent-phase-atlas holdout."""

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

from imri_qpe.layer3_minidisk_1d.tangent_phase_atlas import (  # noqa: E402
    fit_tangent_phase_chart,
    normalized_metric_tangents,
    rolling_tangent_phase_audit,
)
import run_causal_inner_adaptive_metric_chart_cycle_readiness_discrimination_execution_wp10c9d6c7c3b5c4f25fit as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = parent.OPEN_AUTHORIZED_NEXT
CLASSIFICATION = (
    "conservative_overlapping_tangent_phase_atlas_selected_"
    "prospective_holdout_required"
)
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fiv_"
    "conservative_tangent_phase_atlas_holdout_execution"
)
WINDOWS = (12, 16, 20, 24)
SELECTED_WINDOW = 12
PREDICTOR_INCREMENT_COUNT = 4
ARTIFACT = (
    "causal_inner_adaptive_metric_chart_cycle_readiness_phase_atlas_manifest_"
    "wp10c9d6c7c3b5c4f25fiu"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_CONSERVATIVE_TANGENT_PHASE_"
    "ATLAS_MANIFEST_WP10C9D6C7C3B5C4F25FIU_2026-08-24.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_adaptive_metric_chart_cycle_readiness_"
    "phase_atlas_manifest_wp10c9d6c7c3b5c4f25fiu.py"
)
THIS_TEST = (
    "tests/test_causal_inner_adaptive_metric_chart_cycle_readiness_"
    "phase_atlas_manifest_wp10c9d6c7c3b5c4f25fiu.py"
)
PHASE_SOURCE = "src/imri_qpe/layer3_minidisk_1d/tangent_phase_atlas.py"
PHASE_TEST = "tests/test_tangent_phase_atlas.py"


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
        PHASE_SOURCE,
        PHASE_TEST,
        parent.THIS_RUNNER,
        parent.THIS_TEST,
    )
    return {name: helper._sha(ROOT / name) for name in paths}


def _validate_parent(*, require_clean: bool) -> dict:
    helper = _helper()
    hashes = helper._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = helper._read(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = helper._read(
        parent.CANONICAL_DIRECTORY / "forecast_discrimination_metrics.json"
    )
    provenance = helper._read(parent.CANONICAL_DIRECTORY / "provenance.json")
    values = metrics["gate_values"]
    forecast = values["updated_forecast_bundle"]
    if (
        summary["classification"] != parent.OPEN_CLASSIFICATION
        or not summary["passed"]
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["complete_cycle_execution_authorized"]
        or summary["reduced_slow_evolution_authorized"]
        or values["accepted_segments"] != 48
        or values["attempted_segments"] != 48
        or values["terminal_elapsed_seconds"] != 0.16400000000000012
        or values["section_turning_point_observed"]
        or values["section_negative_observed"]
        or not values["secant_acceleration_reversal_observed"]
        or forecast["raw_velocity_forecast_range_seconds"] is not None
        or forecast["orientation_forecast_range_seconds"] is not None
        or len(forecast["raw_velocity_windows_without_forward_zero"]) != 7
        or len(forecast["orientation_windows_without_forward_zero"]) != 7
    ):
        raise RuntimeError("orientation-discrimination classification changed")
    for relative, expected in provenance["source_hashes"].items():
        if helper._sha(ROOT / relative) != expected:
            raise RuntimeError(f"orientation-discrimination source changed: {relative}")
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("tangent-phase manifest requires a clean tracked tree")
    return {
        "hashes": hashes,
        "summary": summary,
        "metrics": metrics,
        "provenance": provenance,
    }


def _trajectory() -> dict[str, np.ndarray]:
    arrays = _load_npz(
        parent.CANONICAL_DIRECTORY / "forecast_discrimination_arrays.npz"
    )
    seed = parent.manifest._seed()
    rates = np.vstack(
        (
            seed["current_coordinate_rate470_per_s"],
            arrays["accepted_endpoint_coordinate_rates470_per_s"],
        )
    )
    times = np.concatenate(
        (
            np.asarray([float(seed["elapsed_seconds"])]),
            arrays["accepted_endpoint_elapsed_seconds"],
        )
    )
    if (
        rates.shape != (49, 470)
        or times.shape != (49,)
        or not np.all(np.diff(times) > 0.0)
        or float(times[0]) != parent.manifest.INITIAL_ELAPSED_SECONDS
        or float(times[-1]) != 0.16400000000000012
    ):
        raise RuntimeError("tangent-phase trajectory changed")
    return {
        "rates470_per_s": rates,
        "times_seconds": times,
        "initial_metric_transform470x470": seed["metric_transform470x470"],
        "terminal_metric_transform470x470": arrays["metric_transform470x470"],
        "parent_arrays": arrays,
    }


def _chart_arrays(prefix: str, chart) -> dict[str, np.ndarray]:
    return {
        f"{prefix}mean_tangent470": chart.mean_tangent,
        f"{prefix}plane_basis470x2": chart.plane_basis,
        f"{prefix}circle_center2": chart.circle_center,
        f"{prefix}circle_radius": np.asarray(chart.circle_radius),
        f"{prefix}orientation_sign": np.asarray(chart.orientation_sign),
        f"{prefix}oriented_angle_origin": np.asarray(
            chart.oriented_angle_origin
        ),
        f"{prefix}training_phases": chart.training_phases,
        f"{prefix}predicted_phase_increment": np.asarray(
            chart.predicted_phase_increment
        ),
        f"{prefix}predicted_unit_tangent470": chart.predicted_unit_tangent(),
    }


def _evaluate(parent_lock: dict) -> tuple[dict, dict[str, np.ndarray], dict]:
    trajectory = _trajectory()
    rates = trajectory["rates470_per_s"]
    initial_transform = trajectory["initial_metric_transform470x470"]
    unit = normalized_metric_tangents(rates, initial_transform)
    audits = {}
    output_arrays = {
        "trajectory_times_seconds": trajectory["times_seconds"],
        "trajectory_raw_rates470_per_s": rates,
        "trajectory_initial_metric_unit_tangents470": unit,
    }
    for window in WINDOWS:
        metrics, arrays = rolling_tangent_phase_audit(
            unit,
            window_size=window,
            predictor_increment_count=PREDICTOR_INCREMENT_COUNT,
        )
        audits[str(window)] = metrics
        for name, value in arrays.items():
            output_arrays[f"window_{window}__{name}"] = value
    selected = audits[str(SELECTED_WINDOW)]
    terminal_transform = trajectory["terminal_metric_transform470x470"]
    terminal_training_rates = rates[-SELECTED_WINDOW:]
    terminal_unit = normalized_metric_tangents(
        terminal_training_rates, terminal_transform
    )
    terminal_chart = fit_tangent_phase_chart(
        terminal_unit, predictor_increment_count=PREDICTOR_INCREMENT_COUNT
    )
    output_arrays.update(
        {
            "terminal_metric_transform470x470": terminal_transform,
            "terminal_training_raw_rates470_per_s": terminal_training_rates,
            "terminal_training_unit_tangents470": terminal_unit,
            **_chart_arrays("terminal_chart__", terminal_chart),
        }
    )
    parent_values = parent_lock["metrics"]["gate_values"]
    forecast = parent_values["updated_forecast_bundle"]
    fixed_section_rejected = bool(
        forecast["raw_velocity_forecast_range_seconds"] is None
        and forecast["orientation_forecast_range_seconds"] is None
        and len(forecast["raw_velocity_windows_without_forward_zero"]) == 7
        and len(forecast["orientation_windows_without_forward_zero"]) == 7
    )
    retrospective_support = bool(
        fixed_section_rejected
        and selected["all_phase_increments_positive"]
        and selected["minimum_training_two_plane_energy_fraction"] >= 0.999
        and selected["maximum_relative_radial_defect"] <= 0.002
        and selected["maximum_out_of_plane_defect"] <= 0.005
        and selected["maximum_direction_prediction_defect_radians"] <= 0.005
        and terminal_chart.two_plane_energy_fraction >= 0.999
    )
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            CLASSIFICATION
            if retrospective_support
            else "tangent_phase_atlas_architecture_not_supported"
        ),
        "passed": retrospective_support,
        "definitions_only": True,
        "new_truth_evaluations": 0,
        "retrospective_cross_validation_is_binding_execution_evidence": False,
        "fixed_section_forecast_rejected": fixed_section_rejected,
        "window_audits": audits,
        "selected_window": SELECTED_WINDOW,
        "selection_is_retrospective": True,
        "terminal_chart": {
            "training_samples": SELECTED_WINDOW,
            "two_plane_energy_fraction": terminal_chart.two_plane_energy_fraction,
            "training_relative_radial_rms": (
                terminal_chart.training_relative_radial_rms
            ),
            "predicted_phase_increment": (
                terminal_chart.predicted_phase_increment
            ),
            "circle_solve_condition_number": (
                terminal_chart.circle_solve_condition_number
            ),
        },
        "complete_cycle_observed": False,
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT if retrospective_support else None,
        "input_lock": {
            "parent_hashes": parent_lock["hashes"],
            "parent_classification": parent_lock["summary"]["classification"],
        },
    }
    contract = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "authorized_execution": AUTHORIZED_NEXT,
        "scope": {
            "initial_elapsed_seconds": 0.16400000000000012,
            "accepted_segments": 16,
            "maximum_attempted_segments": 18,
            "initial_segment_seconds": 2.5e-4,
            "minimum_segment_seconds": 1.25e-4,
            "maximum_segment_seconds": 2.5e-4,
            "maximum_exact_free_field_calls": 20,
            "maximum_retractions": 20,
            "maximum_execution_wall_hours": 2.5,
            "blind_midpoint_frequency": 4,
        },
        "phase_observer": {
            "physical_vector_field": "autonomous original reaction-free field",
            "metric": "frozen terminal conservative transform W_164ms",
            "unit_tangent": "tau=W f_free/||W f_free||_2",
            "trailing_window_samples": SELECTED_WINDOW,
            "affine_plane_rank": 2,
            "circle_fit": "linear least squares in the trailing affine plane",
            "phase_orientation": "strictly increasing training-time orientation",
            "phase_increment_predictor": (
                f"median of the last {PREDICTOR_INCREMENT_COUNT} accepted increments"
            ),
            "refit_after_each_accepted_exact_endpoint": True,
            "prediction_is_frozen_before_holdout_tangent_is_seen": True,
            "rejected_candidate_never_enters_phase_history": True,
        },
        "binding_phase_gates": {
            "minimum_phase_increment_strictly_greater_than": 0.0,
            "maximum_phase_increment": 0.08,
            "minimum_training_two_plane_energy_fraction": 0.999,
            "maximum_training_relative_radial_rms": 0.001,
            "maximum_holdout_relative_radial_defect": 0.002,
            "maximum_holdout_out_of_plane_defect": 0.005,
            "maximum_direction_prediction_defect_radians": 0.005,
            "all_original_physical_and_retraction_gates_unchanged": True,
            "checkpoint_roundtrip_bitwise": True,
            "suffix_history_replay_bitwise": True,
        },
        "classification_branches": {
            "phase_observer_prospectively_validated": (
                "all 16 exact holdouts and every numerical/physical/phase gate pass"
            ),
            "phase_observer_geometry_failed": (
                "physical continuation passes but a phase geometry gate fails"
            ),
            "phase_observer_physical_or_numerical_failed": (
                "any original continuation, physics, checkpoint, or replay gate fails"
            ),
        },
        "authorization_boundary": (
            "a pass authorizes only a phase-lap/recurrence acquisition manifest; "
            "it does not authorize a complete-cycle claim or reduced slow evolution"
        ),
    }
    architecture = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "truth_dynamics": "dq/dt=f_free(q) on the certified conservative atlas",
        "phase_observer": (
            "overlapping oriented tangent-circle charts in a frozen conservative metric"
        ),
        "phase_lap_is_not_a_cycle": True,
        "cycle_certificate_requires": {
            "unwrapped_tangent_phase_advance_at_least": "2*pi",
            "state_return_distance_fraction_of_path_length_at_most": 0.10,
            "same_orientation": True,
            "transverse_registered_section_crossing": True,
            "all_physical_ledgers_and_restart_gates": True,
        },
        "offline_periodic_orbit_if_cycle_is_certified": {
            "method": "phase-conditioned multiple shooting/collocation",
            "unknowns": "orbit nodes, segment durations, and period",
            "equations": "exact free-field flow matching plus periodic closure",
            "phase_condition": "one registered tangent-phase chart",
            "continuation": "continue the periodic solution over slow Q anchors",
        },
        "reduced_slow_solver_if_orbits_are_certified": {
            "averaged_drift": (
                "Fbar(Q)=T(Q)^-1 integral_0^T F_slow(q_star(t;Q),Q) dt"
            ),
            "sensitivities": "bordered periodic adjoint",
            "online_state": "slow macro Q plus discrete mode/event state",
            "online_forbidden": (
                "exact truth calls, fixed-Q reactions, nonlinear roots, and micro-BDF steps"
            ),
        },
        "no_recurrence_branch": (
            "classify the autonomous free trajectory as open/nonperiodic and prohibit "
            "cycle averaging; test equilibrium or nonperiodic attractor closure instead"
        ),
    }
    return metrics, output_arrays, {"contract": contract, "architecture": architecture}


def _continuation_seed() -> dict[str, np.ndarray]:
    arrays = _trajectory()["parent_arrays"]
    names = (
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
    return {name: np.asarray(arrays[name]) for name in names}


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
    arrays: dict[str, np.ndarray],
    definitions: dict,
    parent_lock: dict,
) -> dict:
    helper = _helper()
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("tangent-phase manifest result already exists")
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(
        CANONICAL_DIRECTORY / "retrospective_phase_diagnosis.json", metrics
    )
    helper._write_json(
        CANONICAL_DIRECTORY / "phase_atlas_contract.json",
        definitions["contract"],
    )
    helper._write_json(
        CANONICAL_DIRECTORY / "mathematical_architecture.json",
        definitions["architecture"],
    )
    _save_npz(CANONICAL_DIRECTORY / "phase_atlas_diagnostic_arrays.npz", arrays)
    _save_npz(CANONICAL_DIRECTORY / "continuation_seed.npz", _continuation_seed())
    helper._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_hashes": parent_lock["hashes"],
            "parent_classification": parent_lock["summary"]["classification"],
            "parent_acquisition_commit": parent_lock["provenance"][
                "acquisition_commit"
            ],
            "parent_classifier_commit": parent_lock["provenance"][
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
        "phase_atlas_holdout_execution_authorized": metrics["passed"],
        "phase_atlas_holdout_execution_executed": False,
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
    selected = metrics["window_audits"][str(SELECTED_WINDOW)]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Conservative tangent-phase-atlas manifest",
                "",
                f"Classification: `{metrics['classification']}`.",
                "",
                "All seven fixed-section raw and orientation forecast windows now have positive slopes, so the earlier zero forecasts are rejected. The exact autonomous free-field trajectory and conservative metric atlas remain valid.",
                "",
                f"A retrospective trailing-window audit selected `{SELECTED_WINDOW}` unit tangents in one frozen conservative metric. All `{selected['prediction_count']}` one-step phase increments are positive; maximum radial, out-of-plane, and direction-prediction defects are `{selected['maximum_relative_radial_defect']:.6e}`, `{selected['maximum_out_of_plane_defect']:.6e}`, and `{selected['maximum_direction_prediction_defect_radians']:.6e}` rad. This evidence is supportive but is not counted as prospective execution evidence.",
                "",
                "The next package is a 16-endpoint prospective holdout. Every tangent-circle chart and next-step prediction must be frozen before seeing its exact holdout tangent. Original physical, retraction, checkpoint, and replay gates remain binding.",
                "",
                "A 2*pi tangent-phase advance is necessary but not sufficient for a cycle. State recurrence, same orientation, transverse registered-section return, physical ledgers, and restart replay must also pass before periodic-orbit multiple shooting or slow averaging is authorized.",
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
    parser.add_argument("--run", action="store_true")
    arguments = parser.parse_args()
    if not arguments.run:
        parser.error("--run is required")
    parent_lock = _validate_parent(require_clean=True)
    metrics, arrays, definitions = _evaluate(parent_lock)
    summary = _canonicalize(metrics, arrays, definitions, parent_lock)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
