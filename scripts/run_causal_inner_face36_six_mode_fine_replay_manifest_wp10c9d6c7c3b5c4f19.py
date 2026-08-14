#!/usr/bin/env python3
"""Freeze the fine-only six-mode dynamic-coordinate replay contract.

Definitions only.  This package reuses the recovered middle history and
authorizes one analysis-only fine tangent replay.  It advances no trajectory.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_face36_six_mode_dynamic_coordinate_replay_wp10c9d6c7c3b5c4f17 as c4f17  # noqa: E402
import run_causal_inner_face36_six_mode_numerical_audit_recovery_wp10c9d6c7c3b5c4f18 as c4f18  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f19"
ARTIFACT = (
    "causal_inner_face36_six_mode_fine_replay_manifest_"
    "wp10c9d6c7c3b5c4f19"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_face36_six_mode_fine_replay_manifest_"
    "wp10c9d6c7c3b5c4f19.py"
)
THIS_TEST = (
    "tests/test_causal_inner_face36_six_mode_fine_replay_manifest_"
    "wp10c9d6c7c3b5c4f19.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_FACE36_SIX_MODE_FINE_REPLAY_MANIFEST_"
    "WP10C9D6C7C3B5C4F19_2026-08-13.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "fine_replay_manifest.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

MODE_DIMENSION = 6
LEADING_DIMENSION = 2
AUDIT_TIME_IDS_MICROSECONDS = (5000, 5400, 10000, 16000, 20000)
SELECTED_RELATIVE_STEPS = (5.0e-5, 1.0e-4)


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _authorization() -> tuple[dict, dict]:
    middle = _read(c4f17.SUMMARY_PATH)
    recovery = _read(c4f18.SUMMARY_PATH)
    expected = (
        "WP10c9d6c7c3b5c4f19_definitions_only_fine_six_mode_"
        "dynamic_coordinate_replay_manifest"
    )
    if (
        not middle["middle_completed"]
        or middle["fine_executed"]
        or not recovery["passed"]
        or not recovery["saved_middle_history_reclassified"]
        or not recovery["dual_recovery_passed"]
        or not recovery["face36_directional_JVP_plateau_passed"]
        or recovery["selected_adjacent_step_pair"]
        != list(SELECTED_RELATIVE_STEPS)
        or recovery["fine_executed"]
        or recovery["fixed_Q_micro_solver_authorized"]
        or recovery["authorized_next"] != expected
    ):
        raise RuntimeError("c4f19 authorization changed")
    return middle, recovery


def _manifest(middle: dict, recovery: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            "fine_six_mode_dynamic_coordinate_replay_manifest_frozen_"
            "analysis_only_fine_replay_authorized"
        ),
        "definitions_only": True,
        "trajectory_executed": False,
        "physical_operator_changed": False,
        "parent_decision": {
            "saved_middle_steps": middle["middle"]["steps"],
            "saved_middle_wall_seconds": middle["middle"]["wall_seconds"],
            "saved_middle_maximum_Q3_leakage": middle["middle"][
                "maximum_Q3_leakage"
            ],
            "stable_dual_recovered": recovery["dual_recovery_passed"],
            "face36_directional_JVP_plateau_recovered": recovery[
                "face36_directional_JVP_plateau_passed"
            ],
            "middle_history_scientifically_reclassified": recovery[
                "saved_middle_history_reclassified"
            ],
        },
        "authorized_fine_replay": {
            "work_package": (
                "WP10c9d6c7c3b5c4f20_analysis_only_fine_six_mode_"
                "dynamic_coordinate_replay"
            ),
            "layout": "fine",
            "uses_committed_fine_5_to_20ms_base_history": True,
            "uses_saved_c4f17_middle_state_direction_history": True,
            "reruns_middle_propagation": False,
            "new_nonlinear_trajectory": False,
            "new_tangent_trajectory": True,
            "directions": MODE_DIMENSION,
            "one_factorization_six_RHS_per_step": True,
            "propagate_complete_BDF_history_directions": True,
            "save_state_direction_history_at_all_committed_outputs": True,
            "save_guard_mapped_and_height_history_complement": True,
            "audit_complete_residual_JVP_only_at_time_ids_microseconds": [
                5400,
                10000,
                16000,
                20000,
            ],
            "run_initial_dual_and_face36_audits_before_full_propagation": True,
            "durable_per_step_checkpoint": True,
            "resume_requires_exact_source_and_input_hashes": True,
        },
        "stable_dual_contract": {
            "methods": ["reduced_QR", "thin_SVD"],
            "normal_equations_forbidden": True,
            "maximum_biorthogonality_defect": 1.0e-10,
            "maximum_normalized_slow_lift_annihilation_defect": 1.0e-10,
            "maximum_initial_consensus_coefficient_defect": 1.0e-10,
            "maximum_relative_QR_SVD_dual_difference": 1.0e-8,
            "use_QR_dual_for_reported_amplitude_history": True,
            "recompute_saved_middle_amplitudes_with_recovered_QR_dual": True,
        },
        "face36_directional_JVP_contract": {
            "time_ids_microseconds": list(AUDIT_TIME_IDS_MICROSECONDS),
            "directions": MODE_DIMENSION,
            "selected_relative_steps": list(SELECTED_RELATIVE_STEPS),
            "reference": "five_point_central_directional_difference",
            "central_difference_recorded_as_nonbinding_diagnostic": True,
            "maximum_relative_defect_at_each_selected_step": 1.0e-8,
            "same_pair_required_for_all_times_and_directions": True,
            "tolerance_relaxation_forbidden": True,
        },
        "single_layout_method_gates": {
            "maximum_step_matrix_JVP_relative_defect": 1.0e-8,
            "maximum_block_linear_solve_relative_defect": 1.0e-10,
            "maximum_component_closure_defect": 1.0e-12,
            "maximum_Q3_leakage": 0.10,
            "maximum_initial_state_lift_Q3_defect": 1.0e-10,
            "maximum_initial_scaled_orthogonality_defect": 1.0e-10,
            "incoming_excision_characteristics": 0,
        },
        "cross_resolution_contract": {
            "restrict_middle_and_fine_state_directions_to_common_parent": True,
            "require_exact_common_output_time_ids": True,
            "leading_block_dimensions": [0, LEADING_DIMENSION],
            "weak_enrichment_block_dimensions": [LEADING_DIMENSION, MODE_DIMENSION],
            "align_weak_block_by_orthogonal_Procrustes": True,
            "individual_weak_mode_matching_forbidden": True,
            "minimum_leading_block_projector_cosine": 0.95,
            "minimum_full_subspace_projector_cosine": 0.90,
            "minimum_stable_dual_amplitude_history_cosine": 0.95,
            "maximum_stable_dual_amplitude_history_relative_difference": 0.10,
            "minimum_face36_mode_history_cosine": 0.95,
            "maximum_face36_mode_history_relative_difference": 0.10,
            "maximum_six_mode_output_weighted_RMS_error": 0.10,
            "maximum_six_mode_significant_direction_error": 0.25,
            "fine_only_complement_and_face36_observability_must_be_reported": True,
            "guard_complement_retained_without_smallness_assumption": True,
        },
        "fail_fast_decision": {
            "fine_initial_dual_or_face36_audit_fails": (
                "stop_before_fine_tangent_propagation_and_localize_numerical_audit"
            ),
            "fine_method_gate_fails": (
                "reject_fine_replay_without_interpreting_cross_grid_coordinates"
            ),
            "leading_block_cross_grid_gate_fails": (
                "return_to_memory_basis_localization"
            ),
            "leading_block_passes_but_weak_block_fails": (
                "authorize_definitions_only_leading_two_plus_HMM_manifest"
            ),
            "all_single_layout_and_cross_grid_gates_pass": (
                "authorize_definitions_only_one_Q_constrained_nonlinear_"
                "pilot_manifest"
            ),
        },
        "cost_contract": {
            "measured_middle_replay_wall_hours": middle["middle"][
                "wall_seconds"
            ]
            / 3600.0,
            "measured_middle_recovery_audit_wall_hours": recovery[
                "wall_seconds"
            ]
            / 3600.0,
            "expected_fine_replay_and_selected_audits_wall_hours": [3.5, 5.5],
            "selected_two_step_JVP_audit_replaces_six_step_sweep": True,
            "full_middle_replay_forbidden": True,
            "no_repeated_29_direction_propagation": True,
            "stop_after_initial_audits_if_they_fail": True,
            "wall_time_is_a_scheduling_metric_not_a_scientific_gate": True,
        },
        "hard_stops": [
            "do_not_rerun_the_middle_tangent_history",
            "do_not_relax_any_coordinate_or_derivative_tolerance",
            "do_not_name_weak_enrichment_vectors_as_individual_physical_modes",
            "do_not_apply_a_fixed_Q_reaction",
            "do_not_start_a_nonlinear_microburst",
            "do_not_discard_the_guard_complement",
            "do_not_use_raw_face48_as_slow_exchange",
            "do_not_start_50ms_or_reduced_slow_evolution",
        ],
        "fine_dynamic_coordinate_replay_authorized": True,
        "fixed_Q_micro_solver_authorized": False,
        "nonlinear_retained_mode_pilot_authorized": False,
        "fifty_ms_propagation_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": (
            "WP10c9d6c7c3b5c4f20_analysis_only_fine_six_mode_"
            "dynamic_coordinate_replay"
        ),
    }


def _catalog(summary: dict) -> None:
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
                    "sha256": _sha(path),
                    "scientific_status": "PROSPECTIVE",
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case", "path", "bytes", "sha256", "scientific_status"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": True,
    }
    catalog.update(
        {
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write(CANONICAL_SUMMARY, catalog)


def main() -> None:
    middle, recovery = _authorization()
    manifest = _manifest(middle, recovery)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": manifest["classification"],
        "passed": True,
        "definitions_only": True,
        "trajectory_executed": False,
        "saved_middle_history_reused": True,
        "middle_replay_forbidden": True,
        "fine_dynamic_coordinate_replay_authorized": True,
        "fixed_Q_micro_solver_authorized": False,
        "nonlinear_retained_mode_pilot_authorized": False,
        "fifty_ms_propagation_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "physical_failure_detected": False,
        "guard_complement_retained": True,
        "raw_face48_export_rejection_preserved": True,
        "authorized_next": manifest["authorized_next"],
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "mode_dimension": MODE_DIMENSION,
            "leading_dimension": LEADING_DIMENSION,
            "audit_time_ids_microseconds": list(AUDIT_TIME_IDS_MICROSECONDS),
            "selected_relative_steps": list(SELECTED_RELATIVE_STEPS),
            "shared_exchange_parent_face": 36,
            "raw_face48_exchange_forbidden": True,
        },
    )
    _write(MANIFEST_PATH, manifest)
    _write(SUMMARY_PATH, summary)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# Face-36 six-mode fine replay manifest\n\n"
        f"Classification: `{summary['classification']}`.\n\n"
        "The saved middle six-direction history is accepted after the c4f18 "
        "stable-dual and directional-JVP recovery. This definitions-only "
        "package authorizes one fine-layout analysis-only replay; it forbids "
        "repeating the middle propagation.\n\n"
        "The fine replay must run its stable QR/SVD dual and face-36 five-point "
        "audits before full propagation, then apply the unchanged single-layout "
        "and cross-grid projector, amplitude-history, face-36 history, and "
        "output-closure gates. Only the selected relative-step pair "
        "`(5e-5, 1e-4)` is needed, reducing derivative-audit cost without "
        "relaxing the `1e-8` gate.\n\n"
        "No nonlinear trajectory, fixed-Q reaction, 50 ms propagation, or "
        "reduced slow evolution is authorized. The guard complement and raw "
        "face-48 rejection remain binding.\n",
        encoding="utf-8",
    )
    _write(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "middle_summary_sha256": _sha(c4f17.SUMMARY_PATH),
            "middle_arrays_sha256": _sha(c4f17.DECISIVE_ARRAYS),
            "recovery_summary_sha256": _sha(c4f18.SUMMARY_PATH),
            "recovery_arrays_sha256": _sha(c4f18.DECISIVE_ARRAYS),
            "source_hashes": {
                THIS_RUNNER: _sha(ROOT / THIS_RUNNER),
                THIS_TEST: (
                    _sha(ROOT / THIS_TEST) if (ROOT / THIS_TEST).exists() else None
                ),
            },
        },
    )
    files = (CONFIG_PATH, MANIFEST_PATH, SUMMARY_PATH, PROVENANCE_PATH)
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )
    _catalog(summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
