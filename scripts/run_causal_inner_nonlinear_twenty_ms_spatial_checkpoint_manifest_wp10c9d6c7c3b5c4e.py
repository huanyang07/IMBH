#!/usr/bin/env python3
"""Freeze the staged, cost-bounded 20 ms spatial-checkpoint campaign."""

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

import run_causal_inner_nonlinear_twenty_ms_checkpoint_assessment_wp10c9d6c7c3b5c4d1 as c4d1  # noqa: E402
import run_causal_inner_nonlinear_middle_5ms_completion_wp10c9d6c7c3b5c3h2d1 as middle5  # noqa: E402
import run_causal_inner_nonlinear_fine_5ms_completion_wp10c9d6c7c3b5c3h2e1 as fine5  # noqa: E402
import run_causal_inner_nonlinear_5ms_cumulative_extraction_recovery_audit_wp10c9d6c7c3b5c3h2j1 as extraction5  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4e"
ANALYZED_BASE_COMMIT = "a4ead99e2265bee5d6af463f823c470a8cf6319e"
ANALYZED_BASE_PARENT = "032a2346090901d6498ead0f0ac21239fd172f19"
ANALYZED_BASE_TREE = "cadc8820a69b905fe776e0bedde82987cd084485"

ARTIFACT = (
    "causal_inner_nonlinear_twenty_ms_spatial_checkpoint_manifest_"
    "wp10c9d6c7c3b5c4e"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_twenty_ms_spatial_checkpoint_"
    "manifest_wp10c9d6c7c3b5c4e.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_twenty_ms_spatial_checkpoint_"
    "manifest_wp10c9d6c7c3b5c4e.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_TWENTY_MS_SPATIAL_"
    "CHECKPOINT_MANIFEST_WP10C9D6C7C3B5C4E_2026-08-10.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "spatial_checkpoint_manifest.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

LAYOUTS = (
    "N128_exterior_N128_inner_c48",
    "N128_exterior_N256_inner_c48",
    "N128_exterior_N512_inner_c48",
)
LAYOUT_CELL_COUNTS = (64, 112, 208)
COUPLING_FACE_INDICES = (48, 96, 192)
EXTRACTION_FACE_INDICES = (2, 4, 8)
EXTRACTION_RADIUS_RG = 1.9531594414758637
PROFILES = tuple(middle5.h2b1.PROFILES)
GENERIC_PROFILE = PROFILES[0]
PILOT_TARGET_MICROSECONDS = (5400, 5800, 6000)
COMPLETION_OUTPUT_MICROSECONDS = (
    6000,
    7000,
    8000,
    9000,
    10000,
    12000,
    14000,
    16000,
    18000,
    18800,
    19600,
    19800,
    20000,
)


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _validate_parent() -> tuple[dict, dict, dict, dict]:
    assessment = _read_json(c4d1.SUMMARY_PATH)
    middle = _read_json(middle5.SUMMARY_PATH)
    fine = _read_json(fine5.SUMMARY_PATH)
    extraction = _read_json(extraction5.SUMMARY_PATH)
    if (
        not assessment["passed"]
        or not assessment["twenty_ms_spatial_checkpoint_manifest_authorized"]
        or assessment["fifty_ms_propagation_authorized"]
        or assessment["fixed_q_micro_solver_authorized"]
        or assessment["reduced_slow_evolution_authorized"]
        or assessment["authorized_next"]
        != f"{WORK_PACKAGE}_twenty_ms_spatial_checkpoint_manifest"
    ):
        raise RuntimeError("c4e authorization changed")
    if (
        not middle["passed"]
        or not fine["passed"]
        or not extraction["passed"]
        or not extraction[
            "middle_fine_5ms_extraction_partition_spatial_certificate_issued"
        ]
        or extraction["analysis"]["selected_layout_face_indices"]
        != list(EXTRACTION_FACE_INDICES)
        or extraction["analysis"]["selected_radius_rg"] != EXTRACTION_RADIUS_RG
    ):
        raise RuntimeError("certified 5 ms spatial seed changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c4e analyzed identity changed")
    return assessment, middle, fine, extraction


def _manifest() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            "twenty_ms_spatial_checkpoint_manifest_frozen_middle_six_ms_"
            "cost_pilot_authorized"
        ),
        "passed": True,
        "definitions_only": True,
        "propagation_executed": False,
        "operator_changed": False,
        "production_defaults_changed": False,
        "scientific_scope": {
            "start_seconds": 5.0e-3,
            "checkpoint_seconds": 20.0e-3,
            "layouts": LAYOUTS,
            "layout_cell_counts": LAYOUT_CELL_COUNTS,
            "coupling_face_indices": COUPLING_FACE_INDICES,
            "binding_extraction_face_indices": EXTRACTION_FACE_INDICES,
            "binding_extraction_radius_rg": EXTRACTION_RADIUS_RG,
            "profiles": PROFILES,
            "generic_nonlinear_anchor": GENERIC_PROFILE,
            "raw_inner_face_is_not_a_binding_slow_export": True,
        },
        "canonical_seeds": {
            "middle_5ms_arrays": str(middle5.DECISIVE_ARRAYS.relative_to(ROOT)),
            "fine_5ms_arrays": str(fine5.DECISIVE_ARRAYS.relative_to(ROOT)),
            "coarse_10ms_arrays": str(c4d1.c4b2.DECISIVE_ARRAYS.relative_to(ROOT)),
            "coarse_20ms_arrays": str(c4d1.c4c1.DECISIVE_ARRAYS.relative_to(ROOT)),
            "five_ms_extraction_certificate": str(
                extraction5.SUMMARY_PATH.relative_to(ROOT)
            ),
            "ignored_output_checkpoints_are_not_required": True,
        },
        "execution_stages": (
            {
                "name": "middle_5_to_6ms_cost_pilot",
                "authorized": True,
                "target_microseconds": PILOT_TARGET_MICROSECONDS,
                "work": (
                    "one_nonlinear_base",
                    "one_full_generic_nonlinear_anchor",
                    "one_five_profile_block_tangent",
                    "sampled_step_doubling",
                    "binding_extraction_partition_evaluation",
                ),
            },
            {
                "name": "middle_6_to_20ms_completion",
                "authorized": False,
                "requires_fresh_manifest_after_pilot": True,
                "output_target_microseconds": COMPLETION_OUTPUT_MICROSECONDS,
            },
            {
                "name": "coarse_middle_checkpoint_analysis",
                "authorized": False,
                "requires_middle_completion": True,
            },
            {
                "name": "fine_5_to_20ms_conditional_confirmation",
                "authorized": False,
                "requires_fresh_manifest_after_middle_analysis": True,
            },
        ),
        "pilot_contract": {
            "middle_layout_owns_accepted_schedule": True,
            "coarse_schedule_is_not_acceptance_authority": True,
            "generic_anchor_replays_base_schedule": True,
            "tangent_prediction_is_anchor_newton_initial_guess": True,
            "five_profiles_share_one_matrix_factorization": True,
            "complete_residual_jvp_only_at_declared_audit_steps": True,
            "checkpoint_after_every_completed_stage": True,
            "same_target_restart_replay_bitwise": True,
            "minimum_measured_steps_for_projection": 3,
            "projection_safety_factor": 1.5,
        },
        "method_gates": {
            "maximum_scaled_nonlinear_residual": 1.0e-10,
            "maximum_discrete_ledger_defect": 1.0e-12,
            "maximum_mapped_endpoint_path_closure_defect": 1.0e-9,
            "maximum_extraction_identity_defect": 1.0e-12,
            "minimum_reconstruction_factor": 1.0,
            "maximum_incoming_excision_characteristics": 0,
            "maximum_local_error_estimate": 2.5e-4,
            "maximum_tangent_linear_solve_relative_defect": 1.0e-10,
            "maximum_tangent_matrix_jvp_relative_defect": 1.0e-8,
            "maximum_generic_tangent_discrepancy_fraction_of_response": 0.01,
        },
        "spatial_certificate_gates": {
            "minimum_state_rms_order": 0.75,
            "minimum_state_max_order": 0.75,
            "minimum_significant_state_field_order": 0.75,
            "minimum_extraction_instantaneous_rms_order": 0.75,
            "minimum_extraction_cumulative_rms_order": 0.75,
            "minimum_significant_extraction_component_order": 0.75,
            "minimum_refinement_error_cosine": 0.90,
            "maximum_fine_normalized_difference": 0.05,
            "maximum_temporal_uncertainty_fraction_of_spatial_difference": 0.10,
            "maximum_surrogate_uncertainty_fraction_of_spatial_difference": 0.10,
            "common_parent_restriction_required": True,
        },
        "fine_trigger": {
            "fine_information_required_for_a_measured_order": True,
            "full_fine_nonlinear_base_is_default_if_middle_passes": True,
            "five_profile_fine_block_tangent_is_default_if_middle_passes": True,
            "full_fine_generic_anchor_is_conditional": True,
            "full_fine_anchor_triggers": (
                "middle_anchor_surrogate_uncertainty_exceeds_0p1_of_predicted_medium_fine_difference",
                "predicted_spatial_result_is_within_0p2_of_a_gate",
                "observable_nonlinear_remainder_changes_between_5_and_20ms",
                "fine_tangent_audit_fails",
            ),
            "defect_estimator_is_screening_only_until_independently_calibrated": True,
        },
        "resource_policy": {
            "twenty_four_hours_is_soft_not_binding": True,
            "projected_middle_up_to_hours": 30.0,
            "projected_fine_base_and_tangent_up_to_hours": 36.0,
            "projected_conditional_fine_anchor_additional_hours": 12.0,
            "working_total_hours": 54.0,
            "contingency_total_hours": 78.0,
            "unattended_stage_checkpoint_hours": 4.0,
            "stop_for_optimization_review_if_single_stage_projects_above_hours": 48.0,
        },
        "decision_branches": {
            "pilot_passes_and_projection_acceptable": (
                "freeze_middle_6_to_20ms_completion_manifest"
            ),
            "pilot_scientific_gate_fails": "localize_middle_time_or_spatial_failure",
            "pilot_cost_only_exceeds_projection": "optimize_without_scientific_rejection",
            "middle_checkpoint_fails": "stop_before_fine",
            "middle_checkpoint_passes": "freeze_conditional_fine_manifest",
            "fine_certificate_passes": "fifty_ms_manifest_may_be_considered",
        },
        "hard_stops": (
            "do_not_run_fine_before_middle_checkpoint_analysis",
            "do_not_run_fifty_ms_before_the_spatial_certificate",
            "do_not_use_raw_inner_face_flux_as_the_slow_export",
            "do_not_drop_the_fine_level_without_a_certified_upper_bound",
            "do_not_start_fixed_Q_or_reduced_slow_evolution",
            "do_not_change_operator_profile_or_production_defaults",
        ),
        "authorized_next": (
            "WP10c9d6c7c3b5c4e1_middle_5_to_6ms_spatial_cost_pilot"
        ),
    }


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
    assessment, middle, fine, extraction = _validate_parent()
    manifest = _manifest()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": manifest["classification"],
        "passed": True,
        "definitions_only": True,
        "parent_classification_preserved": assessment["classification"],
        "five_ms_middle_seed_certified": bool(middle["passed"]),
        "five_ms_fine_seed_certified": bool(fine["passed"]),
        "five_ms_extraction_partition_certificate_preserved": bool(
            extraction[
                "middle_fine_5ms_extraction_partition_spatial_certificate_issued"
            ]
        ),
        "middle_six_ms_cost_pilot_authorized": True,
        "middle_twenty_ms_completion_authorized": False,
        "fine_twenty_ms_propagation_authorized": False,
        "twenty_ms_spatial_checkpoint_certified": False,
        "fifty_ms_propagation_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": manifest["authorized_next"],
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "layouts": LAYOUTS,
            "pilot_target_microseconds": PILOT_TARGET_MICROSECONDS,
            "completion_output_microseconds": COMPLETION_OUTPUT_MICROSECONDS,
            "extraction_face_indices": EXTRACTION_FACE_INDICES,
            "extraction_radius_rg": EXTRACTION_RADIUS_RG,
        },
    )
    _write_json(MANIFEST_PATH, manifest)
    _write_json(SUMMARY_PATH, summary)
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
                "checkpoint_assessment": _sha256(c4d1.SUMMARY_PATH),
                "middle_5ms_arrays": _sha256(middle5.DECISIVE_ARRAYS),
                "fine_5ms_arrays": _sha256(fine5.DECISIVE_ARRAYS),
                "five_ms_extraction_certificate": _sha256(extraction5.SUMMARY_PATH),
                "coarse_10ms_arrays": _sha256(c4d1.c4b2.DECISIVE_ARRAYS),
                "coarse_20ms_arrays": _sha256(c4d1.c4c1.DECISIVE_ARRAYS),
            },
            "implementation_source_hashes": {
                path: _sha256(ROOT / path)
                for path in (THIS_RUNNER, THIS_TEST)
                if (ROOT / path).exists()
            },
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "command": f"PYTHONPATH=src:scripts python {THIS_RUNNER}",
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Nonlinear 20 ms spatial-checkpoint manifest WP10c9d6c7c3b5c4e",
                "",
                "## Classification",
                "",
                f"`{manifest['classification']}`",
                "",
                "This definitions-only package freezes a middle-first, fail-fast spatial checkpoint at 20 ms. It executes no state.",
                "",
                "The first authorized work is a middle-layout continuation from 5 to 6 ms using one nonlinear base, one generic nonlinear anchor, and one five-profile block tangent. Its measured cost and scientific gates determine whether a fresh 6-to-20 ms middle completion manifest may be frozen.",
                "",
                "The binding slow export is the certified exterior extraction partition at coarse/middle/fine faces 2/4/8 (`R=1.95315944 r_g`). The raw pointwise horizon flux remains rejected.",
                "",
                "Fine propagation remains conditional on the middle checkpoint. Twenty-four hours is a soft scheduling target; scientific gates are unchanged. The working total is 54 hours with a 78-hour contingency, dominated by nonlinear base trajectories.",
                "",
                "Fifty-millisecond propagation, fixed-Q experiments, and reduced slow evolution remain blocked.",
                "",
            )
        ),
        encoding="utf-8",
    )
    names = (
        "config.json",
        "provenance.json",
        "spatial_checkpoint_manifest.json",
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
