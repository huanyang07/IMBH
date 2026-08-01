#!/usr/bin/env python3
"""Freeze a cost-bounded nonlinear temporal-refinement campaign."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_causal_inner_nonlinear_spatial_export_manifest_wp10c9d6c7c3b2a as c3b2a  # noqa: E402
import run_causal_inner_nonlinear_spatial_export_pilot_wp10c9d6c7c3b2b as c3b2b  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b3a"
ANALYZED_BASE_COMMIT = "bf1ed9e6a11a7e687f8544ae6daae6c3e1cd9203"
ANALYZED_BASE_PARENT = "67b3e70e3eadb43d12229f89daecec5b04d0e7fb"
ANALYZED_BASE_TREE = "b41dd8d4baca809e9948471866e403666347eb0a"

ARTIFACT = (
    "causal_inner_nonlinear_temporal_refinement_manifest_"
    "wp10c9d6c7c3b3a"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_temporal_refinement_manifest_"
    "wp10c9d6c7c3b3a.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_temporal_refinement_manifest_"
    "wp10c9d6c7c3b3a.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_NONLINEAR_TEMPORAL_REFINEMENT_MANIFEST_"
    "WP10C9D6C7C3B3A_2026-08-01.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE

LAYOUTS = c3b2a.LAYOUTS
COARSE_LAYOUT, MIDDLE_LAYOUT, FINE_LAYOUT = LAYOUTS
TIMESTEP_LEVELS_SECONDS = np.asarray((1.0e-5, 5.0e-6, 2.5e-6))
HORIZON_SECONDS = 4.0e-5
COMMON_OUTPUT_TIMES_SECONDS = np.arange(5, dtype=float) * 1.0e-5
STEP_COUNTS = np.rint(HORIZON_SECONDS / TIMESTEP_LEVELS_SECONDS).astype(
    np.int64
)
NEW_REFINED_STEP_COUNT = int(np.sum(STEP_COUNTS[1:]))

PRIMARY_PROFILE = "p3_buffer45__inward_shear"
SECONDARY_PROFILE = "p3_buffer45__outward_shear"
PRIMARY_VARIANT = 1.0
NONLINEAR_CONTROL_VARIANTS = (-1.0, 0.5, -0.5)
PRIMARY_CASE_ID = f"{PRIMARY_PROFILE}__p1"
SECONDARY_CASE_ID = f"{SECONDARY_PROFILE}__p1"

MINIMUM_TEMPORAL_RMS_ORDER = 1.5
MINIMUM_TEMPORAL_MAXIMUM_ORDER = 1.5
MINIMUM_TEMPORAL_COMPONENT_ORDER = 1.5
MAXIMUM_FINE_NORMALIZED_TEMPORAL_DIFFERENCE = 0.05
MAXIMUM_SELECTED_STEP_RICHARDSON_ERROR = 0.005
MINIMUM_HISTORY_COSINE = 0.90
MINIMUM_REFINEMENT_ERROR_COSINE = 0.90
MINIMUM_RELATIVE_ACTIVITY = c3b2a.MINIMUM_RELATIVE_ACTIVITY
OBSERVABILITY_FACTOR = 5.0
MAXIMUM_SCALED_RESIDUAL = 1.0e-10
MAXIMUM_LEDGER_DEFECT = 1.0e-12

PARENT_DIRECTORY = c3b2b.CANONICAL_DIRECTORY
SPATIAL_MANIFEST_DIRECTORY = c3b2a.CANONICAL_DIRECTORY
PREFLIGHT_DIRECTORY = c3b2a.STEP4_DIRECTORY

CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "temporal_refinement_manifest.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_parent() -> tuple[dict, dict]:
    parent = _read_json(PARENT_DIRECTORY / "summary.json")
    spatial_manifest = _read_json(
        SPATIAL_MANIFEST_DIRECTORY
        / "nonlinear_spatial_export_manifest.json"
    )
    if (
        not parent["passed"]
        or parent["classification"]
        != "nonlinear_short_horizon_state_and_tier_I_export_spatial_"
        "pilot_certified_temporal_refinement_manifest_authorized"
        or parent["authorized_next"]
        != "WP10c9d6c7c3b3a_nonlinear_temporal_refinement_"
        "pilot_manifest"
        or parent["temporal_convergence_certified"]
        or parent["long_nonlinear_physical_ladder_authorized"]
        or parent["fixed_q_micro_solver_authorized"]
        or parent["reduced_slow_evolution_authorized"]
    ):
        raise RuntimeError("c3b2b temporal-manifest authorization changed")
    if (
        spatial_manifest["interpretation_limits"]
        ["temporal_convergence_certified"]
        or spatial_manifest["interpretation_limits"]
        ["nonlinear_physical_ladder_authorized"]
    ):
        raise RuntimeError("c3b2a interpretation limits changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT)
        != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c3b3a analyzed identity changed")
    return parent, spatial_manifest


def _case_selection_audit(parent: dict) -> dict:
    state_reports = parent["state_response"]["reports"]
    export_reports = parent["tier_I_exports"]["reports"]
    state_worst_case = min(
        state_reports,
        key=lambda case: state_reports[case]["refinement_error_cosine"],
    )
    instantaneous_worst_case = min(
        export_reports,
        key=lambda case: export_reports[case]["instantaneous"]
        ["refinement_error_cosine"],
    )
    cumulative_worst_case = min(
        export_reports,
        key=lambda case: export_reports[case]["cumulative"]
        ["refinement_error_cosine"],
    )
    if {
        state_worst_case,
        instantaneous_worst_case,
        cumulative_worst_case,
    } != {PRIMARY_CASE_ID}:
        raise RuntimeError("frozen weakest-margin temporal case changed")
    if SECONDARY_CASE_ID not in state_reports:
        raise RuntimeError("frozen outward temporal control changed")
    return {
        "passed": True,
        "selection_rule": (
            "full-amplitude case with the smallest inherited state or "
            "Tier-I refinement-error cosine; add the same-support outward "
            "full-amplitude profile as the propagation-direction control"
        ),
        "primary_case_id": PRIMARY_CASE_ID,
        "secondary_case_id": SECONDARY_CASE_ID,
        "state_worst_case": state_worst_case,
        "instantaneous_export_worst_case": instantaneous_worst_case,
        "cumulative_export_worst_case": cumulative_worst_case,
        "primary_state_error_cosine": state_reports[PRIMARY_CASE_ID]
        ["refinement_error_cosine"],
        "primary_instantaneous_export_error_cosine": export_reports[
            PRIMARY_CASE_ID
        ]["instantaneous"]["refinement_error_cosine"],
        "primary_cumulative_export_error_cosine": export_reports[
            PRIMARY_CASE_ID
        ]["cumulative"]["refinement_error_cosine"],
    }


def _cost_audit() -> dict:
    preflight = _read_json(PREFLIGHT_DIRECTORY / "summary.json")
    elapsed_by_layout: dict[str, list[float]] = {
        layout: [] for layout in LAYOUTS
    }
    for report in preflight["case_reports"]:
        elapsed_by_layout[report["layout"]].append(
            float(report["elapsed_seconds"])
        )
    median_step_seconds = np.asarray(
        [statistics.median(elapsed_by_layout[layout]) for layout in LAYOUTS]
    )
    refined_hours_per_trajectory = (
        median_step_seconds * NEW_REFINED_STEP_COUNT / 3600.0
    )
    stage_trajectory_counts = np.asarray((3, 2, 2, 3), dtype=np.int64)
    stage_layout_indices = np.asarray((0, 1, 2, 0), dtype=np.int64)
    stage_hours = np.asarray(
        [
            stage_trajectory_counts[index]
            * refined_hours_per_trajectory[layout_index]
            for index, layout_index in enumerate(stage_layout_indices)
        ]
    )
    full_matrix_hours = float(
        17.0 * np.sum(refined_hours_per_trajectory)
    )
    staged_hours = float(np.sum(stage_hours))
    if staged_hours >= full_matrix_hours:
        raise RuntimeError("staged temporal campaign no longer saves work")
    return {
        "passed": True,
        "median_step_seconds_by_layout": dict(
            zip(LAYOUTS, median_step_seconds.tolist(), strict=True)
        ),
        "new_refined_steps_per_trajectory": NEW_REFINED_STEP_COUNT,
        "refined_hours_per_trajectory": dict(
            zip(
                LAYOUTS,
                refined_hours_per_trajectory.tolist(),
                strict=True,
            )
        ),
        "stage_trajectory_counts": stage_trajectory_counts.tolist(),
        "stage_estimated_cpu_hours": stage_hours.tolist(),
        "staged_total_estimated_cpu_hours": staged_hours,
        "full_matrix_estimated_cpu_hours": full_matrix_hours,
        "staged_to_full_matrix_cost_ratio": (
            staged_hours / full_matrix_hours
        ),
    }


def _manifest(selection: dict, cost: dict) -> dict:
    temporal_gates = {
        "minimum_rms_order": MINIMUM_TEMPORAL_RMS_ORDER,
        "minimum_maximum_order": MINIMUM_TEMPORAL_MAXIMUM_ORDER,
        "minimum_significant_component_order": (
            MINIMUM_TEMPORAL_COMPONENT_ORDER
        ),
        "maximum_fine_normalized_temporal_difference": (
            MAXIMUM_FINE_NORMALIZED_TEMPORAL_DIFFERENCE
        ),
        "maximum_selected_step_richardson_error": (
            MAXIMUM_SELECTED_STEP_RICHARDSON_ERROR
        ),
        "minimum_history_cosine": MINIMUM_HISTORY_COSINE,
        "minimum_observable_refinement_error_cosine": (
            MINIMUM_REFINEMENT_ERROR_COSINE
        ),
        "minimum_relative_activity": MINIMUM_RELATIVE_ACTIVITY,
        "observability_factor": OBSERVABILITY_FACTOR,
    }
    stages = [
        {
            "work_package": "WP10c9d6c7c3b3b1",
            "name": "coarse_inward_outward_temporal_screen",
            "layout": COARSE_LAYOUT,
            "trajectories": [
                "unperturbed_background",
                PRIMARY_CASE_ID,
                SECONDARY_CASE_ID,
            ],
            "estimated_cpu_hours": cost["stage_estimated_cpu_hours"][0],
            "binding": True,
        },
        {
            "work_package": "WP10c9d6c7c3b3b2",
            "name": "middle_primary_temporal_confirmation",
            "layout": MIDDLE_LAYOUT,
            "trajectories": ["unperturbed_background", PRIMARY_CASE_ID],
            "estimated_cpu_hours": cost["stage_estimated_cpu_hours"][1],
            "conditional_on": "c3b3b1_pass",
        },
        {
            "work_package": "WP10c9d6c7c3b3b3",
            "name": "fine_primary_temporal_confirmation",
            "layout": FINE_LAYOUT,
            "trajectories": ["unperturbed_background", PRIMARY_CASE_ID],
            "estimated_cpu_hours": cost["stage_estimated_cpu_hours"][2],
            "conditional_on": "c3b3b2_pass",
        },
        {
            "work_package": "WP10c9d6c7c3b3b4",
            "name": "coarse_primary_nonlinear_symmetry_controls",
            "layout": COARSE_LAYOUT,
            "trajectories": [
                f"{PRIMARY_PROFILE}__m1",
                f"{PRIMARY_PROFILE}__p0p5",
                f"{PRIMARY_PROFILE}__m0p5",
            ],
            "reuses": ["unperturbed_background", PRIMARY_CASE_ID],
            "estimated_cpu_hours": cost["stage_estimated_cpu_hours"][3],
            "conditional_on": "c3b3b3_pass",
        },
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            "nonlinear_temporal_refinement_manifest_frozen_"
            "coarse_temporal_screen_authorized"
        ),
        "authorized_next": (
            "WP10c9d6c7c3b3b1_coarse_inward_outward_temporal_screen"
        ),
        "operator_changed": False,
        "propagation_executed": False,
        "scope": {
            "purpose": (
                "certify temporal convergence of the nonlinear "
                "perturbed-minus-background state and Tier-I exports "
                "before extending physical time"
            ),
            "timestep_levels_seconds": TIMESTEP_LEVELS_SECONDS.tolist(),
            "horizon_seconds": HORIZON_SECONDS,
            "common_output_times_seconds": (
                COMMON_OUTPUT_TIMES_SECONDS.tolist()
            ),
            "step_counts": STEP_COUNTS.tolist(),
            "each_level_uses_own_BDF1_startup_then_BDF2": True,
            "existing_1e_minus_5_history_reused_by_hash": True,
            "smaller_timestep_histories_require_new_propagation": True,
        },
        "case_selection": selection,
        "temporal_binding_contract": {
            "response_definition": (
                "perturbed nonlinear trajectory minus independently "
                "evolved unperturbed trajectory at the same layout, "
                "timestep and physical output time"
            ),
            "state_and_export_scales": (
                "unchanged c3b2a fixed physical scales"
            ),
            "state_response_binding": True,
            "instantaneous_13_export_response_binding": True,
            "cumulative_13_export_response_binding": True,
            "gates": temporal_gates,
            "temporal_order_binding_only_for_significant_observable_"
            "differences": True,
            "error_cosine_binding_rule": (
                "both temporal refinement-error norms must exceed the "
                "complete frozen numerical uncertainty envelope by the "
                "observability factor"
            ),
            "below_floor_result": (
                "certify an upper bound; do not force an order or angle"
            ),
        },
        "space_time_budget": {
            "binding_selected_step_error_is_richardson_estimate": True,
            "maximum_selected_step_normalized_error": (
                MAXIMUM_SELECTED_STEP_RICHARDSON_ERROR
            ),
            "interpretation": (
                "ten percent of the inherited 0.05 Tier-I normalized "
                "accuracy allowance, not ten percent of a near-zero raw "
                "spatial difference"
            ),
            "temporal_to_spatial_error_ratio": (
                "reported only when the spatial difference exceeds its "
                "independent uncertainty floor; not a binding raw ratio"
            ),
        },
        "numerical_uncertainty_contract": {
            "combination_rule": "conservative_envelope_not_RSS",
            "sources": [
                "nonlinear residual and independent residual replay",
                "checkpoint split/restart replay",
                "dense/colored or independent Jacobian-action spot check",
                "instantaneous export reevaluation",
                "cumulative export quadrature at common output times",
                "roundoff-scale deterministic rerun",
            ],
            "no_uncertainty_source_may_be_set_to_zero_without_evidence": (
                True
            ),
        },
        "nonlinear_response_diagnostics": {
            "status": "explanatory_until_resolved_above_uncertainty",
            "odd_response": "(R_A - R_minus_A) / 2",
            "even_response": "(R_A + R_minus_A) / 2",
            "half_amplitude_defect": "R_A - 2 R_A_over_2",
            "odd_vs_frozen_linear_tangent": True,
            "meaningfully_nonlinear_claim_requires_spatial_and_temporal_"
            "convergence_of_nonzero_remainder": True,
        },
        "method_gates": {
            "maximum_scaled_residual": MAXIMUM_SCALED_RESIDUAL,
            "maximum_discrete_ledger_defect": MAXIMUM_LEDGER_DEFECT,
            "mapped_endpoint_path_closure_gate_inherited": True,
            "checkpoint_roundtrip": "bitwise",
            "split_restart_replay": "bitwise",
            "minimum_reconstruction_factor": 1.0,
            "incoming_excision_characteristics": 0,
            "physical_admissibility_gates_inherited": True,
        },
        "fail_fast_stages": stages,
        "cost_audit": cost,
        "decision": {
            "stage_passes": (
                "commit and push its certificate, then execute only the "
                "next frozen stage"
            ),
            "temporal_order_or_budget_fails": (
                "stop duration extension and localize BDF startup, "
                "temporal storage/path history, residual and export "
                "quadrature; do not redesign the spatial operator"
            ),
            "coarse_passes_but_refined_layout_fails": (
                "audit space-time stiffness and timestep control before "
                "any physical duration extension"
            ),
            "all_four_stages_pass": (
                "authorize a definitions-only short-horizon nonlinear "
                "profile-breadth and efficient-controller manifest"
            ),
        },
        "interpretation_limits": {
            "temporal_convergence_certified": False,
            "meaningfully_nonlinear_dynamics_certified": False,
            "long_nonlinear_physical_ladder_authorized": False,
            "fixed_q_micro_solver_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
        "hard_stops": [
            "do not change the monolithic spatial operator or interface",
            "do not launch the complete 16-case by three-layout matrix",
            "do not bind a ratio to a spatial error below uncertainty",
            "do not run N1024",
            "do not launch the 0.125-second nonlinear ladder",
            "do not begin fixed-Q or reduced slow-time evolution",
            "do not stage unrelated untracked docs/reports/gpt files",
        ],
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
                    "scientific_status": (
                        "CERTIFIED" if summary["passed"] else "REJECTED"
                    ),
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case",
                "path",
                "bytes",
                "sha256",
                "scientific_status",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    global_summary = _read_json(CANONICAL_SUMMARY)
    global_summary.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": summary["passed"],
    }
    global_summary["latest_work_package"] = WORK_PACKAGE
    global_summary["latest_source_parent_commit"] = ANALYZED_BASE_COMMIT
    global_summary["case_count"] = len(global_summary["artifacts"])
    _write_json(CANONICAL_SUMMARY, global_summary)


def _report(manifest: dict) -> str:
    selection = manifest["case_selection"]
    cost = manifest["cost_audit"]
    gates = manifest["temporal_binding_contract"]["gates"]
    return "\n".join(
        [
            "# Nonlinear temporal-refinement manifest "
            "WP10c9d6c7c3b3a",
            "",
            "## Classification",
            "",
            "`nonlinear_temporal_refinement_manifest_frozen_"
            "coarse_temporal_screen_authorized`",
            "",
            "This definitions-only package changes no operator and runs no "
            "new trajectory. It freezes a staged temporal campaign instead "
            "of an immediate full profile/layout matrix.",
            "",
            "## Frozen temporal triplet",
            "",
            "- timesteps: `1e-5`, `5e-6`, `2.5e-6 s`",
            "- common horizon: `4e-5 s`",
            "- common outputs: `0`, `1e-5`, `2e-5`, `3e-5`, "
            "`4e-5 s`",
            "- each level: its own BDF1 startup followed by BDF2",
            "- response: perturbed minus independently evolved background "
            "at the same layout and timestep",
            "",
            "## Evidence-selected screen",
            "",
            f"- primary: `{selection['primary_case_id']}`",
            f"- outward control: `{selection['secondary_case_id']}`",
            "- primary inherited state/export error cosines: "
            f"`{selection['primary_state_error_cosine']:.6f}` / "
            f"`{selection['primary_instantaneous_export_error_cosine']:.6f}` "
            "/ "
            f"`{selection['primary_cumulative_export_error_cosine']:.6f}`",
            "",
            "## Binding gates",
            "",
            "- minimum temporal RMS/max/component order: "
            f"`{gates['minimum_rms_order']:.2f}` / "
            f"`{gates['minimum_maximum_order']:.2f}` / "
            f"`{gates['minimum_significant_component_order']:.2f}`",
            "- maximum fine normalized temporal difference: "
            f"`{gates['maximum_fine_normalized_temporal_difference']:.3f}`",
            "- maximum selected-step Richardson error: "
            f"`{gates['maximum_selected_step_richardson_error']:.3f}`",
            "- error angle binds only above a complete uncertainty envelope",
            "- nonlinear residual `<=1e-10`; ledgers `<=1e-12`; bitwise "
            "restart and zero incoming excision modes",
            "",
            "The Richardson budget is ten percent of the inherited `0.05` "
            "Tier-I accuracy allowance. It is deliberately not ten percent "
            "of the approximately `1e-9` raw spatial difference.",
            "",
            "## Cost-bounded execution",
            "",
            "- coarse inward/outward screen: "
            f"`{cost['stage_estimated_cpu_hours'][0]:.2f} CPU h`",
            "- conditional middle primary confirmation: "
            f"`{cost['stage_estimated_cpu_hours'][1]:.2f} CPU h`",
            "- conditional fine primary confirmation: "
            f"`{cost['stage_estimated_cpu_hours'][2]:.2f} CPU h`",
            "- conditional coarse nonlinear controls: "
            f"`{cost['stage_estimated_cpu_hours'][3]:.2f} CPU h`",
            "- complete staged estimate: "
            f"`{cost['staged_total_estimated_cpu_hours']:.2f} CPU h`",
            "- rejected immediate full-matrix estimate: "
            f"`{cost['full_matrix_estimated_cpu_hours']:.2f} CPU h`",
            "",
            "## Authorized next",
            "",
            "`WP10c9d6c7c3b3b1_coarse_inward_outward_temporal_screen`",
            "",
            "Temporal, long-horizon, fixed-Q and reduced slow evolution "
            "remain uncertified and blocked.",
            "",
        ]
    )


def main() -> None:
    parent, _ = _validate_parent()
    selection = _case_selection_audit(parent)
    cost = _cost_audit()
    manifest = _manifest(selection, cost)
    passed = bool(selection["passed"] and cost["passed"])
    if not passed:
        manifest["classification"] = (
            "nonlinear_temporal_refinement_manifest_failed"
        )
        manifest["authorized_next"] = None

    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "layouts": list(LAYOUTS),
        "timestep_levels_seconds": TIMESTEP_LEVELS_SECONDS.tolist(),
        "horizon_seconds": HORIZON_SECONDS,
        "common_output_times_seconds": COMMON_OUTPUT_TIMES_SECONDS.tolist(),
        "step_counts": STEP_COUNTS.tolist(),
        "primary_case_id": PRIMARY_CASE_ID,
        "secondary_case_id": SECONDARY_CASE_ID,
        "nonlinear_control_variants": list(NONLINEAR_CONTROL_VARIANTS),
        "temporal_gates": manifest["temporal_binding_contract"]["gates"],
        "fail_fast_stages": manifest["fail_fast_stages"],
    }
    _write_json(CONFIG_PATH, config)
    _write_json(MANIFEST_PATH, manifest)

    median_step_seconds = np.asarray(
        [
            cost["median_step_seconds_by_layout"][layout]
            for layout in LAYOUTS
        ]
    )
    refined_hours = np.asarray(
        [cost["refined_hours_per_trajectory"][layout] for layout in LAYOUTS]
    )
    decisive = {
        "timestep_levels_seconds": TIMESTEP_LEVELS_SECONDS,
        "common_output_times_seconds": COMMON_OUTPUT_TIMES_SECONDS,
        "step_counts": STEP_COUNTS,
        "median_step_seconds_by_layout": median_step_seconds,
        "refined_hours_per_trajectory": refined_hours,
        "stage_estimated_cpu_hours": np.asarray(
            cost["stage_estimated_cpu_hours"]
        ),
        "selection_error_cosines": np.asarray(
            (
                selection["primary_state_error_cosine"],
                selection["primary_instantaneous_export_error_cosine"],
                selection["primary_cumulative_export_error_cosine"],
            )
        ),
    }
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)

    input_paths = {
        "parent_summary": PARENT_DIRECTORY / "summary.json",
        "parent_arrays": PARENT_DIRECTORY / "decisive_arrays.npz",
        "spatial_manifest": SPATIAL_MANIFEST_DIRECTORY
        / "nonlinear_spatial_export_manifest.json",
        "preflight_summary": PREFLIGHT_DIRECTORY / "summary.json",
    }
    source_hashes = {
        path: _sha256(ROOT / path)
        for path in (THIS_RUNNER, THIS_TEST)
        if (ROOT / path).exists()
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "analyzed_base_parent_commit": ANALYZED_BASE_PARENT,
        "analyzed_base_tree_sha": ANALYZED_BASE_TREE,
        "passed": passed,
        "classification": manifest["classification"],
        "authorized_next": manifest["authorized_next"],
        "operator_changed": False,
        "propagation_executed": False,
        "parent_classification": parent["classification"],
        "case_selection_audit": selection,
        "cost_audit": cost,
        "coarse_temporal_screen_authorized": passed,
        "temporal_convergence_certified": False,
        "meaningfully_nonlinear_dynamics_certified": False,
        "long_nonlinear_physical_ladder_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "config_sha256": _sha256(CONFIG_PATH),
        "manifest_file_sha256": _sha256(MANIFEST_PATH),
        "decisive_arrays_sha256": _sha256(DECISIVE_ARRAYS),
        "decisive_array_hashes": {
            name: hashlib.sha256(
                np.ascontiguousarray(values).view(np.uint8)
            ).hexdigest()
            for name, values in decisive.items()
        },
        "implementation_source_hashes": source_hashes,
        "input_hashes": {
            name: _sha256(path) for name, path in input_paths.items()
        },
    }
    _write_json(SUMMARY_PATH, summary)
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "command": (
                "PYTHONPATH=src "
                "/Users/huanyang/.cache/codex-runtimes/"
                "codex-primary-runtime/dependencies/python/bin/python3 "
                + THIS_RUNNER
            ),
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "implementation_parent_commit": _git_value(
                "rev-parse", "HEAD"
            ),
            "implementation_parent_tree_sha": _git_value(
                "rev-parse", "HEAD^{tree}"
            ),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "implementation_source_hashes": source_hashes,
            "input_hashes": summary["input_hashes"],
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_report(manifest), encoding="utf-8")
    names = (
        "config.json",
        "temporal_refinement_manifest.json",
        "decisive_arrays.npz",
        "summary.json",
        "provenance.json",
    )
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "\n".join(
            f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}"
            for name in names
        )
        + "\n",
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
