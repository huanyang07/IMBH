#!/usr/bin/env python3
"""Freeze a cost-bounded route to the third-rung spatial certificate."""

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

import run_causal_inner_nonlinear_third_duration_rung_spatial_confirmation_manifest_wp10c9d6c7c3b5c3g as c3g  # noqa: E402
import run_causal_inner_nonlinear_coarse_third_duration_rung_completion_wp10c9d6c7c3b5c3d as c3d  # noqa: E402
import run_causal_inner_nonlinear_coarse_heldout_third_duration_rung_screen_wp10c9d6c7c3b5c3f as c3f  # noqa: E402
import run_causal_inner_nonlinear_profile_breadth_spatial_wp10c9d6c7c3b4b3 as b4b3  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c3g1"
ANALYZED_BASE_COMMIT = "10f3665fb106843c6618a0e1e6e3ac61ace62d27"
ANALYZED_BASE_PARENT = "05ce1e5b63eecdddef06402977d7bb417679e8d9"
ANALYZED_BASE_TREE = "57526454f0c1cb777c6872398ecaf2dbf5d7998c"

PROFILE_NAMES = (
    c3g.GENERIC_PROFILE,
    "p4__inward_acoustic",
    "p4__outward_acoustic",
    "p3_buffer45__material",
    "p4__inward_shear_acoustic_mix",
)
LAYOUTS = tuple(c3g.LAYOUTS)
COARSE_LAYOUT, MIDDLE_LAYOUT, FINE_LAYOUT = LAYOUTS

SPATIAL_GATES = dict(c3g.SPATIAL_GATES)
TANGENT_GATES = {
    "maximum_internal_discrete_residual_jvp_relative_defect": 1.0e-6,
    "maximum_scaled_state_response_discrepancy": 5.0e-3,
    "maximum_scaled_Tier_I_response_discrepancy": 5.0e-3,
    "minimum_state_response_history_cosine": 0.99,
    "minimum_Tier_I_response_history_cosine": 0.99,
    "surrogate_fraction_of_fine_difference_acceptance_budget": 0.10,
    "coarse_long_horizon_profiles_required": PROFILE_NAMES,
    "short_horizon_layouts_required": LAYOUTS,
}
TEMPORAL_SCHEDULE_GATES = {
    "each_layout_owns_its_schedule": True,
    "coarse_schedule_reuse_on_middle_or_fine_forbidden_without_audit": True,
    "maximum_sampled_step_doubling_to_spatial_error_ratio": 0.10,
    "sampled_windows": (
        "first_two_accepted_steps",
        "middle_declared_output_window",
        "final_strict_shadow_window",
    ),
    "same_layout_base_schedule_reuse_for_tangent_and_anchor": True,
}
FINE_DECISION_GATES = {
    "defect_estimator_is_triage_not_final_certificate": True,
    "calibration_required_on_existing_full_fine_short_horizon_evidence": True,
    "minimum_effectivity_samples": 5,
    "conservative_effectivity_safety_factor": 5.0,
    "maximum_estimator_bound_to_fine_difference_acceptance_budget": 0.20,
    "binding_fine_confirmation_minimum": "fine_nonlinear_base_plus_discrete_tangent",
    "full_fine_nonlinear_perturbed_triggered_by": (
        "middle_tangent_anchor_gate_failure",
        "fine_estimator_calibration_failure",
        "fine_estimator_bound_near_spatial_gate",
        "observable_nonlinear_remainder_not_bounded_by_surrogate_budget",
    ),
}
COST_CONTRACT = {
    "historical_bruteforce_lower_bound_hours": 61.7,
    "maximum_total_projected_new_nonlinear_wall_hours_before_execution": 24.0,
    "maximum_single_unattended_stage_wall_hours": 12.0,
    "projection_is_a_scheduling_stop_not_a_scientific_gate": True,
    "mandatory_new_full_nonlinear_trajectories": (
        f"{MIDDLE_LAYOUT}__base",
        f"{MIDDLE_LAYOUT}__generic_anchor",
        f"{FINE_LAYOUT}__base",
    ),
    "conditional_new_full_nonlinear_trajectories": (
        f"{FINE_LAYOUT}__generic_anchor",
    ),
    "block_tangent_profiles_share_one_step_factorization": True,
    "benchmark_before_middle_propagation": True,
}

ARTIFACT = (
    "causal_inner_nonlinear_third_duration_rung_cost_bounded_spatial_"
    "confirmation_manifest_wp10c9d6c7c3b5c3g1"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_third_duration_rung_cost_bounded_"
    "spatial_confirmation_manifest_wp10c9d6c7c3b5c3g1.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_third_duration_rung_cost_bounded_"
    "spatial_confirmation_manifest_wp10c9d6c7c3b5c3g1.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_THIRD_DURATION_RUNG_"
    "COST_BOUNDED_SPATIAL_CONFIRMATION_MANIFEST_"
    "WP10C9D6C7C3B5C3G1_2026-08-05.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "cost_bounded_manifest.json"
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
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_parent() -> dict:
    parent = _read_json(c3g.SUMMARY_PATH)
    if (
        not parent["passed"]
        or parent["classification"]
        != "third_duration_rung_spatial_confirmation_manifest_frozen_"
        "middle_fine_generic_propagation_authorized"
        or parent["middle_fine_generic_spatial_confirmation_executed"]
        or parent["third_duration_rung_spatial_convergence_certified"]
    ):
        raise RuntimeError("c3g status changed before cost-bounded manifest")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("cost-bounded analyzed identity changed")
    return parent


def _manifest() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            "third_duration_rung_cost_bounded_spatial_confirmation_manifest_"
            "frozen_discrete_bdf_tangent_calibration_authorized"
        ),
        "passed": True,
        "definitions_only": True,
        "propagation_executed": False,
        "operator_changed": False,
        "production_defaults_changed": False,
        "c3g_historical_classification_preserved": (
            "third_duration_rung_spatial_confirmation_manifest_frozen_"
            "middle_fine_generic_propagation_authorized"
        ),
        "c3h_bruteforce_execution": {
            "started": True,
            "stopped_before_complete_trajectory": True,
            "completed_scientific_evidence": False,
            "classification_issued": False,
            "prospective_route_superseded": True,
            "reason": "projected_middle_fine_cost_exceeded_user_budget",
        },
        "scientific_assessment": {
            "accepted_suggestion": (
                "use_the_complete_discrete_variable_step_BDF_tangent_and_"
                "conditional_full_nonlinear_anchors"
            ),
            "required_refinements": (
                "linearize_mapped_and_responsive_height_history_actions",
                "derive_a_safe_schedule_separately_for_each_layout",
                "use_the_fine_defect_estimator_only_for_triage",
                "retain_a_binding_fine_base_plus_tangent_confirmation",
            ),
            "frozen_continuous_generator_is_not_an_acceptable_substitute": True,
            "post_hoc_surrogate_tuning_forbidden": True,
        },
        "phase_h1_discrete_tangent_calibration": {
            "authorized": True,
            "new_long_trajectory_required": False,
            "complete_discrete_residual": (
                "variable_step_BDF2_current_state_old_state_primitive_history_"
                "mapped_storage_history_responsive_height_history"
            ),
            "tangent_history_state": (
                "primitive_increment",
                "mapped_storage_path_increment",
                "responsive_height_storage_path_increment",
            ),
            "coarse_long_horizon_evidence": {
                "generic": str(c3d.DECISIVE_ARRAYS.relative_to(ROOT)),
                "heldouts": str(c3f.DECISIVE_ARRAYS.relative_to(ROOT)),
            },
            "short_horizon_three_layout_evidence": str(
                b4b3.DECISIVE_ARRAYS.relative_to(ROOT)
            ),
            "profiles": PROFILE_NAMES,
            "layouts": LAYOUTS,
            "state_and_all_Tier_I_export_JVPs_required": True,
            "gates": TANGENT_GATES,
            "decision": {
                "pass": "authorize_middle_cost_bounded_anchor_manifest",
                "fail": "retain_bruteforce_c3g_as_only_certification_route",
            },
        },
        "phase_h2_middle_confirmation": {
            "authorized": False,
            "definitions_required_after_h1_pass": True,
            "nonlinear_base_uses_layout_owned_adaptive_schedule": True,
            "generic_nonlinear_anchor_replays_frozen_middle_schedule": True,
            "block_tangent_profiles": PROFILE_NAMES,
            "sampled_step_doubling_contract": TEMPORAL_SCHEDULE_GATES,
            "generic_tangent_anchor_gates": TANGENT_GATES,
            "stop_before_fine_on_failure": True,
        },
        "phase_h3_fine_confirmation": {
            "authorized": False,
            "definitions_required_after_middle_pass": True,
            "fine_defect_error_transport_preflight": True,
            "fine_decision_gates": FINE_DECISION_GATES,
            "minimum_binding_execution": (
                "one_fine_nonlinear_base_plus_block_discrete_BDF_tangent"
            ),
            "full_fine_generic_nonlinear_anchor_is_conditional": True,
            "spatial_gates_unchanged": SPATIAL_GATES,
        },
        "cost_contract": COST_CONTRACT,
        "binding_final_certificate": {
            "coarse_evidence": "committed_c3d_and_c3f",
            "middle_evidence": "nonlinear_base_generic_anchor_and_block_tangent",
            "fine_evidence": "nonlinear_base_and_block_tangent_minimum",
            "common_parent_state_response": True,
            "correct_face_Tier_I_exports": True,
            "active_coupling_face_indices": c3g.ACTIVE_COUPLING_FACE_INDICES,
            "spatial_gates": SPATIAL_GATES,
            "temporal_uncertainty_contract": c3g.TEMPORAL_UNCERTAINTY_GATES,
            "surrogate_error_is_added_to_not_subtracted_from_error_budget": True,
            "no_spatial_certificate_from_defect_estimator_alone": True,
        },
        "hard_stops": (
            "do not amend or relabel c3g",
            "do not resume brute_force_c3h before tangent calibration fails",
            "do not use the frozen continuous generator as the discrete tangent",
            "do not reuse a coarse time schedule blindly on refined layouts",
            "do not issue a fine spatial certificate from a defect estimate alone",
            "do not relax the c3g spatial_temporal_method_or_replay_gates",
            "do not begin the fourth duration rung fixed_Q or reduced evolution",
            "do not add tide wind hot_state S_curve or QPE_cycle physics",
            "do not use N1024 as a rescue",
        ),
        "authorized_next": (
            "WP10c9d6c7c3b5c3h1_discrete_BDF_tangent_calibration"
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


def _report(manifest: dict) -> str:
    return "\n".join(
        [
            "# Cost-bounded third-rung spatial-confirmation manifest WP10c9d6c7c3b5c3g1",
            "",
            "## Classification",
            "",
            f"`{manifest['classification']}`",
            "",
            "This definitions-only package preserves c3g and prospectively replaces only its expensive execution route. The incomplete c3h attempt produced no completed trajectory or scientific classification.",
            "",
            "## Assessment of the proposed cost reduction",
            "",
            "The central proposal is accepted: differentiate the complete discrete variable-step BDF residual along an independently evolved nonlinear base, reuse one step factorization for a block of perturbation right-hand sides, and reserve full nonlinear refined perturbations for calibration or triggered confirmation.",
            "",
            "Three refinements are binding. The tangent includes primitive, mapped-storage, and responsive-height BDF histories; every layout establishes its own safe timestep schedule; and a prolonged-state fine defect estimate is a triage device, not a spatial certificate by itself.",
            "",
            "## Frozen route",
            "",
            "1. Calibrate the complete discrete BDF tangent against the committed coarse 5 ms nonlinear generic and four held-out responses, plus the committed three-layout short-horizon evidence.",
            "2. After calibration only, run one middle nonlinear base and one generic nonlinear anchor; propagate all frozen profiles as block tangent right-hand sides and audit time accuracy at declared windows.",
            "3. Calibrate the fine defect/error-transport estimate. The minimum binding fine route remains one nonlinear fine base plus the discrete tangent. A full nonlinear fine perturbed anchor is triggered only by a failed or marginal surrogate bound.",
            "",
            "The original c3g spatial gates remain unchanged, and surrogate uncertainty is added to the temporal/spatial budget. The plan targets no more than 24 projected new nonlinear wall-hours and no unattended stage longer than 12 hours; exceeding that projection stops execution for further optimization rather than weakening science gates.",
            "",
            f"Authorized next: `{manifest['authorized_next']}`.",
            "",
            "The fourth duration rung, fixed-Q experiments, reduced slow evolution, tide, wind, production promotion, and N1024 remain blocked.",
            "",
        ]
    )


def main() -> int:
    parent = _validate_parent()
    manifest = _manifest()
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "profiles": PROFILE_NAMES,
        "layouts": LAYOUTS,
        "spatial_gates": SPATIAL_GATES,
        "tangent_gates": TANGENT_GATES,
        "temporal_schedule_gates": TEMPORAL_SCHEDULE_GATES,
        "fine_decision_gates": FINE_DECISION_GATES,
        "cost_contract": COST_CONTRACT,
        "definitions_only": True,
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": manifest["classification"],
        "passed": True,
        "definitions_only": True,
        "propagation_executed": False,
        "operator_changed": False,
        "production_defaults_changed": False,
        "parent_classification_preserved": parent["classification"],
        "c3h_completed_scientific_evidence": False,
        "bruteforce_c3h_prospectively_authorized": False,
        "discrete_BDF_tangent_calibration_authorized": True,
        "middle_cost_bounded_propagation_authorized": False,
        "fine_cost_bounded_propagation_authorized": False,
        "third_duration_rung_spatial_convergence_certified": False,
        "fourth_duration_rung_manifest_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": manifest["authorized_next"],
    }
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "analyzed_base_parent": ANALYZED_BASE_PARENT,
        "analyzed_base_tree": ANALYZED_BASE_TREE,
        "working_head": _git_value("rev-parse", "HEAD"),
        "working_tree": _git_value("rev-parse", "HEAD^{tree}"),
        "runner": THIS_RUNNER,
        "test": THIS_TEST,
        "report": REPORT_RELATIVE,
        "input_hashes": {
            "c3g_summary": _sha256(c3g.SUMMARY_PATH),
            "c3g_manifest": _sha256(c3g.MANIFEST_PATH),
            "c3d_arrays": _sha256(c3d.DECISIVE_ARRAYS),
            "c3f_arrays": _sha256(c3f.DECISIVE_ARRAYS),
            "b4b3_arrays": _sha256(b4b3.DECISIVE_ARRAYS),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "command": (
            "PYTHONPATH=src:scripts python scripts/"
            "run_causal_inner_nonlinear_third_duration_rung_cost_bounded_"
            "spatial_confirmation_manifest_wp10c9d6c7c3b5c3g1.py"
        ),
    }

    _write_json(CONFIG_PATH, config)
    _write_json(MANIFEST_PATH, manifest)
    _write_json(SUMMARY_PATH, summary)
    _write_json(PROVENANCE_PATH, provenance)
    REPORT_PATH.write_text(_report(manifest), encoding="utf-8")
    sums = []
    for path in (CONFIG_PATH, MANIFEST_PATH, PROVENANCE_PATH, SUMMARY_PATH):
        sums.append(f"{_sha256(path)}  {path.name}")
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "\n".join(sums) + "\n",
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(summary["classification"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
