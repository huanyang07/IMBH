#!/usr/bin/env python3
"""Freeze the one-Q leading-two plus HMM nonlinear-pilot contract.

Definitions only.  The package authorizes an implementation/JVP preflight of
the state-dependent constrained step, not a constrained trajectory.
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

import run_causal_inner_face36_leading_two_plus_hmm_fixed_q_preflight_wp10c9d6c7c3b5c4f22 as c4f22  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f23"
ARTIFACT = (
    "causal_inner_face36_one_q_nonlinear_pilot_manifest_"
    "wp10c9d6c7c3b5c4f23"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_face36_one_q_nonlinear_pilot_manifest_"
    "wp10c9d6c7c3b5c4f23.py"
)
THIS_TEST = (
    "tests/test_causal_inner_face36_one_q_nonlinear_pilot_manifest_"
    "wp10c9d6c7c3b5c4f23.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_FACE36_ONE_Q_NONLINEAR_PILOT_MANIFEST_"
    "WP10C9D6C7C3B5C4F23_2026-08-13.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "pilot_manifest.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


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


def _authorization() -> dict:
    summary = _read(c4f22.SUMMARY_PATH)
    expected = (
        "WP10c9d6c7c3b5c4f23_definitions_only_one_Q_leading_two_plus_HMM_"
        "nonlinear_pilot_manifest"
    )
    if (
        not summary["passed"]
        or not summary["fixed_Q_KKT_algebra_certified"]
        or not summary["frozen_projected_local_tangent_certified"]
        or summary["state_dependent_constrained_tangent_certified"]
        or summary["fixed_Q_micro_solver_authorized"]
        or summary["nonlinear_retained_mode_pilot_authorized"]
        or not summary["one_Q_nonlinear_pilot_manifest_authorized"]
        or summary["authorized_next"] != expected
    ):
        raise RuntimeError("c4f23 authorization changed")
    return summary


def _manifest(parent: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            "one_Q_leading_two_plus_HMM_nonlinear_pilot_manifest_frozen_"
            "state_dependent_constrained_step_preflight_authorized"
        ),
        "definitions_only": True,
        "trajectory_executed": False,
        "physical_operator_changed": False,
        "inherited_architecture": {
            "formal_state": "Q3_plus_a2_plus_Z_guard",
            "Q3": "exact_exterior_domain_M_J_E_state_map",
            "a2": "cross_grid_stable_leading_two_state_amplitudes",
            "Z_guard": (
                "remaining_DAE_state_storage_histories_rotatable_weak_block_"
                "and_refinement_complement"
            ),
            "binding_exchange": "certified_face36_exterior_partition",
            "raw_face48_exchange_forbidden": True,
            "direct_two_mode_face36_law_rejected": True,
            "guard_mixing_or_decay_assumed": False,
            "c4f22_maximum_frozen_state_gain": max(
                parent["middle"]["maximum_frozen_state_transient_gain"],
                parent["fine"]["maximum_frozen_state_transient_gain"],
            ),
        },
        "representative_one_Q_state": {
            "layout": "middle_primary",
            "time_seconds": 0.020,
            "state": "committed_middle_20ms_nonlinear_base_endpoint",
            "history": "exact_committed_variable_step_BDF2_history",
            "Q_target": "Q3_of_the_committed_endpoint",
            "excision_to_face36_buffer_remains_in_Z_guard": True,
        },
        "state_dependent_constrained_system": {
            "continuous_KKT": "[M(p),-B_Q(p);DQ3(p),0]*[p_dot,lambda]=[-R(p),0]",
            "B_Q": "ledger_derived_macro_reaction_from_c4f15_generalized_to_p",
            "finite_step_unknowns": ["new_primitive_state", "three_multipliers"],
            "finite_step_constraint": "Q3(p_new)-Q_target=0",
            "reaction_term": (
                "insert_once_into_the_complete_scaled_monolithic_BDF_residual_"
                "with_sign_and_time_units_derived_from_the_continuous_KKT"
            ),
            "temporal_storage": (
                "unchanged_mapped_endpoint_plus_responsive_height_path_"
                "increments_and_complete_BDF_history"
            ),
            "linearization_must_include": [
                "D_M",
                "D_R",
                "D_DQ3",
                "D_B_Q",
                "multiplier_coupling",
                "mapped_storage_history",
                "responsive_height_history",
            ],
            "frozen_P_times_G_is_reference_only": True,
            "Euclidean_projection_forbidden": True,
            "manual_primitive_freezing_forbidden": True,
            "residual_subtraction_forbidden": True,
            "constraint_reaction_M_J_E_ledgers_required": True,
        },
        "authorized_state_dependent_step_preflight": {
            "work_package": (
                "WP10c9d6c7c3b5c4f24_analysis_only_state_dependent_fixed_Q_"
                "step_and_JVP_preflight"
            ),
            "new_physical_trajectory": False,
            "new_tangent_trajectory": False,
            "implement_augmented_backward_Euler_and_variable_step_BDF2": True,
            "verify_zero_multiplier_reduces_to_unconstrained_residual": True,
            "verify_small_timestep_limit_against_c4f22_continuous_KKT": True,
            "compare_dense_colored_and_directional_JVPs": True,
            "directions": [
                "two_a2_lifts",
                "four_weak_guard_lifts",
                "six_refinement_complement_lifts",
                "twelve_smooth_random_guard_lifts",
                "three_multiplier_directions",
            ],
            "finite_equal_Q_lifts": {
                "method": (
                    "reaction_coordinate_Newton_correction_to_exact_Q3_"
                    "after_each_prospective_scaled_lift"
                ),
                "use_smallest_common_admissible_amplitude": True,
                "amplitude_selected_before_nonlinear_response": True,
                "require_sign_symmetric_admissibility_preflight": True,
                "do_not_run_all_lifts_nonlinearly": True,
            },
        },
        "preflight_gates": {
            "maximum_Q3_endpoint_relative_defect": 1.0e-12,
            "maximum_continuous_KKT_relative_defect": 1.0e-10,
            "maximum_augmented_step_scaled_residual": 1.0e-10,
            "maximum_dense_colored_Jacobian_relative_defect": 1.0e-9,
            "maximum_directional_JVP_relative_defect": 1.0e-8,
            "maximum_zero_multiplier_reduction_defect": 1.0e-12,
            "maximum_small_timestep_KKT_closure_defect": 1.0e-8,
            "maximum_reaction_ledger_relative_defect": 1.0e-12,
            "maximum_constraint_work_ledger_relative_defect": 1.0e-12,
            "maximum_face36_directional_JVP_relative_defect": 1.0e-8,
            "maximum_reconstruction_factor": 1.0,
            "minimum_scattering_optical_depth": 1.0,
            "maximum_H_over_R": 0.12,
            "incoming_excision_characteristics": 0,
        },
        "conditional_one_Q_pilot": {
            "not_authorized_by_this_manifest": True,
            "primary_layout": "middle",
            "starting_time_seconds": 0.020,
            "one_constrained_nonlinear_base": True,
            "one_unconstrained_short_drift_control": True,
            "screen_24_equal_Q_lifts_by_one_block_tangent": True,
            "maximum_full_nonlinear_anchor_lifts": 2,
            "anchor_selection_frozen_rule": [
                "largest_predicted_persistent_face36_effect_from_a2_block",
                "largest_predicted_persistent_face36_effect_from_Z_guard_block",
            ],
            "fail_fast_windows_ms": [0.2, 1.0, 2.0, 5.0, 10.0, 20.0],
            "fine_full_trajectory_automatic": False,
            "fine_check_order": [
                "endpoint_JVP_and_residual_correction",
                "short_temporal_shadow_if_needed",
                "full_fine_only_if_still_inconclusive",
            ],
            "binding_outputs": [
                "instantaneous_face36_exterior_partition",
                "cumulative_face36_exterior_partition",
                "window_mean_face36_exterior_partition",
                "a2_history",
                "Q3_drift",
                "constraint_multiplier_and_reaction_ledgers",
                "guard_state_and_storage_history_statistics",
            ],
            "prospective_scientific_gates": {
                "maximum_scaled_Q3_drift": 1.0e-10,
                "maximum_tangent_to_nonlinear_anchor_fraction": 0.01,
                "maximum_temporal_to_response_uncertainty_fraction": 0.10,
                "maximum_late_window_mean_relative_change": 0.05,
                "maximum_lift_dependent_mean_relative_difference": 0.05,
            },
        },
        "decision_tree": {
            "step_or_JVP_preflight_fails": "stop_before_any_fixed_Q_microburst",
            "method_passes_guard_statistics_converge": (
                "authorize_exact_one_Q_nonlinear_pilot_manifest_execution"
            ),
            "a2_persists_guard_mean_converges": (
                "retain_Q3_plus_a2_with_short_HMM_guard_burst"
            ),
            "guard_mean_is_lift_dependent": (
                "retain_longer_inner_solver_or_explicit_memory_kernel"
            ),
            "multiple_conditional_branches": "add_discrete_hysteresis_state",
            "no_low_dimensional_observable_structure": "retain_inner_micro_solver",
        },
        "cost_contract": {
            "state_dependent_step_preflight_wall_hours": [1.0, 3.0],
            "conditional_middle_one_Q_pilot_wall_hours": [6.0, 15.0],
            "one_factorization_24_RHS": True,
            "sparse_nonlinear_anchors_only": True,
            "adaptive_window_extension": True,
            "no_automatic_fine_or_50ms_run": True,
            "checkpoint_each_fail_fast_window": True,
        },
        "hard_stops": [
            "do_not_treat_frozen_PG_as_the_state_dependent_constrained_JVP",
            "do_not_begin_a_microburst_before_augmented_step_JVP_certification",
            "do_not_fit_a_direct_two_mode_face36_output_law",
            "do_not_discard_Z_guard_or_storage_history",
            "do_not_run_all_24_lifts_nonlinearly",
            "do_not_use_raw_face48_as_slow_exchange",
            "do_not_start_50ms_or_reduced_slow_evolution",
        ],
        "state_dependent_fixed_Q_step_preflight_authorized": True,
        "fixed_Q_micro_solver_authorized": False,
        "one_Q_nonlinear_pilot_propagation_authorized": False,
        "fifty_ms_propagation_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": (
            "WP10c9d6c7c3b5c4f24_analysis_only_state_dependent_fixed_Q_"
            "step_and_JVP_preflight"
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
    parent = _authorization()
    manifest = _manifest(parent)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": manifest["classification"],
        "passed": True,
        "definitions_only": True,
        "trajectory_executed": False,
        "physical_operator_changed": False,
        "inherited_fixed_Q_KKT_algebra_certified": True,
        "inherited_frozen_projected_local_tangent_certified": True,
        "state_dependent_constrained_tangent_certified": False,
        "state_dependent_fixed_Q_step_preflight_authorized": True,
        "fixed_Q_micro_solver_authorized": False,
        "one_Q_nonlinear_pilot_propagation_authorized": False,
        "fifty_ms_propagation_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "guard_mixing_or_decay_claimed": False,
        "raw_face48_export_rejection_preserved": True,
        "authorized_next": manifest["authorized_next"],
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "representative_layout": "middle",
            "representative_time_seconds": 0.020,
            "macro_dimension": 3,
            "explicit_memory_dimension": 2,
            "screened_equal_Q_lifts": 24,
            "maximum_full_nonlinear_anchor_lifts": 2,
            "shared_exchange_parent_face": 36,
            "raw_face48_exchange_forbidden": True,
        },
    )
    _write(MANIFEST_PATH, manifest)
    _write(SUMMARY_PATH, summary)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# Face-36 one-Q nonlinear-pilot manifest\n\n"
        f"Classification: `{summary['classification']}`.\n\n"
        "This package is definitions-only. It freezes the exact state-dependent "
        "constraint, augmented BDF residual/JVP, finite equal-Q lift, cost, and "
        "fail-fast contracts for the first one-Q pilot.\n\n"
        "The next package may implement and audit the augmented constrained "
        "backward-Euler/BDF2 step at the committed middle 20 ms endpoint. It may "
        "not advance a constrained trajectory. The JVP must include derivatives "
        "of `M`, `R`, `DQ3`, `B_Q`, multiplier coupling, and both storage-history "
        "channels; the frozen `P G` operator from c4f22 is reference-only.\n\n"
        "Strong frozen transient amplification prevents any assumption that the "
        "guard rapidly decays. If the step/JVP preflight later passes, a separate "
        "execution manifest may authorize one middle constrained base, a 24-RHS "
        "block tangent, and at most two prospectively selected nonlinear anchors. "
        "Fine and 50 ms runs remain conditional.\n",
        encoding="utf-8",
    )
    _write(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "source_parent_commit": _git("rev-parse", "HEAD"),
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "parent_summary_sha256": _sha(c4f22.SUMMARY_PATH),
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
