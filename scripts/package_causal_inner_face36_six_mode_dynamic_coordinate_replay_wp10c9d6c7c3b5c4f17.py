#!/usr/bin/env python3
"""Package the fail-fast middle-only c4f17 result.

The numerical replay was executed by the c4f17 runner.  This script copies
its immutable checkpoint into canonical storage and freezes the definitions-
only recovery decision.  It performs no tangent or nonlinear propagation.
"""

from __future__ import annotations

import json
import platform
import sys

import numpy as np
import scipy

import run_causal_inner_face36_six_mode_dynamic_coordinate_replay_wp10c9d6c7c3b5c4f17 as c4f17


RECOVERY_WORK_PACKAGE = "WP10c9d6c7c3b5c4f18"
RECOVERY_MANIFEST_PATH = c4f17.CANONICAL_DIRECTORY / "recovery_manifest.json"


def _result() -> tuple[dict, dict, dict]:
    checkpoint_path = c4f17.CHECKPOINT_DIRECTORY / "middle.npz"
    progress_path = c4f17.CHECKPOINT_DIRECTORY / "middle.json"
    report_path = c4f17.CHECKPOINT_DIRECTORY / "middle_summary.json"
    arrays = c4f17._load(checkpoint_path)
    progress = c4f17._read(progress_path)
    middle = c4f17._read(report_path)
    manifest = c4f17._authorization()
    gates = manifest["prospective_dynamic_gates"]
    if (
        int(progress["steps_completed"]) != 39
        or middle["steps"] != 39
        or middle["passed_method_and_single_layout_coordinate_gates"]
        or arrays["times"].size != 40
        or (c4f17.CHECKPOINT_DIRECTORY / "fine.npz").exists()
        or progress["source_identity"].get(c4f17.THIS_RUNNER)
        != c4f17._sha(c4f17.ROOT / c4f17.THIS_RUNNER)
    ):
        raise RuntimeError("c4f17 middle-only rejection evidence changed")

    checks = {
        "initial_state_lift_Q3": bool(
            middle["initial_Q3_defect"]
            <= gates["maximum_initial_state_lift_Q3_defect"]
        ),
        "initial_orthogonality": bool(
            middle["initial_orthogonality_defect"] <= 1.0e-10
        ),
        "dual_biorthogonality": bool(
            middle["dual_biorthogonality_defect"]
            <= gates["maximum_dual_biorthogonality_defect"]
        ),
        "dual_normalized_slow_annihilation": bool(
            middle["dual_normalized_slow_annihilation_defect"]
            <= gates["maximum_normalized_slow_lift_annihilation_defect"]
        ),
        "step_matrix_JVP": bool(
            middle["maximum_JVP_defect"]
            <= gates["maximum_step_matrix_JVP_relative_defect"]
        ),
        "block_linear_solve": bool(
            middle["maximum_linear_solve_defect"]
            <= gates["maximum_block_linear_solve_relative_defect"]
        ),
        "face36_output_map": bool(
            middle["maximum_face36_output_map_defect"]
            <= gates["maximum_face36_output_map_relative_defect"]
        ),
        "Q3_leakage": bool(
            middle["maximum_Q3_leakage"] <= gates["maximum_Q3_leakage"]
        ),
        "outgoing_excision": bool(
            middle["maximum_incoming_characteristics"]
            == gates["incoming_excision_characteristics"]
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed != [
        "dual_normalized_slow_annihilation",
        "face36_output_map",
    ]:
        raise RuntimeError(f"c4f17 unexpected failure set: {failed}")
    return middle, arrays, {
        "gates": gates,
        "checks": checks,
        "failed": failed,
        "execution_source_identity": progress["source_identity"],
    }


def main() -> None:
    middle, arrays, audit = _result()
    c4f17.CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    recovery = {
        "schema_version": 1,
        "work_package": RECOVERY_WORK_PACKAGE,
        "classification": (
            "six_mode_middle_numerical_audit_recovery_manifest_frozen_"
            "analysis_only_recovery_authorized"
        ),
        "definitions_only": True,
        "uses_saved_c4f17_middle_state_direction_history": True,
        "reruns_middle_propagation": False,
        "runs_fine_propagation": False,
        "stable_dual_audit": {
            "replace_normal_equations_in_audit_only": True,
            "methods": ["reduced_QR", "thin_SVD"],
            "compare_with_c4f15_endpoint_metrics": True,
            "maximum_dual_biorthogonality_defect": 1.0e-10,
            "maximum_normalized_slow_lift_annihilation_defect": 1.0e-10,
            "tolerance_relaxation_forbidden": True,
        },
        "face36_directional_JVP_plateau": {
            "times_seconds": [0.005, 0.0054, 0.010, 0.016, 0.020],
            "all_six_saved_directions": True,
            "relative_steps": [
                5.0e-5,
                1.0e-4,
                2.0e-4,
                5.0e-4,
                1.0e-3,
                2.0e-3,
            ],
            "central_and_five_point_references": True,
            "maximum_relative_defect": 1.0e-8,
            "require_visible_plateau_or_U_shaped_curve": True,
            "tolerance_relaxation_forbidden": True,
        },
        "decision": {
            "both_recovery_audits_pass": (
                "reclassify_saved_middle_history_without_repropagation_and_"
                "authorize_definitions_only_fine_replay_manifest"
            ),
            "stable_dual_fails": "reject_six_mode_Petrov_coordinate",
            "directional_JVP_has_no_plateau": (
                "audit_face36_derivative_or_scaling_before_any_fine_work"
            ),
        },
        "expected_wall_hours": [0.5, 1.5],
        "hard_stops": [
            "do_not_relax_the_frozen_tolerances",
            "do_not_rerun_middle_propagation",
            "do_not_start_fine_propagation",
            "do_not_apply_a_fixed_Q_reaction",
            "do_not_start_a_nonlinear_microburst",
            "do_not_start_50ms_or_reduced_slow_evolution",
        ],
        "fixed_Q_micro_solver_authorized": False,
        "nonlinear_retained_mode_pilot_authorized": False,
        "fifty_ms_propagation_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": (
            "WP10c9d6c7c3b5c4f18_analysis_only_stable_dual_and_"
            "face36_directional_JVP_recovery"
        ),
    }
    summary = {
        "schema_version": c4f17.SCHEMA_VERSION,
        "work_package": c4f17.WORK_PACKAGE,
        "classification": (
            "face36_six_mode_middle_dynamic_coordinate_preflight_rejected_"
            "fine_blocked_numerical_audit_recovery_manifest_authorized"
        ),
        "passed": False,
        "middle_completed": True,
        "fine_executed": False,
        "middle": middle,
        "individual_gate_results": audit["checks"],
        "failed_gates": audit["failed"],
        "middle_state_history_scientifically_interpreted": False,
        "new_nonlinear_trajectory": False,
        "fixed_Q_reaction_applied": False,
        "fixed_Q_micro_solver_authorized": False,
        "nonlinear_retained_mode_pilot_authorized": False,
        "fifty_ms_propagation_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "physical_failure_detected": False,
        "guard_complement_retained": True,
        "raw_face48_export_rejection_preserved": True,
        "authorized_next": recovery["authorized_next"],
    }
    c4f17._save(
        c4f17.DECISIVE_ARRAYS,
        times=arrays["times"],
        middle_state_directions=arrays["state_directions"],
        middle_face36_outputs=arrays["face36_outputs"],
        middle_amplitude_transitions=arrays["amplitude_transitions"],
        middle_Q3_leakage=arrays["Q3_leakage"],
        middle_guard_mapped=arrays["guard_mapped"],
        middle_guard_height_history=arrays["guard_height_history"],
        middle_JVP_defects=arrays["JVP_defects"],
        middle_linear_solve_defects=arrays["linear_solve_defects"],
        middle_face36_output_map_defects=arrays["face36_output_map_defects"],
    )
    c4f17._write(
        c4f17.CONFIG_PATH,
        {
            "schema_version": c4f17.SCHEMA_VERSION,
            "work_package": c4f17.WORK_PACKAGE,
            "mode_dimension": c4f17.MODE_DIMENSION,
            "leading_dimension": c4f17.LEADING_DIMENSION,
            "audit_targets_seconds": list(c4f17.AUDIT_TARGETS_SECONDS),
            "prospective_gates": audit["gates"],
        },
    )
    c4f17._write(RECOVERY_MANIFEST_PATH, recovery)
    c4f17._write(c4f17.SUMMARY_PATH, summary)
    c4f17.REPORT_PATH.write_text(
        "# Face-36 six-mode middle dynamic-coordinate replay\n\n"
        f"Classification: `{summary['classification']}`.\n\n"
        "The analysis-only middle replay completed all 39 committed 5--20 ms "
        "steps in 1.865 hours. Fine was not started because the frozen "
        "middle fail-fast gate did not pass.\n\n"
        "The complete BDF step JVP (`5.85e-11`), block solve (`3.31e-16`), "
        "Q3 leakage (`0.004143` versus `0.10`), component closure "
        "(`6.56e-17`), initial Q3 lift (`3.41e-15`), dual "
        "biorthogonality (`1.35e-11`), and outgoing excision all pass.\n\n"
        "Two numerical-audit gates fail: the normal-equation Petrov dual has "
        "normalized slow-lift annihilation `3.84e-10 > 1e-10`, and the "
        "selected face-36 directional finite-difference check gives "
        "`1.37e-7 > 1e-8`. This is not a physical truth-model failure and the "
        "saved dynamic histories are not interpreted before those audits are "
        "recovered.\n\n"
        "The next authorized work is analysis-only: reconstruct the dual with "
        "stable reduced-QR/thin-SVD algebra and establish a predeclared "
        "central/five-point face-36 JVP step plateau on all six saved "
        "directions. No tolerance may be relaxed, middle propagation must not "
        "be repeated, and fine remains blocked.\n",
        encoding="utf-8",
    )
    c4f17._write(
        c4f17.PROVENANCE_PATH,
        {
            "schema_version": c4f17.SCHEMA_VERSION,
            "source_parent_commit": c4f17._read(c4f17.CANONICAL_SUMMARY)[
                "latest_source_parent_commit"
            ],
            "packaging_execution_commit": c4f17._git("rev-parse", "HEAD"),
            "packaging_execution_tree": c4f17._git("rev-parse", "HEAD^{tree}"),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "execution_source_identity": audit["execution_source_identity"],
            "checkpoint_sha256": c4f17._sha(
                c4f17.CHECKPOINT_DIRECTORY / "middle.npz"
            ),
            "checkpoint_summary_sha256": c4f17._sha(
                c4f17.CHECKPOINT_DIRECTORY / "middle_summary.json"
            ),
            "c4f16_manifest_sha256": c4f17._sha(c4f17.c4f16.MANIFEST_PATH),
        },
    )
    files = (
        c4f17.CONFIG_PATH,
        c4f17.DECISIVE_ARRAYS,
        RECOVERY_MANIFEST_PATH,
        c4f17.SUMMARY_PATH,
        c4f17.PROVENANCE_PATH,
    )
    (c4f17.CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{c4f17._sha(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )
    c4f17._catalog(summary)
    print(json.dumps(c4f17._plain(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
