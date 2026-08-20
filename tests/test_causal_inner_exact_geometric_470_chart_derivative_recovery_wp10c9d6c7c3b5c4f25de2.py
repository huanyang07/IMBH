from __future__ import annotations

import numpy as np

import run_causal_inner_exact_geometric_470_chart_derivative_recovery_wp10c9d6c7c3b5c4f25de2 as f25de2


def test_manifest_is_committed_and_authorizes_only_this_audit() -> None:
    frozen = f25de2._validate_manifest(require_clean=False)
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["authorized_next"] == f25de2.WORK_PACKAGE
    assert not frozen["summary"]["branch_root_execution_authorized"]
    assert frozen["contract"]["preserved_negative_certificate"][
        "remains_failed"
    ]


def test_roundoff_statistics_identify_inverse_step_scaling() -> None:
    steps = np.asarray(f25de2.manifest.ROUND_OFF_STEPS)
    defects = 2.0e-10 / steps
    statistics = f25de2._roundoff_statistics(steps, defects)
    assert np.isclose(statistics["loglog_slope"], -1.0)
    assert np.isclose(statistics["loglog_R_squared"], 1.0)
    assert statistics["h_times_defect_coefficient_of_variation"] < 1.0e-12


def test_checks_are_fail_closed_and_keep_truth_budget_zero() -> None:
    steps = np.asarray(f25de2.manifest.ROUND_OFF_STEPS)
    metrics = {
        "maximum_algebraic_relative_defect": 1.0e-12,
        "maximum_common_scale_relative_defect": 5.0e-8,
        "minimum_common_scale_signal_norm": 1.0e-7,
        "roundoff_statistics": f25de2._roundoff_statistics(
            steps, 2.0e-10 / steps
        ),
        "original_defect_reproduction_relative": 0.0,
        "parent_non_derivative_checks_preserved": True,
        "new_coordinate_evaluations": 26,
        "new_coordinate_jacobian_assemblies": 0,
        "new_coordinate_retractions": 0,
        "new_exact_fixed_Q_rate_evaluations": 0,
        "new_complete_generator_assemblies": 0,
        "new_intrinsic_hidden_roots": 0,
        "propagated_states": 0,
        "sealed_16ms_truth_calls": 0,
    }
    assert all(f25de2._checks(metrics).values())
    metrics["new_exact_fixed_Q_rate_evaluations"] = 1
    assert not f25de2._checks(metrics)["rate_budget"]
    metrics["new_exact_fixed_Q_rate_evaluations"] = 0
    metrics["maximum_common_scale_relative_defect"] = 2.0e-6
    assert not f25de2._checks(metrics)["all_eight_common_scale_directions"]


def test_canonical_audit_if_present() -> None:
    if not f25de2.CANONICAL_DIRECTORY.exists():
        return
    f25de2._checksums(f25de2.CANONICAL_DIRECTORY)
    summary = f25de2._read(f25de2.CANONICAL_DIRECTORY / "summary.json")
    payload = f25de2._read(
        f25de2.CANONICAL_DIRECTORY / "derivative_recovery_metrics.json"
    )
    assert summary["parent_negative_certificate_preserved"]
    assert not summary["parent_chart_preflight_reclassified"]
    assert not summary["branch_root_execution_authorized"]
    assert not summary["sealed_16ms_opened"]
    assert payload["metrics"]["new_exact_fixed_Q_rate_evaluations"] == 0
    assert payload["metrics"]["new_intrinsic_hidden_roots"] == 0
    if summary["passed"]:
        assert all(payload["checks"].values())
        assert summary["authorized_next"] == f25de2.AUTHORIZED_NEXT
    else:
        assert summary["authorized_next"] is None
