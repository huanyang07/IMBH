from __future__ import annotations

import run_causal_inner_exact_geometric_470_chart_derivative_recovery_manifest_wp10c9d6c7c3b5c4f25de1 as f25de1


def test_parent_failure_is_preserved_and_narrow() -> None:
    frozen = f25de1._validate_parent(require_clean=False)
    checks = frozen["metrics"]["checks"]
    assert not frozen["summary"]["passed"]
    assert not checks["implicit_derivative"]
    assert all(
        passed for name, passed in checks.items() if name != "implicit_derivative"
    )


def test_scale_aware_audit_is_frozen_before_binding_execution() -> None:
    frozen = f25de1._validate_parent(require_clean=False)
    contract = f25de1._contract(frozen["metrics"])
    execution = contract["prospective_execution"]
    assert execution["common_scale_step"] == 3.0e-3
    assert execution["common_scale_direction_indices"] == list(range(8))
    assert execution["roundoff_ladder"] == [
        3.0e-3,
        1.0e-3,
        3.0e-4,
        1.0e-4,
        3.0e-5,
        1.0e-5,
    ]
    assert execution["roundoff_direction"] == {
        "direction_index": 6,
        "family": "macro",
        "source_index": 53,
    }


def test_roundoff_signature_and_absolute_accuracy_are_both_binding() -> None:
    frozen = f25de1._validate_parent(require_clean=False)
    gates = f25de1._contract(frozen["metrics"])["binding_gates"]
    assert gates["all_eight_algebraic_relative_defects_max"] == 1.0e-10
    assert gates["all_eight_common_scale_relative_defects_max"] == 1.0e-6
    assert gates["best_common_scale_relative_defect_max"] == 1.0e-7
    assert gates["roundoff_loglog_slope_min"] == -1.10
    assert gates["roundoff_loglog_slope_max"] == -0.90
    assert gates["roundoff_loglog_R_squared_min"] == 0.99
    assert (
        gates["roundoff_h_times_defect_coefficient_of_variation_max"] == 0.10
    )


def test_truth_and_authorization_boundaries_fail_closed() -> None:
    frozen = f25de1._validate_parent(require_clean=False)
    contract = f25de1._contract(frozen["metrics"])
    budgets = contract["prospective_execution"]["budgets"]
    assert budgets["new_exact_fixed_Q_rate_evaluations_equal"] == 0
    assert budgets["new_complete_generator_assemblies_equal"] == 0
    assert budgets["new_intrinsic_hidden_roots_equal"] == 0
    assert budgets["propagated_states_equal"] == 0
    assert budgets["sealed_16ms_truth_calls_equal"] == 0
    assert not contract["decision"]["pass"]["authorizes_root_execution_directly"]
    assert not contract["authorization_boundaries"][
        "branch_root_execution_authorized"
    ]
    assert not contract["authorization_boundaries"][
        "reduced_slow_evolution_authorized"
    ]


def test_canonical_manifest_if_present() -> None:
    if not f25de1.CANONICAL_DIRECTORY.exists():
        return
    f25de1._checksums(f25de1.CANONICAL_DIRECTORY)
    summary = f25de1._read(f25de1.CANONICAL_DIRECTORY / "summary.json")
    contract = f25de1._read(
        f25de1.CANONICAL_DIRECTORY / "derivative_recovery_contract.json"
    )
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["parent_negative_certificate_preserved"]
    assert summary["authorized_next"] == f25de1.AUTHORIZED_NEXT
    assert not summary["branch_root_execution_authorized"]
    assert not summary["sealed_16ms_opened"]
    assert contract["preserved_negative_certificate"]["remains_failed"]
