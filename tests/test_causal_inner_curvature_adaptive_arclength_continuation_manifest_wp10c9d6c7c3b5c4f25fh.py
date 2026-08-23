from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_curvature_adaptive_arclength_continuation_manifest_wp10c9d6c7c3b5c4f25fh as target  # noqa: E402


def test_parent_selects_autonomous_wide_transport() -> None:
    lock = target._validate_parent(require_clean=False)
    assert lock["diagnosis_classification"] == target.diagnosis.CLASSIFICATION
    assert lock["execution_classification"] == target.diagnosis.parent.BUDGET_CLASSIFICATION


def test_endpoint_acquisition_is_complete_and_fail_closed() -> None:
    contract = target._execution_contract(target._cost_projection())
    assert contract["truth_system"]["autonomous"]
    assert contract["truth_system"]["external_clock_or_phase"] == "forbidden"
    assert contract["truth_system"]["fixed_Q_rate_or_reaction"] == "forbidden"
    assert contract["segment_validation"]["endpoint_exact_before_propagation"]
    assert contract["segment_validation"]["failed_candidate_is_never_propagated"]
    assert contract["step_policy"]["retry_from_last_accepted_endpoint"]


def test_variable_step_ab2_uses_only_exact_autonomous_rates() -> None:
    formula = target._variable_step_ab2_formula()
    assert "h^2/(2*h_previous)" in formula["definition"]
    assert formula["all_rates"].startswith("exact original unconstrained")
    assert not formula["fixed_Q_rate_used"]
    assert not formula["external_phase_used"]


def test_budget_has_blind_and_rejection_reserve() -> None:
    cost = target._cost_projection()
    assert cost["cost_gate_passed"]
    assert cost["no_rejection_exact_call_count"] == 271
    assert cost["rejection_call_reserve"] == 17
    assert cost["maximum_horizon_at_two_milliseconds_seconds"] == 0.432


def test_endpoint_predictor_is_prevalidated_through_two_milliseconds() -> None:
    backtest = target._predictor_backtest()
    assert backtest["passed"]
    assert backtest["two_millisecond_endpoint_proposal_prevalidated"]
    assert backtest["four_millisecond_predictor_is_diagnostic_only"]
    assert (
        backtest["records"]["steady_2ms"]["maximum_relative_endpoint_defect"]
        < target.MAXIMUM_PREVALIDATED_PREDICTOR_DEFECT
    )
    assert (
        backtest["records"]["steady_4ms"]["maximum_relative_endpoint_defect"]
        > target.MAXIMUM_PREVALIDATED_PREDICTOR_DEFECT
    )


def test_post_cycle_authorizes_refinement_not_slow_evolution() -> None:
    contract = target._execution_contract(target._cost_projection())
    assert "matched-path refinement" in contract["post_cycle_authorization"]
    assert "no slow closure" in contract["post_cycle_authorization"]


def test_canonical_package_when_present() -> None:
    if not target.CANONICAL_DIRECTORY.exists():
        return
    helper = target._helper()
    helper._validate_checksums(target.CANONICAL_DIRECTORY)
    summary = helper._read(target.CANONICAL_DIRECTORY / "summary.json")
    contract = helper._read(
        target.CANONICAL_DIRECTORY / "continuation_execution_contract.json"
    )
    assert summary["passed"]
    assert summary["endpoint_acquisition_contract_complete"]
    assert summary["curvature_adaptive_continuation_execution_authorized"]
    assert not summary["curvature_adaptive_continuation_executed"]
    assert not summary["reduced_slow_evolution_authorized"]
    assert contract["authorized_execution"] == target.AUTHORIZED_NEXT
