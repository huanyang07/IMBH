from __future__ import annotations

import run_causal_inner_hybrid_branch_transition_atlas_manifest_wp10c9d6c7c3b5c4f25db as f25db


def test_slow_branch_schedule_is_sealed_and_pathwise() -> None:
    contract = f25db._branch_contract()
    assert contract["branches"] == f25db.BRANCHES
    assert len(contract["anchor_schedule"]) == 12
    assert len(contract["training_ids"]) == 8
    assert len(contract["sealed_heldout_ids"]) == 4
    assert sum(item["fine_layout"] for item in contract["anchor_schedule"]) == 2
    assert contract["stable_memory"]["orders"] == f25db.MEMORY_ORDERS
    assert contract["stable_memory"]["freeze_before_opening_heldout"]
    assert contract["local_conservative_closure"]["cell_derivative_regression_forbidden"]


def test_fast_transition_schedule_is_not_an_online_ode() -> None:
    contract = f25db._transition_contract()
    assert contract["directions"] == f25db.TRANSITIONS
    assert len(contract["transition_schedule"]) == 8
    assert sum(
        item["role"] == "training" for item in contract["transition_schedule"]
    ) == 6
    assert sum(
        item["role"] == "heldout" for item in contract["transition_schedule"]
    ) == 2
    assert contract["online_transition_ODE_forbidden"]
    assert contract["online_fast_microsteps"] == 0
    assert contract["truth_budget"][
        "first_execution_authorizes_only_one_direction_one_transition_pilot"
    ]


def test_campaign_is_fail_fast_and_no_truth() -> None:
    branch = f25db._branch_contract()
    transition = f25db._transition_contract()
    campaign = f25db._campaign_contract(branch, transition)
    assert campaign["next_package"]["work_package"] == f25db.AUTHORIZED_NEXT
    assert campaign["next_package"]["new_exact_rate_calls"] == 0
    assert campaign["next_package"]["new_nonlinear_roots"] == 0
    assert campaign["runtime_contract"]["truth_and_470_field_online_calls"] == 0
    assert campaign["runtime_contract"][
        "fast_stability_scale_present_in_online_step_restriction"
    ] is False
    assert not campaign["authorization_boundaries"]["truth_campaign_authorized"]
    assert all(f25db._checks(branch, transition, campaign).values())


def test_canonical_manifest_if_present() -> None:
    if not f25db.CANONICAL_DIRECTORY.exists():
        return
    f25db._checksums(f25db.CANONICAL_DIRECTORY)
    summary = f25db._read(f25db.CANONICAL_DIRECTORY / "summary.json")
    checks = f25db._read(f25db.CANONICAL_DIRECTORY / "checks.json")
    branch = f25db._read(f25db.CANONICAL_DIRECTORY / "slow_branch_contract.json")
    transition = f25db._read(
        f25db.CANONICAL_DIRECTORY / "fast_transition_contract.json"
    )
    campaign = f25db._read(f25db.CANONICAL_DIRECTORY / "campaign_contract.json")
    assert summary["passed"]
    assert summary["classification"] == f25db.CLASSIFICATION
    assert summary["authorized_next"] == f25db.AUTHORIZED_NEXT
    assert summary["branch_and_transition_datasets_separated"]
    assert not summary["truth_campaign_authorized"]
    assert not summary["online_reduced_solver_implementation_authorized"]
    assert all(checks["checks"].values())
    assert branch["local_conservative_closure"][
        "global_M_J_E_telescoping_exact_by_construction"
    ]
    assert transition["online_fast_microsteps"] == 0
    assert not campaign["authorization_boundaries"]["predictive_cycle_authorized"]
