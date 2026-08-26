from __future__ import annotations

import hashlib

import run_causal_inner_entropy_complete_radial_substep_diagnosis_execution_wp10c9d6c7c3b5c4f25fizem as target


def test_manifest_is_hash_locked_and_authorizes_only_diagnosis() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["classification"] == target.parent.CLASSIFICATION
    assert validated["summary"]["maximum_new_trajectory_steps"] == 0
    assert validated["contract"]["experiment"]["independent_one_step_trials"]


def test_selection_rule_requires_original_gate_and_headroom() -> None:
    contract = target.parent._contract()
    assert contract["binding_gates"]["maximum_scaled_chart_change_per_step"] == 0.05
    assert contract["binding_gates"]["selected_substep_maximum_scaled_chart_change"] == 0.03
    assert contract["selection"]["no_retrospective_tolerance_change"]


def test_no_trial_endpoint_can_become_history() -> None:
    contract = target.parent._contract()
    assert not contract["experiment"]["propagate_trial_endpoint"]
    assert contract["claim_boundary"]["maximum_new_trajectory_steps"] == 0


def test_canonical_package_closes_if_present() -> None:
    directory = target.CANONICAL_DIRECTORY
    if not directory.exists():
        return
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
    summary = target._utils()._read_json(directory / "summary.json")
    metrics = target._utils()._read_json(directory / "execution_metrics.json")
    assert summary["new_trajectory_steps"] == 0
    assert not summary["trial_endpoints_propagated"]
    assert summary["selected_timestep_seconds"] == metrics["selected_timestep_seconds"]
