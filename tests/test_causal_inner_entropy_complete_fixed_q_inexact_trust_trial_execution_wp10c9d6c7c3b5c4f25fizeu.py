from __future__ import annotations

import hashlib

import run_causal_inner_entropy_complete_fixed_q_inexact_trust_trial_execution_wp10c9d6c7c3b5c4f25fizeu as target


def test_recovery_manifest_is_locked() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["classification"] == target.parent.CLASSIFICATION
    assert validated["summary"]["one_nonpropagating_trial_authorized"]
    assert not validated["summary"]["primary_root_execution_authorized"]


def test_line_search_is_finite_and_nonpropagating() -> None:
    trial = target.parent._contract()["nonpropagating_physical_trial"]
    assert len(trial["ordered_step_factors"]) == 8
    assert trial["maximum_physical_field_calls"] == 8
    assert trial["accepted_trial_is_not_a_root"]
    assert trial["accepted_trial_must_not_be_propagated"]


def test_canonical_package_closes_if_present() -> None:
    directory = target.CANONICAL_DIRECTORY
    if not directory.exists(): return
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
    summary = target._utils()._read_json(directory / "summary.json")
    metrics = target._utils()._read_json(directory / "trial_metrics.json")
    assert summary["new_nonlinear_roots"] == 0
    assert summary["propagated_states"] == 0
    assert metrics["physical_field_calls"] <= metrics["maximum_physical_field_calls"]
