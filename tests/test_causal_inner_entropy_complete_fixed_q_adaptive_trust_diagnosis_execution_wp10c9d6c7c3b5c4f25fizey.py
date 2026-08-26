from __future__ import annotations

import hashlib

import run_causal_inner_entropy_complete_fixed_q_adaptive_trust_diagnosis_execution_wp10c9d6c7c3b5c4f25fizey as target


def test_diagnosis_manifest_is_locked() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["classification"] == target.parent.CLASSIFICATION
    assert validated["summary"]["adaptive_trust_diagnosis_authorized"]
    assert not validated["summary"]["root_retry_authorized"]


def test_execution_budget_and_trust_ladder_are_prospective() -> None:
    contract = target.parent._contract()
    linear = contract["fresh_equation_linearization"]
    candidates = contract["adaptive_trust_candidates"]
    assert linear["independent_central_JVP_directions"] == 2
    assert candidates["ordered_trust_radii"] == [0.02, 0.01, 0.005, 0.0025, 0.001]
    assert candidates["maximum_full_physical_candidate_evaluations"] == 5


def test_candidate_cannot_be_a_root_or_propagated() -> None:
    progress = target.parent._contract()["useful_progress"]
    assert progress["selected_candidate_is_not_a_root_and_must_not_be_propagated"]


def test_canonical_package_closes_if_present() -> None:
    directory = target.CANONICAL_DIRECTORY
    if not directory.exists():
        return
    for line in (directory / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
    summary = target._utils()._read_json(directory / "summary.json")
    metrics = target._utils()._read_json(directory / "diagnosis_metrics.json")
    assert summary["new_nonlinear_roots"] == 0
    assert summary["propagated_states"] == 0
    assert metrics["physical_field_calls"] <= metrics["maximum_physical_field_calls"]
