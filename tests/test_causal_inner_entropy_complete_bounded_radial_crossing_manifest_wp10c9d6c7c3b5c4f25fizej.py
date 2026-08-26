from __future__ import annotations

import hashlib

import run_causal_inner_entropy_complete_bounded_radial_crossing_manifest_wp10c9d6c7c3b5c4f25fizej as target


def test_relaxation_parent_is_locked() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["classification"] == target.parent.PASS_CLASSIFICATION
    assert validated["summary"]["hydrostatic_relaxation_limit_certified"]
    assert validated["metrics"]["first_failure"] is None


def test_trajectory_budget_is_exact_and_bounded() -> None:
    contract = target._contract()
    assert contract["time_integrator"]["method"] == "explicit_SSPRK2_in_seven_primitive_chart"
    assert contract["time_integrator"]["timestep_seconds"] == 6.25e-5
    assert contract["time_integrator"]["accepted_steps"] == 4
    assert contract["time_integrator"]["horizon_seconds"] == 2.5e-4
    assert contract["claim_boundary"]["maximum_new_trajectory_steps"] == 4


def test_rejected_legacy_state_is_not_a_seed() -> None:
    contract = target._contract()
    assert contract["seed"]["profile"] == "accepted_terminal_base_charts5"
    assert contract["seed"]["rejected_legacy_candidate_not_used"]
    assert contract["radial_operator"]["accepted_history_only"]


def test_canonical_package_closes_if_present() -> None:
    directory = target.CANONICAL_DIRECTORY
    if not directory.exists(): return
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
    summary = target._utils()._read_json(directory / "summary.json")
    assert summary["definitions_only"]
    assert summary["maximum_new_trajectory_steps"] == 4
    assert summary["new_trajectory_steps"] == 0
    assert not summary["fixed_Q_invariant_object_authorized"]
