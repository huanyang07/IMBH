from __future__ import annotations

import hashlib

import run_causal_inner_entropy_complete_radial_substep_diagnosis_manifest_wp10c9d6c7c3b5c4f25fizel as target


def test_rejected_crossing_is_hash_locked() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["classification"] == target.parent.PHYSICAL_FAILURE
    assert validated["summary"]["accepted_new_steps"] == 0
    assert validated["metrics"]["first_failure"]["failure_reasons"] == ["physical:chart_change"]


def test_trials_are_independent_and_nonpropagating() -> None:
    contract = target._contract()
    assert contract["experiment"]["timesteps_seconds"] == [3.125e-5, 1.5625e-5, 7.8125e-6]
    assert contract["experiment"]["independent_one_step_trials"]
    assert not contract["experiment"]["propagate_trial_endpoint"]
    assert contract["claim_boundary"]["maximum_new_trajectory_steps"] == 0


def test_original_gate_is_unchanged_and_selection_has_headroom() -> None:
    gates = target._contract()["binding_gates"]
    assert gates["maximum_scaled_chart_change_per_step"] == 0.05
    assert gates["selected_substep_maximum_scaled_chart_change"] == 0.03
    assert gates["minimum_adjacent_chart_change_order"] == 0.8
    assert gates["maximum_adjacent_chart_change_order"] == 1.2


def test_canonical_package_closes_if_present() -> None:
    directory = target.CANONICAL_DIRECTORY
    if not directory.exists():
        return
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
    summary = target._utils()._read_json(directory / "summary.json")
    assert summary["definitions_only"]
    assert summary["maximum_new_trajectory_steps"] == 0
    assert not summary["bounded_crossing_retry_authorized"]
