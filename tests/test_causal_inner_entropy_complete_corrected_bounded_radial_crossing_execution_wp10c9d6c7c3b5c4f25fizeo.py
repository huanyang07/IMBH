from __future__ import annotations

import hashlib

import run_causal_inner_entropy_complete_corrected_bounded_radial_crossing_execution_wp10c9d6c7c3b5c4f25fizeo as target


def test_manifest_is_hash_locked() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["classification"] == target.parent.CLASSIFICATION
    assert validated["summary"]["maximum_new_trajectory_steps"] == 32
    assert validated["contract"]["time_integrator"]["replay_checkpoint_step"] == 28


def test_exact_balance_helper_closes_for_zero_change() -> None:
    import numpy as np
    assert target._relative(np.zeros(6), np.zeros(6), np.zeros(6)) == 0.0


def test_execution_preserves_all_claim_boundaries() -> None:
    contract = target.parent._contract()
    assert contract["claim_boundary"]["corrected_crossing_execution_authorized"]
    assert not contract["claim_boundary"]["fixed_Q_invariant_object_authorized"]
    assert not contract["claim_boundary"]["complete_cycle_execution_authorized"]


def test_canonical_package_closes_if_present() -> None:
    directory = target.CANONICAL_DIRECTORY
    if not directory.exists(): return
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
    summary = target._utils()._read_json(directory / "summary.json")
    metrics = target._utils()._read_json(directory / "execution_metrics.json")
    assert summary["accepted_new_steps"] == metrics["accepted_new_steps"]
    if summary["passed"]:
        assert summary["accepted_new_steps"] == 32
        assert summary["crossed_old_rejected_time"]
        assert summary["first_endpoint_matches_diagnostic_bitwise"]
        assert summary["suffix_replay_bitwise"]
