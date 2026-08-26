from __future__ import annotations

import hashlib

import run_causal_inner_entropy_complete_fixed_q_adaptive_trust_diagnosis_manifest_wp10c9d6c7c3b5c4f25fizex as target


def test_root_rejection_is_locked() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["classification"] == target.parent.FAIL_CLASSIFICATION
    assert not validated["summary"]["root_exists"]
    assert validated["metrics"]["accepted_nonlinear_corrections"] == 2


def test_trust_radii_are_recomputed_not_line_searched() -> None:
    candidates = target._contract()["adaptive_trust_candidates"]
    assert candidates["ordered_trust_radii"] == [0.02, 0.01, 0.005, 0.0025, 0.001]
    assert candidates["new_bounded_TRF_direction_at_each_radius"]
    assert candidates["line_searching_the_old_radius_0p25_direction_is_not_used"]


def test_no_root_or_slow_execution_is_authorized() -> None:
    claim = target._contract()["claim_boundary"]
    assert claim["diagnosis_authorized"]
    assert not claim["root_retry_authorized"]
    assert not claim["complete_cycle_execution_authorized"]


def test_canonical_package_closes_if_present() -> None:
    directory = target.CANONICAL_DIRECTORY
    if not directory.exists(): return
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
    summary = target._utils()._read_json(directory / "summary.json")
    assert summary["primary_root_rejection_preserved"]
    assert summary["adaptive_trust_diagnosis_authorized"]
    assert not summary["root_retry_authorized"]
