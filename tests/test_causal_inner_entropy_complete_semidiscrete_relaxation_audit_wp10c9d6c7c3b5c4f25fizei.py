from __future__ import annotations

import hashlib

import run_causal_inner_entropy_complete_semidiscrete_relaxation_audit_wp10c9d6c7c3b5c4f25fizei as target


def test_relaxation_manifest_is_locked() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["classification"] == target.parent.CLASSIFICATION
    assert validated["summary"]["nonpropagating_relaxation_audit_authorized"]
    assert not validated["summary"]["radial_boundary_implementation_authorized"]


def test_fast_limit_ladder_is_prospective() -> None:
    assert target.FAST_MULTIPLIERS == (1.0, 2.0, 4.0, 8.0)
    gates = target.parent._contract()["binding_gates"]
    assert gates["minimum_fast_relaxation_observed_order"] == 0.8
    assert gates["all_cases_and_all_gates_required"]


def test_canonical_package_closes_if_present() -> None:
    directory = target.CANONICAL_DIRECTORY
    if not directory.exists():
        return
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
    summary = target._utils()._read_json(directory / "summary.json")
    assert summary["new_trajectory_steps"] == 0
    assert not summary["radial_boundary_implementation_authorized"]
    assert not summary["bounded_crossing_trajectory_authorized"]
