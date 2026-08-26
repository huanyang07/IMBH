from __future__ import annotations

import hashlib

import run_causal_inner_entropy_complete_fixed_q_primary_inexact_newton_root_execution_wp10c9d6c7c3b5c4f25fizew as target


def test_primary_root_manifest_and_sources_are_locked() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["classification"] == target.parent.CLASSIFICATION
    assert validated["summary"]["primary_root_execution_authorized"]
    assert not validated["summary"]["heldout_root_execution_authorized"]


def test_inexact_solver_budget_is_finite() -> None:
    contract = target.parent._contract()
    solver = contract["inexact_Newton_solver"]
    assert solver["maximum_total_nonlinear_corrections"] == 12
    assert solver["maximum_solver_colored_Jacobian_assemblies"] == 2
    assert len(solver["ordered_line_search_factors"]) == 8


def test_post_root_audit_cannot_change_root() -> None:
    audit = target.parent._contract()["post_root_certification"]
    assert audit["certification_assembly_cannot_change_the_root"]
    assert audit["independent_direct_physical_rate_central_JVP_directions"] == 4


def test_canonical_package_closes_if_present() -> None:
    directory = target.CANONICAL_DIRECTORY
    if not directory.exists(): return
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
    summary = target._utils()._read_json(directory / "summary.json")
    metrics = target._utils()._read_json(directory / "root_metrics.json")
    assert summary["propagated_states"] == 0
    assert metrics["solver_exact_colored_assemblies"] <= 2
    if summary["passed"]:
        assert summary["root_exists"]
        assert summary["normally_attracting"]
