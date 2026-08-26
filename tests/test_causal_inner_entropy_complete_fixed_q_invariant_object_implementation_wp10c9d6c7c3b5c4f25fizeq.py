from __future__ import annotations

import hashlib

import run_causal_inner_entropy_complete_fixed_q_invariant_object_implementation_wp10c9d6c7c3b5c4f25fizeq as target


def test_manifest_and_sources_are_locked() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["classification"] == target.parent.CLASSIFICATION
    assert validated["summary"]["implementation_authorized"]
    assert not validated["summary"]["nonlinear_root_execution_authorized"]


def test_projected_field_call_budget_matches_coloring_and_audits() -> None:
    assert target.EXPECTED_FIELD_CALLS == 21
    contract = target.parent._contract()
    assert contract["sparse_derivative"]["forward_colored_residual_evaluations_per_assembly"] == 12
    assert contract["sparse_derivative"]["independent_central_directional_audits"] == 4


def test_implementation_cannot_run_a_root() -> None:
    contract = target.parent._contract()
    assert not contract["claim_boundary"]["nonlinear_root_execution_authorized"]
    assert not contract["claim_boundary"]["slow_flux_atlas_authorized"]


def test_canonical_package_closes_if_present() -> None:
    directory = target.CANONICAL_DIRECTORY
    if not directory.exists(): return
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
    summary = target._utils()._read_json(directory / "summary.json")
    metrics = target._utils()._read_json(directory / "implementation_metrics.json")
    assert summary["new_nonlinear_roots"] == 0
    assert summary["propagated_states"] == 0
    if summary["passed"]:
        assert metrics["projected_field_calls"] == target.EXPECTED_FIELD_CALLS
        assert summary["projected_fast_field_certified"]
        assert summary["colored_fast_Jacobian_certified"]
