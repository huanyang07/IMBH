from __future__ import annotations

import hashlib

import run_causal_inner_entropy_complete_fixed_q_inexact_trust_recovery_manifest_wp10c9d6c7c3b5c4f25fizet as target


def test_only_backend_status_failed() -> None:
    validated = target._validate_parent(require_clean=False)
    failed = [name for name, passed in validated["metrics"]["checks"].items() if not passed]
    assert failed == ["bounded_linear_solver"]
    assert validated["summary"]["new_nonlinear_roots"] == 0


def test_saved_direction_meets_inexact_forcing_contract() -> None:
    diagnostics = target._saved_direction_diagnostics()
    limits = target._contract()["inexact_Newton_direction"]
    assert diagnostics["forcing_two_norm"] <= limits["maximum_forcing_two_norm"]
    assert diagnostics["forcing_infinity_norm"] <= limits["maximum_forcing_infinity_norm"]
    assert diagnostics["relative_projected_KKT_infinity"] <= limits["maximum_relative_projected_KKT_infinity"]
    assert diagnostics["normalized_directional_derivative"] <= limits["maximum_normalized_directional_derivative"]


def test_trial_is_nonpropagating_and_not_a_root() -> None:
    contract = target._contract()
    assert contract["nonpropagating_physical_trial"]["accepted_trial_is_not_a_root"]
    assert contract["nonpropagating_physical_trial"]["accepted_trial_must_not_be_propagated"]
    assert not contract["claim_boundary"]["primary_root_execution_authorized"]


def test_canonical_package_closes_if_present() -> None:
    directory = target.CANONICAL_DIRECTORY
    if not directory.exists(): return
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
    summary = target._utils()._read_json(directory / "summary.json")
    assert summary["equation_form_preflight_rejection_preserved"]
    assert summary["saved_inexact_direction_qualified"]
    assert not summary["primary_root_execution_authorized"]
