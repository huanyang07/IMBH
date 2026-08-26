from __future__ import annotations

import hashlib

import run_causal_inner_entropy_complete_fixed_q_primary_inexact_newton_root_execution_manifest_wp10c9d6c7c3b5c4f25fizev as target


def test_nonpropagating_trial_is_locked() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["classification"] == target.parent.PASS_CLASSIFICATION
    assert validated["metrics"]["selected_step_factor"] == 0.25
    assert validated["summary"]["new_nonlinear_roots"] == 0


def test_solver_and_certification_budgets_are_separate() -> None:
    contract = target._contract()
    assert contract["inexact_Newton_solver"]["maximum_solver_colored_Jacobian_assemblies"] == 2
    assert contract["post_root_certification"]["one_fresh_12_color_equation_Jacobian"]
    assert contract["post_root_certification"]["certification_assembly_cannot_change_the_root"]


def test_physical_spectrum_is_binding() -> None:
    contract = target._contract()
    assert contract["normal_attraction"]["maximum_spectral_abscissa_per_second"] == -1.0
    assert contract["normal_attraction"]["minimum_attraction_to_slow_relative_rate_ratio"] == 10.0
    assert not contract["claim_boundary"]["heldout_root_execution_authorized"]


def test_canonical_package_closes_if_present() -> None:
    directory = target.CANONICAL_DIRECTORY
    if not directory.exists(): return
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
    summary = target._utils()._read_json(directory / "summary.json")
    assert summary["definitions_only"]
    assert summary["primary_root_execution_authorized"]
    assert not summary["heldout_root_execution_authorized"]
