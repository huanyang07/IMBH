from __future__ import annotations

import hashlib

import run_causal_inner_entropy_complete_fixed_q_equation_form_root_preflight_wp10c9d6c7c3b5c4f25fizes as target


def test_root_preflight_manifest_is_locked() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["classification"] == target.parent.CLASSIFICATION
    assert validated["summary"]["equation_form_preflight_authorized"]
    assert not validated["summary"]["primary_nonlinear_root_execution_authorized"]


def test_call_budget_is_prospective() -> None:
    assert target.EXPECTED_PROJECTED_FIELD_CALLS == 21
    contract = target.parent._contract()
    assert contract["linearization_preflight"][
        "forward_colored_equation_evaluations"
    ] == 12
    assert contract["linearization_preflight"][
        "independent_central_JVP_directions"
    ] == 4


def test_preflight_executes_no_root() -> None:
    contract = target.parent._contract()
    assert contract["claim_boundary"]["equation_form_preflight_authorized"]
    assert not contract["claim_boundary"][
        "primary_nonlinear_root_execution_authorized"
    ]


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
    metrics = target._utils()._read_json(directory / "preflight_metrics.json")
    assert summary["new_nonlinear_roots"] == 0
    assert summary["propagated_states"] == 0
    if summary["passed"]:
        assert metrics["projected_field_calls"] == target.EXPECTED_PROJECTED_FIELD_CALLS
        assert summary["equation_form_linearization_certified"]
        assert summary["bounded_linear_step_certified"]
