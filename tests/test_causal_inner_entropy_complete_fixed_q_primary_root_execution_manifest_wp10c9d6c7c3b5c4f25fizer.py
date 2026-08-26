from __future__ import annotations

import hashlib

import run_causal_inner_entropy_complete_fixed_q_primary_root_execution_manifest_wp10c9d6c7c3b5c4f25fizer as target


def test_projected_field_certificate_is_locked() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["classification"] == target.parent.PASS_CLASSIFICATION
    assert validated["summary"]["colored_fast_Jacobian_certified"]
    assert validated["metrics"]["new_nonlinear_roots"] == 0


def test_equation_form_removes_base_rate_scaling() -> None:
    contract = target._contract()
    scaling = contract["equation_row_scaling"]
    assert scaling["scales_frozen_at_hash_locked_primary_base"]
    assert scaling["base_projected_rate_magnitude_is_not_a_row_scale"]
    assert contract["root_equivalence"][
        "projected_fast_rate_zero_is_mathematically_equivalent"
    ]


def test_attraction_uses_physical_similarity_scaled_tangent() -> None:
    audit = target._contract()["future_root_attraction_audit"]
    assert audit["solver_row_normalized_Jacobian_eigenvalues_are_not_physical"]
    assert audit["maximum_spectral_abscissa_per_second"] == -1.0
    assert audit["minimum_attraction_to_slow_relative_rate_ratio"] == 10.0


def test_preflight_stops_before_root_execution() -> None:
    claim = target._contract()["claim_boundary"]
    assert claim["equation_form_preflight_authorized"]
    assert not claim["primary_nonlinear_root_execution_authorized"]
    assert not claim["complete_cycle_execution_authorized"]


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
    assert summary["definitions_only"]
    assert summary["equation_form_preflight_authorized"]
    assert not summary["primary_nonlinear_root_execution_authorized"]
