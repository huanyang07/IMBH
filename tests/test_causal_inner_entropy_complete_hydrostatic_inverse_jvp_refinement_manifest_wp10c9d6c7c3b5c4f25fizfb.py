from __future__ import annotations

import hashlib

import run_causal_inner_entropy_complete_hydrostatic_inverse_jvp_refinement_manifest_wp10c9d6c7c3b5c4f25fizfb as target


def test_original_narrow_rejection_is_preserved() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["classification"] == target.parent.FAIL_CLASSIFICATION
    assert not validated["summary"]["passed"]
    assert validated["metrics"]["offline_seven_field_operator_calls"] == 4


def test_refinement_tightens_numerics_without_relaxing_gate() -> None:
    refinement = target._contract()["refinement"]
    assert refinement["reconstruction_tolerance"] == 1.0e-12
    assert refinement["ordered_central_steps"] == (1.0e-5, 5.0e-6, 2.0e-6)
    assert refinement["maximum_JVP_relative_defect"] == 2.0e-5
    assert refinement["new_seven_field_radial_operator_calls"] == 0
    assert refinement["no_gate_or_physical_equation_is_relaxed"]


def test_no_atlas_or_cycle_is_authorized() -> None:
    claim = target._contract()["claim_boundary"]
    assert claim["derivative_refinement_authorized"]
    assert not claim["truth_resampling_authorized"]
    assert not claim["slow_flux_atlas_authorized"]
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
    assert summary["original_inverse_JVP_rejection_preserved"]
    assert not summary["truth_resampling_authorized"]
