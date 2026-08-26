from __future__ import annotations

import hashlib

import run_causal_inner_entropy_complete_semidiscrete_relaxation_manifest_wp10c9d6c7c3b5c4f25fizeh as target


def test_interface_parent_is_locked() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["classification"] == target.parent.PASS_CLASSIFICATION
    assert validated["summary"]["exact_flux_rows_certified"]
    assert validated["metrics"]["first_failure"] is None


def test_source_contract_counts_vertical_energy_once() -> None:
    contract = target._contract()
    source = contract["local_lower_source"]
    assert source["vertical_pressure_work_is_not_an_extra_energy_source"]
    assert source["stream_source_added_once_only_at_cell_integration"]
    assert source["height_row"] == "S_H=D*beta_H/u0"


def test_relaxation_limit_preserves_causal_shear() -> None:
    relaxation = target._contract()["equilibrium_relaxation_limit"]
    assert relaxation["causal_shear_remains_a_finite_rate_fifth_field"]
    assert relaxation["tau_to_zero_at_fixed_viscosity_forbidden"]
    assert tuple(relaxation["fast_vertical_source_multipliers"]) == (1.0, 2.0, 4.0, 8.0)
    assert relaxation["old_failed_face_equivalence_claim_forbidden"]


def test_claim_boundary_is_nonpropagating() -> None:
    claims = target._contract()["claim_boundary"]
    assert claims["local_source_implementation_authorized"]
    assert claims["fixed_geometry_periodic_semidiscrete_implementation_authorized"]
    assert claims["nonpropagating_relaxation_audit_authorized"]
    assert not claims["radial_boundary_implementation_authorized"]
    assert not claims["bounded_crossing_trajectory_authorized"]


def test_canonical_package_closes_if_present() -> None:
    directory = target.CANONICAL_DIRECTORY
    if not directory.exists():
        return
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
    summary = target._utils()._read_json(directory / "summary.json")
    assert summary["classification"] == target.CLASSIFICATION
    assert summary["definitions_only"]
    assert summary["new_trajectory_steps"] == 0
    assert not summary["bounded_crossing_trajectory_authorized"]
