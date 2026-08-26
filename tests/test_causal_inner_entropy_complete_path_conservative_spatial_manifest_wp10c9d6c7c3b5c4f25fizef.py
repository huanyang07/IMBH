from __future__ import annotations

import hashlib
from pathlib import Path

import run_causal_inner_entropy_complete_path_conservative_spatial_manifest_wp10c9d6c7c3b5c4f25fizef as target


def test_local_parent_certificate_is_hash_locked() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["classification"] == target.parent.PASS_CLASSIFICATION
    assert validated["summary"]["complete_reduced_principal_certified"]
    assert validated["summary"]["advective_cluster_certified"]


def test_mixed_spatial_form_preserves_exact_rows() -> None:
    contract = target._contract()
    mixed = contract["mixed_spatial_form"]
    path = contract["DLM_path"]
    assert tuple(mixed["physical_conservative_rows"]) == (0, 1, 2, 3)
    assert tuple(mixed["exact_material_current_rows"]) == (5, 6)
    assert tuple(mixed["nonconservative_projected_shear_row"]) == (4,)
    assert mixed["no_nonconservative_derivative_hidden_as_a_lower_order_source"]
    assert path["conservative_row_exact_flux_difference_parity_required"]
    assert tuple(path["quadrature_ladder"]) == (4, 8, 16)


def test_complete_eigenbasis_fluctuation_is_binding() -> None:
    contract = target._contract()
    dissipation = contract["complete_eigenbasis_dissipation"]
    gates = contract["binding_gates"]
    claims = contract["claim_boundary"]
    assert dissipation["negative_plus_positive_equals_total_path_jump"]
    assert dissipation["shared_flux_from_both_sides_required_on_exact_flux_rows"]
    assert dissipation["characteristic_quadratic_dissipation_nonnegative"]
    assert dissipation["scalar_max_speed_Rusanov_forbidden"]
    assert gates["all_cases_and_all_gates_required"] and gates["fail_closed"]
    assert claims["spatial_operator_implementation_authorized"]
    assert not claims["semidiscrete_cell_operator_authorized"]
    assert not claims["trajectory_authorized"]


def test_canonical_package_closes_if_present() -> None:
    directory = target.CANONICAL_DIRECTORY
    if not directory.exists():
        return
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        assert actual == expected
    summary = target._utils()._read_json(directory / "summary.json")
    assert summary["classification"] == target.CLASSIFICATION
    assert summary["definitions_only"]
    assert summary["local_architecture_certificate_preserved"]
    assert summary["authorized_next"] == target.AUTHORIZED_NEXT
    assert not summary["semidiscrete_cell_operator_authorized"]


def test_declared_paths_are_workspace_relative() -> None:
    for relative in (
        target.THIS_RUNNER,
        target.THIS_TEST,
        target.PHYSICAL_SOURCE,
        target.PHYSICAL_TEST,
        target.SPATIAL_SOURCE,
        target.SPATIAL_TEST,
        target.REPORT_RELATIVE,
    ):
        assert not Path(relative).is_absolute()
