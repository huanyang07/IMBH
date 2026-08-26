from __future__ import annotations

import hashlib
from pathlib import Path

import run_causal_inner_entropy_complete_projected_architecture_correction_manifest_wp10c9d6c7c3b5c4f25fized1 as target


def test_contract_selects_entropy_complete_projected_equation() -> None:
    contract = target._contract()
    model = contract["precise_reduced_model"]
    shear = contract["entropy_complete_shear_equation"]
    standard = contract["causality_and_hyperbolicity_standard"]
    assert model["one_amplitude_model_is_a_projected_disk_closure"]
    assert model[
        "one_amplitude_model_is_not_claimed_invariant_under_the_full_five_component_shear_PDE"
    ]
    assert shear["both_temporal_and_radial_velocity_derivatives_retained"]
    assert shear[
        "stress_density_expansion_terms_cancel_against_the_full_entropy_current_term"
    ]
    assert standard["binding_object"] == (
        "complete_reduced_7_by_7_radial_quasilinear_pencil"
    )


def test_contract_is_fail_closed_and_authorizes_no_trajectory() -> None:
    contract = target._contract()
    gates = contract["binding_gates"]
    claims = contract["claim_boundary"]
    assert gates["all_points_and_all_gates_required"]
    assert gates["fail_closed"]
    assert claims["local_structural_audit_authorized"]
    assert not claims["architecture_certified"]
    assert not claims["spatial_discretization_authorized"]
    assert not claims["trajectory_authorized"]
    assert not claims["complete_cycle_execution_authorized"]


def test_parent_and_implementation_are_hash_locked() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["classification"] == target.parent.CLASSIFICATION
    assert target.parent._sha256(target.ROOT / target.PHYSICAL_SOURCE) == (
        target.PHYSICAL_SOURCE_SHA256
    )
    assert target.parent._sha256(target.ROOT / target.PHYSICAL_TEST) == (
        target.PHYSICAL_TEST_SHA256
    )


def test_canonical_package_closes_if_present() -> None:
    directory = target.CANONICAL_DIRECTORY
    if not directory.exists():
        return
    for line in (directory / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        assert actual == expected
    summary = target.parent._read_json(directory / "summary.json")
    assert summary["classification"] == target.CLASSIFICATION
    assert summary["definitions_only"]
    assert summary["authorized_next"] == target.AUTHORIZED_NEXT
    assert not summary["seven_field_trajectory_authorized"]


def test_declared_paths_are_workspace_relative() -> None:
    for relative in (
        target.THIS_RUNNER,
        target.THIS_TEST,
        target.PHYSICAL_SOURCE,
        target.PHYSICAL_TEST,
        target.REPORT_RELATIVE,
    ):
        assert not Path(relative).is_absolute()
