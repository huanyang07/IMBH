from __future__ import annotations

import hashlib
import json

import pytest

import run_causal_inner_generalized_maxwell_cattaneo_architecture_manifest_wp10c9d6c7c3b5c4f25fized as target


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_failed_entropy_realization_is_the_binding_parent() -> None:
    parent = target._validate_parent(require_clean=False)
    assert not parent["summary"]["passed"]
    assert parent["summary"]["entropy_integrability_passed"] is False
    assert parent["summary"]["stable_order_unity_obstruction"]
    assert parent["metrics"]["minimum_relative_entropy_flux_curl_defect"] > 0.1


def test_corrected_architecture_selects_transient_quasilinear_class() -> None:
    contract = target._contract()
    selected = contract["selected_PDE_class"]
    assert "Maxwell-Cattaneo" in selected["name"]
    assert "nabla_a J" in selected["exact_conservation"]
    assert not selected["global_Godunov_potential_required"]
    assert selected["post_hoc_symmetrization_still_forbidden"]


def test_complete_nonlinear_symbol_and_causality_are_binding() -> None:
    contract = target._contract()
    principal = contract["nonlinear_principal_contract"]
    causality = contract["nonlinear_causality_gates"]
    strong = contract["strong_hyperbolicity_gates"]
    assert principal["complete_symbol_not_isolated_shear_cone_is_binding"]
    assert causality["evaluate_all_specialized_necessary_and_sufficient_inequalities"]
    assert causality["no_linearized_only_certificate"]
    assert strong["all_generalized_eigenvalues_real"]
    assert strong["diagonally_equilibrated_eigenvector_condition_number_max"] == 1.0e8


def test_future_discretization_is_path_conservative_but_not_authorized() -> None:
    spatial = target._contract()["future_spatial_discretization_if_local_audit_passes"]
    assert spatial["method"].startswith("path-conservative")
    assert spatial["path_consistency"].startswith("Dal Maso")
    assert not spatial["this_manifest_authorizes_spatial_discretization"]


def test_manifest_authorizes_only_local_structural_audit() -> None:
    contract = target._contract()
    claims = contract["claim_boundary"]
    budget = contract["budget"]
    assert claims["corrected_transient_architecture_selected"]
    assert not claims["corrected_transient_architecture_certified"]
    assert claims["local_structural_audit_authorized"]
    assert not claims["spatial_discretization_authorized"]
    assert not claims["seven_field_trajectory_authorized"]
    assert not claims["complete_cycle_authorized"]
    assert budget["new_trajectory_steps"] == 0
    assert budget["new_nonlinear_roots"] == 0


@pytest.mark.skipif(
    not target.CANONICAL_DIRECTORY.exists(),
    reason="canonical architecture manifest has not yet frozen",
)
def test_frozen_architecture_package_closes() -> None:
    summary = _read(target.CANONICAL_DIRECTORY / "summary.json")
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["failed_Godunov_realization_preserved"]
    assert summary["local_structural_audit_authorized"]
    assert not summary["seven_field_trajectory_authorized"]
    assert summary["authorized_next"] == target.AUTHORIZED_NEXT
    for line in (target.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        assert _sha256(target.CANONICAL_DIRECTORY / name) == expected
