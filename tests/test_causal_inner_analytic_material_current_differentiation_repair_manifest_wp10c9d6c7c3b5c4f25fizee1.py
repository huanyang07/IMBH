from __future__ import annotations

import hashlib
from pathlib import Path

import run_causal_inner_analytic_material_current_differentiation_repair_manifest_wp10c9d6c7c3b5c4f25fizee1 as target


def test_parent_negative_result_is_preserved() -> None:
    contract = target._contract()
    parent = contract["parent_negative_result"]
    assert parent["classification"] == target.PARENT_CLASSIFICATION
    assert parent["preserved_as_binding"]
    assert parent["retroactive_reclassification_forbidden"]
    assert parent["recorded_maximum_imaginary_speed_over_c"] > parent["frozen_gate"]


def test_repair_is_only_the_exact_product_rule() -> None:
    contract = target._contract()
    repair = contract["authorized_source_repair"]
    assert repair["analytic_identity"] == "d(v*U)=v*dU+U*dv"
    assert tuple(repair["repaired_principal_rows"]) == (0, 5, 6)
    assert repair["stress_energy_rows_other_than_rest_mass_unchanged"]
    assert repair["shear_row_unchanged"]
    assert repair["eigenvalue_clipping_forbidden"]
    assert repair["eigenvalue_projection_forbidden"]
    assert repair["matrix_symmetrization_forbidden"]
    assert repair["threshold_relaxation_forbidden"]


def test_saved_point_certificate_is_nonpropagating_and_fail_closed() -> None:
    contract = target._contract()
    certificate = contract["saved_point_certificate"]
    gates = contract["binding_gates"]
    claims = contract["claim_boundary"]
    assert certificate["label"] == target.SAVED_LABEL
    assert tuple(certificate["derivative_step_factors"]) == (2.0, 1.0, 0.5)
    assert certificate["trajectory_steps"] == 0
    assert gates["maximum_imaginary_speed_over_c"] == 1.0e-10
    assert gates["all_factors_and_all_gates_required"]
    assert gates["fail_closed"]
    assert claims["saved_point_certificate_authorized"]
    assert not claims["full_envelope_retry_authorized"]
    assert not claims["trajectory_authorized"]
    assert not claims["complete_cycle_execution_authorized"]


def test_parent_and_old_implementation_are_hash_locked() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["classification"] == target.PARENT_CLASSIFICATION
    utils = target._utils()
    assert utils._sha256(target.ROOT / target.PHYSICAL_SOURCE) == (
        target.OLD_PHYSICAL_SOURCE_SHA256
    )
    assert utils._sha256(target.ROOT / target.PHYSICAL_TEST) == (
        target.OLD_PHYSICAL_TEST_SHA256
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
    summary = target._utils()._read_json(directory / "summary.json")
    assert summary["classification"] == target.CLASSIFICATION
    assert summary["definitions_only"]
    assert summary["parent_negative_result_preserved"]
    assert summary["authorized_next"] == target.AUTHORIZED_NEXT
    assert not summary["full_envelope_retry_authorized"]


def test_declared_paths_are_workspace_relative() -> None:
    for relative in (
        target.THIS_RUNNER,
        target.THIS_TEST,
        target.PHYSICAL_SOURCE,
        target.PHYSICAL_TEST,
        target.REPORT_RELATIVE,
    ):
        assert not Path(relative).is_absolute()
