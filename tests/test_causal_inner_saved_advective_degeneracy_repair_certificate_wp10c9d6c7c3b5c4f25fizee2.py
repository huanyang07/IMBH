from __future__ import annotations

import hashlib
from pathlib import Path

import run_causal_inner_saved_advective_degeneracy_repair_certificate_wp10c9d6c7c3b5c4f25fizee2 as target


def test_parent_manifest_and_repaired_source_are_hash_locked() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["classification"] == target.parent.CLASSIFICATION
    utils = target._utils()
    assert utils._sha256(target.ROOT / target.PHYSICAL_SOURCE) == (
        target.PHYSICAL_SOURCE_SHA256
    )
    assert utils._sha256(target.ROOT / target.PHYSICAL_TEST) == (
        target.PHYSICAL_TEST_SHA256
    )


def test_certificate_scope_preserves_negative_parent() -> None:
    contract = target._utils()._read_json(
        target.parent.CANONICAL_DIRECTORY / "repair_contract.json"
    )
    assert contract["parent_negative_result"]["preserved_as_binding"]
    assert contract["saved_point_certificate"]["scope"] == (
        "one_saved_point_nonpropagating"
    )
    assert contract["saved_point_certificate"]["trajectory_steps"] == 0
    assert contract["claim_boundary"]["full_envelope_retry_authorized"] is False


def test_canonical_package_closes_and_passes_if_present() -> None:
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
    metrics = target._utils()._read_json(
        directory / "certificate_metrics.json"
    )
    assert summary["classification"] == target.PASS_CLASSIFICATION
    assert summary["passed"]
    assert summary["parent_negative_result_preserved"]
    assert summary["full_envelope_retry_authorized"]
    assert summary["new_trajectory_steps"] == 0
    assert summary["authorized_next"] == target.AUTHORIZED_NEXT_ON_PASS
    assert metrics["maximum_imaginary_speed_over_c"] <= 1.0e-10
    assert max(metrics["material_product_identity_relative_defects"]) <= 1.0e-12
    assert max(metrics["matrix_derivative_ladder"].values()) <= 1.0e-7


def test_declared_paths_are_workspace_relative() -> None:
    for relative in (
        target.THIS_RUNNER,
        target.THIS_TEST,
        target.PHYSICAL_SOURCE,
        target.PHYSICAL_TEST,
        target.REPORT_RELATIVE,
    ):
        assert not Path(relative).is_absolute()
