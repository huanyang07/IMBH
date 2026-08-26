from __future__ import annotations

import hashlib
from pathlib import Path

import run_causal_inner_entropy_complete_projected_local_structural_audit_retry_wp10c9d6c7c3b5c4f25fizee3 as target


def test_saved_point_parent_and_frozen_kernel_are_hash_locked() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["classification"] == target.parent.PASS_CLASSIFICATION
    assert validated["summary"]["parent_negative_result_preserved"]
    assert target._utils()._sha256(target.ROOT / target.FROZEN_AUDIT_RUNNER) == (
        target.FROZEN_AUDIT_RUNNER_SHA256
    )


def test_classification_mapping_is_fail_closed() -> None:
    assert target._retry_classification(
        target.frozen_audit.PASS_CLASSIFICATION
    ) == target.PASS_CLASSIFICATION
    assert target._retry_classification(
        target.frozen_audit.CAUSALITY_FAILURE
    ) == target.CAUSALITY_FAILURE
    assert target._retry_classification(
        target.frozen_audit.HYPERBOLICITY_FAILURE
    ) == target.HYPERBOLICITY_FAILURE
    assert target._retry_classification(
        target.frozen_audit.LEDGER_FAILURE
    ) == target.LEDGER_FAILURE
    assert target._retry_classification(
        target.frozen_audit.DERIVATION_FAILURE
    ) == target.DERIVATION_FAILURE


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
    metrics = target._utils()._read_json(directory / "audit_metrics.json")
    assert summary["classification"] in {
        target.PASS_CLASSIFICATION,
        target.CAUSALITY_FAILURE,
        target.HYPERBOLICITY_FAILURE,
        target.LEDGER_FAILURE,
        target.DERIVATION_FAILURE,
    }
    assert summary["parent_negative_result_preserved"]
    assert summary["saved_point_repair_certificate_preserved"]
    assert summary["new_trajectory_steps"] == 0
    if summary["passed"]:
        assert summary["complete_reduced_principal_certified"]
        assert summary["authorized_next"] == target.AUTHORIZED_NEXT_ON_PASS
        assert metrics["base_points_audited"] == metrics["base_points_planned"]
        assert metrics["first_failure"] is None
    else:
        assert not summary["complete_reduced_principal_certified"]
        assert summary["authorized_next"] is None
        assert metrics["first_failure"] is not None


def test_declared_paths_are_workspace_relative() -> None:
    for relative in (
        target.THIS_RUNNER,
        target.THIS_TEST,
        target.FROZEN_AUDIT_RUNNER,
        target.PHYSICAL_SOURCE,
        target.PHYSICAL_TEST,
        target.REPORT_RELATIVE,
    ):
        assert not Path(relative).is_absolute()
