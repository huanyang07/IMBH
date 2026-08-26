from __future__ import annotations

import hashlib

import numpy as np

import run_causal_inner_entropy_complete_hydrostatic_inverse_order_recovery_execution_wp10c9d6c7c3b5c4f25fizfe as target


def test_order_recovery_manifest_is_locked() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["order_aware_inverse_diagnostic_authorized"]
    assert not validated["summary"]["slow_flux_atlas_authorized"]


def test_relative_defect_is_symmetric_in_scale() -> None:
    first = np.asarray((1.0, 2.0, 3.0))
    second = np.asarray((1.0, 2.0, 3.3))
    assert target._relative_defect(first, second) == target._relative_defect(
        second, first
    )


def test_richardson_contract_keeps_original_gate() -> None:
    audit = target.parent._contract()["order_aware_inverse_audit"]
    assert audit["maximum_each_Richardson_JVP_relative_defect"] == 2.0e-5
    assert audit["maximum_Richardson_pair_relative_disagreement"] == 2.0e-5
    assert audit["maximum_new_local_nonlinear_solves"] == 84


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
    metrics = target._utils()._read_json(directory / "order_recovery_metrics.json")
    assert summary["new_seven_field_operator_calls"] == 0
    assert summary["propagated_states"] == 0
    assert metrics["requested_local_nonlinear_solves"] == 84
    if summary["passed"]:
        assert metrics["minimum_global_worst_raw_defect_order"] >= 1.8
        assert metrics["maximum_Richardson_JVP_relative_defect"] <= 2.0e-5
