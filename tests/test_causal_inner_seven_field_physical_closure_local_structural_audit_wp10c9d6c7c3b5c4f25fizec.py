from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

import run_causal_inner_seven_field_physical_closure_local_structural_audit_wp10c9d6c7c3b5c4f25fizec as target


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_parent_authorizes_only_the_local_structural_audit() -> None:
    parent = target._validate_parent(require_clean=False)
    assert parent["summary"]["classification"] == target.parent.CLASSIFICATION
    assert parent["summary"]["local_structural_audit_authorized"]
    assert not parent["summary"]["seven_field_trajectory_authorized"]
    assert parent["contract"]["physical_entropy_extension"][
        "no_post_hoc_matrix_symmetrization"
    ]


def test_independent_jacobian_is_sixth_order_centered_on_polynomials() -> None:
    chart = np.asarray([0.2, -0.3])
    steps = np.asarray([1.0e-3, 2.0e-3])

    def function(values):
        x, y = values
        return np.asarray([x**4 + 2.0 * y**3, x * y])

    _value, jacobian = target._sixth_order_centered_jacobian(
        function,
        chart,
        steps,
    )
    expected = np.asarray(
        [[4.0 * chart[0] ** 3, 6.0 * chart[1] ** 2], [chart[1], chart[0]]]
    )
    np.testing.assert_allclose(jacobian, expected, rtol=2.0e-10, atol=2.0e-12)


def test_negative_classification_does_not_authorize_propagation() -> None:
    assert target.CLASSIFICATION.endswith("entropy_failed")
    assert target.AUTHORIZED_NEXT.startswith("definitions_only_")


@pytest.mark.skipif(
    not target.CANONICAL_DIRECTORY.exists(),
    reason="canonical Stage-3 audit has not yet executed",
)
def test_frozen_negative_package_closes_and_is_fail_closed() -> None:
    summary = _read(target.CANONICAL_DIRECTORY / "summary.json")
    metrics = _read(target.CANONICAL_DIRECTORY / "audit_metrics.json")
    assert not summary["passed"]
    assert summary["audit_completed"]
    assert not summary["entropy_integrability_passed"]
    assert summary["stable_order_unity_obstruction"]
    assert summary["corrective_architecture_manifest_authorized"]
    assert not summary["seven_field_trajectory_authorized"]
    assert not summary["complete_cycle_execution_authorized"]
    assert metrics["minimum_relative_entropy_flux_curl_defect"] > 0.1
    assert all(
        row["maximum_curl_pair"] == [3, 4]
        for row in metrics["step_ladder"]
    )
    for line in (target.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        assert _sha256(target.CANONICAL_DIRECTORY / name) == expected
