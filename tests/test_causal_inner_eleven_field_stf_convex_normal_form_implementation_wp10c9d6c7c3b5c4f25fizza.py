import hashlib
import json

import pytest

import run_causal_inner_eleven_field_stf_convex_normal_form_implementation_wp10c9d6c7c3b5c4f25fizza as target


def test_architecture_manifest_authorizes_this_package():
    validated = target._validate_manifest(require_clean=False)
    assert validated["summary"]["eleven_field_architecture_selected"]
    assert validated["summary"]["authorized_next"] == target.WORK_PACKAGE


def test_claim_boundary_remains_structural_only():
    assert "normal_form" in target.PASS_CLASSIFICATION
    assert target.AUTHORIZED_NEXT.startswith("definitions_only_")


def test_independent_five_point_gradient_is_exact_for_quadratic_fixture():
    import numpy as np
    from imri_qpe.layer3_minidisk_1d.causal_inner_eleven_field_convex import (
        build_eleven_field_convex_normal_form,
        reference_eleven_field_parameters,
    )

    form = build_eleven_field_convex_normal_form(
        reference_eleven_field_parameters()
    )
    point = np.linspace(-0.1, 0.2, 11)
    gradient = target._five_point_gradient(form.state_potential, point, 2.0e-4)
    np.testing.assert_allclose(
        gradient, form.state_current(point), atol=2.0e-12, rtol=0.0
    )


@pytest.mark.skipif(
    not target.CANONICAL_DIRECTORY.exists(),
    reason="canonical STF normal-form package has not yet run",
)
def test_canonical_package_closes_and_does_not_authorize_trajectory():
    summary = json.loads(
        (target.CANONICAL_DIRECTORY / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["passed"]
    assert summary["five_STF_basis_certified"]
    assert summary["quadratic_convex_normal_form_certified"]
    assert not summary["nonlinear_physical_master_potential_derived"]
    assert not summary["eleven_field_trajectory_authorized"]
    assert not summary["complete_cycle_execution_authorized"]
    for line in (target.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256(
            (target.CANONICAL_DIRECTORY / name).read_bytes()
        ).hexdigest() == expected
