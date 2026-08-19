from __future__ import annotations

import numpy as np

import run_causal_inner_exact_geometric_departure_chart_preflight_wp10c9d6c7c3b5c4f25ay as f25ay


def test_frozen_geometric_chart_manifest_is_locked():
    frozen = f25ay._validate_manifest(require_clean=False)
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["authorized_next"] == f25ay.WORK_PACKAGE
    assert not frozen["contract"]["exact_geometric_retraction"][
        "rate_reaction_lift_used"
    ]


def test_state_local_coordinate_jacobian_reproduces_certified_base_map():
    components = f25ay.coordinate_tools._coordinate_components()
    jacobian, metrics = f25ay._coordinate_jacobian(
        components["state"], components
    )
    assert metrics["rank"] == 162
    assert metrics["condition_number"] <= 1.0e4
    assert f25ay._relative(jacobian, components["jacobian"]) <= 1.0e-12


def test_departure_family_is_tangent_and_has_eight_directions():
    metrics, arrays = f25ay._departure_family()
    assert arrays["departure_basis"].shape == (560, 28)
    assert arrays["energy_directions"].shape == (28, 8)
    assert metrics["departure_base_physical_tangency_defect"] <= 1.0e-10


def test_minimum_norm_coordinate_correction_closes_linear_system():
    rng = np.random.default_rng(25)
    jacobian = rng.normal(size=(7, 13))
    error = rng.normal(size=7)
    correction = f25ay._minimum_norm_coordinate_correction(jacobian, error)
    assert np.allclose(jacobian @ correction, error, rtol=1.0e-12, atol=1.0e-12)
