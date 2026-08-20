from __future__ import annotations

import json

import numpy as np

import run_causal_inner_authentic_center_local_field_overlap_manifest_wp10c9d6c7c3b5c4f25cr as f25cr


def test_authentic_transition_authorizes_local_field_manifest():
    frozen = f25cr._validate_parent(require_clean=False)
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["authentic_center_established"]
    assert frozen["summary"]["authorized_next"] == (
        "definitions_only_authentic_center_local_field_and_overlap_manifest"
    )


def test_direction_design_is_forward_rank_three_and_separated():
    forward = np.arange(1.0, 29.0)
    design = f25cr._direction_design(forward)
    assert design["active_departure_basis"].shape == (28, 3)
    assert design["training_directions"].shape == (4, 28)
    assert design["holdout_directions"].shape == (4, 28)
    assert np.allclose(
        design["active_departure_basis"].T
        @ design["active_departure_basis"],
        np.eye(3),
        atol=1.0e-12,
    )
    assert np.min(
        design["training_directions"] @ design["forward_direction"]
    ) >= 0.85
    assert np.min(
        design["holdout_directions"] @ design["forward_direction"]
    ) >= 0.85
    assert np.max(
        np.abs(
            design["holdout_directions"]
            @ design["training_directions"].T
        )
    ) <= 0.95


def test_affine_residual_fit_recovers_synthetic_map():
    rng = np.random.default_rng(419)
    coordinates = rng.normal(size=(12, 3))
    coefficients = rng.normal(size=(4, 7))
    residuals = f25cr._affine_features(coordinates) @ coefficients
    fitted = f25cr._fit_affine_residual(
        coordinates, residuals, regularization=0.0
    )
    assert np.allclose(fitted, coefficients, atol=1.0e-12)


def test_overlap_weight_is_c2_bounded_and_has_frozen_endpoints():
    assert f25cr._smoothstep_weight(0.0) == 0.0
    assert f25cr._smoothstep_weight(f25cr.OVERLAP_ACTIVATION_LOAD) == 0.0
    assert f25cr._smoothstep_weight(f25cr.OVERLAP_FULL_NEW_LOAD) == 1.0
    assert f25cr._smoothstep_weight(1.0) == 1.0
    values = np.asarray(
        [f25cr._smoothstep_weight(value) for value in np.linspace(0.0, 0.02, 101)]
    )
    assert np.all(values >= 0.0)
    assert np.all(values <= 1.0)
    assert np.all(np.diff(values) >= 0.0)


def test_contract_freezes_no_generator_direct_atlas():
    contract = f25cr._contract()
    architecture = contract["mathematical_architecture"]
    assert contract["definitions_only"]
    assert architecture["online_state_dependent_coordinate_Jacobian_calls"] == 0
    assert architecture["new_complete_generator_assemblies"] == 0
    assert contract["next_geometry_preflight_budget"] == {
        "candidate_count": 8,
        "new_continuous_rate_calls": 0,
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_fixed_Q_roots": 0,
        "propagated_states": 0,
    }
    assert contract["prospective_total_rate_budget_after_geometry_pass"][
        "total_new_exact_continuous_rate_calls"
    ] == 9
    assert not contract["authorization_boundaries"][
        "repeated_authentic_recenter_roots_required_by_architecture"
    ]


def test_canonical_manifest_if_present():
    if not f25cr.CANONICAL_DIRECTORY.exists():
        return
    f25cr._checksums(f25cr.CANONICAL_DIRECTORY)
    summary = json.loads(
        (f25cr.CANONICAL_DIRECTORY / "summary.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (f25cr.CANONICAL_DIRECTORY / "design_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    arrays = f25cr._load_npz(
        f25cr.CANONICAL_DIRECTORY / "center_local_field_design.npz"
    )
    assert summary["classification"] == f25cr.CLASSIFICATION
    assert summary["passed"] and summary["definitions_only"]
    assert summary["authorized_next"] == f25cr.AUTHORIZED_NEXT
    assert not summary["physical_microburst_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    assert metrics["passed"] and all(metrics["checks"].values())
    assert arrays["authentic_center_fixed_restriction"].shape == (470, 560)
    assert arrays["revealed_overlap_local_coordinates"].shape == (16, 470)
    assert arrays["training_directions"].shape == (4, 28)
    assert arrays["holdout_directions"].shape == (4, 28)
