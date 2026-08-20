from __future__ import annotations

import json

import numpy as np

import run_causal_inner_partitioned_authentic_center_field_revision_manifest_wp10c9d6c7c3b5c4f25cv as f25cv


def test_failed_pooled_fit_authorizes_definitions_only_revision():
    frozen = f25cv._validate_parent(require_clean=False)
    assert not frozen["summary"]["passed"]
    assert frozen["summary"]["exact_truth_passed"]
    assert not frozen["summary"]["local_field_training_passed"]
    assert frozen["summary"]["holdout_rate_calls"] == 0
    assert not frozen["metrics"]["field_checks"]["training_q162"]


def test_partition_has_exact_frozen_role_separation():
    frozen = f25cv._validate_parent(require_clean=False)
    database = f25cv._known_database(frozen)
    design = frozen["design"]
    local = np.vstack(
        (
            design["revealed_overlap_local_coordinates"],
            np.zeros((1, 470)),
            design["training_local_coordinates"],
            design["holdout_local_coordinates"],
        )
    )
    weights, center, forward, active = f25cv._partition_weights(
        local, database["active_basis"]
    )
    assert np.array_equal(weights[:16], np.zeros(16))
    assert weights[16] == 1.0
    assert np.array_equal(weights[17:], np.ones(8))
    assert center[16] == 1.0 and forward[16] == 0.0
    assert np.max(active[:16, 0]) < f25cv.FORWARD_ZERO_COORDINATE


def test_partition_is_c2_at_all_frozen_endpoints():
    epsilon = 1.0e-7
    for lower, upper in (
        (f25cv.CENTER_CORE_FULL_RADIUS, f25cv.CENTER_CORE_ZERO_RADIUS),
        (f25cv.FORWARD_ZERO_COORDINATE, f25cv.FORWARD_FULL_COORDINATE),
    ):
        values = np.asarray((lower - epsilon, lower, lower + epsilon, upper - epsilon, upper, upper + epsilon))
        weights = f25cv._smoothstep(values, lower, upper)
        assert weights[0] == weights[1] == 0.0
        assert weights[-2] == weights[-1] == 1.0
        assert weights[2] < 1.0e-12
        assert 1.0 - weights[3] < 1.0e-12


def test_contract_keeps_blind_holdout_and_cost_boundaries():
    contract = f25cv._contract()
    assert contract["blind_holdout_execution"]["count"] == 4
    assert contract["blind_holdout_execution"]["coefficients_may_not_change"]
    assert contract["mathematical_architecture"][
        "online_state_dependent_coordinate_Jacobian_calls"
    ] == 0
    assert contract["authorization_boundaries"] == {
        "physical_microburst_authorized": False,
        "predictive_cycle_authorized": False,
        "fast_average_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }


def test_partitioned_field_implementation_matches_frozen_equations():
    rng = np.random.default_rng(2521)

    class FakeDirect:
        def decoded_delta(self, coordinate):
            return np.resize(np.asarray(coordinate), 560) + 0.25

        def full_state_rate(self, coordinate):
            return 2.0 * np.resize(np.asarray(coordinate), 560) - 0.5

        def field(self, coordinate):
            return 3.0 * np.asarray(coordinate) + 0.75

    center = rng.normal(scale=1.0e-3, size=470)
    direct = FakeDirect()
    restriction = rng.normal(scale=1.0e-2, size=(470, 560))
    active_basis = np.eye(28, 3)
    closure = {
        "authentic_center_absolute_coordinate": center,
        "authentic_center_scaled_delta": rng.normal(size=560),
        "authentic_center_direct_decoded_scaled_delta": direct.decoded_delta(center),
        "authentic_center_fixed_restriction": restriction,
        "active_departure_basis": active_basis,
        "decoder_affine_coefficients": rng.normal(size=(3, 560)),
        "full_rate_affine_coefficients": rng.normal(size=(4, 560)),
        "q162_Jacobian_affine_coefficients": rng.normal(
            scale=1.0e-2, size=(4, 162, 560)
        ),
    }
    field = f25cv.PartitionedAuthenticCenterField(
        closure, model=object(), direct=direct
    )
    local = np.zeros(470)
    local[-28] = 2.0e-2
    active = local[-28:] @ active_basis / f25cv.parent.manifest.ACTIVE_SCALE
    features = np.concatenate(([1.0], active))
    weight = field.weight(local)
    absolute = center + local
    old_delta = direct.decoded_delta(absolute)
    new_delta = (
        closure["authentic_center_scaled_delta"]
        + old_delta
        - closure["authentic_center_direct_decoded_scaled_delta"]
        + active @ closure["decoder_affine_coefficients"]
    )
    assert np.allclose(
        field.decoded_delta(local), old_delta + weight * (new_delta - old_delta)
    )
    old_full = direct.full_state_rate(absolute)
    new_full = old_full + features @ closure["full_rate_affine_coefficients"]
    assert np.allclose(
        field.full_state_rate(local), old_full + weight * (new_full - old_full)
    )
    new_coordinate = restriction @ new_full
    transported = np.einsum(
        "f,fij->ij", features, closure["q162_Jacobian_affine_coefficients"]
    )
    new_coordinate[:162] = transported @ new_full
    old_coordinate = direct.field(absolute)
    assert np.allclose(
        field.field(local),
        old_coordinate + weight * (new_coordinate - old_coordinate),
    )


def test_canonical_revision_manifest_if_present():
    if not f25cv.CANONICAL_DIRECTORY.exists():
        return
    f25cv._checksums(f25cv.CANONICAL_DIRECTORY)
    summary = json.loads(
        (f25cv.CANONICAL_DIRECTORY / "summary.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (f25cv.CANONICAL_DIRECTORY / "design_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    closure = f25cv._load_npz(
        f25cv.CANONICAL_DIRECTORY / "partitioned_local_field.npz"
    )
    assert summary["passed"] and summary["definitions_only"]
    assert summary["classification"] == f25cv.CLASSIFICATION
    assert summary["authorized_next"] == f25cv.AUTHORIZED_NEXT
    assert summary["new_exact_rate_calls"] == 0
    assert summary["blind_holdout_rate_calls"] == 0
    assert metrics["passed"] and all(metrics["checks"].values())
    assert closure["full_rate_affine_coefficients"].shape == (4, 560)
    assert closure["q162_Jacobian_affine_coefficients"].shape == (4, 162, 560)
