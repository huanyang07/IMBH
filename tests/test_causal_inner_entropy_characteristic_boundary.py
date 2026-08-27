import numpy as np
import pytest

from imri_qpe.layer3_minidisk_1d.causal_inner_entropy_characteristic_boundary import (
    audit_outward_entropy_characteristic_boundary,
    build_outward_entropy_characteristic_boundary,
)


def test_negative_radial_speeds_are_excision_at_left_and_incoming_at_right():
    radial = np.diag((-0.8, -0.31, -0.05))
    left = build_outward_entropy_characteristic_boundary(
        radial, outward_normal=-1.0
    )
    right = build_outward_entropy_characteristic_boundary(
        radial, outward_normal=1.0
    )
    assert left.incoming_count == 0
    assert right.incoming_count == 3
    np.testing.assert_array_equal(left.incoming_projector, np.zeros((3, 3)))
    np.testing.assert_allclose(right.incoming_projector, np.eye(3), atol=2e-15)
    assert audit_outward_entropy_characteristic_boundary(left).passed
    assert audit_outward_entropy_characteristic_boundary(right).passed


def test_mixed_boundary_penalty_is_positive_semidefinite_and_projected():
    rotation = np.asarray(((0.8, -0.6), (0.6, 0.8)))
    radial = rotation @ np.diag((-0.4, 0.7)) @ rotation.T
    boundary = build_outward_entropy_characteristic_boundary(
        radial, outward_normal=1.0
    )
    audit = audit_outward_entropy_characteristic_boundary(boundary)
    assert audit.passed
    assert boundary.incoming_count == 1
    assert audit.penalty_minimum_eigenvalue >= -2e-15
    np.testing.assert_allclose(
        boundary.incoming_projector @ boundary.incoming_penalty,
        boundary.incoming_penalty,
        atol=2e-15,
    )


def test_neutral_characteristic_is_not_assigned_incoming():
    boundary = build_outward_entropy_characteristic_boundary(
        np.diag((-0.2, 0.0, 0.3)), outward_normal=1.0
    )
    audit = audit_outward_entropy_characteristic_boundary(boundary)
    assert boundary.incoming_count == 1
    assert audit.neutral_count == 1


def test_nonsymmetric_or_invalid_normal_fails_closed():
    with pytest.raises(ValueError, match="symmetric"):
        build_outward_entropy_characteristic_boundary(
            np.asarray(((0.0, 1.0), (0.0, 0.0))), outward_normal=1.0
        )
    with pytest.raises(ValueError, match="outward_normal"):
        build_outward_entropy_characteristic_boundary(
            np.eye(2), outward_normal=0.0
        )
