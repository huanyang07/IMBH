import numpy as np
import pytest

from imri_qpe.layer3_minidisk_1d.causal_inner_cycle_atlas import (
    locate_full_simplex,
    locate_guard_sheet,
    periodic_phase_weights,
)


def test_full_simplex_is_exact_at_anchors_and_rejects_extrapolation():
    nodes = np.vstack((np.zeros(3), np.eye(3)))
    simplices = np.asarray(((0, 1, 2, 3),), dtype=int)
    anchor = locate_full_simplex(nodes, simplices, nodes[2])
    assert np.array_equal(anchor.weights, np.asarray((0.0, 0.0, 1.0, 0.0)))
    interior = locate_full_simplex(nodes, simplices, np.full(3, 0.2))
    assert np.min(interior.weights) >= 0.0
    assert interior.coordinate_reproduction_defect <= 1e-15
    with pytest.raises(ValueError, match="outside"):
        locate_full_simplex(nodes, simplices, np.asarray((2.0, 2.0, 2.0)))


def test_periodic_phase_weights_close_at_two_pi():
    nodes = np.linspace(0.0, 2.0 * np.pi, 5)
    left_indices, left_weights = periodic_phase_weights(nodes, 0.0)
    right_indices, right_weights = periodic_phase_weights(nodes, 2.0 * np.pi)
    assert np.array_equal(left_indices, right_indices)
    assert np.array_equal(left_weights, right_weights)


def test_guard_sheet_uses_five_vertices_and_signed_normal_distance():
    nodes = np.zeros((5, 5))
    nodes[1:, :4] = np.eye(4)
    simplices = np.asarray(((0, 1, 2, 3, 4),), dtype=int)
    normals = np.broadcast_to(np.asarray((0.0, 0.0, 0.0, 0.0, 1.0)), nodes.shape)
    query = np.asarray((0.1, 0.2, 0.15, 0.05, 0.03))
    location = locate_guard_sheet(nodes, simplices, normals, query)
    assert abs(location.signed_guard_distance - 0.03) <= 1e-15
    assert np.array_equal(location.oriented_normal, normals[0])
    assert location.affine_hull_reproduction_defect <= 1e-15
    with pytest.raises(ValueError, match="outside"):
        locate_guard_sheet(nodes, simplices, normals, np.asarray((1.0, 1.0, 1.0, 1.0, 0.0)))
