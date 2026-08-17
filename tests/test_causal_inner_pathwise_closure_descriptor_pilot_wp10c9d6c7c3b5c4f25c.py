from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

import run_causal_inner_pathwise_closure_descriptor_pilot_wp10c9d6c7c3b5c4f25c as f25c


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_manifest_authorizes_only_the_nonpropagating_pilot(monkeypatch):
    for name, value in f25c.THREAD_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    frozen = f25c._validate_manifest(require_clean=False)
    assert frozen["summary"]["single_anchor_descriptor_pilot_authorized"]
    assert not frozen["summary"]["full_anchor_campaign_authorized"]
    assert frozen["pilot"]["allowed_new_nonlinear_roots"] == 0
    assert frozen["pilot"]["allowed_exact_continuous_descriptor_assemblies"] == 1


def test_coarse_groups_cover_truth_grid_and_keep_face36():
    groups = f25c._coarse_groups(112)
    assert len(groups) == 16
    assert groups[0][0] == 0
    assert groups[-1][1] == 112
    assert groups[7][1] == groups[8][0] == 72
    boundaries = f25c._coarse_boundaries(112)
    assert len(boundaries) == 17
    assert boundaries[8] == 72


def test_incidence_telescopes_to_two_boundaries_exactly():
    incidence = f25c._incidence_matrix()
    total = np.sum(incidence, axis=0)
    expected = np.zeros(17)
    expected[0] = 1.0
    expected[-1] = -1.0
    assert np.array_equal(total, expected)


def test_continuous_generator_includes_constraint_and_lift_derivatives():
    free_generator = np.asarray(((0.2, -0.1), (0.3, -0.4)))
    free_rate = np.asarray((0.7, -0.2))
    constraint = np.asarray(((1.0, 0.0),))
    lift = np.asarray(((1.0,), (0.0,)))
    d_constraint = np.asarray((((0.1, -0.2),), ((-0.3, 0.4),)))
    d_lift = np.asarray((((0.05,), (-0.02,)), ((-0.04,), (0.03,))))
    complete, rate, multiplier_jacobian = f25c._continuous_generator(
        free_generator, free_rate, constraint, lift, d_constraint, d_lift
    )
    multiplier = -constraint @ free_rate
    expected = np.empty_like(free_generator)
    for column in range(2):
        d_multiplier = -(
            d_constraint[column] @ free_rate
            + constraint @ free_generator[:, column]
        )
        expected[:, column] = (
            free_generator[:, column]
            + d_lift[column] @ multiplier
            + lift @ d_multiplier
        )
    assert np.allclose(complete, expected)
    assert np.allclose(rate, free_rate + lift @ multiplier)
    assert multiplier_jacobian.shape == (1, 2)


def test_resolved_projection_builds_identity_and_orthogonal_complement():
    restriction = np.asarray(((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)))
    dual = np.asarray(((0.0, 0.0, 1.0, 0.0),))
    resolved, lifting, complement, metrics = f25c._resolved_projection(
        restriction, dual
    )
    assert np.allclose(resolved @ lifting, np.eye(3))
    assert np.allclose(resolved @ complement, 0.0)
    assert metrics["resolved_rank"] == 3
    assert metrics["unresolved_dimension"] == 1


def test_complex_schur_transfer_closes_and_is_conjugate_symmetric():
    generator = np.asarray(((-2.0, 0.5, 0.0), (0.0, -3.0, 0.2), (0.0, 0.0, -4.0)))
    lifting = np.asarray(((1.0,), (0.0,), (0.0,)))
    complement = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    output = np.asarray(((1.0, 2.0, -1.0),))
    transfer, poles, metrics = f25c._transfer_from_schur(
        generator, lifting, complement, output, np.asarray((0.1, 1.0, 10.0))
    )
    assert transfer.shape == (4, 1, 1)
    assert poles.shape == (2,)
    assert metrics["maximum_frequency_solve_relative_residual"] < 1.0e-12
    assert metrics["maximum_transfer_conjugate_symmetry_relative_defect"] < 1.0e-12
    assert metrics["unstable_unresolved_pole_count_diagnostic"] == 0


def test_canonical_result_when_available():
    summary_path = f25c.CANONICAL_DIRECTORY / "summary.json"
    if not summary_path.exists():
        pytest.skip("descriptor pilot not canonicalized yet")
    summary = _read(summary_path)
    assert summary["pilot_executed"]
    assert summary["new_nonlinear_roots"] == 0
    assert summary["propagated_states"] == 0
    assert not summary["full_anchor_campaign_authorized"]
    assert not summary["online_reduced_solver_implementation_authorized"]
    for line in (f25c.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        actual = hashlib.sha256((f25c.CANONICAL_DIRECTORY / name).read_bytes()).hexdigest()
        assert actual == expected
