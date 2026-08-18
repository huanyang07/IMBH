from __future__ import annotations

import numpy as np

import run_causal_inner_square_root_transfer_seeded_audit_wp10c9d6c7c3b5c4f25aa as f25aa


def test_manifest_is_locked_and_truth_work_is_forbidden(monkeypatch):
    for name, value in f25aa.THREAD_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    frozen = f25aa._validate_manifest(require_clean=False)
    budget = frozen["contract"]["execution_budget"]
    assert budget["allowed_new_nonlinear_roots"] == 0
    assert budget["allowed_propagated_states"] == 0
    assert budget["allowed_new_full_560_direction_generator_assemblies"] == 0


def test_square_root_coordinates_make_conservative_pair_exact_and_stable():
    generator = np.diag((-1.0, -2.0, -3.0, -4.0))
    restriction = np.asarray(
        ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0))
    )
    output = np.asarray(((0.0, 0.0, 1.0, 1.0),))
    system, metrics = f25aa._square_root_stable_system(
        generator,
        output,
        restriction,
        np.empty((4, 0)),
        np.empty((0, 4)),
        np.empty((0, 0)),
    )
    assert metrics["square_root_reconstruction_relative_defect"] < 1.0e-14
    assert metrics["whitened_Lyapunov_relative_defect"] < 1.0e-12
    assert metrics["conservative_lift_identity_defect"] < 1.0e-12
    assert metrics["full_trial_test_biorthogonality_defect"] < 1.0e-12
    assert metrics["full_coordinate_reconstruction_relative_defect"] < 1.0e-12
    assert metrics["full_stable_spectral_abscissa_per_second"] < 0.0
    assert system["hidden_basis"].shape == (4, 2)


def test_balanced_trial_uses_frozen_hankel_ordering():
    balanced = {
        "hankel_singular_values": np.asarray((4.0, 1.0)),
        "controllability_factor": np.eye(2),
        "hankel_right_vectors_transpose": np.eye(2),
    }
    trial = f25aa._balanced_trial(balanced, 2)
    np.testing.assert_allclose(trial, np.diag((0.5, 1.0)))


def test_candidate_score_is_fail_closed_for_zero_cross_anchor_overlap():
    gates = {
        "cross_anchor_hidden_principal_cosine_min": 0.5,
        "trial_test_biorthogonality_defect_max": 1.0,
        "reduced_Lyapunov_identity_relative_defect_max": 1.0,
        "resolved_self_energy": {
            "RMS_normalized_dynamic_transfer_relative_error_max": 1.0
        },
        "conservative_face_flux": {
            "RMS_normalized_dynamic_transfer_relative_error_max": 1.0
        },
    }
    anchor = {
        "trial_test_biorthogonality_defect": 0.0,
        "reduced_Lyapunov_identity_relative_defect": 0.0,
        "blocks": {
            label: {
                f"{prefix}_RMS_normalized_dynamic_transfer_relative_error": 0.0
                for prefix in ("training", "heldout")
            }
            for label in ("resolved_self_energy", "conservative_face_flux")
        },
    }
    item = {
        "cross_anchor_hidden_principal_cosine_min": 0.0,
        "primary": anchor,
        "heldout": anchor,
    }
    assert f25aa._candidate_score(item, gates) > 1.0e100
