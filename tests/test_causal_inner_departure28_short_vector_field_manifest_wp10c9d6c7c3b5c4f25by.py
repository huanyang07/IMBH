from __future__ import annotations

import run_causal_inner_departure28_short_vector_field_manifest_wp10c9d6c7c3b5c4f25by as f25by


def test_parent_authorizes_only_short_vector_field_manifest():
    frozen = f25by._validate_parent(require_clean=False)
    assert frozen["parent_summary"]["passed"]
    assert frozen["parent_summary"]["classification"] == (
        "departure28_dual_polynomial_rate_and_rank4_decoder_"
        "independently_validated"
    )
    assert frozen["parent_summary"]["authorized_next"] == (
        "definitions_only_departure28_short_reduced_vector_field_"
        "validation_manifest"
    )


def test_contract_freezes_consistent_fast_transient_vector_field():
    contract = f25by._contract()
    assert contract["state"]["partition"] == "y_equals_q162_plus_z280_plus_a28"
    assert contract["state"]["dimension"] == 470
    assert contract["architecture_role"]["model_470"] == (
        "offline_fast_transient_and_closure_model"
    )
    assert contract["architecture_role"][
        "model_470_is_not_the_final_cycle_integrator"
    ]
    assert contract["algebraic_decoder"][
        "online_Newton_retractions_per_field_evaluation"
    ] == 0
    assert contract["reduced_vector_field"][
        "online_truth_calls_per_field_evaluation"
    ] == 0
    assert contract["reduced_vector_field"]["physical_rate"] == (
        "q_dot_equals_D_Cphys_at_u_hat_times_r_hat"
    )


def test_forecast_is_locked_before_one_new_truth_root():
    reference = f25by._contract()["reference_sequence"]
    assert reference["readiness_interval"] == (
        "accepted_warm_2_to_accepted_warm_3"
    )
    assert reference["readiness_is_retrospective_and_cannot_by_itself_certify_forecasting"]
    assert reference["prospective_forecast"] == "accepted_warm_3_to_new_warm_4"
    assert reference["forecast_must_be_serialized_and_hashed_before_truth_root"]
    assert reference["truth_predictor"] == (
        "accepted_history_predictor_not_reduced_forecast"
    )


def test_reference_checkpoints_are_accepted_and_locked():
    frozen = f25by._validate_parent(require_clean=False)
    assert frozen["retry_summary"]["accepted_main_BDF2_roots"] == 4
    assert frozen["retry_summary"]["rejected_main_BDF2_roots"] == 0
    inputs = f25by._decisive_inputs()
    assert all(path.is_file() for path in inputs.values())
    assert set(inputs) >= {
        "accepted_warm_2_checkpoint",
        "accepted_warm_3_checkpoint",
        "complete_generator",
        "frozen_coefficients",
    }


def test_decision_stops_before_cycle_evolution():
    contract = f25by._contract()
    assert f25by.AUTHORIZED_NEXT == "WP10c9d6c7c3b5c4f25bz"
    assert contract["decision"]["pass_authorizes_only"] == (
        "definitions_only_fixed_Q_fast_attractor_and_normal_hyperbolicity_manifest"
    )
    assert contract["decision"]["physical_microburst_authorized"] is False
    assert contract["decision"]["predictive_cycle_authorized"] is False
    assert contract["decision"]["reduced_slow_evolution_authorized"] is False
