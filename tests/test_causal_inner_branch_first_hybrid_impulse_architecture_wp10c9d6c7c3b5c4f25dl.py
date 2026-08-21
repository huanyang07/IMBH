from __future__ import annotations

import run_causal_inner_branch_first_hybrid_impulse_architecture_wp10c9d6c7c3b5c4f25dl as f25dl


def test_rank16_parent_authorizes_architecture_only() -> None:
    frozen = f25dl._validate_parent(require_clean=False)
    assert frozen["parent_hashes"]
    assert frozen["rank8_invariance_defect"] > 0.1
    assert frozen["rank12_invariance_defect"] > 0.1
    assert frozen["rank16_invariance_defect"] <= 0.1


def test_online_architecture_has_no_transition_microintegration() -> None:
    architecture = f25dl._architecture()
    online = architecture["mathematical_architecture"]
    runtime = architecture["online_runtime_contract"]
    assert not online["online_transition_ODE"]
    assert online["online_exact_truth_calls"] == 0
    assert online["online_fast_microsteps"] == 0
    assert online["offline_full470_fallback_required"]
    assert not runtime["full_transition_microintegration_online"]
    assert runtime["minimum_average_macrostep_seconds"] >= 5.7888


def test_branch_certification_precedes_transition_sampling() -> None:
    architecture = f25dl._architecture()
    branch = architecture["branch_first_dependency"]
    sampling = architecture["later_offline_transition_sampling"]
    assert branch["required_branch_labels"] == ["cold", "hot"]
    assert branch["branch_candidate_must_not_be_the_exact_20ms_transition_anchor"]
    assert branch["branch_certification_gates"][
        "fast_to_effective_slow_spectral_gap_ratio_min"
    ] == 10.0
    assert not sampling["authorized_in_this_package"]
    assert sampling["prerequisite"] == "both_branch_certificates_and_entry_exit_sections"


def test_next_screen_is_saved_revealed_arrays_only() -> None:
    architecture = f25dl._architecture()
    screen = architecture["prospective_branch_candidate_screen"]
    boundary = architecture["authorization_boundaries"]
    assert screen["truth_policy"] == "saved_revealed_arrays_only"
    assert screen["new_exact_fixed_Q_rate_calls_equal"] == 0
    assert screen["new_complete_generator_assemblies_equal"] == 0
    assert screen["new_nonlinear_roots_equal"] == 0
    assert screen["propagated_states_equal"] == 0
    assert screen["sealed_16ms_truth_calls_equal"] == 0
    assert boundary["branch_candidate_saved_array_screen_authorized"]
    assert not boundary["branch_truth_execution_authorized"]
    assert not boundary["transition_truth_campaign_authorized"]
    assert not boundary["reduced_slow_evolution_authorized"]


def test_canonical_architecture_if_present() -> None:
    if not f25dl.CANONICAL_DIRECTORY.exists():
        return
    f25dl._checksums(f25dl.CANONICAL_DIRECTORY)
    summary = f25dl._read(f25dl.CANONICAL_DIRECTORY / "summary.json")
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["selected_transition_internal_rank"] == 16
    assert summary["branch_first_execution_order_frozen"]
    assert summary["new_exact_fixed_Q_rate_calls"] == 0
    assert summary["new_complete_generator_assemblies"] == 0
    assert summary["new_nonlinear_roots"] == 0
    assert summary["propagated_states"] == 0
    assert not summary["sealed_16ms_opened"]
    assert summary["authorized_next"] == f25dl.AUTHORIZED_NEXT
