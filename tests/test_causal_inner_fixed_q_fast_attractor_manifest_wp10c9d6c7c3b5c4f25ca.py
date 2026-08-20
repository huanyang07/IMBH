from __future__ import annotations

import numpy as np

import run_causal_inner_fixed_q_fast_attractor_manifest_wp10c9d6c7c3b5c4f25ca as f25ca


def test_parent_authorizes_only_fast_attractor_manifest():
    frozen = f25ca._validate_parent(require_clean=False)
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["authorized_next"] == (
        "definitions_only_fixed_Q_fast_attractor_and_normal_hyperbolicity_manifest"
    )
    assert frozen["summary"]["model_470_role"] == (
        "offline_fast_transient_and_closure_model"
    )


def test_split_eliminates_only_certified_stable_memory():
    split = f25ca._contract()["mathematical_split"]
    assert split["active_physical_q_dimension"] == 162
    assert split["strictly_stable_memory_z_dimension"] == 280
    assert split["nonlinear_departure_a_dimension"] == 28
    assert split["z_is_the_only_block_authorized_for_linear_elimination"]
    assert split["a_remains_explicitly_nonlinear"]
    assert split["naive_96_slow_plus_374_fast_split_authorized"] is False


def test_search_design_is_deterministic_and_inside_chart():
    first = f25ca._search_design()
    second = f25ca._search_design()
    assert np.array_equal(first, second)
    assert first.shape == (13, 28)
    assert np.array_equal(first[0], np.zeros(28))
    assert np.max(np.abs(first)) == max(f25ca.SEARCH_AMPLITUDES)
    assert np.max(np.abs(first)) < f25ca.DEPARTURE_COMPONENT_BOUND


def test_decision_branches_do_not_authorize_cycle():
    decision = f25ca._contract()["decision"]
    assert decision["stable_graph"]["authorizes_only"] == (
        "definitions_only_local_fast_graph_continuation_and_"
        "slow_flux_closure_manifest"
    )
    assert decision["clear_nonclosure"]["authorizes_only"] == (
        "definitions_only_guarded_departure_amplitude_expansion_manifest"
    )
    assert decision["physical_microburst_authorized"] is False
    assert decision["predictive_cycle_authorized"] is False
    assert decision["reduced_slow_evolution_authorized"] is False


def test_final_cycle_boundary_requires_multi_anchor_conservative_closure():
    boundary = f25ca._contract()["final_cycle_architecture_boundary"]
    assert boundary["direct_microsecond_marching"] is False
    assert boundary["single_anchor_470_field_is_final_cycle_model"] is False
    assert boundary["target_maximum_cycle_macrosteps"] == 100_000
    assert boundary["required_end_state"].startswith("multi_anchor_conservative_q162")
