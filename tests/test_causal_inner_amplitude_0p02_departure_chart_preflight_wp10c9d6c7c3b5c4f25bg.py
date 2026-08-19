from __future__ import annotations

import run_causal_inner_amplitude_0p02_departure_chart_preflight_wp10c9d6c7c3b5c4f25bg as f25bg


def test_frozen_amplitude_0p02_manifest_is_locked():
    frozen = f25bg._validate_manifest(require_clean=False)
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["authorized_next"] == f25bg.WORK_PACKAGE
    assert frozen["summary"]["maximum_scaled_component_bound"] == 2.0e-2


def test_retraction_family_remains_the_certified_eight_directions():
    metrics, arrays = f25bg.chart_tools._departure_family()
    assert arrays["energy_directions"].shape == (28, 8)
    assert arrays["departure_basis"].shape == (560, 28)
    assert metrics["departure_base_physical_tangency_defect"] <= 1.0e-10


def test_amplitude_0p02_gate_contract_remains_rate_free():
    gates = f25bg.manifest._contract()["binding_preflight_gates"]
    assert gates["completed_candidate_count_equal"] == 16
    assert gates["maximum_final_scaled_component"] == 2.0e-2
    assert gates["nonbase_continuous_rate_evaluations_equal"] == 0
    assert gates["propagated_states_equal"] == 0
