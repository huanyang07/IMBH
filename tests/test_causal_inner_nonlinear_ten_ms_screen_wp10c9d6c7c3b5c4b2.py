from __future__ import annotations

import numpy as np

import run_causal_inner_monolithic_bdf_base_preflight_wp10c9d6c7c3b1a as c3b1a
import run_causal_inner_nonlinear_second_duration_rung_wp10c9d6c7c3b5c2 as c2
import run_causal_inner_nonlinear_ten_ms_screen_manifest_wp10c9d6c7c3b5c4b1 as c4b1
import run_causal_inner_nonlinear_ten_ms_screen_wp10c9d6c7c3b5c4b2 as c4b2


def test_pilot_seed_reconstructs_complete_bdf2_history() -> None:
    context = c3b1a._configurations()[c2.LAYOUT]["context"]
    for name in ("base", "perturbed"):
        restart, arrays = c4b2._pilot_seed(name, context)
        assert restart.elapsed_time_seconds == arrays["output_times"][-1]
        assert restart.next_order == 2
        assert restart.history.previous_timestep_seconds == arrays[
            "accepted_timesteps"
        ][0]
        assert restart.primitive_charts.shape == (64, 5)
        assert arrays["output_states"].shape == (2, 64, 5)
        assert arrays["output_extraction_partition"].shape == (2, 13)


def test_stage_and_target_contracts_are_exact() -> None:
    assert c4b2.STAGE_ORDER == (
        "base_main",
        "perturbed_main",
        "base_replay",
        "perturbed_replay",
        "base_strict",
        "perturbed_strict",
    )
    assert set(c4b2.REPLAY_TARGET_MICROSECONDS).issubset(
        set(c4b1.MASTER_TARGET_MICROSECONDS)
    )
    assert set(c4b2.STRICT_TARGET_MICROSECONDS).issubset(
        set(c4b1.MASTER_TARGET_MICROSECONDS)
    )


def test_transport_sign_diagnostic_is_not_a_binding_audit() -> None:
    audits = np.zeros((2, 7), dtype=float)
    audits[:, 5] = 2.0
    gates = {
        "maximum_shared_conservative_face_defect": 1.0e-12,
        "maximum_incoming_excision_characteristics": 0,
        "maximum_exterior_prefix_identity_defect": 1.0e-12,
    }
    assert c4b2._audit_passed(audits, gates)
