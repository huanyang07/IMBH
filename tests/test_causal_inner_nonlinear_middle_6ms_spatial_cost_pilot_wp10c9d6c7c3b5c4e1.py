from __future__ import annotations

import json

import numpy as np

import run_causal_inner_nonlinear_middle_6ms_spatial_cost_pilot_wp10c9d6c7c3b5c4e1 as c4e1


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_middle_pilot_passes_method_surrogate_and_extraction_gates() -> None:
    summary = _read(c4e1.SUMMARY_PATH)
    assert summary["passed"]
    assert summary["scientific_gates_passed"]
    assert summary["cost_review_passed"]
    assert summary["base"]["passed"]
    assert summary["tangent"]["passed"]
    assert summary["anchor"]["passed"]
    assert summary["surrogate_relative_gate_passed"]
    assert summary["extraction_partition"]["passed"]


def test_pilot_starts_from_canonical_five_ms_histories() -> None:
    with np.load(c4e1.DECISIVE_ARRAYS) as arrays, np.load(
        c4e1.middle5.DECISIVE_ARRAYS
    ) as parent:
        assert np.array_equal(
            arrays["base__accepted_states"][0], parent["base__accepted_states"][-1]
        )
        assert np.array_equal(
            arrays["anchor__anchor_states"][0],
            parent["anchor__anchor_states"][-1],
        )
        assert arrays["base__accepted_times"][0] == 5.0e-3
        assert arrays["base__accepted_times"][-1] == 6.0e-3


def test_pilot_replay_and_authorization_are_narrow() -> None:
    summary = _read(c4e1.SUMMARY_PATH)
    assert all(
        item["checkpoint_roundtrip_bitwise"]
        and item["last_step_replay_bitwise"]
        and item["maximum_scaled_residual"] <= 1.0e-10
        for item in summary["serialized_replays"].values()
    )
    assert summary["middle_twenty_ms_completion_manifest_authorized"]
    assert not summary["middle_twenty_ms_propagation_authorized"]
    assert not summary["fine_twenty_ms_propagation_authorized"]
    assert not summary["twenty_ms_spatial_checkpoint_certified"]
    assert not summary["fifty_ms_propagation_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
