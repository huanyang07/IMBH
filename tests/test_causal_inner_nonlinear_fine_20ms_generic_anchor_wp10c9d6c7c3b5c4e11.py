from __future__ import annotations

import hashlib
import json

import numpy as np

import run_causal_inner_nonlinear_fine_20ms_generic_anchor_wp10c9d6c7c3b5c4e11 as c4e11


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_parent_schedule_and_initial_histories_align():
    parent, fine_start = c4e11._parent_arrays()
    base = c4e11._base(parent)
    arrays = c4e11._initial_arrays(parent, fine_start)
    assert base["accepted_timesteps"].size == 39
    assert arrays["anchor_states"].shape == (1, 208, 5)
    assert np.array_equal(
        fine_start["base__accepted_states"][-1], base["accepted_states"][0]
    )
    assert arrays["anchor_previous_timesteps"].shape == (1,)


def test_only_three_prospective_temporal_audits_are_selected():
    parent, _ = c4e11._parent_arrays()
    indices = c4e11._audit_indices(c4e11._base(parent))
    times = c4e11._base(parent)["accepted_times"][1:]
    assert {_time for _time in (c4e11._time_us(times[i]) for i in indices)} == {
        8_000,
        14_000,
        20_000,
    }


def test_source_and_input_identities_are_complete():
    source = c4e11._source_identity()
    inputs = c4e11._input_identity()
    assert c4e11.THIS_RUNNER in source
    assert c4e11.THIS_TEST in source
    assert "fine_base_tangent_arrays" in inputs
    assert "fine_5ms_arrays" in inputs


def test_canonical_result_when_present():
    if not c4e11.SUMMARY_PATH.exists():
        return
    summary = _read(c4e11.SUMMARY_PATH)
    assert summary["passed"]
    assert summary["fine_generic_anchor_completed"]
    assert summary["final_three_grid_spatial_reanalysis_authorized"]
    assert not summary["fine_twenty_ms_spatial_certificate_issued"]
    assert not summary["fifty_ms_propagation_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    for line in (c4e11.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        assert _sha256(c4e11.CANONICAL_DIRECTORY / name) == expected
