import json

import numpy as np
import pytest

import run_causal_inner_native_grid_five_to_eleven_field_lift_and_prefix_coverage_audit_wp10c9d6c7c3b5c4f25fizzq2 as runner


def test_native_lift_embeds_only_old_Rphi_stress():
    charts=np.asarray(((5.0,-0.2,0.6,14.5,2e-4),(6.0,-0.3,0.7,15.0,-3e-4)))
    lifted=runner._lift_anchor_states(charts)
    assert lifted.shape==(2,11)
    assert np.array_equal(lifted[:,:4],np.zeros((2,4)))
    assert np.array_equal(lifted[:,4:6],np.zeros((2,2)))
    assert np.allclose(lifted[:,6]/np.sqrt(2.0),charts[:,4],rtol=0,atol=1e-19)
    assert np.array_equal(lifted[:,7:],np.zeros((2,4)))


def test_cellwise_cover_is_native_and_deterministic():
    trajectory,witnesses=runner._load_prefix();profiles=np.concatenate((trajectory,witnesses),axis=0)
    cells,indices,nearest,_=runner._select_cellwise_cover(profiles)
    assert len(cells)==len(indices)==913
    assert set(cells)==set(range(112))
    assert np.max(nearest)<=1.0+1e-12


@pytest.mark.skipif(not runner.CANONICAL_DIRECTORY.exists(),reason="audit not executed")
def test_canonical_lift_and_prefix_plan_closes():
    assert runner._u()._validate_checksums(runner.CANONICAL_DIRECTORY)
    summary=json.loads((runner.CANONICAL_DIRECTORY/"summary.json").read_text())
    assert summary["passed"] and summary["native_five_to_eleven_lift_certified"]
    assert summary["prefix_candidate_anchor_count"]==913
    assert not summary["prefix_coefficient_payloads_built"]
    assert not summary["cycle_wide_inputs_complete"]
    assert summary["complete_cycle_steps"]==0
