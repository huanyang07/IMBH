from __future__ import annotations

import numpy as np

import run_causal_inner_active8_projective_kernel_geometry_wp10c9d6c7c3b5c4f25bs as f25bs


def test_manifest_is_frozen_and_authorizes_geometry_only():
    frozen = f25bs._validate_manifest(require_clean=False)
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["definitions_only"]
    assert frozen["summary"]["authorized_next"] == f25bs.WORK_PACKAGE
    assert frozen["summary"]["new_truth_evaluations"] == 0


def test_candidate_specifications_match_frozen_order_and_count():
    specifications = f25bs._candidate_specifications()
    assert len(specifications) == 24
    assert [item["split"] for item in specifications[:16]] == [
        "holdout_high"
    ] * 16
    assert [item["split"] for item in specifications[16:]] == [
        "holdout_low"
    ] * 8
    assert all(item["active_direction"].shape == (8,) for item in specifications)
    assert all(
        np.isclose(np.linalg.norm(item["active_direction"]), 1.0)
        for item in specifications
    )
    assert specifications[0]["component_bound"] == 1.0e-2
    assert specifications[-1]["component_bound"] == 5.0e-3


def test_retraction_contract_preserves_binding_geometry_gates():
    contract = f25bs._retraction_contract()
    assert contract["binding_preflight_gates"] == contract["binding_geometry_gates"]
    assert contract["binding_geometry_gates"][
        "completed_candidate_count_equal"
    ] == 48
    assert contract["binding_geometry_gates"][
        "nonbase_continuous_rate_evaluations_equal"
    ] == 0


def test_engine_adapter_uses_new_manifest_and_scratch():
    engine = f25bs._fresh_engine()
    assert engine.WORK_PACKAGE == f25bs.WORK_PACKAGE
    assert engine.manifest.PLANNED_CANDIDATES == 48
    assert engine.SCRATCH_DIRECTORY == f25bs.SCRATCH_DIRECTORY
    assert engine._candidate_specifications()[0]["split"] == "holdout_high"
