from __future__ import annotations

import json

import numpy as np

import run_causal_inner_nonlinear_twenty_ms_completion_manifest_wp10c9d6c7c3b5c4c as c4c


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_all_auxiliary_targets_are_exact_main_targets() -> None:
    manifest = _read(c4c.MANIFEST_PATH)
    targets = manifest["canonical_targets"]
    main = np.asarray(targets["main_seconds"], dtype=np.float64)
    for name in ("replay", "strict"):
        values = np.asarray(targets[f"{name}_seconds"], dtype=np.float64)
        assert set(values.view(np.uint64)).issubset(set(main.view(np.uint64)))
    assert targets["all_common_strict_outputs_binding"]


def test_restart_seeds_are_canonical_and_hashed() -> None:
    manifest = _read(c4c.MANIFEST_PATH)
    provenance = _read(c4c.PROVENANCE_PATH)
    assert manifest["rung"]["no_new_BDF1_startup"]
    assert manifest["initial_restarts"]["copied_into_canonical_evidence"]
    assert c4c._sha256(c4c.BASE_RESTART_PATH) == provenance[
        "seed_restart_sha256"
    ]["base"]
    assert c4c._sha256(c4c.PERTURBED_RESTART_PATH) == provenance[
        "seed_restart_sha256"
    ]["perturbed"]


def test_runtime_projection_and_durable_execution_are_binding() -> None:
    manifest = _read(c4c.MANIFEST_PATH)
    projection = manifest["runtime_projection"]
    assert projection["raw_projected_wall_hours"] > 0.0
    assert projection["projected_wall_hours_with_safety"] > projection[
        "raw_projected_wall_hours"
    ]
    assert projection["runtime_is_advisory_not_scientific"]
    assert manifest["execution"]["durable_restart_and_arrays_after_every_target"]
    assert manifest["execution"]["stop_on_first_scientific_failure"]


def test_manifest_authorizes_only_twenty_ms_propagation() -> None:
    summary = _read(c4c.SUMMARY_PATH)
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["ten_ms_screen_certified"]
    assert summary["twenty_ms_completion_manifest_authorized"]
    assert summary["twenty_ms_propagation_authorized"]
    assert not summary["twenty_ms_checkpoint_assessment_authorized"]
    assert not summary["fifty_ms_propagation_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
