from __future__ import annotations

import json

import numpy as np

import run_causal_inner_nonlinear_twenty_ms_completion_manifest_wp10c9d6c7c3b5c4c as c4c
import run_causal_inner_nonlinear_twenty_ms_completion_wp10c9d6c7c3b5c4c1 as c4c1


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_execution_uses_manifest_owned_targets() -> None:
    assert np.array_equal(
        c4c1.REPLAY_TARGET_MICROSECONDS,
        c4c.MASTER_TARGET_MICROSECONDS[c4c.REPLAY_TARGET_INDICES],
    )
    assert np.array_equal(
        c4c1.STRICT_TARGET_MICROSECONDS,
        c4c.MASTER_TARGET_MICROSECONDS[c4c.STRICT_TARGET_INDICES],
    )
    assert c4c1.STAGE_ORDER == (
        "base_main",
        "perturbed_main",
        "base_replay",
        "perturbed_replay",
        "base_strict",
        "perturbed_strict",
    )


def test_source_identity_binds_runner_manifest_seeds_and_parent() -> None:
    identity = c4c1._source_identity()
    assert set(identity) == {
        "runner",
        "manifest",
        "base_seed_restart",
        "perturbed_seed_restart",
        "ten_ms_arrays",
        "ten_ms_summary",
    }
    assert all(len(value) == 64 for value in identity.values())


def test_parent_authorizes_only_twenty_ms_execution() -> None:
    parent, manifest = c4c1._validate_parent()
    assert parent["twenty_ms_propagation_authorized"]
    assert not parent["twenty_ms_checkpoint_assessment_authorized"]
    assert not parent["fifty_ms_propagation_authorized"]
    assert manifest["execution"]["durable_restart_and_arrays_after_every_target"]
    assert manifest["execution"]["stop_on_first_scientific_failure"]


def test_progress_identity_is_durable() -> None:
    names = (
        "WORK_PACKAGE",
        "_source_identity",
    )
    previous = {name: getattr(c4c1.engine, name) for name in names}
    try:
        c4c1.engine.WORK_PACKAGE = c4c1.WORK_PACKAGE
        c4c1.engine._source_identity = c4c1._source_identity
        progress = c4c1.engine._new_progress("base_main", 10_000, 4.0e-4)
        assert progress["work_package"] == c4c1.WORK_PACKAGE
        assert progress["current_target_microseconds"] == 10_000
        assert progress["source_identity"] == c4c1._source_identity()
        assert not progress["complete"]
    finally:
        for name, value in previous.items():
            setattr(c4c1.engine, name, value)
