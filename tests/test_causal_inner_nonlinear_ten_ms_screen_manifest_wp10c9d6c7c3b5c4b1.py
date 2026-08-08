from __future__ import annotations

import json

import numpy as np

import run_causal_inner_nonlinear_ten_ms_screen_manifest_wp10c9d6c7c3b5c4b1 as c4b1


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_every_replay_and_strict_target_is_exact_main_target() -> None:
    manifest = _read(c4b1.MANIFEST_PATH)
    targets = manifest["canonical_targets"]
    main = np.asarray(targets["main_seconds"], dtype=np.float64)
    for name in ("replay", "strict", "pilot"):
        values = np.asarray(targets[f"{name}_seconds"], dtype=np.float64)
        assert set(values.view(np.uint64)).issubset(set(main.view(np.uint64)))
    assert targets["all_common_strict_outputs_binding"]


def test_pilot_seed_and_durable_resume_are_binding() -> None:
    manifest = _read(c4b1.MANIFEST_PATH)
    assert manifest["pilot_seed"]["no_new_BDF1_startup"]
    assert manifest["pilot_seed"][
        "complete_BDF2_history_reconstructed_deterministically_from_5p0_and_5p4ms_states"
    ]
    assert manifest["execution"]["durable_restart_and_arrays_after_every_target"]
    assert manifest["execution"]["resume_requires_exact_runner_manifest_and_input_hashes"]


def test_manifest_authorizes_only_ten_ms_screen() -> None:
    summary = _read(c4b1.SUMMARY_PATH)
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["ten_ms_screen_propagation_authorized"]
    assert not summary["twenty_ms_completion_manifest_authorized"]
    assert not summary["twenty_ms_propagation_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
