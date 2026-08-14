import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "results/canonical"
    / "causal_inner_face36_six_mode_fine_dynamic_coordinate_replay_"
    "wp10c9d6c7c3b5c4f20"
)


def _read(name: str) -> dict:
    return json.loads((ARTIFACT / name).read_text(encoding="utf-8"))


def test_c4f20_executes_only_the_fine_tangent_replay():
    summary = _read("summary.json")
    assert summary["fine_executed"]
    assert summary["new_tangent_trajectory"]
    assert not summary["new_nonlinear_trajectory"]
    assert not summary["middle_replayed"]
    assert not summary["fixed_Q_reaction_applied"]


def test_c4f20_fine_method_and_cross_grid_decision_are_consistent():
    summary = _read("summary.json")
    config = _read("config.json")
    fine = summary["fine"]
    assert fine["steps"] == 39
    assert fine["preflight_passed"]
    assert fine["maximum_JVP_defect"] <= config[
        "single_layout_method_gates"
    ]["maximum_step_matrix_JVP_relative_defect"]
    assert fine["maximum_face36_five_point_defect"] <= config[
        "face36_directional_JVP_contract"
    ]["maximum_relative_defect_at_each_selected_step"]
    assert fine["maximum_Q3_leakage"] <= config[
        "single_layout_method_gates"
    ]["maximum_Q3_leakage"]
    if summary["passed"]:
        assert fine["passed"]
        assert summary["cross_resolution"] is not None
        assert summary["classification"] == (
            "face36_six_mode_dynamic_coordinate_replay_certified_"
            "one_Q_manifest_authorized"
        )


def test_c4f20_commits_complete_fine_and_complement_histories():
    with np.load(ARTIFACT / "decisive_arrays.npz", allow_pickle=False) as arrays:
        assert arrays["times"].shape == (40,)
        assert arrays["fine_state_directions"].shape[:2] == (40, 6)
        assert arrays["fine_face36_outputs"].shape == (40, 6, 3)
        assert arrays["fine_amplitude_transitions"].shape == (40, 6, 6)
        assert arrays["fine_Q3_leakage"].shape == (40, 6)
        assert arrays["fine_complement_state_fractions"].shape == (40, 6)
        assert arrays["fine_complement_face36_fractions"].shape == (40, 6)
        assert np.count_nonzero(
            np.isfinite(arrays["fine_face36_five_point_defects"])
        ) == 5 * 6 * 2


def test_c4f20_preserves_hard_stops_and_hashes():
    summary = _read("summary.json")
    assert not summary["fixed_Q_micro_solver_authorized"]
    assert not summary["nonlinear_retained_mode_pilot_authorized"]
    assert not summary["fifty_ms_propagation_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    assert not summary["physical_failure_detected"]
    assert summary["guard_complement_retained"]
    assert summary["raw_face48_export_rejection_preserved"]
    entries = {}
    for line in (ARTIFACT / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    for name, digest in entries.items():
        assert hashlib.sha256((ARTIFACT / name).read_bytes()).hexdigest() == digest
