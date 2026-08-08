from __future__ import annotations

import json

import numpy as np

import run_causal_inner_nonlinear_5ms_cumulative_extraction_recovery_audit_wp10c9d6c7c3b5c3h2j1 as audit


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_selection_follows_innermost_two_consecutive_rule() -> None:
    summary = _read(audit.SUMMARY_PATH)
    faces = summary["analysis"]["faces"]
    expected = None
    for index in range(len(faces) - 1):
        if faces[index]["passed"] and faces[index + 1]["passed"]:
            expected = faces[index]["coarse_face_index"]
            break
    assert summary["analysis"]["selected_coarse_face_index"] == expected
    assert summary["passed"] is (expected is not None and summary["analysis"]["ledger_audits"]["passed"])


def test_selected_surface_passes_both_full_window_channels() -> None:
    summary = _read(audit.SUMMARY_PATH)
    if not summary["passed"]:
        return
    selected = summary["analysis"]["selected_coarse_face_index"]
    face = next(item for item in summary["analysis"]["faces"] if item["coarse_face_index"] == selected)
    assert face["instantaneous"]["temporal_classification"]["passed"]
    assert face["cumulative"]["temporal_classification"]["passed"]
    assert tuple(summary["analysis"]["binding_window_seconds"]) == (0.002, 0.005)


def test_raw_horizon_claims_remain_blocked() -> None:
    summary = _read(audit.SUMMARY_PATH)
    assert not summary["raw_inner_face_spatial_convergence_certified"]
    assert not summary["pointwise_horizon_flux_convergence_certified"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_decisive_histories_cover_every_surface() -> None:
    with np.load(audit.DECISIVE_ARRAYS, allow_pickle=False) as payload:
        times = payload["times_seconds"]
        for layout in audit.LAYOUTS:
            assert payload[f"{layout}__response"].shape == (times.size, len(audit.FACES), 13)
