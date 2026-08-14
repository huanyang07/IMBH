import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "results/canonical"
    / "causal_inner_face36_six_mode_numerical_audit_recovery_wp10c9d6c7c3b5c4f18"
)


def _read(name: str) -> dict:
    return json.loads((ARTIFACT / name).read_text(encoding="utf-8"))


def test_c4f18_uses_stable_duals_and_frozen_plateau_rule():
    summary = _read("summary.json")
    config = _read("config.json")
    assert config["dual_gate"] == 1.0e-10
    assert config["face36_output_map_gate"] == 1.0e-8
    assert not config["tolerance_relaxation"]
    assert summary["dual_QR_passed"]
    assert summary["dual_SVD_passed"]
    assert summary["dual_recovery_passed"]
    assert summary["QR_metrics"]["biorthogonality_defect"] <= 1.0e-10
    assert summary["SVD_metrics"][
        "normalized_slow_lift_annihilation_defect"
    ] <= 1.0e-10


def test_c4f18_classification_matches_plateau_result():
    summary = _read("summary.json")
    if summary["face36_directional_JVP_plateau_passed"]:
        assert summary["passed"]
        assert summary["saved_middle_history_reclassified"]
        assert summary["selected_adjacent_step_pair"] is not None
        assert summary["classification"] == (
            "middle_six_mode_numerical_audits_recovered_saved_dynamic_"
            "history_reclassified_fine_manifest_authorized"
        )
    else:
        assert not summary["passed"]
        assert not summary["saved_middle_history_reclassified"]
        assert summary["classification"] == (
            "face36_directional_JVP_audit_failed_derivative_localization_required"
        )


def test_c4f18_arrays_cover_all_times_directions_and_steps():
    with np.load(ARTIFACT / "decisive_arrays.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(
            arrays["time_ids_microseconds"], [5000, 5400, 10000, 16000, 20000]
        )
        assert arrays["central_relative_defects"].shape == (5, 6, 6)
        assert arrays["five_point_relative_defects"].shape == (5, 6, 6)
        assert arrays["dual_QR"].shape[0] == 6
        assert arrays["dual_SVD"].shape == arrays["dual_QR"].shape


def test_c4f18_hashes_and_hard_stops_are_preserved():
    entries = {}
    for line in (ARTIFACT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    for name, digest in entries.items():
        assert hashlib.sha256((ARTIFACT / name).read_bytes()).hexdigest() == digest
    summary = _read("summary.json")
    assert not summary["new_tangent_trajectory"]
    assert not summary["new_nonlinear_trajectory"]
    assert not summary["fine_executed"]
    assert not summary["fixed_Q_reaction_applied"]
    assert not summary["fixed_Q_micro_solver_authorized"]
    assert not summary["nonlinear_retained_mode_pilot_authorized"]
    assert not summary["fifty_ms_propagation_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
