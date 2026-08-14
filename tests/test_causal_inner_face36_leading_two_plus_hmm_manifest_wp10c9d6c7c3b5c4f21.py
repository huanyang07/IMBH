import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "results/canonical"
    / "causal_inner_face36_leading_two_plus_hmm_manifest_"
    "wp10c9d6c7c3b5c4f21"
)


def _read(name: str) -> dict:
    return json.loads((ARTIFACT / name).read_text(encoding="utf-8"))


def test_c4f21_freezes_hybrid_state_without_running_a_trajectory():
    summary = _read("summary.json")
    assert summary["passed"]
    assert summary["definitions_only"]
    assert not summary["trajectory_executed"]
    assert summary["leading_two_state_coordinate_supported"]
    assert not summary["direct_two_mode_output_closure_supported"]
    assert not summary["six_mode_explicit_coordinate_supported"]
    assert summary["guard_HMM_required"]


def test_c4f21_leading_state_passes_while_weak_explicit_coordinate_fails():
    metrics = _read("summary.json")["architecture_metrics"]
    assert metrics["minimum_leading_block_projector_cosine"] >= 0.95
    assert metrics["leading_amplitude_history_cosine"] >= 0.95
    assert metrics["leading_amplitude_history_relative_difference"] <= 0.10
    assert metrics["minimum_full_six_mode_projector_cosine"] < 0.90
    assert metrics["weak_amplitude_history_relative_difference"] > 0.10
    assert metrics["leading_face36_history_relative_difference"] > 0.10


def test_c4f21_preserves_guard_and_binding_micro_output():
    manifest = _read("hybrid_architecture_manifest.json")
    state = manifest["hybrid_state"]
    interpretation = manifest["interpretation"]
    assert state["formal_state"] == "Q3_plus_a2_plus_Z_guard"
    assert state["slow_exchange"] == "certified_face36_exterior_partition"
    assert state["raw_face48_exchange_forbidden"]
    assert not interpretation["guard_complement_may_be_discarded"]
    assert interpretation["HMM_microstate_supplies_binding_face36_output"]
    assert not interpretation[
        "weak_vectors_are_individually_named_physical_modes"
    ]


def test_c4f21_freezes_descriptor_consistent_fixed_Q_preflight():
    manifest = _read("hybrid_architecture_manifest.json")
    constraint = manifest["fixed_Q_constraint"]
    preflight = manifest["authorized_analysis_only_preflight"]
    assert constraint["Q"] == "Q3_exterior_domain_M_J_E"
    assert constraint["a2_is_not_constrained"]
    assert constraint["constraint_reaction_must_annihilate_a2_dual"]
    assert constraint["record_multiplier_work_in_M_J_E_ledgers"]
    assert constraint["manual_primitive_freezing_forbidden"]
    assert preflight["screen_equal_Q_lifts_as_block_RHS"] == 24
    assert not preflight["new_nonlinear_trajectory"]
    assert not preflight["new_tangent_trajectory"]


def test_c4f21_cost_contract_avoids_brute_force_lifts_and_fine_runs():
    manifest = _read("hybrid_architecture_manifest.json")
    cost = manifest["cost_contract"]
    pilot = manifest["conditional_nonlinear_pilot_contract"]
    assert cost["analysis_only_preflight_wall_hours"] == [1.0, 3.0]
    assert cost["one_factorization_24_RHS"]
    assert cost["no_full_nonlinear_run_for_every_lift"]
    assert cost["no_automatic_fine_or_50ms_trajectory"]
    assert pilot["maximum_full_nonlinear_anchor_lifts"] == 2
    assert pilot["fine_residual_correction_then_short_shadow"]
    assert pilot["adaptive_microburst_windows_ms"] == [2.0, 5.0, 10.0, 20.0]


def test_c4f21_keeps_nonlinear_reduction_and_long_duration_work_blocked():
    summary = _read("summary.json")
    assert summary["fixed_Q_constraint_preflight_authorized"]
    assert not summary["fixed_Q_micro_solver_authorized"]
    assert not summary["nonlinear_retained_mode_pilot_authorized"]
    assert not summary["fifty_ms_propagation_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    assert not summary["physical_failure_detected"]
    assert summary["guard_complement_retained"]
    assert summary["raw_face48_export_rejection_preserved"]


def test_c4f21_metrics_and_hashes_are_self_consistent():
    with np.load(ARTIFACT / "architecture_metrics.npz", allow_pickle=False) as arrays:
        assert arrays["middle_leading_amplitude_history"].shape == (40, 2, 2)
        assert arrays["fine_leading_amplitude_history_aligned"].shape == (40, 2, 2)
        assert arrays["fine_complement_state_fractions"].shape == (40, 6)
    entries = {}
    for line in (ARTIFACT / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    for name, digest in entries.items():
        assert hashlib.sha256((ARTIFACT / name).read_bytes()).hexdigest() == digest
