from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for path in (SCRIPTS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_characteristic_phase_audit_wp10c9a as wp10c9a


def test_wp10c9a_machine_evidence_localizes_unresolved_rusanov_phase() -> None:
    if not wp10c9a.DEFAULT_OUTPUT.exists():
        return
    payload = json.loads(
        wp10c9a.DEFAULT_OUTPUT.read_text(encoding="utf-8")
    )

    assert payload["work_package"] == "WP10c9a"
    assert payload["classification"] == (
        "characteristic_rate_phase_unresolved_operator_redesign_required"
    )
    assert not payload["passed"]
    assert payload["method_contract"]["passed"]
    assert payload["method_contract"]["maximum_shared_flux_defect"] == 0.0
    assert (
        payload["method_contract"]["maximum_storage_action_defect"]
        <= wp10c9a.MAXIMUM_STORAGE_ACTION_DEFECT
    )
    assert (
        payload["method_contract"]["maximum_restart_relative_defect"]
        <= wp10c9a.MAXIMUM_RESTART_DEFECT
    )

    packets = payload["packet_results"]
    assert set(packets) == set(
        wp10c9a.CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES
    )
    for packet in packets.values():
        assert all(
            row["exact_reduced_descriptor_primitive_manifold"]
            and not row["retained_slow_coordinate_null_fiber_imposed"]
            and row["passed"]
            for row in packet["projection"].values()
        )
        assert packet["fine_minimum_signed_cosine"] >= (
            wp10c9a.MINIMUM_SIGNED_COSINE
        )

    decomposition = payload["phase_defect_decomposition"]
    assert decomposition["controlling_family"] == "inward_shear"
    assert decomposition["controlling_term"] == "rusanov_transport"
    assert decomposition["controlling_balance_block"] == (
        "mapped_storage_action"
    )
    for row in decomposition["by_family"].values():
        assert row["controlling_forcing_term"] == "rusanov_transport"

    candidates = payload["candidate_screen"]
    assert all(
        row["smooth_method_gate_passed"] for row in candidates.values()
    )
    assert not any(
        row["full_packet_phase_certified"] for row in candidates.values()
    )
    assert not candidates["horizon_rapidity_quadratic"][
        "meaningful_rate_coefficient_improvement"
    ]
    assert not candidates["characteristic_perturbation_quadratic"][
        "meaningful_rate_coefficient_improvement"
    ]

    decision = payload["decision"]
    assert not decision["passing_candidates"]
    assert not decision["rerun_common_mode_authorized"]
    assert not decision["bounded_nonlinear_patch_truth_authorized"]
    assert not decision["one_more_brute_force_refinement_authorized"]
    assert not decision["targeted_operator_implementation_authorized"]
    assert not decision["fixed_q_averaging_authorized"]
    assert not decision["reduced_model_authorized"]
    assert "Rusanov" in decision["next_operator_target"]
