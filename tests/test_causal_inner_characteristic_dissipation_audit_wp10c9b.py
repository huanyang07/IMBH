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

import run_causal_inner_characteristic_dissipation_audit_wp10c9b as wp10c9b


def test_wp10c9b_machine_evidence_rejects_matrix_dissipation() -> None:
    if not wp10c9b.DEFAULT_OUTPUT.exists():
        return
    payload = json.loads(
        wp10c9b.DEFAULT_OUTPUT.read_text(encoding="utf-8")
    )

    assert payload["work_package"] == "WP10c9b"
    assert payload["classification"] == (
        "characteristic_matrix_rejected_bdf_noise_and_"
        "inward_shear_damping_unresolved"
    )
    assert not payload["passed"]
    assert payload["coordinate_principal_basis"]["passed"]
    for row in payload["coordinate_principal_basis"][
        "by_refinement_ratio"
    ].values():
        assert row["passed"]
        assert row["maximum_imaginary_part"] == 0.0
        assert row["maximum_incoming_inner_characteristics"] == 0
        assert (
            row["maximum_eigenpair_defect"]
            <= wp10c9b.MAXIMUM_EIGENPAIR_DEFECT
        )
        assert (
            row["maximum_biorthogonality_defect"]
            <= wp10c9b.MAXIMUM_BIORTHOGONALITY_DEFECT
        )

    method = payload["method_contract"]
    assert not method["passed_before_packets"]
    assert method["constant_state_dissipative_flux_defect"] == 0.0
    assert (
        method["maximum_shared_flux_defect"]
        <= wp10c9b.MAXIMUM_SHARED_FLUX_DEFECT
    )
    assert (
        method["maximum_storage_action_defect"]
        <= wp10c9b.MAXIMUM_STORAGE_ACTION_DEFECT
    )
    assert method["dense_colored_jacobian"]["passed"]
    assert method["no_incoming_excision_characteristic"]

    bdf = method["bdf2_split_replay"]
    assert not bdf["passed"]
    assert bdf["completed_steps"] == 0
    assert (
        bdf["maximum_scaled_residual"]
        > bdf["binding_residual_tolerance"]
    )
    assert (
        bdf["maximum_discrete_ledger_relative_defect"]
        < bdf["maximum_scaled_residual"]
    )

    packets = payload["packet_results"]
    assert set(packets) == set(
        wp10c9b.CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES
    )
    assert not payload["all_packet_gates_passed"]
    assert not packets["inward_shear"]["passed"]
    assert (
        packets["inward_shear"]["observed_orders"]["damping"]
        < wp10c9b.MINIMUM_PACKET_DAMPING_ORDER
    )
    assert (
        packets["inward_shear"]["fine_minimum_signed_cosine"]
        >= wp10c9b.MINIMUM_SIGNED_COSINE
    )
    assert (
        packets["inward_shear"]["observed_orders"]["smooth_dissipation"]
        >= wp10c9b.MINIMUM_SMOOTH_ORDER
    )
    for family, packet in packets.items():
        if family != "inward_shear":
            assert packet["passed"]

    assert payload["conditional_common_mode"] is None
    scope = payload["scope"]
    assert scope["audit_only_characteristic_matrix_built"]
    assert scope["pure_packet_ladder_run"]
    assert not scope["production_operator_changed"]
    assert not scope["common_mode_rerun"]
    assert not scope["bounded_nonlinear_patch_truth_run"]
    assert not scope["fixed_q_averaging_run"]
    assert not scope["reduced_model_run"]

    decision = payload["decision"]
    assert not decision["candidate_promoted_to_production"]
    assert not decision["pure_packets_passed"]
    assert not decision["common_mode_rerun_authorized_and_run"]
    assert not decision["bounded_nonlinear_patch_truth_authorized"]
    assert not decision["one_more_brute_force_refinement_authorized"]
    assert not decision["fixed_q_averaging_authorized"]
    assert not decision["reduced_model_authorized"]
    assert "path-conservative" in decision["next_step"]
