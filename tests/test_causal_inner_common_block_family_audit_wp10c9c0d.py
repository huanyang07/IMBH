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

import run_causal_inner_common_block_family_audit_wp10c9c0d as wp10c9c0d


def test_wp10c9c0d_machine_evidence_closes_direct_ledger() -> None:
    if not wp10c9c0d.DEFAULT_OUTPUT.exists():
        return
    payload = json.loads(
        wp10c9c0d.DEFAULT_OUTPUT.read_text(encoding="utf-8")
    )

    assert payload["work_package"] == "WP10c9c0d"
    assert payload["base_commit"] == wp10c9c0d.BASE_COMMIT
    assert payload["method_contract_passed"]
    assert payload["decomposition_method_contract"]["passed"]
    assert payload["transfer_method_contract"]["passed"]
    assert all(
        row["coordinate_construction"]
        == "native_wp10c8x_decomposition_then_exact_similarity_rescale"
        for row in payload["decomposition_by_mesh"].values()
    )

    decomposition = payload["decomposition_method_contract"]
    assert (
        decomposition["measurements"][
            "maximum_unattributed_generator_fraction"
        ]
        <= wp10c9c0d.MAXIMUM_UNATTRIBUTED_GENERATOR_FRACTION
    )
    transfer = payload["transfer_method_contract"]
    assert (
        transfer["measurements"]["maximum_rate_action_closure_defect"]
        <= wp10c9c0d.MAXIMUM_RATE_ACTION_CLOSURE_DEFECT
    )
    assert (
        transfer["measurements"]["maximum_cross_work_closure_defect"]
        <= wp10c9c0d.MAXIMUM_CROSS_WORK_CLOSURE_DEFECT
    )

    targeted = payload["targeted_common_shear_interaction"]
    assert not targeted["passed"]
    assert targeted["fine_mesh_mediating_block"] == (
        "transport_inner_boundary"
    )
    assert targeted[
        "fine_mesh_mediating_block_activity_fraction"
    ] >= wp10c9c0d.MINIMUM_COMPONENT_ACTIVITY_FRACTION
    assert targeted["fine_pair_interaction_defect_block"] == (
        "transport_central_perfect"
    )
    assert targeted[
        "fine_pair_interaction_defect_block_activity_fraction"
    ] < wp10c9c0d.MINIMUM_COMPONENT_ACTIVITY_FRACTION
    assert targeted["outward_shear_rate_error_block"] == (
        "transport_central_perfect"
    )
    assert targeted[
        "outward_shear_rate_error_block_activity_fraction"
    ] < wp10c9c0d.MINIMUM_COMPONENT_ACTIVITY_FRACTION
    assert not targeted["checks"][
        "same_block_controls_outward_shear_rate_error"
    ]

    assert not payload["localized_mechanism_gate_passed"]
    assert payload["classification"] == (
        "common_mode_defect_remains_multiblock_after_direct_ledger"
    )
    assert not payload["decision"]["wp10c9c1_path_candidate_authorized"]
    assert not payload["decision"]["production_operator_change_authorized"]
    assert not payload["decision"]["new_truth_trajectory_authorized"]
    assert not payload["decision"]["fixed_q_or_reduction_authorized"]
