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

import run_causal_inner_family_transfer_audit_wp10c9c0c as wp10c9c0c


def test_wp10c9c0c_machine_evidence_closes_family_ledgers() -> None:
    if not wp10c9c0c.DEFAULT_OUTPUT.exists():
        return
    payload = json.loads(
        wp10c9c0c.DEFAULT_OUTPUT.read_text(encoding="utf-8")
    )
    assert payload["work_package"] == "WP10c9c0c"
    assert payload["base_commit"] == wp10c9c0c.BASE_COMMIT
    assert payload["method_contract_passed"]

    method = payload["common_mode_family_decomposition"][
        "method_contract"
    ]
    assert method["passed"]
    assert (
        method["maximum_projector_identity_defect"]
        <= wp10c9c0c.MAXIMUM_PROJECTOR_DEFECT
    )
    assert (
        method["maximum_history_state_sum_defect"]
        <= wp10c9c0c.MAXIMUM_DECOMPOSITION_DEFECT
    )
    assert (
        method["maximum_history_rate_sum_defect"]
        <= wp10c9c0c.MAXIMUM_DECOMPOSITION_DEFECT
    )
    assert (
        method["maximum_cross_work_defect"]
        <= wp10c9c0c.MAXIMUM_PAIRWISE_LEDGER_DEFECT
    )

    local = payload["pure_inward_shear_local_work"]
    assert (
        local["maximum_instantaneous_block_closure_defect"]
        <= wp10c9c0c.MAXIMUM_LOCAL_BLOCK_CLOSURE_DEFECT
    )
    assert not payload["decision"]["wp10c9c1_path_candidate_authorized"]
    assert not payload["decision"]["production_operator_change_authorized"]
    assert not payload["decision"]["new_truth_trajectory_authorized"]
    assert not payload["decision"]["fixed_q_or_reduction_authorized"]
    assert not payload["localized_mechanism_gate_passed"]
    assert (
        local["controlling_block_absolute_fraction"]
        < wp10c9c0c.MINIMUM_CONTROLLING_BLOCK_FRACTION
    )
    assert (
        payload["cross_audit_correlation"]["radial_profile_cosine"]
        < wp10c9c0c.MINIMUM_RADIAL_PROFILE_COSINE
    )
    assert payload["classification"] == (
        "common_mode_failure_remains_multifamily_or_nonlocal"
    )
