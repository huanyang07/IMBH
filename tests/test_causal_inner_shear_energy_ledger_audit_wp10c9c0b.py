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

import run_causal_inner_shear_energy_ledger_wp10c9c0b as wp10c9c0b


def test_wp10c9c0b_machine_evidence_closes_energy_ledger() -> None:
    if not wp10c9c0b.DEFAULT_OUTPUT.exists():
        return
    payload = json.loads(
        wp10c9c0b.DEFAULT_OUTPUT.read_text(encoding="utf-8")
    )

    assert payload["work_package"] == "WP10c9c0b"
    assert payload["base_commit"] == wp10c9c0b.BASE_COMMIT
    assert payload["audit_completed"]
    assert not payload["production_changed"]
    assert not payload["wp10c9c1_path_candidate_authorized"]

    method = payload["method_contract"]
    assert method["passed"]
    assert all(method["checks"].values())
    assert (
        method["measurements"][
            "maximum_instantaneous_block_ledger_defect"
        ]
        <= wp10c9c0b.MAXIMUM_INSTANTANEOUS_LEDGER_DEFECT
    )
    assert (
        method["measurements"][
            "maximum_unattributed_generator_fraction"
        ]
        <= wp10c9c0b.MAXIMUM_UNATTRIBUTED_GENERATOR_FRACTION
    )

    refined = payload["refined_integrated_ledger_contract"]
    assert refined["passed"]
    assert (
        refined["maximum_801_sample_defect"]
        <= wp10c9c0b.MAXIMUM_FINE_INTEGRATED_LEDGER_DEFECT
    )
    assert (
        refined["minimum_observed_order_over_factor_four"]
        >= wp10c9c0b.MINIMUM_INTEGRATED_LEDGER_ORDER
    )

    decision = payload["scientific_decision"]
    assert not decision["legacy_basis_normalization_is_sufficient_explanation"]
    assert not decision["path_inconsistency_proved"]
    assert not decision["wp10c9c1_path_candidate_authorized"]
    assert not decision["fixed_q_reduction_authorized"]
    assert not decision["unique_operator_block_identified"]
    assert decision["multiple_interacting_blocks_implicated"]
    assert (
        decision["full_total_energy_order"]
        >= wp10c9c0b.MINIMUM_SPATIAL_ENERGY_ORDER
    )
    assert (
        decision["full_selected_energy_order"]
        < wp10c9c0b.MINIMUM_SPATIAL_ENERGY_ORDER
    )
    assert (
        decision["fixed_window_selected_energy_order"]
        >= wp10c9c0b.MINIMUM_SPATIAL_ENERGY_ORDER
    )
    assert payload["classification"] == (
        "selected_shear_energy_defect_is_transport_window_"
        "or_family_transfer_sensitive"
    )
