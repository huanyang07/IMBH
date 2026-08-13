import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_existing_state_overlap_consistency_preflight_wp10c9d6c7c3b5c4f11 as c4f11


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_c4f11_certifies_storage_space_overlap_maps_only():
    summary = _read(c4f11.SUMMARY_PATH)
    assert summary["passed"] is True
    assert summary["classification"] == "existing_state_overlap_contract_certified"
    assert summary["method_gates_passed"] is True
    assert summary["spatial_gates_passed"] is True
    assert summary["guard_reaction_observability_gate_passed"] is True
    assert summary["raw_face48_absolute_export_rejection_preserved"] is True


def test_c4f11_closes_restriction_inventory_and_roundtrip():
    summary = _read(c4f11.SUMMARY_PATH)
    manifest = _read(c4f11.c4f10.MANIFEST_PATH)
    gates = manifest["prospective_gates"]
    assert summary["maximum_conservative_restriction_defect"] <= gates["maximum_conservative_restriction_defect"]
    assert summary["maximum_physical_inventory_partition_defect"] <= gates["maximum_physical_inventory_partition_defect"]
    assert summary["maximum_fine_complement_zero_mean_closure"] <= gates["maximum_overlap_sync_roundtrip_defect"]
    assert summary["maximum_storage_increment_roundtrip_defect"] <= gates["maximum_overlap_sync_roundtrip_defect"]
    assert summary["maximum_baseline_plus_response_storage_defect"] <= gates["maximum_baseline_plus_response_scaled_defect"]


def test_c4f11_preserves_fine_complement_and_authorizes_only_manifest():
    summary = _read(c4f11.SUMMARY_PATH)
    assert summary["fine_complement_face36_observability_fraction"] <= 0.10
    assert summary["memory_propagation_authorized"] is False
    assert summary["fixed_Q_micro_solver_authorized"] is False
    assert summary["reduced_slow_evolution_authorized"] is False
    assert summary["fifty_ms_propagation_authorized"] is False
    assert summary["authorized_next"] == (
        "WP10c9d6c7c3b5c4f12_definitions_only_face36_augmented_projected_memory_screen_manifest"
    )
