import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_retained_guard_buffer_micro_macro_manifest_wp10c9d6c7c3b5c4f10 as c4f10


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_c4f10_is_definitions_only_and_preserves_all_hard_stops():
    manifest = _read(c4f10.MANIFEST_PATH)
    summary = _read(c4f10.SUMMARY_PATH)
    assert manifest["definitions_only"] is True
    assert manifest["new_trajectory"] is False
    assert manifest["production_operator_changed"] is False
    assert summary["new_trajectory_authorized"] is False
    assert summary["memory_propagation_authorized"] is False
    assert summary["fixed_Q_micro_solver_authorized"] is False
    assert summary["reduced_slow_evolution_authorized"] is False
    assert summary["fifty_ms_propagation_authorized"] is False


def test_c4f10_freezes_single_physical_ownership_and_no_double_count():
    manifest = _read(c4f10.MANIFEST_PATH)
    physical = manifest["physical_partition"]
    overlap = manifest["numerical_overlap"]
    assert physical["inner_micro_core_parent_cells"] == [0, 36]
    assert physical["macro_exterior_parent_cells"] == [36, 64]
    assert physical["shared_exchange_parent_face"] == 36
    assert physical["raw_face48_flux_forbidden_as_slow_exchange"] is True
    assert overlap["micro_solver_guard_parent_cells"] == [36, 48]
    assert overlap["micro_guard_is_duplicate_numerical_state"] is True
    assert overlap["micro_guard_storage_and_sources_counted_in_physical_inventory"] is False
    assert overlap["macro_overlap_storage_and_sources_counted_exactly_once"] is True
    assert overlap["primitive_recovery_from_restricted_storage_requires_separate_certification"] is True


def test_c4f10_requires_DAE_complete_conservative_synchronization():
    manifest = _read(c4f10.MANIFEST_PATH)
    contract = manifest["synchronization_contract"]
    assert contract["reaction_M_J_E_must_be_ledgered"] is True
    assert contract["responsive_height_history_must_be_transferred"] is True
    assert contract["primitive_only_overwrite_forbidden"] is True
    assert contract["Euclidean_projection_forbidden"] is True
    assert contract["double_counted_storage_forbidden"] is True
    assert "preserving_zero_mean_fine_complement" in contract["macro_to_micro"]
    assert manifest["authorized_next"] == (
        "WP10c9d6c7c3b5c4f11_analysis_only_existing_state_overlap_consistency_preflight"
    )
