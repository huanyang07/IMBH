import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_face36_augmented_memory_screen_manifest_wp10c9d6c7c3b5c4f12 as c4f12


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_c4f12_authorizes_only_analysis_block_tangents():
    manifest = _read(c4f12.MANIFEST_PATH)
    summary = _read(c4f12.SUMMARY_PATH)
    assert manifest["definitions_only"] is True
    assert manifest["propagation_executed"] is False
    assert manifest["memory_propagation_authorized"] is True
    assert manifest["new_nonlinear_trajectory_authorized"] is False
    assert manifest["fixed_Q_authorized"] is False
    assert manifest["reduced_slow_evolution_authorized"] is False
    assert summary["memory_propagation_authorized"] is True
    assert summary["new_nonlinear_trajectory_authorized"] is False


def test_c4f12_uses_face36_and_retains_guard_history_and_complement():
    manifest = _read(c4f12.MANIFEST_PATH)
    assert manifest["binding_output"]["name"] == "shared_face36_M_J_E_flux"
    assert manifest["slow_coordinate"]["raw_face48_flux_forbidden"] is True
    assert manifest["augmented_diagnostic_output"]["fine_complement_retained"] is True
    assert manifest["augmented_diagnostic_output"]["guard_parent_responsive_height_history"]
    assert manifest["initial_lifts"]["per_step_reprojection"] is False
    assert manifest["initial_lifts"]["physical_fixed_Q_constraint_imposed"] is False


def test_c4f12_freezes_cost_and_architecture_decisions_before_results():
    manifest = _read(c4f12.MANIFEST_PATH)
    assert manifest["algorithm"]["all_29_directions_one_block_solve"] is True
    assert manifest["staging"]["run_middle_first"] is True
    assert manifest["staging"]["stop_before_fine_on_any_method_gate_failure"] is True
    assert manifest["staging"]["no_nonlinear_anchor_in_this_package"] is True
    assert manifest["decision"]["large_Q3_leakage"] == (
        "derive_physical_constraint_reaction_map_before_conditional_memory_claim"
    )
    assert manifest["authorized_next"] == (
        "WP10c9d6c7c3b5c4f13_face36_augmented_analysis_only_memory_screen"
    )
