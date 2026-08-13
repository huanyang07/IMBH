import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_recovered_coupling_existing_state_ledger_preflight_wp10c9d6c7c3b5c4f9 as c4f9


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_c4f9_preserves_the_guard_buffer_and_runs_no_trajectory():
    contract = _read(c4f9.CONTRACT_PATH)
    summary = _read(c4f9.SUMMARY_PATH)
    assert contract["new_trajectory"] is False
    assert contract["recovery_parent_face"] == 36
    assert contract["coupling_parent_face"] == 48
    assert contract["control_volume_reconstruction_is_algebraically_dependent"] is True
    assert summary["physical_failure_detected"] is False
    assert summary["memory_propagation_authorized"] is False
    assert summary["fixed_Q_micro_solver_authorized"] is False
    assert summary["reduced_slow_evolution_authorized"] is False


def test_c4f9_exact_BDF_rearrangement_closes_but_is_not_independent():
    summary = _read(c4f9.SUMMARY_PATH)
    arrays = np.load(c4f9.DECISIVE_ARRAYS, allow_pickle=False)
    gate = _read(c4f9.CONTRACT_PATH)["gates"]["maximum_exact_BDF_ledger_defect"]
    assert summary["method_gates_passed"] is True
    assert summary["exact_BDF_ledger_defect"] <= gate
    assert summary["exact_BDF_residual_defect"] <= gate
    assert summary["control_volume_reconstruction_is_independent_evidence"] is False
    direct = arrays["exact_BDF_direct_face48_flux"]
    reconstructed = arrays["exact_BDF_reconstructed_face48_flux"]
    block_scale = np.maximum(
        np.maximum(np.abs(direct), np.abs(reconstructed)),
        np.max(np.abs(arrays["exact_BDF_buffer_block_sums"]), axis=1),
    )
    assert np.max(np.abs(direct - reconstructed) / block_scale) <= gate


def test_c4f9_selects_only_the_spatially_convergent_overlap_state():
    summary = _read(c4f9.SUMMARY_PATH)
    assert summary["passed"] is True
    assert summary["classification"] == "control_volume_identity_dependent_overlap_state_converges"
    assert summary["overlap_state_spatially_convergent"] is True
    assert all(item["passed"] for item in summary["overlap_state_metrics"].values())
    assert summary["direct_face48_absolute_export_spatially_convergent"] is False
    assert not all(item["passed"] for item in summary["control_volume_direct_flux_metrics"].values())
    assert summary["authorized_next"] == (
        "WP10c9d6c7c3b5c4f10_definitions_only_retained_guard_buffer_micro_macro_manifest"
    )


def test_c4f9_zero_height_mass_channel_is_not_divided_by_zero():
    arrays = np.load(c4f9.DECISIVE_ARRAYS, allow_pickle=False)
    height = arrays["guard_responsive_height_history_rates"]
    assert np.array_equal(height[..., 0], np.zeros_like(height[..., 0]))
    assert np.all(np.isfinite(height))
    assert _read(c4f9.SUMMARY_PATH)["overlap_state_metrics"][
        "guard_responsive_height_history_rate"
    ]["passed"] is True
