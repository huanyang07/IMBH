from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PHASE_ROOT = ROOT / "results/canonical/phase_dae_entry_N164"
CLASSIFICATION = (
    ROOT
    / "results/canonical/p0_validity_ledger_outer_manifold"
    / "phase_critical_classification_summary.json"
)


PHASE_CASES = (
    (
        "k12",
        PHASE_ROOT / "k12_state.npz",
        PHASE_ROOT / "k12_summary.json",
        12,
        True,
    ),
    (
        "k13",
        PHASE_ROOT / "state.npz",
        PHASE_ROOT / "k13_summary.json",
        13,
        True,
    ),
    (
        "k14",
        PHASE_ROOT / "k14_state.npz",
        PHASE_ROOT / "k14_summary.json",
        14,
        False,
    ),
)


@pytest.mark.parametrize("_label,checkpoint,table,n_intervals,monotone", PHASE_CASES)
def test_phase_checkpoint_is_complete_and_classified(
    _label: str,
    checkpoint: Path,
    table: Path,
    n_intervals: int,
    monotone: bool,
) -> None:
    assert checkpoint.exists()
    assert table.exists()
    with np.load(checkpoint) as data:
        intervals = np.asarray(data["global_flux_phase_dae_segment_aux_interval_indices"], dtype=int)
        nodes = np.asarray(data["global_flux_phase_dae_segment_aux_node_indices"], dtype=int)
        z = np.asarray(data["global_flux_phase_dae_segment_aux_z"], dtype=float)
        p = np.asarray(data["global_flux_phase_dae_segment_aux_p"], dtype=float)
        p_mid = np.asarray(data["global_flux_phase_dae_segment_aux_p_mid"], dtype=float)
        ds = np.asarray(data["global_flux_phase_dae_segment_aux_ds"], dtype=float)

    assert intervals.shape == (n_intervals,)
    assert nodes.shape == (n_intervals + 1,)
    assert z.shape == (n_intervals + 1, 4)
    assert p.shape == (n_intervals + 1, 4)
    assert p_mid.shape == (n_intervals, 4)
    assert ds.shape == (n_intervals,)
    assert np.array_equal(np.diff(intervals), np.ones(n_intervals - 1, dtype=int))
    assert np.array_equal(nodes[:-1], intervals)
    assert nodes[-1] == intervals[-1] + 1
    assert np.all(np.isfinite(z))
    assert np.all(np.isfinite(p))
    assert np.all(np.isfinite(p_mid))
    assert np.all(np.isfinite(ds))
    assert np.all(np.diff(z[:, 3]) > 0.0)
    assert np.all(ds > 0.0)

    p_r = np.concatenate([p[:, 3], p_mid[:, 3]])
    if monotone:
        assert np.min(p_r) > 0.0
    else:
        assert np.min(p_r) < 0.0

    row = json.loads(table.read_text())
    assert int(row["global_flux_phase_dae_segment_n_intervals"]) == n_intervals
    if "global_flux_phase_dae_segment_accepted_exploratory" in row:
        assert bool(row["global_flux_phase_dae_segment_accepted_exploratory"]) is monotone
    assert int(row["global_flux_phase_dae_segment_final_p_R_sign_changes"]) == (0 if monotone else 2)


def test_k13_direct_physical_residual_gate() -> None:
    row = json.loads((PHASE_ROOT / "k13_summary.json").read_text())
    assert row["global_flux_phase_dae_segment_final_direct_radial_max"] <= 1.0e-4
    assert row["global_flux_phase_dae_segment_final_direct_energy_max"] <= 1.0e-4
    assert row["global_flux_phase_dae_segment_final_fprime_max"] <= 1.0e-5
    assert row["global_flux_phase_dae_segment_final_kinematic_max"] <= 1.0e-3
    assert row["global_flux_phase_dae_segment_final_endpoint_state_mismatch_max"] <= 1.0e-3


def test_phase_homogeneous_rows_match_direct_and_remain_finite_at_zero_pr() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "src"))
    import run_mdot5_global_phase_dae_production as global_phase

    _x, params, _context, _aux, phase = global_phase._load_problem()
    checkpoint = PHASE_ROOT / "exit_refinement_endpoint.npz"
    with np.load(checkpoint) as data:
        z = np.asarray(data["z"], dtype=float)
        p = np.asarray(data["p"], dtype=float)

    point = global_phase.model._global_flux_phase_dae_point_data(
        z[-1], p[-1], params, float(phase["lambda0"])
    )
    assert np.max(np.abs(point["equivalence"])) <= 1.0e-8
    assert abs(float(point["fprime_equivalence"])) <= 1.0e-8

    tangent_zero = np.asarray(p[-1], dtype=float).copy()
    tangent_zero[3] = 0.0
    zero_point = global_phase.model._global_flux_phase_dae_point_data(
        z[-1], tangent_zero, params, float(phase["lambda0"])
    )
    assert np.all(np.isfinite(zero_point["homogeneous_rows"]))


def test_phase_critical_classification_is_step_converged_and_low_u_singular() -> None:
    result = json.loads(CLASSIFICATION.read_text())
    decision = result["decision"]
    assert decision["classification"] == "finite-radius low-u stagnation/singular boundary"
    assert decision["low_u_singular"] is True
    assert decision["finite_state_fold"] is False
    assert decision["regular_critical_point"] is False
    assert decision["global_certified"] is False

    baseline = [row["summary"] for row in result["baseline"]]
    assert len(baseline) == 2
    assert all(row["complete"] for row in baseline)
    assert all(float(row["p_R_final"]) > 0.0 for row in baseline)
    limits = np.asarray([row["R_limit_rg"] for row in baseline], dtype=float)
    assert np.ptp(limits) <= 5.0e-5

    scaling = result["scaling"]
    assert scaling["Sigma"]["power_of_u"] < -0.9
    assert scaling["tau"]["power_of_u"] < -0.9
    assert scaling["rho"]["power_of_u"] < -1.4
    assert scaling["H_over_R"]["power_of_u"] > 0.4
    assert scaling["Mach_eff"]["power_of_u"] > 0.4


def test_phase_critical_classification_source_shape_and_bordered_audits() -> None:
    result = json.loads(CLASSIFICATION.read_text())
    decision = result["decision"]
    assert decision["source_shape_resolved"] is True
    assert decision["source_shape_sensitive"] is False

    source = [row["summary"] for row in result["source_branches"]]
    assert {row["label"] for row in source} == {
        "compact_c2",
        "compact_c4",
        "compact_cinf",
        "compact_c2_wide",
    }
    assert all(row["complete"] for row in source)
    assert all(row["phase_homotopy_accepted"] for row in source)
    assert all(float(row["p_R_final"]) > 0.0 for row in source)
    assert np.ptp(np.asarray([row["R_limit_rg"] for row in source], dtype=float)) < 0.05

    bordered = result["bordered"]
    assert len(bordered) == 2
    assert all(int(row["accepted_steps"]) == 30 for row in bordered)
    assert all(row["crossed"] is False for row in bordered)
    assert all(float(row["final_p_R"]) > 0.0 for row in bordered)
    assert max(float(row["max"]) for row in result["angular_closures"]) < 3.0e-5
