import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "results/canonical"
    / "causal_inner_face36_state_dependent_fixed_q_step_preflight_"
    "wp10c9d6c7c3b5c4f24"
)


def _read(name: str) -> dict:
    return json.loads((ARTIFACT / name).read_text(encoding="utf-8"))


def test_c4f24_is_analysis_only_and_blocks_execution_manifest():
    summary = _read("summary.json")
    assert not summary["passed"]
    assert summary["classification"] == (
        "state_dependent_fixed_Q_step_and_JVP_preflight_failed"
    )
    assert summary["analysis_only"]
    assert not summary["trajectory_executed"]
    assert not summary["physical_operator_changed"]
    assert not summary["state_dependent_constrained_step_certified"]
    assert not summary["state_dependent_constrained_JVP_certified"]
    assert summary["finite_equal_Q_lifts_preflight_certified"]
    assert not summary["one_Q_execution_manifest_authorized"]
    assert not summary["one_Q_nonlinear_pilot_propagation_authorized"]
    assert not summary["fixed_Q_micro_solver_authorized"]
    assert not summary["fifty_ms_propagation_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    assert summary["raw_face48_export_rejection_preserved"]
    assert summary["authorized_next"] is None


def test_c4f24_localizes_passed_and_failed_frozen_gates():
    summary = _read("summary.json")
    gates = summary["gates"]
    metrics = summary["middle_endpoint"]
    assert not metrics["passed"]
    assert metrics["maximum_Q3_endpoint_relative_defect"] <= gates[
        "maximum_Q3_endpoint_relative_defect"
    ]
    assert metrics["maximum_augmented_endpoint_scaled_residual"] <= gates[
        "maximum_augmented_step_scaled_residual"
    ]
    assert metrics["maximum_constraint_work_ledger_relative_defect"] <= gates[
        "maximum_constraint_work_ledger_relative_defect"
    ]
    assert metrics["continuous_KKT_relative_defect"] <= gates[
        "maximum_continuous_KKT_relative_defect"
    ]
    assert metrics["maximum_dense_colored_Jacobian_relative_defect"] > gates[
        "maximum_dense_colored_Jacobian_relative_defect"
    ]
    assert metrics["maximum_directional_five_point_JVP_relative_defect"] > gates[
        "maximum_directional_JVP_relative_defect"
    ]
    assert metrics[
        "maximum_nonzero_multiplier_state_dependent_central_five_point_defect"
    ] > gates["maximum_directional_JVP_relative_defect"]
    assert metrics["maximum_face36_directional_JVP_relative_defect"] <= gates[
        "maximum_face36_directional_JVP_relative_defect"
    ]
    assert metrics["maximum_small_timestep_KKT_closure_defect"] > gates[
        "maximum_small_timestep_KKT_closure_defect"
    ]
    assert metrics["maximum_zero_multiplier_reduction_defect"] <= gates[
        "maximum_zero_multiplier_reduction_defect"
    ]
    assert metrics["incoming_excision_characteristics"] == 0


def test_c4f24_supplementary_derivative_plateau_does_not_rescue_KKT_limit():
    summary = _read("summary.json")
    diagnostics = summary["supplementary_diagnostics"]
    sweep = diagnostics["reaction_derivative_step_sweep"]
    selected = next(item for item in sweep if item["relative_step"] == 5.0e-6)
    assert selected["central_five_relative_defect"] <= summary["gates"][
        "maximum_directional_JVP_relative_defect"
    ]
    assert not diagnostics["small_timestep_limit_gate_passed"]
    assert diagnostics["small_timestep_cubic_extrapolated_residual"] > summary[
        "gates"
    ]["maximum_small_timestep_KKT_closure_defect"]


def test_c4f24_stores_complete_augmented_operator_and_lifts():
    with np.load(ARTIFACT / "decisive_arrays.npz", allow_pickle=False) as arrays:
        q3 = arrays["q3_scaled_derivative"]
        reaction = arrays["reaction_scaled_rows"]
        lift = arrays["reaction_lift"]
        augmented = arrays["augmented_analytic_matrix"]
        assert q3.shape == (3, 560)
        assert reaction.shape == (560, 3)
        assert lift.shape == (560, 3)
        assert augmented.shape == (563, 563)
        assert arrays["colored_augmented_matrix"].shape == (563, 563)
        assert arrays["screened_augmented_directions"].shape == (563, 27)
        assert arrays["finite_equal_Q_scaled_lifts"].shape == (48, 560)
        assert np.allclose(q3 @ lift, np.eye(3), rtol=0.0, atol=2.0e-12)
        assert np.array_equal(augmented[:560, 560:], -reaction)
        assert np.array_equal(augmented[560:, :560], q3)
        assert np.all(np.isfinite(augmented))


def test_c4f24_finite_lifts_preserve_exact_Q3_and_readiness():
    summary = _read("summary.json")
    gates = summary["gates"]
    metrics = summary["middle_endpoint"]
    assert metrics["sign_symmetric_lift_count"] == 48
    assert metrics["maximum_Q3_endpoint_relative_defect"] <= gates[
        "maximum_Q3_endpoint_relative_defect"
    ]
    assert metrics["maximum_H_over_R"] <= gates["maximum_H_over_R"]
    assert metrics["minimum_scattering_optical_depth"] >= gates[
        "minimum_scattering_optical_depth"
    ]
    assert metrics["maximum_reconstruction_factor"] <= gates[
        "maximum_reconstruction_factor"
    ]


def test_c4f24_hashes_and_source_provenance_are_self_consistent():
    entries = {}
    for line in (ARTIFACT / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    for name, digest in entries.items():
        assert hashlib.sha256((ARTIFACT / name).read_bytes()).hexdigest() == digest

    provenance = _read("provenance.json")
    execution_commit = provenance["execution_commit"]
    for relative, digest in provenance["source_hashes"].items():
        committed = subprocess.run(
            ("git", "show", f"{execution_commit}:{relative}"),
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert hashlib.sha256(committed).hexdigest() == digest
