import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "results/canonical"
    / "causal_inner_face36_fixed_q_jacobian_repair_"
    "wp10c9d6c7c3b5c4f24b"
)


def _read(name: str) -> dict:
    return json.loads((ARTIFACT / name).read_text(encoding="utf-8"))


def test_c4f24b_certifies_only_the_repair_preflight():
    summary = _read("summary.json")
    assert summary["passed"]
    assert summary["analysis_only"]
    assert not summary["trajectory_executed"]
    assert not summary["physical_operator_changed"]
    assert summary["classification"] == (
        "fixed_Q_Jacobian_and_exact_BE_limit_repair_passed"
    )
    assert summary["authorized_next"] == (
        "second_state_and_constrained_BDF2_preflight"
    )
    assert not summary["one_Q_execution_manifest_authorized"]
    assert not summary["fixed_Q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_c4f24b_derivative_and_exact_step_gates_pass():
    summary = _read("summary.json")
    derivative = summary["derivative_audit"]
    gates = summary["gates"]
    assert derivative["direct_monolithic_JVP_relative_defect"] <= gates[
        "maximum_direct_monolithic_JVP_relative_defect"
    ]
    assert derivative["direct_augmented_JVP_relative_defect"] <= gates[
        "maximum_direct_augmented_JVP_relative_defect"
    ]
    assert derivative["direct_raw_reaction_JVP_relative_defect"] <= gates[
        "maximum_direct_raw_reaction_JVP_relative_defect"
    ]
    assert all(step["accepted"] for step in summary["exact_steps"])
    assert max(
        step["maximum_scaled_residual"] for step in summary["exact_steps"]
    ) <= gates["maximum_exact_step_scaled_residual"]
    assert max(
        step["maximum_Q3_relative_defect"] for step in summary["exact_steps"]
    ) <= gates["maximum_exact_step_Q3_relative_defect"]
    assert min(summary["rate_convergence_orders"]) >= gates[
        "minimum_rate_convergence_order"
    ]
    assert min(summary["multiplier_convergence_orders"]) >= gates[
        "minimum_multiplier_convergence_order"
    ]


def test_c4f24b_decisive_arrays_and_hashes_are_complete():
    with np.load(ARTIFACT / "decisive_arrays.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(
            arrays["timesteps_seconds"],
            np.asarray((1.0e-7, 5.0e-8, 2.5e-8)),
        )
        assert arrays["derivative_direction"].shape == (563,)
        for index in range(3):
            assert arrays[f"step_{index}_primitive_charts"].shape == (112, 5)
            assert arrays[f"step_{index}_scaled_rate_per_s"].shape == (560,)
            assert arrays[f"step_{index}_multipliers"].shape == (3,)
            assert np.all(np.isfinite(arrays[f"step_{index}_primitive_charts"]))

    entries = {}
    for line in (ARTIFACT / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    assert set(entries) == {
        "config.json",
        "decisive_arrays.npz",
        "provenance.json",
        "summary.json",
    }
    for name, digest in entries.items():
        assert hashlib.sha256((ARTIFACT / name).read_bytes()).hexdigest() == digest


def test_c4f24b_provenance_records_uncommitted_execution_transparently():
    provenance = _read("provenance.json")
    assert not provenance["working_tree_clean"]
    assert provenance["implementation_commit"] is None
    assert provenance["execution_base_commit"]
    assert provenance["implementation_source_bundle_sha256"]
    for relative, digest in provenance["source_hashes"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
