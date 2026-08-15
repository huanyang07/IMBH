import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "results/canonical"
    / "causal_inner_face36_fixed_q_second_state_bdf2_preflight_"
    "wp10c9d6c7c3b5c4f24c"
)


def _read(name: str) -> dict:
    return json.loads((ARTIFACT / name).read_text(encoding="utf-8"))


def test_c4f24c_preserves_the_narrow_negative_classification() -> None:
    summary = _read("summary.json")
    assert not summary["passed"]
    assert summary["analysis_only"]
    assert not summary["trajectory_executed"]
    assert not summary["physical_operator_changed"]
    assert summary["classification"] == (
        "fixed_Q_second_state_Jacobian_and_exact_BDF2_roots_passed_"
        "but_synthetic_history_limit_orders_failed"
    )
    assert summary["authorized_next"] == (
        "definitions_only_constrained_BDF_startup_history_preflight"
    )
    assert not summary["one_Q_execution_manifest_authorized"]
    assert not summary["fixed_Q_micro_solver_authorized"]
    assert not summary["one_Q_propagation_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_c4f24c_derivatives_and_roots_pass_but_orders_fail() -> None:
    summary = _read("summary.json")
    gates = summary["gates"]
    assert summary["second_state_derivative_certified"]
    assert summary["exact_constrained_BDF2_roots_certified"]
    assert not summary["synthetic_history_limit_orders_certified"]
    assert len(summary["exact_bdf2_steps"]) == 5
    assert all(step["accepted"] for step in summary["exact_bdf2_steps"])
    assert max(
        step["maximum_scaled_residual"]
        for step in summary["exact_bdf2_steps"]
    ) <= gates["maximum_exact_step_scaled_residual"]
    assert max(
        step["maximum_Q3_relative_defect"]
        for step in summary["exact_bdf2_steps"]
    ) <= gates["maximum_exact_step_Q3_relative_defect"]
    assert max(
        step["history_Q3_relative_defect"]
        for step in summary["exact_bdf2_steps"]
    ) <= gates["maximum_history_Q3_relative_defect"]
    assert (
        min(summary["rate_convergence_orders"])
        < gates["minimum_rate_convergence_order"]
    )
    assert (
        min(summary["multiplier_convergence_orders"])
        < gates["minimum_multiplier_convergence_order"]
    )


def test_c4f24c_decisive_arrays_and_hashes_are_complete() -> None:
    with np.load(ARTIFACT / "decisive_arrays.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(
            arrays["timesteps_seconds"],
            np.asarray((2.0e-9, 1.0e-9, 5.0e-10, 8.0e-9, 4.0e-9)),
        )
        assert arrays["derivative_direction"].shape == (563,)
        for index in range(5):
            assert arrays[f"step_{index}_primitive_charts"].shape == (112, 5)
            assert arrays[f"step_{index}_scaled_bdf_rate_per_s"].shape == (560,)
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


def test_c4f24c_provenance_hashes_the_committed_source_bundle() -> None:
    provenance = _read("provenance.json")
    assert not provenance["working_tree_clean"]
    assert provenance["implementation_commit"] is None
    artifact_commit = subprocess.run(
        (
            "git",
            "log",
            "-1",
            "--format=%H",
            "--",
            str((ARTIFACT / "provenance.json").relative_to(ROOT)),
        ),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    for relative, digest in provenance["source_hashes"].items():
        if artifact_commit:
            contents = subprocess.run(
                ("git", "show", f"{artifact_commit}:{relative}"),
                cwd=ROOT,
                check=True,
                capture_output=True,
            ).stdout
        else:
            contents = (ROOT / relative).read_bytes()
        assert hashlib.sha256(contents).hexdigest() == digest
