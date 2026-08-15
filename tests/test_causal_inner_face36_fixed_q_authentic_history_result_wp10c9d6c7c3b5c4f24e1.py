import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "results/canonical"
    / "causal_inner_face36_fixed_q_authentic_history_ladder_"
    "wp10c9d6c7c3b5c4f24e1"
)


def _read(name: str) -> dict:
    return json.loads((ARTIFACT / name).read_text(encoding="utf-8"))


def test_c4f24e1_rejects_only_the_one_refresh_solver_policy() -> None:
    summary = _read("summary.json")
    assert not summary["passed"]
    assert summary["first_failed_case"] == "primary_coarse"
    assert summary["failure_reasons"] == [
        "nonlinear_root",
        "complete_residual",
    ]
    assert not summary["physical_failure_detected"]
    assert not summary["continuous_KKT_or_reaction_architecture_rejected"]
    assert summary["one_exact_Jacobian_plus_Broyden_policy_rejected"]
    assert summary["diagnostic_exact_refresh_authorized"]
    assert not summary["fixed_Q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_c4f24e1_fail_fast_and_nonroot_gates_close() -> None:
    decisive = _read("decisive_case.json")
    case = decisive["case"]
    assert case["failed_stage"] == "BDF1"
    assert decisive["all_nonroot_acceptance_gates_passed"]
    assert decisive["residual_gate_ratio"] > 1.0
    assert case["BDF1"]["exact_Jacobian_assemblies"] == 1
    assert case["BDF1"]["maximum_Q3_relative_defect"] <= 1.0e-12
    assert case["BDF1"]["maximum_storage_parity_relative_defect"] <= 1.0e-9
    assert case["BDF1"]["minimum_path_reconstruction_factor"] == 1.0
    assert case["BDF1"]["incoming_excision_characteristics"] == 0
    assert _read("config.json")[
        "diagnostic_refresh_may_not_convert_this_result_to_a_pass"
    ]


def test_c4f24e1_decisive_arrays_replay_the_residual_maximum() -> None:
    decisive = _read("decisive_case.json")
    with np.load(ARTIFACT / "decisive_arrays.npz", allow_pickle=False) as source:
        maximum = float(np.max(np.abs(source["augmented_scaled_residual"])))
    assert maximum == decisive["checkpoint_residual_maximum"]
    assert maximum == decisive["case"]["BDF1"]["maximum_scaled_residual"]


def test_c4f24e1_execution_source_hashes_close() -> None:
    provenance = _read("provenance.json")
    execution = provenance["physical_execution"]
    sources = {
        "scripts/run_causal_inner_face36_fixed_q_authentic_history_ladder_"
        "wp10c9d6c7c3b5c4f24e1.py": execution["runner_sha256"],
        "src/imri_qpe/layer3_minidisk_1d/causal_inner_fixed_q.py": (
            execution["fixed_q_source_sha256"]
        ),
    }
    for relative, digest in sources.items():
        contents = subprocess.run(
            ("git", "show", f"{execution['execution_commit']}:{relative}"),
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert hashlib.sha256(contents).hexdigest() == digest


def test_c4f24e1_checksum_manifest_is_complete() -> None:
    entries = {}
    for line in (ARTIFACT / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    assert set(entries) == {
        "config.json",
        "decisive_arrays.npz",
        "decisive_case.json",
        "provenance.json",
        "summary.json",
    }
    for name, digest in entries.items():
        assert hashlib.sha256((ARTIFACT / name).read_bytes()).hexdigest() == digest
