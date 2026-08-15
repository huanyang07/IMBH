import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_adaptive_refresh_heldout_manifest_"
    "wp10c9d6c7c3b5c4f24e12"
)
RESULT = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_adaptive_refresh_heldout_"
    "wp10c9d6c7c3b5c4f24e12"
)
HISTORICAL = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_remaining_history_ladder_stage_"
    "heldout_coarse_wp10c9d6c7c3b5c4f24e9"
)


def _read(directory: Path, name: str) -> dict:
    return json.loads((directory / name).read_text(encoding="utf-8"))


def test_f24e12_manifest_is_heldout_only() -> None:
    summary = _read(MANIFEST, "summary.json")
    contract = _read(MANIFEST, "execution_manifest.json")
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["heldout_retry_authorized"]
    assert not summary["remaining_ladder_execution_authorized"]
    assert contract["case"] == "heldout_coarse"
    assert contract["exact_jacobian_refresh_policy"] == "on_line_search_failure"
    assert contract["required_BDF2_exact_jacobian_assemblies"] == 2
    assert contract["required_optional_refresh_reason"] == "line_search_failure"
    assert contract["maximum_scaled_residual"] == 1.0e-10
    assert not contract["may_resume_refined_ladder"]


def test_f24e12_result_preserves_historical_rejection() -> None:
    summary = _read(RESULT, "summary.json")
    metrics = _read(RESULT, "metrics.json")
    assert summary["historical_rejection_preserved"]
    assert summary["historical_BDF1_decisive_arrays_bitwise"]
    assert summary["adaptive_refresh_used"]
    assert summary["refined_ladder_manifest_authorized"] == summary["passed"]
    assert not summary["remaining_ladder_execution_authorized"]
    assert not summary["fixed_Q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    assert metrics["BDF1"]["exact_Jacobian_assemblies"] == 1
    assert metrics["BDF2"]["exact_Jacobian_assemblies"] == 2
    assert metrics["BDF2"]["function_evaluations"] == 18


def test_f24e12_bdf1_arrays_are_historical_bitwise() -> None:
    with np.load(RESULT / "decisive_arrays.npz", allow_pickle=False) as current:
        with np.load(
            HISTORICAL / "decisive_arrays.npz", allow_pickle=False
        ) as historical:
            names = [name for name in current.files if name.startswith("bdf1_")]
            assert names
            for name in names:
                np.testing.assert_array_equal(current[name], historical[name])


def test_f24e12_checksum_manifests_are_complete() -> None:
    expected = {
        MANIFEST: {"execution_manifest.json", "provenance.json", "summary.json"},
        RESULT: {
            "contract.json",
            "decisive_arrays.npz",
            "metrics.json",
            "provenance.json",
            "summary.json",
        },
    }
    for directory, names in expected.items():
        entries = {}
        for line in (directory / "SHA256SUMS.txt").read_text().splitlines():
            digest, name = line.split("  ", maxsplit=1)
            entries[name] = digest
        assert set(entries) == names
        for name, digest in entries.items():
            assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
