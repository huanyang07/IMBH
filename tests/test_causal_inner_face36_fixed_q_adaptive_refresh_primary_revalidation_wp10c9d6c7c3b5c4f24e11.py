import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_adaptive_refresh_primary_manifest_"
    "wp10c9d6c7c3b5c4f24e11"
)
RESULT = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_adaptive_refresh_primary_"
    "wp10c9d6c7c3b5c4f24e11"
)
BASELINE = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_primary_retry_wp10c9d6c7c3b5c4f24e8"
)


def _read(directory: Path, name: str) -> dict:
    return json.loads((directory / name).read_text(encoding="utf-8"))


def test_f24e11_manifest_freezes_fail_closed_policy() -> None:
    summary = _read(MANIFEST, "summary.json")
    contract = _read(MANIFEST, "execution_manifest.json")
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["primary_revalidation_authorized"]
    assert not summary["heldout_retry_authorized"]
    assert contract["exact_jacobian_refresh_policy"] == "on_line_search_failure"
    assert contract["maximum_exact_jacobian_assemblies"] == 2
    assert contract[
        "additional_exact_jacobian_allowed_only_after_line_search_failure"
    ]
    assert contract["maximum_scaled_residual"] == 1.0e-10
    assert not contract["may_resume_refined_ladder"]


def test_f24e11_result_is_primary_nonregression_only() -> None:
    summary = _read(RESULT, "summary.json")
    metrics = _read(RESULT, "metrics.json")
    assert summary["historical_results_preserved"]
    assert summary["primary_nonregression_passed"] == summary["passed"]
    assert summary["heldout_retry_manifest_authorized"] == summary["passed"]
    assert not summary["heldout_retry_authorized"]
    assert not summary["remaining_ladder_execution_authorized"]
    assert not summary["fixed_Q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    assert metrics["BDF1_used_only_initial_exact_Jacobian"]
    assert metrics["BDF2_used_only_initial_exact_Jacobian"]
    assert metrics["baseline_decisive_arrays_bitwise"]


def test_f24e11_decisive_arrays_are_bitwise_baseline() -> None:
    with np.load(RESULT / "decisive_arrays.npz", allow_pickle=False) as current:
        with np.load(
            BASELINE / "decisive_arrays.npz", allow_pickle=False
        ) as baseline:
            assert set(current.files) == set(baseline.files)
            for name in current.files:
                np.testing.assert_array_equal(current[name], baseline[name])


def test_f24e11_checksum_manifests_are_complete() -> None:
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
