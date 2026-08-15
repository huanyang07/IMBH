import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_heldout_bdf2_exact_refresh_manifest_"
    "wp10c9d6c7c3b5c4f24e10"
)
RESULT = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_heldout_bdf2_exact_refresh_"
    "wp10c9d6c7c3b5c4f24e10"
)


def _read(directory: Path, name: str) -> dict:
    return json.loads((directory / name).read_text(encoding="utf-8"))


def test_f24e10_manifest_is_one_correction_diagnostic() -> None:
    summary = _read(MANIFEST, "summary.json")
    contract = _read(MANIFEST, "execution_manifest.json")
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["diagnostic_execution_authorized"]
    assert not summary["remaining_ladder_execution_authorized"]
    assert contract["diagnostic_only"]
    assert not contract["may_amend_parent_rejection"]
    assert not contract["may_resume_remaining_ladder"]
    assert contract["maximum_newton_iterations"] == 1
    assert contract["exact_Jacobian_corrections"] == 1
    assert contract["maximum_scaled_residual"] == 1.0e-10


def test_f24e10_result_preserves_parent_and_authorization_boundaries() -> None:
    summary = _read(RESULT, "summary.json")
    assert summary["diagnostic_only"]
    assert summary["parent_rejection_preserved"]
    assert summary["source_endpoint_reproduced_bitwise"]
    assert summary["one_exact_Jacobian_correction_used"]
    assert not summary["remaining_ladder_execution_authorized"]
    assert not summary["one_Q_execution_manifest_authorized"]
    assert not summary["fixed_Q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    assert (
        summary["adaptive_refresh_policy_manifest_authorized"]
        == summary["passed"]
    )


def test_f24e10_decisive_arrays_close_to_metrics() -> None:
    metrics = _read(RESULT, "metrics.json")
    with np.load(RESULT / "decisive_arrays.npz", allow_pickle=False) as source:
        source_maximum = float(
            np.max(np.abs(source["source_augmented_scaled_residual"]))
        )
        corrected_maximum = float(
            np.max(np.abs(source["corrected_augmented_scaled_residual"]))
        )
    assert source_maximum == metrics["source_maximum_scaled_residual"]
    assert corrected_maximum == metrics["maximum_scaled_residual"]
    assert metrics["fresh_initial_maximum_scaled_residual"] == source_maximum
    assert metrics["source_endpoint_reproduced_bitwise"]
    assert metrics["exact_Jacobian_assemblies"] == 1


def test_f24e10_checksum_manifests_are_complete() -> None:
    expected = {
        MANIFEST: {
            "execution_manifest.json",
            "provenance.json",
            "source_restart.npz",
            "summary.json",
        },
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
