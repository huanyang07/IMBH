import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT
    / "results/canonical"
    / "causal_inner_face36_fixed_q_exact_refresh_diagnostic_manifest_"
    "wp10c9d6c7c3b5c4f24e2"
)
RESULT = (
    ROOT
    / "results/canonical"
    / "causal_inner_face36_fixed_q_exact_refresh_diagnostic_"
    "wp10c9d6c7c3b5c4f24e2"
)


def _read(directory: Path, name: str) -> dict:
    return json.loads((directory / name).read_text(encoding="utf-8"))


def test_c4f24e2_manifest_is_diagnostic_only() -> None:
    summary = _read(MANIFEST, "summary.json")
    contract = _read(MANIFEST, "execution_manifest.json")
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["diagnostic_execution_authorized"]
    assert not summary["adaptive_refresh_implementation_authorized"]
    assert contract["diagnostic_only"]
    assert not contract["may_amend_parent_rejection"]
    assert contract["maximum_scaled_residual"] == 1.0e-10
    assert contract["exact_Jacobian_assemblies_in_diagnostic"] == 1


def test_c4f24e2_result_preserves_parent_rejection() -> None:
    summary = _read(RESULT, "summary.json")
    assert summary["diagnostic_only"]
    assert summary["parent_rejection_preserved"]
    assert not summary["one_Q_execution_manifest_authorized"]
    assert not summary["fixed_Q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    assert summary["adaptive_refresh_implementation_authorized"] == summary["passed"]


def test_c4f24e2_result_arrays_close_to_metrics() -> None:
    metrics = _read(RESULT, "metrics.json")
    with np.load(RESULT / "decisive_arrays.npz", allow_pickle=False) as source:
        maximum = float(np.max(np.abs(source["augmented_scaled_residual"])))
    assert maximum == metrics["maximum_scaled_residual"]
    assert metrics["exact_Jacobian_assemblies"] == 1


def test_c4f24e2_checksum_manifests_are_complete() -> None:
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
