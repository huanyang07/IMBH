import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT
    / "results/canonical"
    / "causal_inner_face36_fixed_q_residual_resolution_audit_manifest_"
    "wp10c9d6c7c3b5c4f24e4"
)
RESULT = (
    ROOT
    / "results/canonical"
    / "causal_inner_face36_fixed_q_residual_resolution_audit_"
    "wp10c9d6c7c3b5c4f24e4"
)


def _read(directory: Path, name: str) -> dict:
    return json.loads((directory / name).read_text(encoding="utf-8"))


def test_c4f24e4_manifest_preserves_all_stops() -> None:
    summary = _read(MANIFEST, "summary.json")
    contract = _read(MANIFEST, "execution_manifest.json")
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["analysis_execution_authorized"]
    assert not summary["physical_execution_authorized"]
    assert contract["analysis_only"]
    assert contract["line_search_alphas"] == [2.0 ** (-i) for i in range(8)]
    assert contract["minimum_resolvable_model_error_order"] == 1.5
    assert not contract["may_change_row_scales_or_merit_norm"]
    assert not contract["may_relax_nonlinear_residual_gate"]


def test_c4f24e4_result_is_analysis_only() -> None:
    summary = _read(RESULT, "summary.json")
    assert summary["analysis_only"]
    assert not summary["trajectory_executed"]
    assert not summary["physical_failure_detected"]
    assert summary["parent_rejections_preserved"]
    assert not summary["physical_history_ladder_authorized"]
    assert not summary["adaptive_refresh_implementation_authorized"]
    assert not summary["fixed_Q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_c4f24e4_arrays_replay_model_errors() -> None:
    metrics = _read(RESULT, "metrics.json")
    with np.load(RESULT / "decisive_arrays.npz", allow_pickle=False) as source:
        errors = np.asarray(source["model_errors_to_base_residual"])
        orders = np.asarray(source["model_error_orders"])
    assert np.array_equal(
        errors,
        np.asarray(
            [item["model_error_to_base_residual"] for item in metrics["alpha_metrics"]]
        ),
    )
    assert np.array_equal(orders, np.asarray(metrics["model_error_orders"]))


def test_c4f24e4_checksum_manifests_are_complete() -> None:
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
