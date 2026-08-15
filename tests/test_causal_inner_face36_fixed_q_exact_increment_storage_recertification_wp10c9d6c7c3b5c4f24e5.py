import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "results/canonical"
    / "causal_inner_face36_fixed_q_exact_increment_storage_recertification_"
    "wp10c9d6c7c3b5c4f24e5"
)


def _read(name: str) -> dict:
    return json.loads((RESULT / name).read_text(encoding="utf-8"))


def test_c4f24e5_recertification_preserves_stops() -> None:
    summary = _read("summary.json")
    assert summary["analysis_only"]
    assert not summary["trajectory_executed"]
    assert not summary["physical_failure_detected"]
    assert summary["parent_rejections_preserved"]
    assert not summary["fixed_Q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_c4f24e5_metrics_and_arrays_close() -> None:
    metrics = _read("metrics.json")
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


def test_c4f24e5_result_checksums_close() -> None:
    entries = {}
    for line in (RESULT / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    assert set(entries) == {
        "contract.json",
        "decisive_arrays.npz",
        "metrics.json",
        "provenance.json",
        "summary.json",
    }
    for name, digest in entries.items():
        assert hashlib.sha256((RESULT / name).read_bytes()).hexdigest() == digest
