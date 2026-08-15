import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT
    / "results/canonical"
    / "causal_inner_face36_fixed_q_endpoint_linearization_audit_manifest_"
    "wp10c9d6c7c3b5c4f24e3"
)
RESULT = (
    ROOT
    / "results/canonical"
    / "causal_inner_face36_fixed_q_endpoint_linearization_audit_"
    "wp10c9d6c7c3b5c4f24e3"
)


def _read(directory: Path, name: str) -> dict:
    return json.loads((directory / name).read_text(encoding="utf-8"))


def test_c4f24e3_manifest_is_fail_fast_and_analysis_only() -> None:
    summary = _read(MANIFEST, "summary.json")
    contract = _read(MANIFEST, "execution_manifest.json")
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["analysis_execution_authorized"]
    assert not summary["physical_execution_authorized"]
    assert contract["analysis_only"]
    assert contract["binding_directional_step"] == 1.0e-4
    assert contract["maximum_actual_correction_action_error_to_residual"] == 0.10
    assert contract["fail_fast_before_sweep_if_binding_action_fails"]
    assert not contract["may_relax_nonlinear_residual_gate"]


def test_c4f24e3_result_preserves_all_physical_stops() -> None:
    summary = _read(RESULT, "summary.json")
    assert summary["analysis_only"]
    assert not summary["trajectory_executed"]
    assert not summary["physical_failure_detected"]
    assert summary["parent_rejections_preserved"]
    assert not summary["adaptive_refresh_implementation_authorized"]
    assert not summary["physical_history_ladder_authorized"]
    assert not summary["fixed_Q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_c4f24e3_result_arrays_close_to_metrics() -> None:
    metrics = _read(RESULT, "metrics.json")
    with np.load(RESULT / "decisive_arrays.npz", allow_pickle=False) as source:
        residual = np.asarray(source["base_residual"])
        analytic = np.asarray(source["exact_matrix_action"])
        five = np.asarray(source["five_point_JVP"])
        correction = np.asarray(source["correction"])
    assert float(np.max(np.abs(residual))) == metrics["base_maximum_scaled_residual"]
    ratio = np.linalg.norm(np.linalg.norm(correction) * (five - analytic)) / np.linalg.norm(residual)
    assert np.isclose(
        ratio,
        metrics["actual_correction_action_error_to_residual"],
        rtol=0.0,
        atol=2.0e-15,
    )


def test_c4f24e3_checksum_manifests_are_complete() -> None:
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
