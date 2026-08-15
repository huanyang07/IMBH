import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT
    / "results/canonical"
    / "causal_inner_face36_fixed_q_schur_solve_audit_manifest_"
    "wp10c9d6c7c3b5c4f24e7"
)
RESULT = (
    ROOT
    / "results/canonical"
    / "causal_inner_face36_fixed_q_schur_solve_audit_"
    "wp10c9d6c7c3b5c4f24e7"
)


def _read(directory: Path, name: str) -> dict:
    return json.loads((directory / name).read_text(encoding="utf-8"))


def test_c4f24e7_manifest_is_analysis_only() -> None:
    summary = _read(MANIFEST, "summary.json")
    contract = _read(MANIFEST, "execution_manifest.json")
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["analysis_execution_authorized"]
    assert not summary["physical_execution_authorized"]
    assert contract["maximum_selected_solve_closure_defect"] == 5.0e-13
    assert not contract["may_change_reaction_support_or_physical_rows"]
    assert not contract["may_relax_ledger_or_condition_gates"]


def test_c4f24e7_result_preserves_stops() -> None:
    summary = _read(RESULT, "summary.json")
    assert summary["analysis_only"]
    assert not summary["trajectory_executed"]
    assert not summary["physical_failure_detected"]
    assert not summary["physical_execution_authorized"]
    assert not summary["fixed_Q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_c4f24e7_checksums_close() -> None:
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
