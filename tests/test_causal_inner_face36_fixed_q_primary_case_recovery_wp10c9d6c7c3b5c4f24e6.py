import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT
    / "results/canonical"
    / "causal_inner_face36_fixed_q_primary_case_recovery_manifest_"
    "wp10c9d6c7c3b5c4f24e6"
)
RESULT = (
    ROOT
    / "results/canonical"
    / "causal_inner_face36_fixed_q_primary_case_recovery_"
    "wp10c9d6c7c3b5c4f24e6"
)


def _read(directory: Path, name: str) -> dict:
    return json.loads((directory / name).read_text(encoding="utf-8"))


def test_c4f24e6_manifest_is_bounded_and_preserves_stops() -> None:
    summary = _read(MANIFEST, "summary.json")
    contract = _read(MANIFEST, "execution_manifest.json")
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["primary_case_execution_authorized"]
    assert not summary["remaining_history_ladder_execution_authorized"]
    assert contract["case"] == "primary_coarse"
    assert contract["binding_temporal_form"] == "exact_increment_primary"
    assert contract["maximum_scaled_residual"] == 1.0e-10
    assert not contract["may_change_physical_equations"]
    assert not contract["may_relax_residual_or_physical_gates"]


def test_c4f24e6_result_preserves_reduction_stops() -> None:
    summary = _read(RESULT, "summary.json")
    assert summary["bounded_primary_case_only"]
    assert not summary["physical_failure_detected"]
    assert summary["parent_rejections_preserved"]
    assert not summary["remaining_history_ladder_execution_authorized"]
    assert not summary["fixed_Q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_c4f24e6_checksum_manifests_close() -> None:
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
