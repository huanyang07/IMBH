import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "results/canonical"
    / "causal_inner_face36_fixed_q_exact_increment_storage_repair_manifest_"
    "wp10c9d6c7c3b5c4f24e5"
)


def _read(name: str) -> dict:
    return json.loads((ARTIFACT / name).read_text(encoding="utf-8"))


def test_c4f24e5_manifest_preserves_increment_primary_contract() -> None:
    summary = _read("summary.json")
    contract = _read("execution_manifest.json")
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["implementation_authorized"]
    assert not summary["physical_history_ladder_execution_authorized"]
    assert contract["binding_temporal_form"] == "increment_primary"
    assert contract["direct_rate_role"].endswith("parity_audit_only")
    assert contract["require_exact_endpoint_reconstruction"]
    assert contract["require_inactive_affine_reconstruction"]
    assert contract["maximum_saved_root_scaled_residual"] == 1.0e-10
    assert not contract["may_change_physical_equations"]
    assert not contract["may_change_row_scales_or_merit_norm"]
    assert not contract["may_relax_nonlinear_residual_gate"]


def test_c4f24e5_manifest_checksums_close() -> None:
    entries = {}
    for line in (ARTIFACT / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    assert set(entries) == {
        "execution_manifest.json",
        "provenance.json",
        "summary.json",
    }
    for name, digest in entries.items():
        assert hashlib.sha256((ARTIFACT / name).read_bytes()).hexdigest() == digest
