import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "results/canonical"
    / "causal_inner_face36_fixed_q_history_implementation_preflight_"
    "wp10c9d6c7c3b5c4f24e0"
)


def _read(name: str) -> dict:
    return json.loads((ARTIFACT / name).read_text(encoding="utf-8"))


def test_c4f24e0_certifies_only_the_history_implementation() -> None:
    summary = _read("summary.json")
    assert summary["passed"]
    assert not summary["trajectory_executed"]
    assert summary["focused_tests_passed"]
    assert summary["implementation_contract_passed"]
    assert summary["physical_history_execution_authorized"]
    assert not summary["fixed_Q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_c4f24e0_preserves_increment_primary_and_restart_contracts() -> None:
    contract = _read("implementation_contract.json")
    assert contract["all_checks_passed"]
    assert contract["binding_temporal_form"] == "increment_primary_complete_BDF"
    assert contract["direct_rate_role"] == "post_root_parity_only"
    assert contract["binding_reaction_basis"] == "frozen_normalized"
    assert contract["maximum_complete_Jacobian_assemblies_per_root"] == 1
    assert all(contract["checks"].values())


def test_c4f24e0_focused_tests_passed_at_the_execution_commit() -> None:
    results = _read("test_results.json")
    assert results["passed"]
    assert results["returncode"] == 0
    assert "18 passed" in results["stdout"]


def test_c4f24e0_provenance_hashes_the_committed_sources() -> None:
    provenance = _read("provenance.json")
    assert provenance["tracked_worktree_clean_at_start"]
    for relative, digest in provenance["source_hashes"].items():
        contents = subprocess.run(
            ("git", "show", f"{provenance['execution_commit']}:{relative}"),
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert hashlib.sha256(contents).hexdigest() == digest


def test_c4f24e0_checksum_manifest_is_complete() -> None:
    entries = {}
    for line in (ARTIFACT / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    assert set(entries) == {
        "implementation_contract.json",
        "provenance.json",
        "summary.json",
        "test_results.json",
    }
    for name, digest in entries.items():
        assert hashlib.sha256((ARTIFACT / name).read_bytes()).hexdigest() == digest
