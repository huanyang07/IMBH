import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_adaptive_refresh_refined_ladder_manifest_"
    "wp10c9d6c7c3b5c4f24e13"
)
PRIMARY = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_adaptive_refresh_primary_"
    "wp10c9d6c7c3b5c4f24e11"
)
HELDOUT = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_adaptive_refresh_heldout_"
    "wp10c9d6c7c3b5c4f24e12"
)


def _read(directory: Path, name: str) -> dict:
    return json.loads((directory / name).read_text(encoding="utf-8"))


def test_c4f24e13_manifest_freezes_only_refined_cases() -> None:
    summary = _read(MANIFEST, "summary.json")
    contract = _read(MANIFEST, "execution_manifest.json")
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["next_case"] == "primary_middle"
    assert summary["refined_ladder_execution_authorized"]
    assert contract["reused_cases"] == ["primary_coarse", "heldout_coarse"]
    assert contract["refined_cases"] == [
        "primary_middle",
        "heldout_middle",
        "primary_fine",
        "heldout_fine",
    ]
    assert contract["timesteps_seconds"] == [1.0e-7, 5.0e-8, 2.5e-8]


def test_c4f24e13_preserves_adaptive_and_scientific_gates() -> None:
    summary = _read(MANIFEST, "summary.json")
    contract = _read(MANIFEST, "execution_manifest.json")
    assert contract["binding_temporal_form"] == "exact_increment_primary"
    assert contract["direct_rate_form"] == "post_root_parity_audit_only"
    assert contract["exact_jacobian_refresh_policy"] == "on_line_search_failure"
    assert contract["maximum_exact_jacobian_assemblies_per_root"] == 2
    assert contract["minimum_state_rate_convergence_order"] == 0.9
    assert contract["minimum_reaction_action_convergence_order"] == 0.9
    assert contract["require_bitwise_restart_roundtrip"]
    assert contract["require_bitwise_BDF2_replay"]
    assert contract["fail_fast"]
    assert not contract["may_relax_any_gate"]
    assert not summary["one_Q_execution_manifest_authorized"]
    assert not summary["fixed_Q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_c4f24e13_parent_authorizations_are_positive() -> None:
    primary = _read(PRIMARY, "summary.json")
    heldout = _read(HELDOUT, "summary.json")
    assert primary["passed"]
    assert primary["heldout_retry_manifest_authorized"]
    assert heldout["passed"]
    assert heldout["refined_ladder_manifest_authorized"]


def test_c4f24e13_manifest_checksums_close() -> None:
    entries = {}
    for line in (MANIFEST / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    assert set(entries) == {
        "execution_manifest.json",
        "provenance.json",
        "summary.json",
    }
    for name, digest in entries.items():
        assert hashlib.sha256((MANIFEST / name).read_bytes()).hexdigest() == digest
