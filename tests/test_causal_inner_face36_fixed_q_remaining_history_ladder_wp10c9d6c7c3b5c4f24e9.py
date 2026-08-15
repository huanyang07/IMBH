import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT
    / "results/canonical"
    / "causal_inner_face36_fixed_q_remaining_history_ladder_manifest_"
    "wp10c9d6c7c3b5c4f24e9"
)
HELDOUT = (
    ROOT
    / "results/canonical"
    / "causal_inner_face36_fixed_q_remaining_history_ladder_stage_"
    "heldout_coarse_wp10c9d6c7c3b5c4f24e9"
)


def _read(directory: Path, name: str) -> dict:
    return json.loads((directory / name).read_text(encoding="utf-8"))


def _checksums_close(directory: Path, names: set[str]) -> None:
    entries = {}
    for line in (directory / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    assert set(entries) == names
    for name, digest in entries.items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest


def test_c4f24e9_manifest_freezes_fail_fast_remaining_ladder() -> None:
    summary = _read(MANIFEST, "summary.json")
    contract = _read(MANIFEST, "execution_manifest.json")
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["next_case"] == "heldout_coarse"
    assert summary["remaining_ladder_execution_authorized"]
    assert contract["reused_case"] == "primary_coarse"
    assert contract["remaining_cases"][0] == "heldout_coarse"
    assert contract["minimum_state_rate_convergence_order"] == 0.9
    assert contract["minimum_reaction_action_convergence_order"] == 0.9
    assert not contract["may_relax_any_gate"]
    assert not summary["fixed_Q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_c4f24e9_heldout_stage_preserves_stops() -> None:
    summary = _read(HELDOUT, "summary.json")
    metrics = _read(HELDOUT, "metrics.json")
    assert summary["case"] == "heldout_coarse"
    assert summary["passed"] == metrics["passed"]
    assert not summary["one_Q_execution_manifest_authorized"]
    assert not summary["fixed_Q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_c4f24e9_checksums_close() -> None:
    _checksums_close(
        MANIFEST,
        {"execution_manifest.json", "provenance.json", "summary.json"},
    )
    _checksums_close(
        HELDOUT,
        {
            "contract.json",
            "decisive_arrays.npz",
            "metrics.json",
            "provenance.json",
            "summary.json",
        },
    )
