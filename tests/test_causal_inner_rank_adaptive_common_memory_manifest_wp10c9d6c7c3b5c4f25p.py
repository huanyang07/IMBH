from __future__ import annotations

import hashlib
import json

import run_causal_inner_rank_adaptive_common_memory_manifest_wp10c9d6c7c3b5c4f25p as f25p


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_parent_rejection_is_preserved_and_reassessment_is_authorized():
    summary, metrics, hashes = f25p._validate_parent()
    assert not summary["passed"]
    assert summary["common_numerical_passed"]
    assert not summary["memory_passed"]
    assert metrics["common_promoted_union_dimension"] == 36
    assert "summary.json" in hashes


def test_rank_and_memory_ladders_obey_R320_cap():
    contract = f25p._contract()
    assert contract["common_rank_ladder"]["candidates"] == list(range(18, 37, 2))
    for rank in f25p.COMMON_RANK_CANDIDATES:
        orders = f25p._memory_orders(rank)
        assert orders[0] == 96
        assert 162 + rank + orders[-1] == 320
        assert all(162 + rank + order <= 320 for order in orders)


def test_contract_forbids_new_truth_work_and_physical_failure_claim():
    contract = f25p._contract()
    budget = contract["execution_budget"]
    assert budget["allowed_new_nonlinear_roots"] == 0
    assert budget["allowed_propagated_states"] == 0
    assert budget["allowed_new_full_560_direction_generator_assemblies"] == 0
    assert not contract["claim_boundary"]["physical_failure_can_be_declared"]


def test_canonical_manifest_when_available():
    summary_path = f25p.ARTIFACT_DIRECTORY / "summary.json"
    if not summary_path.exists():
        return
    summary = _read(summary_path)
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["authorized_next"] == "WP10c9d6c7c3b5c4f25q"
    for line in (f25p.ARTIFACT_DIRECTORY / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        actual = hashlib.sha256((f25p.ARTIFACT_DIRECTORY / name).read_bytes()).hexdigest()
        assert actual == expected
