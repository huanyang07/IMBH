from __future__ import annotations

import hashlib
from pathlib import Path

import run_causal_inner_advective_subspace_path_diagnosis_manifest_wp10c9d6c7c3b5c4f25fizee4 as target


def test_parent_failure_is_preserved() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["classification"] == target.parent.HYPERBOLICITY_FAILURE
    contract = target._contract()
    assert contract["parent_negative_result"]["preserved_as_binding"]
    assert contract["parent_negative_result"]["coarse_endpoint_gate_remains_failed"]


def test_path_and_projector_gates_are_prospective() -> None:
    contract = target._contract()
    path = contract["frozen_path"]
    gates = contract["binding_gates"]
    claims = contract["claim_boundary"]
    assert tuple(path["nested_node_counts"]) == (33, 65, 129)
    assert gates["minimum_cluster_complement_gap_over_c"] == 1.0e-4
    assert gates["minimum_adjacent_subspace_cosine"] == 0.99
    assert gates["maximum_projector_jump_refinement_ratio"] == 0.60
    assert gates["all_gates_required"] and gates["fail_closed"]
    assert claims["path_diagnosis_authorized"]
    assert not claims["diagnostic_gate_replacement_authorized"]
    assert not claims["trajectory_authorized"]


def test_canonical_package_closes_if_present() -> None:
    directory = target.CANONICAL_DIRECTORY
    if not directory.exists():
        return
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        assert actual == expected
    summary = target._utils()._read_json(directory / "summary.json")
    assert summary["classification"] == target.CLASSIFICATION
    assert summary["definitions_only"]
    assert summary["parent_negative_result_preserved"]
    assert summary["authorized_next"] == target.AUTHORIZED_NEXT
    assert not summary["diagnostic_gate_replacement_authorized"]


def test_declared_paths_are_workspace_relative() -> None:
    for relative in (
        target.THIS_RUNNER,
        target.THIS_TEST,
        target.PHYSICAL_SOURCE,
        target.PHYSICAL_TEST,
        target.REPORT_RELATIVE,
    ):
        assert not Path(relative).is_absolute()
