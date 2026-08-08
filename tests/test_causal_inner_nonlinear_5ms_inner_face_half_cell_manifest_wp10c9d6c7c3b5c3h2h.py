from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_nonlinear_5ms_inner_face_half_cell_manifest_wp10c9d6c7c3b5c3h2h as h2h  # noqa: E402


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_is_operator_neutral_and_definitions_only():
    manifest = _read(h2h.MANIFEST_PATH)
    summary = _read(h2h.SUMMARY_PATH)
    assert summary["passed"] and summary["definitions_only"]
    assert manifest["parent_rejection_preserved"]
    assert not manifest["propagation_executed"]
    assert not manifest["operator_changed"]
    assert not manifest["production_defaults_changed"]


def test_prefixes_and_complete_balance_are_frozen():
    manifest = _read(h2h.MANIFEST_PATH)
    assert tuple(manifest["common_prefix_coarse_face_indices"]) == h2h.COMMON_PREFIX_COARSE_FACE_INDICES
    assert tuple(manifest["common_prefix_face_multipliers"]) == (1, 2, 4)
    balance = manifest["diagnostics"]["complete_BDF_prefix_balance"]
    assert balance["use_actual_layout_owned_BDF_histories"]
    assert balance["evaluate_only_committed_accepted_steps"]
    assert "minus_inner_face_flux" in balance["blocks"]
    assert "outer_common_face_flux" in balance["blocks"]
    assert "mapped_temporal_storage" in balance["blocks"]


def test_decision_tree_does_not_preselect_a_correction():
    manifest = _read(h2h.MANIFEST_PATH)
    decisions = tuple(manifest["decision_tree"])
    assert any("extraction_surface" in item for item in decisions)
    assert any("half_cell_candidate" in item for item in decisions)
    assert any("space_storage" in item for item in decisions)
    assert any("source_consistency" in item for item in decisions)
    assert not _read(h2h.SUMMARY_PATH)["new_propagation_authorized"]


def test_reduction_and_later_duration_remain_blocked():
    summary = _read(h2h.SUMMARY_PATH)
    assert not summary["fourth_duration_rung_manifest_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    assert summary["authorized_next"].endswith("operator_neutral_inner_face_half_cell_audit")


def test_canonical_hashes_close():
    entries = {}
    for line in (h2h.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        entries[name] = digest
    assert entries
    for name, digest in entries.items():
        assert _sha256(h2h.CANONICAL_DIRECTORY / name) == digest
