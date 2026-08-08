from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_nonlinear_5ms_inner_face_half_cell_audit_wp10c9d6c7c3b5c3h2h1 as h2h1  # noqa: E402


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_operator_neutral_audit_completed():
    summary = _read(h2h1.SUMMARY_PATH)
    assert summary["passed"]
    assert not summary["operator_changed"]
    assert not summary["production_defaults_changed"]
    assert not summary["propagation_executed"]
    assert summary["audit"]["maximum_incoming_excision_characteristics"] == 0


def test_prefix_and_primitive_decompositions_close():
    summary = _read(h2h1.SUMMARY_PATH)
    audit = summary["audit"]
    for payload in audit["final_discrete_step_diagnostics"].values():
        assert payload["maximum_component_closure_defect"] <= 1.0e-10
        assert payload["linear_residual_relative_defect"] <= 1.0e-10
    primitive = audit["inner_flux_primitive_path"]
    if primitive["path_contract_passed"]:
        assert primitive["maximum_path_closure_defect"] <= 1.0e-9
    else:
        assert primitive["maximum_path_closure_defect"] > 1.0e-9
        assert not primitive["stable_field_localizations"]


def test_decision_matches_frozen_branches():
    summary = _read(h2h1.SUMMARY_PATH)
    audit = summary["audit"]
    if audit["common_face_recovery"]["compact_recovery_selected"]:
        assert "extraction_surface" in summary["classification"]
    elif audit["inner_flux_primitive_path"]["stable_field_localizations"].get("mass"):
        near = audit["final_prefix_compensators"]["stable_near_inner_compensators"].get("mass")
        if near:
            assert "half_cell" in summary["classification"]


def test_later_duration_and_reduction_remain_blocked():
    summary = _read(h2h1.SUMMARY_PATH)
    assert not summary["five_ms_spatial_convergence_certified"]
    assert not summary["fourth_duration_rung_manifest_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_canonical_hashes_close():
    entries = {}
    for line in (h2h1.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        entries[name] = digest
    assert entries
    for name, digest in entries.items():
        assert _sha256(h2h1.CANONICAL_DIRECTORY / name) == digest
