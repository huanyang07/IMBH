from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_nonlinear_5ms_tier_i_localization_manifest_wp10c9d6c7c3b5c3h2g as h2g  # noqa: E402


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_freezes_all_four_discriminators():
    manifest = _read(h2g.MANIFEST_PATH)
    assert manifest["definitions_only"]
    assert not manifest["propagation_executed"]
    assert set(manifest["discriminators"]) == {
        "common_parent_export_map",
        "net_drive_balance",
        "time_window_error_energy",
        "tangent_nonlinear_pair_error",
    }
    assert manifest["failed_components"] == list(h2g.FAILED_COMPONENTS)


def test_selection_gates_are_prospective_and_strict():
    manifest = _read(h2g.MANIFEST_PATH)
    common = manifest["discriminators"]["common_parent_export_map"]
    balance = manifest["discriminators"]["net_drive_balance"]
    assert common["selection_gates"]["minimum_layout_map_alignment"] == 0.95
    assert common["selection_gates"]["maximum_common_state_error_fraction"] == 0.25
    assert balance["minimum_dominance_fraction"] == 0.70
    assert balance["minimum_error_alignment"] == 0.90
    assert len(manifest["decision_tree"]) == 6


def test_downstream_work_remains_blocked():
    summary = _read(h2g.SUMMARY_PATH)
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["operator_neutral_localization_authorized"]
    assert not summary["new_propagation_authorized"]
    assert not summary["fourth_duration_rung_manifest_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_canonical_hashes_close():
    entries = {}
    for line in (h2g.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        digest, name = line.split("  ", 1)
        entries[name] = digest
    assert entries
    for name, digest in entries.items():
        assert _sha256(h2g.CANONICAL_DIRECTORY / name) == digest
