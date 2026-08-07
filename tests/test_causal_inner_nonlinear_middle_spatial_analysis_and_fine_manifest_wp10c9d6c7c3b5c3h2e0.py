from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_nonlinear_middle_spatial_analysis_and_fine_manifest_wp10c9d6c7c3b5c3h2e0 as h2e0  # noqa: E402


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_middle_analysis_selects_full_fine_anchor():
    summary = _read(h2e0.SUMMARY_PATH)
    ratios = summary["analysis"][
        "generic_surrogate_to_spatial_difference_ratios"
    ]
    assert summary["passed"]
    assert ratios["state"] < h2e0.SURROGATE_TO_SPATIAL_GATE
    assert ratios["instantaneous_Tier_I"] > h2e0.SURROGATE_TO_SPATIAL_GATE
    assert ratios["cumulative_Tier_I"] > h2e0.SURROGATE_TO_SPATIAL_GATE
    assert summary["full_fine_generic_nonlinear_anchor_required"]


def test_fine_manifest_preserves_binding_scope():
    manifest = _read(h2e0.MANIFEST_PATH)
    assert manifest["definitions_only"]
    assert not manifest["propagation_executed"]
    assert manifest["layout"] == h2e0.FINE_LAYOUT
    assert manifest["active_coupling_face"] == 192
    assert manifest["full_fine_generic_anchor_selected_by_evidence"]
    assert len(manifest["required_trajectories"]) == 3
    assert manifest["execution_contract"]["durable_checkpoint_after_every_target"]
    assert manifest["execution_contract"]["base_and_anchor_last_step_bitwise_replay"]
    assert manifest["spatial_gates_unchanged"] == h2e0.g1.SPATIAL_GATES


def test_reduction_remains_blocked():
    summary = _read(h2e0.SUMMARY_PATH)
    assert summary["fine_cost_bounded_propagation_authorized"]
    assert not summary["middle_fine_5ms_spatial_certificate_issued"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_canonical_hashes_close():
    entries = {}
    for line in (h2e0.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        digest, name = line.split("  ", 1)
        entries[name] = digest
    for name, digest in entries.items():
        assert _sha256(h2e0.CANONICAL_DIRECTORY / name) == digest
