from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_metric_chart_atlas_overlap_preflight_wp10c9d6c7c3b5c4f25fid as target  # noqa: E402


def test_manifest_authorizes_only_overlap_preflight() -> None:
    lock = target._validate_manifest(require_clean=False)
    assert lock["summary"]["authorized_next"] == target.manifest.AUTHORIZED_NEXT
    assert lock["contract"]["scope"]["new_exact_free_field_calls"] == 0
    assert lock["contract"]["scope"]["new_trajectory_segments"] == 0


def test_policy_preserves_original_and_metric_residual_gates() -> None:
    policy = target._policy()
    assert policy.original_coordinate_tolerance == 1.0e-10
    assert policy.metric_coordinate_tolerance == 1.0e-9
    assert policy.gauge_tolerance == 1.0e-10
    assert policy.maximum_metric_augmented_condition == 10.0


def test_saved_overlap_witnesses_are_available() -> None:
    witness = target._selected_witnesses()
    assert witness["primitive_states"].shape == (2, 112, 5)
    assert witness["requested_coordinates470"].shape == (2, 470)
    assert tuple(witness["attempt_indices"]) == (82, 83)


def test_canonical_overlap_when_present() -> None:
    if not target.CANONICAL_DIRECTORY.exists():
        return
    helper = target._helper()
    helper._validate_checksums(target.CANONICAL_DIRECTORY)
    summary = helper._read(target.CANONICAL_DIRECTORY / "summary.json")
    metrics = helper._read(target.CANONICAL_DIRECTORY / "overlap_metrics.json")
    assert summary["classification"] == metrics["classification"]
    assert summary["original_physics_preserved"]
    assert not summary["new_trajectory"]
    assert metrics["gate_values"]["new_exact_free_field_calls"] == 0
