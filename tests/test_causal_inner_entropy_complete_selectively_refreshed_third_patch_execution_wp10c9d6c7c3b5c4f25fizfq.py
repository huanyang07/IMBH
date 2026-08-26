from __future__ import annotations

import hashlib

import numpy as np

import run_causal_inner_entropy_complete_selectively_refreshed_third_patch_execution_wp10c9d6c7c3b5c4f25fizfq as target


def test_selective_refresh_manifest_is_locked() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["selectively_refreshed_third_patch_execution_authorized"]
    assert validated["contract"]["saved_evidence_selection"]["selected_input_field_indices"] == [0, 1]
    assert validated["contract"]["acquisition_cost"]["selectively_refreshed_patch_truth_calls"] == 21


def test_saved_anchor_and_coloring_are_complete() -> None:
    with np.load(target.PATCH_2_ARRAYS) as archive:
        assert archive["endpoint_8ms_primitive_charts"].shape == (112, 7)
        assert archive["endpoint_8ms_truth_packed_outputs"].shape == (115,)
    supports = target.rejected_execution.parent.parent._support_rows
    for color in range(3):
        rows = [set(supports(cell)) for cell in range(color, 16, 3)]
        for index, left in enumerate(rows):
            for right in rows[index + 1 :]:
                assert left.isdisjoint(right)


def test_canonical_package_closes_if_present() -> None:
    directory = target.CANONICAL_DIRECTORY
    if not directory.exists(): return
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1); assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
    summary = target._utils()._read_json(directory / "summary.json")
    metrics = target._utils()._read_json(directory / "selective_refresh_metrics.json")
    assert metrics["new_truth_operator_calls"] == 21
    assert summary["unchanged_transport_rejection_preserved"]
    assert not summary["complete_cycle_execution_authorized"]
    if summary["passed"]: assert summary["selectively_refreshed_third_patch_certified"]
