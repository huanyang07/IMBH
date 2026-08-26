from __future__ import annotations

import hashlib

import numpy as np

import run_causal_inner_entropy_complete_transported_third_macro_patch_execution_wp10c9d6c7c3b5c4f25fizfo as target


def test_transport_manifest_is_locked() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["transported_third_patch_execution_authorized"]
    assert validated["contract"]["acquisition_cost"]["transported_patch_truth_calls"] == 9


def test_saved_8ms_anchor_is_complete() -> None:
    with np.load(target.PATCH_2_ARRAYS) as archive:
        assert archive["patch_2_macro_states"].shape == (5, 16, 5)
        assert archive["endpoint_8ms_primitive_charts"].shape == (112, 7)
        assert archive["endpoint_8ms_truth_packed_outputs"].shape == (115,)


def test_execution_preserves_cycle_boundary() -> None:
    assert target.AUTHORIZED_NEXT_ON_PASS.startswith("definitions_only_")
    assert "cycle_readiness_manifest" in target.AUTHORIZED_NEXT_ON_PASS


def test_canonical_package_closes_if_present() -> None:
    directory = target.CANONICAL_DIRECTORY
    if not directory.exists():
        return
    for line in (directory / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
    summary = target._utils()._read_json(directory / "summary.json")
    metrics = target._utils()._read_json(
        directory / "transported_patch_metrics.json"
    )
    assert metrics["new_truth_operator_calls"] == 9
    assert not summary["complete_cycle_execution_authorized"]
    if summary["passed"]:
        assert summary["transported_third_patch_certified"]
