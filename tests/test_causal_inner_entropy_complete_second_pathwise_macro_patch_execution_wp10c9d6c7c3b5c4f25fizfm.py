from __future__ import annotations

import hashlib

import numpy as np

import run_causal_inner_entropy_complete_second_pathwise_macro_patch_execution_wp10c9d6c7c3b5c4f25fizfm as target


def test_pathwise_manifest_and_input_are_locked() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["second_patch_execution_authorized"]
    assert validated["contract"]["budgets"]["maximum_new_truth_operator_calls"] == 39


def test_saved_integrator_endpoint_is_available() -> None:
    with np.load(target.INTEGRATOR_ARRAYS) as archive:
        assert archive["macro_states"].shape == (5, 16, 5)
        assert archive["endpoint_reconstructed_primitive_charts"].shape == (112, 7)
        assert archive["endpoint_truth_packed_outputs"].shape == (115,)


def test_coloring_supports_are_disjoint() -> None:
    for color in range(3):
        supports = [set(target._support_rows(cell)) for cell in range(color, 16, 3)]
        for first, left in enumerate(supports):
            for right in supports[first + 1 :]: assert left.isdisjoint(right)


def test_canonical_package_closes_if_present() -> None:
    directory = target.CANONICAL_DIRECTORY
    if not directory.exists(): return
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1); assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
    summary = target._utils()._read_json(directory / "summary.json")
    metrics = target._utils()._read_json(directory / "pathwise_patch_metrics.json")
    assert metrics["new_truth_operator_calls"] == 39
    assert not summary["complete_cycle_execution_authorized"]
    if summary["passed"]: assert summary["two_patch_path_certified"]
