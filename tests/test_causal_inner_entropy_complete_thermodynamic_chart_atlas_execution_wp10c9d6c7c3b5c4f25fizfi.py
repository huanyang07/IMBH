from __future__ import annotations

import hashlib

import numpy as np

import run_causal_inner_entropy_complete_thermodynamic_chart_atlas_execution_wp10c9d6c7c3b5c4f25fizfi as target


def test_thermodynamic_chart_manifest_and_sources_are_locked() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["raw_coordinate_atlas_rejection_preserved"]
    assert validated["contract"]["atlas_audit"]["maximum_new_truth_operator_calls"] == 38
    assert validated["contract"]["coordinate_redesign"]["maximum_pullback_condition_number"] == 1.0e5


def test_radius_one_coloring_supports_are_disjoint() -> None:
    for color in range(3):
        supports = [set(target._support_rows(cell)) for cell in range(color, 16, 3)]
        for first, left in enumerate(supports):
            for right in supports[first + 1 :]:
                assert left.isdisjoint(right)


def test_saved_truth_samples_have_expected_shape() -> None:
    with np.load(target.TRUTH_ARRAYS) as archive:
        charts, targets, outputs = target._sample(
            archive, "primary_20ms", "base"
        )
    macro = target.restrict_entropy_complete_macro(targets, charts)
    assert macro.shape == (16, 5)
    assert target.pack_macro_outputs(outputs).shape == (target.OUTPUT_SIZE,)


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
    metrics = target._utils()._read_json(directory / "macro_atlas_metrics.json")
    assert summary["online_truth_calls_per_macrostep"] == 0
    assert metrics["new_truth_operator_calls"] == 38
    assert metrics["propagated_states"] == 0
    if summary["passed"]:
        assert summary["heldout_16ms_profiles_passed"]
        assert metrics["maximum_pullback_condition_number"] <= 1.0e5
