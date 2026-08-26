from __future__ import annotations

import hashlib

import numpy as np

import run_causal_inner_entropy_complete_structure_preserving_macro_integrator_implementation_wp10c9d6c7c3b5c4f25fizfk as target


def test_manifest_and_integrator_source_are_locked() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["bounded_macro_propagation_authorized"]
    assert validated["contract"]["bounded_pilot"]["accepted_macrosteps"] == 4
    assert validated["contract"]["bounded_pilot"]["maximum_new_truth_operator_calls"] == 1


def test_certified_atlas_builds_the_expected_affine_system() -> None:
    with np.load(target.ATLAS_ARRAYS) as archive:
        atlas = target._atlas(archive)
    system = target.ExactAffineMacroSystem.from_atlas(atlas)
    assert system.normalized_rate_matrix.shape == (80, 80)
    assert system.augmented_generator.shape == (81, 81)
    assert np.max(np.real(np.linalg.eigvals(system.normalized_rate_matrix))) < 0.0


def test_canonical_package_closes_if_present() -> None:
    directory = target.CANONICAL_DIRECTORY
    if not directory.exists():
        return
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
    summary = target._utils()._read_json(directory / "summary.json")
    metrics = target._utils()._read_json(directory / "macro_integrator_metrics.json")
    assert summary["accepted_macrosteps"] == 4
    assert not summary["complete_cycle_execution_authorized"]
    assert metrics["new_truth_operator_calls"] == 1
    if summary["passed"]:
        assert summary["exact_affine_macro_integrator_certified"]
        assert metrics["suffix_replay_bitwise"]
