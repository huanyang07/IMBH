from __future__ import annotations

import hashlib

import numpy as np

import run_causal_inner_entropy_complete_hydrostatic_inverse_jvp_refinement_execution_wp10c9d6c7c3b5c4f25fizfc as target


def test_refinement_manifest_and_original_rejection_are_locked() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["original_inverse_JVP_rejection_preserved"]
    assert validated["summary"]["four_truth_samples_preserved"]
    assert validated["contract"]["refinement"][
        "new_seven_field_radial_operator_calls"
    ] == 0


def test_refinement_count_and_gate_are_prospective() -> None:
    refinement = target.parent._contract()["refinement"]
    combinations = (
        len(refinement["profiles"])
        * len(refinement["selected_cells"])
        * len(refinement["ordered_central_steps"])
    )
    assert 2 * combinations == refinement["maximum_new_nonlinear_slow_invariant_solves"]
    assert refinement["maximum_JVP_relative_defect"] == 2.0e-5
    assert refinement["reconstruction_tolerance"] == 1.0e-12


def test_deterministic_directions_are_normalized_and_distinct() -> None:
    directions = target._deterministic_directions()
    assert tuple(directions) == ("primary_20ms", "heldout_16ms")
    combined = np.concatenate(tuple(directions.values()))
    np.testing.assert_allclose(np.linalg.norm(combined, axis=1), 1.0)
    assert np.unique(combined, axis=0).shape[0] == combined.shape[0]


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
    metrics = target._utils()._read_json(directory / "refinement_metrics.json")
    assert summary["new_seven_field_operator_calls"] == 0
    assert summary["propagated_states"] == 0
    assert metrics["requested_local_nonlinear_solves"] == 84
    if summary["passed"]:
        assert summary["hydrostatic_invariant_inverse_certified_by_refinement"]
        assert metrics["maximum_JVP_relative_defect"] <= 2.0e-5
