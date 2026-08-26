from __future__ import annotations

import hashlib

import run_causal_inner_entropy_complete_hydrostatic_invariant_reconstruction_implementation_wp10c9d6c7c3b5c4f25fizfa as target


def test_partial_equilibrium_manifest_and_source_are_locked() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"][
        "partial_equilibrium_Q3_plus_radial_stress_architecture_selected"
    ]
    assert validated["summary"]["local_reconstruction_implementation_authorized"]


def test_primary_and_heldout_design_is_prospective() -> None:
    validation = target.parent._contract()["offline_validation"]
    assert validation["primary_profile"] == "primary_20ms"
    assert validation["heldout_profile"] == "heldout_16ms"
    assert tuple(validation["selected_cell_indices"]) == target.SELECTED_CELLS


def test_truth_sampling_does_not_propagate_or_find_a_root() -> None:
    claim = target.parent._contract()["claim_boundary"]
    assert not claim["slow_flux_atlas_authorized"]
    assert not claim["complete_cycle_execution_authorized"]


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
    metrics = target._utils()._read_json(directory / "implementation_metrics.json")
    assert summary["new_nonlinear_roots"] == 0
    assert summary["propagated_states"] == 0
    if summary["passed"]:
        assert metrics["offline_seven_field_operator_calls"] == 4
        assert summary["primary_and_heldout_truth_samples_certified"]
