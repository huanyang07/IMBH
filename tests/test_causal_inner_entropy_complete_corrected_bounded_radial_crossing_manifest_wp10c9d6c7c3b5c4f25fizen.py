from __future__ import annotations

import hashlib

import run_causal_inner_entropy_complete_corrected_bounded_radial_crossing_manifest_wp10c9d6c7c3b5c4f25fizen as target


def test_substep_certificate_is_locked() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["classification"] == target.parent.PASS_CLASSIFICATION
    assert validated["summary"]["selected_timestep_seconds"] == target.TIMESTEP_SECONDS
    assert validated["metrics"]["new_trajectory_steps"] == 0


def test_corrected_crossing_preserves_horizon() -> None:
    contract = target._contract()
    assert contract["time_integrator"]["timestep_seconds"] == 7.8125e-6
    assert contract["time_integrator"]["accepted_steps"] == 32
    assert contract["time_integrator"]["horizon_seconds"] == 2.5e-4
    assert contract["time_integrator"]["same_horizon_as_rejected_coarse_crossing"]


def test_replay_and_first_endpoint_are_binding() -> None:
    contract = target._contract()
    assert contract["time_integrator"]["first_endpoint_must_match_diagnostic_bitwise"]
    assert contract["time_integrator"]["replay_checkpoint_step"] == 28
    assert contract["time_integrator"]["replay_suffix_steps"] == 4
    assert contract["additional_gates"]["suffix_replay_bitwise"]


def test_original_physical_gates_are_unchanged() -> None:
    assert target._contract()["binding_gates"] == target.original._contract()["binding_gates"]


def test_canonical_package_closes_if_present() -> None:
    directory = target.CANONICAL_DIRECTORY
    if not directory.exists(): return
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
    summary = target._utils()._read_json(directory / "summary.json")
    assert summary["definitions_only"]
    assert summary["maximum_new_trajectory_steps"] == 32
    assert summary["new_trajectory_steps"] == 0
