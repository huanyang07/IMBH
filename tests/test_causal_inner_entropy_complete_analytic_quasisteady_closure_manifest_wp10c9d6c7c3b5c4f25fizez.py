from __future__ import annotations

import hashlib

import run_causal_inner_entropy_complete_analytic_quasisteady_closure_manifest_wp10c9d6c7c3b5c4f25fizez as target


def test_negative_generic_root_result_is_preserved() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["generic_fixed_point_Newton_rejected"]
    assert validated["metrics"]["propagated_states"] == 0


def test_partial_equilibrium_keeps_the_two_unclosed_fast_coordinates() -> None:
    state = target._contract()["mathematical_state"]
    assert state["cellwise_online_dimension"] == 5
    assert state["radial_drift_is_not_forced_to_zero"]
    assert state["causal_stress_is_not_forced_to_its_instantaneous_alpha_target"]
    assert state["vertical_velocity_over_c_is_zero"]


def test_online_cycle_cost_boundary_is_explicit() -> None:
    split = target._contract()["offline_online_split"]
    assert split["online_truth_calls_per_macrostep"] == 0
    assert split["maximum_online_dimension"] <= 96
    assert split["maximum_macrosteps_per_cycle"] == 100_000
    assert split["target_complete_cycle_wall_days"] == 3.0


def test_no_atlas_solver_or_cycle_is_authorized() -> None:
    claim = target._contract()["claim_boundary"]
    assert claim["local_reconstruction_implementation_authorized"]
    assert not claim["slow_flux_atlas_authorized"]
    assert not claim["online_macro_solver_authorized"]
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
    assert summary["definitions_only"]
    assert summary["local_reconstruction_implementation_authorized"]
    assert not summary["complete_cycle_execution_authorized"]
