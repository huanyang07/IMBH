from __future__ import annotations

import hashlib

import run_causal_inner_entropy_complete_multi_patch_growth_and_fast_slaving_manifest_wp10c9d6c7c3b5c4f25fizfn as target


def test_parent_two_patch_certificate_is_locked() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["two_patch_path_certified"]
    assert validated["summary"]["accepted_absolute_horizon_seconds"] == 8.0e-3


def test_saved_stable_bundle_supports_transport() -> None:
    diagnostics = target._evidence_diagnostics()
    gates = target._contract()["stable_bundle_evidence_gates"]
    assert diagnostics["patch_1_spectral_abscissa_per_second"] <= 0.0
    assert diagnostics["patch_2_spectral_abscissa_per_second"] <= 0.0
    assert (
        diagnostics["maximum_fast_efold_to_cycle_ratio"]
        <= gates["maximum_fast_efold_to_cycle_ratio"]
    )
    assert (
        diagnostics["maximum_stable_bundle_principal_angle_degrees"]
        <= gates["maximum_interpatch_stable_bundle_angle_degrees"]
    )
    assert (
        diagnostics["physical_rate_jacobian_relative_infinity_drift"]
        <= gates["maximum_physical_rate_jacobian_relative_infinity_drift"]
    )


def test_transport_contract_is_fail_closed_and_cost_reducing() -> None:
    contract = target._contract()
    assert contract["preserved_boundaries"]["all_80_macro_coordinates_remain_dynamic"]
    assert contract["preserved_boundaries"]["no_instantaneous_global_equilibrium_substitution"]
    assert contract["independent_transport_validation"]["new_JVP_truth_calls"] == 8
    assert contract["acquisition_cost"]["transported_patch_truth_calls"] == 9
    assert (
        contract["acquisition_cost"]["transported_to_full_truth_call_fraction"]
    ) <= contract["acquisition_cost"]["maximum_transported_to_full_truth_call_fraction"]
    assert not contract["claim_boundary"]["complete_cycle_execution_authorized"]


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
    assert summary["moving_full_macro_stable_bundle_supported"]
    assert summary["transported_third_patch_execution_authorized"]
    assert not summary["complete_cycle_execution_authorized"]
