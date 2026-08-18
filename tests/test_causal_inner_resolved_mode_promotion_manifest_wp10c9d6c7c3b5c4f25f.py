from __future__ import annotations

import hashlib
import json

import run_causal_inner_resolved_mode_promotion_manifest_wp10c9d6c7c3b5c4f25f as f25f


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_parent_certificate_is_hash_locked_and_not_a_physical_failure():
    summary, spectrum, hashes = f25f._validate_parent()
    assert summary["passed"]
    assert not summary["physical_failure_detected"]
    assert spectrum["unstable_unresolved_pole_count"] == 24
    assert "projection.npz" in hashes


def test_ordered_schur_attribution_is_diagnostic_only():
    stage = f25f._contract()["ordered_real_schur_attribution"]
    assert stage["ordered_partition"] == "stable_first_then_nonstable"
    assert stage["full_generator_spectrum_is_diagnostic_only"]
    assert stage["compressed_nonstable_poles_are_not_a_physical_instability_claim"]
    assert stage["stability_margin_per_second"] > 0.0


def test_promotion_is_complete_bounded_and_fail_closed():
    stage = f25f._contract()["algebraic_promotion"]
    gates = stage["pass_requires"]
    assert stage["promote_every_nonstable_compressed_real_schur_coordinate"]
    assert stage["maximum_promoted_dimension"] == 32
    assert stage["maximum_augmented_resolved_dimension"] == 114
    assert gates["promoted_dimension"] == 24
    assert gates["remaining_unresolved_spectral_abscissa_per_second_max"] < 0.0


def test_execution_budget_reuses_saved_generator_and_fits_no_memory():
    budget = f25f._contract()["execution_budget"]
    assert budget["allowed_new_nonlinear_roots"] == 0
    assert budget["allowed_propagated_states"] == 0
    assert budget["allowed_new_full_560_direction_descriptor_assemblies"] == 0
    assert budget["saved_complete_generator_must_be_reused"]
    assert budget["allowed_memory_coefficients_fit"] == 0


def test_claim_boundary_keeps_online_cycle_blocked():
    claims = f25f._contract()["claim_boundary"]
    assert not claims["physical_instability_claim_authorized"]
    assert not claims["memory_fit_authorized_in_this_package"]
    assert not claims["full_anchor_campaign_authorized"]
    assert not claims["online_solver_authorized"]
    assert not claims["predictive_cycle_authorized"]
    assert not claims["reduced_slow_evolution_authorized"]


def test_canonical_manifest_when_available():
    summary_path = f25f.ARTIFACT_DIRECTORY / "summary.json"
    if not summary_path.exists():
        return
    summary = _read(summary_path)
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["algebraic_promotion_authorized"]
    assert not summary["physical_instability_claim_authorized"]
    for line in (f25f.ARTIFACT_DIRECTORY / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        actual = hashlib.sha256((f25f.ARTIFACT_DIRECTORY / name).read_bytes()).hexdigest()
        assert actual == expected
