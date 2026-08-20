from __future__ import annotations

import numpy as np

import run_causal_inner_guarded_departure_amplitude_expansion_preflight_wp10c9d6c7c3b5c4f25cd as f25cd


def test_manifest_authorizes_exact_preflight():
    frozen = f25cd._validate_manifest(require_clean=False)
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["authorized_next"] == f25cd.WORK_PACKAGE
    assert frozen["directions"].shape == (10, 28)
    assert np.array_equal(frozen["amplitudes"], np.asarray((0.015, 0.02, 0.03)))


def test_retraction_adapter_uses_frozen_halving_line_search():
    frozen = f25cd._validate_manifest(require_clean=False)
    adapted = f25cd._retraction_contract(frozen["contract"], 0.015)
    assert adapted["binding_preflight_gates"]["maximum_final_scaled_component"] == 0.015
    assert adapted["exact_geometric_retraction"]["line_factors"] == list(f25cd.LINE_FACTORS)
    assert len(f25cd.LINE_FACTORS) == 12


def test_direction_design_matches_exact_chart_energy_basis():
    frozen = f25cd._validate_manifest(require_clean=False)
    _metrics, family = f25cd.prior_geometry.chart_tools._departure_family()
    defects = f25cd._direction_consistency(frozen["directions"], family)
    assert defects["frozen_energy_direction_maximum_absolute_defect"] <= 1.0e-14
    assert defects["all_direction_norm_maximum_absolute_defect"] <= 1.0e-14


def test_canonical_preflight_if_present():
    if not f25cd.CANONICAL_DIRECTORY.exists():
        return
    f25cd._checksums(f25cd.CANONICAL_DIRECTORY)
    summary = f25cd._read(f25cd.CANONICAL_DIRECTORY / "summary.json")
    metrics = f25cd._read(f25cd.CANONICAL_DIRECTORY / "preflight_metrics.json")
    assert summary["passed"]
    assert summary["classification"] in {
        f25cd.FULL_PASS_CLASSIFICATION,
        f25cd.PARTIAL_CLASSIFICATION,
        f25cd.FIRST_FAILURE_CLASSIFICATION,
    }
    assert summary["new_truth_calls"] == 0
    assert summary["stable_memory_remains_dynamic"]
    assert not summary["old_polynomial_extrapolation_used"]
    assert all(metrics["decision"]["budget_checks"].values())


def test_preflight_never_authorizes_cycle_directly():
    frozen = f25cd._validate_manifest(require_clean=False)
    decision = frozen["contract"]["fail_fast_decision"]
    assert not decision["physical_microburst_authorized"]
    assert not decision["predictive_cycle_authorized"]
    assert not decision["reduced_slow_evolution_authorized"]
