from __future__ import annotations

import json

import numpy as np

import run_causal_inner_reduced_slow_atlas_integrator_manifest_wp10c9d6c7c3b5c4f25d as f25d


def test_validated_field_authorizes_only_definitions():
    frozen = f25d._validate_parent(require_clean=False)
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["classification"] == f25d.parent.PASS_CLASSIFICATION
    assert frozen["summary"]["authorized_next"] == (
        "definitions_only_reduced_slow_atlas_integrator_manifest"
    )
    assert frozen["prior_summary"]["selected_architecture"] == (
        "cellwise_Q5_FV_plus_a2_finite_memory_hybrid"
    )


def test_three_layers_separate_truth_atlas_and_online_macrostate():
    frozen = f25d._validate_parent(require_clean=False)
    runtime = f25d._runtime_budget(frozen)
    contract = f25d._architecture_contract(runtime)
    layers = contract["three_layer_architecture"]
    assert layers["offline_truth_layer"]["online_calls"] == 0
    assert layers["offline_atlas_layer"]["direct_cycle_integration_forbidden"]
    assert layers["online_macro_layer"]["continuous_dimension_by_memory_order"] == {
        "0": 82,
        "2": 84,
        "4": 86,
        "6": 88,
    }
    assert layers["online_macro_layer"]["online_truth_calls"] == 0
    assert layers["online_macro_layer"]["online_atlas_microbursts"] == 0


def test_runtime_requires_macro_timescale_elimination():
    runtime = f25d._runtime_budget(f25d._validate_parent(require_clean=False))
    assert runtime["direct_surrogate_wall_years_per_cycle"] > 100.0
    assert runtime["minimum_average_macrostep_seconds"] > 5.0
    assert runtime["reference_100k_macrostep_rhs_wall_fraction"] < 0.10
    assert runtime["required_architectural_change"] == (
        "eliminate_fast_stability_scale_online"
    )


def test_domain_guard_forbids_polynomial_extrapolation():
    contract = f25d._architecture_contract(
        f25d._runtime_budget(f25d._validate_parent(require_clean=False))
    )
    guard = contract["atlas_and_domain_guard"]
    assert not guard["global_forward_half_space_claimed"]
    assert guard["unbounded_polynomial_extrapolation_forbidden"]
    assert guard["leave_trust_domain_action"] == (
        "stop_and_request_offline_patch_expansion"
    )


def test_immediate_preflight_is_no_truth_and_fail_closed():
    contract = f25d._architecture_contract(
        f25d._runtime_budget(f25d._validate_parent(require_clean=False))
    )
    preflight = contract["immediate_local_slaving_preflight"]
    assert preflight["new_exact_rate_calls_equal"] == 0
    assert preflight["new_nonlinear_fixed_Q_roots_equal"] == 0
    assert preflight["binding_gates"]["minimum_spectral_gap_ratio"] == 10.0
    assert contract["decision"]["no_gap_classification"].startswith(
        "compact_slow_graph_rejected"
    )


def test_canonical_manifest_if_present():
    if not f25d.CANONICAL_DIRECTORY.exists():
        return
    f25d._checksums(f25d.CANONICAL_DIRECTORY)
    summary = json.loads(
        (f25d.CANONICAL_DIRECTORY / "summary.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (f25d.CANONICAL_DIRECTORY / "readiness_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    seed = f25d._load_npz(f25d.CANONICAL_DIRECTORY / "local_atlas_seed.npz")
    assert summary["passed"] and summary["definitions_only"]
    assert summary["classification"] == f25d.CLASSIFICATION
    assert summary["authorized_next"] == f25d.AUTHORIZED_NEXT
    assert not summary["local_field_direct_cycle_integration_authorized"]
    assert metrics["passed"] and all(metrics["checks"].values())
    assert seed["seed_local_coordinates"].shape == (13, 470)
    assert seed["seed_exact_full_rates_per_second"].shape == (13, 560)
    assert np.linalg.matrix_rank(
        np.c_[np.ones(13), seed["seed_active_coordinates"]]
    ) == 4
