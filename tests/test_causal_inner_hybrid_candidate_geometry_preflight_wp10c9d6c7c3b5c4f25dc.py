from __future__ import annotations

import numpy as np

import run_causal_inner_hybrid_candidate_geometry_preflight_wp10c9d6c7c3b5c4f25dc as f25dc


def test_state_selection_requires_accepted_identity() -> None:
    source = {
        "base__output_times": np.asarray((0.1,)),
        "base__accepted_times": np.asarray((0.1,)),
        "base__output_states": np.ones((1, 2, 5)),
        "base__accepted_states": np.ones((1, 2, 5)),
    }
    state, exact = f25dc._state_at(source, 0.1)
    assert exact
    assert state.shape == (2, 5)
    source["base__accepted_states"][0, 0, 0] = 2.0
    assert not f25dc._state_at(source, 0.1)[1]


def test_pilot_contract_keeps_candidates_unclassified() -> None:
    metrics = {
        "eligible_existing_atlas_geometry_times_seconds": np.asarray((0.016, 0.020))
    }
    contract = f25dc._pilot_contract(metrics)
    assert not contract["candidate_inventory"]["branch_labels_assigned"]
    assert contract["selected_candidates"]["primary"]["time_seconds"] == 0.020
    assert contract["selected_candidates"]["sealed"]["time_seconds"] == 0.016
    assert contract["next_definitions_only_manifest"]["work_package"] == f25dc.AUTHORIZED_NEXT
    assert contract["next_definitions_only_manifest"]["mathematical_root"][
        "no_artificial_82_channel_physical_reaction"
    ]
    assert not contract["authorization_boundaries"]["new_truth_authorized"]


def test_candidate_checks_fail_on_label_or_trust_leak() -> None:
    source = {
        "candidate_count": 6,
        "all_output_states_equal_accepted_states_bitwise": True,
    }
    geometry = {
        "physical_guard_passes": np.ones(6, dtype=bool),
        "minimum_reconstruction_factors": np.ones(6),
        "maximum_height_ratios": np.full(6, 0.1),
        "minimum_scattering_optical_depths": np.full(6, 10.0),
        "forward_patch_weights": np.zeros(6),
        "eligible_selection_matches_frozen_16ms_20ms_pair": True,
        "macro_path_effective_rank_at_relative_1e_3": 2,
        "primary_to_sealed_macro_separation": 0.1,
        "all_candidates_unclassified": True,
    }
    assert all(f25dc._checks(source, geometry).values())
    geometry["forward_patch_weights"][0] = 0.1
    assert not f25dc._checks(source, geometry)["forward_patch_not_mislabeled"]
    geometry["forward_patch_weights"][0] = 0.0
    geometry["all_candidates_unclassified"] = False
    assert not f25dc._checks(source, geometry)["all_unclassified"]


def test_canonical_preflight_if_present() -> None:
    if not f25dc.CANONICAL_DIRECTORY.exists():
        return
    f25dc._checksums(f25dc.CANONICAL_DIRECTORY)
    summary = f25dc._read(f25dc.CANONICAL_DIRECTORY / "summary.json")
    metrics = f25dc._read(
        f25dc.CANONICAL_DIRECTORY / "candidate_geometry_metrics.json"
    )
    contract = f25dc._read(
        f25dc.CANONICAL_DIRECTORY / "branch_pilot_contract.json"
    )
    assert summary["passed"]
    assert summary["classification"] == f25dc.CLASSIFICATION
    assert summary["authorized_next"] == f25dc.AUTHORIZED_NEXT
    assert summary["all_candidates_unclassified"]
    assert summary["primary_candidate"] == "U20_unclassified_primary"
    assert summary["sealed_candidate"] == "U16_unclassified_sealed"
    assert not summary["branch_root_execution_authorized"]
    assert all(metrics["checks"].values())
    geometry = metrics["geometry"]
    assert geometry["eligible_existing_atlas_geometry_times_seconds"] == [
        0.016,
        0.020,
    ]
    assert geometry["macro_path_effective_rank_at_relative_1e_3"] >= 2
    assert not contract["candidate_inventory"]["branch_labels_assigned"]
