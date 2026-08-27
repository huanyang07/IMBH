import json

import run_causal_inner_physical_cycle_bundle_acquisition_manifest_wp10c9d6c7c3b5c4f25fizzy2 as runner


def test_parent_negative_readiness_certificate_is_hash_locked():
    hashes, summary, readiness, inventory = runner._validate_parent()
    assert "summary.json" in hashes
    assert summary["external_physical_acquisition_block_selected"]
    assert not readiness["complete_cycle_preexecution_readiness_passed"]
    assert inventory["directly_reusable_binding_cycle_field_count"] == 0


def test_v2_contract_matches_production_reduced_hybrid_kernel_inputs():
    _, _, readiness, _ = runner._validate_parent()
    contract = runner._contract(readiness)
    bundle = contract["canonical_physical_cycle_bundle_v2"]
    assert contract["schema_version"] == 2
    assert "atlas_topology.npz" in bundle
    assert "integrated_phase_advance" in bundle["events_train.npz"]
    assert "destination_guard_margin" in bundle["events_train.npz"]
    assert "reduced_guard_normals5" in bundle["events_train.npz"]
    assert "branch_holdout.npz" in bundle
    assert "event_holdout.npz" in bundle
    assert "sequence_holdout.npz" in bundle
    assert "spatial_holdout.npz" in bundle
    assert "production_benchmark.json" in bundle


def test_acquisition_is_prospective_fail_fast_and_external():
    _, _, readiness, _ = runner._validate_parent()
    contract = runner._contract(readiness)
    stages = contract["prospective_acquisition_sequence"]
    assert [stage["stage"] for stage in stages] == list(range(10))
    assert stages[0]["name"] == "external model declaration"
    assert stages[1]["name"] == "prospective split lock"
    assert contract["scientific_authority_boundary"]["external_scientific_authority_required"]
    assert not contract["scientific_authority_boundary"][
        "repository_may_select_unprovided_physical_forcing_or_modes"
    ]
    assert not contract["current_status"]["complete_cycle_execution_authorized"]
    assert contract["cost_and_execution_boundary"]["new_complete_cycle_steps"] == 0


def test_external_request_has_no_synthetic_or_legacy_defaults():
    request = runner._request_template()
    assert request["delivery_status"] == "awaiting_external_scientific_input"
    assert not request["repository_defaults_permitted"]
    assert not request["synthetic_substitution_permitted"]
    assert not request["legacy_prefix_extrapolation_permitted"]
    assert not request["complete_cycle_execution_authorized"]
    assert all(
        value is None or value == []
        for value in request["required_before_repository_execution"].values()
    )


def test_canonical_acquisition_manifest_remains_definitions_only():
    if not runner.CANONICAL_DIRECTORY.exists():
        return
    assert runner._u()._validate_checksums(runner.CANONICAL_DIRECTORY)
    summary = json.loads((runner.CANONICAL_DIRECTORY / "summary.json").read_text())
    assert summary["passed"] and summary["definitions_only"]
    assert summary["external_scientific_input_required"]
    assert not summary["physical_model_declaration_received"]
    assert not summary["complete_cycle_preexecution_readiness_passed"]
    assert not summary["complete_cycle_execution_authorized"]
    assert summary["complete_cycle_steps"] == 0
