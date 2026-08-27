import json

import run_causal_inner_complete_cycle_preexecution_readiness_certificate_wp10c9d6c7c3b5c4f25fizzy1 as runner


def test_parent_and_support_evidence_are_hash_locked():
    assert "summary.json" in runner._validate_parent()
    support = runner._validate_support()
    assert support["legacy"]["direct_binding_reuse_count"] == 0
    assert support["ports"]["candidate_anchor_count"] == 913
    assert support["native_global_AP"]["global_state_dimension"] == 1232
    assert support["schema"]["synthetic_fixture_rejected_when_physical_required"]


def test_repository_inventory_finds_no_complete_physical_cycle_bundle():
    inventory = runner._inventory()
    assert not inventory["unreadable_tracked_npz"]
    assert inventory["complete_bundle_directory_count"] == 0
    assert inventory["physical_metadata_record_count"] == 0
    assert inventory["complete_group_file_count"] == 0
    assert inventory["directly_reusable_binding_cycle_field_count"] == 0
    assert all(not paths for paths in inventory["complete_group_files"].values())


def test_readiness_fails_closed_without_relabeling_prefix_or_synthetic_data():
    support = runner._validate_support()
    readiness = runner._readiness(support, runner._inventory())
    assert readiness["inventory_audit_passed"]
    assert not readiness["complete_cycle_preexecution_readiness_passed"]
    assert not readiness["binding_interpretation"]["physical_model_failure_selected"]
    assert readiness["binding_interpretation"]["external_physical_acquisition_block_selected"]
    assert not readiness["binding_interpretation"]["legacy_prefix_may_be_used_as_binding_cycle_truth"]
    assert not readiness["binding_interpretation"]["synthetic_fixture_may_be_relabeled_physical"]
    assert all(readiness["genuinely_missing_external_evidence"].values())
    assert not readiness["decision"]["complete_cycle_execution_authorized"]
    assert readiness["decision"]["complete_cycle_steps"] == 0


def test_canonical_readiness_certificate_records_supported_negative_finding():
    if not runner.CANONICAL_DIRECTORY.exists():
        return
    assert runner._u()._validate_checksums(runner.CANONICAL_DIRECTORY)
    summary = json.loads((runner.CANONICAL_DIRECTORY / "summary.json").read_text())
    assert summary["passed"] and summary["inventory_audit_passed"]
    assert summary["directly_reusable_binding_cycle_field_count"] == 0
    assert not summary["complete_cycle_preexecution_readiness_passed"]
    assert not summary["complete_cycle_execution_authorized"]
    assert summary["complete_cycle_steps"] == 0
