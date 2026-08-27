#!/usr/bin/env python3
"""Freeze the legacy cycle-evidence compatibility audit."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_production_size_global_ap_dry_run_wp10c9d6c7c3b5c4f25fizzo1 as parent  # noqa: E402


WORK_PACKAGE = parent.AUTHORIZED_NEXT
CLASSIFICATION = "legacy_cycle_evidence_compatibility_audit_frozen"
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizzp1_legacy_cycle_evidence_"
    "compatibility_and_reusable_input_audit"
)
PASS_NEXT = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizzq_"
    "cycle_wide_eleven_field_anchor_coverage_and_lift_manifest"
)
ARTIFACT = (
    "causal_inner_legacy_cycle_evidence_compatibility_manifest_"
    "wp10c9d6c7c3b5c4f25fizzp"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_LEGACY_CYCLE_EVIDENCE_"
    "COMPATIBILITY_MANIFEST_WP10C9D6C7C3B5C4F25FIZZP_2026-08-27.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_legacy_cycle_evidence_compatibility_"
    "manifest_wp10c9d6c7c3b5c4f25fizzp.py"
)
THIS_TEST = (
    "tests/test_causal_inner_legacy_cycle_evidence_compatibility_"
    "manifest_wp10c9d6c7c3b5c4f25fizzp.py"
)
PARENT_SHA256 = "6b6edd61bcccbb0b11c60848dbd14874f278d196b86ad70d28ef5c56ffcb7884"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

LEGACY_INPUTS = {
    "complete_cycle_attempt": {
        "artifact": "causal_inner_adaptive_complete_cycle_execution_wp10c9d6c7c3b5c4f25fe",
        "checksum_manifest_sha256": "0b018b004798f28e5ec5d5f0e70b2bfb26ee6fca0b05a42f53f810246a061aab",
        "expected_classification": "complete_cycle_inconclusive_acquisition_budget_exhausted",
    },
    "five_field_fixed_exterior_boundary": {
        "artifact": "causal_inner_fixed_exterior_continuum_reference_wp10c9d6c7c2c2",
        "checksum_manifest_sha256": "c1ba4e2a181d78931139144c955d077b09bb4ab2846c5484270bb13ebffac9f7",
        "expected_classification": "fixed_exterior_continuum_reference_certified_embedded_propagation_authorized",
    },
    "transition_geometry": {
        "artifact": "causal_inner_transition_tube_geometry_wp10c9d6c7c3b5c4f25ds",
        "checksum_manifest_sha256": "b056fb78d8c66e7b3cc0238ef9c37b4250de5085c72ef8baf770330faac4cf33",
        "expected_classification": "one_scalar_conservative_transition_tube_geometry_supported_hot_exit_unobserved",
    },
    "conservative_transition_tube": {
        "artifact": "causal_inner_conservative_transition_tube_wp10c9d6c7c3b5c4f25du",
        "checksum_manifest_sha256": "4447e9915b9ca5e1fb5259bf74717cbae55b694345a77fb1612397ae3460484a",
        "expected_classification": "train_only_rank_adaptive_conservative_scalar_transition_tube_validated_local_observed_segment",
    },
    "terminal_hot_exit_recovery": {
        "artifact": "causal_inner_hot_exit_half_step_recovery_wp10c9d6c7c3b5c4f25dq_step_12",
        "checksum_manifest_sha256": "393394e853123640e3ea22d47b92db1df85e9bd36eb73290d09861f808364a25",
        "expected_classification": "half_step_hot_exit_recovery_budget_exhausted_exit_not_reached",
    },
    "branch_first_impulse_architecture": {
        "artifact": "causal_inner_branch_first_hybrid_impulse_architecture_wp10c9d6c7c3b5c4f25dl",
        "checksum_manifest_sha256": "4ea1a038e09aa42afb4e1c59d9e659ca2e0f3c360bbe28c22343a4e1e16eaea1",
        "expected_classification": "rank16_transition_internal_candidate_reconciled_branch_first_hybrid_impulse_sampling_architecture_frozen",
    },
    "legacy_cycle_map": {
        "artifact": "causal_inner_cycle_map_architecture_decision_wp10c9d6c7c3b5c4f25ec_v2",
        "checksum_manifest_sha256": "383a04434bffc62a72c098f1d570a1d297379437342e8bc1c1bc1686ae0725f7",
        "expected_classification": "conservative_hybrid_phase_cycle_map_architecture_selected_accepted_anchor_three_mode_prefix_replayed_complete_cycle_calibration_missing",
    },
}


def _u():
    return parent._u()


def _legacy_directory(specification: dict) -> Path:
    return ROOT / "results/canonical" / specification["artifact"]


def _validate_parent(*, require_clean: bool = False):
    utility = _u()
    if utility._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != PARENT_SHA256:
        raise RuntimeError("production-size global AP certificate changed")
    parent_hashes = utility._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utility._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    if (
        not summary["passed"]
        or not summary["production_size_global_AP_dry_run_certified"]
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["complete_cycle_execution_authorized"]
    ):
        raise RuntimeError("global AP dry-run classification changed")
    legacy_hashes = {}
    for label, specification in LEGACY_INPUTS.items():
        directory = _legacy_directory(specification)
        if utility._sha256(directory / "SHA256SUMS.txt") != specification["checksum_manifest_sha256"]:
            raise RuntimeError(f"legacy checksum manifest changed: {label}")
        hashes = utility._validate_checksums(directory)
        legacy_summary = utility._read_json(directory / "summary.json")
        if legacy_summary["classification"] != specification["expected_classification"]:
            raise RuntimeError(f"legacy classification changed: {label}")
        legacy_hashes[label] = hashes
    if require_clean and utility._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("legacy compatibility manifest needs a clean tracked tree")
    return parent_hashes, legacy_hashes


def _contract():
    return {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "frozen_inputs": LEGACY_INPUTS,
        "target_architecture": {
            "radial_cells": 94,
            "fields_per_cell": 11,
            "global_state_dimension": 1034,
            "cycle_seconds": 578880.0,
        },
        "compatibility_classes": [
            "direct_binding_reuse",
            "candidate_seed_after_deterministic_lift_and_new_audit",
            "diagnostic_only",
            "binding_negative_evidence",
        ],
        "binding_checks": {
            "all_input_checksum_manifests_close": True,
            "legacy_profile_shape": [112, 5],
            "legacy_coordinate_dimension": 470,
            "legacy_macro_dimension": 82,
            "old_boundary_field_count": 5,
            "target_field_count": 11,
            "old_cycle_observed": False,
            "old_hot_exit_observed": False,
            "direct_binding_reuse_count": 0,
            "complete_cycle_execution_authorized": False,
        },
        "decision": {
            "pass_classification": (
                "legacy_cycle_evidence_partitioned_no_direct_"
                "eleven_field_cycle_input"
            ),
            "pass_authorized_next": PASS_NEXT,
            "failure_classification": "legacy_cycle_evidence_compatibility_audit_failed",
        },
        "claim_boundary": {
            "new_truth_calls": 0,
            "new_trajectory_steps": 0,
            "cycle_wide_inputs_complete": False,
            "complete_cycle_execution_authorized": False,
        },
        "authorized_next": AUTHORIZED_NEXT,
    }


def _update_catalog(summary):
    utility = _u()
    rows = list(csv.DictReader(CANONICAL_MANIFEST.open(newline="", encoding="utf-8")))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": utility._sha256(path), "scientific_status": "SUPPORTED"})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    catalog = utility._read_json(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": summary["classification"], "passed": True}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": utility._git("rev-parse", "HEAD"), "latest_work_package": WORK_PACKAGE})
    utility._write_json(CANONICAL_SUMMARY, catalog)


def _freeze():
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("legacy compatibility manifest already exists")
    parent_hashes, legacy_hashes = _validate_parent(require_clean=True)
    utility = _u(); CANONICAL_DIRECTORY.mkdir(parents=True)
    utility._write_json(CANONICAL_DIRECTORY / "compatibility_contract.json", _contract())
    summary = {"schema_version": 1, "work_package": WORK_PACKAGE, "classification": CLASSIFICATION, "passed": True, "definitions_only": True, "legacy_input_count": len(LEGACY_INPUTS), "legacy_evidence_compatibility_audited": False, "cycle_wide_inputs_complete": False, "complete_cycle_execution_authorized": False, "complete_cycle_steps": 0, "authorized_next": AUTHORIZED_NEXT}
    utility._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utility._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"parent_artifact": parent.ARTIFACT, "parent_checksum_manifest_sha256": PARENT_SHA256, "parent_hashes": parent_hashes, "legacy_hashes": legacy_hashes})
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("# Legacy cycle-evidence compatibility manifest\n\nSeven checksum-locked legacy packages are audited without new truth calls. The audit separates candidate five-field physical seeds and geometry from diagnostic-only reduced models and binding negative event evidence. No legacy payload is presumed to be a direct eleven-field production input. No cycle step is authorized.\n", encoding="utf-8")
    sources = (THIS_RUNNER, THIS_TEST, REPORT_RELATIVE)
    utility._write_json(CANONICAL_DIRECTORY / "provenance.json", {"implementation_commit": utility._git("rev-parse", "HEAD"), "source_hashes": {source: utility._sha256(ROOT / source) for source in sources}})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utility._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8")
    _update_catalog(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--freeze", action="store_true")
    arguments = parser.parse_args()
    if not arguments.freeze: parser.error("choose --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
