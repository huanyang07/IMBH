#!/usr/bin/env python3
"""Decompose and freeze the cycle-wide inputs and global AP dry-run package."""

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
sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))

import run_causal_inner_complete_cycle_preexecution_manifest_wp10c9d6c7c3b5c4f25fizzn as parent  # noqa: E402


WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fizzo_cycle_wide_offline_atlas_"
    "boundary_event_acquisition_and_global_dry_run"
)
CLASSIFICATION = "cycle_wide_inputs_and_global_AP_dry_run_decomposition_frozen"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25fizzo1_production_size_global_AP_dry_run"
PASS_NEXT = (
    "WP10c9d6c7c3b5c4f25fizzp_legacy_cycle_evidence_"
    "compatibility_and_reusable_input_audit"
)
ARTIFACT = "causal_inner_cycle_wide_inputs_and_global_ap_dry_run_manifest_wp10c9d6c7c3b5c4f25fizzo"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = "docs/reports/current/CODEX_CAUSAL_INNER_CYCLE_WIDE_INPUTS_AND_GLOBAL_AP_DRY_RUN_MANIFEST_WP10C9D6C7C3B5C4F25FIZZO_2026-08-27.md"
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = "scripts/run_causal_inner_cycle_wide_inputs_and_global_ap_dry_run_manifest_wp10c9d6c7c3b5c4f25fizzo.py"
THIS_TEST = "tests/test_causal_inner_cycle_wide_inputs_and_global_ap_dry_run_manifest_wp10c9d6c7c3b5c4f25fizzo.py"
PARENT_SHA256 = "e7fe4eae3d8a6d951f25c429566d1bdef402adb49f23df4d01ef7eb9d533afcf"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _u():
    return parent._u()


def _validate_parent(require_clean=False):
    utility = _u()
    if utility._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != PARENT_SHA256:
        raise RuntimeError("complete-cycle preexecution checksum changed")
    hashes = utility._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utility._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    contract = utility._read_json(parent.CANONICAL_DIRECTORY / "preexecution_contract.json")
    if (
        not summary["passed"]
        or not summary["mathematical_architecture_selected"]
        or summary["cycle_wide_inputs_complete"]
        or summary["global_dry_run_complete"]
        or summary["complete_cycle_execution_ready"]
        or summary["complete_cycle_execution_authorized"]
        or summary["required_next_artifact"] != WORK_PACKAGE
        or contract["decision"]["required_next_artifact"] != WORK_PACKAGE
    ):
        raise RuntimeError("complete-cycle preexecution classification changed")
    if require_clean and utility._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("cycle-wide input decomposition needs a clean tracked tree")
    return hashes


def _contract():
    return {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "decomposition": [
            {
                "stage": 1,
                "name": "production_size_global_AP_dry_run",
                "purpose": "clear the global spatial exponential-action and cost blocker independently",
                "requires_new_truth": False,
            },
            {
                "stage": 2,
                "name": "legacy_cycle_evidence_compatibility_audit",
                "purpose": "classify old 82/470-dimensional boundary, event, forcing, and path artifacts for reuse",
                "requires_new_truth": False,
            },
            {
                "stage": 3,
                "name": "cycle_wide_eleven_field_atlas_and_event_acquisition",
                "purpose": "acquire only inputs not supplied by compatible legacy evidence",
                "requires_new_truth": True,
            },
            {
                "stage": 4,
                "name": "held_out_global_phase_window_validation",
                "purpose": "validate interpolation, boundaries, events, ledgers, restart, and step refinement",
                "requires_new_truth": True,
            },
        ],
        "global_AP_dry_run": {
            "radial_cells": 94,
            "fields_per_cell": 11,
            "global_state_dimension": 1034,
            "boundary": "periodic proof kernel only",
            "spatial_symbol": (
                "centered conservative derivative plus Rusanov entropy dissipation; "
                "each Fourier mode is an exact 11-field affine exponential action"
            ),
            "physical_anchor_paths": {"primary": [0, 10], "held_out": [20, 30]},
            "stiffness_ratios": [1.0, 1000.0],
            "step_counts": [4, 8, 16],
            "reference_step_count": 64,
            "normalized_horizon": 2.0,
            "gates": {
                "minimum_matched_refinement_order": 1.7,
                "maximum_homogeneous_mode_expansivity": 2.0e-10,
                "maximum_core_total_conservation_defect": 2.0e-11,
                "maximum_state_norm": 2.0,
                "minimum_source_nullity": 4,
                "checkpoint_and_suffix_replay": "bitwise",
                "maximum_projected_100k_step_wall_days": 3.0,
                "online_truth_calls": 0,
            },
        },
        "legacy_evidence_frozen_observations": {
            "old_online_state_dimension": 82,
            "old_complete_cycle_attempt_patches": 64,
            "old_exact_free_field_witnesses": 192,
            "old_physical_time_advanced_seconds": 0.016,
            "old_hot_exit_observed": False,
            "reuse_requires_explicit_eleven_field_lift_and_new_hash_lock": True,
        },
        "decision": {
            "global_dry_run_pass_authorized_next": PASS_NEXT,
            "complete_cycle_execution_authorized": False,
        },
        "claim_boundary": {
            "physical_production_boundaries_certified": False,
            "cycle_wide_atlas_complete": False,
            "complete_cycle_execution_authorized": False,
            "complete_cycle_steps": 0,
        },
        "authorized_next": AUTHORIZED_NEXT,
    }


def _update_catalog(summary):
    utility = _u(); rows = list(csv.DictReader(CANONICAL_MANIFEST.open(newline="", encoding="utf-8"))); rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file(): rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": utility._sha256(path), "scientific_status": "SUPPORTED"})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    catalog = utility._read_json(CANONICAL_SUMMARY); catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": CLASSIFICATION, "passed": True}; catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": utility._git("rev-parse", "HEAD"), "latest_work_package": WORK_PACKAGE}); utility._write_json(CANONICAL_SUMMARY, catalog)


def _freeze():
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists(): raise RuntimeError("cycle-wide input decomposition exists")
    hashes = _validate_parent(require_clean=True); utility = _u(); contract = _contract(); CANONICAL_DIRECTORY.mkdir(parents=True)
    utility._write_json(CANONICAL_DIRECTORY / "decomposition_contract.json", contract)
    summary = {"schema_version": 1, "work_package": WORK_PACKAGE, "classification": CLASSIFICATION, "passed": True, "definitions_only": True, "global_AP_dry_run_certified": False, "legacy_evidence_compatibility_audited": False, "cycle_wide_inputs_complete": False, "complete_cycle_execution_authorized": False, "complete_cycle_steps": 0, "authorized_next": AUTHORIZED_NEXT}
    utility._write_json(CANONICAL_DIRECTORY / "summary.json", summary); utility._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"parent_artifact": parent.ARTIFACT, "parent_checksum_manifest_sha256": PARENT_SHA256, "parent_hashes": hashes})
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True); REPORT_PATH.write_text("# Cycle-wide inputs and global AP dry-run decomposition\n\nThe pre-execution blocker is split into four prospective stages so that the production-size global AP cost can be decided without manufacturing missing cycle physics. The first stage is a 1,034-state periodic proof kernel using the same primary and held-out physical ports.\n\nThe previous 82-dimensional cycle attempt is preserved as legacy evidence: it used 64 patches and 192 exact witnesses, advanced only 0.016 physical seconds, and did not observe hot exit. Nothing from that model enters the eleven-field atlas without an explicit lift and a new audit. No complete-cycle execution is authorized.\n", encoding="utf-8")
    sources = (THIS_RUNNER, THIS_TEST, REPORT_RELATIVE); utility._write_json(CANONICAL_DIRECTORY / "provenance.json", {"implementation_commit": utility._git("rev-parse", "HEAD"), "source_hashes": {source: utility._sha256(ROOT / source) for source in sources}}); names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()); (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utility._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8"); _update_catalog(summary); return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--freeze", action="store_true"); arguments = parser.parse_args()
    if not arguments.freeze: parser.error("choose --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
