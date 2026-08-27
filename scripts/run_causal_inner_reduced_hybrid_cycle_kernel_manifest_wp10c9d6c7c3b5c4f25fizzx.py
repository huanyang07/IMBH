#!/usr/bin/env python3
"""Freeze the production reduced-hybrid cycle kernel adapter and cost gates."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path: sys.path.insert(0, str(path))

import run_causal_inner_heldout_atlas_and_hybrid_sequence_validator_wp10c9d6c7c3b5c4f25fizzw1 as parent  # noqa: E402


WORK_PACKAGE = parent.AUTHORIZED_NEXT
CLASSIFICATION = "reduced_hybrid_cycle_kernel_manifest_frozen"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25fizzx1_reduced_hybrid_cycle_kernel_certificate"
PASS_NEXT = "definitions_only_WP10c9d6c7c3b5c4f25fizzy_complete_cycle_preexecution_manifest"
ARTIFACT = "causal_inner_reduced_hybrid_cycle_kernel_manifest_wp10c9d6c7c3b5c4f25fizzx"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = "docs/reports/current/CODEX_CAUSAL_INNER_REDUCED_HYBRID_CYCLE_KERNEL_MANIFEST_WP10C9D6C7C3B5C4F25FIZZX_2026-08-27.md"
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = "scripts/run_causal_inner_reduced_hybrid_cycle_kernel_manifest_wp10c9d6c7c3b5c4f25fizzx.py"
THIS_TEST = "tests/test_causal_inner_reduced_hybrid_cycle_kernel_manifest_wp10c9d6c7c3b5c4f25fizzx.py"
PARENT_SHA256 = "2935dae93d0df83a9b7e5d27c5408c44c7f97993e8f4ea03f4138c7a4d18a582"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _u(): return parent._u()


def _validate_parent(*, require_clean=False):
    utility = _u()
    if utility._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != PARENT_SHA256: raise RuntimeError("heldout sequence validator changed")
    hashes = utility._validate_checksums(parent.CANONICAL_DIRECTORY); summary = utility._read_json(parent.CANONICAL_DIRECTORY / "summary.json"); metrics = utility._read_json(parent.CANONICAL_DIRECTORY / "validator_metrics.json")
    if not summary["passed"] or not summary["heldout_validator_structure_certified"] or not summary["hybrid_sequence_validator_structure_certified"] or not summary["finite_event_phase_advance_certified"] or not summary["synthetic_fixture_only"] or summary["physical_payloads_acquired"] or summary["authorized_next"] != WORK_PACKAGE or summary["complete_cycle_execution_authorized"] or metrics["complete_cycle_steps"] != 0: raise RuntimeError("heldout sequence validator classification changed")
    if require_clean and utility._git("status", "--short", "--untracked-files=no"): raise RuntimeError("reduced cycle kernel manifest needs a clean tracked tree")
    return hashes


def _contract():
    return {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "production_adapter": {
            "inputs": ["validated cycle physical input bundle v2", "hash-locked driver/branch/event simplices", "certified conservation map and minimum-norm normal", "validated heldout decision"],
            "rhs": "driver interpolation returns (qdot,phidot) for every Dormand-Prince stage",
            "branch_endpoint": "accepted endpoint reconstructs z_star,A,S,b and repeats hull/trust/gap/source/boundary audits",
            "event_guard": "all outgoing class-pure guard sheets evaluated in reduced coordinates",
            "event_reset": "interpolated impulse, duration, phase advance and destination mode become one atomic accepted transition",
            "history": "accepted checkpoints persist y,mode,next step,cumulative smooth/event ledgers and provenance",
            "online_truth_calls": 0,
            "online_large_nonlinear_roots": 0,
        },
        "production_fail_closed": {
            "require_physical_model_complete": True,
            "reject_synthetic_fixture": True,
            "require_all_physical_payload_hashes": True,
            "require_heldout_physical_validation_complete": True,
            "require_one_independent_spatial_holdout": True,
            "require_one_independent_full_sequence_or_cycle_holdout": True,
            "outside_hull_trust_or_guard_sheet": "reject step",
            "ambiguous_event": "reject step",
            "cycle_runner_available_in_certificate": False,
        },
        "prefix_and_cost_certificate": {
            "synthetic_fixture_only": True,
            "minimum_prefix_accepted_steps": 16,
            "minimum_prefix_events": 2,
            "minimum_endpoint_structure_audits": 16,
            "maximum_prefix_ledger_relative_defect": 2.0e-12,
            "restart_suffix_replay_bitwise": True,
            "minimum_benchmark_queries": 1000,
            "maximum_projected_100000_step_wall_days": 3.0,
            "assumed_rhs_queries_per_step": 7,
            "assumed_endpoint_branch_queries_per_step": 1,
            "complete_cycle_steps": 0,
        },
        "cost_projection": {
            "fiducial_period_seconds": 578880.0,
            "maximum_online_macrosteps": 100000,
            "required_mean_physical_seconds_per_step": 5.7888,
            "maximum_wall_days": 3.0,
            "report": ["driver RHS query wall time", "branch endpoint query wall time", "guard/reset query wall time", "checkpoint I/O", "projected cycle wall time", "projected physical seconds per wall second"],
        },
        "decision": {"pass_classification": "reduced_hybrid_cycle_kernel_structure_and_cost_certified_synthetic_fixture_only", "failure_classification": "reduced_hybrid_cycle_kernel_or_cost_failed", "pass_authorized_next": PASS_NEXT},
        "scientific_boundary": {"physical_model_complete": False, "physical_payloads_acquired": False, "heldout_physical_validation_complete": False, "complete_cycle_execution_authorized": False, "complete_cycle_steps": 0},
        "authorized_next": AUTHORIZED_NEXT,
    }


def _update(summary):
    utility = _u(); rows = list(csv.DictReader(CANONICAL_MANIFEST.open(newline="", encoding="utf-8"))); rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file(): rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": utility._sha256(path), "scientific_status": "SUPPORTED"})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle: writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    catalog = utility._read_json(CANONICAL_SUMMARY); catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": CLASSIFICATION, "passed": True}; catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": utility._git("rev-parse", "HEAD"), "latest_work_package": WORK_PACKAGE}); utility._write_json(CANONICAL_SUMMARY, catalog)


def _freeze():
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists(): raise RuntimeError("reduced cycle kernel manifest already exists")
    hashes = _validate_parent(require_clean=True); utility = _u(); CANONICAL_DIRECTORY.mkdir(parents=True); utility._write_json(CANONICAL_DIRECTORY / "cycle_kernel_contract.json", _contract())
    summary = {"schema_version": 1, "work_package": WORK_PACKAGE, "classification": CLASSIFICATION, "passed": True, "definitions_only": True, "reduced_hybrid_cycle_kernel_certified": False, "production_adapter_certified": False, "synthetic_fixture_only": True, "physical_model_complete": False, "physical_payloads_acquired": False, "heldout_physical_validation_complete": False, "complete_cycle_execution_authorized": False, "complete_cycle_steps": 0, "authorized_next": AUTHORIZED_NEXT}; utility._write_json(CANONICAL_DIRECTORY / "summary.json", summary); utility._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"parent_artifact": parent.ARTIFACT, "parent_checksum_manifest_sha256": PARENT_SHA256, "parent_hashes": hashes})
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True); REPORT_PATH.write_text("# Reduced hybrid cycle-kernel manifest\n\n" f"Classification: `{CLASSIFICATION}`.\n\n" "The prospective production adapter binds the validated physical driver, structure-preserving branch atlas, oriented event sheets, conservative finite-duration resets, and reduced Dormand-Prince integrator. Every accepted endpoint repeats hull, trust, spectral-gap, source-nullity, and 0/11 boundary audits.\n\n" "Only a synthetic prefix and online cost benchmark are authorized. A production bundle must be complete, nonsynthetic, hash-closed, and pass all physical holdouts before a separate complete-cycle manifest can authorize execution. No cycle runner or cycle step is permitted here.\n", encoding="utf-8")
    sources = (THIS_RUNNER, THIS_TEST, REPORT_RELATIVE); utility._write_json(CANONICAL_DIRECTORY / "provenance.json", {"implementation_commit": utility._git("rev-parse", "HEAD"), "source_hashes": {name: utility._sha256(ROOT / name) for name in sources}}); names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()); (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utility._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8"); _update(summary); return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--freeze", action="store_true"); args = parser.parse_args()
    if not args.freeze: parser.error("choose --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
