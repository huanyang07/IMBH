#!/usr/bin/env python3
"""Freeze heldout atlas validation and reduced hybrid sequence tests."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path: sys.path.insert(0, str(path))

import run_causal_inner_cycle_physical_driver_branch_and_event_interpolator_structure_certificate_wp10c9d6c7c3b5c4f25fizzv1 as parent  # noqa: E402


WORK_PACKAGE = parent.AUTHORIZED_NEXT
CLASSIFICATION = "heldout_atlas_and_hybrid_sequence_validation_manifest_frozen"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25fizzw1_heldout_atlas_and_hybrid_sequence_validator"
PASS_NEXT = "definitions_only_WP10c9d6c7c3b5c4f25fizzx_reduced_hybrid_cycle_kernel_manifest"
ARTIFACT = "causal_inner_heldout_atlas_and_hybrid_sequence_validation_manifest_wp10c9d6c7c3b5c4f25fizzw"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = "docs/reports/current/CODEX_CAUSAL_INNER_HELDOUT_ATLAS_AND_HYBRID_SEQUENCE_VALIDATION_MANIFEST_WP10C9D6C7C3B5C4F25FIZZW_2026-08-27.md"
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = "scripts/run_causal_inner_heldout_atlas_and_hybrid_sequence_validation_manifest_wp10c9d6c7c3b5c4f25fizzw.py"
THIS_TEST = "tests/test_causal_inner_heldout_atlas_and_hybrid_sequence_validation_manifest_wp10c9d6c7c3b5c4f25fizzw.py"
PARENT_SHA256 = "c7cca633d97be79be7a5652b35b84dda5383d81f3e60986ac1382bdcee85bacf"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _u(): return parent._u()


def _validate_parent(*, require_clean: bool = False):
    utility = _u()
    if utility._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != PARENT_SHA256: raise RuntimeError("cycle interpolator certificate changed")
    hashes = utility._validate_checksums(parent.CANONICAL_DIRECTORY); summary = utility._read_json(parent.CANONICAL_DIRECTORY / "summary.json"); metrics = utility._read_json(parent.CANONICAL_DIRECTORY / "interpolator_metrics.json")
    if not summary["passed"] or not summary["cycle_interpolator_structure_certified"] or not summary["event_guard_sheet_dimension_corrected"] or not summary["synthetic_fixture_only"] or summary["physical_payloads_acquired"] or summary["authorized_next"] != WORK_PACKAGE or summary["complete_cycle_execution_authorized"] or metrics["complete_cycle_steps"] != 0: raise RuntimeError("cycle interpolator classification changed")
    if require_clean and utility._git("status", "--short", "--untracked-files=no"): raise RuntimeError("heldout sequence manifest needs a clean tracked tree")
    return hashes


def _contract():
    return {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "event_time_schema_extension": {
            "reason": "a finite-duration compressed event must advance elapsed time and orbital phase",
            "integrated_phase_advance": "Delta phi_event>0 in the same unwrapped phase convention as the driver",
            "post_event_phase": "phi_plus=phi_entry+Delta phi_event modulo 2*pi",
            "consistency": "Delta phi_event equals the physical event-truth phase integral over duration within 2e-3 relative",
            "reset_order": ["localize entry guard", "apply integrated q ledger impulse", "advance elapsed time by duration", "advance unwrapped phase", "switch mode", "require destination guard margin"],
            "zero_duration_instantaneous_event": "permitted only when physical truth declares it and phase advance is exactly zero",
        },
        "smooth_reduced_flow": {
            "state": "y=(q1,q2,q3,q4,phi_unwrapped)",
            "mode": "held fixed between events",
            "rhs": "qdot=distributed_ledger_rate+boundary_ledger_rate; phidot=positive phase rate",
            "method": "Dormand-Prince 5(4) with embedded error and continuous dense output",
            "same_quadrature_ledger": "accepted q increment is exactly the RK quadrature of the interpolated physical ledger",
            "step_acceptance": ["embedded scaled error <=1", "all RK stages inside driver hull", "accepted endpoint inside branch hull and trust", "positive fast gap", "all physical/entropy guards"],
            "rejection": "rejected stages alter no history, event state, or cumulative ledger",
            "maximum_step_ratio": 2.0,
            "maximum_accepted_macrosteps": 100000,
        },
        "event_localization": {
            "outgoing_classes": "all transition classes whose source mode equals the current mode",
            "detection": "oriented endpoint sign change with the declared crossing direction",
            "multiple_crossings": "reject if more than one unresolved class or more than one crossing in a step",
            "localization": "bracketed dense-output root with fresh guard-sheet membership checks",
            "maximum_guard_value": 1.0e-10,
            "maximum_event_time_fraction_of_step": 1.0e-8,
            "minimum_transversality": 1.0e-8,
            "post_reset_dwell": "destination mode must have strictly positive guard margin before smooth stepping resumes",
        },
        "prospective_holdouts": {
            "leakage": "all training/heldout indices and simplex connectivity hash-locked before interpolation",
            "branch": "at least 20 percent per mode, including a hull-edge and a gap-minimum point; never simplex vertices",
            "phase": "at least two contiguous driver windows absent from all fits",
            "events": "at least 20 percent per transition class and one parameter-edge event; never guard-sheet vertices",
            "sequence": "one multi-event physical sequence absent from branch, driver, guard and reset fitting",
            "spatial": "one independently produced refined-grid endpoint comparison",
        },
        "heldout_gates": {
            "maximum_branch_state_relative_defect": 2.0e-2,
            "maximum_branch_rate_relative_defect": 5.0e-2,
            "maximum_physical_port_action_relative_defect": 5.0e-2,
            "maximum_event_time_relative_defect": 2.0e-2,
            "maximum_event_post_state_relative_defect": 5.0e-2,
            "maximum_event_ledger_relative_defect": 2.0e-2,
            "maximum_sequence_endpoint_relative_defect": 5.0e-2,
            "maximum_sequence_ledger_relative_defect": 2.0e-2,
            "all_discrete_modes_and_event_order_exact": True,
            "all_hull_trust_gap_boundary_and_physical_gates": True,
            "restart_suffix_replay_bitwise": True,
        },
        "structure_certificate": {
            "synthetic_fixture_only": True,
            "minimum_smooth_accepted_steps": 8,
            "minimum_events": 2,
            "minimum_modes": 2,
            "matched_refinement_step_counts": [16, 32, 64],
            "minimum_smooth_observed_order": 4.5,
            "maximum_reduced_ledger_defect": 2.0e-12,
            "event_time_and_reset_exact_fixture": True,
            "restart_suffix_replay_bitwise": True,
            "physical_claim": False,
            "complete_cycle_steps": 0,
        },
        "scientific_boundary": {"physical_payloads_acquired": False, "heldout_physical_validation_complete": False, "complete_cycle_execution_authorized": False, "complete_cycle_steps": 0},
        "decision": {"pass_classification": "heldout_atlas_and_hybrid_sequence_validator_certified_synthetic_fixture_only", "failure_classification": "heldout_atlas_or_hybrid_sequence_validator_failed", "pass_authorized_next": PASS_NEXT},
        "authorized_next": AUTHORIZED_NEXT,
    }


def _update(summary):
    utility = _u(); rows = list(csv.DictReader(CANONICAL_MANIFEST.open(newline="", encoding="utf-8"))); rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file(): rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": utility._sha256(path), "scientific_status": "SUPPORTED"})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle: writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    catalog = utility._read_json(CANONICAL_SUMMARY); catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": CLASSIFICATION, "passed": True}; catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": utility._git("rev-parse", "HEAD"), "latest_work_package": WORK_PACKAGE}); utility._write_json(CANONICAL_SUMMARY, catalog)


def _freeze():
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists(): raise RuntimeError("heldout sequence manifest already exists")
    hashes = _validate_parent(require_clean=True); utility = _u(); CANONICAL_DIRECTORY.mkdir(parents=True); utility._write_json(CANONICAL_DIRECTORY / "heldout_and_sequence_contract.json", _contract())
    summary = {"schema_version": 1, "work_package": WORK_PACKAGE, "classification": CLASSIFICATION, "passed": True, "definitions_only": True, "finite_event_phase_advance_frozen": True, "heldout_validator_certified": False, "hybrid_sequence_validator_certified": False, "synthetic_fixture_only": True, "physical_payloads_acquired": False, "heldout_physical_validation_complete": False, "complete_cycle_execution_authorized": False, "complete_cycle_steps": 0, "authorized_next": AUTHORIZED_NEXT}; utility._write_json(CANONICAL_DIRECTORY / "summary.json", summary); utility._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"parent_artifact": parent.ARTIFACT, "parent_checksum_manifest_sha256": PARENT_SHA256, "parent_hashes": hashes})
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True); REPORT_PATH.write_text("# Heldout atlas and hybrid-sequence validation manifest\n\n" f"Classification: `{CLASSIFICATION}`.\n\n" "The prospective reduced flow evolves the four retained physical ledgers and unwrapped orbital phase with Dormand-Prince 5(4), using the same quadrature for state and ledger accounting. Events are localized on oriented guard sheets, then advance q, elapsed time, phase, and mode using a finite-duration compressed event map.\n\n" "The manifest also hash-locks branch, phase-window, event, spatial, and full-sequence holdouts before fitting. It is definitions-only, has no physical payload, and authorizes no complete-cycle step.\n", encoding="utf-8")
    sources = (THIS_RUNNER, THIS_TEST, REPORT_RELATIVE); utility._write_json(CANONICAL_DIRECTORY / "provenance.json", {"implementation_commit": utility._git("rev-parse", "HEAD"), "source_hashes": {name: utility._sha256(ROOT / name) for name in sources}}); names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()); (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utility._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8"); _update(summary); return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--freeze", action="store_true"); args = parser.parse_args()
    if not args.freeze: parser.error("choose --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
