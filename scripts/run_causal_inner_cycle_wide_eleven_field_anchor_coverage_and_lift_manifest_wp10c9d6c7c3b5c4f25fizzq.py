#!/usr/bin/env python3
"""Freeze the native-grid eleven-field lift and cycle-coverage package."""

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

import run_causal_inner_legacy_cycle_evidence_compatibility_audit_wp10c9d6c7c3b5c4f25fizzp1 as parent  # noqa: E402


WORK_PACKAGE = parent.AUTHORIZED_NEXT
CLASSIFICATION = "native_112_cell_eleven_field_lift_and_coverage_manifest_frozen"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25fizzq1_physical_112_cell_global_AP_dry_run"
PASS_NEXT = "WP10c9d6c7c3b5c4f25fizzq2_native_grid_five_to_eleven_field_lift_and_prefix_coverage_audit"
ARTIFACT = "causal_inner_cycle_wide_eleven_field_anchor_coverage_and_lift_manifest_wp10c9d6c7c3b5c4f25fizzq"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = "docs/reports/current/CODEX_CAUSAL_INNER_CYCLE_WIDE_ELEVEN_FIELD_ANCHOR_COVERAGE_AND_LIFT_MANIFEST_WP10C9D6C7C3B5C4F25FIZZQ_2026-08-27.md"
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = "scripts/run_causal_inner_cycle_wide_eleven_field_anchor_coverage_and_lift_manifest_wp10c9d6c7c3b5c4f25fizzq.py"
THIS_TEST = "tests/test_causal_inner_cycle_wide_eleven_field_anchor_coverage_and_lift_manifest_wp10c9d6c7c3b5c4f25fizzq.py"
PARENT_SHA256 = "d6b1b2c6afe76b56a407e9b2baa0a319b6ec6a111a24668938e253b8cb7e5ac3"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _u(): return parent._u()


def _validate_parent(require_clean=False):
    utility = _u()
    if utility._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != PARENT_SHA256: raise RuntimeError("legacy compatibility audit changed")
    hashes = utility._validate_checksums(parent.CANONICAL_DIRECTORY); summary = utility._read_json(parent.CANONICAL_DIRECTORY / "summary.json"); audit = utility._read_json(parent.CANONICAL_DIRECTORY / "compatibility_audit.json")
    if not summary["passed"] or not summary["legacy_evidence_compatibility_audited"] or summary["authorized_next"] != WORK_PACKAGE or audit["direct_binding_reuse_count"] != 0 or audit["facts"]["trajectory_primitive_shape"] != [65,112,5] or summary["complete_cycle_execution_authorized"]: raise RuntimeError("legacy compatibility classification changed")
    if require_clean and utility._git("status", "--short", "--untracked-files=no"): raise RuntimeError("native-grid lift manifest needs a clean tracked tree")
    return hashes, audit


def _contract():
    return {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "grid_correction": {
            "physical_context_cells": 112,
            "legacy_primitive_profile_shape": [112,5],
            "eleven_field_global_dimension": 1232,
            "previous_periodic_proof_cells": 94,
            "previous_periodic_proof_dimension": 1034,
            "previous_certificate_scope_preserved": "94-cell periodic scalability proof",
            "production_size_claim_superseded": True,
            "radial_remap_required_for_native_lift": False,
            "reason": "470 is an implicit constrained-coordinate dimension, not 94 physical radial cells",
        },
        "stages": [
            {"stage": 1, "name": "physical_112_cell_global_AP_dry_run", "new_truth_calls": 0},
            {"stage": 2, "name": "native_grid_five_to_eleven_field_lift_and_prefix_coverage", "new_truth_calls": 0},
            {"stage": 3, "name": "cycle_wide_missing_anchor_acquisition", "new_truth_required": True},
        ],
        "physical_grid_global_AP_dry_run": {
            "radial_cells": 112,
            "fields_per_cell": 11,
            "global_state_dimension": 1232,
            "boundary": "periodic scalability proof only",
            "physical_anchor_paths": {"primary": [0,10], "held_out": [20,30]},
            "stiffness_ratios": [1.0,1000.0],
            "step_counts": [4,8,16],
            "reference_step_count": 64,
            "normalized_horizon": 2.0,
            "gates": {"minimum_matched_refinement_order": 1.7, "maximum_homogeneous_mode_expansivity": 2e-10, "maximum_core_total_conservation_defect": 2e-11, "maximum_state_norm": 2.0, "minimum_source_nullity": 4, "checkpoint_and_suffix_replay": "bitwise", "maximum_projected_100k_step_wall_days": 3.0, "online_truth_calls": 0},
        },
        "native_lift": {
            "old_fields": ["log_surface_density","radial_velocity_over_c","azimuthal_velocity_over_c","log_temperature","specific_Rphi_stress"],
            "new_fields": ["four equilibrium entropy-current coordinates","five physical STF shear amplitudes","log height over anchor","vertical velocity"],
            "shear_embedding": "old R-phi stress maps to sqrt(2) times the normalized STF R-phi amplitude",
            "height": "recover the accepted hydrostatic responsive height cellwise",
            "vertical_velocity": "zero only as an anchor coordinate; source and heldout audits remain binding",
            "native_cellwise_lift": True,
            "accepted_history_only": True,
        },
        "coverage": {
            "first_audit": "all saved 65 trajectory profiles and 192 exact witnesses",
            "binding_metric": "nearest certified anchor in the frozen nonlinear trust metric, not componentwise envelope alone",
            "full_cycle_requirement": "phase [0,2pi], every physical mode, every accepted q-tube, impact and hot-exit neighborhoods",
            "prefix_coverage_cannot_establish_cycle_coverage": True,
        },
        "decision": {"physical_grid_pass_authorized_next": PASS_NEXT, "complete_cycle_execution_authorized": False},
        "claim_boundary": {"cycle_wide_inputs_complete": False, "physical_boundaries_certified": False, "events_and_resets_certified": False, "complete_cycle_execution_authorized": False, "complete_cycle_steps": 0},
        "authorized_next": AUTHORIZED_NEXT,
    }


def _update(summary):
    utility=_u(); rows=list(csv.DictReader(CANONICAL_MANIFEST.open(newline="",encoding="utf-8"))); rows=[row for row in rows if row.get("case")!=ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file(): rows.append({"case":ARTIFACT,"path":str(path.relative_to(ROOT)),"bytes":str(path.stat().st_size),"sha256":utility._sha256(path),"scientific_status":"SUPPORTED"})
    with CANONICAL_MANIFEST.open("w",newline="",encoding="utf-8") as handle: writer=csv.DictWriter(handle,fieldnames=("case","path","bytes","sha256","scientific_status"),lineterminator="\n");writer.writeheader();writer.writerows(rows)
    catalog=utility._read_json(CANONICAL_SUMMARY);catalog.setdefault("artifacts",{})[ARTIFACT]={"path":str(CANONICAL_DIRECTORY.relative_to(ROOT)),"classification":summary["classification"],"passed":True};catalog.update({"case_count":len({row["case"] for row in rows}),"file_count":len(rows),"total_bytes":sum(int(row["bytes"]) for row in rows),"all_payload_hashes_recorded":True,"latest_source_parent_commit":utility._git("rev-parse","HEAD"),"latest_work_package":WORK_PACKAGE});utility._write_json(CANONICAL_SUMMARY,catalog)


def _freeze():
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists(): raise RuntimeError("native-grid lift manifest exists")
    hashes,_=_validate_parent(require_clean=True);utility=_u();CANONICAL_DIRECTORY.mkdir(parents=True);contract=_contract();utility._write_json(CANONICAL_DIRECTORY/"lift_and_coverage_contract.json",contract)
    summary={"schema_version":1,"work_package":WORK_PACKAGE,"classification":CLASSIFICATION,"passed":True,"definitions_only":True,"physical_grid_cells":112,"eleven_field_global_dimension":1232,"previous_94_cell_scalability_certificate_preserved":True,"previous_production_size_label_superseded":True,"physical_112_cell_global_AP_certified":False,"native_five_to_eleven_lift_certified":False,"cycle_wide_inputs_complete":False,"complete_cycle_execution_authorized":False,"complete_cycle_steps":0,"authorized_next":AUTHORIZED_NEXT};utility._write_json(CANONICAL_DIRECTORY/"summary.json",summary);utility._write_json(CANONICAL_DIRECTORY/"input_lock.json",{"parent_artifact":parent.ARTIFACT,"parent_checksum_manifest_sha256":PARENT_SHA256,"parent_hashes":hashes})
    REPORT_PATH.parent.mkdir(parents=True,exist_ok=True);REPORT_PATH.write_text("# Native-grid eleven-field lift and cycle-coverage manifest\n\nThe physical context and every saved primitive profile contain 112 radial cells. The earlier 94-cell periodic run remains a valid scalability proof, but its production-size label is superseded: 470 is the implicit constrained-coordinate dimension, not 94 five-field cells. The binding global rerun therefore uses 112 x 11 = 1,232 variables.\n\nThe old five-field profiles lift cellwise on their native grid; no artificial radial remap is allowed. Prefix coverage is measured in the certified nonlinear trust metric. It cannot establish full-cycle phase/mode/event coverage. No cycle step is authorized.\n",encoding="utf-8")
    sources=(THIS_RUNNER,THIS_TEST,REPORT_RELATIVE);utility._write_json(CANONICAL_DIRECTORY/"provenance.json",{"implementation_commit":utility._git("rev-parse","HEAD"),"source_hashes":{source:utility._sha256(ROOT/source) for source in sources}});names=sorted(path.name for path in CANONICAL_DIRECTORY.iterdir());(CANONICAL_DIRECTORY/"SHA256SUMS.txt").write_text("".join(f"{utility._sha256(CANONICAL_DIRECTORY/name)}  {name}\n" for name in names),encoding="utf-8");_update(summary);return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--freeze",action="store_true");arguments=parser.parse_args()
    if not arguments.freeze:parser.error("choose --freeze")
    print(json.dumps(_freeze(),indent=2,sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())
