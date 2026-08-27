#!/usr/bin/env python3
"""Freeze the native prefix port batch and 11-field boundary lift."""

from __future__ import annotations

import argparse,csv,json,sys
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT/"src",ROOT/"scripts"):
    if str(path) not in sys.path:sys.path.insert(0,str(path))

import run_causal_inner_native_grid_five_to_eleven_field_lift_and_prefix_coverage_audit_wp10c9d6c7c3b5c4f25fizzq2 as parent  # noqa:E402


WORK_PACKAGE=parent.AUTHORIZED_NEXT
CLASSIFICATION="prefix_913_port_payload_batch_and_eleven_field_boundary_lift_manifest_frozen"
AUTHORIZED_NEXT="WP10c9d6c7c3b5c4f25fizzr1_prefix_port_payload_and_boundary_structure_certificate"
PASS_NEXT="definitions_only_WP10c9d6c7c3b5c4f25fizzs_cycle_wide_missing_input_acquisition_and_event_reset_manifest"
ARTIFACT="causal_inner_prefix_anchor_batch_build_and_physical_boundary_lift_manifest_wp10c9d6c7c3b5c4f25fizzr"
CANONICAL_DIRECTORY=ROOT/"results/canonical"/ARTIFACT
REPORT_RELATIVE="docs/reports/current/CODEX_CAUSAL_INNER_PREFIX_ANCHOR_BATCH_BUILD_AND_PHYSICAL_BOUNDARY_LIFT_MANIFEST_WP10C9D6C7C3B5C4F25FIZZR_2026-08-27.md"
REPORT_PATH=ROOT/REPORT_RELATIVE
THIS_RUNNER="scripts/run_causal_inner_prefix_anchor_batch_build_and_physical_boundary_lift_manifest_wp10c9d6c7c3b5c4f25fizzr.py"
THIS_TEST="tests/test_causal_inner_prefix_anchor_batch_build_and_physical_boundary_lift_manifest_wp10c9d6c7c3b5c4f25fizzr.py"
PARENT_SHA256="7cca05eff1dd0d389c11dd7fb8a79314a338c8ec7e782a0f4694993b00233a19"
CANONICAL_MANIFEST=ROOT/"results/manifests/canonical_artifacts.csv";CANONICAL_SUMMARY=ROOT/"results/manifests/canonical_summary.json"


def _u():return parent._u()


def _validate_parent(require_clean=False):
    utility=_u()
    if utility._sha256(parent.CANONICAL_DIRECTORY/"SHA256SUMS.txt")!=PARENT_SHA256:raise RuntimeError("native lift/coverage audit changed")
    hashes=utility._validate_checksums(parent.CANONICAL_DIRECTORY);summary=utility._read_json(parent.CANONICAL_DIRECTORY/"summary.json");metrics=utility._read_json(parent.CANONICAL_DIRECTORY/"lift_and_coverage_metrics.json")
    if not summary["passed"] or not summary["native_five_to_eleven_lift_certified"] or not summary["prefix_candidate_cover_frozen"] or summary["prefix_candidate_anchor_count"]!=913 or summary["authorized_next"]!=WORK_PACKAGE or summary["complete_cycle_execution_authorized"]:raise RuntimeError("native lift/coverage classification changed")
    if metrics["candidate_anchors_per_cell_maximum"]!=17 or metrics["candidate_anchors_per_cell_minimum"]!=2:raise RuntimeError("prefix anchor inventory changed")
    if require_clean and utility._git("status","--short","--untracked-files=no"):raise RuntimeError("prefix batch manifest needs clean tracked tree")
    return hashes,metrics


def _contract():
    return {"schema_version":1,"work_package":WORK_PACKAGE,"classification":CLASSIFICATION,
    "prefix_port_batch":{"candidate_anchor_count":913,"native_radial_cells":112,"payload_per_anchor":["cell and source lineage","radius and primitive chart5/chart7","anchor-local state11","physical entropy congruence","corrected symmetric radial matrix11x11","dissipative/reversible source matrix11x11","characteristic speeds and source nullity","trust and physical guard diagnostics"],"binding_audits":["equilibrium entropy congruence","full port causality and reciprocity","corrected physical core reconstruction","Rphi-to-five-STF embedding","finite payload and lineage","maximum build wall hours"],"maximum_build_wall_hours":1.0,"new_truth_calls":0,"slow_forcing_b_included":False,"claim":"prefix port/coefficient payload only; full slow forcing remains a later input"},
    "boundary_lift":{"representation":"symmetric entropy-port characteristic decomposition of outward-normal A_n","incoming_definition":"negative eigenvalues of outward-normal A_n","penalty":"D_in=V diag(max(-lambda_n,0)) V^T","inner":{"cell":0,"expected_candidate_anchors":17,"expected_incoming_count":0,"mode":"pure excision/outflow"},"outer":{"cell":111,"expected_candidate_anchors":2,"expected_incoming_count":11,"mode":"prescribed incoming physical loading","candidate_loading":"lifted frozen old exterior chart","cycle_wide_loading_complete":False},"gates":{"maximum_symmetry_defect":2e-12,"maximum_projector_idempotence_defect":2e-12,"minimum_penalty_eigenvalue":-2e-12,"maximum_characteristic_reconstruction_defect":2e-12,"maximum_absolute_speed_over_c":0.999}},
    "decision":{"pass_classification":"prefix_913_port_payloads_and_eleven_field_boundary_structure_certified_outer_cycle_loading_missing","pass_authorized_next":PASS_NEXT,"failure_classification":"prefix_port_payload_or_boundary_structure_failed"},
    "claim_boundary":{"prefix_port_payloads_built":False,"full_slow_forcing_complete":False,"outer_cycle_loading_complete":False,"cycle_wide_inputs_complete":False,"events_and_resets_certified":False,"complete_cycle_execution_authorized":False,"complete_cycle_steps":0},"authorized_next":AUTHORIZED_NEXT}


def _update(summary):
    utility=_u();rows=list(csv.DictReader(CANONICAL_MANIFEST.open(newline="",encoding="utf-8")));rows=[row for row in rows if row.get("case")!=ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():rows.append({"case":ARTIFACT,"path":str(path.relative_to(ROOT)),"bytes":str(path.stat().st_size),"sha256":utility._sha256(path),"scientific_status":"SUPPORTED"})
    with CANONICAL_MANIFEST.open("w",newline="",encoding="utf-8") as handle:writer=csv.DictWriter(handle,fieldnames=("case","path","bytes","sha256","scientific_status"),lineterminator="\n");writer.writeheader();writer.writerows(rows)
    catalog=utility._read_json(CANONICAL_SUMMARY);catalog.setdefault("artifacts",{})[ARTIFACT]={"path":str(CANONICAL_DIRECTORY.relative_to(ROOT)),"classification":summary["classification"],"passed":True};catalog.update({"case_count":len({row["case"] for row in rows}),"file_count":len(rows),"total_bytes":sum(int(row["bytes"]) for row in rows),"all_payload_hashes_recorded":True,"latest_source_parent_commit":utility._git("rev-parse","HEAD"),"latest_work_package":WORK_PACKAGE});utility._write_json(CANONICAL_SUMMARY,catalog)


def _freeze():
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():raise RuntimeError("prefix batch manifest exists")
    hashes,_=_validate_parent(require_clean=True);utility=_u();CANONICAL_DIRECTORY.mkdir(parents=True);utility._write_json(CANONICAL_DIRECTORY/"batch_and_boundary_contract.json",_contract());summary={"schema_version":1,"work_package":WORK_PACKAGE,"classification":CLASSIFICATION,"passed":True,"definitions_only":True,"candidate_anchor_count":913,"prefix_port_payloads_built":False,"eleven_field_boundary_structure_certified":False,"outer_cycle_loading_complete":False,"cycle_wide_inputs_complete":False,"events_and_resets_certified":False,"complete_cycle_execution_authorized":False,"complete_cycle_steps":0,"authorized_next":AUTHORIZED_NEXT};utility._write_json(CANONICAL_DIRECTORY/"summary.json",summary);utility._write_json(CANONICAL_DIRECTORY/"input_lock.json",{"parent_artifact":parent.ARTIFACT,"parent_checksum_manifest_sha256":PARENT_SHA256,"parent_hashes":hashes})
    REPORT_PATH.parent.mkdir(parents=True,exist_ok=True);REPORT_PATH.write_text("# Prefix port-payload batch and physical boundary-lift manifest\n\nThe 913 native-grid candidate anchors are built in one context initialization and audited independently. Each stores the corrected 11-field symmetric radial/source operators and exact lineage. Slow forcing is deliberately not manufactured.\n\nBoundary closure is formulated in outward-normal entropy characteristics. The inner edge must remain pure excision. The outer edge requires prescribed incoming data; the old frozen exterior chart is only a prefix candidate and cannot supply full-cycle loading. No cycle step is authorized.\n",encoding="utf-8");sources=(THIS_RUNNER,THIS_TEST,REPORT_RELATIVE);utility._write_json(CANONICAL_DIRECTORY/"provenance.json",{"implementation_commit":utility._git("rev-parse","HEAD"),"source_hashes":{source:utility._sha256(ROOT/source) for source in sources}});names=sorted(path.name for path in CANONICAL_DIRECTORY.iterdir());(CANONICAL_DIRECTORY/"SHA256SUMS.txt").write_text("".join(f"{utility._sha256(CANONICAL_DIRECTORY/name)}  {name}\n" for name in names),encoding="utf-8");_update(summary);return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--freeze",action="store_true");arguments=parser.parse_args()
    if not arguments.freeze:parser.error("choose --freeze")
    print(json.dumps(_freeze(),indent=2,sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())
