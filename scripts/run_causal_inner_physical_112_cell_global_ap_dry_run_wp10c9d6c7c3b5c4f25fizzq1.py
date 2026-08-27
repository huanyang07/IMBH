#!/usr/bin/env python3
"""Execute the corrected 112-cell, 1,232-state global AP dry run."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT/"src",ROOT/"scripts"):
    if str(path) not in sys.path:sys.path.insert(0,str(path))

import run_causal_inner_cycle_wide_eleven_field_anchor_coverage_and_lift_manifest_wp10c9d6c7c3b5c4f25fizzq as manifest  # noqa:E402
import run_causal_inner_production_size_global_ap_dry_run_wp10c9d6c7c3b5c4f25fizzo1 as proof  # noqa:E402
import run_causal_inner_bounded_ap_coarse_trajectory_kernel_wp10c9d6c7c3b5c4f25fizzm1 as physical_builder  # noqa:E402


WORK_PACKAGE=manifest.AUTHORIZED_NEXT
PASS_CLASSIFICATION="physical_112_cell_global_AP_dry_run_certified"
FAIL_CLASSIFICATION="physical_112_cell_global_AP_dry_run_failed"
AUTHORIZED_NEXT=manifest.PASS_NEXT
ARTIFACT="causal_inner_physical_112_cell_global_ap_dry_run_wp10c9d6c7c3b5c4f25fizzq1"
CANONICAL_DIRECTORY=ROOT/"results/canonical"/ARTIFACT
REPORT_RELATIVE="docs/reports/current/CODEX_CAUSAL_INNER_PHYSICAL_112_CELL_GLOBAL_AP_DRY_RUN_WP10C9D6C7C3B5C4F25FIZZQ1_2026-08-27.md"
REPORT_PATH=ROOT/REPORT_RELATIVE
THIS_RUNNER="scripts/run_causal_inner_physical_112_cell_global_ap_dry_run_wp10c9d6c7c3b5c4f25fizzq1.py"
THIS_TEST="tests/test_causal_inner_physical_112_cell_global_ap_dry_run_wp10c9d6c7c3b5c4f25fizzq1.py"
PARENT_SHA256="fa32a3d8fff263c3599e939a31c0c40e18f92c3203ea061866a29306ab9bd9ea"
CANONICAL_MANIFEST=ROOT/"results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY=ROOT/"results/manifests/canonical_summary.json"


def _u():return manifest._u()


def _validate_parent(require_clean=False):
    utility=_u()
    if utility._sha256(manifest.CANONICAL_DIRECTORY/"SHA256SUMS.txt")!=PARENT_SHA256:raise RuntimeError("native-grid lift manifest changed")
    hashes=utility._validate_checksums(manifest.CANONICAL_DIRECTORY);summary=utility._read_json(manifest.CANONICAL_DIRECTORY/"summary.json");contract=utility._read_json(manifest.CANONICAL_DIRECTORY/"lift_and_coverage_contract.json")
    if not summary["passed"] or not summary["definitions_only"] or summary["physical_grid_cells"]!=112 or summary["eleven_field_global_dimension"]!=1232 or summary["authorized_next"]!=WORK_PACKAGE or contract["grid_correction"]["production_size_claim_superseded"] is not True or summary["complete_cycle_execution_authorized"]:raise RuntimeError("native-grid lift classification changed")
    if require_clean and utility._git("status","--short","--untracked-files=no"):raise RuntimeError("112-cell global AP dry run needs a clean tracked tree")
    return hashes,contract


def _physical_context_cell_count():
    witnesses=physical_builder.manifest.parent.witnesses
    source=witnesses.frozen_audit.parent.parent.parent.boundary_diagnostic.manifest.parent.engine.execution.source
    context=source._initial_inputs()["base"]["configuration"]["context"]
    return int(context.grid.centers.size)


def _certificate():
    began=time.perf_counter();_,contract=_validate_parent();spec=contract["physical_grid_global_AP_dry_run"];physical_cells=_physical_context_cell_count()
    if physical_cells!=int(spec["radial_cells"]):raise RuntimeError("physical context cell count disagrees with frozen grid")
    pairs={name:tuple(values) for name,values in spec["physical_anchor_paths"].items()};indices=sorted({index for pair in pairs.values() for index in pair})
    offline_began=time.perf_counter();ports=physical_builder._physical_ports(indices);offline_wall=time.perf_counter()-offline_began
    proof_contract={"global_AP_dry_run":spec}
    with tempfile.TemporaryDirectory(prefix="physical-112-global-ap-") as temporary:cases=[proof._case(name,pair,ports,proof_contract,Path(temporary)) for name,pair in pairs.items()]
    rows=[case[0] for case in cases];passed=bool(physical_cells==112 and len(rows)==2 and all(row["passed"] for row in rows))
    metrics={"schema_version":1,"work_package":WORK_PACKAGE,"classification":PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,"passed":passed,"physical_context_cells":physical_cells,"radial_cells":spec["radial_cells"],"fields_per_cell":spec["fields_per_cell"],"global_state_dimension":spec["global_state_dimension"],"minimum_matched_refinement_order":float(min(row["minimum_matched_refinement_order"] for row in rows)),"maximum_homogeneous_mode_expansivity":float(max(row["maximum_homogeneous_mode_expansivity"] for row in rows)),"maximum_core_total_conservation_defect":float(max(row["maximum_core_total_conservation_defect"] for row in rows)),"maximum_state_norm":float(max(row["maximum_state_norm"] for row in rows)),"maximum_projected_100k_step_wall_days":float(max(row["projected_100k_step_wall_days"] for row in rows)),"all_checkpoints_bitwise":all(row["checkpoint_roundtrip_bitwise"] for row in rows),"all_suffix_replays_bitwise":all(row["suffix_replay_bitwise"] for row in rows),"minimum_source_nullity":min(min(row["source_nullities"]) for row in rows),"offline_physical_anchor_builds":len(ports),"offline_anchor_wall_seconds":offline_wall,"online_truth_calls":0,"previous_94_cell_scalability_certificate_preserved":True,"physical_boundaries_certified":False,"native_five_to_eleven_lift_certified":False,"cycle_wide_inputs_complete":False,"complete_cycle_execution_authorized":False,"complete_cycle_steps":0,"certificate_wall_seconds":time.perf_counter()-began,"rows":rows,"authorized_next":AUTHORIZED_NEXT if passed else None}
    arrays={"final_states":np.asarray([case[1] for case in cases]),"anchor_indices":np.asarray(list(pairs.values()))}
    return metrics,arrays


def _update(summary):
    utility=_u();rows=list(csv.DictReader(CANONICAL_MANIFEST.open(newline="",encoding="utf-8")));rows=[row for row in rows if row.get("case")!=ARTIFACT];status="SUPPORTED" if summary["passed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():rows.append({"case":ARTIFACT,"path":str(path.relative_to(ROOT)),"bytes":str(path.stat().st_size),"sha256":utility._sha256(path),"scientific_status":status})
    with CANONICAL_MANIFEST.open("w",newline="",encoding="utf-8") as handle:writer=csv.DictWriter(handle,fieldnames=("case","path","bytes","sha256","scientific_status"),lineterminator="\n");writer.writeheader();writer.writerows(rows)
    catalog=utility._read_json(CANONICAL_SUMMARY);catalog.setdefault("artifacts",{})[ARTIFACT]={"path":str(CANONICAL_DIRECTORY.relative_to(ROOT)),"classification":summary["classification"],"passed":summary["passed"]};catalog.update({"case_count":len({row["case"] for row in rows}),"file_count":len(rows),"total_bytes":sum(int(row["bytes"]) for row in rows),"all_payload_hashes_recorded":True,"latest_source_parent_commit":utility._git("rev-parse","HEAD"),"latest_work_package":WORK_PACKAGE});utility._write_json(CANONICAL_SUMMARY,catalog)


def _canonicalize(metrics,arrays):
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():raise RuntimeError("112-cell global AP certificate exists")
    hashes,_=_validate_parent(require_clean=True);utility=_u();CANONICAL_DIRECTORY.mkdir(parents=True);utility._write_json(CANONICAL_DIRECTORY/"global_dry_run_metrics.json",metrics);np.savez_compressed(CANONICAL_DIRECTORY/"global_dry_run_arrays.npz",**arrays)
    summary={"schema_version":1,"work_package":WORK_PACKAGE,"classification":metrics["classification"],"passed":metrics["passed"],"physical_112_cell_global_AP_certified":metrics["passed"],"physical_context_cells":metrics["physical_context_cells"],"global_state_dimension":metrics["global_state_dimension"],"previous_94_cell_scalability_certificate_preserved":True,"physical_boundaries_certified":False,"native_five_to_eleven_lift_certified":False,"cycle_wide_inputs_complete":False,"complete_cycle_execution_authorized":False,"complete_cycle_steps":0,"authorized_next":metrics["authorized_next"]};utility._write_json(CANONICAL_DIRECTORY/"summary.json",summary);utility._write_json(CANONICAL_DIRECTORY/"input_lock.json",{"manifest_artifact":manifest.ARTIFACT,"manifest_checksum_manifest_sha256":PARENT_SHA256,"manifest_hashes":hashes})
    REPORT_PATH.parent.mkdir(parents=True,exist_ok=True);REPORT_PATH.write_text(f"# Physical 112-cell global AP dry-run certificate\n\nClassification: `{metrics['classification']}`.\n\nThe corrected native grid advances 112 radial cells x 11 fields = 1,232 variables. The minimum matched order is `{metrics['minimum_matched_refinement_order']:.6f}`, maximum core-total conservation defect `{metrics['maximum_core_total_conservation_defect']:.6e}`, and projected 100,000-step cost `{metrics['maximum_projected_100k_step_wall_days']:.6e}` wall days. Restart and suffix replay are bitwise.\n\nThe old 94-cell result remains a scalability proof. This result clears the native-grid global action only; physical boundaries, the native profile lift, cycle coverage, events, and resets remain blocked. No cycle step occurred.\n",encoding="utf-8")
    sources=(THIS_RUNNER,THIS_TEST,proof.GLOBAL_SOURCE,proof.GLOBAL_TEST,REPORT_RELATIVE);utility._write_json(CANONICAL_DIRECTORY/"provenance.json",{"implementation_commit":utility._git("rev-parse","HEAD"),"source_hashes":{source:utility._sha256(ROOT/source) for source in sources},"python":sys.version,"numpy":np.__version__,"platform":platform.platform(),"thread_environment":{name:os.environ.get(name,"") for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS")}});names=sorted(path.name for path in CANONICAL_DIRECTORY.iterdir());(CANONICAL_DIRECTORY/"SHA256SUMS.txt").write_text("".join(f"{utility._sha256(CANONICAL_DIRECTORY/name)}  {name}\n" for name in names),encoding="utf-8");_update(summary);return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run",action="store_true");arguments=parser.parse_args()
    if not arguments.run:parser.error("choose --run")
    metrics,arrays=_certificate();print(json.dumps(metrics,indent=2,sort_keys=True),flush=True);return 0 if _canonicalize(metrics,arrays)["passed"] else 2


if __name__=="__main__":raise SystemExit(main())
