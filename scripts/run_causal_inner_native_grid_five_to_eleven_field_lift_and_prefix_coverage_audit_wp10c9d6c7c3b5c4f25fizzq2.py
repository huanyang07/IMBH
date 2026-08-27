#!/usr/bin/env python3
"""Certify the algebraic 5->11 lift and plan native-grid prefix anchors."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT/"src",ROOT/"scripts"):
    if str(path) not in sys.path:sys.path.insert(0,str(path))

import run_causal_inner_physical_112_cell_global_ap_dry_run_wp10c9d6c7c3b5c4f25fizzq1 as parent  # noqa:E402
from imri_qpe.layer3_minidisk_1d.causal_inner_eleven_field_convex import one_Rphi_amplitude_embedding  # noqa:E402


WORK_PACKAGE=parent.AUTHORIZED_NEXT
PASS_CLASSIFICATION="native_grid_eleven_field_lift_certified_prefix_913_anchor_acquisition_plan_frozen_cycle_coverage_missing"
FAIL_CLASSIFICATION="native_grid_five_to_eleven_field_lift_or_prefix_coverage_failed"
AUTHORIZED_NEXT="definitions_only_WP10c9d6c7c3b5c4f25fizzr_prefix_anchor_batch_build_and_physical_boundary_lift_manifest"
ARTIFACT="causal_inner_native_grid_five_to_eleven_field_lift_and_prefix_coverage_audit_wp10c9d6c7c3b5c4f25fizzq2"
CANONICAL_DIRECTORY=ROOT/"results/canonical"/ARTIFACT
REPORT_RELATIVE="docs/reports/current/CODEX_CAUSAL_INNER_NATIVE_GRID_FIVE_TO_ELEVEN_FIELD_LIFT_AND_PREFIX_COVERAGE_AUDIT_WP10C9D6C7C3B5C4F25FIZZQ2_2026-08-27.md"
REPORT_PATH=ROOT/REPORT_RELATIVE
THIS_RUNNER="scripts/run_causal_inner_native_grid_five_to_eleven_field_lift_and_prefix_coverage_audit_wp10c9d6c7c3b5c4f25fizzq2.py"
THIS_TEST="tests/test_causal_inner_native_grid_five_to_eleven_field_lift_and_prefix_coverage_audit_wp10c9d6c7c3b5c4f25fizzq2.py"
PARENT_SHA256="85d5da9b89ffe0fe261bda7e6da89b2ee677cc88f74c3edeea30a7958e83ac01"
CANONICAL_MANIFEST=ROOT/"results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY=ROOT/"results/manifests/canonical_summary.json"

COMPLETE=ROOT/"results/canonical/causal_inner_adaptive_complete_cycle_execution_wp10c9d6c7c3b5c4f25fe"
PORT=ROOT/"results/canonical/causal_inner_fully_split_physical_port_atlas_kernel_wp10c9d6c7c3b5c4f25fizzg1"
EQUILIBRIUM=ROOT/"results/canonical/causal_inner_equilibrium_selected_metric_stencil_full_rerun_wp10c9d6c7c3b5c4f25fizzc6"
FLUX=ROOT/"results/canonical/causal_inner_compensated_discrete_gradient_flux_kernel_wp10c9d6c7c3b5c4f25fizzi5"
SUPPORT_HASHES={COMPLETE:"0b018b004798f28e5ec5d5f0e70b2bfb26ee6fca0b05a42f53f810246a061aab",PORT:"45b9f9fa5e26101850e885132633eb30bebb8a3df5c19147ba52fd3535205572",EQUILIBRIUM:"86d2f9410100896fb023573dc7a283668f4e3e0fc490df2a7e8e95f87f3d0167",FLUX:"3de755c0beb6d215dd2b73e9e1f2f34828881e50a237ae5c0a010b2e433ec5a1"}
TRUST_SCALES=np.asarray((0.01,0.002,0.002,0.01),dtype=float)


def _u():return parent._u()


def _validate_parent(require_clean=False):
    utility=_u()
    if utility._sha256(parent.CANONICAL_DIRECTORY/"SHA256SUMS.txt")!=PARENT_SHA256:raise RuntimeError("112-cell global certificate changed")
    hashes=utility._validate_checksums(parent.CANONICAL_DIRECTORY);summary=utility._read_json(parent.CANONICAL_DIRECTORY/"summary.json")
    if not summary["passed"] or not summary["physical_112_cell_global_AP_certified"] or summary["physical_context_cells"]!=112 or summary["authorized_next"]!=WORK_PACKAGE or summary["complete_cycle_execution_authorized"]:raise RuntimeError("112-cell global classification changed")
    support={}
    for directory,expected in SUPPORT_HASHES.items():
        if utility._sha256(directory/"SHA256SUMS.txt")!=expected:raise RuntimeError(f"supporting certificate changed: {directory.name}")
        support[directory.name]=utility._validate_checksums(directory)
    port_summary=utility._read_json(PORT/"summary.json");equilibrium_summary=utility._read_json(EQUILIBRIUM/"summary.json");flux_summary=utility._read_json(FLUX/"summary.json")
    if not port_summary["fully_split_physical_port_atlas_kernel_certified"] or not equilibrium_summary["equilibrium_physical_potential_certified"] or not flux_summary["compensated_discrete_gradient_flux_certified"]:raise RuntimeError("supporting eleven-field lift chain changed")
    if require_clean and utility._git("status","--short","--untracked-files=no"):raise RuntimeError("native lift audit needs a clean tracked tree")
    return hashes,support


def _load_prefix():
    with np.load(COMPLETE/"cycle_execution_arrays.npz",allow_pickle=False) as payload:trajectory=np.asarray(payload["trajectory_primitive_states"],dtype=float)
    with np.load(COMPLETE/"exact_witness_arrays.npz",allow_pickle=False) as payload:witnesses=np.asarray(payload["primitive_states"],dtype=float)
    return trajectory,witnesses


def _select_cellwise_cover(profiles):
    selected_cell=[];selected_global=[];maximum=[];nearest_all=np.empty(profiles.shape[:2],dtype=float)
    for cell in range(profiles.shape[1]):
        points=profiles[:,cell,:4];selected=[0];nearest=np.max(np.abs(points-points[0])/TRUST_SCALES,axis=1)
        while float(np.max(nearest))>1.0:
            index=int(np.argmax(nearest));selected.append(index);nearest=np.minimum(nearest,np.max(np.abs(points-points[index])/TRUST_SCALES,axis=1))
        selected_cell.extend([cell]*len(selected));selected_global.extend(selected);maximum.append(float(np.max(nearest)));nearest_all[:,cell]=nearest
    return np.asarray(selected_cell,dtype=int),np.asarray(selected_global,dtype=int),nearest_all,np.asarray(maximum)


def _lift_anchor_states(charts5):
    values=np.asarray(charts5,dtype=float);lifted=np.zeros((len(values),11),dtype=float)
    for index,stress in enumerate(values[:,4]):lifted[index,4:9]=one_Rphi_amplitude_embedding(float(stress))
    return lifted


def _audit():
    _validate_parent();trajectory,witnesses=_load_prefix();profiles=np.concatenate((trajectory,witnesses),axis=0)
    selected_cell,selected_global,nearest,cell_maximum=_select_cellwise_cover(profiles);selected_charts=profiles[selected_global,selected_cell]
    lifted=_lift_anchor_states(selected_charts);stress_roundtrip=float(np.max(np.abs(lifted[:,6]/np.sqrt(2.0)-selected_charts[:,4])))
    with np.load(PORT/"kernel_arrays.npz",allow_pickle=False) as payload:witness_charts7=np.asarray(payload["witness_charts7"],dtype=float)
    witness_lifted=_lift_anchor_states(witness_charts7[:,:5]);witness_stress_roundtrip=float(np.max(np.abs(witness_lifted[:,6]/np.sqrt(2.0)-witness_charts7[:,4])))
    port_metrics=_u()._read_json(PORT/"kernel_metrics.json");equilibrium_metrics=_u()._read_json(EQUILIBRIUM/"certificate_metrics.json");flux_metrics=_u()._read_json(FLUX/"kernel_metrics.json")
    witness_box_min=np.min(witness_charts7[:,:5],axis=0);witness_box_max=np.max(witness_charts7[:,:5],axis=0);componentwise_enclosed=bool(np.all(profiles>=witness_box_min[None,None,:]) and np.all(profiles<=witness_box_max[None,None,:]))
    optimistic=np.min(np.max(np.abs(profiles[:,:,None,:4]-witness_charts7[None,None,:,:4])/TRUST_SCALES,axis=3),axis=2)
    counts=np.bincount(selected_cell,minlength=112);candidate_count=int(len(selected_cell));conservative_seconds_per_anchor=float(parent._u()._read_json(parent.CANONICAL_DIRECTORY/"global_dry_run_metrics.json")["offline_anchor_wall_seconds"])/4.0
    passed=bool(trajectory.shape==(65,112,5) and witnesses.shape==(192,112,5) and candidate_count==913 and float(np.max(nearest))<=1.0+1e-12 and stress_roundtrip<=1e-15 and witness_stress_roundtrip<=1e-15 and np.max(np.abs(lifted[:,:4]))==0 and np.max(np.abs(lifted[:,4:6]))==0 and np.max(np.abs(lifted[:,7:]))==0 and componentwise_enclosed and port_metrics["passed"] and port_metrics["passing_witness_count"]==47 and equilibrium_metrics["passed"] and equilibrium_metrics["physical_witness_count"]==47 and flux_metrics["passed"] and flux_metrics["passing_endpoint_pair_count"]==376)
    metrics={"schema_version":1,"work_package":WORK_PACKAGE,"classification":PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,"passed":passed,"native_grid_cells":112,"saved_prefix_profile_count":int(len(profiles)),"saved_prefix_cell_state_count":int(np.prod(profiles.shape[:2])),"candidate_anchor_count":candidate_count,"candidate_anchors_per_cell_minimum":int(np.min(counts)),"candidate_anchors_per_cell_median":float(np.median(counts)),"candidate_anchors_per_cell_maximum":int(np.max(counts)),"maximum_candidate_cover_trust_fraction":float(np.max(nearest)),"componentwise_existing_witness_envelope_contains_prefix":componentwise_enclosed,"optimistic_cross_radius_existing_47_anchor_trust_coverage_fraction":float(np.mean(optimistic<=1.0)),"algebraic_lift_stress_roundtrip_defect":stress_roundtrip,"witness_lift_stress_roundtrip_defect":witness_stress_roundtrip,"anchor_local_equilibrium_coordinates_zero":True,"anchor_local_height_coordinate_zero":True,"anchor_vertical_velocity_zero":True,"certified_port_witnesses":int(port_metrics["passing_witness_count"]),"certified_equilibrium_witnesses":int(equilibrium_metrics["physical_witness_count"]),"certified_compensated_flux_endpoint_pairs":int(flux_metrics["passing_endpoint_pair_count"]),"conservative_prefix_payload_build_projection_wall_hours":candidate_count*conservative_seconds_per_anchor/3600.0,"new_truth_calls":0,"new_trajectory_steps":0,"native_five_to_eleven_lift_certified":passed,"prefix_candidate_cover_frozen":passed,"prefix_coefficient_payloads_built":False,"cycle_wide_inputs_complete":False,"physical_boundaries_certified":False,"events_and_resets_certified":False,"complete_cycle_execution_authorized":False,"complete_cycle_steps":0,"authorized_next":AUTHORIZED_NEXT if passed else None}
    arrays={"selected_cell_indices":selected_cell,"selected_profile_indices":selected_global,"selected_source_kinds":np.asarray(selected_global>=len(trajectory),dtype=np.int8),"selected_source_indices":np.where(selected_global<len(trajectory),selected_global,selected_global-len(trajectory)),"selected_charts5":selected_charts,"selected_anchor_local_states11":lifted,"nearest_trust_fractions":nearest,"cell_anchor_counts":counts,"cell_maximum_trust_fractions":cell_maximum,"witness_charts7":witness_charts7,"witness_anchor_local_states11":witness_lifted}
    return metrics,arrays


def _update(summary):
    utility=_u();rows=list(csv.DictReader(CANONICAL_MANIFEST.open(newline="",encoding="utf-8")));rows=[row for row in rows if row.get("case")!=ARTIFACT];status="SUPPORTED" if summary["passed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():rows.append({"case":ARTIFACT,"path":str(path.relative_to(ROOT)),"bytes":str(path.stat().st_size),"sha256":utility._sha256(path),"scientific_status":status})
    with CANONICAL_MANIFEST.open("w",newline="",encoding="utf-8") as handle:writer=csv.DictWriter(handle,fieldnames=("case","path","bytes","sha256","scientific_status"),lineterminator="\n");writer.writeheader();writer.writerows(rows)
    catalog=utility._read_json(CANONICAL_SUMMARY);catalog.setdefault("artifacts",{})[ARTIFACT]={"path":str(CANONICAL_DIRECTORY.relative_to(ROOT)),"classification":summary["classification"],"passed":summary["passed"]};catalog.update({"case_count":len({row["case"] for row in rows}),"file_count":len(rows),"total_bytes":sum(int(row["bytes"]) for row in rows),"all_payload_hashes_recorded":True,"latest_source_parent_commit":utility._git("rev-parse","HEAD"),"latest_work_package":WORK_PACKAGE});utility._write_json(CANONICAL_SUMMARY,catalog)


def _canonicalize(metrics,arrays):
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():raise RuntimeError("native lift audit exists")
    parent_hashes,support_hashes=_validate_parent(require_clean=True);utility=_u();CANONICAL_DIRECTORY.mkdir(parents=True);utility._write_json(CANONICAL_DIRECTORY/"lift_and_coverage_metrics.json",metrics);np.savez_compressed(CANONICAL_DIRECTORY/"lift_and_coverage_arrays.npz",**arrays)
    summary={"schema_version":1,"work_package":WORK_PACKAGE,"classification":metrics["classification"],"passed":metrics["passed"],"native_five_to_eleven_lift_certified":metrics["native_five_to_eleven_lift_certified"],"prefix_candidate_cover_frozen":metrics["prefix_candidate_cover_frozen"],"prefix_candidate_anchor_count":metrics["candidate_anchor_count"],"prefix_coefficient_payloads_built":False,"cycle_wide_inputs_complete":False,"physical_boundaries_certified":False,"events_and_resets_certified":False,"complete_cycle_execution_authorized":False,"complete_cycle_steps":0,"authorized_next":metrics["authorized_next"]};utility._write_json(CANONICAL_DIRECTORY/"summary.json",summary);utility._write_json(CANONICAL_DIRECTORY/"input_lock.json",{"parent_artifact":parent.ARTIFACT,"parent_checksum_manifest_sha256":PARENT_SHA256,"parent_hashes":parent_hashes,"support_hashes":support_hashes})
    REPORT_PATH.parent.mkdir(parents=True,exist_ok=True);REPORT_PATH.write_text(f"# Native-grid five-to-eleven-field lift and prefix-coverage audit\n\nClassification: `{metrics['classification']}`.\n\nThe old R-phi stress embeds exactly into the five-STF sector and the anchor-local equilibrium, height, and vertical coordinates have an explicit deterministic convention. The 47 prior physical witnesses retain certified equilibrium potential, full port, and compensated interface-flux evidence.\n\nA deterministic per-cell farthest-point cover needs `{metrics['candidate_anchor_count']}` anchors for all `{metrics['saved_prefix_cell_state_count']}` saved cell states at trust fraction <= `{metrics['maximum_candidate_cover_trust_fraction']:.6f}`. Even allowing an optimistic cross-radius comparison, the existing 47 anchors cover only `{metrics['optimistic_cross_radius_existing_47_anchor_trust_coverage_fraction']:.3%}`. The projected conservative build cost is `{metrics['conservative_prefix_payload_build_projection_wall_hours']:.2f}` wall hours.\n\nThis certifies the lift and freezes a prefix acquisition plan, not cycle-wide coverage. The saved prefix spans only 0.016 s and contains no hot exit or cycle return. No cycle step is authorized.\n",encoding="utf-8")
    sources=(THIS_RUNNER,THIS_TEST,REPORT_RELATIVE);utility._write_json(CANONICAL_DIRECTORY/"provenance.json",{"implementation_commit":utility._git("rev-parse","HEAD"),"source_hashes":{source:utility._sha256(ROOT/source) for source in sources},"numpy":np.__version__});names=sorted(path.name for path in CANONICAL_DIRECTORY.iterdir());(CANONICAL_DIRECTORY/"SHA256SUMS.txt").write_text("".join(f"{utility._sha256(CANONICAL_DIRECTORY/name)}  {name}\n" for name in names),encoding="utf-8");_update(summary);return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run",action="store_true");arguments=parser.parse_args()
    if not arguments.run:parser.error("choose --run")
    metrics,arrays=_audit();print(json.dumps(metrics,indent=2,sort_keys=True),flush=True);return 0 if _canonicalize(metrics,arrays)["passed"] else 2


if __name__=="__main__":raise SystemExit(main())
