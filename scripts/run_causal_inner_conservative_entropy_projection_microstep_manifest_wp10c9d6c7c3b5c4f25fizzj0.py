#!/usr/bin/env python3
"""Supersede scalar relaxation with a conservative entropy projection."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT/"src",ROOT/"scripts"):
 if str(p) not in sys.path:sys.path.insert(0,str(p))
import run_causal_inner_bounded_nonlinear_split_microstep_manifest_wp10c9d6c7c3b5c4f25fizzj as parent  # noqa:E402
WORK_PACKAGE="definitions_only_WP10c9d6c7c3b5c4f25fizzj0_conservative_entropy_projection_microstep_manifest";CLASSIFICATION="conservative_entropy_projection_microstep_manifest_frozen";AUTHORIZED_NEXT="WP10c9d6c7c3b5c4f25fizzj2_conservative_entropy_projection_microstep_kernel";PASS_NEXT=parent.PASS_NEXT;ARTIFACT="causal_inner_conservative_entropy_projection_microstep_manifest_wp10c9d6c7c3b5c4f25fizzj0";CANONICAL_DIRECTORY=ROOT/"results/canonical"/ARTIFACT;REPORT_RELATIVE="docs/reports/current/CODEX_CAUSAL_INNER_CONSERVATIVE_ENTROPY_PROJECTION_MICROSTEP_MANIFEST_WP10C9D6C7C3B5C4F25FIZZJ0_2026-08-26.md";REPORT_PATH=ROOT/REPORT_RELATIVE;THIS_RUNNER="scripts/run_causal_inner_conservative_entropy_projection_microstep_manifest_wp10c9d6c7c3b5c4f25fizzj0.py";THIS_TEST="tests/test_causal_inner_conservative_entropy_projection_microstep_manifest_wp10c9d6c7c3b5c4f25fizzj0.py";PARENT_SHA256="244f09f277524379e33234e80e255b7bd4d43dd6d68a66538c189d24197bb789";CANONICAL_MANIFEST=ROOT/"results/manifests/canonical_artifacts.csv";CANONICAL_SUMMARY=ROOT/"results/manifests/canonical_summary.json"
def _u():return parent._u()
def _validate_parent(clean=False):
 u=_u()
 if u._sha256(parent.CANONICAL_DIRECTORY/"SHA256SUMS.txt")!=PARENT_SHA256:raise RuntimeError("microstep manifest checksum changed")
 h=u._validate_checksums(parent.CANONICAL_DIRECTORY);s=u._read_json(parent.CANONICAL_DIRECTORY/"summary.json")
 if not s["passed"] or not s["definitions_only"] or s["bounded_nonlinear_microstep_certified"] or s["trajectory_authorized"] or s["complete_cycle_execution_authorized"]:raise RuntimeError("microstep manifest classification changed")
 if clean and u._git("status","--short","--untracked-files=no"):raise RuntimeError("projection manifest needs clean tracked tree")
 return h
def _contract():
 c=parent._contract();c.update({"work_package":WORK_PACKAGE,"classification":CLASSIFICATION,"supersedes":parent.WORK_PACKAGE,"pre_execution_change":True,"scalar_gamma_policy":"withdrawn before binding execution; the nontrivial root is not guaranteed inside [0.8,1.2]","entropy_projection":{"proposal":"same explicit midpoint RK2 conserved proposal","entropy_variables":"recover v_i at the proposal","global_positive_metric":"one diagonal conserved scale M shared by all cells","zero_sum_direction":"z_i=-M*(v_i-mean(v)); therefore sum_i z_i=0 exactly","scalar_solve":"find theta such that sum eta(U_i+theta z_i) equals the initial periodic entropy","root_property":"derivative at theta=0 is -sum ||v_i-mean(v)||_M^2 < 0 unless the patch is uniform","correction_gate":.05,"theta_bound":1.0},"authorized_next":AUTHORIZED_NEXT});c["spatial_step"]["entropy_relaxation"]="conservative zero-sum entropy projection after the RK2 proposal";c["decision"]={"pass_classification":"conservative_entropy_projection_microstep_kernel_certified","pass_authorized_next":PASS_NEXT,"failure_classification":"conservative_entropy_projection_microstep_kernel_failed"};return c
def _update(s):
 u=_u();rows=list(csv.DictReader(CANONICAL_MANIFEST.open(newline="",encoding="utf-8")));rows=[r for r in rows if r.get("case")!=ARTIFACT]
 for p in sorted(CANONICAL_DIRECTORY.iterdir()):
  if p.is_file():rows.append({"case":ARTIFACT,"path":str(p.relative_to(ROOT)),"bytes":str(p.stat().st_size),"sha256":u._sha256(p),"scientific_status":"SUPPORTED"})
 with CANONICAL_MANIFEST.open("w",newline="",encoding="utf-8") as h:w=csv.DictWriter(h,fieldnames=("case","path","bytes","sha256","scientific_status"),lineterminator="\n");w.writeheader();w.writerows(rows)
 c=u._read_json(CANONICAL_SUMMARY);c.setdefault("artifacts",{})[ARTIFACT]={"path":str(CANONICAL_DIRECTORY.relative_to(ROOT)),"classification":CLASSIFICATION,"passed":True};c.update({"case_count":len({r['case'] for r in rows}),"file_count":len(rows),"total_bytes":sum(int(r['bytes']) for r in rows),"all_payload_hashes_recorded":True,"latest_source_parent_commit":u._git("rev-parse","HEAD"),"latest_work_package":WORK_PACKAGE});u._write_json(CANONICAL_SUMMARY,c)
def _freeze():
 if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():raise RuntimeError("projection manifest exists")
 h=_validate_parent(True);u=_u();CANONICAL_DIRECTORY.mkdir(parents=True);u._write_json(CANONICAL_DIRECTORY/"projection_contract.json",_contract());s={"schema_version":1,"work_package":WORK_PACKAGE,"classification":CLASSIFICATION,"passed":True,"definitions_only":True,"prior_microstep_manifest_superseded_before_execution":True,"conservative_entropy_projection_certified":False,"trajectory_authorized":False,"complete_cycle_execution_authorized":False,"authorized_next":AUTHORIZED_NEXT};u._write_json(CANONICAL_DIRECTORY/"summary.json",s);u._write_json(CANONICAL_DIRECTORY/"input_lock.json",{"parent_artifact":parent.ARTIFACT,"parent_checksum_manifest_sha256":PARENT_SHA256,"parent_hashes":h});REPORT_PATH.parent.mkdir(parents=True,exist_ok=True);REPORT_PATH.write_text("# Conservative entropy-projection microstep manifest\n\nThe unexecuted whole-increment relaxation policy is superseded. The RK2 proposal is corrected along a shared-metric, zero-sum direction, so all conserved totals remain exact while a scalar solve enforces the nonlinear entropy target. This is definitions-only and authorizes no trajectory.\n",encoding="utf-8");src=(THIS_RUNNER,THIS_TEST,REPORT_RELATIVE);u._write_json(CANONICAL_DIRECTORY/"provenance.json",{"implementation_commit":u._git("rev-parse","HEAD"),"source_hashes":{p:u._sha256(ROOT/p) for p in src}});names=sorted(p.name for p in CANONICAL_DIRECTORY.iterdir());(CANONICAL_DIRECTORY/"SHA256SUMS.txt").write_text("".join(f"{u._sha256(CANONICAL_DIRECTORY/n)}  {n}\n" for n in names));_update(s);return s
def main():
 p=argparse.ArgumentParser();p.add_argument("--freeze",action="store_true");a=p.parse_args();
 if not a.freeze:p.error("choose --freeze")
 print(json.dumps(_freeze(),indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
