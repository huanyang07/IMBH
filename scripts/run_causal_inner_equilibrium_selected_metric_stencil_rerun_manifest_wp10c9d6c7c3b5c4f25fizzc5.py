#!/usr/bin/env python3
"""Freeze the selected-factor full equilibrium rerun."""
from __future__ import annotations
import argparse,csv,json,os,platform,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT/"src",ROOT/"scripts"):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
sys.setrecursionlimit(max(sys.getrecursionlimit(),5000))
import run_causal_inner_equilibrium_metric_stencil_ladder_diagnostic_wp10c9d6c7c3b5c4f25fizzc4 as parent  # noqa:E402
WORK_PACKAGE=parent.AUTHORIZED_NEXT;CLASSIFICATION="equilibrium_selected_metric_stencil_rerun_manifest_frozen";AUTHORIZED_NEXT="WP10c9d6c7c3b5c4f25fizzc6_equilibrium_selected_metric_stencil_full_rerun"
ARTIFACT="causal_inner_equilibrium_selected_metric_stencil_rerun_manifest_wp10c9d6c7c3b5c4f25fizzc5";CANONICAL_DIRECTORY=ROOT/"results/canonical"/ARTIFACT
REPORT_RELATIVE="docs/reports/current/CODEX_CAUSAL_INNER_EQUILIBRIUM_SELECTED_METRIC_STENCIL_RERUN_MANIFEST_WP10C9D6C7C3B5C4F25FIZZC5_2026-08-26.md";REPORT_PATH=ROOT/REPORT_RELATIVE;THIS_RUNNER="scripts/run_causal_inner_equilibrium_selected_metric_stencil_rerun_manifest_wp10c9d6c7c3b5c4f25fizzc5.py";THIS_TEST="tests/test_causal_inner_equilibrium_selected_metric_stencil_rerun_manifest_wp10c9d6c7c3b5c4f25fizzc5.py"
PARENT_SHA="fe5d6d2e3a8b8ef37c2b72e56b949b36438e7a0dedbdaaf92276b441d715e129";CANONICAL_MANIFEST=ROOT/"results/manifests/canonical_artifacts.csv";CANONICAL_SUMMARY=ROOT/"results/manifests/canonical_summary.json"
def _u():return parent._u()
def _validate(clean=False):
 u=_u();
 if u._sha256(parent.CANONICAL_DIRECTORY/"SHA256SUMS.txt")!=PARENT_SHA:raise RuntimeError("diagnosis changed")
 h=u._validate_checksums(parent.CANONICAL_DIRECTORY);s=u._read_json(parent.CANONICAL_DIRECTORY/"summary.json")
 if not s["passed"] or not s["stencil_conditioning_diagnosed"] or s["selected_step_factor"]!=.5 or s["authorized_next"]!=WORK_PACKAGE:raise RuntimeError("selected factor changed")
 if clean and u._git("status","--short","--untracked-files=no"):raise RuntimeError("manifest needs clean tree")
 return h
def _contract():return {"work_package":WORK_PACKAGE,"classification":CLASSIFICATION,"preserved_rejections":True,"selected_step_factor":.5,"selection_was_prospective":True,"same_47_witnesses":True,"unchanged_gates":{"physical_current":1e-10,"thermodynamics":1e-11,"complex_step":1e-9,"sixth_order":2e-5,"density_roundtrip":2e-9},"physical_EOS_and_potential_unchanged":True,"fail_closed":True,"equilibrium_physical_potential_certified":False,"dynamic_height_potential_certified":False,"complete_cycle_execution_authorized":False,"authorized_next":AUTHORIZED_NEXT}
def _update(s):
 u=_u();rows=list(csv.DictReader(CANONICAL_MANIFEST.open(newline="",encoding="utf-8")));rows=[r for r in rows if r.get("case")!=ARTIFACT]
 for p in sorted(CANONICAL_DIRECTORY.iterdir()):
  if p.is_file():rows.append({"case":ARTIFACT,"path":str(p.relative_to(ROOT)),"bytes":str(p.stat().st_size),"sha256":u._sha256(p),"scientific_status":"SUPPORTED"})
 with CANONICAL_MANIFEST.open("w",newline="",encoding="utf-8") as h:w=csv.DictWriter(h,fieldnames=("case","path","bytes","sha256","scientific_status"),lineterminator="\n");w.writeheader();w.writerows(rows)
 c=u._read_json(CANONICAL_SUMMARY);c.setdefault("artifacts",{})[ARTIFACT]={"path":str(CANONICAL_DIRECTORY.relative_to(ROOT)),"classification":CLASSIFICATION,"passed":True};c.update({"case_count":len({r['case'] for r in rows}),"file_count":len(rows),"total_bytes":sum(int(r['bytes']) for r in rows),"all_payload_hashes_recorded":True,"latest_source_parent_commit":u._git("rev-parse","HEAD"),"latest_work_package":WORK_PACKAGE});u._write_json(CANONICAL_SUMMARY,c)
def _freeze():
 if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():raise RuntimeError("exists")
 h=_validate(True);u=_u();CANONICAL_DIRECTORY.mkdir(parents=True);u._write_json(CANONICAL_DIRECTORY/"rerun_contract.json",_contract());s={"work_package":WORK_PACKAGE,"classification":CLASSIFICATION,"passed":True,"definitions_only":True,"selected_step_factor":.5,"equilibrium_physical_potential_certified":False,"complete_cycle_execution_authorized":False,"authorized_next":AUTHORIZED_NEXT};u._write_json(CANONICAL_DIRECTORY/"summary.json",s);u._write_json(CANONICAL_DIRECTORY/"input_lock.json",{"parent_artifact":parent.ARTIFACT,"parent_sha":PARENT_SHA,"parent_hashes":h});REPORT_PATH.parent.mkdir(parents=True,exist_ok=True);REPORT_PATH.write_text(f"# Selected metric-stencil full-rerun manifest\n\nFactor 0.5 is frozen by the prior selection rule. All 47 witnesses and gates are unchanged.\n\nAuthorized next: `{AUTHORIZED_NEXT}`.\n",encoding="utf-8");u._write_json(CANONICAL_DIRECTORY/"provenance.json",{"implementation_commit":u._git("rev-parse","HEAD"),"source_hashes":{p:u._sha256(ROOT/p) for p in (THIS_RUNNER,THIS_TEST,REPORT_RELATIVE)},"python":sys.version,"numpy":np.__version__,"platform":platform.platform(),"thread_environment":{n:os.environ.get(n,"") for n in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS")}});names=sorted(p.name for p in CANONICAL_DIRECTORY.iterdir());(CANONICAL_DIRECTORY/"SHA256SUMS.txt").write_text("".join(f"{u._sha256(CANONICAL_DIRECTORY/n)}  {n}\n" for n in names));_update(s);return s
def main():
 p=argparse.ArgumentParser();p.add_argument("--freeze",action="store_true");a=p.parse_args();
 if not a.freeze:p.error("choose --freeze")
 print(json.dumps(_freeze(),indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
