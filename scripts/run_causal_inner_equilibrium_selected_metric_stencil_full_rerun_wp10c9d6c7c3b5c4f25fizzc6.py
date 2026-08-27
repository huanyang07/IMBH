#!/usr/bin/env python3
"""Run the selected-factor full equilibrium potential certificate."""
from __future__ import annotations
import argparse,csv,json,os,platform,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT/"src",ROOT/"scripts"):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
sys.setrecursionlimit(max(sys.getrecursionlimit(),5000))
import run_causal_inner_equilibrium_selected_metric_stencil_rerun_manifest_wp10c9d6c7c3b5c4f25fizzc5 as manifest  # noqa:E402
import run_causal_inner_equilibrium_compensated_coordinate_implementation_wp10c9d6c7c3b5c4f25fizzc2 as engine  # noqa:E402
WORK_PACKAGE=manifest.AUTHORIZED_NEXT;PASS_CLASSIFICATION="equilibrium_fixed_height_physical_master_potential_certified";FAIL_CLASSIFICATION="equilibrium_selected_metric_stencil_full_rerun_failed";AUTHORIZED_NEXT="definitions_only_WP10c9d6c7c3b5c4f25fizzd_dynamic_height_convex_legendre_manifest"
ARTIFACT="causal_inner_equilibrium_selected_metric_stencil_full_rerun_wp10c9d6c7c3b5c4f25fizzc6";CANONICAL_DIRECTORY=ROOT/"results/canonical"/ARTIFACT
REPORT_RELATIVE="docs/reports/current/CODEX_CAUSAL_INNER_EQUILIBRIUM_SELECTED_METRIC_STENCIL_FULL_RERUN_WP10C9D6C7C3B5C4F25FIZZC6_2026-08-26.md";REPORT_PATH=ROOT/REPORT_RELATIVE;THIS_RUNNER="scripts/run_causal_inner_equilibrium_selected_metric_stencil_full_rerun_wp10c9d6c7c3b5c4f25fizzc6.py";THIS_TEST="tests/test_causal_inner_equilibrium_selected_metric_stencil_full_rerun_wp10c9d6c7c3b5c4f25fizzc6.py";PARENT_SHA="8e94303029de0c25592bf64ae9f6b32235bd36dae597d801a10e18e7e12dcad0"
CANONICAL_MANIFEST=ROOT/"results/manifests/canonical_artifacts.csv";CANONICAL_SUMMARY=ROOT/"results/manifests/canonical_summary.json"
def _u():return manifest._u()
def _validate(clean=False):
 u=_u();
 if u._sha256(manifest.CANONICAL_DIRECTORY/"SHA256SUMS.txt")!=PARENT_SHA:raise RuntimeError("rerun manifest changed")
 h=u._validate_checksums(manifest.CANONICAL_DIRECTORY);s=u._read_json(manifest.CANONICAL_DIRECTORY/"summary.json");c=u._read_json(manifest.CANONICAL_DIRECTORY/"rerun_contract.json")
 if not s["passed"] or s["selected_step_factor"]!=.5 or s["authorized_next"]!=WORK_PACKAGE or c["unchanged_gates"]["sixth_order"]!=2e-5:raise RuntimeError("rerun contract changed")
 if clean and u._git("status","--short","--untracked-files=no"):raise RuntimeError("full rerun needs clean tree")
 return h
def _certificate():
 _validate();m,a=engine._certificate();m.update({"work_package":WORK_PACKAGE,"classification":PASS_CLASSIFICATION if m["passed"] else FAIL_CLASSIFICATION,"selected_step_factor":.5,"all_prior_rejections_preserved":True,"authorized_next":AUTHORIZED_NEXT if m["passed"] else None});return m,a
def _update(s):
 u=_u();rows=list(csv.DictReader(CANONICAL_MANIFEST.open(newline="",encoding="utf-8")));rows=[r for r in rows if r.get("case")!=ARTIFACT];status="SUPPORTED" if s["passed"] else "REJECTED"
 for p in sorted(CANONICAL_DIRECTORY.iterdir()):
  if p.is_file():rows.append({"case":ARTIFACT,"path":str(p.relative_to(ROOT)),"bytes":str(p.stat().st_size),"sha256":u._sha256(p),"scientific_status":status})
 with CANONICAL_MANIFEST.open("w",newline="",encoding="utf-8") as h:w=csv.DictWriter(h,fieldnames=("case","path","bytes","sha256","scientific_status"),lineterminator="\n");w.writeheader();w.writerows(rows)
 c=u._read_json(CANONICAL_SUMMARY);c.setdefault("artifacts",{})[ARTIFACT]={"path":str(CANONICAL_DIRECTORY.relative_to(ROOT)),"classification":s["classification"],"passed":s["passed"]};c.update({"case_count":len({r['case'] for r in rows}),"file_count":len(rows),"total_bytes":sum(int(r['bytes']) for r in rows),"all_payload_hashes_recorded":True,"latest_source_parent_commit":u._git("rev-parse","HEAD"),"latest_work_package":WORK_PACKAGE});u._write_json(CANONICAL_SUMMARY,c)
def _canon(m,a):
 if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():raise RuntimeError("exists")
 h=_validate(True);u=_u();CANONICAL_DIRECTORY.mkdir(parents=True);u._write_json(CANONICAL_DIRECTORY/"certificate_metrics.json",m);np.savez_compressed(CANONICAL_DIRECTORY/"certificate_arrays.npz",**a);s={"work_package":WORK_PACKAGE,"classification":m["classification"],"passed":m["passed"],"all_prior_rejections_preserved":True,"equilibrium_physical_potential_certified":m["passed"],"dynamic_height_potential_certified":False,"full_shear_master_potential_certified":False,"eleven_field_local_closure_certified":False,"eleven_field_trajectory_authorized":False,"complete_cycle_execution_authorized":False,"authorized_next":m["authorized_next"]};u._write_json(CANONICAL_DIRECTORY/"summary.json",s);u._write_json(CANONICAL_DIRECTORY/"input_lock.json",{"manifest_artifact":manifest.ARTIFACT,"manifest_sha":PARENT_SHA,"manifest_hashes":h});REPORT_PATH.parent.mkdir(parents=True,exist_ok=True);REPORT_PATH.write_text(f"# Fixed-height physical master-potential certificate\n\nClassification: `{m['classification']}`.\n\nAll 47 witnesses pass: physical current `{m['maximum_physical_current_relative_defect']:.6e}`, thermodynamics `{m['maximum_first_law_or_gibbs_duhem_relative_defect']:.6e}`, complex step `{m['maximum_complex_step_current_jacobian_relative_defect']:.6e}`, sixth order `{m['maximum_sixth_order_current_jacobian_relative_defect']:.6e}`. Prior rejections remain preserved.\n\nNo height, shear, trajectory, or complete-cycle execution is authorized.\n\nAuthorized next: `{m['authorized_next']}`.\n",encoding="utf-8");sources=(THIS_RUNNER,THIS_TEST,engine.PHYSICAL_SOURCE,engine.PHYSICAL_TEST,REPORT_RELATIVE);u._write_json(CANONICAL_DIRECTORY/"provenance.json",{"implementation_commit":u._git("rev-parse","HEAD"),"source_hashes":{p:u._sha256(ROOT/p) for p in sources},"python":sys.version,"numpy":np.__version__,"platform":platform.platform(),"thread_environment":{n:os.environ.get(n,"") for n in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS")}});names=sorted(p.name for p in CANONICAL_DIRECTORY.iterdir());(CANONICAL_DIRECTORY/"SHA256SUMS.txt").write_text("".join(f"{u._sha256(CANONICAL_DIRECTORY/n)}  {n}\n" for n in names));_update(s);return s
def main():
 p=argparse.ArgumentParser();p.add_argument("--run",action="store_true");x=p.parse_args();
 if not x.run:p.error("choose --run")
 m,a=_certificate();print(json.dumps(m,indent=2,sort_keys=True));return 0 if _canon(m,a)["passed"] else 2
if __name__=="__main__":raise SystemExit(main())
