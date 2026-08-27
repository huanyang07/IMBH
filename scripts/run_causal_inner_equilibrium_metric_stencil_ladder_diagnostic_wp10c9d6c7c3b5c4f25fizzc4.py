#!/usr/bin/env python3
"""Diagnose the saved equilibrium finite-difference witness."""
from __future__ import annotations
import argparse,csv,json,os,platform,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT/"src",ROOT/"scripts"):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
sys.setrecursionlimit(max(sys.getrecursionlimit(),5000))
import run_causal_inner_equilibrium_metric_stencil_ladder_manifest_wp10c9d6c7c3b5c4f25fizzc3 as manifest  # noqa:E402
import run_causal_inner_equilibrium_column_thermodynamic_potential_implementation_wp10c9d6c7c3b5c4f25fizzc as physical  # noqa:E402
import imri_qpe.layer3_minidisk_1d.causal_inner_equilibrium_potential as potential  # noqa:E402
from imri_qpe.layer3_minidisk_1d.causal_inner_eleven_field_convex import full_shear_rest_frame  # noqa:E402
SCHEMA_VERSION=1;WORK_PACKAGE=manifest.AUTHORIZED_NEXT
PASS_CLASSIFICATION="equilibrium_metric_stencil_conditioning_diagnosed";FAIL_CLASSIFICATION="equilibrium_metric_stencil_conditioning_not_diagnosed"
AUTHORIZED_NEXT="definitions_only_WP10c9d6c7c3b5c4f25fizzc5_equilibrium_selected_metric_stencil_rerun_manifest"
ARTIFACT="causal_inner_equilibrium_metric_stencil_ladder_diagnostic_wp10c9d6c7c3b5c4f25fizzc4";CANONICAL_DIRECTORY=ROOT/"results/canonical"/ARTIFACT
REPORT_RELATIVE="docs/reports/current/CODEX_CAUSAL_INNER_EQUILIBRIUM_METRIC_STENCIL_LADDER_DIAGNOSTIC_WP10C9D6C7C3B5C4F25FIZZC4_2026-08-26.md";REPORT_PATH=ROOT/REPORT_RELATIVE
THIS_RUNNER="scripts/run_causal_inner_equilibrium_metric_stencil_ladder_diagnostic_wp10c9d6c7c3b5c4f25fizzc4.py";THIS_TEST="tests/test_causal_inner_equilibrium_metric_stencil_ladder_diagnostic_wp10c9d6c7c3b5c4f25fizzc4.py"
MANIFEST_SHA="7b301e857d0180304d5c26092acdfa4d166bfde3ef70e7d5653b5ac303038d4e";CANONICAL_MANIFEST=ROOT/"results/manifests/canonical_artifacts.csv";CANONICAL_SUMMARY=ROOT/"results/manifests/canonical_summary.json"
def _u():return manifest._utils()
def _validate(require_clean=False):
    u=_u();
    if u._sha256(manifest.CANONICAL_DIRECTORY/"SHA256SUMS.txt")!=MANIFEST_SHA:raise RuntimeError("manifest changed")
    h=u._validate_checksums(manifest.CANONICAL_DIRECTORY);s=u._read_json(manifest.CANONICAL_DIRECTORY/"summary.json");c=u._read_json(manifest.CANONICAL_DIRECTORY/"diagnostic_contract.json")
    if not s["passed"] or s["authorized_next"]!=WORK_PACKAGE or tuple(c["diagnostic"]["step_factors"])!=manifest.FACTORS or c["diagnostic"]["passing_factor_gate"]!=2e-5:raise RuntimeError("diagnostic contract changed")
    if require_clean and u._git("status","--short","--untracked-files=no"):raise RuntimeError("diagnostic needs clean tree")
    return {"hashes":h,"summary":s,"contract":c}
def _runs(passing):
    runs=[];start=None
    for i,value in enumerate(passing+(False,)):
        if value and start is None:start=i
        elif not value and start is not None:runs.append((start,i));start=None
    return runs
def _diagnose():
    began=time.perf_counter();_validate();w=list(physical._physical_witnesses());index,label,radius,old,chart=w[44]
    if index!=44 or label!="empirical_max_field_2":raise RuntimeError("saved witness changed")
    H=float(np.exp(chart[5]));rho=float(np.exp(chart[0]))/(2*H);T=float(np.exp(chart[3]));frame=full_shear_rest_frame(old.geometry,radial_velocity_over_c=float(chart[1]),azimuthal_velocity_over_c=float(chart[2]),vertical_velocity_over_c=0.)
    alpha,beta=potential.entropy_variables_from_primitive(frame.metric,frame.four_velocity,density=rho,temperature=T);state=potential.equilibrium_column_potential_state(frame.metric,alpha,beta,proper_half_thickness=H);analytic=potential.analytic_potential_current_jacobian(state)
    matrices=[];defects=[]
    for factor in manifest.FACTORS:
        fd=potential.finite_difference_potential_current_jacobian(frame.metric,alpha,beta,proper_half_thickness=H,step_factor=factor);matrices.append(fd);defects.append(potential._columnwise_relative_defect(fd,analytic))
    passing=tuple(d<=2e-5 for d in defects);runs=_runs(passing);eligible=[r for r in runs if r[1]-r[0]>=3]
    selected=None
    if eligible:
        width=max(b-a for a,b in eligible);candidates=[]
        for a,b in eligible:
            if b-a==width:
                i=(a+b-1)//2;candidates.append((abs(manifest.FACTORS[i]-1),i))
        selected=manifest.FACTORS[min(candidates)[1]]
    passed=sum(passing)>=3 and selected is not None
    metrics={"schema_version":1,"work_package":WORK_PACKAGE,"classification":PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,"passed":passed,"witness_index":index,"witness_label":label,"witness_radius_cm":radius,"step_factors":manifest.FACTORS,"relative_defects":defects,"passing_factors":[f for f,p in zip(manifest.FACTORS,passing) if p],"contiguous_passing_runs":[[manifest.FACTORS[a],manifest.FACTORS[b-1]] for a,b in runs],"selected_step_factor":selected,"unchanged_gate":2e-5,"complex_step_relative_defect":potential._columnwise_relative_defect(potential.complex_step_potential_current_jacobian(frame.metric,alpha,beta,proper_half_thickness=H),analytic),"original_rejection_preserved":True,"equilibrium_physical_potential_certified":False,"complete_cycle_execution_authorized":False,"wall_seconds":time.perf_counter()-began,"authorized_next":AUTHORIZED_NEXT if passed else None}
    return metrics,{"step_factors":np.asarray(manifest.FACTORS),"finite_difference_jacobians":np.asarray(matrices),"analytic_jacobian":np.asarray(analytic),"witness_chart7":chart,"metric4":frame.metric}
def _update(s):
    u=_u();rows=list(csv.DictReader(CANONICAL_MANIFEST.open(newline="",encoding="utf-8")));rows=[r for r in rows if r.get("case")!=ARTIFACT];status="SUPPORTED" if s["passed"] else "REJECTED"
    for p in sorted(CANONICAL_DIRECTORY.iterdir()):
        if p.is_file():rows.append({"case":ARTIFACT,"path":str(p.relative_to(ROOT)),"bytes":str(p.stat().st_size),"sha256":u._sha256(p),"scientific_status":status})
    with CANONICAL_MANIFEST.open("w",newline="",encoding="utf-8") as h:w=csv.DictWriter(h,fieldnames=("case","path","bytes","sha256","scientific_status"),lineterminator="\n");w.writeheader();w.writerows(rows)
    c=u._read_json(CANONICAL_SUMMARY);c.setdefault("artifacts",{})[ARTIFACT]={"path":str(CANONICAL_DIRECTORY.relative_to(ROOT)),"classification":s["classification"],"passed":s["passed"]};c.update({"case_count":len({r['case'] for r in rows}),"file_count":len(rows),"total_bytes":sum(int(r['bytes']) for r in rows),"all_payload_hashes_recorded":True,"latest_source_parent_commit":u._git("rev-parse","HEAD"),"latest_work_package":WORK_PACKAGE});u._write_json(CANONICAL_SUMMARY,c)
def _canon(m,a):
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():raise RuntimeError("diagnostic exists")
    v=_validate(True);u=_u();CANONICAL_DIRECTORY.mkdir(parents=True);u._write_json(CANONICAL_DIRECTORY/"diagnostic_metrics.json",m);np.savez_compressed(CANONICAL_DIRECTORY/"diagnostic_arrays.npz",**a);s={"schema_version":1,"work_package":WORK_PACKAGE,"classification":m["classification"],"passed":m["passed"],"original_rejection_preserved":True,"stencil_conditioning_diagnosed":m["passed"],"selected_step_factor":m["selected_step_factor"],"equilibrium_physical_potential_certified":False,"complete_cycle_execution_authorized":False,"authorized_next":m["authorized_next"]};u._write_json(CANONICAL_DIRECTORY/"summary.json",s);u._write_json(CANONICAL_DIRECTORY/"input_lock.json",{"manifest_artifact":manifest.ARTIFACT,"manifest_sha":MANIFEST_SHA,"manifest_hashes":v["hashes"]});REPORT_PATH.parent.mkdir(parents=True,exist_ok=True);REPORT_PATH.write_text(f"# Equilibrium metric-stencil ladder diagnostic\n\nClassification: `{m['classification']}`.\n\nFactors: `{m['step_factors']}`. Defects: `{m['relative_defects']}`. Selected factor: `{m['selected_step_factor']}`. The 2e-5 gate is unchanged and the prior rejection is preserved.\n\nAuthorized next: `{m['authorized_next']}`.\n",encoding="utf-8");u._write_json(CANONICAL_DIRECTORY/"provenance.json",{"schema_version":1,"work_package":WORK_PACKAGE,"implementation_commit":u._git("rev-parse","HEAD"),"source_hashes":{p:u._sha256(ROOT/p) for p in (THIS_RUNNER,THIS_TEST,physical.PHYSICAL_SOURCE,REPORT_RELATIVE)},"python":sys.version,"numpy":np.__version__,"platform":platform.platform(),"thread_environment":{n:os.environ.get(n,"") for n in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS")}});names=sorted(p.name for p in CANONICAL_DIRECTORY.iterdir());(CANONICAL_DIRECTORY/"SHA256SUMS.txt").write_text("".join(f"{u._sha256(CANONICAL_DIRECTORY/n)}  {n}\n" for n in names),encoding="utf-8");_update(s);return s
def main():
    p=argparse.ArgumentParser();p.add_argument("--run",action="store_true");x=p.parse_args();
    if not x.run:p.error("choose --run")
    m,a=_diagnose();print(json.dumps(m,indent=2,sort_keys=True));return 0 if _canon(m,a)["passed"] else 2
if __name__=="__main__":raise SystemExit(main())
