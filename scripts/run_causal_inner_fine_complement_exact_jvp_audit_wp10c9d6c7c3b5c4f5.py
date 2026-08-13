#!/usr/bin/env python3
"""Execute the authorized fine-complement shared-face JVP audit."""
from __future__ import annotations
import csv,hashlib,json,math,os,platform,subprocess,sys,time
from pathlib import Path
import numpy as np
import scipy
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT/"src",ROOT/"scripts"):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
import run_causal_inner_fine_complement_jvp_manifest_wp10c9d6c7c3b5c4f4 as c4f4  # noqa:E402
from imri_qpe.layer3_minidisk_1d.causal_inner_radial_linear_tangent import causal_five_field_radial_analytic_tangent  # noqa:E402

c4f3=c4f4.c4f3;c4f1=c4f3.c4f1
SCHEMA_VERSION=1;WORK_PACKAGE="WP10c9d6c7c3b5c4f5";ANALYZED_CERTIFICATE_COMMIT=c4f4.ANALYZED_CERTIFICATE_COMMIT
ARTIFACT="causal_inner_fine_complement_exact_jvp_audit_wp10c9d6c7c3b5c4f5";THIS_RUNNER="scripts/run_causal_inner_fine_complement_exact_jvp_audit_wp10c9d6c7c3b5c4f5.py";THIS_TEST="tests/test_causal_inner_fine_complement_exact_jvp_audit_wp10c9d6c7c3b5c4f5.py"
REPORT_RELATIVE="docs/reports/current/CODEX_CAUSAL_INNER_FINE_COMPLEMENT_EXACT_JVP_AUDIT_WP10C9D6C7C3B5C4F5_2026-08-13.md";REPORT_PATH=ROOT/REPORT_RELATIVE
CANONICAL_DIRECTORY=ROOT/"results/canonical"/ARTIFACT;CONFIG_PATH=CANONICAL_DIRECTORY/"config.json";CONTRACT_PATH=CANONICAL_DIRECTORY/"analysis_contract.json";SUMMARY_PATH=CANONICAL_DIRECTORY/"summary.json";PROVENANCE_PATH=CANONICAL_DIRECTORY/"provenance.json";DECISIVE_ARRAYS=CANONICAL_DIRECTORY/"decisive_arrays.npz"
CANONICAL_MANIFEST=ROOT/"results/manifests/canonical_artifacts.csv";CANONICAL_SUMMARY=ROOT/"results/manifests/canonical_summary.json";CHECKPOINT=ROOT/"outputs/checkpoints"/ARTIFACT/"jvp.npz";PROGRESS=ROOT/"outputs/checkpoints"/ARTIFACT/"progress.json"
FACES=np.asarray(c4f4.FACES);FIELDS=c4f3.FIELDS;TIMES=c4f3.TIMES
def _plain(v):
    if isinstance(v,dict):return {str(k):_plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [_plain(x) for x in v]
    if isinstance(v,np.ndarray):return _plain(v.tolist())
    if isinstance(v,(np.bool_,bool)):return bool(v)
    if isinstance(v,(np.floating,float)):v=float(v);return v if math.isfinite(v) else None
    if isinstance(v,(np.integer,int)):return int(v)
    return v
def _read(p):return json.loads(p.read_text(encoding="utf-8"))
def _write(p,v):p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+".tmp");t.write_text(json.dumps(_plain(v),indent=2,sort_keys=True)+"\n",encoding="utf-8");os.replace(t,p)
def _load(p):
    with np.load(p,allow_pickle=False) as z:return {k:np.asarray(z[k]) for k in z.files}
def _save(p,**a):p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(".tmp.npz");np.savez_compressed(t,**a);os.replace(t,p)
def _sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def _git(*a):return subprocess.run(("git",*a),cwd=ROOT,check=True,capture_output=True,text=True).stdout.strip()
def _validate():
    s=_read(c4f4.SUMMARY_PATH);m=_read(c4f4.MANIFEST_PATH)
    if not s["passed"] or s["authorized_next"]!="WP10c9d6c7c3b5c4f5_fine_complement_exact_JVP_audit":raise RuntimeError("c4f5 authorization changed")
    return m
def _evaluate(m):
    _,configs=c4f1._configurations();layout,conf=configs["fine"];trajectory=c4f1._fine_trajectory();selected=trajectory["states"][c4f1._indices(trajectory["times"],TIMES)];parent=c4f1.c4f.c4e12.c4e9.c4e4._restrict(selected,layout);lift=parent[:,layout.parent_cell_indices];directions=selected-lift;columns=np.asarray(conf["columns"]);rows=np.asarray(conf["rows"]);face_indices=FACES*int(layout.refinement_ratio)
    if CHECKPOINT.exists() and PROGRESS.exists():arrays=_load(CHECKPOINT);progress=_read(PROGRESS)
    else:
        arrays={"analytic_JVP":np.full((TIMES.size,FACES.size,FIELDS.size),np.nan),"FD_JVP":np.full((TIMES.size,len(c4f4.FD_STEPS),FACES.size,FIELDS.size),np.nan),"matrix_wall_seconds":np.full(TIMES.size,np.nan),"maximum_imaginary_part":np.full(TIMES.size,np.nan),"maximum_projector_closure_defect":np.full(TIMES.size,np.nan),"directions":directions,"lifted_states":lift};progress={"completed":[]}
    done=set(progress["completed"]);fd_times=set(c4f4.FD_TIMES_MICROSECONDS)
    for i,tv in enumerate(TIMES):
        key=str(int(round(tv*1e6)))
        if key in done:continue
        began=time.perf_counter();tangent=causal_five_field_radial_analytic_tangent(conf["context"],selected[i],primitive_column_scales=columns,conservation_row_scales=rows,path_quadrature_order=6,center_broken_within_cell_paths=True);arrays["matrix_wall_seconds"][i]=time.perf_counter()-began
        scaled=directions[i].ravel()/columns;maps=np.asarray(tangent.shared_face_flux_scaled_jacobians)[face_indices][:,FIELDS];arrays["analytic_JVP"][i]=np.einsum("fkd,d->fk",maps,scaled)
        arrays["maximum_imaginary_part"][i]=tangent.maximum_characteristic_imaginary_part;arrays["maximum_projector_closure_defect"][i]=tangent.maximum_projector_closure_defect
        if int(round(tv*1e6)) in fd_times:
            for j,alpha in enumerate(c4f4.FD_STEPS):
                plus,_=c4f3._face_fluxes(conf["context"],selected[i]+alpha*directions[i],int(layout.refinement_ratio));minus,_=c4f3._face_fluxes(conf["context"],selected[i]-alpha*directions[i],int(layout.refinement_ratio));positions=[int(np.flatnonzero(c4f3.FACES==f)[0]) for f in FACES];arrays["FD_JVP"][i,j]=(plus[positions]-minus[positions])/(2*alpha)
        done.add(key);progress["completed"]=sorted(done);_save(CHECKPOINT,**arrays);_write(PROGRESS,progress);print(f"c4f5: t={tv:.6f}s matrix={arrays['matrix_wall_seconds'][i]:.1f}s",flush=True)
    arrays.update({"times_seconds":TIMES,"faces":FACES,"fine_states":selected});return arrays
def _analyze(a,m):
    scales=_load(c4f1.c4f.MIDDLE_ARRAYS)["tangent__export_scales"][:3];face48=int(np.flatnonzero(FACES==48)[0]);analytic=a["analytic_JVP"];parent=_load(c4f3.DECISIVE_ARRAYS);p48=int(np.flatnonzero(parent["parent_face_indices"]==48)[0]);actual_mf=parent["actual_fluxes"][2,:,p48]-parent["actual_fluxes"][1,:,p48]
    fraction=float(np.linalg.norm(analytic[:,face48]/scales)/max(np.linalg.norm(actual_mf/scales),np.finfo(float).tiny));adjacent=float(max(np.linalg.norm(analytic[:,j]/scales) for j,f in enumerate(FACES) if f!=48)/max(np.linalg.norm(analytic[:,face48]/scales),np.finfo(float).tiny))
    fd_indices=[i for i,t in enumerate(TIMES) if int(round(t*1e6)) in set(c4f4.FD_TIMES_MICROSECONDS)];fd=a["FD_JVP"][fd_indices];an=analytic[fd_indices];defect=float(np.linalg.norm(fd-an[:,None])/max(np.linalg.norm(fd),np.linalg.norm(np.repeat(an[:,None],len(c4f4.FD_STEPS),axis=1)),np.finfo(float).tiny));plateau=float(np.linalg.norm(fd[:,-1]-fd[:,0])/max(np.linalg.norm(fd[:,1]),np.finfo(float).tiny));g=m["gates"];fda=m["finite_difference_audit"]
    method=bool(defect<=fda["maximum_analytic_FD_relative_defect"] and plateau<=fda["maximum_step_plateau_relative_change"] and np.max(a["maximum_imaginary_part"])<=g["maximum_characteristic_imaginary_part"] and np.max(a["maximum_projector_closure_defect"])<=g["maximum_projector_closure_defect"])
    if not method:classification="fine_complement_JVP_method_gate_failed";next_="JVP_method_repair_only";passed=False
    elif fraction>g["maximum_JVP_fraction_of_actual_middle_fine_transition_difference"] and adjacent<=g["maximum_adjacent_face_JVP_fraction_of_transition_JVP"]:classification="fine_complement_observable_only_at_transition_recovery_surface_manifest_authorized";next_="definitions_only_coupling_recovery_surface_manifest";passed=True
    elif fraction<=g["maximum_JVP_fraction_of_actual_middle_fine_transition_difference"]:classification="fine_complement_linear_effect_small_fine_anchored_baseline_manifest_authorized";next_="definitions_only_fine_anchored_absolute_baseline_manifest";passed=True
    else:classification="fine_complement_observable_across_adjacent_faces_localization_required";next_="distributed_fine_complement_localization_manifest";passed=False
    return {"schema_version":1,"work_package":WORK_PACKAGE,"classification":classification,"passed":passed,"method_gates_passed":method,"analytic_FD_relative_defect":defect,"FD_step_plateau_relative_change":plateau,"transition_JVP_fraction_of_actual_middle_fine_difference":fraction,"maximum_adjacent_face_fraction_of_transition_JVP":adjacent,"maximum_characteristic_imaginary_part":float(np.max(a["maximum_imaginary_part"])),"maximum_projector_closure_defect":float(np.max(a["maximum_projector_closure_defect"])),"physical_failure_detected":False,"response_certificate_preserved":True,"absolute_closure_fit_authorized":False,"observable_memory_propagation_authorized":False,"fixed_Q_micro_solver_authorized":False,"reduced_slow_evolution_authorized":False,"authorized_next":next_}
def _catalog(s):
    rows=[]
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="",encoding="utf-8") as h:rows=list(csv.DictReader(h))
    rows=[r for r in rows if r.get("case")!=ARTIFACT]
    for p in sorted(CANONICAL_DIRECTORY.iterdir()):
        if p.is_file():rows.append({"case":ARTIFACT,"path":str(p.relative_to(ROOT)),"bytes":str(p.stat().st_size),"sha256":_sha(p),"scientific_status":"CERTIFIED" if s["passed"] else "REJECTED"})
    with CANONICAL_MANIFEST.open("w",newline="",encoding="utf-8") as h:w=csv.DictWriter(h,fieldnames=["case","path","bytes","sha256","scientific_status"],lineterminator="\n");w.writeheader();w.writerows(rows)
    c=_read(CANONICAL_SUMMARY);c.setdefault("artifacts",{})[ARTIFACT]={"path":str(CANONICAL_DIRECTORY.relative_to(ROOT)),"classification":s["classification"],"passed":s["passed"]};c.update({"case_count":len({r['case'] for r in rows}),"file_count":len(rows),"total_bytes":sum(int(r['bytes']) for r in rows),"latest_source_parent_commit":ANALYZED_CERTIFICATE_COMMIT,"latest_work_package":WORK_PACKAGE});_write(CANONICAL_SUMMARY,c)
def _finalize(a,s,m):
    CANONICAL_DIRECTORY.mkdir(parents=True,exist_ok=True);_save(DECISIVE_ARRAYS,**a);_write(CONFIG_PATH,{"schema_version":1,"times_seconds":TIMES,"faces":FACES,"FD_steps":c4f4.FD_STEPS});_write(CONTRACT_PATH,m);_write(SUMMARY_PATH,s);REPORT_PATH.parent.mkdir(parents=True,exist_ok=True);REPORT_PATH.write_text(f"# Fine-complement exact JVP audit\n\nClassification: `{s['classification']}`.\n\nAnalytic/FD defect: `{s['analytic_FD_relative_defect']:.6e}`; step plateau change: `{s['FD_step_plateau_relative_change']:.6e}`.\n\nTransition JVP divided by the actual middle-fine transition difference: `{s['transition_JVP_fraction_of_actual_middle_fine_difference']:.6e}`. Maximum adjacent-face fraction of the transition JVP: `{s['maximum_adjacent_face_fraction_of_transition_JVP']:.6e}`.\n\nNo trajectory, fixed-Q solve, or memory propagation ran.\n",encoding="utf-8")
    p={"schema_version":1,"execution_head":_git("rev-parse","HEAD"),"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"scipy":scipy.__version__,"input_hashes":{str(x.relative_to(ROOT)):_sha(x) for x in (c4f3.DECISIVE_ARRAYS,c4f4.MANIFEST_PATH)},"output_hashes":{}};p["output_hashes"]={str(x.relative_to(ROOT)):_sha(x) for x in (CONFIG_PATH,CONTRACT_PATH,SUMMARY_PATH,DECISIVE_ARRAYS,REPORT_PATH)};_write(PROVENANCE_PATH,p);(CANONICAL_DIRECTORY/"SHA256SUMS.txt").write_text("".join(f"{_sha(x)}  {x.name}\n" for x in (CONFIG_PATH,CONTRACT_PATH,SUMMARY_PATH,PROVENANCE_PATH,DECISIVE_ARRAYS)),encoding="utf-8");_catalog(s)
def main():m=_validate();a=_evaluate(m);s=_analyze(a,m);_finalize(a,s,m);print(json.dumps(_plain(s),indent=2,sort_keys=True))
if __name__=="__main__":main()
