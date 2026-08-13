#!/usr/bin/env python3
"""Freeze the fine-complement analytic JVP audit."""

from __future__ import annotations

import csv, hashlib, json, math, platform, subprocess, sys
from pathlib import Path
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path: sys.path.insert(0, str(path))
import run_causal_inner_selected_time_absolute_coupling_localization_wp10c9d6c7c3b5c4f3 as c4f3  # noqa: E402

SCHEMA_VERSION=1; WORK_PACKAGE="WP10c9d6c7c3b5c4f4"; ANALYZED_CERTIFICATE_COMMIT=c4f3.ANALYZED_CERTIFICATE_COMMIT
ARTIFACT="causal_inner_fine_complement_jvp_manifest_wp10c9d6c7c3b5c4f4"
THIS_RUNNER="scripts/run_causal_inner_fine_complement_jvp_manifest_wp10c9d6c7c3b5c4f4.py"; THIS_TEST="tests/test_causal_inner_fine_complement_jvp_manifest_wp10c9d6c7c3b5c4f4.py"
REPORT_RELATIVE="docs/reports/current/CODEX_CAUSAL_INNER_FINE_COMPLEMENT_JVP_MANIFEST_WP10C9D6C7C3B5C4F4_2026-08-13.md"; REPORT_PATH=ROOT/REPORT_RELATIVE
CANONICAL_DIRECTORY=ROOT/"results/canonical"/ARTIFACT; CONFIG_PATH=CANONICAL_DIRECTORY/"config.json"; MANIFEST_PATH=CANONICAL_DIRECTORY/"jvp_manifest.json"; SUMMARY_PATH=CANONICAL_DIRECTORY/"summary.json"; PROVENANCE_PATH=CANONICAL_DIRECTORY/"provenance.json"
CANONICAL_MANIFEST=ROOT/"results/manifests/canonical_artifacts.csv"; CANONICAL_SUMMARY=ROOT/"results/manifests/canonical_summary.json"
FD_TIMES_MICROSECONDS=(5000,20000); FD_STEPS=(5.0e-4,1.0e-3,2.0e-3); FACES=(46,47,48)

def _plain(v):
    if isinstance(v,dict): return {str(k):_plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [_plain(x) for x in v]
    if isinstance(v,np.ndarray): return _plain(v.tolist())
    if isinstance(v,(np.bool_,bool)): return bool(v)
    if isinstance(v,(np.floating,float)): v=float(v); return v if math.isfinite(v) else None
    if isinstance(v,(np.integer,int)): return int(v)
    return v
def _read(p): return json.loads(p.read_text(encoding="utf-8"))
def _write(p,v): p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(_plain(v),indent=2,sort_keys=True)+"\n",encoding="utf-8")
def _sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def _git(*a): return subprocess.run(("git",*a),cwd=ROOT,check=True,capture_output=True,text=True).stdout.strip()
def _manifest():
    return {"schema_version":SCHEMA_VERSION,"work_package":WORK_PACKAGE,"classification":"fine_complement_JVP_manifest_frozen_analysis_authorized","definitions_only":True,"new_trajectory":False,"layout":"fine","times_microseconds":tuple(c4f3.c4f2.TIMES_MICROSECONDS),"faces":FACES,"direction":"native_fine_state_minus_repeated_conservative_parent_average","direction_is_diagnostic_not_a_physical_lift":True,"analytic_kernel":"forward_AD_radial_shared_face_flux_jacobian","finite_difference_audit":{"times_microseconds":FD_TIMES_MICROSECONDS,"relative_steps":FD_STEPS,"maximum_analytic_FD_relative_defect":1.0e-6,"maximum_step_plateau_relative_change":1.0e-4},"gates":{"maximum_JVP_fraction_of_actual_middle_fine_transition_difference":0.10,"maximum_adjacent_face_JVP_fraction_of_transition_JVP":0.10,"maximum_characteristic_imaginary_part":1.0e-10,"maximum_projector_closure_defect":1.0e-10},"decision":{"transition_JVP_above_point_one_and_adjacent_below_point_one":"coupling_recovery_surface_manifest_authorized","transition_JVP_below_point_one":"fine_anchored_absolute_baseline_manifest_authorized","adjacent_JVP_also_large":"distributed_fine_complement_localization_required","JVP_audit_failure":"method_repair_only"},"response_certificate_preserved":True,"absolute_closure_fit_authorized":False,"observable_memory_propagation_authorized":False,"fixed_Q_authorized":False,"reduced_slow_evolution_authorized":False,"hard_stops":("do_not_promote_the_repeated_parent_state","do_not_change_operator_or_run_trajectory","do_not_relax_failed_absolute_gate","do_not_start_memory_or_fixed_Q"),"authorized_next":"WP10c9d6c7c3b5c4f5_fine_complement_exact_JVP_audit"}
def _catalog(s):
    rows=[]
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="",encoding="utf-8") as h: rows=list(csv.DictReader(h))
    rows=[r for r in rows if r.get("case")!=ARTIFACT]
    for p in sorted(CANONICAL_DIRECTORY.iterdir()):
        if p.is_file(): rows.append({"case":ARTIFACT,"path":str(p.relative_to(ROOT)),"bytes":str(p.stat().st_size),"sha256":_sha(p),"scientific_status":"PROSPECTIVE"})
    with CANONICAL_MANIFEST.open("w",newline="",encoding="utf-8") as h: w=csv.DictWriter(h,fieldnames=["case","path","bytes","sha256","scientific_status"],lineterminator="\n");w.writeheader();w.writerows(rows)
    c=_read(CANONICAL_SUMMARY);c.setdefault("artifacts",{})[ARTIFACT]={"path":str(CANONICAL_DIRECTORY.relative_to(ROOT)),"classification":s["classification"],"passed":True};c.update({"case_count":len({r['case'] for r in rows}),"file_count":len(rows),"total_bytes":sum(int(r['bytes']) for r in rows),"latest_source_parent_commit":ANALYZED_CERTIFICATE_COMMIT,"latest_work_package":WORK_PACKAGE});_write(CANONICAL_SUMMARY,c)
def main():
    p=_read(c4f3.SUMMARY_PATH)
    if not p["passed"] or p["authorized_next"]!="definitions_only_fine_complement_exact_JVP_manifest": raise RuntimeError("c4f4 authorization changed")
    m=_manifest();s={"schema_version":SCHEMA_VERSION,"work_package":WORK_PACKAGE,"classification":m["classification"],"passed":True,"definitions_only":True,"parent_localization_preserved":True,"new_trajectory_authorized":False,"observable_memory_propagation_authorized":False,"fixed_Q_micro_solver_authorized":False,"reduced_slow_evolution_authorized":False,"physical_failure_detected":False,"authorized_next":m["authorized_next"]}
    CANONICAL_DIRECTORY.mkdir(parents=True,exist_ok=True);_write(CONFIG_PATH,{"schema_version":1,"times_microseconds":m["times_microseconds"],"faces":FACES,"FD_steps":FD_STEPS});_write(MANIFEST_PATH,m);_write(SUMMARY_PATH,s)
    REPORT_PATH.parent.mkdir(parents=True,exist_ok=True);REPORT_PATH.write_text(f"# Fine-complement JVP manifest\n\nClassification: `{s['classification']}`.\n\nThis definitions-only package freezes four analytic fine-grid face-flux JVPs and two small-step finite-difference plateaus. It runs no trajectory and does not promote the repeated-parent state.\n",encoding="utf-8")
    q={"schema_version":1,"execution_head":_git("rev-parse","HEAD"),"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"scipy":scipy.__version__,"parent_summary_sha256":_sha(c4f3.SUMMARY_PATH),"output_hashes":{}};q["output_hashes"]={str(x.relative_to(ROOT)):_sha(x) for x in (CONFIG_PATH,MANIFEST_PATH,SUMMARY_PATH,REPORT_PATH)};_write(PROVENANCE_PATH,q)
    (CANONICAL_DIRECTORY/"SHA256SUMS.txt").write_text("".join(f"{_sha(x)}  {x.name}\n" for x in (CONFIG_PATH,MANIFEST_PATH,SUMMARY_PATH,PROVENANCE_PATH)),encoding="utf-8");_catalog(s);print(json.dumps(s,indent=2,sort_keys=True))
if __name__=="__main__": main()
