#!/usr/bin/env python3
"""Freeze the near-transition fine-complement spatial-decay audit."""
from pathlib import Path
import csv,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT/"src",ROOT/"scripts"):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
import run_causal_inner_fine_complement_exact_jvp_audit_wp10c9d6c7c3b5c4f5 as c4f5  # noqa:E402
SCHEMA_VERSION=1;WORK_PACKAGE="WP10c9d6c7c3b5c4f6";ARTIFACT="causal_inner_fine_complement_decay_manifest_wp10c9d6c7c3b5c4f6";THIS_RUNNER="scripts/run_causal_inner_fine_complement_decay_manifest_wp10c9d6c7c3b5c4f6.py";THIS_TEST="tests/test_causal_inner_fine_complement_decay_manifest_wp10c9d6c7c3b5c4f6.py"
REPORT_RELATIVE="docs/reports/current/CODEX_CAUSAL_INNER_FINE_COMPLEMENT_DECAY_MANIFEST_WP10C9D6C7C3B5C4F6_2026-08-13.md";REPORT_PATH=ROOT/REPORT_RELATIVE;CANONICAL_DIRECTORY=ROOT/"results/canonical"/ARTIFACT;CONFIG_PATH=CANONICAL_DIRECTORY/"config.json";MANIFEST_PATH=CANONICAL_DIRECTORY/"decay_manifest.json";SUMMARY_PATH=CANONICAL_DIRECTORY/"summary.json";PROVENANCE_PATH=CANONICAL_DIRECTORY/"provenance.json";CANONICAL_MANIFEST=ROOT/"results/manifests/canonical_artifacts.csv";CANONICAL_SUMMARY=ROOT/"results/manifests/canonical_summary.json"
FACES=(32,36,40,42,44,45,46,47,48);FD_STEP=1e-3
def _read(p):return json.loads(p.read_text())
def _write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,sort_keys=True)+"\n")
def _sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def _manifest():return {"schema_version":1,"work_package":WORK_PACKAGE,"classification":"fine_complement_decay_manifest_frozen_existing_state_audit_authorized","definitions_only":True,"new_trajectory":False,"parent_faces":FACES,"times_microseconds":tuple(c4f5.c4f4.c4f3.c4f2.TIMES_MICROSECONDS),"centered_difference_step":FD_STEP,"step_certified_by_parent_JVP_audit":True,"gates":{"maximum_JVP_fraction_of_transition":0.10,"minimum_consecutive_recovered_faces":2,"minimum_actual_flux_order":0.75,"minimum_actual_flux_cosine":0.90,"maximum_nonmonotone_regrowth_fraction":0.05,"maximum_ledger_defect":1e-12},"selection":"largest_parent_face_below_48_whose_JVP_is_below_point_one_with_two_consecutive_inward_passes_and_whose_actual_flux_passes_spatial_gates","selected_face_is_only_a_recovery_surface_candidate":True,"response_certificate_preserved":True,"absolute_closure_fit_authorized":False,"memory_propagation_authorized":False,"fixed_Q_authorized":False,"reduced_slow_evolution_authorized":False,"authorized_next":"WP10c9d6c7c3b5c4f7_fine_complement_spatial_decay_audit"}
def _catalog(s):
 rows=[]
 if CANONICAL_MANIFEST.exists():
  with CANONICAL_MANIFEST.open(newline='') as h:rows=list(csv.DictReader(h))
 rows=[r for r in rows if r.get('case')!=ARTIFACT]
 for p in sorted(CANONICAL_DIRECTORY.iterdir()):
  if p.is_file():rows.append({'case':ARTIFACT,'path':str(p.relative_to(ROOT)),'bytes':str(p.stat().st_size),'sha256':_sha(p),'scientific_status':'PROSPECTIVE'})
 with CANONICAL_MANIFEST.open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=['case','path','bytes','sha256','scientific_status'],lineterminator='\n');w.writeheader();w.writerows(rows)
 c=_read(CANONICAL_SUMMARY);c.setdefault('artifacts',{})[ARTIFACT]={'path':str(CANONICAL_DIRECTORY.relative_to(ROOT)),'classification':s['classification'],'passed':True};c.update({'case_count':len({r['case'] for r in rows}),'file_count':len(rows),'total_bytes':sum(int(r['bytes']) for r in rows),'latest_work_package':WORK_PACKAGE});_write(CANONICAL_SUMMARY,c)
def main():
 p=_read(c4f5.SUMMARY_PATH)
 if p['authorized_next']!='distributed_fine_complement_localization_manifest':raise RuntimeError('c4f6 authorization changed')
 m=_manifest();s={'schema_version':1,'work_package':WORK_PACKAGE,'classification':m['classification'],'passed':True,'definitions_only':True,'parent_result_preserved':True,'new_trajectory_authorized':False,'memory_propagation_authorized':False,'fixed_Q_micro_solver_authorized':False,'reduced_slow_evolution_authorized':False,'physical_failure_detected':False,'authorized_next':m['authorized_next']};CANONICAL_DIRECTORY.mkdir(parents=True,exist_ok=True);_write(CONFIG_PATH,{'faces':FACES,'FD_step':FD_STEP});_write(MANIFEST_PATH,m);_write(SUMMARY_PATH,s);REPORT_PATH.parent.mkdir(parents=True,exist_ok=True);REPORT_PATH.write_text(f"# Fine-complement spatial-decay manifest\n\nClassification: `{s['classification']}`.\n\nNo trajectory is authorized. The audit may identify only a recovery-surface candidate; a separate control-volume contract remains necessary.\n");_write(PROVENANCE_PATH,{'parent_summary_sha256':_sha(c4f5.SUMMARY_PATH),'output_hashes':{}});(CANONICAL_DIRECTORY/'SHA256SUMS.txt').write_text(''.join(f"{_sha(x)}  {x.name}\n" for x in (CONFIG_PATH,MANIFEST_PATH,SUMMARY_PATH,PROVENANCE_PATH)));_catalog(s);print(json.dumps(s,indent=2))
if __name__=='__main__':main()
