#!/usr/bin/env python3
"""Measure the spatial decay of the certified fine-complement flux JVP."""
from __future__ import annotations
import csv,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT/"src",ROOT/"scripts"):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
import run_causal_inner_fine_complement_decay_manifest_wp10c9d6c7c3b5c4f6 as c4f6  # noqa:E402
c4f5=c4f6.c4f5;c4f3=c4f5.c4f4.c4f3;c4f1=c4f3.c4f1
SCHEMA_VERSION=1;WORK_PACKAGE="WP10c9d6c7c3b5c4f7";ARTIFACT="causal_inner_fine_complement_spatial_decay_audit_wp10c9d6c7c3b5c4f7";THIS_RUNNER="scripts/run_causal_inner_fine_complement_spatial_decay_audit_wp10c9d6c7c3b5c4f7.py";THIS_TEST="tests/test_causal_inner_fine_complement_spatial_decay_audit_wp10c9d6c7c3b5c4f7.py"
REPORT_RELATIVE="docs/reports/current/CODEX_CAUSAL_INNER_FINE_COMPLEMENT_SPATIAL_DECAY_AUDIT_WP10C9D6C7C3B5C4F7_2026-08-13.md";REPORT_PATH=ROOT/REPORT_RELATIVE;CANONICAL_DIRECTORY=ROOT/"results/canonical"/ARTIFACT;CONFIG_PATH=CANONICAL_DIRECTORY/"config.json";CONTRACT_PATH=CANONICAL_DIRECTORY/"analysis_contract.json";SUMMARY_PATH=CANONICAL_DIRECTORY/"summary.json";PROVENANCE_PATH=CANONICAL_DIRECTORY/"provenance.json";DECISIVE_ARRAYS=CANONICAL_DIRECTORY/"decisive_arrays.npz";CANONICAL_MANIFEST=ROOT/"results/manifests/canonical_artifacts.csv";CANONICAL_SUMMARY=ROOT/"results/manifests/canonical_summary.json";CHECKPOINT=ROOT/"outputs/checkpoints"/ARTIFACT/"values.npz";PROGRESS=ROOT/"outputs/checkpoints"/ARTIFACT/"progress.json"
FACES=np.asarray(c4f6.FACES);TIMES=c4f3.TIMES;FIELDS=c4f3.FIELDS
def _read(p):return json.loads(p.read_text())
def _write(p,v):p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');os.replace(t,p)
def _load(p):
 with np.load(p,allow_pickle=False) as z:return {k:np.asarray(z[k]) for k in z.files}
def _save(p,**a):p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp.npz');np.savez_compressed(t,**a);os.replace(t,p)
def _sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def _flux(context,state,multiplier):
 ledger=c4f3.causal_five_field_radial_candidate_ledger(context,state);idx=FACES*int(multiplier);return np.asarray(ledger.interfaces.candidate_shared_face_fluxes_over_c)[idx][:,FIELDS],np.asarray((ledger.interfaces.shared_conservative_face_defect,ledger.local_block_ledger_defect,ledger.source_double_count_defect,ledger.interfaces.incoming_excision_characteristics))
def _evaluate(m):
 _,cfg=c4f1._configurations();selected=c4f3._selected_states();fl,fc=cfg['fine'];fp=c4f1.c4f.c4e12.c4e9.c4e4._restrict(selected['fine'],fl);lift=fp[:,fl.parent_cell_indices];direction=selected['fine']-lift
 shape=(3,TIMES.size,FACES.size,3)
 if CHECKPOINT.exists():a=_load(CHECKPOINT);p=_read(PROGRESS)
 else:a={'actual':np.full(shape,np.nan),'JVP':np.full((TIMES.size,FACES.size,3),np.nan),'audits':np.full((3,TIMES.size,4),np.nan),'FD_audits':np.full((TIMES.size,2,4),np.nan)};p={'completed':[]}
 done=set(p['completed'])
 for li,label in enumerate(c4f1.LAYOUT_LABELS):
  layout,conf=cfg[label]
  for ti,tv in enumerate(TIMES):
   key=f'a:{li}:{ti}'
   if key not in done:
    began=time.perf_counter();a['actual'][li,ti],a['audits'][li,ti]=_flux(conf['context'],selected[label][ti],layout.refinement_ratio);done.add(key);p['completed']=sorted(done);_save(CHECKPOINT,**a);_write(PROGRESS,p);print(f"c4f7 actual {label} {tv:.3f}s {time.perf_counter()-began:.1f}s",flush=True)
 for ti,tv in enumerate(TIMES):
  key=f'j:{ti}'
  if key not in done:
   began=time.perf_counter();alpha=c4f6.FD_STEP;plus,pa=_flux(fc['context'],selected['fine'][ti]+alpha*direction[ti],fl.refinement_ratio);minus,ma=_flux(fc['context'],selected['fine'][ti]-alpha*direction[ti],fl.refinement_ratio);a['JVP'][ti]=(plus-minus)/(2*alpha);a['FD_audits'][ti]=np.asarray((pa,ma));done.add(key);p['completed']=sorted(done);_save(CHECKPOINT,**a);_write(PROGRESS,p);print(f"c4f7 JVP {tv:.3f}s {time.perf_counter()-began:.1f}s",flush=True)
 a['times_seconds']=TIMES;a['faces']=FACES;return a
def _metric(v,sc,g):
 n=v/sc[None,None,:];d1=n[1]-n[0];d2=n[2]-n[1];n1=np.linalg.norm(d1);n2=np.linalg.norm(d2);order=float(np.log2(max(n1,np.finfo(float).tiny)/max(n2,np.finfo(float).tiny)));cos=float(np.vdot(d1.ravel(),d2.ravel()).real/max(n1*n2,np.finfo(float).tiny));return {'order':order,'cosine':cos,'passed':bool(order>=g['minimum_actual_flux_order'] and cos>=g['minimum_actual_flux_cosine'])}
def _analyze(a,m):
 g=m['gates'];sc=_load(c4f1.c4f.MIDDLE_ARRAYS)['tangent__export_scales'][:3];metrics={str(f):_metric(a['actual'][:,:,i],sc,g) for i,f in enumerate(FACES)};norms=np.asarray([np.linalg.norm(a['JVP'][:,i]/sc) for i in range(FACES.size)]);ratios=norms/max(norms[-1],np.finfo(float).tiny);eligible=[]
 for i,f in enumerate(FACES[:-1]):
  consecutive=i>=1 and ratios[i]<=g['maximum_JVP_fraction_of_transition'] and ratios[i-1]<=g['maximum_JVP_fraction_of_transition'];monotone=np.all(np.diff(ratios[:i+1])>=-g['maximum_nonmonotone_regrowth_fraction']);
  if consecutive and monotone and metrics[str(f)]['passed']:eligible.append(int(f))
 selected=max(eligible) if eligible else None;maxaudit=float(max(np.max(np.abs(a['audits'][...,:3])),np.max(np.abs(a['FD_audits'][...,:3]))));incoming=int(max(np.max(a['audits'][...,3]),np.max(a['FD_audits'][...,3])))
 method=maxaudit<=g['maximum_ledger_defect'] and incoming==0
 if not method:cl='fine_complement_decay_method_gate_failed';nxt='method_repair_only';passed=False
 elif selected is not None:cl='fine_complement_recovery_surface_candidate_identified';nxt='definitions_only_recovered_coupling_control_volume_manifest';passed=True
 else:cl='fine_complement_no_compact_recovery_surface';nxt='near_transition_architecture_localization_only';passed=False
 return {'schema_version':1,'work_package':WORK_PACKAGE,'classification':cl,'passed':passed,'method_gates_passed':method,'face_metrics':metrics,'JVP_fraction_of_transition':{str(f):float(r) for f,r in zip(FACES,ratios)},'selected_recovery_parent_face':selected,'eligible_faces':eligible,'maximum_ledger_defect':maxaudit,'incoming_excision_characteristics':incoming,'physical_failure_detected':False,'response_certificate_preserved':True,'absolute_closure_fit_authorized':False,'memory_propagation_authorized':False,'fixed_Q_micro_solver_authorized':False,'reduced_slow_evolution_authorized':False,'authorized_next':nxt}
def _catalog(s):
 rows=[]
 if CANONICAL_MANIFEST.exists():
  with CANONICAL_MANIFEST.open(newline='') as h:rows=list(csv.DictReader(h))
 rows=[r for r in rows if r.get('case')!=ARTIFACT]
 for p in sorted(CANONICAL_DIRECTORY.iterdir()):
  if p.is_file():rows.append({'case':ARTIFACT,'path':str(p.relative_to(ROOT)),'bytes':str(p.stat().st_size),'sha256':_sha(p),'scientific_status':'CERTIFIED' if s['passed'] else 'REJECTED'})
 with CANONICAL_MANIFEST.open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=['case','path','bytes','sha256','scientific_status'],lineterminator='\n');w.writeheader();w.writerows(rows)
 c=_read(CANONICAL_SUMMARY);c.setdefault('artifacts',{})[ARTIFACT]={'path':str(CANONICAL_DIRECTORY.relative_to(ROOT)),'classification':s['classification'],'passed':s['passed']};c.update({'case_count':len({r['case'] for r in rows}),'file_count':len(rows),'total_bytes':sum(int(r['bytes']) for r in rows),'latest_work_package':WORK_PACKAGE});_write(CANONICAL_SUMMARY,c)
def _finalize(a,s,m):
 CANONICAL_DIRECTORY.mkdir(parents=True,exist_ok=True);_save(DECISIVE_ARRAYS,**a);_write(CONFIG_PATH,{'faces':FACES.tolist(),'times_seconds':TIMES.tolist(),'FD_step':c4f6.FD_STEP});_write(CONTRACT_PATH,m);_write(SUMMARY_PATH,s);REPORT_PATH.parent.mkdir(parents=True,exist_ok=True);rows=['# Fine-complement spatial-decay audit','',f"Classification: `{s['classification']}`.",'','| Parent face | JVP/transition | Actual order | Actual cosine | Pass |','|---:|---:|---:|---:|---:|'];[rows.append(f"| {f} | {s['JVP_fraction_of_transition'][str(f)]:.6e} | {s['face_metrics'][str(f)]['order']:.6f} | {s['face_metrics'][str(f)]['cosine']:.6f} | {s['face_metrics'][str(f)]['passed']} |") for f in FACES];rows+=['',f"Selected recovery candidate: `{s['selected_recovery_parent_face']}`.",'','No trajectory ran. A selected face is only a candidate until a separately frozen control-volume identity certifies storage and source bookkeeping.'];REPORT_PATH.write_text('\n'.join(rows)+'\n');_write(PROVENANCE_PATH,{'input_hashes':{str(x.relative_to(ROOT)):_sha(x) for x in (c4f5.DECISIVE_ARRAYS,c4f6.MANIFEST_PATH)},'output_hashes':{}});(CANONICAL_DIRECTORY/'SHA256SUMS.txt').write_text(''.join(f"{_sha(x)}  {x.name}\n" for x in (CONFIG_PATH,CONTRACT_PATH,SUMMARY_PATH,PROVENANCE_PATH,DECISIVE_ARRAYS)));_catalog(s)
def main():
 m=_read(c4f6.MANIFEST_PATH);a=_evaluate(m);s=_analyze(a,m);_finalize(a,s,m);print(json.dumps(s,indent=2,sort_keys=True))
if __name__=='__main__':main()
