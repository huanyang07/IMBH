#!/usr/bin/env python3
"""Freeze the fully modular equilibrium/shear/height port atlas."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT/"src",ROOT/"scripts"):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
sys.setrecursionlimit(max(sys.getrecursionlimit(),5000))
import run_causal_inner_restricted_five_stf_linear_stress_diagnostic_wp10c9d6c7c3b5c4f25fizzf1 as parent  # noqa:E402
SCHEMA_VERSION=1;WORK_PACKAGE=parent.manifest.FAILURE_NEXT;CLASSIFICATION="fully_split_shear_height_port_atlas_manifest_frozen";AUTHORIZED_NEXT="WP10c9d6c7c3b5c4f25fizzg1_fully_split_physical_port_atlas_kernel";PASS_NEXT="definitions_only_WP10c9d6c7c3b5c4f25fizzh_entropy_stable_split_discretization_manifest"
ARTIFACT="causal_inner_fully_split_shear_height_port_atlas_manifest_wp10c9d6c7c3b5c4f25fizzg";CANONICAL_DIRECTORY=ROOT/"results/canonical"/ARTIFACT
REPORT_RELATIVE="docs/reports/current/CODEX_CAUSAL_INNER_FULLY_SPLIT_SHEAR_HEIGHT_PORT_ATLAS_MANIFEST_WP10C9D6C7C3B5C4F25FIZZG_2026-08-26.md";REPORT_PATH=ROOT/REPORT_RELATIVE;THIS_RUNNER="scripts/run_causal_inner_fully_split_shear_height_port_atlas_manifest_wp10c9d6c7c3b5c4f25fizzg.py";THIS_TEST="tests/test_causal_inner_fully_split_shear_height_port_atlas_manifest_wp10c9d6c7c3b5c4f25fizzg.py";PARENT_SHA256="e372ce1a3987b3ba91dd84bd476b061824c3da63137bcacc83e2e47a28c7e93b";CANONICAL_MANIFEST=ROOT/"results/manifests/canonical_artifacts.csv";CANONICAL_SUMMARY=ROOT/"results/manifests/canonical_summary.json"
def _u():return parent._u()
def _validate_parent(clean=False):
 u=_u();
 if u._sha256(parent.CANONICAL_DIRECTORY/"SHA256SUMS.txt")!=PARENT_SHA256:raise RuntimeError("five-STF rejection changed")
 h=u._validate_checksums(parent.CANONICAL_DIRECTORY);s=u._read_json(parent.CANONICAL_DIRECTORY/"summary.json")
 if s["passed"] or not s["audit_completed"] or not s["fully_split_port_atlas_manifest_authorized"] or s["authorized_next"]!=WORK_PACKAGE or s["complete_cycle_execution_authorized"]:raise RuntimeError("five-STF rejection classification changed")
 if clean and u._git("status","--short","--untracked-files=no"):raise RuntimeError("port atlas manifest needs clean tracked tree")
 return h
def _contract():
 return {"schema_version":1,"work_package":WORK_PACKAGE,"classification":CLASSIFICATION,
 "preserved":{"fixed_height_equilibrium_potential_certified":True,"height_port_kernel_certified":True,"moving_five_STF_basis_certified":True,"one_piece_height_potential_rejected":True,"restricted_five_STF_scalar_potential_rejected":True},
 "architecture":{"type":"state-local symmetric-hyperbolic GENERIC/port-Hamiltonian atlas","fields":"4 equilibrium Godunov + 5 physical STF shear ports + 2 vertical ports = 11","global_common_potential_required":False,"anchor_policy":"all thermodynamic metrics, bases and coefficients frozen within an accepted substep and recomputed only at an explicit atlas reanchor","physical_tensor":"pi^{mu nu}=sum_A pi_A E_A^{mu nu}(u,g)","full_tensor_no_projection":True},
 "normalized_local_principal_form":{"A0":"identity after the anchor's fixed entropy-coordinate congruence","rest_acoustic_speed":"c_s/c from the exact gas+radiation EOS","shear_signal_speed":"c_nu/c=sqrt(alpha)*c_s/c","radial_STF_incidence":"C_iA=E_A^{R i} for i=(R,phi,z)","rest_matrix":"symmetric acoustic plus velocity--five-STF incidence couplings","coordinate_matrix":"A_r=(beta_r*I+K)*(I+beta_r*K)^(-1), the symmetric relativistic spectral map","causality":"every eigenvalue strictly inside (-1,1)"},
 "ports":{"shear":{"reversible":"symmetric spatial stress/work incidence is skew-adjoint in the energy ledger","relaxation":"-pi_A/tau_pi","heat":"sum_A reservoir_loss_A is deposited into thermal energy","physical_entropy":"nonnegative"},"height":{"normalized_source":"[[0,Omega_H],[-Omega_H,-gamma_H]]","Omega_H_squared":"Omega_perp**2+R*T/H**2","gamma_H":"alpha*Omega_perp","damping_heat":"gamma_H times vertical reservoir"}},
 "atlas":{"anchors":"same frozen physical envelope, later extended prospectively","stored":"primitive anchor, entropy congruence, STF frame, A0, Ar, source J-R, EOS and coefficient hashes","overlap":"nearest certified anchor only inside a prospectively audited trust radius","outside_overlap":"fail closed and request a new offline anchor","interpolation":"structure-preserving interpolation only after a separate certificate"},
 "kernel":{"witnesses":47,"temporal_minimum_eigenvalue_gate":1e-10,"symmetry_gate":1e-12,"maximum_speed_gate":.999,"STF_constraint_gate":2e-13,"source_positive_part_gate":1e-12,"energy_ledger_gate":1e-12,"trajectory_steps":0},
 "decision":{"pass_classification":"fully_split_physical_port_atlas_kernel_certified","pass_authorized_next":PASS_NEXT,"failure_classification":"fully_split_physical_port_atlas_kernel_failed"},
 "claim_boundary":{"definitions_only":True,"discretization_certified":False,"trajectory_authorized":False,"cycle_execution_authorized":False},"authorized_next":AUTHORIZED_NEXT}
def _update(s):
 u=_u();rows=list(csv.DictReader(CANONICAL_MANIFEST.open(newline="",encoding="utf-8")));rows=[r for r in rows if r.get("case")!=ARTIFACT]
 for p in sorted(CANONICAL_DIRECTORY.iterdir()):
  if p.is_file():rows.append({"case":ARTIFACT,"path":str(p.relative_to(ROOT)),"bytes":str(p.stat().st_size),"sha256":u._sha256(p),"scientific_status":"SUPPORTED"})
 with CANONICAL_MANIFEST.open("w",newline="",encoding="utf-8") as h:w=csv.DictWriter(h,fieldnames=("case","path","bytes","sha256","scientific_status"),lineterminator="\n");w.writeheader();w.writerows(rows)
 c=u._read_json(CANONICAL_SUMMARY);c.setdefault("artifacts",{})[ARTIFACT]={"path":str(CANONICAL_DIRECTORY.relative_to(ROOT)),"classification":CLASSIFICATION,"passed":True};c.update({"case_count":len({r['case'] for r in rows}),"file_count":len(rows),"total_bytes":sum(int(r['bytes']) for r in rows),"all_payload_hashes_recorded":True,"latest_source_parent_commit":u._git("rev-parse","HEAD"),"latest_work_package":WORK_PACKAGE});u._write_json(CANONICAL_SUMMARY,c)
def _freeze():
 if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():raise RuntimeError("port atlas manifest exists")
 h=_validate_parent(True);u=_u();CANONICAL_DIRECTORY.mkdir(parents=True);u._write_json(CANONICAL_DIRECTORY/"atlas_contract.json",_contract());s={"schema_version":1,"work_package":WORK_PACKAGE,"classification":CLASSIFICATION,"passed":True,"definitions_only":True,"prior_rejections_preserved":True,"fully_split_port_atlas_kernel_certified":False,"trajectory_authorized":False,"complete_cycle_execution_authorized":False,"authorized_next":AUTHORIZED_NEXT};u._write_json(CANONICAL_DIRECTORY/"summary.json",s);u._write_json(CANONICAL_DIRECTORY/"input_lock.json",{"parent_artifact":parent.ARTIFACT,"parent_checksum_manifest_sha256":PARENT_SHA256,"parent_hashes":h});REPORT_PATH.parent.mkdir(parents=True,exist_ok=True);REPORT_PATH.write_text("# Fully split shear/height port-atlas manifest\n\nClassification: `fully_split_shear_height_port_atlas_manifest_frozen`.\n\nThe viable eleven-field architecture is modular: the certified fixed-height equilibrium potential supplies the four-field thermodynamic core; five physical moving-STF amplitudes are reciprocal shear ports; height and vertical velocity form the certified vertical port. Each atlas anchor freezes a symmetric local principal form, and relativistic spectral mapping preserves causality.\n\nThis package is definitions-only and authorizes no discretization, trajectory, or cycle execution.\n",encoding="utf-8");sources=(THIS_RUNNER,THIS_TEST,REPORT_RELATIVE);u._write_json(CANONICAL_DIRECTORY/"provenance.json",{"implementation_commit":u._git("rev-parse","HEAD"),"source_hashes":{p:u._sha256(ROOT/p) for p in sources}});names=sorted(p.name for p in CANONICAL_DIRECTORY.iterdir());(CANONICAL_DIRECTORY/"SHA256SUMS.txt").write_text("".join(f"{u._sha256(CANONICAL_DIRECTORY/n)}  {n}\n" for n in names));_update(s);return s
def main():
 p=argparse.ArgumentParser();p.add_argument("--freeze",action="store_true");x=p.parse_args();
 if not x.freeze:p.error("choose --freeze")
 print(json.dumps(_freeze(),indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
