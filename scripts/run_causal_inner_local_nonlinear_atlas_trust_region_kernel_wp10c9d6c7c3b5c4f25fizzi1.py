#!/usr/bin/env python3
"""Certify the local nonlinear entropy-path and moving-STF trust region."""

from __future__ import annotations

import argparse, csv, json, os, platform, sys, time
from dataclasses import asdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path: sys.path.insert(0, str(path))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 5000))

import run_causal_inner_equilibrium_column_thermodynamic_potential_implementation_wp10c9d6c7c3b5c4f25fizzc as witnesses  # noqa:E402
import run_causal_inner_local_nonlinear_atlas_trust_region_manifest_wp10c9d6c7c3b5c4f25fizzi as manifest  # noqa:E402
from imri_qpe.constants import C  # noqa:E402
from imri_qpe.layer3_minidisk_1d.causal_inner import gas_radiation_relativistic_sound_speed_squared  # noqa:E402
from imri_qpe.layer3_minidisk_1d.causal_inner_eleven_field_convex import full_shear_rest_frame  # noqa:E402
from imri_qpe.layer3_minidisk_1d.causal_inner_full_port_atlas import audit_full_port_atlas_anchor, build_full_port_atlas_anchor  # noqa:E402
from imri_qpe.layer3_minidisk_1d.causal_inner_nonlinear_port_atlas import audit_equilibrium_entropy_path_flux, audit_stf_polar_connection, equilibrium_entropy_point_from_primitive  # noqa:E402

SCHEMA_VERSION=1; WORK_PACKAGE=manifest.AUTHORIZED_NEXT
PASS_CLASSIFICATION="local_nonlinear_atlas_trust_region_kernel_certified"; FAIL_CLASSIFICATION="local_nonlinear_atlas_trust_region_kernel_failed"; AUTHORIZED_NEXT=manifest.PASS_NEXT
ARTIFACT="causal_inner_local_nonlinear_atlas_trust_region_kernel_wp10c9d6c7c3b5c4f25fizzi1"; CANONICAL_DIRECTORY=ROOT/"results/canonical"/ARTIFACT
REPORT_RELATIVE="docs/reports/current/CODEX_CAUSAL_INNER_LOCAL_NONLINEAR_ATLAS_TRUST_REGION_KERNEL_WP10C9D6C7C3B5C4F25FIZZI1_2026-08-26.md"; REPORT_PATH=ROOT/REPORT_RELATIVE
THIS_RUNNER="scripts/run_causal_inner_local_nonlinear_atlas_trust_region_kernel_wp10c9d6c7c3b5c4f25fizzi1.py"; THIS_TEST="tests/test_causal_inner_local_nonlinear_atlas_trust_region_kernel_wp10c9d6c7c3b5c4f25fizzi1.py"; PHYSICAL_SOURCE="src/imri_qpe/layer3_minidisk_1d/causal_inner_nonlinear_port_atlas.py"; PHYSICAL_TEST="tests/test_causal_inner_nonlinear_port_atlas.py"
PARENT_SHA256="227fe24e15851c9a72f4a0f9295212fb84ec261f4d76378e0f931ef8aef90dd0"; CANONICAL_MANIFEST=ROOT/"results/manifests/canonical_artifacts.csv"; CANONICAL_SUMMARY=ROOT/"results/manifests/canonical_summary.json"

def _u(): return manifest._u()
def _validate_parent(require_clean=False):
    u=_u()
    if u._sha256(manifest.CANONICAL_DIRECTORY/"SHA256SUMS.txt")!=PARENT_SHA256: raise RuntimeError("nonlinear trust manifest checksum changed")
    hashes=u._validate_checksums(manifest.CANONICAL_DIRECTORY); summary=u._read_json(manifest.CANONICAL_DIRECTORY/"summary.json"); contract=u._read_json(manifest.CANONICAL_DIRECTORY/"trust_region_contract.json")
    if not summary["passed"] or not summary["definitions_only"] or summary["authorized_next"]!=WORK_PACKAGE or contract["kernel"]["trajectory_steps"] or summary["complete_cycle_execution_authorized"]: raise RuntimeError("nonlinear trust contract changed")
    if require_clean and u._git("status","--short","--untracked-files=no"): raise RuntimeError("nonlinear trust kernel needs clean tracked tree")
    return hashes,contract

_DIRECTIONS=np.asarray(((1,0,0,0,0),(0,1,0,0,0),(0,0,1,0,0),(0,0,0,1,0),(1,1,0,0,1),(1,-1,1,0,-1),(-1,1,0,1,1),(1,1,-1,1,-1)),dtype=float)

def _atlas(*,rho,T,H,omega,alpha,tau,ur):
    sound=float(np.sqrt(gas_radiation_relativistic_sound_speed_squared(rho,T)))
    return build_full_port_atlas_anchor(sound_speed=sound,temperature=T,proper_half_thickness=H,proper_vertical_frequency=omega,alpha=alpha,shear_relaxation_time=tau,transport_speed_over_c=ur)

def _atlas_change(left,right,tref):
    radial=float(np.linalg.norm(right.coordinate_radial_matrix-left.coordinate_radial_matrix,2)/max(np.linalg.norm(left.coordinate_radial_matrix,2),np.linalg.norm(right.coordinate_radial_matrix,2),1.0))
    source=float(tref*np.linalg.norm(right.source_matrix-left.source_matrix,2)/max(tref*np.linalg.norm(left.source_matrix,2),tref*np.linalg.norm(right.source_matrix,2),1.0))
    return max(radial,source)

def _certificate():
    began=time.perf_counter();_,contract=_validate_parent(); witness_began=time.perf_counter(); physical=list(witnesses._physical_witnesses()); witness_seconds=time.perf_counter()-witness_began
    rows=[];charts=[];radii=[];tadmor=[];quad=[];orth=[];roundtrip=[];stretch=[];changes=[];minrho=[];minT=[]
    for index,label,radius,old,chart in physical:
        H=float(np.exp(chart[5]));rho=float(np.exp(chart[0]))/(2*H);T=float(np.exp(chart[3]));ur=float(chart[1]);uphi=float(chart[2]);Sigma=float(np.exp(chart[0]));omega=float(np.sqrt(old.thermodynamics.integrated_pressure/(Sigma*H**2)));sound=float(old.thermodynamics.sound_speed);alpha=float((old.closure.viscous_signal_speed_over_c*C/sound)**2);tau=float(old.closure.relaxation_time);tref=min(tau,1/omega)
        charts.append(chart);radii.append(radius)
        for direction_index,direction in enumerate(_DIRECTIONS):
            drho,dtemp,dur,duphi,dheight=direction*np.asarray((.01,.01,.002,.002,.005))
            rho_l=rho*np.exp(-.5*drho);rho_r=rho*np.exp(.5*drho);T_l=T*np.exp(-.5*dtemp);T_r=T*np.exp(.5*dtemp);ur_l=ur-.5*dur;ur_r=ur+.5*dur;uphi_l=uphi-.5*duphi;uphi_r=uphi+.5*duphi;H_l=H*np.exp(-.5*dheight);H_r=H*np.exp(.5*dheight)
            left_point=equilibrium_entropy_point_from_primitive(old.geometry,density=rho_l,temperature=T_l,proper_half_thickness=H,radial_velocity_over_c=ur_l,azimuthal_velocity_over_c=uphi_l)
            right_point=equilibrium_entropy_point_from_primitive(old.geometry,density=rho_r,temperature=T_r,proper_half_thickness=H,radial_velocity_over_c=ur_r,azimuthal_velocity_over_c=uphi_r)
            path_audit=audit_equilibrium_entropy_path_flux(left_point,right_point)
            left_frame=full_shear_rest_frame(old.geometry,radial_velocity_over_c=ur_l,azimuthal_velocity_over_c=uphi_l,vertical_velocity_over_c=0.0);right_frame=full_shear_rest_frame(old.geometry,radial_velocity_over_c=ur_r,azimuthal_velocity_over_c=uphi_r,vertical_velocity_over_c=0.0);frame_audit=audit_stf_polar_connection(left_frame,right_frame)
            left_atlas=_atlas(rho=rho_l,T=T_l,H=H_l,omega=omega,alpha=alpha,tau=tau,ur=ur_l);right_atlas=_atlas(rho=rho_r,T=T_r,H=H_r,omega=omega,alpha=alpha,tau=tau,ur=ur_r);left_atlas_audit=audit_full_port_atlas_anchor(left_atlas);right_atlas_audit=audit_full_port_atlas_anchor(right_atlas);change=_atlas_change(left_atlas,right_atlas,tref)
            passed=bool(path_audit.passed and frame_audit.passed and left_atlas_audit.passed and right_atlas_audit.passed and change<=.08)
            rows.append({"witness_index":index,"witness_label":label,"direction_index":direction_index,"radius_cm":radius,"path_audit":asdict(path_audit),"STF_connection_audit":asdict(frame_audit),"normalized_atlas_change":change,"left_atlas_passed":left_atlas_audit.passed,"right_atlas_passed":right_atlas_audit.passed,"passed":passed})
            tadmor.append(path_audit.tadmor_relative_defect);quad.append(path_audit.quadrature_refinement_relative_defect);orth.append(frame_audit.orthogonality_defect);roundtrip.append(frame_audit.reverse_roundtrip_defect);stretch.append(frame_audit.polar_stretch);changes.append(change);minrho.append(path_audit.minimum_path_density);minT.append(path_audit.minimum_path_temperature)
    passed=bool(len(rows)==47*8 and all(row["passed"] for row in rows));metrics={"schema_version":1,"work_package":WORK_PACKAGE,"classification":PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,"passed":passed,"physical_anchor_count":len(physical),"endpoint_pair_count":len(rows),"passing_endpoint_pair_count":sum(row["passed"] for row in rows),"maximum_tadmor_relative_defect":float(max(tadmor)),"maximum_quadrature_refinement_relative_defect":float(max(quad)),"maximum_STF_connection_orthogonality_defect":float(max(orth)),"maximum_STF_connection_roundtrip_defect":float(max(roundtrip)),"maximum_STF_polar_stretch":float(max(stretch)),"maximum_normalized_atlas_change":float(max(changes)),"minimum_path_density":float(min(minrho)),"minimum_path_temperature":float(min(minT)),"entropy_path_flux_certified":passed,"moving_STF_connection_certified":passed,"split_discretization_preserved":True,"trajectory_steps":0,"complete_cycle_execution_authorized":False,"witness_construction_wall_seconds":witness_seconds,"certificate_wall_seconds":time.perf_counter()-began,"rows":rows,"authorized_next":AUTHORIZED_NEXT if passed else None}
    arrays={"witness_charts7":np.asarray(charts),"witness_radii_cm":np.asarray(radii),"tadmor_relative_defects":np.asarray(tadmor),"quadrature_refinement_relative_defects":np.asarray(quad),"STF_connection_orthogonality_defects":np.asarray(orth),"STF_connection_roundtrip_defects":np.asarray(roundtrip),"STF_polar_stretches":np.asarray(stretch),"normalized_atlas_changes":np.asarray(changes),"minimum_path_densities":np.asarray(minrho),"minimum_path_temperatures":np.asarray(minT)};del contract;return metrics,arrays

def _update(summary):
    u=_u();rows=list(csv.DictReader(CANONICAL_MANIFEST.open(newline="",encoding="utf-8")));rows=[r for r in rows if r.get("case")!=ARTIFACT];status="SUPPORTED" if summary["passed"] else "REJECTED"
    for p in sorted(CANONICAL_DIRECTORY.iterdir()):
        if p.is_file():rows.append({"case":ARTIFACT,"path":str(p.relative_to(ROOT)),"bytes":str(p.stat().st_size),"sha256":u._sha256(p),"scientific_status":status})
    with CANONICAL_MANIFEST.open("w",newline="",encoding="utf-8") as h:w=csv.DictWriter(h,fieldnames=("case","path","bytes","sha256","scientific_status"),lineterminator="\n");w.writeheader();w.writerows(rows)
    c=u._read_json(CANONICAL_SUMMARY);c.setdefault("artifacts",{})[ARTIFACT]={"path":str(CANONICAL_DIRECTORY.relative_to(ROOT)),"classification":summary["classification"],"passed":summary["passed"]};c.update({"case_count":len({r['case'] for r in rows}),"file_count":len(rows),"total_bytes":sum(int(r['bytes']) for r in rows),"all_payload_hashes_recorded":True,"latest_source_parent_commit":u._git("rev-parse","HEAD"),"latest_work_package":WORK_PACKAGE});u._write_json(CANONICAL_SUMMARY,c)

def _canonicalize(metrics,arrays):
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():raise RuntimeError("nonlinear trust certificate exists")
    hashes,_=_validate_parent(True);u=_u();CANONICAL_DIRECTORY.mkdir(parents=True);u._write_json(CANONICAL_DIRECTORY/"kernel_metrics.json",metrics);np.savez_compressed(CANONICAL_DIRECTORY/"kernel_arrays.npz",**arrays)
    summary={"schema_version":1,"work_package":WORK_PACKAGE,"classification":metrics["classification"],"passed":metrics["passed"],"split_discretization_preserved":True,"local_nonlinear_atlas_trust_region_certified":metrics["passed"],"bounded_nonlinear_microstep_authorized":False,"trajectory_authorized":False,"complete_cycle_execution_authorized":False,"authorized_next":metrics["authorized_next"]};u._write_json(CANONICAL_DIRECTORY/"summary.json",summary);u._write_json(CANONICAL_DIRECTORY/"input_lock.json",{"manifest_artifact":manifest.ARTIFACT,"manifest_checksum_manifest_sha256":PARENT_SHA256,"manifest_hashes":hashes})
    REPORT_PATH.parent.mkdir(parents=True,exist_ok=True);REPORT_PATH.write_text(f"# Local nonlinear port-atlas trust-region certificate\n\nClassification: `{metrics['classification']}`.\n\n{metrics['passing_endpoint_pair_count']}/{metrics['endpoint_pair_count']} prospectively frozen endpoint pairs pass. The worst Tadmor defect is `{metrics['maximum_tadmor_relative_defect']:.6e}`, STF roundtrip defect `{metrics['maximum_STF_connection_roundtrip_defect']:.6e}`, polar stretch `{metrics['maximum_STF_polar_stretch']:.6e}`, and normalized atlas change `{metrics['maximum_normalized_atlas_change']:.6e}`.\n\nThis certifies only local nonlinear atlas overlap. No trajectory or complete-cycle execution occurred.\n\nAuthorized next: `{metrics['authorized_next']}`.\n",encoding="utf-8")
    sources=(THIS_RUNNER,THIS_TEST,PHYSICAL_SOURCE,PHYSICAL_TEST,REPORT_RELATIVE);u._write_json(CANONICAL_DIRECTORY/"provenance.json",{"implementation_commit":u._git("rev-parse","HEAD"),"source_hashes":{p:u._sha256(ROOT/p) for p in sources},"python":sys.version,"numpy":np.__version__,"platform":platform.platform(),"thread_environment":{n:os.environ.get(n,"") for n in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS")}});names=sorted(p.name for p in CANONICAL_DIRECTORY.iterdir());(CANONICAL_DIRECTORY/"SHA256SUMS.txt").write_text("".join(f"{u._sha256(CANONICAL_DIRECTORY/n)}  {n}\n" for n in names),encoding="utf-8");_update(summary);return summary

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run",action="store_true");a=p.parse_args()
    if not a.run:p.error("choose --run")
    m,x=_certificate();print(json.dumps(m,indent=2,sort_keys=True),flush=True);return 0 if _canonicalize(m,x)["passed"] else 2
if __name__=="__main__":raise SystemExit(main())
