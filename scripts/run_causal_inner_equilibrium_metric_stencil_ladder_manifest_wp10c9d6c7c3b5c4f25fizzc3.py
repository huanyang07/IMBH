#!/usr/bin/env python3
"""Freeze a nonpropagating metric-stencil factor diagnosis."""

from __future__ import annotations

import argparse, csv, json, os, platform, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path: sys.path.insert(0, str(path))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 5000))
import run_causal_inner_equilibrium_compensated_coordinate_implementation_wp10c9d6c7c3b5c4f25fizzc2 as parent  # noqa: E402

SCHEMA_VERSION = 1
WORK_PACKAGE = "definitions_only_WP10c9d6c7c3b5c4f25fizzc3_equilibrium_metric_stencil_ladder_manifest"
CLASSIFICATION = "equilibrium_metric_stencil_ladder_manifest_frozen"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25fizzc4_equilibrium_metric_stencil_ladder_diagnostic"
ARTIFACT = "causal_inner_equilibrium_metric_stencil_ladder_manifest_wp10c9d6c7c3b5c4f25fizzc3"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = "docs/reports/current/CODEX_CAUSAL_INNER_EQUILIBRIUM_METRIC_STENCIL_LADDER_MANIFEST_WP10C9D6C7C3B5C4F25FIZZC3_2026-08-26.md"
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = "scripts/run_causal_inner_equilibrium_metric_stencil_ladder_manifest_wp10c9d6c7c3b5c4f25fizzc3.py"
THIS_TEST = "tests/test_causal_inner_equilibrium_metric_stencil_ladder_manifest_wp10c9d6c7c3b5c4f25fizzc3.py"
PARENT_SHA = "94cb11f6150b1166a92a6d984af9a4238ec7b431202a8d1fb4a1494eaf9a5369"
FACTORS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

def _utils(): return parent._utils()

def _validate_parent(require_clean=False):
    u=_utils()
    if u._sha256(parent.CANONICAL_DIRECTORY/"SHA256SUMS.txt") != PARENT_SHA: raise RuntimeError("stencil rejection checksum changed")
    hashes=u._validate_checksums(parent.CANONICAL_DIRECTORY); s=u._read_json(parent.CANONICAL_DIRECTORY/"summary.json"); m=u._read_json(parent.CANONICAL_DIRECTORY/"certificate_metrics.json")
    if s["passed"] or s["classification"]!=parent.FAIL_CLASSIFICATION or s["authorized_next"] is not None or m["maximum_physical_current_relative_defect"]>1e-10 or m["maximum_complex_step_current_jacobian_relative_defect"]>1e-9 or m["maximum_sixth_order_current_jacobian_relative_defect"]!=2.138526336420079e-05: raise RuntimeError("stencil-only rejection changed")
    if require_clean and u._git("status","--short","--untracked-files=no"): raise RuntimeError("stencil manifest needs clean tracked tree")
    return {"hashes":hashes,"summary":s,"metrics":m}

def _contract():
    return {"schema_version":1,"work_package":WORK_PACKAGE,"classification":CLASSIFICATION,"preserved_rejection":{"artifact":parent.ARTIFACT,"retroactive_pass_forbidden":True,"only_failed_gate":"sixth_order_current_derivative","failed_value":2.138526336420079e-05,"unchanged_gate":2e-5,"witness_index":44,"witness_label":"empirical_max_field_2"},"diagnostic":{"step_factors":FACTORS,"same_saved_state_and_analytic_jacobian":True,"no_trajectory":True,"no_coefficient_fit":True,"passing_factor_gate":2e-5,"minimum_passing_factors":3,"minimum_contiguous_passing_factors":3,"selection_rule":"center factor of widest contiguous passing run; ties choose center nearest 1","complex_step_gate":1e-9},"outcomes":{"diagnosed":"authorize one same-47-witness rerun using the selected frozen factor","not_diagnosed":"stop equilibrium architecture"},"claim_boundary":{"equilibrium_physical_potential_certified":False,"dynamic_height_potential_certified":False,"eleven_field_trajectory_authorized":False,"complete_cycle_execution_authorized":False},"authorized_next":AUTHORIZED_NEXT}

def _update(summary):
    u=_utils(); rows=list(csv.DictReader(CANONICAL_MANIFEST.open(newline="",encoding="utf-8"))); rows=[r for r in rows if r.get("case")!=ARTIFACT]
    for p in sorted(CANONICAL_DIRECTORY.iterdir()):
        if p.is_file(): rows.append({"case":ARTIFACT,"path":str(p.relative_to(ROOT)),"bytes":str(p.stat().st_size),"sha256":u._sha256(p),"scientific_status":"SUPPORTED"})
    with CANONICAL_MANIFEST.open("w",newline="",encoding="utf-8") as h: w=csv.DictWriter(h,fieldnames=("case","path","bytes","sha256","scientific_status"),lineterminator="\n");w.writeheader();w.writerows(rows)
    c=u._read_json(CANONICAL_SUMMARY);c.setdefault("artifacts",{})[ARTIFACT]={"path":str(CANONICAL_DIRECTORY.relative_to(ROOT)),"classification":CLASSIFICATION,"passed":True};c.update({"case_count":len({r['case'] for r in rows}),"file_count":len(rows),"total_bytes":sum(int(r['bytes']) for r in rows),"all_payload_hashes_recorded":True,"latest_source_parent_commit":u._git("rev-parse","HEAD"),"latest_work_package":WORK_PACKAGE});u._write_json(CANONICAL_SUMMARY,c)

def _freeze():
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists(): raise RuntimeError("stencil manifest exists")
    v=_validate_parent(True);u=_utils();CANONICAL_DIRECTORY.mkdir(parents=True);u._write_json(CANONICAL_DIRECTORY/"diagnostic_contract.json",_contract());s={"schema_version":1,"work_package":WORK_PACKAGE,"classification":CLASSIFICATION,"passed":True,"definitions_only":True,"compensated_rejection_preserved":True,"equilibrium_physical_potential_certified":False,"complete_cycle_execution_authorized":False,"authorized_next":AUTHORIZED_NEXT};u._write_json(CANONICAL_DIRECTORY/"summary.json",s);u._write_json(CANONICAL_DIRECTORY/"input_lock.json",{"parent_artifact":parent.ARTIFACT,"parent_checksum_sha256":PARENT_SHA,"parent_hashes":v["hashes"]})
    REPORT_PATH.parent.mkdir(parents=True,exist_ok=True);REPORT_PATH.write_text(f"# Equilibrium metric-stencil ladder manifest\n\nClassification: `{CLASSIFICATION}`.\n\nThe compensated certificate remains rejected solely because one sixth-order stencil defect was 2.138526e-5 against 2.0e-5. This nonpropagating factor ladder preserves that gate and the saved witness.\n\nAuthorized next: `{AUTHORIZED_NEXT}`.\n",encoding="utf-8")
    u._write_json(CANONICAL_DIRECTORY/"provenance.json",{"schema_version":1,"work_package":WORK_PACKAGE,"implementation_commit":u._git("rev-parse","HEAD"),"source_hashes":{p:u._sha256(ROOT/p) for p in (THIS_RUNNER,THIS_TEST,REPORT_RELATIVE)},"python":sys.version,"numpy":np.__version__,"platform":platform.platform(),"thread_environment":{n:os.environ.get(n,"") for n in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS")}});names=sorted(p.name for p in CANONICAL_DIRECTORY.iterdir());(CANONICAL_DIRECTORY/"SHA256SUMS.txt").write_text("".join(f"{u._sha256(CANONICAL_DIRECTORY/n)}  {n}\n" for n in names),encoding="utf-8");_update(s);return s

def main():
    p=argparse.ArgumentParser();p.add_argument("--freeze",action="store_true");a=p.parse_args();
    if not a.freeze:p.error("choose --freeze")
    print(json.dumps(_freeze(),indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
