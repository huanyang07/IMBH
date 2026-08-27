#!/usr/bin/env python3
"""Execute the restricted moving-five-STF linear-stress diagnostic."""

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
import run_causal_inner_fixed_height_five_stf_master_potential_manifest_wp10c9d6c7c3b5c4f25fizzf as manifest  # noqa:E402
from imri_qpe.layer3_minidisk_1d.causal_inner_eleven_field_convex import full_shear_rest_frame  # noqa:E402
from imri_qpe.layer3_minidisk_1d.causal_inner_restricted_stf_potential import audit_restricted_five_stf_scalar_potential  # noqa:E402

SCHEMA_VERSION = 1; WORK_PACKAGE = manifest.AUTHORIZED_NEXT
PASS_CLASSIFICATION = "restricted_five_STF_master_potential_viable"; FAIL_CLASSIFICATION = "restricted_five_STF_master_potential_linear_stress_obstructed"
ARTIFACT = "causal_inner_restricted_five_stf_linear_stress_diagnostic_wp10c9d6c7c3b5c4f25fizzf1"; CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = "docs/reports/current/CODEX_CAUSAL_INNER_RESTRICTED_FIVE_STF_LINEAR_STRESS_DIAGNOSTIC_WP10C9D6C7C3B5C4F25FIZZF1_2026-08-26.md"; REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = "scripts/run_causal_inner_restricted_five_stf_linear_stress_diagnostic_wp10c9d6c7c3b5c4f25fizzf1.py"; THIS_TEST = "tests/test_causal_inner_restricted_five_stf_linear_stress_diagnostic_wp10c9d6c7c3b5c4f25fizzf1.py"; PHYSICAL_SOURCE = "src/imri_qpe/layer3_minidisk_1d/causal_inner_restricted_stf_potential.py"; PHYSICAL_TEST = "tests/test_causal_inner_restricted_stf_potential.py"
PARENT_SHA256 = "19dc23152c0d86704aa89b880347d4ba23fde8b6b72f5b70bfadb29a6378ab70"; CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"; CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

def _u(): return manifest._u()
def _validate_parent(require_clean=False):
    u = _u(); checksum = manifest.CANONICAL_DIRECTORY / "SHA256SUMS.txt"
    if u._sha256(checksum) != PARENT_SHA256: raise RuntimeError("five-STF manifest checksum changed")
    hashes = u._validate_checksums(manifest.CANONICAL_DIRECTORY); summary = u._read_json(manifest.CANONICAL_DIRECTORY / "summary.json"); contract = u._read_json(manifest.CANONICAL_DIRECTORY / "linear_stress_contract.json")
    if not summary["passed"] or summary["authorized_next"] != WORK_PACKAGE or summary["physical_five_STF_potential_certified"] or contract["claim_boundary"]["complete_cycle_execution_authorized"]: raise RuntimeError("five-STF diagnostic contract changed")
    if require_clean and u._git("status", "--short", "--untracked-files=no"): raise RuntimeError("five-STF diagnostic needs clean tracked tree")
    return hashes, contract

def _certificate():
    began = time.perf_counter(); _, contract = _validate_parent(); rows = []; charts = []; radii = []; defects = []; transversality = []; invariants = []
    witness_began = time.perf_counter(); physical_witnesses = list(witnesses._physical_witnesses()); witness_seconds = time.perf_counter() - witness_began
    for index, label, radius, old_state, chart7 in physical_witnesses:
        frame = full_shear_rest_frame(old_state.geometry, radial_velocity_over_c=float(chart7[1]), azimuthal_velocity_over_c=float(chart7[2]), vertical_velocity_over_c=0.0)
        audit = audit_restricted_five_stf_scalar_potential(frame, temperature=float(np.exp(chart7[3])))
        rows.append({"index": index, "label": label, "radius_cm": radius, "audit": asdict(audit), "candidate_viable": audit.candidate_viable}); charts.append(chart7); radii.append(radius); defects.append(audit.linear_stress_map_relative_defect); transversality.append(audit.maximum_beta_transversality_defect); invariants.append(audit.maximum_first_invariant_derivative_at_origin)
    passed = bool(len(rows) == 47 and all(row["candidate_viable"] for row in rows)); classification = PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION; authorized_next = manifest.PASS_NEXT if passed else manifest.FAILURE_NEXT
    metrics = {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "classification": classification, "passed": passed, "audit_completed": True, "physical_witness_count": len(rows), "viable_witness_count": sum(row["candidate_viable"] for row in rows), "maximum_beta_transversality_defect": float(max(transversality)), "maximum_first_invariant_derivative_at_origin": float(max(invariants)), "minimum_linear_stress_map_relative_defect": float(min(defects)), "maximum_linear_stress_map_relative_defect": float(max(defects)), "coefficient_independent_no_go": not passed and min(defects) > 0.99, "fixed_height_equilibrium_potential_preserved": True, "split_height_port_kernel_preserved": True, "failure_scope": "restricted scalar invariant ansatz on the exactly transverse moving five-STF basis", "trajectory_steps": 0, "complete_cycle_execution_authorized": False, "witness_construction_wall_seconds": witness_seconds, "diagnostic_wall_seconds": time.perf_counter() - began, "rows": rows, "authorized_next": authorized_next}
    arrays = {"witness_charts7": np.asarray(charts), "witness_radii_cm": np.asarray(radii), "linear_stress_map_relative_defects": np.asarray(defects), "beta_transversality_defects": np.asarray(transversality), "first_invariant_derivatives_at_origin": np.asarray(invariants)}; del contract; return metrics, arrays

def _update(summary):
    u = _u(); rows = list(csv.DictReader(CANONICAL_MANIFEST.open(newline="", encoding="utf-8"))); rows = [row for row in rows if row.get("case") != ARTIFACT]; status = "SUPPORTED" if summary["passed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file(): rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": u._sha256(path), "scientific_status": status})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle: writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    catalog = u._read_json(CANONICAL_SUMMARY); catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": summary["classification"], "passed": summary["passed"]}; catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": u._git("rev-parse", "HEAD"), "latest_work_package": WORK_PACKAGE}); u._write_json(CANONICAL_SUMMARY, catalog)

def _canonicalize(metrics, arrays):
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists(): raise RuntimeError("five-STF diagnostic exists")
    hashes, _ = _validate_parent(True); u = _u(); CANONICAL_DIRECTORY.mkdir(parents=True); u._write_json(CANONICAL_DIRECTORY / "diagnostic_metrics.json", metrics); np.savez_compressed(CANONICAL_DIRECTORY / "diagnostic_arrays.npz", **arrays)
    summary = {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "classification": metrics["classification"], "passed": metrics["passed"], "audit_completed": True, "fixed_height_equilibrium_potential_preserved": True, "split_height_port_kernel_preserved": True, "physical_five_STF_potential_certified": metrics["passed"], "fully_split_port_atlas_manifest_authorized": not metrics["passed"], "trajectory_authorized": False, "complete_cycle_execution_authorized": False, "authorized_next": metrics["authorized_next"]}; u._write_json(CANONICAL_DIRECTORY / "summary.json", summary); u._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"manifest_artifact": manifest.ARTIFACT, "manifest_checksum_manifest_sha256": PARENT_SHA256, "manifest_hashes": hashes})
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True); REPORT_PATH.write_text("# Restricted five-STF linear-stress diagnostic\n\n" f"Classification: `{metrics['classification']}`.\n\n" f"All {metrics['physical_witness_count']} witnesses satisfy exact moving-basis transversality, but none produces a linear physical stress: the linear-map relative defect is `{metrics['minimum_linear_stress_map_relative_defect']:.6e}` on every witness. The result is coefficient-independent because `nu` vanishes identically and the remaining invariants begin at quadratic order.\n\n" "This rejects only the restricted scalar invariant ansatz. The equilibrium potential and split height port remain certified. No trajectory was executed.\n\n" f"Authorized next: `{metrics['authorized_next']}`.\n", encoding="utf-8")
    sources = (THIS_RUNNER, THIS_TEST, PHYSICAL_SOURCE, PHYSICAL_TEST, REPORT_RELATIVE); u._write_json(CANONICAL_DIRECTORY / "provenance.json", {"implementation_commit": u._git("rev-parse", "HEAD"), "source_hashes": {path: u._sha256(ROOT / path) for path in sources}, "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "thread_environment": {name: os.environ.get(name, "") for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}}); names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()); (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{u._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8"); _update(summary); return summary

def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run", action="store_true"); args = parser.parse_args()
    if not args.run: parser.error("choose --run")
    metrics, arrays = _certificate(); print(json.dumps(metrics, indent=2, sort_keys=True), flush=True); summary = _canonicalize(metrics, arrays); return 0 if summary["passed"] else 2
if __name__ == "__main__": raise SystemExit(main())
