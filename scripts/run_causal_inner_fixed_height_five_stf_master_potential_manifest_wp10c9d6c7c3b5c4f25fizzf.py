#!/usr/bin/env python3
"""Freeze the restricted five-STF master-potential linear-stress test."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 5000))

import run_causal_inner_split_godunov_port_hamiltonian_proof_kernel_wp10c9d6c7c3b5c4f25fizze1 as parent  # noqa: E402

SCHEMA_VERSION = 1
WORK_PACKAGE = parent.AUTHORIZED_NEXT
CLASSIFICATION = "fixed_height_five_STF_master_potential_manifest_frozen"
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizzf1_"
    "restricted_five_STF_linear_stress_diagnostic"
)
FAILURE_NEXT = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizzg_"
    "fully_split_shear_height_port_atlas_manifest"
)
PASS_NEXT = (
    "WP10c9d6c7c3b5c4f25fizzf2_"
    "fixed_height_five_STF_physical_potential_implementation"
)
ARTIFACT = "causal_inner_fixed_height_five_stf_master_potential_manifest_wp10c9d6c7c3b5c4f25fizzf"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = "docs/reports/current/CODEX_CAUSAL_INNER_FIXED_HEIGHT_FIVE_STF_MASTER_POTENTIAL_MANIFEST_WP10C9D6C7C3B5C4F25FIZZF_2026-08-26.md"
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = "scripts/run_causal_inner_fixed_height_five_stf_master_potential_manifest_wp10c9d6c7c3b5c4f25fizzf.py"
THIS_TEST = "tests/test_causal_inner_fixed_height_five_stf_master_potential_manifest_wp10c9d6c7c3b5c4f25fizzf.py"
PARENT_SHA256 = "4e780e86956007a6d6ac88a5dc18fa0d3b49d54913888468dfba01137b0ae7bb"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _u(): return parent._u()


def _validate_parent(*, require_clean=False):
    u = _u(); checksum = parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt"
    if u._sha256(checksum) != PARENT_SHA256: raise RuntimeError("split proof checksum changed")
    hashes = u._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = u._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    if not summary["passed"] or not summary["split_kernel_certified"] or summary["authorized_next"] != WORK_PACKAGE or summary["complete_cycle_execution_authorized"]: raise RuntimeError("split proof authorization changed")
    if require_clean and u._git("status", "--short", "--untracked-files=no"): raise RuntimeError("five-STF manifest needs a clean tracked tree")
    return hashes


def _contract():
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "preserved_certificates": {"fixed_height_equilibrium_potential": True, "split_height_port_kernel": True, "moving_five_STF_basis": True},
        "restricted_candidate": {
            "shear_coordinates": "five amplitudes in the moving spatial STF basis E_A^{mu nu}(beta)",
            "constraints": "zeta symmetric, tracefree, and beta_mu*zeta^{mu nu}=0 identically",
            "invariants": {"nu": "zeta^{mu nu}*beta_mu*beta_nu", "I2": "zeta_mu_nu*zeta^{mu nu}", "I3": "zeta^mu_nu*zeta^nu_rho*zeta^rho_mu"},
            "scalar_ansatz": "chi=chi_eq+c1*nu+c2*I2+c3*I3+c4*nu**2+higher invariant barrier",
            "coefficients": "arbitrary smooth functions of alpha and beta^2; no witness fit",
            "moving_basis_differentiated": True,
        },
        "necessary_physical_condition": {
            "desired_linear_stress_map": "d T_shear^{mu nu}/d zeta_A at zeta=0 equals a nonzero multiple of E_A^{mu nu}",
            "all_five_components_required": True,
            "one_Rphi_projection_is_not_sufficient": True,
        },
        "diagnostic": {
            "same_47_witnesses": True,
            "transversality_gate": 2.0e-13,
            "invariant_identity_gate": 2.0e-13,
            "linear_stress_map_relative_defect_gate": 1.0e-10,
            "coefficient_independent_no_go_allowed": True,
            "trajectory_steps": 0,
        },
        "decision": {
            "pass": {"classification": "restricted_five_STF_master_potential_viable", "authorized_next": PASS_NEXT},
            "fail": {"classification": "restricted_five_STF_master_potential_linear_stress_obstructed", "authorized_next": FAILURE_NEXT, "interpretation": "reject only the scalar invariant realization on the constrained five-dimensional moving basis"},
        },
        "claim_boundary": {"definitions_only": True, "physical_five_STF_potential_certified": False, "fully_split_port_atlas_certified": False, "trajectory_authorized": False, "complete_cycle_execution_authorized": False},
        "authorized_next": AUTHORIZED_NEXT,
    }


def _update(summary):
    u = _u(); rows = list(csv.DictReader(CANONICAL_MANIFEST.open(newline="", encoding="utf-8"))); rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file(): rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": u._sha256(path), "scientific_status": "SUPPORTED"})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    catalog = u._read_json(CANONICAL_SUMMARY); catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": CLASSIFICATION, "passed": True}; catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": u._git("rev-parse", "HEAD"), "latest_work_package": WORK_PACKAGE}); u._write_json(CANONICAL_SUMMARY, catalog)


def _freeze():
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists(): raise RuntimeError("five-STF manifest exists")
    hashes = _validate_parent(require_clean=True); u = _u(); CANONICAL_DIRECTORY.mkdir(parents=True); u._write_json(CANONICAL_DIRECTORY / "linear_stress_contract.json", _contract())
    summary = {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "classification": CLASSIFICATION, "passed": True, "definitions_only": True, "split_height_port_kernel_preserved": True, "physical_five_STF_potential_certified": False, "trajectory_authorized": False, "complete_cycle_execution_authorized": False, "authorized_next": AUTHORIZED_NEXT}; u._write_json(CANONICAL_DIRECTORY / "summary.json", summary); u._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"parent_artifact": parent.ARTIFACT, "parent_checksum_manifest_sha256": PARENT_SHA256, "parent_hashes": hashes})
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True); REPORT_PATH.write_text("# Restricted five-STF master-potential manifest\n\n" f"Classification: `{CLASSIFICATION}`.\n\n" "This prospectively tests whether the invariant scalar ansatz can generate a nonzero linear physical shear stress when zeta is represented by only the five exactly transverse moving-STF amplitudes. A failure authorizes a fully split shear/height port atlas and does not affect the certified equilibrium potential or height port.\n\nNo trajectory or cycle execution is authorized.\n", encoding="utf-8")
    sources = (THIS_RUNNER, THIS_TEST, REPORT_RELATIVE); u._write_json(CANONICAL_DIRECTORY / "provenance.json", {"implementation_commit": u._git("rev-parse", "HEAD"), "source_hashes": {path: u._sha256(ROOT / path) for path in sources}}); names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()); (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{u._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8"); _update(summary); return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--freeze", action="store_true"); args = parser.parse_args()
    if not args.freeze: parser.error("choose --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
