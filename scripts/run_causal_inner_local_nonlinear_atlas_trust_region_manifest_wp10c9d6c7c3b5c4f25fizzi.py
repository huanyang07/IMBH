#!/usr/bin/env python3
"""Freeze the nonlinear entropy-path and moving-STF atlas trust region."""

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

import run_causal_inner_entropy_stable_split_discretization_kernel_wp10c9d6c7c3b5c4f25fizzh1 as parent  # noqa: E402


WORK_PACKAGE = parent.AUTHORIZED_NEXT
CLASSIFICATION = "local_nonlinear_atlas_trust_region_manifest_frozen"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25fizzi1_local_nonlinear_atlas_trust_region_kernel"
PASS_NEXT = "definitions_only_WP10c9d6c7c3b5c4f25fizzj_bounded_nonlinear_split_microstep_manifest"
ARTIFACT = "causal_inner_local_nonlinear_atlas_trust_region_manifest_wp10c9d6c7c3b5c4f25fizzi"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = "docs/reports/current/CODEX_CAUSAL_INNER_LOCAL_NONLINEAR_ATLAS_TRUST_REGION_MANIFEST_WP10C9D6C7C3B5C4F25FIZZI_2026-08-26.md"
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = "scripts/run_causal_inner_local_nonlinear_atlas_trust_region_manifest_wp10c9d6c7c3b5c4f25fizzi.py"
THIS_TEST = "tests/test_causal_inner_local_nonlinear_atlas_trust_region_manifest_wp10c9d6c7c3b5c4f25fizzi.py"
PARENT_SHA256 = "ab4a91f40f69d3611018718ef77d0b826d3ada4210236d4d4297e23099a5e0d2"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _u():
    return parent._u()


def _validate_parent(*, require_clean=False):
    utils = _u()
    if utils._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != PARENT_SHA256:
        raise RuntimeError("split-discretization certificate checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    if not summary["passed"] or not summary["entropy_stable_split_discretization_certified"] or summary["authorized_next"] != WORK_PACKAGE or summary["complete_cycle_execution_authorized"]:
        raise RuntimeError("split-discretization classification changed")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("nonlinear atlas manifest needs a clean tracked tree")
    return hashes


def _contract():
    return {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "nonlinear_equilibrium_interface": {
            "variables": "four fixed-height entropy variables (alpha,beta_t,beta_R,beta_phi); beta_z is owned by the height port",
            "potential": "certified X^R=2Hp beta^R",
            "entropy_conservative_flux": "16-node Gauss-Legendre integral of grad_v X^R along the straight entropy-variable segment",
            "tadmor_identity": "Delta v dot f_ec = X^R(v_R)-X^R(v_L)",
            "entropy_stable_flux": "f_ec minus one-half lambda times the entropy-coordinate jump",
            "metric_policy": "one frozen Kerr-Schild geometry per local proof interface; geometric interfaces require a later path-conservative source certificate",
        },
        "moving_five_STF_connection": {
            "physical_representation": "full spacetime shear tensor reconstructed from five rest-frame amplitudes",
            "cross_gram": "metric contraction of left and right moving STF frames",
            "connection": "orthogonal polar factor of the cross-Gram matrix",
            "energy_property": "the connection is an isometry in the five-amplitude reservoir norm",
            "roundtrip": "right-to-left is the transpose of left-to-right",
            "polar_stretch": "stored as a geometric-work diagnostic and bounded by the trust gate",
        },
        "trust_region": {
            "primitive_log_density_radius": 0.01,
            "primitive_log_temperature_radius": 0.01,
            "horizontal_velocity_radius_over_c": 0.002,
            "log_height_radius": 0.005,
            "path_nodes_must_remain_physical": True,
            "endpoint_atlases_must_remain_causal_and_entropy_dissipative": True,
            "maximum_STF_polar_stretch": 0.08,
            "maximum_normalized_atlas_change": 0.08,
        },
        "kernel": {
            "physical_anchors": 47,
            "deterministic_endpoint_pairs_per_anchor": 8,
            "tadmor_relative_gate": 2e-9,
            "quadrature_refinement_gate": 2e-9,
            "STF_connection_orthogonality_gate": 2e-12,
            "STF_connection_roundtrip_gate": 2e-12,
            "trajectory_steps": 0,
        },
        "decision": {"pass_classification": "local_nonlinear_atlas_trust_region_kernel_certified", "pass_authorized_next": PASS_NEXT, "failure_classification": "local_nonlinear_atlas_trust_region_kernel_failed"},
        "claim_boundary": {"definitions_only": True, "bounded_nonlinear_microstep_authorized": False, "trajectory_authorized": False, "complete_cycle_execution_authorized": False},
        "authorized_next": AUTHORIZED_NEXT,
    }


def _update(summary):
    utils = _u(); rows = list(csv.DictReader(CANONICAL_MANIFEST.open(newline="", encoding="utf-8"))); rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file(): rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": utils._sha256(path), "scientific_status": "SUPPORTED"})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    catalog = utils._read_json(CANONICAL_SUMMARY); catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": CLASSIFICATION, "passed": True}; catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": utils._git("rev-parse", "HEAD"), "latest_work_package": WORK_PACKAGE}); utils._write_json(CANONICAL_SUMMARY, catalog)


def _freeze():
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists(): raise RuntimeError("nonlinear atlas manifest already exists")
    hashes = _validate_parent(require_clean=True); utils = _u(); CANONICAL_DIRECTORY.mkdir(parents=True); utils._write_json(CANONICAL_DIRECTORY / "trust_region_contract.json", _contract())
    summary = {"schema_version": 1, "work_package": WORK_PACKAGE, "classification": CLASSIFICATION, "passed": True, "definitions_only": True, "split_discretization_preserved": True, "nonlinear_atlas_trust_region_certified": False, "trajectory_authorized": False, "complete_cycle_execution_authorized": False, "authorized_next": AUTHORIZED_NEXT}; utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary); utils._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"parent_artifact": parent.ARTIFACT, "parent_checksum_manifest_sha256": PARENT_SHA256, "parent_hashes": hashes})
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True); REPORT_PATH.write_text("# Local nonlinear atlas trust-region manifest\n\nClassification: `local_nonlinear_atlas_trust_region_manifest_frozen`.\n\nThe equilibrium interface uses the certified radial flux potential and a straight entropy-variable path, while the moving five-STF reservoirs use the polar isometry between neighboring rest-frame bases. Explicit primitive radii bound each local atlas overlap.\n\nThis is definitions-only and authorizes no nonlinear step or trajectory.\n", encoding="utf-8")
    sources = (THIS_RUNNER, THIS_TEST, REPORT_RELATIVE); utils._write_json(CANONICAL_DIRECTORY / "provenance.json", {"implementation_commit": utils._git("rev-parse", "HEAD"), "source_hashes": {path: utils._sha256(ROOT / path) for path in sources}}); names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()); (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utils._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8"); _update(summary); return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--freeze", action="store_true"); args = parser.parse_args()
    if not args.freeze: parser.error("choose --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
