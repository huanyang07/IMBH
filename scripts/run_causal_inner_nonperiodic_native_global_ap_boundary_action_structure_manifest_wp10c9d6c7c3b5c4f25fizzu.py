#!/usr/bin/env python3
"""Freeze the native nonperiodic global AP/SBP-SAT boundary action."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_cycle_physical_input_bundle_schema_and_validator_certificate_wp10c9d6c7c3b5c4f25fizzt1 as parent  # noqa: E402
import run_causal_inner_prefix_port_payload_and_boundary_structure_certificate_wp10c9d6c7c3b5c4f25fizzr1 as ports  # noqa: E402


WORK_PACKAGE = parent.AUTHORIZED_NEXT
CLASSIFICATION = "nonperiodic_native_global_AP_boundary_action_structure_manifest_frozen"
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizzu1_nonperiodic_native_global_AP_boundary_action_"
    "structure_certificate"
)
PASS_NEXT = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizzv_cycle_physical_driver_branch_"
    "and_event_interpolator_manifest"
)
ARTIFACT = (
    "causal_inner_nonperiodic_native_global_ap_boundary_action_structure_manifest_"
    "wp10c9d6c7c3b5c4f25fizzu"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONPERIODIC_NATIVE_GLOBAL_AP_"
    "BOUNDARY_ACTION_STRUCTURE_MANIFEST_WP10C9D6C7C3B5C4F25FIZZU_2026-08-27.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_nonperiodic_native_global_ap_boundary_action_"
    "structure_manifest_wp10c9d6c7c3b5c4f25fizzu.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonperiodic_native_global_ap_boundary_action_"
    "structure_manifest_wp10c9d6c7c3b5c4f25fizzu.py"
)
PARENT_SHA256 = "f852d2b520700e81444c5472d2c00d426c133111de4247739362d0f8fb1e8a1c"
PORT_SHA256 = "4b491cbba2440f6106da7ae69c54c494ecaa5c15137f8fd4aa808cf305b3d9c6"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _u():
    return parent._u()


def _validate_parents(*, require_clean: bool = False):
    utility = _u()
    if utility._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != PARENT_SHA256:
        raise RuntimeError("cycle physical-input validator changed")
    if utility._sha256(ports.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != PORT_SHA256:
        raise RuntimeError("native prefix-port payload changed")
    input_hashes = utility._validate_checksums(parent.CANONICAL_DIRECTORY)
    port_hashes = utility._validate_checksums(ports.CANONICAL_DIRECTORY)
    input_summary = utility._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    port_summary = utility._read_json(ports.CANONICAL_DIRECTORY / "summary.json")
    if (
        not input_summary["passed"]
        or not input_summary["input_schema_and_validator_certified"]
        or input_summary["physical_model_complete"]
        or input_summary["authorized_next"] != WORK_PACKAGE
        or input_summary["complete_cycle_execution_authorized"]
        or not port_summary["passed"]
        or not port_summary["eleven_field_boundary_structure_certified"]
        or port_summary["complete_cycle_execution_authorized"]
    ):
        raise RuntimeError("nonperiodic boundary-action parents changed classification")
    if require_clean and utility._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("nonperiodic boundary-action manifest needs a clean tracked tree")
    return input_hashes, port_hashes


def _contract() -> dict:
    return {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "native_state": {
            "radial_cells": 112,
            "entropy_fields_per_cell": 11,
            "global_dimension": 1232,
            "entropy_weight": "H=diag(m_i I_11), m_i>0 from normalized native cell measures",
            "principal_and_source_payload": "one frozen certified 11-field port per native cell",
        },
        "sbp_split_form": {
            "difference": "Q_ii=(-1/2,0,...,0,+1/2), Q_i,i+1=+1/2, Q_i+1,i=-1/2",
            "identities": ["Q+Q^T=B=diag(-1,0,...,0,+1)", "Q 1=0"],
            "principal_operator": "L_A=-(1/2)H^-1[(Q tensor I)A+A(Q tensor I)]",
            "interior_viscosity": "L_D=-H^-1 G^T R G, R_face=nu_face I_11, nu_face>=rho((A_i+A_i+1)/2)",
            "source_operator": "L_S=blockdiag(S_i), symmetric(S_i)<=0",
            "coefficient_policy": "all A_i,S_i,H,R and boundary eigenspaces frozen within one accepted action",
        },
        "maximally_dissipative_sat": {
            "outward_matrix": "A_n=n A_boundary",
            "incoming_penalty": "P=(-A_n)_+",
            "homogeneous_sat": "-H^-1 E_boundary P E_boundary^T z",
            "affine_control": "+H^-1 E_boundary P V_in a_in",
            "inner_edge": "pure excision: zero incoming eigenvalues, zero SAT control columns",
            "outer_edge": "eleven incoming entropy-characteristic amplitudes",
        },
        "binding_energy_identity": {
            "formula": (
                "H L+L^T H = -sum_boundary E |A_n| E^T - 2 G^T R G "
                "+ H(L_S+L_S^T)"
            ),
            "interpretation": "homogeneous action is nonexpansive in the H entropy norm",
            "affine_power": "outer loading contributes 2 z^T H B_out a_in and is reported separately",
            "no_sample_only_claim": True,
        },
        "certificate": {
            "native_profile": "selected_profile_index=0 at every one of 112 cells",
            "minimum_native_cells": 112,
            "maximum_sbp_identity_defect": 5.0e-14,
            "maximum_energy_identity_relative_defect": 5.0e-12,
            "maximum_homogeneous_entropy_growth_eigenvalue": 5.0e-11,
            "maximum_affine_action_defect": 5.0e-12,
            "minimum_source_nullity": 4,
            "inner_incoming_count": 0,
            "outer_incoming_count": 11,
            "checkpoint_roundtrip_bitwise": True,
            "complete_cycle_steps": 0,
        },
        "scientific_boundary": {
            "certifies": [
                "native 1232-state nonperiodic sparse action structure",
                "pure inner excision",
                "affine eleven-characteristic outer loading",
                "exact semidiscrete entropy identity",
            ],
            "does_not_certify": [
                "physical cycle-wide driver payload",
                "radial accuracy or nonlinear atlas interpolation",
                "physical event calibration or heldout cycle prediction",
                "complete-cycle execution",
            ],
            "complete_cycle_execution_authorized": False,
        },
        "decision": {
            "pass_classification": "nonperiodic_native_global_AP_boundary_action_structure_certified",
            "failure_classification": "nonperiodic_native_global_AP_boundary_action_structure_failed",
            "pass_authorized_next": PASS_NEXT,
        },
        "authorized_next": AUTHORIZED_NEXT,
    }


def _update(summary: dict) -> None:
    utility = _u()
    rows = list(csv.DictReader(CANONICAL_MANIFEST.open(newline="", encoding="utf-8")))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": utility._sha256(path), "scientific_status": "SUPPORTED"})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    catalog = utility._read_json(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": CLASSIFICATION, "passed": True}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": utility._git("rev-parse", "HEAD"), "latest_work_package": WORK_PACKAGE})
    utility._write_json(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("nonperiodic boundary-action manifest already exists")
    input_hashes, port_hashes = _validate_parents(require_clean=True)
    utility = _u(); CANONICAL_DIRECTORY.mkdir(parents=True)
    utility._write_json(CANONICAL_DIRECTORY / "boundary_action_contract.json", _contract())
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "input_schema_and_validator_preserved": True,
        "native_boundary_payload_preserved": True,
        "nonperiodic_global_AP_boundary_action_certified": False,
        "physical_model_complete": False,
        "complete_cycle_execution_authorized": False,
        "complete_cycle_steps": 0,
        "authorized_next": AUTHORIZED_NEXT,
    }
    utility._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utility._write_json(CANONICAL_DIRECTORY / "input_lock.json", {
        "physical_input_parent": parent.ARTIFACT,
        "physical_input_checksum_manifest_sha256": PARENT_SHA256,
        "physical_input_hashes": input_hashes,
        "prefix_port_parent": ports.ARTIFACT,
        "prefix_port_checksum_manifest_sha256": PORT_SHA256,
        "prefix_port_hashes": port_hashes,
    })
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# Nonperiodic native global AP boundary-action structure manifest\n\n"
        f"Classification: `{CLASSIFICATION}`.\n\n"
        "The prospective native operator is an SBP split form with positive jump viscosity and maximally dissipative SAT boundaries. Its homogeneous weighted entropy identity is exact: the interior contributes only certified dissipation, the inner edge remains pure excision, and the outer edge accepts eleven affine incoming entropy-characteristic amplitudes.\n\n"
        "This package is definitions-only. Physical driver data remain absent, radial-accuracy and nonlinear interpolation are not claimed, and no complete-cycle step is authorized.\n",
        encoding="utf-8",
    )
    sources = (THIS_RUNNER, THIS_TEST, REPORT_RELATIVE)
    utility._write_json(CANONICAL_DIRECTORY / "provenance.json", {"implementation_commit": utility._git("rev-parse", "HEAD"), "source_hashes": {name: utility._sha256(ROOT / name) for name in sources}})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utility._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8")
    _update(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--freeze", action="store_true"); args = parser.parse_args()
    if not args.freeze: parser.error("choose --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
