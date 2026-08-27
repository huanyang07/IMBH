#!/usr/bin/env python3
"""Freeze the exact dynamic-height Legendre convexity diagnostic."""

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

import run_causal_inner_equilibrium_selected_metric_stencil_full_rerun_wp10c9d6c7c3b5c4f25fizzc6 as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = parent.AUTHORIZED_NEXT
CLASSIFICATION = "dynamic_height_convex_legendre_manifest_frozen"
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizzd1_"
    "dynamic_height_physical_entropy_convexity_diagnostic"
)
FAILURE_NEXT = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizze_"
    "split_Godunov_port_Hamiltonian_architecture_manifest"
)
PASS_NEXT = (
    "WP10c9d6c7c3b5c4f25fizzd2_"
    "dynamic_height_common_potential_implementation"
)
ARTIFACT = (
    "causal_inner_dynamic_height_convex_legendre_manifest_"
    "wp10c9d6c7c3b5c4f25fizzd"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_DYNAMIC_HEIGHT_CONVEX_"
    "LEGENDRE_MANIFEST_WP10C9D6C7C3B5C4F25FIZZD_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_dynamic_height_convex_legendre_manifest_"
    "wp10c9d6c7c3b5c4f25fizzd.py"
)
THIS_TEST = (
    "tests/test_causal_inner_dynamic_height_convex_legendre_manifest_"
    "wp10c9d6c7c3b5c4f25fizzd.py"
)
PARENT_SHA256 = "86d2f9410100896fb023573dc7a283668f4e3e0fc490df2a7e8e95f87f3d0167"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _u():
    return parent._u()


def _validate_parent(*, require_clean: bool = False) -> dict:
    utils = _u()
    checksum = parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt"
    if utils._sha256(checksum) != PARENT_SHA256:
        raise RuntimeError("fixed-height certificate checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    if (
        not summary["passed"]
        or not summary["equilibrium_physical_potential_certified"]
        or summary["dynamic_height_potential_certified"]
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["complete_cycle_execution_authorized"]
    ):
        raise RuntimeError("fixed-height authorization changed")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("dynamic-height manifest needs a clean tracked tree")
    return hashes


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "parent_fixed_height_potential_certified": True,
        "candidate_common_potential": {
            "rest_frame_conserved_coordinates": (
                "surface_mass, thermal_plus_vertical_energy, "
                "Z_H=surface_mass*H, P_H=surface_mass*w_H"
            ),
            "density": "rho=surface_mass**2/(2*Z_H)",
            "thermal_energy": (
                "surface_mass*(R*T/(gamma-1)+a_rad*T**4/rho)"
            ),
            "vertical_energy": (
                "P_H**2/(2*surface_mass)"
                "+0.5*Omega_perp**2*Z_H**2/surface_mass"
            ),
            "mathematical_entropy": "eta=-surface_mass*s(rho,T)",
            "specific_entropy": (
                "R/(gamma-1)*ln(T/T0)-R*ln(rho/rho0)"
                "+4*a_rad*T**3/(3*rho)"
            ),
            "height_affinity": "eta_H=d eta/d Z_H",
            "necessary_legendre_condition": (
                "the Hessian d2 eta/dU_H2 is positive definite"
            ),
            "hydrostatic_witness": (
                "P_H=0 and Pi=surface_mass*Omega_perp**2*H**2"
            ),
            "force_identity": (
                "-dE/dH|surface_mass,entropy,P_H"
                "=Pi/H-surface_mass*Omega_perp**2*H"
            ),
        },
        "diagnostic": {
            "same_frozen_physical_witnesses": 47,
            "no_coefficient_fit": True,
            "linear_coordinate_scales": (
                "surface_mass, surface_mass*e_thermal, Z_H, surface_mass*c"
            ),
            "centered_Hessian_step_factors": [0.002, 0.001, 0.0005],
            "diagonal_equilibration": True,
            "positive_minimum_eigenvalue_gate": 1.0e-8,
            "Hessian_symmetry_gate": 1.0e-11,
            "force_identity_relative_gate": 1.0e-10,
            "stable_sign_rule": (
                "the minimum eigenvalue must have the same sign at all "
                "three step factors for every classified witness"
            ),
            "trajectory_steps": 0,
        },
        "decision": {
            "pass": {
                "classification": "dynamic_height_common_potential_convex",
                "authorized_next": PASS_NEXT,
            },
            "fail": {
                "classification": (
                    "dynamic_height_common_potential_convexity_obstructed"
                ),
                "authorized_next": FAILURE_NEXT,
                "interpretation": (
                    "reject only the one-piece common-potential height "
                    "completion; preserve the fixed-height potential"
                ),
            },
        },
        "claim_boundary": {
            "definitions_only": True,
            "dynamic_height_potential_certified": False,
            "full_shear_master_potential_certified": False,
            "eleven_field_local_closure_certified": False,
            "trajectory_authorized": False,
            "complete_cycle_execution_authorized": False,
        },
        "authorized_next": AUTHORIZED_NEXT,
    }


def _update_catalog(summary: dict) -> None:
    utils = _u()
    with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": utils._sha256(path),
                    "scientific_status": "SUPPORTED",
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case", "path", "bytes", "sha256", "scientific_status"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = utils._read_json(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": True,
    }
    catalog.update(
        {
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": utils._git("rev-parse", "HEAD"),
            "latest_work_package": WORK_PACKAGE,
        }
    )
    utils._write_json(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("dynamic-height Legendre manifest already exists")
    hashes = _validate_parent(require_clean=True)
    utils = _u()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    contract = _contract()
    utils._write_json(CANONICAL_DIRECTORY / "convexity_contract.json", contract)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "fixed_height_potential_certified": True,
        "dynamic_height_potential_certified": False,
        "trajectory_authorized": False,
        "complete_cycle_execution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utils._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_artifact": parent.ARTIFACT,
            "parent_checksum_manifest_sha256": PARENT_SHA256,
            "parent_hashes": hashes,
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# Dynamic-height convex Legendre diagnostic manifest\n\n"
        f"Classification: `{CLASSIFICATION}`.\n\n"
        "The exact fixed-height gas+radiation potential remains certified. "
        "This package prospectively tests whether the physical column entropy "
        "is strictly convex after promoting `Z_H=surface_mass*H` and vertical "
        "momentum. Positive convexity is a necessary condition for the proposed "
        "one-piece Legendre master potential.\n\n"
        "A failure rejects only that common-potential height completion and "
        "authorizes a definitions-only split Godunov/port-Hamiltonian "
        "architecture. No trajectory or complete-cycle execution is authorized.\n",
        encoding="utf-8",
    )
    sources = (THIS_RUNNER, THIS_TEST, REPORT_RELATIVE)
    utils._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "implementation_commit": utils._git("rev-parse", "HEAD"),
            "source_hashes": {
                path: utils._sha256(ROOT / path) for path in sources
            },
        },
    )
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{utils._sha256(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if not args.freeze:
        parser.error("choose --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
