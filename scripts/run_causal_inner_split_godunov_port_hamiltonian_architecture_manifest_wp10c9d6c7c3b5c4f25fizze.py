#!/usr/bin/env python3
"""Freeze the split Godunov/port-Hamiltonian eleven-field architecture."""

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

import run_causal_inner_dynamic_height_physical_entropy_convexity_diagnostic_wp10c9d6c7c3b5c4f25fizzd1 as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = parent.manifest.FAILURE_NEXT
CLASSIFICATION = "split_Godunov_port_Hamiltonian_architecture_manifest_frozen"
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizze1_"
    "split_Godunov_port_Hamiltonian_proof_kernel"
)
PASS_NEXT = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizzf_"
    "fixed_height_five_STF_master_potential_manifest"
)
ARTIFACT = (
    "causal_inner_split_godunov_port_hamiltonian_architecture_manifest_"
    "wp10c9d6c7c3b5c4f25fizze"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_SPLIT_GODUNOV_PORT_HAMILTONIAN_"
    "ARCHITECTURE_MANIFEST_WP10C9D6C7C3B5C4F25FIZZE_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_split_godunov_port_hamiltonian_architecture_"
    "manifest_wp10c9d6c7c3b5c4f25fizze.py"
)
THIS_TEST = (
    "tests/test_causal_inner_split_godunov_port_hamiltonian_architecture_"
    "manifest_wp10c9d6c7c3b5c4f25fizze.py"
)
PARENT_SHA256 = "14742c4806473cc90650ca72ea59056c6ea3b3c930d212ff1f1498c470009630"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _u():
    return parent._u()


def _validate_parent(*, require_clean: bool = False) -> dict:
    utils = _u()
    checksum = parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt"
    if utils._sha256(checksum) != PARENT_SHA256:
        raise RuntimeError("dynamic-height rejection checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    if (
        summary["passed"]
        or not summary["audit_completed"]
        or not summary["fixed_height_physical_potential_preserved"]
        or not summary["split_architecture_manifest_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["complete_cycle_execution_authorized"]
    ):
        raise RuntimeError("dynamic-height rejection classification changed")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("split architecture manifest needs a clean tracked tree")
    return hashes


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "preserved_results": {
            "fixed_height_equilibrium_potential_certified": True,
            "five_STF_basis_and_quadratic_normal_form_certified": True,
            "one_piece_dynamic_height_common_potential_rejected": True,
            "rejected_candidate_reintroduced": False,
        },
        "field_decomposition": {
            "total_fields": 11,
            "Godunov_transport_fields": 9,
            "transport_coordinates": (
                "surface mass, radial momentum, angular momentum, total "
                "energy, five STF shear amplitudes"
            ),
            "vertical_port_fields": 2,
            "vertical_coordinates": "log(H/H_anchor), w_H/c",
            "field_count_identity": "4 equilibrium + 5 STF + 2 vertical = 11",
        },
        "transport_generator": {
            "height_policy": (
                "H is a frozen positive coefficient during each hyperbolic "
                "transport substep"
            ),
            "equilibrium_potential": "X_eq^mu=2*H*p*beta^mu",
            "shear_completion": (
                "a prospective five-STF common potential in the nine-field "
                "transport block only"
            ),
            "requirements": (
                "A_G^0 positive definite; A_G^r symmetric; full radial "
                "spectrum real and strictly causal"
            ),
        },
        "vertical_port_generator": {
            "height_kinematics": "D ln(H)=w_H/H",
            "physical_force_per_area": (
                "F_H=Pi/H-surface_mass*Omega_perp**2*H"
            ),
            "momentum": "D P_H=F_H-gamma_H*P_H",
            "damping": "gamma_H=alpha*Omega_perp",
            "reversible_operator": "J_H=-J_H^T",
            "dissipative_operator": "R_H=R_H^T positive_semidefinite",
            "energy_ledger": (
                "reversible pressure/gravity work plus kinetic work is zero; "
                "gamma_H*P_H**2/surface_mass is deposited as heat"
            ),
            "entropy_ledger": (
                "reversible contribution is zero and damping produces "
                "gamma_H*P_H**2/(surface_mass*T)>=0"
            ),
        },
        "composition": {
            "method": "symmetric V(dt/2) G(dt) V(dt/2) Strang composition",
            "state_dependent_coefficients": (
                "recomputed only at declared substep boundaries; no hidden "
                "lagging inside a substep"
            ),
            "accepted_history_only": True,
            "discrete_total_energy_ledger_binding": True,
            "discrete_physical_entropy_non_decrease_binding": True,
            "matched_endpoint_half_step_audit_binding": True,
        },
        "proof_kernel": {
            "same_47_physical_witnesses": True,
            "abstract_nine_field_Godunov_block": (
                "the certified vertical-equilibrium restriction of the "
                "eleven-field quadratic normal form"
            ),
            "physical_vertical_frequency_squared": (
                "Omega_H**2=Omega_perp**2+R*T/H**2"
            ),
            "height_rate_matrix": (
                "L_H=[[0,c/H],[-H*Omega_H**2/c,-gamma_H]]"
            ),
            "height_entropy_metric": (
                "M_H=diag(H*Omega_H**2/c,c/H)"
            ),
            "gates": {
                "temporal_minimum_eigenvalue": 1.0e-10,
                "matrix_symmetry_relative": 1.0e-12,
                "maximum_characteristic_speed_over_c": 0.999,
                "port_skew_relative": 1.0e-12,
                "source_entropy_positive_part": 1.0e-12,
                "vertical_energy_ledger_relative": 1.0e-12,
            },
            "trajectory_steps": 0,
        },
        "decision": {
            "pass_classification": "split_Godunov_port_Hamiltonian_kernel_certified",
            "pass_authorized_next": PASS_NEXT,
            "failure_classification": "split_Godunov_port_Hamiltonian_kernel_failed",
            "failure_authorized_next": None,
        },
        "claim_boundary": {
            "definitions_only": True,
            "full_shear_physical_potential_certified": False,
            "split_discretization_certified": False,
            "trajectory_authorized": False,
            "cycle_execution_authorized": False,
        },
        "authorized_next": AUTHORIZED_NEXT,
    }


def _update(summary: dict) -> None:
    utils = _u()
    with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": utils._sha256(path), "scientific_status": "SUPPORTED"})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    catalog = utils._read_json(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": CLASSIFICATION, "passed": True}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": utils._git("rev-parse", "HEAD"), "latest_work_package": WORK_PACKAGE})
    utils._write_json(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("split architecture manifest already exists")
    hashes = _validate_parent(require_clean=True)
    utils = _u(); CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "architecture_contract.json", _contract())
    summary = {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "classification": CLASSIFICATION, "passed": True, "definitions_only": True, "one_piece_height_rejection_preserved": True, "split_kernel_certified": False, "trajectory_authorized": False, "complete_cycle_execution_authorized": False, "authorized_next": AUTHORIZED_NEXT}
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utils._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"parent_artifact": parent.ARTIFACT, "parent_checksum_manifest_sha256": PARENT_SHA256, "parent_hashes": hashes})
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# Split Godunov/port-Hamiltonian architecture manifest\n\n"
        f"Classification: `{CLASSIFICATION}`.\n\n"
        "The eleven fields are split into a nine-field fixed-height Godunov "
        "transport block and a two-field local vertical port. The reversible "
        "vertical oscillator is skew with respect to its positive energy metric; "
        "damping transfers vertical energy to heat. Symmetric Strang composition "
        "replaces the rejected one-piece height Legendre potential.\n\n"
        "This package is definitions-only. It authorizes only the algebraic and "
        "physical proof kernel, not a trajectory or cycle execution.\n",
        encoding="utf-8",
    )
    sources = (THIS_RUNNER, THIS_TEST, REPORT_RELATIVE)
    utils._write_json(CANONICAL_DIRECTORY / "provenance.json", {"implementation_commit": utils._git("rev-parse", "HEAD"), "source_hashes": {path: utils._sha256(ROOT / path) for path in sources}})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utils._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8")
    _update(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--freeze", action="store_true"); args = parser.parse_args()
    if not args.freeze: parser.error("choose --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
