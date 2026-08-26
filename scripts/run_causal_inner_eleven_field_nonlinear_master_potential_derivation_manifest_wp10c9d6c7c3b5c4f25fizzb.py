#!/usr/bin/env python3
"""Freeze the staged nonlinear eleven-field master-potential derivation."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import sys


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_eleven_field_stf_convex_normal_form_implementation_wp10c9d6c7c3b5c4f25fizza as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = parent.AUTHORIZED_NEXT
CLASSIFICATION = "eleven_field_nonlinear_master_potential_derivation_manifest_frozen"
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizzc_"
    "equilibrium_column_thermodynamic_potential_implementation"
)
ARTIFACT = (
    "causal_inner_eleven_field_nonlinear_master_potential_derivation_manifest_"
    "wp10c9d6c7c3b5c4f25fizzb"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ELEVEN_FIELD_NONLINEAR_MASTER_"
    "POTENTIAL_DERIVATION_MANIFEST_"
    "WP10C9D6C7C3B5C4F25FIZZB_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_eleven_field_nonlinear_master_potential_"
    "derivation_manifest_wp10c9d6c7c3b5c4f25fizzb.py"
)
THIS_TEST = (
    "tests/test_causal_inner_eleven_field_nonlinear_master_potential_"
    "derivation_manifest_wp10c9d6c7c3b5c4f25fizzb.py"
)
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "b24a5f936f6e1e5d82916248bda38a52c8526dfc2147d208a8264bcc6d7ef861"
)
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _utils():
    return parent._utils()


def _validate_parent(*, require_clean: bool) -> dict:
    utils = _utils()
    if (
        utils._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt")
        != PARENT_CHECKSUM_MANIFEST_SHA256
    ):
        raise RuntimeError("STF normal-form certificate checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = utils._read_json(
        parent.CANONICAL_DIRECTORY / "certificate_metrics.json"
    )
    if (
        summary["classification"] != parent.PASS_CLASSIFICATION
        or not summary["passed"]
        or not summary["seven_field_rejection_preserved"]
        or not summary["five_STF_basis_certified"]
        or not summary["quadratic_convex_normal_form_certified"]
        or summary["nonlinear_physical_master_potential_derived"]
        or summary["eleven_field_physical_closure_certified"]
        or summary["complete_cycle_execution_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or metrics["fixture_is_physical_calibration"]
    ):
        raise RuntimeError("STF normal-form claim boundary changed")
    for relative, expected in utils._read_json(
        parent.CANONICAL_DIRECTORY / "provenance.json"
    )["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"STF normal-form source changed: {relative}")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("master-potential manifest needs a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "metrics": metrics}


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "failure_avoidance": {
            "preserved_negative_control": (
                "the prior seven-field conservative D*chi candidate has an "
                "order-unity temperature-shear entropy-flux curl"
            ),
            "dominant_failed_pair": "log_temperature,specific_shear_stress",
            "minimum_failed_relative_curl": 1.3683589179402538,
            "root_cause": (
                "a state-dependent shear modulus was inserted after the "
                "thermodynamic potential, omitting its reciprocal mixed derivatives"
            ),
            "repair_rule": (
                "choose the master scalar first in entropy variables and derive "
                "every current and coefficient derivative from it"
            ),
            "post_hoc_symmetrization_forbidden": True,
        },
        "entropy_variables_and_invariants": {
            "mass_affinity": "alpha=mu_mass/T",
            "inverse_temperature_four_vector": "beta_mu=u_mu/T",
            "height_affinity": "eta_H conjugate to Z_H=surface_mass*H",
            "shear_affinity": "zeta_<mu nu> represented by five STF amplitudes",
            "temperature": "T=(-beta_mu*beta^mu)^(-1/2)",
            "invariants": {
                "mu": "beta_mu*beta^mu<0",
                "nu": "zeta^{mu nu} beta_mu beta_nu",
                "I2": "zeta_mu_nu*zeta^{mu nu}",
                "I3": "zeta^mu_nu*zeta^nu_rho*zeta^rho_mu",
            },
            "moving_STF_basis_differentiated": True,
        },
        "equilibrium_thermodynamic_control": {
            "volume_EOS": (
                "p=rho*R*T+a_rad*T^4/3; e=R*T/(gamma-1)+a_rad*T^4/rho"
            ),
            "specific_entropy": (
                "s=R/(gamma-1) ln(T/T0)-R ln(rho/rho0)+4a_rad*T^3/(3rho)"
            ),
            "specific_chemical_potential": "mu_mass=c^2+e+p/rho-T*s",
            "exact_inversion": (
                "ln(rho/rho0)=alpha/R-c^2/(R*T)-gamma/(gamma-1)"
                "+ln(T/T0)/(gamma-1)"
            ),
            "fixed_height_potential_current": "X_eq^mu=2H*p(alpha,T)*beta^mu",
            "required_derivatives": (
                "dX_eq^mu/dalpha=surface-mass current; "
                "dX_eq^mu/dbeta_nu=perfect-fluid column stress-energy"
            ),
            "gas_radiation_first_law_and_Gibbs_Duhem_binding": True,
            "purpose": (
                "certify the exact equilibrium thermodynamic sector before adding "
                "height dynamics or dissipative invariants"
            ),
        },
        "height_completion": {
            "state": "Z_H=surface_mass*H",
            "vertical_momentum": "the beta_z stress-energy component",
            "total_column_energy_terms": (
                "thermal+orbital+P_H^2/(2*surface_mass)+"
                "0.5*surface_mass*Omega_perp^2*H^2"
            ),
            "construction": (
                "convex Legendre extension in eta_H; derive J_H from the same "
                "potential current rather than appending D*H advection"
            ),
            "hydrostatic_force": "Pi/H-surface_mass*Omega_perp^2*H",
            "vertical_exchange_entropy_neutral": True,
            "vertical_damping_heats_internal_energy": True,
        },
        "full_shear_master_scalar": {
            "prospective_ansatz": (
                "chi=chi_eq(alpha,mu,eta_H)+c1(alpha,mu,eta_H)*nu+"
                "c2(alpha,mu,eta_H)*I2+c3(alpha,mu,eta_H)*I3+"
                "c4(alpha,mu,eta_H)*nu^2+higher_convex_barrier"
            ),
            "potential_current": "chi^mu=dchi/dbeta_mu",
            "currents": (
                "N^mu=dchi^mu/dalpha; T^{mu nu}=dchi^mu/dbeta_nu; "
                "J_H^mu=dchi^mu/deta_H; A^{mu<nu rho>}=dchi^mu/dzeta_<nu rho>"
            ),
            "physical_linear_stress": (
                "the c1*nu mixed derivative supplies the complete five-component "
                "linear shear stress in T^{mu nu}"
            ),
            "nonlinear_terms": (
                "I2,I3 and nu^2 retain amplitude dependence and provide the "
                "thermodynamic terms absent from the rejected D*chi construction"
            ),
            "coefficient_calibration_after_differentiation": True,
            "one_Rphi_projection_forbidden": True,
        },
        "derivation_sequence": (
            "exact fixed-height equilibrium gas+radiation potential",
            "dynamic-height convex Legendre completion",
            "five-STF shear invariant extension",
            "source/entropy and full-tensor constraint propagation",
            "complete nonlinear local structural audit",
        ),
        "stage_gates": {
            "equilibrium": (
                "current parity<=1e-10; first-law/Gibbs-Duhem<=1e-11; "
                "independent derivative parity<=1e-9"
            ),
            "height": (
                "A0 positive; exact height current; hydrostatic force/energy ledger<=1e-10"
            ),
            "shear": (
                "all mixed Hessians symmetric; old Rphi near-equilibrium parity; "
                "five-amplitude roundtrip; source entropy nonpositive"
            ),
            "complete_local": (
                "A0 positive, radial Hessian symmetric, full spectrum real/causal, "
                "Cordeiro nonlinear inequalities, constraints propagated on every "
                "frozen envelope state"
            ),
            "fail_closed_at_first_failed_stage": True,
        },
        "frozen_envelope_for_complete_local_audit": {
            "canonical_base_charts": 8401,
            "deterministic_witnesses": 47,
            "includes": (
                "20ms primary; 16ms held-out; all accepted seven-field predecessor "
                "profiles; 179ms endpoint; 179.125ms cell-6 witness"
            ),
            "coefficient_fit_to_witness_forbidden": True,
            "envelope_frozen_before_physical_eigenvalue_results": True,
        },
        "next_package": {
            "scope": AUTHORIZED_NEXT,
            "fixed_height_only": True,
            "exact_physical_gas_radiation_EOS": True,
            "derive_mass_and_stress_energy_currents": True,
            "independent_derivative_ladder": True,
            "add_height_or_shear_terms": False,
            "trajectory": False,
        },
        "claim_boundary": {
            "STF_and_quadratic_normal_form_certified": True,
            "equilibrium_physical_potential_certified": False,
            "dynamic_height_potential_certified": False,
            "full_shear_master_potential_certified": False,
            "eleven_field_local_closure_certified": False,
            "eleven_field_trajectory_authorized": False,
            "complete_cycle_execution_authorized": False,
        },
        "budget": {
            "truth_operator_calls": 0,
            "trajectory_steps": 0,
            "complete_cycle_steps": 0,
            "maximum_wall_minutes": 10,
        },
        "authorized_next": AUTHORIZED_NEXT,
    }


def _update_catalog(summary: dict) -> None:
    utils = _utils()
    with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({
                "case": ARTIFACT,
                "path": str(path.relative_to(ROOT)),
                "bytes": str(path.stat().st_size),
                "sha256": utils._sha256(path),
                "scientific_status": "DEFINITIONS_ONLY",
            })
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    catalog = utils._read_json(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": True,
    }
    catalog.update({
        "case_count": len({row["case"] for row in rows}),
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "all_payload_hashes_recorded": True,
        "latest_source_parent_commit": utils._git("rev-parse", "HEAD"),
        "latest_work_package": WORK_PACKAGE,
    })
    utils._write_json(CANONICAL_SUMMARY, catalog)


def _canonicalize() -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("master-potential derivation manifest already exists")
    validated = _validate_parent(require_clean=True)
    utils = _utils()
    contract = _contract()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "derivation_contract.json", contract)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "prior_entropy_curl_failure_preserved": True,
        "STF_and_quadratic_normal_form_certified": True,
        "nonlinear_physical_master_potential_derived": False,
        "eleven_field_physical_closure_certified": False,
        "eleven_field_trajectory_authorized": False,
        "complete_cycle_execution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utils._write_json(CANONICAL_DIRECTORY / "input_lock.json", {
        "parent_artifact": parent.ARTIFACT,
        "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256,
        "parent_hashes": validated["hashes"],
        "preserved_seven_field_entropy_failure_artifact": (
            "causal_inner_seven_field_physical_closure_local_structural_audit_"
            "wp10c9d6c7c3b5c4f25fizec"
        ),
    })
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join((
        "# Eleven-field nonlinear master-potential derivation manifest",
        "",
        f"Classification: `{CLASSIFICATION}`.",
        "",
        "The derivation begins with the exact gas+radiation thermodynamic potential in entropy variables. Dynamic height and the five shear invariants are added only after the equilibrium current identities pass independently.",
        "",
        "This ordering prevents recurrence of the prior order-unity temperature-shear curl: no state-dependent modulus may be inserted after differentiating the potential. Every reciprocal thermodynamic term must be generated automatically as a mixed derivative of the same master scalar.",
        "",
        "This package is definitions-only. The physical master potential, local eleven-field closure, trajectory, and complete cycle remain uncertified and unauthorized.",
        "",
        f"Authorized next: `{AUTHORIZED_NEXT}`.",
        "",
    )), encoding="utf-8")
    sources = (THIS_RUNNER, THIS_TEST, REPORT_RELATIVE)
    utils._write_json(CANONICAL_DIRECTORY / "provenance.json", {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "implementation_commit": utils._git("rev-parse", "HEAD"),
        "implementation_tree": utils._git("rev-parse", "HEAD^{tree}"),
        "source_hashes": {path: utils._sha256(ROOT / path) for path in sources},
        "python": sys.version,
        "platform": platform.platform(),
        "thread_environment": {name: os.environ.get(name, "") for name in (
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"
        )},
    })
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(
        f"{utils._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names
    ), encoding="utf-8")
    _update_catalog(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if not args.freeze:
        parser.error("choose --freeze")
    print(json.dumps(_canonicalize(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
