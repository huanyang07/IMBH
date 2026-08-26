#!/usr/bin/env python3
"""Freeze the eleven-field full-shear convex/divergence architecture."""

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

import run_causal_inner_projected_shear_hyperbolicity_blocker_certificate_wp10c9d6c7c3b5c4f25fizy as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = parent.AUTHORIZED_NEXT
CLASSIFICATION = "eleven_field_convex_divergence_architecture_manifest_frozen"
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizza_"
    "eleven_field_stf_convex_normal_form_implementation"
)
ARTIFACT = (
    "causal_inner_eleven_field_convex_divergence_architecture_manifest_"
    "wp10c9d6c7c3b5c4f25fizz"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ELEVEN_FIELD_CONVEX_"
    "DIVERGENCE_ARCHITECTURE_MANIFEST_"
    "WP10C9D6C7C3B5C4F25FIZZ_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_eleven_field_convex_divergence_architecture_"
    "manifest_wp10c9d6c7c3b5c4f25fizz.py"
)
THIS_TEST = (
    "tests/test_causal_inner_eleven_field_convex_divergence_architecture_"
    "manifest_wp10c9d6c7c3b5c4f25fizz.py"
)
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "b7f89790881be572d67b6bf222ec4098b2f9e31e428ca0353bc1da9d1e30d313"
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
        raise RuntimeError("projected-shear blocker certificate checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = utils._read_json(
        parent.CANONICAL_DIRECTORY / "certificate_metrics.json"
    )
    if (
        summary["classification"] != parent.PASS_CLASSIFICATION
        or not summary["passed"]
        or not summary["seven_field_rejection_preserved"]
        or not summary["one_amplitude_projected_shear_closure_rejected"]
        or not summary["full_five_component_shear_completion_selected"]
        or summary["eleven_field_physical_closure_certified"]
        or summary["complete_cycle_execution_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or not metrics["derivative_stable_complex_pair"]
        or not metrics["full_tensor_screen_passed"]
        or metrics["selected_local_field_count"] != 11
    ):
        raise RuntimeError("projected-shear blocker decision changed")
    for relative, expected in utils._read_json(
        parent.CANONICAL_DIRECTORY / "provenance.json"
    )["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"projected-shear blocker source changed: {relative}")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("eleven-field architecture freeze needs a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "metrics": metrics}


def _contract() -> dict:
    """Return the prospective mathematical and execution contract."""

    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "decision_basis": {
            "seven_field_one_amplitude_closure_rejected": True,
            "failure_is_derivative_stable": True,
            "full_tensor_causality_screen_at_witness_positive": True,
            "witness_cell": 6,
            "witness_radius_cm": 3035196434.9786267,
            "no_failed_candidate_may_be_propagated": True,
        },
        "state_architecture": {
            "dimension": 11,
            "column_conserved_backbone": (
                "surface rest mass; radial momentum; angular momentum; "
                "total column energy"
            ),
            "vertical_completion": (
                "height content Z_H=surface_mass*H and vertical momentum "
                "P_H=surface_mass*w_H"
            ),
            "dissipative_completion": (
                "five independent amplitudes of the spatial, symmetric, "
                "tracefree shear tensor in the instantaneous fluid rest frame"
            ),
            "primitive_chart": (
                "lnSigma,beta_R,beta_phi,lnT,lnH,beta_H,zeta_1,...,zeta_5"
            ),
            "entropy_variables": (
                "lambda=(alpha,beta_mu,eta_H,zeta_<mu nu>) with dimensions "
                "1+4+1+5; beta_z is conjugate to vertical momentum"
            ),
            "one_Rphi_projection_forbidden": True,
            "algebraic_height_foldback_forbidden": True,
        },
        "covariant_shear_representation": {
            "rest_frame_triad": "e_R,e_phi,e_z orthonormal and orthogonal to u",
            "basis": (
                "five Frobenius-orthonormal symmetric-tracefree tensors: "
                "(RR-pp)/sqrt(2), (RR+pp-2zz)/sqrt(6), "
                "sqrt(2)R(p), sqrt(2)R(z), sqrt(2)p(z)"
            ),
            "reconstruction": "pi^{mu nu}=Sigma*c^2*sum_A zeta_A E_A^{mu nu}",
            "constraints": (
                "pi^{mu nu}=pi^{nu mu}; g_mu_nu*pi^{mu nu}=0; "
                "u_mu*pi^{mu nu}=0"
            ),
            "moving_basis_derivatives_included": True,
            "basis_frozen_inside_principal_part": False,
            "constraint_propagation_binding": True,
        },
        "single_master_potential": {
            "variables": "lambda_A=(alpha,beta_mu,eta_H,zeta_A)",
            "scalar": (
                "chi(alpha,beta^2,eta_H,I2,I3,beta_mu*zeta^{mu nu}*beta_nu;g,z)"
            ),
            "potential_current": "chi^mu=dchi/dbeta_mu",
            "mass_current": "N^mu=dchi^mu/dalpha",
            "stress_energy": "T^{mu nu}=dchi^mu/dbeta_nu",
            "height_current": "J_H^mu=dchi^mu/deta_H",
            "shear_current": "A^{mu nu rho}=dchi^mu/dzeta_<nu rho>",
            "column_reduction": (
                "integrate the four-dimensional currents over the proper "
                "vertical column only after differentiating the same master scalar"
            ),
            "temporal_hessian": (
                "A0_AB=-n_mu*d2chi^mu/(dlambda_A dlambda_B) positive definite"
            ),
            "radial_hessian": (
                "AR_AB=s_R_mu*d2chi^mu/(dlambda_A dlambda_B) symmetric"
            ),
            "reciprocity": (
                "fluid-shear, shear-height, and vertical-work derivative "
                "couplings are mixed derivatives of chi and cannot be added post hoc"
            ),
            "nonlinear_invariant_domain": (
                "strictly convex connected component containing equilibrium; "
                "I2 and I3 are retained rather than collapsed to the Rphi amplitude"
            ),
        },
        "physical_calibration": {
            "equilibrium_limit": (
                "exact gas+radiation Kerr-Schild column EOS and perfect-fluid currents"
            ),
            "vertical_energy": (
                "P_H^2/(2*surface_mass)+0.5*surface_mass*Omega_perp^2*H^2"
            ),
            "shear_near_equilibrium": (
                "tau_pi Delta D pi_<mu nu>+pi_mu_nu=2 eta sigma_mu_nu"
            ),
            "alpha_calibration": (
                "the Rphi Navier-Stokes target matches the existing alpha law; "
                "the other four components are not set to zero"
            ),
            "relaxation_heating": (
                "shear and vertical damping heat internal energy and do not delete total energy"
            ),
            "coefficient_policy": (
                "derive eta,tau_pi and nonlinear coefficients from the EOS/transport "
                "law on a prospectively frozen envelope; never fit the cell-6 witness"
            ),
        },
        "source_and_entropy_structure": {
            "balances": "nabla_mu J_A^mu=S_A",
            "internal_source": (
                "S=-M(lambda)*lambda_diss+J_H(lambda)*lambda with "
                "M symmetric positive semidefinite and J_H skew in A0"
            ),
            "entropy_inequality": "lambda_A*S_A<=0 for mathematical entropy",
            "shear_entropy_production": "zeta_A M_AB zeta_B>=0",
            "vertical_hamiltonian_exchange": (
                "height/vertical-momentum pressure-gravity exchange is entropy neutral"
            ),
            "geometry": "metric and tetrad derivatives are lower-order covariant sources",
            "external_cooling_and_stream_sources": (
                "audited in the complete total-energy and physical-entropy ledgers"
            ),
        },
        "local_theorem_contract": {
            "convexity": "minimum eigenvalue(A0)>0 on the complete frozen envelope",
            "symmetric_hyperbolicity": "AR=AR.T in entropy variables",
            "causality": "all generalized speeds real and inside the Kerr-Schild light cone",
            "nonlinear_full_tensor_screen": (
                "Cordeiro-et-al. necessary-and-sufficient causality inequalities "
                "and sufficient strong-hyperbolicity inequalities"
            ),
            "constraints": "normalization, symmetry, trace, and orthogonality propagate",
            "subcharacteristic": (
                "equilibrium compression interlaces the 11-field generalized spectrum"
            ),
            "independent_derivatives": (
                "analytic/automatic Hessians versus a frozen high-order finite-difference ladder"
            ),
        },
        "numerical_architecture_after_local_certificate": {
            "space": (
                "entropy-stable finite-volume flux in entropy variables plus a "
                "path-conservative treatment of geometry and moving-frame products"
            ),
            "time": (
                "IMEX/AP: explicit causal transport; cell-local implicit five-shear "
                "and vertical relaxation; no global nonlinear truth root"
            ),
            "admissibility": (
                "convex-domain limiter applied to entropy variables before primitive recovery"
            ),
            "restart": "all 11 fields, multistep/IMEX history, and local solver state",
        },
        "reduction_architecture_after_truth_certificate": {
            "offline": (
                "certified 11-field truth microbursts at multiple Q/phase anchors; "
                "identify a normally attracting fast equilibrium or periodic bundle"
            ),
            "slow_coordinates": (
                "conservative mass/angular-momentum/energy coordinates and phase; "
                "all other fields are slaved only after normal-attraction evidence"
            ),
            "online": (
                "interpolate a certified slow flux/source atlas with uncertainty and "
                "domain guards; make no truth calls during the complete cycle"
            ),
            "cycle_cost_goal": "one complete cycle in no more than several wall days",
        },
        "prospective_sequence": (
            "STF representation and convex normal form",
            "nonlinear physical master-potential derivation",
            "complete local envelope structural certificate",
            "entropy-stable interface and IMEX/AP certificate",
            "bounded transient reacquisition through the prior boundary",
            "fast-bundle normal-attraction certificate",
            "multi-anchor slow-atlas cross-validation and cost certificate",
            "definitions-only complete-cycle execution manifest",
        ),
        "next_package": {
            "scope": AUTHORIZED_NEXT,
            "implement_covariant_five_STF_basis": True,
            "implement_quadratic_eleven_field_common_potential_normal_form": True,
            "prove_basis_constraints_and_roundtrip": True,
            "prove_A0_positive_AR_symmetric_entropy_dissipative": True,
            "physical_coefficient_fit_or_trajectory": False,
        },
        "external_mathematical_basis": {
            "geroch_lindblom_divergence_type": (
                "https://doi.org/10.1103/PhysRevD.41.1855"
            ),
            "lehner_reula_rubio_master_scalar": (
                "https://arxiv.org/abs/1710.08033"
            ),
            "gavassino_onsager_symmetric_hyperbolicity": (
                "https://arxiv.org/abs/2210.05067"
            ),
            "cordeiro_full_nonlinear_israel_stewart": (
                "https://arxiv.org/abs/2607.05639"
            ),
            "literature_does_not_certify_this_disk_column_closure": True,
        },
        "claim_boundary": {
            "architecture_selected": True,
            "five_STF_basis_implemented": False,
            "quadratic_normal_form_certified": False,
            "nonlinear_physical_master_potential_derived": False,
            "eleven_field_local_closure_certified": False,
            "eleven_field_trajectory_authorized": False,
            "slow_atlas_authorized": False,
            "complete_cycle_execution_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
        "budget": {
            "truth_operator_calls": 0,
            "new_trajectory_steps": 0,
            "complete_cycle_steps": 0,
            "maximum_wall_minutes": 10,
        },
        "forbidden": (
            "resume or refine the rejected seven-field trajectory",
            "discard four shear amplitudes after forming the full tensor",
            "freeze the moving STF basis inside a derivative",
            "use the full-tensor causality screen as proof of this unimplemented closure",
            "fit coefficients to repair only cell 6",
            "claim a local normal form is the nonlinear physical disk model",
            "run an eleven-field trajectory before the local structural certificate",
            "execute a complete cycle",
        ),
        "authorized_next": AUTHORIZED_NEXT,
    }


def _update_catalog(summary: dict) -> None:
    utils = _utils()
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
                    "scientific_status": "DEFINITIONS_ONLY",
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


def _canonicalize() -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("eleven-field architecture manifest already exists")
    validated = _validate_parent(require_clean=True)
    utils = _utils()
    contract = _contract()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "architecture_contract.json", contract)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "seven_field_rejection_preserved": True,
        "eleven_field_architecture_selected": True,
        "eleven_field_physical_closure_certified": False,
        "eleven_field_trajectory_authorized": False,
        "complete_cycle_execution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utils._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_artifact": parent.ARTIFACT,
            "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256,
            "parent_hashes": validated["hashes"],
            "parent_classification": parent.PASS_CLASSIFICATION,
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Eleven-field convex/divergence architecture manifest",
                "",
                f"Classification: `{CLASSIFICATION}`.",
                "",
                "The derivative-stable seven-field complex pair rejects only the one-amplitude projected-shear closure. The replacement retains mass, full stress-energy (including vertical momentum), and height content, while promoting all five independent rest-frame symmetric-tracefree shear amplitudes. The total local dimension is eleven.",
                "",
                "A single master scalar generates the mass, stress-energy, height, and shear currents. Its temporal Hessian must be positive, its radial Hessian symmetric, and its dissipative source non-positive in entropy variables. Moving rest-frame basis derivatives are binding; no post-hoc R-phi projection is permitted.",
                "",
                "This package selects the architecture only. It does not derive the nonlinear physical master potential, certify the eleven-field closure, authorize a trajectory, or authorize complete-cycle execution.",
                "",
                f"Authorized next: `{AUTHORIZED_NEXT}`.",
                "",
            )
        ),
        encoding="utf-8",
    )
    sources = (THIS_RUNNER, THIS_TEST, REPORT_RELATIVE)
    utils._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "implementation_commit": utils._git("rev-parse", "HEAD"),
            "implementation_tree": utils._git("rev-parse", "HEAD^{tree}"),
            "source_hashes": {
                path: utils._sha256(ROOT / path) for path in sources
            },
            "python": sys.version,
            "platform": platform.platform(),
            "thread_environment": {
                name: os.environ.get(name, "")
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
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
    print(json.dumps(_canonicalize(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
