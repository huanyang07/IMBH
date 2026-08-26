#!/usr/bin/env python3
"""Freeze the corrected generalized Maxwell--Cattaneo architecture.

This definitions-only package responds prospectively to the failed nonlinear
Godunov realization in WP10c9d6c7c3b5c4f25fizec.  It preserves exact mass
and stress-energy conservation, but represents the dissipative shear sector
by its covariant transient equation and binds nonlinear causality and strong
hyperbolicity directly.  It authorizes one local, non-trajectory structural
audit and nothing beyond it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_seven_field_physical_closure_local_structural_audit_wp10c9d6c7c3b5c4f25fizec as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fized_"
    "generalized_Maxwell_Cattaneo_architecture_manifest"
)
CLASSIFICATION = (
    "generalized_Maxwell_Cattaneo_seven_field_architecture_manifest_frozen"
)
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizee_"
    "generalized_Maxwell_Cattaneo_local_structural_audit"
)
ARTIFACT = (
    "causal_inner_generalized_maxwell_cattaneo_architecture_manifest_"
    "wp10c9d6c7c3b5c4f25fized"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_GENERALIZED_MAXWELL_CATTANEO_"
    "ARCHITECTURE_MANIFEST_WP10C9D6C7C3B5C4F25FIZED_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_generalized_maxwell_cattaneo_architecture_"
    "manifest_wp10c9d6c7c3b5c4f25fized.py"
)
THIS_TEST = (
    "tests/test_causal_inner_generalized_maxwell_cattaneo_architecture_"
    "manifest_wp10c9d6c7c3b5c4f25fized.py"
)
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

PARENT_RESULT_COMMIT = "dc7a762aad8e78afa67d1717e829dda68d640a7a"
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "047aee0b1543112767f86b5e31b61a8b8c6cf231e2c89124841c0a7bf4d75fbb"
)
PARENT_SUMMARY_SHA256 = (
    "f94248f2d10c2cd71cc0a00da0290ac111165e211019b6ddc895fc1ba2cb5248"
)
PARENT_METRICS_SHA256 = (
    "9bf6e75b6f24e4efde03987dfdaa775e5d010415307055973f6d29ff771e7548"
)


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_checksums(directory: Path) -> dict[str, str]:
    hashes = {}
    for line in (directory / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        actual = _sha256(directory / name)
        if actual != expected:
            raise RuntimeError(f"checksum mismatch: {directory / name}")
        hashes[name] = actual
    return hashes


def _validate_parent(*, require_clean: bool) -> dict:
    directory = parent.CANONICAL_DIRECTORY
    if _sha256(directory / "SHA256SUMS.txt") != PARENT_CHECKSUM_MANIFEST_SHA256:
        raise RuntimeError("failed-closure checksum manifest changed")
    hashes = _validate_checksums(directory)
    if (
        hashes["summary.json"] != PARENT_SUMMARY_SHA256
        or hashes["audit_metrics.json"] != PARENT_METRICS_SHA256
    ):
        raise RuntimeError("failed-closure decisive evidence changed")
    summary = _read_json(directory / "summary.json")
    metrics = _read_json(directory / "audit_metrics.json")
    provenance = _read_json(directory / "provenance.json")
    if (
        summary["classification"] != parent.CLASSIFICATION
        or summary["passed"]
        or not summary["audit_completed"]
        or summary["entropy_integrability_passed"]
        or not summary["stable_order_unity_obstruction"]
        or not summary["corrective_architecture_manifest_authorized"]
        or summary["authorized_next"]
        != "definitions_only_" + WORK_PACKAGE
        or metrics["minimum_relative_entropy_flux_curl_defect"] <= 0.1
        or metrics["dominant_obstruction_coordinates"]
        != ["log_temperature", "specific_shear_stress"]
        or provenance["implementation_commit"]
        != "a5ed8c85caacd2639332852affeabf80eae75edd"
    ):
        raise RuntimeError("failed-closure classification changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha256(ROOT / relative) != expected:
            raise RuntimeError(f"failed-closure source changed: {relative}")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("architecture freeze requires a clean tracked tree")
    return {
        "hashes": hashes,
        "summary": summary,
        "metrics": metrics,
        "provenance": provenance,
    }


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "authorization_basis": {
            "parent_result_commit": PARENT_RESULT_COMMIT,
            "parent_classification": parent.CLASSIFICATION,
            "parent_entropy_obstruction_preserved": True,
            "scope_is_definitions_only": True,
        },
        "preserved_scientific_boundary": {
            "five_field_trajectory_classification": (
                "hyperbolicity_boundary_bracketed_after_first_half_step"
            ),
            "five_field_accepted_endpoint_count": 72,
            "five_field_accepted_terminal_seconds": 0.18587500000000012,
            "failed_five_field_candidate_propagated": False,
            "failed_Godunov_candidate_propagated": False,
            "no_trajectory_is_restarted_by_this_package": True,
        },
        "selected_PDE_class": {
            "name": (
                "one_shear_amplitude Landau-frame generalized "
                "Maxwell-Cattaneo/Israel-Stewart balance law"
            ),
            "mass_current": "J^a=Sigma*u^a",
            "stress_energy": (
                "T_col^{ab}=epsilon*u^a*u^b+Pi*Delta^{ab}+pi^{ab}"
            ),
            "exact_conservation": (
                "nabla_a J^a=0 and nabla_b T_col^{ab}=external sources"
            ),
            "dissipative_equation": (
                "tau_pi Delta^{ab}_{cd} u^e nabla_e pi^{cd}+pi^{ab}="
                "-2 eta sigma^{ab} plus prospectively declared nonlinear terms"
            ),
            "mathematical_standard": (
                "nonlinear causality plus strong hyperbolicity of the complete "
                "quasilinear principal symbol"
            ),
            "global_Godunov_potential_required": False,
            "reason_global_Godunov_requirement_is_removed": (
                "the rejected D*chi state-dependent-modulus realization is not "
                "the general transient relativistic-fluid PDE class"
            ),
            "post_hoc_symmetrization_still_forbidden": True,
        },
        "physical_variables": {
            "q7": (
                "lnSigma",
                "beta_R",
                "beta_phi",
                "lnT",
                "chi",
                "lnH",
                "beta_H_equals_w_H_over_c",
            ),
            "shear_tensor": (
                "pi^{ab}=Sigma*c^2*chi*(e_R^a e_phi^b+e_phi^a e_R^b)"
            ),
            "shear_constraints": (
                "symmetric, trace free, and orthogonal to u^a by construction"
            ),
            "stress_is_not_an_added_isotropic_energy_reservoir": True,
            "horizontal_velocity_gate": "beta_R^2+beta_phi^2<1",
            "vertical_velocity_gate": "abs(beta_H)<1",
            "dominant_energy_gate": "all principal stresses are below enthalpy",
        },
        "thermodynamic_and_vertical_sector": {
            "density": "rho=Sigma/(2H)",
            "gas_radiation_EOS_at_independent_H": True,
            "height_current": "nabla_a(Sigma*H*u^a)=Sigma*w_H",
            "vertical_momentum_current": (
                "nabla_a(Sigma*w_H*u^a)=Pi/H-Sigma*Omega_perp^2*H-"
                "gamma_H*Sigma*w_H"
            ),
            "vertical_damping": "gamma_H=alpha*Omega_perp",
            "vertical_energy_ledger": (
                "vertical force and damping work are exchanged with the thermal "
                "energy equation with zero hidden total-energy defect"
            ),
            "height_is_finite_inertia_not_algebraically_responsive": True,
            "height_and_vertical_velocity_are_advective_in_the_radial_principal": True,
        },
        "transport_calibration": {
            "alpha_equilibrium_target": "chi_NS=alpha*Pi/(Sigma*c^2)",
            "reference_positive_shear": "gamma_dot_ref=1.5*Omega_perp",
            "specific_viscosity": "nu_s=chi_NS/gamma_dot_ref",
            "linear_signal_target": "eta/(tau_pi*E)=alpha*(c_s/c)^2",
            "relaxation_time": "tau_pi=nu_s/(h_eff*alpha*(c_s/c)^2)",
            "tau_pi_positive": True,
            "eta_positive": True,
            "no_face_dependent_coefficient": True,
            "no_hyperbolicity_floor_or_clip": True,
        },
        "nonlinear_principal_contract": {
            "derive_from": (
                "covariant mass, energy-momentum, projected shear-relaxation, "
                "height-current, and vertical-momentum equations"
            ),
            "unknown_derivative_order": (
                "lnSigma,beta_R,beta_phi,lnT,chi,lnH,beta_H"
            ),
            "temporal_matrix": "M^t(q;R)",
            "radial_matrix": "M^R(q;R)",
            "characteristic_equation": "det(M^R-lambda*M^t)=0",
            "transport_coefficients_differentiated_when_the_covariant_equation_requires": True,
            "normalization_orthogonality_and_trace_constraints_propagated": True,
            "complete_symbol_not_isolated_shear_cone_is_binding": True,
            "coordinate_light_cone_is_Kerr_Schild": True,
        },
        "nonlinear_causality_gates": {
            "paper": "https://arxiv.org/abs/2607.05639",
            "specialization": (
                "zero bulk stress, one physical R-phi shear amplitude, baryon "
                "current retained, fixed Kerr-Schild background"
            ),
            "evaluate_all_specialized_necessary_and_sufficient_inequalities": True,
            "basic_transverse_ratio": "0<=eta_eff/(tau_pi*E)<=1",
            "finite_shear_eigenvalues": "Lambda=(-abs(Sigma*c^2*chi),0,+abs(...))",
            "require_E_plus_each_Lambda_positive": True,
            "inequality_margin_min": 1.0e-8,
            "no_linearized_only_certificate": True,
        },
        "strong_hyperbolicity_gates": {
            "all_generalized_eigenvalues_real": True,
            "maximum_imaginary_coordinate_speed_over_c": 1.0e-10,
            "maximum_absolute_coordinate_speed_inside_local_light_cone": True,
            "generalized_eigenpair_relative_defect_max": 1.0e-8,
            "diagonally_equilibrated_eigenvector_condition_number_max": 1.0e8,
            "constant_geometric_multiplicity_and_complete_eigenbasis_required": True,
            "neighboring_subspace_minimum_cosine_min": 0.90,
            "left_right_projector_relative_defect_max": 1.0e-8,
        },
        "entropy_and_energy_scope": {
            "equilibrium_EOS_entropy_exact": True,
            "original_IS_quadratic_entropy_current_checked": True,
            "entropy_production_nonnegative_on_frozen_stencil": True,
            "DNMR_terms_not_added_without_separate_entropy_accounting": True,
            "global_convex_entropy_or_symmetric_hyperbolicity_claimed": False,
            "strong_hyperbolicity_is_sufficient_for_local_well_posedness": True,
            "mass_angular_momentum_and_Killing_energy_ledgers_remain_binding": True,
        },
        "future_spatial_discretization_if_local_audit_passes": {
            "conservative_rows": "mass and Kerr-Schild stress-energy",
            "nonconservative_rows": "projected shear and material height kinematics",
            "method": (
                "path-conservative finite volume with a prospectively frozen "
                "straight entropy-scaled primitive path"
            ),
            "path_consistency": "Dal Maso-LeFloch-Murat jump integral",
            "smooth_limit_parity_required": True,
            "Riemann_dissipation_uses_complete_real_eigenbasis": True,
            "this_manifest_authorizes_spatial_discretization": False,
        },
        "frozen_local_audit_envelope": {
            "reuse_parent_stage2_audit_envelope_bitwise": True,
            "base_charts": 8401,
            "deterministic_witnesses": 47,
            "equilibrium_embedding_all_base_charts": True,
            "off_equilibrium_height_vertical_stress_stencils": True,
            "failed_face_is_negative_control": True,
            "mutable_scratch_forbidden": True,
        },
        "audit_order": (
            "derive the specialized covariant quasilinear matrices symbolically",
            "audit coefficient and physical-constraint identities",
            "audit the primary and held-out profiles",
            "audit every accepted pre-boundary profile",
            "audit all deterministic off-equilibrium witnesses",
            "audit the rejected profile and failed face without propagation",
            "evaluate specialized nonlinear causality inequalities",
            "evaluate complete strong-hyperbolicity and light-cone gates",
            "freeze a positive or negative local certificate",
        ),
        "decision_classifications": {
            "pass": "generalized_Maxwell_Cattaneo_local_structural_audit_passed",
            "causality_failure": "generalized_Maxwell_Cattaneo_nonlinear_causality_failed",
            "strong_hyperbolicity_failure": "generalized_Maxwell_Cattaneo_strong_hyperbolicity_failed",
            "ledger_failure": "generalized_Maxwell_Cattaneo_energy_entropy_ledger_failed",
            "derivation_failure": "generalized_Maxwell_Cattaneo_derivation_failed",
        },
        "budget": {
            "new_trajectory_steps": 0,
            "new_nonlinear_roots": 0,
            "new_fixed_Q_roots": 0,
            "new_complete_cycle_steps": 0,
            "maximum_manifest_wall_minutes": 10,
        },
        "claim_boundary": {
            "failed_Godunov_realization_preserved": True,
            "corrected_transient_architecture_selected": True,
            "corrected_transient_architecture_certified": False,
            "local_structural_audit_authorized": True,
            "spatial_discretization_authorized": False,
            "seven_field_trajectory_authorized": False,
            "fixed_Q_invariant_object_authorized": False,
            "slow_flux_atlas_authorized": False,
            "complete_cycle_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
        "forbidden": (
            "reinterpret the failed entropy curl as numerical noise",
            "restore the rejected D*chi Godunov flux by post-hoc symmetrization",
            "certify only the linearized or isolated shear cone",
            "fit transport coefficients to the old failed face",
            "hide a nonconservative product inside a lower-order source",
            "advance any five-field or seven-field trajectory",
            "construct a fixed-Q orbit, slow atlas, or complete cycle",
        ),
        "authorized_next": AUTHORIZED_NEXT,
    }


def _source_hashes() -> dict[str, str]:
    paths = (THIS_RUNNER, THIS_TEST, REPORT_RELATIVE)
    return {path: _sha256(ROOT / path) for path in paths}


def _report() -> str:
    return "\n".join(
        (
            "# Generalized Maxwell--Cattaneo seven-field architecture manifest",
            "",
            f"Classification: `{CLASSIFICATION}`.",
            "",
            "This definitions-only package replaces the failed nonlinear `D chi` Godunov realization; it does not erase or reinterpret that failure.",
            "",
            "## Selected mathematical architecture",
            "",
            "Mass and the Kerr--Schild stress-energy tensor remain exact covariant conservation laws. The physical R-phi shear amplitude is evolved with its projected transient Israel--Stewart/Maxwell--Cattaneo equation. Height and vertical momentum remain finite-inertia material balances. The complete seven-field quasilinear principal symbol—not an isolated signal-speed estimate—is binding.",
            "",
            "The dissipative equation is intentionally treated as a nonconservative covariant product. A future spatial implementation, if authorized, must therefore be path-conservative and must reproduce the smooth quasilinear equation and its jump integral. No nonconservative derivative may be disguised as a lower-order source.",
            "",
            "## Mathematical standard",
            "",
            "The local certificate requires the exact one-shear specialization of the nonlinear necessary-and-sufficient causality inequalities of Cordeiro et al., a complete real eigenbasis with controlled condition number, Kerr--Schild light-cone containment, physical constraint propagation, and explicit energy/entropy ledgers. A global convex Godunov potential is not claimed for this transient dissipative class.",
            "",
            "## Frozen evidence",
            "",
            "The audit reuses the bitwise Stage-2 envelope: 8,401 canonical base charts, 47 deterministic witnesses, the primary and held-out profiles, all 72 accepted pre-boundary profiles, the rejected full-step profile, and the independently diagnosed failed face. No coefficient may depend on the failed face.",
            "",
            "## Decision",
            "",
            f"Authorized next: `{AUTHORIZED_NEXT}` only. It may derive and execute the local structural audit, with no spatial step, trajectory, fixed-Q orbit, slow atlas, or complete-cycle execution.",
            "",
        )
    )


def _update_catalog(summary: dict) -> None:
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
                    "sha256": _sha256(path),
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
    catalog = _read_json(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": summary["passed"],
    }
    catalog.update(
        {
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": _git("rev-parse", "HEAD"),
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("Maxwell-Cattaneo architecture manifest already exists")
    parent_data = _validate_parent(require_clean=True)
    contract = _contract()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    _write_json(CANONICAL_DIRECTORY / "architecture_contract.json", contract)
    _write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_artifact": parent.ARTIFACT,
            "parent_result_commit": PARENT_RESULT_COMMIT,
            "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256,
            "parent_hashes": parent_data["hashes"],
            "stage2_audit_envelope_artifact": parent.parent.ARTIFACT,
            "stage2_audit_envelope_sha256": parent_data["metrics"][
                "schema_version"
            ]
            and parent._validate_parent(require_clean=False)["hashes"][
                "audit_envelope.npz"
            ],
            "canonical_sources_only": True,
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "failed_Godunov_realization_preserved": True,
        "corrected_transient_architecture_selected": True,
        "corrected_transient_architecture_certified": False,
        "local_structural_audit_authorized": True,
        "new_trajectory_steps": 0,
        "spatial_discretization_authorized": False,
        "seven_field_trajectory_authorized": False,
        "fixed_Q_invariant_object_authorized": False,
        "slow_flux_atlas_authorized": False,
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_report(), encoding="utf-8")
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "DEFINITIONS_ONLY",
            "implementation_commit": _git("rev-parse", "HEAD"),
            "implementation_tree": _git("rev-parse", "HEAD^{tree}"),
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "source_hashes": _source_hashes(),
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "thread_environment": {
                name: os.environ.get(name, "")
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
        },
    )
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    arguments = parser.parse_args()
    if not arguments.freeze:
        parser.error("choose --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
