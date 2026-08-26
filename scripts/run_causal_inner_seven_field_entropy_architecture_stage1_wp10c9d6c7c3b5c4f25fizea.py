#!/usr/bin/env python3
"""Freeze the post-boundary seven-field entropy architecture, Stage 1."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
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

from imri_qpe.layer3_minidisk_1d.causal_inner_seven_field_entropy import (  # noqa: E402
    N_SEVEN_FIELDS,
    N_VERTICAL_EQUILIBRIUM_FIELDS,
    SEVEN_FIELD_PRIMITIVE_NAMES,
    VERTICAL_EQUILIBRIUM_FIVE_FIELD_NAMES,
    audit_seven_field_entropy_normal_form,
    build_seven_field_entropy_normal_form,
    reference_seven_field_entropy_parameters,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fizea_"
    "seven_field_entropy_architecture_stage1"
)
CLASSIFICATION = (
    "seven_field_entropy_stage1_normal_form_derived_"
    "local_physical_closure_manifest_authorized"
)
AUTHORIZED_NEXT = (
    "definitions_only_seven_field_physical_closure_"
    "and_local_structural_audit_manifest"
)

PARENT_ARTIFACT = (
    "causal_inner_tangent_phase_hyperbolicity_two_half_step_bracket_"
    "execution_wp10c9d6c7c3b5c4f25fizdd"
)
PARENT_DIRECTORY = ROOT / "results/canonical" / PARENT_ARTIFACT
PARENT_CLASSIFICATION = "hyperbolicity_boundary_bracketed_after_first_half_step"
PARENT_RESULT_COMMIT = "3db75a53110f94a3ec9abee4933042bb62f0bd3f"
ARCHITECTURE_COMMIT = "b7546ed9251c79ab9788df0faa59e7c008a2872e"
ARCHITECTURE_TREE = "32b6bdd650cba29b5e204b20772843c9e7ffdcac"
ARCHITECTURE_REPORT = (
    "docs/reports/current/CODEX_CAUSAL_INNER_POST_HYPERBOLICITY_"
    "MATHEMATICAL_ARCHITECTURE_2026-08-25.md"
)
ARCHITECTURE_REPORT_SHA256 = (
    "353c372e933594e785c1a7fc5f8b9401b60ba06f6cf362574b58c4b353c43a33"
)

ARTIFACT = (
    "causal_inner_seven_field_entropy_architecture_stage1_"
    "wp10c9d6c7c3b5c4f25fizea"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_SEVEN_FIELD_ENTROPY_"
    "ARCHITECTURE_STAGE1_WP10C9D6C7C3B5C4F25FIZEA_2026-08-25.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_seven_field_entropy_architecture_stage1_"
    "wp10c9d6c7c3b5c4f25fizea.py"
)
THIS_TEST = (
    "tests/test_causal_inner_seven_field_entropy_architecture_stage1_"
    "wp10c9d6c7c3b5c4f25fizea.py"
)
NORMAL_FORM_SOURCE = (
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_seven_field_entropy.py"
)
NORMAL_FORM_TEST = "tests/test_causal_inner_seven_field_entropy.py"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


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
    recorded = {}
    for line in (directory / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        actual = _sha256(directory / name)
        if actual != expected:
            raise RuntimeError(f"checksum mismatch: {directory / name}")
        recorded[name] = actual
    return recorded


def _validate_parent(*, require_clean: bool) -> dict:
    hashes = _validate_checksums(PARENT_DIRECTORY)
    summary = _read_json(PARENT_DIRECTORY / "summary.json")
    metrics = _read_json(PARENT_DIRECTORY / "execution_metrics.json")
    if (
        summary["classification"] != PARENT_CLASSIFICATION
        or not summary["passed"]
        or summary["trajectory_continuation_passed"]
        or summary["new_accepted_half_steps"] != 1
        or summary["combined_accepted_endpoints"] != 72
        or summary["authorized_next"] is not None
        or summary["complete_cycle_execution_authorized"]
        or summary["reduced_slow_evolution_authorized"]
        or metrics["classification"] != PARENT_CLASSIFICATION
        or metrics["stop_reason"] != "hyperbolicity_boundary"
        or metrics["failed_full_step_propagated"]
        or metrics["gate_values"]["boundary_candidates"] != 1
        or metrics["gate_values"][
            "maximum_boundary_complex_characteristic_component"
        ]
        < 1.0e-2
    ):
        raise RuntimeError("post-boundary parent classification changed")
    if _git("rev-parse", PARENT_RESULT_COMMIT) != PARENT_RESULT_COMMIT:
        raise RuntimeError("parent boundary result commit changed")
    if _git("rev-parse", ARCHITECTURE_COMMIT) != ARCHITECTURE_COMMIT:
        raise RuntimeError("architecture decision commit changed")
    if _git("rev-parse", f"{ARCHITECTURE_COMMIT}^{{tree}}") != ARCHITECTURE_TREE:
        raise RuntimeError("architecture decision tree changed")
    if _sha256(ROOT / ARCHITECTURE_REPORT) != ARCHITECTURE_REPORT_SHA256:
        raise RuntimeError("post-boundary architecture report changed")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("Stage-1 architecture freeze requires a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "metrics": metrics}


def _contract() -> dict:
    """Return the binding Stage-1 derivation and claim boundary."""

    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "authorization_basis": {
            "parent_authorized_next_is_null": True,
            "explicit_user_instruction": (
                "proceed with Stage 1 after review of the post-hyperbolicity "
                "architecture"
            ),
            "scope_is_model_design_only": True,
        },
        "parent_boundary_preserved": {
            "classification": PARENT_CLASSIFICATION,
            "accepted_endpoint_count": 72,
            "accepted_terminal_seconds": 0.18587500000000012,
            "first_complex_face": 3,
            "five_field_trajectory_remains_stopped": True,
            "old_failed_candidate_may_not_be_propagated": True,
        },
        "seven_field_state": {
            "conserved_names": (
                "surface_rest_mass",
                "radial_momentum",
                "angular_momentum",
                "total_column_energy",
                "causal_shear_stress_coordinate",
                "height_content_Z_H_equals_surface_mass_times_H",
                "vertical_momentum_P_H_equals_surface_mass_times_w_H",
            ),
            "primitive_names": SEVEN_FIELD_PRIMITIVE_NAMES,
            "dimension": N_SEVEN_FIELDS,
            "vertical_equilibrium_dimension": N_VERTICAL_EQUILIBRIUM_FIELDS,
            "vertical_equilibrium_names": VERTICAL_EQUILIBRIUM_FIVE_FIELD_NAMES,
            "height_and_vertical_momentum_are_independent": True,
            "shear_stress_remains_a_causal_relaxation_field": True,
        },
        "total_energy_architecture": {
            "density_relation": "rho=Sigma/(2*H)",
            "terms": {
                "orbital": "E_KS(Sigma,m_r,L;g)",
                "internal": "Sigma*e(rho,s)",
                "vertical_kinetic": "P_H**2/(2*Sigma)",
                "vertical_gravity": "0.5*Sigma*Omega_perp**2*H**2",
                "stress_relaxation": "R_pi**2/(2*Sigma*a_pi), a_pi>0",
            },
            "vertical_force_at_fixed_entropy": (
                "-dE/dH=Pi/H-Sigma*Omega_perp**2*H"
            ),
            "hydrostatic_equilibrium": "Pi=Sigma*Omega_perp**2*H**2",
            "matches_existing_gas_radiation_height_equation": True,
            "damping_and_stress_relaxation_heat_internal_energy": True,
            "total_energy_loss_from_internal_relaxation": False,
        },
        "entropy_potential_architecture": {
            "entropy_variables": "w=d_eta/dU",
            "state_potential": "U=d_psi(w)/dw",
            "flux_potential": "F=d_phi(w)/dw",
            "temporal_hessian": "A0=d2_psi/dw2 positive definite",
            "spatial_hessian": "A1=d2_phi/dw2 symmetric",
            "source_inequality": "w dot S <= 0 for mathematical entropy",
            "geometry_derivatives_are_lower_order_sources": True,
            "all_principal_derivative_couplings_must_come_from_phi": True,
            "algebraic_height_derivative_foldback_forbidden": True,
        },
        "proved_local_normal_form": {
            "height_departure": (
                "y_H=dlnH-H_Sigma*dlnSigma-H_T*dlnT"
            ),
            "chart_map": "r=L*q with det(L)=1",
            "positive_metric": "M=diag(m_i)>0",
            "symmetric_flux_hessian": "K=K.T",
            "chart_temporal": "A0=L.T*M*L",
            "chart_spatial": "A1=L.T*K*L",
            "vertical_metric_relation": "m_H=m_w*omega_H**2",
            "vertical_source": "dy_H/dt=w_H; dw_H/dt=-omega_H**2*y_H-gamma_H*w_H",
            "stress_source": "dchi/dt=-r_pi*chi plus entropy-paired shear principal coupling",
            "entropy_production": "-m_chi*r_pi*chi**2-m_w*gamma_H*w_H**2",
            "vertical_equilibrium": "y_H=0 and w_H=0",
            "subcharacteristic_theorem": (
                "five-field vertical-equilibrium compression interlaces the "
                "seven-field symmetric generalized spectrum"
            ),
        },
        "five_field_limit_boundary": {
            "exact_equivalence_required_only_on_certified_pre_boundary_domain": True,
            "exact_equivalence_at_or_beyond_failed_face_forbidden": True,
            "reason": (
                "an entropy-stable relaxation limit cannot be required to "
                "inherit the committed nonhyperbolic equilibrium pencil"
            ),
            "finite_vertical_inertia_is_binding_beyond_old_validity_domain": True,
            "old_model_remains_a_pre_boundary_comparison_control": True,
        },
        "external_mathematical_basis": {
            "chen_levermore_liu_relaxation_entropy": (
                "https://doi.org/10.1002/cpa.3160470602"
            ),
            "linear_onsager_symmetric_hyperbolicity": (
                "https://arxiv.org/abs/2210.05067"
            ),
            "nonlinear_israel_stewart_shear_conditions": (
                "https://arxiv.org/abs/2607.05639"
            ),
            "godunov_variables_bulk_relaxation_example": (
                "https://arxiv.org/abs/2110.15223"
            ),
            "literature_does_not_certify_this_column_reduction": True,
        },
        "stage2_manifest_obligations": {
            "definitions_only": True,
            "derive_physical_Kerr_Schild_state_and_flux_potentials": True,
            "derive_exact_column_entropy_variables": True,
            "map_shear_coefficients_to_nonlinear_causality_conditions": True,
            "derive_vertical_constraint_and_source_propagation": True,
            "freeze_physical_parameter_envelope_before_eigenvalue_scan": True,
            "audit_primary_heldout_boundary_and_failed_face_states": True,
            "independent_automatic_or_finite_difference_hessians": True,
            "no_trajectory": True,
            "no_spatial_discretization_change": True,
        },
        "stage2_local_pass_gates": {
            "A0_positive_on_entire_frozen_envelope": True,
            "A1_symmetric_in_entropy_variables": True,
            "source_entropy_production_nonpositive": True,
            "all_characteristic_speeds_real_and_causal": True,
            "uniform_energy_metric_condition_margin": True,
            "positive_vertical_subcharacteristic_margin_pre_boundary": True,
            "old_pre_boundary_principal_and_source_parity": True,
            "failed_face_is_real_only_in_new_finite_inertia_model": True,
            "analytic_independent_derivative_parity": True,
        },
        "budget": {
            "new_five_field_trajectory_steps": 0,
            "new_seven_field_trajectory_steps": 0,
            "new_nonlinear_roots": 0,
            "new_fixed_Q_roots": 0,
            "complete_cycle_steps": 0,
            "maximum_stage1_wall_minutes": 10,
        },
        "claim_boundary": {
            "quadratic_entropy_normal_form_certified": True,
            "physical_Kerr_Schild_seven_field_closure_certified": False,
            "nonlinear_global_symmetrizer_certified": False,
            "seven_field_local_structural_audit_authorized": False,
            "definitions_only_stage2_manifest_authorized": True,
            "seven_field_trajectory_authorized": False,
            "fixed_Q_invariant_object_authorized": False,
            "slow_flux_atlas_authorized": False,
            "complete_cycle_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
        "forbidden": (
            "resume the stopped five-field trajectory",
            "fit coefficients to make only face 3 real",
            "claim the quadratic normal form is the physical disk closure",
            "force the new model to reproduce the nonhyperbolic old symbol",
            "freeze height algebraically inside the seven-field principal part",
            "discard vertical or stress relaxation heat from total energy",
            "run a seven-field trajectory before the local structural certificate",
            "authorize a fixed-Q orbit, slow atlas, or complete cycle",
        ),
        "authorized_next": AUTHORIZED_NEXT,
    }


def _reference_payload() -> tuple[dict, dict[str, np.ndarray]]:
    parameters = reference_seven_field_entropy_parameters()
    normal_form = build_seven_field_entropy_normal_form(parameters)
    audit = audit_seven_field_entropy_normal_form(parameters)
    if not audit.passed:
        raise RuntimeError("Stage-1 reference normal form failed its identities")
    payload = {
        "parameters": asdict(parameters),
        "audit": asdict(audit),
        "passed": audit.passed,
        "fixture_is_physical_calibration": False,
        "fixture_role": "identity_and_proof_template_only",
    }
    arrays = {
        "chart_to_relaxation": normal_form.chart_to_relaxation,
        "entropy_metric_relaxation": normal_form.entropy_metric_relaxation,
        "flux_hessian_relaxation": normal_form.flux_hessian_relaxation,
        "source_generator_relaxation": normal_form.source_generator_relaxation,
        "temporal_matrix": normal_form.temporal_matrix,
        "spatial_matrix": normal_form.spatial_matrix,
        "source_matrix": normal_form.source_matrix,
        "vertical_equilibrium_embedding": normal_form.vertical_equilibrium_embedding,
        "reduced_temporal_matrix": normal_form.reduced_temporal_matrix,
        "reduced_spatial_matrix": normal_form.reduced_spatial_matrix,
    }
    return payload, arrays


def _source_hashes() -> dict[str, str]:
    paths = (
        THIS_RUNNER,
        THIS_TEST,
        NORMAL_FORM_SOURCE,
        NORMAL_FORM_TEST,
        ARCHITECTURE_REPORT,
    )
    return {path: _sha256(ROOT / path) for path in paths}


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


def _report(reference: dict) -> str:
    audit = reference["audit"]
    return "\n".join(
        (
            "# Seven-field entropy architecture Stage 1",
            "",
            f"Classification: `{CLASSIFICATION}`.",
            "",
            "This definitions-only package preserves the stopped five-field boundary and derives the algebraic proof template for a seven-field entropy-compatible relaxation model. It executes no trajectory and does not claim that a physical Kerr--Schild closure is already certified.",
            "",
            "## Selected state and energy",
            "",
            "The state promotes column height and vertical momentum: `(Sigma, m_r, L, E_tot, R_pi, Z_H, P_H)`, with `Z_H=Sigma H` and `P_H=Sigma w_H`. Total energy contains orbital, gas+radiation internal, vertical kinetic, vertical gravitational, and positive stress-relaxation energy.",
            "",
            "At fixed entropy, `rho=Sigma/(2H)` gives `-dE/dH=Pi/H-Sigma Omega_perp^2 H`; its zero is exactly the existing gas+radiation hydrostatic relation `Pi=Sigma Omega_perp^2 H^2`. Height is nevertheless kept dynamical in the repaired model.",
            "",
            "## Structural derivation",
            "",
            "In entropy variables, `U=dpsi/dw` and `F=dphi/dw`. The binding architecture requires `A0=d2psi/dw2>0`, `A1=d2phi/dw2=A1.T`, and `w dot S<=0`. Every shear and vertical derivative coupling must therefore arise from the same flux potential; algebraic responsive-height foldback is forbidden.",
            "",
            "The implemented normal form maps primitive perturbations to the hydrostatic departure `y_H=dlnH-H_Sigma dlnSigma-H_T dlnT`. For any invertible map `L`, positive metric `M`, and symmetric flux Hessian `K`, `A0=L.T M L` and `A1=L.T K L` are symmetric hyperbolic. Choosing `m_H=m_w omega_H^2` makes the undamped height/vertical-momentum exchange entropy conservative; stress and vertical damping give non-positive entropy production.",
            "",
            "The vertical-equilibrium restriction `y_H=w_H=0` is a five-dimensional compression. Its speeds interlace the seven-field speeds by the symmetric generalized Rayleigh principle, which is the local subcharacteristic proof template.",
            "",
            "## Reference identity audit",
            "",
            f"- temporal minimum eigenvalue: `{audit['temporal_minimum_eigenvalue']:.6e}`",
            f"- temporal condition number: `{audit['temporal_condition_number']:.6e}`",
            f"- maximum characteristic speed over c: `{audit['maximum_absolute_characteristic_speed_over_c']:.6e}`",
            f"- generalized eigenpair defect: `{audit['generalized_eigenpair_defect']:.6e}`",
            f"- energy-metric orthogonality defect: `{audit['energy_metric_orthogonality_defect']:.6e}`",
            f"- entropy-source positive part: `{audit['source_entropy_positive_part']:.6e}`",
            f"- vertical Hamiltonian entropy defect: `{audit['vertical_hamiltonian_entropy_defect']:.6e}`",
            f"- subcharacteristic interlacing violation: `{audit['subcharacteristic_interlacing_violation']:.6e}`",
            "",
            "The coefficients in this fixture are nondimensional and are not a fit or physical calibration.",
            "",
            "## Compatibility boundary",
            "",
            "The repaired system must reproduce the old model only on its certified pre-boundary domain. It may not be forced to inherit the committed nonhyperbolic face-3 symbol. Finite vertical inertia is binding beyond that domain.",
            "",
            "## Literature basis and limitation",
            "",
            "The entropy-extension/interlacing structure follows Chen--Levermore--Liu (1994). The symmetric Onsager normal form follows the linear near-equilibrium structure described by Gavassino (2022). The next package must map the shear subsystem to the nonlinear Israel--Stewart causality and strong-hyperbolicity inequalities of Cordeiro et al. (2026). These references do not certify this disk-column reduction; its exact potentials remain a binding local derivation task.",
            "",
            "## Decision",
            "",
            f"Authorized next artifact: `{AUTHORIZED_NEXT}` only. It must remain definitions-only, derive the physical Kerr--Schild entropy/flux potentials, and freeze the local audit envelope. No seven-field trajectory, fixed-Q orbit, slow atlas, or complete cycle is authorized.",
            "",
        )
    )


def _freeze() -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("Stage-1 seven-field architecture is already frozen")
    parent = _validate_parent(require_clean=True)
    contract = _contract()
    reference, arrays = _reference_payload()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    _write_json(CANONICAL_DIRECTORY / "stage1_contract.json", contract)
    _write_json(CANONICAL_DIRECTORY / "normal_form_audit.json", reference)
    with (CANONICAL_DIRECTORY / "normal_form_matrices.npz").open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    _write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_artifact": PARENT_ARTIFACT,
            "parent_hashes": parent["hashes"],
            "parent_classification": parent["summary"]["classification"],
            "parent_result_commit": PARENT_RESULT_COMMIT,
            "architecture_commit": ARCHITECTURE_COMMIT,
            "architecture_tree": ARCHITECTURE_TREE,
            "architecture_report": ARCHITECTURE_REPORT,
            "architecture_report_sha256": ARCHITECTURE_REPORT_SHA256,
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "seven_field_dimension": N_SEVEN_FIELDS,
        "quadratic_entropy_normal_form_certified": True,
        "reference_identity_audit_passed": reference["passed"],
        "physical_Kerr_Schild_seven_field_closure_certified": False,
        "nonlinear_global_symmetrizer_certified": False,
        "new_trajectory_steps": 0,
        "five_field_trajectory_remains_stopped": True,
        "seven_field_trajectory_authorized": False,
        "fixed_Q_invariant_object_authorized": False,
        "slow_flux_atlas_authorized": False,
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
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
            "normal_form_source": NORMAL_FORM_SOURCE,
            "normal_form_test": NORMAL_FORM_TEST,
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
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_report(reference), encoding="utf-8")
    _update_catalog(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    arguments = parser.parse_args()
    if not arguments.freeze:
        parser.error("--freeze is required")
    print(json.dumps(_plain(_freeze()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
