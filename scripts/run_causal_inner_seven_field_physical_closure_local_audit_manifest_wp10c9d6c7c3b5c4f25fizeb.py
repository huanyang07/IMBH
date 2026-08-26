#!/usr/bin/env python3
"""Freeze the physical seven-field closure and local-audit manifest.

This work package is definitions-only.  It derives the binding mathematical
obligations for the physical Kerr--Schild seven-field closure, freezes the
committed state envelope, and authorizes one local structural audit.  It does
not implement the closure and it does not advance either trajectory.
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

SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fizeb_"
    "seven_field_physical_closure_local_audit_manifest"
)
CLASSIFICATION = (
    "seven_field_physical_closure_local_structural_audit_manifest_frozen"
)
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizec_"
    "seven_field_physical_closure_local_structural_audit"
)

STAGE1_ARTIFACT = (
    "causal_inner_seven_field_entropy_architecture_stage1_"
    "wp10c9d6c7c3b5c4f25fizea"
)
STAGE1_DIRECTORY = ROOT / "results/canonical" / STAGE1_ARTIFACT
STAGE1_CLASSIFICATION = (
    "seven_field_entropy_stage1_normal_form_derived_"
    "local_physical_closure_manifest_authorized"
)
STAGE1_RESULT_COMMIT = "5e4488c4eead3c7044de786d17e4ed2d02cc15ac"
STAGE1_CHECKSUM_MANIFEST_SHA256 = (
    "b8874ae43c3456ffb5607961fe0755c10dab8fa038364bc146735029d63220be"
)
STAGE1_CONTRACT_SHA256 = (
    "8f3195c7ca9455b3620e2e3c51441c6ed6211c3995f299f8357f192ef79cf064"
)
STAGE1_SUMMARY_SHA256 = (
    "605e53aa54da24b6dff8d739f9cfdfae8bc758249cd50dae286246842298edd7"
)

PRIMARY_ARTIFACT = (
    "causal_inner_face36_fixed_q_adaptive_refresh_primary_"
    "wp10c9d6c7c3b5c4f24e11"
)
HELDOUT_ARTIFACT = (
    "causal_inner_face36_fixed_q_adaptive_refresh_heldout_"
    "wp10c9d6c7c3b5c4f24e12"
)
BOUNDARY_EXECUTION_ARTIFACT = (
    "causal_inner_tangent_phase_hyperbolicity_two_half_step_bracket_"
    "execution_wp10c9d6c7c3b5c4f25fizdd"
)
BOUNDARY_MANIFEST_ARTIFACT = (
    "causal_inner_tangent_phase_lap_stage2_hyperbolicity_boundary_"
    "manifest_wp10c9d6c7c3b5c4f25fizda"
)
BOUNDARY_DIAGNOSTIC_ARTIFACT = (
    "causal_inner_tangent_phase_lap_stage2_hyperbolicity_boundary_"
    "diagnostic_wp10c9d6c7c3b5c4f25fizdb"
)

STATE_INPUTS = {
    "primary_20ms": {
        "artifact": PRIMARY_ARTIFACT,
        "checksum_manifest_sha256": (
            "c0194393c7e76f067847baddd7eb35d09aef732aea72c18a9935a4146db9efe4"
        ),
        "array_name": "decisive_arrays.npz",
        "array_sha256": (
            "f5d31e6950c733405c8e46b159ef8ebc561f0f63afcbcac36a9a9ac9b270a945"
        ),
        "classification": (
            "adaptive_refresh_primary_nonregression_passed_"
            "heldout_retry_manifest_authorized"
        ),
    },
    "heldout_16ms": {
        "artifact": HELDOUT_ARTIFACT,
        "checksum_manifest_sha256": (
            "7ddf3247ea7cd998a90b8682e92dfc3eeefe50bea286da4535234c3c65ba3027"
        ),
        "array_name": "decisive_arrays.npz",
        "array_sha256": (
            "3347ba27f6b2e20a2bcb8040d656502c7b8f8dd73ecdfa976ffd526f9da84c59"
        ),
        "classification": (
            "adaptive_refresh_heldout_coarse_passed_"
            "refined_ladder_manifest_authorized"
        ),
    },
    "accepted_boundary_trajectory": {
        "artifact": BOUNDARY_EXECUTION_ARTIFACT,
        "checksum_manifest_sha256": (
            "0bc10e9f1d21822f66db8e4e82ab21b00b71e3fcfec0ece08709cd6d4698389d"
        ),
        "array_name": "execution_arrays.npz",
        "array_sha256": (
            "519767878ec5bc9d24d59a8d13a6c0e76547e3585f7fa501512b7cb5fcd38231"
        ),
        "classification": "hyperbolicity_boundary_bracketed_after_first_half_step",
    },
    "rejected_full_step_profile": {
        "artifact": BOUNDARY_MANIFEST_ARTIFACT,
        "checksum_manifest_sha256": (
            "4dec5f8d5157964820b0c48f9e054554ba22bf38d6c7d22fdae7990db76962d3"
        ),
        "array_name": "boundary_seed.npz",
        "array_sha256": (
            "dc9280f04dca05e704b2c8fa20a12706c7cb5dcd383cd43dbf018cf8285a20e9"
        ),
        "classification": (
            "tangent_phase_lap_stage2_hyperbolicity_boundary_"
            "diagnosis_selected_definitions_only"
        ),
    },
    "failed_face_diagnostic": {
        "artifact": BOUNDARY_DIAGNOSTIC_ARTIFACT,
        "checksum_manifest_sha256": (
            "745fe70be9ea1184caf6247237c011b10c973b9acffd4052aeaa6301a97f4933"
        ),
        "array_name": "diagnostic_arrays.npz",
        "array_sha256": (
            "8f3d48cac8fac22281cbdb09e38e402c97aa05d338a385e5e4c0622d87a4b9a0"
        ),
        "classification": "genuine_local_hyperbolicity_loss_confirmed",
    },
}

ARTIFACT = (
    "causal_inner_seven_field_physical_closure_local_audit_manifest_"
    "wp10c9d6c7c3b5c4f25fizeb"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_SEVEN_FIELD_PHYSICAL_CLOSURE_"
    "LOCAL_AUDIT_MANIFEST_WP10C9D6C7C3B5C4F25FIZEB_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_seven_field_physical_closure_local_audit_"
    "manifest_wp10c9d6c7c3b5c4f25fizeb.py"
)
THIS_TEST = (
    "tests/test_causal_inner_seven_field_physical_closure_local_audit_"
    "manifest_wp10c9d6c7c3b5c4f25fizeb.py"
)
STAGE1_SOURCE = (
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_seven_field_entropy.py"
)
STAGE1_TEST = "tests/test_causal_inner_seven_field_entropy.py"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

SELECTED_CELL_INDICES = (0, 2, 36, 55, 111)
SELECTED_ACCEPTED_ENDPOINT_INDICES = (0, 17, 35, 53, 71)
HEIGHT_DEPARTURE_STENCIL = (-0.10, 0.0, 0.10)
VERTICAL_VELOCITY_OVER_C_STENCIL = (-0.03, 0.0, 0.03)
STRESS_AMPLITUDE_FACTORS = (0.0, 1.0, 1.25)
STRESS_SIGNS = (-1.0, 1.0)
AXIS_PERTURBATION_FRACTION = 0.02
AXIS_PERTURBATION_FLOORS = np.asarray(
    [5.0e-3, 1.0e-3, 1.0e-3, 5.0e-3, 1.0e-6],
    dtype=float,
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
    recorded = {}
    manifest = directory / "SHA256SUMS.txt"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        actual = _sha256(directory / name)
        if actual != expected:
            raise RuntimeError(f"checksum mismatch: {directory / name}")
        recorded[name] = actual
    return recorded


def _validate_stage1(*, require_clean: bool) -> dict:
    if _sha256(STAGE1_DIRECTORY / "SHA256SUMS.txt") != (
        STAGE1_CHECKSUM_MANIFEST_SHA256
    ):
        raise RuntimeError("Stage-1 checksum manifest changed")
    hashes = _validate_checksums(STAGE1_DIRECTORY)
    if (
        hashes["stage1_contract.json"] != STAGE1_CONTRACT_SHA256
        or hashes["summary.json"] != STAGE1_SUMMARY_SHA256
    ):
        raise RuntimeError("Stage-1 decisive definitions changed")
    summary = _read_json(STAGE1_DIRECTORY / "summary.json")
    contract = _read_json(STAGE1_DIRECTORY / "stage1_contract.json")
    claims = contract["claim_boundary"]
    obligations = contract["stage2_manifest_obligations"]
    if (
        summary["classification"] != STAGE1_CLASSIFICATION
        or not summary["passed"]
        or not summary["definitions_only"]
        or summary["authorized_next"]
        != "definitions_only_seven_field_physical_closure_and_local_structural_audit_manifest"
        or summary["physical_Kerr_Schild_seven_field_closure_certified"]
        or summary["seven_field_trajectory_authorized"]
        or not claims["definitions_only_stage2_manifest_authorized"]
        or not obligations["freeze_physical_parameter_envelope_before_eigenvalue_scan"]
        or not obligations["no_trajectory"]
    ):
        raise RuntimeError("Stage-1 authorization or claim boundary changed")
    if _git("rev-parse", STAGE1_RESULT_COMMIT) != STAGE1_RESULT_COMMIT:
        raise RuntimeError("Stage-1 result commit changed")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("Stage-2 manifest freeze requires a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "contract": contract}


def _validate_state_inputs() -> dict[str, dict]:
    validated = {}
    for label, specification in STATE_INPUTS.items():
        directory = ROOT / "results/canonical" / specification["artifact"]
        if _sha256(directory / "SHA256SUMS.txt") != specification[
            "checksum_manifest_sha256"
        ]:
            raise RuntimeError(f"{label} checksum manifest changed")
        hashes = _validate_checksums(directory)
        array_name = specification["array_name"]
        if hashes[array_name] != specification["array_sha256"]:
            raise RuntimeError(f"{label} decisive arrays changed")
        summary = _read_json(directory / "summary.json")
        if (
            summary["classification"] != specification["classification"]
            or not summary["passed"]
            or summary["reduced_slow_evolution_authorized"]
        ):
            raise RuntimeError(f"{label} scientific classification changed")
        validated[label] = {
            "directory": directory,
            "hashes": hashes,
            "summary": summary,
            "specification": specification,
        }
    boundary = validated["accepted_boundary_trajectory"]["summary"]
    diagnostic = validated["failed_face_diagnostic"]["summary"]
    if (
        boundary["combined_accepted_endpoints"] != 72
        or boundary["new_accepted_half_steps"] != 1
        or boundary["failed_full_step_propagated"]
        or boundary["trajectory_continuation_passed"]
        or diagnostic["first_complex_face"] != 3
        or not diagnostic["nonpropagating"]
        or diagnostic["analytic_maximum_imaginary_speed"] <= 1.0e-5
    ):
        raise RuntimeError("boundary state classification changed")
    return validated


def _load_npz(validated: dict, label: str) -> dict[str, np.ndarray]:
    entry = validated[label]
    path = entry["directory"] / entry["specification"]["array_name"]
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _unique_witnesses(
    candidates: list[tuple[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    labels = []
    charts = []
    seen = set()
    for label, chart in candidates:
        value = np.asarray(chart, dtype=float)
        if value.shape != (5,) or np.any(~np.isfinite(value)):
            raise RuntimeError("audit witness chart is invalid")
        key = value.tobytes()
        if key in seen:
            continue
        seen.add(key)
        labels.append(label)
        charts.append(value)
    return np.asarray(charts, dtype=float), np.asarray(labels, dtype="U96")


def _audit_envelope(
    validated: dict[str, dict] | None = None,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Return the canonical five-field envelope and seven-field stencils."""

    inputs = _validate_state_inputs() if validated is None else validated
    primary_data = _load_npz(inputs, "primary_20ms")
    heldout_data = _load_npz(inputs, "heldout_16ms")
    trajectory_data = _load_npz(inputs, "accepted_boundary_trajectory")
    failed_data = _load_npz(inputs, "rejected_full_step_profile")
    face_data = _load_npz(inputs, "failed_face_diagnostic")

    primary = np.asarray(
        primary_data["bdf1_primitive_charts"]
        - primary_data["bdf1_primitive_increment"],
        dtype=float,
    )
    heldout = np.asarray(
        heldout_data["bdf1_primitive_charts"]
        - heldout_data["bdf1_primitive_increment"],
        dtype=float,
    )
    accepted = np.asarray(
        trajectory_data["combined_accepted_endpoint_primitive_states"],
        dtype=float,
    )
    terminal = np.asarray(trajectory_data["current_primitive_state"], dtype=float)
    failed = np.asarray(failed_data["failed_retracted_primitive_state"], dtype=float)
    failed_face = np.asarray(face_data["failing_face_chart5"], dtype=float)
    if (
        primary.shape != (112, 5)
        or heldout.shape != (112, 5)
        or accepted.shape != (72, 112, 5)
        or terminal.shape != (112, 5)
        or failed.shape != (112, 5)
        or failed_face.shape != (5,)
        or not np.array_equal(terminal, accepted[-1])
    ):
        raise RuntimeError("committed audit-envelope shape or endpoint changed")

    pool = np.concatenate(
        (
            primary,
            heldout,
            accepted.reshape(-1, 5),
            failed,
            failed_face.reshape(1, 5),
        ),
        axis=0,
    )
    if np.any(~np.isfinite(pool)):
        raise RuntimeError("committed audit envelope is not finite")
    speed_squared = pool[:, 1] ** 2 + pool[:, 2] ** 2
    if np.max(speed_squared) >= 1.0:
        raise RuntimeError("committed audit envelope is not subluminal")
    lower = np.min(pool, axis=0)
    upper = np.max(pool, axis=0)
    span = upper - lower
    axis_steps = np.maximum(
        AXIS_PERTURBATION_FRACTION * span,
        AXIS_PERTURBATION_FLOORS,
    )

    candidates: list[tuple[str, np.ndarray]] = []
    for profile_label, profile in (
        ("primary_20ms", primary),
        ("heldout_16ms", heldout),
        ("accepted_terminal", terminal),
        ("rejected_full_step", failed),
    ):
        for cell_index in SELECTED_CELL_INDICES:
            candidates.append(
                (f"{profile_label}_cell_{cell_index:03d}", profile[cell_index])
            )
    for endpoint_index in SELECTED_ACCEPTED_ENDPOINT_INDICES:
        for cell_index in SELECTED_CELL_INDICES:
            candidates.append(
                (
                    f"accepted_{endpoint_index:02d}_cell_{cell_index:03d}",
                    accepted[endpoint_index, cell_index],
                )
            )
    candidates.append(("failed_face_003", failed_face))
    for field_index in range(5):
        candidates.append(
            (
                f"empirical_min_field_{field_index}",
                pool[int(np.argmin(pool[:, field_index]))],
            )
        )
        candidates.append(
            (
                f"empirical_max_field_{field_index}",
                pool[int(np.argmax(pool[:, field_index]))],
            )
        )
    witnesses, witness_labels = _unique_witnesses(candidates)
    for field_index in (1, 2):
        for sign in (-1.0, 1.0):
            perturbed = np.array(witnesses, copy=True)
            perturbed[:, field_index] += sign * axis_steps[field_index]
            if np.max(perturbed[:, 1] ** 2 + perturbed[:, 2] ** 2) >= 1.0:
                raise RuntimeError("velocity-axis stencil leaves the physical chart")

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "canonical_sources_only": True,
        "mutable_scratch_files_used": False,
        "five_field_chart_names": (
            "log_surface_density",
            "radial_velocity_over_c",
            "azimuthal_velocity_over_c",
            "log_temperature",
            "specific_shear_stress",
        ),
        "primary_profile_cells": int(primary.shape[0]),
        "heldout_profile_cells": int(heldout.shape[0]),
        "accepted_profiles": int(accepted.shape[0]),
        "accepted_profile_cells": int(accepted.shape[1]),
        "rejected_full_step_profile_cells": int(failed.shape[0]),
        "failed_face_count": 1,
        "empirical_base_chart_count": int(pool.shape[0]),
        "witness_chart_count": int(witnesses.shape[0]),
        "empirical_chart_minimum": lower,
        "empirical_chart_maximum": upper,
        "empirical_chart_span": span,
        "maximum_empirical_horizontal_speed_squared_over_c2": float(
            np.max(speed_squared)
        ),
        "axis_perturbation_fraction_of_empirical_span": (
            AXIS_PERTURBATION_FRACTION
        ),
        "axis_perturbation_floors": AXIS_PERTURBATION_FLOORS,
        "axis_perturbation_steps": axis_steps,
        "height_departure_stencil_log_H_over_H_eq": HEIGHT_DEPARTURE_STENCIL,
        "vertical_velocity_over_c_stencil": VERTICAL_VELOCITY_OVER_C_STENCIL,
        "stress_amplitude_factors": STRESS_AMPLITUDE_FACTORS,
        "stress_signs": STRESS_SIGNS,
        "selected_cell_indices": SELECTED_CELL_INDICES,
        "selected_accepted_endpoint_indices": (
            SELECTED_ACCEPTED_ENDPOINT_INDICES
        ),
        "failed_face_index": int(face_data["failing_face_index"]),
        "failed_face_radius_cm": float(face_data["failing_face_radius"]),
        "old_failed_face_maximum_imaginary_speed_over_c": float(
            np.max(np.abs(np.imag(face_data["analytic_eigenvalues5"])))
        ),
        "old_failed_face_is_diagnostic_only": True,
        "no_hyperrectangle_claim": True,
        "stencil_is_discrete_and_prospective": True,
    }
    arrays = {
        "primary_20ms_base_charts5": primary,
        "heldout_16ms_base_charts5": heldout,
        "accepted_trajectory_base_charts5": accepted,
        "accepted_terminal_base_charts5": terminal,
        "rejected_full_step_base_charts5": failed,
        "failed_face_chart5": failed_face,
        "empirical_chart_minimum5": lower,
        "empirical_chart_maximum5": upper,
        "empirical_chart_span5": span,
        "axis_perturbation_steps5": axis_steps,
        "witness_charts5": witnesses,
        "witness_labels": witness_labels,
        "height_departure_stencil": np.asarray(
            HEIGHT_DEPARTURE_STENCIL, dtype=float
        ),
        "vertical_velocity_over_c_stencil": np.asarray(
            VERTICAL_VELOCITY_OVER_C_STENCIL, dtype=float
        ),
        "stress_amplitude_factors": np.asarray(
            STRESS_AMPLITUDE_FACTORS, dtype=float
        ),
        "stress_signs": np.asarray(STRESS_SIGNS, dtype=float),
    }
    return metadata, arrays


def _contract() -> dict:
    """Return the binding Stage-2 physical derivation and audit contract."""

    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "authorization_basis": {
            "stage1_classification": STAGE1_CLASSIFICATION,
            "stage1_authorized_this_definitions_only_manifest": True,
            "explicit_user_instruction": "proceed with Stage 1 and Stage 2",
            "scope_is_definitions_only": True,
        },
        "preserved_boundary": {
            "five_field_trajectory_classification": (
                "hyperbolicity_boundary_bracketed_after_first_half_step"
            ),
            "accepted_endpoint_count": 72,
            "accepted_terminal_seconds": 0.18587500000000012,
            "first_complex_face": 3,
            "old_failed_candidate_propagated": False,
            "five_field_trajectory_remains_stopped": True,
        },
        "physical_primitives": {
            "q7": (
                "lnSigma",
                "beta_R",
                "beta_phi",
                "lnT",
                "chi",
                "lnH",
                "beta_H_equals_w_H_over_c",
            ),
            "rho": "Sigma/(2*H)",
            "horizontal_velocity_gate": "beta_R**2+beta_phi**2<1",
            "height_gate": "H>0",
            "vertical_velocity_gate": "abs(beta_H)<1",
            "internal_energy_gate": "e_int>0 after reservoir subtraction",
        },
        "covariant_densitization": {
            "rest_surface_density": "Sigma",
            "Valencia_rest_mass_storage": "D=Sigma*W",
            "rest_frame_height_content": "Sigma*H",
            "rest_frame_vertical_momentum": "Sigma*w_H",
            "conserved_height_content": "Z_H=D*H",
            "conserved_vertical_momentum": "P_H=D*w_H",
            "conserved_shear_coordinate": "R_pi=D*chi",
            "stage1_surface_mass_notation_resolved_as_D_in_coordinate_storage": True,
            "height_and_vertical_velocity_recovered_from": (
                "H=Z_H/D and w_H=P_H/D"
            ),
            "no_redundant_algebraic_height_unknown": True,
        },
        "rest_frame_total_energy": {
            "surface_energy": (
                "epsilon_col=Sigma*c**2+Sigma*e(rho,s)+E_H+E_pi"
            ),
            "vertical_reservoir": (
                "E_H=0.5*Sigma*w_H**2+"
                "0.5*Sigma*Omega_perp**2*H**2"
            ),
            "stress_reservoir": "E_pi=Sigma*chi**2/(2*a_pi), a_pi>0",
            "integrated_pressure": (
                "Pi is the thermodynamic derivative of Sigma*e at fixed "
                "column entropy and H"
            ),
            "vertical_force": "Pi/H-Sigma*Omega_perp**2*H",
            "hydrostatic_limit": "Pi=Sigma*Omega_perp**2*H**2",
            "reservoirs_contribute_to_relativistic_inertia": True,
            "quadratic_vertical_energy_is_a_local_model_on_the_frozen_beta_H_stencil": True,
        },
        "Kerr_Schild_state_and_flux": {
            "column_stress_energy": (
                "T_col^{ab}=((epsilon_col+Pi)/c**2)u^a u^b+"
                "(Pi/c**2)g^{ab}+pi^{ab}"
            ),
            "shear_tensor": (
                "pi^{ab}=Sigma*chi*(e_R^a e_phi^b+e_phi^a e_R^b), "
                "pi^a_a=0, pi^{ab}u_b=0"
            ),
            "first_four_state_entries": (
                "Kerr-Schild Killing projections of T_col and the rest-mass current"
            ),
            "first_four_flux_entries": (
                "radial Kerr-Schild Killing fluxes of the same T_col"
            ),
            "remaining_state_entries": "(R_pi,Z_H,P_H)",
            "remaining_fluxes": (
                "mass-current advection plus only couplings derived from the "
                "same entropy flux potential"
            ),
            "geometry_derivatives": "lower-order covariant-divergence sources",
            "existing_Kerr_Schild_projection_code_is_the_pre_boundary_control": True,
        },
        "physical_entropy_extension": {
            "internal_energy_recovery": (
                "subtract horizontal kinetic, vertical, gravitational, and "
                "stress reservoir energies from total column energy"
            ),
            "physical_entropy_density": "S_ext=D*s(rho,e_int)",
            "mathematical_entropy": "eta(U7)=-S_ext",
            "entropy_variables": "w=deta/dU7",
            "Legendre_state_potential": "psi(w)=w dot U7-eta(U7)",
            "entropy_flux": "q_eta is the radial flux of -S_ext",
            "Legendre_flux_potential": "phi(w)=w dot F7-q_eta",
            "state_identity": "U7=dpsi/dw",
            "flux_identity": "F7=dphi/dw",
            "temporal_symmetrizer": "A0=d2psi/dw2 positive definite",
            "spatial_symmetrizer": "A1=d2phi/dw2 symmetric",
            "no_post_hoc_matrix_symmetrization": True,
        },
        "shear_relaxation_closure": {
            "equation": (
                "tau_pi*u^a*nabla_a(chi)+chi="
                "nu_s*gamma_dot_Rphi plus covariantly required nonlinear terms"
            ),
            "alpha_target": "chi_alpha=alpha*Pi/(Sigma*c**2)",
            "reference_shear": "gamma_dot_ref=1.5*Omega_perp",
            "specific_viscosity": "nu_s=chi_alpha/gamma_dot_ref",
            "signal_calibration": "c_nu/c=sqrt(alpha)*c_s/c",
            "relaxation_time": "tau_pi=nu_s/(h_eff*(c_nu/c)**2)",
            "positive_coefficients": "nu_s>0, tau_pi>0, a_pi>0",
            "a_pi_is_fixed_by_entropy_conjugacy_and_the_same_nu_s_tau_pi_pair": True,
            "nonlinear_Israel_Stewart_conditions": (
                "evaluate the exact one-shear-amplitude specialization of the "
                "Cordeiro-et-al. nonlinear causality and strong-hyperbolicity inequalities"
            ),
            "full_seven_field_spectrum_is_binding_not_isolated_c_nu": True,
            "dominant_energy_gate": "abs(chi)<h_eff",
            "boundary_tuned_coefficient_or_floor_forbidden": True,
        },
        "vertical_balance_laws": {
            "height_current": "nabla_a(Sigma*H*u^a)=Sigma*w_H",
            "vertical_momentum_current": (
                "nabla_a(Sigma*w_H*u^a)="
                "Pi/H-Sigma*Omega_perp**2*H-gamma_H*Sigma*w_H"
            ),
            "structural_damping_calibration": "gamma_H=alpha*Omega_perp",
            "damping_is_lower_order": True,
            "constraint_definitions": (
                "C_Z=Z_H-D*H and C_P=P_H-D*w_H"
            ),
            "constraint_propagation": (
                "C_Z=C_P=0 follows identically from the state map, rest-mass "
                "balance, and the two covariant scalar-current balances"
            ),
            "vertical_exchange_is_entropy_conservative_when_gamma_H_zero": True,
            "vertical_damping_heats_internal_energy": True,
        },
        "source_and_energy_ledgers": {
            "stress_relaxation_heat": (
                "loss of E_pi plus mechanical shear work is added to internal energy"
            ),
            "vertical_damping_heat": "gamma_H*Sigma*w_H**2 is added to internal energy",
            "internal_relaxation_total_energy_defect": "identically zero",
            "mathematical_entropy_source": "w dot S<=0",
            "physical_entropy_production": "nabla_a(S_ext*u^a)>=0",
            "radiative_and_external_sources_remain_explicit": True,
            "geometry_source_identity_uses_the_same_T_col": True,
            "component_ledgers_may_not_hide_cancellation": True,
        },
        "frozen_state_envelope": {
            "canonical_sources_only": True,
            "primary_20ms_profile": True,
            "heldout_16ms_profile": True,
            "all_72_accepted_trajectory_profiles": True,
            "accepted_terminal_profile": True,
            "rejected_full_step_profile_nonpropagating": True,
            "independently_diagnosed_failed_face": True,
            "equilibrium_embedding_at_every_base_chart": True,
            "off_equilibrium_stencil_at_deterministic_witnesses": True,
            "height_departure_stencil_log_H_over_H_eq": HEIGHT_DEPARTURE_STENCIL,
            "vertical_velocity_over_c_stencil": VERTICAL_VELOCITY_OVER_C_STENCIL,
            "stress_sign_reflection": True,
            "stress_amplitude_factors": STRESS_AMPLITUDE_FACTORS,
            "five_field_axis_perturbation_fraction": AXIS_PERTURBATION_FRACTION,
            "no_continuous_hyperrectangle_claim": True,
            "mutable_scratch_files_forbidden": True,
        },
        "independent_derivative_audit": {
            "production_derivatives": (
                "analytic or automatic derivatives of U7, F7, eta, psi, and phi"
            ),
            "independent_method": (
                "sixth-order centered finite differences with two step ladders "
                "or a genuinely independent complex-step implementation"
            ),
            "independent_code_path_required": True,
            "differentiate_then_symmetrize_forbidden": True,
            "branch_and_limiter_pattern_frozen_per_stencil_point": True,
            "state_gradient_identity_audited": True,
            "flux_gradient_identity_audited": True,
            "Hessian_parity_audited": True,
            "source_directional_derivative_audited": True,
        },
        "old_model_parity_boundary": {
            "pre_boundary_equilibrium_embedding": "y_H=0 and beta_H=0",
            "pre_boundary_state_flux_source_parity_required": True,
            "pre_boundary_five_field_compressed_principal_parity_required": True,
            "pre_boundary_subcharacteristic_interlacing_required": True,
            "failed_face_exact_principal_parity_forbidden": True,
            "failed_face_old_complex_spectrum_is_a_negative_control": True,
            "failed_face_new_finite_inertia_spectrum_must_be_real_and_causal": True,
        },
        "binding_local_audit_gates": {
            "state_and_flux_potential_relative_gradient_defect_max": 1.0e-8,
            "analytic_independent_Hessian_relative_defect_max": 1.0e-6,
            "A0_relative_symmetry_defect_max": 1.0e-10,
            "A0_diagonally_equilibrated_minimum_eigenvalue_min": 1.0e-10,
            "A0_diagonally_equilibrated_condition_number_max": 1.0e8,
            "A1_relative_symmetry_defect_max": 1.0e-10,
            "maximum_characteristic_imaginary_speed_over_c": 1.0e-10,
            "maximum_absolute_characteristic_speed_over_c": 0.999999,
            "generalized_eigenpair_relative_defect_max": 1.0e-8,
            "energy_metric_orthogonality_relative_defect_max": 1.0e-8,
            "source_entropy_positive_part_max": 1.0e-10,
            "internal_relaxation_energy_ledger_relative_defect_max": 1.0e-10,
            "vertical_constraint_propagation_relative_defect_max": 1.0e-10,
            "pre_boundary_state_flux_source_parity_relative_defect_max": 1.0e-8,
            "pre_boundary_compressed_principal_parity_relative_defect_max": 1.0e-6,
            "pre_boundary_subcharacteristic_interlacing_violation_max": 1.0e-8,
            "failed_face_new_model_maximum_imaginary_speed_over_c": 1.0e-10,
            "failed_face_old_model_imaginary_speed_over_c_min": 1.0e-5,
            "all_points_and_all_gates_required": True,
            "fail_closed": True,
        },
        "stage3_execution_order": (
            "derive exact thermodynamic and Kerr-Schild state map",
            "derive eta, psi, entropy flux, and phi before assembling matrices",
            "prove vertical and shear source energy/entropy ledgers symbolically",
            "evaluate primary and held-out equilibrium profiles",
            "evaluate every accepted pre-boundary equilibrium profile",
            "evaluate deterministic off-equilibrium witness stencils",
            "evaluate rejected profile and failed face only in the new model",
            "run independent derivative audits",
            "freeze a positive or negative local structural certificate",
        ),
        "stage3_classifications": {
            "pass": "seven_field_physical_closure_local_structural_audit_passed",
            "derivation_failure": "seven_field_physical_closure_derivation_failed",
            "entropy_failure": "seven_field_physical_closure_entropy_failed",
            "hyperbolicity_failure": "seven_field_physical_closure_hyperbolicity_failed",
            "pre_boundary_parity_failure": "seven_field_pre_boundary_parity_failed",
            "no_failure_may_be_reclassified_as_a_trajectory_result": True,
        },
        "external_mathematical_basis": {
            "relaxation_entropy_and_subcharacteristic_structure": (
                "https://doi.org/10.1002/cpa.3160470602"
            ),
            "symmetric_hyperbolicity_from_entropy_variables": (
                "https://arxiv.org/abs/2210.05067"
            ),
            "nonlinear_Israel_Stewart_shear_conditions": (
                "https://arxiv.org/abs/2607.05639"
            ),
            "Godunov_variable_relaxation_example": (
                "https://arxiv.org/abs/2110.15223"
            ),
            "literature_does_not_certify_this_column_closure": True,
        },
        "budget": {
            "new_five_field_trajectory_steps": 0,
            "new_seven_field_trajectory_steps": 0,
            "new_nonlinear_roots": 0,
            "new_fixed_Q_roots": 0,
            "new_complete_cycle_steps": 0,
            "maximum_stage2_manifest_wall_minutes": 10,
        },
        "claim_boundary": {
            "stage1_quadratic_normal_form_preserved": True,
            "physical_audit_envelope_frozen": True,
            "physical_Kerr_Schild_seven_field_closure_certified": False,
            "nonlinear_global_symmetrizer_certified": False,
            "local_structural_audit_authorized": True,
            "seven_field_spatial_discretization_authorized": False,
            "seven_field_trajectory_authorized": False,
            "five_field_trajectory_restart_authorized": False,
            "fixed_Q_invariant_object_authorized": False,
            "slow_flux_atlas_authorized": False,
            "complete_cycle_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
        "forbidden": (
            "use mutable scratch output as a binding state",
            "fit a_pi, tau_pi, gamma_H, or a signal speed to face 3",
            "symmetrize a nonsymmetric Jacobian after differentiation",
            "discard vertical or shear relaxation heat",
            "force exact parity with the old complex failed-face pencil",
            "run a seven-field spatial step or trajectory",
            "restart the stopped five-field trajectory",
            "construct a fixed-Q orbit, slow atlas, or complete cycle",
        ),
        "authorized_next": AUTHORIZED_NEXT,
    }


def _source_hashes() -> dict[str, str]:
    paths = (THIS_RUNNER, THIS_TEST, STAGE1_SOURCE, STAGE1_TEST, REPORT_RELATIVE)
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


def _report(metadata: dict) -> str:
    lower = metadata["empirical_chart_minimum"]
    upper = metadata["empirical_chart_maximum"]
    return "\n".join(
        (
            "# Seven-field physical closure and local-audit manifest",
            "",
            f"Classification: `{CLASSIFICATION}`.",
            "",
            "This package is definitions-only. It freezes the physical Kerr--Schild closure obligations and the exact local audit envelope. It executes no eigenvalue campaign, nonlinear root, spatial step, or trajectory.",
            "",
            "## Covariant state architecture",
            "",
            "The seven primitives are `(ln Sigma, beta_R, beta_phi, ln T, chi, ln H, w_H/c)`. The rest-frame height contents `Sigma H` and `Sigma w_H` are densitized consistently with the mass current: the coordinate conserved variables are `Z_H=D H` and `P_H=D w_H`, where `D=Sigma W` is the Valencia rest-mass storage. Thus `H=Z_H/D` and `w_H=P_H/D`; height is not a redundant algebraic unknown.",
            "",
            "The local rest-frame total energy contains gas+radiation internal energy, vertical kinetic and gravitational reservoirs, and a positive shear-relaxation reservoir. These reservoirs enter the relativistic inertia before the Kerr--Schild Killing projections are formed. The same stress-energy tensor supplies state, flux, and geometric source terms.",
            "",
            "## Entropy/Godunov construction",
            "",
            "Internal energy is recovered from total energy after subtracting all mechanical and relaxation reservoirs. The binding mathematical entropy is minus the recovered column entropy. Its exact Legendre state and flux potentials must generate both `U7` and `F7`; post-hoc symmetrization is forbidden. Shear and vertical damping transfer reservoir energy to heat, so internal relaxation has zero total-energy defect and non-positive mathematical-entropy production.",
            "",
            "## Frozen evidence envelope",
            "",
            f"The package contains `{metadata['empirical_base_chart_count']}` canonical five-field base charts and `{metadata['witness_chart_count']}` deterministic witnesses. Inputs are the 20 ms primary profile, 16 ms held-out profile, all 72 accepted pre-boundary profiles, the rejected full-step profile, and the independently diagnosed failed face. Mutable scratch files are not used.",
            "",
            f"Empirical chart minimum: `{lower}`.",
            "",
            f"Empirical chart maximum: `{upper}`.",
            "",
            "Every base chart is audited at vertical equilibrium. Deterministic witnesses additionally use `ln(H/H_eq) in {-0.10,0,0.10}`, `w_H/c in {-0.03,0,0.03}`, sign-reflected shear amplitudes through 1.25 times the physical amplitude, and prospectively frozen axis perturbations. This is a discrete local stencil, not a claim over a continuous hyperrectangle.",
            "",
            "## Shear and vertical closure",
            "",
            "The alpha amplitude, reference shear, viscosity, and relaxation time retain the existing state-local calibration. The exact nonlinear one-shear Israel--Stewart causality and strong-hyperbolicity inequalities are binding in addition to the full seven-field spectrum. No coefficient or floor may be tuned to the failed face. The vertical balance is a covariant advected oscillator with `gamma_H=alpha Omega_perp`; damping is lower order and heats the internal reservoir.",
            "",
            "## Compatibility boundary",
            "",
            "On every certified pre-boundary equilibrium state, state/flux/source parity, compressed principal parity, and relaxation subcharacteristic interlacing are binding. At the failed face, exact principal parity is forbidden because it would force the repaired model to inherit the old complex spectrum. The old face is retained only as a negative control; the finite-inertia seven-field spectrum must be real and causal there.",
            "",
            "## Decision",
            "",
            f"Authorized next: `{AUTHORIZED_NEXT}` only. It may implement the local physical closure and structural audit under the frozen gates. It may not construct a spatial discretization or advance a trajectory. The stopped five-field trajectory remains stopped, and fixed-Q evolution, slow-flux mapping, a complete cycle, and reduced slow evolution remain unauthorized.",
            "",
        )
    )


def _freeze() -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("Stage-2 physical-closure manifest is already frozen")
    stage1 = _validate_stage1(require_clean=True)
    inputs = _validate_state_inputs()
    contract = _contract()
    metadata, arrays = _audit_envelope(inputs)

    CANONICAL_DIRECTORY.mkdir(parents=True)
    _write_json(CANONICAL_DIRECTORY / "local_audit_manifest.json", contract)
    _write_json(CANONICAL_DIRECTORY / "audit_envelope.json", metadata)
    with (CANONICAL_DIRECTORY / "audit_envelope.npz").open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    _write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "stage1_artifact": STAGE1_ARTIFACT,
            "stage1_result_commit": STAGE1_RESULT_COMMIT,
            "stage1_checksum_manifest_sha256": (
                STAGE1_CHECKSUM_MANIFEST_SHA256
            ),
            "stage1_hashes": stage1["hashes"],
            "state_inputs": {
                label: {
                    "artifact": entry["specification"]["artifact"],
                    "classification": entry["summary"]["classification"],
                    "checksum_manifest_sha256": entry["specification"][
                        "checksum_manifest_sha256"
                    ],
                    "decisive_array": entry["specification"]["array_name"],
                    "decisive_array_sha256": entry["specification"][
                        "array_sha256"
                    ],
                }
                for label, entry in inputs.items()
            },
            "canonical_sources_only": True,
            "mutable_scratch_files_used": False,
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "canonical_audit_envelope_frozen": True,
        "empirical_base_chart_count": metadata["empirical_base_chart_count"],
        "witness_chart_count": metadata["witness_chart_count"],
        "new_trajectory_steps": 0,
        "five_field_trajectory_remains_stopped": True,
        "physical_Kerr_Schild_seven_field_closure_certified": False,
        "nonlinear_global_symmetrizer_certified": False,
        "local_structural_audit_authorized": True,
        "seven_field_spatial_discretization_authorized": False,
        "seven_field_trajectory_authorized": False,
        "fixed_Q_invariant_object_authorized": False,
        "slow_flux_atlas_authorized": False,
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_report(metadata), encoding="utf-8")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    arguments = parser.parse_args()
    if not arguments.freeze:
        parser.error("--freeze is required")
    print(json.dumps(_plain(_freeze()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
