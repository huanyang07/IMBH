#!/usr/bin/env python3
"""Freeze the seven-field semidiscrete and equilibrium-relaxation contract."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_entropy_complete_path_conservative_interface_audit_wp10c9d6c7c3b5c4f25fizeg as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fizeh_"
    "entropy_complete_semidiscrete_relaxation_manifest"
)
CLASSIFICATION = "entropy_complete_semidiscrete_relaxation_manifest_frozen"
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizei_"
    "entropy_complete_semidiscrete_relaxation_audit"
)
ARTIFACT = (
    "causal_inner_entropy_complete_semidiscrete_relaxation_manifest_"
    "wp10c9d6c7c3b5c4f25fizeh"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_"
    "SEMIDISCRETE_RELAXATION_MANIFEST_WP10C9D6C7C3B5C4F25FIZEH_"
    "2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_entropy_complete_semidiscrete_relaxation_"
    "manifest_wp10c9d6c7c3b5c4f25fizeh.py"
)
THIS_TEST = (
    "tests/test_causal_inner_entropy_complete_semidiscrete_relaxation_"
    "manifest_wp10c9d6c7c3b5c4f25fizeh.py"
)
SOURCE = (
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_generalized_maxwell_cattaneo_semidiscrete.py"
)
SOURCE_TEST = (
    "tests/test_causal_inner_generalized_maxwell_cattaneo_semidiscrete.py"
)
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "c5541773e176856cc1f3b2eb2236447912dabc8b163d91a04f397e344505103b"
)
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _utils():
    return parent._utils()


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "equation_space": {
            "quasilinear_form": "M_t(q) dq/d(ct) + M_R(q) dq/dR = S_lower(q,R)",
            "exact_flux_rows": (0, 1, 2, 3, 5, 6),
            "projected_shear_row": 4,
            "chart": (
                "log_surface_density",
                "radial_velocity_over_c",
                "azimuthal_velocity_over_c",
                "log_temperature",
                "specific_shear_stress",
                "log_proper_half_thickness",
                "vertical_velocity_over_c",
            ),
        },
        "local_lower_source": {
            "physical_rows": "Kerr_Schild_geometry_plus_radiative_cooling_only",
            "vertical_pressure_work_is_not_an_extra_energy_source": True,
            "reason": (
                "vertical_kinetic_and_potential_energy_are_in_exact_total_"
                "energy_storage; pressure/gravity/damping exchange through_"
                "the coupled temporal map"
            ),
            "shear_row": (
                "S_chi=(nu_over_tau*gamma_connection-chi_over_tau)/c"
            ),
            "shear_connection_evaluated_with_zero_explicit_state_derivative": True,
            "height_row": "S_H=D*beta_H/u0",
            "vertical_momentum_row": "S_PH=D*a_H/(c*u0)",
            "vertical_acceleration": (
                "a_H=Pi/(Sigma*H)-Omega_perp^2*H-"
                "gamma_H*c*beta_H"
            ),
            "vertical_damping": "gamma_H=alpha*Omega_perp",
            "stream_source_added_once_only_at_cell_integration": True,
        },
        "semidiscrete_operator": {
            "first_order_audit_scheme": True,
            "interior_interfaces": "certified_complete_DLM_signed_fluctuations",
            "cell_equation": (
                "V_i*M_t(q_i)*dq_i/d(ct) + Dplus_left + Dminus_right "
                "= V_i*S_lower_i"
            ),
            "exact_rows_equivalent_to_one_shared_flux_difference": True,
            "nonconservative_shear_retained_as_signed_fluctuations": True,
            "temporal_solve": "equilibrated_dense_solve_per_cell",
            "periodic_fixed_geometry_audit_before_radial_boundaries": True,
            "no_reconstruction_in_first_audit": True,
            "no_source_or_interface_double_counting": True,
        },
        "equilibrium_relaxation_limit": {
            "equilibrium_embedding": (
                "E_R(q5)=(q5,log(H_hydrostatic(Sigma,T,R)),beta_H=0)"
            ),
            "height_force_must_vanish_on_embedding": True,
            "height_material_source_must_vanish_on_embedding": True,
            "causal_shear_remains_a_finite_rate_fifth_field": True,
            "tau_to_zero_at_fixed_viscosity_forbidden": True,
            "reason_tau_limit_forbidden": (
                "it would increase the viscous signal speed and violate the "
                "already certified causal subcharacteristic calibration"
            ),
            "fast_vertical_source_multipliers": (1.0, 2.0, 4.0, 8.0),
            "restoring_multiplier": "kappa^2",
            "damping_multiplier": "kappa",
            "well_prepared_limit_target": "hydrostatic_height_and_zero_vertical_velocity",
            "old_five_field_model_role": "pre_boundary_comparison_control_only",
            "old_failed_face_equivalence_claim_forbidden": True,
        },
        "audit_scope": {
            "primary_20ms": True,
            "heldout_16ms": True,
            "accepted_preboundary_states": True,
            "old_failed_face_nonpropagating": True,
            "periodic_constant_and_smooth_stencils": True,
            "independent_source_directional_derivatives": True,
            "no_radial_boundary_condition": True,
            "no_time_step": True,
        },
        "binding_gates": {
            "source_derivative_ladder_relative_defect_max": 1.0e-8,
            "hydrostatic_height_relative_defect_max": 1.0e-12,
            "equilibrium_vertical_source_relative_defect_max": 1.0e-10,
            "vertical_total_energy_ledger_relative_defect_max": 1.0e-10,
            "periodic_constant_operator_absolute_defect_max": 1.0e-10,
            "periodic_exact_flux_global_ledger_relative_defect_max": 1.0e-10,
            "periodic_signed_split_relative_defect_max": 1.0e-8,
            "temporal_solve_relative_residual_max": 1.0e-10,
            "minimum_fast_relaxation_observed_order": 0.8,
            "maximum_imaginary_speed_over_c": 1.0e-10,
            "maximum_light_cone_excess_over_c": 1.0e-10,
            "eigenvector_condition_number_max": 1.0e8,
            "all_cases_and_all_gates_required": True,
            "fail_closed": True,
        },
        "claim_boundary": {
            "local_source_implementation_authorized": True,
            "fixed_geometry_periodic_semidiscrete_implementation_authorized": True,
            "nonpropagating_relaxation_audit_authorized": True,
            "radial_boundary_implementation_authorized": False,
            "bounded_crossing_trajectory_authorized": False,
            "fixed_Q_invariant_object_authorized": False,
            "slow_flux_atlas_authorized": False,
            "complete_cycle_execution_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
        "authorized_next": AUTHORIZED_NEXT,
    }


def _validate_parent(*, require_clean: bool) -> dict:
    utils = _utils()
    if utils._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != (
        PARENT_CHECKSUM_MANIFEST_SHA256
    ):
        raise RuntimeError("interface certificate checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = utils._read_json(parent.CANONICAL_DIRECTORY / "audit_metrics.json")
    if (
        summary["classification"] != parent.PASS_CLASSIFICATION
        or not summary["passed"]
        or not summary["exact_flux_rows_certified"]
        or not summary["complete_eigenbasis_split_certified"]
        or summary["semidiscrete_cell_operator_authorized"]
        or summary["authorized_next"]
        != "definitions_only_WP10c9d6c7c3b5c4f25fizeh_entropy_complete_semidiscrete_relaxation_manifest"
        or metrics["first_failure"] is not None
    ):
        raise RuntimeError("interface certificate authorization changed")
    for relative, expected in utils._read_json(
        parent.CANONICAL_DIRECTORY / "provenance.json"
    )["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"interface certificate source changed: {relative}")
    if require_clean and utils._git(
        "status", "--short", "--untracked-files=no"
    ):
        raise RuntimeError("relaxation manifest requires clean tracked tree")
    return {"hashes": hashes, "summary": summary, "metrics": metrics}


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
        "passed": summary["passed"],
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


def _report() -> str:
    return "\n".join(
        (
            "# Entropy-complete semidiscrete relaxation manifest",
            "",
            f"Classification: `{CLASSIFICATION}`.",
            "",
            "This package freezes the local lower sources and a fixed-geometry periodic first-order DLM cell operator. The vertical relaxation limit is the hydrostatic equilibrium compression with finite causal shear relaxation; an acausal tau-to-zero limit at fixed viscosity is explicitly forbidden.",
            "",
            "The first audit is nonpropagating. Radial boundaries and a bounded crossing trajectory require separate prospective certificates.",
            "",
            f"Authorized next: `{AUTHORIZED_NEXT}` only.",
            "",
        )
    )


def _freeze() -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("semidiscrete relaxation manifest already exists")
    utils = _utils()
    validated = _validate_parent(require_clean=True)
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "relaxation_contract.json", _contract())
    utils._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_artifact": parent.ARTIFACT,
            "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256,
            "parent_hashes": validated["hashes"],
            "parent_metrics": validated["metrics"],
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "interface_certificate_preserved": True,
        "local_source_implementation_authorized": True,
        "periodic_semidiscrete_implementation_authorized": True,
        "nonpropagating_relaxation_audit_authorized": True,
        "radial_boundary_implementation_authorized": False,
        "new_trajectory_steps": 0,
        "bounded_crossing_trajectory_authorized": False,
        "fixed_Q_invariant_object_authorized": False,
        "slow_flux_atlas_authorized": False,
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_report(), encoding="utf-8")
    sources = (THIS_RUNNER, THIS_TEST, parent.SPATIAL_SOURCE, parent.SPATIAL_TEST, REPORT_RELATIVE)
    utils._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "DEFINITIONS_ONLY",
            "implementation_commit": utils._git("rev-parse", "HEAD"),
            "implementation_tree": utils._git("rev-parse", "HEAD^{tree}"),
            "source_hashes": {path: utils._sha256(ROOT / path) for path in sources},
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
    arguments = parser.parse_args()
    if not arguments.freeze:
        parser.error("choose --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
