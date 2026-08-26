#!/usr/bin/env python3
"""Freeze the analytic partial-equilibrium slow architecture."""

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

import run_causal_inner_entropy_complete_fixed_q_adaptive_trust_diagnosis_execution_wp10c9d6c7c3b5c4f25fizey as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fizez_"
    "entropy_complete_analytic_quasisteady_closure_manifest"
)
CLASSIFICATION = (
    "entropy_complete_partial_equilibrium_Q3_plus_radial_stress_"
    "architecture_manifest_frozen"
)
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizfa_"
    "entropy_complete_hydrostatic_invariant_reconstruction_implementation"
)
ARTIFACT = (
    "causal_inner_entropy_complete_analytic_quasisteady_closure_manifest_"
    "wp10c9d6c7c3b5c4f25fizez"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_ANALYTIC_"
    "QUASISTEADY_CLOSURE_MANIFEST_WP10C9D6C7C3B5C4F25FIZEZ_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_entropy_complete_analytic_quasisteady_closure_"
    "manifest_wp10c9d6c7c3b5c4f25fizez.py"
)
THIS_TEST = (
    "tests/test_causal_inner_entropy_complete_analytic_quasisteady_closure_"
    "manifest_wp10c9d6c7c3b5c4f25fizez.py"
)
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "0a1d2c6266f4471c334a49a5b987adf80aa5e9e5224a2381532afb6fd44d7bb0"
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
        raise RuntimeError("adaptive-trust diagnosis checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = utils._read_json(
        parent.CANONICAL_DIRECTORY / "diagnosis_metrics.json"
    )
    if (
        summary["classification"] != parent.NO_DIRECTION_CLASSIFICATION
        or summary["passed"]
        or not summary["fresh_equation_linearization_certified"]
        or not summary["generic_fixed_point_Newton_rejected"]
        or not summary["analytic_quasisteady_closure_manifest_authorized"]
        or summary["authorized_next"]
        != "definitions_only_WP10c9d6c7c3b5c4f25fizez_entropy_complete_fixed_Q_analytic_quasisteady_closure_manifest"
        or metrics["maximum_equation_JVP_relative_defect"] > 2.0e-5
        or metrics["full_physical_candidate_evaluations"] != 0
        or metrics["propagated_states"] != 0
    ):
        raise RuntimeError("generic fixed-point rejection classification changed")
    for relative, expected in utils._read_json(
        parent.CANONICAL_DIRECTORY / "provenance.json"
    )["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"adaptive-trust diagnosis source changed: {relative}")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("analytic closure manifest requires a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "metrics": metrics}


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "preserved_negative_result": {
            "generic_global_fixed_Q_Newton_rejected": True,
            "absence_of_a_physical_fast_attractor_not_claimed": True,
            "no_failed_candidate_propagated": True,
            "no_threshold_relaxed": True,
        },
        "mathematical_state": {
            "cellwise_exact_slow_invariants": (
                "mass_M",
                "angular_momentum_J",
                "total_column_energy_E",
            ),
            "cellwise_resolved_auxiliaries": (
                "radial_velocity_over_c_beta_r",
                "causal_specific_shear_stress_chi",
            ),
            "cellwise_online_dimension": 5,
            "vertical_height_is_local_algebraic_equilibrium": True,
            "vertical_velocity_over_c_is_zero": True,
            "radial_drift_is_not_forced_to_zero": True,
            "causal_stress_is_not_forced_to_its_instantaneous_alpha_target": True,
            "no_global_fast_fixed_point_is_assumed": True,
        },
        "analytic_reconstruction": {
            "given": "(M,J,E,beta_r,chi)_per_cell",
            "unknown_local_charts": ("lnSigma", "beta_phi", "lnT"),
            "height_relation": "Pi(Sigma,T,H)=Sigma*Omega_perp**2*H**2",
            "implementation_height_map": "QuasiHydrostaticGasRadiationColumnEOS",
            "vertical_velocity_over_c": 0.0,
            "exact_constraints": "cell_measure*U7[M,J,E]=(M,J,E)_target",
            "maximum_constraint_relative_defect": 1.0e-10,
            "maximum_newton_corrections": 8,
            "maximum_scaled_local_inverse_condition_number": 1.0e8,
            "maximum_local_inverse_JVP_relative_defect": 2.0e-5,
            "accepted_state_template_may_seed_Newton_but_not_change_the_solution": True,
        },
        "differential_auxiliary_closure": {
            "beta_r_rate": "projected_seven_field_radial_momentum_equation_or_certified_atlas",
            "chi_rate": "projected_Maxwell_Cattaneo_shear_equation_or_certified_atlas",
            "height_and_vertical_momentum_rates": "eliminated_by_the_proved_local_equilibrium_pair",
            "exact_MJE_rates": "single_valued_conservative_face_fluxes_plus_sources",
            "L_stable_implicit_or_IMEX_macro_update_required": True,
            "five_field_reduced_characteristic_pencil_must_not_be_used": True,
            "every_offline_truth_state_is_a_reconstructed_seven_field_state": True,
        },
        "offline_validation": {
            "primary_profile": "primary_20ms",
            "heldout_profile": "heldout_16ms",
            "selected_cell_indices": (0, 18, 36, 55, 74, 92, 111),
            "relative_invariant_perturbation": 1.0e-6,
            "radial_velocity_chart_perturbation": 1.0e-6,
            "stress_relative_perturbation": 1.0e-4,
            "anchor_chart_reproduction_relative_defect": 1.0e-11,
            "all_reconstructed_center_and_face_pencils_must_be_real": True,
            "midpoint_eigenvalue_imaginary_ratio_max": 1.0e-10,
            "existing_height_optical_depth_excision_and_causality_gates_retained": True,
        },
        "offline_online_split": {
            "truth_model": "112_cell_entropy_complete_seven_field_operator",
            "online_primary_radial_cells": 16,
            "online_cellwise_dimension": 5,
            "global_finite_memory_candidates": (0, 2, 4, 6),
            "maximum_online_dimension": 86,
            "online_truth_calls_per_macrostep": 0,
            "maximum_macrosteps_per_cycle": 100_000,
            "maximum_average_wall_seconds_per_macrostep": 1.0,
            "target_complete_cycle_wall_days": 3.0,
            "conservative_restriction_and_prolongation_required": True,
            "heldout_anchor_required_before_cycle_manifest": True,
        },
        "claim_boundary": {
            "local_reconstruction_implementation_authorized": True,
            "slow_flux_atlas_authorized": False,
            "online_macro_solver_authorized": False,
            "complete_cycle_execution_authorized": False,
            "predictive_reduced_slow_evolution_authorized": False,
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


def _freeze() -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("analytic closure manifest already exists")
    validated = _validate_parent(require_clean=True)
    utils = _utils()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(
        CANONICAL_DIRECTORY / "analytic_quasisteady_closure_contract.json",
        _contract(),
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "generic_global_fixed_Q_Newton_rejection_preserved": True,
        "partial_equilibrium_Q3_plus_radial_stress_architecture_selected": True,
        "local_reconstruction_implementation_authorized": True,
        "slow_flux_atlas_authorized": False,
        "online_macro_solver_authorized": False,
        "complete_cycle_execution_authorized": False,
        "predictive_reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utils._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_artifact": parent.ARTIFACT,
            "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256,
            "parent_hashes": validated["hashes"],
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Entropy-complete analytic quasisteady closure manifest",
                "",
                f"Classification: `{CLASSIFICATION}`.",
                "",
                "The selected macro state is cellwise exact `(M,J,E)` plus resolved radial drift and causal stress. Only the vertical height/momentum pair is eliminated analytically. No global fast fixed point is assumed.",
                "",
                "Every offline flux state must be reconstructed in the seven-field model; the failed five-field characteristic pencil is forbidden. This package executes no truth call, root, trajectory, atlas, or cycle.",
                "",
                f"Authorized next: `{AUTHORIZED_NEXT}` only.",
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
            "scientific_status": "DEFINITIONS_ONLY",
            "implementation_commit": utils._git("rev-parse", "HEAD"),
            "implementation_tree": utils._git("rev-parse", "HEAD^{tree}"),
            "source_hashes": {
                path: utils._sha256(ROOT / path) for path in sources
            },
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
    args = parser.parse_args()
    if not args.freeze:
        parser.error("choose --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
