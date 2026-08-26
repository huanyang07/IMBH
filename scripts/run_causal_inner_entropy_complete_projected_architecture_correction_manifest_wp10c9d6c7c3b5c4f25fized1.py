#!/usr/bin/env python3
"""Freeze the entropy-complete projected seven-field architecture.

The parent manifest selected a generalized Maxwell--Cattaneo class before
the complete reduced principal was implemented.  This prospective correction
records the precise one-dimensional projected model actually being audited:
the original Israel--Stewart entropy-current term is retained, the temporal
and radial shear projections are both principal, and direct diagonalization
of the reduced seven-field pencil is binding.  No trajectory is authorized.
"""

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

import run_causal_inner_generalized_maxwell_cattaneo_architecture_manifest_wp10c9d6c7c3b5c4f25fized as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fized1_"
    "entropy_complete_projected_architecture_correction_manifest"
)
CLASSIFICATION = (
    "entropy_complete_projected_seven_field_architecture_manifest_frozen"
)
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizee_"
    "entropy_complete_projected_local_structural_audit"
)
ARTIFACT = (
    "causal_inner_entropy_complete_projected_architecture_correction_manifest_"
    "wp10c9d6c7c3b5c4f25fized1"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_PROJECTED_"
    "ARCHITECTURE_CORRECTION_MANIFEST_WP10C9D6C7C3B5C4F25FIZED1_"
    "2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_entropy_complete_projected_architecture_"
    "correction_manifest_wp10c9d6c7c3b5c4f25fized1.py"
)
THIS_TEST = (
    "tests/test_causal_inner_entropy_complete_projected_architecture_"
    "correction_manifest_wp10c9d6c7c3b5c4f25fized1.py"
)
PHYSICAL_SOURCE = (
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_generalized_maxwell_cattaneo.py"
)
PHYSICAL_TEST = "tests/test_causal_inner_generalized_maxwell_cattaneo.py"
PHYSICAL_IMPLEMENTATION_COMMIT = "926ad5b98e138fb77ce0b0ca533d31153c2ab1d4"
PHYSICAL_SOURCE_SHA256 = (
    "d599d6d3e16f9bcdc3e67dbe7ba4004ee7459b334ec8c0189de5c37e21c2425e"
)
PHYSICAL_TEST_SHA256 = (
    "92ec5feebe894dd711630d3b8a68f6577c74fde7ccde763967792733b3e96387"
)
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "b81f9d6ea9f25d72cc486b1f0fc8aa893f5c3ef761d284f88cf07432d40f29f6"
)
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "parent_architecture": {
            "artifact": parent.ARTIFACT,
            "classification": parent.CLASSIFICATION,
            "generalized_Maxwell_Cattaneo_class_preserved": True,
            "failed_Godunov_realization_preserved": True,
            "no_parent_numerical_result_is_reclassified": True,
        },
        "precise_reduced_model": {
            "domain": "axisymmetric vertically integrated one-dimensional radial disk",
            "variables": (
                "lnSigma",
                "beta_R",
                "beta_phi",
                "lnT",
                "chi_Rphi",
                "lnH",
                "beta_H",
            ),
            "conservative_rows": (
                "rest mass and three Kerr-Schild Killing stress-energy rows"
            ),
            "material_rows": "projected shear, height, and vertical momentum",
            "shear_tensor_background": (
                "Sigma*c^2*chi_Rphi*(e_R e_phi+e_phi e_R)"
            ),
            "one_amplitude_model_is_a_projected_disk_closure": True,
            "one_amplitude_model_is_not_claimed_invariant_under_the_full_five_component_shear_PDE": True,
        },
        "entropy_complete_shear_equation": {
            "coefficient": "b=tau_pi/(nu_s*T)",
            "equation": (
                "tau_pi*D chi+chi=nu_s*gamma_Rphi-"
                "0.5*tau_pi*chi*D log(b)"
            ),
            "gamma_Rphi": "-2*c*sigma_(R)(phi)",
            "both_temporal_and_radial_velocity_derivatives_retained": True,
            "stress_density_expansion_terms_cancel_against_the_full_entropy_current_term": True,
            "truncated_one_sided_expansion_term_forbidden": True,
            "extended_entropy_production": (
                "Sigma*c^2*chi^2/(nu_s*T)>=0"
            ),
        },
        "causality_and_hyperbolicity_standard": {
            "binding_object": "complete_reduced_7_by_7_radial_quasilinear_pencil",
            "characteristic_equation": "det(M^R-lambda*M^t)=0",
            "direct_Kerr_Schild_light_cone_containment": True,
            "complete_real_eigenbasis": True,
            "Cordeiro_Corollary_5_full_tensor_frozen_coefficient_screen": (
                "binding_conservative_reference_screen_but_not_theorem_equivalent_for_projected_model"
            ),
            "no_isolated_signal_speed_certificate": True,
            "no_post_hoc_symmetrization": True,
        },
        "derivative_and_constraint_audit": {
            "physical_state_and_flux_maps": "exact_covariant_formulas",
            "principal_differentiation": "sixth_order_centered_chart_derivatives",
            "derivative_step_factors": (2.0, 1.0, 0.5),
            "representative_and_old_failed_face_ladders_required": True,
            "four_velocity_normalization_required": True,
            "shear_symmetry_trace_and_orthogonality_required": True,
            "vertical_energy_exchange_and_entropy_sign_required": True,
        },
        "binding_gates": {
            "maximum_imaginary_speed_over_c": 1.0e-10,
            "maximum_light_cone_excess_over_c": 1.0e-10,
            "generalized_eigenpair_relative_defect_max": 1.0e-8,
            "eigenvector_condition_number_max": 1.0e8,
            "scaled_temporal_condition_number_max": 1.0e8,
            "biorthogonality_and_projector_defect_max": 1.0e-8,
            "physical_tensor_constraint_relative_defect_max": 1.0e-10,
            "derivative_ladder_relative_defect_max": 1.0e-7,
            "source_energy_ledger_relative_defect_max": 1.0e-10,
            "reference_causality_margin_min": 1.0e-8,
            "dominant_energy_margin_min": 1.0e-8,
            "entropy_production_min": 0.0,
            "advective_neighbor_subspace_cosine_min": 0.90,
            "all_points_and_all_gates_required": True,
            "fail_closed": True,
        },
        "frozen_envelope": {
            "reuse_stage2_audit_envelope_bitwise": True,
            "base_charts": 8401,
            "deterministic_witnesses": 47,
            "include_axis_height_vertical_velocity_and_stress_stencils": True,
            "old_failed_face_is_nonpropagating_negative_control": True,
        },
        "claim_boundary": {
            "architecture_selected": True,
            "architecture_certified": False,
            "local_structural_audit_authorized": True,
            "spatial_discretization_authorized": False,
            "trajectory_authorized": False,
            "fixed_Q_invariant_object_authorized": False,
            "slow_flux_atlas_authorized": False,
            "complete_cycle_execution_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
        "authorized_next": AUTHORIZED_NEXT,
    }


def _validate_parent(*, require_clean: bool) -> dict:
    directory = parent.CANONICAL_DIRECTORY
    if parent._sha256(directory / "SHA256SUMS.txt") != PARENT_CHECKSUM_MANIFEST_SHA256:
        raise RuntimeError("parent architecture checksum manifest changed")
    hashes = parent._validate_checksums(directory)
    summary = parent._read_json(directory / "summary.json")
    contract = parent._read_json(directory / "architecture_contract.json")
    if (
        summary["classification"] != parent.CLASSIFICATION
        or not summary["passed"]
        or not summary["definitions_only"]
        or summary["authorized_next"] != parent.AUTHORIZED_NEXT
        or contract["selected_PDE_class"]["global_Godunov_potential_required"]
        or not contract["claim_boundary"]["local_structural_audit_authorized"]
    ):
        raise RuntimeError("parent architecture classification changed")
    if parent._sha256(ROOT / PHYSICAL_SOURCE) != PHYSICAL_SOURCE_SHA256:
        raise RuntimeError("entropy-complete physical source changed")
    if parent._sha256(ROOT / PHYSICAL_TEST) != PHYSICAL_TEST_SHA256:
        raise RuntimeError("entropy-complete physical test changed")
    if parent._git("rev-parse", PHYSICAL_IMPLEMENTATION_COMMIT) != PHYSICAL_IMPLEMENTATION_COMMIT:
        raise RuntimeError("entropy-complete implementation commit is unavailable")
    if require_clean and parent._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("architecture correction freeze requires a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "contract": contract}


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
                    "sha256": parent._sha256(path),
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
    catalog = parent._read_json(CANONICAL_SUMMARY)
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
            "latest_source_parent_commit": parent._git("rev-parse", "HEAD"),
            "latest_work_package": WORK_PACKAGE,
        }
    )
    parent._write_json(CANONICAL_SUMMARY, catalog)


def _report() -> str:
    return "\n".join(
        (
            "# Entropy-complete projected seven-field architecture correction",
            "",
            f"Classification: `{CLASSIFICATION}`.",
            "",
            "This prospective correction makes the reduced mathematical claim precise before the local audit. The physical one-amplitude R-phi shear is a projected one-dimensional disk closure, not an invariant truncation of the full five-component shear-tensor PDE.",
            "",
            "The binding shear equation is the original Israel--Stewart entropy-current-complete specific-stress law `tau D chi + chi = nu gamma - (tau chi/2) D log(tau/(nu T))`. Both temporal and radial pieces of the projected shear rate are principal. This form has nonnegative extended-entropy production and avoids retaining only one side of the stress-density expansion cancellation.",
            "",
            "Causality and strong hyperbolicity will be decided directly from the complete reduced 7x7 Kerr--Schild radial pencil. The Cordeiro et al. full-tensor inequalities remain a conservative reference screen; they are not presented as a theorem-equivalent proof for the projected model.",
            "",
            f"Authorized next: `{AUTHORIZED_NEXT}` only. No spatial step, trajectory, fixed-Q orbit, slow atlas, or complete-cycle execution is authorized.",
            "",
        )
    )


def _freeze() -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("architecture correction manifest already exists")
    parent_data = _validate_parent(require_clean=True)
    CANONICAL_DIRECTORY.mkdir(parents=True)
    parent._write_json(CANONICAL_DIRECTORY / "architecture_contract.json", _contract())
    parent._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_artifact": parent.ARTIFACT,
            "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256,
            "parent_hashes": parent_data["hashes"],
            "physical_implementation_commit": PHYSICAL_IMPLEMENTATION_COMMIT,
            "physical_source_sha256": PHYSICAL_SOURCE_SHA256,
            "physical_test_sha256": PHYSICAL_TEST_SHA256,
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "architecture_selected": True,
        "architecture_certified": False,
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
    parent._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_report(), encoding="utf-8")
    source_paths = (THIS_RUNNER, THIS_TEST, PHYSICAL_SOURCE, PHYSICAL_TEST, REPORT_RELATIVE)
    parent._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "DEFINITIONS_ONLY",
            "implementation_commit": parent._git("rev-parse", "HEAD"),
            "implementation_tree": parent._git("rev-parse", "HEAD^{tree}"),
            "source_hashes": {
                path: parent._sha256(ROOT / path) for path in source_paths
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
            f"{parent._sha256(CANONICAL_DIRECTORY / name)}  {name}\n"
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
