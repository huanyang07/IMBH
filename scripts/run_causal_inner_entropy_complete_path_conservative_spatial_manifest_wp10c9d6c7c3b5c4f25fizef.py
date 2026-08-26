#!/usr/bin/env python3
"""Freeze the entropy-complete seven-field path-conservative spatial method.

This definitions-only package selects a DLM complete-fluctuation interface
operator after the full local architecture pass.  It authorizes only local
and interface audits.  No semidiscrete trajectory or time step is authorized.
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

import run_causal_inner_invariant_cluster_local_structural_audit_wp10c9d6c7c3b5c4f25fizee7 as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fizef_"
    "entropy_complete_path_conservative_spatial_manifest"
)
CLASSIFICATION = (
    "entropy_complete_path_conservative_spatial_manifest_frozen"
)
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizeg_"
    "entropy_complete_path_conservative_interface_audit"
)
ARTIFACT = (
    "causal_inner_entropy_complete_path_conservative_spatial_manifest_"
    "wp10c9d6c7c3b5c4f25fizef"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_PATH_"
    "CONSERVATIVE_SPATIAL_MANIFEST_WP10C9D6C7C3B5C4F25FIZEF_"
    "2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_entropy_complete_path_conservative_spatial_"
    "manifest_wp10c9d6c7c3b5c4f25fizef.py"
)
THIS_TEST = (
    "tests/test_causal_inner_entropy_complete_path_conservative_spatial_"
    "manifest_wp10c9d6c7c3b5c4f25fizef.py"
)
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "b9d42c17ccadcd3aba9c1e55a0334ab5549ee94031b56b2b80a9fbfd015ce9ad"
)
PHYSICAL_SOURCE = parent.PHYSICAL_SOURCE
PHYSICAL_TEST = parent.PHYSICAL_TEST
PHYSICAL_SOURCE_SHA256 = parent.PHYSICAL_SOURCE_SHA256
PHYSICAL_TEST_SHA256 = parent.PHYSICAL_TEST_SHA256
SPATIAL_SOURCE = (
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_generalized_maxwell_cattaneo_spatial.py"
)
SPATIAL_TEST = (
    "tests/test_causal_inner_generalized_maxwell_cattaneo_spatial.py"
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
        "certified_parent": {
            "artifact": parent.ARTIFACT,
            "classification": parent.PASS_CLASSIFICATION,
            "complete_reduced_principal_certified": True,
            "advective_cluster_certified": True,
            "all_parent_positive_and_negative_results_preserved": True,
        },
        "mixed_spatial_form": {
            "equation": "M_t(q) q_t + M_R(q) q_R = S_lower(q,R)",
            "physical_conservative_rows": (0, 1, 2, 3),
            "exact_material_current_rows": (5, 6),
            "nonconservative_projected_shear_row": (4,),
            "physical_conservation_claim_limited_to_mass_and_Kerr_Schild_stress_energy": True,
            "height_and_vertical_momentum_use_exact_shared_material_fluxes": True,
            "no_nonconservative_derivative_hidden_as_a_lower_order_source": True,
        },
        "DLM_path": {
            "path": "straight_line_in_fixed_entropy_scaled_seven_primitive_chart",
            "fixed_chart_scales": (1.0, 0.1, 0.1, 1.0, 1.0e-4, 1.0, 0.03),
            "interface_geometry_and_transport_coefficients_recomputed_at_each_path_node": True,
            "binding_quadrature_order": 8,
            "quadrature_ladder": (4, 8, 16),
            "total_jump": "integral_0^1 M_R(Psi(s))*Psi_s ds",
            "conservative_row_exact_flux_difference_parity_required": True,
            "path_reversal_antisymmetry_required": True,
            "constant_state_zero_jump_required": True,
        },
        "complete_eigenbasis_dissipation": {
            "basis": "complete_midpoint_generalized_eigenbasis",
            "dissipation": "M_t R abs(Lambda) R_inverse delta_q",
            "negative_fluctuation": "0.5*(path_jump-dissipation)",
            "positive_fluctuation": "0.5*(path_jump+dissipation)",
            "negative_plus_positive_equals_total_path_jump": True,
            "shared_flux_from_both_sides_required_on_exact_flux_rows": True,
            "characteristic_quadratic_dissipation_nonnegative": True,
            "scalar_max_speed_Rusanov_forbidden": True,
            "eigenvalue_clipping_or_post_hoc_matrix_modification_forbidden": True,
        },
        "interface_audit_scope": {
            "primary_and_heldout_representatives": True,
            "saved_complex_split_point": True,
            "old_five_field_failed_face": True,
            "certified_large_jump_pair": True,
            "deterministic_off_equilibrium_witnesses": True,
            "smooth_small_jump_ladders": (1.0e-3, 5.0e-4, 2.5e-4),
            "finite_amplitude_reversal_tests": True,
            "no_cell_residual_or_time_advance": True,
        },
        "binding_gates": {
            "constant_state_absolute_jump_max": 1.0e-12,
            "conservative_and_material_flux_parity_relative_defect_max": 1.0e-8,
            "path_partition_and_split_relative_defect_max": 1.0e-8,
            "path_reversal_relative_defect_max": 1.0e-8,
            "quadrature_ladder_relative_defect_max": 1.0e-7,
            "smooth_limit_relative_defect_max": 1.0e-6,
            "shared_flux_relative_defect_max": 1.0e-8,
            "maximum_imaginary_speed_over_c": 1.0e-10,
            "maximum_light_cone_excess_over_c": 1.0e-10,
            "eigenvector_condition_number_max": 1.0e8,
            "characteristic_quadratic_dissipation_min": 0.0,
            "all_cases_and_all_gates_required": True,
            "fail_closed": True,
        },
        "claim_boundary": {
            "spatial_operator_implementation_authorized": True,
            "nonpropagating_interface_audit_authorized": True,
            "semidiscrete_cell_operator_authorized": False,
            "relaxation_limit_audit_authorized": False,
            "trajectory_authorized": False,
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
        raise RuntimeError("local architecture checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = utils._read_json(parent.CANONICAL_DIRECTORY / "audit_metrics.json")
    provenance = utils._read_json(parent.CANONICAL_DIRECTORY / "provenance.json")
    if (
        summary["classification"] != parent.PASS_CLASSIFICATION
        or not summary["passed"]
        or not summary["all_parent_results_preserved"]
        or not summary["complete_reduced_principal_certified"]
        or not summary["advective_cluster_certified"]
        or not summary["spatial_manifest_authorized"]
        or summary["authorized_next"]
        != "definitions_only_WP10c9d6c7c3b5c4f25fizef_entropy_complete_path_conservative_spatial_manifest"
        or metrics["first_failure"] is not None
    ):
        raise RuntimeError("local architecture authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"local architecture source changed: {relative}")
    if utils._sha256(ROOT / PHYSICAL_SOURCE) != PHYSICAL_SOURCE_SHA256:
        raise RuntimeError("physical source changed")
    if utils._sha256(ROOT / PHYSICAL_TEST) != PHYSICAL_TEST_SHA256:
        raise RuntimeError("physical test changed")
    if require_clean and utils._git(
        "status", "--short", "--untracked-files=no"
    ):
        raise RuntimeError("spatial manifest freeze requires clean tracked tree")
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
            "# Entropy-complete path-conservative spatial manifest",
            "",
            f"Classification: `{CLASSIFICATION}`.",
            "",
            "The certified seven-field PDE is mixed: four exact physical conservation laws, two exact material-current balances, and one projected nonconservative Israel--Stewart shear row. The selected DLM operator integrates the complete radial principal along a fixed straight primitive path and preserves exact flux-difference parity on every exact-flux row.",
            "",
            "Interface dissipation uses the complete midpoint eigenbasis and the absolute characteristic speeds. Negative and positive fluctuations must close to the DLM jump and define one shared flux from both sides on the exact-flux rows. Scalar max-speed Rusanov dissipation is not selected.",
            "",
            f"Authorized next: `{AUTHORIZED_NEXT}` only. It may implement and audit interfaces but may not assemble a cell trajectory or take a time step.",
            "",
        )
    )


def _freeze() -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("spatial manifest already exists")
    utils = _utils()
    parent_data = _validate_parent(require_clean=True)
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "spatial_contract.json", _contract())
    utils._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_artifact": parent.ARTIFACT,
            "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256,
            "parent_hashes": parent_data["hashes"],
            "parent_metrics": parent_data["metrics"],
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
        "local_architecture_certificate_preserved": True,
        "spatial_operator_implementation_authorized": True,
        "nonpropagating_interface_audit_authorized": True,
        "semidiscrete_cell_operator_authorized": False,
        "relaxation_limit_audit_authorized": False,
        "new_trajectory_steps": 0,
        "seven_field_trajectory_authorized": False,
        "fixed_Q_invariant_object_authorized": False,
        "slow_flux_atlas_authorized": False,
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_report(), encoding="utf-8")
    source_paths = (THIS_RUNNER, THIS_TEST, PHYSICAL_SOURCE, PHYSICAL_TEST, REPORT_RELATIVE)
    utils._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "DEFINITIONS_ONLY",
            "implementation_commit": utils._git("rev-parse", "HEAD"),
            "implementation_tree": utils._git("rev-parse", "HEAD^{tree}"),
            "source_hashes": {
                path: utils._sha256(ROOT / path) for path in source_paths
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
    arguments = parser.parse_args()
    if not arguments.freeze:
        parser.error("choose --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
