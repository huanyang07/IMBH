#!/usr/bin/env python3
"""Freeze the analytic material-current differentiation repair.

The parent local audit remains a binding negative result.  Its first failure
is an order-roundoff complex splitting of the exactly repeated advective
root.  This definitions-only package prospectively authorizes replacing
independent finite differences of the three exact product fluxes ``v U`` by
the analytic product rule ``v dU + U dv``.  It authorizes only a saved-point,
nonpropagating certificate before any full-envelope retry.
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

import run_causal_inner_entropy_complete_projected_local_structural_audit_wp10c9d6c7c3b5c4f25fizee as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fizee1_"
    "analytic_material_current_differentiation_repair_manifest"
)
CLASSIFICATION = (
    "analytic_material_current_differentiation_repair_manifest_frozen"
)
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizee2_"
    "saved_advective_degeneracy_repair_certificate"
)
ARTIFACT = (
    "causal_inner_analytic_material_current_differentiation_repair_manifest_"
    "wp10c9d6c7c3b5c4f25fizee1"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ANALYTIC_MATERIAL_CURRENT_"
    "DIFFERENTIATION_REPAIR_MANIFEST_WP10C9D6C7C3B5C4F25FIZEE1_"
    "2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_analytic_material_current_differentiation_"
    "repair_manifest_wp10c9d6c7c3b5c4f25fizee1.py"
)
THIS_TEST = (
    "tests/test_causal_inner_analytic_material_current_differentiation_"
    "repair_manifest_wp10c9d6c7c3b5c4f25fizee1.py"
)
PHYSICAL_SOURCE = parent.PHYSICAL_SOURCE
PHYSICAL_TEST = parent.PHYSICAL_TEST
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "7d17a66591ac87ccaf8516d32a1f199c5697b2cd9ff13331be1f216983222e28"
)
PARENT_CLASSIFICATION = parent.HYPERBOLICITY_FAILURE
OLD_PHYSICAL_SOURCE_SHA256 = (
    "d599d6d3e16f9bcdc3e67dbe7ba4004ee7459b334ec8c0189de5c37e21c2425e"
)
OLD_PHYSICAL_TEST_SHA256 = (
    "92ec5feebe894dd711630d3b8a68f6577c74fde7ccde763967792733b3e96387"
)
SAVED_LABEL = "heldout_16ms_cell_027"
SAVED_RADIUS_CM = 4659926455.691107
SAVED_CHART7 = (
    4.741144066935414,
    -0.3258490203398499,
    0.669416858133401,
    15.03778152477332,
    0.00024494031012566665,
    19.915742072422567,
    0.0,
)
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _utils():
    return parent.parent.parent


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "parent_negative_result": {
            "artifact": parent.ARTIFACT,
            "classification": PARENT_CLASSIFICATION,
            "preserved_as_binding": True,
            "retroactive_reclassification_forbidden": True,
            "first_failure_label": SAVED_LABEL,
            "first_failure_radius_cm": SAVED_RADIUS_CM,
            "first_failure_chart7": SAVED_CHART7,
            "recorded_maximum_imaginary_speed_over_c": (
                1.1962800308420447e-10
            ),
            "frozen_gate": 1.0e-10,
        },
        "diagnosis": {
            "exact_material_currents": (
                "rest_mass",
                "rest_mass_times_height",
                "rest_mass_times_vertical_velocity",
            ),
            "exact_radial_flux_form": "F=v_transport*U",
            "observed_split_is_stencil_sensitive": True,
            "observed_split_is_not_stable_under_step_halving_or_doubling": True,
            "all_non_hyperbolicity_parent_gates_passed_at_saved_point": True,
            "repair_hypothesis": (
                "independent finite differences violate the exact product identity "
                "at roundoff and split the repeated advective root"
            ),
        },
        "authorized_source_repair": {
            "differentiate_state_and_transport_with_the_same_centered_stencil": True,
            "analytic_identity": "d(v*U)=v*dU+U*dv",
            "repaired_principal_rows": (0, 5, 6),
            "stress_energy_rows_other_than_rest_mass_unchanged": True,
            "shear_row_unchanged": True,
            "height_and_vertical_physics_unchanged": True,
            "eigenvalue_clipping_forbidden": True,
            "eigenvalue_projection_forbidden": True,
            "matrix_symmetrization_forbidden": True,
            "threshold_relaxation_forbidden": True,
        },
        "saved_point_certificate": {
            "scope": "one_saved_point_nonpropagating",
            "label": SAVED_LABEL,
            "radius_cm": SAVED_RADIUS_CM,
            "chart7": SAVED_CHART7,
            "derivative_step_factors": (2.0, 1.0, 0.5),
            "all_factors_binding": True,
            "require_exact_material_product_identity": True,
            "require_all_original_point_gates": True,
            "save_complete_temporal_and_radial_matrices": True,
            "save_complete_eigenvalues_and_diagnostics": True,
            "trajectory_steps": 0,
        },
        "binding_gates": {
            "maximum_imaginary_speed_over_c": 1.0e-10,
            "maximum_light_cone_excess_over_c": 1.0e-10,
            "maximum_eigenpair_relative_defect": 1.0e-8,
            "eigenvector_condition_number_max": 1.0e8,
            "scaled_temporal_condition_number_max": 1.0e8,
            "maximum_biorthogonality_defect": 1.0e-8,
            "maximum_projector_idempotence_defect": 1.0e-8,
            "physical_tensor_constraint_relative_defect_max": 1.0e-10,
            "source_energy_ledger_relative_defect_max": 1.0e-10,
            "reference_causality_margin_min": 1.0e-8,
            "dominant_energy_margin_min": 1.0e-8,
            "entropy_production_min": 0.0,
            "material_product_identity_relative_defect_max": 1.0e-12,
            "matrix_derivative_ladder_relative_defect_max": 1.0e-7,
            "all_factors_and_all_gates_required": True,
            "fail_closed": True,
        },
        "claim_boundary": {
            "differentiation_repair_authorized": True,
            "saved_point_certificate_authorized": True,
            "full_envelope_retry_authorized": False,
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
    utils = _utils()
    if utils._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != (
        PARENT_CHECKSUM_MANIFEST_SHA256
    ):
        raise RuntimeError("parent negative audit checksum manifest changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = utils._read_json(
        parent.CANONICAL_DIRECTORY / "audit_metrics.json"
    )
    failure = metrics["first_failure"]
    if (
        summary["classification"] != PARENT_CLASSIFICATION
        or summary["passed"]
        or summary["authorized_next"] is not None
        or failure["label"] != SAVED_LABEL
        or failure["radius_cm"] != SAVED_RADIUS_CM
        or tuple(failure["chart7"]) != SAVED_CHART7
        or failure["reasons"] != ["strong_hyperbolicity:complex_speed"]
    ):
        raise RuntimeError("parent negative classification changed")
    if utils._sha256(ROOT / PHYSICAL_SOURCE) != OLD_PHYSICAL_SOURCE_SHA256:
        raise RuntimeError("pre-repair physical source changed")
    if utils._sha256(ROOT / PHYSICAL_TEST) != OLD_PHYSICAL_TEST_SHA256:
        raise RuntimeError("pre-repair physical test changed")
    if require_clean and utils._git(
        "status", "--short", "--untracked-files=no"
    ):
        raise RuntimeError("repair manifest freeze requires a clean tracked tree")
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
            "# Analytic material-current differentiation repair manifest",
            "",
            f"Classification: `{CLASSIFICATION}`.",
            "",
            f"The parent `{PARENT_CLASSIFICATION}` result remains binding. It is not converted into a pass.",
            "",
            "The saved complex split is not stable under stencil refinement. The three material fluxes are exact products `F=v_transport*U`, but the rejected implementation differentiated `U` and `F` independently. This package prospectively replaces only those three flux derivatives by the analytic identity `d(vU)=v dU+U dv` using the same centered stencil.",
            "",
            "No eigenvalue is clipped or projected, no matrix is symmetrized, and no tolerance is changed. The first execution is restricted to the saved held-out point at factors 2, 1, and 0.5 and advances no trajectory.",
            "",
            f"Authorized next: `{AUTHORIZED_NEXT}` only. A full-envelope retry remains unauthorized until that saved-point certificate passes.",
            "",
        )
    )


def _freeze() -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("repair manifest already exists")
    utils = _utils()
    parent_data = _validate_parent(require_clean=True)
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "repair_contract.json", _contract())
    utils._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_artifact": parent.ARTIFACT,
            "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256,
            "parent_hashes": parent_data["hashes"],
            "old_physical_source_sha256": OLD_PHYSICAL_SOURCE_SHA256,
            "old_physical_test_sha256": OLD_PHYSICAL_TEST_SHA256,
            "saved_failure": parent_data["metrics"]["first_failure"],
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "parent_negative_result_preserved": True,
        "differentiation_repair_authorized": True,
        "saved_point_certificate_authorized": True,
        "full_envelope_retry_authorized": False,
        "new_trajectory_steps": 0,
        "spatial_discretization_authorized": False,
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
    source_paths = (
        THIS_RUNNER,
        THIS_TEST,
        PHYSICAL_SOURCE,
        PHYSICAL_TEST,
        REPORT_RELATIVE,
    )
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
