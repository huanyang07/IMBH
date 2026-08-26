#!/usr/bin/env python3
"""Freeze the invariant correction to the local hyperbolicity audit.

The two parent failures remain binding historical results.  This prospective
correction replaces only the non-invariant coarse endpoint-overlap heuristic
by pointwise uniform diagonalizability and invariant spectral-cluster gates.
The known large-jump path certificate is hash-locked supporting evidence.
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

import run_causal_inner_advective_subspace_path_continuity_certificate_wp10c9d6c7c3b5c4f25fizee5 as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fizee6_"
    "advective_path_continuity_audit_correction_manifest"
)
CLASSIFICATION = (
    "advective_path_continuity_audit_correction_manifest_frozen"
)
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizee7_"
    "invariant_cluster_local_structural_audit"
)
ARTIFACT = (
    "causal_inner_advective_path_continuity_audit_correction_manifest_"
    "wp10c9d6c7c3b5c4f25fizee6"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ADVECTIVE_PATH_CONTINUITY_"
    "AUDIT_CORRECTION_MANIFEST_WP10C9D6C7C3B5C4F25FIZEE6_"
    "2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_advective_path_continuity_audit_correction_"
    "manifest_wp10c9d6c7c3b5c4f25fizee6.py"
)
THIS_TEST = (
    "tests/test_causal_inner_advective_path_continuity_audit_correction_"
    "manifest_wp10c9d6c7c3b5c4f25fizee6.py"
)
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "552206781122ab7bc823c333dcf996d27b305dc881cc9a7a7ef8a87b14b1ea7c"
)
PHYSICAL_SOURCE = parent.PHYSICAL_SOURCE
PHYSICAL_TEST = parent.PHYSICAL_TEST
PHYSICAL_SOURCE_SHA256 = parent.PHYSICAL_SOURCE_SHA256
PHYSICAL_TEST_SHA256 = parent.PHYSICAL_TEST_SHA256
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _utils():
    return parent._utils()


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "preserved_results": {
            "original_complex_split_failure": {
                "classification": parent.parent.parent.parent.parent.PARENT_CLASSIFICATION,
                "preserved": True,
            },
            "coarse_neighbor_overlap_failure": {
                "classification": parent.parent.parent.HYPERBOLICITY_FAILURE,
                "preserved": True,
                "coarse_endpoint_cosine": 0.7380714589652135,
            },
            "saved_product_rule_repair": {
                "classification": parent.parent.parent.parent.PASS_CLASSIFICATION,
                "preserved": True,
            },
            "path_continuity_certificate": {
                "classification": parent.PASS_CLASSIFICATION,
                "preserved": True,
                "minimum_cluster_gap_over_c": 0.008508686578747426,
                "maximum_eigenvector_condition_number": 327.70521769545377,
                "maximum_imaginary_speed_over_c": 0.0,
                "projector_jump_refinement_ratios": (
                    0.5085785346518578,
                    0.5041740069261599,
                ),
            },
            "no_parent_result_reclassified": True,
        },
        "corrected_mathematical_standard": {
            "binding_object": "complete_reduced_7_by_7_radial_quasilinear_pencil",
            "strong_hyperbolicity_requires_real_spectrum_and_uniformly_bounded_complete_eigenbasis": True,
            "advective_cluster_dimension": 3,
            "advective_cluster_selection": "three eigenvalues nearest exact material transport speed",
            "cluster_transport_offset_and_cluster_complement_gap_binding": True,
            "coarse_neighbor_subspace_cosine": "diagnostic_only",
            "reason": (
                "endpoint overlap depends on finite state/geometry separation and "
                "is not an invariant pointwise strong-hyperbolicity condition"
            ),
            "known_large_jump_path_certificate_binding_support": True,
            "post_hoc_eigenvalue_or_matrix_modification_forbidden": True,
        },
        "binding_gates": {
            "maximum_imaginary_speed_over_c": 1.0e-10,
            "maximum_light_cone_excess_over_c": 1.0e-10,
            "generalized_eigenpair_relative_defect_max": 1.0e-8,
            "eigenvector_condition_number_max": 1.0e8,
            "scaled_temporal_condition_number_max": 1.0e8,
            "biorthogonality_and_projector_defect_max": 1.0e-8,
            "advective_cluster_dimension": 3,
            "maximum_advective_cluster_transport_offset_over_c": 1.0e-6,
            "minimum_advective_cluster_complement_gap_over_c": 1.0e-4,
            "physical_tensor_constraint_relative_defect_max": 1.0e-10,
            "derivative_ladder_relative_defect_max": 1.0e-7,
            "source_energy_ledger_relative_defect_max": 1.0e-10,
            "reference_causality_margin_min": 1.0e-8,
            "dominant_energy_margin_min": 1.0e-8,
            "entropy_production_min": 0.0,
            "all_points_and_all_gates_required": True,
            "fail_closed": True,
        },
        "corrected_full_audit": {
            "reuse_frozen_stage2_envelope_bitwise": True,
            "base_points": 8401,
            "all_off_equilibrium_witnesses_required": True,
            "representative_and_old_failed_face_derivative_ladders_required": True,
            "restart_from_first_base_point": True,
            "reuse_partial_negative_arrays_forbidden": True,
            "new_failure_stops_immediately": True,
        },
        "claim_boundary": {
            "corrected_full_local_audit_authorized": True,
            "local_architecture_certified": False,
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
        raise RuntimeError("path certificate checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = utils._read_json(parent.CANONICAL_DIRECTORY / "path_metrics.json")
    provenance = utils._read_json(parent.CANONICAL_DIRECTORY / "provenance.json")
    if (
        summary["classification"] != parent.PASS_CLASSIFICATION
        or not summary["passed"]
        or not summary["parent_negative_result_preserved"]
        or not summary["audit_correction_manifest_authorized"]
        or summary["authorized_next"] != parent.AUTHORIZED_NEXT_ON_PASS
        or metrics["minimum_cluster_complement_gap_over_c"] < 1.0e-4
        or metrics["maximum_eigenvector_condition_number"] > 1.0e8
        or metrics["maximum_imaginary_speed_over_c"] > 1.0e-10
        or max(metrics["maximum_projector_jump_refinement_ratios"]) > 0.60
    ):
        raise RuntimeError("path continuity certificate changed")
    for relative, expected in provenance["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"path certificate source changed: {relative}")
    if utils._sha256(ROOT / PHYSICAL_SOURCE) != PHYSICAL_SOURCE_SHA256:
        raise RuntimeError("physical source changed")
    if utils._sha256(ROOT / PHYSICAL_TEST) != PHYSICAL_TEST_SHA256:
        raise RuntimeError("physical test changed")
    if require_clean and utils._git(
        "status", "--short", "--untracked-files=no"
    ):
        raise RuntimeError("audit correction freeze requires clean tracked tree")
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
            "# Advective path-continuity audit correction manifest",
            "",
            f"Classification: `{CLASSIFICATION}`.",
            "",
            "The parent failures remain binding. The prospective correction changes one diagnostic only: coarse endpoint subspace overlap becomes nonbinding because it depends on finite state and geometry separation. Pointwise real diagonalizability, uniform eigenbasis conditioning, exact three-mode transport attachment, and a nonzero cluster/complement spectral gap are binding invariant replacements.",
            "",
            "The known 0.738 endpoint pair has an independently frozen 129-node path certificate with a uniformly open gap and convergent projectors. The corrected audit restarts from the first frozen base state and may not reuse partial negative arrays.",
            "",
            f"Authorized next: `{AUTHORIZED_NEXT}` only. No spatial step or trajectory is authorized.",
            "",
        )
    )


def _freeze() -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("audit correction manifest already exists")
    utils = _utils()
    parent_data = _validate_parent(require_clean=True)
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "correction_contract.json", _contract())
    utils._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_artifact": parent.ARTIFACT,
            "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256,
            "parent_hashes": parent_data["hashes"],
            "path_metrics": parent_data["metrics"],
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
        "all_parent_results_preserved": True,
        "corrected_full_local_audit_authorized": True,
        "local_architecture_certified": False,
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
