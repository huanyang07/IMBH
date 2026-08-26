#!/usr/bin/env python3
"""Freeze a pathwise diagnosis of the coarse-cell subspace rotation.

The parent retry failure remains binding.  This definitions-only package
tests whether its well-separated three-mode advective cluster has a smooth,
uniformly conditioned invariant subspace along the exact interpolating
state/geometry path.  It authorizes no repair, envelope retry, or trajectory.
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

import run_causal_inner_entropy_complete_projected_local_structural_audit_retry_wp10c9d6c7c3b5c4f25fizee3 as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fizee4_"
    "advective_subspace_path_diagnosis_manifest"
)
CLASSIFICATION = "advective_subspace_path_diagnosis_manifest_frozen"
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizee5_"
    "advective_subspace_path_continuity_certificate"
)
ARTIFACT = (
    "causal_inner_advective_subspace_path_diagnosis_manifest_"
    "wp10c9d6c7c3b5c4f25fizee4"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ADVECTIVE_SUBSPACE_PATH_"
    "DIAGNOSIS_MANIFEST_WP10C9D6C7C3B5C4F25FIZEE4_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_advective_subspace_path_diagnosis_manifest_"
    "wp10c9d6c7c3b5c4f25fizee4.py"
)
THIS_TEST = (
    "tests/test_causal_inner_advective_subspace_path_diagnosis_manifest_"
    "wp10c9d6c7c3b5c4f25fizee4.py"
)
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "2a2dde4fddb6d831f54c1d1f8ba9a88562b647b88aa05d6e0f4ae6dd1a9ad879"
)
PHYSICAL_SOURCE = parent.PHYSICAL_SOURCE
PHYSICAL_TEST = parent.PHYSICAL_TEST
PHYSICAL_SOURCE_SHA256 = parent.PHYSICAL_SOURCE_SHA256
PHYSICAL_TEST_SHA256 = parent.PHYSICAL_TEST_SHA256
LEFT_LABEL = "accepted_00_cell_044"
RIGHT_LABEL = "accepted_00_cell_045"
LEFT_RADIUS_CM = 6593349987.183909
RIGHT_RADIUS_CM = 6729339317.769789
LEFT_CHART7 = (
    4.863366549793431,
    -0.0986781519959265,
    0.8014391603898786,
    15.034538870445314,
    0.0002998298955252985,
    20.821641297092793,
    0.0,
)
RIGHT_CHART7 = (
    4.8497191077958,
    -0.11504099014619346,
    0.6583764461476951,
    14.893727530332326,
    5.571117827765792e-05,
    20.333405656468322,
    0.0,
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
        "parent_negative_result": {
            "artifact": parent.ARTIFACT,
            "classification": parent.HYPERBOLICITY_FAILURE,
            "preserved_as_binding": True,
            "retroactive_reclassification_forbidden": True,
            "failure_reason": "strong_hyperbolicity:advective_subspace_continuity",
            "coarse_endpoint_subspace_cosine": 0.7380714589652135,
            "all_pointwise_spectra_real": True,
            "coarse_endpoint_gate_remains_failed": True,
        },
        "mathematical_question": {
            "strong_hyperbolicity_is_pointwise_with_a_uniformly_bounded_smooth_eigenbasis": True,
            "coarse_endpoint_overlap_is_not_itself_a_pointwise_theorem": True,
            "advective_cluster_dimension": 3,
            "cluster_selection": "three eigenvalues nearest exact material transport speed",
            "invariant_object": "orthogonal projector onto the scaled physical right invariant subspace",
            "coordinate_scaling": (1.0, 0.1, 0.1, 1.0, 1.0e-4, 1.0, 0.03),
        },
        "frozen_path": {
            "left_label": LEFT_LABEL,
            "right_label": RIGHT_LABEL,
            "left_radius_cm": LEFT_RADIUS_CM,
            "right_radius_cm": RIGHT_RADIUS_CM,
            "left_chart7": LEFT_CHART7,
            "right_chart7": RIGHT_CHART7,
            "radius_interpolation": "affine",
            "chart_interpolation": "affine_in_seven_primitive_charts",
            "geometry_and_vertical_frequency": "recomputed_exactly_at_each_interpolated_radius",
            "nested_node_counts": (33, 65, 129),
            "all_nodes_and_all_resolutions_binding": True,
        },
        "binding_gates": {
            "maximum_imaginary_speed_over_c": 1.0e-10,
            "maximum_light_cone_excess_over_c": 1.0e-10,
            "maximum_eigenpair_relative_defect": 1.0e-8,
            "maximum_eigenvector_condition_number": 1.0e8,
            "maximum_temporal_condition_number": 1.0e8,
            "maximum_biorthogonality_or_projector_defect": 1.0e-8,
            "minimum_cluster_complement_gap_over_c": 1.0e-4,
            "minimum_adjacent_subspace_cosine": 0.99,
            "maximum_projector_jump_refinement_ratio": 0.60,
            "physical_tensor_constraint_relative_defect_max": 1.0e-10,
            "source_energy_ledger_relative_defect_max": 1.0e-10,
            "reference_causality_margin_min": 1.0e-8,
            "dominant_energy_margin_min": 1.0e-8,
            "entropy_production_min": 0.0,
            "all_gates_required": True,
            "fail_closed": True,
        },
        "interpretation": {
            "coarse_endpoint_cosine_is_recorded_but_nonbinding_in_this_path_test": True,
            "positive_path_certificate_would_diagnose_a_coarse_sampling_gate_not_a_physical_eigenbasis_loss": True,
            "positive_path_certificate_does_not_reclassify_parent": True,
            "any_cluster_gap_closure_or_unbounded_condition_is_a_structural_failure": True,
        },
        "claim_boundary": {
            "path_diagnosis_authorized": True,
            "diagnostic_gate_replacement_authorized": False,
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
        raise RuntimeError("parent retry checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = utils._read_json(parent.CANONICAL_DIRECTORY / "audit_metrics.json")
    failure = metrics["first_failure"]
    if (
        summary["classification"] != parent.HYPERBOLICITY_FAILURE
        or summary["passed"]
        or summary["authorized_next"] is not None
        or failure["label"] != RIGHT_LABEL
        or failure["reasons"]
        != ["strong_hyperbolicity:advective_subspace_continuity"]
        or metrics["minimum_advective_neighbor_subspace_pair"]
        != [LEFT_LABEL, RIGHT_LABEL]
        or metrics["minimum_advective_neighbor_subspace_cosine"]
        != 0.7380714589652135
    ):
        raise RuntimeError("parent retry boundary changed")
    if utils._sha256(ROOT / PHYSICAL_SOURCE) != PHYSICAL_SOURCE_SHA256:
        raise RuntimeError("physical source changed")
    if utils._sha256(ROOT / PHYSICAL_TEST) != PHYSICAL_TEST_SHA256:
        raise RuntimeError("physical test changed")
    if require_clean and utils._git(
        "status", "--short", "--untracked-files=no"
    ):
        raise RuntimeError("path diagnosis freeze requires clean tracked tree")
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
            "# Advective-subspace path diagnosis manifest",
            "",
            f"Classification: `{CLASSIFICATION}`.",
            "",
            f"The parent `{parent.HYPERBOLICITY_FAILURE}` classification and its coarse endpoint-cosine failure remain binding.",
            "",
            "The two endpoints have real, well-conditioned spectra and a three-mode cluster separated from the complement. This prospective diagnostic resolves the intervening state/geometry path on 33, 65, and 129 nested nodes and tests the invariant projector, cluster gap, complete pointwise principal, and refinement of projector jumps.",
            "",
            "A pass can diagnose the coarse endpoint gate as a sampling diagnostic; it cannot retroactively pass the parent audit or authorize a trajectory.",
            "",
            f"Authorized next: `{AUTHORIZED_NEXT}` only.",
            "",
        )
    )


def _freeze() -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("path diagnosis manifest already exists")
    utils = _utils()
    parent_data = _validate_parent(require_clean=True)
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "diagnosis_contract.json", _contract())
    utils._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_artifact": parent.ARTIFACT,
            "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256,
            "parent_hashes": parent_data["hashes"],
            "parent_first_failure": parent_data["metrics"]["first_failure"],
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
        "parent_negative_result_preserved": True,
        "path_diagnosis_authorized": True,
        "diagnostic_gate_replacement_authorized": False,
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
