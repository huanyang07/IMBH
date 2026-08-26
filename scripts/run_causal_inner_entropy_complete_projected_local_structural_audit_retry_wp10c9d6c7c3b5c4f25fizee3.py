#!/usr/bin/env python3
"""Retry the complete projected seven-field local structural audit.

The original fail-fast audit kernel and all of its gates are hash-locked and
reused unchanged.  The only scientific implementation change is the already
certified analytic differentiation of the three exact material currents.
The original negative result remains preserved as historical evidence.
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

import run_causal_inner_saved_advective_degeneracy_repair_certificate_wp10c9d6c7c3b5c4f25fizee2 as parent  # noqa: E402
import run_causal_inner_entropy_complete_projected_local_structural_audit_wp10c9d6c7c3b5c4f25fizee as frozen_audit  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = parent.AUTHORIZED_NEXT_ON_PASS
PASS_CLASSIFICATION = (
    "entropy_complete_projected_local_structural_audit_retry_passed"
)
CAUSALITY_FAILURE = (
    "entropy_complete_projected_local_structural_audit_retry_causality_failed"
)
HYPERBOLICITY_FAILURE = (
    "entropy_complete_projected_local_structural_audit_retry_hyperbolicity_failed"
)
LEDGER_FAILURE = (
    "entropy_complete_projected_local_structural_audit_retry_ledger_failed"
)
DERIVATION_FAILURE = (
    "entropy_complete_projected_local_structural_audit_retry_derivation_failed"
)
AUTHORIZED_NEXT_ON_PASS = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizef_"
    "entropy_complete_path_conservative_spatial_manifest"
)
ARTIFACT = (
    "causal_inner_entropy_complete_projected_local_structural_audit_retry_"
    "wp10c9d6c7c3b5c4f25fizee3"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_PROJECTED_"
    "LOCAL_STRUCTURAL_AUDIT_RETRY_WP10C9D6C7C3B5C4F25FIZEE3_"
    "2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_entropy_complete_projected_local_structural_"
    "audit_retry_wp10c9d6c7c3b5c4f25fizee3.py"
)
THIS_TEST = (
    "tests/test_causal_inner_entropy_complete_projected_local_structural_"
    "audit_retry_wp10c9d6c7c3b5c4f25fizee3.py"
)
FROZEN_AUDIT_RUNNER = frozen_audit.THIS_RUNNER
FROZEN_AUDIT_RUNNER_SHA256 = (
    "4143bc70e73a673926b192c5f727692c47d9f765e10de91449a436aa3e5354b8"
)
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "c28141bbe883542cd7a0fb5d1d4baeb4069bc86f09d25acd59b40b250ee07849"
)
PHYSICAL_SOURCE = parent.PHYSICAL_SOURCE
PHYSICAL_TEST = parent.PHYSICAL_TEST
PHYSICAL_SOURCE_SHA256 = parent.PHYSICAL_SOURCE_SHA256
PHYSICAL_TEST_SHA256 = parent.PHYSICAL_TEST_SHA256
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _utils():
    return parent._utils()


def _validate_parent(*, require_clean: bool) -> dict:
    utils = _utils()
    if utils._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != (
        PARENT_CHECKSUM_MANIFEST_SHA256
    ):
        raise RuntimeError("saved-point certificate checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = utils._read_json(
        parent.CANONICAL_DIRECTORY / "certificate_metrics.json"
    )
    provenance = utils._read_json(
        parent.CANONICAL_DIRECTORY / "provenance.json"
    )
    if (
        summary["classification"] != parent.PASS_CLASSIFICATION
        or not summary["passed"]
        or not summary["parent_negative_result_preserved"]
        or not summary["full_envelope_retry_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or metrics["maximum_imaginary_speed_over_c"] > 1.0e-10
        or max(metrics["material_product_identity_relative_defects"]) > 1.0e-12
        or max(metrics["matrix_derivative_ladder"].values()) > 1.0e-7
    ):
        raise RuntimeError("saved-point repair certificate changed")
    for relative, expected in provenance["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"saved-point certificate source changed: {relative}")
    if utils._sha256(ROOT / FROZEN_AUDIT_RUNNER) != FROZEN_AUDIT_RUNNER_SHA256:
        raise RuntimeError("original audit kernel changed")
    if utils._sha256(ROOT / PHYSICAL_SOURCE) != PHYSICAL_SOURCE_SHA256:
        raise RuntimeError("certified physical repair changed")
    if utils._sha256(ROOT / PHYSICAL_TEST) != PHYSICAL_TEST_SHA256:
        raise RuntimeError("certified physical tests changed")
    stage2 = frozen_audit.parent.parent.parent.parent
    envelope_hashes = utils._validate_checksums(stage2.CANONICAL_DIRECTORY)
    if require_clean and utils._git(
        "status", "--short", "--untracked-files=no"
    ):
        raise RuntimeError("full local retry requires a clean tracked tree")
    return {
        "hashes": hashes,
        "summary": summary,
        "metrics": metrics,
        "stage2": stage2,
        "envelope_hashes": envelope_hashes,
    }


def _retry_classification(original: str) -> str:
    mapping = {
        frozen_audit.PASS_CLASSIFICATION: PASS_CLASSIFICATION,
        frozen_audit.CAUSALITY_FAILURE: CAUSALITY_FAILURE,
        frozen_audit.HYPERBOLICITY_FAILURE: HYPERBOLICITY_FAILURE,
        frozen_audit.LEDGER_FAILURE: LEDGER_FAILURE,
        frozen_audit.DERIVATION_FAILURE: DERIVATION_FAILURE,
    }
    try:
        return mapping[original]
    except KeyError as error:
        raise RuntimeError(f"unknown frozen audit classification: {original}") from error


def _execute() -> tuple[dict, dict[str, np.ndarray]]:
    """Run the hash-locked audit kernel against the repaired principal.

    The frozen kernel's validator is replaced only for the duration of this
    call because its prospective parent understandably hash-locks the old
    pre-repair physical source.  The retry performs the stronger validation
    above, supplies the identical frozen Stage-2 envelope, and restores the
    original validator immediately afterward.  No audit loop, point gate,
    witness construction, or derivative-ladder calculation is changed.
    """

    validated = _validate_parent(require_clean=False)
    original_validator = frozen_audit._validate_parent

    def validated_retry_inputs(*, require_clean: bool) -> dict:
        if require_clean and _utils()._git(
            "status", "--short", "--untracked-files=no"
        ):
            raise RuntimeError("full local retry requires a clean tracked tree")
        return {"stage2": validated["stage2"]}

    frozen_audit._validate_parent = validated_retry_inputs
    try:
        metrics, arrays = frozen_audit._audit()
    finally:
        frozen_audit._validate_parent = original_validator

    original_classification = str(metrics["classification"])
    classification = _retry_classification(original_classification)
    passed = classification == PASS_CLASSIFICATION
    metrics.update(
        {
            "work_package": WORK_PACKAGE,
            "classification": classification,
            "passed": passed,
            "frozen_audit_kernel_classification": original_classification,
            "frozen_audit_kernel_sha256": FROZEN_AUDIT_RUNNER_SHA256,
            "parent_negative_result_preserved": True,
            "saved_point_repair_certificate_preserved": True,
            "full_envelope_retry_completed": True,
            "authorized_next": AUTHORIZED_NEXT_ON_PASS if passed else None,
        }
    )
    return frozen_audit._plain(metrics), arrays


def _report(metrics: dict) -> str:
    failure = metrics["first_failure"]
    decision = (
        f"Authorized next: `{AUTHORIZED_NEXT_ON_PASS}` only."
        if metrics["passed"]
        else "No later package is authorized; the first retry failure must be diagnosed prospectively."
    )
    return "\n".join(
        (
            "# Entropy-complete projected local structural audit retry",
            "",
            f"Classification: `{metrics['classification']}`.",
            "",
            f"The hash-locked original audit kernel evaluated {metrics['base_points_audited']} of {metrics['base_points_planned']} base points and {metrics['witness_points_audited']} unique off-equilibrium witnesses against the repaired physical principal. The original `{parent.parent.PARENT_CLASSIFICATION}` result remains preserved and is not reclassified.",
            "",
            f"Worst imaginary speed: `{metrics['extremes'].get('maximum_imaginary_speed_over_c', {}).get('value')}`. Worst light-cone excess: `{metrics['extremes'].get('maximum_light_cone_excess_over_c', {}).get('value')}`. Worst eigenvector condition: `{metrics['extremes'].get('eigenvector_condition_number', {}).get('value')}`. Minimum neighboring advective-subspace cosine: `{metrics['minimum_advective_neighbor_subspace_cosine']}`.",
            "",
            f"First failure: `{failure}`. Derivative ladders: `{metrics['derivative_ladders']}`.",
            "",
            decision,
            "No spatial step, seven-field trajectory, fixed-Q invariant object, slow-flux atlas, reduced cycle, or complete-cycle execution is authorized by this local audit.",
            "",
        )
    )


def _update_catalog(summary: dict, status: str) -> None:
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
                    "scientific_status": status,
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


def _canonicalize(metrics: dict, arrays: dict[str, np.ndarray]) -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("full local retry result already exists")
    utils = _utils()
    parent_data = _validate_parent(require_clean=True)
    CANONICAL_DIRECTORY.mkdir(parents=True)
    frozen_audit._save_npz(CANONICAL_DIRECTORY / "audit_arrays.npz", arrays)
    utils._write_json(CANONICAL_DIRECTORY / "audit_metrics.json", metrics)
    utils._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_artifact": parent.ARTIFACT,
            "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256,
            "parent_hashes": parent_data["hashes"],
            "frozen_envelope_artifact": parent_data["stage2"].ARTIFACT,
            "frozen_envelope_hashes": parent_data["envelope_hashes"],
            "frozen_audit_runner": FROZEN_AUDIT_RUNNER,
            "frozen_audit_runner_sha256": FROZEN_AUDIT_RUNNER_SHA256,
            "physical_source_sha256": PHYSICAL_SOURCE_SHA256,
            "physical_test_sha256": PHYSICAL_TEST_SHA256,
        },
    )
    passed = bool(metrics["passed"])
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": passed,
        "parent_negative_result_preserved": True,
        "saved_point_repair_certificate_preserved": True,
        "full_envelope_retry_completed": True,
        "base_points_audited": metrics["base_points_audited"],
        "witness_points_audited": metrics["witness_points_audited"],
        "complete_reduced_principal_certified": passed,
        "physical_tensor_constraints_certified": passed,
        "source_energy_entropy_ledger_certified": passed,
        "new_trajectory_steps": 0,
        "spatial_manifest_authorized": passed,
        "seven_field_trajectory_authorized": False,
        "fixed_Q_invariant_object_authorized": False,
        "slow_flux_atlas_authorized": False,
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT_ON_PASS if passed else None,
    }
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_report(metrics), encoding="utf-8")
    source_paths = (
        THIS_RUNNER,
        THIS_TEST,
        FROZEN_AUDIT_RUNNER,
        PHYSICAL_SOURCE,
        PHYSICAL_TEST,
        REPORT_RELATIVE,
    )
    utils._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "PASS" if passed else "FAIL",
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
    _update_catalog(summary, "PASS" if passed else "FAIL")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    if not arguments.execute:
        parser.error("choose --execute")
    metrics, arrays = _execute()
    summary = _canonicalize(metrics, arrays)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
