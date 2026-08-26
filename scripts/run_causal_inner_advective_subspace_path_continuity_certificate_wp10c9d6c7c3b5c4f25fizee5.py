#!/usr/bin/env python3
"""Certify pathwise continuity of the repeated advective eigenspace.

No trajectory is advanced.  The exact state/geometry segment between the two
coarse audit endpoints is resolved on a nested 129-node path.  Complete local
gates, spectral separation, invariant-projector continuity, and refinement of
the maximum projector jump are binding.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_advective_subspace_path_diagnosis_manifest_wp10c9d6c7c3b5c4f25fizee4 as parent  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo import (  # noqa: E402
    generalized_maxwell_cattaneo_principal,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = parent.AUTHORIZED_NEXT
PASS_CLASSIFICATION = "advective_subspace_path_continuity_passed"
FAIL_CLASSIFICATION = "advective_subspace_path_continuity_failed"
AUTHORIZED_NEXT_ON_PASS = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizee6_"
    "advective_path_continuity_audit_correction_manifest"
)
ARTIFACT = (
    "causal_inner_advective_subspace_path_continuity_certificate_"
    "wp10c9d6c7c3b5c4f25fizee5"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ADVECTIVE_SUBSPACE_PATH_"
    "CONTINUITY_CERTIFICATE_WP10C9D6C7C3B5C4F25FIZEE5_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_advective_subspace_path_continuity_"
    "certificate_wp10c9d6c7c3b5c4f25fizee5.py"
)
THIS_TEST = (
    "tests/test_causal_inner_advective_subspace_path_continuity_"
    "certificate_wp10c9d6c7c3b5c4f25fizee5.py"
)
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "994a65c7aeb218ac2068c66ee2b28472e0a77ad124932d9f350baf2c1235f891"
)
PHYSICAL_SOURCE = parent.PHYSICAL_SOURCE
PHYSICAL_TEST = parent.PHYSICAL_TEST
PHYSICAL_SOURCE_SHA256 = parent.PHYSICAL_SOURCE_SHA256
PHYSICAL_TEST_SHA256 = parent.PHYSICAL_TEST_SHA256
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _utils():
    return parent._utils()


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
    if isinstance(value, (np.complexfloating, complex)):
        number = complex(value)
        return {"real": float(number.real), "imaginary": float(number.imag)}
    return value


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def _validate_parent(*, require_clean: bool) -> dict:
    utils = _utils()
    if utils._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != (
        PARENT_CHECKSUM_MANIFEST_SHA256
    ):
        raise RuntimeError("path diagnosis manifest checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    contract = utils._read_json(
        parent.CANONICAL_DIRECTORY / "diagnosis_contract.json"
    )
    provenance = utils._read_json(
        parent.CANONICAL_DIRECTORY / "provenance.json"
    )
    if (
        summary["classification"] != parent.CLASSIFICATION
        or not summary["passed"]
        or not summary["definitions_only"]
        or not summary["parent_negative_result_preserved"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["authorized_next"] != WORK_PACKAGE
        or not contract["binding_gates"]["fail_closed"]
        or contract["claim_boundary"]["trajectory_authorized"]
    ):
        raise RuntimeError("path diagnosis authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _utils()._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"path diagnosis source changed: {relative}")
    if _utils()._sha256(ROOT / PHYSICAL_SOURCE) != PHYSICAL_SOURCE_SHA256:
        raise RuntimeError("physical source changed")
    if _utils()._sha256(ROOT / PHYSICAL_TEST) != PHYSICAL_TEST_SHA256:
        raise RuntimeError("physical test changed")
    if require_clean and utils._git(
        "status", "--short", "--untracked-files=no"
    ):
        raise RuntimeError("path certificate requires clean tracked tree")
    return {"hashes": hashes, "summary": summary, "contract": contract}


def _scaled_advective_subspace(principal) -> tuple[np.ndarray, np.ndarray, float]:
    transport = float(principal.local_state.transport_velocity_over_c)
    eigenvalues = np.asarray(principal.eigenvalues_over_c)
    if np.max(np.abs(np.imag(eigenvalues))) > 1.0e-10:
        raise RuntimeError("complex spectrum cannot define real path subspace")
    real_values = np.real(eigenvalues)
    selected = np.argsort(np.abs(real_values - transport))[:3]
    complement = np.asarray(
        [index for index in range(7) if index not in set(selected)], dtype=int
    )
    cluster_gap = float(
        np.min(
            np.abs(
                real_values[selected, None] - real_values[complement][None, :]
            )
        )
    )
    physical_vectors = (
        principal.primitive_column_scales[:, None]
        * principal.right_eigenvectors_scaled[:, selected]
    )
    common_scales = np.asarray(
        [1.0, 0.1, 0.1, 1.0, 1.0e-4, 1.0, 0.03], dtype=float
    )
    dimensionless = np.real_if_close(
        physical_vectors / common_scales[:, None], tol=1000
    )
    basis, _ = np.linalg.qr(np.asarray(dimensionless, dtype=float))
    basis = basis[:, :3]
    projector = basis @ basis.T
    return basis, projector, cluster_gap


def _resolution_metrics(bases: np.ndarray, projectors: np.ndarray) -> dict:
    cosines = []
    jumps = []
    for left, right, projector_left, projector_right in zip(
        bases[:-1], bases[1:], projectors[:-1], projectors[1:], strict=True
    ):
        singular = np.linalg.svd(left.T @ right, compute_uv=False)
        cosines.append(float(np.min(np.clip(singular, 0.0, 1.0))))
        jumps.append(float(np.linalg.norm(projector_right - projector_left, ord=2)))
    return {
        "minimum_adjacent_subspace_cosine": min(cosines),
        "maximum_adjacent_projector_jump": max(jumps),
        "sum_adjacent_projector_jumps": sum(jumps),
    }


def _execute() -> tuple[dict, dict[str, np.ndarray]]:
    validated = _validate_parent(require_clean=False)
    contract = validated["contract"]
    frozen_audit = parent.parent.frozen_audit
    source = (
        frozen_audit.parent.parent.parent.parent.boundary_diagnostic.manifest.parent.engine.execution.source
    )
    start = time.perf_counter()
    context = source._initial_inputs()["base"]["configuration"]["context"]
    context_seconds = time.perf_counter() - start
    left_radius = float(parent.LEFT_RADIUS_CM)
    right_radius = float(parent.RIGHT_RADIUS_CM)
    left_chart = np.asarray(parent.LEFT_CHART7, dtype=float)
    right_chart = np.asarray(parent.RIGHT_CHART7, dtype=float)
    radial = frozen_audit.parent.parent.parent.parent.boundary_diagnostic.radial
    fine_count = max(contract["frozen_path"]["nested_node_counts"])
    parameters = np.linspace(0.0, 1.0, fine_count)
    radii = left_radius + parameters * (right_radius - left_radius)
    charts = left_chart[None, :] + parameters[:, None] * (
        right_chart - left_chart
    )[None, :]
    eigenvalues = np.empty((fine_count, 7), dtype=complex)
    bases = np.empty((fine_count, 7, 3), dtype=float)
    projectors = np.empty((fine_count, 7, 7), dtype=float)
    cluster_gaps = np.empty(fine_count, dtype=float)
    conditions = np.empty(fine_count, dtype=float)
    point_reasons: list[tuple[str, ...]] = []
    point_metrics: list[dict] = []
    for index, (radius, chart) in enumerate(zip(radii, charts, strict=True)):
        geometry = radial._cell_state(context, float(radius), chart[:5]).geometry
        principal = generalized_maxwell_cattaneo_principal(
            geometry,
            chart,
            proper_vertical_frequency=float(
                context.vertical_frequency.frequency(float(radius))
            ),
            alpha=float(context.alpha),
            stress_factor=float(context.stress_factor),
        )
        local_metrics, reasons = frozen_audit._point_metrics(
            principal, alpha=float(context.alpha)
        )
        basis, projector, gap = _scaled_advective_subspace(principal)
        if gap < 1.0e-4:
            reasons = tuple(reasons) + ("strong_hyperbolicity:cluster_gap",)
        eigenvalues[index] = principal.eigenvalues_over_c
        bases[index] = basis
        projectors[index] = projector
        cluster_gaps[index] = gap
        conditions[index] = principal.eigenvector_condition_number
        point_reasons.append(tuple(reasons))
        point_metrics.append(local_metrics)

    resolution_metrics = {}
    maximum_jumps = []
    for count in contract["frozen_path"]["nested_node_counts"]:
        stride = (fine_count - 1) // (int(count) - 1)
        selected = np.arange(0, fine_count, stride, dtype=int)
        result = _resolution_metrics(bases[selected], projectors[selected])
        resolution_metrics[str(count)] = result
        maximum_jumps.append(result["maximum_adjacent_projector_jump"])
    refinement_ratios = [
        maximum_jumps[index + 1] / maximum_jumps[index]
        for index in range(len(maximum_jumps) - 1)
    ]
    all_reasons = tuple(reason for reasons in point_reasons for reason in reasons)
    for count, result in resolution_metrics.items():
        if result["minimum_adjacent_subspace_cosine"] < 0.99:
            all_reasons += (f"strong_hyperbolicity:path_cosine_{count}",)
    if max(refinement_ratios) > 0.60:
        all_reasons += ("strong_hyperbolicity:projector_refinement",)
    passed = not all_reasons
    classification = PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION
    endpoint_singular = np.linalg.svd(bases[0].T @ bases[-1], compute_uv=False)
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "parent_negative_result_preserved": True,
        "node_count": fine_count,
        "context_construction_wall_seconds": context_seconds,
        "execution_wall_seconds": time.perf_counter() - start,
        "minimum_cluster_complement_gap_over_c": float(np.min(cluster_gaps)),
        "maximum_eigenvector_condition_number": float(np.max(conditions)),
        "maximum_imaginary_speed_over_c": float(
            np.max(np.abs(np.imag(eigenvalues)))
        ),
        "coarse_endpoint_subspace_cosine": float(np.min(endpoint_singular)),
        "resolution_metrics": resolution_metrics,
        "maximum_projector_jump_refinement_ratios": refinement_ratios,
        "point_failure_reasons": point_reasons,
        "all_failure_reasons": all_reasons,
        "point_metrics": point_metrics,
        "new_trajectory_steps": 0,
        "audit_correction_manifest_authorized": passed,
        "authorized_next": AUTHORIZED_NEXT_ON_PASS if passed else None,
    }
    arrays = {
        "path_parameters": parameters,
        "radii_cm": radii,
        "charts7": charts,
        "eigenvalues_over_c": eigenvalues,
        "advective_bases": bases,
        "advective_projectors": projectors,
        "cluster_complement_gaps_over_c": cluster_gaps,
        "eigenvector_condition_numbers": conditions,
    }
    return _plain(metrics), arrays


def _report(metrics: dict) -> str:
    decision = (
        f"Authorized next: `{AUTHORIZED_NEXT_ON_PASS}` only."
        if metrics["passed"]
        else "No later package is authorized."
    )
    return "\n".join(
        (
            "# Advective-subspace path continuity certificate",
            "",
            f"Classification: `{metrics['classification']}`.",
            "",
            f"The 129-node path retained minimum cluster/complement gap `{metrics['minimum_cluster_complement_gap_over_c']}`, maximum eigenvector condition `{metrics['maximum_eigenvector_condition_number']}`, and maximum imaginary speed `{metrics['maximum_imaginary_speed_over_c']}`.",
            "",
            f"Nested path metrics: `{metrics['resolution_metrics']}`. Maximum projector-jump refinement ratios: `{metrics['maximum_projector_jump_refinement_ratios']}`. The original coarse endpoint cosine `{metrics['coarse_endpoint_subspace_cosine']}` remains recorded but is not reinterpreted as a pointwise gate.",
            "",
            f"The parent `{parent.parent.HYPERBOLICITY_FAILURE}` result remains binding and is not retroactively reclassified.",
            "",
            decision,
            "No audit gate is changed here, and no spatial step or trajectory is authorized.",
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
        raise RuntimeError("path continuity certificate already exists")
    utils = _utils()
    parent_data = _validate_parent(require_clean=True)
    CANONICAL_DIRECTORY.mkdir(parents=True)
    _save_npz(CANONICAL_DIRECTORY / "path_arrays.npz", arrays)
    utils._write_json(CANONICAL_DIRECTORY / "path_metrics.json", metrics)
    utils._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_artifact": parent.ARTIFACT,
            "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256,
            "parent_hashes": parent_data["hashes"],
            "physical_source_sha256": PHYSICAL_SOURCE_SHA256,
            "physical_test_sha256": PHYSICAL_TEST_SHA256,
            "left_radius_cm": parent.LEFT_RADIUS_CM,
            "right_radius_cm": parent.RIGHT_RADIUS_CM,
            "left_chart7": parent.LEFT_CHART7,
            "right_chart7": parent.RIGHT_CHART7,
        },
    )
    passed = bool(metrics["passed"])
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": passed,
        "parent_negative_result_preserved": True,
        "path_continuity_certificate_completed": True,
        "smooth_uniformly_bounded_advective_subspace_on_saved_path": passed,
        "audit_correction_manifest_authorized": passed,
        "new_trajectory_steps": 0,
        "full_envelope_retry_authorized": False,
        "spatial_discretization_authorized": False,
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
    source_paths = (THIS_RUNNER, THIS_TEST, PHYSICAL_SOURCE, PHYSICAL_TEST, REPORT_RELATIVE)
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
