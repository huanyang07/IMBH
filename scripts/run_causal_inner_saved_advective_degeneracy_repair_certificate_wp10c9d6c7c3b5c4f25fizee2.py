#!/usr/bin/env python3
"""Certify the analytic material-current repair at the saved failed point.

This fail-closed execution evaluates exactly one previously rejected local
state at three prospectively frozen derivative step factors.  It advances no
trajectory and does not reclassify the parent full-envelope failure.
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

import run_causal_inner_analytic_material_current_differentiation_repair_manifest_wp10c9d6c7c3b5c4f25fizee1 as parent  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo import (  # noqa: E402
    _sixth_order_centered_jacobian,
    default_primitive_steps,
    generalized_maxwell_cattaneo_local_state,
    generalized_maxwell_cattaneo_principal,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = parent.AUTHORIZED_NEXT
PASS_CLASSIFICATION = "saved_advective_degeneracy_repair_passed"
FAIL_CLASSIFICATION = "saved_advective_degeneracy_repair_failed"
AUTHORIZED_NEXT_ON_PASS = (
    "WP10c9d6c7c3b5c4f25fizee3_"
    "entropy_complete_projected_local_structural_audit_retry"
)
ARTIFACT = (
    "causal_inner_saved_advective_degeneracy_repair_certificate_"
    "wp10c9d6c7c3b5c4f25fizee2"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_SAVED_ADVECTIVE_DEGENERACY_"
    "REPAIR_CERTIFICATE_WP10C9D6C7C3B5C4F25FIZEE2_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_saved_advective_degeneracy_repair_"
    "certificate_wp10c9d6c7c3b5c4f25fizee2.py"
)
THIS_TEST = (
    "tests/test_causal_inner_saved_advective_degeneracy_repair_"
    "certificate_wp10c9d6c7c3b5c4f25fizee2.py"
)
PHYSICAL_SOURCE = parent.PHYSICAL_SOURCE
PHYSICAL_TEST = parent.PHYSICAL_TEST
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "f1025454c0ab619ee943780264230df1b9d6d41dc25e7c97702b56302009f9a3"
)
PHYSICAL_IMPLEMENTATION_COMMIT = "570c36c534d88de5065a7121be1b7321caa1740d"
PHYSICAL_SOURCE_SHA256 = (
    "0b19c790ab32eee5c977b753ae82dfa25c788d312942ba55a4426ad4bb9dafe3"
)
PHYSICAL_TEST_SHA256 = (
    "7121d4b52f1c1076fae4a98f1cfb1ecd8d4d8bea7b0ede34eb4a56cabff61291"
)
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
        return {
            "real": float(np.real(number)),
            "imaginary": float(np.imag(number)),
        }
    return value


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def _validate_parent(*, require_clean: bool) -> dict:
    utils = _utils()
    if utils._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != (
        PARENT_CHECKSUM_MANIFEST_SHA256
    ):
        raise RuntimeError("repair manifest checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    contract = utils._read_json(
        parent.CANONICAL_DIRECTORY / "repair_contract.json"
    )
    if (
        summary["classification"] != parent.CLASSIFICATION
        or not summary["passed"]
        or not summary["definitions_only"]
        or not summary["parent_negative_result_preserved"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["authorized_next"] != WORK_PACKAGE
        or not contract["binding_gates"]["fail_closed"]
        or contract["saved_point_certificate"]["trajectory_steps"] != 0
    ):
        raise RuntimeError("repair manifest authorization changed")
    if utils._git("rev-parse", PHYSICAL_IMPLEMENTATION_COMMIT) != (
        PHYSICAL_IMPLEMENTATION_COMMIT
    ):
        raise RuntimeError("repair implementation commit unavailable")
    if utils._sha256(ROOT / PHYSICAL_SOURCE) != PHYSICAL_SOURCE_SHA256:
        raise RuntimeError("repaired physical source changed")
    if utils._sha256(ROOT / PHYSICAL_TEST) != PHYSICAL_TEST_SHA256:
        raise RuntimeError("repaired physical test changed")
    if require_clean and utils._git(
        "status", "--short", "--untracked-files=no"
    ):
        raise RuntimeError("saved-point certificate requires clean tracked tree")
    return {"hashes": hashes, "summary": summary, "contract": contract}


def _material_product_defect(
    geometry,
    chart7: np.ndarray,
    *,
    proper_vertical_frequency: float,
    alpha: float,
    stress_factor: float,
    factor: float,
    principal,
) -> float:
    def evaluate(candidate):
        return generalized_maxwell_cattaneo_local_state(
            geometry,
            candidate,
            proper_vertical_frequency=proper_vertical_frequency,
            alpha=alpha,
            stress_factor=stress_factor,
        )

    base = evaluate(chart7)
    steps = factor * default_primitive_steps(
        chart7,
        equilibrium_specific_stress=base.equilibrium_specific_stress,
    )
    _, state_jacobian = _sixth_order_centered_jacobian(
        lambda candidate: evaluate(candidate).conservative_state6,
        chart7,
        steps,
    )
    _, transport_jacobian = _sixth_order_centered_jacobian(
        lambda candidate: np.atleast_1d(
            geometry.base.lapse
            * float(candidate[1])
            / np.sqrt(geometry.base.gamma_rr)
            - geometry.base.radial_shift_over_c
        ),
        chart7,
        steps,
    )
    expected = np.asarray(
        [
            base.transport_velocity_over_c * state_jacobian[index]
            + base.conservative_state6[index] * transport_jacobian.ravel()
            for index in (0, 4, 5)
        ]
    )
    actual = np.asarray(principal.radial_matrix[[0, 5, 6]], dtype=float)
    scale = max(float(np.linalg.norm(expected, ord=np.inf)), 1.0)
    return float(np.linalg.norm(actual - expected, ord=np.inf) / scale)


def _execute() -> tuple[dict, dict[str, np.ndarray]]:
    validated = _validate_parent(require_clean=False)
    contract = validated["contract"]
    audit = parent.parent
    source = (
        audit.parent.parent.parent.boundary_diagnostic.manifest.parent.engine.execution.source
    )
    start = time.perf_counter()
    inputs = source._initial_inputs()
    context = inputs["base"]["configuration"]["context"]
    context_seconds = time.perf_counter() - start
    chart7 = np.asarray(parent.SAVED_CHART7, dtype=float)
    radius = float(parent.SAVED_RADIUS_CM)
    old_state, rebuilt_chart7 = audit._chart7_at_equilibrium(
        context, radius, chart7[:5]
    )
    if not np.array_equal(rebuilt_chart7, chart7):
        raise RuntimeError("saved chart cannot be reconstructed bitwise")
    omega = float(context.vertical_frequency.frequency(radius))
    factors = np.asarray(
        contract["saved_point_certificate"]["derivative_step_factors"],
        dtype=float,
    )
    pencils = []
    point_metrics = []
    reasons_by_factor = []
    product_defects = []
    for factor in factors:
        principal = generalized_maxwell_cattaneo_principal(
            old_state.geometry,
            chart7,
            proper_vertical_frequency=omega,
            alpha=float(context.alpha),
            stress_factor=float(context.stress_factor),
            derivative_step_factor=float(factor),
        )
        metrics, reasons = audit._point_metrics(
            principal, alpha=float(context.alpha)
        )
        product_defect = _material_product_defect(
            old_state.geometry,
            chart7,
            proper_vertical_frequency=omega,
            alpha=float(context.alpha),
            stress_factor=float(context.stress_factor),
            factor=float(factor),
            principal=principal,
        )
        if product_defect > 1.0e-12:
            reasons = tuple(reasons) + ("derivation:material_product_identity",)
        pencils.append(principal)
        point_metrics.append(metrics)
        reasons_by_factor.append(tuple(reasons))
        product_defects.append(product_defect)

    temporal = np.asarray([item.temporal_matrix for item in pencils])
    radial = np.asarray([item.radial_matrix for item in pencils])
    eigenvalues = np.asarray([item.eigenvalues_over_c for item in pencils])
    ladder_defects = {}
    for name, matrices in (("temporal", temporal), ("radial", radial)):
        scale = max(float(np.linalg.norm(matrices[-1], ord=np.inf)), 1.0)
        ladder_defects[f"{name}_coarse_middle_relative_defect"] = float(
            np.linalg.norm(matrices[0] - matrices[1], ord=np.inf) / scale
        )
        ladder_defects[f"{name}_middle_fine_relative_defect"] = float(
            np.linalg.norm(matrices[1] - matrices[2], ord=np.inf) / scale
        )
    ladder_passed = max(ladder_defects.values()) <= 1.0e-7
    all_reasons = tuple(
        reason for reasons in reasons_by_factor for reason in reasons
    )
    if not ladder_passed:
        all_reasons += ("derivation:matrix_derivative_ladder",)
    passed = not all_reasons
    classification = PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "parent_negative_classification_preserved": parent.PARENT_CLASSIFICATION,
        "saved_label": parent.SAVED_LABEL,
        "radius_cm": radius,
        "chart7": chart7,
        "derivative_step_factors": factors,
        "point_metrics": point_metrics,
        "reasons_by_factor": reasons_by_factor,
        "material_product_identity_relative_defects": product_defects,
        "matrix_derivative_ladder": ladder_defects,
        "maximum_imaginary_speed_over_c": float(
            np.max(np.abs(np.imag(eigenvalues)))
        ),
        "maximum_eigenvector_condition_number": float(
            max(item.eigenvector_condition_number for item in pencils)
        ),
        "maximum_light_cone_excess_over_c": float(
            max(item.maximum_light_cone_excess_over_c for item in pencils)
        ),
        "context_construction_wall_seconds": context_seconds,
        "execution_wall_seconds": time.perf_counter() - start,
        "new_trajectory_steps": 0,
        "full_envelope_retry_authorized": passed,
        "authorized_next": AUTHORIZED_NEXT_ON_PASS if passed else None,
    }
    arrays = {
        "chart7": chart7,
        "derivative_step_factors": factors,
        "temporal_matrices": temporal,
        "radial_matrices": radial,
        "eigenvalues_over_c": eigenvalues,
        "material_product_identity_relative_defects": np.asarray(
            product_defects, dtype=float
        ),
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
            "# Saved advective-degeneracy repair certificate",
            "",
            f"Classification: `{metrics['classification']}`.",
            "",
            f"The saved parent failure `{parent.PARENT_CLASSIFICATION}` remains binding and is not retroactively reclassified.",
            "",
            f"At derivative factors 2, 1, and 0.5, the largest imaginary characteristic speed was `{metrics['maximum_imaginary_speed_over_c']}`, the largest eigenvector condition number was `{metrics['maximum_eigenvector_condition_number']}`, and the largest light-cone excess was `{metrics['maximum_light_cone_excess_over_c']}`.",
            "",
            f"Exact material-product identity defects: `{metrics['material_product_identity_relative_defects']}`. Matrix derivative ladder: `{metrics['matrix_derivative_ladder']}`.",
            "",
            decision,
            "No spatial step, trajectory state, fixed-Q object, slow atlas, reduced cycle, or complete-cycle execution is authorized by this saved-point result.",
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
        raise RuntimeError("saved-point certificate already exists")
    utils = _utils()
    parent_data = _validate_parent(require_clean=True)
    CANONICAL_DIRECTORY.mkdir(parents=True)
    _save_npz(CANONICAL_DIRECTORY / "certificate_arrays.npz", arrays)
    utils._write_json(CANONICAL_DIRECTORY / "certificate_metrics.json", metrics)
    utils._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_artifact": parent.ARTIFACT,
            "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256,
            "parent_hashes": parent_data["hashes"],
            "physical_implementation_commit": PHYSICAL_IMPLEMENTATION_COMMIT,
            "physical_source_sha256": PHYSICAL_SOURCE_SHA256,
            "physical_test_sha256": PHYSICAL_TEST_SHA256,
            "saved_label": parent.SAVED_LABEL,
            "saved_radius_cm": parent.SAVED_RADIUS_CM,
            "saved_chart7": parent.SAVED_CHART7,
        },
    )
    passed = bool(metrics["passed"])
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": passed,
        "parent_negative_result_preserved": True,
        "saved_point_certificate_completed": True,
        "analytic_material_current_repair_certified": passed,
        "full_envelope_retry_authorized": passed,
        "new_trajectory_steps": 0,
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
