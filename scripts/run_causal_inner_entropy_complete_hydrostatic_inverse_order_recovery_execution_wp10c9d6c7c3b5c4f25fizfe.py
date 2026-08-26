#!/usr/bin/env python3
"""Execute the frozen order-aware hydrostatic inverse recovery."""

from __future__ import annotations

import argparse
import csv
import json
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

import run_causal_inner_entropy_complete_hydrostatic_inverse_order_recovery_manifest_wp10c9d6c7c3b5c4f25fizfd as parent  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_quasisteady import (  # noqa: E402
    hydrostatic_invariant_local_scaled_jacobian,
    reconstruct_hydrostatic_fixed_invariants,
)


previous = parent.parent
original = previous.original
SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fizfe_"
    "entropy_complete_hydrostatic_inverse_order_recovery_execution"
)
PASS_CLASSIFICATION = "entropy_complete_hydrostatic_implicit_inverse_tangent_certified"
FAIL_CLASSIFICATION = "entropy_complete_hydrostatic_implicit_inverse_tangent_rejected"
AUTHORIZED_NEXT_ON_PASS = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizff_"
    "entropy_complete_local_slow_flux_atlas_manifest"
)
ARTIFACT = (
    "causal_inner_entropy_complete_hydrostatic_inverse_order_recovery_"
    "execution_wp10c9d6c7c3b5c4f25fizfe"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_HYDROSTATIC_"
    "INVERSE_ORDER_RECOVERY_EXECUTION_WP10C9D6C7C3B5C4F25FIZFE_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_entropy_complete_hydrostatic_inverse_order_"
    "recovery_execution_wp10c9d6c7c3b5c4f25fizfe.py"
)
THIS_TEST = (
    "tests/test_causal_inner_entropy_complete_hydrostatic_inverse_order_"
    "recovery_execution_wp10c9d6c7c3b5c4f25fizfe.py"
)
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "b6ae90e32e7f81f3989a6943c3c6c8bbf1273c76e8f842d75db79723c58cbb35"
)
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
UNKNOWN_INDICES = np.asarray((0, 2, 3), dtype=int)
UNKNOWN_SCALES = np.asarray((1.0, 0.1, 1.0), dtype=float)


def _utils():
    return parent._utils()


def _validate_parent(*, require_clean: bool) -> dict:
    utils = _utils()
    if (
        utils._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt")
        != PARENT_CHECKSUM_MANIFEST_SHA256
    ):
        raise RuntimeError("order-recovery manifest checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    contract = utils._read_json(
        parent.CANONICAL_DIRECTORY / "order_recovery_contract.json"
    )
    if (
        summary["classification"] != parent.CLASSIFICATION
        or not summary["passed"]
        or not summary["definitions_only"]
        or not summary["fixed_step_inverse_JVP_rejection_preserved"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["authorized_next"] != WORK_PACKAGE
        or contract["order_aware_inverse_audit"]["new_seven_field_operator_calls"] != 0
        or contract["claim_boundary"]["complete_cycle_execution_authorized"]
    ):
        raise RuntimeError("order-recovery authorization changed")
    for relative, expected in utils._read_json(
        parent.CANONICAL_DIRECTORY / "provenance.json"
    )["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"order-recovery source changed: {relative}")
    rejected = parent._validate_parent(require_clean=False)
    if rejected["summary"]["classification"] != previous.FAIL_CLASSIFICATION:
        raise RuntimeError("fixed-step inverse rejection was not preserved")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("order-recovery execution requires a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "contract": contract}


def _relative_defect(value: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.max(np.abs(value - reference))
        / max(
            float(np.max(np.abs(value))),
            float(np.max(np.abs(reference))),
            np.finfo(float).tiny,
        )
    )


def _execute() -> tuple[dict, dict[str, np.ndarray]]:
    validated = _validate_parent(require_clean=False)
    audit = validated["contract"]["order_aware_inverse_audit"]
    context, _profile, _charts = original.fixed_q_implementation._primary_setup()
    directions = previous._deterministic_directions()
    profile_records = {}
    arrays: dict[str, np.ndarray] = {}
    all_raw = []
    all_richardson = []
    all_disagreements = []
    all_constraints = []
    requested_solves = 0
    start = time.perf_counter()
    with np.load(previous.TRUTH_ARRAYS) as archive:
        for label in audit["profiles"]:
            charts = np.asarray(archive[f"{label}_base_charts7"])
            targets = np.asarray(archive[f"{label}_slow_targets_MJE"])
            radial = charts[:, 1]
            stress = charts[:, 4]
            shape = (len(original.SELECTED_CELLS), len(audit["ordered_central_steps"]))
            raw_defects = np.empty(shape)
            finite = np.empty(shape + (3,))
            analytic = np.empty((shape[0], 3))
            plus_constraints = np.empty(shape)
            minus_constraints = np.empty(shape)
            for ordinal, cell in enumerate(original.SELECTED_CELLS):
                matrix = hydrostatic_invariant_local_scaled_jacobian(
                    context, cell, charts[cell]
                )
                direction = directions[label][ordinal]
                analytic[ordinal] = np.linalg.solve(matrix, direction)
                target_scale = np.abs(targets[cell])
                for step_ordinal, step in enumerate(audit["ordered_central_steps"]):
                    plus_targets = np.array(targets, copy=True)
                    minus_targets = np.array(targets, copy=True)
                    plus_targets[cell] += step * target_scale * direction
                    minus_targets[cell] -= step * target_scale * direction
                    plus = reconstruct_hydrostatic_fixed_invariants(
                        context,
                        plus_targets,
                        radial,
                        stress,
                        template_charts=charts,
                        constraint_tolerance=audit["reconstruction_tolerance"],
                    )
                    minus = reconstruct_hydrostatic_fixed_invariants(
                        context,
                        minus_targets,
                        radial,
                        stress,
                        template_charts=charts,
                        constraint_tolerance=audit["reconstruction_tolerance"],
                    )
                    requested_solves += 2
                    finite[ordinal, step_ordinal] = (
                        plus.primitive_charts[cell, UNKNOWN_INDICES]
                        - minus.primitive_charts[cell, UNKNOWN_INDICES]
                    ) / (2.0 * step * UNKNOWN_SCALES)
                    raw_defects[ordinal, step_ordinal] = _relative_defect(
                        finite[ordinal, step_ordinal], analytic[ordinal]
                    )
                    plus_constraints[ordinal, step_ordinal] = (
                        plus.maximum_constraint_relative_defect
                    )
                    minus_constraints[ordinal, step_ordinal] = (
                        minus.maximum_constraint_relative_defect
                    )
            richardson_coarse = (4.0 * finite[:, 1] - finite[:, 0]) / 3.0
            richardson_fine = (4.0 * finite[:, 2] - finite[:, 1]) / 3.0
            richardson_defects = np.asarray(
                [
                    [
                        _relative_defect(richardson_coarse[cell], analytic[cell]),
                        _relative_defect(richardson_fine[cell], analytic[cell]),
                    ]
                    for cell in range(shape[0])
                ]
            )
            disagreements = np.asarray(
                [
                    _relative_defect(richardson_coarse[cell], richardson_fine[cell])
                    for cell in range(shape[0])
                ]
            )
            monotone = bool(
                np.all(raw_defects[:, 2] <= raw_defects[:, 1])
                and np.all(raw_defects[:, 1] <= raw_defects[:, 0])
            )
            constraint_max = float(
                max(np.max(plus_constraints), np.max(minus_constraints))
            )
            profile_passed = bool(
                monotone
                and np.max(richardson_defects)
                <= audit["maximum_each_Richardson_JVP_relative_defect"]
                and np.max(disagreements)
                <= audit["maximum_Richardson_pair_relative_disagreement"]
                and constraint_max
                <= audit["maximum_reconstruction_constraint_relative_defect"]
            )
            profile_records[label] = {
                "passed": profile_passed,
                "raw_defects_nonincreasing": monotone,
                "maximum_raw_JVP_relative_defect": float(np.max(raw_defects)),
                "maximum_Richardson_JVP_relative_defect": float(
                    np.max(richardson_defects)
                ),
                "maximum_Richardson_pair_relative_disagreement": float(
                    np.max(disagreements)
                ),
                "maximum_reconstruction_constraint_relative_defect": constraint_max,
            }
            all_raw.append(raw_defects)
            all_richardson.extend(richardson_defects.ravel())
            all_disagreements.extend(disagreements)
            all_constraints.extend(plus_constraints.ravel())
            all_constraints.extend(minus_constraints.ravel())
            arrays.update(
                {
                    f"{label}_directions": directions[label],
                    f"{label}_selected_cells": np.asarray(original.SELECTED_CELLS),
                    f"{label}_central_steps": np.asarray(audit["ordered_central_steps"]),
                    f"{label}_raw_JVP_relative_defects": raw_defects,
                    f"{label}_finite_scaled_tangents": finite,
                    f"{label}_analytic_scaled_tangents": analytic,
                    f"{label}_Richardson_coarse_scaled_tangents": richardson_coarse,
                    f"{label}_Richardson_fine_scaled_tangents": richardson_fine,
                    f"{label}_Richardson_JVP_relative_defects": richardson_defects,
                    f"{label}_Richardson_pair_relative_disagreements": disagreements,
                    f"{label}_plus_constraint_relative_defects": plus_constraints,
                    f"{label}_minus_constraint_relative_defects": minus_constraints,
                }
            )
    all_raw_array = np.asarray(all_raw)
    global_worst = np.max(all_raw_array, axis=(0, 1))
    observed_orders = np.log(global_worst[:-1] / global_worst[1:]) / np.log(2.0)
    passed = bool(
        all(record["passed"] for record in profile_records.values())
        and np.min(observed_orders) >= audit["minimum_global_worst_defect_order"]
        and requested_solves == audit["maximum_new_local_nonlinear_solves"]
    )
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,
        "passed": passed,
        "fixed_step_inverse_JVP_rejection_preserved": True,
        "profiles": profile_records,
        "global_worst_raw_JVP_relative_defects": global_worst.tolist(),
        "global_worst_raw_defect_observed_orders": observed_orders.tolist(),
        "minimum_global_worst_raw_defect_order": float(np.min(observed_orders)),
        "maximum_Richardson_JVP_relative_defect": float(max(all_richardson)),
        "maximum_Richardson_pair_relative_disagreement": float(
            max(all_disagreements)
        ),
        "maximum_reconstruction_constraint_relative_defect": float(
            max(all_constraints)
        ),
        "requested_local_nonlinear_solves": requested_solves,
        "new_seven_field_operator_calls": 0,
        "new_global_roots": 0,
        "propagated_states": 0,
        "execution_wall_seconds": time.perf_counter() - start,
    }
    return metrics, arrays


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
                    "scientific_status": "SUPPORTED" if summary["passed"] else "REJECTED",
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
        raise RuntimeError("order-recovery execution already exists")
    validated = _validate_parent(require_clean=True)
    utils = _utils()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "order_recovery_metrics.json", metrics)
    np.savez_compressed(CANONICAL_DIRECTORY / "order_recovery_arrays.npz", **arrays)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": bool(metrics["passed"]),
        "fixed_step_inverse_JVP_rejection_preserved": True,
        "hydrostatic_implicit_inverse_tangent_certified": bool(metrics["passed"]),
        "four_saved_truth_samples_certified_for_atlas_use": bool(metrics["passed"]),
        "new_seven_field_operator_calls": 0,
        "new_global_roots": 0,
        "propagated_states": 0,
        "local_slow_flux_atlas_manifest_authorized": bool(metrics["passed"]),
        "slow_flux_atlas_execution_authorized": False,
        "online_macro_solver_authorized": False,
        "complete_cycle_execution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT_ON_PASS if metrics["passed"] else None,
    }
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utils._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_artifact": parent.ARTIFACT,
            "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256,
            "parent_hashes": validated["hashes"],
            "truth_arrays": str(previous.TRUTH_ARRAYS.relative_to(ROOT)),
            "truth_arrays_sha256": previous.TRUTH_ARRAYS_SHA256,
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Hydrostatic inverse order-recovery execution",
                "",
                f"Classification: `{metrics['classification']}`.",
                "",
                f"Worst raw defects: `{metrics['global_worst_raw_JVP_relative_defects']}`; observed orders: `{metrics['global_worst_raw_defect_observed_orders']}`.",
                "",
                f"Maximum Richardson tangent defect: `{metrics['maximum_Richardson_JVP_relative_defect']:.6e}`; maximum independent-limit disagreement: `{metrics['maximum_Richardson_pair_relative_disagreement']:.6e}`.",
                "",
                "The earlier fixed-step audit remains rejected. No seven-field operator, global root, or state propagation was run.",
                "",
                f"Authorized next: `{summary['authorized_next']}`.",
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
            f"{utils._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("choose --run")
    metrics, arrays = _execute()
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    summary = _canonicalize(metrics, arrays)
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
