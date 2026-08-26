#!/usr/bin/env python3
"""Execute the frozen derivative-only hydrostatic inverse refinement."""

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

import run_causal_inner_entropy_complete_hydrostatic_inverse_jvp_refinement_manifest_wp10c9d6c7c3b5c4f25fizfb as parent  # noqa: E402
import run_causal_inner_entropy_complete_hydrostatic_invariant_reconstruction_implementation_wp10c9d6c7c3b5c4f25fizfa as original  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_quasisteady import (  # noqa: E402
    hydrostatic_invariant_local_scaled_jacobian,
    reconstruct_hydrostatic_fixed_invariants,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fizfc_"
    "entropy_complete_hydrostatic_inverse_JVP_refinement_execution"
)
PASS_CLASSIFICATION = "entropy_complete_hydrostatic_inverse_JVP_refinement_certified"
FAIL_CLASSIFICATION = "entropy_complete_hydrostatic_inverse_JVP_refinement_failed"
AUTHORIZED_NEXT_ON_PASS = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizfd_"
    "entropy_complete_local_slow_flux_atlas_manifest"
)
ARTIFACT = (
    "causal_inner_entropy_complete_hydrostatic_inverse_jvp_refinement_"
    "execution_wp10c9d6c7c3b5c4f25fizfc"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_HYDROSTATIC_"
    "INVERSE_JVP_REFINEMENT_EXECUTION_WP10C9D6C7C3B5C4F25FIZFC_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_entropy_complete_hydrostatic_inverse_jvp_"
    "refinement_execution_wp10c9d6c7c3b5c4f25fizfc.py"
)
THIS_TEST = (
    "tests/test_causal_inner_entropy_complete_hydrostatic_inverse_jvp_"
    "refinement_execution_wp10c9d6c7c3b5c4f25fizfc.py"
)
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "b9f602fbde77f409d6c0a3afe048ae309bc06e5781119af46255ec5a3a2e7d7f"
)
TRUTH_ARRAYS = original.CANONICAL_DIRECTORY / "implementation_arrays.npz"
TRUTH_ARRAYS_SHA256 = (
    "98ccc8c6a103e88ed208444021df4f62c539d9e54be1fbeeedbada6bdc6612e0"
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
        raise RuntimeError("inverse-JVP refinement manifest checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    contract = utils._read_json(
        parent.CANONICAL_DIRECTORY / "inverse_jvp_refinement_contract.json"
    )
    if (
        summary["classification"] != parent.CLASSIFICATION
        or not summary["passed"]
        or not summary["definitions_only"]
        or not summary["original_inverse_JVP_rejection_preserved"]
        or not summary["four_truth_samples_preserved"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["authorized_next"] != WORK_PACKAGE
        or contract["refinement"]["new_seven_field_radial_operator_calls"] != 0
        or contract["claim_boundary"]["complete_cycle_execution_authorized"]
    ):
        raise RuntimeError("inverse-JVP refinement authorization changed")
    for relative, expected in utils._read_json(
        parent.CANONICAL_DIRECTORY / "provenance.json"
    )["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"inverse-JVP refinement source changed: {relative}")
    if utils._sha256(TRUTH_ARRAYS) != TRUTH_ARRAYS_SHA256:
        raise RuntimeError("saved seven-field truth arrays changed")
    original_validation = parent._validate_parent(require_clean=False)
    if original_validation["summary"]["classification"] != original.FAIL_CLASSIFICATION:
        raise RuntimeError("original narrow JVP rejection was not preserved")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("inverse-JVP refinement execution requires a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "contract": contract}


def _deterministic_directions() -> dict[str, np.ndarray]:
    generator = np.random.default_rng(original.RANDOM_SEED)
    directions = {}
    for label in ("primary_20ms", "heldout_16ms"):
        values = []
        for _cell in original.SELECTED_CELLS:
            direction = generator.normal(size=3)
            direction /= np.linalg.norm(direction)
            values.append(direction)
        directions[label] = np.asarray(values)
    return directions


def _execute() -> tuple[dict, dict[str, np.ndarray]]:
    validated = _validate_parent(require_clean=False)
    refinement = validated["contract"]["refinement"]
    context, _profile, _charts = original.fixed_q_implementation._primary_setup()
    directions = _deterministic_directions()
    profile_records = {}
    saved_arrays: dict[str, np.ndarray] = {}
    all_defects = []
    all_constraint_defects = []
    requested_solves = 0
    start = time.perf_counter()
    with np.load(TRUTH_ARRAYS) as archive:
        for label in refinement["profiles"]:
            charts = np.asarray(archive[f"{label}_base_charts7"])
            targets = np.asarray(archive[f"{label}_slow_targets_MJE"])
            radial = charts[:, 1]
            stress = charts[:, 4]
            profile_defects = np.empty(
                (len(original.SELECTED_CELLS), len(refinement["ordered_central_steps"]))
            )
            plus_constraints = np.empty_like(profile_defects)
            minus_constraints = np.empty_like(profile_defects)
            corrections = np.empty(profile_defects.shape + (2,), dtype=int)
            finite_tangents = np.empty(profile_defects.shape + (3,))
            analytic_tangents = np.empty((len(original.SELECTED_CELLS), 3))
            conditions = np.empty(len(original.SELECTED_CELLS))
            for ordinal, cell in enumerate(original.SELECTED_CELLS):
                matrix = hydrostatic_invariant_local_scaled_jacobian(
                    context, cell, charts[cell]
                )
                conditions[ordinal] = np.linalg.cond(matrix)
                direction = directions[label][ordinal]
                analytic = np.linalg.solve(matrix, direction)
                analytic_tangents[ordinal] = analytic
                target_scale = np.abs(targets[cell])
                for step_ordinal, step in enumerate(
                    refinement["ordered_central_steps"]
                ):
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
                        constraint_tolerance=refinement["reconstruction_tolerance"],
                    )
                    minus = reconstruct_hydrostatic_fixed_invariants(
                        context,
                        minus_targets,
                        radial,
                        stress,
                        template_charts=charts,
                        constraint_tolerance=refinement["reconstruction_tolerance"],
                    )
                    requested_solves += 2
                    finite = (
                        plus.primitive_charts[cell, UNKNOWN_INDICES]
                        - minus.primitive_charts[cell, UNKNOWN_INDICES]
                    ) / (2.0 * step * UNKNOWN_SCALES)
                    defect = float(
                        np.max(np.abs(finite - analytic))
                        / max(
                            float(np.max(np.abs(finite))),
                            float(np.max(np.abs(analytic))),
                            np.finfo(float).tiny,
                        )
                    )
                    profile_defects[ordinal, step_ordinal] = defect
                    plus_constraints[ordinal, step_ordinal] = (
                        plus.maximum_constraint_relative_defect
                    )
                    minus_constraints[ordinal, step_ordinal] = (
                        minus.maximum_constraint_relative_defect
                    )
                    corrections[ordinal, step_ordinal] = (
                        plus.maximum_newton_corrections,
                        minus.maximum_newton_corrections,
                    )
                    finite_tangents[ordinal, step_ordinal] = finite
            max_defect = float(np.max(profile_defects))
            max_constraint = float(
                max(np.max(plus_constraints), np.max(minus_constraints))
            )
            combination_pass = profile_defects <= refinement[
                "maximum_JVP_relative_defect"
            ]
            constraint_pass = (
                plus_constraints <= refinement["reconstruction_tolerance"]
            ) & (minus_constraints <= refinement["reconstruction_tolerance"])
            profile_records[label] = {
                "passed": bool(np.all(combination_pass) and np.all(constraint_pass)),
                "maximum_JVP_relative_defect": max_defect,
                "maximum_reconstruction_constraint_relative_defect": max_constraint,
                "maximum_scaled_local_inverse_condition_number": float(
                    np.max(conditions)
                ),
                "all_profile_cell_step_combinations_passed": bool(
                    np.all(combination_pass)
                ),
                "all_reconstructions_passed_tight_tolerance": bool(
                    np.all(constraint_pass)
                ),
                "requested_local_nonlinear_solves": int(2 * profile_defects.size),
            }
            all_defects.extend(profile_defects.ravel())
            all_constraint_defects.extend(plus_constraints.ravel())
            all_constraint_defects.extend(minus_constraints.ravel())
            saved_arrays.update(
                {
                    f"{label}_directions": directions[label],
                    f"{label}_selected_cells": np.asarray(original.SELECTED_CELLS),
                    f"{label}_central_steps": np.asarray(
                        refinement["ordered_central_steps"]
                    ),
                    f"{label}_JVP_relative_defects": profile_defects,
                    f"{label}_plus_constraint_relative_defects": plus_constraints,
                    f"{label}_minus_constraint_relative_defects": minus_constraints,
                    f"{label}_newton_corrections": corrections,
                    f"{label}_finite_scaled_tangents": finite_tangents,
                    f"{label}_analytic_scaled_tangents": analytic_tangents,
                    f"{label}_scaled_inverse_condition_numbers": conditions,
                }
            )
    expected_solves = refinement["maximum_new_nonlinear_slow_invariant_solves"]
    passed = (
        all(record["passed"] for record in profile_records.values())
        and requested_solves == expected_solves
        and max(all_defects) <= refinement["maximum_JVP_relative_defect"]
        and max(all_constraint_defects) <= refinement["reconstruction_tolerance"]
    )
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,
        "passed": passed,
        "original_inverse_JVP_rejection_preserved": True,
        "saved_four_truth_samples_reused_by_hash": True,
        "profiles": profile_records,
        "maximum_JVP_relative_defect": float(max(all_defects)),
        "maximum_reconstruction_constraint_relative_defect": float(
            max(all_constraint_defects)
        ),
        "unchanged_JVP_gate": refinement["maximum_JVP_relative_defect"],
        "requested_local_nonlinear_solves": requested_solves,
        "offline_seven_field_operator_calls": 0,
        "new_nonlinear_global_roots": 0,
        "propagated_states": 0,
        "execution_wall_seconds": time.perf_counter() - start,
    }
    return metrics, saved_arrays


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
        raise RuntimeError("inverse-JVP refinement execution already exists")
    validated = _validate_parent(require_clean=True)
    utils = _utils()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "refinement_metrics.json", metrics)
    np.savez_compressed(CANONICAL_DIRECTORY / "refinement_arrays.npz", **arrays)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": bool(metrics["passed"]),
        "original_inverse_JVP_rejection_preserved": True,
        "hydrostatic_invariant_inverse_certified_by_refinement": bool(
            metrics["passed"]
        ),
        "four_saved_truth_samples_certified_for_atlas_use": bool(metrics["passed"]),
        "new_seven_field_operator_calls": 0,
        "new_nonlinear_global_roots": 0,
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
            "truth_arrays": str(TRUTH_ARRAYS.relative_to(ROOT)),
            "truth_arrays_sha256": TRUTH_ARRAYS_SHA256,
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Hydrostatic inverse-JVP derivative refinement",
                "",
                f"Classification: `{metrics['classification']}`.",
                "",
                f"Maximum inverse-JVP defect: `{metrics['maximum_JVP_relative_defect']:.6e}` against the unchanged `2e-5` gate.",
                "",
                f"Maximum reconstruction defect: `{metrics['maximum_reconstruction_constraint_relative_defect']:.6e}`. All `{metrics['requested_local_nonlinear_solves']}` prospectively frozen local solves were evaluated.",
                "",
                "The original narrow rejection remains preserved. No new seven-field truth operator, global root, or propagated state was evaluated.",
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
