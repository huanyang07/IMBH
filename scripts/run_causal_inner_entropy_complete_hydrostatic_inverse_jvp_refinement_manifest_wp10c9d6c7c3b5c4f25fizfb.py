#!/usr/bin/env python3
"""Freeze a derivative-only refinement after the hydrostatic JVP miss."""

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

import run_causal_inner_entropy_complete_hydrostatic_invariant_reconstruction_implementation_wp10c9d6c7c3b5c4f25fizfa as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fizfb_"
    "entropy_complete_hydrostatic_inverse_JVP_refinement_manifest"
)
CLASSIFICATION = (
    "entropy_complete_hydrostatic_inverse_JVP_refinement_manifest_frozen"
)
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizfc_"
    "entropy_complete_hydrostatic_inverse_JVP_refinement_execution"
)
ARTIFACT = (
    "causal_inner_entropy_complete_hydrostatic_inverse_jvp_refinement_"
    "manifest_wp10c9d6c7c3b5c4f25fizfb"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_HYDROSTATIC_"
    "INVERSE_JVP_REFINEMENT_MANIFEST_WP10C9D6C7C3B5C4F25FIZFB_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_entropy_complete_hydrostatic_inverse_jvp_"
    "refinement_manifest_wp10c9d6c7c3b5c4f25fizfb.py"
)
THIS_TEST = (
    "tests/test_causal_inner_entropy_complete_hydrostatic_inverse_jvp_"
    "refinement_manifest_wp10c9d6c7c3b5c4f25fizfb.py"
)
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "f66d02189c1e3f9507c9c977427af951bb1e5bb36505d4d0ba0d3f0663ae46a2"
)
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _utils():
    return parent._utils()


def _validate_parent(*, require_clean: bool) -> dict:
    utils = _utils()
    if (
        utils._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt")
        != PARENT_CHECKSUM_MANIFEST_SHA256
    ):
        raise RuntimeError("hydrostatic inverse rejection checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = utils._read_json(
        parent.CANONICAL_DIRECTORY / "implementation_metrics.json"
    )
    primary = metrics["profiles"]["primary_20ms"]
    heldout = metrics["profiles"]["heldout_16ms"]
    if (
        summary["classification"] != parent.FAIL_CLASSIFICATION
        or summary["passed"]
        or summary["hydrostatic_invariant_inverse_certified"]
        or summary["authorized_next"] is not None
        or metrics["offline_seven_field_operator_calls"] != 4
        or metrics["propagated_states"] != 0
        or metrics["maximum_local_inverse_JVP_relative_defect"]
        != 2.2873090291353505e-05
        or primary["checks"]["local_inverse_JVP"]
        or not all(
            value
            for key, value in primary["checks"].items()
            if key != "local_inverse_JVP"
        )
        or not all(heldout["checks"].values())
    ):
        raise RuntimeError("hydrostatic inverse rejection classification changed")
    for relative, expected in utils._read_json(
        parent.CANONICAL_DIRECTORY / "provenance.json"
    )["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"hydrostatic inverse source changed: {relative}")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("inverse-JVP refinement manifest requires a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "metrics": metrics}


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "preserved_result": {
            "original_implementation_classification": parent.FAIL_CLASSIFICATION,
            "original_maximum_JVP_relative_defect": 2.2873090291353505e-05,
            "original_gate": 2.0e-5,
            "four_seven_field_truth_samples_remain_valid": True,
            "all_original_physical_and_constraint_gates_except_JVP_passed": True,
            "no_truth_operator_rerun_authorized": True,
        },
        "numerical_diagnosis": {
            "failing_profile": "primary_20ms",
            "failing_cell": 111,
            "scaled_local_inverse_condition_number": 987.791865380874,
            "old_reconstruction_tolerance": 1.0e-10,
            "old_central_step": 2.0e-6,
            "old_tolerance_over_step_floor": 5.0e-5,
            "hypothesis": "nonlinear_solve_floor_amplified_by_the_central_difference",
        },
        "refinement": {
            "profiles": ("primary_20ms", "heldout_16ms"),
            "selected_cells": (0, 18, 36, 55, 74, 92, 111),
            "reconstruction_tolerance": 1.0e-12,
            "ordered_central_steps": (1.0e-5, 5.0e-6, 2.0e-6),
            "same_deterministic_directions_as_original_execution": True,
            "maximum_JVP_relative_defect": 2.0e-5,
            "every_profile_cell_step_combination_must_pass": True,
            "maximum_new_nonlinear_slow_invariant_solves": 84,
            "new_seven_field_radial_operator_calls": 0,
            "no_gate_or_physical_equation_is_relaxed": True,
        },
        "decision": {
            "pass": "authorize_definitions_only_local_slow_flux_atlas_manifest_using_saved_truth_samples",
            "fail": "block_partial_equilibrium_architecture_and_require_inverse_map_redesign",
        },
        "claim_boundary": {
            "derivative_refinement_authorized": True,
            "truth_resampling_authorized": False,
            "slow_flux_atlas_authorized": False,
            "online_macro_solver_authorized": False,
            "complete_cycle_execution_authorized": False,
        },
        "authorized_next": AUTHORIZED_NEXT,
    }


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


def _freeze() -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("inverse-JVP refinement manifest already exists")
    validated = _validate_parent(require_clean=True)
    utils = _utils()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(
        CANONICAL_DIRECTORY / "inverse_jvp_refinement_contract.json", _contract()
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "original_inverse_JVP_rejection_preserved": True,
        "four_truth_samples_preserved": True,
        "derivative_refinement_authorized": True,
        "truth_resampling_authorized": False,
        "slow_flux_atlas_authorized": False,
        "complete_cycle_execution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utils._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_artifact": parent.ARTIFACT,
            "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256,
            "parent_hashes": validated["hashes"],
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Hydrostatic inverse-JVP refinement manifest",
                "",
                f"Classification: `{CLASSIFICATION}`.",
                "",
                "The original narrow derivative rejection remains binding. This package freezes a tighter local-solve tolerance and three central steps while retaining the unchanged `2e-5` JVP gate.",
                "",
                "The four accepted seven-field operator samples are reused by hash; no truth operator, root, or trajectory may run.",
                "",
                f"Authorized next: `{AUTHORIZED_NEXT}` only.",
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
            "scientific_status": "DEFINITIONS_ONLY",
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
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if not args.freeze:
        parser.error("choose --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
