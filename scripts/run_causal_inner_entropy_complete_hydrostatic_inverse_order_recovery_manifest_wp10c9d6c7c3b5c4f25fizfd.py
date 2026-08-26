#!/usr/bin/env python3
"""Freeze an order-aware recovery after the fixed-step inverse-JVP rejection."""

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

import run_causal_inner_entropy_complete_hydrostatic_inverse_jvp_refinement_execution_wp10c9d6c7c3b5c4f25fizfc as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fizfd_"
    "entropy_complete_hydrostatic_inverse_order_recovery_manifest"
)
CLASSIFICATION = "entropy_complete_hydrostatic_inverse_order_recovery_manifest_frozen"
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizfe_"
    "entropy_complete_hydrostatic_inverse_order_recovery_execution"
)
ARTIFACT = (
    "causal_inner_entropy_complete_hydrostatic_inverse_order_recovery_"
    "manifest_wp10c9d6c7c3b5c4f25fizfd"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_HYDROSTATIC_"
    "INVERSE_ORDER_RECOVERY_MANIFEST_WP10C9D6C7C3B5C4F25FIZFD_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_entropy_complete_hydrostatic_inverse_order_"
    "recovery_manifest_wp10c9d6c7c3b5c4f25fizfd.py"
)
THIS_TEST = (
    "tests/test_causal_inner_entropy_complete_hydrostatic_inverse_order_"
    "recovery_manifest_wp10c9d6c7c3b5c4f25fizfd.py"
)
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "1277b45ea223bbee2b881a167dc0e012a2a6992d56f21fa992a5ec2ec88501bf"
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
        raise RuntimeError("fixed-step inverse-JVP rejection checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = utils._read_json(parent.CANONICAL_DIRECTORY / "refinement_metrics.json")
    defects = []
    with np.load(parent.CANONICAL_DIRECTORY / "refinement_arrays.npz") as archive:
        for label in ("primary_20ms", "heldout_16ms"):
            defects.append(np.asarray(archive[f"{label}_JVP_relative_defects"]))
    defects = np.asarray(defects)
    if (
        summary["classification"] != parent.FAIL_CLASSIFICATION
        or summary["passed"]
        or summary["authorized_next"] is not None
        or not summary["original_inverse_JVP_rejection_preserved"]
        or metrics["maximum_JVP_relative_defect"] != 0.0005720414141375823
        or metrics["offline_seven_field_operator_calls"] != 0
        or metrics["propagated_states"] != 0
        or not np.all(defects[:, :, 2] <= defects[:, :, 1])
        or not np.all(defects[:, :, 1] <= defects[:, :, 0])
    ):
        raise RuntimeError("fixed-step inverse-JVP rejection classification changed")
    for relative, expected in utils._read_json(
        parent.CANONICAL_DIRECTORY / "provenance.json"
    )["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"fixed-step inverse-JVP source changed: {relative}")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("order-recovery manifest requires a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "metrics": metrics}


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "preserved_rejections": {
            "original_fixed_step_refinement_remains_failed": True,
            "original_maximum_defect": 0.0005720414141375823,
            "no_original_result_is_reclassified": True,
        },
        "diagnosis": {
            "observed_worst_defects_at_1e_5_5e_6_2e_6": (
                0.0005720414141375823,
                0.00014296297085423685,
                2.2873090291353505e-05,
            ),
            "hypothesis": "central_inverse_difference_has_second_order_truncation_on_the_high_condition_outer_cell",
            "required_discriminator": "independent_refinement_and_two_Richardson_limits",
        },
        "order_aware_inverse_audit": {
            "profiles": ("primary_20ms", "heldout_16ms"),
            "selected_cells": tuple(parent.original.SELECTED_CELLS),
            "same_deterministic_directions": True,
            "reconstruction_tolerance": 1.0e-12,
            "ordered_central_steps": (8.0e-6, 4.0e-6, 2.0e-6),
            "raw_defects_must_be_nonincreasing_under_refinement": True,
            "minimum_global_worst_defect_order": 1.8,
            "richardson_formula": "D_R(h,h/2)=(4*D(h/2)-D(h))/3",
            "maximum_each_Richardson_JVP_relative_defect": 2.0e-5,
            "maximum_Richardson_pair_relative_disagreement": 2.0e-5,
            "maximum_reconstruction_constraint_relative_defect": 1.0e-12,
            "maximum_new_local_nonlinear_solves": 84,
            "new_seven_field_operator_calls": 0,
            "new_global_roots": 0,
            "propagated_states": 0,
        },
        "decision": {
            "pass": "certify_implicit_local_inverse_tangent_and_authorize_saved_truth_atlas_manifest",
            "fail": "reject_partial_equilibrium_inverse_and_redesign_local_macro_coordinates",
        },
        "claim_boundary": {
            "diagnostic_execution_authorized": True,
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
        raise RuntimeError("order-recovery manifest already exists")
    validated = _validate_parent(require_clean=True)
    utils = _utils()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "order_recovery_contract.json", _contract())
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "fixed_step_inverse_JVP_rejection_preserved": True,
        "order_aware_inverse_diagnostic_authorized": True,
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
                "# Hydrostatic inverse order-recovery manifest",
                "",
                f"Classification: `{CLASSIFICATION}`.",
                "",
                "The fixed-step refinement remains rejected. Its monotone quadratic defect pattern motivates a new independent 8e-6, 4e-6, 2e-6 ladder and two Richardson limits.",
                "",
                "The original `2e-5` tangent gate and `1e-12` reconstruction gate remain unchanged. No seven-field operator, global root, or trajectory is authorized.",
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
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if not args.freeze:
        parser.error("choose --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
