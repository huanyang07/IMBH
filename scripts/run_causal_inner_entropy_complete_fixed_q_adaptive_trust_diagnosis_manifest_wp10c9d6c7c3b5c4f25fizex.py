#!/usr/bin/env python3
"""Freeze an adaptive-trust/hyperbolicity diagnosis after root rejection."""

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

import run_causal_inner_entropy_complete_fixed_q_primary_inexact_newton_root_execution_wp10c9d6c7c3b5c4f25fizew as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fizex_"
    "entropy_complete_fixed_Q_adaptive_trust_diagnosis_manifest"
)
CLASSIFICATION = "entropy_complete_fixed_Q_adaptive_trust_diagnosis_manifest_frozen"
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizey_"
    "entropy_complete_fixed_Q_adaptive_trust_diagnosis_execution"
)
ARTIFACT = (
    "causal_inner_entropy_complete_fixed_q_adaptive_trust_diagnosis_manifest_"
    "wp10c9d6c7c3b5c4f25fizex"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_FIXED_Q_"
    "ADAPTIVE_TRUST_DIAGNOSIS_MANIFEST_WP10C9D6C7C3B5C4F25FIZEX_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = "scripts/run_causal_inner_entropy_complete_fixed_q_adaptive_trust_diagnosis_manifest_wp10c9d6c7c3b5c4f25fizex.py"
THIS_TEST = "tests/test_causal_inner_entropy_complete_fixed_q_adaptive_trust_diagnosis_manifest_wp10c9d6c7c3b5c4f25fizex.py"
PARENT_CHECKSUM_MANIFEST_SHA256 = "177f2c9b73b3f76b3347e34fe1526396b8716800eee91e58bf836f0662785f60"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _utils(): return parent._utils()


def _validate_parent(*, require_clean: bool) -> dict:
    utils = _utils()
    if utils._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != PARENT_CHECKSUM_MANIFEST_SHA256:
        raise RuntimeError("primary root rejection checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = utils._read_json(parent.CANONICAL_DIRECTORY / "root_metrics.json")
    if (
        summary["classification"] != parent.FAIL_CLASSIFICATION
        or summary["passed"]
        or summary["root_exists"]
        or summary["propagated_states"] != 0
        or summary["authorized_next"] is not None
        or metrics["failure_reason"] != "inexact_direction_or_complete_line_search_failed_after_refresh"
        or metrics["accepted_nonlinear_corrections"] != 2
        or metrics["solver_exact_colored_assemblies"] != 2
    ):
        raise RuntimeError("primary root rejection classification changed")
    for relative, expected in utils._read_json(parent.CANONICAL_DIRECTORY / "provenance.json")["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"primary root source changed: {relative}")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("adaptive-trust manifest requires a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "metrics": metrics}


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "preserved_rejection": {
            "classification": parent.FAIL_CLASSIFICATION,
            "no_root_was_found": True,
            "no_physical_failure_was_certified": True,
            "no_state_was_propagated": True,
        },
        "diagnostic_state": "hash_locked_second_accepted_coordinate_and_residual",
        "fresh_equation_linearization": {
            "one_12_color_equation_Jacobian": True,
            "independent_central_JVP_directions": 2,
            "maximum_JVP_relative_defect": 2.0e-5,
            "assembly_is_diagnostic_and_cannot_update_a_root": True,
        },
        "adaptive_trust_candidates": {
            "ordered_trust_radii": [0.02, 0.01, 0.005, 0.0025, 0.001],
            "new_bounded_TRF_direction_at_each_radius": True,
            "line_searching_the_old_radius_0p25_direction_is_not_used": True,
            "binding_midpoint_eigenvalue_imaginary_ratio": 1.0e-10,
            "fixed_slow_reconstruction_required": True,
            "largest_feasible_radius_evaluated_first": True,
            "maximum_full_physical_candidate_evaluations": 5,
        },
        "useful_progress": {
            "maximum_actual_two_norm_ratio": 0.98,
            "maximum_actual_infinity_norm_ratio": 0.98,
            "all_physical_and_equation_rate_parity_gates_required": True,
            "selected_candidate_is_not_a_root_and_must_not_be_propagated": True,
        },
        "decision": {
            "feasible_useful_progress": "authorize_definitions_only_adaptive_trust_primary_root_retry_manifest",
            "no_feasible_useful_progress": "reject_generic_fixed_point_Newton_and_authorize_analytic_quasisteady_closure_manifest",
            "no_threshold_may_be_relaxed_after_results": True,
        },
        "claim_boundary": {
            "diagnosis_authorized": True,
            "root_retry_authorized": False,
            "heldout_root_authorized": False,
            "slow_flux_atlas_authorized": False,
            "complete_cycle_execution_authorized": False,
        },
        "authorized_next": AUTHORIZED_NEXT,
    }


def _update_catalog(summary: dict) -> None:
    utils = _utils()
    with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file(): rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": utils._sha256(path), "scientific_status": "DEFINITIONS_ONLY"})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    catalog = utils._read_json(CANONICAL_SUMMARY); catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": summary["classification"], "passed": summary["passed"]}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": utils._git("rev-parse", "HEAD"), "latest_work_package": WORK_PACKAGE}); utils._write_json(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists(): raise RuntimeError("adaptive-trust diagnosis manifest exists")
    validated = _validate_parent(require_clean=True); utils = _utils(); CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "adaptive_trust_diagnosis_contract.json", _contract()); utils._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"parent_artifact": parent.ARTIFACT, "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256, "parent_hashes": validated["hashes"], "final_residual_infinity": validated["metrics"]["final_normalized_equation_residual_infinity"]})
    summary = {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "classification": CLASSIFICATION, "passed": True, "definitions_only": True, "primary_root_rejection_preserved": True, "adaptive_trust_diagnosis_authorized": True, "root_retry_authorized": False, "heldout_root_authorized": False, "slow_flux_atlas_authorized": False, "complete_cycle_execution_authorized": False, "authorized_next": AUTHORIZED_NEXT}
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True); REPORT_PATH.write_text("\n".join(("# Entropy-complete fixed-Q adaptive-trust diagnosis manifest", "", f"Classification: `{CLASSIFICATION}`.", "", "The failed large-box Newton policy remains rejected. This package authorizes one exact linearization at the last accepted state and prospectively recomputed smaller trust-region directions. It executes no root.", "", f"Authorized next: `{AUTHORIZED_NEXT}` only.", "")), encoding="utf-8")
    sources = (THIS_RUNNER, THIS_TEST, REPORT_RELATIVE); utils._write_json(CANONICAL_DIRECTORY / "provenance.json", {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "scientific_status": "DEFINITIONS_ONLY", "implementation_commit": utils._git("rev-parse", "HEAD"), "implementation_tree": utils._git("rev-parse", "HEAD^{tree}"), "source_hashes": {path: utils._sha256(ROOT / path) for path in sources}, "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "thread_environment": {name: os.environ.get(name, "") for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()); (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utils._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8"); _update_catalog(summary); return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--freeze", action="store_true"); args = parser.parse_args()
    if not args.freeze: parser.error("choose --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
