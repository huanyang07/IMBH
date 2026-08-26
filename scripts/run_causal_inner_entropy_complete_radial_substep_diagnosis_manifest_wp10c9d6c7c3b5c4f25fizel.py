#!/usr/bin/env python3
"""Freeze a nonpropagating timestep diagnosis of the rejected radial crossing."""

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

import run_causal_inner_entropy_complete_bounded_radial_crossing_execution_wp10c9d6c7c3b5c4f25fizek as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fizel_"
    "entropy_complete_radial_substep_diagnosis_manifest"
)
CLASSIFICATION = "entropy_complete_radial_substep_diagnosis_manifest_frozen"
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizem_"
    "entropy_complete_radial_substep_diagnosis_execution"
)
ARTIFACT = (
    "causal_inner_entropy_complete_radial_substep_diagnosis_manifest_"
    "wp10c9d6c7c3b5c4f25fizel"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_RADIAL_"
    "SUBSTEP_DIAGNOSIS_MANIFEST_WP10C9D6C7C3B5C4F25FIZEL_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = "scripts/run_causal_inner_entropy_complete_radial_substep_diagnosis_manifest_wp10c9d6c7c3b5c4f25fizel.py"
THIS_TEST = "tests/test_causal_inner_entropy_complete_radial_substep_diagnosis_manifest_wp10c9d6c7c3b5c4f25fizel.py"
PARENT_CHECKSUM_MANIFEST_SHA256 = "e5e3ac4c67ab818df2735fa29b1c14dc7bebb41724eb95e19bf3861ebe49bb0e"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
TIMESTEPS_SECONDS = (3.125e-5, 1.5625e-5, 7.8125e-6)
CHART_SCALES = np.asarray((1.0, 0.1, 0.1, 1.0, 1.0e-4, 1.0, 0.03))
HEADROOM_CHANGE = 0.03


def _utils():
    return parent._utils()


def _validate_parent(*, require_clean: bool) -> dict:
    utils = _utils()
    checksum = parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt"
    if utils._sha256(checksum) != PARENT_CHECKSUM_MANIFEST_SHA256:
        raise RuntimeError("bounded crossing rejection checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = utils._read_json(parent.CANONICAL_DIRECTORY / "execution_metrics.json")
    failure = metrics["first_failure"]
    if (
        summary["classification"] != parent.PHYSICAL_FAILURE
        or summary["passed"]
        or summary["accepted_new_steps"] != 0
        or summary["authorized_next"] is not None
        or metrics["accepted_new_steps"] != 0
        or failure["failure_reasons"] != ["physical:chart_change"]
        or failure["maximum_scaled_chart_change"] <= 0.05
        or failure["maximum_CFL"] >= 0.4
    ):
        raise RuntimeError("bounded crossing rejection changed")
    for relative, expected in utils._read_json(
        parent.CANONICAL_DIRECTORY / "provenance.json"
    )["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"bounded crossing rejection source changed: {relative}")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("substep diagnosis manifest requires clean tracked tree")
    return {"hashes": hashes, "summary": summary, "metrics": metrics}


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "diagnosis": {
            "preserved_parent_classification": parent.PHYSICAL_FAILURE,
            "coarse_timestep_seconds": 6.25e-5,
            "coarse_maximum_scaled_chart_change": 0.17443160649335754,
            "first_euler_stage_dominant_component": "causal_shear_chart_chi",
            "first_euler_stage_dominant_component_index": 4,
            "first_euler_stage_scaled_change": 0.17661301219600348,
            "coarse_CFL": 0.018272981381367334,
            "interpretation_to_test": "unresolved_causal_shear_relaxation_substep",
        },
        "experiment": {
            "seed": "same_hash_locked_accepted_terminal_hydrostatic_lift",
            "timesteps_seconds": list(TIMESTEPS_SECONDS),
            "independent_one_step_trials": True,
            "propagate_trial_endpoint": False,
            "quadrature_order": 8,
            "chart_scales": CHART_SCALES.tolist(),
            "all_original_crossing_gates_unchanged": True,
        },
        "binding_gates": {
            "maximum_scaled_chart_change_per_step": 0.05,
            "selected_substep_maximum_scaled_chart_change": HEADROOM_CHANGE,
            "minimum_adjacent_chart_change_order": 0.8,
            "maximum_adjacent_chart_change_order": 1.2,
            "all_non_chart_crossing_gates_pass": True,
            "checkpoint_roundtrip_bitwise": True,
            "fail_closed": True,
        },
        "selection": {
            "rule": "largest_tested_timestep_passing_all_original_gates_and_headroom_change",
            "require_at_least_one_selected_substep": True,
            "no_retrospective_tolerance_change": True,
        },
        "claim_boundary": {
            "diagnosis_execution_authorized": True,
            "maximum_new_trajectory_steps": 0,
            "bounded_crossing_retry_authorized": False,
            "fixed_Q_invariant_object_authorized": False,
            "slow_flux_atlas_authorized": False,
            "complete_cycle_execution_authorized": False,
        },
        "decision": {
            "pass": "authorize_definitions_only_same_horizon_crossing_retry_manifest",
            "failure": "stop_without_propagating_any_trial_endpoint",
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
            rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": utils._sha256(path), "scientific_status": "DEFINITIONS_ONLY"})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    catalog = utils._read_json(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": summary["classification"], "passed": summary["passed"]}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": utils._git("rev-parse", "HEAD"), "latest_work_package": WORK_PACKAGE})
    utils._write_json(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("substep diagnosis manifest already exists")
    utils = _utils(); validated = _validate_parent(require_clean=True)
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "diagnosis_contract.json", _contract())
    utils._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"parent_artifact": parent.ARTIFACT, "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256, "parent_hashes": validated["hashes"], "parent_metrics": validated["metrics"]})
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "bounded_crossing_rejection_preserved": True,
        "diagnosis_execution_authorized": True,
        "maximum_new_trajectory_steps": 0,
        "bounded_crossing_retry_authorized": False,
        "fixed_Q_invariant_object_authorized": False,
        "complete_cycle_execution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(("# Entropy-complete radial substep diagnosis manifest", "", f"Classification: `{CLASSIFICATION}`.", "", "The rejected coarse crossing remains binding. Three independent one-step trials test whether the unchanged chart-change gate resolves under causal-shear substepping; no trial endpoint may be propagated.", "", f"Authorized next: `{AUTHORIZED_NEXT}` only.", "")), encoding="utf-8")
    sources = (THIS_RUNNER, THIS_TEST, REPORT_RELATIVE)
    utils._write_json(CANONICAL_DIRECTORY / "provenance.json", {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "scientific_status": "DEFINITIONS_ONLY", "implementation_commit": utils._git("rev-parse", "HEAD"), "implementation_tree": utils._git("rev-parse", "HEAD^{tree}"), "source_hashes": {path: utils._sha256(ROOT / path) for path in sources}, "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "thread_environment": {name: os.environ.get(name, "") for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utils._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8")
    _update_catalog(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--freeze", action="store_true"); args = parser.parse_args()
    if not args.freeze: parser.error("choose --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
