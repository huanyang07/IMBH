#!/usr/bin/env python3
"""Freeze the corrected same-horizon seven-field radial crossing."""

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

import run_causal_inner_entropy_complete_radial_substep_diagnosis_execution_wp10c9d6c7c3b5c4f25fizem as parent  # noqa: E402
import run_causal_inner_entropy_complete_bounded_radial_crossing_manifest_wp10c9d6c7c3b5c4f25fizej as original  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fizen_"
    "entropy_complete_corrected_bounded_radial_crossing_manifest"
)
CLASSIFICATION = "entropy_complete_corrected_bounded_radial_crossing_manifest_frozen"
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizeo_"
    "entropy_complete_corrected_bounded_radial_crossing_execution"
)
ARTIFACT = (
    "causal_inner_entropy_complete_corrected_bounded_radial_crossing_manifest_"
    "wp10c9d6c7c3b5c4f25fizen"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_CORRECTED_"
    "BOUNDED_RADIAL_CROSSING_MANIFEST_WP10C9D6C7C3B5C4F25FIZEN_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = "scripts/run_causal_inner_entropy_complete_corrected_bounded_radial_crossing_manifest_wp10c9d6c7c3b5c4f25fizen.py"
THIS_TEST = "tests/test_causal_inner_entropy_complete_corrected_bounded_radial_crossing_manifest_wp10c9d6c7c3b5c4f25fizen.py"
PARENT_CHECKSUM_MANIFEST_SHA256 = "25631a66552b5c91d19accb9c9c62bb492f7724874d61b95c54a719f66a76463"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
TIMESTEP_SECONDS = 7.8125e-6
ACCEPTED_STEPS = 32
HORIZON_SECONDS = TIMESTEP_SECONDS * ACCEPTED_STEPS
REPLAY_SUFFIX_STEPS = 4
REPLAY_CHECKPOINT_STEP = ACCEPTED_STEPS - REPLAY_SUFFIX_STEPS


def _utils():
    return parent._utils()


def _validate_parent(*, require_clean: bool) -> dict:
    utils = _utils()
    checksum = parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt"
    if utils._sha256(checksum) != PARENT_CHECKSUM_MANIFEST_SHA256:
        raise RuntimeError("radial substep certificate checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = utils._read_json(parent.CANONICAL_DIRECTORY / "execution_metrics.json")
    if (
        summary["classification"] != parent.PASS_CLASSIFICATION
        or not summary["passed"]
        or not summary["radial_shear_substep_certified"]
        or summary["selected_timestep_seconds"] != TIMESTEP_SECONDS
        or not summary["bounded_crossing_retry_manifest_authorized"]
        or summary["bounded_crossing_retry_execution_authorized"]
        or summary["authorized_next"] != (
            "definitions_only_WP10c9d6c7c3b5c4f25fizen_"
            "entropy_complete_corrected_bounded_radial_crossing_manifest"
        )
        or metrics["new_trajectory_steps"] != 0
        or metrics["trial_endpoints_propagated"]
        or metrics["failure_reasons"]
    ):
        raise RuntimeError("radial substep authorization changed")
    for relative, expected in utils._read_json(
        parent.CANONICAL_DIRECTORY / "provenance.json"
    )["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"radial substep source changed: {relative}")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("corrected crossing manifest requires clean tracked tree")
    return {"hashes": hashes, "summary": summary, "metrics": metrics}


def _contract() -> dict:
    original_contract = original._contract()
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "preserved_results": {
            "coarse_crossing_rejection_preserved": True,
            "shear_substep_certificate_preserved": True,
            "physical_equations_unchanged": True,
            "spatial_operator_unchanged": True,
            "acceptance_gates_unchanged": True,
        },
        "seed": original_contract["seed"],
        "radial_operator": original_contract["radial_operator"],
        "time_integrator": {
            "method": "explicit_SSPRK2_in_seven_primitive_chart",
            "timestep_seconds": TIMESTEP_SECONDS,
            "accepted_steps": ACCEPTED_STEPS,
            "horizon_seconds": HORIZON_SECONDS,
            "same_horizon_as_rejected_coarse_crossing": HORIZON_SECONDS == original.HORIZON_SECONDS,
            "first_endpoint_must_match_diagnostic_bitwise": True,
            "checkpoint_each_endpoint": True,
            "matched_control": "one_1.5625e-5_s_step_vs_two_7.8125e-6_s_steps_from_seed",
            "replay_checkpoint_step": REPLAY_CHECKPOINT_STEP,
            "replay_suffix_steps": REPLAY_SUFFIX_STEPS,
            "suffix_replay_bitwise": True,
            "stagewise_fail_closed_audits": True,
        },
        "binding_gates": original_contract["binding_gates"],
        "additional_gates": {
            "first_endpoint_matches_substep_certificate_bitwise": True,
            "cross_old_rejected_time": True,
            "accepted_terminal_elapsed_seconds_minimum": 0.186125,
            "checkpoint_count": ACCEPTED_STEPS + 1,
            "suffix_replay_bitwise": True,
            "maximum_cumulative_exact_flux_balance_relative_defect": 5.0e-5,
            "fail_closed": True,
        },
        "claim_boundary": {
            "corrected_crossing_execution_authorized": True,
            "maximum_new_trajectory_steps": ACCEPTED_STEPS,
            "fixed_Q_invariant_object_authorized": False,
            "slow_flux_atlas_authorized": False,
            "complete_cycle_execution_authorized": False,
        },
        "decision": {
            "pass": "authorize_definitions_only_fixed_Q_invariant_object_manifest",
            "failure": "stop_without_propagating_failed_candidate",
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
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists(): raise RuntimeError("corrected crossing manifest already exists")
    utils = _utils(); validated = _validate_parent(require_clean=True); CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "crossing_contract.json", _contract()); utils._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"parent_artifact": parent.ARTIFACT, "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256, "parent_hashes": validated["hashes"], "selected_substep_trial": validated["metrics"]["trials"][2]})
    summary = {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "classification": CLASSIFICATION, "passed": True, "definitions_only": True, "coarse_crossing_rejection_preserved": True, "radial_shear_substep_certificate_preserved": True, "corrected_crossing_execution_authorized": True, "maximum_new_trajectory_steps": ACCEPTED_STEPS, "new_trajectory_steps": 0, "fixed_Q_invariant_object_authorized": False, "complete_cycle_execution_authorized": False, "authorized_next": AUTHORIZED_NEXT}
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True); REPORT_PATH.write_text("\n".join(("# Entropy-complete corrected bounded radial crossing manifest", "", f"Classification: `{CLASSIFICATION}`.", "", f"The sole authorized trajectory is `{ACCEPTED_STEPS}` SSPRK2 steps of `{TIMESTEP_SECONDS}` s, preserving the rejected coarse result and the same `{HORIZON_SECONDS}` s horizon.", "", f"Authorized next: `{AUTHORIZED_NEXT}` only.", "")), encoding="utf-8")
    sources = (THIS_RUNNER, THIS_TEST, REPORT_RELATIVE); utils._write_json(CANONICAL_DIRECTORY / "provenance.json", {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "scientific_status": "DEFINITIONS_ONLY", "implementation_commit": utils._git("rev-parse", "HEAD"), "implementation_tree": utils._git("rev-parse", "HEAD^{tree}"), "source_hashes": {path: utils._sha256(ROOT / path) for path in sources}, "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "thread_environment": {name: os.environ.get(name, "") for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()); (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utils._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8"); _update_catalog(summary); return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--freeze", action="store_true"); args = parser.parse_args()
    if not args.freeze: parser.error("choose --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
