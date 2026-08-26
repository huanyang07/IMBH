#!/usr/bin/env python3
"""Freeze adaptive-step recovery of the accepted full-bundle transient."""

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

import run_causal_inner_entropy_complete_bounded_full_bundle_transient_acquisition_execution_wp10c9d6c7c3b5c4f25fizs as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fizt_"
    "entropy_complete_adaptive_full_bundle_transient_recovery_manifest"
)
CLASSIFICATION = (
    "entropy_complete_fixed_step_trust_rejection_preserved_adaptive_transient_"
    "recovery_manifest_frozen"
)
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizu_"
    "entropy_complete_adaptive_full_bundle_transient_recovery_execution"
)
ARTIFACT = (
    "causal_inner_entropy_complete_adaptive_full_bundle_transient_recovery_"
    "manifest_wp10c9d6c7c3b5c4f25fizt"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_ADAPTIVE_FULL_"
    "BUNDLE_TRANSIENT_RECOVERY_MANIFEST_"
    "WP10C9D6C7C3B5C4F25FIZT_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_entropy_complete_adaptive_full_bundle_transient_"
    "recovery_manifest_wp10c9d6c7c3b5c4f25fizt.py"
)
THIS_TEST = (
    "tests/test_causal_inner_entropy_complete_adaptive_full_bundle_transient_"
    "recovery_manifest_wp10c9d6c7c3b5c4f25fizt.py"
)
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "0ec4d7824f408d8ccc83d8df2f2c358aeff7a436b82cc7aec80c3e5e3d920224"
)
PARENT_ARRAYS = parent.CANONICAL_DIRECTORY / "transient_arrays.npz"
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
        raise RuntimeError("fixed-step rejection checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = utils._read_json(parent.CANONICAL_DIRECTORY / "transient_metrics.json")
    if (
        summary["classification"] != parent.FAIL_CLASSIFICATION
        or summary["passed"]
        or summary["accepted_steps"] != 36
        or summary["accepted_absolute_horizon_seconds"]
        != 0.15600000000000003
        or summary["complete_cycle_execution_authorized"]
        or metrics["failure_reason"]
        != "candidate_reconstruction_or_truth_evaluation_failed"
        or metrics["attempted_steps"] != 37
        or metrics["new_truth_operator_calls"] != 36
        or metrics["step_records"][-1]["step"] != 37
        or "reconstruction line search failed"
        not in metrics["step_records"][-1]["exception"]
        or not all(record["passed"] for record in metrics["step_records"][:-1])
    ):
        raise RuntimeError("fixed-step rejection classification changed")
    for relative, expected in utils._read_json(
        parent.CANONICAL_DIRECTORY / "provenance.json"
    )["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"fixed-step execution source changed: {relative}")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("adaptive recovery manifest needs a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "metrics": metrics}


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "authorized_next": AUTHORIZED_NEXT,
        "preserved_rejection": {
            "fixed_4ms_step_37_rejected": True,
            "accepted_horizon_seconds": 0.15600000000000003,
            "failed_candidate_propagated": False,
            "all_36_accepted_steps_physical": True,
            "failure_was_reconstruction_trust_not_physical": True,
            "no_gate_relaxed": True,
        },
        "restart": {
            "seed": "hash-validated final two accepted states, rates, outputs, and terminal primitive charts",
            "previous_timestep_seconds": 0.004,
            "restart_roundtrip_bitwise_required": True,
            "accepted_history_only": True,
        },
        "adaptive_AB2": {
            "variable_step_formula": (
                "X_next=X+h*((1+r/2)*F_n-(r/2)*F_nm1), r=h/h_previous"
            ),
            "initial_timestep_seconds": 0.002,
            "minimum_timestep_seconds": 0.000125,
            "maximum_timestep_seconds": 0.004,
            "shrink_factor": 0.5,
            "growth_factor": 2.0,
            "growth_requires_consecutive_low_defect_steps": 4,
            "growth_chart_coordinate_maximum": 0.06,
            "growth_embedded_defect_maximum": 0.0025,
            "pretruth_reconstruction_failure_is_retryable": True,
            "posttruth_numerical_failure_is_retryable": True,
            "physical_failure_is_not_retryable": True,
        },
        "bounded_execution": {
            "initial_absolute_elapsed_seconds": 0.15600000000000003,
            "target_absolute_elapsed_seconds": 0.212,
            "maximum_attempted_steps": 160,
            "maximum_new_truth_operator_calls": 128,
            "new_global_roots": 0,
            "fixed_Q_reaction_calls": 0,
            "stop_exactly_at_target_horizon": True,
        },
        "binding_gates": {
            "reserved_reconstruction_chart_coordinate": 0.12,
            "maximum_macro_roundtrip_relative_defect": 1.0e-10,
            "maximum_AB2_trapezoidal_embedded_defect": 0.01,
            "maximum_discrete_conservative_ledger_relative_defect": 1.0e-12,
            "all_existing_height_optical_depth_causality_hyperbolicity_and_excision_gates": True,
            "minimum_timestep_failure_is_binding": True,
            "truth_call_budget_failure_is_inconclusive_not_a_physical_failure": True,
        },
        "slaving_observation": {
            "normalized_auxiliary_to_conservative_rate_ratio_maximum": 0.1,
            "normalized_auxiliary_rate_infinity_per_second_maximum": 0.1,
            "required_consecutive_accepted_steps": 8,
            "fresh_normal_attraction_tangent_still_required": True,
            "no_switch_during_this_execution": True,
        },
        "decision": {
            "target_reached_with_persistent_slaving": (
                "authorize definitions-only terminal fast-graph tangent manifest"
            ),
            "target_reached_without_slaving": (
                "authorize definitions-only transient-geometry and cost decision manifest"
            ),
            "budget_exhausted": "classify adaptive transient recovery inconclusive",
            "physical_failure": "stop the seven-field cycle path",
            "no_retrospective_gate_change": True,
        },
        "claim_boundary": {
            "one_adaptive_recovery_execution_authorized": True,
            "complete_cycle_execution_authorized": False,
            "48_coordinate_cycle_solver_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
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
        raise RuntimeError("adaptive recovery manifest already exists")
    validated = _validate_parent(require_clean=True)
    utils = _utils()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(
        CANONICAL_DIRECTORY / "adaptive_recovery_contract.json", _contract()
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "fixed_step_rejection_preserved": True,
        "adaptive_recovery_execution_authorized": True,
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utils._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_artifact": parent.ARTIFACT,
            "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256,
            "parent_hashes": validated["hashes"],
            "parent_arrays_sha256": utils._sha256(PARENT_ARRAYS),
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Entropy-complete adaptive full-bundle transient recovery manifest",
                "",
                f"Classification: `{CLASSIFICATION}`.",
                "",
                "The fixed 4 ms policy remains rejected at step 37 after 36 physical accepted steps through 156 ms. The failure was a pre-truth local reconstruction trust failure; no physical candidate failed and no rejected state entered history.",
                "",
                "The next package restarts from the exact accepted two-step history, begins at 2 ms, and halves prospectively on reconstruction or numerical failure. Physical failures remain nonretryable. The target is 212 ms under a 128-truth-call cap.",
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
