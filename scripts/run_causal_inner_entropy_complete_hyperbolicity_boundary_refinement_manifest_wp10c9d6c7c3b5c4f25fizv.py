#!/usr/bin/env python3
"""Freeze a nonpropagating refinement diagnosis at the 179.5 ms boundary."""

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

import run_causal_inner_entropy_complete_adaptive_full_bundle_transient_recovery_execution_wp10c9d6c7c3b5c4f25fizu as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fizv_"
    "entropy_complete_hyperbolicity_boundary_refinement_manifest"
)
CLASSIFICATION = (
    "entropy_complete_179p5ms_hyperbolicity_boundary_refinement_manifest_frozen"
)
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizw_"
    "entropy_complete_hyperbolicity_boundary_refinement_diagnostic"
)
ARTIFACT = (
    "causal_inner_entropy_complete_hyperbolicity_boundary_refinement_manifest_"
    "wp10c9d6c7c3b5c4f25fizv"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_"
    "HYPERBOLICITY_BOUNDARY_REFINEMENT_MANIFEST_"
    "WP10C9D6C7C3B5C4F25FIZV_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_entropy_complete_hyperbolicity_boundary_"
    "refinement_manifest_wp10c9d6c7c3b5c4f25fizv.py"
)
THIS_TEST = (
    "tests/test_causal_inner_entropy_complete_hyperbolicity_boundary_"
    "refinement_manifest_wp10c9d6c7c3b5c4f25fizv.py"
)
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "c15377d5994abda65d5fa3dd7280b00e88920a9783f08e0b1ab1f5e31ea722b2"
)
PARENT_ARRAYS = parent.CANONICAL_DIRECTORY / "adaptive_arrays.npz"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _utils():
    return parent._utils()


def _contract() -> dict:
    probe_steps = (5.0e-4, 2.5e-4, 1.25e-4)
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "preserved_parent_rejection": {
            "classification": parent.FAIL_CLASSIFICATION,
            "accepted_horizon_seconds": 0.17900000000000005,
            "failed_candidate_elapsed_seconds": 0.17950000000000005,
            "failed_candidate_timestep_seconds": 5.0e-4,
            "failed_candidate_propagated": False,
            "physical_hyperbolicity_failure": True,
            "no_parent_gate_relaxed": True,
        },
        "diagnostic_scope": {
            "nonpropagating": True,
            "restart_from_hash_validated_179ms_checkpoint": True,
            "variable_AB2_formula_unchanged": True,
            "probe_timestep_seconds": probe_steps,
            "scan_timestep_seconds": tuple(np.linspace(0.0, 5.0e-4, 17)),
            "maximum_new_truth_operator_calls": len(probe_steps),
            "new_global_roots": 0,
            "fixed_Q_reaction_calls": 0,
        },
        "binding_gates": {
            "coarse_0p5ms_nonreal_face_reproduced": True,
            "accepted_179ms_endpoint_remains_hyperbolic": True,
            "refined_0p25ms_full_truth_operator_passes": True,
            "refined_0p125ms_full_truth_operator_passes": True,
            "all_existing_physical_gates_on_refined_probes": True,
            "maximum_reconstruction_chart_coordinate": 0.12,
            "maximum_macro_roundtrip_relative_defect": 1.0e-10,
            "maximum_face_imaginary_ratio_for_hyperbolic_probe": 1.0e-10,
        },
        "decision": {
            "coarse_fails_both_refined_pass": (
                "classify explicit_AB2_candidate_overshoot; authorize definitions-only "
                "event-aware hyperbolicity-retry recovery manifest"
            ),
            "refined_probe_fails": (
                "classify unresolved/genuine local hyperbolicity boundary and stop "
                "the seven-field cycle path"
            ),
            "coarse_does_not_reproduce": "method/reproducibility failure only",
            "no_probe_is_propagated": True,
        },
        "claim_boundary": {
            "diagnostic_only": True,
            "adaptive_recovery_reopened": False,
            "complete_cycle_execution_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
        "authorized_next": AUTHORIZED_NEXT,
    }


def _validate_parent(*, require_clean: bool) -> dict:
    utils = _utils()
    if (
        utils._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt")
        != PARENT_CHECKSUM_MANIFEST_SHA256
    ):
        raise RuntimeError("adaptive hyperbolicity rejection checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = utils._read_json(
        parent.CANONICAL_DIRECTORY / "adaptive_metrics.json"
    )
    failed = metrics["attempt_records"][-1]
    if (
        summary["classification"] != parent.FAIL_CLASSIFICATION
        or summary["passed"]
        or summary["accepted_absolute_horizon_seconds"]
        != 0.17900000000000005
        or summary["authorized_next"] is not None
        or metrics["stop_reason"] != "physical_hyperbolicity_truth_gate_failed"
        or not metrics["physical_failure"]
        or metrics["target_reached"]
        or metrics["accepted_new_steps"] != 24
        or metrics["new_truth_operator_calls"] != 25
        or failed["accepted"]
        or failed["retryable"]
        or not failed["physical_hyperbolicity_failure"]
        or failed["candidate_timestep_seconds"] != 5.0e-4
        or failed["candidate_absolute_elapsed_seconds"]
        != 0.17950000000000005
    ):
        raise RuntimeError("adaptive hyperbolicity rejection changed")
    for relative, expected in utils._read_json(
        parent.CANONICAL_DIRECTORY / "provenance.json"
    )["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"adaptive rejection source changed: {relative}")
    with np.load(PARENT_ARRAYS) as archive:
        if (
            archive["accepted_macro_states"].shape[0] < 2
            or archive["accepted_primitive_charts"].shape[0] < 2
            or float(archive["terminal_elapsed_seconds"][0])
            != 0.17900000000000005
            or float(archive["terminal_previous_timestep_seconds"][0])
            != 5.0e-4
        ):
            raise RuntimeError("adaptive rejection checkpoint changed")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("boundary refinement manifest needs a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "metrics": metrics}


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
                    "scientific_status": "SUPPORTED",
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
        "passed": True,
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
        raise RuntimeError("boundary refinement manifest already exists")
    validated = _validate_parent(require_clean=True)
    utils = _utils()
    contract = _contract()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "diagnostic_contract.json", contract)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "parent_rejection_preserved": True,
        "nonpropagating_refinement_diagnostic_authorized": True,
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
            "parent_arrays_sha256": utils._sha256(PARENT_ARRAYS),
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Entropy-complete hyperbolicity-boundary refinement manifest",
                "",
                f"Classification: `{CLASSIFICATION}`.",
                "",
                "The accepted 179 ms endpoint and the nonpropagated 179.5 ms face-hyperbolicity rejection are preserved. This package authorizes only three nonpropagating AB2 probes at 0.5, 0.25, and 0.125 ms plus a local face scan.",
                "",
                "A valid overshoot diagnosis requires the 0.5 ms failure to reproduce and both refined full-truth probes to pass every unchanged physical gate. No probe may enter accepted history.",
                "",
                "Complete-cycle execution and reduced slow evolution remain unauthorized.",
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
