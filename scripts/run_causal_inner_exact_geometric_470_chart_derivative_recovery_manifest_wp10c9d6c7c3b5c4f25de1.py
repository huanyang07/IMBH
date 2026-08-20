#!/usr/bin/env python3
"""Freeze a scale-aware derivative recovery audit for the exact 470 chart."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_exact_geometric_470_chart_preflight_wp10c9d6c7c3b5c4f25de as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25de1"
PARENT_COMMIT = "2dad3286e5418c9eb17a095df54ae49198268942"
PARENT_PARENT = "ecf6515262513ade45e9376d36779cde8467405f"
PARENT_TREE = "cb238f23b448e8e81de2417062b67a0404ce6374"

CLASSIFICATION = (
    "exact_geometric_470_chart_scale_aware_derivative_recovery_manifest_"
    "frozen_zero_truth"
)
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25de2"
PASS_AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25df"

COMMON_SCALE_STEP = 3.0e-3
ROUND_OFF_STEPS = (3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5, 1.0e-5)
FAILED_DIRECTION_INDEX = 6
FAILED_DIRECTION_FAMILY = "macro"
FAILED_DIRECTION_SOURCE_INDEX = 53

ALGEBRAIC_RELATIVE_DEFECT_GATE = 1.0e-10
COMMON_SCALE_RELATIVE_DEFECT_GATE = 1.0e-6
COMMON_SCALE_BEST_DEFECT_GATE = 1.0e-7
MINIMUM_COMMON_SIGNAL_NORM = 1.0e-8
ROUND_OFF_SLOPE_MIN = -1.10
ROUND_OFF_SLOPE_MAX = -0.90
ROUND_OFF_R_SQUARED_MIN = 0.99
ROUND_OFF_COEFFICIENT_OF_VARIATION_MAX = 0.10
ORIGINAL_DEFECT_REPRODUCTION_RELATIVE_GATE = 1.0e-9

ARTIFACT = (
    "causal_inner_exact_geometric_470_chart_derivative_recovery_manifest_"
    "wp10c9d6c7c3b5c4f25de1"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_exact_geometric_470_chart_derivative_recovery_"
    "manifest_wp10c9d6c7c3b5c4f25de1.py"
)
THIS_TEST = (
    "tests/test_causal_inner_exact_geometric_470_chart_derivative_recovery_"
    "manifest_wp10c9d6c7c3b5c4f25de1.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_EXACT_GEOMETRIC_470_CHART_"
    "DERIVATIVE_RECOVERY_MANIFEST_WP10C9D6C7C3B5C4F25DE1_2026-08-20.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _plain(value):
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, int):
        return value
    return value


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _checksums(directory: Path) -> dict[str, str]:
    recorded = {}
    for line in (directory / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        if _sha(directory / name) != expected:
            raise RuntimeError(f"checksum mismatch: {directory / name}")
        recorded[name] = expected
    return recorded


def _validate_parent(*, require_clean: bool) -> dict:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("derivative-recovery parent commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("derivative-recovery parent lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("derivative-recovery parent tree changed")
    hashes = _checksums(parent.CANONICAL_DIRECTORY)
    summary = _read(parent.CANONICAL_DIRECTORY / "summary.json")
    payload = _read(parent.CANONICAL_DIRECTORY / "exact_chart_metrics.json")
    provenance = _read(parent.CANONICAL_DIRECTORY / "provenance.json")
    checks = payload["checks"]
    if (
        summary["passed"]
        or summary["classification"] != parent.FAIL_CLASSIFICATION
        or summary["authorized_next"] is not None
        or summary["branch_root_execution_authorized"]
        or summary["sealed_16ms_opened"]
        or checks["implicit_derivative"]
        or any(
            not passed
            for name, passed in checks.items()
            if name != "implicit_derivative"
        )
        or payload["metrics"]["completed_retraction_count"]
        != parent.PLANNED_RETRACTIONS
        or payload["metrics"]["failed_retraction_count"] != 0
    ):
        raise RuntimeError("exact-chart negative certificate changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"exact-chart source changed: {relative}")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("derivative-recovery manifest requires a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "metrics": payload}


def _contract(parent_payload: dict) -> dict:
    original = parent_payload["metrics"]["implicit_derivative"]
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "preserved_negative_certificate": {
            "work_package": parent.WORK_PACKAGE,
            "classification": parent.FAIL_CLASSIFICATION,
            "remains_failed": True,
            "retroactive_gate_change": False,
            "hidden_root_remains_blocked_in_this_package": True,
            "original_step": original["step"],
            "original_maximum_relative_defect": original[
                "maximum_relative_defect"
            ],
            "only_failed_gate": "implicit_derivative",
        },
        "selection_disclosure": {
            "nonbinding_exploratory_read_only_ladder_informed_scale_range": True,
            "exploratory_values_are_not_binding_evidence": True,
            "all_binding_values_must_be_recomputed_after_this_manifest_commit": True,
            "no_exploratory_result_may_reclassify_the_parent": True,
        },
        "prospective_execution": {
            "work_package": AUTHORIZED_NEXT,
            "purpose": "scale_aware_exact_chart_derivative_recovery_only",
            "common_scale_step": COMMON_SCALE_STEP,
            "common_scale_direction_indices": list(range(parent.DIRECTION_COUNT)),
            "roundoff_ladder": list(ROUND_OFF_STEPS),
            "roundoff_direction": {
                "direction_index": FAILED_DIRECTION_INDEX,
                "family": FAILED_DIRECTION_FAMILY,
                "source_index": FAILED_DIRECTION_SOURCE_INDEX,
            },
            "algebraic_identity": (
                "A560_times_dx_equals_concatenate_dy470_and_zero90"
            ),
            "budgets": {
                "new_coordinate_evaluations_max": 2
                * (parent.DIRECTION_COUNT + len(ROUND_OFF_STEPS)),
                "new_coordinate_jacobian_assemblies_equal": 0,
                "new_coordinate_retractions_equal": 0,
                "new_exact_fixed_Q_rate_evaluations_equal": 0,
                "new_complete_generator_assemblies_equal": 0,
                "new_intrinsic_hidden_roots_equal": 0,
                "propagated_states_equal": 0,
                "sealed_16ms_truth_calls_equal": 0,
            },
        },
        "binding_gates": {
            "all_eight_algebraic_relative_defects_max": (
                ALGEBRAIC_RELATIVE_DEFECT_GATE
            ),
            "all_eight_common_scale_relative_defects_max": (
                COMMON_SCALE_RELATIVE_DEFECT_GATE
            ),
            "best_common_scale_relative_defect_max": (
                COMMON_SCALE_BEST_DEFECT_GATE
            ),
            "minimum_common_scale_signal_norm": MINIMUM_COMMON_SIGNAL_NORM,
            "roundoff_loglog_slope_min": ROUND_OFF_SLOPE_MIN,
            "roundoff_loglog_slope_max": ROUND_OFF_SLOPE_MAX,
            "roundoff_loglog_R_squared_min": ROUND_OFF_R_SQUARED_MIN,
            "roundoff_h_times_defect_coefficient_of_variation_max": (
                ROUND_OFF_COEFFICIENT_OF_VARIATION_MAX
            ),
            "original_step_defect_reproduction_relative_max": (
                ORIGINAL_DEFECT_REPRODUCTION_RELATIVE_GATE
            ),
            "all_parent_non_derivative_gates_remain_passed": True,
        },
        "decision": {
            "pass": {
                "classification": (
                    "exact_geometric_470_chart_derivative_recovered_"
                    "primary_hidden_root_manifest_authorized"
                ),
                "authorized_next": PASS_AUTHORIZED_NEXT,
                "authorizes_root_execution_directly": False,
            },
            "failure": {
                "classification": (
                    "exact_geometric_470_chart_derivative_recovery_failed_"
                    "hidden_root_blocked"
                ),
                "authorized_next": None,
            },
        },
        "authorization_boundaries": {
            "this_package_definitions_only": True,
            "chart_reclassified_in_this_package": False,
            "branch_root_execution_authorized": False,
            "sealed_16ms_opened": False,
            "online_solver_authorized": False,
            "predictive_cycle_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
        "decisive_input_hashes": {
            "parent_summary": _sha(parent.CANONICAL_DIRECTORY / "summary.json"),
            "parent_metrics": _sha(
                parent.CANONICAL_DIRECTORY / "exact_chart_metrics.json"
            ),
            "parent_arrays": _sha(
                parent.CANONICAL_DIRECTORY / "exact_chart_arrays.npz"
            ),
        },
    }


def _update_catalog(summary: dict) -> None:
    rows = []
    if CANONICAL_MANIFEST.exists():
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
                    "sha256": _sha(path),
                    "scientific_status": "DEFINITIONS_ONLY",
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case", "path", "bytes", "sha256", "scientific_status"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read(CANONICAL_SUMMARY)
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
            "latest_source_parent_commit": PARENT_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def _run() -> dict:
    frozen = _validate_parent(require_clean=True)
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("derivative-recovery manifest already exists")
    contract = _contract(frozen["metrics"])
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_json(CANONICAL_DIRECTORY / "derivative_recovery_contract.json", contract)
    _write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_commit": PARENT_COMMIT,
            "parent_parent": PARENT_PARENT,
            "parent_tree": PARENT_TREE,
            "parent_hashes": frozen["hashes"],
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "parent_negative_certificate_preserved": True,
        "new_coordinate_evaluations": 0,
        "new_exact_fixed_Q_rate_evaluations": 0,
        "new_complete_generator_assemblies": 0,
        "new_intrinsic_hidden_roots": 0,
        "propagated_states": 0,
        "sealed_16ms_opened": False,
        "branch_root_execution_authorized": False,
        "derivative_recovery_audit_authorized": True,
        "online_solver_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    source_files = (THIS_RUNNER, THIS_TEST, parent.THIS_RUNNER, parent.THIS_TEST)
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "DEFINITIONS_ONLY",
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "source_hashes": {
                relative: _sha(ROOT / relative) for relative in source_files
            },
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "thread_environment": {
                name: os.environ.get(name)
                for name in parent.manifest.parent.field_manifest.training._thread_environment()
            },
        },
    )
    names = tuple(sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()))
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Exact 470-chart derivative-recovery manifest WP10c9d6c7c3b5c4f25de1",
                "",
                "## Classification",
                "",
                f"`{CLASSIFICATION}`",
                "",
                "The f25de exact-chart preflight remains a binding rejection. Its 18 exact retractions and every physical, closure, rank, condition, trust, and budget gate passed; only the fixed h=1e-4 implicit-derivative audit failed.",
                "",
                "This definitions-only package freezes a replacement scale-aware audit. All eight algebraic identities and all eight finite-difference directions are binding at h=3e-3. The previously failed macro-53 direction also receives a six-step ladder whose 1/h roundoff signature must pass slope, R-squared, and coefficient-of-variation gates.",
                "",
                "An exploratory read-only ladder informed the scale range but is not evidence. Every binding value must be recomputed after this manifest is committed. No fixed-Q rate, generator, retraction, root, propagation, or 16 ms evaluation is permitted.",
                "",
                f"Authorized next artifact: `{AUTHORIZED_NEXT}`. A pass there may authorize only definitions for the primary hidden-root pilot `{PASS_AUTHORIZED_NEXT}`.",
                "",
            )
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("pass --run")
    print(json.dumps(_plain(_run()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
