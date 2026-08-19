#!/usr/bin/env python3
"""Freeze sparse recovery of the missing coordinate-Hessian KKT term."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_bordered_branch_homotopy_launch_wp10c9d6c7c3b5c4f25as as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25at"
CLASSIFICATION = (
    "coordinate_hessian_complete_KKT_diagnosis_manifest_frozen_"
    "zero_new_fixed_Q_rate_calls"
)
PARENT_COMMIT = "54a66b9ea4ebc4155f9a312871b190b5beee55b2"
PARENT_PARENT = "4ef594146efc0862b0f44fb4b03a024ea62bf03c"
PARENT_TREE = "632468d6a7341b44036672e695ea7386246db78b"

ARTIFACT = (
    "causal_inner_coordinate_hessian_diagnosis_manifest_"
    "wp10c9d6c7c3b5c4f25at"
)
ARTIFACT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_coordinate_hessian_diagnosis_manifest_"
    "wp10c9d6c7c3b5c4f25at.py"
)
THIS_TEST = (
    "tests/test_causal_inner_coordinate_hessian_diagnosis_manifest_"
    "wp10c9d6c7c3b5c4f25at.py"
)
NEXT_RUNNER = (
    "scripts/run_causal_inner_coordinate_hessian_diagnosis_"
    "wp10c9d6c7c3b5c4f25au.py"
)
NEXT_TEST = (
    "tests/test_causal_inner_coordinate_hessian_diagnosis_"
    "wp10c9d6c7c3b5c4f25au.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_COORDINATE_HESSIAN_DIAGNOSIS_"
    "MANIFEST_WP10C9D6C7C3B5C4F25AT_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

CELL_HALF_BANDWIDTH = 2
CELL_COLOR_COUNT = 2 * CELL_HALF_BANDWIDTH + 1
FIELD_COUNT = 5
COLORED_DIRECTION_COUNT = CELL_COLOR_COUNT * FIELD_COUNT
CENTRAL_DIFFERENCE_STEP = 1.0e-5
TAU_DIAGNOSTIC = 1.0 / 64.0


def _plain(value):
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


def _write(path: Path, payload) -> None:
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


def _validate_parent() -> dict:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("coordinate-Hessian parent commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("coordinate-Hessian parent lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("coordinate-Hessian parent tree changed")
    hashes = _checksums(parent.CANONICAL_DIRECTORY)
    summary = _read(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = _read(parent.CANONICAL_DIRECTORY / "metrics.json")
    if (
        summary["passed"]
        or summary["classification"]
        != "bordered_homotopy_launch_failed_conditional_branch_path_requires_diagnosis"
        or summary["rate_evaluations"] != 1
        or metrics["nonlinear"]["events"][0]["maximum_scaled_anchor_departure"]
        > 5.0e-3
    ):
        raise RuntimeError("homotopy-launch rejection changed")
    return {"summary": summary, "metrics": metrics, "hashes": hashes}


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "diagnosis": {
            "missing_complete_KKT_term": (
                "D_of_D_C_phys_x_transpose_mu0_with_respect_to_scaled_x"
            ),
            "mathematical_object": (
                "H_C_equals_Hessian_x_of_mu0_transpose_C_phys_x"
            ),
            "complete_stationarity_block": "A0_divided_by_omega_minus_H_C",
            "physical_interpretation": (
                "curvature_of_the_fixed_resolved_coordinate_fiber"
            ),
        },
        "sparse_recovery": {
            "truth_grid_cells": 112,
            "fields_per_cell": FIELD_COUNT,
            "cell_half_bandwidth": CELL_HALF_BANDWIDTH,
            "cell_color_count": CELL_COLOR_COUNT,
            "colored_direction_count": COLORED_DIRECTION_COUNT,
            "central_coordinate_jacobian_evaluations": 2 * COLORED_DIRECTION_COUNT,
            "central_difference_step": CENTRAL_DIFFERENCE_STEP,
            "coloring": "cell_index_modulo_5_cross_each_of_5_fields",
            "random_direction_validation_evaluations": 2,
            "new_fixed_Q_rate_evaluations": 0,
            "new_complete_fixed_Q_generator_evaluations": 0,
        },
        "binding_gates": {
            "recovered_Hessian_relative_symmetry_defect_max": 1.0e-5,
            "random_direction_action_relative_defect_max": 1.0e-4,
            "outside_declared_band_relative_leakage_max": 1.0e-10,
            "complete_equilibrated_KKT_condition_number_max": 1.0e6,
            "complete_tau_1_over_64_tangent_maximum_scaled_component_max": 5.0e-3,
            "complete_linear_solve_relative_residual_max": 1.0e-10,
        },
        "decision": {
            "pass": (
                "coordinate_hessian_complete_KKT_tangent_certified_"
                "definitions_only_corrected_homotopy_launch_manifest_authorized"
            ),
            "unsafe_tangent": (
                "complete_KKT_tangent_requires_smaller_prospective_tau_manifest"
            ),
            "recovery_fail": (
                "coordinate_hessian_recovery_failed_"
                "branch_solver_architecture_requires_revision"
            ),
        },
        "claim_boundary": {
            "new_physical_rate_evaluated": False,
            "failed_homotopy_candidate_reclassified": False,
            "physical_conditional_branch_found": False,
            "normal_hyperbolicity_certified": False,
            "reduced_slow_evolution_authorized": False,
        },
        "decisive_input_hashes": {
            "failed_launch_checkpoint": _sha(
                parent.CANONICAL_DIRECTORY / "homotopy_tau_1_over_64.npz"
            ),
            "failed_launch_metrics": _sha(
                parent.CANONICAL_DIRECTORY / "metrics.json"
            ),
            "preflight_diagnostics": _sha(parent.PREFLIGHT_ARRAYS),
        },
    }


def _update_catalog(summary: dict) -> None:
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(ARTIFACT_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha(path),
                    "scientific_status": "PROSPECTIVE",
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
        "path": str(ARTIFACT_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": True,
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
    _write(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    validated = _validate_parent()
    if _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("coordinate-Hessian manifest requires a clean tracked tree")
    if ARTIFACT_DIRECTORY.exists():
        raise RuntimeError("coordinate-Hessian manifest is already frozen")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "colored_direction_count": COLORED_DIRECTION_COUNT,
        "coordinate_jacobian_evaluations": 2 * COLORED_DIRECTION_COUNT + 2,
        "new_fixed_Q_rate_evaluations": 0,
        "physical_conditional_branch_found": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": "WP10c9d6c7c3b5c4f25au",
    }
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write(ARTIFACT_DIRECTORY / "contract.json", _contract())
    _write(ARTIFACT_DIRECTORY / "summary.json", summary)
    _write(
        ARTIFACT_DIRECTORY / "parent_lock.json",
        {
            "parent_commit": PARENT_COMMIT,
            "parent_parent": PARENT_PARENT,
            "parent_tree": PARENT_TREE,
            "parent_package_hashes": validated["hashes"],
        },
    )
    _write(
        ARTIFACT_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "PROSPECTIVE",
            "definition_commit": _git("rev-parse", "HEAD"),
            "definition_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "authorized_next_runner": NEXT_RUNNER,
            "authorized_next_test": NEXT_TEST,
            "report": REPORT_RELATIVE,
            "source_hashes": {
                THIS_RUNNER: _sha(ROOT / THIS_RUNNER),
                THIS_TEST: _sha(ROOT / THIS_TEST),
            },
            "python": sys.version,
            "platform": platform.platform(),
            "thread_environment": parent.preflight.THREAD_ENVIRONMENT,
        },
    )
    names = tuple(sorted(path.name for path in ARTIFACT_DIRECTORY.iterdir()))
    (ARTIFACT_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(ARTIFACT_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Coordinate-Hessian diagnosis manifest WP10c9d6c7c3b5c4f25at",
                "",
                "## Classification",
                "",
                f"`{CLASSIFICATION}`",
                "",
                "The failed tau=1/64 launch is preserved. The next diagnostic recovers the omitted curvature term Hessian_x(mu0^T C_phys) using 25 five-cell-stencil colors and central coordinate-Jacobian differences.",
                "",
                "The audit uses no new fixed-Q rate or complete-generator evaluation. It verifies symmetry, an independent random Hessian action, sparsity leakage, the complete equilibrated KKT condition number, and the corrected tau=1/64 tangent trust bound.",
                "",
                "A pass authorizes only a definitions-only corrected homotopy launch. No physical branch or reduced evolution is authorized.",
                "",
            )
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if not args.freeze:
        raise SystemExit("pass --freeze")
    print(json.dumps(_plain(_freeze()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
