#!/usr/bin/env python3
"""Freeze the effective-real-rank correction for the exact spectral fiber."""

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


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts",):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_unstable_exact_conservative_fiber_manifest_wp10c9d6c7c3b5c4f25t as original_manifest  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25v"
CLASSIFICATION = (
    "effective_real_rank_correction_manifest_frozen_"
    "saved_generator_reaudit_authorized"
)
PARENT_COMMIT = "720132211c788df2e0e9b1898f9f8920e1e77928"
PARENT_PARENT = "f5fd2deea60808f18074bd621731ed4a5ba8245b"
PARENT_TREE = "9295c8a1726f95aa12953dca1a521372b0fa4bb6"

PARENT_ARTIFACT = (
    "causal_inner_unstable_exact_conservative_fiber_audit_"
    "wp10c9d6c7c3b5c4f25u"
)
PARENT_DIRECTORY = ROOT / "results/canonical" / PARENT_ARTIFACT
ARTIFACT = (
    "causal_inner_effective_real_rank_manifest_wp10c9d6c7c3b5c4f25v"
)
ARTIFACT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_effective_real_rank_manifest_"
    "wp10c9d6c7c3b5c4f25v.py"
)
THIS_TEST = (
    "tests/test_causal_inner_effective_real_rank_manifest_"
    "wp10c9d6c7c3b5c4f25v.py"
)
NEXT_RUNNER = (
    "scripts/run_causal_inner_effective_real_rank_audit_"
    "wp10c9d6c7c3b5c4f25w.py"
)
NEXT_TEST = (
    "tests/test_causal_inner_effective_real_rank_audit_"
    "wp10c9d6c7c3b5c4f25w.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_EFFECTIVE_REAL_RANK_MANIFEST_"
    "WP10C9D6C7C3B5C4F25V_2026-08-18.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

EXPECTED_RANK = 28
EFFECTIVE_RANK_RELATIVE_CUTOFF = 5.0e-10
RETAINED_SINGULAR_VALUE_RELATIVE_FLOOR = 1.0e-6
DISCARDED_TO_RETAINED_GAP_MAX = 5.0e-10


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


def _tracked_tree_clean() -> bool:
    return not _git("status", "--short", "--untracked-files=no")


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


def _validate_parent() -> tuple[dict, dict[str, str]]:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("effective-rank parent commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("effective-rank parent lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("effective-rank parent tree changed")
    hashes = _checksums(PARENT_DIRECTORY)
    summary = _read(PARENT_DIRECTORY / "summary.json")
    metrics = _read(PARENT_DIRECTORY / "metrics.json")
    if (
        summary["passed"]
        or not summary["numerical_passed"]
        or summary["classification"]
        != "unstable_exact_conservative_fiber_failed_reduced_architecture_reassessment_required"
        or summary["physical_failure_detected"]
        or summary["authorized_next"]
        != "definitions_only_unstable_exact_architecture_reassessment_manifest"
        or not metrics["cross_anchor_passed"]
    ):
        raise RuntimeError("effective-rank correction authorization changed")
    return summary, hashes


def _contract() -> dict:
    original = _read(original_manifest.ARTIFACT_DIRECTORY / "contract.json")
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "parent_decisive_hashes": {
            name: _sha(PARENT_DIRECTORY / name)
            for name in ("summary.json", "metrics.json", "decisive_fibers.npz")
        },
        "execution_budget": {
            "allowed_new_nonlinear_roots": 0,
            "allowed_propagated_states": 0,
            "allowed_new_full_560_direction_generator_assemblies": 0,
            "allowed_new_truth_anchors": 0,
            "maximum_wall_minutes": 20.0,
        },
        "effective_real_rank": {
            "expected_rank": EXPECTED_RANK,
            "definition": "count_singular_values_with_sigma_over_sigma_1_strictly_above_cutoff",
            "relative_cutoff": EFFECTIVE_RANK_RELATIVE_CUTOFF,
            "retained_singular_value_relative_floor": RETAINED_SINGULAR_VALUE_RELATIVE_FLOOR,
            "first_discarded_to_last_retained_ratio_max": DISCARDED_TO_RETAINED_GAP_MAX,
            "same_cutoff_for_left_and_right_at_both_anchors": True,
            "machine_epsilon_rank_is_nonbinding_diagnostic": True,
        },
        "unchanged_binding_gates": {
            key: value
            for key, value in original["binding_gates"].items()
            if key not in ("selected_nonstable_dimension_equal",)
        },
        "unchanged_architecture": original["prospective_online_state_if_passed"],
        "decisions": {
            "effective_rank_and_all_unchanged_gates_pass": (
                "two_anchor_effective_rank_unstable_exact_fiber_passed_"
                "constrained_lyapunov_stable_reduction_manifest_authorized"
            ),
            "any_binding_gate_fails": (
                "effective_rank_unstable_exact_fiber_failed_"
                "reduced_architecture_reassessment_required"
            ),
            "numerical_integrity_fails": (
                "effective_rank_unstable_exact_fiber_numerical_failure_stop"
            ),
        },
        "claim_boundary": {
            "online_integrator_implementation_authorized": False,
            "predictive_cycle_authorized": False,
            "reduced_slow_evolution_authorized": False,
            "physical_failure_can_be_declared": False,
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
            rows.append({
                "case": ARTIFACT,
                "path": str(path.relative_to(ROOT)),
                "bytes": str(path.stat().st_size),
                "sha256": _sha(path),
                "scientific_status": "PROSPECTIVE",
            })
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
    catalog.update({
        "case_count": len({row["case"] for row in rows}),
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "all_payload_hashes_recorded": True,
        "latest_source_parent_commit": PARENT_COMMIT,
        "latest_work_package": WORK_PACKAGE,
    })
    _write(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    parent_summary, parent_hashes = _validate_parent()
    if not _tracked_tree_clean():
        raise RuntimeError("effective-rank manifest requires a clean tracked tree")
    if ARTIFACT_DIRECTORY.exists():
        raise RuntimeError("effective-rank manifest is already frozen")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "parent_classification_preserved": parent_summary["classification"],
        "expected_effective_real_rank": EXPECTED_RANK,
        "effective_rank_relative_cutoff": EFFECTIVE_RANK_RELATIVE_CUTOFF,
        "new_truth_trajectory_executed": False,
        "new_fixed_Q_root_executed": False,
        "new_generator_assembly_executed": False,
        "online_integrator_implementation_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "physical_failure_detected": False,
        "authorized_next": WORK_PACKAGE.replace("25v", "25w"),
    }
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write(ARTIFACT_DIRECTORY / "contract.json", _contract())
    _write(ARTIFACT_DIRECTORY / "summary.json", summary)
    _write(ARTIFACT_DIRECTORY / "parent_lock.json", {
        "parent_commit": PARENT_COMMIT,
        "parent_parent": PARENT_PARENT,
        "parent_tree": PARENT_TREE,
        "parent_package_hashes": parent_hashes,
        "original_manifest_package_hashes": _checksums(original_manifest.ARTIFACT_DIRECTORY),
    })
    _write(ARTIFACT_DIRECTORY / "provenance.json", {
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
        "thread_environment": {
            name: os.environ.get(name, "")
            for name in (
                "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
    })
    names = tuple(sorted(path.name for path in ARTIFACT_DIRECTORY.iterdir()))
    (ARTIFACT_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(ARTIFACT_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    REPORT_PATH.write_text("\n".join((
        "# Effective real-rank correction manifest WP10c9d6c7c3b5c4f25v",
        "",
        "## Classification",
        "",
        f"`{CLASSIFICATION}`",
        "",
        "The f25u rejection is preserved. This prospective correction replaces only the generic machine-epsilon rank diagnostic with an explicit effective-rank cutoff tied to the already frozen realification tolerance.",
        "",
        "Effective rank 28 requires sigma_28/sigma_1 >= 1e-6 and sigma_29/sigma_28 <= 5e-10 at both anchors for both left and right fibers. Every other spectral, projector, stability, conservation, alignment, and R320 gate is unchanged.",
        "",
        "No truth assembly, nonlinear root, propagation, online integrator, or predictive cycle is authorized.",
        "",
    )), encoding="utf-8")
    _update_catalog(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    arguments = parser.parse_args()
    if not arguments.freeze:
        raise SystemExit("pass --freeze")
    print(json.dumps(_plain(_freeze()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
