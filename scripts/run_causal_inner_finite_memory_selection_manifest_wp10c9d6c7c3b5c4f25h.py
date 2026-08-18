#!/usr/bin/env python3
"""Freeze the single-anchor stable finite-memory model-selection audit."""

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

SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25h"
CLASSIFICATION = "finite_memory_selection_manifest_frozen_balanced_screen_authorized"
PARENT_COMMIT = "2031ef9"
PARENT_ARTIFACT = "causal_inner_resolved_mode_promotion_audit_wp10c9d6c7c3b5c4f25g"
PARENT_DIRECTORY = ROOT / "results/canonical" / PARENT_ARTIFACT
ARTIFACT = "causal_inner_finite_memory_selection_manifest_wp10c9d6c7c3b5c4f25h"
ARTIFACT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = "scripts/run_causal_inner_finite_memory_selection_manifest_wp10c9d6c7c3b5c4f25h.py"
THIS_TEST = "tests/test_causal_inner_finite_memory_selection_manifest_wp10c9d6c7c3b5c4f25h.py"
NEXT_RUNNER = "scripts/run_causal_inner_finite_memory_selection_audit_wp10c9d6c7c3b5c4f25i.py"
NEXT_TEST = "tests/test_causal_inner_finite_memory_selection_audit_wp10c9d6c7c3b5c4f25i.py"
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_FINITE_MEMORY_SELECTION_"
    "MANIFEST_WP10C9D6C7C3B5C4F25H_2026-08-18.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

MEMORY_ORDERS = (0, 2, 4, 6)
BASE_RESOLVED_DIMENSION = 106
MAXIMUM_ONLINE_CONTINUOUS_DIMENSION = 114


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
    path.write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        actual = _sha(directory / name)
        if actual != expected:
            raise RuntimeError(f"checksum mismatch: {directory / name}")
        recorded[name] = actual
    return recorded


def _validate_parent() -> tuple[dict, dict, dict[str, str]]:
    parent_full = _git("rev-parse", PARENT_COMMIT)
    if not parent_full.startswith(PARENT_COMMIT):
        raise RuntimeError("parent promotion certificate commit changed")
    hashes = _checksums(PARENT_DIRECTORY)
    summary = _read(PARENT_DIRECTORY / "summary.json")
    metrics = _read(PARENT_DIRECTORY / "metrics.json")
    if (
        not summary["passed"]
        or summary["classification"]
        != "resolved_mode_promotion_passed_stable_memory_manifest_authorized"
        or summary["authorized_next"]
        != "definitions_only_mode_selection_and_finite_memory_manifest"
        or summary["augmented_resolved_dimension"] != BASE_RESOLVED_DIMENSION
        or metrics["remaining_unresolved_spectral_abscissa_per_second"] >= 0.0
        or metrics["memory_coefficients_fit"] != 0
    ):
        raise RuntimeError("parent finite-memory authorization changed")
    return summary, metrics, hashes


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "parent_decisive_hashes": {
            name: _sha(PARENT_DIRECTORY / name)
            for name in ("promotion.npz", "transfer_real.npz", "transfer_imag.npz", "metrics.json")
        },
        "execution_budget": {
            "allowed_new_nonlinear_roots": 0,
            "allowed_propagated_states": 0,
            "allowed_new_full_generator_assemblies": 0,
            "allowed_single_anchor_linear_memory_fits": len(MEMORY_ORDERS),
            "allowed_truth_anchors": 0,
            "maximum_wall_hours": 2.0,
        },
        "normalized_stable_system": {
            "source": "hash_locked_R106_promotion_and_454_dimensional_stable_block",
            "input_scaling": "unit_two_norm_of_stacked_stable_forcing_and_direct_column",
            "output_scaling": "unit_two_norm_of_concatenated_stable_observation_and_direct_row",
            "zero_scale_fallback": 1.0,
            "same_frozen_scaling_for_all_candidate_orders": True,
            "DC_and_32_log_frequency_samples_reused": True,
        },
        "balanced_truncation": {
            "candidate_memory_orders": MEMORY_ORDERS,
            "selection_rule": "smallest_candidate_passing_every_binding_gate",
            "post_result_candidate_addition_forbidden": True,
            "continuous_lyapunov_gramians": True,
            "square_root_balancing": True,
            "direct_term_preserved_exactly": True,
            "full_stable_transfer_is_the_single_anchor_reference": True,
            "base_resolved_dimension": BASE_RESOLVED_DIMENSION,
            "maximum_online_continuous_dimension": MAXIMUM_ONLINE_CONTINUOUS_DIMENSION,
        },
        "candidate_pass_requires": {
            "reduced_spectral_abscissa_per_second_max": -1.0e-8,
            "lyapunov_dissipation_residual_max": 1.0e-9,
            "lyapunov_certificate_minimum_eigenvalue_min": 1.0e-14,
            "maximum_normalized_dynamic_transfer_relative_error_max": 0.25,
            "RMS_normalized_dynamic_transfer_relative_error_max": 0.10,
            "DC_normalized_dynamic_transfer_relative_error_max": 0.10,
            "maximum_normalized_total_transfer_relative_error_max": 0.10,
            "database_roundtrip_bitwise": True,
        },
        "full_order_numerical_pass_requires": {
            "controllability_gramian_relative_residual_max": 1.0e-8,
            "observability_gramian_relative_residual_max": 1.0e-8,
            "minimum_retained_hankel_singular_value_positive": True,
            "frequency_reference_roundtrip_bitwise": True,
        },
        "decision": {
            "one_or_more_candidates_pass": (
                "single_anchor_finite_memory_order_selected_cross_anchor_manifest_authorized"
            ),
            "no_candidate_passes": (
                "compact_finite_memory_failed_larger_conservative_coarse_PDE_fallback_required"
            ),
            "full_order_numerical_failure": "finite_memory_balancing_audit_failed_stop",
        },
        "claim_boundary": {
            "selected_coefficients_are_single_anchor_diagnostic_only": True,
            "production_memory_coefficients_authorized": False,
            "cross_anchor_database_authorized_only_after_pass": True,
            "online_solver_authorized": False,
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
        "passed": summary["passed"],
    }
    catalog.update({
        "case_count": len({row["case"] for row in rows}),
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "all_payload_hashes_recorded": True,
        "latest_source_parent_commit": _git("rev-parse", PARENT_COMMIT),
        "latest_work_package": WORK_PACKAGE,
    })
    _write(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    parent_summary, parent_metrics, parent_hashes = _validate_parent()
    if not _tracked_tree_clean():
        raise RuntimeError("finite-memory manifest requires a clean tracked tree")
    if ARTIFACT_DIRECTORY.exists():
        raise RuntimeError("finite-memory manifest is already frozen")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "candidate_memory_orders": MEMORY_ORDERS,
        "base_resolved_dimension": BASE_RESOLVED_DIMENSION,
        "maximum_candidate_online_dimension": BASE_RESOLVED_DIMENSION + max(MEMORY_ORDERS),
        "balanced_truncation_screen_authorized": True,
        "new_truth_trajectory_executed": False,
        "new_fixed_Q_root_executed": False,
        "new_generator_assembly_executed": False,
        "memory_fit_executed": False,
        "production_memory_coefficients_authorized": False,
        "online_reduced_solver_implementation_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "physical_failure_detected": False,
        "parent_classification_preserved": parent_summary["classification"],
        "parent_remaining_spectral_abscissa_per_second": parent_metrics[
            "remaining_unresolved_spectral_abscissa_per_second"
        ],
        "authorized_next": WORK_PACKAGE.replace("25h", "25i"),
    }
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write(ARTIFACT_DIRECTORY / "contract.json", _contract())
    _write(ARTIFACT_DIRECTORY / "summary.json", summary)
    _write(ARTIFACT_DIRECTORY / "parent_lock.json", {
        "parent_commit": _git("rev-parse", PARENT_COMMIT),
        "parent_tree": _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}"),
        "parent_package_hashes": parent_hashes,
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
        "source_hashes": {THIS_RUNNER: _sha(ROOT / THIS_RUNNER), THIS_TEST: _sha(ROOT / THIS_TEST)},
        "python": sys.version,
        "platform": platform.platform(),
        "thread_environment": {
            name: os.environ.get(name, "")
            for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
        },
    })
    names = tuple(sorted(path.name for path in ARTIFACT_DIRECTORY.iterdir()))
    (ARTIFACT_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(ARTIFACT_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        "\n".join((
            "# Finite-memory selection manifest WP10c9d6c7c3b5c4f25h",
            "",
            "## Classification",
            "",
            f"`{CLASSIFICATION}`",
            "",
            "This definitions-only package screens stable balanced-truncation memory orders 0/2/4/6 on the hash-locked single-anchor R106 decomposition. The largest online candidate is R112, below the R114 ceiling.",
            "",
            "The smallest candidate passing all predeclared total-transfer, dynamic-transfer, DC, stability, Lyapunov-dissipation, and bitwise-roundtrip gates is selected. The direct term is preserved exactly. If no order passes, the compact-kernel route stops and the previously declared larger conservative coarse-PDE fallback is required.",
            "",
            "No truth root, propagated state, new generator assembly, production coefficient fit, online solver, predictive cycle, or reduced slow evolution is authorized.",
            "",
        )),
        encoding="utf-8",
    )
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
