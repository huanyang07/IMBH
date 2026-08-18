#!/usr/bin/env python3
"""Freeze the ordered-real-Schur resolved-mode promotion audit."""

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
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25f"
CLASSIFICATION = "resolved_mode_promotion_manifest_frozen_saved_generator_audit_authorized"
PARENT_COMMIT = "c96a62de0382d0d94367ba26b7e751127610d949"
PARENT_PARENT = "ee037367c4ffdca0da1c5334d435c5247370ed10"
PARENT_TREE = "ded17f5aff5f6ef910fc3722e1cbbac8f05f190b"

PARENT_ARTIFACT = "causal_inner_invariant_projection_spectrum_audit_wp10c9d6c7c3b5c4f25e"
PARENT_DIRECTORY = ROOT / "results/canonical" / PARENT_ARTIFACT
GENERATOR_ARTIFACT = "causal_inner_pathwise_closure_descriptor_pilot_wp10c9d6c7c3b5c4f25c"
GENERATOR_DIRECTORY = ROOT / "results/canonical" / GENERATOR_ARTIFACT
ARTIFACT = "causal_inner_resolved_mode_promotion_manifest_wp10c9d6c7c3b5c4f25f"
ARTIFACT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = "scripts/run_causal_inner_resolved_mode_promotion_manifest_wp10c9d6c7c3b5c4f25f.py"
THIS_TEST = "tests/test_causal_inner_resolved_mode_promotion_manifest_wp10c9d6c7c3b5c4f25f.py"
NEXT_RUNNER = "scripts/run_causal_inner_resolved_mode_promotion_audit_wp10c9d6c7c3b5c4f25g.py"
NEXT_TEST = "tests/test_causal_inner_resolved_mode_promotion_audit_wp10c9d6c7c3b5c4f25g.py"
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_RESOLVED_MODE_PROMOTION_"
    "MANIFEST_WP10C9D6C7C3B5C4F25F_2026-08-18.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

TRUTH_DIMENSION = 560
PARENT_RESOLVED_DIMENSION = 82
PARENT_UNRESOLVED_DIMENSION = 478
EXPECTED_NONSTABLE_DIMENSION = 24
MAXIMUM_PROMOTED_DIMENSION = 32
MAXIMUM_AUGMENTED_RESOLVED_DIMENSION = 114
STABILITY_MARGIN_PER_SECOND = 1.0e-8
FREQUENCY_COUNT_INCLUDING_DC = 33


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
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("parent spectrum certificate commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("parent spectrum certificate parent changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("parent spectrum certificate tree changed")
    hashes = _checksums(PARENT_DIRECTORY)
    summary = _read(PARENT_DIRECTORY / "summary.json")
    spectrum = _read(PARENT_DIRECTORY / "stage_2_metrics.json")
    if (
        not summary["passed"]
        or summary["classification"]
        != "invariant_projection_transfer_passed_unstable_modes_require_promotion"
        or summary["authorized_next"] != "definitions_only_resolved_mode_promotion_manifest"
        or summary["physical_failure_detected"]
        or spectrum["unstable_unresolved_pole_count"] != EXPECTED_NONSTABLE_DIMENSION
        or spectrum["memory_coefficients_fit"] != 0
        or spectrum["new_full_560_direction_descriptor_assemblies"] != 0
    ):
        raise RuntimeError("parent promotion authorization changed")
    return summary, spectrum, hashes


def _contract() -> dict:
    decisive = {
        "parent_projection": _sha(PARENT_DIRECTORY / "projection.npz"),
        "parent_poles": _sha(PARENT_DIRECTORY / "unresolved_poles.npz"),
        "saved_generator": _sha(GENERATOR_DIRECTORY / "descriptor_A.npz"),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "parent_decisive_hashes": decisive,
        "execution_budget": {
            "allowed_new_nonlinear_roots": 0,
            "allowed_propagated_states": 0,
            "allowed_new_full_560_direction_descriptor_assemblies": 0,
            "saved_complete_generator_must_be_reused": True,
            "allowed_memory_coefficients_fit": 0,
            "maximum_wall_hours": 2.0,
        },
        "ordered_real_schur_attribution": {
            "parent_resolved_dimension": PARENT_RESOLVED_DIMENSION,
            "parent_unresolved_dimension": PARENT_UNRESOLVED_DIMENSION,
            "expected_nonstable_compressed_dimension": EXPECTED_NONSTABLE_DIMENSION,
            "stability_definition": "real_part_strictly_less_than_minus_margin",
            "stability_margin_per_second": STABILITY_MARGIN_PER_SECOND,
            "ordered_partition": "stable_first_then_nonstable",
            "full_generator_spectrum_is_diagnostic_only": True,
            "compressed_nonstable_poles_are_not_a_physical_instability_claim": True,
            "record_full_nonstable_subspace_capture_and_principal_angles": True,
        },
        "algebraic_promotion": {
            "promote_every_nonstable_compressed_real_schur_coordinate": True,
            "promotion_basis": "Z_times_ordered_real_schur_nonstable_vectors",
            "augmented_restriction": "stack_R_and_Uu_transpose_times_Z_transpose",
            "augmented_lifting": "columns_L_and_Z_times_Uu",
            "remaining_complement": "Z_times_Us",
            "maximum_promoted_dimension": MAXIMUM_PROMOTED_DIMENSION,
            "maximum_augmented_resolved_dimension": MAXIMUM_AUGMENTED_RESOLVED_DIMENSION,
            "pass_requires": {
                "parent_nonstable_dimension": EXPECTED_NONSTABLE_DIMENSION,
                "promoted_dimension": EXPECTED_NONSTABLE_DIMENSION,
                "augmented_resolved_dimension_max": MAXIMUM_AUGMENTED_RESOLVED_DIMENSION,
                "augmented_restriction_lifting_identity_max": 5.0e-11,
                "augmented_restriction_complement_annihilation_max": 5.0e-11,
                "stable_complement_orthogonality_max": 5.0e-11,
                "stable_complement_lifting_annihilation_max": 5.0e-11,
                "ordered_schur_reconstruction_relative_defect_max": 1.0e-10,
                "ordered_schur_orthogonality_defect_max": 5.0e-11,
                "remaining_unresolved_spectral_abscissa_per_second_max": -STABILITY_MARGIN_PER_SECOND,
            },
        },
        "stable_transfer_reaudit": {
            "frequency_count_including_DC": FREQUENCY_COUNT_INCLUDING_DC,
            "transfer": "G(s)=D_aug+C_s*(sI-A_s)^(-1)*A_sr",
            "no_memory_coefficients_are_fit_in_this_package": True,
            "pass_requires": {
                "frequency_solve_relative_residual_max": 1.0e-10,
                "conjugate_symmetry_relative_defect_max": 1.0e-10,
                "database_roundtrip_bitwise": True,
            },
        },
        "decision": {
            "promotion_and_stable_transfer_pass": (
                "definitions_only_mode_selection_and_finite_memory_manifest_authorized"
            ),
            "promotion_budget_exceeded": "reduced_architecture_dimension_budget_failed_stop",
            "remaining_unresolved_not_strictly_stable": (
                "resolved_mode_promotion_failed_remaining_memory_not_stable"
            ),
            "numerical_gate_failure": "resolved_mode_promotion_audit_failed_stop",
        },
        "claim_boundary": {
            "physical_instability_claim_authorized": False,
            "memory_fit_authorized_in_this_package": False,
            "full_anchor_campaign_authorized": False,
            "online_solver_authorized": False,
            "predictive_cycle_authorized": False,
            "reduced_slow_evolution_authorized": False,
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
        "latest_source_parent_commit": PARENT_COMMIT,
        "latest_work_package": WORK_PACKAGE,
    })
    _write(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    parent_summary, parent_spectrum, parent_hashes = _validate_parent()
    if not _tracked_tree_clean():
        raise RuntimeError("resolved-mode promotion manifest requires a clean tracked tree")
    if ARTIFACT_DIRECTORY.exists():
        raise RuntimeError("resolved-mode promotion manifest is already frozen")
    contract = _contract()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "ordered_real_schur_attribution_authorized": True,
        "algebraic_promotion_authorized": True,
        "stable_transfer_reaudit_authorized_only_after_promotion_pass": True,
        "parent_nonstable_compressed_dimension": parent_spectrum["unstable_unresolved_pole_count"],
        "new_truth_trajectory_executed": False,
        "new_fixed_Q_root_executed": False,
        "new_full_descriptor_assembly_executed": False,
        "memory_fit_executed": False,
        "physical_instability_claim_authorized": False,
        "full_anchor_campaign_authorized": False,
        "online_reduced_solver_implementation_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "physical_failure_detected": False,
        "parent_classification_preserved": parent_summary["classification"],
        "authorized_next": WORK_PACKAGE.replace("25f", "25g"),
    }
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write(ARTIFACT_DIRECTORY / "contract.json", contract)
    _write(ARTIFACT_DIRECTORY / "summary.json", summary)
    _write(ARTIFACT_DIRECTORY / "parent_lock.json", {
        "parent_commit": PARENT_COMMIT,
        "parent_parent": PARENT_PARENT,
        "parent_tree": PARENT_TREE,
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
            "# Resolved-mode promotion manifest WP10c9d6c7c3b5c4f25f",
            "",
            "## Classification",
            "",
            f"`{CLASSIFICATION}`",
            "",
            "This definitions-only package preserves the certified R82 projection and its 24 nonstable compressed poles. It authorizes one hash-locked saved-generator audit; it does not authorize a physical instability claim.",
            "",
            "The audit orders the real Schur form of the 478-dimensional unresolved block, promotes every nonstable real Schur coordinate into the explicit resolved state, and requires the remaining unresolved block to be strictly stable. The explicit-state ceiling is 114 coordinates.",
            "",
            "Only after the promotion algebra passes may the 33-point transfer be rebuilt on the stable complement. No nonlinear root, propagation, new full generator assembly, memory fit, anchor campaign, online solver, or cycle is authorized.",
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
