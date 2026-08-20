#!/usr/bin/env python3
"""Freeze a new independent validation of the active-8 projective kernel."""

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

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_active8_projective_kernel_diagnosis_wp10c9d6c7c3b5c4f25bq as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25br"
PARENT_COMMIT = "ac7fc1c9bb3fc9ed4f65eb65a871fdfab4d378d7"
PARENT_PARENT = "7e0d0484751fc67a0ef6755afe0fb18c7537ef72"
PARENT_TREE = "7488e2ca92ce741b45af94a104c1eaaa6c91d735"
CLASSIFICATION = (
    "active8_projective_kernel_new_independent_validation_manifest_frozen"
)
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25bs"

ARTIFACT = (
    "causal_inner_active8_projective_kernel_validation_manifest_"
    "wp10c9d6c7c3b5c4f25br"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_active8_projective_kernel_validation_manifest_"
    "wp10c9d6c7c3b5c4f25br.py"
)
THIS_TEST = (
    "tests/test_causal_inner_active8_projective_kernel_validation_manifest_"
    "wp10c9d6c7c3b5c4f25br.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ACTIVE8_PROJECTIVE_KERNEL_"
    "VALIDATION_MANIFEST_WP10C9D6C7C3B5C4F25BR_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

HIGH_COMPONENT_BOUND = 1.0e-2
LOW_COMPONENT_BOUND = 5.0e-3
REVEALED_HIGH_DIRECTION_COUNT = 144
NEW_HOLDOUT_DIRECTION_COUNT = 16
NEW_RADIAL_DIRECTION_COUNT = 8
NEW_HIGH_CANDIDATE_COUNT = 2 * NEW_HOLDOUT_DIRECTION_COUNT
NEW_LOW_CANDIDATE_COUNT = 2 * NEW_RADIAL_DIRECTION_COUNT
PLANNED_CANDIDATES = NEW_HIGH_CANDIDATE_COUNT + NEW_LOW_CANDIDATE_COUNT


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
    return value


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


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
        raise RuntimeError("projective-kernel diagnosis commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != _git(
        "rev-parse", PARENT_PARENT
    ):
        raise RuntimeError("projective-kernel diagnosis lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("projective-kernel diagnosis tree changed")
    hashes = _checksums(parent.CANONICAL_DIRECTORY)
    summary = _read(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = _read(parent.CANONICAL_DIRECTORY / "metrics.json")
    if (
        not summary["passed"]
        or summary["classification"] != parent.CLASSIFICATION
        or summary["authorized_next"] != parent.AUTHORIZED_NEXT
        or not summary["diagnostic_only"]
        or summary["independent_model_validation_passed"]
        or not all(metrics["checks"].values())
    ):
        raise RuntimeError("projective-kernel diagnosis authorization changed")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("projective-kernel manifest requires a clean tree")
    return {"summary": summary, "metrics": metrics, "hashes": hashes}


def _revealed_high_directions() -> np.ndarray:
    rate = parent.parent
    _metrics, arrays = rate.manifest._design()
    revealed = np.vstack(
        (
            arrays["total_training_directions_active8"].T,
            arrays["new_tuning_directions_active8"].T,
            arrays["new_holdout_directions_active8"].T,
        )
    )
    if revealed.shape != (REVEALED_HIGH_DIRECTION_COUNT, 8):
        raise RuntimeError("revealed high-direction database changed")
    return revealed


def _design() -> tuple[dict, dict[str, np.ndarray]]:
    revealed = _revealed_high_directions()
    pool = parent.parent.architecture._candidate_pool()
    distance = np.min(1.0 - np.abs(pool @ revealed.T), axis=1)
    indices = []
    selection_distances = []
    for _ in range(NEW_HOLDOUT_DIRECTION_COUNT):
        selected = int(np.argmax(distance))
        indices.append(selected)
        selection_distances.append(float(distance[selected]))
        distance = np.minimum(distance, 1.0 - np.abs(pool @ pool[selected]))
        distance[selected] = -1.0
    holdout = pool[indices]
    mutual = 1.0 - np.abs(holdout @ holdout.T) + np.eye(holdout.shape[0])
    metrics = {
        "revealed_high_direction_count": int(revealed.shape[0]),
        "new_holdout_direction_count": int(holdout.shape[0]),
        "new_radial_direction_count": NEW_RADIAL_DIRECTION_COUNT,
        "planned_candidate_count": PLANNED_CANDIDATES,
        "minimum_new_holdout_to_revealed_projective_separation": float(
            np.min(1.0 - np.abs(holdout @ revealed.T))
        ),
        "minimum_new_holdout_mutual_projective_separation": float(
            np.min(mutual)
        ),
        "maximum_absolute_new_direction_component": float(np.max(np.abs(holdout))),
        "selection_distances": selection_distances,
    }
    return metrics, {
        "revealed_high_directions_active8": revealed.T,
        "new_holdout_directions_active8": holdout.T,
        "new_radial_directions_active8": holdout[:NEW_RADIAL_DIRECTION_COUNT].T,
    }


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "objective": (
            "independently_validate_the_frozen_active8_projective_even_"
            "kernel_cubic_odd_rate_closure_on_new_untouched_directions"
        ),
        "mathematical_architecture": {
            "online_state": "q162_plus_z280_plus_a28_equals_470",
            "stable_descriptor_kernel_dimension": 280,
            "active_departure_input_dimension": 8,
            "even_rate_kernel": "dot_squared_plus_dot_fourth_divided_by_320",
            "even_target_weight_exponent": 2.0,
            "even_Tikhonov_regularization": 1.0 / 64.0,
            "odd_rate_closure": "overdetermined_full_homogeneous_cubic120",
            "state_decoder": "linear470_plus_rank4_algebraic_slaved_curvature",
            "dynamic_curvature_augmentation": False,
            "online_truth_calls_per_macrostep": 0,
            "online_Newton_retractions_per_macrostep": 0,
            "stored_nonlinear_coefficients_after_refit": 7_872,
        },
        "database": {
            "revealed_high_directions_reclassified_as_training": 144,
            "new_untouched_holdout_directions_high_radius": 16,
            "new_radial_subset_directions_high_and_low": 8,
            "high_component_bound": HIGH_COMPONENT_BOUND,
            "low_component_bound": LOW_COMPONENT_BOUND,
            "planned_signed_candidates": PLANNED_CANDIDATES,
            "candidate_order": (
                "new_holdout_high_negative_positive_then_first8_low_"
                "negative_positive"
            ),
        },
        "leakage_control": {
            "all_parent_validation_responses": "revealed_training_only",
            "architecture_and_hyperparameters_frozen_before_new_geometry": True,
            "all_coefficients_frozen_and_hashed_before_new_rate_truth": True,
            "new_holdout_cannot_change_any_coefficient_or_threshold": True,
            "geometry_may_test_only_admissibility_not_rate_accuracy": True,
        },
        "design_gates": {
            "revealed_high_direction_count_equal": REVEALED_HIGH_DIRECTION_COUNT,
            "new_holdout_direction_count_equal": NEW_HOLDOUT_DIRECTION_COUNT,
            "new_radial_direction_count_equal": NEW_RADIAL_DIRECTION_COUNT,
            "planned_candidate_count_equal": PLANNED_CANDIDATES,
            "minimum_new_holdout_to_revealed_projective_separation": 0.265,
            "minimum_new_holdout_mutual_projective_separation": 0.27,
            "maximum_absolute_new_direction_component": 0.75,
        },
        "binding_geometry_gates": {
            "completed_candidate_count_equal": PLANNED_CANDIDATES,
            "failed_candidate_count_equal": 0,
            "maximum_coordinate_residual_infinity": 1.0e-10,
            "maximum_normalized_Q3_defect": 1.0e-10,
            "maximum_final_scaled_component": HIGH_COMPONENT_BOUND,
            "minimum_reconstruction_factor": 1.0 - 1.0e-12,
            "maximum_reconstruction_factor": 1.0 + 1.0e-12,
            "maximum_coordinate_Jacobian_condition_number": 5.0e3,
            "minimum_departure_direction_alignment_cosine": 0.995,
            "maximum_departure_transverse_fraction": 0.05,
            "maximum_pair_coordinate_odd_symmetry_defect": 0.02,
            "maximum_H_over_R": 0.12,
            "minimum_scattering_optical_depth": 1.0,
            "nonbase_continuous_rate_evaluations_equal": 0,
            "new_full_generator_assemblies_equal": 0,
            "new_nonlinear_roots_equal": 0,
            "propagated_states_equal": 0,
        },
        "binding_truth_rate_gates": {
            "completed_nonbase_rate_evaluations_equal": PLANNED_CANDIDATES,
            "failed_rate_evaluations_equal": 0,
            "minimum_reconstruction_factor": 1.0 - 1.0e-12,
            "maximum_reconstruction_factor": 1.0 + 1.0e-12,
            "maximum_raw_Schur_condition_number": 1.0e6,
            "maximum_reaction_identity_defect": 1.0e-9,
            "maximum_rate_tangency_relative_defect": 1.0e-8,
            "maximum_coordinate_Jacobian_condition_number": 5.0e3,
            "maximum_H_over_R": 0.12,
            "minimum_scattering_optical_depth": 1.0,
            "maximum_incoming_excision_characteristics_equal": 0,
            "new_complete_generator_assemblies_equal": 0,
            "new_nonlinear_roots_equal": 0,
            "propagated_states_equal": 0,
        },
        "binding_radial_consistency_gates": {
            "maximum_quadratic_target_high_low_relative_difference": 0.10,
            "maximum_cubic_rate_target_high_low_relative_difference": 0.15,
            "maximum_curvature_cubic_target_high_low_relative_difference": 0.10,
        },
        "binding_independent_model_gates": {
            "holdout_median_nonlinear_departure_rate_relative_error": 0.10,
            "holdout_maximum_nonlinear_departure_rate_relative_error": 0.25,
            "holdout_median_full_departure_rate_relative_error": 0.02,
            "holdout_maximum_full_departure_rate_relative_error": 0.05,
            "maximum_curvature_prediction_error_over_full_state_delta": 1.5e-3,
            "maximum_full_scaled_state_decoder_relative_error": 2.5e-3,
            "maximum_reconstructed_C_phys_residual_infinity": 2.5e-4,
            "minimum_reconstructed_state_reconstruction_factor": 1.0 - 1.0e-12,
            "maximum_reconstructed_H_over_R": 0.12,
            "minimum_reconstructed_scattering_optical_depth": 1.0,
        },
        "decision": {
            "pass_classification": (
                "active8_projective_kernel_rate_and_rank4_decoder_"
                "independently_validated"
            ),
            "pass_authorizes_only": (
                "definitions_only_active8_short_reduced_vector_field_"
                "validation_manifest"
            ),
            "failure_classification": (
                "active8_projective_kernel_independent_validation_failed"
            ),
            "failure_authorizes_only": None,
            "predictive_cycle_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
    }


def _design_checks(metrics: dict, gates: dict) -> dict:
    return {
        "revealed_count": metrics["revealed_high_direction_count"]
        == gates["revealed_high_direction_count_equal"],
        "holdout_count": metrics["new_holdout_direction_count"]
        == gates["new_holdout_direction_count_equal"],
        "radial_count": metrics["new_radial_direction_count"]
        == gates["new_radial_direction_count_equal"],
        "candidate_count": metrics["planned_candidate_count"]
        == gates["planned_candidate_count_equal"],
        "training_separation": metrics[
            "minimum_new_holdout_to_revealed_projective_separation"
        ] >= gates["minimum_new_holdout_to_revealed_projective_separation"],
        "mutual_separation": metrics[
            "minimum_new_holdout_mutual_projective_separation"
        ] >= gates["minimum_new_holdout_mutual_projective_separation"],
        "component_bound": metrics["maximum_absolute_new_direction_component"]
        <= gates["maximum_absolute_new_direction_component"],
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
        raise RuntimeError("projective-kernel manifest already canonicalized")
    metrics, arrays = _design()
    contract = _contract()
    checks = _design_checks(metrics, contract["design_gates"])
    if not all(checks.values()):
        raise RuntimeError("projective-kernel holdout design failed")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "design_checks_passed": True,
        "new_truth_evaluations": 0,
        "independent_model_validation_passed": False,
        "authorized_next": AUTHORIZED_NEXT,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    CANONICAL_DIRECTORY.mkdir(parents=True)
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    _write_json(CANONICAL_DIRECTORY / "contract.json", contract)
    _write_json(
        CANONICAL_DIRECTORY / "design_metrics.json",
        {"checks": checks, **metrics},
    )
    _write_npz(CANONICAL_DIRECTORY / "validation_design.npz", arrays)
    _write_json(
        CANONICAL_DIRECTORY / "parent_lock.json",
        {
            "parent_commit": PARENT_COMMIT,
            "parent_parent": _git("rev-parse", PARENT_PARENT),
            "parent_tree": PARENT_TREE,
            "parent_hashes": frozen["hashes"],
        },
    )
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "DEFINITIONS_ONLY",
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "source_hashes": {
                relative: _sha(ROOT / relative)
                for relative in (THIS_RUNNER, THIS_TEST, parent.THIS_RUNNER, parent.THIS_TEST)
            },
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
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
                "# Active-8 projective-kernel validation manifest WP10c9d6c7c3b5c4f25br",
                "",
                "## Classification",
                "",
                f"`{CLASSIFICATION}`",
                "",
                "This definitions-only package freezes the projective even kernel, its weights and regularization, all existing revealed training responses, and 16 new untouched holdout directions before any new geometry or rate response is evaluated.",
                "",
                f"Planned new signed candidates: `{PLANNED_CANDIDATES}`. Eight directions are repeated at half radius for radial consistency.",
                "",
                f"Authorized next work package: `{AUTHORIZED_NEXT}`. No trajectory or reduced slow evolution is authorized.",
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
