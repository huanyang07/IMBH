#!/usr/bin/env python3
"""Diagnose and select a regularized projective active-8 rate architecture."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_active8_tensor_rate_validation_wp10c9d6c7c3b5c4f25bp as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25bq"
PARENT_COMMIT = "86798b5d7030ff2589a44113c0715e6267dc38a3"
PARENT_PARENT = "22bd33f075573e09d673663718dde7739698337c"
PARENT_TREE = "b99730187bca686e8698b692d25f52d9020d29cd"

CLASSIFICATION = (
    "active8_projective_even_kernel_cubic_odd_architecture_"
    "selected_for_new_independent_validation"
)
AUTHORIZED_NEXT = (
    "definitions_only_active8_projective_kernel_"
    "independent_validation_manifest"
)
ARTIFACT = (
    "causal_inner_active8_projective_kernel_diagnosis_"
    "wp10c9d6c7c3b5c4f25bq"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_active8_projective_kernel_diagnosis_"
    "wp10c9d6c7c3b5c4f25bq.py"
)
THIS_TEST = (
    "tests/test_causal_inner_active8_projective_kernel_diagnosis_"
    "wp10c9d6c7c3b5c4f25bq.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ACTIVE8_PROJECTIVE_KERNEL_"
    "DIAGNOSIS_WP10C9D6C7C3B5C4F25BQ_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

# Frozen after, and therefore diagnostic with respect to, the revealed parent
# tuning and holdout sets.  New truth data are required for certification.
EVEN_TARGET_WEIGHT_EXPONENT = 2.0
EVEN_TIKHONOV_REGULARIZATION = 1.0 / 64.0
EVEN_QUARTIC_KERNEL_WEIGHT = 1.0 / 320.0
TRAINING_DIRECTION_COUNT = 120
EVEN_KERNEL_COEFFICIENT_COUNT = 120 * 28
ODD_CUBIC_COEFFICIENT_COUNT = 120 * 28
CURVATURE_COEFFICIENT_COUNT = 120 * 4
TOTAL_NONLINEAR_COEFFICIENT_COUNT = (
    EVEN_KERNEL_COEFFICIENT_COUNT
    + ODD_CUBIC_COEFFICIENT_COUNT
    + CURVATURE_COEFFICIENT_COUNT
)


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
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
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
        raise RuntimeError("tensor-rate rejection commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("tensor-rate rejection lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("tensor-rate rejection tree changed")
    hashes = _checksums(parent.CANONICAL_DIRECTORY)
    summary = _read(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = _read(parent.CANONICAL_DIRECTORY / "metrics.json")
    provenance = _read(parent.CANONICAL_DIRECTORY / "provenance.json")
    if (
        summary["classification"] != parent.FAIL_CLASSIFICATION
        or summary["passed"]
        or not summary["truth_database_passed"]
        or summary["independent_model_validation_passed"]
        or summary["authorized_next"] is not None
        or not all(metrics["truth_checks"].values())
        or all(metrics["model_checks"].values())
    ):
        raise RuntimeError("tensor-rate rejection classification changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"parent source changed: {relative}")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("projective-kernel diagnosis requires a clean tree")
    for name, expected in parent.parent.chart_tools.coordinate_tools.THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    return {"summary": summary, "metrics": metrics, "hashes": hashes}


def _even_kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    dot = np.asarray(left, dtype=float) @ np.asarray(right, dtype=float).T
    return dot**2 + EVEN_QUARTIC_KERNEL_WEIGHT * dot**4


def _relative_errors(
    predicted_nonlinear: np.ndarray,
    truth_nonlinear: np.ndarray,
    truth_full: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    numerator = np.linalg.norm(predicted_nonlinear - truth_nonlinear, axis=1)
    nonlinear = numerator / np.maximum(
        np.linalg.norm(truth_nonlinear, axis=1), np.finfo(float).tiny
    )
    full = numerator / np.maximum(
        np.linalg.norm(truth_full, axis=1), np.finfo(float).tiny
    )
    return nonlinear, full


def _aggregate(values: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    selected = np.asarray(values, dtype=float)[mask]
    return float(np.median(selected)), float(np.max(selected))


def _diagnose() -> tuple[dict, dict[str, np.ndarray]]:
    inputs = parent._load_inputs()
    truth = parent._load_npz(parent.CANONICAL_DIRECTORY / "tensor_closure.npz")
    frozen = parent._load_npz(
        parent.CANONICAL_DIRECTORY / "frozen_coefficients.npz"
    )
    directions = np.asarray(frozen["directions"], dtype=float)
    quadratic_targets = np.asarray(
        frozen["rate_quadratic_targets"], dtype=float
    )
    cubic_targets = np.asarray(frozen["rate_cubic_targets"], dtype=float)
    if (
        directions.shape != (TRAINING_DIRECTION_COUNT, 8)
        or quadratic_targets.shape != (TRAINING_DIRECTION_COUNT, 28)
        or cubic_targets.shape != (TRAINING_DIRECTION_COUNT, 28)
    ):
        raise RuntimeError("revealed tensor-training dimensions changed")

    target_norms = np.linalg.norm(quadratic_targets, axis=1)
    target_scale = float(np.median(target_norms))
    weights = (
        target_scale / np.maximum(target_norms, np.finfo(float).tiny)
    ) ** EVEN_TARGET_WEIGHT_EXPONENT
    kernel = _even_kernel(directions, directions)
    regularized = kernel + EVEN_TIKHONOV_REGULARIZATION * np.diag(
        1.0 / weights
    )
    even_coefficients = np.linalg.solve(regularized, quadratic_targets)
    cubic_features = parent.architecture._cubic_features(directions)
    odd_coefficients = np.linalg.solve(cubic_features, cubic_targets)

    validation_indices = np.arange(128, 192, dtype=int)
    energy = np.asarray(inputs["database"]["energy_directions"], dtype=float)
    active = np.asarray(
        [energy.T @ inputs["coordinates"][index] for index in validation_indices]
    )
    radii = np.linalg.norm(active, axis=1)
    unit = active / radii[:, None]
    predicted_quadratic = _even_kernel(unit, directions) @ even_coefficients
    predicted_cubic = (
        parent.architecture._cubic_features(unit) @ odd_coefficients
    )
    predicted_nonlinear = (
        radii[:, None] ** 2 * predicted_quadratic
        + radii[:, None] ** 3 * predicted_cubic
    )
    truth_full = truth["departure_rate_increments_per_second"][
        validation_indices
    ]
    truth_nonlinear = truth_full - truth[
        "departure_linear_references_per_second"
    ][validation_indices]
    nonlinear_errors, full_errors = _relative_errors(
        predicted_nonlinear, truth_nonlinear, truth_full
    )
    splits = np.asarray(
        [inputs["candidates"][index]["split"] for index in validation_indices]
    )
    tuning = np.asarray(
        [str(value).startswith("tuning") for value in splits], dtype=bool
    )
    holdout = splits == "holdout"
    tuning_nonlinear = _aggregate(nonlinear_errors, tuning)
    tuning_full = _aggregate(full_errors, tuning)
    holdout_nonlinear = _aggregate(nonlinear_errors, holdout)
    holdout_full = _aggregate(full_errors, holdout)

    baseline = truth["predicted_nonlinear_departure_rates_per_second"][
        validation_indices
    ]
    baseline_residual = truth_nonlinear - baseline
    singular_values = np.linalg.svd(baseline_residual, compute_uv=False)
    energy_fraction = np.cumsum(singular_values**2) / np.sum(singular_values**2)
    residual_rank90 = int(np.searchsorted(energy_fraction, 0.90) + 1)
    residual_rank99 = int(np.searchsorted(energy_fraction, 0.99) + 1)

    metrics = {
        "diagnostic_only_revealed_validation": True,
        "training_direction_count": TRAINING_DIRECTION_COUNT,
        "revealed_tuning_candidate_count": int(np.sum(tuning)),
        "revealed_holdout_candidate_count": int(np.sum(holdout)),
        "even_target_weight_exponent": EVEN_TARGET_WEIGHT_EXPONENT,
        "even_Tikhonov_regularization": EVEN_TIKHONOV_REGULARIZATION,
        "even_quartic_kernel_weight": EVEN_QUARTIC_KERNEL_WEIGHT,
        "regularized_even_system_condition_number": float(
            np.linalg.cond(regularized)
        ),
        "odd_cubic_feature_condition_number": float(
            np.linalg.cond(cubic_features)
        ),
        "tuning_median_nonlinear_departure_rate_relative_error": tuning_nonlinear[0],
        "tuning_maximum_nonlinear_departure_rate_relative_error": tuning_nonlinear[1],
        "tuning_median_full_departure_rate_relative_error": tuning_full[0],
        "tuning_maximum_full_departure_rate_relative_error": tuning_full[1],
        "holdout_median_nonlinear_departure_rate_relative_error": holdout_nonlinear[0],
        "holdout_maximum_nonlinear_departure_rate_relative_error": holdout_nonlinear[1],
        "holdout_median_full_departure_rate_relative_error": holdout_full[0],
        "holdout_maximum_full_departure_rate_relative_error": holdout_full[1],
        "baseline_validation_residual_energy_rank90": residual_rank90,
        "baseline_validation_residual_energy_rank99": residual_rank99,
        "nonlinear_coefficient_count": TOTAL_NONLINEAR_COEFFICIENT_COUNT,
        "online_truth_calls_per_macrostep": 0,
        "online_Newton_retractions_per_macrostep": 0,
        "dynamic_state_dimension": 470,
        "dynamic_curvature_augmentation": False,
        "curvature_decoder_kind": "rank4_algebraic_slaved",
    }
    gates = {
        "regularized_even_system_condition_number": 1.0e4,
        "odd_cubic_feature_condition_number": 25.0,
        "tuning_median_nonlinear_departure_rate_relative_error": 0.10,
        "tuning_maximum_nonlinear_departure_rate_relative_error": 0.25,
        "tuning_median_full_departure_rate_relative_error": 0.02,
        "tuning_maximum_full_departure_rate_relative_error": 0.05,
        "holdout_median_nonlinear_departure_rate_relative_error": 0.10,
        "holdout_maximum_nonlinear_departure_rate_relative_error": 0.25,
        "holdout_median_full_departure_rate_relative_error": 0.02,
        "holdout_maximum_full_departure_rate_relative_error": 0.05,
        "online_truth_calls_per_macrostep": 0,
        "online_Newton_retractions_per_macrostep": 0,
        "dynamic_state_dimension": 470,
    }
    checks = {name: metrics[name] <= threshold for name, threshold in gates.items()}
    checks.update(
        {
            "parent_truth_database_passed": True,
            "parent_tensor_validation_remains_rejected": True,
            "new_independent_validation_required": True,
            "no_dynamic_curvature_augmentation": not metrics[
                "dynamic_curvature_augmentation"
            ],
        }
    )
    return {
        "metrics": metrics,
        "gates": gates,
        "checks": checks,
    }, {
        "training_directions_active8": directions,
        "even_kernel_coefficients": even_coefficients,
        "odd_cubic_coefficients": odd_coefficients,
        "validation_candidate_indices": validation_indices,
        "validation_active_coordinates": active,
        "predicted_nonlinear_departure_rates_per_second": predicted_nonlinear,
        "truth_nonlinear_departure_rates_per_second": truth_nonlinear,
        "nonlinear_departure_rate_relative_errors": nonlinear_errors,
        "full_departure_rate_relative_errors": full_errors,
        "baseline_validation_residual_singular_values": singular_values,
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
        raise RuntimeError("projective-kernel diagnosis already canonicalized")
    diagnosis, arrays = _diagnose()
    passed = all(diagnosis["checks"].values())
    if not passed:
        raise RuntimeError("projective-kernel diagnostic selection failed")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "diagnostic_only": True,
        "parent_truth_database_passed": True,
        "parent_tensor_validation_remains_rejected": True,
        "projective_kernel_architecture_selected": True,
        "independent_model_validation_passed": False,
        "authorized_next": AUTHORIZED_NEXT,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    CANONICAL_DIRECTORY.mkdir(parents=True)
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    _write_json(CANONICAL_DIRECTORY / "metrics.json", diagnosis)
    _write_npz(CANONICAL_DIRECTORY / "projective_kernel_diagnosis.npz", arrays)
    _write_json(
        CANONICAL_DIRECTORY / "parent_lock.json",
        {
            "parent_commit": PARENT_COMMIT,
            "parent_parent": PARENT_PARENT,
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
            "thread_environment": parent.parent.chart_tools.coordinate_tools.THREAD_ENVIRONMENT,
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
                "# Active-8 projective-kernel diagnosis WP10c9d6c7c3b5c4f25bq",
                "",
                "## Classification",
                "",
                f"`{CLASSIFICATION}`",
                "",
                "The parent tensor rejection is preserved. All 192 exact truth responses remain accepted; only worst-case closure errors failed.",
                "",
                "A frozen diagnostic architecture replaces the unregularized even tensor by an inverse-square norm-weighted projective kernel `(d_i.d_j)^2 + (d_i.d_j)^4/320` with Tikhonov regularization `1/64`. The odd cubic tensor and rank-4 algebraic curvature decoder remain unchanged.",
                "",
                f"Revealed tuning nonlinear median/max: `{diagnosis['metrics']['tuning_median_nonlinear_departure_rate_relative_error']:.6e}` / `{diagnosis['metrics']['tuning_maximum_nonlinear_departure_rate_relative_error']:.6e}`.",
                "",
                f"Revealed holdout nonlinear median/max: `{diagnosis['metrics']['holdout_median_nonlinear_departure_rate_relative_error']:.6e}` / `{diagnosis['metrics']['holdout_maximum_nonlinear_departure_rate_relative_error']:.6e}`.",
                "",
                "Because those sets informed architecture selection, this is diagnostic evidence only. A newly generated untouched holdout is binding.",
                "",
                f"Authorized next artifact: `{AUTHORIZED_NEXT}`. No trajectory or reduced slow evolution is authorized.",
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
