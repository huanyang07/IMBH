#!/usr/bin/env python3
"""Diagnose the quadratic-cubic low-rank departure architecture."""

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

import run_causal_inner_parity_low_rank_architecture_manifest_wp10c9d6c7c3b5c4f25bh as manifest  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25bi"
MANIFEST_COMMIT = "63881f8ce8eee74b729b6057d2e01117ce95d1c4"
MANIFEST_PARENT = "da92f52e9f617a0137f77584855d1524ac8b6720"
MANIFEST_TREE = "e564908397a93534de26962b9671f9a0ddd0c7b1"

PASS_CLASSIFICATION = (
    "quadratic_cubic_low_rank_departure_architecture_diagnosed_"
    "mixed_direction_database_manifest_authorized"
)
FAIL_CLASSIFICATION = (
    "parity_low_rank_architecture_diagnosis_inconsistent_"
    "mixed_direction_database_blocked"
)

ARTIFACT = (
    "causal_inner_parity_low_rank_architecture_audit_"
    "wp10c9d6c7c3b5c4f25bi"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_parity_low_rank_architecture_audit_"
    "wp10c9d6c7c3b5c4f25bi.py"
)
THIS_TEST = (
    "tests/test_causal_inner_parity_low_rank_architecture_audit_"
    "wp10c9d6c7c3b5c4f25bi.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_PARITY_LOW_RANK_"
    "ARCHITECTURE_AUDIT_WP10C9D6C7C3B5C4F25BI_2026-08-19.md"
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


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name], dtype=float) for name in source.files}


def _validate_manifest(*, require_clean: bool) -> dict:
    if _git("rev-parse", MANIFEST_COMMIT) != MANIFEST_COMMIT:
        raise RuntimeError("parity architecture manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("parity architecture manifest lineage changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("parity architecture manifest tree changed")
    hashes = _checksums(manifest.ARTIFACT_DIRECTORY)
    summary = _read(manifest.ARTIFACT_DIRECTORY / "summary.json")
    contract = _read(manifest.ARTIFACT_DIRECTORY / "contract.json")
    provenance = _read(manifest.ARTIFACT_DIRECTORY / "provenance.json")
    lock = _read(manifest.ARTIFACT_DIRECTORY / "parent_lock.json")
    if (
        not summary["passed"]
        or summary["authorized_next"] != WORK_PACKAGE
        or not summary["post_result_architecture_diagnosis"]
        or summary["independent_validation_claimed"]
        or summary["planned_new_truth_rate_evaluations"] != 0
        or contract["claim_boundary"]["thresholds_were_selected_blind_to_existing_results"]
    ):
        raise RuntimeError("parity architecture diagnosis authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"manifest source changed: {relative}")
    for name, path in (
        ("rate_0p005", manifest.RATE_0P005_PATH),
        ("rate_0p01", manifest.RATE_0P01_PATH),
        ("chart_0p02", manifest.CHART_0P02_PATH),
        ("online_470_geometry", manifest.GEOMETRY_PATH),
    ):
        if _sha(path) != lock["decisive_input_hashes"][name]:
            raise RuntimeError(f"parity architecture input changed: {path}")
    for directory in (
        manifest.parent.CANONICAL_DIRECTORY,
        manifest.rate_0p01.CANONICAL_DIRECTORY,
        manifest.rate_0p005.CANONICAL_DIRECTORY,
        manifest.architecture.CANONICAL_DIRECTORY,
    ):
        _checksums(directory)
    if require_clean and not _tracked_tree_clean():
        raise RuntimeError("parity architecture audit requires a clean tracked tree")
    for name, expected in (
        manifest.rate_0p01.manifest.parent.chart_tools.coordinate_tools.THREAD_ENVIRONMENT.items()
    ):
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    return {"summary": summary, "contract": contract, "hashes": hashes}


def _parity_terms(
    metrics: dict, arrays: dict[str, np.ndarray], amplitude_index: int
) -> dict[str, np.ndarray]:
    candidates = metrics["evaluations"]
    residual = (
        arrays["departure_rate_increments_per_second"]
        - arrays["departure_linear_references_per_second"]
    )
    linear = arrays["departure_linear_references_per_second"]
    coordinates = arrays["candidate_departure_coordinates"]
    radii = []
    even_residuals = []
    odd_residuals = []
    quadratic_coefficients = []
    cubic_coefficients = []
    even_relative_signals = []
    odd_relative_signals = []
    active_directions = []
    for direction_index in range(manifest.ACTIVE_INPUT_DIMENSION):
        indices = [
            index
            for index, item in enumerate(candidates)
            if item["direction_index"] == direction_index
            and item.get("amplitude_index", 0) == amplitude_index
        ]
        if len(indices) != 2:
            raise RuntimeError("signed parity pair is incomplete")
        negative, positive = sorted(
            indices, key=lambda index: candidates[index]["sign"]
        )
        coordinate_odd = 0.5 * (
            coordinates[positive] - coordinates[negative]
        )
        radius = float(np.linalg.norm(coordinate_odd))
        if radius <= np.finfo(float).tiny:
            raise RuntimeError("signed parity radius vanished")
        direction = coordinate_odd / radius
        even = 0.5 * (residual[positive] + residual[negative])
        odd = 0.5 * (residual[positive] - residual[negative])
        linear_odd = 0.5 * (linear[positive] - linear[negative])
        denominator = max(float(np.linalg.norm(linear_odd)), np.finfo(float).tiny)
        radii.append(radius)
        active_directions.append(direction)
        even_residuals.append(even)
        odd_residuals.append(odd)
        quadratic_coefficients.append(even / radius**2)
        cubic_coefficients.append(odd / radius**3)
        even_relative_signals.append(float(np.linalg.norm(even) / denominator))
        odd_relative_signals.append(float(np.linalg.norm(odd) / denominator))
    return {
        "radii": np.asarray(radii),
        "active_directions": np.asarray(active_directions),
        "even_residuals": np.asarray(even_residuals),
        "odd_residuals": np.asarray(odd_residuals),
        "quadratic_coefficients": np.asarray(quadratic_coefficients),
        "cubic_coefficients": np.asarray(cubic_coefficients),
        "even_relative_signals": np.asarray(even_relative_signals),
        "odd_relative_signals": np.asarray(odd_relative_signals),
    }


def _row_balanced_basis(
    coefficients: np.ndarray, rank: int
) -> tuple[np.ndarray, np.ndarray, float]:
    norms = np.linalg.norm(coefficients, axis=1)
    if np.any(norms <= np.finfo(float).tiny):
        raise RuntimeError("parity coefficient row vanished")
    balanced = coefficients / norms[:, None]
    _left, singular, right = np.linalg.svd(balanced, full_matrices=False)
    energy = float(np.sum(singular[:rank] ** 2) / np.sum(singular**2))
    return right[:rank].T, singular, energy


def _coefficient_consistency(
    low: np.ndarray, high: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    low_norm = np.linalg.norm(low, axis=1)
    high_norm = np.linalg.norm(high, axis=1)
    changes = np.linalg.norm(high - low, axis=1) / np.maximum(
        low_norm, np.finfo(float).tiny
    )
    cosines = np.sum(low * high, axis=1) / np.maximum(
        low_norm * high_norm, np.finfo(float).tiny
    )
    return changes, cosines


def _diagnose() -> tuple[dict, dict[str, np.ndarray]]:
    metrics_0p005 = _read(
        manifest.rate_0p005.CANONICAL_DIRECTORY / "metrics.json"
    )
    metrics_0p01 = _read(
        manifest.rate_0p01.CANONICAL_DIRECTORY / "metrics.json"
    )
    metrics_0p02 = _read(manifest.parent.CANONICAL_DIRECTORY / "metrics.json")
    arrays_0p005 = _load_npz(manifest.RATE_0P005_PATH)
    arrays_0p01 = _load_npz(manifest.RATE_0P01_PATH)
    geometry = _load_npz(manifest.GEOMETRY_PATH)
    low = _parity_terms(metrics_0p005, arrays_0p005, 2)
    high = _parity_terms(metrics_0p01, arrays_0p01, 0)
    quadratic_changes, quadratic_cosines = _coefficient_consistency(
        low["quadratic_coefficients"], high["quadratic_coefficients"]
    )
    cubic_changes, cubic_cosines = _coefficient_consistency(
        low["cubic_coefficients"], high["cubic_coefficients"]
    )
    quadratic_basis, quadratic_singular, quadratic_energy = _row_balanced_basis(
        high["quadratic_coefficients"], manifest.QUADRATIC_OUTPUT_RANK
    )
    cubic_basis, cubic_singular, cubic_energy = _row_balanced_basis(
        high["cubic_coefficients"], manifest.CUBIC_OUTPUT_RANK
    )
    even_amplification = high["even_relative_signals"] / np.maximum(
        low["even_relative_signals"], np.finfo(float).tiny
    )
    odd_amplification = high["odd_relative_signals"] / np.maximum(
        low["odd_relative_signals"], np.finfo(float).tiny
    )
    metrics = {
        "signed_pairs_per_amplitude": manifest.ACTIVE_INPUT_DIMENSION,
        "median_even_relative_signal_at_0p005": float(
            np.median(low["even_relative_signals"])
        ),
        "median_even_relative_signal_at_0p01": float(
            np.median(high["even_relative_signals"])
        ),
        "maximum_even_relative_signal_at_0p01": float(
            np.max(high["even_relative_signals"])
        ),
        "median_odd_relative_signal_at_0p005": float(
            np.median(low["odd_relative_signals"])
        ),
        "median_odd_relative_signal_at_0p01": float(
            np.median(high["odd_relative_signals"])
        ),
        "maximum_odd_relative_signal_at_0p01": float(
            np.max(high["odd_relative_signals"])
        ),
        "median_even_relative_amplification": float(np.median(even_amplification)),
        "median_odd_relative_amplification": float(np.median(odd_amplification)),
        "maximum_quadratic_coefficient_relative_change": float(
            np.max(quadratic_changes)
        ),
        "median_quadratic_coefficient_relative_change": float(
            np.median(quadratic_changes)
        ),
        "minimum_quadratic_coefficient_cosine": float(
            np.min(quadratic_cosines)
        ),
        "maximum_cubic_coefficient_relative_change": float(
            np.max(cubic_changes)
        ),
        "median_cubic_coefficient_relative_change": float(
            np.median(cubic_changes)
        ),
        "minimum_cubic_coefficient_cosine": float(np.min(cubic_cosines)),
        "quadratic_row_normalized_rank3_energy": quadratic_energy,
        "cubic_row_normalized_rank4_energy": cubic_energy,
        "amplitude_0p02_maximum_transverse_distortion": metrics_0p02[
            "maximum_departure_transverse_fraction"
        ],
        "active_input_dimension": manifest.ACTIVE_INPUT_DIMENSION,
        "quadratic_output_rank": manifest.QUADRATIC_OUTPUT_RANK,
        "cubic_output_rank": manifest.CUBIC_OUTPUT_RANK,
        "quadratic_feature_count": manifest.QUADRATIC_FEATURE_COUNT,
        "cubic_feature_count": manifest.CUBIC_FEATURE_COUNT,
        "compressed_polynomial_coefficient_upper_bound": (
            manifest.COMPRESSED_POLYNOMIAL_COEFFICIENT_COUNT
        ),
        "new_truth_rate_evaluations": 0,
        "new_retractions": 0,
        "new_roots": 0,
        "propagated_states": 0,
    }
    arrays = {
        "active_input_directions": high["active_directions"].T,
        "departure_coordinate_basis": geometry["departure_coordinate_basis"],
        "radii_0p005": low["radii"],
        "radii_0p01": high["radii"],
        "quadratic_coefficients_0p005": low["quadratic_coefficients"],
        "quadratic_coefficients_0p01": high["quadratic_coefficients"],
        "cubic_coefficients_0p005": low["cubic_coefficients"],
        "cubic_coefficients_0p01": high["cubic_coefficients"],
        "even_relative_signals_0p005": low["even_relative_signals"],
        "even_relative_signals_0p01": high["even_relative_signals"],
        "odd_relative_signals_0p005": low["odd_relative_signals"],
        "odd_relative_signals_0p01": high["odd_relative_signals"],
        "even_relative_amplification": even_amplification,
        "odd_relative_amplification": odd_amplification,
        "quadratic_coefficient_relative_changes": quadratic_changes,
        "quadratic_coefficient_cosines": quadratic_cosines,
        "cubic_coefficient_relative_changes": cubic_changes,
        "cubic_coefficient_cosines": cubic_cosines,
        "quadratic_balanced_output_basis": quadratic_basis,
        "quadratic_balanced_singular_values": quadratic_singular,
        "cubic_balanced_output_basis": cubic_basis,
        "cubic_balanced_singular_values": cubic_singular,
    }
    return metrics, arrays


def _gate_checks(metrics: dict, gates: dict) -> dict:
    return {
        "pair_count": metrics["signed_pairs_per_amplitude"]
        == gates["signed_pairs_per_amplitude_equal"],
        "even_signal": metrics["median_even_relative_signal_at_0p01"]
        >= gates["median_even_relative_signal_at_0p01_min"],
        "even_amplification_min": metrics["median_even_relative_amplification"]
        >= gates["median_even_relative_amplification_min"],
        "even_amplification_max": metrics["median_even_relative_amplification"]
        <= gates["median_even_relative_amplification_max"],
        "odd_amplification_min": metrics["median_odd_relative_amplification"]
        >= gates["median_odd_relative_amplification_min"],
        "odd_amplification_max": metrics["median_odd_relative_amplification"]
        <= gates["median_odd_relative_amplification_max"],
        "quadratic_change": metrics[
            "maximum_quadratic_coefficient_relative_change"
        ]
        <= gates["maximum_quadratic_coefficient_relative_change"],
        "quadratic_cosine": metrics["minimum_quadratic_coefficient_cosine"]
        >= gates["minimum_quadratic_coefficient_cosine"],
        "cubic_change": metrics["maximum_cubic_coefficient_relative_change"]
        <= gates["maximum_cubic_coefficient_relative_change"],
        "cubic_cosine": metrics["minimum_cubic_coefficient_cosine"]
        >= gates["minimum_cubic_coefficient_cosine"],
        "quadratic_rank": metrics["quadratic_row_normalized_rank3_energy"]
        >= gates["quadratic_row_normalized_rank3_energy_min"],
        "cubic_rank": metrics["cubic_row_normalized_rank4_energy"]
        >= gates["cubic_row_normalized_rank4_energy_min"],
        "axial_boundary": metrics[
            "amplitude_0p02_maximum_transverse_distortion"
        ]
        >= gates["amplitude_0p02_transverse_distortion_min"],
        "truth_budget": metrics["new_truth_rate_evaluations"]
        == gates["new_truth_rate_evaluations_equal"],
        "retraction_budget": metrics["new_retractions"]
        == gates["new_retractions_equal"],
        "root_budget": metrics["new_roots"] == gates["new_roots_equal"],
        "propagation_budget": metrics["propagated_states"]
        == gates["propagated_states_equal"],
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
                    "scientific_status": (
                        "DIAGNOSED" if summary["passed"] else "INCONSISTENT"
                    ),
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
            "latest_source_parent_commit": MANIFEST_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def _run() -> dict:
    frozen = _validate_manifest(require_clean=True)
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("parity architecture audit is already canonicalized")
    metrics, arrays = _diagnose()
    checks = _gate_checks(
        metrics, frozen["contract"]["diagnostic_consistency_gates"]
    )
    passed = all(checks.values())
    classification = PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION
    authorized_next = (
        "definitions_only_active8_mixed_direction_parity_database_manifest"
        if passed
        else None
    )
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_json(CANONICAL_DIRECTORY / "metrics.json", {"checks": checks, **metrics})
    np.savez_compressed(
        CANONICAL_DIRECTORY / "parity_low_rank_architecture.npz", **arrays
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "post_result_architecture_diagnosis": True,
        "independent_validation_claimed": False,
        "selected_online_state_dimension": 470,
        "selected_active_nonlinear_input_dimension": manifest.ACTIVE_INPUT_DIMENSION,
        "selected_quadratic_output_rank": manifest.QUADRATIC_OUTPUT_RANK,
        "selected_cubic_output_rank": manifest.CUBIC_OUTPUT_RANK,
        "median_even_relative_signal_at_0p01": metrics[
            "median_even_relative_signal_at_0p01"
        ],
        "median_odd_relative_signal_at_0p01": metrics[
            "median_odd_relative_signal_at_0p01"
        ],
        "mixed_direction_coefficients_identified": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": authorized_next,
    }
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    _write_json(
        CANONICAL_DIRECTORY / "parent_lock.json",
        {
            "manifest_commit": MANIFEST_COMMIT,
            "manifest_parent": MANIFEST_PARENT,
            "manifest_tree": MANIFEST_TREE,
            "manifest_hashes": _checksums(manifest.ARTIFACT_DIRECTORY),
            "rate_0p005_hashes": _checksums(
                manifest.rate_0p005.CANONICAL_DIRECTORY
            ),
            "rate_0p01_hashes": _checksums(manifest.rate_0p01.CANONICAL_DIRECTORY),
            "rejected_0p02_hashes": _checksums(manifest.parent.CANONICAL_DIRECTORY),
        },
    )
    source_files = (THIS_RUNNER, THIS_TEST, manifest.THIS_RUNNER, manifest.THIS_TEST)
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "DIAGNOSED" if passed else "INCONSISTENT",
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "source_hashes": {
                relative: _sha(ROOT / relative) for relative in source_files
            },
            "python": sys.version,
            "platform": platform.platform(),
            "thread_environment": (
                manifest.rate_0p01.manifest.parent.chart_tools.coordinate_tools.THREAD_ENVIRONMENT
            ),
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
                "# Parity low-rank architecture audit WP10c9d6c7c3b5c4f25bi",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                "## Diagnosis",
                "",
                "This is a post-result architecture diagnosis, not independent validation. It performs no new truth evaluation, retraction, root, or propagation.",
                "",
                f"The median even/quadratic relative signal grows from `{metrics['median_even_relative_signal_at_0p005']:.6e}` at 0.005 to `{metrics['median_even_relative_signal_at_0p01']:.6e}` at 0.01. The median odd/cubic signal at 0.01 is `{metrics['median_odd_relative_signal_at_0p01']:.6e}`.",
                "",
                f"Rank-3 captures `{metrics['quadratic_row_normalized_rank3_energy']:.6e}` of balanced quadratic output energy; rank-4 captures `{metrics['cubic_row_normalized_rank4_energy']:.6e}` of balanced cubic output energy.",
                "",
                "The selected candidate keeps the exact 162 physical and 280 stable-memory updates, and models only the 28D departure nonlinearity from an active 8D input with low-rank quadratic and cubic outputs. The compressed full-polynomial upper bound is 588 coefficients before any cubic input-tensor compression.",
                "",
                f"Authorized next artifact: `{authorized_next}`. Mixed-direction coefficients, an online integrator, and a predictive cycle remain unvalidated.",
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
    print(json.dumps(_plain(_run()), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
