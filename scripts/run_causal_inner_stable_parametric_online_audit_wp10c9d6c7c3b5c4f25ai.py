#!/usr/bin/env python3
"""Audit aligned dissipative interpolation and the stable online kernel cost."""

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
import time

import numpy as np
from scipy.linalg import eigvals, expm, lu_factor, lu_solve


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_stable_parametric_online_manifest_wp10c9d6c7c3b5c4f25ah as manifest  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25ai"
MANIFEST_COMMIT = "756797cdd2a70632080e1e5ca156badd93ffbc1f"
MANIFEST_PARENT = "d4a240e4566db64ba3206618cfbe0159e3dab206"
MANIFEST_TREE = "b87af85c42ba396bd0c9da16c0c9b88621e6c555"

PASS_CLASSIFICATION = (
    "stable_parametric_kernel_and_cost_passed_"
    "nonlinear_unstable_bundle_database_manifest_authorized"
)
STRUCTURE_FAIL_CLASSIFICATION = (
    "stable_parametric_kernel_failed_coordinate_or_descriptor_reassessment_required"
)
COST_FAIL_CLASSIFICATION = (
    "stable_parametric_kernel_structural_pass_cost_failed_"
    "online_linear_algebra_optimization_required"
)

ARTIFACT = "causal_inner_stable_parametric_online_audit_wp10c9d6c7c3b5c4f25ai"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_stable_parametric_online_audit_"
    "wp10c9d6c7c3b5c4f25ai.py"
)
THIS_TEST = (
    "tests/test_causal_inner_stable_parametric_online_audit_"
    "wp10c9d6c7c3b5c4f25ai.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_STABLE_PARAMETRIC_ONLINE_"
    "AUDIT_WP10C9D6C7C3B5C4F25AI_2026-08-18.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

THREAD_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


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
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        if _sha(directory / name) != expected:
            raise RuntimeError(f"checksum mismatch: {directory / name}")
        recorded[name] = expected
    return recorded


def _validate_manifest(*, require_clean: bool) -> dict:
    if _git("rev-parse", MANIFEST_COMMIT) != MANIFEST_COMMIT:
        raise RuntimeError("stable-parametric manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("stable-parametric manifest lineage changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("stable-parametric manifest tree changed")
    hashes = _checksums(manifest.ARTIFACT_DIRECTORY)
    summary = _read(manifest.ARTIFACT_DIRECTORY / "summary.json")
    contract = _read(manifest.ARTIFACT_DIRECTORY / "contract.json")
    provenance = _read(manifest.ARTIFACT_DIRECTORY / "provenance.json")
    if (
        not summary["passed"]
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["stable_descriptor_dimension"]
        != manifest.STABLE_REDUCED_DIMENSION
        or summary["unstable_bundle_dimension"] != manifest.UNSTABLE_DIMENSION
    ):
        raise RuntimeError("stable-parametric execution authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"manifest source changed: {relative}")
    for name, expected in contract["parent_decisive_hashes"].items():
        if _sha(manifest.PARENT_DIRECTORY / name) != expected:
            raise RuntimeError(f"parent input changed: {name}")
    for name, expected in contract["fiber_decisive_hashes"].items():
        if _sha(manifest.FIBER_DIRECTORY / name) != expected:
            raise RuntimeError(f"fiber input changed: {name}")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("stable-parametric audit requires a clean tracked tree")
    for name, expected in THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    return {"summary": summary, "contract": contract, "hashes": hashes}


def _orthogonal_procrustes(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    left, _, right = np.linalg.svd(source.T @ target)
    return left @ right


def _descriptor_grid(
    operator_0: np.ndarray,
    metric_0: np.ndarray,
    operator_1: np.ndarray,
    metric_1: np.ndarray,
    parameters: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    K0 = metric_0 @ operator_0
    K1 = metric_1 @ operator_1
    arrays = {
        "parameter": np.asarray(parameters),
        "metric_minimum_eigenvalue": [],
        "metric_condition_number": [],
        "symmetric_dissipation_largest_eigenvalue": [],
        "spectral_abscissa_per_second": [],
        "descriptor_identity_relative_defect": [],
    }
    for parameter in parameters:
        metric = (1.0 - parameter) * metric_0 + parameter * metric_1
        generator = (1.0 - parameter) * K0 + parameter * K1
        metric_eigenvalues = np.linalg.eigvalsh(metric)
        symmetric = generator + generator.T
        operator = np.linalg.solve(metric, generator)
        identity_defect = np.linalg.norm(
            metric @ operator + operator.T @ metric - symmetric
        ) / max(float(np.linalg.norm(symmetric)), np.finfo(float).tiny)
        arrays["metric_minimum_eigenvalue"].append(metric_eigenvalues[0])
        arrays["metric_condition_number"].append(
            metric_eigenvalues[-1] / metric_eigenvalues[0]
        )
        arrays["symmetric_dissipation_largest_eigenvalue"].append(
            np.linalg.eigvalsh(symmetric)[-1]
        )
        arrays["spectral_abscissa_per_second"].append(
            np.max(np.real(eigvals(operator)))
        )
        arrays["descriptor_identity_relative_defect"].append(identity_defect)
    arrays = {name: np.asarray(value) for name, value in arrays.items()}
    endpoint_0 = np.linalg.solve(metric_0, K0)
    endpoint_1 = np.linalg.solve(metric_1, K1)
    metrics = {
        "minimum_metric_eigenvalue": float(
            np.min(arrays["metric_minimum_eigenvalue"])
        ),
        "maximum_metric_condition_number": float(
            np.max(arrays["metric_condition_number"])
        ),
        "maximum_symmetric_dissipation_eigenvalue": float(
            np.max(arrays["symmetric_dissipation_largest_eigenvalue"])
        ),
        "maximum_spectral_abscissa_per_second": float(
            np.max(arrays["spectral_abscissa_per_second"])
        ),
        "maximum_descriptor_identity_relative_defect": float(
            np.max(arrays["descriptor_identity_relative_defect"])
        ),
        "endpoint_0_operator_relative_defect": float(
            np.linalg.norm(endpoint_0 - operator_0)
            / max(float(np.linalg.norm(operator_0)), np.finfo(float).tiny)
        ),
        "endpoint_1_operator_relative_defect": float(
            np.linalg.norm(endpoint_1 - operator_1)
            / max(float(np.linalg.norm(operator_1)), np.finfo(float).tiny)
        ),
    }
    return metrics, arrays


def _unstable_grid(
    operator_0: np.ndarray,
    operator_1: np.ndarray,
    parameters: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    minimum_real_parts = []
    maximum_real_parts = []
    positive_counts = []
    for parameter in parameters:
        poles = eigvals((1.0 - parameter) * operator_0 + parameter * operator_1)
        minimum_real_parts.append(np.min(np.real(poles)))
        maximum_real_parts.append(np.max(np.real(poles)))
        positive_counts.append(np.count_nonzero(np.real(poles) > 1.0e-8))
    arrays = {
        "parameter": np.asarray(parameters),
        "minimum_real_part_per_second": np.asarray(minimum_real_parts),
        "maximum_real_part_per_second": np.asarray(maximum_real_parts),
        "positive_real_part_count": np.asarray(positive_counts),
    }
    return {
        "minimum_real_part_per_second": float(np.min(minimum_real_parts)),
        "maximum_real_part_per_second": float(np.max(maximum_real_parts)),
        "minimum_positive_real_part_count": int(np.min(positive_counts)),
        "maximum_positive_real_part_count": int(np.max(positive_counts)),
        "minimum_growth_timescale_seconds": float(1.0 / np.max(maximum_real_parts)),
        "maximum_growth_timescale_seconds": float(1.0 / np.min(minimum_real_parts)),
        "maximum_macrostep_linear_log_amplification": float(
            manifest.MINIMUM_MACROSTEP_SECONDS * np.max(maximum_real_parts)
        ),
    }, arrays


def _timed_median(function, repetitions: int) -> float:
    values = []
    for _ in range(repetitions):
        began = time.perf_counter()
        function()
        values.append(time.perf_counter() - began)
    return float(np.median(values))


def _benchmark_kernel(
    operator: np.ndarray,
    metric: np.ndarray,
    generator: np.ndarray,
    runtime: dict,
) -> dict:
    dimension = operator.shape[0]
    state = np.linspace(0.5, 1.5, dimension)
    timestep = runtime["minimum_average_macrostep_seconds"]
    implicit_matrix = metric - timestep * generator
    exponential_seconds = _timed_median(
        lambda: expm(timestep * operator),
        runtime["benchmark_exponential_repetitions"],
    )
    factor_seconds = _timed_median(
        lambda: lu_factor(implicit_matrix),
        runtime["benchmark_LU_factor_repetitions"],
    )
    factors = lu_factor(implicit_matrix)
    solve_seconds = _timed_median(
        lambda: lu_solve(factors, state),
        runtime["benchmark_LU_solve_repetitions"],
    )
    matvec_seconds = _timed_median(
        lambda: operator @ state,
        runtime["benchmark_matvec_repetitions"],
    )
    return {
        "stable_dimension": dimension,
        "macrostep_seconds": timestep,
        "exponential_median_wall_seconds": exponential_seconds,
        "LU_factor_median_wall_seconds": factor_seconds,
        "LU_solve_median_wall_seconds": solve_seconds,
        "matvec_median_wall_seconds": matvec_seconds,
        "projected_exponential_cycle_wall_seconds": (
            exponential_seconds * runtime["maximum_macrosteps"]
        ),
        "projected_exponential_cycle_wall_days": (
            exponential_seconds * runtime["maximum_macrosteps"] / 86_400.0
        ),
        "projected_exponential_fraction_of_wall_budget": (
            exponential_seconds
            * runtime["maximum_macrosteps"]
            / runtime["wall_budget_seconds"]
        ),
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
                        "CERTIFIED" if summary["passed"] else "REJECTED"
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
        raise RuntimeError("stable-parametric audit is already canonicalized")
    with np.load(
        manifest.PARENT_DIRECTORY / "decisive_model.npz", allow_pickle=False
    ) as source:
        model = {name: np.asarray(source[name]) for name in source.files}
    with np.load(
        manifest.FIBER_DIRECTORY / "decisive_fibers.npz", allow_pickle=False
    ) as source:
        fiber = {name: np.asarray(source[name]) for name in source.files}

    hidden_rotation = _orthogonal_procrustes(
        model["heldout_hidden_truth_trial"],
        model["primary_hidden_truth_trial"],
    )
    stable_transform = np.eye(manifest.STABLE_REDUCED_DIMENSION)
    stable_transform[
        manifest.CONSERVATIVE_DIMENSION :, manifest.CONSERVATIVE_DIMENSION :
    ] = hidden_rotation
    operator_0 = model["primary_stable_reduced_operator"]
    metric_0 = model["primary_metric"]
    operator_1 = (
        stable_transform.T
        @ model["heldout_stable_reduced_operator"]
        @ stable_transform
    )
    metric_1 = stable_transform.T @ model["heldout_metric"] @ stable_transform
    parameters = np.asarray(manifest.PARAMETER_GRID)
    descriptor_metrics, descriptor_arrays = _descriptor_grid(
        operator_0, metric_0, operator_1, metric_1, parameters
    )
    descriptor_metrics.update(
        {
            "hidden_alignment_orthogonality_defect": float(
                np.max(
                    np.abs(
                        hidden_rotation.T @ hidden_rotation
                        - np.eye(manifest.HIDDEN_DIMENSION)
                    )
                )
            ),
            "aligned_hidden_truth_trial_relative_mismatch": float(
                np.linalg.norm(
                    model["heldout_hidden_truth_trial"] @ hidden_rotation
                    - model["primary_hidden_truth_trial"]
                )
                / np.linalg.norm(model["primary_hidden_truth_trial"])
            ),
        }
    )

    unstable_rotation = fiber["heldout_alignment_rotation"]
    unstable_0 = fiber["primary_unstable_operator"]
    unstable_1 = (
        unstable_rotation.T
        @ fiber["heldout_unstable_operator"]
        @ unstable_rotation
    )
    unstable_metrics, unstable_arrays = _unstable_grid(
        unstable_0, unstable_1, parameters
    )
    unstable_metrics["alignment_orthogonality_defect"] = float(
        np.max(
            np.abs(
                unstable_rotation.T @ unstable_rotation
                - np.eye(manifest.UNSTABLE_DIMENSION)
            )
        )
    )
    unstable_metrics["operator_dimension"] = int(unstable_0.shape[0])

    gates = frozen["contract"]["binding_gates"]
    structural_passed = bool(
        descriptor_metrics["hidden_alignment_orthogonality_defect"]
        <= gates["hidden_alignment_orthogonality_defect_max"]
        and descriptor_metrics["minimum_metric_eigenvalue"]
        >= gates["stable_metric_minimum_eigenvalue_min"]
        and descriptor_metrics["maximum_metric_condition_number"]
        <= gates["stable_metric_condition_number_max"]
        and descriptor_metrics["maximum_symmetric_dissipation_eigenvalue"]
        <= gates["stable_symmetric_dissipation_largest_eigenvalue_max"]
        and descriptor_metrics["maximum_spectral_abscissa_per_second"]
        <= gates["stable_spectral_abscissa_per_second_max"]
        and descriptor_metrics["maximum_descriptor_identity_relative_defect"]
        <= gates["descriptor_identity_relative_defect_max"]
        and max(
            descriptor_metrics["endpoint_0_operator_relative_defect"],
            descriptor_metrics["endpoint_1_operator_relative_defect"],
        )
        <= gates["endpoint_operator_relative_defect_max"]
        and unstable_metrics["alignment_orthogonality_defect"]
        <= gates["unstable_alignment_orthogonality_defect_max"]
        and unstable_metrics["operator_dimension"]
        == gates["unstable_dimension_equal"]
        and unstable_metrics["minimum_positive_real_part_count"]
        == gates["unstable_positive_real_part_count_equal"]
        and unstable_metrics["maximum_positive_real_part_count"]
        == gates["unstable_positive_real_part_count_equal"]
        and unstable_metrics["minimum_real_part_per_second"]
        >= gates["unstable_minimum_real_part_per_second_min"]
    )

    runtime = frozen["contract"]["runtime_contract"]
    benchmark_points = {}
    for label, parameter in (("primary", 0.0), ("midpoint", 0.5), ("heldout", 1.0)):
        metric = (1.0 - parameter) * metric_0 + parameter * metric_1
        generator = (1.0 - parameter) * (metric_0 @ operator_0) + parameter * (
            metric_1 @ operator_1
        )
        operator = np.linalg.solve(metric, generator)
        benchmark_points[label] = _benchmark_kernel(
            operator, metric, generator, runtime
        )
    worst_benchmark = {
        name: max(item[name] for item in benchmark_points.values())
        for name in (
            "exponential_median_wall_seconds",
            "LU_factor_median_wall_seconds",
            "LU_solve_median_wall_seconds",
            "matvec_median_wall_seconds",
            "projected_exponential_cycle_wall_seconds",
            "projected_exponential_cycle_wall_days",
            "projected_exponential_fraction_of_wall_budget",
        )
    }
    cost_passed = bool(
        worst_benchmark["exponential_median_wall_seconds"]
        <= gates["stable_exponential_median_wall_seconds_max"]
        and worst_benchmark["LU_factor_median_wall_seconds"]
        <= gates["stable_LU_factor_median_wall_seconds_max"]
        and worst_benchmark["LU_solve_median_wall_seconds"]
        <= gates["stable_LU_solve_median_wall_seconds_max"]
        and worst_benchmark["matvec_median_wall_seconds"]
        <= gates["stable_matvec_median_wall_seconds_max"]
        and worst_benchmark["projected_exponential_cycle_wall_seconds"]
        <= gates["stable_kernel_projected_cycle_wall_seconds_max"]
    )

    passed = bool(structural_passed and cost_passed)
    if passed:
        classification = PASS_CLASSIFICATION
        authorized_next = (
            "definitions_only_nonlinear_unstable_bundle_offline_database_manifest"
        )
    elif structural_passed:
        classification = COST_FAIL_CLASSIFICATION
        authorized_next = None
    else:
        classification = STRUCTURE_FAIL_CLASSIFICATION
        authorized_next = None

    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_json(
        CANONICAL_DIRECTORY / "metrics.json",
        {
            "descriptor": descriptor_metrics,
            "unstable_bundle": unstable_metrics,
            "benchmark_points": benchmark_points,
            "worst_benchmark": worst_benchmark,
            "structural_passed": structural_passed,
            "cost_passed": cost_passed,
        },
    )
    np.savez_compressed(
        CANONICAL_DIRECTORY / "parametric_diagnostics.npz",
        hidden_alignment_rotation=hidden_rotation,
        stable_coordinate_transform=stable_transform,
        unstable_alignment_rotation=unstable_rotation,
        **{f"stable_{name}": value for name, value in descriptor_arrays.items()},
        **{f"unstable_{name}": value for name, value in unstable_arrays.items()},
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "structural_passed": structural_passed,
        "cost_passed": cost_passed,
        "stable_descriptor_dimension": manifest.STABLE_REDUCED_DIMENSION,
        "unstable_bundle_dimension": manifest.UNSTABLE_DIMENSION,
        "total_architecture_dimension": manifest.TOTAL_ARCHITECTURE_DIMENSION,
        "maximum_stable_spectral_abscissa_per_second": descriptor_metrics[
            "maximum_spectral_abscissa_per_second"
        ],
        "minimum_unstable_real_part_per_second": unstable_metrics[
            "minimum_real_part_per_second"
        ],
        "maximum_unstable_real_part_per_second": unstable_metrics[
            "maximum_real_part_per_second"
        ],
        "projected_stable_kernel_cycle_wall_seconds": worst_benchmark[
            "projected_exponential_cycle_wall_seconds"
        ],
        "projected_stable_kernel_cycle_wall_days": worst_benchmark[
            "projected_exponential_cycle_wall_days"
        ],
        "online_truth_calls_per_macrostep": 0,
        "unstable_bundle_linear_macro_propagation_authorized": False,
        "online_integrator_implementation_authorized": False,
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
            "manifest_package_hashes": _checksums(manifest.ARTIFACT_DIRECTORY),
            "parent_package_hashes": _checksums(manifest.PARENT_DIRECTORY),
            "fiber_package_hashes": _checksums(manifest.FIBER_DIRECTORY),
        },
    )
    source_files = (THIS_RUNNER, THIS_TEST, manifest.THIS_RUNNER, manifest.THIS_TEST)
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "CERTIFIED" if passed else "REJECTED",
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
            "thread_environment": THREAD_ENVIRONMENT,
        },
    )
    names = tuple(sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()))
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names
        ),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Stable parametric online audit WP10c9d6c7c3b5c4f25ai",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                f"Structural pass: `{structural_passed}`. Stable-kernel cost pass: `{cost_passed}`.",
                "",
                f"The aligned descriptor family has maximum stable spectral abscissa `{descriptor_metrics['maximum_spectral_abscissa_per_second']:.6e} s^-1`. All 28 separated modes retain positive growth, from `{unstable_metrics['minimum_real_part_per_second']:.6e}` to `{unstable_metrics['maximum_real_part_per_second']:.6e} s^-1`.",
                "",
                f"Recomputing the stable dense exponential at every one of 100,000 macrosteps projects to `{worst_benchmark['projected_exponential_cycle_wall_seconds']:.6f}` wall seconds (`{worst_benchmark['projected_exponential_cycle_wall_days']:.6e}` days).",
                "",
                f"Authorized next artifact: `{authorized_next}`. The unstable bundle may not be linearly macro-propagated, and no online solver or predictive cycle is authorized.",
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
