#!/usr/bin/env python3
"""Execute the frozen scale-aware derivative recovery for the exact 470 chart."""

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


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_exact_geometric_470_chart_derivative_recovery_manifest_wp10c9d6c7c3b5c4f25de1 as manifest  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25de2"
MANIFEST_COMMIT = "5a01f9de30d3fd47f7e7cf6454dbb34b88e099e2"
MANIFEST_PARENT = "2dad3286e5418c9eb17a095df54ae49198268942"
MANIFEST_TREE = "45ab9f600f17d6d753e763960e9a89126b8095d2"

PASS_CLASSIFICATION = (
    "exact_geometric_470_chart_derivative_recovered_"
    "primary_hidden_root_manifest_authorized"
)
FAIL_CLASSIFICATION = (
    "exact_geometric_470_chart_derivative_recovery_failed_hidden_root_blocked"
)
AUTHORIZED_NEXT = manifest.PASS_AUTHORIZED_NEXT

ARTIFACT = (
    "causal_inner_exact_geometric_470_chart_derivative_recovery_"
    "wp10c9d6c7c3b5c4f25de2"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_exact_geometric_470_chart_derivative_recovery_"
    "wp10c9d6c7c3b5c4f25de2.py"
)
THIS_TEST = (
    "tests/test_causal_inner_exact_geometric_470_chart_derivative_recovery_"
    "wp10c9d6c7c3b5c4f25de2.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_EXACT_GEOMETRIC_470_CHART_"
    "DERIVATIVE_RECOVERY_WP10C9D6C7C3B5C4F25DE2_2026-08-20.md"
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


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


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


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _validate_manifest(*, require_clean: bool) -> dict:
    if _git("rev-parse", MANIFEST_COMMIT) != MANIFEST_COMMIT:
        raise RuntimeError("derivative-recovery manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("derivative-recovery manifest lineage changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("derivative-recovery manifest tree changed")
    hashes = _checksums(manifest.CANONICAL_DIRECTORY)
    summary = _read(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = _read(
        manifest.CANONICAL_DIRECTORY / "derivative_recovery_contract.json"
    )
    provenance = _read(manifest.CANONICAL_DIRECTORY / "provenance.json")
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or summary["authorized_next"] != WORK_PACKAGE
        or not summary["parent_negative_certificate_preserved"]
        or summary["branch_root_execution_authorized"]
        or summary["sealed_16ms_opened"]
        or contract["prospective_execution"]["work_package"] != WORK_PACKAGE
        or not contract["preserved_negative_certificate"]["remains_failed"]
    ):
        raise RuntimeError("derivative-recovery authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"derivative-recovery manifest source changed: {relative}")
    decisive = contract["decisive_input_hashes"]
    for name, path in (
        ("parent_summary", manifest.parent.CANONICAL_DIRECTORY / "summary.json"),
        (
            "parent_metrics",
            manifest.parent.CANONICAL_DIRECTORY / "exact_chart_metrics.json",
        ),
        (
            "parent_arrays",
            manifest.parent.CANONICAL_DIRECTORY / "exact_chart_arrays.npz",
        ),
    ):
        if _sha(path) != decisive[name]:
            raise RuntimeError(f"decisive derivative input changed: {path}")
    parent = manifest._validate_parent(require_clean=False)
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("derivative-recovery audit requires a clean tracked tree")
    for name, expected in provenance["thread_environment"].items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    return {
        "hashes": hashes,
        "summary": summary,
        "contract": contract,
        "parent": parent,
    }


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(left) - np.asarray(right))
        / max(np.linalg.norm(right), np.finfo(float).tiny)
    )


def _central_difference(model, anchor_delta, physical, coordinate, step):
    plus_state = manifest.parent._state_from_delta(
        model, anchor_delta + step * physical
    )
    minus_state = manifest.parent._state_from_delta(
        model, anchor_delta - step * physical
    )
    plus_coordinate, _ = model.coordinate(plus_state)
    minus_coordinate, _ = model.coordinate(minus_state)
    difference = np.asarray(plus_coordinate) - np.asarray(minus_coordinate)
    finite = difference / (2.0 * step)
    return {
        "plus": np.asarray(plus_coordinate),
        "minus": np.asarray(minus_coordinate),
        "finite": finite,
        "signal_norm": float(np.linalg.norm(difference)),
        "relative_defect": _relative(finite, coordinate),
    }


def _roundoff_statistics(steps: np.ndarray, defects: np.ndarray) -> dict:
    log_steps = np.log(np.asarray(steps, dtype=float))
    log_defects = np.log(np.asarray(defects, dtype=float))
    slope, intercept = np.polyfit(log_steps, log_defects, 1)
    fitted = slope * log_steps + intercept
    residual = float(np.sum((log_defects - fitted) ** 2))
    total = float(np.sum((log_defects - np.mean(log_defects)) ** 2))
    r_squared = 1.0 - residual / max(total, np.finfo(float).tiny)
    scaled = np.asarray(steps, dtype=float) * np.asarray(defects, dtype=float)
    coefficient = float(np.std(scaled) / max(np.mean(scaled), np.finfo(float).tiny))
    return {
        "loglog_slope": float(slope),
        "loglog_intercept": float(intercept),
        "loglog_R_squared": float(r_squared),
        "h_times_defect": scaled,
        "h_times_defect_coefficient_of_variation": coefficient,
    }


def _execute() -> tuple[dict, dict[str, np.ndarray]]:
    began = time.perf_counter()
    arrays = _load_npz(
        manifest.parent.CANONICAL_DIRECTORY / "exact_chart_arrays.npz"
    )
    parent_payload = _read(
        manifest.parent.CANONICAL_DIRECTORY / "exact_chart_metrics.json"
    )
    model, _, _ = manifest.parent._model_and_inputs()
    anchor = np.asarray(arrays["anchor_primitive_state"])
    anchor_delta = manifest.parent._delta(model, anchor)
    augmented = np.asarray(arrays["anchor_augmented_chart_jacobian"])
    coordinates = np.asarray(arrays["coordinate_test_directions"])
    physicals = np.asarray(arrays["linear_physical_test_directions"])
    if coordinates.shape != (
        manifest.parent.DIRECTION_COUNT,
        manifest.parent.COORDINATE_DIMENSION,
    ) or physicals.shape != (
        manifest.parent.DIRECTION_COUNT,
        manifest.parent.PHYSICAL_DIMENSION,
    ):
        raise RuntimeError("frozen derivative direction shapes changed")

    algebraic_residuals = []
    algebraic_defects = []
    for coordinate, physical in zip(coordinates, physicals, strict=True):
        right = np.concatenate(
            (coordinate, np.zeros(manifest.parent.GAUGE_DIMENSION))
        )
        residual = augmented @ physical - right
        algebraic_residuals.append(residual)
        algebraic_defects.append(_relative(augmented @ physical, right))

    common_records = []
    common_plus = []
    common_minus = []
    common_finite = []
    for index, (coordinate, physical) in enumerate(
        zip(coordinates, physicals, strict=True)
    ):
        record = _central_difference(
            model,
            anchor_delta,
            physical,
            coordinate,
            manifest.COMMON_SCALE_STEP,
        )
        common_records.append(
            {
                "direction_index": index,
                "step": manifest.COMMON_SCALE_STEP,
                "signal_norm": record["signal_norm"],
                "relative_defect": record["relative_defect"],
            }
        )
        common_plus.append(record["plus"])
        common_minus.append(record["minus"])
        common_finite.append(record["finite"])
        print(
            f"f25de2: common direction={index} "
            f"defect={record['relative_defect']:.6e}",
            flush=True,
        )

    failed_index = manifest.FAILED_DIRECTION_INDEX
    ladder_records = []
    ladder_plus = []
    ladder_minus = []
    ladder_finite = []
    for step in manifest.ROUND_OFF_STEPS:
        if step == manifest.COMMON_SCALE_STEP:
            record = {
                "plus": common_plus[failed_index],
                "minus": common_minus[failed_index],
                "finite": common_finite[failed_index],
                "signal_norm": common_records[failed_index]["signal_norm"],
                "relative_defect": common_records[failed_index][
                    "relative_defect"
                ],
            }
        else:
            record = _central_difference(
                model,
                anchor_delta,
                physicals[failed_index],
                coordinates[failed_index],
                step,
            )
        ladder_records.append(
            {
                "step": step,
                "signal_norm": record["signal_norm"],
                "relative_defect": record["relative_defect"],
            }
        )
        ladder_plus.append(record["plus"])
        ladder_minus.append(record["minus"])
        ladder_finite.append(record["finite"])
        print(
            f"f25de2: ladder step={step:.1e} "
            f"defect={record['relative_defect']:.6e}",
            flush=True,
        )

    steps = np.asarray([record["step"] for record in ladder_records])
    defects = np.asarray(
        [record["relative_defect"] for record in ladder_records]
    )
    roundoff = _roundoff_statistics(steps, defects)
    original_defect = float(
        parent_payload["metrics"]["implicit_derivative"]["relative_defects"][
            failed_index
        ]
    )
    original_index = list(manifest.ROUND_OFF_STEPS).index(
        manifest.parent.DERIVATIVE_STEP
    )
    reproduced = float(defects[original_index])
    original_reproduction = abs(reproduced - original_defect) / max(
        abs(original_defect), np.finfo(float).tiny
    )
    coordinate_evaluations = 2 * (
        manifest.parent.DIRECTION_COUNT + len(manifest.ROUND_OFF_STEPS) - 1
    )
    metrics = {
        "direction_count": manifest.parent.DIRECTION_COUNT,
        "algebraic_relative_defects": np.asarray(algebraic_defects),
        "maximum_algebraic_relative_defect": float(np.max(algebraic_defects)),
        "common_scale_step": manifest.COMMON_SCALE_STEP,
        "common_scale_records": common_records,
        "maximum_common_scale_relative_defect": float(
            max(record["relative_defect"] for record in common_records)
        ),
        "minimum_common_scale_signal_norm": float(
            min(record["signal_norm"] for record in common_records)
        ),
        "roundoff_direction": {
            "direction_index": failed_index,
            "family": manifest.FAILED_DIRECTION_FAMILY,
            "source_index": manifest.FAILED_DIRECTION_SOURCE_INDEX,
        },
        "roundoff_ladder": ladder_records,
        "roundoff_statistics": roundoff,
        "original_step": manifest.parent.DERIVATIVE_STEP,
        "original_committed_relative_defect": original_defect,
        "original_reproduced_relative_defect": reproduced,
        "original_defect_reproduction_relative": original_reproduction,
        "parent_non_derivative_checks_preserved": bool(
            all(
                passed
                for name, passed in parent_payload["checks"].items()
                if name != "implicit_derivative"
            )
        ),
        "new_coordinate_evaluations": coordinate_evaluations,
        "new_coordinate_jacobian_assemblies": 0,
        "new_coordinate_retractions": 0,
        "new_exact_fixed_Q_rate_evaluations": 0,
        "new_complete_generator_assemblies": 0,
        "new_intrinsic_hidden_roots": 0,
        "propagated_states": 0,
        "sealed_16ms_truth_calls": 0,
        "total_wall_seconds": time.perf_counter() - began,
    }
    output_arrays = {
        "steps": steps,
        "coordinate_directions": coordinates,
        "linear_physical_directions": physicals,
        "algebraic_residuals": np.asarray(algebraic_residuals),
        "common_plus_coordinates": np.asarray(common_plus),
        "common_minus_coordinates": np.asarray(common_minus),
        "common_finite_derivatives": np.asarray(common_finite),
        "ladder_plus_coordinates": np.asarray(ladder_plus),
        "ladder_minus_coordinates": np.asarray(ladder_minus),
        "ladder_finite_derivatives": np.asarray(ladder_finite),
    }
    return metrics, output_arrays


def _checks(metrics: dict) -> dict[str, bool]:
    statistics = metrics["roundoff_statistics"]
    budget = 2 * (
        manifest.parent.DIRECTION_COUNT + len(manifest.ROUND_OFF_STEPS)
    )
    return {
        "all_eight_algebraic_identities": metrics[
            "maximum_algebraic_relative_defect"
        ]
        <= manifest.ALGEBRAIC_RELATIVE_DEFECT_GATE,
        "all_eight_common_scale_directions": metrics[
            "maximum_common_scale_relative_defect"
        ]
        <= manifest.COMMON_SCALE_RELATIVE_DEFECT_GATE,
        "strong_common_scale_accuracy": metrics[
            "maximum_common_scale_relative_defect"
        ]
        <= manifest.COMMON_SCALE_BEST_DEFECT_GATE,
        "common_scale_signal": metrics["minimum_common_scale_signal_norm"]
        >= manifest.MINIMUM_COMMON_SIGNAL_NORM,
        "roundoff_slope": manifest.ROUND_OFF_SLOPE_MIN
        <= statistics["loglog_slope"]
        <= manifest.ROUND_OFF_SLOPE_MAX,
        "roundoff_R_squared": statistics["loglog_R_squared"]
        >= manifest.ROUND_OFF_R_SQUARED_MIN,
        "roundoff_scaled_coefficient": statistics[
            "h_times_defect_coefficient_of_variation"
        ]
        <= manifest.ROUND_OFF_COEFFICIENT_OF_VARIATION_MAX,
        "original_defect_reproduction": metrics[
            "original_defect_reproduction_relative"
        ]
        <= manifest.ORIGINAL_DEFECT_REPRODUCTION_RELATIVE_GATE,
        "parent_non_derivative_checks": metrics[
            "parent_non_derivative_checks_preserved"
        ],
        "coordinate_evaluation_budget": metrics["new_coordinate_evaluations"]
        <= budget,
        "coordinate_jacobian_budget": metrics[
            "new_coordinate_jacobian_assemblies"
        ]
        == 0,
        "retraction_budget": metrics["new_coordinate_retractions"] == 0,
        "rate_budget": metrics["new_exact_fixed_Q_rate_evaluations"] == 0,
        "generator_budget": metrics["new_complete_generator_assemblies"] == 0,
        "root_budget": metrics["new_intrinsic_hidden_roots"] == 0,
        "propagation_budget": metrics["propagated_states"] == 0,
        "sealed_budget": metrics["sealed_16ms_truth_calls"] == 0,
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
        raise RuntimeError("derivative-recovery audit already exists")
    metrics, arrays = _execute()
    checks = _checks(metrics)
    passed = bool(all(checks.values()))
    classification = PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_npz(CANONICAL_DIRECTORY / "derivative_recovery_arrays.npz", arrays)
    _write_json(
        CANONICAL_DIRECTORY / "derivative_recovery_metrics.json",
        {"metrics": metrics, "checks": checks, "passed": passed},
    )
    _write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "manifest_commit": MANIFEST_COMMIT,
            "manifest_parent": MANIFEST_PARENT,
            "manifest_tree": MANIFEST_TREE,
            "manifest_hashes": frozen["hashes"],
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "parent_negative_certificate_preserved": True,
        "parent_chart_preflight_reclassified": False,
        "maximum_algebraic_relative_defect": metrics[
            "maximum_algebraic_relative_defect"
        ],
        "maximum_common_scale_relative_defect": metrics[
            "maximum_common_scale_relative_defect"
        ],
        "minimum_common_scale_signal_norm": metrics[
            "minimum_common_scale_signal_norm"
        ],
        "roundoff_loglog_slope": metrics["roundoff_statistics"][
            "loglog_slope"
        ],
        "roundoff_loglog_R_squared": metrics["roundoff_statistics"][
            "loglog_R_squared"
        ],
        "roundoff_h_times_defect_coefficient_of_variation": metrics[
            "roundoff_statistics"
        ]["h_times_defect_coefficient_of_variation"],
        "new_coordinate_evaluations": metrics["new_coordinate_evaluations"],
        "new_exact_fixed_Q_rate_evaluations": 0,
        "new_complete_generator_assemblies": 0,
        "new_intrinsic_hidden_roots": 0,
        "propagated_states": 0,
        "sealed_16ms_opened": False,
        "branch_root_execution_authorized": False,
        "primary_hidden_root_manifest_authorized": passed,
        "online_solver_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT if passed else None,
    }
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    source_files = (
        THIS_RUNNER,
        THIS_TEST,
        manifest.THIS_RUNNER,
        manifest.THIS_TEST,
        manifest.parent.THIS_RUNNER,
        manifest.parent.THIS_TEST,
    )
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "CERTIFIED" if passed else "REJECTED",
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
                for name in manifest.parent.manifest.parent.field_manifest.training._thread_environment()
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
                "# Exact 470-chart derivative recovery WP10c9d6c7c3b5c4f25de2",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                f"All eight algebraic derivative identities have maximum relative defect `{metrics['maximum_algebraic_relative_defect']:.6e}`. At the prospectively frozen common scale h=3e-3, the maximum finite-difference defect is `{metrics['maximum_common_scale_relative_defect']:.6e}` and the minimum coordinate signal norm is `{metrics['minimum_common_scale_signal_norm']:.6e}`.",
                "",
                f"For the previously failed macro-53 direction, the six-step ladder has log-log slope `{metrics['roundoff_statistics']['loglog_slope']:.6f}`, R-squared `{metrics['roundoff_statistics']['loglog_R_squared']:.6f}`, and h-times-defect coefficient of variation `{metrics['roundoff_statistics']['h_times_defect_coefficient_of_variation']:.6f}`.",
                "",
                "The f25de failure remains preserved: this is a new prospective derivative certificate, not a retrospective gate relaxation. No rate, generator, retraction, root, propagated state, or sealed-state truth call occurred.",
                "",
                (
                    f"Passing authorizes only the definitions-only primary hidden-root manifest `{AUTHORIZED_NEXT}`; it does not authorize root execution."
                    if passed
                    else "Failure leaves the hidden root blocked and authorizes no successor."
                ),
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
    summary = _run()
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
