#!/usr/bin/env python3
"""Diagnose the terminal 470-chart condition boundary without propagation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_coordinate_chart_conditioning_diagnosis_manifest_wp10c9d6c7c3b5c4f25fia as manifest  # noqa: E402


parent = manifest.parent
SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25fib"
METRIC_CLASSIFICATION = (
    "terminal_raw_condition_crossing_is_coordinate_metric_artifact"
)
TANGENT_CLASSIFICATION = (
    "terminal_raw_condition_crossing_requires_block_tangent_atlas"
)
INTRINSIC_CLASSIFICATION = "terminal_coordinate_intrinsic_degeneracy_detected"
METHOD_FAILURE_CLASSIFICATION = "terminal_conditioning_diagnosis_method_failed"
AUTHORIZED_METRIC_NEXT = (
    "WP10c9d6c7c3b5c4f25fic_conservative_metric_chart_atlas_manifest"
)
AUTHORIZED_TANGENT_NEXT = (
    "WP10c9d6c7c3b5c4f25fic_conservative_tangent_chart_atlas_manifest"
)
ARTIFACT = (
    "causal_inner_coordinate_chart_conditioning_diagnosis_"
    "wp10c9d6c7c3b5c4f25fib"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_COORDINATE_CHART_CONDITIONING_"
    "DIAGNOSIS_WP10C9D6C7C3B5C4F25FIB_2026-08-24.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_coordinate_chart_conditioning_diagnosis_"
    "wp10c9d6c7c3b5c4f25fib.py"
)
THIS_TEST = (
    "tests/test_causal_inner_coordinate_chart_conditioning_diagnosis_"
    "wp10c9d6c7c3b5c4f25fib.py"
)
RATE_ACTION_RELATIVE_TOLERANCE = 1.0e-10


def _helper():
    return manifest._helper()


def _validate_manifest(*, require_clean: bool) -> dict:
    helper = _helper()
    hashes = helper._validate_checksums(manifest.CANONICAL_DIRECTORY)
    summary = helper._read(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = helper._read(
        manifest.CANONICAL_DIRECTORY / "diagnosis_contract.json"
    )
    witness = helper._read(manifest.CANONICAL_DIRECTORY / "witness_lock.json")
    if (
        summary["classification"] != manifest.CLASSIFICATION
        or not summary["passed"]
        or not summary["definitions_only"]
        or not summary["conditioning_diagnosis_authorized"]
        or summary["conditioning_diagnosis_executed"]
        or summary["trajectory_authorized"]
        or summary["authorized_next"] != manifest.AUTHORIZED_NEXT
        or contract["authorized_execution"] != manifest.AUTHORIZED_NEXT
        or contract["scope"]["new_exact_coordinate_jacobians"] != 6
        or contract["scope"]["new_exact_free_field_calls"] != 0
        or contract["scope"]["new_retractions"] != 0
        or contract["scope"]["new_trajectory_segments"] != 0
        or not witness["all_nonconditioning_physical_gates_passed"]
    ):
        raise RuntimeError("conditioning diagnosis authorization changed")
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("conditioning diagnosis requires a clean tracked tree")
    return {
        "hashes": hashes,
        "summary": summary,
        "contract": contract,
        "witness": witness,
    }


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    return float(
        np.linalg.norm(a - b)
        / max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), np.finfo(float).tiny)
    )


def _condition(singular: np.ndarray) -> float:
    values = np.asarray(singular, dtype=float)
    return float(values[0] / values[-1])


def _row_equilibrate(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(value, axis=1)
    if np.any(norms <= np.finfo(float).tiny):
        raise RuntimeError("coordinate Jacobian contains a zero row")
    return value / norms[:, None], norms


def _whiten_block(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    value = np.asarray(matrix, dtype=float)
    left, singular, right = np.linalg.svd(value, full_matrices=False)
    if len(singular) != value.shape[0] or singular[-1] <= np.finfo(float).tiny:
        raise RuntimeError("coordinate block lost row rank")
    transform = (left * (1.0 / singular)[None, :]) @ left.T
    whitened = transform @ value
    closure = float(
        np.linalg.norm(
            whitened @ whitened.T - np.eye(value.shape[0]), ord=np.inf
        )
    )
    return whitened, transform, closure, right


def _principal_angle_metrics(
    left_row_basis: np.ndarray, right_row_basis: np.ndarray
) -> dict:
    cosines = np.linalg.svd(
        np.asarray(left_row_basis) @ np.asarray(right_row_basis).T,
        compute_uv=False,
    )
    maximum_cosine = float(np.clip(cosines[0], 0.0, 1.0))
    return {
        "maximum_principal_cosine": maximum_cosine,
        "minimum_principal_angle_radians": float(np.arccos(maximum_cosine)),
    }


def _diagnose(lock: dict) -> tuple[dict, dict[str, np.ndarray]]:
    helper = _helper()
    witnesses = helper._load_npz(
        manifest.CANONICAL_DIRECTORY / "conditioning_witnesses.npz"
    )
    states = np.asarray(witnesses["primitive_states"], dtype=float)
    saved_rates = np.asarray(
        witnesses["scaled_free_rates560_per_s"], dtype=float
    )
    saved_coordinate_rates = np.asarray(
        witnesses["coordinate_free_rates470_per_s"], dtype=float
    )
    saved_conditions = np.asarray(witnesses["saved_condition_numbers"], dtype=float)
    attempts = np.asarray(witnesses["attempt_indices"], dtype=int)
    if (
        states.shape != (6, 112, 5)
        or saved_rates.shape != (6, manifest.PHYSICAL_COLUMNS)
        or saved_coordinate_rates.shape != (6, manifest.COORDINATE_ROWS)
        or tuple(attempts) != manifest.WITNESS_ATTEMPTS
    ):
        raise RuntimeError("conditioning witness arrays changed")

    inputs = parent.source._initial_inputs()
    model = inputs["model"]
    exact_chart = parent.source.arclength._exact_chart()
    block_slices = {
        "physical": slice(0, manifest.PHYSICAL_ROWS),
        "memory": slice(
            manifest.PHYSICAL_ROWS,
            manifest.PHYSICAL_ROWS + manifest.MEMORY_ROWS,
        ),
        "departure": slice(
            manifest.PHYSICAL_ROWS + manifest.MEMORY_ROWS,
            manifest.COORDINATE_ROWS,
        ),
    }

    records = []
    raw_spectra = []
    row_spectra = []
    block_spectra = []
    raw_left_critical = []
    raw_right_critical = []
    row_norms_all = []
    terminal_accepted_transform = None
    began = time.perf_counter()
    for index, (attempt, state) in enumerate(zip(attempts, states, strict=True)):
        jacobian, coordinate_metrics = exact_chart._coordinate_jacobian(model, state)
        left, raw_singular, right = np.linalg.svd(jacobian, full_matrices=False)
        raw_condition = _condition(raw_singular)
        condition_reproduction = abs(raw_condition - saved_conditions[index]) / max(
            raw_condition, saved_conditions[index], np.finfo(float).tiny
        )
        rate_defect = _relative(
            jacobian @ saved_rates[index], saved_coordinate_rates[index]
        )

        row_matrix, row_norms = _row_equilibrate(jacobian)
        row_singular = np.linalg.svd(row_matrix, compute_uv=False)
        whitened_blocks = []
        transforms = []
        closures = {}
        row_bases = {}
        block_conditions = {}
        for name, selected in block_slices.items():
            whitened, transform, closure, right_basis = _whiten_block(
                jacobian[selected]
            )
            whitened_blocks.append(whitened)
            transforms.append(transform)
            closures[name] = closure
            row_bases[name] = right_basis
            block_conditions[name] = float(np.linalg.cond(jacobian[selected]))
        block_matrix = np.vstack(whitened_blocks)
        block_singular = np.linalg.svd(block_matrix, compute_uv=False)
        block_transform = np.zeros(
            (manifest.COORDINATE_ROWS, manifest.COORDINATE_ROWS), dtype=float
        )
        offset = 0
        for transform in transforms:
            stop = offset + transform.shape[0]
            block_transform[offset:stop, offset:stop] = transform
            offset = stop
        if int(attempt) == 82:
            terminal_accepted_transform = block_transform.copy()

        records.append({
            "attempt_index": int(attempt),
            "saved_condition_number": float(saved_conditions[index]),
            "raw_rank": int(np.linalg.matrix_rank(jacobian)),
            "raw_condition_number": raw_condition,
            "raw_minimum_singular_value": float(raw_singular[-1]),
            "raw_maximum_singular_value": float(raw_singular[0]),
            "condition_reproduction_relative_defect": condition_reproduction,
            "coordinate_rate_action_relative_defect": rate_defect,
            "minimum_row_norm": float(np.min(row_norms)),
            "maximum_row_norm": float(np.max(row_norms)),
            "row_norm_ratio": float(np.max(row_norms) / np.min(row_norms)),
            "row_equilibrated_condition_number": _condition(row_singular),
            "row_equilibrated_minimum_singular_value": float(row_singular[-1]),
            "block_condition_numbers": block_conditions,
            "block_whitening_closure_defects": closures,
            "maximum_block_whitening_closure_defect": float(max(closures.values())),
            "block_whitened_condition_number": _condition(block_singular),
            "block_whitened_minimum_singular_value": float(block_singular[-1]),
            "physical_memory_angles": _principal_angle_metrics(
                row_bases["physical"], row_bases["memory"]
            ),
            "physical_departure_angles": _principal_angle_metrics(
                row_bases["physical"], row_bases["departure"]
            ),
            "memory_departure_angles": _principal_angle_metrics(
                row_bases["memory"], row_bases["departure"]
            ),
            "coordinate_reconstruction_relative_defect": float(
                coordinate_metrics["coordinate_reconstruction_relative_defect"]
            ),
        })
        raw_spectra.append(raw_singular)
        row_spectra.append(row_singular)
        block_spectra.append(block_singular)
        raw_left_critical.append(left[:, -1])
        raw_right_critical.append(right[-1])
        row_norms_all.append(row_norms)

    if terminal_accepted_transform is None:
        raise RuntimeError("terminal accepted witness was not found")
    execution_wall = float(time.perf_counter() - began)
    raw_reproduction = max(
        item["condition_reproduction_relative_defect"] for item in records
    )
    rate_action = max(
        item["coordinate_rate_action_relative_defect"] for item in records
    )
    maximum_closure = max(
        item["maximum_block_whitening_closure_defect"] for item in records
    )
    maximum_row_condition = max(
        item["row_equilibrated_condition_number"] for item in records
    )
    maximum_block_condition = max(
        item["block_whitened_condition_number"] for item in records
    )
    block_minimum = np.asarray(
        [item["block_whitened_minimum_singular_value"] for item in records]
    )
    terminal_normalized_ratio = float(block_minimum[-1] / block_minimum[0])
    rank_passed = all(item["raw_rank"] == manifest.COORDINATE_ROWS for item in records)
    method_passed = bool(
        rank_passed
        and raw_reproduction
        <= manifest.RECOMPUTED_CONDITION_RELATIVE_TOLERANCE
        and rate_action <= RATE_ACTION_RELATIVE_TOLERANCE
        and maximum_closure <= manifest.MAXIMUM_WHITENING_CLOSURE_DEFECT
        and lock["witness"]["all_nonconditioning_physical_gates_passed"]
        and execution_wall <= 3600.0 * manifest.MAXIMUM_EXECUTION_WALL_HOURS
    )
    row_passed = bool(
        maximum_row_condition <= manifest.MAXIMUM_ROW_EQUILIBRATED_CONDITION
    )
    block_passed = bool(
        maximum_block_condition <= manifest.MAXIMUM_BLOCK_WHITENED_CONDITION
        and terminal_normalized_ratio
        >= manifest.MINIMUM_TERMINAL_NORMALIZED_SINGULAR_RATIO
    )
    if not method_passed:
        classification = METHOD_FAILURE_CLASSIFICATION
        authorized_next = None
    elif row_passed and block_passed:
        classification = METRIC_CLASSIFICATION
        authorized_next = AUTHORIZED_METRIC_NEXT
    elif block_passed:
        classification = TANGENT_CLASSIFICATION
        authorized_next = AUTHORIZED_TANGENT_NEXT
    else:
        classification = INTRINSIC_CLASSIFICATION
        authorized_next = None
    atlas_supported = bool(method_passed and block_passed)
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": method_passed,
        "atlas_supported": atlas_supported,
        "authorized_next": authorized_next,
        "records": records,
        "gate_values": {
            "witness_count": len(records),
            "new_exact_coordinate_jacobians": len(records),
            "new_exact_free_field_calls": 0,
            "new_retractions": 0,
            "new_trajectory_segments": 0,
            "new_physical_time_seconds": 0.0,
            "maximum_raw_condition_reproduction_relative_defect": raw_reproduction,
            "maximum_coordinate_rate_action_relative_defect": rate_action,
            "maximum_block_whitening_closure_defect": maximum_closure,
            "maximum_row_equilibrated_condition_number": maximum_row_condition,
            "maximum_block_whitened_condition_number": maximum_block_condition,
            "terminal_normalized_block_singular_ratio": terminal_normalized_ratio,
            "rank_passed": rank_passed,
            "method_passed": method_passed,
            "row_equilibrated_gate_passed": row_passed,
            "block_whitened_gate_passed": block_passed,
            "execution_wall_seconds": execution_wall,
        },
        "interpretation": (
            "The historical raw condition rejection remains binding. An atlas "
            "is supported only when the saved Jacobians reproduce exactly and "
            "their row- or block-metric forms remain full-rank and bounded."
        ),
        "input_lock": {
            "manifest_hashes": lock["hashes"],
            "manifest_classification": lock["summary"]["classification"],
        },
    }
    arrays = {
        "attempt_indices": attempts,
        "raw_singular_values": np.stack(raw_spectra),
        "row_equilibrated_singular_values": np.stack(row_spectra),
        "block_whitened_singular_values": np.stack(block_spectra),
        "raw_critical_left_vectors": np.stack(raw_left_critical),
        "raw_critical_right_vectors": np.stack(raw_right_critical),
        "coordinate_row_norms": np.stack(row_norms_all),
        "terminal_accepted_block_transform470x470": terminal_accepted_transform,
    }
    return metrics, arrays


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def _update_catalog(summary: dict) -> None:
    helper = _helper()
    manifest_path = ROOT / "results/manifests/canonical_artifacts.csv"
    summary_path = ROOT / "results/manifests/canonical_summary.json"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    status = "SUPPORTED" if summary["passed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({
                "case": ARTIFACT,
                "path": str(path.relative_to(ROOT)),
                "bytes": str(path.stat().st_size),
                "sha256": helper._sha(path),
                "scientific_status": status,
            })
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case", "path", "bytes", "sha256", "scientific_status"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = helper._read(summary_path)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": summary["passed"],
    }
    catalog.update({
        "case_count": len({row["case"] for row in rows}),
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "all_payload_hashes_recorded": True,
        "latest_source_parent_commit": helper._git("rev-parse", "HEAD"),
        "latest_work_package": WORK_PACKAGE,
    })
    helper._write_json(summary_path, catalog)


def _canonicalize(metrics: dict, arrays: dict, lock: dict) -> dict:
    helper = _helper()
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("conditioning diagnosis result already exists")
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "conditioning_metrics.json", metrics)
    _save_npz(CANONICAL_DIRECTORY / "conditioning_arrays.npz", arrays)
    helper._write_json(CANONICAL_DIRECTORY / "input_lock.json", {
        "manifest_hashes": lock["hashes"],
        "manifest_classification": lock["summary"]["classification"],
    })
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": bool(metrics["passed"]),
        "atlas_supported": bool(metrics["atlas_supported"]),
        "historical_raw_condition_rejection_preserved": True,
        "new_trajectory": False,
        "cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": metrics["authorized_next"],
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    helper._write_json(CANONICAL_DIRECTORY / "provenance.json", {
        "runner": THIS_RUNNER,
        "test": THIS_TEST,
        "implementation_commit": helper._git("rev-parse", "HEAD"),
        "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"),
        "source_hashes": {
            THIS_RUNNER: helper._sha(ROOT / THIS_RUNNER),
            THIS_TEST: helper._sha(ROOT / THIS_TEST),
            manifest.THIS_RUNNER: helper._sha(ROOT / manifest.THIS_RUNNER),
            parent.THIS_RUNNER: helper._sha(ROOT / parent.THIS_RUNNER),
        },
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
    })
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{helper._sha(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
    values = metrics["gate_values"]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join((
            "# Coordinate-chart conditioning diagnosis",
            "",
            f"Classification: `{metrics['classification']}`.",
            "",
            f"Six saved terminal Jacobians were recomputed without a free-field call, retraction, or propagated state. The maximum raw-condition reproduction defect was `{values['maximum_raw_condition_reproduction_relative_defect']:.6e}` and the maximum saved-rate action defect was `{values['maximum_coordinate_rate_action_relative_defect']:.6e}`.",
            "",
            f"The maximum row-equilibrated condition number was `{values['maximum_row_equilibrated_condition_number']:.6e}` and the maximum independently block-whitened condition number was `{values['maximum_block_whitened_condition_number']:.6e}`. The terminal normalized block singular ratio was `{values['terminal_normalized_block_singular_ratio']:.6e}`.",
            "",
            "The historical raw 2500 gate and the rejected continuation remain unchanged. A next atlas manifest is authorized only if the frozen intrinsic metric gates pass.",
            "",
            f"Authorized next artifact: `{metrics['authorized_next']}`.",
            "",
        )),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    arguments = parser.parse_args()
    if not arguments.run:
        parser.error("--run is required")
    lock = _validate_manifest(require_clean=True)
    metrics, arrays = _diagnose(lock)
    summary = _canonicalize(metrics, arrays, lock)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
