#!/usr/bin/env python3
"""Recertify the saved fixed-Q endpoint with exact increment storage."""

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

import run_causal_inner_face36_fixed_q_endpoint_linearization_audit_wp10c9d6c7c3b5c4f24e3 as e3  # noqa: E402
import run_causal_inner_face36_fixed_q_residual_resolution_audit_wp10c9d6c7c3b5c4f24e4 as e4  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_fixed_q import (  # noqa: E402
    causal_five_field_fixed_q_augmented_step_matrix,
    evaluate_causal_five_field_fixed_q_bdf,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_monolithic_tangent import (  # noqa: E402
    causal_five_field_monolithic_discrete_step_matrix,
)


WORK_PACKAGE = "WP10c9d6c7c3b5c4f24e5"
ARTIFACT = (
    "causal_inner_face36_fixed_q_exact_increment_storage_recertification_"
    "wp10c9d6c7c3b5c4f24e5"
)
RESULT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CHECKPOINT_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
MANIFEST_DIRECTORY = (
    ROOT
    / "results/canonical"
    / "causal_inner_face36_fixed_q_exact_increment_storage_repair_manifest_"
    "wp10c9d6c7c3b5c4f24e5"
)
PARENT_DIRECTORY = e4.RESULT_DIRECTORY
E3_DIRECTORY = e3.RESULT_DIRECTORY
CATALOG_CSV = ROOT / "results/manifests/canonical_artifacts.csv"
CATALOG_JSON = ROOT / "results/manifests/canonical_summary.json"
THIS_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_exact_increment_storage_"
    "recertification_wp10c9d6c7c3b5c4f24e5.py"
)
THIS_TEST = (
    "tests/test_causal_inner_face36_fixed_q_exact_increment_storage_"
    "recertification_wp10c9d6c7c3b5c4f24e5.py"
)
ALPHAS = tuple(2.0 ** (-index) for index in range(8))


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tracked_tree_is_clean() -> bool:
    return bool(
        subprocess.run(("git", "diff", "--quiet"), cwd=ROOT).returncode == 0
        and subprocess.run(
            ("git", "diff", "--cached", "--quiet"), cwd=ROOT
        ).returncode
        == 0
    )


def _relative(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    scale = max(
        float(np.linalg.norm(left)),
        float(np.linalg.norm(right)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(left - right) / scale)


def _order(coarse: float, fine: float) -> float:
    if coarse <= 0.0 or fine <= 0.0:
        return float("nan")
    return float(math.log(coarse / fine) / math.log(2.0))


def _catalog(summary: dict, status: str) -> None:
    with CATALOG_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row["case"] != ARTIFACT]
    for path in sorted(RESULT_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha(path),
                    "scientific_status": status,
                }
            )
    with CATALOG_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=tuple(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read(CATALOG_JSON)
    catalog["artifacts"][ARTIFACT] = {
        "path": str(RESULT_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": summary["passed"],
    }
    catalog.update(
        {
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write(CATALOG_JSON, catalog)


def _contract() -> dict:
    manifest = _read(MANIFEST_DIRECTORY / "summary.json")
    contract = _read(MANIFEST_DIRECTORY / "execution_manifest.json")
    parent = _read(PARENT_DIRECTORY / "summary.json")
    if (
        not manifest["saved_endpoint_recertification_authorized"]
        or parent["selected_next"] != "cancellation_prone_block_repair"
    ):
        raise RuntimeError("exact-increment recertification is not authorized")
    return contract


def _evaluate_unknown(
    unknown: np.ndarray,
    data: dict,
    *,
    direct_rate: bool = False,
):
    dimensions = int(data["state"].size)
    scaled_increment = np.asarray(unknown[:dimensions], dtype=float)
    candidate = data["state"] + data["columns"] * scaled_increment.reshape(
        data["state"].shape
    )
    return evaluate_causal_five_field_fixed_q_bdf(
        data["state"],
        candidate,
        unknown[dimensions:],
        data["reaction"].q3_value,
        1.0e-7,
        data["context"],
        order=1,
        primitive_column_scales=data["columns"],
        conservation_row_scales=data["rows"],
        parent_cell_indices=data["layout"].parent_cell_indices,
        refinement_ratio=data["layout"].refinement_ratio,
        constraint_row_scales=data["reaction"].q3_derivative_norms,
        reaction_channel_basis="frozen_normalized",
        reaction_channel_transform=data["reaction"].raw_schur_inverse,
        scaled_primitive_increment=scaled_increment,
        scaled_rate_per_s=(
            scaled_increment / 1.0e-7 if direct_rate else None
        ),
        maximum_schur_condition_number=1.0e8,
    )


def _checkpoint(alpha: float) -> Path:
    token = f"{alpha:.12e}".replace("+", "p").replace("-", "m").replace(".", "d")
    return CHECKPOINT_DIRECTORY / f"alpha_{token}.npz"


def _evaluate_alpha(
    alpha: float,
    unknown: np.ndarray,
    correction: np.ndarray,
    data: dict,
) -> dict[str, np.ndarray]:
    path = _checkpoint(alpha)
    if path.exists():
        with np.load(path, allow_pickle=False) as source:
            return {name: np.asarray(source[name]) for name in source.files}
    print(f"f24e5: evaluate alpha={alpha:.8f}", flush=True)
    evaluation = _evaluate_unknown(unknown + alpha * correction, data)
    blocks = e4._blocks(evaluation, data["rows"])
    _write_npz(path, **blocks)
    return blocks


def _execute() -> dict:
    contract = _contract()
    if not _tracked_tree_is_clean():
        raise RuntimeError("recertification requires a clean tracked tree")
    identity = {
        "execution_commit": _git("rev-parse", "HEAD"),
        "execution_tree": _git("rev-parse", "HEAD^{tree}"),
        "runner_sha256": _sha(ROOT / THIS_RUNNER),
        "manifest_summary_sha256": _sha(MANIFEST_DIRECTORY / "summary.json"),
        "parent_arrays_sha256": _sha(E3_DIRECTORY / "decisive_arrays.npz"),
        "source_hashes": {
            name: _sha(ROOT / name)
            for name in (
                "src/imri_qpe/layer3_minidisk_1d/causal_inner_fixed_q.py",
                "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_bdf.py",
                "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_dae.py",
            )
        },
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OMP_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
    }
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    identity_path = CHECKPOINT_DIRECTORY / "execution_identity.json"
    if identity_path.exists() and _read(identity_path) != _plain(identity):
        raise RuntimeError("recertification execution identity changed")
    _write(identity_path, identity)
    began = time.perf_counter()
    data = e4._data()
    dimensions = int(data["state"].size)
    with np.load(E3_DIRECTORY / "decisive_arrays.npz", allow_pickle=False) as source:
        unknown = np.asarray(source["unknown"], dtype=float)
    base_evaluations = [_evaluate_unknown(unknown, data) for _index in range(3)]
    base_blocks = [e4._blocks(item, data["rows"]) for item in base_evaluations]
    repeat_bitwise = all(
        all(
            np.array_equal(base_blocks[0][name], current[name])
            for name in base_blocks[0]
        )
        for current in base_blocks[1:]
    )
    direct = e4._blocks(
        _evaluate_unknown(unknown, data, direct_rate=True),
        data["rows"],
    )
    storage_defects = {
        name: _relative(base_blocks[0][name], direct[name])
        for name in ("mapped_storage", "height_storage")
    }
    endpoint = data["state"] + data["columns"] * unknown[:dimensions].reshape(
        data["state"].shape
    )
    print("f24e5: assemble exact repaired endpoint matrix", flush=True)
    matrix = causal_five_field_fixed_q_augmented_step_matrix(
        data["context"],
        data["state"],
        endpoint,
        unknown[dimensions:],
        1.0e-7,
        None,
        order=1,
        primitive_column_scales=data["columns"],
        conservation_row_scales=data["rows"],
        parent_cell_indices=data["layout"].parent_cell_indices,
        refinement_ratio=data["layout"].refinement_ratio,
        constraint_row_scales=data["reaction"].q3_derivative_norms,
        reaction_channel_basis="frozen_normalized",
        reaction_channel_transform=data["reaction"].raw_schur_inverse,
        reaction=base_evaluations[0].reaction,
    )
    correction, linear_defect = e3._equilibrated_dense_solve(
        matrix.scaled_matrix,
        -base_blocks[0]["augmented"],
    )
    correction_norm = float(np.linalg.norm(correction))
    actual = [
        _evaluate_alpha(alpha, unknown, correction, data) for alpha in ALPHAS
    ]
    base = base_blocks[0]
    base_norm = max(
        float(np.linalg.norm(base["augmented"])),
        np.finfo(float).tiny,
    )
    action = matrix.scaled_matrix @ correction
    model_errors = []
    alpha_metrics = []
    for alpha, blocks in zip(ALPHAS, actual, strict=True):
        predicted = base["augmented"] + alpha * action
        error = float(
            np.linalg.norm(blocks["augmented"] - predicted) / base_norm
        )
        model_errors.append(error)
        alpha_metrics.append(
            {
                "alpha": alpha,
                "augmented_L2_norm": float(
                    np.linalg.norm(blocks["augmented"])
                ),
                "augmented_maximum": float(
                    np.max(np.abs(blocks["augmented"]))
                ),
                "predicted_L2_norm": float(np.linalg.norm(predicted)),
                "model_error_to_base_residual": error,
            }
        )
    model_orders = [
        _order(model_errors[index], model_errors[index + 1])
        for index in range(len(model_errors) - 1)
    ]
    storage = base_evaluations[0].monolithic_evaluation.current_storage_increment
    mapped_closure = float(storage.maximum_mapped_path_closure_defect)
    repaired_root_maximum = alpha_metrics[0]["augmented_maximum"]
    gates = {
        "endpoint_repeatability": repeat_bitwise,
        "increment_direct_storage": max(storage_defects.values())
        <= contract["maximum_increment_direct_storage_relative_defect"],
        "mapped_endpoint_path_closure": mapped_closure
        <= contract["maximum_mapped_endpoint_path_closure_defect"],
        "linear_solve": linear_defect <= 1.0e-10,
        "full_step_model": model_errors[0]
        <= contract["maximum_full_step_model_error_to_base_residual"],
        "first_three_model_orders": min(model_orders[:3])
        >= contract["minimum_first_three_model_error_orders"],
        "saved_root": repaired_root_maximum
        <= contract["maximum_saved_root_scaled_residual"],
    }
    passed = all(gates.values())
    if passed:
        classification = (
            "fixed_Q_exact_increment_storage_repair_certified_"
            "authentic_history_ladder_retry_authorized"
        )
        selected = "retry_authentic_history_ladder_from_first_case"
    elif not gates["mapped_endpoint_path_closure"]:
        classification = "fixed_Q_exact_increment_endpoint_path_closure_failed"
        selected = "mapped_endpoint_audit_repair"
    elif not gates["full_step_model"] or not gates["first_three_model_orders"]:
        classification = "fixed_Q_exact_increment_residual_resolution_failed"
        selected = "exact_increment_path_derivative_repair"
    else:
        classification = "fixed_Q_exact_increment_saved_root_not_certified"
        selected = "saved_endpoint_solver_audit"
    metrics = {
        "endpoint_repeat_bitwise": repeat_bitwise,
        "storage_increment_direct_relative_defects": storage_defects,
        "mapped_endpoint_path_closure_defect": mapped_closure,
        "base_maximum_scaled_residual": float(
            np.max(np.abs(base["augmented"]))
        ),
        "base_residual_L2_norm": base_norm,
        "correction_L2_norm": correction_norm,
        "linear_relative_residual": linear_defect,
        "alpha_metrics": alpha_metrics,
        "model_error_orders": model_orders,
        "gates": gates,
        "selected_next": selected,
        "wall_seconds": time.perf_counter() - began,
    }
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "analysis_only": True,
        "trajectory_executed": False,
        "physical_failure_detected": False,
        "parent_rejections_preserved": True,
        "selected_next": selected,
        "authentic_history_ladder_retry_authorized": passed,
        "fixed_Q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    RESULT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write(RESULT_DIRECTORY / "summary.json", summary)
    _write(RESULT_DIRECTORY / "metrics.json", metrics)
    _write(RESULT_DIRECTORY / "contract.json", contract)
    _write_npz(
        RESULT_DIRECTORY / "decisive_arrays.npz",
        unknown=unknown,
        correction=correction,
        alphas=np.asarray(ALPHAS),
        model_errors_to_base_residual=np.asarray(model_errors),
        model_error_orders=np.asarray(model_orders),
        base_augmented_residual=base["augmented"],
        exact_matrix_action=action,
    )
    _write(
        RESULT_DIRECTORY / "provenance.json",
        {
            "schema_version": 1,
            **identity,
            "tracked_worktree_clean_at_start": True,
            "untracked_files_at_start": _git(
                "ls-files", "--others", "--exclude-standard"
            ).splitlines(),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    )
    names = (
        "contract.json",
        "decisive_arrays.npz",
        "metrics.json",
        "provenance.json",
        "summary.json",
    )
    (RESULT_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha(RESULT_DIRECTORY / name)}  {name}\n" for name in names
        ),
        encoding="utf-8",
    )
    _catalog(summary, "SUPPORTED" if passed else "REJECTED")
    return {"summary": summary, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    if not arguments.execute:
        raise SystemExit("use --execute")
    print(json.dumps(_plain(_execute()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
