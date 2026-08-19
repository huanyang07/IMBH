#!/usr/bin/env python3
"""Recover and audit the coordinate-Hessian term in the branch KKT tangent."""

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

import run_causal_inner_coordinate_hessian_diagnosis_manifest_wp10c9d6c7c3b5c4f25at as manifest  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25au"
MANIFEST_COMMIT = "b8f99886f382d79e3327a6fae8a5406138d0cb33"
MANIFEST_PARENT = "cbcf05bdf187dc9357ab8da8a14d44641c4405fc"
MANIFEST_TREE = "8564a4608b4b14d1a31e87c06cb8ae7925c9bc6e"

PASS_CLASSIFICATION = (
    "coordinate_hessian_complete_KKT_tangent_certified_"
    "corrected_homotopy_launch_manifest_authorized"
)
SMALLER_TAU_CLASSIFICATION = (
    "coordinate_hessian_certified_complete_KKT_tangent_outside_trust_"
    "smaller_tau_manifest_authorized"
)
FAIL_CLASSIFICATION = (
    "coordinate_hessian_recovery_failed_"
    "branch_solver_architecture_requires_revision"
)

ARTIFACT = (
    "causal_inner_coordinate_hessian_diagnosis_"
    "wp10c9d6c7c3b5c4f25au"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_coordinate_hessian_diagnosis_"
    "wp10c9d6c7c3b5c4f25au.py"
)
THIS_TEST = (
    "tests/test_causal_inner_coordinate_hessian_diagnosis_"
    "wp10c9d6c7c3b5c4f25au.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_COORDINATE_HESSIAN_DIAGNOSIS_"
    "WP10C9D6C7C3B5C4F25AU_2026-08-19.md"
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


def _validate_manifest(*, require_clean: bool) -> dict:
    if _git("rev-parse", MANIFEST_COMMIT) != MANIFEST_COMMIT:
        raise RuntimeError("coordinate-Hessian manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("coordinate-Hessian manifest lineage changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("coordinate-Hessian manifest tree changed")
    hashes = _checksums(manifest.ARTIFACT_DIRECTORY)
    summary = _read(manifest.ARTIFACT_DIRECTORY / "summary.json")
    contract = _read(manifest.ARTIFACT_DIRECTORY / "contract.json")
    provenance = _read(manifest.ARTIFACT_DIRECTORY / "provenance.json")
    if (
        not summary["passed"]
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["new_fixed_Q_rate_evaluations"] != 0
        or contract["claim_boundary"]["failed_homotopy_candidate_reclassified"]
    ):
        raise RuntimeError("coordinate-Hessian authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"manifest source changed: {relative}")
    decisive = contract["decisive_input_hashes"]
    for name, path in (
        (
            "failed_launch_checkpoint",
            manifest.parent.CANONICAL_DIRECTORY / "homotopy_tau_1_over_64.npz",
        ),
        (
            "failed_launch_metrics",
            manifest.parent.CANONICAL_DIRECTORY / "metrics.json",
        ),
        ("preflight_diagnostics", manifest.parent.PREFLIGHT_ARRAYS),
    ):
        if _sha(path) != decisive[name]:
            raise RuntimeError(f"decisive input changed: {path}")
    _checksums(manifest.parent.CANONICAL_DIRECTORY)
    if require_clean and not _tracked_tree_clean():
        raise RuntimeError("coordinate-Hessian diagnosis requires a clean tracked tree")
    for name, expected in manifest.parent.preflight.THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    return {"summary": summary, "contract": contract, "hashes": hashes}


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(left) - np.asarray(right))
        / max(
            float(np.linalg.norm(left)),
            float(np.linalg.norm(right)),
            np.finfo(float).tiny,
        )
    )


def _columns_for_color(
    cells: int, field: int, color: int
) -> np.ndarray:
    selected_cells = np.arange(color, cells, manifest.CELL_COLOR_COUNT)
    return manifest.FIELD_COUNT * selected_cells + int(field)


def _row_indices_for_cell(cell: int, cells: int) -> np.ndarray:
    start = max(0, int(cell) - manifest.CELL_HALF_BANDWIDTH)
    stop = min(int(cells), int(cell) + manifest.CELL_HALF_BANDWIDTH + 1)
    return np.arange(
        manifest.FIELD_COUNT * start,
        manifest.FIELD_COUNT * stop,
        dtype=int,
    )


def _recover_from_colored_responses(
    responses: dict[tuple[int, int], np.ndarray], cells: int
) -> tuple[np.ndarray, float]:
    dimension = manifest.FIELD_COUNT * int(cells)
    recovered = np.zeros((dimension, dimension), dtype=float)
    leakage_squared = 0.0
    response_squared = 0.0
    for field in range(manifest.FIELD_COUNT):
        for color in range(manifest.CELL_COLOR_COUNT):
            response = np.asarray(responses[(field, color)], dtype=float)
            if response.shape != (dimension,):
                raise ValueError("colored Hessian response shape changed")
            assigned = np.zeros(dimension, dtype=bool)
            for column in _columns_for_color(cells, field, color):
                cell = int(column // manifest.FIELD_COUNT)
                rows = _row_indices_for_cell(cell, cells)
                if np.any(assigned[rows]):
                    raise RuntimeError("Hessian coloring has overlapping row support")
                recovered[rows, column] = response[rows]
                assigned[rows] = True
            leakage_squared += float(response[~assigned] @ response[~assigned])
            response_squared += float(response @ response)
    leakage = math.sqrt(leakage_squared) / max(
        math.sqrt(response_squared), np.finfo(float).tiny
    )
    return recovered, float(leakage)


def _colored_hessian(system: dict):
    state = system["components"]["state"]
    columns = system["components"]["columns"]
    multiplier = system["multiplier_anchor"]
    cells = state.shape[0]
    epsilon = manifest.CENTRAL_DIFFERENCE_STEP
    responses = {}
    evaluations = 0
    for field in range(manifest.FIELD_COUNT):
        for color in range(manifest.CELL_COLOR_COUNT):
            direction = np.zeros(state.size, dtype=float)
            selected = _columns_for_color(cells, field, color)
            direction[selected] = 1.0
            plus = state + (
                epsilon * columns.ravel() * direction
            ).reshape(state.shape)
            minus = state - (
                epsilon * columns.ravel() * direction
            ).reshape(state.shape)
            plus_jacobian, _ = manifest.parent._coordinate_jacobian(plus, system)
            minus_jacobian, _ = manifest.parent._coordinate_jacobian(minus, system)
            responses[(field, color)] = (
                plus_jacobian.T @ multiplier - minus_jacobian.T @ multiplier
            ) / (2.0 * epsilon)
            evaluations += 2
            print(
                f"f25au: recovered color field={field} color={color} "
                f"evaluations={evaluations}",
                flush=True,
            )
    recovered, leakage = _recover_from_colored_responses(responses, cells)
    return recovered, leakage, evaluations


def _random_direction_audit(system: dict, hessian: np.ndarray):
    rng = np.random.default_rng(20260819)
    direction = rng.normal(size=560)
    direction /= np.linalg.norm(direction)
    epsilon = manifest.CENTRAL_DIFFERENCE_STEP
    state = system["components"]["state"]
    columns = system["components"]["columns"]
    plus = state + (epsilon * columns.ravel() * direction).reshape(state.shape)
    minus = state - (epsilon * columns.ravel() * direction).reshape(state.shape)
    plus_jacobian, _ = manifest.parent._coordinate_jacobian(plus, system)
    minus_jacobian, _ = manifest.parent._coordinate_jacobian(minus, system)
    finite_difference = (
        plus_jacobian.T @ system["multiplier_anchor"]
        - minus_jacobian.T @ system["multiplier_anchor"]
    ) / (2.0 * epsilon)
    recovered_action = hessian @ direction
    return {
        "relative_defect": _relative(finite_difference, recovered_action),
        "direction": direction,
        "finite_difference": finite_difference,
        "recovered_action": recovered_action,
    }


def _complete_tangent(system: dict, hessian: np.ndarray):
    jacobian = system["jacobian_anchor"]
    generator = system["saved"]["complete_fixed_Q_generator"]
    complete_state_block = generator / system["rate_scale"] - hessian
    matrix = np.block(
        [
            [jacobian, np.zeros((162, 162))],
            [complete_state_block, -jacobian.T],
        ]
    )
    right = -system["base_target_residual"]
    predictor, linear = manifest.parent._equilibrated_solve(matrix, right)
    return matrix, complete_state_block, predictor, linear


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
        raise RuntimeError("coordinate-Hessian diagnosis is already canonicalized")
    began = time.perf_counter()
    system = manifest.parent._anchor_system()
    raw_hessian, leakage, colored_evaluations = _colored_hessian(system)
    symmetry = _relative(raw_hessian, raw_hessian.T)
    hessian = 0.5 * (raw_hessian + raw_hessian.T)
    random = _random_direction_audit(system, hessian)
    matrix, state_block, predictor, linear = _complete_tangent(system, hessian)
    gates = frozen["contract"]["binding_gates"]
    recovery_checks = {
        "symmetry": symmetry
        <= gates["recovered_Hessian_relative_symmetry_defect_max"],
        "random_direction": random["relative_defect"]
        <= gates["random_direction_action_relative_defect_max"],
        "band_leakage": leakage
        <= gates["outside_declared_band_relative_leakage_max"],
        "complete_condition": linear["equilibrated_condition_number"]
        <= gates["complete_equilibrated_KKT_condition_number_max"],
        "linear_solve": linear["relative_linear_residual"]
        <= gates["complete_linear_solve_relative_residual_max"],
    }
    recovery_passed = all(recovery_checks.values())
    maximum_component = float(np.max(np.abs(predictor[:560])))
    tangent_safe = bool(
        recovery_passed
        and maximum_component
        <= gates["complete_tau_1_over_64_tangent_maximum_scaled_component_max"]
    )
    if tangent_safe:
        classification = PASS_CLASSIFICATION
        authorized_next = "definitions_only_corrected_homotopy_launch_manifest"
    elif recovery_passed:
        classification = SMALLER_TAU_CLASSIFICATION
        authorized_next = "definitions_only_smaller_tau_homotopy_launch_manifest"
    else:
        classification = FAIL_CLASSIFICATION
        authorized_next = None
    passed = recovery_passed
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    metrics = {
        "recovery": {
            "checks": recovery_checks,
            "cell_half_bandwidth": manifest.CELL_HALF_BANDWIDTH,
            "cell_color_count": manifest.CELL_COLOR_COUNT,
            "colored_direction_count": manifest.COLORED_DIRECTION_COUNT,
            "coordinate_jacobian_evaluations": colored_evaluations + 2,
            "new_fixed_Q_rate_evaluations": 0,
            "relative_symmetry_defect": symmetry,
            "outside_band_relative_leakage": leakage,
            "random_direction_action_relative_defect": random["relative_defect"],
            "raw_Hessian_norm": float(np.linalg.norm(raw_hessian)),
            "symmetrized_Hessian_norm": float(np.linalg.norm(hessian)),
        },
        "complete_tangent": {
            **linear,
            "tau": manifest.TAU_DIAGNOSTIC,
            "predictor_maximum_scaled_component": maximum_component,
            "predictor_scaled_norm": float(np.linalg.norm(predictor[:560])),
            "Gauss_Newton_predictor_maximum_scaled_component": float(
                np.max(np.abs(system["predictor_correction"][:560]))
            ),
            "tangent_safe": tangent_safe,
        },
        "total_wall_seconds": time.perf_counter() - began,
    }
    _write_json(CANONICAL_DIRECTORY / "metrics.json", metrics)
    np.savez_compressed(
        CANONICAL_DIRECTORY / "coordinate_hessian_and_complete_KKT.npz",
        raw_coordinate_hessian=raw_hessian,
        coordinate_hessian=hessian,
        complete_state_block=state_block,
        complete_KKT_matrix=matrix,
        complete_tau_1_over_64_predictor=predictor,
        random_direction=random["direction"],
        random_finite_difference_action=random["finite_difference"],
        random_recovered_action=random["recovered_action"],
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "coordinate_hessian_recovery_passed": recovery_passed,
        "complete_tau_1_over_64_tangent_safe": tangent_safe,
        "coordinate_jacobian_evaluations": colored_evaluations + 2,
        "new_fixed_Q_rate_evaluations": 0,
        "relative_symmetry_defect": symmetry,
        "random_direction_action_relative_defect": random["relative_defect"],
        "complete_equilibrated_KKT_condition_number": linear[
            "equilibrated_condition_number"
        ],
        "complete_predictor_maximum_scaled_component": maximum_component,
        "physical_conditional_branch_found": False,
        "normal_hyperbolicity_certified": False,
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
            "failed_launch_package_hashes": _checksums(
                manifest.parent.CANONICAL_DIRECTORY
            ),
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
            "thread_environment": manifest.parent.preflight.THREAD_ENVIRONMENT,
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
                "# Coordinate-Hessian diagnosis WP10c9d6c7c3b5c4f25au",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                "## Result",
                "",
                f"The 25-color recovery used `{colored_evaluations + 2}` coordinate-Jacobian evaluations and zero new fixed-Q rate evaluations. The raw Hessian symmetry defect is `{symmetry:.6e}`, the independent random-action defect is `{random['relative_defect']:.6e}`, and the outside-band leakage is `{leakage:.6e}`.",
                "",
                f"After adding the missing coordinate curvature, the equilibrated complete KKT condition number is `{linear['equilibrated_condition_number']:.6e}` and the tau=1/64 tangent has maximum scaled state component `{maximum_component:.6e}`.",
                "",
                f"Authorized next artifact: `{authorized_next}`.",
                "",
                "No new physical rate was evaluated, the failed launch remains rejected, and no branch or reduced slow evolution is claimed.",
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
