#!/usr/bin/env python3
"""Audit the exact coordinate map and fail-fast direct branch predictor."""

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
from scipy.linalg import null_space


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_first_conditional_branch_seed_manifest_wp10c9d6c7c3b5c4f25ap as manifest  # noqa: E402
import run_causal_inner_larger_coarse_pde_audit_wp10c9d6c7c3b5c4f25k as r32  # noqa: E402
import run_causal_inner_nonlinear_bundle_screen_wp10c9d6c7c3b5c4f25ak as nonlinear  # noqa: E402
from imri_qpe.constants import C  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_monolithic_dae import (  # noqa: E402
    _integrated_mapped_storage,
    _spatial_nodes,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_monolithic_tangent import (  # noqa: E402
    _descriptor_matrices,
    _node_reconstruction_weights,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25aq"
MANIFEST_COMMIT = "8792be677ea28b4855c936457eae4ef9b995101e"
MANIFEST_PARENT = "dd593f6c79a579e0de0faaa0d511faa10765bf43"
MANIFEST_TREE = "c36c17820e93dfff5f4d1a0dedd5d1efe63aa0a4"

DIRECT_SAFE_CLASSIFICATION = (
    "exact_integrable_coordinate_preflight_passed_"
    "direct_branch_predictor_safe_single_root_manifest_authorized"
)
HOMOTOPY_CLASSIFICATION = (
    "exact_integrable_coordinate_preflight_passed_"
    "direct_branch_predictor_rejected_bordered_hidden_residual_homotopy_"
    "manifest_authorized"
)
COORDINATE_FAIL_CLASSIFICATION = (
    "exact_integrable_coordinate_preflight_failed_"
    "conditional_branch_architecture_blocked"
)

ARTIFACT = (
    "causal_inner_first_conditional_branch_seed_preflight_"
    "wp10c9d6c7c3b5c4f25aq"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_first_conditional_branch_seed_preflight_"
    "wp10c9d6c7c3b5c4f25aq.py"
)
THIS_TEST = (
    "tests/test_causal_inner_first_conditional_branch_seed_preflight_"
    "wp10c9d6c7c3b5c4f25aq.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_FIRST_CONDITIONAL_BRANCH_SEED_"
    "PREFLIGHT_WP10C9D6C7C3B5C4F25AQ_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
GEOMETRY_PATH = manifest.parent.GEOMETRY_DIRECTORY / "intrinsic_geometry.npz"

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
        raise RuntimeError("branch-seed manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("branch-seed manifest lineage changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("branch-seed manifest tree changed")
    hashes = _checksums(manifest.ARTIFACT_DIRECTORY)
    summary = _read(manifest.ARTIFACT_DIRECTORY / "summary.json")
    contract = _read(manifest.ARTIFACT_DIRECTORY / "contract.json")
    provenance = _read(manifest.ARTIFACT_DIRECTORY / "provenance.json")
    if (
        not summary["passed"]
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["physical_branch_root_attempted"]
        or contract["direct_predictor"]["direct_root_may_run_in_this_work_package"]
    ):
        raise RuntimeError("branch-seed preflight authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"manifest source changed: {relative}")
    for name, path in (
        ("primary_generator", manifest.GENERATOR_PATH),
        ("R32_projection", manifest.R32_PATH),
    ):
        if _sha(path) != contract["decisive_input_hashes"][name]:
            raise RuntimeError(f"decisive input changed: {path}")
    _checksums(manifest.parent.CANONICAL_DIRECTORY)
    if require_clean and not _tracked_tree_clean():
        raise RuntimeError("branch-seed preflight requires a clean tracked tree")
    for name, expected in THREAD_ENVIRONMENT.items():
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


def _coordinate_components() -> dict:
    data = nonlinear._anchor_data("primary")
    state = np.asarray(data["state"], dtype=float)
    columns = np.asarray(data["columns"], dtype=float)
    rows = np.asarray(data["rows"], dtype=float)
    context = data["context"]
    (
        node_weights,
        node_cells,
        node_radii,
        node_measures,
        reconstruction_defect,
        partition_defect,
    ) = _node_reconstruction_weights(context, state)
    mapped, _height = _descriptor_matrices(
        context,
        state,
        columns,
        rows,
        node_weights,
        node_cells,
        node_radii,
        node_measures,
    )
    physical_mapped = C * rows.ravel()[:, None] * mapped
    unscaled = np.zeros((manifest.MAPPED_COORDINATES, manifest.FULL_DIMENSION))
    groups = r32._R32_groups(state.shape[0])
    for coarse_cell, (start, stop) in enumerate(groups):
        for field in range(manifest.FIELDS_PER_CELL):
            target = manifest.FIELDS_PER_CELL * coarse_cell + field
            source_rows = manifest.FIELDS_PER_CELL * np.arange(start, stop) + field
            unscaled[target] = np.sum(physical_mapped[source_rows], axis=0)
    row_scales = np.linalg.norm(unscaled, axis=1)
    if np.any(~np.isfinite(row_scales)) or np.any(row_scales <= 0.0):
        raise RuntimeError("mapped-only R32 coordinate map lost a row")
    with np.load(manifest.R32_PATH, allow_pickle=False) as source:
        stable_dual = np.asarray(source["resolved_restriction"][-2:], dtype=float)
    jacobian = np.vstack((unscaled / row_scales[:, None], stable_dual))
    return {
        "data": data,
        "state": state,
        "columns": columns,
        "rows": rows,
        "context": context,
        "groups": groups,
        "mapped_row_scales": row_scales,
        "stable_dual": stable_dual,
        "jacobian": jacobian,
        "mapped_reconstruction_relative_defect": float(reconstruction_defect),
        "reconstruction_partition_defect": float(partition_defect),
    }


def _coordinate_value(state: np.ndarray, components: dict) -> np.ndarray:
    integrated, factors, _node_values = _integrated_mapped_storage(
        components["context"], state, _spatial_nodes(components["context"])
    )
    mapped = np.zeros(manifest.MAPPED_COORDINATES, dtype=float)
    for coarse_cell, (start, stop) in enumerate(components["groups"]):
        block = np.sum(integrated[start:stop], axis=0)
        mapped[
            manifest.FIELDS_PER_CELL * coarse_cell :
            manifest.FIELDS_PER_CELL * (coarse_cell + 1)
        ] = block
    mapped /= components["mapped_row_scales"]
    scaled_delta = (
        (np.asarray(state) - components["state"]) / components["columns"]
    ).ravel()
    stable = components["stable_dual"] @ scaled_delta
    if float(np.min(factors)) < 1.0 - 1.0e-12:
        raise RuntimeError("coordinate evaluation activated reconstruction limiting")
    return np.concatenate((mapped, stable))


def _coordinate_audit(components: dict) -> tuple[dict, dict[str, np.ndarray]]:
    jacobian = components["jacobian"]
    singular = np.linalg.svd(jacobian, compute_uv=False)
    rng = np.random.default_rng(20260819)
    direction = rng.normal(size=manifest.FULL_DIMENSION)
    direction /= np.linalg.norm(direction)
    epsilon = 1.0e-5
    plus = components["state"] + (
        epsilon * components["columns"].ravel() * direction
    ).reshape(components["state"].shape)
    minus = components["state"] - (
        epsilon * components["columns"].ravel() * direction
    ).reshape(components["state"].shape)
    finite_difference = (
        _coordinate_value(plus, components) - _coordinate_value(minus, components)
    ) / (2.0 * epsilon)
    analytic_action = jacobian @ direction
    with np.load(GEOMETRY_PATH, allow_pickle=False) as source:
        q3_rows = np.asarray(source["primary_constraint_rows"], dtype=float)
    rowspace_projection = q3_rows @ np.linalg.pinv(jacobian) @ jacobian
    q3_defect = _relative(rowspace_projection, q3_rows)
    metrics = {
        "coordinate_rank": int(np.linalg.matrix_rank(jacobian)),
        "coordinate_condition_number": float(singular[0] / singular[-1]),
        "coordinate_largest_singular_value": float(singular[0]),
        "coordinate_smallest_singular_value": float(singular[-1]),
        "mapped_reconstruction_relative_defect": components[
            "mapped_reconstruction_relative_defect"
        ],
        "reconstruction_partition_defect": components[
            "reconstruction_partition_defect"
        ],
        "directional_derivative_relative_defect": _relative(
            finite_difference, analytic_action
        ),
        "directional_derivative_step": epsilon,
        "Q3_rowspace_relative_defect": q3_defect,
        "responsive_height_one_form_used": False,
    }
    arrays = {
        "coordinate_jacobian": jacobian,
        "coordinate_singular_values": singular,
        "mapped_coordinate_row_scales": components["mapped_row_scales"],
        "stable_coordinate_duals": components["stable_dual"],
        "directional_derivative_direction": direction,
        "directional_derivative_finite_difference": finite_difference,
        "directional_derivative_analytic_action": analytic_action,
        "Q3_constraint_rows": q3_rows,
        "Q3_rowspace_projection": rowspace_projection,
        "anchor_coordinate_value": _coordinate_value(components["state"], components),
    }
    return metrics, arrays


def _direct_predictor_audit(
    components: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    with np.load(manifest.GENERATOR_PATH, allow_pickle=False) as source:
        generator = np.asarray(source["complete_fixed_Q_generator"], dtype=float)
        rate = np.asarray(source["fixed_Q_rate"], dtype=float)
    jacobian = components["jacobian"]
    hidden = null_space(jacobian)
    if hidden.shape != (manifest.FULL_DIMENSION, manifest.HIDDEN_DIMENSION):
        raise RuntimeError("hidden coordinate dimension changed")
    branch_matrix = np.vstack((jacobian, hidden.T @ generator))
    hidden_rate = hidden.T @ rate
    right_hand_side = np.concatenate(
        (np.zeros(manifest.RESOLVED_DIMENSION), -hidden_rate)
    )
    predictor = np.linalg.solve(branch_matrix, right_hand_side)
    residual = branch_matrix @ predictor - right_hand_side
    relative_residual = float(
        np.linalg.norm(residual)
        / max(float(np.linalg.norm(right_hand_side)), np.finfo(float).tiny)
    )
    metrics = {
        "full_rate_norm_per_second": float(np.linalg.norm(rate)),
        "hidden_rate_norm_per_second": float(np.linalg.norm(hidden_rate)),
        "hidden_basis_shape": list(hidden.shape),
        "hidden_orthogonality_defect": float(
            np.max(np.abs(hidden.T @ hidden - np.eye(hidden.shape[1])))
        ),
        "hidden_coordinate_annihilation_defect": float(
            np.max(np.abs(jacobian @ hidden))
        ),
        "branch_linear_condition_number": float(np.linalg.cond(branch_matrix)),
        "predictor_maximum_scaled_component": float(np.max(np.abs(predictor))),
        "predictor_scaled_norm": float(np.linalg.norm(predictor)),
        "predictor_relative_linear_residual": relative_residual,
        "nonbase_physical_truth_calls": 0,
        "direct_root_attempted": False,
    }
    return metrics, {
        "complete_fixed_Q_generator": generator,
        "fixed_Q_rate": rate,
        "hidden_basis": hidden,
        "hidden_rate": hidden_rate,
        "branch_linear_matrix": branch_matrix,
        "branch_linear_right_hand_side": right_hand_side,
        "direct_branch_predictor": predictor,
        "direct_branch_predictor_linear_residual": residual,
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
        raise RuntimeError("branch-seed preflight is already canonicalized")
    components = _coordinate_components()
    coordinate, coordinate_arrays = _coordinate_audit(components)
    predictor, predictor_arrays = _direct_predictor_audit(components)
    gates = frozen["contract"]["binding_gates"]
    coordinate_passed = bool(
        coordinate["mapped_reconstruction_relative_defect"]
        <= gates["mapped_reconstruction_relative_defect_max"]
        and coordinate["reconstruction_partition_defect"]
        <= gates["reconstruction_partition_defect_max"]
        and coordinate["coordinate_rank"] == gates["coordinate_rank_equal"]
        and coordinate["coordinate_condition_number"]
        <= gates["coordinate_condition_number_max"]
        and coordinate["directional_derivative_relative_defect"]
        <= gates["coordinate_directional_derivative_relative_defect_max"]
        and coordinate["Q3_rowspace_relative_defect"]
        <= gates["Q3_rowspace_relative_defect_max"]
    )
    direct_safe = bool(
        coordinate_passed
        and predictor["branch_linear_condition_number"]
        <= gates["direct_branch_linear_condition_number_max"]
        and predictor["predictor_maximum_scaled_component"]
        <= gates["direct_predictor_maximum_scaled_component_max"]
        and predictor["predictor_relative_linear_residual"]
        <= gates["direct_predictor_relative_linear_residual_max"]
    )
    if not coordinate_passed:
        classification = COORDINATE_FAIL_CLASSIFICATION
        authorized_next = None
    elif direct_safe:
        classification = DIRECT_SAFE_CLASSIFICATION
        authorized_next = "definitions_only_single_direct_branch_root_manifest"
    else:
        classification = HOMOTOPY_CLASSIFICATION
        authorized_next = (
            "definitions_only_bordered_hidden_residual_homotopy_manifest"
        )
    passed = coordinate_passed
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    metrics = {
        "coordinate_map": coordinate,
        "direct_predictor": predictor,
        "coordinate_structure_passed": coordinate_passed,
        "direct_predictor_safe": direct_safe,
        "homotopy_required": bool(coordinate_passed and not direct_safe),
    }
    _write_json(CANONICAL_DIRECTORY / "metrics.json", metrics)
    np.savez_compressed(
        CANONICAL_DIRECTORY / "preflight_diagnostics.npz",
        primitive_anchor=components["state"],
        primitive_column_scales=components["columns"],
        conservation_row_scales=components["rows"],
        **coordinate_arrays,
        **predictor_arrays,
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "coordinate_structure_passed": coordinate_passed,
        "coordinate_rank": coordinate["coordinate_rank"],
        "coordinate_condition_number": coordinate["coordinate_condition_number"],
        "branch_linear_condition_number": predictor[
            "branch_linear_condition_number"
        ],
        "direct_predictor_maximum_scaled_component": predictor[
            "predictor_maximum_scaled_component"
        ],
        "direct_predictor_safe": direct_safe,
        "direct_root_attempted": False,
        "homotopy_required": bool(coordinate_passed and not direct_safe),
        "physical_branch_found": False,
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
            "architecture_package_hashes": _checksums(
                manifest.parent.CANONICAL_DIRECTORY
            ),
            "decisive_input_hashes": frozen["contract"][
                "decisive_input_hashes"
            ],
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
        "".join(f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# First conditional branch seed preflight WP10c9d6c7c3b5c4f25aq",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                "## Result",
                "",
                f"The exact mapped-only 162-coordinate map has rank `{coordinate['coordinate_rank']}` and condition number `{coordinate['coordinate_condition_number']:.6e}`. Its directional derivative defect is `{coordinate['directional_derivative_relative_defect']:.6e}` and the Q3 rowspace defect is `{coordinate['Q3_rowspace_relative_defect']:.6e}`.",
                "",
                f"The direct 560-state branch linearization has condition number `{predictor['branch_linear_condition_number']:.6e}`. Its exact linear solve predicts a maximum scaled component `{predictor['predictor_maximum_scaled_component']:.6e}` and norm `{predictor['predictor_scaled_norm']:.6e}`.",
                "",
                "The coordinate structure passes, but the direct root predictor is rejected prospectively. No physical root was attempted and no candidate was propagated.",
                "",
                "## Next gate",
                "",
                f"Authorized next artifact: `{authorized_next}`.",
                "",
                "The successor must start from the exact anchor through a bordered hidden-residual homotopy. This rejection is not evidence that a conditional branch is absent, and it does not authorize relaxed physical or algebraic tolerances.",
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
