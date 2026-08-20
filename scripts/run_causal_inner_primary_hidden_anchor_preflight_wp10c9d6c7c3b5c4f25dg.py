#!/usr/bin/env python3
"""Execute the fail-fast hidden-rate gate at the exact primary anchor."""

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

import run_causal_inner_face36_fixed_q_authentic_history_ladder_wp10c9d6c7c3b5c4f24e1 as rate_source  # noqa: E402
import run_causal_inner_primary_hidden_fast_root_manifest_wp10c9d6c7c3b5c4f25df as manifest  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25dg"
MANIFEST_COMMIT = "10c7bffdce6fa07c931a31793fba3f1b02908692"
MANIFEST_PARENT = "d831aa88d7b47556c8380f181f11719295a80b78"
MANIFEST_TREE = "b5d9014595a72580ffb6f069a49fd787905ffcd8"

IMPLEMENTATION_PARENT_COMMIT = "ce60c9268141e0bb62e6270c32e4ebdef26644a8"

PASS_CLASSIFICATION = (
    "primary_anchor_hidden_fraction_passed_complete_tangent_stage_authorized"
)
FAIL_CLASSIFICATION = (
    "primary_anchor_not_near_frozen_macro_critical_manifold_root_not_attempted"
)
PASS_AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25dh"

HIDDEN_FRACTION_GATE = 0.25
FIXED_Q_TANGENCY_GATE = 1.0e-12
REACTION_LEDGER_GATE = 1.0e-12
SCHUR_CONDITION_GATE = 1.0e8
RECONSTRUCTION_GATE = 1.0 - 1.0e-12
HEIGHT_RATIO_GATE = 0.5
OPTICAL_DEPTH_GATE = 1.0

ARTIFACT = (
    "causal_inner_primary_hidden_anchor_preflight_"
    "wp10c9d6c7c3b5c4f25dg"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_primary_hidden_anchor_preflight_"
    "wp10c9d6c7c3b5c4f25dg.py"
)
THIS_TEST = (
    "tests/test_causal_inner_primary_hidden_anchor_preflight_"
    "wp10c9d6c7c3b5c4f25dg.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_PRIMARY_HIDDEN_ANCHOR_"
    "PREFLIGHT_WP10C9D6C7C3B5C4F25DG_2026-08-20.md"
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
        raise RuntimeError("hidden-root manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("hidden-root manifest lineage changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("hidden-root manifest tree changed")
    hashes = _checksums(manifest.CANONICAL_DIRECTORY)
    summary = _read(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = _read(
        manifest.CANONICAL_DIRECTORY / "primary_hidden_root_contract.json"
    )
    provenance = _read(manifest.CANONICAL_DIRECTORY / "provenance.json")
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or summary["authorized_next"] != WORK_PACKAGE
        or not summary["dual_consistent_hidden_residual_frozen"]
        or summary["branch_root_in_this_package"]
        or summary["sealed_16ms_opened"]
        or contract["prospective_execution"]["work_package"] != WORK_PACKAGE
        or not contract["mathematical_architecture"][
            "naive_residual_Z_transpose_F_forbidden"
        ]
    ):
        raise RuntimeError("hidden-root manifest authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"hidden-root manifest source changed: {relative}")
    decisive = contract["decisive_input_hashes"]
    if _sha(manifest.CHART_DIRECTORY / "exact_chart_arrays.npz") != decisive[
        "exact_chart_arrays"
    ]:
        raise RuntimeError("exact chart input changed")
    if _sha(manifest.FIBER_DIRECTORY / "fiber_geometry.npz") != decisive[
        "fiber_geometry"
    ]:
        raise RuntimeError("fiber geometry input changed")
    for name, expected in provenance["thread_environment"].items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("primary hidden anchor preflight requires a clean tree")
    return {"manifest_hashes": hashes, "contract": contract}


def _validate_implementation_lock() -> dict:
    if IMPLEMENTATION_PARENT_COMMIT.startswith("__"):
        raise RuntimeError("implementation parent commit is not frozen")
    if _git("rev-parse", "HEAD^") != IMPLEMENTATION_PARENT_COMMIT:
        raise RuntimeError("execution is not based on the frozen implementation")
    return {
        "implementation_parent_commit": IMPLEMENTATION_PARENT_COMMIT,
        "execution_commit": _git("rev-parse", "HEAD"),
        "execution_tree": _git("rev-parse", "HEAD^{tree}"),
    }


def _geometry() -> dict[str, np.ndarray]:
    geometry = _load_npz(manifest.CANONICAL_DIRECTORY / "dual_hidden_geometry.npz")
    chart = _load_npz(manifest.CHART_DIRECTORY / "exact_chart_arrays.npz")
    return {
        "R": np.asarray(geometry["macro_restriction_R82"], dtype=float),
        "L": np.asarray(geometry["macro_lifting_L82"], dtype=float),
        "Z": np.asarray(geometry["hidden_basis_Z388"], dtype=float),
        "P": np.asarray(geometry["fiber_projection_P470"], dtype=float),
        "Q": np.asarray(geometry["hidden_dual_Q388"], dtype=float),
        "anchor_state": np.asarray(chart["anchor_primitive_state"], dtype=float),
        "anchor_coordinate": np.asarray(
            chart["anchor_coordinate_y470"], dtype=float
        ),
        "coordinate_jacobian": np.asarray(
            chart["anchor_coordinate_jacobian"], dtype=float
        ),
    }


def _decompose_rate(
    coordinate_rate: np.ndarray, geometry: dict[str, np.ndarray]
) -> dict[str, np.ndarray | float]:
    F = np.asarray(coordinate_rate, dtype=float)
    G = geometry["R"] @ F
    H = geometry["Q"] @ F
    hidden_action = geometry["Z"] @ H
    macro_action = geometry["L"] @ G
    denominator = max(float(np.linalg.norm(F)), np.finfo(float).tiny)
    return {
        "F": F,
        "G": G,
        "H": H,
        "hidden_action": hidden_action,
        "macro_action": macro_action,
        "projection_action": geometry["P"] @ F,
        "hidden_fraction": float(np.linalg.norm(hidden_action) / denominator),
        "decomposition_relative_defect": float(
            np.linalg.norm(macro_action + hidden_action - F) / denominator
        ),
        "hidden_projection_relative_defect": float(
            np.linalg.norm(hidden_action - geometry["P"] @ F) / denominator
        ),
    }


def _rate_metrics(data: dict, geometry: dict[str, np.ndarray]) -> tuple[dict, dict]:
    state = np.asarray(data["state"], dtype=float)
    columns = np.asarray(data["columns"], dtype=float)
    rate = np.asarray(data["continuous_rate"], dtype=float)
    action = np.asarray(data["continuous_reaction_action"], dtype=float)
    free = np.asarray(data["tangent"].scaled_base_rate_per_s, dtype=float)
    reaction = data["reaction"]
    if not np.array_equal(state, geometry["anchor_state"]):
        raise RuntimeError("fresh fixed-Q rate state is not the exact chart anchor")
    model, _candidate, _fiber = manifest.parent.manifest.parent._model_and_inputs()
    if not np.array_equal(columns, np.asarray(model.columns, dtype=float)):
        raise RuntimeError("fixed-Q and exact-chart primitive scalings differ")
    coordinate_rate = geometry["coordinate_jacobian"] @ rate
    decomposition = _decompose_rate(coordinate_rate, geometry)
    free_constraint = reaction.q3_scaled_derivative @ free
    action_constraint = reaction.q3_scaled_derivative @ action
    total_constraint = reaction.q3_scaled_derivative @ rate
    constraint_scale = max(
        float(np.linalg.norm(free_constraint)),
        float(np.linalg.norm(action_constraint)),
        np.finfo(float).tiny,
    )
    physical = rate_source._state_audit(data["context"], state)
    metrics = {
        "anchor_state_bitwise_exact": True,
        "primitive_scalings_bitwise_exact": True,
        "coordinate_rate_norm": float(np.linalg.norm(coordinate_rate)),
        "macro_rate_norm": float(np.linalg.norm(decomposition["G"])),
        "hidden_rate_norm": float(np.linalg.norm(decomposition["H"])),
        "hidden_coordinate_rate_fraction": decomposition["hidden_fraction"],
        "coordinate_rate_decomposition_relative_defect": decomposition[
            "decomposition_relative_defect"
        ],
        "hidden_action_projection_relative_defect": decomposition[
            "hidden_projection_relative_defect"
        ],
        "fixed_Q_rate_tangency_relative_defect": float(
            np.linalg.norm(total_constraint) / constraint_scale
        ),
        "maximum_reaction_ledger_relative_defect": float(
            reaction.maximum_reaction_ledger_relative_defect
        ),
        "raw_Schur_rank": int(reaction.raw_schur_numerical_rank),
        "raw_Schur_condition_number": float(reaction.raw_schur_condition_number),
        "maximum_raw_Schur_solve_relative_defect": float(
            reaction.maximum_raw_schur_solve_relative_defect
        ),
        "minimum_reconstruction_factor": min(
            float(reaction.minimum_q3_reconstruction_factor),
            float(physical["minimum_reconstruction_factor"]),
        ),
        "maximum_height_ratio": float(physical["maximum_h_over_r"]),
        "minimum_scattering_optical_depth": float(
            physical["minimum_scattering_optical_depth"]
        ),
        "new_exact_fixed_Q_rate_evaluations": 1,
        "new_complete_generator_assemblies": 0,
        "new_intrinsic_hidden_roots": 0,
        "new_coordinate_chart_retractions": 0,
        "propagated_states": 0,
        "sealed_16ms_truth_calls": 0,
    }
    arrays = {
        "anchor_primitive_state": state,
        "primitive_column_scales": columns,
        "conservation_row_scales": np.asarray(data["rows"], dtype=float),
        "continuous_scaled_fixed_Q_rate_per_s": rate,
        "continuous_scaled_free_rate_per_s": free,
        "continuous_scaled_reaction_action_per_s": action,
        "continuous_multiplier": np.asarray(data["continuous_multiplier"]),
        "coordinate_rate_F470_per_s": decomposition["F"],
        "macro_rate_G82_per_s": decomposition["G"],
        "hidden_rate_H388_per_s": decomposition["H"],
        "hidden_action_ZH470_per_s": decomposition["hidden_action"],
        "macro_action_LG470_per_s": decomposition["macro_action"],
        "fiber_projection_PF470_per_s": decomposition["projection_action"],
        "q3_scaled_rate_tangency": total_constraint,
        "q3_scaled_free_action": free_constraint,
        "q3_scaled_reaction_action": action_constraint,
    }
    return metrics, arrays


def _checks(metrics: dict) -> dict[str, bool]:
    return {
        "anchor_state_bitwise_exact": metrics["anchor_state_bitwise_exact"],
        "primitive_scalings_bitwise_exact": metrics[
            "primitive_scalings_bitwise_exact"
        ],
        "coordinate_rate_decomposition": metrics[
            "coordinate_rate_decomposition_relative_defect"
        ]
        <= manifest.DUAL_GEOMETRY_GATE,
        "hidden_action_projection": metrics[
            "hidden_action_projection_relative_defect"
        ]
        <= manifest.DUAL_GEOMETRY_GATE,
        "fixed_Q_rate_tangent": metrics["fixed_Q_rate_tangency_relative_defect"]
        <= FIXED_Q_TANGENCY_GATE,
        "reaction_ledger": metrics["maximum_reaction_ledger_relative_defect"]
        <= REACTION_LEDGER_GATE,
        "Schur_rank": metrics["raw_Schur_rank"] == 3,
        "Schur_condition": metrics["raw_Schur_condition_number"]
        <= SCHUR_CONDITION_GATE,
        "reconstruction": metrics["minimum_reconstruction_factor"]
        >= RECONSTRUCTION_GATE,
        "height": metrics["maximum_height_ratio"] <= HEIGHT_RATIO_GATE,
        "optical_depth": metrics["minimum_scattering_optical_depth"]
        >= OPTICAL_DEPTH_GATE,
        "truth_budget": metrics["new_exact_fixed_Q_rate_evaluations"] == 1,
        "generator_budget": metrics["new_complete_generator_assemblies"] == 0,
        "root_budget": metrics["new_intrinsic_hidden_roots"] == 0,
        "propagation_budget": metrics["propagated_states"] == 0,
        "sealed_budget": metrics["sealed_16ms_truth_calls"] == 0,
    }


def _update_catalog(summary: dict, status: str) -> None:
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
                    "scientific_status": status,
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
    began = time.perf_counter()
    frozen = _validate_manifest(require_clean=True)
    implementation = _validate_implementation_lock()
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("primary hidden anchor evidence already exists")
    geometry = _geometry()
    print("f25dg: evaluate one fresh exact fixed-Q rate at 20 ms", flush=True)
    data = rate_source._state_data("primary_20ms")
    metrics, arrays = _rate_metrics(data, geometry)
    checks = _checks(metrics)
    infrastructure_passed = bool(all(checks.values()))
    hidden_gate_passed = bool(
        metrics["hidden_coordinate_rate_fraction"] <= HIDDEN_FRACTION_GATE
    )
    passed = bool(infrastructure_passed and hidden_gate_passed)
    classification = PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION
    authorized_next = PASS_AUTHORIZED_NEXT if passed else None
    metrics["initial_hidden_fraction_gate"] = HIDDEN_FRACTION_GATE
    metrics["initial_hidden_fraction_gate_passed"] = hidden_gate_passed
    metrics["infrastructure_checks_passed"] = infrastructure_passed
    metrics["total_wall_seconds"] = time.perf_counter() - began

    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_npz(CANONICAL_DIRECTORY / "primary_anchor_rate_arrays.npz", arrays)
    _write_json(
        CANONICAL_DIRECTORY / "primary_anchor_rate_metrics.json",
        {
            "metrics": metrics,
            "checks": checks,
            "hidden_fraction_gate_passed": hidden_gate_passed,
            "passed": passed,
        },
    )
    _write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {**implementation, "manifest_commit": MANIFEST_COMMIT, **frozen},
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "definitions_only": False,
        "parent_hidden_root_manifest_preserved": True,
        "dual_consistent_hidden_rate_used": True,
        "naive_Z_transpose_F_used": False,
        "anchor_hidden_fraction": metrics["hidden_coordinate_rate_fraction"],
        "anchor_hidden_fraction_gate": HIDDEN_FRACTION_GATE,
        "anchor_hidden_fraction_gate_passed": hidden_gate_passed,
        "complete_generator_assembled": False,
        "hidden_root_attempted": False,
        "root_not_attempted_because_anchor_gate_failed": not hidden_gate_passed,
        "new_exact_fixed_Q_rate_evaluations": 1,
        "new_complete_generator_assemblies": 0,
        "new_intrinsic_hidden_roots": 0,
        "propagated_states": 0,
        "sealed_16ms_opened": False,
        "physical_microburst_authorized": False,
        "online_solver_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": authorized_next,
    }
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    source_files = (
        THIS_RUNNER,
        THIS_TEST,
        manifest.THIS_RUNNER,
        manifest.THIS_TEST,
        rate_source.THIS_RUNNER,
        "src/imri_qpe/layer3_minidisk_1d/causal_inner_fixed_q.py",
        "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_tangent.py",
    )
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "SUPPORTED" if passed else "REJECTED",
            **implementation,
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
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
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
                "# Primary hidden-anchor preflight WP10c9d6c7c3b5c4f25dg",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                "One fresh exact fixed-Q rate was evaluated at the bitwise exact 20 ms anchor. The dual-consistent hidden rate H=Z^T(I-LR)F was used; the mathematically invalid shortcut Z^T F was not used.",
                "",
                f"The hidden coordinate-rate fraction is `{metrics['hidden_coordinate_rate_fraction']:.16e}` against the prospectively frozen maximum `{HIDDEN_FRACTION_GATE:.2f}`.",
                "",
                f"The fixed-Q tangency defect is `{metrics['fixed_Q_rate_tangency_relative_defect']:.6e}` and the coordinate decomposition defect is `{metrics['coordinate_rate_decomposition_relative_defect']:.6e}`.",
                "",
                (
                    "The anchor gate passed. A separate complete-tangent and root stage is authorized; no root was attempted here."
                    if passed
                    else "The anchor gate failed. Fail-fast ordering therefore forbade a complete generator assembly and hidden root attempt."
                ),
                "",
                "No state was propagated and the sealed 16 ms state was not opened. No physical microburst, online solver, predictive cycle, or reduced slow evolution is authorized.",
                "",
            )
        ),
        encoding="utf-8",
    )
    _update_catalog(summary, "SUPPORTED" if passed else "REJECTED")
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
