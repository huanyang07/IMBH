#!/usr/bin/env python3
"""Execute the active-8 mixed-direction exact-geometry preflight."""

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
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_active8_mixed_parity_database_manifest_wp10c9d6c7c3b5c4f25bj as manifest  # noqa: E402
import run_causal_inner_expanded_departure_chart_preflight_wp10c9d6c7c3b5c4f25bc as high_chart  # noqa: E402
import run_causal_inner_exact_geometric_departure_chart_preflight_wp10c9d6c7c3b5c4f25ay as chart_tools  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25bk"
MANIFEST_COMMIT = "1e07757fc801154f7c98697d45be762cb910f2ab"
MANIFEST_PARENT = "4cb13373784e00fd962d361f9b0791beb0066695"
MANIFEST_TREE = "72240a2e64abcd741d56444489cf2d9242cad4cc"

PASS_CLASSIFICATION = "active8_mixed_geometry_passed_rate_fit_authorized"
FAIL_CLASSIFICATION = "active8_mixed_geometry_failed_rate_fit_blocked"

ARTIFACT = (
    "causal_inner_active8_mixed_geometry_preflight_"
    "wp10c9d6c7c3b5c4f25bk"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_active8_mixed_geometry_preflight_"
    "wp10c9d6c7c3b5c4f25bk.py"
)
THIS_TEST = (
    "tests/test_causal_inner_active8_mixed_geometry_preflight_"
    "wp10c9d6c7c3b5c4f25bk.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ACTIVE8_MIXED_GEOMETRY_"
    "PREFLIGHT_WP10C9D6C7C3B5C4F25BK_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
DESIGN_PATH = manifest.ARTIFACT_DIRECTORY / "mixed_direction_design.npz"


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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
        return {name: np.asarray(source[name]) for name in source.files}


def _validate_manifest(*, require_clean: bool) -> dict:
    if _git("rev-parse", MANIFEST_COMMIT) != MANIFEST_COMMIT:
        raise RuntimeError("mixed-geometry manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("mixed-geometry manifest lineage changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("mixed-geometry manifest tree changed")
    hashes = _checksums(manifest.ARTIFACT_DIRECTORY)
    summary = _read(manifest.ARTIFACT_DIRECTORY / "summary.json")
    contract = _read(manifest.ARTIFACT_DIRECTORY / "contract.json")
    provenance = _read(manifest.ARTIFACT_DIRECTORY / "provenance.json")
    lock = _read(manifest.ARTIFACT_DIRECTORY / "parent_lock.json")
    if (
        not summary["passed"]
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["planned_candidate_count"] != manifest.PLANNED_CANDIDATES
        or contract["exact_geometry"]["rate_or_reaction_lift_used"]
        or contract["exact_geometry"]["propagated_states"] != 0
    ):
        raise RuntimeError("mixed-geometry execution authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"manifest source changed: {relative}")
    for name, path in (
        ("online_470_geometry", manifest.GEOMETRY_PATH),
        ("low_chart", manifest.LOW_CHART_PATH),
        ("high_chart", manifest.HIGH_CHART_PATH),
        ("low_rate", manifest.LOW_RATE_PATH),
        ("high_rate", manifest.HIGH_RATE_PATH),
    ):
        if _sha(path) != lock["decisive_input_hashes"][name]:
            raise RuntimeError(f"mixed-geometry input changed: {path}")
    if require_clean and not _tracked_tree_clean():
        raise RuntimeError("mixed-geometry preflight requires a clean tracked tree")
    for name, expected in chart_tools.coordinate_tools.THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    return {"summary": summary, "contract": contract, "hashes": hashes}


def _candidate_specifications() -> list[dict]:
    design = _load_npz(DESIGN_PATH)
    groups = (
        (
            "training",
            0,
            design["training_directions_active8"],
            manifest.HIGH_COMPONENT_BOUND,
            "0p01",
        ),
        (
            "tuning_high",
            manifest.TRAINING_DIRECTION_COUNT,
            design["tuning_directions_active8"],
            manifest.HIGH_COMPONENT_BOUND,
            "0p01",
        ),
        (
            "holdout",
            manifest.TRAINING_DIRECTION_COUNT + manifest.TUNING_DIRECTION_COUNT,
            design["holdout_directions_active8"],
            manifest.HIGH_COMPONENT_BOUND,
            "0p01",
        ),
        (
            "tuning_low",
            manifest.TRAINING_DIRECTION_COUNT,
            design["tuning_directions_active8"],
            manifest.LOW_COMPONENT_BOUND,
            "0p005",
        ),
    )
    specifications = []
    for split, offset, directions, bound, amplitude in groups:
        for local_index in range(directions.shape[1]):
            specifications.append(
                {
                    "split": split,
                    "split_direction_index": local_index,
                    "global_direction_index": offset + local_index,
                    "active_direction": np.asarray(
                        directions[:, local_index], dtype=float
                    ),
                    "component_bound": float(bound),
                    "amplitude_label": amplitude,
                }
            )
    return specifications


def _retraction_contract() -> dict:
    contract = manifest._contract()
    contract["binding_preflight_gates"] = contract["binding_geometry_gates"]
    contract["exact_geometric_retraction"] = high_chart.manifest._contract()[
        "exact_geometric_retraction"
    ]
    return contract


def _empty_progress(identity: dict) -> dict:
    return {
        "identity": identity,
        "candidates": [],
        "failures": [],
        "candidate_primitive_states": np.empty((0, 112, 5), dtype=float),
        "candidate_scaled_deltas": np.empty((0, 560), dtype=float),
        "candidate_departure_coordinates": np.empty((0, 28), dtype=float),
        "candidate_active_directions": np.empty((0, 8), dtype=float),
        "candidate_split_codes": np.empty((0,), dtype=np.int64),
        "candidate_global_direction_indices": np.empty((0,), dtype=np.int64),
        "candidate_component_bounds": np.empty((0,), dtype=float),
        "candidate_signs": np.empty((0,), dtype=np.int64),
    }


def _progress_identity() -> dict:
    return {
        "execution_commit": _git("rev-parse", "HEAD"),
        "manifest_commit": MANIFEST_COMMIT,
        "manifest_sha256": _sha(
            manifest.ARTIFACT_DIRECTORY / "SHA256SUMS.txt"
        ),
        "runner_sha256": _sha(ROOT / THIS_RUNNER),
        "test_sha256": _sha(ROOT / THIS_TEST),
        "design_sha256": _sha(DESIGN_PATH),
    }


def _save_progress(progress: dict) -> None:
    SCRATCH_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(
        SCRATCH_DIRECTORY / "progress.json",
        {
            "identity": progress["identity"],
            "candidates": progress["candidates"],
            "failures": progress["failures"],
        },
    )
    _write_npz(
        SCRATCH_DIRECTORY / "progress.npz",
        {
            name: progress[name]
            for name in (
                "candidate_primitive_states",
                "candidate_scaled_deltas",
                "candidate_departure_coordinates",
                "candidate_active_directions",
                "candidate_split_codes",
                "candidate_global_direction_indices",
                "candidate_component_bounds",
                "candidate_signs",
            )
        },
    )


def _load_or_create_progress() -> dict:
    identity = _progress_identity()
    json_path = SCRATCH_DIRECTORY / "progress.json"
    npz_path = SCRATCH_DIRECTORY / "progress.npz"
    if not json_path.exists() and not npz_path.exists():
        return _empty_progress(identity)
    if not json_path.exists() or not npz_path.exists():
        raise RuntimeError("mixed-geometry scratch checkpoint is incomplete")
    recorded = _read(json_path)
    if recorded["identity"] != identity:
        raise RuntimeError("mixed-geometry scratch identity changed")
    arrays = _load_npz(npz_path)
    progress = {
        "identity": identity,
        "candidates": recorded["candidates"],
        "failures": recorded["failures"],
        **arrays,
    }
    count = len(progress["candidates"])
    if count % 2 or any(
        progress[name].shape[0] != count
        for name in (
            "candidate_primitive_states",
            "candidate_scaled_deltas",
            "candidate_departure_coordinates",
            "candidate_active_directions",
            "candidate_split_codes",
            "candidate_global_direction_indices",
            "candidate_component_bounds",
            "candidate_signs",
        )
    ):
        raise RuntimeError("mixed-geometry scratch dimensions changed")
    return progress


def _append(array: np.ndarray, value, *, dtype=float) -> np.ndarray:
    item = np.asarray(value, dtype=dtype)
    return np.concatenate((array, item.reshape((1,) + item.shape)), axis=0)


def _execute() -> tuple[dict, dict[str, np.ndarray]]:
    components = high_chart._prepare_components()
    family_metrics, family = chart_tools._departure_family()
    design = _load_npz(DESIGN_PATH)
    specifications = _candidate_specifications()
    contract = _retraction_contract()
    progress = _load_or_create_progress()
    began = time.perf_counter()
    split_codes = {"training": 0, "tuning_high": 1, "holdout": 2, "tuning_low": 3}
    start_pair = len(progress["candidates"]) // 2
    resumed_candidate_count = len(progress["candidates"])

    if progress["failures"]:
        start_pair = len(specifications)
    for pair_index in range(start_pair, len(specifications)):
        specification = specifications[pair_index]
        direction = family["energy_directions"] @ specification["active_direction"]
        direction /= np.linalg.norm(direction)
        pair_coordinates = []
        pair_indices = []
        for sign in (-1, 1):
            candidate_index = len(progress["candidates"])
            try:
                metrics, arrays = chart_tools._retract_candidate(
                    components,
                    family["departure_basis"],
                    family["stable_memory_basis"],
                    direction,
                    sign,
                    specification["component_bound"],
                    contract,
                )
                metrics.update(
                    {
                        "candidate_index": candidate_index,
                        "pair_index": pair_index,
                        "split": specification["split"],
                        "split_direction_index": specification[
                            "split_direction_index"
                        ],
                        "global_direction_index": specification[
                            "global_direction_index"
                        ],
                        "amplitude_label": specification["amplitude_label"],
                        "component_bound_fraction": (
                            metrics["final_scaled_component"]
                            / specification["component_bound"]
                        ),
                    }
                )
                progress["candidates"].append(metrics)
                progress["candidate_primitive_states"] = _append(
                    progress["candidate_primitive_states"], arrays["primitive_state"]
                )
                progress["candidate_scaled_deltas"] = _append(
                    progress["candidate_scaled_deltas"], arrays["scaled_delta"]
                )
                progress["candidate_departure_coordinates"] = _append(
                    progress["candidate_departure_coordinates"],
                    arrays["departure_coordinates"],
                )
                progress["candidate_active_directions"] = _append(
                    progress["candidate_active_directions"],
                    specification["active_direction"],
                )
                progress["candidate_split_codes"] = _append(
                    progress["candidate_split_codes"],
                    split_codes[specification["split"]],
                    dtype=np.int64,
                )
                progress["candidate_global_direction_indices"] = _append(
                    progress["candidate_global_direction_indices"],
                    specification["global_direction_index"],
                    dtype=np.int64,
                )
                progress["candidate_component_bounds"] = _append(
                    progress["candidate_component_bounds"],
                    specification["component_bound"],
                )
                progress["candidate_signs"] = _append(
                    progress["candidate_signs"], sign, dtype=np.int64
                )
                pair_coordinates.append(arrays["departure_coordinates"])
                pair_indices.append(candidate_index)
                status = "accepted"
            except chart_tools.ChartRetractionFailure as error:
                progress["failures"].append(
                    {
                        "candidate_index": candidate_index,
                        "pair_index": pair_index,
                        "split": specification["split"],
                        "split_direction_index": specification[
                            "split_direction_index"
                        ],
                        "global_direction_index": specification[
                            "global_direction_index"
                        ],
                        "component_bound": specification["component_bound"],
                        "amplitude_label": specification["amplitude_label"],
                        "sign": sign,
                        "reason": str(error),
                        "diagnostics": error.diagnostics,
                    }
                )
                status = "failed"
            print(
                json.dumps(
                    {
                        "candidate": candidate_index + 1,
                        "total": manifest.PLANNED_CANDIDATES,
                        "pair": pair_index,
                        "split": specification["split"],
                        "direction": specification["split_direction_index"],
                        "component_bound": specification["component_bound"],
                        "sign": sign,
                        "status": status,
                        "elapsed_this_process_seconds": time.perf_counter() - began,
                    }
                ),
                flush=True,
            )
            if progress["failures"]:
                break
        if progress["failures"]:
            _save_progress(progress)
            break
        denominator = max(
            float(np.linalg.norm(pair_coordinates[0]))
            + float(np.linalg.norm(pair_coordinates[1])),
            np.finfo(float).tiny,
        )
        odd = float(
            np.linalg.norm(pair_coordinates[0] + pair_coordinates[1]) / denominator
        )
        for index in pair_indices:
            progress["candidates"][index]["pair_coordinate_odd_symmetry_defect"] = odd
        _save_progress(progress)

    candidates = progress["candidates"]
    failures = progress["failures"]

    def maximum(name: str, default=math.inf) -> float:
        values = [item.get(name, default) for item in candidates]
        return float(max(values)) if values else float(default)

    def minimum(name: str, default=-math.inf) -> float:
        values = [item.get(name, default) for item in candidates]
        return float(min(values)) if values else float(default)

    metrics = {
        "departure_family": family_metrics,
        "planned_candidate_count": manifest.PLANNED_CANDIDATES,
        "completed_candidate_count": len(candidates),
        "failed_candidate_count": len(failures),
        "failures": failures,
        "maximum_coordinate_residual_infinity": maximum(
            "coordinate_residual_infinity"
        ),
        "maximum_normalized_Q3_defect": maximum("normalized_Q3_defect"),
        "maximum_final_scaled_component": maximum("final_scaled_component"),
        "maximum_component_bound_fraction": maximum("component_bound_fraction"),
        "minimum_reconstruction_factor": minimum(
            "minimum_reconstruction_factor", math.inf
        ),
        "maximum_reconstruction_factor": maximum("maximum_reconstruction_factor"),
        "maximum_coordinate_Jacobian_condition_number": maximum(
            "maximum_coordinate_Jacobian_condition_number"
        ),
        "minimum_departure_direction_alignment_cosine": minimum(
            "departure_direction_alignment_cosine"
        ),
        "maximum_departure_transverse_fraction": maximum(
            "departure_transverse_fraction"
        ),
        "maximum_pair_coordinate_odd_symmetry_defect": maximum(
            "pair_coordinate_odd_symmetry_defect"
        ),
        "maximum_stable_memory_coordinate_leakage_norm": maximum(
            "stable_memory_coordinate_leakage_norm"
        ),
        "maximum_H_over_R": maximum("maximum_H_over_R"),
        "minimum_scattering_optical_depth": minimum(
            "minimum_scattering_optical_depth"
        ),
        "maximum_Newton_corrections": maximum("Newton_corrections"),
        "maximum_radius_rescalings": maximum("radius_rescalings"),
        "resumed_candidate_count": resumed_candidate_count,
        "wall_seconds_this_process": time.perf_counter() - began,
        "nonbase_continuous_rate_evaluations": 0,
        "new_full_generator_assemblies": 0,
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "candidates": candidates,
    }
    arrays = {
        **family,
        **design,
        **{
            name: progress[name]
            for name in (
                "candidate_primitive_states",
                "candidate_scaled_deltas",
                "candidate_departure_coordinates",
                "candidate_active_directions",
                "candidate_split_codes",
                "candidate_global_direction_indices",
                "candidate_component_bounds",
                "candidate_signs",
            )
        },
    }
    return metrics, arrays


def _gate_checks(metrics: dict, gates: dict) -> dict:
    return {
        "candidate_count": metrics["completed_candidate_count"]
        == gates["completed_candidate_count_equal"],
        "failure_count": metrics["failed_candidate_count"]
        == gates["failed_candidate_count_equal"],
        "coordinate_closure": metrics["maximum_coordinate_residual_infinity"]
        <= gates["maximum_coordinate_residual_infinity"],
        "Q3_closure": metrics["maximum_normalized_Q3_defect"]
        <= gates["maximum_normalized_Q3_defect"],
        "component_trust": metrics["maximum_final_scaled_component"]
        <= gates["maximum_final_scaled_component"] * (1.0 + 1.0e-12),
        "per_candidate_component_trust": metrics["maximum_component_bound_fraction"]
        <= 1.0 + 1.0e-12,
        "reconstruction_minimum": metrics["minimum_reconstruction_factor"]
        >= gates["minimum_reconstruction_factor"],
        "reconstruction_maximum": metrics["maximum_reconstruction_factor"]
        <= gates["maximum_reconstruction_factor"],
        "coordinate_condition": metrics[
            "maximum_coordinate_Jacobian_condition_number"
        ]
        <= gates["maximum_coordinate_Jacobian_condition_number"],
        "direction_alignment": metrics[
            "minimum_departure_direction_alignment_cosine"
        ]
        >= gates["minimum_departure_direction_alignment_cosine"],
        "transverse_distortion": metrics["maximum_departure_transverse_fraction"]
        <= gates["maximum_departure_transverse_fraction"],
        "odd_symmetry": metrics["maximum_pair_coordinate_odd_symmetry_defect"]
        <= gates["maximum_pair_coordinate_odd_symmetry_defect"],
        "height": metrics["maximum_H_over_R"] <= gates["maximum_H_over_R"],
        "optical_depth": metrics["minimum_scattering_optical_depth"]
        >= gates["minimum_scattering_optical_depth"],
        "rate_budget": metrics["nonbase_continuous_rate_evaluations"]
        == gates["nonbase_continuous_rate_evaluations_equal"],
        "generator_budget": metrics["new_full_generator_assemblies"]
        == gates["new_full_generator_assemblies_equal"],
        "root_budget": metrics["new_nonlinear_roots"]
        == gates["new_nonlinear_roots_equal"],
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
                    "scientific_status": "CERTIFIED" if summary["passed"] else "REJECTED",
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
        raise RuntimeError("mixed-geometry preflight is already canonicalized")
    metrics, arrays = _execute()
    checks = _gate_checks(
        metrics, frozen["contract"]["binding_geometry_gates"]
    )
    passed = all(checks.values())
    classification = PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION
    authorized_next = WORK_PACKAGE[:-1] + "l" if passed else None
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_json(CANONICAL_DIRECTORY / "metrics.json", {"checks": checks, **metrics})
    _write_npz(CANONICAL_DIRECTORY / "mixed_geometry_database.npz", arrays)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "completed_candidate_count": metrics["completed_candidate_count"],
        "failed_candidate_count": metrics["failed_candidate_count"],
        "maximum_coordinate_residual_infinity": metrics[
            "maximum_coordinate_residual_infinity"
        ],
        "maximum_normalized_Q3_defect": metrics["maximum_normalized_Q3_defect"],
        "maximum_departure_transverse_fraction": metrics[
            "maximum_departure_transverse_fraction"
        ],
        "nonbase_continuous_rate_evaluations": 0,
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
            "resumed_from_candidate_count": metrics["resumed_candidate_count"],
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "source_hashes": {
                relative: _sha(ROOT / relative) for relative in source_files
            },
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "thread_environment": chart_tools.coordinate_tools.THREAD_ENVIRONMENT,
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
                "# Active-8 mixed geometry preflight WP10c9d6c7c3b5c4f25bk",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                f"Completed `{metrics['completed_candidate_count']}` of `{manifest.PLANNED_CANDIDATES}` planned signed exact retractions; failures: `{metrics['failed_candidate_count']}`.",
                "",
                f"Maximum C_phys closure is `{metrics['maximum_coordinate_residual_infinity']:.6e}`; maximum normalized Q3 defect is `{metrics['maximum_normalized_Q3_defect']:.6e}`; maximum transverse departure fraction is `{metrics['maximum_departure_transverse_fraction']:.6e}`.",
                "",
                f"Authorized next work package: `{authorized_next}`. No nonbase rate was evaluated and no state was propagated.",
                "",
            )
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    if SCRATCH_DIRECTORY.exists():
        shutil.rmtree(SCRATCH_DIRECTORY)
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
