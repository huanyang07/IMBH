#!/usr/bin/env python3
"""Execute the amplitude-0.01 exact geometric departure-chart rung."""

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

import run_causal_inner_expanded_departure_chart_manifest_wp10c9d6c7c3b5c4f25bb as manifest  # noqa: E402
import run_causal_inner_exact_geometric_departure_chart_preflight_wp10c9d6c7c3b5c4f25ay as chart_tools  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_fixed_q import (  # noqa: E402
    causal_five_field_exterior_q3,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25bc"
MANIFEST_COMMIT = "8ec8d8c7392d366aa57fb37f1af85c18785b6e58"
MANIFEST_PARENT = "fec32c52052163ce280dede4a9e45eae669ced04"
MANIFEST_TREE = "40b62caf227fe3cf64e17c44ac76f8c5d7bd9c32"

PASS_CLASSIFICATION = (
    "expanded_exact_departure_chart_amplitude_0p01_passed_"
    "sixteen_rate_screen_manifest_authorized"
)
FAIL_CLASSIFICATION = (
    "expanded_exact_departure_chart_amplitude_0p01_failed_"
    "nonlinear_amplitude_expansion_blocked"
)

ARTIFACT = (
    "causal_inner_expanded_departure_chart_preflight_"
    "wp10c9d6c7c3b5c4f25bc"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_expanded_departure_chart_preflight_"
    "wp10c9d6c7c3b5c4f25bc.py"
)
THIS_TEST = (
    "tests/test_causal_inner_expanded_departure_chart_preflight_"
    "wp10c9d6c7c3b5c4f25bc.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_EXPANDED_DEPARTURE_CHART_"
    "PREFLIGHT_WP10C9D6C7C3B5C4F25BC_2026-08-19.md"
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
        raise RuntimeError("expanded-chart manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("expanded-chart manifest lineage changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("expanded-chart manifest tree changed")
    hashes = _checksums(manifest.ARTIFACT_DIRECTORY)
    summary = _read(manifest.ARTIFACT_DIRECTORY / "summary.json")
    contract = _read(manifest.ARTIFACT_DIRECTORY / "contract.json")
    provenance = _read(manifest.ARTIFACT_DIRECTORY / "provenance.json")
    lock = _read(manifest.ARTIFACT_DIRECTORY / "parent_lock.json")
    if (
        not summary["passed"]
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["maximum_scaled_component_bound"] != manifest.COMPONENT_BOUND
        or summary["planned_nonbase_rate_evaluations"] != 0
        or contract["exact_geometric_retraction"]["rate_reaction_lift_used"]
    ):
        raise RuntimeError("expanded-chart execution authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"manifest source changed: {relative}")
    for name, path in (
        ("base_geometric_chart", manifest.BASE_CHART_PATH),
        ("online_470_geometry", manifest.GEOMETRY_PATH),
    ):
        if _sha(path) != lock["decisive_input_hashes"][name]:
            raise RuntimeError(f"expanded-chart input changed: {path}")
    _checksums(manifest.parent.CANONICAL_DIRECTORY)
    if require_clean and not _tracked_tree_clean():
        raise RuntimeError("expanded-chart preflight requires a clean tracked tree")
    for name, expected in chart_tools.coordinate_tools.THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    return {"summary": summary, "contract": contract, "hashes": hashes}


def _prepare_components() -> dict:
    components = chart_tools.coordinate_tools._coordinate_components()
    components["spatial_nodes"] = chart_tools.coordinate_tools._spatial_nodes(
        components["context"]
    )
    components["coordinate_target"], _factors = (
        chart_tools._coordinate_value_with_factors(
            components["state"], components
        )
    )
    face = 36 * int(components["data"]["layout"].refinement_ratio)
    components["base_q3"], components["base_q3_factors"] = (
        causal_five_field_exterior_q3(
            components["context"],
            components["state"],
            exterior_face_index=face,
        )
    )
    return components


def _execute() -> tuple[dict, dict[str, np.ndarray]]:
    components = _prepare_components()
    family_metrics, family = chart_tools._departure_family()
    contract = manifest._contract()
    candidates = []
    failures = []
    states = []
    deltas = []
    coordinates = []
    began = time.perf_counter()
    for direction_index in range(manifest.DIRECTION_COUNT):
        direction = family["energy_directions"][:, direction_index]
        pair = []
        for sign in manifest.SIGNS:
            index = len(candidates) + len(failures)
            try:
                metrics, arrays = chart_tools._retract_candidate(
                    components,
                    family["departure_basis"],
                    family["stable_memory_basis"],
                    direction,
                    sign,
                    manifest.COMPONENT_BOUND,
                    contract,
                )
                metrics.update(
                    {
                        "candidate_index": index,
                        "direction_index": direction_index,
                        "amplitude_index": 0,
                    }
                )
                candidates.append(metrics)
                states.append(arrays["primitive_state"])
                deltas.append(arrays["scaled_delta"])
                coordinates.append(arrays["departure_coordinates"])
                pair.append(arrays["departure_coordinates"])
                status = "accepted"
            except chart_tools.ChartRetractionFailure as error:
                failures.append(
                    {
                        "candidate_index": index,
                        "direction_index": direction_index,
                        "sign": sign,
                        "reason": str(error),
                        "diagnostics": error.diagnostics,
                    }
                )
                status = "failed"
            print(
                json.dumps(
                    {
                        "candidate": index + 1,
                        "total": manifest.PLANNED_CANDIDATES,
                        "direction": direction_index,
                        "component_bound": manifest.COMPONENT_BOUND,
                        "sign": sign,
                        "status": status,
                        "elapsed_seconds": time.perf_counter() - began,
                    }
                ),
                flush=True,
            )
            if failures:
                break
        if failures:
            break
        denominator = max(
            float(np.linalg.norm(pair[0])) + float(np.linalg.norm(pair[1])),
            np.finfo(float).tiny,
        )
        odd = float(np.linalg.norm(pair[0] + pair[1]) / denominator)
        candidates[-1]["pair_coordinate_odd_symmetry_defect"] = odd
        candidates[-2]["pair_coordinate_odd_symmetry_defect"] = odd

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
        "maximum_coordinate_odd_symmetry_defect": maximum(
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
        "total_wall_seconds": time.perf_counter() - began,
        "nonbase_continuous_rate_evaluations": 0,
        "new_full_generator_assemblies": 0,
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "candidates": candidates,
    }
    arrays = {
        **family,
        "candidate_primitive_states": np.asarray(states, dtype=float),
        "candidate_scaled_deltas": np.asarray(deltas, dtype=float),
        "candidate_departure_coordinates": np.asarray(coordinates, dtype=float),
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
        "odd_symmetry": metrics["maximum_coordinate_odd_symmetry_defect"]
        <= gates["maximum_coordinate_odd_symmetry_defect"],
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
        raise RuntimeError("expanded-chart preflight is already canonicalized")
    metrics, arrays = _execute()
    checks = _gate_checks(metrics, frozen["contract"]["binding_preflight_gates"])
    passed = all(checks.values())
    classification = PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION
    authorized_next = (
        "definitions_only_expanded_amplitude_0p01_sixteen_rate_screen_manifest"
        if passed
        else None
    )
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_json(CANONICAL_DIRECTORY / "metrics.json", {"checks": checks, **metrics})
    np.savez_compressed(
        CANONICAL_DIRECTORY / "expanded_departure_chart.npz", **arrays
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "maximum_scaled_component_bound": manifest.COMPONENT_BOUND,
        "completed_candidate_count": metrics["completed_candidate_count"],
        "failed_candidate_count": metrics["failed_candidate_count"],
        "maximum_coordinate_residual_infinity": metrics[
            "maximum_coordinate_residual_infinity"
        ],
        "maximum_normalized_Q3_defect": metrics["maximum_normalized_Q3_defect"],
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
            "rate_screen_hashes": _checksums(manifest.parent.CANONICAL_DIRECTORY),
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
                "# Expanded departure-chart preflight WP10c9d6c7c3b5c4f25bc",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                f"All `{metrics['completed_candidate_count']}` completed states use the doubled `{manifest.COMPONENT_BOUND:.3e}` scaled-component bound. Failures: `{metrics['failed_candidate_count']}`.",
                "",
                f"Maximum C_phys closure is `{metrics['maximum_coordinate_residual_infinity']:.6e}`; maximum normalized Q3 defect is `{metrics['maximum_normalized_Q3_defect']:.6e}`; maximum H/R is `{metrics['maximum_H_over_R']:.6e}`.",
                "",
                f"Authorized next artifact: `{authorized_next}`. No expanded rate, closure fit, or predictive trajectory is claimed.",
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
