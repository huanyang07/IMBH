#!/usr/bin/env python3
"""Execute the exact departure-chart rung at component bound 0.02."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_amplitude_0p02_departure_chart_manifest_wp10c9d6c7c3b5c4f25bf as manifest  # noqa: E402
import run_causal_inner_expanded_departure_chart_preflight_wp10c9d6c7c3b5c4f25bc as prior_chart  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25bg"
MANIFEST_COMMIT = "8302b7f2713114d16d0e06a41471f98ef34337f5"
MANIFEST_PARENT = "d24ab4c02942e9920eaaa512d6a954da1cba60b8"
MANIFEST_TREE = "41c13136292c7cc9bf6eda80065976fd14d26229"

PASS_CLASSIFICATION = (
    "exact_departure_chart_amplitude_0p02_passed_"
    "sixteen_rate_screen_manifest_authorized"
)
FAIL_CLASSIFICATION = (
    "exact_departure_chart_amplitude_0p02_failed_"
    "nonlinear_amplitude_expansion_blocked"
)

ARTIFACT = (
    "causal_inner_amplitude_0p02_departure_chart_preflight_"
    "wp10c9d6c7c3b5c4f25bg"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_amplitude_0p02_departure_chart_preflight_"
    "wp10c9d6c7c3b5c4f25bg.py"
)
THIS_TEST = (
    "tests/test_causal_inner_amplitude_0p02_departure_chart_preflight_"
    "wp10c9d6c7c3b5c4f25bg.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_AMPLITUDE_0P02_DEPARTURE_"
    "CHART_PREFLIGHT_WP10C9D6C7C3B5C4F25BG_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

FROZEN_RETRACTION_SOURCES = (
    prior_chart.THIS_RUNNER,
    prior_chart.manifest.parent.manifest.parent.THIS_RUNNER,
)

_plain = prior_chart._plain
_read = prior_chart._read
_write_json = prior_chart._write_json
_sha = prior_chart._sha
_git = prior_chart._git
_tracked_tree_clean = prior_chart._tracked_tree_clean
_checksums = prior_chart._checksums
chart_tools = prior_chart.chart_tools


def _validate_manifest(*, require_clean: bool) -> dict:
    if _git("rev-parse", MANIFEST_COMMIT) != MANIFEST_COMMIT:
        raise RuntimeError("amplitude-0.02 chart manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("amplitude-0.02 chart manifest lineage changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("amplitude-0.02 chart manifest tree changed")
    hashes = _checksums(manifest.ARTIFACT_DIRECTORY)
    summary = _read(manifest.ARTIFACT_DIRECTORY / "summary.json")
    contract = _read(manifest.ARTIFACT_DIRECTORY / "contract.json")
    provenance = _read(manifest.ARTIFACT_DIRECTORY / "provenance.json")
    lock = _read(manifest.ARTIFACT_DIRECTORY / "parent_lock.json")
    if (
        not summary["passed"]
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["maximum_scaled_component_bound"] != manifest.COMPONENT_BOUND
        or summary["planned_candidate_count"] != manifest.PLANNED_CANDIDATES
        or summary["planned_nonbase_continuous_rate_evaluations"] != 0
        or contract["exact_geometric_retraction"]["rate_reaction_lift_used"]
        or not contract["candidate_family"][
            "prior_states_are_not_propagated_or_extrapolated"
        ]
    ):
        raise RuntimeError("amplitude-0.02 chart execution authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"manifest source changed: {relative}")
    for name, path in (
        ("prior_expanded_departure_chart", manifest.PRIOR_CHART_PATH),
        ("base_geometric_chart", manifest.BASE_CHART_PATH),
        ("online_470_geometry", manifest.GEOMETRY_PATH),
    ):
        if _sha(path) != lock["decisive_input_hashes"][name]:
            raise RuntimeError(f"amplitude-0.02 chart input changed: {path}")
    _checksums(manifest.parent.CANONICAL_DIRECTORY)
    _checksums(manifest.PRIOR_CHART_DIRECTORY)
    for relative in FROZEN_RETRACTION_SOURCES:
        if _git("hash-object", relative) != _git(
            "rev-parse", f"{MANIFEST_COMMIT}:{relative}"
        ):
            raise RuntimeError(f"retraction source changed after manifest: {relative}")
    if require_clean and not _tracked_tree_clean():
        raise RuntimeError("amplitude-0.02 chart preflight requires a clean tracked tree")
    for name, expected in chart_tools.coordinate_tools.THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    return {"summary": summary, "contract": contract, "hashes": hashes}


def _execute() -> tuple[dict, dict[str, np.ndarray]]:
    components = prior_chart._prepare_components()
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
    return prior_chart._gate_checks(metrics, gates)


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
        raise RuntimeError("amplitude-0.02 chart preflight is already canonicalized")
    metrics, arrays = _execute()
    checks = _gate_checks(metrics, frozen["contract"]["binding_preflight_gates"])
    passed = all(checks.values())
    classification = PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION
    authorized_next = (
        "definitions_only_amplitude_0p02_sixteen_rate_screen_manifest"
        if passed
        else None
    )
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_json(CANONICAL_DIRECTORY / "metrics.json", {"checks": checks, **metrics})
    np.savez_compressed(
        CANONICAL_DIRECTORY / "amplitude_0p02_departure_chart.npz", **arrays
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
            "prior_rate_screen_hashes": _checksums(
                manifest.parent.CANONICAL_DIRECTORY
            ),
        },
    )
    source_files = (
        THIS_RUNNER,
        THIS_TEST,
        manifest.THIS_RUNNER,
        manifest.THIS_TEST,
        *FROZEN_RETRACTION_SOURCES,
    )
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
                "# Amplitude-0.02 departure-chart preflight WP10c9d6c7c3b5c4f25bg",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                f"All `{metrics['completed_candidate_count']}` completed states use component bound `{manifest.COMPONENT_BOUND:.3e}`. Failures: `{metrics['failed_candidate_count']}`.",
                "",
                f"Maximum C_phys closure is `{metrics['maximum_coordinate_residual_infinity']:.6e}`; maximum normalized Q3 defect is `{metrics['maximum_normalized_Q3_defect']:.6e}`; maximum H/R is `{metrics['maximum_H_over_R']:.6e}`.",
                "",
                f"Authorized next artifact: `{authorized_next}`. No amplitude-0.02 rate, closure fit, or predictive trajectory is claimed.",
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
