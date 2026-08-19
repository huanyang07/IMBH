#!/usr/bin/env python3
"""Freeze the first doubled-amplitude exact departure-chart rung."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_guarded_departure_rate_screen_wp10c9d6c7c3b5c4f25ba as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25bb"
CLASSIFICATION = (
    "expanded_exact_departure_chart_amplitude_0p01_manifest_frozen_"
    "geometry_only"
)
PARENT_COMMIT = "6a08ffc0afe18581d5480d295adf9607e21caba4"
PARENT_PARENT = "e0d16a4b432a9b264e95341408b2e9dd189e78e6"
PARENT_TREE = "1ff5eccedd931cc960ed4d5e82fdb06f5a84c236"

ARTIFACT = (
    "causal_inner_expanded_departure_chart_manifest_"
    "wp10c9d6c7c3b5c4f25bb"
)
ARTIFACT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_expanded_departure_chart_manifest_"
    "wp10c9d6c7c3b5c4f25bb.py"
)
THIS_TEST = (
    "tests/test_causal_inner_expanded_departure_chart_manifest_"
    "wp10c9d6c7c3b5c4f25bb.py"
)
NEXT_RUNNER = (
    "scripts/run_causal_inner_expanded_departure_chart_preflight_"
    "wp10c9d6c7c3b5c4f25bc.py"
)
NEXT_TEST = (
    "tests/test_causal_inner_expanded_departure_chart_preflight_"
    "wp10c9d6c7c3b5c4f25bc.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_EXPANDED_DEPARTURE_CHART_"
    "MANIFEST_WP10C9D6C7C3B5C4F25BB_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

BASE_CHART_DIRECTORY = parent.manifest.parent.CANONICAL_DIRECTORY
BASE_CHART_PATH = BASE_CHART_DIRECTORY / "geometric_departure_chart.npz"
GEOMETRY_PATH = parent.manifest.GEOMETRY_PATH

ANCHOR = "primary"
COMPONENT_BOUND = 1.0e-2
DIRECTION_COUNT = 8
SIGNS = (-1, 1)
PLANNED_CANDIDATES = DIRECTION_COUNT * len(SIGNS)


def _plain(value):
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


def _write(path: Path, payload) -> None:
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
    for line in (directory / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        if _sha(directory / name) != expected:
            raise RuntimeError(f"checksum mismatch: {directory / name}")
        recorded[name] = expected
    return recorded


def _validate_parent() -> dict:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("guarded rate-screen commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("guarded rate-screen lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("guarded rate-screen tree changed")
    hashes = _checksums(parent.CANONICAL_DIRECTORY)
    summary = _read(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = _read(parent.CANONICAL_DIRECTORY / "metrics.json")
    if (
        not summary["passed"]
        or summary["nonlinear_signal_resolved"]
        or summary["authorized_next"]
        != "definitions_only_expanded_safe_departure_chart_manifest"
        or summary["median_largest_departure_nonlinear_relative_defect"]
        >= parent.manifest.NONLINEAR_SIGNAL_THRESHOLD
        or not all(metrics["checks"].values())
    ):
        raise RuntimeError("expanded-chart authorization changed")
    return {"summary": summary, "metrics": metrics, "hashes": hashes}


def _contract() -> dict:
    inherited = parent.manifest.parent.manifest._contract()
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "objective": (
            "test_exact_C_phys_chart_geometry_at_the_first_doubled_"
            "component_bound_before_any_expanded_nonbase_rate_call"
        ),
        "candidate_family": {
            "anchor": ANCHOR,
            "energy_directions": DIRECTION_COUNT,
            "signs": list(SIGNS),
            "maximum_scaled_component_bound": COMPONENT_BOUND,
            "planned_candidates": PLANNED_CANDIDATES,
            "direction_source": "hash_locked_original_eight_energy_directions",
        },
        "exact_geometric_retraction": {
            **inherited["exact_geometric_retraction"],
            "maximum_Newton_iterations": 8,
            "maximum_radius_rescalings": 4,
            "rate_reaction_lift_used": False,
        },
        "binding_preflight_gates": {
            "completed_candidate_count_equal": PLANNED_CANDIDATES,
            "failed_candidate_count_equal": 0,
            "maximum_coordinate_residual_infinity": 1.0e-10,
            "maximum_normalized_Q3_defect": 1.0e-10,
            "maximum_final_scaled_component": COMPONENT_BOUND,
            "minimum_reconstruction_factor": 1.0 - 1.0e-12,
            "maximum_reconstruction_factor": 1.0 + 1.0e-12,
            "maximum_coordinate_Jacobian_condition_number": 1.0e4,
            "minimum_departure_direction_alignment_cosine": 0.99,
            "maximum_departure_transverse_fraction": 0.05,
            "maximum_coordinate_odd_symmetry_defect": 0.05,
            "maximum_H_over_R": 0.12,
            "minimum_scattering_optical_depth": 1.0,
            "nonbase_continuous_rate_evaluations_equal": 0,
            "new_full_generator_assemblies_equal": 0,
            "new_nonlinear_roots_equal": 0,
            "propagated_states_equal": 0,
        },
        "decision": {
            "pass": (
                "expanded_exact_departure_chart_amplitude_0p01_passed_"
                "sixteen_rate_screen_manifest_authorized"
            ),
            "fail": (
                "expanded_exact_departure_chart_amplitude_0p01_failed_"
                "nonlinear_amplitude_expansion_blocked"
            ),
            "pass_authorizes_only": (
                "definitions_only_expanded_amplitude_0p01_sixteen_rate_screen_manifest"
            ),
        },
        "claim_boundary": {
            "expanded_nonbase_rate_evaluations_executed": False,
            "nonlinear_closure_identified": False,
            "heldout_state_validated": False,
            "online_trajectory_authorized": False,
            "predictive_cycle_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
    }


def _update_catalog(summary: dict) -> None:
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(ARTIFACT_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha(path),
                    "scientific_status": "PROSPECTIVE",
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
        "path": str(ARTIFACT_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": True,
    }
    catalog.update(
        {
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": PARENT_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    parent_data = _validate_parent()
    if _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("expanded-chart manifest requires a clean tracked tree")
    if ARTIFACT_DIRECTORY.exists():
        raise RuntimeError("expanded-chart manifest is already frozen")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "anchor": ANCHOR,
        "maximum_scaled_component_bound": COMPONENT_BOUND,
        "planned_candidate_count": PLANNED_CANDIDATES,
        "planned_nonbase_rate_evaluations": 0,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": "WP10c9d6c7c3b5c4f25bc",
    }
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write(ARTIFACT_DIRECTORY / "contract.json", _contract())
    _write(ARTIFACT_DIRECTORY / "summary.json", summary)
    _write(
        ARTIFACT_DIRECTORY / "parent_lock.json",
        {
            "parent_commit": PARENT_COMMIT,
            "parent_parent": PARENT_PARENT,
            "parent_tree": PARENT_TREE,
            "guarded_rate_screen_hashes": parent_data["hashes"],
            "decisive_input_hashes": {
                "base_geometric_chart": _sha(BASE_CHART_PATH),
                "online_470_geometry": _sha(GEOMETRY_PATH),
            },
        },
    )
    _write(
        ARTIFACT_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "PROSPECTIVE",
            "definition_commit": _git("rev-parse", "HEAD"),
            "definition_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "authorized_next_runner": NEXT_RUNNER,
            "authorized_next_test": NEXT_TEST,
            "report": REPORT_RELATIVE,
            "source_hashes": {
                THIS_RUNNER: _sha(ROOT / THIS_RUNNER),
                THIS_TEST: _sha(ROOT / THIS_TEST),
            },
            "python": sys.version,
            "platform": platform.platform(),
            "thread_environment": parent.manifest.parent.coordinate_tools.THREAD_ENVIRONMENT,
        },
    )
    names = tuple(sorted(path.name for path in ARTIFACT_DIRECTORY.iterdir()))
    (ARTIFACT_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(ARTIFACT_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Expanded departure-chart manifest WP10c9d6c7c3b5c4f25bb",
                "",
                "## Classification",
                "",
                f"`{CLASSIFICATION}`",
                "",
                "The current 0.005 chart is physically valid but resolves only 0.43% median nonlinear departure signal. This package freezes the first doubled-amplitude geometry rung at 0.01 across the same eight signed energy directions.",
                "",
                "Exact C_phys retraction and every physical guard remain unchanged. No expanded nonbase rate call, closure fit, trajectory, or predictive cycle is authorized until this geometry rung passes.",
                "",
            )
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if not args.freeze:
        raise SystemExit("pass --freeze")
    print(json.dumps(_plain(_freeze()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
