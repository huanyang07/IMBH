#!/usr/bin/env python3
"""Freeze the exact departure-chart rung at component bound 0.02."""

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

import run_causal_inner_expanded_departure_rate_screen_wp10c9d6c7c3b5c4f25be as parent  # noqa: E402
import run_causal_inner_expanded_departure_chart_manifest_wp10c9d6c7c3b5c4f25bb as chart_contract  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25bf"
CLASSIFICATION = (
    "exact_departure_chart_amplitude_0p02_manifest_frozen_geometry_only"
)
PARENT_COMMIT = "eff39fb12c69794e11e5e38f6578b97957d1dcd7"
PARENT_PARENT = "bdc37ccd3293eae812333775f70b73998e2374b9"
PARENT_TREE = "4948f305807f5351c3a1781f2db1849f92ca2abd"

ARTIFACT = (
    "causal_inner_amplitude_0p02_departure_chart_manifest_"
    "wp10c9d6c7c3b5c4f25bf"
)
ARTIFACT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_amplitude_0p02_departure_chart_manifest_"
    "wp10c9d6c7c3b5c4f25bf.py"
)
THIS_TEST = (
    "tests/test_causal_inner_amplitude_0p02_departure_chart_manifest_"
    "wp10c9d6c7c3b5c4f25bf.py"
)
NEXT_RUNNER = (
    "scripts/run_causal_inner_amplitude_0p02_departure_chart_preflight_"
    "wp10c9d6c7c3b5c4f25bg.py"
)
NEXT_TEST = (
    "tests/test_causal_inner_amplitude_0p02_departure_chart_preflight_"
    "wp10c9d6c7c3b5c4f25bg.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_AMPLITUDE_0P02_DEPARTURE_"
    "CHART_MANIFEST_WP10C9D6C7C3B5C4F25BF_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

PRIOR_CHART_DIRECTORY = parent.manifest.parent.CANONICAL_DIRECTORY
PRIOR_CHART_PATH = PRIOR_CHART_DIRECTORY / "expanded_departure_chart.npz"
BASE_CHART_PATH = chart_contract.BASE_CHART_PATH
GEOMETRY_PATH = chart_contract.GEOMETRY_PATH

ANCHOR = "primary"
COMPONENT_BOUND = 2.0e-2
PRIOR_COMPONENT_BOUND = 1.0e-2
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
        raise RuntimeError("amplitude-0.01 rate-screen commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("amplitude-0.01 rate-screen lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("amplitude-0.01 rate-screen tree changed")
    rate_hashes = _checksums(parent.CANONICAL_DIRECTORY)
    rate_summary = _read(parent.CANONICAL_DIRECTORY / "summary.json")
    rate_metrics = _read(parent.CANONICAL_DIRECTORY / "metrics.json")
    chart_hashes = _checksums(PRIOR_CHART_DIRECTORY)
    chart_summary = _read(PRIOR_CHART_DIRECTORY / "summary.json")
    if (
        not rate_summary["passed"]
        or rate_summary["nonlinear_signal_resolved"]
        or rate_summary["component_bound"] != PRIOR_COMPONENT_BOUND
        or rate_summary["authorized_next"]
        != "definitions_only_exact_departure_chart_amplitude_0p02_manifest"
        or rate_summary["median_current_departure_nonlinear_relative_defect"]
        >= parent.manifest.NONLINEAR_SIGNAL_THRESHOLD
        or not all(rate_metrics["checks"].values())
    ):
        raise RuntimeError("amplitude-0.02 chart authorization changed")
    if (
        not chart_summary["passed"]
        or chart_summary["maximum_scaled_component_bound"]
        != PRIOR_COMPONENT_BOUND
        or chart_summary["completed_candidate_count"] != PLANNED_CANDIDATES
    ):
        raise RuntimeError("prior expanded chart changed")
    return {
        "rate_summary": rate_summary,
        "rate_metrics": rate_metrics,
        "rate_hashes": rate_hashes,
        "chart_summary": chart_summary,
        "chart_hashes": chart_hashes,
    }


def _contract() -> dict:
    inherited = chart_contract._contract()
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "objective": (
            "test_exact_C_phys_chart_geometry_at_component_bound_0p02_"
            "before_any_amplitude_0p02_nonbase_rate_call"
        ),
        "candidate_family": {
            "anchor": ANCHOR,
            "energy_directions": DIRECTION_COUNT,
            "signs": list(SIGNS),
            "maximum_scaled_component_bound": COMPONENT_BOUND,
            "prior_maximum_scaled_component_bound": PRIOR_COMPONENT_BOUND,
            "planned_candidates": PLANNED_CANDIDATES,
            "construction": "retract_each_direction_from_the_original_base_state",
            "direction_source": "hash_locked_original_eight_energy_directions",
            "prior_states_are_not_propagated_or_extrapolated": True,
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
                "exact_departure_chart_amplitude_0p02_passed_"
                "sixteen_rate_screen_manifest_authorized"
            ),
            "fail": (
                "exact_departure_chart_amplitude_0p02_failed_"
                "nonlinear_amplitude_expansion_blocked"
            ),
            "pass_authorizes_only": (
                "definitions_only_amplitude_0p02_sixteen_rate_screen_manifest"
            ),
        },
        "claim_boundary": {
            "amplitude_0p02_nonbase_rate_evaluations_executed": False,
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
        raise RuntimeError("amplitude-0.02 chart manifest requires a clean tracked tree")
    if ARTIFACT_DIRECTORY.exists():
        raise RuntimeError("amplitude-0.02 chart manifest is already frozen")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "anchor": ANCHOR,
        "maximum_scaled_component_bound": COMPONENT_BOUND,
        "planned_candidate_count": PLANNED_CANDIDATES,
        "planned_nonbase_continuous_rate_evaluations": 0,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": "WP10c9d6c7c3b5c4f25bg",
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
            "amplitude_0p01_rate_screen_hashes": parent_data["rate_hashes"],
            "amplitude_0p01_chart_hashes": parent_data["chart_hashes"],
            "decisive_input_hashes": {
                "prior_expanded_departure_chart": _sha(PRIOR_CHART_PATH),
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
            "thread_environment": parent.manifest.parent.chart_tools.coordinate_tools.THREAD_ENVIRONMENT,
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
                "# Amplitude-0.02 departure-chart manifest WP10c9d6c7c3b5c4f25bf",
                "",
                "## Classification",
                "",
                f"`{CLASSIFICATION}`",
                "",
                "The next execution constructs 16 exact fixed-Q geometric states at scaled-component bound 0.02 from the original base state and the same eight signed energy directions.",
                "",
                "No amplitude-0.02 rate, root, propagation, closure fit, or equilibrium branch is authorized. Only a complete geometric pass may authorize a separate 16-rate manifest.",
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
