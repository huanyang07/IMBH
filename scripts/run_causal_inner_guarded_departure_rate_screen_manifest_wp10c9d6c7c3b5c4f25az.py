#!/usr/bin/env python3
"""Freeze the guarded primary-anchor nonlinear departure-rate screen."""

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

import run_causal_inner_exact_geometric_departure_chart_preflight_wp10c9d6c7c3b5c4f25ay as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25az"
CLASSIFICATION = (
    "guarded_primary_departure_rate_screen_manifest_frozen_"
    "no_closure_fit_or_trajectory_authorized"
)
PARENT_COMMIT = "48ae358264b7e2cebc1ab66f23b9ed896fb8b0a9"
PARENT_PARENT = "7b35714d7035f8363b519faa0d1acfac78ddd48a"
PARENT_TREE = "84d23d3bf65e62137801cde88faf30f726634451"

ARTIFACT = (
    "causal_inner_guarded_departure_rate_screen_manifest_"
    "wp10c9d6c7c3b5c4f25az"
)
ARTIFACT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_guarded_departure_rate_screen_manifest_"
    "wp10c9d6c7c3b5c4f25az.py"
)
THIS_TEST = (
    "tests/test_causal_inner_guarded_departure_rate_screen_manifest_"
    "wp10c9d6c7c3b5c4f25az.py"
)
NEXT_RUNNER = (
    "scripts/run_causal_inner_guarded_departure_rate_screen_"
    "wp10c9d6c7c3b5c4f25ba.py"
)
NEXT_TEST = (
    "tests/test_causal_inner_guarded_departure_rate_screen_"
    "wp10c9d6c7c3b5c4f25ba.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_GUARDED_DEPARTURE_RATE_"
    "SCREEN_MANIFEST_WP10C9D6C7C3B5C4F25AZ_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

CHART_PATH = parent.CANONICAL_DIRECTORY / "geometric_departure_chart.npz"
GEOMETRY_PATH = parent.manifest.GEOMETRY_PATH
GENERATOR_PATH = parent.coordinate_tools.manifest.GENERATOR_PATH

ANCHOR = "primary"
DEPARTURE_DIMENSION = 28
CANDIDATE_COUNT = 48
SMALLEST_COMPONENT_BOUND = 2.5e-4
LARGEST_COMPONENT_BOUND = 5.0e-3
NONLINEAR_SIGNAL_THRESHOLD = 0.10


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
        raise RuntimeError("geometric chart certificate commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("geometric chart certificate lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("geometric chart certificate tree changed")
    hashes = _checksums(parent.CANONICAL_DIRECTORY)
    summary = _read(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = _read(parent.CANONICAL_DIRECTORY / "metrics.json")
    if (
        not summary["passed"]
        or summary["completed_candidate_count"] != CANDIDATE_COUNT
        or summary["failed_candidate_count"] != 0
        or summary["nonbase_continuous_rate_evaluations"] != 0
        or summary["authorized_next"]
        != "definitions_only_guarded_nonlinear_28D_rate_database_manifest"
        or not all(metrics["checks"].values())
    ):
        raise RuntimeError("geometric chart authorization changed")
    return {"summary": summary, "metrics": metrics, "hashes": hashes}


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "objective": (
            "evaluate_the_exact_continuous_fixed_Q_vector_field_on_the_"
            "48_certified_primary_anchor_chart_states_and_decide_whether_"
            "nonlinear_signal_is_resolved_inside_the_current_trust_region"
        ),
        "inputs": {
            "anchor": ANCHOR,
            "candidate_states": CANDIDATE_COUNT,
            "candidate_source": "hash_locked_exact_geometric_chart_certificate",
            "base_rate_source": "hash_locked_exact_primary_fixed_Q_rate",
            "linear_reference": "hash_locked_complete_primary_fixed_Q_generator",
            "departure_basis_dimension": DEPARTURE_DIMENSION,
        },
        "truth_evaluation": {
            "continuous_rate": (
                "descriptor_solve_of_the_complete_monolithic_stationary_"
                "residual_plus_physical_fixed_Q_reaction_action"
            ),
            "save_free_rate": True,
            "save_physical_reaction_action": True,
            "save_multiplier_coordinates": True,
            "save_total_560_rate": True,
            "save_470_coordinate_rate": True,
            "save_28D_departure_rate": True,
            "nonbase_continuous_rate_evaluations": CANDIDATE_COUNT,
            "new_complete_generator_assemblies": 0,
            "new_nonlinear_roots": 0,
            "propagated_states": 0,
        },
        "linear_limit_audit": {
            "smallest_component_bound": SMALLEST_COMPONENT_BOUND,
            "reference_increment": "complete_generator_times_actual_scaled_delta",
            "state_rate_metric": (
                "norm_of_nonlinear_rate_increment_minus_linear_reference_"
                "over_norm_of_linear_reference"
            ),
            "departure_rate_metric": (
                "same_relative_defect_after_projection_to_the_28D_departure_basis"
            ),
            "pairwise_central_radial_growth_recorded": True,
        },
        "nonlinear_signal_classifier": {
            "largest_component_bound": LARGEST_COMPONENT_BOUND,
            "metric": (
                "median_over_directions_of_departure_rate_nonlinear_"
                "relative_defect_at_the_largest_amplitude"
            ),
            "resolved_if_at_least": NONLINEAR_SIGNAL_THRESHOLD,
            "radial_saturation_is_diagnostic_not_binding": True,
            "equilibrium_branch_selection_is_not_an_outcome": True,
        },
        "binding_evaluator_gates": {
            "completed_nonbase_rate_evaluations_equal": CANDIDATE_COUNT,
            "failed_rate_evaluations_equal": 0,
            "maximum_smallest_state_rate_linear_relative_defect": 0.25,
            "maximum_smallest_departure_rate_linear_relative_defect": 0.25,
            "minimum_reconstruction_factor": 1.0 - 1.0e-12,
            "maximum_reconstruction_factor": 1.0 + 1.0e-12,
            "maximum_raw_Schur_condition_number": 1.0e8,
            "maximum_reaction_identity_defect": 1.0e-10,
            "maximum_rate_tangency_relative_defect": 1.0e-10,
            "maximum_H_over_R": 0.12,
            "minimum_scattering_optical_depth": 1.0,
            "maximum_incoming_excision_characteristics_equal": 0,
            "new_complete_generator_assemblies_equal": 0,
            "new_nonlinear_roots_equal": 0,
            "propagated_states_equal": 0,
        },
        "decision": {
            "nonlinear_signal_resolved": {
                "classification": (
                    "guarded_primary_departure_rate_screen_passed_"
                    "nonlinear_signal_resolved_mixed_direction_database_"
                    "manifest_authorized"
                ),
                "authorizes_only": (
                    "definitions_only_mixed_direction_adaptive_28D_database_manifest"
                ),
            },
            "nonlinear_signal_not_resolved": {
                "classification": (
                    "guarded_primary_departure_rate_screen_passed_"
                    "nonlinear_signal_not_resolved_expanded_chart_manifest_authorized"
                ),
                "authorizes_only": (
                    "definitions_only_expanded_safe_departure_chart_manifest"
                ),
            },
            "evaluator_failed": {
                "classification": (
                    "guarded_primary_departure_rate_screen_failed_"
                    "nonlinear_closure_identification_blocked"
                ),
                "authorizes_only": None,
            },
        },
        "claim_boundary": {
            "48_axial_samples_are_a_full_28D_closure_database": False,
            "nonlinear_coefficients_identified": False,
            "heldout_state_validated": False,
            "online_integrator_implemented": False,
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
        raise RuntimeError("departure-rate screen manifest requires a clean tracked tree")
    if ARTIFACT_DIRECTORY.exists():
        raise RuntimeError("departure-rate screen manifest is already frozen")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "anchor": ANCHOR,
        "planned_nonbase_continuous_rate_evaluations": CANDIDATE_COUNT,
        "planned_new_generator_assemblies": 0,
        "full_closure_database_claimed": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": "WP10c9d6c7c3b5c4f25ba",
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
            "geometric_chart_package_hashes": parent_data["hashes"],
            "decisive_input_hashes": {
                "geometric_departure_chart": _sha(CHART_PATH),
                "online_470_geometry": _sha(GEOMETRY_PATH),
                "complete_primary_generator": _sha(GENERATOR_PATH),
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
            "thread_environment": parent.coordinate_tools.THREAD_ENVIRONMENT,
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
                "# Guarded departure-rate screen manifest WP10c9d6c7c3b5c4f25az",
                "",
                "## Classification",
                "",
                f"`{CLASSIFICATION}`",
                "",
                "The next execution evaluates the exact continuous fixed-Q rate on the 48 already-certified primary-anchor chart states. It stores the free rate, physical reaction action, total rate, and 470/28-dimensional coordinate rates.",
                "",
                "The smallest-amplitude pairs must reproduce the complete saved generator. The largest-amplitude pairs classify whether nonlinear signal is resolved. Radial saturation is diagnostic rather than an architecture-selection requirement.",
                "",
                "No closure fit, held-out claim, online trajectory, or predictive cycle is authorized.",
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
