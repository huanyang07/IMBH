#!/usr/bin/env python3
"""Freeze the amplitude-0.01 primary departure-rate screen."""

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

import run_causal_inner_expanded_departure_chart_preflight_wp10c9d6c7c3b5c4f25bc as parent  # noqa: E402
import run_causal_inner_guarded_departure_rate_screen_wp10c9d6c7c3b5c4f25ba as prior_screen  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25bd"
CLASSIFICATION = (
    "expanded_primary_departure_rate_screen_amplitude_0p01_manifest_frozen_"
    "no_closure_fit_or_trajectory_authorized"
)
PARENT_COMMIT = "157270a635ced4f8a25a93ece6cd7cd5013c1a83"
PARENT_PARENT = "1b9105183bfc26b0fe0967106c97b8047c952461"
PARENT_TREE = "f1a4bd7475742ed1be96d1380a4f101bcd815551"

ARTIFACT = (
    "causal_inner_expanded_departure_rate_screen_manifest_"
    "wp10c9d6c7c3b5c4f25bd"
)
ARTIFACT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_expanded_departure_rate_screen_manifest_"
    "wp10c9d6c7c3b5c4f25bd.py"
)
THIS_TEST = (
    "tests/test_causal_inner_expanded_departure_rate_screen_manifest_"
    "wp10c9d6c7c3b5c4f25bd.py"
)
NEXT_RUNNER = (
    "scripts/run_causal_inner_expanded_departure_rate_screen_"
    "wp10c9d6c7c3b5c4f25be.py"
)
NEXT_TEST = (
    "tests/test_causal_inner_expanded_departure_rate_screen_"
    "wp10c9d6c7c3b5c4f25be.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_EXPANDED_DEPARTURE_RATE_"
    "SCREEN_MANIFEST_WP10C9D6C7C3B5C4F25BD_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

CHART_PATH = parent.CANONICAL_DIRECTORY / "expanded_departure_chart.npz"
GEOMETRY_PATH = parent.manifest.GEOMETRY_PATH
GENERATOR_PATH = prior_screen.manifest.GENERATOR_PATH
PRIOR_SCREEN_PATH = prior_screen.CANONICAL_DIRECTORY / "departure_rate_screen.npz"

ANCHOR = "primary"
DEPARTURE_DIMENSION = 28
DIRECTION_COUNT = 8
CANDIDATE_COUNT = 16
COMPONENT_BOUND = 1.0e-2
PRIOR_COMPONENT_BOUND = 5.0e-3
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
        raise RuntimeError("expanded-chart certificate commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("expanded-chart certificate lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("expanded-chart certificate tree changed")
    chart_hashes = _checksums(parent.CANONICAL_DIRECTORY)
    chart_summary = _read(parent.CANONICAL_DIRECTORY / "summary.json")
    chart_metrics = _read(parent.CANONICAL_DIRECTORY / "metrics.json")
    prior_hashes = _checksums(prior_screen.CANONICAL_DIRECTORY)
    prior_summary = _read(prior_screen.CANONICAL_DIRECTORY / "summary.json")
    prior_metrics = _read(prior_screen.CANONICAL_DIRECTORY / "metrics.json")
    if (
        not chart_summary["passed"]
        or chart_summary["completed_candidate_count"] != CANDIDATE_COUNT
        or chart_summary["failed_candidate_count"] != 0
        or chart_summary["maximum_scaled_component_bound"] != COMPONENT_BOUND
        or chart_summary["nonbase_continuous_rate_evaluations"] != 0
        or chart_summary["authorized_next"]
        != "definitions_only_expanded_amplitude_0p01_sixteen_rate_screen_manifest"
        or not all(chart_metrics["checks"].values())
    ):
        raise RuntimeError("expanded-chart rate-screen authorization changed")
    if (
        not prior_summary["passed"]
        or prior_summary["nonlinear_signal_resolved"]
        or prior_summary["median_largest_departure_nonlinear_relative_defect"]
        >= NONLINEAR_SIGNAL_THRESHOLD
        or not all(prior_metrics["checks"].values())
    ):
        raise RuntimeError("prior amplitude-0.005 rate screen changed")
    return {
        "chart_summary": chart_summary,
        "chart_metrics": chart_metrics,
        "chart_hashes": chart_hashes,
        "prior_summary": prior_summary,
        "prior_metrics": prior_metrics,
        "prior_hashes": prior_hashes,
    }


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "objective": (
            "evaluate_the_exact_continuous_fixed_Q_vector_field_on_the_"
            "sixteen_certified_amplitude_0p01_axial_states_and_decide_"
            "whether_the_28D_departure_nonlinearity_is_resolved"
        ),
        "inputs": {
            "anchor": ANCHOR,
            "candidate_states": CANDIDATE_COUNT,
            "candidate_component_bound": COMPONENT_BOUND,
            "candidate_source": "hash_locked_expanded_exact_geometric_chart",
            "base_rate_source": "hash_locked_exact_primary_fixed_Q_rate",
            "linear_reference": "hash_locked_complete_primary_fixed_Q_generator",
            "prior_rate_screen_component_bound": PRIOR_COMPONENT_BOUND,
            "departure_basis_dimension": DEPARTURE_DIMENSION,
            "energy_directions": DIRECTION_COUNT,
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
        "nonlinear_signal_audit": {
            "current_metric": (
                "median_over_eight_signed_pairs_of_the_central_28D_"
                "rate_increment_defect_relative_to_the_complete_linear_generator"
            ),
            "prior_metric": (
                "same_hash_locked_metric_at_component_bound_0p005"
            ),
            "amplitude_amplification_recorded": True,
            "central_radial_growth_recorded": True,
            "secant_cubic_growth_coefficient_recorded": True,
            "radial_saturation_is_diagnostic_not_binding": True,
            "equilibrium_branch_selection_is_not_an_outcome": True,
        },
        "nonlinear_signal_classifier": {
            "component_bound": COMPONENT_BOUND,
            "metric": "median_central_departure_nonlinear_relative_defect",
            "resolved_if_at_least": NONLINEAR_SIGNAL_THRESHOLD,
        },
        "binding_evaluator_gates": {
            "completed_nonbase_rate_evaluations_equal": CANDIDATE_COUNT,
            "failed_rate_evaluations_equal": 0,
            "minimum_reconstruction_factor": 1.0 - 1.0e-12,
            "maximum_reconstruction_factor": 1.0 + 1.0e-12,
            "maximum_raw_Schur_condition_number": 1.0e8,
            "maximum_reaction_identity_defect": 1.0e-10,
            "maximum_rate_tangency_relative_defect": 1.0e-10,
            "maximum_coordinate_Jacobian_condition_number": 1.0e4,
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
                    "expanded_primary_departure_rate_screen_amplitude_0p01_"
                    "passed_nonlinear_signal_resolved_mixed_direction_"
                    "database_manifest_authorized"
                ),
                "authorizes_only": (
                    "definitions_only_mixed_direction_adaptive_28D_database_manifest"
                ),
            },
            "nonlinear_signal_not_resolved": {
                "classification": (
                    "expanded_primary_departure_rate_screen_amplitude_0p01_"
                    "passed_nonlinear_signal_not_resolved_amplitude_0p02_"
                    "chart_manifest_authorized"
                ),
                "authorizes_only": (
                    "definitions_only_exact_departure_chart_amplitude_0p02_manifest"
                ),
            },
            "evaluator_failed": {
                "classification": (
                    "expanded_primary_departure_rate_screen_amplitude_0p01_"
                    "failed_nonlinear_architecture_identification_blocked"
                ),
                "authorizes_only": None,
            },
        },
        "claim_boundary": {
            "sixteen_axial_samples_are_a_full_28D_closure_database": False,
            "prior_plus_current_axial_samples_are_a_full_28D_database": False,
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
        raise RuntimeError("expanded rate-screen manifest requires a clean tracked tree")
    if ARTIFACT_DIRECTORY.exists():
        raise RuntimeError("expanded rate-screen manifest is already frozen")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "anchor": ANCHOR,
        "component_bound": COMPONENT_BOUND,
        "planned_nonbase_continuous_rate_evaluations": CANDIDATE_COUNT,
        "planned_new_generator_assemblies": 0,
        "full_closure_database_claimed": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": "WP10c9d6c7c3b5c4f25be",
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
            "expanded_chart_package_hashes": parent_data["chart_hashes"],
            "prior_rate_screen_package_hashes": parent_data["prior_hashes"],
            "decisive_input_hashes": {
                "expanded_departure_chart": _sha(CHART_PATH),
                "online_470_geometry": _sha(GEOMETRY_PATH),
                "complete_primary_generator": _sha(GENERATOR_PATH),
                "prior_departure_rate_screen": _sha(PRIOR_SCREEN_PATH),
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
            "thread_environment": parent.chart_tools.coordinate_tools.THREAD_ENVIRONMENT,
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
                "# Expanded departure-rate screen manifest WP10c9d6c7c3b5c4f25bd",
                "",
                "## Classification",
                "",
                f"`{CLASSIFICATION}`",
                "",
                "The next execution evaluates exactly 16 continuous fixed-Q rates on the certified amplitude-0.01 primary-anchor chart. It saves physical reaction actions and 560/470/28-dimensional rates, but performs no root or trajectory propagation.",
                "",
                "The central signed-pair nonlinear fraction is compared with the complete saved generator and the amplitude-0.005 screen. A 10% median signal authorizes mixed-direction sampling; an unresolved signal authorizes only a prospective amplitude-0.02 geometry rung.",
                "",
                "Radial saturation is diagnostic, not required. No closure fit, equilibrium branch, held-out claim, online trajectory, or predictive cycle is authorized.",
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
