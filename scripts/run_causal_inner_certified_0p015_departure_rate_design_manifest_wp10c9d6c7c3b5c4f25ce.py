#!/usr/bin/env python3
"""Freeze a targeted exact-rate design at the certified 0.015 chart bound."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_guarded_departure_amplitude_expansion_preflight_wp10c9d6c7c3b5c4f25cd as parent  # noqa: E402
import run_causal_inner_departure28_rate_validation_wp10c9d6c7c3b5c4f25bx as old_rate  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25ce"
PARENT_COMMIT = "b8c461d852944c3911a546ff028041e7c886d509"
PARENT_PARENT = "d9d702c0eeee8b021733c4c636ea20c4bbad0cbb"
PARENT_TREE = "507d7d0d0fde99fb91ba7d386bd02397b98a1f57"
CLASSIFICATION = "certified_0p015_targeted_departure_rate_design_manifest_frozen"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25cf"

CERTIFIED_COMPONENT_BOUND = 1.5e-2
TARGET_DIRECTION_INDICES = (4, 6, 8, 9)
TARGET_DIRECTION_LABELS = (
    "energy_4",
    "energy_6",
    "screen_escape",
    "accepted_forward_rate",
)
SIGNS = (-1, 1)
SELECTED_CANDIDATE_INDICES = tuple(
    2 * direction + sign_index
    for direction in TARGET_DIRECTION_INDICES
    for sign_index in range(len(SIGNS))
)
PLANNED_RATE_EVALUATIONS = len(SELECTED_CANDIDATE_INDICES)
FORWARD_POSITIVE_LOCAL_INDEX = 7

ARTIFACT = (
    "causal_inner_certified_0p015_departure_rate_design_manifest_"
    "wp10c9d6c7c3b5c4f25ce"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_certified_0p015_departure_rate_design_manifest_"
    "wp10c9d6c7c3b5c4f25ce.py"
)
THIS_TEST = (
    "tests/test_causal_inner_certified_0p015_departure_rate_design_manifest_"
    "wp10c9d6c7c3b5c4f25ce.py"
)
NEXT_RUNNER = (
    "scripts/run_causal_inner_certified_0p015_departure_rate_screen_"
    "wp10c9d6c7c3b5c4f25cf.py"
)
NEXT_TEST = (
    "tests/test_causal_inner_certified_0p015_departure_rate_screen_"
    "wp10c9d6c7c3b5c4f25cf.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_CERTIFIED_0P015_DEPARTURE_"
    "RATE_DESIGN_MANIFEST_WP10C9D6C7C3B5C4F25CE_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

PARENT_ARRAYS = parent.CANONICAL_DIRECTORY / "expanded_chart_states.npz"
OLD_COEFFICIENTS = old_rate.CANONICAL_DIRECTORY / "frozen_coefficients.npz"
ONLINE_GEOMETRY = old_rate.ONLINE_GEOMETRY_PATH
GENERATOR = old_rate.GENERATOR_PATH

_plain = parent._plain
_read = parent._read
_write_json = parent._write_json
_sha = parent._sha
_checksums = parent._checksums


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _validate_parent(*, require_clean: bool) -> dict:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("guarded-amplitude result commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("guarded-amplitude result lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("guarded-amplitude result tree changed")
    hashes = _checksums(parent.CANONICAL_DIRECTORY)
    summary = _read(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = _read(parent.CANONICAL_DIRECTORY / "preflight_metrics.json")
    provenance = _read(parent.CANONICAL_DIRECTORY / "provenance.json")
    if (
        not summary["passed"]
        or summary["classification"] != parent.PARTIAL_CLASSIFICATION
        or summary["largest_passing_component_bound"] != CERTIFIED_COMPONENT_BOUND
        or summary["authorized_next"]
        != "definitions_only_rate_design_within_largest_passing_departure_bound"
        or summary["new_truth_calls"] != 0
        or not summary["stable_memory_remains_dynamic"]
        or summary["old_polynomial_extrapolation_used"]
        or metrics["passing_rung_count"] != 1
        or not metrics["rungs"][0]["passed"]
        or metrics["rungs"][1]["passed"]
    ):
        raise RuntimeError("guarded-amplitude authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"guarded-amplitude source changed: {relative}")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("targeted rate manifest requires a clean tracked tree")
    return {"summary": summary, "metrics": metrics, "hashes": hashes}


def _selected_design() -> tuple[dict[str, np.ndarray], dict]:
    source = _load_npz(PARENT_ARRAYS)
    indices = np.asarray(SELECTED_CANDIDATE_INDICES, dtype=int)
    amplitudes = source["candidate_component_bounds"][indices]
    directions = source["candidate_direction_indices"][indices]
    signs = source["candidate_signs"][indices]
    expected_directions = np.repeat(np.asarray(TARGET_DIRECTION_INDICES), 2)
    expected_signs = np.tile(np.asarray(SIGNS), len(TARGET_DIRECTION_INDICES))
    if (
        source["candidate_primitive_states"].shape != (40, 112, 5)
        or source["candidate_scaled_deltas"].shape != (40, 560)
        or source["candidate_departure_coordinates"].shape != (40, 28)
        or not np.array_equal(amplitudes, np.full(PLANNED_RATE_EVALUATIONS, CERTIFIED_COMPONENT_BOUND))
        or not np.array_equal(directions, expected_directions)
        or not np.array_equal(signs, expected_signs)
    ):
        raise RuntimeError("certified 0.015 rate selection changed")
    coordinates = source["candidate_departure_coordinates"][indices]
    radial_norms = np.linalg.norm(coordinates, axis=1)
    if np.any(radial_norms <= np.finfo(float).tiny):
        raise RuntimeError("selected departure coordinate vanished")
    return (
        {
            "parent_candidate_indices": indices,
            "direction_indices": directions,
            "signs": signs,
            "departure_coordinates": coordinates,
        },
        {
            "direction_labels": list(TARGET_DIRECTION_LABELS),
            "candidate_labels": [
                f"{label}_{'negative' if sign < 0 else 'positive'}"
                for label in TARGET_DIRECTION_LABELS
                for sign in SIGNS
            ],
            "minimum_departure_coordinate_norm": float(np.min(radial_norms)),
            "maximum_departure_coordinate_norm": float(np.max(radial_norms)),
        },
    )


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "objective": (
            "measure_exact_boundary_departure_rates_and_choose_between_"
            "inward_turn_and_recentered_transient_atlas_architectures"
        ),
        "candidate_design": {
            "component_bound": CERTIFIED_COMPONENT_BOUND,
            "parent_candidate_indices": list(SELECTED_CANDIDATE_INDICES),
            "direction_indices": list(TARGET_DIRECTION_INDICES),
            "direction_labels": list(TARGET_DIRECTION_LABELS),
            "signs": list(SIGNS),
            "planned_exact_rate_evaluations": PLANNED_RATE_EVALUATIONS,
            "forward_positive_local_index": FORWARD_POSITIVE_LOCAL_INDEX,
        },
        "binding_exact_rate_gates": {
            "completed_nonbase_rate_evaluations_equal": PLANNED_RATE_EVALUATIONS,
            "failed_rate_evaluations_equal": 0,
            "minimum_reconstruction_factor": 1.0 - 1.0e-12,
            "maximum_reconstruction_factor": 1.0 + 1.0e-12,
            "maximum_raw_Schur_condition_number": 1.0e6,
            "maximum_reaction_identity_defect": 1.0e-9,
            "maximum_rate_tangency_relative_defect": 1.0e-8,
            "maximum_coordinate_Jacobian_condition_number": 5.0e3,
            "maximum_H_over_R": 0.12,
            "minimum_scattering_optical_depth": 1.0,
            "maximum_incoming_excision_characteristics_equal": 0,
            "new_complete_generator_assemblies_equal": 0,
            "new_nonlinear_roots_equal": 0,
            "propagated_states_equal": 0,
        },
        "frozen_old_field_diagnostic": {
            "maximum_full_departure_rate_relative_error": 0.15,
            "model_refit_during_screen": False,
            "old_field_pass_is_required_for_truth_classification": False,
        },
        "forward_boundary_decision": {
            "radial_direction_cosine_threshold": 0.02,
            "outward": {
                "classification_suffix": "forward_chart_exit",
                "requires": "authentic_trajectory_recentered_chart_atlas",
            },
            "inward": {
                "classification_suffix": "forward_inward_turn",
                "requires": "bounded_transient_saturation_validation",
            },
            "nearly_tangential": {
                "classification_suffix": "forward_turn_unresolved",
                "requires": "local_rate_extension_before_trajectory",
            },
        },
        "scientific_boundaries": {
            "new_generator_assembly": False,
            "new_nonlinear_fixed_Q_root": False,
            "propagated_state": False,
            "stationary_stable_memory_elimination": False,
            "amplitude_above_certified_bound": False,
            "physical_microburst_authorized": False,
            "predictive_cycle_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
        "target_mathematical_architecture": {
            "active_state": "q162_physical_coordinates",
            "stable_memory": "z280_dynamic_exponential_or_L_stable_update",
            "departure": "a28_nonlinear_transient_atlas_then_invariant_measure_or_phase_closure",
            "slow_closure": "multi_anchor_conservative_averaged_q_flux",
            "online_truth_calls_per_macrostep": 0,
            "maximum_macrosteps_per_cycle": 100_000,
        },
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
                    "scientific_status": "DEFINITIONS_ONLY",
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
            "latest_source_parent_commit": PARENT_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def _run() -> dict:
    frozen = _validate_parent(require_clean=True)
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("targeted rate design already canonicalized")
    arrays, design = _selected_design()
    contract = _contract()
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(CANONICAL_DIRECTORY / "rate_design.npz", **arrays)
    _write_json(CANONICAL_DIRECTORY / "rate_design.json", design)
    _write_json(CANONICAL_DIRECTORY / "contract.json", contract)
    _write_json(
        CANONICAL_DIRECTORY / "parent_lock.json",
        {
            "parent_commit": PARENT_COMMIT,
            "parent_parent": PARENT_PARENT,
            "parent_tree": PARENT_TREE,
            "parent_hashes": frozen["hashes"],
            "expanded_chart_states_sha256": _sha(PARENT_ARRAYS),
            "old_coefficients_sha256": _sha(OLD_COEFFICIENTS),
            "online_geometry_sha256": _sha(ONLINE_GEOMETRY),
            "generator_sha256": _sha(GENERATOR),
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "certified_component_bound": CERTIFIED_COMPONENT_BOUND,
        "planned_exact_rate_evaluations": PLANNED_RATE_EVALUATIONS,
        "new_truth_calls": 0,
        "stable_memory_remains_dynamic": True,
        "model_refit_authorized": False,
        "trajectory_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "DEFINITIONS_ONLY",
            "definition_commit": _git("rev-parse", "HEAD"),
            "definition_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "source_hashes": {
                THIS_RUNNER: _sha(ROOT / THIS_RUNNER),
                THIS_TEST: _sha(ROOT / THIS_TEST),
                parent.THIS_RUNNER: _sha(ROOT / parent.THIS_RUNNER),
                parent.THIS_TEST: _sha(ROOT / parent.THIS_TEST),
                old_rate.THIS_RUNNER: _sha(ROOT / old_rate.THIS_RUNNER),
                old_rate.THIS_TEST: _sha(ROOT / old_rate.THIS_TEST),
            },
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
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
                "# Certified-0.015 departure-rate design manifest WP10c9d6c7c3b5c4f25ce",
                "",
                "## Classification",
                "",
                f"`{CLASSIFICATION}`",
                "",
                "Eight exact continuous-rate evaluations are frozen at the largest certified departure-chart bound: signed energy-4, energy-6, stationary-screen escape, and accepted-forward-flow states.",
                "",
                "The exact forward-flow radial sign chooses between an inward-turn saturation test and an authentic trajectory-recentered chart atlas. The frozen old departure field is assessed at these states but is neither refit nor treated as truth.",
                "",
                "The 280D stable memory remains dynamic. No generator, nonlinear root, propagated state, physical microburst, cycle evolution, or reduced slow evolution is authorized.",
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
    print(json.dumps(_plain(_run()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
