#!/usr/bin/env python3
"""Freeze the moving exact weighted-arclength transport preflight."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_hot_mode_arclength_event_diagnosis_wp10c9d6c7c3b5c4f25f1 as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25f2"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25f3"
CLASSIFICATION = "moving_exact_arclength_transport_preflight_manifest_frozen"
ARTIFACT = (
    "causal_inner_arclength_transport_manifest_"
    "wp10c9d6c7c3b5c4f25f2"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_arclength_transport_manifest_"
    "wp10c9d6c7c3b5c4f25f2.py"
)
THIS_TEST = (
    "tests/test_causal_inner_arclength_transport_manifest_"
    "wp10c9d6c7c3b5c4f25f2.py"
)
PREFLIGHT_RUNNER = (
    "scripts/run_causal_inner_arclength_transport_preflight_"
    "wp10c9d6c7c3b5c4f25f3.py"
)
PREFLIGHT_TEST = (
    "tests/test_causal_inner_arclength_transport_preflight_"
    "wp10c9d6c7c3b5c4f25f3.py"
)
ARCLENGTH_SOURCE = "src/imri_qpe/layer3_minidisk_1d/arclength_phase.py"
ARCLENGTH_TEST = "tests/test_arclength_phase.py"
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ARCLENGTH_TRANSPORT_MANIFEST_"
    "WP10C9D6C7C3B5C4F25F2_2026-08-22.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE

NODE_COUNT = 5
INITIAL_ARCLENGTH_SPAN = 2.5e-2
MINIMUM_ARCLENGTH_SPAN = 6.25e-3
MAXIMUM_ARCLENGTH_SPAN = 5.0e-2
MAXIMUM_RETRIES = 2
MAXIMUM_TRANSPORT_ITERATIONS = 8
MAXIMUM_TARGET_EXACT_REFRESHES = 1
MAXIMUM_TOTAL_TARGET_EXACT_REFRESHES = 4
REFRESH_ITERATION_RESERVE = 2
STATE_REPLAY_MAXIMUM_SCALED_INFINITY_DEFECT = 1.0e-8


def _helper():
    return parent._helper()


def _decisive_inputs() -> dict[str, Path]:
    window_05 = parent._window_directories()[-1]
    return {
        "diagnosis_summary": parent.CANONICAL_DIRECTORY / "summary.json",
        "diagnosis_metrics": parent.CANONICAL_DIRECTORY / "arclength_event_metrics.json",
        "diagnosis_arrays": parent.CANONICAL_DIRECTORY / "arclength_event_arrays.npz",
        "window_05_summary": window_05 / "summary.json",
        "window_05_metrics": window_05 / "phase_window_metrics.json",
        "window_05_arrays": window_05 / "phase_window_arrays.npz",
    }


def _validate_parent(*, require_clean: bool) -> dict:
    helper = _helper()
    diagnosis_hashes = helper._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = helper._read(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = helper._read(parent.CANONICAL_DIRECTORY / "arclength_event_metrics.json")
    if (
        not summary["passed"]
        or not summary["weighted_coordinate_arclength_selected"]
        or summary["fixed_time_window_06_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or not metrics["passed"]
    ):
        raise RuntimeError("arclength diagnosis changed")
    window_05 = parent._window_directories()[-1]
    window_hashes = helper._validate_checksums(window_05)
    window_summary = helper._read(window_05 / "summary.json")
    if not window_summary["passed"] or int(window_summary["window_index"]) != 5:
        raise RuntimeError("Window 5 transport seed changed")
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("arclength transport manifest requires a clean tracked tree")
    return {
        "diagnosis_hashes": diagnosis_hashes,
        "window_05_hashes": window_hashes,
    }


def _contract(parent_lock: dict) -> dict:
    helper = _helper()
    sources = (
        THIS_RUNNER,
        THIS_TEST,
        PREFLIGHT_RUNNER,
        PREFLIGHT_TEST,
        ARCLENGTH_SOURCE,
        ARCLENGTH_TEST,
        parent.THIS_RUNNER,
        parent.source.THIS_RUNNER,
        parent.source.manifest.EXACT_CHART_SOURCE,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "mathematical_system": {
            "coordinate_phase_speed": "nu(y,t)=||f_Q(y,t)||_2",
            "coordinate_equation": "dy/ds=f_Q(y,t)/nu(y,t)",
            "time_equation": "dt/ds=1/nu(y,t)",
            "phase_is_strictly_monotone_while_nu_is_nonzero": True,
            "equilibrium_candidate_if_nu_collapses": True,
        },
        "collocation": {
            "node_family": "Legendre-Gauss-Lobatto",
            "node_count": NODE_COUNT,
            "projected_Picard_updates": 1,
            "initial_arclength_span": INITIAL_ARCLENGTH_SPAN,
            "minimum_arclength_span": MINIMUM_ARCLENGTH_SPAN,
            "maximum_arclength_span": MAXIMUM_ARCLENGTH_SPAN,
            "maximum_retries": MAXIMUM_RETRIES,
            "failed_segment_never_propagates": True,
        },
        "moving_exact_chart_transport": {
            "one_exact_augmented_coordinate_Jacobian_at_anchor": True,
            "anchor_gauge_is_canonical_and_frozen_within_segment": True,
            "anchor_matrix_initializes_every_target_retraction": True,
            "Broyden_updates_are_target_local": True,
            "maximum_transport_iterations": MAXIMUM_TRANSPORT_ITERATIONS,
            "maximum_exact_refreshes_per_target": MAXIMUM_TARGET_EXACT_REFRESHES,
            "maximum_total_target_exact_refreshes": MAXIMUM_TOTAL_TARGET_EXACT_REFRESHES,
            "iteration_reserve_before_refresh": REFRESH_ITERATION_RESERVE,
            "binding_coordinate_residual": parent.source.manifest.COORDINATE_TOLERANCE,
            "binding_gauge_residual": parent.source.manifest.GAUGE_TOLERANCE,
            "maximum_scaled_anchor_departure": parent.source.manifest.MAXIMUM_SCALED_ANCHOR_DEPARTURE,
            "maximum_augmented_condition_number": parent.source.manifest.MAXIMUM_AUGMENTED_CONDITION_NUMBER,
            "state_replay_maximum_scaled_infinity_defect": STATE_REPLAY_MAXIMUM_SCALED_INFINITY_DEFECT,
            "final_residual_and_physical_audits_are_exact": True,
        },
        "regime_candidates": {
            "legacy_exit_requires_persistence_and_endpoint_refinement": True,
            "equilibrium_requires_speed_collapse_then_stability_refinement": True,
            "recurrence_requires_close_return_tangent_alignment_and_Poincare_refinement": True,
            "no_candidate_is_a_certificate_without_separate_refinement": True,
        },
        "preflight": {
            "replay_all_fifteen_Window_05_exact_chart_targets": True,
            "new_exact_fixed_Q_rate_calls_equal": 0,
            "new_nonlinear_fixed_Q_roots_equal": 0,
            "new_BDF_microsteps_equal": 0,
            "canonical_exact_states_are_binding_replay_references": True,
        },
        "parent_lock": parent_lock,
        "decisive_input_hashes": {
            name: helper._sha(path) for name, path in _decisive_inputs().items()
        },
        "frozen_source_hashes": {
            relative: helper._sha(ROOT / relative) for relative in sources
        },
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }


def _update_catalog(summary: dict) -> None:
    helper = _helper()
    cold = parent.source._post().manifest.transition.manifest.cold.manifest
    with cold.CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({
                "case": ARTIFACT,
                "path": str(path.relative_to(ROOT)),
                "bytes": str(path.stat().st_size),
                "sha256": helper._sha(path),
                "scientific_status": "DEFINITIONS_ONLY",
            })
    with cold.CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case", "path", "bytes", "sha256", "scientific_status"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = helper._read(cold.CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": True,
    }
    catalog.update({
        "case_count": len({row["case"] for row in rows}),
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "all_payload_hashes_recorded": True,
        "latest_source_parent_commit": helper._git("rev-parse", "HEAD"),
        "latest_work_package": WORK_PACKAGE,
    })
    helper._write_json(cold.CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    helper = _helper()
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("arclength transport manifest already exists")
    parent_lock = _validate_parent(require_clean=True)
    contract = _contract(parent_lock)
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "arclength_transport_contract.json", contract)
    helper._write_json(CANONICAL_DIRECTORY / "input_lock.json", parent_lock)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "weighted_arclength_phase": True,
        "transport_preflight_authorized": True,
        "new_truth_execution_authorized": False,
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    helper._write_json(CANONICAL_DIRECTORY / "provenance.json", {
        "runner": THIS_RUNNER,
        "test": THIS_TEST,
        "implementation_commit": helper._git("rev-parse", "HEAD"),
        "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"),
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
    })
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{helper._sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join((
            "# Moving exact arclength transport manifest",
            "",
            f"Classification: `{CLASSIFICATION}`.",
            "",
            "The next preflight replays all 15 accepted Window-5 chart targets without evaluating a new fixed-Q rate. It binds the exact coordinate and gauge residuals, physical reconstruction, the canonical state replay, and a maximum of one anchor plus four target-refresh Jacobians.",
            "",
            "The selected phase equations are dy/ds=f_Q/||f_Q|| and dt/ds=1/||f_Q||. A five-node, 0.025-coordinate-arclength first execution segment remains blocked until transport replay passes.",
            "",
        )),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if not args.freeze:
        parser.error("use --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
