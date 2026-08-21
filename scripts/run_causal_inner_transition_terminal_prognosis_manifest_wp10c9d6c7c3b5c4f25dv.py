#!/usr/bin/env python3
"""Freeze the hot-transition terminal prognosis and architecture decision."""

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

import run_causal_inner_conservative_transition_tube_wp10c9d6c7c3b5c4f25du as tube  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25dv"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25dw"
PARENT_COMMIT = "6401da5c896a473e7bbe031476c945a67bcb39e7"
PARENT_TREE = "d6097d2951ea9633e713acfcde736ab24c4535e7"
CLASSIFICATION = (
    "transition_terminal_prognosis_and_branch_continuation_"
    "architecture_decision_frozen"
)

TAIL_INTERVALS = 6
MAXIMUM_ADDITIONAL_HALF_STEP_ROOTS = 24
MAXIMUM_EXTENSION_WALL_HOURS = 10.0
HIDDEN_EXIT_FRACTION = 0.25
ONLINE_CYCLE_SECONDS = 578_880.0
MAXIMUM_ONLINE_MACROSTEPS = 100_000
MINIMUM_AVERAGE_MACROSTEP_SECONDS = ONLINE_CYCLE_SECONDS / MAXIMUM_ONLINE_MACROSTEPS

ARTIFACT = "causal_inner_transition_terminal_prognosis_manifest_wp10c9d6c7c3b5c4f25dv"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = "scripts/run_causal_inner_transition_terminal_prognosis_manifest_wp10c9d6c7c3b5c4f25dv.py"
THIS_TEST = "tests/test_causal_inner_transition_terminal_prognosis_manifest_wp10c9d6c7c3b5c4f25dv.py"
EXECUTION_RUNNER = "scripts/run_causal_inner_transition_terminal_prognosis_wp10c9d6c7c3b5c4f25dw.py"
EXECUTION_TEST = "tests/test_causal_inner_transition_terminal_prognosis_wp10c9d6c7c3b5c4f25dw.py"
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_TRANSITION_TERMINAL_PROGNOSIS_"
    "MANIFEST_WP10C9D6C7C3B5C4F25DV_2026-08-21.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

COLD_SCREEN_DIRECTORY = ROOT / (
    "results/canonical/causal_inner_branch_candidate_saved_array_screen_"
    "wp10c9d6c7c3b5c4f25dm"
)


def _validate_parent(*, require_clean: bool) -> dict:
    if tube.manifest.geometry._git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("terminal prognosis parent commit changed")
    if tube.manifest.geometry._git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("terminal prognosis parent tree changed")
    tube_hashes = tube.manifest.geometry._validate_checksums(tube.CANONICAL_DIRECTORY)
    cold_hashes = tube.manifest.geometry._validate_checksums(COLD_SCREEN_DIRECTORY)
    tube_summary = tube.manifest.geometry._read(tube.CANONICAL_DIRECTORY / "summary.json")
    cold_metrics = tube.manifest.geometry._read(
        COLD_SCREEN_DIRECTORY / "branch_candidate_screen_metrics.json"
    )
    if (
        not tube_summary["passed"]
        or tube_summary["authorized_next"] != WORK_PACKAGE
        or tube_summary["hot_exit_observed"]
        or cold_metrics["selected_cold_candidate"] != "full_model_12ms"
        or cold_metrics["selected_hot_candidate"] is not None
    ):
        raise RuntimeError("terminal prognosis inputs changed")
    if require_clean and tube.manifest.geometry._git(
        "status", "--short", "--untracked-files=no"
    ):
        raise RuntimeError("terminal prognosis manifest requires a clean tracked tree")
    return {"tube_hashes": tube_hashes, "cold_screen_hashes": cold_hashes}


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "prognosis": {
            "tail_intervals": TAIL_INTERVALS,
            "hidden_exit_fraction": HIDDEN_EXIT_FRACTION,
            "maximum_additional_half_step_roots": MAXIMUM_ADDITIONAL_HALF_STEP_ROOTS,
            "maximum_extension_wall_hours": MAXIMUM_EXTENSION_WALL_HOURS,
            "extension_requires_every_tail_fraction_to_decrease": True,
            "extension_requires_linear_crossing_within_budget": True,
            "extension_requires_projected_wall_time_within_budget": True,
        },
        "selected_mathematical_architecture": {
            "slow_state": "q_in_R82_conservative_finite_volume_coordinates",
            "branch_equation": "G_b(q,h)=Q*F(Lq+Zh)=0",
            "branch_drift": "dq_dt=R*F(Lq+Z*h_b(q))",
            "regular_branch_tangent": "G_h*dh_dq=-G_q",
            "fold_continuation": "pseudo_arclength_bordered_G_equals_zero",
            "event_indicator": "smallest_singular_value_of_G_h_with_orientation",
            "transition_state": "(q_entry,s_scalar)",
            "transition_lift": "y=L(q_entry+ell_q(s))+Z(h0+Ur*a(s))",
            "online_transition": "one_prevalidated_reset_or_small_scalar_tube_evaluation",
            "online_full_truth_calls": 0,
            "online_y470_roots": 0,
        },
        "online_feasibility": {
            "cycle_seconds": ONLINE_CYCLE_SECONDS,
            "maximum_macrosteps": MAXIMUM_ONLINE_MACROSTEPS,
            "minimum_average_macrostep_seconds": MINIMUM_AVERAGE_MACROSTEP_SECONDS,
            "truth_engine_role": "offline_branch_and_transition_corrector_only",
        },
        "decision_policy": {
            "if_extension_warranted": "authorize_only_bounded_additional_hot_exit_truth",
            "if_extension_not_warranted": "stop_time_microstepping_and_authorize_cold_branch_root_manifest",
            "cold_branch_can_be_certified_without_hot_branch": True,
            "hot_branch_and_complete_impulse_remain_blocked": True,
            "reduced_cycle_remains_blocked": True,
        },
        "input_hashes": {
            "tube_summary": tube.manifest.geometry._sha(tube.CANONICAL_DIRECTORY / "summary.json"),
            "tube_metrics": tube.manifest.geometry._sha(tube.CANONICAL_DIRECTORY / "tube_metrics.json"),
            "tube_arrays": tube.manifest.geometry._sha(
                tube.CANONICAL_DIRECTORY / "tube_model_and_validation.npz"
            ),
            "cold_screen_metrics": tube.manifest.geometry._sha(
                COLD_SCREEN_DIRECTORY / "branch_candidate_screen_metrics.json"
            ),
        },
        "frozen_source_hashes": {
            THIS_RUNNER: tube.manifest.geometry._sha(ROOT / THIS_RUNNER),
            THIS_TEST: tube.manifest.geometry._sha(ROOT / THIS_TEST),
            EXECUTION_RUNNER: tube.manifest.geometry._sha(ROOT / EXECUTION_RUNNER),
            EXECUTION_TEST: tube.manifest.geometry._sha(ROOT / EXECUTION_TEST),
        },
    }


def _update_catalog(summary: dict) -> None:
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
                    "sha256": tube.manifest.geometry._sha(path),
                    "scientific_status": "DEFINITIONS_ONLY",
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case", "path", "bytes", "sha256", "scientific_status"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = tube.manifest.geometry._read(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
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
    tube.manifest.geometry._write_json(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("terminal prognosis manifest already exists")
    locks = _validate_parent(require_clean=True)
    contract = _contract()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    tube.manifest.geometry._write_json(CANONICAL_DIRECTORY / "prognosis_contract.json", contract)
    tube.manifest.geometry._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_commit": PARENT_COMMIT,
            "parent_tree": PARENT_TREE,
            **locks,
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "new_truth_calls": 0,
        "hot_branch_blocked": True,
        "complete_impulse_blocked": True,
        "reduced_cycle_blocked": True,
        "authorized_next": AUTHORIZED_NEXT,
    }
    tube.manifest.geometry._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    tube.manifest.geometry._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "runner": THIS_RUNNER,
            "tests": [THIS_TEST, EXECUTION_TEST],
            "execution_runner": EXECUTION_RUNNER,
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "implementation_commit": tube.manifest.geometry._git("rev-parse", "HEAD"),
            "implementation_tree": tube.manifest.geometry._git("rev-parse", "HEAD^{tree}"),
            "source_hashes": contract["frozen_source_hashes"],
        },
    )
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{tube.manifest.geometry._sha(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Transition terminal prognosis manifest WP10c9d6c7c3b5c4f25dv",
                "",
                "This no-new-truth package prospectively decides whether another bounded block of half-step roots is warranted. Extension requires a consistently decreasing hidden fraction, a projected exit within 24 roots, and at most 10 projected wall-hours.",
                "",
                "If extension is not warranted, the strategy changes from time microstepping to branch equations: G(q,h)=QF(Lq+Zh)=0 with pseudo-arclength continuation at folds. The validated scalar tube remains the transition representation.",
                "",
                "A cold branch may be certified independently from the saved 12 ms candidate. Hot branch truth, a complete impulse, and reduced-cycle execution remain blocked.",
                "",
            ]
        ),
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
    print(json.dumps(tube.manifest.geometry._plain(_freeze()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
