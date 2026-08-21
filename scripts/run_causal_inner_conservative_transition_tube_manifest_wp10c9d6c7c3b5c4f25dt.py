#!/usr/bin/env python3
"""Freeze a train-only conservative transition-tube surrogate."""

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

import run_causal_inner_transition_tube_geometry_manifest_wp10c9d6c7c3b5c4f25dr as geometry_manifest  # noqa: E402
import run_causal_inner_transition_tube_geometry_wp10c9d6c7c3b5c4f25ds as geometry  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25dt"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25du"
PARENT_COMMIT = "843d4708fe0f9e733f31f762ed11f78ed5b6c1fc"
PARENT_TREE = "f6bdc69e32e5c4b9eb62029a73797cd59e7a9b7a"
CLASSIFICATION = (
    "train_only_rank_adaptive_conservative_scalar_transition_tube_"
    "surrogate_manifest_frozen"
)

ENERGY_CAPTURE_TARGET = 0.9999
MAXIMUM_HIDDEN_EMBEDDING_RANK = 16
MAXIMUM_HOLDOUT_HIDDEN_ERROR_OVER_PATH = 1.0e-2
MAXIMUM_HOLDOUT_FULL_ERROR_OVER_PATH = 1.5e-2
MAXIMUM_HOLDOUT_FULL_ERROR_OVER_LOCAL_CHORD = 5.0e-2
MAXIMUM_MACRO_LEDGER_HOLDOUT_ERROR = 1.0e-4
MAXIMUM_FINE_SECANT_RELATIVE_ERROR = 1.0e-1
MINIMUM_FINE_SECANT_DIRECTION_COSINE = 0.995
CONSERVATIVE_CLOSURE_TOLERANCE = 1.0e-10
MAXIMUM_ONLINE_LIFT_FLOPS = 1_000_000
MAXIMUM_ONLINE_TABLE_BYTES = 2 * 1024 * 1024

ARTIFACT = "causal_inner_conservative_transition_tube_manifest_wp10c9d6c7c3b5c4f25dt"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = "scripts/run_causal_inner_conservative_transition_tube_manifest_wp10c9d6c7c3b5c4f25dt.py"
THIS_TEST = "tests/test_causal_inner_conservative_transition_tube_manifest_wp10c9d6c7c3b5c4f25dt.py"
EXECUTION_RUNNER = "scripts/run_causal_inner_conservative_transition_tube_wp10c9d6c7c3b5c4f25du.py"
EXECUTION_TEST = "tests/test_causal_inner_conservative_transition_tube_wp10c9d6c7c3b5c4f25du.py"
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_CONSERVATIVE_TRANSITION_TUBE_"
    "MANIFEST_WP10C9D6C7C3B5C4F25DT_2026-08-21.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _validate_parent(*, require_clean: bool) -> dict:
    if geometry._git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("conservative tube parent commit changed")
    if geometry._git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("conservative tube parent tree changed")
    hashes = geometry._validate_checksums(geometry.CANONICAL_DIRECTORY)
    summary = geometry._read(geometry.CANONICAL_DIRECTORY / "summary.json")
    metrics = geometry._read(geometry.CANONICAL_DIRECTORY / "geometry_metrics.json")
    if (
        not summary["passed"]
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["transition_dynamic_dimension"] != 1
        or summary["hot_exit_observed"]
        or metrics["failed_gates"]
    ):
        raise RuntimeError("scalar transition-tube geometry result changed")
    if require_clean and geometry._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("conservative tube manifest requires a clean tracked tree")
    return {"geometry_hashes": hashes}


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "parent_geometry": {
            "work_package": geometry.WORK_PACKAGE,
            "classification": geometry.PASS_CLASSIFICATION,
            "summary_sha256": geometry._sha(
                geometry.CANONICAL_DIRECTORY / "summary.json"
            ),
            "metrics_sha256": geometry._sha(
                geometry.CANONICAL_DIRECTORY / "geometry_metrics.json"
            ),
            "arrays_sha256": geometry._sha(
                geometry.CANONICAL_DIRECTORY / "geometry_arrays.npz"
            ),
        },
        "training_policy": {
            "training_state_indices": geometry_manifest.TRAIN_STATE_INDICES,
            "held_out_state_indices": geometry_manifest.HOLDOUT_STATE_INDICES,
            "basis_fit_uses_training_states_only": True,
            "held_out_states_never_select_rank_or_coefficients": True,
            "rank_selection_energy_target": ENERGY_CAPTURE_TARGET,
            "maximum_hidden_embedding_rank": MAXIMUM_HIDDEN_EMBEDDING_RANK,
        },
        "surrogate": {
            "state": "(q_entry_82,s_scalar)",
            "progress_dynamics": "piecewise_constant_ds_dt_on_training_segments",
            "hidden_decoder": "h0_plus_Ur_times_piecewise_linear_coefficients_of_s",
            "conservative_ledger": "ell_q(s)_82_piecewise_linear",
            "coordinate_lift": "y=L(q_entry+ell_q(s))+Z(h0+Ur*a(s))",
            "partial_endpoint_reset": "Delta_q_partial=ell_q(s_terminal)",
            "partial_reset_is_not_a_hot_exit_impulse": True,
            "online_y470_residual_calls": 0,
            "online_truth_calls": 0,
        },
        "binding_gates": {
            "energy_capture_target": ENERGY_CAPTURE_TARGET,
            "maximum_hidden_embedding_rank": MAXIMUM_HIDDEN_EMBEDDING_RANK,
            "maximum_holdout_hidden_error_over_path": MAXIMUM_HOLDOUT_HIDDEN_ERROR_OVER_PATH,
            "maximum_holdout_full_error_over_path": MAXIMUM_HOLDOUT_FULL_ERROR_OVER_PATH,
            "maximum_holdout_full_error_over_local_chord": MAXIMUM_HOLDOUT_FULL_ERROR_OVER_LOCAL_CHORD,
            "maximum_macro_ledger_holdout_error": MAXIMUM_MACRO_LEDGER_HOLDOUT_ERROR,
            "maximum_fine_secant_relative_error": MAXIMUM_FINE_SECANT_RELATIVE_ERROR,
            "minimum_fine_secant_direction_cosine": MINIMUM_FINE_SECANT_DIRECTION_COSINE,
            "conservative_closure_tolerance": CONSERVATIVE_CLOSURE_TOLERANCE,
            "maximum_online_lift_flops": MAXIMUM_ONLINE_LIFT_FLOPS,
            "maximum_online_table_bytes": MAXIMUM_ONLINE_TABLE_BYTES,
        },
        "decision_policy": {
            "pass": "authorize_tube_forecast_and_targeted_truth_design_only",
            "fail": "reject_piecewise_linear_rank_adaptive_tube_policy",
            "hot_branch_truth_authorized": False,
            "complete_impulse_fit_authorized": False,
            "reduced_cycle_authorized": False,
        },
        "frozen_source_hashes": {
            THIS_RUNNER: geometry._sha(ROOT / THIS_RUNNER),
            THIS_TEST: geometry._sha(ROOT / THIS_TEST),
            EXECUTION_RUNNER: geometry._sha(ROOT / EXECUTION_RUNNER),
            EXECUTION_TEST: geometry._sha(ROOT / EXECUTION_TEST),
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
                    "sha256": geometry._sha(path),
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
    catalog = geometry._read(CANONICAL_SUMMARY)
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
    geometry._write_json(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("conservative transition-tube manifest already exists")
    locks = _validate_parent(require_clean=True)
    contract = _contract()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    geometry._write_json(CANONICAL_DIRECTORY / "tube_contract.json", contract)
    geometry._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_commit": PARENT_COMMIT,
            "parent_tree": PARENT_TREE,
            "geometry_hashes": locks["geometry_hashes"],
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "train_only_rank_selection": True,
        "new_truth_calls": 0,
        "complete_impulse_fit_authorized": False,
        "reduced_cycle_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    geometry._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    geometry._write_json(
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
            "implementation_commit": geometry._git("rev-parse", "HEAD"),
            "implementation_tree": geometry._git("rev-parse", "HEAD^{tree}"),
            "source_hashes": contract["frozen_source_hashes"],
        },
    )
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{geometry._sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names
        ),
        encoding="utf-8",
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Conservative transition-tube manifest WP10c9d6c7c3b5c4f25dt",
                "",
                "A train-only rank-adaptive surrogate is frozen for the certified scalar transition geometry. The online state is (q_entry,s); the conservative ledger ell_q(s) and hidden curve are table-driven.",
                "",
                "The decoder y=L(q_entry+ell_q(s))+Z(h0+Ur a(s)) preserves the 82 macro coordinates algebraically. Alternating trajectory states remain held out.",
                "",
                "The terminal point is only the end of the observed tube. It is not a hot exit and its partial ledger is not a complete impulse map.",
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
    print(json.dumps(geometry._plain(_freeze()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
