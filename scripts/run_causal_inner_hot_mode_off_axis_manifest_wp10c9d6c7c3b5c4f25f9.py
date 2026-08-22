#!/usr/bin/env python3
"""Freeze the off-axis hot-mode free-field validation contract."""

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

import run_causal_inner_hot_free_field_rom_preflight_wp10c9d6c7c3b5c4f25f8 as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25f9"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25fa"
CLASSIFICATION = "hot_discrete_mode_off_axis_free_field_manifest_frozen"
ARTIFACT = "causal_inner_hot_mode_off_axis_manifest_wp10c9d6c7c3b5c4f25f9"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_HOT_MODE_OFF_AXIS_MANIFEST_"
    "WP10C9D6C7C3B5C4F25F9_2026-08-22.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
EXECUTION_RUNNER = (
    "scripts/run_causal_inner_hot_mode_off_axis_preflight_"
    "wp10c9d6c7c3b5c4f25fa.py"
)
EXECUTION_TEST = (
    "tests/test_causal_inner_hot_mode_off_axis_preflight_"
    "wp10c9d6c7c3b5c4f25fa.py"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_hot_mode_off_axis_manifest_"
    "wp10c9d6c7c3b5c4f25f9.py"
)
THIS_TEST = (
    "tests/test_causal_inner_hot_mode_off_axis_manifest_"
    "wp10c9d6c7c3b5c4f25f9.py"
)

HOT_CENTER_INDEX = 2
DIAGONAL_ARCLENGTH_INDEX = 3
PHYSICAL_MACRO_STEP_SECONDS = 2.5e-4
PHYSICAL_AXIS_FRACTIONS = np.asarray((0.5, 1.0))
HIDDEN_RATE_RANKS = (2, 3, 4)
MAXIMUM_NEW_EXACT_FREE_RATE_CALLS = 3
MAXIMUM_SPLIT_IDENTITY_DEFECT = 5.0e-11
MAXIMUM_COORDINATE_DECOMPOSITION_DEFECT = 5.0e-12
MAXIMUM_COORDINATE_RETRACTION_RESIDUAL = 5.0e-10
MAXIMUM_GAUGE_RETRACTION_RESIDUAL = 5.0e-10
MAXIMUM_SCALED_ANCHOR_DEPARTURE = 5.0e-2
MAXIMUM_COORDINATE_JACOBIAN_CONDITION_NUMBER = 2.5e3
MAXIMUM_HIDDEN_RATE_HOLDOUT_DEFECT = 5.0e-2
MAXIMUM_PHYSICAL_AXIS_LINEAR_HOLDOUT_DEFECT = 2.0e-2
MAXIMUM_SEPARABLE_DIAGONAL_OPERATOR_DEFECT = 5.0e-2
MAXIMUM_FREE_RATE_VARIATION = 1.0e-1
MAXIMUM_EULER_HEUN_CORRECTION_FRACTION = 5.0e-2


def _helper():
    return parent._helper()


def _decisive_inputs() -> dict[str, Path]:
    return {
        "hot_preflight_summary": parent.CANONICAL_DIRECTORY / "summary.json",
        "hot_preflight_metrics": parent.CANONICAL_DIRECTORY
        / "hot_free_field_metrics.json",
        "hot_preflight_arrays": parent.CANONICAL_DIRECTORY
        / "hot_free_field_arrays.npz",
    }


def _source_paths() -> tuple[Path, ...]:
    return (
        ROOT / THIS_RUNNER,
        ROOT / THIS_TEST,
        ROOT / EXECUTION_RUNNER,
        ROOT / EXECUTION_TEST,
        ROOT / "src/imri_qpe/layer3_minidisk_1d/conservative_free_field_rom.py",
        ROOT / parent.THIS_RUNNER,
    )


def _validate_parent(*, require_clean: bool) -> dict:
    helper = _helper()
    hashes = helper._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = helper._read(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = helper._read(
        parent.CANONICAL_DIRECTORY / "hot_free_field_metrics.json"
    )
    failed = [name for name, passed in metrics["gates"].items() if not passed]
    if (
        summary["passed"]
        or summary["classification"] != parent.FAIL_CLASSIFICATION
        or metrics["classification"] != parent.FAIL_CLASSIFICATION
        or failed != ["cold_physical_subspace_extension"]
        or not metrics["gates"]["hidden_rate_holdout"]
        or not metrics["gates"]["polynomial_operator_holdout"]
        or not metrics["gates"]["offline_cost"]
    ):
        raise RuntimeError("hot/cold separation diagnosis changed")
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("hot-mode off-axis manifest requires a clean tracked tree")
    return {
        "hot_preflight_hashes": hashes,
        "failed_parent_gates": failed,
        "parent_classification": summary["classification"],
    }


def _contract(locked: dict) -> dict:
    helper = _helper()
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "parent_lock": locked,
        "decisive_input_hashes": {
            name: helper._sha(path) for name, path in _decisive_inputs().items()
        },
        "frozen_source_hashes": {
            str(path.relative_to(ROOT)): helper._sha(path) for path in _source_paths()
        },
        "architecture": {
            "continuous_state": "q82 plus hot hidden amplitudes",
            "discrete_mode": "hot distinct from cold",
            "physical_field": "original unconstrained reaction-free tangent",
            "sampling_axes": (
                "artificial fixed-Q arclength is offline-only; the second axis "
                "is the original free-field direction"
            ),
            "physical_macro_step_seconds": PHYSICAL_MACRO_STEP_SECONDS,
            "physical_axis_fractions": PHYSICAL_AXIS_FRACTIONS.tolist(),
            "diagonal_arclength_index": DIAGONAL_ARCLENGTH_INDEX,
            "hidden_rate_candidate_ranks": list(HIDDEN_RATE_RANKS),
            "macro_ledger_projection": "forbidden; all 82 macro rates retained",
        },
        "truth_budget": {
            "new_exact_free_rate_calls": MAXIMUM_NEW_EXACT_FREE_RATE_CALLS,
            "new_fixed_Q_reaction_calls": 0,
            "new_nonlinear_roots": 0,
            "new_BDF_microsteps": 0,
        },
        "gates": {
            "maximum_split_identity_defect": MAXIMUM_SPLIT_IDENTITY_DEFECT,
            "maximum_coordinate_decomposition_defect": MAXIMUM_COORDINATE_DECOMPOSITION_DEFECT,
            "maximum_coordinate_retraction_residual": MAXIMUM_COORDINATE_RETRACTION_RESIDUAL,
            "maximum_gauge_retraction_residual": MAXIMUM_GAUGE_RETRACTION_RESIDUAL,
            "maximum_scaled_anchor_departure": MAXIMUM_SCALED_ANCHOR_DEPARTURE,
            "maximum_coordinate_jacobian_condition_number": MAXIMUM_COORDINATE_JACOBIAN_CONDITION_NUMBER,
            "maximum_hidden_rate_holdout_defect": MAXIMUM_HIDDEN_RATE_HOLDOUT_DEFECT,
            "maximum_physical_axis_linear_holdout_defect": MAXIMUM_PHYSICAL_AXIS_LINEAR_HOLDOUT_DEFECT,
            "maximum_separable_diagonal_operator_defect": MAXIMUM_SEPARABLE_DIAGONAL_OPERATOR_DEFECT,
            "maximum_free_rate_variation": MAXIMUM_FREE_RATE_VARIATION,
            "maximum_euler_heun_correction_fraction": MAXIMUM_EULER_HEUN_CORRECTION_FRACTION,
            "reconstruction_and_physical_state_gates_unchanged": True,
        },
        "decision": {
            "pass": "authorize a truth-free conservative hot-mode engine replay",
            "off_axis_rank_failure": "increase hot hidden rank prospectively",
            "diagonal_failure": "replace the separable patch with an adaptive multidimensional local atlas",
            "physical_failure": "reject the sampled hot-mode region",
        },
        "fixed_Q_physical_phase_authorized": False,
        "complete_cycle_execution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }


def _update_catalog(summary: dict) -> None:
    helper = _helper()
    cold = parent.manifest.parent.parent._source()._post().manifest.transition.manifest.cold.manifest
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
        raise RuntimeError("hot-mode off-axis manifest already exists")
    locked = _validate_parent(require_clean=True)
    CANONICAL_DIRECTORY.mkdir(parents=True)
    contract = _contract(locked)
    helper._write_json(CANONICAL_DIRECTORY / "hot_mode_off_axis_contract.json", contract)
    helper._write_json(CANONICAL_DIRECTORY / "input_lock.json", locked)
    helper._write_json(CANONICAL_DIRECTORY / "provenance.json", {
        "runner": THIS_RUNNER,
        "test": THIS_TEST,
        "implementation_commit": helper._git("rev-parse", "HEAD"),
        "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"),
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
    })
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "hot_mode_off_axis_preflight_authorized": True,
        "fixed_Q_physical_phase_authorized": False,
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{helper._sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join((
            "# Hot-mode off-axis free-field manifest",
            "",
            f"Classification: `{CLASSIFICATION}`.",
            "",
            "The failed cold-subspace extension is preserved. This package defines a separate hot discrete mode and tests it along the original physical/free direction, not along fixed-Q time.",
            "",
            f"The physical macro step is `{PHYSICAL_MACRO_STEP_SECONDS:.6e}` s. Two axial witnesses and one diagonal cross-witness are authorized; no fixed-Q reaction, nonlinear root, or BDF microstep is allowed.",
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
