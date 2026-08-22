#!/usr/bin/env python3
"""Freeze the prospective hot-state conservative free-field ROM preflight."""

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

import run_causal_inner_reaction_free_field_architecture_diagnosis_wp10c9d6c7c3b5c4f25f6 as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25f7"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25f8"
CLASSIFICATION = "hot_conservative_free_field_rom_preflight_manifest_frozen"
ARTIFACT = "causal_inner_hot_free_field_rom_manifest_wp10c9d6c7c3b5c4f25f7"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_HOT_FREE_FIELD_ROM_MANIFEST_"
    "WP10C9D6C7C3B5C4F25F7_2026-08-22.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_hot_free_field_rom_manifest_"
    "wp10c9d6c7c3b5c4f25f7.py"
)
THIS_TEST = (
    "tests/test_causal_inner_hot_free_field_rom_manifest_"
    "wp10c9d6c7c3b5c4f25f7.py"
)
EXECUTION_RUNNER = (
    "scripts/run_causal_inner_hot_free_field_rom_preflight_"
    "wp10c9d6c7c3b5c4f25f8.py"
)
EXECUTION_TEST = (
    "tests/test_causal_inner_hot_free_field_rom_preflight_"
    "wp10c9d6c7c3b5c4f25f8.py"
)
CORE_SOURCE = "src/imri_qpe/layer3_minidisk_1d/conservative_free_field_rom.py"
CORE_TEST = "tests/test_conservative_free_field_rom.py"

NODE_COUNT = 5
TRAINING_INDICES = np.asarray((0, 2, 4), dtype=int)
HOLDOUT_INDICES = np.asarray((1, 3), dtype=int)
HIDDEN_RATE_RANKS = (2, 3)
MAXIMUM_SPLIT_IDENTITY_DEFECT = 5.0e-11
MAXIMUM_COORDINATE_DECOMPOSITION_DEFECT = 5.0e-12
MAXIMUM_HIDDEN_RATE_HOLDOUT_DEFECT = 5.0e-2
MAXIMUM_POLYNOMIAL_HOLDOUT_DEFECT = 5.0e-2
MAXIMUM_COLD_PHYSICAL_SUBSPACE_DEFECT = 2.5e-1
MAXIMUM_COORDINATE_JACOBIAN_CONDITION_NUMBER = 2.5e3
MAXIMUM_PROJECTED_256_WITNESS_WALL_HOURS = 24.0
MAXIMUM_NEW_EXACT_FREE_RATE_CALLS = NODE_COUNT


def _helper():
    return parent._helper()


def _decisive_inputs() -> dict[str, Path]:
    return {
        "architecture_summary": parent.CANONICAL_DIRECTORY / "summary.json",
        "architecture_metrics": parent.CANONICAL_DIRECTORY
        / "reaction_free_field_metrics.json",
        "architecture_arrays": parent.CANONICAL_DIRECTORY
        / "reaction_free_field_arrays.npz",
        "arclength_summary": parent.ARCLENGTH_DIRECTORY / "summary.json",
        "arclength_metrics": parent.ARCLENGTH_DIRECTORY
        / "arclength_segment_metrics.json",
        "arclength_arrays": parent.ARCLENGTH_DIRECTORY
        / "arclength_segment_arrays.npz",
    }


def _source_paths() -> dict[str, Path]:
    f5 = parent.parent
    modules = {
        "arclength_runner": Path(f5.__file__).resolve(),
        "exact_chart": Path(f5._exact_chart().__file__).resolve(),
        "fixed_rate_source": Path(
            f5._source()._post().exact_rate.rate_source.__file__
        ).resolve(),
    }
    paths = {
        "manifest_runner": ROOT / THIS_RUNNER,
        "manifest_test": ROOT / THIS_TEST,
        "execution_runner": ROOT / EXECUTION_RUNNER,
        "execution_test": ROOT / EXECUTION_TEST,
        "core_source": ROOT / CORE_SOURCE,
        "core_test": ROOT / CORE_TEST,
        "architecture_diagnosis": ROOT / parent.THIS_RUNNER,
        **modules,
    }
    return paths


def _validate_parent(*, require_clean: bool) -> dict:
    helper = _helper()
    architecture_hashes = helper._validate_checksums(parent.CANONICAL_DIRECTORY)
    arclength_hashes = helper._validate_checksums(parent.ARCLENGTH_DIRECTORY)
    architecture = helper._read(parent.CANONICAL_DIRECTORY / "summary.json")
    arclength = helper._read(parent.ARCLENGTH_DIRECTORY / "summary.json")
    if (
        not architecture["passed"]
        or architecture["authorized_next"] != WORK_PACKAGE
        or architecture["fixed_Q_physical_phase_authorized"]
        or not architecture["conservative_free_field_hidden_amplitude_rom_selected"]
    ):
        raise RuntimeError("free-field architecture decision changed")
    if not arclength["passed"] or arclength["work_package"] != parent.parent.WORK_PACKAGE:
        raise RuntimeError("accepted arclength sample changed")
    arrays = helper._load_npz(
        parent.ARCLENGTH_DIRECTORY / "arclength_segment_arrays.npz"
    )
    if np.asarray(arrays["coordinates"]).shape[0] != NODE_COUNT:
        raise RuntimeError("arclength sample node count changed")
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("free-field manifest requires a clean tracked tree")
    return {
        "architecture_hashes": architecture_hashes,
        "arclength_hashes": arclength_hashes,
    }


def _contract(parent_lock: dict) -> dict:
    helper = _helper()
    sources = _source_paths()
    missing = [name for name, path in sources.items() if not path.exists()]
    if missing:
        raise RuntimeError(f"free-field preflight source is missing: {missing}")
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "parent_lock": parent_lock,
        "decisive_input_hashes": {
            name: helper._sha(path) for name, path in _decisive_inputs().items()
        },
        "frozen_source_hashes": {
            str(path.relative_to(ROOT)): helper._sha(path)
            for path in sources.values()
        },
        "evaluation": {
            "states": "five accepted final Lobatto states from WP10c9d6c7c3b5c4f25f5",
            "node_count": NODE_COUNT,
            "training_indices": TRAINING_INDICES.tolist(),
            "holdout_indices": HOLDOUT_INDICES.tolist(),
            "hidden_rate_candidate_ranks": list(HIDDEN_RATE_RANKS),
            "field": "original monolithic free tangent mapped by the exact coordinate Jacobian",
            "fixed_Q_reaction_calls": 0,
            "new_nonlinear_roots": 0,
            "new_BDF_microsteps": 0,
        },
        "gates": {
            "maximum_split_identity_defect": MAXIMUM_SPLIT_IDENTITY_DEFECT,
            "maximum_coordinate_decomposition_defect": MAXIMUM_COORDINATE_DECOMPOSITION_DEFECT,
            "maximum_hidden_rate_holdout_defect": MAXIMUM_HIDDEN_RATE_HOLDOUT_DEFECT,
            "maximum_polynomial_holdout_defect": MAXIMUM_POLYNOMIAL_HOLDOUT_DEFECT,
            "maximum_cold_physical_subspace_defect": MAXIMUM_COLD_PHYSICAL_SUBSPACE_DEFECT,
            "maximum_coordinate_jacobian_condition_number": MAXIMUM_COORDINATE_JACOBIAN_CONDITION_NUMBER,
            "maximum_projected_256_witness_wall_hours": MAXIMUM_PROJECTED_256_WITNESS_WALL_HOURS,
            "maximum_new_exact_free_rate_calls": MAXIMUM_NEW_EXACT_FREE_RATE_CALLS,
            "reconstruction_and_physical_state_gates_unchanged": True,
        },
        "decision": {
            "pass": "authorize a truth-free conservative hidden-amplitude engine replay",
            "rank_holdout_failure": "increase the prospectively declared hidden rank before any trajectory",
            "physical_subspace_failure": "separate the hot discrete mode; do not extrapolate the cold operator",
            "cost_failure": "profile and optimize the direct free evaluator before complete-cycle acquisition",
        },
        "fixed_Q_arclength_physical_time_authorized": False,
        "complete_cycle_execution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }


def _update_catalog(summary: dict) -> None:
    helper = _helper()
    cold = parent.parent._source()._post().manifest.transition.manifest.cold.manifest
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
                "scientific_status": "SUPPORTED",
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
        raise RuntimeError("free-field ROM manifest already exists")
    parent_lock = _validate_parent(require_clean=True)
    contract = _contract(parent_lock)
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "hot_free_field_rom_contract.json", contract)
    helper._write_json(CANONICAL_DIRECTORY / "input_lock.json", parent_lock)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "hot_free_field_rom_preflight_authorized": True,
        "fixed_Q_physical_phase_authorized": False,
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    helper._write_json(CANONICAL_DIRECTORY / "provenance.json", {
        "runner": THIS_RUNNER,
        "test": THIS_TEST,
        "definition_commit": helper._git("rev-parse", "HEAD"),
        "definition_tree": helper._git("rev-parse", "HEAD^{tree}"),
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
    })
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{helper._sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names
        ),
        encoding="utf-8",
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join((
            "# Hot conservative free-field ROM preflight manifest",
            "",
            f"Classification: `{CLASSIFICATION}`.",
            "",
            "Evaluate the original reaction-free physical tangent at the five accepted hot arclength states. Retain the 82-coordinate macro ledger exactly and test a rank-2/rank-3 hidden-rate basis on held-out Lobatto nodes.",
            "",
            "The fixed-Q reaction is forbidden in every binding evaluation. A pass requires conservative coordinate closure, held-out operator accuracy, physical-subspace consistency, unchanged physical guards, and a projected 256-witness offline acquisition cost below 24 wall hours.",
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
