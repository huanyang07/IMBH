#!/usr/bin/env python3
"""Supersede the off-axis manifest after a post-evaluation packaging bug."""

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

import run_causal_inner_hot_mode_off_axis_manifest_wp10c9d6c7c3b5c4f25f9 as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25f9_v2"
AUTHORIZED_NEXT = parent.AUTHORIZED_NEXT
CLASSIFICATION = "hot_discrete_mode_off_axis_manifest_superseded_packaging_fixed"
ARTIFACT = "causal_inner_hot_mode_off_axis_manifest_wp10c9d6c7c3b5c4f25f9_v2"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_HOT_MODE_OFF_AXIS_MANIFEST_"
    "WP10C9D6C7C3B5C4F25F9_V2_2026-08-22.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
EXECUTION_RUNNER = parent.EXECUTION_RUNNER
EXECUTION_TEST = parent.EXECUTION_TEST
THIS_RUNNER = (
    "scripts/run_causal_inner_hot_mode_off_axis_manifest_"
    "wp10c9d6c7c3b5c4f25f9_v2.py"
)
THIS_TEST = (
    "tests/test_causal_inner_hot_mode_off_axis_manifest_"
    "wp10c9d6c7c3b5c4f25f9_v2.py"
)

HOT_CENTER_INDEX = parent.HOT_CENTER_INDEX
DIAGONAL_ARCLENGTH_INDEX = parent.DIAGONAL_ARCLENGTH_INDEX
PHYSICAL_MACRO_STEP_SECONDS = parent.PHYSICAL_MACRO_STEP_SECONDS
PHYSICAL_AXIS_FRACTIONS = parent.PHYSICAL_AXIS_FRACTIONS
HIDDEN_RATE_RANKS = parent.HIDDEN_RATE_RANKS
MAXIMUM_NEW_EXACT_FREE_RATE_CALLS = parent.MAXIMUM_NEW_EXACT_FREE_RATE_CALLS
MAXIMUM_SPLIT_IDENTITY_DEFECT = parent.MAXIMUM_SPLIT_IDENTITY_DEFECT
MAXIMUM_COORDINATE_DECOMPOSITION_DEFECT = parent.MAXIMUM_COORDINATE_DECOMPOSITION_DEFECT
MAXIMUM_COORDINATE_RETRACTION_RESIDUAL = parent.MAXIMUM_COORDINATE_RETRACTION_RESIDUAL
MAXIMUM_GAUGE_RETRACTION_RESIDUAL = parent.MAXIMUM_GAUGE_RETRACTION_RESIDUAL
MAXIMUM_SCALED_ANCHOR_DEPARTURE = parent.MAXIMUM_SCALED_ANCHOR_DEPARTURE
MAXIMUM_COORDINATE_JACOBIAN_CONDITION_NUMBER = parent.MAXIMUM_COORDINATE_JACOBIAN_CONDITION_NUMBER
MAXIMUM_HIDDEN_RATE_HOLDOUT_DEFECT = parent.MAXIMUM_HIDDEN_RATE_HOLDOUT_DEFECT
MAXIMUM_PHYSICAL_AXIS_LINEAR_HOLDOUT_DEFECT = parent.MAXIMUM_PHYSICAL_AXIS_LINEAR_HOLDOUT_DEFECT
MAXIMUM_SEPARABLE_DIAGONAL_OPERATOR_DEFECT = parent.MAXIMUM_SEPARABLE_DIAGONAL_OPERATOR_DEFECT
MAXIMUM_FREE_RATE_VARIATION = parent.MAXIMUM_FREE_RATE_VARIATION
MAXIMUM_EULER_HEUN_CORRECTION_FRACTION = parent.MAXIMUM_EULER_HEUN_CORRECTION_FRACTION


def _helper():
    return parent._helper()


def _decisive_inputs() -> dict[str, Path]:
    return parent._decisive_inputs()


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
    parent_hashes = helper._validate_checksums(parent.CANONICAL_DIRECTORY)
    parent_summary = helper._read(parent.CANONICAL_DIRECTORY / "summary.json")
    if (
        not parent_summary["passed"]
        or not parent_summary["definitions_only"]
        or parent_summary["authorized_next"] != AUTHORIZED_NEXT
    ):
        raise RuntimeError("original off-axis manifest changed")
    underlying = parent._validate_parent(require_clean=False)
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("superseding off-axis manifest requires a clean tracked tree")
    return {
        "superseded_manifest_hashes": parent_hashes,
        "underlying_parent": underlying,
        "execution_failure": (
            "three exact witnesses completed but no scientific classification was "
            "written because the Heun audit used the wrong height-key spelling"
        ),
    }


def _contract(locked: dict) -> dict:
    helper = _helper()
    contract = parent._contract(locked["underlying_parent"])
    contract.update({
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "parent_lock": locked,
        "supersedes": parent.WORK_PACKAGE,
        "scientific_contract_changed": False,
        "execution_repair": {
            "height_key": "maximum_height_ratio",
            "per_witness_exact_scratch_required": True,
            "partial_scientific_result_from_failed_attempt": False,
        },
        "frozen_source_hashes": {
            str(path.relative_to(ROOT)): helper._sha(path) for path in _source_paths()
        },
    })
    return contract


def _update_catalog(summary: dict) -> None:
    helper = _helper()
    cold = parent.parent.manifest.parent.parent._source()._post().manifest.transition.manifest.cold.manifest
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
        writer = csv.DictWriter(handle, fieldnames=(
            "case", "path", "bytes", "sha256", "scientific_status"
        ), lineterminator="\n")
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
        raise RuntimeError("superseding off-axis manifest already exists")
    locked = _validate_parent(require_clean=True)
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(
        CANONICAL_DIRECTORY / "hot_mode_off_axis_contract.json", _contract(locked)
    )
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
        "scientific_contract_changed": False,
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
            "# Superseding hot-mode off-axis manifest",
            "",
            f"Classification: `{CLASSIFICATION}`.",
            "",
            "The scientific states, macro step, truth budget, ranks, holdouts, and thresholds are unchanged. The execution fixes a post-evaluation diagnostic key and requires atomic per-witness scratch persistence.",
            "",
            "The failed attempt produced no classification and authorizes no scientific inference.",
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
