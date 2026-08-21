#!/usr/bin/env python3
"""Supersede the adaptive hot-exit manifest after a packaging-only failure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_adaptive_hot_exit_phase_atlas_manifest_wp10c9d6c7c3b5c4f25ed as original  # noqa: E402


SCHEMA_VERSION = 2
WORK_PACKAGE = original.WORK_PACKAGE
AUTHORIZED_NEXT = original.AUTHORIZED_NEXT
PARENT_COMMIT = "020344d7d0f677e1f215329eece8481edc55cd13"
PARENT_TREE = "028faa166e19eaddf1259b37983da97d5d33de33"
CLASSIFICATION = (
    "adaptive_hot_exit_phase_atlas_manifest_superseded_legacy_namespace_"
    "collision_repaired_window1_truth_cache_recovered_bitwise"
)

NODE_COUNT = original.NODE_COUNT
INITIAL_DURATION_SECONDS = original.INITIAL_DURATION_SECONDS
MAXIMUM_DURATION_SECONDS = original.MAXIMUM_DURATION_SECONDS
MAXIMUM_WINDOWS = original.MAXIMUM_WINDOWS
RATE_BASIS_RANKS = original.RATE_BASIS_RANKS
PICARD_UPDATES = original.PICARD_UPDATES
MAXIMUM_UNIQUE_RATE_STATES_PER_WINDOW = original.MAXIMUM_UNIQUE_RATE_STATES_PER_WINDOW
MAXIMUM_TRAINING_NORMAL_RATE_DEFECT = original.MAXIMUM_TRAINING_NORMAL_RATE_DEFECT
MAXIMUM_PROJECTED_COLLOCATION_DEFECT = original.MAXIMUM_PROJECTED_COLLOCATION_DEFECT
MAXIMUM_FULL_COLLOCATION_DEFECT = original.MAXIMUM_FULL_COLLOCATION_DEFECT
MAXIMUM_NORMAL_RATE_DEFECT = original.MAXIMUM_NORMAL_RATE_DEFECT
MINIMUM_RATE_DIRECTION_COSINE = original.MINIMUM_RATE_DIRECTION_COSINE
MAXIMUM_DECODER_COORDINATE_ERROR_OVER_TRANSITION_PATH = original.MAXIMUM_DECODER_COORDINATE_ERROR_OVER_TRANSITION_PATH
MAXIMUM_Q3_RELATIVE_DRIFT = original.MAXIMUM_Q3_RELATIVE_DRIFT
MINIMUM_RECONSTRUCTION_FACTOR = original.MINIMUM_RECONSTRUCTION_FACTOR
GROW_MAXIMUM_FULL_COLLOCATION_DEFECT = original.GROW_MAXIMUM_FULL_COLLOCATION_DEFECT
GROW_MAXIMUM_NORMAL_RATE_DEFECT = original.GROW_MAXIMUM_NORMAL_RATE_DEFECT
GROW_MINIMUM_RATE_DIRECTION_COSINE = original.GROW_MINIMUM_RATE_DIRECTION_COSINE
GROW_MAXIMUM_DECODER_COORDINATE_ERROR_OVER_TRANSITION_PATH = original.GROW_MAXIMUM_DECODER_COORDINATE_ERROR_OVER_TRANSITION_PATH
GROW_MAXIMUM_Q3_RELATIVE_DRIFT = original.GROW_MAXIMUM_Q3_RELATIVE_DRIFT
HIDDEN_SECANT_FRACTION_MAX = original.HIDDEN_SECANT_FRACTION_MAX
HIDDEN_EXIT_PERSISTENCE_WINDOWS = original.HIDDEN_EXIT_PERSISTENCE_WINDOWS
RANK16_HIDDEN_AMPLITUDE_MIN = original.RANK16_HIDDEN_AMPLITUDE_MIN
MAXIMUM_MACRO_DRIFT_FROM_SEED = original.MAXIMUM_MACRO_DRIFT_FROM_SEED
architecture = original.architecture
legacy_exit = original.legacy_exit

ARTIFACT = "causal_inner_adaptive_hot_exit_phase_atlas_manifest_wp10c9d6c7c3b5c4f25ed_v2"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
RECOVERED_METRICS = CANONICAL_DIRECTORY / "recovered_window_01_truth_metrics.json"
RECOVERED_ARRAYS = CANONICAL_DIRECTORY / "recovered_window_01_truth_arrays.npz"
ORIGINAL_SCRATCH = ROOT / "outputs/checkpoints/causal_inner_adaptive_hot_exit_phase_atlas_wp10c9d6c7c3b5c4f25ee/window_01"
THIS_RUNNER = "scripts/run_causal_inner_adaptive_hot_exit_phase_atlas_manifest_v2_wp10c9d6c7c3b5c4f25ed.py"
THIS_TEST = "tests/test_causal_inner_adaptive_hot_exit_phase_atlas_manifest_v2_wp10c9d6c7c3b5c4f25ed.py"
EXECUTION_RUNNER = original.EXECUTION_RUNNER
EXECUTION_TEST = original.EXECUTION_TEST
REPORT_RELATIVE = "docs/reports/current/CODEX_CAUSAL_INNER_ADAPTIVE_HOT_EXIT_PHASE_ATLAS_MANIFEST_V2_WP10C9D6C7C3B5C4F25ED_2026-08-21.md"
REPORT_PATH = ROOT / REPORT_RELATIVE


def _helper():
    return original._helper()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _scratch_inputs() -> dict[str, Path]:
    index = json.loads((ORIGINAL_SCRATCH / "index.json").read_text(encoding="utf-8"))
    paths = {
        "identity": ORIGINAL_SCRATCH / "identity.json",
        "index": ORIGINAL_SCRATCH / "index.json",
    }
    for number, entry in enumerate(index["records"], start=1):
        paths[f"record_{number:02d}_metrics"] = ORIGINAL_SCRATCH / entry["metrics_file"]
        paths[f"record_{number:02d}_arrays"] = ORIGINAL_SCRATCH / entry["arrays_file"]
    return paths


def _validate_parent(*, require_clean: bool) -> dict:
    helper = _helper()
    if helper._git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("adaptive v2 parent commit changed")
    if helper._git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("adaptive v2 parent tree changed")
    original_hashes = helper._validate_checksums(original.CANONICAL_DIRECTORY)
    original_summary = helper._read(original.CANONICAL_DIRECTORY / "summary.json")
    if not original_summary["passed"] or not original_summary["adaptive_phase_atlas_execution_authorized"]:
        raise RuntimeError("original adaptive manifest changed")
    identity = helper._read(ORIGINAL_SCRATCH / "identity.json")
    index = helper._read(ORIGINAL_SCRATCH / "index.json")
    if (
        identity["manifest_hashes"] != original_hashes
        or identity["window_index"] != 1
        or identity["duration_seconds"] != INITIAL_DURATION_SECONDS
        or identity["basis_rank"] != 4
        or len(index["records"]) != 14
    ):
        raise RuntimeError("recoverable Window-1 truth cache identity changed")
    for entry in index["records"]:
        arrays = helper._load_npz(ORIGINAL_SCRATCH / entry["arrays_file"])
        coordinate = np.ascontiguousarray(arrays["coordinate470"])
        if hashlib.sha256(coordinate.tobytes()).hexdigest() != entry["key"]:
            raise RuntimeError("recoverable exact-rate coordinate key changed")
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("adaptive v2 manifest requires a clean tracked tree")
    return {
        "original_manifest_hashes": original_hashes,
        "recovered_scratch_hashes": {
            name: _sha(path) for name, path in _scratch_inputs().items()
        },
    }


def _recover_cache() -> tuple[dict, dict[str, np.ndarray]]:
    helper = _helper()
    index = helper._read(ORIGINAL_SCRATCH / "index.json")
    records = []
    loaded = []
    for entry in index["records"]:
        records.append(helper._read(ORIGINAL_SCRATCH / entry["metrics_file"]))
        loaded.append(helper._load_npz(ORIGINAL_SCRATCH / entry["arrays_file"]))
    names = (
        "coordinate470",
        "decoded_primitive_state",
        "recovered_coordinate470",
        "Q3",
        "coordinate_rate470_per_s",
        "scaled_fixed_Q_rate560_per_s",
        "scaled_reaction_action560_per_s",
    )
    arrays = {name: np.stack([item[name] for item in loaded]) for name in names}
    metrics = {
        "schema_version": 1,
        "source": "hash-validated interrupted Window-1 scratch",
        "failure_location": "post-truth event-diagnostic namespace collision",
        "scientific_rate_evaluations_completed": True,
        "canonical_window_result_previously_written": False,
        "record_count": len(records),
        "records": records,
    }
    return metrics, arrays


def _decisive_inputs() -> dict[str, Path]:
    inputs = dict(original._decisive_inputs())
    inputs.update({
        "original_manifest_summary": original.CANONICAL_DIRECTORY / "summary.json",
        "original_manifest_contract": original.CANONICAL_DIRECTORY / "adaptive_hot_exit_phase_atlas_contract.json",
        "recovered_truth_metrics": RECOVERED_METRICS,
        "recovered_truth_arrays": RECOVERED_ARRAYS,
    })
    return inputs


def _contract(parent: dict) -> dict:
    helper = _helper()
    contract = original._contract()
    contract.update({
        "schema_version": SCHEMA_VERSION,
        "classification": CLASSIFICATION,
        "supersedes": str(original.CANONICAL_DIRECTORY.relative_to(ROOT)),
        "runtime_repair": {
            "failure_was_after_all_window1_truth_calls": True,
            "failure_was_before_any_canonical_window_result_or_propagation": True,
            "cause": "legacy hot-exit helper module alias mutated by import-time half-step recovery configuration",
            "repair": "self-contained immutable event feature calculation",
            "truth_cache_reused_bitwise": True,
            "physical_equations_or_gates_changed": False,
        },
        "recovered_scratch_hashes": parent["recovered_scratch_hashes"],
        "decisive_input_hashes": {
            name: helper._sha(path) for name, path in _decisive_inputs().items()
        },
        "frozen_source_hashes": {
            relative: helper._sha(ROOT / relative)
            for relative in (
                THIS_RUNNER,
                THIS_TEST,
                EXECUTION_RUNNER,
                EXECUTION_TEST,
                original.PHASE_SOURCE,
                original.FIXED_Q_SOURCE,
                original.EXACT_RATE_SOURCE,
                original.POST_RUNNER,
                original.LEGACY_EXIT_SOURCE,
            )
        },
    })
    return contract


def _update_catalog(summary: dict) -> None:
    helper = _helper()
    rejected = architecture.rejected
    manifest_path = rejected.post.manifest.transition.manifest.cold.manifest.CANONICAL_MANIFEST
    summary_path = rejected.post.manifest.transition.manifest.cold.manifest.CANONICAL_SUMMARY
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": helper._sha(path), "scientific_status": "DEFINITIONS_ONLY_RECOVERED_CACHE"})
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    catalog = helper._read(summary_path)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": summary["classification"], "passed": True}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": PARENT_COMMIT, "latest_work_package": WORK_PACKAGE})
    helper._write_json(summary_path, catalog)


def _freeze() -> dict:
    helper = _helper()
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("adaptive phase-atlas v2 manifest already exists")
    parent = _validate_parent(require_clean=True)
    metrics, arrays = _recover_cache()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(RECOVERED_METRICS, metrics)
    with RECOVERED_ARRAYS.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    contract = _contract(parent)
    helper._write_json(CANONICAL_DIRECTORY / "adaptive_hot_exit_phase_atlas_contract.json", contract)
    helper._write_json(CANONICAL_DIRECTORY / "input_lock.json", parent)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only_with_recovered_truth_cache": True,
        "adaptive_phase_atlas_execution_authorized": True,
        "recovered_window1_exact_rate_calls": metrics["record_count"],
        "new_truth_calls": 0,
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    helper._write_json(CANONICAL_DIRECTORY / "provenance.json", {"runner": THIS_RUNNER, "test": THIS_TEST, "implementation_commit": helper._git("rev-parse", "HEAD"), "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"), "python": sys.version, "numpy": np.__version__, "platform": platform.platform()})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{helper._sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(("# Adaptive hot-exit phase-atlas manifest v2", "", f"Classification: `{CLASSIFICATION}`.", "", "Window 1 completed all 14 prospectively authorized new exact-rate calls, then stopped before canonicalization because an imported legacy helper had mutated its manifest alias. No result or endpoint was propagated.", "", "V2 hash-locks and canonicalizes those witnesses, replaces only the mutable event helper with the same explicit historical formula, and authorizes exact bitwise replay of Window 1. Physics, duration, basis, event thresholds, and every acceptance gate are unchanged.", "")), encoding="utf-8")
    _update_catalog(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if not args.freeze:
        parser.error("use --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
