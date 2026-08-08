#!/usr/bin/env python3
"""Harden and freeze the complete 10 ms nonlinear screen."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import sys

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_nonlinear_fourth_duration_rung_manifest_wp10c9d6c7c3b5c4a as c4a  # noqa: E402
import run_causal_inner_nonlinear_ten_ms_cost_pilot_wp10c9d6c7c3b5c4b as c4b  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4b1"
ANALYZED_BASE_COMMIT = "b2aa5d5c956dd36599c154b096552457a2925f2e"
ANALYZED_BASE_PARENT = "5f248c5c9249e003b0237e063215faab85e34b79"
ANALYZED_BASE_TREE = "1baa66e15958575898e0ab364b133efaf31b4593"

MASTER_TARGET_MICROSECONDS = np.asarray(
    (5000, 5400, 6000, 7000, 8000, 9000, 9600, 9800, 10000), dtype=int
)
MASTER_TARGETS_SECONDS = MASTER_TARGET_MICROSECONDS.astype(float) * 1.0e-6
MAIN_TARGET_INDICES = np.arange(MASTER_TARGET_MICROSECONDS.size, dtype=int)
REPLAY_TARGET_INDICES = np.asarray((5, 6, 7, 8), dtype=int)
STRICT_TARGET_INDICES = np.asarray((6, 7, 8), dtype=int)
PILOT_TARGET_INDICES = np.asarray((0, 1), dtype=int)
MAIN_TARGETS_SECONDS = MASTER_TARGETS_SECONDS[MAIN_TARGET_INDICES]
REPLAY_TARGETS_SECONDS = MASTER_TARGETS_SECONDS[REPLAY_TARGET_INDICES]
STRICT_TARGETS_SECONDS = MASTER_TARGETS_SECONDS[STRICT_TARGET_INDICES]
PILOT_TARGETS_SECONDS = MASTER_TARGETS_SECONDS[PILOT_TARGET_INDICES]

ARTIFACT = "causal_inner_nonlinear_ten_ms_screen_manifest_wp10c9d6c7c3b5c4b1"
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_ten_ms_screen_manifest_"
    "wp10c9d6c7c3b5c4b1.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_ten_ms_screen_manifest_"
    "wp10c9d6c7c3b5c4b1.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_TEN_MS_SCREEN_"
    "MANIFEST_WP10C9D6C7C3B5C4B1_2026-08-08.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "ten_ms_screen_manifest.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _validate_parent() -> dict:
    parent = _read_json(c4b.SUMMARY_PATH)
    if (
        not parent["passed"]
        or not parent["ten_ms_screen_propagation_authorized"]
        or parent["authorized_next"] != f"{WORK_PACKAGE}_ten_ms_screen"
        or parent["physical_failure_detected"]
        or parent["twenty_ms_propagation_authorized"]
        or parent["fixed_q_micro_solver_authorized"]
        or parent["reduced_slow_evolution_authorized"]
    ):
        raise RuntimeError("c4b1 authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c4b1 analyzed identity changed")
    return parent


def _manifest() -> dict:
    parent_manifest = _read_json(c4a.MANIFEST_PATH)
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            "ten_ms_screen_manifest_hardened_all_common_targets_"
            "durable_propagation_authorized"
        ),
        "passed": True,
        "definitions_only": True,
        "propagation_executed": False,
        "operator_changed": False,
        "production_defaults_changed": False,
        "historical_c4a_target_table_preserved": True,
        "prospective_target_hardening": (
            "main_schedule_now_includes_9600_and_9800_microsecond_strict_targets"
        ),
        "canonical_targets": {
            "construction": "one_integer_microsecond_master_table",
            "master_microseconds": MASTER_TARGET_MICROSECONDS,
            "main_indices": MAIN_TARGET_INDICES,
            "replay_indices": REPLAY_TARGET_INDICES,
            "strict_indices": STRICT_TARGET_INDICES,
            "pilot_indices": PILOT_TARGET_INDICES,
            "main_seconds": MAIN_TARGETS_SECONDS,
            "replay_seconds": REPLAY_TARGETS_SECONDS,
            "strict_seconds": STRICT_TARGETS_SECONDS,
            "pilot_seconds": PILOT_TARGETS_SECONDS,
            "every_replay_and_strict_target_is_a_main_target": True,
            "all_common_strict_outputs_binding": True,
            "independent_target_construction_forbidden": True,
        },
        "pilot_seed": {
            "source": "committed_c4b_base_and_perturbed_5p0_to_5p4ms_steps",
            "states_and_extraction_histories_reused_by_hash": True,
            "complete_BDF2_history_reconstructed_deterministically_from_5p0_and_5p4ms_states": True,
            "history_reconstruction_uses_complete_mapped_and_responsive_height_storage_increment": True,
            "no_new_BDF1_startup": True,
            "main_propagation_start_seconds": 5.4e-3,
            "next_candidate_timestep_seconds": 4.0e-4,
        },
        "extraction_partition_contract": parent_manifest[
            "extraction_partition_contract"
        ],
        "main_controller": parent_manifest["main_controller"],
        "strict_controller": parent_manifest["strict_controller"],
        "execution": {
            "stage_order": (
                "base_main",
                "perturbed_main",
                "base_replay",
                "perturbed_replay",
                "base_strict",
                "perturbed_strict",
            ),
            "post_pilot_main_expected_comparisons_per_trajectory": 15,
            "replay_expected_comparisons_per_trajectory": 4,
            "strict_expected_comparisons_per_trajectory": 4,
            "post_pilot_total_expected_comparisons_per_trajectory": 23,
            "projected_wall_hours_with_1p5_safety_factor": (
                (23.0 / 22.0)
                * _read_json(c4b.SUMMARY_PATH)["runtime_projection"][
                    "projected_wall_hours"
                ]
            ),
            "durable_restart_and_arrays_after_every_target": True,
            "resume_requires_exact_runner_manifest_and_input_hashes": True,
            "stop_on_first_scientific_failure": True,
        },
        "binding_gates": parent_manifest["binding_gates"],
        "positive_branch": parent_manifest["positive_branch"],
        "negative_branch": parent_manifest["negative_branch"],
        "hard_stops": parent_manifest["hard_stops"],
        "authorized_next": "WP10c9d6c7c3b5c4b2_ten_ms_screen",
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
                    "sha256": _sha256(path),
                    "scientific_status": "CERTIFIED",
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
    catalog = _read_json(CANONICAL_SUMMARY)
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
            "latest_source_parent_commit": ANALYZED_BASE_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def main() -> int:
    parent = _validate_parent()
    manifest = _manifest()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": manifest["classification"],
        "passed": True,
        "definitions_only": True,
        "ten_ms_screen_propagation_authorized": True,
        "twenty_ms_completion_manifest_authorized": False,
        "twenty_ms_propagation_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": manifest["authorized_next"],
        "parent_classification_preserved": parent["classification"],
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "master_target_microseconds": MASTER_TARGET_MICROSECONDS,
            "main_target_indices": MAIN_TARGET_INDICES,
            "replay_target_indices": REPLAY_TARGET_INDICES,
            "strict_target_indices": STRICT_TARGET_INDICES,
            "pilot_target_indices": PILOT_TARGET_INDICES,
        },
    )
    _write_json(MANIFEST_PATH, manifest)
    _write_json(SUMMARY_PATH, summary)
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "CERTIFIED",
            "source_parent_commit": ANALYZED_BASE_COMMIT,
            "source_parent_tree_sha": ANALYZED_BASE_TREE,
            "working_head": _git_value("rev-parse", "HEAD"),
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "parent_summary_sha256": _sha256(c4b.SUMMARY_PATH),
            "pilot_arrays_sha256": _sha256(c4b.DECISIVE_ARRAYS),
            "implementation_source_hashes": {
                path: _sha256(ROOT / path)
                for path in (THIS_RUNNER, THIS_TEST)
                if (ROOT / path).exists()
            },
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "command": f"PYTHONPATH=src:scripts python {THIS_RUNNER}",
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Nonlinear 10 ms screen manifest WP10c9d6c7c3b5c4b1",
                "",
                "## Classification",
                "",
                f"`{manifest['classification']}`",
                "",
                "This prospective hardening package makes 9.6 and 9.8 ms literal main targets so every strict output is compared against an exact same-target main output. It does not amend the c4a target table.",
                "",
                "The committed 5.0-5.4 ms pilot steps seed the full screen. Their complete primitive, mapped, and responsive-height BDF2 histories are reconstructed deterministically from the committed endpoint states. Every later canonical target receives a durable restart and arrays cache.",
                "",
                "The corrected 23-comparison-per-trajectory projection remains below the 24-hour automatic-continuation tier. The extraction partition semantics and all scientific gates are unchanged.",
                "",
                "The 20 ms propagation, fixed-Q experiments, and reduced slow evolution remain blocked.",
                "",
            )
        ),
        encoding="utf-8",
    )
    names = ("config.json", "provenance.json", "summary.json", "ten_ms_screen_manifest.json")
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
