#!/usr/bin/env python3
"""Freeze the canonical-target replay audit after the c2d formal failure."""

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

import run_causal_inner_nonlinear_second_duration_rung_wp10c9d6c7c3b5c2 as c2  # noqa: E402
import run_causal_inner_nonlinear_second_rung_perturbed_completion_wp10c9d6c7c3b5c2d as c2d  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_canonical_json_sha256,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c2e"
ANALYZED_BASE_COMMIT = "43216e36c21e0554a545a00da3361651a8900df0"
ANALYZED_BASE_PARENT = "38f01107ede022526624a041137903d880e31e63"
ANALYZED_BASE_TREE = "5901a1fce58eec9ba0686a5c830d458ed78b0ef4"

START_SECONDS = 8.0e-4
STOP_SECONDS = 1.0e-3
CANONICAL_TARGETS = np.asarray(c2.OUTPUT_TIMES[8:], dtype=float)
LEGACY_TARGETS = np.asarray(c2.CONTINUATION_OUTPUT_TIMES[6:], dtype=float)
TRAJECTORIES = ("base", "perturbed")

ARTIFACT = "causal_inner_nonlinear_canonical_time_replay_manifest_wp10c9d6c7c3b5c2e"
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_canonical_time_replay_manifest_"
    "wp10c9d6c7c3b5c2e.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_canonical_time_replay_manifest_"
    "wp10c9d6c7c3b5c2e.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_NONLINEAR_CANONICAL_TIME_REPLAY_MANIFEST_"
    "WP10C9D6C7C3B5C2E_2026-08-04.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "canonical_time_replay_manifest.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
PARENT_DIRECTORY = c2d.CANONICAL_DIRECTORY


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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _validate_parent() -> dict:
    parent = _read_json(PARENT_DIRECTORY / "summary.json")
    if (
        parent["passed"]
        or parent["classification"]
        != "second_rung_perturbed_completion_failed_later_duration_blocked"
        or parent["strict_shadow_comparison"]["passed"] is not True
        or parent["third_duration_rung_manifest_authorized"] is not False
        or parent["fixed_q_micro_solver_authorized"] is not False
        or parent["reduced_slow_evolution_authorized"] is not False
    ):
        raise RuntimeError("c2e parent classification changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c2e analyzed identity changed")
    return parent


def _target_grid_diagnostic() -> dict:
    if CANONICAL_TARGETS.shape != LEGACY_TARGETS.shape:
        raise RuntimeError("canonical and legacy target shapes changed")
    delta = LEGACY_TARGETS - CANONICAL_TARGETS
    spacing = np.maximum(
        np.spacing(np.abs(CANONICAL_TARGETS)), np.finfo(float).tiny
    )
    spacing_units = np.abs(delta) / spacing
    differing = np.flatnonzero(delta != 0.0)
    if differing.tolist() != [1] or spacing_units[1] != 1.0:
        raise RuntimeError("expected one-ULP target-grid mismatch changed")
    return {
        "canonical_targets_seconds": CANONICAL_TARGETS,
        "legacy_targets_seconds": LEGACY_TARGETS,
        "canonical_target_hex": [float(value).hex() for value in CANONICAL_TARGETS],
        "legacy_target_hex": [float(value).hex() for value in LEGACY_TARGETS],
        "differing_indices": differing,
        "maximum_spacing_units": float(np.max(spacing_units)),
    }


def _manifest(target_grid: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            "canonical_time_replay_manifest_frozen_"
            "paired_target_grid_audit_authorized"
        ),
        "passed": True,
        "propagation_executed": False,
        "operator_changed": False,
        "production_defaults_changed": False,
        "historical_c2d_failure_preserved": True,
        "root_cause_hypothesis": (
            "main_and_replay_used_independently_generated_target_grids_"
            "whose_9e_minus_4_entries_differ_by_one_ULP"
        ),
        "target_grid_diagnostic": target_grid,
        "audit": {
            "start_seconds": START_SECONDS,
            "stop_seconds": STOP_SECONDS,
            "trajectories": TRAJECTORIES,
            "layout": c2.LAYOUT,
            "profile": c2.PROFILE,
            "coupling_face": c2.COUPLING_FACE,
            "reconstruct_start_from_committed_c2d_state_and_complete_BDF_history": True,
            "start_state_index": 8,
            "start_history_index": 1,
            "previous_timestep_seconds": 1.0e-4,
            "build_one_frozen_tangent_shared_by_all_branches": True,
            "branches_per_trajectory": (
                "legacy_target_direct",
                "canonical_target_direct",
                "canonical_target_serialized",
            ),
            "fixed_BDF2_step_uses_exact_next_target_minus_current_time": True,
            "accepted_steps_per_branch": 2,
            "no_error_estimator_or_controller_retuning": True,
        },
        "binding_gates": {
            "legacy_branch_reproduces_committed_c2d_main_state_export_and_history_bitwise": True,
            "canonical_direct_and_serialized_target_labels_bitwise": True,
            "canonical_direct_and_serialized_states_bitwise": True,
            "canonical_direct_and_serialized_Tier_I_exports_bitwise": True,
            "canonical_direct_and_serialized_complete_BDF_histories_bitwise": True,
            "canonical_direct_and_serialized_restart_payloads_bitwise": True,
            "legacy_and_canonical_first_difference_index": 1,
            "legacy_canonical_maximum_relative_state_norm_difference": 1.0e-12,
            "legacy_canonical_maximum_relative_export_norm_difference": 1.0e-12,
            "legacy_canonical_maximum_relative_history_norm_difference": 1.0e-9,
            "canonical_response_maximum_scaled_state_difference_from_legacy": 1.0e-8,
            "canonical_response_maximum_scaled_Tier_I_difference_from_legacy": 1.0e-8,
            "all_inherited_method_gates": True,
            "no_incoming_excision_characteristics": True,
        },
        "method_gates": _read_json(c2.CONFIG_PATH)["main_controller"][
            "step_method_gates"
        ],
        "positive_branch": {
            "classification": (
                "canonical_target_replay_bitwise_certified_"
                "third_rung_manifest_authorized"
            ),
            "authorized_next": "WP10c9d6c7c3b5c3a_third_duration_rung_manifest",
            "preserve_c2d_as_formal_failure": True,
            "use_only_canonical_target_slices_in_all_future_controllers": True,
            "commit_corrected_base_and_perturbed_BDF2_restart_payloads": True,
        },
        "negative_branch": {
            "classification": (
                "canonical_target_replay_audit_failed_later_duration_blocked"
            ),
            "authorized_next": "none",
            "no_later_duration_propagation": True,
        },
        "hard_stops": [
            "do not amend or relabel the c2d formal failure",
            "do not relax same-target same-tangent bitwise replay gates",
            "do not change the physical or numerical operator",
            "do not begin the third duration rung before the audit passes",
            "do not begin fixed-Q or reduced slow evolution",
            "do not add tide, wind, hot-state, S-curve or QPE-cycle physics",
            "do not use N1024 as a rescue",
        ],
        "authorized_next": "WP10c9d6c7c3b5c2e1_canonical_time_replay_audit",
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
    target_grid = _target_grid_diagnostic()
    manifest = _manifest(target_grid)
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "start_seconds": START_SECONDS,
        "stop_seconds": STOP_SECONDS,
        "canonical_targets_seconds": CANONICAL_TARGETS,
        "legacy_targets_seconds": LEGACY_TARGETS,
        "trajectories": TRAJECTORIES,
        "propagation_executed": False,
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": manifest["classification"],
        "passed": True,
        "propagation_executed": False,
        "operator_changed": False,
        "historical_c2d_classification_preserved": parent["classification"],
        "target_grid_diagnostic": target_grid,
        "canonical_time_replay_audit_authorized": True,
        "third_duration_rung_manifest_authorized": False,
        "third_duration_rung_propagation_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": manifest["authorized_next"],
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(CONFIG_PATH, config)
    _write_json(MANIFEST_PATH, manifest)
    _write_json(SUMMARY_PATH, summary)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "analyzed_base_parent": ANALYZED_BASE_PARENT,
        "analyzed_base_tree": ANALYZED_BASE_TREE,
        "implementation_commit": _git_value("rev-parse", "HEAD"),
        "implementation_tree_before_manifest_commit": _git_value(
            "rev-parse", "HEAD^{tree}"
        ),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "runner": THIS_RUNNER,
        "runner_sha256": _sha256(ROOT / THIS_RUNNER),
        "test": THIS_TEST,
        "input_summary": str((PARENT_DIRECTORY / "summary.json").relative_to(ROOT)),
        "input_summary_sha256": _sha256(PARENT_DIRECTORY / "summary.json"),
        "input_decisive_arrays": str(
            (PARENT_DIRECTORY / "decisive_arrays.npz").relative_to(ROOT)
        ),
        "input_decisive_arrays_sha256": _sha256(
            PARENT_DIRECTORY / "decisive_arrays.npz"
        ),
        "config_sha256": causal_canonical_json_sha256(_plain(config)),
        "manifest_sha256": causal_canonical_json_sha256(_plain(manifest)),
    }
    _write_json(PROVENANCE_PATH, provenance)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# Canonical-time replay manifest WP10c9d6c7c3b5c2e\n\n"
        "## Classification\n\n"
        f"`{manifest['classification']}`\n\n"
        "This definitions-only package preserves the c2d failure and freezes "
        "a paired legacy-target/canonical-target replay audit. The separately "
        "generated target grids differ by exactly one ULP at `9e-4 s`.\n\n"
        f"Authorized next: `{manifest['authorized_next']}`.\n\n"
        "The third duration rung, fixed-Q experiments, and reduced slow "
        "evolution remain blocked.\n",
        encoding="utf-8",
    )
    checksum_paths = (CONFIG_PATH, MANIFEST_PATH, PROVENANCE_PATH, SUMMARY_PATH)
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in checksum_paths),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
