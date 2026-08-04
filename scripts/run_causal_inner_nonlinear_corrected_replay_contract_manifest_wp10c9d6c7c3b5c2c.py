#!/usr/bin/env python3
"""Freeze a corrected, separately instrumented second-rung replay contract."""

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

import run_causal_inner_nonlinear_second_duration_replay_localization_wp10c9d6c7c3b5c2b as c2b  # noqa: E402
import run_causal_inner_nonlinear_second_duration_rung_wp10c9d6c7c3b5c2 as c2  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_canonical_json_sha256,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c2c"
ANALYZED_BASE_COMMIT = "d282f4a43f78b045d0911923fd1ca82aa1211a85"
ANALYZED_BASE_PARENT = "acaeacf18a71509a711e85ab3181dff50380aa5f"
ANALYZED_BASE_TREE = "59d419b64ee9ea4abacae23cd0de7374013c391f"

REPLAY_START_SECONDS = 6.0e-4
REPLAY_STOP_SECONDS = 1.0e-3
REPLAY_TIMESTEP_SECONDS = 1.0e-4
REPLAY_TARGETS_SECONDS = np.linspace(REPLAY_START_SECONDS, REPLAY_STOP_SECONDS, 5)
LAYOUT = c2.LAYOUT
PROFILE = c2.PROFILE
COUPLING_FACE = c2.COUPLING_FACE

ARTIFACT = (
    "causal_inner_nonlinear_corrected_replay_contract_manifest_"
    "wp10c9d6c7c3b5c2c"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_corrected_replay_contract_manifest_"
    "wp10c9d6c7c3b5c2c.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_corrected_replay_contract_manifest_"
    "wp10c9d6c7c3b5c2c.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_NONLINEAR_CORRECTED_REPLAY_CONTRACT_MANIFEST_"
    "WP10C9D6C7C3B5C2C_2026-08-02.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "corrected_replay_contract_manifest.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
PARENT_DIRECTORY = c2b.CANONICAL_DIRECTORY


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


def _validate_parent() -> tuple[dict, dict]:
    parent = _read_json(PARENT_DIRECTORY / "summary.json")
    historical = _read_json(c2.CANONICAL_DIRECTORY / "summary.json")
    if (
        not parent["passed"]
        or not parent["corrected_replay_manifest_authorized"]
        or parent["authorized_next"]
        != "WP10c9d6c7c3b5c2c_corrected_replay_contract_manifest"
        or not parent["historical_replay_failure_preserved"]
        or historical["classification"]
        != "second_nonlinear_duration_rung_failed_later_duration_work_blocked"
    ):
        raise RuntimeError("c2c replay-manifest authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c2c analyzed identity changed")
    return parent, historical


def _manifest() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            "corrected_replay_contract_manifest_frozen_"
            "paired_base_replay_validation_authorized"
        ),
        "passed": True,
        "propagation_executed": False,
        "operator_changed": False,
        "production_defaults_changed": False,
        "historical_c2_failure_preserved": True,
        "layout": LAYOUT,
        "profile": PROFILE,
        "coupling_face": COUPLING_FACE,
        "paired_base_replay": {
            "start_seconds": REPLAY_START_SECONDS,
            "stop_seconds": REPLAY_STOP_SECONDS,
            "fixed_BDF2_timestep_seconds": REPLAY_TIMESTEP_SECONDS,
            "canonical_output_targets_seconds": REPLAY_TARGETS_SECONDS,
            "source_history_times_seconds": [5.0e-4, 6.0e-4],
            "reconstruct_history_from_committed_base_states": True,
            "build_one_frozen_tangent_shared_by_both_branches": True,
            "direct_branch_starts_from_in_memory_restart": True,
            "serialized_branch_starts_from_save_load_of_same_restart": True,
            "run_direct_and_serialized_branches_independently": True,
            "accepted_step_count_per_branch": 4,
        },
        "separate_replay_gates": {
            "canonical_time_labels_bitwise": True,
            "accumulated_elapsed_time_within_ULP": 1.0,
            "primitive_states_bitwise": True,
            "direct_Tier_I_exports_bitwise": True,
            "complete_BDF_history_bitwise": True,
            "restart_roundtrip_bitwise": True,
            "no_combined_short_circuit_boolean": True,
        },
        "method_gates": _read_json(c2.CONFIG_PATH)["main_controller"][
            "step_method_gates"
        ],
        "committed_main_comparison": {
            "scientific_role": "explanatory_fresh_process_check_only",
            "maximum_scaled_state_difference": 1.0e-12,
            "maximum_scaled_export_difference": 1.0e-12,
            "not_a_substitute_for_paired_bitwise_replay": True,
        },
        "positive_branch": {
            "authorized_next": (
                "WP10c9d6c7c3b5c2d_second_rung_perturbed_completion"
            ),
            "reuse_committed_base_main_and_strict_arrays_by_hash": True,
            "run_only_missing_perturbed_main_replay_and_strict_trajectory": True,
            "instrument_time_state_export_replay_separately": True,
            "require_original_response_and_strict_shadow_gates": True,
        },
        "negative_branch": {
            "authorized_next": (
                "WP10c9d6c7c3b5c2c2_paired_replay_localization"
            ),
            "no_perturbed_or_later_duration_propagation": True,
        },
        "hard_stops": [
            "do not amend or relabel the historical c2 failure",
            "do not replace paired same-tangent bitwise gates with tolerances",
            "do not run the missing perturbed trajectory before paired replay passes",
            "do not begin the third duration rung",
            "do not redesign the spatial operator or interface",
            "do not begin fixed-Q or reduced slow evolution",
            "do not add tide, wind, hot-state, S-curve or QPE-cycle physics",
            "do not use N1024 as a rescue",
        ],
        "authorized_next": (
            "WP10c9d6c7c3b5c2c1_paired_base_replay_validation"
        ),
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
    parent, historical = _validate_parent()
    manifest = _manifest()
    if (
        not np.array_equal(
            np.asarray(manifest["paired_base_replay"]["canonical_output_targets_seconds"]),
            REPLAY_TARGETS_SECONDS,
        )
        or manifest["separate_replay_gates"]["no_combined_short_circuit_boolean"]
        is not True
        or manifest["positive_branch"]["run_only_missing_perturbed_main_replay_and_strict_trajectory"]
        is not True
    ):
        raise RuntimeError("c2c corrected replay contract is inconsistent")
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "layout": LAYOUT,
        "profile": PROFILE,
        "coupling_face": COUPLING_FACE,
        "replay_targets_seconds": REPLAY_TARGETS_SECONDS,
    }
    _write_json(CONFIG_PATH, config)
    _write_json(MANIFEST_PATH, manifest)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": manifest["classification"],
        "passed": True,
        "authorized_next": manifest["authorized_next"],
        "parent_classification_preserved": parent["classification"],
        "historical_classification_preserved": historical["classification"],
        "propagation_executed": False,
        "operator_changed": False,
        "production_defaults_changed": False,
        "paired_base_replay_validation_authorized": True,
        "perturbed_second_rung_authorized": False,
        "later_duration_rungs_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "config_sha256": causal_canonical_json_sha256(_plain(config)),
        "manifest_sha256": causal_canonical_json_sha256(_plain(manifest)),
    }
    _write_json(SUMMARY_PATH, summary)
    inputs = {
        "parent_summary": PARENT_DIRECTORY / "summary.json",
        "parent_arrays": PARENT_DIRECTORY / "decisive_arrays.npz",
        "historical_summary": c2.CANONICAL_DIRECTORY / "summary.json",
        "historical_arrays": c2.CANONICAL_DIRECTORY / "decisive_arrays.npz",
    }
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "CERTIFIED",
            "command": f"PYTHONPATH=src:scripts python3 {THIS_RUNNER}",
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "analyzed_base_parent_commit": ANALYZED_BASE_PARENT,
            "analyzed_base_tree_sha": ANALYZED_BASE_TREE,
            "implementation_parent_commit": _git_value("rev-parse", "HEAD"),
            "implementation_parent_tree_sha": _git_value("rev-parse", "HEAD^{tree}"),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "implementation_source_hashes": {
                path: _sha256(ROOT / path)
                for path in (THIS_RUNNER, THIS_TEST, c2b.THIS_RUNNER, c2.THIS_RUNNER)
                if (ROOT / path).exists()
            },
            "input_hashes": {name: _sha256(path) for name, path in inputs.items()},
        },
    )
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Corrected replay-contract manifest WP10c9d6c7c3b5c2c",
                "",
                "## Classification",
                "",
                f"`{manifest['classification']}`",
                "",
                "This definitions-only package preserves the historical c2 failure and",
                "freezes a same-tangent direct-versus-serialized replay through `1e-3 s`.",
                "Time labels, primitive states, complete BDF history, and Tier-I exports",
                "are reported separately; no short-circuit combined Boolean is allowed.",
                "",
                "A pass authorizes only the missing perturbed second-rung completion.",
                "The third rung, fixed-Q experiments, and reduced evolution remain blocked.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    names = (
        "config.json",
        "corrected_replay_contract_manifest.json",
        "summary.json",
        "provenance.json",
    )
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names)
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
