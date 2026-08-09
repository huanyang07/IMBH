#!/usr/bin/env python3
"""Freeze the durable nonlinear continuation from 10 to 20 ms."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_nonlinear_ten_ms_screen_manifest_wp10c9d6c7c3b5c4b1 as c4b1  # noqa: E402
import run_causal_inner_nonlinear_ten_ms_screen_wp10c9d6c7c3b5c4b2 as c4b2  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4c"
ANALYZED_BASE_COMMIT = "14d5c829f54999aecedbe93ae747a7ce32d58bc0"
ANALYZED_BASE_PARENT = "fc8889fe7a00e734c272be416bc7decbd6970e2f"
ANALYZED_BASE_TREE = "642ac2f66859238063c98cc4d9c05d1920f48670"

START_MICROSECONDS = 10_000
HORIZON_MICROSECONDS = 20_000
MASTER_TARGET_MICROSECONDS = np.asarray(
    (10_000, 12_000, 14_000, 16_000, 18_000, 18_800, 19_600, 19_800, 20_000),
    dtype=int,
)
MASTER_TARGETS_SECONDS = MASTER_TARGET_MICROSECONDS.astype(float) * 1.0e-6
MAIN_TARGET_INDICES = np.arange(MASTER_TARGET_MICROSECONDS.size, dtype=int)
REPLAY_TARGET_INDICES = np.asarray((5, 6, 7, 8), dtype=int)
STRICT_TARGET_INDICES = np.asarray((6, 7, 8), dtype=int)
MAIN_TARGETS_SECONDS = MASTER_TARGETS_SECONDS[MAIN_TARGET_INDICES]
REPLAY_TARGETS_SECONDS = MASTER_TARGETS_SECONDS[REPLAY_TARGET_INDICES]
STRICT_TARGETS_SECONDS = MASTER_TARGETS_SECONDS[STRICT_TARGET_INDICES]

EXPECTED_MAIN_COMPARISONS = 26
EXPECTED_REPLAY_COMPARISONS = 4
EXPECTED_STRICT_COMPARISONS = 4
RESOURCE_SAFETY_FACTOR = 1.25

ARTIFACT = "causal_inner_nonlinear_twenty_ms_completion_manifest_wp10c9d6c7c3b5c4c"
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_twenty_ms_completion_manifest_"
    "wp10c9d6c7c3b5c4c.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_twenty_ms_completion_manifest_"
    "wp10c9d6c7c3b5c4c.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_TWENTY_MS_COMPLETION_"
    "MANIFEST_WP10C9D6C7C3B5C4C_2026-08-09.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "twenty_ms_completion_manifest.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
BASE_RESTART_PATH = CANONICAL_DIRECTORY / "base_restart_10000us.npz"
PERTURBED_RESTART_PATH = CANONICAL_DIRECTORY / "perturbed_restart_10000us.npz"
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


def _stage_seed(name: str) -> tuple[Path, dict]:
    stage = f"{name}_main"
    progress_path = c4b2.PROGRESS_DIRECTORY / stage / "progress.json"
    restart_path = c4b2.PROGRESS_DIRECTORY / stage / "restart_10000us.npz"
    progress = _read_json(progress_path)
    if (
        progress.get("stage") != stage
        or not progress.get("complete")
        or int(progress.get("current_target_microseconds", -1))
        != START_MICROSECONDS
        or progress.get("restart_sha256") != _sha256(restart_path)
    ):
        raise RuntimeError(f"{stage} 10 ms restart is not certified")
    return restart_path, progress


def _validate_parent() -> tuple[dict, dict[str, tuple[Path, dict]]]:
    parent = _read_json(c4b2.SUMMARY_PATH)
    if (
        not parent["passed"]
        or not parent["ten_ms_screen_certified"]
        or not parent["twenty_ms_completion_manifest_authorized"]
        or parent["twenty_ms_propagation_authorized"]
        or parent["physical_failure_detected"]
        or parent["pointwise_horizon_flux_convergence_claimed"]
        or not parent["raw_inner_face_rejection_preserved"]
        or parent["fixed_q_micro_solver_authorized"]
        or parent["reduced_slow_evolution_authorized"]
    ):
        raise RuntimeError("c4c authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c4c analyzed identity changed")
    return parent, {name: _stage_seed(name) for name in ("base", "perturbed")}


def _runtime_projection(parent: dict) -> dict:
    reports = parent["stage_reports"]
    projected = 0.0
    comparisons = {
        "main": EXPECTED_MAIN_COMPARISONS,
        "replay": EXPECTED_REPLAY_COMPARISONS,
        "strict": EXPECTED_STRICT_COMPARISONS,
    }
    mean_seconds = {}
    for name in ("base", "perturbed"):
        for kind, count in comparisons.items():
            key = f"{name}_{kind}"
            report = reports[key]
            mean = report["measured_wall_seconds"] / report["accepted_comparisons"]
            mean_seconds[key] = mean
            projected += count * mean
    return {
        "method": "measured_10ms_stage_seconds_per_accepted_comparison",
        "expected_comparisons": comparisons,
        "mean_seconds_per_comparison": mean_seconds,
        "raw_projected_wall_seconds": projected,
        "raw_projected_wall_hours": projected / 3600.0,
        "safety_factor": RESOURCE_SAFETY_FACTOR,
        "projected_wall_seconds_with_safety": projected * RESOURCE_SAFETY_FACTOR,
        "projected_wall_hours_with_safety": projected
        * RESOURCE_SAFETY_FACTOR
        / 3600.0,
        "runtime_is_advisory_not_scientific": True,
    }


def _manifest(parent: dict) -> dict:
    inherited = _read_json(c4b1.MANIFEST_PATH)
    projection = _runtime_projection(parent)
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            "twenty_ms_completion_manifest_frozen_durable_propagation_authorized"
        ),
        "passed": True,
        "definitions_only": True,
        "propagation_executed": False,
        "operator_changed": False,
        "production_defaults_changed": False,
        "rung": {
            "start_seconds": START_MICROSECONDS * 1.0e-6,
            "horizon_seconds": HORIZON_MICROSECONDS * 1.0e-6,
            "fraction_of_stress_relaxation_time": 0.020 / 0.147,
            "approximate_N128_cell_crossing_times": 0.020 / 5.54e-3,
            "layout": "N128-exterior__N128-equivalent-inner",
            "profile": "generic_five_field",
            "trajectories": ("base", "perturbed"),
            "no_new_BDF1_startup": True,
        },
        "canonical_targets": {
            "construction": "one_integer_microsecond_master_table",
            "master_microseconds": MASTER_TARGET_MICROSECONDS,
            "main_indices": MAIN_TARGET_INDICES,
            "replay_indices": REPLAY_TARGET_INDICES,
            "strict_indices": STRICT_TARGET_INDICES,
            "main_seconds": MAIN_TARGETS_SECONDS,
            "replay_seconds": REPLAY_TARGETS_SECONDS,
            "strict_seconds": STRICT_TARGETS_SECONDS,
            "every_replay_and_strict_target_is_a_main_target": True,
            "all_common_strict_outputs_binding": True,
            "independent_target_construction_forbidden": True,
        },
        "initial_restarts": {
            "source": "certified_c4b2_complete_BDF2_restarts_at_10ms",
            "base_canonical_path": str(BASE_RESTART_PATH.relative_to(ROOT)),
            "perturbed_canonical_path": str(
                PERTURBED_RESTART_PATH.relative_to(ROOT)
            ),
            "complete_primitive_mapped_and_responsive_height_histories": True,
            "candidate_timestep_seconds": 4.0e-4,
            "copied_into_canonical_evidence": True,
        },
        "extraction_partition_contract": inherited["extraction_partition_contract"],
        "main_controller": inherited["main_controller"],
        "strict_controller": inherited["strict_controller"],
        "execution": {
            "stage_order": (
                "base_main",
                "perturbed_main",
                "base_replay",
                "perturbed_replay",
                "base_strict",
                "perturbed_strict",
            ),
            "main_expected_comparisons_per_trajectory": EXPECTED_MAIN_COMPARISONS,
            "replay_expected_comparisons_per_trajectory": EXPECTED_REPLAY_COMPARISONS,
            "strict_expected_comparisons_per_trajectory": EXPECTED_STRICT_COMPARISONS,
            "durable_restart_and_arrays_after_every_target": True,
            "resume_requires_exact_runner_manifest_and_input_hashes": True,
            "stop_on_first_scientific_failure": True,
            "base_before_perturbed": True,
        },
        "runtime_projection": projection,
        "resource_policy": {
            "projected_hours_at_or_below_24": "continue_automatically",
            "projected_hours_24_to_36": (
                "continue_with_durable_checkpoints_after_cost_review"
            ),
            "projected_hours_above_36": "stop_and_optimize_before_propagation",
            "runtime_is_not_a_physical_gate": True,
        },
        "binding_gates": inherited["binding_gates"],
        "positive_branch": {
            "classification": (
                "twenty_ms_completion_certified_checkpoint_assessment_manifest_authorized"
            ),
            "authorized_next": (
                "WP10c9d6c7c3b5c4d_twenty_ms_checkpoint_assessment_manifest"
            ),
            "fifty_ms_propagation_requires_fresh_manifest": True,
        },
        "negative_branch": {
            "classification": "twenty_ms_completion_failed_later_duration_blocked",
            "authorized_next": "failure_localization_only",
            "runtime_projection_alone_is_not_scientific_failure": True,
        },
        "hard_stops": (
            "do_not_use_raw_inner_face_flux_as_the_slow_export",
            "do_not_relabel_extraction_flux_as_pointwise_horizon_flux",
            "do_not_change_operator_profile_or_production_defaults",
            "do_not_run_fifty_ms_before_a_fresh_manifest",
            "do_not_start_fixed_Q_or_reduced_slow_evolution",
            "do_not_add_tide_wind_hot_state_S_curve_or_QPE_cycle_physics",
            "do_not_use_N1024_as_a_rescue",
        ),
        "authorized_next": "WP10c9d6c7c3b5c4c1_twenty_ms_completion",
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
    parent, seeds = _validate_parent()
    manifest = _manifest(parent)
    projection = manifest["runtime_projection"]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": manifest["classification"],
        "passed": True,
        "definitions_only": True,
        "parent_classification_preserved": parent["classification"],
        "ten_ms_screen_certified": True,
        "twenty_ms_completion_manifest_authorized": True,
        "twenty_ms_propagation_authorized": True,
        "twenty_ms_checkpoint_assessment_authorized": False,
        "fifty_ms_propagation_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "runtime_projection": projection,
        "authorized_next": manifest["authorized_next"],
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    shutil.copy2(seeds["base"][0], BASE_RESTART_PATH)
    shutil.copy2(seeds["perturbed"][0], PERTURBED_RESTART_PATH)
    _write_json(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "start_microseconds": START_MICROSECONDS,
            "horizon_microseconds": HORIZON_MICROSECONDS,
            "master_target_microseconds": MASTER_TARGET_MICROSECONDS,
            "main_target_indices": MAIN_TARGET_INDICES,
            "replay_target_indices": REPLAY_TARGET_INDICES,
            "strict_target_indices": STRICT_TARGET_INDICES,
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
            "parent_summary_sha256": _sha256(c4b2.SUMMARY_PATH),
            "parent_decisive_arrays_sha256": _sha256(c4b2.DECISIVE_ARRAYS),
            "seed_restart_sha256": {
                name: _sha256(seed[0]) for name, seed in seeds.items()
            },
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
                "# Nonlinear 20 ms completion manifest WP10c9d6c7c3b5c4c",
                "",
                "## Classification",
                "",
                f"`{manifest['classification']}`",
                "",
                "The certified base and generic-five-field perturbed BDF2 restarts at 10 ms are promoted into compact canonical evidence and seed a direct, durable continuation to 20 ms. No BDF1 restart is introduced.",
                "",
                "One integer-microsecond table supplies every main, replay, and strict target. Main propagation checkpoints at 12, 14, 16, 18, 18.8, 19.6, 19.8, and 20 ms; replay begins at 18.8 ms and the strict shadow begins at 19.6 ms.",
                "",
                f"Measured-stage projection: `{projection['raw_projected_wall_hours']:.3f} h` raw and `{projection['projected_wall_hours_with_safety']:.3f} h` with the frozen 1.25 safety factor. Runtime is advisory, not a physical gate.",
                "",
                "The certified exterior-domain extraction partition at `R=1.9531594414758637 r_g` remains the binding slow export. The raw pointwise horizon-flux rejection is preserved.",
                "",
                "A pass authorizes only a definitions-only 20 ms checkpoint assessment. Fifty-millisecond propagation, fixed-Q experiments, and reduced slow evolution remain blocked.",
                "",
            )
        ),
        encoding="utf-8",
    )
    names = (
        "base_restart_10000us.npz",
        "config.json",
        "perturbed_restart_10000us.npz",
        "provenance.json",
        "summary.json",
        "twenty_ms_completion_manifest.json",
    )
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
