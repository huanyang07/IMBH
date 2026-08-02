#!/usr/bin/env python3
"""Run the frozen coarse-layout nonlinear profile-breadth method screen."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_causal_inner_monolithic_bdf_base_preflight_wp10c9d6c7c3b1a as c3b1a  # noqa: E402
import run_causal_inner_nonlinear_profile_breadth_controller_manifest_wp10c9d6c7c3b4a as c3b4a  # noqa: E402
import run_causal_inner_nonlinear_temporal_coarse_screen_wp10c9d6c7c3b3b1 as c3b3b1  # noqa: E402

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    causal_five_field_monolithic_frozen_tangent,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_canonical_json_sha256,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b4b1"
ANALYZED_BASE_COMMIT = "0e3a6feca400b386d9fdb47754aa4b58460cfea3"
ANALYZED_BASE_PARENT = "1e5fcc3900b2d3bd44f792ee768291dab5459f03"
ANALYZED_BASE_TREE = "731924cf0125a35ab00940a7a2ee03a01e12944e"

LAYOUT = c3b4a.LAYOUTS[0]
PROFILES = tuple(c3b4a.PROFILE_NAMES)
TIMESTEP_SECONDS = 1.0e-5
HORIZON_SECONDS = c3b4a.HORIZON_SECONDS
PROFILE_KIND = "primary_physical"

ARTIFACT = (
    "causal_inner_nonlinear_profile_breadth_coarse_screen_"
    "wp10c9d6c7c3b4b1"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_profile_breadth_coarse_screen_"
    "wp10c9d6c7c3b4b1.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_profile_breadth_coarse_screen_"
    "wp10c9d6c7c3b4b1.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_NONLINEAR_PROFILE_BREADTH_COARSE_SCREEN_"
    "WP10C9D6C7C3B4B1_2026-08-02.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE

PARENT_DIRECTORY = c3b4a.CANONICAL_DIRECTORY
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
CHECKPOINT_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
CHECKPOINT_JSON = CHECKPOINT_DIRECTORY / "progress.json"
CHECKPOINT_ARRAYS = CHECKPOINT_DIRECTORY / "progress_arrays.npz"


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
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


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


def _source_identity() -> dict[str, str]:
    paths = (THIS_RUNNER, THIS_TEST, c3b4a.THIS_RUNNER, c3b3b1.THIS_RUNNER)
    return {
        path: _sha256(ROOT / path)
        for path in paths
        if (ROOT / path).exists()
    }


def _validate_parent() -> tuple[dict, dict]:
    summary = _read_json(PARENT_DIRECTORY / "summary.json")
    manifest = _read_json(c3b4a.MANIFEST_PATH)
    if (
        not summary["passed"]
        or summary["authorized_next"]
        != "WP10c9d6c7c3b4b1_coarse_short_horizon_nonlinear_profile_breadth_screen"
        or manifest["classification"]
        != "short_horizon_nonlinear_profile_breadth_and_controller_manifest_frozen_coarse_breadth_screen_authorized"
        or manifest["propagation_executed"]
        or manifest["interpretation_limits"]["long_nonlinear_physical_ladder_authorized"]
        or manifest["interpretation_limits"]["fixed_q_micro_solver_authorized"]
        or manifest["interpretation_limits"]["reduced_slow_evolution_authorized"]
    ):
        raise RuntimeError("b4a coarse profile-breadth authorization changed")
    stage = manifest["campaign_controller"]["stages"][0]
    if (
        stage["work_package"] != WORK_PACKAGE
        or stage["layout"] != LAYOUT
        or stage["timestep_seconds"] != TIMESTEP_SECONDS
        or tuple(stage["profiles"]) != PROFILES
        or not stage["stop_on_any_method_or_readiness_failure"]
    ):
        raise RuntimeError("b4a coarse stage changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("b4b1 analyzed identity changed")
    return summary, manifest


def _task_id(profile: str) -> str:
    return f"{profile}__p1__dt_1e-5"


def _load_progress() -> tuple[dict, dict[str, np.ndarray]]:
    if not CHECKPOINT_JSON.exists():
        return (
            {
                "schema_version": SCHEMA_VERSION,
                "work_package": WORK_PACKAGE,
                "analyzed_base_commit": ANALYZED_BASE_COMMIT,
                "source_identity": _source_identity(),
                "completed_tasks": [],
                "trajectory_reports": {},
                "export_audits": {},
                "failed": False,
            },
            {},
        )
    progress = _read_json(CHECKPOINT_JSON)
    if (
        progress.get("work_package") != WORK_PACKAGE
        or progress.get("analyzed_base_commit") != ANALYZED_BASE_COMMIT
        or progress.get("source_identity") != _source_identity()
    ):
        raise RuntimeError("saved b4b1 progress belongs to different code")
    return progress, _load_npz(CHECKPOINT_ARRAYS)


def _save_progress(progress: dict, arrays: dict[str, np.ndarray]) -> None:
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(CHECKPOINT_JSON, progress)
    np.savez_compressed(CHECKPOINT_ARRAYS, **arrays)


def _run_screen(configuration: dict, tangent) -> tuple[dict, dict[str, np.ndarray]]:
    progress, arrays = _load_progress()
    completed = set(progress["completed_tasks"])
    packets = _load_npz(c3b4a.DECISIVE_ARRAYS)

    original_work_package = c3b3b1.WORK_PACKAGE
    c3b3b1.WORK_PACKAGE = WORK_PACKAGE
    try:
        for profile in PROFILES:
            task = _task_id(profile)
            if task in completed:
                continue
            packet = np.asarray(
                packets[f"{profile}__{LAYOUT}__{PROFILE_KIND}"], dtype=float
            )
            initial_state = np.asarray(configuration["base"], dtype=float) + packet
            report, trajectory = c3b3b1._trajectory(
                configuration,
                tangent,
                initial_state,
                TIMESTEP_SECONDS,
                f"{profile}__p1",
            )
            for name, values in trajectory.items():
                arrays[f"{task}__{name}"] = np.asarray(values)
            if report["passed"]:
                exports, audit = c3b3b1._export_history(
                    configuration["context"], trajectory["states"]
                )
                arrays[f"{task}__direct_exports"] = exports
                progress["export_audits"][task] = audit
                report["export_audit_passed"] = audit["passed"]
                report["passed"] = bool(report["passed"] and audit["passed"])
            else:
                report["export_audit_passed"] = False
            progress["trajectory_reports"][task] = report
            completed.add(task)
            progress["completed_tasks"] = sorted(completed)
            progress["failed"] = bool(progress["failed"] or not report["passed"])
            _save_progress(progress, arrays)
            if not report["passed"]:
                break
    finally:
        c3b3b1.WORK_PACKAGE = original_work_package
    return progress, arrays


def _method_screen(progress: dict) -> dict:
    reports = list(progress["trajectory_reports"].values())
    expected = len(PROFILES)
    passed = bool(
        not progress["failed"]
        and len(reports) == expected
        and all(report["passed"] for report in reports)
    )
    return {
        "passed": passed,
        "profile_count": expected,
        "completed_profile_count": len(reports),
        "all_four_step_trajectories_completed": all(
            report["completed_steps"] == 4 for report in reports
        ),
        "all_export_audits_passed": all(
            report["export_audit_passed"] for report in reports
        ),
        "maximum_scaled_residual": max(
            (report["maximum_scaled_residual"] for report in reports),
            default=None,
        ),
        "maximum_scaled_algebraic_residual": max(
            (report["maximum_scaled_algebraic_residual"] for report in reports),
            default=None,
        ),
        "maximum_discrete_ledger_defect": max(
            (report["maximum_discrete_ledger_defect"] for report in reports),
            default=None,
        ),
        "maximum_mapped_endpoint_path_closure_defect": max(
            (
                report["maximum_mapped_endpoint_path_closure_defect"]
                for report in reports
            ),
            default=None,
        ),
        "minimum_path_reconstruction_factor": min(
            (report["minimum_path_reconstruction_factor"] for report in reports),
            default=None,
        ),
        "maximum_incoming_excision_characteristics": max(
            (
                report["maximum_incoming_excision_characteristics"]
                for report in reports
            ),
            default=None,
        ),
        "all_checkpoint_roundtrips_bitwise": all(
            report["checkpoint_roundtrip_bitwise"] for report in reports
        ),
        "all_split_restart_replays_bitwise": all(
            report["split_restart_replay_bitwise"] for report in reports
        ),
        "trajectory_reports": progress["trajectory_reports"],
        "export_audits": progress["export_audits"],
    }


def _update_catalog(summary: dict) -> None:
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    status = "CERTIFIED" if summary["passed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha256(path),
                    "scientific_status": status,
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


def _report(summary: dict) -> str:
    screen = summary["method_screen"]
    lines = [
        "# Nonlinear profile-breadth coarse screen WP10c9d6c7c3b4b1",
        "",
        "## Classification",
        "",
        f"`{summary['classification']}`",
        "",
        "The five prospectively frozen acoustic, material, mixed and generic "
        "held-outs were evolved with the unchanged coarse embedded operator "
        "for four nonlinear steps at `dt=1e-5 s`.",
        "",
        "## Method results",
        "",
        f"- completed profiles: `{screen['completed_profile_count']}/5`",
        f"- maximum scaled nonlinear residual: `{screen['maximum_scaled_residual']:.3e}`",
        f"- maximum algebraic residual: `{screen['maximum_scaled_algebraic_residual']:.3e}`",
        f"- maximum discrete ledger defect: `{screen['maximum_discrete_ledger_defect']:.3e}`",
        "- maximum mapped endpoint/path closure defect: "
        f"`{screen['maximum_mapped_endpoint_path_closure_defect']:.3e}`",
        "- minimum path reconstruction factor: "
        f"`{screen['minimum_path_reconstruction_factor']:.12f}`",
        "- maximum incoming excision characteristics: "
        f"`{screen['maximum_incoming_excision_characteristics']}`",
        "- all checkpoint roundtrips bitwise: "
        f"`{screen['all_checkpoint_roundtrips_bitwise']}`",
        "- all split/restart replays bitwise: "
        f"`{screen['all_split_restart_replays_bitwise']}`",
        "",
    ]
    for profile in PROFILES:
        item = screen["trajectory_reports"][_task_id(profile)]
        lines.extend(
            [
                f"- `{profile}`: `{'pass' if item['passed'] else 'fail'}`, "
                f"residual `{item['maximum_scaled_residual']:.3e}`",
            ]
        )
    lines.extend(
        [
            "",
            "This package is a fail-fast solver and physical-ledger screen. "
            "It does not certify temporal convergence, spatial convergence, "
            "meaningful nonlinearity, or a longer physical horizon.",
            "",
            "## Authorized next",
            "",
            f"`{summary['authorized_next']}`",
            "",
            "Duration extension, fixed-Q experiments and reduced slow "
            "evolution remain blocked.",
            "",
        ]
    )
    return "\n".join(lines)


def _package(parent: dict, manifest: dict, progress: dict, arrays: dict) -> int:
    screen = _method_screen(progress)
    passed = screen["passed"]
    classification = (
        "coarse_heldout_profile_method_screen_certified_coarse_temporal_refinement_authorized"
        if passed
        else "coarse_heldout_profile_method_screen_failed_profile_breadth_campaign_blocked"
    )
    authorized_next = (
        "WP10c9d6c7c3b4b2_coarse_heldout_profile_temporal_refinement"
        if passed
        else "WP10c9d6c7c3b4b1_method_failure_localization"
    )
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "layout": LAYOUT,
        "profiles": list(PROFILES),
        "binding_multiplier": 1.0,
        "timestep_seconds": TIMESTEP_SECONDS,
        "horizon_seconds": HORIZON_SECONDS,
        "binding_contract": manifest["binding_contract"],
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(CONFIG_PATH, config)
    np.savez_compressed(DECISIVE_ARRAYS, **arrays)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "authorized_next": authorized_next,
        "operator_changed": False,
        "production_defaults_changed": False,
        "parent_classification": parent["classification"],
        "method_screen": screen,
        "coarse_heldout_temporal_refinement_authorized": passed,
        "temporal_convergence_certified": False,
        "spatial_convergence_certified": False,
        "meaningfully_nonlinear_dynamics_certified": False,
        "long_nonlinear_physical_ladder_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "config_sha256": causal_canonical_json_sha256(config),
        "decisive_arrays_sha256": _sha256(DECISIVE_ARRAYS),
        "decisive_array_hashes": {
            name: causal_array_sha256(values) for name, values in arrays.items()
        },
    }
    _write_json(SUMMARY_PATH, summary)
    input_paths = {
        "parent_summary": PARENT_DIRECTORY / "summary.json",
        "parent_manifest": c3b4a.MANIFEST_PATH,
        "parent_arrays": c3b4a.DECISIVE_ARRAYS,
    }
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "command": (
                "PYTHONPATH=src /Users/huanyang/.cache/codex-runtimes/"
                "codex-primary-runtime/dependencies/python/bin/python3 "
                + THIS_RUNNER
            ),
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "analyzed_base_parent_commit": ANALYZED_BASE_PARENT,
            "analyzed_base_tree_sha": ANALYZED_BASE_TREE,
            "implementation_parent_commit": _git_value("rev-parse", "HEAD"),
            "implementation_parent_tree_sha": _git_value("rev-parse", "HEAD^{tree}"),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "implementation_source_hashes": _source_identity(),
            "input_hashes": {name: _sha256(path) for name, path in input_paths.items()},
        },
    )
    REPORT_PATH.write_text(_report(summary), encoding="utf-8")
    names = ("config.json", "summary.json", "provenance.json", "decisive_arrays.npz")
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0 if passed else 1


def main() -> int:
    parent, manifest = _validate_parent()
    configuration = c3b1a._configurations()[LAYOUT]
    print(f"b4b1: build tangent {LAYOUT}", flush=True)
    started = time.perf_counter()
    tangent = causal_five_field_monolithic_frozen_tangent(
        configuration["context"],
        configuration["base"],
        primitive_column_scales=configuration["columns"],
        conservation_row_scales=configuration["rows"],
    )
    print(f"b4b1: tangent built in {time.perf_counter() - started:.2f}s", flush=True)
    progress, arrays = _run_screen(configuration, tangent)
    return _package(parent, manifest, progress, arrays)


if __name__ == "__main__":
    raise SystemExit(main())
