#!/usr/bin/env python3
"""Execute the durable, resumable coarse nonlinear 10-to-20 ms completion."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_nonlinear_ten_ms_screen_wp10c9d6c7c3b5c4b2 as engine  # noqa: E402
import run_causal_inner_nonlinear_twenty_ms_completion_manifest_wp10c9d6c7c3b5c4c as c4c  # noqa: E402

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    load_causal_five_field_monolithic_bdf_restart,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4c1"
ANALYZED_BASE_COMMIT = "a7fb97fa9385a5762065985af147217118a21390"
ANALYZED_BASE_PARENT = "14d5c829f54999aecedbe93ae747a7ce32d58bc0"
ANALYZED_BASE_TREE = "271196fb3333d5f3d725e676341d0ee98c92d6fc"

ARTIFACT = "causal_inner_nonlinear_twenty_ms_completion_wp10c9d6c7c3b5c4c1"
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_twenty_ms_completion_"
    "wp10c9d6c7c3b5c4c1.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_twenty_ms_completion_"
    "wp10c9d6c7c3b5c4c1.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_TWENTY_MS_COMPLETION_"
    "WP10C9D6C7C3B5C4C1_2026-08-09.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
PROGRESS_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
STAGE_ORDER = engine.STAGE_ORDER
REPLAY_TARGET_MICROSECONDS = c4c.MASTER_TARGET_MICROSECONDS[
    c4c.REPLAY_TARGET_INDICES
]
STRICT_TARGET_MICROSECONDS = c4c.MASTER_TARGET_MICROSECONDS[
    c4c.STRICT_TARGET_INDICES
]

_ORIGINAL_FINALIZE = engine._finalize
TEN_MS_DECISIVE_ARRAYS = engine.DECISIVE_ARRAYS
TEN_MS_SUMMARY_PATH = engine.SUMMARY_PATH


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_identity() -> dict[str, str]:
    return {
        "runner": engine._sha256(ROOT / THIS_RUNNER),
        "manifest": engine._sha256(c4c.MANIFEST_PATH),
        "base_seed_restart": engine._sha256(c4c.BASE_RESTART_PATH),
        "perturbed_seed_restart": engine._sha256(c4c.PERTURBED_RESTART_PATH),
        "ten_ms_arrays": engine._sha256(TEN_MS_DECISIVE_ARRAYS),
        "ten_ms_summary": engine._sha256(TEN_MS_SUMMARY_PATH),
    }


def _validate_parent() -> tuple[dict, dict]:
    parent = _read_json(c4c.SUMMARY_PATH)
    manifest = _read_json(c4c.MANIFEST_PATH)
    if (
        not parent["passed"]
        or not parent["ten_ms_screen_certified"]
        or not parent["twenty_ms_completion_manifest_authorized"]
        or not parent["twenty_ms_propagation_authorized"]
        or parent["twenty_ms_checkpoint_assessment_authorized"]
        or parent["fifty_ms_propagation_authorized"]
        or parent["fixed_q_micro_solver_authorized"]
        or parent["reduced_slow_evolution_authorized"]
        or parent["authorized_next"] != f"{WORK_PACKAGE}_twenty_ms_completion"
    ):
        raise RuntimeError("c4c1 authorization changed")
    if (
        engine._git_value("rev-parse", ANALYZED_BASE_COMMIT)
        != ANALYZED_BASE_COMMIT
        or engine._git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or engine._git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c4c1 analyzed identity changed")
    return parent, manifest


def _seed(name: str, context):
    source = engine._load_npz(TEN_MS_DECISIVE_ARRAYS)
    restart_path = (
        c4c.BASE_RESTART_PATH if name == "base" else c4c.PERTURBED_RESTART_PATH
    )
    prefix = f"{name}_main__"
    restart = load_causal_five_field_monolithic_bdf_restart(restart_path, context)
    if (
        restart.elapsed_time_seconds != c4c.START_MICROSECONDS * 1.0e-6
        or restart.next_order != 2
    ):
        raise RuntimeError(f"{name} canonical 10 ms seed changed")
    arrays = {
        "output_times": np.asarray(source[f"{prefix}output_times"][-1:]),
        "output_states": np.asarray(source[f"{prefix}output_states"][-1:]),
        "output_raw_Tier_I": np.asarray(
            source[f"{prefix}output_raw_Tier_I"][-1:]
        ),
        "output_extraction_partition": np.asarray(
            source[f"{prefix}output_extraction_partition"][-1:]
        ),
        "output_extraction_audits": np.asarray(
            source[f"{prefix}output_extraction_audits"][-1:]
        ),
        "accepted_times": np.asarray(
            (c4c.START_MICROSECONDS * 1.0e-6,), dtype=float
        ),
        "accepted_timesteps": np.empty(0, dtype=float),
        "local_error_estimates": np.empty(0, dtype=float),
        "retries": np.empty(0, dtype=int),
        "accepted_step_wall_seconds": np.empty(0, dtype=float),
    }
    return restart, arrays


def _main_stage(
    name: str,
    configuration: dict,
    tangent,
    field_scales: np.ndarray,
    raw_export_scales: np.ndarray,
    manifest: dict,
) -> dict:
    stage = f"{name}_main"
    loaded = engine._load_stage(stage, configuration["context"])
    if loaded is None:
        restart, arrays = _seed(name, configuration["context"])
        progress = engine._new_progress(
            stage, c4c.START_MICROSECONDS, 4.0e-4
        )
        engine._save_stage(
            stage, progress, arrays, configuration["context"], restart
        )
    else:
        progress, arrays, restart = loaded
    progress, arrays, restart = engine._run_progression(
        stage,
        configuration,
        tangent,
        restart,
        arrays,
        progress,
        c4c.MASTER_TARGET_MICROSECONDS,
        field_scales,
        raw_export_scales,
        manifest["main_controller"],
        manifest,
        strict=False,
    )
    if not progress["complete"]:
        raise RuntimeError(f"{stage} did not complete")
    return {"progress": progress, "arrays": arrays, "restart": restart}


def _finalize(
    parent: dict,
    manifest: dict,
    stages: dict[str, dict],
    field_scales: np.ndarray,
    exterior_scales: np.ndarray,
    context,
    started: float,
) -> int:
    previous_targets = (
        engine.c4b1.MASTER_TARGET_MICROSECONDS,
        engine.c4b1.MAIN_TARGETS_SECONDS,
        engine.c4b1.REPLAY_TARGETS_SECONDS,
        engine.c4b1.STRICT_TARGETS_SECONDS,
    )
    try:
        engine.c4b1.MASTER_TARGET_MICROSECONDS = c4c.MASTER_TARGET_MICROSECONDS
        engine.c4b1.MAIN_TARGETS_SECONDS = c4c.MAIN_TARGETS_SECONDS
        engine.c4b1.REPLAY_TARGETS_SECONDS = c4c.REPLAY_TARGETS_SECONDS
        engine.c4b1.STRICT_TARGETS_SECONDS = c4c.STRICT_TARGETS_SECONDS
        result = _ORIGINAL_FINALIZE(
            parent,
            manifest,
            stages,
            field_scales,
            exterior_scales,
            context,
            started,
        )
    finally:
        (
            engine.c4b1.MASTER_TARGET_MICROSECONDS,
            engine.c4b1.MAIN_TARGETS_SECONDS,
            engine.c4b1.REPLAY_TARGETS_SECONDS,
            engine.c4b1.STRICT_TARGETS_SECONDS,
        ) = previous_targets
    summary = _read_json(SUMMARY_PATH)
    passed = bool(summary["passed"])
    summary.update(
        {
            "ten_ms_screen_certified": True,
            "twenty_ms_completion_manifest_authorized": True,
            "twenty_ms_completion_certified": passed,
            "twenty_ms_propagation_authorized": False,
            "twenty_ms_checkpoint_assessment_authorized": passed,
            "fifty_ms_propagation_authorized": False,
            "fixed_q_micro_solver_authorized": False,
            "reduced_slow_evolution_authorized": False,
        }
    )
    engine._write_json(SUMMARY_PATH, summary)
    strict = summary["strict_response"]
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Nonlinear 20 ms completion WP10c9d6c7c3b5c4c1",
                "",
                "## Classification",
                "",
                f"`{summary['classification']}`",
                "",
                f"Completion passed: `{passed}`.",
                "",
                f"Strict extraction-partition response difference: `{strict['maximum_scaled_extraction_partition_difference']:.6e}`.",
                "",
                f"Authorized next: `{summary['authorized_next']}`.",
                "",
                "The binding slow export remains the certified exterior-domain extraction partition at `R=1.9531594414758637 r_g`, not the raw pointwise horizon flux.",
                "",
                "A pass authorizes only a definitions-only 20 ms checkpoint assessment. Fifty-millisecond propagation, fixed-Q experiments, and reduced slow evolution remain blocked.",
                "",
            )
        ),
        encoding="utf-8",
    )
    names = ("config.json", "decisive_arrays.npz", "provenance.json", "summary.json")
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{engine._sha256(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
    engine._update_catalog(summary)
    print(json.dumps(engine._plain(summary), indent=2, sort_keys=True))
    return result


def _configure_engine() -> None:
    replacements = {
        "SCHEMA_VERSION": SCHEMA_VERSION,
        "WORK_PACKAGE": WORK_PACKAGE,
        "ANALYZED_BASE_COMMIT": ANALYZED_BASE_COMMIT,
        "ANALYZED_BASE_PARENT": ANALYZED_BASE_PARENT,
        "ANALYZED_BASE_TREE": ANALYZED_BASE_TREE,
        "ARTIFACT": ARTIFACT,
        "THIS_RUNNER": THIS_RUNNER,
        "THIS_TEST": THIS_TEST,
        "REPORT_RELATIVE": REPORT_RELATIVE,
        "REPORT_PATH": REPORT_PATH,
        "CANONICAL_DIRECTORY": CANONICAL_DIRECTORY,
        "CONFIG_PATH": CONFIG_PATH,
        "SUMMARY_PATH": SUMMARY_PATH,
        "PROVENANCE_PATH": PROVENANCE_PATH,
        "DECISIVE_ARRAYS": DECISIVE_ARRAYS,
        "PROGRESS_DIRECTORY": PROGRESS_DIRECTORY,
        "CANONICAL_MANIFEST": CANONICAL_MANIFEST,
        "CANONICAL_SUMMARY": CANONICAL_SUMMARY,
        "STAGE_ORDER": STAGE_ORDER,
        "REPLAY_TARGET_MICROSECONDS": REPLAY_TARGET_MICROSECONDS,
        "STRICT_TARGET_MICROSECONDS": STRICT_TARGET_MICROSECONDS,
        "_source_identity": _source_identity,
        "_validate_parent": _validate_parent,
        "_main_stage": _main_stage,
        "_finalize": _finalize,
    }
    for name, value in replacements.items():
        setattr(engine, name, value)


def main(argv: list[str] | None = None) -> int:
    _configure_engine()
    started = time.perf_counter()
    result = engine.main(argv)
    if result == 0 and SUMMARY_PATH.exists():
        print(
            json.dumps(
                {
                    "work_package": WORK_PACKAGE,
                    "elapsed_seconds_in_wrapper": time.perf_counter() - started,
                    "summary_path": str(SUMMARY_PATH),
                },
                indent=2,
                sort_keys=True,
            )
        )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
