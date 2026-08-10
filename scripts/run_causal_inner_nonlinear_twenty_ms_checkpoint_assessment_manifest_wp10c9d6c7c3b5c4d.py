#!/usr/bin/env python3
"""Freeze the evidence-only assessment at the certified 20 ms checkpoint."""

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

import run_causal_inner_nonlinear_ten_ms_screen_wp10c9d6c7c3b5c4b2 as c4b2  # noqa: E402
import run_causal_inner_nonlinear_twenty_ms_completion_wp10c9d6c7c3b5c4c1 as c4c1  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4d"
ANALYZED_BASE_COMMIT = "52641403173c84d570ec7890c21d642144165824"
ANALYZED_BASE_PARENT = "27b241476ad8efd35bf05815c76fb37d03ba129c"
ANALYZED_BASE_TREE = "d7bdbb92c31e3cbe89ae0c02de0cd6e84787f041"

ARTIFACT = (
    "causal_inner_nonlinear_twenty_ms_checkpoint_assessment_manifest_"
    "wp10c9d6c7c3b5c4d"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_twenty_ms_checkpoint_assessment_"
    "manifest_wp10c9d6c7c3b5c4d.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_twenty_ms_checkpoint_assessment_"
    "manifest_wp10c9d6c7c3b5c4d.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_TWENTY_MS_CHECKPOINT_"
    "ASSESSMENT_MANIFEST_WP10C9D6C7C3B5C4D_2026-08-10.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "checkpoint_assessment_manifest.json"
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
    parent = _read_json(c4c1.SUMMARY_PATH)
    if (
        not parent["passed"]
        or not parent["twenty_ms_completion_certified"]
        or not parent["twenty_ms_checkpoint_assessment_authorized"]
        or parent["fifty_ms_propagation_authorized"]
        or parent["physical_failure_detected"]
        or parent["fixed_q_micro_solver_authorized"]
        or parent["reduced_slow_evolution_authorized"]
        or parent["authorized_next"]
        != f"{WORK_PACKAGE}_twenty_ms_checkpoint_assessment_manifest"
    ):
        raise RuntimeError("c4d authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c4d analyzed identity changed")
    return parent


def _manifest() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            "twenty_ms_checkpoint_assessment_manifest_frozen_evidence_only_"
            "assessment_authorized"
        ),
        "passed": True,
        "definitions_only": True,
        "propagation_executed": False,
        "operator_changed": False,
        "production_defaults_changed": False,
        "inputs": {
            "ten_ms_summary": str(c4b2.SUMMARY_PATH.relative_to(ROOT)),
            "ten_ms_arrays": str(c4b2.DECISIVE_ARRAYS.relative_to(ROOT)),
            "twenty_ms_summary": str(c4c1.SUMMARY_PATH.relative_to(ROOT)),
            "twenty_ms_arrays": str(c4c1.DECISIVE_ARRAYS.relative_to(ROOT)),
            "all_inputs_must_match_committed_hashes": True,
        },
        "assessment_metrics": {
            "ten_to_twenty_boundary_state_bitwise": True,
            "ten_to_twenty_boundary_raw_Tier_I_bitwise": True,
            "ten_to_twenty_boundary_extraction_partition_bitwise": True,
            "scaled_state_response_rms_and_max_history": True,
            "scaled_extraction_response_rms_and_max_history": True,
            "endpoint_response_ratio_twenty_over_ten": True,
            "endpoint_direction_cosine_twenty_vs_ten": True,
            "base_and_perturbed_readiness_margin_change": True,
            "accepted_local_error_sum_and_stage_cost": True,
            "duration_in_crossing_and_relaxation_clocks": True,
        },
        "binding_gates": {
            "boundary_bitwise": True,
            "all_twenty_ms_stages_passed": True,
            "all_replays_passed": True,
            "strict_response_passed": True,
            "physical_failure_detected": False,
            "maximum_scaled_state_response": 5.0e-2,
            "maximum_scaled_extraction_partition_response": 5.0e-3,
            "minimum_scattering_optical_depth": 1.0,
            "maximum_h_over_r": 0.12,
            "minimum_reconstruction_factor": 1.0,
        },
        "interpretation_contract": {
            "endpoint_ratio_below_0p9": "decreasing_over_10_to_20ms",
            "endpoint_ratio_0p9_to_1p1": "approximately_stationary_amplitude",
            "endpoint_ratio_above_1p1": "increasing_over_10_to_20ms",
            "amplitude_trend_is_not_an_attractor_classification": True,
            "one_perturbation_is_not_multiple_equal_Q_fast_lifts": True,
            "twenty_ms_is_only_0p136_of_reference_stress_relaxation_time": True,
            "no_fixed_Q_or_slow_reduction_claim_can_be_authorized": True,
        },
        "positive_branch": {
            "classification": (
                "twenty_ms_checkpoint_assessed_cost_bounded_spatial_"
                "checkpoint_manifest_authorized"
            ),
            "authorized_next": (
                "WP10c9d6c7c3b5c4e_twenty_ms_spatial_checkpoint_manifest"
            ),
            "fifty_ms_requires_spatial_checkpoint_decision": True,
        },
        "negative_branch": {
            "classification": "twenty_ms_checkpoint_assessment_failed",
            "authorized_next": "failure_localization_only",
        },
        "hard_stops": (
            "do_not_infer_attraction_or_memory_loss_from_one_trajectory_pair",
            "do_not_run_fifty_ms_before_the_spatial_checkpoint_decision",
            "do_not_use_raw_inner_face_flux_as_the_slow_export",
            "do_not_start_fixed_Q_or_reduced_slow_evolution",
            "do_not_change_operator_profile_or_production_defaults",
        ),
        "authorized_next": (
            "WP10c9d6c7c3b5c4d1_twenty_ms_checkpoint_assessment"
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
    parent = _validate_parent()
    manifest = _manifest()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": manifest["classification"],
        "passed": True,
        "definitions_only": True,
        "parent_classification_preserved": parent["classification"],
        "twenty_ms_checkpoint_assessment_authorized": True,
        "twenty_ms_spatial_checkpoint_manifest_authorized": False,
        "fifty_ms_propagation_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": manifest["authorized_next"],
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "assessment_interval_seconds": (0.010, 0.020),
            "reference_stress_relaxation_seconds": 0.147,
            "reference_N128_cell_crossing_seconds": 5.54e-3,
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
            "input_hashes": {
                "ten_ms_summary": _sha256(c4b2.SUMMARY_PATH),
                "ten_ms_arrays": _sha256(c4b2.DECISIVE_ARRAYS),
                "twenty_ms_summary": _sha256(c4c1.SUMMARY_PATH),
                "twenty_ms_arrays": _sha256(c4c1.DECISIVE_ARRAYS),
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
                "# Nonlinear 20 ms checkpoint-assessment manifest WP10c9d6c7c3b5c4d",
                "",
                "## Classification",
                "",
                f"`{manifest['classification']}`",
                "",
                "This definitions-only package freezes an evidence-only assessment of the certified 10 and 20 ms trajectories. It executes no new physical propagation.",
                "",
                "The assessment measures exact continuation closure, response amplitude and direction, readiness margins, accumulated temporal error, physical-clock coverage, and measured runtime. A response trend is diagnostic and cannot establish attraction or memory loss.",
                "",
                "A pass authorizes only a cost-bounded 20 ms spatial-checkpoint manifest. Fifty-millisecond propagation, fixed-Q experiments, and reduced slow evolution remain blocked.",
                "",
            )
        ),
        encoding="utf-8",
    )
    names = (
        "checkpoint_assessment_manifest.json",
        "config.json",
        "provenance.json",
        "summary.json",
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
