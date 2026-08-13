#!/usr/bin/env python3
"""Freeze the response-specific 20 ms temporal-reference hardening contract."""

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

import run_causal_inner_nonlinear_coarse_middle_20ms_checkpoint_analysis_wp10c9d6c7c3b5c4e4 as c4e4  # noqa: E402
import run_causal_inner_nonlinear_twenty_ms_completion_manifest_wp10c9d6c7c3b5c4c as c4c  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4e5"
ANALYZED_BASE_COMMIT = c4e4.ANALYZED_BASE_COMMIT
ANALYZED_BASE_PARENT = c4e4.ANALYZED_BASE_PARENT
ANALYZED_BASE_TREE = c4e4.ANALYZED_BASE_TREE

ARTIFACT = (
    "causal_inner_nonlinear_middle_20ms_temporal_reference_manifest_"
    "wp10c9d6c7c3b5c4e5"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_middle_20ms_temporal_reference_"
    "manifest_wp10c9d6c7c3b5c4e5.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_middle_20ms_temporal_reference_"
    "manifest_wp10c9d6c7c3b5c4e5.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_MIDDLE_20MS_TEMPORAL_"
    "REFERENCE_MANIFEST_WP10C9D6C7C3B5C4E5_2026-08-11.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "temporal_reference_manifest.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

START_MICROSECONDS = 16_000
STOP_MICROSECONDS = 16_400
STRICT_TARGET_MICROSECONDS = (16_000, 16_100, 16_200, 16_300, 16_400)
MAIN_TIMESTEP_SECONDS = 4.0e-4
STRICT_TIMESTEP_SECONDS = 1.0e-4
TEMPORAL_SAFETY_FACTOR = 2.0
MAXIMUM_TEMPORAL_TO_SPATIAL_FRACTION = 0.10
FULL_INTERVAL_SECONDS = 0.015


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
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _manifest() -> dict:
    coarse_manifest = _read_json(c4c.MANIFEST_PATH)
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": "middle_20ms_response_temporal_reference_contract_frozen",
        "definitions_frozen_before_propagation": True,
        "scope": {
            "layouts": ("coarse_N128_inner", "middle_N256_inner"),
            "profile": "generic_five_field",
            "start_microseconds": START_MICROSECONDS,
            "stop_microseconds": STOP_MICROSECONDS,
            "strict_target_microseconds": STRICT_TARGET_MICROSECONDS,
            "coarse_main_reference": (
                "one deterministic 0.4 ms BDF2 continuation from the committed "
                "16 ms restart"
            ),
            "middle_main_reference": "committed c4e3 accepted 16.0 and 16.4 ms states",
            "strict_reference": (
                "unchanged full-step-versus-two-half strict controller with 0.1 ms cap"
            ),
            "slow_export": "certified_exterior_extraction_partition",
            "raw_inner_face_is_not_a_slow_export": True,
        },
        "response_observables": {
            "state": "perturbed_or_anchor_minus_base at common endpoints",
            "instantaneous_extraction": "maximum over common window targets",
            "cumulative_extraction_increment": (
                "trapezoidal integral over 16.0 to 16.4 ms normalized by the "
                "full 5 to 20 ms duration"
            ),
            "window_mean_extraction": "trapezoidal mean over 16.0 to 16.4 ms",
        },
        "temporal_uncertainty": {
            "existing_coarse_windows": "direct main-minus-strict response at 10 and 20 ms",
            "interior_coarse_window": "direct main-minus-strict response at 16 ms",
            "interior_middle_window": "direct main-minus-strict response at 16 ms",
            "combined_formula": (
                "2 * (maximum coarse response-specific discrepancy + "
                "middle response-specific discrepancy)"
            ),
            "safety_factor": TEMPORAL_SAFETY_FACTOR,
            "maximum_fraction_of_spatial_difference": (
                MAXIMUM_TEMPORAL_TO_SPATIAL_FRACTION
            ),
        },
        "method_gates": {
            "maximum_scaled_nonlinear_residual": 1.0e-10,
            "maximum_discrete_ledger_defect": 1.0e-12,
            "maximum_mapped_endpoint_path_closure_defect": 1.0e-9,
            "minimum_reconstruction_factor": 1.0,
            "maximum_incoming_excision_characteristics": 0,
            "maximum_extraction_identity_defect": 1.0e-12,
            "maximum_shared_conservative_face_defect": 1.0e-12,
            "maximum_source_double_count_defect": 1.0e-12,
            "main_local_error_maximum": coarse_manifest["binding_gates"][
                "main_local_error_maximum"
            ],
            "strict_local_error_maximum": coarse_manifest["binding_gates"][
                "strict_local_error_maximum"
            ],
        },
        "decision": {
            "all_response_specific_temporal_ratios_at_most_0.10": (
                "definitions-only cost-bounded fine manifest authorized"
            ),
            "otherwise": "shorter_timestep_or_one_additional_short_shadow_only",
            "fine_propagation_directly_authorized": False,
            "full_fine_generic_anchor_remains_conditional": True,
        },
        "durability": {
            "stage_payload_hashes_required": True,
            "complete_BDF2_histories_required": True,
            "resume_source_identity_required": True,
            "canonical_target_integer_identity_required": True,
        },
        "hard_stops": (
            "do_not_rerun_full_5_to_20ms_coarse_or_middle_histories",
            "do_not_run_fine_in_the_temporal_reference_package",
            "do_not_change_operator_profile_or_thresholds",
            "do_not_run_50ms_fixed_Q_or_reduced_evolution",
            "do_not_use_raw_inner_face_flux_as_slow_export",
        ),
    }


def _validate_parent() -> dict:
    parent = _read_json(c4e4.SUMMARY_PATH)
    if (
        not parent["passed"]
        or parent["coarse_middle_twenty_ms_checkpoint_screen_passed"]
        or parent["fine_completion_manifest_authorized"]
        or parent["fine_twenty_ms_propagation_authorized"]
        or parent["physical_failure_detected"]
        or parent["fixed_q_micro_solver_authorized"]
        or parent["reduced_slow_evolution_authorized"]
        or parent["authorized_next"] != "middle_20ms_temporal_reference_hardening_only"
    ):
        raise RuntimeError("c4e5 parent authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c4e5 analyzed identity changed")
    return parent


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
                    "scientific_status": "PROSPECTIVE",
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
        "classification": (
            "middle_20ms_response_temporal_reference_manifest_frozen_"
            "interior_shadow_authorized"
        ),
        "passed": True,
        "definitions_only": True,
        "temporal_reference_shadow_authorized": True,
        "fine_twenty_ms_manifest_authorized": False,
        "fine_twenty_ms_propagation_authorized": False,
        "fifty_ms_propagation_authorized": False,
        "physical_failure_detected": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": (
            "WP10c9d6c7c3b5c4e6_middle_20ms_response_temporal_reference_shadow"
        ),
        "parent_classification_preserved": parent["classification"],
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "start_microseconds": START_MICROSECONDS,
            "stop_microseconds": STOP_MICROSECONDS,
            "strict_target_microseconds": STRICT_TARGET_MICROSECONDS,
            "main_timestep_seconds": MAIN_TIMESTEP_SECONDS,
            "strict_timestep_seconds": STRICT_TIMESTEP_SECONDS,
            "full_interval_seconds": FULL_INTERVAL_SECONDS,
        },
    )
    _write_json(MANIFEST_PATH, manifest)
    _write_json(SUMMARY_PATH, summary)
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "PROSPECTIVE",
            "command": f"PYTHONPATH=src:scripts python {THIS_RUNNER}",
            "source_parent_commit": ANALYZED_BASE_COMMIT,
            "source_parent_tree_sha": ANALYZED_BASE_TREE,
            "working_head": _git_value("rev-parse", "HEAD"),
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "input_hashes": {
                "c4e4_summary": _sha256(c4e4.SUMMARY_PATH),
                "c4e4_contract": _sha256(c4e4.CONTRACT_PATH),
                "coarse_controller_manifest": _sha256(c4c.MANIFEST_PATH),
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
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Middle 20 ms temporal-reference manifest WP10c9d6c7c3b5c4e5",
                "",
                "## Classification",
                "",
                f"`{summary['classification']}`",
                "",
                "This definitions-only package freezes one response-specific strict shadow from `16.0` to `16.4 ms` on the coarse and middle layouts.",
                "",
                "The binding uncertainty is twice the sum of the maximum coarse and middle response-specific discrepancies. It must remain below `0.10` of each committed coarse-middle spatial difference.",
                "",
                "No propagation occurs here. Fine propagation, 50 ms evolution, fixed-Q experiments, and reduced slow evolution remain blocked.",
                "",
            )
        ),
        encoding="utf-8",
    )
    names = (
        "config.json",
        "temporal_reference_manifest.json",
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
