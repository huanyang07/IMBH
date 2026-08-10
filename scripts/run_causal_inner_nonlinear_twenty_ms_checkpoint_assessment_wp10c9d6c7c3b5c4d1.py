#!/usr/bin/env python3
"""Assess the certified 10 and 20 ms nonlinear checkpoint evidence."""

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
import run_causal_inner_nonlinear_twenty_ms_checkpoint_assessment_manifest_wp10c9d6c7c3b5c4d as c4d  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4d1"
ANALYZED_BASE_COMMIT = "032a2346090901d6498ead0f0ac21239fd172f19"
ANALYZED_BASE_PARENT = "52641403173c84d570ec7890c21d642144165824"
ANALYZED_BASE_TREE = "453d5490f73e02adf6b8ba53f1270a126336b28d"

ARTIFACT = (
    "causal_inner_nonlinear_twenty_ms_checkpoint_assessment_"
    "wp10c9d6c7c3b5c4d1"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_twenty_ms_checkpoint_assessment_"
    "wp10c9d6c7c3b5c4d1.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_twenty_ms_checkpoint_assessment_"
    "wp10c9d6c7c3b5c4d1.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_TWENTY_MS_CHECKPOINT_"
    "ASSESSMENT_WP10C9D6C7C3B5C4D1_2026-08-10.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
ASSESSMENT_PATH = CANONICAL_DIRECTORY / "assessment.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
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


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = np.asarray(left, dtype=float).ravel()
    right_flat = np.asarray(right, dtype=float).ravel()
    denominator = float(np.linalg.norm(left_flat) * np.linalg.norm(right_flat))
    if denominator == 0.0:
        return 1.0 if np.array_equal(left_flat, right_flat) else 0.0
    return float(np.dot(left_flat, right_flat) / denominator)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else math.inf
    return float(numerator / denominator)


def _trend(ratio: float, contract: dict) -> str:
    if ratio < 0.9:
        return contract["endpoint_ratio_below_0p9"]
    if ratio <= 1.1:
        return contract["endpoint_ratio_0p9_to_1p1"]
    return contract["endpoint_ratio_above_1p1"]


def _validate_parent() -> tuple[dict, dict, dict, dict]:
    manifest_summary = _read_json(c4d.SUMMARY_PATH)
    manifest = _read_json(c4d.MANIFEST_PATH)
    ten_summary = _read_json(c4b2.SUMMARY_PATH)
    twenty_summary = _read_json(c4c1.SUMMARY_PATH)
    if (
        not manifest_summary["passed"]
        or not manifest_summary["twenty_ms_checkpoint_assessment_authorized"]
        or manifest_summary["twenty_ms_spatial_checkpoint_manifest_authorized"]
        or manifest_summary["fifty_ms_propagation_authorized"]
        or manifest_summary["fixed_q_micro_solver_authorized"]
        or manifest_summary["reduced_slow_evolution_authorized"]
        or manifest_summary["authorized_next"]
        != f"{WORK_PACKAGE}_twenty_ms_checkpoint_assessment"
    ):
        raise RuntimeError("c4d1 authorization changed")
    if not ten_summary["passed"] or not twenty_summary["passed"]:
        raise RuntimeError("certified duration input changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c4d1 analyzed identity changed")
    return manifest_summary, manifest, ten_summary, twenty_summary


def _boundary_report(ten: dict[str, np.ndarray], twenty: dict[str, np.ndarray]) -> dict:
    report = {}
    for trajectory in ("base_main", "perturbed_main"):
        trajectory_report = {}
        for field in (
            "output_times",
            "output_states",
            "output_raw_Tier_I",
            "output_extraction_partition",
        ):
            left = ten[f"{trajectory}__{field}"][-1]
            right = twenty[f"{trajectory}__{field}"][0]
            trajectory_report[f"{field}_bitwise"] = bool(np.array_equal(left, right))
            trajectory_report[f"{field}_maximum_absolute_difference"] = float(
                np.max(np.abs(left - right))
            )
        trajectory_report["passed"] = all(
            value
            for key, value in trajectory_report.items()
            if key.endswith("_bitwise")
        )
        report[trajectory.removesuffix("_main")] = trajectory_report
    report["passed"] = all(report[name]["passed"] for name in ("base", "perturbed"))
    return report


def _response_histories(ten: dict[str, np.ndarray], twenty: dict[str, np.ndarray]):
    times = np.concatenate(
        (ten["base_main__output_times"][-1:], twenty["base_main__output_times"][1:])
    )
    state_response = np.concatenate(
        (
            ten["perturbed_main__output_states"][-1:]
            - ten["base_main__output_states"][-1:],
            twenty["perturbed_main__output_states"][1:]
            - twenty["base_main__output_states"][1:],
        )
    )
    extraction_response = np.concatenate(
        (
            ten["perturbed_main__output_extraction_partition"][-1:]
            - ten["base_main__output_extraction_partition"][-1:],
            twenty["perturbed_main__output_extraction_partition"][1:]
            - twenty["base_main__output_extraction_partition"][1:],
        )
    )
    field_scales = twenty["field_scales"]
    extraction_scales = twenty["extraction_partition_scales"]
    scaled_state = state_response / field_scales[None, None, :]
    scaled_extraction = extraction_response / extraction_scales[None, :]
    state_rms = np.sqrt(np.mean(scaled_state**2, axis=(1, 2)))
    state_max = np.max(np.abs(scaled_state), axis=(1, 2))
    extraction_rms = np.sqrt(np.mean(scaled_extraction**2, axis=1))
    extraction_max = np.max(np.abs(scaled_extraction), axis=1)
    return {
        "times": times,
        "state_response": state_response,
        "extraction_response": extraction_response,
        "scaled_state": scaled_state,
        "scaled_extraction": scaled_extraction,
        "state_rms": state_rms,
        "state_max": state_max,
        "extraction_rms": extraction_rms,
        "extraction_max": extraction_max,
        "field_scales": field_scales,
        "extraction_scales": extraction_scales,
    }


def _readiness_change(ten: dict, twenty: dict) -> dict:
    result = {}
    for trajectory in ("base", "perturbed"):
        old = ten["final_state_readiness"][trajectory]
        new = twenty["final_state_readiness"][trajectory]
        result[trajectory] = {
            "ten_ms": old,
            "twenty_ms": new,
            "change": {key: float(new[key] - old[key]) for key in old},
        }
    return result


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
                    "scientific_status": "CERTIFIED" if summary["passed"] else "REJECTED",
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
    manifest_summary, manifest, ten_summary, twenty_summary = _validate_parent()
    with np.load(c4b2.DECISIVE_ARRAYS) as source:
        ten = {name: np.asarray(source[name]) for name in source.files}
    with np.load(c4c1.DECISIVE_ARRAYS) as source:
        twenty = {name: np.asarray(source[name]) for name in source.files}

    boundary = _boundary_report(ten, twenty)
    histories = _response_histories(ten, twenty)
    contract = manifest["interpretation_contract"]
    endpoint = {
        "state_rms_ratio_twenty_over_ten": _safe_ratio(
            histories["state_rms"][-1], histories["state_rms"][0]
        ),
        "state_max_ratio_twenty_over_ten": _safe_ratio(
            histories["state_max"][-1], histories["state_max"][0]
        ),
        "extraction_rms_ratio_twenty_over_ten": _safe_ratio(
            histories["extraction_rms"][-1], histories["extraction_rms"][0]
        ),
        "extraction_max_ratio_twenty_over_ten": _safe_ratio(
            histories["extraction_max"][-1], histories["extraction_max"][0]
        ),
        "state_direction_cosine_twenty_vs_ten": _cosine(
            histories["scaled_state"][0], histories["scaled_state"][-1]
        ),
        "extraction_direction_cosine_twenty_vs_ten": _cosine(
            histories["scaled_extraction"][0], histories["scaled_extraction"][-1]
        ),
    }
    endpoint["state_rms_trend"] = _trend(
        endpoint["state_rms_ratio_twenty_over_ten"], contract
    )
    endpoint["extraction_rms_trend"] = _trend(
        endpoint["extraction_rms_ratio_twenty_over_ten"], contract
    )

    maximum_state = float(np.max(histories["state_max"]))
    maximum_extraction = float(np.max(histories["extraction_max"]))
    gates = manifest["binding_gates"]
    all_stages_passed = all(
        report["passed"] for report in twenty_summary["stage_reports"].values()
    )
    all_replays_passed = all(
        report["passed"] for report in twenty_summary["replay_reports"].values()
    )
    readiness = _readiness_change(ten_summary, twenty_summary)
    final_readiness = twenty_summary["final_state_readiness"]
    physical_gate_passed = all(
        item["minimum_scattering_optical_depth"]
        >= gates["minimum_scattering_optical_depth"]
        and item["maximum_h_over_r"] <= gates["maximum_h_over_r"]
        and item["minimum_reconstruction_factor"]
        >= gates["minimum_reconstruction_factor"]
        for item in final_readiness.values()
    )
    passed = bool(
        boundary["passed"]
        and all_stages_passed
        and all_replays_passed
        and twenty_summary["strict_response"]["passed"]
        and not twenty_summary["physical_failure_detected"]
        and maximum_state <= gates["maximum_scaled_state_response"]
        and maximum_extraction
        <= gates["maximum_scaled_extraction_partition_response"]
        and physical_gate_passed
    )

    stage_costs = {
        "ten_ms_elapsed_seconds": ten_summary["elapsed_seconds"],
        "twenty_ms_elapsed_seconds": twenty_summary["elapsed_seconds"],
        "twenty_ms_main_wall_seconds": sum(
            twenty_summary["stage_reports"][name]["measured_wall_seconds"]
            for name in ("base_main", "perturbed_main")
        ),
        "twenty_ms_main_sum_local_error_estimates": sum(
            twenty_summary["stage_reports"][name]["sum_local_error_estimates"]
            for name in ("base_main", "perturbed_main")
        ),
        "twenty_ms_total_elapsed_hours": twenty_summary["elapsed_seconds"] / 3600.0,
    }
    clock_coverage = {
        "duration_seconds": 0.020,
        "reference_N128_cell_crossing_seconds": 5.54e-3,
        "reference_stress_relaxation_seconds": 0.147,
        "N128_cell_crossings": 0.020 / 5.54e-3,
        "stress_relaxation_times": 0.020 / 0.147,
    }
    classification = (
        manifest["positive_branch"]["classification"]
        if passed
        else manifest["negative_branch"]["classification"]
    )
    authorized_next = (
        manifest["positive_branch"]["authorized_next"]
        if passed
        else manifest["negative_branch"]["authorized_next"]
    )
    assessment = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "boundary": boundary,
        "response_history": {
            "maximum_scaled_state_response": maximum_state,
            "maximum_scaled_extraction_partition_response": maximum_extraction,
            "endpoint": endpoint,
        },
        "readiness_change": readiness,
        "stage_costs": stage_costs,
        "physical_clock_coverage": clock_coverage,
        "interpretation": {
            "state_amplitude_is_approximately_stationary_over_10_to_20ms": (
                endpoint["state_rms_trend"]
                == contract["endpoint_ratio_0p9_to_1p1"]
            ),
            "extraction_partition_response_increases_over_10_to_20ms": (
                endpoint["extraction_rms_trend"]
                == contract["endpoint_ratio_above_1p1"]
            ),
            "attraction_or_memory_loss_demonstrated": False,
            "multiple_equal_Q_lifts_tested": False,
            "physical_failure_detected": bool(
                twenty_summary["physical_failure_detected"]
            ),
            "spatial_checkpoint_required_before_fifty_ms": True,
        },
        "gates": {
            "boundary_bitwise": boundary["passed"],
            "all_twenty_ms_stages_passed": all_stages_passed,
            "all_replays_passed": all_replays_passed,
            "strict_response_passed": twenty_summary["strict_response"]["passed"],
            "state_response_cap_passed": maximum_state
            <= gates["maximum_scaled_state_response"],
            "extraction_response_cap_passed": maximum_extraction
            <= gates["maximum_scaled_extraction_partition_response"],
            "physical_readiness_passed": physical_gate_passed,
        },
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "parent_classification_preserved": manifest_summary["classification"],
        "twenty_ms_completion_certified": bool(
            twenty_summary["twenty_ms_completion_certified"]
        ),
        "twenty_ms_checkpoint_assessed": passed,
        "twenty_ms_spatial_checkpoint_manifest_authorized": passed,
        "fifty_ms_propagation_authorized": False,
        "physical_failure_detected": bool(
            twenty_summary["physical_failure_detected"]
        ),
        "pointwise_horizon_flux_convergence_claimed": False,
        "raw_inner_face_rejection_preserved": True,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "maximum_scaled_state_response": maximum_state,
        "maximum_scaled_extraction_partition_response": maximum_extraction,
        "state_rms_ratio_twenty_over_ten": endpoint[
            "state_rms_ratio_twenty_over_ten"
        ],
        "extraction_rms_ratio_twenty_over_ten": endpoint[
            "extraction_rms_ratio_twenty_over_ten"
        ],
        "authorized_next": authorized_next,
    }

    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "assessment_interval_seconds": [0.010, 0.020],
            "binding_gates": gates,
        },
    )
    _write_json(ASSESSMENT_PATH, assessment)
    _write_json(SUMMARY_PATH, summary)
    np.savez_compressed(
        DECISIVE_ARRAYS,
        output_times=histories["times"],
        state_response=histories["state_response"],
        extraction_partition_response=histories["extraction_response"],
        scaled_state_response=histories["scaled_state"],
        scaled_extraction_partition_response=histories["scaled_extraction"],
        scaled_state_rms=histories["state_rms"],
        scaled_state_max=histories["state_max"],
        scaled_extraction_partition_rms=histories["extraction_rms"],
        scaled_extraction_partition_max=histories["extraction_max"],
        field_scales=histories["field_scales"],
        extraction_partition_scales=histories["extraction_scales"],
    )
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "CERTIFIED" if passed else "REJECTED",
            "source_parent_commit": ANALYZED_BASE_COMMIT,
            "source_parent_tree_sha": ANALYZED_BASE_TREE,
            "working_head": _git_value("rev-parse", "HEAD"),
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "input_hashes": {
                "manifest_summary": _sha256(c4d.SUMMARY_PATH),
                "manifest": _sha256(c4d.MANIFEST_PATH),
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
                "# Nonlinear 20 ms checkpoint assessment WP10c9d6c7c3b5c4d1",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                f"Assessment passed: `{passed}`.",
                "",
                "The 10-to-20 ms continuation boundary is bitwise exact for base and perturbed state, raw Tier-I exports, and the certified extraction partition.",
                "",
                f"The maximum scaled state response is `{maximum_state:.8e}` and its 20/10 ms RMS ratio is `{endpoint['state_rms_ratio_twenty_over_ten']:.8f}`. The maximum scaled extraction-partition response is `{maximum_extraction:.8e}` and its RMS ratio is `{endpoint['extraction_rms_ratio_twenty_over_ten']:.8f}`.",
                "",
                f"Twenty milliseconds spans `{clock_coverage['N128_cell_crossings']:.3f}` reference N128 crossing times but only `{clock_coverage['stress_relaxation_times']:.3f}` of the reference stress-relaxation time.",
                "",
                "No physical failure is detected. The nearly stationary state-response amplitude does not demonstrate attraction, while the extraction-partition response continues to accumulate. Multiple equal-Q lifts have not been tested.",
                "",
                f"Authorized next: `{authorized_next}`.",
                "",
                "Fifty-millisecond propagation, fixed-Q experiments, and reduced slow evolution remain blocked pending the cost-bounded 20 ms spatial-checkpoint decision. The binding slow export remains the certified exterior-domain extraction partition, not the raw pointwise horizon flux.",
                "",
            )
        ),
        encoding="utf-8",
    )
    names = (
        "assessment.json",
        "config.json",
        "decisive_arrays.npz",
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
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
