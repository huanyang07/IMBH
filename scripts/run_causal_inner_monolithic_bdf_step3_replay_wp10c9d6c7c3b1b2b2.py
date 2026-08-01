#!/usr/bin/env python3
"""Compare direct and serialized-restart executions of monolithic BDF2 step 3."""

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
import run_causal_inner_monolithic_bdf_step2_screen_wp10c9d6c7c3b1b2a as c3b1b2a  # noqa: E402
import run_causal_inner_monolithic_bdf_restarted_step3_wp10c9d6c7c3b1b2b1 as c3b1b2b1  # noqa: E402

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    CausalFiveFieldMonolithicBDFHistory,
    advance_causal_five_field_monolithic_bdf,
    causal_five_field_monolithic_frozen_tangent,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_canonical_json_sha256,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b1b2b2"
ANALYZED_BASE_COMMIT = "c5cbab9cc7500e6a9684850f663c8532a7ab4e2f"
ANALYZED_BASE_PARENT = "e0ffa7985c6c81b383c81db778d6b412caa0e79a"
ANALYZED_BASE_TREE = "ea49d573da227496dd498ade3b46dadb530ee17d"

LAYOUTS = tuple(c3b1b2a.LAYOUTS)
PROFILES = tuple(c3b1b2a.PROFILES)
VARIANT_MULTIPLIERS = tuple(c3b1b2a.VARIANT_MULTIPLIERS)
TIMESTEP_SECONDS = c3b1b2a.TIMESTEP_SECONDS
MAXIMUM_SCALED_RESIDUAL = c3b1b2a.MAXIMUM_SCALED_RESIDUAL
MAXIMUM_SCALED_ALGEBRAIC_RESIDUAL = (
    c3b1b2a.MAXIMUM_SCALED_ALGEBRAIC_RESIDUAL
)
MAXIMUM_DISCRETE_LEDGER_DEFECT = c3b1b2a.MAXIMUM_DISCRETE_LEDGER_DEFECT
MAXIMUM_SCALED_PRIMITIVE_CHANGE = c3b1b2a.MAXIMUM_SCALED_PRIMITIVE_CHANGE
MAXIMUM_H_OVER_R = c3b1b2a.MAXIMUM_H_OVER_R
MINIMUM_SCATTERING_OPTICAL_DEPTH = (
    c3b1b2a.MINIMUM_SCATTERING_OPTICAL_DEPTH
)
MINIMUM_RECONSTRUCTION_FACTOR = c3b1b2a.MINIMUM_RECONSTRUCTION_FACTOR
MAXIMUM_MAPPED_ENDPOINT_PATH_CLOSURE = (
    c3b1b2a.MAXIMUM_MAPPED_ENDPOINT_PATH_CLOSURE
)

ARTIFACT = "causal_inner_monolithic_bdf_step3_replay_wp10c9d6c7c3b1b2b2"
THIS_RUNNER = (
    "scripts/run_causal_inner_monolithic_bdf_step3_replay_"
    "wp10c9d6c7c3b1b2b2.py"
)
THIS_TEST = (
    "tests/test_causal_inner_monolithic_bdf_step3_replay_"
    "wp10c9d6c7c3b1b2b2.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_MONOLITHIC_BDF_STEP3_REPLAY_"
    "WP10C9D6C7C3B1B2B2_2026-07-31.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
STEP2_DIRECTORY = c3b1b2a.CANONICAL_DIRECTORY
STEP2_ARRAYS = STEP2_DIRECTORY / "decisive_arrays.npz"
RESTARTED_DIRECTORY = c3b1b2b1.CANONICAL_DIRECTORY
RESTARTED_ARRAYS = RESTARTED_DIRECTORY / "decisive_arrays.npz"
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

REPLAY_FIELDS = (
    "step3_old_state",
    "step3_previous_primitive_increment",
    "step3_previous_mapped_storage_increment",
    "step3_previous_height_storage_increment",
    "step3_primitive_increment",
    "step3_final_state",
    "step3_mapped_storage_increment",
    "step3_height_storage_increment",
)


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
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_identity() -> dict[str, str]:
    paths = (
        THIS_RUNNER,
        THIS_TEST,
        c3b1a.THIS_MODULE,
        c3b1a.THIS_RUNNER,
        c3b1b2a.THIS_RUNNER,
        c3b1b2b1.THIS_RUNNER,
    )
    return {
        path: _sha256(ROOT / path)
        for path in paths
        if (ROOT / path).exists()
    }


def _validate_parents() -> tuple[dict, dict]:
    step2 = _read_json(STEP2_DIRECTORY / "summary.json")
    restarted = _read_json(RESTARTED_DIRECTORY / "summary.json")
    if (
        not step2["passed"]
        or not step2["restart_roundtrip_certified"]
        or step2["completed_case_count"] != 48
    ):
        raise RuntimeError("c3b1b2a step-2 certificate changed")
    if (
        restarted["classification"]
        != "full_profile_variant_restarted_bdf2_step3_depth_certified_"
        "direct_split_replay_comparison_authorized"
        or not restarted["passed"]
        or not restarted["direct_split_replay_comparison_authorized"]
        or restarted["completed_case_count"] != 48
    ):
        raise RuntimeError("c3b1b2b1 restarted step-3 certificate changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT)
        != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c3b1b2b2 analyzed identity changed")
    return step2, restarted


def _case_sequence() -> list[tuple[str, float]]:
    return list(c3b1b2a._case_sequence())


def _case_id(profile: str, multiplier: float) -> str:
    return c3b1b2a._case_id(profile, multiplier)


def _load_progress() -> tuple[dict, dict[str, np.ndarray]]:
    if not CHECKPOINT_JSON.exists():
        return (
            {
                "schema_version": SCHEMA_VERSION,
                "work_package": WORK_PACKAGE,
                "analyzed_base_commit": ANALYZED_BASE_COMMIT,
                "source_identity": _source_identity(),
                "reports": [],
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
        raise RuntimeError("saved c3b1b2b2 progress belongs to different code")
    arrays = (
        _load_npz(CHECKPOINT_ARRAYS)
        if CHECKPOINT_ARRAYS.exists()
        else {}
    )
    return progress, arrays


def _save_progress(progress: dict, arrays: dict[str, np.ndarray]) -> None:
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(CHECKPOINT_JSON, progress)
    np.savez_compressed(CHECKPOINT_ARRAYS, **arrays)


def _direct_history(
    configuration: dict,
    step2_arrays: dict[str, np.ndarray],
    prefix: str,
) -> CausalFiveFieldMonolithicBDFHistory:
    return CausalFiveFieldMonolithicBDFHistory(
        previous_primitive_increment=np.asarray(
            step2_arrays[f"{prefix}__step2_primitive_increment"],
            dtype=float,
        ),
        previous_mapped_storage_increment=np.asarray(
            step2_arrays[f"{prefix}__step2_mapped_storage_increment"],
            dtype=float,
        ),
        previous_responsive_height_storage_increment=np.asarray(
            step2_arrays[f"{prefix}__step2_height_storage_increment"],
            dtype=float,
        ),
        previous_timestep_seconds=TIMESTEP_SECONDS,
    ).validated(n_cells=configuration["base"].shape[0])


def _run_case(
    label: str,
    configuration: dict,
    tangent,
    profile: str,
    multiplier: float,
    step2_arrays: dict[str, np.ndarray],
    restarted_arrays: dict[str, np.ndarray],
) -> tuple[dict, dict[str, np.ndarray]]:
    case = _case_id(profile, multiplier)
    prefix = f"{label}__{case}"
    old_state = np.asarray(
        step2_arrays[f"{prefix}__step2_final_state"],
        dtype=float,
    )
    history = _direct_history(configuration, step2_arrays, prefix)
    started = time.perf_counter()
    step = advance_causal_five_field_monolithic_bdf(
        configuration["context"],
        old_state,
        TIMESTEP_SECONDS,
        tangent,
        order=2,
        history=history,
        residual_tolerance=MAXIMUM_SCALED_RESIDUAL,
        ledger_tolerance=MAXIMUM_DISCRETE_LEDGER_DEFECT,
        maximum_scaled_primitive_change=MAXIMUM_SCALED_PRIMITIVE_CHANGE,
    )
    if step.history is None:
        raise RuntimeError(f"{prefix} direct step 3 returned no BDF history")
    direct_arrays = {
        f"{prefix}__step3_old_state": old_state,
        f"{prefix}__step3_previous_primitive_increment": (
            history.previous_primitive_increment
        ),
        f"{prefix}__step3_previous_mapped_storage_increment": (
            history.previous_mapped_storage_increment
        ),
        f"{prefix}__step3_previous_height_storage_increment": (
            history.previous_responsive_height_storage_increment
        ),
        f"{prefix}__step3_primitive_increment": step.primitive_increment,
        f"{prefix}__step3_final_state": step.primitive_charts,
        f"{prefix}__step3_mapped_storage_increment": (
            step.history.previous_mapped_storage_increment
        ),
        f"{prefix}__step3_height_storage_increment": (
            step.history.previous_responsive_height_storage_increment
        ),
    }
    field_parity = {
        field: bool(
            np.array_equal(
                direct_arrays[f"{prefix}__{field}"],
                restarted_arrays[f"{prefix}__{field}"],
                equal_nan=True,
            )
        )
        for field in REPLAY_FIELDS
    }
    field_maximum_absolute_differences = {
        field: float(
            np.max(
                np.abs(
                    direct_arrays[f"{prefix}__{field}"]
                    - restarted_arrays[f"{prefix}__{field}"]
                )
            )
        )
        for field in REPLAY_FIELDS
    }
    final_audit = c3b1a._state_audit(
        configuration["context"],
        step.primitive_charts,
    )
    mapped_closure = (
        step.evaluation.maximum_mapped_endpoint_path_closure_defect
    )
    method_passed = bool(
        step.accepted
        and step.maximum_scaled_residual <= MAXIMUM_SCALED_RESIDUAL
        and step.maximum_scaled_algebraic_residual
        <= MAXIMUM_SCALED_ALGEBRAIC_RESIDUAL
        and step.maximum_discrete_ledger_defect
        <= MAXIMUM_DISCRETE_LEDGER_DEFECT
        and step.maximum_scaled_primitive_change
        <= MAXIMUM_SCALED_PRIMITIVE_CHANGE
        and step.minimum_path_reconstruction_factor
        >= MINIMUM_RECONSTRUCTION_FACTOR
        and mapped_closure <= MAXIMUM_MAPPED_ENDPOINT_PATH_CLOSURE
        and step.incoming_excision_characteristics == 0
        and final_audit["maximum_h_over_r"] <= MAXIMUM_H_OVER_R
        and final_audit["minimum_scattering_optical_depth"]
        > MINIMUM_SCATTERING_OPTICAL_DEPTH
        and final_audit["minimum_reconstruction_factor"]
        >= MINIMUM_RECONSTRUCTION_FACTOR
    )
    replay_passed = bool(all(field_parity.values()))
    report = {
        "layout": label,
        "profile": profile,
        "variant_multiplier": multiplier,
        "case_id": case,
        "step_order": 2,
        "cumulative_completed_steps": 3,
        "direct_execution": True,
        "serialized_restart_reference": True,
        "passed": bool(method_passed and replay_passed),
        "method_passed": method_passed,
        "bitwise_replay_passed": replay_passed,
        "bitwise_field_parity": field_parity,
        "field_maximum_absolute_differences": (
            field_maximum_absolute_differences
        ),
        "accepted": step.accepted,
        "message": step.message,
        "maximum_scaled_residual": step.maximum_scaled_residual,
        "maximum_scaled_algebraic_residual": (
            step.maximum_scaled_algebraic_residual
        ),
        "maximum_discrete_ledger_defect": (
            step.maximum_discrete_ledger_defect
        ),
        "maximum_scaled_primitive_change": (
            step.maximum_scaled_primitive_change
        ),
        "maximum_mapped_endpoint_path_closure_defect": mapped_closure,
        "minimum_path_reconstruction_factor": (
            step.minimum_path_reconstruction_factor
        ),
        "incoming_excision_characteristics": (
            step.incoming_excision_characteristics
        ),
        "final_state_audit": final_audit,
        "iterations": step.iterations,
        "function_evaluations": step.function_evaluations,
        "maximum_linear_residual": step.maximum_linear_residual,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return report, direct_arrays


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
            fieldnames=[
                "case",
                "path",
                "bytes",
                "sha256",
                "scientific_status",
            ],
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
            "case_count": len({str(row["case"]) for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": ANALYZED_BASE_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def _package(
    progress: dict,
    arrays: dict[str, np.ndarray],
    *,
    step2: dict,
    restarted: dict,
) -> dict:
    reports = list(progress["reports"])
    expected_count = len(LAYOUTS) * len(PROFILES) * len(VARIANT_MULTIPLIERS)
    passed = bool(
        len(reports) == expected_count
        and all(report["passed"] for report in reports)
    )
    classification = (
        "full_profile_variant_bdf2_step3_direct_restarted_bitwise_replay_"
        "certified_fourth_step_depth_authorized"
        if passed
        else "full_profile_variant_bdf2_step3_direct_restarted_replay_"
        "failed_nonlinear_depth_blocked"
    )
    authorized_next = (
        "WP10c9d6c7c3b1b3a_fourth_step_depth"
        if passed
        else "WP10c9d6c7c3b1b2b2_replay_failure_localization"
    )
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "layouts": list(LAYOUTS),
        "profiles": list(PROFILES),
        "variant_multipliers": list(VARIANT_MULTIPLIERS),
        "fixed_timestep_seconds": TIMESTEP_SECONDS,
        "step_in_this_stage": 3,
        "step_order": 2,
        "direct_execution_from_canonical_step2": True,
        "reference_execution_from_serialized_step2": True,
        "bitwise_fields": list(REPLAY_FIELDS),
        "full_frozen_method_steps": 4,
        "gates": {
            "all_replay_fields_bitwise_equal": True,
            "maximum_scaled_residual": MAXIMUM_SCALED_RESIDUAL,
            "maximum_scaled_algebraic_residual": (
                MAXIMUM_SCALED_ALGEBRAIC_RESIDUAL
            ),
            "maximum_discrete_ledger_defect": (
                MAXIMUM_DISCRETE_LEDGER_DEFECT
            ),
            "maximum_scaled_primitive_change": (
                MAXIMUM_SCALED_PRIMITIVE_CHANGE
            ),
            "maximum_mapped_endpoint_path_closure_defect": (
                MAXIMUM_MAPPED_ENDPOINT_PATH_CLOSURE
            ),
            "maximum_h_over_r": MAXIMUM_H_OVER_R,
            "minimum_scattering_optical_depth": (
                MINIMUM_SCATTERING_OPTICAL_DEPTH
            ),
            "minimum_reconstruction_factor": (
                MINIMUM_RECONSTRUCTION_FACTOR
            ),
            "incoming_excision_characteristics": 0,
        },
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
        "direct_step3_execution_certified": passed,
        "direct_restarted_step3_bitwise_replay_certified": passed,
        "fourth_step_depth_authorized": passed,
        "long_nonlinear_physical_ladder_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "c3b1b2a_step2_certificate_preserved": True,
        "c3b1b2b1_restarted_step3_certificate_preserved": True,
        "c3a_manufactured_background_rejection_preserved": True,
        "c7c1b_strict_auxiliary_rejection_preserved": True,
        "expected_case_count": expected_count,
        "completed_case_count": len(reports),
        "case_reports": reports,
        "all_cases_bitwise_equal": bool(
            reports and all(report["bitwise_replay_passed"] for report in reports)
        ),
        "maximum_bitwise_replay_absolute_difference": max(
            (
                max(report["field_maximum_absolute_differences"].values())
                for report in reports
            ),
            default=None,
        ),
        "maximum_scaled_residual": max(
            (report["maximum_scaled_residual"] for report in reports),
            default=None,
        ),
        "maximum_discrete_ledger_defect": max(
            (
                report["maximum_discrete_ledger_defect"]
                for report in reports
            ),
            default=None,
        ),
        "maximum_mapped_endpoint_path_closure_defect": max(
            (
                report[
                    "maximum_mapped_endpoint_path_closure_defect"
                ]
                for report in reports
            ),
            default=None,
        ),
        "maximum_function_evaluations": max(
            (report["function_evaluations"] for report in reports),
            default=None,
        ),
        "maximum_iterations": max(
            (report["iterations"] for report in reports),
            default=None,
        ),
        "step2_parent_classification": step2["classification"],
        "restarted_parent_classification": restarted["classification"],
        "config_sha256": causal_canonical_json_sha256(config),
        "decisive_arrays_sha256": _sha256(DECISIVE_ARRAYS),
        "decisive_array_hashes": {
            name: causal_array_sha256(values)
            for name, values in arrays.items()
        },
    }
    _write_json(SUMMARY_PATH, summary)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "analyzed_base_parent_commit": ANALYZED_BASE_PARENT,
        "analyzed_base_tree_sha": ANALYZED_BASE_TREE,
        "implementation_source_hashes": _source_identity(),
        "input_hashes": {
            "c3b1b2a_summary": _sha256(STEP2_DIRECTORY / "summary.json"),
            "c3b1b2a_arrays": _sha256(STEP2_ARRAYS),
            "c3b1b2b1_summary": _sha256(
                RESTARTED_DIRECTORY / "summary.json"
            ),
            "c3b1b2b1_arrays": _sha256(RESTARTED_ARRAYS),
        },
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "command": f"PYTHONPATH=src python3 {THIS_RUNNER}",
    }
    _write_json(PROVENANCE_PATH, provenance)
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Monolithic BDF direct/restarted step-3 replay "
                "WP10c9d6c7c3b1b2b2",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                "Every direct third BDF2 step is independently executed "
                "from canonical step-2 state and history, then compared "
                "bitwise with the serialized-restart execution.",
                "",
                "## Result",
                "",
                f"- Completed cases: `{len(reports)}/{expected_count}`",
                (
                    "- Maximum replay absolute difference: "
                    f"`{summary['maximum_bitwise_replay_absolute_difference']}`"
                ),
                (
                    "- Maximum scaled residual: "
                    f"`{summary['maximum_scaled_residual']}`"
                ),
                (
                    "- Maximum ledger defect: "
                    f"`{summary['maximum_discrete_ledger_defect']}`"
                ),
                "",
                "## Authorized next",
                "",
                f"`{authorized_next}`",
                "",
                "Fourth-step depth remains conditional on this package. "
                "Long nonlinear trajectories, fixed-Q experiments, and "
                "reduced slow evolution remain blocked.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    checksums = {
        path.name: _sha256(path)
        for path in (
            CONFIG_PATH,
            SUMMARY_PATH,
            PROVENANCE_PATH,
            DECISIVE_ARRAYS,
        )
    }
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(checksums.items())
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> int:
    step2, restarted = _validate_parents()
    configurations = c3b1a._configurations()
    step2_arrays = _load_npz(STEP2_ARRAYS)
    restarted_arrays = _load_npz(RESTARTED_ARRAYS)
    progress, arrays = _load_progress()
    completed = {
        (report["layout"], report["case_id"])
        for report in progress["reports"]
    }
    for label in LAYOUTS:
        pending = [
            (profile, multiplier)
            for profile, multiplier in _case_sequence()
            if (label, _case_id(profile, multiplier)) not in completed
        ]
        if not pending or progress["failed"]:
            continue
        configuration = configurations[label]
        print(f"c3b1b2b2: build tangent {label}", flush=True)
        tangent = causal_five_field_monolithic_frozen_tangent(
            configuration["context"],
            configuration["base"],
            primitive_column_scales=configuration["columns"],
            conservation_row_scales=configuration["rows"],
        )
        for profile, multiplier in pending:
            case = _case_id(profile, multiplier)
            print(f"c3b1b2b2: {label} {case}", flush=True)
            report, case_arrays = _run_case(
                label,
                configuration,
                tangent,
                profile,
                multiplier,
                step2_arrays,
                restarted_arrays,
            )
            progress["reports"].append(report)
            arrays.update(case_arrays)
            progress["failed"] = not report["passed"]
            _save_progress(progress, arrays)
            if progress["failed"]:
                break
        if progress["failed"]:
            break
    summary = _package(
        progress,
        arrays,
        step2=step2,
        restarted=restarted,
    )
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
