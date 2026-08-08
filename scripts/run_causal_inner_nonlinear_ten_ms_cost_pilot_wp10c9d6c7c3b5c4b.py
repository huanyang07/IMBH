#!/usr/bin/env python3
"""Measure one accepted coarse step before the complete 10 ms screen."""

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
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_monolithic_bdf_base_preflight_wp10c9d6c7c3b1a as c3b1a  # noqa: E402
import run_causal_inner_nonlinear_5ms_extraction_surface_certificate_wp10c9d6c7c3b5c3h2i1 as h2i1  # noqa: E402
import run_causal_inner_nonlinear_coarse_third_duration_rung_completion_wp10c9d6c7c3b5c3d as c3d  # noqa: E402
import run_causal_inner_nonlinear_fourth_duration_rung_manifest_wp10c9d6c7c3b5c4a as c4a  # noqa: E402
import run_causal_inner_nonlinear_second_duration_rung_wp10c9d6c7c3b5c2 as c2  # noqa: E402
import run_causal_inner_nonlinear_spatial_export_pilot_wp10c9d6c7c3b2b as c3b2b  # noqa: E402
import run_causal_inner_nonlinear_third_duration_rung_screen_wp10c9d6c7c3b5c3b as c3b  # noqa: E402

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    causal_five_field_monolithic_frozen_tangent,
    load_causal_five_field_monolithic_bdf_restart,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_canonical_json_sha256,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4b"
ANALYZED_BASE_COMMIT = "5f248c5c9249e003b0237e063215faab85e34b79"
ANALYZED_BASE_PARENT = "f0c661691e55c70d27bf7825c589671357221872"
ANALYZED_BASE_TREE = "9729c78f1cf4dc7e6a5891ac9b365391cd5401ef"

ARTIFACT = "causal_inner_nonlinear_ten_ms_cost_pilot_wp10c9d6c7c3b5c4b"
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_ten_ms_cost_pilot_"
    "wp10c9d6c7c3b5c4b.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_ten_ms_cost_pilot_"
    "wp10c9d6c7c3b5c4b.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_TEN_MS_COST_PILOT_"
    "WP10C9D6C7C3B5C4B_2026-08-08.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
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


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


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


def _validate_parent() -> tuple[dict, dict]:
    parent = _read_json(c4a.SUMMARY_PATH)
    manifest = _read_json(c4a.MANIFEST_PATH)
    if (
        not parent["passed"]
        or not parent["ten_ms_cost_pilot_authorized"]
        or parent["ten_ms_screen_propagation_authorized"]
        or parent["authorized_next"] != f"{WORK_PACKAGE}_ten_ms_cost_pilot"
        or manifest["propagation_executed"]
        or manifest["authorized_next"] != f"{WORK_PACKAGE}_ten_ms_cost_pilot"
    ):
        raise RuntimeError("c4b authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c4b analyzed identity changed")
    return parent, manifest


def _load_restart(name: str, context):
    path = c3d.PROGRESS_DIRECTORY / name / "final_restart.npz"
    restart = load_causal_five_field_monolithic_bdf_restart(path, context)
    if (
        restart.elapsed_time_seconds != c4a.RUNG_START_SECONDS
        or restart.next_order != 2
        or restart.history.previous_timestep_seconds <= 0.0
    ):
        raise RuntimeError(f"{name} 5 ms restart is incomplete")
    return restart, path


def _exterior_history(context, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = []
    audits = []
    for state in np.asarray(states, dtype=float):
        value, audit = h2i1._exterior_observable(
            context,
            state,
            c4a.SELECTED_EXTRACTION_LAYOUT_FACE_INDICES[0],
            c2.COUPLING_FACE,
        )
        values.append(value)
        audits.append(audit)
    return np.asarray(values), np.asarray(audits)


def _audit_report(audits: np.ndarray, gates: dict) -> dict:
    audits = np.asarray(audits, dtype=float)
    report = {
        "maximum_shared_conservative_face_defect": float(np.max(audits[:, 0])),
        "maximum_local_block_ledger_defect": float(np.max(audits[:, 1])),
        "maximum_source_double_count_defect": float(np.max(audits[:, 2])),
        "maximum_split_closure_defect": float(np.max(audits[:, 3])),
        "maximum_incoming_excision_characteristics": int(np.max(audits[:, 4])),
        "transport_sign_convention_diagnostic": float(np.max(audits[:, 5])),
        "maximum_exterior_prefix_identity_defect": float(np.max(audits[:, 6])),
    }
    report["passed"] = bool(
        report["maximum_shared_conservative_face_defect"]
        <= gates["maximum_shared_conservative_face_defect"]
        and report["maximum_local_block_ledger_defect"] <= 1.0e-12
        and report["maximum_source_double_count_defect"] <= 1.0e-12
        and report["maximum_split_closure_defect"] <= 1.0e-10
        and report["maximum_incoming_excision_characteristics"]
        <= gates["maximum_incoming_excision_characteristics"]
        and report["maximum_exterior_prefix_identity_defect"]
        <= gates["maximum_exterior_prefix_identity_defect"]
    )
    return report


def _reuse_base_attempt(manifest: dict) -> tuple[dict, dict] | None:
    """Reuse the numerically valid first base attempt after its audit-only stop."""
    if not (SUMMARY_PATH.exists() and DECISIVE_ARRAYS.exists()):
        return None
    previous = _read_json(SUMMARY_PATH)
    prior_report = previous.get("trajectory_reports", {}).get("base")
    if (
        previous.get("work_package") != WORK_PACKAGE
        or prior_report is None
        or not prior_report["method"]["method_passed"]
        or prior_report["method"]["accepted_comparisons"] != 1
    ):
        return None
    arrays = _load_npz(DECISIVE_ARRAYS)
    required = {
        "base__output_states",
        "base__output_raw_Tier_I",
        "base__output_extraction_partition",
        "base__extraction_partition_audits",
        "base__accepted_timesteps",
        "base__accepted_step_wall_seconds",
    }
    if not required.issubset(arrays):
        return None
    audit = _audit_report(
        arrays["base__extraction_partition_audits"], manifest["binding_gates"]
    )
    gates = manifest["binding_gates"]
    method = prior_report["method"]
    readiness = prior_report["readiness"]
    passed = bool(
        method["method_passed"]
        and method["maximum_local_error_estimate"]
        <= gates["main_local_error_maximum"]
        and method["sum_local_error_estimates"]
        <= gates["main_local_error_sum_maximum"]
        and readiness["minimum_scattering_optical_depth"]
        >= gates["minimum_scattering_optical_depth"]
        and readiness["maximum_h_over_r"] <= gates["maximum_h_over_r"]
        and readiness["minimum_reconstruction_factor"]
        >= gates["minimum_reconstruction_factor"]
        and audit["passed"]
    )
    report = {
        **prior_report,
        "passed": passed,
        "extraction_partition_audit": audit,
        "reused_numerical_attempt_after_audit_contract_correction": True,
    }
    payload = {
        "segment": {
            "output_states": arrays["base__output_states"],
            "output_exports": arrays["base__output_raw_Tier_I"],
            "accepted_timesteps": arrays["base__accepted_timesteps"],
            "accepted_step_wall_seconds": arrays[
                "base__accepted_step_wall_seconds"
            ],
        },
        "exterior": arrays["base__output_extraction_partition"],
        "audits": arrays["base__extraction_partition_audits"],
    }
    return report, payload


def _run_trajectory(
    name: str,
    configuration: dict,
    tangent,
    restart,
    field_scales: np.ndarray,
    export_scales: np.ndarray,
    manifest: dict,
) -> tuple[dict, dict]:
    print(f"c4b: {name} pilot", flush=True)
    started = time.perf_counter()
    segment = c2._controller_segment(
        configuration,
        tangent,
        restart.primitive_charts,
        restart.history,
        c4a.RUNG_START_SECONDS,
        manifest["main_controller"]["initial_timestep_seconds"],
        field_scales,
        export_scales,
        c2.COUPLING_FACE,
        manifest["main_controller"],
        output_times=c4a.PILOT_TARGETS_SECONDS,
        stop_time=c4a.PILOT_HORIZON_SECONDS,
        include_initial_output=True,
        record_accepted_steps=True,
        log_prefix=f"c4b-{name}",
    )
    elapsed = time.perf_counter() - started
    method = c3b._segment_report(segment, manifest["main_controller"])
    readiness = c3b1a._state_audit(configuration["context"], segment["final_state"])
    exterior, audits = _exterior_history(
        configuration["context"], segment["output_states"]
    )
    audit = _audit_report(audits, manifest["binding_gates"])
    gates = manifest["binding_gates"]
    passed = bool(
        method["method_passed"]
        and method["accepted_comparisons"] == 1
        and method["maximum_local_error_estimate"]
        <= gates["main_local_error_maximum"]
        and method["sum_local_error_estimates"]
        <= gates["main_local_error_sum_maximum"]
        and readiness["minimum_scattering_optical_depth"]
        >= gates["minimum_scattering_optical_depth"]
        and readiness["maximum_h_over_r"] <= gates["maximum_h_over_r"]
        and readiness["minimum_reconstruction_factor"]
        >= gates["minimum_reconstruction_factor"]
        and audit["passed"]
    )
    report = {
        "trajectory": name,
        "passed": passed,
        "elapsed_seconds": elapsed,
        "seconds_per_accepted_comparison": elapsed,
        "method": method,
        "readiness": readiness,
        "extraction_partition_audit": audit,
    }
    payload = {
        "segment": segment,
        "exterior": exterior,
        "audits": audits,
    }
    return report, payload


def _runtime_projection(
    setup_seconds: float, reports: dict[str, dict], manifest: dict
) -> dict:
    execution = manifest["screen_execution"]
    comparisons = sum(
        execution[key]
        for key in (
            "main_expected_comparisons_per_trajectory",
            "replay_expected_comparisons_per_trajectory",
            "strict_expected_comparisons_per_trajectory",
        )
    )
    raw = setup_seconds + comparisons * sum(
        reports[name]["seconds_per_accepted_comparison"]
        for name in ("base", "perturbed")
    )
    projected = manifest["pilot"]["projection_safety_factor"] * raw
    hours = projected / 3600.0
    if hours <= 24.0:
        branch = "continue_automatically"
    elif hours <= 48.0:
        branch = "continue_after_optimization_review"
    else:
        branch = "stop_and_optimize_before_full_screen"
    return {
        "screen_comparisons_per_trajectory": comparisons,
        "raw_projected_wall_seconds": raw,
        "safety_factor": manifest["pilot"]["projection_safety_factor"],
        "projected_wall_seconds": projected,
        "projected_wall_hours": hours,
        "resource_branch": branch,
        "resource_projection_is_not_a_physical_gate": True,
        "full_screen_resource_authorized": hours <= 48.0,
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


def main() -> int:
    started = time.perf_counter()
    parent, manifest = _validate_parent()
    configuration = c3b1a._configurations()[c2.LAYOUT]
    pilot_inputs = _load_npz(c3b2b.CANONICAL_DIRECTORY / "decisive_arrays.npz")
    field_scales = pilot_inputs["field_scales"]
    export_scales = pilot_inputs["fixed_physical_observable_scales"]
    print("c4b: build frozen nonlinear tangent", flush=True)
    tangent_started = time.perf_counter()
    tangent = causal_five_field_monolithic_frozen_tangent(
        configuration["context"],
        configuration["base"],
        primitive_column_scales=configuration["columns"],
        conservation_row_scales=configuration["rows"],
    )
    setup_seconds = time.perf_counter() - tangent_started
    reports = {}
    payloads = {}
    restart_paths = {}
    for name in ("base", "perturbed"):
        restart, path = _load_restart(name, configuration["context"])
        restart_paths[name] = path
        reused = _reuse_base_attempt(manifest) if name == "base" else None
        if reused is not None:
            print("c4b: reuse numerically valid base pilot attempt", flush=True)
            reports[name], payloads[name] = reused
        else:
            reports[name], payloads[name] = _run_trajectory(
                name,
                configuration,
                tangent,
                restart,
                field_scales,
                export_scales,
                manifest,
            )
        if not reports[name]["passed"]:
            break
    both_ran = set(reports) == {"base", "perturbed"}
    scientific_passed = bool(both_ran and all(item["passed"] for item in reports.values()))
    runtime = (
        _runtime_projection(setup_seconds, reports, manifest)
        if scientific_passed
        else None
    )
    resource_authorized = bool(
        scientific_passed and runtime["full_screen_resource_authorized"]
    )
    if resource_authorized:
        classification = (
            "ten_ms_cost_pilot_certified_full_ten_ms_screen_authorized"
        )
        authorized_next = "WP10c9d6c7c3b5c4b1_ten_ms_screen"
    elif scientific_passed:
        classification = (
            "ten_ms_cost_pilot_scientifically_passed_runtime_optimization_required"
        )
        authorized_next = "runtime_optimization_only"
    else:
        classification = "ten_ms_cost_pilot_failed_later_duration_blocked"
        authorized_next = "failure_localization_only"
    arrays = {
        "field_scales": field_scales,
        "export_scales": export_scales,
        "pilot_times_seconds": c4a.PILOT_TARGETS_SECONDS,
    }
    response = None
    if both_ran:
        response = {
            "maximum_absolute_state_response": float(
                np.max(
                    np.abs(
                        payloads["perturbed"]["segment"]["output_states"]
                        - payloads["base"]["segment"]["output_states"]
                    )
                )
            ),
            "maximum_absolute_extraction_partition_response": float(
                np.max(
                    np.abs(
                        payloads["perturbed"]["exterior"]
                        - payloads["base"]["exterior"]
                    )
                )
            ),
        }
    for name, payload in payloads.items():
        arrays[f"{name}__output_states"] = payload["segment"]["output_states"]
        arrays[f"{name}__output_raw_Tier_I"] = payload["segment"]["output_exports"]
        arrays[f"{name}__output_extraction_partition"] = payload["exterior"]
        arrays[f"{name}__extraction_partition_audits"] = payload["audits"]
        arrays[f"{name}__accepted_timesteps"] = payload["segment"]["accepted_timesteps"]
        arrays[f"{name}__accepted_step_wall_seconds"] = payload["segment"]["accepted_step_wall_seconds"]
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "pilot_start_seconds": c4a.RUNG_START_SECONDS,
        "pilot_horizon_seconds": c4a.PILOT_HORIZON_SECONDS,
        "pilot_targets_seconds": c4a.PILOT_TARGETS_SECONDS,
        "selected_extraction_radius_rg": c4a.SELECTED_EXTRACTION_RADIUS_RG,
        "selected_extraction_face": c4a.SELECTED_EXTRACTION_LAYOUT_FACE_INDICES[0],
        "main_controller": manifest["main_controller"],
    }
    _write_json(CONFIG_PATH, config)
    np.savez_compressed(DECISIVE_ARRAYS, **arrays)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": scientific_passed,
        "physical_failure_detected": not scientific_passed,
        "runtime_projection": runtime,
        "trajectory_reports": reports,
        "response": response,
        "setup_seconds": setup_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "ten_ms_screen_propagation_authorized": resource_authorized,
        "twenty_ms_completion_manifest_authorized": False,
        "twenty_ms_propagation_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": authorized_next,
        "parent_classification_preserved": parent["classification"],
        "pointwise_horizon_flux_convergence_claimed": False,
        "raw_inner_face_rejection_preserved": True,
        "config_sha256": causal_canonical_json_sha256(_plain(config)),
        "decisive_arrays_sha256": _sha256(DECISIVE_ARRAYS),
        "decisive_array_hashes": {
            name: causal_array_sha256(value) for name, value in arrays.items()
        },
    }
    _write_json(SUMMARY_PATH, summary)
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "CERTIFIED" if scientific_passed else "REJECTED",
            "command": f"PYTHONPATH=src:scripts python {THIS_RUNNER}",
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
                for path in (THIS_RUNNER, THIS_TEST)
                if (ROOT / path).exists()
            },
            "input_hashes": {
                "manifest_summary": _sha256(c4a.SUMMARY_PATH),
                "manifest": _sha256(c4a.MANIFEST_PATH),
                "base_restart": _sha256(restart_paths["base"]),
                **(
                    {"perturbed_restart": _sha256(restart_paths["perturbed"])}
                    if "perturbed" in restart_paths
                    else {}
                ),
            },
        },
    )
    projected = runtime["projected_wall_hours"] if runtime else float("nan")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Nonlinear 10 ms cost pilot WP10c9d6c7c3b5c4b",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                f"Scientific pilot passed: `{scientific_passed}`.",
                "",
                f"Projected complete-screen wall time: `{projected:.3f} h`.",
                "",
                f"Authorized next: `{authorized_next}`.",
                "",
                "The pilot advances the committed coarse base and generic perturbed BDF2 restarts from 5.0 to 5.4 ms. It retains the full-step/two-half-step estimator and evaluates the certified exterior-domain extraction partition at `R=1.9531594414758637 r_g`.",
                "",
                "Runtime classification is not a physical classification. The raw inner-face flux remains rejected and is not used as the slow export. Fixed-Q experiments and reduced slow evolution remain blocked.",
                "",
            )
        ),
        encoding="utf-8",
    )
    names = ("config.json", "decisive_arrays.npz", "provenance.json", "summary.json")
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0 if scientific_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
