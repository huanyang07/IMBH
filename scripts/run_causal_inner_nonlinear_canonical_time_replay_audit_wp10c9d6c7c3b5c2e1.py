#!/usr/bin/env python3
"""Audit the one-ULP target-grid effect under one frozen tangent."""

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
import run_causal_inner_nonlinear_canonical_time_replay_manifest_wp10c9d6c7c3b5c2e as c2e  # noqa: E402
import run_causal_inner_nonlinear_second_duration_rung_wp10c9d6c7c3b5c2 as c2  # noqa: E402
import run_causal_inner_nonlinear_second_rung_perturbed_completion_wp10c9d6c7c3b5c2d as c2d  # noqa: E402
import run_causal_inner_nonlinear_spatial_export_pilot_wp10c9d6c7c3b2b as c3b2b  # noqa: E402

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    CausalFiveFieldMonolithicBDFHistory,
    CausalFiveFieldMonolithicBDFRestart,
    advance_causal_five_field_monolithic_bdf,
    causal_five_field_monolithic_bdf_restarts_equal,
    causal_five_field_monolithic_frozen_tangent,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_canonical_json_sha256,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c2e1"
ANALYZED_BASE_COMMIT = "d18fe34d01f5f62f8229b109353fc8c846fc685c"
ANALYZED_BASE_PARENT = "c60fa67412d4e6861d5c4b3102ce2d8c2dba46ae"
ANALYZED_BASE_TREE = "7bf9e53881b1c738ee7405b60c97dfb7dc10bb19"

ARTIFACT = "causal_inner_nonlinear_canonical_time_replay_audit_wp10c9d6c7c3b5c2e1"
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_canonical_time_replay_audit_"
    "wp10c9d6c7c3b5c2e1.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_canonical_time_replay_audit_"
    "wp10c9d6c7c3b5c2e1.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_NONLINEAR_CANONICAL_TIME_REPLAY_AUDIT_"
    "WP10C9D6C7C3B5C2E1_2026-08-04.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
PARENT_DIRECTORY = c2e.CANONICAL_DIRECTORY


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
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _validate_parent() -> tuple[dict, dict]:
    parent = _read_json(PARENT_DIRECTORY / "summary.json")
    manifest = _read_json(c2e.MANIFEST_PATH)
    if (
        not parent["passed"]
        or not parent["canonical_time_replay_audit_authorized"]
        or parent["authorized_next"]
        != "WP10c9d6c7c3b5c2e1_canonical_time_replay_audit"
        or manifest["classification"]
        != "canonical_time_replay_manifest_corrected_paired_target_grid_audit_authorized"
        or manifest["propagation_executed"]
    ):
        raise RuntimeError("c2e1 authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c2e1 analyzed identity changed")
    return parent, manifest


def _history_from_arrays(
    arrays: dict[str, np.ndarray], trajectory: str, context
):
    prefix = f"{trajectory}__main_replay_"
    if f"{prefix}primitive_history" not in arrays:
        previous = np.asarray(arrays[f"{trajectory}__states"][7], dtype=float)
        current = np.asarray(arrays[f"{trajectory}__states"][8], dtype=float)
        storage = c2.causal_five_field_monolithic_storage_increment(
            context, previous, current
        )
        return c2.causal_five_field_monolithic_bdf_history(
            current - previous,
            storage,
            1.0e-4,
        )
    return CausalFiveFieldMonolithicBDFHistory(
        previous_primitive_increment=np.array(
            arrays[f"{prefix}primitive_history"][1], copy=True
        ),
        previous_mapped_storage_increment=np.array(
            arrays[f"{prefix}mapped_history"][1], copy=True
        ),
        previous_responsive_height_storage_increment=np.array(
            arrays[f"{prefix}height_history"][1], copy=True
        ),
        previous_timestep_seconds=1.0e-4,
    )


def _restart_from_arrays(
    arrays: dict[str, np.ndarray], trajectory: str, context
) -> CausalFiveFieldMonolithicBDFRestart:
    return CausalFiveFieldMonolithicBDFRestart(
        primitive_charts=np.array(arrays[f"{trajectory}__states"][8], copy=True),
        history=_history_from_arrays(arrays, trajectory, context),
        elapsed_time_seconds=float(c2e.CANONICAL_TARGETS[0]),
        completed_steps=1,
        next_order=2,
        provenance={
            "work_package": WORK_PACKAGE,
            "trajectory": trajectory,
            "source": "committed_c2d_at_8e_minus_4",
        },
    )


def _branch(
    configuration: dict,
    tangent,
    restart: CausalFiveFieldMonolithicBDFRestart,
    targets: np.ndarray,
    *,
    serialized: bool,
    label: str,
    method_contract: dict,
) -> dict:
    context = configuration["context"]
    if serialized:
        active, roundtrip = c2._save_restore(context, restart, label)
    else:
        active = restart
        roundtrip = True
    state = np.array(active.primitive_charts, copy=True)
    history = active.history
    elapsed = float(active.elapsed_time_seconds)
    states = [np.array(state, copy=True)]
    histories = [history]
    exports = [c2.c3b5b._export_value(context, state, c2.COUPLING_FACE)[0]]
    export_ledgers = [
        c2.c3b5b._export_value(context, state, c2.COUPLING_FACE)[1]
    ]
    incoming = [c2.c3b5b._export_value(context, state, c2.COUPLING_FACE)[2]]
    records = []
    for target in np.asarray(targets[1:], dtype=float):
        timestep = float(target - elapsed)
        step = advance_causal_five_field_monolithic_bdf(
            context,
            state,
            timestep,
            tangent,
            order=2,
            history=history,
            residual_tolerance=1.0e-10,
            ledger_tolerance=1.0e-12,
            maximum_scaled_primitive_change=5.0e-3,
        )
        record = c2.c3b5b._step_record(step)
        records.append(record)
        if not c2.c3b5b._step_passed(step, method_contract) or step.history is None:
            raise RuntimeError(f"{label} method gate failed")
        state = np.array(step.primitive_charts, copy=True)
        history = step.history
        elapsed = float(target)
        export, ledger, modes = c2.c3b5b._export_value(
            context, state, c2.COUPLING_FACE
        )
        states.append(state)
        histories.append(history)
        exports.append(export)
        export_ledgers.append(ledger)
        incoming.append(modes)
        print(
            f"c2e1: {label} t={elapsed:.17g} dt={timestep:.17g} "
            f"residual={step.maximum_scaled_residual:.3e}",
            flush=True,
        )
    final_restart = CausalFiveFieldMonolithicBDFRestart(
        primitive_charts=np.array(state, copy=True),
        history=history,
        elapsed_time_seconds=elapsed,
        completed_steps=active.completed_steps + len(targets) - 1,
        next_order=2,
        provenance=active.provenance,
    )
    return {
        "times": np.asarray(targets, dtype=float),
        "states": np.asarray(states, dtype=float),
        "exports": np.asarray(exports, dtype=float),
        "primitive_history": np.asarray(
            [item.previous_primitive_increment for item in histories]
        ),
        "mapped_history": np.asarray(
            [item.previous_mapped_storage_increment for item in histories]
        ),
        "height_history": np.asarray(
            [item.previous_responsive_height_storage_increment for item in histories]
        ),
        "previous_timestep": np.asarray(
            [item.previous_timestep_seconds for item in histories]
        ),
        "records": records,
        "export_ledgers": np.asarray(export_ledgers, dtype=float),
        "incoming": np.asarray(incoming, dtype=int),
        "restart": final_restart,
        "initial_roundtrip_bitwise": roundtrip,
    }


def _relative_norm(left: np.ndarray, right: np.ndarray) -> float:
    delta = np.asarray(left) - np.asarray(right)
    scale = max(float(np.linalg.norm(np.asarray(right).ravel())), np.finfo(float).tiny)
    return float(np.linalg.norm(delta.ravel()) / scale)


def _scaled_max(left: np.ndarray, right: np.ndarray, scales: np.ndarray) -> float:
    return float(np.max(np.abs((np.asarray(left) - np.asarray(right)) / scales)))


def _history_relative(left: dict, right: dict) -> float:
    return max(
        _relative_norm(left[name], right[name])
        for name in ("primitive_history", "mapped_history", "height_history")
    )


def _first_difference(left: dict, right: dict) -> int | None:
    for index in range(left["times"].size):
        if (
            not np.array_equal(left["states"][index], right["states"][index])
            or not np.array_equal(left["exports"][index], right["exports"][index])
            or not np.array_equal(
                left["primitive_history"][index], right["primitive_history"][index]
            )
            or not np.array_equal(
                left["mapped_history"][index], right["mapped_history"][index]
            )
            or not np.array_equal(
                left["height_history"][index], right["height_history"][index]
            )
        ):
            return index
    return None


def _method_report(branches: dict[str, dict], method_contract: dict) -> dict:
    records = [record for branch in branches.values() for record in branch["records"]]
    report = {
        "step_count": len(records),
        "maximum_scaled_residual": max(item["maximum_scaled_residual"] for item in records),
        "maximum_discrete_ledger_defect": max(
            item["maximum_discrete_ledger_defect"] for item in records
        ),
        "maximum_mapped_endpoint_path_closure_defect": max(
            item["maximum_mapped_endpoint_path_closure_defect"] for item in records
        ),
        "minimum_path_reconstruction_factor": min(
            item["minimum_path_reconstruction_factor"] for item in records
        ),
        "maximum_incoming_excision_characteristics": max(
            int(np.max(branch["incoming"])) for branch in branches.values()
        ),
        "maximum_export_ledger_defect": max(
            float(np.max(branch["export_ledgers"])) for branch in branches.values()
        ),
    }
    gates = method_contract["step_method_gates"]
    report["passed"] = bool(
        all(item["accepted"] for item in records)
        and report["maximum_scaled_residual"] <= gates["maximum_scaled_residual"]
        and report["maximum_discrete_ledger_defect"]
        <= gates["maximum_discrete_ledger_defect"]
        and report["maximum_mapped_endpoint_path_closure_defect"]
        <= gates["maximum_mapped_endpoint_path_closure"]
        and report["minimum_path_reconstruction_factor"]
        >= gates["minimum_reconstruction_factor"] - 1.0e-12
        and report["maximum_incoming_excision_characteristics"]
        <= gates["maximum_incoming_excision_characteristics"]
    )
    return report


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
    arrays = _load_npz(c2d.DECISIVE_ARRAYS)
    pilot = _load_npz(c3b2b.CANONICAL_DIRECTORY / "decisive_arrays.npz")
    field_scales = pilot["field_scales"]
    export_scales = pilot["fixed_physical_observable_scales"]
    configuration = c3b1a._configurations()[c2.LAYOUT]
    print("c2e1: build tangent", flush=True)
    tangent = causal_five_field_monolithic_frozen_tangent(
        configuration["context"],
        configuration["base"],
        primitive_column_scales=configuration["columns"],
        conservation_row_scales=configuration["rows"],
    )
    method_contract = {"step_method_gates": manifest["method_gates"]}
    branches: dict[str, dict] = {}
    trajectory_reports = {}
    output_arrays: dict[str, np.ndarray] = {
        "canonical_targets_seconds": c2e.CANONICAL_TARGETS,
        "legacy_targets_seconds": c2e.LEGACY_TARGETS,
        "field_scales": field_scales,
        "export_scales": export_scales,
    }
    for trajectory in c2e.TRAJECTORIES:
        restart = _restart_from_arrays(
            arrays, trajectory, configuration["context"]
        )
        legacy = _branch(
            configuration,
            tangent,
            restart,
            c2e.LEGACY_TARGETS,
            serialized=False,
            label=f"{trajectory}_legacy",
            method_contract=method_contract,
        )
        canonical = _branch(
            configuration,
            tangent,
            restart,
            c2e.CANONICAL_TARGETS,
            serialized=False,
            label=f"{trajectory}_canonical",
            method_contract=method_contract,
        )
        serialized = _branch(
            configuration,
            tangent,
            restart,
            c2e.CANONICAL_TARGETS,
            serialized=True,
            label=f"{trajectory}_canonical_serialized",
            method_contract=method_contract,
        )
        branches.update(
            {
                f"{trajectory}_legacy": legacy,
                f"{trajectory}_canonical": canonical,
                f"{trajectory}_canonical_serialized": serialized,
            }
        )
        committed = {
            "states": arrays[f"{trajectory}__states"][8:],
            "exports": arrays[f"{trajectory}__direct_exports"][8:],
        }
        has_committed_history = (
            f"{trajectory}__main_replay_primitive_history" in arrays
        )
        if has_committed_history:
            committed.update(
                {
                    "primitive_history": arrays[
                        f"{trajectory}__main_replay_primitive_history"
                    ][1:],
                    "mapped_history": arrays[
                        f"{trajectory}__main_replay_mapped_history"
                    ][1:],
                    "height_history": arrays[
                        f"{trajectory}__main_replay_height_history"
                    ][1:],
                }
            )
        fresh_process = {
            "maximum_scaled_state_difference": _scaled_max(
                legacy["states"], committed["states"], field_scales[None, None, :]
            ),
            "maximum_scaled_export_difference": _scaled_max(
                legacy["exports"], committed["exports"], export_scales[None, :]
            ),
            "maximum_relative_history_difference": (
                _history_relative(legacy, committed)
                if has_committed_history
                else 0.0
            ),
            "committed_history_reference_available": has_committed_history,
            "history_reference": (
                "committed_c2d_history"
                if has_committed_history
                else "reconstructed_from_committed_endpoint_states"
            ),
        }
        same_target = {
            "target_labels_bitwise": np.array_equal(
                canonical["times"], serialized["times"]
            ),
            "states_bitwise": np.array_equal(
                canonical["states"], serialized["states"]
            ),
            "Tier_I_exports_bitwise": np.array_equal(
                canonical["exports"], serialized["exports"]
            ),
            "primitive_history_bitwise": np.array_equal(
                canonical["primitive_history"], serialized["primitive_history"]
            ),
            "mapped_history_bitwise": np.array_equal(
                canonical["mapped_history"], serialized["mapped_history"]
            ),
            "height_history_bitwise": np.array_equal(
                canonical["height_history"], serialized["height_history"]
            ),
            "previous_timestep_bitwise": np.array_equal(
                canonical["previous_timestep"], serialized["previous_timestep"]
            ),
            "final_restart_bitwise": causal_five_field_monolithic_bdf_restarts_equal(
                canonical["restart"], serialized["restart"]
            ),
            "serialized_initial_roundtrip_bitwise": serialized[
                "initial_roundtrip_bitwise"
            ],
        }
        legacy_canonical = {
            "first_difference_index": _first_difference(legacy, canonical),
            "relative_state_norm_difference": _relative_norm(
                legacy["states"], canonical["states"]
            ),
            "relative_export_norm_difference": _relative_norm(
                legacy["exports"], canonical["exports"]
            ),
            "relative_history_norm_difference": _history_relative(
                legacy, canonical
            ),
        }
        gates = manifest["binding_gates"]
        trajectory_passed = bool(
            fresh_process["maximum_scaled_state_difference"]
            <= gates["legacy_committed_maximum_scaled_state_difference"]
            and fresh_process["maximum_scaled_export_difference"]
            <= gates["legacy_committed_maximum_scaled_export_difference"]
            and fresh_process["maximum_relative_history_difference"]
            <= gates["legacy_committed_maximum_relative_history_difference"]
            and all(same_target.values())
            and legacy_canonical["first_difference_index"]
            == gates["legacy_and_canonical_first_difference_index"]
            and legacy_canonical["relative_state_norm_difference"]
            <= gates["legacy_canonical_maximum_relative_state_norm_difference"]
            and legacy_canonical["relative_export_norm_difference"]
            <= gates["legacy_canonical_maximum_relative_export_norm_difference"]
            and legacy_canonical["relative_history_norm_difference"]
            <= gates["legacy_canonical_maximum_relative_history_norm_difference"]
        )
        trajectory_reports[trajectory] = {
            "fresh_process_committed_comparison": fresh_process,
            "same_target_direct_serialized": same_target,
            "legacy_canonical_comparison": legacy_canonical,
            "passed": trajectory_passed,
        }
        for branch_name, branch in (
            ("legacy", legacy),
            ("canonical", canonical),
            ("canonical_serialized", serialized),
        ):
            prefix = f"{trajectory}__{branch_name}__"
            for name in (
                "times",
                "states",
                "exports",
                "primitive_history",
                "mapped_history",
                "height_history",
                "previous_timestep",
            ):
                output_arrays[f"{prefix}{name}"] = branch[name]
        output_arrays[f"{trajectory}__canonical_final_state"] = canonical["restart"].primitive_charts
        output_arrays[f"{trajectory}__canonical_final_primitive_history"] = (
            canonical["restart"].history.previous_primitive_increment
        )
        output_arrays[f"{trajectory}__canonical_final_mapped_history"] = (
            canonical["restart"].history.previous_mapped_storage_increment
        )
        output_arrays[f"{trajectory}__canonical_final_height_history"] = (
            canonical[
                "restart"
            ].history.previous_responsive_height_storage_increment
        )
        output_arrays[f"{trajectory}__canonical_final_previous_timestep"] = np.asarray(
            canonical["restart"].history.previous_timestep_seconds
        )

    legacy_state_response = (
        branches["perturbed_legacy"]["states"]
        - branches["base_legacy"]["states"]
    )
    canonical_state_response = (
        branches["perturbed_canonical"]["states"]
        - branches["base_canonical"]["states"]
    )
    legacy_export_response = (
        branches["perturbed_legacy"]["exports"]
        - branches["base_legacy"]["exports"]
    )
    canonical_export_response = (
        branches["perturbed_canonical"]["exports"]
        - branches["base_canonical"]["exports"]
    )
    response = {
        "maximum_scaled_state_difference": _scaled_max(
            canonical_state_response,
            legacy_state_response,
            field_scales[None, None, :],
        ),
        "maximum_scaled_Tier_I_difference": _scaled_max(
            canonical_export_response,
            legacy_export_response,
            export_scales[None, :],
        ),
    }
    gates = manifest["binding_gates"]
    response["passed"] = bool(
        response["maximum_scaled_state_difference"]
        <= gates["canonical_response_maximum_scaled_state_difference_from_legacy"]
        and response["maximum_scaled_Tier_I_difference"]
        <= gates["canonical_response_maximum_scaled_Tier_I_difference_from_legacy"]
    )
    method_report = _method_report(branches, method_contract)
    passed = bool(
        method_report["passed"]
        and response["passed"]
        and all(item["passed"] for item in trajectory_reports.values())
    )
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
    output_arrays["legacy_state_response"] = legacy_state_response
    output_arrays["canonical_state_response"] = canonical_state_response
    output_arrays["legacy_export_response"] = legacy_export_response
    output_arrays["canonical_export_response"] = canonical_export_response

    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "layout": c2.LAYOUT,
        "profile": c2.PROFILE,
        "coupling_face": c2.COUPLING_FACE,
        "legacy_targets_seconds": c2e.LEGACY_TARGETS,
        "canonical_targets_seconds": c2e.CANONICAL_TARGETS,
        "trajectories": c2e.TRAJECTORIES,
    }
    _write_json(CONFIG_PATH, config)
    np.savez_compressed(DECISIVE_ARRAYS, **output_arrays)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "authorized_next": authorized_next,
        "parent_classification_preserved": parent["classification"],
        "historical_c2d_classification_preserved": (
            "second_rung_perturbed_completion_failed_later_duration_blocked"
        ),
        "operator_changed": False,
        "production_defaults_changed": False,
        "trajectory_reports": trajectory_reports,
        "response_comparison": response,
        "method_report": method_report,
        "elapsed_seconds": time.perf_counter() - started,
        "third_duration_rung_manifest_authorized": passed,
        "third_duration_rung_propagation_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "config_sha256": causal_canonical_json_sha256(_plain(config)),
        "decisive_arrays_sha256": _sha256(DECISIVE_ARRAYS),
        "decisive_array_hashes": {
            name: causal_array_sha256(value) for name, value in output_arrays.items()
        },
    }
    _write_json(SUMMARY_PATH, summary)
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "CERTIFIED" if passed else "REJECTED",
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
                for path in (THIS_RUNNER, THIS_TEST, c2e.THIS_RUNNER, c2d.THIS_RUNNER)
                if (ROOT / path).exists()
            },
            "input_hashes": {
                "parent_summary": _sha256(PARENT_DIRECTORY / "summary.json"),
                "parent_manifest": _sha256(c2e.MANIFEST_PATH),
                "c2d_decisive_arrays": _sha256(c2d.DECISIVE_ARRAYS),
            },
        },
    )
    REPORT_PATH.write_text(
        "# Canonical-time replay audit WP10c9d6c7c3b5c2e1\n\n"
        "## Classification\n\n"
        f"`{classification}`\n\n"
        f"Method gates pass: `{method_report['passed']}`.\n\n"
        f"Base paired replay passes: `{trajectory_reports['base']['passed']}`.\n\n"
        f"Perturbed paired replay passes: `{trajectory_reports['perturbed']['passed']}`.\n\n"
        f"Canonical response envelope passes: `{response['passed']}`.\n\n"
        "The historical c2d failure remains unchanged.\n\n"
        f"Authorized next: `{authorized_next}`.\n\n"
        "Fixed-Q experiments and reduced evolution remain blocked.\n",
        encoding="utf-8",
    )
    names = ("config.json", "summary.json", "provenance.json", "decisive_arrays.npz")
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
