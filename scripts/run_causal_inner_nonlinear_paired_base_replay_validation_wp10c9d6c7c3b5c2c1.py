#!/usr/bin/env python3
"""Run the frozen same-tangent paired base replay validation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_monolithic_bdf_base_preflight_wp10c9d6c7c3b1a as c3b1a  # noqa: E402
import run_causal_inner_nonlinear_corrected_replay_contract_manifest_wp10c9d6c7c3b5c2c as c2c  # noqa: E402
import run_causal_inner_nonlinear_duration_controller_validation_wp10c9d6c7c3b5b as c3b5b  # noqa: E402
import run_causal_inner_nonlinear_second_duration_rung_wp10c9d6c7c3b5c2 as c2  # noqa: E402
import run_causal_inner_nonlinear_spatial_export_pilot_wp10c9d6c7c3b2b as c3b2b  # noqa: E402

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    CausalFiveFieldMonolithicBDFRestart,
    advance_causal_five_field_monolithic_bdf,
    causal_five_field_monolithic_bdf_history,
    causal_five_field_monolithic_bdf_restarts_equal,
    causal_five_field_monolithic_frozen_tangent,
    causal_five_field_monolithic_storage_increment,
    load_causal_five_field_monolithic_bdf_restart,
    save_causal_five_field_monolithic_bdf_restart,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_canonical_json_sha256,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c2c1"
ANALYZED_BASE_COMMIT = "2577f42a2252bc1822d4c1e1de729767864ba31a"
ANALYZED_BASE_PARENT = "d282f4a43f78b045d0911923fd1ca82aa1211a85"
ANALYZED_BASE_TREE = "744bd9ad9e5efc16d7d67a507d9f6fffdd430265"
LAYOUT = c2c.LAYOUT
COUPLING_FACE = c2c.COUPLING_FACE
REPLAY_TARGETS = c2c.REPLAY_TARGETS_SECONDS
TIMESTEP = c2c.REPLAY_TIMESTEP_SECONDS
START_INDEX = 6

ARTIFACT = (
    "causal_inner_nonlinear_paired_base_replay_validation_"
    "wp10c9d6c7c3b5c2c1"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_paired_base_replay_validation_"
    "wp10c9d6c7c3b5c2c1.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_paired_base_replay_validation_"
    "wp10c9d6c7c3b5c2c1.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_NONLINEAR_PAIRED_BASE_REPLAY_VALIDATION_"
    "WP10C9D6C7C3B5C2C1_2026-08-02.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
PARENT_DIRECTORY = c2c.CANONICAL_DIRECTORY


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


def _validate_parent() -> dict:
    parent = _read_json(PARENT_DIRECTORY / "summary.json")
    if (
        not parent["passed"]
        or not parent["paired_base_replay_validation_authorized"]
        or parent["authorized_next"]
        != "WP10c9d6c7c3b5c2c1_paired_base_replay_validation"
        or parent["propagation_executed"]
    ):
        raise RuntimeError("c2c1 paired replay authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c2c1 analyzed identity changed")
    return parent


def _initial_restart(context, committed: dict[str, np.ndarray]):
    times = committed["base__times_seconds"]
    states = committed["base__states"]
    previous = states[START_INDEX - 1]
    current = states[START_INDEX]
    storage = causal_five_field_monolithic_storage_increment(context, previous, current)
    history = causal_five_field_monolithic_bdf_history(
        current - previous, storage, TIMESTEP
    )
    return CausalFiveFieldMonolithicBDFRestart(
        primitive_charts=np.array(current, copy=True),
        history=history,
        elapsed_time_seconds=float(REPLAY_TARGETS[0]),
        completed_steps=6,
        next_order=2,
        provenance={"work_package": WORK_PACKAGE, "branch": "paired_base"},
    ), storage


def _serialize_roundtrip(context, restart):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "paired_start.npz"
        save_causal_five_field_monolithic_bdf_restart(path, context, restart)
        restored = load_causal_five_field_monolithic_bdf_restart(
            path, context, expected_provenance=restart.provenance
        )
    return restored, causal_five_field_monolithic_bdf_restarts_equal(restart, restored)


def _branch(context, tangent, restart, contract, label: str):
    state = np.array(restart.primitive_charts, copy=True)
    history = restart.history
    accumulated = float(restart.elapsed_time_seconds)
    restarts = [restart]
    states = [np.array(state, copy=True)]
    exports = [c3b5b._export_value(context, state, COUPLING_FACE)[0]]
    accumulated_times = [accumulated]
    records = []
    for index, target in enumerate(REPLAY_TARGETS[1:], start=1):
        step = advance_causal_five_field_monolithic_bdf(
            context,
            state,
            TIMESTEP,
            tangent,
            order=2,
            history=history,
            residual_tolerance=1.0e-10,
            ledger_tolerance=1.0e-12,
            maximum_scaled_primitive_change=5.0e-3,
        )
        record = c3b5b._step_record(step)
        records.append(record)
        if not c3b5b._step_passed(step, contract):
            raise RuntimeError(f"{label} paired replay method gate failed")
        state = np.array(step.primitive_charts, copy=True)
        history = step.history
        accumulated += TIMESTEP
        states.append(state)
        exports.append(c3b5b._export_value(context, state, COUPLING_FACE)[0])
        accumulated_times.append(accumulated)
        restarts.append(
            CausalFiveFieldMonolithicBDFRestart(
                primitive_charts=np.array(state, copy=True),
                history=history,
                elapsed_time_seconds=float(target),
                completed_steps=6 + index,
                next_order=2,
                provenance=restart.provenance,
            )
        )
        print(f"c2c1: {label} target={target:.8e}", flush=True)
    return {
        "canonical_times": np.array(REPLAY_TARGETS, copy=True),
        "accumulated_times": np.asarray(accumulated_times),
        "states": np.asarray(states),
        "exports": np.asarray(exports),
        "primitive_history": np.asarray(
            [item.history.previous_primitive_increment for item in restarts]
        ),
        "mapped_history": np.asarray(
            [item.history.previous_mapped_storage_increment for item in restarts]
        ),
        "height_history": np.asarray(
            [
                item.history.previous_responsive_height_storage_increment
                for item in restarts
            ]
        ),
        "previous_timesteps": np.asarray(
            [item.history.previous_timestep_seconds for item in restarts]
        ),
        "restarts": restarts,
        "records": records,
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
    manifest = _read_json(c2c.MANIFEST_PATH)
    committed = _load_npz(c2.CANONICAL_DIRECTORY / "decisive_arrays.npz")
    pilot = _load_npz(c3b2b.CANONICAL_DIRECTORY / "decisive_arrays.npz")
    field_scales = pilot["field_scales"]
    export_scales = pilot["fixed_physical_observable_scales"]
    configuration = c3b1a._configurations()[LAYOUT]
    context = configuration["context"]
    print("c2c1: build shared tangent", flush=True)
    tangent = causal_five_field_monolithic_frozen_tangent(
        context,
        configuration["base"],
        primitive_column_scales=configuration["columns"],
        conservation_row_scales=configuration["rows"],
    )
    initial, storage = _initial_restart(context, committed)
    restored, initial_roundtrip = _serialize_roundtrip(context, initial)
    contract = _read_json(c2.CONFIG_PATH)["main_controller"]
    direct = _branch(context, tangent, initial, contract, "direct")
    serialized = _branch(context, tangent, restored, contract, "serialized")
    separate = {
        "canonical_time_labels_bitwise": np.array_equal(
            direct["canonical_times"], serialized["canonical_times"]
        ),
        "primitive_states_bitwise": np.array_equal(
            direct["states"], serialized["states"]
        ),
        "direct_Tier_I_exports_bitwise": np.array_equal(
            direct["exports"], serialized["exports"]
        ),
        "primitive_history_bitwise": np.array_equal(
            direct["primitive_history"], serialized["primitive_history"]
        ),
        "mapped_history_bitwise": np.array_equal(
            direct["mapped_history"], serialized["mapped_history"]
        ),
        "height_history_bitwise": np.array_equal(
            direct["height_history"], serialized["height_history"]
        ),
        "previous_timesteps_bitwise": np.array_equal(
            direct["previous_timesteps"], serialized["previous_timesteps"]
        ),
        "complete_restarts_bitwise_each_target": all(
            causal_five_field_monolithic_bdf_restarts_equal(left, right)
            for left, right in zip(direct["restarts"], serialized["restarts"])
        ),
        "initial_restart_roundtrip_bitwise": initial_roundtrip,
    }
    all_records = direct["records"] + serialized["records"]
    accumulated_delta = np.maximum(
        np.abs(direct["accumulated_times"] - REPLAY_TARGETS),
        np.abs(serialized["accumulated_times"] - REPLAY_TARGETS),
    )
    accumulated_ulp = float(
        np.max(
            accumulated_delta
            / np.maximum(np.spacing(np.abs(REPLAY_TARGETS)), np.finfo(float).tiny)
        )
    )
    committed_states = committed["base__states"][START_INDEX:]
    committed_exports = committed["base__direct_exports"][START_INDEX:]
    explanatory = {
        "maximum_scaled_state_difference": float(
            np.max(
                np.abs(direct["states"] - committed_states)
                / field_scales[None, None, :]
            )
        ),
        "maximum_scaled_export_difference": float(
            np.max(
                np.abs(direct["exports"] - committed_exports)
                / export_scales[None, :]
            )
        ),
    }
    method = {
        "all_steps_accepted": all(item["accepted"] for item in all_records),
        "maximum_scaled_residual": max(
            item["maximum_scaled_residual"] for item in all_records
        ),
        "maximum_discrete_ledger_defect": max(
            item["maximum_discrete_ledger_defect"] for item in all_records
        ),
        "maximum_mapped_closure_defect": max(
            item["maximum_mapped_endpoint_path_closure_defect"]
            for item in all_records
        ),
        "minimum_reconstruction_factor": min(
            item["minimum_path_reconstruction_factor"] for item in all_records
        ),
        "maximum_incoming_excision_characteristics": max(
            item["incoming_excision_characteristics"] for item in all_records
        ),
        "initial_storage_mapped_closure_defect": float(
            storage.maximum_mapped_path_closure_defect
        ),
    }
    paired_bitwise = bool(all(separate.values()))
    passed = bool(
        paired_bitwise
        and accumulated_ulp <= 1.0
        and method["all_steps_accepted"]
        and method["maximum_scaled_residual"] <= 1.0e-10
        and method["maximum_discrete_ledger_defect"] <= 1.0e-12
        and method["maximum_mapped_closure_defect"] <= 1.0e-9
        and method["minimum_reconstruction_factor"] >= 1.0
        and method["maximum_incoming_excision_characteristics"] == 0
        and explanatory["maximum_scaled_state_difference"] <= 1.0e-12
        and explanatory["maximum_scaled_export_difference"] <= 1.0e-12
    )
    classification = (
        "paired_base_replay_bitwise_certified_"
        "missing_perturbed_second_rung_authorized"
        if passed
        else "paired_base_replay_failed_perturbed_and_later_duration_blocked"
    )
    authorized_next = (
        "WP10c9d6c7c3b5c2d_second_rung_perturbed_completion"
        if passed
        else "WP10c9d6c7c3b5c2c2_paired_replay_localization"
    )
    decisive = {
        "canonical_times_seconds": direct["canonical_times"],
        "direct_accumulated_times_seconds": direct["accumulated_times"],
        "serialized_accumulated_times_seconds": serialized["accumulated_times"],
        "direct_states": direct["states"],
        "serialized_states": serialized["states"],
        "direct_exports": direct["exports"],
        "serialized_exports": serialized["exports"],
        "direct_primitive_history": direct["primitive_history"],
        "serialized_primitive_history": serialized["primitive_history"],
        "direct_mapped_history": direct["mapped_history"],
        "serialized_mapped_history": serialized["mapped_history"],
        "direct_height_history": direct["height_history"],
        "serialized_height_history": serialized["height_history"],
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "layout": LAYOUT,
        "coupling_face": COUPLING_FACE,
        "canonical_targets_seconds": REPLAY_TARGETS,
        "fixed_timestep_seconds": TIMESTEP,
        "manifest_sha256": causal_canonical_json_sha256(manifest),
    }
    _write_json(CONFIG_PATH, config)
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "authorized_next": authorized_next,
        "parent_classification_preserved": parent["classification"],
        "historical_c2_failure_preserved": True,
        "operator_changed": False,
        "production_defaults_changed": False,
        "paired_replay_report": separate,
        "paired_replay_bitwise": paired_bitwise,
        "maximum_accumulated_time_spacing_units": accumulated_ulp,
        "fresh_process_committed_main_comparison": explanatory,
        "method_report": method,
        "perturbed_second_rung_authorized": passed,
        "later_duration_rungs_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "config_sha256": causal_canonical_json_sha256(_plain(config)),
        "decisive_arrays_sha256": _sha256(DECISIVE_ARRAYS),
        "decisive_array_hashes": {
            name: causal_array_sha256(value) for name, value in decisive.items()
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
                for path in (THIS_RUNNER, THIS_TEST, c2c.THIS_RUNNER, c2.THIS_RUNNER)
                if (ROOT / path).exists()
            },
            "input_hashes": {
                "parent_summary": _sha256(PARENT_DIRECTORY / "summary.json"),
                "parent_manifest": _sha256(c2c.MANIFEST_PATH),
                "historical_arrays": _sha256(
                    c2.CANONICAL_DIRECTORY / "decisive_arrays.npz"
                ),
            },
        },
    )
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Paired base replay validation WP10c9d6c7c3b5c2c1",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                f"All separate bitwise replay gates pass: `{paired_bitwise}`.",
                f"Maximum accumulated-time offset is `{accumulated_ulp:.1f}` ULP.",
                f"Worst residual is `{method['maximum_scaled_residual']:.3e}`.",
                "",
                f"Authorized next: `{authorized_next}`.",
                "",
                "The historical c2 failure remains unchanged. The third duration rung,",
                "fixed-Q experiments, and reduced evolution remain blocked.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    names = ("config.json", "summary.json", "provenance.json", "decisive_arrays.npz")
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names)
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
