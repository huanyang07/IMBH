#!/usr/bin/env python3
"""Localize the second-rung replay failure without new physical evolution."""

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
WORK_PACKAGE = "WP10c9d6c7c3b5c2b"
ANALYZED_BASE_COMMIT = "acaeacf18a71509a711e85ab3181dff50380aa5f"
ANALYZED_BASE_PARENT = "0d6df95057829f75c40004c44722d9ab664c81d1"
ANALYZED_BASE_TREE = "c20d9a677c16cc4b1c64072902d603bcef799d5d"
LAYOUT = c2.LAYOUT
COUPLING_FACE = c2.COUPLING_FACE
REPLAY_START_INDEX = 6
REPLAY_TIMESTEP_SECONDS = 1.0e-4
FRESH_PROCESS_ROUNDOFF_ENVELOPE = 1.0e-12

ARTIFACT = (
    "causal_inner_nonlinear_second_duration_replay_localization_"
    "wp10c9d6c7c3b5c2b"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_second_duration_replay_localization_"
    "wp10c9d6c7c3b5c2b.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_second_duration_replay_localization_"
    "wp10c9d6c7c3b5c2b.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_NONLINEAR_SECOND_DURATION_REPLAY_LOCALIZATION_"
    "WP10C9D6C7C3B5C2B_2026-08-02.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
PARENT_DIRECTORY = c2.CANONICAL_DIRECTORY


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
        parent["classification"]
        != "second_nonlinear_duration_rung_failed_later_duration_work_blocked"
        or parent["passed"]
        or parent["authorized_next"]
        != "WP10c9d6c7c3b5c2b_second_duration_rung_localization"
        or parent["trajectory_reports"]["base"]["split_restart_replay_bitwise"]
    ):
        raise RuntimeError("c2b parent classification changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c2b analyzed identity changed")
    return parent


def _roundtrip(context, restart):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "c2b_start.npz"
        save_causal_five_field_monolithic_bdf_restart(path, context, restart)
        restored = load_causal_five_field_monolithic_bdf_restart(
            path, context, expected_provenance=restart.provenance
        )
    return restored, causal_five_field_monolithic_bdf_restarts_equal(
        restart, restored
    )


def _run_replay(configuration, tangent, arrays, field_scales, export_scales):
    context = configuration["context"]
    times = arrays["base__times_seconds"]
    states = arrays["base__states"]
    exports = arrays["base__direct_exports"]
    previous = states[REPLAY_START_INDEX - 1]
    current = states[REPLAY_START_INDEX]
    storage = causal_five_field_monolithic_storage_increment(
        context, previous, current
    )
    history = causal_five_field_monolithic_bdf_history(
        current - previous, storage, REPLAY_TIMESTEP_SECONDS
    )
    restart = CausalFiveFieldMonolithicBDFRestart(
        primitive_charts=np.array(current, copy=True),
        history=history,
        elapsed_time_seconds=float(times[REPLAY_START_INDEX]),
        completed_steps=6,
        next_order=2,
        provenance={"work_package": WORK_PACKAGE, "kind": "replay_localization"},
    )
    restored, roundtrip = _roundtrip(context, restart)
    state = restored.primitive_charts
    history = restored.history
    replay_states = [np.array(state, copy=True)]
    replay_exports = [c3b5b._export_value(context, state, COUPLING_FACE)[0]]
    records = []
    for target_index in range(REPLAY_START_INDEX + 1, len(times)):
        step = advance_causal_five_field_monolithic_bdf(
            context,
            state,
            REPLAY_TIMESTEP_SECONDS,
            tangent,
            order=2,
            history=history,
            residual_tolerance=1.0e-10,
            ledger_tolerance=1.0e-12,
            maximum_scaled_primitive_change=5.0e-3,
        )
        record = c3b5b._step_record(step)
        records.append(record)
        if not c3b5b._step_passed(step, _read_json(c2.CONFIG_PATH)["main_controller"]):
            raise RuntimeError("c2b direct replay method gate failed")
        state = np.array(step.primitive_charts, copy=True)
        history = step.history
        replay_states.append(state)
        replay_exports.append(c3b5b._export_value(context, state, COUPLING_FACE)[0])
        print(
            f"c2b: target={times[target_index]:.8e} "
            f"state_bitwise={np.array_equal(state, states[target_index])}",
            flush=True,
        )
    replay_states = np.asarray(replay_states)
    replay_exports = np.asarray(replay_exports)
    main_states = states[REPLAY_START_INDEX:]
    main_exports = exports[REPLAY_START_INDEX:]
    return {
        "restart_roundtrip_bitwise": roundtrip,
        "all_steps_passed": all(item["accepted"] for item in records),
        "state_bitwise": np.array_equal(replay_states, main_states),
        "export_bitwise": np.array_equal(replay_exports, main_exports),
        "maximum_absolute_state_difference": float(
            np.max(np.abs(replay_states - main_states))
        ),
        "maximum_scaled_state_difference": float(
            np.max(
                np.abs(replay_states - main_states)
                / field_scales[None, None, :]
            )
        ),
        "maximum_absolute_export_difference": float(
            np.max(np.abs(replay_exports - main_exports))
        ),
        "maximum_scaled_export_difference": float(
            np.max(
                np.abs(replay_exports - main_exports)
                / export_scales[None, :]
            )
        ),
        "maximum_scaled_residual": max(
            item["maximum_scaled_residual"] for item in records
        ),
        "maximum_discrete_ledger_defect": max(
            item["maximum_discrete_ledger_defect"] for item in records
        ),
        "maximum_mapped_closure_defect": max(
            item["maximum_mapped_endpoint_path_closure_defect"]
            for item in records
        ),
        "storage_mapped_closure_defect": float(
            storage.maximum_mapped_path_closure_defect
        ),
        "storage_minimum_reconstruction_factor": float(
            storage.minimum_path_reconstruction_factor
        ),
    }, {
        "main_tail_states": main_states,
        "replay_tail_states": replay_states,
        "main_tail_exports": main_exports,
        "replay_tail_exports": replay_exports,
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
                    "scientific_status": "DIAGNOSTIC ONLY",
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
    arrays = _load_npz(PARENT_DIRECTORY / "decisive_arrays.npz")
    config = _read_json(c2.CONFIG_PATH)
    configured_times = np.asarray(config["output_times_seconds"])
    stored_times = arrays["base__times_seconds"]
    time_delta = stored_times - configured_times
    mismatch = np.flatnonzero(stored_times != configured_times)
    time_report = {
        "bitwise_equal": np.array_equal(stored_times, configured_times),
        "mismatch_indices": mismatch,
        "maximum_absolute_difference": float(np.max(np.abs(time_delta))),
        "maximum_spacing_units": float(
            np.max(
                np.abs(time_delta)
                / np.maximum(np.spacing(np.abs(configured_times)), np.finfo(float).tiny)
            )
        ),
    }
    pilot = _load_npz(c3b2b.CANONICAL_DIRECTORY / "decisive_arrays.npz")
    field_scales = pilot["field_scales"]
    export_scales = pilot["fixed_physical_observable_scales"]
    if "--package-existing" in sys.argv:
        if not SUMMARY_PATH.exists() or not DECISIVE_ARRAYS.exists():
            raise RuntimeError("existing localization evidence is absent")
        replay = _read_json(SUMMARY_PATH)["direct_replay_report"]
        decisive = _load_npz(DECISIVE_ARRAYS)
    else:
        configuration = c3b1a._configurations()[LAYOUT]
        print("c2b: build tangent", flush=True)
        tangent = causal_five_field_monolithic_frozen_tangent(
            configuration["context"],
            configuration["base"],
            primitive_column_scales=configuration["columns"],
            conservation_row_scales=configuration["rows"],
        )
        replay, decisive = _run_replay(
            configuration, tangent, arrays, field_scales, export_scales
        )
    fresh_replay_within_roundoff_envelope = bool(
        replay["maximum_scaled_state_difference"]
        <= FRESH_PROCESS_ROUNDOFF_ENVELOPE
        and replay["maximum_scaled_export_difference"]
        <= FRESH_PROCESS_ROUNDOFF_ENVELOPE
        and replay["all_steps_passed"]
        and replay["restart_roundtrip_bitwise"]
    )
    localized = bool(
        not time_report["bitwise_equal"]
        and time_report["maximum_spacing_units"] <= 1.0
        and fresh_replay_within_roundoff_envelope
    )
    classification = (
        "second_rung_replay_boolean_localized_to_one_ulp_time_label_"
        "fresh_process_replay_roundoff_scale_paired_replay_required"
        if localized
        else "second_rung_replay_failure_not_localized_later_duration_blocked"
    )
    authorized_next = (
        "WP10c9d6c7c3b5c2c_corrected_replay_contract_manifest"
        if localized
        else "none"
    )
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    run_config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "layout": LAYOUT,
        "coupling_face": COUPLING_FACE,
        "replay_start_index": REPLAY_START_INDEX,
        "replay_timestep_seconds": REPLAY_TIMESTEP_SECONDS,
        "fresh_process_roundoff_envelope": FRESH_PROCESS_ROUNDOFF_ENVELOPE,
    }
    _write_json(CONFIG_PATH, run_config)
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": localized,
        "authorized_next": authorized_next,
        "parent_classification_preserved": parent["classification"],
        "historical_replay_failure_preserved": True,
        "historical_combined_boolean_short_circuited_by_time_label": True,
        "historical_state_export_bitwise_status": "not_recorded_separately",
        "operator_changed": False,
        "production_defaults_changed": False,
        "time_label_report": time_report,
        "direct_replay_report": replay,
        "fresh_process_replay_within_roundoff_envelope": (
            fresh_replay_within_roundoff_envelope
        ),
        "corrected_replay_manifest_authorized": localized,
        "later_duration_rungs_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "config_sha256": causal_canonical_json_sha256(_plain(run_config)),
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
            "scientific_status": "DIAGNOSTIC ONLY",
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
                for path in (THIS_RUNNER, THIS_TEST, c2.THIS_RUNNER)
                if (ROOT / path).exists()
            },
            "input_hashes": {
                "parent_summary": _sha256(PARENT_DIRECTORY / "summary.json"),
                "parent_arrays": _sha256(PARENT_DIRECTORY / "decisive_arrays.npz"),
            },
        },
    )
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Second-duration replay localization WP10c9d6c7c3b5c2b",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                "The historical replay failure is preserved. Its combined Boolean "
                "short-circuited on the time comparison, so historical state/export "
                "bitwise status was not recorded separately. The stored/global "
                f"time labels differ by `{time_report['maximum_absolute_difference']:.3e} s` "
                f"(`{time_report['maximum_spacing_units']:.1f}` ULP). A fresh-process "
                "replay is not bitwise equal to the old committed trajectory, but its "
                f"maximum scaled state/export differences are "
                f"`{replay['maximum_scaled_state_difference']:.3e}` / "
                f"`{replay['maximum_scaled_export_difference']:.3e}`, below the "
                f"diagnostic `{FRESH_PROCESS_ROUNDOFF_ENVELOPE:.1e}` roundoff envelope.",
                "",
                f"Authorized next: `{authorized_next}`.",
                "",
                "Later duration, fixed-Q, and reduced evolution remain blocked.",
                "",
            ]
        )
    )
    names = ("config.json", "summary.json", "provenance.json", "decisive_arrays.npz")
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names)
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0 if localized else 1


if __name__ == "__main__":
    raise SystemExit(main())
