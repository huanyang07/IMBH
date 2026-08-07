#!/usr/bin/env python3
"""Execute the evidence-selected fine 5 ms completion campaign."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import sys

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_nonlinear_middle_1ms_continuation_wp10c9d6c7c3b5c3h2b1 as h2b1  # noqa: E402
import run_causal_inner_nonlinear_middle_spatial_analysis_and_fine_manifest_wp10c9d6c7c3b5c3h2e0 as h2e0  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c3h2e1"
ANALYZED_BASE_COMMIT = "2b64fd4a3943f2dbbc8c8a1a992658c748bddaeb"
ANALYZED_BASE_PARENT = "e3728c4c6fe7148aeca9ca80e10a7c8d97095c24"
ANALYZED_BASE_TREE = "a0a7a616a65ad08c9b9cc9d7998b8b60447f7b0f"

FINE_LAYOUT = h2e0.FINE_LAYOUT
COUPLING_FACE = int(
    h2e0.g1.c3g.ACTIVE_COUPLING_FACE_INDICES[FINE_LAYOUT]
)
PROFILES = tuple(h2e0.PROFILES)
GENERIC_PROFILE = h2e0.GENERIC_PROFILE
GENERIC_INDEX = PROFILES.index(GENERIC_PROFILE)
TARGET_MICROSECONDS = tuple(h2e0.FINE_TARGET_MICROSECONDS)
AUDIT_TARGET_MICROSECONDS = tuple(h2e0.FINE_AUDIT_TARGET_MICROSECONDS)
START_SECONDS = 40.0e-6
STOP_SECONDS = 5.0e-3
INITIAL_PREVIOUS_TIMESTEP_SECONDS = 1.0e-5

ARTIFACT = "causal_inner_nonlinear_fine_5ms_completion_wp10c9d6c7c3b5c3h2e1"
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_fine_5ms_completion_"
    "wp10c9d6c7c3b5c3h2e1.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_fine_5ms_completion_"
    "wp10c9d6c7c3b5c3h2e1.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_FINE_5MS_COMPLETION_"
    "WP10C9D6C7C3B5C3H2E1_2026-08-06.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
CHECKPOINT_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
PROGRESS_PATH = CHECKPOINT_DIRECTORY / "progress.json"
BASE_PATH = CHECKPOINT_DIRECTORY / "base.npz"
TANGENT_PATH = CHECKPOINT_DIRECTORY / "tangent.npz"
ANCHOR_PATH = CHECKPOINT_DIRECTORY / "anchor.npz"


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


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_identity() -> dict[str, str]:
    return {
        path: _sha256(ROOT / path)
        for path in (
            THIS_RUNNER,
            THIS_TEST,
            h2b1.CONTROLLER_RELATIVE,
            h2b1.MODULE_RELATIVE,
        )
        if (ROOT / path).exists()
    }


def _validate_parent() -> None:
    parent = _read_json(h2e0.SUMMARY_PATH)
    if (
        not parent["passed"]
        or not parent["fine_cost_bounded_propagation_authorized"]
        or not parent["full_fine_generic_nonlinear_anchor_required"]
        or parent["middle_fine_5ms_spatial_certificate_issued"]
    ):
        raise RuntimeError("h2e1 authorization changed")
    if (
        h2b1._git_value("rev-parse", ANALYZED_BASE_COMMIT)
        != ANALYZED_BASE_COMMIT
        or h2b1._git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or h2b1._git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("h2e1 analyzed identity changed")


def _short_inputs():
    base_arrays = h2b1._load_npz(h2b1.h2a2.h2.h1.b1a.DECISIVE_ARRAYS)
    spatial_arrays = h2b1._load_npz(h2b1.h2a2.h2.h1.b4b3.DECISIVE_ARRAYS)
    scale_arrays = h2b1._load_npz(h2b1.h2a2.h2.h1.c3d.DECISIVE_ARRAYS)
    base = np.asarray(base_arrays[f"{FINE_LAYOUT}__states"], dtype=float)
    perturbed = []
    for profile in PROFILES:
        task = f"{FINE_LAYOUT}__{profile}__p1__dt_1e-5"
        perturbed.append(np.asarray(spatial_arrays[f"{task}__states"], dtype=float))
    return (
        base,
        np.asarray(perturbed, dtype=float),
        np.asarray(scale_arrays["field_scales"], dtype=float),
        np.asarray(scale_arrays["export_scales"], dtype=float),
    )


def _fine_initial_base(configuration: dict):
    base, _perturbed, _field_scales, _export_scales = _short_inputs()
    context = configuration["context"]
    history = h2b1.h2a2.causal_five_field_monolithic_bdf_history_from_interval(
        context,
        base[3],
        base[4],
        INITIAL_PREVIOUS_TIMESTEP_SECONDS,
    )
    state = np.asarray(base[4], dtype=float)
    value, ledger, incoming = h2b1.controller._export_value(
        context, state, COUPLING_FACE
    )
    contract, _strict = h2b1.h2a2.h2.g._controller_contracts()
    arrays = {
        "accepted_times": np.asarray([START_SECONDS]),
        "accepted_timesteps": np.empty(0, dtype=float),
        "accepted_states": state[None, ...],
        "accepted_primitive_histories": history.previous_primitive_increment[None, ...],
        "accepted_mapped_histories": history.previous_mapped_storage_increment[None, ...],
        "accepted_height_histories": (
            history.previous_responsive_height_storage_increment[None, ...]
        ),
        "accepted_previous_timesteps": np.asarray(
            [history.previous_timestep_seconds], dtype=float
        ),
        "accepted_step_wall_seconds": np.empty(0, dtype=float),
        "local_state_estimates": np.empty(0, dtype=float),
        "local_export_estimates": np.empty(0, dtype=float),
        "local_error_estimates": np.empty(0, dtype=float),
        "retries": np.empty(0, dtype=np.int64),
        "output_times": np.asarray([START_SECONDS]),
        "output_states": state[None, ...],
        "output_exports": np.asarray(value, dtype=float)[None, :],
        "next_candidate_timestep": np.asarray(
            [contract["initial_timestep_seconds"]], dtype=float
        ),
    }
    report = {
        "passed_so_far": bool(ledger <= 1.0e-9 and incoming == 0),
        "accepted_steps": 0,
        "rejected_attempts": 0,
        "wall_seconds": 0.0,
        "maximum_scaled_residual": 0.0,
        "maximum_discrete_ledger_defect": 0.0,
        "maximum_mapped_endpoint_path_closure_defect": 0.0,
        "minimum_path_reconstruction_factor": 1.0,
        "maximum_incoming_excision_characteristics": incoming,
        "maximum_export_ledger_defect": ledger,
    }
    return arrays, report


def _fine_initial_tangent():
    base, perturbed, field_scales, export_scales = _short_inputs()
    configuration = h2b1._configuration()
    context = configuration["context"]
    response = perturbed - base[None, ...]
    matrix = h2b1.causal_five_field_monolithic_discrete_step_matrix(
        context,
        base[3],
        base[4],
        INITIAL_PREVIOUS_TIMESTEP_SECONDS,
        INITIAL_PREVIOUS_TIMESTEP_SECONDS,
        primitive_column_scales=configuration["columns"],
        conservation_row_scales=configuration["rows"],
    )
    direction = np.asarray(response[:, 4], dtype=float)
    history_direction = h2b1.h2a2.causal_five_field_monolithic_bdf_history_direction(
        context,
        base[3],
        base[4],
        response[:, 3],
        response[:, 4],
        analytic_step_matrix=matrix,
    )
    exports, audit = (
        h2b1.causal_five_field_monolithic_discrete_export_directions(
            matrix, direction, COUPLING_FACE
        )
    )
    arrays = {
        "state_directions": direction[None, ...],
        "export_directions": np.asarray(exports, dtype=float)[None, ...],
        "primitive_history_directions": (
            history_direction.previous_primitive_increment[None, ...]
        ),
        "mapped_history_directions": (
            history_direction.previous_mapped_storage_increment[None, ...]
        ),
        "height_history_directions": (
            history_direction.previous_responsive_height_storage_increment[None, ...]
        ),
        "matrix_assembly_wall_seconds": np.empty(0, dtype=float),
        "block_step_wall_seconds": np.empty(0, dtype=float),
        "audit_flags": np.empty(0, dtype=bool),
        "step_ratios": np.empty(0, dtype=float),
        "field_scales": field_scales,
        "export_scales": export_scales,
    }
    report = {
        "maximum_step_matrix_jvp_relative_defect": 0.0,
        "maximum_linear_solve_relative_defect": 0.0,
        "maximum_matrix_component_closure_defect": matrix.maximum_component_closure_defect,
        "maximum_incoming_excision_characteristics": (
            matrix.incoming_excision_characteristics
        ),
        "maximum_export_active_prefix_ledger_defect": (
            audit.active_prefix_ledger_defect
        ),
        "maximum_export_transport_telescoping_defect": (
            audit.conservative_transport_telescoping_defect
        ),
    }
    return arrays, report


def _fine_initial_anchor(configuration: dict, base_arrays: dict[str, np.ndarray]):
    base, perturbed, _field_scales, _export_scales = _short_inputs()
    context = configuration["context"]
    generic = np.asarray(perturbed[GENERIC_INDEX], dtype=float)
    history = h2b1.h2a2.causal_five_field_monolithic_bdf_history_from_interval(
        context,
        generic[3],
        generic[4],
        INITIAL_PREVIOUS_TIMESTEP_SECONDS,
    )
    state = np.asarray(generic[4], dtype=float)
    base_value, base_ledger, base_incoming = h2b1.controller._export_value(
        context, base_arrays["accepted_states"][0], COUPLING_FACE
    )
    anchor_value, anchor_ledger, anchor_incoming = h2b1.controller._export_value(
        context, state, COUPLING_FACE
    )
    arrays = {
        "anchor_states": state[None, ...],
        "anchor_primitive_histories": history.previous_primitive_increment[None, ...],
        "anchor_mapped_histories": history.previous_mapped_storage_increment[None, ...],
        "anchor_height_histories": (
            history.previous_responsive_height_storage_increment[None, ...]
        ),
        "anchor_previous_timesteps": np.asarray(
            [history.previous_timestep_seconds], dtype=float
        ),
        "anchor_predictors": np.empty((0, *state.shape), dtype=float),
        "anchor_step_wall_seconds": np.empty(0, dtype=float),
        "sampled_flags": np.empty(0, dtype=bool),
        "sampled_state_error_estimates": np.empty(0, dtype=float),
        "sampled_export_error_estimates": np.empty(0, dtype=float),
        "base_exports": np.asarray(base_value, dtype=float)[None, :],
        "anchor_exports": np.asarray(anchor_value, dtype=float)[None, :],
    }
    report = {
        "maximum_scaled_residual": 0.0,
        "maximum_discrete_ledger_defect": 0.0,
        "maximum_mapped_endpoint_path_closure_defect": 0.0,
        "minimum_path_reconstruction_factor": 1.0,
        "maximum_incoming_excision_characteristics": 0,
        "maximum_export_ledger_defect": max(base_ledger, anchor_ledger),
        "maximum_export_incoming_characteristics": max(
            base_incoming, anchor_incoming
        ),
    }
    return arrays, report


def _audit_indices(base: dict[str, np.ndarray]) -> set[int]:
    endpoint_us = np.rint(base["accepted_times"][1:] * 1.0e6).astype(int)
    indices = {
        int(index)
        for index, value in enumerate(endpoint_us)
        if int(value) in AUDIT_TARGET_MICROSECONDS
    }
    indices.update((0, int(endpoint_us.size - 1)))
    return indices


def _patch_shared_module() -> None:
    h2b1.WORK_PACKAGE = WORK_PACKAGE
    h2b1.ARTIFACT = ARTIFACT
    h2b1.MIDDLE_LAYOUT = FINE_LAYOUT
    h2b1.COUPLING_FACE = COUPLING_FACE
    h2b1.TARGET_MICROSECONDS = TARGET_MICROSECONDS
    h2b1.START_SECONDS = START_SECONDS
    h2b1.STOP_SECONDS = STOP_SECONDS
    h2b1.CHECKPOINT_DIRECTORY = CHECKPOINT_DIRECTORY
    h2b1.PROGRESS_PATH = PROGRESS_PATH
    h2b1.BASE_PATH = BASE_PATH
    h2b1.TANGENT_PATH = TANGENT_PATH
    h2b1.ANCHOR_PATH = ANCHOR_PATH
    h2b1._source_identity = _source_identity
    h2b1._initial_base = _fine_initial_base
    h2b1._initial_tangent = _fine_initial_tangent
    h2b1._initial_anchor = _fine_initial_anchor
    h2b1._tangent_audit_indices = _audit_indices
    h2b1._anchor_sample_indices = _audit_indices


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
    _validate_parent()
    _patch_shared_module()
    progress = h2b1._progress()
    configuration = h2b1._configuration()
    print("h2e1: build fine frozen nonlinear tangent", flush=True)
    frozen_tangent, setup_seconds = h2b1._build_frozen_tangent(configuration)
    _base_short, _perturbed_short, field_scales, export_scales = _short_inputs()
    contract, _strict = h2b1.h2a2.h2.g._controller_contracts()
    base_report, base = h2b1._run_base_targets(
        progress,
        configuration,
        frozen_tangent,
        field_scales,
        export_scales,
        contract,
    )
    tangent_report, tangent = h2b1._run_tangent(progress, configuration, base)
    anchor_report, anchor = h2b1._run_anchor(
        progress,
        configuration,
        frozen_tangent,
        base,
        tangent,
        field_scales,
        export_scales,
        contract,
    )
    replays = {
        "base": h2b1._serialized_last_step_replay(
            "base",
            configuration,
            frozen_tangent,
            base["accepted_states"],
            base["accepted_primitive_histories"],
            base["accepted_mapped_histories"],
            base["accepted_height_histories"],
            base["accepted_previous_timesteps"],
            base["accepted_timesteps"],
            base["accepted_times"],
            None,
        ),
        "anchor": h2b1._serialized_last_step_replay(
            "anchor",
            configuration,
            frozen_tangent,
            anchor["anchor_states"],
            anchor["anchor_primitive_histories"],
            anchor["anchor_mapped_histories"],
            anchor["anchor_height_histories"],
            anchor["anchor_previous_timesteps"],
            base["accepted_timesteps"],
            base["accepted_times"],
            anchor["anchor_predictors"][-1],
        ),
    }
    replay_passed = all(
        item["checkpoint_roundtrip_bitwise"]
        and item["last_step_replay_bitwise"]
        and item["maximum_scaled_residual"] <= 1.0e-10
        for item in replays.values()
    )
    passed = bool(
        base_report["passed"]
        and tangent_report["passed"]
        and anchor_report["passed"]
        and replay_passed
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            "fine_5ms_completion_passed_final_spatial_certificate_analysis_authorized"
            if passed
            else "fine_5ms_completion_failed_spatial_certificate_blocked"
        ),
        "passed": passed,
        "setup_wall_seconds": setup_seconds,
        "base": base_report,
        "tangent": tangent_report,
        "anchor": anchor_report,
        "serialized_replays": replays,
        "full_fine_generic_anchor_executed": True,
        "final_spatial_certificate_analysis_authorized": passed,
        "middle_fine_5ms_spatial_certificate_issued": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": (
            "WP10c9d6c7c3b5c3h2f_5ms_spatial_certificate"
            if passed
            else None
        ),
    }
    combined = {
        **{f"base__{key}": value for key, value in base.items()},
        **{f"tangent__{key}": value for key, value in tangent.items()},
        **{f"anchor__{key}": value for key, value in anchor.items()},
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "layout": FINE_LAYOUT,
            "profiles": PROFILES,
            "generic_profile": GENERIC_PROFILE,
            "coupling_face": COUPLING_FACE,
            "target_microseconds": TARGET_MICROSECONDS,
            "audit_target_microseconds": AUDIT_TARGET_MICROSECONDS,
            "controller": contract,
        },
    )
    np.savez_compressed(DECISIVE_ARRAYS, **combined)
    _write_json(SUMMARY_PATH, summary)
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "working_head": h2b1._git_value("rev-parse", "HEAD"),
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "input_hashes": {
                "fine_manifest": _sha256(h2e0.MANIFEST_PATH),
                "parent_summary": _sha256(h2e0.SUMMARY_PATH),
                "short_base_arrays": _sha256(
                    h2b1.h2a2.h2.h1.b1a.DECISIVE_ARRAYS
                ),
                "short_profile_arrays": _sha256(
                    h2b1.h2a2.h2.h1.b4b3.DECISIVE_ARRAYS
                ),
            },
            "implementation_source_hashes": _source_identity(),
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
                "# Fine 5 ms completion WP10c9d6c7c3b5c3h2e1",
                "",
                "## Classification",
                "",
                f"`{summary['classification']}`",
                "",
                f"The fine nonlinear base, five-profile discrete tangent, and "
                f"evidence-required generic nonlinear anchor reached 5 ms in "
                f"`{base_report['accepted_steps']}` accepted base steps with "
                f"`{base_report['rejected_attempts']}` rejected attempts.",
                "",
                f"The fine tangent/nonlinear discrepancy is "
                f"`{anchor_report['state']['discrepancy_fraction_of_observable_response']:.6e}` "
                "of the generic state response and "
                f"`{anchor_report['instantaneous_Tier_I']['discrepancy_fraction_of_observable_response']:.6e}` "
                "of the instantaneous Tier-I response. Base and anchor last-step "
                "replays are bitwise.",
                "",
                "A pass authorizes only final coarse/middle/fine spatial analysis. "
                "Fixed-Q experiments and reduced slow evolution remain blocked.",
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
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
