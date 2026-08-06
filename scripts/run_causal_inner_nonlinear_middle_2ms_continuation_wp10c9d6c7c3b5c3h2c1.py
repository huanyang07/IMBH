#!/usr/bin/env python3
"""Execute the cost-bounded middle continuation from 1 to 2 ms."""

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

import run_causal_inner_nonlinear_middle_1ms_continuation_wp10c9d6c7c3b5c3h2b1 as h2b1  # noqa: E402
import run_causal_inner_nonlinear_middle_2ms_continuation_manifest_wp10c9d6c7c3b5c3h2c0 as h2c0  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c3h2c1"
ANALYZED_BASE_COMMIT = "a55083606a65948fe296a3f22a85c29ca6273bf9"
ANALYZED_BASE_PARENT = "ea3afa95ca57d8dabd6cde8a9c31ff53bf179662"
ANALYZED_BASE_TREE = "5f1222c5b2637dc8636882e12114140e4d57319d"
TARGET_MICROSECONDS = tuple(h2c0.TARGET_MICROSECONDS)
ARTIFACT = (
    "causal_inner_nonlinear_middle_2ms_continuation_"
    "wp10c9d6c7c3b5c3h2c1"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_middle_2ms_continuation_"
    "wp10c9d6c7c3b5c3h2c1.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_middle_2ms_continuation_"
    "wp10c9d6c7c3b5c3h2c1.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_MIDDLE_2MS_"
    "CONTINUATION_WP10C9D6C7C3B5C3H2C1_2026-08-06.md"
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
CONTROLLER_RELATIVE = h2b1.CONTROLLER_RELATIVE
MODULE_RELATIVE = h2b1.MODULE_RELATIVE


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
    path.write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _source_identity() -> dict[str, str]:
    return {
        path: _sha256(ROOT / path)
        for path in (THIS_RUNNER, THIS_TEST, CONTROLLER_RELATIVE, MODULE_RELATIVE)
        if (ROOT / path).exists()
    }


def _validate_parent() -> None:
    manifest = _read_json(h2c0.SUMMARY_PATH)
    parent = _read_json(h2b1.SUMMARY_PATH)
    if (
        not manifest["passed"]
        or not manifest["middle_2ms_propagation_authorized"]
        or not parent["passed"]
        or not parent["middle_2ms_continuation_manifest_authorized"]
    ):
        raise RuntimeError("h2c1 authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("h2c1 analyzed identity changed")


def _load_parent_arrays() -> dict[str, np.ndarray]:
    with np.load(h2b1.DECISIVE_ARRAYS, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _patch_shared_module() -> None:
    h2b1.WORK_PACKAGE = WORK_PACKAGE
    h2b1.TARGET_MICROSECONDS = TARGET_MICROSECONDS
    h2b1.START_SECONDS = 1.0e-3
    h2b1.STOP_SECONDS = 2.0e-3
    h2b1.ARTIFACT = ARTIFACT
    h2b1.CHECKPOINT_DIRECTORY = CHECKPOINT_DIRECTORY
    h2b1.PROGRESS_PATH = PROGRESS_PATH
    h2b1.BASE_PATH = BASE_PATH
    h2b1.TANGENT_PATH = TANGENT_PATH
    h2b1.ANCHOR_PATH = ANCHOR_PATH
    h2b1._source_identity = _source_identity
    h2b1._anchor_sample_indices = lambda base: set(
        range(base["accepted_timesteps"].size)
    )


def _seed_checkpoints() -> dict:
    identity = _source_identity()
    if PROGRESS_PATH.exists():
        progress = _read_json(PROGRESS_PATH)
        if progress.get("source_identity") != identity:
            raise RuntimeError("h2c1 checkpoint source identity changed")
        return progress
    parent = _load_parent_arrays()
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    base = {
        "accepted_times": parent["base__accepted_times"][-1:],
        "accepted_states": parent["base__accepted_states"][-1:],
        "accepted_primitive_histories": parent[
            "base__accepted_primitive_histories"
        ][-1:],
        "accepted_mapped_histories": parent["base__accepted_mapped_histories"][-1:],
        "accepted_height_histories": parent["base__accepted_height_histories"][-1:],
        "accepted_previous_timesteps": parent[
            "base__accepted_previous_timesteps"
        ][-1:],
        "accepted_timesteps": np.empty(0, dtype=float),
        "accepted_step_wall_seconds": np.empty(0, dtype=float),
        "local_state_estimates": np.empty(0, dtype=float),
        "local_export_estimates": np.empty(0, dtype=float),
        "local_error_estimates": np.empty(0, dtype=float),
        "retries": np.empty(0, dtype=int),
        "output_times": parent["base__output_times"][-1:],
        "output_states": parent["base__output_states"][-1:],
        "output_exports": parent["base__output_exports"][-1:],
        "next_candidate_timestep": parent["base__next_candidate_timestep"],
    }
    tangent = {
        "state_directions": parent["tangent__state_directions"][-1:],
        "export_directions": parent["tangent__export_directions"][-1:],
        "primitive_history_directions": parent[
            "tangent__primitive_history_directions"
        ][-1:],
        "mapped_history_directions": parent["tangent__mapped_history_directions"][-1:],
        "height_history_directions": parent["tangent__height_history_directions"][-1:],
        "matrix_assembly_wall_seconds": np.empty(0, dtype=float),
        "block_step_wall_seconds": np.empty(0, dtype=float),
        "audit_flags": np.empty(0, dtype=bool),
        "step_ratios": np.empty(0, dtype=float),
        "field_scales": parent["tangent__field_scales"],
        "export_scales": parent["tangent__export_scales"],
    }
    anchor = {
        "anchor_states": parent["anchor__anchor_states"][-1:],
        "anchor_primitive_histories": parent[
            "anchor__anchor_primitive_histories"
        ][-1:],
        "anchor_mapped_histories": parent["anchor__anchor_mapped_histories"][-1:],
        "anchor_height_histories": parent["anchor__anchor_height_histories"][-1:],
        "anchor_previous_timesteps": parent[
            "anchor__anchor_previous_timesteps"
        ][-1:],
        "anchor_predictors": np.empty(
            (0, *parent["anchor__anchor_states"].shape[1:]), dtype=float
        ),
        "anchor_step_wall_seconds": np.empty(0, dtype=float),
        "sampled_flags": np.empty(0, dtype=bool),
        "sampled_state_error_estimates": np.empty(0, dtype=float),
        "sampled_export_error_estimates": np.empty(0, dtype=float),
        "base_exports": parent["anchor__base_exports"][-1:],
        "anchor_exports": parent["anchor__anchor_exports"][-1:],
    }
    base_report = {
        "accepted_steps": 0,
        "rejected_attempts": 0,
        "maximum_scaled_residual": 0.0,
        "maximum_discrete_ledger_defect": 0.0,
        "maximum_mapped_endpoint_path_closure_defect": 0.0,
        "minimum_path_reconstruction_factor": 1.0,
        "maximum_incoming_excision_characteristics": 0,
        "maximum_export_ledger_defect": 0.0,
        "wall_seconds": 0.0,
        "passed_so_far": True,
    }
    tangent_report = {
        "maximum_step_matrix_jvp_relative_defect": 0.0,
        "maximum_linear_solve_relative_defect": 0.0,
        "maximum_matrix_component_closure_defect": 0.0,
        "maximum_incoming_excision_characteristics": 0,
        "maximum_export_active_prefix_ledger_defect": 0.0,
        "maximum_export_transport_telescoping_defect": 0.0,
    }
    anchor_report = {
        "maximum_scaled_residual": 0.0,
        "maximum_discrete_ledger_defect": 0.0,
        "maximum_mapped_endpoint_path_closure_defect": 0.0,
        "minimum_path_reconstruction_factor": 1.0,
        "maximum_incoming_excision_characteristics": 0,
        "maximum_export_ledger_defect": 0.0,
        "maximum_export_incoming_characteristics": 0,
        "maximum_sampled_state_error_estimate": 0.0,
        "maximum_sampled_export_error_estimate": 0.0,
    }
    np.savez_compressed(BASE_PATH, **base)
    np.savez_compressed(TANGENT_PATH, **tangent)
    np.savez_compressed(ANCHOR_PATH, **anchor)
    progress = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "source_identity": identity,
        "base_targets_completed": [],
        "tangent_steps_completed": 0,
        "anchor_steps_completed": 0,
        "reports": {
            "base": base_report,
            "tangent": tangent_report,
            "anchor": anchor_report,
        },
    }
    _write_json(PROGRESS_PATH, progress)
    return progress


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
    _validate_parent()
    _patch_shared_module()
    progress = _seed_checkpoints()
    configuration = h2b1._configuration()
    print("h2c1: build frozen nonlinear tangent", flush=True)
    frozen_tangent, setup_seconds = h2b1._build_frozen_tangent(configuration)
    with np.load(h2b1.DECISIVE_ARRAYS, allow_pickle=False) as parent:
        field_scales = np.asarray(parent["tangent__field_scales"], dtype=float)
        export_scales = np.asarray(parent["tangent__export_scales"], dtype=float)
    contract, _strict = h2b1.h2a2.h2.g._controller_contracts()
    base_report, base = h2b1._run_base_targets(
        progress, configuration, frozen_tangent, field_scales, export_scales, contract
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
    projection = h2b1._remaining_projection(
        base_report,
        base,
        tangent_report,
        tangent,
        anchor_report,
        replays,
        setup_seconds,
        contract,
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
            "middle_2ms_continuation_passed_5ms_completion_manifest_authorized"
            if passed
            else "middle_2ms_continuation_failed_later_middle_and_fine_blocked"
        ),
        "passed": passed,
        "base": base_report,
        "tangent": tangent_report,
        "anchor": anchor_report,
        "serialized_replays": replays,
        "remaining_cost_projection": projection,
        "middle_5ms_completion_manifest_authorized": passed,
        "middle_5ms_propagation_authorized": False,
        "middle_5ms_spatial_confirmation_certified": False,
        "fine_cost_bounded_propagation_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": (
            "WP10c9d6c7c3b5c3h2d0_middle_5ms_completion_manifest"
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
            "layout": h2b1.MIDDLE_LAYOUT,
            "profiles": h2b1.PROFILES,
            "generic_profile": h2b1.GENERIC_PROFILE,
            "coupling_face": h2b1.COUPLING_FACE,
            "target_microseconds": TARGET_MICROSECONDS,
            "controller": contract,
            "surrogate_gates": h2b1.h2a3.h2a1.GATES,
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
            "analyzed_base_parent": ANALYZED_BASE_PARENT,
            "analyzed_base_tree": ANALYZED_BASE_TREE,
            "working_head": _git_value("rev-parse", "HEAD"),
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "input_hashes": {
                "continuation_manifest": _sha256(h2c0.MANIFEST_PATH),
                "parent_summary": _sha256(h2b1.SUMMARY_PATH),
                "parent_arrays": _sha256(h2b1.DECISIVE_ARRAYS),
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
                "# Middle 2 ms continuation WP10c9d6c7c3b5c3h2c1",
                "",
                "## Classification",
                "",
                f"`{summary['classification']}`",
                "",
                f"The middle nonlinear base, generic nonlinear anchor, and corrected five-profile tangent reached 2 ms in `{base_report['accepted_steps']}` new accepted steps with `{base_report['rejected_attempts']}` rejected attempts.",
                "",
                f"The generic tangent discrepancy is `{anchor_report['state']['discrepancy_fraction_of_observable_response']:.6e}` of the state response and `{anchor_report['instantaneous_Tier_I']['discrepancy_fraction_of_observable_response']:.6e}` of the instantaneous Tier-I response. Base and anchor last-step replays are bitwise.",
                "",
                f"The conservative factor-two projection for remaining middle work through 5 ms is `{projection['projected_remaining_wall_hours']:.2f}` hours in the `{projection['resource_tier']}` tier.",
                "",
                "A pass authorizes only a fresh definitions-only 5 ms completion manifest. Fine work, fixed-Q experiments, and reduced slow evolution remain blocked.",
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
