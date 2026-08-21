#!/usr/bin/env python3
"""Freeze a no-new-truth transition-tube geometry audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_bounded_hot_exit_acquisition_wp10c9d6c7c3b5c4f25do as full_step  # noqa: E402
import run_causal_inner_hot_exit_half_step_recovery_wp10c9d6c7c3b5c4f25dq as half_step  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25dr"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25ds"
PARENT_COMMIT = "b2bc5fddf9989376cf3ffdc9bdede764ea321b57"
PARENT_TREE = "74784175bd0f681b5a4f0409829a91c6a9417829"
CLASSIFICATION = (
    "complete_accepted_transition_trajectory_locked_"
    "scalar_tube_geometry_audit_frozen"
)

FULL_STEP_COUNT = 5
HALF_STEP_COUNT = 12
STATE_COUNT = 1 + FULL_STEP_COUNT + HALF_STEP_COUNT
TRAIN_STATE_INDICES = (0, 2, 4, 6, 8, 10, 12, 14, 16, 17)
HOLDOUT_STATE_INDICES = (1, 3, 5, 7, 9, 11, 13, 15)

# These gates distinguish a one-dimensional nonlinear trajectory from a
# rank-one affine line.  The curve has one dynamic progress coordinate, while
# its stored embedding may use as many as the already certified 16 hidden
# directions.
ENERGY_CAPTURE_TARGET = 0.9999
MAXIMUM_HIDDEN_EMBEDDING_RANK = 16
MAXIMUM_TURN_ANGLE_DEGREES = 30.0
MINIMUM_FORWARD_CHORD_COSINE = 0.50
MAXIMUM_HOLDOUT_ERROR_OVER_PATH_LENGTH = 1.0e-2
MAXIMUM_HOLDOUT_ERROR_OVER_LOCAL_CHORD = 5.0e-2
MAXIMUM_HOLDOUT_MACRO_ERROR = 1.0e-4
MINIMUM_NONLOCAL_SEPARATION_OVER_MINIMUM_STEP = 0.50
MAXIMUM_MACRO_DRIFT_FROM_SEED = 2.0e-2
DECOMPOSITION_CLOSURE_TOLERANCE = 1.0e-10
MINIMUM_TRANSITION_HIDDEN_FRACTION = 0.99
MAXIMUM_ONLINE_LIFT_FLOPS = 1_000_000
MAXIMUM_ONLINE_TABLE_BYTES = 2 * 1024 * 1024

ARTIFACT = "causal_inner_transition_tube_geometry_manifest_wp10c9d6c7c3b5c4f25dr"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = "scripts/run_causal_inner_transition_tube_geometry_manifest_wp10c9d6c7c3b5c4f25dr.py"
THIS_TEST = "tests/test_causal_inner_transition_tube_geometry_manifest_wp10c9d6c7c3b5c4f25dr.py"
ANALYSIS_RUNNER = "scripts/run_causal_inner_transition_tube_geometry_wp10c9d6c7c3b5c4f25ds.py"
ANALYSIS_TEST = "tests/test_causal_inner_transition_tube_geometry_wp10c9d6c7c3b5c4f25ds.py"
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_TRANSITION_TUBE_GEOMETRY_"
    "MANIFEST_WP10C9D6C7C3B5C4F25DR_2026-08-21.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

TANGENT_ARRAYS = half_step.manifest.TANGENT_ARRAYS
REJECTED_FULL_STEP = full_step._stage_directory(6)


def _plain(value):
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _checksums(directory: Path) -> dict[str, str]:
    result = {}
    for line in (directory / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        if _sha(directory / name) != expected:
            raise RuntimeError(f"checksum mismatch: {directory / name}")
        result[name] = expected
    return result


def _accepted_stage_directories() -> tuple[Path, ...]:
    return tuple(full_step._stage_directory(index) for index in range(1, 6)) + tuple(
        half_step.base._stage_directory(index) for index in range(1, 13)
    )


def _validate_parents(*, require_clean: bool) -> dict:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("transition-tube parent commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("transition-tube parent tree changed")

    stage_hashes = {}
    for expected_index, directory in enumerate(_accepted_stage_directories(), 1):
        hashes = _checksums(directory)
        summary = _read(directory / "summary.json")
        local_index = expected_index if expected_index <= 5 else expected_index - 5
        if (
            not summary["passed"]
            or not summary["root_accepted"]
            or not summary["checkpoint_roundtrip_bitwise"]
            or summary["step_index"] != local_index
            or not (directory / "hot_exit_feature_arrays.npz").exists()
        ):
            raise RuntimeError(f"accepted transition stage changed: {directory}")
        stage_hashes[directory.name] = hashes

    rejected_hashes = _checksums(REJECTED_FULL_STEP)
    rejected = _read(REJECTED_FULL_STEP / "summary.json")
    if (
        rejected["passed"]
        or rejected["root_accepted"]
        or rejected["step_index"] != 6
        or rejected["next_step_authorized"]
        or (REJECTED_FULL_STEP / "checkpoint_step_06.npz").exists()
    ):
        raise RuntimeError("rejected full-step candidate changed")

    terminal = _read(half_step.base._stage_directory(12) / "summary.json")
    if (
        terminal["classification"]
        != "half_step_hot_exit_recovery_budget_exhausted_exit_not_reached"
        or terminal["hot_exit_reached"]
        or terminal["next_step_authorized"]
    ):
        raise RuntimeError("terminal half-step classification changed")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("transition-tube manifest requires a clean tracked tree")
    return {
        "accepted_stage_hashes": stage_hashes,
        "rejected_full_step_hashes": rejected_hashes,
    }


def _contract() -> dict:
    decisive = {
        "tangent_arrays": _sha(TANGENT_ARRAYS),
        "rejected_full_step_summary": _sha(REJECTED_FULL_STEP / "summary.json"),
    }
    for directory in _accepted_stage_directories():
        decisive[f"{directory.name}__summary"] = _sha(directory / "summary.json")
        decisive[f"{directory.name}__features"] = _sha(
            directory / "hot_exit_feature_arrays.npz"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "trajectory": {
            "accepted_full_step_roots": FULL_STEP_COUNT,
            "accepted_half_step_roots": HALF_STEP_COUNT,
            "state_count_including_seed": STATE_COUNT,
            "rejected_full_step_06_excluded": True,
            "accepted_history_only": True,
            "new_truth_calls": 0,
        },
        "prospective_split": {
            "training_state_indices": TRAIN_STATE_INDICES,
            "held_out_state_indices": HOLDOUT_STATE_INDICES,
            "endpoint_states_always_training": True,
            "interpolation": "time_weighted_piecewise_linear_between_training_knots",
        },
        "mathematical_architecture": {
            "exact_oblique_split": "y=Lq+Zh",
            "macro_coordinates": 82,
            "hidden_coordinates": 388,
            "transition_dynamic_coordinate": "one_scalar_progress_s",
            "embedding_rank_adaptive": True,
            "maximum_hidden_embedding_rank": MAXIMUM_HIDDEN_EMBEDDING_RANK,
            "hot_exit_required_before_impulse_collapse": True,
            "online_full_y470_residual_forbidden": True,
        },
        "binding_gates": {
            "energy_capture_target": ENERGY_CAPTURE_TARGET,
            "maximum_hidden_embedding_rank": MAXIMUM_HIDDEN_EMBEDDING_RANK,
            "maximum_turn_angle_degrees": MAXIMUM_TURN_ANGLE_DEGREES,
            "minimum_forward_chord_cosine": MINIMUM_FORWARD_CHORD_COSINE,
            "maximum_holdout_error_over_path_length": MAXIMUM_HOLDOUT_ERROR_OVER_PATH_LENGTH,
            "maximum_holdout_error_over_local_chord": MAXIMUM_HOLDOUT_ERROR_OVER_LOCAL_CHORD,
            "maximum_holdout_macro_error": MAXIMUM_HOLDOUT_MACRO_ERROR,
            "minimum_nonlocal_separation_over_minimum_step": MINIMUM_NONLOCAL_SEPARATION_OVER_MINIMUM_STEP,
            "maximum_macro_drift_from_seed": MAXIMUM_MACRO_DRIFT_FROM_SEED,
            "decomposition_closure_tolerance": DECOMPOSITION_CLOSURE_TOLERANCE,
            "minimum_transition_hidden_fraction": MINIMUM_TRANSITION_HIDDEN_FRACTION,
            "maximum_online_lift_flops": MAXIMUM_ONLINE_LIFT_FLOPS,
            "maximum_online_table_bytes": MAXIMUM_ONLINE_TABLE_BYTES,
        },
        "decision_policy": {
            "geometry_pass": "authorize_local_transition_tube_surrogate_only",
            "geometry_fail": "stop_and_reject_scalar_transition_tube",
            "hot_branch_truth_authorized": False,
            "transition_impulse_fit_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
        "decisive_input_hashes": decisive,
        "frozen_source_hashes": {
            THIS_RUNNER: _sha(ROOT / THIS_RUNNER),
            THIS_TEST: _sha(ROOT / THIS_TEST),
            ANALYSIS_RUNNER: _sha(ROOT / ANALYSIS_RUNNER),
            ANALYSIS_TEST: _sha(ROOT / ANALYSIS_TEST),
        },
    }


def _update_catalog(summary: dict) -> None:
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
                    "sha256": _sha(path),
                    "scientific_status": "DEFINITIONS_ONLY",
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case", "path", "bytes", "sha256", "scientific_status"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": True,
    }
    catalog.update(
        {
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": PARENT_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("transition-tube geometry manifest already exists")
    locks = _validate_parents(require_clean=True)
    contract = _contract()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    _write_json(CANONICAL_DIRECTORY / "geometry_contract.json", contract)
    _write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_commit": PARENT_COMMIT,
            "parent_tree": PARENT_TREE,
            "validated_parent_hashes": locks,
            "decisive_input_hashes": contract["decisive_input_hashes"],
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "accepted_trajectory_states_locked": STATE_COUNT,
        "rejected_state_excluded": True,
        "new_truth_calls": 0,
        "transition_impulse_fit_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "runner": THIS_RUNNER,
            "tests": [THIS_TEST, ANALYSIS_TEST],
            "analysis_runner": ANALYSIS_RUNNER,
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "implementation_commit": _git("rev-parse", "HEAD"),
            "implementation_tree": _git("rev-parse", "HEAD^{tree}"),
            "source_hashes": contract["frozen_source_hashes"],
        },
    )
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names
        ),
        encoding="utf-8",
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Transition-tube geometry manifest WP10c9d6c7c3b5c4f25dr",
                "",
                "The complete accepted hot-transition trajectory is hash-locked: five 1e-7 s roots, twelve 5e-8 s roots, and their common seed. The rejected full-step candidate is excluded.",
                "",
                "The prospective audit tests an exact conservative oblique split y=Lq+Zh and a one-scalar nonlinear progress tube with rank-adaptive embedding up to 16. Alternating interior states are held out before fitting.",
                "",
                "Passing this package can authorize only a local transition-tube surrogate. It cannot identify a hot branch or collapse the unresolved transition to an impulse.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if not args.freeze:
        parser.error("use --freeze")
    print(json.dumps(_plain(_freeze()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
