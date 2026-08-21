#!/usr/bin/env python3
"""Execute one hash-locked full-y470 hot-exit acquisition step at a time."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_bounded_hot_exit_acquisition_manifest_wp10c9d6c7c3b5c4f25dn as manifest  # noqa: E402
import run_causal_inner_branch_candidate_saved_array_screen_wp10c9d6c7c3b5c4f25dm as screen  # noqa: E402
import run_causal_inner_face36_fixed_q_primary_retry_wp10c9d6c7c3b5c4f24e14l as legacy  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_fixed_q import (  # noqa: E402
    load_causal_five_field_fixed_q_continuation_state,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25do"
MANIFEST_COMMIT = "64613b762828636272a9a50a6e2e3601156651c1"
LOCK_CLASSIFICATION = "bounded_hot_exit_execution_sources_locked_no_root_executed"
CONTINUE_CLASSIFICATION = "bounded_hot_exit_acquisition_stage_passed_exit_not_yet_reached"
EXIT_CLASSIFICATION = "persistent_hot_side_exit_candidate_supported_exact_branch_preflight_manifest_authorized"
BUDGET_CLASSIFICATION = "bounded_hot_exit_acquisition_budget_exhausted_exit_not_reached"
FAILURE_CLASSIFICATION = "bounded_hot_exit_acquisition_failed_no_branch_truth_authorized"

ARTIFACT_PREFIX = "causal_inner_bounded_hot_exit_acquisition_wp10c9d6c7c3b5c4f25do"
LOCK_ARTIFACT = f"{ARTIFACT_PREFIX}_execution_lock"
LOCK_DIRECTORY = ROOT / "results/canonical" / LOCK_ARTIFACT
LOCK_REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_BOUNDED_HOT_EXIT_ACQUISITION_"
    "EXECUTION_LOCK_WP10C9D6C7C3B5C4F25DO_2026-08-21.md"
)
LOCK_REPORT_PATH = ROOT / LOCK_REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_bounded_hot_exit_acquisition_"
    "wp10c9d6c7c3b5c4f25do.py"
)
THIS_TEST = (
    "tests/test_causal_inner_bounded_hot_exit_acquisition_"
    "wp10c9d6c7c3b5c4f25do.py"
)
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
SCRATCH_ROOT = ROOT / "outputs/checkpoints" / ARTIFACT_PREFIX

FEATURE_SOURCE_FILES = (
    manifest.THIS_RUNNER,
    screen.THIS_RUNNER,
    screen.geometry.THIS_RUNNER,
    screen.geometry.field_manifest.THIS_RUNNER,
    screen.geometry.field_manifest.vector_field.THIS_RUNNER,
    screen.geometry.field_manifest.vector_field.manifest.parent.geometry.chart_tools.THIS_RUNNER,
    screen.geometry.field_manifest.vector_field.manifest.parent.geometry.chart_tools.coordinate_tools.THIS_RUNNER,
)


def _plain(value):
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, bool):
        return value
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


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


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


def _clean_tracked_tree() -> bool:
    return not bool(_git("status", "--short", "--untracked-files=no"))


def _checksums(directory: Path) -> dict[str, str]:
    recorded = {}
    for line in (directory / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        if _sha(directory / name) != expected:
            raise RuntimeError(f"checksum mismatch: {directory / name}")
        recorded[name] = expected
    return recorded


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _stage_artifact(index: int) -> str:
    return f"{ARTIFACT_PREFIX}_step_{index:02d}"


def _stage_directory(index: int) -> Path:
    return ROOT / "results/canonical" / _stage_artifact(index)


def _stage_report_path(index: int) -> Path:
    return ROOT / (
        "docs/reports/current/CODEX_CAUSAL_INNER_BOUNDED_HOT_EXIT_"
        f"ACQUISITION_STEP_{index:02d}_WP10C9D6C7C3B5C4F25DO_2026-08-21.md"
    )


def _scratch_directory(index: int) -> Path:
    return SCRATCH_ROOT / f"step_{index:02d}"


def _static_execution_contract() -> dict:
    return {
        "work_package": WORK_PACKAGE,
        "one_root_per_command": True,
        "maximum_steps": manifest.MAXIMUM_NEW_BDF2_ROOTS,
        "timestep_seconds": manifest.TIMESTEP_SECONDS,
        "hidden_fraction_max": manifest.HIDDEN_SECANT_FRACTION_MAX,
        "persistence_steps": manifest.HIDDEN_EXIT_PERSISTENCE_STEPS,
        "rank16_hidden_amplitude_min": manifest.RANK16_HIDDEN_AMPLITUDE_MIN,
        "macro_drift_from_seed_max": manifest.MAXIMUM_MACRO_DRIFT_FROM_SEED,
        "rejected_root_never_propagates": True,
        "full_y470_dynamics_binding": True,
        "rank16_coordinates_diagnostic_only": True,
    }


def _validate_manifest(*, require_clean: bool) -> dict:
    if _git("rev-parse", MANIFEST_COMMIT) != MANIFEST_COMMIT:
        raise RuntimeError("hot-exit manifest commit changed")
    hashes = _checksums(manifest.CANONICAL_DIRECTORY)
    summary = _read(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = _read(
        manifest.CANONICAL_DIRECTORY / "hot_exit_acquisition_contract.json"
    )
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["maximum_new_BDF2_roots"] != manifest.MAXIMUM_NEW_BDF2_ROOTS
        or not summary["one_root_per_command"]
        or summary["branch_root_execution_authorized"]
        or contract["execution_order"]["maximum_new_BDF2_roots"]
        != manifest.MAXIMUM_NEW_BDF2_ROOTS
        or contract["execution_order"]["fixed_equal_BDF2_timestep_seconds"]
        != manifest.TIMESTEP_SECONDS
        or not contract["authorization_boundaries"][
            "stepwise_hot_exit_execution_in_next_package"
        ]
    ):
        raise RuntimeError("hot-exit execution manifest changed")
    for name, path in {
        "parent_summary": manifest.PARENT_SUMMARY,
        "parent_metrics": manifest.PARENT_METRICS,
        "parent_arrays": manifest.PARENT_ARRAYS,
        "parent_acquisition_contract": manifest.PARENT_ACQUISITION,
        "transition_tangent_arrays": manifest.TANGENT_ARRAYS,
        "candidate_geometry_arrays": manifest.GEOMETRY_ARRAYS,
        "seed_checkpoint": manifest.SEED_CHECKPOINT,
        "seed_checkpoint_json": manifest.SEED_CHECKPOINT_JSON,
        "seed_metrics": manifest.SEED_METRICS,
        "fixed_Q_source": ROOT / manifest.FIXED_Q_SOURCE,
        "monolithic_source": ROOT / manifest.MONOLITHIC_SOURCE,
        "legacy_continuation_runner": ROOT / manifest.LEGACY_CONTINUATION_RUNNER,
    }.items():
        if _sha(path) != contract["decisive_input_hashes"][name]:
            raise RuntimeError(f"hot-exit manifest input changed: {name}")
    provenance = _read(manifest.CANONICAL_DIRECTORY / "provenance.json")
    for name, expected in provenance["thread_environment"].items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    if require_clean and not _clean_tracked_tree():
        raise RuntimeError("hot-exit execution requires a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "contract": contract}


def _execution_lock_payload() -> dict:
    contract = _read(
        manifest.CANONICAL_DIRECTORY / "hot_exit_acquisition_contract.json"
    )
    transitive_sources = sorted(set((*legacy.SOURCE_FILES, *FEATURE_SOURCE_FILES)))
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": LOCK_CLASSIFICATION,
        "manifest_commit": MANIFEST_COMMIT,
        "manifest_contract_sha256": _sha(
            manifest.CANONICAL_DIRECTORY / "hot_exit_acquisition_contract.json"
        ),
        "manifest_summary_sha256": _sha(
            manifest.CANONICAL_DIRECTORY / "summary.json"
        ),
        "runner_sha256": _sha(ROOT / THIS_RUNNER),
        "test_sha256": _sha(ROOT / THIS_TEST),
        "fixed_Q_source_sha256": _sha(ROOT / manifest.FIXED_Q_SOURCE),
        "monolithic_source_sha256": _sha(ROOT / manifest.MONOLITHIC_SOURCE),
        "legacy_continuation_runner_sha256": _sha(
            ROOT / manifest.LEGACY_CONTINUATION_RUNNER
        ),
        "transitive_execution_source_hashes": {
            relative: _sha(ROOT / relative) for relative in transitive_sources
        },
        "coordinate_field_arrays_sha256": _sha(screen.geometry.FIELD_ARRAYS),
        "seed_checkpoint_sha256": _sha(manifest.SEED_CHECKPOINT),
        "static_execution_contract": _static_execution_contract(),
        "manifest_decisive_input_hashes": contract["decisive_input_hashes"],
    }


def _update_catalog(artifact: str, directory: Path, summary: dict, status: str) -> None:
    with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != artifact]
    for path in sorted(directory.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": artifact,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha(path),
                    "scientific_status": status,
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
    catalog.setdefault("artifacts", {})[artifact] = {
        "path": str(directory.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": summary["passed"],
    }
    catalog.update(
        {
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": MANIFEST_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def _freeze_lock() -> dict:
    if LOCK_DIRECTORY.exists() or LOCK_REPORT_PATH.exists():
        raise RuntimeError("hot-exit execution lock already exists")
    _validate_manifest(require_clean=True)
    payload = _execution_lock_payload()
    LOCK_DIRECTORY.mkdir(parents=True)
    _write_json(LOCK_DIRECTORY / "execution_lock.json", payload)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": LOCK_CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "runner_and_test_hash_locked": True,
        "step_1_execution_authorized": True,
        "branch_root_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    _write_json(LOCK_DIRECTORY / "summary.json", summary)
    _write_json(
        LOCK_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "implementation_commit": _git("rev-parse", "HEAD"),
            "implementation_tree": _git("rev-parse", "HEAD^{tree}"),
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "source_hashes": {
                THIS_RUNNER: _sha(ROOT / THIS_RUNNER),
                THIS_TEST: _sha(ROOT / THIS_TEST),
            },
        },
    )
    names = sorted(path.name for path in LOCK_DIRECTORY.iterdir())
    (LOCK_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(LOCK_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    LOCK_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_REPORT_PATH.write_text(
        "\n".join(
            [
                "# Bounded hot-exit acquisition execution lock WP10c9d6c7c3b5c4f25do",
                "",
                "The stepwise runner, tests, full fixed-Q residual sources, inherited warm solver policy, and warm_3 seed are hash-locked before any new root.",
                "",
                "Exactly one BDF2 root may be attempted per clean command. A rejected root cannot become history.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _update_catalog(LOCK_ARTIFACT, LOCK_DIRECTORY, summary, "DEFINITIONS_ONLY")
    return summary


def _validate_lock(*, require_clean: bool) -> dict:
    _validate_manifest(require_clean=False)
    hashes = _checksums(LOCK_DIRECTORY)
    summary = _read(LOCK_DIRECTORY / "summary.json")
    lock = _read(LOCK_DIRECTORY / "execution_lock.json")
    expected = _execution_lock_payload()
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or not summary["runner_and_test_hash_locked"]
        or lock != expected
    ):
        raise RuntimeError("hot-exit execution lock changed")
    if require_clean and not _clean_tracked_tree():
        raise RuntimeError("hot-exit step requires a clean tracked tree")
    return {"hashes": hashes, "lock": lock}


def _prior_stage(index: int) -> tuple[Path, dict] | None:
    if index == 1:
        return None
    directory = _stage_directory(index - 1)
    hashes = _checksums(directory)
    summary = _read(directory / "summary.json")
    if (
        not summary["passed"]
        or summary["hot_exit_reached"]
        or not summary["next_step_authorized"]
        or summary["step_index"] != index - 1
        or summary["classification"] != CONTINUE_CLASSIFICATION
    ):
        raise RuntimeError("prior hot-exit stage does not authorize continuation")
    return directory, {"hashes": hashes, "summary": summary}


def _next_step_index() -> int:
    existing = [
        index
        for index in range(1, manifest.MAXIMUM_NEW_BDF2_ROOTS + 1)
        if _stage_directory(index).exists()
    ]
    if not existing:
        return 1
    if existing != list(range(1, max(existing) + 1)):
        raise RuntimeError("hot-exit stages are not contiguous")
    previous = _read(_stage_directory(max(existing)) / "summary.json")
    if not previous["next_step_authorized"]:
        raise RuntimeError("hot-exit acquisition already reached a terminal stage")
    return max(existing) + 1


def _execution_identity(index: int, input_checkpoint: Path) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "stage_index": index,
        "execution_commit": _git("rev-parse", "HEAD"),
        "execution_tree": _git("rev-parse", "HEAD^{tree}"),
        "runner_sha256": _sha(ROOT / THIS_RUNNER),
        "test_sha256": _sha(ROOT / THIS_TEST),
        "execution_lock_sha256": _sha(LOCK_DIRECTORY / "execution_lock.json"),
        "manifest_contract_sha256": _sha(
            manifest.CANONICAL_DIRECTORY / "hot_exit_acquisition_contract.json"
        ),
        "input_checkpoint_sha256": _sha(input_checkpoint),
    }


@contextmanager
def _stage_legacy_runtime(scratch: Path):
    replacements = {
        "WORK_PACKAGE": WORK_PACKAGE,
        "ARTIFACT": ARTIFACT_PREFIX,
        "SCRATCH_DIRECTORY": scratch,
        "THIS_RUNNER": THIS_RUNNER,
        "THIS_TEST": THIS_TEST,
    }
    original = {name: getattr(legacy, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(legacy, name, value)
        with legacy._legacy_runtime():
            yield
    finally:
        for name, value in original.items():
            setattr(legacy, name, value)


def _input_checkpoint(index: int) -> Path:
    if index == 1:
        return manifest.SEED_CHECKPOINT
    return _stage_directory(index - 1) / f"checkpoint_step_{index - 1:02d}.npz"


def _static_feature_data() -> dict:
    tangent_arrays = _load_npz(manifest.TANGENT_ARRAYS)
    geometry_arrays = _load_npz(manifest.GEOMETRY_ARRAYS)
    screen_arrays = _load_npz(manifest.PARENT_ARRAYS)
    field_arrays = _load_npz(screen.geometry.FIELD_ARRAYS)
    field = screen.geometry.field_manifest.ForwardQuadraticAuthenticCenterField(
        field_arrays
    )
    labels = _read(manifest.PARENT_METRICS)["candidate_labels"]
    if labels[-1] != "fixed_Q_warm_3":
        raise RuntimeError("saved warm_3 coordinate label changed")
    return {
        "model": field.model,
        "macro_restriction": tangent_arrays["macro_restriction_R82"],
        "hidden_basis": tangent_arrays["hidden_basis_Z388"],
        "hidden_dual": tangent_arrays["hidden_dual_Q388"],
        "rank16_basis": tangent_arrays["selected_hidden_basis388"],
        "anchor_coordinate": geometry_arrays[
            "candidate_absolute_y470_coordinates"
        ][5],
        "seed_coordinate": screen_arrays[
            "candidate_absolute_y470_coordinates"
        ][-1],
    }


def _exit_features(
    static: dict,
    previous_coordinate: np.ndarray,
    current_state: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    current_coordinate = np.asarray(static["model"].coordinate(current_state)[0])
    secant_rate = (current_coordinate - previous_coordinate) / manifest.TIMESTEP_SECONDS
    hidden_rate = static["hidden_dual"] @ secant_rate
    hidden_action = static["hidden_basis"] @ hidden_rate
    hidden_fraction = float(
        np.linalg.norm(hidden_action)
        / max(float(np.linalg.norm(secant_rate)), np.finfo(float).tiny)
    )
    rank16_secant_capture = float(
        np.linalg.norm(static["rank16_basis"].T @ hidden_rate)
        / max(float(np.linalg.norm(hidden_rate)), np.finfo(float).tiny)
    )
    departure = current_coordinate - static["anchor_coordinate"]
    hidden_departure = static["hidden_dual"] @ departure
    rank16_amplitude = float(
        np.linalg.norm(static["rank16_basis"].T @ hidden_departure)
    )
    macro_drift = float(
        np.linalg.norm(
            static["macro_restriction"]
            @ (current_coordinate - static["seed_coordinate"])
        )
    )
    metrics = {
        "hidden_secant_fraction": hidden_fraction,
        "rank16_secant_capture": rank16_secant_capture,
        "rank16_hidden_amplitude_from_20ms_anchor": rank16_amplitude,
        "macro_drift_from_warm3_seed": macro_drift,
        "hidden_fraction_gate_passed": (
            hidden_fraction <= manifest.HIDDEN_SECANT_FRACTION_MAX
        ),
        "rank16_amplitude_gate_passed": (
            rank16_amplitude >= manifest.RANK16_HIDDEN_AMPLITUDE_MIN
        ),
        "macro_drift_gate_passed": (
            macro_drift <= manifest.MAXIMUM_MACRO_DRIFT_FROM_SEED
        ),
    }
    arrays = {
        "previous_coordinate470": np.asarray(previous_coordinate),
        "current_coordinate470": current_coordinate,
        "coordinate_secant_rate470_per_s": secant_rate,
        "hidden_secant_rate388_per_s": hidden_rate,
        "hidden_secant_action470_per_s": hidden_action,
    }
    return metrics, arrays


def _previous_coordinate(index: int, static: dict) -> np.ndarray:
    if index == 1:
        return np.asarray(static["seed_coordinate"])
    prior = _load_npz(
        _stage_directory(index - 1) / "hot_exit_feature_arrays.npz"
    )
    return np.asarray(prior["current_coordinate470"])


def _prior_hidden_gate_run(index: int) -> int:
    run = 0
    for prior_index in range(index - 1, 0, -1):
        metrics = _read(
            _stage_directory(prior_index) / "hot_exit_feature_metrics.json"
        )
        if not metrics["hidden_fraction_gate_passed"]:
            break
        run += 1
    return run


def _copy_stage_outputs(index: int, scratch: Path, destination: Path) -> None:
    label = f"step_{index:02d}"
    for name in (
        f"result_{label}.npz",
        f"metrics_{label}.json",
        f"checkpoint_{label}.npz",
        f"checkpoint_{label}.json",
        "execution_identity.json",
    ):
        source = scratch / name
        if source.exists():
            shutil.copy2(source, destination / name)


def _canonicalize_stage(
    index: int,
    scratch: Path,
    input_checkpoint: Path,
    root_metrics: dict,
    feature_metrics: dict | None,
    feature_arrays: dict[str, np.ndarray] | None,
    *,
    root_passed: bool,
) -> dict:
    destination = _stage_directory(index)
    if destination.exists() or _stage_report_path(index).exists():
        raise RuntimeError("hot-exit stage output already exists")
    destination.mkdir(parents=True)
    _copy_stage_outputs(index, scratch, destination)
    if feature_metrics is not None and feature_arrays is not None:
        _write_json(destination / "hot_exit_feature_metrics.json", feature_metrics)
        _write_npz(destination / "hot_exit_feature_arrays.npz", feature_arrays)

    persistent_run = (
        0
        if feature_metrics is None or not feature_metrics["hidden_fraction_gate_passed"]
        else _prior_hidden_gate_run(index) + 1
    )
    hot_exit_reached = bool(
        root_passed
        and feature_metrics is not None
        and feature_metrics["rank16_amplitude_gate_passed"]
        and feature_metrics["macro_drift_gate_passed"]
        and persistent_run >= manifest.HIDDEN_EXIT_PERSISTENCE_STEPS
    )
    macro_failed = bool(
        feature_metrics is not None
        and not feature_metrics["macro_drift_gate_passed"]
    )
    budget_exhausted = index >= manifest.MAXIMUM_NEW_BDF2_ROOTS
    if not root_passed or macro_failed:
        classification = FAILURE_CLASSIFICATION
    elif hot_exit_reached:
        classification = EXIT_CLASSIFICATION
    elif budget_exhausted:
        classification = BUDGET_CLASSIFICATION
    else:
        classification = CONTINUE_CLASSIFICATION
    next_step_authorized = bool(
        classification == CONTINUE_CLASSIFICATION
        and index < manifest.MAXIMUM_NEW_BDF2_ROOTS
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "step_index": index,
        "classification": classification,
        "passed": root_passed and not macro_failed,
        "root_accepted": root_passed,
        "checkpoint_roundtrip_bitwise": bool(
            root_metrics.get("checkpoint", {}).get("bitwise_roundtrip", False)
        ),
        "elapsed_time_seconds": (
            None
            if feature_metrics is None
            else float(root_metrics["elapsed_time_seconds"])
        ),
        "hidden_secant_fraction": (
            None
            if feature_metrics is None
            else feature_metrics["hidden_secant_fraction"]
        ),
        "persistent_hidden_exit_run": persistent_run,
        "hot_exit_reached": hot_exit_reached,
        "budget_exhausted": budget_exhausted,
        "next_step_authorized": next_step_authorized,
        "hot_branch_preflight_manifest_authorized": hot_exit_reached,
        "branch_root_execution_authorized": False,
        "transition_impulse_fit_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "sealed_16ms_opened": False,
    }
    _write_json(destination / "summary.json", summary)
    _write_json(
        destination / "input_lock.json",
        {
            "execution_lock_sha256": _sha(LOCK_DIRECTORY / "execution_lock.json"),
            "input_checkpoint": str(input_checkpoint.relative_to(ROOT)),
            "input_checkpoint_sha256": _sha(input_checkpoint),
            "prior_stage_summary_sha256": (
                None
                if index == 1
                else _sha(_stage_directory(index - 1) / "summary.json")
            ),
        },
    )
    _write_json(
        destination / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "step_index": index,
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "runner_sha256": _sha(ROOT / THIS_RUNNER),
            "test_sha256": _sha(ROOT / THIS_TEST),
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    )
    names = sorted(path.name for path in destination.iterdir())
    (destination / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(destination / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    report = _stage_report_path(index)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "\n".join(
            [
                f"# Bounded hot-exit acquisition step {index:02d}",
                "",
                f"Classification: `{classification}`.",
                "",
                f"Root accepted: `{root_passed}`. Hidden secant fraction: `{summary['hidden_secant_fraction']}`. Persistent exit run: `{persistent_run}` of `{manifest.HIDDEN_EXIT_PERSISTENCE_STEPS}`.",
                "",
                f"Next step authorized: `{next_step_authorized}`. Branch-root execution remains forbidden.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    status = (
        "HOT_EXIT_CANDIDATE"
        if hot_exit_reached
        else "REJECTED"
        if classification == FAILURE_CLASSIFICATION
        else "BOUNDED_ACQUISITION_STAGE"
    )
    _update_catalog(_stage_artifact(index), destination, summary, status)
    return summary


def _run_step(index: int) -> dict:
    if index < 1 or index > manifest.MAXIMUM_NEW_BDF2_ROOTS:
        raise ValueError("hot-exit step index is outside the frozen budget")
    _validate_lock(require_clean=True)
    prior = _prior_stage(index)
    input_checkpoint = _input_checkpoint(index)
    if not input_checkpoint.exists():
        raise RuntimeError("hot-exit input checkpoint is missing")
    scratch = _scratch_directory(index)
    if scratch.exists():
        raise RuntimeError("hot-exit scratch stage already exists")
    scratch.mkdir(parents=True)
    identity = _execution_identity(index, input_checkpoint)
    _write_json(scratch / "execution_identity.json", identity)

    data = legacy.e14d.e1._state_data("primary_20ms")
    continuation = load_causal_five_field_fixed_q_continuation_state(
        input_checkpoint,
        data["context"],
    )
    if (
        continuation.current_order != 2
        or continuation.next_order != 2
        or continuation.history.previous_timestep_seconds
        != manifest.TIMESTEP_SECONDS
        or continuation.nonlinear_solver_state is None
        or continuation.nonlinear_solver_state.current_timestep_seconds
        != manifest.TIMESTEP_SECONDS
        or continuation.nonlinear_solver_state.previous_timestep_seconds
        != manifest.TIMESTEP_SECONDS
        or continuation.next_reaction_channel_basis != "frozen_normalized"
    ):
        raise RuntimeError("hot-exit continuation seed semantics changed")

    label = f"step_{index:02d}"
    root_passed = False
    result = None
    root_metrics: dict = {}
    next_continuation = None
    try:
        with _stage_legacy_runtime(scratch):
            result, next_continuation, root_metrics = legacy.e14d._advance(
                label,
                data,
                continuation,
                manifest.TIMESTEP_SECONDS,
                identity,
            )
        root_passed = bool(root_metrics["root_passed"] and result.accepted)
    except legacy.e14d.BindingRootFailure as error:
        result = error.result
        root_metrics = error.metrics
        root_passed = False

    feature_metrics = None
    feature_arrays = None
    if root_passed and next_continuation is not None:
        static = _static_feature_data()
        feature_metrics, feature_arrays = _exit_features(
            static,
            _previous_coordinate(index, static),
            next_continuation.current_primitive_charts,
        )
        feature_metrics.update(
            {
                "step_index": index,
                "elapsed_time_seconds": next_continuation.elapsed_time_seconds,
                "completed_steps": next_continuation.completed_steps,
            }
        )
        root_metrics["elapsed_time_seconds"] = next_continuation.elapsed_time_seconds
        _write_json(scratch / f"metrics_{label}.json", root_metrics)
    else:
        root_metrics.setdefault("elapsed_time_seconds", None)

    return _canonicalize_stage(
        index,
        scratch,
        input_checkpoint,
        root_metrics,
        feature_metrics,
        feature_arrays,
        root_passed=root_passed,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-lock", action="store_true")
    parser.add_argument("--validate-lock", action="store_true")
    parser.add_argument("--next", action="store_true")
    parser.add_argument("--step", type=int)
    arguments = parser.parse_args()
    selected = sum(
        (
            arguments.freeze_lock,
            arguments.validate_lock,
            arguments.next,
            arguments.step is not None,
        )
    )
    if selected != 1:
        parser.error("select exactly one execution mode")
    if arguments.freeze_lock:
        payload = _freeze_lock()
    elif arguments.validate_lock:
        payload = _validate_lock(require_clean=False)
    else:
        index = _next_step_index() if arguments.next else int(arguments.step)
        payload = _run_step(index)
    print(json.dumps(_plain(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
