#!/usr/bin/env python3
"""Execute two fail-closed half steps at the stage-2 hyperbolicity boundary."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import json
from pathlib import Path
import platform
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_tangent_phase_hyperbolicity_two_half_step_bracket_manifest_wp10c9d6c7c3b5c4f25fizdc as manifest  # noqa: E402


boundary_diagnostic = manifest.parent
boundary_manifest = boundary_diagnostic.manifest
stage2 = boundary_manifest.parent
phase = stage2.phase
engine = stage2.engine
suffix = stage2.suffix
SCHEMA_VERSION = 1
WORK_PACKAGE = manifest.AUTHORIZED_NEXT
SUCCESS_CLASSIFICATION = (
    "two_half_steps_hyperbolic_predictor_overshoot_confirmed"
)
FIRST_BOUNDARY_CLASSIFICATION = (
    "hyperbolicity_boundary_before_first_half_step"
)
SECOND_BOUNDARY_CLASSIFICATION = (
    "hyperbolicity_boundary_bracketed_after_first_half_step"
)
AMBIGUOUS_CLASSIFICATION = "hyperbolicity_preflight_ambiguous"
PHYSICAL_FAILURE_CLASSIFICATION = (
    "two_half_step_original_physical_gate_failed"
)
NUMERICAL_FAILURE_CLASSIFICATION = (
    "two_half_step_numerical_phase_or_restart_failed"
)
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizde_"
    "tangent_phase_halved_step_stage2_continuation_manifest"
)
ARTIFACT = (
    "causal_inner_tangent_phase_hyperbolicity_two_half_step_bracket_execution_"
    "wp10c9d6c7c3b5c4f25fizdd"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
V1_LOCK_ARTIFACT = f"{ARTIFACT}_lock"
V1_LOCK_DIRECTORY = ROOT / "results/canonical" / V1_LOCK_ARTIFACT
V1_EXECUTION_COMMIT = "1a92db02312149dcf4334e948297c144d5dc5651"
LOCK_ARTIFACT = f"{ARTIFACT}_lock_v2"
LOCK_DIRECTORY = ROOT / "results/canonical" / LOCK_ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_TANGENT_PHASE_"
    "HYPERBOLICITY_TWO_HALF_STEP_BRACKET_EXECUTION_"
    "WP10C9D6C7C3B5C4F25FIZDD_2026-08-25.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
LOCK_REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_TANGENT_PHASE_"
    "HYPERBOLICITY_TWO_HALF_STEP_BRACKET_EXECUTION_LOCK_V2_"
    "WP10C9D6C7C3B5C4F25FIZDD_2026-08-25.md"
)
LOCK_REPORT_PATH = ROOT / LOCK_REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_tangent_phase_hyperbolicity_"
    "two_half_step_bracket_execution_wp10c9d6c7c3b5c4f25fizdd.py"
)
THIS_TEST = (
    "tests/test_causal_inner_tangent_phase_hyperbolicity_"
    "two_half_step_bracket_execution_wp10c9d6c7c3b5c4f25fizdd.py"
)

# Adapter constants consumed by the certified continuation engine.
INITIAL_ELAPSED_SECONDS = 0.18575000000000014
MINIMUM_SEGMENT_SECONDS = manifest.HALF_STEP_SECONDS
MAXIMUM_SEGMENT_SECONDS = manifest.HALF_STEP_SECONDS
GROWTH_FACTOR = 2.0
ACCEPTED_SEGMENTS_BEFORE_GROWTH = 4
BLIND_MIDPOINT_FREQUENCY = 8
MAXIMUM_ACCEPTED_SEGMENTS = manifest.MAXIMUM_ACCEPTED_HALF_STEPS
MAXIMUM_ATTEMPTED_SEGMENTS = manifest.MAXIMUM_ATTEMPTED_HALF_STEPS
MAXIMUM_EXACT_FREE_FIELD_CALLS = manifest.MAXIMUM_EXACT_FREE_FIELD_CALLS
MAXIMUM_RETRACTIONS = manifest.MAXIMUM_RETRACTIONS
MAXIMUM_EXECUTION_WALL_HOURS = manifest.MAXIMUM_WALL_HOURS
MAXIMUM_ENDPOINT_INTEGRAL_DEFECT = phase.MAXIMUM_ENDPOINT_INTEGRAL_DEFECT
MAXIMUM_BLIND_MIDPOINT_RATE_DEFECT = phase.MAXIMUM_BLIND_MIDPOINT_RATE_DEFECT


_BASE_HELPER_MODULE = manifest._helper()
_ORIGINAL_METRIC_FIELD = suffix._metric_field


class HyperbolicityBoundary(RuntimeError):
    """Raised before a free-field call when a retracted state is nonhyperbolic."""

    def __init__(self, metrics: dict, arrays: dict[str, np.ndarray]):
        super().__init__("retracted endpoint has a complex characteristic pencil")
        self.metrics = metrics
        self.arrays = arrays


def _helper():
    return manifest._helper()


def _stable_engine_helper():
    return _BASE_HELPER_MODULE


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def _contract() -> dict:
    return _helper()._read(
        manifest.CANONICAL_DIRECTORY / "two_half_step_contract.json"
    )


def _seed() -> dict[str, np.ndarray]:
    return _load_npz(manifest.CANONICAL_DIRECTORY / "two_half_step_seed.npz")


def _execution_source_hashes() -> dict[str, str]:
    helper = _helper()
    paths = (
        THIS_RUNNER,
        THIS_TEST,
        manifest.THIS_RUNNER,
        manifest.THIS_TEST,
        boundary_diagnostic.THIS_RUNNER,
        boundary_diagnostic.THIS_TEST,
        boundary_manifest.THIS_RUNNER,
        boundary_manifest.THIS_TEST,
        stage2.THIS_RUNNER,
        stage2.THIS_TEST,
        stage2.phase.THIS_RUNNER,
        stage2.phase.THIS_TEST,
        stage2.engine.THIS_RUNNER,
        stage2.engine.THIS_TEST,
        stage2.suffix.THIS_RUNNER,
        stage2.engine.execution.source.THIS_RUNNER,
        "src/imri_qpe/layer3_minidisk_1d/causal_inner_radial_linear_tangent.py",
        "src/imri_qpe/layer3_minidisk_1d/causal_inner_characteristic_dissipation.py",
    )
    return {name: helper._sha(ROOT / name) for name in paths}


def _validate_manifest(*, require_clean: bool) -> dict:
    helper = _helper()
    hashes = helper._validate_checksums(manifest.CANONICAL_DIRECTORY)
    summary = helper._read(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = _contract()
    provenance = helper._read(manifest.CANONICAL_DIRECTORY / "provenance.json")
    scope = contract["scope"]
    hyperbolicity = contract["binding_hyperbolicity_gate"]
    if (
        summary["classification"] != manifest.CLASSIFICATION
        or not summary["passed"]
        or not summary["definitions_only"]
        or not summary["half_step_execution_authorized"]
        or summary["failed_full_step_propagated"]
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["complete_cycle_execution_authorized"]
        or summary["reduced_slow_evolution_authorized"]
        or contract["authorized_execution"] != WORK_PACKAGE
        or scope["half_step_seconds"] != manifest.HALF_STEP_SECONDS
        or scope["maximum_accepted_half_steps"] != 2
        or scope["maximum_attempted_half_steps"] != 2
        or scope["tentative_segment_numbers"] != [268, 269]
        or scope["blind_midpoint_segment_numbers"] != []
        or scope["maximum_exact_free_field_calls"] != 2
        or scope["maximum_retractions"] != 2
        or not scope["failed_full_step_is_diagnostic_only"]
        or not hyperbolicity["checked_after_retraction_before_exact_field"]
        or hyperbolicity["maximum_imaginary_coordinate_speed"] != 1.0e-10
        or not hyperbolicity["no_complex_flux_split"]
    ):
        raise RuntimeError("two-half-step authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if helper._sha(ROOT / relative) != expected:
            raise RuntimeError(f"two-half-step manifest source changed: {relative}")
    manifest._validate_parent(require_clean=False)
    if require_clean and helper._git(
        "status", "--short", "--untracked-files=no"
    ):
        raise RuntimeError("two-half-step execution requires a clean tracked tree")
    return {
        "hashes": hashes,
        "summary": summary,
        "contract": contract,
        "provenance": provenance,
    }


def _static_execution_contract() -> dict:
    return {
        "work_package": WORK_PACKAGE,
        "half_step_seconds": manifest.HALF_STEP_SECONDS,
        "maximum_attempts": 2,
        "maximum_accepted_steps": 2,
        "all_face_hyperbolicity_preflight": True,
        "preflight_before_each_exact_field": True,
        "real_characteristic_basis_required": True,
        "maximum_real_spectrum_imaginary_speed": (
            manifest.MAXIMUM_REAL_SPECTRUM_IMAGINARY_SPEED
        ),
        "failed_full_step_propagated": False,
        "rejected_half_step_never_propagates": True,
        "original_truth_dynamics_and_gates_unchanged": True,
    }


def _execution_lock_payload() -> dict:
    helper = _helper()
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            "two_half_step_phase_history_filter_repair_locked_"
            "step1_field_reusable_not_propagated"
        ),
        "manifest_hashes": helper._validate_checksums(
            manifest.CANONICAL_DIRECTORY
        ),
        "manifest_contract_sha256": helper._sha(
            manifest.CANONICAL_DIRECTORY / "two_half_step_contract.json"
        ),
        "seed_sha256": helper._sha(
            manifest.CANONICAL_DIRECTORY / "two_half_step_seed.npz"
        ),
        "static_execution_contract": _static_execution_contract(),
        "source_hashes": _execution_source_hashes(),
        "superseded_v1_partial_execution": _v1_partial_snapshot(),
    }


def _v1_partial_file(logical_name: str) -> Path:
    directory = SCRATCH_DIRECTORY / "attempt_0000"
    migrated = directory / f"{Path(logical_name).stem}_prephase_v1{Path(logical_name).suffix}"
    original = directory / logical_name
    if migrated.exists():
        return migrated
    return original


def _v1_partial_snapshot() -> dict:
    helper = _helper()
    v1_hashes = helper._validate_checksums(V1_LOCK_DIRECTORY)
    v1_lock = helper._read(V1_LOCK_DIRECTORY / "execution_lock.json")
    identity_path = SCRATCH_DIRECTORY / "execution_identity.json"
    if not identity_path.exists():
        raise RuntimeError("v1 partial execution identity is missing")
    identity = helper._read(identity_path)
    if (
        identity["implementation_commit"] != V1_EXECUTION_COMMIT
        or identity["lock_hashes"] != v1_hashes
        or identity["source_hashes"] != v1_lock["source_hashes"]
        or identity["contract"] != _static_execution_contract()
    ):
        raise RuntimeError("v1 partial execution identity changed")
    logical_names = (
        "phase_prediction.json",
        "phase_prediction.npz",
        "endpoint_retraction.json",
        "endpoint_retraction.npz",
        "endpoint_field_hyperbolicity.json",
        "endpoint_field_hyperbolicity.npz",
        "endpoint_field.json",
        "endpoint_field.npz",
        "attempt.json",
        "attempt.npz",
    )
    paths = {name: _v1_partial_file(name) for name in logical_names}
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise RuntimeError(f"v1 partial artifact is missing: {missing[0]}")
    attempt = helper._read(paths["attempt.json"])
    field = helper._read(paths["endpoint_field.json"])
    hyperbolicity = helper._read(paths["endpoint_field_hyperbolicity.json"])
    checkpoint = SCRATCH_DIRECTORY / "attempt_0000/accepted_checkpoint.npz"
    if (
        not attempt["accepted"]
        or attempt["physical_failure"]
        or attempt.get("phase_geometry") is not None
        or attempt.get("recurrence_geometry") is not None
        or not field["physical_passed"]
        or not hyperbolicity["passed"]
        or checkpoint.exists()
    ):
        raise RuntimeError("v1 partial attempt was propagated or changed")
    return {
        "v1_lock_hashes": v1_hashes,
        "v1_identity_sha256": helper._sha(identity_path),
        "partial_artifact_hashes": {
            name: helper._sha(path) for name, path in paths.items()
        },
        "attempt_index": 0,
        "strict_retraction_passed": attempt["endpoint_retraction_passed"],
        "all_face_hyperbolicity_passed": hyperbolicity["passed"],
        "exact_field_physical_passed": field["physical_passed"],
        "endpoint_integral_defect": attempt["endpoint_integral_defect"],
        "phase_geometry_completed": False,
        "accepted_checkpoint_written": False,
        "candidate_propagated": False,
    }


def _update_catalog(
    artifact: str,
    directory: Path,
    summary: dict,
    status: str,
) -> None:
    helper = _helper()
    manifest_path = ROOT / "results/manifests/canonical_artifacts.csv"
    summary_path = ROOT / "results/manifests/canonical_summary.json"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != artifact]
    for path in sorted(directory.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": artifact,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": helper._sha(path),
                    "scientific_status": status,
                }
            )
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case", "path", "bytes", "sha256", "scientific_status"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = helper._read(summary_path)
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
            "latest_source_parent_commit": helper._git("rev-parse", "HEAD"),
            "latest_work_package": WORK_PACKAGE,
        }
    )
    helper._write_json(summary_path, catalog)


def _freeze_lock() -> dict:
    helper = _helper()
    if LOCK_DIRECTORY.exists() or LOCK_REPORT_PATH.exists():
        raise RuntimeError("two-half-step execution lock already exists")
    _validate_manifest(require_clean=True)
    payload = _execution_lock_payload()
    LOCK_DIRECTORY.mkdir(parents=True)
    helper._write_json(LOCK_DIRECTORY / "execution_lock.json", payload)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": payload["classification"],
        "passed": True,
        "definitions_only": True,
        "new_free_field_calls": 0,
        "new_retractions": 0,
        "v1_step1_prephase_field_reuse_authorized": True,
        "two_half_step_execution_authorized": True,
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    helper._write_json(LOCK_DIRECTORY / "summary.json", summary)
    helper._write_json(
        LOCK_DIRECTORY / "provenance.json",
        {
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "implementation_commit": helper._git("rev-parse", "HEAD"),
            "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"),
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    )
    names = sorted(path.name for path in LOCK_DIRECTORY.iterdir())
    (LOCK_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{helper._sha(LOCK_DIRECTORY / name)}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
    LOCK_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_REPORT_PATH.write_text(
        "# Two-half-step hyperbolicity bracket execution lock v2\n\n"
        "The v1 step-1 retraction, all-face hyperbolicity audit, and exact "
        "physical field passed but orchestration stopped before phase "
        "registration. No checkpoint was written and no candidate propagated. "
        "Their hashes are frozen for exact reuse. The v2 runner changes only "
        "the prior-history filter and remains bound to the same dynamics and "
        "physical gates.\n",
        encoding="utf-8",
    )
    _update_catalog(LOCK_ARTIFACT, LOCK_DIRECTORY, summary, "DEFINITIONS_ONLY")
    return summary


def _validate_lock(*, require_clean: bool) -> dict:
    helper = _helper()
    _validate_manifest(require_clean=False)
    hashes = helper._validate_checksums(LOCK_DIRECTORY)
    summary = helper._read(LOCK_DIRECTORY / "summary.json")
    lock = helper._read(LOCK_DIRECTORY / "execution_lock.json")
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or not summary["two_half_step_execution_authorized"]
        or not summary["v1_step1_prephase_field_reuse_authorized"]
        or lock != _execution_lock_payload()
    ):
        raise RuntimeError("two-half-step execution lock changed")
    if require_clean and helper._git(
        "status", "--short", "--untracked-files=no"
    ):
        raise RuntimeError("two-half-step run requires a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "lock": lock}


def _identity(lock: dict) -> dict:
    helper = _helper()
    return {
        "work_package": WORK_PACKAGE,
        "implementation_commit": helper._git("rev-parse", "HEAD"),
        "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"),
        "lock_hashes": lock["hashes"],
        "source_hashes": _execution_source_hashes(),
        "contract": _static_execution_contract(),
    }


def _prepare_scratch(lock: dict) -> dict:
    helper = _helper()
    identity = _identity(lock)
    path = SCRATCH_DIRECTORY / "execution_identity_v2.json"
    if not SCRATCH_DIRECTORY.exists():
        raise RuntimeError("v1 partial scratch required for v2 execution")
    if path.exists():
        if helper._read(path) != identity:
            raise RuntimeError("two-half-step v2 scratch identity mismatch")
        return identity
    snapshot = _v1_partial_snapshot()
    if snapshot != lock["lock"]["superseded_v1_partial_execution"]:
        raise RuntimeError("v1 partial reuse lock changed")
    directory = SCRATCH_DIRECTORY / "attempt_0000"
    for name in ("attempt.json", "attempt.npz"):
        source = directory / name
        destination = _v1_partial_file(name)
        if source == destination:
            destination = directory / (
                f"{source.stem}_prephase_v1{source.suffix}"
            )
        if source.exists():
            if destination.exists():
                raise RuntimeError("v1 prephase migration target exists")
            source.rename(destination)
    helper._write_json(
        SCRATCH_DIRECTORY / "v1_to_v2_migration.json",
        {
            "classification": "prephase_attempt_pair_preserved_not_propagated",
            "snapshot": snapshot,
            "moved_attempt_pair_out_of_engine_inventory": True,
        },
    )
    helper._write_json(path, identity)
    return identity


def _initial_progress() -> dict:
    seed = _seed()
    return {
        "previous_coordinate": seed["previous_coordinate470"].copy(),
        "current_coordinate": seed["current_coordinate470"].copy(),
        "previous_state": seed["previous_primitive_state"].copy(),
        "current_state": seed["current_primitive_state"].copy(),
        "previous_rate": seed["previous_coordinate_rate470_per_s"].copy(),
        "current_rate": seed["current_coordinate_rate470_per_s"].copy(),
        "previous_span": float(seed["previous_span_seconds"]),
        "next_span": float(seed["next_span_seconds"]),
        "elapsed_seconds": float(seed["elapsed_seconds"]),
        "accepted_segments_total": int(seed["accepted_segments_total"]),
        "accepted_segments_new": 0,
        "attempts": 0,
        "accepted_since_growth": 0,
        "metric_transform": seed["metric_transform470x470"].copy(),
        "metric_augmented": seed["metric_augmented560x560"].copy(),
        "gauge_basis": seed["gauge_basis560x90"].copy(),
        "section_normal": seed["section_normal470"].copy(),
        "start_coordinate": seed["start_coordinate470"].copy(),
        "stop_reason": None,
    }


def _accepted_attempts() -> list[tuple[dict, dict[str, np.ndarray]]]:
    result = []
    if not SCRATCH_DIRECTORY.exists():
        return result
    for directory in sorted(
        SCRATCH_DIRECTORY.glob("attempt_[0-9][0-9][0-9][0-9]")
    ):
        metrics_path = directory / "attempt.json"
        arrays_path = directory / "attempt.npz"
        if metrics_path.exists() and arrays_path.exists():
            metrics = _helper()._read(metrics_path)
            if (
                metrics.get("accepted")
                and metrics.get("phase_geometry") is not None
                and metrics.get("recurrence_geometry") is not None
            ):
                result.append((metrics, _load_npz(arrays_path)))
    return result


def _phase_history() -> np.ndarray:
    history = _seed()["accepted_endpoint_coordinate_rates470_per_s"].copy()
    new = [
        arrays["accepted_coordinate_rate470_per_s"]
        for _metrics, arrays in _accepted_attempts()
    ]
    if new:
        history = np.vstack((history, np.stack(new)))
    return history[-phase.holdout.manifest.SELECTED_WINDOW :]


def _prior_accumulation() -> dict:
    seed = _seed()
    result = {
        "cumulative_phase_advance_radians": float(
            seed["unwrapped_phase_advance_radians"]
        ),
        "cumulative_metric_path_length": float(
            seed["accumulated_metric_path_length"]
        ),
        "registered_section_value": float(
            seed["accepted_registered_section_values"][-1]
        ),
    }
    for metrics, _arrays in _accepted_attempts():
        recurrence = metrics["recurrence_geometry"]
        result = {
            "cumulative_phase_advance_radians": recurrence[
                "cumulative_phase_advance_radians"
            ],
            "cumulative_metric_path_length": recurrence[
                "cumulative_metric_path_length"
            ],
            "registered_section_value": recurrence[
                "endpoint_registered_section_value"
            ],
        }
    return result


def _hyperbolicity_paths(directory: Path, stem: str) -> tuple[Path, Path]:
    return (
        directory / f"{stem}_hyperbolicity.json",
        directory / f"{stem}_hyperbolicity.npz",
    )


def _face_hyperbolicity(context, state: np.ndarray) -> tuple[dict, dict]:
    charts = boundary_diagnostic._face_charts(context, state)
    values = []
    maximum_imaginary = []
    maximum_vector_imaginary = []
    eigenpair_defects = []
    for radius, chart in zip(context.grid.edges, charts, strict=True):
        pencil = boundary_diagnostic._analytic_pencil(
            context, float(radius), chart
        )
        values.append(pencil["values"])
        maximum_imaginary.append(pencil["maximum_imaginary_speed"])
        primitive_vectors = (
            pencil["column_scales"][:, None] * pencil["vectors"]
        )
        maximum_vector_imaginary.append(
            float(np.max(np.abs(np.imag(primitive_vectors))))
        )
        eigenpair_defects.append(pencil["maximum_eigenpair_defect"])
    imaginary = np.asarray(maximum_imaginary)
    vector_imaginary = np.asarray(maximum_vector_imaginary)
    complex_component = np.maximum(imaginary, vector_imaginary)
    first = np.flatnonzero(
        complex_component > manifest.MAXIMUM_REAL_SPECTRUM_IMAGINARY_SPEED
    )
    maximum_face = int(np.argmax(complex_component))
    passed = bool(
        np.max(complex_component)
        <= manifest.MAXIMUM_REAL_SPECTRUM_IMAGINARY_SPEED
    )
    metrics = {
        "faces_checked": int(len(charts)),
        "passed": passed,
        "maximum_imaginary_coordinate_speed": float(np.max(imaginary)),
        "maximum_imaginary_primitive_eigenvector_component": float(
            np.max(vector_imaginary)
        ),
        "maximum_complex_characteristic_component": float(
            np.max(complex_component)
        ),
        "maximum_imaginary_face": maximum_face,
        "first_complex_face": None if not len(first) else int(first[0]),
        "maximum_eigenpair_defect": float(np.max(eigenpair_defects)),
        "real_spectrum_gate": manifest.MAXIMUM_REAL_SPECTRUM_IMAGINARY_SPEED,
        "confirmed_complex_threshold": manifest.COMPLEX_SPECTRUM_IMAGINARY_SPEED,
    }
    arrays = {
        "face_charts": charts,
        "face_eigenvalues": np.stack(values),
        "face_maximum_imaginary_speeds": imaginary,
        "face_maximum_imaginary_primitive_eigenvector_components": (
            vector_imaginary
        ),
        "face_eigenpair_defects": np.asarray(eigenpair_defects),
    }
    return metrics, arrays


def _guarded_metric_field(
    *,
    directory: Path,
    stem: str,
    inputs: dict,
    exact_chart,
    state: np.ndarray,
    coordinate: np.ndarray,
    retraction: dict,
    anchor_chart,
):
    helper = _helper()
    metrics_path, arrays_path = _hyperbolicity_paths(directory, stem)
    if metrics_path.exists() or arrays_path.exists():
        if not metrics_path.exists() or not arrays_path.exists():
            raise RuntimeError("incomplete hyperbolicity preflight cache")
        hyperbolicity = helper._read(metrics_path)
        hyperbolicity_arrays = _load_npz(arrays_path)
        np.testing.assert_array_equal(
            hyperbolicity_arrays["requested_primitive_state"], state
        )
        np.testing.assert_array_equal(
            hyperbolicity_arrays["requested_coordinate470"], coordinate
        )
    else:
        context = inputs["base"]["configuration"]["context"]
        hyperbolicity, hyperbolicity_arrays = _face_hyperbolicity(
            context, state
        )
        hyperbolicity_arrays.update(
            {
                "requested_primitive_state": np.asarray(state),
                "requested_coordinate470": np.asarray(coordinate),
            }
        )
        helper._write_json(metrics_path, hyperbolicity)
        _save_npz(arrays_path, hyperbolicity_arrays)
    print(
        f"{directory.name}/{stem}: hyperbolic={hyperbolicity['passed']} "
        f"max_complex={hyperbolicity['maximum_complex_characteristic_component']:.6e}",
        flush=True,
    )
    if not hyperbolicity["passed"]:
        raise HyperbolicityBoundary(hyperbolicity, hyperbolicity_arrays)
    metrics, arrays = _ORIGINAL_METRIC_FIELD(
        directory=directory,
        stem=stem,
        inputs=inputs,
        exact_chart=exact_chart,
        state=state,
        coordinate=coordinate,
        retraction=retraction,
        anchor_chart=anchor_chart,
    )
    metrics = dict(metrics)
    metrics["hyperbolicity_preflight"] = hyperbolicity
    return metrics, arrays


def _boundary_record(
    *,
    progress: dict,
    error: HyperbolicityBoundary,
) -> tuple[dict, dict[str, np.ndarray]]:
    helper = _helper()
    index = int(progress["attempts"])
    directory = engine._attempt_directory(index)
    retraction = helper._read(directory / "endpoint_retraction.json")
    retraction_arrays = _load_npz(directory / "endpoint_retraction.npz")
    span = float(progress["next_span"])
    tentative = int(progress["accepted_segments_total"]) + 1
    candidate = engine.execution._variable_step_ab2(
        progress["current_coordinate"],
        progress["current_rate"],
        progress["previous_rate"],
        span,
        progress["previous_span"],
    )
    np.testing.assert_array_equal(
        candidate, retraction_arrays["target_original_coordinate470"]
    )
    metrics = {
        "attempt_index": index,
        "tentative_segment_number": tentative,
        "span_seconds": span,
        "previous_span_seconds": float(progress["previous_span"]),
        "blind_midpoint_required": False,
        "endpoint_integral_defect": None,
        "blind_midpoint_rate_defect": None,
        "endpoint_retraction_passed": retraction["passed"],
        "endpoint_retraction_physical_passed": retraction["physical_passed"],
        "endpoint_metric_augmented_condition": retraction[
            "maximum_metric_augmented_condition_number"
        ],
        "retryable_chart_failure": False,
        "endpoint_physical_passed": None,
        "midpoint_retraction_passed": None,
        "midpoint_physical_passed": None,
        "physical_failure": not retraction["physical_passed"],
        "numerical_passed": False,
        "accepted": False,
        "accepted_since_growth_after": int(progress["accepted_since_growth"]),
        "next_span_seconds": span,
        "stop_reason": "hyperbolicity_boundary",
        "elapsed_seconds_after": float(progress["elapsed_seconds"]),
        "endpoint_field": None,
        "midpoint_field": None,
        "phase_geometry": None,
        "recurrence_geometry": None,
        "hyperbolicity_preflight": error.metrics,
    }
    arrays = {
        "candidate_target470": candidate,
        "endpoint_coordinate470": retraction_arrays[
            "recovered_original_coordinate470"
        ],
        "endpoint_primitive_state": retraction_arrays["primitive_state"],
        "endpoint_coordinate_rate470_per_s": np.full(470, np.nan),
        "midpoint_target470": np.full(470, np.nan),
        "midpoint_hermite_rate470_per_s": np.full(470, np.nan),
        "midpoint_coordinate470": np.full(470, np.nan),
        "midpoint_primitive_state": np.full((112, 5), np.nan),
        "midpoint_coordinate_rate470_per_s": np.full(470, np.nan),
        "accepted_coordinate470": np.asarray(progress["current_coordinate"]),
        "accepted_primitive_state": np.asarray(progress["current_state"]),
        "accepted_coordinate_rate470_per_s": np.asarray(progress["current_rate"]),
        "accepted_metric_transform470x470": np.asarray(
            progress["metric_transform"]
        ),
        "accepted_metric_augmented560x560": np.asarray(
            progress["metric_augmented"]
        ),
        "accepted_gauge_basis560x90": np.asarray(progress["gauge_basis"]),
    }
    helper._write_json(directory / "attempt.json", metrics)
    _save_npz(directory / "attempt.npz", arrays)
    return metrics, arrays


def _guarded_phase_attempt(*, progress: dict, inputs: dict, exact_chart):
    try:
        return phase._phase_attempt(
            progress=progress,
            inputs=inputs,
            exact_chart=exact_chart,
        )
    except HyperbolicityBoundary as error:
        return _boundary_record(progress=progress, error=error)


def _short_restart_replay(
    records: list[tuple[dict, dict[str, np.ndarray]]],
    final_progress: dict,
) -> tuple[bool, int | None]:
    accepted_positions = [
        index
        for index, (metrics, _arrays) in enumerate(records)
        if metrics["accepted"]
    ]
    if not accepted_positions:
        return False, None
    position = accepted_positions[0]
    attempt = records[position][0]["attempt_index"]
    path = engine._attempt_directory(attempt) / "accepted_checkpoint.npz"
    if not path.exists():
        return False, None
    progress = engine._progress_from_checkpoint(_load_npz(path))
    replay = True
    for metrics, arrays in records[position + 1 :]:
        candidate = engine.execution._variable_step_ab2(
            progress["current_coordinate"],
            progress["current_rate"],
            progress["previous_rate"],
            progress["next_span"],
            progress["previous_span"],
        )
        replay = bool(
            replay and np.array_equal(candidate, arrays["candidate_target470"])
        )
        progress = engine._apply_record(progress, metrics, arrays)
    scalar_names = (
        "previous_span",
        "next_span",
        "elapsed_seconds",
        "accepted_segments_total",
        "accepted_segments_new",
        "attempts",
        "accepted_since_growth",
        "stop_reason",
    )
    array_names = (
        "previous_coordinate",
        "current_coordinate",
        "previous_state",
        "current_state",
        "previous_rate",
        "current_rate",
        "metric_transform",
        "metric_augmented",
        "gauge_basis",
        "section_normal",
        "start_coordinate",
    )
    replay = bool(
        replay
        and all(progress[name] == final_progress[name] for name in scalar_names)
        and all(
            np.array_equal(progress[name], final_progress[name])
            for name in array_names
        )
    )
    return replay, attempt


_ENGINE_NAMES = (
    "manifest",
    "WORK_PACKAGE",
    "PASS_CLASSIFICATION",
    "PHYSICAL_FAILURE_CLASSIFICATION",
    "NUMERICAL_FAILURE_CLASSIFICATION",
    "AUTHORIZED_NEXT",
    "SCRATCH_DIRECTORY",
    "_initial_progress",
    "_helper",
    "_attempt",
    "_restart_replay",
)


@contextmanager
def _execution_context():
    engine_saved = {name: getattr(engine, name) for name in _ENGINE_NAMES}
    phase_saved = {
        "_seed": phase._seed,
        "_accepted_phase_history": phase._accepted_phase_history,
        "_prior_accumulation": phase._prior_accumulation,
    }
    block_sizes_saved = suffix._block_sizes
    metric_field_saved = suffix._metric_field
    replacements = {
        "manifest": sys.modules[__name__],
        "WORK_PACKAGE": WORK_PACKAGE,
        "PASS_CLASSIFICATION": SUCCESS_CLASSIFICATION,
        "PHYSICAL_FAILURE_CLASSIFICATION": PHYSICAL_FAILURE_CLASSIFICATION,
        "NUMERICAL_FAILURE_CLASSIFICATION": NUMERICAL_FAILURE_CLASSIFICATION,
        "AUTHORIZED_NEXT": AUTHORIZED_NEXT,
        "SCRATCH_DIRECTORY": SCRATCH_DIRECTORY,
        "_initial_progress": _initial_progress,
        "_helper": _stable_engine_helper,
        "_attempt": _guarded_phase_attempt,
        "_restart_replay": _short_restart_replay,
    }
    try:
        for name, value in replacements.items():
            setattr(engine, name, value)
        phase._seed = _seed
        phase._accepted_phase_history = _phase_history
        phase._prior_accumulation = _prior_accumulation
        suffix._block_sizes = lambda: (442, 28)
        suffix._metric_field = _guarded_metric_field
        yield
    finally:
        for name, value in engine_saved.items():
            setattr(engine, name, value)
        for name, value in phase_saved.items():
            setattr(phase, name, value)
        suffix._block_sizes = block_sizes_saved
        suffix._metric_field = metric_field_saved


def _records() -> list[tuple[dict, dict[str, np.ndarray]]]:
    result = []
    if not SCRATCH_DIRECTORY.exists():
        return result
    for directory in sorted(
        SCRATCH_DIRECTORY.glob("attempt_[0-9][0-9][0-9][0-9]")
    ):
        metrics_path = directory / "attempt.json"
        arrays_path = directory / "attempt.npz"
        if metrics_path.exists() and arrays_path.exists():
            result.append((_helper()._read(metrics_path), _load_npz(arrays_path)))
    return result


def _classification_from_records(records: list[tuple[dict, dict]]) -> dict:
    accepted = [metrics for metrics, _arrays in records if metrics["accepted"]]
    boundaries = [
        metrics
        for metrics, _arrays in records
        if metrics.get("stop_reason") == "hyperbolicity_boundary"
    ]
    confirmed_boundaries = [
        metrics
        for metrics in boundaries
        if metrics["hyperbolicity_preflight"][
            "maximum_complex_characteristic_component"
        ]
        >= manifest.COMPLEX_SPECTRUM_IMAGINARY_SPEED
    ]
    physical = any(metrics.get("physical_failure") for metrics, _ in records)
    phase_failed = any(
        metrics.get("phase_geometry") is not None
        and not metrics["phase_geometry"]["passed"]
        for metrics, _ in records
    )
    if len(accepted) == 2 and not boundaries:
        return {
            "classification": SUCCESS_CLASSIFICATION,
            "passed": True,
            "trajectory_continuation_passed": True,
            "authorized_next": AUTHORIZED_NEXT,
        }
    if boundaries and not confirmed_boundaries:
        return {
            "classification": AMBIGUOUS_CLASSIFICATION,
            "passed": False,
            "trajectory_continuation_passed": False,
            "authorized_next": None,
        }
    if len(accepted) == 0 and confirmed_boundaries:
        return {
            "classification": FIRST_BOUNDARY_CLASSIFICATION,
            "passed": True,
            "trajectory_continuation_passed": False,
            "authorized_next": None,
        }
    if len(accepted) == 1 and confirmed_boundaries:
        return {
            "classification": SECOND_BOUNDARY_CLASSIFICATION,
            "passed": True,
            "trajectory_continuation_passed": False,
            "authorized_next": None,
        }
    if physical:
        classification = PHYSICAL_FAILURE_CLASSIFICATION
    else:
        classification = NUMERICAL_FAILURE_CLASSIFICATION
    return {
        "classification": classification,
        "passed": False,
        "trajectory_continuation_passed": False,
        "authorized_next": None,
        "phase_failed": phase_failed,
    }


def _classify(
    engine_metrics: dict,
    engine_arrays: dict[str, np.ndarray],
) -> tuple[dict, dict[str, np.ndarray]]:
    seed = _seed()
    records = _records()
    outcome = _classification_from_records(records)
    accepted = [(metrics, arrays) for metrics, arrays in records if metrics["accepted"]]
    boundaries = [
        metrics
        for metrics, _arrays in records
        if metrics.get("stop_reason") == "hyperbolicity_boundary"
    ]
    all_hyperbolic = bool(
        accepted
        and all(
            metrics["endpoint_field"]["hyperbolicity_preflight"]["passed"]
            for metrics, _arrays in accepted
        )
    )
    comparison = None
    if len(accepted) == 2:
        final_arrays = accepted[-1][1]
        transform = seed["phase_observer_metric_transform470x470"]
        failed_delta = (
            final_arrays["accepted_coordinate470"]
            - seed["failed_recovered_original_coordinate470"]
        )
        original_step = (
            seed["failed_recovered_original_coordinate470"]
            - seed["current_coordinate470"]
        )
        comparison = {
            "metric_coordinate_distance_to_rejected_full_step": float(
                np.linalg.norm(transform @ failed_delta)
            ),
            "distance_over_rejected_full_step_length": float(
                np.linalg.norm(transform @ failed_delta)
                / max(
                    np.linalg.norm(transform @ original_step),
                    np.finfo(float).tiny,
                )
            ),
            "primitive_relative_distance": float(
                np.linalg.norm(
                    final_arrays["accepted_primitive_state"]
                    - seed["failed_retracted_primitive_state"]
                )
                / max(
                    np.linalg.norm(seed["failed_retracted_primitive_state"]),
                    np.finfo(float).tiny,
                )
            ),
            "diagnostic_only": True,
        }
    values = dict(engine_metrics["gate_values"])
    values.update(
        {
            "prior_accepted_endpoints": 71,
            "new_accepted_half_steps": len(accepted),
            "combined_accepted_endpoints": 71 + len(accepted),
            "all_accepted_endpoints_hyperbolic": all_hyperbolic,
            "boundary_candidates": len(boundaries),
            "maximum_boundary_complex_characteristic_component": max(
                (
                    item["hyperbolicity_preflight"][
                        "maximum_complex_characteristic_component"
                    ]
                    for item in boundaries
                ),
                default=0.0,
            ),
            "two_half_step_to_rejected_full_step_comparison": comparison,
        }
    )
    metrics = {
        **engine_metrics,
        **outcome,
        "gate_values": values,
        "failed_full_step_propagated": False,
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    new_coordinates = (
        np.stack([arrays["accepted_coordinate470"] for _metrics, arrays in accepted])
        if accepted
        else np.empty((0, 470))
    )
    new_states = (
        np.stack([arrays["accepted_primitive_state"] for _metrics, arrays in accepted])
        if accepted
        else np.empty((0, 112, 5))
    )
    new_rates = (
        np.stack(
            [
                arrays["accepted_coordinate_rate470_per_s"]
                for _metrics, arrays in accepted
            ]
        )
        if accepted
        else np.empty((0, 470))
    )
    new_phase = np.asarray(
        [metrics_["phase_geometry"]["phase_increment"] for metrics_, _ in accepted]
    )
    arrays = {
        **engine_arrays,
        "combined_accepted_endpoint_coordinates470": np.vstack(
            (seed["accepted_endpoint_coordinates470"], new_coordinates)
        ),
        "combined_accepted_endpoint_primitive_states": np.concatenate(
            (seed["accepted_endpoint_primitive_states"], new_states), axis=0
        ),
        "combined_accepted_endpoint_coordinate_rates470_per_s": np.vstack(
            (seed["accepted_endpoint_coordinate_rates470_per_s"], new_rates)
        ),
        "new_phase_increments": new_phase,
    }
    return metrics, arrays


def _execute(lock: dict, identity: dict) -> tuple[dict, dict[str, np.ndarray]]:
    with _execution_context():
        metrics, arrays = engine._execute(lock, identity)
    return _classify(metrics, arrays)


def _canonicalize(
    metrics: dict,
    arrays: dict[str, np.ndarray],
    lock: dict,
    identity: dict,
) -> dict:
    helper = _helper()
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("two-half-step execution result already exists")
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "execution_metrics.json", metrics)
    _save_npz(CANONICAL_DIRECTORY / "execution_arrays.npz", arrays)
    helper._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {"execution_lock_hashes": lock["hashes"], "identity": identity},
    )
    values = metrics["gate_values"]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": metrics["passed"],
        "trajectory_continuation_passed": metrics[
            "trajectory_continuation_passed"
        ],
        "new_accepted_half_steps": values["new_accepted_half_steps"],
        "combined_accepted_endpoints": values["combined_accepted_endpoints"],
        "failed_full_step_propagated": False,
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": metrics["authorized_next"],
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    helper._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "implementation_commit": identity["implementation_commit"],
            "implementation_tree": identity["implementation_tree"],
            "source_hashes": identity["source_hashes"],
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    )
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{helper._sha(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
    comparison = values["two_half_step_to_rejected_full_step_comparison"]
    comparison_text = (
        "No two-half-step/full-step comparison was made."
        if comparison is None
        else (
            "The diagnostic metric-coordinate distance between the authentic "
            "two-half-step endpoint and the rejected full-step state is "
            f"`{comparison['metric_coordinate_distance_to_rejected_full_step']:.6e}`."
        )
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Tangent-phase hyperbolicity two-half-step bracket execution",
                "",
                f"Classification: `{metrics['classification']}`.",
                "",
                f"Accepted `{values['new_accepted_half_steps']}` of `{values['attempted_segments']}` attempted 0.125 ms half steps. The accepted chain contains `{values['combined_accepted_endpoints']}` endpoints.",
                "",
                f"Boundary candidates: `{values['boundary_candidates']}`; maximum boundary complex characteristic component: `{values['maximum_boundary_complex_characteristic_component']:.6e}`. All accepted endpoints hyperbolic: `{values['all_accepted_endpoints_hyperbolic']}`.",
                "",
                comparison_text,
                "",
                "The rejected 0.25 ms state was never propagated. Any complex half-step candidate was stopped before exact free-field evaluation and before accepted-history construction.",
                "",
                f"Authorized next artifact: `{metrics['authorized_next']}`. Complete-cycle execution and reduced slow evolution remain unauthorized.",
                "",
            )
        ),
        encoding="utf-8",
    )
    _update_catalog(
        ARTIFACT,
        CANONICAL_DIRECTORY,
        summary,
        "SUPPORTED" if summary["passed"] else "REJECTED",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-lock", action="store_true")
    parser.add_argument("--validate-lock", action="store_true")
    parser.add_argument("--run", action="store_true")
    arguments = parser.parse_args()
    if sum((arguments.freeze_lock, arguments.validate_lock, arguments.run)) != 1:
        parser.error("select exactly one mode")
    if arguments.freeze_lock:
        payload = _freeze_lock()
    elif arguments.validate_lock:
        payload = _validate_lock(require_clean=False)
    else:
        lock = _validate_lock(require_clean=True)
        identity = _prepare_scratch(lock)
        metrics, arrays = _execute(lock, identity)
        payload = _canonicalize(metrics, arrays, lock, identity)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if arguments.run and not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
