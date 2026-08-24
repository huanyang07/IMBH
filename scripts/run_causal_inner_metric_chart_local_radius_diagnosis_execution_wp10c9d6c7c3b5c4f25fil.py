#!/usr/bin/env python3
"""Execute the nonpropagating local metric-chart radius diagnosis."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_metric_chart_local_radius_diagnosis_manifest_wp10c9d6c7c3b5c4f25fik as manifest  # noqa: E402
from imri_qpe.layer3_minidisk_1d.conservative_metric_chart_atlas_v2 import (  # noqa: E402
    ConservativeMetricChart,
    block_whitening_transform,
    metric_augmented_jacobian,
    metric_transport_retract_strict,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25fil"
PASS_CLASSIFICATION = "metric_chart_local_radius_identified"
NO_RADIUS_CLASSIFICATION = "metric_chart_local_radius_not_identified"
PHYSICAL_FAILURE_CLASSIFICATION = "metric_chart_local_radius_physical_gate_failed"
INTEGRITY_FAILURE_CLASSIFICATION = "metric_chart_local_radius_execution_integrity_failed"
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fim_adaptive_metric_chart_radius_recovery_manifest"
)
ARTIFACT = (
    "causal_inner_metric_chart_local_radius_diagnosis_execution_"
    "wp10c9d6c7c3b5c4f25fil"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_METRIC_CHART_LOCAL_RADIUS_"
    "DIAGNOSIS_EXECUTION_WP10C9D6C7C3B5C4F25FIL_2026-08-24.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_metric_chart_local_radius_diagnosis_execution_"
    "wp10c9d6c7c3b5c4f25fil.py"
)
THIS_TEST = (
    "tests/test_causal_inner_metric_chart_local_radius_diagnosis_execution_"
    "wp10c9d6c7c3b5c4f25fil.py"
)


def _helper():
    return manifest._helper()


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _validate_manifest(*, require_clean: bool) -> dict:
    helper = _helper()
    hashes = helper._validate_checksums(manifest.CANONICAL_DIRECTORY)
    summary = helper._read(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = helper._read(
        manifest.CANONICAL_DIRECTORY / "diagnosis_contract.json"
    )
    provenance = helper._read(manifest.CANONICAL_DIRECTORY / "provenance.json")
    if (
        summary["classification"] != manifest.CLASSIFICATION
        or not summary["passed"]
        or not summary["definitions_only"]
        or not summary["metric_chart_local_radius_diagnosis_authorized"]
        or summary["metric_chart_local_radius_diagnosis_executed"]
        or summary["new_trajectory"]
        or summary["authorized_next"] != manifest.AUTHORIZED_NEXT
        or contract["authorized_execution"] != manifest.AUTHORIZED_NEXT
        or contract["scope"]["span_ladder_seconds"]
        != list(manifest.SPAN_LADDER_SECONDS)
        or contract["scope"]["maximum_retractions"]
        != manifest.MAXIMUM_RETRACTIONS
        or contract["scope"]["exact_free_field_calls"] != 0
        or contract["scope"]["new_trajectory"]
        or not contract["gates"][
            "strict_status_requires_physical_closure_and_condition"
        ]
    ):
        raise RuntimeError("local-radius diagnosis authorization changed")
    for relative, frozen_hash in provenance["source_hashes"].items():
        if helper._sha(ROOT / relative) != frozen_hash:
            raise RuntimeError(f"frozen local-radius source changed: {relative}")
    parent_lock = manifest._validate_parent(require_clean=False)
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("local-radius execution requires a clean tracked tree")
    return {
        "hashes": hashes,
        "summary": summary,
        "contract": contract,
        "parent_lock": parent_lock,
    }


def _seed() -> dict[str, np.ndarray]:
    return _load_npz(manifest.CANONICAL_DIRECTORY / "diagnosis_seed.npz")


def _source_hashes() -> dict[str, str]:
    helper = _helper()
    paths = (
        THIS_RUNNER,
        THIS_TEST,
        manifest.THIS_RUNNER,
        manifest.THIS_TEST,
        manifest.STRICT_ATLAS_SOURCE,
        manifest.STRICT_ATLAS_TEST,
        manifest.parent.source.parent.ATLAS_SOURCE,
        manifest.parent.source.THIS_RUNNER,
        manifest.parent.execution.source.THIS_RUNNER,
    )
    return {name: helper._sha(ROOT / name) for name in paths}


def _identity(lock: dict) -> dict:
    helper = _helper()
    return {
        "work_package": WORK_PACKAGE,
        "implementation_commit": helper._git("rev-parse", "HEAD"),
        "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"),
        "manifest_hashes": lock["hashes"],
        "source_hashes": _source_hashes(),
        "span_ladder_seconds": list(manifest.SPAN_LADDER_SECONDS),
    }


def _prepare_scratch(lock: dict) -> dict:
    helper = _helper()
    identity = _identity(lock)
    path = SCRATCH_DIRECTORY / "execution_identity.json"
    if SCRATCH_DIRECTORY.exists():
        if not path.exists() or helper._read(path) != identity:
            raise RuntimeError("local-radius scratch identity mismatch")
    else:
        SCRATCH_DIRECTORY.mkdir(parents=True)
        helper._write_json(path, identity)
    return identity


def _target(seed: dict[str, np.ndarray], span: float) -> np.ndarray:
    return manifest.parent.execution._variable_step_ab2(
        seed["current_coordinate470"],
        seed["current_coordinate_rate470_per_s"],
        seed["previous_coordinate_rate470_per_s"],
        float(span),
        float(seed["previous_span_seconds"]),
    )


def _physical_passed(metrics: dict) -> bool:
    return bool(
        metrics["physical_passed"]
        and metrics["minimum_reconstruction_factor"] >= 1.0 - 1.0e-12
        and metrics["maximum_height_ratio"] <= 0.5
        and metrics["minimum_scattering_optical_depth"] >= 1.0
    )


def _fresh_patch_audit(
    *,
    exact_chart,
    model,
    state: np.ndarray,
    coordinate: np.ndarray,
    anchor_chart: ConservativeMetricChart,
) -> tuple[dict, dict[str, np.ndarray]]:
    began = time.perf_counter()
    jacobian, raw_metrics = exact_chart._coordinate_jacobian(model, state)
    transform, transform_metrics = block_whitening_transform(
        jacobian, manifest.parent.source._block_sizes()
    )
    chart = ConservativeMetricChart(
        coordinate,
        transform,
        manifest.parent.source._block_sizes(),
    )
    gauge = exact_chart._canonical_null_basis(jacobian)
    augmented, augmented_condition = metric_augmented_jacobian(
        jacobian, gauge, chart
    )
    transition = transform @ anchor_chart.inverse_transform
    transition_condition = float(np.linalg.cond(transition))
    passed = bool(
        transform_metrics["metric_jacobian_condition_number"]
        <= manifest.MAXIMUM_METRIC_JACOBIAN_CONDITION
        and augmented_condition <= manifest.MAXIMUM_METRIC_AUGMENTED_CONDITION
        and transition_condition <= manifest.MAXIMUM_PATCH_TRANSITION_CONDITION
        and chart.inverse_closure_defect
        <= manifest.parent.manifest.MAXIMUM_TRANSFORM_INVERSE_CLOSURE
    )
    metrics = {
        "passed": passed,
        "raw_coordinate_jacobian_condition_number": float(
            raw_metrics["condition_number"]
        ),
        **transform_metrics,
        "metric_augmented_condition_number": float(augmented_condition),
        "patch_transition_condition_number": transition_condition,
        "transform_inverse_closure_defect": chart.inverse_closure_defect,
        "wall_seconds": float(time.perf_counter() - began),
    }
    arrays = {
        "coordinate_jacobian470x560": np.asarray(jacobian),
        "metric_transform470x470": np.asarray(transform),
        "metric_augmented560x560": np.asarray(augmented),
        "gauge_basis560x90": np.asarray(gauge),
    }
    return metrics, arrays


def _run_span(
    *,
    index: int,
    span: float,
    seed: dict[str, np.ndarray],
    inputs: dict,
    exact_chart,
    anchor_chart: ConservativeMetricChart,
) -> tuple[dict, dict[str, np.ndarray]]:
    helper = _helper()
    directory = SCRATCH_DIRECTORY / f"span_{index:02d}"
    directory.mkdir(exist_ok=True)
    metrics_path = directory / "diagnosis.json"
    arrays_path = directory / "diagnosis.npz"
    target = _target(seed, span)
    if metrics_path.exists() or arrays_path.exists():
        if not metrics_path.exists() or not arrays_path.exists():
            raise RuntimeError("incomplete local-radius span cache")
        metrics = helper._read(metrics_path)
        arrays = _load_npz(arrays_path)
        np.testing.assert_array_equal(arrays["target_original_coordinate470"], target)
        print(f"span={span:.3e}s reused", flush=True)
        return metrics, arrays
    policy = manifest.parent.source._policy()
    initial = manifest.parent.source.parent.parent._initial_state(
        inputs["model"],
        seed["current_primitive_state"],
        seed["current_coordinate470"],
        target,
    )
    began = time.perf_counter()
    state, matrix, strict = metric_transport_retract_strict(
        exact_chart=exact_chart,
        model=inputs["model"],
        initial_state=initial,
        target_original_coordinate=target,
        gauge_basis=seed["current_gauge_basis560x90"],
        anchor_delta=exact_chart._delta(
            inputs["model"], seed["current_primitive_state"]
        ),
        anchor_metric_augmented=seed["current_metric_augmented560x560"],
        chart=anchor_chart,
        policy=policy,
    )
    recovered, factors = inputs["model"].coordinate(state)
    physical_passed = _physical_passed(strict)
    fresh_metrics = None
    fresh_arrays: dict[str, np.ndarray] = {}
    if span < manifest.SPAN_LADDER_SECONDS[0] and strict["passed"]:
        fresh_metrics, fresh_arrays = _fresh_patch_audit(
            exact_chart=exact_chart,
            model=inputs["model"],
            state=state,
            coordinate=np.asarray(recovered),
            anchor_chart=anchor_chart,
        )
    metrics = {
        "span_seconds": float(span),
        "strict_retraction": strict,
        "physical_passed": physical_passed,
        "fresh_patch": fresh_metrics,
        "passed": bool(
            strict["passed"]
            and physical_passed
            and fresh_metrics is not None
            and fresh_metrics["passed"]
        ),
        "wall_seconds": float(time.perf_counter() - began),
    }
    arrays = {
        "target_original_coordinate470": np.asarray(target),
        "recovered_original_coordinate470": np.asarray(recovered),
        "primitive_state": np.asarray(state),
        "final_metric_broyden560x560": np.asarray(matrix),
        "decoder_reconstruction_factors": np.asarray(factors),
        "fresh_metric_transform470x470": fresh_arrays.get(
            "metric_transform470x470", np.full((470, 470), np.nan)
        ),
        "fresh_metric_augmented560x560": fresh_arrays.get(
            "metric_augmented560x560", np.full((560, 560), np.nan)
        ),
        "fresh_gauge_basis560x90": fresh_arrays.get(
            "gauge_basis560x90", np.full((560, 90), np.nan)
        ),
    }
    helper._write_json(metrics_path, metrics)
    _save_npz(arrays_path, arrays)
    print(
        f"span={span:.3e}s strict={strict['passed']} "
        f"closure={strict['nonlinear_closure_passed']} "
        f"condition={strict['maximum_metric_augmented_condition_number']:.6g} "
        f"fresh={None if fresh_metrics is None else fresh_metrics['passed']}",
        flush=True,
    )
    return metrics, arrays


def _classify(
    records: list[tuple[dict, dict[str, np.ndarray]]],
    *,
    seed_roundtrip_bitwise: bool,
    wall_seconds: float,
) -> tuple[str, bool, float | None]:
    physical = all(metrics["physical_passed"] for metrics, _arrays in records)
    two_ms = records[0][0]["strict_retraction"]
    reproduced = bool(
        not two_ms["passed"]
        and (
            not two_ms["nonlinear_closure_passed"]
            or not two_ms["chart_condition_passed"]
        )
    )
    passing = [
        metrics["span_seconds"]
        for metrics, _arrays in records[1:]
        if metrics["passed"]
    ]
    integrity = bool(
        seed_roundtrip_bitwise
        and len(records) == manifest.MAXIMUM_RETRACTIONS
        and wall_seconds <= 3600.0 * manifest.MAXIMUM_EXECUTION_WALL_HOURS
    )
    if not integrity:
        return INTEGRITY_FAILURE_CLASSIFICATION, False, None
    if not physical:
        return PHYSICAL_FAILURE_CLASSIFICATION, False, None
    if reproduced and passing:
        return PASS_CLASSIFICATION, True, float(max(passing))
    return NO_RADIUS_CLASSIFICATION, False, None


def _execute(lock: dict, identity: dict) -> tuple[dict, dict[str, np.ndarray]]:
    del lock, identity
    began = time.perf_counter()
    seed_first = _seed()
    seed_second = _seed()
    seed_roundtrip = bool(
        set(seed_first) == set(seed_second)
        and all(
            np.array_equal(seed_first[name], seed_second[name])
            for name in seed_first
        )
    )
    inputs = manifest.parent.execution.source._initial_inputs()
    exact_chart = manifest.parent.execution.source.arclength._exact_chart()
    anchor_chart = ConservativeMetricChart(
        seed_first["current_coordinate470"],
        seed_first["current_metric_transform470x470"],
        manifest.parent.source._block_sizes(),
    )
    records = [
        _run_span(
            index=index,
            span=span,
            seed=seed_first,
            inputs=inputs,
            exact_chart=exact_chart,
            anchor_chart=anchor_chart,
        )
        for index, span in enumerate(manifest.SPAN_LADDER_SECONDS)
    ]
    wall_seconds = float(time.perf_counter() - began)
    classification, passed, selected = _classify(
        records,
        seed_roundtrip_bitwise=seed_roundtrip,
        wall_seconds=wall_seconds,
    )
    selected_index = None
    if selected is not None:
        selected_index = manifest.SPAN_LADDER_SECONDS.index(selected)
    strict = [metrics["strict_retraction"] for metrics, _arrays in records]
    fresh = [metrics["fresh_patch"] for metrics, _arrays in records]
    gate_values = {
        "initial_elapsed_seconds": manifest.INITIAL_ELAPSED_SECONDS,
        "span_ladder_seconds": list(manifest.SPAN_LADDER_SECONDS),
        "strict_retraction_passed": [value["passed"] for value in strict],
        "physical_passed": [metrics["physical_passed"] for metrics, _arrays in records],
        "nonlinear_closure_passed": [
            value["nonlinear_closure_passed"] for value in strict
        ],
        "chart_condition_passed": [
            value["chart_condition_passed"] for value in strict
        ],
        "metric_augmented_condition_numbers": [
            value["maximum_metric_augmented_condition_number"] for value in strict
        ],
        "original_coordinate_residuals": [
            value["original_coordinate_residual_infinity"] for value in strict
        ],
        "metric_coordinate_residuals": [
            value["metric_coordinate_residual_infinity"] for value in strict
        ],
        "gauge_residuals": [value["gauge_residual_infinity"] for value in strict],
        "fresh_patch_passed": [
            None if value is None else value["passed"] for value in fresh
        ],
        "fresh_metric_condition_numbers": [
            None
            if value is None
            else value["metric_jacobian_condition_number"]
            for value in fresh
        ],
        "fresh_transition_condition_numbers": [
            None
            if value is None
            else value["patch_transition_condition_number"]
            for value in fresh
        ],
        "selected_local_radius_seconds": selected,
        "seed_roundtrip_bitwise": seed_roundtrip,
        "new_trajectory": False,
        "accepted_history_mutated": False,
        "exact_free_field_calls": 0,
        "fixed_Q_calls": 0,
        "reaction_calls": 0,
        "nonlinear_roots": 0,
        "BDF_microsteps": 0,
        "execution_wall_seconds": wall_seconds,
    }
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "authorized_next": AUTHORIZED_NEXT if passed else None,
        "gate_values": gate_values,
    }
    selected_arrays = (
        records[selected_index][1] if selected_index is not None else None
    )
    arrays = {
        "span_ladder_seconds": np.asarray(manifest.SPAN_LADDER_SECONDS),
        "targets_original_coordinate470": np.stack(
            [arrays["target_original_coordinate470"] for _metrics, arrays in records]
        ),
        "recovered_original_coordinates470": np.stack(
            [arrays["recovered_original_coordinate470"] for _metrics, arrays in records]
        ),
        "primitive_states": np.stack(
            [arrays["primitive_state"] for _metrics, arrays in records]
        ),
        "selected_local_radius_seconds": np.asarray(
            np.nan if selected is None else selected
        ),
        "selected_target_original_coordinate470": (
            np.full(470, np.nan)
            if selected_arrays is None
            else selected_arrays["target_original_coordinate470"]
        ),
        "selected_primitive_state": (
            np.full((112, 5), np.nan)
            if selected_arrays is None
            else selected_arrays["primitive_state"]
        ),
        "selected_metric_transform470x470": (
            np.full((470, 470), np.nan)
            if selected_arrays is None
            else selected_arrays["fresh_metric_transform470x470"]
        ),
        "selected_metric_augmented560x560": (
            np.full((560, 560), np.nan)
            if selected_arrays is None
            else selected_arrays["fresh_metric_augmented560x560"]
        ),
        "selected_gauge_basis560x90": (
            np.full((560, 90), np.nan)
            if selected_arrays is None
            else selected_arrays["fresh_gauge_basis560x90"]
        ),
    }
    return metrics, arrays


def _update_catalog(summary: dict) -> None:
    helper = _helper()
    manifest_path = ROOT / "results/manifests/canonical_artifacts.csv"
    summary_path = ROOT / "results/manifests/canonical_summary.json"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    status = "SUPPORTED" if summary["passed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
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
            "latest_source_parent_commit": helper._git("rev-parse", "HEAD"),
            "latest_work_package": WORK_PACKAGE,
        }
    )
    helper._write_json(summary_path, catalog)


def _canonicalize(metrics: dict, arrays: dict[str, np.ndarray], lock: dict, identity: dict) -> dict:
    helper = _helper()
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("local-radius execution result already exists")
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "diagnosis_metrics.json", metrics)
    _save_npz(CANONICAL_DIRECTORY / "diagnosis_arrays.npz", arrays)
    helper._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "manifest_hashes": lock["hashes"],
            "manifest_classification": lock["summary"]["classification"],
            "execution_identity": identity,
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": metrics["passed"],
        "selected_local_radius_seconds": metrics["gate_values"][
            "selected_local_radius_seconds"
        ],
        "new_trajectory": False,
        "cycle_authorized": False,
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
    values = metrics["gate_values"]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Metric-chart local-radius diagnosis execution",
                "",
                f"Classification: `{metrics['classification']}`.",
                "",
                f"Strict retraction passes over 2.00/1.00/0.50 ms: `{values['strict_retraction_passed']}`. Metric-augmented conditions: `{values['metric_augmented_condition_numbers']}`.",
                "",
                f"Fresh smaller-patch passes: `{values['fresh_patch_passed']}`. Selected local radius: `{values['selected_local_radius_seconds']}` s.",
                "",
                "No slow-field call was made and no diagnosed state was propagated.",
                "",
                f"Authorized next artifact: `{metrics['authorized_next']}`.",
                "",
            )
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    arguments = parser.parse_args()
    if not arguments.run:
        parser.error("--run is required")
    lock = _validate_manifest(require_clean=True)
    identity = _prepare_scratch(lock)
    metrics, arrays = _execute(lock, identity)
    summary = _canonicalize(metrics, arrays, lock, identity)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
