#!/usr/bin/env python3
"""Freeze a no-trajectory metric-chart recovery after the stage-1 boundary."""

from __future__ import annotations

import argparse
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

from imri_qpe.layer3_minidisk_1d.conservative_metric_chart_atlas import (  # noqa: E402
    ConservativeMetricChart,
    block_whitening_transform,
    metric_augmented_jacobian,
)
import run_causal_inner_tangent_phase_lap_recurrence_stage1_execution_wp10c9d6c7c3b5c4f25fix as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fiy_"
    "coupled_physical_memory_metric_chart_recovery_manifest"
)
CLASSIFICATION = (
    "coupled_physical_memory_metric_chart_recovery_selected_definitions_only"
)
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fiz_"
    "coupled_physical_memory_metric_chart_boundary_recovery_execution"
)
ARTIFACT = (
    "causal_inner_coupled_physical_memory_metric_chart_recovery_manifest_"
    "wp10c9d6c7c3b5c4f25fiy"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_COUPLED_PHYSICAL_MEMORY_METRIC_"
    "CHART_RECOVERY_MANIFEST_WP10C9D6C7C3B5C4F25FIY_2026-08-25.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_coupled_physical_memory_metric_chart_recovery_"
    "manifest_wp10c9d6c7c3b5c4f25fiy.py"
)
THIS_TEST = (
    "tests/test_causal_inner_coupled_physical_memory_metric_chart_recovery_"
    "manifest_wp10c9d6c7c3b5c4f25fiy.py"
)

ORIGINAL_BLOCKS = (162, 280, 28)
COUPLED_PHYSICAL_MEMORY_BLOCKS = (442, 28)
FULL_ROW_BLOCKS = (470,)
PARTITION_HIERARCHY = (
    ORIGINAL_BLOCKS,
    COUPLED_PHYSICAL_MEMORY_BLOCKS,
    FULL_ROW_BLOCKS,
)
MAXIMUM_SELECTION_METRIC_CONDITION = 5.0
MAXIMUM_BINDING_METRIC_CONDITION = 10.0
MAXIMUM_PATCH_TRANSITION_CONDITION = 10.0
MAXIMUM_TRANSFORM_INVERSE_CLOSURE = 1.0e-10
MAXIMUM_WHITENING_CLOSURE = 1.0e-9
MAXIMUM_JACOBIAN_RECONSTRUCTION_DEFECT = 1.0e-10
MAXIMUM_COORDINATE_ROUNDTRIP_DEFECT = 1.0e-10
MAXIMUM_RATE_PUSH_PULL_DEFECT = 1.0e-10
RECOVERY_SEGMENT_SECONDS = 2.5e-4


def _helper():
    return parent._helper()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    return float(
        np.linalg.norm(a - b)
        / max(
            float(np.linalg.norm(a)),
            float(np.linalg.norm(b)),
            np.finfo(float).tiny,
        )
    )


def _source_hashes() -> dict[str, str]:
    helper = _helper()
    paths = (
        THIS_RUNNER,
        THIS_TEST,
        parent.THIS_RUNNER,
        parent.THIS_TEST,
        parent.manifest.THIS_RUNNER,
        parent.manifest.THIS_TEST,
        parent.holdout.THIS_RUNNER,
        parent.holdout.THIS_TEST,
        parent.engine.THIS_RUNNER,
        parent.engine.THIS_TEST,
        parent.engine.suffix.THIS_RUNNER,
    )
    return {name: helper._sha(ROOT / name) for name in paths}


def _validate_parent(*, require_clean: bool) -> dict:
    helper = _helper()
    hashes = helper._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = helper._read(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = helper._read(
        parent.CANONICAL_DIRECTORY / "stage1_metrics.json"
    )
    provenance = helper._read(parent.CANONICAL_DIRECTORY / "provenance.json")
    values = metrics["gate_values"]
    if (
        summary["classification"] != parent.PHYSICAL_FAILURE_CLASSIFICATION
        or summary["passed"]
        or summary["authorized_next"] is not None
        or summary["accepted_segments"] != 42
        or summary["phase_lap_observed"]
        or summary["coarse_recurrence_candidate_observed"]
        or summary["complete_cycle_execution_authorized"]
        or summary["reduced_slow_evolution_authorized"]
        or metrics["classification"] != parent.PHYSICAL_FAILURE_CLASSIFICATION
        or metrics["passed"]
        or metrics["stop_reason"] != "physical_failure"
        or values["accepted_segments"] != 42
        or values["attempted_segments"] != 43
        or values["rejected_segments"] != 1
        or values["maximum_metric_coordinate_jacobian_condition"]
        <= MAXIMUM_BINDING_METRIC_CONDITION
        or values["minimum_reconstruction_factor"] < 1.0 - 1.0e-12
        or values["minimum_scattering_optical_depth"] < 1.0
        or values["maximum_height_ratio"] > 0.5
        or values["phase_lap_observed"]
        or values["coarse_recurrence_candidate_observed"]
        or not values["all_accepted_checkpoint_roundtrips_bitwise"]
        or not values["suffix_history_replay_bitwise"]
    ):
        raise RuntimeError("stage-1 metric boundary certificate changed")
    for relative, expected in provenance["source_hashes"].items():
        if helper._sha(ROOT / relative) != expected:
            raise RuntimeError(f"stage-1 source changed: {relative}")
    parent._validate_manifest(require_clean=False)
    if require_clean and helper._git(
        "status", "--short", "--untracked-files=no"
    ):
        raise RuntimeError("metric recovery manifest requires a clean tracked tree")
    return {
        "hashes": hashes,
        "summary": summary,
        "metrics": metrics,
        "provenance": provenance,
    }


def _parent_arrays() -> dict[str, np.ndarray]:
    arrays = _load_npz(parent.CANONICAL_DIRECTORY / "stage1_arrays.npz")
    if (
        arrays["current_coordinate470"].shape != (470,)
        or arrays["current_primitive_state"].shape != (112, 5)
        or arrays["metric_transform470x470"].shape != (470, 470)
        or arrays["metric_augmented560x560"].shape != (560, 560)
        or arrays["gauge_basis560x90"].shape != (560, 90)
        or arrays["accepted_endpoint_coordinates470"].shape != (42, 470)
        or arrays["attempted_acceptance"].shape != (43,)
        or bool(arrays["attempted_acceptance"][-1])
    ):
        raise RuntimeError("stage-1 recovery arrays changed")
    return arrays


def _anchor_jacobian(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
    transform = arrays["metric_transform470x470"]
    metric_rows = arrays["metric_augmented560x560"][:470]
    jacobian = np.linalg.solve(transform, metric_rows)
    defect = _relative(transform @ jacobian, metric_rows)
    return jacobian, defect


def _chart_candidate(
    *,
    arrays: dict[str, np.ndarray],
    jacobian: np.ndarray,
    block_sizes: tuple[int, ...],
) -> tuple[dict, dict[str, np.ndarray]]:
    transform, whitening = block_whitening_transform(jacobian, block_sizes)
    chart = ConservativeMetricChart(
        arrays["current_coordinate470"], transform, block_sizes
    )
    augmented, augmented_condition = metric_augmented_jacobian(
        jacobian,
        arrays["gauge_basis560x90"],
        chart,
    )
    transition = transform @ np.linalg.inv(
        arrays["metric_transform470x470"]
    )
    probe = (
        arrays["current_coordinate470"]
        + RECOVERY_SEGMENT_SECONDS
        * arrays["current_coordinate_rate470_per_s"]
    )
    roundtrip = chart.decode(chart.encode(probe))
    pushed = chart.push_rate(arrays["current_coordinate_rate470_per_s"])
    pulled = chart.pull_rate(pushed)
    metrics = {
        "block_sizes": list(block_sizes),
        "metric_jacobian_condition_number": whitening[
            "metric_jacobian_condition_number"
        ],
        "metric_augmented_condition_number": float(augmented_condition),
        "maximum_block_whitening_closure_defect": whitening[
            "maximum_block_whitening_closure_defect"
        ],
        "transform_condition_number": float(np.linalg.cond(transform)),
        "transform_inverse_closure_defect": chart.inverse_closure_defect,
        "transition_from_parent_condition_number": float(
            np.linalg.cond(transition)
        ),
        "coordinate_roundtrip_relative_defect": _relative(probe, roundtrip),
        "rate_push_pull_relative_defect": _relative(
            arrays["current_coordinate_rate470_per_s"], pulled
        ),
    }
    metrics["selection_passed"] = bool(
        metrics["metric_jacobian_condition_number"]
        <= MAXIMUM_SELECTION_METRIC_CONDITION
        and metrics["metric_augmented_condition_number"]
        <= MAXIMUM_SELECTION_METRIC_CONDITION
        and metrics["transition_from_parent_condition_number"]
        <= MAXIMUM_PATCH_TRANSITION_CONDITION
        and metrics["transform_inverse_closure_defect"]
        <= MAXIMUM_TRANSFORM_INVERSE_CLOSURE
        and metrics["maximum_block_whitening_closure_defect"]
        <= MAXIMUM_WHITENING_CLOSURE
        and metrics["coordinate_roundtrip_relative_defect"]
        <= MAXIMUM_COORDINATE_ROUNDTRIP_DEFECT
        and metrics["rate_push_pull_relative_defect"]
        <= MAXIMUM_RATE_PUSH_PULL_DEFECT
    )
    return metrics, {
        "transform470x470": transform,
        "metric_augmented560x560": augmented,
    }


def _candidate_target(arrays: dict[str, np.ndarray]) -> np.ndarray:
    return parent.engine.execution._variable_step_ab2(
        arrays["current_coordinate470"],
        arrays["current_coordinate_rate470_per_s"],
        arrays["previous_coordinate_rate470_per_s"],
        float(arrays["next_span_seconds"]),
        float(arrays["previous_span_seconds"]),
    )


def _recovery_seed(
    arrays: dict[str, np.ndarray],
    jacobian: np.ndarray,
    selected_metrics: dict,
    selected_arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    original_seed = _load_npz(
        parent.manifest.CANONICAL_DIRECTORY / "continuation_seed.npz"
    )
    result = {name: np.asarray(value) for name, value in arrays.items()}
    for name in (
        "phase_training_raw_rates470_per_s",
        "phase_observer_metric_transform470x470",
        "phase_lap_reference_coordinate470",
        "phase_lap_reference_primitive_state",
        "phase_lap_reference_unit_tangent470",
        "registered_section_covector470",
        "registered_section_reference_value",
        "reference_metric_speed_per_s",
    ):
        result[name] = original_seed[name]
    result.update(
        {
            "superseded_metric_transform470x470": arrays[
                "metric_transform470x470"
            ],
            "reconstructed_anchor_jacobian470x560": jacobian,
            "metric_transform470x470": selected_arrays["transform470x470"],
            "metric_augmented560x560": selected_arrays[
                "metric_augmented560x560"
            ],
            "selected_metric_block_sizes": np.asarray(
                selected_metrics["block_sizes"], dtype=np.int64
            ),
            "next_candidate_target470": _candidate_target(arrays),
            "unwrapped_phase_advance_radians": np.asarray(
                arrays["accepted_cumulative_phase_advance_radians"][-1]
            ),
            "accumulated_metric_path_length": np.asarray(
                arrays["accepted_cumulative_metric_path_lengths"][-1]
            ),
            "acquisition_stage": np.asarray(1),
            "metric_chart_generation": np.asarray(1),
        }
    )
    return result


def _definitions(selected_metrics: dict) -> dict:
    contract = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "authorized_execution": AUTHORIZED_NEXT,
        "preserved_negative_classification": (
            parent.PHYSICAL_FAILURE_CLASSIFICATION
        ),
        "failure_localization": {
            "failed_conditions": [
                "independent endpoint metric Jacobian condition <= 10",
                "independent endpoint metric augmented condition <= 10",
            ],
            "actual_physical_gates_passed": True,
            "rejected_candidate_propagated": False,
            "residual_or_retraction_failure": False,
        },
        "deterministic_chart_hierarchy": [
            list(value) for value in PARTITION_HIERARCHY
        ],
        "selection_rule": (
            "choose the first, least-mixing nested partition whose anchor "
            "metric and augmented conditions are <= 5, whose transition "
            "from the accepted parent chart is <= 10, and whose closures "
            "and roundtrips pass"
        ),
        "selected_partition": selected_metrics["block_sizes"],
        "selected_partition_semantics": {
            "first_block": (
                "joint numerical whitening of the 162 physical and 280 "
                "causal-memory coordinate rows"
            ),
            "second_block": "separate whitening of 28 departure rows",
            "original_coordinate_unchanged": True,
            "primitive_state_unchanged": True,
            "physics_and_ledgers_remain_in_original_coordinates": True,
        },
        "authorized_scope": {
            "one_recomputed_AB2_candidate_from_last_accepted_history": True,
            "maximum_new_accepted_segments": 1,
            "segment_seconds": RECOVERY_SEGMENT_SECONDS,
            "maximum_retractions": 1,
            "maximum_exact_free_field_calls": 1,
            "maximum_wall_hours": 1.0,
            "fixed_Q_calls": 0,
            "reaction_calls": 0,
            "nonlinear_roots": 0,
            "BDF_microsteps": 0,
        },
        "binding_recovery_gates": {
            "maximum_metric_jacobian_condition": (
                MAXIMUM_BINDING_METRIC_CONDITION
            ),
            "maximum_metric_augmented_condition": (
                MAXIMUM_BINDING_METRIC_CONDITION
            ),
            "maximum_patch_transition_condition": (
                MAXIMUM_PATCH_TRANSITION_CONDITION
            ),
            "maximum_transform_inverse_closure": (
                MAXIMUM_TRANSFORM_INVERSE_CLOSURE
            ),
            "all_original_retraction_and_physical_gates_unchanged": True,
            "all_phase_and_recurrence_gates_unchanged": True,
            "checkpoint_roundtrip_bitwise": True,
            "suffix_history_replay_bitwise": True,
            "accepted_history_only_propagation": True,
        },
        "forbidden": [
            "raise or remove the metric condition threshold",
            "reinterpret the rejected endpoint as accepted",
            "change the primitive state or original coordinate",
            "change physical equations, ledgers, phase observer, or section",
            "execute more than one recovery segment",
            "authorize a phase lap, cycle, or reduced slow evolution",
        ],
    }
    architecture = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "problem": (
            "independent three-block whitening lost condition margin because "
            "physical and causal-memory Jacobian row spaces became correlated"
        ),
        "mathematical_repair": (
            "replace W=diag(W_phys,W_mem,W_dep) by "
            "W=diag((J_pm J_pm^T)^(-1/2), "
            "(J_dep J_dep^T)^(-1/2)) at an accepted anchor"
        ),
        "why_root_and_physics_are_invariant": (
            "W is square and invertible and is applied only as a numerical "
            "left coordinate transform; W R(q)=0 iff R(q)=0"
        ),
        "why_two_blocks_not_one": (
            "the nested hierarchy selects the least row mixing with a strong "
            "condition reserve; the coupled 442+28 chart passes, so dense "
            "470-row whitening is retained only as a future fallback"
        ),
        "phase_and_recurrence_invariance": (
            "the tangent phase, path length, state return, and registered "
            "section remain measured in the prospectively frozen 168 ms "
            "observer metric, independent of the computational chart"
        ),
        "longer_term_chart_policy": (
            "at accepted chart boundaries evaluate nested 162+280+28, "
            "442+28, and 470 partitions and select the first with a "
            "prospective factor-two condition reserve"
        ),
        "slow_reduction_status": (
            "this only repairs offline truth-orbit acquisition; the eventual "
            "online slow solver still uses tabulated averaged drift and no "
            "truth integration or micro-time stepping"
        ),
    }
    return {"contract": contract, "architecture": architecture}


def _evaluate(parent_lock: dict) -> tuple[dict, dict[str, np.ndarray], dict]:
    arrays = _parent_arrays()
    jacobian, reconstruction = _anchor_jacobian(arrays)
    candidates = []
    candidate_arrays = []
    for blocks in PARTITION_HIERARCHY:
        metrics, values = _chart_candidate(
            arrays=arrays,
            jacobian=jacobian,
            block_sizes=blocks,
        )
        candidates.append(metrics)
        candidate_arrays.append(values)
    passing = [index for index, item in enumerate(candidates) if item["selection_passed"]]
    selected_index = passing[0] if passing else None
    selected = None if selected_index is None else candidates[selected_index]
    selected_values = None if selected_index is None else candidate_arrays[selected_index]
    supported = bool(
        reconstruction <= MAXIMUM_JACOBIAN_RECONSTRUCTION_DEFECT
        and selected_index == 1
        and not candidates[0]["selection_passed"]
        and candidates[1]["selection_passed"]
    )
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            CLASSIFICATION
            if supported
            else "coupled_physical_memory_metric_chart_recovery_not_supported"
        ),
        "passed": supported,
        "definitions_only": True,
        "new_truth_evaluations": 0,
        "new_retractions": 0,
        "new_accepted_segments": 0,
        "anchor_jacobian_reconstruction_relative_defect": reconstruction,
        "anchor_raw_jacobian_condition_number": float(np.linalg.cond(jacobian)),
        "chart_candidates": candidates,
        "selected_candidate_index": selected_index,
        "selected_block_sizes": None if selected is None else selected["block_sizes"],
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT if supported else None,
        "input_lock": {
            "parent_hashes": parent_lock["hashes"],
            "parent_classification": parent_lock["summary"]["classification"],
        },
    }
    if selected is None or selected_values is None:
        seed = {}
        definitions = _definitions({"block_sizes": []})
    else:
        seed = _recovery_seed(
            arrays,
            jacobian,
            selected,
            selected_values,
        )
        definitions = _definitions(selected)
    return metrics, seed, definitions


def _update_catalog(summary: dict) -> None:
    helper = _helper()
    manifest_path = ROOT / "results/manifests/canonical_artifacts.csv"
    summary_path = ROOT / "results/manifests/canonical_summary.json"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": helper._sha(path),
                    "scientific_status": (
                        "SUPPORTED" if summary["passed"] else "REJECTED"
                    ),
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


def _canonicalize(
    metrics: dict,
    seed: dict[str, np.ndarray],
    definitions: dict,
    parent_lock: dict,
) -> dict:
    helper = _helper()
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("metric recovery manifest result already exists")
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(
        CANONICAL_DIRECTORY / "chart_recovery_metrics.json", metrics
    )
    helper._write_json(
        CANONICAL_DIRECTORY / "recovery_contract.json",
        definitions["contract"],
    )
    helper._write_json(
        CANONICAL_DIRECTORY / "mathematical_architecture.json",
        definitions["architecture"],
    )
    _save_npz(CANONICAL_DIRECTORY / "recovery_seed.npz", seed)
    helper._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_hashes": parent_lock["hashes"],
            "parent_classification": parent_lock["summary"]["classification"],
            "parent_implementation_commit": parent_lock["provenance"][
                "implementation_commit"
            ],
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": metrics["passed"],
        "definitions_only": True,
        "boundary_recovery_execution_authorized": metrics["passed"],
        "boundary_recovery_execution_executed": False,
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
            "implementation_commit": helper._git("rev-parse", "HEAD"),
            "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"),
            "source_hashes": _source_hashes(),
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
    original, coupled, full = metrics["chart_candidates"]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Coupled physical-memory metric-chart recovery manifest",
                "",
                f"Classification: `{metrics['classification']}`.",
                "",
                "Stage 1 remains rejected at attempted endpoint 43. The candidate was not propagated. Its primitive-space guards and ledgers passed; the independently rebuilt three-block metric and augmented conditions reached 10.0455 against the unchanged 10.0 gate.",
                "",
                f"At the last accepted endpoint, the frozen nested chart hierarchy gives metric conditions `{original['metric_jacobian_condition_number']:.6f}` for 162+280+28, `{coupled['metric_jacobian_condition_number']:.6f}` for 442+28, and `{full['metric_jacobian_condition_number']:.6f}` for 470. The least-mixing chart with a factor-two reserve is 442+28; its transition from the accepted chart is `{coupled['transition_from_parent_condition_number']:.6f}`.",
                "",
                "The repair jointly whitens the intrinsically coupled physical and causal-memory rows but retains the 28 departure rows as a separate block. It is an invertible numerical left transform only: the primitive state, original coordinate, physical equations, ledgers, phase observer, registered section, and 10.0 binding condition gate are unchanged.",
                "",
                "Only one recomputed 0.25 ms boundary endpoint is authorized next. No phase lap, cycle, complete-cycle execution, or reduced slow evolution is authorized.",
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
    parent_lock = _validate_parent(require_clean=True)
    metrics, seed, definitions = _evaluate(parent_lock)
    summary = _canonicalize(metrics, seed, definitions, parent_lock)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
