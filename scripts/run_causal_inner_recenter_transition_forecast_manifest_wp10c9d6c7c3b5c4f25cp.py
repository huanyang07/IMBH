#!/usr/bin/env python3
"""Freeze one cheap forecast of the first authentic atlas recentering."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_departure28_short_vector_field_validation_wp10c9d6c7c3b5c4f25bz as warm4  # noqa: E402
import run_causal_inner_direct_coordinate_field_validation_wp10c9d6c7c3b5c4f25co as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25cp"
PARENT_COMMIT = "3855c50e1480e0b0c2136455d5ba6afb246e84a0"
PARENT_PARENT = "cf3401c4893ff6d7856d13fa7d813db9c96bf889"
PARENT_TREE = "7963a38eeff7b59b879a1dce67d1d9476192694d"
CLASSIFICATION = "direct_field_recenter_transition_forecast_frozen"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25cq"

TIMESTEP_SECONDS = 1.0e-7
AUTHENTIC_ROOT_BUDGET = 2
COARSE_SUBSTEPS_PER_ROOT = 1
REFINED_SUBSTEPS_PER_ROOT = 4
AUDIT_SUBSTEPS_PER_ROOT = 8
RECENTER_TRIGGER_LOAD = 1.2e-2
HARD_CHART_LOAD = 1.5e-2

ARTIFACT = (
    "causal_inner_recenter_transition_forecast_manifest_"
    "wp10c9d6c7c3b5c4f25cp"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_recenter_transition_forecast_manifest_"
    "wp10c9d6c7c3b5c4f25cp.py"
)
THIS_TEST = (
    "tests/test_causal_inner_recenter_transition_forecast_manifest_"
    "wp10c9d6c7c3b5c4f25cp.py"
)
NEXT_RUNNER = (
    "scripts/run_causal_inner_recenter_transition_validation_"
    "wp10c9d6c7c3b5c4f25cq.py"
)
NEXT_TEST = (
    "tests/test_causal_inner_recenter_transition_validation_"
    "wp10c9d6c7c3b5c4f25cq.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_RECENTER_TRANSITION_"
    "FORECAST_MANIFEST_WP10C9D6C7C3B5C4F25CP_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

DIRECT_FIELD = (
    parent.manifest.CANONICAL_DIRECTORY / "direct_coordinate_field.npz"
)
DIRECT_CERTIFICATE = parent.CANONICAL_DIRECTORY / "rate_arrays.npz"
WARM4_DIRECTORY = warm4.CANONICAL_DIRECTORY
WARM4_CHECKPOINT = WARM4_DIRECTORY / "checkpoint_warm_4.npz"
WARM4_VALIDATION_ARRAYS = WARM4_DIRECTORY / "validation_arrays.npz"
WARM4_VALIDATION_METRICS = WARM4_DIRECTORY / "validation_metrics.json"
WARM4_METRICS = WARM4_DIRECTORY / "metrics_warm_4.json"

_plain = parent._plain
_read = parent._read
_write_json = parent._write_json
_sha = parent._sha
_checksums = parent._checksums
_load_npz = parent._load_npz


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _relative_error(
    predicted: np.ndarray, truth: np.ndarray, reference_change: np.ndarray
) -> float:
    return float(
        np.linalg.norm(np.asarray(predicted) - np.asarray(truth))
        / max(float(np.linalg.norm(reference_change)), np.finfo(float).tiny)
    )


def _endpoint_errors(
    predicted: np.ndarray, truth: np.ndarray, start: np.ndarray
) -> dict:
    slices = {
        "full": slice(None),
        "q162": slice(0, parent.manifest.PHYSICAL_DIMENSION),
        "z280": slice(
            parent.manifest.PHYSICAL_DIMENSION,
            parent.manifest.PHYSICAL_DIMENSION + parent.manifest.MEMORY_DIMENSION,
        ),
        "a28": slice(-parent.manifest.DEPARTURE_DIMENSION, None),
    }
    return {
        name: _relative_error(
            np.asarray(predicted)[selection],
            np.asarray(truth)[selection],
            np.asarray(truth)[selection] - np.asarray(start)[selection],
        )
        for name, selection in slices.items()
    }


def _validate_parent(*, require_clean: bool) -> dict:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("direct-field certificate commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("direct-field certificate lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("direct-field certificate tree changed")
    hashes = _checksums(parent.CANONICAL_DIRECTORY)
    summary = _read(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = _read(parent.CANONICAL_DIRECTORY / "rate_metrics.json")
    provenance = _read(parent.CANONICAL_DIRECTORY / "provenance.json")
    warm4_hashes = _checksums(WARM4_DIRECTORY)
    warm4_summary = _read(WARM4_DIRECTORY / "summary.json")
    warm4_metrics = _read(WARM4_METRICS)
    warm4_validation = _read(WARM4_VALIDATION_METRICS)
    if (
        not summary["passed"]
        or not summary["truth_passed"]
        or not summary["independent_coordinate_field_passed"]
        or summary["classification"] != parent.PASS_CLASSIFICATION
        or summary["authorized_next"]
        != "definitions_only_one_recentered_transition_forecast_execution_manifest"
        or summary["online_state_dependent_coordinate_Jacobian_calls"] != 0
        or not all(metrics["truth_checks"].values())
        or not all(metrics["field_checks"].values())
        or not warm4_summary["passed"]
        or not warm4_summary["prospective_forecast_passed"]
        or warm4_summary["accepted_truth_roots"] != 1
        or not warm4_metrics["accepted"]
        or not warm4_validation["truth_checkpoint"]["bitwise_roundtrip"]
        or warm4_metrics["timestep_seconds"] != TIMESTEP_SECONDS
    ):
        raise RuntimeError("recenter transition authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"direct-field certificate source changed: {relative}")
    for path in (
        DIRECT_FIELD,
        DIRECT_CERTIFICATE,
        WARM4_CHECKPOINT,
        WARM4_VALIDATION_ARRAYS,
        WARM4_VALIDATION_METRICS,
        WARM4_METRICS,
    ):
        if not path.is_file():
            raise RuntimeError(f"recenter transition input missing: {path}")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("recenter transition manifest requires a clean tracked tree")
    return {
        "summary": summary,
        "metrics": metrics,
        "hashes": hashes,
        "warm4_summary": warm4_summary,
        "warm4_hashes": warm4_hashes,
    }


def _rk4_endpoints(
    field,
    initial: np.ndarray,
    root_count: int,
    substeps_per_root: int,
) -> np.ndarray:
    state = np.asarray(initial, dtype=float).copy()
    endpoints = []
    step = TIMESTEP_SECONDS / int(substeps_per_root)
    for _root in range(int(root_count)):
        for _substep in range(int(substeps_per_root)):
            k1 = field(state)
            k2 = field(state + 0.5 * step * k1)
            k3 = field(state + 0.5 * step * k2)
            k4 = field(state + step * k3)
            state += step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        endpoints.append(np.array(state, copy=True))
    return np.asarray(endpoints)


def _load(direct: parent.manifest.DirectCoordinateField, coordinate: np.ndarray) -> float:
    old_delta, _weight = direct._old_shell(coordinate)
    return float(np.max(np.abs(old_delta)))


def _state_audit(
    direct: parent.manifest.DirectCoordinateField, coordinate: np.ndarray
) -> dict:
    state = direct.decoded_state(coordinate)
    physical = warm4.manifest.parent.geometry.chart_tools._state_audit(
        direct.model.components["context"], state
    )
    decoded_coordinate, factors = direct.model.coordinate(state)
    return {
        "minimum_reconstruction_factor": min(
            float(np.min(factors)), physical["minimum_reconstruction_factor"]
        ),
        "maximum_H_over_R": physical["maximum_h_over_r"],
        "minimum_scattering_optical_depth": physical[
            "minimum_scattering_optical_depth"
        ],
        "decoder_coordinate_relative_mismatch": _relative_error(
            decoded_coordinate, coordinate, coordinate
        ),
    }


def _forecast() -> tuple[dict[str, np.ndarray], dict]:
    closure = _load_npz(DIRECT_FIELD)
    direct = parent.manifest.DirectCoordinateField(closure)
    with np.load(WARM4_VALIDATION_ARRAYS, allow_pickle=False) as source:
        warm4_coordinate = np.asarray(source["truth_coordinate"], dtype=float)
    warm3_coordinate = np.zeros(parent.manifest.ONLINE_DIMENSION)
    began = time.perf_counter()
    retro_coarse = _rk4_endpoints(
        direct.field, warm3_coordinate, 1, COARSE_SUBSTEPS_PER_ROOT
    )[0]
    retro_refined = _rk4_endpoints(
        direct.field, warm3_coordinate, 1, REFINED_SUBSTEPS_PER_ROOT
    )[0]
    retro_audit = _rk4_endpoints(
        direct.field, warm3_coordinate, 1, AUDIT_SUBSTEPS_PER_ROOT
    )[0]
    prospective_coarse = _rk4_endpoints(
        direct.field,
        warm4_coordinate,
        AUTHENTIC_ROOT_BUDGET,
        COARSE_SUBSTEPS_PER_ROOT,
    )
    prospective_refined = _rk4_endpoints(
        direct.field,
        warm4_coordinate,
        AUTHENTIC_ROOT_BUDGET,
        REFINED_SUBSTEPS_PER_ROOT,
    )
    prospective_audit = _rk4_endpoints(
        direct.field,
        warm4_coordinate,
        AUTHENTIC_ROOT_BUDGET,
        AUDIT_SUBSTEPS_PER_ROOT,
    )
    loads = {
        "initial": _load(direct, warm4_coordinate),
        "refined_step_1": _load(direct, prospective_refined[0]),
        "refined_step_2": _load(direct, prospective_refined[1]),
        "audit_step_1": _load(direct, prospective_audit[0]),
        "audit_step_2": _load(direct, prospective_audit[1]),
    }
    retro_errors = _endpoint_errors(
        retro_refined, warm4_coordinate, warm3_coordinate
    )
    retro_coarse_refined = _relative_error(
        retro_coarse, retro_refined, retro_refined - warm3_coordinate
    )
    retro_refined_audit = _relative_error(
        retro_refined, retro_audit, retro_audit - warm3_coordinate
    )
    forecast_coarse_refined = _relative_error(
        prospective_coarse[-1],
        prospective_refined[-1],
        prospective_refined[-1] - warm4_coordinate,
    )
    forecast_refined_audit = _relative_error(
        prospective_refined[-1],
        prospective_audit[-1],
        prospective_audit[-1] - warm4_coordinate,
    )
    audits = [_state_audit(direct, point) for point in prospective_refined]
    gates = _contract()["binding_forecast_gates"]
    checks = {
        "retrospective_full": retro_errors["full"]
        <= gates["retrospective_full_coordinate_relative_error_max"],
        "retrospective_q": retro_errors["q162"]
        <= gates["retrospective_q162_relative_error_max"],
        "retrospective_z": retro_errors["z280"]
        <= gates["retrospective_z280_relative_error_max"],
        "retrospective_a": retro_errors["a28"]
        <= gates["retrospective_a28_relative_error_max"],
        "retrospective_coarse_refined": retro_coarse_refined
        <= gates["retrospective_coarse_refined_difference_max"],
        "retrospective_refined_audit": retro_refined_audit
        <= gates["retrospective_refined_audit_difference_max"],
        "forecast_coarse_refined": forecast_coarse_refined
        <= gates["prospective_coarse_refined_difference_max"],
        "forecast_refined_audit": forecast_refined_audit
        <= gates["prospective_refined_audit_difference_max"],
        "initial_inside": loads["initial"] < RECENTER_TRIGGER_LOAD,
        "refined_step_1_inside": loads["refined_step_1"] < RECENTER_TRIGGER_LOAD,
        "refined_step_2_crosses": loads["refined_step_2"] >= RECENTER_TRIGGER_LOAD,
        "refined_step_2_below_hard": loads["refined_step_2"] < HARD_CHART_LOAD,
        "audit_step_1_inside": loads["audit_step_1"] < RECENTER_TRIGGER_LOAD,
        "audit_step_2_crosses": loads["audit_step_2"] >= RECENTER_TRIGGER_LOAD,
        "audit_step_2_below_hard": loads["audit_step_2"] < HARD_CHART_LOAD,
        "reconstruction": min(
            audit["minimum_reconstruction_factor"] for audit in audits
        ) >= gates["minimum_reconstruction_factor"],
        "height": max(audit["maximum_H_over_R"] for audit in audits)
        <= gates["maximum_H_over_R"],
        "optical_depth": min(
            audit["minimum_scattering_optical_depth"] for audit in audits
        ) >= gates["minimum_scattering_optical_depth"],
    }
    metrics = {
        "checks": checks,
        "passed": all(checks.values()),
        "retrospective_endpoint_relative_errors": retro_errors,
        "retrospective_coarse_refined_relative_difference": retro_coarse_refined,
        "retrospective_refined_audit_relative_difference": retro_refined_audit,
        "prospective_coarse_refined_relative_difference": forecast_coarse_refined,
        "prospective_refined_audit_relative_difference": forecast_refined_audit,
        "predicted_old_decoder_loads": loads,
        "prospective_state_audits": audits,
        "predicted_trigger_root_index": 2,
        "predicted_trigger_elapsed_from_warm4_seconds": 2.0 * TIMESTEP_SECONDS,
        "new_truth_roots": 0,
        "new_continuous_rate_calls": 0,
        "new_complete_generator_assemblies": 0,
        "propagated_states": 0,
        "wall_seconds": time.perf_counter() - began,
    }
    arrays = {
        "warm3_coordinate": warm3_coordinate,
        "warm4_truth_coordinate": warm4_coordinate,
        "retrospective_coarse_coordinate": retro_coarse,
        "retrospective_refined_coordinate": retro_refined,
        "retrospective_audit_coordinate": retro_audit,
        "prospective_coarse_coordinates": prospective_coarse,
        "prospective_refined_coordinates": prospective_refined,
        "prospective_audit_coordinates": prospective_audit,
        "predicted_transition_center_coordinate": prospective_refined[-1],
        "predicted_transition_center_primitive_state": direct.decoded_state(
            prospective_refined[-1]
        ),
    }
    return arrays, metrics


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "objective": "forecast_and_then_authenticate_exactly_one_recenter_transition",
        "online_architecture": {
            "state": "q162_plus_dynamic_z280_plus_a28",
            "field": "independently_validated_direct_470_coordinate_field",
            "state_dependent_coordinate_Jacobian_per_online_evaluation": 0,
            "truth_calls_per_online_evaluation": 0,
            "role": "offline_fast_transient_atlas_builder_not_final_cycle_integrator",
        },
        "forecast": {
            "start": "accepted_authentic_warm_4_checkpoint",
            "timestep_seconds": TIMESTEP_SECONDS,
            "maximum_root_intervals": AUTHENTIC_ROOT_BUDGET,
            "method": "classical_RK4",
            "coarse_substeps_per_root": COARSE_SUBSTEPS_PER_ROOT,
            "refined_substeps_per_root": REFINED_SUBSTEPS_PER_ROOT,
            "audit_substeps_per_root": AUDIT_SUBSTEPS_PER_ROOT,
            "recenter_trigger_old_decoder_load": RECENTER_TRIGGER_LOAD,
            "hard_old_chart_load": HARD_CHART_LOAD,
            "forecast_must_be_hashed_before_truth": True,
        },
        "binding_forecast_gates": {
            "retrospective_full_coordinate_relative_error_max": 1.0e-2,
            "retrospective_q162_relative_error_max": 5.0e-2,
            "retrospective_z280_relative_error_max": 1.0e-2,
            "retrospective_a28_relative_error_max": 1.0e-2,
            "retrospective_coarse_refined_difference_max": 1.0e-5,
            "retrospective_refined_audit_difference_max": 1.0e-6,
            "prospective_coarse_refined_difference_max": 5.0e-3,
            "prospective_refined_audit_difference_max": 2.0e-5,
            "minimum_reconstruction_factor": 1.0 - 1.0e-12,
            "maximum_H_over_R": 0.12,
            "minimum_scattering_optical_depth": 1.0,
        },
        "authentic_execution": {
            "root_budget": AUTHENTIC_ROOT_BUDGET,
            "sequence": "accepted_warm4_to_warm5_to_warm6_fail_fast",
            "accepted_history_only": True,
            "equal_step_BDF2": True,
            "warm_policy": "carried_matrix_with_at_most_one_exact_refresh_per_root",
            "checkpoint_every_accepted_root": True,
            "predicted_center_may_not_become_center": True,
            "first_accepted_state_crossing_trigger_becomes_center": True,
        },
        "binding_execution_gates": {
            "accepted_roots_equal": AUTHENTIC_ROOT_BUDGET,
            "maximum_scaled_residual": 1.0e-10,
            "maximum_Q3_relative_defect": 1.0e-12,
            "minimum_reconstruction_factor": 1.0 - 1.0e-12,
            "maximum_H_over_R": 0.12,
            "minimum_scattering_optical_depth": 1.0,
            "maximum_exact_Jacobian_assemblies_per_root": 1,
            "checkpoint_roundtrip_bitwise": True,
            "forecast_endpoint_full_coordinate_relative_error_max": 0.15,
            "forecast_endpoint_q162_relative_error_max": 0.15,
            "forecast_endpoint_z280_relative_error_max": 0.15,
            "forecast_endpoint_a28_relative_error_max": 0.15,
            "warm5_exact_load_below_trigger": RECENTER_TRIGGER_LOAD,
            "warm6_exact_load_at_least_trigger": RECENTER_TRIGGER_LOAD,
            "warm6_exact_load_below_hard_limit": HARD_CHART_LOAD,
            "translation_roundtrip_infinity_defect_max": 1.0e-14,
        },
        "decision": {
            "pass_classification": "one_authentic_recenter_transition_validated",
            "fail_classification": "recenter_transition_forecast_or_truth_failed",
            "pass_authorizes_only": (
                "definitions_only_authentic_center_local_field_and_overlap_manifest"
            ),
            "fail_authorizes_only": "definitions_only_transition_diagnosis_manifest",
        },
        "authorization_boundaries": {
            "new_truth_roots_during_manifest": 0,
            "new_truth_roots_during_next_execution_max": AUTHENTIC_ROOT_BUDGET,
            "new_continuous_rate_calls": 0,
            "new_generator_assemblies": 0,
            "physical_microburst_authorized": False,
            "predictive_cycle_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
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
                    "sha256": _sha(path),
                    "scientific_status": "DEFINITIONS_ONLY",
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
    catalog = _read(CANONICAL_SUMMARY)
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
            "latest_source_parent_commit": PARENT_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def _run() -> dict:
    frozen = _validate_parent(require_clean=True)
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("recenter transition forecast already canonicalized")
    arrays, metrics = _forecast()
    if not metrics["passed"]:
        raise RuntimeError(f"recenter transition forecast failed: {metrics['checks']}")
    contract = _contract()
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(CANONICAL_DIRECTORY / "forecast.npz", **arrays)
    _write_json(CANONICAL_DIRECTORY / "forecast_metrics.json", metrics)
    _write_json(CANONICAL_DIRECTORY / "contract.json", contract)
    _write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_commit": PARENT_COMMIT,
            "parent_parent": PARENT_PARENT,
            "parent_tree": PARENT_TREE,
            "parent_hashes": frozen["hashes"],
            "warm4_hashes": frozen["warm4_hashes"],
            "direct_field_sha256": _sha(DIRECT_FIELD),
            "direct_certificate_sha256": _sha(DIRECT_CERTIFICATE),
            "warm4_checkpoint_sha256": _sha(WARM4_CHECKPOINT),
            "warm4_validation_arrays_sha256": _sha(WARM4_VALIDATION_ARRAYS),
            "warm4_validation_metrics_sha256": _sha(WARM4_VALIDATION_METRICS),
            "warm4_metrics_sha256": _sha(WARM4_METRICS),
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "retrospective_direct_forecast_passed": True,
        "predicted_trigger_root_index": metrics["predicted_trigger_root_index"],
        "predicted_trigger_elapsed_from_warm4_seconds": metrics[
            "predicted_trigger_elapsed_from_warm4_seconds"
        ],
        "prospective_truth_root_budget": AUTHENTIC_ROOT_BUDGET,
        "new_truth_roots": 0,
        "new_continuous_rate_calls": 0,
        "new_generator_assemblies": 0,
        "propagated_states": 0,
        "predicted_center_became_chart_center": False,
        "physical_microburst_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    source_files = (
        THIS_RUNNER,
        THIS_TEST,
        parent.THIS_RUNNER,
        parent.THIS_TEST,
        parent.manifest.THIS_RUNNER,
        parent.manifest.THIS_TEST,
        warm4.THIS_RUNNER,
        warm4.THIS_TEST,
    )
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "DEFINITIONS_ONLY",
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "source_hashes": {
                relative: _sha(ROOT / relative) for relative in source_files
            },
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    )
    names = tuple(sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()))
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    loads = metrics["predicted_old_decoder_loads"]
    retro = metrics["retrospective_endpoint_relative_errors"]
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Recenter-transition forecast manifest WP10c9d6c7c3b5c4f25cp",
                "",
                "## Classification",
                "",
                f"`{CLASSIFICATION}`",
                "",
                f"The direct field retrospectively predicts warm-3→warm-4 with full/q/z/a relative errors `{retro['full']:.6e}`, `{retro['q162']:.6e}`, `{retro['z280']:.6e}`, and `{retro['a28']:.6e}`.",
                "",
                f"From accepted warm-4 it predicts old-chart loads `{loads['refined_step_1']:.6e}` and `{loads['refined_step_2']:.6e}` after one and two `1e-7 s` intervals. Thus the frozen forecast places the first recenter trigger in the second interval, below the `0.015` hard limit.",
                "",
                "Exactly two fail-fast authentic BDF2 roots are authorized. A predicted state may not become a chart center; only the first accepted authentic state satisfying the frozen trigger may do so.",
                "",
                "No physical microburst, predictive cycle, or reduced slow evolution is authorized.",
                "",
            )
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("pass --run")
    print(json.dumps(_plain(_run()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
