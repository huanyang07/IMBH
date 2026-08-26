#!/usr/bin/env python3
"""Execute the bounded 80-coordinate conservative transient acquisition."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_entropy_complete_adaptive_selective_refresh_cycle_readiness_manifest_wp10c9d6c7c3b5c4f25fizr as parent  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_macro_atlas import (  # noqa: E402
    conservative_ledger_relative_defect,
    pack_macro_outputs,
    truth_outputs_from_radial_operator,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_macro_integrator import (  # noqa: E402
    macro_rate_output_matrix,
    reconstruct_thermodynamic_macro_state,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_radial import (  # noqa: E402
    generalized_maxwell_cattaneo_radial_operator,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_thermodynamic_macro_atlas import (  # noqa: E402
    thermodynamic_macro_chart_pullback,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fizs_"
    "entropy_complete_bounded_full_bundle_transient_acquisition_execution"
)
SLAVING_CLASSIFICATION = (
    "entropy_complete_persistent_auxiliary_slaving_observed_in_bounded_transient"
)
OPEN_CLASSIFICATION = (
    "entropy_complete_bounded_full_bundle_transient_extended_without_slaving"
)
FAIL_CLASSIFICATION = "entropy_complete_bounded_full_bundle_transient_failed"
SLAVING_AUTHORIZED_NEXT = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizt_"
    "entropy_complete_terminal_fast_graph_tangent_certificate_manifest"
)
OPEN_AUTHORIZED_NEXT = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizt_"
    "entropy_complete_cost_bounded_full_bundle_continuation_manifest"
)
ARTIFACT = (
    "causal_inner_entropy_complete_bounded_full_bundle_transient_acquisition_"
    "execution_wp10c9d6c7c3b5c4f25fizs"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_BOUNDED_FULL_"
    "BUNDLE_TRANSIENT_ACQUISITION_EXECUTION_"
    "WP10C9D6C7C3B5C4F25FIZS_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_entropy_complete_bounded_full_bundle_transient_"
    "acquisition_execution_wp10c9d6c7c3b5c4f25fizs.py"
)
THIS_TEST = (
    "tests/test_causal_inner_entropy_complete_bounded_full_bundle_transient_"
    "acquisition_execution_wp10c9d6c7c3b5c4f25fizs.py"
)
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "088c8b5a215288765c3b2e69930e7c8aaa2cf03c16e8813d2493b003adc683e1"
)
PARENT_ARRAYS = parent.PARENT_ARRAYS
PATCH_2_ARRAYS = parent.parent.PATCH_2_ARRAYS
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _utils():
    return parent._utils()


def _relative(defect, *references) -> float:
    scale = max(
        *(float(np.max(np.abs(np.asarray(item)))) for item in references),
        np.finfo(float).tiny,
    )
    return float(np.max(np.abs(np.asarray(defect))) / scale)


def _ab2_candidate(
    current_state: np.ndarray,
    current_rate: np.ndarray,
    previous_rate: np.ndarray,
    timestep_seconds: float,
) -> np.ndarray:
    """Return the equal-step accepted-history-only AB2 candidate."""

    state = np.asarray(current_state, dtype=float)
    current = np.asarray(current_rate, dtype=float)
    previous = np.asarray(previous_rate, dtype=float)
    timestep = float(timestep_seconds)
    if (
        state.shape != (16, 5)
        or current.shape != state.shape
        or previous.shape != state.shape
        or any(np.any(~np.isfinite(item)) for item in (state, current, previous))
        or not np.isfinite(timestep)
        or timestep <= 0.0
    ):
        raise ValueError("AB2 transient inputs are invalid")
    return state + timestep * (1.5 * current - 0.5 * previous)


def _embedded_defect(
    current_state: np.ndarray,
    candidate_state: np.ndarray,
    current_rate: np.ndarray,
    candidate_rate: np.ndarray,
    coordinate_scales: np.ndarray,
    timestep_seconds: float,
) -> float:
    """Compare AB2 against the endpoint trapezoidal estimate."""

    current = np.asarray(current_state, dtype=float)
    candidate = np.asarray(candidate_state, dtype=float)
    rate_0 = np.asarray(current_rate, dtype=float)
    rate_1 = np.asarray(candidate_rate, dtype=float)
    scales = np.asarray(coordinate_scales, dtype=float)
    trapezoidal = current + 0.5 * float(timestep_seconds) * (rate_0 + rate_1)
    if any(item.shape != (16, 5) for item in (current, candidate, rate_0, rate_1, scales)):
        raise ValueError("embedded transient arrays have the wrong shape")
    return float(np.max(np.abs((candidate - trapezoidal) / scales)))


def _slaving_record(
    macro_rate: np.ndarray,
    coordinate_scales: np.ndarray,
    contract: dict,
) -> dict:
    normalized = np.asarray(macro_rate, dtype=float) / np.asarray(
        coordinate_scales, dtype=float
    )
    slow = float(np.max(np.abs(normalized[:, :3])))
    auxiliary = float(np.max(np.abs(normalized[:, 3:])))
    ratio = auxiliary / max(slow, np.finfo(float).tiny)
    gates = contract["slaving_observation"]
    return {
        "normalized_conservative_rate_infinity_per_second": slow,
        "normalized_auxiliary_rate_infinity_per_second": auxiliary,
        "auxiliary_to_conservative_rate_ratio": ratio,
        "instantaneous_slaving_observation_passed": bool(
            auxiliary
            <= gates["normalized_auxiliary_rate_infinity_per_second_maximum"]
            and ratio
            <= gates["normalized_auxiliary_to_conservative_rate_ratio_maximum"]
        ),
    }


def _validate_parent(*, require_clean: bool) -> dict:
    utils = _utils()
    if (
        utils._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt")
        != PARENT_CHECKSUM_MANIFEST_SHA256
    ):
        raise RuntimeError("cycle-readiness manifest checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    contract = utils._read_json(
        parent.CANONICAL_DIRECTORY / "cycle_readiness_contract.json"
    )
    if (
        summary["classification"] != parent.CLASSIFICATION
        or not summary["passed"]
        or not summary["two_regime_architecture_selected"]
        or not summary["bounded_transient_execution_authorized"]
        or summary["complete_cycle_execution_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["authorized_next"] != WORK_PACKAGE
        or contract["bounded_execution"]["new_macrosteps"] != 50
        or contract["bounded_execution"]["maximum_new_truth_operator_calls"]
        != 50
    ):
        raise RuntimeError("bounded transient authorization changed")
    for relative, expected in utils._read_json(
        parent.CANONICAL_DIRECTORY / "provenance.json"
    )["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"cycle-readiness source changed: {relative}")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("bounded transient execution needs a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "contract": contract}


def _execute() -> tuple[dict, dict[str, np.ndarray]]:
    validated = _validate_parent(require_clean=False)
    contract = validated["contract"]
    execution = contract["bounded_execution"]
    numerical = contract["numerical_gates"]
    timestep = execution["fixed_macrostep_seconds"]
    with np.load(PARENT_ARRAYS) as archive:
        parent_arrays = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(PATCH_2_ARRAYS) as archive:
        patch_2_arrays = {name: np.asarray(archive[name]) for name in archive.files}
    state_history = [
        np.array(parent_arrays["combined_macro_states_0_to_12ms"][-1], copy=True)
    ]
    chart_history = [
        np.array(parent_arrays["endpoint_12ms_primitive_charts"], copy=True)
    ]
    packed_output_history = [
        np.array(parent_arrays["endpoint_12ms_truth_packed_outputs"], copy=True)
    ]
    rate_output = macro_rate_output_matrix()
    current_rate = (
        rate_output @ packed_output_history[-1]
    ).reshape(16, 5)
    previous_rate = (
        rate_output @ patch_2_arrays["endpoint_8ms_truth_packed_outputs"]
    ).reshape(16, 5)
    rate_history = [np.array(current_rate, copy=True)]
    truth_execution = parent.parent.rejected_execution.truth_execution
    truth_source = truth_execution.truth_source
    physical_gates = truth_source.fixed_q_implementation.parent._contract()[
        "binding_physical_gates"
    ]
    context_start = time.perf_counter()
    context, _profile, _initial = truth_source.fixed_q_implementation._primary_setup()
    context_seconds = time.perf_counter() - context_start
    truth_calls = 0
    records = []
    packed_integrals = []
    slaving_streak = 0
    maximum_slaving_streak = 0
    execution_start = time.perf_counter()
    failure_reason = None
    for step in range(1, execution["new_macrosteps"] + 1):
        step_start = time.perf_counter()
        current_state = state_history[-1]
        current_charts = chart_history[-1]
        anchor_macro, coordinate_scales, _tangents, pullbacks = (
            thermodynamic_macro_chart_pullback(
                context, current_charts, derivative_step=1.0e-5
            )
        )
        anchor_defect = _relative(anchor_macro - current_state, anchor_macro, current_state)
        candidate_state = _ab2_candidate(
            current_state, current_rate, previous_rate, timestep
        )
        record = {
            "step": step,
            "absolute_elapsed_seconds": execution[
                "initial_absolute_elapsed_seconds"
            ]
            + step * timestep,
            "anchor_roundtrip_relative_defect": anchor_defect,
        }
        try:
            reconstruction = reconstruct_thermodynamic_macro_state(
                context,
                current_charts,
                candidate_state,
                anchor_macro_state=anchor_macro,
                macro_coordinate_scales=coordinate_scales,
                macro_coordinate_pullbacks=pullbacks,
                derivative_step=1.0e-5,
                maximum_newton_corrections=8,
                relative_tolerance=numerical[
                    "maximum_macro_roundtrip_relative_defect"
                ],
                maximum_chart_coordinate_infinity=numerical[
                    "maximum_local_reconstruction_chart_coordinate"
                ],
            )
            prefilter = truth_source.adaptive_diagnosis._midpoint_hyperbolicity_audit(
                context, reconstruction.primitive_charts
            )
            operator = generalized_maxwell_cattaneo_radial_operator(
                context, reconstruction.primitive_charts, quadrature_order=8
            )
            truth_calls += 1
            physical = truth_source._operator_record(operator)
            physical_checks = truth_source._physical_checks(physical, physical_gates)
            physical_checks.update(
                {
                    "prefilter_eigenvalue": prefilter[
                        "maximum_eigenvalue_imaginary_ratio"
                    ]
                    <= 1.0e-10,
                    "prefilter_eigenvector": prefilter[
                        "maximum_eigenvector_imaginary_ratio"
                    ]
                    <= 1.0e-10,
                    "hydrostatic_embedding": parent.parent.rejected_execution._hydrostatic_embedding_defect(
                        context, reconstruction.primitive_charts
                    )
                    <= 1.0e-10,
                }
            )
            outputs = truth_outputs_from_radial_operator(operator)
            packed_output = pack_macro_outputs(outputs)
            candidate_rate = (rate_output @ packed_output).reshape(16, 5)
            embedded = _embedded_defect(
                current_state,
                candidate_state,
                current_rate,
                candidate_rate,
                coordinate_scales,
                timestep,
            )
            integrated_output = timestep * (
                1.5 * packed_output_history[-1]
                - 0.5 * patch_2_arrays["endpoint_8ms_truth_packed_outputs"]
                if step == 1
                else 1.5 * packed_output_history[-1]
                - 0.5 * packed_output_history[-2]
            )
            ledger_change = (rate_output @ integrated_output).reshape(16, 5)
            actual_change = candidate_state - current_state
            ledger_defect = _relative(
                ledger_change - actual_change, ledger_change, actual_change
            )
            slaving = _slaving_record(candidate_rate, coordinate_scales, contract)
            if slaving["instantaneous_slaving_observation_passed"]:
                slaving_streak += 1
            else:
                slaving_streak = 0
            maximum_slaving_streak = max(maximum_slaving_streak, slaving_streak)
            maximum_chart = float(
                np.max(np.abs(reconstruction.chart_coordinates))
            )
            step_passed = bool(
                anchor_defect
                <= numerical["maximum_macro_roundtrip_relative_defect"]
                and reconstruction.maximum_macro_state_roundtrip_relative_defect
                <= numerical["maximum_macro_roundtrip_relative_defect"]
                and maximum_chart
                <= numerical["maximum_step_scaled_coordinate_change"]
                and embedded
                <= numerical["maximum_AB2_trapezoidal_embedded_defect"]
                and ledger_defect
                <= numerical["maximum_discrete_conservative_ledger_relative_defect"]
                and all(physical_checks.values())
                and conservative_ledger_relative_defect(outputs)
                <= numerical["maximum_discrete_conservative_ledger_relative_defect"]
            )
            record.update(
                {
                    "passed": step_passed,
                    "truth_call": truth_calls,
                    "maximum_chart_coordinate": maximum_chart,
                    "macro_roundtrip_relative_defect": reconstruction.maximum_macro_state_roundtrip_relative_defect,
                    "reconstruction_newton_corrections": reconstruction.newton_corrections,
                    "embedded_defect": embedded,
                    "discrete_ledger_relative_defect": ledger_defect,
                    "truth_output_ledger_relative_defect": conservative_ledger_relative_defect(outputs),
                    "physical": physical,
                    "physical_checks": physical_checks,
                    "slaving": slaving,
                    "slaving_streak": slaving_streak,
                    "wall_seconds": time.perf_counter() - step_start,
                }
            )
            records.append(record)
            print(
                f"transient step {step}/{execution['new_macrosteps']}: "
                f"{'passed' if step_passed else 'failed'} "
                f"chart={maximum_chart:.6e} embedded={embedded:.6e} "
                f"aux={slaving['normalized_auxiliary_rate_infinity_per_second']:.6e}",
                flush=True,
            )
            if not step_passed:
                failure_reason = "binding_step_gate_failed"
                break
            state_history.append(np.array(candidate_state, copy=True))
            chart_history.append(
                np.array(reconstruction.primitive_charts, copy=True)
            )
            packed_output_history.append(np.array(packed_output, copy=True))
            rate_history.append(np.array(candidate_rate, copy=True))
            packed_integrals.append(np.array(integrated_output, copy=True))
            previous_rate = current_rate
            current_rate = candidate_rate
        except Exception as exc:  # fail-closed canonical diagnosis
            record.update(
                {
                    "passed": False,
                    "exception": f"{type(exc).__name__}: {exc}",
                    "wall_seconds": time.perf_counter() - step_start,
                }
            )
            records.append(record)
            failure_reason = "candidate_reconstruction_or_truth_evaluation_failed"
            print(
                f"transient step {step}/{execution['new_macrosteps']}: failed "
                f"with {record['exception']}",
                flush=True,
            )
            break
    accepted_steps = len(state_history) - 1
    completed = accepted_steps == execution["new_macrosteps"]
    persistent = bool(
        completed
        and maximum_slaving_streak
        >= contract["slaving_observation"][
            "required_consecutive_accepted_steps"
        ]
    )
    passed = completed
    classification = (
        SLAVING_CLASSIFICATION
        if persistent
        else OPEN_CLASSIFICATION
        if completed
        else FAIL_CLASSIFICATION
    )
    terminal_slaving = (
        _slaving_record(rate_history[-1], coordinate_scales, contract)
        if rate_history
        else None
    )
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "failure_reason": failure_reason,
        "context_construction_wall_seconds": context_seconds,
        "execution_wall_seconds": time.perf_counter() - execution_start,
        "attempted_steps": len(records),
        "accepted_steps": accepted_steps,
        "new_truth_operator_calls": truth_calls,
        "accepted_absolute_horizon_seconds": execution[
            "initial_absolute_elapsed_seconds"
        ]
        + accepted_steps * timestep,
        "step_records": records,
        "maximum_slaving_streak": maximum_slaving_streak,
        "persistent_auxiliary_slaving_observed": persistent,
        "terminal_slaving": terminal_slaving,
        "maximum_accepted_chart_coordinate": max(
            (record.get("maximum_chart_coordinate", 0.0) for record in records if record["passed"]),
            default=0.0,
        ),
        "maximum_accepted_embedded_defect": max(
            (record.get("embedded_defect", 0.0) for record in records if record["passed"]),
            default=0.0,
        ),
        "maximum_accepted_discrete_ledger_relative_defect": max(
            (record.get("discrete_ledger_relative_defect", 0.0) for record in records if record["passed"]),
            default=0.0,
        ),
        "new_global_roots": 0,
        "fixed_Q_reaction_calls": 0,
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    arrays = {
        "accepted_macro_states": np.asarray(state_history),
        "accepted_primitive_charts": np.asarray(chart_history),
        "accepted_truth_packed_outputs": np.asarray(packed_output_history),
        "accepted_macro_rates_per_second": np.asarray(rate_history),
        "accepted_integrated_packed_outputs": np.asarray(packed_integrals),
        "terminal_previous_macro_rate_per_second": previous_rate,
        "terminal_current_macro_rate_per_second": current_rate,
    }
    return metrics, arrays


def _update_catalog(summary: dict) -> None:
    utils = _utils()
    with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
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
                    "sha256": utils._sha256(path),
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
    catalog = utils._read_json(CANONICAL_SUMMARY)
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
            "latest_source_parent_commit": utils._git("rev-parse", "HEAD"),
            "latest_work_package": WORK_PACKAGE,
        }
    )
    utils._write_json(CANONICAL_SUMMARY, catalog)


def _canonicalize(metrics: dict, arrays: dict[str, np.ndarray]) -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("bounded transient result already exists")
    validated = _validate_parent(require_clean=True)
    utils = _utils()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "transient_metrics.json", metrics)
    np.savez_compressed(CANONICAL_DIRECTORY / "transient_arrays.npz", **arrays)
    authorized_next = (
        SLAVING_AUTHORIZED_NEXT
        if metrics["persistent_auxiliary_slaving_observed"]
        else OPEN_AUTHORIZED_NEXT
        if metrics["passed"]
        else None
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": bool(metrics["passed"]),
        "accepted_steps": metrics["accepted_steps"],
        "accepted_absolute_horizon_seconds": metrics[
            "accepted_absolute_horizon_seconds"
        ],
        "persistent_auxiliary_slaving_observed": metrics[
            "persistent_auxiliary_slaving_observed"
        ],
        "terminal_fast_graph_tangent_manifest_authorized": bool(
            metrics["persistent_auxiliary_slaving_observed"]
        ),
        "bounded_continuation_manifest_authorized": bool(
            metrics["passed"]
            and not metrics["persistent_auxiliary_slaving_observed"]
        ),
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": authorized_next,
    }
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utils._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_artifact": parent.ARTIFACT,
            "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256,
            "parent_hashes": validated["hashes"],
            "parent_arrays_sha256": utils._sha256(PARENT_ARRAYS),
            "patch_2_arrays_sha256": utils._sha256(PATCH_2_ARRAYS),
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Entropy-complete bounded full-bundle transient acquisition",
                "",
                f"Classification: `{metrics['classification']}`.",
                "",
                f"Accepted `{metrics['accepted_steps']}` of 50 prospectively frozen 4 ms AB2 steps, reaching `{metrics['accepted_absolute_horizon_seconds']:.6e}` s with `{metrics['new_truth_operator_calls']}` exact seven-field truth calls.",
                "",
                f"Maximum accepted chart/embedded/ledger defects were `{metrics['maximum_accepted_chart_coordinate']:.6e}`, `{metrics['maximum_accepted_embedded_defect']:.6e}`, and `{metrics['maximum_accepted_discrete_ledger_relative_defect']:.6e}`.",
                "",
                f"Persistent auxiliary slaving observed: `{metrics['persistent_auxiliary_slaving_observed']}`. No complete-cycle execution or reduced slow evolution is authorized.",
                "",
                f"Authorized next: `{authorized_next}`.",
                "",
            )
        ),
        encoding="utf-8",
    )
    sources = (THIS_RUNNER, THIS_TEST, REPORT_RELATIVE)
    utils._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "implementation_commit": utils._git("rev-parse", "HEAD"),
            "implementation_tree": utils._git("rev-parse", "HEAD^{tree}"),
            "source_hashes": {
                path: utils._sha256(ROOT / path) for path in sources
            },
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "thread_environment": {
                name: os.environ.get(name, "")
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
        },
    )
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{utils._sha256(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("choose --run")
    metrics, arrays = _execute()
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    summary = _canonicalize(metrics, arrays)
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
