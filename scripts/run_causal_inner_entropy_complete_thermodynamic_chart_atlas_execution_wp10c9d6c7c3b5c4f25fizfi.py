#!/usr/bin/env python3
"""Build and blindly validate the frozen thermodynamic-chart macro atlas."""

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

import run_causal_inner_entropy_complete_conservative_macro_atlas_execution_wp10c9d6c7c3b5c4f25fizfg as raw_execution  # noqa: E402
import run_causal_inner_entropy_complete_hydrostatic_invariant_reconstruction_implementation_wp10c9d6c7c3b5c4f25fizfa as truth_source  # noqa: E402
import run_causal_inner_entropy_complete_thermodynamic_chart_atlas_manifest_wp10c9d6c7c3b5c4f25fizfh as parent  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_macro_atlas import (  # noqa: E402
    ConservativeMacroOutputs,
    OUTPUT_SIZE,
    conservative_ledger_relative_defect,
    macro_output_component_scales,
    pack_macro_outputs,
    restrict_entropy_complete_macro,
    truth_outputs_from_radial_operator,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_radial import (  # noqa: E402
    generalized_maxwell_cattaneo_radial_operator,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_semidiscrete import (  # noqa: E402
    generalized_maxwell_cattaneo_hydrostatic_embedding,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_slow_manifold import (  # noqa: E402
    generalized_maxwell_cattaneo_slow_targets,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_thermodynamic_macro_atlas import (  # noqa: E402
    ThermodynamicAffineMacroAtlas,
    thermodynamic_chart_lift,
    thermodynamic_macro_chart_pullback,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fizfi_"
    "entropy_complete_thermodynamic_chart_atlas_execution"
)
PASS_CLASSIFICATION = (
    "entropy_complete_thermodynamic_chart_conservative_atlas_blindly_validated"
)
FAIL_CLASSIFICATION = "entropy_complete_thermodynamic_chart_atlas_rejected"
AUTHORIZED_NEXT_ON_PASS = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizfj_"
    "entropy_complete_structure_preserving_macro_integrator_manifest"
)
ARTIFACT = (
    "causal_inner_entropy_complete_thermodynamic_chart_atlas_execution_"
    "wp10c9d6c7c3b5c4f25fizfi"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_"
    "THERMODYNAMIC_CHART_ATLAS_EXECUTION_WP10C9D6C7C3B5C4F25FIZFI_"
    "2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_entropy_complete_thermodynamic_chart_atlas_"
    "execution_wp10c9d6c7c3b5c4f25fizfi.py"
)
THIS_TEST = (
    "tests/test_causal_inner_entropy_complete_thermodynamic_chart_atlas_"
    "execution_wp10c9d6c7c3b5c4f25fizfi.py"
)
SOURCE = (
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_generalized_maxwell_cattaneo_macro_atlas.py"
)
SOURCE_TEST = "tests/test_causal_inner_generalized_maxwell_cattaneo_macro_atlas.py"
THERMODYNAMIC_SOURCE = (
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_generalized_maxwell_cattaneo_thermodynamic_macro_atlas.py"
)
THERMODYNAMIC_SOURCE_TEST = (
    "tests/test_causal_inner_generalized_maxwell_cattaneo_"
    "thermodynamic_macro_atlas.py"
)
SOURCE_SHA256 = "a8869394a31ff8056f2b448e10bb42cd043bebde4ab16152c648b25a9f84d1b8"
SOURCE_TEST_SHA256 = "0287e0b4f03be3c241e820f9b7c76ba75c14fc40907421f9b28f4ce000eda4b3"
THERMODYNAMIC_SOURCE_SHA256 = (
    "fcd790c0328f7122c1e63acdb24b98ffe6a31f928d84946a5bf19818ebb9a808"
)
THERMODYNAMIC_SOURCE_TEST_SHA256 = (
    "c9bd83fcdb1d4b9625a2e6703cc04a23b34c2a64525caa049f5b56f313694098"
)
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "3af2ebd46fc055788398f698d401d8c320ac35eb42d18f872ac3e338aeac2cca"
)
TRUTH_ARRAYS = truth_source.CANONICAL_DIRECTORY / "implementation_arrays.npz"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
RANDOM_SEED = 20_260_826


def _utils():
    return parent._utils()


def _validate_parent(*, require_clean: bool) -> dict:
    utils = _utils()
    if (
        utils._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt")
        != PARENT_CHECKSUM_MANIFEST_SHA256
    ):
        raise RuntimeError("thermodynamic-chart manifest checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    contract = utils._read_json(
        parent.CANONICAL_DIRECTORY / "thermodynamic_chart_atlas_contract.json"
    )
    if (
        summary["classification"] != parent.CLASSIFICATION
        or not summary["passed"]
        or not summary["definitions_only"]
        or not summary["raw_coordinate_atlas_rejection_preserved"]
        or not summary["thermodynamic_chart_atlas_execution_authorized"]
        or summary["state_propagation_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["authorized_next"] != WORK_PACKAGE
        or contract["atlas_audit"]["maximum_new_truth_operator_calls"] != 38
    ):
        raise RuntimeError("thermodynamic-chart atlas authorization changed")
    if utils._sha256(ROOT / SOURCE) != SOURCE_SHA256:
        raise RuntimeError("thermodynamic-chart atlas source changed")
    if utils._sha256(ROOT / SOURCE_TEST) != SOURCE_TEST_SHA256:
        raise RuntimeError("thermodynamic-chart atlas source test changed")
    if utils._sha256(ROOT / THERMODYNAMIC_SOURCE) != THERMODYNAMIC_SOURCE_SHA256:
        raise RuntimeError("thermodynamic-chart implementation source changed")
    if (
        utils._sha256(ROOT / THERMODYNAMIC_SOURCE_TEST)
        != THERMODYNAMIC_SOURCE_TEST_SHA256
    ):
        raise RuntimeError("thermodynamic-chart implementation test changed")
    for relative, expected in utils._read_json(
        parent.CANONICAL_DIRECTORY / "provenance.json"
    )["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(
                f"thermodynamic-chart manifest source changed: {relative}"
            )
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("thermodynamic-chart atlas execution needs a clean tree")
    return {"hashes": hashes, "summary": summary, "contract": contract}


def _sample(
    archive, label: str, sample: str
) -> tuple[np.ndarray, np.ndarray, ConservativeMacroOutputs]:
    return raw_execution._sample(archive, label, sample)


def _support_rows(input_cell: int) -> np.ndarray:
    return raw_execution._support_rows(input_cell)


def _relative(value, reference) -> float:
    return raw_execution._relative(value, reference)


def _output_block_defects(
    predicted: ConservativeMacroOutputs, truth: ConservativeMacroOutputs
) -> dict:
    return raw_execution._output_block_defects(predicted, truth)


def _macro_rate_defects(
    predicted: ConservativeMacroOutputs, truth: ConservativeMacroOutputs
) -> dict:
    return raw_execution._macro_rate_defects(predicted, truth)


def _hydrostatic_embedding_defect(context, charts: np.ndarray) -> float:
    rebuilt = np.asarray(
        [
            generalized_maxwell_cattaneo_hydrostatic_embedding(
                chart[:5],
                proper_vertical_frequency=float(
                    context.vertical_frequency.frequency(float(radius))
                ),
            )
            for radius, chart in zip(context.grid.centers, charts, strict=True)
        ]
    )
    return float(
        np.max(np.abs((rebuilt - charts) / truth_source.CHART_SCALES7))
    )


def _execute() -> tuple[dict, dict[str, np.ndarray]]:
    validated = _validate_parent(require_clean=False)
    contract = validated["contract"]
    coordinate_contract = contract["coordinate_redesign"]
    audit = contract["atlas_audit"]
    cost = contract["online_cost"]
    physical_gates = (
        truth_source.fixed_q_implementation.parent._contract()[
            "binding_physical_gates"
        ]
    )
    context_start = time.perf_counter()
    context, _profile, _charts = truth_source.fixed_q_implementation._primary_setup()
    context_seconds = time.perf_counter() - context_start
    with np.load(TRUTH_ARRAYS) as archive:
        anchor_charts, _anchor_targets, base_outputs = _sample(
            archive, "primary_20ms", "base"
        )
        pullback_start = time.perf_counter()
        anchor_macro, coordinate_scales, tangents, pullbacks = (
            thermodynamic_macro_chart_pullback(
                context,
                anchor_charts,
                derivative_step=coordinate_contract["pullback_derivative_step"],
            )
        )
        pullback_seconds = time.perf_counter() - pullback_start
        pullback_conditions = np.linalg.cond(tangents)
        pullback_closure = float(
            np.max(
                np.abs(
                    np.einsum("kij,kjl->kil", pullbacks, tangents)
                    - np.eye(5)[None, :, :]
                )
            )
        )
        output_scales = macro_output_component_scales(base_outputs)
        base_normalized_output = pack_macro_outputs(base_outputs) / output_scales
        zero_lift = thermodynamic_chart_lift(
            context, anchor_charts, np.zeros((16, 5))
        )
        base_chart_defect = float(
            np.max(
                np.abs(
                    (zero_lift - anchor_charts) / truth_source.CHART_SCALES7
                )
            )
        )
        truth_calls = 0
        truth_records: list[dict] = []
        maximum_hydrostatic_defect = 0.0

        def truth_field(coordinate: np.ndarray, *, label: str) -> np.ndarray:
            nonlocal truth_calls, maximum_hydrostatic_defect
            values = np.asarray(coordinate, dtype=float)
            if values.shape != (16, 5) or np.any(~np.isfinite(values)):
                raise ValueError("truth-field chart coordinate is invalid")
            charts = thermodynamic_chart_lift(context, anchor_charts, values)
            hydrostatic_defect = _hydrostatic_embedding_defect(context, charts)
            maximum_hydrostatic_defect = max(
                maximum_hydrostatic_defect, hydrostatic_defect
            )
            slow_targets = generalized_maxwell_cattaneo_slow_targets(
                context, charts
            )
            macro_state = restrict_entropy_complete_macro(slow_targets, charts)
            prefilter = truth_source.adaptive_diagnosis._midpoint_hyperbolicity_audit(
                context, charts
            )
            operator = generalized_maxwell_cattaneo_radial_operator(
                context, charts, quadrature_order=8
            )
            truth_calls += 1
            record = truth_source._operator_record(operator)
            checks = truth_source._physical_checks(record, physical_gates)
            checks.update(
                {
                    "prefilter_eigenvalue": prefilter[
                        "maximum_eigenvalue_imaginary_ratio"
                    ]
                    <= 1.0e-10,
                    "prefilter_eigenvector": prefilter[
                        "maximum_eigenvector_imaginary_ratio"
                    ]
                    <= 1.0e-10,
                    "hydrostatic_embedding": hydrostatic_defect
                    <= audit["maximum_truth_state_constraint_relative_defect"],
                }
            )
            truth_records.append(
                {
                    "label": label,
                    "call": truth_calls,
                    "maximum_chart_coordinate": float(np.max(np.abs(values))),
                    "hydrostatic_embedding_scaled_defect": hydrostatic_defect,
                    "minimum_macro_MJE": float(np.min(macro_state[:, :3])),
                    "physical": record,
                    "checks": checks,
                    "passed": all(checks.values()),
                }
            )
            print(
                f"truth call {truth_calls}/{audit['maximum_new_truth_operator_calls']}: "
                f"{label} {'passed' if all(checks.values()) else 'failed'}",
                flush=True,
            )
            return (
                pack_macro_outputs(truth_outputs_from_radial_operator(operator))
                / output_scales
            )

        jacobian = np.zeros((OUTPUT_SIZE, 80), dtype=float)
        colored_leakage_ratios = []
        colored_step = audit["central_colored_chart_step"]
        for field in range(5):
            for color in range(3):
                direction = np.zeros((16, 5), dtype=float)
                active = list(range(color, 16, 3))
                direction[active, field] = 1.0
                plus = truth_field(
                    colored_step * direction,
                    label=f"colored_f{field}_c{color}_plus",
                )
                minus = truth_field(
                    -colored_step * direction,
                    label=f"colored_f{field}_c{color}_minus",
                )
                derivative = (plus - minus) / (2.0 * colored_step)
                assigned = np.zeros(OUTPUT_SIZE, dtype=bool)
                for cell in active:
                    rows = _support_rows(cell)
                    jacobian[rows, 5 * cell + field] = derivative[rows]
                    assigned[rows] = True
                leakage = (
                    float(np.max(np.abs(derivative[~assigned])))
                    if np.any(~assigned)
                    else 0.0
                )
                derivative_scale = max(
                    float(np.max(np.abs(derivative))), np.finfo(float).tiny
                )
                colored_leakage_ratios.append(leakage / derivative_scale)

        generator = np.random.default_rng(RANDOM_SEED)
        jvp_directions = generator.normal(
            size=(audit["independent_JVP_directions"], 16, 5)
        )
        jvp_directions /= np.max(
            np.abs(jvp_directions), axis=(1, 2)
        )[:, None, None]
        jvp_defects = []
        jvp_step = audit["independent_JVP_chart_step"]
        for index, direction in enumerate(jvp_directions):
            plus = truth_field(jvp_step * direction, label=f"JVP_{index}_plus")
            minus = truth_field(-jvp_step * direction, label=f"JVP_{index}_minus")
            finite = (plus - minus) / (2.0 * jvp_step)
            analytic = jacobian @ direction.ravel()
            jvp_defects.append(_relative(finite, analytic))

        atlas = ThermodynamicAffineMacroAtlas(
            anchor_macro_state=anchor_macro,
            macro_coordinate_scales=coordinate_scales,
            base_normalized_output=base_normalized_output,
            normalized_output_jacobian=jacobian,
            output_component_scales=output_scales,
            trust_coordinate_infinity=audit[
                "maximum_blind_inferred_chart_coordinate_infinity"
            ],
            macro_coordinate_pullback=pullbacks,
        )

        validation_records = {}
        validation_arrays: dict[str, np.ndarray] = {}
        for label, sample, output_gate in (
            (
                "primary_20ms",
                "perturbed",
                audit["maximum_saved_near_witness_output_relative_defect"],
            ),
            (
                "heldout_16ms",
                "base",
                audit["maximum_blind_output_relative_defect_per_block"],
            ),
            (
                "heldout_16ms",
                "perturbed",
                audit["maximum_blind_output_relative_defect_per_block"],
            ),
        ):
            charts, targets, truth = _sample(archive, label, sample)
            state = restrict_entropy_complete_macro(targets, charts)
            raw_coordinate = (state - anchor_macro) / coordinate_scales
            chart_coordinate = np.einsum(
                "kij,kj->ki", pullbacks, raw_coordinate
            )
            predicted = atlas.evaluate(state)
            block_defects = _output_block_defects(predicted, truth)
            rate_defects = _macro_rate_defects(predicted, truth)
            ledger_defect = conservative_ledger_relative_defect(predicted)
            coordinate_infinity = float(np.max(np.abs(chart_coordinate)))
            is_blind = label == "heldout_16ms"
            passed = bool(
                max(block_defects.values()) <= output_gate
                and (
                    not is_blind
                    or max(rate_defects.values())
                    <= audit[
                        "maximum_blind_macro_rate_relative_defect_per_field"
                    ]
                )
                and coordinate_infinity
                <= audit["maximum_blind_inferred_chart_coordinate_infinity"]
                and ledger_defect <= 1.0e-12
            )
            key = f"{label}_{sample}"
            validation_records[key] = {
                "strict_blind": is_blind,
                "passed": passed,
                "inferred_chart_coordinate_infinity": coordinate_infinity,
                "output_block_relative_defects": block_defects,
                "macro_rate_relative_defects": rate_defects,
                "conservative_ledger_relative_defect": ledger_defect,
            }
            validation_arrays.update(
                {
                    f"{key}_macro_state": state,
                    f"{key}_inferred_chart_coordinate": chart_coordinate,
                    f"{key}_truth_outputs": pack_macro_outputs(truth),
                    f"{key}_predicted_outputs": pack_macro_outputs(predicted),
                    f"{key}_truth_macro_rates_per_second": (
                        truth.macro_rates_per_second
                    ),
                    f"{key}_predicted_macro_rates_per_second": (
                        predicted.macro_rates_per_second
                    ),
                }
            )

    benchmark_states = (
        anchor_macro,
        validation_arrays["heldout_16ms_base_macro_state"],
    )
    benchmark_start = time.perf_counter()
    benchmark_checksum = 0.0
    for index in range(cost["benchmark_evaluations"]):
        output = atlas.evaluate(benchmark_states[index & 1])
        benchmark_checksum += float(
            output.macro_rates_per_second[index % 16, index % 5]
        )
    benchmark_seconds = time.perf_counter() - benchmark_start
    all_truth_passed = all(record["passed"] for record in truth_records)
    all_validation_passed = all(
        record["passed"] for record in validation_records.values()
    )
    passed = bool(
        truth_calls == audit["maximum_new_truth_operator_calls"]
        and all_truth_passed
        and float(np.max(pullback_conditions))
        <= coordinate_contract["maximum_pullback_condition_number"]
        and pullback_closure
        <= audit["maximum_truth_state_constraint_relative_defect"]
        and base_chart_defect
        <= audit["maximum_truth_state_constraint_relative_defect"]
        and max(jvp_defects) <= audit["maximum_independent_JVP_relative_defect"]
        and all_validation_passed
        and benchmark_seconds <= cost["maximum_benchmark_wall_seconds"]
    )
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,
        "passed": passed,
        "context_construction_wall_seconds": context_seconds,
        "pullback_construction_wall_seconds": pullback_seconds,
        "base_anchor_chart_reproduction_scaled_infinity": base_chart_defect,
        "pullback_condition_numbers": pullback_conditions.tolist(),
        "maximum_pullback_condition_number": float(np.max(pullback_conditions)),
        "pullback_inverse_closure_infinity": pullback_closure,
        "new_truth_operator_calls": truth_calls,
        "all_truth_physical_gates_passed": all_truth_passed,
        "maximum_truth_hydrostatic_embedding_scaled_defect": (
            maximum_hydrostatic_defect
        ),
        "truth_records": truth_records,
        "maximum_colored_support_leakage_ratio": max(colored_leakage_ratios),
        "independent_JVP_relative_defects": jvp_defects,
        "maximum_independent_JVP_relative_defect": max(jvp_defects),
        "validation": validation_records,
        "all_saved_and_blind_validations_passed": all_validation_passed,
        "online_benchmark_evaluations": cost["benchmark_evaluations"],
        "online_benchmark_wall_seconds": benchmark_seconds,
        "online_average_wall_seconds_per_evaluation": (
            benchmark_seconds / cost["benchmark_evaluations"]
        ),
        "online_benchmark_checksum": benchmark_checksum,
        "new_global_roots": 0,
        "propagated_states": 0,
    }
    arrays = {
        "anchor_macro_state": anchor_macro,
        "macro_coordinate_scales": coordinate_scales,
        "macro_chart_tangents": tangents,
        "macro_coordinate_pullbacks": pullbacks,
        "base_normalized_output": base_normalized_output,
        "normalized_output_chart_jacobian": jacobian,
        "output_component_scales": output_scales,
        "JVP_directions": jvp_directions,
        "JVP_relative_defects": np.asarray(jvp_defects),
        **validation_arrays,
    }
    return metrics, arrays


def _update_catalog(summary: dict) -> None:
    utils = _utils()
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
                    "sha256": utils._sha256(path),
                    "scientific_status": (
                        "SUPPORTED" if summary["passed"] else "REJECTED"
                    ),
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
        raise RuntimeError("thermodynamic-chart atlas execution already exists")
    validated = _validate_parent(require_clean=True)
    utils = _utils()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "macro_atlas_metrics.json", metrics)
    np.savez_compressed(CANONICAL_DIRECTORY / "macro_atlas_arrays.npz", **arrays)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": bool(metrics["passed"]),
        "raw_coordinate_atlas_rejection_preserved": True,
        "hydrostatic_implicit_inverse_tangent_preserved": True,
        "thermodynamic_chart_conservative_macro_atlas_certified": bool(
            metrics["passed"]
        ),
        "heldout_16ms_profiles_passed": bool(metrics["passed"]),
        "online_truth_calls_per_macrostep": 0,
        "new_global_roots": 0,
        "propagated_states": 0,
        "structure_preserving_macro_integrator_manifest_authorized": bool(
            metrics["passed"]
        ),
        "macro_integrator_execution_authorized": False,
        "complete_cycle_execution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT_ON_PASS if metrics["passed"] else None,
    }
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utils._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_artifact": parent.ARTIFACT,
            "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256,
            "parent_hashes": validated["hashes"],
            "source_sha256": SOURCE_SHA256,
            "source_test_sha256": SOURCE_TEST_SHA256,
            "thermodynamic_source_sha256": THERMODYNAMIC_SOURCE_SHA256,
            "thermodynamic_source_test_sha256": THERMODYNAMIC_SOURCE_TEST_SHA256,
            "truth_arrays_sha256": utils._sha256(TRUTH_ARRAYS),
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    validation = metrics.get("validation", {})
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Entropy-complete thermodynamic-chart macro-atlas execution",
                "",
                f"Classification: `{metrics['classification']}`.",
                "",
                "The raw independent M/J/E atlas rejection remains preserved. "
                "Offline perturbations were generated only through admissible "
                "hydrostatic thermodynamic charts, while the online state remains "
                "the exact 16-cell M/J/E/beta_r/chi ledger.",
                "",
                f"Truth calls: `{metrics['new_truth_operator_calls']}`; maximum "
                f"pullback condition: `{metrics['maximum_pullback_condition_number']:.6e}`; "
                f"maximum independent JVP defect: "
                f"`{metrics['maximum_independent_JVP_relative_defect']:.6e}`.",
                "",
                f"Primary near witness passed: "
                f"`{validation.get('primary_20ms_perturbed', {}).get('passed', False)}`; "
                f"held-out base passed: "
                f"`{validation.get('heldout_16ms_base', {}).get('passed', False)}`; "
                f"held-out perturbed passed: "
                f"`{validation.get('heldout_16ms_perturbed', {}).get('passed', False)}`.",
                "",
                f"The `{metrics['online_benchmark_evaluations']}`-evaluation online "
                f"benchmark took `{metrics['online_benchmark_wall_seconds']:.6f}` s "
                "with zero online truth calls, roots, or propagation.",
                "",
                f"Authorized next: `{summary['authorized_next']}`.",
                "",
            )
        ),
        encoding="utf-8",
    )
    sources = (
        THIS_RUNNER,
        THIS_TEST,
        SOURCE,
        SOURCE_TEST,
        THERMODYNAMIC_SOURCE,
        THERMODYNAMIC_SOURCE_TEST,
        REPORT_RELATIVE,
    )
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
