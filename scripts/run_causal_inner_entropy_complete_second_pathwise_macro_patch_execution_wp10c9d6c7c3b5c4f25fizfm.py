#!/usr/bin/env python3
"""Build a second pathwise atlas patch and validate the 8 ms endpoint."""

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

import run_causal_inner_entropy_complete_pathwise_macro_atlas_expansion_manifest_wp10c9d6c7c3b5c4f25fizfl as parent  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_macro_atlas import (  # noqa: E402
    OUTPUT_SIZE,
    conservative_ledger_relative_defect,
    macro_output_component_scales,
    pack_macro_outputs,
    restrict_entropy_complete_macro,
    truth_outputs_from_radial_operator,
    unpack_macro_outputs,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_macro_integrator import (  # noqa: E402
    ExactAffineMacroSystem,
    ExactAffineMacroTransition,
    reconstruct_thermodynamic_macro_state,
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
    "WP10c9d6c7c3b5c4f25fizfm_"
    "entropy_complete_second_pathwise_macro_patch_execution"
)
PASS_CLASSIFICATION = (
    "entropy_complete_two_patch_macro_path_and_8ms_truth_endpoint_certified"
)
FAIL_CLASSIFICATION = "entropy_complete_second_pathwise_macro_patch_rejected"
AUTHORIZED_NEXT_ON_PASS = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizfn_"
    "entropy_complete_multi_patch_growth_and_fast_slaving_manifest"
)
ARTIFACT = (
    "causal_inner_entropy_complete_second_pathwise_macro_patch_execution_"
    "wp10c9d6c7c3b5c4f25fizfm"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_SECOND_PATHWISE_"
    "MACRO_PATCH_EXECUTION_WP10C9D6C7C3B5C4F25FIZFM_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_entropy_complete_second_pathwise_macro_patch_"
    "execution_wp10c9d6c7c3b5c4f25fizfm.py"
)
THIS_TEST = (
    "tests/test_causal_inner_entropy_complete_second_pathwise_macro_patch_"
    "execution_wp10c9d6c7c3b5c4f25fizfm.py"
)
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "043b2f75e589cfd433003afec2dcc4b8a886c9022dca02c56f9d339066392b94"
)
INTEGRATOR_ARRAYS = (
    parent.parent.CANONICAL_DIRECTORY / "macro_integrator_arrays.npz"
)
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
RANDOM_SEED = 20_260_827


def _utils():
    return parent._utils()


def _validate_parent(*, require_clean: bool) -> dict:
    utils = _utils()
    if (
        utils._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt")
        != PARENT_CHECKSUM_MANIFEST_SHA256
    ):
        raise RuntimeError("pathwise patch manifest checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    contract = utils._read_json(
        parent.CANONICAL_DIRECTORY / "pathwise_patch_contract.json"
    )
    if (
        summary["classification"] != parent.CLASSIFICATION
        or not summary["passed"]
        or not summary["definitions_only"]
        or not summary["second_patch_execution_authorized"]
        or summary["unbounded_pathwise_continuation_authorized"]
        or summary["complete_cycle_execution_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["authorized_next"] != WORK_PACKAGE
        or contract["budgets"]["maximum_new_truth_operator_calls"] != 39
    ):
        raise RuntimeError("second pathwise patch authorization changed")
    for relative, expected in utils._read_json(
        parent.CANONICAL_DIRECTORY / "provenance.json"
    )["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"pathwise patch manifest source changed: {relative}")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("second pathwise patch execution needs a clean tree")
    return {"hashes": hashes, "summary": summary, "contract": contract}


def _support_rows(input_cell: int) -> np.ndarray:
    return parent.parent.parent.parent._support_rows(input_cell)


def _relative(value, reference) -> float:
    return parent.parent._relative(value, reference)


def _output_block_defects(predicted, truth) -> dict:
    return parent.parent.parent.parent._output_block_defects(predicted, truth)


def _macro_rate_defects(predicted, truth) -> dict:
    return parent.parent.parent.parent._macro_rate_defects(predicted, truth)


def _patch_1_atlas(arrays) -> ThermodynamicAffineMacroAtlas:
    return ThermodynamicAffineMacroAtlas(
        anchor_macro_state=np.asarray(arrays["atlas_anchor_macro_state"]),
        macro_coordinate_scales=np.asarray(arrays["atlas_macro_coordinate_scales"]),
        base_normalized_output=np.asarray(arrays["atlas_base_normalized_output"]),
        normalized_output_jacobian=np.asarray(
            arrays["atlas_normalized_output_chart_jacobian"]
        ),
        output_component_scales=np.asarray(
            arrays["atlas_output_component_scales"]
        ),
        trust_coordinate_infinity=1.5e-1,
        macro_coordinate_pullback=np.asarray(
            arrays["atlas_macro_coordinate_pullbacks"]
        ),
    )


def _hydrostatic_embedding_defect(context, charts: np.ndarray) -> float:
    truth_source = parent.parent.parent.parent.truth_source
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
    construction = contract["patch_2_construction"]
    validation = contract["overlap_and_dynamic_validation"]
    budgets = contract["budgets"]
    truth_source = parent.parent.parent.parent.truth_source
    physical_gates = truth_source.fixed_q_implementation.parent._contract()[
        "binding_physical_gates"
    ]
    with np.load(INTEGRATOR_ARRAYS) as source:
        inputs = {name: np.asarray(source[name]) for name in source.files}
    patch_1 = _patch_1_atlas(inputs)
    patch_1_states = np.asarray(inputs["macro_states"])
    saved_anchor_target = patch_1_states[-1]
    anchor_charts = np.asarray(inputs["endpoint_reconstructed_primitive_charts"])
    base_outputs = unpack_macro_outputs(inputs["endpoint_truth_packed_outputs"])
    context_start = time.perf_counter()
    context, _profile, _charts = truth_source.fixed_q_implementation._primary_setup()
    context_seconds = time.perf_counter() - context_start
    pullback_start = time.perf_counter()
    anchor_macro, coordinate_scales, tangents, pullbacks = (
        thermodynamic_macro_chart_pullback(
            context,
            anchor_charts,
            derivative_step=construction["pullback_derivative_step"],
        )
    )
    pullback_seconds = time.perf_counter() - pullback_start
    anchor_roundtrip_defect = _relative(anchor_macro, saved_anchor_target)
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
    truth_calls = 0
    truth_records = []

    def truth_field(coordinate: np.ndarray, *, label: str) -> np.ndarray:
        nonlocal truth_calls
        values = np.asarray(coordinate, dtype=float)
        charts = thermodynamic_chart_lift(context, anchor_charts, values)
        hydrostatic_defect = _hydrostatic_embedding_defect(context, charts)
        prefilter = truth_source.adaptive_diagnosis._midpoint_hyperbolicity_audit(
            context, charts
        )
        operator = generalized_maxwell_cattaneo_radial_operator(
            context, charts, quadrature_order=8
        )
        truth_calls += 1
        physical = truth_source._operator_record(operator)
        checks = truth_source._physical_checks(physical, physical_gates)
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
                "hydrostatic_embedding": hydrostatic_defect <= 1.0e-10,
            }
        )
        truth_records.append(
            {
                "call": truth_calls,
                "label": label,
                "maximum_chart_coordinate": float(np.max(np.abs(values))),
                "physical": physical,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
        print(
            f"truth call {truth_calls}/{budgets['maximum_new_truth_operator_calls']}: "
            f"{label} {'passed' if all(checks.values()) else 'failed'}",
            flush=True,
        )
        return pack_macro_outputs(truth_outputs_from_radial_operator(operator)) / output_scales

    jacobian = np.zeros((OUTPUT_SIZE, 80), dtype=float)
    leakage_ratios = []
    step = construction["central_colored_chart_step"]
    for field in range(5):
        for color in range(3):
            direction = np.zeros((16, 5))
            active = list(range(color, 16, 3))
            direction[active, field] = 1.0
            plus = truth_field(step * direction, label=f"colored_f{field}_c{color}_plus")
            minus = truth_field(-step * direction, label=f"colored_f{field}_c{color}_minus")
            derivative = (plus - minus) / (2.0 * step)
            assigned = np.zeros(OUTPUT_SIZE, dtype=bool)
            for cell in active:
                rows = _support_rows(cell)
                jacobian[rows, 5 * cell + field] = derivative[rows]
                assigned[rows] = True
            leakage = float(np.max(np.abs(derivative[~assigned]))) if np.any(~assigned) else 0.0
            leakage_ratios.append(
                leakage
                / max(float(np.max(np.abs(derivative))), np.finfo(float).tiny)
            )
    generator = np.random.default_rng(RANDOM_SEED)
    directions = generator.normal(
        size=(construction["independent_JVP_directions"], 16, 5)
    )
    directions /= np.max(np.abs(directions), axis=(1, 2))[:, None, None]
    jvp_defects = []
    jvp_step = construction["independent_JVP_chart_step"]
    for index, direction in enumerate(directions):
        plus = truth_field(jvp_step * direction, label=f"JVP_{index}_plus")
        minus = truth_field(-jvp_step * direction, label=f"JVP_{index}_minus")
        finite = (plus - minus) / (2.0 * jvp_step)
        jvp_defects.append(_relative(finite, jacobian @ direction.ravel()))
    patch_2 = ThermodynamicAffineMacroAtlas(
        anchor_macro_state=anchor_macro,
        macro_coordinate_scales=coordinate_scales,
        base_normalized_output=base_normalized_output,
        normalized_output_jacobian=jacobian,
        output_component_scales=output_scales,
        trust_coordinate_infinity=construction["atlas_trust_coordinate_infinity"],
        macro_coordinate_pullback=pullbacks,
    )
    overlap_state = patch_1_states[3]
    overlap_1 = patch_1.evaluate(overlap_state)
    overlap_2 = patch_2.evaluate(overlap_state)
    overlap_output_defects = _output_block_defects(overlap_2, overlap_1)
    overlap_rate_defects = _macro_rate_defects(overlap_2, overlap_1)
    overlap_patch_1_coordinate = float(
        np.max(np.abs(patch_1.inferred_chart_coordinate(overlap_state)))
    )
    overlap_patch_2_coordinate = float(
        np.max(np.abs(patch_2.inferred_chart_coordinate(overlap_state)))
    )
    system = ExactAffineMacroSystem.from_atlas(patch_2)
    eigenvalues = np.linalg.eigvals(system.normalized_rate_matrix)
    transition = ExactAffineMacroTransition.build(
        system,
        validation["patch_2_fixed_macrostep_seconds"],
        trust_coordinate_infinity=validation[
            "reserved_trust_coordinate_infinity"
        ],
    )
    patch_2_states = [np.array(anchor_macro, copy=True)]
    integrated_outputs = []
    step_records = []
    for index in range(validation["patch_2_macrosteps"]):
        result = transition.step(patch_2_states[-1])
        patch_2_states.append(result.macro_state)
        integrated_outputs.append(result.integrated_packed_output)
        step_records.append(
            {
                "step": index + 1,
                "absolute_elapsed_seconds": 4.0e-3
                + (index + 1) * validation["patch_2_fixed_macrostep_seconds"],
                "maximum_chart_coordinate": result.maximum_endpoint_chart_coordinate,
                "ledger_relative_defect": result.state_ledger_relative_defect,
                "passed": bool(
                    result.maximum_endpoint_chart_coordinate
                    <= validation["reserved_trust_coordinate_infinity"]
                    and result.state_ledger_relative_defect
                    <= validation["exact_integrated_ledger_relative_defect_max"]
                ),
            }
        )
    endpoint_reconstruction = reconstruct_thermodynamic_macro_state(
        context,
        anchor_charts,
        patch_2_states[-1],
        anchor_macro_state=anchor_macro,
        macro_coordinate_scales=coordinate_scales,
        macro_coordinate_pullbacks=pullbacks,
        derivative_step=construction["pullback_derivative_step"],
        maximum_newton_corrections=8,
        relative_tolerance=validation[
            "maximum_endpoint_macro_roundtrip_relative_defect"
        ],
        maximum_chart_coordinate_infinity=validation[
            "reserved_trust_coordinate_infinity"
        ],
    )
    endpoint_prefilter = truth_source.adaptive_diagnosis._midpoint_hyperbolicity_audit(
        context, endpoint_reconstruction.primitive_charts
    )
    endpoint_operator = generalized_maxwell_cattaneo_radial_operator(
        context, endpoint_reconstruction.primitive_charts, quadrature_order=8
    )
    truth_calls += 1
    endpoint_physical = truth_source._operator_record(endpoint_operator)
    endpoint_checks = truth_source._physical_checks(endpoint_physical, physical_gates)
    endpoint_checks.update(
        {
            "prefilter_eigenvalue": endpoint_prefilter[
                "maximum_eigenvalue_imaginary_ratio"
            ]
            <= 1.0e-10,
            "prefilter_eigenvector": endpoint_prefilter[
                "maximum_eigenvector_imaginary_ratio"
            ]
            <= 1.0e-10,
        }
    )
    print(
        f"truth call {truth_calls}/{budgets['maximum_new_truth_operator_calls']}: "
        f"dynamic_8ms_endpoint {'passed' if all(endpoint_checks.values()) else 'failed'}",
        flush=True,
    )
    endpoint_truth = truth_outputs_from_radial_operator(endpoint_operator)
    endpoint_predicted = patch_2.evaluate(patch_2_states[-1])
    endpoint_output_defects = _output_block_defects(endpoint_predicted, endpoint_truth)
    endpoint_rate_defects = _macro_rate_defects(endpoint_predicted, endpoint_truth)
    cumulative_output = np.sum(np.asarray(integrated_outputs), axis=0)
    q_change = system.normalized_state(patch_2_states[-1]) - system.normalized_state(patch_2_states[0])
    physical_change = np.asarray(coordinate_scales).ravel() * q_change
    cumulative_ledger_defect = _relative(
        physical_change, system.rate_output_matrix @ cumulative_output
    )
    spectral_abscissa = float(np.max(np.real(eigenvalues)))
    all_truth_passed = all(record["passed"] for record in truth_records) and all(
        endpoint_checks.values()
    )
    passed = bool(
        truth_calls == budgets["maximum_new_truth_operator_calls"]
        and all_truth_passed
        and anchor_roundtrip_defect
        <= validation["maximum_endpoint_macro_roundtrip_relative_defect"]
        and float(np.max(pullback_conditions))
        <= construction["maximum_pullback_condition_number"]
        and pullback_closure <= 1.0e-10
        and max(leakage_ratios)
        <= construction["maximum_colored_support_leakage_ratio"]
        and max(jvp_defects)
        <= construction["maximum_independent_JVP_relative_defect"]
        and max(overlap_output_defects.values())
        <= validation["maximum_interpatch_output_relative_defect_per_block"]
        and max(overlap_rate_defects.values())
        <= validation["maximum_interpatch_macro_rate_relative_defect_per_field"]
        and spectral_abscissa
        <= validation["maximum_local_spectral_abscissa_per_second"]
        and all(record["passed"] for record in step_records)
        and cumulative_ledger_defect
        <= validation["exact_integrated_ledger_relative_defect_max"]
        and endpoint_reconstruction.maximum_macro_state_roundtrip_relative_defect
        <= validation["maximum_endpoint_macro_roundtrip_relative_defect"]
        and max(endpoint_output_defects.values())
        <= validation["maximum_endpoint_truth_output_relative_defect_per_block"]
        and max(endpoint_rate_defects.values())
        <= validation["maximum_endpoint_truth_macro_rate_relative_defect_per_field"]
        and conservative_ledger_relative_defect(endpoint_predicted)
        <= validation["exact_integrated_ledger_relative_defect_max"]
    )
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,
        "passed": passed,
        "context_construction_wall_seconds": context_seconds,
        "pullback_construction_wall_seconds": pullback_seconds,
        "patch_2_anchor_roundtrip_relative_defect": anchor_roundtrip_defect,
        "patch_2_pullback_condition_numbers": pullback_conditions.tolist(),
        "patch_2_maximum_pullback_condition_number": float(np.max(pullback_conditions)),
        "patch_2_pullback_inverse_closure_infinity": pullback_closure,
        "new_truth_operator_calls": truth_calls,
        "all_truth_physical_gates_passed": all_truth_passed,
        "truth_records": truth_records,
        "maximum_colored_support_leakage_ratio": max(leakage_ratios),
        "independent_JVP_relative_defects": jvp_defects,
        "maximum_independent_JVP_relative_defect": max(jvp_defects),
        "overlap_patch_1_coordinate_infinity": overlap_patch_1_coordinate,
        "overlap_patch_2_coordinate_infinity": overlap_patch_2_coordinate,
        "overlap_output_block_relative_defects": overlap_output_defects,
        "overlap_macro_rate_relative_defects": overlap_rate_defects,
        "maximum_overlap_output_relative_defect": max(overlap_output_defects.values()),
        "maximum_overlap_macro_rate_relative_defect": max(overlap_rate_defects.values()),
        "patch_2_spectral_abscissa_per_second": spectral_abscissa,
        "patch_2_minimum_spectral_real_part_per_second": float(np.min(np.real(eigenvalues))),
        "patch_2_step_records": step_records,
        "patch_2_cumulative_ledger_relative_defect": cumulative_ledger_defect,
        "endpoint_reconstruction_roundtrip_relative_defect": endpoint_reconstruction.maximum_macro_state_roundtrip_relative_defect,
        "endpoint_reconstruction_chart_coordinate_infinity": float(np.max(np.abs(endpoint_reconstruction.chart_coordinates))),
        "endpoint_reconstruction_newton_corrections": endpoint_reconstruction.newton_corrections,
        "endpoint_truth_physical": endpoint_physical,
        "endpoint_truth_physical_checks": endpoint_checks,
        "endpoint_output_block_relative_defects": endpoint_output_defects,
        "endpoint_macro_rate_relative_defects": endpoint_rate_defects,
        "endpoint_maximum_output_relative_defect": max(endpoint_output_defects.values()),
        "endpoint_maximum_macro_rate_relative_defect": max(endpoint_rate_defects.values()),
        "accepted_new_macrosteps": validation["patch_2_macrosteps"] if passed else 0,
        "accepted_absolute_horizon_seconds": validation["absolute_elapsed_endpoint_seconds"] if passed else 4.0e-3,
        "new_global_roots": 0,
        "complete_cycle_execution_authorized": False,
    }
    arrays = {
        "patch_2_anchor_macro_state": anchor_macro,
        "patch_2_anchor_primitive_charts": anchor_charts,
        "patch_2_macro_coordinate_scales": coordinate_scales,
        "patch_2_macro_chart_tangents": tangents,
        "patch_2_macro_coordinate_pullbacks": pullbacks,
        "patch_2_base_normalized_output": base_normalized_output,
        "patch_2_normalized_output_chart_jacobian": jacobian,
        "patch_2_output_component_scales": output_scales,
        "patch_2_JVP_directions": directions,
        "patch_2_JVP_relative_defects": np.asarray(jvp_defects),
        "patch_2_macro_states": np.asarray(patch_2_states),
        "patch_2_integrated_packed_outputs": np.asarray(integrated_outputs),
        "patch_2_spectral_values_per_second": eigenvalues,
        "combined_macro_states_0_to_8ms": np.concatenate((patch_1_states, np.asarray(patch_2_states[1:])), axis=0),
        "endpoint_8ms_primitive_charts": endpoint_reconstruction.primitive_charts,
        "endpoint_8ms_chart_coordinates": endpoint_reconstruction.chart_coordinates,
        "endpoint_8ms_truth_packed_outputs": pack_macro_outputs(endpoint_truth),
        "endpoint_8ms_predicted_packed_outputs": pack_macro_outputs(endpoint_predicted),
    }
    return metrics, arrays


def _update_catalog(summary: dict) -> None:
    utils = _utils()
    with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file(): rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": utils._sha256(path), "scientific_status": "SUPPORTED" if summary["passed"] else "REJECTED"})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    catalog = utils._read_json(CANONICAL_SUMMARY); catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": summary["classification"], "passed": summary["passed"]}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": utils._git("rev-parse", "HEAD"), "latest_work_package": WORK_PACKAGE}); utils._write_json(CANONICAL_SUMMARY, catalog)


def _canonicalize(metrics: dict, arrays: dict[str, np.ndarray]) -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists(): raise RuntimeError("second pathwise patch result already exists")
    validated = _validate_parent(require_clean=True); utils = _utils(); CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "pathwise_patch_metrics.json", metrics); np.savez_compressed(CANONICAL_DIRECTORY / "pathwise_patch_arrays.npz", **arrays)
    summary = {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "classification": metrics["classification"], "passed": bool(metrics["passed"]), "bounded_macro_integrator_preserved": True, "two_patch_path_certified": bool(metrics["passed"]), "accepted_absolute_horizon_seconds": metrics["accepted_absolute_horizon_seconds"], "multi_patch_growth_and_fast_slaving_manifest_authorized": bool(metrics["passed"]), "complete_cycle_execution_authorized": False, "authorized_next": AUTHORIZED_NEXT_ON_PASS if metrics["passed"] else None}
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utils._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"parent_artifact": parent.ARTIFACT, "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256, "parent_hashes": validated["hashes"], "integrator_arrays_sha256": utils._sha256(INTEGRATOR_ARRAYS)})
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(("# Entropy-complete second pathwise macro-patch execution", "", f"Classification: `{metrics['classification']}`.", "", f"The second 38-call patch had maximum JVP defect `{metrics['maximum_independent_JVP_relative_defect']:.6e}` and spectral abscissa `{metrics['patch_2_spectral_abscissa_per_second']:.6e}` /s.", "", f"Overlap output/rate defects were `{metrics['maximum_overlap_output_relative_defect']:.6e}` / `{metrics['maximum_overlap_macro_rate_relative_defect']:.6e}`. The new 8 ms truth output/rate defects were `{metrics['endpoint_maximum_output_relative_defect']:.6e}` / `{metrics['endpoint_maximum_macro_rate_relative_defect']:.6e}`.", "", "No extrapolation or complete-cycle execution is authorized.", "", f"Authorized next: `{summary['authorized_next']}`.", "")), encoding="utf-8")
    sources = (THIS_RUNNER, THIS_TEST, REPORT_RELATIVE)
    utils._write_json(CANONICAL_DIRECTORY / "provenance.json", {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "implementation_commit": utils._git("rev-parse", "HEAD"), "implementation_tree": utils._git("rev-parse", "HEAD^{tree}"), "source_hashes": {path: utils._sha256(ROOT / path) for path in sources}, "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "thread_environment": {name: os.environ.get(name, "") for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()); (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utils._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8"); _update_catalog(summary); return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run", action="store_true"); args = parser.parse_args()
    if not args.run: parser.error("choose --run")
    metrics, arrays = _execute(); print(json.dumps(metrics, indent=2, sort_keys=True), flush=True); summary = _canonicalize(metrics, arrays); return 0 if summary["passed"] else 2


if __name__ == "__main__": raise SystemExit(main())
