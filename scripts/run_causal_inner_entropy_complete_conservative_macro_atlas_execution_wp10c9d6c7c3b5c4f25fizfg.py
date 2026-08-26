#!/usr/bin/env python3
"""Build and blindly validate the frozen conservative 16-cell macro atlas."""

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

import run_causal_inner_entropy_complete_conservative_macro_atlas_manifest_wp10c9d6c7c3b5c4f25fizff as parent  # noqa: E402
import run_causal_inner_entropy_complete_hydrostatic_invariant_reconstruction_implementation_wp10c9d6c7c3b5c4f25fizfa as truth_source  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_macro_atlas import (  # noqa: E402
    ConservativeAffineMacroAtlas,
    ConservativeMacroOutputs,
    OUTPUT_SIZE,
    conservative_ledger_relative_defect,
    macro_output_component_scales,
    pack_macro_outputs,
    prolong_entropy_complete_macro,
    restrict_entropy_complete_macro,
    restricted_truth_outputs,
    truth_outputs_from_radial_operator,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_radial import (  # noqa: E402
    generalized_maxwell_cattaneo_radial_operator,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fizfg_"
    "entropy_complete_conservative_macro_atlas_execution"
)
PASS_CLASSIFICATION = "entropy_complete_conservative_16_cell_macro_atlas_blindly_validated"
FAIL_CLASSIFICATION = "entropy_complete_conservative_16_cell_macro_atlas_rejected"
AUTHORIZED_NEXT_ON_PASS = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizfh_"
    "entropy_complete_structure_preserving_macro_integrator_manifest"
)
ARTIFACT = (
    "causal_inner_entropy_complete_conservative_macro_atlas_execution_"
    "wp10c9d6c7c3b5c4f25fizfg"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_CONSERVATIVE_"
    "MACRO_ATLAS_EXECUTION_WP10C9D6C7C3B5C4F25FIZFG_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_entropy_complete_conservative_macro_atlas_"
    "execution_wp10c9d6c7c3b5c4f25fizfg.py"
)
THIS_TEST = (
    "tests/test_causal_inner_entropy_complete_conservative_macro_atlas_"
    "execution_wp10c9d6c7c3b5c4f25fizfg.py"
)
SOURCE = (
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_generalized_maxwell_cattaneo_macro_atlas.py"
)
SOURCE_TEST = "tests/test_causal_inner_generalized_maxwell_cattaneo_macro_atlas.py"
SOURCE_SHA256 = "a8869394a31ff8056f2b448e10bb42cd043bebde4ab16152c648b25a9f84d1b8"
SOURCE_TEST_SHA256 = "0287e0b4f03be3c241e820f9b7c76ba75c14fc40907421f9b28f4ce000eda4b3"
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "388abf54b4b428d79ba3dfb0a503d278d57fb1028559fca14eded7ce65fe431b"
)
TRUTH_ARRAYS = truth_source.CANONICAL_DIRECTORY / "implementation_arrays.npz"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
RANDOM_SEED = 20_260_831


def _utils():
    return parent._utils()


def _validate_parent(*, require_clean: bool) -> dict:
    utils = _utils()
    if (
        utils._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt")
        != PARENT_CHECKSUM_MANIFEST_SHA256
    ):
        raise RuntimeError("conservative macro-atlas manifest checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    contract = utils._read_json(parent.CANONICAL_DIRECTORY / "macro_atlas_contract.json")
    if (
        summary["classification"] != parent.CLASSIFICATION
        or not summary["passed"]
        or not summary["definitions_only"]
        or not summary["conservative_16_cell_macro_atlas_selected"]
        or not summary["offline_truth_sampling_authorized"]
        or summary["state_propagation_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["authorized_next"] != WORK_PACKAGE
        or contract["atlas_construction"]["maximum_new_truth_operator_calls"] != 38
    ):
        raise RuntimeError("conservative macro-atlas authorization changed")
    if utils._sha256(ROOT / SOURCE) != SOURCE_SHA256:
        raise RuntimeError("conservative macro-atlas source changed")
    if utils._sha256(ROOT / SOURCE_TEST) != SOURCE_TEST_SHA256:
        raise RuntimeError("conservative macro-atlas source test changed")
    for relative, expected in utils._read_json(
        parent.CANONICAL_DIRECTORY / "provenance.json"
    )["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"conservative macro-atlas manifest source changed: {relative}")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("macro-atlas execution requires a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "contract": contract}


def _sample(archive, label: str, sample: str) -> tuple[np.ndarray, np.ndarray, ConservativeMacroOutputs]:
    charts = np.asarray(
        archive[f"{label}_base_charts7"]
        if sample == "base"
        else archive[f"{label}_perturbed_charts7"]
    )
    targets = np.asarray(
        archive[f"{label}_slow_targets_MJE"]
        if sample == "base"
        else archive[f"{label}_perturbed_targets_MJE"]
    )
    outputs = restricted_truth_outputs(
        slow_targets_MJE=targets,
        primitive_charts=charts,
        weighted_shared_MJE_fluxes_over_c=np.asarray(
            archive[f"{label}_{sample}_weighted_shared_MJE_fluxes_over_c"]
        ),
        weighted_MJE_sources_per_ct=np.asarray(
            archive[f"{label}_{sample}_weighted_MJE_sources_per_ct"]
        ),
        radial_stress_rates_per_second=np.asarray(
            archive[f"{label}_{sample}_radial_stress_rates_per_second"]
        ),
    )
    return charts, targets, outputs


def _support_rows(input_cell: int) -> np.ndarray:
    cell = int(input_cell)
    if cell < 0 or cell >= 16:
        raise ValueError("macro input cell is out of range")
    rows = []
    for face in (cell, cell + 1):
        rows.extend(range(3 * face, 3 * face + 3))
    rows.extend(range(51 + 2 * cell, 51 + 2 * cell + 2))
    for output_cell in range(max(0, cell - 1), min(15, cell + 1) + 1):
        rows.extend(range(83 + 2 * output_cell, 83 + 2 * output_cell + 2))
    return np.asarray(sorted(set(rows)), dtype=int)


def _relative(value, reference) -> float:
    value = np.asarray(value, dtype=float); reference = np.asarray(reference, dtype=float)
    scale = max(float(np.max(np.abs(value))), float(np.max(np.abs(reference))), np.finfo(float).tiny)
    return float(np.max(np.abs(value - reference)) / scale)


def _output_block_defects(predicted: ConservativeMacroOutputs, truth: ConservativeMacroOutputs) -> dict:
    result = {}
    for index, name in enumerate(("M_flux", "J_flux", "E_flux")):
        result[name] = _relative(predicted.MJE_face_fluxes_over_c[:, index], truth.MJE_face_fluxes_over_c[:, index])
    for index, name in enumerate(("J_source", "E_source"), start=1):
        result[name] = _relative(predicted.MJE_cell_sources_per_ct[:, index], truth.MJE_cell_sources_per_ct[:, index])
    for index, name in enumerate(("beta_r_rate", "chi_rate")):
        result[name] = _relative(predicted.auxiliary_rates_per_second[:, index], truth.auxiliary_rates_per_second[:, index])
    return result


def _macro_rate_defects(predicted: ConservativeMacroOutputs, truth: ConservativeMacroOutputs) -> dict:
    return {
        name: _relative(predicted.macro_rates_per_second[:, index], truth.macro_rates_per_second[:, index])
        for index, name in enumerate(("M_rate", "J_rate", "E_rate", "beta_r_rate", "chi_rate"))
    }


def _execute() -> tuple[dict, dict[str, np.ndarray]]:
    validated = _validate_parent(require_clean=False)
    contract = validated["contract"]
    construction = contract["atlas_construction"]
    conservative = contract["conservative_outputs"]
    cost = contract["online_cost"]
    physical_gates = truth_source.fixed_q_implementation.parent._contract()["binding_physical_gates"]
    context_start = time.perf_counter()
    context, _profile, _charts = truth_source.fixed_q_implementation._primary_setup()
    context_seconds = time.perf_counter() - context_start
    with np.load(TRUTH_ARRAYS) as archive:
        anchor_charts, anchor_targets, base_outputs = _sample(
            archive, "primary_20ms", "base"
        )
        anchor_macro = restrict_entropy_complete_macro(anchor_targets, anchor_charts)
        field_max = np.max(np.abs(anchor_macro), axis=0)
        coordinate_scales = np.maximum(
            np.abs(anchor_macro),
            construction["coordinate_scale_floor_fraction"] * field_max[None, :],
        )
        output_scales = macro_output_component_scales(base_outputs)
        base_normalized_output = pack_macro_outputs(base_outputs) / output_scales
        base_lift = prolong_entropy_complete_macro(
            context,
            anchor_targets,
            anchor_charts,
            anchor_macro,
            constraint_tolerance=construction["maximum_truth_reconstruction_relative_defect"],
        )
        base_chart_defect = float(
            np.max(
                np.abs(
                    (base_lift.primitive_charts - anchor_charts)
                    / truth_source.CHART_SCALES7
                )
            )
        )
        truth_calls = 0
        truth_records = []
        maximum_roundtrip_defect = 0.0

        def truth_field(coordinate: np.ndarray, *, label: str) -> np.ndarray:
            nonlocal truth_calls, maximum_roundtrip_defect
            values = np.asarray(coordinate, dtype=float)
            if values.shape != (16, 5) or np.any(~np.isfinite(values)):
                raise ValueError("truth-field coordinate is invalid")
            macro = anchor_macro + coordinate_scales * values
            reconstruction = prolong_entropy_complete_macro(
                context,
                anchor_targets,
                anchor_charts,
                macro,
                constraint_tolerance=construction["maximum_truth_reconstruction_relative_defect"],
            )
            restricted = restrict_entropy_complete_macro(
                reconstruction.slow_targets, reconstruction.primitive_charts
            )
            roundtrip = _relative(restricted, macro)
            maximum_roundtrip_defect = max(maximum_roundtrip_defect, roundtrip)
            prefilter = truth_source.adaptive_diagnosis._midpoint_hyperbolicity_audit(
                context, reconstruction.primitive_charts
            )
            operator = generalized_maxwell_cattaneo_radial_operator(
                context, reconstruction.primitive_charts, quadrature_order=8
            )
            truth_calls += 1
            record = truth_source._operator_record(operator)
            checks = truth_source._physical_checks(record, physical_gates)
            checks.update(
                {
                    "prefilter_eigenvalue": prefilter["maximum_eigenvalue_imaginary_ratio"] <= 1.0e-10,
                    "prefilter_eigenvector": prefilter["maximum_eigenvector_imaginary_ratio"] <= 1.0e-10,
                    "reconstruction": reconstruction.maximum_constraint_relative_defect <= construction["maximum_truth_reconstruction_relative_defect"],
                    "restriction_roundtrip": roundtrip <= conservative["restriction_roundtrip_relative_defect_max"],
                }
            )
            truth_records.append(
                {
                    "label": label,
                    "call": truth_calls,
                    "maximum_coordinate": float(np.max(np.abs(values))),
                    "maximum_constraint_relative_defect": reconstruction.maximum_constraint_relative_defect,
                    "restriction_roundtrip_relative_defect": roundtrip,
                    "physical": record,
                    "checks": checks,
                    "passed": all(checks.values()),
                }
            )
            print(
                f"truth call {truth_calls}/{construction['maximum_new_truth_operator_calls']}: {label} {'passed' if all(checks.values()) else 'failed'}",
                flush=True,
            )
            return pack_macro_outputs(truth_outputs_from_radial_operator(operator)) / output_scales

        jacobian = np.zeros((OUTPUT_SIZE, 80), dtype=float)
        colored_leakages = []
        colored_step = construction["central_colored_coordinate_step"]
        for field in range(5):
            for color in range(3):
                direction = np.zeros((16, 5), dtype=float)
                active = list(range(color, 16, 3))
                direction[active, field] = 1.0
                plus = truth_field(
                    colored_step * direction, label=f"colored_f{field}_c{color}_plus"
                )
                minus = truth_field(
                    -colored_step * direction, label=f"colored_f{field}_c{color}_minus"
                )
                derivative = (plus - minus) / (2.0 * colored_step)
                assigned = np.zeros(OUTPUT_SIZE, dtype=bool)
                for cell in active:
                    rows = _support_rows(cell)
                    jacobian[rows, 5 * cell + field] = derivative[rows]
                    assigned[rows] = True
                colored_leakages.append(float(np.max(np.abs(derivative[~assigned]))) if np.any(~assigned) else 0.0)

        generator = np.random.default_rng(RANDOM_SEED)
        jvp_directions = generator.normal(size=(construction["independent_JVP_directions"], 16, 5))
        jvp_directions /= np.max(np.abs(jvp_directions), axis=(1, 2))[:, None, None]
        jvp_defects = []
        jvp_step = construction["independent_JVP_coordinate_step"]
        for index, direction in enumerate(jvp_directions):
            plus = truth_field(jvp_step * direction, label=f"JVP_{index}_plus")
            minus = truth_field(-jvp_step * direction, label=f"JVP_{index}_minus")
            finite = (plus - minus) / (2.0 * jvp_step)
            analytic = jacobian @ direction.ravel()
            jvp_defects.append(_relative(finite, analytic))

        atlas = ConservativeAffineMacroAtlas(
            anchor_macro_state=anchor_macro,
            macro_coordinate_scales=coordinate_scales,
            base_normalized_output=base_normalized_output,
            normalized_output_jacobian=jacobian,
            output_component_scales=output_scales,
            trust_coordinate_infinity=construction["maximum_blind_coordinate_infinity"],
        )

        validation_records = {}
        validation_arrays = {}
        for label, sample, gate in (
            ("primary_20ms", "perturbed", construction["maximum_saved_near_witness_output_relative_defect"]),
            ("heldout_16ms", "base", construction["maximum_blind_output_relative_defect_per_block"]),
            ("heldout_16ms", "perturbed", construction["maximum_blind_output_relative_defect_per_block"]),
        ):
            charts, targets, truth = _sample(archive, label, sample)
            state = restrict_entropy_complete_macro(targets, charts)
            coordinate = (state - anchor_macro) / coordinate_scales
            predicted = atlas.evaluate(state)
            block_defects = _output_block_defects(predicted, truth)
            rate_defects = _macro_rate_defects(predicted, truth)
            ledger_defect = conservative_ledger_relative_defect(predicted)
            coordinate_infinity = float(np.max(np.abs(coordinate)))
            is_blind = label == "heldout_16ms"
            passed = (
                max(block_defects.values()) <= gate
                and (
                    not is_blind
                    or max(rate_defects.values())
                    <= construction["maximum_blind_macro_rate_relative_defect_per_field"]
                )
                and coordinate_infinity <= construction["maximum_blind_coordinate_infinity"]
                and ledger_defect <= conservative["conservative_ledger_relative_defect_max"]
            )
            key = f"{label}_{sample}"
            validation_records[key] = {
                "strict_blind": is_blind,
                "passed": passed,
                "coordinate_infinity": coordinate_infinity,
                "output_block_relative_defects": block_defects,
                "macro_rate_relative_defects": rate_defects,
                "conservative_ledger_relative_defect": ledger_defect,
            }
            validation_arrays.update(
                {
                    f"{key}_macro_state": state,
                    f"{key}_truth_outputs": pack_macro_outputs(truth),
                    f"{key}_predicted_outputs": pack_macro_outputs(predicted),
                    f"{key}_truth_macro_rates_per_second": truth.macro_rates_per_second,
                    f"{key}_predicted_macro_rates_per_second": predicted.macro_rates_per_second,
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
        benchmark_checksum += float(output.macro_rates_per_second[index % 16, index % 5])
    benchmark_seconds = time.perf_counter() - benchmark_start
    all_truth_passed = all(record["passed"] for record in truth_records)
    all_validation_passed = all(record["passed"] for record in validation_records.values())
    passed = bool(
        truth_calls == construction["maximum_new_truth_operator_calls"]
        and all_truth_passed
        and max(jvp_defects) <= construction["maximum_colored_JVP_relative_defect"]
        and maximum_roundtrip_defect <= conservative["restriction_roundtrip_relative_defect_max"]
        and all_validation_passed
        and benchmark_seconds <= cost["maximum_benchmark_wall_seconds"]
    )
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,
        "passed": passed,
        "context_construction_wall_seconds": context_seconds,
        "base_anchor_chart_reproduction_scaled_infinity": base_chart_defect,
        "new_truth_operator_calls": truth_calls,
        "all_truth_physical_gates_passed": all_truth_passed,
        "truth_records": truth_records,
        "maximum_restriction_roundtrip_relative_defect": maximum_roundtrip_defect,
        "maximum_colored_support_leakage": max(colored_leakages),
        "independent_JVP_relative_defects": jvp_defects,
        "maximum_independent_JVP_relative_defect": max(jvp_defects),
        "validation": validation_records,
        "all_saved_and_blind_validations_passed": all_validation_passed,
        "online_benchmark_evaluations": cost["benchmark_evaluations"],
        "online_benchmark_wall_seconds": benchmark_seconds,
        "online_average_wall_seconds_per_evaluation": benchmark_seconds / cost["benchmark_evaluations"],
        "online_benchmark_checksum": benchmark_checksum,
        "new_global_roots": 0,
        "propagated_states": 0,
    }
    arrays = {
        "anchor_macro_state": anchor_macro,
        "macro_coordinate_scales": coordinate_scales,
        "base_normalized_output": base_normalized_output,
        "normalized_output_jacobian": jacobian,
        "output_component_scales": output_scales,
        "JVP_directions": jvp_directions,
        "JVP_relative_defects": np.asarray(jvp_defects),
        **validation_arrays,
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
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists(): raise RuntimeError("macro-atlas execution already exists")
    validated = _validate_parent(require_clean=True); utils = _utils(); CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "macro_atlas_metrics.json", metrics); np.savez_compressed(CANONICAL_DIRECTORY / "macro_atlas_arrays.npz", **arrays)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": bool(metrics["passed"]),
        "hydrostatic_implicit_inverse_tangent_preserved": True,
        "conservative_16_cell_macro_atlas_certified": bool(metrics["passed"]),
        "heldout_16ms_profiles_passed": bool(metrics["passed"]),
        "online_truth_calls_per_macrostep": 0,
        "new_global_roots": 0,
        "propagated_states": 0,
        "structure_preserving_macro_integrator_manifest_authorized": bool(metrics["passed"]),
        "macro_integrator_execution_authorized": False,
        "complete_cycle_execution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT_ON_PASS if metrics["passed"] else None,
    }
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utils._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"parent_artifact": parent.ARTIFACT, "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256, "parent_hashes": validated["hashes"], "source_sha256": SOURCE_SHA256, "source_test_sha256": SOURCE_TEST_SHA256, "truth_arrays_sha256": utils._sha256(TRUTH_ARRAYS)})
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    blind = metrics.get("validation", {})
    heldout_base = blind.get("heldout_16ms_base", {}).get("passed", False)
    heldout_perturbed = blind.get("heldout_16ms_perturbed", {}).get("passed", False)
    REPORT_PATH.write_text("\n".join(("# Entropy-complete conservative macro-atlas execution", "", f"Classification: `{metrics['classification']}`.", "", f"Offline truth calls: `{metrics.get('new_truth_operator_calls', 0)}`; maximum independent colored-JVP defect: `{metrics.get('maximum_independent_JVP_relative_defect', float('nan')):.6e}`.", "", f"Held-out base passed: `{heldout_base}`; held-out perturbed passed: `{heldout_perturbed}`.", "", f"The `{metrics.get('online_benchmark_evaluations', 0)}`-evaluation online benchmark took `{metrics.get('online_benchmark_wall_seconds', 0.0):.6f}` s with zero online truth calls, roots, or propagation.", "", f"Failure: `{metrics.get('failure', None)}`.", "", f"Authorized next: `{summary['authorized_next']}`.", "")), encoding="utf-8")
    sources = (THIS_RUNNER, THIS_TEST, SOURCE, SOURCE_TEST, REPORT_RELATIVE)
    utils._write_json(CANONICAL_DIRECTORY / "provenance.json", {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "implementation_commit": utils._git("rev-parse", "HEAD"), "implementation_tree": utils._git("rev-parse", "HEAD^{tree}"), "source_hashes": {path: utils._sha256(ROOT / path) for path in sources}, "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "thread_environment": {name: os.environ.get(name, "") for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()); (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utils._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8"); _update_catalog(summary); return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run", action="store_true"); args = parser.parse_args()
    if not args.run: parser.error("choose --run")
    try:
        metrics, arrays = _execute()
    except np.linalg.LinAlgError as error:
        metrics = {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "classification": FAIL_CLASSIFICATION,
            "passed": False,
            "failure": {
                "stage": "first_colored_raw_M_coordinate_plus_lift",
                "type": type(error).__name__,
                "message": str(error),
                "interpretation": "raw_independent_MJE_atlas_coordinate_left_the_locally_invertible_thermodynamic_cone",
            },
            "new_truth_operator_calls": 0,
            "attempted_macro_lifts": 1,
            "new_global_roots": 0,
            "propagated_states": 0,
        }
        arrays = {}
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    summary = _canonicalize(metrics, arrays)
    return 0 if summary["passed"] else 2


if __name__ == "__main__": raise SystemExit(main())
