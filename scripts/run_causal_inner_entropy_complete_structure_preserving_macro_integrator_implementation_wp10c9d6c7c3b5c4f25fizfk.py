#!/usr/bin/env python3
"""Execute the bounded exact-affine macro-integrator certificate."""

from __future__ import annotations

import argparse
import csv
import io
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

import run_causal_inner_entropy_complete_structure_preserving_macro_integrator_manifest_wp10c9d6c7c3b5c4f25fizfj as parent  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_macro_atlas import (  # noqa: E402
    conservative_ledger_relative_defect,
    pack_macro_outputs,
    restrict_entropy_complete_macro,
    truth_outputs_from_radial_operator,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_macro_integrator import (  # noqa: E402
    ExactAffineMacroSystem,
    ExactAffineMacroTransition,
    reconstruct_thermodynamic_macro_state,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_radial import (  # noqa: E402
    generalized_maxwell_cattaneo_radial_operator,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_thermodynamic_macro_atlas import (  # noqa: E402
    ThermodynamicAffineMacroAtlas,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fizfk_"
    "entropy_complete_structure_preserving_macro_integrator_implementation"
)
PASS_CLASSIFICATION = (
    "entropy_complete_exact_affine_macro_integrator_bounded_pilot_certified"
)
SCIENTIFIC_FAIL_CLASSIFICATION = (
    "entropy_complete_exact_affine_macro_integrator_bounded_pilot_rejected"
)
COST_FAIL_CLASSIFICATION = (
    "entropy_complete_exact_affine_macro_integrator_scientific_pass_cost_failed"
)
AUTHORIZED_NEXT_ON_PASS = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizfl_"
    "entropy_complete_pathwise_macro_atlas_expansion_manifest"
)
ARTIFACT = (
    "causal_inner_entropy_complete_structure_preserving_macro_integrator_"
    "implementation_wp10c9d6c7c3b5c4f25fizfk"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_STRUCTURE_"
    "PRESERVING_MACRO_INTEGRATOR_IMPLEMENTATION_WP10C9D6C7C3B5C4F25FIZFK_"
    "2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_entropy_complete_structure_preserving_macro_"
    "integrator_implementation_wp10c9d6c7c3b5c4f25fizfk.py"
)
THIS_TEST = (
    "tests/test_causal_inner_entropy_complete_structure_preserving_macro_"
    "integrator_implementation_wp10c9d6c7c3b5c4f25fizfk.py"
)
SOURCE = (
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_generalized_maxwell_cattaneo_macro_integrator.py"
)
SOURCE_TEST = (
    "tests/test_causal_inner_generalized_maxwell_cattaneo_macro_integrator.py"
)
SOURCE_SHA256 = "9fa619ca694ad8c92e1a36b79fc49c7dee045429a7110b3a029be041e7127e12"
SOURCE_TEST_SHA256 = "d0867f74eb22ffd16bc61a959a225a9937dba563a5743166c5b7c4d9ccf32056"
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "d2f1aa15491edaae201867a55a07580099ce23845de2ff82f7fbf168b2800c77"
)
ATLAS_ARRAYS = parent.parent.CANONICAL_DIRECTORY / "macro_atlas_arrays.npz"
TRUTH_ARRAYS = parent.parent.TRUTH_ARRAYS
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _utils():
    return parent._utils()


def _validate_parent(*, require_clean: bool) -> dict:
    utils = _utils()
    if (
        utils._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt")
        != PARENT_CHECKSUM_MANIFEST_SHA256
    ):
        raise RuntimeError("macro-integrator manifest checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    contract = utils._read_json(
        parent.CANONICAL_DIRECTORY / "macro_integrator_contract.json"
    )
    if (
        summary["classification"] != parent.CLASSIFICATION
        or not summary["passed"]
        or not summary["definitions_only"]
        or not summary["exact_affine_macro_integrator_selected"]
        or not summary["bounded_macro_propagation_authorized"]
        or summary["complete_cycle_execution_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["authorized_next"] != WORK_PACKAGE
        or contract["bounded_pilot"]["maximum_new_truth_operator_calls"] != 1
    ):
        raise RuntimeError("bounded macro-integrator authorization changed")
    if utils._sha256(ROOT / SOURCE) != SOURCE_SHA256:
        raise RuntimeError("macro-integrator source changed")
    if utils._sha256(ROOT / SOURCE_TEST) != SOURCE_TEST_SHA256:
        raise RuntimeError("macro-integrator source test changed")
    for relative, expected in utils._read_json(
        parent.CANONICAL_DIRECTORY / "provenance.json"
    )["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"macro-integrator manifest source changed: {relative}")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("macro-integrator execution requires a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "contract": contract}


def _atlas(archive) -> ThermodynamicAffineMacroAtlas:
    return ThermodynamicAffineMacroAtlas(
        anchor_macro_state=np.asarray(archive["anchor_macro_state"]),
        macro_coordinate_scales=np.asarray(archive["macro_coordinate_scales"]),
        base_normalized_output=np.asarray(archive["base_normalized_output"]),
        normalized_output_jacobian=np.asarray(
            archive["normalized_output_chart_jacobian"]
        ),
        output_component_scales=np.asarray(archive["output_component_scales"]),
        trust_coordinate_infinity=1.5e-1,
        macro_coordinate_pullback=np.asarray(
            archive["macro_coordinate_pullbacks"]
        ),
    )


def _relative(value, reference) -> float:
    return parent.parent._relative(value, reference)


def _execute() -> tuple[dict, dict[str, np.ndarray]]:
    validated = _validate_parent(require_clean=False)
    contract = validated["contract"]
    pilot = contract["bounded_pilot"]
    reconstruction_contract = contract["offline_endpoint_reconstruction"]
    gates = contract["binding_gates"]
    cost = contract["online_cost"]
    with np.load(ATLAS_ARRAYS) as archive:
        atlas = _atlas(archive)
        atlas_arrays = {name: np.asarray(archive[name]) for name in archive.files}
    system = ExactAffineMacroSystem.from_atlas(atlas)
    eigenvalues = np.linalg.eigvals(system.normalized_rate_matrix)
    spectral_abscissa = float(np.max(np.real(eigenvalues)))
    transition_start = time.perf_counter()
    transition = ExactAffineMacroTransition.build(
        system,
        pilot["fixed_macrostep_seconds"],
        trust_coordinate_infinity=pilot[
            "pilot_reserved_trust_coordinate_infinity"
        ],
    )
    full_transition = ExactAffineMacroTransition.build(
        system,
        pilot["horizon_seconds"],
        trust_coordinate_infinity=pilot[
            "pilot_reserved_trust_coordinate_infinity"
        ],
    )
    transition_construction_seconds = time.perf_counter() - transition_start
    states = [np.array(atlas.anchor_macro_state, copy=True)]
    integrated_outputs = []
    step_records = []
    checkpoint_bytes = None
    for index in range(pilot["accepted_macrosteps"]):
        result = transition.step(states[-1])
        states.append(result.macro_state)
        integrated_outputs.append(result.integrated_packed_output)
        step_records.append(
            {
                "step": index + 1,
                "elapsed_seconds": (index + 1) * pilot["fixed_macrostep_seconds"],
                "maximum_chart_coordinate": (
                    result.maximum_endpoint_chart_coordinate
                ),
                "exact_integrated_ledger_relative_defect": (
                    result.state_ledger_relative_defect
                ),
                "minimum_MJE": float(np.min(result.macro_state[:, :3])),
                "maximum_absolute_beta_r": float(
                    np.max(np.abs(result.macro_state[:, 3]))
                ),
                "passed": bool(
                    result.maximum_endpoint_chart_coordinate
                    <= pilot["pilot_reserved_trust_coordinate_infinity"]
                    and result.state_ledger_relative_defect
                    <= gates["maximum_exact_integrated_ledger_relative_defect"]
                    and np.all(result.macro_state[:, :3] > 0.0)
                    and np.all(np.abs(result.macro_state[:, 3]) < 1.0)
                ),
            }
        )
        if index + 1 == pilot["checkpoint_after_step"]:
            buffer = io.BytesIO()
            np.savez_compressed(
                buffer,
                macro_state=result.macro_state,
                completed_steps=np.asarray(index + 1, dtype=np.int64),
                timestep_seconds=np.asarray(
                    pilot["fixed_macrostep_seconds"], dtype=float
                ),
                state_transition=transition.state_transition,
                normalized_output_integral=transition.normalized_output_integral,
            )
            checkpoint_bytes = buffer.getvalue()
    if checkpoint_bytes is None:
        raise RuntimeError("macro-integrator checkpoint was not created")
    with np.load(io.BytesIO(checkpoint_bytes), allow_pickle=False) as restart:
        restart_state = np.asarray(restart["macro_state"])
        checkpoint_roundtrip_bitwise = np.array_equal(
            restart_state, states[pilot["checkpoint_after_step"]]
        )
    replay_states = [restart_state]
    for _ in range(pilot["suffix_steps_to_replay"]):
        replay_states.append(transition.step(replay_states[-1]).macro_state)
    suffix_replay_bitwise = np.array_equal(replay_states[-1], states[-1])
    one_step = full_transition.step(states[0])
    q_many = system.normalized_state(states[-1])
    q_one = system.normalized_state(one_step.macro_state)
    semigroup_defect = _relative(q_many, q_one)
    accumulated_output = np.sum(np.asarray(integrated_outputs), axis=0)
    normalized_change = q_many - system.normalized_state(states[0])
    physical_change = np.asarray(atlas.macro_coordinate_scales).ravel() * (
        normalized_change
    )
    ledger_change = system.rate_output_matrix @ accumulated_output
    cumulative_ledger_defect = _relative(physical_change, ledger_change)

    context_start = time.perf_counter()
    truth_source = parent.parent.truth_source
    context, _profile, _charts = truth_source.fixed_q_implementation._primary_setup()
    context_seconds = time.perf_counter() - context_start
    with np.load(TRUTH_ARRAYS) as truth_archive:
        anchor_charts = np.asarray(truth_archive["primary_20ms_base_charts7"])
    reconstruction_start = time.perf_counter()
    reconstruction = reconstruct_thermodynamic_macro_state(
        context,
        anchor_charts,
        states[-1],
        anchor_macro_state=atlas.anchor_macro_state,
        macro_coordinate_scales=atlas.macro_coordinate_scales,
        macro_coordinate_pullbacks=atlas.macro_coordinate_pullback,
        derivative_step=reconstruction_contract["finite_difference_step"],
        maximum_newton_corrections=reconstruction_contract[
            "maximum_newton_corrections"
        ],
        relative_tolerance=reconstruction_contract[
            "maximum_macro_state_roundtrip_relative_defect"
        ],
        maximum_chart_coordinate_infinity=reconstruction_contract[
            "maximum_chart_coordinate_infinity"
        ],
    )
    reconstruction_seconds = time.perf_counter() - reconstruction_start
    prefilter = truth_source.adaptive_diagnosis._midpoint_hyperbolicity_audit(
        context, reconstruction.primitive_charts
    )
    truth_start = time.perf_counter()
    operator = generalized_maxwell_cattaneo_radial_operator(
        context, reconstruction.primitive_charts, quadrature_order=8
    )
    truth_seconds = time.perf_counter() - truth_start
    physical = truth_source._operator_record(operator)
    physical_checks = truth_source._physical_checks(
        physical,
        truth_source.fixed_q_implementation.parent._contract()[
            "binding_physical_gates"
        ],
    )
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
        }
    )
    truth_outputs = truth_outputs_from_radial_operator(operator)
    predicted_outputs = atlas.evaluate(states[-1])
    output_defects = parent.parent._output_block_defects(
        predicted_outputs, truth_outputs
    )
    rate_defects = parent.parent._macro_rate_defects(
        predicted_outputs, truth_outputs
    )
    predicted_ledger_defect = conservative_ledger_relative_defect(
        predicted_outputs
    )

    benchmark_start = time.perf_counter()
    benchmark_checksum = 0.0
    for _ in range(cost["benchmark_macrosteps"]):
        benchmark_result = transition.step(states[0])
        benchmark_checksum += benchmark_result.state_ledger_relative_defect
    benchmark_seconds = time.perf_counter() - benchmark_start
    scientific_passed = bool(
        spectral_abscissa <= gates["maximum_spectral_abscissa_per_second"]
        and all(record["passed"] for record in step_records)
        and semigroup_defect <= gates["maximum_semigroup_relative_defect"]
        and cumulative_ledger_defect
        <= gates["maximum_exact_integrated_ledger_relative_defect"]
        and checkpoint_roundtrip_bitwise
        and suffix_replay_bitwise
        and reconstruction.maximum_macro_state_roundtrip_relative_defect
        <= reconstruction_contract[
            "maximum_macro_state_roundtrip_relative_defect"
        ]
        and float(np.max(np.abs(reconstruction.chart_coordinates)))
        <= reconstruction_contract["maximum_chart_coordinate_infinity"]
        and all(physical_checks.values())
        and max(output_defects.values())
        <= gates["maximum_endpoint_truth_output_relative_defect_per_block"]
        and max(rate_defects.values())
        <= gates["maximum_endpoint_truth_macro_rate_relative_defect_per_field"]
        and predicted_ledger_defect
        <= gates["maximum_exact_integrated_ledger_relative_defect"]
    )
    cost_passed = benchmark_seconds <= cost["maximum_benchmark_wall_seconds"]
    passed = scientific_passed and cost_passed
    classification = (
        PASS_CLASSIFICATION
        if passed
        else (
            COST_FAIL_CLASSIFICATION
            if scientific_passed
            else SCIENTIFIC_FAIL_CLASSIFICATION
        )
    )
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "scientific_passed": scientific_passed,
        "cost_passed": cost_passed,
        "spectral_abscissa_per_second": spectral_abscissa,
        "minimum_spectral_real_part_per_second": float(
            np.min(np.real(eigenvalues))
        ),
        "maximum_spectral_imaginary_magnitude_per_second": float(
            np.max(np.abs(np.imag(eigenvalues)))
        ),
        "transition_construction_wall_seconds": transition_construction_seconds,
        "step_records": step_records,
        "maximum_step_ledger_relative_defect": max(
            record["exact_integrated_ledger_relative_defect"]
            for record in step_records
        ),
        "cumulative_ledger_relative_defect": cumulative_ledger_defect,
        "same_horizon_semigroup_relative_defect": semigroup_defect,
        "checkpoint_bytes": len(checkpoint_bytes),
        "checkpoint_roundtrip_bitwise": checkpoint_roundtrip_bitwise,
        "suffix_replay_bitwise": suffix_replay_bitwise,
        "context_construction_wall_seconds": context_seconds,
        "endpoint_reconstruction_wall_seconds": reconstruction_seconds,
        "endpoint_reconstruction_newton_corrections": (
            reconstruction.newton_corrections
        ),
        "endpoint_reconstruction_roundtrip_relative_defect": (
            reconstruction.maximum_macro_state_roundtrip_relative_defect
        ),
        "endpoint_reconstruction_chart_coordinate_infinity": float(
            np.max(np.abs(reconstruction.chart_coordinates))
        ),
        "endpoint_reconstruction_maximum_local_condition_number": (
            reconstruction.maximum_local_jacobian_condition_number
        ),
        "endpoint_truth_operator_wall_seconds": truth_seconds,
        "endpoint_truth_physical": physical,
        "endpoint_truth_physical_checks": physical_checks,
        "endpoint_truth_all_physical_gates_passed": all(
            physical_checks.values()
        ),
        "endpoint_output_block_relative_defects": output_defects,
        "endpoint_macro_rate_relative_defects": rate_defects,
        "endpoint_maximum_output_relative_defect": max(output_defects.values()),
        "endpoint_maximum_macro_rate_relative_defect": max(rate_defects.values()),
        "endpoint_predicted_conservative_ledger_relative_defect": (
            predicted_ledger_defect
        ),
        "new_truth_operator_calls": 1,
        "new_global_roots": 0,
        "accepted_macrosteps": pilot["accepted_macrosteps"],
        "accepted_horizon_seconds": pilot["horizon_seconds"],
        "online_benchmark_macrosteps": cost["benchmark_macrosteps"],
        "online_benchmark_wall_seconds": benchmark_seconds,
        "online_average_wall_seconds_per_macrostep": (
            benchmark_seconds / cost["benchmark_macrosteps"]
        ),
        "online_benchmark_checksum": benchmark_checksum,
        "complete_cycle_execution_authorized": False,
    }
    arrays = {
        "macro_states": np.asarray(states),
        "integrated_packed_outputs": np.asarray(integrated_outputs),
        "normalized_rate_matrix": system.normalized_rate_matrix,
        "normalized_rate_offset": system.normalized_rate_offset,
        "augmented_generator": system.augmented_generator,
        "state_transition_1ms": transition.state_transition,
        "normalized_output_integral_1ms": transition.normalized_output_integral,
        "state_transition_4ms": full_transition.state_transition,
        "normalized_output_integral_4ms": full_transition.normalized_output_integral,
        "spectral_values_per_second": eigenvalues,
        "endpoint_reconstructed_primitive_charts": (
            reconstruction.primitive_charts
        ),
        "endpoint_reconstructed_chart_coordinates": (
            reconstruction.chart_coordinates
        ),
        "endpoint_truth_packed_outputs": pack_macro_outputs(truth_outputs),
        "endpoint_predicted_packed_outputs": pack_macro_outputs(
            predicted_outputs
        ),
        "endpoint_truth_macro_rates_per_second": (
            truth_outputs.macro_rates_per_second
        ),
        "endpoint_predicted_macro_rates_per_second": (
            predicted_outputs.macro_rates_per_second
        ),
        "checkpoint_bytes": np.frombuffer(checkpoint_bytes, dtype=np.uint8),
        **{
            f"atlas_{name}": value for name, value in atlas_arrays.items()
        },
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
        raise RuntimeError("macro-integrator implementation already exists")
    validated = _validate_parent(require_clean=True)
    utils = _utils()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "macro_integrator_metrics.json", metrics)
    np.savez_compressed(
        CANONICAL_DIRECTORY / "macro_integrator_arrays.npz", **arrays
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": bool(metrics["passed"]),
        "thermodynamic_chart_macro_atlas_preserved": True,
        "exact_affine_macro_integrator_certified": bool(metrics["scientific_passed"]),
        "online_cost_gate_passed": bool(metrics["cost_passed"]),
        "accepted_macrosteps": metrics["accepted_macrosteps"],
        "accepted_horizon_seconds": metrics["accepted_horizon_seconds"],
        "pathwise_macro_atlas_expansion_manifest_authorized": bool(
            metrics["passed"]
        ),
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
            "atlas_arrays_sha256": utils._sha256(ATLAS_ARRAYS),
            "truth_arrays_sha256": utils._sha256(TRUTH_ARRAYS),
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Entropy-complete structure-preserving macro-integrator implementation",
                "",
                f"Classification: `{metrics['classification']}`.",
                "",
                f"Four exact-affine 1 ms steps reached `{metrics['accepted_horizon_seconds']:.6e}` s. "
                f"The maximum chart coordinate was `{max(r['maximum_chart_coordinate'] for r in metrics['step_records']):.6e}`, "
                f"and the maximum exact integrated ledger defect was "
                f"`{metrics['maximum_step_ledger_relative_defect']:.6e}`.",
                "",
                f"The same-horizon semigroup defect was "
                f"`{metrics['same_horizon_semigroup_relative_defect']:.6e}`; "
                f"checkpoint roundtrip and suffix replay were bitwise: "
                f"`{metrics['checkpoint_roundtrip_bitwise']}` / "
                f"`{metrics['suffix_replay_bitwise']}`.",
                "",
                f"One full endpoint truth call passed physical gates: "
                f"`{metrics['endpoint_truth_all_physical_gates_passed']}`. "
                f"Maximum output/rate defects were "
                f"`{metrics['endpoint_maximum_output_relative_defect']:.6e}` / "
                f"`{metrics['endpoint_maximum_macro_rate_relative_defect']:.6e}`.",
                "",
                f"The 100,000-step online kernel benchmark took "
                f"`{metrics['online_benchmark_wall_seconds']:.6f}` s.",
                "",
                "One local patch does not cover a cycle. No complete-cycle execution is authorized.",
                "",
                f"Authorized next: `{summary['authorized_next']}`.",
                "",
            )
        ),
        encoding="utf-8",
    )
    sources = (THIS_RUNNER, THIS_TEST, SOURCE, SOURCE_TEST, REPORT_RELATIVE)
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
