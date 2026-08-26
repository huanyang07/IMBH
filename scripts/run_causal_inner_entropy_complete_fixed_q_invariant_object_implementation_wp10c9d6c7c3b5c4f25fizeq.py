#!/usr/bin/env python3
"""Certify the authentic cellwise fixed-Q projected field and coloring."""

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

import run_causal_inner_entropy_complete_fixed_q_invariant_object_manifest_wp10c9d6c7c3b5c4f25fizep as parent  # noqa: E402
import run_causal_inner_entropy_complete_bounded_radial_crossing_execution_wp10c9d6c7c3b5c4f25fizek as crossing  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_semidiscrete import (  # noqa: E402
    generalized_maxwell_cattaneo_hydrostatic_embedding,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_slow_manifold import (  # noqa: E402
    FAST_CHART_SCALES,
    directional_jacobian_relative_defect,
    generalized_maxwell_cattaneo_fast_charts,
    generalized_maxwell_cattaneo_fast_rate_scales,
    generalized_maxwell_cattaneo_projected_fast_evaluation,
    generalized_maxwell_cattaneo_slow_targets,
    radius_one_colored_jacobian,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fizeq_"
    "entropy_complete_fixed_Q_invariant_object_implementation"
)
PASS_CLASSIFICATION = "entropy_complete_cellwise_fixed_Q_projected_field_and_coloring_certified"
FAIL_CLASSIFICATION = "entropy_complete_cellwise_fixed_Q_projected_field_or_coloring_failed"
AUTHORIZED_NEXT_ON_PASS = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizer_"
    "entropy_complete_fixed_Q_primary_root_execution_manifest"
)
ARTIFACT = (
    "causal_inner_entropy_complete_fixed_q_invariant_object_implementation_"
    "wp10c9d6c7c3b5c4f25fizeq"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_FIXED_Q_"
    "INVARIANT_OBJECT_IMPLEMENTATION_WP10C9D6C7C3B5C4F25FIZEQ_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = "scripts/run_causal_inner_entropy_complete_fixed_q_invariant_object_implementation_wp10c9d6c7c3b5c4f25fizeq.py"
THIS_TEST = "tests/test_causal_inner_entropy_complete_fixed_q_invariant_object_implementation_wp10c9d6c7c3b5c4f25fizeq.py"
SOURCE = "src/imri_qpe/layer3_minidisk_1d/causal_inner_generalized_maxwell_cattaneo_slow_manifold.py"
SOURCE_TEST = "tests/test_causal_inner_generalized_maxwell_cattaneo_slow_manifold.py"
SOURCE_SHA256 = "060ffa8c038a338e4d8352bbb3465ba88367f11eaab240edd4abd2a89500d594"
SOURCE_TEST_SHA256 = "0d7166175fbf98caef1de22e4c46659500e755d4e488ee1714a59c88d4fe2cd1"
PARENT_CHECKSUM_MANIFEST_SHA256 = "f185811e468dc799ea765f1c1ead9dde244fb1668e6f5b03077aacc747dbd0e8"
AUDIT_ENVELOPE = "results/canonical/causal_inner_seven_field_physical_closure_local_audit_manifest_wp10c9d6c7c3b5c4f25fizeb/audit_envelope.npz"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
COLORED_STEP = 1.0e-6
JVP_STEP = 2.0e-6
JVP_DIRECTION_COUNT = 4
RANDOM_SEED = 20_260_826
EXPECTED_FIELD_CALLS = 1 + 12 + 2 * JVP_DIRECTION_COUNT


def _utils():
    return parent._utils()


def _validate_parent(*, require_clean: bool) -> dict:
    utils = _utils()
    if utils._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != PARENT_CHECKSUM_MANIFEST_SHA256:
        raise RuntimeError("fixed-Q invariant-object manifest checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    contract = utils._read_json(parent.CANONICAL_DIRECTORY / "invariant_object_contract.json")
    if (
        summary["classification"] != parent.CLASSIFICATION
        or not summary["passed"]
        or not summary["definitions_only"]
        or not summary["implementation_authorized"]
        or summary["nonlinear_root_execution_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["authorized_next"] != WORK_PACKAGE
    ):
        raise RuntimeError("fixed-Q implementation authorization changed")
    if utils._sha256(ROOT / SOURCE) != SOURCE_SHA256: raise RuntimeError("fixed-Q slow-manifold source changed")
    if utils._sha256(ROOT / SOURCE_TEST) != SOURCE_TEST_SHA256: raise RuntimeError("fixed-Q slow-manifold source test changed")
    for relative, expected in utils._read_json(parent.CANONICAL_DIRECTORY / "provenance.json")["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected: raise RuntimeError(f"fixed-Q manifest source changed: {relative}")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("fixed-Q implementation audit requires clean tracked tree")
    return {"hashes": hashes, "summary": summary, "contract": contract}


def _primary_setup():
    context, _terminal = crossing._context_and_seed()
    with np.load(ROOT / AUDIT_ENVELOPE, allow_pickle=False) as archive:
        profile = np.asarray(archive["primary_20ms_base_charts5"], dtype=float)
    charts = np.asarray([
        generalized_maxwell_cattaneo_hydrostatic_embedding(
            chart,
            proper_vertical_frequency=float(context.vertical_frequency.frequency(float(radius))),
        )
        for radius, chart in zip(context.grid.centers, profile, strict=True)
    ])
    return context, profile, charts


def _audit() -> tuple[dict, dict[str, np.ndarray]]:
    validated = _validate_parent(require_clean=False)
    contract = validated["contract"]
    setup_start = time.perf_counter(); context, profile, charts = _primary_setup(); setup_seconds = time.perf_counter() - setup_start
    targets = generalized_maxwell_cattaneo_slow_targets(context, charts)
    fast = generalized_maxwell_cattaneo_fast_charts(charts)
    base_start = time.perf_counter()
    base = generalized_maxwell_cattaneo_projected_fast_evaluation(
        context, targets, fast, template_charts=charts, quadrature_order=8
    )
    base_seconds = time.perf_counter() - base_start
    rate_scales = generalized_maxwell_cattaneo_fast_rate_scales(base.projected_fast_rates_per_second)
    base_normalized = base.projected_fast_rates_per_second / rate_scales
    calls = 1

    def field(coordinates: np.ndarray) -> np.ndarray:
        nonlocal calls
        if np.array_equal(coordinates, np.zeros_like(coordinates)):
            return np.array(base_normalized, copy=True)
        calls += 1
        candidate_fast = fast + np.asarray(coordinates) * FAST_CHART_SCALES
        evaluation = generalized_maxwell_cattaneo_projected_fast_evaluation(
            context,
            targets,
            candidate_fast,
            template_charts=charts,
            fast_rate_scales_per_second=rate_scales,
            quadrature_order=8,
        )
        return evaluation.normalized_fast_rates

    coordinate = np.zeros_like(fast)
    jacobian_start = time.perf_counter()
    recovered_base, jacobian = radius_one_colored_jacobian(
        field, coordinate, relative_step=COLORED_STEP
    )
    jacobian_seconds = time.perf_counter() - jacobian_start
    generator = np.random.default_rng(RANDOM_SEED)
    directions = generator.normal(size=(JVP_DIRECTION_COUNT, *coordinate.shape))
    jvp_start = time.perf_counter()
    jvp_defects = np.asarray([
        directional_jacobian_relative_defect(
            field,
            coordinate,
            jacobian,
            direction,
            relative_step=JVP_STEP,
        )
        for direction in directions
    ])
    jvp_seconds = time.perf_counter() - jvp_start
    operator = base.radial_operator
    physical = {
        "maximum_imaginary_speed_over_c": operator.maximum_imaginary_speed_over_c,
        "maximum_light_cone_excess_over_c": operator.maximum_light_cone_excess_over_c,
        "maximum_eigenvector_condition_number": operator.maximum_eigenvector_condition_number,
        "minimum_height_over_radius": operator.minimum_height_over_radius,
        "maximum_height_over_radius": operator.maximum_height_over_radius,
        "minimum_optical_depth": operator.minimum_optical_depth,
        "incoming_inner_characteristics": operator.incoming_inner_characteristics,
    }
    gates = contract["binding_physical_gates"]
    checks = {
        "constraint_reconstruction": base.reconstruction.maximum_constraint_relative_defect <= contract["fixed_slow_reconstruction"]["maximum_constraint_relative_defect"],
        "temporal_projection": base.maximum_temporal_projection_solve_relative_defect <= gates["maximum_temporal_projection_solve_relative_defect"],
        "colored_base": np.array_equal(recovered_base, base_normalized),
        "colored_shape": jacobian.shape == (448, 448),
        "colored_JVP": float(np.max(jvp_defects)) <= contract["sparse_derivative"]["maximum_colored_JVP_relative_defect"],
        "field_call_budget": calls == EXPECTED_FIELD_CALLS,
        "imaginary_speed": physical["maximum_imaginary_speed_over_c"] <= gates["maximum_imaginary_speed_over_c"],
        "light_cone": physical["maximum_light_cone_excess_over_c"] <= gates["maximum_light_cone_excess_over_c"],
        "eigenbasis_condition": physical["maximum_eigenvector_condition_number"] <= gates["eigenvector_condition_number_max"],
        "height": physical["minimum_height_over_radius"] >= gates["minimum_height_over_radius"] and physical["maximum_height_over_radius"] <= gates["maximum_height_over_radius"],
        "optical_depth": physical["minimum_optical_depth"] >= gates["minimum_optical_depth"],
        "inner_excision": physical["incoming_inner_characteristics"] == gates["inner_incoming_characteristics_equal"],
    }
    passed = all(checks.values())
    eigenvalues = np.linalg.eigvals(jacobian)
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,
        "passed": passed,
        "checks": checks,
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "projected_field_calls": calls,
        "expected_projected_field_calls": EXPECTED_FIELD_CALLS,
        "setup_wall_seconds": setup_seconds,
        "base_field_wall_seconds": base_seconds,
        "colored_jacobian_wall_seconds": jacobian_seconds,
        "independent_JVP_wall_seconds": jvp_seconds,
        "maximum_colored_JVP_relative_defect": float(np.max(jvp_defects)),
        "individual_colored_JVP_relative_defects": jvp_defects.tolist(),
        "base_normalized_fast_rate_infinity": float(np.max(np.abs(base_normalized))),
        "base_fast_rate_norm_per_second": float(np.linalg.norm(base.projected_fast_rates_per_second)),
        "base_colored_jacobian_spectral_abscissa_per_second_scaled_coordinates": float(np.max(eigenvalues.real)),
        "maximum_constraint_relative_defect": base.reconstruction.maximum_constraint_relative_defect,
        "maximum_temporal_projection_solve_relative_defect": base.maximum_temporal_projection_solve_relative_defect,
        "physical": physical,
    }
    arrays = {
        "primary_base_charts5": profile,
        "primary_lifted_charts7": charts,
        "slow_targets_MJE": targets,
        "base_fast_charts": fast,
        "fast_chart_scales": FAST_CHART_SCALES,
        "fast_rate_scales_per_second": rate_scales,
        "base_normalized_fast_rates": base_normalized,
        "colored_normalized_fast_rate_jacobian": jacobian,
        "JVP_directions": directions,
        "JVP_relative_defects": jvp_defects,
        "base_slow_integrated_drift_per_second": base.slow_integrated_drift_per_second,
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
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists(): raise RuntimeError("fixed-Q implementation certificate already exists")
    validated = _validate_parent(require_clean=True); utils = _utils(); CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "implementation_metrics.json", metrics); np.savez_compressed(CANONICAL_DIRECTORY / "implementation_arrays.npz", **arrays)
    summary = {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "classification": metrics["classification"], "passed": bool(metrics["passed"]), "corrected_bounded_crossing_preserved": True, "cellwise_fixed_Q_split_preserved": True, "projected_fast_field_certified": bool(metrics["passed"]), "colored_fast_Jacobian_certified": bool(metrics["passed"]), "new_nonlinear_roots": 0, "propagated_states": 0, "primary_root_execution_manifest_authorized": bool(metrics["passed"]), "primary_root_execution_authorized": False, "slow_flux_atlas_authorized": False, "complete_cycle_execution_authorized": False, "authorized_next": AUTHORIZED_NEXT_ON_PASS if metrics["passed"] else None}
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary); utils._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"parent_artifact": parent.ARTIFACT, "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256, "parent_hashes": validated["hashes"], "source_sha256": SOURCE_SHA256, "source_test_sha256": SOURCE_TEST_SHA256, "audit_envelope_sha256": utils._sha256(ROOT / AUDIT_ENVELOPE)})
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True); REPORT_PATH.write_text("\n".join(("# Entropy-complete fixed-Q invariant-object implementation", "", f"Classification: `{metrics['classification']}`.", "", f"Projected-field calls: `{metrics['projected_field_calls']}`; maximum independent colored-JVP defect: `{metrics['maximum_colored_JVP_relative_defect']:.6e}`. No nonlinear root or physical state was propagated.", "", f"Authorized next: `{summary['authorized_next']}`.", "")), encoding="utf-8")
    sources = (THIS_RUNNER, THIS_TEST, SOURCE, SOURCE_TEST, REPORT_RELATIVE); utils._write_json(CANONICAL_DIRECTORY / "provenance.json", {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "implementation_commit": utils._git("rev-parse", "HEAD"), "implementation_tree": utils._git("rev-parse", "HEAD^{tree}"), "source_hashes": {path: utils._sha256(ROOT / path) for path in sources}, "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "thread_environment": {name: os.environ.get(name, "") for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()); (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utils._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8"); _update_catalog(summary); return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run", action="store_true"); args = parser.parse_args()
    if not args.run: parser.error("choose --run")
    metrics, arrays = _audit(); print(json.dumps(metrics, indent=2, sort_keys=True), flush=True); summary = _canonicalize(metrics, arrays); return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
