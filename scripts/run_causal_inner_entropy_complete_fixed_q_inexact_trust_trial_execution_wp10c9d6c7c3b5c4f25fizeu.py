#!/usr/bin/env python3
"""Execute the frozen nonpropagating inexact-Newton physical trial."""

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

import run_causal_inner_entropy_complete_fixed_q_inexact_trust_recovery_manifest_wp10c9d6c7c3b5c4f25fizet as parent  # noqa: E402
import run_causal_inner_entropy_complete_fixed_q_equation_form_root_preflight_wp10c9d6c7c3b5c4f25fizes as preflight  # noqa: E402
import run_causal_inner_entropy_complete_fixed_q_invariant_object_implementation_wp10c9d6c7c3b5c4f25fizeq as implementation  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_fixed_slow_root import (  # noqa: E402
    equation_rate_parity_relative_defect,
    projected_fast_temporal_blocks,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_slow_manifold import (  # noqa: E402
    FAST_EQUATION_ROWS,
    generalized_maxwell_cattaneo_projected_fast_evaluation,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fizeu_"
    "entropy_complete_fixed_Q_inexact_trust_trial_execution"
)
PASS_CLASSIFICATION = "entropy_complete_fixed_Q_inexact_trust_physical_trial_passed"
FAIL_CLASSIFICATION = "entropy_complete_fixed_Q_inexact_trust_physical_trial_failed"
AUTHORIZED_NEXT_ON_PASS = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizev_"
    "entropy_complete_fixed_Q_primary_inexact_Newton_root_execution_manifest"
)
ARTIFACT = (
    "causal_inner_entropy_complete_fixed_q_inexact_trust_trial_execution_"
    "wp10c9d6c7c3b5c4f25fizeu"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_FIXED_Q_"
    "INEXACT_TRUST_TRIAL_EXECUTION_WP10C9D6C7C3B5C4F25FIZEU_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_entropy_complete_fixed_q_inexact_trust_trial_"
    "execution_wp10c9d6c7c3b5c4f25fizeu.py"
)
THIS_TEST = (
    "tests/test_causal_inner_entropy_complete_fixed_q_inexact_trust_trial_"
    "execution_wp10c9d6c7c3b5c4f25fizeu.py"
)
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "fca84fb6546d67efa835f89a0fa7ed0143864eb9e179b74fa6c9cd2aca812d2a"
)
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _utils():
    return parent._utils()


def _validate_parent(*, require_clean: bool) -> dict:
    utils = _utils()
    if utils._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != PARENT_CHECKSUM_MANIFEST_SHA256:
        raise RuntimeError("inexact-trust recovery manifest checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    contract = utils._read_json(parent.CANONICAL_DIRECTORY / "inexact_trust_contract.json")
    if (
        summary["classification"] != parent.CLASSIFICATION
        or not summary["passed"]
        or not summary["definitions_only"]
        or not summary["saved_inexact_direction_qualified"]
        or not summary["one_nonpropagating_trial_authorized"]
        or summary["primary_root_execution_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["authorized_next"] != WORK_PACKAGE
    ):
        raise RuntimeError("inexact-trust physical trial authorization changed")
    for relative, expected in utils._read_json(parent.CANONICAL_DIRECTORY / "provenance.json")["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"inexact-trust manifest source changed: {relative}")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("inexact-trust trial requires a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "contract": contract}


def _physical_checks(evaluation, contract: dict) -> dict:
    gates = preflight.parent._contract()["binding_physical_gates"]
    operator = evaluation.radial_operator
    return {
        "constraint_reconstruction": evaluation.reconstruction.maximum_constraint_relative_defect <= gates["maximum_fixed_slow_reconstruction_relative_defect"],
        "temporal_projection": evaluation.maximum_temporal_projection_solve_relative_defect <= gates["maximum_temporal_projection_solve_relative_defect"],
        "imaginary_speed": operator.maximum_imaginary_speed_over_c <= gates["maximum_imaginary_speed_over_c"],
        "light_cone": operator.maximum_light_cone_excess_over_c <= gates["maximum_light_cone_excess_over_c"],
        "eigenbasis_condition": operator.maximum_eigenvector_condition_number <= gates["eigenvector_condition_number_max"],
        "height": operator.minimum_height_over_radius >= gates["minimum_height_over_radius"] and operator.maximum_height_over_radius <= gates["maximum_height_over_radius"],
        "optical_depth": operator.minimum_optical_depth >= gates["minimum_optical_depth"],
        "inner_excision": operator.incoming_inner_characteristics == gates["inner_incoming_characteristics_equal"],
    }


def _execute() -> tuple[dict, dict[str, np.ndarray]]:
    validated = _validate_parent(require_clean=False)
    contract = validated["contract"]
    trial_contract = contract["nonpropagating_physical_trial"]
    context, _profile, setup_charts = implementation._primary_setup()
    with np.load(preflight.CANONICAL_DIRECTORY / "preflight_arrays.npz") as archive:
        charts = archive["primary_lifted_charts7"]
        targets = archive["slow_targets_MJE"]
        fast = archive["base_fast_charts"]
        chart_scales = archive["fast_chart_scales"]
        equation_scales = archive["equation_row_scales_per_cm"]
        base_residual = archive["base_normalized_equation_residual"]
        step = archive["bounded_linear_step"]
    if not np.array_equal(setup_charts, charts):
        raise RuntimeError("primary setup no longer reproduces the saved charts")
    base_two = float(np.linalg.norm(base_residual))
    base_inf = float(np.max(np.abs(base_residual)))
    records = []
    selected = None
    selected_evaluation = None
    selected_residual = None
    start = time.perf_counter()
    for factor in trial_contract["ordered_step_factors"]:
        candidate_fast = fast + float(factor) * step * chart_scales
        record = {"factor": float(factor)}
        try:
            evaluation = generalized_maxwell_cattaneo_projected_fast_evaluation(
                context,
                targets,
                candidate_fast,
                template_charts=charts,
                quadrature_order=8,
            )
            right = evaluation.radial_operator.equation_right_hand_sides_per_cm[:, FAST_EQUATION_ROWS]
            residual = right / equation_scales
            temporal_blocks, _tangent = projected_fast_temporal_blocks(
                context, evaluation.reconstruction.primitive_charts
            )
            parity = equation_rate_parity_relative_defect(
                temporal_blocks,
                evaluation.projected_fast_rates_per_second,
                right,
            )
            checks = _physical_checks(evaluation, contract)
            checks["equation_rate_parity"] = parity <= preflight.parent._contract()["root_equivalence"]["maximum_equation_rate_parity_relative_defect"]
            two = float(np.linalg.norm(residual))
            inf = float(np.max(np.abs(residual)))
            armijo = two <= (1.0 - trial_contract["two_norm_Armijo_coefficient"] * float(factor)) * base_two
            strict_inf = inf < base_inf
            record.update({"two_norm": two, "infinity_norm": inf, "two_norm_ratio": two / base_two, "infinity_norm_ratio": inf / base_inf, "equation_rate_parity_relative_defect": parity, "physical_checks": checks, "Armijo_two_norm": armijo, "strict_infinity_decrease": strict_inf, "accepted": bool(armijo and strict_inf and all(checks.values()))})
            if record["accepted"]:
                selected = float(factor)
                selected_evaluation = evaluation
                selected_residual = residual
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
            record.update({"accepted": False, "exception": f"{type(error).__name__}: {error}"})
        records.append(record)
        if selected is not None:
            break
    wall = time.perf_counter() - start
    passed = selected is not None
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,
        "passed": passed,
        "equation_form_preflight_rejection_preserved": True,
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "physical_field_calls": len(records),
        "maximum_physical_field_calls": trial_contract["maximum_physical_field_calls"],
        "selected_step_factor": selected,
        "base_two_norm": base_two,
        "base_infinity_norm": base_inf,
        "trial_records": records,
        "execution_wall_seconds": wall,
    }
    arrays = {
        "primary_lifted_charts7": charts,
        "slow_targets_MJE": targets,
        "base_fast_charts": fast,
        "fast_chart_scales": chart_scales,
        "equation_row_scales_per_cm": equation_scales,
        "base_normalized_equation_residual": base_residual,
        "saved_bounded_linear_step": step,
    }
    if passed:
        arrays.update({
            "selected_trial_fast_charts": fast + selected * step * chart_scales,
            "selected_trial_primitive_charts7": selected_evaluation.reconstruction.primitive_charts,
            "selected_trial_normalized_equation_residual": selected_residual,
            "selected_trial_projected_fast_rates_per_second": selected_evaluation.projected_fast_rates_per_second,
        })
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
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists(): raise RuntimeError("inexact-trust trial certificate already exists")
    validated = _validate_parent(require_clean=True); utils = _utils(); CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "trial_metrics.json", metrics); np.savez_compressed(CANONICAL_DIRECTORY / "trial_arrays.npz", **arrays)
    summary = {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "classification": metrics["classification"], "passed": bool(metrics["passed"]), "equation_form_preflight_rejection_preserved": True, "nonpropagating_inexact_trial_passed": bool(metrics["passed"]), "new_nonlinear_roots": 0, "propagated_states": 0, "primary_inexact_Newton_root_execution_manifest_authorized": bool(metrics["passed"]), "primary_root_execution_authorized": False, "slow_flux_atlas_authorized": False, "complete_cycle_execution_authorized": False, "authorized_next": AUTHORIZED_NEXT_ON_PASS if metrics["passed"] else None}
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary); utils._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"parent_artifact": parent.ARTIFACT, "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256, "parent_hashes": validated["hashes"]})
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True); REPORT_PATH.write_text("\n".join(("# Entropy-complete fixed-Q inexact-trust physical trial", "", f"Classification: `{metrics['classification']}`.", "", f"Selected step factor: `{metrics['selected_step_factor']}` after `{metrics['physical_field_calls']}` physical call(s). No nonlinear root or physical state was propagated.", "", f"Authorized next: `{summary['authorized_next']}`.", "")), encoding="utf-8")
    sources = (THIS_RUNNER, THIS_TEST, REPORT_RELATIVE); utils._write_json(CANONICAL_DIRECTORY / "provenance.json", {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "implementation_commit": utils._git("rev-parse", "HEAD"), "implementation_tree": utils._git("rev-parse", "HEAD^{tree}"), "source_hashes": {path: utils._sha256(ROOT / path) for path in sources}, "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "thread_environment": {name: os.environ.get(name, "") for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()); (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utils._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8"); _update_catalog(summary); return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run", action="store_true"); args = parser.parse_args()
    if not args.run: parser.error("choose --run")
    metrics, arrays = _execute(); print(json.dumps(metrics, indent=2, sort_keys=True), flush=True); summary = _canonicalize(metrics, arrays); return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
