#!/usr/bin/env python3
"""Freeze a nonpropagating inexact-Newton recovery trial."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_entropy_complete_fixed_q_equation_form_root_preflight_wp10c9d6c7c3b5c4f25fizes as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fizet_"
    "entropy_complete_fixed_Q_inexact_trust_recovery_manifest"
)
CLASSIFICATION = "entropy_complete_fixed_Q_inexact_trust_recovery_manifest_frozen"
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizeu_"
    "entropy_complete_fixed_Q_inexact_trust_trial_execution"
)
ARTIFACT = (
    "causal_inner_entropy_complete_fixed_q_inexact_trust_recovery_manifest_"
    "wp10c9d6c7c3b5c4f25fizet"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_FIXED_Q_"
    "INEXACT_TRUST_RECOVERY_MANIFEST_WP10C9D6C7C3B5C4F25FIZET_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_entropy_complete_fixed_q_inexact_trust_"
    "recovery_manifest_wp10c9d6c7c3b5c4f25fizet.py"
)
THIS_TEST = (
    "tests/test_causal_inner_entropy_complete_fixed_q_inexact_trust_"
    "recovery_manifest_wp10c9d6c7c3b5c4f25fizet.py"
)
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "dc55606d79dd93d141e9c9b0c574f201e48d55632e26d252b271f124bf7dd6d8"
)
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _utils():
    return parent._utils()


def _saved_direction_diagnostics() -> dict:
    with np.load(parent.CANONICAL_DIRECTORY / "preflight_arrays.npz") as archive:
        jacobian = archive["colored_normalized_equation_jacobian"]
        residual = archive["base_normalized_equation_residual"].ravel()
        step = archive["bounded_linear_step"].ravel()
    predicted = residual + jacobian @ step
    gradient = jacobian.T @ residual
    final_gradient = jacobian.T @ predicted
    trust = 0.25
    tolerance = trust * 1.0e-8
    lower = step <= -trust + tolerance
    upper = step >= trust - tolerance
    interior = ~(lower | upper)
    violation = np.zeros_like(final_gradient)
    violation[interior] = np.abs(final_gradient[interior])
    violation[lower] = np.maximum(-final_gradient[lower], 0.0)
    violation[upper] = np.maximum(final_gradient[upper], 0.0)
    scale = max(float(np.max(np.abs(gradient))), np.finfo(float).tiny)
    return {
        "forcing_two_norm": float(np.linalg.norm(predicted) / np.linalg.norm(residual)),
        "forcing_infinity_norm": float(
            np.max(np.abs(predicted)) / np.max(np.abs(residual))
        ),
        "normalized_directional_derivative": float(
            (gradient @ step) / (residual @ residual)
        ),
        "relative_projected_KKT_infinity": float(np.max(violation) / scale),
        "maximum_absolute_step": float(np.max(np.abs(step))),
        "active_bound_count": int(np.sum(lower | upper)),
    }


def _validate_parent(*, require_clean: bool) -> dict:
    utils = _utils()
    if (
        utils._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt")
        != PARENT_CHECKSUM_MANIFEST_SHA256
    ):
        raise RuntimeError("equation-form preflight rejection checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = utils._read_json(parent.CANONICAL_DIRECTORY / "preflight_metrics.json")
    failed = [name for name, passed in metrics["checks"].items() if not passed]
    if (
        summary["classification"] != parent.FAIL_CLASSIFICATION
        or summary["passed"]
        or summary["authorized_next"] is not None
        or summary["new_nonlinear_roots"] != 0
        or summary["propagated_states"] != 0
        or failed != ["bounded_linear_solver"]
        or not metrics["checks"]["predicted_infinity_reduction"]
        or not metrics["checks"]["predicted_two_norm_reduction"]
        or not metrics["checks"]["colored_JVP"]
        or not metrics["checks"]["equation_rate_parity"]
    ):
        raise RuntimeError("equation-form rejection classification changed")
    for relative, expected in utils._read_json(
        parent.CANONICAL_DIRECTORY / "provenance.json"
    )["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"equation-form preflight source changed: {relative}")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("inexact-trust manifest requires a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "metrics": metrics}


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "preserved_rejection": {
            "classification": parent.FAIL_CLASSIFICATION,
            "failed_gate": "bounded_linear_solver_library_success_status",
            "not_a_physical_failure": True,
            "not_a_derivative_failure": True,
            "not_retroactively_converted_to_pass": True,
        },
        "inexact_Newton_direction": {
            "source": "hash_locked_saved_500_iteration_TRF_bounded_step",
            "maximum_forcing_two_norm": 0.80,
            "maximum_forcing_infinity_norm": 0.70,
            "maximum_relative_projected_KKT_infinity": 0.10,
            "maximum_normalized_directional_derivative": -0.10,
            "maximum_absolute_scaled_step": 0.25,
            "global_linear_subproblem_optimality_is_not_required": True,
            "sufficient_bounded_descent_is_binding": True,
        },
        "nonpropagating_physical_trial": {
            "ordered_step_factors": [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125],
            "accept_first_factor_only": True,
            "two_norm_Armijo_coefficient": 1.0e-4,
            "strict_infinity_norm_decrease_required": True,
            "unchanged_fixed_slow_targets_and_equation_row_scales": True,
            "all_existing_physical_gates_required": True,
            "maximum_physical_field_calls": 8,
            "accepted_trial_is_not_a_root": True,
            "accepted_trial_must_not_be_propagated": True,
        },
        "decision": {
            "actual_descent_and_physical_gates_pass": (
                "authorize_definitions_only_primary_inexact_Newton_root_execution_manifest"
            ),
            "all_backtracks_fail": (
                "stop_and_design_structured_block_or_pseudo_transient_solver"
            ),
            "no_threshold_may_be_relaxed_after_observing_results": True,
        },
        "claim_boundary": {
            "one_nonpropagating_trial_authorized": True,
            "primary_root_execution_authorized": False,
            "heldout_root_execution_authorized": False,
            "slow_flux_atlas_authorized": False,
            "complete_cycle_execution_authorized": False,
        },
        "authorized_next": AUTHORIZED_NEXT,
    }


def _update_catalog(summary: dict) -> None:
    utils = _utils()
    with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": utils._sha256(path), "scientific_status": "DEFINITIONS_ONLY"})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    catalog = utils._read_json(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": summary["classification"], "passed": summary["passed"]}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": utils._git("rev-parse", "HEAD"), "latest_work_package": WORK_PACKAGE})
    utils._write_json(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("inexact-trust recovery manifest already exists")
    validated = _validate_parent(require_clean=True)
    diagnostics = _saved_direction_diagnostics()
    contract = _contract()
    limits = contract["inexact_Newton_direction"]
    if not (
        diagnostics["forcing_two_norm"] <= limits["maximum_forcing_two_norm"]
        and diagnostics["forcing_infinity_norm"] <= limits["maximum_forcing_infinity_norm"]
        and diagnostics["relative_projected_KKT_infinity"] <= limits["maximum_relative_projected_KKT_infinity"]
        and diagnostics["normalized_directional_derivative"] <= limits["maximum_normalized_directional_derivative"]
        and diagnostics["maximum_absolute_step"] <= limits["maximum_absolute_scaled_step"] * (1.0 + 1.0e-12)
    ):
        raise RuntimeError("saved inexact direction does not meet the prospective recovery gates")
    utils = _utils(); CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "inexact_trust_contract.json", contract)
    utils._write_json(CANONICAL_DIRECTORY / "saved_direction_diagnostics.json", diagnostics)
    utils._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"parent_artifact": parent.ARTIFACT, "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256, "parent_hashes": validated["hashes"]})
    summary = {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "classification": CLASSIFICATION, "passed": True, "definitions_only": True, "equation_form_preflight_rejection_preserved": True, "saved_inexact_direction_qualified": True, "one_nonpropagating_trial_authorized": True, "primary_root_execution_authorized": False, "slow_flux_atlas_authorized": False, "complete_cycle_execution_authorized": False, "authorized_next": AUTHORIZED_NEXT}
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(("# Entropy-complete fixed-Q inexact-trust recovery manifest", "", f"Classification: `{CLASSIFICATION}`.", "", "The prior TRF status rejection remains binding. Its saved bounded step is reclassified prospectively only as an inexact-Newton descent direction and will be evaluated by a nonpropagating physical line search.", "", f"Authorized next: `{AUTHORIZED_NEXT}` only.", "")), encoding="utf-8")
    sources = (THIS_RUNNER, THIS_TEST, REPORT_RELATIVE)
    utils._write_json(CANONICAL_DIRECTORY / "provenance.json", {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "scientific_status": "DEFINITIONS_ONLY", "implementation_commit": utils._git("rev-parse", "HEAD"), "implementation_tree": utils._git("rev-parse", "HEAD^{tree}"), "source_hashes": {path: utils._sha256(ROOT / path) for path in sources}, "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "thread_environment": {name: os.environ.get(name, "") for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utils._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8")
    _update_catalog(summary); return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--freeze", action="store_true"); args = parser.parse_args()
    if not args.freeze: parser.error("choose --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
