#!/usr/bin/env python3
"""Freeze the bounded full-bundle transient acquisition architecture."""

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

import run_causal_inner_entropy_complete_selectively_refreshed_third_patch_execution_wp10c9d6c7c3b5c4f25fizfq as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fizr_"
    "entropy_complete_adaptive_selective_refresh_cycle_readiness_manifest"
)
CLASSIFICATION = (
    "entropy_complete_two_regime_full_bundle_transient_to_conservative_cycle_"
    "readiness_manifest_frozen"
)
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizs_"
    "entropy_complete_bounded_full_bundle_transient_acquisition_execution"
)
ARTIFACT = (
    "causal_inner_entropy_complete_adaptive_selective_refresh_cycle_readiness_"
    "manifest_wp10c9d6c7c3b5c4f25fizr"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_ADAPTIVE_"
    "SELECTIVE_REFRESH_CYCLE_READINESS_MANIFEST_"
    "WP10C9D6C7C3B5C4F25FIZR_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_entropy_complete_adaptive_selective_refresh_"
    "cycle_readiness_manifest_wp10c9d6c7c3b5c4f25fizr.py"
)
THIS_TEST = (
    "tests/test_causal_inner_entropy_complete_adaptive_selective_refresh_"
    "cycle_readiness_manifest_wp10c9d6c7c3b5c4f25fizr.py"
)
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "1ae2793b1407c7efa63ae543fd9687eb9128e55932edef9ae0d4d6eb554b2c63"
)
PARENT_ARRAYS = parent.CANONICAL_DIRECTORY / "selective_refresh_arrays.npz"
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
        raise RuntimeError("selectively refreshed patch checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = utils._read_json(
        parent.CANONICAL_DIRECTORY / "selective_refresh_metrics.json"
    )
    if (
        summary["classification"] != parent.PASS_CLASSIFICATION
        or not summary["passed"]
        or not summary["selectively_refreshed_third_patch_certified"]
        or not summary["adaptive_selective_refresh_cycle_readiness_manifest_authorized"]
        or summary["complete_cycle_execution_authorized"]
        or summary["authorized_next"] != f"definitions_only_{WORK_PACKAGE}"
        or not metrics["passed"]
        or metrics["accepted_absolute_horizon_seconds"] != 0.012
        or metrics["maximum_independent_JVP_relative_defect"] > 0.05
        or metrics["endpoint_maximum_output_relative_defect"] > 0.05
    ):
        raise RuntimeError("selectively refreshed patch authorization changed")
    for relative, expected in utils._read_json(
        parent.CANONICAL_DIRECTORY / "provenance.json"
    )["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"selective patch source changed: {relative}")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("cycle-readiness manifest needs a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "metrics": metrics}


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "authorized_next": AUTHORIZED_NEXT,
        "mathematical_architecture": {
            "regime_A": (
                "accepted 80-coordinate (M,J,E,beta_r,chi) conservative transient"
            ),
            "regime_A_integrator": (
                "equal-step AB2 predictor with one exact seven-field truth endpoint "
                "and a trapezoidal embedded defect"
            ),
            "regime_A_reason": (
                "the instantaneous auxiliary fixed point is not certified and all "
                "80 macro coordinates remain dynamic during acquisition"
            ),
            "regime_B": (
                "48-coordinate conservative M,J,E evolution with beta_r and chi "
                "supplied by a separately certified attracting graph"
            ),
            "regime_B_entry": (
                "persistent auxiliary-slaving observation followed by a fresh "
                "selected/full tangent and normal-attraction certificate"
            ),
            "online_cycle_truth_calls": 0,
            "online_cycle_global_roots": 0,
            "online_cycle_micro_BDF_steps": 0,
            "single_valued_face_fluxes_and_exact_MJE_ledger": True,
        },
        "preserved_negative_results": {
            "generic_fixed_Q_global_root_rejected": True,
            "unchanged_chain_rule_tangent_rejected": True,
            "instantaneous_32_auxiliary_coordinate_root_not_authorized": True,
            "no_global_fixed_point_or_periodic_orbit_assumed": True,
            "no_failed_candidate_may_define_history": True,
        },
        "bounded_execution": {
            "seed": "hash-validated certified 8 ms and 12 ms truth endpoints",
            "initial_absolute_elapsed_seconds": 0.012,
            "fixed_macrostep_seconds": 0.004,
            "new_macrosteps": 50,
            "target_absolute_elapsed_seconds": 0.212,
            "truth_operator_calls_per_attempted_step": 1,
            "maximum_new_truth_operator_calls": 50,
            "new_global_roots": 0,
            "fixed_Q_reaction_calls": 0,
            "reconstruct_every_candidate_in_the_seven_field_model": True,
            "recenter_thermodynamic_chart_after_every_accepted_step": True,
            "accepted_history_only": True,
        },
        "numerical_gates": {
            "maximum_local_reconstruction_chart_coordinate": 0.15,
            "maximum_macro_roundtrip_relative_defect": 1.0e-10,
            "maximum_AB2_trapezoidal_embedded_defect": 0.05,
            "maximum_discrete_conservative_ledger_relative_defect": 1.0e-12,
            "maximum_step_scaled_coordinate_change": 0.15,
            "fail_on_first_rejected_step": True,
        },
        "physical_gates": {
            "maximum_height_over_radius": 0.5,
            "minimum_height_over_radius": 1.0e-4,
            "minimum_optical_depth": 1.0,
            "maximum_imaginary_speed_over_c": 1.0e-10,
            "maximum_light_cone_excess_over_c": 1.0e-10,
            "maximum_eigenvector_condition_number": 1.0e8,
            "incoming_inner_characteristics_equal": 0,
            "maximum_temporal_solve_relative_defect": 1.0e-10,
            "all_midpoint_pencils_real": True,
        },
        "slaving_observation": {
            "diagnostic_only_during_this_execution": True,
            "normalized_auxiliary_to_conservative_rate_ratio_maximum": 0.1,
            "normalized_auxiliary_rate_infinity_per_second_maximum": 0.1,
            "required_consecutive_accepted_steps": 8,
            "fresh_tangent_certificate_still_required": True,
            "maximum_fast_block_spectral_abscissa_per_second": -1.0,
            "instantaneous_switch_during_the_bounded_run": False,
        },
        "decision": {
            "persistent_slaving_observed": (
                "authorize definitions-only terminal fast-graph tangent certificate"
            ),
            "physical_transient_extended_without_slaving": (
                "authorize one cost-bounded continuation manifest only"
            ),
            "numerical_or_physical_failure": "stop the cycle architecture path",
            "no_retrospective_gate_change": True,
        },
        "claim_boundary": {
            "bounded_transient_execution_authorized": True,
            "instantaneous_fast_graph_authorized": False,
            "48_coordinate_cycle_solver_authorized": False,
            "complete_cycle_execution_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
    }


def _diagnostics(validated: dict) -> dict:
    metrics = validated["metrics"]
    with np.load(PARENT_ARRAYS) as archive:
        eigenvalues = np.asarray(archive["patch_3_spectral_values_per_second"])
    real = np.real(eigenvalues)
    cycle_seconds = 578_880.0
    step = _contract()["bounded_execution"]["fixed_macrostep_seconds"]
    return {
        "certified_absolute_horizon_seconds": metrics[
            "accepted_absolute_horizon_seconds"
        ],
        "patch_3_spectral_abscissa_per_second": float(np.max(real)),
        "patch_3_minimum_spectral_real_part_per_second": float(np.min(real)),
        "slowest_local_efold_seconds": float(-1.0 / np.max(real)),
        "complete_cycle_seconds": cycle_seconds,
        "naive_fixed_patch_steps_per_cycle": int(np.ceil(cycle_seconds / step)),
        "naive_selective_truth_calls_per_cycle": int(
            np.ceil(cycle_seconds / step) * metrics["new_truth_operator_calls"]
        ),
        "naive_patchwise_cycle_route_rejected_as_infeasible": True,
        "selected_architecture_requires_offline_transient_then_zero_truth_online": True,
        "parent_endpoint_maximum_output_defect": metrics[
            "endpoint_maximum_output_relative_defect"
        ],
        "parent_maximum_blind_JVP_defect": metrics[
            "maximum_independent_JVP_relative_defect"
        ],
    }


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
                    "scientific_status": "DEFINITIONS_ONLY",
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


def _freeze() -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("cycle-readiness manifest already exists")
    validated = _validate_parent(require_clean=True)
    utils = _utils()
    contract = _contract()
    diagnostics = _diagnostics(validated)
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(
        CANONICAL_DIRECTORY / "cycle_readiness_contract.json", contract
    )
    utils._write_json(
        CANONICAL_DIRECTORY / "cycle_readiness_diagnostics.json", diagnostics
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "two_regime_architecture_selected": True,
        "naive_patchwise_cycle_route_rejected_as_infeasible": True,
        "bounded_transient_execution_authorized": True,
        "instantaneous_fast_graph_authorized": False,
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utils._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_artifact": parent.ARTIFACT,
            "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256,
            "parent_hashes": validated["hashes"],
            "parent_arrays_sha256": utils._sha256(PARENT_ARRAYS),
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Entropy-complete adaptive cycle-readiness manifest",
                "",
                f"Classification: `{CLASSIFICATION}`.",
                "",
                "The certified 4 ms selective patch cannot be tiled across a 578,880 s cycle: that naive route would require more than 144 million patches and three billion truth calls. The selected architecture therefore separates an offline 80-coordinate physical transient from a zero-truth online conservative cycle regime.",
                "",
                "The next execution advances only 0.2 s with 50 accepted-history-only AB2 macro steps and one exact seven-field endpoint per step. A 48-coordinate cycle solver remains forbidden until persistent auxiliary slaving and a fresh normal-attraction tangent certificate both pass.",
                "",
                f"Authorized next: `{AUTHORIZED_NEXT}` only.",
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
            "scientific_status": "DEFINITIONS_ONLY",
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
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if not args.freeze:
        parser.error("choose --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
