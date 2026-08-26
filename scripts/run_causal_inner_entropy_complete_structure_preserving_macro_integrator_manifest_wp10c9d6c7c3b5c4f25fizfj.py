#!/usr/bin/env python3
"""Freeze the bounded structure-preserving affine macro-integrator pilot."""

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

import run_causal_inner_entropy_complete_thermodynamic_chart_atlas_execution_wp10c9d6c7c3b5c4f25fizfi as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fizfj_"
    "entropy_complete_structure_preserving_macro_integrator_manifest"
)
CLASSIFICATION = (
    "entropy_complete_exact_affine_macro_integrator_bounded_pilot_manifest_frozen"
)
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizfk_"
    "entropy_complete_structure_preserving_macro_integrator_implementation"
)
ARTIFACT = (
    "causal_inner_entropy_complete_structure_preserving_macro_integrator_"
    "manifest_wp10c9d6c7c3b5c4f25fizfj"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_STRUCTURE_"
    "PRESERVING_MACRO_INTEGRATOR_MANIFEST_WP10C9D6C7C3B5C4F25FIZFJ_"
    "2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_entropy_complete_structure_preserving_macro_"
    "integrator_manifest_wp10c9d6c7c3b5c4f25fizfj.py"
)
THIS_TEST = (
    "tests/test_causal_inner_entropy_complete_structure_preserving_macro_"
    "integrator_manifest_wp10c9d6c7c3b5c4f25fizfj.py"
)
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "cc8028d5de740a6e5983acdf0afb6510f7628adc4ce122a2cb2c066538b44642"
)
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
        raise RuntimeError("thermodynamic chart atlas checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = utils._read_json(
        parent.CANONICAL_DIRECTORY / "macro_atlas_metrics.json"
    )
    if (
        summary["classification"] != parent.PASS_CLASSIFICATION
        or not summary["passed"]
        or not summary["thermodynamic_chart_conservative_macro_atlas_certified"]
        or not summary["heldout_16ms_profiles_passed"]
        or summary["online_truth_calls_per_macrostep"] != 0
        or summary["propagated_states"] != 0
        or not summary["structure_preserving_macro_integrator_manifest_authorized"]
        or summary["macro_integrator_execution_authorized"]
        or summary["complete_cycle_execution_authorized"]
        or summary["authorized_next"] != (
            "definitions_only_WP10c9d6c7c3b5c4f25fizfj_"
            "entropy_complete_structure_preserving_macro_integrator_manifest"
        )
        or metrics["new_truth_operator_calls"] != 38
        or metrics["maximum_independent_JVP_relative_defect"] > 5.0e-2
        or not metrics["all_saved_and_blind_validations_passed"]
    ):
        raise RuntimeError("macro-integrator authorization changed")
    for relative, expected in utils._read_json(
        parent.CANONICAL_DIRECTORY / "provenance.json"
    )["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"certified atlas source changed: {relative}")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("macro-integrator manifest requires a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "metrics": metrics}


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "mathematical_system": {
            "online_state": "X=16_cell_exact_(M,J,E,beta_r,chi)",
            "normalized_state": "q=(X-X_anchor)/S_X",
            "inferred_chart": "z=Pq",
            "packed_output": "y=S_y*(y0+J_z*z)",
            "conservative_rate": "dX/dt=R*y",
            "normalized_affine_rate": "dq/dt=c+Bq",
            "augmented_generator": "K=[[B,c],[0,0]]",
            "state_transition": "[q_next,1]=exp(dt*K)*[q,1]",
            "integrated_output": "Gamma(dt)*[q,1] from one augmented exponential",
            "single_valued_face_fluxes": True,
            "exact_MJE_telescoping_ledger": True,
            "truth_calls_per_online_step": 0,
            "nonlinear_roots_per_online_step": 0,
        },
        "bounded_pilot": {
            "anchor": "primary_20ms_base",
            "fixed_macrostep_seconds": 1.0e-3,
            "accepted_macrosteps": 4,
            "horizon_seconds": 4.0e-3,
            "atlas_absolute_trust_coordinate_infinity": 1.5e-1,
            "pilot_reserved_trust_coordinate_infinity": 1.2e-1,
            "checkpoint_after_step": 2,
            "suffix_steps_to_replay": 2,
            "same_horizon_one_step_comparison": True,
            "one_dynamic_endpoint_truth_call": True,
            "maximum_new_truth_operator_calls": 1,
            "complete_cycle_seconds": 578880.0,
            "complete_cycle_execution_authorized": False,
        },
        "offline_endpoint_reconstruction": {
            "method": "cellwise_5x5_thermodynamic_chart_Newton",
            "initial_guess": "certified_linear_macro_chart_pullback",
            "finite_difference_step": 1.0e-5,
            "maximum_newton_corrections": 8,
            "maximum_macro_state_roundtrip_relative_defect": 1.0e-10,
            "maximum_chart_coordinate_infinity": 1.2e-1,
            "reconstruction_is_not_online": True,
        },
        "binding_gates": {
            "maximum_spectral_abscissa_per_second": 0.0,
            "maximum_semigroup_relative_defect": 1.0e-12,
            "maximum_exact_integrated_ledger_relative_defect": 1.0e-12,
            "maximum_checkpoint_roundtrip_relative_defect": 0.0,
            "suffix_replay_bitwise": True,
            "maximum_endpoint_truth_output_relative_defect_per_block": 5.0e-2,
            "maximum_endpoint_truth_macro_rate_relative_defect_per_field": 5.0e-2,
            "endpoint_truth_all_physical_gates": True,
            "state_MJE_positive": True,
            "state_radial_velocity_subluminal": True,
            "rejected_or_out_of_trust_state_never_propagated": True,
        },
        "online_cost": {
            "benchmark_macrosteps": 100000,
            "maximum_benchmark_wall_seconds": 10.0,
            "maximum_macrosteps_per_cycle": 100000,
            "minimum_average_physical_seconds_per_cycle_macrostep": 5.7888,
            "precompute_transition_once_per_patch_and_timestep": True,
        },
        "decision": {
            "pass": "authorize_definitions_only_pathwise_macro_atlas_expansion_manifest",
            "scientific_fail": "reject_local_macro_integrator_or_affine_dynamic_closure",
            "trust_exit": "stop_and_expand_offline_atlas_without_extrapolation",
        },
        "claim_boundary": {
            "bounded_macro_propagation_authorized": True,
            "pathwise_patch_expansion_authorized": False,
            "complete_cycle_execution_authorized": False,
            "reduced_slow_cycle_claim_authorized": False,
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
            "latest_source_parent_commit": _utils()._git("rev-parse", "HEAD"),
            "latest_work_package": WORK_PACKAGE,
        }
    )
    utils._write_json(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("macro-integrator manifest already exists")
    validated = _validate_parent(require_clean=True)
    utils = _utils()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(
        CANONICAL_DIRECTORY / "macro_integrator_contract.json", _contract()
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "thermodynamic_chart_macro_atlas_preserved": True,
        "exact_affine_macro_integrator_selected": True,
        "bounded_macro_propagation_authorized": True,
        "pathwise_patch_expansion_authorized": False,
        "complete_cycle_execution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utils._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_artifact": parent.ARTIFACT,
            "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256,
            "parent_hashes": validated["hashes"],
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Entropy-complete structure-preserving macro-integrator manifest",
                "",
                f"Classification: `{CLASSIFICATION}`.",
                "",
                "The certified 80D macro field is affine inside one local chart. "
                "The selected integrator therefore uses a precomputed augmented "
                "matrix exponential, with exact time-integrated face fluxes and "
                "sources for the M/J/E ledgers.",
                "",
                "The binding pilot is four 1 ms steps (4 ms total), an arbitrary-step "
                "checkpoint and two-step bitwise suffix replay, one same-horizon "
                "semigroup comparison, and one full physical truth audit at the "
                "endpoint. The pilot stops at chart coordinate 0.12, below the "
                "certified 0.15 atlas boundary.",
                "",
                "This package does not claim that one patch covers a cycle; a pass "
                "authorizes only pathwise offline patch expansion.",
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
