#!/usr/bin/env python3
"""Freeze the decisive projected-shear hyperbolicity blocker audit."""

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

import run_causal_inner_entropy_complete_hyperbolicity_boundary_refinement_diagnostic_wp10c9d6c7c3b5c4f25fizw as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fizx_"
    "projected_shear_hyperbolicity_blocker_manifest"
)
CLASSIFICATION = "projected_shear_hyperbolicity_blocker_manifest_frozen"
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizy_"
    "projected_shear_hyperbolicity_blocker_certificate"
)
ARTIFACT = (
    "causal_inner_projected_shear_hyperbolicity_blocker_manifest_"
    "wp10c9d6c7c3b5c4f25fizx"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_PROJECTED_SHEAR_"
    "HYPERBOLICITY_BLOCKER_MANIFEST_"
    "WP10C9D6C7C3B5C4F25FIZX_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_projected_shear_hyperbolicity_blocker_"
    "manifest_wp10c9d6c7c3b5c4f25fizx.py"
)
THIS_TEST = (
    "tests/test_causal_inner_projected_shear_hyperbolicity_blocker_"
    "manifest_wp10c9d6c7c3b5c4f25fizx.py"
)
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "3c43d50572ae931a18634bbeb90d2beea3660b309d78a9d1c996b353900f2864"
)
PARENT_ARRAYS = parent.CANONICAL_DIRECTORY / "diagnostic_arrays.npz"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _utils():
    return parent._utils()


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "saved_witness": {
            "source_probe_timestep_seconds": 1.25e-4,
            "expected_first_failing_cell": 6,
            "expected_radius_cm": 3035196434.9786267,
            "nonpropagating": True,
        },
        "independent_derivative_ladder": {
            "derivative_step_factors": (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
            "minimum_imaginary_speed_over_c": 1.0e-5,
            "maximum_imaginary_speed_relative_spread": 0.01,
            "maximum_eigenpair_relative_defect": 1.0e-12,
            "maximum_eigenvector_condition_number": 1.0e5,
        },
        "physical_discriminator": {
            "full_tensor_nonlinear_causality_minimum_margin_strictly_positive": True,
            "source_entropy_production_nonnegative": True,
            "vertical_energy_ledger_relative_defect_maximum": 1.0e-12,
            "reduced_projected_principal_is_binding_for_current_model": True,
        },
        "decision": {
            "stable_complex_pair_and_full_tensor_screen_positive": (
                "reject one-amplitude projected shear closure; select a full five-component "
                "rest-frame shear convex/divergence-type completion"
            ),
            "complex_pair_not_stable": "method repair only",
            "full_tensor_screen_fails": "physical transport envelope repair required",
        },
        "selected_next_architecture_if_certified": {
            "conserved_backbone": "mass, radial momentum, angular momentum, total energy",
            "vertical_pair": "finite-inertia height and vertical momentum",
            "dissipative_extension": "five independent rest-frame symmetric tracefree shear amplitudes",
            "total_local_field_count": 11,
            "binding_structure": (
                "one common convex generating potential; symmetric temporal Hessian; "
                "symmetric radial Hessian; dissipative source negative in entropy variables"
            ),
            "project_to_one_Rphi_amplitude": False,
        },
        "claim_boundary": {
            "one_amplitude_seven_field_trajectory_blocked": True,
            "eleven_field_physical_closure_certified": False,
            "complete_cycle_execution_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
        "authorized_next": AUTHORIZED_NEXT,
    }


def _validate_parent(*, require_clean: bool) -> dict:
    utils = _utils()
    if (
        utils._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt")
        != PARENT_CHECKSUM_MANIFEST_SHA256
    ):
        raise RuntimeError("refined hyperbolicity diagnostic checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = utils._read_json(
        parent.CANONICAL_DIRECTORY / "diagnostic_metrics.json"
    )
    if (
        summary["classification"] != parent.BOUNDARY_CLASSIFICATION
        or summary["passed"]
        or summary["authorized_next"] is not None
        or not summary["coarse_failure_reproduced"]
        or summary["both_refined_truth_probes_passed"]
        or not summary["all_probes_nonpropagating"]
        or not metrics["accepted_endpoint_hyperbolic"]
        or metrics["largest_scanned_hyperbolic_timestep_seconds"] != 1.875e-4
        or metrics["smallest_scanned_nonreal_timestep_seconds"] != 2.1875e-4
    ):
        raise RuntimeError("refined hyperbolicity diagnostic changed")
    for relative, expected in utils._read_json(
        parent.CANONICAL_DIRECTORY / "provenance.json"
    )["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"refined diagnostic source changed: {relative}")
    with np.load(PARENT_ARRAYS) as archive:
        if (
            "probe_2_primitive_charts" not in archive.files
            or archive["probe_2_primitive_charts"].shape[1] != 7
        ):
            raise RuntimeError("saved refined witness changed")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("blocker manifest needs a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "metrics": metrics}


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
                    "scientific_status": "SUPPORTED",
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
        "passed": True,
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
        raise RuntimeError("projected-shear blocker manifest already exists")
    validated = _validate_parent(require_clean=True)
    utils = _utils()
    contract = _contract()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "blocker_contract.json", contract)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "seven_field_rejection_preserved": True,
        "blocker_certificate_authorized": True,
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
            "parent_arrays_sha256": utils._sha256(PARENT_ARRAYS),
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Projected-shear hyperbolicity blocker manifest",
                "",
                f"Classification: `{CLASSIFICATION}`.",
                "",
                "This definitions-only package preserves the 179 ms stopped trajectory and freezes an independent derivative-step ladder at the nonpropagated 0.125 ms witness.",
                "",
                "Only if the reduced complex pair is stable across the ladder while the full-tensor causality and entropy screens remain positive may the one-amplitude projection be rejected in favor of an eleven-field full-shear convex extension.",
                "",
                "No new trajectory or complete-cycle execution is authorized.",
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
