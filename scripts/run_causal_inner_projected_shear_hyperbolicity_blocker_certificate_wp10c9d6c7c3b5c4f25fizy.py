#!/usr/bin/env python3
"""Certify the projected-shear blocker at the saved 179.125 ms witness."""

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

import run_causal_inner_projected_shear_hyperbolicity_blocker_manifest_wp10c9d6c7c3b5c4f25fizx as manifest  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo import (  # noqa: E402
    audit_generalized_maxwell_cattaneo_source_ledger,
    audit_specialized_nonlinear_causality,
    generalized_maxwell_cattaneo_principal,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_geometry import (  # noqa: E402
    kerr_schild_column_geometry,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = manifest.AUTHORIZED_NEXT
PASS_CLASSIFICATION = (
    "one_amplitude_projected_shear_closure_rejected_full_shear_completion_selected"
)
METHOD_CLASSIFICATION = "projected_shear_complex_pair_not_derivative_stable"
PHYSICAL_CLASSIFICATION = "full_tensor_transport_envelope_also_failed"
AUTHORIZED_NEXT = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizz_"
    "eleven_field_convex_divergence_architecture_manifest"
)
ARTIFACT = (
    "causal_inner_projected_shear_hyperbolicity_blocker_certificate_"
    "wp10c9d6c7c3b5c4f25fizy"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_PROJECTED_SHEAR_"
    "HYPERBOLICITY_BLOCKER_CERTIFICATE_"
    "WP10C9D6C7C3B5C4F25FIZY_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_projected_shear_hyperbolicity_blocker_"
    "certificate_wp10c9d6c7c3b5c4f25fizy.py"
)
THIS_TEST = (
    "tests/test_causal_inner_projected_shear_hyperbolicity_blocker_"
    "certificate_wp10c9d6c7c3b5c4f25fizy.py"
)
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "4f831a95bd32d881373c98c2065976a5d4cbd4e94213cae7a1444e60d4d438dd"
)
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _utils():
    return manifest._utils()


def _validate_manifest(*, require_clean: bool) -> dict:
    utils = _utils()
    if (
        utils._sha256(manifest.CANONICAL_DIRECTORY / "SHA256SUMS.txt")
        != PARENT_CHECKSUM_MANIFEST_SHA256
    ):
        raise RuntimeError("projected-shear blocker manifest checksum changed")
    hashes = utils._validate_checksums(manifest.CANONICAL_DIRECTORY)
    summary = utils._read_json(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = utils._read_json(
        manifest.CANONICAL_DIRECTORY / "blocker_contract.json"
    )
    if (
        summary["classification"] != manifest.CLASSIFICATION
        or not summary["passed"]
        or not summary["definitions_only"]
        or not summary["seven_field_rejection_preserved"]
        or not summary["blocker_certificate_authorized"]
        or summary["complete_cycle_execution_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["authorized_next"] != WORK_PACKAGE
        or contract["selected_next_architecture_if_certified"][
            "total_local_field_count"
        ]
        != 11
    ):
        raise RuntimeError("projected-shear blocker authorization changed")
    for relative, expected in utils._read_json(
        manifest.CANONICAL_DIRECTORY / "provenance.json"
    )["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"projected-shear manifest source changed: {relative}")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("projected-shear certificate needs a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "contract": contract}


def _certificate() -> tuple[dict, dict[str, np.ndarray]]:
    began = time.perf_counter()
    contract = _validate_manifest(require_clean=False)["contract"]
    ladder = contract["independent_derivative_ladder"]
    witness = contract["saved_witness"]
    diagnostic = manifest.parent
    truth_source, _fizu = diagnostic._truth_modules()
    with np.load(manifest.PARENT_ARRAYS) as archive:
        charts = np.asarray(archive["probe_2_primitive_charts"])
    context_start = time.perf_counter()
    context, _profile, _initial = truth_source.fixed_q_implementation._primary_setup()
    context_seconds = time.perf_counter() - context_start
    base_rows = []
    base_principals = []
    for cell, radius in enumerate(np.asarray(context.grid.centers, dtype=float)):
        principal = generalized_maxwell_cattaneo_principal(
            kerr_schild_column_geometry(
                float(radius), context.grid.gravitational_radius
            ),
            charts[cell],
            proper_vertical_frequency=float(
                context.vertical_frequency.frequency(float(radius))
            ),
            alpha=float(context.alpha),
            stress_factor=float(context.stress_factor),
        )
        base_rows.append(
            {
                "cell": cell,
                "radius_cm": float(radius),
                "maximum_imaginary_speed_over_c": principal.maximum_imaginary_speed_over_c,
            }
        )
        base_principals.append(principal)
    maximum_row = max(
        base_rows, key=lambda item: item["maximum_imaginary_speed_over_c"]
    )
    cell = maximum_row["cell"]
    radius = maximum_row["radius_cm"]
    geometry = kerr_schild_column_geometry(
        radius, context.grid.gravitational_radius
    )
    rows = []
    eigenvalues = []
    temporal_matrices = []
    radial_matrices = []
    principals = []
    for factor in ladder["derivative_step_factors"]:
        principal = generalized_maxwell_cattaneo_principal(
            geometry,
            charts[cell],
            proper_vertical_frequency=float(
                context.vertical_frequency.frequency(radius)
            ),
            alpha=float(context.alpha),
            stress_factor=float(context.stress_factor),
            derivative_step_factor=float(factor),
        )
        principals.append(principal)
        rows.append(
            {
                "derivative_step_factor": float(factor),
                "maximum_imaginary_speed_over_c": principal.maximum_imaginary_speed_over_c,
                "maximum_eigenpair_relative_defect": principal.maximum_eigenpair_relative_defect,
                "eigenvector_condition_number": principal.eigenvector_condition_number,
                "scaled_temporal_condition_number": principal.scaled_temporal_condition_number,
            }
        )
        eigenvalues.append(principal.eigenvalues_over_c)
        temporal_matrices.append(principal.temporal_matrix)
        radial_matrices.append(principal.radial_matrix)
    imaginary = np.asarray(
        [item["maximum_imaginary_speed_over_c"] for item in rows]
    )
    relative_spread = float(
        (np.max(imaginary) - np.min(imaginary))
        / max(float(np.mean(imaginary)), np.finfo(float).tiny)
    )
    representative = principals[
        list(ladder["derivative_step_factors"]).index(1.0)
    ]
    causality = audit_specialized_nonlinear_causality(
        representative.local_state
    )
    ledger = audit_generalized_maxwell_cattaneo_source_ledger(
        representative.local_state, alpha=float(context.alpha)
    )
    derivative_stable = bool(
        cell == witness["expected_first_failing_cell"]
        and np.isclose(radius, witness["expected_radius_cm"], rtol=0.0, atol=1.0e-6)
        and np.min(imaginary) >= ladder["minimum_imaginary_speed_over_c"]
        and relative_spread
        <= ladder["maximum_imaginary_speed_relative_spread"]
        and max(item["maximum_eigenpair_relative_defect"] for item in rows)
        <= ladder["maximum_eigenpair_relative_defect"]
        and max(item["eigenvector_condition_number"] for item in rows)
        <= ladder["maximum_eigenvector_condition_number"]
    )
    full_tensor_screen = bool(
        causality.minimum_margin > 0.0
        and ledger.minimum_entropy_production_rate >= 0.0
        and ledger.vertical_total_energy_relative_defect
        <= contract["physical_discriminator"][
            "vertical_energy_ledger_relative_defect_maximum"
        ]
        and ledger.vertical_reversible_exchange_relative_defect
        <= contract["physical_discriminator"][
            "vertical_energy_ledger_relative_defect_maximum"
        ]
    )
    passed = derivative_stable and full_tensor_screen
    classification = (
        PASS_CLASSIFICATION
        if passed
        else METHOD_CLASSIFICATION
        if not derivative_stable
        else PHYSICAL_CLASSIFICATION
    )
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "seven_field_parent_rejection_preserved": True,
        "witness_cell": cell,
        "witness_radius_cm": radius,
        "witness_chart7": charts[cell].tolist(),
        "derivative_rows": rows,
        "imaginary_speed_relative_spread": relative_spread,
        "derivative_stable_complex_pair": derivative_stable,
        "full_tensor_causality_minimum_margin": causality.minimum_margin,
        "full_tensor_shear_signal_ratio": causality.shear_signal_ratio,
        "full_tensor_sound_speed_squared_over_c2": causality.sound_speed_squared_over_c2,
        "full_tensor_E_plus_Lambda_minimum": causality.E_plus_Lambda_minimum,
        "full_tensor_screen_passed": full_tensor_screen,
        "minimum_entropy_production_rate": ledger.minimum_entropy_production_rate,
        "vertical_total_energy_relative_defect": ledger.vertical_total_energy_relative_defect,
        "vertical_reversible_exchange_relative_defect": ledger.vertical_reversible_exchange_relative_defect,
        "context_construction_wall_seconds": context_seconds,
        "certificate_wall_seconds": time.perf_counter() - began,
        "selected_local_field_count": 11 if passed else None,
        "one_amplitude_projected_shear_closure_authorized": False,
        "eleven_field_physical_closure_certified": False,
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT if passed else None,
    }
    arrays = {
        "witness_chart7": charts[cell],
        "derivative_step_factors": np.asarray(
            ladder["derivative_step_factors"], dtype=float
        ),
        "eigenvalues7_by_derivative_step": np.asarray(eigenvalues),
        "temporal_matrices7x7": np.asarray(temporal_matrices),
        "radial_matrices7x7": np.asarray(radial_matrices),
        "full_tensor_causality_inequality_margins": np.asarray(
            causality.inequality_margins
        ),
    }
    return metrics, arrays


def _update_catalog(summary: dict) -> None:
    utils = _utils()
    with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    status = "SUPPORTED" if summary["passed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": utils._sha256(path),
                    "scientific_status": status,
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
        raise RuntimeError("projected-shear blocker certificate already exists")
    validated = _validate_manifest(require_clean=True)
    utils = _utils()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "certificate_metrics.json", metrics)
    np.savez_compressed(CANONICAL_DIRECTORY / "certificate_arrays.npz", **arrays)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": bool(metrics["passed"]),
        "seven_field_rejection_preserved": True,
        "one_amplitude_projected_shear_closure_rejected": bool(metrics["passed"]),
        "full_five_component_shear_completion_selected": bool(metrics["passed"]),
        "eleven_field_physical_closure_certified": False,
        "complete_cycle_execution_authorized": False,
        "authorized_next": metrics["authorized_next"],
    }
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utils._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_artifact": manifest.ARTIFACT,
            "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256,
            "parent_hashes": validated["hashes"],
            "witness_arrays_sha256": utils._sha256(manifest.PARENT_ARRAYS),
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Projected-shear hyperbolicity blocker certificate",
                "",
                f"Classification: `{metrics['classification']}`.",
                "",
                f"At cell `{metrics['witness_cell']}` the reduced complex-pair imaginary speed is derivative-stable with relative spread `{metrics['imaginary_speed_relative_spread']:.6e}`. The independent full-tensor causality screen has positive minimum margin `{metrics['full_tensor_causality_minimum_margin']:.6e}`.",
                "",
                "The one-amplitude projected shear closure is therefore rejected. The conserved four-field backbone and finite-inertia vertical pair are retained; the selected next architecture promotes all five rest-frame symmetric-tracefree shear amplitudes and derives every principal coupling from one convex generating potential.",
                "",
                "The eleven-field physical closure is not yet certified and complete-cycle execution remains unauthorized.",
                "",
                f"Authorized next: `{metrics['authorized_next']}`.",
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
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("choose --run")
    metrics, arrays = _certificate()
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    summary = _canonicalize(metrics, arrays)
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
