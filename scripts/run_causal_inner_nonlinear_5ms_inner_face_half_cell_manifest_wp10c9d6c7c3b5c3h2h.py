#!/usr/bin/env python3
"""Freeze the operator-neutral 5 ms inner-face/half-cell audit."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import sys

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_nonlinear_5ms_tier_i_localization_wp10c9d6c7c3b5c3h2g1 as h2g1  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c3h2h"
ANALYZED_BASE_COMMIT = "f5d743acf516a9f491c978efd5599b8f763d6500"
ANALYZED_BASE_PARENT = "242966f54dada9fa0bbde67dd5d56cc4b9d7b488"
ANALYZED_BASE_TREE = "2c9087f21d42f55f9ea40f660139f99f9c9942e3"

ARTIFACT = (
    "causal_inner_nonlinear_5ms_inner_face_half_cell_manifest_"
    "wp10c9d6c7c3b5c3h2h"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_5ms_inner_face_half_cell_manifest_"
    "wp10c9d6c7c3b5c3h2h.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_5ms_inner_face_half_cell_manifest_"
    "wp10c9d6c7c3b5c3h2h.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_5MS_INNER_FACE_"
    "HALF_CELL_MANIFEST_WP10C9D6C7C3B5C3H2H_2026-08-08.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "audit_manifest.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

PRIMITIVE_NAMES = (
    "log_surface_density",
    "radial_velocity_over_c",
    "azimuthal_velocity_over_c",
    "log_temperature",
    "specific_stress",
)
CONSERVATIVE_CHANNELS = ("mass", "angular_momentum", "killing_energy")
COMMON_PREFIX_COARSE_FACE_INDICES = (1, 2, 4, 8, 12, 16, 24, 32, 40, 48)
COMMON_PREFIX_FACE_MULTIPLIERS = (1, 2, 4)
AUDIT_GATES = {
    "maximum_identity_closure_defect": 1.0e-10,
    "maximum_inner_flux_field_path_closure_defect": 1.0e-9,
    "minimum_error_dominance_fraction": 0.70,
    "minimum_error_alignment": 0.90,
    "minimum_spatial_order": 0.75,
    "maximum_fine_normalized_difference": 0.05,
    "minimum_refinement_error_cosine": 0.90,
    "minimum_consecutive_recovery_faces": 2,
    "maximum_temporal_uncertainty_fraction": 0.10,
}


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_identity() -> dict[str, str]:
    return {
        path: _sha256(ROOT / path)
        for path in (THIS_RUNNER, THIS_TEST)
        if (ROOT / path).exists()
    }


def _validate_parent() -> dict:
    parent = _read_json(h2g1.SUMMARY_PATH)
    if (
        not parent["passed"]
        or parent["classification"]
        != "five_ms_Tier_I_failure_localized_to_inner_face_response_"
        "half_cell_audit_manifest_authorized"
        or parent["authorized_next"]
        != "WP10c9d6c7c3b5c3h2h_inner_face_half_cell_audit_manifest"
        or parent["fourth_duration_rung_manifest_authorized"]
        or parent["fixed_q_micro_solver_authorized"]
        or parent["reduced_slow_evolution_authorized"]
    ):
        raise RuntimeError("h2h authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("h2h analyzed identity changed")
    return parent


def _manifest() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            "inner_face_half_cell_audit_manifest_frozen_operator_neutral_"
            "control_volume_diagnostics_authorized"
        ),
        "definitions_only": True,
        "propagation_executed": False,
        "operator_changed": False,
        "production_defaults_changed": False,
        "parent_rejection_preserved": True,
        "primitive_names": PRIMITIVE_NAMES,
        "conservative_channels": CONSERVATIVE_CHANNELS,
        "common_prefix_coarse_face_indices": COMMON_PREFIX_COARSE_FACE_INDICES,
        "common_prefix_face_multipliers": COMMON_PREFIX_FACE_MULTIPLIERS,
        "audit_gates": AUDIT_GATES,
        "diagnostics": {
            "complete_BDF_prefix_balance": {
                "identity": (
                    "temporal_storage_plus_outer_common_face_flux_minus_"
                    "inner_face_flux_plus_distributed_principal_and_lower_"
                    "sources_equals_zero"
                ),
                "use_actual_layout_owned_BDF_histories": True,
                "compare_base_to_full_generic_anchor": True,
                "evaluate_only_committed_accepted_steps": True,
                "blocks": (
                    "minus_inner_face_flux",
                    "outer_common_face_flux",
                    "mapped_temporal_storage",
                    "responsive_height_temporal_storage",
                    "shear_principal",
                    "height_principal",
                    "local_stress_relaxation",
                    "geometry",
                    "cooling",
                    "stream",
                    "lower_height_work",
                ),
            },
            "inner_flux_primitive_path": {
                "definition": (
                    "integrate_the_analytic_inner_flux_jacobian_along_the_"
                    "straight_base_to_anchor_first_cell_path_and_record_each_"
                    "primitive_column_contribution"
                ),
                "quadrature_order": 8,
                "no_fitted_coefficients": True,
            },
            "common_face_recovery": {
                "requires_every_conservative_channel": True,
                "requires_instantaneous_response_contract": True,
                "requires_consecutive_faces": AUDIT_GATES[
                    "minimum_consecutive_recovery_faces"
                ],
            },
            "time_scope": {
                "use_common_accepted_targets_with_complete_BDF_history": True,
                "retain_late_window_as_binding": True,
                "no_time_interpolation": True,
            },
        },
        "decision_tree": (
            "two_consecutive_common_faces_recover__authorize_conservative_extraction_surface_manifest",
            "stable_inner_flux_and_one_primitive_column_dominate__authorize_one_outgoing_half_cell_candidate_manifest",
            "stable_temporal_storage_block_dominates__authorize_space_storage_consistency_manifest",
            "stable_principal_or_lower_source_block_dominates__authorize_one_targeted_source_consistency_manifest",
            "otherwise__authorize_monolithic_near_horizon_space_storage_redesign_manifest_only",
        ),
        "hard_stops": (
            "do_not_propagate_new_state",
            "do_not_change_excision_trace_flux_path_storage_or_source",
            "do_not_tune_a_candidate_before_the_audit_selects_one_branch",
            "do_not_relax_spatial_temporal_or_ledger_gates",
            "do_not_infer_physical_instability",
            "do_not_start_fourth_duration_rung_fixed_Q_or_reduction",
        ),
        "authorized_next": (
            "WP10c9d6c7c3b5c3h2h1_operator_neutral_inner_face_half_cell_audit"
        ),
    }


def _update_catalog(summary: dict) -> None:
    rows = []
    if CANONICAL_MANIFEST.exists():
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
                    "sha256": _sha256(path),
                    "scientific_status": "CERTIFIED",
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case", "path", "bytes", "sha256", "scientific_status"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read_json(CANONICAL_SUMMARY)
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
            "latest_source_parent_commit": ANALYZED_BASE_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def main() -> int:
    parent = _validate_parent()
    manifest = _manifest()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": manifest["classification"],
        "passed": True,
        "definitions_only": True,
        "parent_classification_preserved": parent["classification"],
        "operator_neutral_half_cell_audit_authorized": True,
        "new_propagation_authorized": False,
        "fourth_duration_rung_manifest_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": manifest["authorized_next"],
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "common_prefix_coarse_face_indices": COMMON_PREFIX_COARSE_FACE_INDICES,
            "primitive_names": PRIMITIVE_NAMES,
            "conservative_channels": CONSERVATIVE_CHANNELS,
            "audit_gates": AUDIT_GATES,
        },
    )
    _write_json(MANIFEST_PATH, manifest)
    _write_json(SUMMARY_PATH, summary)
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "source_parent_commit": ANALYZED_BASE_COMMIT,
            "scientific_status": "CERTIFIED",
            "working_head": _git_value("rev-parse", "HEAD"),
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "parent_summary_sha256": _sha256(h2g1.SUMMARY_PATH),
            "implementation_source_hashes": _source_identity(),
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "command": f"PYTHONPATH=src:scripts python {THIS_RUNNER}",
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Nonlinear 5 ms inner-face/half-cell audit manifest WP10c9d6c7c3b5c3h2h",
                "",
                "## Classification",
                "",
                f"`{manifest['classification']}`",
                "",
                "This definitions-only package preserves the rejected 5 ms spatial certificate and the h2g1 inner-face localization. It freezes complete accepted-BDF control-volume balances on ten nested common physical faces and a path-integrated analytic decomposition of the outgoing inner flux into the five primitive coordinates.",
                "",
                "The audit may select one extraction-surface, half-cell, space-storage, source-consistency, or broader near-horizon redesign manifest. It may not change or propagate the operator. Later duration, fixed-Q experiments, and reduced slow evolution remain blocked.",
                "",
            )
        ),
        encoding="utf-8",
    )
    names = ("audit_manifest.json", "config.json", "provenance.json", "summary.json")
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
