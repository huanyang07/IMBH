#!/usr/bin/env python3
"""Freeze the retained guard-buffer overlap architecture.

Definitions only: no state is advanced and no production operator changes.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_recovered_coupling_existing_state_ledger_preflight_wp10c9d6c7c3b5c4f9 as c4f9  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f10"
ARTIFACT = "causal_inner_retained_guard_buffer_micro_macro_manifest_wp10c9d6c7c3b5c4f10"
THIS_RUNNER = "scripts/run_causal_inner_retained_guard_buffer_micro_macro_manifest_wp10c9d6c7c3b5c4f10.py"
THIS_TEST = "tests/test_causal_inner_retained_guard_buffer_micro_macro_manifest_wp10c9d6c7c3b5c4f10.py"
REPORT_RELATIVE = "docs/reports/current/CODEX_CAUSAL_INNER_RETAINED_GUARD_BUFFER_MICRO_MACRO_MANIFEST_WP10C9D6C7C3B5C4F10_2026-08-13.md"
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "overlap_manifest.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _manifest():
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": "retained_guard_buffer_overlap_architecture_frozen_analysis_only_preflight_authorized",
        "definitions_only": True,
        "new_trajectory": False,
        "production_operator_changed": False,
        "physical_partition": {
            "inner_micro_core_parent_cells": [0, 36],
            "macro_exterior_parent_cells": [36, 64],
            "shared_exchange_parent_face": 36,
            "binding_exchange_observable": "shared_face36_M_J_E_flux",
            "raw_face48_flux_forbidden_as_slow_exchange": True,
        },
        "numerical_overlap": {
            "micro_solver_guard_parent_cells": [36, 48],
            "original_numerical_coupling_parent_face": 48,
            "macro_owned_overlap_representation": "exact_parent_sums_of_mapped_storage_integrals_plus_responsive_height_BDF_history",
            "primitive_recovery_from_restricted_storage_requires_separate_certification": True,
            "micro_guard_is_duplicate_numerical_state": True,
            "micro_guard_storage_and_sources_counted_in_physical_inventory": False,
            "macro_overlap_storage_and_sources_counted_exactly_once": True,
            "micro_guard_must_remain_evolved_for_boundary_insulation": True,
        },
        "synchronization_contract": {
            "macro_to_micro": "storage_space_parent_increment_distributed_by_child_measure_while_preserving_zero_mean_fine_complement",
            "micro_to_macro": "conservative_restriction_plus_explicit_reaction_increment",
            "reaction_M_J_E_must_be_ledgered": True,
            "responsive_height_history_must_be_transferred": True,
            "primitive_only_overwrite_forbidden": True,
            "Euclidean_projection_forbidden": True,
            "double_counted_storage_forbidden": True,
        },
        "slow_state_candidates": {
            "Q3": ["macro_exterior_M", "macro_exterior_J", "macro_exterior_E"],
            "Q4": ["macro_exterior_M", "macro_exterior_J", "macro_exterior_E", "shell_thermal_height_storage"],
            "guard_auxiliary_state": [
                "restricted_guard_mapped_M_J_E",
                "restricted_guard_responsive_height_BDF_history",
                "previous_guard_timestep",
            ],
            "guard_auxiliary_is_not_assumed_to_be_a_slow_coordinate": True,
        },
        "existing_state_overlap_preflight": {
            "new_trajectory": False,
            "verify_conservative_restriction_and_physical_inventory_partition": True,
            "verify_face36_flux_and_guard_storage_three_grid_convergence": True,
            "verify_baseline_plus_response_reconstruction": True,
            "verify_macro_owned_overlap_state_convergence": True,
            "verify_no_face48_substitution": True,
            "verify_responsive_height_history_transfer": True,
            "projected_memory_propagation": False,
        },
        "prospective_gates": {
            "maximum_conservative_restriction_defect": 1.0e-12,
            "maximum_physical_inventory_partition_defect": 1.0e-12,
            "maximum_overlap_sync_roundtrip_defect": 1.0e-10,
            "minimum_spatial_RMS_order": 0.75,
            "minimum_spatial_error_direction_cosine": 0.90,
            "maximum_baseline_plus_response_scaled_defect": 1.0e-10,
            "maximum_guard_reaction_fraction_for_projection_free_memory_screen": 0.10,
        },
        "decision": {
            "existing_state_overlap_contract_passes": "authorize_analysis_only_face36_augmented_projected_memory_screen_manifest",
            "restriction_or_inventory_fails": "repair_overlap_maps_only",
            "guard_reaction_is_large": "retain_overlap_microburst_do_not_project_memory_without_reaction",
            "absolute_overlap_state_fails": "absolute_slow_closure_remains_blocked",
        },
        "cost_contract": {
            "reuse_committed_5_to_20_ms_trajectories_first": True,
            "block_tangent_directions": "16_to_32_after_overlap_preflight_only",
            "new_nonlinear_anchor_limit_before_memory_dimension_known": 0,
            "full_fine_or_50ms_trajectory_authorized": False,
        },
        "hard_stops": [
            "do_not_relabel_face36_as_face48_or_horizon_flux",
            "do_not_drop_or_double_count_guard_storage_or_sources",
            "do_not_overwrite_primitive_guard_state_without_DAE_history_transfer",
            "do_not_run_fixed_Q_or_reduced_evolution",
            "do_not_start_50ms_propagation",
        ],
        "response_certificate_preserved": True,
        "absolute_closure_fit_authorized": False,
        "memory_propagation_authorized": False,
        "fixed_Q_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": "WP10c9d6c7c3b5c4f11_analysis_only_existing_state_overlap_consistency_preflight",
    }


def _catalog(summary):
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": _sha(path), "scientific_status": "PROSPECTIVE"})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": summary["classification"], "passed": True}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "latest_work_package": WORK_PACKAGE})
    _write(CANONICAL_SUMMARY, catalog)


def main():
    parent = _read(c4f9.SUMMARY_PATH)
    if (
        not parent["passed"]
        or not parent["overlap_state_spatially_convergent"]
        or parent["direct_face48_absolute_export_spatially_convergent"]
        or parent["authorized_next"] != "WP10c9d6c7c3b5c4f10_definitions_only_retained_guard_buffer_micro_macro_manifest"
    ):
        raise RuntimeError("c4f10 authorization changed")
    manifest = _manifest()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": manifest["classification"],
        "passed": True,
        "definitions_only": True,
        "overlap_physical_ownership_frozen": True,
        "guard_double_count_forbidden": True,
        "face48_absolute_export_rejection_preserved": True,
        "new_trajectory_authorized": False,
        "memory_propagation_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "fifty_ms_propagation_authorized": False,
        "physical_failure_detected": False,
        "authorized_next": manifest["authorized_next"],
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write(CONFIG_PATH, {"schema_version": SCHEMA_VERSION, "shared_face": 36, "guard_parent_cells": [36, 48], "macro_parent_cells": [36, 64]})
    _write(MANIFEST_PATH, manifest)
    _write(SUMMARY_PATH, summary)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# Retained guard-buffer micro-macro manifest\n\n"
        f"Classification: `{summary['classification']}`.\n\n"
        "The physical inner/macro exchange is the shared M/J/E flux at parent face 36. The macro exterior owns cells 36:64. The inner micro-solver may continue through cells 36:48 only as a duplicate numerical guard; those duplicate storage and source terms are not counted in the physical inventory.\n\n"
        "Conservative restriction/prolongation and every synchronization reaction must be ledgered, including responsive-height BDF history. Face 36 is not relabelled face 48 or a horizon flux.\n\n"
        "Only an existing-state overlap consistency preflight is authorized.\n",
        encoding="utf-8",
    )
    _write(PROVENANCE_PATH, {"schema_version": SCHEMA_VERSION, "parent_summary_sha256": _sha(c4f9.SUMMARY_PATH), "source_hashes": {THIS_RUNNER: _sha(ROOT / THIS_RUNNER), THIS_TEST: _sha(ROOT / THIS_TEST) if (ROOT / THIS_TEST).exists() else None}})
    files = (CONFIG_PATH, MANIFEST_PATH, SUMMARY_PATH, PROVENANCE_PATH)
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{_sha(path)}  {path.name}\n" for path in files), encoding="utf-8")
    _catalog(summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
