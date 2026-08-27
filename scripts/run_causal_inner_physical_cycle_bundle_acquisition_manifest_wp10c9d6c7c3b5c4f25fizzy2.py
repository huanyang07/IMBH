#!/usr/bin/env python3
"""Freeze the external physical-cycle bundle acquisition contract."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_complete_cycle_preexecution_readiness_certificate_wp10c9d6c7c3b5c4f25fizzy1 as parent  # noqa: E402


WORK_PACKAGE = parent.AUTHORIZED_NEXT
CLASSIFICATION = (
    "physical_cycle_bundle_v2_acquisition_manifest_frozen_"
    "external_model_declaration_required"
)
AUTHORIZED_NEXT = (
    "external_input_WP10c9d6c7c3b5c4f25fizzy3_"
    "physical_model_declaration_and_prospective_split_lock"
)
ARTIFACT = (
    "causal_inner_physical_cycle_bundle_acquisition_manifest_"
    "wp10c9d6c7c3b5c4f25fizzy2"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_PHYSICAL_CYCLE_BUNDLE_"
    "ACQUISITION_MANIFEST_WP10C9D6C7C3B5C4F25FIZZY2_2026-08-27.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_physical_cycle_bundle_acquisition_manifest_"
    "wp10c9d6c7c3b5c4f25fizzy2.py"
)
THIS_TEST = (
    "tests/test_causal_inner_physical_cycle_bundle_acquisition_manifest_"
    "wp10c9d6c7c3b5c4f25fizzy2.py"
)
PARENT_SHA256 = "58545f60a020a3f7cff7c1d84d907227d9c81aa2d6ac4177e174182464558a91"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _u():
    return parent._u()


def _validate_parent(*, require_clean: bool = False):
    utility = _u()
    checksum = utility._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt")
    if checksum != PARENT_SHA256:
        raise RuntimeError("physical-readiness certificate changed")
    hashes = utility._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utility._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    readiness = utility._read_json(parent.CANONICAL_DIRECTORY / "readiness_audit.json")
    inventory = utility._read_json(
        parent.CANONICAL_DIRECTORY / "repository_payload_inventory.json"
    )
    if (
        not summary["passed"]
        or not summary["inventory_audit_passed"]
        or summary["directly_reusable_binding_cycle_field_count"] != 0
        or summary["physical_cycle_bundle_v2_acquired"]
        or summary["complete_cycle_preexecution_readiness_passed"]
        or summary["complete_cycle_execution_authorized"]
        or summary["complete_cycle_steps"] != 0
        or summary["authorized_next"] != WORK_PACKAGE
        or readiness["complete_cycle_preexecution_readiness_passed"]
        or not readiness["binding_interpretation"][
            "external_physical_acquisition_block_selected"
        ]
        or inventory["complete_bundle_directory_count"] != 0
        or inventory["physical_metadata_record_count"] != 0
        or inventory["complete_group_file_count"] != 0
    ):
        raise RuntimeError("physical-readiness negative finding changed")
    if require_clean and utility._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("physical acquisition manifest needs a clean tracked tree")
    return hashes, summary, readiness, inventory


def _contract(readiness: dict) -> dict:
    heldout_gates = {
        "maximum_branch_state_relative_defect": 2.0e-2,
        "maximum_branch_rate_relative_defect": 5.0e-2,
        "maximum_port_action_relative_defect": 5.0e-2,
        "maximum_event_time_relative_defect": 2.0e-2,
        "maximum_event_post_state_relative_defect": 5.0e-2,
        "maximum_event_ledger_relative_defect": 2.0e-2,
        "maximum_sequence_endpoint_relative_defect": 5.0e-2,
        "maximum_sequence_ledger_relative_defect": 2.0e-2,
        "discrete_modes_and_event_order_exact": True,
    }
    return {
        "schema_version": 2,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "supersedes": {
            "artifact": (
                "causal_inner_cycle_wide_physical_driver_boundary_loading_and_"
                "event_truth_acquisition_manifest_wp10c9d6c7c3b5c4f25fizzt"
            ),
            "reason": (
                "the final five-coordinate reduced-hybrid production kernel now "
                "requires explicit atlas topology, finite phase-advancing resets, "
                "independent truth payloads, a locked initial state, and a production "
                "runtime benchmark in addition to the original schema"
            ),
        },
        "scientific_authority_boundary": {
            "repository_may_define_numerical_schema_and_validation": True,
            "repository_may_select_unprovided_physical_forcing_or_modes": False,
            "external_scientific_authority_required": True,
            "forbidden_defaults": [
                "do not treat 578880 seconds as a measured period or phase law",
                "do not extend the 0.016-second no-tide/no-wind prefix around a cycle",
                "do not promote the static exterior candidate into physical outer loading",
                "do not infer hot/cooling/recovery modes or resets from unobserved events",
                "do not label any deterministic or synthetic fixture as physical truth",
            ],
        },
        "external_physical_model_declaration": {
            "identity_and_provenance": [
                "physical_model_id and semantic version",
                "source-code repository, exact commit, dependency lock, and license",
                "citable sources for every external constitutive or forcing law",
                "CGS units, coordinate conventions, sign conventions, and parameter table",
                "declared uncertainty ranges and which parameters are calibrated",
            ],
            "orbital_driver": [
                "ephemeris phi(t), strictly positive dphi/dt, and physical period",
                "impact geometry, orientation, and phase-zero convention",
                "derivation and provenance independent of the runtime target",
            ],
            "distributed_and_boundary_physics": [
                "1232-state distributed forcing over (Q,phase,mode)",
                "four retained-ledger source rates with mass/angular-momentum/energy bookkeeping",
                "physical exterior state or incoming-flux law",
                "all eleven incoming outer entropy-characteristic amplitudes",
                "inner excision ledger and any tide, torque, wind, heating, or cooling terms",
            ],
            "constitutive_modes_and_events": [
                "cold, hot, cooling, recovery, or other physically chosen mode definitions",
                "oriented transition guards, hysteresis, and destination-mode policy",
                "entry-to-exit physical event solver and dense guard localization",
                "event duration, phase advance, four-ledger impulse, and constitutive jump",
            ],
            "initial_condition": [
                "physical 1232-state profile and retained invariants",
                "unwrapped phase, discrete mode, next transition inventory, and epoch",
                "source artifact hashes and parameter realization",
            ],
        },
        "canonical_physical_cycle_bundle_v2": {
            "metadata.json": {
                "required": [
                    "schema_version=2",
                    "physical_model_id",
                    "physical_model_complete=true",
                    "synthetic_fixture=false",
                    "unit_system=cgs",
                    "source_citations",
                    "source_code_commit",
                    "parameter_hash",
                    "split_frozen_before_fit=true",
                    "independent_spatial_holdout_complete=true only after validation",
                    "independent_sequence_or_cycle_holdout_complete=true only after validation",
                    "heldout_physical_validation_complete=true only after validation",
                    "physical_payload_hashes_complete=true only after final checksums close",
                    "physical_bundle_sha256 only after canonicalization",
                ]
            },
            "driver.npz": {
                "phase_nodes": "(N_phi,) exact 0..2*pi periodic grid",
                "phase_rate_per_second": "(N_phi,) positive physical ephemeris",
                "retained_invariant_nodes4": "(N_q,4)",
                "mode_labels": "(N_mode,) unique physical labels",
                "slow_forcing1232_per_second": "(N_phi,N_q,N_mode,1232)",
                "distributed_source_ledger_rate4": "(N_phi,N_q,N_mode,4)",
                "boundary_ledger_rate4": "(N_phi,N_q,N_mode,4)",
                "outer_incoming_characteristics11": "(N_phi,N_q,N_mode,11)",
            },
            "branch_train.npz": {
                "anchor_states1232": "(N_branch,1232)",
                "anchor_phase": "(N_branch,)",
                "anchor_invariants4": "(N_branch,4)",
                "anchor_mode_index": "(N_branch,)",
                "radial_matrices112x11x11": "(N_branch,112,11,11)",
                "source_matrices112x11x11": "(N_branch,112,11,11)",
                "forcing1232_per_second": "(N_branch,1232)",
                "trust_radii": "(N_branch,) positive",
                "stable_spectral_gaps_per_second": "(N_branch,) positive",
                "guard_margins": "(N_branch,N_transition)",
                "pseudo_arclength_tangents": "(N_branch,1237)",
            },
            "events_train.npz": {
                "pre_states1232": "(N_event,1232)",
                "post_states1232": "(N_event,1232)",
                "pre_invariants4": "(N_event,4)",
                "phase": "(N_event,)",
                "source_mode_index": "(N_event,)",
                "destination_mode_index": "(N_event,)",
                "transition_class_index": "(N_event,)",
                "duration_seconds": "(N_event,) positive",
                "integrated_phase_advance": "(N_event,) positive",
                "integrated_ledger_impulse4": "(N_event,4)",
                "ledger_null_constitutive_jump1232": "(N_event,1232)",
                "guard_value_and_direction": "(N_event,2)",
                "reduced_guard_normals5": "(N_event,5) oriented",
                "destination_guard_margin": "(N_event,) positive",
            },
            "atlas_topology.npz": {
                "q_simplices": "(N_q_simplex,5)",
                "q_scales": "(4,) positive physical scales",
                "branch_simplices": "(N_branch_simplex,6) mode-pure",
                "branch_simplex_modes": "(N_branch_simplex,)",
                "phase_scale": "positive scalar",
                "event_simplices": "(N_event_simplex,5)",
                "event_simplex_classes": "(N_event_simplex,)",
            },
            "transition_specs.json": [
                "unique transition name and class index",
                "source and destination mode index",
                "crossing direction +/-1",
                "hysteresis and re-entry policy",
            ],
            "split_lock.json": [
                "training and holdout identifiers for phase, Q, branch, and every event class",
                "two or more contiguous withheld phase windows",
                "at least 20 percent branch anchors including fold-adjacent cases",
                "at least 20 percent of each event class plus a parameter-edge case",
                "hash and timestamp fixed before the first interpolation or reset fit",
            ],
            "branch_holdout.npz": [
                "independent 1232-state anchors, rates, port/source matrices, ledgers, and guards",
                "no identifier or sample duplicated in any training payload",
            ],
            "event_holdout.npz": [
                "independent entry/exit states, event times, modes, impulses, and guard truth",
                "no event used to construct a guard sheet or reset fit",
            ],
            "sequence_holdout.npz": [
                "one independent full event sequence or full cycle",
                "mode order, event times, endpoint, port actions, and cumulative ledgers",
            ],
            "spatial_holdout.npz": [
                "one independently generated native or refined grid with at least 112 cells",
                "state, rate, port action, physical guards, and resolution provenance",
            ],
            "initial_condition.npz": [
                "physical state1232, invariants4, phase, mode, epoch, and transition inventory",
                "bitwise roundtrip and physical bundle hash",
            ],
            "production_benchmark.json": [
                "at least 1000 mixed driver/branch/guard/reset queries",
                "exact machine, dependency, and thread configuration",
                "query counts, wall time, projected 100000-step wall days, and peak memory",
            ],
            "SHA256SUMS.txt": "every decisive file; one canonical aggregate bundle digest",
        },
        "prospective_acquisition_sequence": [
            {
                "stage": 0,
                "name": "external model declaration",
                "binding_pass": (
                    "all physical equations, ephemeris, sources, boundary environment, "
                    "modes, transitions, citations, parameters, and initial-condition "
                    "provenance are supplied without repository-invented defaults"
                ),
            },
            {
                "stage": 1,
                "name": "prospective split lock",
                "binding_pass": (
                    "training/holdout identifiers and raw truth hashes are committed "
                    "before any coefficient, simplex, guard, or reset fit"
                ),
            },
            {
                "stage": 2,
                "name": "sparse driver and outer-boundary preflight",
                "binding_pass": (
                    "positive periodic phase law, exact four-ledger closure, physical "
                    "outer incoming characteristics, and all thermodynamic guards pass"
                ),
            },
            {
                "stage": 3,
                "name": "cold branch continuation preflight",
                "binding_pass": (
                    "mode-pure pseudo-arclength anchors cover the declared cold Q/phase "
                    "tube with positive trust radii and normal-hyperbolic gaps"
                ),
            },
            {
                "stage": 4,
                "name": "remaining branch sheets",
                "binding_pass": (
                    "each additional physical mode passes the cold-sheet structure and "
                    "prospective branch holdout gates before the next mode is acquired"
                ),
            },
            {
                "stage": 5,
                "name": "event classes",
                "binding_pass": (
                    "each transition class separately passes oriented guard, duration, "
                    "phase advance, ledger impulse, reset, and event-holdout gates"
                ),
            },
            {
                "stage": 6,
                "name": "atlas topology and hull closure",
                "binding_pass": (
                    "every prospective cycle query lies inside a mode-pure trusted "
                    "driver/branch/guard simplex; no fallback or extrapolation exists"
                ),
            },
            {
                "stage": 7,
                "name": "independent spatial and sequence validation",
                "binding_pass": "all frozen heldout thresholds pass without refitting",
            },
            {
                "stage": 8,
                "name": "production payload benchmark and initial-state lock",
                "binding_pass": (
                    "the exact final payload projects at most three wall days for 100000 "
                    "macrosteps and the initial checkpoint closes bitwise"
                ),
            },
            {
                "stage": 9,
                "name": "physical pre-execution readiness certificate",
                "binding_pass": (
                    "all prior stage artifacts are independently checksum-validated; "
                    "only then may a definitions-only single-cycle execution package be frozen"
                ),
            },
        ],
        "heldout_acceptance_gates": heldout_gates,
        "structure_and_conservation_gates": {
            "driver_periodicity_and_first_derivative_close": True,
            "phase_rate_strictly_positive": True,
            "forcing_ledger_relative_defect_maximum": 2.0e-12,
            "branch_invariant_relative_defect_maximum": 2.0e-12,
            "radial_symmetry_defect_maximum": 2.0e-12,
            "source_positive_eigenvalue_maximum": 2.0e-12,
            "source_nullity_exact": 4,
            "inner_incoming_characteristic_count_exact": 0,
            "outer_incoming_characteristic_count_exact": 11,
            "stable_spectral_gap_strictly_positive": True,
            "event_reset_ledger_relative_defect_maximum": 2.0e-12,
            "event_constitutive_null_relative_defect_maximum": 2.0e-12,
            "no_unlocked_extrapolation_or_fallback": True,
        },
        "cost_and_execution_boundary": {
            "complete_cycle_runner_in_this_package": False,
            "new_physical_truth_calls_in_this_package": 0,
            "new_complete_cycle_steps": 0,
            "maximum_future_online_macrosteps": 100000,
            "maximum_future_complete_cycle_wall_days": 3.0,
            "minimum_average_physical_seconds_per_macrostep_for_fiducial_target": 5.7888,
            "runtime_target_is_not_physical_period_evidence": True,
        },
        "current_status": {
            "physical_model_declaration_received": False,
            "prospective_split_locked": False,
            "physical_cycle_bundle_v2_acquired": False,
            "heldout_physical_validation_complete": False,
            "production_runtime_benchmark_complete": False,
            "physical_initial_condition_locked": False,
            "complete_cycle_preexecution_readiness_passed": False,
            "complete_cycle_execution_authorized": False,
            "complete_cycle_steps": 0,
            "repository_seed_evidence_preserved": readiness[
                "evidence_disposition"
            ]["candidate_seed_only"],
        },
        "fail_closed_decision": {
            "blocked_on": "external physical model declaration and raw truth acquisition",
            "first_required_delivery": AUTHORIZED_NEXT,
            "automatic_progression": (
                "after each stage passes, proceed to the next stage; stop immediately "
                "on any scientific, heldout, provenance, or cost failure"
            ),
            "complete_cycle_execution_authorized": False,
        },
        "authorized_next": AUTHORIZED_NEXT,
    }


def _request_template() -> dict:
    return {
        "schema_version": 1,
        "delivery_name": AUTHORIZED_NEXT,
        "delivery_status": "awaiting_external_scientific_input",
        "required_before_repository_execution": {
            "physical_model_id": None,
            "source_repository_and_commit": None,
            "source_citations": [],
            "parameter_table_with_units": None,
            "orbital_ephemeris_and_period": None,
            "impact_geometry_and_phase_zero": None,
            "distributed_source_equations": None,
            "outer_environment_and_incoming_characteristic_law": None,
            "physical_mode_definitions": None,
            "guard_hysteresis_and_destination_policy": None,
            "event_truth_solver_definition": None,
            "physical_initial_condition_source": None,
            "raw_truth_archive_location_and_hashes": None,
        },
        "repository_defaults_permitted": False,
        "synthetic_substitution_permitted": False,
        "legacy_prefix_extrapolation_permitted": False,
        "complete_cycle_execution_authorized": False,
    }


def _update(summary):
    utility = _u()
    rows = list(csv.DictReader(CANONICAL_MANIFEST.open(newline="", encoding="utf-8")))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": utility._sha256(path),
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
    catalog = utility._read_json(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": CLASSIFICATION,
        "passed": True,
    }
    catalog.update(
        {
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": utility._git("rev-parse", "HEAD"),
            "latest_work_package": WORK_PACKAGE,
        }
    )
    utility._write_json(CANONICAL_SUMMARY, catalog)


def _freeze():
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("physical cycle-bundle acquisition manifest already exists")
    hashes, parent_summary, readiness, inventory = _validate_parent(require_clean=True)
    contract = _contract(readiness)
    request = _request_template()
    if (
        contract["current_status"]["physical_model_declaration_received"]
        or contract["current_status"]["physical_cycle_bundle_v2_acquired"]
        or contract["current_status"]["complete_cycle_execution_authorized"]
        or contract["cost_and_execution_boundary"]["new_complete_cycle_steps"] != 0
        or request["repository_defaults_permitted"]
        or inventory["directly_reusable_binding_cycle_field_count"] != 0
    ):
        raise RuntimeError("acquisition manifest crossed its definitions-only boundary")

    utility = _u()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utility._write_json(
        CANONICAL_DIRECTORY / "physical_cycle_bundle_v2_acquisition_contract.json",
        contract,
    )
    utility._write_json(
        CANONICAL_DIRECTORY / "external_physical_model_request.json", request
    )
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "physical_readiness_negative_finding_preserved": True,
        "physical_model_declaration_received": False,
        "prospective_split_locked": False,
        "physical_cycle_bundle_v2_acquired": False,
        "heldout_physical_validation_complete": False,
        "production_runtime_benchmark_complete": False,
        "physical_initial_condition_locked": False,
        "complete_cycle_preexecution_readiness_passed": False,
        "complete_cycle_execution_authorized": False,
        "complete_cycle_steps": 0,
        "external_scientific_input_required": True,
        "authorized_next": AUTHORIZED_NEXT,
    }
    utility._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utility._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_artifact": parent.ARTIFACT,
            "parent_checksum_manifest_sha256": PARENT_SHA256,
            "parent_hashes": hashes,
            "parent_classification": parent_summary["classification"],
            "parent_directly_reusable_binding_cycle_field_count": inventory[
                "directly_reusable_binding_cycle_field_count"
            ],
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# Physical cycle-bundle v2 acquisition manifest\n\n"
        f"Classification: `{CLASSIFICATION}`.\n\n"
        "The mathematical and software architecture is fixed; this package freezes the "
        "physical evidence required to instantiate it. The first delivery must be an "
        "externally authorized physical-model declaration: ephemeris, period, distributed "
        "sources, outer incoming loading, mode and transition definitions, citations, "
        "parameters, and initial-condition provenance. The repository is forbidden from "
        "inventing defaults from the 6.7-day runtime target or the 0.016-second prefix.\n\n"
        "The v2 bundle separates training branch/event payloads from physical branch, "
        "event, spatial, and sequence holdouts; adds the exact simplex topology and finite "
        "phase-advancing reset fields used by the production kernel; and requires a "
        "1000-query production benchmark. Acquisition proceeds fail-fast from model "
        "declaration through split lock, sparse driver/boundary preflight, branch sheets, "
        "events, topology, independent validation, and benchmark.\n\n"
        "No physical declaration or truth archive has been supplied in this package. It "
        "contains no cycle runner, performs no new physical truth call, and executes zero "
        "complete-cycle steps. Work must stop at the external-input request until a "
        "scientifically authorized model and raw truth provenance are provided.\n",
        encoding="utf-8",
    )
    sources = (THIS_RUNNER, THIS_TEST, REPORT_RELATIVE)
    utility._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "implementation_commit": utility._git("rev-parse", "HEAD"),
            "source_hashes": {
                source: utility._sha256(ROOT / source) for source in sources
            },
        },
    )
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{utility._sha256(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
    _update(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    arguments = parser.parse_args()
    if not arguments.freeze:
        parser.error("choose --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
