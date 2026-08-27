#!/usr/bin/env python3
"""Freeze the physical driver, boundary, branch, and event-truth acquisition."""

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

import run_causal_inner_conservative_entropy_reset_and_guard_localization_structure_certificate_wp10c9d6c7c3b5c4f25fizzs1 as parent  # noqa: E402


WORK_PACKAGE = parent.AUTHORIZED_NEXT
CLASSIFICATION = (
    "cycle_wide_physical_driver_boundary_branch_and_event_truth_acquisition_"
    "manifest_frozen_external_physics_inputs_missing"
)
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizzt1_cycle_physical_input_bundle_schema_and_"
    "validator_certificate"
)
PASS_NEXT = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizzu_nonperiodic_native_global_AP_"
    "boundary_action_structure_manifest"
)
ARTIFACT = (
    "causal_inner_cycle_wide_physical_driver_boundary_loading_and_event_truth_"
    "acquisition_manifest_wp10c9d6c7c3b5c4f25fizzt"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_CYCLE_WIDE_PHYSICAL_DRIVER_BOUNDARY_"
    "LOADING_AND_EVENT_TRUTH_ACQUISITION_MANIFEST_WP10C9D6C7C3B5C4F25FIZZT_"
    "2026-08-27.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_cycle_wide_physical_driver_boundary_loading_and_"
    "event_truth_acquisition_manifest_wp10c9d6c7c3b5c4f25fizzt.py"
)
THIS_TEST = (
    "tests/test_causal_inner_cycle_wide_physical_driver_boundary_loading_and_"
    "event_truth_acquisition_manifest_wp10c9d6c7c3b5c4f25fizzt.py"
)
PARENT_SHA256 = "fb8830d557a2afc13dc0d72557df69654884f491e6f3ff3fd81759ae3a3d3ff5"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _u():
    return parent._u()


def _validate_parent(require_clean: bool = False):
    utility = _u()
    if utility._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != PARENT_SHA256:
        raise RuntimeError("reset/localization structure certificate changed")
    hashes = utility._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utility._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = utility._read_json(
        parent.CANONICAL_DIRECTORY / "reset_and_guard_metrics.json"
    )
    if (
        not summary["passed"]
        or not summary["reset_and_guard_structure_certified"]
        or summary["events_and_resets_physically_calibrated"]
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["complete_cycle_execution_authorized"]
        or metrics["complete_cycle_steps"] != 0
    ):
        raise RuntimeError("reset/localization structure classification changed")
    if require_clean and utility._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("physical-input acquisition manifest needs a clean tracked tree")
    return hashes, summary, metrics


def _contract() -> dict:
    cycle_seconds = 578880.0
    return {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "scientific_boundary": {
            "fiducial_cycle_seconds": cycle_seconds,
            "fiducial_period_role": (
                "runtime target only until a physical orbital/driver model supplies and "
                "validates the phase law"
            ),
            "certified_short_time_model_scope": (
                "no tide, no wind, no hot-state/S-curve cycle closure, static prefix "
                "exterior candidate, and 0.016 seconds of revealed dynamics"
            ),
            "forbidden_inference": (
                "the 6.7-day target, 16-ms prefix, or old static exterior state may not be "
                "promoted into cycle-wide physical forcing"
            ),
        },
        "required_external_physical_model": {
            "orbital_ephemeris": [
                "phase phi(t) on S1 and strictly positive phase rate",
                "physical period and its parameter provenance",
                "impact geometry and orientation relative to the disk",
            ],
            "distributed_sources": [
                "rest-mass injection rate and native radial profile",
                "specific angular momentum of injected material",
                "specific total energy/entropy of injected material",
                "tidal work and torque if present",
                "wind mass, angular-momentum, and energy loss if present",
            ],
            "outer_environment": [
                "physical exterior primitive/entropy state or incoming flux law",
                "all eleven incoming entropy-characteristic amplitudes at the outer edge",
                "phase, retained-invariant, and mode dependence",
            ],
            "constitutive_cycle_closure": [
                "cold/hot/cooling/recovery mode definitions",
                "physical transition guards with crossing orientation",
                "destination mode and hysteresis policy",
            ],
            "currently_present_in_repository_as_cycle_wide_binding_data": False,
        },
        "canonical_input_bundle": {
            "metadata_json": [
                "schema version and physical model identifier",
                "units and coordinate conventions",
                "parameter values, priors, and source citations",
                "source-code commit and environment hashes",
                "training/validation/heldout split fixed before fitting",
            ],
            "driver_npz": {
                "phase_nodes": "(N_phi,) including 0 and 2*pi",
                "phase_rate_per_second": "(N_phi,) strictly positive and periodic",
                "source_ledger_rate4": "(N_phi,N_q,N_mode,4)",
                "slow_forcing1232_per_second": "(N_phi,N_q,N_mode,1232)",
                "outer_incoming_characteristics11": "(N_phi,N_q,N_mode,11)",
                "retained_invariant_nodes4": "(N_q,4)",
                "mode_labels": "(N_mode,) fixed-width strings",
            },
            "branch_npz": {
                "anchor_states1232": "(N_anchor,1232)",
                "anchor_phase": "(N_anchor,)",
                "anchor_invariants4": "(N_anchor,4)",
                "anchor_mode_index": "(N_anchor,)",
                "radial_matrices112x11x11": "(N_anchor,112,11,11)",
                "source_matrices112x11x11": "(N_anchor,112,11,11)",
                "forcing1232_per_second": "(N_anchor,1232)",
                "trust_radii": "(N_anchor,)",
                "stable_spectral_gaps_per_second": "(N_anchor,)",
                "guard_margins": "(N_anchor,N_guard)",
                "pseudo_arclength_tangents": "(N_anchor,1232+5)",
            },
            "events_npz": {
                "pre_states1232": "(N_event,1232)",
                "post_states1232": "(N_event,1232)",
                "pre_invariants4": "(N_event,4)",
                "phase": "(N_event,)",
                "source_mode_index": "(N_event,)",
                "destination_mode_index": "(N_event,)",
                "duration_seconds": "(N_event,)",
                "integrated_ledger_impulse4": "(N_event,4)",
                "ledger_null_constitutive_jump1232": "(N_event,1232)",
                "guard_value_and_direction": "(N_event,2)",
            },
            "heldout_truth_npz": {
                "withheld_branch_anchors": "complete states, fields, and ledgers",
                "withheld_phase_windows": "matched endpoints and integrated ledgers",
                "withheld_events": "entry/exit states, event time, mode, and impulse",
                "withheld_sequence": "one independent full event sequence or full cycle",
            },
        },
        "offline_branch_problem": {
            "coordinates": "(z,q,phi,m) with z in R^1232 and q=Cz in R^4",
            "stationarity": (
                "P_fast(z,q,phi,m) F_physical(z,q,phi,m)=0 with C z=q and physical "
                "boundary/source constraints"
            ),
            "continuation": (
                "bordered pseudo-arclength in (q,phi) with adaptive arclength, fold "
                "crossing, exact residual correction, and accepted-anchor-only history"
            ),
            "normal_hyperbolicity": (
                "all eliminated fast eigenvalues strictly stable and separated from retained "
                "rates by at least the prospectively frozen gap ratio"
            ),
            "fold_policy": "fold or stability loss becomes a physical guard candidate, not extrapolation",
        },
        "driver_and_boundary_gates": {
            "phase_periodicity": "values and first derivatives close at 0/2*pi",
            "positive_phase_rate": True,
            "outer_projection": (
                "physical exterior state/flux is projected onto the eleven certified "
                "incoming outward-normal entropy characteristics"
            ),
            "ledger_identity": (
                "C b equals integrated distributed source plus outer/inner boundary ledger rate"
            ),
            "thermodynamic_admissibility": [
                "finite positive density and temperature",
                "height/radius below 0.5",
                "scattering optical depth above 1",
                "subluminal characteristic speeds",
            ],
            "no_unlocked_extrapolation": True,
        },
        "event_truth_acquisition": {
            "truth_unit": (
                "one offline entry-to-exit physical solve with dense guard localization and "
                "complete integrated face/source/constraint ledgers"
            ),
            "reset_fit": (
                "DeltaQ_event and a ledger-null constitutive jump are fit separately; the "
                "certified weighted reset geometry enforces the ledger exactly"
            ),
            "minimum_event_classes": ["impact", "hot_exit", "cooling_entry", "recovery_exit"],
            "required_variations": [
                "retained invariant",
                "orbital phase/impact parameter",
                "entry direction",
                "source and destination mode",
            ],
            "online_event_truth_calls": 0,
        },
        "prospective_holdouts": {
            "branch": "at least 20 percent of anchors, including fold-adjacent anchors",
            "phase": "at least two contiguous phase windows absent from all fits",
            "events": "at least 20 percent per event class and one parameter-edge sample",
            "spatial": "one independently generated native-grid or refined-grid truth comparison",
            "sequence": "one complete event sequence or cycle never used for fitting",
            "leakage": "hash-lock all split indices before the first coefficient fit",
        },
        "binding_validation": {
            "regular_anchor_state_relative_defect_maximum": 2.0e-2,
            "regular_anchor_rate_relative_defect_maximum": 5.0e-2,
            "physical_reaction_or_port_action_relative_defect_maximum": 5.0e-2,
            "event_time_relative_defect_maximum": 2.0e-2,
            "event_post_state_relative_defect_maximum": 5.0e-2,
            "event_ledger_relative_defect_maximum": 2.0e-2,
            "full_sequence_ledger_relative_defect_maximum": 2.0e-2,
            "all_physical_and_entropy_guards_binding": True,
        },
        "fail_fast_acquisition_order": [
            "validate external physical model metadata and phase law",
            "acquire outer loading and distributed source ledgers on a sparse phase grid",
            "continue one cold branch sheet and validate prospective cold holdouts",
            "continue hot/cooling/recovery sheets only after the cold sheet passes",
            "acquire each event class independently and validate event holdouts immediately",
            "build the full adaptive atlas only after all sparse preflights pass",
            "run one heldout event sequence or cycle only after the atlas is frozen",
        ],
        "cost_and_execution_boundary": {
            "maximum_online_macrosteps": 100000,
            "minimum_average_macrostep_seconds_for_fiducial_target": cycle_seconds / 100000.0,
            "maximum_cycle_wall_days": 3.0,
            "online_truth_calls": 0,
            "online_large_nonlinear_roots": 0,
            "complete_cycle_runner_may_exist_in_this_package": False,
            "complete_cycle_steps": 0,
        },
        "current_missing_payloads": {
            "external_physical_model_metadata": True,
            "cycle_wide_phase_law": True,
            "cycle_wide_distributed_forcing": True,
            "cycle_wide_outer_incoming_loading": True,
            "cold_hot_cooling_recovery_branch_sheets": True,
            "calibrated_physical_events": True,
            "heldout_event_sequence_or_cycle": True,
        },
        "claim_boundary": {
            "input_schema_and_validator_certified": False,
            "physical_model_complete": False,
            "physical_payloads_acquired": False,
            "events_and_resets_physically_calibrated": False,
            "heldout_cycle_validation_complete": False,
            "complete_cycle_execution_authorized": False,
            "complete_cycle_steps": 0,
        },
        "authorized_next": AUTHORIZED_NEXT,
        "pass_authorized_next": PASS_NEXT,
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
        "classification": summary["classification"],
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
        raise RuntimeError("physical-input acquisition manifest exists")
    hashes, _, _ = _validate_parent(require_clean=True)
    utility = _u()
    contract = _contract()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utility._write_json(CANONICAL_DIRECTORY / "physical_input_acquisition_contract.json", contract)
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "reset_and_guard_structure_certified": True,
        "input_schema_and_validator_certified": False,
        "physical_model_complete": False,
        "physical_payloads_acquired": False,
        "events_and_resets_physically_calibrated": False,
        "heldout_cycle_validation_complete": False,
        "complete_cycle_execution_authorized": False,
        "complete_cycle_steps": 0,
        "authorized_next": AUTHORIZED_NEXT,
    }
    utility._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utility._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_artifact": parent.ARTIFACT,
            "parent_checksum_manifest_sha256": PARENT_SHA256,
            "parent_hashes": hashes,
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# Cycle-wide physical-input and event-truth acquisition manifest\n\n"
        f"Classification: `{CLASSIFICATION}`.\n\n"
        "The physical interface is now explicit: orbital phase law, distributed mass/"
        "angular-momentum/energy sources, all eleven outer incoming characteristics, "
        "cold/hot/cooling/recovery branches, and entry-to-exit event truth are separate "
        "binding inputs. The existing 6.7-day value is a runtime target, not a derived "
        "period of the certified no-tide/no-wind short-time equations.\n\n"
        "Branch sheets use bordered pseudo-arclength and normal-hyperbolicity gates; folds "
        "and stability loss become event guards. Each event is calibrated from an offline "
        "physical entry-to-exit solve and separated into a four-ledger impulse plus a "
        "ledger-null constitutive jump. Prospective branch, phase, event, spatial, and full-"
        "sequence holdouts are frozen before fitting.\n\n"
        "None of the required cycle-wide physical payloads currently exists as binding "
        "repository data. The next package certifies a fail-closed bundle validator only. "
        "No cycle runner or cycle step is authorized.\n",
        encoding="utf-8",
    )
    sources = (THIS_RUNNER, THIS_TEST, REPORT_RELATIVE)
    utility._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "implementation_commit": utility._git("rev-parse", "HEAD"),
            "source_hashes": {source: utility._sha256(ROOT / source) for source in sources},
        },
    )
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{utility._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names
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
