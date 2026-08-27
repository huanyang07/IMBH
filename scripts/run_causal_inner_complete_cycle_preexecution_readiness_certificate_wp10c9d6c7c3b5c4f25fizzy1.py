#!/usr/bin/env python3
"""Inventory physical evidence and certify complete-cycle pre-execution readiness."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORK_PACKAGE = (
    "physical_WP10c9d6c7c3b5c4f25fizzy1_"
    "complete_cycle_preexecution_readiness_certificate"
)
CLASSIFICATION = (
    "complete_cycle_preexecution_readiness_audit_passed_"
    "external_physical_bundle_absent"
)
AUTHORIZED_NEXT = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizzy2_"
    "physical_cycle_bundle_acquisition_manifest"
)
ARTIFACT = (
    "causal_inner_complete_cycle_preexecution_readiness_certificate_"
    "wp10c9d6c7c3b5c4f25fizzy1"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_COMPLETE_CYCLE_PREEXECUTION_"
    "READINESS_CERTIFICATE_WP10C9D6C7C3B5C4F25FIZZY1_2026-08-27.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_complete_cycle_preexecution_readiness_"
    "certificate_wp10c9d6c7c3b5c4f25fizzy1.py"
)
THIS_TEST = (
    "tests/test_causal_inner_complete_cycle_preexecution_readiness_"
    "certificate_wp10c9d6c7c3b5c4f25fizzy1.py"
)
PARENT_SHA256 = "a572d1030313a0af2e66ee6e7cb75f04659d204a516f85bd546a3da1e2f0ac3d"
PARENT_ARTIFACT = "causal_inner_complete_cycle_preexecution_manifest_wp10c9d6c7c3b5c4f25fizzy"
PARENT_DIRECTORY = ROOT / "results/canonical" / PARENT_ARTIFACT
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

SUPPORT_ARTIFACTS = {
    "legacy_cycle_evidence": (
        "causal_inner_legacy_cycle_evidence_compatibility_audit_"
        "wp10c9d6c7c3b5c4f25fizzp1",
        "d6b1b2c6afe76b56a407e9b2baa0a319b6ec6a111a24668938e253b8cb7e5ac3",
    ),
    "native_global_AP": (
        "causal_inner_physical_112_cell_global_ap_dry_run_"
        "wp10c9d6c7c3b5c4f25fizzq1",
        "85d5da9b89ffe0fe261bda7e6da89b2ee677cc88f74c3edeea30a7958e83ac01",
    ),
    "prefix_ports_and_boundaries": (
        "causal_inner_prefix_port_payload_and_boundary_structure_certificate_"
        "wp10c9d6c7c3b5c4f25fizzr1",
        "4b491cbba2440f6106da7ae69c54c494ecaa5c15137f8fd4aa808cf305b3d9c6",
    ),
    "physical_bundle_acquisition_contract": (
        "causal_inner_cycle_wide_physical_driver_boundary_loading_and_event_"
        "truth_acquisition_manifest_wp10c9d6c7c3b5c4f25fizzt",
        "af99169da4ceb552d48779245f8505a6a78ada7b4b54541f6940c9b1a8f1b1d4",
    ),
    "physical_bundle_schema": (
        "causal_inner_cycle_physical_input_bundle_schema_and_validator_"
        "certificate_wp10c9d6c7c3b5c4f25fizzt1",
        "f852d2b520700e81444c5472d2c00d426c133111de4247739362d0f8fb1e8a1c",
    ),
}

REQUIRED_METADATA_FIELDS = (
    "schema_version",
    "physical_model_id",
    "physical_model_complete",
    "synthetic_fixture",
    "period_seconds",
    "unit_system",
    "source_citations",
    "source_code_commit",
    "split_frozen_before_fit",
)

REQUIRED_ARRAY_FIELDS = {
    "driver": (
        "phase_nodes",
        "phase_rate_per_second",
        "retained_invariant_nodes4",
        "mode_labels",
        "slow_forcing1232_per_second",
        "distributed_source_ledger_rate4",
        "boundary_ledger_rate4",
        "outer_incoming_characteristics11",
    ),
    "branch": (
        "anchor_states1232",
        "anchor_phase",
        "anchor_invariants4",
        "anchor_mode_index",
        "radial_matrices112x11x11",
        "source_matrices112x11x11",
        "forcing1232_per_second",
        "trust_radii",
        "stable_spectral_gaps_per_second",
        "guard_margins",
        "pseudo_arclength_tangents",
    ),
    "events": (
        "pre_states1232",
        "post_states1232",
        "pre_invariants4",
        "phase",
        "source_mode_index",
        "destination_mode_index",
        "duration_seconds",
        "integrated_ledger_impulse4",
        "ledger_null_constitutive_jump1232",
        "guard_value_and_direction",
    ),
    "heldout_truth": (
        "withheld_branch_anchor_indices",
        "withheld_event_indices",
        "withheld_phase_windows",
        "sequence_mode_indices",
        "sequence_ledger_increments4",
        "spatial_truth_grid_cells",
    ),
}

EXPECTED_BUNDLE_FILES = (
    "metadata.json",
    "driver.npz",
    "branch.npz",
    "events.npz",
    "heldout_truth.npz",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_checksums(directory: Path) -> dict[str, str]:
    manifest = directory / "SHA256SUMS.txt"
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        expected[name] = digest
    actual_names = {
        path.name for path in directory.iterdir() if path.is_file() and path.name != manifest.name
    }
    if set(expected) != actual_names:
        raise RuntimeError(f"checksum inventory changed: {directory}")
    for name, digest in expected.items():
        if _sha256(directory / name) != digest:
            raise RuntimeError(f"checksum mismatch: {directory / name}")
    return expected


def _u():
    return sys.modules[__name__]


def _validate_parent(*, require_clean: bool = False):
    utility = _u()
    if utility._sha256(PARENT_DIRECTORY / "SHA256SUMS.txt") != PARENT_SHA256:
        raise RuntimeError("complete-cycle preexecution manifest changed")
    hashes = utility._validate_checksums(PARENT_DIRECTORY)
    summary = utility._read_json(PARENT_DIRECTORY / "summary.json")
    status = utility._read_json(PARENT_DIRECTORY / "readiness_status.json")
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or not summary["mathematical_architecture_verified"]
        or summary["physical_payloads_acquired"]
        or summary["complete_cycle_execution_ready"]
        or summary["complete_cycle_execution_authorized"]
        or summary["complete_cycle_steps"] != 0
        or summary["authorized_next"] != WORK_PACKAGE
        or status["physical_cycle_bundle_v2_acquired"]
        or status["complete_cycle_execution_ready"]
    ):
        raise RuntimeError("complete-cycle preexecution classification changed")
    if require_clean and utility._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("readiness certificate needs a clean tracked tree")
    return hashes


def _validate_support() -> dict:
    utility = _u()
    hashes = {}
    for label, (artifact, expected_sha) in SUPPORT_ARTIFACTS.items():
        directory = ROOT / "results/canonical" / artifact
        actual_sha = utility._sha256(directory / "SHA256SUMS.txt")
        if actual_sha != expected_sha:
            raise RuntimeError(f"support artifact changed: {label}")
        hashes[label] = {
            "artifact": artifact,
            "checksum_manifest_sha256": actual_sha,
            "payload_hashes": utility._validate_checksums(directory),
        }

    legacy_directory = ROOT / "results/canonical" / SUPPORT_ARTIFACTS[
        "legacy_cycle_evidence"
    ][0]
    legacy = utility._read_json(legacy_directory / "compatibility_audit.json")
    port_directory = ROOT / "results/canonical" / SUPPORT_ARTIFACTS[
        "prefix_ports_and_boundaries"
    ][0]
    ports = utility._read_json(port_directory / "port_and_boundary_metrics.json")
    ap_directory = ROOT / "results/canonical" / SUPPORT_ARTIFACTS["native_global_AP"][0]
    ap = utility._read_json(ap_directory / "global_dry_run_metrics.json")
    schema_directory = ROOT / "results/canonical" / SUPPORT_ARTIFACTS[
        "physical_bundle_schema"
    ][0]
    schema = utility._read_json(schema_directory / "validator_metrics.json")
    if (
        not legacy["passed"]
        or legacy["direct_binding_reuse_count"] != 0
        or legacy["facts"]["old_cycle_observed"]
        or legacy["facts"]["old_hot_exit_observed"]
        or not ports["passed"]
        or ports["candidate_anchor_count"] != 913
        or ports["outer_cycle_loading_complete"]
        or ports["slow_forcing_b_included"]
        or not ap["passed"]
        or ap["global_state_dimension"] != 1232
        or ap["physical_context_cells"] != 112
        or ap["physical_boundaries_certified"]
        or ap["cycle_wide_inputs_complete"]
        or not schema["passed"]
        or not schema["structurally_passed"]
        or not schema["synthetic_fixture_rejected_when_physical_required"]
        or schema["physically_usable"]
    ):
        raise RuntimeError("supporting physical-evidence classification changed")
    return {
        "hash_locks": hashes,
        "legacy": legacy,
        "ports": ports,
        "native_global_AP": ap,
        "schema": schema,
    }


def _tracked_paths() -> tuple[str, ...]:
    listing = _u()._git("ls-files")
    return tuple(sorted(line for line in listing.splitlines() if line))


def _npz_keys(path: Path) -> tuple[str, ...]:
    with zipfile.ZipFile(path) as archive:
        return tuple(
            sorted(
                name[:-4]
                for name in archive.namelist()
                if name.endswith(".npy") and "/" not in name
            )
        )


def _inventory() -> dict:
    tracked = _tracked_paths()
    tracked_set = set(tracked)
    npz_paths = tuple(path for path in tracked if path.endswith(".npz"))
    json_paths = tuple(path for path in tracked if path.endswith(".json"))

    keys_by_npz = {}
    unreadable_npz = []
    for relative in npz_paths:
        try:
            keys_by_npz[relative] = _npz_keys(ROOT / relative)
        except (OSError, zipfile.BadZipFile):
            unreadable_npz.append(relative)

    complete_group_files = {}
    exact_occurrences = {}
    for group, fields in REQUIRED_ARRAY_FIELDS.items():
        required = set(fields)
        complete_group_files[group] = sorted(
            path for path, keys in keys_by_npz.items() if required.issubset(keys)
        )
        exact_occurrences[group] = {
            field: sorted(
                path for path, keys in keys_by_npz.items() if field in set(keys)
            )
            for field in fields
        }

    complete_bundle_directories = []
    for relative in tracked:
        if Path(relative).name != "metadata.json":
            continue
        directory = Path(relative).parent
        expected = {str(directory / name) for name in EXPECTED_BUNDLE_FILES}
        if expected.issubset(tracked_set):
            complete_bundle_directories.append(str(directory))

    metadata_records = []
    required_metadata = set(REQUIRED_METADATA_FIELDS)
    for relative in json_paths:
        try:
            payload = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and required_metadata.issubset(payload):
            metadata_records.append(
                {
                    "path": relative,
                    "physical_model_complete": bool(payload["physical_model_complete"]),
                    "synthetic_fixture": bool(payload["synthetic_fixture"]),
                }
            )
    physical_metadata_records = [
        record
        for record in metadata_records
        if record["physical_model_complete"] and not record["synthetic_fixture"]
    ]

    required_qualified_fields = sum(len(fields) for fields in REQUIRED_ARRAY_FIELDS.values())
    complete_group_count = sum(len(paths) for paths in complete_group_files.values())
    return {
        "inventory_scope": "all git-tracked JSON and NPZ payloads",
        "tracked_file_count": len(tracked),
        "tracked_json_count": len(json_paths),
        "tracked_npz_count": len(npz_paths),
        "unreadable_tracked_npz": unreadable_npz,
        "required_metadata_fields": list(REQUIRED_METADATA_FIELDS),
        "required_array_fields_by_group": {
            group: list(fields) for group, fields in REQUIRED_ARRAY_FIELDS.items()
        },
        "required_qualified_array_field_count": required_qualified_fields,
        "exact_field_occurrences": exact_occurrences,
        "complete_group_files": complete_group_files,
        "complete_group_file_count": complete_group_count,
        "complete_bundle_directories": sorted(complete_bundle_directories),
        "complete_bundle_directory_count": len(complete_bundle_directories),
        "metadata_records_matching_schema": metadata_records,
        "physical_metadata_records": physical_metadata_records,
        "physical_metadata_record_count": len(physical_metadata_records),
        "directly_reusable_binding_cycle_field_count": 0,
    }


def _readiness(support: dict, inventory: dict) -> dict:
    legacy = support["legacy"]
    ports = support["ports"]
    ap = support["native_global_AP"]
    facts = legacy["facts"]
    reusable = {
        "candidate_seed_only": [
            {
                "evidence": "accepted short physical trajectory profiles",
                "count": facts["trajectory_primitive_shape"][0],
                "shape": facts["trajectory_primitive_shape"],
                "observed_physical_seconds": facts["observed_physical_seconds"],
                "binding_limit": "short five-field prefix; no full cycle or hot exit",
            },
            {
                "evidence": "exact short-time witness profiles",
                "count": facts["exact_witness_primitive_shape"][0],
                "shape": facts["exact_witness_primitive_shape"],
                "binding_limit": "five-field witnesses; no cycle-wide branch coverage",
            },
            {
                "evidence": "local eleven-field port/source anchors",
                "count": ports["candidate_anchor_count"],
                "native_radial_cells": ports["native_radial_cells"],
                "binding_limit": (
                    "per-cell prefix anchors; no 1232-state cycle branch sheets, "
                    "slow forcing, or physical outer loading"
                ),
            },
            {
                "evidence": "accepted 20.0017-ms terminal checkpoint",
                "count": 1,
                "binding_limit": "physical seed only; not a transition or reset endpoint",
            },
        ],
        "structure_only": [
            "four-invariant conservation map and minimum-norm correction",
            f"native {ap['physical_context_cells']}-cell/{ap['global_state_dimension']}-state AP operator structure",
            "inner excision and eleven-incoming outer characteristic structure",
            "reduced hybrid driver/branch/guard/reset production interfaces",
        ],
        "synthetic_only": [
            "complete physical-input schema fixture",
            "reduced driver/branch/event kernel fixtures and runtime projections",
        ],
        "direct_binding_cycle_payloads": [],
    }
    missing = {
        "external_physical_model_metadata": True,
        "physical_orbital_ephemeris_and_phase_law": True,
        "cycle_wide_distributed_forcing": True,
        "cycle_wide_outer_incoming_loading": True,
        "mode_pure_1232_state_branch_sheets": True,
        "stable_spectral_gap_and_trust_data_over_full_tube": True,
        "calibrated_transition_guard_and_reset_truth": True,
        "prospectively_split_branch_event_and_phase_holdouts": True,
        "independent_spatial_truth": True,
        "independent_full_sequence_or_cycle_truth": True,
        "physical_production_payload_runtime_benchmark": True,
        "hash_locked_initial_phase_mode_state_and_transition_inventory": True,
    }
    gates = {
        "repository_inventory_complete": not inventory["unreadable_tracked_npz"],
        "prior_evidence_hash_locked": True,
        "mathematical_architecture_verified": True,
        "physical_bundle_schema_and_fail_closed_validator_verified": True,
        "complete_physical_bundle_directory_present": bool(
            inventory["complete_bundle_directory_count"]
        ),
        "complete_nonsynthetic_metadata_present": bool(
            inventory["physical_metadata_record_count"]
        ),
        "complete_driver_group_present": bool(inventory["complete_group_files"]["driver"]),
        "complete_branch_group_present": bool(inventory["complete_group_files"]["branch"]),
        "complete_event_group_present": bool(inventory["complete_group_files"]["events"]),
        "complete_heldout_group_present": bool(
            inventory["complete_group_files"]["heldout_truth"]
        ),
        "independent_spatial_holdout_complete": False,
        "independent_sequence_or_cycle_holdout_complete": False,
        "physical_runtime_benchmark_complete": False,
        "physical_initial_condition_locked": False,
    }
    readiness_passed = bool(
        gates["repository_inventory_complete"]
        and gates["prior_evidence_hash_locked"]
        and gates["complete_physical_bundle_directory_present"]
        and gates["complete_nonsynthetic_metadata_present"]
        and gates["complete_driver_group_present"]
        and gates["complete_branch_group_present"]
        and gates["complete_event_group_present"]
        and gates["complete_heldout_group_present"]
        and gates["independent_spatial_holdout_complete"]
        and gates["independent_sequence_or_cycle_holdout_complete"]
        and gates["physical_runtime_benchmark_complete"]
        and gates["physical_initial_condition_locked"]
    )
    return {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "inventory_audit_passed": gates["repository_inventory_complete"],
        "complete_cycle_preexecution_readiness_passed": readiness_passed,
        "evidence_disposition": reusable,
        "genuinely_missing_external_evidence": missing,
        "readiness_gates": gates,
        "binding_interpretation": {
            "physical_model_failure_selected": False,
            "mathematical_architecture_failure_selected": False,
            "repository_corruption_or_unreadable_payload_selected": False,
            "external_physical_acquisition_block_selected": True,
            "legacy_prefix_may_be_used_as_binding_cycle_truth": False,
            "synthetic_fixture_may_be_relabeled_physical": False,
        },
        "decision": {
            "status": "blocked_before_physical_complete_cycle_preexecution",
            "reason": (
                "no complete nonsynthetic physical driver, branch, event, heldout, "
                "runtime-benchmark, and initial-condition package exists"
            ),
            "authorized_next": AUTHORIZED_NEXT,
            "complete_cycle_execution_authorized": False,
            "complete_cycle_steps": 0,
        },
        "authorized_next": AUTHORIZED_NEXT,
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
        raise RuntimeError("complete-cycle readiness certificate already exists")
    parent_hashes = _validate_parent(require_clean=True)
    support = _validate_support()
    inventory = _inventory()
    readiness = _readiness(support, inventory)
    if (
        not readiness["inventory_audit_passed"]
        or readiness["complete_cycle_preexecution_readiness_passed"]
        or inventory["complete_bundle_directory_count"] != 0
        or inventory["physical_metadata_record_count"] != 0
        or inventory["complete_group_file_count"] != 0
    ):
        raise RuntimeError("repository evidence no longer supports the frozen negative finding")

    utility = _u()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utility._write_json(CANONICAL_DIRECTORY / "repository_payload_inventory.json", inventory)
    utility._write_json(CANONICAL_DIRECTORY / "readiness_audit.json", readiness)
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "inventory_audit_passed": True,
        "mathematical_architecture_verified": True,
        "directly_reusable_binding_cycle_field_count": 0,
        "candidate_local_port_anchor_count": support["ports"]["candidate_anchor_count"],
        "candidate_short_profile_count": (
            support["legacy"]["facts"]["trajectory_primitive_shape"][0]
            + support["legacy"]["facts"]["exact_witness_primitive_shape"][0]
        ),
        "physical_cycle_bundle_v2_acquired": False,
        "heldout_physical_validation_complete": False,
        "physical_runtime_benchmark_complete": False,
        "physical_initial_condition_locked": False,
        "complete_cycle_preexecution_readiness_passed": False,
        "complete_cycle_execution_authorized": False,
        "complete_cycle_steps": 0,
        "physical_model_failure_selected": False,
        "external_physical_acquisition_block_selected": True,
        "authorized_next": AUTHORIZED_NEXT,
    }
    utility._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utility._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_artifact": PARENT_ARTIFACT,
            "parent_checksum_manifest_sha256": PARENT_SHA256,
            "parent_hashes": parent_hashes,
            "support_artifacts": support["hash_locks"],
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# Complete-cycle physical pre-execution readiness certificate\n\n"
        f"Classification: `{CLASSIFICATION}`.\n\n"
        "A mechanical inventory of every git-tracked JSON and NPZ payload found no "
        "canonical five-file physical bundle, no complete nonsynthetic metadata record, "
        "and no complete driver, branch, event, or heldout array group. Therefore the "
        "physical complete-cycle readiness gate fails closed. This is an external-data "
        "acquisition block, not a failure of the verified reduced-hybrid architecture or "
        "of the short-time physical equations.\n\n"
        "The repository evidence remains useful but has a narrower role: 65 accepted "
        "short physical profiles, 192 exact witnesses, 913 local eleven-field port/source "
        "anchors, native 112-cell conservation/operator structure, and the accepted "
        "20.0017-ms terminal state are retained as candidate seeds or structural evidence. "
        "None is relabeled as cycle-wide branch, event, or heldout truth. The complete "
        "schema fixture remains explicitly synthetic.\n\n"
        "The next authorized artifact is a definitions-only acquisition manifest for the "
        "genuinely missing physical model, ephemeris, forcing, boundary loading, branch "
        "sheets, event truth, prospective holdouts, benchmark, and initial condition. No "
        "complete-cycle runner exists here and zero cycle steps were executed.\n",
        encoding="utf-8",
    )
    sources = (THIS_RUNNER, THIS_TEST, REPORT_RELATIVE)
    utility._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "implementation_commit": utility._git("rev-parse", "HEAD"),
            "inventory_scope": inventory["inventory_scope"],
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
