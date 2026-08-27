#!/usr/bin/env python3
"""Freeze cycle-wide physical-input acquisition and event-reset architecture."""

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
sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))

import run_causal_inner_prefix_port_payload_and_boundary_structure_certificate_wp10c9d6c7c3b5c4f25fizzr1 as parent  # noqa: E402


WORK_PACKAGE = parent.AUTHORIZED_NEXT
CLASSIFICATION = (
    "cycle_wide_missing_input_acquisition_and_conservative_event_reset_"
    "manifest_frozen"
)
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizzs1_conservative_entropy_reset_and_guard_"
    "localization_structure_certificate"
)
PASS_NEXT = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizzt_cycle_wide_physical_driver_"
    "boundary_loading_and_event_truth_acquisition_manifest"
)
ARTIFACT = (
    "causal_inner_cycle_wide_missing_input_acquisition_and_event_reset_manifest_"
    "wp10c9d6c7c3b5c4f25fizzs"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_CYCLE_WIDE_MISSING_INPUT_ACQUISITION_"
    "AND_EVENT_RESET_MANIFEST_WP10C9D6C7C3B5C4F25FIZZS_2026-08-27.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_cycle_wide_missing_input_acquisition_and_event_"
    "reset_manifest_wp10c9d6c7c3b5c4f25fizzs.py"
)
THIS_TEST = (
    "tests/test_causal_inner_cycle_wide_missing_input_acquisition_and_event_"
    "reset_manifest_wp10c9d6c7c3b5c4f25fizzs.py"
)
PARENT_SHA256 = "4b491cbba2440f6106da7ae69c54c494ecaa5c15137f8fd4aa808cf305b3d9c6"
SUPPORT = {
    "causal_inner_complete_cycle_preexecution_manifest_wp10c9d6c7c3b5c4f25fizzn": "e7fe4eae3d8a6d951f25c429566d1bdef402adb49f23df4d01ef7eb9d533afcf",
    "causal_inner_legacy_cycle_evidence_compatibility_audit_wp10c9d6c7c3b5c4f25fizzp1": "d6b1b2c6afe76b56a407e9b2baa0a319b6ec6a111a24668938e253b8cb7e5ac3",
    "causal_inner_adaptive_complete_cycle_execution_wp10c9d6c7c3b5c4f25fe": "0b018b004798f28e5ec5d5f0e70b2bfb26ee6fca0b05a42f53f810246a061aab",
    "causal_inner_transition_terminal_prognosis_wp10c9d6c7c3b5c4f25dw": "8e90aea893909df4f96dae5655e14a009fcc38dafa379ec83a3a7feb5d01cdaf",
    "causal_inner_equilibrium_centered_hybrid_architecture_audit_wp10c9d6c7c3b5c4f25ao": "e4ea8883ca866af6d803d63cbc87136adb77df6096016619675aa97ce8784d40",
    "causal_inner_cycle_map_architecture_decision_wp10c9d6c7c3b5c4f25ec_v2": "383a04434bffc62a72c098f1d570a1d297379437342e8bc1c1bc1686ae0725f7",
}
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _u():
    return parent._u()


def _support_directory(name: str) -> Path:
    return ROOT / "results/canonical" / name


def _validate_parent(require_clean: bool = False):
    utility = _u()
    if utility._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != PARENT_SHA256:
        raise RuntimeError("prefix port/boundary certificate changed")
    parent_hashes = utility._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utility._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    if (
        not summary["passed"]
        or not summary["prefix_port_payloads_built"]
        or not summary["eleven_field_boundary_structure_certified"]
        or summary["outer_cycle_loading_complete"]
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["complete_cycle_execution_authorized"]
    ):
        raise RuntimeError("prefix port/boundary classification changed")
    support_hashes = {}
    for name, expected in SUPPORT.items():
        directory = _support_directory(name)
        if utility._sha256(directory / "SHA256SUMS.txt") != expected:
            raise RuntimeError(f"support evidence changed: {name}")
        support_hashes[name] = utility._validate_checksums(directory)
    old_cycle = utility._read_json(
        _support_directory(
            "causal_inner_adaptive_complete_cycle_execution_wp10c9d6c7c3b5c4f25fe"
        )
        / "cycle_execution_metrics.json"
    )
    prognosis = utility._read_json(
        _support_directory(
            "causal_inner_transition_terminal_prognosis_wp10c9d6c7c3b5c4f25dw"
        )
        / "prognosis_metrics.json"
    )
    if (
        old_cycle["classification"]
        != "complete_cycle_inconclusive_acquisition_budget_exhausted"
        or old_cycle["gate_values"]["cycle_observed"]
        or old_cycle["event"] is not None
        or prognosis["additional_hot_exit_microsteps_authorized"]
    ):
        raise RuntimeError("negative event evidence changed")
    if require_clean and utility._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("cycle input/reset manifest needs a clean tracked tree")
    return parent_hashes, support_hashes, old_cycle, prognosis


def _contract(old_cycle: dict, prognosis: dict) -> dict:
    gate_values = old_cycle["gate_values"]
    return {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "physical_target": {
            "cycle_seconds": 578880.0,
            "maximum_online_macrosteps": 100000,
            "minimum_average_macrostep_seconds": 5.7888,
            "maximum_wall_days": 3.0,
            "online_truth_calls": 0,
        },
        "certified_now": {
            "native_radial_cells": 112,
            "global_state_dimension": 1232,
            "prefix_physical_anchor_payloads": 913,
            "prefix_saved_profiles": 257,
            "prefix_physical_seconds": 0.016,
            "prefix_cycle_fraction": 0.016 / 578880.0,
            "inner_entropy_characteristic_excision": True,
            "outer_entropy_characteristic_operator": True,
            "outer_physical_loading": False,
            "periodic_and_native_size_global_AP_method_proofs": True,
        },
        "binding_negative_evidence": {
            "old_acquisition_wall_seconds": gate_values["execution_wall_seconds"],
            "old_exact_witnesses": gate_values["exact_free_field_witnesses"],
            "old_completed_patches": gate_values["completed_patches"],
            "old_mode_switches": gate_values["mode_switches"],
            "old_cycle_return_observed": False,
            "old_hot_exit_observed": False,
            "continued_hot_exit_microstepping_cost_justified": False,
            "median_old_half_step_root_wall_seconds": prognosis["prognosis"][
                "median_half_step_root_wall_seconds"
            ],
            "interpretation": (
                "local prefix geometry is useful, but uniform microstepping did not identify "
                "a hot exit or a cycle and is not the acquisition architecture"
            ),
        },
        "selected_acquisition_architecture": {
            "slow_branches": (
                "offline pseudo-arclength branch sheets parameterized by retained physical "
                "invariants, orbital phase, and discrete mode"
            ),
            "regular_segments": (
                "adaptive overlap atlas of physical entropy ports, slow forcing, outer "
                "incoming loading, trust radii, spectral gaps, and guard margins"
            ),
            "events": (
                "dense-output guard localization followed by one calibrated conservative "
                "entry-to-exit reset evaluation"
            ),
            "transition_truth": (
                "offline event-to-event solves with full physical ledger integration; never "
                "an online nanosecond ODE"
            ),
            "heldout": (
                "prospectively withheld branch anchors, phase windows, event parameters, and "
                "one full-cycle or complete event-sequence truth trajectory"
            ),
        },
        "conservative_reset_structure": {
            "online_state": "z in R^(112*11) in anchor-local physical entropy coordinates",
            "physical_increment_map": (
                "delta U_cell = diag(conserved_scales) * entropy_root_inverse * z_core"
            ),
            "global_ledger_map": "Delta Q = C z using native cell measures",
            "entropy_weight": "W=blockdiag(cell_measure_i*I_11)",
            "minimum_norm_normal": "N=W^-1 C^T (C W^-1 C^T)^-1",
            "ledger_null_projector": "P=I-NC",
            "reset": "z_plus=z_minus+N DeltaQ_event+P xi_event",
            "identity": "C(z_plus-z_minus)=DeltaQ_event",
            "physical_calibration_missing": [
                "impact/hot-exit guard surfaces and orientation",
                "integrated physical ledger impulse DeltaQ_event",
                "ledger-null constitutive jump xi_event",
                "destination mode and event duration",
            ],
        },
        "event_localization_structure": {
            "dense_state": "cubic Hermite state from endpoint states and rates",
            "root": "bracketed scalar guard root on theta in [0,1]",
            "fail_closed": [
                "unbracketed or tangential event",
                "multiple unresolved guard crossings",
                "event outside atlas trust domain",
                "reset ledger or physical guard failure",
            ],
        },
        "required_physical_inputs": {
            "cycle_wide_slow_forcing_b": False,
            "cycle_wide_outer_incoming_loading": False,
            "impact_guard_and_reset_truth": False,
            "hot_exit_guard_and_reset_truth": False,
            "cooling_recovery_branch_sheet": False,
            "heldout_event_sequence_or_cycle_truth": False,
        },
        "execution_order": [
            "certify conservative entropy reset and dense guard localization structure",
            "freeze physical driver, outer loading, branch, and event truth acquisition",
            "acquire revealed training payloads with prospective heldouts",
            "certify nonperiodic 112-cell global AP action with physical boundary operators",
            "validate matched regular segments and calibrated event resets",
            "freeze final complete-cycle execution manifest and stop before running it",
        ],
        "prohibitions": {
            "continue_old_hot_exit_microstepping": True,
            "infer_cycle_from_prefix": True,
            "manufacture_slow_forcing_or_outer_loading": True,
            "fit_event_reset_without_entry_exit_truth": True,
            "online_truth_or_fixed_Q_calls": True,
            "complete_cycle_runner_or_step": True,
        },
        "claim_boundary": {
            "reset_and_guard_structure_certified": False,
            "cycle_wide_physical_inputs_complete": False,
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
        raise RuntimeError("cycle input/reset manifest exists")
    parent_hashes, support_hashes, old_cycle, prognosis = _validate_parent(
        require_clean=True
    )
    utility = _u()
    contract = _contract(old_cycle, prognosis)
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utility._write_json(CANONICAL_DIRECTORY / "acquisition_and_reset_contract.json", contract)
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "prefix_port_payloads_built": True,
        "eleven_field_boundary_structure_certified": True,
        "reset_and_guard_structure_certified": False,
        "cycle_wide_physical_inputs_complete": False,
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
            "parent_hashes": parent_hashes,
            "support_hashes": support_hashes,
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# Cycle-wide input acquisition and conservative event-reset manifest\n\n"
        f"Classification: `{CLASSIFICATION}`.\n\n"
        "The native 1,232-state AP operator, 913 prefix port payloads, inner excision, "
        "and outer entropy-characteristic structure are certified. The outer edge has "
        "eleven incoming modes, so physical loading is data, not a numerical closure.\n\n"
        f"The old acquisition used `{old_cycle['gate_values']['execution_wall_seconds']:.1f}` "
        f"wall seconds, `{old_cycle['gate_values']['exact_free_field_witnesses']}` witnesses, "
        f"and `{old_cycle['gate_values']['completed_patches']}` patches without observing a "
        "cycle return. Hot-exit microstepping is explicitly not continued.\n\n"
        "The selected path is branch/event driven: acquire slow branch sheets and regular "
        "physical atlas payloads offline, localize events with dense output, and apply one "
        "ledger-exact calibrated reset. The next package certifies this reset/localization "
        "structure only. Physical forcing, outer loading, impact/hot-exit truth, recovery "
        "branches, and held-out cycle evidence remain missing. No cycle step is authorized.\n",
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
