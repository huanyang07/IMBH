#!/usr/bin/env python3
"""Freeze the final complete-cycle pre-execution architecture and blockers."""

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

import run_causal_inner_bounded_ap_coarse_trajectory_kernel_wp10c9d6c7c3b5c4f25fizzm1 as parent  # noqa: E402


WORK_PACKAGE = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizzn_"
    "complete_cycle_preexecution_manifest"
)
CLASSIFICATION = "complete_cycle_preexecution_architecture_frozen_inputs_blocked"
ARTIFACT = "causal_inner_complete_cycle_preexecution_manifest_wp10c9d6c7c3b5c4f25fizzn"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = "docs/reports/current/CODEX_CAUSAL_INNER_COMPLETE_CYCLE_PREEXECUTION_MANIFEST_WP10C9D6C7C3B5C4F25FIZZN_2026-08-26.md"
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = "scripts/run_causal_inner_complete_cycle_preexecution_manifest_wp10c9d6c7c3b5c4f25fizzn.py"
THIS_TEST = "tests/test_causal_inner_complete_cycle_preexecution_manifest_wp10c9d6c7c3b5c4f25fizzn.py"
PARENT_SHA256 = "d5ddb6c74502d62ecc1d4667c571d3e6a3b3f18f2042c46b50f5ae3868ecde52"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
REQUIRED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizzo_cycle_wide_offline_atlas_"
    "boundary_event_acquisition_and_global_dry_run"
)


def _u():
    return parent._u()


def _validate_parent(require_clean=False):
    utility = _u()
    if utility._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != PARENT_SHA256:
        raise RuntimeError("bounded AP trajectory certificate checksum changed")
    hashes = utility._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utility._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = utility._read_json(parent.CANONICAL_DIRECTORY / "trajectory_metrics.json")
    if (
        not summary["passed"]
        or not summary["bounded_AP_coarse_trajectory_certified"]
        or not summary["arbitrary_step_restart_certified"]
        or not summary["online_truth_call_free"]
        or not summary["complete_cycle_preexecution_manifest_authorized"]
        or summary["cycle_wide_coefficient_atlas_complete"]
        or summary["complete_cycle_execution_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or metrics["online_truth_calls"] != 0
    ):
        raise RuntimeError("bounded AP trajectory classification changed")
    if require_clean and utility._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("complete-cycle preexecution manifest needs a clean tracked tree")
    return hashes, metrics


def _contract(parent_metrics):
    local_seconds = float(
        max(row["median_online_step_wall_seconds"] for row in parent_metrics["rows"])
        * 100000.0
    )
    certified = {
        "physical_four_current_entropy_congruence": True,
        "Kerr_Schild_Valencia_characteristics": True,
        "eleven_field_symmetric_port_and_dissipative_source": True,
        "exponential_AP_stiff_limit": True,
        "conservative_entropy_projection_microstep": True,
        "bounded_two_anchor_AP_trajectory": True,
        "second_order_matched_refinement": True,
        "arbitrary_step_bitwise_restart": True,
        "zero_online_truth_calls": True,
    }
    missing = {
        "cycle_wide_physical_coefficient_atlas": True,
        "production_radial_boundary_port_maps": True,
        "hot_exit_guard_and_conservative_reset": True,
        "impact_impulse_map": True,
        "full_cycle_slow_forcing_and_loading": True,
        "global_spatial_exponential_action_benchmark": True,
        "held_out_full_cycle_or_phase_window_truth": True,
    }
    return {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "selected_mathematical_architecture": {
            "name": "hybrid conservative entropy-port AP slow-cycle solver",
            "online_state": (
                "z in R^(11*N_R), retained slow invariants q=Cz, orbital phase s, "
                "and a discrete physical mode m"
            ),
            "offline_atlas": (
                "hash-locked maps (q,s,m,R)->(H,A_port,S_port,b,trust_radius,guards,resets)"
            ),
            "global_fast_operator": (
                "L(q,s,m)=-D_R^ES*A_port+S_port with physical boundary ports; "
                "D_R^ES is the certified entropy-stable conservative spatial operator"
            ),
            "macrostep": (
                "second-order commutator-free exponential/phi action for L and b, "
                "followed by the conservative nonlinear entropy projection"
            ),
            "AP_rule": (
                "neutral and weak modes remain explicit; only spectrally separated "
                "stable coordinates are exponentially slaved, never microstepped"
            ),
            "hybrid_events": (
                "dense-output guard localization followed by hash-locked conservative "
                "reset maps for impact and hot-exit transitions"
            ),
            "online_prohibitions": {
                "truth_residual_calls": 0,
                "fixed_Q_microsteps": 0,
                "large_nonlinear_roots": 0,
                "unlocked_extrapolation": 0,
            },
        },
        "cycle_execution_budget": {
            "physical_cycle_seconds": 578880.0,
            "maximum_online_macrosteps": 100000,
            "minimum_average_macrostep_seconds": 5.7888,
            "maximum_wall_days": 3.0,
            "certified_local_100k_step_projection_seconds": local_seconds,
            "global_projection_status": "not certified until production N_R dry run",
        },
        "adaptive_step_controller": {
            "accepted_error_estimator": "one macrostep versus two half macrosteps",
            "reject_on": [
                "atlas trust-domain exit",
                "entropy/conservation gate failure",
                "loss of source spectral gap",
                "unlocalized event guard",
                "boundary-port admissibility failure",
                "restart/checksum mismatch",
            ],
            "maximum_step_ratio": 2.0,
            "event_steps_may_be_shorter_than_cycle_average": True,
        },
        "offline_acquisition_contract": {
            "phase_coverage": "full [0,2*pi] in every discrete physical mode",
            "state_coverage": "adaptive overlap cover of every accepted q trajectory tube",
            "required_payload_per_anchor": [
                "physical state and geometry",
                "entropy metric and congruence",
                "A_port and S_port",
                "slow forcing b",
                "trust radius and spectral gap",
                "physical guard margins",
            ],
            "required_events": ["impact", "hot_exit", "mode_entry", "mode_exit"],
            "required_boundaries": ["inner_excision", "outer_loading_or_outflow"],
            "held_out_validation": (
                "prospectively withheld anchors and phase windows must pass state, "
                "reaction/port action, conservation, entropy, and event-time gates"
            ),
            "all_payloads_hash_locked_before_cycle_run": True,
        },
        "preexecution_readiness": {
            "certified_components": certified,
            "missing_binding_inputs": missing,
            "architecture_complete": True,
            "inputs_complete": False,
            "global_dry_run_complete": False,
            "complete_cycle_execution_ready": False,
        },
        "decision": {
            "status": "blocked_before_complete_cycle_execution",
            "reason": (
                "the online AP engine is locally certified and cost-feasible, but the "
                "cycle-wide physical atlas, boundaries, forcing, and event reset data do not exist"
            ),
            "required_next_artifact": REQUIRED_NEXT,
            "complete_cycle_runner_must_not_exist_yet": True,
        },
        "claim_boundary": {
            "local_mathematical_architecture_verified": True,
            "predictive_complete_cycle_model_verified": False,
            "complete_cycle_execution_authorized": False,
            "complete_cycle_steps": 0,
        },
        "authorized_next": None,
    }


def _update_catalog(summary):
    utility = _u()
    rows = list(csv.DictReader(CANONICAL_MANIFEST.open(newline="", encoding="utf-8")))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": utility._sha256(path), "scientific_status": "SUPPORTED"})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    catalog = utility._read_json(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": CLASSIFICATION, "passed": True}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": utility._git("rev-parse", "HEAD"), "latest_work_package": WORK_PACKAGE})
    utility._write_json(CANONICAL_SUMMARY, catalog)


def _freeze():
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("complete-cycle preexecution manifest exists")
    hashes, parent_metrics = _validate_parent(require_clean=True); utility = _u()
    contract = _contract(parent_metrics)
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utility._write_json(CANONICAL_DIRECTORY / "preexecution_contract.json", contract)
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "mathematical_architecture_selected": True,
        "local_AP_engine_certified": True,
        "cycle_wide_inputs_complete": False,
        "global_dry_run_complete": False,
        "complete_cycle_execution_ready": False,
        "complete_cycle_execution_authorized": False,
        "complete_cycle_steps": 0,
        "required_next_artifact": REQUIRED_NEXT,
        "authorized_next": None,
    }
    utility._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utility._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"parent_artifact": parent.ARTIFACT, "parent_checksum_manifest_sha256": PARENT_SHA256, "parent_hashes": hashes})
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# Complete-cycle pre-execution architecture\n\n"
        f"Classification: `{CLASSIFICATION}`.\n\n"
        "The selected solver is a hybrid conservative entropy-port AP system: a hash-locked physical atlas supplies the global eleven-field entropy operator, a second-order exponential/phi macrostep removes the fast CFL restriction, and conservative entropy projection plus event-localized reset maps preserve the physical ledgers. The local physical engine, stiff limit, second-order trajectory, restart, and zero-truth-call cost model are certified.\n\n"
        "Complete-cycle execution is deliberately blocked. A cycle-wide coefficient/forcing atlas, production boundary ports, impact and hot-exit reset maps, a production-size global exponential-action benchmark, and held-out cycle-window truth are missing. Four local physical anchors cannot be extrapolated into a predictive 6.7-day orbit. No complete-cycle runner or step was created.\n\n"
        f"Required next artifact: `{REQUIRED_NEXT}`.\n",
        encoding="utf-8",
    )
    sources = (THIS_RUNNER, THIS_TEST, REPORT_RELATIVE)
    utility._write_json(CANONICAL_DIRECTORY / "provenance.json", {"implementation_commit": utility._git("rev-parse", "HEAD"), "source_hashes": {source: utility._sha256(ROOT / source) for source in sources}})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utility._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8")
    _update_catalog(summary); return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--freeze", action="store_true"); arguments = parser.parse_args()
    if not arguments.freeze: parser.error("choose --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
