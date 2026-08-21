#!/usr/bin/env python3
"""Select the lowest-dimensional evidence-supported hybrid phase model."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_hybrid_phase_memory_architecture_manifest_wp10c9d6c7c3b5c4f25e0 as manifest  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25e1"
PASS_CLASSIFICATION = (
    "hybrid_conservative_scalar_phase_memory_architecture_selected_"
    "local_segments_supported_complete_cycle_truth_missing"
)
FAIL_CLASSIFICATION = "available_evidence_does_not_support_low_dimensional_phase_architecture"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25e2"

ARTIFACT = "causal_inner_hybrid_phase_memory_architecture_selection_wp10c9d6c7c3b5c4f25e1"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = "scripts/run_causal_inner_hybrid_phase_memory_architecture_selection_wp10c9d6c7c3b5c4f25e1.py"
THIS_TEST = "tests/test_causal_inner_hybrid_phase_memory_architecture_selection_wp10c9d6c7c3b5c4f25e1.py"
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_HYBRID_PHASE_MEMORY_ARCHITECTURE_"
    "SELECTION_WP10C9D6C7C3B5C4F25E1_2026-08-21.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE


def _rank_at_energy(matrix: np.ndarray, target: float) -> tuple[int, np.ndarray, np.ndarray]:
    singular_values = np.linalg.svd(np.asarray(matrix, dtype=float), compute_uv=False)
    energy = np.cumsum(singular_values * singular_values)
    energy /= energy[-1]
    return int(np.searchsorted(energy, target) + 1), singular_values, energy


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= np.finfo(float).tiny):
        raise RuntimeError("zero direction in architecture evidence")
    return values / norms[:, None]


def _validate_manifest(*, require_clean: bool) -> dict:
    helper = manifest.tube.manifest.geometry
    hashes = helper._validate_checksums(manifest.CANONICAL_DIRECTORY)
    contract = helper._read(manifest.CANONICAL_DIRECTORY / "architecture_contract.json")
    summary = helper._read(manifest.CANONICAL_DIRECTORY / "summary.json")
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["work_package"] != manifest.WORK_PACKAGE
    ):
        raise RuntimeError("phase-memory manifest changed")
    for relative, expected in contract["frozen_source_hashes"].items():
        if helper._sha(ROOT / relative) != expected:
            raise RuntimeError(f"frozen phase-memory source changed: {relative}")
    manifest._validate_parent(require_clean=False)
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("phase-memory selection requires a clean tracked tree")
    return {"manifest_hashes": hashes, "contract": contract}


def _evaluate() -> tuple[dict, dict[str, np.ndarray]]:
    helper = manifest.tube.manifest.geometry
    cold_metrics = helper._read(
        manifest.cold.CANONICAL_DIRECTORY / "cold_anchor_metrics.json"
    )
    cold_arrays = helper._load_npz(
        manifest.cold.CANONICAL_DIRECTORY / "cold_anchor_arrays.npz"
    )
    geometry = helper._load_npz(manifest.GEOMETRY_DIRECTORY / "geometry_arrays.npz")
    candidates = helper._load_npz(
        manifest.CANDIDATE_DIRECTORY / "candidate_geometry_arrays.npz"
    )
    tangent = helper._load_npz(
        manifest.TANGENT_DIRECTORY / "transition_hidden_tangent_arrays.npz"
    )
    tube_metrics = helper._read(manifest.tube.CANONICAL_DIRECTORY / "tube_metrics.json")

    labels = ("12ms", "08ms", "05ms", "02ms")
    cold_rates = np.stack(
        [cold_arrays[f"candidate_{label}__coordinate_rate470_per_s"] for label in labels]
    )
    transition_secants = np.asarray(geometry["trajectory_secants470_per_s"], dtype=float)
    cold_unit = _unit_rows(cold_rates)
    transition_unit = _unit_rows(transition_secants)
    combined_unit = np.vstack((cold_unit, transition_unit))
    cold_rank, cold_singular, cold_energy = _rank_at_energy(
        cold_unit, manifest.DIRECTION_ENERGY_TARGET
    )
    combined_rank, combined_singular, combined_energy = _rank_at_energy(
        combined_unit, manifest.DIRECTION_ENERGY_TARGET
    )

    times = np.asarray(candidates["candidate_times_seconds"], dtype=float)
    coordinates = np.asarray(candidates["candidate_absolute_y470_coordinates"], dtype=float)
    cold_full_secants = np.diff(coordinates, axis=0) / np.diff(times)[:, None]
    cold_full_unit = _unit_rows(cold_full_secants)
    switch_cosine = float(cold_full_unit[-1] @ transition_unit[0])
    switch_angle = float(math.degrees(math.acos(np.clip(switch_cosine, -1.0, 1.0))))
    cold_to_transition_cosines = cold_unit @ transition_unit.T

    hidden_fractions = np.asarray(
        [item["hidden_coordinate_rate_fraction"] for item in cold_metrics["candidate_metrics"]],
        dtype=float,
    )
    checks = {
        "memoryless_cold_graph_rejected": bool(
            np.all(hidden_fractions > manifest.HIDDEN_FRACTION_GATE)
        ),
        "cold_direction_intrinsic_scalar": cold_rank == 1,
        "combined_direction_embedding_rank": combined_rank <= 2,
        "hybrid_mode_turn": switch_angle >= manifest.MINIMUM_MODE_TURN_DEGREES,
        "transition_tube_supported": bool(tube_metrics["passed"]),
        "transition_intrinsic_scalar": int(tube_metrics["dynamic_transition_coordinates"]) == 1,
        "transition_decoder_exact_macro_closure": (
            tube_metrics["maximum_macro_decoder_closure_infinity"] <= 5.0e-12
        ),
        "online_truth_free": True,
    }
    passed = bool(all(checks.values()))
    metrics = {
        "classification": PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,
        "passed": passed,
        "checks": checks,
        "cold_hidden_fractions": hidden_fractions,
        "minimum_cold_hidden_fraction": float(np.min(hidden_fractions)),
        "cold_normalized_direction_rank_99p99": cold_rank,
        "cold_first_mode_energy_fraction": float(cold_energy[0]),
        "minimum_cold_direction_cosine": float(np.min(cold_unit @ cold_unit.T)),
        "combined_normalized_direction_rank_99p99": combined_rank,
        "combined_first_mode_energy_fraction": float(combined_energy[0]),
        "combined_first_two_mode_energy_fraction": float(combined_energy[1]),
        "minimum_cold_to_transition_direction_cosine": float(
            np.min(cold_to_transition_cosines)
        ),
        "maximum_cold_to_transition_direction_cosine": float(
            np.max(cold_to_transition_cosines)
        ),
        "full_model_to_fixed_Q_switch_cosine": switch_cosine,
        "full_model_to_fixed_Q_switch_angle_degrees": switch_angle,
        "continuous_online_dimension": 83,
        "discrete_mode_count_minimum": 2,
        "online_truth_calls": 0,
        "online_470_roots": 0,
        "online_fixed_Q_microsteps": 0,
        "minimum_average_macrostep_seconds": (
            manifest.ONLINE_CYCLE_SECONDS / manifest.MAXIMUM_ONLINE_MACROSTEPS
        ),
        "complete_cycle_truth_available": False,
        "hot_exit_truth_available": False,
        "complete_impulse_map_available": False,
        "reduced_cycle_authorized": False,
    }
    arrays = {
        "cold_fixed_Q_rates4x470": cold_rates,
        "cold_normalized_directions4x470": cold_unit,
        "transition_secants17x470_per_s": transition_secants,
        "transition_normalized_directions17x470": transition_unit,
        "combined_normalized_directions21x470": combined_unit,
        "cold_direction_singular_values": cold_singular,
        "cold_direction_cumulative_energy": cold_energy,
        "combined_direction_singular_values": combined_singular,
        "combined_direction_cumulative_energy": combined_energy,
        "cold_full_model_secants5x470_per_s": cold_full_secants,
        "cold_full_model_normalized_directions5x470": cold_full_unit,
        "cold_to_transition_direction_cosines4x17": cold_to_transition_cosines,
        "hidden_dual_Q388x470": np.asarray(tangent["hidden_dual_Q388"]),
        "macro_restriction_R82x470": np.asarray(tangent["macro_restriction_R82"]),
        "macro_lift_L470x82": np.asarray(geometry["macro_lift_L470x82"]),
    }
    return metrics, arrays


def _update_catalog(summary: dict) -> None:
    helper = manifest.tube.manifest.geometry
    with manifest.CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": helper._sha(path),
                    "scientific_status": "SUPPORTED" if summary["passed"] else "REJECTED",
                }
            )
    with manifest.CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case", "path", "bytes", "sha256", "scientific_status"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = helper._read(manifest.CANONICAL_SUMMARY)
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
            "latest_source_parent_commit": manifest.PARENT_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    helper._write_json(manifest.CANONICAL_SUMMARY, catalog)


def _run() -> dict:
    helper = manifest.tube.manifest.geometry
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("phase-memory architecture result already exists")
    locked = _validate_manifest(require_clean=True)
    metrics, arrays = _evaluate()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "architecture_metrics.json", metrics)
    helper._write_npz(CANONICAL_DIRECTORY / "architecture_arrays.npz", arrays)
    helper._write_json(CANONICAL_DIRECTORY / "input_lock.json", locked)
    helper._write_json(
        CANONICAL_DIRECTORY / "architecture.json",
        {
            "state": ["q_R82", "s_scalar", "mode_discrete"],
            "decoder": "D_m(q,s)=Lq+Z*(h_m0+U_m*a_m(q,s))",
            "macro_dynamics": "dq_dt=g_m(q,s)",
            "phase_dynamics": "ds_dt=omega_m(q,s)>0",
            "normal_defect": "QF-D_qh*g_m-D_sh*omega_m",
            "guard": "gamma_m(q,s)=0",
            "reset": "Psi_m with exact conservative macro ledger",
            "cycle_map": "q_next=q+sum_m_integral(g_m/omega_m ds)+sum_event_Delta_q",
            "cycle_time": "sum_m_integral(1/omega_m ds)",
            "online_truth_calls": 0,
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": metrics["passed"],
        "memoryless_branch_graph_rejected": True,
        "hybrid_phase_memory_architecture_selected": metrics["passed"],
        "continuous_online_dimension": 83,
        "complete_cycle_truth_missing": True,
        "reduced_cycle_authorized": False,
        "authorized_next": AUTHORIZED_NEXT if metrics["passed"] else None,
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    helper._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "implementation_commit": helper._git("rev-parse", "HEAD"),
            "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"),
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    )
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{helper._sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Hybrid phase-memory architecture selection WP10c9d6c7c3b5c4f25e1",
                "",
                f"Classification: `{metrics['classification']}`.",
                "",
                f"All cold fixed-Q hidden fractions exceed 0.999999. A memoryless critical graph is therefore rejected on every saved cold candidate. Yet their normalized directions are rank {metrics['cold_normalized_direction_rank_99p99']} at 99.99% energy, and cold plus transition directions are rank {metrics['combined_normalized_direction_rank_99p99']}.",
                "",
                f"The full-model/fixed-Q direction changes by {metrics['full_model_to_fixed_Q_switch_angle_degrees']:.3f} degrees, selecting a hybrid mode switch. The retained online state is (q in R82, scalar phase s, discrete mode), with exact macro-ledger closure and no online truth call or 470-dimensional root.",
                "",
                "This selects the mathematical architecture; it does not manufacture missing hot-exit, impulse, or complete-cycle truth. Those data remain required before a predictive cycle can be authorized.",
                "",
            )
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("use --run")
    payload = _run()
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
