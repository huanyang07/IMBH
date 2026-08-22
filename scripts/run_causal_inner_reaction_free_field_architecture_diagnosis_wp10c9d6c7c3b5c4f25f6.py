#!/usr/bin/env python3
"""Select the physical free-field ROM or retain fixed-Q as a physical phase.

The fixed-Q equations remain valid constrained equations.  This package asks
the separate modelling question that matters for a reduced physical cycle:
does their reaction merely remove slow drift, or does it create the tangent
motion that was being interpreted as a fast phase?  Only committed cold-branch
rates and committed full-model secants are used; no new rate, root, or BDF
step is evaluated.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_arclength_segment_wp10c9d6c7c3b5c4f25f5 as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25f6"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25f7"
CLASSIFICATION = (
    "conservative_free_field_hidden_amplitude_rom_selected_"
    "fixed_Q_arclength_retained_sampling_only"
)
FAIL_CLASSIFICATION = "reaction_free_field_architecture_diagnosis_rejected"
ARTIFACT = (
    "causal_inner_reaction_free_field_architecture_diagnosis_"
    "wp10c9d6c7c3b5c4f25f6"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_REACTION_FREE_FIELD_"
    "ARCHITECTURE_DIAGNOSIS_WP10C9D6C7C3B5C4F25F6_2026-08-22.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_reaction_free_field_architecture_diagnosis_"
    "wp10c9d6c7c3b5c4f25f6.py"
)
THIS_TEST = (
    "tests/test_causal_inner_reaction_free_field_architecture_diagnosis_"
    "wp10c9d6c7c3b5c4f25f6.py"
)

COLD_DIRECTORY = ROOT / (
    "results/canonical/causal_inner_cold_branch_anchor_preflight_"
    "wp10c9d6c7c3b5c4f25dy"
)
ARCHITECTURE_DIRECTORY = ROOT / (
    "results/canonical/causal_inner_hybrid_phase_memory_architecture_"
    "selection_wp10c9d6c7c3b5c4f25e1_v2"
)
PRIMARY_DIRECTORY = ROOT / (
    "results/canonical/causal_inner_primary_hidden_anchor_preflight_"
    "wp10c9d6c7c3b5c4f25dg"
)
ARCLENGTH_DIRECTORY = parent.CANONICAL_DIRECTORY

LABELS = ("12ms", "08ms", "05ms", "02ms")
MAXIMUM_ADDITIVE_CLOSURE_DEFECT = 5.0e-13
MAXIMUM_COORDINATE_RATE_PARITY_DEFECT = 5.0e-12
MAXIMUM_FREE_TO_FIXED_RATE_RATIO = 5.0e-4
MINIMUM_REACTION_TO_FIXED_RATE_RATIO = 0.999
MINIMUM_REACTION_FIXED_DIRECTION_COSINE = 0.999
MINIMUM_REACTION_TANGENT_PROJECTION_FRACTION = 0.99
MAXIMUM_FREE_PHYSICAL_SUBSPACE_DEFECT = 1.0e-2
MINIMUM_FIXED_PHYSICAL_SUBSPACE_DEFECT = 0.95
MAXIMUM_NEAREST_PHYSICAL_SECANT_RELATIVE_DEFECT = 5.0e-2
MINIMUM_RANK_TWO_DIRECTION_ENERGY = 0.9999


def _helper():
    return parent._helper()


def _validate_inputs(*, require_clean: bool) -> dict:
    helper = _helper()
    hashes = {
        "cold_anchor": helper._validate_checksums(COLD_DIRECTORY),
        "hybrid_architecture": helper._validate_checksums(ARCHITECTURE_DIRECTORY),
        "primary_anchor": helper._validate_checksums(PRIMARY_DIRECTORY),
        "arclength_segment": helper._validate_checksums(ARCLENGTH_DIRECTORY),
    }
    cold = helper._read(COLD_DIRECTORY / "summary.json")
    architecture = helper._read(ARCHITECTURE_DIRECTORY / "summary.json")
    primary = helper._read(PRIMARY_DIRECTORY / "summary.json")
    arclength = helper._read(ARCLENGTH_DIRECTORY / "summary.json")
    if cold["classification"] != "saved_cold_candidates_not_near_fixed_macro_critical_manifold":
        raise RuntimeError("cold reaction/free-rate evidence changed")
    if not architecture["passed"] or not architecture[
        "hybrid_phase_memory_architecture_selected"
    ]:
        raise RuntimeError("prior hybrid architecture evidence changed")
    if primary["classification"] != "primary_anchor_not_near_frozen_macro_critical_manifold_root_not_attempted":
        raise RuntimeError("primary fixed-Q anchor evidence changed")
    if not arclength["passed"] or arclength["work_package"] != parent.WORK_PACKAGE:
        raise RuntimeError("first arclength segment changed")
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("reaction/free-field diagnosis requires a clean tracked tree")
    return {
        "input_hashes": hashes,
        "parent_classifications": {
            "cold": cold["classification"],
            "hybrid": architecture["classification"],
            "primary": primary["classification"],
            "arclength": arclength["classification"],
        },
    }


def _unit_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= np.finfo(float).tiny):
        raise RuntimeError("zero rate in reaction/free-field diagnosis")
    return array / norms[:, None]


def _rank_two_basis(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unit = _unit_rows(values)
    _left, singular, right = np.linalg.svd(unit, full_matrices=False)
    energy = np.cumsum(singular * singular) / np.sum(singular * singular)
    return np.asarray(right[:2].T), singular, energy


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(left) - np.asarray(right))
        / max(
            float(np.linalg.norm(left)),
            float(np.linalg.norm(right)),
            np.finfo(float).tiny,
        )
    )


def _evaluate(locked: dict) -> tuple[dict, dict[str, np.ndarray], dict]:
    helper = _helper()
    cold = helper._load_npz(COLD_DIRECTORY / "cold_anchor_arrays.npz")
    old = helper._load_npz(ARCHITECTURE_DIRECTORY / "architecture_arrays.npz")
    primary = helper._load_npz(PRIMARY_DIRECTORY / "primary_anchor_rate_arrays.npz")
    physical_secants = np.asarray(old["cold_full_model_secants5x470_per_s"])
    physical_basis, physical_singular, physical_energy = _rank_two_basis(
        physical_secants
    )

    records = []
    fixed_coordinate_rates = []
    free_coordinate_rates = []
    reaction_coordinate_rates = []
    for label in LABELS:
        prefix = f"candidate_{label}__"
        jacobian = np.asarray(cold[prefix + "coordinate_jacobian470x560"])
        fixed = np.asarray(cold[prefix + "scaled_fixed_Q_rate560_per_s"])
        free = np.asarray(cold[prefix + "scaled_free_rate560_per_s"])
        reaction = np.asarray(cold[prefix + "scaled_reaction_action560_per_s"])
        coordinate_fixed = jacobian @ fixed
        coordinate_free = jacobian @ free
        coordinate_reaction = jacobian @ reaction
        saved_coordinate = np.asarray(cold[prefix + "coordinate_rate470_per_s"])
        fixed_norm = float(np.linalg.norm(coordinate_fixed))
        reaction_norm = float(np.linalg.norm(coordinate_reaction))
        free_unit = coordinate_free / np.linalg.norm(coordinate_free)
        fixed_unit = coordinate_fixed / fixed_norm
        free_normal = float(
            np.linalg.norm(free_unit - physical_basis @ (physical_basis.T @ free_unit))
        )
        fixed_normal = float(
            np.linalg.norm(
                fixed_unit - physical_basis @ (physical_basis.T @ fixed_unit)
            )
        )
        secant_relative = np.asarray(
            [_relative(coordinate_free, secant) for secant in physical_secants]
        )
        secant_cosines = _unit_rows(physical_secants) @ free_unit
        records.append({
            "label": label,
            "state_additive_closure_relative_defect": _relative(
                fixed, free + reaction
            ),
            "coordinate_additive_closure_relative_defect": _relative(
                coordinate_fixed, coordinate_free + coordinate_reaction
            ),
            "coordinate_rate_parity_relative_defect": _relative(
                coordinate_fixed, saved_coordinate
            ),
            "state_free_to_fixed_rate_ratio": float(
                np.linalg.norm(free) / np.linalg.norm(fixed)
            ),
            "coordinate_free_to_fixed_rate_ratio": float(
                np.linalg.norm(coordinate_free) / fixed_norm
            ),
            "coordinate_reaction_to_fixed_rate_ratio": reaction_norm / fixed_norm,
            "coordinate_reaction_fixed_direction_cosine": float(
                coordinate_reaction @ coordinate_fixed / (reaction_norm * fixed_norm)
            ),
            "reaction_projection_on_fixed_tangent_fraction": float(
                abs(coordinate_reaction @ coordinate_fixed) / (fixed_norm * fixed_norm)
            ),
            "free_physical_rank_two_subspace_defect": free_normal,
            "fixed_Q_physical_rank_two_subspace_defect": fixed_normal,
            "nearest_physical_secant_relative_defect": float(
                np.min(secant_relative)
            ),
            "nearest_physical_secant_direction_cosine": float(
                np.max(secant_cosines)
            ),
            "coordinate_free_rate_norm_per_second": float(
                np.linalg.norm(coordinate_free)
            ),
            "coordinate_fixed_Q_rate_norm_per_second": fixed_norm,
        })
        fixed_coordinate_rates.append(coordinate_fixed)
        free_coordinate_rates.append(coordinate_free)
        reaction_coordinate_rates.append(coordinate_reaction)

    fixed_coordinate_rates = np.asarray(fixed_coordinate_rates)
    free_coordinate_rates = np.asarray(free_coordinate_rates)
    reaction_coordinate_rates = np.asarray(reaction_coordinate_rates)
    free_basis, free_singular, free_energy = _rank_two_basis(free_coordinate_rates)
    primary_ratio = float(
        np.linalg.norm(primary["continuous_scaled_free_rate_per_s"])
        / np.linalg.norm(primary["continuous_scaled_fixed_Q_rate_per_s"])
    )

    maxima = {
        "maximum_state_additive_closure_relative_defect": max(
            item["state_additive_closure_relative_defect"] for item in records
        ),
        "maximum_coordinate_additive_closure_relative_defect": max(
            item["coordinate_additive_closure_relative_defect"] for item in records
        ),
        "maximum_coordinate_rate_parity_relative_defect": max(
            item["coordinate_rate_parity_relative_defect"] for item in records
        ),
        "maximum_state_free_to_fixed_rate_ratio": max(
            item["state_free_to_fixed_rate_ratio"] for item in records
        ),
        "maximum_coordinate_free_to_fixed_rate_ratio": max(
            item["coordinate_free_to_fixed_rate_ratio"] for item in records
        ),
        "minimum_coordinate_reaction_to_fixed_rate_ratio": min(
            item["coordinate_reaction_to_fixed_rate_ratio"] for item in records
        ),
        "minimum_coordinate_reaction_fixed_direction_cosine": min(
            item["coordinate_reaction_fixed_direction_cosine"] for item in records
        ),
        "minimum_reaction_projection_on_fixed_tangent_fraction": min(
            item["reaction_projection_on_fixed_tangent_fraction"] for item in records
        ),
        "maximum_free_physical_rank_two_subspace_defect": max(
            item["free_physical_rank_two_subspace_defect"] for item in records
        ),
        "minimum_fixed_Q_physical_rank_two_subspace_defect": min(
            item["fixed_Q_physical_rank_two_subspace_defect"] for item in records
        ),
        "maximum_nearest_physical_secant_relative_defect": max(
            item["nearest_physical_secant_relative_defect"] for item in records
        ),
        "minimum_nearest_physical_secant_direction_cosine": min(
            item["nearest_physical_secant_direction_cosine"] for item in records
        ),
        "free_direction_rank_two_energy": float(free_energy[1]),
        "physical_secant_direction_rank_two_energy": float(physical_energy[1]),
        "primary_20ms_state_free_to_fixed_rate_ratio": primary_ratio,
    }
    gates = {
        "state_additive_closure": maxima[
            "maximum_state_additive_closure_relative_defect"
        ] <= MAXIMUM_ADDITIVE_CLOSURE_DEFECT,
        "coordinate_additive_closure": maxima[
            "maximum_coordinate_additive_closure_relative_defect"
        ] <= MAXIMUM_ADDITIVE_CLOSURE_DEFECT,
        "coordinate_rate_parity": maxima[
            "maximum_coordinate_rate_parity_relative_defect"
        ] <= MAXIMUM_COORDINATE_RATE_PARITY_DEFECT,
        "free_rate_is_not_fixed_Q_rate": maxima[
            "maximum_coordinate_free_to_fixed_rate_ratio"
        ] <= MAXIMUM_FREE_TO_FIXED_RATE_RATIO,
        "reaction_dominates_fixed_Q_rate": maxima[
            "minimum_coordinate_reaction_to_fixed_rate_ratio"
        ] >= MINIMUM_REACTION_TO_FIXED_RATE_RATIO,
        "reaction_aligned_with_fixed_Q_tangent": maxima[
            "minimum_coordinate_reaction_fixed_direction_cosine"
        ] >= MINIMUM_REACTION_FIXED_DIRECTION_COSINE,
        "reaction_has_material_tangent_projection": maxima[
            "minimum_reaction_projection_on_fixed_tangent_fraction"
        ] >= MINIMUM_REACTION_TANGENT_PROJECTION_FRACTION,
        "free_rate_matches_physical_subspace": maxima[
            "maximum_free_physical_rank_two_subspace_defect"
        ] <= MAXIMUM_FREE_PHYSICAL_SUBSPACE_DEFECT,
        "fixed_Q_rate_rejected_by_physical_subspace": maxima[
            "minimum_fixed_Q_physical_rank_two_subspace_defect"
        ] >= MINIMUM_FIXED_PHYSICAL_SUBSPACE_DEFECT,
        "free_rate_matches_nearby_physical_secants": maxima[
            "maximum_nearest_physical_secant_relative_defect"
        ] <= MAXIMUM_NEAREST_PHYSICAL_SECANT_RELATIVE_DEFECT,
        "free_direction_is_rank_two": maxima["free_direction_rank_two_energy"]
        >= MINIMUM_RANK_TWO_DIRECTION_ENERGY,
        "physical_secants_are_rank_two": maxima[
            "physical_secant_direction_rank_two_energy"
        ] >= MINIMUM_RANK_TWO_DIRECTION_ENERGY,
        "primary_anchor_corroborates_reaction_dominance": primary_ratio
        <= MAXIMUM_FREE_TO_FIXED_RATE_RATIO,
        "no_new_truth_roots_or_microsteps": True,
    }
    passed = bool(all(gates.values()))
    classification = CLASSIFICATION if passed else FAIL_CLASSIFICATION
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "gates": gates,
        "gate_values": maxima,
        "records": records,
        "new_exact_rate_calls": 0,
        "new_nonlinear_fixed_Q_roots": 0,
        "new_BDF_microsteps": 0,
        "input_lock": locked,
    }
    arrays = {
        "physical_full_model_secants5x470_per_s": physical_secants,
        "physical_rank_two_basis470x2": physical_basis,
        "physical_direction_singular_values": physical_singular,
        "physical_direction_cumulative_energy": physical_energy,
        "free_coordinate_rates4x470_per_s": free_coordinate_rates,
        "fixed_Q_coordinate_rates4x470_per_s": fixed_coordinate_rates,
        "reaction_coordinate_actions4x470_per_s": reaction_coordinate_rates,
        "free_rank_two_basis470x2": free_basis,
        "free_direction_singular_values": free_singular,
        "free_direction_cumulative_energy": free_energy,
    }
    architecture = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "prior_fixed_Q_certificates_preserved": True,
        "fixed_Q_interpretation": {
            "constrained_residual_and_jacobian_remain_valid": True,
            "physical_phase_clock_rejected": passed,
            "arclength_retained_use": "offline_state_space_sampling_only",
            "reason": (
                "the imposed reaction is aligned with the constrained tangent "
                "and supplies essentially the entire constrained rate, while "
                "the original free rate matches accepted full-model secants"
            ),
        },
        "selected_online_state": (
            "x=(q in R^82, hidden amplitudes a in R^r, forcing phase theta, "
            "discrete mode sigma)"
        ),
        "exact_coordinate_split": "y=L q + Z(h0_sigma + V_sigma a)",
        "physical_vector_field": "r_free(y,t)=Dchi(u) f_free(u,t)",
        "reduced_equations": {
            "macro": "dq/dt=R r_hat_free(q,a,theta,sigma)",
            "hidden": "da/dt=V_sigma^T Q r_hat_free(q,a,theta,sigma)",
            "forcing": "dtheta/dt=Omega_forcing",
            "discarded_hidden_gate": (
                "||(I-VV^T)Q r_free||/||Q r_free|| must remain below its "
                "prospectively frozen tolerance"
            ),
        },
        "events": (
            "bracket g_sigma_to_tau(q,a,theta)=0, refine in the reduced field, "
            "and apply a conservative macro reset"
        ),
        "offline_sampling": (
            "adaptive exact free-rate witnesses; constrained arclength may "
            "propose states but supplies neither physical time nor drift"
        ),
        "online_forbidden": (
            "fixed-Q reaction, monolithic truth calls, nonlinear roots, "
            "and nanosecond BDF microsteps"
        ),
        "next_verification": (
            "prospective hot-state exact free-coordinate-rate and conservative "
            "hidden-amplitude ROM preflight"
        ),
    }
    return metrics, arrays, architecture


def _update_catalog(summary: dict) -> None:
    helper = _helper()
    cold = parent._source()._post().manifest.transition.manifest.cold.manifest
    with cold.CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    status = "SUPPORTED" if summary["passed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({
                "case": ARTIFACT,
                "path": str(path.relative_to(ROOT)),
                "bytes": str(path.stat().st_size),
                "sha256": helper._sha(path),
                "scientific_status": status,
            })
    with cold.CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case", "path", "bytes", "sha256", "scientific_status"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = helper._read(cold.CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": summary["passed"],
    }
    catalog.update({
        "case_count": len({row["case"] for row in rows}),
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "all_payload_hashes_recorded": True,
        "latest_source_parent_commit": helper._git("rev-parse", "HEAD"),
        "latest_work_package": WORK_PACKAGE,
    })
    helper._write_json(cold.CANONICAL_SUMMARY, catalog)


def _run() -> dict:
    helper = _helper()
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("reaction/free-field diagnosis already exists")
    locked = _validate_inputs(require_clean=True)
    metrics, arrays, architecture = _evaluate(locked)
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "reaction_free_field_metrics.json", metrics)
    helper._write_json(CANONICAL_DIRECTORY / "mathematical_architecture.json", architecture)
    helper._write_json(CANONICAL_DIRECTORY / "input_lock.json", locked)
    with (CANONICAL_DIRECTORY / "reaction_free_field_arrays.npz").open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": metrics["passed"],
        "fixed_Q_physical_phase_authorized": False,
        "fixed_Q_arclength_sampling_only": metrics["passed"],
        "conservative_free_field_hidden_amplitude_rom_selected": metrics["passed"],
        "hot_free_field_preflight_manifest_authorized": metrics["passed"],
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT if metrics["passed"] else None,
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    helper._write_json(CANONICAL_DIRECTORY / "provenance.json", {
        "runner": THIS_RUNNER,
        "test": THIS_TEST,
        "implementation_commit": helper._git("rev-parse", "HEAD"),
        "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"),
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
    })
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{helper._sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names
        ),
        encoding="utf-8",
    )
    values = metrics["gate_values"]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join((
            "# Reaction/free-field mathematical architecture diagnosis",
            "",
            f"Classification: `{metrics['classification']}`.",
            "",
            "The fixed-Q equations and their numerical certificates remain valid, but their constrained trajectory is not accepted as the physical fast-time clock. Across four committed cold anchors, the physical free coordinate rate is at most "
            f"`{values['maximum_coordinate_free_to_fixed_rate_ratio']:.6e}` of the constrained rate. The imposed reaction projects onto the fixed-Q tangent by at least `{values['minimum_reaction_projection_on_fixed_tangent_fraction']:.6e}`.",
            "",
            "The free rate lies within "
            f"`{values['maximum_free_physical_rank_two_subspace_defect']:.6e}` of the accepted full-model rank-two secant subspace, while the fixed-Q rate remains at least `{values['minimum_fixed_Q_physical_rank_two_subspace_defect']:.6e}` outside it. The worst nearest full-model secant defect is `{values['maximum_nearest_physical_secant_relative_defect']:.6e}`.",
            "",
            "Select the exact conservative split `y=Lq+Z(h0+Va)` driven by the original free field. Fixed-Q arclength remains useful only to propose offline sample states; it supplies neither physical duration nor reduced drift. No new truth rate, root, or BDF microstep was executed.",
            "",
        )),
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
    print(json.dumps(_run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
