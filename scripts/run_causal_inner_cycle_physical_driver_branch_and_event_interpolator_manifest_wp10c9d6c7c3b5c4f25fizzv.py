#!/usr/bin/env python3
"""Freeze the structure-preserving cycle driver/branch/event interpolator."""

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

import run_causal_inner_nonperiodic_native_global_ap_boundary_action_structure_certificate_wp10c9d6c7c3b5c4f25fizzu1 as parent  # noqa: E402


WORK_PACKAGE = parent.AUTHORIZED_NEXT
CLASSIFICATION = "cycle_physical_driver_branch_and_event_interpolator_manifest_frozen"
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizzv1_cycle_physical_driver_branch_and_event_"
    "interpolator_structure_certificate"
)
PASS_NEXT = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizzw_heldout_atlas_and_hybrid_"
    "sequence_validation_manifest"
)
ARTIFACT = (
    "causal_inner_cycle_physical_driver_branch_and_event_interpolator_manifest_"
    "wp10c9d6c7c3b5c4f25fizzv"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_CYCLE_PHYSICAL_DRIVER_BRANCH_AND_"
    "EVENT_INTERPOLATOR_MANIFEST_WP10C9D6C7C3B5C4F25FIZZV_2026-08-27.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_cycle_physical_driver_branch_and_event_"
    "interpolator_manifest_wp10c9d6c7c3b5c4f25fizzv.py"
)
THIS_TEST = (
    "tests/test_causal_inner_cycle_physical_driver_branch_and_event_"
    "interpolator_manifest_wp10c9d6c7c3b5c4f25fizzv.py"
)
PARENT_SHA256 = "ae6b44d84a6a83d9639a19e7f924d29c37046b2555da6bc94df850324f1e974f"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _u():
    return parent._u()


def _validate_parent(*, require_clean: bool = False):
    utility = _u()
    if utility._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != PARENT_SHA256:
        raise RuntimeError("nonperiodic native boundary certificate changed")
    hashes = utility._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utility._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    if (
        not summary["passed"]
        or not summary["nonperiodic_global_AP_boundary_action_certified"]
        or not summary["pure_inner_excision_certified"]
        or not summary["eleven_characteristic_outer_affine_loading_certified"]
        or summary["physical_model_complete"]
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["complete_cycle_execution_authorized"]
        or summary["complete_cycle_steps"] != 0
    ):
        raise RuntimeError("nonperiodic native boundary classification changed")
    if require_clean and utility._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("cycle interpolator manifest needs a clean tracked tree")
    return hashes


def _contract() -> dict:
    return {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "online_hybrid_state": {
            "continuous": "(q,phi) in R^4 x S1, q=Cz",
            "discrete": "physical branch mode m",
            "reconstruction": "z_star(q,phi,m) plus optional certified fast correction",
            "truth_calls_online": 0,
            "large_nonlinear_roots_online": 0,
        },
        "schema_v2_extension": {
            "reason": (
                "schema v1 records event crossing direction but no reduced guard normal; "
                "it cannot define an online guard surface"
            ),
            "driver_additions": {
                "q_simplices5": "precomputed 4-D invariant simplices, five vertices each",
                "q_scales4": "positive frozen coordinate scales",
            },
            "branch_additions": {
                "branch_simplices6": "precomputed mode-pure simplices in local (q4,unwrapped phi)",
                "branch_simplex_mode_index": "one mode per simplex",
                "branch_q_scales4": "positive frozen coordinate scales",
                "branch_phase_scale": "positive frozen local phase scale",
                "common_local_source_null_coordinates": "indices [0,1,2,3] at every radial cell",
            },
            "event_additions": {
                "transition_class_labels": "fixed source-mode/destination-mode/event-class label",
                "reduced_guard_normals5": "normalized gradients with respect to (q4,unwrapped phi)",
                "reduced_guard_offsets": "signed affine offsets at each event truth point",
                "event_simplices6": "class-pure convex simplices on each guard sheet",
                "event_simplex_class_index": "one transition class per simplex",
            },
            "heldout_additions": {
                "withheld_guard_points": "signed values on both sides plus transverse crossing times",
                "withheld_full_sequence": "never used for simplex construction or coefficient fitting",
            },
        },
        "driver_interpolation": {
            "phase": "periodic piecewise-linear interpolation on S1 using two adjacent nodes",
            "invariants": "nonnegative barycentric weights on one precomputed 4-simplex",
            "tensor_weights": "ten nonnegative products summing to one",
            "same_weights_for": [
                "phase rate",
                "slow forcing",
                "distributed and boundary ledger rates",
                "outer incoming characteristic amplitudes",
            ],
            "preserved": ["positive phase rate", "periodicity", "C b=ledger rate"],
            "outside_hull": "fail closed; nearest-neighbor or extrapolated fallback forbidden",
        },
        "branch_interpolation": {
            "locator": "one mode-pure 5-simplex in scaled (q4,unwrapped phi)",
            "weights": "six nonnegative barycentric coordinates summing to one",
            "state": "convex state interpolation followed by z<-z+N(q-Cz)",
            "radial_matrix": "convex interpolation, preserving entropy symmetry",
            "source_matrix": (
                "convex interpolation with the common first-four-coordinate nullspace "
                "verified before use"
            ),
            "forcing_and_guards": "same convex weights",
            "trust": "query distance and simplex diameter must remain inside every contributing trust radius",
            "normal_hyperbolicity": "interpolated fast spectral gap must remain strictly positive",
            "boundary_reaudit": "interpolated inner/outer incoming counts must remain 0/11",
            "outside_hull_or_trust": "fail closed",
        },
        "event_and_reset_interpolation": {
            "guard": "class-pure convex interpolation of reduced guard normals and offsets",
            "orientation": "all contributing normals have positive dot product after frozen orientation",
            "transversality": "grad(g) dot (qdot,phidot) has prospectively frozen nonzero margin",
            "event_map": "convex duration, ledger impulse and ledger-null constitutive jump",
            "reset": "certified weighted reset geometry re-enforces C(z_plus-z_minus)=Delta q_event exactly",
            "destination_mode": "single-valued and fixed within each transition class",
            "outside_guard_sheet": "fail closed",
        },
        "binding_structure_gates": {
            "maximum_weight_sum_defect": 2.0e-13,
            "minimum_barycentric_weight": -2.0e-13,
            "maximum_coordinate_reproduction_defect": 2.0e-12,
            "maximum_invariant_reconstruction_relative_defect": 2.0e-12,
            "maximum_forcing_ledger_relative_defect": 2.0e-12,
            "maximum_radial_symmetry_defect": 2.0e-12,
            "maximum_source_positive_eigenvalue": 2.0e-12,
            "minimum_source_nullity": 4,
            "minimum_fast_spectral_gap_per_second": 0.0,
            "minimum_guard_transversality": 1.0e-8,
            "maximum_event_reset_ledger_relative_defect": 2.0e-12,
            "driver_and_branch_anchor_reproduction_bitwise": True,
            "outside_hull_rejected": True,
            "checkpoint_roundtrip_bitwise": True,
            "complete_cycle_steps": 0,
        },
        "certificate_fixture": {
            "synthetic_structure_only": True,
            "minimum_modes": 2,
            "minimum_driver_q_nodes": 6,
            "minimum_branch_anchors_per_mode": 8,
            "minimum_event_truths_per_class": 8,
            "heldout_points": True,
            "physical_claim": False,
        },
        "scientific_boundary": {
            "physical_payloads_acquired": False,
            "heldout_physical_validation_complete": False,
            "complete_cycle_execution_authorized": False,
            "complete_cycle_steps": 0,
        },
        "decision": {
            "pass_classification": "cycle_driver_branch_event_interpolator_structure_certified_synthetic_fixture_only",
            "failure_classification": "cycle_driver_branch_event_interpolator_structure_failed",
            "pass_authorized_next": PASS_NEXT,
        },
        "authorized_next": AUTHORIZED_NEXT,
    }


def _update(summary: dict) -> None:
    utility = _u(); rows = list(csv.DictReader(CANONICAL_MANIFEST.open(newline="", encoding="utf-8"))); rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file(): rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": utility._sha256(path), "scientific_status": "SUPPORTED"})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    catalog = utility._read_json(CANONICAL_SUMMARY); catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": CLASSIFICATION, "passed": True}; catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": utility._git("rev-parse", "HEAD"), "latest_work_package": WORK_PACKAGE}); utility._write_json(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists(): raise RuntimeError("cycle interpolator manifest already exists")
    hashes = _validate_parent(require_clean=True); utility = _u(); CANONICAL_DIRECTORY.mkdir(parents=True)
    utility._write_json(CANONICAL_DIRECTORY / "interpolator_contract.json", _contract())
    summary = {"schema_version": 1, "work_package": WORK_PACKAGE, "classification": CLASSIFICATION, "passed": True, "definitions_only": True, "schema_v2_guard_geometry_frozen": True, "cycle_interpolator_certified": False, "synthetic_fixture_only": True, "physical_model_complete": False, "physical_payloads_acquired": False, "heldout_physical_validation_complete": False, "complete_cycle_execution_authorized": False, "complete_cycle_steps": 0, "authorized_next": AUTHORIZED_NEXT}
    utility._write_json(CANONICAL_DIRECTORY / "summary.json", summary); utility._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"parent_artifact": parent.ARTIFACT, "parent_checksum_manifest_sha256": PARENT_SHA256, "parent_hashes": hashes})
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True); REPORT_PATH.write_text(
        "# Cycle physical driver, branch, and event interpolator manifest\n\n"
        f"Classification: `{CLASSIFICATION}`.\n\n"
        "The prospective online atlas uses only nonnegative barycentric weights on precomputed, mode-pure simplices. It preserves phase periodicity, invariant ledgers, radial symmetry, dissipative source structure, the common four-dimensional source kernel, and reset ledgers. Queries outside a convex hull or trust region fail closed.\n\n"
        "Schema v1 is insufficient for online events because it lacks a reduced guard gradient. This package prospectively adds oriented five-dimensional guard normals, offsets, and class-pure guard-sheet simplices. It is definitions-only, uses no physical payload, and authorizes no cycle step.\n",
        encoding="utf-8",
    )
    sources = (THIS_RUNNER, THIS_TEST, REPORT_RELATIVE); utility._write_json(CANONICAL_DIRECTORY / "provenance.json", {"implementation_commit": utility._git("rev-parse", "HEAD"), "source_hashes": {name: utility._sha256(ROOT / name) for name in sources}})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()); (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utility._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8"); _update(summary); return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--freeze", action="store_true"); args = parser.parse_args()
    if not args.freeze: parser.error("choose --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
