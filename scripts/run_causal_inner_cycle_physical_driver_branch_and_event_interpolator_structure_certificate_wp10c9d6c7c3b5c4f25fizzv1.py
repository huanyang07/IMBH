#!/usr/bin/env python3
"""Certify structure-preserving cycle-atlas interpolation on a fixture."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path: sys.path.insert(0, str(path))

import run_causal_inner_cycle_interpolator_guard_sheet_dimension_correction_manifest_wp10c9d6c7c3b5c4f25fizzv0 as manifest  # noqa: E402
import run_causal_inner_cycle_physical_input_bundle_schema_and_validator_certificate_wp10c9d6c7c3b5c4f25fizzt1 as input_validator  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_cycle_atlas import (  # noqa: E402
    interpolate_cycle_branch,
    interpolate_cycle_driver,
    interpolate_cycle_event,
)


WORK_PACKAGE = manifest.AUTHORIZED_NEXT
PASS_CLASSIFICATION = "cycle_driver_branch_event_interpolator_structure_certified_synthetic_fixture_only"
FAIL_CLASSIFICATION = "cycle_driver_branch_event_interpolator_structure_failed"
AUTHORIZED_NEXT = manifest.PASS_NEXT
ARTIFACT = (
    "causal_inner_cycle_physical_driver_branch_and_event_interpolator_structure_"
    "certificate_wp10c9d6c7c3b5c4f25fizzv1"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_CYCLE_PHYSICAL_DRIVER_BRANCH_AND_"
    "EVENT_INTERPOLATOR_STRUCTURE_CERTIFICATE_WP10C9D6C7C3B5C4F25FIZZV1_2026-08-27.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_cycle_physical_driver_branch_and_event_"
    "interpolator_structure_certificate_wp10c9d6c7c3b5c4f25fizzv1.py"
)
THIS_TEST = (
    "tests/test_causal_inner_cycle_physical_driver_branch_and_event_"
    "interpolator_structure_certificate_wp10c9d6c7c3b5c4f25fizzv1.py"
)
PHYSICAL_SOURCE = "src/imri_qpe/layer3_minidisk_1d/causal_inner_cycle_atlas.py"
PHYSICAL_TEST = "tests/test_causal_inner_cycle_atlas.py"
PARENT_SHA256 = "b61b93f0cee5f66c21031a68336e195873cc2a7e4fc7b9383f818bbd21585ec3"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _u(): return manifest._u()


def _validate_parent(*, require_clean: bool = False):
    utility = _u()
    if utility._sha256(manifest.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != PARENT_SHA256: raise RuntimeError("guard-sheet correction manifest changed")
    hashes = utility._validate_checksums(manifest.CANONICAL_DIRECTORY); summary = utility._read_json(manifest.CANONICAL_DIRECTORY / "summary.json"); correction = utility._read_json(manifest.CANONICAL_DIRECTORY / "corrected_interpolator_contract.json"); original = utility._read_json(manifest.parent.CANONICAL_DIRECTORY / "interpolator_contract.json")
    if not summary["passed"] or not summary["definitions_only"] or not summary["supersedes_prior_interpolator_manifest"] or summary["event_guard_intrinsic_dimension"] != 4 or summary["event_simplex_vertex_count"] != 5 or summary["authorized_next"] != WORK_PACKAGE or summary["complete_cycle_execution_authorized"] or not correction["binding_correction"]["event_simplices6_forbidden"] or original["binding_structure_gates"]["complete_cycle_steps"] != 0: raise RuntimeError("corrected interpolator contract changed")
    if require_clean and utility._git("status", "--short", "--untracked-files=no"): raise RuntimeError("cycle interpolator certificate needs a clean tracked tree")
    return hashes, original, correction


def _geometry(): return input_validator._reset_geometry_arrays()


def _fixture():
    conservation, normal = _geometry(); rng = np.random.default_rng(2026082705)
    q_scales = np.asarray((1.0e-3, 1.2e-3, 0.8e-3, 1.5e-3)); phase_scale = 0.2
    q_nodes = np.zeros((6, 4))
    for index in range(4): q_nodes[index + 1, index] = 0.08 * q_scales[index]
    q_nodes[5] = 0.025 * q_scales
    q_simplices = np.asarray(((0, 1, 2, 3, 4),), dtype=int)
    phases = np.linspace(0.0, 2.0 * np.pi, 5); mode_count = 2; common = (len(phases), len(q_nodes), mode_count)
    base = rng.normal(scale=2e-4, size=common[1:] + (1232,)); cosine = rng.normal(scale=4e-5, size=common[1:] + (1232,)); sine = rng.normal(scale=3e-5, size=common[1:] + (1232,))
    forcing = np.asarray([base + np.cos(value) * cosine + np.sin(value) * sine for value in phases]); forcing[-1] = forcing[0]
    total = np.einsum("as,pqms->pqma", conservation, forcing); distributed = 0.6 * total; boundary = total - distributed
    incoming_base = rng.normal(scale=2e-3, size=common[1:] + (11,)); incoming_cos = rng.normal(scale=3e-4, size=common[1:] + (11,)); incoming = np.asarray([incoming_base + np.cos(value) * incoming_cos for value in phases]); incoming[-1] = incoming[0]
    rates = 1.0e-5 * (1.0 + 0.1 * np.cos(phases)); rates[-1] = rates[0]
    driver = {"phase_nodes": phases, "phase_rate_per_second": rates, "retained_invariant_nodes4": q_nodes, "mode_labels": np.asarray(("cold_fixture", "hot_fixture")), "slow_forcing1232_per_second": forcing, "distributed_source_ledger_rate4": distributed, "boundary_ledger_rate4": boundary, "outer_incoming_characteristics11": incoming}

    per_mode = 8; anchor_count = 2 * per_mode
    anchor_q = np.zeros((anchor_count, 4)); anchor_phase = np.zeros(anchor_count); anchor_mode = np.repeat(np.arange(2), per_mode)
    branch_simplices = []; branch_simplex_modes = []
    for mode in range(2):
        start = mode * per_mode; base_q = (0.01 + 0.015 * mode) * q_scales; base_phase = 1.0 + 3.0 * mode
        anchor_q[start] = base_q; anchor_phase[start] = base_phase
        for coordinate in range(4): anchor_q[start + coordinate + 1] = base_q; anchor_q[start + coordinate + 1, coordinate] += 0.04 * q_scales[coordinate]; anchor_phase[start + coordinate + 1] = base_phase
        anchor_q[start + 5] = base_q; anchor_phase[start + 5] = base_phase + 0.04 * phase_scale
        anchor_q[start + 6] = base_q + 0.01 * q_scales; anchor_phase[start + 6] = base_phase + 0.01 * phase_scale
        anchor_q[start + 7] = base_q + np.asarray((0.015, 0.005, 0.01, 0.02)) * q_scales; anchor_phase[start + 7] = base_phase + 0.015 * phase_scale
        branch_simplices.append(np.arange(start, start + 6)); branch_simplex_modes.append(mode)
    states = []
    for value in anchor_q:
        candidate = rng.normal(scale=2e-4, size=1232); candidate -= normal @ (conservation @ candidate); states.append(candidate + normal @ value)
    radial = np.empty((anchor_count, 112, 11, 11)); source = np.empty_like(radial)
    for anchor in range(anchor_count):
        for cell in range(112):
            radial[anchor, cell] = np.diag(-np.linspace(0.8, 0.3, 11) * (1.0 + 2e-3 * anchor + 1e-4 * cell))
            source[anchor, cell] = np.diag(np.concatenate((np.zeros(4), -np.linspace(0.5, 1.1, 7) * (1.0 + 1e-3 * anchor))))
    branch = {"anchor_states1232": np.asarray(states), "anchor_phase": anchor_phase, "anchor_invariants4": anchor_q, "anchor_mode_index": anchor_mode, "radial_matrices112x11x11": radial, "source_matrices112x11x11": source, "forcing1232_per_second": rng.normal(scale=1e-4, size=(anchor_count, 1232)), "trust_radii": np.full(anchor_count, 0.2), "stable_spectral_gaps_per_second": np.linspace(0.4, 0.7, anchor_count), "guard_margins": np.full((anchor_count, 2), 0.5)}

    event_per_class = 8; event_count = 2 * event_per_class; event_q = np.zeros((event_count, 4)); event_phase = np.zeros(event_count); event_classes = np.repeat(np.arange(2), event_per_class); event_source = np.repeat(np.asarray((0, 1)), event_per_class); event_destination = np.repeat(np.asarray((1, 0)), event_per_class); event_simplices = []; event_simplex_classes = []
    guard_normals = np.zeros((event_count, 5)); guard_normals[:, 4] = 1.0; guard_offsets = np.zeros(event_count)
    for event_class in range(2):
        start = event_class * event_per_class; base_q = (0.012 + 0.014 * event_class) * q_scales; event_phi = 1.5 + 3.0 * event_class
        event_q[start] = base_q; event_phase[start] = event_phi
        for coordinate in range(4): event_q[start + coordinate + 1] = base_q; event_q[start + coordinate + 1, coordinate] += 0.04 * q_scales[coordinate]; event_phase[start + coordinate + 1] = event_phi
        event_q[start + 5] = base_q + 0.01 * q_scales; event_q[start + 6] = base_q + 0.02 * q_scales; event_q[start + 7] = base_q + np.asarray((0.015, 0.005, 0.01, 0.02)) * q_scales; event_phase[start + 5:start + 8] = event_phi
        guard_offsets[start:start + event_per_class] = -event_phi / phase_scale; event_simplices.append(np.arange(start, start + 5)); event_simplex_classes.append(event_class)
    impulses = rng.normal(scale=1e-4, size=(event_count, 4)); constitutive = rng.normal(scale=1e-4, size=(event_count, 1232)); constitutive -= (constitutive @ conservation.T) @ normal.T
    pre_states = []
    for value in event_q:
        candidate = rng.normal(scale=1e-4, size=1232); candidate -= normal @ (conservation @ candidate); pre_states.append(candidate + normal @ value)
    events = {"pre_states1232": np.asarray(pre_states), "pre_invariants4": event_q, "phase": event_phase, "source_mode_index": event_source, "destination_mode_index": event_destination, "transition_class_index": event_classes, "duration_seconds": np.linspace(0.2, 0.6, event_count), "integrated_ledger_impulse4": impulses, "ledger_null_constitutive_jump1232": constitutive, "reduced_guard_normals5": guard_normals, "reduced_guard_offsets": guard_offsets}
    additions = {"q_simplices": q_simplices, "q_scales": q_scales, "branch_simplices": np.asarray(branch_simplices), "branch_simplex_modes": np.asarray(branch_simplex_modes), "phase_scale": phase_scale, "event_simplices": np.asarray(event_simplices), "event_simplex_classes": np.asarray(event_simplex_classes)}
    return driver, branch, events, additions, conservation, normal


def _all_bitwise(left, right): return bool(np.array_equal(np.asarray(left), np.asarray(right)))


def _certificate():
    began = time.perf_counter(); _validate_parent(); driver, branch, events, additions, conservation, normal = _fixture()
    driver_values = []; branch_values = []; event_values = []; anchor_bitwise = []
    bary = np.full(5, 0.2)
    query_driver_q = bary @ driver["retained_invariant_nodes4"][additions["q_simplices"][0]]
    for mode in range(2):
        driver_values.append(interpolate_cycle_driver(driver, q_simplices=additions["q_simplices"], q_scales=additions["q_scales"], query_invariants=query_driver_q, phase=0.37, mode_index=mode, conservation_map=conservation))
        anchor = interpolate_cycle_driver(driver, q_simplices=additions["q_simplices"], q_scales=additions["q_scales"], query_invariants=driver["retained_invariant_nodes4"][0], phase=0.0, mode_index=mode, conservation_map=conservation)
        anchor_bitwise.append(_all_bitwise(anchor.slow_forcing_per_second, driver["slow_forcing1232_per_second"][0, 0, mode]))
        indices = additions["branch_simplices"][mode]; weights = np.asarray((0.16, 0.16, 0.16, 0.16, 0.16, 0.20)); query_q = weights @ branch["anchor_invariants4"][indices]; phases = branch["anchor_phase"][indices]; query_phase = float(weights @ phases)
        branch_values.append(interpolate_cycle_branch(branch, branch_simplices=additions["branch_simplices"], branch_simplex_modes=additions["branch_simplex_modes"], q_scales=additions["q_scales"], phase_scale=additions["phase_scale"], query_invariants=query_q, phase=query_phase, mode_index=mode, conservation_map=conservation, minimum_norm_normal=normal))
        branch_anchor = interpolate_cycle_branch(branch, branch_simplices=additions["branch_simplices"], branch_simplex_modes=additions["branch_simplex_modes"], q_scales=additions["q_scales"], phase_scale=additions["phase_scale"], query_invariants=branch["anchor_invariants4"][indices[0]], phase=branch["anchor_phase"][indices[0]], mode_index=mode, conservation_map=conservation, minimum_norm_normal=normal)
        anchor_bitwise.extend((_all_bitwise(branch_anchor.state, branch["anchor_states1232"][indices[0]]), _all_bitwise(branch_anchor.radial_matrices, branch["radial_matrices112x11x11"][indices[0]]), _all_bitwise(branch_anchor.source_matrices, branch["source_matrices112x11x11"][indices[0]])))
        event_indices = additions["event_simplices"][mode]; event_weights = np.full(5, 0.2); event_q = event_weights @ events["pre_invariants4"][event_indices]; event_phase = float(events["phase"][event_indices[0]]); pre_state = normal @ event_q
        event = interpolate_cycle_event(events, event_simplices=additions["event_simplices"], event_simplex_classes=additions["event_simplex_classes"], q_scales=additions["q_scales"], phase_scale=additions["phase_scale"], query_invariants=event_q, phase=event_phase, transition_class=mode, reduced_flow_scaled=np.asarray((0.0, 0.0, 0.0, 0.0, 0.2)), pre_state=pre_state, conservation_map=conservation, minimum_norm_normal=normal)
        plus = interpolate_cycle_event(events, event_simplices=additions["event_simplices"], event_simplex_classes=additions["event_simplex_classes"], q_scales=additions["q_scales"], phase_scale=additions["phase_scale"], query_invariants=event_q, phase=event_phase + 0.002 * additions["phase_scale"], transition_class=mode, reduced_flow_scaled=np.asarray((0.0, 0.0, 0.0, 0.0, 0.2)), pre_state=pre_state, conservation_map=conservation, minimum_norm_normal=normal, require_on_guard=False)
        minus = interpolate_cycle_event(events, event_simplices=additions["event_simplices"], event_simplex_classes=additions["event_simplex_classes"], q_scales=additions["q_scales"], phase_scale=additions["phase_scale"], query_invariants=event_q, phase=event_phase - 0.002 * additions["phase_scale"], transition_class=mode, reduced_flow_scaled=np.asarray((0.0, 0.0, 0.0, 0.0, 0.2)), pre_state=pre_state, conservation_map=conservation, minimum_norm_normal=normal, require_on_guard=False)
        if plus.guard.signed_guard_distance * minus.guard.signed_guard_distance >= 0.0: raise RuntimeError("guard sheet does not separate its two sides")
        event_values.append(event)
    outside_rejections = []
    try: interpolate_cycle_driver(driver, q_simplices=additions["q_simplices"], q_scales=additions["q_scales"], query_invariants=np.full(4, 100.0), phase=0.2, mode_index=0, conservation_map=conservation)
    except ValueError: outside_rejections.append(True)
    try: interpolate_cycle_branch(branch, branch_simplices=additions["branch_simplices"], branch_simplex_modes=additions["branch_simplex_modes"], q_scales=additions["q_scales"], phase_scale=additions["phase_scale"], query_invariants=np.full(4, 100.0), phase=0.2, mode_index=0, conservation_map=conservation, minimum_norm_normal=normal)
    except ValueError: outside_rejections.append(True)
    decisive = {"driver_forcing": np.asarray([value.slow_forcing_per_second for value in driver_values]), "branch_states": np.asarray([value.state for value in branch_values]), "branch_radial": np.asarray([value.radial_matrices for value in branch_values]), "event_post_states": np.asarray([value.post_state for value in event_values]), "event_guard_normals": np.asarray([value.guard.oriented_normal for value in event_values])}
    with tempfile.TemporaryDirectory(prefix="cycle_atlas_fixture_") as directory:
        path = Path(directory) / "checkpoint.npz"; np.savez_compressed(path, **decisive)
        with np.load(path, allow_pickle=False) as payload: checkpoint_bitwise = all(np.array_equal(payload[name], value) for name, value in decisive.items())
    minima = [value.q_location.minimum_weight for value in driver_values] + [value.location.minimum_weight for value in branch_values] + [value.guard.minimum_weight for value in event_values]
    weight_defects = [value.q_location.weight_sum_defect for value in driver_values] + [value.location.weight_sum_defect for value in branch_values] + [value.guard.weight_sum_defect for value in event_values]
    coordinate_defects = [value.q_location.coordinate_reproduction_defect for value in driver_values] + [value.location.coordinate_reproduction_defect for value in branch_values] + [value.guard.affine_hull_reproduction_defect for value in event_values]
    metrics = {"schema_version": 1, "work_package": WORK_PACKAGE, "classification": PASS_CLASSIFICATION, "passed": True, "synthetic_fixture_only": True, "physical_claim": False, "driver_modes_tested": len(driver_values), "branch_modes_tested": len(branch_values), "event_classes_tested": len(event_values), "minimum_barycentric_weight": float(min(minima)), "maximum_weight_sum_defect": float(max(weight_defects)), "maximum_coordinate_reproduction_defect": float(max(coordinate_defects)), "maximum_driver_forcing_ledger_relative_defect": float(max(value.forcing_ledger_relative_defect for value in driver_values)), "maximum_branch_invariant_relative_defect": float(max(value.invariant_relative_defect for value in branch_values)), "maximum_branch_radial_symmetry_defect": float(max(value.maximum_radial_symmetry_defect for value in branch_values)), "maximum_branch_source_positive_eigenvalue": float(max(value.maximum_source_entropy_positive_eigenvalue for value in branch_values)), "minimum_branch_source_nullity": int(min(value.minimum_source_nullity for value in branch_values)), "minimum_branch_fast_spectral_gap_per_second": float(min(value.fast_spectral_gap_per_second for value in branch_values)), "branch_boundary_incoming_counts": [[value.inner_incoming_count, value.outer_incoming_count] for value in branch_values], "minimum_event_transversality": float(min(abs(value.transversality) for value in event_values)), "maximum_event_reset_ledger_relative_defect": float(max(value.reset_ledger_relative_defect for value in event_values)), "maximum_event_constitutive_null_relative_defect": float(max(value.constitutive_null_relative_defect for value in event_values)), "all_anchor_reproductions_bitwise": all(anchor_bitwise), "outside_hull_rejections": len(outside_rejections), "checkpoint_roundtrip_bitwise": checkpoint_bitwise, "physical_model_complete": False, "physical_payloads_acquired": False, "heldout_physical_validation_complete": False, "complete_cycle_execution_authorized": False, "complete_cycle_steps": 0, "certificate_wall_seconds": time.perf_counter() - began, "authorized_next": AUTHORIZED_NEXT}
    original = _validate_parent()[1]["binding_structure_gates"]
    metrics["passed"] = bool(metrics["minimum_barycentric_weight"] >= original["minimum_barycentric_weight"] and metrics["maximum_weight_sum_defect"] <= original["maximum_weight_sum_defect"] and metrics["maximum_coordinate_reproduction_defect"] <= original["maximum_coordinate_reproduction_defect"] and metrics["maximum_driver_forcing_ledger_relative_defect"] <= original["maximum_forcing_ledger_relative_defect"] and metrics["maximum_branch_invariant_relative_defect"] <= original["maximum_invariant_reconstruction_relative_defect"] and metrics["maximum_branch_radial_symmetry_defect"] <= original["maximum_radial_symmetry_defect"] and metrics["maximum_branch_source_positive_eigenvalue"] <= original["maximum_source_positive_eigenvalue"] and metrics["minimum_branch_source_nullity"] >= original["minimum_source_nullity"] and metrics["minimum_branch_fast_spectral_gap_per_second"] > original["minimum_fast_spectral_gap_per_second"] and all(value == [0, 11] for value in metrics["branch_boundary_incoming_counts"]) and metrics["minimum_event_transversality"] >= original["minimum_guard_transversality"] and metrics["maximum_event_reset_ledger_relative_defect"] <= original["maximum_event_reset_ledger_relative_defect"] and metrics["all_anchor_reproductions_bitwise"] and metrics["outside_hull_rejections"] == 2 and checkpoint_bitwise)
    if not metrics["passed"]: metrics["classification"] = FAIL_CLASSIFICATION; metrics["authorized_next"] = None
    return metrics, decisive


def _update(summary):
    utility = _u(); rows = list(csv.DictReader(CANONICAL_MANIFEST.open(newline="", encoding="utf-8"))); rows = [row for row in rows if row.get("case") != ARTIFACT]; status = "SUPPORTED" if summary["passed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file(): rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": utility._sha256(path), "scientific_status": status})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle: writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    catalog = utility._read_json(CANONICAL_SUMMARY); catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": summary["classification"], "passed": summary["passed"]}; catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": utility._git("rev-parse", "HEAD"), "latest_work_package": WORK_PACKAGE}); utility._write_json(CANONICAL_SUMMARY, catalog)


def _canonicalize(metrics, arrays):
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists(): raise RuntimeError("cycle interpolator certificate already exists")
    hashes, _, _ = _validate_parent(require_clean=True); utility = _u(); CANONICAL_DIRECTORY.mkdir(parents=True); utility._write_json(CANONICAL_DIRECTORY / "interpolator_metrics.json", metrics); np.savez_compressed(CANONICAL_DIRECTORY / "interpolator_arrays.npz", **arrays)
    summary = {"schema_version": 1, "work_package": WORK_PACKAGE, "classification": metrics["classification"], "passed": metrics["passed"], "cycle_interpolator_structure_certified": metrics["passed"], "event_guard_sheet_dimension_corrected": True, "synthetic_fixture_only": True, "physical_model_complete": False, "physical_payloads_acquired": False, "heldout_physical_validation_complete": False, "complete_cycle_execution_authorized": False, "complete_cycle_steps": 0, "authorized_next": metrics["authorized_next"]}; utility._write_json(CANONICAL_DIRECTORY / "summary.json", summary); utility._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"manifest_artifact": manifest.ARTIFACT, "manifest_checksum_manifest_sha256": PARENT_SHA256, "manifest_hashes": hashes})
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True); REPORT_PATH.write_text("# Cycle driver, branch, and event interpolator structure certificate\n\n" f"Classification: `{metrics['classification']}`.\n\n" f"The synthetic structure fixture closes the driver ledger at `{metrics['maximum_driver_forcing_ledger_relative_defect']:.6e}`, branch invariants at `{metrics['maximum_branch_invariant_relative_defect']:.6e}`, and event reset ledgers at `{metrics['maximum_event_reset_ledger_relative_defect']:.6e}`. The common source nullity is `{metrics['minimum_branch_source_nullity']}`, boundary counts are 0/11, event sheets have five vertices, anchor reproduction and checkpoint reload are bitwise, and out-of-hull queries fail closed.\n\n" "This is not physical calibration. External cycle inputs and all heldout physical validation remain missing, and no cycle step occurred.\n", encoding="utf-8")
    sources = (THIS_RUNNER, THIS_TEST, PHYSICAL_SOURCE, PHYSICAL_TEST, REPORT_RELATIVE); utility._write_json(CANONICAL_DIRECTORY / "provenance.json", {"implementation_commit": utility._git("rev-parse", "HEAD"), "source_hashes": {name: utility._sha256(ROOT / name) for name in sources}, "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "thread_environment": {name: os.environ.get(name, "") for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}}); names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()); (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utility._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8"); _update(summary); return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run", action="store_true"); args = parser.parse_args()
    if not args.run: parser.error("choose --run")
    metrics, arrays = _certificate(); print(json.dumps(metrics, indent=2, sort_keys=True), flush=True); return 0 if _canonicalize(metrics, arrays)["passed"] else 2


if __name__ == "__main__": raise SystemExit(main())
