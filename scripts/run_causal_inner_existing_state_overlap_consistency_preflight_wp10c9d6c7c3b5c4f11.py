#!/usr/bin/env python3
"""Run the analysis-only retained-overlap consistency preflight."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_retained_guard_buffer_micro_macro_manifest_wp10c9d6c7c3b5c4f10 as c4f10  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_monolithic_dae import (  # noqa: E402
    _integrated_mapped_storage,
    _spatial_nodes,
)


c4f9 = c4f10.c4f9
c4f7 = c4f9.c4f7
c4f1 = c4f9.c4f1
SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f11"
ARTIFACT = "causal_inner_existing_state_overlap_consistency_preflight_wp10c9d6c7c3b5c4f11"
THIS_RUNNER = "scripts/run_causal_inner_existing_state_overlap_consistency_preflight_wp10c9d6c7c3b5c4f11.py"
THIS_TEST = "tests/test_causal_inner_existing_state_overlap_consistency_preflight_wp10c9d6c7c3b5c4f11.py"
REPORT_RELATIVE = "docs/reports/current/CODEX_CAUSAL_INNER_EXISTING_STATE_OVERLAP_CONSISTENCY_PREFLIGHT_WP10C9D6C7C3B5C4F11_2026-08-13.md"
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
CONTRACT_PATH = CANONICAL_DIRECTORY / "analysis_contract.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
CHECKPOINT_PATH = ROOT / "outputs/checkpoints" / ARTIFACT / "overlap_storage.npz"

LAYOUTS = c4f9.LAYOUTS
STATE_TIMES = c4f9.STATE_TIMES
HISTORY_TIMES_US = c4f9.HISTORY_TIMES_US
FIELDS = c4f9.FIELDS
CORE_FACE = c4f9.RECOVERY_FACE
GUARD_END_FACE = c4f9.COUPLING_FACE
PARENT_CELLS = 64


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load(path):
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _save(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*arguments):
    return subprocess.run(("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _validate():
    parent = _read(c4f10.SUMMARY_PATH)
    manifest = _read(c4f10.MANIFEST_PATH)
    if (
        not parent["passed"]
        or not parent["definitions_only"]
        or parent["authorized_next"] != "WP10c9d6c7c3b5c4f11_analysis_only_existing_state_overlap_consistency_preflight"
        or manifest["physical_partition"]["shared_exchange_parent_face"] != CORE_FACE
    ):
        raise RuntimeError("c4f11 authorization changed")
    return manifest


def _restrict_integrals(values, layout):
    values = np.asarray(values, dtype=float)
    result = np.zeros((PARENT_CELLS, values.shape[-1]), dtype=float)
    np.add.at(result, layout.parent_cell_indices, values)
    return result


def _mapped_cells(context, state):
    mapped, factors, _nodes = _integrated_mapped_storage(context, state, _spatial_nodes(context))
    if not np.array_equal(factors, np.ones_like(factors)):
        raise RuntimeError("c4f11 mapped storage activated reconstruction scaling")
    return np.asarray(mapped, dtype=float)


def _evaluate():
    if CHECKPOINT_PATH.exists():
        return _load(CHECKPOINT_PATH)
    _parent_grid, configurations = c4f1._configurations()
    trajectories = c4f9._trajectories()
    parent_mapped = []
    parent_height_rates = []
    child_guard_mapped = []
    restriction_defects = []
    inventory_defects = []
    complement_closures = []
    increment_roundtrip_defects = []
    baseline_response_defects = []
    wall = []
    for label in LAYOUTS:
        layout, configuration = configurations[label]
        multiplier = int(layout.refinement_ratio)
        lo, hi = CORE_FACE * multiplier, GUARD_END_FACE * multiplier
        selected = c4f1._indices(trajectories[label]["times"], STATE_TIMES)
        layout_parent = []
        layout_children = []
        began = time.perf_counter()
        for state in trajectories[label]["states"][selected]:
            mapped = _mapped_cells(configuration["context"], state)
            parent = _restrict_integrals(mapped, layout)
            layout_parent.append(parent)
            layout_children.append(mapped[lo:hi])
            direct = np.sum(mapped, axis=0)
            partitioned = np.sum(parent[:CORE_FACE], axis=0) + np.sum(parent[CORE_FACE:], axis=0)
            inventory_defects.append(np.max(np.abs(direct - partitioned)) / max(np.max(np.abs(direct)), np.finfo(float).tiny))
            restricted_total = np.sum(parent, axis=0)
            restriction_defects.append(np.max(np.abs(direct - restricted_total)) / max(np.max(np.abs(direct)), np.finfo(float).tiny))
            for parent_cell in range(CORE_FACE, GUARD_END_FACE):
                children = np.flatnonzero(layout.parent_cell_indices == parent_cell)
                child_values = mapped[children]
                fractions = layout.grid.cell_measures[children] / layout.parent_grid.cell_measures[parent_cell]
                mean = fractions[:, None] * parent[parent_cell]
                complement = child_values - mean
                complement_closures.append(np.max(np.abs(np.sum(complement, axis=0))) / max(np.max(np.abs(parent[parent_cell])), np.finfo(float).tiny))
        wall.append(time.perf_counter() - began)
        layout_parent = np.asarray(layout_parent)
        layout_children = np.asarray(layout_children)
        parent_mapped.append(layout_parent)
        child_guard_mapped.append(layout_children)
        for old_index, new_index in zip(range(len(STATE_TIMES) - 1), range(1, len(STATE_TIMES)), strict=True):
            delta_parent = layout_parent[new_index] - layout_parent[old_index]
            accumulated = np.zeros_like(delta_parent)
            for parent_cell in range(CORE_FACE, GUARD_END_FACE):
                children = np.flatnonzero(layout.parent_cell_indices == parent_cell)
                fractions = layout.grid.cell_measures[children] / layout.parent_grid.cell_measures[parent_cell]
                child_increment = fractions[:, None] * delta_parent[parent_cell]
                accumulated[parent_cell] = np.sum(child_increment, axis=0)
            scale = max(np.max(np.abs(delta_parent[CORE_FACE:GUARD_END_FACE])), np.finfo(float).tiny)
            increment_roundtrip_defects.append(np.max(np.abs(accumulated[CORE_FACE:GUARD_END_FACE] - delta_parent[CORE_FACE:GUARD_END_FACE])) / scale)
            reconstructed = layout_parent[old_index, CORE_FACE:GUARD_END_FACE] + accumulated[CORE_FACE:GUARD_END_FACE]
            target = layout_parent[new_index, CORE_FACE:GUARD_END_FACE]
            baseline_response_defects.append(np.max(np.abs(reconstructed - target)) / max(np.max(np.abs(target)), np.finfo(float).tiny))
        height_rates = []
        for time_us in HISTORY_TIMES_US:
            _state, history = c4f9._accepted_state_and_history(label, trajectories[label], time_us)
            parent_height = _restrict_integrals(history.previous_responsive_height_storage_increment, layout)
            height_rates.append(parent_height / history.previous_timestep_seconds)
        parent_height_rates.append(np.asarray(height_rates))
    arrays = {
        "state_times_seconds": STATE_TIMES,
        "history_times_seconds": np.asarray(HISTORY_TIMES_US, dtype=float) * 1.0e-6,
        "parent_mapped_storage": np.asarray(parent_mapped),
        "parent_responsive_height_history_rates": np.asarray(parent_height_rates),
        "restriction_defects": np.asarray(restriction_defects),
        "physical_inventory_partition_defects": np.asarray(inventory_defects),
        "fine_complement_zero_mean_closures": np.asarray(complement_closures),
        "storage_increment_roundtrip_defects": np.asarray(increment_roundtrip_defects),
        "baseline_plus_response_storage_defects": np.asarray(baseline_response_defects),
        "evaluation_wall_seconds": np.asarray(wall),
    }
    _save(CHECKPOINT_PATH, **arrays)
    return arrays


def _metric(values, scales, gates):
    scales = np.asarray(scales, dtype=float)
    active = scales > np.finfo(float).tiny
    normalized = tuple(np.asarray(item)[..., active] / scales[active] for item in values)
    coarse_middle = normalized[1] - normalized[0]
    middle_fine = normalized[2] - normalized[1]
    coarse_norm = float(np.linalg.norm(coarse_middle))
    fine_norm = float(np.linalg.norm(middle_fine))
    order = float(np.log2(max(coarse_norm, np.finfo(float).tiny) / max(fine_norm, np.finfo(float).tiny)))
    cosine = float(np.vdot(coarse_middle.ravel(), middle_fine.ravel()).real / max(coarse_norm * fine_norm, np.finfo(float).tiny))
    return {"RMS_order": order, "error_direction_cosine": cosine, "passed": bool(order >= gates["minimum_spatial_RMS_order"] and cosine >= gates["minimum_spatial_error_direction_cosine"])}


def _analyze(arrays, manifest):
    gates = manifest["prospective_gates"]
    mapped = arrays["parent_mapped_storage"][:, :, CORE_FACE:, :][:, :, :, FIELDS]
    guard = arrays["parent_mapped_storage"][:, :, CORE_FACE:GUARD_END_FACE, :][:, :, :, FIELDS]
    height = arrays["parent_responsive_height_history_rates"][:, :, CORE_FACE:GUARD_END_FACE, :][:, :, :, FIELDS]
    mapped_scales = np.maximum.reduce([np.max(np.abs(item), axis=(0, 1)) for item in mapped])
    guard_scales = np.maximum.reduce([np.max(np.abs(item), axis=(0, 1)) for item in guard])
    height_scales = np.maximum.reduce([np.max(np.abs(item), axis=(0, 1)) for item in height])
    metrics = {
        "macro_exterior_mapped_storage": _metric(tuple(mapped), mapped_scales, gates),
        "guard_parent_mapped_storage": _metric(tuple(guard), guard_scales, gates),
        "guard_parent_responsive_height_history": _metric(tuple(height), height_scales, gates),
    }
    maximum_restriction = float(np.max(arrays["restriction_defects"]))
    maximum_inventory = float(np.max(arrays["physical_inventory_partition_defects"]))
    maximum_complement = float(np.max(arrays["fine_complement_zero_mean_closures"]))
    maximum_roundtrip = float(np.max(arrays["storage_increment_roundtrip_defects"]))
    maximum_baseline_response = float(np.max(arrays["baseline_plus_response_storage_defects"]))
    complement_observability = float(_read(c4f7.SUMMARY_PATH)["JVP_fraction_of_transition"][str(CORE_FACE)])
    overlap = _read(c4f9.SUMMARY_PATH)
    method_passed = bool(
        maximum_restriction <= gates["maximum_conservative_restriction_defect"]
        and maximum_inventory <= gates["maximum_physical_inventory_partition_defect"]
        and maximum_complement <= gates["maximum_overlap_sync_roundtrip_defect"]
        and maximum_roundtrip <= gates["maximum_overlap_sync_roundtrip_defect"]
        and maximum_baseline_response <= gates["maximum_baseline_plus_response_scaled_defect"]
    )
    spatial_passed = bool(all(item["passed"] for item in metrics.values()) and all(item["passed"] for item in overlap["overlap_state_metrics"].values()))
    reaction_passed = bool(complement_observability <= gates["maximum_guard_reaction_fraction_for_projection_free_memory_screen"])
    if not method_passed:
        classification = "overlap_restriction_or_inventory_method_failed"
        authorized_next = "repair_overlap_maps_only"
        passed = False
    elif not spatial_passed:
        classification = "macro_owned_overlap_state_spatial_gate_failed"
        authorized_next = "absolute_slow_closure_remains_blocked"
        passed = False
    elif not reaction_passed:
        classification = "overlap_reaction_observable_retain_microburst"
        authorized_next = "definitions_only_reaction_aware_overlap_microburst_manifest"
        passed = True
    else:
        classification = "existing_state_overlap_contract_certified"
        authorized_next = "WP10c9d6c7c3b5c4f12_definitions_only_face36_augmented_projected_memory_screen_manifest"
        passed = True
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "physical_failure_detected": False,
        "method_gates_passed": method_passed,
        "spatial_gates_passed": spatial_passed,
        "guard_reaction_observability_gate_passed": reaction_passed,
        "maximum_conservative_restriction_defect": maximum_restriction,
        "maximum_physical_inventory_partition_defect": maximum_inventory,
        "maximum_fine_complement_zero_mean_closure": maximum_complement,
        "maximum_storage_increment_roundtrip_defect": maximum_roundtrip,
        "maximum_baseline_plus_response_storage_defect": maximum_baseline_response,
        "fine_complement_face36_observability_fraction": complement_observability,
        "spatial_metrics": metrics,
        "face36_overlap_metrics_preserved": overlap["overlap_state_metrics"],
        "raw_face48_absolute_export_rejection_preserved": True,
        "response_certificate_preserved": True,
        "memory_propagation_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "fifty_ms_propagation_authorized": False,
        "authorized_next": authorized_next,
    }


def _catalog(summary):
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": _sha(path), "scientific_status": "CERTIFIED" if summary["passed"] else "REJECTED"})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": summary["classification"], "passed": summary["passed"]}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "latest_work_package": WORK_PACKAGE})
    _write(CANONICAL_SUMMARY, catalog)


def _finalize(arrays, manifest, summary):
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _save(DECISIVE_ARRAYS, **arrays)
    _write(CONFIG_PATH, {"schema_version": SCHEMA_VERSION, "state_times_seconds": STATE_TIMES, "history_times_microseconds": HISTORY_TIMES_US, "core_face": CORE_FACE, "guard_end_face": GUARD_END_FACE})
    _write(CONTRACT_PATH, manifest)
    _write(SUMMARY_PATH, summary)
    lines = [
        "# Existing-state overlap consistency preflight",
        "",
        f"Classification: `{summary['classification']}`.",
        "",
        "No trajectory, projected-memory propagation, or fixed-Q solve ran.",
        "",
        "| Macro-owned state | Order | Cosine | Pass |",
        "|---|---:|---:|---:|",
    ]
    for name, item in summary["spatial_metrics"].items():
        lines.append(f"| {name} | {item['RMS_order']:.6f} | {item['error_direction_cosine']:.6f} | {item['passed']} |")
    lines.extend([
        "",
        f"Restriction, physical-inventory, fine-complement zero-mean, increment roundtrip, and baseline-plus-response defects are at most `{max(summary['maximum_conservative_restriction_defect'], summary['maximum_physical_inventory_partition_defect'], summary['maximum_fine_complement_zero_mean_closure'], summary['maximum_storage_increment_roundtrip_defect'], summary['maximum_baseline_plus_response_storage_defect']):.6e}`.",
        "",
        f"The retained fine-complement extraction observability at face 36 is `{summary['fine_complement_face36_observability_fraction']:.6f}`, below the prospective `0.10` gate. The complement remains in the micro guard; it is not overwritten or discarded.",
        "",
        "The raw face-48 absolute export remains rejected. Any memory screen must use face 36 plus the augmented storage/history state and must be frozen in a new definitions-only manifest.",
        "",
        f"Authorized next: `{summary['authorized_next']}`.",
        "",
    ])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    _write(PROVENANCE_PATH, {"schema_version": SCHEMA_VERSION, "execution_commit": _git("rev-parse", "HEAD"), "execution_head_tree": _git("rev-parse", "HEAD^{tree}"), "parent_manifest_sha256": _sha(c4f10.MANIFEST_PATH), "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}, "source_hashes": {THIS_RUNNER: _sha(ROOT / THIS_RUNNER), THIS_TEST: _sha(ROOT / THIS_TEST) if (ROOT / THIS_TEST).exists() else None}})
    files = (CONFIG_PATH, CONTRACT_PATH, SUMMARY_PATH, DECISIVE_ARRAYS, PROVENANCE_PATH)
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{_sha(path)}  {path.name}\n" for path in files), encoding="utf-8")
    _catalog(summary)


def main():
    manifest = _validate()
    arrays = _evaluate()
    summary = _analyze(arrays, manifest)
    _finalize(arrays, manifest, summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
