#!/usr/bin/env python3
"""Run the frozen endpoint/interface regularity controls embedded.

WP10c9d6c7c1a certified all sixteen c7c0 variants uniformly.  This package
propagates the same immutable variants on the three c7a embedded layouts and
applies the unchanged direct state, physical-export, coupling-face,
interface-state, and characteristic-energy contracts.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_causal_inner_embedded_regularity_uniform_wp10c9d6c7c1a as c7c1a
import run_causal_inner_embedded_validation_wp10c9d6c7b as c7b

from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_canonical_json_sha256,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c1b"
ANALYZED_BASE_COMMIT = "d9300e938f6636d92518cc242ac16231a65d6716"
ANALYZED_BASE_PARENT = "406ab7e18d6094d07647efb91dccce06808087b4"
ANALYZED_BASE_TREE = "9249a03bdd6403866e411fe131843ce35b22a244"
FROZEN_MANIFEST_SHA256 = (
    "b230ce7a3c7e7546d0d706ee8f9bcfa3102c6c69be5f67a29aa451e1b5d9706b"
)
THIS_RUNNER = (
    "scripts/"
    "run_causal_inner_embedded_regularity_wp10c9d6c7c1b.py"
)

LABELS = tuple(c7b.LABELS)
BASE_PROFILES = (
    "p4__inward_shear",
    "p4__outward_shear",
    "p3_buffer45__inward_shear",
    "p3_buffer45__outward_shear",
)
MAXIMUM_EXACT_INTEGRAL_RESIDUAL = 1.0e-12

C7C0_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_embedded_regularity_manifest_wp10c9d6c7c0"
)
C7C0_MANIFEST = C7C0_DIRECTORY / "regularity_manifest.json"
C7C0_SUMMARY = C7C0_DIRECTORY / "summary.json"
C7C0_ARRAYS = C7C0_DIRECTORY / "decisive_arrays.npz"
C7C0_PROVENANCE = C7C0_DIRECTORY / "provenance.json"

C7C1A_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_embedded_regularity_uniform_wp10c9d6c7c1a"
)
C7C1A_SUMMARY = C7C1A_DIRECTORY / "summary.json"
C7C1A_ARRAYS = C7C1A_DIRECTORY / "decisive_arrays.npz"
C7C1A_PROVENANCE = C7C1A_DIRECTORY / "provenance.json"

CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_embedded_regularity_wp10c9d6c7c1b"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

CHECKPOINT_DIRECTORY = ROOT / "outputs/checkpoints"
PROPAGATION_CHECKPOINT = (
    CHECKPOINT_DIRECTORY
    / "causal_inner_embedded_regularity_"
    "wp10c9d6c7c1b_propagated.npz"
)
PROPAGATION_CHECKPOINT_REPORT = (
    CHECKPOINT_DIRECTORY
    / "causal_inner_embedded_regularity_"
    "wp10c9d6c7c1b_report.json"
)

IMPLEMENTATION_SOURCES = (
    THIS_RUNNER,
    "scripts/"
    "run_causal_inner_embedded_regularity_uniform_wp10c9d6c7c1a.py",
    "scripts/"
    "run_causal_inner_embedded_regularity_manifest_wp10c9d6c7c0.py",
    "scripts/run_causal_inner_embedded_validation_wp10c9d6c7b.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_embedded_validation.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_embedded_patch.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_monolithic_tangent.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_packet_validation.py",
    "tests/"
    "test_causal_inner_embedded_regularity_wp10c9d6c7c1b.py",
)


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


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _plain(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_analyzed_git_identity() -> dict:
    resolved = _git_value("rev-parse", ANALYZED_BASE_COMMIT)
    parent = _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
    tree = _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
    if (
        resolved != ANALYZED_BASE_COMMIT
        or parent != ANALYZED_BASE_PARENT
        or tree != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("WP10c9d6c7c1b analyzed git identity changed")
    return {
        "analyzed_base_commit": resolved,
        "analyzed_base_parent": parent,
        "analyzed_base_tree": tree,
    }


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {
            name: np.array(source[name], copy=True)
            for name in source.files
        }


def _array_hashes(arrays: dict[str, np.ndarray]) -> dict[str, str]:
    return {
        name: causal_array_sha256(values)
        for name, values in sorted(arrays.items())
    }


def _source_manifest() -> tuple[dict[str, str], str]:
    hashes = {
        path: _sha256(ROOT / path)
        for path in IMPLEMENTATION_SOURCES
        if (ROOT / path).is_file()
    }
    digest = hashlib.sha256()
    for path, value in sorted(hashes.items()):
        digest.update(path.encode("utf-8"))
        digest.update(value.encode("ascii"))
    return hashes, digest.hexdigest()


def _refresh_sha256s(directory: Path) -> None:
    entries = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.name == "SHA256SUMS.txt":
            continue
        entries.append(f"{_sha256(path)}  {path.name}")
    (directory / "SHA256SUMS.txt").write_text(
        "\n".join(entries) + "\n",
        encoding="utf-8",
    )


def _refresh_canonical_catalog() -> None:
    rows: list[dict[str, str | int]] = []
    for case in sorted(CANONICAL_DIRECTORY.parent.iterdir()):
        if not case.is_dir():
            continue
        provenance_path = case / "provenance.json"
        if not provenance_path.is_file():
            continue
        provenance = json.loads(
            provenance_path.read_text(encoding="utf-8")
        )
        status = provenance.get(
            "scientific_status",
            provenance.get("numerical_status", "DIAGNOSTIC ONLY"),
        )
        for path in sorted(case.iterdir()):
            if not path.is_file():
                continue
            rows.append(
                {
                    "case": case.name,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "scientific_status": status,
                }
            )
    with CANONICAL_MANIFEST.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "case",
                "path",
                "bytes",
                "sha256",
                "scientific_status",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    summary = json.loads(CANONICAL_SUMMARY.read_text(encoding="utf-8"))
    summary.update(
        {
            "case_count": len({str(row["case"]) for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": ANALYZED_BASE_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, summary)


def _load_authorized_inputs() -> tuple[dict, dict, dict, dict, dict, dict]:
    c7c0_summary = json.loads(C7C0_SUMMARY.read_text(encoding="utf-8"))
    manifest = json.loads(C7C0_MANIFEST.read_text(encoding="utf-8"))
    arrays = _load_npz(C7C0_ARRAYS)
    c7c1a_summary = json.loads(C7C1A_SUMMARY.read_text(encoding="utf-8"))
    if (
        c7c0_summary["manifest_sha256"] != FROZEN_MANIFEST_SHA256
        or not c7c0_summary["passed"]
        or c7c1a_summary["manifest_sha256"] != FROZEN_MANIFEST_SHA256
        or c7c1a_summary["classification"]
        != (
            "endpoint_interface_regularity_uniform_controls_certified_"
            "embedded_discrimination_authorized"
        )
        or not c7c1a_summary["passed"]
        or not c7c1a_summary[
            "embedded_regularity_discrimination_authorized"
        ]
    ):
        raise RuntimeError("WP10c9d6c7c1b authorization changed")
    stored = manifest.pop("manifest_sha256")
    calculated = causal_canonical_json_sha256(manifest)
    manifest["manifest_sha256"] = stored
    if (
        stored != FROZEN_MANIFEST_SHA256
        or calculated != FROZEN_MANIFEST_SHA256
        or manifest["propagation_executed"]
    ):
        raise RuntimeError("WP10c9d6c7c0 manifest changed")
    if set(arrays) != set(c7c0_summary["decisive_array_hashes"]):
        raise RuntimeError("WP10c9d6c7c0 decisive array set changed")
    for name, expected in c7c0_summary["decisive_array_hashes"].items():
        if causal_array_sha256(arrays[name]) != expected:
            raise RuntimeError(f"WP10c9d6c7c0 array changed: {name}")
    c7a_parent, c7a_manifest, c7a_arrays, _f1 = (
        c7b._load_frozen_inputs()
    )
    return (
        c7c0_summary,
        manifest,
        arrays,
        c7c1a_summary,
        c7a_parent,
        (c7a_manifest, c7a_arrays),
    )


def _variant_metadata(manifest: dict) -> dict:
    variants = manifest["profile_variants"]
    packet_ids = [item["profile_id"] for item in variants]
    base_names = list(BASE_PROFILES)
    base_indices = np.asarray(
        [base_names.index(item["base_profile"]) for item in variants],
        dtype=int,
    )
    multipliers = np.asarray(
        [
            float(item["amplitude_factor"]) * int(item["sign"])
            for item in variants
        ],
        dtype=float,
    )
    base_variant_indices = []
    for index in range(len(base_names)):
        matches = np.flatnonzero(
            (base_indices == index) & (multipliers == 1.0)
        )
        if matches.size != 1:
            raise RuntimeError("frozen c7c0 base variant changed")
        base_variant_indices.append(int(matches[0]))
    return {
        "variants": variants,
        "packet_ids": packet_ids,
        "base_names": base_names,
        "base_indices": base_indices,
        "multipliers": multipliers,
        "base_variant_indices": np.asarray(
            base_variant_indices,
            dtype=int,
        ),
    }


def _build_inputs(
    manifest: dict,
    arrays: dict[str, np.ndarray],
    c7a_manifest: dict,
    c7a_arrays: dict[str, np.ndarray],
) -> tuple[dict, dict, dict, dict]:
    configurations, layouts, _old_metadata = c7b._build_inputs(
        c7a_manifest,
        c7a_arrays,
    )
    for label in LABELS:
        columns = np.asarray(
            configurations[label]["primitive_column_scales"],
            dtype=float,
        )
        configurations[label]["initial_directions"] = {
            "common_mode": (
                np.asarray(
                    arrays[
                        f"p4__inward_shear__{label}__primary_physical"
                    ],
                    dtype=float,
                ).ravel()
                / columns
            ),
            "heldout_near_excision": (
                np.asarray(
                    arrays[
                        "p3_buffer45__outward_shear__"
                        f"{label}__primary_physical"
                    ],
                    dtype=float,
                ).ravel()
                / columns
            ),
        }
    metadata = _variant_metadata(manifest)
    directions = c7b._variant_directions(
        configurations,
        arrays,
        metadata,
    )
    return configurations, layouts, metadata, directions


@contextmanager
def _regularity_profiles():
    historical = c7b.BASE_PROFILES
    c7b.BASE_PROFILES = BASE_PROFILES
    try:
        yield
    finally:
        c7b.BASE_PROFILES = historical


def _save_checkpoint(
    propagated: dict,
    propagation_report: dict,
    method_reports: dict,
) -> None:
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    payload = {}
    for label in LABELS:
        for name, values in propagated[label].items():
            payload[f"{label}::{name}"] = np.asarray(values)
    np.savez_compressed(PROPAGATION_CHECKPOINT, **payload)
    _write_json(
        PROPAGATION_CHECKPOINT_REPORT,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "manifest_sha256": FROZEN_MANIFEST_SHA256,
            "c7c0_arrays_sha256": _sha256(C7C0_ARRAYS),
            "labels": LABELS,
            "propagation_report": propagation_report,
            "method_reports": method_reports,
            "checkpoint_sha256": _sha256(PROPAGATION_CHECKPOINT),
        },
    )


def _load_checkpoint() -> tuple[dict, dict, dict]:
    if (
        not PROPAGATION_CHECKPOINT.is_file()
        or not PROPAGATION_CHECKPOINT_REPORT.is_file()
    ):
        raise RuntimeError("WP10c9d6c7c1b checkpoint is unavailable")
    report = json.loads(
        PROPAGATION_CHECKPOINT_REPORT.read_text(encoding="utf-8")
    )
    if (
        report["manifest_sha256"] != FROZEN_MANIFEST_SHA256
        or report["c7c0_arrays_sha256"] != _sha256(C7C0_ARRAYS)
        or tuple(report["labels"]) != LABELS
        or report["checkpoint_sha256"] != _sha256(PROPAGATION_CHECKPOINT)
    ):
        raise RuntimeError("WP10c9d6c7c1b checkpoint changed")
    raw = _load_npz(PROPAGATION_CHECKPOINT)
    propagated = {label: {} for label in LABELS}
    for key, values in raw.items():
        label, name = key.split("::", 1)
        propagated[label][name] = values
    return (
        propagated,
        report["propagation_report"],
        report["method_reports"],
    )


def _prospective_decision(
    direct: dict,
    coupling: dict,
    metadata: dict,
) -> dict:
    profile_reports = {}
    for profile in BASE_PROFILES:
        variants = [
            packet_id
            for packet_id, base_name in zip(
                metadata["packet_ids"],
                (
                    metadata["base_names"][index]
                    for index in metadata["base_indices"]
                ),
                strict=True,
            )
            if base_name == profile
        ]
        direct_passed = all(
            direct["packet_reports"][packet_id]["passed"]
            for packet_id in variants
        )
        energy_passed = coupling["energy_reports"][profile]["passed"]
        interface_passed = coupling["interface_state_reports"][profile][
            "passed"
        ]
        coupling_face_passed = coupling["common_face_flux_reports"][
            profile
        ][str(c7b.c7a.PARENT_COUPLING_FACE)]["passed"]
        profile_reports[profile] = {
            "variant_ids": variants,
            "direct_state_and_export_contract_passed": direct_passed,
            "characteristic_energy_contract_passed": energy_passed,
            "interface_state_contract_passed": interface_passed,
            "coupling_face_flux_contract_passed": coupling_face_passed,
            "passed": bool(
                direct_passed
                and energy_passed
                and interface_passed
                and coupling_face_passed
            ),
        }
    p4_passed = all(
        profile_reports[name]["passed"]
        for name in (
            "p4__inward_shear",
            "p4__outward_shear",
        )
    )
    buffered_passed = all(
        profile_reports[name]["passed"]
        for name in (
            "p3_buffer45__inward_shear",
            "p3_buffer45__outward_shear",
        )
    )
    if p4_passed and buffered_passed:
        classification = (
            "endpoint_interface_regularity_crossover_certified_"
            "no_operator_redesign"
        )
        authorized_next = (
            "WP10c9d6c7c2_embedded_admissible_common_mode_manifest"
        )
        decision_branch = "both_prospective_classes_pass"
    elif not p4_passed and buffered_passed:
        classification = (
            "active_endpoint_coupling_hypothesis_selected_"
            "local_audit_only"
        )
        authorized_next = (
            "WP10c9d6c7c2_active_endpoint_local_truncation_audit"
        )
        decision_branch = "active_C3_endpoint_fails_buffered_C2_passes"
    elif p4_passed and not buffered_passed:
        classification = (
            "short_support_p3_preasymptotic_hypothesis_selected_"
            "no_interface_redesign"
        )
        authorized_next = (
            "WP10c9d6c7c2_short_support_preasymptotic_audit"
        )
        decision_branch = "active_C3_endpoint_passes_buffered_C2_fails"
    else:
        classification = "no_regularized_embedded_profile_class_selected"
        authorized_next = None
        decision_branch = "both_prospective_classes_fail"
    return {
        "classification": classification,
        "authorized_next": authorized_next,
        "decision_branch": decision_branch,
        "profile_reports": profile_reports,
        "p4_active_endpoint_class_passed": p4_passed,
        "p3_exact_zero_buffer_class_passed": buffered_passed,
        "passed": bool(p4_passed and buffered_passed),
        "failed_variants": sorted(direct["failed_packets"]),
        "direct_variant_count": 16 - len(direct["failed_packets"]),
        "alternate_variant_count": 0,
    }


def _parent_hashes() -> dict[str, str]:
    paths = (
        C7C0_MANIFEST,
        C7C0_SUMMARY,
        C7C0_ARRAYS,
        C7C0_PROVENANCE,
        C7C1A_SUMMARY,
        C7C1A_ARRAYS,
        C7C1A_PROVENANCE,
        c7b.PARENT_MANIFEST,
        c7b.PARENT_SUMMARY,
        c7b.PARENT_ARRAYS,
    )
    return {
        str(path.relative_to(ROOT)): _sha256(path)
        for path in paths
    }


def _config(
    manifest: dict,
    c7a_manifest: dict,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "operator_change": False,
        "manifest_sha256": FROZEN_MANIFEST_SHA256,
        "labels": LABELS,
        "base_profiles": BASE_PROFILES,
        "profile_variants": manifest["profile_variants"],
        "decision_contract": manifest["decision_contract"],
        "observable_names": c7b.OBSERVABLE_NAMES,
        "observable_contract": c7a_manifest["observable_contract"],
        "coupling_diagnostic_contract": (
            c7a_manifest["coupling_diagnostic_contract"]
        ),
        "prospective_propagation_contract": (
            c7a_manifest["prospective_propagation_contract"]
        ),
    }


def _finalize(
    *,
    identity: dict,
    manifest: dict,
    c7a_manifest: dict,
    summary: dict,
    arrays: dict[str, np.ndarray],
) -> dict:
    source_hashes, source_manifest = _source_manifest()
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DECISIVE_ARRAYS, **arrays)
    result = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        **identity,
        **summary,
        "operator_changed": False,
        "c7b_rejection_preserved": True,
        "c7c0_manifest_preserved": True,
        "c7c1a_uniform_certification_preserved": True,
        "historical_classifications_preserved": True,
        "manifest_sha256": FROZEN_MANIFEST_SHA256,
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "decisive_arrays_path": str(DECISIVE_ARRAYS.relative_to(ROOT)),
        "decisive_arrays_sha256": _sha256(DECISIVE_ARRAYS),
        "decisive_array_hashes": _array_hashes(arrays),
        "bounded_nonlinear_common_mode_authorized": False,
        "nonlinear_physical_trajectory_authorized": False,
        "production_operator_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "scientific_status": (
            "REGULARIZED EMBEDDED CLASS CERTIFIED"
            if result.get("passed")
            else "REGULARITY DISCRIMINATION DIAGNOSTIC"
        ),
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        **identity,
        "implementation_base_tree": _git_value("rev-parse", "HEAD^{tree}"),
        "working_tree_status": _git_value("status", "--short"),
        "command": (
            "PYTHONPATH=src:scripts python scripts/"
            "run_causal_inner_embedded_regularity_"
            "wp10c9d6c7c1b.py"
        ),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "parent_canonical_hashes": _parent_hashes(),
    }
    _write_json(CONFIG_PATH, _config(manifest, c7a_manifest))
    _write_json(SUMMARY_PATH, result)
    _write_json(PROVENANCE_PATH, provenance)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    _refresh_canonical_catalog()
    return result


def run(*, reuse_propagation_checkpoint: bool = False) -> dict:
    started = time.perf_counter()
    identity = _validate_analyzed_git_identity()
    (
        _c7c0_summary,
        manifest,
        arrays,
        _c7c1a_summary,
        _c7a_parent,
        c7a_inputs,
    ) = _load_authorized_inputs()
    c7a_manifest, c7a_arrays = c7a_inputs
    configurations, layouts, metadata, directions = _build_inputs(
        manifest,
        arrays,
        c7a_manifest,
        c7a_arrays,
    )
    result_arrays: dict[str, np.ndarray] = {
        "times": np.asarray(c7a_arrays["times"], dtype=float),
        "fixed_physical_observable_scales": np.asarray(
            c7a_arrays["fixed_physical_observable_scales"],
            dtype=float,
        ),
        "field_scales": np.asarray(arrays["field_scales"], dtype=float),
        "common_parent_face_indices": np.asarray(
            c7a_arrays["common_parent_face_indices"],
            dtype=int,
        ),
        "common_face_radii_over_rg": np.asarray(
            c7a_arrays["common_face_radii_over_rg"],
            dtype=float,
        ),
    }
    if reuse_propagation_checkpoint:
        propagated, propagation_report, method_reports = _load_checkpoint()
        method_passed = all(
            report["passed"] for report in method_reports.values()
        )
    else:
        tangents, active_audits, method_reports, method_passed = (
            c7b._build_tangents(configurations)
        )
        if not method_passed:
            return _finalize(
                identity=identity,
                manifest=manifest,
                c7a_manifest=c7a_manifest,
                arrays=result_arrays,
                summary={
                    "classification": (
                        "embedded_regularity_method_gate_failed"
                    ),
                    "authorized_next": None,
                    "passed": False,
                    "propagation_executed": False,
                    "method_reports": method_reports,
                    "method_passed": False,
                    "embedded_regularized_profile_class_certified": False,
                    "runtime_seconds": float(
                        time.perf_counter() - started
                    ),
                },
            )
        propagated, propagation_report = c7b._propagate(
            configurations,
            layouts,
            tangents,
            active_audits,
            directions,
            metadata,
            c7a_manifest,
        )
        _save_checkpoint(
            propagated,
            propagation_report,
            method_reports,
        )
    observable_scales = np.asarray(
        c7a_arrays["fixed_physical_observable_scales"],
        dtype=float,
    )
    field_scales = np.asarray(arrays["field_scales"], dtype=float)
    direct, direct_arrays = c7b._comparison_report(
        c7a_manifest,
        layouts,
        metadata,
        propagated,
        observable_scales,
        field_scales,
    )
    with _regularity_profiles():
        coupling, coupling_arrays = c7b._coupling_diagnostics(
            c7a_manifest,
            layouts,
            configurations,
            metadata,
            propagated,
            observable_scales,
            field_scales,
        )
    decision = _prospective_decision(direct, coupling, metadata)
    result_arrays.update(direct_arrays)
    result_arrays.update(coupling_arrays)
    with _regularity_profiles():
        localization = c7b._interface_localization_report(result_arrays)
    maximum_integral_residual = max(
        report["maximum_exact_integral_relative_solve_residual"]
        for report in propagation_report.values()
    )
    passed = bool(
        method_passed
        and maximum_integral_residual <= MAXIMUM_EXACT_INTEGRAL_RESIDUAL
        and decision["passed"]
    )
    classification = (
        decision["classification"]
        if method_passed
        and maximum_integral_residual <= MAXIMUM_EXACT_INTEGRAL_RESIDUAL
        else "embedded_regularity_method_or_integral_gate_failed"
    )
    authorized_next = decision["authorized_next"] if passed else (
        decision["authorized_next"]
        if method_passed
        and maximum_integral_residual <= MAXIMUM_EXACT_INTEGRAL_RESIDUAL
        else None
    )
    result = _finalize(
        identity=identity,
        manifest=manifest,
        c7a_manifest=c7a_manifest,
        arrays=result_arrays,
        summary={
            "classification": classification,
            "authorized_next": authorized_next,
            "passed": passed,
            "propagation_executed": True,
            "method_reports": method_reports,
            "method_passed": method_passed,
            "propagation_report": propagation_report,
            "maximum_exact_integral_relative_solve_residual": (
                maximum_integral_residual
            ),
            "historical_direct_contract_report": direct,
            "prospective_decision": decision,
            "coupling_diagnostic_report": coupling,
            "interface_localization_report": localization,
            "embedded_regularized_profile_class_certified": passed,
            "runtime_seconds": float(time.perf_counter() - started),
        },
    )
    print(
        json.dumps(
            {
                "classification": classification,
                "authorized_next": authorized_next,
                "decision_branch": decision["decision_branch"],
                "direct_failed_count": len(direct["failed_packets"]),
                "direct_variant_count": decision["direct_variant_count"],
                "coupling_diagnostics_passed": coupling["passed"],
                "maximum_exact_integral_residual": (
                    maximum_integral_residual
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def refresh_metadata_only() -> dict:
    if not SUMMARY_PATH.is_file() or not DECISIVE_ARRAYS.is_file():
        raise RuntimeError("WP10c9d6c7c1b canonical evidence is unavailable")
    identity = _validate_analyzed_git_identity()
    (
        _c7c0_summary,
        manifest,
        _arrays,
        _c7c1a_summary,
        _c7a_parent,
        c7a_inputs,
    ) = _load_authorized_inputs()
    c7a_manifest, _c7a_arrays = c7a_inputs
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    arrays = _load_npz(DECISIVE_ARRAYS)
    immutable = {
        key: value
        for key, value in summary.items()
        if key
        not in {
            "implementation_source_hashes",
            "implementation_source_manifest_sha256",
            "decisive_arrays_path",
            "decisive_arrays_sha256",
            "decisive_array_hashes",
        }
    }
    with _regularity_profiles():
        immutable["interface_localization_report"] = (
            c7b._interface_localization_report(arrays)
        )
    return _finalize(
        identity=identity,
        manifest=manifest,
        c7a_manifest=c7a_manifest,
        summary=immutable,
        arrays=arrays,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-propagation-checkpoint", action="store_true")
    parser.add_argument("--refresh-metadata-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.refresh_metadata_only:
        refresh_metadata_only()
    else:
        run(
            reuse_propagation_checkpoint=(
                arguments.reuse_propagation_checkpoint
            )
        )


if __name__ == "__main__":
    main()
