#!/usr/bin/env python3
"""Run the frozen c7c0 endpoint-regularity controls uniformly.

This is phase one of the c7c1 fail-fast contract.  The four new C3-endpoint
and C2-buffered shear bases are propagated on the unchanged uniform
N128/N256/N512 monolithic tangents.  Embedded propagation is authorized only
if all sixteen frozen sign/amplitude variants pass the direct state and
thirteen-export contract.
"""

from __future__ import annotations

import argparse
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

import run_causal_inner_continuum_lift_wp10c9d6c3 as c3
import run_causal_inner_embedded_regularity_manifest_wp10c9d6c7c0 as c7c0
import run_causal_inner_integral_conditioning_validation_wp10c9d6c6e1 as c6e1
import run_causal_inner_packet_validation_wp10c9d6c6c as c6c

from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_canonical_json_sha256,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c1a"
ANALYZED_BASE_COMMIT = "406ab7e18d6094d07647efb91dccce06808087b4"
ANALYZED_BASE_PARENT = "fad76852220d7c304fff9016ff99ada64d404eff"
ANALYZED_BASE_TREE = "c14275ee307e90911d76298575fb633d0bd205ba"
FROZEN_MANIFEST_SHA256 = (
    "b230ce7a3c7e7546d0d706ee8f9bcfa3102c6c69be5f67a29aa451e1b5d9706b"
)
THIS_RUNNER = (
    "scripts/"
    "run_causal_inner_embedded_regularity_uniform_wp10c9d6c7c1a.py"
)

LABELS = tuple(c6e1.LABELS)
MAXIMUM_EXACT_INTEGRAL_RESIDUAL = 1.0e-12
MAXIMUM_PROPAGATION_SCALING_DEFECT = 1.0e-10

PARENT_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_embedded_regularity_manifest_wp10c9d6c7c0"
)
PARENT_MANIFEST = PARENT_DIRECTORY / "regularity_manifest.json"
PARENT_SUMMARY = PARENT_DIRECTORY / "summary.json"
PARENT_ARRAYS = PARENT_DIRECTORY / "decisive_arrays.npz"
PARENT_PROVENANCE = PARENT_DIRECTORY / "provenance.json"

F0_MANIFEST = (
    ROOT
    / "results/canonical/"
    "causal_inner_band_envelope_manifest_wp10c9d6c6f0/"
    "band_envelope_manifest.json"
)
C3_ARRAYS = (
    ROOT
    / "results/canonical/"
    "causal_inner_continuum_lift_wp10c9d6c3/decisive_arrays.npz"
)

CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_embedded_regularity_uniform_wp10c9d6c7c1a"
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
    / "causal_inner_embedded_regularity_uniform_"
    "wp10c9d6c7c1a_propagated.npz"
)
PROPAGATION_CHECKPOINT_REPORT = (
    CHECKPOINT_DIRECTORY
    / "causal_inner_embedded_regularity_uniform_"
    "wp10c9d6c7c1a_report.json"
)

IMPLEMENTATION_SOURCES = (
    THIS_RUNNER,
    "scripts/"
    "run_causal_inner_embedded_regularity_manifest_wp10c9d6c7c0.py",
    "scripts/"
    "run_causal_inner_integral_conditioning_validation_wp10c9d6c6e1.py",
    "scripts/run_causal_inner_packet_validation_wp10c9d6c6c.py",
    "scripts/run_causal_inner_continuum_lift_wp10c9d6c3.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_packet_validation.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_monolithic_tangent.py",
    "tests/"
    "test_causal_inner_embedded_regularity_uniform_wp10c9d6c7c1a.py",
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
        raise RuntimeError("WP10c9d6c7c1a analyzed git identity changed")
    return {
        "analyzed_base_commit": resolved,
        "analyzed_base_parent": parent,
        "analyzed_base_tree": tree,
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


def _load_parent() -> tuple[dict, dict, dict[str, np.ndarray], dict]:
    parent = json.loads(PARENT_SUMMARY.read_text(encoding="utf-8"))
    manifest = json.loads(PARENT_MANIFEST.read_text(encoding="utf-8"))
    arrays = _load_npz(PARENT_ARRAYS)
    if (
        parent["classification"]
        != "endpoint_interface_regularity_manifest_frozen_"
        "uniform_control_preflight_authorized"
        or parent["manifest_sha256"] != FROZEN_MANIFEST_SHA256
        or not parent["passed"]
        or manifest["manifest_sha256"] != FROZEN_MANIFEST_SHA256
        or manifest["propagation_executed"]
    ):
        raise RuntimeError("frozen WP10c9d6c7c0 authorization changed")
    stored = manifest.pop("manifest_sha256")
    calculated = causal_canonical_json_sha256(manifest)
    manifest["manifest_sha256"] = stored
    if calculated != FROZEN_MANIFEST_SHA256:
        raise RuntimeError("WP10c9d6c7c0 manifest hash changed")
    if set(arrays) != set(parent["decisive_array_hashes"]):
        raise RuntimeError("WP10c9d6c7c0 array set changed")
    for name, expected in parent["decisive_array_hashes"].items():
        if causal_array_sha256(arrays[name]) != expected:
            raise RuntimeError(f"WP10c9d6c7c0 array changed: {name}")
    f0 = json.loads(F0_MANIFEST.read_text(encoding="utf-8"))
    return parent, manifest, arrays, f0


def _adapter(manifest: dict, f0: dict) -> dict:
    return {
        "base_profile_definitions": manifest["profile_definitions"],
        "profile_variants": manifest["profile_variants"],
        "prospective_propagation_contract": dict(
            f0["prospective_propagation_contract"]
        ),
    }


def _build_inputs(
    manifest: dict,
    arrays: dict[str, np.ndarray],
    f0: dict,
) -> tuple[dict, dict, dict, dict]:
    configurations, construction_arrays, construction_report = (
        c3._build_continuum_configurations()
    )
    base_directions = {label: {} for label in LABELS}
    for label in LABELS:
        columns = np.asarray(
            configurations[label]["primitive_column_scales"],
            dtype=float,
        )
        for name in manifest["profile_definitions"]:
            primary = np.asarray(
                arrays[f"{name}__{label}__primary_physical"],
                dtype=float,
            )
            secondary = np.asarray(
                arrays[f"{name}__{label}__secondary_physical"],
                dtype=float,
            )
            base_directions[label][name] = {
                "primary_scaled": primary.ravel() / columns,
                "secondary_scaled": secondary.ravel() / columns,
            }
    adapter = _adapter(manifest, f0)
    directions, metadata = c6e1._variant_directions(
        adapter,
        base_directions,
    )
    return (
        configurations,
        construction_arrays,
        directions,
        metadata,
    )


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
            "parent_decisive_arrays_sha256": _sha256(PARENT_ARRAYS),
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
        raise RuntimeError("WP10c9d6c7c1a checkpoint is unavailable")
    report = json.loads(
        PROPAGATION_CHECKPOINT_REPORT.read_text(encoding="utf-8")
    )
    if (
        report["manifest_sha256"] != FROZEN_MANIFEST_SHA256
        or report["parent_decisive_arrays_sha256"] != _sha256(PARENT_ARRAYS)
        or tuple(report["labels"]) != LABELS
        or report["checkpoint_sha256"] != _sha256(PROPAGATION_CHECKPOINT)
    ):
        raise RuntimeError("WP10c9d6c7c1a checkpoint changed")
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


def _direct_decision(
    manifest: dict,
    direct: dict,
    metadata: dict,
) -> dict:
    reports = {}
    for packet_id, base_index, multiplier in zip(
        metadata["packet_ids"],
        metadata["base_indices"],
        metadata["multipliers"],
        strict=True,
    ):
        parent = direct["packet_reports"][packet_id]
        scaling = max(parent["propagation_scaling_defects"].values())
        passed = bool(
            parent["passed"]
            and scaling <= MAXIMUM_PROPAGATION_SCALING_DEFECT
        )
        reports[packet_id] = {
            "base_profile": metadata["base_names"][int(base_index)],
            "multiplier": float(multiplier),
            "route": "direct_contract" if passed else "failed",
            "passed": passed,
        }
    return {
        "variant_reports": reports,
        "failed_variants": sorted(
            name for name, report in reports.items() if not report["passed"]
        ),
        "direct_variant_count": sum(
            report["passed"] for report in reports.values()
        ),
        "passed": all(report["passed"] for report in reports.values()),
        "profile_definitions_preserved": (
            set(manifest["profile_definitions"])
            == set(metadata["base_names"])
        ),
    }


def _parent_hashes() -> dict[str, str]:
    paths = (
        PARENT_MANIFEST,
        PARENT_SUMMARY,
        PARENT_ARRAYS,
        PARENT_PROVENANCE,
        F0_MANIFEST,
        C3_ARRAYS,
    )
    return {
        str(path.relative_to(ROOT)): _sha256(path)
        for path in paths
    }


def _config(manifest: dict, f0: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "operator_change": False,
        "manifest_sha256": FROZEN_MANIFEST_SHA256,
        "labels": LABELS,
        "profile_definitions": manifest["profile_definitions"],
        "profile_variants": manifest["profile_variants"],
        "prospective_propagation_contract": (
            f0["prospective_propagation_contract"]
        ),
        "phase_contract": (
            manifest["future_propagation_contract"]["phase_1"]
        ),
    }


def _finalize(
    *,
    identity: dict,
    parent: dict,
    manifest: dict,
    f0: dict,
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
        "parent_classification": parent["classification"],
        "parent_classification_preserved": True,
        "c7b_rejection_preserved": True,
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
            "UNIFORM CONTROLS CERTIFIED"
            if result.get("passed")
            else "UNIFORM CONTROLS REJECTED"
        ),
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        **identity,
        "implementation_base_tree": _git_value("rev-parse", "HEAD^{tree}"),
        "working_tree_status": _git_value("status", "--short"),
        "command": (
            "PYTHONPATH=src:scripts python scripts/"
            "run_causal_inner_embedded_regularity_uniform_"
            "wp10c9d6c7c1a.py"
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
    _write_json(CONFIG_PATH, _config(manifest, f0))
    _write_json(SUMMARY_PATH, result)
    _write_json(PROVENANCE_PATH, provenance)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    _refresh_canonical_catalog()
    return result


def run(*, reuse_propagation_checkpoint: bool = False) -> dict:
    started = time.perf_counter()
    identity = _validate_analyzed_git_identity()
    parent, manifest, parent_arrays, f0 = _load_parent()
    (
        configurations,
        construction_arrays,
        directions,
        metadata,
    ) = _build_inputs(manifest, parent_arrays, f0)
    if reuse_propagation_checkpoint:
        propagated, propagation_report, method_reports = (
            _load_checkpoint()
        )
        method_passed = all(
            report["passed"] for report in method_reports.values()
        )
    else:
        print(
            "WP10c9d6c7c1a: build unchanged uniform monolithic tangents",
            flush=True,
        )
        tangents, observable_maps, method_reports, _baselines = (
            c3._build_tangents(configurations, construction_arrays)
        )
        method_passed = all(
            method_reports[label]["passed"] for label in LABELS
        )
        if not method_passed:
            return _finalize(
                identity=identity,
                parent=parent,
                manifest=manifest,
                f0=f0,
                arrays={
                    "field_scales": np.asarray(
                        parent_arrays["field_scales"],
                        dtype=float,
                    )
                },
                summary={
                    "classification": (
                        "endpoint_interface_regularity_uniform_"
                        "method_gate_failed"
                    ),
                    "authorized_next": None,
                    "passed": False,
                    "propagation_executed": False,
                    "method_reports": method_reports,
                    "method_passed": False,
                    "uniform_controls_certified": False,
                    "embedded_regularity_discrimination_authorized": False,
                    "runtime_seconds": float(
                        time.perf_counter() - started
                    ),
                },
            )
        propagated, propagation_report = c6e1._propagate(
            configurations,
            tangents,
            observable_maps,
            directions,
            metadata,
        )
        _save_checkpoint(
            propagated,
            propagation_report,
            method_reports,
        )

    pseudo_manifest = c6e1._parent_contract_manifest(
        _adapter(manifest, f0)
    )
    direct, direct_arrays = c6c._comparison_report(
        pseudo_manifest,
        configurations,
        construction_arrays,
        metadata,
        propagated,
    )
    decision = _direct_decision(manifest, direct, metadata)
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
        "endpoint_interface_regularity_uniform_controls_certified_"
        "embedded_discrimination_authorized"
        if passed
        else "endpoint_interface_regularity_uniform_controls_failed"
    )
    authorized_next = (
        "WP10c9d6c7c1b_embedded_regularity_discrimination"
        if passed
        else None
    )
    result_arrays = dict(direct_arrays)
    result_arrays["field_scales"] = np.asarray(
        parent_arrays["field_scales"],
        dtype=float,
    )
    result = _finalize(
        identity=identity,
        parent=parent,
        manifest=manifest,
        f0=f0,
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
            "uniform_controls_certified": passed,
            "embedded_regularity_discrimination_authorized": passed,
            "runtime_seconds": float(time.perf_counter() - started),
        },
    )
    print(
        json.dumps(
            {
                "classification": classification,
                "authorized_next": authorized_next,
                "failed_variants": decision["failed_variants"],
                "direct_variant_count": decision[
                    "direct_variant_count"
                ],
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
        raise RuntimeError("WP10c9d6c7c1a canonical evidence is unavailable")
    identity = _validate_analyzed_git_identity()
    parent, manifest, _parent_arrays, f0 = _load_parent()
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
    return _finalize(
        identity=identity,
        parent=parent,
        manifest=manifest,
        f0=f0,
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
