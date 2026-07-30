#!/usr/bin/env python3
"""Propagate the frozen WP10c9d6c6f0 band-envelope manifest.

The exact 20 inherited sign/amplitude variants are propagated with the
unchanged N128/N256/N512 monolithic tangents.  The historical direct
component gate remains primary.  Only lower-height-work angular momentum
may use the frozen proof-style signed-band error envelope.
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
import run_causal_inner_integral_conditioning_validation_wp10c9d6c6e1 as c6e1
import run_causal_inner_packet_validation_wp10c9d6c6c as c6c

from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_canonical_json_sha256,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c6f1"
ANALYZED_BASE_COMMIT = "7005aa11bb22f40862abf886bc3f7fee26ec68b8"
ANALYZED_BASE_PARENT = "595a200bd2218eb0dfdfc2478f2706f917bc561b"
ANALYZED_BASE_TREE = "ec566666a6447c3cc9186da101bc2430b8fff75d"
FROZEN_MANIFEST_SHA256 = (
    "221a271dd861226bbc09eaf430dfc6bef47ad39a5b5d7e6e53520f9d75fcb643"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_band_envelope_validation_"
    "wp10c9d6c6f1.py"
)

LABELS = ("uniform_N128", "uniform_N256", "uniform_N512")
TARGET_OBSERVABLE_NAME = "vertical_work_angular_momentum"
MAXIMUM_PROPAGATION_SCALING_DEFECT = 1.0e-10

PARENT_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_band_envelope_manifest_wp10c9d6c6f0"
)
PARENT_CONFIG = PARENT_DIRECTORY / "config.json"
PARENT_SUMMARY = PARENT_DIRECTORY / "summary.json"
PARENT_MANIFEST = PARENT_DIRECTORY / "band_envelope_manifest.json"
PARENT_PROVENANCE = PARENT_DIRECTORY / "provenance.json"

E0_MANIFEST = (
    ROOT
    / "results/canonical/"
    "causal_inner_integral_conditioning_manifest_wp10c9d6c6e0/"
    "conditioning_manifest.json"
)
E1_ARRAYS = (
    ROOT
    / "results/canonical/"
    "causal_inner_integral_conditioning_validation_wp10c9d6c6e1/"
    "decisive_arrays.npz"
)
C3_ARRAYS = (
    ROOT
    / "results/canonical/"
    "causal_inner_continuum_lift_wp10c9d6c3/decisive_arrays.npz"
)

CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_band_envelope_validation_wp10c9d6c6f1"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

IMPLEMENTATION_SOURCES = (
    THIS_RUNNER,
    "scripts/run_causal_inner_band_envelope_manifest_wp10c9d6c6f0.py",
    "scripts/"
    "run_causal_inner_integral_conditioning_validation_wp10c9d6c6e1.py",
    "scripts/run_causal_inner_packet_validation_wp10c9d6c6c.py",
    "scripts/run_causal_inner_height_localization_wp10c9d6c6d.py",
    "scripts/run_causal_inner_continuum_lift_wp10c9d6c3.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_integral_conditioning.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_height_localization.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_packet_validation.py",
    "tests/test_causal_inner_band_envelope_validation_wp10c9d6c6f1.py",
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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
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
        raise RuntimeError("WP10c9d6c6f1 analyzed git identity changed")
    return {
        "analyzed_base_commit": resolved,
        "analyzed_base_parent": parent,
        "analyzed_base_tree": tree,
    }


def _source_manifest() -> tuple[dict[str, str], str]:
    hashes = {path: _sha256(ROOT / path) for path in IMPLEMENTATION_SOURCES}
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
        "w", newline="", encoding="utf-8"
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


def _load_manifest() -> tuple[dict, dict, dict]:
    parent = json.loads(PARENT_SUMMARY.read_text(encoding="utf-8"))
    manifest = json.loads(PARENT_MANIFEST.read_text(encoding="utf-8"))
    if (
        parent["classification"]
        != (
            "band_envelope_contract_and_heldout_profiles_frozen_"
            "uniform_propagation_authorized"
        )
        or parent["authorized_next"]
        != "WP10c9d6c6f1_prospective_band_envelope_propagation"
        or not parent["passed"]
    ):
        raise RuntimeError("WP10c9d6c6f0 propagation authorization changed")
    stored = manifest.pop("manifest_sha256")
    calculated = causal_canonical_json_sha256(manifest)
    manifest["manifest_sha256"] = stored
    if (
        stored != FROZEN_MANIFEST_SHA256
        or calculated != FROZEN_MANIFEST_SHA256
        or parent["manifest_sha256"] != FROZEN_MANIFEST_SHA256
        or len(manifest["base_profile_definitions"]) != 5
        or len(manifest["profile_variants"]) != 20
        or manifest["propagation_executed"]
        or not all(
            item["binding"] for item in manifest["profile_variants"]
        )
    ):
        raise RuntimeError("frozen WP10c9d6c6f0 manifest changed")
    e0 = json.loads(E0_MANIFEST.read_text(encoding="utf-8"))
    adapter = dict(manifest)
    adapter["eligibility_contract"] = e0["eligibility_contract"]
    conditioning = dict(manifest["component_route_contract"])
    conditioning["historical_minimum_relative_activity"] = conditioning[
        "minimum_relative_activity"
    ]
    adapter["integral_conditioning_contract"] = conditioning
    return parent, manifest, adapter


def _verify_regenerated_projections(
    manifest: dict,
    arrays: dict[str, np.ndarray],
) -> dict:
    expected = manifest["profile_projection_hashes"]
    missing = sorted(set(expected) - set(arrays))
    mismatched = sorted(
        name
        for name, digest in expected.items()
        if name in arrays and causal_array_sha256(arrays[name]) != digest
    )
    report = {
        "expected_projection_count": len(expected),
        "missing_projection_arrays": missing,
        "mismatched_projection_arrays": mismatched,
        "maximum_projection_hash_defect": (
            1.0 if missing or mismatched else 0.0
        ),
        "passed": not missing and not mismatched,
    }
    return report


def _prospective_decision(
    manifest: dict,
    direct: dict,
    conditioning: dict,
    metadata: dict,
) -> dict:
    propagation = manifest["prospective_propagation_contract"]
    reports = {}
    alternate_bases = set()
    direct_bases = set()
    for packet_id, base_index, multiplier in zip(
        metadata["packet_ids"],
        metadata["base_indices"],
        metadata["multipliers"],
        strict=True,
    ):
        name = metadata["base_names"][int(base_index)]
        parent = direct["packet_reports"][packet_id]
        if parent["passed"]:
            passed = True
            route = "historical_direct_contract"
            direct_bases.add(name)
        else:
            histories_pass = True
            used_alternate = False
            for history_name, metric_name in (
                ("instantaneous", "instantaneous_exports"),
                ("cumulative", "cumulative_exports"),
            ):
                metric = parent[metric_name]
                low = {
                    component
                    for component, order in metric[
                        "component_orders"
                    ].items()
                    if order < 0.75
                }
                if low and low != {TARGET_OBSERVABLE_NAME}:
                    histories_pass = False
                if TARGET_OBSERVABLE_NAME in low:
                    decision = conditioning[name][history_name]["decision"]
                    histories_pass = bool(
                        histories_pass
                        and decision["passed"]
                        and decision["route"]
                        == "cancellation_conditioned_band_envelope"
                    )
                    used_alternate = True
                else:
                    histories_pass = bool(
                        histories_pass
                        and metric["minimum_significant_component_order"]
                        >= 0.75
                    )
                histories_pass = bool(
                    histories_pass
                    and c6e1._other_export_gates_pass(
                        metric,
                        propagation,
                    )
                )
            passed = bool(
                histories_pass
                and used_alternate
                and parent["state_reference"]["passed"]
                and max(parent["propagation_scaling_defects"].values())
                <= MAXIMUM_PROPAGATION_SCALING_DEFECT
            )
            route = (
                "proof_style_cancellation_conditioned_band_envelope"
                if passed
                else "failed"
            )
            if passed:
                alternate_bases.add(name)
        reports[packet_id] = {
            "base_profile": name,
            "multiplier": float(multiplier),
            "historical_direct_passed": parent["passed"],
            "route": route,
            "passed": passed,
        }
    return {
        "variant_reports": reports,
        "all_variants_passed": all(
            item["passed"] for item in reports.values()
        ),
        "direct_base_profiles": sorted(direct_bases),
        "alternate_base_profiles": sorted(alternate_bases),
        "direct_variant_count": sum(
            item["route"] == "historical_direct_contract"
            for item in reports.values()
        ),
        "alternate_variant_count": sum(
            item["route"]
            == "proof_style_cancellation_conditioned_band_envelope"
            for item in reports.values()
        ),
        "failed_variants": sorted(
            name for name, item in reports.items() if not item["passed"]
        ),
        "passed": all(item["passed"] for item in reports.values()),
    }


def _config(manifest: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "operator_change": False,
        "manifest_sha256": FROZEN_MANIFEST_SHA256,
        "labels": LABELS,
        "time_horizon_s": 0.125,
        "time_sample_count": 65,
        "component_route_contract": manifest["component_route_contract"],
        "prospective_propagation_contract": (
            manifest["prospective_propagation_contract"]
        ),
    }


def _parent_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): _sha256(path)
        for path in (
            PARENT_CONFIG,
            PARENT_SUMMARY,
            PARENT_MANIFEST,
            PARENT_PROVENANCE,
            E0_MANIFEST,
            E1_ARRAYS,
        )
    }


def _finalize(
    *,
    identity: dict,
    parent: dict,
    manifest: dict,
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
        "historical_classifications_preserved": True,
        "c6c_rejection_preserved": True,
        "manifest_sha256": FROZEN_MANIFEST_SHA256,
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "decisive_arrays_path": str(DECISIVE_ARRAYS.relative_to(ROOT)),
        "decisive_arrays_sha256": _sha256(DECISIVE_ARRAYS),
        "decisive_array_hashes": {
            name: _array_sha256(values)
            for name, values in sorted(arrays.items())
        },
        "nonlinear_physical_trajectory_authorized": False,
        "production_operator_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "scientific_status": (
            "PROSPECTIVE UNIFORM CLASS CERTIFIED"
            if result.get("passed")
            else "REJECTED OR STOPPED"
        ),
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        **identity,
        "implementation_base_tree": _git_value(
            "rev-parse", "HEAD^{tree}"
        ),
        "working_tree_status": _git_value("status", "--short"),
        "command": (
            "PYTHONPATH=src:scripts python scripts/"
            "run_causal_inner_band_envelope_validation_wp10c9d6c6f1.py"
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
    _write_json(CONFIG_PATH, _config(manifest))
    _write_json(SUMMARY_PATH, result)
    _write_json(PROVENANCE_PATH, provenance)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    _refresh_canonical_catalog()
    return result


def run() -> dict:
    started = time.perf_counter()
    identity = _validate_analyzed_git_identity()
    parent, manifest, adapter = _load_manifest()
    (
        configurations,
        construction_arrays,
        base_directions,
        _evaluators,
        continuum_references,
        eligibility,
        arrays,
    ) = c6e1._build_inputs(adapter)
    projection_replay = _verify_regenerated_projections(manifest, arrays)
    construction_passed = bool(
        eligibility["passed"] and projection_replay["passed"]
    )
    if not construction_passed:
        result = _finalize(
            identity=identity,
            parent=parent,
            manifest=manifest,
            arrays=arrays,
            summary={
                "classification": (
                    "band_envelope_profile_reconstruction_failed"
                ),
                "authorized_next": "none",
                "passed": False,
                "propagation_executed": False,
                "eligibility_report": eligibility,
                "projection_replay_report": projection_replay,
                "embedded_export_discrimination_authorized": False,
                "runtime_seconds": float(time.perf_counter() - started),
            },
        )
        return result

    variant_directions, metadata = c6e1._variant_directions(
        adapter,
        base_directions,
    )
    print("WP10c9d6c6f1: build unchanged monolithic tangents", flush=True)
    tangents, observable_maps, method_reports, _baselines = (
        c3._build_tangents(configurations, construction_arrays)
    )
    method_passed = all(
        method_reports[label]["passed"] for label in LABELS
    )
    propagated, propagation_report = c6e1._propagate(
        configurations,
        tangents,
        observable_maps,
        variant_directions,
        metadata,
    )
    pseudo_manifest = c6e1._parent_contract_manifest(adapter)
    direct, direct_arrays = c6c._comparison_report(
        pseudo_manifest,
        configurations,
        construction_arrays,
        metadata,
        propagated,
    )
    with np.load(C3_ARRAYS, allow_pickle=False) as source:
        observable_scales = np.asarray(
            source["fixed_physical_observable_scales"],
            dtype=float,
        )
    continuum, continuum_arrays = c6e1._continuum_ratios(
        configurations,
        tangents,
        propagated,
        continuum_references,
        metadata,
        observable_scales,
    )
    conditioning, conditioning_arrays = c6e1._conditioning_report(
        adapter,
        configurations,
        propagated,
        metadata,
        continuum,
        observable_scales,
    )
    prospective = _prospective_decision(
        manifest,
        direct,
        conditioning,
        metadata,
    )
    maximum_integral_residual = max(
        report["maximum_exact_integral_relative_solve_residual"]
        for report in propagation_report.values()
    )
    passed = bool(
        construction_passed
        and method_passed
        and maximum_integral_residual <= 1.0e-12
        and prospective["passed"]
    )
    if passed:
        classification = (
            "prospective_band_envelope_uniform_validation_certified"
        )
        authorized_next = "WP10c9d6c7_embedded_discrimination"
    else:
        classification = "prospective_band_envelope_uniform_validation_failed"
        authorized_next = "none"
    arrays.update(direct_arrays)
    arrays.update(continuum_arrays)
    arrays.update(conditioning_arrays)
    arrays["fixed_physical_observable_scales"] = observable_scales
    for label in LABELS:
        arrays[f"{label}__base_cell_actions"] = propagated[label][
            "base_cell_actions"
        ]
        arrays[f"{label}__base_cumulative_cell_actions"] = propagated[
            label
        ]["base_cumulative_cell_actions"]
    result = _finalize(
        identity=identity,
        parent=parent,
        manifest=manifest,
        arrays=arrays,
        summary={
            "classification": classification,
            "authorized_next": authorized_next,
            "passed": passed,
            "propagation_executed": True,
            "eligibility_report": eligibility,
            "projection_replay_report": projection_replay,
            "method_reports": {
                label: method_reports[label] for label in LABELS
            },
            "method_passed": method_passed,
            "propagation_report": propagation_report,
            "maximum_exact_integral_relative_solve_residual": (
                maximum_integral_residual
            ),
            "historical_direct_contract_report": direct,
            "continuum_reference_report": continuum,
            "band_envelope_report": conditioning,
            "prospective_decision": prospective,
            "uniform_profile_class_certified": passed,
            "embedded_export_discrimination_authorized": passed,
            "runtime_seconds": float(time.perf_counter() - started),
        },
    )
    print(
        json.dumps(
            {
                "classification": classification,
                "authorized_next": authorized_next,
                "historical_direct_failed_count": len(
                    direct["failed_packets"]
                ),
                "prospective_failed_count": len(
                    prospective["failed_variants"]
                ),
                "direct_variant_count": prospective[
                    "direct_variant_count"
                ],
                "alternate_variant_count": prospective[
                    "alternate_variant_count"
                ],
                "alternate_base_profiles": prospective[
                    "alternate_base_profiles"
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def refresh_metadata_only() -> dict:
    if not SUMMARY_PATH.exists() or not DECISIVE_ARRAYS.exists():
        raise RuntimeError("WP10c9d6c6f1 canonical evidence is unavailable")
    identity = _validate_analyzed_git_identity()
    parent, manifest, _adapter = _load_manifest()
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    with np.load(DECISIVE_ARRAYS, allow_pickle=False) as source:
        arrays = {
            name: np.array(source[name], copy=True)
            for name in source.files
        }
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
        summary=immutable,
        arrays=arrays,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-metadata-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.refresh_metadata_only:
        refresh_metadata_only()
    else:
        run()


if __name__ == "__main__":
    main()
