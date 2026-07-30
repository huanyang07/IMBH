#!/usr/bin/env python3
"""Freeze a deterministic band-limited cancellation feasibility search."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import sys

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_canonical_json_sha256,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c6e2a"
ANALYZED_BASE_COMMIT = "5c644d2d5912ceca6c661bec6f5db55c798095a9"
ANALYZED_BASE_PARENT = "8e7b567d5f64b28db8405726586e1bf78fe9da67"
ANALYZED_BASE_TREE = "eab4a8acabe556fd28d781f33512351bcd82242f"
THIS_RUNNER = (
    "scripts/"
    "run_causal_inner_bandlimited_balance_manifest_wp10c9d6c6e2a.py"
)

CANDIDATE_LIBRARY = tuple(
    {
        "candidate_id": f"p{power}_cos{harmonic}",
        "base_power": power,
        "modulation_harmonic": harmonic,
        "envelope_formula": (
            "sin(pi*x)^base_power * "
            "(1 + alpha*cos(modulation_harmonic*pi*x))"
        ),
    }
    for power in (2, 3, 4)
    for harmonic in (1, 2)
)

SEARCH_CONTRACT = {
    "families": ("inward_shear", "outward_shear"),
    "coordinate": (
        "x=(lnR-lnR_inner)/(lnR_outer-lnR_inner)"
    ),
    "amplitude": 1.0e-2,
    "balance_functional": (
        "primary_769_node_continuum_global_"
        "candidate_lower_height_work_angular_action"
    ),
    "coefficient_rule": (
        "alpha=-L[sin(pi*x)^p]/"
        "L[sin(pi*x)^p*cos(m*pi*x)]"
    ),
    "primary_continuum_nodes": 769,
    "secondary_continuum_nodes": 513,
    "maximum_absolute_coefficient": 50.0,
    "maximum_coefficient_relative_769_513_difference": 1.0e-6,
    "maximum_secondary_initial_cancellation_ratio": 1.0e-6,
    "spectral_energy_quantile": 0.99,
    "maximum_theta_99": 0.30,
    "maximum_nyquist_alias_fraction": 1.0e-3,
    "maximum_endpoint_cell_fraction": 5.0e-3,
    "minimum_global_family_purity": 0.995,
    "minimum_active_cell_family_purity": 0.98,
    "maximum_projection_defect": 2.0e-12,
    "maximum_inward_outward_coefficient_difference": 1.0e-10,
    "pair_must_pass_both_shear_families": True,
    "selection_key": (
        "maximum_theta_99_across_families",
        "maximum_alias_fraction_across_families",
        "base_power",
        "modulation_harmonic",
    ),
    "selection_is_lexicographic_ascending": True,
    "propagated_history_objective_forbidden": True,
    "operator_changed": False,
    "propagation_executed": False,
}

PARENT_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_integral_conditioning_validation_wp10c9d6c6e1"
)
PARENT_SUMMARY = PARENT_DIRECTORY / "summary.json"
PARENT_CONFIG = PARENT_DIRECTORY / "config.json"
PARENT_ARRAYS = PARENT_DIRECTORY / "decisive_arrays.npz"
PARENT_PROVENANCE = PARENT_DIRECTORY / "provenance.json"
CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_bandlimited_balance_manifest_wp10c9d6c6e2a"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "search_manifest.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

IMPLEMENTATION_SOURCES = (
    THIS_RUNNER,
    "tests/"
    "test_causal_inner_bandlimited_balance_manifest_wp10c9d6c6e2a.py",
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


def _git_value(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _identity() -> dict:
    resolved = _git_value("rev-parse", ANALYZED_BASE_COMMIT)
    parent = _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
    tree = _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
    if (
        resolved != ANALYZED_BASE_COMMIT
        or parent != ANALYZED_BASE_PARENT
        or tree != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("WP10c9d6c6e2a analyzed git identity changed")
    return {
        "analyzed_base_commit": resolved,
        "analyzed_base_parent": parent,
        "analyzed_base_tree": tree,
    }


def _source_manifest() -> tuple[dict[str, str], str]:
    hashes = {
        path: _sha256(ROOT / path) for path in IMPLEMENTATION_SOURCES
    }
    digest = hashlib.sha256()
    for path, value in sorted(hashes.items()):
        digest.update(path.encode("utf-8"))
        digest.update(value.encode("ascii"))
    return hashes, digest.hexdigest()


def _refresh_sha256s() -> None:
    entries = []
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if not path.is_file() or path.name == "SHA256SUMS.txt":
            continue
        entries.append(f"{_sha256(path)}  {path.name}")
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "\n".join(entries) + "\n",
        encoding="utf-8",
    )


def _refresh_catalog() -> None:
    rows = []
    for case in sorted(CANONICAL_DIRECTORY.parent.iterdir()):
        provenance_path = case / "provenance.json"
        if not case.is_dir() or not provenance_path.is_file():
            continue
        provenance = json.loads(
            provenance_path.read_text(encoding="utf-8")
        )
        status = provenance.get(
            "scientific_status",
            provenance.get("numerical_status", "DIAGNOSTIC ONLY"),
        )
        for path in sorted(case.iterdir()):
            if path.is_file():
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
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": ANALYZED_BASE_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, summary)


def run() -> dict:
    identity = _identity()
    parent = json.loads(PARENT_SUMMARY.read_text(encoding="utf-8"))
    if (
        parent["classification"] != "frozen_integral_profiles_ineligible"
        or parent["propagation_executed"]
        or parent["authorized_next"] != "none"
    ):
        raise RuntimeError("c6e2a predecessor status changed")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "parent_commit": ANALYZED_BASE_COMMIT,
        "candidate_library": CANDIDATE_LIBRARY,
        "search_contract": SEARCH_CONTRACT,
        "evaluation_executed": False,
        "propagation_executed": False,
        "operator_changed": False,
    }
    manifest = {
        **payload,
        "manifest_sha256": causal_canonical_json_sha256(payload),
    }
    source_hashes, source_manifest = _source_manifest()
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(MANIFEST_PATH, manifest)
    _write_json(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "candidate_count": len(CANDIDATE_LIBRARY),
            "search_contract": SEARCH_CONTRACT,
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        **identity,
        "classification": (
            "bandlimited_conditioning_search_frozen_"
            "feasibility_authorized"
        ),
        "authorized_next": (
            "WP10c9d6c6e2b_bandlimited_balance_feasibility"
        ),
        "passed": True,
        "operator_changed": False,
        "evaluation_executed": False,
        "propagation_executed": False,
        "candidate_count": len(CANDIDATE_LIBRARY),
        "manifest_sha256": manifest["manifest_sha256"],
        "parent_classification": parent["classification"],
        "parent_classification_preserved": True,
        "c6c_c6d_c6e0_c6e1_status_preserved": True,
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "embedded_export_discrimination_authorized": False,
        "nonlinear_physical_trajectory_authorized": False,
        "production_operator_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    _write_json(SUMMARY_PATH, summary)
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "DEFINITIONS-ONLY SEARCH MANIFEST",
            "source_parent_commit": ANALYZED_BASE_COMMIT,
            **identity,
            "implementation_base_tree": _git_value(
                "rev-parse",
                "HEAD^{tree}",
            ),
            "working_tree_status": _git_value("status", "--short"),
            "command": (
                "PYTHONPATH=src python "
                "scripts/"
                "run_causal_inner_bandlimited_balance_manifest_"
                "wp10c9d6c6e2a.py"
            ),
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "platform": platform.platform(),
            },
            "implementation_source_hashes": source_hashes,
            "implementation_source_manifest_sha256": source_manifest,
            "parent_canonical_hashes": {
                str(path.relative_to(ROOT)): _sha256(path)
                for path in (
                    PARENT_CONFIG,
                    PARENT_SUMMARY,
                    PARENT_ARRAYS,
                    PARENT_PROVENANCE,
                )
            },
        },
    )
    _refresh_sha256s()
    _refresh_catalog()
    print(
        json.dumps(
            {
                "classification": summary["classification"],
                "candidate_count": summary["candidate_count"],
                "manifest_sha256": summary["manifest_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return summary


if __name__ == "__main__":
    run()
