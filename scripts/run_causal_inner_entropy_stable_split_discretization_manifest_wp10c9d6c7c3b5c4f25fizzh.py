#!/usr/bin/env python3
"""Freeze the entropy-stable split discretization for the port atlas."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_fully_split_physical_port_atlas_kernel_wp10c9d6c7c3b5c4f25fizzg1 as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = parent.AUTHORIZED_NEXT
CLASSIFICATION = "entropy_stable_split_discretization_manifest_frozen"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25fizzh1_entropy_stable_split_discretization_kernel"
PASS_NEXT = "definitions_only_WP10c9d6c7c3b5c4f25fizzi_local_nonlinear_atlas_trust_region_manifest"
ARTIFACT = "causal_inner_entropy_stable_split_discretization_manifest_wp10c9d6c7c3b5c4f25fizzh"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_STABLE_SPLIT_DISCRETIZATION_MANIFEST_WP10C9D6C7C3B5C4F25FIZZH_2026-08-26.md"
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = "scripts/run_causal_inner_entropy_stable_split_discretization_manifest_wp10c9d6c7c3b5c4f25fizzh.py"
THIS_TEST = "tests/test_causal_inner_entropy_stable_split_discretization_manifest_wp10c9d6c7c3b5c4f25fizzh.py"
PARENT_SHA256 = "45b9f9fa5e26101850e885132633eb30bebb8a3df5c19147ba52fd3535205572"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _u():
    return parent._u()


def _validate_parent(*, require_clean: bool = False) -> dict:
    utils = _u()
    if utils._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != PARENT_SHA256:
        raise RuntimeError("physical port-atlas certificate checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = utils._read_json(parent.CANONICAL_DIRECTORY / "kernel_metrics.json")
    if (
        not summary["passed"]
        or not summary["fully_split_physical_port_atlas_kernel_certified"]
        or summary["authorized_next"] != WORK_PACKAGE
        or metrics["field_count"] != 11
        or metrics["trajectory_steps"] != 0
        or summary["complete_cycle_execution_authorized"]
    ):
        raise RuntimeError("physical port-atlas classification changed")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("split discretization manifest needs a clean tracked tree")
    return hashes


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "parent_architecture": "certified state-local 4+5+2 physical port atlas",
        "spatial_discretization": {
            "grid": "periodic uniform proof grid; boundaries require a later SAT certificate",
            "entropy_coordinates": "the frozen anchor coordinates with A0=I",
            "two_point_flux": "f*(yL,yR)=A_r*(yL+yR)/2",
            "entropy_dissipation": "-(lambda/2)*(yR-yL), lambda >= spectral_radius(A_r)",
            "equivalent_operator": "skew centered derivative tensor A_r plus symmetric negative jump penalty",
            "constant_state_preservation": True,
            "interface_entropy_inequality": True,
        },
        "time_discretization": {
            "reversible_transport": "implicit midpoint/Cayley update",
            "local_ports": "implicit midpoint/Cayley update of the frozen source generator",
            "composition": "source half-step, transport full-step, source half-step",
            "formal_order": 2,
            "transport_ledger": "energy change equals midpoint numerical-interface dissipation",
            "source_ledger": "reservoir loss equals thermal heat deposit exactly at the midpoint",
        },
        "nonlinear_policy": {
            "coefficients": "frozen for one accepted split step",
            "reanchor": "explicit only after the step; no mid-step coefficient mutation",
            "acceptance": "all substeps and the combined step must close their entropy/heat ledgers",
            "rejection": "rejected states never enter history",
        },
        "kernel": {
            "physical_anchors": 47,
            "proof_grid_cells": 7,
            "deterministic_state_probes_per_anchor": 3,
            "time_step_factors": [0.2, 0.1, 0.05, 0.025],
            "operator_symmetry_gate": 2e-13,
            "energy_ledger_gate": 2e-12,
            "constant_state_gate": 2e-13,
            "second_order_gate": 1.8,
            "trajectory_steps": 0,
        },
        "decision": {
            "pass_classification": "entropy_stable_split_discretization_kernel_certified",
            "pass_authorized_next": PASS_NEXT,
            "failure_classification": "entropy_stable_split_discretization_kernel_failed",
        },
        "claim_boundary": {
            "definitions_only": True,
            "nonlinear_trust_region_certified": False,
            "trajectory_authorized": False,
            "complete_cycle_execution_authorized": False,
        },
        "authorized_next": AUTHORIZED_NEXT,
    }


def _update(summary: dict) -> None:
    utils = _u()
    rows = list(csv.DictReader(CANONICAL_MANIFEST.open(newline="", encoding="utf-8")))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": utils._sha256(path), "scientific_status": "SUPPORTED"})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    catalog = utils._read_json(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": CLASSIFICATION, "passed": True}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": utils._git("rev-parse", "HEAD"), "latest_work_package": WORK_PACKAGE})
    utils._write_json(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("split discretization manifest already exists")
    parent_hashes = _validate_parent(require_clean=True)
    utils = _u()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "discretization_contract.json", _contract())
    summary = {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "classification": CLASSIFICATION, "passed": True, "definitions_only": True, "physical_port_atlas_preserved": True, "split_discretization_certified": False, "trajectory_authorized": False, "complete_cycle_execution_authorized": False, "authorized_next": AUTHORIZED_NEXT}
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utils._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"parent_artifact": parent.ARTIFACT, "parent_checksum_manifest_sha256": PARENT_SHA256, "parent_hashes": parent_hashes})
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# Entropy-stable split-discretization manifest\n\n"
        "Classification: `entropy_stable_split_discretization_manifest_frozen`.\n\n"
        "The prospective proof method combines a symmetric two-point entropy flux, jump dissipation, midpoint/Cayley transport and port updates, and Strang composition. Every reservoir loss is paired with a thermal heat deposit. Coefficients remain fixed inside each accepted substep.\n\n"
        "This package is definitions-only. It authorizes no nonlinear trust region, trajectory, or complete-cycle execution.\n",
        encoding="utf-8",
    )
    sources = (THIS_RUNNER, THIS_TEST, REPORT_RELATIVE)
    utils._write_json(CANONICAL_DIRECTORY / "provenance.json", {"implementation_commit": utils._git("rev-parse", "HEAD"), "source_hashes": {path: utils._sha256(ROOT / path) for path in sources}})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utils._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8")
    _update(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if not args.freeze:
        parser.error("choose --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
