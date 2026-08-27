#!/usr/bin/env python3
"""Freeze the equilibrium entropy-coordinate numerical repair."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_equilibrium_column_thermodynamic_potential_implementation_wp10c9d6c7c3b5c4f25fizzc as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizzc1_"
    "equilibrium_entropy_coordinate_numerics_repair_manifest"
)
CLASSIFICATION = "equilibrium_entropy_coordinate_numerics_repair_manifest_frozen"
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizzc2_"
    "equilibrium_compensated_coordinate_implementation"
)
ARTIFACT = (
    "causal_inner_equilibrium_entropy_coordinate_numerics_repair_manifest_"
    "wp10c9d6c7c3b5c4f25fizzc1"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_EQUILIBRIUM_ENTROPY_"
    "COORDINATE_NUMERICS_REPAIR_MANIFEST_"
    "WP10C9D6C7C3B5C4F25FIZZC1_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_equilibrium_entropy_coordinate_numerics_"
    "repair_manifest_wp10c9d6c7c3b5c4f25fizzc1.py"
)
THIS_TEST = (
    "tests/test_causal_inner_equilibrium_entropy_coordinate_numerics_"
    "repair_manifest_wp10c9d6c7c3b5c4f25fizzc1.py"
)
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "ca6b3e787d1b133a90e5caf3086f0d3a0df52ab6158d6e97e04b8a46f2026c75"
)
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _utils():
    return parent._utils()


def _validate_parent(*, require_clean: bool) -> dict:
    utils = _utils()
    if utils._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != PARENT_CHECKSUM_MANIFEST_SHA256:
        raise RuntimeError("equilibrium rejection checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = utils._read_json(parent.CANONICAL_DIRECTORY / "certificate_metrics.json")
    if (
        summary["classification"] != parent.FAIL_CLASSIFICATION
        or summary["passed"]
        or summary["equilibrium_physical_potential_certified"]
        or summary["authorized_next"] is not None
        or summary["complete_cycle_execution_authorized"]
        or metrics["maximum_physical_current_relative_defect"] != 2.1729141848488282e-09
        or metrics["maximum_complex_step_current_jacobian_relative_defect"] > 1.0e-9
        or metrics["maximum_first_law_or_gibbs_duhem_relative_defect"] > 1.0e-11
    ):
        raise RuntimeError("equilibrium rejection classification changed")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("repair manifest needs a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "metrics": metrics}


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "preserved_rejection": {
            "artifact": parent.ARTIFACT,
            "classification": parent.FAIL_CLASSIFICATION,
            "retroactive_pass_forbidden": True,
            "maximum_physical_current_relative_defect": 2.1729141848488282e-09,
            "sixth_order_derivative_finite": False,
        },
        "diagnosis": {
            "physical_thermodynamic_failure_selected": False,
            "evidence": {
                "maximum_first_law_or_Gibbs_Duhem_defect": 3.6985658598517593e-16,
                "maximum_complex_step_current_derivative_defect": 4.656616497605051e-10,
            },
            "method_failures": (
                "alpha=mu/T was rounded to one float before cancellation of c^2/(R*T)",
                "one global coordinate scale let the Kerr azimuthal covector set time/radial finite-difference steps",
            ),
            "interpretation": (
                "the exact master potential remains mathematically supported; "
                "its numerical entropy chart and independent stencil were not conditioned"
            ),
        },
        "prospective_repair": {
            "physical_master_potential_unchanged": "X^mu=2H*p(alpha,T)*beta^mu",
            "physical_EOS_unchanged": True,
            "entropy_field_count_unchanged": 5,
            "alpha_representation": (
                "construct and carry the single mathematical alpha coordinate in "
                "extended precision so its rest-mass and thermal parts are not "
                "prematurely rounded; this is representation, not an added field"
            ),
            "beta_representation": (
                "canonicalize the unit-timelike velocity and form beta in extended "
                "precision; return physical currents in ordinary floating precision"
            ),
            "finite_difference_scaling": (
                "scale each beta_i with 1/(T*sqrt(abs(g^ii))) and its own magnitude, "
                "then divide by c^2/(R*T); no global coordinate-component maximum"
            ),
            "complex_step": "retain the real-axis-preserving complex evaluation",
            "coefficient_fit_to_witness_forbidden": True,
        },
        "binding_rerun": {
            "same_47_frozen_physical_witnesses": True,
            "same_parent_envelope": True,
            "physical_current_relative_defect_maximum": 1.0e-10,
            "first_law_or_Gibbs_Duhem_relative_defect_maximum": 1.0e-11,
            "complex_step_current_derivative_relative_defect_maximum": 1.0e-9,
            "sixth_order_current_derivative_relative_defect_maximum": 2.0e-5,
            "density_affinity_roundtrip_relative_defect_maximum": 2.0e-9,
            "fail_closed": True,
        },
        "claim_boundary": {
            "equilibrium_physical_potential_certified": False,
            "dynamic_height_potential_certified": False,
            "full_shear_master_potential_certified": False,
            "eleven_field_local_closure_certified": False,
            "eleven_field_trajectory_authorized": False,
            "complete_cycle_execution_authorized": False,
        },
        "budget": {"trajectory_steps": 0, "complete_cycle_steps": 0, "maximum_wall_minutes": 10},
        "authorized_next": AUTHORIZED_NEXT,
    }


def _update_catalog(summary: dict) -> None:
    utils = _utils()
    with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": utils._sha256(path), "scientific_status": "SUPPORTED"})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    catalog = utils._read_json(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": CLASSIFICATION, "passed": True}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": utils._git("rev-parse", "HEAD"), "latest_work_package": WORK_PACKAGE})
    utils._write_json(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("repair manifest already exists")
    validated = _validate_parent(require_clean=True)
    utils = _utils(); contract = _contract()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "repair_contract.json", contract)
    summary = {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "classification": CLASSIFICATION, "passed": True, "definitions_only": True, "equilibrium_rejection_preserved": True, "physical_thermodynamic_failure_selected": False, "equilibrium_physical_potential_certified": False, "dynamic_height_potential_certified": False, "eleven_field_trajectory_authorized": False, "complete_cycle_execution_authorized": False, "authorized_next": AUTHORIZED_NEXT}
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utils._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"parent_artifact": parent.ARTIFACT, "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256, "parent_hashes": validated["hashes"]})
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(("# Equilibrium entropy-coordinate numerical repair manifest", "", f"Classification: `{CLASSIFICATION}`.", "", "The fixed-height certificate remains rejected. Its exact first-law, Gibbs-Duhem, and complex-step identities passed, while a prematurely rounded rest-mass affinity and a coordinate-global Kerr stencil selected the failure.", "", "The repair changes numerical representation and metric-local perturbation scales only. The EOS, master potential, physical gates, 47 witnesses, and claim boundary remain unchanged.", "", f"Authorized next: `{AUTHORIZED_NEXT}`.", "")), encoding="utf-8")
    sources = (THIS_RUNNER, THIS_TEST, REPORT_RELATIVE)
    utils._write_json(CANONICAL_DIRECTORY / "provenance.json", {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "implementation_commit": utils._git("rev-parse", "HEAD"), "implementation_tree": utils._git("rev-parse", "HEAD^{tree}"), "source_hashes": {path: utils._sha256(ROOT / path) for path in sources}, "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "thread_environment": {name: os.environ.get(name, "") for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utils._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8")
    _update_catalog(summary); return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--freeze", action="store_true"); args = parser.parse_args()
    if not args.freeze: parser.error("choose --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
