#!/usr/bin/env python3
"""Freeze the physical entropy congruence and AP macrostep architecture."""

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

import run_causal_inner_short_restartable_equilibrium_core_trajectory_wp10c9d6c7c3b5c4f25fizzk1 as parent  # noqa: E402


WORK_PACKAGE = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizzl_"
    "physical_entropy_congruence_and_AP_macrostep_manifest"
)
CLASSIFICATION = "physical_entropy_congruence_and_AP_macrostep_manifest_frozen"
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizzl1_"
    "physical_entropy_congruence_and_AP_kernel"
)
PASS_NEXT = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizzm_"
    "bounded_AP_coarse_trajectory_manifest"
)
ARTIFACT = (
    "causal_inner_physical_entropy_congruence_and_ap_macrostep_manifest_"
    "wp10c9d6c7c3b5c4f25fizzl"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_PHYSICAL_ENTROPY_CONGRUENCE_AND_AP_MACROSTEP_MANIFEST_"
    "WP10C9D6C7C3B5C4F25FIZZL_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_physical_entropy_congruence_and_ap_macrostep_"
    "manifest_wp10c9d6c7c3b5c4f25fizzl.py"
)
THIS_TEST = (
    "tests/test_causal_inner_physical_entropy_congruence_and_ap_macrostep_"
    "manifest_wp10c9d6c7c3b5c4f25fizzl.py"
)
PARENT_SHA256 = "69bf3c9b74e5c11033b45d1ead440cd56d52804e7d4ed0366d186a971a3f41b4"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _u():
    return parent._u()


def _validate_parent(require_clean=False):
    utility = _u()
    if utility._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != PARENT_SHA256:
        raise RuntimeError("short trajectory certificate checksum changed")
    hashes = utility._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utility._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    if (
        not summary["passed"]
        or not summary["short_restartable_equilibrium_core_trajectory_certified"]
        or not summary["physical_entropy_congruence_manifest_authorized"]
        or summary["full_eleven_field_trajectory_certified"]
        or summary["complete_cycle_execution_authorized"]
    ):
        raise RuntimeError("short trajectory certificate classification changed")
    if require_clean and utility._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("congruence manifest needs a clean tracked tree")
    return hashes


def _contract():
    return {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "diagnosis": {
            "old_algebraic_kernel_preserved": True,
            "old_coordinate_map_status": (
                "the special-relativistic scalar Mobius map is only an "
                "algebraic rest-frame fixture; it is not the physical "
                "Kerr-Schild congruence when shift and transverse velocity are nonzero"
            ),
            "required_repair": (
                "derive the four-current entropy metric from physical tangents and "
                "use the exact Valencia/Kerr-Schild characteristic map"
            ),
        },
        "physical_congruence": {
            "primitive_chart": "q=(log rho, log T, v_R/c, v_phi/c)",
            "tangents": "fourth-order centered physical-primitive stencil",
            "primitive_step": 3.0e-4,
            "matrices": {
                "U_q": "derivative of (J^t,T^tt,T^tR,T^tphi)",
                "F_q": "derivative of (J^R,T^Rt,T^RR,T^Rphi)",
                "v_q": "derivative of compensated (alpha,beta_t,beta_R,beta_phi)",
                "H": "v_q U_q^{-1}=partial v/partial U",
                "K": "H-whitened physical flux Jacobian F_q U_q^{-1}",
            },
            "witness_indices": [0, 10, 20, 30, 40, 46],
            "gates": {
                "minimum_scaled_entropy_metric_eigenvalue": 1.0e-12,
                "maximum_whitened_symmetry_defect": 2.0e-6,
                "maximum_Valencia_spectrum_defect": 2.0e-6,
                "maximum_core_reconstruction_defect": 2.0e-6,
            },
        },
        "corrected_eleven_field_port": {
            "rest_matrix": "certified acoustic plus five-STF incidence matrix",
            "coordinate_map": (
                "spectral Valencia map including lapse, radial shift, radial "
                "velocity, and transverse velocity"
            ),
            "core_orientation": (
                "orthogonal spectral congruence aligns the four rest-core "
                "coordinates with the physical H-whitened core"
            ),
            "source": "same dissipative five-shear plus damped Hamiltonian height port",
            "gates": {
                "symmetry": 2.0e-12,
                "source_positive_part": 2.0e-12,
                "local_light_cone_margin": 2.0e-12,
            },
        },
        "AP_macrostep": {
            "fast_generator": "L_k=-i k A_port+S_port for each coarse radial mode",
            "propagator": "exact or rational-Krylov exponential/phi action",
            "mode_policy": (
                "retain neutral and weakly damped modes explicitly; eliminate "
                "only stable modes separated by a prospectively verified gap"
            ),
            "slow_forcing": (
                "conservative loading, radiation, tide, wind and boundary ports "
                "enter through phi_1(dt L) forcing, followed by entropy retraction"
            ),
            "stiff_limit": (
                "stable fast coordinates converge to their forced slow manifold "
                "without shrinking the macrostep"
            ),
            "kernel_step_ratios": [1.0e-3, 1.0, 1.0e3],
            "gates": {
                "maximum_semigroup_expansivity": 1.0e-10,
                "maximum_composition_defect": 2.0e-11,
                "maximum_stiff_limit_defect": 2.0e-8,
            },
        },
        "cycle_cost_contract": {
            "cycle_seconds": 578880.0,
            "maximum_online_macrosteps": 100000,
            "maximum_wall_days": 3.0,
            "online_truth_residual_calls": 0,
            "online_fixed_Q_microsteps": 0,
            "coefficients": "offline hash-locked physical atlas only",
        },
        "decision": {
            "pass_classification": "physical_entropy_congruence_and_AP_kernel_certified",
            "failure_classification": "physical_entropy_congruence_and_AP_kernel_failed",
            "pass_authorized_next": PASS_NEXT,
        },
        "claim_boundary": {
            "bounded_AP_trajectory_authorized": False,
            "complete_cycle_execution_authorized": False,
        },
        "authorized_next": AUTHORIZED_NEXT,
    }


def _update_catalog(summary):
    utility = _u()
    rows = list(csv.DictReader(CANONICAL_MANIFEST.open(newline="", encoding="utf-8")))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": utility._sha256(path), "scientific_status": "SUPPORTED"})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    catalog = utility._read_json(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": CLASSIFICATION, "passed": True}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": utility._git("rev-parse", "HEAD"), "latest_work_package": WORK_PACKAGE})
    utility._write_json(CANONICAL_SUMMARY, catalog)


def _freeze():
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("congruence/AP manifest exists")
    hashes = _validate_parent(require_clean=True); utility = _u()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utility._write_json(CANONICAL_DIRECTORY / "architecture_contract.json", _contract())
    summary = {"schema_version": 1, "work_package": WORK_PACKAGE, "classification": CLASSIFICATION, "passed": True, "definitions_only": True, "old_algebraic_kernel_preserved": True, "physical_entropy_congruence_certified": False, "AP_macrostep_certified": False, "bounded_AP_trajectory_authorized": False, "complete_cycle_execution_authorized": False, "authorized_next": AUTHORIZED_NEXT}
    utility._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utility._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"parent_artifact": parent.ARTIFACT, "parent_checksum_manifest_sha256": PARENT_SHA256, "parent_hashes": hashes})
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("# Physical entropy-congruence and AP macrostep manifest\n\nThe short nonlinear core trajectory passed, but a full eleven-field trajectory is not yet authorized. This package prospectively derives the physical entropy congruence, replaces the algebraic special-relativistic coordinate fixture with the exact Kerr-Schild/Valencia spectral map, and tests an exponential asymptotic-preserving fast propagator.\n\nThe target cycle architecture retains neutral and weakly damped modes, eliminates only spectrally separated stable modes, and uses zero online truth residuals. No bounded AP trajectory or complete cycle is executed here.\n", encoding="utf-8")
    sources = (THIS_RUNNER, THIS_TEST, REPORT_RELATIVE)
    utility._write_json(CANONICAL_DIRECTORY / "provenance.json", {"implementation_commit": utility._git("rev-parse", "HEAD"), "source_hashes": {source: utility._sha256(ROOT / source) for source in sources}})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utility._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8")
    _update_catalog(summary); return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--freeze", action="store_true"); arguments = parser.parse_args()
    if not arguments.freeze: parser.error("choose --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
