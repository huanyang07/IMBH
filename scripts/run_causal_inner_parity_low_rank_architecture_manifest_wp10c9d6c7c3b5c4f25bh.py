#!/usr/bin/env python3
"""Freeze a no-new-truth parity and low-rank architecture diagnosis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_amplitude_0p02_departure_chart_preflight_wp10c9d6c7c3b5c4f25bg as parent  # noqa: E402
import run_causal_inner_expanded_departure_rate_screen_wp10c9d6c7c3b5c4f25be as rate_0p01  # noqa: E402
import run_causal_inner_guarded_departure_rate_screen_wp10c9d6c7c3b5c4f25ba as rate_0p005  # noqa: E402
import run_causal_inner_explicit_nonlinear_470_architecture_audit_wp10c9d6c7c3b5c4f25aw as architecture  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25bh"
CLASSIFICATION = (
    "post_result_parity_low_rank_architecture_diagnosis_manifest_frozen_"
    "no_new_truth_calls"
)
PARENT_COMMIT = "0f268de618b9f1c2a4e0611e1cbeea7cd045c887"
PARENT_PARENT = "8fafa60b2a28e622585a875eae281ac847e4591d"
PARENT_TREE = "121128366cf917b5d0c9280156dd4c54f517b114"

ARTIFACT = (
    "causal_inner_parity_low_rank_architecture_manifest_"
    "wp10c9d6c7c3b5c4f25bh"
)
ARTIFACT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_parity_low_rank_architecture_manifest_"
    "wp10c9d6c7c3b5c4f25bh.py"
)
THIS_TEST = (
    "tests/test_causal_inner_parity_low_rank_architecture_manifest_"
    "wp10c9d6c7c3b5c4f25bh.py"
)
NEXT_RUNNER = (
    "scripts/run_causal_inner_parity_low_rank_architecture_audit_"
    "wp10c9d6c7c3b5c4f25bi.py"
)
NEXT_TEST = (
    "tests/test_causal_inner_parity_low_rank_architecture_audit_"
    "wp10c9d6c7c3b5c4f25bi.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_PARITY_LOW_RANK_"
    "ARCHITECTURE_MANIFEST_WP10C9D6C7C3B5C4F25BH_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

RATE_0P005_PATH = rate_0p005.CANONICAL_DIRECTORY / "departure_rate_screen.npz"
RATE_0P01_PATH = rate_0p01.CANONICAL_DIRECTORY / "departure_rate_screen.npz"
CHART_0P02_PATH = parent.CANONICAL_DIRECTORY / "amplitude_0p02_departure_chart.npz"
GEOMETRY_PATH = architecture.CANONICAL_DIRECTORY / "online_470_geometry.npz"

ACTIVE_INPUT_DIMENSION = 8
DEPARTURE_DIMENSION = 28
QUADRATIC_OUTPUT_RANK = 3
CUBIC_OUTPUT_RANK = 4
QUADRATIC_FEATURE_COUNT = 36
CUBIC_FEATURE_COUNT = 120
COMPRESSED_POLYNOMIAL_COEFFICIENT_COUNT = 588
INHERITED_NONLINEAR_SIGNAL_THRESHOLD = 0.10


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, int):
        return value
    return value


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _checksums(directory: Path) -> dict[str, str]:
    recorded = {}
    for line in (directory / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        if _sha(directory / name) != expected:
            raise RuntimeError(f"checksum mismatch: {directory / name}")
        recorded[name] = expected
    return recorded


def _validate_parents() -> dict:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("parity diagnosis parent commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("parity diagnosis parent lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("parity diagnosis parent tree changed")
    packages = {
        "rejected_0p02_chart": parent.CANONICAL_DIRECTORY,
        "rate_0p01": rate_0p01.CANONICAL_DIRECTORY,
        "rate_0p005": rate_0p005.CANONICAL_DIRECTORY,
        "architecture_470": architecture.CANONICAL_DIRECTORY,
    }
    hashes = {name: _checksums(path) for name, path in packages.items()}
    rejected = _read(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics_0p02 = _read(parent.CANONICAL_DIRECTORY / "metrics.json")
    summary_0p01 = _read(rate_0p01.CANONICAL_DIRECTORY / "summary.json")
    summary_0p005 = _read(rate_0p005.CANONICAL_DIRECTORY / "summary.json")
    architecture_summary = _read(architecture.CANONICAL_DIRECTORY / "summary.json")
    if (
        rejected["passed"]
        or rejected["classification"] != parent.FAIL_CLASSIFICATION
        or metrics_0p02["checks"]["transverse_distortion"]
        or not all(
            value
            for key, value in metrics_0p02["checks"].items()
            if key != "transverse_distortion"
        )
    ):
        raise RuntimeError("amplitude-0.02 axial-chart rejection changed")
    if (
        not summary_0p01["passed"]
        or not summary_0p005["passed"]
        or summary_0p01["component_bound"] != 1.0e-2
        or summary_0p01["completed_nonbase_rate_evaluations"] != 16
        or summary_0p005["completed_nonbase_rate_evaluations"] != 48
    ):
        raise RuntimeError("signed-pair rate evidence changed")
    if (
        not architecture_summary["passed"]
        or architecture_summary["selected_architecture"]
        != "explicit_nonlinear_conservative_IMEX_470"
    ):
        raise RuntimeError("470-state architecture certificate changed")
    return {
        "hashes": hashes,
        "rejected": rejected,
        "metrics_0p02": metrics_0p02,
        "summary_0p01": summary_0p01,
        "summary_0p005": summary_0p005,
        "architecture_summary": architecture_summary,
    }


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "post_result_architecture_diagnosis": True,
        "independent_validation_claimed": False,
        "objective": (
            "decompose_the_existing_signed_pair_rate_residuals_into_even_"
            "quadratic_and_odd_cubic_terms_and_select_a_testable_low_rank_"
            "nonlinear_departure_architecture_without_new_truth_calls"
        ),
        "fixed_470_state_partition": {
            "exact_physical_coordinates": 162,
            "stable_memory_coordinates": 280,
            "nonlinear_departure_coordinates": DEPARTURE_DIMENSION,
            "truncated_stable_remainder": 90,
            "online_truth_calls_per_macrostep": 0,
        },
        "parity_decomposition": {
            "quadratic_even_term": (
                "one_half_of_the_sum_of_positive_and_negative_nonlinear_"
                "departure_rate_residuals_divided_by_radius_squared"
            ),
            "cubic_odd_term": (
                "one_half_of_the_difference_of_positive_and_negative_"
                "nonlinear_departure_rate_residuals_divided_by_radius_cubed"
            ),
            "amplitudes": [5.0e-3, 1.0e-2],
            "signed_direction_pairs_per_amplitude": ACTIVE_INPUT_DIMENSION,
            "new_truth_rate_evaluations": 0,
            "new_retractions": 0,
            "new_roots": 0,
            "propagated_states": 0,
        },
        "candidate_departure_architecture": {
            "active_input": "xi_equals_W8_transpose_times_a28",
            "quadratic_output_rank": QUADRATIC_OUTPUT_RANK,
            "cubic_output_rank": CUBIC_OUTPUT_RANK,
            "form": (
                "a_dot_equals_La_plus_U2_B2_phi2_xi_plus_U3_C3_xi_plus_"
                "parametric_physical_memory_coupling"
            ),
            "symmetric_quadratic_features": QUADRATIC_FEATURE_COUNT,
            "symmetric_cubic_features": CUBIC_FEATURE_COUNT,
            "compressed_full_polynomial_coefficient_upper_bound": (
                COMPRESSED_POLYNOMIAL_COEFFICIENT_COUNT
            ),
            "cubic_input_tensor_structure_must_be_selected_on_heldout_mixed_data": True,
        },
        "diagnostic_consistency_gates": {
            "signed_pairs_per_amplitude_equal": ACTIVE_INPUT_DIMENSION,
            "median_even_relative_signal_at_0p01_min": (
                INHERITED_NONLINEAR_SIGNAL_THRESHOLD
            ),
            "median_even_relative_amplification_min": 1.5,
            "median_even_relative_amplification_max": 2.5,
            "median_odd_relative_amplification_min": 3.0,
            "median_odd_relative_amplification_max": 5.0,
            "maximum_quadratic_coefficient_relative_change": 0.10,
            "minimum_quadratic_coefficient_cosine": 0.999,
            "maximum_cubic_coefficient_relative_change": 0.25,
            "minimum_cubic_coefficient_cosine": 0.995,
            "quadratic_row_normalized_rank3_energy_min": 0.95,
            "cubic_row_normalized_rank4_energy_min": 0.95,
            "amplitude_0p02_transverse_distortion_min": 0.05,
            "new_truth_rate_evaluations_equal": 0,
            "new_retractions_equal": 0,
            "new_roots_equal": 0,
            "propagated_states_equal": 0,
        },
        "decision": {
            "diagnosis_consistent": {
                "classification": (
                    "quadratic_cubic_low_rank_departure_architecture_"
                    "diagnosed_mixed_direction_database_manifest_authorized"
                ),
                "authorizes_only": (
                    "definitions_only_active8_mixed_direction_parity_database_manifest"
                ),
            },
            "diagnosis_inconsistent": {
                "classification": (
                    "parity_low_rank_architecture_diagnosis_inconsistent_"
                    "mixed_direction_database_blocked"
                ),
                "authorizes_only": None,
            },
        },
        "claim_boundary": {
            "thresholds_were_selected_blind_to_existing_results": False,
            "mixed_direction_coefficients_identified": False,
            "cubic_input_tensor_structure_identified": False,
            "heldout_mixed_direction_validation_completed": False,
            "online_integrator_implemented": False,
            "predictive_cycle_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
    }


def _update_catalog(summary: dict) -> None:
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(ARTIFACT_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha(path),
                    "scientific_status": "PROSPECTIVE_POST_RESULT_DIAGNOSTIC",
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case", "path", "bytes", "sha256", "scientific_status"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(ARTIFACT_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": True,
    }
    catalog.update(
        {
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": PARENT_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    parent_data = _validate_parents()
    if _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("parity architecture manifest requires a clean tracked tree")
    if ARTIFACT_DIRECTORY.exists():
        raise RuntimeError("parity architecture manifest is already frozen")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "post_result_architecture_diagnosis": True,
        "independent_validation_claimed": False,
        "planned_new_truth_rate_evaluations": 0,
        "planned_new_retractions": 0,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": "WP10c9d6c7c3b5c4f25bi",
    }
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write(ARTIFACT_DIRECTORY / "contract.json", _contract())
    _write(ARTIFACT_DIRECTORY / "summary.json", summary)
    _write(
        ARTIFACT_DIRECTORY / "parent_lock.json",
        {
            "parent_commit": PARENT_COMMIT,
            "parent_parent": PARENT_PARENT,
            "parent_tree": PARENT_TREE,
            "package_hashes": parent_data["hashes"],
            "decisive_input_hashes": {
                "rate_0p005": _sha(RATE_0P005_PATH),
                "rate_0p01": _sha(RATE_0P01_PATH),
                "chart_0p02": _sha(CHART_0P02_PATH),
                "online_470_geometry": _sha(GEOMETRY_PATH),
            },
        },
    )
    _write(
        ARTIFACT_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "PROSPECTIVE_POST_RESULT_DIAGNOSTIC",
            "definition_commit": _git("rev-parse", "HEAD"),
            "definition_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "authorized_next_runner": NEXT_RUNNER,
            "authorized_next_test": NEXT_TEST,
            "report": REPORT_RELATIVE,
            "source_hashes": {
                THIS_RUNNER: _sha(ROOT / THIS_RUNNER),
                THIS_TEST: _sha(ROOT / THIS_TEST),
            },
            "python": sys.version,
            "platform": platform.platform(),
            "thread_environment": rate_0p01.manifest.parent.chart_tools.coordinate_tools.THREAD_ENVIRONMENT,
        },
    )
    names = tuple(sorted(path.name for path in ARTIFACT_DIRECTORY.iterdir()))
    (ARTIFACT_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(ARTIFACT_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Parity low-rank architecture manifest WP10c9d6c7c3b5c4f25bh",
                "",
                "## Classification",
                "",
                f"`{CLASSIFICATION}`",
                "",
                "This is an explicitly post-result architecture diagnosis, not an independent validation. It makes no new truth calls and decomposes the already committed signed-pair residuals into even/quadratic and odd/cubic components.",
                "",
                "The candidate keeps the certified 162+280+28 state partition and adds only a low-rank quadratic-cubic closure to the departure block. A pass may authorize only a prospective active-8 mixed-direction database manifest.",
                "",
            )
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if not args.freeze:
        raise SystemExit("pass --freeze")
    print(json.dumps(_plain(_freeze()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
