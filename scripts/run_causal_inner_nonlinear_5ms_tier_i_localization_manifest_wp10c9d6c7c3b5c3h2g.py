#!/usr/bin/env python3
"""Freeze the operator-neutral 5 ms Tier-I localization contract."""

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
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_nonlinear_5ms_spatial_certificate_wp10c9d6c7c3b5c3h2f as h2f  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c3h2g"
ANALYZED_BASE_COMMIT = "42a1a020e3c1a24094d9504444b88c5e15963ab3"
ANALYZED_BASE_PARENT = "e2c6979e67ee25e8020dd750eeb4951a5cae5fcb"
ANALYZED_BASE_TREE = "3fc08daccbf65192ae0613064d38d88e99a4aeb9"

ARTIFACT = (
    "causal_inner_nonlinear_5ms_tier_i_localization_manifest_"
    "wp10c9d6c7c3b5c3h2g"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_5ms_tier_i_localization_manifest_"
    "wp10c9d6c7c3b5c3h2g.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_5ms_tier_i_localization_manifest_"
    "wp10c9d6c7c3b5c3h2g.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_5MS_TIER_I_"
    "LOCALIZATION_MANIFEST_WP10C9D6C7C3B5C3H2G_2026-08-08.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "localization_manifest.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

FAILED_COMPONENTS = (
    "inner_flux_mass",
    "inner_flux_angular_momentum",
    "net_drive_mass",
    "net_drive_angular_momentum",
)
TIME_WINDOWS = {
    "early": (0, 1, 2),
    "middle": (3, 4, 5),
    "late": (6, 7, 8),
}
LOCALIZATION_GATES = {
    "maximum_decomposition_closure_defect": 1.0e-12,
    "minimum_layout_map_alignment": 0.95,
    "maximum_common_state_error_fraction": 0.25,
    "minimum_term_error_dominance_fraction": 0.70,
    "minimum_term_error_alignment": 0.90,
    "minimum_time_window_error_energy_fraction": 0.70,
    "maximum_nonlinear_remainder_pair_fraction": 0.25,
    "minimum_tangent_actual_pair_error_cosine": 0.95,
    "maximum_temporal_uncertainty_fraction": 0.10,
}


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


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
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


def _source_identity() -> dict[str, str]:
    return {
        path: _sha256(ROOT / path)
        for path in (THIS_RUNNER, THIS_TEST)
        if (ROOT / path).exists()
    }


def _validate_parent() -> dict:
    parent = _read_json(h2f.SUMMARY_PATH)
    if (
        parent["passed"]
        or parent["classification"]
        != "five_ms_spatial_certificate_rejected_Tier_I_exports_"
        "nonconvergent_later_duration_blocked"
        or parent["authorized_next"]
        != "WP10c9d6c7c3b5c3h2g_Tier_I_spatial_failure_localization_manifest"
        or parent["fourth_duration_rung_manifest_authorized"]
        or parent["fixed_q_micro_solver_authorized"]
        or parent["reduced_slow_evolution_authorized"]
    ):
        raise RuntimeError("h2g localization authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("h2g analyzed identity changed")
    return parent


def _manifest() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            "tier_I_spatial_failure_localization_manifest_frozen_"
            "operator_neutral_diagnostics_authorized"
        ),
        "definitions_only": True,
        "propagation_executed": False,
        "operator_changed": False,
        "production_defaults_changed": False,
        "parent_rejection_preserved": True,
        "failed_components": FAILED_COMPONENTS,
        "common_history_window_seconds": (2.0e-3, 5.0e-3),
        "common_target_count": 9,
        "discriminators": {
            "common_parent_export_map": {
                "definition": (
                    "restrict_each_native_base_and_generic_anchor_state_to_"
                    "the_same_64_cell_parent_then_evaluate_one_coarse_"
                    "nonlinear_export_map_at_face_48"
                ),
                "pairwise_identity": (
                    "native_error_equals_common_state_error_plus_layout_map_error"
                ),
                "selection_gates": {
                    "common_parent_spatial_contract_must_pass": True,
                    "minimum_layout_map_alignment": LOCALIZATION_GATES[
                        "minimum_layout_map_alignment"
                    ],
                    "maximum_common_state_error_fraction": LOCALIZATION_GATES[
                        "maximum_common_state_error_fraction"
                    ],
                    "maximum_closure_defect": LOCALIZATION_GATES[
                        "maximum_decomposition_closure_defect"
                    ],
                },
            },
            "net_drive_balance": {
                "exact_identity": (
                    "net_drive_equals_inner_flux_minus_interface_flux_plus_"
                    "source_remainder"
                ),
                "source_remainder_definition": (
                    "net_drive_minus_inner_flux_plus_interface_flux"
                ),
                "channels": ("mass", "angular_momentum"),
                "terms": ("inner_flux", "minus_interface_flux", "source_remainder"),
                "selection_requires_same_dominant_term_on_both_grid_pairs": True,
                "minimum_dominance_fraction": LOCALIZATION_GATES[
                    "minimum_term_error_dominance_fraction"
                ],
                "minimum_error_alignment": LOCALIZATION_GATES[
                    "minimum_term_error_alignment"
                ],
                "maximum_closure_defect": LOCALIZATION_GATES[
                    "maximum_decomposition_closure_defect"
                ],
            },
            "time_window_error_energy": {
                "nonoverlapping_target_indices": TIME_WINDOWS,
                "selection_requires_same_window_on_both_grid_pairs": True,
                "minimum_error_energy_fraction": LOCALIZATION_GATES[
                    "minimum_time_window_error_energy_fraction"
                ],
            },
            "tangent_nonlinear_pair_error": {
                "definition": (
                    "compare_actual_middle_fine_error_with_discrete_tangent_"
                    "middle_fine_error_and_their_nonlinear_remainder_correction"
                ),
                "maximum_remainder_pair_fraction": LOCALIZATION_GATES[
                    "maximum_nonlinear_remainder_pair_fraction"
                ],
                "minimum_tangent_actual_error_cosine": LOCALIZATION_GATES[
                    "minimum_tangent_actual_pair_error_cosine"
                ],
                "maximum_sampled_temporal_uncertainty_fraction": (
                    LOCALIZATION_GATES["maximum_temporal_uncertainty_fraction"]
                ),
            },
        },
        "decision_tree": (
            "common_map_passes_and_layout_map_gates_pass__authorize_layout_map_audit",
            "stable_inner_term_dominance__authorize_inner_face_half_cell_audit_manifest",
            "stable_interface_term_dominance__authorize_coupling_face_audit_manifest",
            "stable_source_term_dominance__authorize_source_prefix_audit_manifest",
            "stable_time_window_localization__authorize_window_specific_diagnostic_manifest",
            "otherwise__authorize_distributed_observable_coupling_manifest_only",
        ),
        "hard_stops": (
            "do_not_propagate_new_state",
            "do_not_change_operator_path_boundary_or_export_definition",
            "do_not_relax_spatial_or_temporal_gates",
            "do_not_infer_physical_instability_from_export_nonconvergence",
            "do_not_begin_fourth_duration_rung_fixed_Q_or_reduced_evolution",
            "do_not_use_N1024_as_rescue",
        ),
        "authorized_next": (
            "WP10c9d6c7c3b5c3h2g1_operator_neutral_Tier_I_localization"
        ),
    }


def _update_catalog(summary: dict) -> None:
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha256(path),
                    "scientific_status": "CERTIFIED",
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
    catalog = _read_json(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": summary["passed"],
    }
    catalog.update(
        {
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": ANALYZED_BASE_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def main() -> int:
    parent = _validate_parent()
    manifest = _manifest()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": manifest["classification"],
        "passed": True,
        "definitions_only": True,
        "parent_classification_preserved": parent["classification"],
        "operator_neutral_localization_authorized": True,
        "new_propagation_authorized": False,
        "fourth_duration_rung_manifest_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": manifest["authorized_next"],
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "failed_components": FAILED_COMPONENTS,
            "time_windows": TIME_WINDOWS,
            "localization_gates": LOCALIZATION_GATES,
        },
    )
    _write_json(MANIFEST_PATH, manifest)
    _write_json(SUMMARY_PATH, summary)
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "source_parent_commit": ANALYZED_BASE_COMMIT,
            "scientific_status": "CERTIFIED",
            "working_head": _git_value("rev-parse", "HEAD"),
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "parent_summary_sha256": _sha256(h2f.SUMMARY_PATH),
            "implementation_source_hashes": _source_identity(),
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "command": f"PYTHONPATH=src:scripts python {THIS_RUNNER}",
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Nonlinear 5 ms Tier-I localization manifest WP10c9d6c7c3b5c3h2g",
                "",
                "## Classification",
                "",
                f"`{manifest['classification']}`",
                "",
                "This definitions-only package preserves the rejected 5 ms spatial "
                "certificate and freezes four operator-neutral discriminators: one "
                "common-parent export map, an exact inner/interface/source net-drive "
                "identity, non-overlapping time-window error energies, and the "
                "actual-versus-discrete-tangent middle/fine error.",
                "",
                "No state is propagated and no operator, boundary, path, export "
                "definition, tolerance, or production default changes. Only the "
                "operator-neutral localization is authorized next. Later duration, "
                "fixed-Q experiments, and reduced slow evolution remain blocked.",
                "",
            )
        ),
        encoding="utf-8",
    )
    names = ("config.json", "localization_manifest.json", "provenance.json", "summary.json")
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
