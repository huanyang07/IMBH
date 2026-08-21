#!/usr/bin/env python3
"""Execute fail-fast exact fixed-Q preflight of saved cold candidates."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_cold_branch_anchor_preflight_manifest_wp10c9d6c7c3b5c4f25dx as manifest  # noqa: E402
import run_causal_inner_exact_geometric_470_chart_preflight_wp10c9d6c7c3b5c4f25de as exact_chart  # noqa: E402
import run_causal_inner_face36_fixed_q_authentic_history_ladder_wp10c9d6c7c3b5c4f24e1 as rate_source  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25dy"
PASS_CLASSIFICATION = "saved_cold_exact_fixed_Q_anchor_supported_hidden_root_manifest_authorized"
FAIL_CLASSIFICATION = "saved_cold_candidates_not_near_fixed_macro_critical_manifold"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25dz"

ARTIFACT = "causal_inner_cold_branch_anchor_preflight_wp10c9d6c7c3b5c4f25dy"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = "scripts/run_causal_inner_cold_branch_anchor_preflight_wp10c9d6c7c3b5c4f25dy.py"
THIS_TEST = "tests/test_causal_inner_cold_branch_anchor_preflight_wp10c9d6c7c3b5c4f25dy.py"
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_COLD_BRANCH_ANCHOR_PREFLIGHT_"
    "WP10C9D6C7C3B5C4F25DY_2026-08-21.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _validate_lock(*, require_clean: bool) -> dict:
    helper = manifest.decision.manifest.tube.manifest.geometry
    hashes = helper._validate_checksums(manifest.CANONICAL_DIRECTORY)
    contract = helper._read(manifest.CANONICAL_DIRECTORY / "cold_anchor_contract.json")
    summary = helper._read(manifest.CANONICAL_DIRECTORY / "summary.json")
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["work_package"] != manifest.WORK_PACKAGE
    ):
        raise RuntimeError("cold-anchor manifest classification changed")
    for relative, expected in contract["frozen_source_hashes"].items():
        if helper._sha(ROOT / relative) != expected:
            raise RuntimeError(f"frozen cold-anchor source changed: {relative}")
    manifest._validate_parent(require_clean=False)
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("cold-anchor execution requires a clean tracked tree")
    return {"manifest_hashes": hashes, "contract": contract}


def _candidate_states() -> dict[float, np.ndarray]:
    helper = manifest.decision.manifest.tube.manifest.geometry
    arrays = helper._load_npz(manifest.CANDIDATE_DIRECTORY / "candidate_geometry_arrays.npz")
    times = np.asarray(arrays["candidate_times_seconds"], dtype=float)
    states = np.asarray(arrays["candidate_primitive_states"], dtype=float)
    result = {}
    for target in manifest.CANDIDATE_TIMES_SECONDS:
        matches = np.flatnonzero(np.isclose(times, target, atol=1.0e-14, rtol=0.0))
        if len(matches) != 1:
            raise RuntimeError(f"unique saved cold candidate unavailable: {target}")
        result[target] = np.asarray(states[int(matches[0])], dtype=float)
    return result


def _geometry() -> dict[str, np.ndarray]:
    helper = manifest.decision.manifest.tube.manifest.geometry
    tangent = helper._load_npz(manifest.TANGENT_ARRAYS)
    tube_geometry = helper._load_npz(helper.CANONICAL_DIRECTORY / "geometry_arrays.npz")
    return {
        "R": np.asarray(tangent["macro_restriction_R82"], dtype=float),
        "L": np.asarray(tube_geometry["macro_lift_L470x82"], dtype=float),
        "Z": np.asarray(tangent["hidden_basis_Z388"], dtype=float),
        "Q": np.asarray(tangent["hidden_dual_Q388"], dtype=float),
    }


def _decompose_rate(
    coordinate_rate: np.ndarray, geometry: dict[str, np.ndarray]
) -> dict[str, np.ndarray | float]:
    tiny = np.finfo(float).tiny
    rate = np.asarray(coordinate_rate, dtype=float)
    macro_rate = geometry["R"] @ rate
    hidden_rate = geometry["Q"] @ rate
    macro_action = geometry["L"] @ macro_rate
    hidden_action = geometry["Z"] @ hidden_rate
    scale = max(float(np.linalg.norm(rate)), tiny)
    return {
        "coordinate_rate": rate,
        "macro_rate": macro_rate,
        "hidden_rate": hidden_rate,
        "macro_action": macro_action,
        "hidden_action": hidden_action,
        "hidden_fraction": float(np.linalg.norm(hidden_action) / scale),
        "decomposition_relative_defect": float(
            np.linalg.norm(macro_action + hidden_action - rate) / scale
        ),
    }


def _evaluate_candidate(
    time_seconds: float,
    state: np.ndarray,
    geometry: dict[str, np.ndarray],
    model,
    layout,
    configuration: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    started = time.perf_counter()
    context = configuration["context"]
    columns = np.asarray(configuration["columns"], dtype=float).reshape(state.shape)
    rows = np.asarray(configuration["rows"], dtype=float).reshape(state.shape)
    reaction = rate_source.causal_five_field_fixed_q_reaction(
        context,
        state,
        primitive_column_scales=columns,
        conservation_row_scales=rows,
        parent_cell_indices=layout.parent_cell_indices,
        refinement_ratio=layout.refinement_ratio,
        maximum_schur_condition_number=manifest.SCHUR_CONDITION_GATE,
    )
    tangent = rate_source.causal_five_field_monolithic_frozen_tangent(
        context,
        state,
        primitive_column_scales=columns,
        conservation_row_scales=rows,
    )
    multiplier = -reaction.q3_scaled_derivative @ tangent.scaled_base_rate_per_s
    reaction_action = reaction.reaction_lift @ multiplier
    scaled_rate = tangent.scaled_base_rate_per_s + reaction_action
    coordinate_jacobian, coordinate_metrics = exact_chart._coordinate_jacobian(model, state)
    coordinate_rate = coordinate_jacobian @ scaled_rate
    decomposition = _decompose_rate(coordinate_rate, geometry)

    free_constraint = reaction.q3_scaled_derivative @ tangent.scaled_base_rate_per_s
    action_constraint = reaction.q3_scaled_derivative @ reaction_action
    total_constraint = reaction.q3_scaled_derivative @ scaled_rate
    constraint_scale = max(
        float(np.linalg.norm(free_constraint)),
        float(np.linalg.norm(action_constraint)),
        np.finfo(float).tiny,
    )
    physical = rate_source._state_audit(context, state)
    metrics = {
        "time_seconds": time_seconds,
        "hidden_coordinate_rate_fraction": decomposition["hidden_fraction"],
        "coordinate_rate_decomposition_relative_defect": decomposition[
            "decomposition_relative_defect"
        ],
        "coordinate_jacobian_rank": int(coordinate_metrics["rank"]),
        "coordinate_jacobian_condition_number": float(
            coordinate_metrics["condition_number"]
        ),
        "coordinate_reconstruction_relative_defect": float(
            coordinate_metrics["coordinate_reconstruction_relative_defect"]
        ),
        "fixed_Q_rate_tangency_relative_defect": float(
            np.linalg.norm(total_constraint) / constraint_scale
        ),
        "maximum_reaction_ledger_relative_defect": float(
            reaction.maximum_reaction_ledger_relative_defect
        ),
        "raw_Schur_rank": int(reaction.raw_schur_numerical_rank),
        "raw_Schur_condition_number": float(reaction.raw_schur_condition_number),
        "minimum_reconstruction_factor": min(
            float(reaction.minimum_q3_reconstruction_factor),
            float(physical["minimum_reconstruction_factor"]),
        ),
        "maximum_height_ratio": float(physical["maximum_h_over_r"]),
        "minimum_scattering_optical_depth": float(
            physical["minimum_scattering_optical_depth"]
        ),
        "wall_seconds": float(time.perf_counter() - started),
    }
    gates = {
        "hidden_fraction": metrics["hidden_coordinate_rate_fraction"]
        <= manifest.HIDDEN_FRACTION_GATE,
        "coordinate_decomposition": metrics[
            "coordinate_rate_decomposition_relative_defect"
        ]
        <= manifest.DECOMPOSITION_GATE,
        "coordinate_rank": metrics["coordinate_jacobian_rank"] == 470,
        "coordinate_condition": metrics["coordinate_jacobian_condition_number"]
        <= manifest.COORDINATE_JACOBIAN_CONDITION_GATE,
        "fixed_Q_tangency": metrics["fixed_Q_rate_tangency_relative_defect"]
        <= manifest.FIXED_Q_TANGENCY_GATE,
        "reaction_ledger": metrics["maximum_reaction_ledger_relative_defect"]
        <= manifest.REACTION_LEDGER_GATE,
        "Schur_rank": metrics["raw_Schur_rank"] == 3,
        "Schur_condition": metrics["raw_Schur_condition_number"]
        <= manifest.SCHUR_CONDITION_GATE,
        "reconstruction": metrics["minimum_reconstruction_factor"]
        >= manifest.RECONSTRUCTION_GATE,
        "height": metrics["maximum_height_ratio"] <= manifest.HEIGHT_RATIO_GATE,
        "optical_depth": metrics["minimum_scattering_optical_depth"]
        >= manifest.OPTICAL_DEPTH_GATE,
    }
    metrics["gates"] = gates
    metrics["failed_gates"] = [name for name, passed in gates.items() if not passed]
    metrics["complete_pass"] = all(gates.values())
    arrays = {
        "primitive_state": state,
        "primitive_column_scales": columns,
        "conservation_row_scales": rows,
        "coordinate_jacobian470x560": coordinate_jacobian,
        "scaled_fixed_Q_rate560_per_s": scaled_rate,
        "scaled_free_rate560_per_s": tangent.scaled_base_rate_per_s,
        "scaled_reaction_action560_per_s": reaction_action,
        "continuous_multiplier3": multiplier,
        "coordinate_rate470_per_s": decomposition["coordinate_rate"],
        "macro_rate82_per_s": decomposition["macro_rate"],
        "hidden_rate388_per_s": decomposition["hidden_rate"],
        "macro_action470_per_s": decomposition["macro_action"],
        "hidden_action470_per_s": decomposition["hidden_action"],
        "q3_scaled_total_tangency": total_constraint,
    }
    return metrics, arrays


def _select_first_pass(candidate_metrics: list[dict]) -> int | None:
    for index, metrics in enumerate(candidate_metrics):
        if metrics["complete_pass"]:
            return index
    return None


def _update_catalog(summary: dict) -> None:
    helper = manifest.decision.manifest.tube.manifest.geometry
    with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    status = "SUPPORTED" if summary["passed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": helper._sha(path),
                    "scientific_status": status,
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case", "path", "bytes", "sha256", "scientific_status"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = helper._read(CANONICAL_SUMMARY)
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
            "latest_source_parent_commit": manifest.PARENT_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    helper._write_json(CANONICAL_SUMMARY, catalog)


def _execute() -> dict:
    helper = manifest.decision.manifest.tube.manifest.geometry
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("cold-anchor preflight result already exists")
    lock = _validate_lock(require_clean=True)
    states = _candidate_states()
    geometry = _geometry()
    model, _candidate, _fiber = exact_chart._model_inputs()
    layout, configuration, _trajectory, *_unused = rate_source.c4f24._endpoint_data()

    candidate_metrics = []
    candidate_arrays = {}
    for time_seconds in manifest.CANDIDATE_TIMES_SECONDS:
        metrics, arrays = _evaluate_candidate(
            time_seconds,
            states[time_seconds],
            geometry,
            model,
            layout,
            configuration,
        )
        candidate_metrics.append(metrics)
        prefix = f"candidate_{int(round(time_seconds * 1000)):02d}ms"
        candidate_arrays.update({f"{prefix}__{name}": value for name, value in arrays.items()})
        if metrics["complete_pass"]:
            break
    selected_index = _select_first_pass(candidate_metrics)
    selected_time = (
        None
        if selected_index is None
        else float(candidate_metrics[selected_index]["time_seconds"])
    )
    passed = selected_index is not None
    metrics = {
        "classification": PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,
        "passed": passed,
        "candidate_order_seconds": manifest.CANDIDATE_TIMES_SECONDS,
        "evaluated_candidate_count": len(candidate_metrics),
        "candidate_metrics": candidate_metrics,
        "selected_candidate_time_seconds": selected_time,
        "exact_fixed_Q_rate_evaluations": len(candidate_metrics),
        "complete_generator_assemblies": 0,
        "hidden_branch_roots": 0,
        "propagated_states": 0,
        "new_transition_microsteps": 0,
        "sealed_16ms_truth_calls": 0,
    }
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "cold_anchor_metrics.json", metrics)
    np.savez_compressed(CANONICAL_DIRECTORY / "cold_anchor_arrays.npz", **candidate_arrays)
    helper._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "manifest_directory": str(manifest.CANONICAL_DIRECTORY.relative_to(ROOT)),
            "manifest_hashes": lock["manifest_hashes"],
            "input_hashes": lock["contract"]["input_hashes"],
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": passed,
        "evaluated_candidate_count": len(candidate_metrics),
        "selected_candidate_time_seconds": selected_time,
        "cold_hidden_root_manifest_authorized": passed,
        "branch_root_executed": False,
        "hot_branch_truth_established": False,
        "complete_impulse_fit_authorized": False,
        "reduced_cycle_authorized": False,
        "authorized_next": AUTHORIZED_NEXT if passed else None,
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    helper._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "implementation_commit": helper._git("rev-parse", "HEAD"),
            "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"),
            "source_hashes": lock["contract"]["frozen_source_hashes"],
        },
    )
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{helper._sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fraction_lines = [
        f"- {item['time_seconds'] * 1000:g} ms: hidden fraction `{item['hidden_coordinate_rate_fraction']:.9e}`, pass `{item['complete_pass']}`, failed `{item['failed_gates']}`"
        for item in candidate_metrics
    ]
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Cold-branch anchor preflight WP10c9d6c7c3b5c4f25dy",
                "",
                "## Classification",
                "",
                f"`{summary['classification']}`",
                "",
                *fraction_lines,
                "",
                f"Selected candidate time: `{selected_time}` s. Exact fixed-Q rate calls: `{len(candidate_metrics)}`. No generator or root was executed.",
                "",
                "A pass authorizes only definitions for one cold hidden-root attempt. Hot branch truth, a complete impulse, and reduced-cycle execution remain blocked.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("use --run")
    payload = _execute()
    print(
        json.dumps(
            manifest.decision.manifest.tube.manifest.geometry._plain(payload),
            indent=2,
            sort_keys=True,
        )
    )
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
