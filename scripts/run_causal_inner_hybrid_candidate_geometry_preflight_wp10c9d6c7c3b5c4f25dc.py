#!/usr/bin/env python3
"""Inventory accepted states for the hybrid branch/event atlas without new truth."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_forward_quadratic_field_revision_manifest_wp10c9d6c7c3b5c4f25cx as field_manifest  # noqa: E402
import run_causal_inner_hybrid_branch_transition_atlas_manifest_wp10c9d6c7c3b5c4f25db as parent  # noqa: E402
import run_causal_inner_local_slaving_transition_diagnosis_wp10c9d6c7c3b5c4f25da as diagnosis  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25dc"
PARENT_COMMIT = "c94f6cd17fa5273cd0c98ce3d332fcd84480b5b0"
PARENT_PARENT = "d09b51dc6ad3e967453cdf48f233cd0199744dbd"
PARENT_TREE = "bc020b45a1956c5ccad5cfb24397c739b105a6e3"

CLASSIFICATION = (
    "hybrid_candidate_geometry_passed_unclassified_20ms_primary_"
    "16ms_sealed_branch_pilot_manifest_authorized"
)
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25dd"

CANDIDATE_TIMES_SECONDS = (0.002, 0.005, 0.008, 0.012, 0.016, 0.020)
PRIMARY_TIME_SECONDS = 0.020
SEALED_TIME_SECONDS = 0.016
DECODER_RELATIVE_ERROR_GATE = 5.0e-2
HEIGHT_RATIO_GATE = 0.5
OPTICAL_DEPTH_GATE = 1.0
RECONSTRUCTION_GATE = 1.0 - 1.0e-12
PATH_RELATIVE_RANK_TOLERANCE = 1.0e-3

ARTIFACT = (
    "causal_inner_hybrid_candidate_geometry_preflight_"
    "wp10c9d6c7c3b5c4f25dc"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_hybrid_candidate_geometry_preflight_"
    "wp10c9d6c7c3b5c4f25dc.py"
)
THIS_TEST = (
    "tests/test_causal_inner_hybrid_candidate_geometry_preflight_"
    "wp10c9d6c7c3b5c4f25dc.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_HYBRID_CANDIDATE_GEOMETRY_"
    "PREFLIGHT_WP10C9D6C7C3B5C4F25DC_2026-08-20.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

MIDDLE_5_DIRECTORY = ROOT / (
    "results/canonical/causal_inner_nonlinear_middle_5ms_completion_"
    "wp10c9d6c7c3b5c3h2d1"
)
MIDDLE_20_DIRECTORY = ROOT / (
    "results/canonical/causal_inner_nonlinear_optimized_middle_20ms_completion_"
    "wp10c9d6c7c3b5c4e3"
)
DIAGNOSIS_ARRAYS = diagnosis.CANONICAL_DIRECTORY / "diagnostic_arrays.npz"
FIELD_ARRAYS = field_manifest.CANONICAL_DIRECTORY / "forward_quadratic_local_field.npz"


def _plain(value):
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _plain(value.item())
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


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


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


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _validate_parent(*, require_clean: bool) -> dict:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("candidate geometry parent commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("candidate geometry parent lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("candidate geometry parent tree changed")
    hashes = _checksums(parent.CANONICAL_DIRECTORY)
    summary = _read(parent.CANONICAL_DIRECTORY / "summary.json")
    campaign = _read(parent.CANONICAL_DIRECTORY / "campaign_contract.json")
    provenance = _read(parent.CANONICAL_DIRECTORY / "provenance.json")
    if (
        not summary["passed"]
        or summary["classification"] != parent.CLASSIFICATION
        or summary["authorized_next"] != WORK_PACKAGE
        or campaign["next_package"]["work_package"] != WORK_PACKAGE
        or campaign["next_package"]["new_exact_rate_calls"] != 0
        or campaign["next_package"]["new_nonlinear_roots"] != 0
    ):
        raise RuntimeError("candidate geometry authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"hybrid manifest source changed: {relative}")
    for name, expected in provenance["thread_environment"].items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    diagnosis_hashes = _checksums(diagnosis.CANONICAL_DIRECTORY)
    field_hashes = _checksums(field_manifest.CANONICAL_DIRECTORY)
    trajectory_hashes = {
        "middle_5ms": _checksums(MIDDLE_5_DIRECTORY),
        "middle_20ms": _checksums(MIDDLE_20_DIRECTORY),
    }
    for directory in (MIDDLE_5_DIRECTORY, MIDDLE_20_DIRECTORY):
        source_summary = _read(directory / "summary.json")
        if not source_summary["passed"] or not source_summary["base"]["passed"]:
            raise RuntimeError(f"trajectory source is not accepted: {directory}")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("candidate geometry preflight requires a clean tracked tree")
    return {
        "hashes": hashes,
        "diagnosis_hashes": diagnosis_hashes,
        "field_hashes": field_hashes,
        "trajectory_hashes": trajectory_hashes,
    }


def _state_at(source: dict[str, np.ndarray], time_seconds: float) -> tuple[np.ndarray, bool]:
    output_indices = np.flatnonzero(
        np.isclose(source["base__output_times"], time_seconds, atol=1.0e-14)
    )
    accepted_indices = np.flatnonzero(
        np.isclose(source["base__accepted_times"], time_seconds, atol=1.0e-14)
    )
    if len(output_indices) != 1 or len(accepted_indices) != 1:
        raise RuntimeError(f"candidate time is not unique: {time_seconds}")
    output = np.asarray(source["base__output_states"][output_indices[0]], dtype=float)
    accepted = np.asarray(
        source["base__accepted_states"][accepted_indices[0]], dtype=float
    )
    return output, bool(np.array_equal(output, accepted))


def _candidate_states() -> tuple[dict[str, np.ndarray], dict]:
    middle_5 = _load_npz(MIDDLE_5_DIRECTORY / "decisive_arrays.npz")
    middle_20 = _load_npz(MIDDLE_20_DIRECTORY / "decisive_arrays.npz")
    states = []
    exact_accepted = []
    source_codes = []
    for time_seconds in CANDIDATE_TIMES_SECONDS:
        if time_seconds <= 0.005:
            source = middle_5
            source_code = 5
        else:
            source = middle_20
            source_code = 20
        state, exact = _state_at(source, time_seconds)
        states.append(state)
        exact_accepted.append(exact)
        source_codes.append(source_code)
    return {
        "candidate_times_seconds": np.asarray(CANDIDATE_TIMES_SECONDS),
        "candidate_primitive_states": np.asarray(states),
        "candidate_source_duration_ms": np.asarray(source_codes, dtype=np.int64),
        "candidate_output_equals_accepted_state_bitwise": np.asarray(
            exact_accepted, dtype=bool
        ),
    }, {
        "candidate_count": len(states),
        "all_output_states_equal_accepted_states_bitwise": bool(all(exact_accepted)),
    }


def _relative_scaled_decoder_error(
    model, state: np.ndarray, decoded: np.ndarray
) -> tuple[float, float]:
    truth = ((np.asarray(state) - model.base_state) / model.columns).ravel()
    prediction = ((np.asarray(decoded) - model.base_state) / model.columns).ravel()
    difference = prediction - truth
    return float(
        np.linalg.norm(difference)
        / max(np.linalg.norm(truth), np.finfo(float).tiny)
    ), float(np.max(np.abs(difference)))


def _geometry(
    state_arrays: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict]:
    field_closure = _load_npz(FIELD_ARRAYS)
    field = field_manifest.ForwardQuadraticAuthenticCenterField(field_closure)
    model = field.model
    diagnostic = _load_npz(DIAGNOSIS_ARRAYS)
    macro_restriction = np.asarray(diagnostic["macro_restriction"], dtype=float)

    absolute_coordinates = []
    local_coordinates = []
    macro_coordinates = []
    decoded_states = []
    active_coordinates = []
    field_weights = []
    decoder_errors = []
    decoder_maximums = []
    reconstruction_factors = []
    height_ratios = []
    optical_depths = []
    physical_passes = []
    for state in state_arrays["candidate_primitive_states"]:
        absolute, coordinate_factors = model.coordinate(state)
        local = absolute - field.center_coordinate
        decoded = field.decoded_state(local)
        decoder_error, decoder_maximum = _relative_scaled_decoder_error(
            model, state, decoded
        )
        physical = field_manifest.vector_field.manifest.parent.geometry.chart_tools._state_audit(
            model.components["context"], state
        )
        reconstruction = min(
            float(np.min(coordinate_factors)),
            float(physical["minimum_reconstruction_factor"]),
        )
        height = float(physical["maximum_h_over_r"])
        optical = float(physical["minimum_scattering_optical_depth"])
        physical_pass = bool(
            reconstruction >= RECONSTRUCTION_GATE
            and height <= HEIGHT_RATIO_GATE
            and optical >= OPTICAL_DEPTH_GATE
        )
        absolute_coordinates.append(absolute)
        local_coordinates.append(local)
        macro_coordinates.append(macro_restriction @ absolute)
        decoded_states.append(decoded)
        active_coordinates.append(field._active(local))
        field_weights.append(field.weight(local))
        decoder_errors.append(decoder_error)
        decoder_maximums.append(decoder_maximum)
        reconstruction_factors.append(reconstruction)
        height_ratios.append(height)
        optical_depths.append(optical)
        physical_passes.append(physical_pass)

    macro = np.asarray(macro_coordinates)
    centered = macro - np.mean(macro, axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    relative_threshold = PATH_RELATIVE_RANK_TOLERANCE * singular[0]
    effective_rank = int(np.count_nonzero(singular > relative_threshold))
    decoder_errors_array = np.asarray(decoder_errors)
    physical_passes_array = np.asarray(physical_passes)
    eligible = np.logical_and(
        decoder_errors_array <= DECODER_RELATIVE_ERROR_GATE,
        physical_passes_array,
    )
    eligible_times = state_arrays["candidate_times_seconds"][eligible]
    expected_eligible = np.asarray((SEALED_TIME_SECONDS, PRIMARY_TIME_SECONDS))
    selection_matches = bool(np.array_equal(eligible_times, expected_eligible))
    primary_index = int(
        np.flatnonzero(
            np.isclose(
                state_arrays["candidate_times_seconds"],
                PRIMARY_TIME_SECONDS,
                atol=1.0e-14,
            )
        )[0]
    )
    sealed_index = int(
        np.flatnonzero(
            np.isclose(
                state_arrays["candidate_times_seconds"],
                SEALED_TIME_SECONDS,
                atol=1.0e-14,
            )
        )[0]
    )
    selected_separation = float(np.linalg.norm(macro[primary_index] - macro[sealed_index]))
    metrics = {
        "candidate_count": len(macro),
        "candidate_times_seconds": CANDIDATE_TIMES_SECONDS,
        "decoder_relative_errors": decoder_errors_array,
        "maximum_scaled_decoder_mismatches": np.asarray(decoder_maximums),
        "minimum_reconstruction_factors": np.asarray(reconstruction_factors),
        "maximum_height_ratios": np.asarray(height_ratios),
        "minimum_scattering_optical_depths": np.asarray(optical_depths),
        "physical_guard_passes": physical_passes_array,
        "forward_patch_weights": np.asarray(field_weights),
        "eligible_existing_atlas_geometry_times_seconds": eligible_times,
        "eligible_selection_matches_frozen_16ms_20ms_pair": selection_matches,
        "macro_path_singular_values": singular,
        "macro_path_effective_rank_at_relative_1e_3": effective_rank,
        "macro_path_arclength": float(
            np.sum(np.linalg.norm(np.diff(macro, axis=0), axis=1))
        ),
        "primary_to_sealed_macro_separation": selected_separation,
        "primary_time_seconds": PRIMARY_TIME_SECONDS,
        "sealed_time_seconds": SEALED_TIME_SECONDS,
        "all_candidates_unclassified": True,
        "forward_patch_labels_any_candidate": False,
    }
    arrays = {
        "candidate_absolute_y470_coordinates": np.asarray(absolute_coordinates),
        "candidate_local_forward_patch_coordinates": np.asarray(local_coordinates),
        "candidate_macro_U80_a2_coordinates": macro,
        "candidate_active_coordinates": np.asarray(active_coordinates),
        "candidate_decoded_primitive_states": np.asarray(decoded_states),
        "candidate_decoder_relative_errors": decoder_errors_array,
        "candidate_forward_patch_weights": np.asarray(field_weights),
        "candidate_physical_guard_passes": physical_passes_array,
        "eligible_candidate_mask": eligible,
        "primary_candidate_index": np.asarray(primary_index),
        "sealed_candidate_index": np.asarray(sealed_index),
    }
    return arrays, metrics


def _checks(source: dict, geometry: dict) -> dict[str, bool]:
    return {
        "candidate_count": source["candidate_count"] == len(CANDIDATE_TIMES_SECONDS),
        "accepted_state_identity": source[
            "all_output_states_equal_accepted_states_bitwise"
        ],
        "all_physical_guards": bool(np.all(geometry["physical_guard_passes"])),
        "reconstruction": bool(
            np.min(geometry["minimum_reconstruction_factors"])
            >= RECONSTRUCTION_GATE
        ),
        "height": bool(np.max(geometry["maximum_height_ratios"]) <= HEIGHT_RATIO_GATE),
        "optical_depth": bool(
            np.min(geometry["minimum_scattering_optical_depths"])
            >= OPTICAL_DEPTH_GATE
        ),
        "forward_patch_not_mislabeled": bool(
            np.max(np.abs(geometry["forward_patch_weights"])) <= 1.0e-15
        ),
        "eligible_pair": geometry[
            "eligible_selection_matches_frozen_16ms_20ms_pair"
        ],
        "path_has_more_than_one_effective_direction": geometry[
            "macro_path_effective_rank_at_relative_1e_3"
        ]
        >= 2,
        "selected_pair_separated": geometry["primary_to_sealed_macro_separation"]
        >= 1.0e-2,
        "all_unclassified": geometry["all_candidates_unclassified"],
        "no_truth": True,
        "no_roots": True,
        "no_propagation": True,
    }


def _pilot_contract(metrics: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "candidate_inventory": {
            "times_seconds": CANDIDATE_TIMES_SECONDS,
            "all_are_accepted_full_model_states": True,
            "all_are_physically_admissible": True,
            "existing_atlas_geometry_eligible_times_seconds": metrics[
                "eligible_existing_atlas_geometry_times_seconds"
            ],
            "earlier_states": (
                "conservative_path_geometry_only_outside_current_decoder_"
                "trust_region_requires_new_branch_local_patches"
            ),
            "branch_labels_assigned": False,
        },
        "selected_candidates": {
            "primary": {
                "id": "U20_unclassified_primary",
                "time_seconds": PRIMARY_TIME_SECONDS,
                "role": "future_training_geometry_seed",
            },
            "sealed": {
                "id": "U16_unclassified_sealed",
                "time_seconds": SEALED_TIME_SECONDS,
                "role": "independent_geometry_holdout_not_solver_tuning",
            },
            "a2_coordinates_are_patch_local_not_global_branch_labels": True,
        },
        "next_definitions_only_manifest": {
            "work_package": AUTHORIZED_NEXT,
            "purpose": "freeze_one_primary_20ms_hidden_fast_branch_existence_pilot",
            "mathematical_root": {
                "resolved_constraint": "R82_y_equals_X82_primary",
                "hidden_residual": "Z_transpose_F_y_equals_zero",
                "state_form": "y_equals_L82_X82_plus_Z_z",
                "solver": "surrogate_seeded_hidden_Newton_with_exact_truth_validation_separate",
                "no_artificial_82_channel_physical_reaction": True,
            },
            "must_compare": [
                "surrogate_candidate_to_exact_complete_residual",
                "hidden_fast_stability_and_gap",
                "slow_graph_invariance",
                "physical_and_decoder_guards",
            ],
            "execution_budget_not_yet_authorized": True,
            "branch_label_remains_unclassified_until_stable_root_and_fold_geometry_pass": True,
        },
        "authorization_boundaries": {
            "new_truth_authorized": False,
            "branch_root_execution_authorized": False,
            "transition_execution_authorized": False,
            "online_solver_authorized": False,
            "cycle_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
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
                    "sha256": _sha(path),
                    "scientific_status": "GEOMETRY_PREFLIGHT",
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
            "latest_source_parent_commit": PARENT_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def _run() -> dict:
    frozen = _validate_parent(require_clean=True)
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("hybrid candidate geometry preflight already exists")
    state_arrays, source_metrics = _candidate_states()
    geometry_arrays, geometry_metrics = _geometry(state_arrays)
    checks = _checks(source_metrics, geometry_metrics)
    if not all(checks.values()):
        raise RuntimeError(f"candidate geometry failed: {checks}")
    contract = _pilot_contract(geometry_metrics)
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_npz(
        CANONICAL_DIRECTORY / "candidate_geometry_arrays.npz",
        {**state_arrays, **geometry_arrays},
    )
    _write_json(
        CANONICAL_DIRECTORY / "candidate_geometry_metrics.json",
        {
            "checks": checks,
            "passed": True,
            "source": source_metrics,
            "geometry": geometry_metrics,
            "new_exact_rate_calls": 0,
            "new_complete_generator_assemblies": 0,
            "new_nonlinear_fixed_Q_roots": 0,
            "propagated_states": 0,
        },
    )
    _write_json(CANONICAL_DIRECTORY / "branch_pilot_contract.json", contract)
    _write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_commit": PARENT_COMMIT,
            "parent_parent": PARENT_PARENT,
            "parent_tree": PARENT_TREE,
            "parent_hashes": frozen["hashes"],
            "diagnosis_hashes": frozen["diagnosis_hashes"],
            "field_hashes": frozen["field_hashes"],
            "trajectory_hashes": frozen["trajectory_hashes"],
            "diagnosis_arrays_sha256": _sha(DIAGNOSIS_ARRAYS),
            "field_arrays_sha256": _sha(FIELD_ARRAYS),
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "existing_candidate_state_count": len(CANDIDATE_TIMES_SECONDS),
        "all_candidates_physically_admissible": True,
        "all_candidates_unclassified": True,
        "primary_candidate": "U20_unclassified_primary",
        "sealed_candidate": "U16_unclassified_sealed",
        "earlier_candidate_states_inside_current_decoder_trust": False,
        "new_exact_rate_calls": 0,
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_fixed_Q_roots": 0,
        "propagated_states": 0,
        "branch_root_execution_authorized": False,
        "online_reduced_solver_implementation_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    source_files = (
        THIS_RUNNER,
        THIS_TEST,
        parent.THIS_RUNNER,
        parent.THIS_TEST,
        diagnosis.THIS_RUNNER,
        diagnosis.THIS_TEST,
        field_manifest.THIS_RUNNER,
        field_manifest.THIS_TEST,
    )
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "GEOMETRY_PREFLIGHT",
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "source_hashes": {
                relative: _sha(ROOT / relative) for relative in source_files
            },
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "thread_environment": {
                name: os.environ.get(name)
                for name in field_manifest.training._thread_environment()
            },
        },
    )
    names = tuple(sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()))
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Hybrid candidate geometry preflight WP10c9d6c7c3b5c4f25dc",
                "",
                "## Classification",
                "",
                f"`{CLASSIFICATION}`",
                "",
                "Six existing accepted middle-layout states at 2, 5, 8, 12, 16, and 20 ms were mapped into the exact conservative U80+a2 coordinates. Every state passes reconstruction, height, and optical-depth guards, and every selected output state is bitwise identical to an accepted trajectory state.",
                "",
                f"The current decoder errors decrease from `{geometry_metrics['decoder_relative_errors'][0]:.6e}` at 2 ms to `{geometry_metrics['decoder_relative_errors'][-1]:.6e}` at 20 ms. Only the 16 and 20 ms states pass the frozen `{DECODER_RELATIVE_ERROR_GATE:.2%}` geometry gate. Earlier states remain useful conservative path points but require new branch-local atlas patches.",
                "",
                f"The six-state macro path has effective rank `{geometry_metrics['macro_path_effective_rank_at_relative_1e_3']}` at relative tolerance `{PATH_RELATIVE_RANK_TOLERANCE:.1e}`. The 20 ms primary and sealed 16 ms candidate are separated by `{geometry_metrics['primary_to_sealed_macro_separation']:.6e}` in normalized U80+a2 coordinates.",
                "",
                "All candidates remain unclassified. The authentic forward patch has zero partition weight on them and is not used to assign cold/hot labels. The next package may only define a hidden-fast branch-root pilot at the 20 ms resolved state; it may not execute that root yet.",
                "",
                f"Authorized next artifact: `{AUTHORIZED_NEXT}`. No new truth, branch root, transition, online solver, or cycle is authorized.",
                "",
            )
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("pass --run")
    print(json.dumps(_plain(_run()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
