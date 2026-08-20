#!/usr/bin/env python3
"""Freeze a geometry-only decoder repair with exact field compensation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_shell_gated_atlas_rate_validation_wp10c9d6c7c3b5c4f25ck as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25cl"
PARENT_COMMIT = "2e8e0368b53d338e8d093ddd3d79ae001caec2ad"
PARENT_PARENT = "ff6d7fd00e54c3037dd1186b4b7cb8ca231b5151"
PARENT_TREE = "87cbd0012e47f01178ba2501d6ff1c04812c8461"
CLASSIFICATION = "compensated_geometry_decoder_repair_manifest_frozen"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25cm"

RBF_LENGTH_SCALE = 1.5e-2
RBF_REGULARIZATION = 1.0e-8
HOLDOUT_MIXING = 0.25
HOLDOUT_COMPONENT_BOUNDS = (1.25e-2, 1.5e-2)
HOLDOUT_DIRECTION_COUNT = 4
PLANNED_GEOMETRY_CANDIDATES = 8

ARTIFACT = (
    "causal_inner_compensated_decoder_repair_manifest_"
    "wp10c9d6c7c3b5c4f25cl"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_compensated_decoder_repair_manifest_"
    "wp10c9d6c7c3b5c4f25cl.py"
)
THIS_TEST = (
    "tests/test_causal_inner_compensated_decoder_repair_manifest_"
    "wp10c9d6c7c3b5c4f25cl.py"
)
NEXT_RUNNER = (
    "scripts/run_causal_inner_compensated_decoder_geometry_preflight_"
    "wp10c9d6c7c3b5c4f25cm.py"
)
NEXT_TEST = (
    "tests/test_causal_inner_compensated_decoder_geometry_preflight_"
    "wp10c9d6c7c3b5c4f25cm.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_COMPENSATED_DECODER_"
    "REPAIR_MANIFEST_WP10C9D6C7C3B5C4F25CL_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

OLD_EXTENSION = parent.atlas.CANONICAL_DIRECTORY / "local_atlas_extension.npz"
FAILED_RATE_ARRAYS = parent.CANONICAL_DIRECTORY / "rate_arrays.npz"
DIRECTION_DESIGN = parent.atlas.DIRECTION_DESIGN
INNER_CLOSURE = parent.atlas.OLD_CLOSURE

_plain = parent._plain
_read = parent._read
_write_json = parent._write_json
_sha = parent._sha
_checksums = parent._checksums
_load_npz = parent._load_npz


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _validate_parent(*, require_clean: bool) -> dict:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("independent atlas rejection commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("independent atlas rejection lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("independent atlas rejection tree changed")
    hashes = _checksums(parent.CANONICAL_DIRECTORY)
    summary = _read(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = _read(parent.CANONICAL_DIRECTORY / "rate_metrics.json")
    provenance = _read(parent.CANONICAL_DIRECTORY / "provenance.json")
    if (
        summary["passed"]
        or not summary["truth_passed"]
        or summary["independent_model_passed"]
        or summary["classification"] != parent.FAIL_CLASSIFICATION
        or summary["authorized_next"] != parent.FAIL_AUTHORIZED_NEXT
        or summary["completed_exact_rate_evaluations"] != 8
        or summary["failed_rate_evaluations"] != 0
        or summary["coefficients_refit_after_holdout_truth"]
        or not all(metrics["truth_checks"].values())
        or metrics["model_checks"]["decoder_full_state_error"]
        or not all(
            passed
            for name, passed in metrics["model_checks"].items()
            if name != "decoder_full_state_error"
        )
        or summary["maximum_decoder_full_state_relative_error"] <= 5.0e-3
        or summary["maximum_full_state_rate_relative_error"] > 0.15
        or summary["maximum_a28_rate_relative_error"] > 0.15
        or summary["radial_sign_disagreement_count"] != 0
    ):
        raise RuntimeError("decoder-only repair authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"independent validation source changed: {relative}")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("compensated decoder manifest requires a clean tracked tree")
    return {"summary": summary, "metrics": metrics, "hashes": hashes}


def _gaussian_kernel(
    evaluation: np.ndarray, centers: np.ndarray, length_scale: float
) -> np.ndarray:
    left = np.asarray(evaluation, dtype=float)
    right = np.asarray(centers, dtype=float)
    squared = np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=2)
    return np.exp(-0.5 * squared / float(length_scale) ** 2)


def _repair_value(
    departure: np.ndarray,
    centers: np.ndarray,
    coefficients: np.ndarray,
    length_scale: float = RBF_LENGTH_SCALE,
) -> np.ndarray:
    kernel = _gaussian_kernel(
        np.asarray(departure, dtype=float).reshape(1, -1),
        centers,
        length_scale,
    )
    return np.asarray(kernel @ coefficients, dtype=float).reshape(-1)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= np.finfo(float).tiny):
        raise RuntimeError("decoder holdout direction vanished")
    return array / norms[:, None]


def _holdout_design() -> tuple[np.ndarray, tuple[str, ...], dict]:
    source = _load_npz(DIRECTION_DESIGN)
    directions = np.asarray(source["directions"], dtype=float)
    energy6 = directions[6]
    escape = directions[8]
    forward = directions[9]
    holdouts = _normalize_rows(
        np.vstack(
            (
                forward + HOLDOUT_MIXING * escape + HOLDOUT_MIXING * energy6,
                forward + HOLDOUT_MIXING * escape - HOLDOUT_MIXING * energy6,
                forward - HOLDOUT_MIXING * escape + HOLDOUT_MIXING * energy6,
                forward - HOLDOUT_MIXING * escape - HOLDOUT_MIXING * energy6,
            )
        )
    )
    labels = (
        "forward_plus_escape_plus_energy6",
        "forward_plus_escape_minus_energy6",
        "forward_minus_escape_plus_energy6",
        "forward_minus_escape_minus_energy6",
    )
    old = _load_npz(OLD_EXTENSION)
    revealed = _load_npz(FAILED_RATE_ARRAYS)
    training = np.vstack(
        (
            old["training_departure_coordinates"],
            revealed["candidate_departure_coordinates"],
        )
    )
    normalized_training = _normalize_rows(training)
    return holdouts, labels, {
        "maximum_holdout_to_training_absolute_cosine": float(
            np.max(np.abs(holdouts @ normalized_training.T))
        ),
        "minimum_holdout_pair_separation": float(
            np.min(
                np.linalg.norm(
                    holdouts[:, None, :]
                    - holdouts[None, :, :]
                    + np.eye(HOLDOUT_DIRECTION_COUNT)[:, :, None] * 1.0e6,
                    axis=2,
                )
            )
        ),
    }


def _fit_repair(
    *, include_physical_audits: bool = True
) -> tuple[dict[str, np.ndarray], dict]:
    model = parent.vector_field.ReducedVectorField()
    old = _load_npz(OLD_EXTENSION)
    revealed = _load_npz(FAILED_RATE_ARRAYS)
    revealed_metrics = _read(parent.CANONICAL_DIRECTORY / "rate_metrics.json")
    departures = np.vstack(
        (
            old["training_departure_coordinates"],
            revealed["candidate_departure_coordinates"],
        )
    )
    exact_deltas = np.vstack(
        (
            old["training_exact_scaled_deltas"],
            revealed["candidate_scaled_deltas"],
        )
    )
    old_extended_deltas = np.vstack(
        (
            old["training_extended_decoded_deltas"],
            revealed["extended_decoded_scaled_deltas"],
        )
    )
    old_rates = np.vstack(
        (
            old["training_extended_full_state_rates_per_second"],
            revealed["predicted_full_state_rates_per_second"],
        )
    )
    shell_weights = np.concatenate(
        (
            old["training_shell_weights"],
            np.asarray(
                [
                    item["shell_weight"]
                    for item in revealed_metrics["evaluations"]
                ],
                dtype=float,
            ),
        )
    )
    if (
        departures.shape != (16, 28)
        or exact_deltas.shape != (16, 560)
        or old_extended_deltas.shape != (16, 560)
        or old_rates.shape != (16, 560)
        or np.min(shell_weights) <= 0.0
    ):
        raise RuntimeError("decoder-repair training database changed")
    kernel = _gaussian_kernel(departures, departures, RBF_LENGTH_SCALE)
    regularized = kernel + RBF_REGULARIZATION * np.eye(departures.shape[0])
    targets = (exact_deltas - old_extended_deltas) / shell_weights[:, None]
    coefficients = np.linalg.solve(regularized, targets)
    corrections = shell_weights[:, None] * (kernel @ coefficients)
    repaired_deltas = old_extended_deltas + corrections
    generator_corrections = corrections @ model.generator.T
    uncompensated_rates = old_rates + generator_corrections
    compensated_rates = uncompensated_rates - generator_corrections

    decoder_errors = np.linalg.norm(
        repaired_deltas - exact_deltas, axis=1
    ) / np.maximum(np.linalg.norm(exact_deltas, axis=1), np.finfo(float).tiny)
    old_errors = np.linalg.norm(
        old_extended_deltas - exact_deltas, axis=1
    ) / np.maximum(np.linalg.norm(exact_deltas, axis=1), np.finfo(float).tiny)
    invariance_errors = np.linalg.norm(
        compensated_rates - old_rates, axis=1
    ) / np.maximum(np.linalg.norm(old_rates, axis=1), np.finfo(float).tiny)

    coordinate_mismatches = []
    state_audits = []
    if include_physical_audits:
        memories = exact_deltas @ model.memory_basis
        online_coordinates = np.concatenate(
            (
                np.zeros((16, 162), dtype=float),
                memories,
                departures,
            ),
            axis=1,
        )
        for target, delta in zip(online_coordinates, repaired_deltas):
            state = model.base_state + (
                model.columns.ravel() * delta
            ).reshape(model.base_state.shape)
            decoded, factors = model.coordinate(state)
            physical = parent.vector_field.manifest.parent.geometry.chart_tools._state_audit(
                model.components["context"], state
            )
            coordinate_mismatches.append(
                float(
                    np.linalg.norm(decoded - target)
                    / max(float(np.linalg.norm(target)), np.finfo(float).tiny)
                )
            )
            state_audits.append(
                {
                    "minimum_reconstruction_factor": min(
                        float(np.min(factors)),
                        physical["minimum_reconstruction_factor"],
                    ),
                    "maximum_H_over_R": physical["maximum_h_over_r"],
                    "minimum_scattering_optical_depth": physical[
                        "minimum_scattering_optical_depth"
                    ],
                }
            )

    inner = _load_npz(INNER_CLOSURE)
    inner_deltas = np.asarray(inner["predicted_scaled_deltas"], dtype=float)
    inner_deltas = inner_deltas[np.all(np.isfinite(inner_deltas), axis=1)]
    inner_weights = np.asarray(
        [
            parent.atlas._shell_weight(float(np.max(np.abs(delta))))
            for delta in inner_deltas
        ]
    )
    metrics = {
        "training_count": int(departures.shape[0]),
        "kernel_rank": int(np.linalg.matrix_rank(regularized)),
        "kernel_condition_number": float(np.linalg.cond(regularized)),
        "maximum_old_decoder_relative_error": float(np.max(old_errors)),
        "maximum_repaired_decoder_relative_error": float(
            np.max(decoder_errors)
        ),
        "median_repaired_decoder_relative_error": float(
            np.median(decoder_errors)
        ),
        "maximum_compensated_full_rate_invariance_defect": float(
            np.max(invariance_errors)
        ),
        "maximum_inner_certificate_shell_weight": float(np.max(inner_weights)),
        "maximum_repaired_decoder_coordinate_relative_mismatch": (
            float(np.max(coordinate_mismatches))
            if coordinate_mismatches
            else math.nan
        ),
        "minimum_repaired_state_reconstruction_factor": (
            float(
                min(item["minimum_reconstruction_factor"] for item in state_audits)
            )
            if state_audits
            else math.nan
        ),
        "maximum_repaired_state_H_over_R": (
            float(max(item["maximum_H_over_R"] for item in state_audits))
            if state_audits
            else math.nan
        ),
        "minimum_repaired_state_scattering_optical_depth": (
            float(
                min(
                    item["minimum_scattering_optical_depth"]
                    for item in state_audits
                )
            )
            if state_audits
            else math.nan
        ),
        "new_continuous_rate_evaluations": 0,
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
    }
    arrays = {
        "training_departure_coordinates": departures,
        "training_exact_scaled_deltas": exact_deltas,
        "training_old_extended_scaled_deltas": old_extended_deltas,
        "training_repaired_scaled_deltas": repaired_deltas,
        "training_shell_weights": shell_weights,
        "training_old_full_state_rates_per_second": old_rates,
        "training_compensated_full_state_rates_per_second": compensated_rates,
        "decoder_repair_centers": departures,
        "decoder_repair_coefficients": coefficients,
        "decoder_repair_kernel_matrix": kernel,
        "decoder_repair_kernel_singular_values": np.linalg.svd(
            regularized, compute_uv=False
        ),
        "inner_certificate_shell_weights": inner_weights,
    }
    return arrays, metrics


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "diagnosis": {
            "exact_truth_layer_passed": True,
            "independent_full_rate_submodel_passed": True,
            "independent_a28_rate_submodel_passed": True,
            "independent_radial_sign_submodel_passed": True,
            "only_failed_binding_gate": "decoder_full_state_error",
        },
        "decoder_repair": {
            "input": "a28_departure_coordinate",
            "output": "scaled_full_state_delta_560",
            "kernel": "isotropic_Gaussian",
            "length_scale": RBF_LENGTH_SCALE,
            "regularization": RBF_REGULARIZATION,
            "training_geometry_count": 16,
            "uses_new_rate_truth": False,
            "shell_gate": parent.atlas._contract()["shell_gate"],
        },
        "exact_field_compensation": {
            "new_decoder": "D_new_equals_D_old_plus_C_geometry",
            "new_closure": "N_new_equals_N_old_minus_G_times_C_geometry",
            "full_field_identity": "G_D_new_plus_N_new_equals_G_D_old_plus_N_old",
            "independently_validated_full_state_rate_is_unchanged": True,
        },
        "binding_revealed_geometry_gates": {
            "maximum_kernel_condition_number": 1.0e4,
            "maximum_repaired_decoder_relative_error": 1.0e-6,
            "maximum_compensated_full_rate_invariance_defect": 1.0e-12,
            "maximum_inner_certificate_shell_weight": 0.0,
            "maximum_repaired_decoder_coordinate_relative_mismatch": 5.0e-4,
            "minimum_repaired_state_reconstruction_factor": 1.0 - 1.0e-12,
            "maximum_repaired_state_H_over_R": 0.12,
            "minimum_repaired_state_scattering_optical_depth": 1.0,
        },
        "independent_holdout_geometry": {
            "component_bounds": list(HOLDOUT_COMPONENT_BOUNDS),
            "direction_count": HOLDOUT_DIRECTION_COUNT,
            "mixing_coefficient": HOLDOUT_MIXING,
            "planned_candidate_count": PLANNED_GEOMETRY_CANDIDATES,
            "directions_revealed_after_repair_frozen": True,
            "new_rate_calls_equal": 0,
        },
        "future_independent_geometry_gates": {
            "completed_candidate_count_equal": PLANNED_GEOMETRY_CANDIDATES,
            "failed_candidate_count_equal": 0,
            "maximum_coordinate_residual_infinity": 5.0e-10,
            "maximum_normalized_Q3_defect": 1.0e-10,
            "minimum_reconstruction_factor": 1.0 - 1.0e-12,
            "maximum_reconstruction_factor": 1.0 + 1.0e-12,
            "maximum_coordinate_Jacobian_condition_number": 5.0e3,
            "minimum_departure_direction_alignment_cosine": 0.99,
            "maximum_departure_transverse_fraction": 0.15,
            "maximum_H_over_R": 0.12,
            "minimum_scattering_optical_depth": 1.0,
            "maximum_repaired_decoder_full_state_relative_error": 5.0e-3,
            "maximum_repaired_decoder_coordinate_relative_mismatch": 5.0e-3,
            "maximum_compensated_full_rate_invariance_defect": 1.0e-12,
        },
        "decision": {
            "full_pass_classification": "compensated_decoder_independent_geometry_valid_to_0p015",
            "partial_pass_classification": "compensated_decoder_independent_geometry_valid_to_0p0125",
            "fail_classification": "compensated_decoder_independent_geometry_failed",
            "full_pass_authorizes_only": "definitions_only_recentered_transition_forecast_manifest",
            "partial_or_fail_authorizes_only": "definitions_only_decoder_architecture_revision_manifest",
        },
        "authorization_boundaries": {
            "new_truth_rate_calls": 0,
            "new_generator_assemblies": 0,
            "new_nonlinear_roots": 0,
            "propagated_states": 0,
            "trajectory_authorized": False,
            "physical_microburst_authorized": False,
            "predictive_cycle_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
    }


def _fit_checks(metrics: dict, gates: dict) -> dict:
    return {
        "kernel_condition": metrics["kernel_condition_number"]
        <= gates["maximum_kernel_condition_number"],
        "decoder_error": metrics["maximum_repaired_decoder_relative_error"]
        <= gates["maximum_repaired_decoder_relative_error"],
        "rate_invariance": metrics[
            "maximum_compensated_full_rate_invariance_defect"
        ] <= gates["maximum_compensated_full_rate_invariance_defect"],
        "inner_preservation": metrics["maximum_inner_certificate_shell_weight"]
        <= gates["maximum_inner_certificate_shell_weight"],
        "coordinate_mismatch": metrics[
            "maximum_repaired_decoder_coordinate_relative_mismatch"
        ] <= gates["maximum_repaired_decoder_coordinate_relative_mismatch"],
        "reconstruction": metrics["minimum_repaired_state_reconstruction_factor"]
        >= gates["minimum_repaired_state_reconstruction_factor"],
        "height": metrics["maximum_repaired_state_H_over_R"]
        <= gates["maximum_repaired_state_H_over_R"],
        "optical_depth": metrics[
            "minimum_repaired_state_scattering_optical_depth"
        ] >= gates["minimum_repaired_state_scattering_optical_depth"],
        "rate_budget": metrics["new_continuous_rate_evaluations"] == 0,
        "generator_budget": metrics["new_complete_generator_assemblies"] == 0,
        "root_budget": metrics["new_nonlinear_roots"] == 0,
        "propagation_budget": metrics["propagated_states"] == 0,
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
                    "scientific_status": "DEFINITIONS_ONLY",
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case",
                "path",
                "bytes",
                "sha256",
                "scientific_status",
            ],
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
        raise RuntimeError("compensated decoder manifest already canonicalized")
    arrays, metrics = _fit_repair()
    holdouts, labels, holdout_metrics = _holdout_design()
    contract = _contract()
    checks = _fit_checks(
        metrics, contract["binding_revealed_geometry_gates"]
    )
    if not all(checks.values()):
        raise RuntimeError(f"compensated decoder repair failed: {checks}")
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        CANONICAL_DIRECTORY / "compensated_decoder_repair.npz", **arrays
    )
    np.savez_compressed(
        CANONICAL_DIRECTORY / "holdout_design.npz",
        directions=holdouts,
        component_bounds=np.asarray(HOLDOUT_COMPONENT_BOUNDS),
    )
    _write_json(
        CANONICAL_DIRECTORY / "design_metrics.json",
        {"checks": checks, "fit": metrics, "holdout": holdout_metrics},
    )
    _write_json(
        CANONICAL_DIRECTORY / "holdout_design.json", {"labels": labels}
    )
    _write_json(CANONICAL_DIRECTORY / "contract.json", contract)
    _write_json(
        CANONICAL_DIRECTORY / "parent_lock.json",
        {
            "parent_commit": PARENT_COMMIT,
            "parent_parent": PARENT_PARENT,
            "parent_tree": PARENT_TREE,
            "parent_hashes": frozen["hashes"],
            "old_extension_sha256": _sha(OLD_EXTENSION),
            "failed_rate_arrays_sha256": _sha(FAILED_RATE_ARRAYS),
            "direction_design_sha256": _sha(DIRECTION_DESIGN),
            "inner_closure_sha256": _sha(INNER_CLOSURE),
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "only_parent_failure_repaired": "decoder_full_state_error",
        "independently_validated_rate_field_preserved_algebraically": True,
        "maximum_repaired_training_decoder_relative_error": metrics[
            "maximum_repaired_decoder_relative_error"
        ],
        "maximum_compensated_full_rate_invariance_defect": metrics[
            "maximum_compensated_full_rate_invariance_defect"
        ],
        "planned_independent_geometry_candidates": PLANNED_GEOMETRY_CANDIDATES,
        "new_truth_rate_calls": 0,
        "new_generator_assemblies": 0,
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "trajectory_authorized": False,
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
        parent.manifest.THIS_RUNNER,
        parent.manifest.THIS_TEST,
        parent.atlas.THIS_RUNNER,
        parent.atlas.THIS_TEST,
    )
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "DEFINITIONS_ONLY",
            "definition_commit": _git("rev-parse", "HEAD"),
            "definition_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "source_hashes": {
                relative: _sha(ROOT / relative) for relative in source_files
            },
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    )
    names = tuple(sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()))
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names
        ),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Compensated decoder-repair manifest WP10c9d6c7c3b5c4f25cl",
                "",
                "## Classification",
                "",
                f"`{CLASSIFICATION}`",
                "",
                "The exact truth layer and every independent vector-field gate passed. Only the 0.5% full-state decoder gate failed. This manifest repairs only that map with a geometry-trained Gaussian correction.",
                "",
                "The physical rate is preserved algebraically: the added decoder term is canceled by an equal `-G C_geometry` closure term. Therefore no passing rate coefficient is refit and the already validated full-state field is unchanged.",
                "",
                f"The maximum repaired training decoder error is `{metrics['maximum_repaired_decoder_relative_error']:.6e}`; the compensated full-rate invariance defect is `{metrics['maximum_compensated_full_rate_invariance_defect']:.6e}`. Eight new mixed-corner geometry states are frozen for independent validation.",
                "",
                "No new rate truth, generator, nonlinear root, propagation, trajectory, physical microburst, cycle evolution, or reduced slow evolution is authorized.",
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
