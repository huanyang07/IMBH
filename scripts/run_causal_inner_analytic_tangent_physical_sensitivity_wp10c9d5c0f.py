#!/usr/bin/env python3
"""Run WP10c9d5c0f derivative-choice physical sensitivity.

This production-neutral audit compares the rejected historical frozen
candidate generator with the cross-grid-certified analytic frozen-subspace
generator.  Both propagations use one common sixth-order physical export map,
so the comparison isolates the generator representation rather than changing
the physical observable definition at the same time.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_causal_inner_cross_grid_analytic_tangent_wp10c9d5c0e as wp10c9d5c0e
import run_causal_inner_cross_grid_hardening_wp10c9d5c0 as wp10c9d5c0
import run_causal_inner_derivative_repair_wp10c9d5c0a as wp10c9d5c0a
import run_causal_inner_micro_export_preflight_wp10c9d0 as wp10c9d0


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d5c0f"
ANALYZED_BASE_COMMIT = "e5fd93352aea3dc920e528bb566b60fa7a3c8b0c"
ANALYZED_BASE_PARENT = "d57bcc3e63bcd778823736a795a9311592173bd9"
ANALYZED_BASE_TREE = "4316485d9358abe2462878d56c95282aa22217c2"
THIS_RUNNER = (
    "scripts/"
    "run_causal_inner_analytic_tangent_physical_sensitivity_wp10c9d5c0f.py"
)

LABELS = tuple(wp10c9d5c0e.LABELS)
METHODS = ("historical_finite_difference", "analytic_frozen_subspace")
PERTURBATIONS = ("common_mode", "heldout_near_excision")
OUTPUT_REFERENCE_ORDER = 6
OUTPUT_REFERENCE_METHOD = wp10c9d5c0a.METHOD_NAMES[1]

MAXIMUM_DERIVATIVE_EXPORT_DIFFERENCE = 5.0e-3
MAXIMUM_DERIVATIVE_TO_SPATIAL_RATIO = 0.10
MAXIMUM_RESTART_DEFECT = 1.0e-10

PARENT_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_cross_grid_analytic_tangent_wp10c9d5c0e"
)
PARENT_SUMMARY = PARENT_DIRECTORY / "summary.json"
PARENT_DECISIVE_ARRAYS = PARENT_DIRECTORY / "decisive_arrays.npz"
PARENT_REPLAY_CONTEXTS = PARENT_DIRECTORY / "replay_contexts.json"
PARENT_REPLAY_INPUTS = PARENT_DIRECTORY / "replay_inputs.npz"

CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_analytic_tangent_physical_sensitivity_wp10c9d5c0f"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"

IMPLEMENTATION_SOURCES = (
    THIS_RUNNER,
    "tests/"
    "test_causal_inner_analytic_tangent_physical_sensitivity_wp10c9d5c0f.py",
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


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def _relative_difference(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    scale = max(
        float(np.linalg.norm(left)),
        float(np.linalg.norm(right)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(left - right) / scale)


def _refresh_sha256s(directory: Path) -> None:
    entries = []
    for path in sorted(directory.iterdir()):
        if path.name == "SHA256SUMS.txt" or not path.is_file():
            continue
        entries.append(f"{_sha256(path)}  {path.name}")
    (directory / "SHA256SUMS.txt").write_text(
        "\n".join(entries) + "\n",
        encoding="utf-8",
    )


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_analyzed_git_identity() -> dict:
    commit = _git("rev-parse", ANALYZED_BASE_COMMIT)
    parent = _git("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
    tree = _git("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
    if (commit, parent, tree) != (
        ANALYZED_BASE_COMMIT,
        ANALYZED_BASE_PARENT,
        ANALYZED_BASE_TREE,
    ):
        raise RuntimeError("WP10c9d5c0f analyzed Git identity changed")
    return {
        "analyzed_base_commit": commit,
        "analyzed_base_parent_commit": parent,
        "analyzed_base_tree_sha": tree,
    }


def _source_manifest() -> tuple[dict[str, str], str]:
    hashes = {
        path: _sha256(ROOT / path)
        for path in IMPLEMENTATION_SOURCES
        if (ROOT / path).exists()
    }
    digest = hashlib.sha256()
    for path, value in sorted(hashes.items()):
        digest.update(path.encode("utf-8"))
        digest.update(value.encode("ascii"))
    return hashes, digest.hexdigest()


def _load_parent() -> tuple[dict, dict, dict[str, np.ndarray]]:
    parent = json.loads(PARENT_SUMMARY.read_text(encoding="utf-8"))
    if (
        not parent["passed"]
        or not parent["cross_grid_analytic_tangent_certified"]
        or not parent["derivative_choice_physical_sensitivity_authorized"]
        or parent["wp10c9d5c1_extended_localization_authorized"]
        or not parent["parent_wp10c9d5_candidate_remains_rejected"]
    ):
        raise RuntimeError("WP10c9d5c0e binding classification changed")
    replay_payload, replay_arrays = wp10c9d5c0e._load_replay_inputs()
    if _sha256(PARENT_REPLAY_INPUTS) != parent["replay_inputs_sha256"]:
        raise RuntimeError("WP10c9d5c0e replay input archive changed")
    if _sha256(PARENT_REPLAY_CONTEXTS) != parent["replay_contexts_sha256"]:
        raise RuntimeError("WP10c9d5c0e replay context changed")
    with np.load(PARENT_DECISIVE_ARRAYS, allow_pickle=False) as source:
        decisive = {
            name: np.asarray(source[name])
            for name in source.files
        }
    if _sha256(PARENT_DECISIVE_ARRAYS) != parent[
        "decisive_arrays_sha256"
    ]:
        raise RuntimeError("WP10c9d5c0e decisive arrays changed")
    return parent, wp10c9d5c0e._configurations(
        replay_payload,
        replay_arrays,
    ), decisive


def _analytic_generator(
    label: str,
    parent_arrays: dict[str, np.ndarray],
) -> np.ndarray:
    return wp10c9d5c0e._unpack_sparse(
        f"{label}__analytic_candidate_generator",
        parent_arrays,
    ).toarray()


def _perturbation_initial(
    configuration: dict,
    perturbation: str,
) -> np.ndarray:
    if perturbation == "common_mode":
        return np.asarray(configuration["initial"], dtype=float)
    if perturbation != "heldout_near_excision":
        raise ValueError(f"unknown perturbation: {perturbation}")
    scaled = np.asarray(
        configuration["directions"]["near_excision_0"],
        dtype=float,
    )
    columns = np.asarray(
        configuration["candidate_native"]["primitive_column_scales"],
        dtype=float,
    )
    amplitudes = np.asarray(
        configuration["amplitudes"],
        dtype=float,
    ).ravel()
    return (scaled * columns / amplitudes).reshape(
        np.asarray(configuration["initial"]).shape
    )


def _component_rms_differences(
    first: np.ndarray,
    second: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    difference = (
        np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    ) / np.asarray(scales, dtype=float)[None, :]
    return np.sqrt(np.mean(difference * difference, axis=0))


def _state_difference_history(
    historical: np.ndarray,
    analytic: np.ndarray,
) -> np.ndarray:
    left = np.asarray(historical, dtype=float).reshape(
        historical.shape[0],
        -1,
    )
    right = np.asarray(analytic, dtype=float).reshape(
        analytic.shape[0],
        -1,
    )
    numerator = np.linalg.norm(left - right, axis=1)
    scale = np.maximum(
        np.maximum(
            np.linalg.norm(left, axis=1),
            np.linalg.norm(right, axis=1),
        ),
        np.finfo(float).tiny,
    )
    return numerator / scale


def _history(
    configuration: dict,
    generator: np.ndarray,
    blocks: dict,
    inner_flux_matrix: np.ndarray,
    initial: np.ndarray,
) -> dict:
    run_configuration = {
        **configuration,
        "initial": np.asarray(initial, dtype=float),
    }
    return wp10c9d5c0._observable_history(
        run_configuration,
        generator,
        blocks,
        inner_flux_matrix,
    )


def _pair_report(
    historical: dict,
    analytic: dict,
    observable_scales: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    times = np.asarray(analytic["times"], dtype=float)
    duration = max(float(times[-1]), np.finfo(float).tiny)
    signal_components = _component_rms_differences(
        historical["signals"],
        analytic["signals"],
        observable_scales,
    )
    cumulative_components = _component_rms_differences(
        historical["cumulative_signals"],
        analytic["cumulative_signals"],
        observable_scales * duration,
    )
    state_difference = _state_difference_history(
        historical["state"],
        analytic["state"],
    )
    first_cell_difference = _state_difference_history(
        historical["state"][:, :1, :],
        analytic["state"][:, :1, :],
    )
    maximum_export = float(
        max(
            np.max(signal_components),
            np.max(cumulative_components),
        )
    )
    return {
        "signal_component_rms_differences": dict(
            zip(
                wp10c9d0.OBSERVABLE_NAMES,
                signal_components,
                strict=True,
            )
        ),
        "cumulative_component_rms_differences": dict(
            zip(
                wp10c9d0.OBSERVABLE_NAMES,
                cumulative_components,
                strict=True,
            )
        ),
        "signal_difference": float(np.max(signal_components)),
        "cumulative_difference": float(
            np.max(cumulative_components)
        ),
        "maximum_export_difference": maximum_export,
        "maximum_state_action_difference": float(
            np.max(state_difference)
        ),
        "endpoint_state_action_difference": float(
            state_difference[-1]
        ),
        "maximum_first_cell_state_difference": float(
            np.max(first_cell_difference)
        ),
        "historical_restart_defect": float(
            historical["restart_defect"]
        ),
        "analytic_restart_defect": float(analytic["restart_defect"]),
    }, {
        "signal_component_rms_differences": signal_components,
        "cumulative_component_rms_differences": cumulative_components,
        "state_action_difference_history": state_difference,
        "first_cell_state_difference_history": first_cell_difference,
    }


def run() -> dict:
    started = time.perf_counter()
    identity = _validate_analyzed_git_identity()
    parent, configurations, parent_arrays = _load_parent()
    observable_scales = wp10c9d5c0._fixed_observable_scales(configurations)
    decisive: dict[str, np.ndarray] = {
        "fixed_observable_scales": observable_scales,
    }
    histories = {
        perturbation: {method: {} for method in METHODS}
        for perturbation in PERTURBATIONS
    }
    grid_reports = {perturbation: {} for perturbation in PERTURBATIONS}

    for label in LABELS:
        configuration = configurations[label]
        historical_generator = np.asarray(
            configuration["candidate_native"]["candidate_generator"],
            dtype=float,
        )
        analytic_generator = _analytic_generator(label, parent_arrays)
        generators = {
            METHODS[0]: historical_generator,
            METHODS[1]: analytic_generator,
        }
        blocks = configuration["references"][OUTPUT_REFERENCE_ORDER]
        print(f"WP10c9d5c0f: inner export map {label}", flush=True)
        inner_flux_matrix = wp10c9d5c0a._inner_flux_matrices(
            configuration
        )[OUTPUT_REFERENCE_METHOD]
        decisive[f"{label}__inner_flux_matrix"] = inner_flux_matrix
        decisive[f"{label}__generator_relative_difference"] = np.asarray(
            [_relative_difference(
                historical_generator,
                analytic_generator,
            )],
            dtype=float,
        )

        for perturbation in PERTURBATIONS:
            initial = _perturbation_initial(configuration, perturbation)
            decisive[f"{perturbation}__{label}__initial"] = initial
            for method, generator in generators.items():
                print(
                    f"WP10c9d5c0f: propagate {label} "
                    f"{perturbation} {method}",
                    flush=True,
                )
                history = _history(
                    configuration,
                    generator,
                    blocks,
                    inner_flux_matrix,
                    initial,
                )
                histories[perturbation][method][label] = history
                for name in (
                    "times",
                    "signals",
                    "cumulative_signals",
                    "face_fluxes",
                    "scaled_state",
                ):
                    decisive[
                        f"{perturbation}__{method}__{label}__{name}"
                    ] = np.asarray(history[name], dtype=float)
                decisive[
                    f"{perturbation}__{method}__{label}__first_cell_state"
                ] = np.asarray(history["state"], dtype=float)[:, 0, :]

            report, arrays = _pair_report(
                histories[perturbation][METHODS[0]][label],
                histories[perturbation][METHODS[1]][label],
                observable_scales,
            )
            grid_reports[perturbation][label] = report
            for name, values in arrays.items():
                decisive[
                    f"{perturbation}__{label}__comparison__{name}"
                ] = values

    perturbation_reports = {}
    passed = True
    for perturbation in PERTURBATIONS:
        medium = histories[perturbation][METHODS[1]][LABELS[1]]
        fine = histories[perturbation][METHODS[1]][LABELS[2]]
        duration = max(
            float(np.asarray(fine["times"])[-1]),
            np.finfo(float).tiny,
        )
        spatial_signal = wp10c9d5c0._maximum_component_rms_difference(
            medium["signals"],
            fine["signals"],
            observable_scales,
        )
        spatial_cumulative = (
            wp10c9d5c0._maximum_component_rms_difference(
                medium["cumulative_signals"],
                fine["cumulative_signals"],
                observable_scales * duration,
            )
        )
        binding_spatial = max(spatial_signal, spatial_cumulative)
        maximum_derivative = max(
            grid_reports[perturbation][label][
                "maximum_export_difference"
            ]
            for label in LABELS
        )
        ratio = maximum_derivative / max(
            binding_spatial,
            np.finfo(float).tiny,
        )
        restart = max(
            grid_reports[perturbation][label][key]
            for label in LABELS
            for key in (
                "historical_restart_defect",
                "analytic_restart_defect",
            )
        )
        perturbation_passed = bool(
            maximum_derivative
            <= MAXIMUM_DERIVATIVE_EXPORT_DIFFERENCE
            and ratio <= MAXIMUM_DERIVATIVE_TO_SPATIAL_RATIO
            and restart <= MAXIMUM_RESTART_DEFECT
        )
        passed = bool(passed and perturbation_passed)
        perturbation_reports[perturbation] = {
            "grids": grid_reports[perturbation],
            "maximum_derivative_export_difference": maximum_derivative,
            "binding_medium_fine_signal_difference": spatial_signal,
            "binding_medium_fine_cumulative_difference": (
                spatial_cumulative
            ),
            "binding_medium_fine_spatial_difference": binding_spatial,
            "derivative_to_spatial_ratio": ratio,
            "maximum_restart_defect": restart,
            "passed": perturbation_passed,
        }

    classification = (
        "analytic_tangent_physical_sensitivity_passed_"
        "extended_non_tautological_localization_authorized"
        if passed
        else "analytic_tangent_physical_sensitivity_failed_"
        "extended_localization_blocked"
    )
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)
    source_hashes, source_manifest = _source_manifest()
    gates = {
        "maximum_derivative_export_difference": (
            MAXIMUM_DERIVATIVE_EXPORT_DIFFERENCE
        ),
        "maximum_derivative_to_spatial_ratio": (
            MAXIMUM_DERIVATIVE_TO_SPATIAL_RATIO
        ),
        "maximum_restart_defect": MAXIMUM_RESTART_DEFECT,
    }
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "labels": LABELS,
        "methods": METHODS,
        "perturbations": PERTURBATIONS,
        "observable_names": wp10c9d0.OBSERVABLE_NAMES,
        "output_reference_order": OUTPUT_REFERENCE_ORDER,
        "output_reference_method": OUTPUT_REFERENCE_METHOD,
        "gates": gates,
    }
    _write_json(CONFIG_PATH, config)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        **identity,
        "parent_wp10c9d5c0e_summary_path": _relative(PARENT_SUMMARY),
        "parent_wp10c9d5c0e_summary_sha256": _sha256(PARENT_SUMMARY),
        "parent_wp10c9d5c0e_decisive_arrays_path": _relative(
            PARENT_DECISIVE_ARRAYS
        ),
        "parent_wp10c9d5c0e_decisive_arrays_sha256": _sha256(
            PARENT_DECISIVE_ARRAYS
        ),
        "comparison_scope": (
            "historical finite-difference candidate generator versus "
            "cross-grid-certified analytic frozen-subspace generator; "
            "common sixth-order moving-projector export map"
        ),
        "fixed_observable_scales": dict(
            zip(
                wp10c9d0.OBSERVABLE_NAMES,
                observable_scales,
                strict=True,
            )
        ),
        "perturbations": perturbation_reports,
        "parent_wp10c9d5_candidate_remains_rejected": True,
        "parent_wp10c9d5b_branch_d_preserved": True,
        "cross_grid_analytic_tangent_remains_certified": bool(
            parent["cross_grid_analytic_tangent_certified"]
        ),
        "derivative_choice_physical_sensitivity_passed": passed,
        "wp10c9d5c1_extended_localization_authorized": passed,
        "self_consistent_tangent_authorized": False,
        "frozen_candidate_recertification_authorized": False,
        "production_operator_authorized": False,
        "nonlinear_candidate_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "decisive_arrays_path": _relative(DECISIVE_ARRAYS),
        "decisive_arrays_sha256": _sha256(DECISIVE_ARRAYS),
        "decisive_array_hashes": {
            name: _array_sha256(values)
            for name, values in decisive.items()
        },
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "environment": wp10c9d5c0._environment(),
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_json(SUMMARY_PATH, summary)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        **identity,
        "generation_command": (
            "PYTHONPATH=src python3 "
            "scripts/"
            "run_causal_inner_analytic_tangent_physical_sensitivity_"
            "wp10c9d5c0f.py"
        ),
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        "parent_canonical_hashes": {
            _relative(path): _sha256(path)
            for path in (
                PARENT_SUMMARY,
                PARENT_DECISIVE_ARRAYS,
                PARENT_REPLAY_CONTEXTS,
                PARENT_REPLAY_INPUTS,
            )
        },
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "scientific_status": "CERTIFIED" if passed else "REJECTED",
        "authorization_status": (
            "EXTENDED NON-TAUTOLOGICAL LOCALIZATION ONLY"
            if passed
            else "DERIVATIVE/OPERATOR DIAGNOSIS ONLY"
        ),
        "establishes": (
            "Whether replacing the historical finite-difference generator "
            "by the cross-grid-certified analytic frozen-subspace generator "
            "materially changes common-mode or held-out near-excision "
            "physical export histories."
        ),
        "does_not_establish": (
            "Physical export convergence, a recovery radius, a repaired "
            "operator, nonlinear convergence, fixed-Q closure, or reduced "
            "evolution."
        ),
    }
    _write_json(PROVENANCE_PATH, provenance)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    return summary


def main() -> None:
    print(json.dumps(_plain(run()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
