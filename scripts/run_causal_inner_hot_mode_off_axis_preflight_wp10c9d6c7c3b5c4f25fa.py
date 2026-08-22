#!/usr/bin/env python3
"""Test the hot free-field mode away from the artificial sampling curve."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from imri_qpe.layer3_minidisk_1d.conservative_free_field_rom import (  # noqa: E402
    ConservativeCoordinateSplit,
    canonical_rate_basis,
    polynomial_holdout,
    relative_projection_defects,
)
import run_causal_inner_arclength_segment_wp10c9d6c7c3b5c4f25f5 as arclength  # noqa: E402
import run_causal_inner_hot_free_field_rom_preflight_wp10c9d6c7c3b5c4f25f8 as hot  # noqa: E402
import run_causal_inner_hot_mode_off_axis_manifest_wp10c9d6c7c3b5c4f25f9_v2 as manifest  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25fa"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25fb"
PASS_CLASSIFICATION = "hot_discrete_mode_off_axis_conservative_patch_passed"
FAIL_CLASSIFICATION = "hot_discrete_mode_off_axis_conservative_patch_rejected"
ARTIFACT = "causal_inner_hot_mode_off_axis_preflight_wp10c9d6c7c3b5c4f25fa"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_HOT_MODE_OFF_AXIS_PREFLIGHT_"
    "WP10C9D6C7C3B5C4F25FA_2026-08-22.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = manifest.EXECUTION_RUNNER
THIS_TEST = manifest.EXECUTION_TEST


def _helper():
    return manifest._helper()


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(left) - np.asarray(right))
        / max(
            float(np.linalg.norm(left)),
            float(np.linalg.norm(right)),
            np.finfo(float).tiny,
        )
    )


def _validate_manifest(*, require_clean: bool) -> dict:
    helper = _helper()
    hashes = helper._validate_checksums(manifest.CANONICAL_DIRECTORY)
    summary = helper._read(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = helper._read(
        manifest.CANONICAL_DIRECTORY / "hot_mode_off_axis_contract.json"
    )
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or not summary["hot_mode_off_axis_preflight_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["work_package"] != manifest.WORK_PACKAGE
    ):
        raise RuntimeError("hot-mode off-axis manifest changed")
    current = {
        name: helper._sha(path) for name, path in manifest._decisive_inputs().items()
    }
    if current != contract["decisive_input_hashes"]:
        raise RuntimeError("hot-mode off-axis decisive input changed")
    for relative, expected in contract["frozen_source_hashes"].items():
        if helper._sha(ROOT / relative) != expected:
            raise RuntimeError(f"hot-mode off-axis source changed: {relative}")
    manifest._validate_parent(require_clean=False)
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("hot-mode off-axis preflight requires a clean tracked tree")
    return {"manifest_hashes": hashes, "contract": contract}


def _retraction_context() -> dict:
    source = arclength._source()
    exact_chart = arclength._exact_chart()
    base = source._base_inputs()
    seed = arclength._seed(base)
    model = base["model"]
    anchor_coordinate, _factors = model.coordinate(seed["state"])
    np.testing.assert_array_equal(anchor_coordinate, seed["coordinate"])
    coordinate_jacobian, coordinate_metrics = exact_chart._coordinate_jacobian(
        model, seed["state"]
    )
    gauge_basis = exact_chart._canonical_null_basis(coordinate_jacobian)
    began = time.perf_counter()
    anchor_augmented, augmented_metrics = exact_chart._augmented_jacobian(
        model, seed["state"], gauge_basis
    )
    anchor_wall = float(time.perf_counter() - began)
    if augmented_metrics["augmented_rank"] != exact_chart.PHYSICAL_DIMENSION:
        raise RuntimeError("off-axis anchor augmented Jacobian lost rank")
    return {
        "base": base,
        "model": model,
        "anchor_state": np.asarray(seed["state"]),
        "anchor_coordinate": np.asarray(anchor_coordinate),
        "anchor_model_state": np.asarray(model.decoded_state(anchor_coordinate)),
        "gauge_basis": gauge_basis,
        "anchor_delta": exact_chart._delta(model, seed["state"]),
        "anchor_augmented": anchor_augmented,
        "anchor_assembly_wall_seconds": anchor_wall,
        "anchor_coordinate_metrics": coordinate_metrics,
    }


def _retract(context: dict, coordinate: np.ndarray) -> tuple[np.ndarray, dict]:
    model = context["model"]
    target = np.asarray(coordinate, dtype=float)
    decoded = np.asarray(model.decoded_state(target), dtype=float)
    initial = context["anchor_state"] + decoded - context["anchor_model_state"]
    state, _matrix, metrics = arclength._transport()._transport_retract(
        model=model,
        initial_state=initial,
        target=target,
        gauge_basis=context["gauge_basis"],
        anchor_delta=context["anchor_delta"],
        anchor_augmented=context["anchor_augmented"],
    )
    recovered, factors = model.coordinate(state)
    metrics = {
        **metrics,
        "recovered_coordinate_relative_defect": _relative(recovered, target),
        "minimum_decoder_reconstruction_factor": float(np.min(factors)),
    }
    return np.asarray(state), metrics


def _evaluate_target(
    context: dict,
    coordinate: np.ndarray,
    *,
    label: str,
) -> tuple[dict, dict[str, np.ndarray]]:
    began = time.perf_counter()
    state, retraction = _retract(context, coordinate)
    retraction_wall = float(time.perf_counter() - began)
    source = arclength._source()._post().exact_rate.rate_source
    metrics, arrays = hot._evaluate_free_state(
        state,
        model=context["model"],
        configuration=context["base"]["configuration"],
        rate_source=source,
        exact_chart=arclength._exact_chart(),
    )
    metrics.update({
        "label": label,
        "retraction": retraction,
        "retraction_wall_seconds": retraction_wall,
    })
    arrays.update({
        "coordinate470": np.asarray(coordinate),
        "primitive_state": state,
    })
    print(
        f"off-axis {label}: |r|={metrics['coordinate_free_rate_norm_per_second']:.6e}/s "
        f"retract={retraction_wall:.3f}s free={metrics['total_free_evaluation_wall_seconds']:.3f}s",
        flush=True,
    )
    return metrics, arrays


def _select_basis(
    hidden_rates: np.ndarray,
    training_indices: np.ndarray,
    holdout_indices: np.ndarray,
) -> tuple[np.ndarray, int, dict]:
    attempts = {}
    selected = None
    selected_rank = 0
    for rank in manifest.HIDDEN_RATE_RANKS:
        basis, singular, energy = canonical_rate_basis(
            hidden_rates[training_indices], rank
        )
        defects = relative_projection_defects(hidden_rates, basis)
        attempts[str(rank)] = {
            "training_singular_values": singular.tolist(),
            "training_cumulative_energy": energy.tolist(),
            "maximum_training_defect": float(np.max(defects[training_indices])),
            "maximum_holdout_defect": float(np.max(defects[holdout_indices])),
        }
        selected = basis
        selected_rank = rank
        if float(np.max(defects[holdout_indices])) <= manifest.MAXIMUM_HIDDEN_RATE_HOLDOUT_DEFECT:
            break
    if selected is None:
        raise RuntimeError("no off-axis hot hidden basis constructed")
    return selected, selected_rank, attempts


def _target_coordinates(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    coordinates = np.asarray(arrays["coordinates5x470"])
    rates = np.asarray(arrays["coordinate_free_rates5x470_per_s"])
    center = manifest.HOT_CENTER_INDEX
    diagonal = manifest.DIAGONAL_ARCLENGTH_INDEX
    step = manifest.PHYSICAL_MACRO_STEP_SECONDS
    return {
        "physical_half": coordinates[center]
        + manifest.PHYSICAL_AXIS_FRACTIONS[0] * step * rates[center],
        "physical_full": coordinates[center]
        + manifest.PHYSICAL_AXIS_FRACTIONS[1] * step * rates[center],
        "diagonal_full": coordinates[diagonal] + step * rates[diagonal],
    }


def _evaluate_targets(
    context: dict,
    targets: dict[str, np.ndarray],
    locked: dict,
) -> tuple[list[dict], dict[str, dict[str, np.ndarray]]]:
    helper = _helper()
    identity = {
        "work_package": WORK_PACKAGE,
        "manifest_hashes": locked["manifest_hashes"],
        "target_hashes": {
            label: hashlib.sha256(
                np.ascontiguousarray(coordinate).tobytes()
            ).hexdigest()
            for label, coordinate in targets.items()
        },
    }
    identity_path = SCRATCH_DIRECTORY / "identity.json"
    if SCRATCH_DIRECTORY.exists():
        if not identity_path.exists() or helper._read(identity_path) != identity:
            raise RuntimeError("off-axis scratch identity changed")
    else:
        SCRATCH_DIRECTORY.mkdir(parents=True)
        helper._write_json(identity_path, identity)
    records = []
    evaluated = {}
    for label, coordinate in targets.items():
        metrics_path = SCRATCH_DIRECTORY / f"{label}.json"
        arrays_path = SCRATCH_DIRECTORY / f"{label}.npz"
        if metrics_path.exists() != arrays_path.exists():
            raise RuntimeError(f"partial off-axis scratch witness: {label}")
        if metrics_path.exists():
            metrics = helper._read(metrics_path)
            arrays = helper._load_npz(arrays_path)
            np.testing.assert_array_equal(arrays["coordinate470"], coordinate)
            print(f"off-axis {label}: reused exact scratch witness", flush=True)
        else:
            metrics, arrays = _evaluate_target(context, coordinate, label=label)
            helper._write_json(metrics_path, metrics)
            with arrays_path.open("wb") as handle:
                np.savez_compressed(handle, **arrays)
        records.append(metrics)
        evaluated[label] = arrays
    return records, evaluated


def _evaluate(locked: dict) -> tuple[dict, dict[str, np.ndarray]]:
    helper = _helper()
    hot_arrays = helper._load_npz(
        hot.CANONICAL_DIRECTORY / "hot_free_field_arrays.npz"
    )
    context = _retraction_context()
    targets = _target_coordinates(hot_arrays)
    records, evaluated = _evaluate_targets(context, targets, locked)

    geometry = context["base"]["geometry"]
    split = ConservativeCoordinateSplit(
        macro_restriction=geometry["R"],
        macro_lift=geometry["L"],
        hidden_dual=geometry["Q"],
        hidden_lift=geometry["Z"],
        tolerance=manifest.MAXIMUM_SPLIT_IDENTITY_DEFECT,
    )
    arc_coordinate_rates = np.asarray(hot_arrays["coordinate_free_rates5x470_per_s"])
    arc_hidden_rates = np.asarray(hot_arrays["hidden_free_rates5x388_per_s"])
    off_coordinate_rates = np.stack(
        [evaluated[label]["coordinate_free_rate470_per_s"] for label in targets]
    )
    off_macro_rates = []
    off_hidden_rates = []
    decomposition_defects = []
    for rate in off_coordinate_rates:
        macro, hidden = split.split_rate(rate)
        reconstructed = split.compose(macro, hidden)
        off_macro_rates.append(macro)
        off_hidden_rates.append(hidden)
        decomposition_defects.append(_relative(reconstructed, rate))
    off_macro_rates = np.asarray(off_macro_rates)
    off_hidden_rates = np.asarray(off_hidden_rates)

    all_hidden_rates = np.vstack((arc_hidden_rates, off_hidden_rates))
    training_indices = np.asarray((0, 2, 4, 6), dtype=int)
    holdout_indices = np.asarray((1, 3, 5, 7), dtype=int)
    hidden_basis, hidden_rank, basis_attempts = _select_basis(
        all_hidden_rates, training_indices, holdout_indices
    )
    hidden_defects = relative_projection_defects(all_hidden_rates, hidden_basis)

    center = manifest.HOT_CENTER_INDEX
    diagonal = manifest.DIAGONAL_ARCLENGTH_INDEX
    f0 = arc_coordinate_rates[center]
    fhalf, ffull, fdiag = off_coordinate_rates
    physical_half_prediction = 0.5 * (f0 + ffull)
    physical_axis_defect = _relative(physical_half_prediction, fhalf)
    nodes = np.asarray(hot_arrays["nodes"])
    heldout, arc_predictions, _arc_defects = polynomial_holdout(
        nodes, arc_coordinate_rates, np.asarray((0, 2, 4), dtype=int)
    )
    arc_prediction = arc_predictions[int(np.flatnonzero(heldout == diagonal)[0])]
    diagonal_prediction = arc_prediction + (ffull - f0)
    diagonal_defect = _relative(diagonal_prediction, fdiag)
    rate_variations = np.asarray((
        _relative(fhalf, f0),
        _relative(ffull, f0),
        _relative(fdiag, arc_coordinate_rates[diagonal]),
    ))
    heun_correction = float(
        0.5 * np.linalg.norm(ffull - f0)
        / max(float(np.linalg.norm(f0)), np.finfo(float).tiny)
    )
    heun_coordinate = (
        np.asarray(hot_arrays["coordinates5x470"])[center]
        + 0.5 * manifest.PHYSICAL_MACRO_STEP_SECONDS * (f0 + ffull)
    )
    heun_state, heun_retraction = _retract(context, heun_coordinate)
    heun_physical = arclength._exact_chart()._physical_audit(
        context["model"], heun_state, context["model"].coordinate(heun_state)[1]
    )

    all_retractions = [item["retraction"] for item in records] + [heun_retraction]
    maxima = {
        "maximum_split_identity_defect": max(split.identity_defects.values()),
        "maximum_coordinate_decomposition_defect": float(max(decomposition_defects)),
        "maximum_coordinate_retraction_residual": float(max(
            item["coordinate_residual_infinity"] for item in all_retractions
        )),
        "maximum_gauge_retraction_residual": float(max(
            item["gauge_residual_infinity"] for item in all_retractions
        )),
        "maximum_scaled_anchor_departure": float(max(
            item["maximum_scaled_anchor_departure"] for item in all_retractions
        )),
        "maximum_coordinate_jacobian_condition_number": float(max(
            item["coordinate_jacobian_condition_number"] for item in records
        )),
        "minimum_reconstruction_factor": float(min(
            [item["minimum_reconstruction_factor"] for item in records]
            + [heun_physical["minimum_reconstruction_factor"]]
        )),
        "maximum_height_ratio": float(max(
            [item["maximum_height_ratio"] for item in records]
            + [heun_physical["maximum_height_ratio"]]
        )),
        "minimum_scattering_optical_depth": float(min(
            [item["minimum_scattering_optical_depth"] for item in records]
            + [heun_physical["minimum_scattering_optical_depth"]]
        )),
        "selected_hidden_rate_rank": hidden_rank,
        "maximum_hidden_rate_training_defect": float(np.max(hidden_defects[training_indices])),
        "maximum_hidden_rate_holdout_defect": float(np.max(hidden_defects[holdout_indices])),
        "physical_axis_linear_holdout_defect": physical_axis_defect,
        "separable_diagonal_operator_defect": diagonal_defect,
        "maximum_free_rate_variation": float(np.max(rate_variations)),
        "euler_heun_correction_fraction": heun_correction,
        "anchor_exact_assembly_wall_seconds": context["anchor_assembly_wall_seconds"],
        "new_exact_free_rate_calls": len(records),
        "new_fixed_Q_reaction_calls": 0,
    }
    gates = {
        "split_identities": maxima["maximum_split_identity_defect"] <= manifest.MAXIMUM_SPLIT_IDENTITY_DEFECT,
        "coordinate_decomposition": maxima["maximum_coordinate_decomposition_defect"] <= manifest.MAXIMUM_COORDINATE_DECOMPOSITION_DEFECT,
        "coordinate_retraction": maxima["maximum_coordinate_retraction_residual"] <= manifest.MAXIMUM_COORDINATE_RETRACTION_RESIDUAL,
        "gauge_retraction": maxima["maximum_gauge_retraction_residual"] <= manifest.MAXIMUM_GAUGE_RETRACTION_RESIDUAL,
        "local_departure": maxima["maximum_scaled_anchor_departure"] <= manifest.MAXIMUM_SCALED_ANCHOR_DEPARTURE,
        "coordinate_jacobian_rank": all(item["coordinate_jacobian_rank"] == 470 for item in records),
        "coordinate_jacobian_condition": maxima["maximum_coordinate_jacobian_condition_number"] <= manifest.MAXIMUM_COORDINATE_JACOBIAN_CONDITION_NUMBER,
        "reconstruction": maxima["minimum_reconstruction_factor"] >= 1.0 - 1.0e-12,
        "height": maxima["maximum_height_ratio"] <= 0.5,
        "optical_depth": maxima["minimum_scattering_optical_depth"] >= 1.0,
        "hidden_rate_holdout": maxima["maximum_hidden_rate_holdout_defect"] <= manifest.MAXIMUM_HIDDEN_RATE_HOLDOUT_DEFECT,
        "physical_axis_linear_holdout": physical_axis_defect <= manifest.MAXIMUM_PHYSICAL_AXIS_LINEAR_HOLDOUT_DEFECT,
        "separable_diagonal_operator": diagonal_defect <= manifest.MAXIMUM_SEPARABLE_DIAGONAL_OPERATOR_DEFECT,
        "free_rate_variation": maxima["maximum_free_rate_variation"] <= manifest.MAXIMUM_FREE_RATE_VARIATION,
        "euler_heun_correction": heun_correction <= manifest.MAXIMUM_EULER_HEUN_CORRECTION_FRACTION,
        "truth_budget": len(records) <= manifest.MAXIMUM_NEW_EXACT_FREE_RATE_CALLS,
        "reaction_forbidden": maxima["new_fixed_Q_reaction_calls"] == 0,
        "no_roots_or_microsteps": True,
    }
    passed = bool(all(gates.values()))
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,
        "passed": passed,
        "gates": gates,
        "gate_values": maxima,
        "basis_attempts": basis_attempts,
        "records": records,
        "input_lock": locked,
        "new_nonlinear_fixed_Q_roots": 0,
        "new_BDF_microsteps": 0,
    }
    arrays = {
        "arc_nodes": nodes,
        "arc_coordinates5x470": np.asarray(hot_arrays["coordinates5x470"]),
        "arc_coordinate_free_rates5x470_per_s": arc_coordinate_rates,
        "target_coordinates3x470": np.stack(list(targets.values())),
        "target_primitive_states": np.stack(
            [evaluated[label]["primitive_state"] for label in targets]
        ),
        "off_axis_coordinate_free_rates3x470_per_s": off_coordinate_rates,
        "off_axis_macro_free_rates3x82_per_s": off_macro_rates,
        "off_axis_hidden_free_rates3x388_per_s": off_hidden_rates,
        "extended_hidden_rate_basis388xr": hidden_basis,
        "extended_hidden_rate_projection_defects": hidden_defects,
        "physical_half_prediction470_per_s": physical_half_prediction,
        "diagonal_prediction470_per_s": diagonal_prediction,
        "rate_variations": rate_variations,
        "heun_coordinate470": heun_coordinate,
        "heun_primitive_state": heun_state,
    }
    return metrics, arrays


def _update_catalog(summary: dict) -> None:
    helper = _helper()
    cold = arclength._source()._post().manifest.transition.manifest.cold.manifest
    with cold.CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    status = "SUPPORTED" if summary["passed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({
                "case": ARTIFACT,
                "path": str(path.relative_to(ROOT)),
                "bytes": str(path.stat().st_size),
                "sha256": helper._sha(path),
                "scientific_status": status,
            })
    with cold.CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "case", "path", "bytes", "sha256", "scientific_status"
        ), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    catalog = helper._read(cold.CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": summary["passed"],
    }
    catalog.update({
        "case_count": len({row["case"] for row in rows}),
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "all_payload_hashes_recorded": True,
        "latest_source_parent_commit": helper._git("rev-parse", "HEAD"),
        "latest_work_package": WORK_PACKAGE,
    })
    helper._write_json(cold.CANONICAL_SUMMARY, catalog)


def _run() -> dict:
    helper = _helper()
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("hot-mode off-axis preflight already exists")
    locked = _validate_manifest(require_clean=True)
    metrics, arrays = _evaluate(locked)
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "hot_mode_off_axis_metrics.json", metrics)
    with (CANONICAL_DIRECTORY / "hot_mode_off_axis_arrays.npz").open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": metrics["passed"],
        "truth_free_hot_mode_engine_manifest_authorized": metrics["passed"],
        "fixed_Q_physical_phase_authorized": False,
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT if metrics["passed"] else None,
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    helper._write_json(CANONICAL_DIRECTORY / "provenance.json", {
        "runner": THIS_RUNNER,
        "test": THIS_TEST,
        "implementation_commit": helper._git("rev-parse", "HEAD"),
        "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"),
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
    })
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{helper._sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    values = metrics["gate_values"]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join((
            "# Hot-mode off-axis free-field preflight",
            "",
            f"Classification: `{metrics['classification']}`.",
            "",
            f"Selected hot hidden rank: `{values['selected_hidden_rate_rank']}`; held-out hidden-rate defect: `{values['maximum_hidden_rate_holdout_defect']:.6e}`.",
            "",
            f"Physical-axis linear holdout defect: `{values['physical_axis_linear_holdout_defect']:.6e}`; separable diagonal defect: `{values['separable_diagonal_operator_defect']:.6e}`; Euler/Heun correction fraction: `{values['euler_heun_correction_fraction']:.6e}`.",
            "",
            f"The tested physical macro step was `{manifest.PHYSICAL_MACRO_STEP_SECONDS:.6e}` s. Exactly `{values['new_exact_free_rate_calls']}` original free-field calls and zero fixed-Q reactions, roots, or BDF microsteps were used.",
            "",
        )),
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
    print(json.dumps(_run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
