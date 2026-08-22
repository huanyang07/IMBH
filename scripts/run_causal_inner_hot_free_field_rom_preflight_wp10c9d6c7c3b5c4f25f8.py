#!/usr/bin/env python3
"""Evaluate and compress the original free field at accepted hot states."""

from __future__ import annotations

import argparse
import csv
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
import run_causal_inner_hot_free_field_rom_manifest_wp10c9d6c7c3b5c4f25f7 as manifest  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25f8"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25f9"
CLASSIFICATION = "hot_exact_free_field_conservative_hidden_amplitude_rom_preflight_passed"
FAIL_CLASSIFICATION = "hot_exact_free_field_conservative_rom_preflight_rejected"
ARTIFACT = "causal_inner_hot_free_field_rom_preflight_wp10c9d6c7c3b5c4f25f8"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_HOT_FREE_FIELD_ROM_PREFLIGHT_"
    "WP10C9D6C7C3B5C4F25F8_2026-08-22.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = manifest.EXECUTION_RUNNER
THIS_TEST = manifest.EXECUTION_TEST


def _helper():
    return manifest._helper()


def _validate_manifest(*, require_clean: bool) -> dict:
    helper = _helper()
    hashes = helper._validate_checksums(manifest.CANONICAL_DIRECTORY)
    summary = helper._read(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = helper._read(
        manifest.CANONICAL_DIRECTORY / "hot_free_field_rom_contract.json"
    )
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or not summary["hot_free_field_rom_preflight_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["work_package"] != manifest.WORK_PACKAGE
    ):
        raise RuntimeError("hot free-field ROM manifest changed")
    current_inputs = {
        name: helper._sha(path) for name, path in manifest._decisive_inputs().items()
    }
    if current_inputs != contract["decisive_input_hashes"]:
        raise RuntimeError("hot free-field decisive input changed")
    for relative, expected in contract["frozen_source_hashes"].items():
        if helper._sha(ROOT / relative) != expected:
            raise RuntimeError(f"hot free-field source changed: {relative}")
    manifest._validate_parent(require_clean=False)
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("hot free-field preflight requires a clean tracked tree")
    return {"manifest_hashes": hashes, "contract": contract}


def _states_and_coordinates() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    helper = _helper()
    arrays = helper._load_npz(
        manifest.parent.ARCLENGTH_DIRECTORY / "arclength_segment_arrays.npz"
    )
    nodes = np.asarray(arrays["nodes"], dtype=float)
    coordinates = np.asarray(arrays["coordinates"], dtype=float)
    exact_coordinates = np.asarray(arrays["exact_evaluation_coordinates470"])
    exact_states = np.asarray(arrays["exact_evaluation_primitive_states"])
    states = []
    for coordinate in coordinates:
        matches = np.flatnonzero(np.all(exact_coordinates == coordinate, axis=1))
        if len(matches) != 1:
            raise RuntimeError("final Lobatto state lacks one exact witness")
        states.append(exact_states[int(matches[0])])
    if len(nodes) != manifest.NODE_COUNT:
        raise RuntimeError("hot free-field node count changed")
    return nodes, coordinates, np.asarray(states)


def _evaluate_free_state(
    state: np.ndarray,
    *,
    model,
    configuration: dict,
    rate_source,
    exact_chart,
) -> tuple[dict, dict[str, np.ndarray]]:
    context = configuration["context"]
    columns = np.asarray(configuration["columns"], dtype=float).reshape(state.shape)
    rows = np.asarray(configuration["rows"], dtype=float).reshape(state.shape)
    began_tangent = time.perf_counter()
    tangent = rate_source.causal_five_field_monolithic_frozen_tangent(
        context,
        state,
        primitive_column_scales=columns,
        conservation_row_scales=rows,
    )
    tangent_wall = float(time.perf_counter() - began_tangent)
    began_jacobian = time.perf_counter()
    coordinate_jacobian, coordinate_metrics = exact_chart._coordinate_jacobian(
        model, state
    )
    jacobian_wall = float(time.perf_counter() - began_jacobian)
    coordinate_rate = coordinate_jacobian @ tangent.scaled_base_rate_per_s
    physical = rate_source._state_audit(context, state)
    metrics = {
        "tangent_wall_seconds": tangent_wall,
        "coordinate_jacobian_wall_seconds": jacobian_wall,
        "total_free_evaluation_wall_seconds": tangent_wall + jacobian_wall,
        "coordinate_jacobian_rank": int(coordinate_metrics["rank"]),
        "coordinate_jacobian_condition_number": float(
            coordinate_metrics["condition_number"]
        ),
        "coordinate_reconstruction_relative_defect": float(
            coordinate_metrics["coordinate_reconstruction_relative_defect"]
        ),
        "minimum_reconstruction_factor": float(
            physical["minimum_reconstruction_factor"]
        ),
        "maximum_height_ratio": float(physical["maximum_h_over_r"]),
        "minimum_scattering_optical_depth": float(
            physical["minimum_scattering_optical_depth"]
        ),
        "scaled_free_rate_norm_per_second": float(
            np.linalg.norm(tangent.scaled_base_rate_per_s)
        ),
        "coordinate_free_rate_norm_per_second": float(
            np.linalg.norm(coordinate_rate)
        ),
    }
    arrays = {
        "scaled_free_rate560_per_s": np.asarray(tangent.scaled_base_rate_per_s),
        "coordinate_jacobian470x560": coordinate_jacobian,
        "coordinate_free_rate470_per_s": coordinate_rate,
    }
    return metrics, arrays


def _select_hidden_basis(hidden_rates: np.ndarray) -> tuple[np.ndarray, int, dict]:
    training = hidden_rates[manifest.TRAINING_INDICES]
    attempts = {}
    selected = None
    selected_rank = 0
    for rank in manifest.HIDDEN_RATE_RANKS:
        basis, singular, energy = canonical_rate_basis(training, rank)
        defects = relative_projection_defects(hidden_rates, basis)
        holdout = defects[manifest.HOLDOUT_INDICES]
        attempts[str(rank)] = {
            "training_singular_values": singular.tolist(),
            "training_cumulative_energy": energy.tolist(),
            "maximum_training_defect": float(
                np.max(defects[manifest.TRAINING_INDICES])
            ),
            "maximum_holdout_defect": float(np.max(holdout)),
        }
        selected = basis
        selected_rank = rank
        if float(np.max(holdout)) <= manifest.MAXIMUM_HIDDEN_RATE_HOLDOUT_DEFECT:
            break
    if selected is None:
        raise RuntimeError("no hidden free-rate basis was constructed")
    return selected, selected_rank, attempts


def _evaluate(locked: dict) -> tuple[dict, dict[str, np.ndarray]]:
    helper = _helper()
    f5 = manifest.parent.parent
    base = f5._source()._base_inputs()
    geometry = base["geometry"]
    split = ConservativeCoordinateSplit(
        macro_restriction=geometry["R"],
        macro_lift=geometry["L"],
        hidden_dual=geometry["Q"],
        hidden_lift=geometry["Z"],
        tolerance=manifest.MAXIMUM_SPLIT_IDENTITY_DEFECT,
    )
    nodes, coordinates, states = _states_and_coordinates()
    rate_source = f5._source()._post().exact_rate.rate_source
    exact_chart = f5._exact_chart()
    records = []
    evaluated = []
    for index, state in enumerate(states):
        metrics, arrays = _evaluate_free_state(
            state,
            model=base["model"],
            configuration=base["configuration"],
            rate_source=rate_source,
            exact_chart=exact_chart,
        )
        macro_rate, hidden_rate = split.split_rate(
            arrays["coordinate_free_rate470_per_s"]
        )
        reconstructed = split.compose(macro_rate, hidden_rate)
        metrics.update({
            "node_index": index,
            "node": float(nodes[index]),
            "coordinate_decomposition_relative_defect": float(
                np.linalg.norm(
                    reconstructed - arrays["coordinate_free_rate470_per_s"]
                )
                / max(
                    float(np.linalg.norm(arrays["coordinate_free_rate470_per_s"])),
                    np.finfo(float).tiny,
                )
            ),
        })
        arrays.update({
            "macro_free_rate82_per_s": macro_rate,
            "hidden_free_rate388_per_s": hidden_rate,
        })
        records.append(metrics)
        evaluated.append(arrays)
        print(
            f"free-field node {index + 1:02d}/{manifest.NODE_COUNT}: "
            f"|r|={metrics['coordinate_free_rate_norm_per_second']:.6e}/s "
            f"wall={metrics['total_free_evaluation_wall_seconds']:.3f}s",
            flush=True,
        )

    coordinate_rates = np.stack(
        [item["coordinate_free_rate470_per_s"] for item in evaluated]
    )
    macro_rates = np.stack([item["macro_free_rate82_per_s"] for item in evaluated])
    hidden_rates = np.stack(
        [item["hidden_free_rate388_per_s"] for item in evaluated]
    )
    hidden_basis, hidden_rank, basis_attempts = _select_hidden_basis(hidden_rates)
    hidden_defects = relative_projection_defects(hidden_rates, hidden_basis)
    reduced_rates = np.hstack((macro_rates, hidden_rates @ hidden_basis))
    heldout, predictions, polynomial_defects = polynomial_holdout(
        nodes, reduced_rates, manifest.TRAINING_INDICES
    )
    architecture_arrays = helper._load_npz(
        manifest.parent.CANONICAL_DIRECTORY / "reaction_free_field_arrays.npz"
    )
    physical_basis = np.asarray(architecture_arrays["physical_rank_two_basis470x2"])
    physical_subspace_defects = relative_projection_defects(
        coordinate_rates, physical_basis
    )
    arclength_metrics = helper._read(
        manifest.parent.ARCLENGTH_DIRECTORY / "arclength_segment_metrics.json"
    )
    retraction_times = np.asarray([
        item["retraction"]["wall_seconds"]
        for item in arclength_metrics["exact_rate_metrics"]
        if item["retraction"]["wall_seconds"] > 0.0
    ])
    direct_times = np.asarray(
        [item["total_free_evaluation_wall_seconds"] for item in records]
    )
    median_retraction = float(np.median(retraction_times))
    median_direct = float(np.median(direct_times))
    projected_hours = 256.0 * (median_retraction + median_direct) / 3600.0
    maxima = {
        "maximum_split_identity_defect": max(split.identity_defects.values()),
        "maximum_coordinate_decomposition_defect": max(
            item["coordinate_decomposition_relative_defect"] for item in records
        ),
        "maximum_coordinate_jacobian_condition_number": max(
            item["coordinate_jacobian_condition_number"] for item in records
        ),
        "minimum_reconstruction_factor": min(
            item["minimum_reconstruction_factor"] for item in records
        ),
        "maximum_height_ratio": max(item["maximum_height_ratio"] for item in records),
        "minimum_scattering_optical_depth": min(
            item["minimum_scattering_optical_depth"] for item in records
        ),
        "selected_hidden_rate_rank": hidden_rank,
        "maximum_hidden_rate_training_defect": float(
            np.max(hidden_defects[manifest.TRAINING_INDICES])
        ),
        "maximum_hidden_rate_holdout_defect": float(
            np.max(hidden_defects[manifest.HOLDOUT_INDICES])
        ),
        "maximum_polynomial_holdout_defect": float(np.max(polynomial_defects)),
        "maximum_cold_physical_subspace_defect": float(
            np.max(physical_subspace_defects)
        ),
        "median_direct_free_evaluation_wall_seconds": median_direct,
        "median_historical_retraction_wall_seconds": median_retraction,
        "projected_256_witness_wall_hours": projected_hours,
        "new_exact_free_rate_calls": len(records),
        "new_fixed_Q_reaction_calls": 0,
    }
    gates = {
        "split_identities": maxima["maximum_split_identity_defect"]
        <= manifest.MAXIMUM_SPLIT_IDENTITY_DEFECT,
        "coordinate_decomposition": maxima[
            "maximum_coordinate_decomposition_defect"
        ] <= manifest.MAXIMUM_COORDINATE_DECOMPOSITION_DEFECT,
        "coordinate_jacobian_rank": all(
            item["coordinate_jacobian_rank"] == 470 for item in records
        ),
        "coordinate_jacobian_condition": maxima[
            "maximum_coordinate_jacobian_condition_number"
        ] <= manifest.MAXIMUM_COORDINATE_JACOBIAN_CONDITION_NUMBER,
        "reconstruction": maxima["minimum_reconstruction_factor"] >= 1.0 - 1.0e-12,
        "height": maxima["maximum_height_ratio"] <= 0.5,
        "optical_depth": maxima["minimum_scattering_optical_depth"] >= 1.0,
        "hidden_rate_holdout": maxima["maximum_hidden_rate_holdout_defect"]
        <= manifest.MAXIMUM_HIDDEN_RATE_HOLDOUT_DEFECT,
        "polynomial_operator_holdout": maxima[
            "maximum_polynomial_holdout_defect"
        ] <= manifest.MAXIMUM_POLYNOMIAL_HOLDOUT_DEFECT,
        "cold_physical_subspace_extension": maxima[
            "maximum_cold_physical_subspace_defect"
        ] <= manifest.MAXIMUM_COLD_PHYSICAL_SUBSPACE_DEFECT,
        "offline_cost": projected_hours
        <= manifest.MAXIMUM_PROJECTED_256_WITNESS_WALL_HOURS,
        "truth_budget": len(records) <= manifest.MAXIMUM_NEW_EXACT_FREE_RATE_CALLS,
        "reaction_forbidden": maxima["new_fixed_Q_reaction_calls"] == 0,
        "no_roots_or_microsteps": True,
    }
    passed = bool(all(gates.values()))
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION if passed else FAIL_CLASSIFICATION,
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
        "nodes": nodes,
        "coordinates5x470": coordinates,
        "primitive_states": states,
        "scaled_free_rates5x560_per_s": np.stack(
            [item["scaled_free_rate560_per_s"] for item in evaluated]
        ),
        "coordinate_free_rates5x470_per_s": coordinate_rates,
        "macro_free_rates5x82_per_s": macro_rates,
        "hidden_free_rates5x388_per_s": hidden_rates,
        "hidden_rate_basis388xr": hidden_basis,
        "hidden_rate_projection_defects": hidden_defects,
        "reduced_free_rates5x": reduced_rates,
        "holdout_indices": heldout,
        "holdout_predictions": predictions,
        "polynomial_holdout_defects": polynomial_defects,
        "cold_physical_subspace_defects": physical_subspace_defects,
    }
    return metrics, arrays


def _update_catalog(summary: dict) -> None:
    helper = _helper()
    cold = manifest.parent.parent._source()._post().manifest.transition.manifest.cold.manifest
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
        writer = csv.DictWriter(
            handle,
            fieldnames=("case", "path", "bytes", "sha256", "scientific_status"),
            lineterminator="\n",
        )
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
        raise RuntimeError("hot free-field preflight already exists")
    locked = _validate_manifest(require_clean=True)
    metrics, arrays = _evaluate(locked)
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "hot_free_field_metrics.json", metrics)
    with (CANONICAL_DIRECTORY / "hot_free_field_arrays.npz").open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": metrics["passed"],
        "truth_free_hidden_amplitude_engine_manifest_authorized": metrics["passed"],
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
        "".join(
            f"{helper._sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names
        ),
        encoding="utf-8",
    )
    values = metrics["gate_values"]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join((
            "# Hot conservative free-field ROM preflight",
            "",
            f"Classification: `{metrics['classification']}`.",
            "",
            f"Five original free-field witnesses selected hidden rank `{values['selected_hidden_rate_rank']}`. Maximum hidden-rate holdout defect: `{values['maximum_hidden_rate_holdout_defect']:.6e}`; reduced-operator polynomial holdout: `{values['maximum_polynomial_holdout_defect']:.6e}`; cold physical-subspace extension defect: `{values['maximum_cold_physical_subspace_defect']:.6e}`.",
            "",
            f"Median direct free evaluation: `{values['median_direct_free_evaluation_wall_seconds']:.3f}` s; median historical exact retraction: `{values['median_historical_retraction_wall_seconds']:.3f}` s; projected 256-witness acquisition: `{values['projected_256_witness_wall_hours']:.3f}` wall hours.",
            "",
            "The exact 82-coordinate macro ledger is retained. No fixed-Q reaction, nonlinear root, or BDF microstep was used.",
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
