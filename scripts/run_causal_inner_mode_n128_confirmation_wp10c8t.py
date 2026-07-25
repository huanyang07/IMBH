"""Run the architecture-controlling N128 confirmation for WP10c8t.

The N64 WP10c8t result leaves the complete-rate mode-0 ambiguity well above
the healing gate through 0.125 s.  This runner constructs a new exact
finite-amplitude equal-q34 pair directly on N128 from the matched complete-
rate tangent direction, then runs only that pair with the same nested
1.25e-3/6.25e-4 s BDF contract.

The package is deliberately narrow.  It does not fit a relaxation law,
change the reduced coordinates, run another nonlinear mode, or authorize a
macrostep.  Its only binding question is whether the localized N64
persistence classification is supported by an exact nonlinear N128 pair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_complete_rate_healing_wp10c8s as wp10c8s
import run_causal_extended_healing_wp10c8q as wp10c8q
import run_causal_inner_mode_healing_wp10c8t as wp10c8t
import run_causal_interface_state_sufficiency_wp10c8r as wp10c8r
import run_causal_natural_healing_wp10c8p as wp10c8p
import run_causal_nonlinear_fiber_audit_wp10c8o as wp10c8o

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    causal_five_field_loading_time,
    causal_refined_spread_upper_bound,
)


BASE_COMMIT = wp10c8t.BASE_COMMIT
WORK_PACKAGE = "WP10c8t-N128"
SCHEMA_VERSION = 1
THIS_RUNNER = (
    "scripts/run_causal_inner_mode_n128_confirmation_wp10c8t.py"
)
CASE_ID = "mode_0_inner_stress_n128_exact"
N_CELLS = 128

PARENT_JSON = wp10c8r.DEFAULT_OUTPUT
PARENT_ARRAYS = wp10c8r.DEFAULT_ARRAYS
N64_JSON = wp10c8t.DEFAULT_OUTPUT
N64_ARRAYS = wp10c8t.DEFAULT_ARRAYS
OPERATOR_N128 = (
    ROOT
    / "outputs/checkpoints/causal_five_field_wp10c8i"
    / "N128_t_0p025_moment_operators.npz"
)
CHECKPOINT_DIRECTORY = (
    ROOT / "outputs/checkpoints/causal_five_field_wp10c8t_n128"
)
PAIR_JSON = CHECKPOINT_DIRECTORY / "mode_0_exact_pair.json"
PAIR_ARRAYS = CHECKPOINT_DIRECTORY / "mode_0_exact_pair_arrays.npz"
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/tables/"
    "causal_inner_mode_n128_confirmation_wp10c8t.json"
)
DEFAULT_ARRAYS = (
    ROOT
    / "outputs/tables/"
    "causal_inner_mode_n128_confirmation_wp10c8t_arrays.npz"
)

OUTPUT_OFFSETS_SECONDS = (0.0, wp10c8t.TARGET_DURATION_SECONDS)
MAXIMUM_FINAL_ENDPOINT_TEMPORAL_UNCERTAINTY = 0.10
MINIMUM_PERSISTENT_LOWER_BOUND = 0.25
MINIMUM_CROSS_MESH_RATE_COSINE = 0.90
MAXIMUM_CROSS_MESH_AMPLITUDE_RATIO_DEFECT = 0.50
MINIMUM_LOCALIZED_SHELL_FRACTION = wp10c8s.LOCALIZATION_FRACTION_GATE


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    return wp10c8t._array_sha256(values)


def _plain(value):
    return wp10c8t._plain(value)


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _pair_expected(seed_direction: np.ndarray) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "base_commit": BASE_COMMIT,
        "case_id": CASE_ID,
        "n_cells": N_CELLS,
        "mode_index": 0,
        "seed_multiplier": wp10c8s.SEED_MULTIPLIER,
        "seed_sha256": _array_sha256(seed_direction),
        "parent_json_sha256": _sha256(PARENT_JSON),
        "parent_arrays_sha256": _sha256(PARENT_ARRAYS),
        "operator_sha256": _sha256(OPERATOR_N128),
    }


def _pair_binding(row: dict) -> dict:
    slow_rate_maximum = float(
        row["slow_rate_audit"][
            "maximum_absolute_half_difference_per_unit_slow_time"
        ]
    )
    binding = {
        "lift_valid": bool(row["lift_valid"]),
        "fresh_rate_output_evaluated": bool(
            row["fresh_rate_output_evaluated"]
        ),
        "all_fresh_rate_gates_passed": bool(
            row["full_output"]["all_fresh_rate_gates_passed"]
        ),
        "all_binding_dae_storage_audits_passed": bool(
            row["full_output"][
                "all_binding_dae_storage_audits_passed"
            ]
        ),
        "slow_rate_counterexample": bool(
            slow_rate_maximum > wp10c8t.NONLINEAR_SIGNIFICANCE_GATE
        ),
        "windowed_full_output_counterexample": bool(
            row["full_output"]["counterexample"]
        ),
        "maximum_slow_rate_half_spread": slow_rate_maximum,
    }
    binding["passed"] = bool(
        binding["lift_valid"]
        and binding["fresh_rate_output_evaluated"]
        and binding["all_fresh_rate_gates_passed"]
        and binding["all_binding_dae_storage_audits_passed"]
        and binding["slow_rate_counterexample"]
    )
    return binding


def _build_or_load_pair(
    *,
    seed_direction: np.ndarray,
    initial_by_mesh: dict,
    vectors_by_mesh: dict,
    contract: dict,
    operator_arrays: dict[str, np.ndarray],
    force: bool,
) -> tuple[dict, dict[str, np.ndarray]]:
    expected = _pair_expected(seed_direction)
    if PAIR_JSON.exists() and PAIR_ARRAYS.exists() and not force:
        payload = json.loads(PAIR_JSON.read_text(encoding="utf-8"))
        if not (
            all(payload.get(key) == value for key, value in expected.items())
            and payload.get("arrays_sha256") == _sha256(PAIR_ARRAYS)
        ):
            raise RuntimeError("stale WP10c8t N128 pair cache")
        row = payload["row"]
        binding = _pair_binding(row)
        if row.get("n128_pair_binding") != binding:
            row["n128_pair_binding"] = binding
            payload["row"] = row
            PAIR_JSON.write_text(
                json.dumps(
                    payload,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
        return row, _load_npz(PAIR_ARRAYS)

    started = time.perf_counter()
    row, _pair_arrays, runtime = wp10c8o._build_pair(
        case_id=CASE_ID,
        seed_name="wp10c8r_complete_rate_mode_0_n128",
        seed_origin=(
            "matched N128 significance-gated complete-rate singular "
            "direction 0 at t=0.025 s"
        ),
        seed_direction=seed_direction,
        seed_multiplier=wp10c8s.SEED_MULTIPLIER,
        initial=initial_by_mesh[N_CELLS],
        vector=vectors_by_mesh[N_CELLS][wp10c8o.PRIMARY_ANCHOR],
        cache=operator_arrays,
        shell_edges_rg=contract["shell_edges_rg"],
        require_face58_switch=False,
    )
    wp10c8o._complete_pair_rates(
        row,
        runtime,
        binding_dae_storage_audit=True,
    )
    loading_time = causal_five_field_loading_time(
        contract["context"],
        contract["anchor_vector"],
    )
    wp10c8q._actual_slow_rate_row(row, runtime, loading_time)
    row["mode_index"] = 0
    row["family"] = "inner_stress"
    row["n_cells"] = N_CELLS
    row["total_pair_wall_seconds"] = time.perf_counter() - started
    row["n128_pair_binding"] = _pair_binding(row)

    arrays = {
        name: np.asarray(value)
        for name, value in runtime["arrays"].items()
    }
    PAIR_ARRAYS.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PAIR_ARRAYS, **arrays)
    payload = {
        **expected,
        "row": _plain(row),
        "arrays_path": _relative(PAIR_ARRAYS),
        "arrays_sha256": _sha256(PAIR_ARRAYS),
    }
    PAIR_JSON.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return row, arrays


def _case_from_pair(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    required = (
        "minus_state_vector",
        "plus_state_vector",
        "coordinate_names",
        "coordinate_scales",
        "interface_flux_scales",
    )
    if not all(name in arrays for name in required):
        raise RuntimeError("exact N128 pair arrays are incomplete")
    return {name: np.asarray(arrays[name]) for name in required}


def _load_contract(
    *,
    force_pair: bool,
) -> tuple[dict, dict[str, np.ndarray], dict, dict]:
    required = (
        PARENT_JSON,
        PARENT_ARRAYS,
        N64_JSON,
        N64_ARRAYS,
        OPERATOR_N128,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"required WP10c8t input is missing: {path}")
    parent = json.loads(PARENT_JSON.read_text(encoding="utf-8"))
    n64 = json.loads(N64_JSON.read_text(encoding="utf-8"))
    if not (
        parent.get("work_package") == "WP10c8r"
        and parent.get("artifacts", {}).get("arrays_sha256")
        == _sha256(PARENT_ARRAYS)
        and n64.get("work_package") == "WP10c8t"
        and n64.get("artifacts", {}).get("arrays_sha256")
        == _sha256(N64_ARRAYS)
        and n64.get("decision")
        == "n64_persistent_localized_inner_mode_through_0p125s"
    ):
        raise RuntimeError("WP10c8t N128 parent provenance failed")

    parent_arrays = _load_npz(PARENT_ARRAYS)
    tested = np.asarray(
        parent_arrays[
            "n128_t_0p025_top_tested_state_directions"
        ],
        dtype=float,
    )
    seed_direction = tested[0] / wp10c8r.AUDIT_SEED_MULTIPLIER
    initial_by_mesh, vectors_by_mesh, state_provenance, contracts = (
        wp10c8q._runtime_contracts()
    )
    contract = dict(contracts[N_CELLS])
    contract["checkpoint_directory"] = CHECKPOINT_DIRECTORY
    contract["bitwise_parent_replay_required"] = False
    operator_arrays, operator_metadata = wp10c8r._load_operator_cache(
        OPERATOR_N128
    )
    row, pair_arrays = _build_or_load_pair(
        seed_direction=seed_direction,
        initial_by_mesh=initial_by_mesh,
        vectors_by_mesh=vectors_by_mesh,
        contract=contract,
        operator_arrays=operator_arrays,
        force=force_pair,
    )
    return (
        contract,
        _case_from_pair(pair_arrays),
        {
            "row": row,
            "arrays": pair_arrays,
            "seed_direction": seed_direction,
            "state_provenance": _plain(state_provenance),
        },
        {
            "arrays": operator_arrays,
            "metadata": operator_metadata,
        },
    )


def _signed_direction_comparison(
    left: np.ndarray,
    right: np.ndarray,
) -> dict:
    first = np.asarray(left, dtype=float)
    second = np.asarray(right, dtype=float)
    denominator = max(
        float(np.linalg.norm(first) * np.linalg.norm(second)),
        np.finfo(float).tiny,
    )
    cosine = float(np.dot(first, second) / denominator)
    first_max = float(np.max(np.abs(first)))
    second_max = float(np.max(np.abs(second)))
    ratio = second_max / max(first_max, np.finfo(float).tiny)
    return {
        "signed_cosine": cosine,
        "absolute_cosine": abs(cosine),
        "n64_maximum": first_max,
        "n128_maximum": second_max,
        "n128_to_n64_maximum_ratio": ratio,
        "amplitude_ratio_defect": abs(ratio - 1.0),
        "passed": bool(
            abs(cosine) >= MINIMUM_CROSS_MESH_RATE_COSINE
            and abs(ratio - 1.0)
            <= MAXIMUM_CROSS_MESH_AMPLITUDE_RATIO_DEFECT
        ),
    }


def _n128_decision(
    *,
    pair_arrays: dict[str, dict[str, np.ndarray]],
    localizations: dict[str, dict],
    all_contracts_passed: bool,
) -> tuple[dict, dict[str, np.ndarray]]:
    coarse = pair_arrays["coarse"]
    fine = pair_arrays["fine"]
    if not (
        np.array_equal(coarse["times"], fine["times"])
        and np.array_equal(coarse["full_names"], fine["full_names"])
    ):
        raise RuntimeError("N128 coarse/fine diagnostic schemas differ")
    uncertainty, upper = causal_refined_spread_upper_bound(
        coarse["full_spreads"],
        fine["full_spreads"],
    )
    lower = np.maximum(fine["full_spreads"] - uncertainty, 0.0)
    significant = upper[0] >= wp10c8t.NONLINEAR_SIGNIFICANCE_GATE
    final_uncertainty = float(
        np.max(uncertainty[-1, significant])
        if np.any(significant)
        else np.inf
    )
    endpoint_temporal_passed = bool(
        final_uncertainty
        <= MAXIMUM_FINAL_ENDPOINT_TEMPORAL_UNCERTAINTY
    )
    persistent = bool(
        np.any(significant)
        and np.any(
            lower[-1, significant] >= MINIMUM_PERSISTENT_LOWER_BOUND
        )
    )
    healed = bool(
        np.any(significant)
        and np.all(
            upper[-1, significant]
            <= wp10c8t.HEALING_FINAL_SPREAD_GATE
        )
    )

    n64_arrays = _load_npz(N64_ARRAYS)
    n64_coordinate_names = np.asarray(
        n64_arrays["fine_minus_coordinate_names"]
    )
    if not np.array_equal(
        n64_coordinate_names,
        fine["coordinate_names"],
    ):
        raise RuntimeError("N64/N128 coordinate-rate schemas differ")
    initial_direction = _signed_direction_comparison(
        n64_arrays["fine_pair_signed_slow_rate_half_difference"][0],
        fine["signed_slow_rate_half_difference"][0],
    )
    final_direction = _signed_direction_comparison(
        n64_arrays["fine_pair_signed_slow_rate_half_difference"][-1],
        fine["signed_slow_rate_half_difference"][-1],
    )
    localization_passed = bool(
        localizations["fine"]["final_state_controlling_shell"] == 0
        and localizations["fine"]["final_rate_controlling_shell"] == 0
        and localizations["fine"]["state_support"][-1][
            "controlling_shell_l1_fraction"
        ]
        >= MINIMUM_LOCALIZED_SHELL_FRACTION
        and localizations["fine"]["primitive_rate_support"][-1][
            "controlling_shell_l1_fraction"
        ]
        >= MINIMUM_LOCALIZED_SHELL_FRACTION
    )
    cross_mesh_passed = bool(
        initial_direction["passed"] and final_direction["passed"]
    )
    if (
        all_contracts_passed
        and endpoint_temporal_passed
        and persistent
        and localization_passed
        and cross_mesh_passed
    ):
        classification = (
            "mesh_supported_persistent_localized_inner_mode_through_0p125s"
        )
    elif (
        all_contracts_passed
        and endpoint_temporal_passed
        and healed
    ):
        classification = "n64_n128_healing_classification_disagrees"
    else:
        classification = "n128_architecture_confirmation_inconclusive"
    return {
        "classification": classification,
        "all_n128_contracts_passed": all_contracts_passed,
        "maximum_final_endpoint_temporal_uncertainty": final_uncertainty,
        "maximum_final_endpoint_temporal_uncertainty_gate": (
            MAXIMUM_FINAL_ENDPOINT_TEMPORAL_UNCERTAINTY
        ),
        "endpoint_temporal_control_passed": endpoint_temporal_passed,
        "initial_maximum_uncertainty_inclusive_spread": float(
            np.max(upper[0])
        ),
        "final_maximum_uncertainty_inclusive_spread": float(
            np.max(upper[-1])
        ),
        "final_maximum_uncertainty_exclusive_lower_spread": float(
            np.max(lower[-1])
        ),
        "minimum_persistent_lower_bound": (
            MINIMUM_PERSISTENT_LOWER_BOUND
        ),
        "persistent_lower_bound_passed": persistent,
        "final_healing_gate_passed": healed,
        "initial_cross_mesh_rate_direction": initial_direction,
        "final_cross_mesh_rate_direction": final_direction,
        "cross_mesh_rate_direction_passed": cross_mesh_passed,
        "final_localization_passed": localization_passed,
        "architecture_confirmation_passed": bool(
            classification
            == (
                "mesh_supported_persistent_localized_inner_mode_"
                "through_0p125s"
            )
        ),
    }, {
        "temporal_uncertainty": uncertainty,
        "uncertainty_inclusive_spreads": upper,
        "uncertainty_exclusive_lower_spreads": lower,
        "significant_initial_output_mask": significant,
    }


def _run_diagnostics(
    *,
    contract: dict,
    case: dict[str, np.ndarray],
    operator_arrays: dict[str, np.ndarray],
    trajectories: dict[str, dict],
    compute_fresh_rates: bool,
) -> tuple[dict, dict[str, np.ndarray]]:
    diagnostics = {}
    all_arrays: dict[str, np.ndarray] = {}
    rate_cache: dict[
        str,
        tuple[np.ndarray, dict, dict[str, np.ndarray]],
    ] = {}
    for resolution in ("coarse", "fine"):
        for side in ("minus", "plus"):
            label = f"{resolution}_{side}"
            summary, arrays = wp10c8t._trajectory_diagnostics_with_rates(
                contract=contract,
                case=case,
                operator_arrays=operator_arrays,
                states=trajectories[label]["states"],
                subdivisions=wp10c8t.TOTAL_SUBDIVISIONS[resolution],
                rate_cache=rate_cache,
                compute_fresh_rates=compute_fresh_rates,
                duration_seconds=wp10c8t.TARGET_DURATION_SECONDS,
                output_offsets_seconds=OUTPUT_OFFSETS_SECONDS,
            )
            diagnostics[label] = summary
            all_arrays.update(
                {
                    f"{label}_{name}": values
                    for name, values in arrays.items()
                }
            )

    loading_time = causal_five_field_loading_time(
        contract["context"],
        contract["anchor_vector"],
    )
    pair_arrays = {}
    pair_ledgers = {}
    localizations = {}
    for resolution in ("coarse", "fine"):
        minus = {
            name.removeprefix(f"{resolution}_minus_"): values
            for name, values in all_arrays.items()
            if name.startswith(f"{resolution}_minus_")
        }
        plus = {
            name.removeprefix(f"{resolution}_plus_"): values
            for name, values in all_arrays.items()
            if name.startswith(f"{resolution}_plus_")
        }
        pair = wp10c8t._pair_arrays(
            minus=minus,
            plus=plus,
            coordinate_scales=np.asarray(
                case["coordinate_scales"], dtype=float
            ),
            loading_time_seconds=loading_time,
        )
        if not np.array_equal(
            minus["coordinate_names"],
            plus["coordinate_names"],
        ):
            raise RuntimeError(
                f"N128 {resolution} plus/minus coordinate schemas differ"
            )
        pair["coordinate_names"] = np.asarray(minus["coordinate_names"])
        ledger_summary, ledger_arrays = wp10c8p._pair_diagnostics(
            minus=minus,
            plus=plus,
            coordinate_scales=np.asarray(
                case["coordinate_scales"], dtype=float
            ),
            coordinate_names=tuple(
                str(value) for value in case["coordinate_names"]
            ),
        )
        localization_summary, localization_arrays = wp10c8t._localization(
            context=contract["context"],
            minus=minus,
            plus=plus,
            operator_arrays=operator_arrays,
        )
        pair_arrays[resolution] = pair
        pair_ledgers[resolution] = ledger_summary
        localizations[resolution] = localization_summary
        all_arrays.update(
            {
                f"{resolution}_pair_{name}": values
                for name, values in pair.items()
            }
        )
        all_arrays.update(
            {
                f"{resolution}_ledger_{name}": values
                for name, values in ledger_arrays.items()
            }
        )
        all_arrays.update(
            {
                f"{resolution}_localization_{name}": values
                for name, values in localization_arrays.items()
            }
        )

    trajectory_contracts = bool(
        all(row["summary"]["passed"] for row in trajectories.values())
    )
    diagnostic_contracts = bool(
        compute_fresh_rates
        and all(
            row["maximum_physical_mje_shell_ledger_relative_defect"]
            <= wp10c8t.MAXIMUM_SHELL_LEDGER_RELATIVE_DEFECT
            and row["maximum_flux_reconstruction_defect"]
            <= wp10c8p.MAXIMUM_FLUX_RECONSTRUCTION_DEFECT
            and row["all_output_state_gates_passed"]
            and row["all_fresh_rate_audits_passed"]
            for row in diagnostics.values()
        )
    )
    decision, decision_arrays = _n128_decision(
        pair_arrays=pair_arrays,
        localizations=localizations,
        all_contracts_passed=bool(
            trajectory_contracts and diagnostic_contracts
        ),
    )
    all_arrays.update(
        {
            f"decision_{name}": values
            for name, values in decision_arrays.items()
        }
    )
    return {
        "loading_time_seconds": loading_time,
        "trajectory_contracts_passed": trajectory_contracts,
        "diagnostic_contracts_passed": diagnostic_contracts,
        "trajectory_diagnostics": diagnostics,
        "pair_ledgers": pair_ledgers,
        "localization": localizations,
        "decision": decision,
    }, all_arrays


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair-only",
        action="store_true",
        help="Construct and verify only the exact nonlinear N128 pair.",
    )
    parser.add_argument(
        "--trajectory-only",
        choices=(
            "coarse-minus",
            "coarse-plus",
            "fine-minus",
            "fine-plus",
        ),
        default=None,
        help="Populate exactly one N128 trajectory cache and exit.",
    )
    parser.add_argument(
        "--skip-fresh-rates",
        action="store_true",
        help="Development-only: assemble without binding fresh rates.",
    )
    parser.add_argument(
        "--force-pair",
        action="store_true",
        help="Rebuild the exact nonlinear N128 pair.",
    )
    parser.add_argument(
        "--force-trajectory",
        action="store_true",
        help="Recompute the selected N128 trajectory caches.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--arrays", type=Path, default=DEFAULT_ARRAYS)
    return parser.parse_args()


def main() -> None:
    wp10c8t._validate_schedule()
    args = _arguments()
    contract, case, pair, operator = _load_contract(
        force_pair=args.force_pair
    )
    if args.pair_only:
        print(
            json.dumps(
                _plain(
                    {
                        "work_package": WORK_PACKAGE,
                        "pair_path": _relative(PAIR_JSON),
                        "pair_arrays_path": _relative(PAIR_ARRAYS),
                        "pair_arrays_sha256": _sha256(PAIR_ARRAYS),
                        "binding": pair["row"]["n128_pair_binding"],
                        "maximum_slow_rate_half_spread": pair["row"][
                            "n128_pair_binding"
                        ]["maximum_slow_rate_half_spread"],
                    }
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return

    if not pair["row"]["n128_pair_binding"]["passed"]:
        raise RuntimeError(
            "exact N128 mode-0 pair failed its binding gates: "
            + json.dumps(
                pair["row"]["n128_pair_binding"],
                sort_keys=True,
            )
        )

    if args.trajectory_only is not None:
        resolution, side = args.trajectory_only.split("-", maxsplit=1)
        trajectory = wp10c8t._run_or_load_trajectory(
            contract=contract,
            case=case,
            resolution=resolution,
            side=side,
            force=args.force_trajectory,
        )
        print(
            json.dumps(
                _plain(
                    {
                        "work_package": WORK_PACKAGE,
                        "trajectory_only": args.trajectory_only,
                        "path": _relative(trajectory["path"]),
                        "sha256": trajectory["sha256"],
                        "final_restart": (
                            trajectory["final_restart_evidence"]
                        ),
                        "summary": trajectory["summary"],
                        "cached": trajectory["cached"],
                    }
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return

    trajectories = {}
    for resolution in ("coarse", "fine"):
        for side in ("minus", "plus"):
            label = f"{resolution}_{side}"
            trajectories[label] = wp10c8t._run_or_load_trajectory(
                contract=contract,
                case=case,
                resolution=resolution,
                side=side,
                force=args.force_trajectory,
            )
    diagnostics, arrays = _run_diagnostics(
        contract=contract,
        case=case,
        operator_arrays=operator["arrays"],
        trajectories=trajectories,
        compute_fresh_rates=not args.skip_fresh_rates,
    )
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    arrays_path = args.arrays if args.arrays.is_absolute() else ROOT / args.arrays
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **arrays)
    source_paths = (
        ROOT / THIS_RUNNER,
        ROOT / wp10c8t.THIS_RUNNER,
        ROOT / "scripts/run_causal_complete_rate_healing_wp10c8s.py",
        ROOT / "scripts/run_causal_extended_healing_wp10c8q.py",
        ROOT / "scripts/run_causal_nonlinear_fiber_audit_wp10c8o.py",
        ROOT
        / "src/imri_qpe/layer3_minidisk_1d/causal_inner_bdf_restart.py",
        ROOT
        / "src/imri_qpe/layer3_minidisk_1d/"
        "causal_inner_bdf_evolution.py",
    )
    output = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "base_commit": BASE_COMMIT,
        "case_id": CASE_ID,
        "scope": {
            "n_cells": N_CELLS,
            "production_physics_changed": False,
            "production_flux_changed": False,
            "production_descriptor_changed": False,
            "production_time_integrator_changed": False,
            "moment_ladder_changed": False,
            "reduced_evolution_run": False,
            "relaxation_law_fit": False,
            "only_architecture_controlling_case_run": True,
        },
        "contract": {
            "target_duration_seconds": wp10c8t.TARGET_DURATION_SECONDS,
            "output_offsets_seconds": OUTPUT_OFFSETS_SECONDS,
            "timestep_seconds": wp10c8t.TIMESTEP_SECONDS,
            "total_subdivisions": wp10c8t.TOTAL_SUBDIVISIONS,
        },
        "gates": {
            "maximum_final_endpoint_temporal_uncertainty": (
                MAXIMUM_FINAL_ENDPOINT_TEMPORAL_UNCERTAINTY
            ),
            "minimum_persistent_lower_bound": (
                MINIMUM_PERSISTENT_LOWER_BOUND
            ),
            "minimum_cross_mesh_rate_cosine": (
                MINIMUM_CROSS_MESH_RATE_COSINE
            ),
            "maximum_cross_mesh_amplitude_ratio_defect": (
                MAXIMUM_CROSS_MESH_AMPLITUDE_RATIO_DEFECT
            ),
            "minimum_localized_shell_fraction": (
                MINIMUM_LOCALIZED_SHELL_FRACTION
            ),
            "maximum_shell_ledger_relative_defect": (
                wp10c8t.MAXIMUM_SHELL_LEDGER_RELATIVE_DEFECT
            ),
        },
        "authorization": {
            "n64_json_path": _relative(N64_JSON),
            "n64_json_sha256": _sha256(N64_JSON),
            "n64_arrays_path": _relative(N64_ARRAYS),
            "n64_arrays_sha256": _sha256(N64_ARRAYS),
            "pair_json_path": _relative(PAIR_JSON),
            "pair_json_sha256": _sha256(PAIR_JSON),
            "pair_arrays_path": _relative(PAIR_ARRAYS),
            "pair_arrays_sha256": _sha256(PAIR_ARRAYS),
        },
        "pair": pair["row"],
        "trajectory_provenance": {
            label: {
                "path": _relative(row["path"]),
                "sha256": row["sha256"],
                "initial_history": row["initial_history"],
                "final_restart": row["final_restart_evidence"],
                "segments": row["segments"],
                "cached": row["cached"],
                "summary": row["summary"],
            }
            for label, row in trajectories.items()
        },
        "diagnostics": diagnostics,
        "decision": diagnostics["decision"]["classification"],
        "next_action": (
            "run_local_inner_state_rank_and_physical_attribution_audit"
            if diagnostics["decision"]["architecture_confirmation_passed"]
            else "repair_only_the_failed_n128_confirmation_gate"
        ),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "source_hashes": {
            _relative(path): _sha256(path) for path in source_paths
        },
        "artifacts": {
            "arrays_path": _relative(arrays_path),
            "arrays_sha256": _sha256(arrays_path),
        },
    }
    output_path.write_text(
        json.dumps(_plain(output), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "work_package": WORK_PACKAGE,
                "decision": output["decision"],
                "next_action": output["next_action"],
                "output": _relative(output_path),
                "arrays": _relative(arrays_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
