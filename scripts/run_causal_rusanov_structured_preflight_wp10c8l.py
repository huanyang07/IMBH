"""Run the nonbinding structured Rusanov feasibility preflight for WP10c8l-B.

Track A did not certify a final nominal generator, so this runner deliberately
uses the immutable WP10c8j parent operators and labels every result
nonbinding.  It tests only the richest storage-consistent coordinate level
and only the cached consequential branches.  A failure is a hard stop before
the complete all-face possible-winner set or any finite-neighborhood work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import run_causal_moment_sufficiency_audit_wp10c8i as wp10c8i
from imri_qpe.layer3_minidisk_1d.causal_inner_rusanov_certification import (
    rusanov_structured_zero_remainder_preflight,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_mixed_reduction import (
    causal_weighted_constraint_null_basis,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "4dc5cea0342d35135e31078669e7e71ba7d16cf9"
WORK_PACKAGE = "WP10c8l-B"
TRACK_A_RESULT = ROOT / "outputs/tables/causal_tangent_descriptor_wp10c8l.json"
PARENT_DIRECTORY = (
    ROOT
    / "outputs/checkpoints/causal_five_field_wp10c8j/operator_sources"
)
LOCKED_CASES = (
    ("n64_t_0", PARENT_DIRECTORY / "wp10c8j_N064_t_0_moment_operators.npz"),
    (
        "n64_t_0p025",
        PARENT_DIRECTORY / "wp10c8j_N064_t_0p025_moment_operators.npz",
    ),
)
LOCKED_HORIZONS_SECONDS = (1.0e-2, 2.5e-2)
LOCKED_TIME_PANELS = (64, 128)
LEVEL_INDEX = 4
ALLOWED_MAXIMUM_GATE_FRACTION = 1.0e-2
DEFAULT_OUTPUT = (
    ROOT / "outputs/tables/causal_rusanov_structured_preflight_wp10c8l.json"
)
DEFAULT_ARRAYS = (
    ROOT
    / "outputs/tables/causal_rusanov_structured_preflight_wp10c8l_arrays.npz"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--arrays", type=Path, default=DEFAULT_ARRAYS)
    return parser.parse_args()


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_operator(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: np.array(source[name], copy=True) for name in source.files}
    metadata = json.loads(str(arrays.pop("metadata_json").item()))
    return arrays, metadata


def _preflight_case(
    case_id: str,
    path: Path,
) -> tuple[dict, dict[str, np.ndarray]]:
    arrays, metadata = _load_operator(path)
    response, gates, names, blocks = wp10c8i._response_stack(
        arrays,
        metadata,
        LEVEL_INDEX,
    )
    constraints = np.asarray(arrays[f"level_{LEVEL_INDEX}_constraints"])
    state_weights = np.asarray(arrays["state_weights"])
    basis_audit = causal_weighted_constraint_null_basis(
        constraints,
        state_weights=state_weights,
    )
    basis = np.asarray(basis_audit.basis, dtype=float)
    left = np.asarray(
        arrays["production_rusanov_kink_generator_left_factors"],
        dtype=float,
    )
    right = np.asarray(
        arrays["production_rusanov_kink_generator_right_factors"],
        dtype=float,
    )
    faces = np.asarray(
        arrays["production_rusanov_kink_face_indices"],
        dtype=int,
    )
    direct = wp10c8i._rusanov_kink_instantaneous_output_deltas(
        arrays,
        metadata,
        LEVEL_INDEX,
    )

    rows: dict[str, dict] = {}
    output_arrays: dict[str, np.ndarray] = {
        f"{case_id}_constraint_null_basis": basis,
        f"{case_id}_generator_left_factors": left,
        f"{case_id}_generator_right_factors": right,
        f"{case_id}_branch_face_indices": faces,
    }
    for horizon in LOCKED_HORIZONS_SECONDS:
        horizon_key = f"{horizon:.3e}"
        rows[horizon_key] = {}
        for panels in LOCKED_TIME_PANELS:
            result = rusanov_structured_zero_remainder_preflight(
                base_generator_per_s=np.asarray(arrays["dynamic"]),
                output_operator=response,
                generator_left_factors=left,
                generator_right_factors=right,
                branch_face_indices=faces,
                initial_basis=basis,
                horizon_seconds=horizon,
                output_gates=gates,
                direct_output_deltas=direct,
                time_steps=panels,
                maximum_gate_fraction=ALLOWED_MAXIMUM_GATE_FRACTION,
            )
            row = result.as_dict()
            fractions = np.asarray(result.per_output_gate_fractions)
            control = int(np.argmax(fractions))
            row["controlling_output"] = names[control]
            row["controlling_output_index"] = control
            row["controlling_output_gate_fraction"] = float(fractions[control])
            row.pop("per_output_dynamic_bounds")
            row.pop("per_output_direct_bounds")
            row.pop("per_output_total_bounds")
            row.pop("per_output_gate_fractions")
            rows[horizon_key][str(panels)] = row
            prefix = f"{case_id}_h_{horizon_key}_p_{panels}"
            output_arrays[f"{prefix}_dynamic_bounds"] = (
                result.per_output_dynamic_bounds
            )
            output_arrays[f"{prefix}_direct_bounds"] = (
                result.per_output_direct_bounds
            )
            output_arrays[f"{prefix}_gate_fractions"] = fractions

    convergence: dict[str, dict] = {}
    case_feasible = True
    for horizon_key, panel_rows in rows.items():
        coarse = float(panel_rows["64"]["maximum_gate_fraction"])
        fine = float(panel_rows["128"]["maximum_gate_fraction"])
        scale = max(abs(fine), np.finfo(float).tiny)
        convergence[horizon_key] = {
            "coarse_panels": 64,
            "fine_panels": 128,
            "coarse_maximum_gate_fraction": coarse,
            "fine_maximum_gate_fraction": fine,
            "relative_panel_change": abs(fine - coarse) / scale,
            "fine_within_locked_reserve": bool(
                fine <= ALLOWED_MAXIMUM_GATE_FRACTION
            ),
        }
        case_feasible = bool(
            case_feasible
            and fine <= ALLOWED_MAXIMUM_GATE_FRACTION
        )

    return (
        {
            "source_path": _relative(path),
            "source_sha256": _sha256(path),
            "nominal_generator_source": "immutable_wp10c8j_parent",
            "candidate_scope": "cached_consequential_branches_only",
            "coordinate_level_index": LEVEL_INDEX,
            "coordinate_count": int(constraints.shape[0]),
            "constraint_rank": int(basis_audit.constraint_rank),
            "constraint_null_dimension": int(basis.shape[1]),
            "branch_count": int(left.shape[1]),
            "face_count": int(np.unique(faces).size),
            "output_blocks": blocks,
            "rows": rows,
            "time_panel_convergence": convergence,
            "feasible_under_locked_preflight": case_feasible,
        },
        output_arrays,
    )


def main() -> None:
    arguments = _arguments()
    output_path = _absolute(arguments.output)
    arrays_path = _absolute(arguments.arrays)
    if not TRACK_A_RESULT.exists():
        raise FileNotFoundError(TRACK_A_RESULT)
    track_a = json.loads(TRACK_A_RESULT.read_text())
    if track_a.get("decision") != "wp10c8l_a_locked_n64_failed":
        raise RuntimeError(
            "WP10c8l-B runner is locked to the failed Track-A decision"
        )

    case_rows: dict[str, dict] = {}
    output_arrays: dict[str, np.ndarray] = {}
    for case_id, path in LOCKED_CASES:
        row, arrays = _preflight_case(case_id, path)
        case_rows[case_id] = row
        output_arrays.update(arrays)

    cached_scope_feasible = bool(
        all(row["feasible_under_locked_preflight"] for row in case_rows.values())
    )
    decision = (
        "wp10c8l_b_cached_scope_feasible_nonbinding"
        if cached_scope_feasible
        else "wp10c8l_b_cached_scope_infeasible_nonbinding"
    )
    output = {
        "base_commit": BASE_COMMIT,
        "work_package": WORK_PACKAGE,
        "scope": {
            "binding": False,
            "zero_nonlinear_remainder": True,
            "nominal_generator_is_final_track_a_generator": False,
            "weighted_constraint_null_initial_space": True,
            "direct_output_deltas_included": True,
            "per_face_mutual_exclusivity": True,
            "simultaneous_switching_across_faces": True,
            "candidate_scope": "cached_consequential_branches_only",
            "complete_all_face_possible_winner_scope_run": False,
            "finite_neighborhood_contract_run": False,
        },
        "gates": {
            "allowed_maximum_gate_fraction": (
                ALLOWED_MAXIMUM_GATE_FRACTION
            ),
            "locked_horizons_seconds": list(LOCKED_HORIZONS_SECONDS),
            "locked_time_panels": list(LOCKED_TIME_PANELS),
        },
        "track_a": {
            "path": _relative(TRACK_A_RESULT),
            "sha256": _sha256(TRACK_A_RESULT),
            "decision": track_a["decision"],
        },
        "cases": case_rows,
        "decision": decision,
        "next_action": (
            "stop_before_all_face_and_finite_neighborhood_work"
            if not cached_scope_feasible
            else "build_complete_all_face_possible_winner_preflight_only_after_track_a_passes"
        ),
        "semantics": (
            "This is a nonbinding zero-remainder structured feasibility "
            "preflight.  Track A failed, so the nominal semigroup and "
            "generator-level branch factors are parent evidence and must be "
            "recomputed after any future Track-A certification."
        ),
    }

    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **output_arrays)
    output["artifacts"] = {
        "arrays_path": _relative(arrays_path),
        "arrays_sha256": _sha256(arrays_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
