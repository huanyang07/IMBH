"""Staged branch-locked regular-domain refinement for the sonic-buffer root."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix, vstack

from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot

from run_transonic_outer_slope_calibration_audit import fmt
from run_transonic_two_domain_mesh_validation import combined_profile_arrays, load_checkpoint, make_params
from run_transonic_two_domain_sonic_refinement_sprint import (
    CHECKPOINT_DIR as SPRINT_CHECKPOINT_DIR,
    SOURCE_CHECKPOINT,
    attach_stage,
    buffer_audit,
    buffer_inner_grid,
    buffer_residual,
    buffer_row_with_reference,
    buffer_sparsity,
    interval_residual_between,
    make_buffer_params,
    pack_two_domain,
    pchip_extrap,
    row_json,
    solve_buffer,
    source_first_slope,
    state_bounds_two_domain,
    unpack_buffer,
)
from run_transonic_two_domain_outer_extension import outer_grid


ROOT = Path(__file__).resolve().parents[1]
TABLE_OUTPUT = ROOT / "outputs" / "tables" / "transonic_two_domain_staged_regular_refinement.md"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "transonic_two_domain_staged_regular_refinement"
BEST_BUFFER_CHECKPOINT = SPRINT_CHECKPOINT_DIR / "delta0p02_Nreg64_defect_preserving_0p90277664.npz"
N_SEQUENCE = (80, 96, 112, 128)
SCIENCE_LIMIT = 5.0e-6
STOP_LIMIT = 5.0e-3
FALLBACK_TRIGGER = 5.0e-5
CONTINUE_LIMIT = 1.0e-4
MAX_NFEV_CHAIN_SPLIT = 120


@dataclass(frozen=True)
class StageSpec:
    name: str
    family: str
    max_nfev: int
    state_half_width: float
    rson_half_rg: float
    lambda_half: float
    lock_weight: float = 0.0
    sigma_rson_rg: float = 0.03
    sigma_lambda: float = 1.0e-3


STAGES = (
    StageSpec("A_new_nodes", "new", 220, 2.0, 1.0e-6, 1.0e-8),
    StageSpec("B_neighbors", "neighbors", 260, 1.5, 1.0e-6, 1.0e-8),
    StageSpec("C_inner_fixed_eigen", "inner_regular", 320, 1.0, 1.0e-6, 1.0e-8),
    StageSpec("D_inner_eigen", "inner_eigen", 360, 0.75, 0.035, 1.2e-3, lock_weight=3.0e-5),
    StageSpec("E_full_polish", "full", 360, 0.45, 0.025, 8.0e-4, lock_weight=1.5e-5),
)

FALLBACK_RELEASE = StageSpec("F_loose_release", "loose_full", 300, 0.0, 0.0, 0.0)
FALLBACK_POLISH = StageSpec("G_loose_polish", "loose_full", 150, 0.0, 0.0, 0.0)


def load_row(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["x"], dtype=float), json.loads(str(data["row_json"].item()))


def save_checkpoint(label: str, x: np.ndarray, row: dict[str, object]) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in row.items() if key != "x"}
    np.savez_compressed(
        CHECKPOINT_DIR / f"{label}_0p90277664.npz",
        x=np.asarray(x, dtype=float),
        row_json=np.array(row_json(payload)),
    )


def aligned_nodes(old_logR: np.ndarray, new_logR: np.ndarray, atol: float = 1.0e-10) -> np.ndarray:
    return np.asarray([bool(np.min(np.abs(old_logR - value)) <= atol) for value in new_logR], dtype=bool)


def chain_defect_preserving_values(
    x_old: np.ndarray,
    y_old: np.ndarray,
    x_new: np.ndarray,
    lambda0: float,
    physics,
    prior_weight: float = 0.02,
) -> tuple[np.ndarray, dict[str, object]]:
    x_old = np.asarray(x_old, dtype=float)
    y_old = np.asarray(y_old, dtype=float)
    x_new = np.asarray(x_new, dtype=float)
    y_new = np.column_stack(
        [
            pchip_extrap(x_old, y_old[:, 0], x_new),
            pchip_extrap(x_old, y_old[:, 1], x_new),
        ]
    )
    lower_pair = np.array([physics.logu_bounds[0], physics.logT_bounds[0]], dtype=float)
    upper_pair = np.array([physics.logu_bounds[1], physics.logT_bounds[1]], dtype=float)
    local_defects: list[float] = []
    local_success: list[bool] = []
    copied = 0

    for idx, x_value in enumerate(x_new):
        exact = np.where(np.isclose(x_old, x_value, rtol=0.0, atol=1.0e-11))[0]
        if len(exact):
            y_new[idx] = y_old[int(exact[0])]
            copied += 1

    for old_idx in range(len(x_old) - 1):
        left_x = float(x_old[old_idx])
        right_x = float(x_old[old_idx + 1])
        child_indices = np.where((x_new > left_x + 1.0e-11) & (x_new < right_x - 1.0e-11))[0]
        if len(child_indices) == 0:
            continue
        child_x = x_new[child_indices]
        interp = y_new[child_indices].copy()
        scale = np.maximum(1.0, np.abs(interp))

        def unpack_vars(values: np.ndarray) -> np.ndarray:
            return np.asarray(values, dtype=float).reshape((len(child_indices), 2))

        def residual(values: np.ndarray) -> np.ndarray:
            child_y = unpack_vars(values)
            xs = np.concatenate([[left_x], child_x, [right_x]])
            ys = np.vstack([y_old[old_idx], child_y, y_old[old_idx + 1]])
            rows = [
                interval_residual_between(float(xs[j]), ys[j], float(xs[j + 1]), ys[j + 1], lambda0, physics)
                for j in range(len(xs) - 1)
            ]
            prior = prior_weight * (child_y - interp) / scale
            rows.append(prior.ravel())
            return np.concatenate(rows)

        lower = np.tile(lower_pair, len(child_indices))
        upper = np.tile(upper_pair, len(child_indices))
        result = least_squares(
            residual,
            np.clip(interp.ravel(), lower + 1.0e-12, upper - 1.0e-12),
            bounds=(lower, upper),
            x_scale="jac",
            diff_step=2.0e-5,
            ftol=1.0e-12,
            xtol=1.0e-12,
            gtol=1.0e-12,
            max_nfev=MAX_NFEV_CHAIN_SPLIT,
        )
        y_new[child_indices] = unpack_vars(result.x)
        local_defects.append(float(np.max(np.abs(residual(result.x)))))
        local_success.append(bool(result.success))

    stats = {
        "local_splits": int(sum(1 for value in local_defects if value > 0.0)),
        "copied": int(copied),
        "local_defect_max": float(max(local_defects)) if local_defects else 0.0,
        "local_defect_median": float(np.median(local_defects)) if local_defects else 0.0,
        "local_success_fraction": float(np.mean(local_success)) if local_success else 1.0,
    }
    return y_new, stats


def chain_buffer_to_buffer_seed(x_old: np.ndarray, old_params, new_params) -> tuple[np.ndarray, dict[str, object]]:
    logu_i, logT_i, logu_o, logT_o, logR_son, lambda0 = unpack_buffer(x_old, old_params)
    old_logR_i = buffer_inner_grid(logR_son, old_params)
    old_logR_o = outer_grid(old_params)  # type: ignore[arg-type]
    new_logR_i = buffer_inner_grid(logR_son, new_params)
    new_logR_o = outer_grid(new_params)  # type: ignore[arg-type]
    old_y = np.column_stack([logu_i, logT_i])
    new_y, stats = chain_defect_preserving_values(old_logR_i, old_y, new_logR_i, lambda0, new_params.physics)
    new_y[0] = old_y[0]
    new_y[1] = old_y[1]
    logu_o_new = pchip_extrap(old_logR_o, logu_o, new_logR_o)
    logT_o_new = pchip_extrap(old_logR_o, logT_o, new_logR_o)
    logu_o_new[0] = float(new_y[-1, 0])
    logT_o_new[0] = float(new_y[-1, 1])
    return pack_two_domain(new_y[:, 0], new_y[:, 1], logu_o_new, logT_o_new, logR_son, lambda0), stats


def inner_state_columns(node_indices: list[int], params) -> list[int]:
    ni = params.n_inner
    cols = []
    for idx in node_indices:
        cols.append(idx)
        cols.append(ni + idx)
    return cols


def stage_free_columns(stage: StageSpec, aligned: np.ndarray, params) -> list[int]:
    ni = params.n_inner
    no = params.n_outer
    logR_col = 2 * ni + 2 * no
    lambda_col = logR_col + 1

    if stage.family == "new":
        nodes = [idx for idx in range(2, ni - 1) if not aligned[idx]]
        return inner_state_columns(nodes, params)
    if stage.family == "neighbors":
        node_set: set[int] = set()
        for idx in range(2, ni - 1):
            if not aligned[idx]:
                for neighbor in (idx - 1, idx, idx + 1):
                    if 2 <= neighbor <= ni - 1:
                        node_set.add(neighbor)
        return inner_state_columns(sorted(node_set), params)
    if stage.family == "inner_regular":
        return inner_state_columns(list(range(2, ni)), params)
    if stage.family == "inner_eigen":
        return inner_state_columns(list(range(0, ni)), params) + [logR_col, lambda_col]
    if stage.family == "full":
        return list(range(2 * ni + 2 * no + 2))
    raise ValueError(f"unknown stage family {stage.family!r}")


def column_half_width(col: int, seed: np.ndarray, params, stage: StageSpec) -> float:
    ni = params.n_inner
    no = params.n_outer
    logR_col = 2 * ni + 2 * no
    lambda_col = logR_col + 1
    if col == logR_col:
        rson_rg = float(np.exp(seed[logR_col]) / params.r_g)
        lower_rg = max(rson_rg - stage.rson_half_rg, 2.05)
        upper_rg = rson_rg + stage.rson_half_rg
        return max(abs(np.log(upper_rg / rson_rg)), abs(np.log(rson_rg / lower_rg)), 1.0e-12)
    if col == lambda_col:
        return stage.lambda_half
    if stage.family == "full" and 2 * ni <= col < 2 * ni + 2 * no:
        return min(stage.state_half_width, 0.2)
    return stage.state_half_width


def staged_bounds(seed: np.ndarray, params, free_cols: list[int], stage: StageSpec) -> tuple[np.ndarray, np.ndarray]:
    global_lower, global_upper = state_bounds_two_domain(params)  # type: ignore[arg-type]
    seed = np.asarray(seed, dtype=float)
    lower = seed.copy()
    upper = seed.copy()
    fixed_eps = 1.0e-11 * np.maximum(1.0, np.abs(seed))
    lower -= fixed_eps
    upper += fixed_eps
    free = np.asarray(sorted(set(free_cols)), dtype=int)
    for col in free:
        width = column_half_width(int(col), seed, params, stage)
        lower[col] = seed[col] - width
        upper[col] = seed[col] + width
    lower = np.maximum(lower, global_lower)
    upper = np.minimum(upper, global_upper)
    bad = lower >= upper
    if np.any(bad):
        lower[bad] = np.maximum(global_lower[bad], seed[bad] - fixed_eps[bad])
        upper[bad] = np.minimum(global_upper[bad], seed[bad] + fixed_eps[bad])
    return lower, upper


def augmented_residual(x: np.ndarray, params, g_s: np.ndarray, stage: StageSpec, ref_rson_rg: float, ref_lambda: float) -> np.ndarray:
    base = buffer_residual(x, params, g_s)
    if stage.lock_weight <= 0.0:
        return base
    logR_col = 2 * params.n_inner + 2 * params.n_outer
    rson_rg = float(np.exp(x[logR_col]) / params.r_g)
    locks = np.array(
        [
            stage.lock_weight * (rson_rg - ref_rson_rg) / stage.sigma_rson_rg,
            stage.lock_weight * (float(x[logR_col + 1]) - ref_lambda) / stage.sigma_lambda,
        ],
        dtype=float,
    )
    return np.concatenate([base, locks])


def augmented_sparsity(params, stage: StageSpec):
    base = buffer_sparsity(params)
    if stage.lock_weight <= 0.0:
        return base
    extra = lil_matrix((2, base.shape[1]), dtype=int)
    logR_col = 2 * params.n_inner + 2 * params.n_outer
    extra[0, logR_col] = 1
    extra[1, logR_col + 1] = 1
    return vstack([base, extra.tocsr()]).tocsr()


def solve_stage(seed: np.ndarray, params, g_s: np.ndarray, stage: StageSpec, free_cols: list[int], ref_rson_rg: float, ref_lambda: float):
    lower, upper = staged_bounds(seed, params, free_cols, stage)
    x0 = np.clip(np.asarray(seed, dtype=float), lower + 1.0e-13, upper - 1.0e-13)
    return least_squares(
        lambda trial: augmented_residual(trial, params, g_s, stage, ref_rson_rg, ref_lambda),
        x0,
        jac_sparsity=augmented_sparsity(params, stage),
        bounds=(lower, upper),
        x_scale="jac",
        diff_step=2.0e-5,
        ftol=1.0e-11,
        xtol=1.0e-11,
        gtol=1.0e-10,
        max_nfev=stage.max_nfev,
    )


def write_table(rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Two-Domain Staged Branch-Locked Regular Refinement",
        "",
        "Generated by `scripts/run_transonic_two_domain_staged_regular_refinement.py`.",
        "",
        "Starts from the successful `Delta_s=0.02`, `N_regular=64` sonic-buffer checkpoint and follows `64 -> 80 -> 96 -> 112 -> 128`. Each target runs staged releases: new nodes only, new nodes plus neighbors, regular inner domain with fixed eigenparameters, inner+eigen release, and final full polish with soft branch locks.",
        "",
        "| target | stage | family | N regular | N inner | free cols | physical | pass | dominant | patch | regular R | regular E | outer R | far omega | D | C1 | C2 | K | Rson/rg | step dRson | total dRson | lambda0 | step dlambda | total dlambda | int adv | step dint | total dint | nfev | success | message |",
        "|---|---|---|---:|---:|---:|---:|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {target} | {stage} | {family} | {n_regular} | {n_inner} | {free_count} | {physical_active} | "
            "{passes_science} | {dominant} | {patch} | {regular_R} | {regular_E} | {outer_R} | {far_omega} | "
            "{D} | {C1} | {C2} | {K} | {Rson_rg} | {step_delta_Rson_rg} | {delta_Rson_rg} | {lambda0} | "
            "{step_delta_lambda0} | {delta_lambda0} | {int_adv} | {step_delta_int_adv} | {delta_int_adv} | "
            "{nfev} | {success} | {message} |".format(
                target=row["target"],
                stage=row["stage"],
                family=row["family"],
                n_regular=row["n_regular"],
                n_inner=row["n_inner"],
                free_count=row["free_count"],
                physical_active=fmt(float(row["physical_active"])),
                passes_science="yes" if row["passes_science"] else "no",
                dominant=row["dominant"],
                patch=fmt(float(row["patch"])),
                regular_R=fmt(float(row["regular_R"])),
                regular_E=fmt(float(row["regular_E"])),
                outer_R=fmt(float(row["outer_R"])),
                far_omega=fmt(float(row["far_omega"])),
                D=fmt(float(row["D"])),
                C1=fmt(float(row["C1"])),
                C2=fmt(float(row["C2"])),
                K=fmt(float(row["K"])),
                Rson_rg=fmt(float(row["Rson_rg"])),
                step_delta_Rson_rg=fmt(float(row["step_delta_Rson_rg"])),
                delta_Rson_rg=fmt(float(row["delta_Rson_rg"])),
                lambda0=fmt(float(row["lambda0"])),
                step_delta_lambda0=fmt(float(row["step_delta_lambda0"])),
                delta_lambda0=fmt(float(row["delta_lambda0"])),
                int_adv=fmt(float(row["int_adv"])),
                step_delta_int_adv=fmt(float(row["step_delta_int_adv"])),
                delta_int_adv=fmt(float(row["delta_int_adv"])),
                nfev=row["nfev"],
                success="yes" if row["success"] else "no",
                message=str(row["message"]).replace("|", "/"),
            )
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")


def stage_row(
    target: int,
    stage: StageSpec,
    free_count: int,
    x: np.ndarray,
    params,
    g_s: np.ndarray,
    result,
    ref_row: dict[str, float],
    ref_arrays: dict[str, np.ndarray],
    step_ref: dict[str, float],
) -> dict[str, object]:
    row = buffer_row_with_reference(buffer_audit(f"Nreg{target}_{stage.name}", x, params, g_s, result), ref_row, ref_arrays, params)
    row["target"] = f"Nreg{target}"
    row["stage"] = stage.name
    row["family"] = stage.family
    row["free_count"] = free_count
    row["passes_science"] = bool(float(row["physical_active"]) <= SCIENCE_LIMIT)
    row["step_delta_Rson_rg"] = float(row["Rson_rg"] - step_ref["Rson_rg"])
    row["step_delta_lambda0"] = float(row["lambda0"] - step_ref["lambda0"])
    row["step_delta_int_adv"] = float(row["int_adv"] - step_ref["int_adv"])
    return row


def seed_row(
    target: int,
    stats: dict[str, object],
    x: np.ndarray,
    params,
    g_s: np.ndarray,
    ref_row: dict[str, float],
    ref_arrays: dict[str, np.ndarray],
    step_ref: dict[str, float],
) -> dict[str, object]:
    row = buffer_row_with_reference(buffer_audit(f"Nreg{target}_seed", x, params, g_s), ref_row, ref_arrays, params)
    attach_stage(row, f"Nreg{target}", "seed", "defect_preserving", stats)
    row["target"] = f"Nreg{target}"
    row["family"] = "seed"
    row["free_count"] = 0
    row["passes_science"] = bool(float(row["physical_active"]) <= SCIENCE_LIMIT)
    row["step_delta_Rson_rg"] = float(row["Rson_rg"] - step_ref["Rson_rg"])
    row["step_delta_lambda0"] = float(row["lambda0"] - step_ref["lambda0"])
    row["step_delta_int_adv"] = float(row["int_adv"] - step_ref["int_adv"])
    return row


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    source_x, source_meta = load_checkpoint(SOURCE_CHECKPOINT)
    ratio = float(source_meta["ratio"])
    source_params = make_params(
        fiducial,
        ratio,
        mdot_edd,
        int(source_meta["n_inner"]),
        int(source_meta["n_outer"]),
        float(source_meta["R_far_rg"]),
    )
    ref_arrays = combined_profile_arrays(source_x, source_params)
    ref_row = {
        "Rson_rg": float(source_meta["Rson_rg"]),
        "lambda0": float(source_meta["lambda0"]),
        "int_adv": float(source_meta["int_adv"]),
    }
    g_s = source_first_slope(source_x, source_params)
    current_x, current_meta = load_row(BEST_BUFFER_CHECKPOINT)
    current_params = make_buffer_params(
        fiducial,
        ratio,
        mdot_edd,
        int(current_meta["n_regular"]),
        int(current_meta["n_outer"]),
        float(current_meta["R_far_rg"]),
        float(current_meta["delta_s"]),
    )
    current_audit = buffer_audit("start", current_x, current_params, g_s)
    rows: list[dict[str, object]] = []
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for target in N_SEQUENCE:
        step_ref = {
            "Rson_rg": float(current_audit["Rson_rg"]),
            "lambda0": float(current_audit["lambda0"]),
            "int_adv": float(current_audit["int_adv"]),
        }
        target_params = make_buffer_params(
            fiducial,
            ratio,
            mdot_edd,
            target,
            current_params.n_outer,
            current_params.R_far_rg,
            current_params.delta_s,
        )
        seed, stats = chain_buffer_to_buffer_seed(current_x, current_params, target_params)
        old_logR = buffer_inner_grid(float(current_x[-2]), current_params)
        new_logR = buffer_inner_grid(float(seed[-2]), target_params)
        aligned = aligned_nodes(old_logR, new_logR)
        row = seed_row(target, stats, seed, target_params, g_s, ref_row, ref_arrays, step_ref)
        rows.append(row)
        print(
            f"Nreg{target} seed physical={row['physical_active']:.3e} dominant={row['dominant']} "
            f"local={stats['local_defect_max']:.3e}",
            flush=True,
        )

        stage_seed = seed
        final_row = row
        for stage in STAGES:
            free_cols = stage_free_columns(stage, aligned, target_params)
            result = solve_stage(
                stage_seed,
                target_params,
                g_s,
                stage,
                free_cols,
                step_ref["Rson_rg"],
                step_ref["lambda0"],
            )
            row = stage_row(target, stage, len(set(free_cols)), result.x, target_params, g_s, result, ref_row, ref_arrays, step_ref)
            rows.append(row)
            final_row = row
            stage_seed = np.asarray(result.x, dtype=float)
            write_table(rows)
            print(
                f"Nreg{target} {stage.name} physical={row['physical_active']:.3e} "
                f"dominant={row['dominant']} free={row['free_count']} nfev={result.nfev}",
                flush=True,
            )
        if float(final_row["physical_active"]) > FALLBACK_TRIGGER:
            release = solve_buffer(seed, target_params, g_s, FALLBACK_RELEASE.max_nfev)
            row = stage_row(
                target,
                FALLBACK_RELEASE,
                2 * target_params.n_inner + 2 * target_params.n_outer + 2,
                release.x,
                target_params,
                g_s,
                release,
                ref_row,
                ref_arrays,
                step_ref,
            )
            rows.append(row)
            write_table(rows)
            print(
                f"Nreg{target} {FALLBACK_RELEASE.name} physical={row['physical_active']:.3e} "
                f"dominant={row['dominant']} nfev={release.nfev}",
                flush=True,
            )
            polish = solve_buffer(release.x, target_params, g_s, FALLBACK_POLISH.max_nfev)
            row = stage_row(
                target,
                FALLBACK_POLISH,
                2 * target_params.n_inner + 2 * target_params.n_outer + 2,
                polish.x,
                target_params,
                g_s,
                polish,
                ref_row,
                ref_arrays,
                step_ref,
            )
            rows.append(row)
            write_table(rows)
            final_row = row
            stage_seed = np.asarray(polish.x, dtype=float)
            print(
                f"Nreg{target} {FALLBACK_POLISH.name} physical={row['physical_active']:.3e} "
                f"dominant={row['dominant']} nfev={polish.nfev}",
                flush=True,
            )
        save_checkpoint(f"Nreg{target}", stage_seed, final_row)
        current_x = stage_seed
        current_params = target_params
        current_audit = buffer_audit(f"Nreg{target}", current_x, current_params, g_s)
        limit = CONTINUE_LIMIT if final_row["family"] == "loose_full" else STOP_LIMIT
        if float(final_row["physical_active"]) > limit:
            print(
                f"stopping after Nreg{target}: physical={final_row['physical_active']:.3e} exceeds {limit:.1e}",
                flush=True,
            )
            break

    write_table(rows)
    print(f"wrote {TABLE_OUTPUT}")


if __name__ == "__main__":
    main()
