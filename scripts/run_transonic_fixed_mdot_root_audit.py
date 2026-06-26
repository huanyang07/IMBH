"""Fixed-Mdot algebraic-pivot root audit for transonic square systems."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from imri_qpe.layer3_minidisk_1d import TransonicSlimParams, remap_profile_to_new_sonic_grid
from imri_qpe.layer3_minidisk_1d.transonic_collocation import (
    jacobian_directional_error,
    profile_from_state_vector,
    residual_audit_from_state_vector,
    square_collocation_jacobian,
    square_collocation_residual,
    state_bounds,
    unused_sonic_compatibility,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "transonic_n64_arclength_adaptive"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "transonic_fixed_mdot_root_audit"
TABLE_OUTPUT = ROOT / "outputs" / "tables" / "transonic_fixed_mdot_root_audit.md"

TARGET_FILES = (
    "030_arc_x1_0p90277664.npz",
    "032_arc_x1_0p96532914.npz",
    "033_arc_x1_0p99629052.npz",
)
NODES = (32, 48, 64)
PIVOTS = ("C1", "C2")
JACOBIAN_REL_STEPS = (1.0e-4, 3.0e-5, 1.0e-5)
RCONDS = (1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6)
ROOT_TOL = 1.0e-6
RESIDUAL_TOL = 3.0e-4
R_OUT_RG = 3000.0


def fmt(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.4g}"


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    return value


def row_json(row: dict[str, object]) -> str:
    return json.dumps({key: json_safe(value) for key, value in row.items()}, sort_keys=True)


def params_for(fiducial: FiducialParams, mdot_edd: float, ratio: float, n_nodes: int) -> TransonicSlimParams:
    return TransonicSlimParams(
        M2_g=fiducial.M2_g,
        Mdot_g_s=ratio * mdot_edd,
        alpha=fiducial.alpha_cool,
        n_nodes=n_nodes,
        R_out_rg=R_OUT_RG,
        residual_tol=ROOT_TOL,
        max_nfev=500,
    )


def load_n64_record(path: Path, fiducial: FiducialParams, mdot_edd: float) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        z = np.asarray(data["z"], dtype=float)
        ratio = float(data["ratio"])
    params = params_for(fiducial, mdot_edd, ratio, 64)
    profile = profile_from_state_vector(z, params)
    return {"ratio": ratio, "z": z, "params": params, "profile": profile}


def initial_guess(record: dict[str, object], params: TransonicSlimParams) -> np.ndarray:
    if params.n_nodes == record["params"].n_nodes:
        return np.asarray(record["z"], dtype=float)
    return remap_profile_to_new_sonic_grid(record["profile"], params)


def residual_merit(residual: np.ndarray) -> float:
    residual = np.asarray(residual, dtype=float)
    return 0.5 * float(np.dot(residual, residual))


def active_block_max(audit) -> float:
    return max(
        abs(audit.interval_radial_max),
        abs(audit.interval_energy_max),
        abs(audit.outer_omega),
        abs(audit.outer_energy),
        abs(audit.sonic_D),
        abs(audit.sonic_K),
    )


def dominant_block(audit) -> str:
    values = {
        "interval_R": abs(audit.interval_radial_max),
        "interval_E": abs(audit.interval_energy_max),
        "outer_Omega": abs(audit.outer_omega),
        "outer_E": abs(audit.outer_energy),
        "D": abs(audit.sonic_D),
        "K": abs(audit.sonic_K),
        "C1": abs(audit.sonic_C1),
        "C2": abs(audit.sonic_C2),
    }
    return max(values, key=values.get)


def max_alpha_inside_bounds(z: np.ndarray, step: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    alpha = 1.0
    positive = step > 0.0
    if np.any(positive):
        alpha = min(alpha, float(np.min((upper[positive] - z[positive]) / step[positive])))
    negative = step < 0.0
    if np.any(negative):
        alpha = min(alpha, float(np.min((lower[negative] - z[negative]) / step[negative])))
    if not np.isfinite(alpha):
        return 0.0
    return max(0.0, min(1.0, 0.999 * alpha))


def ruiz_equilibrate_dense(matrix: np.ndarray, n_iter: int = 5, floor: float = 1.0e-300) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return row/column scaled dense matrix and diagonal scale vectors."""

    equilibrated = np.asarray(matrix, dtype=float).copy()
    row_scale = np.ones(equilibrated.shape[0], dtype=float)
    col_scale = np.ones(equilibrated.shape[1], dtype=float)
    for _ in range(n_iter):
        row_norm = np.linalg.norm(equilibrated, axis=1)
        row_factor = 1.0 / np.maximum(row_norm, floor)
        equilibrated = row_factor[:, None] * equilibrated
        row_scale *= row_factor
        col_norm = np.linalg.norm(equilibrated, axis=0)
        col_factor = 1.0 / np.maximum(col_norm, floor)
        equilibrated = equilibrated * col_factor[None, :]
        col_scale *= col_factor
    return equilibrated, row_scale, col_scale


def svd_step_from_equilibrated(
    jac: np.ndarray,
    residual: np.ndarray,
    *,
    rcond: float,
) -> tuple[np.ndarray, float, float, float]:
    """Return an equilibrated dense SVD Newton step and conditioning diagnostics."""

    jac_eq, row_scale, col_scale = ruiz_equilibrate_dense(jac)
    rhs_eq = -row_scale * residual
    singular_values = np.linalg.svd(jac_eq, compute_uv=False)
    cond_eq = float(singular_values[0] / max(singular_values[-1], 1.0e-300))
    raw_singular_values = np.linalg.svd(jac, compute_uv=False)
    cond_raw = float(raw_singular_values[0] / max(raw_singular_values[-1], 1.0e-300))
    U, s, Vt = np.linalg.svd(jac_eq, full_matrices=False)
    cutoff = rcond * s[0]
    denom = np.maximum(s, cutoff)
    step_eq = Vt.T @ ((U.T @ rhs_eq) / denom)
    step = col_scale * step_eq
    return step, cond_raw, cond_eq, float(s[-1])


def dense_svd_newton(
    z0: np.ndarray,
    params: TransonicSlimParams,
    *,
    pivot: str,
    jacobian_rel_step: float,
    max_iter: int = 10,
    max_step_norm: float = 1.0,
    line_search_max_reductions: int = 14,
) -> dict[str, object]:
    lower, upper = state_bounds(params)
    z = np.clip(np.asarray(z0, dtype=float), lower + 1.0e-12, upper - 1.0e-12)
    residual = square_collocation_residual(z, params, pivot=pivot)
    initial_max = float(np.max(np.abs(residual)))
    best_z = np.array(z, copy=True)
    best_residual = np.array(residual, copy=True)
    best_merit = residual_merit(residual)
    best_message = "initial state"
    total_cuts = 0
    cond_raw = np.nan
    cond_eq = np.nan
    smin_eq = np.nan
    final_rcond = np.nan
    final_step_norm = 0.0
    iterations = 0
    nfev = 1
    njev = 0
    for iteration in range(max_iter + 1):
        current_max = float(np.max(np.abs(residual)))
        if current_max <= ROOT_TOL:
            return {
                "z": z,
                "initial_max": initial_max,
                "final_max": current_max,
                "iterations": iteration,
                "nfev": nfev,
                "njev": njev,
                "cuts": total_cuts,
                "cond_raw": cond_raw,
                "cond_eq": cond_eq,
                "smin_eq": smin_eq,
                "rcond": final_rcond,
                "step_norm": final_step_norm,
                "message": "dense SVD Newton converged",
            }
        if iteration == max_iter:
            break
        jac = square_collocation_jacobian(z, params, pivot=pivot, rel_step=jacobian_rel_step).toarray()
        njev += 1
        merit = residual_merit(residual)
        accepted = False
        best_trial = None
        best_trial_merit = np.inf
        best_trial_meta = None
        for rcond in RCONDS:
            step, raw_cond, eq_cond, eq_smin = svd_step_from_equilibrated(jac, residual, rcond=rcond)
            step_norm = float(np.linalg.norm(step, ord=np.inf))
            if not np.isfinite(step_norm) or step_norm <= 0.0:
                continue
            if step_norm > max_step_norm:
                step = step * (max_step_norm / step_norm)
                step_norm = max_step_norm
            alpha = max_alpha_inside_bounds(z, step, lower, upper)
            reductions = 0
            while alpha >= 1.0e-8 and reductions <= line_search_max_reductions:
                trial_z = np.clip(z + alpha * step, lower + 1.0e-12, upper - 1.0e-12)
                trial_residual = square_collocation_residual(trial_z, params, pivot=pivot)
                nfev += 1
                trial_merit = residual_merit(trial_residual)
                if trial_merit < best_trial_merit:
                    best_trial = (trial_z, trial_residual)
                    best_trial_merit = trial_merit
                    best_trial_meta = (rcond, reductions, raw_cond, eq_cond, eq_smin, step_norm)
                if trial_merit < merit:
                    accepted = True
                    break
                alpha *= 0.5
                reductions += 1
            if accepted:
                break
        if best_trial is not None and best_trial_merit < best_merit:
            best_z = np.array(best_trial[0], copy=True)
            best_residual = np.array(best_trial[1], copy=True)
            best_merit = best_trial_merit
        if best_trial_meta is None:
            best_message = "dense SVD step failed"
            break
        final_rcond, reductions, cond_raw, cond_eq, smin_eq, final_step_norm = best_trial_meta
        total_cuts += int(reductions)
        iterations = iteration + 1
        if accepted:
            z, residual = best_trial
            best_message = "maximum dense SVD Newton iterations reached"
        else:
            best_message = "dense SVD line search failed to reduce residual"
            break
    final_residual = square_collocation_residual(best_z, params, pivot=pivot)
    return {
        "z": best_z,
        "initial_max": initial_max,
        "final_max": float(np.max(np.abs(final_residual))),
        "iterations": iterations,
        "nfev": nfev,
        "njev": njev,
        "cuts": total_cuts,
        "cond_raw": cond_raw,
        "cond_eq": cond_eq,
        "smin_eq": smin_eq,
        "rcond": final_rcond,
        "step_norm": final_step_norm,
        "message": best_message,
    }


def run_case(record: dict[str, object], fiducial: FiducialParams, mdot_edd: float, n_nodes: int, pivot: str) -> dict[str, object]:
    ratio = float(record["ratio"])
    params = params_for(fiducial, mdot_edd, ratio, n_nodes)
    z0 = initial_guess(record, params)
    best = None
    for rel_step in JACOBIAN_REL_STEPS:
        result = dense_svd_newton(z0, params, pivot=pivot, jacobian_rel_step=rel_step)
        result["jacobian_rel_step"] = rel_step
        if best is None or result["final_max"] < best["final_max"]:
            best = result
        print(
            f"ratio={ratio:.6g} N={n_nodes} pivot={pivot} hJ={rel_step:.1e} "
            f"{result['initial_max']:.3e}->{result['final_max']:.3e} {result['message']}",
            flush=True,
        )
    z = np.asarray(best["z"], dtype=float)
    profile = profile_from_state_vector(z, params)
    audit = residual_audit_from_state_vector(z, params)
    square = square_collocation_residual(z, params, pivot=pivot)
    square_max = float(np.max(np.abs(square)))
    compat_max = float(max(abs(audit.sonic_C1), abs(audit.sonic_C2), abs(audit.sonic_K)))
    unused = float(unused_sonic_compatibility(z, params, pivot=pivot))
    try:
        directional = jacobian_directional_error(
            z,
            params,
            pivot=pivot,
            steps=(1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5, 1.0e-5, 3.0e-6, 1.0e-6, 3.0e-7),
            n_directions=2,
            seed=1000 + 10 * n_nodes + (0 if pivot == "C1" else 1),
            jacobian_rel_step=float(best["jacobian_rel_step"]),
        )
        fd_best_step = directional.best_step
        fd_best_error = directional.best_median_error
    except Exception:
        fd_best_step = np.nan
        fd_best_error = np.nan
    return {
        "ratio": ratio,
        "n_nodes": n_nodes,
        "pivot": pivot,
        "root": square_max <= ROOT_TOL and compat_max <= ROOT_TOL,
        "initial_max": float(best["initial_max"]),
        "square_max": square_max,
        "active_max": active_block_max(audit),
        "dominant": dominant_block(audit),
        "D": audit.sonic_D,
        "K": audit.sonic_K,
        "C1": audit.sonic_C1,
        "C2": audit.sonic_C2,
        "unused": unused,
        "compat_max": compat_max,
        "smin": audit.sonic_smin_over_smax,
        "max_HR": float(np.max(profile.H_over_R)),
        "int_adv": profile.integrated_advective_fraction,
        "iterations": int(best["iterations"]),
        "nfev": int(best["nfev"]),
        "njev": int(best["njev"]),
        "cuts": int(best["cuts"]),
        "jacobian_rel_step": float(best["jacobian_rel_step"]),
        "rcond": float(best["rcond"]),
        "step_norm": float(best["step_norm"]),
        "cond_raw": float(best["cond_raw"]),
        "cond_eq": float(best["cond_eq"]),
        "smin_eq": float(best["smin_eq"]),
        "fd_best_step": float(fd_best_step),
        "fd_best_error": float(fd_best_error),
        "message": str(best["message"]),
        "z": z,
        "params": params,
    }


def checkpoint_path(row: dict[str, object]) -> Path:
    ratio = f"{float(row['ratio']):.8f}".replace(".", "p")
    return CHECKPOINT_DIR / f"ratio_{ratio}_N{row['n_nodes']}_{row['pivot']}.npz"


def save_checkpoint(row: dict[str, object]) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    params = row["params"]
    path = checkpoint_path(row)
    payload = {key: value for key, value in row.items() if key not in {"z", "params"}}
    np.savez_compressed(
        path,
        ratio=np.array(row["ratio"]),
        n_nodes=np.array(row["n_nodes"]),
        pivot=np.array(row["pivot"]),
        z=np.asarray(row["z"], dtype=float),
        R_out_rg=np.array(params.R_out_rg),
        residual_tol=np.array(params.residual_tol),
        row_json=np.array(row_json(payload)),
    )


def write_table(rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Fixed-Mdot Algebraic-Pivot Root Audit",
        "",
        "Generated by `scripts/run_transonic_fixed_mdot_root_audit.py`.",
        "",
        f"Root criterion: square max and `max(|C1|, |C2|, |K|)` both below `{ROOT_TOL:g}`.",
        "",
        "| Mdot/Edd | N | pivot | root | square initial | square final | active max | dominant | D | C1 | C2 | K | unused | compat max | smin | max H/R | int adv | iter | cuts | hJ | rcond | raw cond | eq cond | eq smin | FD best h | FD err | message |",
        "|---:|---:|:---:|:---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {ratio} | {n_nodes} | {pivot} | {root} | {initial_max} | {square_max} | {active_max} | "
            "{dominant} | {D} | {C1} | {C2} | {K} | {unused} | {compat_max} | {smin} | "
            "{max_HR} | {int_adv} | {iterations} | {cuts} | {jacobian_rel_step} | {rcond} | "
            "{cond_raw} | {cond_eq} | {smin_eq} | {fd_best_step} | {fd_best_error} | {message} |".format(
                ratio=fmt(float(row["ratio"])),
                n_nodes=row["n_nodes"],
                pivot=row["pivot"],
                root="yes" if row["root"] else "no",
                initial_max=fmt(float(row["initial_max"])),
                square_max=fmt(float(row["square_max"])),
                active_max=fmt(float(row["active_max"])),
                dominant=row["dominant"],
                D=fmt(float(row["D"])),
                C1=fmt(float(row["C1"])),
                C2=fmt(float(row["C2"])),
                K=fmt(float(row["K"])),
                unused=fmt(float(row["unused"])),
                compat_max=fmt(float(row["compat_max"])),
                smin=fmt(float(row["smin"])),
                max_HR=fmt(float(row["max_HR"])),
                int_adv=fmt(float(row["int_adv"])),
                iterations=row["iterations"],
                cuts=row["cuts"],
                jacobian_rel_step=fmt(float(row["jacobian_rel_step"])),
                rcond=fmt(float(row["rcond"])),
                cond_raw=fmt(float(row["cond_raw"])),
                cond_eq=fmt(float(row["cond_eq"])),
                smin_eq=fmt(float(row["smin_eq"])),
                fd_best_step=fmt(float(row["fd_best_step"])),
                fd_best_error=fmt(float(row["fd_best_error"])),
                message=str(row["message"]).replace("|", "/"),
            )
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    source_records = [load_n64_record(SOURCE_CHECKPOINT_DIR / filename, fiducial, mdot_edd) for filename in TARGET_FILES]
    rows: list[dict[str, object]] = []
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for old_path in CHECKPOINT_DIR.glob("*.npz"):
        old_path.unlink()
    for record in source_records:
        for n_nodes in NODES:
            for pivot in PIVOTS:
                row = run_case(record, fiducial, mdot_edd, n_nodes, pivot)
                rows.append(row)
                save_checkpoint(row)
    write_table(rows)
    print(f"wrote {TABLE_OUTPUT}")


if __name__ == "__main__":
    main()
