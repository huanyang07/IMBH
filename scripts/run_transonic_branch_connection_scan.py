"""Scan whether the exact sonic branch connects to existing tail profiles."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares

from imri_qpe.layer3_minidisk_1d.transonic_collocation import _heating_terms_from_gradient
from imri_qpe.layer3_minidisk_1d.transonic_local import (
    algebraic_state,
    local_ode_rhs,
    sonic_derivative_branches,
    sonic_diagnostics,
    sonic_lhopital_residual_form,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot

from run_transonic_outer_slope_calibration_audit import fmt
from run_transonic_reverse_seed_graded_flowmap_bvp import (
    graded_inner_grid,
    load_start_checkpoint,
)
from run_transonic_sonic_reverse_fit import (
    EPS_INFER,
    INITIAL_DLAMBDA,
    INITIAL_DX,
    MAX_NFEV,
    SONIC_SCALE,
    SOURCE_CHECKPOINT,
    evaluate_trial,
    load_context,
    solve_fit,
)
from run_transonic_two_domain_dynamic_sonic_patch import load_row
from run_transonic_two_domain_sonic_flowmap import (
    LHOPITAL_EPS,
    branch_from_a,
    branch_gradient_from_a,
    outer_grid,
    unpack_smooth_flowmap,
)
from run_transonic_two_domain_sonic_refinement_sprint import buffer_inner_grid, unpack_buffer


ROOT = Path(__file__).resolve().parents[1]
TABLE_OUTPUT = ROOT / os.environ.get(
    "IMBH_BRANCH_CONNECTION_TABLE",
    "outputs/tables/transonic_branch_connection_scan.md",
)
FIGURE_OUTPUT = ROOT / os.environ.get(
    "IMBH_BRANCH_CONNECTION_FIGURE",
    "outputs/figures/transonic_branch_connection_scan.png",
)
BEST_CHECKPOINT = ROOT / os.environ.get(
    "IMBH_BRANCH_CONNECTION_BEST_CHECKPOINT",
    "outputs/checkpoints/transonic_two_domain_sonic_flowmap/"
    "graded_graded_transition_ode_blend_eps0p002_branch1_N96_0p90277664.npz",
)
R_JOIN_RG = tuple(
    float(piece)
    for piece in os.environ.get("IMBH_BRANCH_CONNECTION_RJOIN", "5.95,6,6.1,6.2,6.3,6.4,6.5,7,8,10,12,15")
    .replace(":", ",")
    .split(",")
    if piece.strip()
)
FIXED_DLAMBDA = tuple(
    float(piece)
    for piece in os.environ.get(
        "IMBH_BRANCH_CONNECTION_DLAMBDAS",
        "0,-5e-4,-1e-3,-1.5e-3,-1.53e-3,-2e-3",
    )
    .replace(":", ",")
    .split(",")
    if piece.strip()
)
BRANCH_SEQUENCE = tuple(
    int(piece)
    for piece in os.environ.get("IMBH_BRANCH_CONNECTION_BRANCHES", "0,1").replace(":", ",").split(",")
    if piece.strip()
)
FLOW_EPS0 = float(os.environ.get("IMBH_BRANCH_CONNECTION_EPS0", "1e-6"))
FLOW_MAX_STEP = float(os.environ.get("IMBH_BRANCH_CONNECTION_MAX_STEP", "1e-3"))
FLOW_RTOL = float(os.environ.get("IMBH_BRANCH_CONNECTION_RTOL", "1e-8"))
FLOW_ATOL = float(os.environ.get("IMBH_BRANCH_CONNECTION_ATOL", "1e-10"))
BRANCH_HALF_WIDTH = float(os.environ.get("IMBH_BRANCH_CONNECTION_BRANCH_HALF_WIDTH", "1000"))
BRANCH_POINTS = int(os.environ.get("IMBH_BRANCH_CONNECTION_BRANCH_POINTS", "2001"))
TARGET_NAMES = tuple(
    piece.strip()
    for piece in os.environ.get("IMBH_BRANCH_CONNECTION_TARGETS", "best_exact_bvp,dynamic_patch_source")
    .replace(":", ",")
    .split(",")
    if piece.strip()
)


@dataclass(frozen=True)
class TargetProfile:
    name: str
    logR: np.ndarray
    logu: np.ndarray
    logT: np.ndarray
    lambda0: float
    params: object


@dataclass(frozen=True)
class CriticalCase:
    label: str
    mode: str
    dlambda_seed: float
    dlambda_actual: float
    logR_son: float
    y_s: np.ndarray
    lambda0: float
    D: float
    C1: float
    C2: float
    K: float
    critK: float
    nfev: int
    success: bool
    message: str


def safe_float(value: float) -> float:
    return float(value) if np.isfinite(value) else np.nan


def tag_float(value: float) -> str:
    return f"{value:+.5g}".replace("+", "p").replace("-", "m").replace(".", "p")


def json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def entropy_proxy(state) -> float:
    return float(np.log(state.P) - (5.0 / 3.0) * np.log(state.rho))


def profile_interpolators(target: TargetProfile):
    order = np.argsort(target.logR)
    logR = np.asarray(target.logR[order], dtype=float)
    unique = np.concatenate([[True], np.diff(logR) > 1.0e-13])
    logR = logR[unique]
    logu = np.asarray(target.logu[order][unique], dtype=float)
    logT = np.asarray(target.logT[order][unique], dtype=float)
    return (
        PchipInterpolator(logR, logu, extrapolate=True),
        PchipInterpolator(logR, logT, extrapolate=True),
        float(logR[0]),
        float(logR[-1]),
    )


def state_metrics(logR: float, y: np.ndarray, g: np.ndarray, lambda0: float, params) -> dict[str, float]:
    state = algebraic_state(logR, float(y[0]), float(y[1]), lambda0, params.physics)
    q_visc, _q_rad, q_adv, _energy = _heating_terms_from_gradient(logR, y, g, lambda0, params.physics)
    return {
        "H_R": float(state.H_over_R),
        "omega_ratio": float(state.Omega / state.Omega_K),
        "entropy_proxy": entropy_proxy(state),
        "qadv_qvisc": safe_float(float(q_adv / (q_visc + 1.0e-300))),
        "qvisc_sign": float(np.sign(q_visc)),
    }


def target_at(target: TargetProfile, logR: float) -> tuple[np.ndarray, np.ndarray, dict[str, float], bool]:
    interp_u, interp_T, xmin, xmax = profile_interpolators(target)
    y = np.array([float(interp_u(logR)), float(interp_T(logR))], dtype=float)
    g = np.array([float(interp_u.derivative()(logR)), float(interp_T.derivative()(logR))], dtype=float)
    in_range = bool(xmin <= logR <= xmax)
    return y, g, state_metrics(logR, y, g, target.lambda0, target.params), in_range


def load_best_target(fiducial: FiducialParams, source_meta: dict[str, object], mdot_edd: float) -> TargetProfile:
    x, params, row = load_start_checkpoint(BEST_CHECKPOINT, fiducial, source_meta, mdot_edd)
    y_s, logu_i, logT_i, logu_o, logT_o, logR_son, lambda0, _a = unpack_smooth_flowmap(x, params)
    grid = graded_inner_grid(logR_son, params)
    logR_o = outer_grid(params)  # type: ignore[arg-type]
    return TargetProfile(
        name=f"best_exact_bvp:{row.get('label', BEST_CHECKPOINT.name)}",
        logR=np.concatenate([np.array([logR_son]), grid.logR, logR_o[1:]]),
        logu=np.concatenate([np.array([float(y_s[0])]), logu_i, logu_o[1:]]),
        logT=np.concatenate([np.array([float(y_s[1])]), logT_i, logT_o[1:]]),
        lambda0=float(lambda0),
        params=params,
    )


def load_dynamic_target(ctx, source_x: np.ndarray) -> TargetProfile:
    logu_i, logT_i, logu_o, logT_o, logR_son, lambda0 = unpack_buffer(source_x, ctx.params)
    logR_i = buffer_inner_grid(logR_son, ctx.params)
    logR_o = outer_grid(ctx.params)  # type: ignore[arg-type]
    return TargetProfile(
        name="dynamic_patch_source",
        logR=np.concatenate([logR_i, logR_o[1:]]),
        logu=np.concatenate([logu_i, logu_o[1:]]),
        logT=np.concatenate([logT_i, logT_o[1:]]),
        lambda0=float(lambda0),
        params=ctx.params,
    )


def critical_case_from_info(label: str, mode: str, dlambda_seed: float, result, info, ctx) -> CriticalCase:
    sonic = info["sonic"]
    return CriticalCase(
        label=label,
        mode=mode,
        dlambda_seed=float(dlambda_seed),
        dlambda_actual=float(info["lambda0"] - ctx.lambda0_old),
        logR_son=float(info["logR_son"]),
        y_s=np.asarray(info["y_s"], dtype=float),
        lambda0=float(info["lambda0"]),
        D=float(sonic.D),
        C1=float(sonic.C1),
        C2=float(sonic.C2),
        K=float(sonic.compatibility),
        critK=max(abs(float(sonic.D)), abs(float(sonic.C1)), abs(float(sonic.C2)), abs(float(sonic.compatibility))),
        nfev=int(result.nfev),
        success=bool(result.success),
        message=str(result.message),
    )


def solve_fixed_dlambda(ctx, dlambda: float) -> CriticalCase:
    lower = np.array([max(EPS_INFER * 5.0, 1.0e-6)], dtype=float)
    upper = np.array([5.0e-2], dtype=float)
    seed_dx = np.array([INITIAL_DX], dtype=float)

    def residual(trial: np.ndarray) -> np.ndarray:
        try:
            info = evaluate_trial(np.array([float(trial[0]), float(dlambda)], dtype=float), ctx, "fixed_buffer")
            sonic = info["sonic"]
            return np.array([sonic.D, sonic.C1, sonic.C2, sonic.compatibility], dtype=float) / SONIC_SCALE
        except Exception:
            return np.full(4, 1.0e8)

    result = least_squares(
        residual,
        seed_dx,
        bounds=(lower, upper),
        x_scale=np.array([1.0e-3], dtype=float),
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-12,
        max_nfev=MAX_NFEV,
    )
    try:
        info = evaluate_trial(np.array([float(result.x[0]), float(dlambda)], dtype=float), ctx, "fixed_buffer")
    except Exception as exc:
        return CriticalCase(
            label=f"fixed_dlambda_{tag_float(dlambda)}",
            mode="fixed_dlambda",
            dlambda_seed=float(dlambda),
            dlambda_actual=float(dlambda),
            logR_son=float(ctx.logR_buffer - float(result.x[0])),
            y_s=np.array([np.nan, np.nan], dtype=float),
            lambda0=float(ctx.lambda0_old + dlambda),
            D=np.nan,
            C1=np.nan,
            C2=np.nan,
            K=np.nan,
            critK=np.inf,
            nfev=int(result.nfev),
            success=False,
            message=str(exc),
        )
    return critical_case_from_info(f"fixed_dlambda_{tag_float(dlambda)}", "fixed_dlambda", dlambda, result, info, ctx)


def critical_cases(ctx) -> list[CriticalCase]:
    result, info = solve_fit(ctx, "fixed_buffer", None)
    rows = [critical_case_from_info("free_reverse_fit", "free_dlambda", INITIAL_DLAMBDA, result, info, ctx)]
    for dlambda in FIXED_DLAMBDA:
        rows.append(solve_fixed_dlambda(ctx, float(dlambda)))
    return rows


def integrate_branch(case: CriticalCase, branch, join_logR: np.ndarray, params) -> tuple[np.ndarray | None, dict[str, object]]:
    x0 = case.logR_son + FLOW_EPS0
    x1 = float(np.max(join_logR))
    if np.any(join_logR <= x0):
        raise ValueError("join radii must lie outside the sonic startup offset")
    y0 = case.y_s + FLOW_EPS0 * branch.gradient

    def rhs(x_value: float, y_value: np.ndarray) -> np.ndarray:
        return local_ode_rhs(float(x_value), y_value, case.lambda0, params.physics)

    try:
        sol = solve_ivp(
            rhs,
            (x0, x1),
            y0,
            method="Radau",
            dense_output=True,
            max_step=FLOW_MAX_STEP,
            rtol=FLOW_RTOL,
            atol=FLOW_ATOL,
        )
        reached_logR = float(sol.t[-1]) if sol.t.size else np.nan
        reached_R_rg = float(np.exp(reached_logR) / params.r_g) if np.isfinite(reached_logR) else np.nan
        meta = {
            "success": bool(sol.success),
            "nfev": int(sol.nfev),
            "message": str(sol.message),
            "reached_logR": reached_logR,
            "reached_R_rg": reached_R_rg,
            "reached_dx": float(reached_logR - case.logR_son) if np.isfinite(reached_logR) else np.nan,
        }
        if sol.sol is None:
            return None, meta
        values = np.full((len(join_logR), 2), np.nan, dtype=float)
        reachable = join_logR <= reached_logR + 1.0e-12
        if np.any(reachable):
            values[reachable] = np.asarray(sol.sol(join_logR[reachable]).T, dtype=float)
        if values.shape != (len(join_logR), 2) or not np.all(np.isfinite(values)):
            if not np.any(np.all(np.isfinite(values), axis=1)):
                return None, meta
        return values, meta
    except Exception as exc:
        return None, {
            "success": False,
            "nfev": 0,
            "message": str(exc),
            "reached_logR": np.nan,
            "reached_R_rg": np.nan,
            "reached_dx": np.nan,
        }


def scan_rows(ctx, cases: list[CriticalCase], targets: list[TargetProfile]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    join_logR = np.log(np.asarray(R_JOIN_RG, dtype=float) * ctx.params.r_g)
    for case in cases:
        if not np.isfinite(case.critK) or not np.all(np.isfinite(case.y_s)):
            for branch_index in BRANCH_SEQUENCE:
                rows.append(
                    {
                        "case": case.label,
                        "mode": case.mode,
                        "branch": branch_index,
                        "status": "critical_failed",
                        "message": case.message,
                        "dlambda_seed": case.dlambda_seed,
                        "dlambda_actual": case.dlambda_actual,
                        "lambda0": case.lambda0,
                        "Rson_rg": float(np.exp(case.logR_son) / ctx.params.r_g),
                        "critK": case.critK,
                        "critical_success": case.success,
                        "critical_nfev": case.nfev,
                    }
                )
            continue
        branches = sonic_derivative_branches(
            case.logR_son,
            case.y_s,
            case.lambda0,
            ctx.params.physics,
            eps=LHOPITAL_EPS,
            form="scaled",
            half_width=BRANCH_HALF_WIDTH,
            scan_points=BRANCH_POINTS,
        )
        for branch_index in BRANCH_SEQUENCE:
            if branch_index >= len(branches):
                rows.append(
                    {
                        "case": case.label,
                        "mode": case.mode,
                        "branch": branch_index,
                        "status": "missing_branch",
                        "message": f"found only {len(branches)} branches",
                        "dlambda_seed": case.dlambda_seed,
                        "dlambda_actual": case.dlambda_actual,
                        "lambda0": case.lambda0,
                        "Rson_rg": float(np.exp(case.logR_son) / ctx.params.r_g),
                        "critK": case.critK,
                    }
                )
                continue
            branch = branches[branch_index]
            values, flow = integrate_branch(case, branch, join_logR, ctx.params)
            for idx, R_join in enumerate(R_JOIN_RG):
                for target in targets:
                    base = {
                        "case": case.label,
                        "mode": case.mode,
                        "target": target.name,
                        "R_join_rg": float(R_join),
                        "branch": int(branch_index),
                        "a": float(branch.a),
                        "g_s_u": float(branch.gradient[0]),
                        "g_s_T": float(branch.gradient[1]),
                        "L": float(branch.lhopital_normalized),
                        "dlambda_seed": case.dlambda_seed,
                        "dlambda_actual": case.dlambda_actual,
                        "lambda0": case.lambda0,
                        "Rson_rg": float(np.exp(case.logR_son) / ctx.params.r_g),
                        "D": case.D,
                        "C1": case.C1,
                        "C2": case.C2,
                        "K": case.K,
                        "critK": case.critK,
                        "critical_success": case.success,
                        "critical_nfev": case.nfev,
                        "flow_nfev": flow["nfev"],
                        "flow_reached_R_rg": flow.get("reached_R_rg", np.nan),
                        "flow_reached_dx": flow.get("reached_dx", np.nan),
                        "message": flow["message"],
                    }
                    if values is None or not np.all(np.isfinite(values[idx])):
                        rows.append({**base, "status": "flow_failed"})
                        continue
                    logR = float(join_logR[idx])
                    y_exact = values[idx]
                    g_exact = local_ode_rhs(logR, y_exact, case.lambda0, ctx.params.physics)
                    exact_metrics = state_metrics(logR, y_exact, g_exact, case.lambda0, ctx.params)
                    y_target, g_target, target_metrics, in_target_range = target_at(target, logR)
                    dy = y_exact - y_target
                    dg = g_exact - g_target
                    rows.append(
                        {
                            **base,
                            "status": "ok" if flow["success"] else "flow_partial",
                            "target_in_range": in_target_range,
                            "exact_logu": float(y_exact[0]),
                            "exact_logT": float(y_exact[1]),
                            "target_logu": float(y_target[0]),
                            "target_logT": float(y_target[1]),
                            "delta_logu": float(dy[0]),
                            "delta_logT": float(dy[1]),
                            "distance_y": float(np.max(np.abs(dy))),
                            "exact_g_u": float(g_exact[0]),
                            "exact_g_T": float(g_exact[1]),
                            "target_g_u": float(g_target[0]),
                            "target_g_T": float(g_target[1]),
                            "delta_g_u": float(dg[0]),
                            "delta_g_T": float(dg[1]),
                            "distance_g": float(np.max(np.abs(dg))),
                            "delta_entropy_proxy": float(exact_metrics["entropy_proxy"] - target_metrics["entropy_proxy"]),
                            "exact_omega_over_K": exact_metrics["omega_ratio"],
                            "target_omega_over_K": target_metrics["omega_ratio"],
                            "delta_omega_over_K": float(exact_metrics["omega_ratio"] - target_metrics["omega_ratio"]),
                            "exact_H_R": exact_metrics["H_R"],
                            "target_H_R": target_metrics["H_R"],
                            "delta_H_R": float(exact_metrics["H_R"] - target_metrics["H_R"]),
                            "exact_qadv_qvisc": exact_metrics["qadv_qvisc"],
                            "target_qadv_qvisc": target_metrics["qadv_qvisc"],
                            "delta_qadv_qvisc": float(exact_metrics["qadv_qvisc"] - target_metrics["qadv_qvisc"]),
                        }
                    )
    return rows


def write_table(rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Transonic Branch-Connection Scan",
        "",
        "Generated by `scripts/run_transonic_branch_connection_scan.py`.",
        "",
        f"Best checkpoint: `{BEST_CHECKPOINT.relative_to(ROOT) if BEST_CHECKPOINT.is_relative_to(ROOT) else BEST_CHECKPOINT}`.",
        "",
        f"R_join/rg: `{', '.join(fmt(value) for value in R_JOIN_RG)}`.",
        "",
        f"Fixed dlambda seeds: `{', '.join(fmt(value) for value in FIXED_DLAMBDA)}`.",
        "",
        "| case | mode | target | R_join/rg | branch | status | dist y | dlogu | dlogT | dist g | dg_u | dg_T | dS proxy | dOmega/K | dH/R | dQadv/Qvisc | critK | D | C1 | C2 | K | Rson/rg | reached R/rg | reached dx | lambda0 | dlambda | a | g_s_u | g_s_T | L | flow nfev | message |",
        "|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {case} | {mode} | {target} | {R_join_rg} | {branch} | {status} | {distance_y} | "
            "{delta_logu} | {delta_logT} | {distance_g} | {delta_g_u} | {delta_g_T} | "
            "{delta_entropy_proxy} | {delta_omega_over_K} | {delta_H_R} | {delta_qadv_qvisc} | "
            "{critK} | {D} | {C1} | {C2} | {K} | {Rson_rg} | {flow_reached_R_rg} | "
            "{flow_reached_dx} | {lambda0} | {dlambda_actual} | {a} | {g_s_u} | {g_s_T} | "
            "{L} | {flow_nfev} | {message} |".format(
                case=row.get("case", ""),
                mode=row.get("mode", ""),
                target=str(row.get("target", "")).replace("|", "/"),
                R_join_rg=fmt(float(row.get("R_join_rg", np.nan))),
                branch=row.get("branch", ""),
                status=row.get("status", ""),
                distance_y=fmt(float(row.get("distance_y", np.nan))),
                delta_logu=fmt(float(row.get("delta_logu", np.nan))),
                delta_logT=fmt(float(row.get("delta_logT", np.nan))),
                distance_g=fmt(float(row.get("distance_g", np.nan))),
                delta_g_u=fmt(float(row.get("delta_g_u", np.nan))),
                delta_g_T=fmt(float(row.get("delta_g_T", np.nan))),
                delta_entropy_proxy=fmt(float(row.get("delta_entropy_proxy", np.nan))),
                delta_omega_over_K=fmt(float(row.get("delta_omega_over_K", np.nan))),
                delta_H_R=fmt(float(row.get("delta_H_R", np.nan))),
                delta_qadv_qvisc=fmt(float(row.get("delta_qadv_qvisc", np.nan))),
                critK=fmt(float(row.get("critK", np.nan))),
                D=fmt(float(row.get("D", np.nan))),
                C1=fmt(float(row.get("C1", np.nan))),
                C2=fmt(float(row.get("C2", np.nan))),
                K=fmt(float(row.get("K", np.nan))),
                Rson_rg=fmt(float(row.get("Rson_rg", np.nan))),
                flow_reached_R_rg=fmt(float(row.get("flow_reached_R_rg", np.nan))),
                flow_reached_dx=fmt(float(row.get("flow_reached_dx", np.nan))),
                lambda0=fmt(float(row.get("lambda0", np.nan))),
                dlambda_actual=fmt(float(row.get("dlambda_actual", np.nan))),
                a=fmt(float(row.get("a", np.nan))),
                g_s_u=fmt(float(row.get("g_s_u", np.nan))),
                g_s_T=fmt(float(row.get("g_s_T", np.nan))),
                L=fmt(float(row.get("L", np.nan))),
                flow_nfev=row.get("flow_nfev", ""),
                message=str(row.get("message", "")).replace("|", "/"),
            )
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")


def write_json(rows: list[dict[str, object]]) -> None:
    path = TABLE_OUTPUT.with_suffix(".json")
    payload = [{key: json_safe(value) for key, value in row.items()} for row in rows]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_figure(rows: list[dict[str, object]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable, using PIL figure fallback: {exc}", flush=True)
        write_pil_figure(rows)
        return

    ok_rows = [row for row in rows if row.get("status") in {"ok", "flow_partial"} and int(row.get("branch", -1)) == 1]
    if not ok_rows:
        return
    targets = [name for name in TARGET_NAMES if any(str(row.get("target", "")).startswith(name) for row in ok_rows)]
    if not targets:
        targets = sorted({str(row.get("target", "")) for row in ok_rows})
    fig, axes = plt.subplots(1, len(targets), figsize=(6.0 * len(targets), 4.2), squeeze=False)
    for ax, target_prefix in zip(axes[0], targets):
        target_rows = [row for row in ok_rows if str(row.get("target", "")).startswith(target_prefix)]
        labels = list(dict.fromkeys(str(row["case"]) for row in target_rows))
        for label in labels:
            case_rows = sorted(
                [row for row in target_rows if row["case"] == label],
                key=lambda item: float(item["R_join_rg"]),
            )
            if not case_rows:
                continue
            ax.semilogy(
                [float(row["R_join_rg"]) for row in case_rows],
                [float(row["distance_y"]) for row in case_rows],
                marker="o",
                linewidth=1.4,
                markersize=3.5,
                label=label.replace("fixed_dlambda_", "fix "),
            )
        ax.axhline(0.05, color="0.25", linestyle="--", linewidth=1.0, label="0.05 target")
        ax.set_title(target_prefix)
        ax.set_xlabel(r"$R_{\rm join}/r_g$")
        ax.set_ylabel(r"max $|\Delta \log u|, |\Delta \log T|$")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7)
    fig.tight_layout()
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_OUTPUT, dpi=180)
    plt.close(fig)


def write_pil_figure(rows: list[dict[str, object]]) -> None:
    from PIL import Image, ImageDraw

    ok_rows = [
        row
        for row in rows
        if row.get("status") in {"ok", "flow_partial"}
        and int(row.get("branch", -1)) == 1
        and str(row.get("target", "")).startswith("best_exact_bvp")
        and np.isfinite(float(row.get("distance_y", np.nan)))
    ]
    reach_rows = [
        row
        for row in rows
        if row.get("status") in {"flow_failed", "flow_partial", "ok"}
        and np.isfinite(float(row.get("flow_reached_R_rg", np.nan)))
    ]
    width, height = 1200, 640
    margin_l, margin_r, margin_t, margin_b = 80, 40, 60, 90
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((margin_l, 20), "Transonic branch-connection scan", fill=(20, 20, 20))
    plot_l, plot_t = margin_l, margin_t
    plot_r, plot_b = width - margin_r, height - margin_b
    draw.rectangle((plot_l, plot_t, plot_r, plot_b), outline=(40, 40, 40), width=1)
    if ok_rows:
        ymin = min(float(row["distance_y"]) for row in ok_rows)
        ymax = max(float(row["distance_y"]) for row in ok_rows)
        ymin = max(ymin * 0.6, 1.0e-8)
        ymax = max(ymax * 1.4, ymin * 10.0)
        xmin = min(float(row["R_join_rg"]) for row in ok_rows)
        xmax = max(float(row["R_join_rg"]) for row in ok_rows)
        pad = max(0.02, 0.08 * (xmax - xmin))
        xmin -= pad
        xmax += pad

        def sx(value: float) -> float:
            return plot_l + (value - xmin) / (xmax - xmin + 1.0e-300) * (plot_r - plot_l)

        def sy(value: float) -> float:
            log_min = np.log10(ymin)
            log_max = np.log10(ymax)
            return plot_b - (np.log10(value) - log_min) / (log_max - log_min + 1.0e-300) * (plot_b - plot_t)

        colors = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189), (255, 127, 14), (23, 190, 207), (127, 127, 127)]
        for frac in np.linspace(0.0, 1.0, 5):
            xtick = xmin + frac * (xmax - xmin)
            xpix = sx(xtick)
            draw.line((xpix, plot_b, xpix, plot_b + 5), fill=(20, 20, 20), width=1)
            draw.text((xpix - 16, plot_b + 10), f"{xtick:.2f}", fill=(20, 20, 20))
        for ytick in (1.0e-3, 1.0e-2, 5.0e-2, 1.0e-1, 1.0):
            if ymin <= ytick <= ymax:
                ypix = sy(ytick)
                draw.line((plot_l - 5, ypix, plot_l, ypix), fill=(20, 20, 20), width=1)
                draw.text((plot_l - 62, ypix - 7), f"{ytick:g}", fill=(20, 20, 20))
        labels = list(dict.fromkeys(str(row["case"]) for row in ok_rows))
        for label_index, label in enumerate(labels):
            case_rows = sorted([row for row in ok_rows if row["case"] == label], key=lambda item: float(item["R_join_rg"]))
            points = [(sx(float(row["R_join_rg"])), sy(float(row["distance_y"]))) for row in case_rows]
            color = colors[label_index % len(colors)]
            if len(points) >= 2:
                draw.line(points, fill=color, width=3)
            for x_value, y_value in points:
                draw.ellipse((x_value - 4, y_value - 4, x_value + 4, y_value + 4), fill=color)
            draw.text((plot_l + 10, plot_t + 18 * label_index + 8), label.replace("fixed_dlambda_", "fix "), fill=color)
        y_target = sy(0.05) if ymin <= 0.05 <= ymax else None
        if y_target is not None:
            draw.line((plot_l, y_target, plot_r, y_target), fill=(80, 80, 80), width=1)
            draw.text((plot_r - 80, y_target - 14), "0.05", fill=(80, 80, 80))
        draw.text((plot_l, plot_b + 30), "R_join / rg", fill=(20, 20, 20))
        draw.text((10, plot_t + 20), "distance_y", fill=(20, 20, 20))
    elif reach_rows:
        unique: dict[tuple[str, int], float] = {}
        for row in reach_rows:
            key = (str(row.get("case", "")), int(row.get("branch", -1)))
            unique[key] = max(unique.get(key, 0.0), float(row["flow_reached_R_rg"]))
        items = list(unique.items())
        ymax = max(value for _key, value in items) * 1.05
        ymin = min(float(row.get("Rson_rg", 0.0)) for row in reach_rows if np.isfinite(float(row.get("Rson_rg", np.nan))))
        bar_w = max(12, int((plot_r - plot_l) / max(len(items), 1) * 0.55))
        for idx, ((label, branch), value) in enumerate(items):
            x = plot_l + (idx + 0.5) * (plot_r - plot_l) / max(len(items), 1)
            y = plot_b - (value - ymin) / (ymax - ymin + 1.0e-300) * (plot_b - plot_t)
            color = (31, 119, 180) if branch == 1 else (180, 80, 80)
            draw.rectangle((x - bar_w / 2, y, x + bar_w / 2, plot_b), fill=color)
            draw.text((x - bar_w, plot_b + 8), f"b{branch}", fill=(20, 20, 20))
            draw.text((x - bar_w, y - 16), f"{value:.3g}", fill=(20, 20, 20))
        draw.text((plot_l, plot_b + 38), "case/branch bars; label is max reached R/rg", fill=(20, 20, 20))
        draw.text((10, plot_t + 20), "reached R/rg", fill=(20, 20, 20))
    else:
        draw.text((plot_l + 20, plot_t + 20), "No finite branch-connection data.", fill=(160, 0, 0))
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(FIGURE_OUTPUT)


def summarize(rows: list[dict[str, object]]) -> None:
    ok_rows = [row for row in rows if row.get("status") in {"ok", "flow_partial"} and np.isfinite(float(row.get("distance_y", np.nan)))]
    print(f"wrote {TABLE_OUTPUT}", flush=True)
    print(f"wrote {TABLE_OUTPUT.with_suffix('.json')}", flush=True)
    if FIGURE_OUTPUT.exists():
        print(f"wrote {FIGURE_OUTPUT}", flush=True)
    for target_name in sorted({str(row.get("target", "")) for row in ok_rows}):
        subset = [row for row in ok_rows if row.get("target") == target_name]
        if not subset:
            continue
        best = min(subset, key=lambda row: float(row["distance_y"]))
        print(
            f"best {target_name}: case={best['case']} branch={best['branch']} "
            f"Rjoin={best['R_join_rg']:.4g} dist_y={best['distance_y']:.3e} "
            f"dlogu={best['delta_logu']:.3e} dlogT={best['delta_logT']:.3e} "
            f"critK={best['critK']:.3e}",
            flush=True,
        )
    if not ok_rows:
        reach_rows = [row for row in rows if np.isfinite(float(row.get("flow_reached_R_rg", np.nan)))]
        if reach_rows:
            best_reach = max(reach_rows, key=lambda row: float(row["flow_reached_R_rg"]))
            print(
                f"no finite join matches; farthest IVP reach case={best_reach['case']} "
                f"branch={best_reach['branch']} reached_R={best_reach['flow_reached_R_rg']:.6g} rg "
                f"message={best_reach.get('message', '')}",
                flush=True,
            )


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    ctx = load_context()
    source_x, source_meta = load_row(SOURCE_CHECKPOINT)
    targets: list[TargetProfile] = []
    if "best_exact_bvp" in TARGET_NAMES:
        targets.append(load_best_target(fiducial, source_meta, mdot_edd))
    if "dynamic_patch_source" in TARGET_NAMES:
        targets.append(load_dynamic_target(ctx, source_x))
    if not targets:
        raise RuntimeError("no target profiles selected")
    cases = critical_cases(ctx)
    print(f"loaded {len(cases)} critical cases and {len(targets)} targets", flush=True)
    rows = scan_rows(ctx, cases, targets)
    write_table(rows)
    write_json(rows)
    write_figure(rows)
    summarize(rows)


if __name__ == "__main__":
    main()
