"""Reverse local fit from the dynamic buffer to a nearby sonic point."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from imri_qpe.layer3_minidisk_1d.transonic_local import (
    algebraic_state,
    local_ode_rhs,
    sonic_derivative_branches,
    sonic_diagnostics,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot

from run_transonic_outer_slope_calibration_audit import fmt
from run_transonic_two_domain_dynamic_sonic_patch import CHECKPOINT_DIR as DYNAMIC_CHECKPOINT_DIR, load_row
from run_transonic_two_domain_sonic_refinement_sprint import buffer_inner_grid, make_buffer_params, unpack_buffer


ROOT = Path(__file__).resolve().parents[1]
TABLE_OUTPUT = ROOT / "outputs" / "tables" / "transonic_sonic_reverse_fit.md"
SOURCE_CHECKPOINT = DYNAMIC_CHECKPOINT_DIR / os.environ.get("IMBH_REVERSE_SOURCE", "Nreg64_0p90277664.npz")
BUFFER_INDEX = int(os.environ.get("IMBH_REVERSE_BUFFER_INDEX", "1"))
EPS_INFER = float(os.environ.get("IMBH_REVERSE_EPS_INFER", "1e-6"))
FLOW_EPS0 = float(os.environ.get("IMBH_REVERSE_FLOW_EPS0", "1e-6"))
MAX_STEP = float(os.environ.get("IMBH_REVERSE_MAX_STEP", "2e-5"))
RTOL = float(os.environ.get("IMBH_REVERSE_RTOL", "1e-9"))
ATOL = float(os.environ.get("IMBH_REVERSE_ATOL", "1e-11"))
SONIC_SCALE = float(os.environ.get("IMBH_REVERSE_SONIC_SCALE", "1e-6"))
MAX_NFEV = int(os.environ.get("IMBH_REVERSE_MAX_NFEV", "120"))
BRANCH_HALF_WIDTH = float(os.environ.get("IMBH_REVERSE_BRANCH_HALF_WIDTH", "1000"))
BRANCH_POINTS = int(os.environ.get("IMBH_REVERSE_BRANCH_POINTS", "2001"))
INITIAL_DX = float(os.environ.get("IMBH_REVERSE_INITIAL_DX", "9.5e-4"))
INITIAL_DLAMBDA = float(os.environ.get("IMBH_REVERSE_INITIAL_DLAMBDA", "-1.53e-3"))
SOFT_SIGMAS = tuple(float(piece) for piece in os.environ.get("IMBH_REVERSE_SOFT_SIGMAS", "1e-3,3e-3,1e-2").replace(":", ",").split(",") if piece.strip())


@dataclass(frozen=True)
class ReverseContext:
    params: object
    logR_buffer: float
    y_buffer: np.ndarray
    logR_son_old: float
    lambda0_old: float
    source_meta: dict[str, object]


def json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def load_context() -> ReverseContext:
    x_source, meta = load_row(SOURCE_CHECKPOINT)
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    params = make_buffer_params(
        fiducial,
        float(meta["ratio"]),
        mdot_edd,
        int(meta["n_regular"]),
        int(meta["n_outer"]),
        float(meta["R_far_rg"]),
        float(meta["delta_s"]),
    )
    logu_i, logT_i, _logu_o, _logT_o, logR_son, lambda0 = unpack_buffer(x_source, params)
    logR_i = buffer_inner_grid(logR_son, params)
    if BUFFER_INDEX <= 0 or BUFFER_INDEX >= len(logR_i):
        raise ValueError("BUFFER_INDEX must select a regular buffer node")
    return ReverseContext(
        params=params,
        logR_buffer=float(logR_i[BUFFER_INDEX]),
        y_buffer=np.array([logu_i[BUFFER_INDEX], logT_i[BUFFER_INDEX]], dtype=float),
        logR_son_old=float(logR_son),
        lambda0_old=float(lambda0),
        source_meta=meta,
    )


def integrate_to_offset(logR_son: float, y_buffer: np.ndarray, lambda0: float, ctx: ReverseContext):
    if not logR_son + EPS_INFER < ctx.logR_buffer:
        raise ValueError("sonic point must be below buffer by more than EPS_INFER")
    x_offset = logR_son + EPS_INFER

    def rhs(x_value: float, y_value: np.ndarray) -> np.ndarray:
        return local_ode_rhs(float(x_value), y_value, lambda0, ctx.params.physics)

    sol = solve_ivp(
        rhs,
        (ctx.logR_buffer, x_offset),
        np.asarray(y_buffer, dtype=float),
        method="Radau",
        max_step=MAX_STEP,
        rtol=RTOL,
        atol=ATOL,
    )
    if (not sol.success) or sol.y.shape[1] == 0 or not np.all(np.isfinite(sol.y[:, -1])):
        raise RuntimeError(sol.message)
    y_offset = np.asarray(sol.y[:, -1], dtype=float)
    g_offset = local_ode_rhs(x_offset, y_offset, lambda0, ctx.params.physics)
    y_s = y_offset - EPS_INFER * g_offset
    return y_s, y_offset, g_offset, sol


def unpack_trial(trial: np.ndarray, mode: str) -> tuple[float, float, float, float]:
    dx = float(trial[0])
    dlambda = float(trial[1])
    dlogu = float(trial[2]) if mode == "soft_buffer" else 0.0
    dlogT = float(trial[3]) if mode == "soft_buffer" else 0.0
    return dx, dlambda, dlogu, dlogT


def evaluate_trial(trial: np.ndarray, ctx: ReverseContext, mode: str):
    dx, dlambda, dlogu, dlogT = unpack_trial(trial, mode)
    logR_son = ctx.logR_buffer - dx
    lambda0 = ctx.lambda0_old + dlambda
    y_buffer = ctx.y_buffer + np.array([dlogu, dlogT], dtype=float)
    y_s, y_offset, g_offset, sol = integrate_to_offset(logR_son, y_buffer, lambda0, ctx)
    sonic = sonic_diagnostics(logR_son, y_s, lambda0, ctx.params.physics)
    state = algebraic_state(logR_son, float(y_s[0]), float(y_s[1]), lambda0, ctx.params.physics)
    return {
        "dx": dx,
        "dlambda": dlambda,
        "dlogu_b": dlogu,
        "dlogT_b": dlogT,
        "logR_son": logR_son,
        "Rson_rg": float(np.exp(logR_son) / ctx.params.r_g),
        "lambda0": lambda0,
        "y_buffer": y_buffer,
        "y_s": y_s,
        "y_offset": y_offset,
        "g_offset": g_offset,
        "sonic": sonic,
        "state": state,
        "nfev_ivp": int(sol.nfev),
        "steps_ivp": int(sol.y.shape[1]),
    }


def residual_vector(trial: np.ndarray, ctx: ReverseContext, mode: str, sigma: float | None) -> np.ndarray:
    try:
        info = evaluate_trial(trial, ctx, mode)
        sonic = info["sonic"]
        rows = np.array([sonic.D, sonic.C1, sonic.C2, sonic.compatibility], dtype=float) / SONIC_SCALE
        if mode == "soft_buffer":
            _dx, _dlambda, dlogu, dlogT = unpack_trial(trial, mode)
            assert sigma is not None
            rows = np.concatenate([rows, np.array([dlogu / sigma, dlogT / sigma], dtype=float)])
        return rows
    except Exception:
        size = 6 if mode == "soft_buffer" else 4
        return np.full(size, 1.0e8)


def solve_fit(ctx: ReverseContext, mode: str, sigma: float | None):
    if mode == "fixed_buffer":
        seed = np.array([INITIAL_DX, INITIAL_DLAMBDA], dtype=float)
        lower = np.array([max(EPS_INFER * 5.0, 1.0e-6), -1.0e-2], dtype=float)
        upper = np.array([5.0e-2, 5.0e-3], dtype=float)
        x_scale = np.array([1.0e-3, 1.0e-3], dtype=float)
    elif mode == "soft_buffer":
        seed = np.array([INITIAL_DX, INITIAL_DLAMBDA, 0.0, 0.0], dtype=float)
        lower = np.array([max(EPS_INFER * 5.0, 1.0e-6), -1.0e-2, -3.0e-2, -3.0e-2], dtype=float)
        upper = np.array([5.0e-2, 5.0e-3, 3.0e-2, 3.0e-2], dtype=float)
        x_scale = np.array([1.0e-3, 1.0e-3, 1.0e-3, 1.0e-3], dtype=float)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    result = least_squares(
        lambda trial: residual_vector(trial, ctx, mode, sigma),
        seed,
        bounds=(lower, upper),
        x_scale=x_scale,
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-12,
        max_nfev=MAX_NFEV,
    )
    return result, evaluate_trial(result.x, ctx, mode)


def flowmap_branch_audit(info: dict[str, object], ctx: ReverseContext) -> list[dict[str, object]]:
    logR_son = float(info["logR_son"])
    y_s = np.asarray(info["y_s"], dtype=float)
    lambda0 = float(info["lambda0"])
    y_target = np.asarray(info["y_buffer"], dtype=float)
    dx = ctx.logR_buffer - logR_son
    eps0 = min(FLOW_EPS0, 0.1 * dx)
    branches = sonic_derivative_branches(
        logR_son,
        y_s,
        lambda0,
        ctx.params.physics,
        eps=1.0e-5,
        form="scaled",
        half_width=BRANCH_HALF_WIDTH,
        scan_points=BRANCH_POINTS,
    )
    rows: list[dict[str, object]] = []
    for index, branch in enumerate(branches):
        y0 = y_s + eps0 * branch.gradient

        def rhs(x_value: float, y_value: np.ndarray) -> np.ndarray:
            return local_ode_rhs(float(x_value), y_value, lambda0, ctx.params.physics)

        try:
            sol = solve_ivp(
                rhs,
                (logR_son + eps0, ctx.logR_buffer),
                y0,
                method="Radau",
                max_step=MAX_STEP,
                rtol=RTOL,
                atol=ATOL,
            )
            success = bool(sol.success)
            message = str(sol.message)
            y_end = np.asarray(sol.y[:, -1], dtype=float)
            diff = y_end - y_target
            max_HR = max(
                algebraic_state(float(x_val), float(y_val[0]), float(y_val[1]), lambda0, ctx.params.physics).H_over_R
                for x_val, y_val in zip(sol.t, sol.y.T)
            )
            nfev = int(sol.nfev)
            n_steps = int(sol.y.shape[1])
        except Exception as exc:
            success = False
            message = str(exc)
            diff = np.array([np.nan, np.nan], dtype=float)
            max_HR = np.nan
            nfev = 0
            n_steps = 0
        rows.append(
            {
                "branch": index,
                "kind": branch.kind,
                "a": float(branch.a),
                "g_u": float(branch.gradient[0]),
                "g_T": float(branch.gradient[1]),
                "L": float(branch.lhopital_normalized),
                "flow_success": success,
                "flow_message": message,
                "flow_eps0": eps0,
                "flow_dx": dx,
                "flow_dlogu": float(diff[0]),
                "flow_dlogT": float(diff[1]),
                "flow_distance": float(np.max(np.abs(diff))) if np.all(np.isfinite(diff)) else np.nan,
                "flow_max_HR": float(max_HR),
                "flow_nfev": nfev,
                "flow_steps": n_steps,
            }
        )
    return rows


def fit_row(label: str, mode: str, sigma: float | None, result, info: dict[str, object], flow_rows: list[dict[str, object]]) -> dict[str, object]:
    sonic = info["sonic"]
    physical_sonic = max(abs(float(sonic.D)), abs(float(sonic.C1)), abs(float(sonic.C2)), abs(float(sonic.compatibility)))
    best_flow = min(flow_rows, key=lambda row: float(row["flow_distance"]) if np.isfinite(float(row["flow_distance"])) else np.inf) if flow_rows else {}
    return {
        "label": label,
        "mode": mode,
        "sigma": np.nan if sigma is None else float(sigma),
        "success": bool(result.success),
        "nfev": int(result.nfev),
        "cost": float(result.cost),
        "message": str(result.message),
        "dx": float(info["dx"]),
        "Rson_rg": float(info["Rson_rg"]),
        "x_minus_old_xs": float(info["logR_son"] - ctx_global.logR_son_old),
        "dlambda": float(info["dlambda"]),
        "lambda0": float(info["lambda0"]),
        "dlogu_b": float(info["dlogu_b"]),
        "dlogT_b": float(info["dlogT_b"]),
        "D": float(sonic.D),
        "C1": float(sonic.C1),
        "C2": float(sonic.C2),
        "K": float(sonic.compatibility),
        "physical_sonic": physical_sonic,
        "smin_over_smax": float(sonic.smin_over_smax),
        "M_eff": float(sonic.M_eff),
        "H_R": float(info["state"].H_over_R),
        "g_offset_u": float(info["g_offset"][0]),
        "g_offset_T": float(info["g_offset"][1]),
        "ivp_nfev": int(info["nfev_ivp"]),
        "ivp_steps": int(info["steps_ivp"]),
        "n_branches": len(flow_rows),
        "best_flow_branch": best_flow.get("branch", -1),
        "best_flow_distance": best_flow.get("flow_distance", np.nan),
        "best_flow_dlogu": best_flow.get("flow_dlogu", np.nan),
        "best_flow_dlogT": best_flow.get("flow_dlogT", np.nan),
        "best_flow_a": best_flow.get("a", np.nan),
        "best_flow_g_u": best_flow.get("g_u", np.nan),
        "best_flow_g_T": best_flow.get("g_T", np.nan),
        "best_flow_L": best_flow.get("L", np.nan),
        "best_flow_success": best_flow.get("flow_success", False),
        "best_flow_message": best_flow.get("flow_message", ""),
    }


def write_table(rows: list[dict[str, object]], branch_rows: list[dict[str, object]], ctx: ReverseContext) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Sonic Reverse Local Fit",
        "",
        "Generated by `scripts/run_transonic_sonic_reverse_fit.py`.",
        "",
        f"Source checkpoint: `{SOURCE_CHECKPOINT}`.",
        f"Old dynamic branch: `Rson={float(ctx.source_meta['Rson_rg']):.8g} rg`, `lambda0={float(ctx.source_meta['lambda0']):.8g}`, `delta_s={float(ctx.source_meta['delta_s']):g}`.",
        f"Buffer index `{BUFFER_INDEX}` at `R={np.exp(ctx.logR_buffer)/ctx.params.r_g:.8g} rg`; `EPS_INFER={EPS_INFER:g}`, `SONIC_SCALE={SONIC_SCALE:g}`.",
        "",
        "## Fit Summary",
        "",
        "| label | mode | sigma | success | physical sonic | D | C1 | C2 | K | Rson/rg | x-old xs | dx buffer | lambda0 | dlambda | dlogu_b | dlogT_b | smin/smax | M_eff | H/R | g_u offset | g_T offset | branches | best flow | branch | flow dlogu | flow dlogT | flow a | flow g_u | flow g_T | flow L | nfev | message |",
        "|---|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {mode} | {sigma} | {success} | {physical_sonic} | {D} | {C1} | {C2} | {K} | "
            "{Rson_rg} | {x_minus_old_xs} | {dx} | {lambda0} | {dlambda} | {dlogu_b} | {dlogT_b} | "
            "{smin_over_smax} | {M_eff} | {H_R} | {g_offset_u} | {g_offset_T} | {n_branches} | "
            "{best_flow_distance} | {best_flow_branch} | {best_flow_dlogu} | {best_flow_dlogT} | "
            "{best_flow_a} | {best_flow_g_u} | {best_flow_g_T} | {best_flow_L} | {nfev} | {message} |".format(
                label=row["label"],
                mode=row["mode"],
                sigma=fmt(float(row["sigma"])),
                success="yes" if row["success"] else "no",
                physical_sonic=fmt(float(row["physical_sonic"])),
                D=fmt(float(row["D"])),
                C1=fmt(float(row["C1"])),
                C2=fmt(float(row["C2"])),
                K=fmt(float(row["K"])),
                Rson_rg=fmt(float(row["Rson_rg"])),
                x_minus_old_xs=fmt(float(row["x_minus_old_xs"])),
                dx=fmt(float(row["dx"])),
                lambda0=fmt(float(row["lambda0"])),
                dlambda=fmt(float(row["dlambda"])),
                dlogu_b=fmt(float(row["dlogu_b"])),
                dlogT_b=fmt(float(row["dlogT_b"])),
                smin_over_smax=fmt(float(row["smin_over_smax"])),
                M_eff=fmt(float(row["M_eff"])),
                H_R=fmt(float(row["H_R"])),
                g_offset_u=fmt(float(row["g_offset_u"])),
                g_offset_T=fmt(float(row["g_offset_T"])),
                n_branches=row["n_branches"],
                best_flow_distance=fmt(float(row["best_flow_distance"])),
                best_flow_branch=row["best_flow_branch"],
                best_flow_dlogu=fmt(float(row["best_flow_dlogu"])),
                best_flow_dlogT=fmt(float(row["best_flow_dlogT"])),
                best_flow_a=fmt(float(row["best_flow_a"])),
                best_flow_g_u=fmt(float(row["best_flow_g_u"])),
                best_flow_g_T=fmt(float(row["best_flow_g_T"])),
                best_flow_L=fmt(float(row["best_flow_L"])),
                nfev=row["nfev"],
                message=str(row["message"]).replace("|", "/"),
            )
        )
    lines.extend(
        [
            "",
            "## Branch Flow-Map Audit",
            "",
            "| fit | branch | a | g_u | g_T | L | success | flow max | dlogu | dlogT | eps0 | dx | max H/R | nfev | message |",
            "|---|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in branch_rows:
        lines.append(
            "| {fit} | {branch} | {a} | {g_u} | {g_T} | {L} | {success} | {flow_distance} | "
            "{flow_dlogu} | {flow_dlogT} | {flow_eps0} | {flow_dx} | {flow_max_HR} | {flow_nfev} | {message} |".format(
                fit=row["fit"],
                branch=row["branch"],
                a=fmt(float(row["a"])),
                g_u=fmt(float(row["g_u"])),
                g_T=fmt(float(row["g_T"])),
                L=fmt(float(row["L"])),
                success="yes" if row["flow_success"] else "no",
                flow_distance=fmt(float(row["flow_distance"])),
                flow_dlogu=fmt(float(row["flow_dlogu"])),
                flow_dlogT=fmt(float(row["flow_dlogT"])),
                flow_eps0=fmt(float(row["flow_eps0"])),
                flow_dx=fmt(float(row["flow_dx"])),
                flow_max_HR=fmt(float(row["flow_max_HR"])),
                flow_nfev=row["flow_nfev"],
                message=str(row["flow_message"]).replace("|", "/"),
            )
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")


ctx_global: ReverseContext


def main() -> None:
    global ctx_global
    ctx_global = load_context()
    print(
        f"reverse fit source={SOURCE_CHECKPOINT.name} old_Rson={np.exp(ctx_global.logR_son_old)/ctx_global.params.r_g:.6g} "
        f"buffer_R={np.exp(ctx_global.logR_buffer)/ctx_global.params.r_g:.6g} old_lambda={ctx_global.lambda0_old:.6g}",
        flush=True,
    )
    fit_rows: list[dict[str, object]] = []
    branch_rows: list[dict[str, object]] = []

    run_specs: list[tuple[str, str, float | None]] = [("fixed_buffer", "fixed_buffer", None)]
    run_specs.extend((f"soft_sigma_{sigma:g}", "soft_buffer", sigma) for sigma in SOFT_SIGMAS)
    for label, mode, sigma in run_specs:
        result, info = solve_fit(ctx_global, mode, sigma)
        flow_rows = flowmap_branch_audit(info, ctx_global)
        for row in flow_rows:
            row["fit"] = label
            branch_rows.append(row)
        row = fit_row(label, mode, sigma, result, info, flow_rows)
        fit_rows.append(row)
        print(
            f"{label}: sonic={row['physical_sonic']:.3e} R={row['Rson_rg']:.6g} "
            f"lambda={row['lambda0']:.6g} flow={row['best_flow_distance']:.3e} "
            f"branches={row['n_branches']} success={row['success']}",
            flush=True,
        )

    write_table(fit_rows, branch_rows, ctx_global)
    print(f"wrote {TABLE_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
