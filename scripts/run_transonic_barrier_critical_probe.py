"""Probe whether the near-tail singular barrier has a compatible critical point."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from imri_qpe.layer3_minidisk_1d.transonic_local import algebraic_state, sonic_diagnostics

from run_transonic_branch_connection_scan import solve_fixed_dlambda
from run_transonic_branch_barrier_audit import POINTS_OUTPUT, critical_case_from_free_fit
from run_transonic_outer_slope_calibration_audit import fmt
from run_transonic_sonic_reverse_fit import load_context


ROOT = Path(__file__).resolve().parents[1]
TABLE_OUTPUT = ROOT / os.environ.get(
    "IMBH_BARRIER_CRITICAL_PROBE_TABLE",
    "outputs/tables/transonic_barrier_critical_probe.md",
)
FIXED_DLAMBDA = float(os.environ.get("IMBH_BARRIER_CRITICAL_FIXED_DLAMBDA", "-0.00153"))
LOGR_HALF_WIDTH = float(os.environ.get("IMBH_BARRIER_CRITICAL_LOGR_HALF_WIDTH", "0.05"))
Y_HALF_WIDTH = float(os.environ.get("IMBH_BARRIER_CRITICAL_Y_HALF_WIDTH", "0.8"))
LAMBDA_HALF_WIDTH = float(os.environ.get("IMBH_BARRIER_CRITICAL_LAMBDA_HALF_WIDTH", "0.02"))
MAX_NFEV = int(os.environ.get("IMBH_BARRIER_CRITICAL_MAX_NFEV", "300"))
RESIDUAL_SCALE = float(os.environ.get("IMBH_BARRIER_CRITICAL_RESIDUAL_SCALE", "1e-4"))


def load_seed_rows() -> list[dict[str, object]]:
    if not POINTS_OUTPUT.exists():
        raise RuntimeError(f"missing barrier points file {POINTS_OUTPUT}")
    rows = json.loads(POINTS_OUTPUT.read_text())
    seeds: list[dict[str, object]] = []
    for case in ("free_reverse_fit", "fixed_dlambda_m0p00153"):
        for branch in (0, 1):
            subset = [
                row
                for row in rows
                if row.get("case") == case
                and int(row.get("branch", -1)) == branch
                and np.isfinite(float(row.get("R_rg", np.nan)))
            ]
            if subset:
                seeds.append(subset[-1])
    return seeds


def case_params(ctx, seed: dict[str, object]):
    if seed["case"] == "free_reverse_fit":
        return critical_case_from_free_fit(ctx)
    if seed["case"] == "fixed_dlambda_m0p00153":
        return solve_fixed_dlambda(ctx, FIXED_DLAMBDA)
    raise ValueError(f"unknown case {seed['case']!r}")


def residual(trial: np.ndarray, params, fixed_lambda0: float | None = None) -> np.ndarray:
    if fixed_lambda0 is None:
        logR, logu, logT, lambda0 = map(float, trial)
    else:
        logR, logu, logT = map(float, trial)
        lambda0 = float(fixed_lambda0)
    try:
        sonic = sonic_diagnostics(logR, np.array([logu, logT], dtype=float), lambda0, params.physics)
        return np.array([sonic.D, sonic.C1, sonic.C2, sonic.compatibility], dtype=float) / RESIDUAL_SCALE
    except Exception:
        return np.full(4, 1.0e8)


def solve_probe(ctx, seed: dict[str, object], *, free_lambda: bool) -> dict[str, object]:
    case = case_params(ctx, seed)
    lambda_seed = float(seed["lambda0"])
    if free_lambda:
        x0 = np.array(
            [
                float(seed["logR"]),
                float(seed["logu"]),
                float(seed["logT"]),
                lambda_seed,
            ],
            dtype=float,
        )
        lower = np.array(
            [
                x0[0] - LOGR_HALF_WIDTH,
                x0[1] - Y_HALF_WIDTH,
                x0[2] - Y_HALF_WIDTH,
                x0[3] - LAMBDA_HALF_WIDTH,
            ],
            dtype=float,
        )
        upper = np.array(
            [
                x0[0] + LOGR_HALF_WIDTH,
                x0[1] + Y_HALF_WIDTH,
                x0[2] + Y_HALF_WIDTH,
                x0[3] + LAMBDA_HALF_WIDTH,
            ],
            dtype=float,
        )
        x_scale = np.array([0.02, 0.2, 0.2, 0.005], dtype=float)
        fun = lambda trial: residual(trial, ctx.params, None)
    else:
        x0 = np.array([float(seed["logR"]), float(seed["logu"]), float(seed["logT"])], dtype=float)
        lower = np.array([x0[0] - LOGR_HALF_WIDTH, x0[1] - Y_HALF_WIDTH, x0[2] - Y_HALF_WIDTH], dtype=float)
        upper = np.array([x0[0] + LOGR_HALF_WIDTH, x0[1] + Y_HALF_WIDTH, x0[2] + Y_HALF_WIDTH], dtype=float)
        x_scale = np.array([0.02, 0.2, 0.2], dtype=float)
        fun = lambda trial: residual(trial, ctx.params, lambda_seed)
    result = least_squares(
        fun,
        x0,
        bounds=(lower, upper),
        x_scale=x_scale,
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-12,
        max_nfev=MAX_NFEV,
    )
    if free_lambda:
        logR, logu, logT, lambda0 = map(float, result.x)
    else:
        logR, logu, logT = map(float, result.x)
        lambda0 = lambda_seed
    sonic = sonic_diagnostics(logR, np.array([logu, logT], dtype=float), lambda0, ctx.params.physics)
    state = algebraic_state(logR, logu, logT, lambda0, ctx.params.physics)
    values = np.array([sonic.D, sonic.C1, sonic.C2, sonic.compatibility], dtype=float)
    return {
        "case": seed["case"],
        "branch": int(seed["branch"]),
        "mode": "free_lambda" if free_lambda else "fixed_lambda",
        "success": bool(result.success),
        "nfev": int(result.nfev),
        "message": str(result.message),
        "initial_R_rg": float(seed["R_rg"]),
        "initial_logu": float(seed["logu"]),
        "initial_logT": float(seed["logT"]),
        "initial_lambda0": float(seed["lambda0"]),
        "R_rg": float(np.exp(logR) / ctx.params.r_g),
        "logu": logu,
        "logT": logT,
        "lambda0": lambda0,
        "delta_logR": float(logR - float(seed["logR"])),
        "delta_logu": float(logu - float(seed["logu"])),
        "delta_logT": float(logT - float(seed["logT"])),
        "delta_lambda0": float(lambda0 - lambda_seed),
        "D": float(sonic.D),
        "C1": float(sonic.C1),
        "C2": float(sonic.C2),
        "K": float(sonic.compatibility),
        "critK": float(np.max(np.abs(values))),
        "smin_over_smax": float(sonic.smin_over_smax),
        "M_eff": float(sonic.M_eff),
        "H_R": float(state.H_over_R),
        "Omega_over_K": float(state.Omega / state.Omega_K),
        "candidate_regular": bool(np.max(np.abs(values)) <= 1.0e-6),
        "same_neighborhood": bool(abs(logR - x0[0]) < 0.03 and abs(logu - x0[1]) < 0.4 and abs(logT - x0[2]) < 0.4),
        "source_Rson_rg": float(np.exp(case.logR_son) / ctx.params.r_g),
        "source_critK": float(case.critK),
    }


def write_table(rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Barrier Critical-Compatibility Probe",
        "",
        "Generated by `scripts/run_transonic_barrier_critical_probe.py`.",
        "",
        f"Bounds: `dlogR={LOGR_HALF_WIDTH:g}`, `dy={Y_HALF_WIDTH:g}`, `dlambda={LAMBDA_HALF_WIDTH:g}`.",
        "",
        "| case | branch | mode | candidate | local | critK | D | C1 | C2 | K | R/rg | dlogR | dlogu | dlogT | dlambda | H/R | Omega/K | M_eff | smin/smax | nfev | message |",
        "|---|---:|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {case} | {branch} | {mode} | {candidate_regular} | {same_neighborhood} | {critK} | {D} | {C1} | "
            "{C2} | {K} | {R_rg} | {delta_logR} | {delta_logu} | {delta_logT} | {delta_lambda0} | "
            "{H_R} | {Omega_over_K} | {M_eff} | {smin_over_smax} | {nfev} | {message} |".format(
                case=row["case"],
                branch=row["branch"],
                mode=row["mode"],
                candidate_regular="yes" if row["candidate_regular"] else "no",
                same_neighborhood="yes" if row["same_neighborhood"] else "no",
                critK=fmt(float(row["critK"])),
                D=fmt(float(row["D"])),
                C1=fmt(float(row["C1"])),
                C2=fmt(float(row["C2"])),
                K=fmt(float(row["K"])),
                R_rg=fmt(float(row["R_rg"])),
                delta_logR=fmt(float(row["delta_logR"])),
                delta_logu=fmt(float(row["delta_logu"])),
                delta_logT=fmt(float(row["delta_logT"])),
                delta_lambda0=fmt(float(row["delta_lambda0"])),
                H_R=fmt(float(row["H_R"])),
                Omega_over_K=fmt(float(row["Omega_over_K"])),
                M_eff=fmt(float(row["M_eff"])),
                smin_over_smax=fmt(float(row["smin_over_smax"])),
                nfev=row["nfev"],
                message=str(row["message"]).replace("|", "/"),
            )
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")


def main() -> None:
    ctx = load_context()
    rows = []
    for seed in load_seed_rows():
        rows.append(solve_probe(ctx, seed, free_lambda=True))
        rows.append(solve_probe(ctx, seed, free_lambda=False))
    write_table(rows)
    print(f"wrote {TABLE_OUTPUT}", flush=True)
    for row in rows:
        print(
            f"{row['case']} branch={row['branch']} mode={row['mode']} candidate={row['candidate_regular']} "
            f"local={row['same_neighborhood']} critK={row['critK']:.3e} R={row['R_rg']:.6g}",
            flush=True,
        )


if __name__ == "__main__":
    main()
