"""Flow-map probe from the compatible second-critical candidates."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator

from imri_qpe.layer3_minidisk_1d.transonic_local import algebraic_state, local_ode_rhs, sonic_derivative_branches

from run_transonic_barrier_critical_probe import load_seed_rows, solve_probe
from run_transonic_branch_barrier_audit import POINTS_OUTPUT
from run_transonic_outer_slope_calibration_audit import fmt
from run_transonic_sonic_reverse_fit import load_context
from run_transonic_two_domain_sonic_flowmap import LHOPITAL_EPS


ROOT = Path(__file__).resolve().parents[1]
TABLE_OUTPUT = ROOT / os.environ.get(
    "IMBH_SECOND_CRITICAL_FLOWMAP_TABLE",
    "outputs/tables/transonic_second_critical_flowmap_probe.md",
)
EPS0 = float(os.environ.get("IMBH_SECOND_CRITICAL_EPS0", "1e-6"))
RTOL = float(os.environ.get("IMBH_SECOND_CRITICAL_RTOL", "1e-8"))
ATOL = float(os.environ.get("IMBH_SECOND_CRITICAL_ATOL", "1e-10"))
MAX_STEP = float(os.environ.get("IMBH_SECOND_CRITICAL_MAX_STEP", "5e-5"))
BRANCH_HALF_WIDTH = float(os.environ.get("IMBH_SECOND_CRITICAL_BRANCH_HALF_WIDTH", "1000"))
BRANCH_POINTS = int(os.environ.get("IMBH_SECOND_CRITICAL_BRANCH_POINTS", "2001"))
MATCH_R_RG = float(os.environ.get("IMBH_SECOND_CRITICAL_MATCH_R_RG", "6.1"))
OUT_R_RG = float(os.environ.get("IMBH_SECOND_CRITICAL_OUT_R_RG", "6.5"))


def incoming_interpolators() -> tuple[PchipInterpolator, PchipInterpolator]:
    rows = json.loads(POINTS_OUTPUT.read_text())
    subset = [
        row
        for row in rows
        if row.get("case") == "free_reverse_fit"
        and int(row.get("branch", -1)) == 1
        and np.isfinite(float(row.get("logR", np.nan)))
    ]
    if not subset:
        raise RuntimeError("missing incoming free_reverse_fit branch=1 barrier-audit points")
    logR = np.asarray([float(row["logR"]) for row in subset], dtype=float)
    logu = np.asarray([float(row["logu"]) for row in subset], dtype=float)
    logT = np.asarray([float(row["logT"]) for row in subset], dtype=float)
    order = np.argsort(logR)
    return (
        PchipInterpolator(logR[order], logu[order], extrapolate=True),
        PchipInterpolator(logR[order], logT[order], extrapolate=True),
    )


def integrate_from_critical(logR_c: float, y_c: np.ndarray, lambda0: float, branch, params, direction: str, target_logR: float):
    sign = 1.0 if direction == "out" else -1.0
    x0 = float(logR_c + sign * EPS0)
    y0 = np.asarray(y_c, dtype=float) + sign * EPS0 * np.asarray(branch.gradient, dtype=float)

    def rhs(x_value: float, y_value: np.ndarray) -> np.ndarray:
        return local_ode_rhs(float(x_value), y_value, lambda0, params.physics)

    try:
        sol = solve_ivp(
            rhs,
            (x0, target_logR),
            y0,
            method="Radau",
            dense_output=True,
            max_step=MAX_STEP,
            rtol=RTOL,
            atol=ATOL,
        )
        reached_logR = float(sol.t[-1]) if sol.t.size else np.nan
        if sol.sol is not None:
            if direction == "out":
                can_eval = target_logR <= reached_logR + 1.0e-12
            else:
                can_eval = target_logR >= reached_logR - 1.0e-12
            y_target = np.asarray(sol.sol(target_logR), dtype=float) if can_eval else np.array([np.nan, np.nan])
        else:
            y_target = np.array([np.nan, np.nan])
        return {
            "success": bool(sol.success),
            "message": str(sol.message),
            "nfev": int(sol.nfev),
            "reached_R_rg": float(np.exp(reached_logR) / params.r_g) if np.isfinite(reached_logR) else np.nan,
            "target_y": y_target,
        }
    except Exception as exc:
        return {
            "success": False,
            "message": str(exc),
            "nfev": 0,
            "reached_R_rg": np.nan,
            "target_y": np.array([np.nan, np.nan]),
        }


def run_probe() -> list[dict[str, object]]:
    ctx = load_context()
    interp_u, interp_T = incoming_interpolators()
    match_logR = float(np.log(MATCH_R_RG * ctx.params.r_g))
    out_logR = float(np.log(OUT_R_RG * ctx.params.r_g))
    incoming_y = np.array([float(interp_u(match_logR)), float(interp_T(match_logR))], dtype=float)
    rows: list[dict[str, object]] = []
    for seed in load_seed_rows():
        if seed.get("case") != "free_reverse_fit" or int(seed.get("branch", -1)) != 1:
            continue
        for free_lambda in (False, True):
            candidate = solve_probe(ctx, seed, free_lambda=free_lambda)
            y_c = np.array([float(candidate["logu"]), float(candidate["logT"])], dtype=float)
            logR_c = float(np.log(float(candidate["R_rg"]) * ctx.params.r_g))
            lambda0 = float(candidate["lambda0"])
            branches = sonic_derivative_branches(
                logR_c,
                y_c,
                lambda0,
                ctx.params.physics,
                eps=LHOPITAL_EPS,
                form="scaled",
                half_width=BRANCH_HALF_WIDTH,
                scan_points=BRANCH_POINTS,
            )
            if not branches:
                rows.append(
                    {
                        "candidate_mode": "free_lambda" if free_lambda else "fixed_lambda",
                        "candidate_R_rg": float(candidate["R_rg"]),
                        "candidate_lambda0": lambda0,
                        "candidate_critK": float(candidate["critK"]),
                        "second_branch": -1,
                        "a": np.nan,
                        "g_u": np.nan,
                        "g_T": np.nan,
                        "L": np.nan,
                        "in_success": False,
                        "in_reached_R_rg": np.nan,
                        "in_nfev": 0,
                        "in_message": "no L'Hopital derivative branches found",
                        "match_R_rg": MATCH_R_RG,
                        "match_dlogu": np.nan,
                        "match_dlogT": np.nan,
                        "match_distance": np.nan,
                        "out_success": False,
                        "out_reached_R_rg": np.nan,
                        "out_nfev": 0,
                        "out_message": "no L'Hopital derivative branches found",
                        "out_H_R": np.nan,
                        "out_Omega_over_K": np.nan,
                    }
                )
                continue
            for branch_index, branch in enumerate(branches):
                inward = integrate_from_critical(logR_c, y_c, lambda0, branch, ctx.params, "in", match_logR)
                outward = integrate_from_critical(logR_c, y_c, lambda0, branch, ctx.params, "out", out_logR)
                inward_y = np.asarray(inward["target_y"], dtype=float)
                dy = inward_y - incoming_y
                try:
                    state = algebraic_state(out_logR, float(outward["target_y"][0]), float(outward["target_y"][1]), lambda0, ctx.params.physics)
                    out_H_R = float(state.H_over_R)
                    out_omega = float(state.Omega / state.Omega_K)
                except Exception:
                    out_H_R = np.nan
                    out_omega = np.nan
                rows.append(
                    {
                        "candidate_mode": "free_lambda" if free_lambda else "fixed_lambda",
                        "candidate_R_rg": float(candidate["R_rg"]),
                        "candidate_lambda0": lambda0,
                        "candidate_critK": float(candidate["critK"]),
                        "second_branch": branch_index,
                        "a": float(branch.a),
                        "g_u": float(branch.gradient[0]),
                        "g_T": float(branch.gradient[1]),
                        "L": float(branch.lhopital_normalized),
                        "in_success": bool(inward["success"]),
                        "in_reached_R_rg": float(inward["reached_R_rg"]),
                        "in_nfev": int(inward["nfev"]),
                        "in_message": str(inward["message"]),
                        "match_R_rg": MATCH_R_RG,
                        "match_dlogu": float(dy[0]) if np.all(np.isfinite(dy)) else np.nan,
                        "match_dlogT": float(dy[1]) if np.all(np.isfinite(dy)) else np.nan,
                        "match_distance": float(np.max(np.abs(dy))) if np.all(np.isfinite(dy)) else np.nan,
                        "out_success": bool(outward["success"]),
                        "out_reached_R_rg": float(outward["reached_R_rg"]),
                        "out_nfev": int(outward["nfev"]),
                        "out_message": str(outward["message"]),
                        "out_H_R": out_H_R,
                        "out_Omega_over_K": out_omega,
                    }
                )
    return rows


def write_table(rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Second-Critical Flow-Map Probe",
        "",
        "Generated by `scripts/run_transonic_second_critical_flowmap_probe.py`.",
        "",
        f"Inward match target: `{MATCH_R_RG:g} rg`; outward target: `{OUT_R_RG:g} rg`.",
        "",
        "| mode | Rcrit/rg | lambda0 | branch | a | g_u | g_T | in ok | in reached | match dist | dlogu | dlogT | out ok | out reached | out H/R | out Omega/K | L | messages |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {candidate_mode} | {candidate_R_rg} | {candidate_lambda0} | {second_branch} | {a} | {g_u} | "
            "{g_T} | {in_success} | {in_reached_R_rg} | {match_distance} | {match_dlogu} | {match_dlogT} | "
            "{out_success} | {out_reached_R_rg} | {out_H_R} | {out_Omega_over_K} | {L} | {messages} |".format(
                candidate_mode=row["candidate_mode"],
                candidate_R_rg=fmt(float(row["candidate_R_rg"])),
                candidate_lambda0=fmt(float(row["candidate_lambda0"])),
                second_branch=row["second_branch"],
                a=fmt(float(row["a"])),
                g_u=fmt(float(row["g_u"])),
                g_T=fmt(float(row["g_T"])),
                in_success="yes" if row["in_success"] else "no",
                in_reached_R_rg=fmt(float(row["in_reached_R_rg"])),
                match_distance=fmt(float(row["match_distance"])),
                match_dlogu=fmt(float(row["match_dlogu"])),
                match_dlogT=fmt(float(row["match_dlogT"])),
                out_success="yes" if row["out_success"] else "no",
                out_reached_R_rg=fmt(float(row["out_reached_R_rg"])),
                out_H_R=fmt(float(row["out_H_R"])),
                out_Omega_over_K=fmt(float(row["out_Omega_over_K"])),
                L=fmt(float(row["L"])),
                messages=(str(row["in_message"]) + " / " + str(row["out_message"])).replace("|", "/"),
            )
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")


def main() -> None:
    rows = run_probe()
    write_table(rows)
    print(f"wrote {TABLE_OUTPUT}", flush=True)
    for row in rows:
        print(
            f"{row['candidate_mode']} branch={row['second_branch']} "
            f"match={row['match_distance']:.3e} out={row['out_success']} "
            f"out_reach={row['out_reached_R_rg']:.6g}",
            flush=True,
        )


if __name__ == "__main__":
    main()
