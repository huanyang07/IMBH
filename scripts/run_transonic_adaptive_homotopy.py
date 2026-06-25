"""Adaptive staged continuation for the transonic low-to-moderate Mdot branch."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    TransonicSlimParams,
    collocation_residual,
    pack_state,
    remap_profile_to_new_sonic_grid,
    solve_low_mdot_transonic_homotopy,
    state_bounds,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot


ROOT = Path(__file__).resolve().parents[1]
TABLE_OUTPUT = ROOT / "outputs" / "tables" / "transonic_adaptive_homotopy.md"
USE_SECANT_PREDICTOR = False


def fmt(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.4g}"


def row_is_usable(status) -> bool:
    return bool(
        status.physically_valid
        or (
            status.optimizer_acceptable
            and status.equations_converged
            and status.sonic_regular
            and status.thin_limit_ok
            and status.active_bounds_clear
            and status.outer_thin
        )
    )


def state_from_profile(profile, params: TransonicSlimParams) -> np.ndarray:
    return pack_state(np.log(profile.u), np.log(profile.T), np.log(profile.sonic_radius), profile.lambda0)


def secant_initial_guess(accepted, params: TransonicSlimParams):
    if len(accepted) == 0:
        return None
    if len(accepted) == 1:
        return remap_profile_to_new_sonic_grid(accepted[-1]["profile"], params)

    previous = accepted[-2]
    current = accepted[-1]
    z_previous = remap_profile_to_new_sonic_grid(previous["profile"], params)
    z_current = remap_profile_to_new_sonic_grid(current["profile"], params)
    denominator = np.log(current["ratio"]) - np.log(previous["ratio"])
    if abs(denominator) < 1.0e-12:
        return z_current
    factor = (np.log(params.mdot_edd_ratio) - np.log(current["ratio"])) / denominator
    candidate = z_current + 0.5 * factor * (z_current - z_previous)
    lower, upper = state_bounds(params)
    candidate = np.clip(candidate, lower + 1.0e-12, upper - 1.0e-12)
    candidate_residual = float(np.max(np.abs(collocation_residual(candidate, params))))
    remap_residual = float(np.max(np.abs(collocation_residual(z_current, params))))
    if np.isfinite(candidate_residual) and candidate_residual <= 1.2 * remap_residual:
        return candidate
    return z_current


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    target_ratio = 0.1
    ratio = 1.0e-3
    step = 0.18
    min_step = 0.02
    max_step = 0.28
    previous_profile = None
    accepted_ratio = ratio
    accepted = []
    rows = []

    while ratio <= target_ratio * (1.0 + 1.0e-12):
        params = TransonicSlimParams(
            M2_g=fiducial.M2_g,
            Mdot_g_s=ratio * mdot_edd,
            alpha=fiducial.alpha_cool,
            n_nodes=18,
            R_out_rg=300.0,
            max_nfev=1200,
            residual_tol=3.0e-4,
        )
        initial_guess = secant_initial_guess(accepted, params) if USE_SECANT_PREDICTOR else None
        if initial_guess is None and previous_profile is not None:
            initial_guess = remap_profile_to_new_sonic_grid(previous_profile, params)
        result = solve_low_mdot_transonic_homotopy(
            params,
            initial_guess=initial_guess,
            max_nfev_per_stage=700,
            final_max_nfev=1200,
        )
        final = result.final_result
        profile = final.profile
        audit = final.residual_audit
        status = final.status
        usable = row_is_usable(status)
        rows.append(
            {
                "ratio": ratio,
                "step": step,
                "usable": usable,
                "physical": status.physically_valid,
                "optimizer_ok": status.optimizer_acceptable,
                "optimizer_raw": status.optimizer_converged,
                "equations": status.equations_converged,
                "sonic": status.sonic_regular,
                "lambda_ratio": audit.lambda0_over_lK_isco,
                "lambda0": profile.lambda0,
                "Rson_rg": profile.sonic_radius / params.r_g,
                "max_HR": float(np.max(profile.H_over_R)),
                "max_residual": final.max_residual,
                "interval_radial": audit.interval_radial_max,
                "outer_omega": audit.outer_omega,
                "sonic_D": audit.sonic_D,
                "sonic_C1": audit.sonic_C1,
                "sonic_C2": audit.sonic_C2,
                "nfev": final.nfev,
                "message": final.message,
            }
        )
        print(
            f"ratio={ratio:.5g} usable={usable} physical={status.physically_valid} "
            f"lambda/lK={audit.lambda0_over_lK_isco:.4g} max_res={final.max_residual:.4g} H/R={np.max(profile.H_over_R):.4g}",
            flush=True,
        )
        if usable:
            previous_profile = profile
            accepted.append(
                {
                    "ratio": ratio,
                    "profile": profile,
                    "z": state_from_profile(profile, params),
                }
            )
            accepted_ratio = ratio
            next_ratio = ratio * np.exp(step)
            step = min(max_step, 1.2 * step)
            ratio = next_ratio
        else:
            step *= 0.5
            if step < min_step:
                break
            ratio = accepted_ratio * np.exp(step)

    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Adaptive Transonic Homotopy Audit",
        "",
        "Generated by `scripts/run_transonic_adaptive_homotopy.py`.",
        "",
        "Rows are accepted for continuation when the hardened residual/sonic/thin checks pass, even if raw SciPy termination reached `max_nfev` after satisfying the residual criteria.",
        "",
        "| Mdot/Mdot_Edd | step dlnM | usable | physical | opt ok | opt raw | equations | sonic | lambda/lK_ISCO | lambda0 | R_son/r_g | max H/R | max residual | interval R | outer Omega | D | C1 | C2 | nfev | message |",
        "|---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {ratio} | {step} | {usable} | {physical} | {optimizer_ok} | {optimizer_raw} | "
            "{equations} | {sonic} | {lambda_ratio} | {lambda0} | {Rson_rg} | {max_HR} | "
            "{max_residual} | {interval_radial} | {outer_omega} | {sonic_D} | {sonic_C1} | "
            "{sonic_C2} | {nfev} | {message} |".format(
                ratio=fmt(row["ratio"]),
                step=fmt(row["step"]),
                usable="yes" if row["usable"] else "no",
                physical="yes" if row["physical"] else "no",
                optimizer_ok="yes" if row["optimizer_ok"] else "no",
                optimizer_raw="yes" if row["optimizer_raw"] else "no",
                equations="yes" if row["equations"] else "no",
                sonic="yes" if row["sonic"] else "no",
                lambda_ratio=fmt(row["lambda_ratio"]),
                lambda0=fmt(row["lambda0"]),
                Rson_rg=fmt(row["Rson_rg"]),
                max_HR=fmt(row["max_HR"]),
                max_residual=fmt(row["max_residual"]),
                interval_radial=fmt(row["interval_radial"]),
                outer_omega=fmt(row["outer_omega"]),
                sonic_D=fmt(row["sonic_D"]),
                sonic_C1=fmt(row["sonic_C1"]),
                sonic_C2=fmt(row["sonic_C2"]),
                nfev=row["nfev"],
                message=row["message"].replace("|", "/"),
            )
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {TABLE_OUTPUT}")


if __name__ == "__main__":
    main()
