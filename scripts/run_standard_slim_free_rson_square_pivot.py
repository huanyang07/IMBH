"""Free-Rson and square-sonic-pivot benchmark for the standard slim disk."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from imri_qpe.constants import C
from imri_qpe.layer3_minidisk_1d import (
    TransonicSlimParams,
    collocation_residual,
    residual_audit_from_state_vector,
    solve_square_transonic_polish,
    square_collocation_residual,
    transonic_profile_from_state_vector,
)
from imri_qpe.layer3_minidisk_1d.transonic_collocation import (
    _free_rson_jac_sparsity_pattern,
    _free_rson_jacobian_from_unknowns,
    _free_rson_residual_from_unknowns,
    _pack_profile_unknowns_to_state,
    _profile_unknown_bounds,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot
from run_standard_slim_analytic_seed_audit import (
    ALPHA,
    MDOT_RATIO,
    R_SON_RG,
    analytic_seed_state,
    fmt,
    json_safe,
    n_nodes_for,
    pressure_support_diagnostics,
)
from run_standard_slim_benchmark_thin_limit import thin_metrics
from run_standard_slim_fixed_eigen_profile import (
    profile_unknowns_from_state,
    solve_profile_stage,
)


ROOT = Path(__file__).resolve().parents[1]
TABLE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_FREE_RSON_TABLE",
    "outputs/tables/slim_benchmark_free_rson_square_pivot.md",
)
JSON_OUTPUT = TABLE_OUTPUT.with_suffix(".json")
FIGURE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_FREE_RSON_FIGURE",
    "outputs/figures/slim_benchmark_free_rson_square_pivot.png",
)

R_OUT_CASES = tuple(
    float(piece)
    for piece in os.environ.get("IMBH_STANDARD_SLIM_FREE_RSON_ROUTS", "300").replace(":", ",").split(",")
    if piece.strip()
)
STRESS_FACTORS = tuple(
    float(piece)
    for piece in os.environ.get("IMBH_STANDARD_SLIM_FREE_RSON_STRESS_FACTORS", "1.0").replace(":", ",").split(",")
    if piece.strip()
)
PIVOTS = tuple(
    piece.strip()
    for piece in os.environ.get("IMBH_STANDARD_SLIM_FREE_RSON_PIVOTS", "C1,C2").replace(":", ",").split(",")
    if piece.strip()
)
SOLVE_FORM = os.environ.get("IMBH_STANDARD_SLIM_FREE_RSON_SOLVE_FORM", "integrated")
SOLVE_WEIGHTING = os.environ.get("IMBH_STANDARD_SLIM_FREE_RSON_SOLVE_WEIGHTING", "inverse_sqrt_dx")
FIXED_PROFILE_MAX_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_FREE_RSON_FIXED_PROFILE_NFEV", "100"))
FREE_RSON_MAX_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_FREE_RSON_NFEV", "160"))
SQUARE_MAX_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_FREE_RSON_SQUARE_NFEV", "160"))
RUN_DIFFERENTIAL_SQUARE = os.environ.get("IMBH_STANDARD_SLIM_FREE_RSON_DIFFERENTIAL_SQUARE", "0") != "0"
USE_BLOCK_JACOBIAN = os.environ.get("IMBH_STANDARD_SLIM_FREE_RSON_BLOCK_JACOBIAN", "0") != "0"
RESIDUAL_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_FREE_RSON_RESIDUAL_TOL", "1e-7"))


def make_params(fiducial: FiducialParams, mdot_edd: float, R_out_rg: float, stress_factor: float, *, form: str, weighting: str) -> TransonicSlimParams:
    return TransonicSlimParams(
        M2_g=fiducial.M2_g,
        Mdot_g_s=MDOT_RATIO * mdot_edd,
        alpha=ALPHA,
        mu_stress=0.0,
        stress_factor=float(stress_factor),
        R_out_rg=float(R_out_rg),
        n_nodes=n_nodes_for(float(R_out_rg)),
        max_nfev=max(FREE_RSON_MAX_NFEV, SQUARE_MAX_NFEV),
        residual_tol=RESIDUAL_TOL,
        outer_closure="thin_value",
        interval_residual_form=form,
        integrated_residual_weighting=weighting,
    )


def solve_free_rson_stage(
    x0_profile: np.ndarray,
    logR_son0: float,
    params: TransonicSlimParams,
    fixed_lambda0: float,
):
    sonic_components = ("D",)
    profile_lower, profile_upper = _profile_unknown_bounds(params)
    state_lower, state_upper = _state_bounds_for_local_import(params)
    lower = np.concatenate([profile_lower, np.array([state_lower[-2]])])
    upper = np.concatenate([profile_upper, np.array([state_upper[-2]])])
    x0 = np.clip(
        np.concatenate([np.asarray(x0_profile, dtype=float), np.array([float(logR_son0)])]),
        lower + 1.0e-12,
        upper - 1.0e-12,
    )
    kwargs = (
        {
            "jac": lambda trial: _free_rson_jacobian_from_unknowns(
                trial,
                params,
                fixed_lambda0,
                sonic_components,
                sonic_weight=1.0,
            )
        }
        if USE_BLOCK_JACOBIAN
        else {"jac_sparsity": _free_rson_jac_sparsity_pattern(params, len(sonic_components))}
    )
    return least_squares(
        lambda trial: _free_rson_residual_from_unknowns(trial, params, fixed_lambda0, sonic_components, sonic_weight=1.0),
        x0,
        bounds=(lower, upper),
        x_scale="jac",
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=1.0e-10,
        max_nfev=FREE_RSON_MAX_NFEV,
        **kwargs,
    )


def _state_bounds_for_local_import(params: TransonicSlimParams) -> tuple[np.ndarray, np.ndarray]:
    from imri_qpe.layer3_minidisk_1d import state_bounds

    return state_bounds(params)


def z_from_free_rson_unknowns(unknowns: np.ndarray, params: TransonicSlimParams, lambda0: float) -> np.ndarray:
    return _pack_profile_unknowns_to_state(np.asarray(unknowns[: 2 * params.n_nodes], dtype=float), params, float(unknowns[-1]), lambda0)


def dominant_block(row: dict[str, object]) -> str:
    keys = (
        "selected_max",
        "differential_selected_max",
        "full_overdetermined_max",
        "square_C1_max",
        "square_C2_max",
        "interval_R_max",
        "interval_E_max",
        "outer_omega",
        "outer_energy",
        "D",
        "C1",
        "C2",
        "K",
    )
    values = {key: abs(float(row[key])) for key in keys if key in row}
    return max(values, key=values.get)


def row_for_z(
    *,
    label: str,
    solve_params: TransonicSlimParams,
    audit_params: TransonicSlimParams,
    z: np.ndarray,
    selected_residual: np.ndarray,
    differential_selected_residual: np.ndarray,
    stress_factor: float,
    pivot: str,
    nfev: int,
    success: bool,
    cost: float,
    optimality: float,
    message: str,
) -> dict[str, object]:
    audit = residual_audit_from_state_vector(z, audit_params)
    profile = transonic_profile_from_state_vector(z, audit_params)
    thin = thin_metrics(profile, audit_params)
    diag = pressure_support_diagnostics(z, audit_params)
    mask = np.ones(len(profile.R), dtype=bool)
    if len(mask) > 8:
        mask[:2] = False
        mask[-2:] = False
    square_c1 = square_collocation_residual(z, audit_params, pivot="C1")
    square_c2 = square_collocation_residual(z, audit_params, pivot="C2")
    row = {
        "label": label,
        "ratio": float(audit_params.mdot_edd_ratio),
        "stress_factor": float(stress_factor),
        "R_out_rg": float(audit_params.R_out_rg),
        "N": int(audit_params.n_nodes),
        "pivot": pivot,
        "solve_form": solve_params.interval_residual_form,
        "solve_weighting": solve_params.integrated_residual_weighting,
        "selected_max": float(np.max(np.abs(selected_residual))),
        "differential_selected_max": float(np.max(np.abs(differential_selected_residual))),
        "full_overdetermined_max": float(np.max(np.abs(collocation_residual(z, audit_params)))),
        "square_C1_max": float(np.max(np.abs(square_c1))),
        "square_C2_max": float(np.max(np.abs(square_c2))),
        "interval_R_max": float(audit.interval_radial_max),
        "interval_E_max": float(audit.interval_energy_max),
        "outer_omega": float(audit.outer_omega),
        "outer_energy": float(audit.outer_energy),
        "D": float(audit.sonic_D),
        "C1": float(audit.sonic_C1),
        "C2": float(audit.sonic_C2),
        "K": float(audit.sonic_K),
        "sonic_smin": float(audit.sonic_smin_over_smax),
        "Rson_rg": float(profile.sonic_radius / audit_params.r_g),
        "lambda0_over_lK_isco": float(audit.lambda0_over_lK_isco),
        "max_abs_omega": float(thin["max_abs_omega_frac"]),
        "max_abs_qadv_qvisc": float(thin["max_abs_qadv_qvisc"]),
        "max_abs_qbalance": float(thin["max_abs_qbalance_thin"]),
        "max_HR": float(thin["max_HR"]),
        "median_pw_nusigma_error": float(thin["median_pw_nusigma_error"]),
        "pressure_frac": float(np.max(np.abs(diag["pressure_frac"][mask]))),
        "inertia_frac": float(np.max(np.abs(diag["inertia_frac"][mask]))),
        "nfev": int(nfev),
        "success": bool(success),
        "cost": float(cost),
        "optimality": float(optimality),
        "message": str(message),
    }
    row["dominant"] = dominant_block(row)
    return row


def write_table(rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Standard Slim Free-Rson and Square-Pivot Benchmark",
        "",
        "Generated by `scripts/run_standard_slim_free_rson_square_pivot.py`.",
        "",
        "This continues the `Mdot/Edd=1e-3` no-wind benchmark from the analytic seed: fixed-eigen profile, free `Rson` with fixed `lambda0=lK_ISCO`, then square sonic pivots.",
        "",
        f"Defaults: `R_out={','.join(f'{value:g}' for value in R_OUT_CASES)} rg`, stress `{','.join(f'{value:g}' for value in STRESS_FACTORS)}`, pivots `{','.join(PIVOTS)}`, solve form `{SOLVE_FORM}`, weighting `{SOLVE_WEIGHTING}`, fixed profile nfev `{FIXED_PROFILE_MAX_NFEV}`, free-Rson nfev `{FREE_RSON_MAX_NFEV}`, square nfev `{SQUARE_MAX_NFEV}`.",
        "",
        "| label | stress | R_out/rg | N | pivot | selected | diff selected | full overdet | dominant | square C1 | square C2 | int R | int E | outer omega | outer E | D | C1 | C2 | K | Rson/rg | lambda/lK | Omega err | Qadv/Qvisc | qbalance | H/R | PW med err | p support | inertia | nfev | success | optimality | message |",
        "|---|---:|---:|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---|",
    ]
    for row in rows:
        formatted = {key: fmt(value) if isinstance(value, (float, int, np.floating, np.integer)) else value for key, value in row.items()}
        lines.append(
            "| {label} | {stress_factor} | {R_out_rg} | {N} | {pivot} | {selected_max} | {differential_selected_max} | "
            "{full_overdetermined_max} | {dominant} | {square_C1_max} | {square_C2_max} | {interval_R_max} | {interval_E_max} | "
            "{outer_omega} | {outer_energy} | {D} | {C1} | {C2} | {K} | {Rson_rg} | {lambda0_over_lK_isco} | "
            "{max_abs_omega} | {max_abs_qadv_qvisc} | {max_abs_qbalance} | {max_HR} | {median_pw_nusigma_error} | "
            "{pressure_frac} | {inertia_frac} | {nfev} | {success} | {optimality} | {message} |".format(**formatted)
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")
    JSON_OUTPUT.write_text(json.dumps([{key: json_safe(value) for key, value in row.items()} for row in rows], indent=2, sort_keys=True) + "\n")


def write_figure(rows: list[dict[str, object]]) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        print(f"skipping figure: PIL unavailable ({exc})", flush=True)
        return
    if not rows:
        return
    primary = [row for row in rows if float(row["R_out_rg"]) == min(float(item["R_out_rg"]) for item in rows)]
    labels = [str(row["label"]) if row["pivot"] == "-" else f"{row['label']} {row['pivot']}" for row in primary]
    values = np.asarray([max(abs(float(row["full_overdetermined_max"])), 1.0e-16) for row in primary], dtype=float)
    y_plot = np.log10(values)
    width, height = 1000, 520
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    x0, y0, x1, y1 = 90, 80, 940, 420
    draw.rectangle((x0, y0, x1, y1), outline=(60, 60, 60), width=2)
    draw.text((90, 25), "Free-Rson/square-pivot benchmark, log10 full overdetermined residual", fill=(20, 20, 20), font=font)
    y_min, y_max = float(np.min(y_plot)), float(np.max(y_plot))
    if y_max <= y_min:
        y_min -= 1.0
        y_max += 1.0
    pad = 0.08 * (y_max - y_min)
    y_min -= pad
    y_max += pad
    count = max(1, len(primary))
    bar_w = max(20, int((x1 - x0) / (2 * count)))
    for idx, (label, yy) in enumerate(zip(labels, y_plot)):
        cx = x0 + int((idx + 0.5) * (x1 - x0) / count)
        py = y1 - int((yy - y_min) / (y_max - y_min) * (y1 - y0))
        draw.rectangle((cx - bar_w // 2, py, cx + bar_w // 2, y1), fill=(31, 119, 180))
        draw.text((cx - bar_w, y1 + 10), label[:18], fill=(50, 50, 50), font=font)
    draw.text((x0 + 5, y0 + 5), f"{y_max:.2e}", fill=(80, 80, 80), font=font)
    draw.text((x0 + 5, y1 - 18), f"{y_min:.2e}", fill=(80, 80, 80), font=font)
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(FIGURE_OUTPUT)


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    rows: list[dict[str, object]] = []
    for stress_factor in STRESS_FACTORS:
        for R_out_rg in R_OUT_CASES:
            solve_params = make_params(fiducial, mdot_edd, R_out_rg, stress_factor, form=SOLVE_FORM, weighting=SOLVE_WEIGHTING)
            audit_params = replace(solve_params, interval_residual_form="differential", integrated_residual_weighting="none")
            lambda0 = float(audit_params.potential.l_k(audit_params.potential.r_isco) / (audit_params.r_g * C))
            fixed_logR_son = float(np.log(R_SON_RG * audit_params.r_g))
            print(f"case stress={stress_factor:g} R_out={R_out_rg:g} N={audit_params.n_nodes}", flush=True)
            z_seed, _arrays = analytic_seed_state(audit_params, fixed_logR_son, lambda0)
            fixed_profile = solve_profile_stage(
                profile_unknowns_from_state(z_seed, audit_params),
                solve_params,
                fixed_logR_son,
                lambda0,
                max_nfev=FIXED_PROFILE_MAX_NFEV,
            )
            z_fixed = _pack_profile_unknowns_to_state(fixed_profile.x, audit_params, fixed_logR_son, lambda0)
            fixed_selected = _free_rson_residual_from_unknowns(np.concatenate([fixed_profile.x, [fixed_logR_son]]), solve_params, lambda0, ("D",))
            fixed_diff = _free_rson_residual_from_unknowns(np.concatenate([fixed_profile.x, [fixed_logR_son]]), audit_params, lambda0, ("D",))
            rows.append(
                row_for_z(
                    label="fixed_eigen",
                    solve_params=solve_params,
                    audit_params=audit_params,
                    z=z_fixed,
                    selected_residual=fixed_selected,
                    differential_selected_residual=fixed_diff,
                    stress_factor=stress_factor,
                    pivot="-",
                    nfev=fixed_profile.nfev,
                    success=bool(fixed_profile.success),
                    cost=float(fixed_profile.cost),
                    optimality=float(fixed_profile.optimality),
                    message=str(fixed_profile.message),
                )
            )
            print(
                f"  fixed full={rows[-1]['full_overdetermined_max']:.3e} "
                f"D={rows[-1]['D']:.3e} C1={rows[-1]['C1']:.3e} C2={rows[-1]['C2']:.3e}",
                flush=True,
            )

            free_rson = solve_free_rson_stage(fixed_profile.x, fixed_logR_son, solve_params, lambda0)
            z_free = z_from_free_rson_unknowns(free_rson.x, audit_params, lambda0)
            free_selected = _free_rson_residual_from_unknowns(free_rson.x, solve_params, lambda0, ("D",))
            free_diff = _free_rson_residual_from_unknowns(free_rson.x, audit_params, lambda0, ("D",))
            rows.append(
                row_for_z(
                    label="free_Rson_fixed_lambda",
                    solve_params=solve_params,
                    audit_params=audit_params,
                    z=z_free,
                    selected_residual=free_selected,
                    differential_selected_residual=free_diff,
                    stress_factor=stress_factor,
                    pivot="-",
                    nfev=free_rson.nfev,
                    success=bool(free_rson.success),
                    cost=float(free_rson.cost),
                    optimality=float(free_rson.optimality),
                    message=str(free_rson.message),
                )
            )
            print(
                f"  free Rson={rows[-1]['Rson_rg']:.4g} full={rows[-1]['full_overdetermined_max']:.3e} "
                f"D={rows[-1]['D']:.3e} C1={rows[-1]['C1']:.3e} C2={rows[-1]['C2']:.3e}",
                flush=True,
            )

            for pivot in PIVOTS:
                square = solve_square_transonic_polish(
                    solve_params,
                    z_free,
                    pivot=pivot,
                    method="least_squares",
                    max_nfev=SQUARE_MAX_NFEV,
                    residual_tol=RESIDUAL_TOL,
                    use_block_jacobian=USE_BLOCK_JACOBIAN,
                )
                square_selected = square_collocation_residual(square.z, solve_params, pivot=pivot)
                square_diff = square_collocation_residual(square.z, audit_params, pivot=pivot)
                rows.append(
                    row_for_z(
                        label="square_integrated",
                        solve_params=solve_params,
                        audit_params=audit_params,
                        z=square.z,
                        selected_residual=square_selected,
                        differential_selected_residual=square_diff,
                        stress_factor=stress_factor,
                        pivot=pivot,
                        nfev=square.result.nfev,
                        success=bool(square.result.optimizer_success),
                        cost=float(square.result.cost),
                        optimality=float(square.result.optimality),
                        message=str(square.result.message),
                    )
                )
                print(
                    f"  square {pivot} Rson={rows[-1]['Rson_rg']:.4g} lambda={rows[-1]['lambda0_over_lK_isco']:.4g} "
                    f"selected={rows[-1]['selected_max']:.3e} diff={rows[-1]['differential_selected_max']:.3e} "
                    f"full={rows[-1]['full_overdetermined_max']:.3e}",
                    flush=True,
                )
                if RUN_DIFFERENTIAL_SQUARE:
                    differential_square = solve_square_transonic_polish(
                        audit_params,
                        square.z,
                        pivot=pivot,
                        method="least_squares",
                        max_nfev=SQUARE_MAX_NFEV,
                        residual_tol=RESIDUAL_TOL,
                        use_block_jacobian=USE_BLOCK_JACOBIAN,
                    )
                    diff_selected = square_collocation_residual(differential_square.z, audit_params, pivot=pivot)
                    rows.append(
                        row_for_z(
                            label="square_differential",
                            solve_params=audit_params,
                            audit_params=audit_params,
                            z=differential_square.z,
                            selected_residual=diff_selected,
                            differential_selected_residual=diff_selected,
                            stress_factor=stress_factor,
                            pivot=pivot,
                            nfev=differential_square.result.nfev,
                            success=bool(differential_square.result.optimizer_success),
                            cost=float(differential_square.result.cost),
                            optimality=float(differential_square.result.optimality),
                            message=str(differential_square.result.message),
                        )
                    )
                    print(
                        f"  square differential {pivot} selected={rows[-1]['selected_max']:.3e} "
                        f"full={rows[-1]['full_overdetermined_max']:.3e}",
                        flush=True,
                    )
                write_table(rows)
                write_figure(rows)
            write_table(rows)
            write_figure(rows)
    write_table(rows)
    write_figure(rows)
    print(f"wrote {TABLE_OUTPUT}", flush=True)
    print(f"wrote {FIGURE_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
