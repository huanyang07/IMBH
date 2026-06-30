"""Local sonic-compatibility probe for the standard low-Mdot benchmark."""

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
    solve_square_transonic_polish,
    state_bounds,
    unpack_state,
)
from imri_qpe.layer3_minidisk_1d.transonic_local import algebraic_state, sonic_derivative_branches, sonic_diagnostics
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
)
from run_standard_slim_fixed_eigen_profile import profile_unknowns_from_state, solve_profile_stage
from run_standard_slim_free_rson_square_pivot import (
    solve_free_rson_stage,
    z_from_free_rson_unknowns,
)


ROOT = Path(__file__).resolve().parents[1]
TABLE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_SONIC_PROBE_TABLE",
    "outputs/tables/slim_benchmark_sonic_compatibility_probe.md",
)
JSON_OUTPUT = TABLE_OUTPUT.with_suffix(".json")

R_OUT_RG = float(os.environ.get("IMBH_STANDARD_SLIM_SONIC_PROBE_ROUT", "300"))
STRESS_FACTOR = float(os.environ.get("IMBH_STANDARD_SLIM_SONIC_PROBE_STRESS", "1.0"))
FIXED_PROFILE_MAX_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_SONIC_PROBE_FIXED_PROFILE_NFEV", "100"))
FREE_RSON_MAX_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_SONIC_PROBE_FREE_RSON_NFEV", "160"))
SQUARE_MAX_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_SONIC_PROBE_SQUARE_NFEV", "160"))
LOCAL_MAX_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_SONIC_PROBE_LOCAL_NFEV", "800"))
RESIDUAL_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_SONIC_PROBE_RESIDUAL_TOL", "1e-10"))
MULTISTART_OFFSETS = tuple(
    float(piece)
    for piece in os.environ.get("IMBH_STANDARD_SLIM_SONIC_PROBE_OFFSETS", "-0.02,0,0.02").replace(":", ",").split(",")
    if piece.strip()
)


def make_params(fiducial: FiducialParams, mdot_edd: float) -> TransonicSlimParams:
    return TransonicSlimParams(
        M2_g=fiducial.M2_g,
        Mdot_g_s=MDOT_RATIO * mdot_edd,
        alpha=ALPHA,
        mu_stress=0.0,
        stress_factor=STRESS_FACTOR,
        R_out_rg=R_OUT_RG,
        n_nodes=n_nodes_for(R_OUT_RG),
        max_nfev=max(FREE_RSON_MAX_NFEV, SQUARE_MAX_NFEV),
        residual_tol=RESIDUAL_TOL,
        outer_closure="thin_value",
        interval_residual_form="differential",
        integrated_residual_weighting="none",
    )


def sonic_components(local: np.ndarray, params: TransonicSlimParams) -> dict[str, float]:
    local = np.asarray(local, dtype=float)
    sonic = sonic_diagnostics(float(local[2]), np.array([local[0], local[1]], dtype=float), float(local[3]), params)
    state = algebraic_state(float(local[2]), float(local[0]), float(local[1]), float(local[3]), params)
    return {
        "D": float(sonic.D),
        "C1": float(sonic.C1),
        "C2": float(sonic.C2),
        "K": float(sonic.compatibility),
        "N": float(sonic.N),
        "smin": float(sonic.smin_over_smax),
        "M_eff": float(sonic.M_eff),
        "Rson_rg": float(np.exp(local[2]) / params.r_g),
        "lambda0_over_lK_isco": float(local[3] * params.r_g * C / params.potential.l_k(params.potential.r_isco)),
        "logu": float(local[0]),
        "logT": float(local[1]),
        "H_R": float(state.H_over_R),
        "Omega_frac": float(state.Omega / state.Omega_K - 1.0),
    }


def component_vector(local: np.ndarray, params: TransonicSlimParams, names: tuple[str, ...]) -> np.ndarray:
    values = sonic_components(local, params)
    return np.asarray([values[name] for name in names], dtype=float)


def local_bounds(params: TransonicSlimParams, *, free_lambda: bool) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = state_bounds(params)
    lo = np.array([lower[0], lower[params.n_nodes], lower[-2], lower[-1]], dtype=float)
    hi = np.array([upper[0], upper[params.n_nodes], upper[-2], upper[-1]], dtype=float)
    if not free_lambda:
        lo = lo[:3]
        hi = hi[:3]
    return lo, hi


def solve_local(
    seed: np.ndarray,
    params: TransonicSlimParams,
    *,
    names: tuple[str, ...],
    free_lambda: bool,
    fixed_lambda0: float,
):
    if free_lambda:
        x0 = np.asarray(seed, dtype=float)

        def residual(trial: np.ndarray) -> np.ndarray:
            return component_vector(trial, params, names)

    else:
        x0 = np.asarray(seed[:3], dtype=float)

        def residual(trial: np.ndarray) -> np.ndarray:
            local = np.array([trial[0], trial[1], trial[2], fixed_lambda0], dtype=float)
            return component_vector(local, params, names)

    lower, upper = local_bounds(params, free_lambda=free_lambda)
    result = least_squares(
        residual,
        np.clip(x0, lower + 1.0e-12, upper - 1.0e-12),
        bounds=(lower, upper),
        x_scale="jac",
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-12,
        max_nfev=LOCAL_MAX_NFEV,
    )
    if free_lambda:
        local = np.asarray(result.x, dtype=float)
    else:
        local = np.array([result.x[0], result.x[1], result.x[2], fixed_lambda0], dtype=float)
    return result, local


def row_from_local(
    *,
    stage: str,
    seed_label: str,
    solve_names: tuple[str, ...],
    free_lambda: bool,
    result,
    local: np.ndarray,
    params: TransonicSlimParams,
):
    values = sonic_components(local, params)
    residual = component_vector(local, params, solve_names)
    all_residual = component_vector(local, params, ("D", "C1", "C2", "K"))
    try:
        branches = sonic_derivative_branches(
            float(local[2]),
            np.array([local[0], local[1]], dtype=float),
            float(local[3]),
            params,
            half_width=300.0,
            scan_points=1201,
        )
    except Exception:
        branches = ()
    row = {
        "stage": stage,
        "seed": seed_label,
        "solve_components": ",".join(solve_names),
        "free_lambda": bool(free_lambda),
        "selected_max": float(np.max(np.abs(residual))),
        "all_D_C1_C2_K_max": float(np.max(np.abs(all_residual))),
        "D": values["D"],
        "C1": values["C1"],
        "C2": values["C2"],
        "K": values["K"],
        "N": values["N"],
        "smin": values["smin"],
        "M_eff": values["M_eff"],
        "Rson_rg": values["Rson_rg"],
        "lambda0_over_lK_isco": values["lambda0_over_lK_isco"],
        "logu": values["logu"],
        "logT": values["logT"],
        "H_R": values["H_R"],
        "Omega_frac": values["Omega_frac"],
        "branches": int(len(branches)),
        "branch_a_min": float(min((branch.a for branch in branches), default=np.nan)),
        "branch_a_max": float(max((branch.a for branch in branches), default=np.nan)),
        "nfev": int(result.nfev),
        "success": bool(result.success),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "message": str(result.message),
    }
    return row


def global_seed_rows_and_locals(params: TransonicSlimParams) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    lambda0 = float(params.potential.l_k(params.potential.r_isco) / (params.r_g * C))
    fixed_logR_son = float(np.log(R_SON_RG * params.r_g))
    z_seed, _arrays = analytic_seed_state(params, fixed_logR_son, lambda0)
    fixed_profile = solve_profile_stage(
        profile_unknowns_from_state(z_seed, params),
        params,
        fixed_logR_son,
        lambda0,
        max_nfev=FIXED_PROFILE_MAX_NFEV,
    )
    free_rson = solve_free_rson_stage(fixed_profile.x, fixed_logR_son, params, lambda0)
    z_free = z_from_free_rson_unknowns(free_rson.x, params, lambda0)
    square_c1 = solve_square_transonic_polish(
        params,
        z_free,
        pivot="C1",
        method="least_squares",
        max_nfev=SQUARE_MAX_NFEV,
        residual_tol=RESIDUAL_TOL,
        use_block_jacobian=False,
    )
    square_c2 = solve_square_transonic_polish(
        params,
        z_free,
        pivot="C2",
        method="least_squares",
        max_nfev=SQUARE_MAX_NFEV,
        residual_tol=RESIDUAL_TOL,
        use_block_jacobian=False,
    )
    locals_by_seed = {}
    stage_rows = []
    for label, z in (
        ("free_Rson", z_free),
        ("square_C1", square_c1.z),
        ("square_C2", square_c2.z),
    ):
        logu, logT, logR_son, lam, _logR = unpack_state(z, params)
        local = np.array([logu[0], logT[0], logR_son, lam], dtype=float)
        locals_by_seed[label] = local
        fake = type("FakeResult", (), {"nfev": 0, "success": True, "cost": 0.0, "optimality": 0.0, "message": "global seed evaluation"})()
        stage_rows.append(
            row_from_local(
                stage="global_seed",
                seed_label=label,
                solve_names=("D", "C1", "C2", "K"),
                free_lambda=True,
                result=fake,
                local=local,
                params=params,
            )
        )
    return stage_rows, locals_by_seed


def local_multistart_seeds(locals_by_seed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    seeds: dict[str, np.ndarray] = {}
    for label, local in locals_by_seed.items():
        seeds[label] = np.asarray(local, dtype=float)
        for offset in MULTISTART_OFFSETS:
            if offset == 0.0:
                continue
            perturbed = np.asarray(local, dtype=float).copy()
            perturbed[2] += offset
            seeds[f"{label}_dlogR_{offset:g}"] = perturbed
    return seeds


def write_table(rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Standard Slim Local Sonic Compatibility Probe",
        "",
        "Generated by `scripts/run_standard_slim_sonic_compatibility_probe.py`.",
        "",
        "This probe asks whether the sonic equations can be made mutually small near the current low-Mdot benchmark branch, independent of the global profile residual.",
        "",
        f"Config: `Mdot/Edd={MDOT_RATIO:g}`, `R_out={R_OUT_RG:g} rg`, `stress_factor={STRESS_FACTOR:g}`, local max_nfev `{LOCAL_MAX_NFEV}`.",
        "",
        "| stage | seed | components | free lambda | selected | all max | D | C1 | C2 | K | N | smin | M_eff | Rson/rg | lambda/lK | H/R | Omega err | branches | branch a min | branch a max | nfev | success | optimality | message |",
        "|---|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---|",
    ]
    for row in rows:
        formatted = {key: fmt(value) if isinstance(value, (float, int, np.floating, np.integer)) else value for key, value in row.items()}
        lines.append(
            "| {stage} | {seed} | {solve_components} | {free_lambda} | {selected_max} | {all_D_C1_C2_K_max} | "
            "{D} | {C1} | {C2} | {K} | {N} | {smin} | {M_eff} | {Rson_rg} | {lambda0_over_lK_isco} | "
            "{H_R} | {Omega_frac} | {branches} | {branch_a_min} | {branch_a_max} | {nfev} | {success} | "
            "{optimality} | {message} |".format(**formatted)
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")
    JSON_OUTPUT.write_text(json.dumps([{key: json_safe(value) for key, value in row.items()} for row in rows], indent=2, sort_keys=True) + "\n")


def main() -> None:
    fiducial = FiducialParams()
    params = make_params(fiducial, eddington_mdot(fiducial.M2_g))
    params = replace(params, interval_residual_form="differential", integrated_residual_weighting="none")
    rows, locals_by_seed = global_seed_rows_and_locals(params)
    write_table(rows)

    fixed_lambda0 = float(params.potential.l_k(params.potential.r_isco) / (params.r_g * C))
    for seed_label, seed in local_multistart_seeds(locals_by_seed).items():
        for names, free_lambda in (
            (("D", "C1", "C2"), False),
            (("D", "C1", "C2", "K"), True),
            (("D", "C1", "C2"), True),
        ):
            result, local = solve_local(seed, params, names=names, free_lambda=free_lambda, fixed_lambda0=fixed_lambda0)
            rows.append(
                row_from_local(
                    stage="local_solve",
                    seed_label=seed_label,
                    solve_names=names,
                    free_lambda=free_lambda,
                    result=result,
                    local=local,
                    params=params,
                )
            )
            print(
                f"{seed_label} names={','.join(names)} free_lambda={free_lambda} "
                f"selected={rows[-1]['selected_max']:.3e} all={rows[-1]['all_D_C1_C2_K_max']:.3e} "
                f"R={rows[-1]['Rson_rg']:.4g} lambda={rows[-1]['lambda0_over_lK_isco']:.4g}",
                flush=True,
            )
            write_table(rows)
    write_table(rows)
    print(f"wrote {TABLE_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
