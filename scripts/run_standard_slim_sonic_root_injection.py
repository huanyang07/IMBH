"""Inject local sonic roots into the global standard-slim benchmark seed."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np

from imri_qpe.constants import C
from imri_qpe.layer3_minidisk_1d import (
    TransonicSlimParams,
    collocation_residual,
    remap_profile_to_new_sonic_grid,
    residual_audit_from_state_vector,
    solve_square_transonic_polish,
    square_collocation_residual,
    transonic_profile_from_state_vector,
    unpack_state,
)
from imri_qpe.layer3_minidisk_1d.transonic_collocation import pack_state
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
from run_standard_slim_free_rson_square_pivot import solve_free_rson_stage, z_from_free_rson_unknowns
from run_standard_slim_sonic_compatibility_probe import component_vector, solve_local, sonic_components


ROOT = Path(__file__).resolve().parents[1]
TABLE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_SONIC_INJECTION_TABLE",
    "outputs/tables/slim_benchmark_sonic_root_injection.md",
)
JSON_OUTPUT = TABLE_OUTPUT.with_suffix(".json")

R_OUT_RG = float(os.environ.get("IMBH_STANDARD_SLIM_SONIC_INJECTION_ROUT", "300"))
STRESS_FACTOR = float(os.environ.get("IMBH_STANDARD_SLIM_SONIC_INJECTION_STRESS", "1.0"))
FIXED_PROFILE_MAX_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_SONIC_INJECTION_FIXED_PROFILE_NFEV", "100"))
FREE_RSON_MAX_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_SONIC_INJECTION_FREE_RSON_NFEV", "160"))
SEED_SQUARE_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_SONIC_INJECTION_SEED_SQUARE_NFEV", "160"))
POLISH_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_SONIC_INJECTION_POLISH_NFEV", "300"))
RESIDUAL_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_SONIC_INJECTION_RESIDUAL_TOL", "1e-8"))
SOURCE_FILTER = tuple(
    piece.strip()
    for piece in os.environ.get("IMBH_STANDARD_SLIM_SONIC_INJECTION_SOURCE_FILTER", "").replace(":", ",").split(",")
    if piece.strip()
)
ROOT_FILTER = tuple(
    piece.strip()
    for piece in os.environ.get("IMBH_STANDARD_SLIM_SONIC_INJECTION_ROOT_FILTER", "").replace(":", ",").split(",")
    if piece.strip()
)
PIVOT_FILTER = tuple(
    piece.strip()
    for piece in os.environ.get("IMBH_STANDARD_SLIM_SONIC_INJECTION_PIVOT_FILTER", "").replace(":", ",").split(",")
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
        max_nfev=max(SEED_SQUARE_NFEV, POLISH_NFEV),
        residual_tol=RESIDUAL_TOL,
        outer_closure="thin_value",
        interval_residual_form="differential",
        integrated_residual_weighting="none",
    )


def build_source_states(params: TransonicSlimParams) -> dict[str, np.ndarray]:
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
        max_nfev=SEED_SQUARE_NFEV,
        residual_tol=RESIDUAL_TOL,
        use_block_jacobian=False,
    )
    square_c2 = solve_square_transonic_polish(
        params,
        z_free,
        pivot="C2",
        method="least_squares",
        max_nfev=SEED_SQUARE_NFEV,
        residual_tol=RESIDUAL_TOL,
        use_block_jacobian=False,
    )
    return {"free_Rson": z_free, "square_C1": square_c1.z, "square_C2": square_c2.z}


def local_from_z(z: np.ndarray, params: TransonicSlimParams) -> np.ndarray:
    logu, logT, logR_son, lambda0, _logR = unpack_state(z, params)
    return np.array([logu[0], logT[0], logR_son, lambda0], dtype=float)


def injected_state(source_z: np.ndarray, local_root: np.ndarray, params: TransonicSlimParams) -> np.ndarray:
    profile = transonic_profile_from_state_vector(source_z, params)
    profile = replace(profile, sonic_radius=float(np.exp(local_root[2])), lambda0=float(local_root[3]))
    z_remap = remap_profile_to_new_sonic_grid(profile, params)
    logu, logT, _logR_son, _lambda0, _logR = unpack_state(z_remap, params)
    logu = np.array(logu, copy=True)
    logT = np.array(logT, copy=True)
    logu[0] = float(local_root[0])
    logT[0] = float(local_root[1])
    return pack_state(logu, logT, float(local_root[2]), float(local_root[3]))


def dominant(audit) -> str:
    values = {
        "interval_R": abs(audit.interval_radial_max),
        "interval_E": abs(audit.interval_energy_max),
        "outer_omega": abs(audit.outer_omega),
        "outer_energy": abs(audit.outer_energy),
        "D": abs(audit.sonic_D),
        "C1": abs(audit.sonic_C1),
        "C2": abs(audit.sonic_C2),
        "K": abs(audit.sonic_K),
    }
    return max(values, key=values.get)


def row_for_z(
    *,
    label: str,
    source: str,
    root_kind: str,
    pivot: str,
    z: np.ndarray,
    params: TransonicSlimParams,
    nfev: int,
    success: bool,
    message: str,
) -> dict[str, object]:
    audit = residual_audit_from_state_vector(z, params)
    local = local_from_z(z, params)
    sonic = sonic_components(local, params)
    square_c1 = square_collocation_residual(z, params, pivot="C1")
    square_c2 = square_collocation_residual(z, params, pivot="C2")
    return {
        "label": label,
        "source": source,
        "root_kind": root_kind,
        "pivot": pivot,
        "full": float(np.max(np.abs(collocation_residual(z, params)))),
        "square_C1": float(np.max(np.abs(square_c1))),
        "square_C2": float(np.max(np.abs(square_c2))),
        "dominant": dominant(audit),
        "interval_R": float(audit.interval_radial_max),
        "interval_E": float(audit.interval_energy_max),
        "outer_omega": float(audit.outer_omega),
        "outer_energy": float(audit.outer_energy),
        "D": float(audit.sonic_D),
        "C1": float(audit.sonic_C1),
        "C2": float(audit.sonic_C2),
        "K": float(audit.sonic_K),
        "local_all": float(np.max(np.abs(component_vector(local, params, ("D", "C1", "C2", "K"))))),
        "Rson_rg": float(sonic["Rson_rg"]),
        "lambda0_over_lK_isco": float(sonic["lambda0_over_lK_isco"]),
        "M_eff": float(sonic["M_eff"]),
        "H_R": float(sonic["H_R"]),
        "nfev": int(nfev),
        "success": bool(success),
        "message": str(message),
    }


def write_table(rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Standard Slim Sonic Root Injection",
        "",
        "Generated by `scripts/run_standard_slim_sonic_root_injection.py`.",
        "",
        "Rows replace the global seed's inner node with a machine-precision local sonic root, remap the remaining profile to that sonic radius, then optionally repolish the global square system.",
        "",
        "| label | source | root | pivot | full | square C1 | square C2 | dominant | int R | int E | outer omega | outer E | D | C1 | C2 | K | local all | Rson/rg | lambda/lK | M_eff | H/R | nfev | success | message |",
        "|---|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        formatted = {key: fmt(value) if isinstance(value, (float, int, np.floating, np.integer)) else value for key, value in row.items()}
        lines.append(
            "| {label} | {source} | {root_kind} | {pivot} | {full} | {square_C1} | {square_C2} | {dominant} | "
            "{interval_R} | {interval_E} | {outer_omega} | {outer_energy} | {D} | {C1} | {C2} | {K} | "
            "{local_all} | {Rson_rg} | {lambda0_over_lK_isco} | {M_eff} | {H_R} | {nfev} | {success} | {message} |".format(**formatted)
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")
    JSON_OUTPUT.write_text(json.dumps([{key: json_safe(value) for key, value in row.items()} for row in rows], indent=2, sort_keys=True) + "\n")


def main() -> None:
    fiducial = FiducialParams()
    params = make_params(fiducial, eddington_mdot(fiducial.M2_g))
    sources = build_source_states(params)
    fixed_lambda0 = float(params.potential.l_k(params.potential.r_isco) / (params.r_g * C))
    rows: list[dict[str, object]] = []
    for source_label, source_z in sources.items():
        if SOURCE_FILTER and source_label not in SOURCE_FILTER:
            continue
        seed_local = local_from_z(source_z, params)
        root_specs = []
        fixed_result, fixed_root = solve_local(seed_local, params, names=("D", "C1", "C2"), free_lambda=False, fixed_lambda0=fixed_lambda0)
        root_specs.append(("fixed_lambda_D_C1_C2", fixed_result, fixed_root))
        free_result, free_root = solve_local(seed_local, params, names=("D", "C1", "C2", "K"), free_lambda=True, fixed_lambda0=fixed_lambda0)
        root_specs.append(("free_lambda_D_C1_C2_K", free_result, free_root))
        for root_kind, root_result, root in root_specs:
            if ROOT_FILTER and root_kind not in ROOT_FILTER:
                continue
            z_injected = injected_state(source_z, root, params)
            rows.append(
                row_for_z(
                    label="injected_seed",
                    source=source_label,
                    root_kind=root_kind,
                    pivot="-",
                    z=z_injected,
                    params=params,
                    nfev=root_result.nfev,
                    success=bool(root_result.success),
                    message=f"local root: {root_result.message}",
                )
            )
            print(
                f"{source_label} {root_kind} injected full={rows[-1]['full']:.3e} "
                f"dom={rows[-1]['dominant']}",
                flush=True,
            )
            for pivot in ("C1", "C2"):
                if PIVOT_FILTER and pivot not in PIVOT_FILTER:
                    continue
                polished = solve_square_transonic_polish(
                    params,
                    z_injected,
                    pivot=pivot,
                    method="least_squares",
                    max_nfev=POLISH_NFEV,
                    residual_tol=RESIDUAL_TOL,
                    use_block_jacobian=False,
                )
                rows.append(
                    row_for_z(
                        label="polished",
                        source=source_label,
                        root_kind=root_kind,
                        pivot=pivot,
                        z=polished.z,
                        params=params,
                        nfev=polished.result.nfev,
                        success=bool(polished.result.optimizer_success),
                        message=str(polished.result.message),
                    )
                )
                print(
                    f"  pivot={pivot} full={rows[-1]['full']:.3e} square_C1={rows[-1]['square_C1']:.3e} "
                    f"square_C2={rows[-1]['square_C2']:.3e} dom={rows[-1]['dominant']}",
                    flush=True,
                )
            write_table(rows)
    write_table(rows)
    print(f"wrote {TABLE_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
