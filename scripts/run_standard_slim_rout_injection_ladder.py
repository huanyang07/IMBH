"""Gradual R_out continuation using sonic-root injection for the slim benchmark."""

from __future__ import annotations

import json
import os
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
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot
from run_standard_slim_analytic_seed_audit import ALPHA, MDOT_RATIO, fmt, json_safe, n_nodes_for
from run_standard_slim_sonic_compatibility_probe import component_vector, solve_local, sonic_components
from run_standard_slim_sonic_root_injection import (
    build_source_states,
    injected_state,
    local_from_z,
)


ROOT = Path(__file__).resolve().parents[1]
TABLE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_ROUT_INJECTION_TABLE",
    "outputs/tables/slim_benchmark_rout_injection_ladder.md",
)
JSON_OUTPUT = TABLE_OUTPUT.with_suffix(".json")
CHECKPOINT_DIR = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_ROUT_INJECTION_CHECKPOINTS",
    "outputs/checkpoints/slim_benchmark_rout_injection_ladder",
)

R_OUT_LADDER = tuple(
    float(piece)
    for piece in os.environ.get("IMBH_STANDARD_SLIM_ROUT_INJECTION_LADDER", "1000,1500,2000,3000,5000,7000,10000").replace(":", ",").split(",")
    if piece.strip()
)
STRESS_FACTOR = float(os.environ.get("IMBH_STANDARD_SLIM_ROUT_INJECTION_STRESS", "1.0"))
ACCEPTANCE_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_ROUT_INJECTION_ACCEPTANCE_TOL", "1e-5"))
SOLVER_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_ROUT_INJECTION_SOLVER_TOL", "1e-8"))
SOURCE_POLISH_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_ROUT_INJECTION_SOURCE_NFEV", "500"))
POLISH_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_ROUT_INJECTION_POLISH_NFEV", "1200"))
FALLBACK_INDEPENDENT = os.environ.get("IMBH_STANDARD_SLIM_ROUT_INJECTION_FALLBACK_INDEPENDENT", "1") != "0"


def make_params(fiducial: FiducialParams, mdot_edd: float, R_out_rg: float) -> TransonicSlimParams:
    return TransonicSlimParams(
        M2_g=fiducial.M2_g,
        Mdot_g_s=MDOT_RATIO * mdot_edd,
        alpha=ALPHA,
        mu_stress=0.0,
        stress_factor=STRESS_FACTOR,
        R_out_rg=float(R_out_rg),
        n_nodes=n_nodes_for(float(R_out_rg)),
        max_nfev=max(SOURCE_POLISH_NFEV, POLISH_NFEV),
        residual_tol=SOLVER_TOL,
        outer_closure="thin_value",
        interval_residual_form="differential",
        integrated_residual_weighting="none",
    )


def max_residual(z: np.ndarray, params: TransonicSlimParams) -> float:
    return float(np.max(np.abs(collocation_residual(z, params))))


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
    stage: str,
    source: str,
    R_out_rg: float,
    params: TransonicSlimParams,
    z: np.ndarray,
    nfev: int,
    success: bool,
    message: str,
) -> dict[str, object]:
    audit = residual_audit_from_state_vector(z, params)
    local = local_from_z(z, params)
    sonic = sonic_components(local, params)
    square_c1 = square_collocation_residual(z, params, pivot="C1")
    square_c2 = square_collocation_residual(z, params, pivot="C2")
    local_all = float(np.max(np.abs(component_vector(local, params, ("D", "C1", "C2", "K")))))
    full = max_residual(z, params)
    return {
        "stage": stage,
        "source": source,
        "accepted": bool(full <= ACCEPTANCE_TOL),
        "R_out_rg": float(R_out_rg),
        "N": int(params.n_nodes),
        "full": full,
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
        "local_all": local_all,
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
        "# Standard Slim R_out Sonic-Root Injection Ladder",
        "",
        "Generated by `scripts/run_standard_slim_rout_injection_ladder.py`.",
        "",
        "Each radius uses a square-C1 source, a local free-lambda sonic root injection, and a C2 square polish. Accepted means full overdetermined residual is below the configured tolerance.",
        "",
        f"Config: `Mdot/Edd={MDOT_RATIO:g}`, stress `{STRESS_FACTOR:g}`, ladder `{','.join(f'{value:g}' for value in R_OUT_LADDER)}`, acceptance tolerance `{ACCEPTANCE_TOL:g}`, solver tolerance `{SOLVER_TOL:g}`, source nfev `{SOURCE_POLISH_NFEV}`, polish nfev `{POLISH_NFEV}`.",
        "",
        "| stage | source | accepted | R_out/rg | N | full | square C1 | square C2 | dominant | int R | int E | outer omega | outer E | D | C1 | C2 | K | local all | Rson/rg | lambda/lK | M_eff | H/R | nfev | success | message |",
        "|---|---|:---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        formatted = {key: fmt(value) if isinstance(value, (float, int, np.floating, np.integer)) else value for key, value in row.items()}
        lines.append(
            "| {stage} | {source} | {accepted} | {R_out_rg} | {N} | {full} | {square_C1} | {square_C2} | {dominant} | "
            "{interval_R} | {interval_E} | {outer_omega} | {outer_energy} | {D} | {C1} | {C2} | {K} | {local_all} | "
            "{Rson_rg} | {lambda0_over_lK_isco} | {M_eff} | {H_R} | {nfev} | {success} | {message} |".format(**formatted)
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")
    JSON_OUTPUT.write_text(json.dumps([{key: json_safe(value) for key, value in row.items()} for row in rows], indent=2, sort_keys=True) + "\n")


def save_checkpoint(label: str, z: np.ndarray, params: TransonicSlimParams, row: dict[str, object]) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    safe = label.replace(".", "p").replace("-", "m")
    path = CHECKPOINT_DIR / f"{safe}.npz"
    np.savez(
        path,
        z=np.asarray(z, dtype=float),
        R_out_rg=np.array(params.R_out_rg),
        n_nodes=np.array(params.n_nodes),
        ratio=np.array(params.mdot_edd_ratio),
        full=np.array(row["full"]),
        accepted=np.array(row["accepted"]),
    )


def source_from_previous(previous_z: np.ndarray, previous_params: TransonicSlimParams, params: TransonicSlimParams) -> np.ndarray:
    previous_profile = transonic_profile_from_state_vector(previous_z, previous_params)
    z0 = remap_profile_to_new_sonic_grid(previous_profile, params)
    polish = solve_square_transonic_polish(
        params,
        z0,
        pivot="C1",
        method="least_squares",
        max_nfev=SOURCE_POLISH_NFEV,
        residual_tol=SOLVER_TOL,
        use_block_jacobian=False,
    )
    return polish.z


def independent_source(params: TransonicSlimParams) -> np.ndarray:
    return build_source_states(params)["square_C1"]


def injected_polish(source_z: np.ndarray, params: TransonicSlimParams) -> tuple[np.ndarray, object, object]:
    fixed_lambda0 = float(params.potential.l_k(params.potential.r_isco) / (params.r_g * C))
    seed_local = local_from_z(source_z, params)
    local_result, local_root = solve_local(seed_local, params, names=("D", "C1", "C2", "K"), free_lambda=True, fixed_lambda0=fixed_lambda0)
    z_injected = injected_state(source_z, local_root, params)
    polish = solve_square_transonic_polish(
        params,
        z_injected,
        pivot="C2",
        method="least_squares",
        max_nfev=POLISH_NFEV,
        residual_tol=SOLVER_TOL,
        use_block_jacobian=False,
    )
    return z_injected, local_result, polish


def solve_radius(
    params: TransonicSlimParams,
    *,
    previous_z: np.ndarray | None,
    previous_params: TransonicSlimParams | None,
) -> tuple[np.ndarray, str]:
    source_labels: list[tuple[str, np.ndarray]] = []
    if previous_z is not None and previous_params is not None:
        source_labels.append(("remap_previous", source_from_previous(previous_z, previous_params, params)))
    if not source_labels or FALLBACK_INDEPENDENT:
        source_labels.append(("independent", independent_source(params)))

    best_z = None
    best_source = ""
    best_full = np.inf
    for source_label, source_z in source_labels:
        _z_injected, _local_result, polish = injected_polish(source_z, params)
        full = max_residual(polish.z, params)
        if full < best_full:
            best_full = full
            best_z = polish.z
            best_source = source_label
        if full <= ACCEPTANCE_TOL:
            break
    if best_z is None:
        raise RuntimeError("failed to produce any R_out candidate")
    return best_z, best_source


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    rows: list[dict[str, object]] = []
    previous_z: np.ndarray | None = None
    previous_params: TransonicSlimParams | None = None
    for R_out_rg in R_OUT_LADDER:
        params = make_params(fiducial, mdot_edd, R_out_rg)
        print(f"R_out={R_out_rg:g} N={params.n_nodes}", flush=True)
        z, source = solve_radius(params, previous_z=previous_z, previous_params=previous_params)
        row = row_for_z(
            stage="polished",
            source=source,
            R_out_rg=R_out_rg,
            params=params,
            z=z,
            nfev=POLISH_NFEV,
            success=max_residual(z, params) <= ACCEPTANCE_TOL,
            message="C2 square polish after free-lambda local sonic root injection",
        )
        rows.append(row)
        save_checkpoint(f"Rout_{R_out_rg:g}", z, params, row)
        write_table(rows)
        print(
            f"  source={source} full={row['full']:.3e} dom={row['dominant']} "
            f"Rson={row['Rson_rg']:.4g} lambda={row['lambda0_over_lK_isco']:.4g} accepted={row['accepted']}",
            flush=True,
        )
        previous_z = z
        previous_params = params
        if not row["accepted"]:
            print("  stopping ladder at first non-accepted radius", flush=True)
            break
    write_table(rows)
    print(f"wrote {TABLE_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
