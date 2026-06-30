"""Mdot continuation from the accepted standard-slim sonic-root benchmark."""

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
from run_standard_slim_analytic_seed_audit import ALPHA, fmt, json_safe, n_nodes_for
from run_standard_slim_rout_injection_ladder import ACCEPTANCE_TOL as DEFAULT_ACCEPTANCE_TOL
from run_standard_slim_sonic_compatibility_probe import component_vector, solve_local, sonic_components
from run_standard_slim_sonic_root_injection import build_source_states, injected_state, local_from_z


ROOT = Path(__file__).resolve().parents[1]
TABLE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_MDOT_INJECTION_TABLE",
    "outputs/tables/slim_benchmark_mdot_injection_ladder.md",
)
JSON_OUTPUT = TABLE_OUTPUT.with_suffix(".json")
CHECKPOINT_DIR = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_MDOT_INJECTION_CHECKPOINTS",
    "outputs/checkpoints/slim_benchmark_mdot_injection_ladder",
)
ANCHOR_CHECKPOINT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_MDOT_INJECTION_ANCHOR",
    "outputs/checkpoints/slim_benchmark_rout_injection_ladder/Rout_10000.npz",
)

R_OUT_RG = float(os.environ.get("IMBH_STANDARD_SLIM_MDOT_INJECTION_ROUT", "10000"))
ANCHOR_RATIO = float(os.environ.get("IMBH_STANDARD_SLIM_MDOT_INJECTION_ANCHOR_RATIO", "1e-3"))
DOWN_LADDER = tuple(
    float(piece)
    for piece in os.environ.get("IMBH_STANDARD_SLIM_MDOT_INJECTION_DOWN", "7e-4,5e-4,3e-4,1e-4").replace(":", ",").split(",")
    if piece.strip()
)
UP_LADDER = tuple(
    float(piece)
    for piece in os.environ.get("IMBH_STANDARD_SLIM_MDOT_INJECTION_UP", "1.5e-3,2e-3,3e-3,5e-3,7e-3,1e-2").replace(":", ",").split(",")
    if piece.strip()
)
POLISH_PIVOTS = tuple(
    piece.strip()
    for piece in os.environ.get("IMBH_STANDARD_SLIM_MDOT_INJECTION_POLISH_PIVOTS", "C1,C2,K").replace(":", ",").split(",")
    if piece.strip()
)
STRESS_FACTOR = float(os.environ.get("IMBH_STANDARD_SLIM_MDOT_INJECTION_STRESS", "1.0"))
ACCEPTANCE_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_MDOT_INJECTION_ACCEPTANCE_TOL", str(DEFAULT_ACCEPTANCE_TOL)))
SOLVER_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_MDOT_INJECTION_SOLVER_TOL", "1e-8"))
SOURCE_POLISH_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_MDOT_INJECTION_SOURCE_NFEV", "700"))
POLISH_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_MDOT_INJECTION_POLISH_NFEV", "1800"))
FALLBACK_LSQ_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_MDOT_INJECTION_FALLBACK_LSQ_NFEV", "220"))
FALLBACK_INDEPENDENT = os.environ.get("IMBH_STANDARD_SLIM_MDOT_INJECTION_FALLBACK_INDEPENDENT", "1") != "0"


def make_params(fiducial: FiducialParams, mdot_edd: float, ratio: float) -> TransonicSlimParams:
    return TransonicSlimParams(
        M2_g=fiducial.M2_g,
        Mdot_g_s=float(ratio) * mdot_edd,
        alpha=ALPHA,
        mu_stress=0.0,
        stress_factor=STRESS_FACTOR,
        R_out_rg=R_OUT_RG,
        n_nodes=n_nodes_for(R_OUT_RG),
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


def local_root_result(source_z: np.ndarray, params: TransonicSlimParams):
    fixed_lambda0 = float(params.potential.l_k(params.potential.r_isco) / (params.r_g * C))
    seed_local = local_from_z(source_z, params)
    return solve_local(seed_local, params, names=("D", "C1", "C2", "K"), free_lambda=True, fixed_lambda0=fixed_lambda0)


def square_polish(z0: np.ndarray, params: TransonicSlimParams, *, pivot: str, max_nfev: int, fallback_lsq: bool):
    newton = solve_square_transonic_polish(
        params,
        z0,
        pivot=pivot,
        method="newton",
        max_nfev=max_nfev,
        residual_tol=SOLVER_TOL,
        use_block_jacobian=True,
        linear_solver="direct",
        max_step_norm=0.5,
    )
    if not fallback_lsq or max_residual(newton.z, params) <= ACCEPTANCE_TOL:
        return newton
    return solve_square_transonic_polish(
        params,
        newton.z,
        pivot=pivot,
        method="least_squares",
        max_nfev=FALLBACK_LSQ_NFEV,
        residual_tol=SOLVER_TOL,
        use_block_jacobian=True,
    )


def polish_from_source(source_z: np.ndarray, params: TransonicSlimParams) -> tuple[np.ndarray, object, object, float]:
    local_result, local_root = local_root_result(source_z, params)
    z_injected = injected_state(source_z, local_root, params)
    injected_full = max_residual(z_injected, params)
    best_polish = None
    best_full = np.inf
    for pivot in POLISH_PIVOTS:
        polish = square_polish(z_injected, params, pivot=pivot, max_nfev=POLISH_NFEV, fallback_lsq=True)
        full = max_residual(polish.z, params)
        if full < best_full:
            best_full = full
            best_polish = polish
        if full <= ACCEPTANCE_TOL:
            break
    if best_polish is None:
        raise RuntimeError("POLISH_PIVOTS did not contain any usable pivot")
    return z_injected, local_result, best_polish, injected_full


def remap_previous_source(previous_z: np.ndarray, previous_params: TransonicSlimParams, params: TransonicSlimParams) -> tuple[np.ndarray, float]:
    previous_profile = transonic_profile_from_state_vector(previous_z, previous_params)
    z0 = remap_profile_to_new_sonic_grid(previous_profile, params)
    remap_full = max_residual(z0, params)
    polish = square_polish(z0, params, pivot="C1", max_nfev=SOURCE_POLISH_NFEV, fallback_lsq=False)
    return polish.z, remap_full


def independent_source(params: TransonicSlimParams) -> tuple[np.ndarray, float]:
    source_z = build_source_states(params)["square_C1"]
    return source_z, max_residual(source_z, params)


def row_for_z(
    *,
    branch: str,
    stage: str,
    source: str,
    ratio: float,
    params: TransonicSlimParams,
    z: np.ndarray,
    remap_full: float,
    source_full: float,
    injected_full: float,
    polish_pivot: str,
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
        "branch": branch,
        "stage": stage,
        "source": source,
        "polish_pivot": str(polish_pivot),
        "accepted": bool(full <= ACCEPTANCE_TOL),
        "ratio": float(ratio),
        "R_out_rg": float(params.R_out_rg),
        "N": int(params.n_nodes),
        "remap_full": float(remap_full),
        "source_full": float(source_full),
        "injected_full": float(injected_full),
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
        "# Standard Slim Mdot Sonic-Root Injection Ladder",
        "",
        "Generated by `scripts/run_standard_slim_mdot_injection_ladder.py`.",
        "",
        "Each target ratio remaps the previous accepted profile, polishes a square-C1 source, reinjects a local free-lambda sonic root, and then C2-polishes the square global system. Accepted means the full overdetermined residual is below the configured tolerance.",
        "",
        f"Config: `R_out={R_OUT_RG:g} rg`, anchor `{ANCHOR_RATIO:g}`, down ladder `{','.join(f'{value:g}' for value in DOWN_LADDER)}`, up ladder `{','.join(f'{value:g}' for value in UP_LADDER)}`, acceptance tolerance `{ACCEPTANCE_TOL:g}`, solver tolerance `{SOLVER_TOL:g}`, source nfev `{SOURCE_POLISH_NFEV}`, polish nfev `{POLISH_NFEV}`.",
        "",
        "| branch | stage | source | pivot | accepted | Mdot/Edd | R_out/rg | N | remap full | source full | injected full | full | square C1 | square C2 | dominant | int R | int E | outer omega | outer E | D | C1 | C2 | K | local all | Rson/rg | lambda/lK | M_eff | H/R | nfev | success | message |",
        "|---|---|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        formatted = {key: fmt(value) if isinstance(value, (float, int, np.floating, np.integer)) else value for key, value in row.items()}
        lines.append(
            "| {branch} | {stage} | {source} | {polish_pivot} | {accepted} | {ratio} | {R_out_rg} | {N} | {remap_full} | {source_full} | {injected_full} | {full} | "
            "{square_C1} | {square_C2} | {dominant} | {interval_R} | {interval_E} | {outer_omega} | {outer_energy} | {D} | {C1} | {C2} | {K} | "
            "{local_all} | {Rson_rg} | {lambda0_over_lK_isco} | {M_eff} | {H_R} | {nfev} | {success} | {message} |".format(**formatted)
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
        branch=np.array(row["branch"]),
    )


def anchor_solution(fiducial: FiducialParams, mdot_edd: float) -> tuple[np.ndarray, TransonicSlimParams]:
    params = make_params(fiducial, mdot_edd, ANCHOR_RATIO)
    if ANCHOR_CHECKPOINT.exists():
        data = np.load(ANCHOR_CHECKPOINT, allow_pickle=True)
        z = np.asarray(data["z"], dtype=float)
        if z.shape == (2 * params.n_nodes + 2,):
            return z, params
    source_z = build_source_states(params)["square_C1"]
    _z_injected, _local_result, polish, _injected_full = polish_from_source(source_z, params)
    return polish.z, params


def solve_ratio(
    ratio: float,
    params: TransonicSlimParams,
    previous_z: np.ndarray,
    previous_params: TransonicSlimParams,
) -> tuple[np.ndarray, str, float, float, float, object]:
    candidates: list[tuple[str, np.ndarray, float]] = []
    remap_source, remap_full = remap_previous_source(previous_z, previous_params, params)
    candidates.append(("remap_previous", remap_source, remap_full))
    if FALLBACK_INDEPENDENT:
        source_z, independent_full = independent_source(params)
        candidates.append(("independent", source_z, independent_full))

    best_z: np.ndarray | None = None
    best_source = ""
    best_remap_full = np.nan
    best_source_full = np.nan
    best_injected_full = np.nan
    best_polish = None
    best_full = np.inf
    for source_label, source_z, candidate_remap_full in candidates:
        source_full = max_residual(source_z, params)
        _z_injected, _local_result, polish, injected_full = polish_from_source(source_z, params)
        full = max_residual(polish.z, params)
        if full < best_full:
            best_full = full
            best_z = polish.z
            best_source = source_label
            best_remap_full = candidate_remap_full
            best_source_full = source_full
            best_injected_full = injected_full
            best_polish = polish
        if full <= ACCEPTANCE_TOL:
            break
    if best_z is None or best_polish is None:
        raise RuntimeError(f"failed to produce any Mdot candidate for ratio={ratio:g}")
    return best_z, best_source, best_remap_full, best_source_full, best_injected_full, best_polish


def run_branch(
    *,
    branch: str,
    ratios: tuple[float, ...],
    anchor_z: np.ndarray,
    anchor_params: TransonicSlimParams,
    rows: list[dict[str, object]],
    fiducial: FiducialParams,
    mdot_edd: float,
) -> None:
    previous_z = anchor_z
    previous_params = anchor_params
    for ratio in ratios:
        params = make_params(fiducial, mdot_edd, ratio)
        print(f"{branch} ratio={ratio:g} N={params.n_nodes}", flush=True)
        z, source, remap_full, source_full, injected_full, polish = solve_ratio(ratio, params, previous_z, previous_params)
        row = row_for_z(
            branch=branch,
            stage="polished",
            source=source,
            ratio=ratio,
            params=params,
            z=z,
            remap_full=remap_full,
            source_full=source_full,
            injected_full=injected_full,
            polish_pivot=polish.pivot,
            nfev=polish.result.nfev,
            success=polish.result.optimizer_success,
            message=polish.result.message,
        )
        rows.append(row)
        save_checkpoint(f"{branch}_mdot_{ratio:g}", z, params, row)
        write_table(rows)
        print(
            f"  source={source} full={row['full']:.3e} dom={row['dominant']} "
            f"Rson={row['Rson_rg']:.4g} M_eff={row['M_eff']:.4g} accepted={row['accepted']}",
            flush=True,
        )
        if not row["accepted"]:
            print(f"  stopping {branch} ladder at first non-accepted ratio", flush=True)
            break
        previous_z = z
        previous_params = params


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    anchor_z, anchor_params = anchor_solution(fiducial, mdot_edd)
    rows: list[dict[str, object]] = [
        row_for_z(
            branch="anchor",
            stage="checkpoint",
            source=str(ANCHOR_CHECKPOINT.relative_to(ROOT)) if ANCHOR_CHECKPOINT.exists() else "rebuilt",
            ratio=ANCHOR_RATIO,
            params=anchor_params,
            z=anchor_z,
            remap_full=np.nan,
            source_full=np.nan,
            injected_full=np.nan,
            polish_pivot="-",
            nfev=0,
            success=True,
            message="accepted R_out ladder anchor",
        )
    ]
    save_checkpoint(f"anchor_mdot_{ANCHOR_RATIO:g}", anchor_z, anchor_params, rows[0])
    write_table(rows)
    print(f"anchor ratio={ANCHOR_RATIO:g} full={rows[0]['full']:.3e} accepted={rows[0]['accepted']}", flush=True)
    run_branch(
        branch="down",
        ratios=DOWN_LADDER,
        anchor_z=anchor_z,
        anchor_params=anchor_params,
        rows=rows,
        fiducial=fiducial,
        mdot_edd=mdot_edd,
    )
    run_branch(
        branch="up",
        ratios=UP_LADDER,
        anchor_z=anchor_z,
        anchor_params=anchor_params,
        rows=rows,
        fiducial=fiducial,
        mdot_edd=mdot_edd,
    )
    write_table(rows)
    print(f"wrote {TABLE_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
