"""Block-isolation audit for the fixed-Mdot transonic residual floor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from imri_qpe.layer3_minidisk_1d import TransonicSlimParams, remap_profile_to_new_sonic_grid
from imri_qpe.layer3_minidisk_1d.transonic_collocation import (
    profile_from_state_vector,
    residual_audit_from_state_vector,
    square_collocation_jacobian,
    square_collocation_residual,
    state_bounds,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CHECKPOINT = ROOT / "outputs" / "checkpoints" / "transonic_n64_arclength_adaptive" / "030_arc_x1_0p90277664.npz"
TABLE_OUTPUT = ROOT / "outputs" / "tables" / "transonic_fixed_mdot_block_audit.md"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "transonic_fixed_mdot_block_audit"

RESIDUAL_TOL = 1.0e-6
N_NODES = 64
BASE_R_OUT_RG = 3000.0
PIVOTS = ("C1", "C2")
R_OUT_SWEEP = (300.0, 1000.0, 3000.0, 10000.0)
MAX_NFEV = 80


def fmt(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.4g}"


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    return value


def row_json(row: dict[str, object]) -> str:
    return json.dumps({key: json_safe(value) for key, value in row.items()}, sort_keys=True)


def params_for(fiducial: FiducialParams, mdot_edd: float, ratio: float, R_out_rg: float) -> TransonicSlimParams:
    return TransonicSlimParams(
        M2_g=fiducial.M2_g,
        Mdot_g_s=ratio * mdot_edd,
        alpha=fiducial.alpha_cool,
        n_nodes=N_NODES,
        R_out_rg=R_out_rg,
        residual_tol=RESIDUAL_TOL,
        max_nfev=MAX_NFEV,
    )


def load_source(fiducial: FiducialParams, mdot_edd: float) -> dict[str, object]:
    with np.load(SOURCE_CHECKPOINT, allow_pickle=False) as data:
        z = np.asarray(data["z"], dtype=float)
        ratio = float(data["ratio"])
    params = params_for(fiducial, mdot_edd, ratio, BASE_R_OUT_RG)
    profile = profile_from_state_vector(z, params)
    return {"ratio": ratio, "z": z, "params": params, "profile": profile}


def row_slices(params: TransonicSlimParams) -> dict[str, np.ndarray]:
    interval = np.arange(0, 2 * (params.n_nodes - 1), dtype=int)
    outer_omega = np.array([2 * (params.n_nodes - 1)], dtype=int)
    outer_energy = np.array([2 * (params.n_nodes - 1) + 1], dtype=int)
    sonic = np.array([2 * (params.n_nodes - 1) + 2, 2 * (params.n_nodes - 1) + 3], dtype=int)
    all_rows = np.concatenate([interval, outer_omega, outer_energy, sonic])
    return {
        "all": all_rows,
        "interval": interval,
        "outer_omega": outer_omega,
        "outer_energy": outer_energy,
        "outer": np.concatenate([outer_omega, outer_energy]),
        "sonic": sonic,
    }


def block_variants(params: TransonicSlimParams) -> list[dict[str, object]]:
    rows = row_slices(params)
    return [
        {"label": "full", "rows": rows["all"], "outer_weight": 1.0, "description": "interval + both outer rows + sonic pair"},
        {"label": "no_outer", "rows": np.concatenate([rows["interval"], rows["sonic"]]), "outer_weight": 1.0, "description": "interval + sonic pair"},
        {"label": "no_sonic", "rows": np.concatenate([rows["interval"], rows["outer"]]), "outer_weight": 1.0, "description": "interval + both outer rows"},
        {"label": "interval_only", "rows": rows["interval"], "outer_weight": 1.0, "description": "interval equations only"},
        {
            "label": "drop_outer_omega",
            "rows": np.concatenate([rows["interval"], rows["outer_energy"], rows["sonic"]]),
            "outer_weight": 1.0,
            "description": "interval + outer energy + sonic pair",
        },
        {
            "label": "drop_outer_energy",
            "rows": np.concatenate([rows["interval"], rows["outer_omega"], rows["sonic"]]),
            "outer_weight": 1.0,
            "description": "interval + outer omega + sonic pair",
        },
        {"label": "outer_weight_1e-1", "rows": rows["all"], "outer_weight": 1.0e-1, "description": "full system with outer rows weighted by 0.1"},
        {"label": "outer_weight_1e-2", "rows": rows["all"], "outer_weight": 1.0e-2, "description": "full system with outer rows weighted by 0.01"},
        {"label": "outer_weight_1e-3", "rows": rows["all"], "outer_weight": 1.0e-3, "description": "full system with outer rows weighted by 0.001"},
    ]


def weights_for(params: TransonicSlimParams, outer_weight: float) -> np.ndarray:
    weights = np.ones(2 * params.n_nodes + 2, dtype=float)
    rows = row_slices(params)
    weights[rows["outer"]] = outer_weight
    return weights


def selected_residual(z, params: TransonicSlimParams, pivot: str, rows: np.ndarray, outer_weight: float) -> np.ndarray:
    residual = square_collocation_residual(z, params, pivot=pivot)
    return (weights_for(params, outer_weight) * residual)[rows]


def selected_jacobian(z, params: TransonicSlimParams, pivot: str, rows: np.ndarray, outer_weight: float):
    jac = square_collocation_jacobian(z, params, pivot=pivot, rel_step=3.0e-5)
    weights = weights_for(params, outer_weight)
    return jac[rows, :].multiply(weights[rows, None])


def active_max(audit) -> float:
    return max(
        abs(audit.interval_radial_max),
        abs(audit.interval_energy_max),
        abs(audit.outer_omega),
        abs(audit.outer_energy),
        abs(audit.sonic_D),
        abs(audit.sonic_K),
    )


def dominant_block(audit) -> str:
    values = {
        "interval_R": abs(audit.interval_radial_max),
        "interval_E": abs(audit.interval_energy_max),
        "outer_Omega": abs(audit.outer_omega),
        "outer_E": abs(audit.outer_energy),
        "D": abs(audit.sonic_D),
        "K": abs(audit.sonic_K),
        "C1": abs(audit.sonic_C1),
        "C2": abs(audit.sonic_C2),
    }
    return max(values, key=values.get)


def profile_changes(before, after) -> tuple[float, float, float, float]:
    return (
        float(np.max(np.abs(np.log(after.u) - np.log(before.u)))),
        float(np.max(np.abs(np.log(after.T) - np.log(before.T)))),
        float(abs(np.log(after.sonic_radius / before.sonic_radius))),
        float(abs(after.lambda0 - before.lambda0)),
    )


def solve_variant(
    z0: np.ndarray,
    params: TransonicSlimParams,
    pivot: str,
    rows: np.ndarray,
    outer_weight: float,
) -> tuple[np.ndarray, object, float]:
    lower, upper = state_bounds(params)
    z_start = np.clip(np.asarray(z0, dtype=float), lower + 1.0e-12, upper - 1.0e-12)
    initial_selected = float(np.max(np.abs(selected_residual(z_start, params, pivot, rows, outer_weight))))
    result = least_squares(
        lambda z: selected_residual(z, params, pivot, rows, outer_weight),
        z_start,
        jac=lambda z: selected_jacobian(z, params, pivot, rows, outer_weight),
        bounds=(lower, upper),
        x_scale="jac",
        ftol=1.0e-11,
        xtol=1.0e-11,
        gtol=1.0e-10,
        max_nfev=MAX_NFEV,
    )
    return np.asarray(result.x, dtype=float), result, initial_selected


def row_from_solution(
    *,
    family: str,
    label: str,
    description: str,
    ratio: float,
    params: TransonicSlimParams,
    pivot: str,
    z0: np.ndarray,
    z: np.ndarray,
    result,
    initial_selected: float,
    rows: np.ndarray,
    outer_weight: float,
) -> dict[str, object]:
    before = profile_from_state_vector(z0, params)
    profile = profile_from_state_vector(z, params)
    audit = residual_audit_from_state_vector(z, params)
    selected_final = float(np.max(np.abs(selected_residual(z, params, pivot, rows, outer_weight))))
    full = square_collocation_residual(z, params, pivot=pivot)
    dlogu, dlogT, dlogRson, dlambda0 = profile_changes(before, profile)
    return {
        "family": family,
        "label": label,
        "description": description,
        "ratio": ratio,
        "R_out_rg": params.R_out_rg,
        "pivot": pivot,
        "n_rows": int(len(rows)),
        "selected_initial": initial_selected,
        "selected_final": selected_final,
        "full_square": float(np.max(np.abs(full))),
        "active_max": active_max(audit),
        "dominant": dominant_block(audit),
        "interval_R": audit.interval_radial_max,
        "interval_E": audit.interval_energy_max,
        "outer_Omega": audit.outer_omega,
        "outer_E": audit.outer_energy,
        "D": audit.sonic_D,
        "C1": audit.sonic_C1,
        "C2": audit.sonic_C2,
        "K": audit.sonic_K,
        "max_HR": float(np.max(profile.H_over_R)),
        "int_adv": profile.integrated_advective_fraction,
        "dlogu": dlogu,
        "dlogT": dlogT,
        "dlogRson": dlogRson,
        "dlambda0": dlambda0,
        "nfev": int(result.nfev),
        "njev": -1 if result.njev is None else int(result.njev),
        "success": bool(result.success),
        "message": str(result.message),
        "z": z,
        "params": params,
    }


def save_checkpoint(row: dict[str, object]) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{row['family']}_{row['label']}_R{float(row['R_out_rg']):.0f}_"
        f"{row['pivot']}_{float(row['ratio']):.8f}"
    ).replace(".", "p")
    filename = f"{stem}.npz"
    payload = {key: value for key, value in row.items() if key not in {"z", "params"}}
    np.savez_compressed(
        CHECKPOINT_DIR / filename,
        z=np.asarray(row["z"], dtype=float),
        ratio=np.array(row["ratio"]),
        R_out_rg=np.array(row["R_out_rg"]),
        pivot=np.array(row["pivot"]),
        row_json=np.array(row_json(payload)),
    )


def write_table(rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Fixed-Mdot Residual Block Audit",
        "",
        "Generated by `scripts/run_transonic_fixed_mdot_block_audit.py`.",
        "",
        "Rows solve selected residual subsets at the native `Mdot/Edd ~= 0.9028` checkpoint and report the unweighted full block audit afterward.",
        "",
        "| family | label | R_out/rg | pivot | rows | selected initial | selected final | full square | active max | dominant | interval R | interval E | outer Omega | outer E | D | C1 | C2 | K | max H/R | int adv | dlogu | dlogT | dlog Rson | d lambda0 | nfev | success | message |",
        "|---|---|---:|:---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {family} | {label} | {R_out_rg} | {pivot} | {n_rows} | {selected_initial} | {selected_final} | "
            "{full_square} | {active_max} | {dominant} | {interval_R} | {interval_E} | {outer_Omega} | "
            "{outer_E} | {D} | {C1} | {C2} | {K} | {max_HR} | {int_adv} | {dlogu} | {dlogT} | "
            "{dlogRson} | {dlambda0} | {nfev} | {success} | {message} |".format(
                family=row["family"],
                label=row["label"],
                R_out_rg=fmt(float(row["R_out_rg"])),
                pivot=row["pivot"],
                n_rows=row["n_rows"],
                selected_initial=fmt(float(row["selected_initial"])),
                selected_final=fmt(float(row["selected_final"])),
                full_square=fmt(float(row["full_square"])),
                active_max=fmt(float(row["active_max"])),
                dominant=row["dominant"],
                interval_R=fmt(float(row["interval_R"])),
                interval_E=fmt(float(row["interval_E"])),
                outer_Omega=fmt(float(row["outer_Omega"])),
                outer_E=fmt(float(row["outer_E"])),
                D=fmt(float(row["D"])),
                C1=fmt(float(row["C1"])),
                C2=fmt(float(row["C2"])),
                K=fmt(float(row["K"])),
                max_HR=fmt(float(row["max_HR"])),
                int_adv=fmt(float(row["int_adv"])),
                dlogu=fmt(float(row["dlogu"])),
                dlogT=fmt(float(row["dlogT"])),
                dlogRson=fmt(float(row["dlogRson"])),
                dlambda0=fmt(float(row["dlambda0"])),
                nfev=row["nfev"],
                success="yes" if row["success"] else "no",
                message=str(row["message"]).replace("|", "/"),
            )
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    source = load_source(fiducial, mdot_edd)
    rows: list[dict[str, object]] = []
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for old_path in CHECKPOINT_DIR.glob("*.npz"):
        old_path.unlink()

    base_params = params_for(fiducial, mdot_edd, float(source["ratio"]), BASE_R_OUT_RG)
    for pivot in PIVOTS:
        for variant in block_variants(base_params):
            z, result, initial_selected = solve_variant(
                np.asarray(source["z"], dtype=float),
                base_params,
                pivot,
                np.asarray(variant["rows"], dtype=int),
                float(variant["outer_weight"]),
            )
            row = row_from_solution(
                family="subset",
                label=str(variant["label"]),
                description=str(variant["description"]),
                ratio=float(source["ratio"]),
                params=base_params,
                pivot=pivot,
                z0=np.asarray(source["z"], dtype=float),
                z=z,
                result=result,
                initial_selected=initial_selected,
                rows=np.asarray(variant["rows"], dtype=int),
                outer_weight=float(variant["outer_weight"]),
            )
            rows.append(row)
            save_checkpoint(row)
            print(
                f"subset {variant['label']} pivot={pivot} selected {initial_selected:.3e}->{row['selected_final']:.3e} "
                f"active={row['active_max']:.3e} dom={row['dominant']}",
                flush=True,
            )

    for R_out_rg in R_OUT_SWEEP:
        params = params_for(fiducial, mdot_edd, float(source["ratio"]), R_out_rg)
        z0 = (
            np.asarray(source["z"], dtype=float)
            if R_out_rg == BASE_R_OUT_RG
            else remap_profile_to_new_sonic_grid(source["profile"], params)
        )
        for pivot in PIVOTS:
            rowspec = row_slices(params)["all"]
            z, result, initial_selected = solve_variant(z0, params, pivot, rowspec, 1.0)
            row = row_from_solution(
                family="R_out",
                label="full",
                description="full square system at varied outer radius",
                ratio=float(source["ratio"]),
                params=params,
                pivot=pivot,
                z0=z0,
                z=z,
                result=result,
                initial_selected=initial_selected,
                rows=rowspec,
                outer_weight=1.0,
            )
            rows.append(row)
            save_checkpoint(row)
            print(
                f"R_out={R_out_rg:g} pivot={pivot} selected {initial_selected:.3e}->{row['selected_final']:.3e} "
                f"active={row['active_max']:.3e} dom={row['dominant']}",
                flush=True,
            )

    write_table(rows)
    print(f"wrote {TABLE_OUTPUT}")


if __name__ == "__main__":
    main()
