"""Audit sensitivity of the direct full-slope outer closure to matched slopes."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from imri_qpe.layer3_minidisk_1d import TransonicSlimParams
from imri_qpe.layer3_minidisk_1d.transonic_collocation import (
    residual_audit_from_state_vector,
    square_collocation_residual,
    square_jac_sparsity_pattern,
    state_bounds,
    unpack_state,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot


ROOT = Path(__file__).resolve().parents[1]
SLOPE_SOURCE_CHECKPOINT = ROOT / "outputs" / "checkpoints" / "transonic_rout_continuation_audit" / "R5000_refresh2_0p90277664.npz"
SEED_CHECKPOINT = ROOT / "outputs" / "checkpoints" / "transonic_full_slope_match_audit" / "full_slope_differential_C1_0p90277664.npz"
TABLE_OUTPUT = ROOT / "outputs" / "tables" / "transonic_full_slope_sensitivity_audit.md"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "transonic_full_slope_sensitivity_audit"

R_OUT_RG = 5000.0
N_NODES = 64
PIVOT = "C1"
MAX_NFEV = 200
DELTAS = (1.0e-3, 3.0e-3, 1.0e-2)


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


def load_checkpoint(path: Path) -> tuple[np.ndarray, float]:
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["z"], dtype=float), float(data["ratio"])


def params_for(fiducial: FiducialParams, mdot_edd: float, ratio: float, slopes: tuple[float, float]) -> TransonicSlimParams:
    return TransonicSlimParams(
        M2_g=fiducial.M2_g,
        Mdot_g_s=ratio * mdot_edd,
        alpha=fiducial.alpha_cool,
        n_nodes=N_NODES,
        R_out_rg=R_OUT_RG,
        residual_tol=1.0e-6,
        max_nfev=MAX_NFEV,
        outer_closure="full_slope_match",
        outer_match_log_slopes=slopes,
        interval_residual_form="differential",
    )


def polyfit_outer_log_slopes(z: np.ndarray, params: TransonicSlimParams, n_fit: int = 8) -> tuple[float, float]:
    logu, logT, _logR_son, _lambda0, logR = unpack_state(z, params)
    count = min(int(n_fit), len(logR))
    x = logR[-count:] - logR[-1]
    gu = float(np.polyder(np.poly1d(np.polyfit(x, logu[-count:], min(2, count - 1))))(0.0))
    gT = float(np.polyder(np.poly1d(np.polyfit(x, logT[-count:], min(2, count - 1))))(0.0))
    return gu, gT


def slope_specs(base_slopes: tuple[float, float]) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = [
        {"label": "baseline", "axis": "none", "delta": 0.0, "slopes": base_slopes},
    ]
    for delta in DELTAS:
        for axis, index in (("g_u", 0), ("g_T", 1)):
            for sign in (-1.0, 1.0):
                slopes = list(base_slopes)
                slopes[index] += sign * delta
                specs.append(
                    {
                        "label": f"{axis}_{sign:+.0f}_{delta:.0e}",
                        "axis": axis,
                        "delta": sign * delta,
                        "slopes": (float(slopes[0]), float(slopes[1])),
                    }
                )
    return specs


def solve_variant(z0: np.ndarray, params: TransonicSlimParams):
    lower, upper = state_bounds(params)
    z_start = np.clip(np.asarray(z0, dtype=float), lower + 1.0e-12, upper - 1.0e-12)
    return least_squares(
        lambda trial: square_collocation_residual(trial, params, pivot=PIVOT),
        z_start,
        jac_sparsity=square_jac_sparsity_pattern(params),
        bounds=(lower, upper),
        x_scale="jac",
        diff_step=3.0e-5,
        ftol=1.0e-11,
        xtol=1.0e-11,
        gtol=1.0e-10,
        max_nfev=MAX_NFEV,
    )


def active_physical_max(audit) -> float:
    return max(
        abs(audit.interval_radial_max),
        abs(audit.interval_energy_max),
        abs(audit.outer_omega),
        abs(audit.outer_energy),
        abs(audit.sonic_D),
        abs(audit.sonic_C1),
        abs(audit.sonic_C2),
        abs(audit.sonic_K),
    )


def dominant_block(audit) -> str:
    values = {
        "interval_R": abs(audit.interval_radial_max),
        "interval_E": abs(audit.interval_energy_max),
        "outer_1": abs(audit.outer_omega),
        "outer_2": abs(audit.outer_energy),
        "D": abs(audit.sonic_D),
        "C1": abs(audit.sonic_C1),
        "C2": abs(audit.sonic_C2),
        "K": abs(audit.sonic_K),
    }
    return max(values, key=values.get)


def row_from_result(
    *,
    spec: dict[str, object],
    ratio: float,
    params: TransonicSlimParams,
    z0: np.ndarray,
    result,
) -> dict[str, object]:
    z = np.asarray(result.x, dtype=float)
    initial = square_collocation_residual(z0, params, pivot=PIVOT)
    final = square_collocation_residual(z, params, pivot=PIVOT)
    initial_audit = residual_audit_from_state_vector(z0, params)
    audit = residual_audit_from_state_vector(z, params)
    legacy_audit = residual_audit_from_state_vector(z, replace(params, outer_closure="thin_value", outer_match_log_slopes=None))
    output_slopes = polyfit_outer_log_slopes(z, params)
    slopes = params.outer_match_log_slopes or (np.nan, np.nan)
    return {
        "label": spec["label"],
        "axis": spec["axis"],
        "delta": float(spec["delta"]),
        "ratio": ratio,
        "g_u": float(slopes[0]),
        "g_T": float(slopes[1]),
        "g_u_out": float(output_slopes[0]),
        "g_T_out": float(output_slopes[1]),
        "slope_delta_out": float(np.max(np.abs(np.asarray(output_slopes) - np.asarray(slopes, dtype=float)))),
        "initial_selected": float(np.max(np.abs(initial))),
        "initial_physical": active_physical_max(initial_audit),
        "initial_dominant": dominant_block(initial_audit),
        "final_selected": float(np.max(np.abs(final))),
        "final_physical": active_physical_max(audit),
        "dominant": dominant_block(audit),
        "interval_R": audit.interval_radial_max,
        "interval_E": audit.interval_energy_max,
        "outer_1": audit.outer_omega,
        "outer_2": audit.outer_energy,
        "legacy_outer_Omega": legacy_audit.outer_omega,
        "legacy_outer_E": legacy_audit.outer_energy,
        "D": audit.sonic_D,
        "C1": audit.sonic_C1,
        "C2": audit.sonic_C2,
        "K": audit.sonic_K,
        "compat_max": max(abs(audit.sonic_C1), abs(audit.sonic_C2), abs(audit.sonic_K)),
        "nfev": int(result.nfev),
        "success": bool(result.success),
        "message": str(result.message),
        "z": z,
    }


def save_checkpoint(row: dict[str, object]) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{row['label']}_{float(row['ratio']):.8f}".replace("+", "p").replace("-", "m").replace(".", "p")
    payload = {key: value for key, value in row.items() if key != "z"}
    np.savez_compressed(
        CHECKPOINT_DIR / f"{stem}.npz",
        z=np.asarray(row["z"], dtype=float),
        ratio=np.array(row["ratio"]),
        row_json=np.array(row_json(payload)),
    )


def write_table(rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Full Slope Match Sensitivity Audit",
        "",
        "Generated by `scripts/run_transonic_full_slope_sensitivity_audit.py`.",
        "",
        "Perturbs the smooth polyfit slopes used by `outer_closure='full_slope_match'` at `R_out=5000 rg`, `Mdot/Edd ~= 0.9028`, `N=64`, pivot `C1`. The seed is the previous `full_slope_differential_C1` polished state.",
        "",
        "| label | axis | delta | g_u | g_T | initial selected | initial physical | initial dominant | final selected | final physical | dominant | slope delta out | g_u out | g_T out | interval R | interval E | outer 1 | outer 2 | legacy Omega | legacy E | D | C1 | C2 | K | compat max | nfev | success | message |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {axis} | {delta} | {g_u} | {g_T} | {initial_selected} | {initial_physical} | {initial_dominant} | "
            "{final_selected} | {final_physical} | {dominant} | {slope_delta_out} | {g_u_out} | {g_T_out} | "
            "{interval_R} | {interval_E} | {outer_1} | {outer_2} | {legacy_outer_Omega} | {legacy_outer_E} | "
            "{D} | {C1} | {C2} | {K} | {compat_max} | {nfev} | {success} | {message} |".format(
                label=row["label"],
                axis=row["axis"],
                delta=fmt(float(row["delta"])),
                g_u=fmt(float(row["g_u"])),
                g_T=fmt(float(row["g_T"])),
                initial_selected=fmt(float(row["initial_selected"])),
                initial_physical=fmt(float(row["initial_physical"])),
                initial_dominant=row["initial_dominant"],
                final_selected=fmt(float(row["final_selected"])),
                final_physical=fmt(float(row["final_physical"])),
                dominant=row["dominant"],
                slope_delta_out=fmt(float(row["slope_delta_out"])),
                g_u_out=fmt(float(row["g_u_out"])),
                g_T_out=fmt(float(row["g_T_out"])),
                interval_R=fmt(float(row["interval_R"])),
                interval_E=fmt(float(row["interval_E"])),
                outer_1=fmt(float(row["outer_1"])),
                outer_2=fmt(float(row["outer_2"])),
                legacy_outer_Omega=fmt(float(row["legacy_outer_Omega"])),
                legacy_outer_E=fmt(float(row["legacy_outer_E"])),
                D=fmt(float(row["D"])),
                C1=fmt(float(row["C1"])),
                C2=fmt(float(row["C2"])),
                K=fmt(float(row["K"])),
                compat_max=fmt(float(row["compat_max"])),
                nfev=row["nfev"],
                success="yes" if row["success"] else "no",
                message=str(row["message"]).replace("|", "/"),
            )
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    slope_z, ratio = load_checkpoint(SLOPE_SOURCE_CHECKPOINT)
    seed_z, seed_ratio = load_checkpoint(SEED_CHECKPOINT)
    if not np.isclose(ratio, seed_ratio):
        raise RuntimeError("slope and seed checkpoints have different accretion rates")
    slope_params = params_for(fiducial, mdot_edd, ratio, slopes=(0.0, 0.0))
    base_slopes = polyfit_outer_log_slopes(slope_z, slope_params)

    rows: list[dict[str, object]] = []
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for old_path in CHECKPOINT_DIR.glob("*.npz"):
        old_path.unlink()

    for spec in slope_specs(base_slopes):
        params = params_for(fiducial, mdot_edd, ratio, slopes=spec["slopes"])
        result = solve_variant(seed_z, params)
        row = row_from_result(spec=spec, ratio=ratio, params=params, z0=seed_z, result=result)
        rows.append(row)
        save_checkpoint(row)
        write_table(rows)
        print(
            f"{row['label']} initial={row['initial_physical']:.3e} final={row['final_physical']:.3e} "
            f"dom={row['dominant']} nfev={row['nfev']}",
            flush=True,
        )

    print(f"wrote {TABLE_OUTPUT}")


if __name__ == "__main__":
    main()
