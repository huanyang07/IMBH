"""Gradually move the matched outer boundary at fixed Mdot."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from imri_qpe.layer3_minidisk_1d.transonic_collocation import (
    profile_from_state_vector,
    residual_audit_from_state_vector,
    square_collocation_residual,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot

from run_transonic_outer_thermal_match_audit import (
    SOURCE_R_OUT_RG,
    custom_square_residual,
    fmt,
    load_source,
    outer_residual,
    params_for,
    polyfit_outer_log_slopes,
    remap_state_with_outer_extrapolation,
    solve_variant,
)


ROOT = Path(__file__).resolve().parents[1]
TABLE_OUTPUT = ROOT / "outputs" / "tables" / "transonic_rout_continuation_audit.md"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "transonic_rout_continuation_audit"

PIVOT = "C1"
CLOSURE = "matched_full"
R_OUT_SEQUENCE_RG = (3000.0, 3500.0, 4000.0, 5000.0, 6500.0, 8000.0, 10000.0)
SLOPE_REFRESHES = 3


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    return value


def active_physical_max(audit, outer) -> float:
    return max(
        abs(audit.interval_radial_max),
        abs(audit.interval_energy_max),
        abs(float(outer[0])),
        abs(float(outer[1])),
        abs(audit.sonic_D),
        abs(audit.sonic_K),
    )


def dominant_block(audit, outer) -> str:
    values = {
        "interval_R": abs(audit.interval_radial_max),
        "interval_E": abs(audit.interval_energy_max),
        "outer_1": abs(float(outer[0])),
        "outer_2": abs(float(outer[1])),
        "D": abs(audit.sonic_D),
        "K": abs(audit.sonic_K),
        "C1": abs(audit.sonic_C1),
        "C2": abs(audit.sonic_C2),
    }
    return max(values, key=values.get)


def slope_delta(input_slopes: tuple[float, float], output_slopes: tuple[float, float]) -> float:
    return float(np.max(np.abs(np.asarray(output_slopes, dtype=float) - np.asarray(input_slopes, dtype=float))))


def row_from_result(
    *,
    ratio: float,
    R_out_rg: float,
    refresh: int,
    params,
    z0: np.ndarray,
    slopes: tuple[float, float],
    result,
) -> dict[str, object]:
    z = np.asarray(result.x, dtype=float)
    selected_initial = custom_square_residual(z0, params, PIVOT, CLOSURE, slopes)
    selected_final = custom_square_residual(z, params, PIVOT, CLOSURE, slopes)
    outer = outer_residual(z, params, CLOSURE, slopes)
    audit = residual_audit_from_state_vector(z, params)
    legacy_audit = residual_audit_from_state_vector(z, replace(params, outer_closure="thin_value", outer_match_log_slopes=None))
    profile = profile_from_state_vector(z, params)
    output_slopes = polyfit_outer_log_slopes(z, params)
    legacy_square = square_collocation_residual(z, params, pivot=PIVOT)
    return {
        "ratio": ratio,
        "R_out_rg": R_out_rg,
        "refresh": refresh,
        "pivot": PIVOT,
        "closure": CLOSURE,
        "g_u_in": float(slopes[0]),
        "g_T_in": float(slopes[1]),
        "g_u_out": float(output_slopes[0]),
        "g_T_out": float(output_slopes[1]),
        "slope_delta": slope_delta(slopes, output_slopes),
        "selected_initial": float(np.max(np.abs(selected_initial))),
        "selected_final": float(np.max(np.abs(selected_final))),
        "legacy_square": float(np.max(np.abs(legacy_square))),
        "physical_active": active_physical_max(audit, outer),
        "dominant": dominant_block(audit, outer),
        "interval_R": audit.interval_radial_max,
        "interval_E": audit.interval_energy_max,
        "outer_1": float(outer[0]),
        "outer_2": float(outer[1]),
        "legacy_outer_Omega": legacy_audit.outer_omega,
        "legacy_outer_E": legacy_audit.outer_energy,
        "D": audit.sonic_D,
        "C1": audit.sonic_C1,
        "C2": audit.sonic_C2,
        "K": audit.sonic_K,
        "compat_max": max(abs(audit.sonic_C1), abs(audit.sonic_C2), abs(audit.sonic_K)),
        "Rson_rg": float(profile.sonic_radius / params.r_g),
        "lambda0": float(profile.lambda0),
        "max_HR": float(np.max(profile.H_over_R)),
        "outer_HR": audit.outer_H_over_R,
        "outer_Qadv_Qvisc": audit.outer_Qadv_over_Qvisc,
        "int_adv": profile.integrated_advective_fraction,
        "nfev": int(result.nfev),
        "success": bool(result.success),
        "message": str(result.message),
        "z": z,
    }


def save_checkpoint(row: dict[str, object]) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"R{float(row['R_out_rg']):.0f}_refresh{int(row['refresh'])}_{float(row['ratio']):.8f}".replace(".", "p")
    payload = {key: json_safe(value) for key, value in row.items() if key != "z"}
    np.savez_compressed(
        CHECKPOINT_DIR / f"{stem}.npz",
        z=np.asarray(row["z"], dtype=float),
        ratio=np.array(row["ratio"]),
        R_out_rg=np.array(row["R_out_rg"]),
        refresh=np.array(row["refresh"]),
        row_json=np.array(json.dumps(payload, sort_keys=True)),
    )


def write_table(rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Gradual R_out Continuation Audit",
        "",
        "Generated by `scripts/run_transonic_rout_continuation_audit.py`.",
        "",
        "Fixed `Mdot/Edd ~= 0.9028`, `N=64`, `matched_full` outer value closure, `C1` sonic pivot. At each radius the smooth outer polyfit slopes are refreshed between solves. This is an R_out diagnostic, not a replacement for independent low-Mdot continuation.",
        "",
        "| R_out/rg | refresh | selected initial | selected final | physical active | dominant | slope delta | g_u in | g_T in | g_u out | g_T out | interval R | interval E | outer 1 | outer 2 | legacy Omega | legacy E | D | C1 | C2 | K | compat max | Rson/rg | lambda0 | max H/R | outer H/R | outer Qadv/Qvisc | int adv | nfev | success | message |",
        "|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {R_out_rg} | {refresh} | {selected_initial} | {selected_final} | {physical_active} | {dominant} | "
            "{slope_delta} | {g_u_in} | {g_T_in} | {g_u_out} | {g_T_out} | {interval_R} | {interval_E} | "
            "{outer_1} | {outer_2} | {legacy_outer_Omega} | {legacy_outer_E} | {D} | {C1} | {C2} | {K} | "
            "{compat_max} | {Rson_rg} | {lambda0} | {max_HR} | {outer_HR} | {outer_Qadv_Qvisc} | {int_adv} | "
            "{nfev} | {success} | {message} |".format(
                R_out_rg=fmt(float(row["R_out_rg"])),
                refresh=row["refresh"],
                selected_initial=fmt(float(row["selected_initial"])),
                selected_final=fmt(float(row["selected_final"])),
                physical_active=fmt(float(row["physical_active"])),
                dominant=row["dominant"],
                slope_delta=fmt(float(row["slope_delta"])),
                g_u_in=fmt(float(row["g_u_in"])),
                g_T_in=fmt(float(row["g_T_in"])),
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
                Rson_rg=fmt(float(row["Rson_rg"])),
                lambda0=fmt(float(row["lambda0"])),
                max_HR=fmt(float(row["max_HR"])),
                outer_HR=fmt(float(row["outer_HR"])),
                outer_Qadv_Qvisc=fmt(float(row["outer_Qadv_Qvisc"])),
                int_adv=fmt(float(row["int_adv"])),
                nfev=row["nfev"],
                success="yes" if row["success"] else "no",
                message=str(row["message"]).replace("|", "/"),
            )
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")


def best_row(rows: list[dict[str, object]]) -> dict[str, object]:
    return min(rows, key=lambda row: float(row["physical_active"]))


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    source_z, ratio = load_source()
    previous_params = params_for(fiducial, mdot_edd, ratio, SOURCE_R_OUT_RG)
    previous_z = source_z

    rows: list[dict[str, object]] = []
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for old_path in CHECKPOINT_DIR.glob("*.npz"):
        old_path.unlink()

    for R_out_rg in R_OUT_SEQUENCE_RG:
        params = params_for(fiducial, mdot_edd, ratio, R_out_rg)
        if np.isclose(R_out_rg, previous_params.R_out_rg):
            z_seed = previous_z
        else:
            z_seed = remap_state_with_outer_extrapolation(previous_z, previous_params, params)
        radius_rows: list[dict[str, object]] = []
        for refresh in range(SLOPE_REFRESHES):
            slopes = polyfit_outer_log_slopes(z_seed, params)
            result = solve_variant(z_seed, params, PIVOT, CLOSURE, slopes)
            row = row_from_result(
                ratio=ratio,
                R_out_rg=R_out_rg,
                refresh=refresh,
                params=params,
                z0=z_seed,
                slopes=slopes,
                result=result,
            )
            rows.append(row)
            radius_rows.append(row)
            save_checkpoint(row)
            write_table(rows)
            print(
                f"R={R_out_rg:.0f} refresh={refresh} selected {row['selected_initial']:.3e}->{row['selected_final']:.3e} "
                f"physical={row['physical_active']:.3e} dom={row['dominant']} slope_delta={row['slope_delta']:.3e}",
                flush=True,
            )
            z_seed = np.asarray(row["z"], dtype=float)
        chosen = best_row(radius_rows)
        previous_z = np.asarray(chosen["z"], dtype=float)
        previous_params = params

    print(f"wrote {TABLE_OUTPUT}")


if __name__ == "__main__":
    main()
