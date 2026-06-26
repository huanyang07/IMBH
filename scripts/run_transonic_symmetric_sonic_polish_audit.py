"""Polish the best fixed-Mdot R_out anchor with symmetric sonic residuals."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from imri_qpe.layer3_minidisk_1d.transonic_collocation import (
    _sonic_component_values,
    profile_from_state_vector,
    residual_audit_from_state_vector,
    square_collocation_residual,
    state_bounds,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot

from run_transonic_outer_thermal_match_audit import (
    custom_square_residual,
    fmt,
    outer_residual,
    params_for,
    polyfit_outer_log_slopes,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CHECKPOINT = ROOT / "outputs" / "checkpoints" / "transonic_rout_continuation_audit" / "R5000_refresh2_0p90277664.npz"
TABLE_OUTPUT = ROOT / "outputs" / "tables" / "transonic_symmetric_sonic_polish_audit.md"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "transonic_symmetric_sonic_polish_audit"

R_OUT_RG = 5000.0
CLOSURE = "matched_full"
SLOPE_REFRESHES = 2
MAX_NFEV = 80

VARIANTS = (
    ("square_C1", ("D", "C1"), 1.0),
    ("square_C2", ("D", "C2"), 1.0),
    ("symmetric_D_C1_C2", ("D", "C1", "C2"), 1.0),
    ("symmetric_D_C1_C2_K_w0p5", ("D", "C1", "C2", "K"), 0.5),
    ("symmetric_D_C1_C2_K_w1", ("D", "C1", "C2", "K"), 1.0),
    ("symmetric_D_C1_C2_K_w2", ("D", "C1", "C2", "K"), 2.0),
)


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def load_seed() -> tuple[np.ndarray, float]:
    with np.load(SOURCE_CHECKPOINT, allow_pickle=False) as data:
        return np.asarray(data["z"], dtype=float), float(data["ratio"])


def residual_vector(z: np.ndarray, params, slopes, sonic_components: tuple[str, ...], sonic_weight: float) -> np.ndarray:
    interval_end = 2 * (params.n_nodes - 1)
    base = custom_square_residual(z, params, "C1", CLOSURE, slopes)
    residual = np.zeros(interval_end + 2 + len(sonic_components), dtype=float)
    residual[:interval_end] = base[:interval_end]
    residual[interval_end : interval_end + 2] = outer_residual(z, params, CLOSURE, slopes)
    residual[interval_end + 2 :] = sonic_weight * _sonic_component_values(z, params, sonic_components)
    return residual


def sparsity_pattern(params, n_sonic: int):
    from scipy.sparse import lil_matrix

    unknown_size = 2 * params.n_nodes + 2
    n_rows = 2 * (params.n_nodes - 1) + 2 + n_sonic
    pattern = lil_matrix((n_rows, unknown_size), dtype=int)
    row = 0
    for idx in range(params.n_nodes - 1):
        columns = (
            idx,
            idx + 1,
            params.n_nodes + idx,
            params.n_nodes + idx + 1,
            unknown_size - 2,
            unknown_size - 1,
        )
        for col in columns:
            pattern[row : row + 2, col] = 1
        row += 2
    for col in (params.n_nodes - 1, 2 * params.n_nodes - 1, unknown_size - 2, unknown_size - 1):
        pattern[row : row + 2, col] = 1
    row += 2
    for col in (0, params.n_nodes, unknown_size - 2, unknown_size - 1):
        pattern[row : row + n_sonic, col] = 1
    return pattern.tocsr()


def solve_variant(z0: np.ndarray, params, slopes, sonic_components: tuple[str, ...], sonic_weight: float):
    lower, upper = state_bounds(params)
    z_start = np.clip(np.asarray(z0, dtype=float), lower + 1.0e-12, upper - 1.0e-12)
    return least_squares(
        lambda trial: residual_vector(trial, params, slopes, sonic_components, sonic_weight),
        z_start,
        jac_sparsity=sparsity_pattern(params, len(sonic_components)),
        bounds=(lower, upper),
        x_scale="jac",
        diff_step=3.0e-5,
        ftol=1.0e-11,
        xtol=1.0e-11,
        gtol=1.0e-10,
        max_nfev=MAX_NFEV,
    )


def active_physical_max(audit, outer) -> float:
    return max(
        abs(audit.interval_radial_max),
        abs(audit.interval_energy_max),
        abs(float(outer[0])),
        abs(float(outer[1])),
        abs(audit.sonic_D),
        abs(audit.sonic_C1),
        abs(audit.sonic_C2),
        abs(audit.sonic_K),
    )


def dominant_block(audit, outer) -> str:
    values = {
        "interval_R": abs(audit.interval_radial_max),
        "interval_E": abs(audit.interval_energy_max),
        "outer_1": abs(float(outer[0])),
        "outer_2": abs(float(outer[1])),
        "D": abs(audit.sonic_D),
        "C1": abs(audit.sonic_C1),
        "C2": abs(audit.sonic_C2),
        "K": abs(audit.sonic_K),
    }
    return max(values, key=values.get)


def row_from_result(
    *,
    label: str,
    ratio: float,
    refresh: int,
    params,
    z0: np.ndarray,
    slopes: tuple[float, float],
    sonic_components: tuple[str, ...],
    sonic_weight: float,
    result,
) -> dict[str, object]:
    z = np.asarray(result.x, dtype=float)
    selected_initial = residual_vector(z0, params, slopes, sonic_components, sonic_weight)
    selected_final = residual_vector(z, params, slopes, sonic_components, sonic_weight)
    square_c1 = custom_square_residual(z, params, "C1", CLOSURE, slopes)
    square_c2 = custom_square_residual(z, params, "C2", CLOSURE, slopes)
    outer = outer_residual(z, params, CLOSURE, slopes)
    audit = residual_audit_from_state_vector(z, params)
    legacy_audit = residual_audit_from_state_vector(z, replace(params, outer_closure="thin_value", outer_match_log_slopes=None))
    profile = profile_from_state_vector(z, params)
    output_slopes = polyfit_outer_log_slopes(z, params)
    return {
        "label": label,
        "ratio": ratio,
        "R_out_rg": params.R_out_rg,
        "refresh": refresh,
        "sonic_components": sonic_components,
        "sonic_weight": sonic_weight,
        "g_u_in": float(slopes[0]),
        "g_T_in": float(slopes[1]),
        "g_u_out": float(output_slopes[0]),
        "g_T_out": float(output_slopes[1]),
        "slope_delta": float(np.max(np.abs(np.asarray(output_slopes) - np.asarray(slopes)))),
        "selected_initial": float(np.max(np.abs(selected_initial))),
        "selected_final": float(np.max(np.abs(selected_final))),
        "square_C1": float(np.max(np.abs(square_c1))),
        "square_C2": float(np.max(np.abs(square_c2))),
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
    stem = f"{row['label']}_refresh{int(row['refresh'])}_{float(row['ratio']):.8f}".replace(".", "p")
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
        "# Symmetric Sonic Polish Audit",
        "",
        "Generated by `scripts/run_transonic_symmetric_sonic_polish_audit.py`.",
        "",
        "Starts from the best gradual `R_out=5000 rg` matched-full checkpoint. Rows compare square-pivot sonic residuals with overdetermined symmetric sonic residuals.",
        "",
        "| label | refresh | sonic components | sonic weight | selected initial | selected final | square C1 | square C2 | physical active | dominant | slope delta | outer 1 | outer 2 | D | C1 | C2 | K | compat max | interval R | interval E | legacy Omega | legacy E | Rson/rg | lambda0 | max H/R | outer H/R | outer Qadv/Qvisc | int adv | nfev | success | message |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {refresh} | {sonic_components} | {sonic_weight} | {selected_initial} | {selected_final} | "
            "{square_C1} | {square_C2} | {physical_active} | {dominant} | {slope_delta} | {outer_1} | {outer_2} | "
            "{D} | {C1} | {C2} | {K} | {compat_max} | {interval_R} | {interval_E} | {legacy_outer_Omega} | "
            "{legacy_outer_E} | {Rson_rg} | {lambda0} | {max_HR} | {outer_HR} | {outer_Qadv_Qvisc} | {int_adv} | "
            "{nfev} | {success} | {message} |".format(
                label=row["label"],
                refresh=row["refresh"],
                sonic_components=",".join(row["sonic_components"]),
                sonic_weight=fmt(float(row["sonic_weight"])),
                selected_initial=fmt(float(row["selected_initial"])),
                selected_final=fmt(float(row["selected_final"])),
                square_C1=fmt(float(row["square_C1"])),
                square_C2=fmt(float(row["square_C2"])),
                physical_active=fmt(float(row["physical_active"])),
                dominant=row["dominant"],
                slope_delta=fmt(float(row["slope_delta"])),
                outer_1=fmt(float(row["outer_1"])),
                outer_2=fmt(float(row["outer_2"])),
                D=fmt(float(row["D"])),
                C1=fmt(float(row["C1"])),
                C2=fmt(float(row["C2"])),
                K=fmt(float(row["K"])),
                compat_max=fmt(float(row["compat_max"])),
                interval_R=fmt(float(row["interval_R"])),
                interval_E=fmt(float(row["interval_E"])),
                legacy_outer_Omega=fmt(float(row["legacy_outer_Omega"])),
                legacy_outer_E=fmt(float(row["legacy_outer_E"])),
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


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    seed_z, ratio = load_seed()
    params = params_for(fiducial, mdot_edd, ratio, R_OUT_RG)

    rows: list[dict[str, object]] = []
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for old_path in CHECKPOINT_DIR.glob("*.npz"):
        old_path.unlink()

    for label, sonic_components, sonic_weight in VARIANTS:
        z_seed = np.asarray(seed_z, dtype=float)
        for refresh in range(SLOPE_REFRESHES):
            slopes = polyfit_outer_log_slopes(z_seed, params)
            result = solve_variant(z_seed, params, slopes, sonic_components, sonic_weight)
            row = row_from_result(
                label=label,
                ratio=ratio,
                refresh=refresh,
                params=params,
                z0=z_seed,
                slopes=slopes,
                sonic_components=sonic_components,
                sonic_weight=sonic_weight,
                result=result,
            )
            rows.append(row)
            save_checkpoint(row)
            write_table(rows)
            print(
                f"{label} refresh={refresh} selected {row['selected_initial']:.3e}->{row['selected_final']:.3e} "
                f"physical={row['physical_active']:.3e} dom={row['dominant']}",
                flush=True,
            )
            z_seed = np.asarray(row["z"], dtype=float)

    print(f"wrote {TABLE_OUTPUT}")


if __name__ == "__main__":
    main()
