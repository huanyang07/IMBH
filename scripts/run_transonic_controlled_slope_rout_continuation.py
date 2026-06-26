"""Continue R_out while treating outer slopes as controlled checkpoint data."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from imri_qpe.layer3_minidisk_1d import TransonicSlimParams
from imri_qpe.layer3_minidisk_1d.transonic_collocation import (
    computational_grid,
    pack_state,
    profile_from_state_vector,
    reduced_outer_log_slopes,
    residual_audit_from_state_vector,
    state_bounds,
    unpack_state,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot

from run_transonic_full_slope_tuned_sonic_audit import residual_vector, sparsity_pattern
from run_transonic_outer_slope_calibration_audit import (
    CHECKPOINT_DIR as CALIBRATION_CHECKPOINT_DIR,
    fmt,
    polyfit_outer_log_slopes,
)


ROOT = Path(__file__).resolve().parents[1]
TABLE_OUTPUT = ROOT / "outputs" / "tables" / "transonic_controlled_slope_rout_continuation.md"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "transonic_controlled_slope_rout_continuation"

R_OUT_SEQUENCE_RG = (5000.0, 5500.0, 6000.0, 6500.0)
N_NODES = 64
MAX_NFEV = 250
SONIC_MODE = "symmetric"
SONIC_PIVOT = "C1"
SONIC_COMPONENTS = ("D", "C1", "C2")
EMA_ETA = 0.35


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def local_row_json(row: dict[str, object]) -> str:
    return json.dumps({key: json_safe(value) for key, value in row.items()}, sort_keys=True)


def params_for(
    fiducial: FiducialParams,
    mdot_edd: float,
    ratio: float,
    R_out_rg: float,
    slopes: tuple[float, float],
) -> TransonicSlimParams:
    return TransonicSlimParams(
        M2_g=fiducial.M2_g,
        Mdot_g_s=ratio * mdot_edd,
        alpha=fiducial.alpha_cool,
        n_nodes=N_NODES,
        R_out_rg=R_out_rg,
        residual_tol=1.0e-6,
        max_nfev=MAX_NFEV,
        outer_closure="full_slope_match",
        outer_match_log_slopes=slopes,
        interval_residual_form="differential",
    )


def calibration_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(CALIBRATION_CHECKPOINT_DIR.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["row_json"].item()))
            metadata["checkpoint"] = str(path)
            metadata["z"] = np.asarray(data["z"], dtype=float)
            records.append(metadata)
    if not records:
        raise RuntimeError("run run_transonic_outer_slope_calibration_audit.py before controlled R_out continuation")
    return records


def best_calibration_record() -> dict[str, object]:
    return min(calibration_records(), key=lambda row: float(row["final_physical"]))


def remap_state_with_slopes(
    z: np.ndarray,
    old_params: TransonicSlimParams,
    new_params: TransonicSlimParams,
    slopes: tuple[float, float],
) -> np.ndarray:
    logu_old, logT_old, logR_son, lambda0, logR_old = unpack_state(z, old_params)
    logR_new = computational_grid(new_params, logR_son)
    logu_new = np.interp(logR_new, logR_old, logu_old)
    logT_new = np.interp(logR_new, logR_old, logT_old)
    high = logR_new > logR_old[-1]
    if np.any(high):
        logu_new[high] = logu_old[-1] + slopes[0] * (logR_new[high] - logR_old[-1])
        logT_new[high] = logT_old[-1] + slopes[1] * (logR_new[high] - logR_old[-1])
    return pack_state(logu_new, logT_new, logR_son, lambda0)


def target_slopes(
    z: np.ndarray,
    params: TransonicSlimParams,
    spec: dict[str, object],
    *,
    use_baseline: bool,
) -> tuple[float, float]:
    if use_baseline:
        return polyfit_outer_log_slopes(z, params, n_fit=8, degree=2, skip_outer=0)
    if spec["estimator"] == "reduced":
        _logu, _logT, _logR_son, lambda0, _logR = unpack_state(z, params)
        slopes = reduced_outer_log_slopes(params, lambda0)
    else:
        slopes = polyfit_outer_log_slopes(
            z,
            params,
            n_fit=int(spec["n_fit"]),
            degree=int(spec["degree"]),
            skip_outer=int(spec["skip_outer"]),
        )
    return float(slopes[0] + float(spec["dg_u"])), float(slopes[1] + float(spec["dg_T"]))


def update_slopes(mode: str, previous: tuple[float, float] | None, target: tuple[float, float]) -> tuple[float, float]:
    if mode.startswith("raw"):
        return target
    if mode.startswith("held"):
        return previous if previous is not None else target
    if previous is None:
        return target
    previous_array = np.asarray(previous, dtype=float)
    target_array = np.asarray(target, dtype=float)
    used = (1.0 - EMA_ETA) * previous_array + EMA_ETA * target_array
    return float(used[0]), float(used[1])


def solve_variant(z0: np.ndarray, params: TransonicSlimParams):
    lower, upper = state_bounds(params)
    z_start = np.clip(np.asarray(z0, dtype=float), lower + 1.0e-12, upper - 1.0e-12)
    return least_squares(
        lambda trial: residual_vector(trial, params, SONIC_MODE, SONIC_PIVOT, SONIC_COMPONENTS),
        z_start,
        jac_sparsity=sparsity_pattern(params, SONIC_MODE, SONIC_COMPONENTS),
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
    mode: str,
    calibration_label: str,
    ratio: float,
    R_out_rg: float,
    target: tuple[float, float],
    used: tuple[float, float],
    params: TransonicSlimParams,
    z0: np.ndarray,
    result,
) -> dict[str, object]:
    z = np.asarray(result.x, dtype=float)
    initial = residual_vector(z0, params, SONIC_MODE, SONIC_PIVOT, SONIC_COMPONENTS)
    final = residual_vector(z, params, SONIC_MODE, SONIC_PIVOT, SONIC_COMPONENTS)
    audit = residual_audit_from_state_vector(z, params)
    legacy_audit = residual_audit_from_state_vector(z, replace(params, outer_closure="thin_value", outer_match_log_slopes=None))
    output_slopes = polyfit_outer_log_slopes(z, params, n_fit=8, degree=2, skip_outer=0)
    profile = profile_from_state_vector(z, params)
    return {
        "mode": mode,
        "calibration_label": calibration_label,
        "ratio": ratio,
        "R_out_rg": R_out_rg,
        "g_u_target": float(target[0]),
        "g_T_target": float(target[1]),
        "g_u_used": float(used[0]),
        "g_T_used": float(used[1]),
        "g_u_out": float(output_slopes[0]),
        "g_T_out": float(output_slopes[1]),
        "target_used_delta": float(np.max(np.abs(np.asarray(target) - np.asarray(used)))),
        "used_out_delta": float(np.max(np.abs(np.asarray(output_slopes) - np.asarray(used)))),
        "initial_selected": float(np.max(np.abs(initial))),
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
    stem = f"{row['mode']}_R{float(row['R_out_rg']):.0f}_{float(row['ratio']):.8f}".replace(".", "p")
    payload = {key: value for key, value in row.items() if key != "z"}
    np.savez_compressed(
        CHECKPOINT_DIR / f"{stem}.npz",
        z=np.asarray(row["z"], dtype=float),
        ratio=np.array(row["ratio"]),
        R_out_rg=np.array(row["R_out_rg"]),
        g_u_used=np.array(row["g_u_used"]),
        g_T_used=np.array(row["g_T_used"]),
        row_json=np.array(local_row_json(payload)),
    )


def write_table(rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Controlled Slope R_out Continuation",
        "",
        "Generated by `scripts/run_transonic_controlled_slope_rout_continuation.py`.",
        "",
        "Starts from the best slope-calibration checkpoint and moves `R_out` outward while storing the target, used, and output outer slopes at every checkpoint. `raw` recomputes slopes directly, `ema` carries an exponential moving average, and `held` keeps the initial calibrated slopes fixed.",
        "",
        "| mode | calibration | R_out/rg | final physical | dominant | target-used | used-out | g_u target | g_T target | g_u used | g_T used | g_u out | g_T out | interval R | interval E | outer 1 | outer 2 | D | C1 | C2 | K | compat max | Rson/rg | lambda0 | max H/R | outer H/R | int adv | nfev | success | message |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {mode} | {calibration_label} | {R_out_rg} | {final_physical} | {dominant} | "
            "{target_used_delta} | {used_out_delta} | {g_u_target} | {g_T_target} | {g_u_used} | {g_T_used} | "
            "{g_u_out} | {g_T_out} | {interval_R} | {interval_E} | {outer_1} | {outer_2} | {D} | {C1} | "
            "{C2} | {K} | {compat_max} | {Rson_rg} | {lambda0} | {max_HR} | {outer_HR} | {int_adv} | "
            "{nfev} | {success} | {message} |".format(
                mode=row["mode"],
                calibration_label=row["calibration_label"],
                R_out_rg=fmt(float(row["R_out_rg"])),
                final_physical=fmt(float(row["final_physical"])),
                dominant=row["dominant"],
                target_used_delta=fmt(float(row["target_used_delta"])),
                used_out_delta=fmt(float(row["used_out_delta"])),
                g_u_target=fmt(float(row["g_u_target"])),
                g_T_target=fmt(float(row["g_T_target"])),
                g_u_used=fmt(float(row["g_u_used"])),
                g_T_used=fmt(float(row["g_T_used"])),
                g_u_out=fmt(float(row["g_u_out"])),
                g_T_out=fmt(float(row["g_T_out"])),
                interval_R=fmt(float(row["interval_R"])),
                interval_E=fmt(float(row["interval_E"])),
                outer_1=fmt(float(row["outer_1"])),
                outer_2=fmt(float(row["outer_2"])),
                D=fmt(float(row["D"])),
                C1=fmt(float(row["C1"])),
                C2=fmt(float(row["C2"])),
                K=fmt(float(row["K"])),
                compat_max=fmt(float(row["compat_max"])),
                Rson_rg=fmt(float(row["Rson_rg"])),
                lambda0=fmt(float(row["lambda0"])),
                max_HR=fmt(float(row["max_HR"])),
                outer_HR=fmt(float(row["outer_HR"])),
                int_adv=fmt(float(row["int_adv"])),
                nfev=row["nfev"],
                success="yes" if row["success"] else "no",
                message=str(row["message"]).replace("|", "/"),
            )
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")


def mode_specs(best_record: dict[str, object]) -> tuple[dict[str, object], ...]:
    return (
        {"mode": "raw_calibrated", "spec": best_record, "baseline": False},
        {"mode": "ema_calibrated_eta0p35", "spec": best_record, "baseline": False},
        {"mode": "held_calibrated", "spec": best_record, "baseline": False},
        {"mode": "ema_baseline_o2_n8_eta0p35", "spec": best_record, "baseline": True},
    )


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    best = best_calibration_record()
    ratio = float(best["ratio"])
    seed_z = np.asarray(best["z"], dtype=float)
    seed_slopes = (float(best["g_u"]), float(best["g_T"]))
    calibration_label = str(best["label"])

    rows: list[dict[str, object]] = []
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for old_path in CHECKPOINT_DIR.glob("*.npz"):
        old_path.unlink()

    for mode_spec in mode_specs(best):
        mode = str(mode_spec["mode"])
        use_baseline = bool(mode_spec["baseline"])
        z_seed = np.asarray(seed_z, dtype=float)
        previous_params = params_for(fiducial, mdot_edd, ratio, R_OUT_SEQUENCE_RG[0], seed_slopes)
        previous_used: tuple[float, float] | None = None
        for R_out_rg in R_OUT_SEQUENCE_RG:
            rough_params = params_for(fiducial, mdot_edd, ratio, R_out_rg, seed_slopes)
            if not np.isclose(R_out_rg, previous_params.R_out_rg):
                z_seed = remap_state_with_slopes(z_seed, previous_params, rough_params, previous_used or seed_slopes)
            target = target_slopes(z_seed, rough_params, mode_spec["spec"], use_baseline=use_baseline)
            used = update_slopes(mode, previous_used, target)
            params = params_for(fiducial, mdot_edd, ratio, R_out_rg, used)
            result = solve_variant(z_seed, params)
            row = row_from_result(
                mode=mode,
                calibration_label=calibration_label,
                ratio=ratio,
                R_out_rg=R_out_rg,
                target=target,
                used=used,
                params=params,
                z0=z_seed,
                result=result,
            )
            rows.append(row)
            save_checkpoint(row)
            write_table(rows)
            print(
                f"{mode} R={R_out_rg:.0f} final={row['final_physical']:.3e} dom={row['dominant']} "
                f"g_used=({row['g_u_used']:.5f},{row['g_T_used']:.5f})",
                flush=True,
            )
            z_seed = np.asarray(row["z"], dtype=float)
            previous_params = params
            previous_used = used

    print(f"wrote {TABLE_OUTPUT}")


if __name__ == "__main__":
    main()
