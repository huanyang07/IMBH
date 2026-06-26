"""Validate tuned full-slope matches with alternate sonic residual sets."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from imri_qpe.layer3_minidisk_1d import TransonicSlimParams
from imri_qpe.layer3_minidisk_1d.transonic_collocation import (
    _sonic_component_values,
    residual_audit_from_state_vector,
    square_collocation_residual,
    square_jac_sparsity_pattern,
    state_bounds,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot

from run_transonic_full_slope_sensitivity_audit import (
    SLOPE_SOURCE_CHECKPOINT,
    load_checkpoint,
    polyfit_outer_log_slopes,
)


ROOT = Path(__file__).resolve().parents[1]
TABLE_OUTPUT = ROOT / "outputs" / "tables" / "transonic_full_slope_tuned_sonic_audit.md"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "transonic_full_slope_tuned_sonic_audit"

R_OUT_RG = 5000.0
N_NODES = 64
MAX_NFEV = 300
SLOPE_SPECS = (
    ("g_T_plus_1e-3", "outputs/checkpoints/transonic_full_slope_sensitivity_audit/g_T_p1_1em03_0p90277664.npz", 0.0, 1.0e-3),
    ("g_u_minus_3e-3", "outputs/checkpoints/transonic_full_slope_sensitivity_audit/g_u_m1_3em03_0p90277664.npz", -3.0e-3, 0.0),
)
SONIC_SPECS = (
    ("pivot_C1", "C1", ("D", "C1"), 1.0),
    ("pivot_C2", "C2", ("D", "C2"), 1.0),
    ("symmetric_D_C1_C2", "symmetric", ("D", "C1", "C2"), 1.0),
    ("symmetric_D_C1_C2_K", "symmetric", ("D", "C1", "C2", "K"), 1.0),
)


def fmt(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.4g}"


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def row_json(row: dict[str, object]) -> str:
    return json.dumps({key: json_safe(value) for key, value in row.items()}, sort_keys=True)


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


def residual_vector(z: np.ndarray, params: TransonicSlimParams, mode: str, pivot: str, sonic_components: tuple[str, ...]) -> np.ndarray:
    if mode != "symmetric":
        return square_collocation_residual(z, params, pivot=pivot)
    interval_end = 2 * (params.n_nodes - 1)
    base = square_collocation_residual(z, params, pivot="C1")
    return np.concatenate(
        [
            base[: interval_end + 2],
            _sonic_component_values(z, params, sonic_components),
        ]
    )


def sparsity_pattern(params: TransonicSlimParams, mode: str, sonic_components: tuple[str, ...]):
    if mode != "symmetric":
        return square_jac_sparsity_pattern(params)
    from scipy.sparse import lil_matrix

    unknown_size = 2 * params.n_nodes + 2
    n_rows = 2 * (params.n_nodes - 1) + 2 + len(sonic_components)
    pattern = lil_matrix((n_rows, unknown_size), dtype=int)
    row = 0
    for idx in range(params.n_nodes - 1):
        for col in (idx, idx + 1, params.n_nodes + idx, params.n_nodes + idx + 1, unknown_size - 2, unknown_size - 1):
            pattern[row : row + 2, col] = 1
        row += 2
    for col in (params.n_nodes - 1, 2 * params.n_nodes - 1, unknown_size - 2, unknown_size - 1):
        pattern[row : row + 2, col] = 1
    row += 2
    for col in (0, params.n_nodes, unknown_size - 2, unknown_size - 1):
        pattern[row : row + len(sonic_components), col] = 1
    return pattern.tocsr()


def solve_variant(z0: np.ndarray, params: TransonicSlimParams, mode: str, pivot: str, sonic_components: tuple[str, ...]):
    lower, upper = state_bounds(params)
    z_start = np.clip(np.asarray(z0, dtype=float), lower + 1.0e-12, upper - 1.0e-12)
    return least_squares(
        lambda trial: residual_vector(trial, params, mode, pivot, sonic_components),
        z_start,
        jac_sparsity=sparsity_pattern(params, mode, sonic_components),
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


def row_from_result(label: str, sonic_label: str, mode: str, pivot: str, sonic_components: tuple[str, ...], ratio: float, params, z0, result) -> dict[str, object]:
    z = np.asarray(result.x, dtype=float)
    initial = residual_vector(z0, params, mode, pivot, sonic_components)
    final = residual_vector(z, params, mode, pivot, sonic_components)
    audit = residual_audit_from_state_vector(z, params)
    legacy_audit = residual_audit_from_state_vector(z, replace(params, outer_closure="thin_value", outer_match_log_slopes=None))
    return {
        "label": label,
        "sonic_label": sonic_label,
        "mode": mode,
        "pivot": pivot,
        "sonic_components": sonic_components,
        "ratio": ratio,
        "g_u": float(params.outer_match_log_slopes[0]),
        "g_T": float(params.outer_match_log_slopes[1]),
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
        "nfev": int(result.nfev),
        "success": bool(result.success),
        "message": str(result.message),
        "z": z,
    }


def save_checkpoint(row: dict[str, object]) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{row['label']}_{row['sonic_label']}_{float(row['ratio']):.8f}".replace(".", "p")
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
        "# Tuned Full-Slope Sonic Audit",
        "",
        "Generated by `scripts/run_transonic_full_slope_tuned_sonic_audit.py`.",
        "",
        "Uses the two best slope perturbations from the C1 slope-sensitivity audit and tests C1, C2, and symmetric sonic residual sets.",
        "",
        "| slope label | sonic label | mode | components | g_u | g_T | selected initial | selected final | final physical | dominant | interval R | interval E | outer 1 | outer 2 | legacy Omega | legacy E | D | C1 | C2 | K | compat max | nfev | success | message |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {sonic_label} | {mode} | {sonic_components} | {g_u} | {g_T} | {initial_selected} | {final_selected} | "
            "{final_physical} | {dominant} | {interval_R} | {interval_E} | {outer_1} | {outer_2} | "
            "{legacy_outer_Omega} | {legacy_outer_E} | {D} | {C1} | {C2} | {K} | {compat_max} | {nfev} | "
            "{success} | {message} |".format(
                label=row["label"],
                sonic_label=row["sonic_label"],
                mode=row["mode"],
                sonic_components=",".join(row["sonic_components"]),
                g_u=fmt(float(row["g_u"])),
                g_T=fmt(float(row["g_T"])),
                initial_selected=fmt(float(row["initial_selected"])),
                final_selected=fmt(float(row["final_selected"])),
                final_physical=fmt(float(row["final_physical"])),
                dominant=row["dominant"],
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
    slope_params = params_for(fiducial, mdot_edd, ratio, slopes=(0.0, 0.0))
    base_slopes = polyfit_outer_log_slopes(slope_z, slope_params)

    rows: list[dict[str, object]] = []
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for old_path in CHECKPOINT_DIR.glob("*.npz"):
        old_path.unlink()

    for label, seed_path, dg_u, dg_T in SLOPE_SPECS:
        seed_z, seed_ratio = load_checkpoint(ROOT / seed_path)
        if not np.isclose(seed_ratio, ratio):
            raise RuntimeError("seed checkpoint ratio mismatch")
        slopes = (base_slopes[0] + dg_u, base_slopes[1] + dg_T)
        params = params_for(fiducial, mdot_edd, ratio, slopes=slopes)
        for sonic_label, mode, sonic_components, _weight in SONIC_SPECS:
            pivot = "C1" if mode == "symmetric" else mode
            result = solve_variant(seed_z, params, mode, pivot, sonic_components)
            row = row_from_result(label, sonic_label, mode, pivot, sonic_components, ratio, params, seed_z, result)
            rows.append(row)
            save_checkpoint(row)
            write_table(rows)
            print(
                f"{label} {sonic_label} final={row['final_physical']:.3e} dom={row['dominant']} nfev={row['nfev']}",
                flush=True,
            )

    print(f"wrote {TABLE_OUTPUT}")


if __name__ == "__main__":
    main()
